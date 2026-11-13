"""Experiment — the top-level Python API entry point (Section 15).

Wraps the experiment directory: creation, opening, and recording access. All
experiment-state mutation and path resolution happen here or in ``Recording``, never
in callers (Tenet 2).
"""

import contextlib
import logging
import os
import re
import shutil
import threading
import time

from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml


from detail.api import models as api_models
from detail.api import run_info
from detail.api import spatial_categories
from detail.api._runner_modules import PIPELINE_RUNNERS as _PIPELINE_RUNNERS
from detail.api.models import ImportedModelRef  # re-exported; machine-global (§5)
from detail.api.recording import Recording
from detail.configs.defaults import none_means_default
from detail.configs.distill import DistillConfig
from detail.configs.labeling import LabelingConfig
from detail.core import atomic, disk_space
from detail.core import camera_view as camera_view_mod
from detail.core import class_names as class_names_mod
from detail.core import progress as progress_mod
from detail.core.protocols import ModelNotReadyError
from detail.core.dataclasses import (
    ACTIVE_STATUSES,
    AUTHORED_STATUSES,
    AWAITING_REFERENCES,
    DISTILL_STAGE_WEIGHTS,
    KEYPOINT_SETUP_CLASSES_STAGE_WEIGHTS,
    KEYPOINT_SETUP_STAGE_WEIGHTS,
    BatchProgressEvent,
    ComputeTarget,
    DatasetExportJob,
    DistillProgress,
    ExportJob,
    InstallProgress,
    JobStatus,
    LabelingClass,
    LabelingJob,
    ModelReadiness,
    ProcessingProgress,
    ProgressEvent,
    Prompt,
    TrainingJob,
    distill_overall_fraction,
    distill_progress_from_dict,
    distill_progress_to_dict,
    processing_inflight_fraction,
    processing_progress_from_dict,
    processing_progress_to_dict,
    reconciled_status,
)
from detail.core.progress import remaining_seconds
from detail.core.registries import TRACKERS, TRAINERS
from detail.models import distilled as distilled_store
from detail.core.utils import (
    check_cuda_indices,
    default_compute,
    effective_backend,
    resolve_device,
    utc_timestamp,
)
from detail.core.visualize import rgb_to_bgr
from detail.executor import rundir, run_metrics, ssh_transport, video_staging
from detail.executor.base import RunHandle
from detail.executor.handles import ExecutorRunHandle
from detail.executor.local import (
    LocalExecutor,
    cancel_run_dir,
    coordinator_alive,
    kill_coordinator,
    live_process_pid,
    pid_alive,
    reap_coordinator,
    request_stop,
    spawn_coordinator,
    worker_alive,
)
from detail.executor.remote import executor_for
from detail.executor.runners import (
    ANALYSIS_MODULES,
    PROCESSING_MODULES,
    import_runner_modules,
    job_params,
)
from detail.storage import activity
from detail.storage import analysis_runs as analysis_store
from detail.storage import credentials
from detail.storage import datasets as datasets_store
from detail.storage import dirscan
from detail.storage import distill_runs as distill_store
from detail.storage import heading_definitions as heading_store
from detail.storage import export as export_store
from detail.storage import export_runs as export_runs_store
from detail.storage import locations
from detail.storage import metadata as metadata_store
from detail.storage import models as models_store
from detail.storage import notes as notes_store
from detail.storage import experiment as experiment_store
from detail.storage import runs as runs_store
from detail.storage import read_cache
from detail.storage import recording as recording_store
from detail.storage import remote_targets
from detail.storage import stream_presets
from detail.trainers import import_trainers

if TYPE_CHECKING:
    from detail.api import previews


class _RecordingSteps:
    """A per-recording hook reported through a ``(stage, fraction, detail)`` callback, for a
    loop the experiment does not own (a module's ``resolve_shared_config``), worded as
    ``progress.reporting_each`` words its own: ``Rat 12 (3 of 48)``.

    Called with each recording's id before the loop reads it. ``finish`` reports the end, and
    only for a loop that reported a step, so a hook a module does not override says nothing.
    """

    def __init__(
        self,
        names: dict[str, str],
        progress: Callable[[str, float, str], None] | None,
        stage: str,
        total: int,
    ) -> None:
        self._names = names
        self._progress = progress
        self._stage = stage
        self._total = total
        self._done = 0

    def __call__(self, recording_id: str) -> None:
        if self._progress is not None:
            name = self._names.get(recording_id, recording_id)
            self._progress(
                self._stage,
                self._done / self._total,
                f"{name} ({self._done + 1} of {self._total})",
            )
        self._done += 1

    def finish(self) -> None:
        if self._progress is not None and self._done:
            self._progress(self._stage, 1.0, f"{self._total} of {self._total}")


def _with_model_defaults(module, config: dict) -> dict:
    """``config`` plus the module model's default for every setting the caller omitted.

    A run must be reproducible from its own record (Tenet 9), and this record is what **Start
    from** re-seeds a form with and what ``retry_run`` re-validates. A defaulted key left out
    of it is not "unset" — it silently means *whatever the default is on the day someone reads
    it back*, so changing a default would re-run an existing run differently.

    Only absent keys are filled, and only where the model declares a default: the caller's own
    values win, and a *required* field is left alone because at this level it may still hold an
    unresolved batch spec (``detection_spec`` rather than ``detection_run_id``) that each
    recording resolves for itself. That is also why ``module.validate``'s enriched config is
    not what gets stored — its ``arena_ids`` are one recording's, and this record spans several.
    """
    model = getattr(module, "config_model", None)
    if model is None:
        return config
    filled = dict(config)
    for name, field in model.model_fields.items():
        if name not in filled and not field.is_required():
            filled[name] = field.get_default(call_default_factory=True)
    return filled


def _space_record(space: disk_space.SpaceEstimate) -> dict:
    """What a run record keeps of its space check (#743): the numbers only. ``where`` can be
    an absolute path, which no experiment file may hold."""
    return {
        "estimated_output_bytes": int(space.bytes),
        "free_bytes_at_launch": int(space.free_bytes),
        "fits": bool(space.fits),
    }


def _jsonable(value):
    """Best-effort conversion to a YAML/JSON-native value for the run record.

    Typed config values (Prompt, ComputeTarget, …) expose ``to_dict``; the full
    reproducible config also lives in each per-recording ``job.json`` (Tenet 9), so
    this record is an index, not the source of truth. ``_typed_config`` is the
    inverse for the coordinator, which reads the record back across a process
    boundary and must restore the typed objects before ``Recording.run``.
    """
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _record_compute(config: dict) -> dict | None:
    """A processing run record's top-level ``compute`` — where the run executes.

    Written at reservation as well as by the coordinator: the run page builds its "Where" and
    live-cost rows from the record it opens on, which is the reserved one when it opens from
    the launch dialog. A reservation without it read as "local", with no cost ticker, for the
    whole life of a remote run.
    """
    return _jsonable(config.get("compute"))


#: The training statuses that can leave a resumable checkpoint — a worker killed (``cancelled``)
#: or crashed mid-epoch (``failed``). Not ``partial``: training only lands there when
#: Ultralytics' ``train()`` *returned*, and a return always finalizes the run, stripping
#: ``last.pt`` of its epoch and optimizer. Offering Resume on one trained a new model on
#: Ultralytics' COCO8 sample instead (#486). The one list the API, the distill resume and the
#: model page's Resume button all read — three hand-copied tuples before.
RESUMABLE_TRAINING_STATUSES = ("cancelled", "failed")

#: Where a swallowed bookkeeping failure goes. There is no per-run activity log yet, so this is
#: the only record that a progress write or a run-record write gave up — which matters most on
#: the share where it happens (#289 was found by a status stream the user ran alongside the app).
_log = logging.getLogger(__name__)

# The typed config primitives ``_jsonable`` flattens to dicts for the run record,
# keyed by the config field they live under. A run record is JSON, so type is lost;
# the detached coordinator restores these before re-running each recording because the
# validators read them as objects (``compute.backend``, ``prompt.targets``).
_TYPED_CONFIG_FIELDS = {
    "compute": ComputeTarget.from_dict,
    "prompt": Prompt.from_dict,
}


def _typed_config(config: dict) -> dict:
    """Restore the typed config values a run record flattened — the inverse of
    ``_jsonable`` for the known typed fields (§15).

    Only fields still carrying a plain ``dict`` are rebuilt, so an absent key or an
    already-typed value is left untouched (safe on a config that never round-tripped).
    """
    restored = dict(config)
    for key, from_dict in _TYPED_CONFIG_FIELDS.items():
        value = restored.get(key)
        if isinstance(value, dict):
            restored[key] = from_dict(value)
    return restored


def _video_staging_target(
    label_compute: ComputeTarget, train_compute: ComputeTarget
) -> str | None:
    """The remote target a distill's background video upload goes to, or ``None`` when neither
    stage is remote: the training compute's (the trained model, and so the detection run that
    reads every recording, lives there), else the labeling compute's."""
    for compute in (train_compute, label_compute):
        if compute.backend == "remote" and compute.remote_target:
            return compute.remote_target
    return None


def _require_remote_target_registered(compute: ComputeTarget) -> None:
    """A remote ``compute`` names a target that is actually registered (§6).

    Shared by every compute-accepting entry point (``label_dataset``, ``train_model``,
    ``distill``'s two computes, ``Recording.run`` via its own module validation) so a
    typo'd or never-added target fails here, synchronously, rather than deep inside a
    detached coordinator's first remote job submission.

    Raises:
        ValueError: ``compute.backend`` is ``"remote"`` with no ``remote_target``, or one
            that names a target nothing has registered.
    """
    remote_targets.require_registered(compute)


def _run_status(run) -> str:
    """A job handle's status, or ``failed`` if its envelope cannot be read at all.

    ``run.status`` is a file read. Asking for it is the last thing the orchestrator does with a
    recording, and it must not be able to raise there — a status we cannot read after the job
    has finished is not a healthy recording, so the honest answer is ``failed`` rather than an
    exception escaping into the batch loop.
    """
    # noqa: BLE001 — an envelope we cannot read is a recording we cannot vouch for.
    try:
        return run.status
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not read a job's status (%s)", exc)
        return "failed"


def _base_name(path: str) -> str:
    """The last component of a path written on either platform — a cell pasted from
    Windows Explorer splits on backslashes even when this runs on Linux."""
    return re.split(r"[\\/]", path.strip())[-1]


def _video_file_key(path: str) -> str:
    return _base_name(path).lower()


def _match_error(
    what: str, value: str, recording_ids: list[str], names: dict[str, str]
) -> str | None:
    """The refusal for a key that names no recording or several, in the user's terms."""
    if not recording_ids:
        return f"unknown {what} {value!r} — not in this experiment"
    if len(recording_ids) > 1:
        matched = ", ".join(repr(names.get(rid, rid)) for rid in recording_ids)
        return f"{what} {value!r} matches several recordings: {matched}"
    return None


class RunOrchestrator:
    """Sequences one processing module over a recording selection (§15). Runs **inside
    the detached coordinator subprocess** (``executor/coordinator.py``): its ``emit``
    sink publishes each aggregate ``ProcessingProgress`` to ``runs/<id>/progress.json``
    and it keeps the experiment-level ``run.yaml`` index current, so the whole run
    completes even with the launching app closed. The per-recording executions are
    ordinary ``Recording.run`` calls (each its own worker + output dir, Tenet 5)."""

    def __init__(
        self,
        experiment: "Experiment",
        run_id: str,
        module_id: str,
        recording_ids: list[str],
        config: dict,
        emit: Callable[[ProcessingProgress], None],
        created_at: str,
        per_recording: dict[str, dict] | None = None,
        prior_duration_s: float = 0.0,
    ) -> None:
        self.id = run_id
        self._experiment = experiment
        self._module_id = module_id
        self._recording_ids = list(recording_ids)
        # The coordinator reads config from the (JSON) run record, where _jsonable
        # flattened the typed values; restore them before Recording.run re-validates,
        # whose checks read compute.backend / prompt.targets (not dicts).
        self._config = _typed_config(config)
        # Wrapped once, here, rather than guarded at each of the five call sites: publishing the
        # aggregate bar is reporting, and one of those sites is in `run()` itself — outside every
        # handler — where a failed write escapes to the coordinator's backstop and fails the
        # whole run. A rule applied per call site is a rule the sixth call site will not get.
        # The activity log rides the same wrapper, for the reason the comment above gives: one
        # site rather than five, so the sixth gets it too. It is sampled *after* the publish —
        # a bar frame is the thing a user is waiting on, and the log is the thing they read
        # afterwards. Its own write is `best_effort` in turn, so a log error cannot even reach
        # this wrapper's guard, let alone the coordinator's backstop.
        self._activity = activity.ActivityLog(
            runs_store.run_dir(experiment.dir, run_id)
        )
        #: A per-recording ETA needs a clock that restarts when the recording does, which is
        #: exactly what `PhaseClock` already does for phases — the run-wide rate cannot answer
        #: "is *this* recording stuck" when recordings differ by hours.
        self._unit_clock = progress_mod.PhaseClock(time.monotonic())

        def publish(p: ProcessingProgress) -> None:
            emit(p)
            self._log_activity(p)

        self._emit = progress_mod.best_effort("run progress")(publish)
        self._created_at = created_at
        self._start = time.monotonic()
        #: Time the run had already been going before this coordinator started — non-zero only on
        #: a retry, which spawns a fresh coordinator whose clock starts at zero. Kept apart from
        #: ``_start`` because the ETA must divide by the work done *since this coordinator began*
        #: (see ``_done_at_start``); only the duration the record reports is the whole run's.
        self._prior_duration_s = float(prior_duration_s or 0.0)
        self._cancelled = threading.Event()
        self._live: dict[str, RunHandle] = {}  # recordings currently in flight
        self._lock = threading.Lock()  # guards _live / _per_recording / _done / record
        #: Each in-flight recording's latest progress event, published together as
        #: ``ProcessingProgress.live`` so a bar over several recordings is their sum.
        self._live_events: dict[str, ProgressEvent] = {}
        #: Serializes *publishing*: a snapshot taken, then written by a thread that lost the race
        #: to another's newer one, would put an older bar over a newer one. Taken before
        #: ``_lock`` and never inside it, so the two cannot deadlock.
        self._emit_lock = threading.Lock()
        #: How many recordings this run keeps in flight (recorded in ``run.yaml``).
        self._parallelism = 1
        # A retry re-enters an existing record: seeding the prior per-recording results
        # makes this run drive only what isn't complete, while _finalize still sees a
        # status for every recording (it rolls up complete only when len(statuses) == n).
        # A fresh run passes nothing, so every recording is pending — unchanged.
        self._per_recording: dict[str, dict] = dict(per_recording or {})
        self._done = sum(
            1 for v in self._per_recording.values() if v.get("status") == "complete"
        )
        # The mark this coordinator started from. The bar spans the whole run, but
        # _elapsed() only buys the recordings run from here, so the ETA divides by the
        # work since this point (0 for a fresh run).
        self._done_at_start = self._done
        # Iterate recording_ids, not _per_recording: a recording the coordinator never
        # launched is absent from the map entirely and must still count as pending.
        self._pending_ids = [
            sid
            for sid in self._recording_ids
            if self._per_recording.get(sid, {}).get("status") != "complete"
        ]
        self._status = "running"

    def run(self) -> None:
        # Isolation boundary (like the executor worker): a per-recording failure is
        # recorded and the batch continues (D5); the finally guarantees a terminal
        # run.yaml even if the loop raises.
        n = len(self._recording_ids)
        if not self._pending_ids:  # a retry with nothing left to drive
            self._finalize(n)
            return
        # Publish recordings done/n immediately so a reattaching client always has a bar.
        # done is 0 for a fresh run and the seeded count for a retry, and the named
        # recording is the first *pending* one — recording_ids[0] may already be finished.
        self._emit(
            ProcessingProgress(
                self._done,
                n,
                self._pending_ids[0],
                None,
                self._elapsed(),
                self._done_at_start,
            )
        )
        try:
            module = PROCESSING_MODULES.get(self._module_id)
            if module is not None and module.batch:
                self._run_batch(module, n)  # one job over the whole subset (§6.5)
            else:
                k = module.recording_parallelism() if module is not None else 1
                k = self._remote_parallelism(k)
                k = max(1, min(k, len(self._pending_ids)))
                self._parallelism = k
                if k == 1:
                    self._orchestrate_serial()  # unchanged path for GPU/serial stages
                else:
                    self._orchestrate_parallel(k)
        finally:
            self._finalize(n)

    def _remote_parallelism(self, local_k: int) -> int:
        """How many recordings to keep in flight: the target's ``max_concurrent_jobs`` for a
        remote run, else what the module says.

        ``recording_parallelism`` exists to stop two jobs fighting over one *local* GPU, and says
        nothing about a cluster, where every recording is its own Slurm job. A remote run follows
        its target, which is why it is not folded into the module's answer.

        **Without rsync it is 1, and says so.** Concurrent jobs rely on the upload leaving other
        jobs' run directories alone, which only the rsync path can do
        (``ssh_transport._rsync_mirror``); the ``scp`` fallback would let one job's upload reset
        another's, so the run goes serially instead of risking that.
        """
        compute = self._config.get("compute")
        if getattr(compute, "backend", None) != "remote":
            return local_k
        try:
            target = remote_targets.get_remote_target(compute.remote_target)
        except KeyError:
            return local_k  # `Recording.run` reports an unregistered target
        wanted = target.max_concurrent_jobs
        if wanted > 1 and not ssh_transport.rsync_available():
            _log.warning(
                "rsync is not installed on this machine, so remote target %r runs one "
                "recording at a time instead of %d: concurrent jobs need rsync to keep one "
                "job's upload from overwriting another's",
                target.name,
                wanted,
            )
            return 1
        return wanted

    def _publish(self, sid: str, event, n: int) -> None:
        """Publish the aggregate bar, with ``event`` as ``sid``'s latest (``None`` when it ended).

        Snapshot and write under one lock, so concurrent recordings cannot publish out of order.
        """
        with self._emit_lock:
            with self._lock:
                if event is not None:
                    self._live_events[sid] = event
                done, live = self._done, dict(self._live_events)
            self._emit(
                ProcessingProgress(
                    done,
                    n,
                    sid,
                    event,
                    self._elapsed(),
                    self._done_at_start,
                    live,
                )
            )

    def _run_batch(self, module, n: int) -> None:
        """Run a batch module once over the whole subset, mapping its
        ``(stage, fraction, detail)`` progress onto the aggregate bar. On success every
        recording shares the produced run id; on failure the whole batch is failed (a batch
        is atomic — retry re-runs it whole)."""

        def on_progress(stage: str, fraction: float, detail: str) -> None:
            # Fold the run's fraction into the aggregate bar: recordings_done is the whole
            # recordings written, and the remainder rides `inner` so the bar moves *within*
            # a recording — a single-recording batch would otherwise sit at 0 until the end.
            fraction = min(max(fraction, 0.0), 1.0)
            # Cap the whole-recordings count at n-1 so the "(k/n)" label never reads
            # "(n+1/n)" at fraction 1.0; the remainder rides `within`, so the folded
            # fraction stays exact.
            done = min(int(fraction * n), max(n - 1, 0))
            within = fraction * n - done
            current = (
                detail if detail in self._recording_ids else self._recording_ids[0]
            )
            inner = BatchProgressEvent(
                frames_done=round(within * 1000),
                frames_total=1000,
                message=stage,
            )
            # `self._emit` cannot raise (see __init__). It matters most here: this call is made
            # from inside the batch module's own loop, so a raised write would surface as the
            # *module* failing and take every recording in the run with it.
            self._emit(
                ProcessingProgress(
                    done, n, current, inner, self._elapsed(), self._done_at_start
                )
            )

        try:
            produced = module.run_batch(
                self._experiment,
                self._recording_ids,
                self._config,
                on_progress,
                self._cancelled,
            )
            entry = {"run_id": produced, "status": "complete"}
        except Exception as exc:  # noqa: BLE001 — boundary: record + finalize (D5)
            entry = {
                "run_id": None,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        with self._lock:
            for sid in self._recording_ids:
                self._per_recording[sid] = dict(entry)
            self._done = sum(
                1 for v in self._per_recording.values() if v.get("status") == "complete"
            )
            self._write_record()

    def _log_activity(self, p: "ProcessingProgress") -> None:
        """One activity row for this emission, with both ETAs kept apart.

        ``eta_run_s`` divides the run's own fraction by this coordinator's elapsed;
        ``eta_unit_s`` divides the recording's frames by the time since *that recording* started.
        Deriving the second from the first is the tempting shortcut and is wrong by hours on a
        cohort whose recordings differ by hours — which is the case this exists for.
        """
        inner = p.inner
        done = getattr(inner, "frames_done", 0) or 0
        total = getattr(inner, "frames_total", 0) or 0
        now = time.monotonic()
        unit_fraction = (done / total) if total else 0.0
        run_fraction = (
            (p.recordings_done + min(unit_fraction, 1.0)) / p.recordings_total
            if p.recordings_total
            else 0.0
        )
        self._activity.sample(
            activity.make_row(
                elapsed_s=p.elapsed_s,
                unit=p.current_recording,
                unit_index=min(p.recordings_done + 1, p.recordings_total or 1),
                unit_total=p.recordings_total,
                phase=getattr(inner, "message", "") or "",
                done=done,
                total=total,
                eta_unit_s=self._unit_clock.eta(
                    p.current_recording, unit_fraction, now
                ),
                eta_run_s=progress_mod.remaining_seconds(run_fraction, p.elapsed_s),
            )
        )

    def _elapsed(self) -> float:
        return time.monotonic() - self._start

    def _orchestrate_serial(self) -> None:
        for sid in self._pending_ids:
            # The on-disk cancel sentinel (Experiment.cancel_run) stops the batch so
            # _finalize records it as cancelled (D5) — works from a fresh process.
            if self._cancel_requested():
                break
            self._run_recording(sid)

    def _orchestrate_parallel(self, k: int) -> None:
        # CPU transcode stages fan out across cores. Each recording is still its own
        # detached subprocess (own LocalExecutor + run dir), so these pool threads
        # only spawn workers + pump their progress files — the compute is elsewhere.
        with ThreadPoolExecutor(max_workers=k) as pool:
            futures = []
            for sid in self._pending_ids:
                if self._cancel_requested():
                    break  # don't launch further recordings once cancelled
                futures.append(pool.submit(self._run_recording, sid))
            for future in futures:
                future.result()  # _run_recording is self-contained; this just joins

    def _cancel_requested(self) -> bool:
        if self._cancelled.is_set() or runs_store.cancel_requested(
            self._experiment.dir, self.id
        ):
            self._cancelled.set()
            return True
        return False

    def _run_recording(self, sid: str) -> None:
        n = len(self._recording_ids)
        if self._cancel_requested():  # a queued pool task whose batch was cancelled
            return
        # **Only the work's own outcome may fail the recording.** The `except` covers the
        # launch and nothing else; everything after it is bookkeeping — writing the run record,
        # pumping progress — and a bookkeeping failure says nothing about the job. Sharing one
        # handler is #289: a transient Windows error on the coordinator's own `progress.json`
        # marked a healthy recording failed, and because the handler sits before `wait()` it
        # also abandoned a live worker, which kept the GPU and stalled the rest of the batch.
        try:
            recording_config = dict(self._config)
            recording_config["processing_run_id"] = self.id
            run = self._experiment.get_recording(sid).run(
                self._module_id, recording_config
            )
        except Exception as exc:  # noqa: BLE001 — boundary; record + continue
            prior = self._per_recording.get(sid, {})
            entry = {
                "run_id": prior.get("run_id"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            with self._lock:
                self._live[sid] = run
                self._per_recording[sid] = {
                    "run_id": run.id,
                    "status": _run_status(run),
                }
                self._record_best_effort()
            self._pump(run, sid, n)
            # Always reached, and never skipped by a bookkeeping error above: `wait` is the
            # only writer of this job's terminal status into its `recording.yaml`, and every
            # downstream gate reads that copy.
            with contextlib.suppress(Exception):
                run.wait()
            entry = {"run_id": run.id, "status": _run_status(run)}
            if entry["status"] != "complete":  # surface the worker's reason (D5)
                entry["error"] = run.error
            # Roll up whatever _adopt_status just merged into the per-recording
            # record (§ run location, duration, cost) — read back, not recomputed.
            with contextlib.suppress(Exception):
                module = PROCESSING_MODULES[self._module_id]
                recording_data = recording_store.read_recording(
                    self._experiment.get_recording(sid).dir
                )
                run_record = next(
                    r
                    for r in recording_data.get(module.runs_key, [])
                    if r.get("id") == run.id
                )
                entry.update(_stage_metrics(run_record))
        with self._lock:
            self._live.pop(sid, None)
            self._live_events.pop(sid, None)
            self._per_recording[sid] = entry
            self._done += 1
            self._record_best_effort()
        # Republish so recordings_done advances promptly between recordings. Outside `_lock`:
        # `_publish` takes its own locks in the order that cannot deadlock.
        self._publish(sid, None, n)

    def _pump(self, run, sid: str, n: int) -> None:
        """Forward a recording's progress to the aggregate bar. Reporting only — never raises.

        The emit cannot raise on its own (see ``__init__``); what this adds is the other half,
        the **stream**. ``run.progress()`` polls the worker's envelope, so it is file I/O too,
        and an exhausted read would otherwise reach the caller's boundary and be recorded as the
        job failing. Ending the pump early costs a stalled bar and nothing else — the wait that
        follows is what decides the recording's fate.
        """
        try:
            for event in run.progress():
                self._publish(sid, event, n)
        except Exception as exc:  # noqa: BLE001 — reporting on the work, not the work
            _log.warning(
                "progress stream for %s stopped early (%s: %s) — the run is unaffected",
                sid,
                type(exc).__name__,
                exc,
            )

    def _record_best_effort(self) -> None:
        """Persist the run record, tolerating a transient write failure.

        The record is re-written on every recording boundary, so a lost write is corrected by
        the next one. The last one has no next one, and :meth:`Experiment._reconcile_run` is
        what corrects that: it recomputes the status from each recording's own ``job.json``
        rather than trusting this index, on the next experiment open or run attach.

        Raising instead would let the index that *describes* the run decide whether the run
        failed. (This used to say `_finalize` was the corrector, which stopped being true the
        moment `_finalize` became the thing that needed correcting.)
        """
        try:
            self._write_record()
        except Exception as exc:  # noqa: BLE001 — the index, not the work
            _log.warning(
                "could not update run record %s (%s: %s) — retried at the next boundary",
                self.id,
                type(exc).__name__,
                exc,
            )

    def _finalize(self, n: int) -> None:
        """Compute the run's terminal status and record it. Best-effort, like every other
        write in this class.

        It runs in ``run()``'s ``finally``, outside every handler, so raising here does not
        merely lose the record — it escapes into ``executor/coordinator.py``'s backstop and is
        written down as ``failed: coordinator error``. A run whose every recording completed,
        recorded as failed, because of one transient ``PermissionError`` on a network share.

        Swallowing it loses nothing, which is the part worth knowing: the record stays
        ``running`` — still in ``ACTIVE_STATUSES`` — and :meth:`Experiment._reconcile_run`
        recomputes the status from each recording's own ``job.json``, the evidence rather than
        the index, on the next experiment open or run attach. That is the same backstop which
        already has to survive a SIGKILL, and a lost write is strictly less damage than one.
        """
        statuses = [v["status"] for v in self._per_recording.values()]
        if self._cancelled.is_set():
            self._status = "cancelled"
        elif len(statuses) == n and all(s == "complete" for s in statuses):
            self._status = "complete"
        else:
            self._status = "failed"
        self._record_best_effort()

    def _write_record(self) -> None:
        record = {
            "id": self.id,
            "module_id": self._module_id,
            "recording_ids": self._recording_ids,
            "config": _jsonable(self._config),
            "per_recording": self._per_recording,
            "status": self._status,
            "created_at": self._created_at,
            # One compute choice covers the whole batch (§ run location, duration,
            # cost) — a per-recording override doesn't exist today.
            "compute": _record_compute(self._config),
            "duration_s": self._prior_duration_s + self._elapsed(),
            "parallelism": self._parallelism,
        }
        costs = [
            entry["estimated_cost_usd"]
            for entry in self._per_recording.values()
            if entry.get("estimated_cost_usd") is not None
        ]
        if costs:
            record["total_estimated_cost_usd"] = sum(costs)
        runs_store.write_run(self._experiment.dir, self.id, record)


class AttachedRun(RunHandle):
    """Reattachable handle over a durable processing run (§15). Works from **any**
    process — the launching app, a GUI reopen, ``detail process attach`` — by
    polling the coordinator's ``runs/<id>/progress.json`` + ``run.yaml`` rather than
    holding an in-memory orchestrator. If the coordinator is gone, it reconciles the
    run to a terminal status (the SIGKILL backstop) so ``progress`` never hangs."""

    def __init__(
        self, experiment: "Experiment", run_id: str, poll_interval: float = 0.25
    ):
        self.id = run_id
        self._experiment = experiment
        self._poll = poll_interval

    @property
    def owns_worker(self) -> bool:
        """False — the worker belongs to whoever launched it, which is the point of reattaching.

        This is what makes ``Ctrl-C`` detach rather than cancel on every surface that streams a
        reattached run, without any of them having to remember to ask for it.
        """
        return False

    @property
    def status(self) -> str:
        return runs_store.read_run(self._experiment.dir, self.id).get("status")

    @property
    def error(self) -> str | None:
        record = runs_store.read_run(self._experiment.dir, self.id)
        for entry in (record.get("per_recording") or {}).values():
            if entry.get("error"):
                return entry["error"]
        return None

    def progress(self) -> Iterator[ProcessingProgress]:
        last = None
        dead_strikes = 0
        while True:
            record = runs_store.read_run(self._experiment.dir, self.id)
            data = runs_store.read_progress(self._experiment.dir, self.id)
            if data is not None and data != last:
                yield processing_progress_from_dict(data)
                last = data
            if record.get("status") not in ACTIVE_STATUSES:
                reap_coordinator(
                    runs_store.read_coordinator_pid(self._experiment.dir, self.id)
                )
                return
            # Cheap liveness on the hot path (no ps/PowerShell every poll); the
            # authoritative PID-reuse-guarded check runs inside _reconcile_run only
            # once the coordinator looks gone for two consecutive polls.
            pid = runs_store.read_coordinator_pid(self._experiment.dir, self.id)
            if pid is not None and not pid_alive(pid):
                dead_strikes += 1
                if dead_strikes >= 2:
                    self._experiment._reconcile_run(record)
                    if (
                        runs_store.read_run(self._experiment.dir, self.id).get("status")
                        not in ACTIVE_STATUSES
                    ):
                        return
                    dead_strikes = 0
            else:
                dead_strikes = 0
            time.sleep(self._poll)

    def wait(self) -> "AttachedRun":
        for _ in self.progress():
            pass
        return self

    def cancel(self) -> None:
        self._experiment.cancel_run(self.id)


@dataclass
class DatasetRef:
    """Lightweight index entry for a training dataset (§12) — for listings."""

    id: str
    name: str
    status: str
    n_train: int
    n_val: int
    #: Read by `api/active.py` for the Active-runs "started" column, and to sort the
    #: listing. Every ref here lacked it, so four of the six run kinds showed a blank
    #: cell and sorted last in arbitrary order.
    created_at: str = ""
    #: Where it ran, and how long it took (§ run location, duration, cost) — merged
    #: into `dataset.yaml` by `set_dataset_status` once the job is terminal; absent
    #: (all `None`) for one still running or one written before this field existed.
    compute: dict | None = None
    duration_s: float | None = None
    estimated_cost_usd: float | None = None


class AttachedJobRun(RunHandle):
    """Reattachable handle over a **single-job** run — analysis, dataset export, training,
    export (§12/§16). The one-job counterpart to :class:`AttachedRun`.

    The launch-time handles (:class:`AnalysisRun`, :class:`DatasetRun`, :class:`ModelRun`)
    wrap an ``ExecutorRunHandle`` over a ``LocalExecutor`` that keeps run
    dirs in **memory**, populated by ``submit``. That is the whole reason a reopened page
    could not stream a run: a fresh executor has no entry for a job it did not launch. This
    polls the run dir's ``job.json`` instead — the same file the worker writes — so any
    process can stream any run, after an app restart included.

    It is also the SIGKILL backstop for these runs. When the worker is gone and the record
    still reads ``running``, ``progress`` reconciles it to a terminal status instead of
    streaming forever — without which a killed run reads ``running`` for good *and cannot
    be deleted*, since the delete methods refuse an active run.
    """

    def __init__(
        self,
        run_dir: Path | str,
        run_id: str,
        finalize: Callable[[Path, str], None] | None = None,
        poll_interval: float = 0.25,
    ):
        self.id = run_id
        self._run_dir = Path(run_dir)
        self._finalize_record = finalize
        self._poll = poll_interval

    @property
    def owns_worker(self) -> bool:
        """False — the worker belongs to whoever launched it, which is the point of reattaching.

        This is what makes ``Ctrl-C`` detach rather than cancel on every surface that streams a
        reattached run, without any of them having to remember to ask for it.
        """
        return False

    @property
    def status(self) -> str:
        return rundir.read_status(self._run_dir).value

    @property
    def error(self) -> str | None:
        return rundir.read_envelope(self._run_dir).get("error")

    def progress(self) -> Iterator[ProgressEvent]:
        last = None
        dead_strikes = 0
        while True:
            status = rundir.read_status(self._run_dir)
            event = rundir.read_progress(self._run_dir)
            if event is not None and event != last:
                yield event
                last = event
            if status.value not in ACTIVE_STATUSES:
                self._finalize(status.value)
                return
            # Two consecutive dead reads before believing it: the pid is recorded just
            # after spawn, so a single miss can be a race with a worker still starting.
            pid = rundir.read_pid(self._run_dir)
            if pid is not None and not pid_alive(pid):
                dead_strikes += 1
                if dead_strikes >= 2:
                    rundir.set_status(
                        self._run_dir,
                        JobStatus.FAILED,
                        error="worker exited without completing",
                    )
                    self._finalize(JobStatus.FAILED.value)
                    return
            else:
                dead_strikes = 0
            time.sleep(self._poll)

    def _finalize(self, status: str) -> None:
        """Fold the job's terminal status into the run's own record.

        The runner writes the authoritative record on success, so only a record it never
        finalized needs this — a crash or kill leaves the initial ``running`` behind. The
        caller supplies that guarded write, because each run kind keeps its own record
        (``analysis.yaml`` / ``dataset.yaml`` / ``model.yaml`` / ``export_run.yaml``).
        """
        if self._finalize_record is None:
            return
        with contextlib.suppress(FileNotFoundError):
            self._finalize_record(self._run_dir, status)

    def wait(self) -> "AttachedJobRun":
        for _ in self.progress():
            pass
        return self

    def cancel(self) -> None:
        cancel_run_dir(self._run_dir)


class DatasetRun(RunHandle):
    """Run handle for a dataset export; reconciles ``dataset.yaml``'s status to
    the executor's terminal status on ``wait`` (the runner writes the full record
    on success; a failure leaves the initial ``running`` record to fix up)."""

    def __init__(
        self, inner: ExecutorRunHandle, experiment: "Experiment", dataset_id: str
    ):
        self._inner = inner
        self._experiment = experiment
        self.id = dataset_id

    @property
    def status(self) -> str:
        return self._inner.status

    @property
    def error(self) -> str | None:
        return self._inner.error

    def progress(self):
        return self._inner.progress()

    def wait(self) -> "DatasetRun":
        self._inner.wait()
        datasets_store.set_dataset_status(
            datasets_store.dataset_dir(self._experiment.dir, self.id),
            self._inner.status,
        )
        return self

    def cancel(self) -> None:
        self._inner.cancel()


def check_base_model_task(dataset_id: str, task: str | None, base_model: str) -> None:
    """Refuse a base model that cannot learn the dataset's labels: Ultralytics takes the task
    from the checkpoint, so a pose dataset needs a ``-pose`` base and anything else must not
    have one. Caught here rather than as an Ultralytics label-format error an hour in.
    """
    is_pose_base = "-pose" in Path(str(base_model)).stem
    if task == "pose" and not is_pose_base:
        raise ValueError(
            f"dataset {dataset_id!r} holds keypoint labels; train it from a YOLO-pose base "
            f"model (e.g. yolo11n-pose.pt), not {base_model!r}"
        )
    if task != "pose" and is_pose_base:
        raise ValueError(
            f"{base_model!r} is a YOLO-pose base model; dataset {dataset_id!r} holds "
            "detection labels"
        )


@dataclass
class ModelRef:
    """Lightweight index entry for a trained model (§12) — for listings."""

    id: str
    name: str
    status: str
    trainer_id: str
    dataset_id: str
    #: Read by `api/active.py` for the Active-runs "started" column, and to sort the
    #: listing. Every ref here lacked it, so four of the six run kinds showed a blank
    #: cell and sorted last in arbitrary order.
    created_at: str = ""
    #: See DatasetRef's identical fields (§ run location, duration, cost).
    compute: dict | None = None
    duration_s: float | None = None
    estimated_cost_usd: float | None = None
    #: What the model is for: ``"detection"`` (a segmentation / detection student) or
    #: ``"keypoints"`` (a pose student, §12 *Distill keypoints*). The detection picker skips
    #: pose models and the keypoint method picker offers them, by this and nothing else.
    role: str = "detection"


class ModelRun(RunHandle):
    """Run handle for a training run; reconciles ``model.yaml``'s status to the
    executor's terminal status on ``wait`` (the runner fills the full record on
    success; a failure leaves the initial ``running`` record to fix up)."""

    def __init__(
        self, inner: ExecutorRunHandle, experiment: "Experiment", model_id: str
    ):
        self._inner = inner
        self._experiment = experiment
        self.id = model_id

    @property
    def status(self) -> str:
        return self._inner.status

    @property
    def error(self) -> str | None:
        return self._inner.error

    def progress(self):
        return self._inner.progress()

    def wait(self) -> "ModelRun":
        self._inner.wait()
        models_store.set_model_status(
            models_store.model_dir(self._experiment.dir, self.id), self._inner.status
        )
        return self

    def cancel(self) -> None:
        self._inner.cancel()


class _StageFailed(RuntimeError):
    """A distill stage (label / train) finished non-``complete``; the orchestrator
    fails the whole run with this stage's reason."""


#: A keypoint setup's teacher stages, each a processing run (``stages.<name>.processing_run_id``).
_TEACHER_STAGES = ("detect", "track", "merge", "keypoints")


def _prior_distill_elapsed(experiment_dir, run_id: str) -> float:
    """Seconds the previous attempt of this distill run had already been going.

    Read from the persisted progress rather than tracked in the record: the coordinator
    writes a DistillProgress on every emit, so the last one is the run's own account of
    itself, and a resume that cannot read it simply starts from zero.
    """
    with contextlib.suppress(Exception):
        data = distill_store.read_progress(experiment_dir, run_id)
        if data:
            return float(data.get("elapsed_s") or 0.0)
    return 0.0


def _stage_metrics(record: dict) -> dict:
    """The subset of a just-completed dataset/model record worth folding into a
    distill run's own stage dict (§ run location, duration, cost) — read back
    rather than recomputed, since ``set_dataset_status``/``set_model_status``
    already merged them in by the time ``run.wait()`` returns."""
    return {
        key: record[key]
        for key in ("compute", "duration_s", "estimated_cost_usd")
        if key in record
    }


#: A distill record's stages, in run order, by its ``kind``: the detector distill is
#: label → train; a keypoint distill (§12 *Distill keypoints*) registers the student after.
DISTILL_STAGES_BY_KIND = {
    "detection": ("label", "train"),
    "keypoints": ("label", "train", "register"),
    # §26 / #717: the teacher's windowed detections and tracks, the pause for an example
    # set clicked on them, then the keypoint distill's own three.
    "keypoint_setup": ("detect", "track", "examples", "label", "train", "register"),
}
#: A keypoint setup with two or more animal classes (§26 *One example set per detection
#: class*): each class tracked on its own, then the tracks merged per teacher recording.
KEYPOINT_SETUP_CLASSES_STAGES = (
    "detect",
    "track",
    "merge",
    "examples",
    "label",
    "train",
    "register",
)

#: The progress phase each stage reports under (the stage a phase's bar segment belongs to).
_PHASE_STAGE = {
    "detecting": "detect",
    "tracking": "track",
    "merging": "merge",
    "labeling": "label",
    "training": "train",
    "registering": "register",
}


def _processing_run_status(experiment_dir, run_id: str) -> str | None:
    """A processing run's recorded status, or None when it has no record."""
    try:
        return runs_store.read_run(experiment_dir, run_id).get("status")
    except FileNotFoundError:
        return None


def distill_kind(record: dict) -> str:
    """A distill record's kind. ``record["kind"]``; a record written by the remote-detail line
    before the two lines met carries it in ``config`` instead, and both are read."""
    return record.get("kind") or (record.get("config") or {}).get("kind") or "detection"


def distill_stage_names(record: dict) -> tuple[str, ...]:
    """The stages a distill record runs, in order — what cancel, resume, reconcile and the
    roll-up iterate, so a third stage is one declaration rather than four hand-written
    loops over ``("label", "train")``. A keypoint setup with ``class_names`` merges its
    per-class tracks; one without (every record before it) keeps its six stages."""
    kind = distill_kind(record)
    if kind == "keypoint_setup" and setup_class_names(record):
        return KEYPOINT_SETUP_CLASSES_STAGES
    return DISTILL_STAGES_BY_KIND[kind]


def setup_class_names(record: dict) -> list[str]:
    """A keypoint setup's animal classes when it has two or more (``class_names``), else
    ``[]`` — the one-class setup, ``class_name``."""
    return list((record.get("config") or {}).get("class_names") or [])


def setup_teacher_tracks(record: dict) -> dict[str, str]:
    """``{recording_id: tracking run id}`` a setup's patch teacher labels: the merged tracks
    for a setup with several classes, else its one class's tracks."""
    stage = "merge" if setup_class_names(record) else "track"
    return dict(((record.get("stages") or {}).get(stage) or {}).get("run_ids") or {})


def distill_final_stage(record: dict) -> str:
    """The stage whose completion makes the run ``complete``."""
    return distill_stage_names(record)[-1]


class DistillOrchestrator:
    """Sequences the distill stages — label → train, and for a keypoint distill → register —
    as one durable run (§12). Runs **inside the detached distill coordinator subprocess**
    (``executor/distill_coordinator.py``): it composes ``Experiment.label_dataset`` (or
    ``keypoint_dataset``) and ``Experiment.train_model`` verbatim (each its own worker +
    output dir), folding their dissimilar progress (frame-fraction, then per-epoch) into one
    ``DistillProgress`` it publishes to ``distill/<id>/progress.json``. On resume it reuses a
    complete dataset, a complete model, and resumes a partial model."""

    def __init__(
        self,
        experiment: "Experiment",
        run_id: str,
        record: dict,
        emit: Callable[[DistillProgress], None],
    ) -> None:
        self.id = run_id
        self._experiment = experiment
        self._record = dict(record)
        self._config = record["config"]
        self._kind = distill_kind(record)
        self._stage_names = distill_stage_names(record)
        self._weights = (
            (
                KEYPOINT_SETUP_CLASSES_STAGE_WEIGHTS
                if setup_class_names(record)
                else KEYPOINT_SETUP_STAGE_WEIGHTS
            )
            if self._kind == "keypoint_setup"
            else None
        )
        # Same rule as `RunOrchestrator`, and the stakes are higher here: this driver's `run()`
        # catches only `_StageFailed`, so a raised progress write escapes to the distill
        # coordinator's backstop and fails a **multi-hour training run** on a bookkeeping error.
        self._emit = progress_mod.best_effort("distill progress")(emit)
        self._stages = dict(record.get("stages") or {})
        self._resume = bool(record.get("resume"))
        # On a resume the bar keeps the prior run's progress, so the clock has to keep its
        # elapsed too or the two describe different runs. Seeded from the last persisted
        # DistillProgress; zero for a fresh run, which is the identity. (The ETA no longer
        # uses this clock at all — see _emit_fold — but `elapsed_s` is still read as "how
        # long this run has been going".)
        self._start = time.monotonic() - (
            _prior_distill_elapsed(experiment.dir, run_id)
            if record.get("resume")
            else 0.0
        )
        self._phase: str | None = None
        self._stage_start = self._start  # reset at each phase boundary (see _emit_fold)
        self._status = "running"
        self._error: str | None = None

    def run(self) -> None:
        try:
            if self._cancel_requested():
                self._finalize("cancelled")
                return
            self._start_video_staging()
            if self._kind == "keypoint_setup" and not self._teacher_stages():
                return
            dataset_id = self._label_stage()
            if self._cancel_requested():
                self._finalize("cancelled")
                return
            self._train_stage(dataset_id)
            if "register" in self._stage_names:
                if self._cancel_requested():
                    self._finalize("cancelled")
                    return
                self._register_stage()
            self._finalize("complete")
        except _StageFailed as exc:
            self._finalize("failed", str(exc))

    def _start_video_staging(self) -> None:
        """Begin the opt-in background video upload, ahead of labeling, and carry on.

        Detached from this coordinator (``executor/video_staging.py``), so it outlives the two
        stages and is there for the detection run that follows. It is an optimisation: a copy that
        cannot start costs the later run its own upload and never fails the distill, so only the
        launch is guarded (``OSError`` from the spawn) and the reason is left in this log.
        """
        cfg = self._config
        target = cfg.get("stage_video_target")
        if not cfg.get("stage_videos") or not target:
            return
        try:
            video_staging.start(
                self._experiment.dir,
                target,
                list(cfg.get("source_recording_ids") or []),
            )
        except OSError as exc:
            print(
                f"{utc_timestamp()} could not start the background video upload "
                f"({type(exc).__name__}: {exc}); the detection run will upload the videos "
                "itself",
                flush=True,
            )

    def _label_stage(self) -> str:
        stage = self._stages.get("label") or {}
        if (
            self._resume
            and stage.get("status") == JobStatus.COMPLETE.value
            and stage.get("dataset_id")
        ):
            self._emit_fold("labeling", 1.0, "reused labeled dataset")
            return stage["dataset_id"]
        cfg = self._config
        # Forced phase-boundary emit before the slow SAM3 load, so the bar never sits
        # frozen on the prior phase (§6 progress rule).
        self._emit_fold("labeling", 0.0, "starting labeling")
        if self._kind in ("keypoints", "keypoint_setup"):
            # The teacher is finished keypoint detect runs, or the patch method resolved
            # and copied at launch (§12 *Distill keypoints*); its config was resolved then.
            # A setup's patch teacher is resolved once its tracks exist (`_setup_teacher`).
            if self._kind == "keypoint_setup":
                self._setup_teacher()
            teacher = cfg.get("teacher") or {}
            if teacher.get("kind") == "model":
                run = self._experiment.keypoint_dataset(
                    name=cfg["name"], teacher=teacher, **cfg["keypoint_dataset"]
                )
            else:
                # The tracking runs resolved at launch, replayed (a record from before
                # §26's 2026-10-21 revision has none and resolves them now).
                run = self._experiment.keypoint_dataset(
                    name=cfg["name"],
                    detect_runs=teacher["detect_runs"],
                    tracking=teacher.get("tracking_runs"),
                    **cfg["keypoint_dataset"],
                )
        else:
            lab = cfg["labeling"]
            classes = [LabelingClass.from_dict(c) for c in cfg["classes"]]
            run = self._experiment.label_dataset(
                name=cfg["name"],
                classes=classes,
                n_frames=lab["n_frames"],
                source_recording_ids=cfg.get("source_recording_ids"),
                logit_threshold=lab["logit_threshold"],
                min_contour_area=lab["min_contour_area"],
                val_split=lab["val_split"],
                # A distill recorded before the field existed labelled with no background.
                background_share=lab.get("background_share", 0.0),
                compute=ComputeTarget.from_dict(cfg["label_compute"]),
                sampling=lab["sampling"],
                model_id=lab["model_id"],
                model_variant=lab["model_variant"],
            )
        self._set_stage("label", "running", dataset_id=run.id)

        def report(ev) -> None:
            frac = ev.frames_done / ev.frames_total if ev.frames_total else 0.0
            self._emit_fold("labeling", frac, ev.message or "labeling frames", ev.eta_s)

        self._pump(run, report, "labeling")
        run.wait()
        if run.status != JobStatus.COMPLETE.value:
            self._set_stage("label", run.status)
            raise _StageFailed(f"labeling {run.status}: {run.error}")
        metrics: dict = {}
        with contextlib.suppress(Exception):
            dataset_record = datasets_store.read_dataset_yaml(
                datasets_store.dataset_dir(self._experiment.dir, run.id)
            )
            metrics = _stage_metrics(dataset_record)
        self._set_stage("label", "complete", dataset_id=run.id, **metrics)
        return run.id

    def _train_stage(self, dataset_id: str) -> None:
        cfg = self._config
        stage = self._stages.get("train") or {}
        model_id = stage.get("model_id")
        if (
            self._resume
            and model_id
            and stage.get("status") == JobStatus.COMPLETE.value
            and self._stage_names[-1] != "train"
        ):
            # A keypoint distill that failed at register keeps its trained student; a
            # resume registers it rather than training it again.
            self._emit_fold("training", 1.0, "reused trained model")
            return
        # Forced phase-boundary emit before the base-model spin-up.
        self._emit_fold("training", 0.0, "starting training")
        if self._resume and model_id and self._model_resumable(model_id):
            run = self._experiment.resume_training(model_id)
        else:
            run = self._experiment.train_model(
                name=cfg["name"],
                dataset_id=dataset_id,
                trainer_id=cfg["trainer_id"],
                base_model=cfg["base_model"],
                hyperparams=cfg["training"],
                compute=ComputeTarget.from_dict(cfg["train_compute"]),
            )
        self._set_stage("train", "running", model_id=run.id)

        def report(ev) -> None:
            frac = ev.epoch / ev.total_epochs if ev.total_epochs else 0.0
            message = ev.message or f"epoch {ev.epoch}/{ev.total_epochs}"
            self._emit_fold("training", frac, message, ev.eta_s)

        self._pump(run, report, "training")
        run.wait()
        if run.status != JobStatus.COMPLETE.value:
            self._set_stage("train", run.status)
            raise _StageFailed(f"training {run.status}: {run.error}")
        metrics: dict = {}
        with contextlib.suppress(Exception):
            model_record = models_store.read_model_yaml(
                models_store.model_dir(self._experiment.dir, run.id)
            )
            metrics = _stage_metrics(model_record)
        self._set_stage("train", "complete", model_id=run.id, **metrics)

    def _register_stage(self) -> None:
        """The keypoint distill's third stage: promote the trained student into the catalog
        (``api/keypoint_distill.register_student``). Sub-second and idempotent, so it gets
        the two phase-boundary emits and no pump."""
        from detail.api import keypoint_distill

        self._emit_fold("registering", 0.0, "registering student")
        self._set_stage("register", "running")
        try:
            model = keypoint_distill.register_student(
                self._experiment, self.id, {**self._record, "stages": self._stages}
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            self._set_stage("register", "failed")
            raise _StageFailed(f"register failed: {exc}") from exc
        model_id = (self._stages.get("train") or {}).get("model_id")
        self._set_stage("register", "complete", model_id=model_id)
        self._emit_fold("registering", 1.0, f"registered {model.get('name')}")

    # -- §26 / #717: a keypoint setup's teacher, before the distill's own stages ---- #
    def _teacher_stages(self) -> bool:
        """Detect and track the first minutes of one recording per acquisition setup (two
        processing runs), then wait for an example set clicked on those tracks. False when
        the run stopped here: to wait (the coordinator exits, and
        ``Experiment.continue_keypoint_setup`` relaunches it with the set, reusing every
        finished stage), or on a cancel."""
        from detail.api import run_spec

        cfg = self._config
        first = cfg["recording_ids"][0]
        detections = self._processing_stage(
            "detect",
            "detecting",
            "detection",
            lambda: {
                "model_id": cfg["detector_model_id"],
                "prompt": Prompt(targets=[]),  # a distilled detector is closed-set
                "conf_threshold": cfg["conf_threshold"],
                "compute": ComputeTarget.from_dict(cfg["detect_compute"]),
                "teacher_window_s": cfg["window_s"],
            },
        )
        if self._cancel_requested():
            self._finalize("cancelled")
            return False
        class_names = setup_class_names(self._record)
        if class_names:
            # Each animal class tracked on its own, then merged per teacher recording: the
            # flow a social experiment uses (§26 *One example set per detection class*).
            tracks = self._class_tracking_stage(class_names, first, detections)
            if self._cancel_requested():
                self._finalize("cancelled")
                return False
            recording = self._experiment.get_recording(first)
            self._processing_stage(
                "merge",
                "merging",
                "track_merge",
                lambda: {
                    "source_specs": [
                        run_spec.spec_from_run(
                            recording, "tracking", tracks[cls][first]
                        )
                        for cls in class_names
                    ]
                },
            )
        else:
            self._processing_stage(
                "track",
                "tracking",
                "tracking",
                lambda: {
                    "tracker_id": cfg.get("tracker_id"),
                    "class_name": cfg["class_name"],
                    "detection_spec": run_spec.spec_from_run(
                        self._experiment.get_recording(first),
                        "detection",
                        detections[first],
                    ),
                },
            )
        sets = (
            {"class_example_sets": cfg.get("class_example_sets")}
            if class_names
            else {"example_set_id": cfg.get("example_set_id")}
        )
        if not next(iter(sets.values())):
            self._set_stage("examples", AWAITING_REFERENCES)
            self._emit_fold("labeling", 0.0, "waiting for example clicks")
            self._finalize(AWAITING_REFERENCES)
            return False
        self._set_stage("examples", "complete", **sets)
        if self._cancel_requested():
            self._finalize("cancelled")
            return False
        return True

    def _class_tracking_stage(
        self, class_names: list[str], first: str, detections: dict
    ) -> dict[str, dict[str, str]]:
        """A setup's ``track`` stage with several classes: one class-filtered tracking run
        per class over the teacher recordings, ``{class: {recording_id: tracking run id}}``.
        Each class's processing run is recorded as it starts (``processing_run_ids``), so a
        cancel reaches it and a relaunch adopts one that finished after this coordinator
        died, as ``_processing_stage`` does for one run.
        """
        from detail.api import run_spec

        stage = dict(self._stages.get("track") or {})
        if stage.get("status") == JobStatus.COMPLETE.value and stage.get(
            "run_ids_by_class"
        ):
            self._emit_fold("tracking", 1.0, "reused tracking")
            return {c: dict(r) for c, r in stage["run_ids_by_class"].items()}
        done = {c: dict(r) for c, r in (stage.get("run_ids_by_class") or {}).items()}
        processing = dict(stage.get("processing_run_ids") or {})
        detection_spec = run_spec.spec_from_run(
            self._experiment.get_recording(first), "detection", detections[first]
        )
        n = len(class_names)
        for i, cls in enumerate(class_names):
            if cls in done:
                continue
            prior = processing.get(cls)
            if not (
                prior
                and _processing_run_status(self._experiment.dir, prior) == "complete"
            ):
                self._emit_fold("tracking", i / n, f"tracking {cls}")
                try:
                    run = self._experiment.run_processing(
                        "tracking",
                        list(self._config["recording_ids"]),
                        {
                            "tracker_id": self._config.get("tracker_id"),
                            "class_name": cls,
                            "detection_spec": detection_spec,
                        },
                    )
                except (ValueError, FileNotFoundError, ModelNotReadyError) as exc:
                    self._set_stage("track", "failed")
                    raise _StageFailed(f"tracking {cls}: {exc}") from None
                processing[cls] = run.id
                self._set_stage(
                    "track",
                    "running",
                    processing_run_ids=dict(processing),
                    run_ids_by_class=done,
                )

                def report(ev, i=i, cls=cls) -> None:
                    total = ev.recordings_total or 1
                    frac = (
                        ev.recordings_done + processing_inflight_fraction(ev)
                    ) / total
                    self._emit_fold("tracking", (i + frac) / n, f"tracking {cls}")

                self._pump(run, report, "tracking")
                run.wait()
                if run.status != JobStatus.COMPLETE.value:
                    self._set_stage(
                        "track", run.status, processing_run_ids=dict(processing)
                    )
                    raise _StageFailed(f"tracking {cls} {run.status}: {run.error}")
                prior = run.id
            record = runs_store.read_run(self._experiment.dir, prior)
            done[cls] = {
                sid: entry["run_id"]
                for sid, entry in (record.get("per_recording") or {}).items()
            }
            self._set_stage(
                "track",
                "running",
                processing_run_ids=dict(processing),
                run_ids_by_class=done,
            )
        self._set_stage(
            "track",
            "complete",
            processing_run_ids=dict(processing),
            run_ids_by_class=done,
        )
        self._emit_fold("tracking", 1.0, "tracking done")
        return done

    def _setup_teacher(self) -> None:
        """Resolve a setup's patch teacher now that its tracks exist: the method with the
        example set, given the teacher tracking runs (``keypoint_distill._resolve_patch_teacher``,
        the same check a distill's launch makes), copied into the record so a relaunch
        replays it rather than the live set."""
        from detail.api import keypoint_distill

        cfg = self._config
        if cfg.get("teacher"):
            return
        # With several classes, the merged tracks, each class matched against its own set.
        tracks = setup_teacher_tracks({**self._record, "stages": self._stages})
        try:
            teacher, compute = keypoint_distill._resolve_patch_teacher(
                self._experiment,
                cfg.get("example_set_id"),
                tracks,
                cfg["keypoints_model_id"],
                cfg.get("keypoints_model_variant"),
                ComputeTarget.from_dict(cfg["keypoints_compute"]),
                cfg.get("keypoint_settings"),
                (
                    cfg.get("class_example_sets")
                    if setup_class_names(self._record)
                    else None
                ),
            )
        except (ValueError, FileNotFoundError, ModelNotReadyError) as exc:
            self._set_stage("label", "failed")
            raise _StageFailed(f"the teacher: {exc}") from None
        cfg["teacher"] = {**teacher, "compute": compute.to_dict()}
        self._record_best_effort()

    def _processing_stage(
        self, name: str, phase: str, module_id: str, config: Callable[[], dict]
    ) -> dict[str, str]:
        """One teacher stage as a processing run over the setup's recordings; returns each
        recording's run id. ``config`` is built only when the stage runs, so a reused stage
        never resolves an upstream it does not need.

        Reused when it already finished — on the relaunch after the example clicks, and on a
        resume — including a processing run that finished after this coordinator died.
        """
        stage = self._stages.get(name) or {}
        if stage.get("status") == JobStatus.COMPLETE.value and stage.get("run_ids"):
            self._emit_fold(phase, 1.0, f"reused {phase}")
            return dict(stage["run_ids"])
        prior = stage.get("processing_run_id")
        if prior and _processing_run_status(self._experiment.dir, prior) == "complete":
            return self._adopt_processing_run(name, phase, prior)
        self._emit_fold(phase, 0.0, f"starting {phase}")
        try:
            run = self._experiment.run_processing(
                module_id, list(self._config["recording_ids"]), config()
            )
        except (ValueError, FileNotFoundError, ModelNotReadyError) as exc:
            self._set_stage(name, "failed")
            raise _StageFailed(f"{phase}: {exc}") from None
        self._set_stage(name, "running", processing_run_id=run.id)

        def report(ev) -> None:
            total = ev.recordings_total or 1
            frac = (ev.recordings_done + processing_inflight_fraction(ev)) / total
            message = getattr(ev.inner, "message", None) or (
                f"{phase}: {ev.recordings_done}/{total} recordings"
            )
            self._emit_fold(phase, frac, message)

        self._pump(run, report, phase)
        run.wait()
        if run.status != JobStatus.COMPLETE.value:
            self._set_stage(name, run.status, processing_run_id=run.id)
            raise _StageFailed(f"{phase} {run.status}: {run.error}")
        return self._adopt_processing_run(name, phase, run.id)

    def _adopt_processing_run(self, name: str, phase: str, run_id: str) -> dict:
        record = runs_store.read_run(self._experiment.dir, run_id)
        run_ids = {
            sid: entry["run_id"]
            for sid, entry in (record.get("per_recording") or {}).items()
        }
        self._set_stage(name, "complete", processing_run_id=run_id, run_ids=run_ids)
        self._emit_fold(phase, 1.0, f"{phase} done")
        return run_ids

    def _model_resumable(self, model_id: str) -> bool:
        with contextlib.suppress(FileNotFoundError):
            record = models_store.read_model_yaml(
                models_store.model_dir(self._experiment.dir, model_id)
            )
            return record.get("status") in RESUMABLE_TRAINING_STATUSES
        return False

    def _emit_fold(
        self,
        phase: str,
        inner_fraction: float,
        message: str,
        eta_s: float | None = None,
    ) -> None:
        """Emit one folded progress event.

        ``overall_fraction`` is the **stage-weighted** fold — for the bar, and only the
        bar. The ETA must not be derived from it: the weights are nominal (a stage's share
        of the bar is not its share of wall clock), so feeding them to a time
        extrapolation under-reported the remaining time. The ETA comes from the running stage
        instead — its own runner's ``eta_s`` where there is one (the trainer already
        computes an exact per-epoch estimate), else a linear estimate over the stage's own
        clock, which is why ``_stage_start`` resets at each phase boundary.

        During training — the last stage — stage-remaining is run-remaining, so the number
        is exact. During labeling it is labeling's own remaining time: a whole-run estimate
        is unknowable there, because ``total_epochs`` does not exist until training starts.
        """
        if phase != self._phase:
            self._phase = phase
            self._stage_start = time.monotonic()
        overall = distill_overall_fraction(phase, inner_fraction, self._weights)
        now = time.monotonic()
        stage = _PHASE_STAGE.get(phase)
        stage_index = (
            self._stage_names.index(stage) if stage in self._stage_names else 0
        )
        if eta_s is None:
            eta_s = remaining_seconds(inner_fraction, now - self._stage_start)
        self._emit(
            DistillProgress(
                phase=phase,
                stage_index=stage_index,
                stage_count=len(self._stage_names),
                inner_fraction=max(0.0, min(1.0, inner_fraction)),
                overall_fraction=overall,
                elapsed_s=now - self._start,
                eta_s=eta_s,
                message=message,
            )
        )

    def _pump(self, run, report: Callable, phase: str) -> None:
        """Forward a stage's progress to the bar. Reporting only — never raises.

        ``run.progress()`` polls the worker's envelope, so it is file I/O. Letting it raise
        would skip the ``run.wait()`` that follows — the only adoption of the job's real status —
        and, because this driver catches only ``_StageFailed``, would fail a whole distill run
        from a read that had nothing to do with the training.
        """
        try:
            for event in run.progress():
                report(event)
        except Exception as exc:  # noqa: BLE001 — reporting on the work, not the work
            _log.warning(
                "%s progress stream stopped early (%s: %s) — the stage is unaffected",
                phase,
                type(exc).__name__,
                exc,
            )

    def _cancel_requested(self) -> bool:
        return distill_store.cancel_requested(self._experiment.dir, self.id)

    def _set_stage(self, name: str, status: str, **ids) -> None:
        """Record a stage boundary. Best-effort, like its processing-run twin.

        #334 gave this class the `emit` half of that fix and not the record half: a stage
        write is a note about where the run has got to, and `_set_stage("train", "complete",
        model_id=…)` runs *after* the training succeeded. Raising there would let the file
        that describes the run decide whether the run failed — and because `run()`'s handler
        is narrowly `_StageFailed`, a `PermissionError` here does not even become a distill
        failure, it escapes to the coordinator's backstop and becomes "coordinator error".

        The terminal write in `_finalize` stays raising: dropping it would leave the record
        without the status the run actually reached, which is a different problem.
        """
        stage = dict(self._stages.get(name) or {})
        stage["status"] = status
        stage.update(ids)
        self._stages[name] = stage
        self._record_best_effort()

    def _record_best_effort(self) -> None:
        try:
            self._write_record()
        except Exception as exc:  # noqa: BLE001 — the index, not the work
            _log.warning(
                "could not update distill record %s (%s: %s) — retried at the next stage",
                self.id,
                type(exc).__name__,
                exc,
            )

    def _finalize(self, status: str, error: str | None = None) -> None:
        self._status = status
        self._error = error
        self._write_record()

    def _write_record(self) -> None:
        self._record["stages"] = self._stages
        self._record["status"] = self._status
        if self._error:
            self._record["error"] = self._error
        self._record["duration_s"] = time.monotonic() - self._start
        costs = [
            stage["estimated_cost_usd"]
            for stage in self._stages.values()
            if stage.get("estimated_cost_usd") is not None
        ]
        if costs:
            self._record["total_estimated_cost_usd"] = sum(costs)
        distill_store.write_run(self._experiment.dir, self.id, self._record)


class AttachedDistillRun(RunHandle):
    """Reattachable handle over a durable distill run (§12). Works from **any** process —
    the launching app, a GUI reopen, ``detail distill attach`` — by polling the
    coordinator's ``distill/<id>/progress.json`` + ``distill.yaml``. If the coordinator is
    gone, it reconciles the run to a terminal status so ``progress`` never hangs."""

    def __init__(
        self, experiment: "Experiment", run_id: str, poll_interval: float = 0.25
    ):
        self.id = run_id
        self._experiment = experiment
        self._poll = poll_interval

    @property
    def owns_worker(self) -> bool:
        """False — the worker belongs to whoever launched it, which is the point of reattaching.

        This is what makes ``Ctrl-C`` detach rather than cancel on every surface that streams a
        reattached run, without any of them having to remember to ask for it.
        """
        return False

    @property
    def experiment(self) -> "Experiment":
        """The experiment this run belongs to (a distill can be launched for any experiment
        from the machine-global Distill surface, so a caller needs to know which)."""
        return self._experiment

    @property
    def status(self) -> str:
        return distill_store.read_run(self._experiment.dir, self.id).get("status")

    @property
    def error(self) -> str | None:
        return distill_store.read_run(self._experiment.dir, self.id).get("error")

    def progress(self) -> Iterator[DistillProgress]:
        last = None
        dead_strikes = 0
        while True:
            record = distill_store.read_run(self._experiment.dir, self.id)
            data = distill_store.read_progress(self._experiment.dir, self.id)
            if data is not None and data != last:
                yield distill_progress_from_dict(data)
                last = data
            if record.get("status") not in ACTIVE_STATUSES:
                reap_coordinator(
                    distill_store.read_coordinator_pid(self._experiment.dir, self.id)
                )
                return
            pid = distill_store.read_coordinator_pid(self._experiment.dir, self.id)
            if pid is not None and not pid_alive(pid):
                dead_strikes += 1
                if dead_strikes >= 2:
                    self._experiment._reconcile_distill_run(record)
                    if (
                        distill_store.read_run(self._experiment.dir, self.id).get(
                            "status"
                        )
                        not in ACTIVE_STATUSES
                    ):
                        return
                    dead_strikes = 0
            else:
                dead_strikes = 0
            time.sleep(self._poll)

    def wait(self) -> "AttachedDistillRun":
        for _ in self.progress():
            pass
        return self

    def cancel(self) -> None:
        self._experiment.cancel_distill(self.id)


@dataclass
class AnalysisRunRef:
    """Lightweight index entry for an analysis run (§16) — for listings."""

    id: str
    module_id: str
    recording_ids: list[str]
    status: str
    #: Read by `api/active.py` for the Active-runs "started" column, and to sort the
    #: listing. Every ref here lacked it, so four of the six run kinds showed a blank
    #: cell and sorted last in arbitrary order.
    created_at: str = ""
    #: An analysis run is local-only today — see ClustererRef's identical fields.
    compute: dict | None = None
    duration_s: float | None = None
    estimated_cost_usd: float | None = None
    #: What a person called the run, when it has a name (a duplicate is "<source> (copy)");
    #: ``""`` for the runs that have only their id.
    name: str = ""


@dataclass
class ExportRunRef:
    """Lightweight index entry for an export run (#354) — for listings.

    Carries ``created_at``, which the other refs do not: `api/active.py` reads it per record to
    fill the Active-runs "started" column and to sort the listing, so a ref without one shows a
    blank cell and sorts last.
    """

    id: str
    subject_kind: str
    subject_id: str
    target_ids: list[str]
    status: str
    created_at: str = ""
    #: The ``{root, rel}`` alias and the folder under it — together, where the files went. Free
    #: to carry (the listing already reads the whole record to build this ref) and the one thing
    #: a *finished* export is looked up for: its output is outside the experiment, so unlike
    #: every other run kind the id does not lead to it.
    folder: str = ""
    destination: dict = field(default_factory=dict)
    #: An export run is local-only today — see ClustererRef's identical fields.
    compute: dict | None = None
    duration_s: float | None = None
    estimated_cost_usd: float | None = None


def _scope_dict(scope) -> dict:
    """An `ExportScope` as the job's JSON-native dict — empty when it means "everything"."""
    if scope is None or scope.is_all():
        return {}
    out: dict = {}
    if scope.cluster is not None:
        out["cluster"] = scope.cluster
    if scope.instance_ids:
        out["instance_ids"] = list(scope.instance_ids)
    if scope.frames is not None:
        # A list, not a tuple: this goes through YAML, and a tuple writes as `!!python/tuple`,
        # which `yaml.safe_load` then refuses — the run does the work and dies reporting it.
        # `ExportScope.__post_init__` turns it back into a tuple on the way out.
        out["frames"] = list(scope.frames)
    if scope.recording_ids:
        out["recording_ids"] = list(
            scope.recording_ids
        )  # a list, for the same reason as `frames`
    return out


class ExportRun(RunHandle):
    """Run handle for an export run (#354); reconciles the record's status to the executor's
    terminal status on ``wait``, the way every other single-job run does."""

    def __init__(self, inner: ExecutorRunHandle, experiment: "Experiment", run_id: str):
        self._inner = inner
        self._experiment = experiment
        self.id = run_id

    @property
    def status(self) -> str:
        return self._inner.status

    @property
    def error(self) -> str | None:
        return self._inner.error

    def progress(self):
        return self._inner.progress()

    def cancel(self) -> None:
        self._inner.cancel()

    def wait(self) -> "ExportRun":
        self._inner.wait()
        rdir = export_runs_store.export_dir(self._experiment.dir, self.id)
        status = reconciled_status(
            export_runs_store.read_export_yaml(rdir).get("status"), self._inner.status
        )
        if status is not None:
            export_runs_store.set_export_status(rdir, status, self._inner.error)
        return self


def _remote_compute(value) -> ComputeTarget | None:
    """``value`` (a ``ComputeTarget``, its dict, or ``None``) as a target when it is remote."""
    if value is None:
        return None
    target = (
        value if isinstance(value, ComputeTarget) else ComputeTarget.from_dict(value)
    )
    return target if target.backend == "remote" else None


class AnalysisRun(RunHandle):
    """Run handle for an analysis run (§16); reconciles ``analysis.yaml``'s status
    to the executor's terminal status on ``wait`` (the runner writes the full
    record on success; a failure leaves the initial ``running`` record to fix up)."""

    def __init__(self, inner: ExecutorRunHandle, experiment: "Experiment", run_id: str):
        self._inner = inner
        self._experiment = experiment
        self.id = run_id

    @property
    def status(self) -> str:
        return self._inner.status

    @property
    def error(self) -> str | None:
        return self._inner.error

    def progress(self):
        return self._inner.progress()

    def wait(self) -> "AnalysisRun":
        self._inner.wait()
        adir = analysis_store.analysis_dir(self._experiment.dir, self.id)
        # The runner writes the authoritative terminal status (complete / partial /
        # failed, with per-recording detail); `reconciled_status` says when the job
        # envelope overrides it — a record the runner never finalized, and any record
        # claiming something other than a failure over a failed envelope.
        status = reconciled_status(
            analysis_store.read_analysis_yaml(adir).get("status"), self._inner.status
        )
        if status is not None:
            analysis_store.set_analysis_status(adir, status, self._inner.error)
        return self

    def cancel(self) -> None:
        self._inner.cancel()


# The cached still that identifies an experiment in a listing (``preview_image``). Lives
# in the experiment, beside its config, the way a model's preview lives beside model.yaml.
PREVIEW_FILENAME = "preview.jpg"
PREVIEW_WIDTH = 320


def _refuse_name_in_another_location(experiment_id: str, location: str) -> None:
    """Refuse a name another connected location already holds.

    The Experiments list shows every location together, so two experiments with one name
    would be two cards nobody can tell apart.
    """
    for path in locations.list_experiment_dirs():
        if path.name != experiment_id:
            continue
        other = locations.location_of(path)
        if other != location:
            raise FileExistsError(
                f"an experiment named {experiment_id!r} already exists in location "
                f"{other!r} ({path.parent.as_posix()}); choose another name"
            )


class Experiment:
    def __init__(self, experiment_dir: Path | str) -> None:
        self._dir = Path(experiment_dir)

    @classmethod
    def create(
        cls,
        name: str,
        parent_dir: Path | str | None = None,
        *,
        location: str | None = None,
    ) -> "Experiment":
        """Create the experiment directory structure and config.yaml.

        Args:
            name: The experiment's name, which is also its folder's name.
            parent_dir: An explicit folder to create it in. Not listed by
                :meth:`list_all` unless that folder is an experiment location.
            location: The experiment location (§2) to create it in. With neither this
                nor ``parent_dir``, the ``default`` location.

        Raises:
            ValueError: Both ``parent_dir`` and ``location``, an empty name, or an
                unknown location.
            LocationNotConnectedError: The location's drive is not plugged in.
            FileExistsError: The name is taken in this location or another one.
        """
        if parent_dir is not None and location is not None:
            raise ValueError("pass parent_dir or location, not both")
        if parent_dir is None:
            location = location or locations.DEFAULT_LOCATION
            parent_dir = locations.location_root(location)
            _refuse_name_in_another_location(name, location)
        return cls(experiment_store.create_experiment(parent_dir, name))

    @classmethod
    def open(cls, experiment_dir: Path | str) -> "Experiment":
        """Open an existing experiment (validates config.yaml)."""
        experiment_store.read_config(experiment_dir)
        return cls(experiment_dir)

    @classmethod
    def list_all(cls) -> list["Experiment"]:
        """All experiments in every connected experiment location (§2), archived ones
        included (filter on :attr:`archived`)."""
        return [cls(p) for p in locations.list_experiment_dirs()]

    @classmethod
    def import_archive(
        cls,
        archive: Path | str,
        *,
        replace: bool = False,
        name: str | None = None,
        location: str | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> "Experiment":
        """Import an exported experiment bundle into an experiment location (§15).
        Creates it, or replaces a same-id experiment when ``replace`` is True; ``name``
        overrides the target id (import as a separate copy). ``location`` defaults to
        the ``default`` location.

        Args:
            progress: ``(stage, fraction, detail)`` callback — ``reading`` →
                ``extracting`` → ``verifying`` → ``placing``. A callback that raises stops
                the import with nothing placed.

        A recording whose video is referenced in place is looked for where the exporting machine
        had it, and its video root registered here when a video there matches (#916);
        `referenced_video_missing` lists what was not found.

        Raises:
            ValueError: Malformed/corrupt archive, unsupported schema_version, or an
                unknown location.
            LocationNotConnectedError: The location's drive is not plugged in.
            FileExistsError: Target id exists and ``replace`` is False, or exists in
                another location.
        """
        location = location or locations.DEFAULT_LOCATION
        dest_root = locations.location_root(location)
        _refuse_name_in_another_location(
            name or export_store.read_manifest(archive)["id"], location
        )
        dest = export_store.import_experiment(
            archive, dest_root, replace=replace, name=name, progress=progress
        )
        experiment = cls(dest)
        # Archiving is the sender's tidying, not a fact about the data: an import of an
        # archived experiment would otherwise land hidden from the list it was imported into.
        if experiment.archived:
            experiment.unarchive()
        return experiment

    @property
    def location(self) -> str | None:
        """The experiment location this experiment is in, or ``None`` for a folder that
        is not one (created with an explicit ``parent_dir``)."""
        return locations.location_of(self._dir)

    def export(
        self,
        dest: Path | str | None = None,
        *,
        include_video: bool = False,
        include_datasets: bool = False,
        gzip: bool = False,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> Path:
        """Pack this experiment into a single portable archive (§15). Slim by default
        (outputs + metadata + models; no source video / datasets) — pass
        ``include_video`` / ``include_datasets`` for a full transfer. Returns the
        archive path (default ``<workspace>/<id>.segproj.tar``).

        ``progress`` is a ``(stage, fraction, detail)`` callback — ``scanning`` →
        ``checking`` → ``hashing`` → ``packing``. The GUI does not call this directly: its
        Export runs the catalogue's ``bundle`` target as an export run (`run_export`), so the
        export is listed under Active processes and survives the window closing."""
        if dest is None:
            dest = self._dir.parent / f"{self._dir.name}.segproj.tar"
        return export_store.export_experiment(
            self._dir,
            dest,
            include_video=include_video,
            include_datasets=include_datasets,
            gzip=gzip,
            progress=progress,
        )

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def name(self) -> str:
        """Experiment display name (from config.yaml; decoupled from the folder)."""
        return experiment_store.read_config(self._dir)["name"]

    @property
    def description(self) -> str:
        """Experiment description (empty string if unset)."""
        return experiment_store.read_config(self._dir).get("description", "")

    @property
    def modified_at(self) -> float:
        """Unix mtime of the most recent experiment-level change.

        The max mtime across config.yaml, ``recordings/`` and ``runs/`` — a cheap
        proxy for "last activity" without walking the whole tree.
        """
        candidates = [
            self._dir / "config.yaml",
            self._dir / "recordings",
            self._dir / "runs",
        ]
        return max((p.stat().st_mtime for p in candidates if p.exists()), default=0.0)

    def preview_image(self, width: int = PREVIEW_WIDTH) -> Path | None:
        """A cached still identifying this experiment, or ``None`` if one can't be made.

        The frame comes from the **middle** of the first recording whose video is on this
        machine: the opening frames are often a blank lead-in and the last ones an empty
        arena, so a midpoint is the one place likely to show the animal. It is written once
        to ``<experiment>/preview.jpg`` and regenerated only when older than
        ``modified_at``, so listing a workspace costs a file read rather than a decode per
        experiment.

        Never raises. ``None`` means there is nothing to show — no recordings, no video, a
        referenced video that isn't on this machine (§5 reference-in-place), or a cache that
        couldn't be written (read-only media) — and the caller renders a placeholder.
        """
        import cv2

        path = self._dir / PREVIEW_FILENAME
        with contextlib.suppress(OSError):
            if path.is_file() and path.stat().st_mtime >= self.modified_at:
                return path
        frame = self._middle_frame()
        if frame is None:
            return None
        height = max(int(frame.shape[0] * width / max(frame.shape[1], 1)), 1)
        thumb = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        try:
            cv2.imwrite(str(path), rgb_to_bgr(thumb))  # get_frame returns RGB
        except (OSError, cv2.error):
            return None
        return path if path.is_file() else None

    def _middle_frame(self):
        """The midpoint frame of the first recording with usable video, else ``None``."""
        for recording in self.recordings:
            if not recording.video_available:
                continue
            counts = recording.part_frame_counts
            total = sum(counts)
            if total <= 0:
                continue
            target = total // 2
            for part_index, count in enumerate(counts):
                if target < count:
                    with contextlib.suppress(IndexError, ValueError, OSError):
                        return recording.get_frame(part_index, target)
                    break
                target -= count
        return None

    def set_metadata(
        self, name: str | None = None, description: str | None = None
    ) -> None:
        """Edit experiment identity (name / description) in config.yaml.

        Only the given fields change; the experiment directory is not renamed.

        Raises:
            ValueError: If ``name`` is given but empty.
        """
        experiment_store.set_metadata(self._dir, name=name, description=description)

    # --- notes (§15) -------------------------------------------------------- #
    def notes(self, kind: str, entry_id: str = "") -> str:
        """The user's notes on one entry, or ``""`` when it has none.

        ``kind`` is one of ``storage.notes.KINDS`` (``experiment``, ``recording``, ``run``,
        ``analysis_run``, ``dataset``, ``model``, ``distill``, ``export_run``,
        ``comparison``); ``experiment`` takes no ``entry_id``.

        Raises:
            ValueError: If ``kind`` is not an entry kind, or the id is missing or extra.
            FileNotFoundError: If there is no such entry.
        """
        return notes_store.read_notes(notes_store.entry_dir(self._dir, kind, entry_id))

    def set_notes(self, kind: str, entry_id: str, text: str) -> None:
        """Replace the user's notes on one entry; empty ``text`` clears them.

        Safe while the entry's run is still going: the notes are a file of their own
        (``notes.md``), never a field in a record a worker is rewriting.

        Raises:
            ValueError: If ``kind`` is not an entry kind, or the id is missing or extra.
            FileNotFoundError: If there is no such entry.
        """
        notes_store.write_notes(notes_store.entry_dir(self._dir, kind, entry_id), text)

    @property
    def archived(self) -> bool:
        """Whether the experiment is archived: kept whole, but left off the Experiments
        list unless archived experiments are asked for."""
        return bool(
            experiment_store.read_config(self._dir).get(experiment_store.ARCHIVED_KEY)
        )

    def archive(self) -> None:
        """Archive the experiment. Nothing in it changes and it still opens and runs;
        only the Experiments list hides it. Undone by :meth:`unarchive`."""
        experiment_store.set_archived(self._dir, True)

    def unarchive(self) -> None:
        """Put an archived experiment back on the Experiments list."""
        experiment_store.set_archived(self._dir, False)

    # --- camera view (§10) -------------------------------------------------- #
    @property
    def camera_view(self) -> camera_view_mod.CameraView:
        """The experiment's default camera view — what every recording inherits unless it
        sets its own (``Recording.camera_view`` resolves the two). ``view`` is ``None`` when
        no default is set."""
        return camera_view_mod.resolve({}, experiment_store.read_config(self._dir))

    def set_camera_view(self, view: str | None, note: str | None = None) -> None:
        """Set the experiment's default camera view, or remove it with ``None``.

        ``view`` is one of ``core.camera_view.VIEWS`` (``top_down``, ``side``,
        ``angled_down``, ``other``); ``note`` describes an ``other`` view and is refused
        on any other.

        Raises:
            ValueError: If the view is unknown, ``other`` has no note, or a note is given
                with a view other than ``other``.
        """
        experiment_store.set_camera_view(self._dir, view, note)

    # --- class names (§1) --------------------------------------------------- #
    def classes(self) -> list[str]:
        """Every detection class this experiment knows, by stored name, sorted.

        Read off records, never zarr: each detection run's prompt targets and per-class
        thresholds, the classes of any imported model a run used, and every trained
        detection model's classes — so a class can be named before it is first detected.
        A renamed class stays listed after its runs are deleted, so its name can be cleared.
        """
        roster: set[str] = set(experiment_store.read_class_names(self._dir))
        run_models: set[str] = set()
        for record in self.processing_runs:
            if record.get("module_id") != "detection":
                continue
            config = record.get("config") or {}
            for target in (config.get("prompt") or {}).get("targets") or []:
                if target.get("class_name"):
                    roster.add(str(target["class_name"]))
            roster.update(str(c) for c in config.get("class_conf_thresholds") or {})
            if config.get("model_id"):
                run_models.add(str(config["model_id"]))
        for rec in dirscan.read_each(
            models_store.discover_models(self._dir),
            lambda m: models_store.read_model_yaml(
                models_store.model_dir(self._dir, m)
            ),
        ):
            role = rec.get("role") or (
                "keypoints" if rec.get("task") == "pose" else "detection"
            )
            run_models.discard(rec.get("id"))
            if role == "detection":
                roster.update(str(c["name"]) for c in rec.get("classes") or [])
        # Only a run's model from outside the experiment is an imported one worth reading.
        for ref in api_models.imported_models("detection") if run_models else []:
            if ref.id in run_models:
                roster.update(ref.classes)
        return sorted(roster)

    def class_names(self) -> dict[str, str]:
        """``{stored class name: what it is called}`` for every class in :meth:`classes` —
        the display map, the class twin of :meth:`recording_names`."""
        names = experiment_store.read_class_names(self._dir)
        return {
            stored: class_names_mod.label(stored, names) for stored in self.classes()
        }

    def class_label(self, stored: str) -> str:
        """What the class stored as ``stored`` is called in this experiment."""
        return class_names_mod.label(stored, self.renamed_classes())

    def renamed_classes(self) -> dict[str, str]:
        """``{stored: name}`` for the renamed classes only — one mtime-cached config read,
        no roster, so a surface labels classes it already holds on the UI thread
        (``core.class_names.label``)."""
        return experiment_store.read_class_names(self._dir)

    def rename_class(self, stored: str, name: str | None) -> None:
        """Call the class stored as ``stored`` ``name`` on every surface; ``None``, an empty
        name or the stored name itself puts back the stored name. No run is rewritten.

        Raises:
            KeyError: ``stored`` is not a class this experiment knows.
            ValueError: Another class is already called ``name``, or has it as its stored
                name.
        """
        self.rename_classes({stored: name})

    def rename_classes(self, changes: dict[str, str | None]) -> None:
        """:meth:`rename_class` for several classes at once, checked together — so two
        classes can trade names, which one at a time would refuse midway.

        Raises:
            KeyError: A class this experiment does not know.
            ValueError: As :meth:`rename_class`, for the names once all are applied.
        """
        known = self.classes()
        unknown = sorted(set(changes) - set(known))
        if unknown:
            raise KeyError(
                f"no class {', '.join(map(repr, unknown))} in this experiment — known: "
                f"{', '.join(known) or '(none)'}"
            )
        names = experiment_store.read_class_names(self._dir)
        for stored, name in changes.items():
            names = class_names_mod.renamed(names, stored, name)
        class_names_mod.check(names, known=known)
        experiment_store.write_class_names(self._dir, names)

    def resolve_class(self, ref: str) -> str:
        """The stored name of the class ``ref`` names, by stored name **or** name — what a
        user types may be either. A stored name wins.

        Raises:
            KeyError: No class matches.
        """
        known = self.classes()
        stored = class_names_mod.resolve(
            ref, experiment_store.read_class_names(self._dir), known=known
        )
        if stored not in known:
            raise KeyError(
                f"no class {ref!r} in this experiment — known: "
                f"{', '.join(self.class_names().values()) or '(none)'}"
            )
        return stored

    def resolve_class_refs(self, config: dict) -> dict:
        """``config`` with every class it names resolved to the stored name
        (``core.class_names.CLASS_REF_FIELDS``) — applied where a run's config enters, so
        the record holds stored names whatever the user typed."""
        names = experiment_store.read_class_names(self._dir)
        if not names:
            return dict(config)
        return class_names_mod.resolve_config(config, names, known=self.classes())

    @property
    def recording_ids(self) -> list[str]:
        """The recording ids in id order — the listing, without the records.

        :attr:`recordings` builds a :class:`Recording` per id, and each one reads its
        ``recording.yaml`` to learn its name. A caller that only needs *how many* there
        are, or which ids exist, pays a file read per recording for a number it could
        have had from the directory listing.
        """
        return experiment_store.discover_recordings(self._dir)

    @property
    def recordings(self) -> list[Recording]:
        return [Recording(self._dir, sid) for sid in self.recording_ids]

    def create_recording(
        self, recording_id: str, metadata: dict[str, str] | None = None
    ) -> Recording:
        """Create a recording **at a chosen id** (identity only — video/streams are
        optional, added later) and seed its ``metadata.csv`` row (§11).

        The id is the caller's, which is what a script or a test wants. An *application*
        adding a recording a user is naming should call :meth:`add_recording` instead —
        it mints the id and leaves the name to the user.
        """
        recording_store.create_recording(
            self._dir, recording_id=recording_id, name=recording_id
        )
        metadata_store.upsert_row(self._dir, recording_id, metadata)
        return Recording(self._dir, recording_id)

    def add_recording(
        self, name: str, metadata: dict[str, str] | None = None
    ) -> Recording:
        """Add a recording the user has *named*: the tool mints the id, the name is theirs.

        ``name`` is normally the video's filename — a good starting label and a terrible
        identity, since the file gets moved, renamed, or joined by a second part. The id
        is opaque and permanent (``recording_000``); the name is editable and unique, and
        is what every surface displays.

        A name already taken is **suffixed** (``m27_day3`` → ``m27_day3_2``) rather than
        refused: the caller here is the tool, importing a folder where two files can
        legitimately share a name, and failing the import over a label would be absurd.
        A name a *user* types goes through :meth:`Recording.set_name`, which refuses.
        """
        recording_id = experiment_store.next_recording_id(self._dir)
        recording_store.create_recording(
            self._dir, recording_id=recording_id, name=self.unique_name(name)
        )
        metadata_store.upsert_row(self._dir, recording_id, metadata)
        return Recording(self._dir, recording_id)

    def missing_previews(self, recording_ids: list[str] | None = None) -> list:
        """Preview builds missing on this machine across ``recording_ids`` (default: every
        recording), proxies before masks within each recording (§3 *Video preview cache*).
        One stat per part once this machine's cache knows the file."""
        from detail.api import previews

        tasks = []
        for recording_id in recording_ids or self.recording_ids:
            recording = self.get_recording(recording_id)
            tasks.extend(previews.recording_tasks(recording.dir, recording_id))
        return tasks

    def build_previews(
        self,
        recording_ids: list[str] | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> "previews.PreviewBuildReport":
        """Build every missing preview for ``recording_ids`` (default: all) under one bar;
        returns a ``previews.PreviewBuildReport`` (built count, and each failure with why).
        The bar is weighed by video length across the whole batch, and its phase line names
        the recording, part and build in hand."""
        from detail.api import previews

        return previews.build_tasks(
            self.dir,
            self.missing_previews(recording_ids),
            self.recording_names(),
            progress,
        )

    def recording_names(self) -> dict[str, str]:
        """``{recording_id: name}`` for every recording — the display map.

        The name is the handle on every surface, so this is read per page build. It rides
        the mtime-cached ``recording_store.read_recording``, the same read the recordings
        table already makes several of per row (no new I/O class — the UI-thread rule).
        """
        return {r.id: r.name for r in self.recordings}

    def unique_name(self, name: str, *, exclude: str | None = None) -> str:
        """``name``, or the first free ``name_2`` / ``name_3`` … variant.

        ``exclude`` skips one recording's own name, so renaming a recording to what it is
        already called is not a collision with itself.
        """
        taken = {n for rid, n in self.recording_names().items() if rid != exclude}
        if name not in taken:
            return name
        suffix = 2
        while f"{name}_{suffix}" in taken:
            suffix += 1
        return f"{name}_{suffix}"

    def get_recording(self, recording_id: str) -> Recording:
        return Recording(self._dir, recording_id)

    def referenced_video_missing(self) -> list[str]:
        """Ids of recordings whose video is referenced in place and is not on this machine.

        What is left after an import has looked where the exporting machine kept the video:
        locating one file of such a recording (`Recording.relink_video_part`) registers its
        folder, and every recording whose video is in that folder finds it too. A stat per part,
        so a surface calls this off the UI thread.
        """
        return [
            recording_id
            for recording_id in export_store.referenced_video_recordings(self._dir)
            if not self.get_recording(recording_id).video_available
        ]

    def resolve_recording(self, ref: str) -> Recording:
        """A recording by id **or** name — what a user types is the name (§11).

        An exact id wins, so a recording perversely *named* another's id can still be
        reached by its own id. Then a unique name. Anything else raises rather than
        guessing.

        Raises:
            KeyError: no recording matches, or the name is ambiguous (only possible in an
                experiment whose names predate the uniqueness rule).
        """
        names = self.recording_names()
        if ref in names:
            return Recording(self._dir, ref)
        matches = [rid for rid, name in names.items() if name == ref]
        if len(matches) == 1:
            return Recording(self._dir, matches[0])
        if not matches:
            raise KeyError(
                f"no recording {ref!r} in this experiment — "
                f"known: {', '.join(sorted(names.values())) or '(none)'}"
            )
        raise KeyError(f"{ref!r} is the name of several recordings: {sorted(matches)}")

    # --- spatial-setup propagation (§10, #342) ------------------------------ #
    # Push one recording's spatial setup onto a chosen set of others — the batch form of
    # the per-recording `import_*_from` pulls, run the other way (issue #342 — the
    # one-recording-at-a-time pulls, batched). The four categories are one table in `api.spatial_categories`,
    # which also owns the two rules that make this safe: a copy never replaces, and a
    # definition's keyframe motion and active window do not survive the trip.

    def propagate_spatial(
        self,
        category_key: str,
        source: str,
        *,
        targets: list[str] | None = None,
        names: list[str] | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> spatial_categories.PropagationResult:
        """Copy ``category_key`` from ``source`` onto ``targets`` (default: every other
        recording), never replacing what a target already has.

        A target that already holds this category is **skipped and reported**, not
        overwritten — the operation has no undo and spans many recordings, so a
        correction is a deliberate delete on the recording that needs it. So is a target
        the category does not fit — a calibration onto a video of another shape, which is
        another rig. A target that refuses for any other reason is skipped too rather than
        aborting the batch, the same contract as ``apply_stream_preset``.

        Returns:
            A ``PropagationResult`` — the recordings that took it, the ones that did not
            and why, and what the trip changed about the definitions themselves.

        Raises:
            KeyError: ``source`` (or a ref in ``targets``) matches no recording.
            ValueError: unknown ``category_key``, or ``source`` has nothing to copy —
                checked before any recording is touched (Tenet 10).
        """
        cat = spatial_categories.category(category_key)
        src = self.resolve_recording(source)
        if not cat.has_data(src):
            raise ValueError(f"recording {src.id!r} has no {cat.label.lower()} to copy")
        dests = self._propagate_targets(src, targets)
        stage = f"copying {cat.label.lower()}"
        total = len(dests)
        copied: list[str] = []
        skipped: list[tuple[str, str]] = []
        flattened: list[str] = []
        refused: list[tuple[str, str]] = []
        for index, dest in enumerate(dests):
            detail = f"{dest.name} ({index + 1}/{total})"
            if progress is not None:
                progress(stage, index / total, detail)
            reason = cat.occupied(dest) or cat.unfit(src, dest)
            if reason is not None:
                skipped.append((dest.id, reason))
            else:
                try:
                    added, flattened, refused = cat.give(src, dest, names)
                except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
                    skipped.append((dest.id, str(exc)))
                else:
                    # Only a target that actually received something counts as copied. A
                    # set-shaped kind can have every one of its names refused, which is
                    # not a failure and not a copy.
                    (copied if added else skipped).append(
                        dest.id
                        if added
                        else (dest.id, "nothing new to add — see the notes below")
                    )
            if progress is not None:
                progress(stage, (index + 1) / total, detail)
        return spatial_categories.PropagationResult(
            copied=copied, skipped=skipped, flattened=flattened, refused=refused
        )

    def _propagate_targets(
        self, src: Recording, targets: list[str] | None
    ) -> list[Recording]:
        """``targets`` resolved to recordings, defaulting to every other recording; the
        source is excluded defensively even if named explicitly."""
        if targets is None:
            return [r for r in self.recordings if r.id != src.id]
        return [r for ref in targets if (r := self.resolve_recording(ref)).id != src.id]

    def propagate_field_of_view(
        self, source: str, **kwargs
    ) -> spatial_categories.PropagationResult:
        """Copy ``source``'s field of view onto recordings that have none (Section 10).
        See ``propagate_spatial``."""
        return self.propagate_spatial("fov", source, **kwargs)

    def propagate_calibration(
        self, source: str, **kwargs
    ) -> spatial_categories.PropagationResult:
        """Copy ``source``'s distance calibration onto recordings that have none
        (Section 10). See ``propagate_spatial``."""
        return self.propagate_spatial("calibration", source, **kwargs)

    def propagate_rois(
        self, source: str, **kwargs
    ) -> spatial_categories.PropagationResult:
        """Copy ``source``'s regions of interest onto recordings that have none
        (Section 10). See ``propagate_spatial``."""
        return self.propagate_spatial("roi", source, **kwargs)

    def propagate_points(
        self, source: str, **kwargs
    ) -> spatial_categories.PropagationResult:
        """Copy ``source``'s points of interest onto recordings that have none
        (Section 10). See ``propagate_spatial``."""
        return self.propagate_spatial("point", source, **kwargs)

    def import_behavior_labels(
        self,
        manifest_path: Path | str,
        *,
        name: str = "behavior",
        importer: str = "behavior_csv",
        progress: Callable[[int, int, str, float, str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Batch-import behavior ground-truth labels onto existing recordings (Section 3).

        Loads + validates the labels manifest (all problems reported upfront,
        Tenet 10), then adds each recording's file as a ``behavior``-role stream named
        ``name`` (default ``"behavior"``), each resolved against its own timestamp
        map. Every file in one batch uses the same ``importer`` (``behavior_csv`` or
        ``boris``). ``progress(index, total, recording_id, fraction, status)`` reports
        per recording. Returns the skipped recordings as ``[(recording_id, error), ...]``.

        Raises:
            FileNotFoundError: If the manifest file is absent.
            ValueError: If the manifest is malformed or references an unknown recording
                or missing file (nothing is imported).
        """
        from detail.api import imports as imports_mod
        from detail.storage import labels_manifest as labels_store

        manifest = labels_store.load_labels_manifest(manifest_path)
        mapping = labels_store.resolve_labels_manifest(
            manifest_path, manifest, experiment_store.discover_recordings(self._dir)
        )
        return imports_mod.import_label_batch(
            self, mapping, name=name, importer=importer, on_progress=progress
        )

    # --- heading definitions (§14 egocentric alignment) ----------------------- #
    def heading_definitions(self) -> list[dict]:
        """Saved heading definitions, oldest first: ``{name, front, back}`` (``[]`` if none)."""
        return heading_store.list_definitions(self._dir)

    def get_heading_definition(self, name: str) -> dict:
        """Raises: ValueError if no definition named ``name`` exists."""
        return heading_store.get_definition(self._dir, name)

    def save_heading_definition(
        self, name: str, front: list[str], back: list[str], *, overwrite: bool = False
    ) -> dict:
        """Save a heading definition: the points whose mean is the front and the points whose
        mean is the back. Checked against a skeleton when a run uses it, not here.

        Raises:
            ValueError: An invalid name or definition, or the name exists and ``overwrite``
                is False.
        """
        return heading_store.save_definition(
            self._dir, name, front, back, overwrite=overwrite
        )

    def delete_heading_definition(self, name: str) -> None:
        """Raises: ValueError if no definition named ``name`` exists."""
        heading_store.delete_definition(self._dir, name)

    def export_heading_definition(self, name: str, out_path: Path | str) -> None:
        """Write a saved definition to a standalone, shareable YAML file.

        Raises:
            ValueError: No definition named ``name`` exists.
        """
        heading_store.export_definition(self._dir, name, out_path)

    def import_heading_definition(
        self, in_path: Path | str, *, name: str | None = None, overwrite: bool = False
    ) -> str:
        """Import a definition from a standalone file; returns the name it was saved under.

        Raises:
            FileNotFoundError: ``in_path`` does not exist.
            ValueError: A malformed file, the wrong schema version, or (without
                ``overwrite``) the name already exists.
        """
        return heading_store.import_definition(
            self._dir, in_path, name=name, overwrite=overwrite
        )

    # --- stream presets (§3) ------------------------------------------------ #
    def list_stream_presets(self) -> list[dict]:
        """Saved importer setups for this experiment, oldest first (``[]`` if none)."""
        return stream_presets.list_presets(self._dir)

    def get_stream_preset(self, name: str) -> dict:
        """Raises: ValueError if no preset named ``name`` exists."""
        return stream_presets.get_preset(self._dir, name)

    def save_stream_preset(
        self,
        name: str,
        importer: str,
        config: dict | None,
        *,
        offset_ns: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """Save an importer + its config for reuse on this experiment's other recordings.

        Raises:
            ValueError: An invalid name, or the name exists and ``overwrite`` is False.
        """
        stream_presets.save_preset(
            self._dir, name, importer, config, offset_ns=offset_ns, overwrite=overwrite
        )

    def delete_stream_preset(self, name: str) -> None:
        """Raises: ValueError if no preset named ``name`` exists."""
        stream_presets.delete_preset(self._dir, name)

    def export_stream_preset(self, name: str, out_path: Path | str) -> None:
        """Write a saved preset to a standalone, shareable YAML file.

        Raises:
            ValueError: No preset named ``name`` exists.
        """
        stream_presets.export_preset(self._dir, name, out_path)

    def import_stream_preset(
        self,
        in_path: Path | str,
        *,
        name: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """Import a preset from a standalone file; returns the name it was saved under.

        Raises:
            FileNotFoundError: ``in_path`` does not exist.
            ValueError: A malformed file, the wrong schema version, or (without
                ``overwrite``) the name already exists.
        """
        return stream_presets.import_preset(
            self._dir, in_path, name=name, overwrite=overwrite
        )

    def apply_stream_preset(
        self,
        preset: str,
        sources: list[tuple[str, Path | str, int | None]],
        *,
        name: str,
        by_video: bool = False,
        progress: Callable[[int, int, str, float, str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Import one stream per recording from a saved preset (Section 3).

        The batch shape for a mapping worked out once. ``sources`` is
        ``[(recording_ref, file, offset_ns | None), ...]`` — **one row per recording, with
        its own file and its own offset**, because those are exactly what a preset cannot
        carry: each video and logger pair has its own skew, so one number applied to sixty
        recordings is one recording's answer given to all of them.

        A recording that fails is recorded and skipped, so one unreadable file never sinks
        the batch — the same contract as ``import_behavior_labels``.

        Args:
            name: The stream name to give each import (the same on every recording).
            by_video: Each ``recording_ref`` is a **video file name**, matched by the same
                rule as a metadata import's file key (base name, any case, extension
                optional) — the key a table made alongside the recordings carries (#330).
                ``False``: an id or a name.
            progress: ``progress(index, total, recording_id, fraction, status)``.

        Returns:
            The skipped recordings as ``[(recording_ref, error), ...]``.

        Raises:
            ValueError: If ``preset`` names nothing saved (nothing is imported).
        """
        self.get_stream_preset(preset)  # fail before touching any recording (Tenet 10)
        total = len(sources)
        skipped: list[tuple[str, str]] = []
        if by_video:
            names = self.recording_names()
            matches = self._video_file_matches(ref for ref, _, _ in sources)
        for index, (ref, path, offset_ns) in enumerate(sources):
            if progress is not None:
                progress(index, total, ref, 0.0, "importing")
            error = (
                _match_error("video file", ref, matches[ref], names)
                if by_video
                else None
            )
            if error is None:
                try:
                    recording = (
                        self.get_recording(matches[ref][0])
                        if by_video
                        else self.resolve_recording(ref)
                    )
                    recording.add_stream_from_preset(
                        name, path, preset, offset_ns=offset_ns
                    )
                except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
                    error = str(exc)
            if error is not None:
                skipped.append((ref, error))
            if progress is not None:
                progress(index, total, ref, 1.0, "done" if error is None else "skipped")
        return skipped

    # --- grouping variables (§11) ------------------------------------------- #
    @property
    def recording_metadata(self) -> dict[str, dict[str, str]]:
        """The cohort metadata table: ``{recording_id: {column: value}}`` (§11)."""
        return metadata_store.read_metadata(self._dir)

    @property
    def metadata_columns(self) -> list[str]:
        """User-defined grouping-variable columns (excluding ``recording_id``)."""
        return metadata_store.columns(self._dir)

    def set_recording_metadata(self, recording_id: str, values: dict[str, str]) -> None:
        """Set/merge a recording's grouping variables; new columns are added (§11)."""
        metadata_store.upsert_row(self._dir, recording_id, values)

    def add_metadata_column(self, name: str) -> None:
        """Add an empty grouping-variable column to the cohort table (§11)."""
        metadata_store.add_column(self._dir, name)

    def delete_metadata_column(self, name: str) -> None:
        """Remove a grouping-variable column and its values from every recording (§11).
        Idempotent; ``recording_id`` cannot be deleted."""
        metadata_store.delete_column(self._dir, name)

    def set_metadata_table(
        self, columns: list[str], rows: dict[str, dict[str, str]]
    ) -> None:
        """Replace the whole cohort table (§11): ``columns`` in order, ``rows`` keyed by
        recording id. Commits a fully edited table — added/deleted columns and cell edits
        together — in one write; the bulk counterpart to :meth:`set_recording_metadata`.
        """
        metadata_store.write_metadata(self._dir, columns, rows)

    def metadata_csv_key(self, path: Path | str) -> tuple[str | None, list[str]]:
        """``(key_column, header)`` for a metadata CSV: the column an import would key on
        by default — ``None`` when none is recognised — and every column it has, so a
        surface can ask which one holds the video file name (#330).

        Raises:
            FileNotFoundError: If the file is absent.
            ValueError: An empty file, or an empty or duplicate column header.
        """
        header = metadata_store.read_import_header(path)
        return metadata_store.detect_key_column(header), header

    def import_metadata_csv(
        self,
        path: Path | str,
        *,
        dry_run: bool = False,
        key_column: str | None = None,
    ) -> tuple[list[str], list[str], int]:
        """Bulk-apply recording grouping variables from a CSV (§11): merge-only,
        blank cells skipped, a row matching no recording rejected. Validates the whole
        file first — on **any** error nothing is applied (Tenet 10).

        Each row is matched to a recording by its **key column**: ``recording_id``, the
        recording's ``name``, or its **video file name** — the key a sheet prepared before
        import actually carries, since the id is minted at import and the name is right
        only once every recording has been renamed (#330). A video file name is the base
        name, any case, extension optional, and matches any of the recording's video parts.

        Args:
            key_column: The column to key on. ``None`` detects it — ``recording_id``, then
                ``name``, then a video file name column
                (``metadata_store.VIDEO_KEY_HEADERS``). A column named here that is not
                the id or the name is matched as a video file name.

        A name or file key is resolved to its id here and never *sets* anything: renaming
        is ``Recording.set_name``, which enforces uniqueness.

        Returns ``(errors, warnings, recordings_updated)``.
        """
        try:
            kind, columns, rows = metadata_store.load_import_csv(path, key_column)
        except (FileNotFoundError, ValueError) as exc:
            return [str(exc)], [], 0
        if kind != metadata_store.RECORDING_KEY:
            rows, errors = self._rekey_import_rows(kind, rows)
            if errors:
                return errors, [], 0
        errors, warnings = metadata_store.validate_import(
            columns, rows, [s.id for s in self.recordings]
        )
        if errors or dry_run:
            return errors, warnings, 0
        applied = metadata_store.apply_import(self._dir, columns, rows)
        return errors, warnings, applied

    def _rekey_import_rows(self, kind: str, rows: list) -> tuple[list, list[str]]:
        """Re-key name- or file-addressed import rows onto recording ids.

        Returns the re-keyed rows and one error per row that matched no recording, matched
        several, or matched a recording another row already did — all of them, in the
        user's own terms, rather than the first (Tenet 10).
        """
        names = self.recording_names()
        values = [value for value, _ in rows]
        if kind == metadata_store.NAME_KEY:
            ids_by_name = {name: rid for rid, name in names.items()}
            matches = {v: [ids_by_name[v]] if v in ids_by_name else [] for v in values}
            what = "recording name"
        else:
            matches = self._video_file_matches(values)
            what = "video file"
        rekeyed, errors = [], []
        first_value_by_id: dict[str, str] = {}
        for value, cells in rows:
            if not value:
                errors.append(f"a row has an empty {what}")
                continue
            error = _match_error(what, value, matches[value], names)
            if error:
                errors.append(error)
                continue
            recording_id = matches[value][0]
            if recording_id in first_value_by_id:
                errors.append(
                    f"{what}s {first_value_by_id[recording_id]!r} and {value!r} both "
                    f"match recording {names[recording_id]!r} — one row per recording"
                )
                continue
            first_value_by_id[recording_id] = value
            rekeyed.append((recording_id, cells))
        return rekeyed, errors

    def _video_file_matches(self, values: Iterable[str]) -> dict[str, list[str]]:
        """``{value: [recording_id, ...]}`` — the recordings whose video each value names.

        The one matching rule for a video-file key, wherever a table carries one (#330):
        the **base name** of the value (a pasted path is fine), **any case**, **extension
        optional** — ``M27.MP4``, ``m27`` and ``D:\\raw\\m27.mp4`` all name ``m27.mp4`` —
        matched against every video part of every recording. An empty list is no match;
        more than one is an ambiguity the caller refuses rather than resolves.
        """
        by_file: dict[str, set[str]] = {}
        by_stem: dict[str, set[str]] = {}
        with read_cache.pinned():
            for recording in self.recordings:
                for source in recording.video_sources:
                    if not source:
                        continue
                    file_name = _video_file_key(source)
                    by_file.setdefault(file_name, set()).add(recording.id)
                    stem = os.path.splitext(file_name)[0]
                    by_stem.setdefault(stem, set()).add(recording.id)
        out = {}
        for value in values:
            key = _video_file_key(value)
            out[value] = sorted(by_file.get(key) or by_stem.get(key) or ())
        return out

    def export_metadata_csv(self, path: Path | str) -> None:
        """Write the current grouping variables to ``path`` as a fill-in CSV template
        (§11) — ``recording_id``, ``name``, ``file``, then every column, one row per
        recording.

        ``name`` and ``file`` (the first video part's file name, as the table shows it)
        ride along so the sheet is legible — an id alone tells a person filling it in
        nothing — and so the filled sheet can be keyed back on any of the three. Neither
        is a stored value: editing one here renames nothing.
        """
        with read_cache.pinned():
            files = {
                recording.id: next(
                    (_base_name(s) for s in recording.video_sources if s), ""
                )
                for recording in self.recordings
            }
        metadata_store.export_metadata(
            self._dir, path, names=self.recording_names(), files=files
        )

    #: How an import's own status vocabulary maps onto the shared run one. An import is
    #: ``queued`` before it starts and ``importing`` while it runs — the two states every other
    #: family calls ``pending`` and ``running``, in a dialect only this one speaks. Translating
    #: here rather than teaching every reader the dialect is what lets one listing, one quit
    #: guard and one "is it still going" predicate cover a family that keeps its status on a
    #: recording instead of in a run record. ``ready`` reads as ``complete``: on a local disk an
    #: import is usually over within a second of being asked for, so a listing that dropped
    #: finished ones showed nothing to a user who had just imported 533 MB of video.
    _IMPORT_RUN_STATUS = {
        "queued": JobStatus.PENDING.value,
        "importing": JobStatus.RUNNING.value,
        "ready": JobStatus.COMPLETE.value,
        "failed": JobStatus.FAILED.value,
        "cancelled": JobStatus.CANCELLED.value,
    }

    @property
    def import_runs(self) -> list[dict]:
        """Every recording that was imported, shaped like a run record (§3).

        An import is the one long operation this app runs that is *not* a run: it writes no run
        record, and its state lives on the recording as ``import_status``. This is the adapter
        that lets it be listed, counted and cancelled beside the families that do — the shape
        ``api.active.RUN_KINDS`` reads, carrying the recording id a surface needs to open it and
        the name a surface should show instead of that id (§11).

        ``created_at`` is the recording's own, which is the right answer rather than a
        convenient one: ``create_shell`` mints the recording synchronously when the import is
        submitted, so it is the moment the import was asked for.

        Read under a pin because every :class:`Recording` property is its own ``recording.yaml``
        read and this wants three of them per recording — and the quit guard asks this question
        **on the UI thread**, where the difference is one round trip per recording instead of
        three (`storage/read_cache.py`).
        """
        runs: list[dict] = []
        with read_cache.pinned():
            for recording in self.recordings:
                record = recording_store.read_recording(recording.dir)
                # The key, not `Recording.import_status`, which defaults a missing one to
                # "ready". Only an import writes it: a recording made before the field existed,
                # or given video by `recording add-video`, never went through one.
                status = self._IMPORT_RUN_STATUS.get(record.get("import_status"))
                if status is None:
                    continue
                # A shell created with metadata only is `ready` the moment it exists and carried
                # nothing in — calling it a finished import would list every such recording.
                imported_something = (record.get("video") or {}).get(
                    "files"
                ) or record.get("streams")
                if status == JobStatus.COMPLETE.value and not imported_something:
                    continue
                runs.append(
                    {
                        "id": recording.id,
                        "name": recording.name,
                        "status": status,
                        "created_at": record.get("created_at", ""),
                    }
                )
        return runs

    def cancel_import(self, recording_id: str) -> None:
        """Mark a recording's unfinished import cancelled. A no-op once it is terminal.

        This records a decision; it does not reach into a worker. An import runs on a thread
        owned by whichever app has this experiment open (`gui/import_queue.py`), so there is no
        cross-process handle to pull — and on the path that matters, quitting, the worker dies
        with the process regardless. What the caller gets is the difference between a record
        that says ``cancelled`` and one left at ``queued`` for the next open to call ``failed``:
        the same outcome, one of them blaming the user for it.

        Raises:
            FileNotFoundError: If the recording doesn't exist.
        """
        recording = self.get_recording(recording_id)
        if recording.import_status not in ("queued", "importing"):
            return
        recording_store.set_import_status(recording.dir, "cancelled")

    def reconcile_imports(self) -> None:
        """Flip orphaned import statuses (``queued`` / ``importing`` left by a
        crash or close, with no active queue) to ``failed`` so they're visible +
        deletable, not stuck. Called on experiment open before the import queue starts
        (the in-memory queue can't resume — §3)."""
        for recording in self.recordings:
            if recording.import_status in ("queued", "importing"):
                recording_store.set_import_status(recording.dir, "failed")

    def reconcile_runs(self) -> None:
        """Fold each still-active processing run's real on-disk status back into its
        record on experiment open (§15). A run's coordinator + workers are detached and
        may outlive the app, but a crashed/SIGKILLed run can read ``running`` forever.
        Evidence-based (see :meth:`_reconcile_run`). Mirrors :meth:`reconcile_imports`.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        for record in runs_store.list_runs(self._dir):
            self._reconcile_run(record)

    def _reconcile_run(self, record: dict) -> bool:
        """Reconcile one run record against on-disk evidence; returns True if changed.
        The **coordinator pid** is the liveness authority: alive → leave running; gone
        → adopt each recording's terminal ``job.json`` status, keep an individually-alive
        recording worker running, fail recordings the coordinator never launched, and roll
        up. This is the SIGKILL backstop that guarantees a terminal status."""
        if record.get("status") not in ACTIVE_STATUSES:
            return False
        module = PROCESSING_MODULES.get(record.get("module_id"))
        if module is None:
            return False
        run_id = record["id"]
        pid = runs_store.read_coordinator_pid(self._dir, run_id)
        if pid is None:
            return False  # pid not recorded yet — just launched, leave running
        if coordinator_alive(runs_store.run_dir(self._dir, run_id), pid):
            return False  # the coordinator is driving it
        per_recording = dict(record.get("per_recording") or {})
        changed = False
        for sid, entry in per_recording.items():
            stage_run_id = entry.get("run_id")
            if stage_run_id is None or entry.get("status") not in ACTIVE_STATUSES:
                continue
            sess_run_dir = experiment_store.recording_path(
                self._dir, sid
            ) / module.output_path(stage_run_id)
            try:
                status = rundir.read_status(sess_run_dir).value
            except (FileNotFoundError, KeyError, ValueError):
                status = None
            if status is None or status in ACTIVE_STATUSES:
                if status is not None and worker_alive(sess_run_dir):
                    continue  # a detached recording worker still finishing — leave it
                status = JobStatus.FAILED.value
                entry["error"] = "interrupted: coordinator gone (reconciled on open)"
            entry["status"] = status
            changed = True
            extra: dict = {}
            with contextlib.suppress(Exception):
                job = rundir.read_job(sess_run_dir)
                compute = getattr(job, "compute", None)
                extra = run_metrics.completion_fields(
                    sess_run_dir,
                    compute.to_dict() if compute is not None else None,
                )
            with contextlib.suppress(FileNotFoundError, ValueError, KeyError):
                self.get_recording(sid)._set_run_status(
                    module.runs_key, stage_run_id, status, **extra
                )
        for sid in record.get("recording_ids") or []:
            if sid not in per_recording:  # coordinator died before launching it
                per_recording[sid] = {
                    "run_id": None,
                    "status": JobStatus.FAILED.value,
                    "error": "interrupted: coordinator gone before launch",
                }
                changed = True
        record["per_recording"] = per_recording
        statuses = [e.get("status") for e in per_recording.values()]
        if any(s in ACTIVE_STATUSES for s in statuses):
            new_status = "running"
        elif statuses and all(s == JobStatus.COMPLETE.value for s in statuses):
            new_status = "complete"
        else:
            new_status = "failed"
        if new_status != record.get("status"):
            record["status"] = new_status
            changed = True
        if changed:
            runs_store.write_run(self._dir, run_id, record)
        return changed

    def delete_recording(self, recording_id: str) -> None:
        """Delete a recording and everything it owns — its imported video, runs
        (detection / tracking / embedding / downsampling), arenas, and masks.

        A processing run merely *indexes* per-recording runs (its outputs live in
        the recordings, Tenet 5), so the recording is pruned from every processing
        run's ``per_recording`` map and an emptied run record is removed — not
        blocked. Nothing outside a recording is computed from a whole recording
        today; a future cross-recording analysis would register as a dependent.

        Raises:
            FileNotFoundError: If no such recording exists.
        """
        sdir = experiment_store.recording_path(self._dir, recording_id)
        if not sdir.is_dir():
            raise FileNotFoundError(f"recording {recording_id!r} not found")
        shutil.rmtree(sdir)
        metadata_store.remove_row(
            self._dir, recording_id
        )  # prune its grouping vars (§11)
        for record in runs_store.list_runs(self._dir):
            per = record.get("per_recording", {})
            if recording_id not in per:
                continue
            del per[recording_id]
            if per:
                record["per_recording"] = per
                runs_store.write_run(self._dir, record["id"], record)
            else:
                runs_store.delete_run(self._dir, record["id"])

    def delete(self) -> None:
        """Delete the whole experiment directory — every recording, run, model, and
        dataset it owns. Nothing outside a experiment depends on it, so this is
        never blocked (the cohesive delete strategy, gui_design §1.1). The
        ``Experiment`` instance is dead afterwards.

        Raises:
            FileNotFoundError: If the experiment directory no longer exists.
        """
        if not self._dir.is_dir():
            raise FileNotFoundError(f"experiment not found: {self._dir}")
        shutil.rmtree(self._dir)

    # --- Processing runs over a recording selection (§15) --------------------- #
    def run_processing(
        self,
        module_id: str,
        recording_ids: list[str],
        config: dict,
        *,
        ignore_free_space: bool = False,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> "AttachedRun":
        """Run a processing module over a selection of recordings.

        Validates every selected recording up front (Tenet 10), checks the run's estimated
        output fits where the experiment is (#743), reserves the run record, then spawns a
        **detached coordinator** (``executor/coordinator.py``) that sequences the
        per-recording ``Recording.run`` calls and publishes aggregate progress. Returns an
        :class:`AttachedRun` polling that progress — so the run survives the app closing and
        any process can reattach (§15). Per-recording outputs land in each recording
        (Tenet 5). The estimate and the free space it was checked against are kept on the
        run record (``space``).

        Args:
            ignore_free_space: Launch even when the estimate does not fit.
            progress: ``progress(stage, fraction, detail)``, called while the up-front work
                runs — every recording is read twice (``Validating recordings``, then
                ``Estimating disk space``), which is minutes over a large experiment or a
                network share. The run itself is reported by the returned handle.

        ``progress`` takes the standard ``(stage, fraction, detail)`` callback and names
        each recording as the up-front checks reach it: over a large selection they read
        every recording's outputs, which is minutes, not a moment.

        Raises:
            ValueError: For an unknown ``module_id``, an empty selection, or any
                recording that fails the module's pre-run validation.
            InsufficientSpaceError: The estimated output does not fit (a ``ValueError``,
                so every surface that reports a refused launch reports this one too).
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        module = PROCESSING_MODULES.get(module_id)
        if module is None:
            raise ValueError(
                f"unknown module_id {module_id!r}; "
                f"available: {sorted(PROCESSING_MODULES)}"
            )
        recording_ids = list(recording_ids)
        if not recording_ids:
            raise ValueError("recording_ids must be non-empty")
        config, validated = self._validate_processing(
            module, recording_ids, config, progress
        )
        space = self._check_space(module, validated, ignore_free_space, progress)
        config = _with_model_defaults(module, config)
        # Allocate and reserve under one lock: `next_run_id` is max+1 over a directory scan,
        # so two launches racing between the scan and the write both take the same id and the
        # loser's atomic replace fails on a tmp file the winner already consumed.
        # Reserve before spawning so the run is discoverable + a coordinator that starts and
        # then dies reconciles to failed (not stuck running).
        with runs_store.runs_lock(self._dir):
            run_id = runs_store.next_run_id(self._dir)
            runs_store.write_run(
                self._dir,
                run_id,
                {
                    "id": run_id,
                    "module_id": module_id,
                    "recording_ids": recording_ids,
                    "config": _jsonable(config),
                    "per_recording": {},
                    "status": "running",
                    "created_at": utc_timestamp(),
                    "compute": _record_compute(config),
                    "space": _space_record(space),
                },
            )
        try:
            pid = spawn_coordinator(
                runs_store.run_dir(self._dir, run_id), tuple(_PIPELINE_RUNNERS)
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 — re-raised; must not leave a stuck record
            # The coordinator never launched (e.g. Popen OSError). Without a recorded
            # pid, reconcile can't tell this from a just-launched run, so mark the
            # reserved record failed here rather than leave it stuck "running".
            self._fail_run_record(run_id, f"coordinator failed to launch: {exc}")
            raise
        runs_store.set_coordinator_pid(self._dir, run_id, pid)
        return self.attach_run(run_id)

    def _validate_processing(
        self,
        module,
        recording_ids: list[str],
        config: dict,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> tuple[dict, dict[str, dict]]:
        """``(config to record, {recording_id: its validated config})`` — every selected
        recording validated up front (Tenet 10), once.

        A batch module validates the whole subset once, and its cross-recording step needs
        the normalized config in the record (the coordinator does not re-validate). Otherwise,
        values that must agree across the selection (embedding's shared canvas) are resolved
        once and enrich the stored config — a per-recording validate cannot see the other
        recordings (§6.5) — and each recording's batch spec is resolved before its validate;
        the record keeps the spec, so the coordinator re-resolves.

        A class the user named by its experiment name is resolved to its stored name first
        (§1 *Class names*), so what is validated and recorded is the stored name.
        """
        config = none_means_default(
            module.config_model, self.resolve_class_refs(config)
        )
        if module.batch:
            config = module.validate_batch(self._dir, recording_ids, dict(config))
            return config, {sid: config for sid in recording_ids}
        names = self.recording_names() if progress is not None else {}
        shared = _RecordingSteps(
            names, progress, progress_mod.RESOLVING_STAGE, len(recording_ids)
        )
        config = module.resolve_shared_config(
            self._dir, recording_ids, dict(config), on_recording=shared
        )
        shared.finish()
        validated = {}
        for sid in progress_mod.reporting_each(
            recording_ids,
            progress_mod.VALIDATING_STAGE,
            progress,
            label=lambda sid: names.get(sid, sid),
        ):
            sdir = experiment_store.recording_path(self._dir, sid)
            sdata = recording_store.read_recording(sdir)
            resolved = module.resolve_upstream_spec(sdir, sdata, dict(config))
            validated[sid] = module.validate(sdir, sdata, resolved)
            # `validate` reads recording.yaml, where a run the OS lost after a shutdown still
            # says complete; this names it here instead of in a worker a second later.
            module.check_upstream_intact(sdir, sdata, resolved)
        return config, validated

    def _check_space(
        self,
        module,
        validated: dict[str, dict],
        ignore_free_space: bool,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> disk_space.SpaceEstimate:
        """The run's estimated output against the free space where the experiment is
        (#743); refused unless ``ignore_free_space``. Each recording is estimated from its own
        validated config, so its own frame count, tracks and stride are what is counted.
        """
        per_recording = {}
        names = self.recording_names() if progress is not None else {}
        for sid in progress_mod.reporting_each(
            validated,
            progress_mod.ESTIMATING_STAGE,
            progress,
            label=lambda sid: names.get(sid, sid),
        ):
            sdir = experiment_store.recording_path(self._dir, sid)
            per_recording[sid] = int(
                module.estimate_output_bytes(
                    sdir, recording_store.read_recording(sdir), validated[sid]
                )
            )
        free, where = disk_space.free_space(self._dir)
        estimate = disk_space.SpaceEstimate(
            bytes=sum(per_recording.values()),
            free_bytes=free,
            where=where,
            per_recording=per_recording,
        )
        if not estimate.fits and not ignore_free_space:
            raise disk_space.InsufficientSpaceError(estimate)
        return estimate

    def retry_run(
        self,
        run_id: str,
        *,
        ignore_free_space: bool = False,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> "AttachedRun":
        """Re-run a terminal processing run's incomplete recordings, into the same record.

        Gap-fills rather than minting a new run: each recording already owns its own stage
        run id in ``per_recording``, so the record is a map with holes in it. The re-spawned
        coordinator seeds those results and drives only what isn't ``complete``, and the
        run rolls up to ``complete`` once every recording is (§15). Retried recordings get a
        *fresh* stage run id from ``Recording.run``, so nothing appends into a half-written
        zarr — a recording is re-run whole, not resumed mid-way (§7 resume is not built).

        Covers ``failed`` and ``cancelled`` runs alike: the set is every recording that
        isn't complete, including ones the coordinator never launched.

        ``progress(stage, fraction, detail)`` reports the up-front validation and space
        estimate, exactly as :meth:`run_processing` does.

        Returns:
            An :class:`AttachedRun` streaming the re-spawned coordinator.

        Raises:
            FileNotFoundError: If the run record does not exist.
            ValueError: If the run is still running, its module is unknown, every recording
                is already complete, or any incomplete recording no longer validates.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        record = runs_store.read_run(self._dir, run_id)
        # A SIGKILLed coordinator leaves the record "running" forever and no CLI path
        # calls reconcile, so gating on a terminal status would reject a run that is
        # genuinely retryable. Reconcile mutates + persists in place.
        self._reconcile_run(record)
        if record.get("status") in ACTIVE_STATUSES:
            raise ValueError(
                f"processing run {run_id!r} is still running — cancel it before retrying"
            )
        module_id = record.get("module_id")
        module = PROCESSING_MODULES.get(module_id)
        if module is None:
            raise ValueError(
                f"unknown module_id {module_id!r} for run {run_id!r}; "
                f"available: {sorted(PROCESSING_MODULES)}"
            )
        per_recording = dict(record.get("per_recording") or {})
        pending = [
            sid
            for sid in record.get("recording_ids") or []
            if per_recording.get(sid, {}).get("status") != JobStatus.COMPLETE.value
        ]
        if not pending:
            raise ValueError(
                f"processing run {run_id!r}: every recording is complete — nothing to retry"
            )
        # The record stores _jsonable(config); restore the typed values before validating,
        # whose checks read compute.backend / prompt.targets as objects (cf. __init__).
        config = _typed_config(record.get("config") or {})
        errors, validated = self._validate_retry_recordings(
            module, pending, config, progress
        )
        if errors:
            detail = "; ".join(f"{sid}: {err}" for sid, err in errors.items())
            raise ValueError(f"cannot retry run {run_id!r} — {detail}")
        # A retry re-runs each pending recording whole, so it writes as much as a launch of
        # those recordings would.
        record["space"] = _space_record(
            self._check_space(module, validated, ignore_free_space, progress)
        )
        for sid in pending:
            per_recording[sid] = {"run_id": None, "status": JobStatus.PENDING.value}
        record["per_recording"] = per_recording
        record["status"] = "running"
        record.pop("error", None)
        runs_store.write_run(self._dir, run_id, record)
        # A cancelled run's sentinel outlives its coordinator; leave it and the fresh one
        # stops on its first recording and re-records cancelled (cf. resume_training).
        runs_store.clear_cancel(self._dir, run_id)
        # Seed the aggregate bar so a client reattaching before the coordinator's first
        # emit sees the real standing, not the finished run's stale recordings_done.
        runs_store.write_progress(
            self._dir,
            run_id,
            processing_progress_to_dict(
                ProcessingProgress(
                    len(record["recording_ids"]) - len(pending),
                    len(record["recording_ids"]),
                    pending[0],
                    None,
                    0.0,
                    len(record["recording_ids"]) - len(pending),
                )
            ),
        )
        try:
            pid = spawn_coordinator(
                runs_store.run_dir(self._dir, run_id), tuple(_PIPELINE_RUNNERS)
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 — re-raised; must not leave it "running"
            self._fail_run_record(run_id, f"coordinator failed to launch: {exc}")
            raise
        runs_store.set_coordinator_pid(self._dir, run_id, pid)
        return self.attach_run(run_id)

    def _validate_retry_recordings(
        self,
        module,
        recording_ids: list[str],
        config: dict,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> tuple[dict[str, str], dict[str, dict]]:
        """``({recording_id: reason}, {recording_id: validated config})`` — every recording
        that no longer validates, and the validated config of every one that does (what the
        space check estimates from, #743).

        Collects instead of raising on the first: a retry names *every* broken recording so
        the user fixes them in one pass. Broad except is the aggregation boundary — every
        reason is re-raised by the caller, nothing is swallowed.
        """
        errors: dict[str, str] = {}
        validated: dict[str, dict] = {}
        config = none_means_default(module.config_model, config)
        if module.batch:
            # A batch run has no per-recording validate; re-validate the whole subset once.
            try:
                checked = module.validate_batch(self._dir, recording_ids, dict(config))
            except Exception as exc:  # noqa: BLE001 — aggregation boundary
                errors[recording_ids[0]] = f"{type(exc).__name__}: {exc}"
                return errors, validated
            return errors, {sid: checked for sid in recording_ids}
        names = self.recording_names() if progress is not None else {}
        for sid in progress_mod.reporting_each(
            recording_ids,
            progress_mod.VALIDATING_STAGE,
            progress,
            label=lambda sid: names.get(sid, sid),
        ):
            sdir = experiment_store.recording_path(self._dir, sid)
            try:
                sdata = recording_store.read_recording(sdir)
                resolved = module.resolve_upstream_spec(sdir, sdata, dict(config))
                validated[sid] = module.validate(sdir, sdata, resolved)
                module.check_upstream_intact(sdir, sdata, resolved)
            except (
                Exception
            ) as exc:  # noqa: BLE001 — aggregation boundary; see docstring
                errors[sid] = f"{type(exc).__name__}: {exc}"
        return errors, validated

    def _fail_run_record(self, run_id: str, error: str) -> None:
        record = runs_store.read_run(self._dir, run_id)
        record["status"] = "failed"
        record.setdefault("error", error)
        runs_store.write_run(self._dir, run_id, record)

    def attach_run(self, run_id: str) -> "AttachedRun":
        """Reattach to a processing run by id — a live one streams its coordinator's
        progress, a terminal one returns at once. Works after an app restart because
        the coordinator + workers are detached and progress is on disk (§15)."""
        return AttachedRun(self, run_id)

    def attach_active_run(self, run_id: str) -> "AttachedRun | None":
        """An :class:`AttachedRun` when the run is still active (for a live reattach
        after reopen), else ``None`` — so the GUI shows a progress bar for a run that
        is still going and a static record for a finished one."""
        with contextlib.suppress(FileNotFoundError):
            if runs_store.read_run(self._dir, run_id).get("status") in ACTIVE_STATUSES:
                return self.attach_run(run_id)
        return None

    # --- Distill runs (label → train, one durable run — §12) ------------------ #
    def distill(
        self,
        name: str,
        classes: list[LabelingClass],
        n_frames: int,
        *,
        source_recording_ids: list[str] | None = None,
        sampling: str = "stratified",
        logit_threshold: float = 0.5,
        min_contour_area: int = 100,
        val_split: float = 0.15,
        background_share: float = LabelingConfig.model_fields[
            "background_share"
        ].default,
        label_model_id: str = "sam3",
        label_model_variant: str | None = None,
        label_compute: ComputeTarget | None = None,
        base_model: str = "yolo11s-seg.pt",
        trainer_id: str = "ultralytics_yolo",
        hyperparams: dict | None = None,
        train_compute: ComputeTarget | None = None,
        stage_videos: bool = False,
    ) -> "AttachedDistillRun":
        """Distill a fast detector in one durable run (§12): label a frame sample with a
        prompt-capable model, then train a closed-set YOLO on those labels.

        ``stage_videos=True`` also copies every source recording's video to the remote target
        in a detached background process while labeling and training run, so the detection run
        that follows (which reads them all) does not have to upload them first. The target is
        the training compute's (where the trained model, and so detection, will be), else the
        labeling compute's. Opt-in because it can be 100+ GB of cluster storage. See
        :meth:`video_staging`.

        Validates the union of the labeling and training pre-flight up front (Tenet 10),
        reserves the run record, then spawns a **detached distill coordinator** that
        sequences the two stages and publishes folded progress. Returns an
        :class:`AttachedDistillRun` — the run survives the app closing and any process can
        reattach. The trained model lands in ``models/<id>/`` (usable in this experiment at
        once); promote it with :meth:`promote_model` for cross-experiment reuse.

        Raises:
            ValueError: no/duplicate classes, a non-prompt-capable label model, a recording
                without video, ``n_frames`` above the frames available, the label model not
                ready, an unknown/unready trainer, an unsupported train backend, or invalid
                hyperparameters.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        recording_ids = list(source_recording_ids or [s.id for s in self.recordings])
        if not classes:
            raise ValueError("labeling needs at least one class (a name + text prompt)")
        prompts = [c.prompt for c in classes]
        if len(set(prompts)) != len(prompts):
            raise ValueError(
                "two classes share a text prompt; prompts must be distinct"
            )
        capable = api_models.prompt_capable_models()
        if label_model_id not in capable:
            raise ValueError(
                f"model {label_model_id!r} cannot label — labeling needs a prompt-capable "
                f"model; choose one of {capable}"
            )
        lab_cfg = LabelingConfig.model_validate(
            {
                "model_id": label_model_id,
                "model_variant": label_model_variant,
                "n_frames": n_frames,
                "sampling": sampling,
                "logit_threshold": logit_threshold,
                "min_contour_area": min_contour_area,
                "val_split": val_split,
                "background_share": background_share,
            }
        )
        trainer_cls = TRAINERS.get(trainer_id)
        if trainer_cls is None:
            raise ValueError(
                f"unknown trainer_id {trainer_id!r}; available: {sorted(TRAINERS)}"
            )
        trainer = trainer_cls()
        train_compute = train_compute or default_compute(trainer.supported_backends)
        # A fact about *this* machine's own backend (e.g. is ultralytics importable
        # here) — a remote job trains entirely on the cluster, so gating submission on
        # it would demand this machine have the training framework for a job that
        # never trains here. Mirrors label_dataset's / detection's identical guard.
        if train_compute.backend != "remote" and not trainer.is_ready():
            raise ValueError(
                f"trainer {trainer_id!r} is not ready (install its backend)"
            )
        if effective_backend(train_compute) not in trainer.supported_backends:
            raise ValueError(
                f"trainer {trainer_id!r} does not support the "
                f"{train_compute.backend!r} backend; supports {trainer.supported_backends}"
            )
        _require_remote_target_registered(train_compute)
        if train_compute.backend == "cuda" and train_compute.device_indices:
            check_cuda_indices(train_compute.device_indices)
        hyperparams = trainer.validate_hyperparams(hyperparams or {})
        # Enforce the run-level composition + a non-empty name before the recording sweep.
        DistillConfig.model_validate(
            {
                "name": name,
                "base_model": base_model,
                "trainer_id": trainer_id,
                "stage_videos": stage_videos,
                "labeling": lab_cfg.model_dump(),
                "training": hyperparams,
            }
        )
        if not recording_ids:
            raise ValueError("labeling needs at least one source recording")
        available = 0
        for sid in recording_ids:
            recording = self.get_recording(sid)  # raises if the recording is unknown
            data = recording_store.read_recording(recording.dir)
            files = data.get("video", {}).get("files", [])
            if not files:
                raise ValueError(f"recording {sid!r} has no video to sample")
            available += sum(
                len(recording_store.load_active_timestamps(recording.dir, data, i))
                for i in range(len(files))
            )
        if lab_cfg.n_frames > available:
            raise ValueError(
                f"n_frames={lab_cfg.n_frames} exceeds the {available} frames available "
                "across the selected recordings"
            )
        label_compute = label_compute or default_compute(
            api_models.supported_backends(lab_cfg.model_id)
        )
        # A fact about *this* machine's disk/network access — a remote labeling job
        # never loads the model here; the shepherd's own readiness probe checks it on
        # the cluster before the sbatch submission. Mirrors label_dataset's identical
        # guard (§12).
        if label_compute.backend != "remote":
            access = self.check_model_access(lab_cfg.model_id)
            if not access.ready:
                raise ValueError(f"{lab_cfg.model_id} is not ready — {access.message}")
        if effective_backend(label_compute) not in api_models.supported_backends(
            lab_cfg.model_id
        ):
            raise ValueError(
                f"model {lab_cfg.model_id!r} does not support the "
                f"{label_compute.backend!r} backend; it requires one of "
                f"{api_models.supported_backends(lab_cfg.model_id)}"
            )
        _require_remote_target_registered(label_compute)
        stage_target = _video_staging_target(label_compute, train_compute)
        if stage_videos and stage_target is None:
            raise ValueError(
                "stage_videos uploads the videos to a remote cluster, but neither the labeling "
                "nor the training compute is remote; pick a remote target or leave it off"
            )
        run_id = distill_store.next_distill_id(self._dir)
        config = {
            "name": name,
            "base_model": base_model,
            "trainer_id": trainer_id,
            "classes": [c.to_dict() for c in classes],
            "source_recording_ids": recording_ids,
            "labeling": lab_cfg.model_dump(),
            "training": hyperparams,
            "label_compute": label_compute.to_dict(),
            "train_compute": train_compute.to_dict(),
            "stage_videos": stage_videos,
            "stage_video_target": stage_target if stage_videos else None,
        }
        # Reserve before spawning so the run is discoverable and a coordinator that dies on
        # launch reconciles to failed (not stuck running).
        distill_store.write_run(
            self._dir,
            run_id,
            {
                "id": run_id,
                "name": name,
                "status": "running",
                "created_at": utc_timestamp(),
                "config": config,
                "stages": {
                    "label": {"status": "pending", "dataset_id": None},
                    "train": {"status": "pending", "model_id": None},
                },
                "resume": False,
            },
        )
        return self._spawn_distill_coordinator(run_id)

    def video_staging(self) -> list[dict]:
        """The background video uploads this experiment has started, one per remote target.

        Each is a dict with ``target``, ``status`` (``running`` / ``complete`` / ``failed`` /
        ``cancelled``; a ``running`` one whose process is gone reads ``failed``), ``files_done`` /
        ``files_total``, ``bytes_done`` / ``bytes_total``, ``eta_s`` and ``message``. Started by
        ``distill(stage_videos=True)``.
        """
        states = []
        for state in video_staging.all_states(self._dir):
            if state["status"] == "running" and not video_staging.is_active(
                self._dir, state
            ):
                state = {
                    **state,
                    "status": "failed",
                    "error": "the upload process is no longer running",
                    "eta_s": None,
                }
            states.append(state)
        return states

    def start_video_staging(
        self, target: str, recording_ids: list[str] | None = None
    ) -> dict:
        """Start, or restart, the background video upload to ``target``.

        With no ``recording_ids`` it takes up where the last upload to this target left off (the
        same recordings), else every recording in the experiment. Files already on the cluster are
        skipped and a half-sent one resumes, so restarting a stopped or failed upload costs only
        what was missing. Does nothing new when one is already running: that one is returned.

        Returns:
            The upload's recorded state (see :meth:`video_staging`).

        Raises:
            ValueError: ``target`` is not a registered remote target.
            FileNotFoundError: a recording in ``recording_ids`` does not exist.
        """
        try:
            remote_targets.get_remote_target(target)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if not recording_ids:
            previous = video_staging.read_state(self._dir, target) or {}
            recording_ids = previous.get("recording_ids") or [
                s.id for s in self.recordings
            ]
        for recording_id in recording_ids:
            self.get_recording(recording_id)  # raises if the recording is unknown
        return video_staging.start(self._dir, target, list(recording_ids))

    def stop_video_staging(self, target: str | None = None) -> list[str]:
        """Stop the background video upload to ``target`` (every one when ``None``).

        What already reached the cluster stays there, so a later run reuses it. Returns the
        targets whose upload was running and is now stopped.
        """
        names = (
            [target]
            if target is not None
            else [state["target"] for state in video_staging.all_states(self._dir)]
        )
        return [name for name in names if video_staging.stop(self._dir, name)]

    # --- keypoint setup (§26, #717) ------------------------------------------------ #
    def _spawn_distill_coordinator(self, run_id: str) -> "AttachedDistillRun":
        """Launch the detached coordinator for a reserved distill record and attach to it."""
        try:
            pid = spawn_coordinator(
                distill_store.run_dir(self._dir, run_id),
                tuple(_PIPELINE_RUNNERS),
                entrypoint="detail.executor.distill_coordinator",
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 — re-raised; must not leave it "running"
            self._fail_distill_record(run_id, f"coordinator failed to launch: {exc}")
            raise
        distill_store.set_coordinator_pid(self._dir, run_id, pid)
        return self.attach_distill_run(run_id)

    def resume_distill(self, run_id: str) -> "AttachedDistillRun":
        """Resume a terminal (failed / cancelled) distill run into the same record: reuse a
        complete labeled dataset, resume a partial model, and re-spawn the coordinator.

        Raises:
            FileNotFoundError: If the run record does not exist.
            ValueError: If the run is still running or already complete.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        record = distill_store.read_run(self._dir, run_id)
        # A SIGKILLed coordinator reads "running" forever; reconcile mutates + persists.
        self._reconcile_distill_run(record)
        if record.get("status") in ACTIVE_STATUSES:
            raise ValueError(
                f"distill run {run_id!r} is still running — cancel it before resuming"
            )
        if record.get("status") == AWAITING_REFERENCES:
            raise ValueError(
                f"distill run {run_id!r} is waiting for a reference set, not stopped — "
                "click one on its teacher tracks, then continue it with "
                "continue_keypoint_setup"
            )
        stages = record.get("stages") or {}
        last = distill_final_stage(record)
        if (stages.get(last) or {}).get("status") == JobStatus.COMPLETE.value:
            raise ValueError(
                f"distill run {run_id!r} is already complete — nothing to resume"
            )
        record["status"] = "running"
        record["resume"] = True
        record.pop("error", None)
        distill_store.write_run(self._dir, run_id, record)
        # A cancelled run's sentinel outlives its coordinator; clear it so the fresh one
        # doesn't stop on its first stage (cf. retry_run / resume_training).
        distill_store.clear_cancel(self._dir, run_id)
        return self._spawn_distill_coordinator(run_id)

    def cancel_distill(self, run_id: str) -> None:
        """Stop an active distill run: kill the coordinator, cancel the live stage worker,
        and record the run's real status (``complete`` if training actually finished,
        else ``cancelled``). No-op once terminal. Raises FileNotFoundError if absent."""
        record = distill_store.read_run(self._dir, run_id)
        if record.get("status") == AWAITING_REFERENCES:
            record["status"] = JobStatus.CANCELLED.value
            distill_store.write_run(self._dir, run_id, record)
            return
        if record.get("status") not in ACTIVE_STATUSES:
            return
        distill_store.request_cancel(self._dir, run_id)
        kill_coordinator(
            distill_store.run_dir(self._dir, run_id),
            distill_store.read_coordinator_pid(self._dir, run_id),
        )
        stages = dict(record.get("stages") or {})
        for name in distill_stage_names(record):
            if name in _TEACHER_STAGES:
                # A setup's teacher stages are processing runs, cancelled by the loop
                # below. Marked cancelled here first, that loop never saw them active, so
                # it never stopped their runs: the record said cancelled, the job ran on.
                continue
            stage = dict(stages.get(name) or {})
            if stage.get("status") in ACTIVE_STATUSES:
                _, run_dir = self._distill_stage_status(name, stage)
                if run_dir is not None:
                    cancel_run_dir(run_dir)
                    try:
                        stage["status"] = rundir.read_status(run_dir).value
                    except (FileNotFoundError, KeyError, ValueError):
                        stage["status"] = "cancelled"
                else:
                    stage["status"] = "cancelled"
                stages[name] = stage
        for name in _TEACHER_STAGES:
            stage = dict(stages.get(name) or {})
            if stage.get("status") in ACTIVE_STATUSES:
                # One processing run, or one per class (a setup's per-class tracking).
                for processing_run_id in [
                    stage.get("processing_run_id"),
                    *(stage.get("processing_run_ids") or {}).values(),
                ]:
                    if processing_run_id:
                        with contextlib.suppress(FileNotFoundError, ValueError):
                            self.cancel_run(processing_run_id)
                stage["status"] = JobStatus.CANCELLED.value
                stages[name] = stage
        record["stages"] = stages
        final_status = (stages.get(distill_final_stage(record)) or {}).get("status")
        record["status"] = (
            "complete" if final_status == JobStatus.COMPLETE.value else "cancelled"
        )
        distill_store.write_run(self._dir, run_id, record)

    def delete_distill(self, run_id: str) -> None:
        """Delete a distill run's index record only (``distill/<id>/``). The produced
        dataset, trained model, and any promoted global model are independent artifacts
        with their own lifecycles and are left intact — delete them explicitly.

        Raises:
            FileNotFoundError: If the run record does not exist.
        """
        distill_store.read_run(self._dir, run_id)  # FileNotFoundError if absent
        distill_store.delete_run(self._dir, run_id)

    def attach_distill_run(self, run_id: str) -> "AttachedDistillRun":
        """Reattach to a distill run by id — a live one streams its coordinator's progress,
        a terminal one returns at once (§12)."""
        return AttachedDistillRun(self, run_id)

    def attach_active_distill_run(self, run_id: str) -> "AttachedDistillRun | None":
        """An :class:`AttachedDistillRun` when the run is still active, else ``None`` — so
        the GUI shows a live bar for a running distill and a static record for a done one.
        """
        with contextlib.suppress(FileNotFoundError):
            if (
                distill_store.read_run(self._dir, run_id).get("status")
                in ACTIVE_STATUSES
            ):
                return self.attach_distill_run(run_id)
        return None

    @property
    def distill_runs(self) -> list[dict]:
        """All distill-run records (§12), reconciled to their real on-disk status."""
        self.reconcile_distill_runs()
        return distill_store.list_runs(self._dir)

    def distill_run_record(self, run_id: str) -> dict:
        """The full ``distill.yaml`` for one run, reconciled (Tenet 2).

        Raises:
            FileNotFoundError: If the run record does not exist.
        """
        record = distill_store.read_run(self._dir, run_id)
        if self._reconcile_distill_run(record):
            record = distill_store.read_run(self._dir, run_id)
        return record

    def reconcile_distill_runs(self) -> None:
        """Fold each still-active distill run's real on-disk status back into its record on
        experiment open (§12) — the SIGKILL backstop, mirroring :meth:`reconcile_runs`.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        for record in distill_store.list_runs(self._dir):
            self._reconcile_distill_run(record)

    def _distill_stage_status(
        self, name: str, stage: dict
    ) -> tuple[str | None, Path | None]:
        """The stage's on-disk ``job.json`` status + its run dir (``datasets/<id>/`` for
        label, ``models/<id>/`` for train), or ``(None, None)`` if it never produced one.
        The register stage runs inside the coordinator and has neither.
        """
        if name == "register":
            return None, None
        art_id = stage.get("dataset_id") if name == "label" else stage.get("model_id")
        if art_id is None:
            return None, None
        run_dir = (
            datasets_store.dataset_dir(self._dir, art_id)
            if name == "label"
            else models_store.model_dir(self._dir, art_id)
        )
        try:
            return rundir.read_status(run_dir).value, run_dir
        except (FileNotFoundError, KeyError, ValueError):
            return None, run_dir

    def _reconcile_distill_run(self, record: dict) -> bool:
        """Reconcile one distill record against on-disk evidence; returns True if changed.
        The **coordinator pid** is the liveness authority: alive → leave running; gone →
        adopt each stage's terminal ``job.json`` status, keep an individually-alive stage
        worker running, fail an active stage with no live worker, and roll up (complete only
        when training is)."""
        if record.get("status") not in ACTIVE_STATUSES:
            return False
        run_id = record["id"]
        pid = distill_store.read_coordinator_pid(self._dir, run_id)
        if pid is None:
            return False  # pid not recorded yet — just launched, leave running
        if coordinator_alive(distill_store.run_dir(self._dir, run_id), pid):
            return False  # the coordinator is driving it
        stages = dict(record.get("stages") or {})
        changed = False
        alive = False
        for name in distill_stage_names(record):
            stage = dict(stages.get(name) or {})
            if stage.get("status") not in ACTIVE_STATUSES:
                stages[name] = stage
                continue
            status, run_dir = self._distill_stage_status(name, stage)
            if status is None or status in ACTIVE_STATUSES:
                if run_dir is not None and status is not None and worker_alive(run_dir):
                    alive = True  # a detached stage worker still finishing — leave it
                    stages[name] = stage
                    continue
                stage["status"] = JobStatus.FAILED.value
                stage.setdefault(
                    "error", "interrupted: coordinator gone (reconciled on open)"
                )
            else:
                stage["status"] = status
                # Roll up whatever set_dataset_status/set_model_status just merged
                # into the stage's own record (§ run location, duration, cost).
                with contextlib.suppress(Exception):
                    reader = (
                        datasets_store.read_dataset_yaml
                        if name == "label"
                        else models_store.read_model_yaml
                    )
                    stage.update(_stage_metrics(reader(run_dir)))
            changed = True
            stages[name] = stage
        for name in _TEACHER_STAGES:
            stage = dict(stages.get(name) or {})
            if stage.get("status") not in ACTIVE_STATUSES:
                continue
            # The stage's processing run has its own coordinator and outlives this one; a
            # resume adopts it if it finished (`DistillOrchestrator._processing_stage`).
            stage["status"] = JobStatus.FAILED.value
            stage.setdefault(
                "error", "interrupted: coordinator gone (reconciled on open)"
            )
            stages[name] = stage
            changed = True
        record["stages"] = stages
        final = stages.get(distill_final_stage(record)) or {}
        if alive:
            new_status = "running"
        elif final.get("status") == JobStatus.COMPLETE.value:
            new_status = "complete"
        else:
            new_status = "failed"
        if new_status != record.get("status"):
            record["status"] = new_status
            changed = True
        if changed:
            distill_store.write_run(self._dir, run_id, record)
        return changed

    def _fail_distill_record(self, run_id: str, error: str) -> None:
        record = distill_store.read_run(self._dir, run_id)
        record["status"] = "failed"
        record.setdefault("error", error)
        distill_store.write_run(self._dir, run_id, record)

    def promote_model(self, model_id: str, name: str | None = None) -> str:
        """Promote a completed trained model into the machine-global catalog so it can be
        selected in *any* experiment (§12) — the on-demand step after a distill. The global
        model records its provenance (this experiment + the source dataset + base model +
        metrics + dataset composition + the per-epoch training history), and its weights are
        copied so it stays self-contained. Returns the assigned global model id.

        Raises:
            FileNotFoundError: If the model (or its weights file) does not exist.
            ValueError: If the model is not complete or the derived global id collides.
        """
        import_runner_modules(
            _PIPELINE_RUNNERS
        )  # populate registries for id validation
        # Computed here, not inside register_distilled_model: this is the one place that
        # already knows how to read a trainer's on-disk history (model_metric_history), and
        # the storage layer must stay free of trainer/registry imports (Tenet 13).
        training_history = self.model_metric_history(model_id)
        return distilled_store.register_distilled_model(
            self._dir,
            model_id,
            source_experiment_name=self.name,
            source_experiment_id=self._dir.name,
            name=name,
            training_history=training_history,
        )

    @property
    def processing_runs(self) -> list[dict]:
        """Experiment-level processing-run records — the runs index (§15)."""
        return runs_store.list_runs(self._dir)

    # --- Training datasets (§12) -------------------------------------------- #
    @property
    def datasets(self) -> list[DatasetRef]:
        """Training datasets in the experiment (scans ``datasets/``)."""
        refs = []
        for rec in dirscan.read_each(
            datasets_store.discover_datasets(self._dir),
            lambda d: datasets_store.read_dataset_yaml(
                datasets_store.dataset_dir(self._dir, d)
            ),
        ):
            refs.append(
                DatasetRef(
                    id=rec["id"],
                    name=rec.get("name", ""),
                    status=rec.get("status", "unknown"),
                    n_train=rec.get("n_train", 0),
                    n_val=rec.get("n_val", 0),
                    created_at=rec.get("created_at", ""),
                    compute=rec.get("compute"),
                    duration_s=rec.get("duration_s"),
                    estimated_cost_usd=rec.get("estimated_cost_usd"),
                )
            )
        return refs

    def dataset_record(self, dataset_id: str) -> dict:
        """The full ``dataset.yaml`` for one dataset (for detail views — Tenet 2)."""
        return datasets_store.read_dataset_yaml(
            datasets_store.dataset_dir(self._dir, dataset_id)
        )

    def generate_dataset(
        self,
        name: str,
        source_run_id: str,
        frame_selection: dict | None = None,
        val_split: float = 0.15,
        min_contour_area: int = 100,
    ) -> DatasetRun:
        """Export a YOLO training dataset from a completed detection/tracking run
        (§12). No new inference — reuses the run's masks. Returns a run handle.

        Raises:
            FileNotFoundError: If the source run record is absent.
            ValueError: If the source run is not a complete detection/tracking run.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        record = runs_store.read_run(self._dir, source_run_id)  # raises if missing
        if record.get("module_id") not in ("detection", "tracking"):
            raise ValueError(
                f"dataset source must be a detection or tracking run, got "
                f"{record.get('module_id')!r}"
            )
        if record.get("status") != "complete":
            raise ValueError(
                f"source run {source_run_id!r} is {record.get('status')!r}, "
                "not complete"
            )
        dataset_id = datasets_store.next_dataset_id(self._dir)
        job = DatasetExportJob(
            job_id=dataset_id,
            dataset_id=dataset_id,
            name=name,
            source_run_id=source_run_id,
            frame_selection=frame_selection or {"mode": "all"},
            val_split=val_split,
            min_contour_area=min_contour_area,
            created_at=utc_timestamp(),
        )
        # An initial record so the dataset is discoverable while it runs, written
        # *before* submit so the runner's completion write (full record) is always
        # last — never clobbered by this one (the worker can finish fast).
        ddir = datasets_store.dataset_dir(self._dir, dataset_id)
        ddir.mkdir(parents=True, exist_ok=True)
        datasets_store.write_dataset_yaml(
            ddir,
            {
                "id": dataset_id,
                "name": name,
                "source": {"kind": "run_export", "run_id": source_run_id},
                "status": "running",
            },
        )
        executor = LocalExecutor(self._dir, runner_modules=_PIPELINE_RUNNERS)
        executor.submit(job)  # run_dir = datasets/<id>/; writes job.json there
        return DatasetRun(ExecutorRunHandle(executor, dataset_id), self, dataset_id)

    def label_dataset(
        self,
        name: str,
        classes: list[LabelingClass],
        n_frames: int,
        source_recording_ids: list[str] | None = None,
        logit_threshold: float = 0.5,
        min_contour_area: int = 100,
        val_split: float = 0.15,
        compute: ComputeTarget | None = None,
        sampling: str = "stratified",
        model_id: str = "sam3",
        model_variant: str | None = None,
        background_share: float = LabelingConfig.model_fields[
            "background_share"
        ].default,
    ) -> DatasetRun:
        """Build a YOLO dataset by running a prompt-capable model on a sample of frames (§12) —
        the distill-into-YOLO source, no existing detection run needed. ``model_id`` picks which
        prompt-capable model labels (``sam3`` by default), ``model_variant`` its size
        (``None`` = the model's default). ``n_frames`` is sampled proportionally to recording
        length; ``sampling`` picks how frames are drawn within each video (``"stratified"`` spreads
        them evenly across time, ``"uniform"`` is fully random). ``source_recording_ids=None`` uses
        every recording. Frames where the model found none of the classes are kept with an
        empty label, up to ``background_share`` of the dataset. Returns a run handle.

        The flat scalar parameters (``model_id``, ``model_variant``, ``n_frames``, ``sampling``,
        ``logit_threshold``, ``min_contour_area``, ``val_split``, ``background_share``) are
        validated by
        :class:`LabelingConfig`; the checks below need recording data and stay here.

        Raises:
            ValueError: no classes, duplicate class prompts, a scalar out of its model bounds
                (bad ``sampling`` / ``val_split`` / ``n_frames`` / thresholds), a model that is not
                prompt-capable, a recording without video, ``n_frames`` above the frames available,
                or the chosen model not ready.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        recording_ids = list(source_recording_ids or [s.id for s in self.recordings])
        compute = compute or default_compute(api_models.supported_backends(model_id))
        if effective_backend(compute) not in api_models.supported_backends(model_id):
            raise ValueError(
                f"model {model_id!r} does not support the {compute.backend!r} backend; "
                f"it requires one of {api_models.supported_backends(model_id)}"
            )
        _require_remote_target_registered(compute)
        if not classes:
            raise ValueError("labeling needs at least one class (a name + text prompt)")
        prompts = [c.prompt for c in classes]
        if len(set(prompts)) != len(prompts):
            raise ValueError(
                "two classes share a text prompt; prompts must be distinct"
            )
        capable = api_models.prompt_capable_models()
        if model_id not in capable:
            raise ValueError(
                f"model {model_id!r} cannot label — labeling needs a prompt-capable model; "
                f"choose one of {capable}"
            )
        cfg = LabelingConfig.model_validate(
            {
                "model_id": model_id,
                "model_variant": model_variant,
                "n_frames": n_frames,
                "sampling": sampling,
                "logit_threshold": logit_threshold,
                "min_contour_area": min_contour_area,
                "val_split": val_split,
                "background_share": background_share,
            }
        )
        if not recording_ids:
            raise ValueError("labeling needs at least one source recording")
        available = 0
        for sid in recording_ids:
            recording = self.get_recording(sid)  # raises if the recording is unknown
            data = recording_store.read_recording(recording.dir)
            files = data.get("video", {}).get("files", [])
            if not files:
                raise ValueError(f"recording {sid!r} has no video to sample")
            # Frame count from the PTS timestamp map (Tenet 8 / CLAUDE.md frame-count rule), not
            # cv2's CAP_PROP_FRAME_COUNT estimate — so the n_frames check matches what's sampleable.
            available += sum(
                len(recording_store.load_active_timestamps(recording.dir, data, i))
                for i in range(len(files))
            )
        if cfg.n_frames > available:
            raise ValueError(
                f"n_frames={cfg.n_frames} exceeds the {available} frames available across "
                "the selected recordings"
            )
        # A fact about *this* machine's disk/network access to the model. A remote job
        # never loads it here — the shepherd rewrites compute to the on-cluster backend
        # and the model actually loads (and is readiness-checked again) on the cluster
        # by remote_shepherd.py before the sbatch submission — so gating submission on
        # this machine's own access would demand it for a job that never runs here.
        # Mirrors detection.py's / embedding.py's identical guard.
        if compute.backend != "remote":
            status = self.check_model_access(cfg.model_id)
            if not status.ready:
                raise ValueError(f"{cfg.model_id} is not ready — {status.message}")

        dataset_id = datasets_store.next_dataset_id(self._dir)
        job = LabelingJob(
            job_id=dataset_id,
            dataset_id=dataset_id,
            name=name,
            source_recording_ids=recording_ids,
            classes=classes,
            n_frames=cfg.n_frames,
            logit_threshold=cfg.logit_threshold,
            min_contour_area=cfg.min_contour_area,
            val_split=cfg.val_split,
            background_share=cfg.background_share,
            compute=compute,
            created_at=utc_timestamp(),
            sampling=cfg.sampling,
            model_id=cfg.model_id,
            model_variant=cfg.model_variant,
        )
        ddir = datasets_store.dataset_dir(self._dir, dataset_id)
        ddir.mkdir(parents=True, exist_ok=True)
        datasets_store.write_dataset_yaml(
            ddir,
            {
                "id": dataset_id,
                "name": name,
                "source": {"kind": "labeling", "model": cfg.model_id},
                "status": "running",
                "created_at": job.created_at,
                # Known at reservation time — where it will run. duration_s/
                # estimated_cost_usd arrive later, merged in by set_dataset_status
                # once the job is terminal (§ run location/duration/cost).
                "compute": compute.to_dict(),
            },
        )
        executor = executor_for(self._dir, compute, _PIPELINE_RUNNERS)
        executor.submit(job)
        return DatasetRun(ExecutorRunHandle(executor, dataset_id), self, dataset_id)

    # --- Keypoints (§26): one-line methods delegating to api/keypoints.py -------- #
    # Each keypoint PR (2 example sets, 4 detect) adds its
    # methods here as one-liners that import `detail.api.keypoints` locally and forward
    # to it — the shape `query_run` has — so three PRs never edit one region's body.
    # -- keypoints (§26): PR 8 -- distill keypoints (api/keypoint_distill.py)
    def keypoint_dataset(
        self, name: str, detect_runs=None, n_frames: int | None = None, **options
    ) -> "DatasetRun":
        """Build a YOLO-pose dataset (§12 *Distill keypoints*, the label stage on its own).
        The teacher is ``detect_runs`` (finished keypoint detect runs: recording names or ids,
        each with a run id or ``None`` for the only one; ``tracking=`` may name the tracking
        run each recording's poses are assigned to, else the one on the detect run's
        detection run) or the patch method (``example_set=``, ``tracking=``, ``model_id=``,
        ``label_compute=``); the other options are ``KeypointDatasetConfig``'s. Returns a
        ``DatasetRun``."""
        from detail.api import keypoint_distill as _kd

        return _kd.keypoint_dataset(self, name, detect_runs, n_frames, **options)

    def test_keypoints(self, detect_runs=None, **options):
        """Test a keypoint teacher on about a dozen frames drawn the way a distill would, with
        the held-out check against the example clicks; writes nothing (§12, *Test
        cut-off…*). Takes ``distill_keypoints``'s teacher keywords plus ``n_frames`` and
        ``seed``. Returns a ``TeacherTest``."""
        from detail.api import keypoint_distill as _kd

        return _kd.keypoint_test(self, detect_runs, **options)

    def keypoint_teacher_plan(self, window_s: float = 300.0) -> list[dict]:
        """One proposed teacher recording per acquisition setup (§26, #717); see
        ``api.keypoint_distill.keypoint_teacher_plan``."""
        from detail.api import keypoint_distill as _kd

        return _kd.keypoint_teacher_plan(self, window_s)

    def start_keypoint_setup(
        self,
        name: str,
        detector_model_id: str,
        class_name: str | None = None,
        **options,
    ) -> "AttachedDistillRun":
        """Build a keypoint teacher across the acquisition setups and distill its student,
        as one durable run that pauses for an example set (§26, #717); ``class_names=[…]``
        in place of ``class_name`` sets up two or more animals, each tracked, then merged,
        with one set per class. See ``api.keypoint_distill.start_keypoint_setup`` for the
        options."""
        from detail.api import keypoint_distill as _kd

        return _kd.start_keypoint_setup(
            self, name, detector_model_id, class_name, **options
        )

    def continue_keypoint_setup(
        self,
        run_id: str,
        example_set: str | None = None,
        score_cutoff: float | None = None,
        class_example_sets: dict[str, str] | None = None,
    ) -> "AttachedDistillRun":
        """Carry a keypoint setup on past its pause with ``example_set`` — or, for a setup
        of several animals, ``class_example_sets`` (one set per class) (§26, #717)."""
        from detail.api import keypoint_distill as _kd

        return _kd.continue_keypoint_setup(
            self, run_id, example_set, score_cutoff, class_example_sets
        )

    def distill_keypoints(
        self, name: str, detect_runs=None, n_frames: int | None = None, **options
    ) -> "AttachedDistillRun":
        """Distill a teacher into a fast YOLO-pose student in one durable run (label →
        train → register). The teacher is ``detect_runs`` or the patch method
        (``example_set=`` + ``tracking=``), as :meth:`keypoint_dataset` takes it. Options:
        ``KeypointDatasetConfig``'s fields plus ``base_model`` (a ``-pose`` checkpoint),
        ``trainer_id``, ``hyperparams``, ``train_compute`` and ``label_compute``. Returns an
        :class:`AttachedDistillRun`."""
        from detail.api import keypoint_distill as _kd

        return _kd.distill_keypoints(self, name, detect_runs, n_frames, **options)

    # -- keypoints (§26): PR 2 -- example sets
    def example_sets(self, purpose: str | None = "example") -> list:
        """This experiment's example sets of one purpose (``None`` for all), id order."""
        from detail.api import keypoints as _keypoints

        return _keypoints.example_sets(self, purpose)

    def example_set(self, set_id: str):
        """One example set; ``FileNotFoundError`` names the sets that exist."""
        from detail.api import keypoints as _keypoints

        return _keypoints.example_set(self, set_id)

    def create_example_set(self, name: str, skeleton, *, purpose: str = "example"):
        """A new, empty example set holding a copy of ``skeleton``."""
        from detail.api import keypoints as _keypoints

        return _keypoints.create_example_set(self, name, skeleton, purpose=purpose)

    def sample_example_frames(self, set_id: str, sources, n_frames: int = 5, **options):
        """Copy tracked frames into an example set (``varied`` or ``spread``); see
        ``api.keypoints.sample_example_frames`` for the options."""
        from detail.api import keypoints as _keypoints

        return _keypoints.sample_example_frames(
            self, set_id, sources, n_frames, **options
        )

    def add_example_frame(
        self, set_id, recording, tracking_run_id, part_id, frame_idx, track_id
    ):
        """Copy one navigated-to frame into an example set."""
        from detail.api import keypoints as _keypoints

        return _keypoints.add_example_frame(
            self, set_id, recording, tracking_run_id, part_id, frame_idx, track_id
        )

    def set_example_marks(self, set_id: str, frame_index: int, marks):
        """Replace one frame's marks and write the set atomically."""
        from detail.api import keypoints as _keypoints

        return _keypoints.set_example_marks(self, set_id, frame_index, marks)

    def remove_example_frame(self, set_id: str, frame_index: int):
        """Drop one frame from an example set (its PNG stays)."""
        from detail.api import keypoints as _keypoints

        return _keypoints.remove_example_frame(self, set_id, frame_index)

    def reject_example_frame(self, set_id: str, frame_index: int, bucket=None):
        """Drop one frame and never offer it again (for ``bucket``, or for any)."""
        from detail.api import keypoints as _keypoints

        return _keypoints.reject_example_frame(self, set_id, frame_index, bucket)

    def reject_candidate(self, set_id: str, key, bucket=None):
        """Never offer a frame not in the set again (for ``bucket``, or for any)."""
        from detail.api import keypoints as _keypoints

        return _keypoints.reject_candidate(self, set_id, key, bucket)

    def heading_pool(self, sources, **options):
        """Every frame's heading bucket over a recording subset (§26 *Headings*); see
        ``api.keypoints.heading_pool``."""
        from detail.api import keypoints as _keypoints

        return _keypoints.heading_pool(self, sources, **options)

    def heading_candidates(self, set_id: str, pool, bucket: str, n: int = 8):
        """A heading bucket's next candidates from ``pool``, best first."""
        from detail.api import keypoints as _keypoints

        return _keypoints.heading_candidates(self, set_id, pool, bucket, n)

    def add_heading_candidates(self, set_id: str, chosen, bucket: str):
        """Copy chosen heading candidates into an example set."""
        from detail.api import keypoints as _keypoints

        return _keypoints.add_heading_candidates(self, set_id, chosen, bucket)

    def heading_coverage(self, set_id: str, pool=None):
        """Per heading bucket, the set's frames in it and how many are marked."""
        from detail.api import keypoints as _keypoints

        return _keypoints.heading_coverage(self, set_id, pool)

    def heading_applies(self, recording_ids) -> bool:
        """Whether sampling by heading means anything for these recordings: not when every
        one's camera view is top down (§26 *Headings*, §10)."""
        from detail.api import keypoints as _keypoints

        return _keypoints.heading_applies(self, recording_ids)

    def example_frame_image(self, set_id: str, frame_index: int):
        """One example frame's picture as RGB uint8."""
        from detail.api import keypoints as _keypoints

        return _keypoints.example_frame_image(self, set_id, frame_index)

    def update_example_set(self, set_id: str, *, name=None, skeleton=None):
        """Rename an example set, and/or change its skeleton until its first mark."""
        from detail.api import keypoints as _keypoints

        return _keypoints.update_example_set(self, set_id, name=name, skeleton=skeleton)

    def duplicate_example_set(self, set_id: str, *, name=None):
        """A new, empty set holding a copy of this set's skeleton."""
        from detail.api import keypoints as _keypoints

        return _keypoints.duplicate_example_set(self, set_id, name=name)

    def export_example_set(self, set_id: str, out_dir):
        """Write a set (``examples.yaml`` + its frame images) to a folder."""
        from detail.api import keypoints as _keypoints

        return _keypoints.export_example_set(self, set_id, out_dir)

    def example_set_runs(self, set_id: str) -> list:
        """``(recording_id, run_id)`` of the keypoint detect runs that list a set."""
        from detail.api import keypoints as _keypoints

        return _keypoints.example_set_runs(self, set_id)

    def import_example_set(self, yaml_path):
        """Read a prepared example set (YAML + images beside it) in under a fresh id."""
        from detail.api import keypoints as _keypoints

        return _keypoints.import_example_set(self, yaml_path)

    def delete_example_set(self, set_id: str) -> None:
        """Delete an example set no keypoint run lists."""
        from detail.api import keypoints as _keypoints

        _keypoints.delete_example_set(self, set_id)

    # -- keypoints (§26): tuning a track method
    def tune_example_sets(self, keypoint_detect_spec: dict | None = None) -> list[dict]:
        """The example sets Tune tracking could score with, each marked usable or not with
        the reason (the detect run's own set; another skeleton). Hashes images: off-thread.
        """
        from detail.api import keypoints as _keypoints

        return _keypoints.tune_example_sets(self, keypoint_detect_spec)

    def tune_keypoint_tracking(self, example_set_id: str, **options):
        """Score keypoint tracking settings against an example set the detect run did not
        learn from; derived, never stored. Options: ``detect_run_id`` or
        ``keypoint_detect_spec``, ``tracking_run_id`` or ``tracking_spec`` (neither: the one
        on the detection run), ``match``, ``base_config``, ``seed``, ``progress``
        (``api.keypoints.tune_keypoint_tracking``)."""
        from detail.api import keypoints as _keypoints

        return _keypoints.tune_keypoint_tracking(self, example_set_id, **options)

    # -- keypoints (§26): previewing a track method
    def preview_keypoint_tracking(
        self, config: dict, recordings: list[str] | None = None, **options
    ):
        """Run keypoint tracking in memory over a short window of a few recordings — the
        current settings, detection alone and up to two variants — and describe each, without
        labels; derived, never stored. ``config`` is the track stage's launch config.
        Options: ``starts``, ``duration_s``, ``variants``, ``grid`` (every *Tune tracking*
        grid setting, the provisional front marked), ``progress``
        (``api.keypoints.preview_keypoint_tracking``)."""
        from detail.api import keypoints as _keypoints

        return _keypoints.preview_keypoint_tracking(self, config, recordings, **options)

    # --- Trained models (§12) ---------------------------------------------- #
    @property
    def models(self) -> list[ModelRef]:
        """Trained models in the experiment (scans ``models/``)."""
        refs = []
        for rec in dirscan.read_each(
            models_store.discover_models(self._dir),
            lambda m: models_store.read_model_yaml(
                models_store.model_dir(self._dir, m)
            ),
        ):
            refs.append(
                ModelRef(
                    id=rec["id"],
                    name=rec.get("name", ""),
                    status=rec.get("status", "unknown"),
                    trainer_id=rec.get("trainer_id", ""),
                    dataset_id=rec.get("dataset_id", ""),
                    created_at=rec.get("created_at", ""),
                    compute=rec.get("compute"),
                    duration_s=rec.get("duration_s"),
                    estimated_cost_usd=rec.get("estimated_cost_usd"),
                    role=rec.get("role")
                    or ("keypoints" if rec.get("task") == "pose" else "detection"),
                )
            )
        return refs

    def model_names(self) -> dict[str, str]:
        """``{model_id: name}`` for every model this experiment can name — its own trained
        models plus the machine-global imported ones (§5/§12). The model twin of
        :meth:`recording_names`.

        Records key on the model **id**, which is permanent; the **name** is what people
        read (§11). A label built from this falls back to the id for anything absent — a
        built-in model, whose id *is* its name, or a model deleted since the run. Cost is
        O(models), not O(recordings), so it is cheap enough to build per page render.
        """
        names = {m.id: m.name for m in self.models if m.name}
        names.update({m.id: m.name for m in self.imported_models if m.name})
        return names

    def model_record(self, model_id: str) -> dict:
        """The full ``model.yaml`` for one trained model (for detail views — Tenet 2)."""
        return models_store.read_model_yaml(models_store.model_dir(self._dir, model_id))

    def model_metric_history(self, model_id: str) -> list[dict[str, float]]:
        """Per-epoch metrics for a trained model, oldest first; ``[]`` if it has none yet.

        The series behind `model_record`'s summary. `final_metrics` answers *how good is this
        model*; someone deciding whether to stop training early is asking the other question —
        *is it still getting better* — and one number cannot tell a run that has been improving
        for an hour from one that plateaued forty epochs ago.

        Readable while the run is live, because it comes off the trainer's own on-disk record
        rather than progress events — so it is whole on a page opened after the run began, and a
        model that trained overnight has its curve in the morning.

        Returns ``[]`` rather than raising when the trainer is not installed on this machine:
        reading a finished run's record must not require the backend that produced it (a model
        trained elsewhere and imported has no Ultralytics here), and a missing curve is the
        honest rendering of a history we cannot read.
        """
        record = self.model_record(model_id)  # FileNotFoundError if the model is gone
        # The registry's entrypoint, not a hope that something earlier in this process ran it:
        # `train_model` does, but a GUI that opens an existing model — or watches a distill whose
        # training runs in its detached coordinator — never trains here, found `TRAINERS` empty,
        # and drew no curve at all. The test session imports every registry up front, so the
        # suite could not see it.
        import_trainers()
        trainer_cls = TRAINERS.get(record.get("trainer_id", ""))
        if trainer_cls is None:
            return []
        return trainer_cls().metric_history(models_store.model_dir(self._dir, model_id))

    def train_model(
        self,
        name: str,
        dataset_id: str,
        trainer_id: str,
        base_model: str,
        hyperparams: dict | None = None,
        compute: ComputeTarget | None = None,
    ) -> ModelRun:
        """Train a custom detection model from a dataset (§12). Returns a handle.

        Raises:
            FileNotFoundError: If the dataset does not exist.
            ValueError: If the dataset is not complete, the trainer is unknown or
                not ready, the backend is unsupported, or hyperparams are invalid.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        ddir = datasets_store.dataset_dir(self._dir, dataset_id)
        if not (ddir / datasets_store.DATASET_FILENAME).is_file():
            raise FileNotFoundError(f"dataset {dataset_id!r} not found")
        dataset = datasets_store.read_dataset_yaml(ddir)
        if dataset.get("status") != "complete":
            raise ValueError(f"dataset {dataset_id!r} is not complete")
        check_base_model_task(dataset_id, dataset.get("task"), base_model)
        trainer_cls = TRAINERS.get(trainer_id)
        if trainer_cls is None:
            raise ValueError(
                f"unknown trainer_id {trainer_id!r}; available: {sorted(TRAINERS)}"
            )
        trainer = trainer_cls()
        compute = compute or default_compute(trainer.supported_backends)
        # A fact about *this* machine's own backend (e.g. is ultralytics importable
        # here) — a remote job trains entirely on the cluster, so gating submission on
        # it would demand this machine have the training framework for a job that
        # never trains here. Mirrors label_dataset's / detection's identical guard.
        if compute.backend != "remote" and not trainer.is_ready():
            raise ValueError(
                f"trainer {trainer_id!r} is not ready (install its backend)"
            )
        if effective_backend(compute) not in trainer.supported_backends:
            raise ValueError(
                f"trainer {trainer_id!r} does not support the {compute.backend!r} "
                f"backend; supports {trainer.supported_backends}"
            )
        _require_remote_target_registered(compute)
        if compute.backend == "cuda" and compute.device_indices:
            check_cuda_indices(compute.device_indices)
        hyperparams = trainer.validate_hyperparams(hyperparams or {})
        # A base checkpoint that cannot learn the dataset's labels (a -seg base on a pose
        # dataset, or the reverse) fails here, before anything is downloaded or written.
        check_base = getattr(trainer, "check_base_model", None)
        if check_base is not None:
            check_base(base_model, ddir)
        model_id = models_store.next_model_id(self._dir)
        job = TrainingJob(
            job_id=model_id,
            model_id=model_id,
            dataset_id=dataset_id,
            trainer_id=trainer_id,
            base_model=base_model,
            hyperparams=hyperparams,
            compute=compute,
            created_at=utc_timestamp(),
        )
        # Initial record (before submit) carrying the name + lineage; the runner
        # fills in classes/weights/metrics/status on completion.
        mdir = models_store.model_dir(self._dir, model_id)
        mdir.mkdir(parents=True, exist_ok=True)
        models_store.write_model_yaml(
            mdir,
            {
                "id": model_id,
                "name": name,
                "trainer_id": trainer_id,
                "dataset_id": dataset_id,
                "base_model": base_model,
                # Resolved (defaults filled) before the run starts: the live training curve
                # and `train status` read the epoch count from here.
                "hyperparams": dict(hyperparams),
                "status": "running",
                "created_at": job.created_at,
                # Known at reservation time — where it will run (see label_dataset's
                # identical field). duration_s/estimated_cost_usd arrive later, merged
                # in by set_model_status once the job is terminal.
                "compute": compute.to_dict(),
            },
        )
        executor = executor_for(self._dir, compute, _PIPELINE_RUNNERS)
        executor.submit(job)  # run_dir = models/<id>/; writes job.json there
        return ModelRun(ExecutorRunHandle(executor, model_id), self, model_id)

    def resume_training(self, model_id: str) -> ModelRun:
        """Resume a training that was killed or crashed mid-run (cancelled / failed) from
        its last checkpoint instead of restarting from epoch 0 (§12). Re-submits the original
        job; the runner picks up ``run/weights/last.pt``. Returns a handle.

        A ``partial`` model is refused: it stopped cleanly, so its checkpoint was finalized
        and holds nothing to resume (:data:`RESUMABLE_TRAINING_STATUSES`).

        Raises:
            FileNotFoundError: If the model does not exist.
            ValueError: If the model finished (complete / partial) or is running, its
                recorded training job is missing/unreadable, its trainer is unknown, or it has
                no checkpoint to resume from.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        mdir = models_store.model_dir(self._dir, model_id)
        record = models_store.read_model_yaml(mdir)  # FileNotFoundError if absent
        status = record.get("status")
        if status == "partial":
            raise ValueError(
                f"model {model_id!r} stopped cleanly, so its training is finished and its "
                "checkpoint holds nothing to resume. Its best weights are usable as they are; "
                "train a new model to train further."
            )
        if status not in RESUMABLE_TRAINING_STATUSES:
            raise ValueError(
                f"model {model_id!r} is {status!r}; only a cancelled or failed training "
                "can be resumed"
            )
        try:
            job = rundir.read_job(
                mdir
            )  # the original TrainingJob (run_dir = models/<id>/)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(
                f"model {model_id!r} has no recorded training job to resume ({exc})"
            ) from exc
        trainer_cls = TRAINERS.get(job.trainer_id)
        if trainer_cls is None:
            raise ValueError(
                f"unknown trainer_id {job.trainer_id!r} for model {model_id!r}"
            )
        if trainer_cls().checkpoint_path(mdir) is None:
            raise ValueError(
                f"model {model_id!r} has no checkpoint to resume from — train it first"
            )
        live = live_process_pid(mdir)
        if live is not None:
            raise ValueError(
                f"model {model_id!r} still has a training process running (pid {live}) — "
                "cancel it in DETAIL before resuming. A force-kill of the trainer alone "
                "leaves the worker above it alive and holding the GPU."
            )
        models_store.set_model_status(mdir, "running")
        # A cancelled/partial run left a `cancel` marker in the run dir; clear it or the resumed
        # worker's cancel watcher sees it and immediately re-cancels (cf. api/annotation.py).
        rundir.clear_cancel(mdir)
        # job.compute is the original run's target, already validated when it was first
        # submitted — resume re-dispatches through it unchanged, the same way a local
        # resume never re-checks CUDA availability either.
        executor = executor_for(self._dir, job.compute, _PIPELINE_RUNNERS)
        executor.submit(job)  # same run dir: last.pt is present, so the runner resumes
        return ModelRun(ExecutorRunHandle(executor, model_id), self, model_id)

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a training dataset (images + labels + ``dataset.yaml``).

        Blocked only while *active* work would lose its inputs (gui_design §1.1):
        the dataset's own export, or a model still training from it. A *completed*
        model is self-contained (it copied its weights), so its lineage reference
        is left to dangle — it does not block.

        Raises:
            FileNotFoundError: If no such dataset exists.
            ValueError: If the dataset's export or a model training from it is active.
        """
        record = datasets_store.read_dataset_yaml(  # FileNotFoundError if absent
            datasets_store.dataset_dir(self._dir, dataset_id)
        )
        if record.get("status") in ACTIVE_STATUSES:
            raise ValueError(
                f"dataset {dataset_id!r} is still being generated — wait for it"
            )
        blockers = [
            m.id
            for m in self.models
            if m.dataset_id == dataset_id and m.status in ACTIVE_STATUSES
        ]
        if blockers:
            raise ValueError(
                f"cannot delete dataset {dataset_id!r} — a model is training from "
                f"it: {', '.join(blockers)}"
            )
        datasets_store.delete_dataset(self._dir, dataset_id)

    def delete_model(self, model_id: str) -> None:
        """Delete a trained model (``model.yaml`` + trainer run + weights).

        Blocked only while *active* work would lose its inputs (gui_design §1.1):
        the model's own training, or a detection run still using it. A *completed*
        detection run keeps its own masks, so its lineage reference is left to
        dangle — it does not block.

        Raises:
            FileNotFoundError: If no such model exists.
            ValueError: If the model's training or a detection run using it is active.
        """
        record = models_store.read_model_yaml(  # FileNotFoundError if absent
            models_store.model_dir(self._dir, model_id)
        )
        if record.get("status") in ACTIVE_STATUSES:
            raise ValueError(f"model {model_id!r} is still training — wait for it")
        blockers = [
            run["id"]
            for run in self.processing_runs
            if run.get("module_id") == "detection"
            and run.get("status") in ACTIVE_STATUSES
            and run.get("config", {}).get("weights_source") == "trained"
            and run.get("config", {}).get("model_id") == model_id
        ]
        if blockers:
            raise ValueError(
                f"cannot delete model {model_id!r} — a detection run is using it: "
                f"{', '.join(blockers)}"
            )
        models_store.delete_model(self._dir, model_id)

    def delete_run(self, run_id: str) -> None:
        """Delete a processing run: its per-recording outputs (zarr stores +
        recording.yaml entries) and the experiment-level run record.

        Refuses if any per-recording output is still depended on by a downstream
        run (e.g. tracking built on this detection) — delete those first. Nothing
        is removed until the whole run is clear to delete (Tenet 10).

        Raises:
            FileNotFoundError: If no such run record exists.
            ValueError: If the run is still active, or a downstream run depends on it.
        """
        record = runs_store.read_run(self._dir, run_id)
        if record.get("status") in ACTIVE_STATUSES:
            raise ValueError(f"run {run_id!r} is still active — wait for it to finish")
        module_id = record["module_id"]
        per_recording = record.get("per_recording", {})

        blockers = []
        for recording_id, info in per_recording.items():
            stage_run_id = info.get("run_id")
            if stage_run_id is None:
                continue
            recording = self.get_recording(recording_id)
            for dep_module, dep_run_id in recording.run_dependents(
                module_id, stage_run_id
            ):
                blockers.append(
                    f"  {recording_id}: {dep_module} run {dep_run_id!r} depends on "
                    f"{stage_run_id!r}"
                )
        if blockers:
            raise ValueError(
                f"cannot delete run {run_id!r} — delete its dependents first:\n"
                + "\n".join(blockers)
            )

        active = self._active_analysis_blockers(module_id, per_recording)
        if active:
            raise ValueError(
                f"cannot delete run {run_id!r} — an analysis run is still reading it:\n"
                + "\n".join(active)
            )

        for recording_id, info in per_recording.items():
            stage_run_id = info.get("run_id")
            if stage_run_id is not None:
                self.get_recording(recording_id).delete_run(module_id, stage_run_id)
        runs_store.delete_run(self._dir, run_id)

    def _active_analysis_blockers(
        self, module_id: str, per_recording: dict
    ) -> list[str]:
        """Active analysis runs that consume one of this run's per-recording stage runs.

        A *completed* analysis run copied its derived values into its own CSVs, so it
        never blocks (it dangles, like a dataset's source) — only an **active** run,
        mid-read of the source, does. Location export consumes tracking runs
        (`track_merge`'s included — it writes into the same `tracking_runs` collection and
        `analysis_record`'s `tracking_run_id` names either kind indifferently); future
        analysis modules consuming other stages would extend the field check here.
        """
        if module_id not in ("tracking", "track_merge"):
            return []
        stage_ids = {info.get("run_id") for info in per_recording.values()}
        recordings = set(per_recording)
        blockers = []
        for ref in self.analysis_runs:
            if ref.status not in ACTIVE_STATUSES:
                continue
            rec = self.analysis_record(ref.id)
            if rec.get("tracking_run_id") in stage_ids and (
                set(rec.get("recording_ids", [])) & recordings
            ):
                blockers.append(f"  active analysis run {ref.id!r} is using it")
        return blockers

    def cancel_run(self, run_id: str) -> None:
        """Stop an active processing run: signal a live orchestrator (the on-disk
        cancel sentinel — works even after the app restarted), cancel each in-flight
        per-recording worker, and record each recording's real on-disk status. The run
        record becomes ``cancelled`` unless every recording actually completed (then
        ``complete`` — e.g. a worker that finished after the GUI was closed, so its
        results aren't hidden). No-op once terminal.

        Raises:
            FileNotFoundError: If no such run record exists.
        """
        record = runs_store.read_run(self._dir, run_id)
        if record.get("status") not in ACTIVE_STATUSES:
            return
        runs_store.request_cancel(
            self._dir, run_id
        )  # cooperative stop for the coordinator
        # Kill the coordinator so it can't launch the next recording before it notices
        # the flag; its own finalize and this record write both converge on cancelled.
        kill_coordinator(
            runs_store.run_dir(self._dir, run_id),
            runs_store.read_coordinator_pid(self._dir, run_id),
        )
        module_id = record["module_id"]
        per_recording = record.get("per_recording", {})
        for recording_id, info in per_recording.items():
            if info.get("status") not in ACTIVE_STATUSES:
                continue  # a recording that already finished keeps its real status
            stage_run_id = info.get("run_id")
            if stage_run_id is not None:
                info["status"] = self.get_recording(recording_id).cancel_run(
                    module_id, stage_run_id
                )
            else:
                info["status"] = "cancelled"
        statuses = [info.get("status") for info in per_recording.values()]
        record["status"] = (
            "complete"
            if statuses and all(s == "complete" for s in statuses)
            else "cancelled"
        )
        runs_store.write_run(self._dir, run_id, record)

    def cancel_model(self, model_id: str) -> None:
        """Stop an active model-training run (kill its worker), then record its real
        on-disk status — ``cancelled`` if killed, but ``complete`` if the worker
        actually finished (so a done run isn't mislabeled). No-op once terminal.
        Raises FileNotFoundError if the model doesn't exist."""
        mdir = models_store.model_dir(self._dir, model_id)
        record = models_store.read_model_yaml(mdir)  # FileNotFoundError if absent
        if record.get("status") not in ACTIVE_STATUSES:
            return
        cancel_run_dir(mdir)  # models/<id>/ is the run dir (job.json lives there)
        models_store.set_model_status(mdir, rundir.read_status(mdir).value)

    def stop_model(self, model_id: str) -> None:
        """Ask a training run to stop at its next epoch and **keep what it has**.

        The counterpart to :meth:`cancel_model`, and the reason the two exist separately: cancel
        force-kills after three seconds, which on a run whose epochs take minutes means the
        worker dies mid-epoch and never writes ``weights.best`` or ``metrics`` into the record.
        The checkpoint file is on disk and nothing references it.

        This only asks. The run stops at its next epoch boundary, runs its own completion path,
        and records the best weights it reached — which is what *"the metrics look good, stop
        here"* has to mean to be worth offering. The record's status stays as the worker leaves
        it (``complete``), because the run did finish, early.

        No-op once terminal. Raises FileNotFoundError if the model does not exist.
        """
        mdir = models_store.model_dir(self._dir, model_id)
        record = models_store.read_model_yaml(mdir)  # FileNotFoundError if absent
        if record.get("status") not in ACTIVE_STATUSES:
            return
        request_stop(mdir)

    def cancel_dataset(self, dataset_id: str) -> None:
        """Stop an active dataset-export run (kill its worker), then record its real
        on-disk status (``cancelled`` if killed, ``complete`` if it finished). No-op
        once terminal. Raises FileNotFoundError if the dataset doesn't exist."""
        ddir = datasets_store.dataset_dir(self._dir, dataset_id)
        record = datasets_store.read_dataset_yaml(ddir)  # FileNotFoundError if absent
        if record.get("status") not in ACTIVE_STATUSES:
            return
        cancel_run_dir(ddir)
        datasets_store.set_dataset_status(ddir, rundir.read_status(ddir).value)

    # --- Reattach to a single-job run (§12/§16) ------------------------------ #
    # The four single-job run kinds differ only in where their record lives, so one table
    # drives every attach. Each entry is (run_dir, read_record, set_status); the run dir is
    # also the executor's, holding the job.json AttachedJobRun polls.
    _JOB_RUN_KINDS = {
        "analysis": (
            analysis_store.analysis_dir,
            analysis_store.read_analysis_yaml,
            analysis_store.set_analysis_status,
            analysis_store.discover_analyses,
        ),
        "dataset": (
            datasets_store.dataset_dir,
            datasets_store.read_dataset_yaml,
            datasets_store.set_dataset_status,
            datasets_store.discover_datasets,
        ),
        "model": (
            models_store.model_dir,
            models_store.read_model_yaml,
            models_store.set_model_status,
            models_store.discover_models,
        ),
        # `export_runs_store`, not `export_store` — the latter is the experiment *bundle*
        # (`storage/export.py`). Three different things want the word "export" in this package
        # and the run store is the newest of them, so it is the one that qualifies its name.
        "export": (
            export_runs_store.export_dir,
            export_runs_store.read_export_yaml,
            export_runs_store.set_export_status,
            export_runs_store.discover_exports,
        ),
    }

    def attach_job_run(self, kind: str, run_id: str) -> AttachedJobRun:
        """Reattach to a single-job run — a live one streams its worker's progress, a
        terminal one returns at once. Works from any process, after an app restart
        included: the worker is detached and its progress is on disk.

        ``kind`` is a key of ``_JOB_RUN_KINDS``. The typed wrappers below are what callers
        use; this is the shared body.

        **The runner modules are imported first, and that is not optional.** A progress event is
        rebuilt from ``job.json`` by tag, and every tag outside `core/dataclasses.py` — every
        analysis module's, and the export's — is registered as a side effect of importing the
        module that defines it. Reattaching without that raises ``unknown progress event type``
        the moment the first event is read. No test saw it because ``conftest``'s autouse
        ``_isolate_registries`` imports every runner into every test, so the registry is full in
        the suite and empty in a bare ``detail analysis attach``.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        run_dir_for, read_record, set_status, _discover = self._JOB_RUN_KINDS[kind]
        run_dir = run_dir_for(self._dir, run_id)

        def finalize(rdir: Path, status: str) -> None:
            # On success the runner writes the full record, with its own detail;
            # `reconciled_status` is what decides when the envelope overrides it.
            adopted = reconciled_status(read_record(rdir).get("status"), status)
            if adopted is not None:
                set_status(rdir, adopted, rundir.read_envelope(rdir).get("error"))

        return AttachedJobRun(run_dir, run_id, finalize)

    def attach_active_job_run(self, kind: str, run_id: str) -> AttachedJobRun | None:
        """An :class:`AttachedJobRun` when that run is still active, else ``None`` — so the
        GUI shows a live bar for a running one and a static record for a finished one. The
        single-job twin of :meth:`attach_active_run`."""
        if self.job_run_active(kind, run_id):
            return self.attach_job_run(kind, run_id)
        return None

    def job_run_active(self, kind: str, run_id: str) -> bool:
        """Whether that run's job is still working, whatever its record already claims.

        The record is usually enough. An **authored** run (§21) is the exception: its runner
        writes ``open`` before the work that ends the job, so for that stretch the record says
        "ready to interpret" while the job is still writing what interpreting reads. Believed,
        the run page built the workspace mid-fit and its tabs read an artifact that did not
        exist yet ("no behavior_output.yaml"). So an authored status is checked against the job
        envelope, which is the one place that knows the job has not finished.
        """
        run_dir_for, read_record, _set, _discover = self._JOB_RUN_KINDS[kind]
        run_dir = run_dir_for(self._dir, run_id)
        try:
            status = read_record(run_dir).get("status")
        except FileNotFoundError:
            return False
        if status in ACTIVE_STATUSES:
            return True
        if status not in AUTHORED_STATUSES:
            return False
        try:
            return rundir.read_status(run_dir).value in ACTIVE_STATUSES
        except (FileNotFoundError, KeyError, ValueError):
            return False

    def analysis_job_active(self, run_id: str) -> bool:
        """:meth:`job_run_active` for an analysis run."""
        return self.job_run_active("analysis", run_id)

    def attach_analysis_run(self, run_id: str) -> AttachedJobRun:
        """Reattach to an analysis run by id (§16)."""
        return self.attach_job_run("analysis", run_id)

    def attach_active_analysis_run(self, run_id: str) -> AttachedJobRun | None:
        """A handle for a still-active analysis run, else ``None`` (§16)."""
        return self.attach_active_job_run("analysis", run_id)

    def attach_dataset_run(self, dataset_id: str) -> AttachedJobRun:
        """Reattach to a dataset-export run by id (§12)."""
        return self.attach_job_run("dataset", dataset_id)

    def attach_active_dataset_run(self, dataset_id: str) -> AttachedJobRun | None:
        """A handle for a still-active dataset export, else ``None`` (§12)."""
        return self.attach_active_job_run("dataset", dataset_id)

    def attach_model_run(self, model_id: str) -> AttachedJobRun:
        """Reattach to a training run by id (§12)."""
        return self.attach_job_run("model", model_id)

    def attach_active_model_run(self, model_id: str) -> AttachedJobRun | None:
        """A handle for a still-active training run, else ``None`` (§12)."""
        return self.attach_active_job_run("model", model_id)

    def attach_export_run(self, run_id: str) -> AttachedJobRun:
        """Reattach to an export run by id (#354)."""
        return self.attach_job_run("export", run_id)

    def attach_active_export_run(self, run_id: str) -> AttachedJobRun | None:
        """A handle for a still-active export, else ``None`` (#354)."""
        return self.attach_active_job_run("export", run_id)

    # --- Analysis runs (§16) ------------------------------------------------ #

    def reconcile_job_runs(self) -> None:
        """Fold every still-active single-job run's real on-disk status back into its record
        on experiment open — the SIGKILL backstop for analysis, dataset, model and export
        runs alike (§12/§16), beside :meth:`reconcile_runs` for processing runs.

        Driven by ``_JOB_RUN_KINDS``, so a fifth kind inherits the backstop by adding a
        table entry. Without it a killed worker leaves a record reading ``running`` forever,
        which also cannot be deleted — every delete method refuses an active record.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        for kind, (_dir_for, _read, _set, discover) in self._JOB_RUN_KINDS.items():
            for run_id in discover(self._dir):
                self._reconcile_job_run(kind, run_id)

    def _reconcile_job_run(self, kind: str, run_id: str) -> bool:
        """Reconcile one single-job record against on-disk evidence; True if changed.

        The job envelope is the authority: ``reconciled_status`` says when its terminal
        status overrides the record's, and a record whose job is still nominally active but
        whose worker pid is gone is failed. A record with no envelope at all is left alone —
        that job was reserved but never submitted.

        Deliberately **not** gated on the record being ``pending`` / ``running``. An authored
        run (§21) writes ``open`` mid-job, so gating on an active record skipped exactly the
        runs that write a terminal-looking status before the work that justifies it — which is
        how a failed pose-clustering run went on reporting ``open`` through every surface.
        """
        run_dir_for, read_record, set_status, _discover = self._JOB_RUN_KINDS[kind]
        run_dir = run_dir_for(self._dir, run_id)
        try:
            record = read_record(run_dir)
        except (FileNotFoundError, ValueError, yaml.YAMLError):
            # **A record we cannot parse is one we cannot reconcile**, and that is all it is.
            # This runs on every experiment open, so an exception here does not fail a run — it
            # makes the *experiment* impossible to open, and there is no way back to the record
            # from inside the app to fix it. One malformed file, written by a half-finished
            # write, a hand-edit, or a version that stored something this one cannot read, took
            # the whole project down.
            #
            # The isolation holds the read and nothing else (CLAUDE.md): the reconciliation
            # below is a verdict on the run, and must not be skipped for any reason except that
            # there is no record to pass a verdict on.
            return False
        record_status = record.get("status")
        try:
            status = rundir.read_status(run_dir)
        except (FileNotFoundError, KeyError, ValueError):
            return False
        adopted = reconciled_status(record_status, status.value)
        if adopted is not None:
            set_status(run_dir, adopted, rundir.read_envelope(run_dir).get("error"))
            return True
        if status.value not in ACTIVE_STATUSES:
            return False
        # The envelope still says the job is running. That is only believable while the
        # record is one the job is still writing — active, or an authored run mid-projection.
        if record_status not in ACTIVE_STATUSES + AUTHORED_STATUSES:
            return False
        pid = rundir.read_pid(run_dir)
        if pid is not None and not pid_alive(pid):
            reason = "worker exited without completing"
            rundir.set_status(run_dir, JobStatus.FAILED, error=reason)
            set_status(run_dir, JobStatus.FAILED.value, reason)
            return True
        return False

    def reconcile_analysis_runs(self) -> None:
        """The analysis slice of :meth:`reconcile_job_runs` — kept because callers name it,
        and because a caller that only touched analysis should not sweep every kind."""
        import_runner_modules(_PIPELINE_RUNNERS)
        for run_id in analysis_store.discover_analyses(self._dir):
            self._reconcile_job_run("analysis", run_id)

    def reconcile_analysis_run(self, run_id: str) -> bool:
        """Reconcile one analysis run against its job envelope; ``True`` if the record moved.

        One ``job.json`` read, so a surface can afford it on the way into a run's page rather
        than trusting a record last reconciled when the experiment was opened. That gap is
        real: a run that fails *during* a session keeps its optimistic record until something
        sweeps, and the surface that shows it is exactly the one about to offer its results.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        return self._reconcile_job_run("analysis", run_id)

    @property
    def analysis_runs(self) -> list[AnalysisRunRef]:
        """Analysis runs in the experiment (scans ``analysis/``)."""
        refs = []
        for rec in dirscan.read_each(
            analysis_store.discover_analyses(self._dir),
            lambda r: analysis_store.read_analysis_yaml(
                analysis_store.analysis_dir(self._dir, r)
            ),
        ):
            refs.append(
                AnalysisRunRef(
                    id=rec["id"],
                    module_id=rec.get("module_id", ""),
                    recording_ids=rec.get("recording_ids", []),
                    status=rec.get("status", "unknown"),
                    created_at=rec.get("created_at", ""),
                    duration_s=rec.get("duration_s"),
                    name=str(rec.get("name") or ""),
                )
            )
        return refs

    # --- Export runs (#354) -------------------------------------------------- #

    @property
    def export_runs(self) -> list["ExportRunRef"]:
        """Every export run in the experiment (scans ``exports/``)."""
        refs = []
        for record in dirscan.read_each(
            export_runs_store.discover_exports(self._dir),
            lambda r: export_runs_store.read_export_yaml(
                export_runs_store.export_dir(self._dir, r)
            ),
        ):
            refs.append(
                ExportRunRef(
                    id=record["id"],
                    subject_kind=record.get("subject_kind", ""),
                    subject_id=record.get("subject_id", ""),
                    target_ids=list(record.get("target_ids") or []),
                    status=record.get("status", "unknown"),
                    created_at=record.get("created_at", ""),
                    folder=record.get("folder", ""),
                    destination=dict(record.get("destination") or {}),
                    duration_s=record.get("duration_s"),
                )
            )
        return refs

    def export_record(self, run_id: str) -> dict:
        """The full record for one export run (for detail views)."""
        return export_runs_store.read_export_yaml(
            export_runs_store.export_dir(self._dir, run_id)
        )

    def run_export(
        self,
        subject_kind: str,
        subject_id: str,
        target_ids: Sequence[str],
        parent: Path | str,
        *,
        scope=None,
        **options,
    ) -> "ExportRun":
        """Export ``target_ids`` from a subject into a new folder under ``parent``, as a run.

        **One request is one run**, however many targets it covers: they share a destination
        folder and a manifest because that is what was asked for once. The targets become the
        run's units.

        ``parent`` is resolved to a machine-local alias (`storage/roots.py`) before it reaches
        the job: everything in a job lands in the run record via ``runners.job_params``, a record
        is ``.yaml``, and an absolute path in one would make ``experiment.export`` refuse this
        experiment permanently (Tenet 6).

        The folder name is minted here rather than in the worker so a surface can say where the
        files are going before the run starts.

        Raises:
            ExportError: if a target is unknown, or cannot run for this subject.
            ValueError: if ``options["theme"]`` is not a chart theme (``light`` / ``dark``).
        """
        from detail.api import exporters
        from detail.pipeline.export_run import EXPORT_ROOTS_KEY
        from detail.storage import roots

        import_runner_modules(_PIPELINE_RUNNERS)
        subject = exporters.ExportSubject(subject_kind, self._dir, subject_id)
        targets = [exporters.target(subject_kind, t) for t in target_ids]
        for target in targets:
            reason = exporters.unavailable_reason(target, subject)
            if reason is not None:
                raise exporters.ExportError(
                    f"cannot export {target.label.lower()}: {reason}"
                )
        self._check_export_recordings(subject, scope)
        if options.get("theme") is not None:
            from detail.core.chart_theme import theme as chart_theme

            # Refused here, where the caller can see it, rather than in the worker minutes later
            # as a failed run.
            chart_theme(options["theme"])

        parent = Path(parent).expanduser()
        parent.mkdir(parents=True, exist_ok=True)
        alias = roots.resolve_or_register(EXPORT_ROOTS_KEY, parent)
        rel = parent.resolve().relative_to(
            roots.root(EXPORT_ROOTS_KEY, alias).resolve()
        )

        run_id = export_runs_store.reserve_export_id(self._dir)
        job = ExportJob(
            job_id=run_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            target_ids=[t.id for t in targets],
            destination={"root": alias, "rel": rel.as_posix()},
            folder=exporters.folder_name(subject),
            created_at=utc_timestamp(),
            scope=_scope_dict(scope),
            options=dict(options),
        )
        # Written before submit so the run is discoverable while it runs, and so the runner's
        # completion write is always last — the worker can finish before this returns.
        rdir = export_runs_store.export_dir(self._dir, run_id)
        export_runs_store.write_export_yaml(
            rdir,
            {
                "id": run_id,
                "subject_kind": subject_kind,
                "subject_id": subject_id,
                "target_ids": [t.id for t in targets],
                "folder": job.folder,
                "destination": dict(job.destination),
                "params": job_params(job),
                "status": "running",
                "created_at": job.created_at,
            },
        )
        executor = LocalExecutor(self._dir, runner_modules=_PIPELINE_RUNNERS)
        executor.submit(job)
        return ExportRun(ExecutorRunHandle(executor, run_id), self, run_id)

    def _check_export_recordings(self, subject, scope) -> None:
        """Refuse a recordings scope that names something this export does not span.

        Refused here, where the caller can see it, rather than writing an empty folder: a
        misspelt id on a large run would otherwise read as "nothing to write". Only a run
        or analysis run has a set of recordings to choose between; for anything else the one
        thing a scope may name is the subject's own recording.
        """
        from detail.api import exporters

        if scope is None or not scope.recording_ids:
            return
        if subject.kind in ("run", "analysis_run"):
            known = set(exporters.recordings_of(subject))
            unknown = [r for r in scope.recording_ids if r not in known]
            if unknown:
                raise exporters.ExportError(
                    f"{', '.join(unknown)} is not one of the recordings in this "
                    f"{subject.kind.replace('_', ' ')} ({len(known)}: "
                    f"{', '.join(sorted(known)[:6])}{', ...' if len(known) > 6 else ''})"
                )
            return
        if subject.kind == "recording" and set(scope.recording_ids) <= {subject.id}:
            return
        raise exporters.ExportError(
            f"a {subject.kind.replace('_', ' ')} export has no recordings to choose between; "
            "choose recordings when exporting a run or an analysis run"
        )

    def estimate_export(
        self,
        subject_kind: str,
        subject_id: str,
        target_ids: Sequence[str],
        parent: Path | str | None = None,
        *,
        scope=None,
        **options,
    ):
        """How big an export would be and how long it would take, without writing it.

        The same arguments as :meth:`run_export`, and the answer is what the Export dialog and
        ``detail export run`` show before they start: an :class:`exporters.ExportEstimate`, one
        entry per target plus the free space on ``parent``'s drive. A video target times a few
        dozen frames on this machine to answer (about a second); everything else reads sizes and
        row counts from disk.

        Raises:
            ExportError: if a target is unknown, or ``scope`` names a recording the export does
                not span.
        """
        from detail.api import exporters

        subject = exporters.ExportSubject(subject_kind, self._dir, subject_id)
        targets = [exporters.target(subject_kind, t) for t in target_ids]
        scope = scope or exporters.ExportScope()
        self._check_export_recordings(subject, scope)
        return exporters.estimate(
            subject, targets, scope=scope, parent=parent, **options
        )

    def cancel_export(self, run_id: str) -> None:
        """Stop an active export run. No-op once terminal.

        Raises:
            FileNotFoundError: If the run doesn't exist.
        """
        rdir = export_runs_store.export_dir(self._dir, run_id)
        record = export_runs_store.read_export_yaml(rdir)
        if record.get("status") not in ACTIVE_STATUSES:
            return
        cancel_run_dir(rdir)
        export_runs_store.set_export_status(rdir, rundir.read_status(rdir).value)

    def delete_export(self, run_id: str) -> None:
        """Delete an export run's record. Blocked while it is still active.

        Deletes the record, never the exported files: those are in the user's own folder,
        outside the experiment, and forgetting a run is not a request to delete its output.
        """
        record = export_runs_store.read_export_yaml(
            export_runs_store.export_dir(self._dir, run_id)
        )
        if record.get("status") in ACTIVE_STATUSES:
            raise ValueError(
                f"export run {run_id!r} is still {record.get('status')}; cancel it first"
            )
        export_runs_store.delete_export(self._dir, run_id)

    def analysis_record(self, run_id: str) -> dict:
        """The full ``analysis.yaml`` for one analysis run (for detail views)."""
        return analysis_store.read_analysis_yaml(
            analysis_store.analysis_dir(self._dir, run_id)
        )

    def analysis_run_dir(self, run_id: str) -> Path:
        """The on-disk output directory for an analysis run (``analysis/<run_id>/``)
        — for opening it in a file manager."""
        return analysis_store.analysis_dir(self._dir, run_id)

    def location_plots(self, run_id: str):
        """Draw a completed location-export run's paths, occupancy and speed — an
        ``api.location_plots.LocationPlots`` over the run's own CSVs (§16 "Location plots").

        Raises:
            ValueError: The run is not a location export, or has no completed recording.
        """
        from detail.api.location_plots import LocationPlots

        return LocationPlots(self, run_id)

    def analysis_info(self, run_id: str) -> run_info.RunInfo:
        """Every YAML file an analysis run keeps (its meta-information), as on disk.

        ``analysis.yaml`` first, then the module's own ``<module>.yaml``, then the rest
        (``machine.yaml``, ``resources.yaml``, …) alphabetically.

        Raises:
            FileNotFoundError: No such run.
        """
        run_dir = self.analysis_run_dir(run_id)
        module_id = self.analysis_record(run_id).get("module_id")
        first = (analysis_store.ANALYSIS_FILENAME,) + (
            (f"{module_id}.yaml",) if module_id else ()
        )
        return run_info.read_run_info(run_dir, run_id, first=first)

    def export_analysis_info(self, run_id: str, path: Path | str) -> Path:
        """Write an analysis run's meta-information (:meth:`analysis_info`) to one zip.

        Returns:
            The zip's path.

        Raises:
            FileNotFoundError: No such run.
            ValueError: The run has no YAML files.
        """
        return run_info.export_run_info(self.analysis_info(run_id), path)

    def run_analysis(self, module_id: str, config: dict) -> AnalysisRun:
        """Launch a experiment-scoped analysis run (§16): one module over a selection
        of recordings, producing artifacts in ``analysis/<run_id>/``. A single
        executor job (not a per-recording fan-out). Returns a run handle.

        ``config["compute"]`` (a ``ComputeTarget`` or its dict) sends the job to a remote
        target, for a module that declares ``remote_capable`` (§6 "Remote analysis").

        Raises:
            ValueError: If the module is unknown, config validation fails, or a remote
                ``compute`` is asked of a module that cannot run remotely.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        module = ANALYSIS_MODULES.get(module_id)
        if module is None:
            raise ValueError(f"unknown analysis module: {module_id!r}")
        remote = _remote_compute(config.get("compute"))
        if remote is not None and not module.remote_capable:
            raise ValueError(
                f"{module_id} cannot run on a remote target ({remote.remote_target}); "
                "run it on this machine (leave compute unset)"
            )
        config = module.validate(
            self._dir, none_means_default(module.config_model, config)
        )
        validated_remote = _remote_compute(config.get("compute"))
        if remote is not None and validated_remote is None:
            # A remote-capable module whose validate() lost the target would run here.
            raise ValueError(
                f"{module_id}'s validated config dropped its remote compute target"
            )
        run_id = analysis_store.reserve_analysis_id(self._dir)
        job = module.build_job(run_id, config)
        # An initial record so the run is discoverable while it runs, written
        # *before* submit so the runner's completion write (full record) is always
        # last — never clobbered by this one (the worker can finish fast).
        adir = analysis_store.analysis_dir(self._dir, run_id)
        adir.mkdir(parents=True, exist_ok=True)
        analysis_store.write_analysis_yaml(
            adir,
            {
                "id": run_id,
                "module_id": module_id,
                "recording_ids": config.get("source_recording_ids", []),
                "status": "running",
                "created_at": utc_timestamp(),
                **(
                    {"compute": validated_remote.to_dict()}
                    if validated_remote is not None
                    else {}
                ),
            },
        )
        executor = executor_for(self._dir, validated_remote, _PIPELINE_RUNNERS)
        executor.submit(job)  # run_dir = analysis/<run_id>/; writes job.json there
        return AnalysisRun(ExecutorRunHandle(executor, run_id), self, run_id)

    def run_comparison(self, config, *, progress=None, stop_check=None):
        """Run a §24 comparison — group statistics across recording metadata — and save it."""
        from detail.api.comparison import run_comparison

        return run_comparison(self, config, progress=progress, stop_check=stop_check)

    def get_comparison(self, comparison_id: str):
        """Handle for an existing comparison in this experiment."""
        from detail.api.comparison import Comparison

        return Comparison(self, comparison_id)

    def measures(self, run_id: str):
        """The §23 measures of one analysis run's behavior artifact (§19)."""
        from detail.api.measures import Measures

        return Measures(self, run_id)

    @property
    def comparisons(self) -> list:
        """All saved comparisons in this experiment."""
        from detail.api.comparison import Comparison
        from detail.storage import comparisons as comparisons_store

        return [
            Comparison(self, cid)
            for cid in comparisons_store.discover_comparisons(self._dir)
        ]

    #: Which handle an authored module's runs are reached through (§21). One table rather than a
    #: copy per caller: the GUI, the CLI and the comparison path all need it, and three parallel
    #: dicts is how the fourth authored module gets forgotten by one of them.
    _AUTHORED_RUN_ACCESSORS = {
        "temporal_clustering": "temporal_clustering_run",
        "query": "query_run",
    }

    def authored_run(self, run_id: str):
        """The §21 handle for an authored run, or ``None`` if this run is not one.

        A computed run (a location export, an eval) has no authored state and pins nothing, so it
        has no handle of this kind — the caller shows no snapshot surface rather than an empty one.
        """
        try:
            record = self.analysis_record(run_id)
        except (FileNotFoundError, KeyError):
            return None
        accessor = self._AUTHORED_RUN_ACCESSORS.get(record.get("module_id"))
        if accessor is None:
            return None
        try:
            return getattr(self, accessor)(run_id)
        except (KeyError, ValueError, FileNotFoundError):
            return None

    # --- cluster profiles (§22 "Cluster profile") ------------------------------ #
    def profile_cluster(
        self,
        run_id: str,
        node: int,
        config: dict | None = None,
        *,
        save: bool = True,
        progress: Callable[[str, float, str], None] | None = None,
    ):
        """Which measures set ``node`` of clustering run ``run_id`` apart, and how far to
        trust that — a ``ClusterProfile`` (``api/cluster_profile.py``).

        ``config`` is a ``ClusterProfileConfig`` field dict (comparison — the rest, the
        sibling, or another cluster by id — explainer, time budget or a fixed count, margin,
        whether position and facing are left out, seed). Saved beside the run unless
        ``save=False``; a comparison with another cluster in its own folder, so the node's
        own profile is kept. ``progress(stage, fraction, detail)`` hears sampling,
        measuring, fitting and context.

        Raises:
            ValueError: Not a clustering run, the root compared with its sibling, another
                cluster that holds this one or sits inside it, or an explainer named whose
                modules are missing.
            FileNotFoundError: The run has no node cache yet.
        """
        from detail.api import cluster_profile

        run = self._clustering_run(run_id)
        profile = cluster_profile.profile_cluster(
            self, run, node, config, progress=progress
        )
        if save:
            cluster_profile.save_profile(self.analysis_run_dir(run_id), profile)
        return profile

    def cluster_profile(self, run_id: str, node: int, against: int | None = None):
        """The saved profile of ``node`` — or, with ``against``, its saved comparison with
        that other cluster — or ``None`` when it was never made."""
        from detail.api import cluster_profile

        return cluster_profile.load_profile(
            self.analysis_run_dir(run_id), node, against
        )

    def _clustering_run(self, run_id: str):
        run = self.authored_run(run_id)
        if run is None or not hasattr(run, "node_view"):
            raise ValueError(
                f"{run_id} is not a temporal-clustering run of this experiment"
            )
        return run

    def comparisons_using(self, run_id: str) -> dict[str, list[str]]:
        """``{snapshot_id: [comparison_id, ...]}`` for one analysis run's pinned artifacts.

        **Derived, never stored.** A snapshot records why it was taken but not what consumed it:
        an export has no stable identifier to keep (its path lies outside the experiment, and an
        absolute path in an experiment file is a hard rule), and a comparison id does not exist
        until after the pin. So the link is read back off the consumers, which is also the §21 rule
        that a fact checkable from the store gets no second writer.

        One pass over ``comparisons/``, which holds tens of records in a real experiment rather
        than thousands. Records written before ``source_snapshot`` existed carry none and are
        skipped rather than guessed at.
        """
        from detail.storage import comparisons as comparisons_store

        out: dict[str, list[str]] = {}
        for cid in comparisons_store.discover_comparisons(self._dir):
            try:
                record = comparisons_store.read_comparison_yaml(
                    comparisons_store.comparison_dir(self._dir, cid)
                )
            except dirscan.UNREADABLE_RECORD_ERRORS:
                continue  # a half-written or foreign folder is not a comparison
            if not isinstance(record, dict) or record.get("source_run_id") != run_id:
                continue
            snapshot = record.get("source_snapshot")
            if snapshot:
                out.setdefault(str(snapshot), []).append(cid)
        return out

    def delete_comparison(self, comparison_id: str) -> None:
        """Delete a comparison and everything it wrote, exports included.

        Raises:
            FileNotFoundError: no comparison with that id.
        """
        from detail.storage import comparisons as comparisons_store

        if comparison_id not in comparisons_store.discover_comparisons(self._dir):
            raise FileNotFoundError(
                f"no comparison {comparison_id!r} in {self._dir}; "
                f"have {comparisons_store.discover_comparisons(self._dir)}"
            )
        comparisons_store.delete_comparison(self._dir, comparison_id)

    def query_run(self, run_id: str):
        """Reopen an existing §20 Query run as a ``QueryRun`` handle (no active search job) —
        for calibrating a cutoff, naming a behavior, or exporting the artifact.

        The way a searched run is reached to be calibrated: the search leaves the run ``open``
        rather than ``complete``, and the whole of calibration happens through this handle.
        """
        from detail.api.query import QueryRun

        return QueryRun(None, self, run_id)

    def temporal_clustering_run(self, run_id: str):
        """Reopen an existing §22 temporal-clustering run as a ``TemporalClusteringRun`` handle
        (no active fit job) — for choosing a granularity, naming clusters, or exporting.

        The way a fitted run is reached to be interpreted: the fit leaves the run ``open`` rather
        than ``complete``, and the whole of interpretation happens through this handle.
        """
        from detail.api.temporal_clustering import TemporalClusteringRun

        return TemporalClusteringRun(None, self, run_id)

    def cancel_analysis(self, run_id: str) -> None:
        """Stop an active analysis run (kill its worker), then record its real
        on-disk status (``cancelled`` if killed, ``complete`` if it finished). No-op
        once terminal. Raises FileNotFoundError if the run doesn't exist."""
        adir = analysis_store.analysis_dir(self._dir, run_id)
        record = analysis_store.read_analysis_yaml(adir)  # FileNotFoundError if absent
        if record.get("status") not in ACTIVE_STATUSES:
            return
        cancel_run_dir(adir)
        analysis_store.set_analysis_status(adir, rundir.read_status(adir).value)

    def delete_analysis(self, run_id: str) -> None:
        """Delete an analysis run (its ``analysis/<run_id>/`` directory).

        A *completed* analysis run is self-contained (it copied the derived values
        into its own CSVs), so it never blocks deleting its source runs — the
        analogue of the dataset/model lineage-dangle. Blocked only while the run is
        still active.

        Raises:
            FileNotFoundError: If no such analysis run exists.
            ValueError: If the run is still active.
        """
        record = analysis_store.read_analysis_yaml(  # FileNotFoundError if absent
            analysis_store.analysis_dir(self._dir, run_id)
        )
        if record.get("status") in ACTIVE_STATUSES:
            raise ValueError(
                f"analysis run {run_id!r} is still active — cancel it first"
            )
        analysis_store.delete_analysis(self._dir, run_id)

    def duplicate_analysis(
        self,
        run_id: str,
        *,
        name: str | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> str:
        """Copy analysis run ``run_id`` whole — its fit, tree, node cache, measures and
        profiles — to a new ``an_NN``, leaving the original untouched; the new id.

        The copy is named ``name`` (``"<original name or id> (copy)"`` by default) and records
        ``duplicated_from``. Ids inside its records are rewritten to the new one; a cancel
        request, a lock or a half-written file is not copied (``api/duplicate_analysis.py``).
        ``progress(stage, fraction, detail)`` hears the copy by bytes; an
        ``OperationCancelled`` raised from it removes the partial copy.

        Raises:
            FileNotFoundError: No such analysis run.
            ValueError: The run is still running.
        """
        from detail.api import duplicate_analysis

        return duplicate_analysis.duplicate_analysis(
            self, run_id, name=name, progress=progress
        )

    def profile_describe_clusters(
        self, run_id: str, config: dict | None = None
    ) -> AnalysisRun:
        """Profile many clusters of clustering run ``run_id`` as one recorded job (§22
        "Profile clusters"): a ``profile_describe`` analysis run (the id it has always had,
        so earlier runs still open).

        ``config`` is a ``ProfileDescribeConfig`` field dict without ``source_run_id``: which
        clusters (``floor`` by default: at or above the run's minimum cluster time), the profile
        settings and ``skip_existing``. Each cluster's profile is saved beside ``run_id`` as the
        single-cluster call saves it; the new run's record lists every cluster's status.
        Returns the run handle (progress, wait, cancel).

        ``config["compute"]``, a remote ``ComputeTarget`` (or its dict), runs the job on that
        cluster; its results are placed beside ``run_id`` once they are copied back.

        Raises:
            ValueError: Not a clustering run, no node cache, or nothing to run.
        """
        return self.run_analysis(
            "profile_describe", {**(config or {}), "source_run_id": run_id}
        )

    def find_lookalikes(self, run_id: str, config: dict | None = None) -> AnalysisRun:
        """Compare pairs of clustering run ``run_id``'s clusters on body measures as one
        recorded job (§22 "Look-alike clusters"): a ``cluster_lookalikes`` analysis run.

        ``config`` is a ``ClusterLookalikesConfig`` field dict without ``source_run_id``:
        ``pairs`` (``"all"``, the default: every pair of branch tops; ``"labelled"``: each
        labelled branch top against every other), the profile settings, ``skip_existing``,
        and ``compute`` (a ``ComputeTarget`` dict) to run it on a registered remote target.
        Every pair is compute intensive: :meth:`count_lookalike_pairs` says how many pairs
        each mode compares, and about how long, before starting. Each pair's comparison is
        placed beside ``run_id`` — after the copy-back, for a remote run — where
        ``lookalike_groups()`` on its run handle reads the groups. Returns the run handle
        (progress, wait, cancel).

        Raises:
            ValueError: Not a clustering run, no node cache, no pair to compare,
                ``labelled`` with no labelled cluster, or a remote target not registered
                here.
        """
        return self.run_analysis(
            "cluster_lookalikes", {**(config or {}), "source_run_id": run_id}
        )

    def count_lookalike_pairs(self, run_id: str, config: dict | None = None):
        """How many pairs :meth:`find_lookalikes` compares on run ``run_id`` in each mode,
        and about how long on this machine — a ``cluster_lookalikes.PairCounts`` (``.counts``
        ``{"all": n, "labelled": n}``, ``.seconds(mode)``, ``.warning()`` the text every
        surface shows before comparing every pair). ``config`` as for :meth:`find_lookalikes`;
        its profile settings decide which pairs are already compared (skipped). Reads the
        node cache and the saved comparisons: never on a UI thread.

        Raises:
            ValueError: Not a clustering run, or it has no node cache yet.
        """
        from detail.api import cluster_lookalikes as lookalikes
        from detail.configs.cluster_lookalikes import (
            RETIRED_FIELDS,
            ClusterLookalikesConfig,
        )

        given = {**(config or {}), "source_run_id": run_id}
        for name in ("compute", "pair_list", "source_recording_ids", *RETIRED_FIELDS):
            given.pop(name, None)  # where it runs does not change what it compares
        model = ClusterLookalikesConfig.model_validate(given)
        run = self._clustering_run(run_id)
        try:
            view = run.node_view()
        except FileNotFoundError:
            raise ValueError(
                f"{run_id} has no node cache yet; open it once (or rebuild its clips) "
                "before comparing its clusters"
            ) from None
        return lookalikes.count_pairs(
            view,
            dict(run.state().node_labels),
            lookalikes.read_evidence(self.analysis_run_dir(run_id)),
            model,
        )

    def estimate_profile_describe(self, run_id: str, config: dict | None = None):
        """How long :meth:`profile_describe_clusters` should take on this machine — a
        ``profile_describe.BatchEstimate`` (``.seconds``; ``.text()`` for a person), counting
        only the clusters not profiled yet. Reads files only.
        """
        from detail.api import profile_describe
        from detail.configs.profile_describe import ProfileDescribeConfig

        given = {**(config or {}), "source_run_id": run_id}
        given.pop("compute", None)  # where it runs does not change what is left to do
        return profile_describe.estimate(
            self, run_id, ProfileDescribeConfig.model_validate(given)
        )

    # --- Credentials and model install (Section 5) -------------------------- #
    def set_hf_token(self, token: str) -> str:
        """Validate and store the Hugging Face token; return the account name.

        Writes the machine-global ``~/.config/detail/credentials.yaml`` (not
        experiment state), exposed here as the single API boundary (Tenet 2).
        """
        return credentials.set_hf_token(token)

    # Models are machine-global (§5) — these delegate to the experiment-free
    # ``api.models`` catalog (the GUI's top-level Models tab uses it directly).
    def check_model_access(
        self, model_id: str, variant: str | None = None
    ) -> ModelReadiness:
        """Proactive readiness for a model — drives 'set up access' UI states."""
        return api_models.model_readiness(model_id, variant)

    def install_model(
        self,
        model_id: str,
        progress_cb: Callable[[InstallProgress], None] | None = None,
        *,
        variant: str | None = None,
    ) -> None:
        """Ensure a model's weights are present; raises ModelNotReadyError if blocked.

        ``variant`` picks the size — each is a separate download (§5)."""
        api_models.install_model(model_id, progress_cb, variant=variant)

    def import_model(
        self,
        weights_path: Path | str,
        *,
        family: str | None = None,
        skeleton=None,
        name: str | None = None,
        made_by: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        preview: Path | str | None = None,
    ) -> str:
        """Import an external checkpoint into the machine-global catalog — usable in *any*
        experiment. A pose checkpoint lands as a keypoint model and needs ``skeleton`` (a
        preset, library id or skeleton YAML) that fits it; see
        :func:`detail.api.models.import_model`. Optional ``made_by`` / ``description`` /
        ``notes`` / ``preview`` are stored as catalog metadata (§12). Returns the assigned
        id."""
        return api_models.import_model(
            weights_path,
            family=family,
            skeleton=skeleton,
            name=name,
            made_by=made_by,
            description=description,
            notes=notes,
            preview=preview,
        )

    def import_keypoint_model(
        self,
        path: Path | str,
        *,
        name: str | None = None,
        skeleton=None,
        made_by: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        preview: Path | str | None = None,
    ) -> str:
        """Import a keypoint model exported to ONNX (``sleap-nn export ... --format onnx``)
        into the machine-global catalog (§26), where any experiment's keypoint detect can
        name it. See ``api.models.import_keypoint_model``. Returns the assigned id."""
        return api_models.import_keypoint_model(
            path,
            name=name,
            skeleton=skeleton,
            made_by=made_by,
            description=description,
            notes=notes,
            preview=preview,
        )

    @property
    def imported_models(self) -> list[ImportedModelRef]:
        """The machine-global imported models, every role (shared across all experiments);
        each ref says its ``role`` — a detection picker filters on it."""
        return api_models.imported_models(role=None)

    def remove_imported_model(self, model_id: str) -> None:
        """Delete a machine-global imported model (raises if absent)."""
        api_models.remove_imported_model(model_id)
