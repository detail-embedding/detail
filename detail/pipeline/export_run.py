"""The export runner — one user request, carried out as a durable job (#354).

Every export goes through here. There is no inline path and no `durable` flag, because "long"
was never a property of a *target*: `tracks` takes seconds on a test fixture and many
minutes on a real recording, same target and same code. A per-target flag would be wrong
the first time somebody exported `instances` across a large cohort. The price of routing
everything through a worker is about a second, to spawn it and register the runner
modules, and what it buys is one mechanism instead of two, with cancel, reattach, an
activity log and a survivable app close thrown in.

The body is `exporters.write_export`, unchanged and still callable directly: the function is the
work, the run is the managed way to invoke it, exactly as a module's `runner` relates to
`Experiment.run_analysis`.

**Units are targets.** The event field names are the §16 analysis shape because that is the one
`gui/workers.py::fit_progress_fraction` already renders — adding a fourth event→render mapping
for this would be the thing DESIGN_PLAN §6 forbids, and the third (`StageProgressWorker._emit`,
which drives the inline path) already has no ETA and no test.
"""

from __future__ import annotations

import time
from pathlib import Path

from dataclasses import dataclass
from typing import ClassVar

from detail.core.dataclasses import ExportJob, register_progress_event_type
from detail.core.progress import PhaseClock, ThrottledProgress
from detail.executor import rundir
from detail.executor.runners import register_runner
from detail.storage import experiment as experiment_store
from detail.storage import export_runs as export_store

#: The ``storage.yaml`` section export destinations are registered under (`storage/roots.py`).
#: An export's folder is chosen by the user and lives outside the experiment, so the job carries
#: an alias into this section rather than a path — see `ExportJob.destination` for what an
#: absolute one would break.
EXPORT_ROOTS_KEY = "export_roots"


@dataclass
class ExportProgressEvent:
    """Progress for an export run, in the §16 analysis event's field names.

    Those names are a **contract with `gui/workers.py::fit_progress_fraction`**, which folds
    them as ``(recordings_done + frames_done/frames_total) / recordings_total``. Reusing them is
    what lets an export render through the one mapping that already exists and has a test,
    rather than becoming a fourth.

    The units are *targets*, not recordings — `LiveRunPanel.set_units` takes its heading as a
    parameter for exactly this reason. ``frames_*`` is then position within the running target.
    """

    recordings_done: int
    recordings_total: int
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None
    #: What a surface should call these units. Without it every renderer that speaks about a
    #: recording-keyed event says "recording", and a two-recording masks export reads
    #: "recording 1/1" — a wrong answer rather than an ugly one.
    unit_label: str = "target"
    #: The fraction gives every target an equal share, so an ETA extrapolated from it is the
    #: whole run's guess, not the time left: a surface shows ``eta_s`` or nothing. A class
    #: attribute, so it is not serialized and is true of every event read back.
    eta_from_runner_only: ClassVar[bool] = True


register_progress_event_type("export", ExportProgressEvent)


def run_dir_for(experiment_root, job: ExportJob) -> Path:
    """``exports/<run_id>/`` — an experiment-level run dir holding the record and the envelope.

    Not where the exported files go: those land in the folder the user chose, outside the
    experiment. This directory is the run's bookkeeping and nothing else.
    """
    return experiment_store.exports_dir(experiment_root) / job.job_id


def _reporter(progress_cb, n_targets: int, clock=time.monotonic):
    """`write_export`'s ``(stage, fraction, detail)`` in the shape the GUI already renders.

    ``fraction`` is the whole export's, already folded across targets by `_slice_progress`, so
    the outer pair is *targets* and the inner is position within the one running. Note what that
    fold assumes and this cannot fix: every target is given an equal share of the bar, and
    `clips` beside `instances` is many minutes beside a fraction of a second. The unit rows
    carry the truth; the aggregate is coarse, and saying so here is better than pretending it is
    linear.

    **So the ETA is not extrapolated from it.** It is the running target's own remaining
    (`PhaseClock`, one phase per target): an experiment bundle beside its metadata CSV was given
    half the bar, and a whole-run extrapolation read 20 minutes with 7 to go (#916).

    A change of stage is forced past the throttle, so the label never lags the work.
    """
    emit = ThrottledProgress(progress_cb)
    started = clock()
    phases = PhaseClock(started)
    total = n_targets or 1
    last_stage = [""]

    def report(stage: str, fraction: float, detail: str = "") -> None:
        now = clock()
        done = min(int(fraction * total), total - 1)
        within = fraction * total - done
        changed = stage != last_stage[0]
        last_stage[0] = stage
        emit(
            ExportProgressEvent(
                recordings_done=done,
                recordings_total=total,
                frames_done=int(within * 1000),
                frames_total=1000,
                elapsed_s=now - started,
                eta_s=phases.eta(str(done), within, now),
                message=f"{stage} · {detail}" if detail else stage,
            ),
            force=changed or detail == "done",
        )

    return report


@register_runner("export", run_dir_for=run_dir_for)
def run(job: ExportJob, run_dir, progress_cb, stop_event) -> None:
    from detail.api import export_targets  # noqa: F401  (registers the catalogue)
    from detail.api import exporters
    from detail.storage import roots

    run_dir = Path(run_dir)
    experiment_dir = run_dir.parent.parent  # <experiment>/exports/<run_id>/
    subject = exporters.ExportSubject(job.subject_kind, experiment_dir, job.subject_id)
    targets = [exporters.target(job.subject_kind, t) for t in job.target_ids]
    parent = roots.resolve_ref(
        EXPORT_ROOTS_KEY,
        job.destination,
        hint="re-run the export and choose the folder again",
    )

    result = exporters.write_export(
        subject,
        targets,
        parent,
        scope=exporters.ExportScope(**job.scope) if job.scope else None,
        progress=_reporter(progress_cb, len(targets)),
        stop_check=stop_event.is_set,
        folder=job.folder,
        **job.options,
    )

    record = export_store.read_export_yaml(run_dir)
    record.update(
        # `job.folder` is minted at submit time and can be renamed by `new_dir` inside
        # `write_export` if it collides with a folder already on disk (#475) — write back what
        # was actually created so `exporters.destination_of` never points at the wrong folder.
        folder=result.dest.name,
        status=result.status,
        files=list(result.files),
        outcomes=[o.to_dict() for o in result.outcomes],
    )
    if result.empty_reason:
        record["empty"] = result.empty_reason
    if result.status == "failed":
        record["error"] = "; ".join(
            f"{o.target_id}: {o.error}" for o in result.failures
        )
    export_store.write_export_yaml(run_dir, record)
    # The envelope's own terminal status is the worker's to write, and it settles `partial`
    # from the cancel flag. A run that finished with a failed target reports that here instead.
    if result.status != "complete" and not rundir.cancel_requested(run_dir):
        rundir.update_payload(run_dir, export_status=result.status)
