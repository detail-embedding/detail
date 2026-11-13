"""Local shepherd subprocess for a remote (Slurm) job (§6).

Invoked as ``python -m detail.executor.remote_shepherd <run_dir> <target_name>
<experiment_root>`` by :class:`detail.executor.remote.RemoteExecutor`. To every local
liveness/reattach check this is an ordinary worker: a plain subprocess with a pid that
is the only writer of this run's local ``job.json``/``progress.json`` once it starts.
It never runs pipeline code itself — it:

1. Mirrors the experiment directory to the cluster (same relative layout, so the
   remote worker resolves every path exactly like a local one — Tenet: no absolute
   paths in experiment files is what makes this a straight mirror).
2. Checks — over ssh, in the job's own module/conda environment — that a job
   carrying a model (detection, today) has that model installed *on the cluster*.
   A local ``DetectionModule.validate()`` deliberately skips this for a remote job
   (it is not this machine's job to have the weights); this is where that
   responsibility actually lands, before an sbatch submission burns a queue slot
   and a GPU allocation on a job that can only fail.
3. Rewrites the job's ``compute`` to the concrete on-cluster target
   (``backend="cuda"``) and writes the *remote* ``job.json`` — the remote worker
   never sees ``"remote"`` as a backend.
4. Submits an sbatch script that runs ``python -m detail.executor.worker
   <remote_run_dir>`` — the same, unmodified worker entrypoint a local job uses.
5. Polls the scheduler + mirrors the remote ``progress.json``/``job.json`` into the
   local envelope until the remote job is terminal.
6. Syncs the job's own run folder back, then writes the final local status — in that order,
   so a reader never sees "complete" before the outputs are actually there. The sync leaves
   the local ``job.json`` alone (#742): the cluster's copy already says the job is over, and it
   used to arrive first, so the coordinator's reaper killed the copy ten seconds in and runs
   were recorded complete without their data. The worker's own fields (a stage's results,
   its clock) are merged in after the data (``_merge_worker_envelope``), and the status is
   the last write.

The upload (step 1) still syncs the **whole experiment directory**, not just what a given job
depends on — simple and correct for any job type, at the cost of transfer time on a large
experiment. The copy *back* is scoped to the run folder: it must be, because a job only ever
writes there, and because the experiment on this machine keeps changing while the job runs (the
app is open, another run is going), so a whole-tree copy down puts the cluster's stale snapshot
over live local state. Scoping the upload to a job's actual dependencies is a documented
follow-up, not something this shepherd attempts.

Every one of those steps before the remote worker's first event is a **named phase** the run
publishes (``_TransferProgress``): uploading experiment files, uploading videos (weighed by
bytes, with a measured ETA), checking the model, submitting, waiting in the queue, syncing results
back. The video upload alone can run for an hour on a large library, and without this a healthy
run was indistinguishable from a hung one. The same lines go to ``shepherd.log``.
"""

import dataclasses
import json
import os
import posixpath
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from detail.core.dataclasses import (
    ComputeTarget,
    DetectionProgressEvent,
    EmbeddingProgressEvent,
    JobStatus,
    LabelingJob,
    LabelingProgressEvent,
    ModelReadiness,
    TrackingProgressEvent,
    TrainingProgressEvent,
    job_to_dict,
    job_type_tag,
)
from detail.core import atomic
from detail.core import progress as core_progress
from detail.core.progress import PhaseClock, ThrottledProgress, format_duration
from detail.core.protocols import ModelNotReadyError
from detail.core.utils import utc_timestamp
from detail.executor import rundir, ssh_transport, video_staging
from detail.executor.remote_python import remote_python
from detail.executor.runners import (
    ANALYSIS_MODULES,
    JOB_RUNNERS,
    RUNNER_MODULES_ENV,
    import_runner_modules,
)
from detail.executor.schedulers.base import TERMINAL_STATES, JobState
from detail.executor.schedulers.slurm import SlurmScheduler, render_sbatch_script
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import remote_targets as remote_targets_store
from detail.storage import video_roots

#: Remote round trips (an ssh call each) are expensive compared to a local file read —
#: no need for the in-process 0.5s live-bar cadence.
POLL_INTERVAL_S = 5.0

_SCHEDULERS = {"slurm": SlurmScheduler()}
_WORKER_TERMINAL_VALUES = {
    JobStatus.COMPLETE.value,
    JobStatus.FAILED.value,
    JobStatus.PARTIAL.value,
    JobStatus.CANCELLED.value,
}
_SCHEDULER_TO_JOB_STATUS = {
    JobState.CANCELLED: JobStatus.CANCELLED,
    JobState.FAILED: JobStatus.FAILED,
    JobState.COMPLETED: JobStatus.FAILED,  # scheduler-terminal with no worker status: see below
}


def _phase_event(job, *, message: str, elapsed_s: float, eta_s: float | None):
    """A zero-work event of ``job``'s own family that carries only a phase ``message`` and ETA.

    Every surface already renders ``message`` and ``eta_s`` off each family's event, so the
    stretch before the job exists on the cluster (uploading, checking, queueing) reuses them
    rather than adding a type each surface would have to learn. The fraction stays 0: nothing the
    *stage* does has started, and a fraction invented here would run its bar to 100% and drop it
    back to 0 when the real work begins. ``None`` for a job whose family has no such event.
    """
    try:
        tag = job_type_tag(job)
    except TypeError:
        return None
    common = {"elapsed_s": elapsed_s, "eta_s": eta_s, "message": message}
    if tag == "detection":
        return DetectionProgressEvent(0, 0, 0, 0, **common)
    if tag in ("labeling", "dataset_export"):
        return LabelingProgressEvent(0, 0, 0, 0, **common)
    if tag == "training":
        return TrainingProgressEvent(0, 0, {}, **common)
    if tag == "tracking":
        return TrackingProgressEvent(0, 0, 0, 0, 0, **common)
    if tag == "embedding":
        return EmbeddingProgressEvent(0, 0, 0, 0, 0, **common)
    if tag in ("keypoint_detect", "keypoint_track"):
        from detail.core.keypoints import KeypointProgressEvent

        return KeypointProgressEvent(0, 0, 0, 0, 0, **common)
    if (
        tag == "keypoint_dataset"
    ):  # the label stage of a keypoint distill, labeling's event
        return LabelingProgressEvent(0, 0, 0, 0, **common)
    module = ANALYSIS_MODULES.get(tag)
    if module is not None and module.remote_capable and module.progress_event_class:
        # A remote-capable analysis module (§6 "Remote analysis"): §16's analysis event, whose
        # four leading fields are the units done and total and the running unit's share.
        return module.progress_event_class(0, 0, 0, 0, **common)
    return None


def _log_line(message: str) -> None:
    """One timestamped line in ``shepherd.log`` (this process's stdout), so a run that stops
    reporting still says what it was last doing."""
    print(f"{utc_timestamp()} {message}", flush=True)


class _TransferProgress:
    """What the shepherd publishes while the job does not yet exist on the cluster.

    Before this, the whole stretch from "starting" to the first event the remote worker wrote
    was silent: on an 80-video library that is ~116 GB of uploads, and a healthy run read as a
    hung one (#644). Each phase now has a message that advances, the video upload carries a
    measured ETA, and the same lines go to ``shepherd.log``.

    The upload is weighed by **bytes**, not files: the videos are not the same size, and a bar
    counting them would claim they cost the same.
    """

    def __init__(self, run_dir: Path, job) -> None:
        self._run_dir = run_dir
        self._job = job
        self._start = time.monotonic()
        self._clock = PhaseClock(self._start)
        self._throttle = ThrottledProgress(self._write)
        self._last_message: str | None = None
        self._logged_tenth = -1

    def frames(self, done: int, total: int) -> None:
        """Extracting the sampled frames from the videos, ahead of the upload.

        The message is stable (the CLI prints a line for each change) and the ETA moves; the log
        gets a line at each tenth. Already throttled by the extractor, so always published.
        """
        message = f"extracting {total:,} frames from the videos"
        if message != self._last_message:
            self._last_message = message
            _log_line(message)
        tenth = done * 10 // total if total else 0
        if tenth > self._logged_tenth:
            self._logged_tenth = tenth
            _log_line(f"extracted {done:,}/{total:,} frames")
        fraction = done / total if total else 0.0
        eta = self._clock.eta("frames", min(fraction, 1.0), time.monotonic())
        self._throttle(self._event(message, eta), force=True)

    def phase(self, message: str) -> None:
        """Enter a phase with nothing to measure. Published at once, never throttled away."""
        if message == self._last_message:
            return
        self._last_message = message
        _log_line(message)
        self._throttle(self._event(message, None), force=True)

    def videos(
        self,
        files_done: int,
        files_total: int,
        bytes_done: int,
        bytes_total: int,
        current_bytes: int = 0,
        *,
        force: bool = False,
    ) -> None:
        """The video upload: the message moves per finished file, the ETA with every byte.

        ``bytes_done`` counts finished files and ``current_bytes`` the one in flight, so the ETA
        keeps falling through a single multi-gigabyte file. The message stays put inside it,
        because the CLI prints a line each time the message changes.
        """
        message = (
            f"uploading videos · {files_done}/{files_total} files · "
            f"{bytes_done / 1e9:.1f}/{bytes_total / 1e9:.1f} GB"
        )
        if message != self._last_message:
            self._last_message = message
            _log_line(message)
        fraction = (bytes_done + current_bytes) / bytes_total if bytes_total else 0.0
        eta = self._clock.eta("videos", min(fraction, 1.0), time.monotonic())
        self._throttle(self._event(message, eta), force=force)

    def staging(self, state: dict) -> None:
        """Waiting on the background video upload a distill started, which this job needs.

        The staging run's own message and measured ETA are this phase's: it is the same upload,
        just owned by another process. Published each poll, never throttled away.
        """
        message = f"waiting for the background video upload · {state['message']}"
        if message != self._last_message:
            self._last_message = message
            _log_line(message)
        self._throttle(self._event(message, state.get("eta_s")), force=True)

    def _event(self, message: str, eta_s: float | None):
        return _phase_event(
            self._job,
            message=message,
            elapsed_s=time.monotonic() - self._start,
            eta_s=eta_s,
        )

    def _write(self, event) -> None:
        if event is not None:
            rundir.set_progress(self._run_dir, event)


def _size_of(path) -> int:
    """A file's size, or 0 when it cannot be read: the copy that follows reports a missing
    file in rsync's own words, which is more useful than failing a size probe here."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _effective_compute(target) -> ComputeTarget:
    """The on-cluster compute target the remote worker actually runs under — never
    ``"remote"``, so `resolve_device` and every runner are unmodified."""
    return ComputeTarget(
        backend="cuda",
        device_indices=list(range(target.gpus_per_job)),
        cloud_provider=None,
        cloud_instance=None,
    )


def _remote_paths(target, experiment_root: Path, run_dir: Path) -> tuple[str, str]:
    rel = Path(run_dir).relative_to(experiment_root)
    remote_experiment_root = (
        f"{target.remote_workdir.rstrip('/')}/{experiment_root.name}"
    )
    remote_run_dir = f"{remote_experiment_root}/{rel.as_posix()}"
    return remote_experiment_root, remote_run_dir


def _clear_foreign_run_dir(host: str, remote_run_dir: str, job) -> bool:
    """Remove the cluster's copy of this run's directory when it belongs to another job.

    The upload never overwrites a run directory the cluster already has (``copy_to_remote``'s
    ``never_overwrite``), on the premise that this job's own is not there yet. A run id that was
    deleted here and handed out again breaks that premise: the cluster still holds the deleted
    run's output, and the new job ran inside it — a 12-keypoint run opened the old 4-keypoint
    zarr, and a run whose cancel sentinel survived stopped in a second.

    Removed only on proof: the directory's ``job.json`` names a different ``job_id`` or
    ``created_at``. One that names this job is a previous attempt at it (a resume, a restarted
    shepherd) and its work is kept; one with no readable ``job.json`` is left as it always was.
    The ``cat`` cannot fail on a missing file, so an exception here is the connection, raised as
    it would be by the upload that follows.

    Returns:
        Whether a directory was removed.
    """
    result = ssh_transport.run_remote(
        host,
        f"cat {shlex.quote(posixpath.join(remote_run_dir, rundir.JOB_FILENAME))}"
        " 2>/dev/null || true",
    )
    try:
        remote_job = json.loads(result.stdout)["job"]
        remote_id = (remote_job["job_id"], remote_job.get("created_at"))
    except (ValueError, KeyError, TypeError):
        return False
    if remote_id == (job.job_id, getattr(job, "created_at", None)):
        return False
    ssh_transport.run_remote(host, f"rm -rf {shlex.quote(remote_run_dir)}")
    _log_line(
        f"replaced a stale copy of {posixpath.basename(remote_run_dir)} on the cluster "
        f"(it held job created {remote_id[1]})"
    )
    return True


def _extract_frames_for_cluster(run_dir: Path, experiment_root: Path, job, transfer):
    """The job to submit: a labeling job is sent its sampled frames instead of the videos.

    A labeling job reads ~1,000 randomly chosen frames across the recordings, and which frames is
    decided from the timestamp maps and the dataset id alone, so it is decided and read **here**,
    on the machine that holds the video, and only those images cross to the cluster (~3 GB, where
    the video library was ~116 GB and about an hour). Any other job, and one already pointing at
    frames, is returned unchanged.
    """
    if not isinstance(job, LabelingJob) or job.source_frames:
        return job
    from detail.pipeline import labeling

    labeling.extract_source_frames(
        experiment_root,
        job,
        run_dir / labeling.SOURCE_FRAMES_DIR,
        progress_cb=transfer.frames,
    )
    return dataclasses.replace(job, source_frames=labeling.SOURCE_FRAMES_DIR)


def _claim_local_inputs(experiment_root: Path, job) -> None:
    """Claim, on this machine and before the upload, an input the job's runner claims when it
    starts — because the cluster's claim lands in its own copy of the experiment, and only the
    run folder comes back.

    No job type claims anything today: the keypoints run that marked its reference set used
    (§26, before the rebuild) became ``keypoint_detect``, whose job carries a full copy of its
    example set, so a later edit of the set cannot change what a finished run recorded. Kept as
    the seam a future claim goes through.
    """
    del experiment_root, job


def _job_recording_ids(job) -> list[str]:
    """Which recordings, if any, this job's runner reads raw video for.

    None for a labeling job whose frames were already extracted: it reads images.

    A job with a single ``recording_id`` (detection, tracking, embedding) reads that
    one; a job with ``source_recording_ids`` (labeling) reads several — already
    resolved to a concrete list by the API method that built the job
    (``label_dataset`` / ``distill`` default an empty selection to every recording
    *before* constructing the job, so this never sees an "all recordings" sentinel
    here). Everything else (training: works from an already-labeled dataset;
    tracking: works from an existing detection run, not raw frames) has none."""
    if getattr(job, "source_frames", None):
        return []
    if hasattr(job, "recording_id"):
        return [job.recording_id]
    if hasattr(job, "source_recording_ids"):
        return list(job.source_recording_ids)
    return []


def _wait_for_staging(
    target, experiment_root: Path, run_dir: Path, job, transfer: "_TransferProgress"
) -> bool:
    """Wait out a background video upload covering this job's recordings; ``True`` if cancelled.

    A distill started with ``stage_videos`` copies every recording's video to this target while
    it labels and trains, so by the time its detection run starts the files are usually already
    there. Uploading them here as well, alongside, would copy the same bytes twice at once. So a
    live staging run is waited on, publishing its progress as this run's phase. If it failed
    or was stopped, ``_sync_video_roots`` below uploads whatever is missing itself: the cache is
    shared, so a failed background copy costs time and never correctness.
    """
    recording_ids = _job_recording_ids(job)
    if not recording_ids:
        return False
    last = video_staging.wait_for_active(
        experiment_root,
        target.name,
        recording_ids,
        transfer.staging,
        cancelled=lambda: rundir.cancel_requested(run_dir),
    )
    if last is not None and rundir.cancel_requested(run_dir):
        return True
    if last is not None:
        _log_line(
            f"background video upload ended ({last['status']})"
            + (f": {last['error']}" if last.get("error") else "")
        )
    return False


def _referenced_video_sources(
    experiment_root: Path, recording_ids: list[str]
) -> list[dict]:
    """Every reference-in-place ``{root, rel}`` video source the given recordings use.

    A **copied** part (a plain recording-relative ``file``, no ``source`` key) needs
    nothing here — it already lives inside the experiment directory the whole-tree
    mirror in :func:`run` sweeps up."""
    sources = []
    for recording_id in recording_ids:
        recording_dir = experiment_store.recording_path(experiment_root, recording_id)
        data = recording_store.read_recording(recording_dir)
        for file in data.get("video", {}).get("files", []):
            source = file.get("source")
            if source is not None:
                sources.append(source)
    return sources


def _sync_video_roots(
    target, experiment_root: Path, job, progress: "_TransferProgress | None" = None
) -> None:
    """Copy the video files a remote job's recordings reference — outside the
    experiment directory (Tenet: no absolute paths in experiment files is exactly
    what makes a video reference a portable ``{root, rel}`` alias rather than a path,
    per ``storage/video_roots.py``) and so never swept up by the whole-experiment
    mirror in :func:`run` — into a shared cache on the cluster, then registers each
    alias there the same way ``detail video-root set`` would locally.

    Without this, a remote detection/labeling/embedding job's worker fails
    immediately with ``MissingVideoRootError`` — the alias is registered on *this*
    machine, never on the one the job actually runs on. Found live, submitting a
    real distill job against a recording whose video root nothing on the cluster had
    ever heard of.

    A no-op for a job type with no raw-video dependency (an empty source list costs
    nothing) — see :func:`_job_recording_ids`.

    This is the long pre-submit phase of a large library (80 videos was ~116 GB), so it reports
    through ``progress`` — per finished file and per byte — when one is given.
    """
    sources = _referenced_video_sources(experiment_root, _job_recording_ids(job))
    if not sources:
        return
    remote_cache_root = f"{target.remote_workdir.rstrip('/')}/_video_roots"
    plan = []
    for source in sources:
        local_path = video_roots.resolve_video_ref(source)
        remote_path = posixpath.join(remote_cache_root, source["root"], source["rel"])
        plan.append((source["root"], local_path, remote_path, _size_of(local_path)))
    bytes_total = sum(size for *_rest, size in plan)
    bytes_done = 0
    if progress is not None:
        progress.videos(0, len(plan), 0, bytes_total, force=True)
    aliases = set()
    for index, (alias, local_path, remote_path, size) in enumerate(plan):

        def report(sent: int, index=index, finished=bytes_done) -> None:
            progress.videos(index, len(plan), finished, bytes_total, sent)

        ssh_transport.copy_file_to_remote(
            target.host,
            local_path,
            remote_path,
            on_bytes=None if progress is None else report,
        )
        bytes_done += size
        if progress is not None:
            progress.videos(index + 1, len(plan), bytes_done, bytes_total, force=True)
        aliases.add(alias)
    for alias in sorted(aliases):
        remote_alias_dir = posixpath.join(remote_cache_root, alias)
        script = (
            "from detail.storage import video_roots\n"
            f"video_roots.register_video_root({alias!r}, {remote_alias_dir!r})\n"
        )
        remote_python(
            target,
            script,
            f"could not register video root {alias!r} on remote target "
            f"{target.name!r} ({target.host})",
        )


def _model_readiness_probe(
    job, remote_experiment_root: str, runner_modules: str
) -> str | None:
    """Python source that, run on the remote host in the job's own module/conda
    environment, checks whether ``job``'s model is installed *there* and prints one
    JSON line describing a ``ModelReadiness``. ``None`` if the job carries no model,
    or if its job type has no model-resolution branch below (a model-less remote job
    type has nothing here to check).

    Resolves by **job type**, not just ``model_id`` — a detection/labeling job's id
    resolves through ``resolve_detection_model`` (the ``MODELS``/trained/imported
    registries — a labeling job's ``model_id`` is a *prompt-capable* member of the
    same ``MODELS`` registry, e.g. SAM 3, so it shares detection's branch exactly),
    an embedding job's through ``EMBEDDING_MODELS``, and the two are disjoint
    namespaces: probing an embedding job's id through the detection resolver raises
    ``unknown model_id`` for every real embedding model, since none of them are
    registered there. A **training** job's ``model_id`` is deliberately not handled
    here — it names the *model being created* by this job, not an existing one to
    check readiness of; ``TrainingJob.model_id`` only exists as an artifact after
    the job completes, so this probe has nothing to resolve for it (this machine's
    own equivalent — the trainer *framework* being importable — is checked, and
    skipped for a remote job, by ``Experiment.train_model``'s own guard instead).
    Add a branch here for each new job type embedding/detection's remote wiring
    gets copied to, mirroring the local worker's own resolution for that job type
    (``pipeline/embedding.py``'s ``EMBEDDING_MODELS[job.model_id]`` +
    ``construct_embedding_model``, not a second, independent guess at it).

    ``resolve_detection_model`` looks up a *trained*-model's family in
    ``TRAINED_LOADERS`` (populated only by importing the module that registers it,
    e.g. ``detail.trainers.ultralytics_yolo`` for "yolo") -- exactly the class of gap
    ``executor/worker.py`` exists to close for the worker itself
    (``import_runner_modules`` before it does anything else). This probe is a second,
    independent entry point into the same resolver and needs the identical import,
    or every trained/imported model's family registry is empty here: found live
    against a real cluster as "no trained-model loader for family 'yolo';
    available: []" for a model that was genuinely installed."""
    model_id = getattr(job, "model_id", None)
    if model_id is None:
        return None
    modules = [m for m in runner_modules.split(",") if m]
    import_lines = "".join(f"import {m}\n" for m in modules)
    job_type = job_type_tag(job)
    if job_type in ("detection", "labeling"):
        resolve_lines = (
            "from detail.models.resolve import resolve_detection_model\n"
            "model = resolve_detection_model(\n"
            f"    Path({remote_experiment_root!r}), {model_id!r},\n"
            f"    {getattr(job, 'weights_source', 'pretrained')!r}, "
            f"{getattr(job, 'model_variant', None)!r},\n"
            ")\n"
        )
    elif job_type == "embedding":
        resolve_lines = (
            "from detail.core.registries import "
            "EMBEDDING_MODELS, construct_embedding_model\n"
            f"model = construct_embedding_model(\n"
            f"    EMBEDDING_MODELS[{model_id!r}], {getattr(job, 'model_variant', None)!r},\n"
            f"    readout={getattr(job, 'readout', None)!r}, "
            f"input_size={getattr(job, 'input_size', None)!r},\n"
            ")\n"
        )
    elif job_type == "keypoint_detect":
        # A registered method resolves by id without its example set (a matching method's
        # readiness is its backbone's weights); an experiment-trained student is read from the
        # uploaded experiment, as ``KeypointDetectModule.validate`` reads it locally (§26).
        resolve_lines = (
            "from detail.core.registries import KEYPOINT_MODELS\n"
            "from detail.models.resolve import resolve_keypoint_model\n"
            f"if {model_id!r} in KEYPOINT_MODELS:\n"
            f"    model = KEYPOINT_MODELS[{model_id!r}]("
            f"{getattr(job, 'model_variant', None)!r})\n"
            "else:\n"
            f"    model = resolve_keypoint_model({model_id!r}, "
            f"experiment_dir=Path({remote_experiment_root!r}))\n"
        )
    else:
        return None
    return (
        "import dataclasses, json\n"
        "from pathlib import Path\n"
        f"{import_lines}"
        f"{resolve_lines}"
        "print(json.dumps(dataclasses.asdict(model.readiness())))\n"
    )


def _detection_catalog_model(job) -> str | None:
    return job.model_id if job.weights_source == "imported" else None


def _keypoint_catalog_model(job) -> str | None:
    """A keypoint detect job's model when it is in the catalog (``role: keypoints``): an imported
    keypoint model, or a student promoted there. A registered method is fetched by the cluster,
    and an experiment-trained student rides the experiment upload."""
    from detail.models import imported as imported_store

    return (
        job.model_id if job.model_id in imported_store.discover("keypoints") else None
    )


#: Per job type, the catalog model (``models/imported.py``) a job runs, if any — the one kind
#: of model the experiment upload never carries (an *experiment-trained* model lives under
#: ``models/`` in the experiment; a built-in is fetched by the cluster itself). Each entry asks
#: the question that job type's own resolver asks, so the two cannot disagree on what is a
#: catalog model. Every other job type that names a model is listed, with the reason it names no
#: catalog model, in ``tests/executor/test_imported_model_upload.py``.
_CATALOG_MODEL_OF: dict[str, Callable[[object], str | None]] = {
    "detection": _detection_catalog_model,  # weights_source == "imported"
    "keypoint_detect": _keypoint_catalog_model,  # role: keypoints in the catalog
}


def _catalog_model_id(job) -> str | None:
    """The catalog model ``job`` runs, or ``None`` if its model comes from anywhere else."""
    catalog_model_of = _CATALOG_MODEL_OF.get(job_type_tag(job))
    return None if catalog_model_of is None else catalog_model_of(job)


def _sync_catalog_model(
    target, job, announce: Callable[[str], None] | None = None
) -> None:
    """Put a **catalog** model on the cluster before the readiness check.

    A catalog model (a distilled or imported YOLO detector, an imported ONNX keypoint model, a
    distilled YOLO-pose keypoint student) lives in this machine's model library
    (``~/.cache/detail/models/imported/<id>``), outside the experiment, so the whole-experiment
    upload never carries it; the cluster found no ``model.yaml`` and every recording of a run
    failed at "could not check model readiness" (77 of 77 on a C57 detection). It is the model
    twin of :func:`_sync_video_roots`: the files a job needs that live outside the experiment are
    sent before the job is submitted.

    Where the model lands is **the cluster's own answer** (``imported.model_dir`` run there),
    not a path built here, so a cluster with its model cache elsewhere still finds it. The copy is
    rsync, so a model already there costs a comparison; and several recordings starting at once
    each send the same files to the same place, which is safe, since rsync writes each file under
    a temporary name and renames it, so a file is whole or absent and every sender finishes with
    all of them in place.

    Raises:
        FileNotFoundError: the model is not in this machine's library.
        RuntimeError: the cluster could not say where its imported models live.
    """
    model_id = _catalog_model_id(job)
    if model_id is None:
        return
    from detail.models import imported as imported_store
    from detail.storage import models as models_store

    local_dir = imported_store.model_dir(model_id)
    if not (local_dir / models_store.MODEL_FILENAME).is_file():
        raise FileNotFoundError(
            f"the model {model_id!r} is not in this machine's model library "
            f"({local_dir}), so there is nothing to upload to {target.name}"
        )
    if announce is not None:
        announce(f"uploading the model {model_id} to {target.name}")
    script = (
        "from detail.models import imported\n"
        f"print(imported.model_dir({model_id!r}))\n"
    )
    failed = (
        f"could not find where {target.name!r} ({target.host}) keeps its imported models, "
        f"to upload {model_id!r}"
    )
    result = remote_python(target, script, failed)
    # The last line: an activated environment's banner comes before it (see the readiness probe).
    lines = result.stdout.strip().splitlines()
    remote_dir = lines[-1].strip() if lines else ""
    if not posixpath.isabs(remote_dir):
        raise RuntimeError(
            f"{failed}: {result.stderr.strip() or result.stdout.strip() or 'no answer'}"
        )
    ssh_transport.copy_to_remote(target.host, local_dir, remote_dir)


def _check_model_readiness(
    target,
    job,
    remote_experiment_root: str,
    runner_modules: str,
    announce: Callable[[], None] | None = None,
) -> None:
    """Raise :class:`ModelNotReadyError` if ``job``'s model is not installed on
    ``target`` — reusing the same exception + message ``DetectionModule.validate()``
    raises for a local run, so the failure reads the same way regardless of where it
    was caught. A no-op for a job type with no model (``announce``, which names the
    phase, is only called when there is a check to run)."""
    script = _model_readiness_probe(job, remote_experiment_root, runner_modules)
    if script is None:
        return
    if announce is not None:
        announce()
    result = remote_python(
        target,
        script,
        f"could not check model readiness on remote target {target.name!r} "
        f"({target.host})",
    )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"could not parse model readiness output from remote target "
            f"{target.name!r} ({target.host}): {result.stdout!r}"
        ) from exc
    readiness = ModelReadiness(**payload)
    if not readiness.ready:
        raise ModelNotReadyError(readiness)


def _ask_remote_envelope(target, remote_run_dir: str) -> tuple[dict | None, bool]:
    """``(envelope, asked)``: the worker's own ``job.json`` as read over ssh, and whether ssh
    could run the question at all.

    ``(None, True)`` is an answer — the file is not there yet, or a torn read (a remote file has
    no atomicity guarantee over ssh; the worker's own writes are atomic, so it settles on retry).
    ``(None, False)`` is **not** an answer: the connection refused or dropped the question, and
    nothing is known about the file. The two used to look the same, and a caller deciding what a
    job's end meant read "could not ask" as "nothing written".
    """
    try:
        text = ssh_transport.read_remote_file(
            target.host,
            f"{remote_run_dir}/job.json",
            timeout_s=ssh_transport.ASK_TIMEOUT_S,
            refused_retries=ssh_transport.ASK_REFUSED_RETRIES,
        )
    except subprocess.TimeoutExpired:
        return None, False
    except ssh_transport.RemoteCommandError as exc:
        return None, exc.returncode != ssh_transport.SSH_ERROR_EXIT
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return None, True


def _read_remote_envelope(target, remote_run_dir: str) -> dict | None:
    return _ask_remote_envelope(target, remote_run_dir)[0]


def _live_files(job) -> tuple[str, ...]:
    """The run-dir-relative files a page reads while ``job`` is still running.

    A training run's curve reads the trainer's own epoch log off this machine's disk, and the
    copy-back brings that file home only once the job is over — so a cluster run drew no curve
    for its whole length. The trainer names the file (`Trainer.live_files`), because the path
    is its backend's, not this module's.
    """
    if job_type_tag(job) != "training":
        return ()
    from detail.core.registries import TRAINERS
    from detail.trainers import import_trainers

    import_trainers()
    trainer_cls = TRAINERS.get(job.trainer_id)
    return () if trainer_cls is None else tuple(trainer_cls.live_files)


def _mirror_live_files(
    target, remote_run_dir: str, run_dir: Path, live_files: tuple[str, ...]
) -> None:
    """Copy each live file from the cluster over its local twin; best-effort.

    A missed copy is retried at the next progress event, and the copy-back at the end brings
    the final file regardless, so a failure here only delays a picture. Written atomically:
    the page reading it is a different thread, at any moment.
    """
    for rel in live_files:
        try:
            text = ssh_transport.read_remote_file(
                target.host,
                f"{remote_run_dir}/{rel}",
                timeout_s=ssh_transport.ASK_TIMEOUT_S,
                refused_retries=ssh_transport.ASK_REFUSED_RETRIES,
            )
        except (subprocess.TimeoutExpired, ssh_transport.RemoteCommandError):
            continue  # not written yet (no epoch finished), or the ask did not get through
        local = run_dir / rel
        tmp = local.with_name(local.name + ".tmp")
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(text, encoding="utf-8", newline="")
            atomic.replace(tmp, local)
        except OSError:
            traceback.print_exc()


#: How many times, and how far apart, the shepherd re-reads the worker's envelope once the
#: scheduler says the job is over, before it believes the worker never wrote a terminal status.
SETTLE_READS = 4
SETTLE_WAIT_S = 2.0


def _settled_envelope(target, remote_run_dir: str) -> tuple[dict | None, bool]:
    """The worker's envelope once the scheduler says the job has ended: ``(envelope, asked)``.

    The poll reads the envelope, *then* asks the scheduler, and a job that finishes between the
    two has written its terminal status after the read the poll is still holding. So a scheduler
    that says "completed" is not yet a verdict on the job: the envelope is read again, a few
    times, until it is terminal. ``asked`` is False when ssh could not run the question — the
    caller waits for the connection instead of deciding.
    """
    envelope: dict | None = None
    for read in range(SETTLE_READS):
        envelope, asked = _ask_remote_envelope(target, remote_run_dir)
        if not asked:
            return None, False
        if envelope is not None and envelope.get("status") in _WORKER_TERMINAL_VALUES:
            return envelope, True
        if read + 1 < SETTLE_READS:
            time.sleep(SETTLE_WAIT_S)
    return envelope, True


#: Longest the shepherd waits between asks while the cluster cannot be reached. Backs off from
#: ``POLL_INTERVAL_S`` so a person who is away for the night is not a poll every 5 s.
MAX_UNREACHABLE_POLL_S = 60.0

#: How long a run waits to reconnect before giving up. A week is longer than any Slurm walltime
#: this tool submits, so a job that outlived it is gone and nobody is coming back for it.
MAX_UNREACHABLE_S = 7 * 24 * 3600


class _Reconnect:
    """The shepherd's state while it cannot reach the cluster, and the one place it says so.

    **Being unable to ask is not an answer about the job.** The poll loop used to read a dropped
    ssh session as "the scheduler has no record of this job" and fail a healthy run (a laptop
    sleeping through a four-hour detection did exactly that). This holds the run in a visible
    waiting state instead — the Slurm job is unaffected, and it carries on the moment the person
    logs in again — and gives up only after :data:`MAX_UNREACHABLE_S`.
    """

    def __init__(self, run_dir: Path, target, transfer, clock=time.monotonic) -> None:
        self._run_dir = run_dir
        self._target = target
        self._transfer = transfer
        self._clock = clock
        self._since: float | None = None
        self._since_label = ""
        self._asks = 0
        self._full = False

    @property
    def waiting(self) -> bool:
        return self._since is not None

    def lost(self, last_progress: dict | None) -> float:
        """Record one more failed ask, publish it, and return how long to wait before the next.

        Raises:
            TimeoutError: it has been unreachable for longer than :data:`MAX_UNREACHABLE_S`.
        """
        now = self._clock()
        full = ssh_transport.connection_full(self._target.host)
        began = self._since is None
        if began:
            self._since = now
            self._since_label = datetime.now().strftime("%H:%M")
            self._asks = 0
        if began or full != self._full:
            _log_line(
                f"the ssh connection to {self._target.host} is full; waiting for a free "
                f"session"
                if full
                else f"cannot reach {self._target.host} over ssh; waiting to reconnect"
            )
        self._full = full
        if now - self._since > MAX_UNREACHABLE_S:
            raise TimeoutError(
                f"{self._target.host} was unreachable over ssh for "
                f"{format_duration(now - self._since)}; the remote job may still be running "
                f"(check `squeue` on the cluster)"
            )
        self._publish(last_progress)
        self._asks += 1
        return min(POLL_INTERVAL_S * 2**self._asks, MAX_UNREACHABLE_POLL_S)

    def found(self) -> None:
        """Reachable again: say so in the log. The next real event replaces the waiting message."""
        if self._since is None:
            return
        _log_line(
            f"reconnected to {self._target.host} after "
            f"{format_duration(self._clock() - self._since)}"
        )
        self._since = None

    @core_progress.best_effort("reconnect message")
    def _publish(self, last_progress: dict | None) -> None:
        """Say it is waiting, and why. Progress is a statement *about* the work, so it cannot fail
        it.

        A full connection says so rather than "log in again": ssh exits the same way for both,
        and telling someone to log in to a connection that is working sent them to fix the wrong
        thing while the run waited on itself.
        """
        if self._full:
            message = (
                f"the ssh connection to {self._target.name} is full (too many commands at "
                f"once) since {self._since_label} — the job keeps running there, and this "
                f"continues by itself when a session frees up"
            )
        else:
            message = (
                f"cannot reach {self._target.name} over ssh since {self._since_label} — the "
                f"job keeps running there; log in again (ssh {self._target.host}) and this "
                f"continues"
            )
        if last_progress is not None:
            # The worker had already reported real numbers: keep them, change only the words,
            # so the bar does not fall to zero because the network did.
            rundir.update_payload(
                self._run_dir, progress={**last_progress, "message": message}
            )
        elif self._transfer is not None:
            self._transfer.phase(message)


def _wait_until_reachable(run_dir: Path, target, transfer) -> None:
    """Block until ``target`` answers over ssh, before the results are copied back.

    The job can finish while the connection is down, and copying then fails — after which the
    run was recorded ``complete`` with nothing on disk. The results are sitting on the cluster;
    waiting for the connection (as the poll loop does) is what gets them.

    Raises:
        TimeoutError: unreachable for longer than :data:`MAX_UNREACHABLE_S`.
    """
    reconnect = _Reconnect(run_dir, target, transfer)
    while not ssh_transport.is_reachable(target.host):
        time.sleep(reconnect.lost(None))
    reconnect.found()


def _poll_until_terminal(
    run_dir: Path,
    target,
    scheduler,
    remote_job_id: str,
    remote_run_dir: str,
    transfer: "_TransferProgress | None" = None,
    live_files: tuple[str, ...] = (),
) -> tuple[JobStatus, str | None]:
    cancelled_here = False
    worker_has_reported = False
    last_progress: dict | None = None
    reconnect = _Reconnect(run_dir, target, transfer)
    while True:
        if not cancelled_here and rundir.cancel_requested(run_dir):
            # Delivered, or tried again next poll: `False` is "could not reach the scheduler".
            cancelled_here = scheduler.cancel(target, remote_job_id) is not False

        envelope = _read_remote_envelope(target, remote_run_dir)
        if envelope is not None:
            progress = envelope.get("progress")
            if progress is not None:
                worker_has_reported = True
                if live_files and progress != last_progress:
                    # Before the progress is published, not after: a page redraws on the new
                    # epoch once, and one that found the old file would wait out a whole epoch.
                    _mirror_live_files(target, remote_run_dir, run_dir, live_files)
                last_progress = progress
                rundir.update_payload(run_dir, progress=progress)
            remote_status = envelope.get("status")
            if remote_status in _WORKER_TERMINAL_VALUES:
                return JobStatus(remote_status), envelope.get("error")

        state = scheduler.poll_state(target, remote_job_id)
        if state == JobState.UNREACHABLE:
            time.sleep(reconnect.lost(last_progress))
            continue
        reconnect.found()
        if transfer is not None and not worker_has_reported:
            # Until the worker's first event the envelope would still read the last phase the
            # shepherd named ("submitting"), through a queue wait that can run for hours.
            transfer.phase(
                f"waiting in the {target.name} queue"
                if state == JobState.PENDING
                else f"starting on {target.name} (loading the model)"
            )
        if state in TERMINAL_STATES:
            settled, asked = _settled_envelope(target, remote_run_dir)
            if settled is not None and settled.get("status") in _WORKER_TERMINAL_VALUES:
                return JobStatus(settled["status"]), settled.get("error")
            if not asked and state == JobState.COMPLETED:
                # "Completed" is the one scheduler word whose meaning is the worker's to give
                # (finished cleanly, or exited without saying), and it could not be asked: wait
                # for the connection, as for any other ask, rather than fail a job that may have
                # finished cleanly (a refused session read as "completed without a status" lost
                # a finished detection). A job the scheduler says *failed* or *was cancelled* is
                # that either way, so those do not wait.
                time.sleep(reconnect.lost(last_progress))
                continue
            # The scheduler says the job ended, but the worker never wrote its own
            # terminal status (crashed before that write, OOM-killed, node failure,
            # preempted, ...) — mirrors the local worker's own "exited without
            # completing" fallback (executor/local.py).
            return (
                _SCHEDULER_TO_JOB_STATUS.get(state, JobStatus.FAILED),
                f"remote job ended ({state.value}) without a terminal status",
            )
        time.sleep(POLL_INTERVAL_S)


def run(run_dir: Path, target_name: str, experiment_root: Path) -> JobStatus:
    run_dir = Path(run_dir)
    experiment_root = Path(experiment_root)
    try:
        # **Before the job is decoded.** A job type decodes only once the module that registers it
        # has been imported, and `keypoints` (like `cluster_fit` / `cluster_apply`) registers
        # itself on import, so reading the job first failed every remote keypoints run, in under a
        # second, with `unknown job type: 'keypoints'`. The local worker imports its modules before
        # reading the job; this process is a second entry point to the same decode and needs the
        # same import (the list comes from `RemoteExecutor`, which sets it in this environment).
        import_runner_modules(
            [m for m in os.environ.get(RUNNER_MODULES_ENV, "").split(",") if m]
        )
        target = remote_targets_store.get_remote_target(target_name)
        scheduler = _SCHEDULERS.get(target.scheduler)
        if scheduler is None:
            raise ValueError(f"unsupported scheduler {target.scheduler!r}")
        job = rundir.read_job(
            run_dir
        )  # as submitted: restored into job.json at the end
        target = _sized_for(target, job)
        # What the job holds, for its cost: a module may ask for fewer GPUs than the target's.
        rundir.update_payload(run_dir, remote_gpus=target.gpus_per_job)
        remote_experiment_root, remote_run_dir = _remote_paths(
            target, experiment_root, run_dir
        )

        transfer = _TransferProgress(run_dir, job)
        # The job the cluster runs. A labeling job reads ~1,000 sampled frames, so they are
        # extracted here, before the mirror below carries them, and no video goes up for it.
        cluster_job = _extract_frames_for_cluster(
            run_dir, experiment_root, job, transfer
        )
        _claim_local_inputs(experiment_root, job)
        transfer.phase("uploading experiment files")
        # A run directory the cluster already has is its worker's, not ours: other jobs in flight
        # there have rewritten theirs since we last saw them, so only the ones it lacks (this
        # job's own, among them) are sent. See `ssh_transport._rsync_mirror`. "This job's own"
        # is only absent if nothing else ever used its id, which is what is made true first.
        _clear_foreign_run_dir(target.host, remote_run_dir, job)
        ssh_transport.copy_to_remote(
            target.host,
            experiment_root,
            remote_experiment_root,
            never_overwrite=rundir.discover_run_dirs(experiment_root),
            # A run directory made after that list (the batch's next recording) would otherwise
            # have its local envelope laid over the one its own shepherd wrote on the cluster.
            never_send=[rundir.JOB_FILENAME],
        )
        _refresh_inputs(target, experiment_root, remote_experiment_root, job, transfer)

        if _wait_for_staging(target, experiment_root, run_dir, cluster_job, transfer):
            rundir.set_status(run_dir, JobStatus.CANCELLED)
            return JobStatus.CANCELLED

        # Raw video lives outside the experiment directory (a video_roots alias), so
        # the mirror above never carries it — see _sync_video_roots.
        _sync_video_roots(target, experiment_root, cluster_job, transfer)

        # RemoteExecutor.submit() set this in the shepherd's own environment
        # (executor/remote.py); forwarded to both the readiness probe and the sbatch
        # script because the compute node (and this probe's own ssh command) start a
        # fresh shell/process that inherits none of it (see render_sbatch_script).
        runner_modules = os.environ.get(RUNNER_MODULES_ENV, "")

        # Fail before the sbatch submission — not after it queues, runs, and dies —
        # when the model this job needs isn't installed on the cluster. The local
        # validate() skipped this on purpose (§ pipeline/detection.py): it is not
        # this machine's job to have the weights for a job that never runs here.
        # A catalog model is outside the experiment, so it is sent here, ahead of the check
        # that would otherwise fail on its absence.
        _sync_catalog_model(target, cluster_job, announce=transfer.phase)
        _check_model_readiness(
            target,
            cluster_job,
            remote_experiment_root,
            runner_modules,
            announce=lambda: transfer.phase("checking the model on the cluster"),
        )

        if rundir.cancel_requested(run_dir):
            # Cancelled during the upload: nothing is on the scheduler yet, so nothing to ask it.
            rundir.set_status(run_dir, JobStatus.CANCELLED)
            return JobStatus.CANCELLED
        transfer.phase("submitting the job")
        remote_job = dataclasses.replace(
            cluster_job, compute=_effective_compute(target)
        )
        remote_envelope = {
            "schema_version": 1,
            "status": JobStatus.PENDING.value,
            "job": job_to_dict(remote_job),
            "progress": None,
            "error": None,
        }
        ssh_transport.write_remote_file(
            target.host, f"{remote_run_dir}/job.json", json.dumps(remote_envelope)
        )

        script = render_sbatch_script(
            target, job.job_id, remote_run_dir, runner_modules
        )
        remote_job_id = scheduler.submit_script(target, script, remote_run_dir)
        rundir.update_payload(run_dir, remote_job_id=remote_job_id)
        rundir.set_status(run_dir, JobStatus.RUNNING)
    except Exception as exc:  # isolation boundary, same contract as executor/worker.py
        traceback.print_exc()
        rundir.set_status(
            run_dir, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
        )
        return JobStatus.FAILED

    try:
        status, error = _poll_until_terminal(
            run_dir,
            target,
            scheduler,
            remote_job_id,
            remote_run_dir,
            transfer,
            live_files=_live_files(job),
        )
    except Exception as exc:
        traceback.print_exc()
        status, error = (
            JobStatus.FAILED,
            f"shepherd polling error: {type(exc).__name__}: {exc}",
        )
        # A TimeoutError here is the poll giving up on a connection that never came back, so
        # there is nothing to wait for again below.
        reachable = not isinstance(exc, TimeoutError)
    else:
        reachable = True

    transfer.phase("syncing results back")
    worker_envelope: dict | None = None
    sync_error: str | None = None
    if reachable:
        try:
            _wait_until_reachable(run_dir, target, transfer)
        except TimeoutError as exc:
            # The job is over but its results are out of reach, so the run did not deliver them.
            traceback.print_exc()
            reachable = False
            if status == JobStatus.COMPLETE:
                status, error = (
                    JobStatus.FAILED,
                    f"results could not be copied back: {exc}",
                )
    if reachable:
        try:
            # **This job's own run folder, never the experiment.** A job writes only inside its
            # run dir (detection, embedding, labeling and training all do; `run_dir_for` is where
            # each says so), so that is all there is to bring back. Mirroring the whole remote
            # experiment down over the local one, as this used to, copied the *cluster's*
            # snapshot of every other file on top of what is on this machine now: an edit made
            # while a four-hour job ran (a renamed recording, another run's live `job.json`) was
            # reverted to what was uploaded at the start, and a run deleted locally came back.
            #
            # **Never the envelope (#742).** The cluster's `job.json` already says the job is
            # over, and every reader of this run trusts that word: the coordinator's reaper
            # (`LocalExecutor._reap`) gives a process with a terminal status ten seconds and
            # then kills its process group — this one, and the rsync inside it. The envelope
            # sorts before the data, so it arrived first and 153 of 153 embedding runs were
            # recorded complete with most of their chunks still on the cluster. So the copy
            # leaves it alone, the worker's own fields are merged in after the data is here
            # (`_merge_worker_envelope`), and the terminal status is the last thing written.
            ssh_transport.copy_from_remote(
                target.host,
                remote_run_dir,
                run_dir,
                keep_local=[
                    rundir.SHEPHERD_LOG_FILENAME,  # this process's own stdout
                    rundir.JOB_FILENAME,
                ],
            )
            worker_envelope = _read_remote_envelope(target, remote_run_dir)
            if worker_envelope is None:
                raise RuntimeError(
                    "the results were copied back but the job's envelope could not be read "
                    "from the cluster"
                )
        except Exception as exc:
            # The log records it; whether it changes the run's status is decided below.
            traceback.print_exc()
            sync_error = f"{type(exc).__name__}: {exc}"
    if status == JobStatus.COMPLETE and sync_error is not None:
        # A run that finished on the cluster but whose results never arrived is **not complete
        # here**: it would read complete with nothing on disk (the copy was refused and the
        # exception only printed). Failing it names the problem, and says where the results are.
        status, error = (
            JobStatus.FAILED,
            f"the job finished on the cluster but its results could not be copied back "
            f"({sync_error.splitlines()[0]}); they are in {remote_run_dir} on "
            f"{target.name}",
        )

    if worker_envelope is not None:
        _merge_worker_envelope(run_dir, worker_envelope)
        status, error = _after_copy_back(experiment_root, job, run_dir, status, error)
    if status == JobStatus.COMPLETE and cluster_job is not job:
        # The labeled examples are in the dataset now; the extracted source frames are scratch.
        # ignore_errors: a leftover scratch directory must not turn a complete run into a failure.
        shutil.rmtree(run_dir / cluster_job.source_frames, ignore_errors=True)
    rundir.set_status(run_dir, status, error=error)
    return status


def _runner_spec(job):
    return JOB_RUNNERS.get(job_type_tag(job))


def _sized_for(target, job):
    """``target`` as this job needs it: the GPUs its module asks for
    (``AnalysisModule.remote_resources``), with the target's CPUs and memory scaled with them,
    since a profile's fit grows with the node."""
    spec = _runner_spec(job)
    gpus = (spec.remote_resources(job) if spec is not None else {}).get("gpus")
    if not gpus or gpus == target.gpus_per_job:
        return target
    scale = gpus / target.gpus_per_job
    return target.model_copy(
        update={
            "gpus_per_job": gpus,
            "cpus_per_task": max(1, round(target.cpus_per_task * scale)),
            "mem_gb": max(1, round(target.mem_gb * scale)),
        }
    )


def _refresh_inputs(
    target, experiment_root: Path, remote_experiment_root: str, job, transfer
) -> None:
    """Re-send the run directories ``job`` reads over the cluster's copy of them.

    The upload above leaves alone every run directory the cluster already has, so a run this
    job *reads* stays as it first went up: a clustering run whose node cache was rebuilt here,
    or whose profiles were placed here since, would be read stale — and a resume that skips
    what is already placed would redo it all. The module names those runs
    (``AnalysisModule.refreshed_inputs``); each is sent again, its ``job.json`` aside, so the
    envelope the cluster holds for it is never replaced. The copy is exact: what was removed
    here is removed there, so a profile a cancelled job left on the cluster's copy (and never
    brought back) cannot make a later job skip its cluster.
    """
    spec = _runner_spec(job)
    if spec is None:
        return
    for directory in spec.refreshed_inputs(experiment_root, job):
        rel = Path(directory).relative_to(experiment_root).as_posix()
        transfer.phase(f"refreshing {rel} on the cluster")
        ssh_transport.mirror_to_remote(
            target.host,
            Path(directory),
            f"{remote_experiment_root}/{rel}",
            keep_remote=[rundir.JOB_FILENAME],
        )


def _after_copy_back(
    experiment_root: Path, job, run_dir: Path, status: JobStatus, error: str | None
) -> tuple[JobStatus, str | None]:
    """Run the module's own step on the copied-back results; ``(status, error)`` as they stand.

    Run whatever the job's status, since a stopped or partial run's finished work belongs where
    a complete run's does. A failure is the run's: it is failed with the reason and where the
    results are, rather than recorded complete with its results somewhere no surface looks.
    The step is idempotent, so it can be repeated over the same folder.
    """
    spec = _runner_spec(job)
    if spec is None:
        return status, error
    try:
        spec.after_copy_back(experiment_root, job, run_dir)
    except Exception as exc:  # the step's own failure, reported as the run's
        traceback.print_exc()
        return (
            JobStatus.FAILED,
            f"the results were copied back but could not be put in place "
            f"({type(exc).__name__}: {exc}); they are in {run_dir}",
        )
    return status, error


#: Envelope fields only this machine knows, which the worker's copy must never replace: the job
#: as submitted (``compute`` reads ``"remote"``, not the ``"cuda"`` `_effective_compute`
#: substituted for the cluster), when it was submitted here, the scheduler's id, the local pid,
#: and the verdict, which `rundir.set_status` writes last.
_SHEPHERD_OWNED = frozenset(
    {"schema_version", "status", "error", "job", "started_at", "completed_at", "pid"}
    | {"remote_job_id", "remote_gpus"}
)


def _merge_worker_envelope(run_dir: Path, worker: dict) -> None:
    """Bring the worker's own envelope fields into the local one, after the data is here (#742).

    Everything the worker wrote that this machine does not own: a stage's own results
    (detection's ``mask_resolution`` and ``parts``, an export's ``export_status``), its final
    ``progress``, and whatever a stage writes there next — a list of the fields known today is
    complete until the day one is added. Plus the worker's own clock: ``started_at`` here is when
    the job was *submitted*, and a job can sit in the queue for hours, so the cost estimate bills
    ``worker_started_at`` / ``worker_completed_at`` — written even when ``None`` (a job cancelled
    while queued never reached a GPU), because their presence says the record knows its running
    time.
    """
    fields = {k: v for k, v in worker.items() if k not in _SHEPHERD_OWNED}
    fields["worker_started_at"] = worker.get("started_at")
    fields["worker_completed_at"] = worker.get("completed_at")
    rundir.update_payload(run_dir, **fields)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        raise SystemExit(
            "usage: python -m detail.executor.remote_shepherd <run_dir> <target_name> "
            "<experiment_root>"
        )
    status = run(Path(argv[0]), argv[1], Path(argv[2]))
    return 0 if status == JobStatus.COMPLETE else 1


if __name__ == "__main__":
    sys.exit(main())
