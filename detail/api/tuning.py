"""Tune a detector's detection and tracking settings (§12 *Tune detection and tracking
settings*), behind ``Experiment``'s one-liners.

A tuning measures a trained detector on a sample of recordings and recommends a detection
confidence plus ``anchor_tracker`` hyperparameters (``pipeline/tune_tracking.py``). It lives
with the model, whoever started it (a distill's ``tune`` stage or a standalone tune), and is a
**report**: nothing defaults to it. A surface offers it through :func:`tuned_settings` (by
model) or :func:`tuned_settings_for_detection` (by the detection a tracking form
tracks), which take the newest complete tuning and say which one that is.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from detail.analysis.track_quality import Setting
from detail.configs.tuning import TuningConfig
from detail.core.dataclasses import ACTIVE_STATUSES, ComputeTarget
from detail.core.protocols import ModelNotReadyError
from detail.core.utils import default_compute, utc_timestamp
from detail.executor import rundir
from detail.executor.base import RunHandle
from detail.executor.handles import ExecutorRunHandle
from detail.storage import models as models_store
from detail.storage import recording as recording_store
from detail.storage import tunings as tunings_store

if TYPE_CHECKING:
    from detail.api.experiment import Experiment


class TuningRun(RunHandle):
    """Handle for one tuning job. ``wait`` returns once ``tuning.yaml`` is terminal; a
    worker that died before writing it is folded in from ``job.json``."""

    def __init__(self, inner: ExecutorRunHandle, experiment_dir: Path, model_id: str):
        self._inner = inner
        self._dir = Path(experiment_dir)
        self.model_id = model_id
        self.id = inner.id

    @property
    def status(self) -> str:
        return self._inner.status

    @property
    def error(self) -> str | None:
        return self._inner.error

    def progress(self) -> Iterator:
        return self._inner.progress()

    def wait(self) -> "TuningRun":
        self._inner.wait()
        reconcile(self._dir, self.model_id, self.id)
        return self

    def cancel(self) -> None:
        self._inner.cancel()

    def record(self) -> dict:
        """The tuning's ``tuning.yaml``, reconciled."""
        return read(self._dir, self.model_id, self.id)


def launch(
    experiment: "Experiment",
    model_id: str,
    source_recording_ids: list[str] | None,
    *,
    sample_seconds: float,
    compute: ComputeTarget | None,
    source: dict,
) -> TuningRun:
    """Validate and submit one tuning job (the body of ``Experiment.tune_tracking``)."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.local import LocalExecutor
    from detail.executor.runners import import_runner_modules
    from detail.models.resolve import resolve_detection_model
    from detail.pipeline.tune_tracking import TuneTrackingJob

    import_runner_modules(PIPELINE_RUNNERS)
    config = TuningConfig.model_validate({"sample_seconds": sample_seconds})
    model = require_tunable(experiment.dir, model_id)
    recording_ids = experiment._sampling_recordings(source_recording_ids)
    if not recording_ids:
        raise ValueError("tuning needs at least one recording with video")
    for recording_id in recording_ids:
        recording = experiment.get_recording(recording_id)  # raises if unknown
        data = recording_store.read_recording(recording.dir)
        if not (data.get("video") or {}).get("files"):
            raise ValueError(
                f"recording {recording.name!r} has no video to measure the model on; "
                "choose recordings that have video, or leave the choice empty to use all"
            )
    detector = resolve_detection_model(experiment.dir, model_id, "trained", None)
    compute = compute or default_compute(list(detector.supported_backends))
    if compute.backend not in detector.supported_backends:
        raise ValueError(
            f"model {model_id!r} cannot run on {compute.backend!r}; it supports "
            f"{list(detector.supported_backends)}"
        )
    status = detector.readiness()
    if not status.ready:
        raise ModelNotReadyError(status)
    for other in list_for_model(experiment.dir, model_id):
        if other.get("status") in ACTIVE_STATUSES:
            raise ValueError(
                f"model {model_id!r} is already being tuned ({other['id']}); wait for it "
                "or cancel it first"
            )
    tuning_id = tunings_store.next_tuning_id(experiment.dir, model_id)
    job = TuneTrackingJob(
        job_id=tuning_id,
        model_id=model_id,
        weights_source="trained",
        model_variant=model.get("model_variant"),
        source_recording_ids=recording_ids,
        sample_seconds=config.sample_seconds,
        compute=compute,
        created_at=utc_timestamp(),
        source=dict(source),
    )
    executor = LocalExecutor(experiment.dir, runner_modules=PIPELINE_RUNNERS)
    executor.submit(job)
    return TuningRun(ExecutorRunHandle(executor, tuning_id), experiment.dir, model_id)


def require_tunable(experiment_dir: Path | str, model_id: str) -> dict:
    """The model's record, when it is a finished detector of this experiment.

    Raises:
        FileNotFoundError: no such model.
        ValueError: it is still training, failed, or is a keypoint model.
    """
    record = models_store.read_model_yaml(
        models_store.model_dir(experiment_dir, model_id)
    )
    if record.get("task") == "pose":
        raise ValueError(
            f"model {model_id!r} places keypoints; tuning measures a detector's masks "
            "through the tracker"
        )
    if record.get("status") not in ("complete", "partial"):
        raise ValueError(
            f"model {model_id!r} is {record.get('status')!r}; tune a model that has "
            "finished training"
        )
    return record


def reconcile(experiment_dir: Path | str, model_id: str, tuning_id: str) -> None:
    """Fold a terminal ``job.json`` into a ``tuning.yaml`` still reading ``running`` (a
    worker killed before it could write its own status)."""
    directory = tunings_store.tuning_dir(experiment_dir, model_id, tuning_id)
    try:
        record = tunings_store.read_tuning(directory)
        status = rundir.read_status(directory).value
    except (FileNotFoundError, KeyError, ValueError):
        return
    if record.get("status") in ACTIVE_STATUSES and status not in ACTIVE_STATUSES:
        record["status"] = status
        error = rundir.read_envelope(directory).get("error")
        if error:
            record.setdefault("error", error)
        tunings_store.write_tuning(directory, record)


def read(experiment_dir: Path | str, model_id: str, tuning_id: str) -> dict:
    reconcile(experiment_dir, model_id, tuning_id)
    return tunings_store.read_tuning(
        tunings_store.tuning_dir(experiment_dir, model_id, tuning_id)
    )


def list_for_model(experiment_dir: Path | str, model_id: str) -> list[dict]:
    """Every tuning of the model, newest first, reconciled."""
    for record in tunings_store.list_tunings(experiment_dir, model_id):
        reconcile(experiment_dir, model_id, record["id"])
    return tunings_store.list_tunings(experiment_dir, model_id)


def source_label(record: dict) -> str:
    """Who made a tuning, in words: ``distill_003`` or ``a standalone tune``."""
    source = record.get("source") or {}
    if source.get("kind") == "distill" and source.get("distill_id"):
        return f"distill {source['distill_id']}"
    return "a standalone tune"


@dataclass(frozen=True)
class TunedSettings:
    """The newest complete tuning of a model, as a surface offers it."""

    model_id: str
    tuning_id: str
    source: str
    created_at: str
    n_complete: int
    setting: Setting | None
    keep_defaults: bool
    reason: str
    warnings: list[str] = field(default_factory=list)
    #: Set by :func:`tuned_settings_for_detection`: the confidence that detection was made
    #: at, which is the least it kept.
    detection_conf_used: float | None = None

    @property
    def needs_new_detection(self) -> bool:
        return (
            self.setting is not None
            and self.detection_conf_used is not None
            and self.detection_conf_used > self.setting.detection_conf + 1e-9
        )

    def describe(self) -> str:
        """Which tuning this is: ``tune_002 (from distill_003), the newest of 2``."""
        text = f"{self.tuning_id} (from {self.source}, {self.created_at})"
        if self.n_complete > 1:
            text += f", the newest of {self.n_complete}"
        return text

    def flags(self) -> tuple[str, str] | None:
        """The detection and tracking CLI flags that apply it, or None with no setting."""
        if self.setting is None:
            return None
        s = self.setting
        return (
            f"--conf {s.detection_conf:g}",
            f"--conf-threshold {s.conf_threshold:g} --secondary-conf "
            f"{s.secondary_conf:g} --max-secondary {s.max_secondary}",
        )


def tuned_settings(experiment_dir: Path | str, model_id: str) -> TunedSettings | None:
    """The newest complete tuning of ``model_id`` (from a distill or a standalone tune), or
    None when it has never been tuned to completion."""
    for record in tunings_store.list_tunings(experiment_dir, model_id):
        reconcile(experiment_dir, model_id, record["id"])
    record, n = tunings_store.latest_complete(experiment_dir, model_id)
    if record is None:
        return None
    pick = record.get("pick") or {}
    setting = pick.get("setting")
    return TunedSettings(
        model_id=model_id,
        tuning_id=record["id"],
        source=source_label(record),
        created_at=str(record.get("finished_at") or record.get("created_at") or ""),
        n_complete=n,
        setting=Setting.from_dict(setting) if setting else None,
        keep_defaults=bool(pick.get("keep_defaults")),
        reason=str(pick.get("reason") or ""),
        warnings=list(record.get("warnings") or []),
    )


def tuned_settings_for_detection(
    experiment_dir: Path | str, detection: dict
) -> TunedSettings | None:
    """The tuning of the model a detection used, with the confidence that detection kept, so
    a tracking form can say when it must be made again at a lower one.

    ``detection`` is a detection run's record or a selection's identity: anything carrying
    ``model_id``, ``weights_source`` and ``conf_threshold``. None when the model is not one
    this experiment trained, or was never tuned."""
    if detection.get("weights_source") != "trained" or not detection.get("model_id"):
        return None
    found = tuned_settings(experiment_dir, detection["model_id"])
    if found is None:
        return None
    return dataclasses.replace(
        found, detection_conf_used=float(detection.get("conf_threshold") or 0.0)
    )
