"""Serializable data types shared across the pipeline.

This module is the data layer every other component implements against:
prompt and cue types, the standardized detection output, compute targeting,
the per-stage job and progress-event dataclasses, and their unions. It has no
backend or Qt dependency — only stdlib and numpy.

Job dataclasses are JSON-serializable (Tenet 4) via ``to_dict`` / ``from_dict``.
Runtime result types (``Detection``, ``FrameResult``, ``RenderPreviewResult``)
carry ``np.ndarray`` payloads and are intentionally not serialized.
"""

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field

from enum import Enum

import numpy as np


# --------------------------------------------------------------------------- #
# Job status
# --------------------------------------------------------------------------- #
class JobStatus(str, Enum):
    """Lifecycle state of a submitted job.

    A ``str`` subclass (not ``StrEnum``, which is 3.11+) so members serialize as
    their plain string value through ``json.dumps`` / ``yaml.dump``.
    """

    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"  # interrupted but resumable (Tenet 12)
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"  # via Executor.cancel


# Statuses a run can be in while it still occupies the recording — used by module
# validation to reject launching a second concurrent run of the same stage.
ACTIVE_STATUSES = (JobStatus.PENDING.value, JobStatus.RUNNING.value)
# A computed run has output to show once it is terminal. `partial` counts: some recordings
# failed or were cancelled, and its artifacts list only the ones that finished.
OUTPUT_STATUSES = (JobStatus.COMPLETE.value, JobStatus.PARTIAL.value)
# An **authored** run (§21) has output while it is still being worked on — the statistics are
# what the user is deciding against, so withholding them until they finish inverts the
# workflow. `annotating` is clip annotation's, `open` is every later authored module's.
AUTHORED_STATUSES = ("annotating", "open")
# A terminal status that means the job did not finish its work. No record may claim
# otherwise — see `reconciled_status`.
FAILED_STATUSES = (JobStatus.FAILED.value, JobStatus.CANCELLED.value)


def reconciled_status(record_status: str | None, job_status: str | None) -> str | None:
    """The status a run record should adopt given its job envelope, or ``None`` to leave it.

    The envelope (``job.json``) is written by the worker's isolation boundary and is the only
    place a crash is recorded. The record is written by the *runner*, from inside the job, and
    can therefore describe work the job never went on to finish.

    Two rules, and the second is the one that was missing:

    - a record the runner never finalized — absent, ``pending``, ``running`` — adopts the
      envelope's terminal status. A crash or a kill leaves the initial record, and the
      envelope is the best we have.
    - a record claiming anything **other than a failure** adopts a ``failed`` / ``cancelled``
      envelope. A runner cannot pre-declare an outcome it did not reach, and an authored run
      (§21) does exactly that: it writes ``status: open`` *before* projecting its artifact,
      because an open run is supposed to have one. When that projection died on a network
      share, ``analysis.yaml`` went on saying ``open`` — a failed run listed beside healthy
      ones, offered for interpretation, and re-running the write that killed it the moment
      anybody opened it. The old guard read ``status in (None, "running")``, which is correct
      for eight analysis modules and silently inapplicable to the two that need it most.

    A *successful* authored run is untouched: ``open`` against a ``complete`` envelope is the
    runner's own statement and stays (§21 — nobody has said they are done with it yet).
    """
    if job_status is None or job_status in ACTIVE_STATUSES:
        return None
    if record_status is None or record_status in ACTIVE_STATUSES:
        return job_status
    if job_status in FAILED_STATUSES and record_status not in FAILED_STATUSES:
        return job_status
    return None


# --------------------------------------------------------------------------- #
# Prompt and cue types
# --------------------------------------------------------------------------- #
@dataclass
class TextCue:
    text: str


@dataclass
class PointCue:
    points: list[tuple[float, float]]
    labels: list[int]  # 1=foreground, 0=background


@dataclass
class BoxCue:
    boxes: list[tuple[float, float, float, float]]  # x1,y1,x2,y2 normalized


@dataclass
class ImageCue:
    image: np.ndarray | None  # runtime pixels; None after deserialization until loaded
    path: str | None  # experiment-relative path once persisted; used for serialization


Cue = TextCue | PointCue | BoxCue | ImageCue


@dataclass
class PromptTarget:
    class_name: str
    cue: Cue


def _cue_to_serialized(cue: Cue) -> tuple[str, object]:
    """Return the ``(cue_type, payload)`` pair for a cue.

    Raises:
        ValueError: if an ``ImageCue`` has no persisted ``path``.
        TypeError: if the cue is not a known cue type.
    """
    if isinstance(cue, TextCue):
        return "text", cue.text
    if isinstance(cue, PointCue):
        return "point", {
            "points": [list(p) for p in cue.points],
            "labels": list(cue.labels),
        }
    if isinstance(cue, BoxCue):
        return "box", {"boxes": [list(b) for b in cue.boxes]}
    if isinstance(cue, ImageCue):
        if cue.path is None:
            raise ValueError(
                "ImageCue must be persisted to a experiment-relative path before "
                "serialization; the API layer writes the image under the run "
                "directory and sets ImageCue.path."
            )
        return "image", cue.path
    raise TypeError(f"unknown cue type: {type(cue).__name__}")


def _cue_from_serialized(cue_type: str, payload: object) -> Cue:
    """Reconstruct a cue from its serialized ``(cue_type, payload)`` form."""
    if cue_type == "text":
        return TextCue(text=payload)
    if cue_type == "point":
        return PointCue(
            points=[tuple(p) for p in payload["points"]],
            labels=list(payload["labels"]),
        )
    if cue_type == "box":
        return BoxCue(boxes=[tuple(b) for b in payload["boxes"]])
    if cue_type == "image":
        return ImageCue(image=None, path=payload)
    raise ValueError(f"unknown cue_type: {cue_type!r}")


@dataclass
class Prompt:
    targets: list[PromptTarget]

    def to_dict(self) -> dict:
        """Serialize to a JSON-native dict with a ``cue_type`` discriminator."""
        targets = []
        for target in self.targets:
            cue_type, payload = _cue_to_serialized(target.cue)
            targets.append(
                {"class_name": target.class_name, "cue_type": cue_type, "cue": payload}
            )
        return {"targets": targets}

    @classmethod
    def from_dict(cls, d: dict) -> "Prompt":
        targets = [
            PromptTarget(
                class_name=t["class_name"],
                cue=_cue_from_serialized(t["cue_type"], t["cue"]),
            )
            for t in d["targets"]
        ]
        return cls(targets=targets)


# --------------------------------------------------------------------------- #
# Standardized model output (runtime types — not serialized)
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One row of a detection ``FrameResult`` — the view the detection code reads."""

    mask: np.ndarray | None  # binary (H, W) bool
    bbox: tuple[float, float, float, float] | None  # x1,y1,x2,y2 normalized
    class_name: str
    confidence: float
    object_id: int | None  # = track_id once tracking has assigned one


_FRAME_RESULT_ROWS = (
    "track_id",
    "class_name",
    "confidence",
    "bbox",
    "mask",
    "points",
    "point_scores",
    "candidates",
    "candidate_scores",
    "detection_row",
)


@dataclass
class FrameResult:
    """A frame's instances as equal-length rows (§5 "Standardized output").

    Every model that locates something in a frame returns one. A detection model fills
    ``class_name`` / ``confidence`` / ``bbox`` / ``mask``; a keypoint model (§26) fills
    ``points`` / ``point_scores`` (and ``candidates`` when it keeps more than one per point);
    a model that does both (a YOLO-pose student, §12) writes one result. Each field is optional
    and present when the model produces it; every array's first axis is ``n``. ``frame_idx``
    and ``t_ns`` are stamped by the runner, not the model.
    """

    frame_idx: int  # index within the video part
    t_ns: int  # from the active timestamp map (Tenet 8)
    n: int  # rows; every array below has n as its first axis
    track_id: np.ndarray | None = None  # (n,) int32; None before tracking
    class_name: list[str] | None = None  # (n,); a keypoint model leaves it None
    confidence: np.ndarray | None = None  # (n,) float32; the instance's own score
    bbox: np.ndarray | None = None  # (n, 4) float32 normalized xyxy; a NaN row = no box
    mask: list[np.ndarray | None] | None = None  # n binary (H, W) masks
    points: np.ndarray | None = (
        None  # (n, K, 2) float32 normalized full-frame; NaN = not placed
    )
    point_scores: np.ndarray | None = None  # (n, K) float32, the method's own
    candidates: np.ndarray | None = None  # (n, K, C, 2) float32; absent when C = 1
    candidate_scores: np.ndarray | None = None  # (n, K, C) float32
    # (n,) int32: the detection part row a per-detection keypoint method placed each instance
    # on; −1 = from no detection (a whole-frame method, §26)
    detection_row: np.ndarray | None = None
    extras: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # method-only, first axis n

    def __post_init__(self) -> None:
        for name in _FRAME_RESULT_ROWS:
            value = getattr(self, name)
            if value is not None and len(value) != self.n:
                raise ValueError(
                    f"FrameResult.{name} has {len(value)} rows but n is {self.n}"
                )
        for name, value in self.extras.items():
            if len(value) != self.n:
                raise ValueError(
                    f"FrameResult.extras[{name!r}] has {len(value)} rows but n is {self.n}"
                )

    @classmethod
    def from_detections(
        cls, frame_idx: int, t_ns: int, detections: list[Detection]
    ) -> "FrameResult":
        """The row form of ``Detection`` rows — how a detection model builds its result."""
        rows = list(detections)
        n = len(rows)
        bbox = np.full((n, 4), np.nan, dtype=np.float32)
        for i, det in enumerate(rows):
            if det.bbox is not None:
                bbox[i] = det.bbox
        ids = [det.object_id for det in rows]
        return cls(
            frame_idx=frame_idx,
            t_ns=t_ns,
            n=n,
            track_id=(
                None
                if all(i is None for i in ids)
                else np.asarray([-1 if i is None else i for i in ids], dtype=np.int32)
            ),
            class_name=[det.class_name for det in rows],
            confidence=np.asarray([det.confidence for det in rows], dtype=np.float32),
            bbox=bbox,
            mask=[det.mask for det in rows],
        )

    def detections(self) -> list[Detection]:
        """The rows as ``Detection``s — what the detection writer, the overlays and the
        labeling export read."""
        out: list[Detection] = []
        for i in range(self.n):
            box = None
            if self.bbox is not None and not np.isnan(self.bbox[i]).any():
                box = tuple(float(v) for v in self.bbox[i])
            track = None
            if self.track_id is not None and int(self.track_id[i]) >= 0:
                track = int(self.track_id[i])
            out.append(
                Detection(
                    mask=None if self.mask is None else self.mask[i],
                    bbox=box,
                    class_name="" if self.class_name is None else self.class_name[i],
                    confidence=(
                        float("nan")
                        if self.confidence is None
                        else float(self.confidence[i])
                    ),
                    object_id=track,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Compute targeting
# --------------------------------------------------------------------------- #
@dataclass
class ComputeTarget:
    backend: str  # "cpu" | "cuda" | "mps" | "cloud"
    device_indices: list[int] | None  # cuda: [0], [0,1], ...; None = all available
    cloud_provider: str | None  # cloud: "modal" | "aws" | ...
    cloud_instance: str | None  # cloud: "A100" | "g4dn.xlarge" | ...

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "device_indices": self.device_indices,
            "cloud_provider": self.cloud_provider,
            "cloud_instance": self.cloud_instance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ComputeTarget":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Model readiness and install (Section 5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelLicense:
    """The terms a model's weights come under (§5 ``Model.license``): shown before install
    when they restrict use. Nothing reads it yet; the first model to set one brings the
    surface (§26 *Decisions during build*)."""

    name: str  # e.g. "DINOv3 License"
    url: str | None  # where the terms are read


@dataclass
class ModelReadiness:
    """Structured, cache-first status of a model's weights and credentials.

    Drives proactive "set up access" UI and the pre-run validation check. The
    network is consulted only when the weights are not installed (see
    ``models/hf_access``), so an installed model resolves offline. ``ready`` means
    installed *and* runnable: access alone earns ``not_installed``, never ``ok``,
    because ``load()`` will not download.
    """

    ready: bool
    reason: str  # "ok" | "not_installed" | "no_token" | "no_access" | "offline" | "bad_token"
    message: str  # human, actionable
    action_url: str | None  # token settings, or the exact gate page
    repo_id: str | None


@dataclass
class InstallProgress:
    """One progress tick while a model's weights download (Section 5).

    A plain callback payload, not a ProgressEvent — install is an API-level step,
    not an executor job.
    """

    bytes_done: int
    bytes_total: int | None  # None until the total is known
    current_file: str | None


# --------------------------------------------------------------------------- #
# Job dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class DetectionJob:
    job_id: str
    recording_id: str
    model_id: str
    prompt: Prompt
    conf_threshold: float
    compute: ComputeTarget
    created_at: str
    model_variant: str | None = None  # None = the model's default variant
    # "pretrained" = look model_id up in the MODELS registry; "trained" = model_id
    # is a experiment model.yaml id, loaded via TRAINED_LOADERS (closed-set, §12).
    weights_source: str = "pretrained"
    batch_size: int = 1  # frames per GPU forward pass (batch-capable models; §7)
    # Process every Nth source frame (1 = every frame). Rows keep the *source* frame
    # index, so nothing downstream renumbers (§4).
    every_n_frames: int = 1

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "recording_id": self.recording_id,
            "every_n_frames": self.every_n_frames,
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "weights_source": self.weights_source,
            "prompt": self.prompt.to_dict(),
            "conf_threshold": self.conf_threshold,
            "batch_size": self.batch_size,
            "compute": self.compute.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DetectionJob":
        return cls(
            job_id=d["job_id"],
            recording_id=d["recording_id"],
            every_n_frames=int(d.get("every_n_frames") or 1),
            model_id=d["model_id"],
            prompt=Prompt.from_dict(d["prompt"]),
            conf_threshold=d["conf_threshold"],
            compute=ComputeTarget.from_dict(d["compute"]),
            created_at=d["created_at"],
            model_variant=d.get("model_variant"),
            weights_source=d.get("weights_source", "pretrained"),
            batch_size=d.get("batch_size", 1),
        )


@dataclass
class LabelingClass:
    class_id: int
    name: str
    prompt: str  # text prompt for this class

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "name": self.name, "prompt": self.prompt}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelingClass":
        return cls(**d)


@dataclass
class LabelingJob:
    job_id: str
    dataset_id: str
    name: str
    source_recording_ids: list[str]  # empty = all recordings in experiment
    classes: list[LabelingClass]
    n_frames: int
    logit_threshold: float
    min_contour_area: (
        int  # px; drop contours smaller than this (every larger one is kept)
    )
    val_split: float  # fraction held out for validation (0.0 = no val set)
    compute: ComputeTarget
    created_at: str
    sampling: str = "stratified"  # within-part frame draw: "stratified" | "uniform"
    model_id: str = "sam3"  # prompt-capable detection model doing the labeling
    model_variant: str | None = None  # None = the model's default variant
    # Largest share of the dataset kept as empty-label background frames. The API always
    # passes LabelingConfig's resolved value; 0.0 is what a job recorded before the field
    # existed did.
    background_share: float = 0.0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source_recording_ids": list(self.source_recording_ids),
            "classes": [c.to_dict() for c in self.classes],
            "n_frames": self.n_frames,
            "logit_threshold": self.logit_threshold,
            "min_contour_area": self.min_contour_area,
            "val_split": self.val_split,
            "compute": self.compute.to_dict(),
            "created_at": self.created_at,
            "sampling": self.sampling,
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "background_share": self.background_share,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelingJob":
        return cls(
            job_id=d["job_id"],
            dataset_id=d["dataset_id"],
            name=d["name"],
            source_recording_ids=list(d["source_recording_ids"]),
            classes=[LabelingClass.from_dict(c) for c in d["classes"]],
            n_frames=d["n_frames"],
            logit_threshold=d["logit_threshold"],
            min_contour_area=d["min_contour_area"],
            val_split=d["val_split"],
            compute=ComputeTarget.from_dict(d["compute"]),
            created_at=d["created_at"],
            sampling=d.get("sampling", "stratified"),
            model_id=d.get("model_id", "sam3"),
            model_variant=d.get("model_variant"),
            background_share=d.get("background_share", 0.0),
        )


@dataclass
class TrainingJob:
    job_id: str
    model_id: str
    dataset_id: str
    trainer_id: str  # key into TRAINERS registry
    base_model: str  # pretrained starting point — trainer interprets this string
    hyperparams: dict  # trainer-specific; validated by trainer.validate_hyperparams()
    compute: ComputeTarget
    created_at: str

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "trainer_id": self.trainer_id,
            "base_model": self.base_model,
            "hyperparams": dict(self.hyperparams),
            "compute": self.compute.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingJob":
        return cls(
            job_id=d["job_id"],
            model_id=d["model_id"],
            dataset_id=d["dataset_id"],
            trainer_id=d["trainer_id"],
            base_model=d["base_model"],
            hyperparams=dict(d["hyperparams"]),
            compute=ComputeTarget.from_dict(d["compute"]),
            created_at=d["created_at"],
        )


def _json_native(value):
    """``value`` with every tuple and set flattened to a list, recursively.

    A job's fields must survive two very different writers: ``json.dump`` into ``job.json``,
    which is forgiving, and ``yaml.dump`` into the run record via ``runners.job_params``, which
    is not — it preserves a tuple as ``!!python/tuple`` and ``yaml.safe_load`` refuses to read
    that back. Anything that reaches a job from a *surface* rather than from a typed field has
    to come through here.
    """
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, set, frozenset)):
        return [_json_native(item) for item in value]
    if isinstance(value, list):
        return [_json_native(item) for item in value]
    return value


@dataclass
class ExportJob:
    """One export a user asked for, carried out as a run (#354).

    **One request is one job**, however many targets it covers: they share a destination folder
    and a manifest, because that is what was asked for once. ``target_ids`` is therefore a list
    and the run's units are its entries.

    ``destination`` is a ``{root, rel}`` reference, never an absolute path. Everything in a job
    reaches the run record through ``runners.job_params``, and a record is ``.yaml`` — which
    ``storage/export.py::portability_offenses`` scans, and which would make
    ``detail experiment export`` refuse this experiment for good (Tenet 6). The alias resolves
    through ``storage/roots.py`` in the worker.

    ``folder`` is minted at submit rather than in the worker so the surface can tell the user
    where the export is going before it starts, and so a retry cannot silently land somewhere
    else.
    """

    job_id: str
    subject_kind: str
    subject_id: str
    target_ids: list[str]
    destination: dict  # {"root": <alias>, "rel": <parent folder, relative to the root>}
    folder: str
    created_at: str
    scope: dict = dataclasses.field(default_factory=dict)
    options: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "target_ids": list(self.target_ids),
            "destination": dict(self.destination),
            "folder": self.folder,
            "created_at": self.created_at,
            # **Coerced, not copied.** Every other job hand-lists its fields with known types;
            # this one takes an opaque bag from whichever surface built it, and the GUI's clip
            # knobs put a *tuple* in `renders`. `json.dump` would quietly make it a list, but a
            # job also reaches the run record through `runners.job_params` — and that record is
            # YAML, where `yaml.dump` writes `!!python/tuple` and `yaml.safe_load` then refuses
            # to read it back. The runner reads its own record to finish, so the run does the
            # work and *then* dies: envelope `failed`, files on disk, nothing to show for it.
            "scope": _json_native(self.scope),
            "options": _json_native(self.options),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExportJob":
        return cls(
            job_id=d["job_id"],
            subject_kind=d["subject_kind"],
            subject_id=d.get("subject_id", ""),
            target_ids=list(d["target_ids"]),
            destination=dict(d["destination"]),
            folder=d["folder"],
            created_at=d["created_at"],
            scope=dict(d.get("scope") or {}),
            options=dict(d.get("options") or {}),
        )


@dataclass
class DatasetExportJob:
    """Export a training dataset from an existing detection/tracking run (§12)."""

    job_id: str
    dataset_id: str
    name: str
    source_run_id: str  # experiment processing run (tracking | detection)
    frame_selection: dict  # {"mode": "every_n", "n": int} | {"mode": "all"}
    val_split: float
    min_contour_area: int  # px; drop tiny contours
    created_at: str

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source_run_id": self.source_run_id,
            "frame_selection": dict(self.frame_selection),
            "val_split": self.val_split,
            "min_contour_area": self.min_contour_area,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetExportJob":
        return cls(
            job_id=d["job_id"],
            dataset_id=d["dataset_id"],
            name=d["name"],
            source_run_id=d["source_run_id"],
            frame_selection=dict(d["frame_selection"]),
            val_split=d["val_split"],
            min_contour_area=d["min_contour_area"],
            created_at=d["created_at"],
        )


@dataclass
class TrackingJob:
    job_id: str
    recording_id: str
    track_id: str
    detection_run_id: str
    tracker_id: str  # key into TRACKERS registry
    hyperparams: dict  # validated by tracker.validate_hyperparams()
    arena_ids: list[str]  # empty = no arena restriction
    created_at: str
    # Link every Nth *source* frame — a multiple of the stride detection ran at (§4).
    # The trackers age tracks by stepping this, so max_missed counts processed frames.
    every_n_frames: int = 1

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "recording_id": self.recording_id,
            "track_id": self.track_id,
            "detection_run_id": self.detection_run_id,
            "tracker_id": self.tracker_id,
            "hyperparams": dict(self.hyperparams),
            "arena_ids": list(self.arena_ids),
            "created_at": self.created_at,
            "every_n_frames": self.every_n_frames,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrackingJob":
        return cls(
            job_id=d["job_id"],
            recording_id=d["recording_id"],
            track_id=d["track_id"],
            detection_run_id=d["detection_run_id"],
            tracker_id=d["tracker_id"],
            hyperparams=dict(d["hyperparams"]),
            arena_ids=list(d["arena_ids"]),
            created_at=d["created_at"],
            every_n_frames=int(d.get("every_n_frames") or 1),
        )


@dataclass
class EmbeddingJob:
    job_id: str
    recording_id: str
    embedding_run_id: str
    tracking_run_id: str
    track_ids: list[int]  # which tracks to embed
    model_id: str
    renderer_spec: list[dict]  # serializable MaskRenderer pipeline
    batch_size: int
    num_workers: int  # not read (§14 *Execution*: threads); kept for runs on disk
    compute: ComputeTarget
    created_at: str
    model_variant: str | None = None  # None = the model's default variant
    constructor: str | None = None  # named recipe that produced renderer_spec, if any
    readout: str | None = None  # feature read-out; None = the model's default
    input_size: int | None = (
        None  # model input resolution (px); None = the model's default
    )
    # Scope each data-derived constructor parameter was resolved at (§14). Recorded so a
    # later comparability check knows whether the canvas is *allowed* to differ between
    # recordings: a per_recording canvas is a deliberate per-recording normalization, a
    # shared one must be identical everywhere. The background is always resolved across the
    # run's selection; a job from before that was resolved per recording (``from_dict``).
    canvas_scope: str = "per_recording"
    bg_color_scope: str = "shared"
    # Embed every Nth *source* frame — a multiple of the stride detection ran at (§4).
    every_n_frames: int = 1
    # §14 *Egocentric alignment*: ``none`` leaves every field below at its default, so two
    # unaligned jobs never differ in a setting that did nothing.
    align: str = "none"
    keypoint_run_id: str | None = (
        None  # the keypoint track run (kt_NN) the heading reads
    )
    heading_definition: dict | None = None  # {front: [...], back: [...]}, resolved
    heading_cutoff: float | None = None  # the track run's decode cut-off, if it had one
    max_gap_s: float = 0.5
    mirror_hold_s: float = 1.0
    facing_camera_ratio: float = 0.5

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "recording_id": self.recording_id,
            "embedding_run_id": self.embedding_run_id,
            "tracking_run_id": self.tracking_run_id,
            "track_ids": list(self.track_ids),
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "constructor": self.constructor,
            "readout": self.readout,
            "input_size": self.input_size,
            "renderer_spec": [dict(step) for step in self.renderer_spec],
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "compute": self.compute.to_dict(),
            "created_at": self.created_at,
            "canvas_scope": self.canvas_scope,
            "bg_color_scope": self.bg_color_scope,
            "every_n_frames": self.every_n_frames,
            "align": self.align,
            "keypoint_run_id": self.keypoint_run_id,
            "heading_definition": (
                None
                if self.heading_definition is None
                else dict(self.heading_definition)
            ),
            "heading_cutoff": self.heading_cutoff,
            "max_gap_s": self.max_gap_s,
            "mirror_hold_s": self.mirror_hold_s,
            "facing_camera_ratio": self.facing_camera_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingJob":
        return cls(
            job_id=d["job_id"],
            recording_id=d["recording_id"],
            embedding_run_id=d["embedding_run_id"],
            tracking_run_id=d["tracking_run_id"],
            track_ids=list(d["track_ids"]),
            model_id=d["model_id"],
            renderer_spec=[dict(step) for step in d["renderer_spec"]],
            batch_size=d["batch_size"],
            num_workers=d["num_workers"],
            compute=ComputeTarget.from_dict(d["compute"]),
            created_at=d["created_at"],
            model_variant=d.get("model_variant"),
            constructor=d.get("constructor"),
            readout=d.get("readout"),
            input_size=d.get("input_size"),
            # A job written before the scopes existed resolved both per recording.
            canvas_scope=d.get("canvas_scope", "per_recording"),
            bg_color_scope=d.get("bg_color_scope", "per_recording"),
            every_n_frames=int(d.get("every_n_frames") or 1),
            # A job written before alignment existed embedded the image as cropped.
            align=str(d.get("align") or "none"),
            keypoint_run_id=d.get("keypoint_run_id"),
            heading_definition=d.get("heading_definition"),
            heading_cutoff=d.get("heading_cutoff"),
            max_gap_s=float(d.get("max_gap_s", 0.5)),
            mirror_hold_s=float(d.get("mirror_hold_s", 1.0)),
            facing_camera_ratio=float(d.get("facing_camera_ratio", 0.5)),
        )


Job = (
    DetectionJob
    | LabelingJob
    | TrainingJob
    | DatasetExportJob
    | TrackingJob
    | EmbeddingJob
)

# Job-type registry (§6.5). The hardcoded map is gone: the first-party job
# types defined in this module seed themselves below, and pipeline modules
# register their own types via register_processing_module — so a new module's
# serialization is wired without editing core. job_from_dict requires the
# owning module imported first (the worker and executor import runner modules
# before touching jobs).
_JOB_TYPES: dict[str, type] = {}
_JOB_TAGS: dict[type, str] = {}


def register_job_type(tag: str, cls: type) -> None:
    """Register a Job subclass under its serialization ``tag`` (§6.5)."""
    _JOB_TYPES[tag] = cls
    _JOB_TAGS[cls] = tag


for _tag, _cls in (
    ("detection", DetectionJob),
    ("labeling", LabelingJob),
    ("training", TrainingJob),
    ("dataset_export", DatasetExportJob),
    ("tracking", TrackingJob),
    ("embedding", EmbeddingJob),
    ("export", ExportJob),
):
    register_job_type(_tag, _cls)


def job_to_dict(job: Job) -> dict:
    """Serialize any job to a JSON-native dict tagged with its ``type``.

    The tag lets a subprocess worker reconstruct the right class from job.json
    without knowing it in advance (Section 6).
    """
    tag = _JOB_TAGS.get(type(job))
    if tag is None:
        raise TypeError(f"not a known Job type: {type(job).__name__}")
    return {"type": tag, **job.to_dict()}


def job_from_dict(d: dict) -> Job:
    """Reconstruct a job from its tagged dict (inverse of ``job_to_dict``)."""
    d = dict(d)
    tag = d.pop("type")
    cls = _JOB_TYPES.get(tag)
    if cls is None:
        raise ValueError(f"unknown job type: {tag!r}")
    return cls.from_dict(d)


def job_type_tag(job: Job) -> str:
    """The discriminator tag for a job (e.g. ``DetectionJob`` -> ``"detection"``)."""
    tag = _JOB_TAGS.get(type(job))
    if tag is None:
        raise TypeError(f"not a known Job type: {type(job).__name__}")
    return tag


# --------------------------------------------------------------------------- #
# Progress events
# --------------------------------------------------------------------------- #
# ``message`` is ephemeral phase text (e.g. "loading model"), shown by the CLI;
# None on ordinary frame ticks. Appended with a default so existing positional
# and keyword constructions stay valid.
@dataclass
class DetectionProgressEvent:
    parts_done: int
    parts_total: int
    frames_done: int  # aggregate across all parallel parts
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


@dataclass
class LabelingProgressEvent:
    frames_done: int
    frames_total: int
    classes_done: int  # classes fully labeled across the sampled frames
    classes_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


@dataclass
class TrainingProgressEvent:
    epoch: int
    total_epochs: int
    metrics: dict[str, float]  # trainer-specific keys, e.g. {"box_map50": 0.85}
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


@dataclass
class TrackingProgressEvent:
    parts_done: int
    parts_total: int
    frames_done: int
    frames_total: int
    n_active_tracks: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


@dataclass
class EmbeddingProgressEvent:
    track_id: int
    parts_done: int
    parts_total: int
    frames_done: int  # run-wide frames embedded so far (all units + in-flight)
    frames_total: int  # run-wide unique-frame total — the bar's real denominator
    elapsed_s: float
    eta_s: float | None
    units_done: int = 0  # (track, part) units completed across the run
    units_total: int = 0  # total (track, part) units — shown in the bar postfix
    message: str | None = None


@dataclass
class BatchProgressEvent:
    """Within-run progress for a **batch** processing run (§6.5) — the `inner` of a
    ``ProcessingProgress`` for a module that runs once over the whole subset.
    ``frames_done``/``frames_total`` carry the run's fine-grained
    fraction (a resolution, not literal frames), so the aggregate bar moves *within* a
    single-recording batch instead of jumping 0→100%; ``message`` names the phase."""

    frames_done: int
    frames_total: int
    message: str | None = None


ProgressEvent = (
    DetectionProgressEvent
    | LabelingProgressEvent
    | TrainingProgressEvent
    | TrackingProgressEvent
    | EmbeddingProgressEvent
    | BatchProgressEvent
)

# Progress-event registry (§6.5) — same dynamic-registration pattern as jobs.
_PROGRESS_EVENT_TYPES: dict[str, type] = {}
_PROGRESS_EVENT_TAGS: dict[type, str] = {}


def register_progress_event_type(tag: str, cls: type) -> None:
    """Register a ProgressEvent subclass under its serialization ``tag`` (§6.5)."""
    _PROGRESS_EVENT_TYPES[tag] = cls
    _PROGRESS_EVENT_TAGS[cls] = tag


for _tag, _cls in (
    ("detection", DetectionProgressEvent),
    ("labeling", LabelingProgressEvent),
    ("training", TrainingProgressEvent),
    ("tracking", TrackingProgressEvent),
    ("embedding", EmbeddingProgressEvent),
    ("batch", BatchProgressEvent),
):
    register_progress_event_type(_tag, _cls)


def progress_event_to_dict(event: ProgressEvent) -> dict:
    """Serialize a progress event to a JSON-native dict tagged with its ``type``.

    The executor streams progress across the subprocess boundary as these dicts
    (Section 6).
    """
    tag = _PROGRESS_EVENT_TAGS.get(type(event))
    if tag is None:
        raise TypeError(f"not a known ProgressEvent type: {type(event).__name__}")
    return {"type": tag, **dataclasses.asdict(event)}


def progress_event_from_dict(d: dict) -> ProgressEvent:
    """Reconstruct a progress event from its tagged dict."""
    d = dict(d)
    tag = d.pop("type")
    cls = _PROGRESS_EVENT_TYPES.get(tag)
    if cls is None:
        raise ValueError(f"unknown progress event type: {tag!r}")
    return cls(**d)


@dataclass
class ProcessingProgress:
    """Aggregate progress for a experiment-level processing run (§15).

    Carries the current recording's per-stage ``ProgressEvent`` plus the run-level
    ``elapsed_s`` (for a whole-run ETA). A detached run coordinator writes these to
    ``runs/<id>/progress.json`` so any process — GUI reopen, CLI ``attach`` — can
    reattach to a run that outlived the app.
    """

    recordings_done: int
    recordings_total: int
    current_recording: str
    inner: ProgressEvent | None
    elapsed_s: float = 0.0  # this coordinator's wall-clock, for the ETA
    # How many recordings were already complete when this coordinator started — non-zero
    # only for a retry (§15), which drives the rest into an existing record. The bar
    # spans the whole run (recordings_done / recordings_total), but ``elapsed_s`` only ever
    # buys the recordings *this* coordinator ran, so an ETA must divide by the work since
    # this mark. Dividing the run-wide fraction by it instead reads 30 min remaining on
    # a 15 h retry.
    recordings_at_start: int = 0


def processing_progress_to_dict(p: ProcessingProgress) -> dict:
    """Serialize aggregate run progress for the run-level ``progress.json`` (§15)."""
    return {
        "recordings_done": p.recordings_done,
        "recordings_total": p.recordings_total,
        "current_recording": p.current_recording,
        "inner": progress_event_to_dict(p.inner) if p.inner is not None else None,
        "elapsed_s": p.elapsed_s,
        "recordings_at_start": p.recordings_at_start,
    }


def processing_progress_from_dict(d: dict) -> ProcessingProgress:
    """Reconstruct aggregate run progress from its dict."""
    inner = d.get("inner")
    return ProcessingProgress(
        recordings_done=d["recordings_done"],
        recordings_total=d["recordings_total"],
        current_recording=d["current_recording"],
        inner=progress_event_from_dict(inner) if inner is not None else None,
        elapsed_s=d.get("elapsed_s", 0.0),
        recordings_at_start=d.get("recordings_at_start", 0),
    )


# Stage weights fold the two distill stages into one bar (§12). Training usually
# dominates wall-clock, so it carries the larger share; the fraction is nominal — the
# renderer shows the phase message alongside, so the exact split need not be measured.
DISTILL_STAGE_WEIGHTS = {"labeling": 0.45, "training": 0.55}
DISTILL_STAGE_ORDER = ("labeling", "training")


def distill_overall_fraction(phase: str, inner_fraction: float) -> float:
    """Fold a stage's inner fraction into the whole-run fraction (stage-weighted, §12).

    Stages before ``phase`` count fully; the active stage contributes its clamped
    ``inner_fraction`` scaled by its weight. An unrecognized phase reads as all named
    stages done.
    """
    offset = 0.0
    clamped = max(0.0, min(1.0, inner_fraction))
    for name in DISTILL_STAGE_ORDER:
        if name == phase:
            return offset + DISTILL_STAGE_WEIGHTS[name] * clamped
        offset += DISTILL_STAGE_WEIGHTS[name]
    return offset


@dataclass
class DistillProgress:
    """Aggregate progress for a distill run (§12) — the two stages (label → train)
    folded into one bar. A detached distill coordinator writes these to
    ``distill/<id>/progress.json`` so any process — GUI reopen, CLI ``attach`` — can
    reattach to a run that outlived the app.
    """

    phase: str  # "labeling" | "training" | "registering" (a keypoint distill's third)
    stage_index: int  # 0-based index of the active stage
    stage_count: int  # 2, or 3 for a keypoint distill (§12 *Distill keypoints*)
    inner_fraction: float  # 0..1 within the active stage
    overall_fraction: float  # 0..1 stage-weighted across the whole run
    elapsed_s: float = 0.0  # this coordinator's wall-clock, for the ETA
    eta_s: float | None = None
    message: str = ""  # advancing phase label (never frozen — §6)


def distill_progress_to_dict(p: DistillProgress) -> dict:
    """Serialize distill progress for the run-level ``progress.json`` (§12)."""
    return {
        "phase": p.phase,
        "stage_index": p.stage_index,
        "stage_count": p.stage_count,
        "inner_fraction": p.inner_fraction,
        "overall_fraction": p.overall_fraction,
        "elapsed_s": p.elapsed_s,
        "eta_s": p.eta_s,
        "message": p.message,
    }


def distill_progress_from_dict(d: dict) -> DistillProgress:
    """Reconstruct distill progress from its dict."""
    return DistillProgress(
        phase=d["phase"],
        stage_index=d["stage_index"],
        stage_count=d["stage_count"],
        inner_fraction=d["inner_fraction"],
        overall_fraction=d["overall_fraction"],
        elapsed_s=d.get("elapsed_s", 0.0),
        eta_s=d.get("eta_s"),
        message=d.get("message", ""),
    )


# --------------------------------------------------------------------------- #
# Preview / prompt-test (runtime types — not serialized)
# --------------------------------------------------------------------------- #
@dataclass
class PromptTestJob:
    """Lightweight pre-detection prompt evaluation (Section 8).

    Synchronous: runs on n_sample_frames with the given prompts and returns
    results directly. Not an executor Job — no zarr output, no recovery.

    No confidence threshold: the run keeps every proposal so the threshold can be chosen
    while looking at the result (``configs/display.ConfidenceView``).
    """

    recording_id: str
    model_id: str
    prompts: list[Prompt]  # N prompts to compare
    n_sample_frames: int  # M random frames
    compute: ComputeTarget
    model_variant: str | None = None  # None = the model's default variant


@dataclass
class PromptTestResult:
    frame_idx: int
    t_ns: int
    frame_image: np.ndarray  # the sampled frame (H, W, 3)
    results: dict[str, FrameResult]  # prompt label -> that prompt's rows


@dataclass
class RenderPreviewJob:
    tracking_run_id: str
    track_ids: list[int]
    renderer_spec: list[dict]
    n_sample_frames: int


@dataclass
class RenderPreviewResult:
    frame_idx: int
    t_ns: int
    track_id: int
    source_frame: np.ndarray  # original video frame with mask overlay — for reference
    rendered_image: np.ndarray  # output of the renderer pipeline — what the model sees
    det_row: int  # back-reference to source detection zarr row


@dataclass
class RenderContext:
    """Input/output carried through a MaskRenderer's transform pipeline (Section 14)."""

    frame: np.ndarray | None  # decoded RGB frame (H, W, 3); None when the renderer
    # reads no pixels (silhouette constructors) and video decode was skipped (§14)
    mask: np.ndarray | None  # binary mask (H, W) bool, decoded from COCO RLE
    bbox: tuple[float, float, float, float] | None  # x1,y1,x2,y2 normalized
    image: np.ndarray | None  # current rendered image — updated by each transform
    frame_idx: int
    t_ns: int
    class_name: str
    confidence: float
    object_id: int
    aux: dict  # open slot: secondary masks, metadata, research extensions


# --------------------------------------------------------------------------- #
# Spatial definitions (Section 10)
# --------------------------------------------------------------------------- #
@dataclass
class PolygonKeyframes:
    """Time-varying polygon for an arena/ROI (Section 10).

    Each keyframe is a ``(frame_idx, vertices)`` pair; ``vertices`` is exactly
    ``n_vertices`` normalized ``(x, y)`` points. The active polygon at an
    arbitrary frame is obtained by linear interpolation between bracketing
    keyframes, clamped at the ends. Lookup/interpolation helpers live in the
    spatial subsystem; this type only holds the keyframe table.
    """

    n_vertices: int
    keyframes: list[tuple[int, list[tuple[float, float]]]]


# --------------------------------------------------------------------------- #
# Streams (Section 3)
# --------------------------------------------------------------------------- #
@dataclass
class AuxInput:
    """An extra file a ``StreamImporter`` needs beyond the primary data file.

    Lets a multi-file importer declare its slots so the API/GUI can collect them
    (e.g. MED-PC needs its ``.MPC`` program for the event-code → label map).
    ``key`` indexes the ``aux_paths`` dict passed to ``import_stream``.
    """

    key: str
    label: str
    extensions: list[str]  # e.g. [".mpc"] — for the GUI file picker filter


@dataclass
class StreamImportResult:
    """What a ``StreamImporter`` returns after parsing one source (Section 3).

    The importer parses to canonical rows; the storage layer writes the canonical
    CSV and composes the recording.yaml manifest. ``columns`` are the canonical CSV
    header (e.g. ``["t_ns", "code", "label"]``); ``rows`` are the parsed rows;
    ``metadata`` carries source provenance (subject, program, code→label map, …).

    Clock/shape (Section 3):
        ``shape`` is the temporal layout — ``"event"`` | ``"interval"`` |
        ``"continuous"`` (generalizes the legacy ``type``; when ``None`` the
        storage layer derives it from ``type``). An importer whose native clock is
        the **video frame index** leaves the ``t_ns`` columns unresolved and
        declares ``frame_columns`` ``{t_ns_col: source_frame_col}``; the import
        layer fills each ``t_ns`` column from the recording's active timestamp map
        (Approach Y — the importer stays recording-agnostic). An importer already in
        video ``t_ns`` (e.g. BORIS seconds → ns) instead declares ``time_columns``
        ``{frame_col: source_t_ns_col}`` so the import layer fills the convenience
        frame columns by the inverse map. ``dtypes`` maps each final column name to
        its dtype (``"int64"`` | ``"int32"`` | ``"str"``) for the typed read.
    """

    type: str
    columns: list[str]
    rows: list[tuple]
    sample_rate_hz: float | None = None
    metadata: dict | None = None
    shape: str | None = None
    clock: str | None = None
    frame_columns: dict[str, str] | None = None
    time_columns: dict[str, str] | None = None
    dtypes: dict[str, str] | None = None


# --------------------------------------------------------------------------- #
# Tools (Section 9)
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    """What a ``Tool.run`` returns — a JSON-native summary the run store folds into
    the ``tool.yaml`` manifest's ``result`` field (Section 9). No numpy payloads: a
    tool persists its heavy output to ``out_dir`` and names it in ``outputs`` (a path
    relative to the run dir); playback / mp4 export re-read from there.
    """

    n_processed: int  # frames actually processed (≤ the requested count)
    frames_total: int  # approximate total frames in the source(s)
    fps: float
    resolution: tuple[int, int]  # (width, height)
    class_counts: dict[str, int]  # detections kept, per class name
    outputs: dict[str, str]  # logical name -> path relative to the run dir
    summary: str
    # True when the user stopped the run early. The outputs are real but partial, so the
    # run store records it as ``stopped`` rather than passing it off as a complete result.
    stopped: bool = False

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["resolution"] = list(self.resolution)  # yaml-native (no !!python/tuple)
        return data


class CheckStatus(str, Enum):
    """A ``Check``'s outcome — advisory (nothing blocks). ``str``-valued so a result is
    trivially JSON-serialisable across the subprocess probe boundary."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"  # the probe couldn't answer (e.g. the subprocess timed out)
    SKIPPED = "skipped"  # a prerequisite check wasn't satisfied


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome (the live checker framework, §1). ``detail`` is a human
    one-liner; ``to_dict`` / ``from_dict`` cross the probe subprocess as JSON.

    ``target_id`` names the object a per-recording check ran against (a recording id), or
    ``None`` for a machine-global check — so the surface can group results per recording.
    """

    check_id: str
    status: CheckStatus
    detail: str = ""
    target_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "detail": self.detail,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckResult":
        return cls(
            data["check_id"],
            CheckStatus(data["status"]),
            data.get("detail", ""),
            data.get("target_id"),
        )


@dataclass(frozen=True)
class RecordingTarget:
    """The object a per-recording ``Check`` runs against — passed to ``run`` for
    ``target_kind == "recording"`` checks (the machine-global checks ignore it)."""

    experiment_dir: str
    recording_id: str
    recording_dir: str
    recording_data: dict


# --- Query methods (§20) ------------------------------------------------------------- #
# A query method searches a trajectory for stretches resembling user-marked example clips.
# These are the declarations the Query module reads *before* running anything: what a method
# needs, what it can use, and how it may be split. They are properties of the method's
# implementation, not user config, so they live on the class rather than in a config model.


@dataclass(frozen=True)
class ExemplarBounds:
    """What a method requires of the example clips it is given (§20).

    Enforced at ``validate()``, before any compute, naming the bound and what was given —
    a method that genuinely needs exactly one exemplar says ``min_clips == max_clips == 1``
    rather than failing obscurely on the second.

    The clip count and the row total are **independent** constraints: a method that pools
    every exemplar frame into one index cares about the rows and barely about how many clips
    they came from, while a method that scores each clip separately and fuses cares about the
    count, because its cost is linear in it.

    Rows here, seconds elsewhere, and that is deliberate. The unit-convention guard fails a
    *temporal window* declared in rows, because the same count is a different duration at
    every cadence. These are not windows — they are a sample budget, cadence-independent in
    exactly the way a window is not.

    ``requires_contiguous`` is the fourth bound and the one that is easy to put in the wrong
    place. A method that matches a **path** (the removed ``dtw`` did) needs its exemplar to be one
    unbroken stretch, because a clip spanning a tracking gap is not a trajectory the animal traced.
    A method that pools every exemplar frame into an unordered index — ``soft_knn`` — never
    consults the order, so the same rule refuses clips it could have used. It lived in the
    module for as long as there was one method, which meant the default method inherited a
    requirement it does not have and users were told to re-pick a clip that would have worked.
    """

    min_clips: int = 1
    max_clips: int | None = None
    min_total_rows: int = 1
    max_total_rows: int | None = None
    #: Must each clip lie inside one continuous trajectory run? ``True`` is the conservative
    #: default: a new method is refused a gap-spanning clip until it says it can take one.
    requires_contiguous: bool = True


@dataclass(frozen=True)
class NegativeSupport:
    """Whether a method can use user-marked non-examples, and how many (§20).

    ``unusable`` methods get the control **greyed out with the reason** rather than hidden:
    a user who marks fifteen careful non-examples into a method that silently discards them
    has been misled by an empty affordance. ``required`` refuses to launch without them.

    ``reason`` is that sentence, and it belongs to the method for the same reason
    ``score_meaning`` does: only the method knows why. The GUI used to hold one — "scores
    distance to the examples and has nothing to score them against" — which was true of the
    removed ``dtw`` and of nothing else that could ever be registered. Left empty, the surface falls
    back to naming the method.
    """

    support: str = "unusable"  # unusable | optional | required
    min_clips: int = 0
    max_clips: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class SamplingRequest:
    """The background a method needs, if any (§20).

    The **module** draws it, never the method, so every draw is recorded identically
    (strategy, scope, size, seed, and the rows drawn) and a run reproduces. ``scope``
    is also a comparability decision: a background drawn per recording calibrates
    confidence to that recording, so one cutoff means different things in different
    animals, while a cohort draw keeps one cutoff meaning one thing.
    """

    size: int
    scope: str = "cohort"  # subject | recording | cohort
    strategy: str = "uniform"


@dataclass(frozen=True)
class QueryCandidate:
    """One stretch a method found, before any cutoff is applied (§20).

    Rows are inclusive and **trajectory-global** — the module may hand a method a chunk of a
    long trajectory, and the method offsets before returning, so a merged result never mixes
    coordinate systems.

    ``confidence`` is canonical: ``[0, 1]``, higher is better, uniform across methods. That
    is what makes the cutoff axis, the audit stratification and the artifact's score column
    method-independent — and it makes the score-direction hazard (a matcher emitting a
    similarity where a distance is expected, reversing the ranking with no error)
    inexpressible rather than a declaration the module has to trust. ``score`` carries the
    method's own native number alongside it, for a reader who knows what it means.
    """

    start_row: int
    end_row: int
    confidence: float
    score: float
    example_idx: int = 0

    @property
    def n_rows(self) -> int:
        """Inclusive row length."""
        return self.end_row - self.start_row + 1


@dataclass(frozen=True)
class ClusterInstance:
    """One run of frames holding a discovered cluster (§22).

    Rows are inclusive and trajectory-global, like ``QueryCandidate``'s. Deliberately *not* that
    class: its ``confidence`` / ``score`` / ``example_idx`` have no meaning for a cluster
    assignment, and a made-up confidence is worse than none — §19's artifact carries a score
    column whose meaning the method declares, and a number invented to fill a field would be read
    as calibrated.

    ``cluster_id`` is the method's own id at the granularity the instance was produced at. It is
    what §19 stores as ``source_label``; the user's name for it lives in ``label``, and many
    clusters may share one name.
    """

    start_row: int
    end_row: int
    cluster_id: int

    @property
    def n_rows(self) -> int:
        """Inclusive row length."""
        return self.end_row - self.start_row + 1


@dataclass(frozen=True)
class MemberPlaces:
    """Where every member a cluster method labels sits, and how much time it stands for (§22).

    Indexed like ``member_leaves`` and ``instances_for``: one entry per member, whether a member is
    an analysed frame or a bout. It is what the node cache sums per leaf, so a node's time, its
    occupancy per recording and its sample can all be answered with no cut.

    ``time_ns`` is **measured**: a frame's is the gap to the next analysed frame inside its
    continuity block, so a fit analysed at every *n*-th frame counts *n* frames of time per member,
    and a bout's is its span plus its measured frame step. Never a count divided by an advertised
    fps, which over-reports whenever a file's header disagrees with its timestamps.
    """

    recording_ids: tuple[str, ...]
    recording_index: np.ndarray  # (M,) int64 into ``recording_ids``
    subject: np.ndarray  # (M,) int64
    start_ns: np.ndarray  # (M,) int64
    end_ns: np.ndarray  # (M,) int64, inclusive: a frame's own t_ns
    time_ns: np.ndarray  # (M,) int64


#: How a node's clips are scored for typicality (``ClipScoring.kind``).
#:
#: - ``distance``: every member carries a distance that does not depend on the node (a codebook
#:   decode's distance to the codeword it was decoded into), so a clip's score is the mean over
#:   its members and the cache keeps a sum per leaf run.
#: - ``centroid``: a clip's score is ``1 - cos`` between its members' mean vector and the node's
#:   centroid, so the cache keeps a vector sum per leaf run and per leaf.
#: - ``medoid``: the score depends on the node and is not additive (a distance to the node's
#:   medoid); clips are ranked on demand from the fit, by ``rank_members``.
#: - ``none``: the method offers no typicality; clips carry ``NaN``.
CLIP_SCORING_KINDS = ("distance", "centroid", "medoid", "none")


@dataclass(frozen=True)
class ClipScoring:
    """What the node cache needs from a cluster method to list and score every node's clips (§22).

    A node's clips are its actions: each leaf run under it on its own, never joined with the runs
    beside it, kept when it holds at least ``min_members`` members, the rule the method's own
    instances follow per run. A leaf run is a maximal stretch of touching members in one leaf when
    the method's instances are runs of like-labelled members (``join``), and one member (a bout)
    otherwise. Member-indexed arrays are indexed like ``member_leaves``.
    """

    kind: str  # one of CLIP_SCORING_KINDS
    #: The fewest members a clip holds; the method's minimum instance length, in members.
    min_members: int = 1
    #: Whether a leaf's touching members make one run (``instances_are_runs``).
    join: bool = True
    #: ``(M,)`` bool, the members that start one of the continuity blocks the method's
    #: ``instances_for`` cuts runs on, under the run's params (its gap threshold included). The
    #: cache never lets a run cross one. ``None``: runs are cut where the members' measured time
    #: breaks (``member_places``).
    member_block_start: np.ndarray | None = None
    #: ``kind == "distance"``: ``(M,)`` each member's distance, smaller is more typical.
    member_distance: np.ndarray | None = None
    #: ``kind == "centroid"``: ``(lo, hi, vectors)`` blocks covering every member once, each
    #: ``vectors`` the ``(hi - lo, D)`` unit vectors of members ``lo..hi-1``. Blocks, so a whole
    #: study's vectors are never held at once.
    member_vectors: Iterable[tuple[int, int, np.ndarray]] | None = None
    #: ``kind == "centroid"``: ``(S, D)`` vectors representative of the members (the fit sample),
    #: from which the cache learns the reduced basis it stores sums in.
    vector_sample: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.kind not in CLIP_SCORING_KINDS:
            raise ValueError(
                f"unknown clip scoring {self.kind!r}: one of {', '.join(CLIP_SCORING_KINDS)}"
            )
        if self.kind == "distance" and self.member_distance is None:
            raise ValueError("distance scoring needs member_distance")
        if self.kind == "centroid" and (
            self.member_vectors is None or self.vector_sample is None
        ):
            raise ValueError("centroid scoring needs member_vectors and vector_sample")


@dataclass(frozen=True)
class ClusterTree:
    """The full merge order behind a fit's hierarchy — a dendrogram in scipy's sense (§22).

    A cluster method produces a handful of nested partitions and, between them, nothing: asking
    for a count it did not produce used to mean merging the closest pair repeatedly, from scratch,
    every time. This records that sequence **once**, so every granularity becomes a lookup.

    **Leaves are the finest partition's clusters, not rows.** A fit sampled at 6,000 rows has on
    the order of a thousand leaves, so the tree is small and independent of how much data was
    fitted; ``leaf_of_row`` is the map back. Node ids follow scipy's convention — ``0..K-1`` are
    the leaves, merge ``m`` creates node ``K + m``, and the root is ``2K - 2`` — so a node id is
    stable for the life of a fit and does not move when the tree is cut somewhere else. That is
    what a name can be attached to; a cluster id at a granularity is not, because the refinement
    that produces it renumbers on every merge.

    **Merges are stored transition by transition, not sorted by height.** Each is constrained to
    stay inside one group of the next-coarser partition, which is what makes the method's own
    partitions land exactly on merge boundaries (``rung_merges``). Because every cut is then a
    prefix of one sequence, **every cut refines every coarser cut** — a property the previous
    refinement did not have (the removed ``api/pose_clustering.py`` recorded a real fit where 50
    refines 12 and 30 does not). The cost is that heights are only non-decreasing within a
    transition, so a renderer must not assume a globally monotone dendrogram.
    """

    children: np.ndarray  # (M, 2) int32, the two node ids each merge joined
    height: np.ndarray  # (M,) float64, average linkage at the merge
    size: np.ndarray  # (2K-1,) int32, leaves under each node
    n_rows: np.ndarray  # (2K-1,) int32, fitted rows under each node
    leaf_of_row: np.ndarray  # (rows,) int32, the finest partition
    rung_merges: tuple[
        int, ...
    ]  # merges completed at each of the method's own partitions

    @property
    def n_leaves(self) -> int:
        """Leaves in the tree, which is the finest partition's cluster count."""
        return (len(self.size) + 1) // 2

    @property
    def n_nodes(self) -> int:
        """Leaves plus internal nodes."""
        return len(self.size)

    @property
    def rungs(self) -> tuple[int, ...]:
        """The method's own partition sizes, finest first."""
        return tuple(self.n_leaves - merges for merges in self.rung_merges)

    @property
    def parent(self) -> np.ndarray:
        """Each node's parent, ``-1`` at the root. Derived in one pass over the merges.

        Not stored: it is ``children`` read the other way round, and two arrays saying one thing is
        how they come to disagree.
        """
        out = np.full(max(self.n_nodes, 0), -1, dtype=np.int64)
        for merge in range(len(self.children)):
            out[self.children[merge, 0]] = self.n_leaves + merge
            out[self.children[merge, 1]] = self.n_leaves + merge
        return out

    def deepest_named(self, node: int, named: set[int]) -> int:
        """The nearest member of ``named`` at or above ``node``, or ``-1``.

        The resolution rule for a name on a node: a leaf's name is its **deepest** named ancestor,
        so a coarse name ("immobility") and a fine one ("rearing left") coexist without either
        having to know about the other. Walking up from the node means the first hit *is* the
        deepest, so there is nothing to compare.

        Takes a set of ids rather than the name map, because whose names they are is not the
        tree's business.
        """
        parent = self.parent
        current = int(node)
        while current >= 0:
            if current in named:
                return current
            current = int(parent[current])
        return -1

    def leaf_nodes_at(self, k: int) -> np.ndarray:
        """The node id each leaf belongs to once the tree is cut to ``k`` clusters."""
        leaves = self.n_leaves
        applied = leaves - max(1, min(int(k), leaves))
        parent = np.arange(max(self.n_nodes, 1), dtype=np.int64)
        for merge in range(applied):
            node = leaves + merge
            parent[int(self.children[merge, 0])] = node
            parent[int(self.children[merge, 1])] = node

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        return np.array([find(leaf) for leaf in range(leaves)], dtype=np.int64)

    def cut_at(self, k: int) -> np.ndarray:
        """The leaf → cluster labelling with ``k`` clusters, compact and finest-first stable.

        Out-of-range counts clamp rather than raise: one cluster is the floor and the leaf
        partition is the ceiling, which is what asking for more clusters than the fit found means.
        """
        roots = self.leaf_nodes_at(k)
        order = {root: i for i, root in enumerate(sorted(set(roots.tolist())))}
        return np.array([order[root] for root in roots.tolist()], dtype=np.int64)

    def nodes_at(self, k: int) -> np.ndarray:
        """The node ids of the ``k`` clusters, in the same order ``cut_at`` labels them."""
        roots = self.leaf_nodes_at(k)
        return np.array(sorted(set(roots.tolist())), dtype=np.int64)

    def leaf_groups(self, named: set[int]) -> np.ndarray:
        """Each leaf's deepest ancestor in ``named`` (itself included), or ``-1``.

        ``deepest_named`` for every leaf at once: the partition a set of labelled nodes draws over
        the tree, which is what a method's ``instances_for`` takes. One pass from the root down,
        because a merge's node id is always above its children's, so a parent is resolved before
        its children are reached.
        """
        owner = np.full(max(self.n_nodes, 0), -1, dtype=np.int64)
        parent = self.parent
        for node in range(self.n_nodes - 1, -1, -1):
            if node in named:
                owner[node] = node
            elif parent[node] >= 0:
                owner[node] = owner[parent[node]]
        return owner[: self.n_leaves]

    def leaves_under(self, node: int) -> np.ndarray:
        """Every leaf beneath ``node``, ascending. A leaf is beneath itself."""
        leaves = self.n_leaves
        start = int(node)
        if start < leaves:
            return np.array([start], dtype=np.int64)
        found: list[int] = []
        stack = [start]
        while stack:
            current = stack.pop()
            if current < leaves:
                found.append(current)
            else:
                stack.extend(int(child) for child in self.children[current - leaves])
        return np.array(sorted(found), dtype=np.int64)


@dataclass(frozen=True)
class GranularitySpec:
    """What a cluster method's granularity knob *is*, so the module renders it (§22).

    §22 requires this to be method-declared rather than assumed: "a FINCH level, an HDBSCAN
    ``min_cluster_size``, and a ``k`` are not the same knob, and the module renders whatever the
    method declares rather than assuming a number of clusters."

    ``rungs`` are the partitions the **method itself produced** — the ones it can defend, because
    nothing chose them. Everything else in ``[minimum, maximum]`` is a cut of the merge tree
    between two of them: legitimate, and ours rather than the algorithm's. That is the distinction
    now, and it is a claim about *provenance*.

    It used to be a claim about nesting, because it had to be: a setting between rungs was derived
    by refining afresh from whichever rung was tightest, so two of them could disagree about which
    cluster a row was in. :class:`ClusterTree` removed that — every cut is a prefix of one merge
    sequence, so **every setting nests in every coarser setting** and a rung is no longer the only
    place a name survives. What a rung still buys is the answer to "why this number", which is the
    question a reader of the method asks and the one no interpolation can answer.
    """

    kind: str  # "cluster_count" today; named so a future knob is not silently assumed to be one
    minimum: int
    maximum: int
    rungs: tuple[int, ...]

    def is_rung(self, value: int) -> bool:
        """Whether ``value`` is one of the nested settings."""
        return int(value) in self.rungs

    def nearest_rung(self, value: int) -> int:
        """The rung closest to ``value`` in log space.

        Log space because rungs fall geometrically — against 63, a rung at 12 and one at 290 are
        equally far in ratio, and picking by absolute difference would take the 12.
        """
        import math

        return min(
            self.rungs,
            key=lambda r: abs(math.log(max(r, 1)) - math.log(max(int(value), 1))),
        )
