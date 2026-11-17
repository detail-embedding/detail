"""Abstract protocols every backend implements against.

These ABCs are the seams the registries dispatch on (Tenet 1): detection
models, trainers, trackers, embedding models, and mask transforms. Concrete
implementations live in their own packages and self-register; this module
imports no backend.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

import numpy as np

from detail.core.dataclasses import (
    AuxInput,
    CheckResult,
    ClipScoring,
    ClusterTree,
    MemberPlaces,
    Cue,
    ExemplarBounds,
    FrameResult,
    GranularitySpec,
    InstallProgress,
    ModelLicense,
    ModelReadiness,
    NegativeSupport,
    PolygonKeyframes,
    Prompt,
    RecordingTarget,
    RenderContext,
    SamplingRequest,
    StreamImportResult,
    ToolResult,
    TrackingJob,
    TrackingProgressEvent,
    TrainingJob,
    TrainingProgressEvent,
)
from detail.core.keypoints import InstanceFrame, ScoreSpec, TrackPart
from detail.core.skeleton import Skeleton

if TYPE_CHECKING:
    # torch is referenced only in annotations. Guarding it here means it is
    # never imported at runtime (TYPE_CHECKING is False then), so importing this
    # module pulls in no backend — the lazy-backend guarantee (Tenet 13).
    import torch

    from detail.help import HelpTopic


class ModelNotReadyError(RuntimeError):
    """A model cannot run because its weights or credentials are not in place.

    Carries the ``ModelReadiness`` so callers (API, GUI, CLI) can surface the
    actionable message and ``action_url`` (Tenet 17).
    """

    def __init__(self, readiness: ModelReadiness) -> None:
        super().__init__(readiness.message)
        self.readiness = readiness

    @property
    def action_url(self) -> str | None:
        return self.readiness.action_url


def _ready() -> ModelReadiness:
    """Default readiness for a model with nothing to fetch."""
    return ModelReadiness(
        ready=True, reason="ok", message="", action_url=None, repo_id=None
    )


class VariantedModel:
    """Optional model-size/variant axis (Section 5).

    A model declares ``VARIANTS`` (variant name → backend weight identifier, e.g.
    a ``.pt`` filename or an HF repo) and a ``DEFAULT_VARIANT``; it is constructed
    with the chosen variant, exposed as ``self.variant`` with the resolved weight
    on ``self.weight``. Models with no size options leave ``VARIANTS`` empty and
    are constructed with ``variant=None``.
    """

    VARIANTS: dict[str, str] = {}
    DEFAULT_VARIANT: str | None = None
    architecture: str = "—"  # family/backbone tag (yolo / sam / convnext) for catalogs

    def __init__(self, variant: str | None = None) -> None:
        variants = type(self).VARIANTS
        model_id = getattr(self, "model_id", "?")
        if not variants:
            if variant is not None:
                raise ValueError(
                    f"model {model_id!r} has no variants; got variant={variant!r}"
                )
            self.variant: str | None = None
            self.weight: str | None = None
            return
        resolved = variant or self.DEFAULT_VARIANT or next(iter(variants))
        if resolved not in variants:
            raise ValueError(
                f"unknown variant {resolved!r} for model {model_id!r}; "
                f"available: {sorted(variants)}"
            )
        self.variant = resolved
        self.weight = variants[resolved]

    @classmethod
    def variants(cls) -> list[str]:
        """Available variant names (empty if the model has no size options)."""
        return list(cls.VARIANTS)

    @classmethod
    def default_variant(cls) -> str | None:
        if not cls.VARIANTS:
            return None
        return cls.DEFAULT_VARIANT or next(iter(cls.VARIANTS))


class Model(VariantedModel, ABC):
    """What every model shares (§5 "Model protocol"): identity, the backends it runs on,
    the install / load contract, and the two hooks §26 generalised.

    ``DetectionModel``, ``EmbeddingModel`` (§14) and ``KeypointModel`` (§26) each add their
    own ``predict``. This base is the only place the install contract is written — the three
    copies it replaced (one per protocol, the third on #626) are the reason it exists.
    """

    model_id: str
    supported_backends: list[str]  # e.g. ["cuda", "cpu"] — for device resolution
    # Shown before install when the weights' terms restrict use. None = nothing to show.
    license: ModelLicense | None = None
    # Another model_id whose download this one reuses (§26: dinov3_match on dinov3_vit).
    shares_weights_with: str | None = None

    def check_backend(self, backend: str) -> None:
        """Raise when this *install* cannot run on a backend the class supports (a
        CPU-only torch asked for CUDA), so ``validate`` refuses before the run starts.
        ``supported_backends`` is the class's claim; this is the machine's. Default: nothing.
        """

    def resolved(self) -> dict:
        """What ``load()`` decided that the config does not say, written to the run record
        under ``resolved`` (Tenet 9). Default: ``{}``."""
        return {}

    @abstractmethod
    def load(self, device: str) -> None:
        """Load present weights into memory on the resolved device; call
        ``model.eval()`` before returning. Assumes ``install()`` has run.

        **Never touches the network** (§5): resolve the weights to a local path
        under ``model_cache_dir()`` and pass it to the backend explicitly
        (``cache_dir=`` + ``local_files_only=True``, or a resolved checkpoint path).
        This runs inside a pool worker subprocess with no progress channel, so a
        download here is invisible by construction; raise instead and let the
        pre-run readiness check point the user at ``install()``.
        """

    def install(
        self, progress_cb: Callable[[InstallProgress], None] | None = None
    ) -> None:
        """Make weights present locally (the one-time download). Idempotent.

        Default is a no-op; weight-bearing models inherit a source mixin (e.g.
        ``HFHubModel``) that overrides this (Section 5).
        """

    def readiness(self) -> ModelReadiness:
        """Cache-first install/credential status. Default: ready (nothing to fetch)."""
        return _ready()

    def is_installed(self) -> bool:
        """Are the weights present on this disk? A pure disk check, never network.
        Default: True (nothing to fetch)."""
        return True

    def is_ready(self) -> bool:
        return self.readiness().ready


class DetectionModel(Model):
    """Detect objects in a frame given a prompt (Section 5)."""

    supported_cue_types: list[type[Cue]]  # [] = closed set; [TextCue] = prompt-capable
    supports_batching: bool = (
        False  # True → predict_batch runs one batched forward pass
    )

    def set_conf_threshold(self, threshold: float) -> None:
        """Push the run's confidence threshold into the model (Section 5).

        Optional: backends with an internal score threshold (e.g. SAM3 filters in
        ``Sam3Processor.confidence_threshold``) override this to make the
        caller's threshold actually effective. Default no-op — backends without
        an internal threshold rely on the pipeline's post-predict filter. Must
        be called after ``load()``.
        """

    @abstractmethod
    def predict(self, frame: np.ndarray, prompt: Prompt) -> FrameResult:
        """Run one forward pass; must be wrapped in ``torch.inference_mode()``.
        Returns the shared row container (§5 "Standardized output"); build it with
        ``FrameResult.from_detections``."""

    @abstractmethod
    def predict_batch(
        self, frames: list[np.ndarray], prompts: list[Prompt]
    ) -> list[FrameResult]:
        """Batched forward pass; ``inference_mode`` + CUDA ``autocast`` per Section 5."""

    def prepare_batch(self, frames: list[np.ndarray], prompts: list[Prompt]) -> object:
        """The CPU half of ``predict_batch``: whatever the model does to a batch before it
        touches the device (a letterbox, a stack). Must need no GPU and be safe to run on a
        thread other than the one calling ``run_prepared``.

        The detection worker runs it a batch ahead on its own thread, so the GPU does not wait
        on it (§7 *Execution*). A model that splits must keep ``predict_batch(f, p) ==
        finish_batch(run_prepared(prepare_batch(f, p)))`` exactly. Default: no CPU half.
        """
        return frames, prompts

    def run_prepared(self, prepared: object) -> object:
        """The device half, given ``prepare_batch``'s output. **Every device-to-host copy
        happens here**, on the thread that runs the model: a copy made on another thread would
        queue behind the next batch's forward on the same CUDA stream. Default:
        ``predict_batch``, the whole call when there is no CPU half.
        """
        frames, prompts = prepared
        return self.predict_batch(frames, prompts)

    def finish_batch(self, raw: object) -> list[FrameResult]:
        """The host half: ``run_prepared``'s host arrays to one ``FrameResult`` per frame, off
        the model's thread (the worker's writer). Default: nothing left to do."""
        return raw


class Trainer(ABC):
    """Train a custom detection model from a labeled dataset (Section 12)."""

    trainer_id: str
    model_family: str
    supported_cue_types: list[type[Cue]]
    supported_backends: list[str]
    # Pydantic model for this trainer's hyperparameters (§12) — the single source
    # for their type/bounds/default/help/label; feeds validate_hyperparams, the
    # GUI form, and the CLI. None until a trainer declares one.
    config_model: type | None = None

    @abstractmethod
    def is_ready(self) -> bool:
        """Base weights downloadable/present and runtime installed."""

    @abstractmethod
    def prepare_dataset(self, dataset_dir: Path, work_dir: Path) -> Path:
        """Convert/validate the dataset into this trainer's format; cache in work_dir."""

    @abstractmethod
    def train(
        self,
        job: TrainingJob,
        dataset_dir: Path,
        model_dir: Path,
        device: str,
        progress_cb: Callable[[TrainingProgressEvent], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None: ...

    @abstractmethod
    def resume(
        self,
        job: TrainingJob,
        dataset_dir: Path,
        model_dir: Path,
        device: str,
        progress_cb: Callable[[TrainingProgressEvent], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None: ...

    @abstractmethod
    def checkpoint_path(self, model_dir: Path) -> Path | None:
        """Last checkpoint available for resume, or None."""

    @abstractmethod
    def best_weights_path(self, model_dir: Path) -> Path | None:
        """Best weights after training completes, or None."""

    @abstractmethod
    def final_metrics(self, model_dir: Path) -> dict[str, float]:
        """Trainer-specific metrics dict — recorded into model.yaml on completion."""

    @abstractmethod
    def metric_history(self, model_dir: Path) -> list[dict[str, float]]:
        """One row per finished epoch, oldest first — what :meth:`final_metrics` summarises.

        The pair matters: ``final_metrics`` answers *how good is this model*, and a user deciding
        whether to stop training early is asking the other question — *is it still getting
        better*. A single number cannot answer that, which is why a run that had been improving
        for an hour and a run that plateaued forty epochs ago looked identical on this page.

        Read from the trainer's own on-disk record rather than accumulated from progress events,
        so the history is whole on a page opened after the run started, and survives the app
        closing. Rows carry ``epoch`` plus whatever metrics the trainer has; a reader takes the
        keys it knows and must tolerate their absence (a detection-only run has no mask mAP).
        Empty when the run has not finished an epoch yet.
        """

    @abstractmethod
    def validate_hyperparams(self, hyperparams: dict) -> dict:
        """Fill defaults; raise ValueError on unknown or wrong-type keys."""

    @abstractmethod
    def resolved_settings(self, model_dir: Path) -> dict:
        """What the backend actually trained with, recorded as ``model.yaml → trained_with``.

        ``hyperparams`` are the few settings this tool asks for; the backend resolves every
        other one from its own defaults, which change between its releases. So the record
        carries the backend's name, its version, and its full resolved settings — the part of
        "what trained this model" nobody typed. Values that are paths on this machine are left
        out (no absolute paths in experiment files). ``{}`` when the backend wrote nothing.
        """


class Tracker(ABC):
    """Assign a consistent object_id to detections across frames (Section 13)."""

    tracker_id: str
    supported_backends: list[str]

    @abstractmethod
    def validate_hyperparams(self, hyperparams: dict) -> dict:
        """Fill defaults; raise ValueError on unknown or wrong-type keys."""

    @abstractmethod
    def track(
        self,
        job: TrackingJob,
        detection_run_dir: Path,
        output_dir: Path,
        arena_polygons: dict[str, PolygonKeyframes] | None,
        part_frame_offsets: list[int] | None = None,
        progress_cb: Callable[[TrackingProgressEvent], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Link detections into tracks and write the assignment + summary zarrs.

        ``part_frame_offsets[i]`` is the recording-global frame index of part i's
        frame 0, so a tracker uses ``part_frame_offsets[i] + local_frame`` for
        arena polygon lookup (Section 10). ``None`` means part-local indexing
        only — appropriate for single-part recordings or no-arena runs.
        """


class EmbeddingModel(Model):
    """Embed a rendered image into a feature vector (Section 14). The install / load
    contract is ``Model``'s."""

    output_dim: int
    # The square pixel size the model consumes (its image processor resizes any
    # input to this). The renderer targets it directly so the model isn't handed a
    # large canvas it re-downscales every frame (§14). Known without loading weights.
    input_size: int = 224

    @abstractmethod
    def embed(self, image: np.ndarray) -> np.ndarray:
        """Single image (H,W,C) -> (D,). Wraps ``embed_batch`` internally."""

    @abstractmethod
    def embed_batch(self, images: torch.Tensor) -> torch.Tensor:
        """(B,H,W,C) -> (B,D); ``inference_mode`` + CUDA ``autocast`` per Section 14."""

    def prepare_batch(self, images: list[np.ndarray]) -> object:
        """The CPU half of ``embed_batch``: whatever the model does to a batch before it
        touches the device (an image processor's resize / rescale / normalise). Must need no
        GPU and be safe to run on a thread other than the one calling ``embed_prepared``.

        The embedding worker runs it a batch ahead on its own thread, so the GPU does not wait
        on it (§14 *Execution*). A model that splits must keep
        ``embed_batch(x) == embed_prepared(prepare_batch(x))`` exactly. Default: no CPU half.
        """
        return images

    def embed_prepared(self, prepared: object) -> torch.Tensor:
        """The device half of ``embed_batch``, given ``prepare_batch``'s output. Default:
        ``embed_batch`` on it, which is the whole call when ``prepare_batch`` is the identity.
        """
        return self.embed_batch(prepared)


class PatchFeatures(ABC):
    """Dense per-patch features, a capability an embedding model declares (§14).

    A patch-matching keypoint method (§26) needs the backbone's per-patch features, not the
    pooled vector an embedding reads out. Same model, same weights, so this is mixed into the
    ``EmbeddingModel`` that offers it rather than registered as a second model: one download
    is listed once and read for both uses.
    """

    patch_size: ClassVar[int]  # an image's sides must be multiples of it

    @abstractmethod
    def patch_grids(self, images: list[np.ndarray]) -> list:
        """One L2-normalised feature grid ``(C, H/patch_size, W/patch_size)`` per RGB uint8
        image — the final layer's patch tokens after the model's final LayerNorm, CLS and
        register tokens dropped, never resized (a patch is exactly ``patch_size`` image
        pixels). fp16 on CUDA, fp32 elsewhere. After ``load()``; never touches the network.
        """


class KeypointModel(Model):
    """Place a skeleton's points on the animals in a frame (§26 "KeypointModel protocol").
    Never assigns a subject: that is ``keypoint_track``'s first step. The install / load
    contract, ``check_backend`` and ``resolved`` are ``Model``'s.
    """

    needs_examples: ClassVar[bool] = False  # built with an example set (dinov3_match)
    # True: one instance per detection it is handed (dinov3_match); False: the whole frame,
    # every animal it finds (the student). No default — each method states which it is.
    per_detection: ClassVar[bool]
    needs_mask: ClassVar[bool] = (
        False  # matching restricted to the detection's mask (dinov3_match)
    )
    n_candidates: ClassVar[int] = 1  # candidates per point the detect stage stores
    # each instance carries a confidence min_pose_confidence applies to (the student)
    reports_confidence: ClassVar[bool] = False
    score_spec: ClassVar[ScoreSpec]
    default_track_method: ClassVar[str]  # the KeypointTrackMethod this output suits

    @abstractmethod
    def skeleton(self) -> Skeleton:
        """Fixed when the model is built: the example set's, or the model's own."""

    @abstractmethod
    def predict_batch(self, frames: list[InstanceFrame]) -> list[FrameResult]:
        """One ``FrameResult`` per ``InstanceFrame``, one row per instance found, in order;
        ``track_id`` unset; a per-detection method sets each row's ``detection_row``; every
        forward pass under ``torch.inference_mode()``."""

    def prepare_batch(self, frames: list[InstanceFrame]) -> object:
        """The CPU half of ``predict_batch``: whatever the method does to a batch before it
        touches the device (a letterbox, crops). Must need no GPU and be safe to run on a
        thread other than the one calling ``run_prepared``.

        The detect stage runs it a batch ahead on its own thread, so the GPU does not wait on
        it (§26 *Execution*). A method that splits must keep ``predict_batch(x) ==
        finish_batch(run_prepared(prepare_batch(x)))`` exactly. Default: no CPU half.
        """
        return frames

    def run_prepared(self, prepared: object) -> object:
        """The device half, given ``prepare_batch``'s output. **Every device-to-host copy
        happens here**, on the thread that runs the model: a copy made on another thread would
        queue behind the next batch's forward on the same CUDA stream. Default:
        ``predict_batch``, the whole call when there is no CPU half.
        """
        return self.predict_batch(prepared)

    def finish_batch(self, raw: object) -> list[FrameResult]:
        """The host half: ``run_prepared``'s host arrays to one ``FrameResult`` per frame, off
        the model's thread (the stage's writer). Default: nothing left to do."""
        return raw

    def set_min_pose_confidence(self, threshold: float) -> None:
        """Apply the run's ``min_pose_confidence`` (§26 *Visibility*): instances below it are
        not returned. Only a method that ``reports_confidence`` takes one; the detect stage
        calls this after ``load()`` and never on any other method."""
        raise TypeError(
            f"{type(self).__name__} reports no instance confidence; min_pose_confidence "
            "applies only to a method that does"
        )


class KeypointTrackMethod(ABC):
    """Turn a detect run's per-frame placements into final points over time (§26 "Track
    methods"). Owns its filtering: no filter runs in the detect stage, and none applies to a
    method that did not declare it."""

    method_id: str
    config_model: type  # pydantic (§6.5); every field is identity
    needs_candidates: ClassVar[bool] = (
        False  # refused on a detect run with n_candidates = 1
    )

    @abstractmethod
    def track(
        self,
        rows: dict,
        skeleton: Skeleton,
        config,
        *,
        rows_per_second: float,
        frame_wh: tuple[int, int],
        body_scale_px: np.ndarray,
        phase: Callable[[str], None] | None = None,
    ) -> TrackPart:
        """One subject's part after *Assignment* (§26), pure arrays in and out: ``rows`` has a
        row per subject-frame at the run's cadence — ``frame_idx``, ``t_ns``,
        ``instance_row``, ``points``, ``point_scores``, and ``candidates`` /
        ``candidate_scores`` / ``extras/<name>`` when the detect run stored them — NaN where
        no instance was assigned (``instance_row`` −1). ``config`` is the method's validated
        model, ``body_scale_px`` the subject's scale per row; the stage reads and writes.
        ``phase`` names each step as it starts (``decoding``, ``filtering``) so a long pass
        is never silent.
        """


# -- keypoints (§26): PR 6/8 -- nothing; the keypoint protocols end here --


class MaskTransform(ABC):
    """One step in a MaskRenderer pipeline (Section 14). Must be picklable."""

    # True if ``apply`` *requires* ``ctx.frame`` (the decoded video pixels). When a
    # renderer's transforms all leave this False (the silhouette constructors build
    # from ``ctx.mask``), the embedding worker skips video decode entirely and passes
    # ``frame=None`` (§14). Only ``ApplyMask`` — which reads the frame's colors — sets
    # it True; transforms that merely crop the frame when present do not.
    reads_frame: ClassVar[bool] = False

    @abstractmethod
    def apply(self, ctx: RenderContext) -> RenderContext: ...

    @abstractmethod
    def to_spec(self) -> dict:
        """Serialize to ``{"type": "...", param: value, ...}``."""


class StreamImporter(ABC):
    """Convert an external time-series file to canonical stream rows (Section 3).

    One adapter per device/format; adding support for a new source is a new
    subclass that self-registers (Tenet 1). The importer **parses** to canonical
    rows; the storage layer writes the canonical CSV and the recording.yaml manifest.

    Two declared axes classify what it produces (Section 3):
        ``role`` — the semantic meaning a consuming module checks (e.g.
        ``"behavior"`` for ground-truth labels); ``None`` for an untyped signal.
        ``clock`` — how the importer's native time reaches canonical video
        ``t_ns``: ``"stream_seconds"`` (the default — the importer emits ``t_ns``
        itself, aligned later by a read-time offset, e.g. MED-PC), ``"video_frames"``
        (native = the video frame index; the import layer resolves it via the
        recording's active timestamp map), or ``"video_seconds"`` (native = seconds
        on the video clock). ``stream_type`` is legacy; ``StreamImportResult.shape``
        supersedes it. ``aux_inputs`` declares extra files the importer needs
        beyond the primary data file (empty for single-file sources), keyed into
        the ``aux_paths`` dict passed to ``import_stream``.

    ``config_model`` is the pydantic model single-sourcing this importer's parameters
    (type/bounds/default/help/label) for the form, the CLI and the generated docs — the
    configs rule, the same channel ``Tool`` and ``ProcessingModule`` carry. ``None`` for an
    importer whose parsing is fully determined by the file (MED-PC, BORIS). Without it a
    configurable importer has nowhere to be told *how* to read a file, which is what forced
    the first column-mapping attempt to bypass this registry entirely — and with it went the
    picker, the CLI ``--importer`` id and the bundle manifest.
    """

    importer_id: str
    display_name: str
    stream_type: str
    role: str | None = None
    clock: str = "stream_seconds"
    aux_inputs: list[AuxInput] = []
    config_model: type | None = None
    #: Whether this importer's config can make it read a file nothing recognizes (a CSV whose
    #: columns are mapped by hand). The Add-stream dialog falls back to it for an unclaimed
    #: file. A config that only *picks within* a recognized file (MED-PC's session, BORIS's
    #: observation and subject) does not qualify, so this is declared, not inferred from
    #: ``config_model``.
    maps_unrecognized_files: bool = False

    @abstractmethod
    def can_import(self, path: Path) -> bool:
        """True if this importer recognizes the primary data file's format."""

    def resolve_clock(self, path: Path, config: dict | None = None) -> str:
        """The clock importing ``path`` with ``config`` would produce, without importing it.

        Defaults to the declared ``clock``. An importer whose clock depends on the file (a CSV
        whose time column is a frame index reports ``video_frames`` from a ``stream_seconds``
        importer) overrides it, so a surface gating alignment controls reads the clock the
        import will actually record rather than the class attribute (Section 3). Cheap: a
        header sniff at most.
        """
        return self.clock

    @abstractmethod
    def import_stream(
        self,
        data_path: Path,
        out_dir: Path,
        aux_paths: dict[str, Path],
        config: dict | None = None,
    ) -> StreamImportResult:
        """Parse the data file (+ any ``aux_paths``) into a ``StreamImportResult``.

        ``out_dir`` is provided for importers that must spill intermediate
        artifacts; the canonical CSV itself is written by the storage layer from
        the returned rows.

        ``config`` is this importer's ``config_model`` as a dict, or ``None`` when the
        caller supplied none — which every importer must handle, since it is what the
        bundle manifest and a bare ``add_stream`` pass. An importer with no
        ``config_model`` ignores it.

        Raises:
            ValueError: If the file cannot be parsed into the canonical form.
        """


class Tool(ABC):
    """A one-off, experiment-free operation — the extensible Tools framework (Section 27).

    A tool takes ad-hoc inputs (e.g. a video path), runs synchronously in-process, and
    writes managed outputs to a run directory the framework owns (never into an
    experiment). Adding a tool is a new subclass that self-registers (Tenet 1) plus a
    GUI ``ToolUI``; the shell hosts it generically.

    ``config_model`` is the pydantic model that single-sources this tool's scalar
    parameters (type/bounds/default/help/label) for the form + CLI (the configs rule).
    Non-scalar inputs (a video path, a ``Prompt``, a ``ComputeTarget``) are assembled by
    each surface and passed in ``config``.
    """

    tool_id: str  # registry key + GUI/CLI id
    label: str  # display name
    description: str  # one line, shown in the tools list
    category: str  # grouping in the tools list (e.g. "Detection & segmentation")
    config_model: type  # pydantic model for the scalar parameters

    @abstractmethod
    def run(
        self,
        config: dict,
        out_dir: Path,
        progress: Callable[[str, float, str], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> ToolResult:
        """Run the tool, writing outputs under ``out_dir``; return a ``ToolResult``.

        ``progress(stage, fraction, detail)`` must be driven inside any loop that can
        run longer than a moment (the long-op rule) — a silent long run is a bug,
        indistinguishable from a hang.

        ``stop_check()`` is polled in that same loop; a truthy return means the user asked
        to stop, and the tool must finish early and return a ``ToolResult`` describing what
        it *did* complete (partial output a surface can still open) rather than raising.
        A tool whose work is unbounded — Quick segment can be pointed at a whole
        multi-hour recording — is unusable without it: the only other way out is killing
        the app.
        """


class Check(ABC):
    """A single readiness / data-integrity check — the live checker framework (§1).

    A check answers one question ("is torch installed?", "does this recording's video
    still exist?") and returns a ``CheckResult``. Adding one is a new subclass that
    self-registers (Tenet 1); the runner + surfaces host it generically.

    ``isolation`` decides HOW the runner executes it. ``"inproc"`` checks run in the app's
    interpreter (disk / credential / state probes). ``"subprocess"`` checks run in a
    throwaway interpreter (import-sensitive probes — torch/CUDA, ``import sam3``), so a
    mid-session ``pip install`` is reflected on the next re-check **without restarting the
    app**: a C-extension like torch can't be hot-swapped in a running process, so an
    in-process re-check would only re-read the already-loaded module.

    ``fix`` points at the docs page that explains how to resolve a failure. It is advisory:
    a check detects and points at the fix; it never performs it (v1). ``target_kind`` is
    ``"machine"`` for global checks; ``"recording"`` for a per-recording check, whose ``run``
    receives the ``RecordingTarget`` it applies to (machine checks ignore the argument, and
    the runner enumerates recordings for per-recording checks).
    """

    check_id: str  # registry key + id
    label: str  # display name
    group: str  # grouping in the surface (e.g. "Environment", "Hugging Face")
    target_kind: str = "machine"  # "machine" (global) | "recording" (per recording)
    isolation: str = "inproc"  # "inproc" | "subprocess"
    fix: HelpTopic | None = None  # docs pointer shown for a failing check

    @abstractmethod
    def run(self, target: "RecordingTarget | None" = None) -> CheckResult:
        """Run the check and return its result. Machine checks take no target; a
        ``target_kind == "recording"`` check receives the ``RecordingTarget`` it applies to.
        Must not raise: a probe that cannot answer returns ``CheckStatus.UNKNOWN`` (the
        runner also guards this boundary).
        """


class QueryMethod(ABC):
    """Search a trajectory for stretches resembling user-marked example clips (§20).

    Deliberately lighter than ``DetectionModel``: no weights, no install, no device
    management. The same shape as ``Tracker`` and ``TRACKERS`` — the Query module needs no
    change when a method is added.

    **The module never looks inside a ``SearchResult``.** It is whatever the method needs,
    the method owns its serialization, and the module stores it, shards it, and hands it
    back. What the module *does* read is ``instances_at`` — where a cutoff becomes instances,
    and therefore where extent is decided. A windowed matcher returns windows near the
    exemplar's length; a per-frame method thresholds and run-length-encodes, so its extents
    follow the behavior. Both are correct for their method and neither is the module's
    business, which is why a method **documents how it defines extent**: that definition is
    what a duration measure will report.

    **No parameter carries a label.** ``prepare`` takes user-chosen exemplars and a
    module-drawn sample; ``search`` takes a trajectory. There is no argument through which
    ground truth could enter — the leakage rule made structural rather than conventional.
    """

    method_id: str
    config_model: type  # its own pydantic parameters, single-sourced (§6.5)
    sampling: SamplingRequest | None = None
    exemplars: ExemplarBounds
    negatives: NegativeSupport
    # The extension `serialize` writes. Declared rather than implied because a serializer may
    # append its own (``np.savez_compressed`` adds ``.npz``), and a module that built the path
    # without it would write one file and read another. No default: a method that forgets it
    # must fail the registry guard, not silently inherit "".
    shard_suffix: str
    # What the artifact's score column means, in the method's own words. §19 carries "the
    # method's confidence with its declared meaning", and the declaring belongs here rather
    # than in the module — a module writing that sentence on a method's behalf is guessing at
    # what the number it never looks inside actually measures.
    #
    # Defaulted rather than abstract because `confidence` is already canonical (`[0, 1]`,
    # higher better, uniform across methods), so the general sentence is *true* of every
    # method. A method overrides it to say what its confidence is derived from.
    score_meaning: str = (
        "confidence that the stretch is an instance of the behavior: [0, 1], higher is better"
    )

    @abstractmethod
    def prepare(
        self,
        exemplars: Sequence[np.ndarray],
        params: dict,
        *,
        negatives: Sequence[np.ndarray] | None = None,
        sample: np.ndarray | None = None,
    ) -> object:
        """Everything derivable from the inputs alone, computed once per run.

        ``exemplars`` are the user's marked clips as ``(rows, dim)`` embedding blocks;
        ``negatives`` their marked non-examples; ``sample`` the module-drawn background.
        Returns opaque state the module carries to every ``search`` call.
        """

    @abstractmethod
    def search(
        self,
        state: object,
        trajectory: np.ndarray,
        runs: Sequence[tuple[int, int]],
        params: dict,
        *,
        row_offset: int = 0,
        progress_cb: Callable[[float], None] | None = None,
        min_rows: int = 0,
    ) -> object:
        """Search one unit (or one chunk of one) and return a ``SearchResult``.

        ``runs`` are half-open continuous-row ranges a match may not cross, so a match never
        spans a tracking gap. ``row_offset`` is the chunk's position in the whole trajectory;
        the method adds it before returning, so a merged result never mixes coordinate
        systems.

        ``min_rows`` is the run's minimum match length (``QueryConfig.min_match_s``), already
        converted to this unit's rows. No match shorter than it is ever returned, and it is
        applied **while matches are formed**, not filtered afterwards: a too-short span must
        not suppress a valid neighbour before being dropped itself. ``0`` = no minimum.

        **No cutoff is applied here.** The search config carries no threshold at all (§20),
        so there is no way to spend hours searching and then find that a change of mind costs
        another search.
        """

    @abstractmethod
    def instances_at(self, result: object, cutoff: float, params: dict) -> list:
        """The instances a confidence ``cutoff`` yields, as ``QueryCandidate``s.

        Called across the confidence range to build the calibration curve, so it must be
        cheap. Instance count is **not guaranteed monotonic** in the cutoff — a method that
        re-forms extents may split one instance into two as the cutoff tightens.
        """

    @abstractmethod
    def merge(
        self, chunk_results: Sequence[object], state: object, params: dict
    ) -> object:
        """Combine chunk results into one, deduplicating across the overlap.

        The method's job rather than the module's, because what counts as a duplicate is
        method knowledge — and the module still never looks inside a result.

        Takes ``state`` because the deduplication rule usually has to be the *same* rule the
        in-chunk search applied, and for a windowed method that rule is derived from the
        exemplars. Reconstructing it from the chunk results alone would let the seam be
        judged more loosely than the interior, which is the chunk-equivalence failure.
        """

    @abstractmethod
    def chunk_overlap_rows(self, state: object, params: dict) -> int | None:
        """Rows of overlap a split needs, or ``None`` when the method cannot be split.

        A method whose scoring is row-local returns ``0``; a windowed method returns its
        longest window so no match is cut. It takes ``state`` because the longest window is
        usually a property of the exemplars, not of the class.

        The module decides *whether* to split, from worker count and unit imbalance. This
        also bounds memory: an unsplittable method is the one that can exhaust it on a long
        recording, which the module then knows up front rather than at hour six.
        """

    def candidate_cap(self, params: dict) -> int | None:
        """The most instances ``instances_at`` will return for one unit, or ``None`` if
        unbounded.

        The calibration curve needs this and cannot infer it: §20 warns that "a curve that
        flattens because storage ran out looks exactly like a curve that flattens because the
        animal stopped". A method that truncates says so here, and the curve marks the points
        the truncation bound rather than leaving a reader to misread a ceiling as a finding.

        Concrete rather than abstract, defaulting to "no cap", because a method with no limit
        answers this correctly by saying nothing.
        """
        return None

    @abstractmethod
    def serialize(self, result: object, path: Path) -> None:
        """Write a ``SearchResult`` to ``path``. The extension is the method's."""

    @abstractmethod
    def load(self, path: Path) -> object:
        """Read back what ``serialize`` wrote."""


class PoseClusterMethod(ABC):
    """Discover a vocabulary of **postures** from pose trajectories, and label frames with it (§22).

    Deliberately separate from a temporal (bout) clustering protocol rather than one interface with
    a mode flag. §22: "The shapes are identical enough that sharing is tempting, and that is the
    trap: a shared interface would let a bout method be selected in the pose module, where it
    answers a different question." A cluster here is a posture and an instance is a run of frames
    holding it; for the temporal module a cluster is a behavior and an instance is a segment of
    movement.

    Lighter than ``DetectionModel`` in the same way ``QueryMethod`` is: no weights, no install, no
    device management.

    **The module never looks inside a fit.** It is whatever the method needs — a codebook, a
    neighbour graph, a hierarchy — the method owns its serialization, and the module stores it and
    hands it back. What the module reads is ``instances_at``, where the authored granularity becomes
    instances, and therefore where the temporal decode and the minimum duration live. A frame-level
    labeling flickers, and suppressing that is the method's business, not a module-level cleanup
    pass.

    **No parameter carries a label.** ``fit`` takes trajectories and config; there is no argument
    through which ground truth could enter. The leakage rule made structural rather than
    conventional — a posture vocabulary a user could not reproduce is not a vocabulary.
    """

    method_id: str
    config_model: type  # its own pydantic parameters, single-sourced (§6.5)
    # What the authored knob is, its type and its range — rendered by the module rather than
    # assumed to be a cluster count (§22).
    granularity: GranularitySpec
    # Whether this rule leaves members unassigned. §19's measures report "noise" and "unnamed"
    # separately, and a method with `has_noise = False` reports full assignment by construction —
    # which must not be read as more complete than one that admits it leaves frames out.
    has_noise: bool
    # Reserved: apply-to-new-recordings is out of the MVP (§22), but declaring the capability now
    # costs nothing where retrofitting a base contract later is not free.
    supports_apply: bool = False
    # The extension `serialize` writes. Declared rather than implied because a serializer may append
    # its own (`np.savez_compressed` adds `.npz`), and a module that built the path without it would
    # write one file and read another.
    shard_suffix: str
    #: Whether an instance is a **run of like-labelled items** the method built, rather than an
    #: extent it found. When it is, two adjacent instances the user has given one name become one
    #: behavior; when it is not, they stay two, because joining them would be a claim the method
    #: did not make. Declared rather than inferred: the module cannot tell the two apart by
    #: looking, and guessing wrong silently rewrites what the artifact says happened.
    instances_are_runs: bool = True
    #: Importable modules the method needs beyond the base package (an optional extra's backend).
    #: Checked at ``validate`` so a missing extra is an actionable refusal before any work, not an
    #: ImportError in a worker after loading every trajectory.
    requires: tuple[str, ...] = ()

    @abstractmethod
    def fit(
        self,
        trajectories: Sequence[tuple[str, int, np.ndarray, np.ndarray]],
        params: dict,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> object:
        """Find the vocabulary. Computed once, expensive, immutable (§21).

        ``trajectories`` is one entry per subject: ``(recording_id, subject, features, t_ns)``,
        where ``features`` is ``(rows, dim)`` and ``t_ns`` is the video time of each row. The
        method keeps whatever it needs to resolve a member back to a frame later, because the node
        card shows members as clips.

        ``progress_cb`` takes ``(fraction, phase)``. The fit is executor-backed and reports a
        **global** phase fraction rather than per-recording progress, so the phase name is what
        moves while the fraction is pinned inside a long step.

        Returns opaque state. The module stores it and never reads it.
        """

    @abstractmethod
    def hierarchy_of(self, fit: object) -> ClusterTree:
        """The full merge order behind this fit's granularities (§22).

        Declared on the protocol rather than reached for on a fit, which is what
        ``granularity_of`` should have been and is not. A surface that draws the hierarchy, names a
        node, or cuts it somewhere the method did not itself produce needs the order the merges
        happened in — the sequence every agglomerative rule computes and every one of them used to
        discard.

        Two guarantees the caller may rely on. The method's own partitions land exactly on merge
        boundaries (``rung_merges``), and **every cut refines every coarser cut**, because a cut is
        a prefix of one sequence. A method that has no nested partitions of its own still satisfies
        both: its finest partition is the leaf set and the whole tree is one unconstrained run, so
        the second guarantee holds trivially and the first names only the leaves.
        """

    @abstractmethod
    def member_leaves(self, fit: object) -> np.ndarray:
        """Each member's leaf of ``hierarchy_of(fit)``, in the index space ``instances_for`` uses.

        Membership with no cut (§22): a node's members are the members whose leaf lies under it,
        so any node of the tree can be asked about, not only the ones a granularity passes through.
        A member is whatever the method's instances index: an analysed row for a frame method, a
        bout for a bout method.
        """

    @abstractmethod
    def member_places(self, fit: object) -> MemberPlaces:
        """Where each member sits and the time it stands for, indexed like ``member_leaves``.

        What the node cache sums per leaf (§22): member time, occupancy per recording, and the
        member spans a sample is drawn from. ``time_ns`` is measured from the fit's own
        timestamps, never a count over an advertised fps (see ``MemberPlaces``).
        """

    @abstractmethod
    def rank_members(self, fit: object, leaves: np.ndarray) -> list:
        """The members under ``leaves``, most typical first, as ``[(instance, distance)]``.

        For any set of leaves, which is to say for any node: the node's own centre (centroid or
        medoid, whichever the method groups by) and its members' distances to it. What makes a node
        interpretable — which members show it best depends on how the method grouped them, so the
        method ranks and the module renders.

        ``distance`` is the method's own measure of *untypicality*, so **smaller is more typical**
        whatever the rule underneath. It is not a probability and must not be rendered as one (the
        same refusal §19's ``score`` column makes). The **whole** ordering, not a top-n: the tail
        is what tells a user whether a node is one thing.

        This replaced the per-cut ``member_order`` / ``representatives`` hooks (and ``projection``,
        whose 2-D layout the interpretation screen no longer draws), which were removed (§22).
        """

    def clip_scoring(
        self, fit: object, params: dict, trajectories: Sequence | None = None
    ) -> ClipScoring:
        """What the node cache needs to list and score **every** node's clips (§22).

        ``rank_members`` ranks one node from the fit, which is right for an export and too slow
        for a click on a large node. This hands the cache build what makes typicality a sum over
        leaf runs instead (``ClipScoring.kind``), plus the method's minimum instance length, so
        selecting any node lists its clips from the cache alone. ``params`` are the run's method
        params; ``trajectories`` are the fit's own inputs, in the order it was given them, for a
        method whose typicality needs every member's vector (``None`` when the caller has not
        loaded them: such a method then says what it needs). The default offers no typicality,
        makes runs of touching members when instances are runs, and keeps every clip.
        """
        return ClipScoring(
            kind="none", min_members=1, join=bool(self.instances_are_runs)
        )

    @abstractmethod
    def node_separations(self, fit: object) -> np.ndarray:
        """``(n_nodes,)``: how different each internal node's two children are; ``NaN`` on leaves.

        The distance between the children's centres over their mean within-child spread, in the
        method's own distance, so a node card can say whether a split is two things (§22). One
        batch over the whole tree, because the cache asks for every node.
        """

    @abstractmethod
    def instances_for(self, fit: object, leaf_groups: np.ndarray, params: dict) -> list:
        """The ``ClusterInstance``s a partition of the tree yields, one group id per leaf.

        ``leaf_groups[leaf]`` is the node that leaf is reported under, or ``-1`` for a leaf under
        no labelled node. The module resolves a user's labels to this partition and maps node ids
        to names afterwards, so the method is handed its own tree's grouping and **no names**: the
        same kind of input a granularity cut was, and still nothing a label could reach it through
        (``tests/cluster_methods/test_registry.py``).

        The method's decisions stay here, as they did in ``instances_at``: the continuity blocks
        runs never cross, and the minimum duration below which a run is not an instance. Runs of
        ``-1`` are instances too; unlabelled time is still time.
        """

    @abstractmethod
    def instances_at(self, fit: object, granularity: int, params: dict) -> list:
        """The ``ClusterInstance``s this granularity yields.

        Authored and instant (§21): if changing granularity triggers the fit, it is not an authored
        decision. Whether that holds is method-specific — a method for which it is expensive is not
        wrong, it just means its granularity is a launch parameter and its interaction is different.

        This is where the temporal decode and the **minimum duration below which a run is not an
        instance** live, because a frame-level labeling flickers and every flicker would otherwise
        become an instance.
        """

    def assign(self, fit: object, trajectory: np.ndarray, params: dict) -> list:
        """Label a recording that was not in the fit. Only if ``supports_apply``.

        Reserved (§22). Refitting after adding recordings changes every cluster id and invalidates
        the user's naming work, which is expensive human effort — that is the reason to build this
        eventually, and the reason its slot is declared now.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support apply-to-new-recordings"
        )

    @abstractmethod
    def serialize(self, fit: object, path: Path) -> None:
        """Write a fit to ``path``. The extension is the method's."""

    @abstractmethod
    def load(self, path: Path) -> object:
        """Read back what ``serialize`` wrote."""


#: ``member_groups`` value for "no override: the member's leaf decides its group".
KEEP_GROUP = -2


def member_groups_of(
    leaf_groups: np.ndarray, member_leaves: np.ndarray, member_groups: np.ndarray | None
) -> np.ndarray:
    """Each member's group: its leaf's, unless ``member_groups`` overrides it (``KEEP_GROUP``)."""
    groups = np.asarray(leaf_groups, dtype=np.int64)[np.asarray(member_leaves)]
    if member_groups is None:
        return groups
    override = np.asarray(member_groups, dtype=np.int64)
    if override.shape != groups.shape:
        raise ValueError(
            f"member_groups has {override.shape[0]} entries for {groups.shape[0]} members"
        )
    return np.where(override != KEEP_GROUP, override, groups)


class TemporalClusterMethod(ABC):
    """Discover a vocabulary of **behaviors** from pose trajectories, as segments of movement (§22).

    Deliberately *not* ``PoseClusterMethod``, and not a shared interface with a mode flag. §22: "The
    shapes are identical enough that sharing is tempting, and that is the trap: a shared interface
    would let a bout method be selected in the pose module, where it answers a different question."
    A cluster here is a behavior and an instance is a stretch of movement; there, a cluster is a
    posture and an instance is a run of frames holding it.

    The two protocols are near-identical on purpose. What separates them is which registry a method
    lands in, and therefore which module can offer it — a distinction that has to be in the types to
    be unexpressible rather than merely discouraged.

    **The module never looks inside a fit.** For a bout method it is typically a distance matrix
    plus the provenance of each bout; the method owns its serialization, and the module stores it
    and hands it back.

    **Segmentation belongs to the method.** Where a bout starts and ends, and how short is too short
    to be one, are decisions the grouping rule is entitled to make differently — so they live behind
    ``fit`` rather than in the module. This is the temporal counterpart of the pose protocol putting
    the decode and the minimum run length inside ``instances_at``.

    **No parameter carries a label.** ``fit`` takes trajectories and config; there is no argument
    through which ground truth could enter.
    """

    method_id: str
    config_model: type  # its own pydantic parameters, single-sourced (§6.5)
    # What the authored knob is, its type and its range. Rendered by the module rather than assumed
    # to be a cluster count: a FINCH level, an HDBSCAN `min_cluster_size` and a `k` are not the same
    # knob, which is also why a rule is a *method* here rather than a parameter of one.
    granularity: GranularitySpec
    # Whether this rule leaves bouts unassigned. §19's measures report "noise" and "unnamed"
    # separately, and a method with `has_noise = False` reports full assignment **of the bouts it
    # compared** — which is not the same as full coverage of the recording, because bouts do not
    # tile a timeline the way frames do.
    has_noise: bool
    # Reserved: apply-to-new-recordings is out of the MVP (§22).
    supports_apply: bool = False
    # The extension `serialize` writes, declared because a serializer may append its own.
    shard_suffix: str
    #: Whether an instance is a **run of like-labelled items** the method built, rather than an
    #: extent it found. ``dtw_finch`` sets this ``False``: a bout is an extent the segmentation
    #: found, and joining two of them because they share a name would be a claim it did not make.
    instances_are_runs: bool = True
    #: Importable modules the method needs beyond the base package (an optional extra's backend).
    #: Checked at ``validate``, as for pose methods, so a missing extra is a refusal before any work.
    requires: tuple[str, ...] = ()

    @abstractmethod
    def fit(
        self,
        trajectories: Sequence[tuple[str, int, np.ndarray, np.ndarray]],
        params: dict,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> object:
        """Cut the trajectories into bouts and find the vocabulary. Expensive, immutable (§21).

        ``trajectories`` is one entry per subject: ``(recording_id, subject, features, t_ns)``,
        matching the pose protocol so a module can load either the same way.

        ``progress_cb`` takes ``(fraction, phase)`` and reports a **global** phase fraction rather
        than per-recording progress. A bout method's phases are unequal in cost by orders of
        magnitude — the pairwise comparison dwarfs the segmentation — so the phase name is what
        moves while the fraction is pinned inside the long step.

        Returns opaque state. The module stores it and never reads it.
        """

    @abstractmethod
    def hierarchy_of(self, fit: object) -> ClusterTree:
        """The full merge order behind this fit's granularities (§22).

        Declared on the protocol rather than reached for on a fit, which is what
        ``granularity_of`` should have been and is not. A surface that draws the hierarchy, names a
        node, or cuts it somewhere the method did not itself produce needs the order the merges
        happened in — the sequence every agglomerative rule computes and every one of them used to
        discard.

        Two guarantees the caller may rely on. The method's own partitions land exactly on merge
        boundaries (``rung_merges``), and **every cut refines every coarser cut**, because a cut is
        a prefix of one sequence. A method that has no nested partitions of its own still satisfies
        both: its finest partition is the leaf set and the whole tree is one unconstrained run, so
        the second guarantee holds trivially and the first names only the leaves.
        """

    @abstractmethod
    def member_leaves(self, fit: object) -> np.ndarray:
        """Each member's leaf of ``hierarchy_of(fit)``, in the index space ``instances_for`` uses.

        Membership with no cut (§22): a node's members are the members whose leaf lies under it,
        so any node of the tree can be asked about, not only the ones a granularity passes through.
        A member is whatever the method's instances index: an analysed row for a frame method, a
        bout for a bout method.
        """

    @abstractmethod
    def member_places(self, fit: object) -> MemberPlaces:
        """Where each member sits and the time it stands for, indexed like ``member_leaves``.

        What the node cache sums per leaf (§22): member time, occupancy per recording, and the
        member spans a sample is drawn from. ``time_ns`` is measured from the fit's own
        timestamps, never a count over an advertised fps (see ``MemberPlaces``).
        """

    @abstractmethod
    def rank_members(self, fit: object, leaves: np.ndarray) -> list:
        """The members under ``leaves``, most typical first, as ``[(instance, distance)]``.

        For any set of leaves, which is to say for any node: the node's own centre (centroid or
        medoid, whichever the method groups by) and its members' distances to it. What makes a node
        interpretable — which members show it best depends on how the method grouped them, so the
        method ranks and the module renders.

        ``distance`` is the method's own measure of *untypicality*, so **smaller is more typical**
        whatever the rule underneath. It is not a probability and must not be rendered as one (the
        same refusal §19's ``score`` column makes). The **whole** ordering, not a top-n: the tail
        is what tells a user whether a node is one thing.

        This replaced the per-cut ``member_order`` / ``representatives`` hooks (and ``projection``,
        whose 2-D layout the interpretation screen no longer draws), which were removed (§22).
        """

    def clip_scoring(
        self, fit: object, params: dict, trajectories: Sequence | None = None
    ) -> ClipScoring:
        """What the node cache needs to list and score **every** node's clips (§22).

        ``rank_members`` ranks one node from the fit, which is right for an export and too slow
        for a click on a large node. This hands the cache build what makes typicality a sum over
        leaf runs instead (``ClipScoring.kind``), plus the method's minimum instance length, so
        selecting any node lists its clips from the cache alone. ``params`` are the run's method
        params; ``trajectories`` are the fit's own inputs, in the order it was given them, for a
        method whose typicality needs every member's vector (``None`` when the caller has not
        loaded them: such a method then says what it needs). The default offers no typicality,
        makes runs of touching members when instances are runs, and keeps every clip.
        """
        return ClipScoring(
            kind="none", min_members=1, join=bool(self.instances_are_runs)
        )

    @abstractmethod
    def node_separations(self, fit: object) -> np.ndarray:
        """``(n_nodes,)``: how different each internal node's two children are; ``NaN`` on leaves.

        The distance between the children's centres over their mean within-child spread, in the
        method's own distance, so a node card can say whether a split is two things (§22). One
        batch over the whole tree, because the cache asks for every node.
        """

    @abstractmethod
    def instances_for(
        self,
        fit: object,
        leaf_groups: np.ndarray,
        params: dict,
        member_groups: np.ndarray | None = None,
    ) -> list:
        """The ``ClusterInstance``s a partition of the tree yields, one group id per leaf.

        ``leaf_groups[leaf]`` is the node that leaf is reported under, or ``-1`` for a leaf under
        no labelled node. The module resolves a user's labels to this partition and maps node ids
        to names afterwards, so the method is handed its own tree's grouping and **no names**: the
        same kind of input a granularity cut was, and still nothing a label could reach it through
        (``tests/cluster_methods/test_registry.py``).

        The method's decisions stay here, as they did in ``instances_at``: the continuity blocks
        runs never cross, and the minimum duration below which a run is not an instance. Runs of
        ``-1`` are instances too; unlabelled time is still time.

        ``member_groups`` is a per-member override, indexed like ``member_leaves``: a group id, or
        :data:`KEEP_GROUP` where the member's leaf decides (:func:`member_groups_of`). It is how a
        clip a user took out of a label (``storage/clip_edits.py``) reports as ``-1``, or as the
        label they moved it to. Group ids again, never a name. The override is applied before runs
        are cut, so a clip taken out of the middle of a run splits it, and a piece left shorter
        than the method's minimum is dropped, as any short run is.
        """

    @abstractmethod
    def instances_at(self, fit: object, granularity: int, params: dict) -> list:
        """The ``ClusterInstance``s this granularity yields — one per bout the fit compared.

        Authored and instant (§21). For a cached-distance method this is a re-cluster of a matrix
        that already exists, which is what makes granularity an authored knob here rather than a
        launch parameter.

        Unlike the pose protocol there is no run-length encoding step: a bout is already an extent,
        so the minimum duration was applied during segmentation, before the comparison existed.
        """

    @abstractmethod
    def resolve(self, fit: object, instance) -> dict:
        """Where one instance sits in the world: recording, subject, frames and times.

        ``{recording_id, subject, start_frame, end_frame, t_ns, end_t_ns}``.

        **The module never looks inside a fit** (§22), and this is what makes that possible for
        the one thing it genuinely needs — turning an instance into a §19 row. A module that
        indexed the fit's own arrays instead would be reading a layout only one method has: it
        works until a second method stores something else, and then it raises deep inside a
        reprojection rather than at a boundary. That is not hypothetical; it is what happened.

        An instance whose rows span two recordings is a bug in the method. Returning it anyway
        would write a §19 row claiming a span across two recordings, so a method that cannot
        resolve one returns ``{}`` and the module drops it.
        """

    def fit_summary(self, fit: object) -> dict:
        """Counts worth recording in the §19 manifest, if this method has any.

        The companion to ``resolve``: the module records what a method reports rather than
        reaching into the fit for numbers only one method's layout has. Empty by default, because
        most methods have nothing extra to say and an empty dict adds no keys.
        """
        return {}

    def subsample_note(self, fit: object) -> str | None:
        """What the fit left out, in a sentence, or ``None`` if it left nothing out.

        §22 requires the surface to state a subsample, because a fit over a fraction of the bouts
        and a fit over all of them are identical in every other respect — same screen, same
        vocabulary shape, same confident-looking clusters. A method that always compares everything
        returns ``None`` and the surface shows no line.
        """
        return None

    def assign(self, fit: object, trajectory: np.ndarray, params: dict) -> list:
        """Label a recording that was not in the fit. Only if ``supports_apply``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support apply-to-new-recordings"
        )

    @abstractmethod
    def serialize(self, fit: object, path: Path) -> None:
        """Write a fit to ``path``. The extension is the method's."""

    @abstractmethod
    def load(self, path: Path) -> object:
        """Read back what ``serialize`` wrote."""
