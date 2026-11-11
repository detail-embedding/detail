"""Backend registries and their self-registration decorators (Tenet 1).

A backend is referenced by string ID; callers never import concrete classes.
Each concrete implementation calls its ``register_*`` decorator on import.
Excluding a backend means not importing its module — the registry never sees it.
"""

from __future__ import annotations

from collections.abc import Callable

from detail.core.protocols import (
    Check,
    DetectionModel,
    EmbeddingModel,
    KeypointModel,
    KeypointTrackMethod,
    MaskTransform,
    Model,
    PoseClusterMethod,
    QueryMethod,
    StreamImporter,
    TemporalClusterMethod,
    Tool,
    Tracker,
    Trainer,
)

MODELS: dict[str, type[DetectionModel]] = {}
# DetectionModels that load locally-trained weights, keyed by the ``model_family``
# recorded in a model.yaml (§12) — detection resolves a ``weights_source="trained"``
# model id to one of these (a separate axis from MODELS, which is keyed by model id).
TRAINED_LOADERS: dict[str, type[DetectionModel]] = {}
TRAINERS: dict[str, type[Trainer]] = {}
TRACKERS: dict[str, type[Tracker]] = {}
EMBEDDING_MODELS: dict[str, type[EmbeddingModel]] = {}
TRANSFORMS: dict[str, type[MaskTransform]] = {}
# Keypoint methods (§26), keyed by model_id — one namespace with the registries above.
KEYPOINT_MODELS: dict[str, type[KeypointModel]] = {}
# Keypoint track methods (§26 "Track methods"), keyed by method_id. Not a backend in the
# weights sense — no install, no device — so it stays as light as TRACKERS.
KEYPOINT_TRACK_METHODS: dict[str, type[KeypointTrackMethod]] = {}
#: Every registry keyed by a ``model_id``, with its role (§5 "Model registry"). The dicts stay
#: separate because each runner dispatches on its own, but a model id is **one namespace**:
#: the ``register_*`` decorators refuse an id another role already holds, and
#: ``api.models.all_models`` resolves an id through this tuple rather than a hand-written
#: ``MODELS.get(id) or EMBEDDING_MODELS.get(id)`` chain, which answered a duplicate with the
#: detection model silently.
MODEL_REGISTRIES: tuple[tuple[str, dict[str, type[Model]]], ...] = (
    ("detection", MODELS),
    ("embedding", EMBEDDING_MODELS),
    ("keypoint", KEYPOINT_MODELS),
)
STREAM_IMPORTERS: dict[str, type[StreamImporter]] = {}
# Behavior-query search methods (§20), keyed by method_id. A method is not a backend
# in the weights sense — no install, no device — so this registry stays as light as
# TRACKERS, and the Query module needs no change when one is added.
QUERY_METHODS: dict[str, type[QueryMethod]] = {}
# Pose-clustering methods (§22), keyed by method_id. Separate from any temporal-clustering
# registry on purpose: a bout method selected in the pose module would answer a different
# question, and keeping the registries apart makes that unexpressible rather than a mistake
# the module has to guard against.
POSE_CLUSTER_METHODS: dict[str, type[PoseClusterMethod]] = {}
# Temporal (bout) clustering methods (§22), keyed by method_id. The sibling of the registry above,
# kept apart for the reason stated there — and note that a *cluster rule* is a method here rather
# than a parameter of one: FINCH offers a nested cluster count, HDBSCAN a `min_cluster_size`, and
# only one of them leaves bouts unassigned, so a single rule-parameterized method would have to
# declare a `granularity` and a `has_noise` that are false for some of its own settings.
TEMPORAL_CLUSTER_METHODS: dict[str, type[TemporalClusterMethod]] = {}
# One-off, experiment-free operations — the Tools framework (§9). Keyed by tool_id.
TOOLS: dict[str, type[Tool]] = {}
# Readiness / data-integrity checks — the live checker framework (§1). Keyed by check_id.
CHECKS: dict[str, type[Check]] = {}
# Consumers of a stream role, keyed by ``role`` (e.g. "behavior"). Each callable
# takes ``(recording_dir, stream_name)`` and returns the run ids that build on the
# stream and would be orphaned by deleting it. Empty in stage 1 (no stream consumer
# exists yet); the seam lets a future consumer register without editing
# ``Recording.stream_dependents``.
STREAM_CONSUMERS: dict[str, list[Callable[[object, str], list[str]]]] = {}


def _refuse_duplicate_model_id(model_id: str, role: str) -> None:
    """A model id already registered under another role is an error at import (§5)."""
    for other_role, registry in MODEL_REGISTRIES:
        if other_role != role and model_id in registry:
            raise ValueError(
                f"model id {model_id!r} is already registered as a {other_role} model; "
                f"a model_id is one namespace across every model registry, so a "
                f"{role} model needs its own id"
            )


def register_model(
    model_id: str,
) -> Callable[[type[DetectionModel]], type[DetectionModel]]:
    """Register a DetectionModel under ``model_id``."""

    def decorator(cls: type[DetectionModel]) -> type[DetectionModel]:
        _refuse_duplicate_model_id(model_id, "detection")
        MODELS[model_id] = cls
        return cls

    return decorator


def register_trained_loader(
    model_family: str,
) -> Callable[[type[DetectionModel]], type[DetectionModel]]:
    """Register a DetectionModel that loads locally-trained weights, keyed by
    ``model_family`` (§12). Detection resolves a ``weights_source="trained"``
    model id to this loader via the family recorded in its ``model.yaml``."""

    def decorator(cls: type[DetectionModel]) -> type[DetectionModel]:
        TRAINED_LOADERS[model_family] = cls
        return cls

    return decorator


def register_trainer(
    trainer_id: str,
) -> Callable[[type[Trainer]], type[Trainer]]:
    """Register a Trainer under ``trainer_id``."""

    def decorator(cls: type[Trainer]) -> type[Trainer]:
        TRAINERS[trainer_id] = cls
        return cls

    return decorator


def register_tracker(
    tracker_id: str,
) -> Callable[[type[Tracker]], type[Tracker]]:
    """Register a Tracker under ``tracker_id``."""

    def decorator(cls: type[Tracker]) -> type[Tracker]:
        TRACKERS[tracker_id] = cls
        return cls

    return decorator


def register_query_method(
    method_id: str,
) -> Callable[[type[QueryMethod]], type[QueryMethod]]:
    """Register a QueryMethod under ``method_id`` (§20)."""

    def decorator(cls: type[QueryMethod]) -> type[QueryMethod]:
        QUERY_METHODS[method_id] = cls
        return cls

    return decorator


def register_pose_cluster_method(
    method_id: str,
) -> Callable[[type[PoseClusterMethod]], type[PoseClusterMethod]]:
    """Register a PoseClusterMethod under ``method_id`` (§22)."""

    def decorator(cls: type[PoseClusterMethod]) -> type[PoseClusterMethod]:
        POSE_CLUSTER_METHODS[method_id] = cls
        return cls

    return decorator


def register_temporal_cluster_method(
    method_id: str,
) -> Callable[[type[TemporalClusterMethod]], type[TemporalClusterMethod]]:
    """Register a TemporalClusterMethod under ``method_id`` (§22)."""

    def decorator(cls: type[TemporalClusterMethod]) -> type[TemporalClusterMethod]:
        TEMPORAL_CLUSTER_METHODS[method_id] = cls
        return cls

    return decorator


def register_embedding_model(
    model_id: str,
) -> Callable[[type[EmbeddingModel]], type[EmbeddingModel]]:
    """Register an EmbeddingModel under ``model_id``."""

    def decorator(cls: type[EmbeddingModel]) -> type[EmbeddingModel]:
        _refuse_duplicate_model_id(model_id, "embedding")
        EMBEDDING_MODELS[model_id] = cls
        return cls

    return decorator


def register_keypoint_model(
    model_id: str,
) -> Callable[[type[KeypointModel]], type[KeypointModel]]:
    """Register a KeypointModel under ``model_id`` (§26)."""

    def decorator(cls: type[KeypointModel]) -> type[KeypointModel]:
        _refuse_duplicate_model_id(model_id, "keypoint")
        KEYPOINT_MODELS[model_id] = cls
        return cls

    return decorator


def register_keypoint_track_method(
    method_id: str,
) -> Callable[[type[KeypointTrackMethod]], type[KeypointTrackMethod]]:
    """Register a KeypointTrackMethod under ``method_id`` (§26 "Track methods")."""

    def decorator(cls: type[KeypointTrackMethod]) -> type[KeypointTrackMethod]:
        KEYPOINT_TRACK_METHODS[method_id] = cls
        return cls

    return decorator


def construct_embedding_model(
    model_cls: type[EmbeddingModel],
    variant: str | None = None,
    *,
    readout: str | None = None,
    input_size: int | None = None,
) -> EmbeddingModel:
    """Construct an embedding model, forwarding ``readout`` / ``input_size`` only to models
    that accept them.

    Not every backbone has a selectable read-out or a resizable input. Passing an unsupported override raises an actionable ``ValueError`` here
    rather than a bare ``TypeError`` deep inside a pool worker.
    """
    import inspect

    params = inspect.signature(model_cls.__init__).parameters
    model_name = getattr(model_cls, "model_id", model_cls.__name__)
    kwargs: dict = {}
    if readout is not None:
        if "readout" not in params:
            raise ValueError(f"model {model_name!r} has no selectable read-out")
        kwargs["readout"] = readout
    if input_size is not None:
        if "input_size" not in params:
            fixed = getattr(model_cls, "input_size", None)
            raise ValueError(
                f"model {model_name!r} has a fixed input size of {fixed} and cannot be resized"
            )
        kwargs["input_size"] = input_size
    return model_cls(variant, **kwargs)


def register_transform(
    name: str,
) -> Callable[[type[MaskTransform]], type[MaskTransform]]:
    """Register a MaskTransform under ``name``."""

    def decorator(cls: type[MaskTransform]) -> type[MaskTransform]:
        TRANSFORMS[name] = cls
        return cls

    return decorator


def register_stream_importer(
    importer_id: str,
) -> Callable[[type[StreamImporter]], type[StreamImporter]]:
    """Register a StreamImporter under ``importer_id``."""

    def decorator(cls: type[StreamImporter]) -> type[StreamImporter]:
        STREAM_IMPORTERS[importer_id] = cls
        return cls

    return decorator


def register_tool(tool_id: str) -> Callable[[type[Tool]], type[Tool]]:
    """Register a Tool under ``tool_id`` (Section 9)."""

    def decorator(cls: type[Tool]) -> type[Tool]:
        TOOLS[tool_id] = cls
        return cls

    return decorator


def register_check(check_id: str) -> Callable[[type[Check]], type[Check]]:
    """Register a Check under ``check_id`` (the live checker framework, §1)."""

    def decorator(cls: type[Check]) -> type[Check]:
        CHECKS[check_id] = cls
        return cls

    return decorator


def register_stream_consumer(
    role: str,
) -> Callable[[Callable[[object, str], list[str]]], Callable[[object, str], list[str]]]:
    """Register a dependents-listing callable for a stream ``role`` (Section 3).

    The callable takes ``(recording_dir, stream_name)`` and returns the run ids that
    consume the stream. ``Recording.stream_dependents`` aggregates every consumer
    registered for the stream's role; none is registered in stage 1.
    """

    def decorator(
        fn: Callable[[object, str], list[str]],
    ) -> Callable[[object, str], list[str]]:
        STREAM_CONSUMERS.setdefault(role, []).append(fn)
        return fn

    return decorator
