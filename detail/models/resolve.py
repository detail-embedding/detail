"""Resolve a detection ``model_id`` to a DetectionModel.

A model id resolves from one of three sources, tried in order by
``resolve_detection_model_auto``: the ``MODELS`` registry (pretrained,
code-defined — ``sam3``/…), a experiment ``models/<id>`` (trained in this
experiment), or the machine-global imported-model catalog (``models/imported.py``).
Experiment-trained and imported models share the same on-disk shape — a ``model.yaml``
(Section 12) + a local ``best.pt`` constructed through the family's
``TRAINED_LOADERS`` loader — and differ only in *where* the directory lives. The
discovered ``weights_source`` is recorded on the run so the worker re-resolves the
same model. A pose (``task: pose``) record is a keypoint model and detection refuses
it; the keypoint functions below resolve it in the same order (built-in
``KEYPOINT_MODELS`` → experiment → catalog) and record the source the same way (§12
*Keypoint models in the catalog*). Backend imports stay inside the model classes (Tenet 13); this module
imports none.
"""

from pathlib import Path

from detail.core.protocols import DetectionModel
from detail.core.registries import MODELS, TRAINED_LOADERS
from detail.models import imported as imported_store
from detail.storage import models as models_store


def resolve_detection_model(
    experiment_root: Path | str,
    model_id: str,
    weights_source: str = "pretrained",
    model_variant: str | None = None,
) -> DetectionModel:
    """Construct (but do not ``load``) the DetectionModel for a *known* source.

    The worker calls this with the concrete ``weights_source`` recorded on the job;
    new code that doesn't yet know the source should use
    ``resolve_detection_model_auto``.

    Raises:
        ValueError: Unknown ``weights_source``, unknown pretrained id, no loader
            for a local model's family, or an incomplete local model.
        FileNotFoundError: ``weights_source`` is local but no such ``model.yaml``.
    """
    if weights_source == "trained":
        return _load_from_model_yaml(models_store.model_dir(experiment_root, model_id))
    if weights_source == "imported":
        return _load_from_model_yaml(imported_store.model_dir(model_id))
    if weights_source != "pretrained":
        raise ValueError(
            f"unknown weights_source {weights_source!r}; "
            "expected 'pretrained', 'trained', or 'imported'"
        )
    model_cls = MODELS.get(model_id)
    if model_cls is None:
        raise ValueError(f"unknown model_id {model_id!r}; available: {sorted(MODELS)}")
    return model_cls(model_variant)


def resolve_detection_model_auto(
    experiment_root: Path | str,
    model_id: str,
    model_variant: str | None = None,
) -> tuple[DetectionModel, str]:
    """Resolve a model id *without* being told its source — try the pretrained
    registry → a experiment-trained model → a machine-imported model, and return
    ``(model, weights_source)``. The id namespaces are disjoint by construction
    (registry ids are names; experiment ids are ``model_NNN``; imports reject both),
    so the first hit is unambiguous.

    Raises:
        ValueError: ``model_id`` matches no source.
    """
    if model_id in MODELS:
        return MODELS[model_id](model_variant), "pretrained"
    experiment_dir = models_store.model_dir(experiment_root, model_id)
    if (experiment_dir / models_store.MODEL_FILENAME).is_file():
        return _load_from_model_yaml(experiment_dir), "trained"
    imported_dir = imported_store.model_dir(model_id)
    if (imported_dir / models_store.MODEL_FILENAME).is_file():
        return _load_from_model_yaml(imported_dir), "imported"
    raise ValueError(
        f"unknown model {model_id!r} — not a built-in model ({sorted(MODELS)}), a "
        "trained model in this experiment, or an imported model (see `model list`)"
    )


def resolve_pretrained_or_imported(
    model_id: str, model_variant: str | None = None
) -> DetectionModel:
    """Resolve a *machine-global* model (no experiment scope): a built-in registry model,
    else an imported/distilled catalog model. This is what a **tool** (§9) uses — it has no
    experiment, so the experiment-trained source doesn't apply.

    Raises:
        ValueError: ``model_id`` matches neither a built-in nor an imported model.
    """
    model_cls = MODELS.get(model_id)
    if model_cls is not None:
        return model_cls(model_variant)
    if model_id in imported_store.discover():
        return _load_from_model_yaml(imported_store.model_dir(model_id))
    raise ValueError(
        f"unknown model {model_id!r} — not a built-in model ({sorted(MODELS)}) or an "
        "imported/distilled model (see `model list`)"
    )


def _load_from_model_yaml(model_dir: Path | str) -> DetectionModel:
    """Construct a closed-set DetectionModel from a ``model.yaml`` directory
    (experiment-trained or machine-imported) via its family's ``TRAINED_LOADERS``
    loader with the local weights path + recorded class names."""
    model_dir = Path(model_dir)
    record = models_store.read_model_yaml(model_dir)  # FileNotFoundError if absent
    if imported_store.is_pose_record(record):
        raise ValueError(
            f"model {model_dir.name!r} is a keypoint (pose) model, not a detector; run it "
            "in keypoint detection (keypoint_detect), not detection"
        )
    status = record.get("status")
    if status != "complete":
        raise ValueError(
            f"model {model_dir.name!r} is not complete (status={status!r})"
        )
    family = record.get("model_family")
    loader = TRAINED_LOADERS.get(family)
    if loader is None:
        raise ValueError(
            f"no trained-model loader for family {family!r}; "
            f"available: {sorted(TRAINED_LOADERS)}"
        )
    best = record.get("weights", {}).get("best")
    if best is None:
        raise ValueError(f"model {model_dir.name!r} has no best-weights path")
    classes = [c["name"] for c in record.get("classes", [])]
    return loader(weights_path=model_dir / best, classes=classes)


#: Where a keypoint student's weights live, by the ``weights_source`` a run records (§12
#: *Keypoint models in the catalog*). A registered method has no source (None).
KEYPOINT_WEIGHTS_SOURCES = ("trained", "imported")


def resolve_keypoint_model(model_id: str, model_variant: str | None = None, **kwargs):
    """A keypoint method (§26) by id, constructed with the inputs it takes.

    ``kwargs`` are the method's build inputs — ``examples`` / ``example_images`` for one that
    ``needs_examples``, ``decode`` / ``side_rule`` / ``crop_px`` for the patch matcher. A
    ``None`` is "not given" and is dropped; a value a method does not take is refused by name
    rather than swallowed, the way ``construct_embedding_model`` treats a read-out.

    ``experiment_dir`` and ``weights_source`` say where a student is: with a recorded
    ``weights_source`` (``"trained"`` / ``"imported"``) only that place is read, so a run
    rebuilds the model it validated; without one the order is built-in → this experiment's
    students → the machine catalog (§12 *Keypoint models in the catalog*).
    """
    import inspect

    from detail.core.registries import KEYPOINT_MODELS

    experiment_dir = kwargs.pop("experiment_dir", None)
    weights_source = kwargs.pop("weights_source", None)
    if weights_source is not None:
        return _load_keypoint_student(
            _student_at_source(experiment_dir, model_id, weights_source),
            model_id,
            kwargs,
        )
    model_cls = KEYPOINT_MODELS.get(model_id)
    if model_cls is None:
        hit = _find_keypoint_student(experiment_dir, model_id)
        if hit is not None:
            return _load_keypoint_student(hit, model_id, kwargs)
        students = [s["id"] for s in keypoint_students(experiment_dir)]
        raise ValueError(
            f"unknown keypoint method {model_id!r}; available: {sorted(KEYPOINT_MODELS)}"
            + (f" and the keypoint models {students}" if students else "")
        )
    params = inspect.signature(model_cls.__init__).parameters
    given = {k: v for k, v in kwargs.items() if v is not None}
    unknown = sorted(k for k in given if k not in params)
    if unknown:
        raise ValueError(f"keypoint method {model_id!r} takes no {unknown}")
    return model_cls(model_variant, **given)


def resolve_keypoint_weights_source(
    experiment_dir: Path | str | None, model_id: str
) -> str | None:
    """Where a keypoint method id resolves, in the lookup order: None for a registered
    method, ``"trained"`` for this experiment's student, ``"imported"`` for a catalog model.
    A keypoint detect run records it (Tenet 9) because an experiment model and a catalog
    model can share an id.

    Raises:
        ValueError: ``model_id`` is none of the three.
    """
    from detail.core.registries import KEYPOINT_MODELS

    if model_id in KEYPOINT_MODELS:
        return None
    hit = _find_keypoint_student(experiment_dir, model_id)
    if hit is None:
        raise ValueError(
            f"unknown keypoint method {model_id!r}; available: {sorted(KEYPOINT_MODELS)}, "
            "this experiment's keypoint students and the catalog's keypoint models"
        )
    return hit[2]


def keypoint_method_class(experiment_dir: Path | str | None, model_id: str | None):
    """The class behind a keypoint method id, for its class-level facts (``score_spec``,
    ``default_track_method``, ``needs_examples``): a registered method, or ``TrainedYoloPose``
    for a pose student — this experiment's (§12 *Distill keypoints*) or the catalog's (§12
    *Keypoint models in the catalog*); None when it is neither.

    **The one lookup.** A student is not in ``KEYPOINT_MODELS`` (it is a model the
    experiment trained, not a registered method), so every site that asked the registry alone
    read a student's runs as having no score cut-off, no score meaning and no default track
    method: the exports showed every point, the heading counted every point, and a student's
    detect run could not be tracked without naming a method.
    """
    from detail.core.registries import KEYPOINT_MODELS

    if not model_id:
        return None
    cls = KEYPOINT_MODELS.get(str(model_id))
    if cls is not None:
        return cls
    if _find_keypoint_student(experiment_dir, str(model_id)) is not None:
        from detail.models.yolo_pose import TrainedYoloPose

        return TrainedYoloPose
    return None


def keypoint_students(experiment_dir: Path | str | None) -> list[dict]:
    """Every complete pose model a keypoint method picker offers beside ``KEYPOINT_MODELS``:
    this experiment's students (oldest first), then the catalog's pose models. Each row is
    the ``model.yaml`` record plus ``source`` (``"trained"`` | ``"imported"``). A catalog
    model whose id an experiment student also has is left out: the id resolves to the
    student, so offering the catalog row would offer a model the run cannot reach."""
    out = []
    taken = set()
    if experiment_dir is not None:
        for model_id in models_store.discover_models(experiment_dir):
            record = _pose_record_at(models_store.model_dir(experiment_dir, model_id))
            if record is not None and record.get("status") == "complete":
                out.append({**record, "source": "trained"})
                taken.add(model_id)
    for model_id in imported_store.discover():
        if model_id in taken:
            continue
        record = _pose_record_at(imported_store.model_dir(model_id))
        if record is not None and record.get("status") == "complete":
            out.append({**record, "source": "imported"})
    return out


def _pose_record_at(model_dir: Path) -> dict | None:
    try:
        record = models_store.read_model_yaml(model_dir)
    except FileNotFoundError:
        return None
    return record if imported_store.is_pose_record(record) else None


def _student_dir(
    experiment_dir: Path | str | None, model_id: str, weights_source: str
) -> Path | None:
    if weights_source == "trained":
        if experiment_dir is None:
            return None
        return models_store.model_dir(experiment_dir, model_id)
    if weights_source == "imported":
        return imported_store.model_dir(model_id)
    raise ValueError(
        f"unknown keypoint weights_source {weights_source!r}; expected one of "
        f"{list(KEYPOINT_WEIGHTS_SOURCES)} (or none for a registered method)"
    )


def _find_keypoint_student(
    experiment_dir: Path | str | None, model_id: str
) -> tuple[dict, Path, str] | None:
    """``(record, model dir, weights_source)`` of the first pose model with this id —
    the experiment's, then the catalog's — or None."""
    for source in KEYPOINT_WEIGHTS_SOURCES:
        model_dir = _student_dir(experiment_dir, model_id, source)
        if model_dir is None:
            continue
        record = _pose_record_at(model_dir)
        if record is not None:
            return record, model_dir, source
    return None


def _student_at_source(
    experiment_dir: Path | str | None, model_id: str, weights_source: str
) -> tuple[dict, Path, str]:
    model_dir = _student_dir(experiment_dir, model_id, weights_source)
    record = None if model_dir is None else _pose_record_at(model_dir)
    if record is None:
        where = (
            "this experiment's models"
            if weights_source == "trained"
            else "the model catalog"
        )
        raise ValueError(
            f"keypoint model {model_id!r} (weights_source={weights_source!r}) is not in "
            f"{where}; it was removed or renamed since the run was configured"
        )
    return record, model_dir, weights_source


def _load_keypoint_student(hit: tuple[dict, Path, str], model_id: str, kwargs: dict):
    """Construct the student from its record: local weights under the directory of the
    branch that matched + the skeleton copy it was trained on. Takes no example set and
    ignores the patch matcher's settings, which are not its; a non-None ``examples`` is
    refused rather than dropped."""
    from detail.core.skeleton import Skeleton
    from detail.models.yolo_pose import TrainedYoloPose

    record, model_dir, _source = hit
    if record.get("status") != "complete":
        raise ValueError(
            f"keypoint student {model_id!r} is not complete (status={record.get('status')!r})"
        )
    if kwargs.get("examples") is not None:
        raise ValueError(
            f"keypoint student {model_id!r} takes no example set; it was trained once and "
            "needs no clicks"
        )
    best = (record.get("weights") or {}).get("best")
    if not best:
        raise ValueError(f"keypoint student {model_id!r} has no best-weights path")
    skeleton = record.get("skeleton")
    if not skeleton:
        raise ValueError(f"keypoint student {model_id!r} records no skeleton")
    return TrainedYoloPose(Path(model_dir) / best, Skeleton.from_dict(skeleton))
