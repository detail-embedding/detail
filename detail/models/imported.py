"""Machine-global imported-model catalog (Section 5/12).

User-imported weights live **outside any experiment**, shared across all of
them — the data-defined twin of the code-defined pretrained registries.

A record has a **role** — ``detection`` (a ``.pt`` run through its family's
``TRAINED_LOADERS`` loader) or ``keypoints`` (an ONNX pose model, §26, run by
``models/onnx_keypoints.py``). A record written before roles existed is a detection
model, which is what every record then was. ``discover`` / ``list_records`` take the
role as a **required** argument: a caller that wants detection models has to say so,
because a keypoint model handed to a detection loader is the failure a default would
hide.
Each imported model is a directory under ``model_cache_dir()/imported/<id>/``
holding a ``model.yaml`` (the same schema as a experiment trained model — Section 12)
and ``weights/best.pt``. Class names are read straight from the checkpoint at
import time (the per-family ``introspect`` hook), so sharing a model is just
sharing its ``.pt``.

Discovered by scanning; constructed by ``models/resolve.py`` via the family's
``TRAINED_LOADERS`` loader — identical to a experiment trained model, only the
directory is machine-global instead of experiment-local.

A **keypoint (YOLO-pose) model** shares the catalog (§12 *Keypoint models in the catalog*):
its record has ``task: pose``, ``role: keypoints``, ``model_family: yolo_pose`` and the skeleton
copy it is rebuilt from. :func:`is_pose_record` is the one test every catalog reader asks;
detection paths skip or refuse such a record, and ``TrainedYoloPose`` is deliberately absent
from ``TRAINED_LOADERS`` so detection resolution can never build one.

The registries (``MODELS`` / ``TRAINED_LOADERS``) must be populated before
``import_model`` (the caller does ``import_runner_modules`` — same assumption as
``resolve.py``); this module imports no backend itself (Tenet 13).
"""

import hashlib
import re
import shutil
from pathlib import Path

from detail.core.registries import (
    EMBEDDING_MODELS,
    KEYPOINT_MODELS,
    MODELS,
    TRAINED_LOADERS,
)
from detail.core.skeleton import Skeleton
from detail.core.skeleton import check_against_model
from detail.core.utils import utc_timestamp
from detail.models.hf_access import model_cache_dir
from detail.storage import dirscan
from detail.storage import models as models_store
from detail.storage import skeleton_library

WEIGHTS_RELPATH = "weights/best.pt"
ROLES = ("detection", "keypoints")
POSE_TASK = "pose"
POSE_FAMILY = "yolo_pose"
KEYPOINTS_ROLE = "keypoints"
# The loader that reads a checkpoint's task when no family is given: every importable
# checkpoint today is an Ultralytics one, and its task decides detection vs keypoints (§12).
_PROBE_FAMILY = "yolo"
# Experiment trained-model ids are model_NNN; reserve that shape so a `--model <id>`
# never has to guess between a experiment model and an imported one.
_RESERVED_ID = re.compile(r"^model_\d+$")


def catalog_dir() -> Path:
    return model_cache_dir() / "imported"


def model_dir(model_id: str) -> Path:
    return catalog_dir() / model_id


def role_of(record: dict) -> str:
    """A record's role; one written before roles existed is a detection model, unless it is a
    pose model (§12), which is a keypoints one whatever it wrote."""
    if record.get("role"):
        return record["role"]
    return KEYPOINTS_ROLE if is_pose_record(record) else "detection"


def discover(role: str | None) -> list[str]:
    """Sorted ids of imported models with ``role`` (``None``: every role).

    Raises:
        ValueError: for an unknown role.
    """
    if role is not None and role not in ROLES:
        raise ValueError(f"role must be one of {ROLES} or None, got {role!r}")
    root = catalog_dir()
    if not root.is_dir():
        return []
    ids = sorted(
        p.name for p in root.iterdir() if (p / models_store.MODEL_FILENAME).is_file()
    )
    if role is None:
        return ids
    return [i for i in ids if role_of(read(i)) == role]


def read(model_id: str) -> dict:
    """One imported model's record (FileNotFoundError if absent)."""
    return models_store.read_model_yaml(model_dir(model_id))


def list_records(role: str | None) -> list[dict]:
    """Every imported model's record with ``role`` (``None``: every role)."""
    return dirscan.read_each(discover(role), read)


def is_pose_record(record: dict) -> bool:
    """Whether a catalog record is a keypoint (YOLO-pose) model rather than a detector.

    The ``task`` decides. A record that wrote none falls back to ``model_family:
    yolo_pose``: a student remote-detail promoted before §12's catalog shape (2026-10-21)
    wrote only the family, and is still a pose model."""
    task = record.get("task")
    if task is not None:
        return task == POSE_TASK
    return record.get("model_family") == POSE_FAMILY


def remove(model_id: str) -> None:
    """Delete an imported model's catalog directory (idempotent)."""
    shutil.rmtree(model_dir(model_id), ignore_errors=True)


def import_model(
    weights_path: Path | str,
    *,
    family: str | None = None,
    skeleton: Skeleton | dict | None = None,
    skeleton_name: str | None = None,
    name: str | None = None,
    made_by: str | None = None,
    description: str | None = None,
    notes: str | None = None,
    preview: Path | str | None = None,
) -> str:
    """Copy ``weights_path`` into the catalog as a new imported model, reading its
    classes and task from the checkpoint. Returns the assigned (name-based) id.

    The task decides what the model is (§12 *Keypoint models in the catalog*). A pose
    checkpoint imports as a ``yolo_pose`` keypoint model and needs ``skeleton``, which is
    checked against the checkpoint's point count and ``flip_idx``, copied into the record,
    and added to the skeleton library (under ``skeleton_name`` when it is new there). Any
    other task imports as a detector of ``family``. ``family`` is optional; when given it
    must agree with the task.

    ``made_by`` / ``description`` / ``notes`` are user-supplied metadata (stored so a
    model isn't "lost in a folder" — §12); ``preview`` is an optional image copied in
    as the model's preview thumbnail.

    Raises:
        FileNotFoundError: ``weights_path`` (or ``preview``) does not exist.
        ValueError: Unknown / non-importable family, a family that disagrees with the
            checkpoint's task, a pose checkpoint without a skeleton (or with one that does
            not fit it), a skeleton for a detection checkpoint, a class-less checkpoint, an
            id that collides with a built-in model id or the reserved ``model_NNN`` shape,
            or an id already in the catalog.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    probe = _PROBE_FAMILY if family in (None, POSE_FAMILY) else family
    loader = TRAINED_LOADERS.get(probe)
    if loader is None:
        raise ValueError(
            f"unknown model family {family!r}; available: "
            f"{sorted([*TRAINED_LOADERS, POSE_FAMILY])}"
        )
    introspect = getattr(loader, "introspect", None)
    if introspect is None:
        raise ValueError(f"family {probe!r} does not support import")
    classes, task = introspect(weights_path)

    pose_skeleton: Skeleton | None = None
    if task == POSE_TASK:
        if family not in (None, POSE_FAMILY):
            raise ValueError(
                f"{weights_path.name} is a pose (keypoint) checkpoint, which imports as "
                f"{POSE_FAMILY!r}, not {family!r}; leave the family out"
            )
        if skeleton is None:
            raise ValueError(
                f"{weights_path.name} is a pose (keypoint) checkpoint; pass the skeleton its "
                "points follow (a preset, a library skeleton or a skeleton YAML)"
            )
        family = POSE_FAMILY
        pose_skeleton = (
            skeleton if isinstance(skeleton, Skeleton) else Skeleton.from_dict(skeleton)
        )
        k, flip_idx = _pose_shape(weights_path)
        check_against_model(pose_skeleton, k, flip_idx)
    else:
        if family == POSE_FAMILY:
            raise ValueError(
                f"{weights_path.name} is a {task} checkpoint, not a pose one; it imports as "
                "a detection model — leave the family out"
            )
        if skeleton is not None:
            raise ValueError(
                f"{weights_path.name} is a {task} (detection) checkpoint; a skeleton applies "
                "only to a pose (keypoint) checkpoint"
            )
        family = probe

    model_id = _slugify(name or weights_path.stem)
    _validate_id(model_id)
    mdir = model_dir(model_id)
    if mdir.exists():
        raise ValueError(
            f"imported model {model_id!r} already exists; remove it or pass a "
            "different --name"
        )

    record = {
        "id": model_id,
        "name": name or model_id,
        "source": "imported",
        "model_family": family,
        "task": task,
        "classes": [{"id": i, "name": n} for i, n in enumerate(classes)],
        "weights": {"best": WEIGHTS_RELPATH},
        "status": "complete",
        "created_at": utc_timestamp(),
        "made_by": made_by or "",
        "description": description or "",
        "notes": notes or "",
    }
    (mdir / "weights").mkdir(parents=True)
    shutil.copy2(weights_path, mdir / WEIGHTS_RELPATH)
    record["preview"] = set_preview(mdir, preview) if preview is not None else None
    if pose_skeleton is not None:
        record.update(
            pose_record_fields(
                pose_skeleton,
                skeleton_name or f"{name or model_id} skeleton",
                mdir / WEIGHTS_RELPATH,
            )
        )
    models_store.write_model_yaml(mdir, record)
    return model_id


def pose_record_fields(skeleton: Skeleton, skeleton_name: str, weights: Path) -> dict:
    """The fields a catalog keypoint model adds to its record (§12): role, task and family,
    the skeleton copy the model uses with its digest, the library entry it was matched or
    added to (:func:`skeleton_library.ensure_in_library`), and the weights' sha256."""
    ref = skeleton_library.ensure_in_library(skeleton, skeleton_name)
    return {
        "role": KEYPOINTS_ROLE,
        "task": POSE_TASK,
        "model_family": POSE_FAMILY,
        "skeleton": skeleton.to_dict(),
        "skeleton_digest": skeleton.digest(),
        "skeleton_ref": ref,
        "weights_sha256": _file_sha256(weights),
    }


def _pose_shape(weights_path: Path) -> tuple[int, list[int] | None]:
    """``(K, flip_idx)`` of a pose checkpoint, through the pose model's own reader."""
    from detail.models.yolo_pose import TrainedYoloPose

    return TrainedYoloPose.introspect(weights_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_preview(model_dir: Path | str, image_path: Path | str) -> str:
    """Copy ``image_path`` into ``model_dir`` as the model's preview, replacing any
    existing one, and return the recorded relpath (``preview<ext>``). The caller records
    it in ``model.yaml`` (a relpath — Tenet 6). Raises FileNotFoundError if absent."""
    model_dir = Path(model_dir)
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"preview image not found: {image_path}")
    for old in model_dir.glob("preview.*"):  # never leave a stale prior preview
        old.unlink()
    rel = f"preview{image_path.suffix.lower() or '.png'}"
    shutil.copy2(image_path, model_dir / rel)
    return rel


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_.")
    if not slug:
        raise ValueError("could not derive a model id from the name/filename")
    return slug


def _validate_id(model_id: str) -> None:
    """Keep the global model-id namespace unambiguous for ``--model <id>`` — across every
    built-in registry, since a keypoints ``--model`` and a detection one share it."""
    if (
        model_id in MODELS
        or model_id in EMBEDDING_MODELS
        or model_id in KEYPOINT_MODELS
    ):
        raise ValueError(
            f"{model_id!r} collides with a built-in model id; pass a different --name"
        )
    if _RESERVED_ID.match(model_id):
        raise ValueError(
            f"{model_id!r} matches the reserved experiment-model shape model_NNN; "
            "pass a different --name"
        )
