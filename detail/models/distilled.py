"""Promote an experiment-trained model into the machine-global catalog (§12).

A distilled model — a YOLO detector trained from SAM3 labels — is usable in its own
experiment immediately (``weights_source: "trained"``). Promoting copies it into the same
machine-global ``imported/`` catalog an imported ``.pt`` lands in (``models/imported.py``),
so any experiment can select it — tagged ``source: "distilled"`` with a ``provenance`` block
(source experiment + dataset + base model + metrics) so a user can see where it came from.

Promotion is **on-demand** (a discrete action, not a distill stage), so a user can iterate on
several distill attempts and promote only the keeper. Provenance is **metadata only** — names,
ids, and metrics, never an absolute path into the source experiment (Tenet 6) — and the weights
are **copied**, so the global model is self-contained and survives deleting the source
experiment. Imports no backend (Tenet 13).

A **keypoint student** (``task: pose``, §12 *Keypoint models in the catalog*) promotes the
same way and carries what it is rebuilt from: its skeleton copy (added to the machine's
skeleton library unless one with its digest is there), ``role: keypoints``, the weights'
sha256 and the teacher runs under ``provenance.teacher``. A pose student without a skeleton
is refused — the catalog copy could not be built.
"""

import shutil

from detail.core import atomic
from detail.core.skeleton import Skeleton
from detail.core.utils import utc_timestamp
from detail.models import imported as imported_store
from detail.storage import models as models_store


def register_distilled_model(
    experiment_root,
    model_id: str,
    *,
    source_experiment_name: str,
    source_experiment_id: str,
    name: str | None = None,
) -> str:
    """Copy a completed experiment-trained model into the machine-global catalog as a
    distilled model, carrying its provenance. Returns the assigned global id.

    Raises:
        FileNotFoundError: The source model (or its best-weights file) does not exist.
        ValueError: The source model is not ``complete``, has no best-weights path, is a
            pose student without a valid skeleton, or the derived global id collides with a built-in id, the reserved ``model_NNN`` shape,
            or an existing catalog entry.
    """
    src_dir = models_store.model_dir(experiment_root, model_id)
    record = models_store.read_model_yaml(src_dir)  # FileNotFoundError if absent
    if record.get("status") != "complete":
        raise ValueError(
            f"model {model_id!r} is {record.get('status')!r}; only a complete model can "
            "be promoted to the global catalog"
        )
    best_rel = (record.get("weights") or {}).get("best")
    if not best_rel:
        raise ValueError(f"model {model_id!r} has no best-weights path to promote")
    best = src_dir / best_rel
    if not best.is_file():
        raise FileNotFoundError(f"best weights missing for model {model_id!r}: {best}")

    pose = imported_store.is_pose_record(record)
    skeleton = None
    if pose:
        if not record.get("skeleton"):
            raise ValueError(
                f"keypoint model {model_id!r} has no skeleton in its model.yaml, so a "
                "catalog copy could not be rebuilt; it cannot be promoted"
            )
        skeleton = Skeleton.from_dict(record["skeleton"])
        skeleton.validate()

    gid = imported_store._slugify(name or record.get("name") or model_id)
    imported_store._validate_id(
        gid
    )  # reject built-in ids + the reserved model_NNN shape
    gdir = imported_store.model_dir(gid)
    if gdir.exists():
        raise ValueError(
            f"a global model {gid!r} already exists; remove it or pass a different name"
        )

    # Built aside and moved into the catalog whole, as an import is
    # (`imported.import_model`): a copy cut short in place left a folder with no model.yaml
    # that blocked the name (#29).
    with atomic.staged_dir(gdir) as stage:
        (stage / "weights").mkdir()
        shutil.copy2(
            best, stage / imported_store.WEIGHTS_RELPATH
        )  # self-contained copy
        # Carry the auto-captured overlay preview (§12) if the trained model has one, so the
        # promoted model shows the same thumbnail the experiment model does.
        preview_rel = _copy_preview(src_dir, record.get("preview"), stage)
        promoted = {
            "id": gid,
            "name": name or record.get("name") or gid,
            "source": "distilled",
            "model_family": record.get("model_family", "yolo"),
            "task": record.get("task", "segment"),
            "classes": record.get("classes", []),
            "weights": {"best": imported_store.WEIGHTS_RELPATH},
            "status": "complete",
            "created_at": utc_timestamp(),
            "made_by": record.get("made_by", ""),
            "description": record.get("description", ""),
            "notes": record.get("notes", ""),
            "preview": preview_rel,
            "provenance": {
                # A list so one model can name several source experiments later; the
                # distill flow trains from one experiment today, so it writes one entry.
                "source_experiments": [
                    {"name": source_experiment_name, "id": source_experiment_id}
                ],
                "source_dataset_id": record.get("dataset_id"),
                "base_model": record.get("base_model"),
                "trainer_id": record.get("trainer_id"),
                "distilled_at": utc_timestamp(),
                "metrics": record.get("metrics", {}),
                # What trained it travels with it, or a promoted model forgets.
                "hyperparams": record.get("hyperparams", {}),
                "trained_with": record.get("trained_with", {}),
            },
        }
        if pose:
            if record.get("teacher"):
                promoted["provenance"]["teacher"] = record["teacher"]
            library_name = (record.get("skeleton_ref") or {}).get("name") or (
                f"{promoted['name']} skeleton"
            )
            promoted.update(
                imported_store.pose_record_fields(
                    skeleton, library_name, stage / imported_store.WEIGHTS_RELPATH
                )
            )
        models_store.write_model_yaml(stage, promoted)
    return gid


def _copy_preview(src_dir, preview_rel: str | None, dest_dir) -> str | None:
    """Copy a trained model's preview image into the promoted model dir, if it has one.
    Returns the recorded relpath (unchanged) or None."""
    if not preview_rel:
        return None
    src = src_dir / preview_rel
    if not src.is_file():
        return None
    shutil.copy2(src, dest_dir / preview_rel)
    return preview_rel
