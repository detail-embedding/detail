"""Trained-model storage (Section 12).

A trained model lives in ``models/<model_id>/`` with ``model.yaml`` (full
metadata, ``schema_version`` first — Tenet 7) and ``run/`` (the trainer's own
output: weights, logs). ``run/`` is trainer-owned scratch and may contain
machine-local paths; DETAIL's authoritative files (``model.yaml``, the
weight paths it records relative to the model dir) are what every other stage
reads. Models are discovered by **scanning** ``models/`` — no central mutable
list (the same pattern as recordings, runs, datasets, and analysis §16).
"""

import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import dirscan
from detail.storage import experiment as experiment_store
from detail.storage import id_allocator

SCHEMA_VERSION = 1
MODEL_FILENAME = "model.yaml"


def model_dir(experiment_dir: Path | str, model_id: str) -> Path:
    return experiment_store.models_dir(experiment_dir) / model_id


def discover_models(experiment_dir: Path | str) -> list[str]:
    """Sorted ids of models (subdirs of ``models/`` holding a model.yaml)."""
    root = experiment_store.models_dir(experiment_dir)
    return dirscan.discover_ids(root, MODEL_FILENAME)


def next_model_id(experiment_dir: Path | str) -> str:
    """Claim the next ``model_NNN`` id (creating its directory); a deleted model's id is never
    handed out again (``storage/id_allocator.py``)."""
    return id_allocator.claim_dir(
        experiment_store.models_dir(experiment_dir), "model", digits=3
    )


def write_model_yaml(model_dir: Path | str, record: dict) -> None:
    """Write ``model.yaml`` with ``schema_version`` as the first key (Tenet 7).

    Atomic (tmp + Windows-safe ``core.atomic.replace``): the training worker subprocess rewrites
    the authoritative status here while the GUI reads it, so a plain truncating write would let a
    reader parse an empty file — the atomic-write hard rule."""
    out = {"schema_version": SCHEMA_VERSION}
    out.update({k: v for k, v in record.items() if k != "schema_version"})
    path = Path(model_dir) / MODEL_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(out, f, sort_keys=False)
    atomic.replace(tmp, path)


def read_model_yaml(model_dir: Path | str) -> dict:
    path = Path(model_dir) / MODEL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {MODEL_FILENAME} in {model_dir}")
    return yaml.safe_load(atomic.read_text(path))


def set_model_status(
    model_dir: Path | str, status: str, error: str | None = None
) -> None:
    """Update the ``status`` field of an existing ``model.yaml``, in place.

    ``error`` (a failure message) is recorded into the record so readers — the
    GUI detail page in particular — can show why a run failed.
    """
    record = read_model_yaml(model_dir)
    record["status"] = status
    if error is not None:
        record["error"] = error
    write_model_yaml(model_dir, record)


def delete_model(experiment_dir: Path | str, model_id: str) -> None:
    """Remove a model's directory (model.yaml + trainer run scratch + weights).
    Idempotent — a missing directory is a no-op (the store owns its dir ops)."""
    shutil.rmtree(model_dir(experiment_dir, model_id), ignore_errors=True)
