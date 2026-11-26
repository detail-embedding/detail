"""Saved recording-visualization recipes (DESIGN_PLAN §22 "Recording visualization").

A recipe is a ``RecordingVisualizationConfig`` dump, kept beside the clustering run it draws:
``analysis/<an_NN>/visualizations/<viz_id>/visualization.yaml``. Nothing rendered is stored, so a
recipe never goes stale and the experiment never holds video. Written through ``tmp`` +
``core.atomic.replace`` with ``schema_version`` first, read through ``core.atomic.read_text``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import analysis_runs
from detail.storage import id_allocator
from detail.storage import user_yaml

SCHEMA_VERSION = 1
DIRNAME = "visualizations"
FILENAME = "visualization.yaml"


def visualizations_dir(experiment_dir: Path | str, run_id: str) -> Path:
    """Where a clustering run's recipes live."""
    return analysis_runs.analysis_dir(experiment_dir, run_id) / DIRNAME


def recipe_path(experiment_dir: Path | str, run_id: str, viz_id: str) -> Path:
    return visualizations_dir(experiment_dir, run_id) / viz_id / FILENAME


def list_recipes(experiment_dir: Path | str, run_id: str) -> list[str]:
    """The run's saved recipe ids, oldest first."""
    root = visualizations_dir(experiment_dir, run_id)
    if not root.is_dir():
        return []
    ids = [p.name for p in root.iterdir() if (p / FILENAME).is_file()]
    return sorted(ids, key=_number)


def claim_id(experiment_dir: Path | str, run_id: str) -> str:
    """Claim the next ``viz_NNN`` (its folder made), never reusing one: a deleted
    visualization's number is not handed to the next (`storage/id_allocator`)."""
    return id_allocator.claim_dir(
        visualizations_dir(experiment_dir, run_id), "viz", digits=3
    )


def write_recipe(
    experiment_dir: Path | str, run_id: str, viz_id: str, recipe: dict
) -> Path:
    """Write one recipe, ``schema_version`` first, atomically."""
    path = recipe_path(experiment_dir, run_id, viz_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION, "id": viz_id, "run_id": run_id}
    for key, value in recipe.items():
        if key not in ordered:
            ordered[key] = value
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, path)
    return path


def read_recipe(experiment_dir: Path | str, run_id: str, viz_id: str) -> dict:
    """One recipe as saved, its bookkeeping keys included.

    Raises:
        FileNotFoundError: The run has no recipe ``viz_id``.
        ValueError: The file is not a recipe this version can read.
    """
    path = recipe_path(experiment_dir, run_id, viz_id)
    if not path.is_file():
        raise FileNotFoundError(f"{run_id} has no saved visualization {viz_id}")
    data = user_yaml.read_user_yaml(path)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is not a visualization recipe this version reads "
            f"(schema_version {data.get('schema_version') if isinstance(data, dict) else None})"
        )
    return data


def delete_recipe(experiment_dir: Path | str, run_id: str, viz_id: str) -> None:
    """Remove one recipe; the number is free again only once its folder is gone."""
    import shutil

    folder = visualizations_dir(experiment_dir, run_id) / viz_id
    if not (folder / FILENAME).is_file():
        raise FileNotFoundError(f"{run_id} has no saved visualization {viz_id}")
    shutil.rmtree(folder)


def _number(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 0
