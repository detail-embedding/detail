"""Training-dataset storage (Section 12).

A dataset lives in ``datasets/<dataset_id>/`` with ``dataset.yaml`` (full
metadata, ``schema_version`` first — Tenet 7), the trainer ``data.yaml`` (YOLO
format, not a DETAIL schema), and ``images/`` + ``labels/`` (``train`` /
``val``). Datasets are discovered by **scanning** ``datasets/`` — no central
mutable list (the same pattern as recordings, runs, and analysis §16); each
``dataset.yaml`` is the source of truth.
"""

import contextlib
import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.storage import dirscan
from detail.storage import experiment as experiment_store

SCHEMA_VERSION = 1
DATASET_FILENAME = "dataset.yaml"


def dataset_dir(experiment_dir: Path | str, dataset_id: str) -> Path:
    return experiment_store.datasets_dir(experiment_dir) / dataset_id


def discover_datasets(experiment_dir: Path | str) -> list[str]:
    """Sorted ids of datasets (subdirs of ``datasets/`` holding a dataset.yaml)."""
    root = experiment_store.datasets_dir(experiment_dir)
    return dirscan.discover_ids(root, DATASET_FILENAME)


def next_dataset_id(experiment_dir: Path | str) -> str:
    """Experiment-monotonic ``dataset_NNN`` id, one past the highest existing."""
    nums = []
    for did in discover_datasets(experiment_dir):
        try:
            nums.append(int(did.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return f"dataset_{max(nums, default=0) + 1:03d}"


def write_dataset_yaml(dataset_dir: Path | str, record: dict) -> None:
    """Write ``dataset.yaml`` with ``schema_version`` as the first key (Tenet 7).

    Atomic (tmp + Windows-safe ``core.atomic.replace``): the labeling / export worker subprocess
    rewrites this at finalize (and on crash via ``set_dataset_status``) while the GUI reads it, so a
    plain truncating write would let a reader parse an empty file — the atomic-write hard rule.
    """
    out = {"schema_version": SCHEMA_VERSION}
    out.update({k: v for k, v in record.items() if k != "schema_version"})
    path = Path(dataset_dir) / DATASET_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(out, f, sort_keys=False)
    atomic.replace(tmp, path)


def read_dataset_yaml(dataset_dir: Path | str) -> dict:
    path = Path(dataset_dir) / DATASET_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {DATASET_FILENAME} in {dataset_dir}")
    return yaml_load.safe_load(atomic.read_text(path))


def set_dataset_status(
    dataset_dir: Path | str, status: str, error: str | None = None
) -> None:
    """Update the ``status`` field of an existing ``dataset.yaml``, in place.

    ``error`` (a failure message) is recorded into the record so readers — the
    GUI detail page in particular — can show why a run failed. Also merges in
    ``duration_s``/``compute``/(if remote and a rate is configured)
    ``estimated_cost_usd`` from the job's own envelope (§ run location, duration,
    cost) — ``dataset_dir`` is the run's own directory (``job.json`` lives right
    beside ``dataset.yaml``), so this needs no extra argument threaded through
    every caller.
    """
    from detail.executor import rundir, run_metrics

    record = read_dataset_yaml(dataset_dir)
    record["status"] = status
    if error is not None:
        record["error"] = error
    with contextlib.suppress(Exception):
        job = rundir.read_job(dataset_dir)
        compute = getattr(job, "compute", None)
        record.update(
            run_metrics.completion_fields(
                dataset_dir, compute.to_dict() if compute is not None else None
            )
        )
    write_dataset_yaml(dataset_dir, record)


def delete_dataset(experiment_dir: Path | str, dataset_id: str) -> None:
    """Remove a dataset's directory and everything it owns. Idempotent — a
    missing directory is a no-op (the store owns its dir ops, like runs)."""
    shutil.rmtree(dataset_dir(experiment_dir, dataset_id), ignore_errors=True)


def write_data_yaml(
    dataset_dir: Path | str,
    names: dict[int, str],
    *,
    kpt_shape: tuple[int, int] | list[int] | None = None,
    flip_idx: list[int] | None = None,
) -> None:
    """Write the trainer-facing ``data.yaml`` (YOLO format).

    ``path: .`` keeps the descriptor portable — no absolute paths in a experiment
    file (Tenet 6). It is *not* resolved relative to this file at train time:
    Ultralytics resolves a relative ``path:`` against the current working
    directory, so the trainer rewrites ``path`` to the absolute dataset dir in a
    machine-local copy before training (see ``UltralyticsYOLOTrainer``). Not a
    DETAIL schema file, so it carries no ``schema_version``.

    A **pose** dataset (§12 *Distill keypoints*) adds ``kpt_shape: [K, 3]`` and ``flip_idx``,
    the index each point swaps with under a left-right flip (from the skeleton's pairs; a
    midline point maps to itself), so Ultralytics' flip augmentation does not teach
    left/right swaps. ``flip_idx`` without ``kpt_shape`` is refused: it describes points.
    """
    if flip_idx is not None and kpt_shape is None:
        raise ValueError("flip_idx describes keypoints; give kpt_shape with it")
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in names.items()},
    }
    if kpt_shape is not None:
        data["kpt_shape"] = [int(v) for v in kpt_shape]
        if flip_idx is not None:
            if len(flip_idx) != int(kpt_shape[0]):
                raise ValueError(
                    f"flip_idx has {len(flip_idx)} entries for {int(kpt_shape[0])} keypoints"
                )
            data["flip_idx"] = [int(i) for i in flip_idx]
    with open(Path(dataset_dir) / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
