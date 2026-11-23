"""Detection and tracking setting tunings of a trained model (§12 *Tune detection and tracking
settings*).

A tuning lives with the model it measured, in ``models/<model_id>/tuning/<tuning_id>/``, whoever
started it: a distill's ``tune`` stage or a standalone tune of an existing model. Its
``tuning.yaml`` (``schema_version`` first) holds the sweep's table and pick, and ``source``
says which of the two made it. The run directory is also the job's (``job.json``); the scratch
detections a sweep writes there are removed when it finishes.

Tunings are found by scanning, newest first, so a model tuned twice offers the newest and can
say which one that is.
"""

from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import dirscan
from detail.storage import id_allocator
from detail.storage import models as models_store
from detail.storage import user_yaml

SCHEMA_VERSION = 1
TUNING_FILENAME = "tuning.yaml"
TUNING_DIRNAME = "tuning"


def tunings_root(experiment_dir: Path | str, model_id: str) -> Path:
    return models_store.model_dir(experiment_dir, model_id) / TUNING_DIRNAME


def tuning_dir(experiment_dir: Path | str, model_id: str, tuning_id: str) -> Path:
    return tunings_root(experiment_dir, model_id) / tuning_id


def next_tuning_id(experiment_dir: Path | str, model_id: str) -> str:
    """Claim the next ``tune_NNN`` id under the model (creating its directory)."""
    root = tunings_root(experiment_dir, model_id)
    root.mkdir(parents=True, exist_ok=True)
    return id_allocator.claim_dir(root, "tune", digits=3)


def write_tuning(directory: Path | str, record: dict) -> None:
    """Write ``tuning.yaml`` atomically with ``schema_version`` first (Tenet 7): the worker
    rewrites it while the GUI may be reading it."""
    out = {"schema_version": SCHEMA_VERSION}
    out.update({k: v for k, v in record.items() if k != "schema_version"})
    path = Path(directory) / TUNING_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(out, f, sort_keys=False)
    atomic.replace(tmp, path)


def read_tuning(directory: Path | str) -> dict:
    path = Path(directory) / TUNING_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {TUNING_FILENAME} in {directory}")
    return user_yaml.read_user_yaml(path)


def list_tunings(experiment_dir: Path | str, model_id: str) -> list[dict]:
    """Every tuning record of the model, newest first (by ``created_at``, then id)."""
    root = tunings_root(experiment_dir, model_id)
    records = []
    for tuning_id in dirscan.discover_ids(root, TUNING_FILENAME):
        try:
            records.append(read_tuning(root / tuning_id))
        except FileNotFoundError:
            continue
    return sorted(
        records,
        key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or "")),
        reverse=True,
    )


def latest_complete(
    experiment_dir: Path | str, model_id: str
) -> tuple[dict | None, int]:
    """The newest complete tuning of the model and how many complete ones there are, so a
    caller offering it can say which one it took when there is more than one."""
    done = [
        r
        for r in list_tunings(experiment_dir, model_id)
        if r.get("status") == "complete"
    ]
    return (done[0] if done else None), len(done)
