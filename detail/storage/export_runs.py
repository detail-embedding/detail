"""Experiment-level export-run records: ``exports/<run_id>/export_run.yaml`` (#354).

An *export run* is one thing a user asked for — a subject, some targets, a scope — carried out
as a single executor job so it survives the app closing. ``Experiment.run_export`` submits it;
its run directory ``exports/<run_id>/`` holds the executor envelope (``job.json``) and the
record. The **exported files do not live here**: they go to the folder the user chose, outside
the experiment, and the record names it through a machine-local alias (Tenet 6). Runs are
discovered by scanning ``exports/`` — no central mutable list, the same as analyses and datasets.

**One request is one run**, however many targets it covers. Eight targets are eight *units* of
one run, with one folder and one manifest between them, because that is what the user asked for
once. The per-target outcomes live in the record so a `partial` run can say which target failed
and why.

Cloned from ``analysis_runs.py`` and not from ``datasets``/``models``: those predate the
atomic-write rule and their own sibling calls them "borderline, not a template here". The record
is written **atomically** (sibling ``*.tmp`` + ``os.replace``) because the runner subprocess
rewrites it on completion while the GUI's progress view reads it.

The filename is ``export_run.yaml``, not ``export.yaml``: that name is already taken twice — by
``storage/export.py``'s bundle manifest and by the manifest ``write_export`` drops in every
export folder — and three documents behind one name is how a reader stops trusting any of them.
"""

import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import dirscan
from detail.storage import experiment as experiment_store
from detail.storage import read_cache

SCHEMA_VERSION = 1
EXPORT_FILENAME = "export_run.yaml"


def export_dir(experiment_dir: Path | str, run_id: str) -> Path:
    return experiment_store.exports_dir(experiment_dir) / run_id


def discover_exports(experiment_dir: Path | str) -> list[str]:
    """Sorted ids of export runs (subdirs of ``exports/`` holding the record)."""
    root = experiment_store.exports_dir(experiment_dir)
    return dirscan.discover_ids(root, EXPORT_FILENAME)


def _highest_claimed(experiment_dir: Path | str) -> int:
    """The highest ``export_NN`` number **claimed**, whether or not it has a record yet.

    Deliberately not ``discover_exports``, which counts only directories holding a record — the
    right definition for listing, the wrong one for allocating: a directory reserved a moment ago
    has no record yet, and skipping it hands the same id out again.
    """
    root = experiment_store.exports_dir(experiment_dir)
    nums = []
    for path in dirscan.child_dirs(root):
        try:
            nums.append(int(path.name.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return max(nums, default=0)


def reserve_export_id(experiment_dir: Path | str, attempts: int = 50) -> str:
    """Claim the next ``export_NN`` id by creating its directory — safe under concurrency.

    Scanning for max+1 is a *read*: two exports launched at once both see the same highest id
    and the second overwrites the first's record — a wrong answer rather than an error. Creating
    the directory with ``exist_ok=False`` makes the claim atomic, so a loser rescans and takes
    the next one. The same idiom as :func:`analysis_runs.reserve_analysis_id`.

    Raises:
        RuntimeError: If ``attempts`` consecutive claims are all lost.
    """
    root = Path(experiment_dir)
    experiment_store.exports_dir(root).mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        run_id = f"export_{_highest_claimed(root) + 1:03d}"
        try:
            export_dir(root, run_id).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_id
    raise RuntimeError(
        f"could not reserve an export id under {root} after {attempts} attempts — "
        "too many exports are starting at once"
    )


def write_export_yaml(run_dir: Path | str, record: dict) -> None:
    """Write ``export_run.yaml`` with ``schema_version`` as the first key (Tenet 7).

    Atomic (tmp + ``os.replace``) — the runner subprocess rewrites this on completion while the
    GUI detail page reads it; a reader never sees a torn file.
    """
    rdir = Path(run_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in record.items():
        if key != "schema_version":
            ordered[key] = value
    path = rdir / EXPORT_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, path)


def read_export_yaml(run_dir: Path | str) -> dict:
    """Read one export-run record.

    Raises:
        FileNotFoundError: If the record is absent.
    """
    path = Path(run_dir) / EXPORT_FILENAME
    try:
        return read_cache.cached_document(
            path, lambda: yaml.safe_load(atomic.read_text(path))
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"no {EXPORT_FILENAME} in {run_dir}") from None


def set_export_status(
    run_dir: Path | str, status: str, error: str | None = None
) -> None:
    """Update the ``status`` field of an existing record, in place.

    ``error`` is recorded so a surface can say why a run failed. Nothing validates ``status``:
    the reconciler writes whatever the executor envelope settled on, and a store that argued
    with it would be a second opinion about a fact it does not own.
    """
    record = read_export_yaml(run_dir)
    record["status"] = status
    if error is not None:
        record["error"] = error
    write_export_yaml(run_dir, record)


def delete_export(experiment_dir: Path | str, run_id: str) -> None:
    """Remove an export run's directory. Idempotent.

    Deletes the *record*, never the exported files: those are in the user's own folder, outside
    the experiment, and deleting a run is not a request to delete what it produced.
    """
    shutil.rmtree(export_dir(experiment_dir, run_id), ignore_errors=True)
