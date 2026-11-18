"""Experiment-level comparison records: ``comparisons/<id>/comparison.yaml`` (§24).

A *comparison* answers "does this differ between these groups?" — it filters recordings, groups
them by columns of ``metadata.csv`` (§11), and summarizes §23 measures over the §19 behavior
artifact. The record holds what was asked (source artifact, filter, group-by, aggregation path,
windows, measures, seed) plus publication-grade provenance, because these numbers go into papers
and a figure without its inputs recorded is not reproducible.

**Not a benchmark.** The removed ``storage/benchmarks.py`` (§18) held the other thing: a set of
*evaluation* runs varying one method dimension, scored against ground truth. They shared this
directory and this filename until the §18 rename; they have nothing in common but an export tree.

Same discipline as every sibling store: discovered by scanning ``comparisons/``, written atomically
with ``schema_version`` first, never a central mutable list.
"""

import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import dirscan
from detail.storage import experiment as experiment_store
from detail.storage import id_allocator

SCHEMA_VERSION = 1
COMPARISON_FILENAME = "comparison.yaml"


def comparisons_dir(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / "comparisons"


def comparison_dir(experiment_dir: Path | str, comparison_id: str) -> Path:
    return comparisons_dir(experiment_dir) / comparison_id


def discover_comparisons(experiment_dir: Path | str) -> list[str]:
    """Sorted ``cmp_NN`` ids, from the folders that actually carry a record.

    A directory without ``comparison.yaml`` is not a comparison — a half-written or hand-made
    folder is skipped rather than surfacing as an id that fails to read.
    """
    return dirscan.discover_ids(comparisons_dir(experiment_dir), COMPARISON_FILENAME)


def next_comparison_id(experiment_dir: Path | str) -> str:
    """Claim the next ``cmp_NN`` id (creating its directory); a deleted comparison's id is
    never handed out again (``storage/id_allocator.py``)."""
    return id_allocator.claim_dir(comparisons_dir(experiment_dir), "cmp", digits=2)


def write_comparison_yaml(comparison_dir: Path | str, record: dict) -> None:
    """Write the record atomically, ``schema_version`` first.

    Atomic because a reader — the GUI list, a re-export — may look while this writes, and a
    truncating open lets it parse an empty file (the concurrently-read-file rule).
    """
    cdir = Path(comparison_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in record.items():
        if key != "schema_version":
            ordered[key] = value
    path = cdir / COMPARISON_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(ordered, handle, sort_keys=False)
    atomic.replace(tmp, path)


def read_comparison_yaml(comparison_dir: Path | str) -> dict:
    path = Path(comparison_dir) / COMPARISON_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {COMPARISON_FILENAME} in {comparison_dir}")
    record = yaml.safe_load(atomic.read_text(path))
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not a mapping")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version is {record.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    return record


def delete_comparison(experiment_dir: Path | str, comparison_id: str) -> None:
    """Remove a comparison and everything it wrote, exports included.

    There is no matching ``set_status``. ``run_comparison`` is synchronous: it computes in one
    pass and writes the record once, at the end, already ``complete``. A failure partway writes
    nothing at all, so there is no record left behind to mark ``failed`` — the status a mutator
    would exist to set is one no reader can ever observe.

    The removed ``storage/benchmarks.py`` had a ``set_benchmark_status``, but it was not evidence
    for keeping one here: it had no caller either and sat in the orphan allow list. The transition
    becomes real if a comparison ever gains an executor-backed run — a record written at launch and
    updated on completion — and it should be written then, against that caller.
    """
    cdir = comparison_dir(experiment_dir, comparison_id)
    if cdir.is_dir():
        shutil.rmtree(cdir)
