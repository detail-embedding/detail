"""A crash mid-write must not leave a record that takes a whole page down with it.

**The class — two halves of one failure.** *A record swapped into place before its bytes are
durable*, read by *a listing that fails whole when one of its records cannot be parsed*.

The instance: a distill run was labeling when the machine lost power. Its ``distill.yaml``,
``dataset.yaml`` and ``job.json`` had each just been rewritten through ``core.atomic.replace``,
which renamed the temp file into place without flushing it. NTFS journals the rename and the
new length as metadata and writes the data later, so after the reboot all three files were the
right size and **every byte was zero**. Then every listing that read them — the Models page
and the Datasets page among them — raised ``yaml.ReaderError`` on the first NUL, and the user
could not reach the app screen that would have let them delete the dead run.

**How it hid.** Both halves are invisible to behavioural tests. A missing flush is correct at
every line where it is written, and only an unclean shutdown between two syscalls exposes it.
A bare listing is correct as long as the records are, and every fixture writes good records.
``api/experiment.py::export_runs`` already skipped an unparseable record and said why;
nine other listings did not, and nothing compared them.

**The rule.**

- ``core.atomic.replace`` flushes ``src`` before renaming it, so the swap yields the old
  record or the new one and never a zero-filled one. Every writer in the package goes
  through it (``tests/storage/test_zarr_seam.py`` forbids a bare ``os.replace``), so the
  flush lives there rather than at each writer.
- A listing — a view over every record of a kind — reads through ``storage.dirscan.read_each``,
  which leaves out an unreadable record, logs it, and keeps the rest. A lookup of *one* record
  by id still raises: the caller asked for that record.

The scan below is the fail-closed half: a loop over a ``discover_*`` scan that reads each
record bare, outside a ``try``, fails here unless it is listed in ``NOT_LISTINGS`` with why.
"""

import ast
import os
import pathlib
import re
from types import SimpleNamespace

import pytest

from detail.api.authored_run import AuthoredRun
from detail.api.experiment import Experiment
from detail.core import atomic
from detail.models import imported
from detail.storage import analysis_runs as analysis_store
from detail.storage import datasets as datasets_store
from detail.storage import distill_runs as distill_store
from detail.storage import export_runs as export_runs_store
from detail.storage import locations
from detail.storage import models as models_store
from detail.storage import runs as runs_store
from detail.storage import snapshots as snapshots_store
from detail.storage import tool_runs as tool_runs_store

REPO = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO / "detail"

# --------------------------------------------------------------------------- #
# Write side: the flush comes before the rename
# --------------------------------------------------------------------------- #


def test_replace_flushes_the_new_contents_before_renaming(tmp_path, monkeypatch):
    src = tmp_path / "record.yaml.tmp"
    dst = tmp_path / "record.yaml"
    src.write_bytes(b"id: run_001\n")
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", os.fstat(fd).st_size))
        real_fsync(fd)

    def replace(a, b):
        events.append(("replace", pathlib.Path(a).name))
        real_replace(a, b)

    monkeypatch.setattr(atomic.os, "fsync", fsync)
    monkeypatch.setattr(atomic.os, "replace", replace)

    atomic.replace(src, dst)

    assert events == [("fsync", len(b"id: run_001\n")), ("replace", src.name)]
    assert dst.read_bytes() == b"id: run_001\n"


def test_replace_moves_a_directory_without_flushing_it(tmp_path, monkeypatch):
    staged = tmp_path / "view.zarr.staged"
    staged.mkdir()
    (staged / "zarr.json").write_text("{}", encoding="utf-8")
    flushed = []
    monkeypatch.setattr(atomic.os, "fsync", flushed.append)

    atomic.replace(staged, tmp_path / "view.zarr")

    assert flushed == []
    assert (tmp_path / "view.zarr" / "zarr.json").is_file()


# --------------------------------------------------------------------------- #
# Read side: one unreadable record costs its row, not the listing
# --------------------------------------------------------------------------- #


def _processing_run(e, rid):
    runs_store.write_run(e, rid, {"id": rid, "status": "complete"})
    return runs_store.run_dir(e, rid) / runs_store.RUN_FILENAME


def _distill_run(e, rid):
    distill_store.write_run(e, rid, {"id": rid, "status": "complete"})
    return distill_store.run_dir(e, rid) / distill_store.DISTILL_FILENAME


def _dataset(e, rid):
    ddir = datasets_store.dataset_dir(e, rid)
    ddir.mkdir(parents=True, exist_ok=True)
    datasets_store.write_dataset_yaml(ddir, {"id": rid, "status": "complete"})
    return ddir / datasets_store.DATASET_FILENAME


def _model(e, rid):
    mdir = models_store.model_dir(e, rid)
    mdir.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(mdir, {"id": rid, "status": "complete"})
    return mdir / models_store.MODEL_FILENAME


def _analysis(e, rid):
    adir = analysis_store.analysis_dir(e, rid)
    adir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(adir, {"id": rid, "status": "complete"})
    return adir / analysis_store.ANALYSIS_FILENAME


def _export(e, rid):
    xdir = export_runs_store.export_dir(e, rid)
    xdir.mkdir(parents=True, exist_ok=True)
    export_runs_store.write_export_yaml(xdir, {"id": rid, "status": "complete"})
    return xdir / export_runs_store.EXPORT_FILENAME


def _tool_run(_e, rid):
    tool_runs_store.write_tool_run("probe", rid, {"id": rid, "status": "complete"})
    return tool_runs_store.tool_run_dir("probe", rid) / tool_runs_store.RUN_FILENAME


def _imported_model(_e, rid):
    mdir = imported.model_dir(rid)
    mdir.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(mdir, {"id": rid, "status": "complete"})
    return mdir / models_store.MODEL_FILENAME


def _snapshot(e, rid):
    sdir = snapshots_store.snapshot_dir(e / "analysis" / "analysis_001", rid)
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / snapshots_store.SNAPSHOT_FILENAME
    path.write_text(f"schema_version: 1\nid: {rid}\n", encoding="utf-8")
    return path


def _snapshots(e):
    run = SimpleNamespace(_run_dir=lambda: e / "analysis" / "analysis_001")
    return AuthoredRun.snapshots(run)


def _ids(rows):
    return sorted(r["id"] if isinstance(r, dict) else r.id for r in rows)


#: ``(name, write one record -> its file, the listing a user reaches)``. The listing is taken
#: through the API where one exists, because that is the surface a page reads.
LISTINGS = [
    ("processing_runs", _processing_run, lambda e: Experiment(e).processing_runs),
    ("distill_runs", _distill_run, lambda e: Experiment(e).distill_runs),
    ("datasets", _dataset, lambda e: Experiment(e).datasets),
    ("models", _model, lambda e: Experiment(e).models),
    ("analysis_runs", _analysis, lambda e: Experiment(e).analysis_runs),
    ("export_runs", _export, lambda e: Experiment(e).export_runs),
    ("tool_runs", _tool_run, lambda e: tool_runs_store.list_tool_runs()),
    ("imported_models", _imported_model, lambda e: imported.list_records(role=None)),
    ("snapshots", _snapshot, _snapshots),
]

#: How a crash leaves a record. ``zero_filled`` is the NTFS shape that was observed — the length
#: survives, the data does not. ``empty`` is the truncated shape; it parses to ``None`` rather
#: than raising, which is the case a bare ``except`` around the parse would miss.
DAMAGE = {
    "zero_filled": lambda size: b"\0" * size,
    "empty": lambda size: b"",
}


def _prefix(name: str) -> str:
    return {"snapshots": "snap", "tool_runs": "run"}.get(name, name.rstrip("s"))


@pytest.mark.parametrize("damage", sorted(DAMAGE))
@pytest.mark.parametrize(
    ("name", "write", "listing"), LISTINGS, ids=[row[0] for row in LISTINGS]
)
def test_one_unreadable_record_costs_its_row_not_the_listing(
    experiment_dir, tmp_path, monkeypatch, name, write, listing, damage
):
    # Tool runs live beside the experiments, not in one; the default root is the user's.
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path / "workspace")
    prefix = _prefix(name)
    broken = write(experiment_dir, f"{prefix}_001")
    write(experiment_dir, f"{prefix}_002")
    broken.write_bytes(DAMAGE[damage](broken.stat().st_size))

    assert _ids(listing(experiment_dir)) == [f"{prefix}_002"]


def test_a_lookup_by_id_still_raises_on_an_unreadable_record(experiment_dir):
    path = _distill_run(experiment_dir, "distill_001")
    path.write_bytes(b"\0" * path.stat().st_size)

    # Raised, not skipped: the listing above tolerates the record, a lookup by id must not. What
    # it raises is PyYAML's "special characters are not allowed" with no path no longer; it is
    # the damaged-file error, which names this file (`tests/core/test_damaged_files.py`).
    with pytest.raises(atomic.DamagedFileError, match="distill.yaml is all zero bytes"):
        distill_store.read_run(experiment_dir, "distill_001")


# --------------------------------------------------------------------------- #
# Fail-closed scan: a listing that reads its records bare
# --------------------------------------------------------------------------- #

_SCAN = re.compile(r"discover|^_run_dirs$|^dirs_containing$")
_READ = re.compile(r"^read|^safe_load$|^loads$")

#: Loops over a record scan that read bare on purpose, and why. **An entry is a claim**: one the
#: scan no longer flags fails below rather than sitting here excusing nothing.
NOT_LISTINGS: dict[str, str] = {}


def _name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _reads(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and _READ.search(_name(n.func)) for n in ast.walk(node)
    )


def _scans(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and bool(_SCAN.search(_name(node.func)))


def bare_listings(source: str, rel: str) -> list[str]:
    """``path::function`` for every loop over a record scan that reads a record outside a
    ``try``. A comprehension can hold no ``try``, so any read inside one counts."""
    tree = ast.parse(source)
    found = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
                bare = _scans(node.generators[0].iter) and _reads(node.elt)
            elif isinstance(node, ast.DictComp):
                bare = _scans(node.generators[0].iter) and (
                    _reads(node.key) or _reads(node.value)
                )
            elif isinstance(node, ast.For):
                bare = _scans(node.iter) and any(
                    _reads(stmt) for stmt in node.body if not isinstance(stmt, ast.Try)
                )
            else:
                continue
            if bare:
                found.append(f"{rel}::{func.name}")
    return sorted(set(found))


def _package_bare_listings() -> list[str]:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        found += bare_listings(path.read_text(encoding="utf-8-sig"), rel)
    return found


def test_no_listing_reads_its_records_bare():
    offenders = [site for site in _package_bare_listings() if site not in NOT_LISTINGS]
    assert not offenders, (
        "a listing reads each record outside a try, so one unreadable record fails the "
        "whole listing — read through storage.dirscan.read_each (or record it in "
        f"NOT_LISTINGS with why): {offenders}"
    )


def test_every_not_listing_entry_still_excuses_something():
    stale = sorted(set(NOT_LISTINGS) - set(_package_bare_listings()))
    assert not stale, f"NOT_LISTINGS entries the scan no longer flags: {stale}"


@pytest.mark.parametrize(
    "source",
    [
        "def f(e):\n    return [read_run(e, r) for r in discover_runs(e)]\n",
        "def f(e):\n    out = []\n    for r in store.discover_models(e):\n"
        "        out.append(store.read_model_yaml(r))\n    return out\n",
        "def f(t):\n    return {d: yaml.safe_load(d) for d in _run_dirs(t)}\n",
    ],
    ids=["comprehension", "for_loop", "dict_comprehension"],
)
def test_scan_catches_a_bare_listing(source):
    assert bare_listings(source, "planted.py") == ["planted.py::f"]


def test_scan_accepts_a_guarded_read():
    source = (
        "def f(e):\n    for r in discover_runs(e):\n        try:\n"
        "            rec = read_run(e, r)\n        except ValueError:\n            continue\n"
    )
    assert bare_listings(source, "planted.py") == []
