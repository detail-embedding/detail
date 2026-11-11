"""A generated id is sorted by its number, not its text.

Every id the tool mints is `<stem>_<n>` zero-padded to a fixed width. Sorting those as
strings is correct only while the number stays inside the pad: at the boundary `an_100`
sorts before `an_99` and `run_1000` before `run_999`, and every listing built on `sorted()`
silently reorders itself once an experiment grows past it.

Already true on real data: an experiment here holds 744 analysis runs whose highest id is
`an_795` (2-digit pad). It sorts to row 724, and the Analysis table, the project rail and
`detail analysis list` all end at `an_99`.

PR #170 removed a re-sort from the project rail on the grounds that the store already
guarantees chronological order. This is the assumption underneath that.
"""

import re
from pathlib import Path

import pytest

import detail
from detail.core.ids import numeric_id_key, sorted_by_id
from detail.storage import analysis_runs, datasets
from detail.storage import distill_runs, experiment, export_runs, models, runs

# Every allocator: (module, discover fn, containing dir under the experiment, marker file).
_FAMILIES = {
    "analysis": (analysis_runs, "discover_analyses", "analysis", "analysis.yaml"),
    "dataset": (datasets, "discover_datasets", "datasets", "dataset.yaml"),
    "distill": (distill_runs, "discover_runs", "distill", "distill.yaml"),
    "export": (export_runs, "discover_exports", "exports", "export_run.yaml"),
    "model": (models, "discover_models", "models", "model.yaml"),
    "recording": (experiment, "discover_recordings", "recordings", "recording.yaml"),
    "run": (runs, "discover_runs", "runs", "run.yaml"),
}


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #
def test_ids_order_by_number_across_the_pad_boundary():
    assert sorted_by_id(["an_100", "an_9", "an_795", "an_10", "an_99"]) == [
        "an_9",
        "an_10",
        "an_99",
        "an_100",
        "an_795",
    ]
    assert sorted_by_id(["run_1000", "run_999", "run_001"]) == [
        "run_001",
        "run_999",
        "run_1000",
    ]


def test_the_order_is_total_even_for_names_that_do_not_conform():
    """A directory can hold anything; the listing must still be deterministic."""
    names = ["an_2", "scratch", "an_10", "notes.bak"]
    assert sorted_by_id(names) == sorted_by_id(list(reversed(names)))
    assert set(sorted_by_id(names)) == set(names)


def test_the_key_separates_stems_before_numbers():
    assert numeric_id_key("an_10") < numeric_id_key("bn_2")
    assert numeric_id_key("an_2") < numeric_id_key("an_10")


# --------------------------------------------------------------------------- #
# every allocator, through the real function
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_discover_orders_numerically(tmp_path, family):
    module, fn_name, subdir, marker = _FAMILIES[family]
    stem = {"recording": "recording", "run": "run", "distill": "distill"}.get(
        family, family
    )
    ids = [f"{stem}_9", f"{stem}_10", f"{stem}_99", f"{stem}_100", f"{stem}_795"]
    root = tmp_path / subdir
    for name in reversed(ids):  # created out of order, so nothing rides on mtime
        (root / name).mkdir(parents=True)
        (root / name / marker).write_text("schema_version: 1\n", encoding="utf-8")

    assert getattr(module, fn_name)(tmp_path) == ids


# --------------------------------------------------------------------------- #
# the guard: a ninth allocator must not reintroduce it
# --------------------------------------------------------------------------- #
_STORAGE = Path(detail.__file__).parent / "storage"
_PLAIN_SORTED = re.compile(r"\bsorted\(")


def _discover_bodies():
    for path in sorted(_STORAGE.glob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        for match in re.finditer(r"^def (discover_\w+)\(", source, re.M):
            start = match.start()
            end = source.find("\ndef ", start + 1)
            yield path.name, match.group(1), source[
                start : end if end > 0 else len(source)
            ]


def _sorts_through_the_key(body: str) -> bool:
    """Whether a ``discover_*`` body orders through ``core.ids``.

    Either directly, or by delegating the whole listing to ``dirscan.discover_ids`` --
    the shared scandir enumeration, which is itself a ``discover_*`` this scan checks,
    so the delegation cannot be where the ordering is quietly lost.
    """
    if "dirscan.discover_ids(" in body:
        return True
    return "sorted_by_id(" in body and not _PLAIN_SORTED.search(body)


def test_the_shared_enumeration_is_itself_scanned():
    """The delegation is only safe while ``dirscan.discover_ids`` is in the sweep."""
    assert ("dirscan.py", "discover_ids") in {
        (file, fn) for file, fn, _body in _discover_bodies()
    }


def test_every_discover_function_sorts_through_the_shared_key():
    offenders = [
        f"{file}::{fn}"
        for file, fn, body in _discover_bodies()
        if not _sorts_through_the_key(body)
    ]
    assert not offenders, (
        "a generated id sorted as a string is only ordered below its pad width, and every "
        f"listing inherits it: {offenders}. Sort through core.ids.sorted_by_id."
    )


def test_the_scan_covers_every_family_the_tests_know_about():
    """Guard the guard, both ways: the scan must see real functions, and the behavioural
    table above must not go stale as allocators are added."""
    # keyed on (file, function): `discover_runs` exists in both runs.py and distill_runs.py
    found = {(file, fn) for file, fn, _body in _discover_bodies()}
    assert len(found) >= len(_FAMILIES)
    covered = {
        (Path(module.__file__).name, fn) for module, fn, _d, _f in _FAMILIES.values()
    }
    assert covered <= found


def test_the_scan_would_catch_a_plain_sorted():
    body = "def discover_things(d):\n    return sorted(p.name for p in d.iterdir())\n"
    assert not _sorts_through_the_key(body)
    # ...and delegating is not a way past it: a body that re-sorts the shared helper's
    # output by text is still an offender.
    delegating = (
        "def discover_things(d):\n    return sorted(dirscan.dirs_containing(d, 'x'))\n"
    )
    assert not _sorts_through_the_key(delegating)
