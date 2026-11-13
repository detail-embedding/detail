"""Every example set a keypoint detect run learned from is read through one function.

The class: a question asked of a run's *one* input after the run can have several. A per-class
detect run (§26 *One example set per detection class*) records one example-set snapshot per
detection class under ``example_sets`` and leaves the single ``example_set`` null. Every check
that asks "did this run learn from that set" — the delete refusal, *Tune tracking*'s refusal to
score on a set that built the method (scoring the answer key), the distill dataset's own-click
frames, the audit's exclusion of build frames — was written against ``example_set`` alone. On a
per-class run each of them reads ``None`` and answers "no", without failing: the set is
deletable while a run lists it, and *Tune tracking* scores a method on the clicks it learned
from.

So the reads go through ``keypoint_runs.detect_example_sets``, and the scan below fails on a new
direct read of ``example_set`` anywhere in the package. The sites it allows read something that
is not a detect run's record (a patch teacher's config, the resolver writing its own config) and
say so.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from detail.storage import example_sets
from detail.storage import keypoint_runs

PACKAGE = Path(__file__).resolve().parents[2] / "detail"
DIRECT_READ = re.compile(
    r"""\.get\(\s*["']example_set["']\s*\)|\[\s*["']example_set["']\s*\]"""
)

#: (path, count of direct reads, why each is not a read of a detect run's sets)
ALLOWED = {
    "storage/keypoint_runs.py": (1, "detect_example_sets itself"),
    "pipeline/keypoint_detect.py": (
        5,
        "resolve_method_model writing the run's own config, and build_job carrying it",
    ),
    "pipeline/keypoint_preview.py": (
        1,
        "handed to build_method beside example_sets_by_class, which takes both forms",
    ),
    "pipeline/keypoint_dataset.py": (
        4,
        "the patch-method teacher's own config (one set until §26 phase 2), not a run",
    ),
    "api/keypoint_distill.py": (
        3,
        "the patch-method teacher's resolved config and the model record it writes",
    ),
    "gui/keypoint_distill.py": (
        1,
        "teacher_info's result, which already went through detect_example_sets",
    ),
}


def test_one_set_per_class_and_none_read_the_same_way():
    single = {"example_set": {"id": "ex_01", "digest": "a"}}
    per_class = {
        "example_set": None,
        "example_sets": {"Resident": {"id": "ex_02"}, "Intruder": {"id": "ex_03"}},
    }
    assert keypoint_runs.detect_example_sets(single) == {
        None: {"id": "ex_01", "digest": "a"}
    }
    assert keypoint_runs.detect_example_sets(per_class) == {
        "Resident": {"id": "ex_02"},
        "Intruder": {"id": "ex_03"},
    }
    assert keypoint_runs.detect_example_sets({"example_set": None}) == {}
    assert keypoint_runs.detect_example_sets({}) == {}  # a record from before the field


@pytest.mark.parametrize("cls", ["Resident", "Intruder"])
def test_a_set_used_by_either_class_cannot_be_deleted(recording_dir, cls):
    experiment_dir = recording_dir.parent.parent
    run_dir = keypoint_runs.detect_run_dir(recording_dir, "kd_01")
    run_dir.mkdir(parents=True)
    used = {"Resident": "ex_02", "Intruder": "ex_03"}
    keypoint_runs.write_detect_yaml(
        run_dir,
        {
            "schema_version": 1,
            "id": "kd_01",
            "example_set": None,
            "example_sets": {c: {"id": s, "marks": []} for c, s in used.items()},
        },
    )
    assert example_sets.runs_using(experiment_dir, used[cls]) == [
        (recording_dir.name, "kd_01")
    ]
    assert example_sets.runs_using(experiment_dir, "ex_01") == []


def _direct_reads() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        n = len(DIRECT_READ.findall(path.read_text(encoding="utf-8-sig")))
        if n:
            found[path.relative_to(PACKAGE).as_posix()] = n
    return found


def test_no_new_direct_read_of_a_runs_example_set():
    found = _direct_reads()
    new = {p: n for p, n in found.items() if n > ALLOWED.get(p, (0, ""))[0]}
    assert not new, (
        f"direct reads of 'example_set' {new}: read a detect run's sets through "
        "keypoint_runs.detect_example_sets (a per-class run's are under example_sets), or "
        "add the site to ALLOWED with why it is not a detect run's record"
    )
    stale = {p: n for p, (n, _why) in ALLOWED.items() if found.get(p, 0) < n}
    assert not stale, f"ALLOWED excuses reads that are gone; lower or drop: {stale}"


def test_the_scan_sees_a_direct_read():
    """The counterexample: the line every reader had before ``detect_example_sets``."""
    assert DIRECT_READ.findall('ref = data.get("example_set") or {}')
    assert DIRECT_READ.findall("snap = meta['example_set']")
    assert not DIRECT_READ.findall("keypoint_runs.detect_example_sets(meta)")
