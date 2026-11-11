"""Measures reachable on their own (§23), not only through a group comparison.

The numbers have been computable since #207 and askable by nobody: the only caller was §24's
Comparison, so getting the per-unit table meant running a group contrast and naming an
aggregation path you might not want. These tests assert the values a person reads, not that the
call returned -- a table of plausible numbers is exactly what this layer fails as.

The fixture is `test_comparison.py::_cohort`, reused rather than re-synthesized: it already
writes a §19 artifact *and* a frame view per recording, which is what a windowed denominator
needs. `tests/test_comparison_cli.py` already imports it cross-package, so this is the
established shape.
"""

import pytest

from detail.api.measures import MeasuresError
from detail.configs.windows import WindowSpec
from tests.api.test_comparison import N_FRAMES, PERIOD, UNBALANCED, _cohort


def _value(rows, recording, measure, label="rearing"):
    match = [
        row
        for row in rows
        if row["recording_id"] == recording
        and row["measure"] == measure
        and row["label"] == label
    ]
    assert len(match) == 1, f"expected one {measure} row for {recording}, got {match}"
    return match[0]["value"]


def test_the_table_counts_the_bouts_the_artifact_holds(tmp_path):
    """`UNBALANCED` gives r1 two bouts and r4 twenty; the count is the count."""
    exp = _cohort(tmp_path, UNBALANCED)
    rows = exp.measures("an_01").table(measures=["n_instances"])

    assert _value(rows, "r1", "n_instances") == 2
    assert _value(rows, "r4", "n_instances") == 20


def test_a_rate_is_per_second_of_observed_time(tmp_path):
    """The fixture observes all 100 frames at 100 Hz — one second — so r4's twenty bouts are
    twenty per second. A rate against *window length* rather than observed time would agree
    here only because coverage is complete; the denominator column is what distinguishes them.
    """
    exp = _cohort(tmp_path, UNBALANCED)
    rows = exp.measures("an_01").table(measures=["rate"])

    assert _value(rows, "r4", "rate") == pytest.approx(20.0)
    denominators = {row["denominator_ns"] for row in rows}
    assert denominators == {N_FRAMES * PERIOD}


def test_a_windowed_measure_splits_the_recording(tmp_path):
    """What Comparison could already do and nobody could ask for directly."""
    exp = _cohort(tmp_path, UNBALANCED)
    rows = exp.measures("an_01").table(
        windows=WindowSpec(kind="normalized", parts=2), measures=["n_instances"]
    )

    windows = sorted({row["window_id"] for row in rows})
    assert windows == ["part_00", "part_01"]
    halves = [
        row["value"]
        for row in rows
        if row["recording_id"] == "r4" and row["label"] == "rearing"
    ]
    assert sum(halves) == 20, "every bout falls in exactly one half"
    assert all(value > 0 for value in halves), "r4's bouts span both halves"


def test_the_export_writes_the_table_the_frames_and_its_manifest(tmp_path):
    import csv

    from detail.analysis import measures as measures_lib
    from detail.api import measures as api

    exp = _cohort(tmp_path, UNBALANCED)
    out = exp.measures("an_01").export(tmp_path / "out", measures=["n_instances"])

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [api.FRAMES_FILENAME, api.MEASURES_FILENAME, api.MANIFEST_FILENAME]
    )
    with open(out / api.MEASURES_FILENAME, encoding="utf-8-sig") as handle:
        assert next(csv.reader(handle)) == measures_lib.COLUMNS
    with open(out / api.FRAMES_FILENAME, encoding="utf-8-sig") as handle:
        frames = list(csv.DictReader(handle))
    assert len(frames) == N_FRAMES * len(UNBALANCED), "one row per frame per unit"
    assert {row["observed"] for row in frames} == {
        "1"
    }, "the fixture observes everything"
    assert any(
        row["label"] == "rearing" for row in frames
    ), "bouts reach the frame table"


def test_the_export_does_not_copy_the_instance_table(tmp_path):
    """The artifact already writes `instances.csv` beside the run. A second copy is two files
    that can disagree, and the export names where the first one lives instead."""
    from detail.api import measures as api

    exp = _cohort(tmp_path, UNBALANCED)
    out = exp.measures("an_01").export(tmp_path / "out")

    assert not (out / "instances.csv").exists()
    assert (exp.analysis_run_dir("an_01") / "instances.csv").is_file()


def test_a_run_with_no_artifact_says_so(tmp_path):
    """Not every analysis produces behavior in time, and the refusal names why rather than
    failing deep inside a reader."""
    exp = _cohort(tmp_path, UNBALANCED)
    exp.create_recording("r9")

    with pytest.raises(MeasuresError, match="no behavior output"):
        exp.measures("an_99").table()
