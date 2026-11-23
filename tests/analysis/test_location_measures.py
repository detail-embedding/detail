"""Measures over a location-export run: distance, time moving, speed and body shape (§24).

**Distance that is not movement (detail#70).** A raw centre path adds distance for an animal
standing still, and a path joined across a tracking gap or a mis-tracked sample adds a step
nobody took. The path is smoothed, and breaks at a gap and at a jump.

**A rule that means something else at another sampling rate.** The first version of the jump
rule was "a step longer than 1.5 body lengths": a fifth of a second of travel at five rows a
second and a sixtieth at sixty, so the same teleport was a jump in one export and movement in
another. It is a speed now, and the test below samples one path at two rates.

**A window that does not add up.** The path is smoothed once per track and each step belongs
to the window its time is in, so distance over adjacent windows sums to the distance over
their union.

**A measure a surface cannot print.** A measure with no unit renders as a bare float and is
given a count's pseudo-count of one bout, which would be added to a ratio of 2.4.
"""

import numpy as np
import pytest

from detail.analysis import location_measures as lm
from detail.analysis import measures
from detail.configs.windows import WindowSpec
from tests.location_fixtures import NS, track, write_run

SETTINGS = lm.Settings(sample_fps=None, smooth_s=1.0, moving_cm_s=2.0, jump_speed=7.5)
WHOLE = WindowSpec(kind="whole").windows_for


def whole(tr: lm.Track, settings: lm.Settings = SETTINGS) -> dict:
    values = lm.window_values(
        tr,
        lm.smoothed_path(tr, settings),
        int(tr.t_ns[0]),
        int(tr.t_ns[-1]) + 1,
        settings,
    )
    return {name: value for name, (value, _denominator, _n) in values.items()}


def test_a_steady_walk_is_measured_as_walked():
    got = whole(track(np.arange(200) * 10.0))  # 10 mm every 0.2 s: 5 cm/s for 40 s
    assert got["distance"] == pytest.approx(1.94, abs=0.02)
    assert got["time_moving"] == pytest.approx(1.0)
    assert got["speed_moving"] == pytest.approx(5.0, abs=0.01)
    assert got["body_length"] == 180.0 and got["body_width"] == 75.0
    assert got["body_elongation"] == pytest.approx(2.4)
    assert got["body_area"] == pytest.approx(
        180.0 * 75.0 * 0.8 / 100.0
    )  # cm², from mm²


def test_a_still_animal_with_jitter_travels_almost_nothing():
    rng = np.random.default_rng(0)
    still = track(100 + rng.normal(0, 1.0, 600), 100 + rng.normal(0, 1.0, 600))
    raw_m = np.hypot(np.diff(still.x_mm), np.diff(still.y_mm)).sum() / 1000.0
    got = whole(still)
    assert raw_m > 0.9  # the unsmoothed path "travels" about a metre
    assert got["distance"] < 0.35 * raw_m
    assert got["time_moving"] == 0.0
    assert got["speed_moving"] is None  # no value, rather than a speed of zero


def test_a_gap_and_a_jump_break_the_path_and_add_no_distance():
    x = np.r_[np.arange(100) * 10.0, 5000 + np.arange(100) * 10.0]  # a 4 m teleport
    t = np.r_[np.arange(100) * 0.2, 60 + np.arange(100) * 0.2]  # across a 40 s gap
    assert whole(track(x, t=t))["distance"] == pytest.approx(2 * 0.94, abs=0.03)
    # the same teleport with no gap in time: only the jump rule can catch it
    assert whole(track(x))["distance"] == pytest.approx(2 * 0.94, abs=0.03)


def test_the_jump_rule_means_the_same_at_any_sampling_rate():
    def walk_with_teleport(hz: float) -> lm.Track:
        t = np.arange(0, 40, 1 / hz)
        x = 50.0 * t  # 5 cm/s
        x[len(x) // 2 :] += 180.0  # one body length, between two samples
        return track(x, hz=hz)

    slow, fast = whole(walk_with_teleport(5.0)), whole(walk_with_teleport(25.0))
    # 180 mm in one sample is 5 body lengths a second at 5 Hz and 25 at 25 Hz. A rule in
    # body lengths *per step* called it movement at both; in body lengths per second it is
    # a jump only where the animal could not have made it.
    assert fast["distance"] == pytest.approx(2.0 - 0.05 * 2.0, abs=0.08)
    assert slow["distance"] == pytest.approx(2.0 + 0.18, abs=0.08)
    assert lm.smoothed_path(walk_with_teleport(25.0), SETTINGS) is not None


def test_distance_over_adjacent_windows_sums_to_the_whole():
    rng = np.random.default_rng(1)
    tr = track(np.cumsum(rng.uniform(0, 30, 900)), np.cumsum(rng.normal(0, 5, 900)))
    path = lm.smoothed_path(tr, SETTINGS)
    start, end = int(tr.t_ns[0]), int(tr.t_ns[-1]) + 1
    total = lm.window_values(tr, path, start, end, SETTINGS)["distance"][0]
    thirds = WindowSpec(kind="normalized", parts=3).windows_for(start, end)
    parts = [
        lm.window_values(tr, path, w.start_t_ns, w.end_t_ns, SETTINGS)["distance"][0]
        for w in thirds
    ]
    assert len(parts) == 3 and all(p > 0 for p in parts)
    assert sum(parts) == pytest.approx(total, rel=1e-12)


def test_a_window_with_nothing_in_it_has_no_values_not_zeros():
    tr = track(np.arange(50) * 10.0)
    values = lm.window_values(
        tr, lm.smoothed_path(tr, SETTINGS), 10**12, 2 * 10**12, SETTINGS
    )
    assert all(value is None for value, _denominator, _n in values.values())


def test_time_moving_carries_the_path_time_it_is_a_share_of():
    tr = track(np.arange(200) * 10.0)
    path = lm.smoothed_path(tr, SETTINGS)
    value, denominator, n = lm.window_values(
        tr, path, int(tr.t_ns[0]), int(tr.t_ns[-1]) + 1, SETTINGS
    )["time_moving"]
    assert value == 1.0 and n == len(path.step_mm)
    assert denominator == pytest.approx(n * 0.2 * NS)


def test_rows_are_thinned_by_the_exports_own_rule_and_never_invented():
    dense = track(np.arange(1000) * 2.0, hz=50.0)  # 20 s at 50 rows a second
    thinned = dense.thinned(5.0)
    assert len(thinned.t_ns) == 101  # ticks at 0, 0.2, ... 20.0 s
    # every tick keeps the row nearest it; the last tick, at 20 s, keeps the last row
    assert np.all(np.diff(thinned.t_ns)[:-1] == NS // 5)
    assert thinned.t_ns[-1] == dense.t_ns[-1]
    sparse = track(np.arange(40) * 10.0, hz=2.0)
    assert (
        len(sparse.thinned(5.0).t_ns) == 40
    )  # exported slower than asked: read as it is
    assert dense.thinned(None) is dense
    # and the measure is the same whichever rate the run was exported at
    assert whole(thinned)["distance"] == pytest.approx(
        whole(track(np.arange(101) * 20.0))["distance"], rel=0.01
    )


# --------------------------------------------------------------------------------------------- #
# reading a run
# --------------------------------------------------------------------------------------------- #
def test_rows_come_out_in_the_measures_schema_one_per_track_window_and_measure(
    tmp_path,
):
    two = {0: track(np.arange(100) * 10.0), 1: track(np.arange(100) * 20.0)}
    write_run(tmp_path, {"r1": two})
    rows = lm.location_rows(
        tmp_path, "an_01", ["r1"], settings=SETTINGS, windows_for=WHOLE
    )
    assert all(list(row) == measures.COLUMNS for row in rows)
    assert len(rows) == 2 * len(lm.MEASURES)
    by = {(row["subject"], row["measure"]): row for row in rows}
    assert by[("1", "distance")]["value"] == pytest.approx(
        2 * by[("0", "distance")]["value"]
    )
    assert {row["window_id"] for row in rows} == {"whole"}
    assert {row["label"] for row in rows} == {""}


def test_a_blank_cell_is_a_missing_sample_not_a_zero(tmp_path):
    tr = track(np.arange(60) * 10.0)
    tr.length_mm[10:20] = np.nan  # ten frames whose mask was empty
    write_run(tmp_path, {"r1": {0: tr}})
    read = lm.read_tracks(tmp_path / "r1")["0"]
    assert np.isnan(read.length_mm[10:20]).all() and read.length_mm[0] == 180.0
    rows = lm.location_rows(
        tmp_path, "an_01", ["r1"], settings=SETTINGS, windows_for=WHOLE,
        measures=["body_length"],
    )  # fmt: skip
    assert rows[0]["value"] == 180.0  # the median of the frames that had a body


def test_only_the_measures_asked_for_are_emitted_and_an_unknown_one_is_refused(
    tmp_path,
):
    write_run(tmp_path, {"r1": {0: track(np.arange(60) * 10.0)}})
    rows = lm.location_rows(
        tmp_path, "an_01", ["r1"], settings=SETTINGS, windows_for=WHOLE,
        measures=["distance", "body_width"],
    )  # fmt: skip
    assert [row["measure"] for row in rows] == ["distance", "body_width"]
    with pytest.raises(
        lm.LocationSourceError, match="no measure n_instances; it has distance"
    ):
        lm.location_rows(
            tmp_path, "an_01", ["r1"], settings=SETTINGS, windows_for=WHOLE,
            measures=["n_instances"],
        )  # fmt: skip


def test_a_run_from_before_the_body_was_measured_is_refused_by_name(tmp_path):
    write_run(tmp_path, {"r1": {0: track(np.arange(60) * 10.0)}}, body=False)
    with pytest.raises(
        lm.LocationSourceError, match="an_03 was made before the export measured"
    ):
        lm.location_rows(
            tmp_path, "an_03", ["r1"], settings=SETTINGS, windows_for=WHOLE
        )


def test_a_run_that_removed_thin_parts_is_refused_by_name(tmp_path):
    # the option was taken out: what it measured no other tool can reproduce
    write_run(
        tmp_path, {"r1": {0: track(np.arange(60) * 10.0)}}, thin_parts_removed=True
    )
    with pytest.raises(
        lm.LocationSourceError, match="an_04 measured r1's body with its thin parts"
    ):
        lm.location_rows(
            tmp_path, "an_04", ["r1"], settings=SETTINGS, windows_for=WHOLE
        )


def test_a_recording_exported_without_a_calibration_is_refused_by_name(tmp_path):
    write_run(tmp_path, {"r1": {0: track(np.arange(60) * 10.0)}}, pixels_per_mm=None)
    with pytest.raises(lm.LocationSourceError, match="r1 had no distance calibration"):
        lm.location_rows(
            tmp_path, "an_01", ["r1"], settings=SETTINGS, windows_for=WHOLE
        )


def test_a_recording_the_run_did_not_write_is_named(tmp_path):
    write_run(tmp_path, {"r1": {0: track(np.arange(60) * 10.0)}})
    with pytest.raises(lm.LocationSourceError, match="r9 has no locations.yaml"):
        lm.location_rows(
            tmp_path, "an_01", ["r1", "r9"], settings=SETTINGS, windows_for=WHOLE
        )


def test_the_fingerprint_moves_when_a_table_or_a_manifest_does(tmp_path):
    write_run(tmp_path, {"r1": {0: track(np.arange(60) * 10.0)}})
    before = lm.run_fingerprint(tmp_path, ["r1"])
    assert lm.run_fingerprint(tmp_path, ["r1"]) == before
    write_run(tmp_path, {"r1": {0: track(np.arange(61) * 10.0)}})  # one more row
    assert lm.run_fingerprint(tmp_path, ["r1"]) != before


# --------------------------------------------------------------------------------------------- #
# what a surface needs to print one
# --------------------------------------------------------------------------------------------- #
def test_every_location_measure_can_be_printed_and_none_shadows_a_behavior_measure():
    assert set(lm.MEASURES).isdisjoint(measures.MEASURES)
    for measure_id, spec in lm.MEASURES.items():
        assert spec.measure_id == measure_id
        assert spec.label.strip() and spec.description.strip()
        # "" is a plain count to the report: its pseudo-count is one bout
        assert spec.unit, f"{measure_id} has no unit"
        assert spec.fans_out is False
        assert measures.spec_of(measure_id) is spec
    assert measures.spec_of("rate") is measures.MEASURES["rate"]
    assert measures.spec_of("no_such_measure") is None


def test_the_report_reads_a_location_measures_unit_label_and_pseudo_count():
    from detail.analysis import comparison_report as report

    key = ("distance", "", "", "whole")
    assert report.measure_unit(key) == "m"
    assert report.measure_axis_label(key) == "Distance travelled (m)"
    assert report.measure_title(key, frozenset()) == "Distance travelled"
    assert (
        report.measure_axis_label(("time_moving", "", "", "whole")) == "Time moving (%)"
    )
    # a ratio has a unit, so it is not read as a count, and prints none
    ratio = ("body_elongation", "", "", "whole")
    assert report.measure_unit(ratio) == report.RATIO_UNIT
    assert report.measure_axis_label(ratio) == "Body length : width"
    assert report.shown_value(2.4321, report.RATIO_UNIT) == ("2.432", "")
    pseudo = report.pseudo_count(1.0)
    # one second means nothing in metres, millimetres or a ratio: nothing is added
    for measure_id in (
        "distance",
        "speed_moving",
        "body_length",
        "body_elongation",
        "body_area",
    ):
        assert pseudo((measure_id, "", "", "whole"), None) == 0.0
    # and one second of a share of time is one second over the path time
    assert pseudo(("time_moving", "", "", "whole"), 1000 * NS) == pytest.approx(0.001)


def test_location_exports_are_found_by_their_tables_not_by_a_run_record(tmp_path):
    """A picker lists them on the UI thread, so discovery reads no run record: the first
    version read every run's ``analysis.yaml`` to ask its module, and the UI-thread guard
    refused it."""
    from detail.storage import location_export

    analysis = tmp_path / "analysis"
    write_run(analysis / "an_02", {"r1": {0: track(np.arange(30) * 10.0)}})
    write_run(analysis / "an_10", {"r1": {0: track(np.arange(30) * 10.0)}})
    (analysis / "an_03" / "r1").mkdir(parents=True)  # a run of another kind: no tables
    (analysis / "an_03" / "behavior_output.yaml").write_text("x: 1\n", encoding="utf-8")
    assert location_export.discover_location_exports(tmp_path) == ["an_02", "an_10"]
    assert location_export.discover_location_exports(tmp_path / "nowhere") == []
    assert location_export.has_location_tables(analysis / "an_02")
    assert not location_export.has_location_tables(analysis / "an_03")
    manifest = location_export.read_location_manifest(analysis / "an_02" / "r1")
    assert (
        manifest["pixels_per_mm"] == 2.0
        and "body_length_mm" in manifest["columns"]["tracks"]
    )
    assert location_export.read_location_manifest(analysis / "an_03" / "r1") is None
