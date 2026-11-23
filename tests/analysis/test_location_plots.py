"""Location plots' pure layer (DESIGN_PLAN §16 "Location plots"): paths, breaks, occupancy, speed,
the body, units and groups, built from in-memory traces and hand-written ``tracks.csv`` files.

**Two rules this file guards.**

*One path, measured one way.* A speed plot and a comparison's speed measure the same smoothed
path, broken at the same gaps and jumps (``location_measures.smoothed_segments``). The plots
first came over with a speed of their own — the raw centroid step, broken at five sampling
intervals — so the Plots view and a comparison of the same run would have reported different
speeds, and a tracking jump was drawn as a line across the arena and counted as movement.
``test_speed_is_the_speed_a_comparison_measures`` holds the steps equal.

*Each animal counts once in a group.* A group's heat map normalizes each animal's occupancy to
1 before the mean; pooling samples let an animal tracked longer outweigh its group-mates (0.909
against the expected 0.5 in ``test_a_group_heat_map_counts_each_animal_equally``).
"""

import csv

import numpy as np
import pytest

from detail.analysis import location_measures as lm
from detail.analysis import location_plots as lp
from detail.configs.location_plots import LocationPlotView

VIEW = LocationPlotView()


def _body(n, *, length=40.0, width=10.0, angle=0.0, cx=None, cy=None):
    cx = np.full(n, 0.5) if cx is None else np.asarray(cx, float)
    cy = np.full(n, 0.5) if cy is None else np.asarray(cy, float)
    return lp.Body(
        cx_norm=cx,
        cy_norm=cy,
        length_px=np.full(n, float(length)),
        width_px=np.full(n, float(width)),
        angle_deg=np.full(n, float(angle)),
        area_px=np.full(n, float(length) * float(width) * 0.8),
        mask_area_px=np.full(n, float(length) * float(width)),
    )


def _trace(
    sid="r1",
    n=100,
    *,
    fps=10.0,
    ppm=None,
    size=(200, 100),
    x=None,
    y=None,
    t=None,
    body=True,
    length=40.0,
):
    t_s = np.arange(n) / fps if t is None else np.asarray(t, dtype=float)
    return lp.Trace(
        recording_id=sid,
        name=sid,
        frame=lp.FrameInfo(float(size[0]), float(size[1]), ppm),
        t_s=t_s,
        x_norm=np.linspace(0.1, 0.9, len(t_s)) if x is None else np.asarray(x, float),
        y_norm=np.full(len(t_s), 0.5) if y is None else np.asarray(y, float),
        body=_body(len(t_s), length=length) if body else None,
    )


# --------------------------------------------------------------------------- #
# The path
# --------------------------------------------------------------------------- #
def test_units_convert_from_normalized_with_the_manifest_frame():
    tr = _trace(n=2, ppm=2.0, x=[0.0, 0.5], y=[0.0, 1.0])
    assert tr.xy("px")[0].tolist() == [0.0, 100.0]
    assert tr.xy("px")[1].tolist() == [0.0, 100.0]
    assert tr.xy("mm")[0].tolist() == [0.0, 50.0]
    assert tr.xy("norm")[1].tolist() == [0.0, 1.0]


def test_speed_is_the_speed_a_comparison_measures():
    """The plot's steps are the comparison's, step for step, at any unit."""
    rng = np.random.default_rng(3)
    n, ppm = 600, 2.0
    x_mm = np.cumsum(rng.normal(2.0, 1.0, n)) % 90 + 5
    y_mm = 40 + 10 * np.sin(np.arange(n) / 20)
    t_s = np.arange(n) / 5.0
    track = lm.Track(
        t_ns=np.rint(t_s * 1e9).astype(np.int64),
        x_mm=x_mm,
        y_mm=y_mm,
        length_mm=np.full(n, 18.0),
        width_mm=np.full(n, 7.0),
        area_mm2=np.full(n, 100.0),
    )
    size = (100 * ppm, 100 * ppm)
    tr = _trace(
        n=n,
        t=t_s,
        ppm=ppm,
        size=size,
        x=x_mm * ppm / size[0],
        y=y_mm * ppm / size[1],
        length=18.0 * ppm,
    )
    settings = lm.Settings(
        sample_fps=None, smooth_s=1.0, moving_cm_s=2.0, jump_speed=7.5
    )
    path = lm.smoothed_path(track, settings)
    t, dist, dt = tr.steps(LocationPlotView(smooth_s=1.0, jump_speed=7.5), "mm")
    np.testing.assert_allclose(t * 1e9, path.t_ns, rtol=0, atol=1)
    np.testing.assert_allclose(dist, path.step_mm, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(dist / dt / 10.0, path.speed_cm_s, rtol=1e-9)


def test_a_jump_breaks_the_path_draws_no_line_and_adds_no_distance():
    n = 200
    x = np.linspace(0.2, 0.4, n)
    x[100] = 0.95  # one sample on the far side of the arena: a tracking jump
    tr = _trace(n=n, x=x, size=(1000, 1000), length=40.0)
    breaks = tr.breaks(VIEW)
    assert breaks[100] and breaks[101] and breaks.sum() == 2
    (drawn_x,) = lp.drawn(tr, [tr.x_norm], VIEW)
    assert np.isnan(drawn_x).sum() == 2  # the jump sample stands alone, never joined
    _t, dist, _dt = tr.steps(VIEW, "px")
    clean = _trace(n=n, x=np.linspace(0.2, 0.4, n), size=(1000, 1000), length=40.0)
    _t, clean_dist, _dt = clean.steps(VIEW, "px")
    # Walking 0.2 → 0.4 of a 1000 px frame is 200 px, less what smoothing trims off each end
    # of a stretch; the jump, joined, would have added about 1100.
    assert 150.0 < dist.sum() <= clean_dist.sum()


def test_a_gap_in_tracking_breaks_the_path():
    t = np.r_[np.arange(30), np.arange(80, 110)] / 10.0  # a 5 s dropout at sample 30
    tr = _trace(t=t)
    breaks = tr.breaks(VIEW)
    assert breaks.tolist().index(True) == 30 and breaks.sum() == 1
    speed = tr.speed(VIEW, "px")
    assert np.isfinite(speed).all()  # each side is long enough to have a smoothed path


def test_a_stretch_too_short_to_smooth_has_no_speed():
    t = np.r_[np.arange(30), [10.0, 10.1], np.arange(200, 230) / 10.0]
    tr = _trace(t=np.sort(t / np.r_[np.full(30, 10.0), 1.0, 1.0, np.ones(30)]))
    speed = tr.speed(VIEW, "px")
    assert np.isnan(speed[30:32]).all() and np.isfinite(speed[:30]).all()


def test_drawn_thins_to_the_cap_and_keeps_both_ends():
    tr = _trace(n=10_001)
    (x,) = lp.drawn(tr, [tr.x_norm], LocationPlotView(max_points=500))
    assert len(x) <= 500 and x[0] == tr.x_norm[0] and x[-1] == tr.x_norm[-1]


def test_binned_speed_is_distance_over_time_and_nan_where_untracked():
    t = np.r_[np.arange(0, 100), np.arange(300, 400)] / 10.0
    x = np.r_[np.linspace(0.0, 0.5, 100), np.linspace(0.5, 1.0, 100)]
    tr = _trace(t=t, x=x, size=(1000, 1000))
    centres, speed = lp.binned_speed(tr, VIEW, "px", t_end=40.0)
    assert len(centres) == 40
    assert np.isnan(speed[15])  # the dropout reads as missing, not as standing still
    assert speed[5] == pytest.approx(500 / 9.9, rel=0.05)


# --------------------------------------------------------------------------- #
# Occupancy, units and groups
# --------------------------------------------------------------------------- #
def test_occupancy_sums_to_one_and_counts_the_far_edge():
    edges = np.array([0.0, 0.5, 1.0])
    grid = lp.occupancy(
        np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.2]), edges, edges
    )
    assert grid.sum() == pytest.approx(1.0) and grid[1, 1] == pytest.approx(1 / 3)


def test_cells_are_square_along_the_longer_side():
    x_edges, y_edges = lp.cell_edges((200.0, 100.0), 10)
    assert len(x_edges) == 11 and len(y_edges) == 6


def test_units_default_to_mm_only_when_every_recording_is_calibrated():
    calibrated, bare = _trace("a", ppm=2.0), _trace("b")
    assert lp.resolve_units([calibrated], VIEW) == "mm"
    assert lp.resolve_units([calibrated, bare], VIEW) == "px"
    with pytest.raises(ValueError, match="not calibrated: b"):
        lp.resolve_units([calibrated, bare], LocationPlotView(units="mm"))


def test_pooled_positions_are_normalized_but_a_group_speed_is_not():
    traces = [_trace("a", ppm=2.0)]
    assert lp.resolve_units(traces, LocationPlotView(group_by="g")) == "norm"
    assert (
        lp.resolve_units(traces, LocationPlotView(kind="speed", group_by="g")) == "mm"
    )


def test_a_blank_group_value_is_its_own_group_and_sorts_last():
    groups = lp.group_traces(
        [_trace("a"), _trace("b"), _trace("c")], {"a": "wt", "b": " "}
    )
    assert [g for g, _ in groups] == ["wt", lp.NO_VALUE]
    assert [t.recording_id for t in groups[1][1]] == ["b", "c"]


def test_a_group_heat_map_counts_each_animal_equally():
    """An animal tracked ten times longer must not outweigh its group-mate."""
    long = _trace("a", n=1000, x=np.full(1000, 0.1), y=np.full(1000, 0.1))
    short = _trace("b", n=100, x=np.full(100, 0.9), y=np.full(100, 0.9))
    spec = lp.build_plot(
        [long, short], LocationPlotView(kind="heatmap", group_by="g", bins=5), {}
    )
    grid = spec.panels[0].heat.grid
    assert grid[0, 0] == pytest.approx(0.5) and grid[-1, -1] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kind", ["trace", "heatmap", "speed", "trace3d", "body", "body_hist"]
)
@pytest.mark.parametrize("group_by", [None, "g"])
def test_every_kind_builds_per_recording_and_by_group(kind, group_by):
    traces = [_trace("a", ppm=2.0), _trace("b", ppm=2.0), _trace("c", ppm=2.0)]
    values = {"a": "wt", "b": "wt", "c": "ko"} if group_by else None
    spec = lp.build_plot(traces, LocationPlotView(kind=kind, group_by=group_by), values)
    grouped_panels = 1 if kind in ("speed", "body", "body_hist") else 2
    assert len(spec.panels) == (grouped_panels if group_by else 3)
    assert spec.legend == bool(group_by and grouped_panels == 1)


def test_group_by_needs_values():
    with pytest.raises(ValueError, match="no group values"):
        lp.build_plot([_trace()], LocationPlotView(group_by="g"))


def test_the_speed_colour_scale_tops_out_at_the_percentile():
    tr = _trace(n=400, x=np.r_[np.linspace(0.1, 0.5, 399), 0.5], size=(1000, 1000))
    spec = lp.build_plot([tr], LocationPlotView(color_by="speed", speed_clip_pct=50))
    speeds = tr.speed(LocationPlotView(color_by="speed"), "px")
    assert spec.color.vmax == pytest.approx(
        np.percentile(speeds[np.isfinite(speeds)], 50)
    )


# --------------------------------------------------------------------------- #
# One animal per recording, read from tracks.csv
# --------------------------------------------------------------------------- #
def _write_tracks(path, rows, *, body=True, reverse=False):
    columns = ["frame_idx", "t_ns", "object_id"] + list(lp._CENTRE_COLUMNS)
    if body:
        columns += list(lp.BODY_COLUMNS) + ["mask_area_px"]
    if reverse:
        columns = columns[::-1]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _row(frame, oid=0, *, x=0.5, y=0.5, bx=None):
    row = {"frame_idx": frame, "t_ns": frame * 100_000_000, "object_id": oid}
    for src in ("mask", "bbox"):
        row[f"{src}_cx_norm"], row[f"{src}_cy_norm"] = x, y
    row.update(
        body_cx_norm=x if bx is None else bx, body_cy_norm=y, body_length_px=40,
        body_width_px=10, body_angle_deg=0, body_area_px=320, mask_area_px=400,
    )  # fmt: skip
    return row


def test_tracks_are_read_by_column_name_in_any_order(tmp_path):
    path = tmp_path / "tracks.csv"
    _write_tracks(path, [_row(1, x=0.7), _row(0, x=0.6)], reverse=True)
    table = lp.read_tracks(path)
    assert table.has_body
    cols = table.columns[0]
    assert cols["frame_idx"].tolist() == [0, 1]  # in frame order
    assert cols["mask_cx_norm"].tolist() == [0.6, 0.7]


def test_a_blank_measure_is_nan_not_an_error(tmp_path):
    path = tmp_path / "tracks.csv"
    row = _row(0)
    row["body_length_px"] = ""
    _write_tracks(path, [row])
    assert np.isnan(lp.read_tracks(path).columns[0]["body_length_px"][0])


def test_the_body_is_the_mask_on_an_export_from_before_the_body(tmp_path):
    """Such an export never removed a tail, and with thin parts kept the two are equal."""
    path = tmp_path / "tracks.csv"
    _write_tracks(path, [_row(0, x=0.3), _row(1, x=0.4)], body=False)
    table = lp.read_tracks(path)
    assert not table.has_body
    tr, note = lp.trace(
        table, lp.FrameInfo(100, 100, None), VIEW, recording_id="r", name="r"
    )
    assert note is None and tr.body is None
    assert tr.x_norm.tolist() == [0.3, 0.4]


def test_the_body_source_reads_the_body_s_centre(tmp_path):
    path = tmp_path / "tracks.csv"
    _write_tracks(path, [_row(0, x=0.3, bx=0.31), _row(1, x=0.4, bx=0.41)])
    table = lp.read_tracks(path)
    frame = lp.FrameInfo(100, 100, None)
    body, _ = lp.trace(table, frame, VIEW, recording_id="r", name="r")
    mask, _ = lp.trace(
        table, frame, LocationPlotView(source="mask"), recording_id="r", name="r"
    )
    assert body.x_norm.tolist() == [0.31, 0.41] and mask.x_norm.tolist() == [0.3, 0.4]


def test_a_recording_with_several_animals_is_named_not_guessed_at(tmp_path):
    path = tmp_path / "tracks.csv"
    _write_tracks(path, [_row(0, 0), _row(0, 3), _row(1, 0)])
    tr, note = lp.trace(
        lp.read_tracks(path),
        lp.FrameInfo(100, 100, None),
        VIEW,
        recording_id="r",
        name="box 3",
    )
    assert tr is None
    assert (
        note
        == "box 3: 2 animals (tracks 0, 3); location plots show one animal per recording"
    )


def test_a_recording_with_no_rows_is_named(tmp_path):
    path = tmp_path / "tracks.csv"
    _write_tracks(path, [])
    tr, note = lp.trace(
        lp.read_tracks(path),
        lp.FrameInfo(100, 100, None),
        VIEW,
        recording_id="r",
        name="box 3",
    )
    assert tr is None and note == "box 3: no tracked rows"


def test_nothing_to_plot_says_what_was_left_out():
    with pytest.raises(ValueError, match="box 3: no tracked rows"):
        lp.build_plot([], VIEW, notes=["box 3: no tracked rows"])


# --------------------------------------------------------------------------- #
# The body
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "measure, units, expected",
    [
        ("length", "px", 40.0),
        ("length", "mm", 20.0),
        ("width", "mm", 5.0),
        ("area", "mm", 80.0),  # 320 px² at 2 px/mm
        ("mask_area", "px", 400.0),
        ("elongation", "mm", 4.0),
    ],
)
def test_a_body_measure_is_read_in_its_units(measure, units, expected):
    tr = _trace(ppm=2.0)
    assert lp.body_values(tr, measure, units)[0] == pytest.approx(expected)


def test_a_body_in_fractions_of_the_frame_is_refused():
    with pytest.raises(ValueError, match="px or mm"):
        lp.build_plot([_trace()], LocationPlotView(kind="body", units="norm"))


def test_a_body_plot_names_the_recordings_exported_before_the_body():
    measured, old = _trace("new"), _trace("old", body=False)
    spec = lp.build_plot([measured, old], LocationPlotView(kind="body"))
    assert [p.title for p in spec.panels] == ["new"]
    assert "old: exported before the body was measured" in spec.note
    with pytest.raises(ValueError, match="old: exported before the body"):
        lp.build_plot([old], LocationPlotView(kind="body"))


def test_a_body_histogram_is_each_animal_s_share_of_samples():
    tr = _trace(n=200)
    tr.body.length_px[:50] = 20.0  # a quarter of the time at half the length
    spec = lp.build_plot(
        [tr], LocationPlotView(kind="body_hist", hist_bins=10, units="px")
    )
    line = spec.panels[0].lines[0]
    assert line.step and line.y.sum() == pytest.approx(1.0)
    assert line.y[0] == pytest.approx(0.25) and line.y[-1] == pytest.approx(0.75)
    assert spec.x.label == "body length (px)"


def test_a_body_over_time_is_the_mean_per_time_bin():
    tr = _trace(n=100, fps=10.0)
    tr.body.width_px[50:] = 20.0
    spec = lp.build_plot(
        [tr],
        LocationPlotView(kind="body", body_measure="width", units="px", time_bin_s=5),
    )
    assert spec.panels[0].lines[0].y.tolist() == pytest.approx([10.0, 20.0])
    assert spec.y.label == "body width (px)"


@pytest.mark.parametrize("angle, extent", [(0.0, (20.0, 5.0)), (90.0, (5.0, 20.0))])
def test_a_body_outline_is_the_body_s_ellipse_along_its_long_axis(angle, extent):
    """Half the length along the long axis and half the width across it, at the body's
    centre, drawn in pixels however the plot is scaled."""
    tr = _trace(n=101, fps=10.0, size=(200, 100))
    tr.body.angle_deg[:] = angle
    line = lp.posture_outlines(tr, LocationPlotView(posture_every_s=5.0), "px", None)
    outlines = np.split(line.x, np.flatnonzero(np.isnan(line.x)) + 1)
    outlines = [o[np.isfinite(o)] for o in outlines if np.isfinite(o).any()]
    assert len(outlines) == 3  # at 0 s, 5 s and 10 s
    first_x = outlines[0]
    first_y = line.y[: len(first_x)]
    assert np.ptp(first_x) / 2 == pytest.approx(extent[0], abs=0.2)
    assert np.ptp(first_y) / 2 == pytest.approx(extent[1], abs=0.2)
    centre = (first_x.max() + first_x.min()) / 2
    assert centre == pytest.approx(100.0, abs=0.2)  # the body's centre, 0.5 of 200 px


def test_a_body_outline_follows_the_plot_s_units_and_colour():
    tr = _trace(n=101, fps=10.0, size=(200, 100))
    line = lp.posture_outlines(
        tr, LocationPlotView(posture_every_s=5.0), "norm", tr.t_s
    )
    finite = np.isfinite(line.x)
    assert line.x[finite].max() == pytest.approx(0.6, abs=0.01)  # (100 + 20) / 200
    assert sorted(set(line.values[np.isfinite(line.values)])) == [0.0, 5.0, 10.0]


def test_a_path_plot_draws_outlines_only_where_the_body_was_measured():
    view = LocationPlotView(posture_every_s=5.0)
    spec = lp.build_plot([_trace("new"), _trace("old", body=False)], view)
    assert [len(p.lines) for p in spec.panels] == [2, 1]
    assert "no body outline: old" in spec.note
