"""Location plots' pure layer (DESIGN_PLAN §16 "Location plots"): paths, gaps, occupancy, speed
bins, units and groups, built from in-memory traces."""

import numpy as np
import pytest

from detail.analysis import location_plots as lp
from detail.configs.location_plots import LocationPlotView


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
    subject=None,
    object_id=0,
):
    t_s = np.arange(n) / fps if t is None else np.asarray(t, dtype=float)
    return lp.Trace(
        recording_id=sid,
        name=sid,
        object_id=object_id,
        subject=subject or f"subject {object_id}",
        show_subject=subject is not None,
        subject_labelled=subject is not None,
        frame=lp.FrameInfo(float(size[0]), float(size[1]), ppm),
        t_s=t_s,
        x_norm=np.linspace(0.1, 0.9, len(t_s)) if x is None else np.asarray(x, float),
        y_norm=np.full(len(t_s), 0.5) if y is None else np.asarray(y, float),
    )


def test_units_convert_from_normalized_with_the_manifest_frame():
    tr = _trace(n=2, ppm=2.0, x=[0.0, 0.5], y=[0.0, 1.0])
    assert tr.xy("px")[0].tolist() == [0.0, 100.0]
    assert tr.xy("px")[1].tolist() == [0.0, 100.0]
    assert tr.xy("mm")[0].tolist() == [0.0, 50.0]
    assert tr.xy("norm")[1].tolist() == [0.0, 1.0]


def test_a_gap_breaks_the_drawn_line_and_carries_no_speed():
    t = np.r_[np.arange(10), np.arange(50, 60)] / 10.0  # a 4 s dropout at sample 10
    tr = _trace(t=t)
    breaks = tr.breaks()
    assert breaks.tolist().index(True) == 10 and breaks.sum() == 1
    dist, dt = tr.steps("px")
    assert np.isnan(dist[9]) and np.isnan(dt[9])
    (x,) = lp.drawn(tr, [tr.x_norm], max_points=1000)
    assert len(x) == 21 and np.isnan(x[10])


def test_drawn_thins_to_the_cap_and_keeps_both_ends():
    tr = _trace(n=10_001)
    (x,) = lp.drawn(tr, [tr.x_norm], max_points=500)
    assert len(x) <= 500
    assert x[0] == tr.x_norm[0] and x[-1] == tr.x_norm[-1]


def test_occupancy_sums_to_one_and_counts_the_far_edge():
    edges = np.linspace(0, 1, 3)
    grid = lp.occupancy(
        np.array([0.1, 0.9, 1.0]), np.array([0.1, 0.1, 1.0]), edges, edges
    )
    assert grid.sum() == pytest.approx(1.0)
    assert grid[1, 1] == pytest.approx(1 / 3)  # (1.0, 1.0) lands in the last cell


def test_cells_are_square_along_the_longer_side():
    x_edges, y_edges = lp.cell_edges((200.0, 100.0), bins=40)
    assert len(x_edges) == 41 and len(y_edges) == 21
    assert np.diff(x_edges)[0] == pytest.approx(np.diff(y_edges)[0])


def test_binned_speed_is_distance_over_tracked_time_and_nan_where_untracked():
    t = np.r_[np.arange(0, 10), np.arange(30, 40)] / 10.0
    tr = _trace(t=t, x=np.linspace(0, 1, 20), size=(100, 100))
    centres, speed = lp.binned_speed(tr, "px", bin_s=1.0, t_end=4.0)
    assert len(centres) == 4
    assert np.isfinite(speed[0]) and np.isnan(speed[1]) and np.isnan(speed[2])
    step = 100.0 / 19  # px per sample
    assert speed[0] == pytest.approx(step / 0.1)


def test_mean_sem_ignores_nan_and_has_no_sem_for_one():
    mean, sem, n = lp.mean_sem([np.array([1.0, np.nan]), np.array([3.0, 4.0])])
    assert mean.tolist() == [2.0, 4.0]
    assert n.tolist() == [2, 1]
    assert sem[0] == pytest.approx(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
    assert np.isnan(sem[1])


def test_units_default_to_mm_only_when_every_recording_is_calibrated():
    view = LocationPlotView()
    assert lp.resolve_units([_trace(ppm=2.0), _trace(ppm=3.0)], view) == "mm"
    assert lp.resolve_units([_trace(ppm=2.0), _trace()], view) == "px"
    with pytest.raises(ValueError, match="not calibrated: r1"):
        lp.resolve_units([_trace()], LocationPlotView(units="mm"))


def test_pooled_positions_are_normalized_but_a_group_speed_is_not():
    traces = [_trace(ppm=2.0)]
    assert (
        lp.resolve_units(traces, LocationPlotView(group_by="g", kind="heatmap"))
        == "norm"
    )
    assert (
        lp.resolve_units(traces, LocationPlotView(group_by="g", kind="speed")) == "mm"
    )


def test_a_blank_group_value_is_its_own_group_and_sorts_last():
    traces = [_trace("a"), _trace("b"), _trace("c")]
    groups = lp.group_traces(
        traces, {"a": "wt", "b": " ", "c": "Mut"}, LocationPlotView(group_by="g")
    )
    assert [g for g, _ in groups] == ["Mut", "wt", lp.NO_VALUE]


def test_group_heat_map_counts_each_animal_equally():
    """A recording tracked ten times longer must not outweigh its group-mate (decision 5)."""
    long = _trace("long", n=1000, x=np.full(1000, 0.1))
    short = _trace("short", n=100, x=np.full(100, 0.9))
    view = LocationPlotView(kind="heatmap", group_by="g", bins=10)
    spec = lp.build_plot([long, short], view, {"long": "x", "short": "x"})
    grid = spec.panels[0].heat.grid
    assert grid.sum() == pytest.approx(1.0)
    left, right = grid[:, :5].sum(), grid[:, 5:].sum()
    assert left == pytest.approx(0.5) and right == pytest.approx(0.5)


def test_group_heat_cells_follow_a_shared_aspect_and_fall_back_to_square_grid():
    view = LocationPlotView(kind="heatmap", group_by="g", bins=20)
    same = lp.build_plot([_trace("a"), _trace("b")], view, {"a": "x", "b": "x"})
    heat = same.panels[0].heat
    assert (len(heat.x_edges), len(heat.y_edges)) == (21, 11)
    assert heat.y_edges[-1] == pytest.approx(1.0)
    mixed = lp.build_plot(
        [_trace("a"), _trace("b", size=(100, 100))], view, {"a": "x", "b": "x"}
    )
    heat = mixed.panels[0].heat
    assert (len(heat.x_edges), len(heat.y_edges)) == (21, 21)


@pytest.mark.parametrize("kind", ["trace", "heatmap", "speed", "trace3d"])
@pytest.mark.parametrize("group_by", [None, "g"])
def test_every_kind_builds_per_recording_and_by_group(kind, group_by):
    traces = [_trace("a", ppm=2.0), _trace("b", ppm=2.0), _trace("c", ppm=2.0)]
    values = {"a": "wt", "b": "wt", "c": "mut"}
    spec = lp.build_plot(traces, LocationPlotView(kind=kind, group_by=group_by), values)
    if group_by is None:
        assert len(spec.panels) == 3
    elif kind == "speed":
        assert len(spec.panels) == 1 and len(spec.panels[0].lines) == 2
        assert spec.legend
    else:
        assert [p.title for p in spec.panels] == ["mut (n=1)", "wt (n=2)"]
    assert spec.three_d == (kind == "trace3d")
    if kind == "trace3d":
        assert spec.camera is not None and spec.z.range[1] == pytest.approx(9.9)


def test_speed_colour_scale_tops_out_at_the_percentile():
    tr = _trace(n=101, x=np.r_[np.linspace(0, 0.5, 100), 1.0], size=(100, 100))
    spec = lp.build_plot([tr], LocationPlotView(color_by="speed", speed_clip_pct=50))
    speeds = tr.speed("px")
    assert spec.color.vmax == pytest.approx(np.percentile(speeds, 50))
    assert spec.color.vmax < speeds.max()


def test_group_by_needs_values():
    with pytest.raises(ValueError, match="no group values"):
        lp.build_plot([_trace()], LocationPlotView(group_by="g"))


def test_nothing_to_plot_raises_and_says_what_was_left_out():
    with pytest.raises(ValueError, match="no animals selected.*r1: no tracked rows"):
        lp.build_plot([], LocationPlotView(), notes=["r1: no tracked rows"])


# --------------------------------------------------------------------------- #
# Several animals in one recording
# --------------------------------------------------------------------------- #
def _table(rows_by_id: dict[int, int]) -> lp.TrackTable:
    """A tracks table with ``rows`` samples for each object id, sharing one clock."""
    columns = {}
    for oid, n in rows_by_id.items():
        t_ns = (np.arange(n) * 100_000_000 + oid * 50_000_000).astype(np.int64)
        cols = {"frame_idx": np.arange(n), "t_ns": t_ns}
        for name in ("mask_cx_norm", "mask_cy_norm", "bbox_cx_norm", "bbox_cy_norm"):
            cols[name] = np.linspace(0.2, 0.8, n) if n else np.zeros(0)
        columns[oid] = cols
    return lp.TrackTable(columns)


_FRAME = lp.FrameInfo(200.0, 100.0, None)


def test_every_animal_is_plotted_by_default_and_named_by_its_label():
    traces, notes = lp.recording_traces(
        _table({0: 50, 1: 30}),
        _FRAME,
        {0: "resident", 1: "intruder"},
        LocationPlotView(),
        recording_id="r1",
        name="r1",
    )
    assert [t.subject for t in traces] == ["intruder", "resident"]  # by label
    assert notes == []


def test_both_animals_share_the_recording_s_clock():
    traces, _ = lp.recording_traces(
        _table({0: 5, 1: 5}),
        _FRAME,
        None,
        LocationPlotView(),
        recording_id="r",
        name="r",
    )
    assert traces[0].t_s[0] == 0.0 and traces[1].t_s[0] == pytest.approx(0.05)


def test_a_label_picks_its_animal_and_an_unlabelled_run_falls_back_to_track_numbers():
    view = LocationPlotView(subjects=["intruder"])
    traces, _ = lp.recording_traces(
        _table({0: 5, 1: 5}),
        _FRAME,
        {0: "resident", 1: "intruder"},
        view,
        recording_id="r",
        name="r",
    )
    assert [(t.subject, t.object_id) for t in traces] == [("intruder", 1)]
    traces, _ = lp.recording_traces(
        _table({0: 5, 1: 5}),
        _FRAME,
        None,
        LocationPlotView(subjects=["subject 1"]),
        recording_id="r",
        name="r",
    )
    assert [t.object_id for t in traces] == [1]


def test_a_labelled_animal_with_no_rows_is_named_not_dropped_silently():
    traces, notes = lp.recording_traces(
        _table({0: 5}),
        _FRAME,
        {0: "resident", 1: "intruder"},
        LocationPlotView(),
        recording_id="r",
        name="box 3",
    )
    assert [t.subject for t in traces] == ["resident"]
    assert notes == ["box 3 · intruder: no tracked rows"]


def test_a_recording_without_the_chosen_animal_is_named():
    traces, notes = lp.recording_traces(
        _table({0: 5}),
        _FRAME,
        {0: "resident"},
        LocationPlotView(subjects=["intruder"]),
        recording_id="r",
        name="box 3",
    )
    assert traces == [] and notes == ["box 3: none of intruder"]


def _pair(sid, ids=(0, 1), labels=("resident", "intruder")):
    return [
        _trace(sid, subject=labels[0], object_id=ids[0], ppm=2.0),
        _trace(sid, subject=labels[1], object_id=ids[1], ppm=2.0),
    ]


def test_a_path_panel_holds_a_recording_s_animals_coloured_by_animal():
    spec = lp.build_plot(_pair("a") + _pair("b"), LocationPlotView(kind="trace"))
    assert [p.title for p in spec.panels] == ["a", "b"]
    assert [[ln.label for ln in p.lines] for p in spec.panels] == [
        ["resident", "intruder"],
        ["resident", "intruder"],
    ]
    assert spec.color is None and spec.legend
    identities = {ln.label: ln.identity for p in spec.panels for ln in p.lines}
    assert identities == {"intruder": 0, "resident": 1}  # sorted by label


def test_auto_colour_is_time_when_every_panel_holds_one_animal():
    spec = lp.build_plot([_trace("a"), _trace("b")], LocationPlotView(kind="trace"))
    assert spec.color.label == "elapsed time (s)" and not spec.legend


def test_heat_and_speed_give_each_animal_its_own_panel():
    for kind in ("heatmap", "speed"):
        spec = lp.build_plot(_pair("a"), LocationPlotView(kind=kind))
        assert [p.title for p in spec.panels] == ["a · resident", "a · intruder"]


def test_the_colour_order_comes_from_the_run_not_the_plotted_animals():
    order = lp.subject_order(["intruder", "resident"])
    spec = lp.build_plot(
        [_trace("a", subject="resident", ppm=2.0)],
        LocationPlotView(kind="trace", color_by="subject"),
        order=order,
    )
    assert spec.panels[0].lines[0].identity == 1  # still resident's colour


def test_group_by_animal_alone_and_crossed_with_a_column():
    traces = _pair("a") + _pair("b") + _pair("c")
    values = {"a": "wt", "b": "wt", "c": "mut"}
    alone = lp.build_plot(
        traces, LocationPlotView(kind="heatmap", split_by_subject=True)
    )
    assert [p.title for p in alone.panels] == ["intruder (n=3)", "resident (n=3)"]
    assert alone.title == "Occupancy by animal"
    crossed = lp.build_plot(
        traces,
        LocationPlotView(kind="heatmap", group_by="genotype", split_by_subject=True),
        values,
    )
    assert [p.title for p in crossed.panels] == [
        "mut · intruder (n=1)",
        "mut · resident (n=1)",
        "wt · intruder (n=2)",
        "wt · resident (n=2)",
    ]
    assert crossed.title == "Occupancy by genotype × animal"


def test_a_group_heat_map_counts_animals_not_recordings():
    """Two animals of one recording are two units: each sums to 1 before the mean."""
    resident = _trace("a", subject="resident", x=np.full(100, 0.1))
    intruder = _trace("a", subject="intruder", object_id=1, x=np.full(10, 0.9), n=10)
    view = LocationPlotView(kind="heatmap", group_by="g", bins=10)
    spec = lp.build_plot([resident, intruder], view, {"a": "x"})
    (panel,) = spec.panels
    assert panel.title == "x (n=2)"
    grid = panel.heat.grid
    assert grid[:, :5].sum() == pytest.approx(0.5) and grid[
        :, 5:
    ].sum() == pytest.approx(0.5)


def test_a_by_animal_speed_line_takes_the_animal_s_colour():
    traces = _pair("a") + _pair("b")
    spec = lp.build_plot(traces, LocationPlotView(kind="speed", split_by_subject=True))
    lines = {ln.label: ln.identity for ln in spec.panels[0].lines}
    assert lines == {"intruder (n=2)": 0, "resident (n=2)": 1}
    assert spec.panels[0].note == "mean ± SEM across animals"


# --------------------------------------------------------------------------- #
# Distance between animals: the pure helpers
# --------------------------------------------------------------------------- #
def test_a_pair_is_named_by_its_labels_sorted():
    assert lp.pair_label("Resident", "Intruder") == "Intruder – Resident"
    assert lp.pair_label("subject 10", "subject 2") == "subject 10 – subject 2"


def test_binned_mean_leaves_an_empty_bin_as_a_gap():
    t = np.array([0.1, 0.2, 2.5])
    centres, mean = lp.binned_mean(t, np.array([1.0, 3.0, 5.0]), 1.0, 3.0)
    assert centres.tolist() == [0.5, 1.5, 2.5]
    assert mean[0] == 2.0 and np.isnan(mean[1]) and mean[2] == 5.0


def test_share_histogram_sums_to_one_and_ignores_unmeasured_frames():
    share = lp.share_histogram(
        np.array([0.5, 1.5, 1.6, np.nan]), np.array([0.0, 1.0, 2.0])
    )
    assert share.tolist() == pytest.approx([1 / 3, 2 / 3])
