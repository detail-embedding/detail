"""The matplotlib renderer for ``report/plot_spec.PlotSpec``: every mark kind draws, colours come
from the theme, and a 3-D grid is one camera over shared ranges."""

import numpy as np
import pytest

from detail.core import chart_theme
from detail.report import plot_render
from detail.report.plot_spec import (
    Axis,
    Band,
    Camera,
    ColorScale,
    Heat,
    Line,
    Panel,
    PlotSpec,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _path_spec(n_panels=3, *, three_d=False):
    t = np.linspace(0, 10, 50)
    x = np.r_[np.linspace(0, 100, 25), np.nan, np.linspace(0, 100, 24)]
    panels = [
        Panel(
            f"r{i}",
            lines=[Line(x, x * 0.5, z=t if three_d else None, values=t, label=f"r{i}")],
        )
        for i in range(n_panels)
    ]
    return PlotSpec(
        kind="trace3d" if three_d else "trace",
        title="Path",
        panels=panels,
        x=Axis("x (px)", (0.0, 100.0)),
        y=Axis("y (px)", (0.0, 50.0)),
        z=Axis("time (s)", (0.0, 10.0)) if three_d else None,
        color=ColorScale("elapsed time (s)", 0.0, 10.0),
        camera=Camera(30.0, -45.0) if three_d else None,
        aspect=1.0,
        invert_y=True,
    )


def test_a_3d_grid_shares_one_camera_and_one_set_of_ranges():
    fig = plot_render.figure(_path_spec(4, three_d=True))
    axes = [ax for ax in fig.axes if hasattr(ax, "get_zlim")]
    assert len(axes) == 4
    assert {(ax.elev, ax.azim) for ax in axes} == {(30.0, -45.0)}
    assert {ax.get_zlim() for ax in axes} == {(0.0, 10.0)}
    # y runs down, as in the video
    assert {ax.get_ylim() for ax in axes} == {(50.0, 0.0)}


@pytest.mark.parametrize("n_panels", [1, 2, 3, 4])
def test_the_colour_bar_covers_no_3d_axis_label(n_panels):
    """A 3-D panel's z label sits outside its box, and the colour bar was placed at the
    2-D gap: on the user guide's three-recording grid it hid the right column's
    ``time (s)``."""
    spec = _path_spec(n_panels, three_d=True)
    spec = PlotSpec(**{**spec.__dict__, "camera": Camera(22.0, -60.0)})
    fig = plot_render.figure(spec)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scenes = [ax for ax in fig.axes if hasattr(ax, "get_zlim")]
    (bar,) = [ax for ax in fig.axes if not hasattr(ax, "get_zlim")]
    bar_box = bar.get_tightbbox(renderer)
    for ax in scenes:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            label = axis.label.get_window_extent(renderer)
            assert not label.overlaps(bar_box), axis.label.get_text()


@pytest.mark.parametrize("kind", ["body", "body_hist", "speed"])
@pytest.mark.parametrize("group_by", [None, "g"])
def test_a_short_panel_s_axis_label_fits_beside_it(kind, group_by):
    """A curve-over-time panel is short, and a y label that named the whole measure
    (``distance between mask centroids (px)``, before this port) was taller than the panel and
    ran into the figure's title on a three-recording grid."""
    from detail.analysis import location_plots as lp
    from detail.configs.location_plots import LocationPlotView

    t = np.linspace(0.0, 300.0, 600)
    traces = [
        lp.Trace(
            sid, sid, lp.FrameInfo(1280.0, 720.0, None), t,
            0.5 + 0.3 * np.sin(t / 20), np.full(len(t), 0.5),
            lp.Body(
                np.full(len(t), 0.5), np.full(len(t), 0.5),
                150 + 20 * np.sin(t / 30), np.full(len(t), 60.0), np.zeros(len(t)),
                np.full(len(t), 7000.0), np.full(len(t), 9000.0),
            ),
        )
        for sid in ("test5", "test9", "test10")
    ]  # fmt: skip
    view = LocationPlotView(kind=kind, group_by=group_by, body_measure="mask_area")
    values = {"test5": "a", "test9": "a", "test10": "b"} if group_by else None
    fig = plot_render.figure(lp.build_plot(traces, view, values))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title = fig._suptitle.get_window_extent(renderer)
    for ax in fig.axes:
        label = ax.yaxis.label.get_window_extent(renderer)
        assert label.height <= ax.get_window_extent(renderer).height
        assert not label.overlaps(title), ax.yaxis.label.get_text()


def test_2d_position_panels_run_y_down():
    fig = plot_render.figure(_path_spec(2))
    assert fig.axes[0].get_ylim() == (50.0, 0.0)


def test_a_path_skips_the_palest_ramp_steps_and_a_heat_map_does_not():
    path_fig = plot_render.figure(_path_spec(1))
    coll = path_fig.axes[0].collections[0]
    low = coll.cmap(0.0)
    assert low[:3] == pytest.approx(
        [c / 255 for c in chart_theme.rgb(chart_theme.LIGHT.sequential[2])], abs=0.01
    )
    heat = PlotSpec(
        kind="heatmap",
        title="Occupancy",
        panels=[
            Panel(
                "r",
                heat=Heat(np.eye(2) / 2, np.linspace(0, 1, 3), np.linspace(0, 1, 3)),
            )
        ],
        x=Axis("x", (0.0, 1.0)),
        y=Axis("y", (0.0, 1.0)),
        color=ColorScale("share", 0.0, 0.5),
    )
    mesh = plot_render.figure(heat).axes[0].collections[0]
    assert mesh.cmap(0.0)[:3] == pytest.approx(
        [c / 255 for c in chart_theme.rgb(chart_theme.LIGHT.sequential[0])], abs=0.01
    )
    # A cell the animal never entered is drawn as nothing: the ramp's palest step is a grey,
    # and a frame of it read as a floor the animal had covered.
    drawn = mesh.get_array()
    assert np.ravel(drawn.mask).tolist() == [False, True, True, False]


def test_identity_lines_and_bands_take_the_theme_categorical_colours():
    x = np.arange(5.0)
    spec = PlotSpec(
        kind="speed",
        title="Speed",
        panels=[
            Panel(
                "",
                lines=[Line(x, x, identity=1, label="wt")],
                bands=[Band(x, x - 1, x + 1, identity=1)],
            )
        ],
        x=Axis("time (s)"),
        y=Axis("speed"),
        legend=True,
    )
    fig = plot_render.figure(spec, plot_render.DEFAULT_STYLE.with_theme("dark"))
    from matplotlib.colors import to_hex

    assert (
        to_hex(fig.axes[0].lines[0].get_color()) == chart_theme.DARK.identity(1).lower()
    )
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["wt"]


def test_an_animal_in_several_panels_is_named_once_in_the_legend():
    x = np.arange(5.0)
    panels = [
        Panel(
            p,
            lines=[
                Line(x, x, identity=0, label="intruder"),
                Line(x, x, identity=1, label="resident"),
            ],
        )
        for p in ("a", "b")
    ]
    spec = PlotSpec(
        kind="trace", title="Path", panels=panels, x=Axis("x"), y=Axis("y"), legend=True,
        note="Not plotted: c · intruder: no tracked rows",
    )  # fmt: skip
    fig = plot_render.figure(spec)
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["intruder", "resident"]
    assert "Not plotted: c · intruder" in fig._suptitle.get_text()


def _drawn_segments(polylines, centres):
    """``[(start, end, colour value)]`` in the order a collection of ``polylines`` draws them."""
    out = []
    for poly, value in zip(polylines, centres):
        for a, b in zip(poly[:-1], poly[1:]):
            out.append((tuple(a), tuple(b), value))
    return out


@pytest.mark.parametrize("dims", [2, 3])
def test_a_coloured_line_draws_each_segment_once_in_its_own_order(dims):
    """Drawn as runs of one colour step, a line is still every segment the per-segment rule
    drew — a non-finite end or value leaves one out — in the line's order, so what lies on
    top where a path crosses itself is what came later, each within half a step of its value.
    """
    rng = np.random.default_rng(3)
    n = 600
    points = np.cumsum(rng.normal(size=(n, dims)), axis=0)
    points[100:104] = np.nan  # a gap in the path
    values = rng.uniform(0.0, 50.0, n)  # a speed: a different step nearly every sample
    values[300] = np.nan
    values[400:420] = 80.0  # above the scale: the top step, as the colour map clips it
    polylines, centres = plot_render.colour_steps(points, values, 0.0, 64.0)
    mid = (values[:-1] + values[1:]) / 2.0
    expected = [
        (tuple(points[i]), tuple(points[i + 1]), min(mid[i], 64.0))
        for i in range(n - 1)
        if np.isfinite(points[i : i + 2]).all() and np.isfinite(mid[i])
    ]
    drawn = _drawn_segments(polylines, centres)
    assert [(a, b) for a, b, _ in drawn] == [(a, b) for a, b, _ in expected]
    half_step = 64.0 / plot_render.COLOUR_STEPS / 2.0
    assert max(abs(d[2] - e[2]) for d, e in zip(drawn, expected)) <= half_step + 1e-9


def test_a_time_coloured_path_is_a_few_polylines_not_one_per_segment():
    """The redraw cost: one drawn path per segment made a group panel of 25 half-hour paths
    225,000 paths, and a change of *Group by* took seconds to show."""
    t = np.linspace(0.0, 1800.0, 9000)
    points = np.column_stack([np.cos(t), np.sin(t)])
    points[4000:4010] = np.nan
    polylines, _ = plot_render.colour_steps(points, t, 0.0, 1800.0)
    assert len(polylines) <= plot_render.COLOUR_STEPS + 1  # a step split by the gap
    assert sum(len(p) for p in polylines) < len(t) * 1.02


def test_a_figure_reports_each_panel_and_a_render_its_write(tmp_path):
    """Each report is where a newer draw stops this one (`gui/busy.run_progress`), so a grid
    reports between panels rather than once at the end."""
    heard = []
    plot_render.render(
        _path_spec(4),
        tmp_path / "fig",
        formats=("png", "svg"),
        progress=lambda stage, fraction, detail: heard.append((stage, fraction)),
    )
    assert heard == [
        ("drawing", 0.0),
        ("drawing", 0.25),
        ("drawing", 0.5),
        ("drawing", 0.75),
        ("writing", 0.0),
        ("writing", 1.0),
    ]
    heard.clear()
    plot_render.render_png(
        _path_spec(2), progress=lambda s, f, d: heard.append((s, f, d))
    )
    assert heard == [("drawing", 0.0, "r0"), ("drawing", 0.5, "r1")]


def test_render_writes_each_format_and_png_bytes_are_a_png(tmp_path):
    paths = plot_render.render(
        _path_spec(), tmp_path / "a" / "fig", formats=("png", "pdf")
    )
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    data = plot_render.render_png(_path_spec(), theme="dark")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("n, cols", [(1, 1), (4, 2), (5, 3), (84, 10)])
def test_columns_stay_about_square_and_capped(n, cols):
    assert plot_render.columns_for(n) == cols
