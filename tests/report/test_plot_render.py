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


@pytest.mark.parametrize("distance_by", ["mask_centroid", "bbox_centroid", "mask_edge"])
@pytest.mark.parametrize("kind", ["distance", "distance_hist"])
def test_a_distance_plot_s_axis_labels_fit_beside_their_panels(kind, distance_by):
    """A distance-over-time panel is short, and its y label once named the whole measure
    (``distance between mask centroids (px)``): taller than the panel, it ran into the
    figure's title on the user guide's three-recording grid."""
    from detail.analysis import location_plots as lp
    from detail.configs.location_plots import LocationPlotView

    t = np.linspace(0.0, 300.0, 600)
    frame = lp.FrameInfo(1280.0, 720.0, None)
    pairs = [
        lp.PairSeries(
            sid, sid, "intruder – resident", (0, 1), frame, t,
            np.abs(np.sin(t / 20)) * 0.5, np.abs(np.sin(t / 20)) * 600,
        )
        for sid in ("test5", "test9", "test10")
    ]  # fmt: skip
    view = LocationPlotView(kind=kind, distance_by=distance_by, units="px")
    fig = plot_render.figure(lp.build_distance_plot(pairs, view))
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
