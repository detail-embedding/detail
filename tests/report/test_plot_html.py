"""The interactive HTML renderer for ``report/plot_spec.PlotSpec`` (§16 "Location plots"): the
matplotlib renderer's twin — same panels, ranges, colour roles, legend and note — and a 3-D grid
that stays one view."""

import numpy as np

from detail.core import chart_theme
from detail.report import plot_html
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


def _path_spec(n=2, *, three_d=False):
    t = np.linspace(0, 10, 20)
    x = np.r_[np.linspace(0, 100, 10), np.nan, np.linspace(0, 100, 9)]
    return PlotSpec(
        kind="trace3d" if three_d else "trace",
        title="Path",
        panels=[
            Panel(f"r{i}", lines=[Line(x, x / 2, z=t if three_d else None, values=t)])
            for i in range(n)
        ],
        x=Axis("x (px)", (0.0, 100.0)),
        y=Axis("y (px)", (0.0, 50.0)),
        z=Axis("time (s)", (0.0, 10.0)) if three_d else None,
        color=ColorScale("elapsed time (s)", 0.0, 10.0),
        camera=Camera(22.0, -60.0) if three_d else None,
        aspect=1.0,
        invert_y=True,
        note="Not plotted: r9: no tracked rows",
    )


def test_a_path_plot_keeps_the_ranges_scale_and_y_down():
    fig = plot_html.figure(_path_spec())
    layout = fig.layout
    assert layout.coloraxis.cmin == 0.0 and layout.coloraxis.cmax == 10.0
    assert layout.coloraxis.colorbar.title.text == "elapsed time (s)"
    assert tuple(layout.yaxis.range) == (50.0, 0.0)  # y runs down, as in the video
    assert layout.yaxis.scaleanchor == "x" and layout.yaxis.scaleratio == 1.0
    assert tuple(layout.xaxis2.range) == (0.0, 100.0)
    assert "Not plotted: r9" in layout.title.text
    (first,) = [t for t in fig.data if t.xaxis == "x"]
    assert first.marker.coloraxis == "coloraxis"
    assert None in first.x  # the gap is a break, not a jump


def test_a_path_skips_the_palest_steps_and_a_heat_map_keeps_them():
    fig = plot_html.figure(_path_spec(1))
    low = fig.layout.coloraxis.colorscale[0][1]
    assert low.lower() == chart_theme.LIGHT.sequential[2].lower()
    heat = PlotSpec(
        kind="heatmap",
        title="Occupancy",
        panels=[
            Panel(
                "r",
                heat=Heat(np.eye(2) / 2, np.linspace(0, 1, 3), np.linspace(0, 1, 3)),
            )
        ],
        x=Axis("x"),
        y=Axis("y"),
        color=ColorScale("share", 0.0, 0.5),
    )
    fig = plot_html.figure(heat)
    assert fig.data[0].type == "heatmap" and fig.data[0].coloraxis == "coloraxis"
    assert (
        fig.layout.coloraxis.colorscale[0][1].lower() == chart_theme.LIGHT.sequential[0]
    )


def test_an_animal_in_several_panels_has_one_legend_entry_in_its_colour():
    x = np.arange(5.0)
    panels = [
        Panel(
            p,
            lines=[
                Line(x, x, identity=0, label="Intruder"),
                Line(x, x, identity=1, label="Resident"),
            ],
        )
        for p in ("a", "b")
    ]
    spec = PlotSpec(
        kind="trace", title="Path", panels=panels, x=Axis("x"), y=Axis("y"), legend=True
    )
    fig = plot_html.figure(spec, theme="dark")
    shown = [(t.name, t.line.color) for t in fig.data if t.showlegend]
    assert shown == [
        ("Intruder", chart_theme.DARK.identity(0)),
        ("Resident", chart_theme.DARK.identity(1)),
    ]
    assert fig.layout.paper_bgcolor == chart_theme.DARK.surface


def test_a_histogram_steps_and_a_band_fills_under_it():
    x = np.arange(5.0)
    spec = PlotSpec(
        kind="distance_hist",
        title="Distances",
        panels=[
            Panel(
                "",
                lines=[Line(x, x, identity=0, label="wt (n=2)", step=True)],
                bands=[Band(x, x - 1, x + 1, identity=0, step=True)],
            )
        ],
        x=Axis("distance"),
        y=Axis("share of frames"),
        legend=True,
    )
    fig = plot_html.figure(spec)
    lines = [t for t in fig.data if t.name == "wt (n=2)"]
    assert lines[0].line.shape == "hvh"
    assert any(t.fill == "tonexty" for t in fig.data)


def test_a_3d_grid_shares_one_camera_and_the_page_links_them(tmp_path):
    spec = _path_spec(3, three_d=True)
    fig = plot_html.figure(spec)
    eyes = {
        (round(s.camera.eye.x, 6), round(s.camera.eye.y, 6), round(s.camera.eye.z, 6))
        for s in (fig.layout.scene, fig.layout.scene2, fig.layout.scene3)
    }
    assert len(eyes) == 1
    assert tuple(fig.layout.scene.yaxis.range) == (50.0, 0.0)
    page = plot_html.render_html(spec, tmp_path / "out" / "p.html").read_text(
        encoding="utf-8-sig"
    )
    assert plot_html.LINK_CAMERAS_MARK in page  # turning one scene turns the rest
    assert "<title>Path</title>" in page
    # plotly.js is embedded, not fetched: the page opens offline.
    assert '<script src="http' not in page and "<script src='http" not in page


def test_a_2d_page_carries_no_camera_script(tmp_path):
    page = plot_html.render_html(_path_spec(), tmp_path / "p.html").read_text(
        encoding="utf-8-sig"
    )
    assert plot_html.LINK_CAMERAS_MARK not in page


def test_a_gap_in_the_colour_values_does_not_stop_the_page(tmp_path):
    """Real paths carry NaN speeds after a gap (and the drawn gap rows); plotly refuses a
    missing colour, which failed every 3-D page of a real run."""
    t = np.linspace(0, 10, 20)
    values = t.copy()
    values[5] = np.nan
    x = np.linspace(0, 100, 20)
    x[5] = np.nan
    for three_d in (False, True):
        spec = PlotSpec(
            kind="trace3d" if three_d else "trace",
            title="Path",
            panels=[
                Panel("r", lines=[Line(x, x, z=t if three_d else None, values=values)])
            ],
            x=Axis("x"),
            y=Axis("y"),
            z=Axis("t") if three_d else None,
            color=ColorScale("speed", 0.0, 10.0),
            camera=Camera(22.0, -60.0) if three_d else None,
        )
        assert plot_html.render_html(spec, tmp_path / f"p{three_d}.html").is_file()
