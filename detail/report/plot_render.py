"""Draw a ``report/plot_spec.PlotSpec`` with matplotlib — files, or PNG bytes for the GUI.

The spec's colour roles are resolved here against the chart theme (``report/style.py``), so the
same spec renders light or dark. 2-D panels go through the house style (``figures.styled`` /
``save_figure``); 3-D panels share the spec's one camera and axis ranges, so a grid of them is
one view of every recording, not several views that happen to sit side by side.
"""

import io
import math
import textwrap
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detail.report.figures import caption, save_figure, styled
from detail.report.plot_spec import Line, PlotSpec
from detail.report.style import DEFAULT_STYLE, FigureStyle, sequential_cmap

#: Width of one panel, in inches. A grid of panels is this times its column count.
PANEL_WIDTH_IN = 2.6

#: The most panels a row holds; past this a grid grows downward.
MAX_COLUMNS = 10

#: The GUI's picture resolution — enough to read at the result page's size, small enough to
#: redraw while a camera slider moves.
SCREEN_DPI = 110


def columns_for(n: int) -> int:
    """Panels per row for ``n`` panels: about square, never more than ``MAX_COLUMNS``."""
    return max(1, min(MAX_COLUMNS, math.ceil(math.sqrt(n))))


def figure(spec: PlotSpec, style: FigureStyle = DEFAULT_STYLE):
    """Build the matplotlib figure for ``spec``. The caller saves or closes it."""
    n = len(spec.panels)
    cols = columns_for(n)
    rows = -(-n // cols)
    panel_h = _panel_height(spec)
    with styled(style):
        fig = plt.figure(figsize=(PANEL_WIDTH_IN * cols + 0.9, panel_h * rows + 0.5))
        projection = "3d" if spec.three_d else None
        axes = [
            fig.add_subplot(rows, cols, i + 1, projection=projection) for i in range(n)
        ]
        has_heat = any(p.heat is not None for p in spec.panels)
        cmap = sequential_cmap(style) if has_heat else _line_cmap(style)
        for ax, panel in zip(axes, spec.panels):
            _draw_panel(ax, spec, panel, style, cmap)
        if spec.color is not None:
            mappable = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(spec.color.vmin, spec.color.vmax)
            )
            bar = fig.colorbar(
                mappable,
                ax=axes,
                shrink=0.8 if n > 1 else 0.9,
                pad=_BAR_PAD_3D if spec.three_d else 0.05,
            )
            bar.outline.set_visible(False)
            bar.set_label(spec.color.label)
        if spec.legend and axes:
            _legend(fig, spec, style)
        heading = spec.title
        if spec.note:
            width = int(fig.get_figwidth() * _CHARS_PER_INCH)
            heading += "\n" + textwrap.fill(spec.note, width=max(width, 40))
        fig.suptitle(heading, x=0.0, ha="left")
    return fig


#: The colour bar's gap from a 3-D grid, as a fraction of the figure: a 3-D panel's z label
#: sits outside its box, where matplotlib's 2-D gap (0.05) puts the bar over it.
_BAR_PAD_3D = 0.12

#: Roughly how many characters of the figure title's size fit in an inch, for wrapping a note.
_CHARS_PER_INCH = 13


def _legend(fig, spec: PlotSpec, style: FigureStyle) -> None:
    """One entry per distinct (identity, label) over every panel, outside the plot area."""
    from matplotlib.lines import Line2D

    seen: dict[tuple[int, str], None] = {}
    for panel in spec.panels:
        for line in panel.lines:
            if line.identity is not None and line.label:
                seen.setdefault((line.identity, line.label), None)
    handles = [
        Line2D(
            [], [], color=style.color(identity), lw=style.marks.line_width, label=label
        )
        for identity, label in sorted(seen, key=lambda key: key[0])
    ]
    if handles:
        fig.legend(handles=handles, loc="outside right upper", frameon=False)


#: How many of the palest sequential steps a coloured *line* skips. The ramp's low end sits near
#: the surface so an empty heat-map cell reads as nothing; a thin line in that colour is invisible.
_LINE_RAMP_SKIP = 2


def _line_cmap(style: FigureStyle):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"detail_sequential_line_{style.theme}",
        list(style.chart.sequential[_LINE_RAMP_SKIP:]),
    )


def _panel_height(spec: PlotSpec) -> float:
    if spec.three_d:
        return PANEL_WIDTH_IN * 1.15
    if spec.aspect is not None and spec.x.range and spec.y.range:
        span_x = spec.x.range[1] - spec.x.range[0]
        span_y = spec.y.range[1] - spec.y.range[0]
        ratio = (span_y / span_x) * spec.aspect if span_x else 1.0
        # The plot area is the panel width less its y-axis gutter; add the title + x-axis strip.
        plot_w = PANEL_WIDTH_IN - 0.55
        return max(1.2, min(PANEL_WIDTH_IN * 1.6, plot_w * ratio + 0.75))
    return PANEL_WIDTH_IN * 0.62


def _draw_panel(ax, spec: PlotSpec, panel, style: FigureStyle, cmap) -> None:
    chart = style.chart
    if panel.heat is not None:
        heat = panel.heat
        # A cell the animal never entered is drawn as nothing, not as the ramp's palest step:
        # the chart theme's sequential ramp starts at a light grey, and a frame of it read as
        # a floor the animal had covered.
        grid = np.ma.masked_where(~(np.asarray(heat.grid) > 0), heat.grid)
        ax.pcolormesh(
            heat.x_edges,
            heat.y_edges,
            grid,
            cmap=cmap,
            vmin=spec.color.vmin if spec.color else None,
            vmax=spec.color.vmax if spec.color else None,
            shading="flat",
            rasterized=True,
        )
    for band in panel.bands:
        ax.fill_between(
            band.x,
            band.lo,
            band.hi,
            color=style.color(band.identity),
            alpha=style.marks.fill_alpha,
            lw=0,
            zorder=1,
            step="mid" if band.step else None,
        )
    for line in panel.lines:
        _draw_line(ax, spec, line, style, cmap)
    if spec.x.range:
        ax.set_xlim(*spec.x.range)
    if spec.y.range:
        lo, hi = spec.y.range
        ax.set_ylim((hi, lo) if spec.invert_y and not spec.three_d else (lo, hi))
    ax.set_xlabel(spec.x.label)
    ax.set_ylabel(spec.y.label)
    if spec.three_d:
        _dress_3d(ax, spec, chart)
    elif spec.aspect is not None:
        ax.set_aspect(spec.aspect, adjustable="box")
    if panel.title:
        ax.set_title(panel.title)
    caption(ax, panel.note, style=style)


def _draw_line(ax, spec: PlotSpec, line: Line, style: FigureStyle, cmap) -> None:
    if line.values is None:
        width = style.marks.line_width
        color = (
            style.color(line.identity)
            if line.identity is not None
            else style.chart.text
        )
        if line.z is None:
            ax.plot(
                line.x,
                line.y,
                color=color,
                lw=width,
                label=line.label or None,
                drawstyle="steps-mid" if line.step else "default",
            )
        else:
            ax.plot(line.x, line.y, line.z, color=color, lw=width)
        return
    if len(line.x) < 2:
        return
    # A path is drawn at the thin weight: it is dense, and a group panel overlays several.
    width = style.marks.thin_line_width
    norm = plt.Normalize(spec.color.vmin, spec.color.vmax)
    mid = (np.asarray(line.values[:-1]) + np.asarray(line.values[1:])) / 2.0
    if line.z is None:
        from matplotlib.collections import LineCollection

        pts = np.column_stack([line.x, line.y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        keep = np.isfinite(segs).all(axis=(1, 2)) & np.isfinite(mid)
        coll = LineCollection(segs[keep], cmap=cmap, norm=norm, linewidths=width)
        coll.set_array(mid[keep])
        ax.add_collection(coll)
    else:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pts = np.column_stack([line.x, line.y, line.z])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        keep = np.isfinite(segs).all(axis=(1, 2)) & np.isfinite(mid)
        coll = Line3DCollection(segs[keep], cmap=cmap, norm=norm, linewidths=width)
        coll.set_array(mid[keep])
        ax.add_collection3d(coll)


def _dress_3d(ax, spec: PlotSpec, chart) -> None:
    if spec.z is not None:
        ax.set_zlabel(spec.z.label)
        if spec.z.range:
            ax.set_zlim(*spec.z.range)
    if spec.y.range and spec.invert_y:
        lo, hi = spec.y.range
        ax.set_ylim(hi, lo)
    if spec.camera is not None:
        ax.view_init(elev=spec.camera.elev, azim=spec.camera.azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.line.set_color(chart.axis)
    ax.tick_params(colors=chart.text_secondary)
    if spec.aspect is not None and spec.x.range and spec.y.range:
        span_x = spec.x.range[1] - spec.x.range[0]
        span_y = (spec.y.range[1] - spec.y.range[0]) * spec.aspect
        top = max(span_x, span_y) or 1.0
        ax.set_box_aspect((span_x / top, span_y / top, 1.0))


def render(
    spec: PlotSpec,
    out_stem: Path | str,
    *,
    formats: tuple[str, ...] = ("png",),
    theme: str = "light",
) -> list[Path]:
    """Write ``spec`` as ``<out_stem>.<fmt>`` for each format; returns the paths."""
    style = DEFAULT_STYLE.with_theme(theme)
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    return save_figure(figure(spec, style), out_stem, formats)


def render_png(spec: PlotSpec, *, theme: str = "light", dpi: int = SCREEN_DPI) -> bytes:
    """``spec`` as PNG bytes at screen resolution — the GUI's picture."""
    style = DEFAULT_STYLE.with_theme(theme)
    fig = figure(spec, style)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()
