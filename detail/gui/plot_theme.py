"""The pyqtgraph and QPainter adapter for the chart design language.

``core/chart_theme.py`` holds every colour, weight and size a chart is drawn with; this module
turns them into Qt's vocabulary — a themed ``PlotWidget``, pens and brushes by *role* rather than
by RGB triple, and ``QColor`` for the hand-drawn lanes and maps. The matplotlib twin is
``report/style.py``; a plot on screen and the same chart exported are one design.

**A plot follows the application's theme.** It used to be ``setBackground("w")`` at every
construction site — "a plot is on white" — which in a dark window was a row of lit panels with
grey tick labels on them. The theme is read from the palette the window is actually drawn with,
so a plot is a card one step lighter than its page in either theme (``ChartTheme.surface``).

**Chrome follows a live switch; data follows its next draw.** A system light/dark flip reaches the
application as a new palette, and every plot made here re-themes its background, axes and labels
when its widget sees that change. The marks on it — a curve, a scatter — carry the colour they
were drawn with until their widget redraws; the two palettes differ by one categorical step
(blue), so a stale mark is the right hue on the wrong surface for as long as it lasts.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette

from detail.core.chart_theme import DARK, LIGHT, SCREEN_MARKS, ChartTheme

#: A plot's marks, in screen pixels: line widths, marker sizes, tick length.
MARKS = SCREEN_MARKS


def current() -> ChartTheme:
    """The chart theme matching the palette the application is drawn with.

    The colour scheme when the platform reports one; otherwise the window colour's lightness,
    which is what an offscreen test platform or a hand-set palette actually shows.
    """
    app = QGuiApplication.instance()
    if app is None:
        return LIGHT
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return DARK
    if scheme == Qt.ColorScheme.Light:
        return LIGHT
    window = app.palette().color(QPalette.ColorRole.Window)
    return DARK if window.lightness() < 128 else LIGHT


def qcolor(color: str, alpha: int | None = None) -> QColor:
    """A theme colour as a ``QColor``, optionally with an alpha (0-255)."""
    out = QColor(color)
    if alpha is not None:
        out.setAlpha(alpha)
    return out


def identity(index: int, theme: ChartTheme | None = None) -> QColor:
    """The identity colour for series ``index`` (see ``ChartTheme.identity``)."""
    return QColor((theme or current()).identity(index))


def pen(color: str | QColor, *, width: float | None = None, dashed: bool = False):
    """A cosmetic pen — ``width`` in screen pixels whatever the view's zoom."""
    import pyqtgraph as pg

    style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
    return pg.mkPen(
        QColor(color), width=MARKS.line_width if width is None else width, style=style
    )


def brush(color: str | QColor, alpha: int | None = None):
    """A fill in ``color``, optionally translucent."""
    import pyqtgraph as pg

    fill = QColor(color)
    if alpha is not None:
        fill.setAlpha(alpha)
    return pg.mkBrush(fill)


def fill_alpha() -> int:
    """The opacity of a filled region under data (a band, a violin body), as 0-255."""
    return round(MARKS.fill_alpha * 255)


def apply(plot, theme: ChartTheme | None = None) -> None:
    """Theme a ``PlotWidget``'s chrome: its surface, axis lines, ticks and labels.

    Idempotent, so a theme change re-applies it. Ticks point *out* of the plot area, as on paper;
    tick labels wear the secondary ink and axis titles the primary one, never a series colour. The
    SI-prefix rewrite is off on both axes — pyqtgraph otherwise relabels a sub-1 axis
    "300 (x0.001)", a unit a confidence does not have (``gui_design`` §3.1).
    """
    theme = theme or current()
    plot.setBackground(QColor(theme.surface))
    plot.setAntialiasing(True)
    axis_pen = pen(theme.axis, width=MARKS.axis_width)
    for name in ("left", "bottom", "right", "top"):
        axis = plot.getAxis(name)
        axis.setPen(axis_pen)
        axis.setTickPen(axis_pen)
        axis.setTextPen(QColor(theme.text_secondary))
        # The axis title in primary ink, the ticks in secondary. `setTextPen` colours both, and a
        # later `setLabel` with no style keeps `labelStyle` — so the title's ink goes here, and an
        # existing title is re-set with its own text to pick it up.
        axis.labelStyle["color"] = theme.text
        if axis.labelText or axis.labelUnits:
            axis.setLabel(axis.labelText, axis.labelUnits, axis.labelUnitPrefix)
        # pyqtgraph: a positive tick length points toward the labels, i.e. out of the plot. Major
        # ticks only — its two minor levels drew a comb along every axis that carried no value.
        axis.setStyle(
            tickLength=int(MARKS.tick_length), tickTextOffset=4, maxTickLevel=0
        )
        axis.enableAutoSIPrefix(False)
    plot._chart_theme = theme.name


#: The events that mean "the palette this widget is drawn with may have changed" — a system
#: light/dark switch arrives as the second, a palette set on an ancestor as the first.
_PALETTE_EVENTS = (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange)

_THEMED_CLASS = None


def _themed_class():
    """``PlotWidget`` that re-themes its chrome when its palette changes — built on first use,
    so importing this module does not import pyqtgraph."""
    global _THEMED_CLASS
    if _THEMED_CLASS is None:
        import pyqtgraph as pg

        class ThemedPlotWidget(pg.PlotWidget):
            """A ``PlotWidget`` that follows the application's chart theme."""

            def changeEvent(self, event) -> None:  # noqa: N802 — Qt's name
                super().changeEvent(event)
                # pyqtgraph's own constructor delivers a palette event before the plot item
                # exists; only a plot `apply` has already themed once is re-themed.
                if event.type() in _PALETTE_EVENTS and hasattr(self, "_chart_theme"):
                    apply(self)

        _THEMED_CLASS = ThemedPlotWidget
    return _THEMED_CLASS


def themed_plot(parent=None):
    """A ``PlotWidget`` in the current chart theme, re-themed on a light/dark switch.

    The one way a plot is built in this package: a construction site that sets its own
    background or axis pens is a second copy of the design, and the copies drift.
    """
    plot = _themed_class()(parent=parent)
    apply(plot)
    return plot


#: Where :func:`add_legend` anchors, as pyqtgraph offsets from the view's corner.
_LEGEND_CORNERS = {
    "top-left": (8, 8),
    "top-right": (-8, 8),
    "bottom-left": (8, -8),
    "bottom-right": (-8, -8),
}


def add_legend(plot, *, corner: str, theme: ChartTheme | None = None):
    """A borderless legend in ``corner``, on a translucent card of the plot's own surface.

    ``corner`` has no default because only the site knows where its data is not: a training curve
    rises into the top right and a PR curve falls out of the top left. The card lets a curve pass
    under the legend without its line striking through the labels.
    """
    theme = theme or current()
    return plot.addLegend(
        offset=_LEGEND_CORNERS[corner],
        labelTextColor=QColor(theme.text_secondary),
        brush=brush(theme.surface, alpha=215),
        pen=None,
        verSpacing=-4,
    )


def effect_colour(value: float, theme: ChartTheme | None = None) -> QColor:
    """The shade for a cluster whose effect sits at ``value`` on a fixed −1..+1 axis.

    The theme's ``diverging`` scale: its negative pole at −1, its neutral middle at 0, its
    positive pole at +1. Zero is ``neutral`` exactly — a cluster inside the floor is "not shown
    to differ", drawn the same grey as one with no clips to count (NaN) — so a tree shaded by an
    effect colours only what cleared, and the sign says which way. One function for the tree's
    blocks and the key beside them, as :func:`match_colour` is for the share ramp.
    """
    theme = theme or current()
    if value != value or value == 0:
        return QColor(theme.neutral)
    stops = theme.diverging
    position = (min(max(float(value), -1.0), 1.0) + 1.0) / 2.0 * (len(stops) - 1)
    low = min(int(position), len(stops) - 2)
    t = position - low
    a, b = QColor(stops[low]), QColor(stops[low + 1])
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


def size_colour(value: float, theme: ChartTheme | None = None) -> QColor:
    """The shade for a cluster whose difference has a size and no direction, ``value`` in 0..1.

    A profile (three or more named levels) scores a cluster by the pair of levels it differs
    most between, so its sign belongs to that pair and not to the cluster. Its ramp is its
    own: from the diverging scale's neutral middle to the theme's ink, no hue. It once reused
    the scale's "more" half, so one colour meant *more* on a two-value screen and *differs
    either way* on this one (user, 2026-11-16). Zero and NaN are ``neutral`` exactly, as for
    :func:`effect_colour`.
    """
    theme = theme or current()
    if value != value or value <= 0:
        return QColor(theme.neutral)
    t = min(float(value), 1.0)
    a = QColor(theme.diverging[len(theme.diverging) // 2])
    b = QColor(theme.text)
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


def match_colour(share: float, theme: ChartTheme | None = None) -> QColor:
    """The shade for a cluster of which ``share`` (0..1) of the clips match a selection.

    **Always the fixed 0–100% scale, never stretched to the clusters' own spread.** Stretched,
    an even spread (every cluster 43–55%) fills the whole ramp and reads as a strong pattern; on
    the fixed scale it is one even mid shade, which is what it is. The ramp is the theme's
    ``sequential`` magnitude scale, lightness with no hue, so its low end sits near the
    surface in either theme and a share of zero (or nothing to count) is that low end: "none
    of it". It was ``neutral`` while the ramp was blue; on a grey ramp ``neutral`` is a shade
    part-way up it, and a cluster with no match would read as one with some.

    One function for the tree's blocks and the key under it (`gui/match_key.py`), so the key
    cannot disagree with the picture it explains.
    """
    theme = theme or current()
    stops = theme.sequential
    if share != share or share <= 0:
        return QColor(stops[0])
    position = min(float(share), 1.0) * (len(stops) - 1)
    low = min(int(position), len(stops) - 2)
    t = position - low
    a, b = QColor(stops[low]), QColor(stops[low + 1])
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )
