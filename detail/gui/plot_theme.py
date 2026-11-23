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

import math

from PyQt6.QtCore import QEvent, QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QGuiApplication,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QRegion,
)
from PyQt6.QtWidgets import QApplication, QStyle

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


def label_identity(index: int, theme: ChartTheme | None = None) -> QColor:
    """The colour for a behavior label's slot ``index`` (see ``ChartTheme.label_identity``):
    never one of the two the diverging scale keeps for less and more."""
    return QColor((theme or current()).label_identity(index))


#: A label's **second channel**: the fill drawn over its colour, by slot. Two labels must never
#: differ by hue alone — a reader who cannot tell the slot-0 green from the slot-1 pink (the
#: user who asked for this, 2026-11-23) had nothing else to tell "groom" from "hi" by on BROWSE's
#: ethogram and in its key. Slot 0 is solid, so a one-behavior run looks as it always did.
#:
#: Nothing here has a vertical stroke: a band one or two pixels wide can sit entirely on a
#: vertical line, and would then read as the surface — a gap — rather than as a label.
#:
#: **One pattern per label colour**, so the pair wraps with the colour, at the limit
#: ``ChartTheme.label_identity`` already documents: every slot with a colour of its own has a
#: pattern of its own, and all twelve are distinct in greyscale whatever their colours
#: (``tests/gui/test_identity_marks.py``). Eight shapes are stroked in the theme's surface colour;
#: the last four repeat the boldest of them in its ink (``inked …``), the opposite lightness.
LABEL_PATTERNS: tuple[str, ...] = (
    "solid",
    "diagonal",
    "dots",
    "horizontal",
    "crosshatch",
    "back diagonal",
    "zigzag",
    "plus",
    "inked diagonal",
    "inked dots",
    "inked horizontal",
    "inked crosshatch",
)

#: The prefix that strokes a pattern in the theme's ink rather than its surface.
_INKED = "inked "

#: A dot's side. Square, and drawn without antialiasing: at three pixels a square reads as a
#: dot, and antialiased round points cost 12 ms of a 33 ms ethogram paint (six rows, 1,865
#: bands, offscreen).
_DOT = 3.0

#: The distance between a pattern's rows of marks, and a stroke's thickness, in pixels: two or
#: three rows across the smallest thing one is drawn on, a 16px swatch or band. Thin enough that
#: a pattern covers about a quarter of the fill, so the colour still reads as the colour.
_PERIOD = 6.0
_STROKE = 1.5
#: The zigzag's stroke and swing above and below its row. Bolder than the rest: its strokes are
#: steep, so antialiasing spreads them thin, and at the common stroke it was the one pattern a
#: greyscale reader could lose on a light colour (slot 6's teal, against solid slot 0).
_ZIGZAG = 2.0


def label_mark(index: int, theme: ChartTheme | None = None) -> tuple[QColor, str]:
    """A behavior label's slot ``index`` as it is drawn: ``(colour, pattern)``.

    One mapping, so a band, a list swatch and the key beside them cannot disagree about which
    label is which — the colour from :func:`label_identity`, the pattern from
    :data:`LABEL_PATTERNS`, both from the same slot.
    """
    return label_identity(index, theme), LABEL_PATTERNS[index % len(LABEL_PATTERNS)]


def _grid(rect, period_x: float, period_y: float, *, stagger: bool):
    """``(x, y)`` of every cell of a grid anchored at the painter's origin that can touch
    ``rect`` — anchored there rather than at ``rect``, so neighbouring fills join up."""
    row = math.floor(rect.top() / period_y) - 1
    while row * period_y <= rect.bottom() + period_y:
        shift = period_x / 2 if stagger and row % 2 else 0.0
        col = math.floor((rect.left() - shift) / period_x) - 1
        while col * period_x + shift <= rect.right() + period_x:
            yield col * period_x + shift, row * period_y
            col += 1
        row += 1


def _diagonals(rect, *, rising: bool, period: float) -> list[QLineF]:
    """Lines at 45°, one every ``period`` along the x axis, anchored at the origin."""
    top, bottom = rect.top(), rect.bottom()
    lines = []
    if rising:  # x + y = c, smallest at the top-left corner
        c = math.floor((rect.left() + top) / period) * period
        while c <= rect.right() + bottom:
            lines.append(QLineF(c - bottom, bottom, c - top, top))
            c += period
    else:  # x - y = c, smallest at the bottom-left corner
        c = math.floor((rect.left() - bottom) / period) * period
        while c <= rect.right() - top:
            lines.append(QLineF(c + top, top, c + bottom, bottom))
            c += period
    return lines


def _rows(rect, period: float):
    """``y`` of every row of marks, anchored at the origin, that can touch ``rect``: once each,
    since a line drawn twice over a dimmed band doubles its alpha."""
    row = math.floor(rect.top() / period) - 1
    while row * period <= rect.bottom() + period:
        yield row * period
        row += 1


def _horizontals(rect) -> list[QLineF]:
    left, right = rect.left() - 1, rect.right() + 1
    return [QLineF(left, y, right, y) for y in _rows(rect, _PERIOD)]


def _zigzags(rect) -> list[QLineF]:
    half = _PERIOD / 2
    lines = []
    for y in _rows(rect, _PERIOD * 1.25):
        x = math.floor(rect.left() / _PERIOD) * _PERIOD - _PERIOD
        while x <= rect.right() + _PERIOD:
            lines.append(QLineF(x, y - _ZIGZAG, x + half, y + _ZIGZAG))
            lines.append(QLineF(x + half, y + _ZIGZAG, x + _PERIOD, y - _ZIGZAG))
            x += _PERIOD
    return lines


def _dots(rect) -> list[QRectF]:
    half = _DOT / 2
    return [
        QRectF(x - half, y - half, _DOT, _DOT)
        for x, y in _grid(rect, _PERIOD * 1.2, _PERIOD, stagger=True)
    ]


def _pluses(rect) -> list[QLineF]:
    arm = _PERIOD * 0.45
    lines = []
    for x, y in _grid(rect, _PERIOD * 1.4, _PERIOD * 1.1, stagger=True):
        lines.append(QLineF(x - arm, y, x + arm, y))
        lines.append(QLineF(x, y - arm, x, y + arm))
    return lines


#: Each pattern's marks over a rectangle: lines (``QLineF``) or dots (``QRectF``), returned
#: rather than drawn, so a pattern is one ``drawLines`` / ``drawRects`` call — drawn mark by
#: mark, the dots alone were 6,400 calls into Qt per ethogram paint.
_PATTERN_MARKS = {
    "diagonal": lambda r: _diagonals(r, rising=True, period=_PERIOD * 1.25),
    "back diagonal": lambda r: _diagonals(r, rising=False, period=_PERIOD * 1.25),
    "crosshatch": lambda r: (
        _diagonals(r, rising=True, period=_PERIOD * 1.6)
        + _diagonals(r, rising=False, period=_PERIOD * 1.6)
    ),
    "horizontal": _horizontals,
    "zigzag": _zigzags,
    "dots": _dots,
    "plus": _pluses,
}


def _paint_pattern(painter, rect, pattern: str, ink: QColor) -> None:
    shape = pattern.removeprefix(_INKED)
    marks = _PATTERN_MARKS[shape](rect)
    if marks and isinstance(marks[0], QRectF):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ink)
        painter.drawRects(marks)
    else:
        pen = QPen(ink)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen.setWidthF(_ZIGZAG if shape == "zigzag" else _STROKE)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLines(marks)


def paint_label_mark(
    painter,
    rects,
    index: int,
    *,
    color: QColor | None = None,
    theme: ChartTheme | None = None,
) -> None:
    """Fill ``rects`` — one ``QRectF``, or several — with label slot ``index``'s mark: its
    colour, its pattern over it.

    The pattern is drawn in the theme's surface colour, so it reads as the page showing through
    the label's colour rather than as a third colour — or, for an ``inked`` one, in its text
    colour, the other end of the lightness scale. It carries the colour's own alpha, so a dimmed
    band dims its pattern with it. ``color`` stands in for the slot's colour (a dimmed one, or the
    canonical step a caller already holds); the pattern is always the slot's.

    **Several at once is how a strip draws**: the pattern is stroked once across all of them,
    clipped to their union, instead of once per rectangle. A recording holds hundreds of bouts,
    and per band the pattern tripled the ethogram's paint time; once per identity a row, it adds
    about 6 ms to 16 (six rows, 1,865 bands, offscreen).
    """
    if isinstance(rects, QRectF):
        rects = [rects]
    rects = [r for r in rects if r.width() > 0 and r.height() > 0]
    if not rects:
        return
    theme = theme or current()
    own, pattern = label_mark(index, theme)
    fill = QColor(own if color is None else color)
    for rect in rects:
        painter.fillRect(rect, fill)
    if pattern == "solid":
        return
    outline = QRegion()
    for rect in rects:
        outline = outline.united(rect.toAlignedRect())
    inked = pattern.startswith(_INKED)
    ink = QColor(theme.text if inked else theme.surface)
    ink.setAlphaF(fill.alphaF() * (0.8 if inked else 1.0))
    painter.save()
    painter.setClipRegion(outline, Qt.ClipOperation.IntersectClip)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    _paint_pattern(painter, QRectF(outline.boundingRect()), pattern, ink)
    painter.restore()


def label_swatch(index: int, theme: ChartTheme | None = None) -> QPixmap:
    """The key for label slot ``index`` in a list: its colour **and** its pattern.

    For ``DecorationRole``, where a view draws a pixmap at its own size. Never a bare ``QColor``
    there — that draws the hue and nothing else
    (``requirements.an_identity_swatch_is_never_colour_alone``). A one-pixel rim of the colour
    keeps a pattern's surface-coloured strokes from merging with the row behind it. Drawn at the
    application's pixel ratio, so its strokes stay crisp on a high-density screen.
    """
    app = QApplication.instance()
    side = app.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize) if app else 16
    ratio = app.devicePixelRatio() if app else 1.0
    pixmap = QPixmap(round(side * ratio), round(side * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    theme = theme or current()
    color, _pattern = label_mark(index, theme)
    painter.fillRect(QRectF(0, 0, side, side), color)
    paint_label_mark(painter, QRectF(1, 1, side - 2, side - 2), index, theme=theme)
    painter.end()
    return pixmap


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
