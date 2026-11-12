"""The match key at the top of the filter dock, and the label swatches above the tree.

:class:`MatchKey` — ``Clips that match`` over ``0% [ramp] 100%``: the ramp on a fixed 0–100%
axis, the selected cluster's share as a tall bar with a triangle over it and ``selected 2.3%``
above, and the whole run's share as a thinner line with ``run 23%`` under it. It sits at the top
of the filter dock, beside what sets it, always: greyed with no marks while nothing is filtered,
in the same space, so the fields under it never move (review round 8). It once shared the row
above the tree, where it crowded the label swatches.
:class:`TreeKeyRow` — the label swatches, with the time under each label.

**The axis never stretches.** The ramp is always 0 to 100%; a cluster at 52% sits at 52% of the
ramp's width whatever the other clusters do. Stretched to the clusters' own min–max, an even
spread (every cluster 43–55%) would fill the whole ramp and read as a strong pattern; on the fixed
axis the selected cluster's bar sits beside the run's line, which is the honest picture. The
colours come from :func:`plot_theme.match_colour`, the same function that shades the tree's blocks.

It once also drew a tick per cluster above the ramp. Hundreds of clusters made that a barcode
nobody could read, and the tree's own shading already says where each cluster sits, so review
round 3 took it out.

**The effect mode.** While a difference screen is the only filter on (§22, *The difference
screen*), the key reads the screen's statistic instead of a share: the theme's diverging scale,
less at the left and more at the right, on a **fixed** axis too — ρ from −1 to +1 for a trend,
÷20 to ×20 for a log ratio — with the floor marked at both ends of the neutral middle and the
selected cluster's effect as the tall bar. Inside the floor is neutral grey on the tree and on the
key: not shown to differ. :func:`effect_position` is the one map from an effect to the axis, for
the key and for the tree's blocks (`cluster_filters.effect_shades`), so they cannot disagree.

The widgets take plain numbers: nothing here reads the node cache.
"""

from __future__ import annotations

import math

from collections.abc import Sequence
import html

from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPalette, QPen, QPolygonF
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from detail.core.progress import format_duration
from detail.gui import plot_theme

#: The key's caption, over the ramp.
CAPTION = "Clips that match"
#: The ramp's width. It never grows past it: it is a scale, and a scale that grows with the window
#: would make the same share sit at a different place on every screen. It does shrink, to
#: ``_RAMP_MIN_PX``, in a filter dock dragged narrow.
RAMP_PX = 230
#: The narrowest the ramp is squeezed to.
_RAMP_MIN_PX = 120
#: Inset of the bar inside the ramp widget, so a mark at 0% or 100% is not clipped.
_INSET = 4
#: The selected cluster's label row, its triangle, the bar, and the run's label row, top down.
_LABEL_H = 13
_TRIANGLE_TOP = _LABEL_H + 2
_TRIANGLE_H = 5
_BAR_TOP = _TRIANGLE_TOP + _TRIANGLE_H + 1
_BAR_H = 10
_RUN_TOP = _BAR_TOP + _BAR_H + 3
_RAMP_H = _RUN_TOP + _LABEL_H
#: How faint the ramp is drawn while no filter is on.
_INACTIVE_OPACITY = 0.35
#: The key's tooltip while no filter is on: why it is greyed.
#: The ramp's tooltip while a filter is on.
RAMP_TEXT = (
    "The tall bar is the selected cluster's share; the thin line is the whole run's."
)
INACTIVE_TEXT = (
    "Pick values in the filters below to shade the tree by how many clips match."
)
#: The effect mode's ramp tooltip.
EFFECT_TEXT = (
    "The tall bar is the selected cluster's effect; the ticks are the floor, and a cluster "
    "inside it is grey: not shown to differ."
)
#: How far the log-ratio axis reaches either way: a fixed 20× (ln 20), so the same effect sits
#: at the same place on every screen. Beyond it a mark sits at the end.
LOG_RATIO_EXTENT = math.log(20.0)


def effect_extent(design: str) -> float:
    """How far the effect axis reaches from its middle: 1 for a trend's ρ, ln 20 otherwise."""
    return 1.0 if design == "trend" else LOG_RATIO_EXTENT


def one_sided(design: str) -> bool:
    """Whether the effect axis is a size with no direction (§22). A profile's effect is the
    largest difference between any two of its levels: its sign belongs to that pair, not to
    the measure, so the axis runs from 1× up and the tree takes one ramp by size."""
    return design == "profile"


def effect_position(
    effect, clears: bool, extent: float, *, one_sided: bool = False
) -> float:
    """Where an effect sits on the −1..+1 axis: ``NaN`` for an undefined effect (``None`` or
    NaN), ``0`` for one that did not clear the floor (neutral, not shown to differ), else the
    signed effect over ``extent``, clamped to the ends. ``one_sided`` places its size on the
    0..+1 half instead, for a profile."""
    if effect is None or effect != effect:
        return float("nan")
    if not clears:
        return 0.0
    if one_sided:
        return min(abs(float(effect)) / float(extent), 1.0)
    return min(max(float(effect) / float(extent), -1.0), 1.0)


def effect_axis_labels(design: str) -> tuple[str, str]:
    """The axis ends: ``("ρ −1", "ρ +1")`` for a trend, ``("1×", "×20")`` for a
    profile's size, ``("÷20", "×20")`` for a log ratio."""
    if design == "trend":
        return ("ρ −1", "ρ +1")
    if one_sided(design):
        return ("1×", "×20")
    return ("÷20", "×20")


def effect_value_text(effect: float | None, design: str) -> str:
    """``12.4×``, ``÷2.1`` or ``ρ −0.74`` — an effect as a person reads it. A profile's is a
    size, ``12.4×`` whichever way its pair runs."""
    if effect is None or effect != effect:
        return "undefined"
    if design == "trend":
        return f"ρ {effect:+.2f}"
    factor = math.exp(abs(float(effect)))
    if one_sided(design):
        return f"{factor:.1f}×"
    return f"{factor:.1f}×" if effect >= 0 else f"÷{factor:.1f}"


def share_text(share: float) -> str:
    """``2.3%`` under 10%, where a tenth of a percent is most of the difference; ``23%`` above."""
    percent = 100 * float(share)
    return f"{percent:.1f}%" if percent < 10 else f"{round(percent)}%"


class _Ramp(QWidget):
    """The painted part: the gradient bar, the selected cluster's bar, the run's line."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # RAMP_PX wide when there is room, narrower in a dock dragged narrow: a fixed 230 put
        # "100%" over the bar's end at the dock's 240px floor.
        self.setFixedHeight(_RAMP_H)
        self.setMinimumWidth(_RAMP_MIN_PX)
        self.setMaximumWidth(RAMP_PX)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        #: False while no filter is on: the ramp is drawn faint, with no marks.
        self.active = True
        #: The effect mode: the diverging scale over −1..+1 with the floor's ticks.
        self.diverging = False
        #: The effect mode's one-sided form: a size over 0..+1 on the scale's upper half, one
        #: floor tick — a profile's axis.
        self.half = False
        self._floor: float | None = None
        self._selected_text: str | None = None
        self._selected: float | None = None
        self._run: float | None = None
        self.setToolTip(RAMP_TEXT)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        return QSize(RAMP_PX, _RAMP_H)

    def set_marks(self, selected: float | None, run: float | None) -> None:
        self._selected = (
            None if selected is None else min(max(float(selected), 0.0), 1.0)
        )
        self._run = None if run is None else min(max(float(run), 0.0), 1.0)
        self.update()

    def set_effect_marks(
        self, selected: float | None, selected_text: str | None, floor: float | None
    ) -> None:
        """The effect mode's marks: the selected cluster at ``selected`` (−1..+1) with its
        words, and the floor at ±``floor`` (0..1 of the half axis)."""
        self._selected = (
            None
            if selected is None or selected != selected
            else min(max(float(selected), -1.0), 1.0)
        )
        self._selected_text = selected_text
        self._floor = None if floor is None else min(max(float(floor), 0.0), 1.0)
        self._run = None
        self.update()

    def x_for(self, share: float) -> float:
        """Where ``share`` sits on the bar — a fixed map from 0..1 (or −1..+1 in the effect
        mode), whatever the data."""
        span = self.width() - 2 * _INSET
        if self.diverging and not self.half:
            return _INSET + span * (min(max(float(share), -1.0), 1.0) + 1.0) / 2.0
        return _INSET + span * min(max(float(share), 0.0), 1.0)

    def labels(self) -> list[str]:
        """The marks' words, selected first — for tests and the key's own text."""
        out = []
        if self._selected is not None:
            if self.diverging:
                out.append(f"selected {self._selected_text or ''}".rstrip())
            else:
                out.append(f"selected {share_text(self._selected)}")
        if self._run is not None:
            out.append(f"run {share_text(self._run)}")
        if self.diverging and self._floor is not None:
            out.append("floor")
        return out

    def label_rect(self, text: str, share: float, top: float) -> QRectF:
        """``text`` centred over ``share``, slid inward at either end so it is never clipped."""
        width = self.fontMetrics().horizontalAdvance(text) + 2
        left = min(max(self.x_for(share) - width / 2, 0.0), self.width() - width)
        return QRectF(left, top, width, _LABEL_H)

    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        theme = plot_theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.active:
            # Greyed with the labels beside it: a scale with nothing on it yet.
            painter.setOpacity(_INACTIVE_OPACITY)
        bar = QRectF(_INSET, _BAR_TOP, self.width() - 2 * _INSET, _BAR_H)
        gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
        if self.diverging and self.half:
            # The upper half of the diverging scale, neutral to the "more" pole: the same
            # colours `plot_theme.effect_colour` gives a block at 0..+1.
            stops = theme.diverging[len(theme.diverging) // 2 :]
        elif self.diverging:
            stops = theme.diverging
        else:
            stops = theme.sequential
        for i, stop in enumerate(stops):
            gradient.setColorAt(i / (len(stops) - 1), QColor(stop))
        painter.fillRect(bar, gradient)
        painter.setPen(QPen(QColor(theme.axis), 0.75))
        painter.drawRect(bar)
        ink, muted = QColor(theme.text), QColor(theme.text_secondary)
        centre = Qt.AlignmentFlag.AlignCenter
        if self.diverging and self._floor is not None:
            # The floor, both ways from the middle: inside it a cluster is grey on the tree.
            painter.setPen(QPen(muted, 1.5))
            sides = (self._floor,) if self.half else (-self._floor, self._floor)
            for side in sides:
                x = self.x_for(side)
                painter.drawLine(QPointF(x, _BAR_TOP - 2), QPointF(x, _RUN_TOP))
            text = "floor"
            painter.drawText(self.label_rect(text, self._floor, _RUN_TOP), centre, text)
        if self._run is not None:
            x = self.x_for(self._run)
            painter.setPen(QPen(muted, 1.5))
            painter.drawLine(QPointF(x, _BAR_TOP - 2), QPointF(x, _RUN_TOP))
            text = f"run {share_text(self._run)}"
            painter.drawText(self.label_rect(text, self._run, _RUN_TOP), centre, text)
        # The selected cluster last, so the run's line cannot hide it at the same share.
        if self._selected is not None:
            x = self.x_for(self._selected)
            painter.fillRect(QRectF(x - 1.5, _BAR_TOP - 2, 3, _BAR_H + 4), ink)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(ink)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 4, _TRIANGLE_TOP),
                        QPointF(x + 4, _TRIANGLE_TOP),
                        QPointF(x, _TRIANGLE_TOP + _TRIANGLE_H),
                    ]
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(ink)
            text = (
                f"selected {self._selected_text or ''}".rstrip()
                if self.diverging
                else f"selected {share_text(self._selected)}"
            )
            painter.drawText(self.label_rect(text, self._selected, 0), centre, text)
        painter.end()


class MatchKey(QWidget):
    """``Clips that match`` over ``0% [ramp] 100%``, always the same height: with its marks
    while a filter is on, greyed with none while no filter is (user, review round 8 — round 7
    hid it, and the column jumped when the first filter went on)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        self._caption = QLabel(CAPTION)
        column.addWidget(self._caption)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._low = QLabel("0%")
        self.ramp = _Ramp()
        self._high = QLabel("100%")
        for widget in (self._low, self.ramp, self._high):
            row.addWidget(widget)
        row.addStretch(1)
        column.addLayout(row)
        self.set_filter(False)

    def is_active(self) -> bool:
        """Whether a filter is on — False while the key is greyed with no marks."""
        return self.ramp.active

    def set_filter(
        self,
        active: bool,
        *,
        selected_share: float | None = None,
        run_share: float | None = None,
    ) -> None:
        """Show the marks while a filter is active; grey the key, with no marks, while none is.

        Args:
            active: Whether any filter is on.
            selected_share: The selected cluster's matching share (0..1), or ``None`` when no
                cluster is selected or it has nothing to count.
            run_share: The whole run's matching share (0..1), or ``None``.
        """
        # Greyed by colour, not by disabling: a disabled label cannot be selected and copied.
        self.ramp.active = bool(active)
        self.ramp.diverging = False
        self.ramp.half = False
        self._caption.setText(CAPTION)
        self._low.setText("0%")
        self._high.setText("100%")
        self.setToolTip("" if active else INACTIVE_TEXT)
        self.ramp.setToolTip(RAMP_TEXT if active else INACTIVE_TEXT)
        self._grey_labels()
        self.ramp.set_marks(
            selected_share if active else None, run_share if active else None
        )

    def set_effect(
        self,
        caption: str,
        design: str,
        *,
        floor: float | None,
        selected_effect: float | None,
    ) -> None:
        """The effect mode: ``caption`` over the diverging axis for ``design``, the floor's
        ticks at ±``floor`` (in the effect's own unit) and the selected cluster's effect.

        Args:
            caption: What the axis is, e.g. ``"Less ↔ more under drug"``.
            design: ``"trend"`` for a ρ axis, ``"profile"`` for a one-sided size axis, else a
                log-ratio axis.
            floor: The screen's floor in the effect's unit, or ``None`` when it has none.
            selected_effect: The selected cluster's effect, ``None`` for none or undefined.
        """
        extent = effect_extent(design)
        half = one_sided(design)
        self.ramp.active = True
        self.ramp.diverging = True
        self.ramp.half = half
        self._caption.setText(caption)
        low, high = effect_axis_labels(design)
        self._low.setText(low)
        self._high.setText(high)
        self.setToolTip("")
        self.ramp.setToolTip(EFFECT_TEXT)
        self._grey_labels()
        position = (
            None
            if selected_effect is None or selected_effect != selected_effect
            else effect_position(selected_effect, True, extent, one_sided=half)
        )
        self.ramp.set_effect_marks(
            position,
            None if position is None else effect_value_text(selected_effect, design),
            None if floor is None else float(floor) / extent,
        )

    def _grey_labels(self) -> None:
        palette = QPalette(self.palette())
        if not self.ramp.active:
            palette.setColor(
                QPalette.ColorRole.WindowText,
                palette.color(
                    QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText
                ),
            )
        for label in (self._caption, self._low, self._high):
            label.setPalette(palette)

    def changeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        # A theme switch reaches this widget; the labels hold a copy, so re-derive it.
        if event.type() == QEvent.Type.PaletteChange:
            self._grey_labels()
        super().changeEvent(event)

    def is_effect(self) -> bool:
        """Whether the key reads an effect rather than a share."""
        return self.ramp.diverging

    def text(self) -> str:
        """What the key says, for tests and the page's own checks: the scale, then the marks
        (none while greyed)."""
        scale = f"{self._low.text()} … {self._high.text()}"
        return " | ".join([self._caption.text(), scale, *self.ramp.labels()])


class LabelSwatches(QWidget):
    """The labels in use, each a colour swatch, its name and the time it covers."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(12)
        self._labels: list[QLabel] = []

    def set_labels(self, rows: Sequence[tuple[str, QColor | str, float]]) -> None:
        """``(name, colour, seconds)`` per label, in the order to show them."""
        for label in self._labels:
            label.setParent(None)
            label.deleteLater()
        self._labels = []
        for name, colour, seconds in rows:
            hex_colour = QColor(colour).name()
            label = QLabel(
                f'<span style="color:{hex_colour}">&#9632;</span> '
                f"{html.escape(name)} "
                f"<span>{html.escape(format_duration(seconds))}</span>"
            )
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setToolTip(f"{name}: {format_duration(seconds)} labelled")
            self._row.addWidget(label)
            self._labels.append(label)

    def texts(self) -> list[str]:
        """What each swatch says, as plain text — for tests and for the page's own checks."""
        from PyQt6.QtGui import QTextDocument

        out = []
        for label in self._labels:
            doc = QTextDocument()
            doc.setHtml(label.text())
            out.append(doc.toPlainText())
        return out


class TreeKeyRow(QWidget):
    """The row above the tree: the label swatches. (The match key is in the filter dock.)"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.swatches = LabelSwatches()
        row.addWidget(self.swatches)
        row.addStretch(1)
        self.setMaximumWidth(RAMP_PX)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
