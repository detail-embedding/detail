"""The clip popup's strips under the slider: what the animal is doing, frame by frame, on the
slider's own x — speed, mask area, distance to the nearest animal, where not to trust the
numbers, and which cluster each moment belongs to.

**Under the slider, on its travel.** Each strip maps a window frame to x through
``FrameScrubber.handle_travel``, the same measurement the slider's own marks use, so a dip in
speed sits exactly under the frame the handle points at; a strip that measured its own width
would be off by half a handle at each end. The name of each strip sits in the column the
elapsed time occupies above it, the value at the playhead in the column the end time does —
the reading a person wants, without hovering. The tooltip gives the unit and the value under
the pointer.

**Thin, in fine print.** The strips describe the picture above them; they are not the
picture. Each signal row is about a line of fine print tall (``sizing.small_font``) — about
half what they were — so five of them, the quality row and the cluster row together take
roughly a hundred pixels and the footage keeps the window. The reading is still there: the
value at the playhead at the right end, the value under the pointer on hover.

**Shape, not scale.** Each signal row is scaled to its own range in the window (speed from 0),
because the question is "when does it slow, shrink, close in", and a fixed scale either clips a
fast clip or flattens a slow one. The value readout carries the number.

**Where the data comes from** is ``ClipQueryPreview.clip_signals`` (the three signals and the
quality flags) and an opener-supplied lane (the clustering page's ``NodeView.lane``); the widget
reads nothing itself.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtWidgets import QWidget

from detail.core.visualize import MASK_TINT, track_color
from detail.gui.sizing import small_font

#: Row heights in pixels, at least: a signal row, the quality row, the cluster row. A row with
#: words in it (a signal's reading, a cluster's name) is never shorter than its fine print.
_SIGNAL_H = 15
_QUALITY_H = 9
_LANE_H = 13
_GAP = 2
#: The short names drawn left of each row, in the elapsed-time column.
_SHORT = {
    "speed": "speed",
    "mask area": "area",
    "to nearest animal": "nearest",
    "body length": "length",
}
#: Strips whose sign is the reading (a bend or a turn one way or the other): drawn about a
#: zero at the row's middle, so which way reads at a glance.
_SIGNED = {"bend", "turning"}


class ClipStrips(QWidget):
    """The rows under a clip player's slider. Fed by :meth:`set_signals` / :meth:`set_lane`;
    :meth:`set_clip_span` shades the clip and :meth:`set_frame` moves the playhead."""

    def __init__(self, scrubber, parent=None) -> None:
        super().__init__(parent)
        self._scrubber = scrubber
        self._signals = None
        self._lane: list[dict] = []
        self._clip: tuple[int, int] | None = None  # window-relative inclusive
        self._frame = 0
        self._message = ""
        self.setFont(small_font(self))
        self.setMouseTracking(True)
        self._resize_for(0)

    # -- inputs ---------------------------------------------------------------- #
    def set_signals(self, signals) -> None:
        """A ``ClipSignals`` for the window the scrubber spans, or ``None`` (cleared)."""
        self._signals = signals
        self._message = ""
        self._resize_for(0 if signals is None else len(signals.signals))
        self.update()

    def set_message(self, text: str) -> None:
        """Say why there are no strips (a tile of several animals, a read that failed)."""
        self._signals = None
        self._message = text
        self._resize_for(0)
        self.update()

    def set_lane(self, lane: list[dict]) -> None:
        """The cluster row: ``[{start_ns, end_ns, kind, node, name}]`` (``NodeView.lane``)."""
        self._lane = list(lane or [])
        self.update()

    def set_clip_span(self, span: tuple[int, int] | None) -> None:
        self._clip = span
        self.update()

    def set_frame(self, step: int) -> None:
        self._frame = int(step)
        self.update()

    def has_lane(self) -> bool:
        return bool(self._lane)

    def signals(self):
        return self._signals

    # -- geometry -------------------------------------------------------------- #
    def _heights(self) -> tuple[int, int, int]:
        """``(signal, quality, lane)`` row heights: the floors, raised to the fine print."""
        text = self.fontMetrics().height()
        return max(_SIGNAL_H, text), max(_QUALITY_H, text - 2), max(_LANE_H, text)

    def _resize_for(self, n_signals: int) -> None:
        signal_h, quality_h, lane_h = self._heights()
        rows = n_signals * (signal_h + _GAP) + quality_h + _GAP + lane_h
        self.setFixedHeight(rows + 2 if n_signals else lane_h + 4)

    def _rows(self) -> list[tuple[str, int, int]]:
        """``(kind, top, height)`` per row, top down; kind is a signal index or a name."""
        signal_h, quality_h, lane_h = self._heights()
        out: list[tuple[str, int, int]] = []
        y = 1
        n = 0 if self._signals is None else len(self._signals.signals)
        for i in range(n):
            out.append((f"signal:{i}", y, signal_h))
            y += signal_h + _GAP
        if n:
            out.append(("quality", y, quality_h))
            y += quality_h + _GAP
        out.append(("lane", y, lane_h))
        return out

    def _x(self, step: float, left: int, span: int, n: int) -> float:
        return left + (step / max(n - 1, 1)) * span

    def _travel(self) -> tuple[int, int]:
        left, span = self._scrubber.handle_travel()
        return left, max(min(span, self.width() - left), 1)

    def _n(self) -> int:
        return 0 if self._signals is None else len(self._signals.frames)

    # -- drawing --------------------------------------------------------------- #
    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        base = palette.color(QPalette.ColorRole.Base)
        ink = palette.color(QPalette.ColorRole.Text)
        label_ink = palette.color(QPalette.ColorRole.WindowText)
        left, span = self._travel()
        n = self._n()
        for kind, top, height in self._rows():
            painter.fillRect(left, top, span, height, base)
            self._shade_clip(painter, left, span, top, height, n)
            name = self._row_name(kind)
            if left >= self.fontMetrics().horizontalAdvance(name) + 6:
                painter.setPen(label_ink)
                painter.drawText(
                    QRect(0, top, left - 6, height),
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    name,
                )
            if kind.startswith("signal:"):
                self._draw_signal(painter, int(kind[7:]), left, span, top, height, ink)
            elif kind == "quality":
                self._draw_quality(painter, left, span, top, height)
            else:
                self._draw_lane(painter, left, span, top, height, ink)
        if n:
            x = self._x(min(self._frame, n - 1), left, span, n)
            painter.setPen(QPen(palette.color(QPalette.ColorRole.Highlight), 1.5))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        if self._message:
            painter.setPen(label_ink)
            painter.drawText(
                QRect(left, 0, span, self.height()),
                int(Qt.AlignmentFlag.AlignCenter),
                self._message,
            )

    def _row_name(self, kind: str) -> str:
        if kind == "quality":
            return "quality"
        if kind == "lane":
            return "clusters"
        strip = self._signals.signals[int(kind[7:])]
        return _SHORT.get(strip.name, strip.name)

    def _shade_clip(self, painter, left, span, top, height, n) -> None:
        if self._clip is None or not n:
            return
        x0 = self._x(self._clip[0], left, span, n)
        x1 = self._x(self._clip[1], left, span, n)
        tint = QColor(120, 190, 120, 55)
        painter.fillRect(QRect(int(x0), top, max(int(x1 - x0), 1), height), tint)

    def _draw_signal(self, painter, index, left, span, top, height, ink) -> None:
        strip = self._signals.signals[index]
        values = np.asarray(strip.values, dtype=np.float64)
        finite = np.isfinite(values)
        n = len(values)
        if not finite.any():
            return
        if strip.name in _SIGNED:
            reach = max(float(np.nanmax(np.abs(values))), 1e-9)
            low, high = -reach, reach
        else:
            low = 0.0 if strip.name == "speed" else float(np.nanmin(values))
            high = float(np.nanmax(values))
            if high - low < 1e-9:  # a flat series: drawn through the middle of the row
                low, high = low - 0.5, high + 0.5
        path = QPainterPath()
        pen_down = False
        for i in range(n):
            if not finite[i]:
                pen_down = False
                continue
            x = self._x(i, left, span, n)
            y = top + height - 1 - (values[i] - low) / (high - low) * (height - 2)
            if pen_down:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                pen_down = True
        painter.setPen(QPen(ink, 1.0))
        painter.drawPath(path)
        frame = min(self._frame, n - 1)
        reading = _fmt(values[frame]) if finite[frame] else "–"
        right = left + span + 6
        painter.drawText(
            QRect(right, top, max(self.width() - right, 0), height),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            reading,
        )

    def _draw_quality(self, painter, left, span, top, height) -> None:
        flags = np.asarray(self._signals.unreliable, dtype=bool)
        n = len(flags)
        bad = QColor(220, 110, 110)
        i = 0
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            x0 = self._x(i - 0.5, left, span, n)
            x1 = self._x(j + 0.5, left, span, n)
            painter.fillRect(QRect(int(x0), top, max(int(x1 - x0), 1), height), bad)
            i = j + 1

    def _draw_lane(self, painter, left, span, top, height, ink) -> None:
        if not self._lane or self._signals is None or not len(self._signals.t_ns):
            return
        t_ns = np.asarray(self._signals.t_ns, dtype=np.int64)
        n = len(t_ns)
        for segment in self._lane:
            a = int(np.searchsorted(t_ns, segment["start_ns"], side="left"))
            b = int(np.searchsorted(t_ns, segment["end_ns"], side="right")) - 1
            if b < a:
                continue
            x0 = self._x(a - 0.5, left, span, n)
            x1 = self._x(b + 0.5, left, span, n)
            x0, x1 = max(x0, left), min(x1, left + span)
            rect = QRect(int(x0), top, max(int(x1 - x0), 1), height)
            painter.fillRect(rect, _lane_colour(segment))
            text = _lane_text(segment)
            if painter.fontMetrics().horizontalAdvance(text) + 6 < rect.width():
                painter.setPen(QColor(30, 30, 30))
                painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)

    # -- reading under the pointer ---------------------------------------------- #
    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self.setToolTip(self.describe(event.position().x(), event.position().y()))
        super().mouseMoveEvent(event)

    def describe(self, x: float, y: float) -> str:
        """The tooltip at a point: which row, the value or the cluster there, and its unit."""
        n = self._n()
        left, span = self._travel()
        if not n:
            return self._message
        step = int(round((x - left) / span * (n - 1)))
        step = min(max(step, 0), n - 1)
        for kind, top, height in self._rows():
            if not top <= y < top + height:
                continue
            if kind.startswith("signal:"):
                strip = self._signals.signals[int(kind[7:])]
                value = strip.values[step]
                shown = _fmt(value) if np.isfinite(value) else "not tracked here"
                return (
                    f"{strip.name}: {shown} {strip.unit}. {self._signals.units_note}."
                )
            if kind == "quality":
                if self._signals.unreliable[step]:
                    return (
                        "Not trustworthy here: the animal is not tracked on this frame, or "
                        "its detection is unsure."
                    )
                return "Tracked, with a confident detection."
            t = int(self._signals.t_ns[step])
            for segment in self._lane:
                if segment["start_ns"] <= t <= segment["end_ns"]:
                    return f"cluster here: {_lane_text(segment)}"
            return "No cluster here (not tracked, or under the size floor)."
        return ""


def _fmt(value: float) -> str:
    return f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"


def _lane_text(segment: dict) -> str:
    if segment["kind"] == "this":
        return f"#{segment['node']} (this)"
    if segment["kind"] == "label":
        return segment["name"] or f"#{segment['node']}"
    return f"#{segment['node']}"


def _lane_colour(segment: dict) -> QColor:
    if segment["kind"] == "this":
        return QColor(*MASK_TINT)
    if segment["kind"] == "label":
        r, g, b = track_color(int(segment["node"]))
        return QColor(r, g, b, 150)
    return QColor(200, 200, 200)
