"""The clip window's strips under the slider: what the animal is doing, frame by frame, on the
slider's own x — the curves ``api.clip_curves`` read (speed, area, and with keypoints the body's
length, bend and turning), and where the track has no rows.

**Under the slider, on its travel.** Each strip maps a window frame to x through
``FrameScrubber.handle_travel``, the same measurement the slider's own marks use, so a dip in
speed sits exactly under the frame the handle points at; a strip that measured its own width
would be off by half a handle at each end. The name of each strip sits in the column the
elapsed time occupies above it, the value at the playhead in the column the end time does —
the reading a person wants, without hovering. The tooltip gives the unit and the value under
the pointer.

**Thin, in fine print.** The strips describe the picture above them; they are not the
picture. Each row is about a line of fine print tall (``sizing.small_font``), so five of them
and the untracked row take roughly a hundred pixels and the footage keeps the window.

**Shape, not scale.** Each curve is scaled to its own range in the window — from zero when the
curve says so (a speed, an area), about zero when it is signed (a bend, a turn) — because the
question is "when does it slow, shrink, turn", and a fixed scale either clips a fast clip or
flattens a slow one. The value readout carries the number.

**Data-driven.** A row is a ``Curve``: its name, unit and how to draw its range come from the
curve, so a signal added to ``api.clip_curves`` appears here with no change to this file. The
port this replaced hand-listed the names and which were signed — the recorded defect "a shared
surface summarizing a module-shaped fact by hardcoding one module's shape" — and was already
inconsistent about them. The widget reads nothing itself.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRect, Qt
from PyQt6.QtGui import QPainter, QPainterPath, QPalette, QPen
from PyQt6.QtWidgets import QWidget

from detail.gui import plot_theme, status_color
from detail.gui.sizing import small_font

#: Row heights in pixels, at least: a curve row, the untracked row. A row with words in it (a
#: curve's reading) is never shorter than its fine print.
_CURVE_H = 15
_UNTRACKED_H = 9
_GAP = 2
#: The clip's own frames are shaded on every row, in the chart theme's green, faint.
_CLIP_SHADE_ALPHA = 55


class ClipStrips(QWidget):
    """The rows under a clip player's slider. Fed by :meth:`set_curves`; :meth:`set_clip_span`
    shades the clip and :meth:`set_frame` moves the playhead."""

    def __init__(self, scrubber, parent=None) -> None:
        super().__init__(parent)
        self._scrubber = scrubber
        self._curves = None
        self._clip: tuple[int, int] | None = None  # window-relative inclusive
        self._frame = 0
        self._message = ""
        self.setFont(small_font(self))
        self.setMouseTracking(True)
        self._resize_for(0)

    # -- inputs ---------------------------------------------------------------- #
    def set_curves(self, curves) -> None:
        """A ``ClipCurves`` for the window the scrubber spans, or ``None`` (cleared). An
        empty one (no sidecar) shows its note where the rows would be."""
        self._curves = curves
        self._message = "" if curves is None or curves.curves else curves.note
        self._resize_for(0 if curves is None else len(curves.curves))
        self.update()

    def set_message(self, text: str) -> None:
        """Say why there are no strips (a read that failed)."""
        self._curves = None
        self._message = text
        self._resize_for(0)
        self.update()

    def set_clip_span(self, span: tuple[int, int] | None) -> None:
        self._clip = span
        self.update()

    def set_frame(self, step: int) -> None:
        self._frame = int(step)
        self.update()

    def curves(self):
        return self._curves

    def names(self) -> list[str]:
        """The rows, top down, by the curve names they draw."""
        return [] if self._curves is None else list(self._curves.names)

    # -- geometry -------------------------------------------------------------- #
    def _heights(self) -> tuple[int, int]:
        text = self.fontMetrics().height()
        return max(_CURVE_H, text), max(_UNTRACKED_H, text - 2)

    def _resize_for(self, n_curves: int) -> None:
        curve_h, untracked_h = self._heights()
        if not n_curves:
            self.setFixedHeight(untracked_h + 4)
            return
        self.setFixedHeight(n_curves * (curve_h + _GAP) + untracked_h + 2)

    def _rows(self) -> list[tuple[str, int, int]]:
        """``(kind, top, height)`` per row, top down: ``curve:<i>`` rows, then ``untracked``."""
        curve_h, untracked_h = self._heights()
        out: list[tuple[str, int, int]] = []
        y = 1
        n = 0 if self._curves is None else len(self._curves.curves)
        for i in range(n):
            out.append((f"curve:{i}", y, curve_h))
            y += curve_h + _GAP
        if n:
            out.append(("untracked", y, untracked_h))
        return out

    def _window(self) -> tuple[int, int]:
        """The inclusive window frames the bar spans, from the curves (nothing without them)."""
        return (0, 0) if self._curves is None else self._curves.window

    def _x(self, frame: float, left: int, span: int) -> float:
        first, last = self._window()
        return left + (float(frame) - first) / max(last - first, 1) * span

    def _travel(self) -> tuple[int, int]:
        left, span = self._scrubber.handle_travel()
        return left, max(min(span, self.width() - left), 1)

    # -- drawing --------------------------------------------------------------- #
    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        base = palette.color(QPalette.ColorRole.Base)
        ink = palette.color(QPalette.ColorRole.Text)
        label_ink = palette.color(QPalette.ColorRole.WindowText)
        left, span = self._travel()
        for kind, top, height in self._rows():
            painter.fillRect(left, top, span, height, base)
            self._shade_clip(painter, left, span, top, height)
            name = self._row_name(kind)
            if left >= self.fontMetrics().horizontalAdvance(name) + 6:
                painter.setPen(label_ink)
                painter.drawText(
                    QRect(0, top, left - 6, height),
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    name,
                )
            if kind.startswith("curve:"):
                self._draw_curve(painter, int(kind[6:]), left, span, top, height, ink)
            else:
                self._draw_untracked(painter, left, span, top, height)
        if self._curves is not None and self._curves.curves:
            first, _last = self._window()
            x = self._x(first + self._frame, left, span)
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
        if kind == "untracked":
            return "untracked"
        return self._curves.curves[int(kind[6:])].name

    def _shade_clip(self, painter, left, span, top, height) -> None:
        if self._clip is None or self._curves is None:
            return
        first, _last = self._window()
        x0 = self._x(first + self._clip[0], left, span)
        x1 = self._x(first + self._clip[1], left, span)
        tint = plot_theme.qcolor(plot_theme.current().hue("green"), _CLIP_SHADE_ALPHA)
        painter.fillRect(QRect(int(x0), top, max(int(x1 - x0), 1), height), tint)

    def _draw_curve(self, painter, index, left, span, top, height, ink) -> None:
        curve = self._curves.curves[index]
        values = np.asarray(curve.values, dtype=np.float64)
        frames = np.asarray(self._curves.frames, dtype=np.int64)
        finite = np.isfinite(values)
        if not finite.any():
            return
        if curve.signed:
            reach = max(float(np.nanmax(np.abs(values))), 1e-9)
            low, high = -reach, reach
        else:
            low = 0.0 if curve.floor_zero else float(np.nanmin(values))
            high = float(np.nanmax(values))
            if high - low < 1e-9:  # a flat series: drawn through the middle of the row
                low, high = low - 0.5, high + 0.5
        path = QPainterPath()
        pen_down = False
        for i in range(len(values)):
            if not finite[i]:
                pen_down = False
                continue
            x = self._x(frames[i], left, span)
            y = top + height - 1 - (values[i] - low) / (high - low) * (height - 2)
            if pen_down:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                pen_down = True
        painter.setPen(QPen(ink, 1.0))
        painter.drawPath(path)
        row = self._row_at_playhead()
        reading = _fmt(values[row]) if row is not None and finite[row] else "–"
        right = left + span + 6
        painter.drawText(
            QRect(right, top, max(self.width() - right, 0), height),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            reading,
        )

    def _draw_untracked(self, painter, left, span, top, height) -> None:
        for a, b in self._curves.untracked:
            x0 = self._x(a - 0.5, left, span)
            x1 = self._x(b + 0.5, left, span)
            painter.fillRect(
                QRect(int(x0), top, max(int(x1 - x0), 1), height), status_color.MUTED
            )

    def _row_at_playhead(self) -> int | None:
        """The curve row at or just before the playhead's frame, or ``None`` before the first."""
        if self._curves is None or not len(self._curves.frames):
            return None
        first, _last = self._window()
        frames = np.asarray(self._curves.frames, dtype=np.int64)
        at = int(np.searchsorted(frames, first + self._frame, side="right")) - 1
        return at if at >= 0 else None

    # -- reading under the pointer ---------------------------------------------- #
    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self.setToolTip(self.describe(event.position().x(), event.position().y()))
        super().mouseMoveEvent(event)

    def describe(self, x: float, y: float) -> str:
        """The tooltip at a point: which row, the value there, and its unit."""
        if self._curves is None or not self._curves.curves:
            return self._message
        left, span = self._travel()
        first, last = self._window()
        frame = int(round(first + (x - left) / span * max(last - first, 1)))
        frame = min(max(frame, first), last)
        frames = np.asarray(self._curves.frames, dtype=np.int64)
        for kind, top, height in self._rows():
            if not top <= y < top + height:
                continue
            if kind == "untracked":
                if any(a <= frame <= b for a, b in self._curves.untracked):
                    return "Not tracked here: the animal has no embedded row on this frame."
                return "Tracked."
            curve = self._curves.curves[int(kind[6:])]
            at = int(np.searchsorted(frames, frame, side="right")) - 1
            value = curve.values[at] if at >= 0 else np.nan
            shown = _fmt(value) if np.isfinite(value) else "not tracked here"
            return f"{curve.label}: {shown} {curve.unit}. {self._curves.note}."
        return ""


def _fmt(value: float) -> str:
    return f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"
