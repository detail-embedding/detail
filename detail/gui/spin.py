"""The number boxes every form uses. Build one of these, never a bare Q*SpinBox.

Qt pads a spin box's text to *exactly* ``decimals`` places and its validator rejects any
candidate carrying more, so a 4-decimal confidence box renders ``0.5000`` and then swallows
every digit you type into it — the field is permanently "full" and the only way in is to
delete first. These trim the display instead (``0.5``), leaving the precision cap as room to
type. Two smaller cures ride along: clicking in selects the value, so typing replaces it,
and an unfocused box ignores the wheel, so scrolling a long dialog cannot nudge a threshold
under the pointer.

A float box's ``kind`` fixes its decimals and step in one place. ``COUNT`` names the integer
box rather than a float kind; it is here so callers (and ``_pydantic_form.field_widget``)
can talk about all four in one vocabulary.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QSlider,
    QSpinBox,
    QWidget,
)

FRACTION = "fraction"  # a 0-1 ratio: confidence, IoU, a split
SECONDS = "seconds"  # a duration
COUNT = "count"  # a whole number of things -> IntSpin
PLAIN = "plain"  # any other real number

# kind -> (decimals, single step). Decimals is a *cap* on typed precision, not a display
# width, because the display trims. SECONDS carries three because this project's durations
# routinely run shorter than a frame -- one frame at 59 Hz is 0.017 s -- and a two-decimal cap
# cannot express one, so it rounds a sub-frame default on the way into the form (it was
# rounding `abd_smooth_half_s`, 0.034, to 0.03).
_FLOAT_KINDS: dict[str, tuple[int, float]] = {
    FRACTION: (4, 0.05),
    SECONDS: (3, 0.1),
    PLAIN: (3, 0.1),
}


def float_format(kind: str) -> tuple[int, float]:
    """``(decimals, single step)`` for a float kind — ``COUNT`` steps by 1 as an int box."""
    return (0, 1.0) if kind == COUNT else _FLOAT_KINDS[kind]


class _Typeable:
    """Focus and wheel behaviour shared by both boxes.

    Mixed in ahead of the Qt class so ``super()`` reaches the spin box implementation.
    """

    def _init_typeable(self) -> None:
        # StrongFocus (not the spin-box default WheelFocus) so a wheel roll cannot take
        # focus and then act on it.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Owned by the box, so a form closed between the focus and the timeout takes the
        # pending select-all with it rather than calling into a deleted widget.
        self._select_all_timer = QTimer(self)
        self._select_all_timer.setSingleShot(True)
        self._select_all_timer.timeout.connect(self.selectAll)

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        # The mouse press that gave us focus places the cursor *after* this event, undoing
        # a selection made now — so select once the press has been handled.
        self._select_all_timer.start(0)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class IntSpin(_Typeable, QSpinBox):
    """A whole-number box."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._init_typeable()


class FloatSpin(_Typeable, QDoubleSpinBox):
    """A real-number box that shows ``0.5``, not ``0.5000``."""

    def __init__(self, kind: str = PLAIN, parent=None) -> None:
        super().__init__(parent)
        self._init_typeable()
        decimals, step = _FLOAT_KINDS[kind]
        self.setDecimals(decimals)
        self.setSingleStep(step)

    def textFromValue(self, value: float) -> str:
        """Qt's own text with the padding zeros taken off the fraction.

        Trims Qt's output rather than formatting from scratch, so the locale's decimal
        point and group separators keep working. At least one decimal survives, so the
        box still reads as a decimal (``1.0``, not ``1``).
        """
        text = super().textFromValue(value)
        point = self.locale().decimalPoint()
        if point not in text:
            return text
        whole, _, fraction = text.partition(point)
        return f"{whole}{point}{fraction.rstrip('0') or '0'}"

    def sizeHint(self):
        return self._widen(super().sizeHint())

    def minimumSizeHint(self):
        return self._widen(super().minimumSizeHint())

    def _widen(self, hint):
        """Reserve room for a full-precision entry.

        Qt sizes a spin box from its *displayed* min/max text, which trimming shortens — a
        0-1 box would come out 49 px against the untrimmed 72 px, too narrow to see
        ``0.9999`` while typing it. Add back the difference between the trimmed and
        untrimmed extremes; the arrow and frame allowance rides along in the base hint, so
        nothing here depends on the style's pixel metrics.
        """
        metrics = self.fontMetrics()

        def widest(render) -> int:
            return max(
                metrics.horizontalAdvance(render(value))
                for value in (self.minimum(), self.maximum())
            )

        padding = widest(lambda v: QDoubleSpinBox.textFromValue(self, v)) - widest(
            self.textFromValue
        )
        hint.setWidth(hint.width() + max(0, padding))
        return hint


class ThresholdSlider(QWidget):
    """A slider tied to a typeable number box, for filtering what a result *shows*.

    Generic over any bounded numeric field — nothing here knows what is being thresholded.
    Its first two users filter detections by confidence; ``DESIGN_PLAN`` §13's tracking
    preview is the third, and it should reuse this rather than grow its own.

    Two behaviours belong to the widget, not to each consumer, because a consumer that
    re-derives them gets them subtly wrong:

    - **Debounced.** Dragging a slider emits a value per pixel. Each emit re-renders a
      frame, so an undebounced drag queues hundreds of renders and the picture lags the
      handle. Changes coalesce into one ``threshold_changed`` after ``debounce_ms`` of
      quiet; releasing the handle emits immediately rather than waiting out the timer.
    - **Two-way, without a loop.** The slider is integral and the value is not, so the two
      are kept in step through ``step``-sized notches with signals blocked on the way back.
    """

    threshold_changed = pyqtSignal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        *,
        step: float,
        kind: str = FRACTION,
        debounce_ms: int = 120,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        if maximum <= minimum:
            raise ValueError(f"maximum ({maximum}) must exceed minimum ({minimum})")
        self._minimum = float(minimum)
        self._step = float(step)

        self._spin = FloatSpin(kind)
        self._spin.setRange(float(minimum), float(maximum))
        self._spin.setSingleStep(float(step))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._notch(maximum))
        # A wheel roll over an unfocused slider must not move it, for the same reason the
        # spin boxes above ignore one: scrolling a dialog would nudge whatever sits under
        # the pointer.
        self._slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._slider.wheelEvent = self._slider_wheel_event

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._slider, 1)
        row.addWidget(self._spin)

        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(debounce_ms)
        self._emit_timer.timeout.connect(self._emit)

        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderReleased.connect(self._emit_now)
        self._spin.valueChanged.connect(self._on_spin)
        self.set_value(minimum)

    # -- value --------------------------------------------------------------- #
    def value(self) -> float:
        return self._spin.value()

    def set_value(self, value: float) -> None:
        """Show ``value`` without emitting — for seeding the widget from a caller."""
        self._spin.blockSignals(True)
        self._spin.setValue(float(value))
        self._spin.blockSignals(False)
        self._sync_slider_to_spin()

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt's own name
        super().setToolTip(text)
        self._spin.setToolTip(text)
        self._slider.setToolTip(text)

    # -- internals ----------------------------------------------------------- #
    def _notch(self, value: float) -> int:
        return int(round((float(value) - self._minimum) / self._step))

    def _slider_wheel_event(self, event) -> None:
        if self._slider.hasFocus():
            QSlider.wheelEvent(self._slider, event)
        else:
            event.ignore()

    def _sync_slider_to_spin(self) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._notch(self._spin.value()))
        self._slider.blockSignals(False)

    def _on_slider(self, notch: int) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(self._minimum + notch * self._step)
        self._spin.blockSignals(False)
        self._emit_timer.start()

    def _on_spin(self, _value: float) -> None:
        self._sync_slider_to_spin()
        self._emit_timer.start()

    def _emit(self) -> None:
        self.threshold_changed.emit(self.value())

    def _emit_now(self) -> None:
        self._emit_timer.stop()
        self._emit()
