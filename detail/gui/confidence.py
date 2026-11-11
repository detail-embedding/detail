"""The confidence-view control every detection result carries.

One widget, used by every surface that shows detections a model produced wide open: the
Quick segment result views and the prompt-test grid today, ``DESIGN_PLAN`` §13's tracking
preview next. A surface that hand-rolls its own slider beside a count label is the bug —
the three parts below are exactly the ones that were about to be written three times.

- The **slider** comes from ``configs/display.ConfidenceView`` through ``field_slider``, so
  its range, step and help are the model's and not a number typed into a widget.
- The **caption** is the count at the current threshold. Every count a run stores was
  counted over all proposals, so a stored number cannot answer "how many at 0.7?" — the
  view has to recount, and a view that forgets to shows a confident wrong total.
- The **debounce** lives in ``ThresholdSlider``, so a drag re-renders once rather than per
  pixel.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from detail.configs.display import ConfidenceView
from detail.gui._pydantic_form import field_meta, field_slider
from detail.gui.status_color import MUTED, set_label_color

HINT = "slide to change"


class ConfidenceRow(QFrame):
    """A named confidence slider plus a live "what this keeps" caption.

    Deliberately **not** a bare horizontal slider. Every surface that shows this also
    carries a ``FrameScrubber`` — itself a horizontal slider — and two unlabelled tracks
    stacked in one view read as the same control. Three things separate them: a titled
    box (``QFrame`` panel, no stylesheet — GUI style), a heading that names what is being
    thresholded rather than describing the effect, and a two-line shape against the
    scrubber's single row.
    """

    threshold_changed = pyqtSignal(float)

    def __init__(
        self,
        initial: float | None = None,
        parent=None,
        *,
        heading: str | None = None,
        help_text: str | None = None,
    ) -> None:
        """``heading`` names what is being thresholded on *this* surface.

        The default is the detection-confidence field's label, which is right for the surfaces
        that filter detections and wrong everywhere else — §20's calibration cutoff is a
        behavior's match confidence, not a detector's. The heading naming the subject rather
        than the effect is this widget's own rule (see the class docstring), so it has to be
        the caller's to set; the range, step and slider behaviour stay shared.
        """
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        meta = field_meta(ConfidenceView, "min_conf")
        self._slider = field_slider(ConfidenceView, "min_conf")
        self._slider.set_value(
            ConfidenceView().min_conf if initial is None else float(initial)
        )
        self._slider.threshold_changed.connect(self.threshold_changed)

        self._caption = QLabel("")
        set_label_color(self._caption, MUTED)

        column = QVBoxLayout(self)
        self._heading = QLabel(f"{heading or meta['label']} ({HINT})")
        self._heading.setToolTip(help_text or meta["help"])
        column.addWidget(self._heading)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._slider, 1)
        row.addWidget(self._caption)
        column.addLayout(row)

    def value(self) -> float:
        return self._slider.value()

    def set_value(self, value: float) -> None:
        """Show ``value`` without emitting — for seeding from a caller."""
        self._slider.set_value(value)

    def set_caption(self, text: str) -> None:
        """The count this threshold keeps, e.g. ``showing 3 of 214``."""
        self._caption.setText(text)


def showing_caption(shown: int, total: int) -> str:
    """The standard caption text, so every surface words it the same way."""
    return f"showing {shown} of {total}"
