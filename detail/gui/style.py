"""The app's one Qt style: Fusion, with scrollbars that have no arrow buttons.

**One place sets how a scrollbar looks**, so every bar in the tool is the same widget drawn the
same way — the page, a table, the cluster tree's levels and the zoom overviews alike
(`gui/zoom.py`'s `OverviewBar` is a real `QScrollBar` for exactly this reason). The overviews used
to be hand-drawn by each widget that zoomed, and read as a different control from the scrollbar
beside them.

**No arrows.** A trackpad and a wheel scroll; the handle drags; the track pages. The two buttons
cost a handle-length of track at each end and were the least-used way to move.

A proxy style rather than a stylesheet: stylesheets are banned here (`requirements.py`,
`the_package_writes_no_stylesheets`), and a proxy keeps Fusion's own drawing — only *where* the
parts sit changes, so the handle, groove and both themes are Fusion's untouched.
"""

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtWidgets import QApplication, QProxyStyle, QStyle, QStyleOptionSlider

_SCROLLBAR = QStyle.ComplexControl.CC_ScrollBar
_ARROWS = (QStyle.SubControl.SC_ScrollBarAddLine, QStyle.SubControl.SC_ScrollBarSubLine)


class DetailStyle(QProxyStyle):
    """Fusion, laying a scrollbar's handle over its whole length with no arrow buttons."""

    def __init__(self) -> None:
        super().__init__("Fusion")

    def subControlRect(
        self, control, option, sub, widget=None
    ):  # noqa: N802 — Qt's name
        if control != _SCROLLBAR or not isinstance(option, QStyleOptionSlider):
            return super().subControlRect(control, option, sub, widget)
        if sub in _ARROWS:
            return QRect()
        rect = QRect(option.rect)
        if sub == QStyle.SubControl.SC_ScrollBarGroove:
            return rect
        horizontal = option.orientation == Qt.Orientation.Horizontal
        length = rect.width() if horizontal else rect.height()
        start, size = self._handle(option, length, widget)
        if sub == QStyle.SubControl.SC_ScrollBarSlider:
            span = (start, size)
        elif sub == QStyle.SubControl.SC_ScrollBarSubPage:
            span = (0, start)
        elif sub == QStyle.SubControl.SC_ScrollBarAddPage:
            span = (start + size, length - start - size)
        else:
            return super().subControlRect(control, option, sub, widget)
        offset, extent = span
        if horizontal:
            part = QRect(rect.x() + offset, rect.y(), extent, rect.height())
        else:
            part = QRect(rect.x(), rect.y() + offset, rect.width(), extent)
        return QStyle.visualRect(option.direction, rect, part)

    def _handle(
        self, option: QStyleOptionSlider, length: int, widget
    ) -> tuple[int, int]:
        """Where the handle starts and how long it is, over ``length`` — Qt's own proportion
        (page step against the range), with Fusion's minimum handle length."""
        span = option.maximum - option.minimum
        if span <= 0:
            return 0, length
        size = int(option.pageStep * length / (span + option.pageStep))
        floor = self.pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarSliderMin, option, widget
        )
        size = min(max(size, floor), length)
        start = QStyle.sliderPositionFromValue(
            option.minimum,
            option.maximum,
            option.sliderPosition,
            length - size,
            option.upsideDown,
        )
        return start, size


def apply_style(app: QApplication) -> None:
    """Give ``app`` the tool's style. The app's entry point and the test suite both call this,
    so a test renders what a user sees."""
    app.setStyle(DetailStyle())
