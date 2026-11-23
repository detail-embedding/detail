"""The app's style lays a scrollbar out with no arrow buttons, and its handle covers the bar.

`gui/style.py` keeps Fusion's drawing and changes only where a scrollbar's parts sit, so the
checks are on those places — through the style the widget actually has, which is the one the
app and the test `qapp` both install.
"""

import pytest

from detail.gui.qt import QScrollBar, QStyle, QStyleOptionSlider, Qt

_BAR = QStyle.ComplexControl.CC_ScrollBar
_SUB = QStyle.SubControl


def _rect(bar, sub):
    option = QStyleOptionSlider()
    bar.initStyleOption(option)
    return bar.style().subControlRect(_BAR, option, sub, bar)


def _bar(orientation, value):
    bar = QScrollBar(orientation)
    bar.setRange(0, 100)
    bar.setPageStep(40)
    bar.setValue(value)
    (
        bar.resize(400, 18)
        if orientation == Qt.Orientation.Horizontal
        else bar.resize(18, 400)
    )
    return bar


@pytest.mark.parametrize(
    "orientation", [Qt.Orientation.Horizontal, Qt.Orientation.Vertical]
)
def test_a_scrollbar_has_no_arrow_buttons(qapp, orientation):
    bar = _bar(orientation, 50)
    assert _rect(bar, _SUB.SC_ScrollBarAddLine).isEmpty()
    assert _rect(bar, _SUB.SC_ScrollBarSubLine).isEmpty()


@pytest.mark.parametrize(
    "orientation", [Qt.Orientation.Horizontal, Qt.Orientation.Vertical]
)
def test_the_handle_reaches_both_ends_of_the_bar(qapp, orientation):
    """With the arrows gone the track is the whole bar, so the handle at its minimum starts at
    the bar's start and at its maximum ends at the bar's end — no dead handle-length at either.
    """
    horizontal = orientation == Qt.Orientation.Horizontal
    length = 400
    at_min = _rect(_bar(orientation, 0), _SUB.SC_ScrollBarSlider)
    at_max = _rect(_bar(orientation, 100), _SUB.SC_ScrollBarSlider)
    assert (at_min.left() if horizontal else at_min.top()) == 0
    assert (at_max.right() if horizontal else at_max.bottom()) == length - 1
    # Qt's proportion: a page of 40 over a range of 100 shows 40/140 of the track.
    size = at_min.width() if horizontal else at_min.height()
    assert size == int(40 * length / 140)


def test_plain_fusion_still_has_arrows(qapp):
    """The negative case: without the app's style the arrows are there, so the two tests above
    are measuring the style and not a bar that never had them."""
    from detail.gui.qt import QStyleFactory

    bar = _bar(Qt.Orientation.Horizontal, 50)
    plain = QStyleFactory.create("Fusion")
    bar.setStyle(plain)
    assert not _rect(bar, _SUB.SC_ScrollBarAddLine).isEmpty()
