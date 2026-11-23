"""The shared horizontal zoom (`gui/zoom.py`): the window, the overview bar, the gestures."""

import pytest

from detail.gui.zoom import OverviewBar, ZoomWindow, zoom_wheel


def test_the_window_stays_inside_the_span_and_snaps_to_whole():
    window = ZoomWindow()
    window.set_view(-0.2, 0.3)
    assert window.view == pytest.approx((0.0, 0.5))
    window.set_view(0.9, 1.4)
    assert window.view == pytest.approx((0.5, 1.0))
    window.set_view(-1.0, 2.0)
    assert window.view == (0.0, 1.0) and not window.is_zoomed()


def test_the_window_never_narrows_past_its_floor():
    window = ZoomWindow()
    window.set_floor(0.1)
    window.zoom_by(1e-6, 0.5)
    assert window.size == pytest.approx(0.1)
    window.set_view(0.2, 0.21)
    assert window.size == pytest.approx(0.1)


def test_a_zoom_keeps_the_anchored_point_where_it_is():
    window = ZoomWindow()
    window.set_view(0.2, 0.6)
    anchor = 0.25
    at = 0.2 + anchor * 0.4
    window.zoom_by(0.5, anchor)
    left, right = window.view
    assert left + anchor * (right - left) == pytest.approx(at)
    assert right - left == pytest.approx(0.2)


def test_pan_and_centre_keep_the_size():
    window = ZoomWindow()
    window.set_view(0.2, 0.4)
    window.pan_by(0.5)
    assert window.view == pytest.approx((0.3, 0.5))
    window.centre_on(0.8)
    assert window.view == pytest.approx((0.7, 0.9))


def test_changed_fires_only_on_a_change():
    window = ZoomWindow()
    seen = []
    window.changed.connect(lambda: seen.append(window.view))
    window.fit()
    assert seen == []
    window.set_view(0.1, 0.2)
    window.set_view(0.1, 0.2)
    assert len(seen) == 1


def test_the_overview_is_the_window_both_ways(qapp):
    from detail.gui.qt import QVBoxLayout, QWidget

    window = ZoomWindow()
    host = QWidget()
    bar = OverviewBar(window)
    QVBoxLayout(host).addWidget(bar)
    host.show()
    assert not bar.isVisible(), "nothing to move at the whole span: no empty box"
    assert bar.sizePolicy().retainSizeWhenHidden(), "its room is kept, so nothing jumps"
    window.set_view(0.25, 0.5)
    assert bar.isVisible()
    steps = OverviewBar.RESOLUTION
    assert bar.pageStep() == steps // 4
    assert bar.value() == steps // 4
    bar.setValue(steps // 2)  # dragging the handle
    assert window.view == pytest.approx((0.5, 0.75))
    window.fit()
    assert not bar.isVisible()
    host.close()


def test_the_overview_is_a_real_scrollbar_with_the_apps_style(qapp):
    """The point of the class: the same control as every other scrollbar, so it looks the
    same — not a picture of one."""
    from detail.gui.qt import QScrollBar

    from detail.gui.style import DetailStyle

    bar = OverviewBar(ZoomWindow())
    assert isinstance(bar, QScrollBar)
    assert isinstance(bar.style(), DetailStyle)


def _wheel(dx, dy):
    from detail.gui.qt import QPoint, QPointF, QWheelEvent, Qt

    return QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, 0),
        QPoint(dx, dy),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_the_wheel_zooms_and_a_sideways_scroll_pans(qapp):
    window = ZoomWindow()
    assert zoom_wheel(window, _wheel(0, 120), 0.5)
    assert window.size == pytest.approx(0.8)
    left = window.view[0]
    assert zoom_wheel(window, _wheel(-120, 0), 0.5)
    assert window.view[0] > left, "scrolling sideways toward the end moves later"
    assert not zoom_wheel(window, _wheel(0, 0), 0.5)
