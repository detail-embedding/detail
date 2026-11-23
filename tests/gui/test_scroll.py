"""A page taller than the window is reachable, and the wheel reaches it.

The class: **a content-derived minimum, summed up a layout tree, with nothing able to absorb the
remainder.** Qt propagates ``minimumSizeHint`` upward until it is the window's own minimum and
then clamps ``resize()`` to it, so the failure is not a clipped page with a scrollbar — it is a
window that *grows past the display* and content that goes under the Dock. Measured on a 1470x833
screen: a 920px analysis page in a 739px content area, and nothing would go smaller.

Two halves, and shipping either alone is worse than shipping neither:

**The container.** ``gui/scroll.py::PageScroll``. Guarded here by asserting the *floor*, because
that is the thing that was wrong — a test that only checks "a scrollbar appeared" passes on a page
that scrolls 400px it did not need to.

**The wheel.** This is the half nobody would think to write a test for, and the half that is
guaranteed to break: ``pyqtgraph.ViewBox.wheelEvent`` ends in ``ev.accept()`` and
``GLViewWidget.wheelEvent`` neither calls ``ignore()`` nor chains to ``super()``. Plots are most of
the area on exactly the pages this exists for, so without routing, the page acquires a scrollbar it
cannot be scrolled with over most of its surface. The bug is invisible to every other kind of test:
the page renders, the scrollbar is there, it is correct, and the wheel does nothing.

The routing rules are asserted one by one below rather than as one happy path, because each is a
case where the *child* must keep the wheel and a too-eager container would silently break it — a
long table that stops scrolling its own rows is a regression the container's own test would never
see.
"""

import pytest

from detail.gui.qt import (
    QApplication,
    QLabel,
    QPoint,
    QPointF,
    QSize,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWheelEvent,
    QWidget,
    Qt,
)
from detail.gui.scroll import OWNS_WHEEL, PageScroll

VIEWPORT = 739  # the measured content-area height of a 13" MacBook


def _tall_page(rows: int = 30) -> QWidget:
    page = QWidget()
    box = QVBoxLayout(page)
    for i in range(rows):
        label = QLabel(f"row {i}")
        label.setMinimumHeight(30)
        box.addWidget(label)
    return page


def _shown(widget, height: int = VIEWPORT):
    widget.resize(1000, height)
    widget.show()
    QApplication.processEvents()
    widget.resize(1000, height)
    QApplication.processEvents()
    return widget


def _wheel(target, notches: int = -3, modifiers=Qt.KeyboardModifier.NoModifier):
    """A wheel event as a real mouse sends one: 120 eighths-of-a-degree per notch."""
    return QWheelEvent(
        QPointF(target.rect().center()),
        QPointF(target.mapToGlobal(target.rect().center())),
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


# --- the container ---------------------------------------------------------------- #


def test_a_page_taller_than_the_viewport_scrolls_instead_of_forcing_the_window(qapp):
    """The bug, stated as a number: the floor was 920 against a 739px viewport."""
    scroll = PageScroll()
    scroll.setWidget(_tall_page())
    _shown(scroll)

    assert scroll.minimumSizeHint().height() < VIEWPORT, (
        "the scroll container must not inherit its page's floor — that is the whole "
        f"mechanism, and it reports {scroll.minimumSizeHint().height()}px"
    )
    assert scroll.verticalScrollBar().maximum() > 0, "nothing to scroll with"


def test_a_page_that_fits_is_given_the_whole_viewport_and_does_not_scroll(qapp):
    """The negative case. A container that always scrolls would 'fix' the bug by adding
    400px of slack to every page that was already fine."""
    scroll = PageScroll()
    page = _tall_page(rows=3)
    scroll.setWidget(page)
    _shown(scroll)

    assert scroll.verticalScrollBar().maximum() == 0
    assert page.height() == scroll.viewport().height(), (
        "a short page must fill the viewport, not sit at its own hint with dead space "
        "below it"
    )


def test_the_page_never_scrolls_horizontally(qapp):
    """Apple's HIG: scrollers must not nest on the same axis. The film strips inside these
    pages scroll horizontally, which is the pairing the HIG allows — but only while this
    container stays vertical-only."""
    scroll = PageScroll()
    scroll.setWidget(_tall_page())
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


# --- the wheel -------------------------------------------------------------------- #


def test_the_wheel_over_a_child_that_swallows_it_still_scrolls_the_page(qapp):
    """The half that would otherwise ship broken.

    A plain ``QWidget`` stands in for the plot: what matters is that the event is delivered to a
    *descendant* rather than to the viewport, which is the case ``QScrollArea`` does not handle.
    The real swallowers are asserted in ``test_pyqtgraph_really_does_swallow_the_wheel``.
    """
    scroll = PageScroll()
    page = _tall_page()
    scroll.setWidget(page)
    _shown(scroll)
    child = page.findChildren(QLabel)[0]
    before = scroll.verticalScrollBar().value()

    QApplication.sendEvent(child, _wheel(child))

    assert scroll.verticalScrollBar().value() > before, (
        "the wheel over a child left the page where it was — every plot on the page is a "
        "hole the page cannot be scrolled through"
    )


def test_pyqtgraph_really_does_swallow_the_wheel(qapp):
    """Guard the guard: the routing above is only worth having while this is true.

    If a future pyqtgraph starts propagating the wheel, the routing becomes dead weight and
    the reason it exists should be re-read rather than inherited.
    """
    pg = pytest.importorskip("pyqtgraph")
    import inspect

    assert "accept()" in inspect.getsource(pg.ViewBox.wheelEvent)
    gl = pytest.importorskip("pyqtgraph.opengl")
    source = inspect.getsource(gl.GLViewWidget.wheelEvent)
    assert "ignore" not in source and "super(" not in source


def test_a_page_that_fits_leaves_the_wheel_to_its_children(qapp):
    """Scroll-to-zoom on a plot must keep working wherever the page is not scrolling —
    which, on a large monitor, is everywhere."""
    scroll = PageScroll()
    page = _tall_page(rows=3)
    scroll.setWidget(page)
    _shown(scroll)
    child = page.findChildren(QLabel)[0]

    event = _wheel(child)
    assert not scroll.eventFilter(
        child, event
    ), "with nothing to scroll the container must not take the wheel"


def test_alt_leaves_the_wheel_to_the_child(qapp):
    """The escape hatch: zooming a plot deliberately, on a page that does scroll."""
    scroll = PageScroll()
    page = _tall_page()
    scroll.setWidget(page)
    _shown(scroll)
    child = page.findChildren(QLabel)[0]

    event = _wheel(child, modifiers=Qt.KeyboardModifier.AltModifier)
    assert not scroll.eventFilter(child, event)


def test_a_child_scroll_area_with_somewhere_to_go_keeps_the_wheel(qapp):
    """A long table inside the page scrolls its own rows before the page moves under it.

    Asserted on a child that *can* scroll and, below, on one that cannot — because the check
    is on the child's range, not its type: a ``PlotWidget`` is a ``QAbstractScrollArea`` whose
    bars have nowhere to go, and keying on the type would hand every plot the wheel and
    reinstate the bug this file exists for.
    """
    scroll = PageScroll()
    page = QWidget()
    box = QVBoxLayout(page)
    inner = PageScroll()
    inner.setWidget(_tall_page())
    inner.setMinimumHeight(120)
    box.addWidget(inner)
    box.addWidget(_tall_page())
    scroll.setWidget(page)
    _shown(scroll)
    assert (
        inner.verticalScrollBar().maximum() > 0
    ), "the inner area has nothing to scroll"

    event = _wheel(inner)
    assert not scroll.eventFilter(inner.viewport(), event)


def test_a_child_scroll_area_with_nowhere_to_go_does_not_keep_the_wheel(qapp):
    """The other half of the rule above — this is the shape a plot has."""
    scroll = PageScroll()
    page = QWidget()
    box = QVBoxLayout(page)
    inner = PageScroll()
    inner.setWidget(QLabel("short"))
    inner.setMinimumHeight(60)
    box.addWidget(inner)
    box.addWidget(_tall_page())
    scroll.setWidget(page)
    _shown(scroll)
    assert inner.verticalScrollBar().maximum() == 0

    event = _wheel(inner)
    assert scroll.eventFilter(inner.viewport(), event)


def test_a_widget_that_declares_it_owns_the_wheel_keeps_it(qapp):
    """Opt-in, for the one surface where the wheel is the interaction — the 3-D scatter."""
    scroll = PageScroll()
    page = _tall_page()
    scroll.setWidget(page)
    _shown(scroll)
    child = page.findChildren(QLabel)[0]
    child.setProperty(OWNS_WHEEL, True)

    event = _wheel(child)
    assert not scroll.eventFilter(child, event)


def test_a_page_swapped_in_later_is_covered_too(qapp):
    """``MainWindow._show`` replaces the whole page on every navigation, so a one-shot walk
    at construction would cover the first page opened and nothing after it."""
    scroll = PageScroll()
    stack = QStackedWidget()
    scroll.setWidget(stack)
    _shown(scroll)

    page = _tall_page()
    stack.addWidget(page)
    stack.setCurrentWidget(page)
    QApplication.processEvents()
    scroll.resize(1000, VIEWPORT)
    QApplication.processEvents()
    assert scroll.verticalScrollBar().maximum() > 0

    child = page.findChildren(QLabel)[0]
    before = scroll.verticalScrollBar().value()
    QApplication.sendEvent(child, _wheel(child))
    assert scroll.verticalScrollBar().value() > before


# --- the shell ---------------------------------------------------------------------- #


def test_the_shells_content_area_scrolls(qapp, tmp_path, monkeypatch):
    """Through the real entry point. Constructing a ``PageScroll`` proves the class works;
    it does not prove the app uses it, which is the thing that was broken."""
    from detail.gui import app as gui_app
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = gui_app.MainWindow()

    assert isinstance(window._scroll, PageScroll)
    assert window._scroll.widget() is window._content, (
        "the stack must still be what pages are added to — `_show` and `_discard_content` "
        "index into it"
    )

    tall = QWidget()
    box = QVBoxLayout(tall)
    filler = QWidget()
    filler.setMinimumHeight(1400)
    filler.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    box.addWidget(filler)
    window._show(tall, "tall")
    # Shown and resized: an unlaid-out window reports a zero scroll range whatever its page
    # holds, so this assertion would pass against its own reverted fix without it.
    window.show()
    window.resize(1000, 700)
    QApplication.processEvents()
    window.resize(1000, 700)
    QApplication.processEvents()

    assert window._scroll.verticalScrollBar().maximum() > 0, (
        "a page taller than the window must scroll; without this the window grows past "
        "the screen instead"
    )


def test_the_window_never_opens_larger_than_the_screen(qapp):
    """``resize()`` does not fail on a size the display cannot hold — it succeeds, and the
    title bar ends up somewhere the user cannot reach. Measured: the hardcoded 900 asked for
    67px more than a 13" MacBook has."""
    from detail.gui.app import _opening_size

    available = QApplication.primaryScreen().availableGeometry()
    size = _opening_size()
    assert size.width() <= available.width()
    assert size.height() <= available.height()


class _Pane(QWidget):
    """A pane whose floor and whose preferred size differ, which is the whole point.

    Every other fixture in this file has ``minimumSizeHint == sizeHint``, so it cannot tell the
    two sizing rules apart and would pass under either — the vacuous-fixture trap. This one asks
    for 460 and can live at 172, the measured pair for the 3-D scatter.
    """

    FLOOR = 172
    WANTS = 460

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(self.FLOOR)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):  # noqa: N802 — Qt's name
        return QSize(400, self.WANTS)


def test_a_pane_on_a_scrolling_page_gets_its_preferred_size_not_its_floor(qapp):
    """The bug the user reported, in one assertion.

    A page sized to its *floor* scrolls less and lays every pane out at its minimum — measured,
    that put the 3-D scatter at 172px, which is the letterboxing this change exists to fix. The
    page is deliberately sized to its **hint** instead and pays in scrolling. Reverse the rule and
    this fails while every other test in the file still passes, which is why it is here.
    """
    scroll = PageScroll()
    page = QWidget()
    box = QVBoxLayout(page)
    pane = _Pane()
    box.addWidget(pane, 1)
    for i in range(15):  # 460 + 15*30 = 910, comfortably past the 739px viewport
        row = QLabel(f"row {i}")
        row.setMinimumHeight(30)
        box.addWidget(row)
    scroll.setWidget(page)
    _shown(scroll)

    assert (
        scroll.verticalScrollBar().maximum() > 0
    ), "the fixture must not fit, or it proves nothing"
    assert pane.height() >= _Pane.WANTS, (
        f"the pane was laid out at {pane.height()}px against a preferred {_Pane.WANTS} — a "
        "scrolling page must not crush its panes to their floors"
    )


def test_a_page_whose_content_fits_does_not_scroll(qapp):
    """The other side: scrolling is the backstop, not the normal state."""
    scroll = PageScroll()
    page = QWidget()
    box = QVBoxLayout(page)
    split = QSplitter(Qt.Orientation.Horizontal)
    for _ in range(2):
        pane = QWidget()
        pane.setMinimumHeight(280)
        pane.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        split.addWidget(pane)
    box.addWidget(split, 1)
    scroll.setWidget(page)
    _shown(scroll)

    assert scroll.verticalScrollBar().maximum() == 0
    assert page.height() == scroll.viewport().height()
