"""A page that is taller than the window scrolls, rather than going out of reach.

The class: **a content-derived minimum, summed up a layout tree, with nothing able to absorb the
remainder.** Qt propagates ``minimumSizeHint`` upward until it becomes the window's own minimum,
and then *clamps* ``resize()`` to it — so a page taller than the screen does not get clipped with
a scrollbar, it silently **enlarges the window past the display** and the bottom goes under the
Dock. Measured on a 1470x833 screen: the temporal-clustering analysis page's floor is 920px
against a 739px content area, and neither the window nor the page would go smaller.

The fix is structural rather than a shave. This is the same mechanism CSS names in Flexbox §4.5 —
an automatic minimum is content-based *"for scroll containers the automatic minimum size is
zero"* — so making the content area a scroll container is not a workaround beside the real fix,
it **is** the fix, and it is the only one that changes the arithmetic instead of the constants.
Measured: it takes the content area's floor from 920px to 68px.

Shipping it alone would trade one bug for another, which is why the wheel handling below is not
optional. ``pyqtgraph``'s ``ViewBox.wheelEvent`` ends in ``ev.accept()`` and ``GLViewWidget``'s
neither calls ``ignore()`` nor chains to ``super()``, so **every plot and the 3-D view is a hole
the page cannot be scrolled through** — and those are most of the area on the pages this exists
for. QGIS hit this once and fixed it once, globally (``QgsScrollArea``: *"All QGIS code and
plugins should use QgsScrollArea in place of QScrollArea"*), rather than patching each widget.
Same here: one class, used everywhere, with the rule baked in.
"""

from detail.gui.qt import (
    QAbstractScrollArea,
    QApplication,
    QEvent,
    QScrollArea,
    QWidget,
    Qt,
)

#: Marks a widget that genuinely needs the wheel for itself — set with
#: ``widget.setProperty(OWNS_WHEEL, True)``.
#:
#: Opt-**in**, and deliberately not a list of classes here. The default has to be "the page
#: scrolls", because the widgets that swallow the wheel by accident (a plot, a spin box, a combo
#: inside a form) vastly outnumber the ones that mean to, and a class list in this module would
#: be a second place to remember. A widget that means it says so at its own construction, where
#: the reason is visible.
OWNS_WHEEL = "detail_owns_wheel"


class PageScroll(QScrollArea):
    """A vertical scroll container for a page, with the wheel going to the page by default.

    Vertical only. Apple's HIG is explicit that scrollers must not nest on the same axis
    (*"Avoid putting a scroll view inside another scroll view with the same orientation"*), and a
    horizontal bar here would fight the film strips, which scroll horizontally inside it — that
    pairing is the one the HIG expressly allows.

    The page is given ``max(viewport, its preferred height)`` — see :meth:`_fit`, which states
    that rule rather than leaving it to ``setWidgetResizable``.

    Wheel routing, in order:

    1. Nothing to scroll → the child keeps it, so a page that fits behaves exactly as it does
       today. This is what keeps the change from regressing a large monitor at all.
    2. ``Alt`` held → the child keeps it. The escape hatch for zooming a plot deliberately.
    3. A descendant scroll area with somewhere to go → it keeps it. A long table inside the page
       scrolls its own rows before the page moves under it.
    4. A widget carrying :data:`OWNS_WHEEL` → it keeps it.
    5. Otherwise the page scrolls.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # **Not** `setWidgetResizable(True)`, which delegates the height to `qSmartMinSize` and
        # so to whether the page's size policy happens to carry Qt's Shrink flag: the same
        # container gave a page its *hint* in the shell (a `QStackedWidget` between it and the
        # page) and its *minimum* in a test that set the page directly. Two behaviours, no
        # visible difference in the calling code. `_fit` below states the rule instead.
        self.setWidgetResizable(False)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 — Qt's name
        super().setWidget(widget)
        self._watch(widget)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        """Size the page to the viewport, or to what it asks for when that is taller.

        **The hint, not the floor**, and the difference is the whole point. A page laid out at its
        floor is a page with every pane crushed to its minimum: measured on the
        temporal-clustering page, floor-sizing gave 920px and put the 3-D scatter back at
        **172px** — the letterboxing this change exists to fix — where hint-sizing gives 1296px
        and the scatter 460px. The cost is scrolling, and scrolling is recoverable in a way that a
        pane too small to read is not.

        So the remaining work is at the panes: every pixel by which a floor is dishonest is a
        pixel of scrolling nobody asked for, but it is never a pane the user cannot use.
        """
        page = self.widget()
        if page is None:
            return
        viewport = self.viewport()
        # Width first: a word-wrapped label's height depends on the width it is given, so a hint
        # read before the resize is the hint for the wrong width.
        page.resize(viewport.width(), page.height())
        wanted = max(page.sizeHint().height(), page.minimumSizeHint().height())
        page.resize(viewport.width(), max(viewport.height(), wanted))

    def _watch(self, widget: QWidget) -> None:
        """Filter ``widget`` and everything under it, now and as it grows.

        The subtree is rebuilt constantly — ``MainWindow._show`` swaps a whole page in — so a
        one-shot walk at construction would cover the first page and nothing after it. Filtering
        ``ChildAdded`` on every watched widget is what keeps new descendants covered: a widget
        added under a watched one announces itself on its new parent.

        Qt drops an event filter automatically when either object dies, and this object is an
        ancestor of everything it filters, so it always outlives them — the lifetime trap in
        ``tests/gui/test_workers.py`` does not apply.
        """
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 — Qt's name
        kind = event.type()
        if kind == QEvent.Type.ChildAdded:
            child = event.child()
            if child is not None and child.isWidgetType():
                self._watch(child)
            return False
        if kind == QEvent.Type.LayoutRequest and obj is self.widget():
            # The page's own size changed — a new page swapped in, a band folded, a table
            # refilled. Without this the fit is only ever computed against whatever the page
            # held when it was *set*, which for the shell is an empty stack: the scroll then
            # never appears and the page is silently squeezed into the viewport instead.
            self._fit()
            return False
        if kind != QEvent.Type.Wheel:
            return False
        if not self._page_takes_wheel(obj, event):
            return False
        return self._scroll_page(event)

    def _page_takes_wheel(self, obj, event) -> bool:
        if self.verticalScrollBar().maximum() <= 0:
            return False
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            return False
        widget = obj if isinstance(obj, QWidget) else None
        while widget is not None and widget is not self:
            if bool(widget.property(OWNS_WHEEL)):
                return False
            # A `PlotWidget` is a `QGraphicsView` and so passes the isinstance check — its own
            # bars have nowhere to go, which is why this asks the range rather than the type.
            if (
                isinstance(widget, QAbstractScrollArea)
                and widget.verticalScrollBar().maximum() > 0
            ):
                return False
            widget = widget.parentWidget()
        return True

    def _scroll_page(self, event) -> bool:
        """Move the page by ``event``'s delta, and say the event is spent.

        Re-dispatching the event to the viewport would be the obvious thing and is wrong: the
        child already accepted it, and Qt gives no way to un-accept it for a second delivery.
        Driving the scrollbar directly is what every hand-rolled version of this does.
        """
        bar = self.verticalScrollBar()
        # A trackpad reports real pixels; a wheel reports eighths of a degree, 120 per notch.
        # Using the angle for a trackpad turns its fine gesture into whole-line jumps.
        pixels = event.pixelDelta().y()
        if pixels:
            bar.setValue(bar.value() - pixels)
        else:
            notches = event.angleDelta().y() / 120.0
            step = bar.singleStep() * QApplication.wheelScrollLines()
            bar.setValue(bar.value() - round(notches * step))
        event.accept()
        return True
