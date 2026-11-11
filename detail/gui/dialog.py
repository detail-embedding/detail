"""The base every dialog in this app inherits, so a policy applied to one is applied to all.

**Why a base rather than a call each dialog makes.** ~100 labels across ~30 dialogs could not be
selected, so a dialog's error message — the text people most want to paste into a bug report —
could not be copied. `MainWindow._show` had given pages that for a long time; a dialog never
passes through the shell, so a dialog never got it.

There was no seam to add it at. `sizing.finish_dialog` looked like one and reached **8 of 44
dialogs**; the other 36 added their button box themselves. A rule that lives where a surface has
to *remember* to visit is a rule that gets missed, and the evidence that it gets missed is that it
already had been, thirty-six times. This class is now closed from both ends: every dialog inherits
this base, and `finish_dialog` has been deleted so there is nothing left to call instead.

**And not an application-wide event filter**, which was tried and segfaults — reproducibly, about
one suite run in two:

    QCoreApplicationPrivate::sendThroughApplicationEventFilters(QObject*, QEvent*)
      sipQObject::eventFilter(QObject*, QEvent*)
        sip_api_call_method → sipWrapInstance → QObject::connect
          EXC_BAD_ACCESS ... (possible pointer authentication failure)

An application filter is offered *every* object's events, including posted ones bound for an
object that has since been destroyed, and sip must build a Python wrapper for that pointer before
any of the filter's own code runs. No guard inside the filter can help, because the crash happens
upstream of its first line. **A dialog applying a policy to itself has none of that exposure**:
the only object involved is the one being shown, which is alive by definition.

`showEvent`, not `__init__`, because a dialog's children are commonly built after
``super().__init__()`` and sometimes filled in later still — a `QMessageBox` grown from a worker
result, a form rebuilt when the selection changes. Show is the last moment before the user can
read anything, which is exactly when the claim "you can copy this" has to be true.

**The base also owns the dialog's structure**, which the text policy deliberately does not.
Reported from the app: *"windows open with no scroll bar, hiding content."* The launch dialog it
was reported on has eight items in its top-level layout and **one** of them scrolls — a
`PageScroll` around the module's config form. The other seven, the recordings list among them,
cannot scroll at all, so they set a floor Qt clamps ``resize()`` to: asked for 300px that dialog
returns **323**, and on a cohort large enough the floor passes the screen and the window grows
past the display with its buttons under the Dock.

An inner scroll area cannot fix that, and no dialog can be trusted to remember an outer one — the
same argument that put the text policy here. So :meth:`Dialog.body` is where a dialog's rows go,
and the base wraps them in one scroll with the buttons pinned outside it.

Structure can live in ``__init__`` where the text policy could not: the layout exists from
construction, and rows added later land inside the body either way.

**A dialog shown non-modally has no default button.** ``QDialog`` makes every ``QPushButton`` in
it *auto-default* and, on show, makes the first of them in the focus chain the *default* — the one
Enter clicks from anywhere in the window that does not use Enter itself, a line edit included
(it passes Return on after acting on it). In a modal OK / Cancel dialog that is the point: Enter
is OK. In a window that stays open beside its page it is a trap: the cluster window's first button
was **Remove**, so Enter in its label field labelled the cluster and then removed the label it had
just written (reported 2026-11-09). A window beside a page has no "OK" to press, so at show — and
whenever its content is refreshed — the base takes default and auto-default off every button of a
dialog that is not modal, and Enter reaching it clicks nothing (:meth:`keyPressEvent`). Enforced
by ``requirements.a_non_modal_window_has_no_default_button``.
"""

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QPaintEvent, QPainter, QPalette, QShowEvent
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.gui.scroll import PageScroll
from detail.gui.selectable import make_text_selectable

#: How far the dialog's outline sits from the window fill, as a fraction of the distance to the
#: text colour. 0.25 measures at 56 points of lightness on a dark palette and 60 on a light one —
#: matched across themes, and close enough to the old `Mid` on a light one (55) that light-mode
#: appearance is unchanged.
_BORDER_MIX = 0.25

#: The floor :func:`border_color` must clear, in lightness. Well under the 56/60 it actually
#: measures, and well over the 10 that shipped — a margin, not a tripwire on the current value.
MIN_BORDER_CONTRAST = 24


def border_color(palette: QPalette) -> QColor:
    """The dialog outline's colour: the window fill, nudged a quarter of the way to the text.

    **Derived, not a palette role**, because no role is visible in both themes. Measured across
    the two palettes this app actually renders under — `windows11` dark and the `fusion` light
    the offscreen test platform uses::

        role        dark gap   light gap
        Mid               10          55     <- what shipped; invisible on a dark box
        Light             90          16     <- invisible on a light one
        Dark               0          80     <- *identical* to Window on windows11 dark
        Midlight          60          37
        derived           56          60

    `Mid` was chosen against the light palette the tests run under, where it is fine. On
    `windows11` dark it is `#282828` on `#1e1e1e`: ten points of lightness, which is not a
    visible line. That is the same mistake the bug itself was — assuming one platform's
    rendering — so the colour is computed against whatever palette is live rather than picked.

    **Not the house `darker(122)`/`lighter(145)` tint** (`app.py`'s mode banner,
    `pages.py`'s row tint). Those are multiplicative, so they scale with the value they start
    from: on `#1e1e1e` the lighter form yields `#2b2b2b`, a gap of 13. That reads fine as a
    large tinted panel and disappears as a one-pixel line. Blending toward `WindowText`
    instead gives a gap that does not collapse on a very dark or very light window.
    """
    window = palette.color(QPalette.ColorRole.Window)
    ink = palette.color(QPalette.ColorRole.WindowText)
    return QColor(
        round(window.red() + (ink.red() - window.red()) * _BORDER_MIX),
        round(window.green() + (ink.green() - window.green()) * _BORDER_MIX),
        round(window.blue() + (ink.blue() - window.blue()) * _BORDER_MIX),
    )


class Dialog(QDialog):
    """A `QDialog` that carries this app's dialog-wide policies.

    Inherit this rather than `QDialog` directly. `tests/gui/test_requirements.py` sweeps every
    dialog it can build, so a new one that skips this is caught the day it is written rather than
    the day somebody tries to copy an error message out of it.

    Put the dialog's rows in :meth:`body` and its button box through :meth:`set_buttons`; the
    base supplies the scroll between them.
    """

    #: A **settings** dialog — one built around a config form — sets this. It then opens, and
    #: grows whenever its form changes, tall enough to show every row outside *Advanced
    #: settings* without scrolling, up to the screen. Never shrinks: a size someone dragged
    #: the window to is theirs.
    #:
    #: Reported on the launch-analysis dialog: its opening 520px put Query's behavior buttons
    #: below the fold, where nothing said they existed. A scroll is the right answer for a
    #: dialog whose content is genuinely longer than the screen, and the wrong one for a form
    #: that fits and was simply opened short. Opt-in rather than for every dialog, because a
    #: clip or frame viewer is sized for its picture, not its content; `tests/gui/
    #: test_settings_dialogs.py` fails on a dialog that builds a config form and does not say
    #: which it is.
    fits_content: bool = False

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._outer: QVBoxLayout | None = None
        self._body_layout: QVBoxLayout | None = None
        self._scroll: PageScroll | None = None
        #: Coalesces the refit a changing body asks for — see :meth:`eventFilter`.
        self._refit_timer: QTimer | None = None
        #: Whether the trailing stretch has been added. `showEvent` fires on every show, and a
        #: dialog shown twice must not collect a spacer each time.
        self._stretched = False
        #: Where that spacer sits, so it can be re-decided when the body gains content after
        #: the show that placed it (`_restretch`). `None` means there is none to move.
        self._stretch_index: int | None = None
        self._body_widget: QWidget | None = None
        self._restretch_timer: QTimer | None = None

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt's name
        """Outline the dialog, since its fill is otherwise indistinguishable from what's behind it.

        **Reported from the app**: a launch dialog opened over the main window and the two were
        the same colour edge to edge — on a platform/theme whose window manager draws no visible
        frame of its own, there was nothing left to show where one window ends and the next
        begins. A light line is the whole fix.

        The colour comes from :func:`border_color` rather than a palette role, because no single
        role is visible in both themes; see there for the measurements.

        **Painted, not a `QFrame` wrapped around the content.** The house style is zero QSS (a
        `QFrame` shape or a palette colour instead), and a `QFrame` is the usual way to get one —
        but here it would have to sit inside :attr:`_outer`, the dialog's top-level layout, which
        `a_dialog_scrolls_its_content` (`gui/requirements.py`) requires to hold *only* the scrolled
        body and the button box; a wrapping frame would read as a stray third widget and fail that
        sweep on every dialog in the app. Painting on the dialog itself reaches the same pixels
        without touching that layout.
        """
        super().paintEvent(event)
        with QPainter(self) as painter:
            painter.setPen(border_color(self.palette()))
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    # -- structure ----------------------------------------------------------- #
    def body(self) -> QVBoxLayout:
        """The layout a dialog's rows go in — inside the scroll, above the buttons.

        Built on first use rather than in ``__init__`` **so an un-migrated dialog is untouched**.
        Qt refuses a second layout on a widget that already has one (it warns and keeps the
        first), so a base that claimed the layout eagerly would silently orphan the
        ``QVBoxLayout(self)`` every dialog currently writes — thirty-odd dialogs broken at once,
        in a way that renders as an empty window. Lazily, a dialog that has not moved over builds
        its own layout exactly as before, and the requirement sweep names it as remaining work.
        """
        if self._body_layout is None:
            self._build()
        return self._body_layout

    def set_buttons(self, buttons: QDialogButtonBox) -> None:
        """Pin the button box **outside** the scroll, so the way out never scrolls away.

        This is the half that makes scrolling safe. A button box inside the scrolled body is
        reachable only by scrolling to it, which is the reported failure one step removed — and
        the more annoying half, because the way out of the dialog becomes the thing you cannot
        find.

        The top-justify is **not** decided here; see :meth:`showEvent` for why not.
        """
        if self._body_layout is None:
            self._build()
        self._outer.addWidget(buttons)

    def _build(self) -> None:
        """Claim the dialog's layout: one scroll, with room beneath it for the buttons."""
        self._outer = QVBoxLayout(self)
        body = QWidget()
        self._body_widget = body
        # The body is where a dialog's rows land, including ones a worker adds after the dialog
        # is on screen — which is what `_restretch` watches for.
        body.installEventFilter(self)
        # Owned by this dialog, so it cannot outlive it. A bare `QTimer.singleShot` would post a
        # callback that runs after a dialog that closed in the meantime — `full_frame`'s window
        # is `WA_DeleteOnClose`, so that is a live path, not a hypothetical. A timer parented
        # here is destroyed with the dialog and its pending shot goes with it.
        self._restretch_timer = QTimer(self)
        self._restretch_timer.setSingleShot(True)
        self._restretch_timer.timeout.connect(self._restretch)
        # Its own timer, not the restretch one: moving the spacer is itself a layout change,
        # so one trigger for both would re-decide the spacer forever.
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.timeout.connect(self._refit)
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        # `PageScroll` rather than a bare `QScrollArea`: a dialog is not reached through the
        # content area, so it gets no page scroll of its own, and this is where its wheel
        # handling has to come from — a config form is mostly spin boxes and combos, every one
        # of which swallows the wheel *and changes its value* doing it.
        self._scroll = PageScroll()
        self._scroll.setWidget(body)
        self._outer.addWidget(self._scroll, 1)

    def sizeHint(self):
        """Open at the size the content asks for, capped to the screen.

        A `QScrollArea` reports a generic hint — measured 540x360 — with no reference to what is
        inside it, so putting the body in one made every small dialog inflate:
        `NewExperimentDialog`, which is two fields wanting 246x52, opened at **413px** of mostly
        empty window. `QAbstractScrollArea.AdjustToContents` does not help, because `PageScroll`
        sets ``widgetResizable(False)`` and the policy then reads the widget's current size —
        which `PageScroll._fit` derives from the viewport. Circular; the dialog has to answer.

        The cap is the other half. A dialog whose content is genuinely enormous must open at the
        screen rather than past it — the failure this whole change exists to fix — and from
        there the scroll takes over.
        """
        base = super().sizeHint()
        if self._scroll is None or self._scroll.widget() is None:
            return base
        content = self._scroll.widget().sizeHint()
        chrome = base.height() - self._scroll.sizeHint().height()
        wanted = content.height() + max(0, chrome)
        screen = QApplication.primaryScreen()
        if screen is not None:
            wanted = min(wanted, screen.availableGeometry().height())
        return QSize(base.width(), max(self.minimumSizeHint().height(), wanted))

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.refresh_content_policy()

    def refresh_content_policy(self) -> None:
        """Apply the show-time policies to whatever the body holds *now*.

        Called from ``showEvent``, and again by ``gui/busy.py`` after a ``run_busy`` callback has
        run against this dialog — because show is not the last moment content arrives.

        ``_top_justify``'s docstring already records moving this from ``set_buttons`` to show, since
        the launch dialogs populate their recordings list after ``__init__``. A ``run_busy``
        callback is the *third* moment, and it lands after show: `StreamPreviewDialog` builds every
        one of its labels in ``_populate``, delivered off-thread, so none of them existed when
        ``showEvent`` ran and none was selectable — text a bug reporter could read and not quote
        (#563). All three policies were missed, not only selectability.
        """
        make_text_selectable(self)
        self._no_default_button()
        self._top_justify()
        self._refit()

    def _no_default_button(self) -> None:
        """A dialog that is not modal — a window beside its page — gets no default button and no
        auto-default one: Enter in it never clicks a button (module docstring)."""
        if self.isModal():
            return
        for button in self.findChildren(QPushButton):
            button.setDefault(False)
            button.setAutoDefault(False)

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Enter in a non-modal dialog clicks nothing, even a button that made itself default
        after the show; a modal one keeps Qt's Enter-is-OK."""
        if not self.isModal() and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.ignore()
            return
        super().keyPressEvent(event)

    def _top_justify(self) -> None:
        """Give the body a trailing stretch, unless something in it already takes the slack.

        **At show, not when the buttons are set**, and the difference is a bug I shipped for an
        afternoon. `set_buttons` is the last call in a dialog's ``__init__``, which reads like
        the right moment and is not: the launch dialogs populate their recordings list *after*
        it, in ``_on_module_changed``. Asked before that, the list is still uncapped and reports
        as absorbing the slack, so no stretch was added — and a short module's dialog spread its
        rows down the window, its 15px labels laying out at **77px**.

        The same reason `make_text_selectable` runs here: a dialog's contents are commonly
        finished after its constructor, so anything that has to reason about them has to wait
        for the last moment before the user sees it.
        """
        from detail.gui.sizing import has_slack

        if self._body_layout is None or self._stretched:
            return
        self._stretched = True
        if not has_slack(self._body_layout):
            # Stretch factor **1**, not `addStretch()`'s default of 0: at 0 the spacer is merely
            # *expanding* and competes with every `Preferred` widget above it, which a
            # word-wrapped QLabel is — measured at 149px on the dialog that prompted that rule.
            self._body_layout.addStretch(1)
            self._stretch_index = self._body_layout.count() - 1

    def _restretch(self) -> None:
        """Re-decide the trailing stretch once the body has content.

        **A dialog that loads its content off the UI thread is empty when it is shown**, so the
        decision above is made against nothing: no child takes slack, a spacer goes in, and every
        row `_populate` adds later lands *below* it. `StreamPreviewDialog` is the measured case —
        150px of dead window above the video, and the video's own block squeezed to share what
        was left with the spacer, 50/50, because both carried stretch 1.

        Qt has no "a layout gained an item" signal, so this rides `ChildAdded` on the body widget
        (see :meth:`eventFilter`) and runs after that child is in place. Two outcomes, the same
        two the show-time decision has: the new content takes the slack itself and the spacer is
        removed, or it does not and the spacer moves back to the end where it belongs.
        """
        from detail.gui.sizing import has_slack

        layout = self._body_layout
        if layout is None or self._stretch_index is None:
            return
        item = layout.itemAt(self._stretch_index)
        if item is None or item.spacerItem() is None:
            self._stretch_index = None  # somebody rebuilt the body; not ours to move
            return
        layout.takeAt(self._stretch_index)
        self._stretch_index = None
        if not has_slack(layout):
            layout.addStretch(1)
            self._stretch_index = layout.count() - 1

    def eventFilter(self, watched, event):
        if watched is self._body_widget and event.type() in (
            QEvent.Type.ChildAdded,
            QEvent.Type.LayoutRequest,
        ):
            # Queued: the child is parented before it is in the layout, so asking now reads the
            # layout as it was. Restarting the same timer coalesces a burst — `_populate` adds
            # several widgets, and only the last one's shot survives.
            if event.type() == QEvent.Type.ChildAdded:
                self._restretch_timer.start(0)
            # Any change to what the body holds or shows — a form swapped for another module's,
            # a method's rows revealed — can change what the window must be to show it.
            self._refit_timer.start(0)
        return super().eventFilter(watched, event)

    def _refit(self) -> None:
        """Size the window to its content: wide enough always, tall enough when it declares
        :attr:`fits_content`.

        **Not only at show.** Both rules were applied once, when the dialog opened — and the
        launch dialogs replace their whole form when you pick another module, *after* that
        show. So the width floor stayed the one the first module's form needed, and Query's
        row of five buttons was crushed inside it until "Edit non-examples" read as a stub.
        The same moment the body changes is the moment its size has to be re-asked.
        """
        if not self.isVisible():
            return
        self._keep_wide_enough()
        if self.fits_content:
            self._grow_to_content()

    def _grow_to_content(self) -> None:
        """Grow to what :meth:`sizeHint` asks — the body unscrolled, capped to the screen —
        and keep the grown window on the screen."""
        hint = self.sizeHint()
        screen = self.screen() or QApplication.primaryScreen()
        width = max(self.width(), hint.width())
        height = max(self.height(), hint.height())
        if screen is not None:
            available = screen.availableGeometry()
            # The cap in `sizeHint` is for the client area; the title bar comes on top of it.
            frame = self.frameGeometry().height() - self.height()
            height = min(height, max(available.height() - frame, self.height()))
        if (width, height) == (self.width(), self.height()):
            return
        self.resize(width, height)
        if screen is not None:
            # A window grown downward from where it opened can put its buttons under the Dock,
            # which is the failure the scroll exists to prevent. Slide it up instead.
            bounds = self.frameGeometry()
            if bounds.bottom() > available.bottom():
                self.move(
                    bounds.x(),
                    max(available.top(), available.bottom() - bounds.height()),
                )

    def _keep_wide_enough(self) -> None:
        """The window may never be narrower than its content needs.

        A scroll area answers for its content's size, so putting the body in one collapsed the
        dialog's own minimum width from what the rows need to **188px** — and a five-button row
        that used to force the window wider was simply crushed instead, rendering as
        "Add behavio" and ": non-examp".

        This scrolls **vertically only** (`PageScroll` turns the horizontal bar off, following
        Apple's rule against nesting scrollers on one axis). Vertical-only has a corollary that
        has to be enforced rather than assumed: whatever the content needs horizontally, the
        window has to give it, because there is no other way for the user to reach it.
        """
        if self._scroll is None:
            return
        content = self._scroll.widget()
        if content is None:
            return
        bar = self._scroll.verticalScrollBar()
        self._scroll.setMinimumWidth(
            content.minimumSizeHint().width() + (bar.sizeHint().width() if bar else 0)
        )
