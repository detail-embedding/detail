"""How a widget or dialog is sized — the rules, in a leaf both hubs and leaves can reach.

Three copies of it existed: `pages.add_body_widget` (which hands out the stretch),
`pages._can_grow` (which decides whether a dialog already has slack), and
`tests/gui/layout_checks._wants_to_fill` (which decides whether a label that grew was entitled
to). The test's docstring even said "same flag test `pages.add_body_widget` uses", which is an
acknowledged duplicate — and a *guard* that keeps its own copy of the predicate it checks stops
being a guard the moment the two drift.

A leaf, not `pages.py`, for the reason `gui/tables.py` is one: importing that 5000-line hub pulls
in the module registry, so nothing below it can call in (`tests/gui/test_import_cycles.py`).

`fit_list_height` moved here for that reason and not for tidiness. It lived in `pages.py`, so
`gui/streams.py` — a leaf — *could not call it*, and its two import dialogs duly added their
button box with a bare `addWidget` and spread their rows down the window. The rule was written,
the helper existed, and the file that needed it could not reach it: exactly what happened to
`fit_table_height` before it became `gui/tables.py`.
"""

from PyQt6.QtCore import QEvent, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QFrame,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QWidget,
)

from detail.gui.lifetime import cleared

#: Qt's "no maximum" sentinel. A widget capped below it cannot absorb extra height however
#: expansive its size policy claims to be.
UNBOUNDED_HEIGHT = 16777215


def mono_font() -> QFont:
    """The monospace face for ids, paths, and any column where alignment matters.

    Built on call, not at import: a ``QFont`` constructed before the ``QApplication`` exists
    does not resolve against the interface font, the same reason :func:`lines` is a function.
    (``pages.MONO`` and two local copies predate this leaf; they are equivalent, and a leaf
    below ``pages.py`` could not reach that one.)
    """
    font = QFont("Monospace")
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def text_width(sample: str) -> int:
    """How wide ``sample`` is in the interface font — the horizontal twin of :func:`lines`.

    For a label whose text changes and whose neighbours must not move because of it. Sized from
    the widest thing it can say rather than from a pixel number, for the same reason `lines` is:
    a number picked by looking at one screen is right on that screen and on no other, and it
    cannot follow a user who has asked for larger text.
    """
    return QFontMetrics(QApplication.font()).horizontalAdvance(sample)


def emphasis_font(widget, steps: int = 1) -> QFont:
    """``widget``'s font, bold and ``steps`` sizes larger — the one way to make a heading.

    A leaf, for the reason :func:`fit_list_height` is one. The sanctioned helper was
    ``pages._status_label``, and ``pages.py`` is the 5000-line hub that imports the module
    registry — so ``audit_tab.py``, a leaf, *could not reach it* and built the same three lines
    by hand, twice. Exactly what happened to ``finish_dialog`` before it became a base class.

    And they drifted, in the way a copied number always does: the guidelines said "one size
    larger", ``_status_label`` used **+2**, and the hand-rolled pair used **+1** — three
    spellings of one gesture, none of which knew about the others. ``steps`` is a parameter
    because a page's status line and a panel's intro heading are legitimately different weights;
    what is not legitimate is each deciding it privately.
    """
    font = QFont(widget.font())
    font.setBold(True)
    font.setPointSize(font.pointSize() + steps)
    return font


def larger_font(widget, steps: int = 2) -> QFont:
    """``widget``'s font ``steps`` points larger, weight unchanged — text that should be easy to
    hit and read without being a heading, like INTERPRET's label field. The plain sibling of
    :func:`emphasis_font`."""
    font = QFont(widget.font())
    font.setPointSize(font.pointSize() + steps)
    return font


def small_font(widget, steps: float = 1.5) -> QFont:
    """``widget``'s font ``steps`` points smaller (never below 7pt) — the one way to make fine
    print: a reading that should stay legible and out of the way, like the clip window's strips
    under the picture they describe. The smaller sibling of :func:`emphasis_font`."""
    font = QFont(widget.font())
    font.setPointSizeF(max(font.pointSizeF() - steps, 7.0))
    return font


def lines(count: float) -> int:
    """``count`` lines of the interface font, in pixels — this package's ``em``.

    Every vertical size in the GUI was a raw pixel number picked by looking at one screen, and a
    raw pixel number is correct on exactly that screen. Two things it cannot follow: a display
    whose text is scaled, and a user who has *asked* for larger text — on whom the boxes stay the
    same size and only the words inside them grow, which is the opposite of what they wanted.

    Qt Widgets has no ``rem``; the idiom (ilastik's, among others) is font metrics, and it is four
    lines of code. Measured here: one line is 16px at the 13pt system font, so the constants this
    replaces read as ``_PLOT_PX = 240`` → 15 lines and ``setMinimumHeight(420)`` → 26 lines.

    Note this is a *typographic* unit, not a DPI fix — Qt 6 already works in logical pixels. It
    buys the text-size half, and it makes a size say how much it means to hold rather than how
    many pixels somebody's monitor had.
    """
    return round(count * QFontMetrics(QApplication.font()).height())


#: The policy flags that mean "give me more than my hint".
#:
#: **Two flags, not one.** ``ExpandFlag`` is the obvious one and it is what ``Expanding`` and
#: ``MinimumExpanding`` carry. ``Ignored`` does not carry it — measured, its value is
#: ``Grow | Shrink | Ignore`` — and yet Qt documents it as *"the sizeHint() is ignored; the widget
#: will get as much space as possible"*, which is the same intent by a different route: one says
#: "more than my hint", the other says "I have no hint, fill me".
#:
#: This function's docstring claimed to cover ``Ignored`` and did not, and nothing noticed while
#: no widget used it. The first one that did — the video pane, which must not size itself from
#: the pixmap it is handed — was immediately read as a label spreading down its pane.
_WANTS_MORE_THAN_ITS_HINT = (
    QSizePolicy.PolicyFlag.ExpandFlag.value | QSizePolicy.PolicyFlag.IgnoreFlag.value
)


def expands_vertically(widget) -> bool:
    """Whether ``widget``'s size policy asks for more vertical space than its hint.

    ``.value`` on both sides: PyQt6 refuses to ``&`` a ``Policy`` with a ``PolicyFlag``
    directly, and the flags are what carry the intent across ``Expanding``,
    ``MinimumExpanding`` and ``Ignored`` alike — enumerating those three by name would miss the
    next one.
    """
    return bool(widget.sizePolicy().verticalPolicy().value & _WANTS_MORE_THAN_ITS_HINT)


def can_grow(widget) -> bool:
    """Whether ``widget`` would *actually* absorb extra vertical space.

    Both halves matter. A policy that expands is not enough — a ``QListWidget`` sized to its rows
    with ``fit_list_height`` still reports ``Expanding``, and counting it as slack is how a dialog
    that visibly spread its rows passed a check for spreading.
    """
    return expands_vertically(widget) and widget.maximumHeight() >= UNBOUNDED_HEIGHT


def _lays_out_vertically(layout) -> bool:
    """Whether ``layout``'s own stretches and spacers distribute *vertical* space.

    A `QHBoxLayout`'s do not, and that is the hole this closes: `ComparisonExportDialog` put its
    destination field in a row as `addWidget(field, 1)`, the recursion found a stretch factor of
    1, and the dialog was declared already-stretched. It got no spacer and spread its rows down
    the window — a 60px instructions label rendered at 432px. A `QFormLayout` or `QGridLayout`
    has no direction and does stack rows vertically, so it counts.
    """
    direction = getattr(layout, "direction", None)
    if direction is None:
        return True
    return direction() in (
        QBoxLayout.Direction.TopToBottom,
        QBoxLayout.Direction.BottomToTop,
    )


def _unbounded(item) -> bool:
    """Whether a layout item may grow past its current height at all.

    A stretch factor on a widget capped at its content — a table sized to its rows in a pop-up
    (`shows_every_row`), say — is a claim on slack the widget cannot take, and the slack then
    lands on the word-wrapped labels around it.
    """
    widget = item.widget()
    if widget is not None:
        return widget.maximumHeight() >= UNBOUNDED_HEIGHT
    nested = item.layout()
    return nested is None or nested.maximumSize().height() >= UNBOUNDED_HEIGHT


def has_slack(layout) -> bool:
    """Whether anything in ``layout`` already takes the vertical slack."""
    # `stretch(i)` is a QBoxLayout method; a QFormLayout has no per-item stretch at all, so a
    # form's slack can only come from a child that asks for it.
    stretch_of = getattr(layout, "stretch", None)
    vertical = _lays_out_vertically(layout)
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if vertical and item.spacerItem() is not None:
            return True
        if (
            vertical
            and stretch_of is not None
            and stretch_of(i) > 0
            and _unbounded(item)
        ):
            return True
        # A widget or a nested layout counts whichever way this one runs: a child that grows
        # vertically makes its row grow, and a row that grows takes the column's slack.
        widget = item.widget()
        if widget is not None and can_grow(widget):
            return True
        if item.layout() is not None and has_slack(item.layout()):
            return True
    return False


def set_available(control, available: bool, reason: str) -> None:
    """Enable or disable ``control``, and say why when it is off.

    "Grey out, don't hide" is only half a rule. A control that is present and inert with nothing
    saying *why* reads as broken rather than as not-yet — and the app did both: seven buttons
    explained themselves ("Import a video part first"), twelve said nothing, and they are the
    same kind of disablement.

    One call rather than a ``setEnabled`` and a ``setToolTip`` side by side, because the pair
    separates. It separates especially where the state is recomputed — a selection handler that
    flips ``setEnabled`` on every row change has to remember to flip the reason with it, and the
    twelve silent buttons were all in handlers like that.

    The reason is cleared when the control is available, so an enabled button never carries a
    tooltip explaining a condition that no longer applies.
    """
    control.setEnabled(available)
    control.setToolTip("" if available else reason)


def _hbar_height(listwidget) -> int:
    """The height a horizontal scrollbar will take out of ``listwidget``, or 0.

    The same question `tables._horizontal_scrollbar_height` asks of a table, asked of the widget
    that has no header to measure: does the **content** overflow, not merely is a scrollbar
    allowed. Reserving it on the policy alone over-reserves by a bar's height — 18px here, about
    1.3 rows — which puts the slice of a row back that removing the old `+ 4` had just taken out.

    Two copies of a shape, which this package tolerates; a third is when it becomes one function.
    """
    from PyQt6.QtCore import Qt

    if listwidget.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
        return 0
    model = listwidget.model()
    needed = listwidget.sizeHintForColumn(0) if model and model.rowCount() else 0
    available = listwidget.viewport().width() or listwidget.width()
    if available and needed <= available:
        return 0
    return listwidget.horizontalScrollBar().sizeHint().height()


def fit_list_height(listwidget, *, max_rows: int = 8) -> None:
    """Size a list to its rows, so a four-item list does not open a nine-row void.

    The table twin is :func:`tables.fit_table_height`, and this function was the half of that
    class that never got swept. The table learned that **a display height belongs on the
    minimum** — a maximum is a cap the widget may never reach, and a hint is a suggestion the
    layout may ignore — and its list counterpart went on setting only the maximum.

    What that looks like: an Activity band capped at 202px was handed **192** by the page it sat
    on, which is 13.7 of its 14 rows, so the bottom row rendered sliced in half. Not a missing
    row — a visibly cut one, the same thing a user had already reported about a table ("the row
    is partially cut off"). The maximum was correct throughout and nothing was obliged to honour
    it.

    Three numbers, three jobs, as on the table: the **minimum** is the height it must be given,
    so a squeeze cannot land between rows; the **maximum** is the content height, so a short list
    still hugs its rows. Safe because the content area scrolls (``gui/scroll.py``) — a floor on a
    page that could not scroll would push the window past the display instead.

    ``max_rows`` is a **page's** cap. In a pop-up (:func:`shows_every_row`) the list shows every
    row and the window grows to hold it, up to its own cap; a list the user has dragged
    (:class:`RowsGrip`) keeps the height they gave it.
    """
    if not remember_row_fit(
        listwidget, lambda: fit_list_height(listwidget, max_rows=max_rows)
    ):
        return
    model = listwidget.model()
    count = model.rowCount() if model is not None else 0
    rows = max(count, 1)
    if not shows_every_row(listwidget):
        rows = min(rows, max_rows)
    row_h = listwidget.sizeHintForRow(0) if count else 18
    # Chrome is the frame plus, if the list can scroll sideways, the bar that sits *inside* the
    # height it is given — the table twin's third piece. The spare `+ 4` that used to stand in
    # for both is what left a 4px slice of row 15 showing under a 14-row band: not a missing row
    # but a visibly cut one, which is the complaint this whole helper exists to answer.
    chrome = 2 * listwidget.frameWidth() + _hbar_height(listwidget)
    height = rows * row_h + chrome
    listwidget.setMinimumHeight(height)
    listwidget.setMaximumHeight(height)


#: How wide a column of forms and rows is allowed to get.
#:
#: Not a style preference: a label at one edge of a 1900 px row and its button at the other have no
#: visual connection left, and a table given the same width hands the slack to whichever column
#: happens to be last — on the Model manager that was `Kind`, holding the word `imported`, in about
#: 900 px. Every platform that hit this shipped the same primitive rather than a rule people have to
#: remember: GNOME's `AdwClamp`, UIKit's `readableContentGuide`, the `max-width` container on the
#: web. This is that widget, in the terms this app already uses.
#:
#: 840 px is a form column, not a text measure — the fields here are paths and names rather than
#: prose, so the 50–75 character guidance for body text does not set it. Picked by rendering the
#: same page at 720 / 840 / 960 / 1200 and looking: a folder path still fits without truncating,
#: and a row's two ends stay close enough to read as one object.
CONTENT_WIDTH = 840


def clamp_width(widget, width: int = CONTENT_WIDTH):
    """Cap ``widget`` so it stops growing at a readable width, and return it.

    For **forms and rows**, never for a table or a canvas: a table earns its width by having
    columns to put in it, and capping one would only move the problem into a horizontal scrollbar.
    A table that is drawing empty grid wants `tables.fit_table_width`, which sizes it to its
    columns instead.

    The surplus becomes margin on the right rather than centring the content, because everything
    else on these pages is left-aligned and a centred block would not line up with the heading
    above it.
    """
    widget.setMaximumWidth(width)
    return widget


#: Where a band's remembered size lives, beside the fold state `collapse.py` writes under the
#: same prefix. One place, because three surfaces remember a size and three copies of a
#: `QSettings` key is how two of them end up spelling it differently.
_SIZE_PREFIX = "bands/"


#: A fitted picture's stretch against its column's trailing one: large, so the picture fills to
#: its cap before any spare height goes under the rows below it.
PICTURE_STRETCH = 100


def add_fitted_picture(column: QBoxLayout, block) -> None:
    """Add a player ``block`` so the *layout* picks its height, inside :func:`bound_picture`'s range.

    The three columns that pin a player over its controls — AUDIT's judging pane and the two
    example dialogs — each set the block's height from their own: ``setFixedHeight(min(want,
    self.height() - taken))``. That makes the block's height the column's minimum, so whatever
    sizes the column from its hint hands back the same height plus any rounding, and the next
    refit takes it: AUDIT's clip crept 344px to 630px in 2px steps and the run page scrolled by
    290. With the hint ignored and the height a range, nothing the block does feeds back.
    """
    block.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
    column.addWidget(block, PICTURE_STRETCH)


def bound_picture(block, width: int, floor: int) -> None:
    """At least ``floor``, at most what the picture wants at ``width`` — never taller.

    Given every spare pixel instead, a clip grew to 620px on a wide pane with a 230px gap under
    its transport, and on a laptop pushed the answers off the bottom. Past the cap the picture
    letterboxes inside the block, as any player does.
    """
    block.setMinimumHeight(floor)
    block.setMaximumHeight(max(block.height_for_width(max(width, 1)), floor))


def remembered_size(key: str, default: int) -> int:
    """The size stored for ``key``, or ``default``.

    A dragged size is a view preference the surface cannot pick on the user's behalf — the same
    category as the thumbnail size the repo already persists — and at a hundred recordings it is
    set once and wanted every time after.
    """
    stored = QSettings().value(f"{_SIZE_PREFIX}{key}/size", None)
    try:
        return int(stored)
    except (TypeError, ValueError):
        return default


def remember_size(key: str, value: int) -> None:
    """Store ``value`` as the size for ``key``."""
    QSettings().setValue(f"{_SIZE_PREFIX}{key}/size", int(value))


class ResizeHandle(QFrame):
    """A band's lower edge, dragged to resize it.

    Emits a pixel delta and holds no size of its own — what a drag *means* differs per band (the
    clip strip trades how many members fit against how big they are; the spatial map trades how
    many recordings fit against how well you can read a corner bias), and only the band knows its
    own floor and ceiling.

    Shared rather than copied because the gesture is taught once on a screen that has two of
    them: a band you can drag beside a band you cannot is the thing that gets reported as a bug.
    """

    dragged = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(6)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        # Hover feedback comes from the style, the same way a splitter's does — see paintEvent.
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._press_y: int | None = None

    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        """Draw the grip Qt draws between splitter panes.

        `CE_Splitter` is the primitive `QSplitterHandle` paints itself with, so this is the same
        mark the divider between the 3-D scatter and the cluster table already carries — which
        is what makes a dragged band read as draggable. The alternative was a stylesheet, and
        this package has none by rule.

        It replaces a bare `QFrame.Shape.HLine`: one hairline centred in 6px, indistinguishable
        from the separators that are *not* draggable, which is how a resizable band and a fixed
        one came to look the same.
        """
        option = QStyleOption()
        option.initFrom(self)
        # **No `State_Horizontal`.** Qt's flag names the splitter's *orientation*, not the
        # handle's: a horizontal splitter puts panes side by side and its handle is a vertical
        # bar. This one divides top from bottom, so the flag is off and the grip lies flat.
        # Set, it drew a column of dots across a 6px-tall bar.
        if self._press_y is not None:
            option.state |= QStyle.StateFlag.State_Sunken
        painter = QPainter(self)
        self.style().drawControl(
            QStyle.ControlElement.CE_Splitter, option, painter, self
        )
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._press_y = int(event.globalPosition().y())
        self.update()  # the grip draws sunken while held

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._press_y is None:
            return
        delta = int(event.globalPosition().y()) - self._press_y
        self._press_y = int(event.globalPosition().y())
        if delta:
            self.dragged.emit(delta)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._press_y = None
        self.update()


# --------------------------------------------------------------------------- #
# a pop-up's size: its content, up to most of the screen
# --------------------------------------------------------------------------- #
#: The most of its screen's free area a pop-up opens at when its content is larger than that.
#: The rest scrolls. Not the whole screen: a window edge-to-edge over the one it came from hides
#: where it came from, and a title bar pushed to the very top is one the user cannot grab.
SCREEN_SHARE = 0.9

#: How many measure-and-grow passes a fit may take. A pass grows by what is still unseen; a
#: layout that shares the extra between two growing panes needs a second, and nothing measured
#: needed more than three.
_FIT_PASSES = 6


def fit_screen(window):
    """The screen ``window`` opens on: its parent's, or its own, or the primary one."""
    parent = window.parentWidget()
    if parent is not None and parent.window().screen() is not None:
        return parent.window().screen()
    return window.screen() or QApplication.primaryScreen()


def _frame_extent(window) -> QSize:
    """The title bar and borders the window system adds around ``window``'s client area."""
    handle = window.windowHandle()
    if handle is None:
        return QSize(0, 0)
    margins = handle.frameMargins()
    return QSize(margins.left() + margins.right(), margins.top() + margins.bottom())


def window_cap(window) -> QSize:
    """The largest client area ``window`` may open at: :data:`SCREEN_SHARE` of its screen's
    free area, less the frame the window system draws around it."""
    screen = fit_screen(window)
    if screen is None:
        return QSize(UNBOUNDED_HEIGHT, UNBOUNDED_HEIGHT)
    free = screen.availableGeometry()
    frame = _frame_extent(window)
    return QSize(
        int(free.width() * SCREEN_SHARE) - frame.width(),
        int(free.height() * SCREEN_SHARE) - frame.height(),
    )


def _room_to_grow(widget) -> tuple[bool, bool]:
    """Whether ``widget`` could take more width, more height, if the window gave it some."""
    return (
        widget.maximumWidth() > widget.width(),
        widget.maximumHeight() > widget.height(),
    )


def hidden_extent(area) -> tuple[int, int]:
    """How many **pixels** of ``area``'s content are scrolled out of view, across and down.

    A scroll bar's range is pixels for most scroll areas and **items** for an item view in its
    default ``ScrollPerItem`` mode — a table with eighteen rows hidden reports 18, and a window
    grown by 18px shows less than one of them. A table answers exactly from its headers; any
    other item view from the size of its first row or column.
    """
    from PyQt6.QtWidgets import QAbstractItemView, QTableView

    across = area.horizontalScrollBar().maximum()
    down = area.verticalScrollBar().maximum()
    if not isinstance(area, QAbstractItemView):
        return across, down
    viewport = area.viewport()
    if isinstance(area, QTableView):
        return (
            (
                max(0, area.horizontalHeader().length() - viewport.width())
                if across
                else 0
            ),
            max(0, area.verticalHeader().length() - viewport.height()) if down else 0,
        )
    per_item = QAbstractItemView.ScrollMode.ScrollPerItem
    if down and area.verticalScrollMode() == per_item:
        down *= max(area.sizeHintForRow(0), 1)
    if across and area.horizontalScrollMode() == per_item:
        across *= max(area.sizeHintForColumn(0), 1)
    return across, down


def unseen(window) -> QSize:
    """How much larger ``window`` would have to be to show everything in it unscrolled.

    Three kinds of content, one answer each, and the largest per axis wins this pass:

    - **The window's own scroll** (a `Dialog`'s `PageScroll`): the height the body asks for
      past what the viewport shows.
    - **Any other scroll area in it** — a table, a list, a text box — by its scroll range, on an
      axis it can grow on. A list sized to its rows (`fit_list_height`) has a maximum height
      and so says nothing: growing the window would not show it one more row.
    - **A picture** — any widget with a ``natural_size()`` (`ImageView`, the field-of-view
      editor's `ArenaCanvas`): that size — the frame at its own pixels, or what a host composing
      to fit declared — past the pane it has. A pane already at that size asks for nothing.

    Read off the laid-out window, so it is only true after a layout pass at the current size.
    """
    from PyQt6.QtWidgets import QAbstractScrollArea, QWidget

    from detail.gui.scroll import PageScroll

    need_w = need_h = 0
    for area in window.findChildren(QAbstractScrollArea):
        if not area.isVisibleTo(window):
            continue
        if isinstance(area, PageScroll):
            body = area.widget()
            if body is not None and area.parentWidget() is window:
                need_h = max(need_h, area.verticalScrollBar().maximum())
                need_w = max(need_w, body.sizeHint().width() - area.viewport().width())
            continue
        wider, taller = _room_to_grow(area)
        across, down = hidden_extent(area)
        off = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        if wider and area.horizontalScrollBarPolicy() != off:
            need_w = max(need_w, across)
        if taller and area.verticalScrollBarPolicy() != off:
            need_h = max(need_h, down)
    for view in window.findChildren(QWidget):
        natural_size = getattr(view, "natural_size", None)
        if natural_size is None or not view.isVisibleTo(window):
            continue
        natural = natural_size()
        if natural is None:
            continue
        need_w = max(need_w, natural.width() - view.width())
        need_h = max(need_h, natural.height() - view.height())
    return QSize(max(need_w, 0), max(need_h, 0))


def announce_natural_size(widget) -> None:
    """Tell ``widget``'s window that its picture now wants more room than it last said.

    For a picture whose frame arrives after the window opened — decoded off the UI thread, as
    every clip and frame view's is. Duck-typed on the window's ``content_grew``
    (`gui.dialog.Dialog`'s): a page's window has none, which is right, since a pane on a page
    is sized by the page. A frame no larger than the last (every frame of a clip) costs one
    comparison.
    """
    natural = widget.natural_size()
    if natural is None:
        return
    before = getattr(widget, "_announced_natural_size", None)
    if before is not None and natural.boundedTo(before) == natural:
        return
    widget._announced_natural_size = QSize(natural)
    grew = getattr(widget.window(), "content_grew", None)
    if grew is not None:
        grew()


def _apply(window, size: QSize) -> None:
    """Resize ``window`` and run its layout now, so :func:`unseen` reads the new geometry."""
    window.resize(size)
    QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
    layout = window.layout()
    if layout is not None:
        layout.activate()


def fit_to_content(window, *, grow_only: bool) -> None:
    """Size ``window`` to show all of its content, capped at :func:`window_cap`.

    Starts from the window's own hint (a `Dialog`'s is its body unscrolled), or from where it is
    with ``grow_only``; then measures what is still :func:`unseen` and grows by it, pass by
    pass. **An axis that did not help is given back**: content that cannot use the room — a
    picture already letterboxed by its other side, a pane with a maximum — would otherwise turn
    the cap into the size of every window that holds one.

    ``grow_only`` is for a window already on screen: content arriving late (a frame decoded, a
    list filled) can make it larger, never smaller.
    """
    # Hints are cached per layout and refreshed when its LayoutRequest is handled: read before
    # that, a hint can describe the window as it was a moment ago (a list just refitted, a grip
    # just added) and open it that size.
    QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
    cap = window_cap(window)
    floor = window.minimumSizeHint().expandedTo(window.minimumSize())
    wanted = window.sizeHint().boundedTo(cap)
    # Growing never shrinks: a window already past the cap (put there by the program, or
    # dragged by a user the caller did not know about) keeps its size.
    size = (window.size().expandedTo(wanted) if grow_only else wanted).expandedTo(floor)
    if not grow_only or size != window.size():
        _apply(window, size)
    for _ in range(_FIT_PASSES):
        need = unseen(window)
        grown = QSize(
            min(size.width() + need.width(), max(cap.width(), size.width())),
            min(size.height() + need.height(), max(cap.height(), size.height())),
        )
        if grown == size:
            return
        _apply(window, grown)
        after = unseen(window)
        kept = QSize(
            grown.width() if after.width() < need.width() else size.width(),
            grown.height() if after.height() < need.height() else size.height(),
        )
        if kept != grown:
            _apply(window, kept)
        if kept == size:
            return
        size = kept


def fit_height(window, wanted: int) -> None:
    """Set ``window``'s height to ``wanted`` — what its content measured it needs at its width
    (`Dialog.content_height`) — capped at :func:`window_cap`, never under its minimum. The width
    is left where it is."""
    height = max(
        min(wanted, window_cap(window).height()), window.minimumSizeHint().height()
    )
    if height != window.height():
        _apply(window, QSize(window.width(), height))


def centre_on_screen(window, *, about=None) -> None:
    """Put ``window`` centred on ``about`` (a point; the parent window's centre by default,
    else its screen's), then wholly inside its screen's free area."""
    screen = fit_screen(window)
    if screen is None:
        return
    free = screen.availableGeometry()
    frame = _frame_extent(window)
    width = window.width() + frame.width()
    height = window.height() + frame.height()
    if about is None:
        parent = window.parentWidget()
        if parent is not None and parent.window().isVisible():
            about = parent.window().frameGeometry().center()
        else:
            about = free.center()
    x = min(max(about.x() - width // 2, free.left()), free.right() + 1 - width)
    y = min(max(about.y() - height // 2, free.top()), free.bottom() + 1 - height)
    window.move(max(x, free.left()), max(y, free.top()))


# --------------------------------------------------------------------------- #
# a list in a pop-up: every row, and an edge the user can drag
# --------------------------------------------------------------------------- #
#: Set on a view the user has dragged to a height of their own (:class:`RowsGrip`). The row fit
#: leaves it alone from then on, for as long as the window is open: content arriving later does
#: not undo a size somebody chose, the same rule a dragged window keeps.
_DRAGGED = "_detail_rows_dragged"

#: Where a row fit leaves a way to run itself again (:func:`refit_rows`). A list sized while it
#: was still being built — before it was in its dialog — was sized as a page's would be.
_REFIT = "_detail_fit_rows"


def shows_every_row(widget) -> bool:
    """Whether ``widget`` is in a window whose lists and tables show every row.

    A **pop-up's** (`gui.dialog.Dialog.SHOWS_EVERY_ROW`): it is sized to its content up to most
    of the screen (:func:`fit_to_content`), so a list capped at a few rows only hides what the
    window had room to show. A **page** keeps its cap — it shares the window with everything
    else on it, and a page of several lists each as tall as its rows is a page of scrolling.
    """
    return bool(getattr(widget.window(), "SHOWS_EVERY_ROW", False))


def remember_row_fit(view, fit) -> bool:
    """Keep ``fit`` (a call that re-sizes ``view`` to its rows) for :func:`refit_rows`, and say
    whether it should run now — not once the user has dragged the view."""
    setattr(view, _REFIT, fit)
    return not getattr(view, _DRAGGED, False)


def refit_rows(window, *, forget_drags: bool = False) -> None:
    """Run every list's and table's row fit again, now that they are where they will be shown.

    A fit run in a constructor, before the view was in its dialog, applied a page's cap; run
    again, :func:`shows_every_row` is true and the cap is gone. ``forget_drags`` is for a window
    opening afresh: a height dragged in an earlier open is not remembered.
    """
    from PyQt6.QtWidgets import QAbstractItemView, QListView, QTableView

    from detail.gui.tables import fit_table_height

    for view in window.findChildren(QAbstractItemView):
        if view.window() is not window:
            continue  # a combo box's drop-down, in its own popup window
        if forget_drags:
            setattr(view, _DRAGGED, False)
        fit = getattr(view, _REFIT, None)
        if fit is not None:
            fit()
        elif isinstance(view, QTableView):
            # Nobody sized it: on a page that is `a_table_is_sized_to_its_rows_or_declared_to_
            # fill`'s business, and a table declared to fill may; in a pop-up every table
            # shows its rows, so it gets the fit it would have had.
            fit_table_height(view)
        elif isinstance(view, QListView):
            fit_list_height(view)


def _layout_holding(layout, widget):
    """The layout under ``layout`` (itself or nested) whose item is ``widget``, or ``None``."""
    if layout is None:
        return None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is widget:
            return layout
        found = _layout_holding(item.layout(), widget)
        if found is not None:
            return found
    return None


#: Where a view keeps its :class:`RowsGrip`, so it gets one and only one, and so a check can ask.
_GRIP = "_detail_row_grip"


class RowsGrip(ResizeHandle):
    """The lower edge of a list or table in a pop-up, dragged to make the view taller or shorter.

    Every list and table in a pop-up gets one (:func:`attach_row_grips`). The window is not
    resized by the drag: the view's height changes, the dialog's body grows or shrinks and
    scrolls if it has to, and the window stops re-fitting itself for the rest of that open,
    since the user has now chosen its layout. Nothing is remembered between opens.

    A :class:`ResizeHandle`, the one band edge this package draws, so a list's edge reads the
    same as the clip strip's and the spatial map's. It sits **beside** the view, in the view's
    own layout, rather than around it: a container would move the view out of the layout that
    placed it, and code that addresses it there — ``QFormLayout.setRowVisible(view, …)``, which
    the distill form does — would silently stop finding it.
    """

    def __init__(self, view) -> None:
        super().__init__()
        self.view = view
        self.setToolTip("Drag to make the list taller or shorter")
        self.setVisible(not view.isHidden())
        self.dragged.connect(self._drag)
        # Shown and hidden with its view, however the view is shown or hidden.
        view.installEventFilter(self)
        view.destroyed.connect(self.deleteLater)
        setattr(view, _GRIP, self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        if cleared(self):
            return False
        if watched is self.view:
            if event.type() == QEvent.Type.HideToParent:
                self.hide()
            elif event.type() == QEvent.Type.ShowToParent:
                self.show()
        return False

    def _drag(self, delta: int) -> None:
        view = self.view
        model = view.model()
        row = view.sizeHintForRow(0) if model is not None and model.rowCount() else 0
        floor = 2 * view.frameWidth() + max(row, lines(1))
        height = max(floor, view.height() + delta)
        setattr(view, _DRAGGED, True)
        view.setMinimumHeight(height)
        view.setMaximumHeight(height)
        window = self.window()
        if hasattr(window, "content_grew"):
            window._user_sized = True


def row_grip(view):
    """``view``'s :class:`RowsGrip`, or ``None``."""
    from detail.gui import lifetime

    grip = getattr(view, _GRIP, None)
    return grip if grip is not None and lifetime.alive(grip) else None


def attach_row_grips(window) -> None:
    """Give every list and table in ``window`` a :class:`RowsGrip`, once, under it.

    Into the layout that holds the view, directly beneath it: the next item of a box layout,
    or a row of its own under the view's in a form. So no dialog has to remember to — the
    reason this lives on the base rather than at forty call sites. Not given one: a combo box's
    drop-down (its own popup window), and a view in a ``QSplitter``, whose divider already
    resizes it.
    """
    from PyQt6.QtWidgets import QAbstractItemView, QFormLayout, QSplitter

    for view in window.findChildren(QAbstractItemView):
        parent = view.parentWidget()
        if view.window() is not window or parent is None or row_grip(view) is not None:
            continue
        if isinstance(parent, QSplitter):
            continue
        holder = _layout_holding(parent.layout(), view)
        if isinstance(holder, QBoxLayout):
            holder.insertWidget(holder.indexOf(view) + 1, RowsGrip(view))
        elif isinstance(holder, QFormLayout):
            row, role = holder.getWidgetPosition(view)
            if role == QFormLayout.ItemRole.SpanningRole:
                holder.insertRow(row + 1, RowsGrip(view))
            elif role == QFormLayout.ItemRole.FieldRole:
                holder.insertRow(row + 1, QWidget(), RowsGrip(view))


class ElidedLabel(QLabel):
    """A one-line label that shortens its text with "…" to the width it is given, and says the
    whole line in its tooltip while it is shortened.

    A plain ``QLabel`` cut short by its layout is clipped mid-letter with nothing to say so: the
    label bar's figures read "9.6% of the run · sepa" and ran into the chip beside them. Its
    minimum is a few characters, so a row can give it what is left rather than being widened
    by it; :meth:`text` is always the whole line.
    """

    #: How much of the line survives at the narrowest: enough to see that text is there.
    MIN_CHARS = 6

    def __init__(
        self, text: str = "", parent=None, *, min_chars: int | None = None
    ) -> None:
        super().__init__(parent)
        self._full = ""
        self._tip = ""
        #: This label's floor, when a row should cut its neighbour first (a filter's summary
        #: beside the field's name); a constant either way, never the text's own width.
        self._min_chars = self.MIN_CHARS if min_chars is None else min_chars
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt's name
        self._full = text or ""
        self.updateGeometry()
        self._elide()

    def text(self) -> str:
        """The whole line, however much of it is shown."""
        return self._full

    def shown_text(self) -> str:
        """What is drawn: the line, or its start and "…"."""
        return super().text()

    def is_elided(self) -> bool:
        return self.shown_text() != self._full

    def full_width(self) -> int:
        """The width the whole line needs."""
        return self.sizeHint().width()

    def set_extra_tooltip(self, tip: str) -> None:
        """A tooltip of the label's own, shown under the whole line while that is cut."""
        self._tip = tip
        self._elide()

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        hint = super().sizeHint()
        width = self.fontMetrics().horizontalAdvance(self._full) + 2
        return QSize(width + 2 * self.margin(), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        width = self.fontMetrics().horizontalAdvance("0" * self._min_chars + "…")
        full = self.fontMetrics().horizontalAdvance(self._full) + 2
        return QSize(min(width, full), super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        room = max(self.contentsRect().width() - 2 * self.margin(), 1)
        shown = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, room
        )
        if shown != super().text():
            super().setText(shown)
        cut = shown != self._full
        lines_ = [self._full] if cut else []
        if self._tip:
            lines_.append(self._tip)
        self.setToolTip("\n".join(lines_))
