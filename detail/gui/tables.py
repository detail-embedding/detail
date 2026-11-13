"""How a list table is built and sized — the rule, in a leaf both hubs and leaves can reach.

``fit_table_height`` used to live in ``gui/pages.py``. That is a 5000-line hub whose import
pulls in the module registry, so a leaf could not call it (``tests/gui/test_import_cycles.py``
— both of this package's shipped import cycles formed exactly that way), and the audit tab
duly stretched its summary tables to fill the pane instead. One behavior and one recording
rendered as two rows above ~700 px of empty grid, with the panel below pushed off screen.

The sizing rule is tool-wide (``docs/gui_design.md`` §3), so it belongs where everything can
reach it rather than being restated at each call site. Same move, same reason, as
``gui/status_color.py``.
"""

from PyQt6.QtCore import QEvent, QModelIndex, QObject, QSize, Qt
from PyQt6.QtGui import QStandardItem, QStandardItem
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from detail.gui.lifetime import cleared

#: The trailing glyph marking a row that opens something. Text, not an icon set.
NAV_GLYPH = "\u203a"

#: The role a row's own identity is stamped under (column 0), so a click handler can
#: recover it without trusting the row's *position* in the view.
ROW_ID_ROLE = Qt.ItemDataRole.UserRole


def stamp_row_id(item: QStandardItem, key) -> None:
    """Attach ``key`` \u2014 the record this row is *about* \u2014 to its own model item.

    The class this exists for: **a click handler that recovered identity from view
    position** (``self._ids[index.row()]``), which is only correct while nothing can
    reorder the view. Once a table sorts, ``index.row()`` is a screen position and
    ``self._ids`` is still in insertion order \u2014 the two agree by coincidence, not by
    construction, and every double-click after a sort opens the wrong row. Stamping the
    key onto column 0's item survives a sort because ``QStandardItemModel.sort()`` moves
    whole rows, item data included \u2014 the pattern ``cluster_workspace.py``'s ``_show_row``
    already used by hand for one table. Read it back with :func:`row_id`.

    ``QStandardItem`` only. ``QTableWidgetItem.setData`` takes its arguments in the
    **opposite order** (``setData(role, value)`` vs. ``QStandardItem``'s
    ``setData(value, role)``) \u2014 a real Qt inconsistency, not a typo \u2014 so a
    ``QTableWidget`` site stamps identity with ``item.setData(ROW_ID_ROLE, key)`` directly
    instead of calling this. :func:`row_id` reads either back the same way: ``.data(role)``
    takes only the role on both.
    """
    item.setData(key, ROW_ID_ROLE)


def row_id(index: QModelIndex):
    """The identity :func:`stamp_row_id` attached to ``index``'s row, or ``None``.

    Reads column 0 of ``index``'s row rather than ``index`` itself, so a handler
    connected to any column's cell (not just the first) still resolves the row's key.
    """
    model = index.model()
    if model is None:
        return None
    return model.index(index.row(), 0, index.parent()).data(ROW_ID_ROLE)


def make_columns_resizable(table) -> None:
    """Let the user drag column borders to resize them.

    Every table in the package used to hard-code ``ResizeMode.ResizeToContents``
    (or ``Stretch`` on one column), and neither lets a user drag a border \u2014 the only
    two resize modes anywhere in ``detail/gui/`` before this. ``Interactive`` still
    starts each column sized to its content (:meth:`QTableView.resizeColumnsToContents`
    below does that once, the same starting widths ``ResizeToContents`` gave); the
    difference is a user can then widen or narrow one. To hand one column the spare width,
    call :func:`give_slack_to` (any column) or :func:`fill_width` (the last) afterwards;
    never ``setSectionResizeMode(col, Stretch)``, which refuses a drag on that column and
    makes every column to its right grow the wrong way (see :func:`give_slack_to`).
    """
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.resizeColumnsToContents()


class _Slack(QObject):
    """One table's slack column, kept filling the viewport until a user sizes a column."""

    def __init__(self, table, column: int) -> None:
        super().__init__(table)
        self._table = table
        self.column = column
        self._width = -1
        self._fitting = False
        self._fitted = False
        table.viewport().installEventFilter(self)
        table.horizontalHeader().sectionResized.connect(self._on_section_resized)

    def fit(self) -> None:
        header = self._table.horizontalHeader()
        self._fitted = True
        if self.column == SPREAD:
            self._spread()
            return
        if self.column >= header.count():
            return
        others = sum(
            header.sectionSize(c) for c in range(header.count()) if c != self.column
        )
        room = self._table.viewport().width() - others
        self._fitting = True
        try:
            header.resizeSection(
                self.column, max(header.sectionSizeHint(self.column), room)
            )
        finally:
            self._fitting = False

    def _spread(self) -> None:
        """Every column its content width plus an equal share of the spare width."""
        table, header = self._table, self._table.horizontalHeader()
        count = header.count()
        if not count:
            return
        content = [
            max(header.sectionSizeHint(c), table.sizeHintForColumn(c))
            for c in range(count)
        ]
        room = max(table.viewport().width() - sum(content), 0)
        share, rest = divmod(room, count)
        self._fitting = True
        try:
            for c, width in enumerate(content):
                header.resizeSection(c, width + share + (rest if c == count - 1 else 0))
        finally:
            self._fitting = False

    def _on_section_resized(self, *_args) -> None:
        # Any resize that is not this fit — a user's drag, or a resize to contents that a
        # `refit_slack` will follow — means the widths are no longer the fitted ones.
        if not self._fitting:
            self._fitted = False

    def eventFilter(self, watched, event) -> bool:
        # Only while still fitted. After a drag the widths are the user's: widening a column
        # past the view brings a horizontal scrollbar, the rows then need a vertical one, the
        # viewport narrows by its width — and re-fitting on that took back 60px the user had
        # just dragged into the columns (measured, on the Measures table).
        # `Show` as well as `Resize`: a viewport resized again while hidden gets no second
        # `Resize` (measured: one at 682px, then shown at 698 with none).
        if cleared(self):
            return False
        kinds = (QEvent.Type.Resize, QEvent.Type.Show)
        if event.type() in kinds and watched.width() != self._width:
            self._width = watched.width()
            if self._fitted:
                self.fit()
        return False


def give_slack_to(table, column: int) -> None:
    """Hand ``column`` the table's spare width while every border stays draggable.

    The ``Stretch`` resize mode does the first half and breaks the second twice over. A
    ``Stretch`` section refuses a drag outright, and it holds the header's length to the
    viewport's, so widening any column to its **right** takes the width back out of the
    stretched one: the dragged border stays where it was and the column grows leftward, away
    from the cursor. Measured on the clustering measurements table, whose name column stretched:
    dragging a border 60px right left that border where it was and moved the column's left edge
    60px left. Stretching only the *last* section (:func:`fill_width`) has neither problem,
    which is why this is only for a column with others after it.

    This leaves every column ``Interactive`` and sets ``column`` to exactly the room the others
    leave, as ``Stretch`` did (eliding its text rather than growing a scrollbar), but never
    narrower than its header — and keeps it so as the viewport's width changes, **until a user
    drags a border**. From then on the widths are theirs, as in a spreadsheet: a column dragged
    wider pushes the rest right, a scrollbar appears once they no longer fit, and a window
    resize moves nothing. :func:`refit_slack` (the rows changed) fits it again.

    Call it again to move the slack to another column; the previous one goes back to its
    content width.
    """
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    if column >= 0:
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
    slack = getattr(table, "_slack", None)
    if slack is None:
        slack = table._slack = _Slack(table, column)
    elif slack.column != column:
        previous = slack.column
        slack.column = column
        if 0 <= previous < header.count():
            table.resizeColumnToContents(previous)
    slack.fit()


#: :func:`give_slack_to`'s column for "share it out across every column" (:func:`spread_slack`).
SPREAD = -1


def spread_slack(table) -> None:
    """Share the table's spare width out **evenly across every column**, each starting from its
    content width; borders stay draggable, and a drag makes the widths the user's, exactly as
    :func:`give_slack_to`.

    For a wide table whose columns are all short, where handing the slack to one column opens a
    gulf in the middle of every row: the Measures table at 1440px gave Label 678px and left
    Measure, Value and Unit at their content widths, the numbers a screen away from the names
    they belong to.
    """
    give_slack_to(table, SPREAD)


def refit_slack(table) -> None:
    """Re-fit a :func:`give_slack_to` column after the rows changed; a no-op without one.

    For a fill that resizes columns to their contents, which shrinks the slack column back to
    its own content and leaves the spare width as dead grid on the right.
    """
    slack = getattr(table, "_slack", None)
    if slack is not None:
        slack.fit()


def enable_sorting(table) -> None:
    """Turn on click-a-header-to-sort, starting from the table's own (unsorted) order.

    ``docs/gui_design.md`` §3.1 has required sortable headers since before this
    function existed; ``setSortingEnabled`` was never actually called except by two
    hand-rolled tables. Call this **after** the model has its rows \u2014 enabling sorting
    on an empty table is harmless but pointless, and calling it before populating means
    every subsequent insert pays for a re-sort.

    **The class this fixes: a table that reorders itself before anyone touches it.**
    A ``QHeaderView`` that has never had ``setSortIndicator`` called reports section
    **0** anyway, not "no column" \u2014 there is no such state by default. Qt's own docs say
    plainly that turning sorting on "immediately triggers a call to sortByColumn() with
    the current sort section and sort order", so a bare ``setSortingEnabled(True)`` on a
    fresh table silently re-sorts every row by column 0, **descending, as text** \u2014 a run
    table's newest-first order, a query's ranked order, a registry's declared order, all
    quietly replaced by column-0-descending the moment this was called, with no user
    interaction and nothing to make it visible except the row order being wrong. Caught
    here (rather than shipped) because one call site's own regression test asserted an
    id-based row still where it was placed. Clearing the indicator to "no section" first
    is the documented way to ask for none, and only a user's own header click sets a real
    one after that.

    Enabling this on a table whose click/selection handler reads identity from a row's
    *position* (``self._ids[index.row()]``) is a correctness bug once a user sorts \u2014
    see :func:`stamp_row_id` / :func:`row_id`, which is the fix. A numeric or time
    column sorting on its display text (`"10.0 s"` before `"9.0 s"`) needs
    :class:`SortCell` instead of a plain ``QStandardItem``.
    """
    # Order matters: enabling sorting itself performs the initial sort, so the indicator
    # must already read "no column" *before* that call, not after.
    table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    table.setSortingEnabled(True)


def refill_table_widget(table, fill) -> None:
    """Run ``fill()`` — a bulk repopulation of a sortable ``QTableWidget`` — with sorting
    suspended, then restore it.

    The class this exists for: **``QTableWidget`` resorts itself after every single
    ``setItem`` call** while ``isSortingEnabled()`` is true (Qt's own docs warn of exactly
    this: adding items while sorting is enabled can move a row out from under you). A
    rebuild loop that writes one row's columns one at a time —
    ``for r, sid in enumerate(ids): table.setItem(r, 0, key); ...; table.setItem(r, c,
    value)`` — assumes row ``r`` in the widget stays row ``r`` until the loop finishes
    writing it. With live sorting, the first ``setItem`` in a row can trigger a resort
    that moves row ``r`` elsewhere before the rest of that row's cells are written,
    scrambling which cells land in which row. Not needed for a ``QTableView`` over a
    ``QStandardItemModel`` (``pages._table`` / ``tables.list_table``): that model only
    resorts on an explicit ``sort()`` call (wired to the header by
    :func:`enable_sorting`), never as a side effect of ``setItem``.
    """
    was_sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    try:
        fill()
    finally:
        table.setSortingEnabled(was_sorting)


#: Roughly what a page spends on everything that is not the table — banner, heading, caption,
#: the action button under it, and the margins between. Measured on the collection pages, which
#: all have the same furniture.
_PAGE_CHROME_PX = 220


def max_visible_rows(row_height: int) -> int:
    """How many rows fit on *this* display, above the page's own furniture.

    It was a constant 16, chosen as what fits a 13" laptop. Two things were wrong with that, in
    opposite directions: on a tall window a 60-row table showed 16 and left 700px of empty page
    underneath, and the number could never be right on a display it was not picked on.

    Unbounded is not the answer either — it was tried. A 60-row `MetadataPage` asked for 1869px
    inside a 907px window, and since the page scrolls rather than the table, the **header scrolls
    away with it**: you lose the column names exactly when you have enough rows to need them.

    So the budget is the display, which is the thing that actually knows. A 13" laptop gets ~20
    rows, a desktop monitor ~40, and either way the table keeps its header and scrolls its own
    content past that.
    """
    from PyQt6.QtWidgets import QApplication

    screen = QApplication.primaryScreen()
    available = screen.availableGeometry().height() if screen is not None else 900
    return max(1, (available - _PAGE_CHROME_PX) // max(row_height, 1))


#: Rows a clip list shows before it scrolls — the two Browse tabs, cluster and behavior-scan,
#: and the audit tab's own walk list, each with a player beside it. Past this the list is
#: taller than the player it drives.
BROWSE_ROWS = 20


#: How few rows a table may be squeezed to before it stops giving ground.
#:
#: The floor is what stops a table being the only compressible thing on a page and absorbing all
#: of it — measured, a 20-row table laid out at one row beside a plot that yielded nothing. A
#: table that sits *beside* a fill element rather than under one passes ``min_rows=0``: there the
#: splitter decides the share, and a floor only pushes the whole pane's minimum up.
MIN_VISIBLE_ROWS = 4


def _horizontal_scrollbar_height(table) -> int:
    """The height a horizontal scrollbar will take out of ``table``, or 0.

    Chrome was the header and the frame, and a table that scrolls sideways has a third piece:
    the scrollbar sits *inside* the height it is given and eats a row. On a wide table that is
    the last row, every time — measured on Recording information at five recordings, 171px of
    table with 18px of scrollbar, so five rows of data had 4.4 rows of room and the fifth was
    always cut off. The table hugged its rows correctly and still could not show them.

    Asked of the columns rather than of the scrollbar's current visibility, because this runs
    while the table is being built: the bar has not been laid out yet and reports hidden, which
    is the answer that produces the bug.
    """
    from PyQt6.QtCore import Qt

    if table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
        return 0
    header = table.horizontalHeader()
    needed = sum(header.sectionSize(c) for c in range(header.count()))
    available = table.viewport().width() or table.width()
    if available and needed <= available:
        return 0
    return table.horizontalScrollBar().sizeHint().height()


def fit_table_height(
    table,
    *,
    fixed: bool = False,
    max_rows: int | None = None,
    min_rows: int = MIN_VISIBLE_ROWS,
    fill: bool = False,
) -> None:
    """Size a table to its rows, up to what the display can show, then scroll its own content.

    ``max_rows=None`` (the default) asks :func:`max_visible_rows` — the number of rows that fit
    on this screen above the page's own furniture. That replaced a constant 16, which was too
    few on a big monitor and could never have been right on a display it was not chosen on.

    The class this fixes: **a cap the table can never reach.** This function has always set the
    *maximum* height to the content height, but never the **preferred** one — and
    ``QTableView.sizeHint()`` is a constant **192px ≈ 5 rows whatever the row count**. With a
    ``Preferred`` policy the layout hands a widget exactly its hint, so every list in the package
    showed at most five rows: measured on a 60-run Analysis page, 5 of 60 with the cap allowing
    1819px and 723px of empty page underneath it. Nothing was squeezing it; it simply never asked.

    So the table's own **preferred** height carries the display height — an instance
    ``sizeHint`` — and the minimum is only how far it may be squeezed. The minimum carried it
    first, and one number serving as both pushed a pane's floor up by the table's whole display
    height (the comment at the assignment below has the measurement).

    The maximum stays the content height, so a short table still hugs its rows rather than
    drawing blank grid below them, which is the failure this function originally existed for.

    A deliberate consequence: a long table no longer collapses under pressure. It used to be the
    lowest floor on an over-budget page and so absorbed *all* of the compression — measured, a
    20-row Measures table whose cap allowed 619px laid out at 70px, one row, while a fixed-height
    plot beside it gave up nothing. It now holds its ground and the page scrolls instead, which is
    only safe because it does (``gui/scroll.py``); this must not be reintroduced without that.

    ``fixed=True`` for a table with a **bounded** row count that must never scroll — a top-N
    panel, where a scrollbar means the panel is not showing its own N. It gets exactly its
    content height and competes for nothing.

    ``fill=True`` for a table that is its page's primary content and is added to the page's
    layout **with a stretch factor**: it asks only for its floor, and the stretch hands it the
    rest of the *window*, up to its rows. Sized to the *screen* instead, a 60-row Recording
    information table was 1213px in a 602px pane — the page scrolled rather than the table, and
    the table's horizontal scrollbar, the only way to its right-hand columns, sat at its bottom
    edge 600px below the fold. A table that scrolls on both axes has to fit the pane it is in.

    **Call it once before the table is first laid out** — on the empty table, where it is
    built — and again whenever its rows change. The hint is a method assigned on the instance,
    and sip looks for a Python override of a C++ virtual only until the first time C++ asks:
    finding none, it caches that answer for the instance, and every later assignment is
    invisible to the layout. Measured on Browse, whose lists were first sized after they had been
    shown: ``sizeHint()`` read 618px from Python and 192 from the layout, and thirty instances
    showed as five rows.

    **In a pop-up** (`sizing.shows_every_row`) there is no row budget: the table shows every row
    and the window grows to hold it, up to its own cap. A table the user has dragged
    (`sizing.RowsGrip`) keeps the height they gave it, and this returns without touching it.
    """
    from detail.gui.sizing import remember_row_fit, shows_every_row

    if not remember_row_fit(
        table,
        lambda: fit_table_height(
            table, fixed=fixed, max_rows=max_rows, min_rows=min_rows, fill=fill
        ),
    ):
        return
    model = table.model()
    rows = model.rowCount() if model is not None else 0
    row_height = table.verticalHeader().defaultSectionSize()
    chrome = table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth()
    chrome += _horizontal_scrollbar_height(table)
    content = chrome + rows * row_height
    budget = max_visible_rows(row_height) if max_rows is None else max_rows
    if shows_every_row(table):
        # A pop-up shows every row and grows to hold them (`sizing.shows_every_row`); the
        # budget above is a page's.
        budget = rows
    shown = content if fixed else min(content, chrome + budget * row_height)
    floor = shown if fixed else min(shown, chrome + min_rows * row_height)
    if shows_every_row(table):
        # Every row as the *floor*, not only the hint: the hint may already have been read
        # (see below), and a pop-up's body scrolls, so a floor here costs nothing but scrolling.
        floor = shown
    if fill:
        shown = floor

    table.setMaximumHeight(content)
    table.setMinimumHeight(floor)
    # The **preferred** height, which is the whole point and cannot be expressed any other way:
    # `QTableView.sizeHint()` is a constant 192px — about five rows — whatever the table holds,
    # so a `Preferred` widget is handed five rows however many it has. Assigning the hint on the
    # instance is what PyQt's virtual dispatch reads, and it works the same on the `QTableView`s
    # and the `QTableWidget`s this is called on (checked on both).
    #
    # Carried here rather than on the minimum, which is where it started: the minimum is how far
    # a table may be *squeezed*, and using one number for both meant a table beside a fill
    # element pushed its whole pane's floor up by its own display height.
    width = table.sizeHint().width()
    table.sizeHint = lambda: QSize(width, shown)
    # `Preferred`, not `Expanding`. A capped-but-still-expanding table keeps *asking* for slack it
    # cannot use, and the container it sits in takes the slack on its behalf: measured on a
    # three-row `IndexTable`, an expanding view left its wrapper 333px tall around 109px of rows —
    # 224px of empty grey. The minimum above is what makes the table tall enough; the policy only
    # has to stop it demanding more.
    table.setSizePolicy(
        table.sizePolicy().horizontalPolicy(),
        QSizePolicy.Policy.Fixed if fixed else QSizePolicy.Policy.Preferred,
    )


def top_of_pane(table) -> QWidget:
    """``table`` at the top of a pane of its own, the rest of the pane left empty.

    For a list in a splitter, beside something that fills. A splitter hands each child its
    whole height whatever it prefers, so a list sized to twenty rows by
    :func:`fit_table_height` drew as many rows as the pane was tall. Capping the list's
    *maximum* instead caps the splitter's, since a splitter is only as tall as its shortest
    child may be — and the whole row then sank to the bottom of its tab. Here the pane takes the
    height and the list keeps its hint, at the top, where the player beside it starts.
    """
    pane = QWidget()
    box = QVBoxLayout(pane)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(table)
    box.addStretch(1)
    return pane


def fit_table_width(table) -> None:
    """Cap a table's width to its columns, the way :func:`fit_table_height` caps its height.

    ``ResizeToContents`` sizes each column to its content but does nothing about what is left:
    a two-column summary occupying 133 px inside a 547 px pane draws ~410 px of empty grid on
    every row. Same rule, other axis. A table that is a page's primary content must NOT call
    this — it is meant to fill.
    """
    model = table.model()
    columns = model.columnCount() if model is not None else 0
    table.setMaximumWidth(
        sum(table.columnWidth(c) for c in range(columns)) + 2 * table.frameWidth()
    )


def nav_cell() -> QStandardItem:
    """The trailing ``›`` cell of a navigable row. Inert — openers key off :func:`row_id`."""
    cell = QStandardItem(NAV_GLYPH)
    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return cell


def list_table(*, stretch: bool = False, sortable: bool = True) -> QTableView:
    """A read-only list table: row selection, no editing, no row numbers, columns to content.

    The shape every list in this package uses. Call :func:`fit_table_height` after filling it
    unless it is a page's primary content.

    ``stretch=True`` widens the last data column to fill the view instead of leaving dead
    grid to its right — right for a list that *is* its pane's content. Its opposite is
    :func:`fit_table_width`, for a summary that should hug what it holds.

    ``sortable=False`` opts *only* out of :func:`enable_sorting` — a bounded top-N panel
    where reordering would defeat the point, or a table whose row order is a registry's
    declared position. It never opts a table out of :func:`make_columns_resizable`, which
    every table gets unconditionally: `docs/gui_design.md` says "never opt out of
    resizing", and there is no reason column-dragging should depend on whether the rows
    happen to be sortable. A row-position-dependent click handler must read identity via
    :func:`row_id`, never ``index.row()``, once sorting is on.
    """
    view = QTableView()
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.verticalHeader().setVisible(False)
    # Off, like `pages._table`: Qt defaults it on, no caller in this package has ever wanted it,
    # and a wrapped cell makes `fit_table_height` short by the rows it grew.
    view.setWordWrap(False)
    make_columns_resizable(view)
    if sortable:
        enable_sorting(view)
    if stretch:
        view.horizontalHeader().setStretchLastSection(True)
    return view


def fill_width(table) -> None:
    """Stretch the last column so the table has no dead grid to its right.

    The counterpart to :func:`fit_table_width`: that one shrinks a summary to what it holds,
    this one spreads a table across the pane it was given. One of the two is right for every
    table; neither is the Qt default, which sizes columns to content and leaves the remainder
    blank — measured at 1400px wide, ten tables in this package trailed between 60px and
    1265px of empty grid on every row.
    """
    table.horizontalHeader().setStretchLastSection(True)


class SortCell(QStandardItem):
    """A read-only cell that sorts on a key it is given, not on the text it displays.

    `QStandardItem` compares display strings, which sorts `10.0 s` before `9.0 s` and orders
    confidences by their leading characters. The key is the value the column is *about*.

    Here rather than in one page because both result workspaces list instances and both want
    their columns to sort on the value — a copy in each is how the two came to disagree about
    everything else.
    """

    def __init__(self, text: str, key) -> None:
        super().__init__(text)
        self.setEditable(False)
        self._key = key

    def __lt__(self, other) -> bool:
        mine, theirs = self._key, getattr(other, "_key", other)
        try:
            return mine < theirs
        except TypeError:
            # A column with a blank in it — "no score yet" beside a float — would otherwise
            # raise inside Qt's sort, where the traceback is swallowed and the rows simply
            # stop moving. Compare as text when the values are not comparable, which is what
            # the reader sees anyway.
            return str(mine) < str(theirs)


def num_cell(value, text: str | None = None) -> SortCell:
    """A numeric cell: shows ``text`` (or ``str(value)``), sorts on ``value``.

    The gap this closes: :func:`enable_sorting` made every table in the package sortable, and
    `pages._cell` gives the two shared builders numeric ordering — but a table that builds its
    own cells got neither. **Twenty-one of them did**, across the analysis module result views
    and the query workspaces: scores, counts, durations and IoUs, all sorting as text, so a
    score column ordered `0.9` above `0.12` and a count column ran 1, 11, 12, 2.

    ``None`` is ordered below every number rather than raising. A blank means "not scored", and
    the alternative is a `TypeError` thrown inside Qt's own sort where nothing surfaces it — the
    rows just stop responding to the header.
    """
    key = float("-inf") if value is None else value
    return SortCell(text if text is not None else str(value), key)
