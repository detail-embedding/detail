"""A table shows the rows it has room for, and hugs them when it has few.

The class: **a cap the thing it caps can never reach.** ``fit_table_height`` computed a maximum
height from the row count and set it, and never touched the *preferred* height — but
``QTableView.sizeHint()`` is a constant **192px ≈ 5 rows whatever the row count**, and a
``Preferred`` widget is handed exactly its hint. So every list in the package showed at most five
rows: measured on a 60-run Analysis page, 5 of 60, with the computed cap allowing 1819px and 723px
of empty page sitting underneath it.

Nothing was squeezing the table and nothing failed. The page rendered, the cap was correct, the
table simply never asked for what it had been granted — which is why the only test that can catch
it is one that counts **rows on screen** rather than checking that a widget exists or that a
helper returned.

The second half is the mirror, and it is why the fix is a minimum rather than a bigger hint: a
table that keeps *asking* for slack it cannot use gets it granted to the container around it.
Measured on a three-row ``IndexTable`` with an expanding view: a 333px wrapper around 109px of
rows, 224px of empty grey. Both directions are asserted here because a fix for either one alone
looks right on the page it was written for.

A deliberate consequence, asserted below: a long table no longer collapses under pressure. It used
to have the lowest floor on an over-budget page and so absorbed *all* the compression — a 20-row
Measures table whose cap allowed 619px laid out at 70px, one row, while a fixed-height plot beside
it gave up nothing. That is only safe because the page now scrolls (``tests/gui/test_scroll.py``);
the two changes must not be separated.
"""

import pytest

from detail.gui.pages import IndexTable, _heading, _table, add_body_widget
from detail.gui.qt import QApplication, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from detail.gui.tables import (
    MIN_VISIBLE_ROWS,
    fit_table_height,
    max_visible_rows,
)

VIEWPORT = 739  # the measured content-area height of a 13" MacBook

#: What the table showed before the fix, whatever it held. Named so the assertions below read as
#: "more than the bug", not as an arbitrary threshold.
ROWS_AT_THE_DEFAULT_HINT = 5


def _rows_visible(view) -> int:
    header = view.horizontalHeader().sizeHint().height()
    return max(
        (view.height() - header) // view.verticalHeader().defaultSectionSize(), 0
    )


#: The cap this replaced. A 739px page fits more rows than this.
THE_OLD_CAP = 16


def _index_page(rows: int):
    """A collection page as the shell builds one: heading, table, action, trailing stretch."""
    index = IndexTable(
        ["Analysis", "Module", "Status", "Recordings"],
        empty_text="(no analyses yet)",
        navigable=True,
    )
    index.set_rows(
        [(f"an_{i:02d}", "temporal_clustering", "open", "rec_a") for i in range(rows)]
    )
    page = QWidget()
    box = QVBoxLayout(page)
    box.addWidget(_heading("Analysis"))
    add_body_widget(box, index)
    box.addWidget(QPushButton("New analysis…"))
    box.addStretch()
    # Shown and resized: an unlaid-out view reports heights that have nothing to do with the
    # layout, so every assertion here would pass against its own reverted fix.
    page.resize(1000, VIEWPORT)
    page.show()
    QApplication.processEvents()
    page.resize(1000, VIEWPORT)
    QApplication.processEvents()
    return page, index


@pytest.mark.parametrize("rows", [8, 25, 60])
def test_a_long_table_shows_more_than_the_five_rows_of_the_default_hint(qapp, rows):
    """The bug, counted. Reverting to a maximum-only fit puts every one of these back at 5."""
    _page, index = _index_page(rows)
    visible = _rows_visible(index.view)
    assert visible > ROWS_AT_THE_DEFAULT_HINT, (
        f"a {rows}-row table showed {visible} rows — the cap it computes allows "
        f"{index.view.maximumHeight()}px and it is asking for its default hint instead"
    )
    if rows <= THE_OLD_CAP:
        assert (
            visible == rows
        ), f"a {rows}-row table showed {visible} — it should hug its rows"
    else:
        assert visible > THE_OLD_CAP, (
            f"a {rows}-row table showed {visible}, which the old fixed cap of {THE_OLD_CAP} "
            f"would also have produced — the {VIEWPORT}px page has room for more"
        )


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_a_short_table_still_hugs_its_rows(qapp, rows):
    """The negative case, and the failure this helper originally existed for: a table that
    keeps asking for slack it cannot use has it granted to the container around it."""
    _page, index = _index_page(rows)
    assert _rows_visible(index.view) == rows
    assert (
        index.height() - index.view.height() == 0
    ), f"{index.height() - index.view.height()}px of empty grid around a {rows}-row table"


def test_a_very_long_table_is_bounded_by_the_page_and_scrolls_its_own_content(qapp):
    """A 600-row table must not make the page 18,000px tall — but what bounds it is the page,
    not a constant.

    The old rule capped at 16 rows, which is a number chosen without reference to how much page
    there is: on a tall window it left 700px empty under a table that had 60 more rows to show,
    and on a short one it was still too many. The table now asks for all 600, the layout gives
    it what the viewport has, and the rest scrolls inside the table.
    """
    _page, index = _index_page(600)
    visible = _rows_visible(index.view)

    # The bound is the page, so a table with far more rows than fit shows the *same* number as
    # one with merely more than fits. Asserted as a comparison rather than against a computed
    # number: predicting the allotment means reimplementing the layout in the test, which is how
    # the first version of this got it wrong by the two rows the heading and button take.
    _page60, index60 = _index_page(60)
    assert visible == _rows_visible(index60.view), (
        f"600 rows showed {visible} and 60 showed {_rows_visible(index60.view)} — if the bound "
        "were the page, both would show the same"
    )
    assert visible > THE_OLD_CAP
    assert index.view.height() <= VIEWPORT, (
        f"the table is {index.view.height()}px inside a {VIEWPORT}px page — asking for every "
        "row has to stay a request the layout can refuse"
    )
    assert (
        index.view.verticalScrollBar().maximum() > 0
    ), "the rows past the page are unreachable"


def test_the_display_height_and_the_squeeze_floor_are_different_numbers(qapp):
    """Pin the mechanism, and keep the two jobs apart.

    ``QTableView.sizeHint()`` is a constant 192px — about five rows — whatever the table holds,
    so the display height has to be stated somewhere. It lives on the **hint** (assigned on the
    instance, which is what PyQt's virtual dispatch reads, and which works the same on the
    ``QTableWidget``s `comparison` and `MetadataPage` pass in).

    It used to live on the *minimum*, and that was one number doing two jobs: "how tall this
    draws" and "how far this may be squeezed". A table beside a fill element then pushed its
    whole pane's floor up by its own display height — 135px of the interpret tab's minimum, on
    the screen that is short of it. The floor is now its own, smaller number, and a caller that
    wants none passes ``min_rows=0``.
    """
    _page, index = _index_page(25)
    view = index.view
    row = view.verticalHeader().defaultSectionSize()
    chrome = view.horizontalHeader().sizeHint().height() + 2 * view.frameWidth()

    # The hint is every row the table holds, capped by what the *display* can show — not by a
    # constant. What it then *gets* is the layout's answer, which is why the minimum below is a
    # separate, smaller number.
    budget = max_visible_rows(row)
    assert view.sizeHint().height() == chrome + min(25, budget) * row, "display height"
    assert view.minimumHeight() == chrome + MIN_VISIBLE_ROWS * row, "squeeze floor"
    assert (
        view.minimumHeight() < view.sizeHint().height()
    ), "one number for both is what made a table rigid in a pane that needed it to yield"
    assert (
        view.maximumHeight() == chrome + 25 * row
    ), "the cap must stay the content height"
    assert view.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Preferred


def test_a_table_no_longer_absorbs_a_whole_over_budget_page(qapp):
    """It used to have the lowest floor on the page and so took all of the compression: a
    20-row table laid out at 70px — one row — beside a plot that gave up nothing.

    Only safe because the page scrolls now; this is the assertion that couples the two.
    """
    greedy = QWidget()  # stands in for the workspace that will not yield
    greedy.setMinimumHeight(640)
    greedy.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    table = _table(
        ["Recording", "Subject", "Label", "Measure", "Value", "Unit", "n"],
        [(f"rec{i}", "0", "rear", "Bouts", str(i), "", "1") for i in range(20)],
    )
    page = QWidget()
    box = QVBoxLayout(page)
    add_body_widget(box, greedy)
    add_body_widget(box, table)
    box.addStretch()
    page.resize(1000, VIEWPORT)
    page.show()
    QApplication.processEvents()
    page.resize(1000, VIEWPORT)
    QApplication.processEvents()

    assert _rows_visible(table) >= MIN_VISIBLE_ROWS, (
        f"the table gave up its rows ({_rows_visible(table)} shown) so the page could fit — "
        "the page is what scrolls now. It may yield down to its floor and no further; it used "
        "to yield to a single row while a fixed-height plot beside it gave up nothing."
    )


def test_a_fixed_table_gets_exactly_its_rows_and_never_scrolls(qapp):
    """``fixed=True`` is a promise that the panel shows its own N — a scrollbar on it means the
    top-6 panel is not showing six."""
    rows = 20
    table = _table(["a", "b"], [(str(i), "x") for i in range(rows)])
    fit_table_height(table, fixed=True)
    page = QWidget()
    box = QVBoxLayout(page)
    add_body_widget(box, table)
    box.addStretch()
    page.resize(600, 1200)
    page.show()
    QApplication.processEvents()
    page.resize(600, 1200)
    QApplication.processEvents()

    assert _rows_visible(table) == rows, "a fixed panel must show every row it has"
    assert table.verticalScrollBar().maximum() == 0


# --- the list twin ----------------------------------------------------------- #
# `fit_list_height` was the half of this class that never got swept. The table learned that a
# display height belongs on the **minimum** (a maximum is a cap the widget may never reach) and
# that chrome includes the horizontal scrollbar, which sits inside the height it is given. Its
# list counterpart went on setting only a maximum, with a bare `+ 4` standing in for all chrome.
#
# What that cost: an Activity band capped at 202px was handed 192 by its page — 13.7 of its 14
# rows — so the bottom row drew sliced in half. Not a missing row, a visibly cut one, which is
# the thing a user had already reported about a table ("the row is partially cut off").


def _log_list(qapp, n_rows: int, width: int = 900, squeeze: int = 0):
    """A sized list **inside a parent that squeezes it**, which is where the bug lives.

    Sizing it standalone and then resizing it to its own numbers cannot see this class: with the
    max-only version the floor is 0, the resize lands on 0, and "the viewport holds whole rows"
    is true of nothing at all. The failure needs a layout handing the list less than it asked
    for — a page whose other content has taken the room — so the parent is the fixture.
    """
    from detail.gui.qt import QListWidget, QVBoxLayout, QWidget

    from detail.gui.sizing import fit_list_height, mono_font

    holder = QWidget()
    box = QVBoxLayout(holder)
    box.setContentsMargins(0, 0, 0, 0)
    lw = QListWidget()
    lw.setFont(mono_font())
    for i in range(n_rows):
        lw.addItem(f"20:44:24 . training ({i}/60) . epoch {i} . 35% . ~58m30s left")
    box.addWidget(lw)
    holder.resize(width, 400)
    holder.show()
    qapp.processEvents()

    fit_list_height(lw, max_rows=14)
    # The squeeze: the parent offers a little less than the list wants, exactly as a page does.
    holder.resize(width, max(lw.maximumHeight() - squeeze, 1))
    holder.layout().activate()
    qapp.processEvents()
    # Both, because the parent owns the C++ side: returning only the child lets the holder be
    # collected on the way out and every assertion then hits a deleted QListWidget.
    return lw, holder


def test_a_capped_list_shows_whole_rows(qapp):
    """The property, not a proxy for it: the viewport holds a **whole number** of rows.

    Asserting "the height is what the formula gives" would pass against every version of the
    formula, including the two that shipped a sliced row. What makes the fix a fix is that
    nothing is drawn in part.
    """
    lw, _holder = _log_list(qapp, 40, squeeze=10)
    row_h = lw.sizeHintForRow(0)
    leftover = lw.viewport().height() % row_h

    assert leftover == 0, (
        f"viewport is {lw.viewport().height()}px against a {row_h}px row — {leftover}px of a "
        "row is drawn, which is the cut-off row this helper exists to prevent"
    )


def test_a_short_list_hugs_its_rows(qapp):
    """The failure the helper originally existed for, kept: three rows must not open a void."""
    lw, _holder = _log_list(qapp, 3)

    assert (
        lw.height() <= 3 * lw.sizeHintForRow(0) + 12
    ), f"a 3-row list laid out at {lw.height()}px — the dead-grid failure, back again"


def test_the_display_height_is_on_the_minimum(qapp):
    """A maximum alone is a cap the widget may never reach — the recorded class, list side.

    The band was squeezed **below** its cap by the page it sat on, which a maximum cannot
    prevent and a minimum can. Guard the guard: this fails if someone drops back to setting
    only `setMaximumHeight`.
    """
    lw, _holder = _log_list(qapp, 40)

    assert (
        lw.minimumHeight() > 0
    ), "no floor — a layout may squeeze this to any height it likes"
    assert (
        lw.minimumHeight() == lw.maximumHeight()
    ), "a capped list whose floor and cap differ can still land between rows"


def test_chrome_is_not_reserved_for_a_scrollbar_that_will_not_appear(qapp):
    """Reserving on the *policy* over-reserves by a bar's height; ask whether content overflows.

    18px is about 1.3 rows here, which puts back exactly the slice that dropping the old `+ 4`
    had just removed — so this is the counterexample for the middle of the three attempts, not
    only the first.
    """
    from detail.gui.sizing import _hbar_height

    wide, _h1 = _log_list(qapp, 40, width=900)
    narrow, _h2 = _log_list(qapp, 40, width=120)

    assert (
        _hbar_height(wide) == 0
    ), "reserved scrollbar height on a list whose lines fit"
    assert (
        _hbar_height(narrow) > 0
    ), "reserved nothing on a list that must scroll sideways — its last row will be cut"
