"""Every table sorts and resizes — and a row-click handler survives being sorted.

**The rule, and where it already lived.** `docs/gui_design.md` §3.1 has required a `QTableView`
with sortable headers for multi-column lists since before this file existed. Of ~45 tables built
across `detail/gui/`, two called `setSortingEnabled(True)` by hand and none offered resizable
columns — `QHeaderView.ResizeMode.Interactive` (the mode that lets a user drag a column border)
appeared nowhere in the package. `gui/tables.py`'s `make_columns_resizable` / `enable_sorting`
fix this once, in the two shared table builders (`pages._table` / `tables.list_table`) plus every
raw `QTableWidget`/`QTableView` construction, so this file's job is to make sure it *stays* fixed
— for tables that exist today and any built tomorrow, since both sweeps below are **discovered**
(the same registries `test_smoke_construct.PAGES` and `test_dialog_space._dialogs()` already use),
not hand-listed.

**The correctness class this also guards.** Turning sorting on is only safe once a row's
click/selection handler can recover *which record it means* without trusting the clicked row's
*position* — `self._ids[index.row()]` silently opens, plays, or (in `MetadataPage`'s Save path)
**overwrites the wrong record** the instant a user sorts, because `index.row()` is a screen
position and `self._ids` stays in insertion order. `tables.stamp_row_id` / `tables.row_id` fix
this by stamping identity onto the row's own model item, which survives a sort because
`QStandardItemModel.sort()` moves whole rows, item data included. `test_a_sorted_metadata_table_
saves_the_edit_to_the_right_recording` below is that bug, caught: it fails immediately against
the pre-fix `self._recordings[r]` mapping the moment the table is sorted.

**A second, unrelated trap the same investigation surfaced**: `setSortingEnabled(True)` on a
table that has never had `setSortIndicator` called does not start "unsorted" — Qt's own docs say
enabling it "immediately triggers a call to sortByColumn() with the current sort section and sort
order", and a fresh `QHeaderView` reports section **0** for that, not "none". A bare
`setSortingEnabled(True)` therefore silently re-sorts every freshly-built table by column 0,
descending, as text — a run table's newest-first order, a query's ranked order, all replaced
before a user ever touches anything. `enable_sorting` clears the indicator to "no section" first;
`test_a_freshly_built_table_keeps_its_own_order` pins that.

**How the source scan missed two live sites (#26).** The scan matched a `.row()` only *inside*
a subscript, `self._ids[index.row()]`. The Recording page's Streams table held the position in a
name first (`idx = table.currentIndex().row()`, then `self._stream_names[idx]`), so after a sort
*Delete stream* deleted a different stream; its Video parts table did the same with `row`, then
`order[row]`, so *Move up / down* reordered parts nobody picked. Both passed the scan for as long
as it existed. `_row_positions_held` now follows a name holding a row position (and arithmetic on
it) to any subscript, or to any call that is not the view's own positional API
(`_POSITIONAL_APIS`); `test_the_row_position_scan_would_catch_the_two_step_shape` is both live
spellings. A table whose row order *is* the data (the parts table edits order) is the other
answer: it leaves sorting off, is named in `SORT_EXEMPT`, and still stamps its rows. The
behavioural halves: `tests/gui/test_streams.py::test_a_sorted_streams_table_acts_on_the_stream_
the_row_shows` and `tests/gui/test_pages.py::test_move_acts_on_the_part_the_row_shows_even_when_
reordered`.
"""

import ast
import pathlib

import pytest

from detail.api.experiment import Experiment
from detail.gui.pages import MetadataPage, _table
from detail.gui.qt import (
    QHeaderView,
    QPoint,
    QStandardItem,
    QStandardItemModel,
    QTableView,
    QTableWidget,
    QTest,
    Qt,
)
from detail.gui.tables import (
    enable_sorting,
    give_slack_to,
    make_columns_resizable,
    refit_slack,
    row_id,
    stamp_row_id,
)
from tests.gui.layout_checks import (
    assert_border_follows_the_drag,
    column_edges,
    drag_column_border,
    locked_columns,
)
from tests.gui.test_dialog_space import BUILDABLE, _build
from tests.gui.test_smoke_construct import PAGES, _experiment

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Tables that deliberately do not sort, named (`setObjectName`) so this list keys on an
#: identity rather than headers that can vary. Each is a claim, checked by `test_every_exemption_is_still_found`: an entry that
#: matches nothing has gone stale, exactly like `test_dialog_space.NEEDS_CONTEXT`.
SORT_EXEMPT = {
    # `skeleton_points` left here when the skeleton editor's table became a canvas plus a
    # point list (2026-10-08, §26 *Decisions during build*); a list is not a table.
    "group_measures": (
        "registry order is the point of this table (a fixed position a reader learns across "
        "runs) — see comparison.py's docstring on it. The ranked view is the top-N table beside "
        "it. Resizable, not sortable."
    ),
    "example_behaviors": (
        "the Mark examples checklist: rows are the behaviors in the order the user named them, "
        "the one being annotated is its selected row, and its Examples column is a progress "
        "bar placed per index (setIndexWidget), which a header-click reorder would leave on "
        "the wrong row. The clip list beside it sorts. Resizable, not sortable."
    ),
    "video_parts": (
        "the Recording page's Video parts table: its row order is the recording's part order, "
        "which Move up / Move down edit. A header-click sort would show an order the recording "
        "does not have, and the move handler once read the sorted row's position as the part "
        "(#26). Rows still carry their part index (`ids=`), so re-enabling sorting cannot "
        "bring that back. Resizable, not sortable."
    ),
    "add_recordings_staging": (
        "rows are recordings-to-be-created in the order the user is entering them, not records "
        "to rank; a header-click reorder mid-entry would be confusing. Resizable, not sortable."
    ),
}

#: Exemptions this file's sweep cannot reach on its own: `group_measures`'s full table needs a
#: real computed comparison (`tests/gui/test_comparison_gui.py`'s `_ran`) — heavier than this
#: guard file should carry just to double-check an exemption reason. Its resizability and
#: non-sortability are still exercised for real: `ComparisonDetailPage` is built and its tables
#: populated in that file's own passing tests. `add_recordings_staging` has no such excuse and is
#: built directly below instead.
NOT_SWEPT_HERE = {"group_measures"}


def _tables_in(widget) -> list:
    return list(widget.findChildren(QTableView)) + list(
        widget.findChildren(QTableWidget)
    )


def _built_pages(qapp, tmp_path):
    proj = _experiment(tmp_path)
    for name, factory in PAGES.items():
        try:
            yield name, factory(proj)
        except (
            Exception
        ) as exc:  # pragma: no cover - surfaced as a failure, not skipped
            pytest.fail(f"{name} failed to build: {exc}")


def _built_dialogs(tmp_path):
    for cls in BUILDABLE:
        # Each gets its own subdirectory: `_build` creates a fresh "p" experiment per call,
        # and `test_dialog_space.py` only ever calls it once per test — looping here would
        # collide on the second class otherwise.
        subdir = tmp_path / cls.__name__
        subdir.mkdir()
        try:
            yield cls.__name__, _build(cls, subdir)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"{cls.__name__} failed to build: {exc}")


def _all_tables(qapp, tmp_path):
    """(container name, table) for every table on every discovered page and dialog.

    Each tuple carries the page/dialog too (even though only ``name``/``table`` are used
    below): a parentless top-level widget's C++ side is tied to its Python wrapper's
    refcount, so a page that goes out of scope gets torn down — and its child tables along
    with it — the moment nothing references it any more. Dropping the container here reads
    fine and then raises "wrapped C/C++ object ... has been deleted" on the very next access.
    """
    found = []
    for name, page in _built_pages(qapp, tmp_path):
        found.extend((name, page, t) for t in _tables_in(page))
    for name, dialog in _built_dialogs(tmp_path):
        found.extend((name, dialog, t) for t in _tables_in(dialog))
    # `AddRecordingsDialog` used to be built here by hand, because the shared builder could not
    # supply its `on_submit` callback and it sat in the exemption list. The builder now fills
    # arguments from a table of providers, a no-op callback among them, so `_built_dialogs`
    # covers it — and building it *again* here collided on the same directory. A special case
    # that outlives its reason does not merely sit there; it breaks.
    return found


def test_the_sweep_finds_enough_tables_to_mean_something(qapp, tmp_path):
    """Guard the guard: a sweep that stops finding tables would pass both checks below
    vacuously forever."""
    found = _all_tables(qapp, tmp_path)
    assert len(found) >= 15, f"only found {len(found)} tables — the sweep is broken"


def test_every_table_has_a_draggable_column(qapp, tmp_path):
    """No table anywhere still uses the old, undraggable `ResizeToContents` default —
    on **every** column, not just one of them.

    **This used to check only that a table had *some* non-`ResizeToContents` section**,
    which passes even when every column but one is still locked: a table that redirects
    `_table`'s stretch column onto something other than the last one did so by looping
    `setSectionResizeMode(c, ResizeToContents)` over the rest, which relocks them — the
    exact mode `gui/tables.py`'s `make_columns_resizable` exists to replace. That shape
    shipped in four places (`active_processes.py`'s Runs/Exports/Imports tables,
    `measurements.py`'s ranked table, and two tables in `pages.py`) and this check passed
    on all four, because each still had its one `Stretch` column. `docs/gui_design.md`
    says "never opt out of resizing" — for every column, not "at least one".

    **`Stretch` is not exempt either.** This once argued a stretched column needn't be
    draggable itself. Opened in the app, it broke the columns around it: a stretched section
    refuses a drag, and a column to its right dragged wider grows *leftward*, because the
    stretched one hands the width back (measured: the dragged border stayed put and the left
    edge moved 60px). So the rule is that every section is `Interactive`; the slack goes to a
    column through `tables.give_slack_to`, or to the last through `setStretchLastSection`,
    which reports `Interactive` and drags normally.

    **This sweep reaches only one of those four sites** — `MetadataPage`. The other three
    live in tables that are filled after construction (Active processes' refresh, a results
    workspace's measurement band, the run page's Measures panel), which a page built against
    an empty experiment never shows; each has its own regression test beside its surface,
    calling the same `locked_columns`.
    """
    for name, _container, table in _all_tables(qapp, tmp_path):
        locked = locked_columns(table)
        assert not locked, (
            f"{name}'s table has column(s) {locked} not `Interactive` — a user cannot "
            "drag their border normally. Call make_columns_resizable (it still starts a "
            "column sized to its content), and give a column the spare width with "
            "tables.give_slack_to, never ResizeToContents or Stretch."
        )


def test_every_table_is_sortable_unless_named_in_sort_exempt(qapp, tmp_path):
    found = _all_tables(qapp, tmp_path)
    unsortable = [
        f"{name}.{table.objectName() or '<unnamed>'}"
        for name, _container, table in found
        if table.objectName() not in SORT_EXEMPT and not table.isSortingEnabled()
    ]
    assert (
        not unsortable
    ), f"these tables are not sortable and are not in SORT_EXEMPT: {unsortable}"


def test_every_exemption_is_still_found(qapp, tmp_path):
    """A stale exemption is how this list stops being complete — the entry that would have
    caught a regression instead certifies one that no longer exists.

    Skips `NOT_SWEPT_HERE` (documented above): those two need a heavier fixture than this
    file builds, and are exercised for real in their own test files instead.
    """
    found_names = {
        table.objectName() for _name, _container, table in _all_tables(qapp, tmp_path)
    }
    checkable = set(SORT_EXEMPT) - NOT_SWEPT_HERE
    stale = sorted(checkable - found_names)
    assert not stale, f"SORT_EXEMPT names tables no longer found by the sweep: {stale}"


# --------------------------------------------------------------------------------------- #
# A freshly built table keeps its own order — the Qt default-indicator trap
# --------------------------------------------------------------------------------------- #
def test_a_freshly_built_table_keeps_its_own_order(qapp):
    """`enable_sorting` must not itself reorder the table it is enabling sorting on.

    Revert to a bare `table.setSortingEnabled(True)` (no `setSortIndicator(-1, ...)` first)
    and this fails: Qt's default indicator (column 0) applies immediately, and a table built
    in "10, 2, 1" order renders as "10, 2, 1" sorted descending as text instead.
    """
    table = _table(["Label"], [("10",), ("2",), ("1",)])
    model = table.model()
    assert [model.item(r, 0).text() for r in range(3)] == ["10", "2", "1"]


def test_a_user_sort_still_applies_after_the_table_refreshes(qapp):
    """The mirror case: once a user *has* sorted, a live refresh must not silently drop it."""
    from detail.gui.pages import IndexTable

    index = IndexTable(["Label"], empty_text="(none)")
    index.set_rows([("10",), ("2",), ("1",)])
    index.view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    model = index.view.model()
    assert [model.item(r, 0).text() for r in range(3)] == ["1", "10", "2"]  # text order

    index.set_rows([("30",), ("4",), ("100",)])
    assert [model.item(r, 0).text() for r in range(3)] == [
        "100",
        "30",
        "4",
    ]  # still asc


# --------------------------------------------------------------------------------------- #
# Row identity survives a sort — the correctness class this PR exists for
# --------------------------------------------------------------------------------------- #
def test_row_id_recovers_identity_after_a_manual_sort():
    """The mechanism, isolated: `stamp_row_id` + `row_id` round-trip through a sort."""
    model = QStandardItemModel(3, 1)
    for r, (text, key) in enumerate([("b", "row-B"), ("a", "row-A"), ("c", "row-C")]):
        item = QStandardItem(text)
        stamp_row_id(item, key)
        model.setItem(r, 0, item)
    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert [model.item(r, 0).text() for r in range(3)] == ["a", "b", "c"]
    assert [row_id(model.index(r, 0)) for r in range(3)] == ["row-A", "row-B", "row-C"]


def test_a_navigable_table_opens_the_right_row_after_a_sort(qapp):
    """The bug this PR fixes, in the shared builder every navigable index table uses.

    Revert `_table`'s ``ids=`` stamping (back to reading ``self._ids[idx.row()]``) and this
    fails: after sorting, row 0's *position* is a different id than row 0 held at insertion.
    """
    rows = [("run_003", "tracking"), ("run_001", "detection"), ("run_002", "embedding")]
    ids = [r[0] for r in rows]
    table = _table(["Run", "Module"], rows, navigable=True, ids=ids)
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)  # run_001, run_002, run_003
    opened = [row_id(table.model().index(r, 0)) for r in range(3)]
    assert opened == ["run_001", "run_002", "run_003"]


def test_a_sorted_metadata_table_saves_the_edit_to_the_right_recording(qapp, tmp_path):
    """`MetadataPage` — the "Recording information" table from the screenshot this PR fixes.

    Sort by a data column (scrambling widget-row-position vs. insertion order), edit a cell,
    Save, and confirm the edit landed on the recording the row is actually showing — not
    whichever recording used to sit at that row position. Revert `_save`'s
    `_row_recording_id` lookup (back to `self._recordings[r]`) and this fails.
    """
    proj = Experiment.create("p", tmp_path)
    for sid, trial in [("s1", "3"), ("s2", "1"), ("s3", "2")]:
        proj.create_recording(sid)
    proj.set_metadata_table(
        ["Trial"], {"s1": {"Trial": "3"}, "s2": {"Trial": "1"}, "s3": {"Trial": "2"}}
    )

    page = MetadataPage(proj)
    # The user's own columns do not start at 1 — `recording_id` and the derived `file`
    # column sit in front of them. Asked for by name rather than by a literal, so the next
    # column added in front does not quietly turn this into a test about sorting by file.
    trial = page._USER_COL_START
    page._table.sortByColumn(
        trial, Qt.SortOrder.AscendingOrder
    )  # by Trial: s2(1), s3(2), s1(3)
    assert [page._row_recording_id(r) for r in range(3)] == ["s2", "s3", "s1"]

    page._enter_edit()
    page._table.item(0, trial).setText("100")  # row 0 is now s2 post-sort
    page._save()

    assert proj.recording_metadata["s2"]["Trial"] == "100"
    assert proj.recording_metadata["s1"]["Trial"] == "3"  # untouched
    assert proj.recording_metadata["s3"]["Trial"] == "2"  # untouched


def test_a_sorted_metadata_table_opens_the_recording_you_clicked(qapp, tmp_path):
    """The other half of the same page, and the half the original fix missed.

    ``_save`` was converted to ``_row_recording_id`` when sorting landed; ``_open_row`` —
    the double-click that leaves the page — was left reading ``self._recordings[index.row()]``
    twelve lines above the accessor written for it. So sorting the table and double-clicking
    opened a *different recording* than the one under the cursor, on the page whose whole job
    is picking a recording. Nothing failed: a real recording opens, just the wrong one.

    Revert ``_open_row`` to the positional read and this fails.
    """
    proj = Experiment.create("p", tmp_path)
    for sid in ("s1", "s2", "s3"):
        proj.create_recording(sid)
    proj.set_metadata_table(
        ["Trial"], {"s1": {"Trial": "3"}, "s2": {"Trial": "1"}, "s3": {"Trial": "2"}}
    )

    opened: list[str] = []
    page = MetadataPage(proj, on_open=opened.append)
    trial = page._USER_COL_START
    page._table.sortByColumn(trial, Qt.SortOrder.AscendingOrder)  # s2(1), s3(2), s1(3)
    assert [page._row_recording_id(r) for r in range(3)] == ["s2", "s3", "s1"]

    page._open_row(page._table.model().index(0, 0))

    # Row 0 shows s2 after the sort; s1 is what the pre-fix positional read would give.
    assert opened == ["s2"]


# --------------------------------------------------------------------------- #
# the source scan: no handler recovers a record from a row's position
# --------------------------------------------------------------------------- #
#: Functions (``path::function``) where a widget row's position is used *as* a position on
#: purpose. Claims, not exemptions: ``test_every_row_position_exemption_is_still_there`` fails
#: on a stale one, and an entry covers only the function it names.
ROW_POSITION_ALLOWED: dict[str, str] = {
    "detail/gui/keypoint_labeling.py::_remove": (
        "the frame list is a QListWidget that does not sort, so its row is the frame's place "
        "in the set by construction, and the API removes a frame by that place. What made it "
        "unsafe was an asynchronous change to the list (a mark save writing by position while "
        "the removal shifted every later frame): the two are serialised by the dialog's "
        "`_removing` flag and the saver's `busy`, guarded in tests/gui/test_keypoint_labeling.py "
        "(`test_a_mark_made_while_a_frame_is_being_removed_does_not_land_on_another_frame`)."
    ),
}

_ROW_CALLS = ("row", "currentRow")

#: Calls that take a row *as a position* in the view it came from: handing a position back to
#: the widget that produced it is correct whatever the sort. Anything else receiving one is
#: being told "this is the record", which is the bug.
_POSITIONAL_APIS = {
    "item",
    "index",
    "selectRow",
    "setCurrentRow",
    "takeItem",
    "removeRow",
    "cellWidget",
    "setItem",
    "showRow",
    "hideRow",
    "setRowHidden",
    "isRowHidden",
    "rowHeight",
    "setRowHeight",
    "verticalHeaderItem",
    "indexWidget",
    "setIndexWidget",
    "setCellWidget",
    "itemWidget",
}


def _is_row_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _ROW_CALLS
        and not node.args
    )


def _functions(tree):
    """``(name, node)`` for every function in ``tree``, plus ``"<module>"`` for the rest."""
    yield "<module>", tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def _row_position_reads(source: str) -> list[tuple[str, int]]:
    """``(function, line)`` of every ``<anything>[... .row() ...]`` subscript in ``source``.

    The shape behind the recorded class: a handler recovering a record from a list it holds
    in *build* order, keyed by the clicked row's *screen* position. The two agree only while
    nothing can reorder the view — which stopped being true when every table gained a
    sortable header, and stops being true silently, because the wrong record is still a
    real record. ``tables.row_id`` is the fix; this is the scan that keeps it applied.
    """
    tree = ast.parse(source)
    owner = {}
    for name, fn in _functions(tree):
        for node in ast.walk(fn):
            if isinstance(node, ast.Subscript):
                owner[id(node)] = (name, node)  # the innermost function wins
    out = set()
    for name, node in owner.values():
        if any(_is_row_call(inner) for inner in ast.walk(node.slice)):
            out.add((name, node.lineno))
    return sorted(out, key=lambda hit: hit[1])


def _is_position(node, held: set[str]) -> bool:
    """Whether ``node`` *is* a row position: ``x.row()``, a name already holding one, or
    arithmetic on either (``row + delta``, ``int(row)``). A call that merely *takes* a row
    (``model.item(row, 0).data(...)``, ``self._row_at(row)``) returns something else."""
    if _is_row_call(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in held
    if isinstance(node, ast.BinOp):
        return _is_position(node.left, held) or _is_position(node.right, held)
    if isinstance(node, ast.UnaryOp):
        return _is_position(node.operand, held)
    if isinstance(node, ast.IfExp):
        return _is_position(node.body, held) or _is_position(node.orelse, held)
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") in (
        "int",
        "max",
        "min",
    ):
        return any(_is_position(arg, held) for arg in node.args)
    return False


def _row_positions_held(source: str) -> list[tuple[str, int]]:
    """``(function, line)`` where a name holding a row position is used as an identity.

    The two-step spelling of the same class, and the one ``_row_position_reads`` never saw:
    ``idx = table.currentIndex().row()`` and then ``names[idx]``, or ``row = ...`` and then
    ``order[row]`` / ``reorder(row)``. Both live instances of #26 were this shape. A use is a
    subscript containing the name, or the name passed to a call that is not one of the view's
    own positional APIs (``_POSITIONAL_APIS``).
    """
    tree = ast.parse(source)
    out = set()
    for name, fn in _functions(tree):
        if name == "<module>":
            continue
        held: set[str] = set()
        # Assignments in source order, so `b = a + 1` after `a = x.row()` is held too.
        assigns = sorted(
            (
                n
                for n in ast.walk(fn)
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
            ),
            key=lambda n: (n.lineno, n.col_offset),
        )
        for node in assigns:
            if node.value is None or not _is_position(node.value, held):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            held.update(t.id for t in targets if isinstance(t, ast.Name))
        if not held:
            continue

        def mentions(expr) -> bool:
            return any(isinstance(x, ast.Name) and x.id in held for x in ast.walk(expr))

        for node in ast.walk(fn):
            if isinstance(node, ast.Subscript) and mentions(node.slice):
                out.add((name, node.lineno))
            elif isinstance(node, ast.Call):
                callee = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", "")
                )
                if callee in _POSITIONAL_APIS or callee in ("int", "max", "min"):
                    continue
                if any(
                    mentions(arg)
                    for arg in [*node.args, *(k.value for k in node.keywords)]
                ):
                    out.add((name, node.lineno))
    return sorted(out, key=lambda hit: hit[1])


def _row_position_offenders(source: str) -> list[tuple[str, int]]:
    return sorted(
        set(_row_position_reads(source)) | set(_row_positions_held(source)),
        key=lambda hit: hit[1],
    )


def test_no_gui_handler_reads_identity_from_a_row_position():
    offenders = []
    for path in sorted((REPO / "detail" / "gui").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        lines = source.splitlines()
        for function, lineno in _row_position_offenders(source):
            if f"{rel}::{function}" in ROW_POSITION_ALLOWED:
                continue
            offenders.append(
                f"{rel}:{lineno} ({function}): {lines[lineno - 1].strip()}"
            )
    assert not offenders, (
        "a row's position is not a record's identity once a table can be sorted — every "
        "table in this package can. Stamp identity with tables.stamp_row_id (or _table's "
        "`ids=`) and read it back with tables.row_id:\n" + "\n".join(offenders)
    )


def test_the_row_position_scan_would_catch_one():
    # Guard the guard, in the two spellings the package actually used.
    assert _row_position_offenders("self._on_open(self._recordings[index.row()])") == [
        ("<module>", 1)
    ]
    assert _row_position_offenders("x = self._ids[self._table.currentRow()]") == [
        ("<module>", 1)
    ]
    # And does not fire on the fix, or on a bare position used as a position.
    assert _row_position_offenders("name = row_id(index)") == []
    assert _row_position_offenders("self._table.item(index.row(), 0)") == []


def test_the_row_position_scan_would_catch_the_two_step_shape():
    """The shape both live #26 sites had, which the subscript-only scan passed over."""
    streams = (
        "def _selected(self):\n"
        "    idx = self._streams_table.currentIndex().row()\n"
        "    if 0 <= idx < len(self._stream_names):\n"
        "        return self._stream_names[idx]\n"
    )
    assert _row_position_offenders(streams) == [("_selected", 4)]
    parts = (
        "def _move(self, delta):\n"
        "    row = self._parts_table.currentIndex().row()\n"
        "    order = list(range(3))\n"
        "    order[row], order[row + delta] = order[row + delta], order[row]\n"
        "    self._recording.reorder(row)\n"
        "    self._parts_table.selectRow(row + delta)\n"
    )
    assert _row_position_offenders(parts) == [("_move", 4), ("_move", 5)]
    # Taint follows arithmetic through a second name.
    shifted = "def f(t, xs):\n    r = t.currentRow()\n    j = r + 1\n    return xs[j]\n"
    assert _row_position_offenders(shifted) == [("f", 4)]
    # Not: a value *looked up* by row in the view's own model, or the row handed back to
    # the view as a position.
    looked_up = (
        "def f(self, current):\n"
        "    index = self._model.item(current.row(), 0).data(ROLE)\n"
        "    row = self._row_at(current.row())\n"
        "    self._table.selectRow(current.row())\n"
        "    return self._rows[int(index)], row\n"
    )
    assert _row_position_offenders(looked_up) == []


def test_every_row_position_exemption_is_still_there():
    for key, reason in ROW_POSITION_ALLOWED.items():
        rel, function = key.split("::")
        path = REPO / rel
        assert path.is_file(), f"{rel} is gone; drop its entry ({reason})"
        hits = _row_position_offenders(path.read_text(encoding="utf-8-sig"))
        assert any(
            name == function for name, _ in hits
        ), f"{key} no longer reads a record by row position; drop its entry"


# --------------------------------------------------------------------------- #
# what a column sorts *on*
# --------------------------------------------------------------------------- #
def test_a_numeric_column_sorts_as_numbers(qapp):
    """The bug a user sees first: a column of counts ordered 1, 11, 12, 2, 3, 100.

    `QStandardItem` compares display strings. `SortCell` has existed for this since the two
    result workspaces needed it, and the shared builder every index table goes through simply
    never used it — so switching sorting on made every numeric column in the app wrong at once,
    which is a worse state than the unsortable tables it replaced.

    Driven through the real `pages._table`, not through a hand-built model: the defect was in the
    builder, so a test that builds its own cells would have passed against it.
    """
    from detail.gui.pages import _table

    values = [1, 2, 3, 11, 12, 100]
    table = _table(["Frames"], [(v,) for v in values])
    model = table.model()
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    shown = [int(model.item(r, 0).text()) for r in range(model.rowCount())]
    assert shown == sorted(values), (
        f"a numeric column sorted to {shown}; it is comparing display text, so 11 lands "
        "before 2. The cell needs the value it is about, not just the string of it."
    )


def test_a_text_column_still_sorts_as_text(qapp):
    """The other half of the pair — otherwise the fix could be "treat everything as a number"
    and this would not notice."""
    from detail.gui.pages import _table

    names = ["charlie", "alpha", "bravo"]
    table = _table(["Name"], [(n,) for n in names])
    model = table.model()
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    shown = [model.item(r, 0).text() for r in range(model.rowCount())]
    assert shown == sorted(names), shown


def test_a_yes_no_column_sorts_alphabetically(qapp):
    """`bool` is an `int` in Python, so a naive numeric branch would order a Yes/No column by
    False-then-True and read as arbitrary. The column shows words; it sorts as words."""
    from detail.gui.pages import _table

    table = _table(["Done"], [(True,), (False,), (True,)])
    model = table.model()
    table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    shown = [model.item(r, 0).text() for r in range(model.rowCount())]
    assert shown == ["False", "True", "True"], shown


# --------------------------------------------------------------------------- #
# sorting and editing cannot both be live
# --------------------------------------------------------------------------- #
def test_editing_a_sorted_table_does_not_move_the_row_being_edited(qapp, tmp_path):
    """Measured, on a plain `QTableWidget` with sorting on: edit row 0 of a sorted column and
    the row leaves — `[alpha, bravo, charlie]` becomes `[bravo, charlie, zulu]`. The user is
    then typing into whatever row slid into its place.

    `refill_table_widget` covers the *repopulate* path. This is the other one, and it is the one
    a person meets: they are in edit mode, they have sorted, and they type.

    A `QTableView` over a `QStandardItemModel` does not do this — that model resorts only on an
    explicit `sort()`. So the rule is about the widget that does.
    """
    from detail.gui.pages import MetadataPage

    experiment = Experiment.create("p", tmp_path)
    for name in ("alpha", "bravo", "charlie"):
        experiment.add_recording(name)
    page = MetadataPage(experiment)

    assert page._table.isSortingEnabled(), "the table should be sortable in view mode"
    page._enter_edit()
    assert not page._table.isSortingEnabled(), (
        "sorting is still live in edit mode — typing into a cell of the sorted column moves "
        "the row out from under the cursor"
    )
    page._leave_edit()
    assert (
        page._table.isSortingEnabled()
    ), "leaving edit mode left the table unsortable; the mode has to restore what it took"


# --------------------------------------------------------------------------- #
# driven as input, not as method calls
# --------------------------------------------------------------------------- #
def _click_header(qapp, table, column: int) -> None:
    """Click the header section for ``column``, the way a user does.

    Every other sort test in this file calls ``sortByColumn`` directly, which exercises Qt
    rather than the app: it works whether or not the header is interactive, whether or not
    sorting is enabled, and whether or not a section is even clickable. Those are three
    separate ways for "the user can sort by clicking a column" to be false while the
    method-call tests all pass.
    """
    header = table.horizontalHeader()
    table.resize(600, 300)
    table.show()
    qapp.processEvents()
    x = header.sectionPosition(column) + header.sectionSize(column) // 2
    QTest.mouseClick(
        header.viewport(),
        Qt.MouseButton.LeftButton,
        pos=QPoint(x, header.height() // 2),
    )
    qapp.processEvents()


def test_clicking_a_header_sorts_the_table(qapp):
    """The gesture the rule is actually about."""
    values = [3, 1, 2]
    table = _table(["Frames"], [(v,) for v in values])
    model = table.model()
    _click_header(qapp, table, 0)
    shown = [int(model.item(r, 0).text()) for r in range(model.rowCount())]
    assert shown == sorted(values), (
        f"clicking the header left the rows as {shown}; sorting is reachable from code but "
        "not from the gesture the guideline promises"
    )


def test_clicking_a_header_twice_reverses_the_order(qapp):
    """A sort you cannot reverse is half a control, and the second click is a different code
    path — the first sets an indicator where there was none, the second flips one."""
    values = [3, 1, 2]
    table = _table(["Frames"], [(v,) for v in values])
    model = table.model()
    _click_header(qapp, table, 0)
    _click_header(qapp, table, 0)
    shown = [int(model.item(r, 0).text()) for r in range(model.rowCount())]
    assert shown == sorted(values, reverse=True), shown


def test_a_column_can_actually_be_dragged(qapp):
    """`sectionResizeMode` is a *claim*; this moves the divider.

    The existing check asserts the mode is not `ResizeToContents`, which is necessary and not
    sufficient — a `Fixed` section reports neither and still cannot be dragged, and a header
    with `setSectionsMovable`/resizable off ignores the drag entirely.
    """
    table = _table(["A", "B"], [(1, 2), (3, 4)])
    header = table.horizontalHeader()
    table.resize(600, 300)
    table.show()
    qapp.processEvents()
    before = header.sectionSize(0)
    header.resizeSection(0, before + 40)
    qapp.processEvents()
    assert header.sectionSize(0) == before + 40, (
        f"section 0 would not resize ({before} -> {header.sectionSize(0)}); its mode lets the "
        "check pass while the divider does not move"
    )


def _slack_table(qapp, width: int = 700):
    """Four columns, the slack on column 1 — the shape of a name column with others after it."""
    table = _table(
        ["id", "name", "status", "n"],
        [("a1", "a long recording name", "done", 3), ("b2", "short", "failed", 12)],
    )
    give_slack_to(table, 1)
    table.resize(width, 300)
    table.show()
    qapp.processEvents()
    return table


def _dead_grid(table) -> int:
    header = table.horizontalHeader()
    return table.viewport().width() - header.length()


def test_the_slack_column_fills_the_table_and_follows_the_window(qapp):
    """What `Stretch` was there for, kept: no dead grid, and the spare width on the name."""
    table = _slack_table(qapp)
    header = table.horizontalHeader()
    assert _dead_grid(table) == 0
    widest = max(range(header.count()), key=header.sectionSize)
    assert widest == 1, f"column {widest} took the slack, not the name"
    before = header.sectionSize(1)
    table.resize(1000, 300)
    qapp.processEvents()
    assert _dead_grid(table) == 0
    assert header.sectionSize(1) == before + 300


def test_after_a_drag_the_widths_are_the_users(qapp):
    """Re-fitting after a drag took back what was dragged: a column widened past the view
    brings a horizontal scrollbar, the rows then need a vertical one, the viewport narrows, and
    a fit on that shrank the slack column by the width the user had just added. Once a border
    has been dragged, a width change moves nothing."""
    table = _slack_table(qapp)
    drag_column_border(table, 2, 200)
    widths = column_edges(table)
    table.resize(500, 300)
    qapp.processEvents()
    table.resize(900, 300)
    qapp.processEvents()
    assert column_edges(table) == widths


def test_a_refill_keeps_the_slack_where_it_was(qapp):
    """`IndexTable.set_rows` resizes every column to its content, which shrinks the slack
    column back to its text; `refit_slack` is what gives the spare width back."""
    table = _slack_table(qapp)
    table.resizeColumnsToContents()
    qapp.processEvents()
    assert _dead_grid(table) > 0, "the resize to contents did not shrink anything"
    refit_slack(table)
    qapp.processEvents()
    assert _dead_grid(table) == 0


@pytest.mark.parametrize("column", [1, 2], ids=["the-slack-column", "right-of-it"])
def test_a_border_follows_the_drag_beside_the_slack_column(qapp, column):
    """The reported bug: in the clustering measurements table, a border dragged right stayed
    where it was and the column grew leftward, because the stretched name column to its left
    gave the width back. The slack column itself could not be dragged at all."""
    table = _slack_table(qapp)
    assert_border_follows_the_drag(table, column)


def test_the_drag_check_fails_on_a_stretched_column(qapp):
    """Guard the guard: the drag assertion above, run on the shape it replaced, fails for
    both columns — `Stretch` on the name, a real mouse drag, the same measurement."""
    for column in (1, 2):
        table = _table(
            ["id", "name", "status", "n"],
            [("a1", "a long recording name", "done", 3)],
        )
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.resize(700, 300)
        table.show()
        qapp.processEvents()
        with pytest.raises(AssertionError, match="should follow the cursor"):
            assert_border_follows_the_drag(table, column)


def test_the_draggable_column_check_would_catch_one_relocked_column(qapp):
    """Guard the guard: `test_every_table_has_a_draggable_column` must fail on the exact
    shapes it was written for — one column redirected back to `ResizeToContents`, and one
    given the slack with `Stretch`, beside a column that is fine. Reverting the check to
    "at least one section is not ResizeToContents" passes this table, which is precisely
    why that version missed the four real sites.

    It calls `locked_columns`, the predicate the sweep runs, rather than restating it: an
    inline copy here kept passing whatever the sweep's own check was weakened to.
    """
    table = _table(["A", "B", "C"], [(1, 2, 3)])
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    assert locked_columns(table) == [0, 1]


# --------------------------------------------------------------------------- #
# the edit boundary, in both directions
# --------------------------------------------------------------------------- #
def test_the_only_editable_table_declares_itself_one(qapp, tmp_path):
    """`requirements.EDITOR_PROPERTY` is what makes the exception visible.

    The conformance grid asserts every table is read-only, and only ever sees them in view mode —
    so before this property existed the guidelines said "no table in this app is an editor" while
    one was. A declaration cannot be forgotten silently: drop it and the grid fails, because the
    table then looks like an ordinary one that happens to flip its edit triggers.
    """
    from detail.gui import requirements

    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("alpha")
    page = MetadataPage(experiment)
    assert page._table.property(requirements.EDITOR_PROPERTY) is True


def test_an_editable_table_is_read_only_until_edit_mode(qapp, tmp_path):
    """Half of the pair. A table you can type into without asking is one you edit by accident —
    and on this page a stray keystroke rewrites a recording's metadata."""
    from detail.gui.qt import QAbstractItemView

    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("alpha")
    page = MetadataPage(experiment)
    assert (
        page._table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    ), "the table is editable before anyone pressed Edit"

    page._enter_edit()
    assert (
        page._table.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers
    ), "Edit mode did not actually make the cells editable"


def test_an_editable_table_goes_back_to_read_only_when_the_edit_ends(qapp, tmp_path):
    """The other half, and the one nothing checked. A mode that does not put back what it took
    leaves the page permanently editable after one trip through Edit — with the Edit button
    showing again, so nothing on screen says the cells are still live.

    Both directions, because a test that only enters passes against a `_leave_edit` that does
    nothing at all. That single-direction shape is a recorded class here.
    """
    from detail.gui.qt import QAbstractItemView

    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("alpha")
    page = MetadataPage(experiment)
    page._enter_edit()
    page._leave_edit()
    assert (
        page._table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    ), "leaving edit mode left the cells editable"
