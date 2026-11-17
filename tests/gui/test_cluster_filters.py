"""The clustering page's filters: the docked sections → one Selection → each field's line, and back.

**A recorded class lives here: a selectable disabled label painted bright.** A label made
selectable (every label on a page is, `selectable.make_text_selectable`) draws through a text
document, and a disabled one takes the palette's disabled *Text* colour — not the disabled
*WindowText* a plain label greys with. A palette that greys disabled WindowText but sets Text for
every group (a dark palette built role by role, as the laptop render's was) drew the dock's
one-value lines — "animal  F3", disabled on purpose — as bright as a live field, while the light
theme greyed them. Nothing fails: the row is disabled and says why in its tooltip; only the
picture is wrong. `ConstantRow` gives its labels the disabled WindowText colour, re-read on a
palette change; the guard is
`test_a_one_value_line_is_greyed_under_a_palette_that_leaves_disabled_text_bright`, which fails
with `_mute_labels` emptied. The macOS system palette greys both roles, so the bug did not show
there; the guard builds the palette that shows it.

**A second: a value's text sets the panel's minimum width.** The filters live in a scroll area
that never scrolls sideways, and a plain ``QLabel``'s minimum width is its whole text. So each
value a user picked — "saline_vehicle_control only", a spreadsheet's long column name, Time in
session's "50%–100%" with its Reset beside it — raised the column's minimum; past the panel's
width the column grew wider than its viewport and its right edge was cut ("end" → "enc", "F3" →
"F", Location's "any"), and it shifted while a handle was dragged because the text width changed
with each step (user, dark theme, 2026-11-09). The same held above the scroll: "12,345 of 99,999
clips match" with Reset all pushed the strip to 259px of a 240px panel. Nothing fails — the
widgets are all there, only cut. The lines' words (name, summary, a one-value field's value, a
range's from–to, the count line) are `sizing.ElidedLabel`s, whose floor is a constant number of
characters (a summary holds more than the name beside it), with the whole text in the line's
tooltip; notes wrap. Guard: `test_no_value_makes_the_filters_panel_wider_than_itself`, which sets
every section to its longest realistic value at 350 and 240px and asserts the content fits and no
line moves sideways — 2 failed on the code before, 2 failed with the floor made the text's width
again, 1 with only the count line put back to a plain label; its negative case plants a plain
label in a line.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QAbstractButton, QLabel, QPushButton, QWidget  # noqa: E402

from detail.gui import cluster_filters as cf  # noqa: E402
from detail.gui.cluster_filters import (  # noqa: E402
    CATEGORY,
    DATE,
    LOCATION,
    NUMBER,
    SESSION,
    TIME,
    VIDEO,
    Column,
    RecordingChoice,
    Selection,
)

#: More than ``MAX_TICK_VALUES`` distinct values each, so these two are ranges.
WEIGHTS = [float(w) for w in range(20, 50)]
BIRTHS = [f"2026-01-{d:02d}" for d in range(1, 16)]
COLUMNS = [
    Column("treatment", CATEGORY, {"drug_A": 120, "vehicle": 98, "": 5}),
    Column("animal_id", CATEGORY, {f"A{i:02d}": 10 + i for i in range(1, 25)}),
    Column("dose_mg", NUMBER, {"0": 98, "0.5": 30, "1": 28, "2": 31, "5": 29, "": 4}),
    Column("recorded", DATE, {"2026-03-02": 20, "2026-03-11": 40, "2026-03-27": 15}),
    Column("weight_g", NUMBER, {f"{w:g}": 7 for w in WEIGHTS} | {"": 3}),
    Column("born", DATE, {b: 5 for b in BIRTHS}),
    Column("animal", CATEGORY, {"F3": 223}),
    Column("sex", CATEGORY, {"F": 223}),
    Column("cohort", NUMBER, {"1": 200, "": 23}),
    Column("start", TIME, {"17:23:19": 40, "9:05": 30, "14:00:21": 50}),
    Column("clock", TIME, {f"{h:02d}:30": 4 for h in range(6, 20)}),
]
BY_NAME = {c.name: c for c in COLUMNS}
RECORDINGS = [
    RecordingChoice(f"recording_{i:03d}", f"A{i:02d}_drug", 9 + i) for i in range(1, 7)
]


@pytest.fixture
def bar(qapp):
    widget = cf.FilterBar()
    widget.set_columns(COLUMNS)
    widget.set_recordings(RECORDINGS)
    widget.set_match_count(None, 1004)
    seen: list = []
    widget.selection_changed.connect(seen.append)
    widget.seen = seen
    widget.dock.resize(420, 2400)
    widget.dock.show()
    # Every field open, so the sections are laid out for the gestures below; the lines
    # themselves (closed by default) are tested on `closed_bar`.
    for row in widget.field_rows():
        row.set_open(True)
    yield widget
    widget.dock.close()


def _check(checklist, key, on: bool) -> None:
    for i in range(checklist.list.count()):
        item = checklist.list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == key:
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            return
    raise AssertionError(f"no entry {key!r}")


# --------------------------------------------------------------------------- #
# the widget picked per type
# --------------------------------------------------------------------------- #
def test_each_column_gets_the_widget_for_its_values(bar):
    sections = bar._meta_sections
    assert isinstance(sections["treatment"], cf.Checklist)
    assert isinstance(sections["dose_mg"], cf.Checklist), "5 numbers: tick boxes"
    assert isinstance(sections["recorded"], cf.Checklist), "3 dates: tick boxes"
    assert isinstance(sections["weight_g"], cf.NumberRange), "30 numbers: a range"
    assert isinstance(sections["born"], cf.DateRange), "15 dates: a range"
    assert isinstance(
        sections["cohort"], cf.Checklist
    ), "one value and blanks: a choice"


def test_a_column_with_one_value_is_a_greyed_line_at_the_bottom(bar):
    assert "animal" not in bar._meta_sections and "sex" not in bar._meta_sections
    rows = bar.constant_rows()
    assert [r.texts() for r in rows] == [
        ("animal", "F3"),
        ("sex", "F"),
    ]
    assert all(not r.isEnabled() and not r.isHidden() for r in rows)
    assert rows[0].toolTip() == (
        "All 223 clips have animal F3, so there is nothing to filter."
    )
    layout = bar.dock.fields_box.layout()
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    last_field = max(order.index(r) for r in bar.field_rows())
    assert all(order.index(r) > last_field for r in rows), "below every openable field"


def test_the_tick_box_limit_counts_parsed_values():
    """ "1" and "1.0" are one number, so twelve numbers written two ways are still twelve."""
    twice = {f"{n}": 1 for n in range(12)} | {f"{n}.0": 1 for n in range(12)}
    assert cf.section_kind(Column("n", NUMBER, twice)) == "checklist"
    thirteen = {f"{n}": 1 for n in range(13)}
    assert cf.section_kind(Column("n", NUMBER, thirteen)) == "range"
    assert cf.section_kind(Column("n", NUMBER, {"1": 4, "1.0": 2})) == "same"
    assert cf.section_kind(Column("n", NUMBER, {"1": 4, "": 2})) == "checklist"
    assert cf.section_kind(Column("c", CATEGORY, {"": 9})) == "same"


def test_a_number_checklist_lists_values_in_number_order(bar):
    assert bar._meta_sections["dose_mg"].keys() == ["0", "0.5", "1", "2", "5", ""]


def test_a_category_lists_no_value_with_its_count(bar):
    texts = [
        bar._meta_sections["treatment"].list.item(i).text()
        for i in range(bar._meta_sections["treatment"].list.count())
    ]
    assert texts[-1].startswith(cf.NO_VALUE_TEXT) and texts[-1].endswith("5")
    assert texts[0] == "drug_A   120"


def test_type_ahead_appears_only_over_twenty_values_and_filters(bar):
    short, long = bar._meta_sections["treatment"], bar._meta_sections["animal_id"]
    assert short.search is None
    assert long.search is not None
    long.search.setText("a1")
    assert long.visible_keys() == [f"A{i:02d}" for i in range(10, 20)]
    long.search.setText("")
    assert len(long.visible_keys()) == 24


def test_column_kind_is_checked(qapp):
    with pytest.raises(ValueError, match="column kind must be one of"):
        Column("x", "text", {})


# --------------------------------------------------------------------------- #
# gestures → the selection
# --------------------------------------------------------------------------- #
def test_unchecking_one_value_makes_a_condition_and_checking_it_back_removes_it(bar):
    section = bar._meta_sections["treatment"]
    _check(section, "vehicle", False)
    assert bar.selection().values == {"treatment": frozenset({"drug_A", ""})}
    _check(section, "vehicle", True)
    assert bar.selection().is_empty
    assert [s.is_empty for s in bar.seen] == [False, True]


def test_a_fields_reset_is_in_its_line_and_shows_only_while_it_has_a_filter(bar):
    """User, review round 8: a full-size Reset under every open field's values was one more
    button per field. It is a small one in the field's own line now, there only while the field
    has something to reset."""
    row = bar.field_row("treatment")
    section = row.section
    assert row.reset_button.isHidden(), "no filter: no Reset"
    assert [b.text() for b in section.findChildren(QPushButton)] == ["only"]
    _check(section, "vehicle", False)
    assert not row.reset_button.isHidden()
    assert row.reset_button.parent() is row.head, "in the field's own line"
    assert row.reset_button.toolTip() == "Remove the filter on treatment"
    assert row.reset_button.font().pointSizeF() < row.title.font().pointSizeF(), "small"
    row.reset_button.click()
    assert bar.selection().is_empty
    assert section.value() is None
    assert row.reset_button.isHidden()
    bar.set_selection(Selection(values={"treatment": frozenset({"drug_A"})}))
    assert not row.reset_button.isHidden(), "set from outside: the Reset follows"


@pytest.mark.parametrize("field", ["Video", "weight_g", "born", "clock"])
def test_video_and_range_fields_get_a_reset_too(bar, field):
    selection = {
        "Video": Selection(recording_ids=frozenset({"recording_002"})),
        "weight_g": Selection(ranges={"weight_g": (25.0, 30.0)}),
        "born": Selection(ranges={"born": ("2026-01-02", "2026-01-05")}),
        "clock": Selection(ranges={"clock": ("08:30", "10:30")}),
    }[field]
    row = bar.field_row(field)
    assert row.reset_button.isHidden()
    bar.set_selection(selection)
    assert not row.reset_button.isHidden()
    others = [r for r in bar.field_rows() if r is not row]
    assert all(
        r.reset_button.isHidden() for r in others
    ), "only the field with a filter"
    row.reset_button.click()
    assert bar.selection().is_empty
    assert row.reset_button.isHidden()


def test_only_ticks_just_the_row_it_sits_on(bar):
    section = bar._meta_sections["treatment"]
    assert section.only_key() is None, "hidden until a row is hovered"
    item = section.list.item(1)
    section.list.itemEntered.emit(item)
    assert section.only_key() == "vehicle"
    assert section.only_button.toolTip() == "Tick only vehicle"
    rect = section.list.visualItemRect(item)
    assert rect.contains(section.only_button.geometry().center()), "on its row"
    section.only_button.click()
    assert bar.selection().values == {"treatment": frozenset({"vehicle"})}
    assert bar.field_row("treatment").summary_text() == "vehicle only"


def test_only_follows_the_keyboard_current_row(bar):
    section = bar._meta_sections["treatment"]
    section.list.setCurrentRow(0)
    assert section.only_key() == "drug_A"
    section.only_button.click()
    assert bar.selection().values == {"treatment": frozenset({"drug_A"})}


def test_a_number_checklist_stores_values_not_a_range(bar):
    """The negative case for the shape rule: unticking a dose on a *number* column is a set of
    values. Decided by the column's type, it was stored as a range, which the API refuses for
    a set and the field's line printed as one."""
    _check(bar._meta_sections["dose_mg"], "0", False)
    assert bar.selection().values == {"dose_mg": frozenset({"0.5", "1", "2", "5", ""})}
    assert bar.selection().ranges == {}
    assert bar.field_row("dose_mg").summary_text() == "5 of 6"


def test_a_conditions_shape_decides_values_or_range():
    number = Column("dose_mg", NUMBER, {"0": 1, "1": 1})
    picked = cf.with_condition(Selection(), number, frozenset({"1"}))
    assert picked.values == {"dose_mg": frozenset({"1"})} and picked.ranges == {}
    span = cf.with_condition(Selection(), "dose_mg", (0.0, 1.0))
    assert span.ranges == {"dose_mg": (0.0, 1.0)} and span.values == {}
    assert cf.with_condition(span, number, None).is_empty
    with pytest.raises(TypeError, match="a set of values or a"):
        cf.with_condition(Selection(), number, [1.0, 2.0, 3.0])
    names = {"dose_mg": number}
    assert cf.field_summary(picked, ("meta", "dose_mg"), names, {}) == "1 only"
    assert cf.field_summary(span, ("meta", "dose_mg"), names, {}) == "0–1"


def _bar_centre(histogram, index: int) -> QPoint:
    width = histogram.width() / len(histogram._counts)
    return QPoint(int((index + 0.5) * width), histogram.height() // 2)


def _drag(widget, points) -> None:
    """Press at the first point, move through the rest, release at the last."""
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=points[0])
    for point in points[1:]:
        QTest.mouseMove(widget, point)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=points[-1])


def test_dragging_a_handle_picks_a_number_range(bar):
    section = bar._meta_sections["weight_g"]
    histogram = section.histogram
    assert histogram.width() > 100, "the dock must be laid out for the handles to land"
    QTest.mousePress(histogram, Qt.MouseButton.LeftButton, pos=QPoint(1, 5))
    QTest.mouseMove(histogram, _bar_centre(histogram, 5))
    assert bar.selection().ranges == {"weight_g": (25.0, 49.0)}, "live, before release"
    QTest.mouseRelease(
        histogram, Qt.MouseButton.LeftButton, pos=_bar_centre(histogram, 6)
    )
    assert bar.selection().ranges == {"weight_g": (26.0, 49.0)}
    assert section.span_label.text() == "26 – 49"
    _drag(histogram, [QPoint(histogram.width() - 1, 5), _bar_centre(histogram, 10)])
    assert bar.selection().ranges == {"weight_g": (26.0, 30.0)}
    assert bar.field_row("weight_g").summary_text() == "26–30"


def test_a_handle_never_passes_the_other(bar):
    histogram = bar._meta_sections["weight_g"].histogram
    _drag(histogram, [QPoint(1, 5), _bar_centre(histogram, 12)])
    _drag(histogram, [_bar_centre(histogram, 29), _bar_centre(histogram, 3)])
    assert histogram.indices() == (12, 12)


def test_dragging_a_handle_picks_a_date_range(bar):
    histogram = bar._meta_sections["born"].histogram
    _drag(histogram, [QPoint(histogram.width() - 1, 5), _bar_centre(histogram, 9)])
    assert bar.selection().ranges == {"born": ("2026-01-01", "2026-01-10")}
    assert bar.field_row("born").summary_text() == "Jan 1 – Jan 10"


def test_video_checklist_emits_recording_ids_and_shows_names(bar):
    bar._video_list.only("recording_003")
    assert bar.selection().recording_ids == frozenset({"recording_003"})
    assert bar.field_row("Video").summary_text() == "A03_drug only"


def test_location_click_and_drag_pick_cells(bar):
    picker = bar.region
    cell = picker.map.cell_at
    # Click the first cell, then drag a rectangle; both go through DensityMap's own gestures.
    first = QPoint(6, 6)
    assert cell(first.toPointF()) == (0, 0)
    QTest.mouseClick(picker.map, Qt.MouseButton.LeftButton, pos=first)
    assert bar.selection().cells == frozenset({(0, 0)})
    assert not bar.field_row("Location").reset_button.isHidden()
    QTest.mouseClick(picker.map, Qt.MouseButton.LeftButton, pos=first)
    assert bar.selection().cells is None
    assert bar.field_row("Location").reset_button.isHidden()
    # A drag from cell (0, 0) to the cell under `far` picks the whole rectangle between.
    far = QPoint(picker.map.width() // 4, picker.map.height() // 4)
    bx, by = cell(far.toPointF())
    _drag(picker.map, [first, far])
    assert bar.selection().cells == frozenset(
        (x, y) for x in range(bx + 1) for y in range(by + 1)
    )
    assert bx >= 1 and by >= 1


def _centre(picker, x: int, y: int) -> QPoint:
    cell_w, cell_h, _cols, _rows = picker.map._cell_size()
    return QPoint(int(4 + (x + 0.5) * cell_w), int(4 + (y + 0.5) * cell_h))


def _box(x0, y0, x1, y1) -> frozenset:
    return frozenset((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


def test_a_box_is_drawn_while_dragging(bar):
    picker = bar.region
    QTest.mousePress(picker.map, Qt.MouseButton.LeftButton, pos=_centre(picker, 2, 2))
    QTest.mouseMove(picker.map, _centre(picker, 5, 4))
    assert picker.map.region() == _box(2, 2, 5, 4), "drawn under the hand"
    assert bar.selection().cells == _box(2, 2, 5, 4), "and the tree shades, live"
    QTest.mouseRelease(picker.map, Qt.MouseButton.LeftButton, pos=_centre(picker, 6, 5))
    assert bar.selection().cells == _box(2, 2, 6, 5)


def test_dragging_inside_the_box_moves_it_and_it_stays_on_the_map(bar):
    picker = bar.region
    picker.set_region(_box(2, 2, 5, 4))
    _drag(picker.map, [_centre(picker, 3, 3), _centre(picker, 6, 5)])
    assert bar.selection().cells == _box(5, 4, 8, 6)
    _drag(picker.map, [_centre(picker, 6, 5), _centre(picker, 15, 15)])
    assert bar.selection().cells == _box(12, 13, 15, 15), "held at the edge, same size"


def test_dragging_a_corner_resizes_the_box(bar):
    picker = bar.region
    picker.set_region(_box(2, 2, 5, 4))
    corners = picker.map._corners((2, 2, 5, 4))
    bottom_right = corners[3].toPoint()
    assert picker.map.corner_at(corners[3]) == 3
    _drag(picker.map, [bottom_right, _centre(picker, 9, 8)])
    assert bar.selection().cells == _box(2, 2, 9, 8)
    top_left = picker.map._corners((2, 2, 9, 8))[0].toPoint()
    _drag(picker.map, [top_left, _centre(picker, 4, 5)])
    assert bar.selection().cells == _box(4, 5, 9, 8), "the opposite corner stays put"


def test_dragging_outside_the_box_starts_a_new_one(bar):
    picker = bar.region
    picker.set_region(_box(2, 2, 5, 4))
    _drag(picker.map, [_centre(picker, 10, 10), _centre(picker, 12, 11)])
    assert bar.selection().cells == _box(10, 10, 12, 11)


def test_a_click_inside_the_box_toggles_one_cell_and_the_handles_go(bar):
    from detail.gui.density_map import box_of

    picker = bar.region
    picker.set_region(_box(2, 2, 5, 4))
    assert box_of(picker.map.region()) == (2, 2, 5, 4)
    QTest.mouseClick(picker.map, Qt.MouseButton.LeftButton, pos=_centre(picker, 3, 3))
    assert bar.selection().cells == _box(2, 2, 5, 4) - {(3, 3)}
    assert box_of(picker.map.region()) is None, "a box with a hole has no corners"
    assert picker.map.corner_at(picker.map._corners((2, 2, 5, 4))[3]) is None


# --------------------------------------------------------------------------- #
# the lines ↔ selection
# --------------------------------------------------------------------------- #
FULL = Selection(
    values={"animal_id": frozenset({"A01", "A02", "A03", "A04"})},
    ranges={"weight_g": (25.0, 30.0)},
    recording_ids=frozenset({"recording_001", "recording_002"}),
    cells=frozenset({(0, 0), (1, 0)}),
    session=(0.25, 0.5),
)


def test_set_selection_shows_in_every_section_and_line(bar):
    bar.set_selection(FULL)
    assert [
        bar.field_row(name).summary_text()
        for name in ("animal_id", "weight_g", "Video")
    ] == ["4 of 24", "25–30", "2 of 6"]
    assert not bar.field_row("Location").reset_button.isHidden()
    assert bar._meta_sections["animal_id"].value() == FULL.values["animal_id"]
    assert bar._meta_sections["weight_g"].value() == (25.0, 30.0)
    assert bar._meta_sections["weight_g"].span_label.text() == "25 – 30"
    assert bar._video_list.value() == FULL.recording_ids
    assert bar.region.region() == FULL.cells
    assert bar.session.value() == FULL.session
    assert bar.field_row("Time in session").summary_text() == "25%–50%"
    assert not bar.field_row("Time in session").reset_button.isHidden()


def test_a_fields_reset_removes_exactly_its_condition(bar):
    bar.set_selection(FULL)
    bar.field_row("Video").reset_button.click()
    assert bar.selection() == FULL.without(VIDEO)
    assert bar._video_list.value() is None
    assert bar.field_row("Video").summary_text() == "6 recordings"
    bar.field_row("Location").reset_button.click()
    assert bar.selection() == FULL.without(VIDEO).without(LOCATION)


def test_round_trip_every_reset_clicked_leaves_nothing(bar):
    bar.set_selection(FULL)
    for name in ("animal_id", "weight_g", "Video"):
        bar.field_row(name).reset_button.click()
    bar.field_row("Location").reset_button.click()
    bar.field_row("Time in session").reset_button.click()
    assert bar.selection().is_empty
    assert all(r.reset_button.isHidden() for r in bar.field_rows())
    assert bar.field_row("Location").reset_button.isHidden()
    assert bar.field_row("Time in session").reset_button.isHidden()
    assert bar._meta_sections["weight_g"].value() is None
    assert bar.region.region() == frozenset()
    assert bar.session.value() is None
    assert bar.field_row("Time in session").summary_text() == "any"


def test_the_filter_column_has_no_pills(bar):
    """User, review round 8: the pills repeated what each field's line already says."""
    assert not hasattr(cf, "pill_text") and not hasattr(bar, "pill_texts")
    bar.set_selection(FULL)
    texts = [b.text() for b in bar.dock.body.findChildren(QPushButton)]
    assert "×" not in texts


def test_clear_all_empties_and_is_hidden_when_empty(bar):
    assert bar.clear_button.isHidden()
    bar.set_selection(FULL)
    assert bar.clear_button.isVisible()
    bar.clear_button.click()
    assert bar.selection().is_empty
    assert bar.clear_button.isHidden()


def test_reset_all_leaves_the_ticked_screen_which_is_on_the_other_tab(bar):
    """Reset all is on the Filters tab and removes that tab's filters. The screen is ticked
    on the Differences tab: a button that unticked it from here would change what is out of
    sight, and a screen alone gives Reset all nothing to do."""
    bar.set_screens(SCREENS)
    ticked = Selection(leaves=frozenset({0, 1}), leaves_source="cmp_02")
    bar.set_selection(ticked)
    assert bar.clear_button.isHidden(), "a screen alone is not a filter of this tab"
    bar.set_selection(
        Selection(leaves=frozenset({0, 1}), leaves_source="cmp_02", session=(0.0, 0.3))
    )
    assert bar.clear_button.isVisible()
    bar.clear_button.click()
    assert bar.selection() == ticked


def test_undoing_a_filter_sits_in_the_dock_under_the_key(bar):
    """User, review round 4: "putting the filter clearing options not in the filter
    column/section is weird and unintuitive". The count and Reset all are in the dock, between
    the match key and the sections, and nothing of the filters is drawn outside it."""
    dock = bar.dock
    bar.set_selection(FULL)
    bar.set_match_count(212, 1004)
    dock.key.set_filter(True, run_share=0.2)
    for widget in (bar.count_label, bar.clear_button):
        assert dock.body.isAncestorOf(widget), widget
        assert not dock.scroll.isAncestorOf(widget), "it scrolls away with the sections"
    assert not isinstance(bar, QWidget), "the filters draw nothing outside the dock"

    def top(widget):
        return widget.mapTo(dock, QPoint(0, 0)).y()

    assert top(dock.key) < top(bar.clear_button) < top(dock.scroll)


def test_the_shut_strip_counts_the_filters_that_are_on(bar):
    """Shut, the dock is a strip whose name says how many filters are on, so an active filter is
    never out of sight (the splitter paints it; this is the name it is handed)."""
    from detail.api.cluster_filter import Selection
    from detail.gui.cluster_filters import panel_name

    assert panel_name(Selection()) == "Filters"
    assert panel_name(FULL) == f"Filters · {len(FULL.keys())} active"
    assert len(FULL.keys()) >= 2, "the fixture puts on more than one filter"


def test_the_dock_has_no_hide_button_of_its_own(bar):
    """It folds by its splitter's grip, like every side panel (`gui/grip_splitter.py`)."""
    texts = [b.text() for b in bar.dock.findChildren(QPushButton)]
    assert not any("Hide" in t or "▸" in t or "◂" in t for t in texts), texts


def test_match_count_label(bar):
    assert bar.count_label.text() == "1,004 clips" and not bar.count_label.isHidden()
    bar.set_selection(FULL)
    bar.set_match_count(212, 1004)
    assert bar.count_label.text() == "212 of 1,004 clips match"
    assert not bar.count_label.isHidden()


# --------------------------------------------------------------------------- #
# the hover breakdown asks the API's rule, one condition at a time
# --------------------------------------------------------------------------- #
METADATA = {
    "recording_003": {"treatment": "drug_A", "animal_id": "A03", "dose_mg": "1"},
    "recording_004": {"treatment": "vehicle", "animal_id": "A04", "dose_mg": "5"},
}


def _clips():
    import numpy as np

    from detail.api.node_view import ClipTable

    n = 4
    return ClipTable(
        recording_ids=("recording_003", "recording_004"),
        recording=np.array([0, 0, 1, 1], dtype=np.int32),
        subject=np.zeros(n, np.int64),
        start_ns=np.arange(n, dtype=np.int64),
        end_ns=np.arange(n, dtype=np.int64),
        cx=np.array([0.01, 0.9, 0.01, 0.9], np.float32),
        cy=np.array([0.01, 0.9, 0.01, 0.9], np.float32),
        members=np.ones(n, np.int64),
        typicality=np.zeros(n),
        leaf=np.zeros(n, np.int64),
    )


def test_breakdown_marks_each_filter():
    sel = Selection(values={"animal_id": frozenset({"A03"})}, cells=frozenset({(5, 5)}))
    passed = {("meta", "animal_id"): True, LOCATION: False}
    text = cf.breakdown(sel, passed, METADATA["recording_003"])
    assert text == "✓ animal_id A03  ✗ location"


def test_every_condition_passing_is_exactly_a_match():
    """One rule: the per-condition ✓s the hover shows AND to the API's whole-selection match,
    for every clip — so a tile can never be grey while its hover says it passes everything.
    """
    import numpy as np

    from detail.api.cluster_filter import clip_matches

    clips = _clips()
    sel = Selection(
        values={"animal_id": frozenset({"A03", "A04"})},
        ranges={"dose_mg": (0.5, 2.0)},
        recording_ids=frozenset({"recording_003", "recording_004"}),
        cells=frozenset({(0, 0)}),
        session=(0.0, 0.5),
    )
    spans = {"recording_003": (0, 4), "recording_004": (0, 4)}
    assert SESSION in sel.keys()
    whole = clip_matches(sel, clips, METADATA, spans=spans)
    each = np.ones(len(clips), dtype=bool)
    for key in sel.keys():
        each &= clip_matches(sel.only(key), clips, METADATA, spans=spans)
    assert list(whole) == list(each) == [True, False, False, False]


def test_only_isolates_one_condition():
    assert FULL.only(LOCATION) == Selection(cells=FULL.cells)
    assert FULL.only(("meta", "weight_g")) == Selection(
        ranges={"weight_g": (25.0, 30.0)}
    )
    assert FULL.without(VIDEO).recording_ids is None


def test_search_fields_read_as_plain_english(bar):
    long = bar._meta_sections["animal_id"]
    assert long.search.placeholderText() == "Search animal_id…"
    assert bar._video_list.search.placeholderText() == "Search recordings…"


# --------------------------------------------------------------------------- #
# headings, the key at the top, time-of-day columns
# --------------------------------------------------------------------------- #
def test_a_field_line_is_the_column_name_without_its_type(bar):
    """ "date · date" and "time · categories" read as noise to the user who asked."""
    words = ("categories", "category", "number", "date", "time")
    for row in bar.field_rows():
        for text in (row.title.text(), row.summary_text()):
            assert not any(w in text.split() for w in words), text


def _visible_texts(widget) -> list[str]:
    return [
        label.text()
        for label in widget.findChildren(QLabel)
        if label.isVisibleTo(widget) and label.text()
    ]


def _top(dock, widget) -> int:
    return widget.mapTo(dock, QPoint(0, 0)).y()


def test_with_no_filter_the_key_is_greyed_over_the_runs_clip_count(bar, qapp):
    """User, review round 8 (option B): the key stays at the top, greyed with no marks, and the
    line says how many clips the run has. No Reset all: there is nothing to reset."""
    dock = bar.dock
    dock.resize(420, 900)
    dock.show()
    bar.set_match_count(None, 2781)
    dock.key.set_filter(False)
    qapp.processEvents()
    assert dock.key.isVisible() and not dock.key.is_active()
    assert dock.key.ramp.labels() == [], "no marks"
    assert dock.key.toolTip(), "greyed, saying why"
    assert bar.count_label.isVisible() and bar.count_label.text() == "2,781 clips"
    assert bar.count_label.toolTip() == cf.MATCH_RULE_TEXT
    assert bar.clear_button.isHidden()
    above = [
        label.text()
        for label in dock.body.findChildren(QLabel)
        if label.isVisibleTo(dock.body)
        and label.text()
        and not dock.scroll.isAncestorOf(label)
    ]
    assert above == ["Clips that match", "0%", "100%", "2,781 clips"], above
    tabs = dock.tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Filters", "Differences"]
    assert bar.open_tab() == cf.FILTERS_TAB, "the tab every run has something on"
    assert tabs.currentIndex() == 0, "first, and the one the dock opens on"


def test_the_fields_do_not_move_when_the_first_filter_goes_on(bar, qapp):
    """The key and the count line keep their space with or without a filter, so Location's
    heading — and every field under it — stays at one y (review round 8; round 7 hid the key
    and the column jumped)."""
    dock = bar.dock
    dock.resize(420, 900)
    dock.show()
    bar.set_match_count(None, 1004)
    dock.key.set_filter(False)
    qapp.processEvents()
    heading = bar.location_heading
    without = _top(dock, heading)
    bar.set_selection(FULL)
    bar.set_match_count(212, 1004)
    dock.key.set_filter(True, selected_share=0.3, run_share=0.2)
    qapp.processEvents()
    assert not bar.clear_button.isHidden()
    assert _top(dock, heading) == without
    bar.clear()
    bar.set_match_count(None, 1004)
    dock.key.set_filter(False)
    qapp.processEvents()
    assert _top(dock, heading) == without


def test_with_a_filter_the_key_and_count_line_say_what_matches(bar, qapp):
    dock = bar.dock
    dock.resize(420, 900)
    dock.show()
    bar.set_selection(FULL)
    bar.set_match_count(212, 1004)
    dock.key.set_filter(True, selected_share=0.3, run_share=0.2)
    qapp.processEvents()
    assert dock.key.isVisible() and dock.key.is_active()
    assert dock.key.ramp.labels() == ["selected 30%", "run 20%"]
    assert bar.count_label.isVisible() and bar.clear_button.isVisible()
    assert bar.count_label.text() == "212 of 1,004 clips match"
    assert bar.count_label.toolTip() == cf.MATCH_RULE_TEXT
    assert "every field" in cf.MATCH_RULE_TEXT and "any of them" in cf.MATCH_RULE_TEXT
    assert cf.MATCH_RULE_TEXT not in _visible_texts(dock.body), "the rule is a tooltip"
    assert (
        abs(_top(dock, bar.clear_button) - _top(dock, bar.count_label)) < 12
    ), "Reset all at the right of the count line"
    assert (
        _top(dock, dock.key)
        < _top(dock, bar.count_label)
        < _top(dock, dock.location_box)
    )
    bar.clear()
    qapp.processEvents()
    assert dock.key.isVisible() and bar.clear_button.isHidden()


def test_location_says_how_to_pick_in_a_tooltip_and_resets_from_its_line(bar, qapp):
    """User, review round 8: the instructions under the map read as clutter once learned. The
    Reset is the Location line's own, like every field's (laptop layout, 2026-11-09)."""
    picker = bar.region
    row = bar.field_row("Location")
    hint = "Click a square to add or remove it. Drag to draw a box"
    assert hint in picker.map.toolTip()
    assert not any(hint in t for t in _visible_texts(bar.dock.location_box))
    assert row.reset_button.isHidden()
    picker.set_region(_box(1, 1, 2, 2))
    qapp.processEvents()
    assert not row.reset_button.isHidden()
    assert row.reset_button.parent() is row.head, "in the Location line"
    assert picker.reset_button.isHidden(), "one Reset, not the map's own as well"
    assert row.summary_text() == "1 region chosen"
    row.reset_button.click()
    assert bar.selection().cells is None
    assert row.reset_button.isHidden()
    assert row.summary_text() == "any"


def test_a_time_column_with_few_values_is_ticked_in_clock_order(bar):
    section = bar._meta_sections["start"]
    assert isinstance(section, cf.Checklist)
    assert section.keys() == ["9:05", "14:00:21", "17:23:19"], "by time, not by text"
    _check(section, "9:05", False)
    assert bar.selection().values == {"start": frozenset({"14:00:21", "17:23:19"})}


def test_a_time_column_with_many_values_is_a_range_of_its_own_times(bar):
    section = bar._meta_sections["clock"]
    assert isinstance(section, cf.TimeRange), "14 times: a range"
    histogram = section.histogram
    _drag(histogram, [QPoint(histogram.width() - 1, 5), _bar_centre(histogram, 3)])
    assert bar.selection().ranges == {"clock": ("06:30", "09:30")}
    assert section.span_label.text() == "06:30 – 09:30"
    assert bar.field_row("clock").summary_text() == "06:30 – 09:30"
    section.set_value(("08:00", "12:30"))
    assert histogram.indices() == (2, 6)


# --------------------------------------------------------------------------- #
# one line per field
# --------------------------------------------------------------------------- #
@pytest.fixture
def closed_bar(qapp):
    """The bar as the page opens it: every field line closed."""
    widget = cf.FilterBar()
    widget.set_columns(COLUMNS)
    widget.set_recordings(RECORDINGS)
    widget.set_match_count(None, 1004)
    widget.dock.resize(420, 1200)
    widget.dock.show()
    yield widget
    widget.dock.close()


def _names(bar) -> list[str]:
    return [row.name for row in bar.field_rows() if not row.isHidden()]


def test_each_field_is_one_closed_line_with_its_value_count(closed_bar):
    bar = closed_bar
    rows = {row.name: row for row in bar.field_rows()}
    assert _names(bar) == [
        "Video",
        "treatment",
        "animal_id",
        "dose_mg",
        "recorded",
        "weight_g",
        "born",
        "cohort",
        "start",
        "clock",
    ], "the usual order: Video, then the columns; one-value columns are not openable"
    assert all(not row.is_open() for row in rows.values())
    assert rows["treatment"].title.text() == "▸ treatment"
    assert rows["treatment"].summary_text() == "3 values"
    assert rows["Video"].summary_text() == "6 recordings"
    assert rows["weight_g"].summary_text() == "31 values"
    assert bar.in_use_label.isHidden() and bar.in_use_divider.isHidden()


def test_clicking_a_line_opens_and_closes_its_values(closed_bar):
    row = closed_bar.field_row("treatment")
    QTest.mouseClick(row.head, Qt.MouseButton.LeftButton, pos=QPoint(8, 6))
    assert row.is_open() and row.title.text() == "▾ treatment"
    assert row.section.isVisible()
    summary = row.summary.mapTo(row.head, QPoint(2, 2))
    QTest.mouseClick(row.head, Qt.MouseButton.LeftButton, pos=summary)
    assert not row.is_open(), "anywhere on the line, the words included"
    assert row.head.toolTip() == "Show or hide the values of treatment"


def test_a_field_line_is_not_a_button(closed_bar):
    """User, review round 8: the names rose into buttons under the pointer — "too button-y and
    busy". The line is text; its only button is the Reset it shows while filtered."""
    row = closed_bar.field_row("treatment")
    assert not isinstance(row.head, QAbstractButton)
    assert [type(b).__name__ for b in row.head.findChildren(QAbstractButton)] == [
        "QPushButton"
    ]
    assert row.head.findChildren(QAbstractButton) == [row.reset_button]
    assert isinstance(row.title, QLabel)


def test_a_field_line_is_reachable_from_the_keyboard(closed_bar, qapp):
    row = closed_bar.field_row("treatment")
    assert row.head.focusPolicy() & Qt.FocusPolicy.TabFocus
    row.head.setFocus()
    qapp.processEvents()
    QTest.keyClick(row.head, Qt.Key.Key_Space)
    assert row.is_open()
    QTest.keyClick(row.head, Qt.Key.Key_Return)
    assert not row.is_open()


def test_a_field_line_tints_under_the_pointer(closed_bar, qapp):
    from PyQt6.QtGui import QEnterEvent
    from PyQt6.QtCore import QEvent, QPointF

    row = closed_bar.field_row("treatment")
    head = row.head
    head.resize(head.width(), head.height())
    plain = head.grab().toImage().pixelColor(head.width() // 2, 2)
    point = QPointF(head.width() / 2, 2)
    qapp.sendEvent(head, QEnterEvent(point, point, point))
    assert head.is_hovered()
    tinted = head.grab().toImage().pixelColor(head.width() // 2, 2)
    assert tinted != plain, "the pointer over the line shades it"
    qapp.sendEvent(head, QEvent(QEvent.Type.Leave))
    assert not head.is_hovered()


def test_the_summary_says_the_filter_on_the_field():
    names = {c.name: c for c in COLUMNS}
    recs = {r.recording_id: r.name for r in RECORDINGS}

    def summary(selection, key):
        return cf.field_summary(selection, key, names, recs)

    treatment = ("meta", "treatment")
    assert summary(Selection(), treatment) == "3 values"
    one = Selection(values={"treatment": frozenset({"drug_A"})})
    assert summary(one, treatment) == "drug_A only"
    two = Selection(values={"treatment": frozenset({"drug_A", ""})})
    assert summary(two, treatment) == "2 of 3"
    assert summary(Selection(values={"treatment": frozenset()}), treatment) == "none"
    span = Selection(ranges={"weight_g": (25.0, 30.0)})
    assert summary(span, ("meta", "weight_g")) == "25–30"
    video = Selection(recording_ids=frozenset({"recording_002"}))
    assert summary(video, VIDEO) == "A02_drug only"


def test_fields_in_use_open_and_sort_to_the_top_under_a_caption(closed_bar):
    bar = closed_bar
    bar.set_selection(
        Selection(
            values={"dose_mg": frozenset({"1"})},
            recording_ids=frozenset({"recording_001", "recording_002"}),
        )
    )
    assert _names(bar)[:2] == ["Video", "dose_mg"], "in use first, in the usual order"
    assert _names(bar)[2:4] == ["treatment", "animal_id"], "then the rest"
    assert bar.field_row("dose_mg").is_open() and bar.field_row("Video").is_open()
    assert not bar.field_row("treatment").is_open()
    assert bar.field_row("dose_mg").summary_text() == "1 only"
    assert bar.field_row("Video").summary_text() == "2 of 6"
    assert not bar.in_use_label.isHidden() and not bar.in_use_divider.isHidden()
    layout = bar.dock.fields_box.layout()
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert (
        order.index(bar.in_use_label)
        < order.index(bar.field_row("dose_mg"))
        < order.index(bar.in_use_divider)
        < order.index(bar.field_row("treatment"))
    )
    bar.clear()
    assert _names(bar)[:2] == ["Video", "treatment"], "cleared: back in place"
    assert bar.in_use_label.isHidden()


def test_a_filter_set_inside_a_field_waits_for_the_pointer_to_leave(closed_bar):
    """Moving the line between two clicks would put the next click on another field."""
    bar = closed_bar
    row = bar.field_row("cohort")
    row.set_open(True)
    _check(row.section, "1", False)
    assert _names(bar).index("cohort") > 1, "not moved under the hand"
    assert row.summary_text() == "(no value) only", "the summary is live"
    row.left.emit()
    assert _names(bar)[0] == "cohort"
    assert not bar.in_use_label.isHidden()


def test_find_a_field_filters_the_lines_by_name(closed_bar):
    bar = closed_bar
    find = bar.dock.find_field
    assert not bar.dock.find_box.isHidden(), "12 fields: more than 8"
    assert find.placeholderText() == "Find a field (12)"
    find.setText("an")
    assert _names(bar) == ["animal_id"]
    assert [r.name for r in bar.constant_rows() if not r.isHidden()] == ["animal"]
    find.setText("")
    assert len(_names(bar)) == 10


def test_find_a_field_is_absent_over_eight_fields_or_fewer(qapp):
    widget = cf.FilterBar()
    widget.set_columns(COLUMNS[:3])
    widget.set_recordings(RECORDINGS)
    assert widget.dock.find_box.isHidden()
    widget.dock.close()


def test_a_folded_location_line_is_spaced_like_every_field_line(qapp):
    """Folded, Location is one more line in the list: the same height, and the same distance to
    the line under it as any field to the next. A gap after it (``addSpacing`` plus the column's
    own spacing, against none between field lines) made it read as a taller row (user,
    2026-11-09). Measured top to top, so a gap anywhere between two lines counts."""
    widget = cf.FilterBar()
    widget.set_columns(COLUMNS[:3])
    widget.set_recordings(RECORDINGS)
    widget.set_match_count(None, 1004)
    widget.dock.resize(420, 900)
    widget.dock.show()
    widget.location_row.set_open(False)
    widget.session_row.set_open(False)
    qapp.processEvents()
    dock = widget.dock
    rows = [
        widget.location_row,
        widget.session_row,
        *widget.field_rows(),
    ]
    assert len(rows) >= 3 and all(not r.is_open() for r in rows)
    tops = [r.mapTo(dock, QPoint(0, 0)).y() for r in rows]
    pitches = [b - a for a, b in zip(tops, tops[1:])]
    heights = {r.height() for r in rows}
    assert len(heights) == 1, f"folded lines of different heights: {heights}"
    assert (
        len(set(pitches)) == 1
    ), f"Location to the next line is {pitches[0]}px, a field to the next {pitches[1:]}"
    widget.dock.close()


def test_the_map_fills_the_docks_width_within_its_bounds(bar, qapp):
    """Laptop layout, decision 2 C: the dock drags to 240px and the map follows it, rather than
    holding the dock at the map's fixed 320."""
    dock = bar.dock
    assert dock.body.minimumWidth() == cf.DOCK_MIN_PX == 240
    dock.show()
    widths = {}
    try:
        for width in (cf.DOCK_MIN_PX, 420, 1200):
            dock.resize(width, 900)
            qapp.processEvents()
            widths[width] = bar.region.map.side
            assert bar.region.map.side <= dock.scroll.viewport().width()
        assert widths[cf.DOCK_MIN_PX] < widths[420], "narrower dock, smaller map"
        assert widths[1200] == cf.MAP_MAX_PX, "and it stops growing"
        assert widths[cf.DOCK_MIN_PX] >= cf.MAP_MIN_PX
    finally:
        dock.hide()


def test_location_folds_like_a_field_says_its_pick_and_is_remembered(qapp):
    first = cf.FilterBar()
    row = first.field_row("Location")
    assert row.is_open(), "open until folded"
    assert row.summary_text() == "any"
    row.head.toggled.emit()  # a click on the line
    assert not row.is_open()
    first.set_selection(cf.Selection(cells=frozenset({(1, 1), (1, 2), (9, 9)})))
    assert row.summary_text() == "2 regions chosen"
    assert not row.reset_button.isHidden()
    again = cf.FilterBar()
    assert not again.field_row("Location").is_open(), "folded, as it was left"
    again.field_row("Location").head.toggled.emit()
    assert cf.FilterBar().field_row("Location").is_open()


def test_location_summary_counts_touching_squares_as_one_region():
    assert cf.location_summary(None) == "any"
    assert cf.location_summary({(0, 0), (0, 1), (1, 1)}) == "1 region chosen"
    assert (
        cf.location_summary({(0, 0), (1, 1)}) == "2 regions chosen"
    ), "corners don't join"


def test_a_one_value_line_is_greyed_under_a_palette_that_leaves_disabled_text_bright(
    qapp,
):
    """The dark render's bug: a selectable label paints with the Text role, and a palette that
    sets Text for every group (a dark palette built role by role) left these lines as bright as a
    live field. They take the colour a plain disabled label uses."""
    from PyQt6.QtGui import QColor, QPalette

    from detail.gui.selectable import make_text_selectable

    palette = QPalette(QColor(60, 60, 60), QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.Text, QColor(230, 230, 230))
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(120, 120, 120),
    )
    holder = QWidget()
    holder.setPalette(palette)
    row = cf.ConstantRow("animal", "F3", 3)
    row.setParent(holder)
    make_text_selectable(holder)
    qapp.processEvents()
    for label in (row.name_label, row.value_label):
        got = label.palette().color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text
        )
        assert got == QColor(120, 120, 120), label.text()


# --------------------------------------------------------------------------- #
# Time in session
# --------------------------------------------------------------------------- #
def _session_bar(qapp):
    widget = cf.FilterBar()
    widget.set_columns(COLUMNS[:3])
    widget.set_recordings(RECORDINGS)
    widget.set_match_count(None, 1004)
    widget.seen = []
    widget.selection_changed.connect(widget.seen.append)
    widget.dock.resize(420, 1400)
    widget.dock.show()
    row = widget.field_row("Time in session")
    row.set_open(True)
    qapp.processEvents()
    return widget


def _at(slider, fraction: float) -> QPoint:
    """The point ``fraction`` of the way along the track, which is set in by half a knob."""
    inset = slider.KNOB_W_PX / 2
    x = inset + fraction * (slider.width() - 2 * inset)
    return QPoint(int(round(x)), slider.height() // 2)


def test_session_is_a_plain_slider_with_start_middle_end_and_its_line(qapp):
    """No bars: the same range widget with ``bars=False``, one step per percent, a plain track;
    the ticks read start · middle · end and the line "Whole session" until a range is picked.
    """
    bar = _session_bar(qapp)
    try:
        section = bar.session
        slider = section.slider
        assert isinstance(slider, cf.RangeHistogram) and not slider._bars
        assert slider.indices() == (0, cf.SESSION_STEPS - 1)
        assert [t.text() for t in section.ticks] == ["start", "middle", "end"]
        assert section.span_label.text() == "Whole session"
        assert bar.field_row("Time in session").summary_text() == "any"
        assert section.value() is None
    finally:
        bar.dock.close()


def test_dragging_the_session_handles_picks_a_range_and_says_it(qapp):
    bar = _session_bar(qapp)
    try:
        slider = bar.session.slider
        assert slider.width() > 200, "the dock must be laid out for the handles to land"
        _drag(slider, [_at(slider, 0.0), _at(slider, 0.12)])
        _drag(slider, [_at(slider, 1.0), _at(slider, 0.4)])
        assert bar.selection().session == (0.12, 0.4)
        assert bar.selection().keys() == [SESSION]
        row = bar.field_row("Time in session")
        assert row.summary_text() == "12%–40%"
        assert not row.reset_button.isHidden()
        assert bar.session.span_label.text() == "12%–40% of each recording"
        bar.set_session_count(840)
        assert bar.session.span_label.text() == "12%–40% of each recording · 840 clips"
        assert cf.panel_name(bar.selection()) == "Filters · 1 active"
        row.reset_button.click()
        assert bar.selection().is_empty
        assert bar.session.span_label.text() == "Whole session"
        assert slider.indices() == (0, cf.SESSION_STEPS - 1)
    finally:
        bar.dock.close()


def test_the_session_handles_move_from_the_keyboard(qapp):
    """Tab reaches the left handle and Tab again the right; the arrows move the focused one by a
    step (Shift: ten), and every move is a selection change."""
    bar = _session_bar(qapp)
    try:
        slider = bar.session.slider
        slider.setFocus(Qt.FocusReason.TabFocusReason)
        qapp.processEvents()
        assert slider.key_handle() == "first"
        QTest.keyClick(slider, Qt.Key.Key_Right)
        QTest.keyClick(slider, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        assert bar.selection().session == (0.11, 1.0)
        QTest.keyClick(slider, Qt.Key.Key_Tab)
        assert slider.key_handle() == "last" and slider.hasFocus()
        QTest.keyClick(slider, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(slider, Qt.Key.Key_Left)
        assert bar.selection().session == (0.11, 0.89)
        assert bar.field_row("Time in session").summary_text() == "11%–89%"
        for _ in range(200):
            QTest.keyClick(slider, Qt.Key.Key_Left)
        assert slider.indices()[1] == slider.indices()[0], "never past the other handle"
        QTest.keyClick(slider, Qt.Key.Key_Backtab)
        assert slider.key_handle() == "first"
    finally:
        bar.dock.close()


def test_session_folds_like_location_and_its_fold_is_remembered(qapp):
    first = cf.FilterBar()
    row = first.field_row("Time in session")
    assert not row.is_open(), "shut until opened"
    row.head.toggled.emit()
    assert row.is_open()
    assert cf.FilterBar().field_row("Time in session").is_open(), "open, as it was left"
    cf.FilterBar().field_row("Time in session").head.toggled.emit()
    assert not cf.FilterBar().field_row("Time in session").is_open()


def test_session_is_greyed_with_its_reason_when_no_recording_has_a_map(qapp):
    bar = _session_bar(qapp)
    try:
        bar.set_session_available(False, "No timestamp maps.")
        assert not bar.session.slider.isEnabled()
        assert bar.session.slider.toolTip() == "No timestamp maps."
        assert not bar.session.note.isHidden()
        bar.set_session_available(True, "")
        assert bar.session.slider.isEnabled() and bar.session.note.isHidden()
        assert "arrow keys" in bar.session.slider.toolTip()
    finally:
        bar.dock.close()


def test_the_session_summary_and_breakdown():
    assert cf.session_summary(None) == "any"
    assert cf.session_summary((0.12, 0.4)) == "12%–40%"
    sel = Selection(session=(0.1, 0.2))
    assert cf.breakdown(sel, {SESSION: False}, {}) == "✗ time in session"


# --------------------------------------------------------------------------- #
# no value widens the panel (the recorded class in the module docstring)
# --------------------------------------------------------------------------- #
LONG_TREATMENT = "saline_vehicle_control_with_a_very_long_condition_name"
LONG_COLUMNS = [
    Column("treatment", CATEGORY, {LONG_TREATMENT: 120, "treatment_a": 98}),
    Column(
        "a_long_metadata_column_name_from_a_spreadsheet", CATEGORY, {"x": 1, "y": 2}
    ),
    Column("weight_g", NUMBER, {f"{w:g}": 7 for w in WEIGHTS} | {"": 3}),
    Column("born", DATE, {b: 5 for b in BIRTHS}),
    Column("animal", CATEGORY, {"S1_with_a_long_identifier_from_the_rig_sheet": 223}),
    Column("clock", TIME, {f"{h:02d}:30:15": 4 for h in range(6, 20)}),
]
LONG_RECORDINGS = [
    RecordingChoice(
        f"recording_{i:03d}", f"drug_treatment_pair_cohort2_day{i:02d}_cam1", 9
    )
    for i in range(1, 5)
]


def _long_steps():
    """The values a user sets one after another, each the longest a real run would show."""
    session = [None, (0.0, 1.0), (0.5, 1.0), (0.12, 0.4)]
    steps = [Selection(session=s) for s in session if s is not None]
    steps.insert(0, Selection())
    full = Selection(
        session=(0.12, 0.4),
        recording_ids=frozenset({"recording_001"}),
        cells=frozenset({(1, 1), (5, 5), (9, 9)}),
    )
    full = cf.with_condition(full, "treatment", frozenset({LONG_TREATMENT}))
    full = cf.with_condition(full, "weight_g", (21.0, 48.0))
    full = cf.with_condition(full, "born", ("2026-01-02", "2026-01-14"))
    full = cf.with_condition(full, "clock", ("07:30:15", "18:30:15"))
    return [*steps, full, Selection()]


def overflow(dock) -> list[str]:
    """How the panel's content is wider than the panel, as sentences; empty when it fits.

    Two places: the scrolled column (wider than its viewport, its right edge is cut — the scroll
    never scrolls sideways) and the strip above it (key, count line, Reset all), which cannot be
    narrower than its minimum and so pushes the dock wider than the splitter gave it."""
    problems = []
    viewport = dock.scroll.viewport().width()
    if dock.sections.width() > viewport:
        problems.append(f"fields {dock.sections.width()}px in a {viewport}px viewport")
    if dock.body.minimumSizeHint().width() > dock.width():
        problems.append(
            f"panel needs {dock.body.minimumSizeHint().width()}px of {dock.width()}px"
        )
    return problems


def _laid_out(dock, qapp) -> None:
    qapp.processEvents()
    dock.sections.adjustSize()
    dock.sections.layout().activate()
    qapp.processEvents()


def _columns_x(bar) -> dict[str, tuple[int, int]]:
    """Each line's left edge and width in the dock — what must not move sideways."""
    rows = [bar.location_row, bar.session_row, *bar.field_rows(), *bar.constant_rows()]
    return {
        getattr(r, "name", "?"): (r.mapTo(bar.dock, QPoint(0, 0)).x(), r.width())
        for r in rows
        if r.isVisibleTo(bar.dock)
    }


@pytest.mark.parametrize("width", [350, cf.DOCK_MIN_PX])
def test_no_value_makes_the_filters_panel_wider_than_itself(qapp, width):
    """User, 2026-11-09: dragging Time in session to 50%–100% put "50%–100%" and a Reset in its
    line, the column grew wider than the panel, and its right edge was cut ("end" → "enc", "F3" →
    "F"). Every section at its longest realistic value, at the default width and the narrowest:
    the content fits, and no line moves sideways as values change."""
    bar = cf.FilterBar()
    bar.set_columns(LONG_COLUMNS)
    bar.set_recordings(LONG_RECORDINGS)
    bar.set_match_count(None, 99_999)
    dock = bar.dock
    dock.resize(width, 3000)
    dock.show()
    try:
        for row in [bar.location_row, bar.session_row, *bar.field_rows()]:
            row.set_open(True)
        _laid_out(dock, qapp)
        first = _columns_x(bar)
        assert len(first) >= 8, first
        for step in _long_steps():
            bar.set_selection(step)
            bar.set_session_count(None if step.session is None else 12_345)
            bar.set_match_count(None if step.is_empty else 12_345, 99_999)
            dock.key.set_filter(
                not step.is_empty, selected_share=0.333, run_share=0.125
            )
            _laid_out(dock, qapp)
            assert overflow(dock) == [], (step, overflow(dock))
            assert _columns_x(bar) == first, step
            assert bar.session.slider.width() <= dock.scroll.viewport().width()
    finally:
        dock.close()


def test_the_overflow_check_would_catch_a_value_that_widens_a_line(qapp):
    """The negative case: a plain label carrying a long value in a field's line is the bug, and
    the check sees it."""
    bar = cf.FilterBar()
    bar.set_columns(LONG_COLUMNS)
    bar.set_recordings(LONG_RECORDINGS)
    bar.set_match_count(None, 1004)
    dock = bar.dock
    dock.resize(cf.DOCK_MIN_PX, 1500)
    dock.show()
    try:
        _laid_out(dock, qapp)
        assert overflow(dock) == []
        wide = QLabel(LONG_TREATMENT + " only")
        bar.session_row.head.layout().addWidget(wide)
        _laid_out(dock, qapp)
        assert overflow(dock), "a plain label's minimum is its whole text"
    finally:
        dock.close()


def test_an_elided_line_says_its_whole_text_in_its_tooltip(qapp):
    bar = cf.FilterBar()
    bar.set_columns(LONG_COLUMNS)
    bar.set_recordings(LONG_RECORDINGS)
    bar.set_match_count(None, 1004)
    dock = bar.dock
    dock.resize(cf.DOCK_MIN_PX, 1500)
    dock.show()
    try:
        bar.set_selection(
            cf.with_condition(Selection(), "treatment", frozenset({LONG_TREATMENT}))
        )
        _laid_out(dock, qapp)
        row = bar.field_row("treatment")
        assert (
            row.summary_text() == f"{LONG_TREATMENT} only"
        ), "text() is the whole value"
        assert row.summary.is_elided()
        assert f"{LONG_TREATMENT} only" in row.head.tooltip_text()
        assert (
            row.head.toolTip() in row.head.tooltip_text()
        ), "and still what the line does"
    finally:
        dock.close()


def _ink_box(image, colours, x0: int, x1: int) -> tuple[int, int]:
    """Width and height of the pixels exactly one of ``colours`` in columns ``x0..x1``."""
    wanted = {colour.rgb() for colour in colours}
    xs, ys = [], []
    for x in range(max(x0, 0), min(x1, image.width())):
        for y in range(image.height()):
            if image.pixelColor(x, y).rgb() in wanted:
                xs.append(x)
                ys.append(y)
    if not xs:
        return 0, 0
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def test_the_session_slider_is_big_enough_to_grab(qapp):
    """User, 2026-11-09: the bar-less slider read too small — a 3px track and 8px squares. The
    track is at least 6px and each handle about a line of text tall, measured off the picture.
    """
    from PyQt6.QtGui import QColor, QPalette

    from detail.gui import plot_theme

    slider = cf.RangeHistogram([i / 100 for i in range(100)], [0] * 100, bars=False)
    slider.resize(300, slider.height())
    slider.set_indices(30, 69)
    slider.show()
    qapp.processEvents()
    image = slider.grab().toImage()
    ratio = image.devicePixelRatio()
    theme = plot_theme.current()
    left, right = slider.handle_x()
    # The handle is drawn in the text colour, or the highlight while it has the keyboard.
    inks = [QColor(theme.text), slider.palette().color(QPalette.ColorRole.Highlight)]
    handle_w, handle_h = _ink_box(
        image, inks, int((left - 12) * ratio), int((left + 12) * ratio)
    )
    _track_w, track_h = _ink_box(
        image, [QColor(theme.neutral)], int(10 * ratio), int(40 * ratio)
    )
    # Measured inside the knob's 1px border and its anti-aliased edge: ~2px under its size.
    line = cf.lines(1)
    assert handle_h / ratio >= 0.85 * line and handle_w / ratio >= 8, (
        handle_w,
        handle_h,
    )
    assert track_h / ratio >= 6, track_h
    slider.close()


# --------------------------------------------------------------------------- #
# a refill never leaves a condition the dock cannot show
# --------------------------------------------------------------------------- #
def _only(qapp, checklist, key) -> None:
    """The row's **only**, pressed with the mouse."""
    row = checklist.keys().index(key)
    checklist.list.setCurrentRow(row)
    qapp.processEvents()
    QTest.mouseClick(checklist.only_button, Qt.MouseButton.LeftButton)


def _refilled(columns, **counts):
    """``columns`` with the named ones' clips per value replaced."""
    return [
        Column(c.name, c.kind, counts[c.name]) if c.name in counts else c
        for c in columns
    ]


@pytest.mark.parametrize(
    "case",
    ["one value now", "removed", "a range on a checklist now", "its values gone"],
)
def test_a_refill_drops_a_condition_it_can_no_longer_show_and_says_so(qapp, bar, case):
    """The page refills the columns whenever its metadata is read again. A condition on a column
    that became one value (a greyed line with no Reset), went away, changed from a range to tick
    boxes, or lost every value it ticked stayed in the selection: still applied, still counted
    in "Filters · 1 active", with nothing in the dock showing it or able to reset it. And the
    prune that did happen (a removed column) emitted nothing, so the page kept shading by the
    condition the dock had dropped."""
    if case == "a range on a checklist now":
        histogram = bar._meta_sections["weight_g"].histogram
        _drag(histogram, [QPoint(1, 5), _bar_centre(histogram, 5)])
        assert "weight_g" in bar.selection().ranges
        columns = _refilled(COLUMNS, weight_g={"20": 5, "21": 6})
    else:
        _only(qapp, bar._meta_sections["treatment"], "drug_A")
        assert bar.selection().values == {"treatment": frozenset({"drug_A"})}
        if case == "one value now":
            columns = _refilled(COLUMNS, treatment={"vehicle": 223})
        elif case == "removed":
            columns = [c for c in COLUMNS if c.name != "treatment"]
        else:
            columns = _refilled(COLUMNS, treatment={"vehicle": 200, "saline": 23})
    bar.seen.clear()
    bar.set_columns(columns)
    assert bar.selection() == Selection(), "a condition the dock cannot show"
    assert cf.panel_name(bar.selection()) == "Filters"
    assert bar.seen == [Selection()], "the page was not told"


def test_a_refill_that_keeps_every_condition_emits_nothing(qapp, bar):
    _only(qapp, bar._meta_sections["treatment"], "drug_A")
    bar.seen.clear()
    bar.set_columns(COLUMNS)
    bar.set_recordings(RECORDINGS)
    assert bar.selection().values == {"treatment": frozenset({"drug_A"})}
    assert bar.seen == []


def test_a_refill_drops_recordings_the_run_no_longer_has(qapp, bar):
    _only(qapp, bar._video_list, "recording_003")
    assert bar.selection().recording_ids == frozenset({"recording_003"})
    bar.seen.clear()
    bar.set_recordings([r for r in RECORDINGS if r.recording_id != "recording_003"])
    assert bar.selection().recording_ids is None
    assert bar.seen == [Selection()]


def test_the_location_map_is_reached_and_picked_from_the_keyboard(qapp, bar):
    """The map took no focus and read no key, so a keyboard user could not reach the Location
    filter at all. Tab reaches it, the arrows move a ringed cell, Space adds or removes it.
    """
    picker = bar.region
    picker.map.set_side(320)
    qapp.processEvents()
    assert picker.map.focusPolicy() & Qt.FocusPolicy.TabFocus, "Tab skips the map"
    picker.map.setFocus()
    qapp.processEvents()
    assert picker.map.cursor_cell() == (0, 0)
    for key in (Qt.Key.Key_Right, Qt.Key.Key_Right, Qt.Key.Key_Down):
        QTest.keyClick(picker.map, key)
    assert picker.map.cursor_cell() == (2, 1)
    QTest.keyClick(picker.map, Qt.Key.Key_Space)
    assert bar.selection().cells == frozenset({(2, 1)})
    QTest.keyClick(picker.map, Qt.Key.Key_Left)
    QTest.keyClick(picker.map, Qt.Key.Key_Space)
    assert bar.selection().cells == frozenset({(2, 1), (1, 1)})
    for _ in range(40):
        QTest.keyClick(picker.map, Qt.Key.Key_Up)
    assert picker.map.cursor_cell() == (1, 0), "held to the grid's edge"
    QTest.keyClick(picker.map, Qt.Key.Key_Down)
    QTest.keyClick(picker.map, Qt.Key.Key_Space)
    assert bar.selection().cells == frozenset({(2, 1)}), "Space takes a cell back out"


# --------------------------------------------------------------------------- #
# the Differences line: saved screens, one at a time, and Run a new screen… (§22)
# --------------------------------------------------------------------------- #

SCREENS = [
    cf.ScreenChoice(
        "cmp_02", "drug vs vehicle · paired", "1 kept of 4", frozenset({0, 1})
    ),
    cf.ScreenChoice("cmp_01", "dose · trend", "cannot be read: no stats.csv", None),
]


def _tick_screen(bar, key: str, on: bool = True) -> None:
    lst = bar.screens.list
    for i in range(lst.count()):
        item = lst.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == key:
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            return
    raise AssertionError(f"{key!r} is not offered")


def test_the_differences_line_lists_the_runs_screens_and_greys_one_it_cannot_read(bar):
    assert bar.screens.note.text() == cf.NO_SCREENS_TEXT
    bar.set_screens([], loading=True)
    assert bar.screens.note.text() == cf.SCREENS_LOADING_TEXT
    bar.set_screens(SCREENS)
    lst = bar.screens.list
    assert [lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lst.count())] == [
        "cmp_02",
        "cmp_01",
    ]
    assert "1 kept of 4" in lst.item(0).text()
    assert not (lst.item(1).flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "cannot be read" in lst.item(1).toolTip()
    assert bar.differences_summary() == "none"
    assert bar.screens.note.text() == cf.SCREENS_HINT_TEXT
    assert bar.dock.differences_box.isAncestorOf(lst), "on the Differences tab"
    assert not bar.dock.filters_page.isAncestorOf(lst)
    # The row is cut with "…", so why it cannot be ticked is said in full under the list.
    assert bar.screens.unavailable.text() == (
        "Can't be ticked: dose · trend (cannot be read: no stats.csv)."
    )
    assert not bar.screens.unavailable.isHidden() and bar.screens.unavailable.wordWrap()
    bar.set_screens([SCREENS[0]])
    assert (
        bar.screens.unavailable.isHidden()
    ), "every screen can be ticked: nothing to say"


def test_ticking_a_screen_is_the_clusters_condition_and_its_reset_removes_it(bar):
    bar.set_screens(SCREENS)
    _tick_screen(bar, "cmp_02")
    selection = bar.seen[-1]
    assert selection.leaves == frozenset({0, 1})
    assert selection.leaves_source == "cmp_02"
    assert selection.keys() == [cf.CLUSTERS]
    assert bar.differences_summary() == "drug vs vehicle · paired"
    tabs = bar.dock.tabs
    assert tabs.tabText(1) == "Differences · 1", "the tab says a screen is on"
    assert tabs.tabText(0) == "Filters", "and that no field is"
    bar.remove(cf.CLUSTERS)
    assert bar.seen[-1].is_empty
    assert bar.screens.value() is None
    assert bar.differences_summary() == "none"
    assert tabs.tabText(1) == "Differences"


def test_one_screen_at_a_time(bar):
    bar.set_screens(
        [
            *SCREENS,
            cf.ScreenChoice("cmp_03", "sex · between", "0 kept of 0", frozenset({2})),
        ]
    )
    _tick_screen(bar, "cmp_02")
    _tick_screen(bar, "cmp_03")
    assert bar.screens.value() == "cmp_03"
    assert bar.seen[-1].leaves == frozenset({2})
    _tick_screen(bar, "cmp_03", on=False)
    assert bar.seen[-1].leaves is None


def test_a_screen_no_longer_offered_takes_its_condition_with_it(bar):
    """The rule every other field keeps: a filter with no line to show and reset it is
    removed, never applied out of sight."""
    bar.set_screens(SCREENS)
    _tick_screen(bar, "cmp_02")
    bar.set_screens([SCREENS[1]])
    assert bar.selection().leaves is None
    assert bar.differences_summary() == "none"


def test_a_ticked_screen_survives_the_list_being_read_again(bar):
    """The page re-reads its screens on every load and after a screen is run. While it
    reads, the offer is unknown, not empty: the ticked screen stays applied, under its
    label, and is checked against the list when it lands."""
    bar.set_screens(SCREENS)
    _tick_screen(bar, "cmp_02")
    bar.set_screens([], loading=True)
    assert bar.selection().leaves == frozenset({0, 1})
    assert bar.selection().leaves_source == "cmp_02"
    assert bar.differences_summary() == "drug vs vehicle · paired"
    bar.set_screens(SCREENS)
    assert bar.selection().leaves_source == "cmp_02"
    assert bar.screens.value() == "cmp_02"


def test_a_screen_read_again_over_a_re_cut_tree_applies_its_new_leaves(bar):
    bar.set_screens(SCREENS)
    _tick_screen(bar, "cmp_02")
    recut = cf.ScreenChoice(
        "cmp_02", "drug vs vehicle · paired", "1 kept of 4", frozenset({0, 1, 2})
    )
    bar.set_screens([recut, SCREENS[1]])
    assert bar.selection().leaves == frozenset({0, 1, 2})
    assert bar.seen[-1].leaves == frozenset({0, 1, 2}), "the page is told"


def test_a_selection_naming_a_screen_shows_in_the_line(bar):
    bar.set_screens(SCREENS)
    bar.set_selection(Selection(leaves=frozenset({0, 1}), leaves_source="cmp_02"))
    assert bar.screens.value() == "cmp_02"
    assert bar.differences_summary() == "drug vs vehicle · paired"
    assert cf.panel_name(bar.selection()) == "Filters · 1 active"


def test_run_a_new_screen_is_the_pages_to_answer(bar):
    asked = []
    bar.screen_requested.connect(lambda: asked.append(True))
    bar.screens.run_button.click()
    assert asked == [True]


def test_the_breakdown_names_the_clusters_condition():
    selection = Selection(
        leaves=frozenset({0}), leaves_source="cmp_02", recording_ids=frozenset({"r1"})
    )
    text = cf.breakdown(selection, {cf.CLUSTERS: True, VIDEO: False}, {})
    assert "✓ differs" in text and "✗ video" in text


def test_effect_shades_is_one_map_from_a_screen_to_the_blocks():
    """Every node's place on the −1..+1 axis and its hover line, through the key's own
    `effect_position`: NaN where the screen has no row or the effect is undefined, 0 inside the
    floor, the signed effect over the axis's reach otherwise."""
    import math

    from detail.gui.match_key import LOG_RATIO_EXTENT

    choice = cf.ScreenChoice(
        "cmp_02",
        "drug vs vehicle · paired",
        "1 kept of 2",
        frozenset({0, 1}),
        effects={
            4: cf.NodeEffect(math.log(12.4), True, True, "more under drug, 12.4×"),
            5: cf.NodeEffect(math.log(1.3), False, False, "more under drug, 1.3×"),
            2: cf.NodeEffect(-math.log(3.0), True, False, "less under drug, 3.0×"),
            3: cf.NodeEffect(None, False, False, "undefined: present in 2 ID(s)"),
        },
        floor=math.log(7.14),
        design="paired",
        caption="Less ↔ more under drug",
    )
    shades, texts = cf.effect_shades(choice, 7)
    assert shades[4] == pytest.approx(math.log(12.4) / LOG_RATIO_EXTENT)
    assert shades[2] == pytest.approx(-math.log(3.0) / LOG_RATIO_EXTENT)
    assert shades[5] == 0.0
    assert (
        shades[3] != shades[3] and shades[6] != shades[6]
    ), "undefined and no row: NaN"
    assert texts[4] == "more under drug, 12.4× · kept"
    assert texts[2].endswith("clears the floor, nested under a kept cluster")
    assert texts[5].endswith("inside the floor")
    assert texts[3] == "undefined: present in 2 ID(s)"
    assert 6 not in texts


def test_effect_shades_read_a_profile_as_a_size():
    """A profile screen's blocks sit on the 0..+1 half by magnitude, whichever way each
    cluster's own pair runs, and inside the floor is still neutral (§22)."""
    import math

    from detail.gui.match_key import LOG_RATIO_EXTENT

    choice = cf.ScreenChoice(
        "cmp_03",
        "session · profile",
        "1 kept of 2",
        frozenset({0}),
        effects={
            4: cf.NodeEffect(
                -math.log(12.4),
                True,
                True,
                "differs most between fent and nalox: less under nalox than fent, 12.4×",
            ),
            5: cf.NodeEffect(math.log(1.3), False, False, "1.3×"),
        },
        floor=math.log(4.0),
        design="profile",
        caption="Differs across session",
    )
    shades, texts = cf.effect_shades(choice, 6)
    assert shades[4] == pytest.approx(math.log(12.4) / LOG_RATIO_EXTENT), "a size"
    assert shades[5] == 0.0
    assert texts[4].endswith("· kept")


def test_the_effect_caption_of_a_profile_names_what_it_differs_across(qapp):
    """A profile's axis is a size, so its caption says what the levels are of rather than a
    direction nobody chose; a two-level screen keeps its less-to-more words (§22)."""
    from detail.analysis import compare
    from detail.gui.node_workspace import _effect_caption

    profile = compare.Ranking(
        [], None, 0, 0, design="profile", level_values=("EXT", "REI", "SA")
    )
    assert (
        _effect_caption(profile, {"design_column": "SessionType"})
        == "Differs across SessionType"
    )
    between = compare.Ranking(
        [], None, 0, 0, design="between", level_values=("a", "b", "c")
    )
    assert _effect_caption(between, {"group_by": ["strain"]}) == "Differs across strain"
    paired = compare.Ranking([], None, 0, 0, design="paired", levels=("veh", "fent"))
    assert _effect_caption(paired, {}) == "Less ↔ more under fent (against veh)"


def test_a_node_effect_carries_the_pair_its_screen_found(qapp):
    """The workspace's row for a cluster keeps the item's pair, so the key can name it at the
    mark (§22); the sentence is the Comparisons page's, pair first."""
    import math

    from detail.analysis import compare
    from detail.gui.node_workspace import _node_effects

    item = compare.Effect(
        ("time_fraction", "node:12", "", "whole"),
        math.log(9.6),
        8,
        8,
        7,
        True,
        pair=("REI", "SA"),
    )
    ranking = compare.Ranking(
        [item],
        math.log(4.0),
        200,
        0,
        design="profile",
        unit=("ID",),
        level_values=("EXT", "REI", "SA"),
    )
    effects = _node_effects(
        ranking, {"design_column": "SessionType", "aggregate_to": ["ID"]}
    )
    assert effects[12].pair == ("REI", "SA") and effects[12].clears
    assert effects[12].text.startswith(
        "differs most between REI and SA: more under SA than REI, 9.6×"
    )


# --------------------------------------------------------------------------- #
# Strictness: a screen read looser or stricter than its floor (§22)
# --------------------------------------------------------------------------- #

LEVELS = (
    cf.ScreenLevel(0.01, 3.0, (), frozenset(), 0),
    cf.ScreenLevel(0.02, 2.6, (4,), frozenset({0}), 1),
    cf.ScreenLevel(0.05, 2.0, (4,), frozenset({0, 1}), 2),
    cf.ScreenLevel(0.1, 1.6, (4,), frozenset({0, 1}), 2),
    cf.ScreenLevel(0.2, 1.2, (4, 5), frozenset({0, 1, 2}), 3),
    cf.ScreenLevel(0.3, 1.0, (4, 5), frozenset({0, 1, 2}), 3),
    cf.ScreenLevel(0.5, 0.6, (4, 5, 3), frozenset({0, 1, 2, 3}), 4),
)


def _levelled_screen():
    import math

    return cf.ScreenChoice(
        "cmp_02",
        "drug vs vehicle · paired",
        "1 kept of 2 clearing",
        frozenset({0, 1}),
        effects={
            4: cf.NodeEffect(
                math.log(12.4),
                True,
                True,
                "more under drug, 12.4×",
                family_p=0.02,
                own_p=0.004,
                q=0.5,
            ),
            2: cf.NodeEffect(
                -math.log(3.0),
                True,
                False,
                "less under drug, 3.0×",
                family_p=0.03,
                own_p=0.004,
                q=0.5,
            ),
            5: cf.NodeEffect(
                math.log(2.5),
                False,
                False,
                "more under drug, 2.5×",
                family_p=0.2,
                own_p=0.03,
                q=0.6,
            ),
            3: cf.NodeEffect(
                math.log(1.5),
                False,
                False,
                "more under drug, 1.5×",
                family_p=0.5,
                own_p=0.4,
                q=0.9,
            ),
        },
        floor=2.0,
        design="paired",
        caption="Less ↔ more under drug",
        kept=(4,),
        column="drug",
        levels=LEVELS,
    )


def test_the_strictness_slider_reads_the_ticked_screen_at_another_level(bar):
    """Looser shows more: the slider's level picks that level's leaves and travels on the
    selection as ``alpha``; the line under it says the level and what is kept there; the
    Differences line's summary names it; Reset takes the screen and its level off together;
    and a re-read of the list keeps the level, applying that level's leaves as listed now.
    """
    screen = _levelled_screen()
    bar.set_screens([screen])
    lst = bar.screens
    assert lst.strictness_box.isHidden(), "no screen ticked: nothing to read at a level"
    _tick_screen(bar, "cmp_02")
    assert not lst.strictness_box.isHidden()
    assert lst.alpha() is None and bar.selection().alpha is None
    assert bar.selection().leaves == frozenset({0, 1})
    assert lst.strictness_label.text() == (
        "p ≤ 0.05 (the screen's own floor) · 1 kept of 2 clearing"
    )

    lst.strictness.setValue(cf.STRICTNESS_LEVELS.index(0.2))
    selection = bar.seen[-1]
    assert selection.alpha == 0.2 and selection.leaves_source == "cmp_02"
    assert selection.leaves == frozenset(
        {0, 1, 2}
    ), "the level's leaves, not the floor's"
    assert lst.strictness_label.text() == (
        "p ≤ 0.2 (looser than its floor) · 2 kept of 3 clearing"
    )
    assert bar.differences_summary() == ("drug vs vehicle · paired · p ≤ 0.2")

    lst.strictness.setValue(cf.STRICTNESS_LEVELS.index(0.01))
    assert bar.selection().alpha == 0.01 and bar.selection().leaves == frozenset()
    assert "stricter than its floor" in lst.strictness_label.text()

    lst.strictness.setValue(cf.STRICTNESS_LEVELS.index(0.05))
    assert bar.selection().alpha is None, "the floor itself is no level"
    assert bar.selection().leaves == frozenset({0, 1})

    lst.strictness.setValue(cf.STRICTNESS_LEVELS.index(0.2))
    bar.set_screens([], loading=True)
    assert bar.selection().alpha == 0.2, "a re-read in flight keeps the level"
    recut = cf.ScreenChoice(
        "cmp_02",
        "drug vs vehicle · paired",
        "1 kept of 2 clearing",
        frozenset({0, 1}),
        effects=screen.effects,
        floor=2.0,
        design="paired",
        kept=(4,),
        levels=tuple(
            cf.ScreenLevel(lv.alpha, lv.floor, lv.kept, lv.leaves | {9}, lv.clearing)
            for lv in LEVELS
        ),
    )
    bar.set_screens([recut])
    assert bar.selection().alpha == 0.2
    assert bar.selection().leaves == frozenset({0, 1, 2, 9}), "that level's new leaves"
    assert lst.strictness.value() == cf.STRICTNESS_LEVELS.index(0.2)

    bar.remove(cf.CLUSTERS)
    assert bar.selection().is_empty and bar.selection().alpha is None
    assert lst.strictness_box.isHidden()


def test_a_screen_with_no_floor_has_no_levels_to_read_at(bar):
    """No floor, no slider: the screen is ticked at what it is."""
    screen = cf.ScreenChoice(
        "cmp_05", "sex · between", "no floor: one unit", frozenset({2}), floor=None
    )
    bar.set_screens([screen])
    _tick_screen(bar, "cmp_05")
    assert bar.screens.strictness_box.isHidden()
    assert bar.selection().leaves == frozenset({2}) and bar.selection().alpha is None
    bar.set_selection(
        Selection(leaves=frozenset({2}), leaves_source="cmp_05", alpha=0.2)
    )
    assert bar.screens.strictness_box.isHidden()
    bar.set_screens([screen])
    assert (
        bar.selection().alpha is None
    ), "a level the screen was not read at is dropped on the next read"
    assert bar.selection().leaves == frozenset({2})


def test_a_selection_at_a_level_shows_the_slider_there(bar):
    bar.set_screens([_levelled_screen()])
    bar.set_selection(
        Selection(leaves=frozenset({0, 1, 2}), leaves_source="cmp_02", alpha=0.2)
    )
    assert bar.screens.value() == "cmp_02"
    assert bar.screens.strictness.value() == cf.STRICTNESS_LEVELS.index(0.2)
    assert bar.screens.alpha() == 0.2


def test_effect_shades_at_a_level_take_that_levels_floor_and_kept_set_and_say_the_p():
    """At p ≤ 0.2 the cluster the floor hid is coloured and *kept at p ≤ 0.2*; at the
    screen's own floor it is grey and *inside the floor*; every hover carries the cluster's
    family-wise p, its own p and its q, in one wording (``numbers_text``)."""
    import math

    from detail.gui.match_key import LOG_RATIO_EXTENT

    screen = _levelled_screen()
    shades, texts = cf.effect_shades(screen, 6)
    assert shades[5] == 0.0
    assert texts[5] == (
        "more under drug, 2.5× · inside the floor · family-wise p 0.2, own p 0.03, q 0.6"
    )
    assert texts[4] == (
        "more under drug, 12.4× · kept · family-wise p 0.02, own p 0.004, q 0.5"
    )
    shades, texts = cf.effect_shades(screen, 6, alpha=0.2)
    assert shades[5] == pytest.approx(math.log(2.5) / LOG_RATIO_EXTENT)
    assert texts[5] == (
        "more under drug, 2.5× · kept at p ≤ 0.2 · family-wise p 0.2, own p 0.03, q 0.6"
    )
    assert texts[4].startswith("more under drug, 12.4× · kept at p ≤ 0.2 ·")
    assert texts[2] == (
        "less under drug, 3.0× · clears the floor at p ≤ 0.2, nested under a kept cluster"
        " · family-wise p 0.03, own p 0.004, q 0.5"
    )
    assert shades[3] == 0.0
    assert texts[3].endswith(
        "inside the floor at p ≤ 0.2 · family-wise p 0.5, own p 0.4, q 0.9"
    )
    shades, _texts = cf.effect_shades(screen, 6, alpha=0.5)
    assert shades[3] == pytest.approx(math.log(1.5) / LOG_RATIO_EXTENT)
    # The level decides by the row's p, as the API does (`Ranking.clears_at`), so a slider's
    # threshold and a hover's p cannot disagree.
    assert screen.clears(5, 0.2) and not screen.clears(5, 0.1)
    assert not screen.clears(5, None)
    assert screen.kept_texts(0.2) == {
        4: "more under drug, 12.4×",
        5: "more under drug, 2.5×",
    }
    assert screen.kept_numbers(0.2)[5] == {"family_p": 0.2, "own_p": 0.03, "q": 0.6}
    assert screen.kept_texts() == {4: "more under drug, 12.4×"}
    assert cf.numbers_text(None, None, None) == ""


def test_effect_shades_on_a_studentized_screen_sit_by_the_t():
    """On a screen scored by the moderated t the blocks and the key share the t axis: a
    block's position is its t over the axis's reach, the floor is in t, a row with no p
    clears a level by its |t| against that level's floor, and the hover carries the t beside
    the effect's words (§22, §24 *The statistic*). The same rows on a screen scored by the
    effect sit by their effect, with no t in the words."""
    import dataclasses
    import math

    from detail.gui.match_key import LOG_RATIO_EXTENT, T_EXTENT

    choice = cf.ScreenChoice(
        "cmp_04",
        "drug vs vehicle · paired",
        "1 kept of 2",
        frozenset({0, 1}),
        effects={
            4: cf.NodeEffect(
                math.log(12.4),
                True,
                True,
                "more under drug, 12.4×",
                statistic=8.84,
                family_p=0.01,
            ),
            5: cf.NodeEffect(
                math.log(1.3), False, False, "more under drug, 1.3×", statistic=0.6
            ),
            2: cf.NodeEffect(
                -math.log(3.0), True, False, "less under drug, 3.0×", statistic=-4.5
            ),
        },
        floor=4.1,
        design="paired",
        caption="Less ↔ more under drug",
        statistic="moderated_t",
        levels=(cf.ScreenLevel(0.2, 3.0, (4, 5), frozenset({0, 1, 2}), 3),),
    )
    assert choice.studentized
    shades, texts = cf.effect_shades(choice, 6)
    assert shades[4] == pytest.approx(8.84 / T_EXTENT)
    assert shades[2] == pytest.approx(-4.5 / T_EXTENT)
    assert shades[5] == 0.0
    assert texts[4] == "more under drug, 12.4× · kept · t +8.8 · family-wise p 0.01"
    assert texts[5] == "more under drug, 1.3× · inside the floor · t +0.6"
    assert choice.clears(2, 0.2) and not choice.clears(5, 0.2), "|t| against the floor"

    plain = dataclasses.replace(choice, statistic="effect")
    assert not plain.studentized
    shades, texts = cf.effect_shades(plain, 6)
    assert shades[4] == pytest.approx(math.log(12.4) / LOG_RATIO_EXTENT)
    assert "t +8.8" not in texts[4]
