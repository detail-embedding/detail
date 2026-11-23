"""AUDIT's review pane, as a widget: click to remove, Shift for a run, the menu, Finish and the
reviewed box.

The page's reads and writes are `tests/gui/test_label_check.py`'s; here the pane alone, fed
tiles by hand, so every disabled control is checked for saying why.
"""

import pytest

from detail.gui.label_clean import (
    CLICK_HINT,
    NOTHING_TO_UNDO,
    REVIEWED_READ_ONLY,
    SCROLL_FIRST,
    ReviewPane,
    crossed_for,
    day_of,
    note_for,
    tag_for,
)
from detail.gui.live_montage import MontageClip

SECOND = 1_000_000_000


def _tiles(n, faded=()):
    return [
        MontageClip(
            SECOND,
            i + 1,
            f"rec {i}",
            faded=i in faded,
            tag="not climb" if i in faded else None,
        )
        for i in range(n)
    ]


def _pane(qapp, n=6, faded=(2,)):
    from tests.gui.conftest import export_source

    view = ReviewPane(export=export_source())
    view.resize(900, 600)
    view.show()
    view.show_cluster(1321, "climb")
    view.set_targets(["rear", "climb/left"])
    view.montage.set_clips(_tiles(n, faded=set(faded)))
    view.set_editable("")
    view.start_review()
    qapp.processEvents()
    return view


@pytest.fixture
def pane(qapp):
    view = _pane(qapp)
    yield view
    view.montage._worker.stop()
    view.close()
    view.deleteLater()


def test_the_top_row_names_the_cluster_its_label_and_how_far_the_scroll_reached(pane):
    assert pane.title.text() == "cluster #1321"
    assert pane.label_name.text() == "climb"
    assert pane.reached.text() == f"reached {pane.montage.seen()} of 6"
    assert pane.progress.maximum() == 6 and pane.progress.value() == pane.montage.seen()
    assert pane.finish.text() == "✓ Finish review" and pane.finish.isDefault()
    assert pane.finish_counts.text() == "5 climb · 1 not climb"
    assert not pane.result.isVisible() and not pane.again.isVisible()
    assert pane.montage.order_button.isHidden()


def _clicked(pane, index, shift=False):
    """Through the montage's own click path, so a Shift-click runs from its real anchor."""
    from detail.gui.qt import Qt

    modifier = Qt.KeyboardModifier.ShiftModifier if shift else None
    pane.montage._clicked(index, modifier)


def _apply(pane):
    """Stand in for the page: fade what the pane asks to remove, lift what it puts back."""
    seen = []

    def remove(indices, to):
        seen.append(("out", list(indices), to))
        _mark(pane, indices, True)

    def back(indices):
        seen.append(("back", list(indices)))
        _mark(pane, indices, False)

    pane.take_out_requested.connect(remove)
    pane.put_back_requested.connect(back)
    return seen


def _mark(pane, indices, faded):
    from dataclasses import replace

    tiles = pane.montage.clips()
    for i in indices:
        tiles[i] = replace(tiles[i], faded=faded, tag="not climb" if faded else None)
    pane.montage.set_clips(tiles, keep_scroll=True)
    pane.refresh()


def test_before_any_change_the_line_says_how_to_start(pane):
    assert pane.undo_text.text() == CLICK_HINT
    assert pane.undo_text.text() == (
        "Click a clip to remove it. Drag, or hold Shift, to remove a run."
    )
    assert not pane.undo.isEnabled() and pane.undo.toolTip() == NOTHING_TO_UNDO
    assert not hasattr(pane, "bar") and not hasattr(pane, "mark_no"), "no selection bar"


def test_a_click_removes_and_a_second_click_puts_back(pane):
    seen = _apply(pane)
    _clicked(pane, 1)
    _clicked(pane, 2)  # already removed: put back
    _clicked(pane, 1)
    assert seen == [("out", [1], None), ("back", [2]), ("back", [1])]


def test_shift_click_removes_the_run_skipping_removed_ones(pane):
    seen = _apply(pane)
    _clicked(pane, 0)
    _clicked(pane, 4, shift=True)  # 2 was removed already
    assert seen == [("out", [0], None), ("out", [1, 3, 4], None)]
    _clicked(pane, 1, shift=True)  # from 4 back to 1: all removed now
    assert len(seen) == 2, "nothing left to remove in that run"


def test_the_finish_count_follows_every_click(pane):
    _apply(pane)
    assert pane.finish_counts.text() == "5 climb · 1 not climb"
    _clicked(pane, 0)
    assert pane.finish_counts.text() == "4 climb · 2 not climb"
    _clicked(pane, 2)
    assert pane.finish_counts.text() == "5 climb · 1 not climb"


def test_the_hover_hint_says_what_a_click_will_do(pane):
    assert pane.montage.is_clickable()
    assert pane.montage.click_hint(0) == "Click: remove from climb · double-click: open"
    assert pane.montage.click_hint(2) == "Click: put back in climb · double-click: open"


def test_right_click_moves_or_puts_back(pane):
    seen = _apply(pane)
    menu = pane.clip_menu(0)
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts == ["Move to rear", "Move to climb › left"]
    menu.actions()[0].trigger()
    assert seen == [("out", [0], "rear")]
    removed = pane.clip_menu(2)
    actions = [a for a in removed.actions() if not a.isSeparator()]
    assert actions[-1].text() == "Put back in climb"
    actions[-1].trigger()
    assert seen[-1] == ("back", [2])


def _press(pane, index, *, double=False):
    """A real press and release on tile ``index`` — ``double``: the second press of a
    double-click, as a pointer sends it (`QTest.mouseDClick` then the release)."""
    from detail.gui.qt import QTest, Qt

    canvas = pane.montage.canvas
    at = pane.montage.tile_rect(index).center()
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    if double:
        QTest.mouseDClick(canvas, left, none, at)
    else:
        QTest.mousePress(canvas, left, none, at)
    QTest.mouseRelease(canvas, left, none, at)


def test_a_double_click_retracts_its_first_press(pane, qapp):
    """Qt delivers a double-click as a press (a click: remove) then a double-click (open): the
    pane takes the click back, so opening a clip does not remove it."""
    _apply(pane)
    undone = []
    pane.undo_requested.connect(lambda: undone.append(True))
    _press(pane, 0)
    pane.set_undo("Removed 1 from climb")
    _press(pane, 0, double=True)
    assert undone == [True]
    _press(pane, 0, double=True)
    assert undone == [True], "only the click right before it"
    _press(pane, 3)
    _press(pane, 4, double=True)
    assert undone == [True], "a double-click elsewhere retracts nothing"


def test_space_after_a_click_opens_and_retracts_nothing(pane, qapp):
    """Space opens the clip as a double-click does, and arrived inside the double-click
    interval: the pane took the click before it back."""
    from detail.gui.qt import QTest, Qt

    _apply(pane)
    undone, opened = [], []
    pane.undo_requested.connect(lambda: undone.append(True))
    pane.montage.clip_activated.connect(opened.append)
    _press(pane, 1)
    pane.set_undo("Removed 1 from climb")
    QTest.keyClick(pane.montage.canvas, Qt.Key.Key_Space)
    assert opened == [1]
    assert undone == [], "a key is not the second half of a click"
    assert pane.montage.clips()[1].faded


def test_a_click_while_the_review_is_saved_changes_nothing(pane, qapp):
    """Finish greyed itself while saving, but the grid still took clicks: the box then said
    "0 removed" over a grid with one removed."""
    from detail.gui.label_clean import SAVING

    seen = _apply(pane)
    pane.show_saving()
    assert not pane.can_mark() and not pane.montage.is_clickable()
    _press(pane, 0)
    assert seen == [] and not pane.montage.clips()[0].faded
    assert pane.undo_text.text() == SAVING
    assert pane.finish.toolTip() == SAVING
    assert pane.clip_menu(0).actions()[0].isEnabled() is False


def test_enter_does_not_finish_while_picking_clips(qapp):
    """Pick mode has the grid's clicks; Enter there finished the review behind it."""
    from detail.gui.qt import QTest, Qt

    from detail.gui.label_clean import PICKING_FINISH

    view = _pane(qapp, n=4, faded=())
    try:
        assert view.reached_end()
        fired = []
        view.finish_requested.connect(lambda: fired.append(True))
        view.montage.exports.start_picking()
        qapp.processEvents()
        view.montage.canvas.setFocus()
        QTest.keyClick(view.montage.canvas, Qt.Key.Key_Return)
        assert fired == [], "Enter finished the review from pick mode"
        assert not view.finish.isEnabled() and view.finish.toolTip() == PICKING_FINISH
        QTest.keyClick(view.montage.canvas, Qt.Key.Key_Escape)
        qapp.processEvents()
        assert view.montage.click_layer() is None
        QTest.keyClick(view.montage.canvas, Qt.Key.Key_Return)
        assert fired == [True], "and does again once picking is over"
    finally:
        view.montage._worker.stop()
        view.deleteLater()


def test_the_undo_line_says_what_the_last_change_did(pane):
    asked = []
    pane.undo_requested.connect(lambda: asked.append(True))
    pane.set_undo("Removed 3 from climb")
    assert pane.undo_text.text() == "Removed 3 from climb" and pane.undo.isEnabled()
    assert pane.undo.text() == "Undo (⌘Z)"
    pane.undo.click()
    keys = {k.key().toString(): k for k in pane._keys}
    keys["Ctrl+Z"].activated.emit()
    assert asked == [True, True]
    pane.set_undo(None)
    assert pane.undo_text.text() == CLICK_HINT
    assert not pane.undo.isEnabled()
    assert set(keys) == {
        "Ctrl+Z",
        "Return",
        "Enter",
    }, "no ⌘A, Esc, B or C: no selection"


def test_finish_waits_for_the_last_row_and_says_so(qapp):
    view = _pane(qapp, n=120, faded=())
    try:
        view.montage.set_tile_px(160)
        qapp.processEvents()
        assert view.montage.seen() < 120
        assert not view.finish.isEnabled()
        assert view.finish.toolTip() == SCROLL_FIRST
        assert view.finish_counts.text() == "120 climb · 0 not climb"
        keys = {k.key().toString(): k for k in view._keys}
        fired = []
        view.finish_requested.connect(lambda: fired.append(True))
        keys["Return"].activated.emit()
        assert fired == [], "Enter does nothing while Finish is greyed"
        assert view.reached.text() == f"reached {view.montage.seen()} of 120"
        bar = view.montage.grid.verticalScrollBar()
        bar.setValue(bar.maximum())
        qapp.processEvents()
        assert view.finish.isEnabled()
        assert view.finish.isDefault()
        assert view.reached.text() == "reached 120 of 120"
        assert view.progress.value() == view.progress.maximum()
        keys["Return"].activated.emit()
        assert fired == [True], "Enter presses Finish once it is open"
        view.finish.click()
        assert fired == [True, True]
        assert view.finish.autoDefault()
        assert view.finish.font().bold()
        assert view.finish.font().pointSizeF() > view.font().pointSizeF()
    finally:
        view.montage._worker.stop()
        view.deleteLater()


def test_the_reviewed_box_replaces_the_top_row_and_stops_clicks(pane, qapp):
    seen = _apply(pane)
    pane.show_result(
        "83%", "5 of 6 are climb · 1 removed", 871, removed=1, when="8 Nov"
    )
    qapp.processEvents()
    assert pane.box.isVisible() and not pane.top.isVisible()
    assert pane.box.frameShape() != pane.box.Shape.NoFrame
    assert pane.reviewed.text() == "✓ Reviewed"
    assert (pane.what.text(), pane.box_label.text(), pane.when.text()) == (
        "cluster #1321",
        "climb",
        "· 8 Nov",
    )
    assert pane.precision.text() == "83%"
    assert pane.precision.font().pointSizeF() > pane.result.font().pointSizeF()
    assert pane.result.text() == "5 of 6 are climb · 1 removed"
    assert pane.next.isVisible() and pane.next.text() == "Next cluster: #871"
    assert pane.again.isVisible() and pane.again.isEnabled()
    assert pane.again.text() == "Reopen review"
    assert pane.again.toolTip() == (
        "Change this cluster's removals, then finish again to update 83%. "
        "Your 1 removal is kept."
    )
    assert not pane.finish.isVisible() and not pane.undo_row.isVisible()
    assert not pane.montage.is_clickable(), "read-only: no click, no hover cue"
    _clicked(pane, 0)
    assert seen == []
    menu = pane.clip_menu(2)
    assert all(
        not a.isEnabled() and a.toolTip() == REVIEWED_READ_ONLY
        for a in menu.actions()
        if not a.isSeparator()
    )
    pane.show_result("100%", "6 of 6 are climb · 0 removed", None, removed=0)
    assert pane.again.toolTip().endswith("update 100%. Nothing was removed.")
    assert not pane.next.isVisible(), "no unreviewed cluster left"
    pane.start_review()
    assert pane.montage.is_clickable() and pane.finish.isVisible()
    assert not pane.box.isVisible()


def test_a_snapshot_disables_every_edit_with_the_reason(pane, qapp):
    seen = _apply(pane)
    pane.set_editable("a snapshot")
    bar = pane.montage.grid.verticalScrollBar()
    bar.setValue(bar.maximum())
    qapp.processEvents()
    pane.set_undo("Removed 1 from climb")
    for button in (pane.finish, pane.undo):
        assert not button.isEnabled() and button.toolTip() == "a snapshot"
    assert not pane.montage.is_clickable()
    _clicked(pane, 0)
    assert seen == []
    assert all(
        not a.isEnabled() and a.toolTip() == "a snapshot"
        for a in pane.clip_menu(0).actions()
    )
    pane.show_result("90%", "x", 3, removed=0)
    assert not pane.again.isEnabled() and pane.again.toolTip() == "a snapshot"


def test_tags_and_notes_say_what_happened():
    out = {"from_label": "climb", "to_label": None, "at": "2026-11-08T10:00:00+00:00"}
    moved = {**out, "to_label": "climb/left"}
    assert tag_for(out) == "not climb"
    assert tag_for(moved) == "→ climb › left"
    assert note_for(out) == "Removed from climb in review, 8 Nov"
    assert note_for(moved).startswith("Moved from climb to climb › left")
    assert crossed_for(out) and not crossed_for(moved) and not crossed_for(None)
    assert day_of("2026-11-08T10:00:00+00:00") == "8 Nov" and day_of(None) == ""


# --- a run, previewed (review round 14) --------------------------------------------------------- #


def _drag(pane, qapp, start, end, *, release=True):
    from tests.gui.test_live_montage import _drag as drag

    drag(pane.montage, qapp, start, end, release=release)


def _release(pane, qapp, index):
    from detail.gui.qt import Qt

    from tests.gui.test_live_montage import _mouse

    _mouse(
        pane.montage,
        qapp,
        "release",
        pane.montage.tile_rect(index).center(),
        Qt.MouseButton.NoButton,
    )


def test_a_drag_removes_the_run_as_one_step_and_says_so_first(pane, qapp):
    from detail.gui.qt import Qt

    seen = _apply(pane)
    _drag(pane, qapp, 0, 4, release=False)
    assert pane.undo_text.text() == "Let go to remove 4 clips · Esc cancels"
    assert pane.undo_text.font().bold(), "the preview line is bold"
    from detail.gui.qt import QPalette

    from detail.gui.status_color import MUTED

    role = QPalette.ColorRole.WindowText
    assert pane.undo_text.palette().color(role) != MUTED, "in ink, not the hint's grey"
    _release(pane, qapp, 4)
    assert seen == [
        ("out", [0, 1, 3, 4], None)
    ], "one request: one undo step; 2 skipped"
    assert pane.undo_text.text() == CLICK_HINT and not pane.undo_text.font().bold()
    assert pane.undo_text.palette().color(role) == MUTED
    _drag(pane, qapp, 3, 5, release=False)
    assert pane.undo_text.text() == "Let go to put back 2 clips"
    from tests.gui.test_live_montage import _key

    _key(pane.montage, qapp, Qt.Key.Key_Escape)
    assert pane.undo_text.text() == CLICK_HINT
    _release(pane, qapp, 5)
    assert len(seen) == 1, "Esc cancelled it"
    _drag(pane, qapp, 1, 3)
    assert seen[-1] == (
        "back",
        [1, 2, 3],
    ), "a drag from a removed clip puts the run back"


def test_holding_shift_says_what_a_shift_click_will_remove(pane, qapp):
    from detail.gui.qt import Qt

    from tests.gui.test_live_montage import _key

    seen = _apply(pane)
    _clicked(pane, 0)
    pane.montage._set_hovered(4)
    _key(pane.montage, qapp, Qt.Key.Key_Shift)
    assert pane.montage.preview().indices == (1, 3, 4)
    assert pane.undo_text.text() == "Shift-click removes 3 clips"
    _clicked(pane, 4, shift=True)
    assert seen[-1] == ("out", [1, 3, 4], None)
    assert "Shift-click removes nothing" in pane.undo_text.text(), "all removed now"


def test_the_preview_line_counts_one_clip_and_an_empty_run():
    from detail.gui.label_clean import preview_text
    from detail.gui.live_montage import RunPreview

    assert preview_text(RunPreview("drag", False, (3,))) == (
        "Let go to remove 1 clip · Esc cancels"
    )
    assert preview_text(RunPreview("drag", True, ())).startswith("Nothing removed")
    assert preview_text(RunPreview("drag", False, ())).startswith("Every clip")


def test_a_reviewed_cluster_offers_no_run(pane, qapp):
    from detail.gui.qt import Qt

    from tests.gui.test_live_montage import _key

    seen = _apply(pane)
    _clicked(pane, 0)
    pane.show_result("83%", "5 of 6 are climb · 1 removed", None, removed=1)
    _drag(pane, qapp, 1, 4, release=False)
    assert pane.montage.preview() is None and not pane.montage.is_dragging()
    pane.montage._set_hovered(4)
    _key(pane.montage, qapp, Qt.Key.Key_Shift)
    assert pane.montage.preview() is None
    _release(pane, qapp, 4)
    assert seen == [("out", [0], None)], "read-only: nothing changed"


def test_finish_is_the_large_default_button_with_its_counts_under_it(pane, qapp):
    finish, counts = pane.finish, pane.finish_counts
    assert finish.text() == "✓ Finish review"
    assert finish.isDefault() and finish.autoDefault()
    assert finish.font().bold()
    assert counts.text() == "5 climb · 1 not climb"
    assert counts.isVisible()
    gap = counts.geometry().top() - finish.geometry().bottom()
    assert 0 <= gap <= 6, "the counts sit directly under the button"
    assert abs(counts.geometry().center().x() - finish.geometry().center().x()) <= 2


def test_finish_fills_with_the_accent_only_once_it_can_be_pressed(qapp):
    from detail.gui.qt import QApplication, QPalette

    view = _pane(qapp, n=120, faded=())
    try:
        view.montage.set_tile_px(160)
        qapp.processEvents()
        accent = QApplication.palette().color(QPalette.ColorRole.Accent)
        button = view.finish.palette().color(QPalette.ColorRole.Button)
        assert not view.finish.isEnabled() and button != accent
        bar = view.montage.grid.verticalScrollBar()
        bar.setValue(bar.maximum())
        qapp.processEvents()
        assert view.finish.isEnabled()
        from detail.gui.label_clean import FINISH_ACCENT_DARKER

        fill = accent.darker(FINISH_ACCENT_DARKER)
        assert view.finish.palette().color(QPalette.ColorRole.Button) == fill
    finally:
        view.montage._worker.stop()
        view.deleteLater()
