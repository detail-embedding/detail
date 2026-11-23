"""The labeling screen (§26): a player with the example set beside it.

Driven over the fixture experiment with a real 6-frame video and a real tracking run, so the
decode, the tracked boxes and every save go through the paths a user's clicks take.

**A class guarded here: an index held across an asynchronous change to its list (#26).** A mark
save and a frame removal both address a frame by its position in the set, each off the UI
thread. Pressing N while a removal was in flight (the keys reach the dialog; the busy overlay
covers only the view) saved the mark to a position the removal had just shifted, onto the next
frame. The dialog serialises the two: Remove waits for the saver, and no mark is saved while a
removal runs. Its two orders are the last two tests in this file."""

import pytest

from detail.core.skeleton import Skeleton
from detail.gui import keypoint_labeling as mod
from detail.gui.qt import QMessageBox, QTest, Qt
from tests.gui import subjects
from tests.gui.conftest import drain_busy_calls, drain_until

SK = Skeleton(names=("nose", "tail_base"), edges=(("nose", "tail_base"),))
TRACKED = (0, 2, 4)  # the fixture's detections (tests/gui/subjects.py `_FRAMES`)


@pytest.fixture
def proj(tmp_path):
    return subjects.an_experiment_with_completed_runs(tmp_path)


def _open(qapp, proj, set_id):
    dialog = mod.KeypointLabelingDialog(proj, set_id)
    dialog.show()
    qapp.processEvents()
    drain_busy_calls(qapp)  # the offered videos
    drain_busy_calls(qapp)  # the chosen video's preview and boxes
    assert drain_until(qapp, lambda: dialog._raw is not None), "no frame rendered"
    return dialog


def _seek(qapp, dialog, frame: int) -> None:
    dialog.scrubber().set_frame(frame)
    assert drain_until(
        qapp, lambda: dialog._raw is not None and dialog._raw[2] == frame
    ), f"frame {frame} never rendered"


def _settled(qapp, dialog) -> None:
    drain_busy_calls(qapp)
    assert drain_until(qapp, lambda: not dialog.saving()), "a save never settled"


# --- the player ------------------------------------------------------------------ #
def test_offers_the_tracked_video_and_shows_its_first_frame(qapp, proj):
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)

    assert dialog._recording.count() == 1
    assert dialog._recording.currentData() == ("recording_001", "track_01")
    assert [dialog._track.itemData(i) for i in range(dialog._track.count())] == [0]
    assert dialog._raw[0] == "recording_001" and dialog._raw[1] == "part_001"
    assert dialog.image_view().pixmap_size() is not None
    assert dialog._annotating.text() == "Annotating: nose"
    assert "Not ready yet" in dialog._ready.text()
    assert dialog.windowTitle().startswith("Mark example frames")


def test_a_click_adds_the_frame_places_the_point_and_saves(qapp, proj):
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 2)
    assert "a click adds it" in dialog._status.text()

    dialog.image_view().clicked_at.emit(0.25, 0.5)
    _settled(qapp, dialog)

    stored = proj.example_set(created.id)
    assert [f.frame_idx for f in stored.frames] == [2]
    assert stored.frames[0].sampled_by is None
    mark = stored.frames[0].mark_for("nose")
    assert (mark.x, mark.y) == (0.25, 0.5) and mark.visible
    assert dialog._annotating.text() == "Annotating: tail_base", "did not advance"
    assert "example frame 1 of 1" in dialog._status.text()
    assert dialog.scrubber().marks()[0] == (2,), "the frame is not a tick on the slider"


def test_the_skeleton_can_change_until_the_first_mark(qapp, proj):
    from detail.core.skeleton import PRESETS

    created = proj.create_example_set("rats", "quadruped_side_5")
    dialog = _open(qapp, proj, created.id)
    combo = dialog._skeleton
    assert combo.currentData() == "quadruped_side_5" and combo.isEnabled()
    combo.setCurrentIndex(combo.findData("midline_3"))
    combo.activated.emit(combo.currentIndex())
    qapp.processEvents()
    names = PRESETS["midline_3"][1].names
    assert proj.example_set(created.id).skeleton.names == names, "written"
    assert dialog._keypoints.count() == len(names)
    assert dialog._annotating.text() == f"Annotating: {names[0]}"

    _seek(qapp, dialog, 2)
    dialog.image_view().clicked_at.emit(0.25, 0.5)
    _settled(qapp, dialog)
    assert not combo.isEnabled() and "make a new example set" in combo.toolTip()


def test_a_set_on_no_library_skeleton_says_so(qapp, proj):
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    assert dialog._skeleton.currentData() == ""
    assert dialog._skeleton.currentText().startswith("this set's own skeleton")


def test_every_new_frame_starts_at_the_first_point(qapp, proj):
    """The order a user clicks in is the skeleton's, on every frame: finishing one frame on
    its last point once carried that point over, so the next frame began at tail_base.
    """
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 2)
    dialog.image_view().clicked_at.emit(0.25, 0.5)
    _settled(qapp, dialog)
    dialog.image_view().clicked_at.emit(0.75, 0.5)
    _settled(qapp, dialog)
    assert (
        dialog._annotating.text() == "Annotating: tail_base"
    ), "stays on the last point"

    _seek(qapp, dialog, 4)
    assert dialog._annotating.text() == "Annotating: nose"
    dialog.image_view().clicked_at.emit(0.3, 0.5)
    _settled(qapp, dialog)

    frame = proj.example_set(created.id).frames[1]
    assert frame.frame_idx == 4 and [m.target for m in frame.marks] == ["nose"]
    assert dialog._annotating.text() == "Annotating: tail_base"


def test_not_visible_and_clear_reach_the_disk(qapp, proj):
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 2)
    dialog.image_view().clicked_at.emit(0.25, 0.5)
    _settled(qapp, dialog)

    dialog._hidden_button.click()  # tail_base, the current point
    _settled(qapp, dialog)
    stored = proj.example_set(created.id)
    assert stored.frames[0].mark_for("tail_base").visible is False
    assert stored.labeled(0)
    # every point is decided on this frame, but tail_base is visible nowhere, so a run cannot
    # learn it yet — the readiness line says which point still needs a visible mark
    assert (
        "Not ready yet" in dialog._ready.text() and "tail_base" in dialog._ready.text()
    )

    dialog._keypoints.setCurrentRow(0)  # nose
    dialog._clear_button.click()
    _settled(qapp, dialog)
    assert proj.example_set(created.id).frames[0].mark_for("nose") is None
    assert not proj.example_set(created.id).labeled(0)


def test_a_click_where_the_track_has_no_subject_places_nothing(qapp, proj):
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 1)
    assert "no tracked subject" in dialog._status.text()

    dialog.image_view().clicked_at.emit(0.25, 0.5)
    _settled(qapp, dialog)

    assert proj.example_set(created.id).frames == []
    assert "no subject on this frame" in dialog._status.text()


def test_example_frames_are_ticks_and_d_a_step_through_them(qapp, proj):
    created = proj.create_example_set("rats", SK)
    for frame in (0, 4):
        proj.add_example_frame(
            created.id, "recording_001", "track_01", "part_001", frame, 0
        )
    dialog = _open(qapp, proj, created.id)
    assert dialog.scrubber().marks()[0] == (0, 4)
    _seek(qapp, dialog, 2)

    dialog._step_example(+1)
    assert drain_until(qapp, lambda: dialog._raw[2] == 0)
    assert dialog._frames.currentRow() == 0
    dialog._step_example(+1)
    assert drain_until(qapp, lambda: dialog._raw[2] == 4)
    assert dialog._frames.currentRow() == 1
    assert dialog._frames.item(1).text().startswith("2. recording_001, frame 4")
    assert "(picked)" not in dialog._frames.item(1).text()


def test_add_frames_samples_with_the_chosen_strategy(qapp, proj):
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    dialog._n_frames.setValue(2)
    dialog._strategy.setCurrentIndex(dialog._strategy.findData("spread"))

    dialog._add_frames.click()
    drain_busy_calls(qapp)

    stored = proj.example_set(created.id)
    assert len(stored.frames) == 2
    assert {f.sampled_by for f in stored.frames} == {"spread"}
    assert {f.frame_idx for f in stored.frames} <= set(TRACKED)
    assert dialog._frames.count() == 2
    assert "(spread)" in dialog._frames.item(0).text()
    assert dialog._add_frames.isEnabled()


def test_removing_a_marked_frame_asks_first(qapp, proj, monkeypatch):
    created = proj.create_example_set("rats", SK)
    proj.add_example_frame(created.id, "recording_001", "track_01", "part_001", 2, 0)
    proj.set_example_marks(created.id, 0, [{"target": "nose", "x": 0.1, "y": 0.1}])
    dialog = _open(qapp, proj, created.id)
    asked = []

    def refuse(*args, **kwargs):
        asked.append(args)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", refuse)
    dialog._frames.setCurrentRow(0)
    dialog._remove_frame.click()
    drain_busy_calls(qapp)
    assert len(asked) == 1
    assert len(proj.example_set(created.id).frames) == 1

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    dialog._remove_frame.click()
    drain_busy_calls(qapp)
    assert proj.example_set(created.id).frames == []
    assert not dialog._remove_frame.isEnabled() and dialog._remove_frame.toolTip()


def test_the_tool_row_offers_point_alone_and_ok_reads_use_these_frames(qapp, proj):
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)

    point, region, box = (dialog._tool_buttons[k] for k in ("point", "region", "box"))
    assert point.isEnabled() and point.isChecked()
    for reserved in (region, box):
        assert not reserved.isEnabled()
        assert "reserved" in reserved.toolTip()
    from detail.gui.qt import QDialogButtonBox

    ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
    assert ok.text() == "Use these frames"


def test_keys_choose_a_point_and_mark_it(qapp, proj):
    from detail.gui.qt import QTest

    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 4)

    QTest.keyClick(dialog, Qt.Key.Key_2)
    assert dialog._current == "tail_base"
    QTest.keyClick(dialog, Qt.Key.Key_N)
    _settled(qapp, dialog)
    stored = proj.example_set(created.id)
    assert stored.frames[0].frame_idx == 4
    assert stored.frames[0].mark_for("tail_base").visible is False


# --- dragging a placed point ---------------------------------------------------------- #
def test_a_placed_point_can_be_dragged_to_move_it(qapp, proj):
    """The ordered clicks stay as they are; a placed point is also grabbable, the scene
    editor's drag, and the move saves without advancing to the next point."""
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 2)
    dialog.image_view().clicked_at.emit(0.25, 0.5)
    _settled(qapp, dialog)
    assert dialog._annotating.text() == "Annotating: tail_base"

    view = dialog.image_view()
    on_nose = view.widget_position(0.25, 0.5)
    assert view.hit_test(on_nose.toPoint()), "the placed nose is grabbable"
    assert not view.hit_test(view.widget_position(0.75, 0.9).toPoint())

    dialog._drag_started(0.25, 0.5)
    assert dialog._dragging == "nose"
    dialog._dragged(0.4, 0.6)
    dialog._drag_finished(0.4, 0.6)
    _settled(qapp, dialog)

    mark = proj.example_set(created.id).frames[0].mark_for("nose")
    assert (mark.x, mark.y) == (0.4, 0.6) and mark.visible
    assert dialog._annotating.text() == "Annotating: nose", "a drag is not a placement"
    assert dialog._dragging is None


def test_the_view_turns_a_press_on_a_grabbable_thing_into_a_drag_not_a_click(qapp):
    from detail.gui.qt import QEvent, QMouseEvent, QPixmap, QPointF

    from detail.gui.image_view import ImageView

    view = ImageView(zoomable=True)
    view.resize(200, 100)
    view.set_frame(QPixmap(200, 100))
    view.show()
    qapp.processEvents()
    events: list = []
    view.clicked.connect(lambda: events.append("clicked"))
    view.mark_drag_started.connect(lambda u, v: events.append(("start", u, v)))
    view.mark_dragged.connect(lambda u, v: events.append(("drag", u, v)))
    view.mark_drag_finished.connect(lambda u, v: events.append(("end", u, v)))
    view.hit_test = lambda pos: pos.x() < 100  # the left half is grabbable

    def mouse(kind, x, y):
        button = (
            Qt.MouseButton.LeftButton
            if kind != QEvent.Type.MouseMove
            else Qt.MouseButton.NoButton
        )
        event = QMouseEvent(
            kind,
            QPointF(x, y),
            button,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        {
            QEvent.Type.MouseButtonPress: view.mousePressEvent,
            QEvent.Type.MouseMove: view.mouseMoveEvent,
            QEvent.Type.MouseButtonRelease: view.mouseReleaseEvent,
        }[kind](event)

    mouse(QEvent.Type.MouseButtonPress, 50, 50)
    mouse(QEvent.Type.MouseMove, 80, 60)
    mouse(QEvent.Type.MouseButtonRelease, 300, 60)  # off the picture: clamped, kept
    assert [e[0] if isinstance(e, tuple) else e for e in events] == [
        "start",
        "drag",
        "end",
    ]
    assert events[-1][1] == 1.0, "a drag that leaves the picture keeps its grip"
    events.clear()
    mouse(QEvent.Type.MouseButtonPress, 150, 50)  # not grabbable: an ordinary click
    mouse(QEvent.Type.MouseButtonRelease, 150, 50)
    assert events == ["clicked"]


# --- a removal and a mark save never race (#26) ------------------------------- #
def _two_frame_set(proj):
    created = proj.create_example_set("rats", SK)
    for frame in (2, 4):
        proj.add_example_frame(
            created.id, "recording_001", "track_01", "part_001", frame, 0
        )
    return created


def test_a_mark_made_while_a_frame_is_being_removed_does_not_land_on_another_frame(
    qapp, proj, monkeypatch
):
    """Remove the frame on screen, and before the removal has come back press N (mark the
    current point hidden): the keys reach the dialog while the busy overlay covers only
    the view. Both writes address the frame by its *position* in the set, so on main the
    mark saved to position 0 after the removal had moved frame 4 into it. The mark must
    never land on a frame it was not made on, and the removal must stick."""
    import threading

    created = _two_frame_set(proj)
    removed = threading.Event()
    real_remove, real_marks = proj.remove_example_frame, proj.set_example_marks

    def remove_then_signal(*a, **k):
        try:
            return real_remove(*a, **k)
        finally:
            removed.set()

    def marks_after_the_removal(*a, **k):
        removed.wait(5)  # the worst order: the mark write lands second
        return real_marks(*a, **k)

    monkeypatch.setattr(proj, "remove_example_frame", remove_then_signal)
    monkeypatch.setattr(proj, "set_example_marks", marks_after_the_removal)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 2)
    assert dialog._on_screen() == 0

    dialog._frames.setCurrentRow(0)
    dialog._remove_frame.click()
    QTest.keyClick(dialog, Qt.Key.Key_N)  # before the removal's result is delivered
    removed.wait(5)
    _settled(qapp, dialog)

    stored = proj.example_set(created.id)
    assert [f.frame_idx for f in stored.frames] == [4], "the removal did not stick"
    assert stored.frames[0].marks == [], "frame 2's mark landed on frame 4"


def test_remove_waits_for_a_mark_that_is_still_saving(qapp, proj, monkeypatch):
    """The other order: a mark is saving when Remove is pressed. Remove is refused until
    the save settles, so the save cannot write by a position the removal has shifted."""
    import threading

    created = _two_frame_set(proj)
    release = threading.Event()
    real_marks = proj.set_example_marks

    def slow_marks(*a, **k):
        release.wait(5)
        return real_marks(*a, **k)

    monkeypatch.setattr(proj, "set_example_marks", slow_marks)
    dialog = _open(qapp, proj, created.id)
    _seek(qapp, dialog, 4)
    assert dialog._on_screen() == 1
    dialog.image_view().clicked_at.emit(0.25, 0.5)  # nose on frame 4, saving
    assert dialog.saving()

    dialog._frames.setCurrentRow(0)
    dialog._remove()  # straight at the handler, not only through the greyed button
    release.set()
    _settled(qapp, dialog)

    stored = proj.example_set(created.id)
    assert [f.frame_idx for f in stored.frames] == [2, 4], "removed mid-save"
    assert stored.frames[1].mark_for("nose") is not None
    assert stored.frames[0].marks == []


def test_marks_saving_when_the_dialog_closes_still_reach_the_disk(
    qapp, proj, monkeypatch
):
    """Closing the labeling screen with saves queued must not lose them (#24, finding 1's
    sibling). The saver's queue lived on the dialog: each next save started from the previous
    one's ``done``, which is dropped once the dialog is gone, so every frame's marks after the
    one in flight were lost. The saves now go through the app's write queue
    (``gui/run_writes.py``), which outlives the dialog."""
    import threading

    from detail.gui.qt import QEvent

    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    for frame in TRACKED:  # the frames join the set first, each with its nose marked
        _seek(qapp, dialog, frame)
        dialog.image_view().clicked_at.emit(0.25, 0.5)
        _settled(qapp, dialog)

    gate = threading.Event()
    real = type(proj).set_example_marks

    def slow(self, *args, **kwargs):
        gate.wait(20)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(proj), "set_example_marks", slow)

    def tails() -> int:
        frames = proj.example_set(created.id).frames
        return sum(f.mark_for("tail_base") is not None for f in frames)

    try:
        for frame in TRACKED:  # tail_base is the current point on each frame now
            _seek(qapp, dialog, frame)
            dialog.image_view().clicked_at.emit(0.75, 0.5)
            qapp.processEvents()
        assert dialog.saving()
        dialog.close()
        dialog.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    finally:
        gate.set()
    assert drain_until(
        qapp, lambda: tails() == len(TRACKED), 20_000
    ), f"only {tails()} of {len(TRACKED)} frames' marks reached the disk"
