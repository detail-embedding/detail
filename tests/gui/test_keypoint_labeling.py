"""The labeling screen (§26): a player with the example set beside it.

Driven over the fixture experiment with a real 6-frame video and a real tracking run, so the
decode, the tracked boxes and every save go through the paths a user's clicks take."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from detail.core.skeleton import Skeleton  # noqa: E402
from detail.gui import keypoint_labeling as mod  # noqa: E402
from tests.gui import subjects  # noqa: E402
from tests.gui.conftest import drain_busy_calls, drain_until  # noqa: E402

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


def test_the_strategy_picker_offers_every_strategy_in_the_table(qapp, proj):
    from detail.pipeline import subject_frames

    dialog = _open(qapp, proj, proj.create_example_set("rats", SK).id)
    offered = tuple(
        dialog._strategy.itemData(i) for i in range(dialog._strategy.count())
    )
    assert offered == subject_frames.STRATEGIES


def _offers_heading(dialog) -> bool:
    strategies = {dialog._strategy.itemData(i) for i in range(dialog._strategy.count())}
    shown = not dialog._heading_box.isHidden()
    assert shown == ("heading" in strategies), "the compass and By heading disagree"
    assert shown == dialog._compass.isVisible() == dialog._floor.isVisible()
    assert dialog._heading_note.isVisible() != shown
    return shown


def test_a_top_down_experiment_drops_the_compass_and_by_heading(qapp, proj):
    """From above the animal never faces toward or away from the camera (§26 *Headings*):
    the compass, the floor and By heading go, and one line says why."""
    proj.set_camera_view("top_down")
    dialog = _open(qapp, proj, proj.create_example_set("rats", SK).id)

    assert not _offers_heading(dialog)
    assert "Top-down camera" in dialog._heading_note.text()
    assert dialog._strategy.currentData() == "varied"
    dialog._strategy.setCurrentIndex(dialog._strategy.findData("spread"))
    assert not dialog._n_recordings.isEnabled()  # the strategy controls still follow


@pytest.mark.parametrize("view", [None, "side", "angled_down"])
def test_any_other_view_or_none_keeps_the_compass(qapp, proj, view):
    """Not set is unknown, not top down (§10), so it keeps the sampler."""
    if view is not None:
        proj.set_camera_view(view)
    dialog = _open(qapp, proj, proj.create_example_set("rats", SK).id)
    assert _offers_heading(dialog)


def test_a_recording_own_view_overrides_a_top_down_default(qapp, proj):
    proj.set_camera_view("top_down")
    proj.get_recording("recording_001").set_camera_view("side")
    dialog = _open(qapp, proj, proj.create_example_set("rats", SK).id)
    assert _offers_heading(dialog)


def test_spread_over_time_greys_out_the_recordings_controls(qapp, proj):
    proj.set_recording_metadata("recording_001", {"setup": "1"})
    dialog = _open(qapp, proj, proj.create_example_set("rats", SK).id)
    assert [
        dialog._spread_by.itemData(i) for i in range(dialog._spread_by.count())
    ] == [
        None,
        "acquisition_setup",
        "setup",
    ]

    dialog._strategy.setCurrentIndex(dialog._strategy.findData("spread"))
    assert not dialog._n_recordings.isEnabled()
    assert not dialog._spread_by.isEnabled()
    assert "reads every recording" in dialog._n_recordings.toolTip()

    dialog._strategy.setCurrentIndex(dialog._strategy.findData("varied"))
    assert dialog._n_recordings.isEnabled() and dialog._spread_by.isEnabled()


def test_every_experiment_offers_its_acquisition_setup_to_spread_across(qapp, proj):
    """Every experiment has the canonical acquisition_setup column (storage/metadata.py), so a
    sample can always be spread across setups, even before anyone fills the column in.
    """
    dialog = _open(qapp, proj, proj.create_example_set("rats", SK).id)
    dialog._strategy.setCurrentIndex(dialog._strategy.findData("varied"))
    assert dialog._n_recordings.isEnabled() and dialog._spread_by.isEnabled()
    assert dialog._spread_by.findData("acquisition_setup") >= 0


def test_add_frames_passes_the_recordings_count_and_column(qapp, proj, monkeypatch):
    proj.set_recording_metadata("recording_001", {"setup": "1"})
    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    assert dialog._n_recordings.maximum() == 1  # the fixture tracks one recording
    dialog._strategy.setCurrentIndex(dialog._strategy.findData("varied"))
    dialog._spread_by.setCurrentIndex(dialog._spread_by.findData("setup"))
    seen = {}

    def capture(set_id, sources, n, **options):
        seen.update(options)
        return proj.example_set(set_id)

    monkeypatch.setattr(proj, "sample_example_frames", capture)
    dialog._add_frames.click()
    drain_busy_calls(qapp)
    assert seen["strategy"] == "varied"
    assert seen["from_recordings"] == 1 and seen["spread_by"] == "setup"


def test_add_frames_samples_the_tracking_chosen_when_a_recording_has_several(
    qapp, proj, monkeypatch
):
    """A recording with several tracking runs (each animal's track and their merge) was
    folded into one ``{recording: run}`` map, so the last run listed always won and an
    Intruder set sampled the merged run. *From tracking* appears then, chooses nothing for
    the person, and the chosen kind of run is what Add frames hands the sampler."""
    from detail.storage import recording as recording_store

    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    assert dialog._sample_tracking_row.isHidden()  # one tracking run: nothing to ask
    (only,) = dialog._offered
    assert dialog.sample_sources() == {only["recording_id"]: only["run_id"]}
    dialog.close()

    rec = proj.get_recording(only["recording_id"])
    data = recording_store.read_recording(rec.dir)
    first = next(e for e in data["tracking_runs"] if e["id"] == only["run_id"])
    data["tracking_runs"].append(
        {**first, "id": "track_09", "tracker_id": "track_merge", "class_name": "a+b"}
    )
    recording_store.write_recording(rec.dir, data)
    dialog = _open(qapp, proj, created.id)
    assert not dialog._sample_tracking_row.isHidden()
    assert dialog._sample_tracking.currentData() is None
    assert dialog.sample_sources() == {}
    assert not dialog._add_frames.isEnabled()
    assert "From tracking" in dialog._add_frames.toolTip()

    index = dialog._sample_tracking.findData("a+b · track_merge")
    assert index > 0, [
        dialog._sample_tracking.itemText(i)
        for i in range(dialog._sample_tracking.count())
    ]
    dialog._sample_tracking.setCurrentIndex(index)
    assert dialog._add_frames.isEnabled()
    seen = {}

    def capture(set_id, sources, n, **options):
        seen["sources"] = dict(sources)
        return proj.example_set(set_id)

    monkeypatch.setattr(proj, "sample_example_frames", capture)
    dialog._add_frames.click()
    drain_busy_calls(qapp)
    assert seen["sources"] == {only["recording_id"]: "track_09"}


def test_from_tracking_names_each_class_as_the_experiment_does(qapp, proj):
    """*From tracking* labels runs by class: the experiment's name for each, not the stored
    one (§1 *Class names*), and a merged run's classes each by theirs."""
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    created = proj.create_example_set("rats", SK)
    dialog = _open(qapp, proj, created.id)
    (only,) = dialog._offered
    dialog.close()
    rec = proj.get_recording(only["recording_id"])
    data = recording_store.read_recording(rec.dir)
    first = next(e for e in data["tracking_runs"] if e["id"] == only["run_id"])
    data["tracking_runs"].append(
        {**first, "id": "track_09", "tracker_id": "track_merge", "class_name": "a+b"}
    )
    recording_store.write_recording(rec.dir, data)
    experiment_store.write_class_names(proj.dir, {"a": "Resident"})

    dialog = _open(qapp, proj, created.id)
    texts = [
        dialog._sample_tracking.itemText(i)
        for i in range(dialog._sample_tracking.count())
    ]
    assert any(t.startswith("Resident+b · track_merge") for t in texts), texts
    assert not any("a+b" in t for t in texts), texts


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
    from PyQt6.QtWidgets import QDialogButtonBox

    ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
    assert ok.text() == "Use these frames"


def test_keys_choose_a_point_and_mark_it(qapp, proj):
    from PyQt6.QtTest import QTest

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
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent, QPixmap

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
