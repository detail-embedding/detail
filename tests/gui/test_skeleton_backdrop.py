"""A frame from a recording behind the skeleton canvas (§26 *The Keypoints manager*).

The whole path a user takes — **Frame from a recording…** in the skeleton editor, a recording
picked by name, played and scrubbed on the shared ``FrameScrubber``, a crop dragged on the frame,
**Use this frame** — runs against a real experiment with a real video, the decode off the UI
thread as in the app. The crop gestures are driven as real mouse events.
"""

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QDialog  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.skeleton import Skeleton  # noqa: E402
from detail.gui import keypoints_manager  # noqa: E402
from detail.gui import skeleton_backdrop as mod  # noqa: E402
from detail.gui import skeleton_editor  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
W, H, FRAMES = 80, 40, 8


def _video(path):
    """Frames that differ from each other and across the width, so a wrong frame or a wrong
    crop is a different picture."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    ramp = np.linspace(0, 200, W, dtype=np.uint8)
    for i in range(FRAMES):
        frame = np.empty((H, W, 3), np.uint8)
        frame[:] = ramp[None, :, None]
        frame[:, :, 0] = (i * 30) % 256
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def experiment(tmp_path):
    """Two recordings, named: one with a video, one without."""
    exp = Experiment.create("larvae", tmp_path)
    fish = exp.add_recording("fish_A1")
    fish.add_video_part(_video(tmp_path / "fish_A1.mp4"))
    exp.add_recording("no_video_yet")
    return exp


def _picker(qapp, experiment):
    dialog = mod.FramePickerDialog(experiment)
    dialog.resize(900, 700)
    dialog.show()
    assert drain_until(qapp, lambda: dialog._shown is not None)
    return dialog


def _at(canvas, u, v):
    rect = canvas._image_rect()
    return QPointF(rect.x() + u * rect.width(), rect.y() + v * rect.height())


def _mouse(canvas, kind, pos, button):
    event = QMouseEvent(kind, pos, button, button, Qt.KeyboardModifier.NoModifier)
    {
        QEvent.Type.MouseButtonPress: canvas.mousePressEvent,
        QEvent.Type.MouseMove: canvas.mouseMoveEvent,
        QEvent.Type.MouseButtonRelease: canvas.mouseReleaseEvent,
    }[kind](event)


def _drag(canvas, start, end, button=LEFT):
    _mouse(canvas, QEvent.Type.MouseButtonPress, _at(canvas, *start), button)
    _mouse(canvas, QEvent.Type.MouseMove, _at(canvas, *end), button)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, _at(canvas, *end), button)


# -- the pure helpers ----------------------------------------------------------- #
def test_square_backdrop_centres_the_crop_on_the_canvas_grey():
    image = np.full((10, 30, 3), 200, np.uint8)
    out = mod.square_backdrop(image)
    assert out.shape == (30, 30, 3)
    assert (out[10:20] == 200).all()
    assert (out[:10] == mod.PAD_VALUE).all() and (out[20:] == mod.PAD_VALUE).all()
    assert mod.PAD_VALUE == int(skeleton_editor.BLANK[0, 0, 0])


def test_crop_box_takes_every_pixel_touched_and_is_never_empty():
    assert mod.crop_box((0.25, 0.5, 0.5, 1.0), 80, 40) == (20, 20, 40, 40)
    assert mod.crop_box((0.251, 0.0, 0.499, 0.1), 80, 40) == (20, 0, 40, 4)
    left, top, right, bottom = mod.crop_box((1.0, 1.0, 1.0, 1.0), 80, 40)
    assert right - left == 1 and bottom - top == 1


# -- the picker ------------------------------------------------------------------- #
def test_picker_lists_recordings_by_name_and_opens_the_first_in_the_player(
    qapp, experiment
):
    dialog = _picker(qapp, experiment)
    names = [dialog._recordings.itemText(i) for i in range(dialog._recordings.count())]
    assert names == ["fish_A1", "no_video_yet"]  # names, never recording_NNN
    scrubber = dialog.scrubber()
    assert scrubber.is_playable() and scrubber.play_button() is not None
    assert scrubber._total == FRAMES
    assert scrubber.time_text() is not None  # the timestamp map reached the row
    assert dialog._use.isEnabled()
    assert not dialog._whole.isEnabled()  # no crop yet: already the whole frame
    dialog.close()


def test_a_recording_without_video_says_so_and_offers_no_frame(qapp, experiment):
    dialog = _picker(qapp, experiment)
    dialog.select_recording(dialog._recordings.itemData(1))
    assert drain_until(
        qapp, lambda: dialog.canvas()._unavailable == "No video imported"
    )
    assert not dialog._use.isEnabled()
    assert dialog._use.toolTip() == mod._NO_FRAME
    assert not dialog.scrubber().isEnabled()
    dialog.close()


def test_a_frame_of_the_previous_recording_is_never_shown_under_the_next(
    qapp, experiment
):
    dialog = _picker(qapp, experiment)
    shown = dialog._shown
    dialog._on_rendered(5, ("recording_999", np.zeros((H, W, 3), np.uint8), None))
    assert dialog._shown is shown and dialog._shown_frame == 0
    dialog.close()


def test_crop_gestures_draw_move_resize_and_a_click_changes_nothing(qapp, experiment):
    dialog = _picker(qapp, experiment)
    canvas = dialog.canvas()
    _drag(canvas, (0.25, 0.25), (0.75, 0.75))
    assert canvas.crop() == pytest.approx((0.25, 0.25, 0.75, 0.75), abs=0.01)
    assert dialog._whole.isEnabled()
    assert "40 × 20 px" in dialog._crop_label.text()

    _drag(canvas, (0.5, 0.5), (0.4, 0.5))  # inside: moves, keeps its size
    assert canvas.crop() == pytest.approx((0.15, 0.25, 0.65, 0.75), abs=0.01)
    _drag(canvas, (0.65, 0.75), (0.9, 0.9))  # a corner: resizes about the opposite one
    assert canvas.crop() == pytest.approx((0.15, 0.25, 0.9, 0.9), abs=0.01)
    before = canvas.crop()
    _drag(canvas, (0.05, 0.05), (0.05, 0.05))  # a click outside: no new crop
    assert canvas.crop() == before
    _drag(canvas, (0.05, 0.05), (0.3, 0.3), button=RIGHT)  # a right drag pans
    assert canvas.crop() == before

    dialog._whole.click()
    assert canvas.crop() is None
    dialog.close()


def test_use_this_frame_takes_the_scrubbed_frame_cropped(qapp, experiment):
    dialog = _picker(qapp, experiment)
    dialog.scrubber().set_frame(5)
    assert drain_until(qapp, lambda: dialog._shown_frame == 5)
    _drag(dialog.canvas(), (0.5, 0.0), (1.0, 0.5))
    dialog.use_frame()

    assert dialog.result() == QDialog.DialogCode.Accepted
    recording = experiment.resolve_recording("fish_A1")
    expected = recording.get_frame(0, 5)[0:20, 40:80]
    assert np.array_equal(dialog.image(), expected)
    assert dialog.source() == "frame 5 of fish_A1, 40 × 20 px"


# -- the editor ------------------------------------------------------------------- #
def test_the_button_puts_the_picked_frame_behind_the_canvas(
    qapp, experiment, monkeypatch
):
    """Through the real entry point: the New skeleton dialog's button opens the picker."""

    def pick_and_crop(picker):
        assert drain_until(qapp, lambda: picker._shown is not None)
        picker.scrubber().set_frame(3)
        assert drain_until(qapp, lambda: picker._shown_frame == 3)
        _drag(picker.canvas(), (0.0, 0.0), (0.5, 1.0))
        picker.use_frame()
        return picker.result()

    monkeypatch.setattr(mod.FramePickerDialog, "exec", pick_and_crop)
    dialog = keypoints_manager.SkeletonEditorDialog(
        None, [], set(), experiment=experiment
    )
    dialog.resize(900, 800)
    dialog.show()
    editor = dialog.editor()
    edits = []
    editor.edited.connect(lambda: edits.append(1))
    assert not editor._plain.isEnabled()
    before = editor.skeleton().to_dict()

    editor._pick_frame.click()

    backdrop = editor.backdrop()
    assert backdrop.shape == (H, W // 2, 3)
    canvas = editor.canvas()
    assert (canvas._width, canvas._height) == (H, H)  # padded to the canvas's square
    assert "frame 3 of fish_A1" in editor._backdrop_label.text()
    assert editor._plain.isEnabled()
    assert (
        not edits and editor.skeleton().to_dict() == before
    )  # a drawing aid, not an edit

    # A point lands where it was drawn, in the same fractions as on the plain canvas.
    pos = _at(canvas, 0.5, 0.25)
    _mouse(canvas, QEvent.Type.MouseButtonPress, pos, LEFT)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, pos, LEFT)
    assert editor.skeleton().layout == ((0.5, 0.25),)

    editor._plain.click()
    assert editor.backdrop() is None
    assert (canvas._width, canvas._height) == skeleton_editor.BLANK.shape[:2]
    assert editor.skeleton().layout == ((0.5, 0.25),)
    dialog.close()


def test_without_an_open_experiment_the_button_says_why(qapp):
    editor = skeleton_editor.SkeletonEditor(experiment=None)
    editor.load(Skeleton(names=("a",)))
    assert not editor._pick_frame.isEnabled()
    assert "Open an experiment" in editor._pick_frame.toolTip()
    assert editor._backdrop_label.text() == "Background: plain"


def test_the_keypoints_manager_hands_the_open_experiment_to_the_editor(
    qapp, experiment, monkeypatch
):
    opened = []

    def capture(dialog):
        opened.append(dialog.editor()._experiment)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(keypoints_manager.SkeletonEditorDialog, "exec", capture)
    page = keypoints_manager.KeypointsManagerPage(experiment=experiment)
    assert drain_until(qapp, lambda: not page._reading)
    page._new_skeleton()
    assert opened == [experiment]
