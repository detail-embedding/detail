"""Arena canvas geometry + normalized polygon (the mouse→model mapping) and the spatial
workspace's state machine (select → scrub → ghost → set keyframe → save all).

`SceneGeometryDialog` replaced five dialogs and four editing idioms with one canvas that
edits every spatial kind, so the tests that used to be split per dialog are here as one set:
what a *kind* differs by is a row in `SPATIAL_KINDS`, and what a *definition* differs by is a
row in `_regions`.
"""

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from tests.gui.conftest import drain_thread_pool  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402


def _video_recording(tmp_path, n_frames=6, w=64, h=48, fps=30.0):
    """A recording backed by a tiny real video so get_frame works."""
    from detail.api.recording import Recording
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    proj_dir = experiment_store.create_experiment(tmp_path, "proj")
    recording_store.create_recording(proj_dir, "recording_001")
    recording = Recording(proj_dir, "recording_001")
    vid = recording.dir / "video" / "part_001.mp4"
    vid.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), (i * 30) % 256, np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(vid))
    m = 0
    while cap.read()[0]:
        m += 1
    cap.release()
    recording_store.write_timestamps_csv(
        recording.dir / "video/timestamps/generated/part_001.csv",
        np.arange(m, dtype=np.int64) * int(1e9 / fps),
    )
    data = recording_store.read_recording(recording.dir)
    data["video"] = {
        "storage": "default",
        "fps_container": fps,
        "fps_actual": fps,
        "fps_source": "container",
        "resolution": [w, h],
        "total_duration_s": m / fps,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    }
    recording_store.write_recording(recording.dir, data)
    return recording


def _row_for(dialog, index: int) -> int:
    """The list row showing region ``index``.

    Never `index` itself: the per-kind headings offset every row, so a region's position in
    the view and its identity in ``_regions`` are two different numbers — which is the whole
    reason the index is carried in the item's data.
    """
    from PyQt6.QtCore import Qt

    for row in range(dialog._list.count()):
        if dialog._list.item(row).data(Qt.ItemDataRole.UserRole) == index:
            return row
    raise AssertionError(f"region {index} has no row in the list")


def _select(dialog, index: int) -> None:
    """Select a region through the list, the way a click does — so `_on_select` runs."""
    dialog._list.setCurrentRow(_row_for(dialog, index))


def _click(canvas, x, y):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(x, y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    _release(canvas, x, y)


def _release(canvas, x, y, button=None):
    """A release as Qt sends one: ``button`` is the one that went up."""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    button = Qt.MouseButton.LeftButton if button is None else button
    canvas.mouseReleaseEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(x, y),
            button,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def _right_press(canvas, x, y):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(x, y),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def _right_move(canvas, x, y):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    canvas.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(x, y),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def _right_click(canvas, x, y):
    """A right press-release with no movement — deletes the vertex under it, if any."""
    from PyQt6.QtCore import Qt

    _right_press(canvas, x, y)
    _release(canvas, x, y, Qt.MouseButton.RightButton)


def _right_drag(canvas, x0, y0, x1, y1):
    """A right-button press-drag-release — the way a pan is actually driven, and the
    click-vs-drag distinction that keeps it from also deleting whatever vertex sat under
    the press."""
    from PyQt6.QtCore import Qt

    _right_press(canvas, x0, y0)
    _right_move(canvas, x1, y1)
    _release(canvas, x1, y1, Qt.MouseButton.RightButton)


def _wheel(canvas, x, y, angle_delta_y):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    canvas.wheelEvent(
        QWheelEvent(
            QPointF(x, y),
            QPointF(x, y),
            QPoint(0, 0),
            QPoint(0, angle_delta_y),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
    )


# --- the canvas ------------------------------------------------------------- #
def test_arena_canvas_image_rect_centers(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))  # 64×48, aspect 4:3
    canvas.resize(800, 360)  # wider than 4:3 → fit by height, letterbox + center in x
    rect = canvas._image_rect()
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (
        160.0,
        0.0,
        480.0,
        360.0,
    )


def test_arena_canvas_normalized_polygon(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas._points = [(0.1, 0.2), (0.9, 0.2), (0.5, 0.8)]
    assert canvas.normalized_polygon() == [(0.1, 0.2), (0.9, 0.2), (0.5, 0.8)]
    canvas.undo()
    assert len(canvas.normalized_polygon()) == 2
    canvas.clear()
    assert canvas.normalized_polygon() == []


def test_arena_canvas_loads_initial_points(qapp):
    from detail.gui.arena import ArenaCanvas

    points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8), points)
    assert canvas.normalized_polygon() == points


def test_arena_canvas_vertex_hit_test(qapp):
    from PyQt6.QtCore import QPointF

    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8), [(0.0, 0.0), (1.0, 1.0)])
    canvas.resize(480, 360)  # 64×48 fills exactly → rect (0,0,480,360)
    rect = canvas._image_rect()
    assert canvas._vertex_at(QPointF(rect.x(), rect.y())) == 0
    assert canvas._vertex_at(QPointF(rect.right(), rect.bottom())) == 1
    centre = QPointF(rect.x() + rect.width() / 2, rect.y() + rect.height() / 2)
    assert canvas._vertex_at(centre) is None  # far from every vertex


def test_arena_canvas_prominent_is_non_interactive(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    canvas.set_polygon(
        [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)], editable=False, prominent=True
    )
    assert canvas._editable is False and canvas._prominent is True
    _click(canvas, 10, 10)  # a click adds no point — not editable
    assert canvas.normalized_polygon() == [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]


def test_point_canvas_keeps_single_vertex(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8), max_points=1)
    canvas.resize(480, 360)  # fills exactly → rect (0,0,480,360)
    _click(canvas, 100, 100)
    _click(canvas, 300, 300)  # a second placement replaces — never two points
    assert len(canvas.normalized_polygon()) == 1


def test_calibration_canvas_caps_at_two_points(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8), max_points=2)
    canvas.resize(480, 360)  # 64×48 fills exactly → rect (0,0,480,360)
    _click(canvas, 50, 50)
    _click(canvas, 300, 200)
    _click(canvas, 100, 300)  # a third click is ignored — a calibration is two points
    assert len(canvas.normalized_polygon()) == 2


def test_canvas_max_points_can_be_changed_after_construction(qapp):
    """The cap is a property of what is *selected*, not of the canvas.

    One canvas now serves every kind, so a cap fixed at construction would let a click put a
    third vertex on a calibration the moment the selection moved off a polygon.
    """
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))  # unbounded polygon
    canvas.resize(480, 360)
    canvas.set_max_points(2)
    _click(canvas, 50, 50)
    _click(canvas, 300, 200)
    _click(canvas, 100, 300)
    assert len(canvas.normalized_polygon()) == 2
    canvas.clear()
    canvas.set_max_points(None)  # back to a polygon — the third click lands again
    _click(canvas, 50, 50)
    _click(canvas, 300, 200)
    _click(canvas, 100, 300)
    assert len(canvas.normalized_polygon()) == 3


def test_canvas_refuses_clicks_while_it_has_no_frame(qapp):
    """A dropped backdrop decode used to leave the previous frame on screen, so the next
    click placed a vertex against a picture the user was not looking at."""
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(64, 48)
    canvas.set_unavailable("(could not decode frame)")
    assert canvas.is_unavailable()

    _click(canvas, 10, 10)
    assert canvas.normalized_polygon() == []

    canvas.set_image(np.zeros((48, 64, 3), np.uint8))  # a frame decodes again
    assert not canvas.is_unavailable()
    _click(canvas, 10, 10)
    assert len(canvas.normalized_polygon()) == 1


def test_arena_scrubber_names_a_backdrop_it_cannot_decode(qapp):
    from detail.gui import frame_source
    from detail.gui.arena import _paint_decoded, ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    _paint_decoded(canvas, (None, frame_source.MISSING))
    assert canvas._unavailable == "(source video not in this export)"
    _paint_decoded(canvas, (None, frame_source.UNDECODABLE))
    assert canvas._unavailable == "(could not decode frame)"
    _paint_decoded(canvas, (np.zeros((48, 64, 3), np.uint8), None))
    assert not canvas.is_unavailable()


# --- the canvas: zoom and pan ------------------------------------------------ #
def test_zoom_in_and_out_change_scale_and_clamp(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    assert canvas.zoom_level() == 1.0
    assert not canvas.can_zoom_out()  # already at the floor — fit-to-widget

    canvas.zoom_in()
    assert canvas.zoom_level() == pytest.approx(1.25)
    assert canvas.can_zoom_out()

    canvas.zoom_out()
    assert canvas.zoom_level() == pytest.approx(1.0)
    assert not canvas.can_zoom_out()

    canvas.zoom_out()  # already at the floor — one more press does nothing
    assert canvas.zoom_level() == pytest.approx(1.0)

    for _ in range(30):  # past the ceiling
        canvas.zoom_in()
    assert canvas.zoom_level() == pytest.approx(canvas._MAX_ZOOM)
    assert not canvas.can_zoom_in()


def test_zoom_emits_zoom_changed(qapp):
    """The dialog's readout and button state are computed from this signal, not polled."""
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    seen = []
    canvas.zoom_changed.connect(seen.append)

    canvas.zoom_in()
    assert seen == [pytest.approx(1.25)]
    canvas.reset_zoom()
    assert seen == [pytest.approx(1.25), pytest.approx(1.0)]
    canvas.reset_zoom()  # already fit — no redundant emit
    assert len(seen) == 2


def test_zoom_button_keeps_the_widget_center_fixed(qapp):
    """Zoom in/out is anchored on the *widget* center, so the point in the middle of the
    canvas is the same image point before and after — the frame grows/shrinks around what
    you were already looking at, not around some other corner."""
    from PyQt6.QtCore import QPointF

    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    center = QPointF(240, 180)
    before = canvas._normalized(center)
    canvas.zoom_in()
    canvas.zoom_in()
    after = canvas._normalized(center)
    assert before == pytest.approx(after, abs=1e-9)


def test_wheel_zooms_toward_the_cursor(qapp):
    """Scrolling over a point keeps that point under the cursor — the anchor a vertex sits
    at has to stay put or zooming in to place one just pushes it out from under the mouse.
    """
    from PyQt6.QtCore import QPointF

    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    anchor = QPointF(120, 90)  # off-center, on purpose
    before = canvas._normalized(anchor)

    _wheel(canvas, 120, 90, 120)  # one notch in
    assert canvas.zoom_level() > 1.0
    after = canvas._normalized(anchor)
    assert before == pytest.approx(after, abs=1e-9)


def test_wheel_notch_direction_matches_in_and_out(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    _wheel(canvas, 240, 180, 120)  # away from the user — zoom in
    assert canvas.zoom_level() > 1.0
    zoomed = canvas.zoom_level()
    _wheel(canvas, 240, 180, -120)  # toward the user — zoom out
    assert canvas.zoom_level() < zoomed


def test_wheel_is_inert_with_no_frame_on_screen(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    canvas.set_unavailable("(could not decode frame)")
    _wheel(canvas, 240, 180, 120)
    assert canvas.zoom_level() == 1.0


def test_right_drag_pans_the_view(qapp):
    """Not middle-drag or Ctrl-left — `docs/gui_design.md` §3.1's "a binding is only real
    on the platform it can be produced on" records the 3-D scatter (now the pedigree) that
    shipped pan bound to exactly those, which a MacBook trackpad cannot produce. A plain
    right-drag is a device every pointer has."""
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    canvas.zoom_in()  # nothing to pan toward at zoom 1.0 — the whole frame already fits
    rect_before = canvas._image_rect()

    _right_drag(canvas, 240, 180, 200, 150)  # drag left and up

    rect_after = canvas._image_rect()
    assert (rect_after.x(), rect_after.y()) != (rect_before.x(), rect_before.y())


def test_right_click_still_deletes_a_vertex(qapp):
    """The pan gesture shares right-click with vertex deletion — a plain click (no drag)
    must still delete, or panning would have silently taken away the way to remove a
    point."""
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(
        np.zeros((48, 64, 3), np.uint8), [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    )
    canvas.resize(480, 360)  # fills exactly → rect (0,0,480,360)
    rect = canvas._image_rect()

    _right_click(canvas, rect.x(), rect.y())  # the (0, 0) vertex, no movement

    assert canvas.normalized_polygon() == [(1.0, 0.0), (1.0, 1.0)]


def test_a_right_drag_starting_on_a_vertex_pans_instead_of_deleting_it(qapp):
    """The click/drag distinction has to hold at the vertex itself, not just on empty
    space — a press that starts on a vertex and then moves is a pan, not a delete."""
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(
        np.zeros((48, 64, 3), np.uint8), [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    )
    canvas.resize(480, 360)
    canvas.zoom_in()  # something to actually pan toward
    rect = canvas._image_rect()

    _right_drag(canvas, rect.x(), rect.y(), rect.x() - 40, rect.y() - 40)

    assert len(canvas.normalized_polygon()) == 3  # nothing deleted


def test_a_left_release_during_a_right_press_deletes_nothing(qapp):
    """Only the right button's own release can finish a right click. A left press and
    release while the right button is still held is a different gesture, and the vertex
    under the right press has to survive it."""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    from detail.gui.arena import ArenaCanvas

    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8), points)
    canvas.resize(480, 360)
    rect = canvas._image_rect()

    _right_press(canvas, rect.x(), rect.y())  # held on the (0, 0) vertex
    canvas.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(240, 180),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    _release(canvas, 240, 180, Qt.MouseButton.LeftButton)

    assert canvas.normalized_polygon() == points


def test_right_click_on_empty_space_does_nothing(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(
        np.zeros((48, 64, 3), np.uint8), [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    )
    canvas.resize(480, 360)
    _right_click(canvas, 240, 180)  # far from every vertex
    assert len(canvas.normalized_polygon()) == 3


def test_pan_and_zoom_never_touch_the_shape_or_dirty_it(qapp):
    """Zoom and pan are a view of the shape, not an edit to it — `edited` is Save's only
    signal that something changed, and a canvas that fires it on every scroll would light
    Save up for looking at a corner rather than for moving a vertex."""
    from detail.gui.arena import ArenaCanvas

    points = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8), points=list(points))
    canvas.resize(480, 360)
    seen = []
    canvas.edited.connect(lambda: seen.append(1))

    canvas.zoom_in()
    _right_drag(canvas, 240, 180, 200, 150)
    _wheel(canvas, 240, 180, 120)

    assert canvas.normalized_polygon() == points
    assert seen == []


def test_pan_is_clamped_so_the_image_can_never_fully_leave_the_widget(qapp):
    """A drag can bring any part of the frame to the widget's center but never past it —
    otherwise a long drag could pan the whole picture away with no way back but Fit."""
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    canvas.zoom_in()
    canvas.zoom_in()

    _right_drag(canvas, 240, 180, 240 + 5000, 180 + 5000)  # a wildly large drag
    assert 0.0 <= canvas._center.x() <= 1.0
    assert 0.0 <= canvas._center.y() <= 1.0


def test_reset_zoom_restores_the_original_fit(qapp):
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(
        800, 360
    )  # the asymmetric fit from test_arena_canvas_image_rect_centers
    fit_rect = canvas._image_rect()

    canvas.zoom_in()
    _right_drag(canvas, 400, 180, 300, 100)
    assert canvas._image_rect() != fit_rect

    canvas.reset_zoom()
    rect = canvas._image_rect()
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (
        fit_rect.x(),
        fit_rect.y(),
        fit_rect.width(),
        fit_rect.height(),
    )


def test_canvas_claims_the_wheel(qapp):
    """Without this, `PageScroll` (every dialog's body is wrapped in one) takes the wheel
    for itself the moment the dialog is tall enough to scroll, and this canvas never sees
    a wheel event at all — scroll-to-zoom silently stops working past some window size.
    """
    from detail.gui.arena import ArenaCanvas
    from detail.gui.scroll import OWNS_WHEEL

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    assert bool(canvas.property(OWNS_WHEEL)) is True


def test_alt_wheel_is_the_escape_hatch_back_to_the_page(qapp):
    """Alt+wheel is `gui/scroll.py`'s documented way to reach the page even over a widget
    that owns the wheel — matching `ImageView`'s own convention."""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((48, 64, 3), np.uint8))
    canvas.resize(480, 360)
    event = QWheelEvent(
        QPointF(240, 180),
        QPointF(240, 180),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.AltModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(event)
    assert canvas.zoom_level() == 1.0
    assert not event.isAccepted()


# --- the workspace: zoom controls --------------------------------------------- #
def test_workspace_zoom_buttons_drive_the_canvas_and_readout(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    assert dialog._zoom_label.text() == "100%"
    assert not dialog._zoom_out_btn.isEnabled()  # already fit — nothing to zoom out to
    assert dialog._zoom_in_btn.isEnabled()

    dialog._zoom_in_btn.click()
    assert dialog._zoom_label.text() == "125%"
    assert dialog._zoom_out_btn.isEnabled()

    dialog._zoom_fit_btn.click()
    assert dialog._zoom_label.text() == "100%"
    assert not dialog._zoom_out_btn.isEnabled()

    # Drain the off-thread frame-zero decode this dialog kicked off at construction —
    # otherwise it can still be running when a later test's teardown tears down Qt
    # objects out from under it (`test_spatial_workspace_decodes_frame_zero_off_thread`'s
    # own reason for doing the same).
    assert drain_until(qapp, lambda: dialog._canvas._pixmap.width() == 64)


# --- the workspace: frame decode -------------------------------------------- #
def test_spatial_workspace_defers_frame_decode_off_thread(qapp, tmp_path, monkeypatch):
    # The first frame must decode off-thread: constructing the workspace must not call
    # get_frame on the UI thread (else opening it on a large video freezes to a blank
    # window). Stub get_frame to raise; construction must still succeed.
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)

    def _raise(*args, **kwargs):
        raise AssertionError("get_frame called on the UI thread")

    monkeypatch.setattr(recording, "get_frame", _raise)
    dialog = SceneGeometryDialog(recording)  # deferred → no raise here
    assert dialog._scrubber is not None
    drain_thread_pool(qapp)  # let the (raising) decode drain


def test_spatial_workspace_decodes_frame_zero_off_thread(qapp, tmp_path):
    """The placeholder is replaced by the *real* frame once the worker delivers.

    Deferring the decode is only half the contract — a workspace that defers it and never
    paints the result leaves you drawing on the blank stand-in.
    """
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)  # 64×48
    dialog = SceneGeometryDialog(recording)
    assert dialog._canvas._pixmap.width() == 480  # the _BLANK_FRAME placeholder
    assert drain_until(qapp, lambda: dialog._canvas._pixmap.width() == 64)


# --- the workspace: the definition list ------------------------------------- #
def test_workspace_lists_every_kind_under_its_own_heading(qapp, tmp_path):
    """One heading per kind, always — an empty kind still says it is empty."""
    from PyQt6.QtCore import Qt

    from detail.gui.arena import SPATIAL_KINDS, SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)

    rows = [dialog._list.item(r) for r in range(dialog._list.count())]
    assert [r.text() for r in rows] == [f"{k.label} (0)" for k in SPATIAL_KINDS]
    for item in rows:
        assert item.data(Qt.ItemDataRole.UserRole) is None  # a heading is not a region
        assert not (item.flags() & Qt.ItemFlag.ItemIsSelectable)
    assert dialog._selected is None


def test_workspace_row_identity_is_the_region_not_the_row(qapp, tmp_path):
    """A row's position in the view is not the record it names.

    The headings offset every row, so selecting the ROI's row must select the ROI — the
    index rides in the item's data, and reading it off `row()` would edit the arena.
    """
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.define_arena("rig", [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])])
    recording.set_rois(
        [{"name": "zone", "keyframes": [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])]}]
    )
    dialog = SceneGeometryDialog(recording)

    roi_index = next(i for i, r in enumerate(dialog._regions) if r["kind"] == "roi")
    row = _row_for(dialog, roi_index)
    assert row != roi_index  # the headings have moved it
    dialog._list.setCurrentRow(row)
    assert dialog._selected == roi_index
    assert dialog._name.text() == "zone"


def test_workspace_rename_keeps_the_selection_on_the_same_region(qapp, tmp_path):
    """Renaming rebuilds the list, which is where a selection read off the row is lost."""
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")
    dialog._canvas._points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    dialog._name.setText("feeder")
    dialog._on_rename()

    assert dialog._regions[0]["name"] == "feeder"
    assert dialog._selected == 0
    assert dialog._list.item(_row_for(dialog, 0)).text() == "feeder"
    # the list follows too — a rebuilt list that lost its current row leaves the workspace
    # editing a region the list no longer shows as chosen
    assert dialog._list.currentRow() == _row_for(dialog, 0)
    dialog._save_all()
    assert [r["name"] for r in recording.rois] == ["feeder"]


# --- the workspace: keyframes ------------------------------------------------ #
def test_workspace_keyframes_scrub_ghost_set_and_save(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    assert dialog._scrubber._slider.maximum() == recording.part_frame_counts[0] - 1
    dialog._add_region("arena")
    dialog._canvas._points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]  # draw at frame 0
    assert "keyframe (editable)" in dialog._kf_label.text()

    dialog._scrubber.set_frame(3, decode=False)  # captures kf0, shows a ghost at 3
    assert dialog._canvas._editable is False  # interpolated, read-only
    assert dialog._canvas._prominent is False  # a dimmed ghost, not a keyframe
    assert "interpolated" in dialog._kf_label.text()
    assert dialog._set_kf.isEnabled() and not dialog._del_kf.isEnabled()

    dialog._set_keyframe()  # frame 3 becomes a keyframe
    assert sorted(dialog._regions[0]["keyframes"]) == [0, 3]
    assert dialog._canvas._editable is True
    assert dialog._canvas._allow_topology is False  # locked with ≥2 keyframes

    dialog._canvas._points[0] = (0.2, 0.2)  # drag a vertex at frame 3
    dialog._save_all()
    keyframes = recording.arena_keyframes(recording.arenas[0]["id"])
    assert [f for f, _ in keyframes] == [0, 3]
    assert keyframes[0][1] == [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    assert keyframes[1][1][0] == (0.2, 0.2)


def test_workspace_delete_keyframe(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("arena")
    dialog._canvas._points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    dialog._scrubber.set_frame(3, decode=False)
    dialog._set_keyframe()
    assert sorted(dialog._regions[0]["keyframes"]) == [0, 3]

    dialog._delete_keyframe()  # removes the keyframe at frame 3
    assert sorted(dialog._regions[0]["keyframes"]) == [0]
    assert dialog._canvas._allow_topology is True  # topology unlocked again
    assert not dialog._del_kf.isEnabled()  # the last keyframe cannot be deleted


def test_workspace_shows_the_interpolated_shape_between_keyframes(qapp, tmp_path):
    """A non-keyframe frame shows where the shape *is* there, not the nearest keyframe.

    Asserting only "not editable" passes on a canvas that draws the first keyframe at every
    frame, which is the thing that makes a moving arena unreadable.
    """
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.define_arena(
        "rig",
        [
            (0, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),
            (4, [(0.4, 0.0), (1.0, 0.0), (0.0, 1.0)]),
        ],
    )
    dialog = SceneGeometryDialog(recording)
    _select(dialog, 0)
    dialog._scrubber.set_frame(2, decode=False)  # halfway between the two keyframes

    assert dialog._canvas._editable is False
    assert dialog._canvas.normalized_polygon()[0] == (0.2, 0.0)  # the midpoint
    assert "interpolated" in dialog._kf_label.text()


# --- the workspace: a locked definition -------------------------------------- #
def test_workspace_locked_definition_is_shown_but_not_editable(
    qapp, tmp_path, monkeypatch
):
    """A field of view a tracking run depends on is visible and inert.

    The run recorded that polygon, so every edit path has to be closed at once — not just
    the canvas: a greyed canvas beside a live Delete still loses the run's record.
    """
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.define_arena("rig", [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])])
    monkeypatch.setattr(recording, "arena_dependents", lambda aid: ["track_01"])

    dialog = SceneGeometryDialog(recording)
    assert "(locked)" in dialog._list.item(_row_for(dialog, 0)).text()
    _select(dialog, 0)

    # shown — frame 0 is a keyframe, so it draws as a solid outline, never grabbable
    assert dialog._canvas._editable is False and dialog._canvas._prominent is True
    assert dialog._canvas.normalized_polygon() == [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    assert "track_01" in dialog._kf_label.text()  # and it says why
    assert not dialog._name.isEnabled()
    for button in (
        dialog._delete,
        dialog._set_kf,
        dialog._del_kf,
        dialog._undo,
        dialog._clear,
    ):
        assert not button.isEnabled()

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dialog._delete_region()
    assert "track_01" in warned[0][2]  # refused, and the refusal names the run
    assert len(dialog._regions) == 1

    dialog._canvas.resize(480, 360)
    _click(dialog._canvas, 100, 100)  # a click neither edits it nor starts a region
    assert dialog._canvas.normalized_polygon() == [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    assert len(dialog._regions) == 1


# --- the workspace: regions and points --------------------------------------- #
def test_workspace_saves_a_region_whole_video(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")
    dialog._canvas._points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    dialog._save_all()

    roi = recording.rois[0]
    assert roi["n_vertices"] == 3 and roi["active"] is None  # whole video by default
    assert recording.roi_keyframes(roi["id"])[0][1] == [
        (0.1, 0.1),
        (0.9, 0.1),
        (0.5, 0.9),
    ]


def test_workspace_saves_a_point(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("point")
    dialog._canvas._points = [(0.5, 0.5)]
    dialog._save_all()

    point = recording.points[0]
    assert point["name"] == "point 1" and point["n_vertices"] == 1
    assert recording.point_keyframes(point["id"])[0][1] == [(0.5, 0.5)]


def test_workspace_adds_and_saves_multiple_regions(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")
    dialog._canvas._points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    dialog._add_region("roi")  # switching selection captures the first one
    dialog._canvas._points = [(0.2, 0.2), (0.4, 0.2), (0.3, 0.5)]
    dialog._save_all()

    assert [r["name"] for r in recording.rois] == ["region 1", "region 2"]
    assert [r["n_vertices"] for r in recording.rois] == [3, 3]


def test_workspace_cancel_discards(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.set_rois(
        [{"name": "keep", "keyframes": [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])]}]
    )
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")  # stage a new region in memory
    dialog._canvas._points = [(0.2, 0.2), (0.4, 0.2), (0.3, 0.5)]
    dialog.reject()  # Cancel — nothing written

    assert [r["name"] for r in recording.rois] == ["keep"]


def test_workspace_click_on_the_frame_starts_nothing(qapp, tmp_path):
    """Clicking the frame no longer begins a shape — Add is the only way in.

    It was a click people made by accident and could not undo without first noticing that
    a region had quietly appeared in the list. With four kinds on one canvas a bare click
    cannot mean all of them anyway, so the gesture has to name a kind, which is what the
    Add menu does.
    """
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    assert dialog._selected is None
    before = list(dialog._regions)

    dialog._canvas.resize(400, 300)
    dialog._canvas.show()
    qapp.processEvents()
    _click(dialog._canvas, 200, 150)

    assert dialog._regions == before, "a bare click created a definition"


def test_workspace_active_window_persists(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")
    dialog._canvas._points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    dialog._whole_video.setChecked(False)
    dialog._scrubber.set_frame(1, decode=False)
    dialog._set_active_bound(start=True)
    dialog._scrubber.set_frame(4, decode=False)
    dialog._set_active_bound(start=False)
    assert dialog._active_label.text() == "Active: frames 1–4"

    dialog._save_all()
    assert recording.rois[0]["active"] == {"start": 1, "end": 4}


def test_workspace_active_controls_are_hidden_for_a_field_of_view(qapp, tmp_path):
    """A field of view is always in effect, so the control that would limit it is gone."""
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")
    assert not dialog._active_row.isHidden()

    dialog._add_region("arena")
    assert dialog._active_row.isHidden()
    assert dialog._active_caption.isHidden() and dialog._active_label.isHidden()


# --- the workspace: every definition on one canvas ---------------------------- #
def test_workspace_selected_definition_keeps_its_siblings_as_context(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.set_rois(
        [
            {"name": "a", "keyframes": [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])]},
            {"name": "b", "keyframes": [(0, [(0.2, 0.2), (0.4, 0.2), (0.3, 0.5)])]},
        ]
    )
    dialog = SceneGeometryDialog(recording)
    _select(dialog, 0)

    drawn = [points for points, _c, _i, _v in dialog._canvas._context_shapes]
    assert drawn == [[(0.2, 0.2), (0.4, 0.2), (0.3, 0.5)]]  # the other one, as context
    assert dialog._canvas.normalized_polygon() == [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    assert dialog._region_color(0) != dialog._region_color(1)


def test_workspace_unchecking_a_definition_hides_it(qapp, tmp_path):
    """The checkbox is how you get a definition out of the way while drawing another."""
    from PyQt6.QtCore import Qt

    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.set_rois(
        [
            {"name": "a", "keyframes": [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])]},
            {"name": "b", "keyframes": [(0, [(0.2, 0.2), (0.4, 0.2), (0.3, 0.5)])]},
        ]
    )
    dialog = SceneGeometryDialog(recording)
    _select(dialog, 0)
    assert len(dialog._canvas._context_shapes) == 1

    dialog._list.item(_row_for(dialog, 1)).setCheckState(Qt.CheckState.Unchecked)
    assert dialog._regions[1]["visible"] is False
    assert dialog._canvas._context_shapes == []  # off the frame
    assert dialog._list.item(_row_for(dialog, 1)).text() == "b"  # still in the list


# --- the workspace: distance calibration -------------------------------------- #
def test_workspace_calibration_saves_and_derives_pixels_per_mm(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)  # 64×48
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("calibration")
    dialog._canvas._points = [(0.25, 0.5), (0.75, 0.5)]  # half the width → 32 px
    dialog._distance.setValue(200.0)  # 200 mm
    dialog._save_all()

    cal = recording.distance_calibration
    assert cal is not None and abs(cal["pixels_per_mm"] - 32 / 200) < 1e-9


def test_workspace_calibration_cm_unit_converts(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("calibration")
    dialog._canvas._points = [(0.25, 0.5), (0.75, 0.5)]  # 32 px
    dialog._unit.setCurrentText("cm")
    dialog._distance.setValue(20.0)  # 20 cm == 200 mm
    dialog._save_all()

    assert abs(recording.distance_calibration["pixels_per_mm"] - 32 / 200) < 1e-9


def test_workspace_calibration_fields_only_show_for_a_calibration(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")
    assert dialog._distance_holder.isHidden() and dialog._distance_label.isHidden()

    dialog._add_region("calibration")
    assert not dialog._distance_holder.isHidden()
    assert not dialog._distance_label.isHidden()


def test_workspace_seeds_an_existing_calibration(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)

    dialog = SceneGeometryDialog(recording)
    index = next(i for i, r in enumerate(dialog._regions) if r["kind"] == "calibration")
    _select(dialog, index)
    assert dialog._canvas.normalized_polygon() == [(0.25, 0.5), (0.75, 0.5)]
    assert dialog._distance.value() == 200.0
    assert dialog._unit.currentText() == "mm"


def test_workspace_calibration_is_a_singleton(qapp, tmp_path, monkeypatch):
    """There is one scale for the frame, so a second calibration is refused, not stacked."""
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    dialog = SceneGeometryDialog(recording)

    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: told.append(a))
    dialog._add_region("calibration")
    assert "distance calibration" in told[0][2] and "only be one" in told[0][2]
    assert len([r for r in dialog._regions if r["kind"] == "calibration"]) == 1


def test_workspace_untouched_calibration_is_never_rewritten(
    qapp, tmp_path, monkeypatch
):
    """Saving without touching the calibration must not revalidate it (#327/§342).

    A calibration set from the CLI's ``--pixels-per-mm`` has *no points* — nothing to put on
    a canvas — so validating it like a fresh two-point pick would refuse the whole save, and
    treating its empty canvas as a deletion would drop it. Untouched means untouched.
    """
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.set_pixels_per_mm(3.0)
    before = recording.pixels_per_mm

    dialog = SceneGeometryDialog(recording)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dialog._save_all()  # nothing moved

    assert not warned
    assert recording.pixels_per_mm == before


# --- the workspace: refusing a save it cannot make ---------------------------- #
def test_workspace_refuses_a_calibration_with_one_point(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QMessageBox

    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("calibration")
    dialog._canvas._points = [(0.25, 0.5)]  # only one point
    dialog._distance.setValue(200.0)

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dialog._save_all()

    assert "two points" in warned[0][2]
    assert recording.distance_calibration is None
    assert dialog.result() != QDialog.DialogCode.Accepted  # and it stays open


def test_workspace_refuses_a_region_below_its_minimum(qapp, tmp_path, monkeypatch):
    """A polygon of two points is a line; the refusal names the definition and the count."""
    from PyQt6.QtWidgets import QDialog, QMessageBox

    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    dialog = SceneGeometryDialog(recording)
    dialog._add_region("roi")
    dialog._canvas._points = [(0.1, 0.1), (0.9, 0.1)]

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dialog._save_all()

    assert "'region 1' needs at least 3 points on each keyframe." == warned[0][2]
    assert recording.rois == []
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_workspace_deletes_a_definition(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    recording = _video_recording(tmp_path)
    recording.set_rois(
        [
            {"name": "a", "keyframes": [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])]},
            {"name": "b", "keyframes": [(0, [(0.2, 0.2), (0.4, 0.2), (0.3, 0.5)])]},
        ]
    )
    dialog = SceneGeometryDialog(recording)
    _select(dialog, 0)
    dialog._delete_region()
    assert dialog._selected is None  # the selection went with it
    dialog._save_all()

    assert [r["name"] for r in recording.rois] == ["b"]


# --- the recording page's one way in ------------------------------------------ #
def test_recording_page_needs_a_video_to_open_the_workspace(qapp, tmp_path):
    """Every definition is drawn on a frame, so with no video there is nothing to draw on."""
    from PyQt6.QtWidgets import QPushButton

    from detail.api.experiment import Experiment
    from detail.gui.pages import RecordingDetailPage

    exp = Experiment.create("p", tmp_path)
    recording = exp.create_recording("s1")  # no video imported
    page = RecordingDetailPage(recording, experiment=exp)
    buttons = [
        b for b in page.findChildren(QPushButton) if b.text() == "Edit scene geometry…"
    ]
    assert buttons and all(not b.isEnabled() for b in buttons)
    assert all("Import a video part first" in b.toolTip() for b in buttons)


def test_recording_page_opens_the_spatial_workspace(qapp, tmp_path, monkeypatch):
    """One surface for all four kinds — the page's button opens the workspace itself."""
    from PyQt6.QtWidgets import QDialog, QPushButton

    from detail.gui import arena as arena_module
    from detail.gui.pages import RecordingDetailPage

    recording = _video_recording(tmp_path)
    recording.define_arena("rig", [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])])

    opened = []
    monkeypatch.setattr(
        arena_module.SceneGeometryDialog,
        "exec",
        lambda self: (opened.append(self), QDialog.DialogCode.Accepted)[1],
    )

    page = RecordingDetailPage(recording, experiment=None)
    buttons = [
        b for b in page.findChildren(QPushButton) if b.text() == "Edit scene geometry…"
    ]
    assert buttons and all(b.isEnabled() for b in buttons)
    buttons[0].click()
    assert len(opened) == 1


def test_recording_detail_pipeline_and_analysis_bands(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel

    from detail.gui.pages import RecordingDetailPage

    recording = _video_recording(tmp_path)
    page = RecordingDetailPage(recording)
    texts = {lbl.text() for lbl in page.findChildren(QLabel)}
    assert "DATA PIPELINE" in texts
    assert "ANALYSIS" in texts


def test_recording_page_summarizes_every_spatial_kind(qapp, tmp_path):
    """The four page sections became one summary, and it still prints what they printed.

    Every kind gets a line whether or not it is set — a kind that disappears when empty is a
    capability the user cannot find.
    """
    from PyQt6.QtWidgets import QLabel

    from detail.gui.arena import SPATIAL_KINDS
    from detail.gui.pages import RecordingDetailPage

    recording = _video_recording(tmp_path)
    recording.define_arena("rig", [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])])
    recording.define_roi(
        "window", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])], active=(1, 4)
    )
    page = RecordingDetailPage(recording)
    texts = {lbl.text() for lbl in page.findChildren(QLabel)}

    assert "Field of view (1)" in texts
    assert "Distance calibration" in texts  # the singleton keeps no count
    assert "Points of interest (0)" in texts
    assert "(none yet)" in texts  # an unset kind says so rather than vanishing
    assert any("roi_001 · window · 3 verts · frames 1–4" in t for t in texts)
    for kind in SPATIAL_KINDS:
        assert f"— {kind.scope}" in texts  # who reads it, per kind


def _as_comparable(shapes: list[tuple]) -> list[tuple]:
    return [(pts, color.name(), is_point, vis) for pts, color, is_point, vis in shapes]


def test_geometry_snapshot_matches_the_dialogs_own_resting_view(tmp_path):
    """`geometry_snapshot` is a second derivation of what `SceneGeometryDialog` draws
    with nothing selected (`_context_shapes`), so the two are held equal here — every
    shape, in order, points and colour both.

    The first version of this test compared the *set of colour names* over one
    definition per kind, which is the same set however a second definition is coloured
    and wherever any shape is drawn: both mutations passed it. So there are two ROIs
    (the second takes an unreserved colour), a calibration, and a region scoped past
    frame 0, which the resting view must leave out.
    """
    from detail.gui.arena import SceneGeometryDialog, geometry_snapshot

    recording = _video_recording(tmp_path)
    recording.define_arena("rig", [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])])
    recording.set_distance_calibration([(0.1, 0.5), (0.9, 0.5)], distance_mm=100.0)
    recording.define_roi("left", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])])
    recording.define_roi("right", [(0, [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0)])])
    recording.define_roi(
        "later", [(0, [(0.2, 0.2), (0.4, 0.2), (0.4, 0.4)])], active=(3, 5)
    )
    recording.define_point("lever", [(0, [(0.2, 0.3)])])

    shapes = _as_comparable(geometry_snapshot(recording, 0))
    dialog = SceneGeometryDialog(recording, None)
    assert dialog._selected is None  # the resting view: nothing is being edited
    assert shapes == _as_comparable(dialog._context_shapes())

    # arena, calibration, two ROIs, point — "later" is not active at frame 0
    assert len(shapes) == 5
    assert shapes[2][1] != shapes[3][1]  # a kind's second definition is not its first


def test_geometry_snapshot_skips_a_region_inactive_at_frame_zero(tmp_path):
    """The one thing a picture of frame 0 cannot show: a region scoped to later frames.
    `geometry_snapshot` draws frame 0, so it draws nothing for one — matching the
    recording page's own text summary of the same fact."""
    from detail.gui.arena import geometry_snapshot

    recording = _video_recording(tmp_path)
    recording.define_roi(
        "post-injection", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])], active=(5, 8)
    )
    assert geometry_snapshot(recording, 0) == []
    assert len(geometry_snapshot(recording, 6)) == 1


def test_geometry_preview_shows_no_video_placeholder(qapp, tmp_path):
    """The "(click to enlarge)" hint is a claim about what a click does — it must not
    show over a placeholder a click does nothing to."""
    from detail.api.experiment import Experiment
    from detail.gui.arena import GeometryPreview

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    preview = GeometryPreview(recording)
    assert preview._canvas.is_unavailable()
    # `isVisible()` is False for any never-shown widget regardless of this — `isHidden()`
    # is the one that is independent of the ancestor chain and reflects the deliberate
    # `setVisible(False)`.
    assert preview._hint.isHidden()


def test_geometry_preview_draws_every_defined_shape(qapp, tmp_path):
    from detail.gui.arena import GeometryPreview

    recording = _video_recording(tmp_path)
    recording.define_arena("rig", [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])])
    recording.define_roi("window", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])])

    preview = GeometryPreview(recording)
    # `is_unavailable()` is False from construction (the blank placeholder is not an
    # "unavailable" state) — wait for the real frame, not the placeholder, to land.
    assert drain_until(qapp, lambda: preview._canvas._width == 64)
    assert len(preview._canvas._context_shapes) == 2
    assert (
        not preview._hint.isHidden()
    )  # a real picture landed — the click now does something


def test_the_wheel_over_the_page_preview_scrolls_the_page_not_the_picture(
    qapp, tmp_path
):
    """`ArenaCanvas` claims the wheel so the editor can zoom. The recording page embeds the
    same canvas as a thumbnail, where a claim on the wheel would make the page stop
    scrolling wherever the pointer crosses the picture. The thumbnail's canvas is
    transparent to the mouse, so Qt delivers the wheel to the preview around it, and
    `PageScroll` never sees the canvas's claim.

    The receiver is found the way Qt finds it (`childAt`, which skips mouse-transparent
    widgets) rather than by sending to the canvas directly, which would skip exactly the
    step under test.
    """
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    from detail.gui.arena import GeometryPreview
    from detail.gui.scroll import PageScroll

    recording = _video_recording(tmp_path)
    scroll = PageScroll()
    page = QWidget()
    box = QVBoxLayout(page)
    preview = GeometryPreview(recording)
    preview.setMinimumHeight(300)
    box.addWidget(preview)
    for i in range(40):
        label = QLabel(f"row {i}")
        label.setMinimumHeight(30)
        box.addWidget(label)
    scroll.setWidget(page)
    scroll.resize(800, 600)
    scroll.show()
    assert drain_until(qapp, lambda: preview._canvas._width == 64)
    assert not preview._canvas.is_unavailable()  # a canvas with no frame ignores wheels
    assert scroll.verticalScrollBar().maximum() > 0

    center = preview._canvas.geometry().center()
    receiver = page.childAt(preview.mapTo(page, center))
    assert receiver is not None and receiver is not preview._canvas
    local = receiver.mapFrom(page, preview.mapTo(page, center))
    before = scroll.verticalScrollBar().value()
    QApplication.sendEvent(
        receiver,
        QWheelEvent(
            QPointF(local),
            QPointF(receiver.mapToGlobal(local)),
            QPoint(0, 0),
            QPoint(0, -360),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        ),
    )

    assert (
        scroll.verticalScrollBar().value() > before
    ), "the page stopped scrolling over the geometry preview"
    assert preview._canvas.zoom_level() == 1.0
    scroll.close()


def test_geometry_preview_click_opens_full_size_dialog(qapp, tmp_path, monkeypatch):
    """Not interactive on its own terms — a click hands off to the full-size read-only
    dialog rather than doing anything itself."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QDialog

    from detail.gui import arena as arena_module

    recording = _video_recording(tmp_path)
    preview = arena_module.GeometryPreview(recording)
    assert drain_until(qapp, lambda: preview._canvas._width == 64)

    opened = []
    monkeypatch.setattr(
        arena_module.GeometryPreviewDialog,
        "exec",
        lambda self: (opened.append(self), QDialog.DialogCode.Accepted)[1],
    )
    QTest.mouseClick(preview, Qt.MouseButton.LeftButton)
    assert len(opened) == 1


@pytest.mark.parametrize("state", ["decoding", "no_video"])
def test_geometry_preview_click_does_nothing_without_a_picture(
    qapp, tmp_path, monkeypatch, state
):
    """The hint, the pointing hand and the click read one flag. The click used to check
    only for a placeholder, so while frame 0 was still decoding — no placeholder yet,
    hint hidden — it opened the enlarged dialog on a blank frame."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from detail.api.experiment import Experiment
    from detail.gui import arena as arena_module

    if state == "decoding":
        # Hold the decode in flight for the whole test, rather than racing the click
        # against a background thread.
        recording = _video_recording(tmp_path)
        monkeypatch.setattr(arena_module, "run_busy", lambda *a, **k: None)
    else:
        recording = Experiment.create("p", tmp_path).create_recording("s1")
    opened = []
    monkeypatch.setattr(
        arena_module.GeometryPreviewDialog, "exec", lambda self: opened.append(self)
    )
    preview = arena_module.GeometryPreview(recording)
    QTest.mouseClick(preview, Qt.MouseButton.LeftButton)
    assert opened == []
    assert preview._hint.isHidden()
    assert preview.cursor().shape() != Qt.CursorShape.PointingHandCursor


def test_geometry_preview_dialog_shows_the_same_shapes(qapp, tmp_path):
    from detail.gui.arena import GeometryPreviewDialog, geometry_snapshot

    recording = _video_recording(tmp_path)
    recording.define_arena("rig", [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])])

    dialog = GeometryPreviewDialog(recording)
    assert drain_until(qapp, lambda: dialog._canvas._width == 64)
    assert len(dialog._canvas._context_shapes) == len(geometry_snapshot(recording, 0))
