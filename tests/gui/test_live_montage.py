"""The live montage: a clock per tile, hover pauses the rest, only visible tiles decode.

**Recorded class: a request queue served in the order requests were made, not the order they
are wanted.** The picture worker took the oldest-posted tile first and never dropped a request,
so a fast scroll to the bottom of a large cluster drew hundreds of tiles already scrolled
past before the ones in view. It now serves the first picture of every tile in view, in
display order, before any tile's second; then one row past the view in the scroll
direction; then later frames; and every scroll, resize and size change drops the requests
of tiles that left (``_PictureWorker.focus``), so none is stale. A tile asked again
while its first picture is in hand is a later frame, so it cannot jump another tile's first. The
order tests at the bottom of this file gate a slow source on its first call, move the view while
it is held, then read the order the source was called in.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtWidgets import QProgressBar, QPushButton  # noqa: E402

from detail.gui import live_montage as lm  # noqa: E402
from detail.gui.live_montage import LiveMontage, MontageClip  # noqa: E402


def _montage(**keys) -> LiveMontage:
    """A montage on its own: the page's keys as given, a context key and no exports."""
    from tests.gui.conftest import export_source

    keys.setdefault("context_key", "interpret/test_context")
    keys.setdefault("export", export_source())
    return LiveMontage(**keys)


SECOND = 1_000_000_000


class FakeSource:
    """Records every call, with the thread it ran on; draws a flat colour per match state."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, bool]] = []
        #: ``masks`` of every call, in the same order as ``calls``.
        self.masks: list[bool] = []
        self.looks: list = []
        self.threads: set[int] = set()
        self._lock = threading.Lock()

    def frame(self, clip_index, offset_ns, tile_px, matching, *, look):
        masks = look.masks
        self.looks.append(look)
        with self._lock:
            self.calls.append((clip_index, offset_ns, tile_px, matching))
            self.masks.append(masks)
            self.threads.add(threading.get_ident())
        value = (200 if matching else 120) if masks else 90
        return np.full((tile_px, tile_px, 3), value, np.uint8)


def _drain(qapp, predicate, timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _clips(n: int, seconds=lambda i: 1 + i % 3, matching=lambda i: True):
    return [
        MontageClip(int(seconds(i) * SECOND), i + 1, f"rec_{i}", matching(i))
        for i in range(n)
    ]


@pytest.fixture
def montage(qapp):
    widget = _montage(
        aligned_key="interpret/test_aligned", look_key="interpret/test_look"
    )
    widget.resize(700, 400)
    widget.show()
    qapp.processEvents()
    yield widget
    widget._worker.stop()
    widget.close()
    widget.deleteLater()


def test_each_tile_loops_on_its_own_clock(montage):
    montage.set_clips(_clips(3, seconds=lambda i: [1, 3, 0.5][i]))
    montage._clock.stop()  # drive the clocks by hand
    montage.advance(int(1.5 * SECOND))
    assert [montage.offset_ns(i) for i in range(3)] == [
        int(0.5 * SECOND),
        int(1.5 * SECOND),
        0,
    ]


def test_hovering_a_tile_pauses_every_other_one(montage):
    montage.set_clips(_clips(4))
    montage._clock.stop()
    montage._set_hovered(2)
    montage.advance(SECOND // 4)
    assert [montage.offset_ns(i) for i in range(4)] == [0, 0, SECOND // 4, 0]
    montage._set_hovered(None)
    montage.advance(SECOND // 4)
    assert [montage.offset_ns(i) for i in range(4)] == [SECOND // 4] * 2 + [
        SECOND // 2,
        SECOND // 4,
    ]


def test_mouse_over_a_tile_is_the_hover(montage, qapp):
    """Sent to the canvas directly: a synthesized cursor move is dropped under a loaded box."""
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    montage.set_clips(_clips(6))
    centre = QPointF(montage.tile_rect(1).center())
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        centre,
        montage.canvas.mapToGlobal(centre),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    qapp.sendEvent(montage.canvas, move)
    assert montage.hovered() == 1
    qapp.sendEvent(montage.canvas, QEvent(QEvent.Type.Leave))
    assert montage.hovered() is None


def test_one_timer_drives_every_tile_at_fifteen_frames_a_second(montage, qapp):
    from PyQt6.QtCore import QTimer

    montage.set_clips(_clips(8))
    assert montage._clock.isActive() and montage._clock.interval() == lm.TICK_MS
    assert len(montage.findChildren(QTimer)) >= 1
    timers = [t for t in montage.findChildren(QTimer) if t.isActive()]
    assert timers == [montage._clock]
    assert _drain(qapp, lambda: montage.offset_ns(0) > 0)


def test_only_visible_tiles_and_the_next_row_ask_for_pictures(montage, qapp):
    montage.set_clips(_clips(400))
    montage._clock.stop()
    _settle(qapp)
    source = FakeSource()
    montage.set_source(source)
    visible = set(montage.visible_indices()) | set(montage.lookahead_indices())
    assert 0 < len(visible) < 60
    assert _drain(qapp, lambda: {c[0] for c in source.calls} >= visible)
    montage.grid.verticalScrollBar().setValue(montage.tile_rect(200).top())
    later = set(montage.visible_indices()) | set(montage.lookahead_indices())
    assert 200 in later
    assert _drain(qapp, lambda: {c[0] for c in source.calls} >= later)
    assert {c[0] for c in source.calls} <= visible | later


def test_pictures_are_drawn_off_the_ui_thread(montage, qapp):
    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(4))
    assert _drain(qapp, lambda: len(montage._images) == 4)
    assert threading.get_ident() not in source.threads


def test_a_paused_tile_is_not_asked_again(montage, qapp):
    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(3))
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 3)
    montage._set_hovered(0)
    before = len(source.calls)
    montage.advance(SECOND // 15)
    for index in montage.visible_indices():
        montage._show_frame(index)
    assert _drain(qapp, lambda: len(source.calls) == before + 1)
    assert source.calls[-1][0] == 0


def test_a_non_matching_clip_is_drawn_from_the_grey_picture_and_keeps_playing(
    montage, qapp
):
    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(4))
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 4)
    montage.set_matching([True, False, True, True], filtered=True)
    assert _drain(qapp, lambda: (1, 0, lm.TILE_PX, False) in source.calls)
    montage.advance(SECOND // 2)
    assert montage.offset_ns(1) == SECOND // 2
    for index in montage.visible_indices():
        montage._show_frame(index)
    assert _drain(qapp, lambda: (1, SECOND // 2, lm.TILE_PX, False) in source.calls)


def test_set_matching_rejects_a_wrong_length(montage):
    montage.set_clips(_clips(3))
    with pytest.raises(ValueError, match="2 flags for 3 clips"):
        montage.set_matching([True, False], filtered=True)


def test_a_failing_source_marks_the_tile_and_the_rest_still_draw(montage, qapp):
    class Broken(FakeSource):
        def frame(self, clip_index, offset_ns, tile_px, matching, *, look):
            if clip_index == 1:
                raise OSError("proxy missing")
            return super().frame(clip_index, offset_ns, tile_px, matching, look=look)

    montage.set_source(Broken())
    montage.set_clips(_clips(3))
    assert _drain(qapp, lambda: montage._failed == {1} and len(montage._images) == 2)


def test_tile_size_and_order_controls_emit(montage, qapp):
    sizes, orders = [], []
    montage.tile_px_changed.connect(sizes.append)
    montage.order_changed.connect(orders.append)
    montage.size_slider.setValue(180)
    montage.order_actions["random"].trigger()
    assert sizes == [180] and montage.tile_px() == 180
    assert orders == ["random"] and montage.order() == "random"
    montage.set_order("typical")
    assert orders == ["random"] and montage.order() == "typical"


def test_hover_text_comes_from_the_callback(montage):
    montage.set_clips(_clips(3))
    montage.set_hover_text(lambda i: f"rank {i + 1} · ✓ video")
    index = montage.index_at(montage.tile_rect(2).center())
    assert montage.hover_text(index) == "rank 3 · ✓ video"
    assert montage.index_at(QPoint(-5, -5)) is None


def test_blocked_shows_the_progress_bar_and_stops_the_clock(montage, qapp):
    cancelled = []
    montage.set_source(FakeSource())
    montage.set_clips(_clips(3))
    montage.set_blocked(
        "Building previews · proxy · rec_1", 0.25, 90.0, lambda: cancelled.append(1)
    )
    assert montage.is_blocked() and not montage._clock.isActive()
    overlay = montage._overlay
    bar = overlay.findChild(QProgressBar)
    assert bar.value() == 250
    assert overlay._label.text() == (
        "Building previews · proxy · rec_1 · about 1m30s left"
    )
    montage.set_blocked(
        "Building previews · masks · rec_1", 0.5, None, lambda: cancelled.append(2)
    )
    assert bar.value() == 500
    assert overlay._label.text() == "Building previews · masks · rec_1"
    assert len(overlay.findChildren(QPushButton)) == 1
    assert _drain(qapp, lambda: overlay._cancel.isVisible())
    overlay._cancel.click()
    assert len(cancelled) == 1
    montage.clear_blocked()
    assert not montage.is_blocked() and montage._clock.isActive()


def test_blocked_asks_for_no_pictures(montage, qapp):
    source = FakeSource()
    montage.set_blocked("Building previews", 0.0)
    montage.set_source(source)
    montage.set_clips(_clips(3))
    qapp.processEvents()
    assert source.calls == []
    montage.clear_blocked()
    assert _drain(qapp, lambda: len(source.calls) == 3)


def test_hidden_montage_does_not_tick(montage):
    montage.set_clips(_clips(3))
    assert montage._clock.isActive()
    montage.hide()
    assert not montage._clock.isActive()


# --- Show ▾: the layers -------------------------------------------------------------------


def test_masks_toggle_asks_every_visible_tile_again_without_masks(montage, qapp):
    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(3))
    montage._clock.stop()
    masks = montage.show_button.layer_actions["masks"]
    assert montage.masks() and masks.isChecked()
    assert _drain(qapp, lambda: len(source.calls) >= 3)
    assert set(source.masks) == {True}
    before = len(source.calls)
    masks.trigger()
    assert not montage.masks() and not masks.isChecked()
    assert _drain(
        qapp, lambda: {c[0] for c in source.calls[before:]} >= {0, 1, 2}
    ), "a toggle redraws every visible tile at once, paused or not"
    assert set(source.masks[before:]) == {False}
    assert _drain(qapp, lambda: montage._images[0].pixelColor(5, 5).red() == 90)


def test_masks_choice_is_kept_for_the_next_montage(montage, qapp):
    montage.set_masks(False)
    again = _montage(
        aligned_key="interpret/test_aligned", look_key="interpret/test_look"
    )
    try:
        assert not again.masks()
        assert not again.show_button.layer_actions["masks"].isChecked()
    finally:
        again._worker.stop()
        again.deleteLater()
    montage.set_masks(True)
    fresh = _montage(
        aligned_key="interpret/test_aligned", look_key="interpret/test_look"
    )
    try:
        assert fresh.masks()
    finally:
        fresh._worker.stop()
        fresh.deleteLater()


def test_without_masks_a_clip_outside_the_filters_is_still_told_apart(montage, qapp):
    """With no grey mask to say so, a miss is washed out — and a filter change asks the source
    for nothing new, since the plain picture is the same either way."""
    source = FakeSource()
    montage.set_source(source)
    montage.set_masks(False)
    montage.set_clips(_clips(2, matching=lambda i: i == 0))
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 2)
    assert {c[3] for c in source.calls} == {True}, "no mask, nothing to colour"
    image = montage.canvas.grab().toImage()

    def centre(index):
        return image.pixelColor(montage.tile_rect(index).center())

    hit, miss = centre(0), centre(1)
    assert abs(hit.lightness() - miss.lightness()) > 40, (hit.name(), miss.name())
    before = len(source.calls)
    montage.set_matching([True, True], filtered=True)
    qapp.processEvents()
    assert len(source.calls) == before


class BlackSource(FakeSource):
    """Footage whose every row is black — the letterbox at a real proxy's foot, all over."""

    def frame(self, clip_index, offset_ns, tile_px, matching, *, look):
        super().frame(clip_index, offset_ns, tile_px, matching, look=look)
        return np.zeros((tile_px, tile_px, 3), np.uint8)


def _filled_run(image, track, ink) -> int:
    """How many pixels of the loop line's middle row, from its left end, are the fill's ink."""
    y = track.top() + track.height() // 2
    run = 0
    for x in range(track.left(), track.right() + 1):
        if image.pixelColor(x, y).name() != ink:
            break
        run += 1
    return run


def test_the_loop_line_fills_from_the_left_as_playback_advances_and_stays_off_the_footage(
    montage, qapp
):
    """The class: a progress mark drawn over footage, whose colours the theme does not control.
    On a black foot the near-black fill vanished and the unplayed remainder read as the bar,
    shrinking from the right — "the bar goes backwards". The line sits under the picture now,
    on a track whose colours the theme picks."""
    from PyQt6.QtGui import QColor

    from detail.gui import plot_theme

    theme = plot_theme.current()
    ink, track_ink = QColor(theme.summary).name(), QColor(theme.grid).name()
    montage.set_source(BlackSource())
    montage.set_clips(_clips(1, seconds=lambda i: 4))
    montage._clock.stop()
    montage._set_hovered(0)
    assert _drain(qapp, lambda: 0 in montage._images)
    cell = montage.tile_rect(0)
    picture = cell.adjusted(0, 0, 0, -(cell.height() - lm.TILE_PX))
    track, _fill = lm.progress_rects(picture, 0.0)
    assert not track.intersects(picture), "the loop line is drawn over the footage"
    assert cell.contains(track)
    runs = []
    for _step in range(3):
        montage.advance(SECOND)
        image = montage.canvas.grab().toImage()
        runs.append(_filled_run(image, track, ink))
        right = image.pixelColor(track.right(), track.top() + 1).name()
        assert right == track_ink, "the unplayed part is the track, not the footage"
        # Just inside the hover ring (2 px from the edge), the footage's lowest unringed row.
        foot = image.pixelColor(picture.center().x(), picture.bottom() - 3)
        assert foot.name() == "#000000", "the footage's foot is painted over"
    width = track.width()
    assert runs == sorted(runs) and runs[0] < runs[1] < runs[2], runs
    assert [round(r / width, 1) for r in runs] == [0.2, 0.5, 0.8]


def _band(montage, index):
    """The colours in tile ``index``'s loop-line band, from a fresh grab."""
    image = montage.canvas.grab().toImage()
    cell = montage.tile_rect(index)
    track, _fill = lm.progress_rects(
        cell.adjusted(0, 0, 0, -(cell.height() - montage._tile_px)), 0.0
    )
    return {
        image.pixelColor(x, y).name()
        for x in range(track.left(), track.right() + 1)
        for y in range(track.top(), track.bottom() + 1)
    }


def test_only_the_hovered_tile_draws_its_loop_line_and_hovering_moves_nothing(
    montage, qapp
):
    """Thirty moving bars pulled the eye off the footage: the line belongs to the one tile still
    playing. Its band is reserved on every tile, so the grid does not shift on hover."""
    from PyQt6.QtGui import QColor

    from detail.gui import plot_theme

    theme = plot_theme.current()
    line = {QColor(theme.summary).name(), QColor(theme.grid).name()}
    montage.set_source(FakeSource())
    montage.set_clips(_clips(3, seconds=lambda i: 4))
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 3)
    montage.advance(2 * SECOND)
    cells = [montage.tile_rect(i) for i in range(3)]
    for index in range(3):
        assert not _band(montage, index) & line, f"tile {index} draws a line unhovered"
    montage._set_hovered(1)
    montage.advance(SECOND)  # only the hovered tile plays
    assert [montage.tile_rect(i) for i in range(3)] == cells, "hovering moved the grid"
    assert QColor(theme.summary).name() in _band(montage, 1)
    assert QColor(theme.grid).name() in _band(montage, 1)
    for index in (0, 2):
        assert not _band(montage, index) & line, f"tile {index} draws a line"
    montage._set_hovered(None)
    qapp.processEvents()
    assert not _band(montage, 1) & line


def test_tiles_carry_no_rank_badge(montage, qapp):
    """The rank lives in the tooltip; the picture's corner is the flat footage, nothing over it."""
    montage.set_source(FakeSource())
    montage.set_clips(_clips(2))
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 2)
    image = montage.canvas.grab().toImage()
    corner = montage.tile_rect(1).topLeft()
    colours = {
        image.pixelColor(corner.x() + dx, corner.y() + dy).name()
        for dx in range(0, 40)
        for dy in range(0, 24)
    }
    assert colours == {"#c8c8c8"}, colours


def test_the_progress_line_contrasts_with_its_track_in_both_themes():
    from detail.core import chart_theme

    def luminance(hex_):
        rgb = [int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    for theme in (chart_theme.LIGHT, chart_theme.DARK):
        a, b = sorted((luminance(theme.summary), luminance(theme.grid)))
        assert (b + 0.05) / (a + 0.05) >= 3, theme


def _frame_pixel(montage, index):
    image = montage.canvas.grab().toImage()
    rect = montage.tile_rect(index)
    return image.pixelColor(rect.left(), rect.top() + lm.TILE_PX // 2).name()


@pytest.mark.parametrize("masks", [True, False])
def test_a_matching_tile_wears_the_match_frame_only_while_a_filter_is_on(
    montage, qapp, masks
):
    match, _miss = lm.match_colours()
    montage.set_source(BlackSource())
    montage.set_masks(masks)
    montage.set_clips(_clips(2))
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 2)
    assert _frame_pixel(montage, 0) != match.name(), "no filter, no frame"
    montage.set_matching([True, False], filtered=True)
    assert _drain(qapp, lambda: _frame_pixel(montage, 0) == match.name())
    assert _frame_pixel(montage, 1) != match.name()
    montage.set_matching([True, True], filtered=False)
    qapp.processEvents()
    assert _frame_pixel(montage, 0) != match.name()


def test_the_match_key_shows_only_while_a_filter_is_on(montage):
    montage.set_clips(_clips(2))
    assert not montage.match_key.isVisible()
    montage.set_matching([True, False], filtered=True)
    assert montage.match_key.isVisible()
    assert montage.match_key.texts() == ("matches filters",)
    montage.set_matching([True, True], filtered=False)
    assert not montage.match_key.isVisible()


def _click(
    montage, qapp, index, modifiers=Qt.KeyboardModifier.NoModifier, *, double=False
):
    """A press and release on the centre of tile ``index`` — and, ``double``, the second press
    of a double-click as Qt delivers it."""
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    centre = QPointF(montage.tile_rect(index).center())
    kinds = [QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease]
    if double:
        kinds += [QEvent.Type.MouseButtonDblClick, QEvent.Type.MouseButtonRelease]
    for kind in kinds:
        event = QMouseEvent(
            kind,
            centre,
            montage.canvas.mapToGlobal(centre),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            modifiers,
        )
        qapp.sendEvent(montage.canvas, event)


def _key(montage, qapp, key, modifiers=Qt.KeyboardModifier.NoModifier):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QKeyEvent

    qapp.sendEvent(montage.canvas, QKeyEvent(QEvent.Type.KeyPress, key, modifiers))


def test_set_matching_keeps_every_other_field_of_a_clip(montage):
    """It rebuilt clips by position, which would drop a field added after it was written."""
    clip = MontageClip(SECOND, 1, "rec", True, tag="not climb", faded=True, note="n")
    montage.set_clips([clip])
    montage.set_matching([False], filtered=True)
    (after,) = montage.clips()
    assert (after.tag, after.faded, after.note, after.matching) == (
        "not climb",
        True,
        "n",
        False,
    )


def test_clicks_do_nothing_without_a_hint(montage, qapp):
    """Interpret: no hint, so a click is not reported and nothing claims it would be."""
    clicked = []
    montage.clip_clicked.connect(lambda i, s: clicked.append(i))
    montage.set_clips(_clips(4))
    _click(montage, qapp, 1)
    assert clicked == [] and not montage.is_clickable()


def test_a_click_reports_the_tile_and_a_shift_click_reports_the_run(montage, qapp):
    """A Shift-click with no anchor is a click; with one it is the run from the anchor."""
    clicked, runs = [], []
    montage.clip_clicked.connect(lambda i, s: clicked.append((i, s)))
    montage.run_marked.connect(lambda r, b: runs.append((r, b)))
    montage.set_click_hint(lambda i: f"Click: remove {i}")
    montage.set_clips(_clips(8))
    _click(montage, qapp, 3, Qt.KeyboardModifier.ShiftModifier)
    _click(montage, qapp, 2)
    _click(montage, qapp, 5, Qt.KeyboardModifier.ShiftModifier)
    assert clicked == [(3, True), (2, False)]
    assert runs == [([2, 3, 4, 5], False)]
    assert montage.last_clicked() == 5


def test_the_click_cue_is_derived_from_the_wiring(montage, qapp):
    """A hint with nothing connected to `clip_clicked` advertises nothing: no hand, no ✕."""
    montage.set_click_hint(lambda i: "Click: remove from climb")
    montage.set_clips(_clips(2))
    assert not montage.is_clickable()
    assert montage.click_hint(0) == ""
    montage.clip_clicked.connect(lambda i, s: None)
    assert montage.is_clickable()
    assert montage.click_hint(0) == "Click: remove from climb"
    montage.set_click_hint(None)
    assert not montage.is_clickable()


def test_a_double_click_or_space_opens_the_clip(montage, qapp):
    opened = []
    montage.clip_activated.connect(opened.append)
    montage.set_clips(_clips(6))
    _click(montage, qapp, 3, double=True)
    assert opened == [3], "a double-click opens with or without click-to-mark"
    montage.clip_clicked.connect(lambda i, s: None)
    montage.set_click_hint(lambda i: "hint")
    _click(montage, qapp, 1)
    _key(montage, qapp, Qt.Key.Key_Space)
    assert opened == [3, 1], "Space opens the last clicked clip"
    montage._set_hovered(4)
    _key(montage, qapp, Qt.Key.Key_Space)
    assert opened == [3, 1, 4], "and the hovered one first"


def _cross_colours(montage, index) -> set:
    image = montage.canvas.grab().toImage()
    rect = montage.cross_rect(index)
    return {
        image.pixelColor(x, y).name()
        for x in range(rect.left() + 2, rect.right() - 1)
        for y in range(rect.top() + 2, rect.bottom() - 1)
    }


def test_hovering_a_clickable_tile_shows_the_hand_a_faint_cross_and_the_hint(
    montage, qapp
):
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    montage.set_source(BlackSource())
    montage.set_clips(_clips(3))
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 3)
    montage._set_hovered(0)
    qapp.processEvents()
    assert _cross_colours(montage, 0) == {"#000000"}, "no cue while not clickable"
    montage.clip_clicked.connect(lambda i, s: None)
    montage.set_click_hint(lambda i: "Click: remove from climb · double-click: open")
    centre = QPointF(montage.tile_rect(0).center())
    qapp.sendEvent(
        montage.canvas,
        QMouseEvent(
            QEvent.Type.MouseMove,
            centre,
            montage.canvas.mapToGlobal(centre),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    qapp.processEvents()
    assert montage.canvas.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert len(_cross_colours(montage, 0)) > 2, "the hovered tile shows a faint ✕"
    assert _cross_colours(montage, 1) == {"#000000"}, "only the hovered one"
    assert montage.hover_text(0).startswith("Click: remove from climb")


def test_a_crossed_tile_wears_a_bold_cross_whatever_the_mode(montage, qapp):
    """INTERPRET too: a removed clip looks the same wherever it is shown."""
    montage.set_source(BlackSource())
    montage.set_clips(
        [
            MontageClip(SECOND, 1, "a"),
            MontageClip(SECOND, 2, "b", tag="not climb", faded=True, crossed=True),
            MontageClip(SECOND, 3, "c", tag="→ rear", faded=True),
        ]
    )
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 3)
    assert _cross_colours(montage, 0) == {"#000000"}
    assert len(_cross_colours(montage, 1)) > 2, "removed: the ✕"
    plain_faded = _cross_colours(montage, 2)
    assert len(plain_faded) == 1, "moved: faded, no ✕"


def test_sections_start_on_a_new_row_under_their_heading(montage, qapp):
    clips = _clips(7)
    cols = montage._columns()
    assert cols > 3, "the fixture's width fits several columns"
    montage.set_clips(clips, sections=[("Removed (2)", 2), ("Kept as climb (5)", 5)])
    qapp.processEvents()
    (removed, r_rect), (kept, k_rect) = montage.heading_rects()
    assert (removed, kept) == ("Removed (2)", "Kept as climb (5)")
    assert montage.tile_rect(0).top() == r_rect.bottom() + 1
    assert (
        montage.tile_rect(2).left() == montage.tile_rect(0).left()
    ), "the second section starts a new row"
    assert montage.tile_rect(2).top() == k_rect.bottom() + 1
    assert k_rect.top() > montage.tile_rect(1).bottom()
    for index in range(7):
        assert montage.index_at(montage.tile_rect(index).center()) == index
    assert montage.index_at(k_rect.center()) is None, "a heading is not a tile"
    assert list(montage.visible_indices()) == list(range(7))
    assert montage.seen() == 7
    with pytest.raises(ValueError, match="sections hold 3 clips, not 7"):
        montage.set_clips(clips, sections=[("a", 1), ("b", 2)])
    montage.set_clips(clips)
    assert montage.heading_rects() == [] and montage.tile_rect(0).top() == 0


def test_a_section_heading_is_painted(montage, qapp):
    montage.set_clips(_clips(3), sections=[("Kept as climb (3)", 3)])
    qapp.processEvents()
    ((_title, rect),) = montage.heading_rects()
    image = montage.canvas.grab().toImage()
    colours = {
        image.pixelColor(x, y).name()
        for x in range(rect.left(), rect.left() + 80)
        for y in range(rect.top(), rect.bottom())
    }
    assert len(colours) > 2, "no heading text drawn"


def test_seen_grows_with_the_scroll_and_reaches_the_clip_count_at_the_end(
    montage, qapp
):
    reached = []
    montage.seen_changed.connect(reached.append)
    montage.set_clips(_clips(120))
    qapp.processEvents()
    first = montage.seen()
    assert 0 < first < 120
    bar = montage.grid.verticalScrollBar()
    bar.setValue(bar.maximum())
    qapp.processEvents()
    assert montage.seen() == 120 and reached[-1] == 120
    bar.setValue(0)
    assert montage.seen() == 120, "scrolling back up does not take it back"
    montage.set_clips(_clips(120))
    assert 0 < montage.seen() < 120, "new clips start over"


def test_a_hidden_montage_has_seen_nothing(qapp):
    hidden = _montage(
        aligned_key="interpret/test_aligned", look_key="interpret/test_look"
    )
    hidden.set_clips(_clips(3))
    assert hidden.seen() == 0
    hidden._worker.stop()


def test_a_faded_tile_is_washed_out_and_wears_its_tag(montage, qapp):
    """Masks on: a faded clip still gets the scrim, unlike a filter miss."""
    montage.set_source(BlackSource())
    montage.set_masks(True)
    montage.set_clips(
        [
            MontageClip(SECOND, 1, "a"),
            MontageClip(SECOND, 2, "b", tag="not climb", faded=True, note="Removed"),
        ]
    )
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) == 2)
    image = montage.canvas.grab().toImage()
    plain = image.pixelColor(montage.tile_rect(0).center())
    faded = image.pixelColor(montage.tile_rect(1).center())
    assert plain.name() == "#000000"
    assert faded.lightness() - plain.lightness() > 40, faded.name()
    corner = montage.tile_rect(1).topLeft()
    bottom = corner.y() + lm.TILE_PX
    tag = {
        image.pixelColor(corner.x() + dx, bottom - dy).name()
        for dx in range(3, 30)
        for dy in range(4, 12)
    }
    assert len(tag) > 2, "no tag painted in the bottom-left corner"
    assert montage.hover_text(1).startswith("Removed")


def test_keep_scroll_keeps_the_scroll_and_clocks(montage, qapp):
    clips = _clips(60)
    montage.set_clips(clips)
    qapp.processEvents()
    bar = montage.grid.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    montage._clock.stop()
    montage.advance(SECOND // 2)
    visible = montage.visible_indices()[0]
    offset = montage.offset_ns(visible)
    scroll = bar.value()
    assert scroll > 0
    faded = [replace_faded(c) if i == visible else c for i, c in enumerate(clips)]
    montage.set_clips(faded, keep_scroll=True)
    assert bar.value() == scroll
    assert montage.offset_ns(visible) == offset
    montage.set_clips(clips)
    assert bar.value() == 0


def replace_faded(clip):
    from dataclasses import replace

    return replace(clip, faded=True, tag="not climb")


def test_right_click_asks_for_a_menu_for_that_clip(montage, qapp):
    from PyQt6.QtGui import QContextMenuEvent

    seen = []
    montage.clip_context_requested.connect(lambda i, pos: seen.append(i))
    montage.set_clips(_clips(3))
    centre = montage.tile_rect(2).center()
    qapp.sendEvent(
        montage.canvas,
        QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, centre, montage.canvas.mapToGlobal(centre)
        ),
    )
    assert seen == [2]


# --- a run: drag, or Shift, with one preview (review round 14) --------------------------------- #


def _mouse(
    montage, qapp, kind, point, buttons, modifiers=Qt.KeyboardModifier.NoModifier
):
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    kinds = {
        "press": QEvent.Type.MouseButtonPress,
        "move": QEvent.Type.MouseMove,
        "release": QEvent.Type.MouseButtonRelease,
    }
    at = QPointF(point)
    button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    qapp.sendEvent(
        montage.canvas,
        QMouseEvent(
            kinds[kind], at, montage.canvas.mapToGlobal(at), button, buttons, modifiers
        ),
    )


def _drag(montage, qapp, start, end, *, release=True):
    """Press on tile ``start``, move through the tiles to ``end``, and (``release``) let go."""
    held = Qt.MouseButton.LeftButton
    _mouse(montage, qapp, "press", montage.tile_rect(start).center(), held)
    step = 1 if end >= start else -1
    for index in range(start, end + step, step):
        _mouse(montage, qapp, "move", montage.tile_rect(index).center(), held)
    if release:
        _mouse(
            montage,
            qapp,
            "release",
            montage.tile_rect(end).center(),
            Qt.MouseButton.NoButton,
        )


def _marking(montage, faded=()):
    runs, clicked = [], []
    montage.run_marked.connect(lambda r, b: runs.append((list(r), b)))
    montage.clip_clicked.connect(lambda i, s: clicked.append(i))
    montage.set_click_hint(lambda i: "Click: remove")
    montage.set_clips(
        [replace_faded(c) if i in faded else c for i, c in enumerate(_clips(10))]
    )
    return runs, clicked


def test_a_drag_previews_the_run_and_marks_it_once_on_letting_go(montage, qapp):
    runs, clicked = _marking(montage, faded={3})
    changed = []
    montage.preview_changed.connect(lambda: changed.append(montage.preview()))
    _drag(montage, qapp, 1, 5, release=False)
    preview = montage.preview()
    assert preview == lm.RunPreview("drag", False, (1, 2, 4, 5)), "3 is removed already"
    assert montage.is_dragging() and changed
    _mouse(
        montage, qapp, "release", montage.tile_rect(5).center(), Qt.MouseButton.NoButton
    )
    assert runs == [([1, 2, 4, 5], False)] and clicked == [], "one run, no click"
    assert montage.preview() is None and not montage.is_dragging()
    assert montage.last_clicked() == 5, "a Shift-click after it runs from its end"


def test_a_drag_back_through_the_grid_runs_in_reading_order(montage, qapp):
    runs, _clicked = _marking(montage)
    _drag(montage, qapp, 6, 2)
    assert runs == [([2, 3, 4, 5, 6], False)]


def test_a_drag_that_starts_on_a_removed_clip_puts_the_run_back(montage, qapp):
    runs, _clicked = _marking(montage, faded={2, 3, 5})
    _drag(montage, qapp, 2, 6, release=False)
    assert montage.preview() == lm.RunPreview(
        "drag", True, (2, 3, 5)
    ), "plain ones skipped"
    _mouse(
        montage, qapp, "release", montage.tile_rect(6).center(), Qt.MouseButton.NoButton
    )
    assert runs == [([2, 3, 5], True)]


def test_esc_cancels_a_drag(montage, qapp):
    runs, clicked = _marking(montage)
    _drag(montage, qapp, 0, 4, release=False)
    assert montage.preview() is not None
    _key(montage, qapp, Qt.Key.Key_Escape)
    assert montage.preview() is None and not montage.is_dragging()
    _mouse(
        montage, qapp, "release", montage.tile_rect(4).center(), Qt.MouseButton.NoButton
    )
    assert runs == [] and clicked == [], "the release after Esc does nothing"


def test_a_shaky_click_is_still_a_click(montage, qapp):
    from PyQt6.QtWidgets import QApplication

    runs, clicked = _marking(montage)
    centre = montage.tile_rect(2).center()
    held = Qt.MouseButton.LeftButton
    shake = QPoint(QApplication.startDragDistance() - 1, 0)
    _mouse(montage, qapp, "press", centre, held)
    _mouse(montage, qapp, "move", centre + shake, held)
    assert not montage.is_dragging() and montage.preview() is None
    _mouse(montage, qapp, "release", centre + shake, Qt.MouseButton.NoButton)
    assert clicked == [2] and runs == []


def test_holding_shift_previews_the_run_a_shift_click_removes(montage, qapp):
    runs, _clicked = _marking(montage, faded={4})
    _click(montage, qapp, 1)
    shift = Qt.KeyboardModifier.ShiftModifier
    _mouse(
        montage,
        qapp,
        "move",
        montage.tile_rect(6).center(),
        Qt.MouseButton.NoButton,
        shift,
    )
    assert montage.preview() == lm.RunPreview("shift", False, (1, 2, 3, 5, 6))
    _mouse(
        montage, qapp, "move", montage.tile_rect(6).center(), Qt.MouseButton.NoButton
    )
    assert montage.preview() is None, "letting go of Shift clears it"
    _key(montage, qapp, Qt.Key.Key_Shift)
    assert montage.preview() == lm.RunPreview("shift", False, (1, 2, 3, 5, 6))
    from PyQt6.QtCore import QEvent

    qapp.sendEvent(montage.canvas, QEvent(QEvent.Type.Leave))
    assert montage.preview() is None, "leaving the grid clears it"
    _key(montage, qapp, Qt.Key.Key_Shift)
    montage._set_hovered(6)
    _click(montage, qapp, 6, shift)
    assert runs == [([1, 2, 3, 5, 6], False)], "the Shift-click marks what was shown"


def test_nothing_previews_on_a_montage_that_is_not_clickable(montage, qapp):
    runs, clicked = _marking(montage)
    _click(montage, qapp, 1)
    montage.set_click_hint(None)
    _drag(montage, qapp, 1, 5, release=False)
    assert montage.preview() is None and not montage.is_dragging()
    _key(montage, qapp, Qt.Key.Key_Shift)
    montage._set_hovered(5)
    assert montage.preview() is None
    _mouse(
        montage, qapp, "release", montage.tile_rect(5).center(), Qt.MouseButton.NoButton
    )
    assert runs == [] and clicked == [1]


def test_turning_clicks_off_mid_drag_drops_the_preview(montage, qapp):
    runs, _clicked = _marking(montage)
    _drag(montage, qapp, 0, 3, release=False)
    montage.set_click_hint(None)
    assert montage.preview() is None and not montage.is_dragging()
    _mouse(
        montage, qapp, "release", montage.tile_rect(3).center(), Qt.MouseButton.NoButton
    )
    assert runs == []


def _tint_share(montage, index, colour) -> float:
    """The share of a picture's pixels within a few steps of ``colour`` over black."""
    from PyQt6.QtGui import QColor

    image = montage.canvas.grab().toImage()
    rect = montage.tile_rect(index)
    want = QColor(colour)
    hits = total = 0
    for x in range(rect.left() + 8, rect.left() + montage.tile_px() - 8, 4):
        for y in range(rect.top() + 30, rect.top() + montage.tile_px() - 30, 4):
            got = image.pixelColor(x, y)
            total += 1
            hits += (
                abs(got.red() - want.red())
                + abs(got.green() - want.green())
                + abs(got.blue() - want.blue())
                <= 12
            )
    return hits / total


def test_the_preview_tints_exactly_its_run_in_the_colour_of_what_it_does(montage, qapp):
    """Painted over black footage: a previewed tile carries the tint, one outside the run does
    not, and a put-back run is the other colour."""
    from PyQt6.QtGui import QColor

    montage.set_source(BlackSource())
    _marking(montage, faded={7})
    montage._clock.stop()
    assert _drain(qapp, lambda: len(montage._images) >= 8)

    def tint(put_back):
        colour = lm.preview_colour(put_back)
        over = QColor(colour)
        alpha = lm.PREVIEW_ALPHA / 255
        return QColor(
            round(over.red() * alpha),
            round(over.green() * alpha),
            round(over.blue() * alpha),
        )

    remove, back = tint(False), tint(True)
    assert remove != back
    _drag(montage, qapp, 1, 3, release=False)
    qapp.processEvents()
    assert _tint_share(montage, 2, remove) > 0.9
    assert _tint_share(montage, 5, remove) == 0, "outside the run: untouched"
    image = montage.canvas.grab().toImage()
    corner = montage.tile_rect(3).topLeft()
    ring = {
        image.pixelColor(corner.x() + 40, corner.y() + d).name()
        for d in range(lm.PREVIEW_RING_PX)
    }
    assert montage.hovered() == 3
    assert ring == {lm.preview_colour(False).name()}, "the hovered end keeps the ring"
    _key(montage, qapp, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert _tint_share(montage, 2, remove) == 0, "Esc takes the tint away"
    _drag(montage, qapp, 7, 7, release=False)
    _mouse(
        montage,
        qapp,
        "move",
        montage.tile_rect(7).center() + QPoint(12, 0),
        Qt.MouseButton.LeftButton,
    )
    qapp.processEvents()
    assert montage.preview() == lm.RunPreview("drag", True, (7,))
    centre = montage.canvas.grab().toImage().pixelColor(montage.tile_rect(7).center())
    assert centre.blue() > centre.red() + 20, "putting back is the blue one"


def test_dragging_at_the_bottom_edge_scrolls_and_grows_the_run(montage, qapp):
    runs, _clicked = _marking(montage)
    montage.set_clips(_clips(200))
    qapp.processEvents()
    bar = montage.grid.verticalScrollBar()
    assert bar.maximum() > 0
    _drag(montage, qapp, 0, 1, release=False)
    before = montage.preview().indices[-1]
    viewport = montage.grid.viewport()
    # Over a column of tiles: the columns are centred, so the viewport's middle can be a gap.
    montage._drag_global = viewport.mapToGlobal(
        QPoint(montage.tile_rect(1).center().x(), viewport.height() - 4)
    )
    for _ in range(10):
        montage._autoscroll_tick()
    assert bar.value() > 0, "the grid scrolled itself"
    assert montage.preview().indices[-1] > before, "and the run followed the mouse"
    assert montage._autoscroll.isActive()
    _key(montage, qapp, Qt.Key.Key_Escape)
    assert not montage._autoscroll.isActive()


# --- the orientation switch (As recorded | Aligned) --------------------------------------- #
def test_aligned_is_greyed_until_the_page_says_the_run_can_align(montage, qapp):
    from detail.gui.live_montage import ALIGN_CHECKING

    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(2))
    aligned = montage.show_button.orientation_actions[True]
    recorded = montage.show_button.orientation_actions[False]
    assert not aligned.isEnabled()
    assert aligned.toolTip() == ALIGN_CHECKING
    montage.set_alignment("Embedding run emb_02 used no keypoints, so …")
    assert not aligned.isEnabled()
    assert aligned.toolTip().startswith("Embedding run emb_02")
    assert recorded.isChecked()
    montage.set_alignment(None)
    assert aligned.isEnabled()
    aligned.trigger()  # choosing "Aligned" from Show ▾
    assert montage.is_aligned() and montage.look().aligned
    assert aligned.isChecked()
    assert _drain(qapp, lambda: source.looks and source.looks[-1].aligned)


def test_the_orientation_is_kept_per_page_and_drawn_only_while_the_run_can(qapp):
    first = _montage(aligned_key="interpret/test_aligned_a", look_key=None)
    first.set_alignment(None)
    first.set_aligned(True)
    again = _montage(aligned_key="interpret/test_aligned_a", look_key=None)
    other = _montage(aligned_key="interpret/test_aligned_b", look_key=None)
    again.set_alignment(None)
    other.set_alignment(None)
    assert again.is_aligned(), "remembered under the page's key"
    assert not other.is_aligned(), "and no other page's"
    again.set_alignment("no heading definition fits")
    assert not again.is_aligned()
    assert again.show_button.orientation_actions[False].isChecked()
    again.set_alignment(None)
    assert again.is_aligned(), "the choice outlives a run that could not align"
    for widget in (first, again, other):
        widget._worker.stop()
        widget.deleteLater()


def test_the_look_reaches_the_source_and_a_new_look_redraws(montage, qapp):
    from detail.api.clip_tiles import TileLook

    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(2))
    assert _drain(qapp, lambda: len(source.looks) >= 2)
    montage.set_points(None)  # the run has keypoints to draw
    montage.set_look(TileLook(footage=False, masks=True, keypoints=True))
    assert _drain(qapp, lambda: source.looks[-1].keypoints)
    assert not source.looks[-1].footage


# --- a click layer: dropped, numbered, and what a click toggles ---------------------------- #
def _layer(off=(), numbers=None, seen=None):
    seen = seen if seen is not None else []
    return lm.ClickLayer(
        hint=lambda i: "Click: toggle",
        is_off=lambda i: i in off,
        clicked=lambda i: seen.append(("click", i)),
        run=lambda indices, back: seen.append(("run", list(indices), back)),
        dropped=lambda i: i in off,
        number=lambda i: (numbers or {}).get(i),
    )


def test_a_dropped_clip_is_washed_out_and_a_pick_wears_its_number(montage, qapp):
    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(3))
    montage.set_click_layer(_layer(off={0}, numbers={1: 2}))
    assert _drain(qapp, lambda: len(montage._images) == 3)
    image = montage.canvas.grab().toImage()
    inside = montage.pick_rect(1, 2).center()
    ink = image.pixelColor(inside.x(), inside.y())
    plain = montage.tile_rect(2)
    clear = image.pixelColor(plain.x() + 8, plain.y() + 8)
    assert ink != clear, "the number's box is painted on the picked tile"
    washed = image.pixelColor(montage.tile_rect(0).center())
    unwashed = image.pixelColor(montage.tile_rect(2).center())
    assert washed != unwashed


def test_a_click_layer_takes_the_clicks_and_gives_them_back(montage, qapp):
    """While a layer is set, it decides what is "off" for a run and receives the click and the
    run; the page's own receiver hears nothing. Taken off, the page's click is back unchanged.
    """
    from dataclasses import replace

    montage.set_source(FakeSource())
    heard = []
    montage.set_click_hint(lambda i: "Click: remove")
    montage.clip_clicked.connect(lambda i, _s: heard.append(("page", i)))
    montage.run_marked.connect(lambda ix, back: heard.append(("page run", ix, back)))
    clips = [replace(c, faded=(i == 1)) for i, c in enumerate(_clips(3))]
    montage.set_clips(clips)
    assert montage.run_between(0, 2, True) == [1], "faded: the page's state"
    seen = []
    montage.set_click_layer(_layer(off={0}, seen=seen))
    assert montage.is_off(0) and not montage.is_off(1)
    assert montage.run_between(0, 2, True) == [0]
    assert montage.click_hint(2) == "Click: toggle"
    montage._clicked(2, Qt.KeyboardModifier.NoModifier)
    montage._clicked(1, Qt.KeyboardModifier.ShiftModifier)
    assert seen == [("click", 2), ("run", [1, 2], False)] and heard == []
    montage.set_click_layer(None)
    assert montage.click_hint(2) == "Click: remove"
    assert montage.run_between(0, 2, True) == [1]
    montage._clicked(2, Qt.KeyboardModifier.NoModifier)
    assert heard == [("page", 2)]


def test_a_click_layer_draws_no_remove_cue(montage, qapp):
    """The ✕ cue on the hovered tile says "a click removes": under a layer it would lie."""
    painted = []
    montage._paint_cross = lambda painter, index, *, cue: painted.append(cue)
    montage.set_source(FakeSource())
    montage.set_click_hint(lambda i: "Click: remove")
    montage.clip_clicked.connect(lambda *_a: None)
    montage.set_clips(_clips(2))
    montage._set_hovered(0)
    montage.canvas.grab()
    assert True in painted, "the page's click shows its cue"
    painted.clear()
    montage.set_click_layer(_layer())
    montage._set_hovered(0)
    montage.canvas.grab()
    assert True not in painted


def test_show_menu_draws_every_layer_and_keeps_them_for_the_page(montage, qapp):
    """Show ▾ is the cluster window's Show row on the page's own montage: each layer reaches the
    source's look, the last one cannot go, keypoints need the run's, and the page keeps it.
    """
    from detail.gui.show_menu import LAST_LAYER, TRAIL_NEEDS_MASKS

    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(2))
    montage._clock.stop()
    actions = montage.show_button.layer_actions
    assert [a.text() for a in actions.values()] == [
        "Video",
        "Masks",
        "Mask trail",
        "Keypoints",
    ]
    montage.set_points("Embedding run emb_02 read no keypoints.")
    assert not actions["keypoints"].isEnabled()
    assert actions["keypoints"].toolTip().startswith("Embedding run emb_02")
    montage.set_points(None)
    actions["keypoints"].trigger()
    actions["footage"].trigger()
    assert _drain(
        qapp,
        lambda: source.looks
        and source.looks[-1].keypoints
        and not source.looks[-1].footage,
    ), "masks on white with keypoints"
    actions["masks"].trigger()
    assert not actions["mask_trail"].isEnabled()
    assert actions["mask_trail"].toolTip() == TRAIL_NEEDS_MASKS
    assert not actions["keypoints"].isEnabled(), "the last layer left"
    assert actions["keypoints"].toolTip() == LAST_LAYER
    again = _montage(
        aligned_key="interpret/test_aligned", look_key="interpret/test_look"
    )
    try:
        again.set_points(None)
        look = again.look()
        assert look.keypoints and not look.footage and not look.masks
    finally:
        again._worker.stop()
        again.deleteLater()


def test_a_page_with_no_look_of_its_own_reads_the_old_show_masks_once(qapp):
    from PyQt6.QtCore import QSettings

    from detail.gui.live_montage import MASKS_KEY

    QSettings().remove("interpret/test_look_fresh")
    QSettings().setValue(MASKS_KEY, "0")
    fresh = _montage(
        aligned_key="interpret/test_aligned", look_key="interpret/test_look_fresh"
    )
    try:
        assert not fresh.masks() and not fresh.look().mask_trail
        assert fresh.look().footage
    finally:
        fresh._worker.stop()
        fresh.deleteLater()
        QSettings().remove(MASKS_KEY)


# -- which picture next: the tiles in view first (the montage-priority change) -------------- #
#
# The class: a request queue that serves in the order requests were *made* rather than the order
# they are *wanted*. Before this, the worker took the oldest-posted tile first and never dropped a
# request, so a fast scroll through a large cluster left hundreds of tiles already scrolled
# past to be drawn before the ones in view. Each test gates a
# slow source on its first call, moves the view while that call is held, then lets go and reads
# the order the source was called in.


class GatedSource(FakeSource):
    """Holds the first call until :attr:`gate` is set, so requests pile up behind it."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.holding = threading.Event()

    def frame(self, clip_index, offset_ns, tile_px, matching, *, look):
        if not self.holding.is_set():
            self.holding.set()
            self.gate.wait(5)
        return super().frame(clip_index, offset_ns, tile_px, matching, look=look)


def _held(montage, qapp, n=400) -> GatedSource:
    """A montage of ``n`` clips at the top, clock stopped, its first call held by the source."""
    source = GatedSource()
    montage.set_source(source)
    montage.set_clips(_clips(n))
    montage._clock.stop()
    assert source.holding.wait(5)
    _settle(qapp)
    return source


def _settle(qapp) -> None:
    """Let the grid lay out: its scroll bar appearing narrows the view by a column."""
    for _ in range(5):
        qapp.processEvents()


def test_after_a_jump_the_next_pictures_drawn_are_the_tiles_in_view(montage, qapp):
    source = _held(montage, qapp)
    montage.grid.verticalScrollBar().setValue(montage.tile_rect(200).top())
    view = list(montage.visible_indices())
    assert 200 in view and len(view) > 3
    source.gate.set()
    assert _drain(qapp, lambda: all(i in montage._images for i in view))
    served = [
        call[0] for call in source.calls[1:]
    ]  # [0] was in hand when the view moved
    assert served[: len(view)] == view


def test_every_tile_in_view_gets_its_first_picture_before_any_tile_a_second(
    montage, qapp
):
    montage.resize(700, 640)  # four rows: the last drawn one, the lookahead, two new
    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(400))
    montage._clock.stop()
    _settle(qapp)
    top = list(montage.visible_indices())
    ahead = montage.lookahead_indices()
    assert _drain(qapp, lambda: all(i in montage._images for i in top + ahead))
    gated = GatedSource()
    montage._source = (
        gated  # same tiles and pictures; only the calls from here are recorded
    )
    montage.advance(SECOND // 15)
    montage._request_visible()  # every drawn tile asks for its next frame; one is held
    assert gated.holding.wait(5)
    # the view starts on the last row drawn and still asking for its next frame; under it the
    # drawn lookahead row, then rows nobody has asked for
    montage.grid.verticalScrollBar().setValue(montage.tile_rect(top[-1]).top())
    view = list(montage.visible_indices())
    new = [i for i in view if i not in montage._images]
    assert new and len(new) < len(view)
    montage.advance(SECOND // 15)
    montage._request_visible()
    gated.gate.set()
    assert _drain(qapp, lambda: all(i in montage._images for i in new))
    served = [call[0] for call in gated.calls[1:]]
    assert served[: len(new)] == new


def test_a_tile_scrolled_out_and_back_before_its_picture_still_gets_one(montage, qapp):
    source = _held(montage, qapp)
    top = list(montage.visible_indices())
    bar = montage.grid.verticalScrollBar()
    bar.setValue(montage.tile_rect(200).top())
    bar.setValue(0)
    source.gate.set()
    assert _drain(qapp, lambda: all(i in montage._images for i in top))


def test_requests_for_tiles_scrolled_past_are_dropped(montage, qapp):
    source = _held(montage, qapp)
    bar = montage.grid.verticalScrollBar()
    passed: set[int] = set()
    for target in (60, 120, 180, 240):
        bar.setValue(montage.tile_rect(target).top())
        passed |= set(montage.visible_indices())
    view = set(montage.visible_indices())
    ahead = set(montage.lookahead_indices())
    source.gate.set()
    assert _drain(qapp, lambda: all(i in montage._images for i in view | ahead))
    qapp.processEvents()
    served = {call[0] for call in source.calls[1:]}
    assert served <= view | ahead
    assert not (passed - view - ahead) & served


def test_the_lookahead_is_one_row_in_the_direction_of_the_scroll(montage, qapp):
    source = FakeSource()
    montage.set_source(source)
    montage.set_clips(_clips(400))
    montage._clock.stop()
    _settle(qapp)
    cols = montage._columns()
    bar = montage.grid.verticalScrollBar()
    bar.setValue(montage.tile_rect(200).top())
    view = montage.visible_indices()
    assert montage.lookahead_indices() == list(range(view.stop, view.stop + cols))
    below = set(montage.lookahead_indices())
    assert _drain(qapp, lambda: below <= set(montage._images))
    bar.setValue(montage.tile_rect(100).top())
    view = montage.visible_indices()
    assert montage.lookahead_indices() == list(
        range(view.start - 1, view.start - cols - 1, -1)
    )
    above = set(montage.lookahead_indices())
    assert _drain(qapp, lambda: above <= set(montage._images))
    asked = {call[0] for call in source.calls}
    assert view.start - cols - 1 not in asked


def test_a_tile_asked_again_while_its_first_picture_is_drawn_waits_its_turn(
    montage, qapp
):
    source = _held(montage, qapp)  # tile 0's first picture is in hand
    view = list(montage.visible_indices())
    assert view[0] == 0 and len(view) > 3
    montage.advance(SECOND // 15)
    montage._request_visible()  # tile 0 asks for its next frame before its first arrives
    source.gate.set()
    assert _drain(qapp, lambda: all(i in montage._images for i in view))
    served = [call[0] for call in source.calls[1:]]
    assert served[: len(view) - 1] == view[1:]


# --- what a montage costs while nobody can see it, and after it is gone (review 2026-11-09) -- #
def test_a_minimised_window_stops_the_clock_and_restoring_starts_it(montage, qapp):
    """A minimised window's montage is still ``isVisible()`` to Qt, so only ``hideEvent``
    stopped the clock and every open cluster window kept decoding at 15 fps for nobody.
    """
    montage.set_clips(_clips(6))
    assert montage._clock.isActive() and montage.is_on_screen()
    montage.window().showMinimized()
    assert _drain(qapp, lambda: not montage._clock.isActive())
    assert not montage.is_on_screen()
    montage.window().showNormal()
    assert _drain(qapp, montage._clock.isActive)


class ClosingSource(FakeSource):
    def __init__(self) -> None:
        super().__init__()
        self.closed_on: list[int] = []

    def close(self) -> None:
        self.closed_on.append(threading.get_ident())


def test_a_montage_destroyed_closes_the_source_it_was_using(qapp):
    """``destroyed → worker.stop`` dropped the queue but never closed the source in use, so a
    closed cluster window's open clips and drawn tiles waited on the garbage collector.
    """
    widget = _montage(
        aligned_key="interpret/test_aligned", look_key="interpret/test_look"
    )
    widget.resize(300, 200)
    widget.show()
    first, second = ClosingSource(), ClosingSource()
    widget.set_source(first)
    widget.set_clips(_clips(4))
    widget.set_source(second)
    assert _drain(qapp, lambda: first.closed_on), "a replaced source is closed"
    from PyQt6.QtCore import QCoreApplication, QEvent

    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert _drain(qapp, lambda: second.closed_on), "the one in use, on destroy"
    assert threading.get_ident() not in second.closed_on, "on the worker, never the UI"


def test_a_request_posted_as_the_worker_goes_idle_is_still_served(qapp, monkeypatch):
    """The worker exits after :data:`_IDLE_S` idle. A request posted after it decided to exit
    and before its thread finished found ``isRunning()`` true, started nothing, and sat until
    the next request. The window is held open here by slowing the thread's last step."""
    from detail.api.clip_tiles import TileLook
    from tests.waits import finished, until

    at_exit = threading.Event()

    class SlowExit(lm._PictureWorker):
        def run(self) -> None:
            super().run()
            at_exit.set()
            time.sleep(0.3)

    monkeypatch.setattr(lm, "_IDLE_S", 0.05)
    worker = SlowExit(receiver=None)
    served: list[int] = []
    worker.delivered.connect(lambda _g, index, _o, _image: served.append(index))
    source = FakeSource()
    worker.post(0, (source, 0, 0, 16, True, TileLook()))
    assert at_exit.wait(30), "the worker never went idle"
    worker.post(1, (source, 0, 0, 16, True, TileLook()))
    until(qapp, lambda: 1 in served, "the request posted as the worker went idle")
    worker.stop()
    finished(worker, "the stopped picture worker")
