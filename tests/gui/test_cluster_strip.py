"""The member strip and the ethograms under a cluster (§22).

These are the parts of the interpretation screen a person actually looks at while deciding whether
a cluster is one thing, and every check here is about what is on it:

- **The cell is the picture.** The old cell was a framed panel with margins and a caption, so a
  16:9 crop sat in a grey surround — chrome around the data rather than the data.
- **Every member, not the first eight.** Eight is the right number to lead with and the wrong
  number to judge a cluster on; the strip scrolls through all of them.
- **The mask is tinted in the cluster's colour**, so a thumbnail, its point in the scatter and its
  band in the ethogram are recognisably one thing.
- **Clicking plays in place**, with one player moved into the clicked cell rather than ninety
  players each owning a decoder.
- **The key lists every cluster and follows the names**, because an unnamed cluster is work
  outstanding and a key that skipped it would leave the ethogram's commonest colours unidentified.
"""

import numpy as np
import pytest

from detail.gui.behavior_query import ClipSpec
from detail.gui.cluster_strip import (
    ClusterClipStrip,
    EthogramStrip,
    _Thumb,
)
from detail.gui.cluster_strip import cluster_colors
from detail.gui.qt import QFrame
from detail.gui.sizing import lines


def _at(strip, fraction: float, row: int = 0, *, lane: bool = False):
    """The point ``fraction`` along ``row``'s bar (or lane), read off the strip's own geometry
    rather than a pixel copied from it — a test that hard-codes the layout tests the layout.
    """
    from detail.gui.qt import QPointF

    if lane:
        y = strip._lane_top(row) + strip.LANE_PX / 2
    else:
        y = strip._bar_top(row) + strip.BAR_PX / 2
    return QPointF(float(strip._x(fraction, row)), float(y))


def _mouse(strip, kind, point, buttons=None):
    from detail.gui.qt import QEvent, QMouseEvent, Qt

    left = Qt.MouseButton.LeftButton
    event = QMouseEvent(
        getattr(QEvent.Type, kind),
        point,
        left,
        left if buttons is None else buttons,
        Qt.KeyboardModifier.NoModifier,
    )
    handler = {
        "MouseButtonPress": strip.mousePressEvent,
        "MouseButtonRelease": strip.mouseReleaseEvent,
        "MouseMove": strip.mouseMoveEvent,
        "MouseButtonDblClick": strip.mouseDoubleClickEvent,
    }[kind]
    handler(event)


def _click(strip, point):
    """A real press and release: a click is decided on release, which is what tells it from a
    drag that pans."""
    _mouse(strip, "MouseButtonPress", point)
    _mouse(strip, "MouseButtonRelease", point)


def _clips(n: int) -> list:
    return [
        ClipSpec("recording_001", 0, "emb_01", i * 1_000_000, (i + 1) * 1_000_000, "c")
        for i in range(n)
    ]


@pytest.fixture
def experiment(tmp_path):
    from detail.api.experiment import Experiment

    return Experiment.create("p", tmp_path)


# --- the cell is the picture ------------------------------------------------------------------


def test_a_thumbnail_has_no_frame_and_no_caption(qapp):
    """What "a rectangular frame around them with empty space" was describing: the cell was a
    ``QFrame`` with a styled panel, 2px margins and a caption row under a square image box.
    """
    thumb = _Thumb(120)
    # `QLabel` *is* a `QFrame` subclass, so the type says nothing — the shape is what draws a
    # border, and a layout is what puts margins around the picture.
    assert thumb.frameShape() == QFrame.Shape.NoFrame
    assert thumb.layout() is None, "a layout means margins around the picture"
    assert thumb.size().width() == 120 and thumb.size().height() == 120


def test_a_thumbnail_says_why_it_is_blank(qapp):
    """Named, not blank: which failure it is decides what the user does about it."""
    from detail.gui import frame_source

    thumb = _Thumb(120)
    thumb.set_image(None, frame_source.UNDECODABLE)
    assert thumb.text()
    assert thumb.pixmap().isNull()


# --- every member ------------------------------------------------------------------------------


def test_the_strip_holds_every_member_not_a_fixed_number(qapp, experiment):
    """Eight is the right number to lead with and the wrong number to judge a cluster on."""
    strip = ClusterClipStrip(experiment, size_px=64)
    strip.show_clips(_clips(37))
    assert len(strip._thumbs) == 37


def test_the_strip_shrinks_back_for_a_smaller_cluster(qapp, experiment):
    """Cells are reused rather than rebuilt, so the ones a smaller cluster does not need must be
    hidden — otherwise the strip keeps the previous cluster's tail on screen."""
    strip = ClusterClipStrip(experiment, size_px=64)
    strip.show_clips(_clips(20))
    strip.show_clips(_clips(3))
    strip.show()
    qapp.processEvents()
    assert len(strip._thumbs) == 20
    assert sum(1 for t in strip._thumbs if not t.isHidden()) == 3


def test_the_strip_scrolls_horizontally_and_stays_one_row(qapp, experiment):
    """A row, not a pane: the members read left to right and the page keeps its height."""
    from detail.gui.qt import Qt

    strip = ClusterClipStrip(experiment, size_px=64)
    assert (
        strip._scroll.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        strip._scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert strip._scroll.height() < 64 * 2


def test_an_empty_cluster_says_so(qapp, experiment):
    """A row of blank cells reads as frames that failed to load rather than nothing chosen."""
    strip = ClusterClipStrip(experiment, size_px=64)
    strip.show_clips([], empty_text="nothing here")
    strip.show()
    qapp.processEvents()
    assert not strip._empty.isHidden()
    assert strip._empty.text() == "nothing here"


# --- the tint ties the surfaces together --------------------------------------------------------


def test_the_tint_reaches_the_frame_producer(qapp, experiment):
    """The mask is drawn in the *cluster's* colour, which is what makes a thumbnail, a point in
    the scatter and a band in the ethogram read as one thing.

    Checked at the producer, because the alternative — asserting on decoded pixels — needs a real
    video and would pass on a tint of the wrong colour anyway.
    """
    strip = ClusterClipStrip(experiment, size_px=64)
    strip.show_clips(_clips(1), tint=(10, 20, 30))
    assert strip._tint == (10, 20, 30)

    seen = {}
    strip._preview.source_crop_masked = lambda *a, **k: seen.update(k) or None
    strip._preview.t_ns_to_frame = lambda *a, **k: 0
    strip._producer_for(_clips(1)[0])()
    assert seen["tint"] == (10, 20, 30)


def test_no_tint_means_no_mask_decode(qapp, experiment):
    """Masks are the expensive half of a frame read, so a strip that wants none must not pay."""
    strip = ClusterClipStrip(experiment, size_px=64)
    strip.show_clips(_clips(1), tint=None)
    seen = {}
    strip._preview.source_crop_masked = lambda *a, **k: seen.update(k) or None
    strip._preview.t_ns_to_frame = lambda *a, **k: 0
    strip._producer_for(_clips(1)[0])()
    assert seen["tint"] is None


# --- play in place -------------------------------------------------------------------------------


def test_clicking_a_cell_plays_in_that_cell(qapp, experiment):
    """In place, not in a side pane — and one playing cell for the strip, because ninety members
    cannot each own a decoder and a timer."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(5)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[2])
    assert strip._playing == 2
    assert strip._thumbs[2].isHidden(), "the picture should give its slot to the player"
    assert strip._row_layout.indexOf(strip._cell) == 2


def test_clicking_the_playing_cell_again_stops_it(qapp, experiment):
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(5)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[2])
    strip._on_clicked(clips[2])
    assert strip._playing is None
    assert not strip._thumbs[2].isHidden()


def test_playing_moves_rather_than_multiplying(qapp, experiment):
    """Clicking a second cell moves the one player, so two clips never play at once."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(5)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    strip._on_clicked(clips[3])
    assert strip._playing == 3
    assert not strip._thumbs[1].isHidden()
    assert strip._thumbs[3].isHidden()


def test_showing_another_cluster_stops_playback(qapp, experiment):
    """Otherwise the player keeps running a clip from a cluster nobody is looking at."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(5)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()
    strip._on_clicked(clips[1])

    strip.show_clips(_clips(4))
    assert strip._playing is None


# --- the ethograms ---------------------------------------------------------------------------------


def test_the_ethogram_draws_a_row_per_recording(qapp):
    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 0.2, 0), (0.5, 0.7, 1)]), ("rec_b", [(0.1, 0.4, 1)])],
        cluster_colors(4),
        selected=1,
        lengths_ns=[1_000_000_000]
        * len([("rec_a", [(0.0, 0.2, 0), (0.5, 0.7, 1)]), ("rec_b", [(0.1, 0.4, 1)])]),
    )
    strip.resize(400, strip.height())
    strip.show()
    qapp.processEvents()
    assert len(strip._rows) == 2
    assert strip.height() >= 2 * strip.BAR_PX


def test_the_selected_cluster_is_solid_and_the_rest_dimmed(qapp):
    """The row's question is where *this* cluster sits among the others, which needs the others
    visible but not competing."""
    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 0.2, 0)])],
        cluster_colors(4),
        selected=1,
        lengths_ns=[1_000_000_000] * len([("rec_a", [(0.0, 0.2, 0)])]),
    )
    assert strip._color_for(1).alpha() == 255
    assert strip._color_for(0).alpha() < 255


def test_no_selection_leaves_every_cluster_solid(qapp):
    """The negative case: dimming everything would make the row unreadable before a cluster is
    picked."""
    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 0.2, 0)])],
        cluster_colors(4),
        selected=None,
        lengths_ns=[1_000_000_000] * len([("rec_a", [(0.0, 0.2, 0)])]),
    )
    assert strip._color_for(0).alpha() == 255


def test_an_empty_row_still_draws(qapp):
    """A recording where this interpretation found nothing is a result, and its row says so by
    being empty rather than by being absent."""
    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [])],
        cluster_colors(2),
        selected=0,
        lengths_ns=[1_000_000_000] * len([("rec_a", [])]),
    )
    strip.resize(200, strip.height())
    strip.show()
    qapp.processEvents()
    assert len(strip._rows) == 1


# --- the row scales ------------------------------------------------------------------------------


def test_dragging_the_handle_grows_the_pictures(qapp, experiment):
    """Drag down for bigger pictures and fewer of them, which is the trade a person makes when a
    behavior stops being distinguishable at thumbnail size."""
    strip = ClusterClipStrip(experiment, size_px=100)
    strip.show_clips(_clips(3))
    before = strip._thumbs[0].height()

    strip.resize_by(60)
    assert strip._size_px == 160
    assert strip._thumbs[0].height() == 160 > before
    assert strip._scroll.height() > before


def test_dragging_up_shrinks_them(qapp, experiment):
    strip = ClusterClipStrip(experiment, size_px=200)
    strip.show_clips(_clips(3))
    strip.resize_by(-60)
    assert strip._size_px == 140
    assert strip._thumbs[0].height() == 140


def test_the_size_is_bounded_at_both_ends(qapp, experiment):
    """A row dragged to nothing has no pictures in it, and one dragged past the window pushes the
    ethograms off the screen."""
    from detail.gui.cluster_strip import MAX_THUMB_PX, MIN_THUMB_PX

    strip = ClusterClipStrip(experiment, size_px=100)
    strip.resize_by(-10_000)
    assert strip._size_px == MIN_THUMB_PX
    strip.resize_by(10_000)
    assert strip._size_px == MAX_THUMB_PX


def test_resizing_redraws_rather_than_re_decoding(qapp, experiment):
    """Frames are kept as decoded arrays, so growing the row costs no video seeks — the whole
    reason a drag can be smooth."""
    strip = ClusterClipStrip(experiment, size_px=100)
    strip.show_clips(_clips(2))
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    strip._thumbs[0].set_image(rgb, None)

    decodes = []
    strip._preview.source_crop_masked = lambda *a, **k: decodes.append(1) or None
    strip.resize_by(50)
    assert decodes == []
    assert strip._thumbs[0]._rgb is rgb


# --- playback is the same crop --------------------------------------------------------------------


def test_playback_decodes_the_same_crop_with_the_same_tint(qapp, experiment):
    """Clicking swaps a still picture for a moving one and changes nothing else. A player showing
    the recording's whole frame would answer a different question than the strip is for.
    """
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(2)
    strip.show_clips(clips, tint=(9, 8, 7))

    seen = []
    strip._preview.t_ns_to_frame = lambda *a, **k: 0
    strip._preview.frame_rate = lambda *a, **k: 30.0
    strip._preview.source_crops_masked = lambda *a, **k: seen.append(k) or iter(())
    strip._clip_producer(clips[0])()
    assert seen and all(call["tint"] == (9, 8, 7) for call in seen)


def test_a_long_clip_is_strided_not_truncated(qapp, experiment):
    """Playing a slowed-down whole is closer to the behavior than playing its first half at
    speed — and the interval carries the stride, so the bout still takes its real duration.
    """
    from detail.gui.cluster_strip import MAX_PLAY_FRAMES

    strip = ClusterClipStrip(experiment, size_px=64)
    clip = _clips(1)[0]
    frames = {"n": 0}

    def fake_crops(_r, _e, _t, first, last, _size, *, tint, step=1, **k):
        # The span is asked for once and strided by the reader, which is the point: the old
        # shape called a single-frame decode per index and re-sought the file every time.
        frames["n"] += 1
        for _ in range(first, last + 1, step):
            yield np.zeros((8, 8, 3), dtype=np.uint8)

    strip._preview.t_ns_to_frame = lambda _r, _e, t: 0 if t == clip.start_t_ns else 1000
    strip._preview.frame_rate = lambda *a, **k: 50.0
    strip._preview.source_crops_masked = fake_crops

    decoded, interval = strip._clip_producer(clip)()
    assert len(decoded) <= MAX_PLAY_FRAMES
    assert frames["n"] == 1
    stride = -(-1001 // MAX_PLAY_FRAMES)
    assert interval == round(1000.0 * stride / 50.0)


def test_a_stale_decode_never_reaches_the_cell(qapp, experiment):
    """Clicking a second member while the first is decoding must not have the first arrive on
    top of it — the same generation guard the thumbnails use."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[0])
    stale = strip._play_generation
    strip._on_clicked(clips[1])
    strip._on_clip_decoded(stale, ([np.zeros((8, 8, 3), dtype=np.uint8)], 40))
    assert strip._cell._frames == []


def test_the_playing_cell_has_a_play_button_and_a_scrub_bar_and_nothing_else(qapp):
    """Two controls, deliberately. The cell is a member of a set being compared against its
    neighbours, and every control added to it is width taken from the comparison."""
    from detail.gui.qt import QPushButton, QSlider

    from detail.gui.quick_preview import QuickPreview

    cell = QuickPreview(80)
    assert len(cell.findChildren(QPushButton)) == 1
    assert len(cell.findChildren(QSlider)) == 1


def test_scrubbing_pauses_so_the_chosen_frame_stays(qapp):
    """Otherwise the next tick replaces the frame the user just picked."""
    from detail.gui.quick_preview import QuickPreview

    cell = QuickPreview(80)
    cell.set_frames([np.full((8, 8, 3), i, dtype=np.uint8) for i in range(5)], 40)
    assert cell._play.isChecked()

    cell._scrub.setValue(3)
    assert not cell._play.isChecked()
    assert cell._index == 3


def test_playback_stops_at_the_end_rather_than_looping(qapp):
    """A clip that restarts on its own keeps pulling the eye back to one member while you are
    trying to look across the row."""
    from detail.gui.quick_preview import QuickPreview

    cell = QuickPreview(80)
    cell.set_frames([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)], 40)
    cell._index = 2
    cell._advance()
    assert cell._index == 2
    assert not cell._play.isChecked()
    assert not cell._timer.isActive()


def test_pressing_play_at_the_end_starts_it_over(qapp):
    """Otherwise the button sits on the final frame doing nothing, which reads as broken."""
    from detail.gui.quick_preview import QuickPreview

    cell = QuickPreview(80)
    cell.set_frames([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)], 40)
    cell._play.setChecked(False)
    cell._index = 2
    cell._play.setChecked(True)
    assert cell._index == 0
    assert cell._timer.isActive()


def test_a_decoding_clip_shows_the_thumbnail_it_replaced(qapp, experiment):
    """Decoding a clip is a second of video walked frame by frame, and there is no reason to
    stare at an empty square while it happens — the thumbnail already holds that frame.
    """
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    rgb = np.full((8, 8, 3), 7, dtype=np.uint8)
    strip._thumbs[1].set_image(rgb, None)
    strip._on_clicked(clips[1])
    assert strip._cell._still is rgb
    assert not strip._cell._image.pixmap().isNull()
    assert not strip._cell._image.text()
    # No controls until there is something to control.
    assert not strip._cell._play.isEnabled()


def test_the_controls_come_back_when_the_clip_lands(qapp, experiment):
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    strip._on_clip_decoded(
        strip._play_generation,
        ([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)], 40),
    )
    assert strip._cell._play.isEnabled()
    assert strip._cell._scrub.isEnabled()
    assert strip._cell._still is None


def test_a_clip_that_cannot_be_decoded_says_so(qapp):
    from detail.gui.quick_preview import QuickPreview

    cell = QuickPreview(80)
    cell.set_frames([], 40)
    assert "decode" in cell._image.text()
    assert not cell._play.isChecked()


def test_waiting_for_a_decode_is_not_reported_as_a_failed_one(qapp):
    """Clearing the cell by handing it an empty frame list made it announce "could not decode
    this clip" the instant it was clicked, every time, for as long as the decode took — which is
    what a user sees and reports as a bug. A placeholder must not claim a failure that has not
    happened."""
    from detail.gui.quick_preview import QuickPreview

    cell = QuickPreview(80)
    cell.set_loading()
    assert "decode" not in cell._image.text()
    assert cell._frames == []


def test_clicking_shows_the_waiting_state_not_the_failure(qapp, experiment):
    """The path the user actually takes, since the two states differ only in how they are
    reached."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    assert "decode" not in strip._cell._image.text()


def test_a_decode_that_really_comes_back_empty_still_says_so(qapp, experiment):
    """The negative case: the distinction is only worth having if a genuine failure is still
    reported."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    strip._on_clip_decoded(strip._play_generation, ([], 40))
    assert "decode" in strip._cell._image.text()


# --- the ethogram is a selector --------------------------------------------------------------------


def test_clicking_a_band_selects_its_cluster(qapp):
    """The same destination a scatter click takes: both are pictures of the vocabulary, and the
    thing a person does on seeing an interesting stretch is point at it."""
    from detail.gui.qt import QEvent, QMouseEvent, QPointF, Qt

    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 0.25, 3), (0.5, 0.9, 7)])],
        cluster_colors(8),
        None,
        lengths_ns=[1_000_000_000] * len([("rec_a", [(0.0, 0.25, 3), (0.5, 0.9, 7)])]),
    )
    strip.resize(400, strip.height())

    picked = []
    strip.cluster_picked.connect(picked.append)
    # A real event, so the handler and the hit-test are covered together rather than only the
    # half a stub can reach. 70% across, inside the first row's bar.
    _click(strip, _at(strip, 0.7))
    assert picked == [7]


def test_the_hit_test_finds_the_band_under_the_point(qapp):
    from detail.gui.qt import QPointF

    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 0.25, 3), (0.5, 0.9, 7)])],
        cluster_colors(8),
        None,
        lengths_ns=[1_000_000_000] * len([("rec_a", [(0.0, 0.25, 3), (0.5, 0.9, 7)])]),
    )
    strip.resize(400, strip.height())
    assert strip._cluster_at(_at(strip, 0.1)) == 3
    assert strip._cluster_at(_at(strip, 0.7)) == 7


def test_clicking_blank_time_selects_nothing(qapp):
    """The gaps are where the subject was not observed or the vocabulary accounts for nothing;
    snapping to a neighbour would claim a band covers time it does not."""
    from detail.gui.qt import QPointF

    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 0.25, 3)])],
        cluster_colors(8),
        None,
        lengths_ns=[1_000_000_000] * len([("rec_a", [(0.0, 0.25, 3)])]),
    )
    strip.resize(400, strip.height())
    assert strip._cluster_at(_at(strip, 0.75)) is None


def test_clicking_off_the_bars_selects_nothing(qapp):
    from detail.gui.qt import QPointF

    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 1.0, 3)])],
        cluster_colors(8),
        None,
        lengths_ns=[1_000_000_000] * len([("rec_a", [(0.0, 1.0, 3)])]),
    )
    strip.resize(400, strip.height())
    assert strip._cluster_at(QPointF(200.0, 0.5)) is None  # the gap above the bar
    # The gutter holds the name: a click on it is not a click on time zero.
    bar_y = strip._bar_top(0) + strip.BAR_PX / 2
    assert strip._gutter > 0
    assert strip._cluster_at(QPointF(strip._gutter / 2, bar_y)) is None


def test_the_hit_test_knows_which_row_it_is_in(qapp):
    """Two recordings, two rows: a click in the second must not read the first one's bands."""
    from detail.gui.qt import QPointF

    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 1.0, 1)]), ("rec_b", [(0.0, 1.0, 5)])],
        cluster_colors(8),
        None,
        lengths_ns=[1_000_000_000]
        * len([("rec_a", [(0.0, 1.0, 1)]), ("rec_b", [(0.0, 1.0, 5)])]),
    )
    strip.resize(400, strip.height())
    assert strip._cluster_at(_at(strip, 0.5, 0)) == 1
    assert strip._cluster_at(_at(strip, 0.5, 1)) == 5


def test_a_decoding_clip_shows_the_busy_indicator(qapp, experiment):
    """A decode is a second or two of walking video, and the long-op rule says a wait that long
    has to show itself — a greyed play button and a still picture are indistinguishable from a
    hang. `BusyOverlay` is the same indicator every other slow GUI load uses."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    busy = strip._cell._busy
    # Armed rather than shown: the overlay waits `SHOW_DELAY_MS` so a quick decode does not
    # flash. Forcing the reveal is what a slow one does.
    assert busy._delay.isActive()
    busy._reveal()
    assert not busy.isHidden()
    assert "Decoding" in busy._label.text()
    # Indeterminate — a clip decode has no fraction to report.
    assert busy._bar.minimum() == busy._bar.maximum() == 0


def test_the_busy_indicator_clears_when_the_clip_lands(qapp, experiment):
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    strip._cell._busy._reveal()
    strip._on_clip_decoded(
        strip._play_generation,
        ([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)], 40),
    )
    assert strip._cell._busy.isHidden()
    assert not strip._cell._busy._delay.isActive()


def test_the_busy_indicator_clears_when_the_clip_cannot_be_decoded(qapp, experiment):
    """Otherwise a failed decode leaves a spinner running over the failure message."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    strip._on_clip_decoded(strip._play_generation, ([], 40))
    assert strip._cell._busy.isHidden()
    assert "decode" in strip._cell._image.text()


def test_the_busy_indicator_clears_when_playback_is_stopped(qapp, experiment):
    """Clicking a second member while the first is still decoding must not leave the overlay up
    on the cell it moved away from."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(3)
    strip.show_clips(clips)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[1])
    strip._on_clicked(clips[2])
    assert strip._playing == 2
    strip._on_clicked(clips[2])
    assert strip._cell._busy.isHidden()


def test_a_playing_cell_is_no_wider_than_the_picture(qapp, experiment):
    """A `QSlider` is Expanding, and that policy propagates up through the cell — so a cluster
    with two members played one of them under a scrub bar four times the width of its own
    picture, stretched across the empty part of the row."""
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(2)
    strip.show_clips(clips)
    strip.resize(900, 200)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[0])
    qapp.processEvents()
    assert strip._cell.width() == 64
    assert strip._cell.maximumWidth() == 64


def test_the_playing_cell_follows_the_dragged_size(qapp, experiment):
    strip = ClusterClipStrip(experiment, size_px=64)
    clips = _clips(2)
    strip.show_clips(clips)
    strip.resize(900, 200)
    strip.show()
    qapp.processEvents()

    strip._on_clicked(clips[0])
    strip.set_thumb_size(180)
    qapp.processEvents()
    assert strip._cell.maximumWidth() == 180
    assert strip._thumbs[1].width() == 180


# --- the selection lane ----------------------------------------------------------------------------


def _laned(qapp):
    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", [(0.0, 0.25, 3)]), ("rec_b", [(0.5, 0.9, 7)])],
        cluster_colors(8),
        None,
        lengths_ns=[1_000_000_000]
        * len([("rec_a", [(0.0, 0.25, 3)]), ("rec_b", [(0.5, 0.9, 7)])]),
    )
    strip.set_lane([[(0.6, 0.7)], [(0.1, 0.2)]])
    strip.resize(400, strip.height())
    return strip


def test_a_lane_grows_each_row_and_clearing_it_restores_the_height(qapp):
    strip = EthogramStrip()
    strip.set_rows(
        [("rec_a", []), ("rec_b", [])],
        cluster_colors(2),
        None,
        lengths_ns=[1_000_000_000] * len([("rec_a", []), ("rec_b", [])]),
    )
    plain = strip.height()
    assert strip.row_px() == strip.BAR_PX + strip.GAP_PX
    strip.set_lane([[], []])
    assert strip.height() == plain + 2 * (strip.LANE_PX + 2)
    strip.set_lane(None)
    assert strip.height() == plain


def test_a_redraw_of_the_rows_keeps_the_lane(qapp):
    """The lane is a selection, a second fact on the picture: relabelling must not drop it."""
    strip = _laned(qapp)
    strip.set_rows(
        [("rec_a", []), ("rec_b", [])],
        cluster_colors(2),
        None,
        lengths_ns=[1_000_000_000] * len([("rec_a", []), ("rec_b", [])]),
    )
    assert strip._lane == [[(0.6, 0.7)], [(0.1, 0.2)]]


def test_bars_and_lane_marks_are_hit_at_the_laned_row_pitch(qapp):
    """Every row's offset comes from one pitch. A copy still stepping by the lane-less pitch would
    put row two's bar where row one's lane is."""
    from detail.gui.qt import QPointF

    strip = _laned(qapp)
    assert strip.row_px() == strip.BAR_PX + strip.GAP_PX + strip.LANE_PX + 2
    # Pixels, not the helpers: a second row at the lane-less pitch would land row two's bar
    # where row one's lane is, and the helpers would move with it.
    assert strip._bar_top(1) == strip.row_px() + strip.GAP_PX // 2
    assert strip._cluster_at(_at(strip, 0.1, 0)) == 3
    assert strip._cluster_at(_at(strip, 0.7, 1)) == 7
    assert strip.band_at(_at(strip, 0.7, 1)) == (1, 0)
    assert strip.lane_at(_at(strip, 0.65, 0, lane=True)) == (0, 0)
    assert strip.lane_at(_at(strip, 0.15, 1, lane=True)) == (1, 0)
    assert strip.lane_at(_at(strip, 0.4, 0, lane=True)) is None
    assert strip.lane_at(_at(strip, 0.65, 0)) is None


def test_clicking_a_lane_mark_says_which(qapp):
    from detail.gui.qt import QEvent, QMouseEvent, QPointF, Qt

    strip = _laned(qapp)
    got = []
    strip.lane_activated.connect(lambda row, index: got.append((row, index)))
    _click(strip, _at(strip, 0.65, 0, lane=True))
    assert got == [(0, 0)]


def test_marking_a_row_outlines_it_for_a_moment(qapp):
    strip = _laned(qapp)
    strip.mark_row(1, for_ms=10)
    assert strip._marked == 1 and strip._unmark.isActive()
    strip._unmark.timeout.emit()
    assert strip._marked is None


def test_a_long_name_is_elided_in_the_gutter_and_whole_on_hover(qapp):
    from detail.gui.qt import QPointF

    long = "subject_03_treatment_25_" * 6
    strip = EthogramStrip()
    strip.set_rows(
        [(long, [(0.0, 1.0, 1)])],
        cluster_colors(2),
        None,
        lengths_ns=[1_000_000_000] * len([(long, [(0.0, 1.0, 1)])]),
    )
    strip.resize(1200, strip.height())
    assert strip._gutter <= lines(strip.GUTTER_LINES) + 8
    assert strip._name_at(QPointF(5.0, strip._bar_top(0) + 4)) == long


# --- one time axis, and the zoom (10/4) ----------------------------------------------------------

MIN = 60_000_000_000


def _two_lengths(qapp):
    """A 30-minute and a 60-minute recording, a band at the same fraction of each."""
    strip = EthogramStrip()
    strip.set_rows(
        [("short", [(0.5, 0.6, 1)]), ("long", [(0.5, 0.6, 2)])],
        cluster_colors(4),
        None,
        lengths_ns=[30 * MIN, 60 * MIN],
    )
    strip.resize(1000, strip.height())
    return strip


def _wheel(strip, x, notches, *, shift=False):
    from detail.gui.qt import QPoint, QPointF, QWheelEvent, Qt

    modifiers = (
        Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    )
    event = QWheelEvent(
        QPointF(x, strip._bar_top(0) + 4),
        QPointF(x, strip._bar_top(0) + 4),
        QPoint(0, 0),
        QPoint(0, int(120 * notches)),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    strip.wheelEvent(event)


def test_rows_share_one_time_axis_and_a_shorter_recording_ends_early(qapp):
    """Each row used to be stretched to its own recording, so one screen position was a
    different minute in every row — the thing a zoom makes visible and a ruler cannot label.
    """
    strip = _two_lengths(qapp)
    left, width = strip.track()
    assert strip._x(1.0, 1) == left + width
    assert abs(strip._x(1.0, 0) - (left + width / 2)) <= 1, "30 min ends half way"
    # Minute 15 of the short recording and minute 15 of the long one are one x.
    assert abs(strip._x(0.5, 0) - strip._x(0.25, 1)) <= 1
    # Past the short recording's end there is no time to point at.
    assert strip.band_at(QPointF_at(strip, left + width * 0.75, 0)) is None


def QPointF_at(strip, x, row):  # noqa: N802 — reads as the constructor it wraps
    from detail.gui.qt import QPointF

    return QPointF(float(x), float(strip._bar_top(row) + strip.BAR_PX / 2))


def test_lengths_are_required_one_per_row(qapp):
    strip = EthogramStrip()
    with pytest.raises(TypeError):
        strip.set_rows([("a", [])], cluster_colors(2), None)  # no lengths at all
    with pytest.raises(ValueError, match="one per row"):
        strip.set_rows([("a", []), ("b", [])], cluster_colors(2), None, lengths_ns=[1])


def test_the_wheel_zooms_around_the_pointer(qapp):
    strip = _two_lengths(qapp)
    left, width = strip.track()
    x = left + width * 0.3
    before = strip._fraction(x, 1)
    _wheel(strip, x, 3)
    assert strip.is_zoomed()
    v0, v1 = strip.view
    assert v1 - v0 < 0.6, "three notches in narrow the window"
    assert (
        abs(strip._fraction(x, 1) - before) < 1e-6
    ), "the time under the pointer stays put"
    _wheel(strip, x, -50)
    assert not strip.is_zoomed(), "zooming out stops at the whole span"


def test_the_zoom_stops_at_two_seconds_across(qapp):
    from detail.gui.cluster_strip import MIN_VIEW_NS

    strip = _two_lengths(qapp)
    strip.zoom_by(1e-9)
    v0, v1 = strip.view
    assert (v1 - v0) * strip.axis_ns == pytest.approx(MIN_VIEW_NS, rel=1e-6)


def test_a_drag_pans_and_does_not_click(qapp):
    strip = _two_lengths(qapp)
    strip.set_view(0.4, 0.6)
    picked = []
    strip.cluster_picked.connect(picked.append)
    start = _at(strip, 0.55, 1)
    _mouse(strip, "MouseButtonPress", start)
    from detail.gui.qt import QPointF, Qt

    _mouse(
        strip,
        "MouseMove",
        QPointF(start.x() - 100, start.y()),
        Qt.MouseButton.LeftButton,
    )
    _mouse(strip, "MouseButtonRelease", QPointF(start.x() - 100, start.y()))
    assert picked == [], "a drag pans; it is not a click on where it ended"
    assert strip.view[0] > 0.4, "dragging left moves the window later"


def test_a_click_on_a_zoomed_strip_picks_what_is_drawn_there(qapp):
    strip = _two_lengths(qapp)
    strip.set_view(
        0.25, 0.35
    )  # minutes 15-21 of the axis: the long row's band, 30-36, is off
    picked = []
    strip.cluster_picked.connect(picked.append)
    _click(strip, _at(strip, 0.55, 0))  # short row, minute 16.5, inside its band
    assert picked == [1]


def test_esc_and_a_double_click_on_blank_time_show_the_whole_span(qapp):
    from detail.gui.qt import QEvent, QKeyEvent, Qt

    strip = _two_lengths(qapp)
    strip.set_view(0.2, 0.3)
    strip.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
    )
    assert not strip.is_zoomed()
    strip.set_view(0.2, 0.3)
    _mouse(
        strip, "MouseButtonDblClick", _at(strip, 0.9, 1)
    )  # blank time in the long row
    assert not strip.is_zoomed()


def test_shift_wheel_moves_through_the_recordings_not_the_time(qapp):
    from detail.gui.qt import QScrollArea

    strip = EthogramStrip()
    strip.set_rows(
        [(f"v{i}", [(0.1, 0.2, 0)]) for i in range(40)],
        cluster_colors(2),
        None,
        lengths_ns=[MIN] * 40,
    )
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(strip)
    area.resize(800, 120)
    area.show()
    qapp.processEvents()
    _wheel(strip, 400, -2, shift=True)
    assert area.verticalScrollBar().value() > 0
    assert not strip.is_zoomed()
    area.close()


def test_a_new_axis_resets_the_zoom_and_the_same_axis_keeps_it(qapp):
    """Clicking another cluster redraws the rows over the same recordings, and the minutes you
    zoomed into are the ones you want to compare it in. A different set of recordings is a
    different axis, where the old window means nothing."""
    strip = _two_lengths(qapp)
    strip.set_view(0.2, 0.3)
    strip.set_rows(
        [("short", []), ("long", [])],
        cluster_colors(2),
        None,
        lengths_ns=[30 * MIN, 60 * MIN],
    )
    assert strip.view == (0.2, 0.3)
    strip.set_rows([("other", [])], cluster_colors(2), None, lengths_ns=[10 * MIN])
    assert not strip.is_zoomed()


def test_the_header_swaps_its_hint_for_whole_and_the_overview_while_zoomed(qapp):
    """One row, one height: the hint names the gestures until you use them, then gives its place
    to **Whole** and the overview — a real scrollbar bound to the strip's zoom."""
    from detail.gui.qt import QScrollBar

    from detail.gui.cluster_strip import ZOOM_HINT, ruled

    strip = _two_lengths(qapp)
    holder = ruled(strip)
    holder.resize(1000, 200)
    holder.show()
    qapp.processEvents()
    assert holder.hint.isVisible() and holder.hint.text() == ZOOM_HINT
    assert not holder.whole.isVisible() and not holder.overview.isVisible()
    flat = holder.sizeHint().height()
    strip.set_view(0.0, 0.1)
    qapp.processEvents()
    assert not holder.hint.isVisible()
    assert holder.whole.isVisible() and holder.overview.isVisible()
    assert holder.sizeHint().height() == flat, "the strips must not jump"
    assert isinstance(holder.overview, QScrollBar)
    # Dragging the overview moves the window.
    bar = holder.overview
    bar.setValue(bar.maximum())
    assert strip.view[1] == pytest.approx(1.0)
    holder.whole.click()
    assert not strip.is_zoomed() and holder.hint.isVisible()
    holder.close()


def test_the_ruler_labels_minutes(qapp):
    from detail.gui.cluster_strip import clock

    assert clock(0) == "0:00"
    assert clock(754 * 1e9) == "12:34"
    assert clock(3750 * 1e9) == "1:02:30"


def test_the_wheel_zooms_anywhere_in_the_plot_area_not_only_on_a_row(qapp):
    """10/5: with two recordings, most of the band is blank under the rows, and a wheel there
    did nothing while the rows just above zoomed. The blank area and the ruler zoom too, at the
    time under the pointer; Shift+wheel there still moves through the recordings."""
    from detail.gui.qt import (
        QApplication,
        QEvent,
        QPoint,
        QPointF,
        QScrollArea,
        QWheelEvent,
        Qt,
    )

    from detail.gui.cluster_strip import ruled

    strip = _two_lengths(qapp)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(strip)
    holder = ruled(strip, area)
    holder.resize(1000, 400)
    holder.show()
    qapp.processEvents()
    viewport = area.viewport()
    blank = QPoint(500, viewport.height() - 10)
    assert strip.geometry().bottom() < blank.y(), "the point must be below every row"

    def wheel(target, point, *, shift=False):
        global_point = QPointF(target.mapToGlobal(point))
        event = QWheelEvent(
            QPointF(point),
            global_point,
            QPoint(0, 0),
            QPoint(0, 240),
            Qt.MouseButton.NoButton,
            (
                Qt.KeyboardModifier.ShiftModifier
                if shift
                else Qt.KeyboardModifier.NoModifier
            ),
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(target, event)

    wheel(viewport, blank, shift=True)
    assert not strip.is_zoomed(), "Shift over the blank area is not a zoom"
    wheel(viewport, blank)
    assert strip.is_zoomed(), "a wheel over the blank area under the rows zooms"
    strip.fit()
    wheel(holder.ruler, QPoint(500, 5))
    assert strip.is_zoomed(), "a wheel over the ruler zooms"
    holder.close()
