"""Detection run-preview widgets: pixmap conversion + the overview strip."""

import numpy as np
import pytest
from tests import waits  # noqa: E402

pytest.importorskip("PyQt6")


def test_to_pixmap_dimensions(qapp):
    from detail.gui.run_preview import _to_pixmap

    pixmap = _to_pixmap(np.zeros((20, 30, 3), np.uint8))
    assert pixmap.width() == 30 and pixmap.height() == 20


def test_count_strip(qapp):
    from detail.gui.run_preview import _CountStrip

    strip = _CountStrip(np.array([0, 2, 1, 0, 3]))
    strip.set_cursor(3)  # repaint cursor, no crash
    assert strip.height() == 44


def test_track_timeline(qapp):
    from detail.gui.run_preview import _TrackTimeline

    presence = {
        0: np.array([True, False, True, False, False]),
        1: np.array([False, True, True, False, False]),
    }
    timeline = _TrackTimeline([0, 1], presence, total=5)
    timeline.set_cursor(2)  # repaint cursor, no crash
    assert timeline.height() == 2 * timeline._ROW_H + 4  # one row per track
    # Sized in lines of the interface font, so it follows a user who asked for larger
    # text instead of keeping the box and growing only the words inside it.
    from detail.gui.sizing import lines

    assert timeline._ROW_H == lines(_TrackTimeline._ROW_LINES)


def _naive_lost(row, plot_w: int) -> list[int]:
    """Frames lost per column, written the obvious way, to check the vectorized one."""
    n = len(row)
    out = []
    for px in range(plot_w):
        lo = int(px / plot_w * n)
        hi = max(int((px + 1) / plot_w * n), lo + 1)
        chunk = row[lo:hi]
        out.append(int(len(chunk) - chunk.sum()))
    return out


def test_track_timeline_buckets_match_naive(qapp):
    """The vectorized bucketing must agree with the obvious per-column version.

    It used to be checked against ``row[lo:hi].any()`` — the behaviour that made a lost
    frame invisible — so the test was pinning the bug rather than guarding against it.
    """
    from detail.gui.run_preview import _TrackTimeline

    rng = np.random.default_rng(0)
    for n in (3, 50, 200):
        row = rng.random(n) < 0.4
        timeline = _TrackTimeline([0], {0: row}, total=n)
        for plot_w in (1, 7, n, n + 13, 3 * n):
            got = timeline._buckets_for(plot_w)[0]
            assert got.tolist() == _naive_lost(row, plot_w), (n, plot_w)


def test_a_lossless_stretch_draws_nothing_and_a_dropout_draws_tall(qapp):
    """The picture has to separate a real dropout from the background scatter.

    At the scale this runs at — 400,000 frames over ~1,000 pixels, 420 a column — a run
    that lost 1% of its frames has *some* loss in nearly every column. An encoding keyed
    on "was anything lost here" lights all of them; one keyed on the covered fraction
    puts 87% and 99% both at full height in a 14px row. Neither separates the four real
    dropouts from the noise, which is the only thing the plot can usefully say.
    """
    from detail.gui.run_preview import _TrackTimeline

    n, plot_w = 42_000, 100  # 420 frames per column
    row = np.ones(n, dtype=bool)
    rng = np.random.default_rng(0)
    row[rng.choice(n, 400, replace=False)] = False  # diffuse 1% scatter
    row[20_000:20_200] = False  # and one real dropout

    timeline = _TrackTimeline([0], {0: row}, total=n)
    lost = timeline._buckets_for(plot_w)[0]

    # The tallest column is where the dropout is. It straddles a column boundary — 420
    # frames a column, 200 lost from frame 20,000 — so the loss lands in two columns and
    # neither holds all of it; the assertion is about *which* column, not how many.
    dropout = int(lost.argmax())
    assert dropout in (47, 48), f"the tallest column is {dropout}, not the dropout"
    # An order of magnitude between the dropout and a typical column is what makes the
    # scatter read as a floor rather than as more dropouts.
    assert lost[dropout] > 10 * np.median(lost)


def test_a_column_that_lost_nothing_is_empty(qapp):
    """Empty means nothing lost. Rounding a small loss up to a pixel is what made the
    previous encoding light every column and say less than the one it replaced."""
    from detail.gui.run_preview import _TrackTimeline

    row = np.ones(1000, dtype=bool)
    row[500] = False
    timeline = _TrackTimeline([0], {0: row}, total=1000)
    lost = timeline._buckets_for(10)[0]
    assert lost[5] == 1
    assert [v for i, v in enumerate(lost) if i != 5] == [0] * 9


class _FakeRunPreview:
    part_frame_counts = [5]
    frames_processed = 5  # every_n_frames=1, so every source frame was looked at

    def processed_mask(self):
        return np.ones(sum(self.part_frame_counts), dtype=bool)

    def counts(self, part):
        # Frame 1 has a detection the tracker does not take; frame 3 has none at all.
        # Those are different failures and the summary has to separate them, so the two
        # ratios must differ here — they used to be equal, and a summary that printed one
        # number twice would have passed.
        return np.array([1, 1, 2, 0, 1])

    def frame(self, part, local, with_masks=False):
        return (local, np.zeros((8, 8, 3), np.uint8), [])

    def track_presence(self):
        return [0], {0: np.array([True, False, True, False, True])}

    def timestamps(self):
        # The run's own frame times — 30 fps from zero, one per source frame.
        return np.arange(sum(self.part_frame_counts), dtype=np.int64) * 33_333_333


class _FakeRunRecording:
    video_fps = 30.0  # the preview plays at the recording's rate, so the fake has one

    def detection_preview(self, run_id):
        return _FakeRunPreview()

    def tracking_preview(self, run_id):
        return _FakeRunPreview()


def _wait_populated(w, attr="_scrubber", timeout_ms=2000):
    """The run's counts load off-thread, so the strip/timeline/scrubber appear only
    after _populate; spin the event loop until they do."""
    from PyQt6.QtTest import QTest

    for _ in range(timeout_ms // 10):
        if hasattr(w, attr):
            return
        QTest.qWait(10)
    raise AssertionError(f"run preview did not populate ({attr}) off-thread")


def test_detection_preview_builds_and_strip_click_jumps(qapp):
    from PyQt6.QtTest import QTest

    from detail.gui.run_preview import DetectionRunPreview

    w = DetectionRunPreview(_FakeRunRecording(), "det_01")
    _wait_populated(w, "_strip")  # off-thread load populates the strip + scrubber
    assert w._masks.isChecked() and w._masks_on  # masks overlaid by default
    # Detection still reports detection counts. The summary moved behind a hook so
    # tracking could say something else; a refactor that quietly changed *this* line too
    # would have gone unnoticed, since nothing asserted it.
    assert w._summary.text() == "5 detections · 4/5 frames with ≥1 · up to 2/frame"
    w._strip.clicked.emit(2)  # strip click → scrubber.set_frame
    QTest.qWait(60)
    assert w._frame == 2


def test_tracking_preview_builds_and_timeline_click_jumps(qapp):
    from PyQt6.QtTest import QTest

    from detail.gui.run_preview import TrackingRunPreview

    w = TrackingRunPreview(_FakeRunRecording(), "track_01")  # builds the track timeline
    _wait_populated(w, "_timeline")
    w._timeline.clicked.emit(3)  # the click that used to crash on the removed _go
    QTest.qWait(60)
    assert w._frame == 3


class _FakeTwoSubjectPreview(_FakeRunPreview):
    def track_presence(self):
        return [0, 1], {
            0: np.array([True, False, True, False, True]),
            1: np.array([True, True, False, True, True]),
        }


class _FakeTwoSubjectRecording(_FakeRunRecording):
    def tracking_preview(self, run_id):
        return _FakeTwoSubjectPreview()


def _painted_cursor_x(strip, host) -> int:
    """Where ``strip`` actually drew its playhead, in ``host``'s coordinates.

    Read off the pixels rather than asked of the widget: the alignment that was reported
    broken is a fact about the picture, and a mapping both strips agree on is no help if
    one of them paints somewhere else. Row 0 is clear of both strips' bars (they start two
    pixels down), so the only ink there is the cursor.
    """
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QColor

    from detail.gui import plot_theme

    ink = QColor(plot_theme.current().text).rgb()
    image = strip.grab().toImage()
    columns = [x for x in range(image.width()) if image.pixel(x, 0) == ink]
    assert columns, (
        f"no playhead painted on {type(strip).__name__}'s top row — only "
        f"{sorted({hex(image.pixel(x, 0)) for x in range(image.width())})}"
    )
    return strip.mapTo(host, QPoint(columns[0], 0)).x()


@pytest.mark.parametrize(
    "recording",
    [_FakeRunRecording, _FakeTwoSubjectRecording],
    ids=["one-subject", "two-subjects"],
)
def test_the_count_strip_and_the_lost_frames_timeline_share_one_axis(qapp, recording):
    """Reported: the lost-frames raster "doesn't line up with the top raster because there
    is blank space and a text 'zero' on the left side". The timeline reserved a 26px label
    gutter the count strip above it did not, so every frame sat 26px apart on the two.

    Both topologies, because the fix is not "delete the gutter": with more than one subject
    the labels are how the rows are told apart, so the strip above moves in to match.
    """
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    from detail.gui.run_preview import TrackingRunPreview

    w = TrackingRunPreview(recording(), "track_01")
    _wait_populated(w, "_timeline")
    w.resize(900, 700)
    w.show()
    qapp.processEvents()
    try:
        for frame in (0, 2, 4):
            w._scrubber.set_frame(frame)
            qapp.processEvents()
            assert _painted_cursor_x(w._strip, w) == _painted_cursor_x(
                w._timeline, w
            ), f"frame {frame} is drawn at two different places on the two strips"

        # And the inverse: one position on the page is one frame on either strip.
        picked = []
        w._strip.clicked.connect(lambda f: picked.append(("strip", f)))
        w._timeline.clicked.connect(lambda f: picked.append(("timeline", f)))
        x = w._timeline.mapTo(w, QPoint(w._timeline.width() * 3 // 5, 0)).x()
        for strip in (w._strip, w._timeline):
            local = strip.mapFrom(w, QPoint(x, 0)).x()
            QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(local, 5))
        assert picked[0][1] == picked[1][1], picked
    finally:
        w.close()


def test_one_subject_gets_no_label_gutter(qapp):
    """A single row labelled ``0`` tells the reader nothing, so it costs the plot nothing."""
    from detail.gui.run_preview import _TrackTimeline

    one = _TrackTimeline([0], {0: np.ones(5, bool)}, total=5)
    two = _TrackTimeline([0, 1], {0: np.ones(5, bool), 1: np.ones(5, bool)}, total=5)
    assert one.gutter == 0
    assert two.gutter > 0


@pytest.mark.parametrize("which", ["DetectionRunPreview", "TrackingRunPreview"])
def test_the_frame_is_on_the_page_and_its_zoom_survives_the_next_frame(qapp, which):
    """Detection and Tracking Results zoom, and zoom is only usable if scrubbing keeps it.

    Also that the picture is *on the page*: the picture-and-controls block is built after the
    off-thread load, and re-parenting the picture into it drops the picture's layout item — a
    `replaceWidget` written after that found nothing to replace, the block was never added,
    and the preview rendered its strips around an empty space. Construction, the smoke corpus
    and the controls-under-picture rule all passed against it; the block measured itself.
    """
    from PyQt6.QtTest import QTest

    from detail.gui import run_preview
    from detail.gui.image_view import ImageView

    w = getattr(run_preview, which)(_FakeRunRecording(), "run_01")
    w.resize(900, 700)
    w.show()
    _wait_populated(w, "_block")
    waits.until(qapp, lambda: w._image.pixmap_size() is not None, "the first frame")

    assert isinstance(w._image, ImageView) and w._image.zoomable
    assert (
        w._block.parentWidget() is w
    ), "the picture's block was never added to the page"
    assert w._image.isVisible() and w._scrubber.isVisible()

    for _ in range(4):
        w._image._set_zoom(w._image.zoom * 1.25, about=None)
    zoomed = w._image.zoom
    assert zoomed > 1.0
    w._scrubber.set_frame(3)
    waits.until(qapp, lambda: w._frame == 3, "the scrub")
    QTest.qWait(50)
    assert w._image.zoom == zoomed, "a new frame reset the zoom, so scrubbing loses it"


def test_a_tracking_run_reports_its_coverage_not_its_detection_counts(qapp):
    """What a user opens a tracking run to find out.

    The summary line is inherited from the detection preview, so tracking reported how
    many *detections* there were and never how much of the recording it ended up holding
    — which is the thing being asked about when frames go missing. The fake covers 3 of
    5 frames with two one-frame gaps.
    """
    from detail.gui.run_preview import TrackingRunPreview

    w = TrackingRunPreview(_FakeRunRecording(), "track_01")
    _wait_populated(w, "_timeline")
    text = w._summary.text()

    assert "subject 0" in text, f"the numbers do not say whose they are: {text!r}"
    # Two ratios, and they must differ: 3 frames carry a subject, 4 carry a detection.
    # Frame 1 is the difference — the detector found something and the tracker did not
    # take it, which is the only one of the two a user can act on. One number hides it.
    assert "3/5 frames with tracked detections" in text, text
    assert "4/5 frames with detections" in text, text
    assert "2 gaps" in text, text
    # The gap duration is stated, because "61 frames" means nothing without a rate.
    assert "longest 1 frames (0.0 s)" in text, text


def test_the_denominator_is_frames_the_run_looked_at_not_source_frames(qapp):
    """A cadence must not read as loss.

    At ``every_n_frames=4`` three frames in four were never processed. Measured against
    the *source* frame count a perfectly healthy run reports having missed 75% of the
    recording — alarming, wrong, and entirely an artifact of the denominator. These
    numbers exist to tell someone whether their tracking is sound; ones that panic over a
    setting are worse than none.
    """
    from detail.gui.run_preview import TrackingRunPreview

    class _EveryFourth(_FakeRunPreview):
        part_frame_counts = [20]
        frames_processed = 5  # frames 0, 4, 8, 12, 16

        def processed_mask(self):
            m = np.zeros(20, dtype=bool)
            m[::4] = True
            return m

        def counts(self, part):
            c = np.zeros(20, dtype=int)
            c[::4] = 1  # a detection on every processed frame
            return c

        def track_presence(self):
            p = np.zeros(20, dtype=bool)
            p[::4] = True  # and a subject on every one of them
            return [0], {0: p}

    class _Recording(_FakeRunRecording):
        def tracking_preview(self, run_id):
            return _EveryFourth()

    w = TrackingRunPreview(_Recording(), "track_01")
    _wait_populated(w, "_timeline")
    text = w._summary.text()
    assert "5/5 frames with tracked detections" in text, text
    assert "5/5 frames with detections" in text, text
    assert "/20" not in text, f"the source frame count leaked into the ratio: {text!r}"
    # Nor into the gaps: every processed frame holds the subject, so there are none. Over
    # source frames each run of three unprocessed frames read as a gap (#32).
    assert "gap" not in text, f"frames never looked at were counted as gaps: {text!r}"


def test_a_tracking_run_with_no_subject_says_so(qapp):
    """An empty state is a claim about the data, and this one is worth making loudly."""
    from detail.gui.run_preview import TrackingRunPreview

    class _Nothing(_FakeRunPreview):
        def track_presence(self):
            return [], {}

    class _Recording(_FakeRunRecording):
        def tracking_preview(self, run_id):
            return _Nothing()

    w = TrackingRunPreview(_Recording(), "track_01")
    _wait_populated(w, "_timeline")
    assert "no subject" in w._summary.text().lower()


class _FakePreview:
    part_frame_counts = [3]

    def input_at(self, part, local, track):
        rgb = np.full((48, 64, 3), 100, np.uint8)
        # only frame 0 has a detection for the track; others have no input
        model_input = np.full((32, 32, 3), 255, np.uint8) if local == 0 else None
        return local, rgb, model_input

    def timestamps(self):
        # The run's own frame times — 30 fps from zero, one per source frame.
        return np.arange(sum(self.part_frame_counts), dtype=np.int64) * 33_333_333


class _FakeRecording:
    id = "s1"
    name = "Fish tank A"  # pickers show the name (§11); the id is the data
    video_fps = 25.0  # the scrubber plays at the recording's rate, so the fake has one

    def resolve_constructor(
        self, constructor, tracking_run_id, track_ids, model_id=None
    ):
        return [{"type": "init_binary_mask"}]

    def input_image_preview(self, tracking_run_id, renderer_spec):
        return _FakePreview()


def test_input_image_scrubber(qapp):
    from PyQt6.QtTest import QTest

    from detail.gui.run_preview import InputImageScrubber

    scrubber = InputImageScrubber(
        [_FakeRecording()], {"s1": "track_01"}, [0, 1], "full_frame"
    )
    for _ in range(200):  # the constructor-spec resolve + preview open runs off-thread
        if scrubber._total:
            break
        QTest.qWait(10)
    assert scrubber._total == 3  # scrubber spans every source frame
    assert scrubber._scrubber._slider.maximum() == 2
    scrubber._display(0, scrubber._compute(0))  # frame with a detection -> both panes
    assert not scrubber._orig.pixmap().isNull()
    assert not scrubber._input.pixmap().isNull()
    scrubber._display(1, scrubber._compute(1))  # track absent -> input pane cleared
    assert not scrubber._orig.pixmap().isNull()
    assert scrubber._input.pixmap().isNull()


class _FakeEmbPreview:
    def metadata(self):
        return {
            "model_id": "dinov3",
            "constructor": "full_frame",
            "output_dim": 4,
            "status": "complete",
        }

    def coverage(self):
        return [{"track_id": 0, "n_frames": 2, "n_parts": 1}]

    def pca_2d(self, track_ids=None, max_points=None):
        coords = np.array([[0.0, 0.0], [1.0, 1.0]], np.float32)
        locators = [(0, "part_001", 0, 0), (0, "part_001", 2, 66)]
        return coords, np.array([0, 0]), locators

    def frame_view(self, track_id, stem, frame_idx):
        return (
            frame_idx,
            np.full((48, 64, 3), 100, np.uint8),
            np.full((32, 32, 3), 255, np.uint8),
        )

    def frame_position(self, stem, frame_idx):
        # As if part_001 were the recording's second part, after 100 frames of the first —
        # so a title that showed the part-local index would read 0, not 100.
        return 100 + frame_idx, 250


class _FakeEmbRecording:
    def embedding_preview(self, run_id):
        return _FakeEmbPreview()


def test_embedding_run_preview(qapp):
    import types

    from PyQt6.QtTest import QTest

    pytest.importorskip("pyqtgraph")
    from detail.gui.run_preview import EmbeddingRunPreview

    rec = _FakeEmbRecording()
    widget = EmbeddingRunPreview(rec, "emb_01")
    assert len(widget._scatter.points()) == 0  # load is async — shell only so far
    # Wait for the widget's own load rather than driving a second one: a synchronous
    # `_on_loaded` here left the real one in flight, and when it landed after the click below
    # it reset the status to "Click a point to inspect" — green alone, red in a loaded run.
    waits.until(
        qapp, lambda: len(widget._scatter.points()) == 2, "the embedding load"
    )  # one point per embedded frame
    # pyqtgraph hands sigClicked a numpy array of spot items (>1 when points
    # overlap) — `if not points` raised "ambiguous truth value" on it
    points = np.array(
        [types.SimpleNamespace(data=lambda: 0) for _ in range(2)], dtype=object
    )
    widget._on_clicked(None, points)  # the frame decode runs off-thread now
    # A failsafe, not a budget: this was 200 spins of 10ms, and a 2s ceiling on an off-thread
    # decode is a claim about the machine. It timed out inside a loaded suite and passed alone.
    waits.until(
        qapp,
        lambda: widget._orig.pixmap() is not None
        and not widget._orig.pixmap().isNull(),
        "the clicked frame's decode",
    )
    assert not widget._orig.pixmap().isNull()
    assert not widget._input.pixmap().isNull()
    # The clicked frame names itself, in the scrubbers' `v / last` form and in recording
    # frames; the status line no longer repeats the track and a part-local index.
    assert widget._frame_title.text() == "Frame 100 / 249"
    assert "track" not in widget._status.text()
    assert "frame" not in widget._status.text()


def test_embedding_preview_puts_scatter_frame_and_input_side_by_side(qapp):
    """Reported: the scatter could be smaller so it, the frame and the model input each take
    a third of the width — and the coverage table under it said nothing worth its room.
    """
    from PyQt6.QtWidgets import QTableView, QTableWidget

    pytest.importorskip("pyqtgraph")
    from detail.gui.run_preview import EmbeddingRunPreview

    widget = EmbeddingRunPreview(_FakeEmbRecording(), "emb_01")
    waits.until(qapp, lambda: len(widget._scatter.points()) == 2, "the embedding load")
    widget.resize(1500, 800)
    widget.show()
    qapp.processEvents()
    try:
        columns = [widget._plot, widget._orig, widget._input]
        tops = {c.mapTo(widget, c.rect().topLeft()).y() for c in columns}
        assert len(tops) == 1, f"the three panes are not in one row: tops {tops}"
        lefts = [c.mapTo(widget, c.rect().topLeft()).x() for c in columns]
        assert lefts == sorted(lefts), "scatter, frame, model input — in that order"
        widths = [c.width() for c in columns]
        assert max(widths) - min(widths) <= 4, f"not thirds: {widths}"
        assert min(widths) >= widget.width() // 3 - 40, f"thirds of what? {widths}"
        assert not widget.findChildren(QTableWidget)
        assert not widget.findChildren(QTableView)
    finally:
        widget.close()
