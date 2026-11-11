"""Tests for storage/video.py — container probing, timestamp maps, and seek reads.

Class guarded here (with ``tests/pipeline/test_detection_worker.py``): **a timestamp map
built from something other than the frames a decoder shows.** The map counted MP4
edit-list packets (``Packet.is_discard``) that cv2 never returns, and kept the file's
first PTS instead of starting at 0. A trimmed part's map was longer than its video, so
detection placed every row late by the number trimmed and called the last ones
dropped; a part whose PTS started at 3 s overlapped the part after it. It hid because
every fixture started at PTS 0 with no edit list — ``write_vfr_video``'s ``start_ms``
and ``b_frames`` options are the fixtures that reach it. The rule: the map has one row
per shown frame, starts at 0, and every reader converts between map and file time with
one per-part constant (``video._map_shift_ns``). Two parts joining in time is
``tests/api/test_recording.py::test_parts_join_in_time_whatever_each_file_starts_at``.
"""

import unittest.mock

import av
import cv2
import numpy as np
import pytest

from detail.storage import video

H, W, FPS = 24, 32, 30.0


def _write_video(path, n_frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 10) % 256, np.uint8))
    vw.release()


def _write_distinct_video(path, n_frames):
    """A video whose frames are pairwise-distinct, so a seek's output can be
    checked against the exact sequential frame it must return."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, :, 0] = (i * 37) % 256
        frame[:, :, 1] = (i * 91 + 20) % 256
        frame[:, :, 2] = (i * 13 + 50) % 256
        vw.write(frame)
    vw.release()


def _decode_all_bgr(path):
    """Every frame in presentation order via PyAV (the same decoder + indexing
    ``read_frame_at`` uses), so a frame-for-frame comparison is exact."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [f.to_ndarray(format="bgr24") for f in container.decode(stream)]


def test_probe_video(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 7)
    info = video.probe_video(p)
    assert round(info["fps"]) == 30
    assert info["resolution"] == [W, H]


def test_generate_timestamps(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 7)
    t = video.generate_timestamps(p, FPS)
    assert t.dtype == np.int64
    assert len(t) == 7
    assert t[0] == 0
    assert abs(int(t[1]) - 33_333_333) < 50_000  # ~33.33 ms at 30 fps
    assert (np.diff(t) > 0).all()  # strictly increasing


def test_demux_and_decode_timestamps_agree(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 9)
    demux = video._timestamps_via_demux(p)
    decode = video._timestamps_via_decode(p, FPS)
    assert demux is not None
    # the fast path must reproduce the decode ground truth bit-for-bit
    assert demux.tolist() == decode.tolist()


def test_generate_timestamps_prefers_demux(tmp_path, monkeypatch):
    p = tmp_path / "v.mp4"
    _write_video(p, 7)

    def _boom(*_args, **_kwargs):
        raise AssertionError("should not decode when demux is trustworthy")

    monkeypatch.setattr(video, "_timestamps_via_decode", _boom)
    # no cv2 frame-count gate: a clean demux is used directly (the fast path)
    assert len(video.generate_timestamps(p, FPS)) == 7


def test_generate_timestamps_falls_back_when_demux_untrusted(tmp_path, monkeypatch):
    p = tmp_path / "v.mp4"
    _write_video(p, 5)
    # demux signalling it can't be trusted (None) is the *only* thing that
    # forces the decode ground truth now
    monkeypatch.setattr(video, "_timestamps_via_demux", lambda _p, _prog=None: None)
    assert len(video.generate_timestamps(p, FPS)) == 5


def test_generate_timestamps_reports_progress(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 7)
    seen = []
    video.generate_timestamps(p, FPS, progress=seen.append)
    assert seen and seen[-1] == 1.0  # the scan reports through to 100%


def test_copy_with_progress_copies_and_reports(tmp_path):
    src = tmp_path / "src.mp4"
    _write_video(src, 5)
    dst = tmp_path / "dst.mp4"
    seen = []
    total = video.copy_with_progress(
        src, dst, progress=lambda d, t: seen.append((d, t))
    )
    assert dst.read_bytes() == src.read_bytes()  # exact copy
    assert total == src.stat().st_size
    assert seen[-1] == (total, total)  # last tick reaches 100%


def test_probe_missing_stream_raises(tmp_path):
    import pytest

    bad = tmp_path / "notavideo.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(Exception):  # av raises on an invalid container
        video.probe_video(bad)


# --- read_frame_at --------------------------------------------------------- #
def test_read_frame_at_returns_bgr_shape(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 5)
    result = video.read_frame_at(p, 2)
    assert result is not None
    actual, frame = result
    assert isinstance(actual, int)
    assert frame.shape == (H, W, 3)
    assert frame.dtype == np.uint8


def test_read_frame_at_distinct_frames(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 5)
    f0 = video.read_frame_at(p, 0)[1]
    f3 = video.read_frame_at(p, 3)[1]
    assert not np.array_equal(f0, f3)  # different requested frames -> different output


def test_read_frame_at_far_out_of_range_returns_none(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 5)
    assert video.read_frame_at(p, 9999) is None


def test_read_frame_at_matches_sequential_decode(tmp_path):
    # The seek must return the same picture the sequential decode calls frame N
    # (the indexing detection assigns) and report that true index — not cv2's
    # fps-derived CAP_PROP_POS_FRAMES cursor.
    p = tmp_path / "v.mp4"
    n = 40
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    for target in (0, 1, 7, 15, 23, n - 1):
        actual, frame = video.read_frame_at(p, target)
        assert actual == target
        assert np.array_equal(frame, seq[target])


def test_read_frame_at_follows_pts_index_not_frame_number(tmp_path, monkeypatch):
    # The bug: preview seeked by frame *number* (cv2 POS_FRAMES, fps-based), so on
    # a file whose advertised fps differs from its true PTS rate the picture and
    # the overlay (looked up by frame index) came from different frames. Inject a
    # non-identity PTS index and assert retrieval follows the PTS, not the ordinal
    # — a frame-number seek cannot satisfy this.
    p = tmp_path / "v.mp4"
    n, shift = 40, 7
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    real = video._cached_frame_pts.__wrapped__(str(p), 0, 0)  # bypass the cache
    shifted = real[shift:]  # logical frame k -> real presentation frame k+shift
    monkeypatch.setattr(video, "_cached_frame_pts", lambda *a, **k: shifted)
    for k in (0, 5, 12, 20, 30):
        actual, frame = video.read_frame_at(p, k)
        assert actual == k
        assert np.array_equal(frame, seq[k + shift])


def test_read_frame_at_stays_exact_when_pts_untrusted(
    tmp_path, monkeypatch, write_vfr_video
):
    # Previously this fell back to the cv2 POS_FRAMES seek — "best effort" — which
    # returns the wrong picture AND reports the requested index back, so overlays
    # were drawn on a frame that wasn't shown. With no trustworthy PTS index the
    # retrieval must still be exact (counting decoded frames), never approximate.
    # The VFR fixture is what exposes this: on a constant-rate file the old seek
    # happened to be right, so this test passed against the bug.
    p = write_vfr_video(tmp_path / "v.mp4", n_frames=60)
    seq = _decode_all_bgr(p)
    monkeypatch.setattr(video, "_cached_frame_pts", lambda *a, **k: None)
    for target in (0, 10, 25, 40, 55):
        actual, frame = video.read_frame_at(p, target)
        assert actual == target
        assert np.array_equal(frame, seq[target])


def test_read_frame_at_uses_the_timestamp_map_without_scanning_packets(
    tmp_path, monkeypatch
):
    # The map is the authoritative index, so passing it must resolve the frame
    # exactly *and* skip the whole-file packet scan (the ~11s first-seek cost on a
    # 1GB part). Failing the scan hard proves it isn't consulted.
    p = tmp_path / "v.mp4"
    n = 20
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    timestamps = video.generate_timestamps(p, FPS)
    assert len(timestamps) == n

    def boom(*a, **k):
        raise AssertionError("packet scan must not run when a timestamp map is given")

    monkeypatch.setattr(video, "_cached_frame_pts", boom)
    for target in (0, 3, 11, n - 1):
        actual, frame = video.read_frame_at(p, target, timestamps=timestamps)
        assert actual == target
        assert np.array_equal(frame, seq[target])


def test_read_frame_at_falls_through_a_map_that_is_not_this_files_pts(tmp_path):
    # A map does not always describe the file it is passed with: a **downsampled**
    # part's map holds the ORIGINAL frames' t_ns by design (§4), and
    # _timestamps_via_decode synthesises i/fps for frames missing a PTS. When the
    # decoded frame's t_ns isn't in the map, drop to the packet index rather than
    # return a mismatched (picture, index) pair — and never raise.
    p = tmp_path / "v.mp4"
    n = 10
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    bogus = np.arange(n, dtype=np.int64) * 7_000_000 + 3  # not this file's PTS
    for target in (0, 4, 9):
        actual, frame = video.read_frame_at(p, target, timestamps=bogus)
        assert actual == target
        assert np.array_equal(frame, seq[target])


def test_read_frame_at_falls_all_the_way_to_decode_when_nothing_aligns(
    tmp_path, monkeypatch
):
    # Map doesn't match the file AND the packet index is unusable: the last rung
    # still has to be exact.
    p = tmp_path / "v.mp4"
    n = 10
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    bogus = np.arange(n, dtype=np.int64) * 7_000_000 + 3
    monkeypatch.setattr(video, "_cached_frame_pts", lambda *a, **k: None)
    actual, frame = video.read_frame_at(p, 6, timestamps=bogus)
    assert actual == 6
    assert np.array_equal(frame, seq[6])


def test_read_frame_at_caps_the_sequential_decode(tmp_path, monkeypatch):
    # Without an index the only exact route is decoding from frame 0, which is
    # minutes deep in a long video — fail loudly instead of silently stalling.
    import pytest

    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 4)
    monkeypatch.setattr(video, "_cached_frame_pts", lambda *a, **k: None)
    monkeypatch.setattr(video, "SEQUENTIAL_DECODE_FRAME_CAP", 3)
    with pytest.raises(ValueError, match="capped at 3 frames"):
        video.read_frame_at(p, 9)


def test_cached_frame_pts_strictly_increasing_and_full_length(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 9)
    video._cached_frame_pts.cache_clear()
    stat = p.stat()
    index = video._cached_frame_pts(str(p), stat.st_size, stat.st_mtime_ns)
    assert index is not None
    assert len(index) == 9
    assert (np.diff(index) > 0).all()


def test_cached_frame_pts_is_memoized(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, 5)
    video._cached_frame_pts.cache_clear()
    stat = p.stat()
    args = (str(p), stat.st_size, stat.st_mtime_ns)
    video._cached_frame_pts(*args)
    video._cached_frame_pts(*args)
    info = video._cached_frame_pts.cache_info()
    assert info.misses == 1
    assert info.hits == 1


# --- interruptible import (should_stop) ------------------------------------- #
def test_copy_with_progress_stops(tmp_path):
    import pytest

    src = tmp_path / "v.mp4"
    _write_video(src, 7)
    dst = tmp_path / "out.mp4"
    with pytest.raises(video.ImportStopped):
        video.copy_with_progress(src, dst, should_stop=lambda: True)


def test_read_frame_at_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="video not present"):
        video.read_frame_at(tmp_path / "nope.mp4", 0)


# --- open_video_writer ------------------------------------------------------------------ #
def test_open_video_writer_returns_a_usable_writer_and_names_its_codec(tmp_path):
    """The fallback stays observable through the returned codec — that, not stderr, is the
    contract callers can act on."""
    out = tmp_path / "w.mp4"
    writer, codec = video.open_video_writer(out, FPS, W, H)
    assert codec in ("avc1", "mp4v")
    try:
        writer.write(np.zeros((H, W, 3), np.uint8))
    finally:
        writer.release()
    assert out.stat().st_size > 0


def test_open_video_writer_raises_when_no_codec_can_open_the_path(tmp_path):
    missing = tmp_path / "no_such_dir" / "w.mp4"
    with pytest.raises(RuntimeError, match="could not open a VideoWriter"):
        video.open_video_writer(missing, FPS, W, H)


def test_an_unwritable_path_does_not_convince_us_the_build_lacks_h264(
    tmp_path, monkeypatch
):
    """A bad path fails at *every* codec, so it must not be mistaken for a missing encoder —
    otherwise one unwritable destination silently downgrades the rest of the process."""
    monkeypatch.setattr(video, "_h264_unavailable", False)
    with pytest.raises(RuntimeError):
        video.open_video_writer(tmp_path / "no_such_dir" / "w.mp4", FPS, W, H)
    assert video._h264_unavailable is False


def test_the_h264_probe_is_not_repeated_once_the_build_has_failed_it(
    tmp_path, monkeypatch
):
    """On a build without H.264 the probe fails every time and prints four error lines when it
    does; repeating it per clip buries a long export in its own noise."""
    monkeypatch.setattr(video, "_h264_unavailable", False)
    first, codec = video.open_video_writer(tmp_path / "a.mp4", FPS, W, H)
    first.release()
    if codec == "avc1":
        pytest.skip("this cv2 build has H.264, so there is no failed probe to skip")
    assert video._h264_unavailable is True
    with unittest.mock.patch("cv2.VideoWriter", wraps=cv2.VideoWriter) as spy:
        second, again = video.open_video_writer(tmp_path / "b.mp4", FPS, W, H)
    second.release()
    assert again == "mp4v" and spy.call_count == 1


# --- read_frames_range ------------------------------------------------------ #
def test_read_frames_range_returns_the_span_in_order(tmp_path):
    p = tmp_path / "v.mp4"
    n = 20
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    timestamps = video.generate_timestamps(p, FPS)

    got = list(video.read_frames_range(p, 4, 11, timestamps=timestamps))
    assert [idx for idx, _ in got] == list(range(4, 12))
    assert all(np.array_equal(frame, seq[idx]) for idx, frame in got)


def test_read_frames_range_matches_read_frame_at_frame_for_frame(tmp_path):
    """The span reader is an optimization, so it has to be indistinguishable from the
    single-frame path it replaces — same pictures, same indices."""
    p = tmp_path / "v.mp4"
    n = 16
    _write_distinct_video(p, n)
    timestamps = video.generate_timestamps(p, FPS)

    span = list(video.read_frames_range(p, 0, n - 1, timestamps=timestamps))
    one_at_a_time = [video.read_frame_at(p, i, timestamps=timestamps) for i in range(n)]
    assert [i for i, _ in span] == [i for i, _ in one_at_a_time]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(span, one_at_a_time))


def test_read_frames_range_opens_the_container_once(tmp_path):
    """The bug this exists for: ``read_frame_at`` opens, seeks and decodes forward *per
    frame*, so playing a clip re-opened a 1080p file once per frame. Counting opens is
    what makes that regression visible — a slower implementation
    that still returns the right pictures passes every other test here."""
    p = tmp_path / "v.mp4"
    n = 24
    _write_distinct_video(p, n)
    timestamps = video.generate_timestamps(p, FPS)

    opens = {"n": 0}
    real_open = av.open

    def counting_open(*a, **k):
        opens["n"] += 1
        return real_open(*a, **k)

    with unittest.mock.patch.object(video.av, "open", counting_open):
        got = list(video.read_frames_range(p, 2, n - 1, timestamps=timestamps))
    assert len(got) == n - 2
    assert opens["n"] == 1


def test_the_open_count_guard_would_catch_a_per_frame_reader(tmp_path):
    """Guard the guard: the open-count assertion has to fail on the shape it forbids."""
    p = tmp_path / "v.mp4"
    n = 8
    _write_distinct_video(p, n)
    timestamps = video.generate_timestamps(p, FPS)

    opens = {"n": 0}
    real_open = av.open

    def counting_open(*a, **k):
        opens["n"] += 1
        return real_open(*a, **k)

    with unittest.mock.patch.object(video.av, "open", counting_open):
        [video.read_frame_at(p, i, timestamps=timestamps) for i in range(n)]
    assert opens["n"] > 1


def test_read_frames_range_strides_without_restarting_the_decode(tmp_path):
    p = tmp_path / "v.mp4"
    n = 20
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    timestamps = video.generate_timestamps(p, FPS)

    got = list(video.read_frames_range(p, 0, n - 1, timestamps=timestamps, step=4))
    assert [idx for idx, _ in got] == [0, 4, 8, 12, 16]
    assert all(np.array_equal(frame, seq[idx]) for idx, frame in got)


def test_read_frames_range_falls_back_when_the_map_is_not_this_files_pts(tmp_path):
    """Same trust rule as the single-frame path — a map that does not describe this file is
    not used to walk a span either, and the caller still gets the right frames."""
    p = tmp_path / "v.mp4"
    n = 10
    _write_distinct_video(p, n)
    seq = _decode_all_bgr(p)
    bogus = np.arange(n, dtype=np.int64) * 7_000_000 + 3

    got = list(video.read_frames_range(p, 1, 5, timestamps=bogus))
    assert [idx for idx, _ in got] == [1, 2, 3, 4, 5]
    assert all(np.array_equal(frame, seq[idx]) for idx, frame in got)


def test_read_frames_range_of_a_single_frame(tmp_path):
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 6)
    seq = _decode_all_bgr(p)
    timestamps = video.generate_timestamps(p, FPS)

    got = list(video.read_frames_range(p, 3, 3, timestamps=timestamps))
    assert len(got) == 1
    assert got[0][0] == 3
    assert np.array_equal(got[0][1], seq[3])


def test_read_frames_range_of_an_inverted_span_is_empty(tmp_path):
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 6)
    timestamps = video.generate_timestamps(p, FPS)
    assert list(video.read_frames_range(p, 4, 2, timestamps=timestamps)) == []


def test_read_frames_range_on_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="video not present"):
        list(video.read_frames_range(tmp_path / "nope.mp4", 0, 3))


# --- decoding threads ---------------------------------------------------------------------- #
# FFmpeg decodes on one core unless told otherwise, and PyAV does not tell it — nothing in this
# package set `thread_type` until now (`grep thread_type detail/` was empty), and both a span
# read and the per-frame seek path get faster with it. The frames that come back are identical,
# which is why no existing test could see it either way — so the property is asserted in two
# halves, because neither alone is the thing: the helper sets it, and every decoding read uses the
# helper.


def test_the_decoding_stream_asks_for_more_than_one_thread(tmp_path):
    """Half one: the helper leaves the stream configured to decode on several threads."""
    p = tmp_path / "v.mp4"
    _write_video(p, 4)

    with av.open(str(p)) as container:
        stream = video._decoding_stream(container)
        assert "AUTO" in str(
            stream.thread_type
        ), f"the decoding stream is left at {stream.thread_type!r}, so FFmpeg uses one core"


def test_a_decoding_read_does_not_open_its_stream_by_hand(tmp_path):
    """Half two: a read that skips the helper decodes single-threaded, and nothing else says so.

    A source scan, because that is the only shape that can see this — a hand-opened stream
    returns identical frames, just slower, so nothing is wrong at the line where it is written.
    The `test_path_serialization.py` idiom.

    Exempt are the reads that **demux without decoding**: configuring a decoder none of them runs
    would be the setting cargo-culted rather than applied.
    """
    import ast
    import pathlib

    #: Opens a container and never calls ``container.decode``, with why.
    NOT_DECODING = {
        "probe_video": "reads container metadata; no decoder runs",
        "_cached_frame_pts": "demuxes packets for their PTS; no decoder runs",
        "_timestamps_via_demux": "demuxes packets for their PTS; no decoder runs",
        "_decoding_stream": "is the helper",
    }

    tree = ast.parse(pathlib.Path(video.__file__).read_text(encoding="utf-8-sig"))

    # A class is the unit where there is one: `FrameCursor` configures its stream once, in the
    # method that opens the container, and decodes from it in another. Asking each method in
    # isolation would report the second as hand-rolled, which is how a guard ends up with
    # exemptions that describe nothing.
    configured_classes = {
        cls.name
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_decoding_stream"
            for n in ast.walk(cls)
        )
    }
    in_configured_class = {
        node.name
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name in configured_classes
        for node in ast.walk(cls)
        if isinstance(node, ast.FunctionDef)
    }

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in NOT_DECODING:
            continue
        if node.name in in_configured_class:
            continue
        calls = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "decode"
        ]
        if not calls:
            continue
        names = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if "_decoding_stream" not in names:
            offenders.append(node.name)

    assert not offenders, (
        f"{offenders} decode frames from a stream opened by hand, so FFmpeg runs single-threaded "
        "there. Take the stream from `_decoding_stream`, or add the function to NOT_DECODING "
        "with the reason it runs no decoder."
    )


def test_the_scan_would_catch_a_hand_opened_decoder():
    """Guard the guard: a scan that matches nothing passes vacuously forever."""
    import ast

    hand_rolled = ast.parse(
        "def read(path):\n"
        "    with av.open(path) as container:\n"
        "        stream = container.streams.video[0]\n"
        "        for frame in container.decode(stream):\n"
        "            return frame\n"
    )
    node = hand_rolled.body[0]
    decodes = any(
        isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "decode"
        for n in ast.walk(node)
    )
    uses_helper = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_decoding_stream"
        for n in ast.walk(node)
    )
    assert (
        decodes and not uses_helper
    ), "the scan's own predicate no longer detects the shape"


# --- the frame cursor ---------------------------------------------------------------------- #
# `read_frame_at` re-opens and re-seeks per call. That is the right shape for a scrub — one frame,
# anywhere, now — and the wrong one for playback, which asks for the *next* frame forty times a
# second. Re-seeking per frame was too slow for 1x playback to show every frame, and a cursor
# that decodes forward is not.
#
# These assert **counts**, not durations. Three or more agent sessions share this machine, so a
# wall clock here measures whoever else is busy; how many times a container is opened is a
# property of the code and is the same on any machine. Same rule the network-share class records.


class _CountingContainer:
    """A container that reports how often it was opened and sought, and is otherwise itself.

    **Seeks, not opens.** Counting opens was the obvious guard and it is the wrong one: a cursor
    that has stopped walking forward still holds its container open and simply seeks on it, so
    the open count stays at one while every frame costs the keyframe walk the cursor exists to
    avoid. Watched passing against exactly that mutation before this replaced it.
    """

    def __init__(self, inner, counts):
        self._inner = inner
        self._counts = counts

    def seek(self, *a, **k):
        self._counts["seek"] += 1
        return self._inner.seek(*a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *a):
        return self._inner.__exit__(*a)


def _counting_open(monkeypatch):
    """Count `av.open` calls and every seek made on what they return."""
    counts = {"n": 0, "seek": 0}
    real = av.open

    def counted(*a, **k):
        counts["n"] += 1
        return _CountingContainer(real(*a, **k), counts)

    monkeypatch.setattr(av, "open", counted)
    return counts


def test_playing_a_span_seeks_once(tmp_path, monkeypatch):
    """The number that must not scale with the frame count.

    One seek to get there, then the decoder walks. A cursor that seeks per frame is
    `read_frame_at` with extra steps — it returns the right pictures and pays the whole cost.
    """
    p = tmp_path / "v.mp4"
    n = 24
    _write_distinct_video(p, n)
    timestamps = video.generate_timestamps(p, FPS)

    cursor = video.FrameCursor(p, timestamps=timestamps)
    counts = _counting_open(monkeypatch)
    for i in range(n):
        assert cursor.frame(i) is not None

    assert counts["seek"] == 1, (
        f"{n} consecutive frames cost {counts['seek']} seeks. One gets there; the decoder walks "
        "the rest."
    )
    assert counts["n"] == 1, f"and one open, not {counts['n']}"


def test_reading_one_at_a_time_would_seek_every_time(tmp_path, monkeypatch):
    """Guard the guard: the counts above have to be a *result*, not something always true here."""
    p = tmp_path / "v.mp4"
    n = 24
    _write_distinct_video(p, n)
    timestamps = video.generate_timestamps(p, FPS)

    counts = _counting_open(monkeypatch)
    for i in range(n):
        video.read_frame_at(p, i, timestamps=timestamps)

    assert counts["seek"] >= n and counts["n"] >= n, (
        f"the per-frame path cost {counts['seek']} seeks and {counts['n']} opens for {n} frames — "
        "if that is no longer true, the test above is measuring nothing"
    )


def test_a_cursor_returns_the_same_frames_as_reading_them_one_at_a_time(tmp_path):
    """Correctness over the four request shapes, because speed is worthless without it.

    Consecutive, skipped (what the scrubber's latest-only decode produces under load), backwards
    (a scrub), and a jump far enough that the cursor must re-seek.
    """
    p = tmp_path / "v.mp4"
    n = 40
    _write_distinct_video(p, n)
    timestamps = video.generate_timestamps(p, FPS)

    cursor = video.FrameCursor(p, timestamps=timestamps)
    for i in [0, 1, 2, 5, 6, 3, 30, 31, 8, n - 1]:
        got = cursor.frame(i)
        expected = video.read_frame_at(p, i, timestamps=timestamps)
        assert (got is None) == (
            expected is None
        ), f"frame {i}: disagreed about decodability"
        assert (
            got[0] == expected[0]
        ), f"frame {i}: cursor said {got[0]}, expected {expected[0]}"
        assert np.array_equal(got[1], expected[1]), f"frame {i}: different picture"


def test_a_backwards_request_does_not_serve_the_frame_it_was_already_on(tmp_path):
    """The failure a cursor invites: keeping a place and then ignoring where it was asked to go.

    Asserted on the returned **index**, not that a call returned — a stale frame is a plausible
    frame, which is exactly the shape that passes a smoke test.
    """
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 30)
    timestamps = video.generate_timestamps(p, FPS)

    cursor = video.FrameCursor(p, timestamps=timestamps)
    assert cursor.frame(20)[0] == 20
    assert (
        cursor.frame(4)[0] == 4
    ), "a backwards request was served from where the cursor stood"
    assert cursor.frame(5)[0] == 5, "and the cursor must be usable afterwards"


def test_two_cursors_on_one_video_do_not_disturb_each_other(tmp_path):
    """Two surfaces can show the same recording at different positions — a player and a strip.

    Each holds its own container, so interleaving them is not a shared-state question. Asserted
    because the tempting cheap version of this is one cursor per *path* in a module-level cache,
    which turns two readers into one that re-seeks on every call.
    """
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 40)
    timestamps = video.generate_timestamps(p, FPS)

    a = video.FrameCursor(p, timestamps=timestamps)
    b = video.FrameCursor(p, timestamps=timestamps)
    for i in range(5):
        assert a.frame(i)[0] == i
        assert b.frame(30 + i)[0] == 30 + i


def test_a_cursor_on_a_missing_video_says_so(tmp_path):
    """The contract it claims to share with `read_frame_at` includes how it refuses.

    `av.open` on an absent path raises its own error, which says only that a path does not exist.
    A video-excluded export is a thing a preview explains (`gui/frame_source.py`), so the cursor
    has to raise what the rest of the package raises.
    """
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 4)
    timestamps = video.generate_timestamps(p, FPS)
    p.unlink()

    cursor = video.FrameCursor(p, timestamps=timestamps)
    with pytest.raises(FileNotFoundError, match="video not present"):
        cursor.frame(0)


def test_a_cursor_past_the_end_returns_none_and_stays_usable(tmp_path):
    """A drained decoder refuses the next packet even after a seek, so it has to be dropped.

    Found by a real failure: reading the last frame and then any other raised
    `av.error.EOFError` out of `avcodec_send_packet`, because the decode generator was being
    recreated per call rather than held — re-entering the decoder instead of resuming it.
    """
    p = tmp_path / "v.mp4"
    n = 12
    _write_distinct_video(p, n)
    timestamps = video.generate_timestamps(p, FPS)

    cursor = video.FrameCursor(p, timestamps=timestamps)
    assert cursor.frame(n - 1)[0] == n - 1
    assert cursor.frame(n) is None, "past the end is None, not an exception"
    assert cursor.frame(0)[0] == 0, "and the cursor still works afterwards"


# --- the map describes the frames a decoder shows -------------------------- #
# Class: a timestamp map built from something other than the frames a decoder shows.
# Two container facts did it. A file's first PTS need not be 0, and the map kept it,
# so a part started seconds into its own recording and two parts could overlap. An MP4
# edit list keeps the frames it trims as ``is_discard`` packets, and the map counted
# them, so it was longer than the video and every detection row was placed that many
# frames late. Both hid because every fixture started at PTS 0 with no edit list.
_VFR_DIFFS_NS = [10_000_000] * 30 + [30_000_000] * 29  # write_vfr_video's spacing


def _cv2_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


@pytest.mark.parametrize("suffix", ["mp4", "mkv"])
@pytest.mark.parametrize("start_ms", [0, 3000])
@pytest.mark.parametrize("path", ["demux", "decode"])
def test_a_map_starts_at_zero_wherever_the_file_starts(
    tmp_path, write_vfr_video, suffix, start_ms, path
):
    p = write_vfr_video(tmp_path / f"v.{suffix}", start_ms=start_ms)
    with av.open(str(p)) as container:  # the fixture reaches the non-zero start
        first = container.streams.video[0].start_time
        assert (first != 0) == (start_ms != 0)
    if path == "demux":
        t_ns = video._timestamps_via_demux(p)
    else:
        t_ns = video._timestamps_via_decode(p, 50.0)
    assert t_ns[0] == 0
    assert np.diff(t_ns).tolist() == _VFR_DIFFS_NS


def test_an_edit_list_trimmed_file_maps_exactly_the_frames_a_decoder_shows(
    tmp_path, write_vfr_video
):
    p = write_vfr_video(tmp_path / "trim.mp4", start_ms=-50, b_frames=2)
    with av.open(str(p)) as container:
        packets = [pk for pk in container.demux(video=0) if pk.size]
    assert any(pk.is_discard for pk in packets), "fixture must carry an edit list"

    shown = _cv2_frame_count(p)
    assert shown == 55  # frames 0..4 sit before PTS 0 and are trimmed
    t_ns = video.generate_timestamps(p, 50.0)
    assert len(t_ns) == shown
    assert t_ns[0] == 0 and (t_ns >= 0).all()
    assert video.shown_frame_count(p) == shown
    # The preview seek index comes through the same demux loop, so it agrees too.
    assert len(video._cached_frame_pts.__wrapped__(str(p), 0, 0)) == shown


def test_previews_of_a_trimmed_file_show_the_frame_its_index_names(
    tmp_path, write_vfr_video, monkeypatch
):
    """Every frame a preview returns is the frame its returned index names.

    Exactness is asserted on frames a seek can land on. A B-frame that references
    the group before its keyframe is not reachable by a keyframe seek on this
    encoder (on an untrimmed B-frame file too), so the reader returns the next frame
    and says so — an honest snap, not this class. Before the fix the packet index
    counted the 5 trimmed frames, and frame ``k`` showed the picture of another index.
    """
    p = write_vfr_video(tmp_path / "trim.mp4", start_ms=-50, b_frames=2)
    seq = _decode_all_bgr(p)
    assert len(seq) == 55
    t_ns = video.generate_timestamps(p, 50.0)
    exact = (1, 20, 54)

    def check(timestamps):
        for target in range(len(seq)):
            actual, frame = video.read_frame_at(p, target, timestamps=timestamps)
            assert np.array_equal(frame, seq[actual]), (target, actual)
            if target in exact:
                assert actual == target

    check(None)  # packet-index rung

    def boom(*a, **k):
        raise AssertionError("a map that describes the file must not need a scan")

    monkeypatch.setattr(video, "_cached_frame_pts", boom)
    check(t_ns)  # map rung


@pytest.mark.parametrize("offset_ns", [0, 7_000_000_000])
def test_previews_seek_by_a_map_that_is_rebased_and_offset(
    tmp_path, write_vfr_video, monkeypatch, offset_ns
):
    """A map is part-local from 0 plus the part's recording offset, while the file's
    PTS start wherever the container put them. The map rung converts between the two,
    so neither a late-starting file nor a second part pays the whole-file packet scan.
    """
    p = write_vfr_video(tmp_path / "v.mp4", start_ms=3000)
    seq = _decode_all_bgr(p)
    t_ns = video.generate_timestamps(p, 50.0) + offset_ns

    def boom(*a, **k):
        raise AssertionError("a map that describes the file must not need a scan")

    monkeypatch.setattr(video, "_cached_frame_pts", boom)
    for target in (0, 7, 31, 59):
        actual, frame = video.read_frame_at(p, target, timestamps=t_ns)
        assert actual == target
        assert np.array_equal(frame, seq[target])
    span = list(video.read_frames_range(p, 3, 40, timestamps=t_ns))
    assert [idx for idx, _ in span] == list(range(3, 41))
    assert all(np.array_equal(frame, seq[idx]) for idx, frame in span)
    cursor = video.FrameCursor(p, timestamps=t_ns)
    for target in (5, 6, 50, 2):
        actual, frame = cursor.frame(target)
        assert actual == target
        assert np.array_equal(frame, seq[target])
    cursor.close()
