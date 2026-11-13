"""Video container metadata, timestamp-map generation, and seek reads (Section 3,
Tenet 15).

Uses PyAV for container metadata and PTS — never the FFmpeg CLI. Sequential
pipeline reads use ``cv2.VideoCapture`` (see ``pipeline/``); this module owns
container probing, the per-frame ``t_ns`` map, and the seek-and-read helpers used
by preview / inspection paths exempted under Tenet 11 — ``read_frame_at`` for one
frame anywhere, ``read_frames_range`` for every frame of a span.

It also owns the one export writer, ``open_video_writer`` (Tenet 15): ``h264`` through
PyAV's linked libav, ``mp4v`` through ``cv2.VideoWriter``.
"""

import functools
import heapq
import itertools
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
from detail.core.video_codecs import VIDEO_CODEC_LABELS
from detail.core.video_codecs import VIDEO_CODECS

NS_PER_SECOND = 1_000_000_000

# How deep ``read_frame_at`` will decode from frame 0 when no trustworthy seek index
# exists. Decode runs at a few hundred fps, so this bounds that path to seconds. The
# only caller that can reach it is the Quick segment tool on a raw clip, itself capped
# at ``configs/quick_segment.QUICK_SEGMENT_FRAME_CAP`` (1000).
SEQUENTIAL_DECODE_FRAME_CAP = 2000


def _decoding_stream(container, threads: int | None = None):
    """The video stream of ``container``, configured to decode on more than one thread.

    **FFmpeg decodes single-threaded unless asked.** PyAV leaves ``thread_type`` at its default,
    so every decode in this package has been using one core. Reading a span on a 1080p
    part, it about halves the time a frame takes, for one setting. It is free in the
    sense that matters: no new API, no call-site changes, and nothing
    about the frames that come back differs.

    ``"AUTO"`` rather than ``"FRAME"`` or a thread count. Frame-threading has a latency cost — the
    decoder holds several frames in flight — which is wrong for a scrub asking for one frame and
    right for a span; ``AUTO`` lets FFmpeg pick per codec, and measured the same as both explicit
    settings while staying correct on a codec that only supports slice threading.

    Applied through one helper rather than at each ``av.open``: this is the same shape as the
    network-share retry that was applied per call site and so was complete on the day it was
    written (``storage/zarr_seam``). A new decoding read asks for its stream here and inherits it.

    **``threads`` caps the pool, for a reader the app holds many of at once.** FFmpeg sizes it to
    the machine — about one decode thread per core — and those threads live as long as the
    container. Right for the one scrubber on a 1080p part; wrong for a montage tile, where five
    cluster windows held well over a hundred proxy decoders open and so thousands of native threads
    (:class:`FrameCursor`'s ``threads``). ``None`` leaves FFmpeg's choice.
    """
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if threads is not None:
        stream.codec_context.thread_count = int(threads)
    return stream


def probe_video(path: Path | str) -> dict:
    """Return container metadata: ``fps`` (float) and ``resolution`` ([width, height])."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        if stream.average_rate is None:
            raise ValueError(f"cannot determine fps for {path}")
        return {
            "fps": float(stream.average_rate),
            "resolution": [stream.codec_context.width, stream.codec_context.height],
        }


def part_duration_ns(t_ns, fallback_fps: float) -> int:
    """A video part's duration in nanoseconds, from its **measured** timestamps.

    The part spans from its first frame's presentation time to the end of its last
    frame's display interval — so the duration is the PTS span plus one frame interval,
    and the interval itself is measured (``span / (n - 1)``) rather than assumed. That
    makes the whole value a function of the timestamps we demuxed, which is the point:
    dividing a measured frame count by the container's *advertised* fps re-introduces
    exactly the error ``generate_timestamps`` exists to remove — Tenet 8 forbids the
    mirror image of it (``CAP_PROP_FRAME_COUNT`` = duration x advertised fps) for the same
    reason. A file whose advertised fps differs from its true rate skews every later
    part's offset by the same proportion.

    ``fallback_fps`` is used only for a **single-frame** part, which has no interval to
    measure. A part with no frames has no duration.
    """
    n = len(t_ns)
    if n == 0:
        return 0
    if n == 1:
        return int(round(NS_PER_SECOND / fallback_fps)) if fallback_fps > 0 else 0
    span = int(t_ns[-1]) - int(t_ns[0])
    return span + round(span / (n - 1))


def read_frame_at(
    video_path: Path | str,
    frame_idx: int,
    *,
    timestamps: np.ndarray | None = None,
) -> tuple[int, np.ndarray] | None:
    """Decode presentation-frame ``frame_idx`` of a video part as one BGR frame.

    The returned picture is exactly the ``frame_idx``-th frame in presentation
    order — the same indexing the sequential detection pass assigns (Tenets 8/11)
    — so preview overlays keyed by ``frame_idx`` line up with the image. The index
    is never guessed: retrieval draws on the first of three sources that can be
    trusted, and there is no best-effort path.

    1. ``timestamps`` — the part's authoritative ``frame_idx -> t_ns`` map
       (``TimestampMap.values``), when the caller has one. Seeks by ``t_ns``, so it
       skips the whole-file packet scan below entirely: on a 1 GB / 108k-frame part
       that is ~0.1s instead of ~11s, and it scales with frame count rather than
       bytes read. Skipped when the map doesn't describe *this* file's PTS — a
       map is the frame index a run's rows are numbered in (§4).
    2. The packet-PTS index (``_cached_frame_pts``) — when no map is passed, or the
       map turned out not to describe this file.
    3. A bounded sequential decode from frame 0 when neither is available — exact
       by construction, since counting decoded frames *is* the indexing detection
       assigns.

    This deliberately avoids cv2's ``CAP_PROP_POS_FRAMES`` seek, which converts the
    frame number to a seek *time* using the container's *advertised* fps: on a file
    whose advertised fps differs from its true PTS rate that lands on the wrong
    frame **and** reports the requested index back, desyncing overlays from the
    image by an amount that grows with the frame number. Permitted only in preview
    / inspection paths per Tenet 11.

    Args:
        timestamps: The part's ``t_ns`` per frame, if the caller has it. Passed as a
            plain array rather than read here so this module stays free of
            ``storage`` imports.

    Returns:
        ``(actual_frame_idx, BGR ndarray)`` on success — ``actual_frame_idx`` is
        the true presentation index of the decoded frame (``== frame_idx`` except
        on the packet-PTS path, where a keyframe-snap can land elsewhere);
        ``None`` if the frame is out of range or can't be decoded.

    Raises:
        FileNotFoundError: If the video file is absent — distinguishes a
            video-excluded export (the blob is on the source machine) from a
            decode failure, so previews can show a clear message.
        ValueError: If no trustworthy index is available and ``frame_idx`` is
            deeper than ``SEQUENTIAL_DECODE_FRAME_CAP`` — decoding that far would
            take minutes, so it fails loudly instead.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(
            f"video not present: {video_path} — likely a video-excluded export; "
            "bring the source video to this machine to preview frames"
        )
    frame_idx = int(frame_idx)
    if frame_idx < 0:
        return None
    if timestamps is not None:
        if frame_idx >= len(timestamps):
            return None
        found = _read_frame_by_t_ns(video_path, frame_idx, timestamps)
        if found is not None:
            return found
        # The map doesn't describe this file's PTS, so its arithmetic can't be
        # trusted — try the next rung rather than guess. Two ways this happens: a
        # map is authoritative for this part's frame -> t_ns by design
        # (§4), and ``_timestamps_via_decode`` synthesises ``i / fps`` for frames
        # genuinely missing a PTS.
    stat = os.stat(video_path)
    pts_index = _cached_frame_pts(str(video_path), stat.st_size, stat.st_mtime_ns)
    if pts_index is None:
        return _read_frame_by_sequential_decode(video_path, frame_idx)
    if frame_idx >= len(pts_index):
        return None
    target_pts = int(pts_index[frame_idx])
    with av.open(str(video_path)) as container:
        stream = _decoding_stream(container)
        # Seek to the keyframe at/before the target PTS, then decode forward in
        # presentation order until we reach it (PyAV reorders B-frames for us).
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        decoded = None
        for frame in container.decode(stream):
            decoded = frame
            if frame.pts is not None and frame.pts >= target_pts:
                break
        if decoded is None:
            return None
        actual = (
            int(np.searchsorted(pts_index, int(decoded.pts), side="left"))
            if decoded.pts is not None
            else frame_idx
        )
        actual = min(max(actual, 0), len(pts_index) - 1)
        return actual, decoded.to_ndarray(format="bgr24")


_log = logging.getLogger(__name__)


class _Pending:
    """A release waiting on :data:`_IDLE_RELEASES`; ``cancel`` is a flag, so it never blocks."""

    __slots__ = ("cancelled",)

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _IdleReleases:
    """One thread running every open :class:`FrameCursor`'s pending idle release.

    A ``threading.Timer`` per open cursor is a native thread per open cursor, and a montage holds
    one cursor per tile: 165 timer threads with five cluster windows open. One thread waits on the
    earliest deadline instead, and exits when nothing is pending, so a process with no open
    cursor has no thread here at all.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._heap: list[tuple[float, int, _Pending, Callable[[], None]]] = []
        self._order = itertools.count()
        self._thread: threading.Thread | None = None

    def schedule(self, delay_s: float, release: Callable[[], None]) -> _Pending:
        pending = _Pending()
        with self._cond:
            due = time.monotonic() + delay_s
            heapq.heappush(self._heap, (due, next(self._order), pending, release))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="FrameCursor-idle-release", daemon=True
                )
                self._thread.start()
            self._cond.notify()
        return pending

    def _next_due(self) -> tuple[_Pending, Callable[[], None]] | None:
        """Wait for the earliest live release to fall due; None when nothing is left. Lock held."""
        while True:
            while self._heap and self._heap[0][2].cancelled:
                heapq.heappop(self._heap)
            if not self._heap:
                self._thread = None
                return None
            wait = self._heap[0][0] - time.monotonic()
            if wait <= 0:
                _due, _order, pending, release = heapq.heappop(self._heap)
                return pending, release
            self._cond.wait(wait)

    def _run(self) -> None:
        while True:
            with self._cond:
                due = self._next_due()
            if due is None:
                return
            pending, release = due
            if pending.cancelled:
                continue
            # Isolation boundary: one cursor's failed close must not end every other cursor's
            # release, which would leave their files held for the life of the process.
            try:
                release()
            except Exception:  # noqa: BLE001
                _log.exception("releasing an idle video cursor failed")


_IDLE_RELEASES = _IdleReleases()


class FrameCursor:
    """One video, read frame by frame, keeping its place.

    **The cost :func:`read_frame_at` cannot avoid is that it forgets.** Every call opens the
    container, seeks to the keyframe at or before the target, and decodes forward to it — the
    right shape for a scrub, which asks for one frame anywhere, and ruinous for playback, which
    asks for the *next* frame forty times a second. On a 1080p part a frame costs over
    an order of magnitude more that way than read in sequence, so a surface playing at
    1x could show only about half its frames, and 4x was unreachable.

    A cursor holds the container open and walks forward when the next request is at or after
    where the decoder already stands. Asking in order costs one decode; asking backwards, or far
    ahead, re-seeks — which is exactly what ``read_frame_at`` would have done anyway.

    **The scrubber's dropped frames cost nothing here.** Its decode is latest-only, so playback
    under load asks for 100 and then 103; the cursor walks three frames rather than re-seeking.

    **Only the timestamp-map rung.** ``read_frame_at`` documents three sources it will trust, in
    order; this handles the first and returns ``None`` for the rest so the caller falls back to
    it. Deliberate rather than unfinished: the map is what every GUI surface passes, and a cursor
    that re-derived the packet index and the counted-decode fallback would be a second copy of
    the trust ladder — the kind of copy that agrees with the original until the day it does not.

    The lock is not incidental: a ``FrameScrubber`` decodes on pool workers, one at a time but
    not always the same one.

    **It lets go of the file when nobody is asking.** An open container is an open file handle,
    and on Windows an open handle pins the file: it cannot be moved, renamed or deleted until
    the handle is closed. A cursor that held its container for its owner's lifetime turned one
    geometry preview on a recording page into "the file is in use" for as long as the page was
    open — the page whose *Locate video* exists for the user who just moved it. So
    :attr:`IDLE_CLOSE_S` after the last request the container is closed, and the next request
    opens it again. Playback asks far more often than that and never notices; the first frame
    after a pause costs one seek (~33 ms on the 1080p part), which is what a scrub after a
    pause pays anyway. Releasing here rather than in each owner is the point: a ``Recording``,
    a run preview and whatever holds a cursor next all inherit it.
    ``tests/storage/test_video_handle_release.py`` is the guard.

    **``threads`` bounds what an open cursor costs in native threads.** FFmpeg gives an open
    decoder a pool sized to the machine, and the conversion to BGR another: about two
    threads per core per open cursor, for as long as it is open. One cursor on a 1080p
    part is worth that (see :func:`_decoding_stream`); a montage holds one per tile, and
    five cluster windows ran well over a hundred open at once — thousands of threads and
    ``can't start new thread``. A caller that
    holds many passes ``threads=`` (``api/previews.TILE_DECODE_THREADS``), which caps the decoder
    and converts on the calling thread; ``None`` leaves FFmpeg's choice. The idle releases share one thread (:class:`_IdleReleases`) for the same reason.
    ``tests/storage/test_video_threads.py`` is the guard.
    """

    #: How far ahead to walk before re-seeking instead. A seek costs the walk from the preceding
    #: keyframe, so walking is cheaper right up until the distance exceeds about half a group of
    #: pictures. On a 1080p part the break-even is near twenty frames walked. 64 keeps a
    #: margin for footage
    #: with longer groups, and being wrong here costs a seek, never a wrong picture.
    FORWARD_WALK_CAP = 64

    #: Seconds without a request before the cursor closes its container. Far longer than any gap
    #: in playback (0.25x of a 10 fps part asks every 0.4 s) and shorter than switching to a file
    #: manager and finding the file. Being short costs one seek; being long costs the user the
    #: file.
    IDLE_CLOSE_S = 2.0

    def __init__(
        self,
        video_path: Path | str,
        *,
        timestamps: np.ndarray | None = None,
        threads: int | None = None,
    ) -> None:
        self._path = Path(video_path)
        self._timestamps = timestamps
        #: Decode threads while open; None is FFmpeg's machine-sized pool.
        self._threads = None if threads is None else max(1, int(threads))
        #: The conversion to BGR is one frame per request, so a capped cursor converts on the
        #: calling thread: at proxy size that is faster than handing it to
        #: FFmpeg's pool, which would otherwise be another ``cores`` threads held while open.
        self._convert_threads = None if threads is None else 1
        self._lock = threading.Lock()
        self._container = None
        self._stream = None
        self._time_base = None
        #: Map value minus file time for this part (``_map_shift_ns``); set on open.
        self._shift_ns = 0
        #: The one decode generator. `container.decode(stream)` hands back a *new* generator each
        #: call, so asking per request re-enters the decoder instead of resuming it — right
        #: frames on a long file, `av.error.EOFError` on a short one. Keeping the place means
        #: keeping this.
        self._frames = None
        #: Map index of the frame the decoder will yield next; None when not positioned.
        self._next_idx: int | None = None
        #: When the last request finished, on ``_clock``; an attribute so a test can drive it.
        self._clock = time.monotonic
        self._last_used = 0.0
        #: The pending idle release, armed while the container is open and at most one at a time
        #: — one per open cursor, not per frame, and all of them on one thread.
        self._idle_timer: _Pending | None = None

    def frame(self, frame_idx: int) -> tuple[int, np.ndarray] | None:
        """``(actual_frame_idx, BGR ndarray)``, exactly as :func:`read_frame_at` returns it.

        Same contract, same guarantees, same exceptions — a caller cannot tell the two apart
        except by how long it takes.
        """
        if not self._path.is_file():
            # The same refusal `read_frame_at` makes, and for the same reason: a video-excluded
            # export is a thing to explain, not a decode failure. Raised here rather than left to
            # `av.open`, whose own error says only that a path does not exist.
            raise FileNotFoundError(
                f"video not present: {self._path} — likely a video-excluded export; "
                "bring the source video to this machine to preview frames"
            )
        if self._timestamps is None:
            return read_frame_at(self._path, frame_idx)
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= len(self._timestamps):
            return None
        with self._lock:
            try:
                found = self._walk(frame_idx)
                if found is None:
                    found = self._seek_and_walk(frame_idx)
            except av.error.EOFError:
                # The decoder was drained and will refuse the next packet whatever we seek to.
                # Dropping it is the whole remedy; the retry below opens a fresh one.
                self._close()
                found = None
            finally:
                # Whatever happened — including an exception out of the decoder — a container
                # still open has its release pending.
                self._touch()
            if found is not None:
                return found
            self._close()
        # The map does not describe this file's PTS. Hand back to the function that owns the
        # other two rungs rather than inventing a third here.
        return read_frame_at(self._path, frame_idx, timestamps=self._timestamps)

    def close(self) -> None:
        """Let go of the container. Safe to call twice, and safe to keep using the cursor."""
        with self._lock:
            self._close()

    @property
    def is_open(self) -> bool:
        """Whether the cursor holds its container — and so the file — right now."""
        return self._container is not None

    def __del__(self) -> None:  # noqa: D105 — a file handle must not outlive its holder
        try:
            self._close()
        except Exception:  # noqa: BLE001 — interpreter teardown reports nowhere
            pass

    # -- internals ----------------------------------------------------------------------- #
    def _touch(self) -> None:
        """Note a request, and make sure an open container has its release pending. Lock held."""
        self._last_used = self._clock()
        if self._container is not None and self._idle_timer is None:
            self._arm_idle_release(self.IDLE_CLOSE_S)

    def _arm_idle_release(self, delay_s: float) -> None:
        self._idle_timer = _IDLE_RELEASES.schedule(delay_s, self._release_if_idle)

    def _release_if_idle(self) -> None:
        """Close the container if it has gone unused for ``IDLE_CLOSE_S``; else wait the rest.

        Re-arming for the remainder rather than restarting the timer on every request is what
        keeps playback at one pending release per idle window instead of one per frame.
        """
        with self._lock:
            self._idle_timer = None
            if self._container is None:
                return
            remaining = self._last_used + self.IDLE_CLOSE_S - self._clock()
            if remaining > 0:
                self._arm_idle_release(remaining)
            else:
                self._close()

    def _walk(self, frame_idx: int) -> tuple[int, np.ndarray] | None:
        """Decode forward from where the decoder stands, if that is the cheap way there."""
        if self._container is None or self._next_idx is None:
            return None
        ahead = frame_idx - self._next_idx
        if ahead < 0 or ahead > self.FORWARD_WALK_CAP:
            return None  # backwards, or far enough that seeking wins
        return self._decode_until(int(self._timestamps[frame_idx]))

    def _seek_and_walk(self, frame_idx: int) -> tuple[int, np.ndarray] | None:
        """Open if needed, seek to the keyframe at or before the target, decode forward to it."""
        target_t_ns = int(self._timestamps[frame_idx])
        if self._container is None:
            container = av.open(str(self._path))
            stream = _decoding_stream(container, self._threads)
            shift_ns = _map_shift_ns(stream, self._timestamps)
            if stream.time_base is None or shift_ns is None:
                container.close()
                return None
            self._container, self._stream = container, stream
            self._time_base = stream.time_base
            self._shift_ns = shift_ns
        seek_target = int(
            (target_t_ns - self._shift_ns) / (float(self._time_base) * NS_PER_SECOND)
        )
        self._container.seek(
            seek_target, stream=self._stream, backward=True, any_frame=False
        )
        self._frames = self._container.decode(self._stream)
        self._next_idx = 0  # unknown until the first decoded frame says so
        return self._decode_until(target_t_ns)

    def _decode_until(self, target_t_ns: int) -> tuple[int, np.ndarray] | None:
        """Pull frames until one lands at or past ``target_t_ns``, keeping the place.

        The alignment check is the one :func:`_read_frame_by_t_ns` makes, and for the same
        reason: a map that does not describe this file's PTS must produce a refusal, never an
        approximate frame.
        """
        timestamps = self._timestamps
        if self._frames is None:
            return None
        for frame in self._frames:
            if frame.pts is None:
                self._close()
                return None
            t_ns = _pts_ns(frame.pts, self._time_base) + self._shift_ns
            actual = int(np.searchsorted(timestamps, t_ns, side="left"))
            if actual >= len(timestamps) or int(timestamps[actual]) != t_ns:
                self._close()  # not PTS-aligned — do not guess, and do not stay positioned
                return None
            self._next_idx = actual + 1
            if t_ns >= target_t_ns:
                return actual, frame.to_ndarray(
                    format="bgr24", threads=self._convert_threads
                )
        self._close()  # ran off the end of the stream
        return None

    def _close(self) -> None:
        self._frames = None
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self._container is not None:
            try:
                self._container.close()
            finally:
                self._container = None
                self._stream = None
                self._time_base = None
                self._next_idx = None


def read_frames_range(
    video_path: Path | str,
    first_frame: int,
    last_frame: int,
    *,
    timestamps: np.ndarray | None = None,
    step: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode a contiguous span of a part in **one pass** — one open, one seek, then
    sequential decode forward.

    ``read_frame_at`` opens the container, seeks to a keyframe and decodes forward for
    *every* frame it is asked for. That is the right shape for a scrub — one frame,
    anywhere, right now — and the wrong one for a clip, which is every frame of a span.
    On a 1080p part a clip costs over an order of magnitude more that way than in
    one pass. Tenet 11 permits the seek because this is a
    preview path; what it does not permit is N seeks in a loop.

    ``step`` skips frames without skipping the decode — the container has to be walked
    either way, so a stride saves the colour conversion, not the parsing.

    Falls back to ``read_frame_at`` per frame when the timestamp map cannot be trusted
    for this file, which keeps the trust rules identical to the single-frame path
    rather than inventing a looser one for playback.

    Yields:
        ``(actual_frame_idx, BGR ndarray)`` in presentation order.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(
            f"video not present: {video_path} — likely a video-excluded export; "
            "bring the source video to this machine to preview frames"
        )
    first_frame, last_frame = int(first_frame), int(last_frame)
    step = max(1, int(step))
    if first_frame < 0 or last_frame < first_frame:
        return
    if timestamps is not None and last_frame < len(timestamps):
        streamed = yield from _read_span_by_t_ns(
            video_path, first_frame, last_frame, step, timestamps
        )
        if streamed:
            return
    for idx in range(first_frame, last_frame + 1, step):
        found = read_frame_at(video_path, idx, timestamps=timestamps)
        if found is not None:
            yield found


def read_selected_frames(
    video_path: Path | str,
    frames: np.ndarray,
    *,
    timestamps: np.ndarray,
    size: tuple[int, int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode a whole part **once, from the start**, and hand back only ``frames``.

    The preview build's read (§3 *Video preview cache*): it keeps about one frame in four of
    a part whose rate is not a multiple of the preview's, at an irregular stride, so
    ``read_frames_range``'s fixed ``step`` cannot
    say which. Every packet is still decoded (H.264 cannot skip one), but only a kept frame
    is converted, and with ``size=(width, height)`` it is shrunk by swscale in the same call —
    a full-resolution BGR copy of every frame was the larger cost. No seek at all (Tenet 11).

    Frames are identified by their ``t_ns`` against the timestamp map, exactly as
    ``_read_span_by_t_ns`` does; when the map does not describe this file, the read falls
    back to ``read_frames_range`` and resizes with cv2, so the trust rules stay the ones every
    other reader uses.

    Yields:
        ``(frame_idx, BGR ndarray)`` for each wanted frame, in order.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"video not present: {video_path}")
    wanted = np.unique(np.asarray(frames, dtype=np.int64))
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if not len(wanted) or not len(timestamps):
        return
    keep = set(int(i) for i in wanted)
    last_wanted = int(wanted[-1])
    with av.open(str(video_path)) as container:
        stream = _decoding_stream(container)
        time_base = stream.time_base
        shift_ns = _map_shift_ns(stream, timestamps)
        aligned = time_base is not None and shift_ns is not None
        checked = False
        if aligned:
            for frame in container.decode(stream):
                if frame.pts is None:
                    if checked:
                        # Mid-file, a frame with no time cannot be placed, and ending here
                        # would hand back a silently short read as if it were the whole part.
                        raise ValueError(
                            f"{Path(video_path).name}: a decoded frame carries no timestamp "
                            f"after frame {actual}; the part cannot be read frame-exactly"
                        )
                    aligned = False
                    break
                t_ns = _pts_ns(frame.pts, time_base) + shift_ns
                actual = int(np.searchsorted(timestamps, t_ns, side="left"))
                if actual >= len(timestamps) or int(timestamps[actual]) != t_ns:
                    if checked:
                        raise ValueError(
                            f"{Path(video_path).name}: frame at {t_ns} ns is not in the "
                            "part's timestamp map — the map no longer describes the file"
                        )
                    aligned = False
                    break
                checked = True
                if actual in keep:
                    if size is None:
                        yield actual, frame.to_ndarray(format="bgr24")
                    else:
                        yield actual, frame.to_ndarray(
                            width=int(size[0]),
                            height=int(size[1]),
                            format="bgr24",
                            interpolation="AREA",
                        )
                if actual >= last_wanted:
                    return
            if aligned:
                return
    for idx, bgr in read_frames_range(
        video_path, 0, last_wanted, timestamps=timestamps
    ):
        if idx not in keep:
            continue
        if size is not None and (bgr.shape[1], bgr.shape[0]) != tuple(size):
            bgr = cv2.resize(
                bgr, tuple(int(v) for v in size), interpolation=cv2.INTER_AREA
            )
        yield idx, bgr


def _read_span_by_t_ns(
    video_path: Path | str,
    first_frame: int,
    last_frame: int,
    step: int,
    timestamps: np.ndarray,
) -> Iterator[tuple[int, np.ndarray]]:
    """One seek to the span's start, then decode forward to its end.

    Alignment is checked on the first frame at or after the target, exactly as
    ``_read_frame_by_t_ns`` checks it, and the span is abandoned before anything is
    yielded if the map does not describe this file — the caller then falls back rather
    than trusting a mismatched pair. Returns whether the span was streamed.
    """
    start_t_ns, end_t_ns = int(timestamps[first_frame]), int(timestamps[last_frame])
    with av.open(str(video_path)) as container:
        stream = _decoding_stream(container)
        time_base = stream.time_base
        shift_ns = _map_shift_ns(stream, timestamps)
        if time_base is None or shift_ns is None:
            return False
        seek_target = int((start_t_ns - shift_ns) / (float(time_base) * NS_PER_SECOND))
        container.seek(seek_target, stream=stream, backward=True, any_frame=False)
        checked = False
        for frame in container.decode(stream):
            if frame.pts is None:
                return checked
            t_ns = _pts_ns(frame.pts, time_base) + shift_ns
            if t_ns < start_t_ns:
                continue  # the seek landed on the keyframe at or before the span
            if t_ns > end_t_ns:
                return True
            actual = int(np.searchsorted(timestamps, t_ns, side="left"))
            if actual >= len(timestamps) or int(timestamps[actual]) != t_ns:
                return checked  # not PTS-aligned — don't guess
            checked = True
            if (actual - first_frame) % step == 0:
                yield actual, frame.to_ndarray(format="bgr24")
    return checked


def _read_frame_by_t_ns(
    video_path: Path | str, frame_idx: int, timestamps: np.ndarray
) -> tuple[int, np.ndarray] | None:
    """Seek by the frame's ``t_ns`` from the part's timestamp map.

    Matching happens in ``t_ns`` space, using the same formula the map was built
    with, so it survives the rounding that a ``t_ns -> pts`` inversion would risk.
    The map is part-local from 0 plus the part's recording offset, and the file's
    PTS start wherever the container starts them; ``_map_shift_ns`` is the one
    constant between the two.
    Returns None when the decoded frame's ``t_ns`` is absent from the map — proof
    the map isn't PTS-aligned — so the caller can fall back rather than trust a
    mismatched pair.
    """
    target_t_ns = int(timestamps[frame_idx])
    with av.open(str(video_path)) as container:
        stream = _decoding_stream(container)
        time_base = stream.time_base
        shift_ns = _map_shift_ns(stream, timestamps)
        if time_base is None or shift_ns is None:
            return None
        seek_target = int((target_t_ns - shift_ns) / (float(time_base) * NS_PER_SECOND))
        container.seek(seek_target, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                return None
            t_ns = _pts_ns(frame.pts, time_base) + shift_ns
            if t_ns >= target_t_ns:
                actual = int(np.searchsorted(timestamps, t_ns, side="left"))
                if actual >= len(timestamps) or int(timestamps[actual]) != t_ns:
                    return None  # not PTS-aligned — don't guess
                return actual, frame.to_ndarray(format="bgr24")
    return None


def _read_frame_by_sequential_decode(
    video_path: Path | str, frame_idx: int
) -> tuple[int, np.ndarray] | None:
    """Decode from frame 0, counting, until ``frame_idx``.

    The last-resort retrieval for a file whose PTS can't be trusted and whose map
    isn't usable. Exact by construction — counting decoded frames is precisely the
    indexing the sequential detection pass assigns — but O(frame_idx), so it is
    capped rather than allowed to silently spend minutes.

    Raises:
        ValueError: If ``frame_idx`` exceeds ``SEQUENTIAL_DECODE_FRAME_CAP``.
    """
    if frame_idx > SEQUENTIAL_DECODE_FRAME_CAP:
        raise ValueError(
            f"cannot retrieve frame {frame_idx} of {Path(video_path).name}: its "
            "packet timestamps are unreliable, so the frame can only be reached by "
            f"decoding from the start, and that is capped at "
            f"{SEQUENTIAL_DECODE_FRAME_CAP} frames. Pass the part's timestamp map "
            "(preview paths do) to seek directly."
        )
    with av.open(str(video_path)) as container:
        stream = _decoding_stream(container)
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                return frame_idx, frame.to_ndarray(format="bgr24")
    return None


def generate_timestamps(
    path: Path | str,
    fps: float,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Per-frame ``t_ns`` for one part (part-local, starting at 0).

    **The map describes the frames a decoder actually shows**, which is what makes
    index ``i`` the ``i``-th frame ``cv2.VideoCapture`` returns (Tenets 8/11). Two
    things in a container are not shown frames, and both are removed here:

    - **Packets an edit list discards** (``Packet.is_discard``). A trimmed MP4 keeps
      the cut frames' packets so the decoder can reference them; cv2 and PyAV's
      decode never output them. Counting them made the map longer than the video
      and shifted every row a run wrote by the number cut.
    - **The file's start time.** Presentation timestamps need not start at 0 (an
      MPEG-TS or a camera file typically does not, and B-frame reordering alone
      offsets the first PTS). The first shown frame's time is subtracted, so
      ``t_ns[0] == 0`` on every path and a part's recording-global offset is the
      only thing that places it in the recording.

    Each value is rounded to integer ns *before* the origin is subtracted
    (``_pts_ns``), so a reader that re-derives a frame's ``t_ns`` from its PTS
    (``_map_shift_ns``) reproduces the map exactly.

    Reads PTS from the container's *packets* (no full decode) — fast even on
    large files. Demux is trusted on its own integrity: every video packet has a
    PTS (the trailing flush packet aside) and, sorted, the PTS are strictly
    increasing. Only when that can't be established does it fall back to decoding
    every frame — the conservative ground truth — so the ``frame_idx → t_ns``
    alignment the pipeline depends on (Tenets 8/11) is never silently wrong.

    (We deliberately do *not* gate on cv2's ``CAP_PROP_FRAME_COUNT``: that value
    is an estimate that's commonly off by a frame on real H.264, which would
    spuriously trigger a full-decode pass on every large import.)

    ``progress`` reports a ``0..1`` fraction through the file as it scans.
    """
    demuxed = _timestamps_via_demux(path, progress)
    if demuxed is not None:
        return demuxed
    return _timestamps_via_decode(path, fps, progress)


def _pts_ns(pts: int, time_base) -> int:
    """A PTS in integer nanoseconds — the one rounding every map and every reader uses."""
    return int(round(float(pts * time_base) * NS_PER_SECOND))


def _map_shift_ns(stream, timestamps: np.ndarray) -> int | None:
    """``map t_ns − file t_ns`` for a part's timestamp map against its own stream.

    A map is part-local from 0 (``generate_timestamps`` subtracts the first shown
    frame's time) plus the part's offset in the recording, so a frame's map value is
    its file time plus one constant: ``timestamps[0]`` minus the stream's start. A
    seek converts with it in both directions.

    The start is ``stream.start_time`` — FFmpeg's first presentation time, which
    already honours an edit list — read from the open container, so it costs no
    I/O. It is not *proved* equal to the origin the import subtracted, and it does
    not need to be: a shift that is wrong for any reason, including a map that does
    not describe this file, fails the per-frame ``t_ns`` membership check after the
    seek, and the caller falls back to the packet index. None when there is no start
    time to use.
    """
    if stream.start_time is None or stream.time_base is None or not len(timestamps):
        return None
    return int(timestamps[0]) - _pts_ns(stream.start_time, stream.time_base)


def _stream_duration_pts(stream) -> int | None:
    """Stream duration in ``time_base`` units, for a scan-progress denominator."""
    if stream.duration is not None:
        return int(stream.duration)
    if stream.frames and stream.average_rate and stream.time_base:
        return int(stream.frames / (stream.average_rate * stream.time_base))
    return None


def _sorted_video_pts(
    stream,
    container,
    progress: Callable[[float], None] | None = None,
    duration: int | None = None,
) -> list[int] | None:
    """PTS (``time_base`` units) of the frames a decoder *shows*, sorted into
    presentation order, or None when the PTS can't be trusted for index alignment —
    any non-flush packet without a PTS, or PTS not strictly increasing once sorted
    (duplicates). The trailing decoder-flush packet (empty, PTS ``None``) is skipped,
    and so is every packet an edit list discards (``is_discard``): the decoder needs
    it as a reference but never outputs it, so counting it would put a frame in the
    index that cv2 never returns. Both the import map and the preview seek index
    (``_cached_frame_pts``) come through here, so they cannot disagree on it."""
    pts: list[int] = []
    for i, packet in enumerate(container.demux(stream)):
        if packet.is_discard:  # edit-list trimmed: decoded as a reference, never shown
            continue
        if packet.pts is None:
            if packet.size == 0:  # decoder-flush packet at end of stream
                continue
            return None  # a real packet with no PTS — can't align
        pts.append(int(packet.pts))
        if progress is not None and duration and i % 500 == 0:
            progress(min(packet.pts / duration, 1.0))
    if not pts:
        return None
    pts.sort()
    if any(b <= a for a, b in zip(pts, pts[1:])):  # not strictly increasing
        return None
    return pts


class FrameCountMismatchError(ValueError):
    """A part's timestamp map and its video disagree on how many frames it shows.

    Every row a sequential pass writes is placed by ``map[frame_idx]``, so a map of a
    different length places every row at the wrong time — not just the tail.
    """


def shown_frame_count(path: Path | str) -> int | None:
    """How many frames a decoder shows for this file, from its packets (no decode).

    The count ``generate_timestamps`` gives a map on its demux path. None when the
    packets cannot be trusted for it (``_sorted_video_pts``) — the map then came from a
    full decode, and nothing short of another one can recount it.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = _sorted_video_pts(stream, container)
    return None if pts is None else len(pts)


@functools.lru_cache(maxsize=8)
def _cached_frame_pts(path_str: str, size: int, mtime_ns: int) -> np.ndarray | None:
    """Sorted per-presentation-frame PTS (stream ``time_base`` units) for one file,
    the seek index behind ``read_frame_at``; None when packet PTS can't be trusted
    for index alignment (``_sorted_video_pts``). Cached by ``(path, size, mtime)``
    so repeated preview seeks on a long video pay the packet scan only once."""
    with av.open(path_str) as container:
        stream = container.streams.video[0]
        if stream.time_base is None:
            return None
        pts = _sorted_video_pts(stream, container)
    return np.asarray(pts, dtype=np.int64) if pts is not None else None


def _timestamps_via_demux(
    path: Path | str, progress: Callable[[float], None] | None = None
) -> np.ndarray | None:
    """Per-frame ``t_ns`` from packet PTS, sorted into presentation order and
    re-based so the first shown frame is 0.

    Returns None when PTS can't be trusted for index alignment — a missing
    ``time_base``, any non-flush packet without a PTS, or PTS that aren't
    strictly increasing once sorted (duplicates) — so the caller decodes
    instead. The trailing flush packet (empty, PTS ``None``) is skipped.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            return None
        duration = _stream_duration_pts(stream)
        pts = _sorted_video_pts(stream, container, progress, duration)
    if pts is None:
        return None
    if progress is not None:
        progress(1.0)
    t_ns = np.asarray([_pts_ns(p, time_base) for p in pts], dtype=np.int64)
    return t_ns - t_ns[0]


def _timestamps_via_decode(
    path: Path | str, fps: float, progress: Callable[[float], None] | None = None
) -> np.ndarray:
    """Per-frame ``t_ns`` by decoding every frame (presentation order), re-based so
    the first frame is 0. Uses each frame's PTS, falling back to ``frame_idx / fps``
    for any frame missing one.

    The origin is the first frame that has a PTS, placed at its own ``i / fps``
    position — so a PTS-bearing frame and a synthesised one share a time base, and a
    file whose very first frame lacks a PTS still starts at 0."""
    out: list[int] = []
    origin_ns: int | None = None
    with av.open(str(path)) as container:
        stream = _decoding_stream(container)
        time_base = stream.time_base
        duration = _stream_duration_pts(stream)
        for i, frame in enumerate(container.decode(stream)):
            synthetic_ns = int(round(i / fps * NS_PER_SECOND))
            if frame.pts is not None and time_base is not None:
                pts_ns = _pts_ns(frame.pts, time_base)
                if origin_ns is None:
                    origin_ns = pts_ns - synthetic_ns
                out.append(pts_ns - origin_ns)
            else:
                out.append(synthetic_ns)
            if (
                progress is not None
                and duration
                and frame.pts is not None
                and i % 200 == 0
            ):
                progress(min(frame.pts / duration, 1.0))
    if progress is not None:
        progress(1.0)
    arr = np.asarray(out, dtype=np.int64)
    # Same rebasing as _timestamps_via_demux, and for the same reason: the first
    # decoded frame's raw PTS is this stream's zero point, not necessarily 0 itself.
    if len(arr):
        arr = arr - arr[0]
    return arr


@functools.cache
def h264_available() -> bool:
    """Whether this machine's PyAV can encode H.264 (``libx264``) — asked once per process,
    since a build does not gain an encoder mid-run."""
    try:
        av.codec.Codec("libx264", "w")
    except (av.codec.codec.UnknownCodecError, ValueError):
        return False
    return True


def written_codec(codec: str = DEFAULT_VIDEO_CODEC) -> str:
    """The codec an export asking for ``codec`` writes on this machine.

    Raises:
        ValueError: ``codec`` is not one of :data:`VIDEO_CODECS`.
    """
    if codec not in VIDEO_CODECS:
        raise ValueError(
            f"unknown video codec {codec!r}; choose one of {', '.join(VIDEO_CODECS)}"
        )
    if codec == "h264" and not h264_available():
        return "mp4v"
    return codec


def codec_note(codec: str = DEFAULT_VIDEO_CODEC) -> str | None:
    """What a surface tells the user when ``codec`` cannot be written here, else ``None``.

    The fallback is never silent: before ``h264`` went through PyAV, every MP4 export on a
    Linux cv2 wheel quietly became ``mp4v`` and nobody knew until PowerPoint refused it.
    """
    written = written_codec(codec)
    if written == codec:
        return None
    return (
        f"{VIDEO_CODEC_LABELS[codec]} is not available on this machine (its PyAV has no "
        f"libx264 encoder), so video is written as {VIDEO_CODEC_LABELS[written]}, which "
        "PowerPoint may not play."
    )


class _H264Writer:
    """An H.264 ``.mp4`` written through PyAV's linked ``libx264`` — the ``cv2.VideoWriter``
    shape (``write`` a BGR frame, ``release``), so a caller never knows which it holds.

    High profile, ``yuv420p``, CRF 23, ``+faststart``: what PowerPoint, QuickTime and every
    browser play. ``yuv420p`` needs even dimensions, so an odd last row or column is dropped —
    one pixel nobody sees, against a file nothing can open.

    Preset ``veryfast``, because a whole-recording export is where the time shows. Measured in
    WSL on 300 frames of 1280x720 footage (median of 3 alternating rounds, under load):
    ``medium`` 72 fps / 2.27 MB, ``veryfast`` 117 fps / 1.42 MB, ``mp4v`` 153 fps / 5.02 MB.
    """

    def __init__(self, out_path: Path | str, fps: float, width: int, height: int):
        self._width, self._height = width // 2 * 2, height // 2 * 2
        if self._width < 2 or self._height < 2:
            raise ValueError(f"a {width}x{height} frame is too small to encode")
        # PyAV opens the file only when the first packet is muxed; creating it here makes an
        # unwritable destination fail at open, where the caller handles it, not mid-export.
        with open(out_path, "wb"):
            pass
        self._container = av.open(
            str(out_path), "w", format="mp4", options={"movflags": "+faststart"}
        )
        stream = self._container.add_stream(
            "libx264", rate=Fraction(float(fps)).limit_denominator(1001)
        )
        stream.width, stream.height, stream.pix_fmt = (
            self._width,
            self._height,
            "yuv420p",
        )
        stream.options = {"crf": "23", "preset": "veryfast", "profile": "high"}
        self._stream = stream
        self._open = True

    def isOpened(self) -> bool:  # noqa: N802 — the cv2.VideoWriter spelling
        return self._open

    def write(self, frame_bgr: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame_bgr[: self._height, : self._width]),
            format="bgr24",
        )
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def release(self) -> None:
        if not self._open:
            return
        self._open = False
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()


def open_video_writer(
    out_path: Path | str,
    fps: float,
    width: int,
    height: int,
    codec: str = DEFAULT_VIDEO_CODEC,
):
    """Open an ``.mp4`` writer at ``codec`` — the one way an export writes video (Tenet 15).

    Returns ``(writer, written)``: ``written`` is the codec actually used, which differs from
    ``codec`` only when this machine cannot encode H.264 (:func:`codec_note` says so in words).
    The writer has ``write(bgr_frame)`` and ``release()``.

    Raises:
        ValueError: ``codec`` is not one of :data:`VIDEO_CODECS`.
        RuntimeError: the writer could not be opened at ``out_path``.
    """
    written = written_codec(codec)
    if written == "h264":
        try:
            return _H264Writer(out_path, fps, width, height), written
        except (OSError, av.error.FFmpegError) as exc:
            raise RuntimeError(
                f"could not open an H.264 writer for {out_path}: {exc}"
            ) from exc
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*written), float(fps), (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV could not open a VideoWriter for {out_path}")
    return writer, written


class ImportStopped(Exception):
    """Raised by the copy/transcode loops when ``should_stop()`` becomes true, so a
    long import can be cancelled mid-file. The caller removes the partial output."""


def copy_with_progress(
    src: Path | str,
    dst: Path | str,
    progress: Callable[[int, int], None] | None = None,
    chunk_size: int = 1 << 23,  # 8 MiB
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Copy ``src`` → ``dst`` in chunks, calling ``progress(bytes_done, total)``
    as it goes (so a multi-GB import shows real progress). Returns total bytes.
    Raises ``ImportStopped`` if ``should_stop()`` fires (cancel mid-copy)."""
    total = os.path.getsize(src)
    done = 0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            if should_stop is not None and should_stop():
                raise ImportStopped()
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
            done += len(buf)
            if progress is not None:
                progress(done, total)
    return total
