"""An open video reader the app holds many of costs no native threads of its own.

**The class: a per-instance thread pool sized to the machine, in an object held many at once.**
FFmpeg gives every open decoder a pool of ``cores + 1`` frame threads, and PyAV's conversion to
BGR another of ``cores``; a ``threading.Timer`` per open cursor added one more. All of them live
as long as the container is open. One `FrameCursor` on a 1080p part is worth that. A montage
holds one per tile: a few montage windows held enough cursors open at once, each with a
pool sized to the machine, to reach ``RuntimeError: can't start new thread`` in the log.

**How it hid.** Python saw only the timers' threads; the FFmpeg threads are native and appear
only in ``ps -M`` / ``psutil.Process().num_threads()``. Every frame came back right.

**The rule.** A caller that holds many cursors passes ``threads=`` (`PreviewClip` passes
``previews.TILE_DECODE_THREADS``, 8), which caps the decoder and converts on the calling
thread; the idle releases of every cursor share one thread that exits when none is pending
(`video._IdleReleases`).

**The guard counts the OS's threads** (psutil, on macOS and Linux), and guards itself: a cursor
left at FFmpeg's default must be seen to cost threads, or a count that cannot see native threads
would pass everything below.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from detail.storage import video
from tests.storage.test_video import FPS
from tests.storage.test_video import _write_distinct_video
from tests.storage.test_video_handle_release import _experiment_recording

psutil = pytest.importorskip("psutil")

#: Clips a montage has open at once in the test; a real window holds more.
N_OPEN = 24
#: The stated bound per open preview clip: its capped decoder (``previews.TILE_DECODE_THREADS``,
#: 8) and nothing for its conversion or its idle release. Before the fix an open clip cost
#: ``2 * cores + 2``.
MAX_THREADS_PER_OPEN_CLIP = 8
#: The shared release thread, plus anything a first decode starts lazily.
SLACK = 4
#: A failsafe, never a budget: how long the shared release thread may take to exit.
_EXIT_TIMEOUT_S = 10.0


def _native_threads() -> int:
    try:
        return psutil.Process().num_threads()
    except (psutil.Error, NotImplementedError) as err:  # pragma: no cover — platform
        pytest.skip(f"native thread count unavailable here: {err}")


def _wait_for_threads(at_most: int) -> int:
    deadline = time.monotonic() + _EXIT_TIMEOUT_S
    while time.monotonic() < deadline:
        n = _native_threads()
        if n <= at_most:
            return n
        time.sleep(0.02)
    return _native_threads()


def _release_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "FrameCursor-idle-release"]


def _wait_for_no_release_thread() -> list[threading.Thread]:
    """Wait for the shared release thread to exit; what is still alive at the failsafe.

    A ``close`` only flags its pending release, so the thread exits on its next wake-up, not at
    once. A cursor an earlier test left open keeps it alive until that release falls due.
    """
    for thread in _release_threads():
        thread.join(_EXIT_TIMEOUT_S)
    return _release_threads()


def _cursors(path, n, threads):
    timestamps = video.generate_timestamps(path, FPS)
    cursors = [
        video.FrameCursor(path, timestamps=timestamps, threads=threads)
        for _ in range(n)
    ]
    for i, cursor in enumerate(cursors):
        assert cursor.frame(i % 8)[0] == i % 8
        assert cursor.is_open
    return cursors


def test_the_count_sees_a_default_cursors_decoder_threads(tmp_path):
    """Guard the guard: FFmpeg's machine-sized pool is visible to the count, so the bounds
    below are not passing because the count is blind to native threads."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("FFmpeg runs one thread on a one-core box; nothing to see")
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 12)
    before = _native_threads()
    cursors = _cursors(p, 4, None)
    try:
        added = _native_threads() - before
        assert added > 4, (
            f"4 cursors at FFmpeg's default added {added} native threads — the count cannot see "
            "the decoder's pool, so every bound in this module would pass with the bug in place"
        )
    finally:
        for cursor in cursors:
            cursor.close()


def test_open_cursors_with_one_thread_cost_no_thread_each_and_closing_returns_the_count(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(video.FrameCursor, "IDLE_CLOSE_S", 0.05)
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 12)
    assert not _wait_for_no_release_thread(), "an earlier release never finished"
    before = _native_threads()
    cursors = _cursors(p, N_OPEN, 1)
    added = _native_threads() - before
    assert added <= SLACK, (
        f"{N_OPEN} open single-threaded cursors added {added} native threads; zero each is the "
        f"design, {SLACK} for all of them the bound"
    )
    assert (
        len(_release_threads()) == 1
    ), "every open cursor's idle release shares one thread"
    for cursor in cursors:
        cursor.close()
    after = _wait_for_threads(at_most=before)
    assert after <= before, f"closing every cursor left {after - before} threads behind"
    assert (
        not _wait_for_no_release_thread()
    ), "the release thread outlived the last pending release"


def test_idle_releases_still_close_every_cursor_on_one_thread(tmp_path, monkeypatch):
    """The shared thread does the job each timer did: every idle cursor lets go."""
    monkeypatch.setattr(video.FrameCursor, "IDLE_CLOSE_S", 0.05)
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, 12)
    cursors = _cursors(p, 6, 1)
    deadline = time.monotonic() + _EXIT_TIMEOUT_S
    while any(c.is_open for c in cursors) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any(c.is_open for c in cursors), "an idle cursor kept its container"
    assert cursors[0].frame(5)[0] == 5, "and reopens on the next request"
    cursors[0].close()


def test_a_montage_of_open_preview_clips_stays_under_the_thread_bound(
    tmp_path, monkeypatch
):
    """Through the real holder: ``PreviewClip``, the one a montage opens per tile."""
    from detail.api import previews

    monkeypatch.setattr(video.FrameCursor, "IDLE_CLOSE_S", 0.05)
    recording, _source = _experiment_recording(tmp_path)
    recording.build_preview()
    reader = previews.PreviewReader.open(recording, 0)
    warm = reader.clip(None, 0, 10**9, 32)
    assert warm.frame_at(1).shape == (
        32,
        32,
        3,
    )  # whatever a first decode starts lazily
    before = _native_threads()
    clips = [reader.clip(None, 0, 10**9, 32) for _ in range(N_OPEN)]
    for i, clip in enumerate(clips):
        assert clip.frame_at(i % 4).shape == (32, 32, 3)
    added = _native_threads() - before
    bound = N_OPEN * MAX_THREADS_PER_OPEN_CLIP + SLACK
    assert added <= bound, (
        f"{N_OPEN} open preview clips added {added} native threads (bound {bound}) — a "
        "montage tile's decoder or conversion is running FFmpeg's machine-sized pool again"
    )
    for clip in (warm, *clips):
        clip.close()
    after = _wait_for_threads(at_most=before)
    assert after <= before, f"closing every clip left {after - before} threads behind"
