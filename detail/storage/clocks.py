"""Stream clock resolvers: an importer's native time → canonical video ``t_ns`` (Section 3).

A ``StreamImporter`` declares a ``clock`` and stays recording-agnostic (Approach Y); the
import layer — which has the recording's active timestamp map — resolves the native-time
columns here. Two map-dependent operations, driven by what the importer declares on its
``StreamImportResult``:

* ``frame_columns`` ``{t_ns_col: frame_col}`` — a frame-clock source (``video_frames``):
  fill each ``t_ns`` column from the source frame column via the forward map.
* ``time_columns`` ``{frame_col: t_ns_col}`` — a source already in video ``t_ns`` (e.g.
  BORIS seconds → ns, done in the importer): fill the convenience frame columns via the
  inverse map.

``t_ns`` stays the anchor; the frame columns ride alongside (Tenet 8). An importer that
emits ``t_ns`` directly and needs no frame columns (MED-PC) declares neither and passes
through unchanged.
"""

from __future__ import annotations

import numpy as np

from detail.core import utils
from detail.core.errors import UserInputError

_NS_PER_SECOND = 1_000_000_000

# Clocks already expressed in the video's own time reference. A stream on one of
# these is anchored at import — its alignment offset is definitionally 0 — so the
# sync-to-video / edit-offset operations do not apply to it. Only the default
# external clock ("stream_seconds", e.g. MED-PC) is aligned by a read-time offset.
VIDEO_ANCHORED_CLOCKS = frozenset({"video_frames", "video_seconds"})


def needs_alignment(clock: str) -> bool:
    """Whether a stream on this clock needs a video-alignment offset (Section 3).

    A ``video_frames`` / ``video_seconds`` stream is already on the video clock, so
    its offset is 0 and no sync-to-video step applies; only an external-clock stream
    (``stream_seconds``, e.g. MED-PC) is aligned by a read-time offset. Every
    alignment surface (API/GUI) gates on this — a stream that doesn't need alignment
    never exposes a sync or offset control.
    """
    return clock not in VIDEO_ANCHORED_CLOCKS


def seconds_to_t_ns(seconds: float, *, what: str = "time") -> int:
    """Video-relative seconds → int64 ``t_ns`` (Tenet 8). No recording context needed.

    Delegates to the one checked conversion, ``core.utils.seconds_to_t_ns``, so a time beyond
    int64 is refused here too rather than stored.
    """
    return utils.seconds_to_t_ns(seconds, what=what)


def frame_to_t_ns(frame_idx: int, frame_t_ns: np.ndarray) -> int:
    """The video ``t_ns`` of a recording-global frame index via the active map.

    Raises:
        ValueError: If ``frame_idx`` is negative or past the video's last frame.
    """
    n = len(frame_t_ns)
    if not 0 <= frame_idx < n:
        raise UserInputError(
            f"frame {frame_idx} is outside the video, whose frames are numbered 0 to {n - 1} "
            f"({n} frames). Was the file written for another video, or does it count frames "
            "from 1?"
        )
    return int(frame_t_ns[frame_idx])


def _nominal_period_ns(frame_t_ns: np.ndarray) -> int:
    """Median inter-frame step (the cadence) — the boundary tolerance unit."""
    if len(frame_t_ns) < 2:
        return 0
    return int(np.median(np.diff(np.asarray(frame_t_ns, dtype=np.int64))))


def t_ns_to_frame(t_ns: int, frame_t_ns: np.ndarray) -> int:
    """The recording-global frame whose interval contains ``t_ns`` (inverse map).

    Time → frame is inherently approximate (continuous time, discrete frames), so a
    ``t_ns`` within **one frame period** beyond either end — a behavior scored right up
    to the last moment of the video, where its exact time can round just past the last
    frame's PTS — clamps to the boundary frame rather than failing. The convenience
    ``start_frame``/``end_frame`` columns absorb the rounding; the exact ``t_ns`` anchor
    is unchanged. Beyond that tolerance the label is genuinely outside the video.

    Raises:
        ValueError: If ``t_ns`` falls more than one frame period before the first or
            after the last frame.
    """
    if len(frame_t_ns) == 0:
        raise ValueError("cannot resolve a frame: the video has no frames")
    first, last = int(frame_t_ns[0]), int(frame_t_ns[-1])
    tol = _nominal_period_ns(frame_t_ns)
    if t_ns < first - tol or t_ns > last + tol:
        raise UserInputError(
            f"a row at {t_ns / _NS_PER_SECOND:.3f} s is outside the video, which runs from "
            f"{first / _NS_PER_SECOND:.3f} s to {last / _NS_PER_SECOND:.3f} s (beyond one "
            "frame of tolerance). Was the file written for another video?"
        )
    if t_ns <= first:
        return 0
    if t_ns >= last:
        return len(frame_t_ns) - 1
    return int(np.searchsorted(frame_t_ns, t_ns, side="right")) - 1


def resolve_native_columns(
    columns: list[str],
    rows: list[tuple],
    frame_columns: dict[str, str] | None,
    time_columns: dict[str, str] | None,
    frame_t_ns: np.ndarray,
) -> list[tuple]:
    """Fill a result's ``t_ns``/frame columns from the recording's active map.

    ``frame_columns`` fills each ``t_ns`` column from its source frame column
    (forward); ``time_columns`` fills each frame column from its source ``t_ns``
    column (inverse). Returns the completed rows; a passthrough (both ``None``)
    returns ``rows`` unchanged.

    Raises:
        ValueError: If resolution is required but the recording has no active
            timestamp map (no video to anchor frames to), or a value is out of range.
    """
    if not frame_columns and not time_columns:
        return rows
    if len(frame_t_ns) == 0:
        raise UserInputError(
            "cannot import this file: its times are video frames, and the recording has "
            "no video to anchor frames to yet. Add the video first"
        )
    idx = {c: i for i, c in enumerate(columns)}
    out: list[tuple] = []
    for row in rows:
        cells = list(row)
        for t_col, f_col in (frame_columns or {}).items():
            cells[idx[t_col]] = frame_to_t_ns(int(cells[idx[f_col]]), frame_t_ns)
        for f_col, t_col in (time_columns or {}).items():
            cells[idx[f_col]] = t_ns_to_frame(int(cells[idx[t_col]]), frame_t_ns)
        out.append(tuple(cells))
    return out
