"""Stream clock resolvers: native time -> canonical video t_ns (Section 3)."""

import numpy as np
import pytest

from detail.core.errors import UserInputError
from detail.storage import clocks

FRAME_T_NS = np.array([0, 100, 200, 300, 400], dtype=np.int64)
COLS = ["t_ns", "end_t_ns", "label", "subject", "start_frame", "end_frame"]


def test_seconds_to_t_ns():
    assert clocks.seconds_to_t_ns(1.5) == 1_500_000_000
    assert clocks.seconds_to_t_ns(0) == 0


def test_frame_to_t_ns():
    assert clocks.frame_to_t_ns(0, FRAME_T_NS) == 0
    assert clocks.frame_to_t_ns(3, FRAME_T_NS) == 300


def test_frame_to_t_ns_out_of_range():
    """Named for any stream, and for a negative frame too: it said "behavior label references
    frame -1, past the video's 5 frames" for a sensor CSV (#33)."""
    for frame in (5, -1):
        with pytest.raises(UserInputError) as err:
            clocks.frame_to_t_ns(frame, FRAME_T_NS)
        assert str(err.value).startswith(
            f"frame {frame} is outside the video, whose frames are numbered 0 to 4 (5 frames)"
        )
        assert "behavior" not in str(err.value)


def test_t_ns_to_frame():
    assert clocks.t_ns_to_frame(0, FRAME_T_NS) == 0
    assert clocks.t_ns_to_frame(150, FRAME_T_NS) == 1
    assert clocks.t_ns_to_frame(300, FRAME_T_NS) == 3
    assert clocks.t_ns_to_frame(400, FRAME_T_NS) == 4


def test_resolve_forward_fills_t_ns_keeps_frames():
    rows = [(None, None, "groom", 1, 1, 3)]
    out = clocks.resolve_native_columns(
        COLS, rows, {"t_ns": "start_frame", "end_t_ns": "end_frame"}, None, FRAME_T_NS
    )
    assert out == [(100, 300, "groom", 1, 1, 3)]


def test_resolve_reverse_fills_frames_from_t_ns():
    rows = [(150, 350, "rear", None, None, None)]
    out = clocks.resolve_native_columns(
        COLS, rows, None, {"start_frame": "t_ns", "end_frame": "end_t_ns"}, FRAME_T_NS
    )
    assert out == [(150, 350, "rear", None, 1, 3)]


def test_resolve_passthrough_when_no_columns():
    rows = [(0, 5, "x")]
    out = clocks.resolve_native_columns(
        ["t_ns", "code", "label"], rows, None, None, np.empty(0, dtype=np.int64)
    )
    assert out == rows


def test_resolve_no_video_raises():
    rows = [(None, None, "groom", 1, 1, 3)]
    with pytest.raises(ValueError, match="no video to anchor"):
        clocks.resolve_native_columns(
            COLS,
            rows,
            {"t_ns": "start_frame", "end_t_ns": "end_frame"},
            None,
            np.empty(0, dtype=np.int64),
        )


def test_t_ns_to_frame_boundary_tolerance():
    # 5 frames at 100ns cadence: [0,100,200,300,400]. A t_ns just past the last
    # frame (within one frame period) clamps to the last frame, not an error —
    # a behavior scored to the final moment of the video.
    assert clocks.t_ns_to_frame(450, FRAME_T_NS) == 4
    assert clocks.t_ns_to_frame(500, FRAME_T_NS) == 4  # exactly one period past
    assert clocks.t_ns_to_frame(-50, FRAME_T_NS) == 0  # just before the first
    # beyond one frame period is genuinely outside the video
    with pytest.raises(ValueError, match="outside the video"):
        clocks.t_ns_to_frame(600, FRAME_T_NS)
    with pytest.raises(ValueError, match="outside the video"):
        clocks.t_ns_to_frame(-150, FRAME_T_NS)


def test_needs_alignment_only_for_external_clock():
    # Video-anchored clocks are aligned at import (offset 0); only an independent
    # device clock (stream_seconds, the default — e.g. MED-PC) needs a read-time
    # offset. Every alignment surface (API/GUI) gates on this predicate.
    assert not clocks.needs_alignment("video_frames")
    assert not clocks.needs_alignment("video_seconds")
    assert clocks.needs_alignment("stream_seconds")
    assert clocks.VIDEO_ANCHORED_CLOCKS == frozenset({"video_frames", "video_seconds"})
