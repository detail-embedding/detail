"""`api.clip_curves`: the clip window's strips, read off the kinematics sidecar.

The edges: a run with no sidecar says so instead of computing a substitute; the window cuts rows
but keeps the step *into* its first row; frames with no row are reported as spans, not drawn
through; the body's shape rides on the same rows as the rest.
"""

import numpy as np
import pytest

from detail.analysis.body_axis import BodyAxis
from detail.api import clip_curves
from detail.storage import recording as recording_store

from tests.experiments import write_embedding_run

EMB = "emb_01"
FRAME_NS = 33_333_333  # 30 fps


def _set_cadence(recording_dir, every_n_frames: int) -> None:
    data = recording_store.read_recording(recording_dir)
    data["tracking_runs"][0]["every_n_frames"] = every_n_frames
    recording_store.write_recording(recording_dir, data)


def _sidecar_run(recording_dir, frames, *, cx, cy=None, stride: int = 2):
    """An embedding run over ``frames`` whose sidecar puts the subject at ``(cx, cy)`` on a
    30 fps clock; the helper's sidecar has a fixed 0.05 mask area and a full-frame box.
    """
    frames = np.asarray(frames)
    _set_cadence(
        recording_dir, stride
    )  # every `stride`th frame, and the record says so
    return write_embedding_run(
        recording_dir,
        np.zeros((len(frames), 4), np.float32),
        frames,
        frames * FRAME_NS,
        tracking_run_id="track_01",
        cx=cx,
        cy=cy if cy is not None else [0.5] * len(frames),
    )


def _read(sdir, first, last, **kwargs):
    kwargs.setdefault("pixels_per_mm", None)
    return clip_curves.read_clip_curves(
        sdir, embed_run_id=EMB, track_id=0, first=first, last=last, **kwargs
    )


def test_without_a_sidecar_the_curves_are_empty_and_the_note_names_backfill(
    processed_recording,
):
    sdir = processed_recording.recording.dir
    write_embedding_run(
        sdir,
        np.zeros((3, 4), np.float32),
        [0, 1, 2],
        [0, 1, 2],
        tracking_run_id="track_01",
    )

    got = _read(sdir, 0, 4)

    assert got.curves == [] and got.names == []
    assert len(got.frames) == 0 and got.frames.dtype == np.int64
    assert "backfill-kinematics" in got.note and EMB in got.note
    assert got.untracked == []


def test_speed_and_area_are_per_row_in_pixels_by_default(processed_recording):
    """40x40 recording: the centroid moves 10 px every 2 frames at 30 fps, so 150 px/s into
    every row but the first; the helper's 0.05 of the frame is 80 px²."""
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4], cx=[0.25, 0.5, 0.75])

    got = _read(sdir, 0, 4)

    assert got.window == (0, 4)
    np.testing.assert_array_equal(got.frames, [0, 2, 4])
    np.testing.assert_array_equal(got.t_ns, np.array([0, 2, 4]) * FRAME_NS)
    assert got.names == ["speed", "area"]
    speed, area = got.curves
    assert (speed.unit, area.unit) == ("px/s", "px²")
    assert speed.floor_zero and area.floor_zero and not speed.signed
    assert np.isnan(speed.values[0])
    np.testing.assert_allclose(speed.values[1:], [150.0, 150.0], rtol=1e-6)
    np.testing.assert_allclose(area.values, [80.0] * 3)
    assert got.untracked == []
    assert got.note == "speed and area in px"


def test_a_calibrated_recording_reads_in_millimetres(processed_recording):
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4], cx=[0.25, 0.5, 0.75])

    got = _read(sdir, 0, 4, pixels_per_mm=20.0)

    speed, area = got.curves
    assert (speed.unit, area.unit) == ("mm/s", "mm²")
    np.testing.assert_allclose(speed.values[1:], [7.5, 7.5], rtol=1e-6)
    np.testing.assert_allclose(area.values, [0.2] * 3)
    assert got.note == "speed and area in mm"


def test_the_window_cuts_rows_but_keeps_the_step_into_its_first_row(
    processed_recording,
):
    """A clip that starts mid-track still knows how fast the animal arrived at its first
    frame: the step is computed over the whole track, then cut."""
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4, 6], cx=[0.2, 0.4, 0.6, 0.8])

    got = _read(sdir, 2, 5)

    np.testing.assert_array_equal(got.frames, [2, 4])
    speed = got.curves[0]
    assert np.isfinite(speed.values).all(), "the step into frame 2 comes from frame 0"
    assert got.untracked == [(5, 5)]


def test_a_tracking_gap_and_the_window_edges_are_untracked_spans(processed_recording):
    """Rows at 2, 4 and 10 in a window of 0..12 at stride 2: nothing before 2, a gap between
    4 and 10 (not adjacent at the stride), nothing after 10. Speed across the gap is NaN.
    """
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [2, 4, 10], cx=[0.2, 0.4, 0.6])

    got = _read(sdir, 0, 12)

    np.testing.assert_array_equal(got.frames, [2, 4, 10])
    assert got.untracked == [(0, 1), (5, 9), (11, 12)]
    speed = got.curves[0]
    assert np.isnan(speed.values[2]), "no step across a gap"
    assert np.isfinite(speed.values[1])


def test_untracked_spans_alone():
    assert clip_curves.untracked_spans(np.array([2, 4, 10]), 2, 0, 12) == [
        (0, 1),
        (5, 9),
        (11, 12),
    ]
    assert clip_curves.untracked_spans(np.array([0, 1, 2]), 1, 0, 2) == []
    assert clip_curves.untracked_spans(np.array([], dtype=np.int64), 1, 3, 7) == [
        (3, 7)
    ]
    assert clip_curves.untracked_spans(np.array([0, 20]), 1, 5, 15) == [(5, 15)]


def _straight_pose(n: int, *, turn_per_frame_deg: float = 0.0):
    """A three-point skeleton (front, middle, back) per window frame on a 40 px frame: a
    20 px axis, the middle on the line, rotating ``turn_per_frame_deg`` a frame."""
    xy = np.full((n, 3, 2), np.nan)
    for i in range(n):
        a = np.radians(-90.0 + turn_per_frame_deg * i)  # -90°: facing up, image y down
        d = np.array([np.cos(a), np.sin(a)]) * 10.0 / 40.0
        xy[i, 0] = 0.5 + d  # front
        xy[i, 1] = (0.5, 0.5)  # middle, on the axis
        xy[i, 2] = 0.5 - d  # back
    return xy, np.ones((n, 3))


def _axis(middle=(1,)):
    return BodyAxis(
        front=(0,), back=(2,), middle=middle, frame_wh=(40, 40), cutoff=None,
        typical_length_px=10.0,
    )  # fmt: skip


def test_the_bodys_shape_rides_on_the_embedded_rows(processed_recording):
    """Pose rows are given for every window frame; the curves take the ones at the embedded
    frames: a 20 px axis over a 10 px typical length reads 2.0 on every row, the middle on
    the line bends 0°, and a steady 3°-a-frame turn is 90°/s at 30 fps."""
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4], cx=[0.5, 0.5, 0.5])
    xy, scores = _straight_pose(5, turn_per_frame_deg=3.0)

    got = _read(sdir, 0, 4, pose=(xy, scores, _axis()))

    assert got.names == ["speed", "area", "length", "bend", "turning"]
    length, bend, turning = got.curves[2:]
    assert len(length.values) == 3 and not length.signed
    np.testing.assert_allclose(length.values, [2.0] * 3)
    assert bend.signed and turning.signed
    np.testing.assert_allclose(bend.values, [0.0] * 3, atol=1e-9)
    np.testing.assert_allclose(turning.values, [90.0] * 3, rtol=1e-6)
    assert (length.unit, bend.unit, turning.unit) == ("x typical", "°", "°/s")


def test_without_a_midline_point_there_is_no_bend_curve(processed_recording):
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4], cx=[0.5, 0.5, 0.5])
    xy, scores = _straight_pose(5)

    got = _read(sdir, 0, 4, pose=(xy, scores, _axis(middle=())))

    assert got.names == ["speed", "area", "length", "turning"]


def test_a_pose_that_does_not_cover_the_window_is_refused(processed_recording):
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4], cx=[0.5, 0.5, 0.5])
    xy, scores = _straight_pose(3)

    with pytest.raises(ValueError, match="3 rows for a window of 5"):
        _read(sdir, 0, 4, pose=(xy, scores, _axis()))


def test_an_inverted_window_is_refused(processed_recording):
    with pytest.raises(ValueError, match="runs forward"):
        _read(processed_recording.recording.dir, 4, 0)
