"""The pose block (§14 *Pose in the embedding*): the same pose gives the same numbers wherever
the animal is, however it is turned, and whatever its size.

**The class: a feature that encodes where the animal is instead of how it is posed.** The
appended numbers are the body's shape in the body's own frame (centred, turned so the heading
points along +x, divided by the track's body length). A block that skips one of those steps
still comes out well-formed, and clustering then reads the arena and the camera instead of the
posture. Guarded as invariance: the same pose placed, turned and scaled differently must give
the same numbers.

**And a frame is found by part and frame together.** A part's frame numbers restart at zero, so
joining a keypoint run's rows to the embedded frames by frame number alone takes another part's
frame with the same number. ``embedding.poses_for_run`` reads each part's rows on their own;
``tests/pipeline/test_embedding_pose.py::test_a_frame_is_found_by_part_and_frame_together``
holds it to that on a two-part recording.
"""

import numpy as np
import pytest

from detail.analysis.pose_features import pose_block, visible_points

WH = (100, 100)


def _pose(centre, deg, scale, n=1):
    """Three points of one rigid pose (nose ahead, two side points behind), placed at
    ``centre`` (px), turned to ``deg`` (image degrees) and scaled by ``scale``."""
    shape = np.array([[10.0, 0.0], [-5.0, 4.0], [-5.0, -4.0]]) * scale
    t = np.radians(deg)
    turn = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    px = shape @ turn.T + np.asarray(centre)
    xy = np.repeat((px / np.array(WH))[None], n, axis=0)
    return xy, np.full(xy.shape[:2], 0.9)


def _block(xy, score, deg, length, *, mirrored=None, mode="rotate", min_score=None):
    n = xy.shape[0]
    return pose_block(
        xy,
        score,
        deg=np.full(n, deg, dtype=float),
        mirrored=np.zeros(n, bool) if mirrored is None else mirrored,
        body_length_px=length,
        frame_wh=WH,
        min_score=min_score,
        frame_mode=mode,
    )


def test_the_same_pose_anywhere_turned_and_scaled_gives_the_same_numbers():
    base, base_score = _pose((30, 30), 0, 1.0)
    want, _n = _block(base, base_score, 0.0, 15.0)
    moved, score = _pose((70, 55), 90, 2.0)
    got, _n = _block(
        moved, score, 90.0, 30.0
    )  # the heading and body length move with it
    np.testing.assert_allclose(got, want, atol=1e-5)
    assert np.linalg.norm(want[0]) == pytest.approx(1.0)  # unit length


def test_mirror_flips_a_frame_the_run_flips_and_turns_nothing():
    right, score = _pose((50, 50), 0, 1.0)
    left, _ = _pose((50, 50), 180, 1.0)
    want, _ = _block(right, score, 0.0, 15.0, mode="mirror")
    flipped, _ = _block(
        left, score, 180.0, 15.0, mode="mirror", mirrored=np.array([True])
    )
    # flipped left-right only: the y offsets keep their sign under a flip, so compare x and |y|
    w, f = want.reshape(3, 2), flipped.reshape(3, 2)
    np.testing.assert_allclose(f[:, 0], w[:, 0], atol=1e-5)
    np.testing.assert_allclose(np.abs(f[:, 1]), np.abs(w[:, 1]), atol=1e-5)


def test_a_point_below_the_floor_is_left_out_and_counted():
    xy, score = _pose((50, 50), 0, 1.0)
    score[0, 2] = 0.1
    vec, n_visible = _block(xy, score, 0.0, 15.0, min_score=0.5)
    assert n_visible.tolist() == [2]
    assert vec[0, 4:].tolist() == [0.0, 0.0]  # the dropped point sits at the centre


def test_a_frame_with_one_point_or_no_heading_gets_a_zero_block():
    xy, score = _pose((50, 50), 0, 1.0)
    xy[0, 1:] = np.nan
    vec, n_visible = _block(xy, score, 0.0, 15.0)
    assert n_visible.tolist() == [1] and not vec.any()
    xy, score = _pose((50, 50), 0, 1.0)
    vec, n_visible = _block(xy, score, np.nan, 15.0)
    assert n_visible.tolist() == [3] and not vec.any()  # never a made-up turn


def test_no_body_length_is_refused_and_a_bad_mode_too():
    xy, score = _pose((50, 50), 0, 1.0)
    with pytest.raises(ValueError, match="body lengths"):
        _block(xy, score, 0.0, None)
    with pytest.raises(ValueError, match="rotate or mirror"):
        _block(xy, score, 0.0, 15.0, mode="sideways")


def test_visible_points_reads_placed_and_scored():
    xy = np.array([[[0.1, 0.1], [np.nan, np.nan], [0.2, 0.2]]])
    score = np.array([[0.9, 0.9, 0.1]])
    assert visible_points(xy, score, None).tolist() == [[True, False, True]]
    assert visible_points(xy, score, 0.5).tolist() == [[True, False, False]]
