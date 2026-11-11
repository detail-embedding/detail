"""The keypoint filters and the candidate path (§26 "Track methods"): each filter held equal to
the per-row research implementation it was measured as, and the two dynamic programs on
evidence a test can see."""

import numpy as np
import pytest

from detail.analysis import keypoint_filters
from detail.analysis.keypoint_filters import (
    hampel_despike,
    rows_for_seconds,
    smooth_local_quadratic,
    two_state_path,
    viterbi_path,
)

FPS = 30.0


def _reference(xy, scale, half=3, k=3.0, floor=0.03):
    """The research implementation of the Hampel despike, verbatim in
    behaviour: the per-frame loop the measured results came from."""
    n, kp, _ = xy.shape
    out = xy.copy()
    replaced = np.zeros((n, kp), dtype=bool)
    finite = np.isfinite(xy).all(axis=2)
    for j in range(kp):
        p = xy[:, j]
        for t in range(n):
            if not finite[t, j]:
                continue
            lo, hi = max(0, t - half), min(n, t + half + 1)
            w = p[lo:hi][finite[lo:hi, j]]
            if len(w) < half + 1:
                continue
            med = np.median(w, axis=0)
            dev = 1.4826 * np.median(np.linalg.norm(w - med, axis=1))
            if np.linalg.norm(p[t] - med) > k * max(dev, floor * scale[t]):
                out[t, j] = med
                replaced[t, j] = True
    return out, replaced


def _track(seed: int, n: int = 400, kp: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None, None]
    xy = (
        300
        + 40 * np.sin(t / 25.0 + np.arange(kp)[None, :, None])
        + rng.normal(0, 1.5, (n, kp, 2))
    )
    spikes = rng.random((n, kp)) < 0.04
    xy[spikes] += rng.normal(0, 60, (int(spikes.sum()), 2))
    xy[rng.random((n, kp)) < 0.08] = np.nan  # absent points, some in runs
    xy[120:135, 2] = np.nan
    scale = np.full(n, 120.0) + rng.normal(0, 5, n)
    return xy, scale


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("half,k", [(3, 3.0), (2, 2.5), (5, 4.0)])
def test_despike_matches_the_research_filter(seed, half, k):
    xy, scale = _track(seed)
    ours, ours_replaced = hampel_despike(xy, scale, half_window=half, k=k)
    ref, ref_replaced = _reference(xy, scale, half=half, k=k)
    np.testing.assert_array_equal(ours_replaced, ref_replaced)
    np.testing.assert_allclose(ours, ref, equal_nan=True)


def test_a_short_jump_is_removed_and_steady_motion_kept():
    n = 60
    xy = np.stack([np.arange(n) * 2.0, np.full(n, 100.0)], axis=1)[:, None, :]
    xy = np.repeat(xy, 2, axis=1)
    clean = xy.copy()
    xy[30, 0] += (80.0, -80.0)  # one-frame jump on point 0
    out, replaced = hampel_despike(xy, np.full(n, 100.0), half_window=3, k=3.0)
    assert replaced.sum() == 1 and replaced[30, 0]
    np.testing.assert_allclose(out[:, 1], clean[:, 1])  # steady motion untouched
    assert np.linalg.norm(out[30, 0] - clean[30, 0]) < 3.0


def test_absence_stays_absent_and_sparse_points_are_left_alone():
    xy, scale = _track(3)
    out, _ = hampel_despike(xy, scale)
    assert np.isnan(out[np.isnan(xy)]).all()
    sparse = np.full((10, 1, 2), np.nan)
    sparse[4] = (0.0, 0.0)
    sparse[5] = (500.0, 500.0)  # a jump with no neighbours cannot be judged
    out, replaced = hampel_despike(sparse, np.full(10, 100.0))
    assert not replaced.any()
    assert hampel_despike(np.zeros((0, 2, 2)), np.zeros(0))[0].shape == (0, 2, 2)


def _smooth_reference(xy, t_ns, window_s, truncate=3.0, min_points=5):
    """The evaluation's per-row smoothing loop (kernel "quad"),
    the arm the measured results came from."""
    t_s = np.asarray(t_ns, dtype=np.float64) / 1e9
    out = xy.copy()
    present = np.isfinite(xy).all(axis=2)
    filled = np.where(present[..., None], xy, 0.0)
    lo = np.searchsorted(t_s, t_s - truncate * window_s, side="left")
    hi = np.searchsorted(t_s, t_s + truncate * window_s, side="right")
    for i in range(len(t_s)):
        sl = slice(lo[i], hi[i])
        dt = t_s[sl] - t_s[i]
        weight = np.exp(-0.5 * (dt / window_s) ** 2)[:, None] * present[sl]
        den = weight.sum(axis=0)
        value = (weight[..., None] * filled[sl]).sum(axis=0) / np.where(
            den > 0, den, np.nan
        )[:, None]
        design = np.stack([np.ones_like(dt), dt, dt * dt], axis=1)
        normal = np.einsum("mk,mi,mj->kij", weight, design, design)
        rhs = np.einsum("mk,mi,mkd->kid", weight, design, filled[sl])
        ok = ((weight > 1e-6).sum(axis=0) >= min_points) & (
            np.abs(np.linalg.det(normal)) > 1e-18
        )
        if ok.any():
            value = value.copy()
            value[ok] = np.linalg.solve(normal[ok], rhs[ok])[:, 0, :]
        out[i] = np.where(present[i][:, None], value, np.nan)
    return out


def _times(frames) -> np.ndarray:
    return np.round(np.asarray(frames) / FPS * 1e9).astype(np.int64)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("window_s", [0.033, 0.133, 0.267])
def test_smoothing_matches_the_study_arm(seed, window_s):
    xy, _scale = _track(seed)
    frames = np.arange(len(xy))
    frames[200:] += 40  # the track loses the subject for 40 frames
    t_ns = _times(frames)
    ours = smooth_local_quadratic(xy, t_ns, window_s)
    np.testing.assert_allclose(
        ours, _smooth_reference(xy, t_ns, window_s), equal_nan=True
    )


def test_smoothing_keeps_curved_motion_and_removes_noise():
    n = 200
    t = np.arange(n) / FPS
    curve = np.stack([100 + 50 * t * t, 300 - 30 * t], axis=1)[:, None, :]
    rng = np.random.default_rng(0)
    noisy = curve + rng.normal(0, 2.0, curve.shape)
    smoothed = smooth_local_quadratic(noisy, _times(np.arange(n)), 0.133)
    inner = slice(20, -20)
    assert (
        np.abs(smoothed[inner] - curve[inner]).mean()
        < np.abs(noisy[inner] - curve[inner]).mean() / 2
    )
    exact = smooth_local_quadratic(curve, _times(np.arange(n)), 0.133)
    np.testing.assert_allclose(exact, curve, atol=1e-6)  # a quadratic is kept exactly


def test_smoothing_never_fills_absence_and_does_not_reach_across_a_gap():
    xy, _scale = _track(4)
    frames = np.arange(len(xy))
    frames[200:] += 300  # a 10 s gap: nothing on either side reaches the other
    t_ns = _times(frames)
    out = smooth_local_quadratic(xy, t_ns, 0.133)
    assert np.isnan(out[np.isnan(xy)]).all()
    left = smooth_local_quadratic(xy[:200], t_ns[:200], 0.133)
    np.testing.assert_allclose(out[:200], left, equal_nan=True)
    assert smooth_local_quadratic(
        np.zeros((0, 1, 2)), np.zeros(0, np.int64), 0.1
    ).shape == (
        0,
        1,
        2,
    )


def test_a_window_in_seconds_becomes_rows_at_the_cadence():
    assert rows_for_seconds(0.1, 30.0) == 3
    assert rows_for_seconds(0.1, 15.0) == 2  # the same duration, fewer rows
    assert rows_for_seconds(0.001, 30.0) == 1  # never below one row
    with pytest.raises(ValueError, match="positive"):
        rows_for_seconds(0.1, 0.0)
    assert keypoint_filters.SMOOTH_KERNEL == "local_quadratic"


# --------------------------------------------------------------------------- #
# The candidate path
# --------------------------------------------------------------------------- #
def test_the_path_prefers_a_consistent_candidate_over_a_louder_decoy():
    """Candidate 0 is a slightly weaker but steady place; candidate 1 jumps far away on every
    row. With a move cost, the steady one wins the whole path; without one, the loud decoy.
    """
    n = 20
    emission = np.tile([0.6, 0.7], (n, 1))
    positions = np.zeros((n, 2, 2))
    positions[:, 0] = (100.0, 100.0)
    positions[:, 1] = np.where(
        (np.arange(n) % 2 == 0)[:, None], (900.0, 900.0), (100.0, 900.0)
    )

    def cost(i, sigma=20.0):
        d = np.linalg.norm(
            positions[i][None, :, :] - positions[i - 1][:, None, :], axis=2
        )
        return 0.5 * (d / sigma) ** 2

    assert viterbi_path(emission, cost, missing_penalty=2.0).tolist() == [0] * n
    free = viterbi_path(emission, lambda i: np.zeros((2, 2)), missing_penalty=2.0)
    assert free.tolist() == [1] * n


def test_the_path_goes_missing_where_every_candidate_is_absent_or_too_cheap():
    emission = np.array([[0.5, np.nan], [np.nan, np.nan], [0.4, 0.1], [-9.0, -9.0]])
    path = viterbi_path(emission, lambda i: np.zeros((2, 2)), missing_penalty=1.0)
    assert path.tolist() == [
        0,
        -1,
        0,
        -1,
    ]  # absent row and a row cheaper than the penalty
    assert viterbi_path(np.zeros((0, 2)), lambda i: None, 1.0).shape == (0,)


def test_a_swap_must_hold_over_frames_to_win():
    """Evidence for 'swapped' that flickers for one frame is held at 'normal'; a run of it
    switches once and stays."""
    flicker = np.full(30, -1.0)
    flicker[10] = 3.0
    assert not two_state_path(flicker, switch_cost=2.0).any()
    held = np.full(30, -1.0)
    held[12:] = 1.0
    swapped = two_state_path(held, switch_cost=2.0)
    assert swapped[:12].sum() == 0 and swapped[12:].all()
    assert two_state_path(np.full(5, np.nan), 1.0).tolist() == [False] * 5
    assert two_state_path(np.zeros(0), 1.0).shape == (0,)
