"""Tests for the bout-segmentation core (detail/analysis/segmentation.py).

The v2 core builds **window-pooled features** in their own feature-index space, so cases cover:
the feature matrix shape + center mapping (``build_features``), that detection/segmentation stay
inside continuous runs and map bouts back to trajectory rows, ``medoid_order`` ranking, and the
live-progress contract on ``pairwise_dtw``. The scipy-backed detectors are guarded with
``importorskip`` (binseg + the pure helpers need neither).
"""

import numpy as np
import pytest

from detail.analysis import segmentation


def _block_traj(directions, block_len=12, dim=8, noise=0.01, seed=0):
    """A trajectory of constant-direction blocks — clean synthetic behavior bouts."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in directions:
        base = np.zeros(dim)
        base[d % dim] = 1.0
        rows.append(base[None, :] + noise * rng.standard_normal((block_len, dim)))
    return np.vstack(rows).astype(np.float32)


# --- build_features --------------------------------------------------------------


def test_build_features_window_mean_shape_and_centers():
    traj = _block_traj([0, 1], block_len=10, dim=6)  # 20 rows, one run
    X, row_of_X, xruns = segmentation.build_features(
        traj, [(0, 20)], window=4, pool="mean"
    )
    assert X.shape == (17, 6)  # 20 - 4 + 1 sliding windows
    assert xruns == [(0, 17)]
    assert list(row_of_X) == list(range(2, 19))  # center = start + window//2


def test_build_features_flatten_shape():
    traj = _block_traj([0], block_len=10, dim=6)
    X, _, _ = segmentation.build_features(traj, [(0, 10)], window=4, pool="flatten")
    assert X.shape == (7, 4 * 6)  # window*D per feature


def test_build_features_max_shape_and_values():
    traj = _block_traj([0], block_len=10, dim=6)
    X, row_of_X, xruns = segmentation.build_features(
        traj, [(0, 10)], window=4, pool="max"
    )
    assert X.shape == (7, 6)  # per-dimension max keeps the source width, like mean
    assert xruns == [(0, 7)] and list(row_of_X) == list(range(2, 9))
    tn = segmentation._normalize(traj.astype(np.float64))
    assert np.allclose(X[0], tn[0:4].max(axis=0))


def test_build_features_max_keeps_a_spike_that_mean_dilutes():
    """The reason max exists: one distinctive frame in a window survives pooling."""
    traj = np.zeros((8, 3), dtype=np.float64)
    traj[:, 0] = 1.0  # a constant baseline pose
    traj[3] = [0.0, 1.0, 0.0]  # one frame pointing elsewhere
    mean_X, _, _ = segmentation.build_features(traj, [(0, 8)], window=4, pool="mean")
    max_X, _, _ = segmentation.build_features(traj, [(0, 8)], window=4, pool="max")
    # The window covering row 3 carries the spike's dimension at full height under max,
    # but diluted to 1/window under mean.
    assert max_X[0, 1] == pytest.approx(1.0)
    assert mean_X[0, 1] == pytest.approx(0.25)


def test_build_features_no_features_width_matches_pool():
    """An empty run must return the same width a populated one would, per pool mode."""
    traj = _block_traj([0], block_len=3, dim=4)  # 3 rows, window 4 -> nothing builds
    for pool, expected in (("mean", 4), ("max", 4), ("flatten", 16)):
        X, _row_of_X, xruns = segmentation.build_features(
            traj, [(0, 3)], window=4, pool=pool
        )
        assert X.shape == (0, expected), pool
        assert xruns == []


def test_build_features_window_1_is_per_frame():
    traj = _block_traj([0, 1], block_len=5, dim=6)  # 10 rows
    X, row_of_X, xruns = segmentation.build_features(traj, [(0, 10)], window=1)
    assert X.shape == (10, 6)
    assert list(row_of_X) == list(range(10))
    assert np.allclose(X, segmentation._normalize(traj.astype(np.float64)))


def test_build_features_run_shorter_than_window_is_skipped():
    traj = _block_traj([0], block_len=10, dim=4)
    X, row_of_X, xruns = segmentation.build_features(traj, [(0, 8), (8, 10)], window=4)
    assert (
        X.shape[0] == 5
    )  # run (0,8) yields 5 windows; run (8,10) too short -> skipped
    assert xruns == [(0, 5)]
    assert list(row_of_X) == [2, 3, 4, 5, 6]


def test_build_features_no_features_returns_empty_not_raise():
    traj = _block_traj([0], block_len=3, dim=4)  # 3 rows, window 4 -> nothing
    X, row_of_X, xruns = segmentation.build_features(traj, [(0, 3)], window=4)
    assert X.shape == (0, 4) and len(row_of_X) == 0 and xruns == []


def test_build_features_bad_pool_raises():
    with pytest.raises(ValueError, match="pool"):
        segmentation.build_features(np.zeros((5, 3)), [(0, 5)], window=2, pool="nope")


# --- segment_bouts (feature edges -> trajectory rows) ----------------------------


def test_segment_bouts_maps_feature_edges_to_center_rows():
    row_of_X = np.arange(2, 12)  # 10 features -> center rows 2..11
    bouts = segmentation.segment_bouts([4], row_of_X, [(0, 10)], min_len=1)
    assert bouts == [(2, 5), (6, 11)]


def test_segment_bouts_respects_runs_and_min_len():
    row_of_X = np.arange(20)  # identity (window=1)
    bouts = segmentation.segment_bouts(
        [3, 15], row_of_X, [(0, 10), (10, 20)], min_len=1
    )
    assert bouts == [(0, 2), (3, 9), (10, 14), (15, 19)]
    assert all(not (a < 10 <= b) for a, b in bouts)  # none straddles the run gap
    # min_len keeps only the 7-row bout
    assert segmentation.segment_bouts([3, 15], row_of_X, [(0, 10), (10, 20)], 6) == [
        (3, 9)
    ]


# --- detectors -------------------------------------------------------------------


def test_detect_transitions_unknown_detector_raises():
    with pytest.raises(ValueError, match="unknown detector"):
        segmentation.detect_transitions(np.zeros((5, 3)), [(0, 5)], "nope", {})


def test_detect_transitions_binseg_finds_boundary_in_feature_space():
    traj = _block_traj([0, 1], block_len=20, dim=6)  # boundary at row 20
    X, row_of_X, xruns = segmentation.build_features(traj, [(0, 40)], window=4)
    cuts = segmentation.detect_transitions(
        X, xruns, "binseg", {"n_bkps": 1, "min_size": 4}
    )
    assert len(cuts) == 1
    assert 16 <= int(row_of_X[cuts[0]]) <= 24  # maps back near the true boundary


def test_novelty_and_speed_return_valid_feature_cuts():
    pytest.importorskip("scipy")
    traj = _block_traj([0, 1, 2], block_len=12, dim=6)
    X, _, xruns = segmentation.build_features(traj, [(0, 36)], window=4)
    for detector in ("novelty", "speed"):
        cuts = segmentation.detect_transitions(
            X, xruns, detector, {"percentile": 80.0, "min_size": 3}
        )
        assert all(0 < c < X.shape[0] for c in cuts)


# --- medoid_order ---------------------------------------------------------------


def test_medoid_order_central_first_ascending_scores():
    dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    order = segmentation.medoid_order(dist, [0, 1, 2])
    assert order[0] == (1, pytest.approx(1.0))  # member 1 is central (mean 1.0)
    assert [s for _, s in order] == sorted(s for _, s in order)


def test_medoid_order_single_and_empty():
    assert segmentation.medoid_order(np.zeros((2, 2)), [1]) == [(1, 0.0)]
    assert segmentation.medoid_order(np.zeros((0, 0)), []) == []


# --- pairwise_dtw: progress + max_segments ---------------------------------------


def test_pairwise_dtw_reports_intermediate_progress():
    pytest.importorskip("numba")  # the DTW kernel compiles via _dtw_numba
    traj = _block_traj([0, 1, 0, 1], block_len=10)
    bouts = [(0, 9), (10, 19), (20, 29), (30, 39)]
    events: list[tuple[int, int]] = []
    _, dist = segmentation.pairwise_dtw(
        traj, bouts, band=2, progress_cb=lambda d, t: events.append((d, t))
    )
    assert dist.shape == (4, 4)
    total = 4 * 3 // 2
    assert any(0 < d < total for d, t in events)  # single-subject live movement (§6)
    assert events[-1] == (total, total)


def test_pairwise_dtw_max_segments_caps_the_matrix():
    pytest.importorskip("numba")
    traj = _block_traj([0] * 10, block_len=5)
    bouts = [(i * 5, i * 5 + 4) for i in range(10)]
    kept, dist = segmentation.pairwise_dtw(traj, bouts, band=2, max_segments=4)
    assert len(kept) == 4 and dist.shape == (4, 4)


def test_pairwise_dtw_matches_per_pair_reference():
    # The fused numba kernel must reproduce the per-pair reference
    # (retrieval.dtw_distance over cosine_cost) — varying bout lengths so the
    # eff = max(band, |li-lj|) widening is exercised — plus be symmetric with a zero diagonal.
    pytest.importorskip("numba")
    from detail.analysis import retrieval

    rng = np.random.default_rng(11)
    lens = [7, 12, 9, 15, 6, 11]
    traj = rng.standard_normal((sum(lens), 8)).astype(np.float32)
    bouts, r = [], 0
    for length in lens:
        bouts.append((r, r + length - 1))
        r += length
    band = 3
    _, dist = segmentation.pairwise_dtw(traj, bouts, band=band)

    segs = [np.asarray(traj[a : b + 1], dtype=np.float64) for a, b in bouts]
    ref = np.zeros((len(segs), len(segs)))
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            eff = max(band, abs(len(segs[i]) - len(segs[j])))
            ref[i, j] = ref[j, i] = retrieval.dtw_distance(
                retrieval.cosine_cost(segs[i], segs[j]), eff
            )
    assert np.allclose(dist, ref, atol=1e-6)
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)
