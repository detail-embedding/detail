"""Clustering-eval metric core — pure functions scored against deterministic toys.

The toys reproduce the exact headline numbers taught in the field-guide artifact (over-split
B-cubed F1 = 0.667 at purity 1.0, coverage 0.85), so a change that shifts a metric definition breaks a test with a known-good value, not a fuzzy bound.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)

from detail.analysis import cluster_metrics as cm

# Four imbalanced behaviors over 100 frames — the field-guide's running toy.
TRUTH = np.array([0] * 40 + [1] * 30 + [2] * 20 + [3] * 10)


def _oversplit_in_half(truth: np.ndarray) -> np.ndarray:
    """Split each behavior's frames into two equal, pure clusters (8 clusters for 4 behaviors)."""
    pred = np.empty(truth.size, dtype=np.int64)
    next_cluster = 0
    for behavior in np.unique(truth):
        idx = np.flatnonzero(truth == behavior)
        half = idx.size // 2
        pred[idx[:half]] = next_cluster
        pred[idx[half:]] = next_cluster + 1
        next_cluster += 2
    return pred


# --- B-cubed / purity / homogeneity: the over-split signature ------------------------ #
def test_oversplit_is_pure_and_homogeneous_but_low_bcubed():
    pred = _oversplit_in_half(TRUTH)
    assert cm.purity(TRUTH, pred) == 1.0  # every cluster is single-behavior
    homogeneity, completeness = cm.homogeneity_completeness(TRUTH, pred)
    assert homogeneity == pytest.approx(1.0)
    assert completeness < 1.0  # each behavior is spread over two clusters
    precision, recall, f1 = cm.bcubed(TRUTH, pred)
    assert precision == pytest.approx(1.0)  # each item's cluster is pure
    assert recall == pytest.approx(0.5)  # each behavior split evenly in half
    assert f1 == pytest.approx(2 / 3, abs=1e-9)  # the taught 0.667 — purity hides this


def test_perfect_clustering_scores_one_everywhere():
    pred = TRUTH.copy()
    precision, recall, f1 = cm.bcubed(TRUTH, pred)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)
    assert cm.purity(TRUTH, pred) == 1.0
    assert cm.homogeneity_completeness(TRUTH, pred) == (1.0, 1.0)
    assert cm.adjusted_rand_index(TRUTH, pred) == pytest.approx(1.0)
    assert cm.adjusted_mutual_information(TRUTH, pred) == pytest.approx(1.0)


def test_bcubed_matches_a_hand_computed_mixed_case():
    # cluster 0 = {a:2, b:1}, cluster 1 = {b:2}. Precision: 2*(2/3) + 3*(2/... ) worked below.
    truth = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1])
    # item precisions: three in cluster0 (sizes: class0=2,class1=1) -> 2/3,2/3,1/3;
    # two in cluster1 (both class1) -> 1,1. mean = (2/3+2/3+1/3+1+1)/5 = (11/3)/5
    precision, recall, f1 = cm.bcubed(truth, pred)
    assert precision == pytest.approx((2 / 3 + 2 / 3 + 1 / 3 + 1 + 1) / 5)
    # item recalls: class0 has 2 items both in cluster0 -> 2/2,2/2; class1 has 3 items:
    # the one in cluster0 shares with 0 other class1 -> 1/3; the two in cluster1 -> 2/3,2/3
    assert recall == pytest.approx((1 + 1 + 1 / 3 + 2 / 3 + 2 / 3) / 5)


# --- sklearn parity: the wrappers must equal sklearn --------------------------------- #
def test_wrappers_equal_sklearn():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 5, 200)
    pred = rng.integers(0, 8, 200)
    assert cm.adjusted_mutual_information(truth, pred) == adjusted_mutual_info_score(
        truth, pred
    )
    assert cm.adjusted_rand_index(truth, pred) == adjusted_rand_score(truth, pred)
    h, c, _v = homogeneity_completeness_v_measure(truth, pred)
    assert cm.homogeneity_completeness(truth, pred) == (h, c)


# --- noise policy: coverage 0.85, excluded beats singletons -------------------------- #
def test_noise_policy_coverage_and_two_readings():
    pred = TRUTH.copy()
    pred[:15] = -1  # 15 frames unassigned -> coverage 0.85
    panel = cm.agreement_panel(TRUTH, pred, noise_label=-1)
    assert panel["coverage"] == pytest.approx(0.85)
    assert panel["noise_fraction"] == pytest.approx(0.15)
    assert panel["n_assigned"] == 85
    # assigned frames are perfectly clustered; singletons fragment the 15 noise frames
    assert panel["excluded"]["ami"] == pytest.approx(1.0)
    assert 0.0 < panel["singletons"]["ami"] < 1.0
    assert panel["excluded"]["ami"] > panel["singletons"]["ami"]


def test_all_noise_gives_zero_coverage_and_empty_excluded():
    pred = np.full(TRUTH.size, -1)
    panel = cm.agreement_panel(TRUTH, pred, noise_label=-1)
    assert panel["coverage"] == 0.0
    assert panel["excluded"] == cm._empty_partition_metrics()
    assert panel["n_assigned"] == 0


# --- majority-vote map + per-behavior ------------------------------------------------ #
def test_majority_vote_map_ignores_noise_and_picks_the_plurality():
    truth = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([5, 5, 5, 5, 9, -1])  # cluster 5 is mostly behavior 0 (3 of 4)
    mapping = cm.majority_vote_map(truth, pred, noise_label=-1)
    assert mapping == {5: 0, 9: 1}  # cluster -1 (noise) omitted
    prf = cm.per_behavior_frame_prf(truth, pred, mapping, noise_label=-1)
    # behavior 0: predicted at frames of cluster5 (4 frames), true at 3 -> tp3 fp1 fn0
    assert prf[0]["recall"] == pytest.approx(1.0)
    assert prf[0]["precision"] == pytest.approx(3 / 4)
    assert prf[2]["support"] == 1 and prf[2]["recall"] == 0.0  # behavior 2 all noise
