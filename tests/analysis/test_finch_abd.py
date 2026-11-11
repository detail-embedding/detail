"""FINCH, the agglomerative refinement, and the ABD detector — the claims each one makes.

The generic contracts (labels per bout, the single-bout edge, the registry dispatch) are covered
data-driven in ``test_cluster_methods_docs.py``. What is here is what makes each worth adding.
"""

import numpy as np
import pytest

from detail.analysis import segmentation as seg


def _grouped_distances(sizes=(8, 8, 8), spread=10.0, seed=0):
    """A distance matrix with tight, well-separated groups of the given sizes."""
    rng = np.random.default_rng(seed)
    centers = np.concatenate([[i * spread] * n for i, n in enumerate(sizes)])
    noise = rng.normal(0, 0.01, (len(centers), len(centers))) ** 2
    dist = np.abs(centers[:, None] - centers[None, :]) + noise
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return dist, centers


def _blocks(X_len, n_blocks, dim=6, noise=0.25, drift=0.0, seed=0):
    """Block-constant features with optional baseline drift; returns (X, true cut indices)."""
    rng = np.random.default_rng(seed)
    per = X_len // n_blocks
    X = np.vstack([np.tile(v, (per, 1)) for v in rng.normal(0, 1, (n_blocks, dim))])
    X = X + rng.normal(0, noise, X.shape)
    X = X + np.linspace(0, drift, len(X))[:, None] * 0.4
    return X, [per * i for i in range(1, n_blocks)]


# --- FINCH ---------------------------------------------------------------------------- #
def test_finch_builds_a_hierarchy_that_coarsens():
    """Each level must be strictly coarser, or the recursion is not doing anything."""
    dist, _ = _grouped_distances()
    levels = seg.finch_hierarchy(dist)
    counts = [len(set(lab.tolist())) for lab in levels]
    assert len(counts) >= 2
    assert counts == sorted(counts, reverse=True), counts


def test_finch_levels_are_nested():
    """A coarse level merges fine clusters whole,
    never splitting one across two."""
    dist, _ = _grouped_distances()
    levels = seg.finch_hierarchy(dist)
    for fine, coarse in zip(levels, levels[1:]):
        for cluster in set(fine.tolist()):
            members = coarse[fine == cluster]
            assert (
                len(set(members.tolist())) == 1
            ), "a fine cluster was split by a coarser level"


def test_finch_assigns_every_bout():
    """No noise class — which is why it always reports full coverage."""
    dist, _ = _grouped_distances()
    labels = seg.cluster_labels(dist, "finch", {"n_clusters": 3})
    assert (labels >= 0).all()


def test_finch_hits_the_requested_count():
    dist, _ = _grouped_distances()
    for target in (2, 3, 5):
        labels = seg.cluster_labels(dist, "finch", {"n_clusters": target})
        assert len(set(labels.tolist())) == target


def test_finch_recovers_the_true_groups():
    dist, centers = _grouped_distances()
    labels = seg.cluster_labels(dist, "finch", {"n_clusters": 3})
    for value in set(centers.tolist()):
        block = labels[centers == value]
        assert len(set(block.tolist())) == 1


def test_finch_partition_selects_a_level():
    dist, _ = _grouped_distances()
    levels = seg.finch_hierarchy(dist)
    picked = seg.cluster_labels(dist, "finch", {"partition": 0})
    np.testing.assert_array_equal(picked, levels[0])


# --- the agglomerative refinement (ABD's second half) --------------------------------- #
def test_agglomerative_hits_the_requested_count():
    dist, _ = _grouped_distances()
    labels = seg.cluster_labels(dist, "agglomerative", {"n_clusters": 3})
    assert len(set(labels.tolist())) == 3


def test_agglomerative_never_asks_for_more_clusters_than_bouts():
    """The schema edge: a target above the bout count must clamp, not raise."""
    dist, _ = _grouped_distances(sizes=(2,))
    labels = seg.cluster_labels(dist, "agglomerative", {"n_clusters": 50})
    assert len(labels) == len(dist)


def test_agglomerative_merges_over_segmentation_back_together():
    """ABD's design: cut generously, then merge the pieces of one behavior back."""
    # six fragments that are really two behaviors
    dist, centers = _grouped_distances(sizes=(3, 3, 3, 3), spread=0.0, seed=1)
    dist[:6, 6:] += 10.0
    dist[6:, :6] += 10.0
    np.fill_diagonal(dist, 0.0)
    labels = seg.cluster_labels(dist, "agglomerative", {"n_clusters": 2})
    assert len(set(labels[:6].tolist())) == 1
    assert len(set(labels[6:].tolist())) == 1


# --- the ABD detector ------------------------------------------------------------------ #
def test_abd_finds_every_true_boundary():
    """Recall is the metric that matters for a detector meant to over-cut: a spurious cut can be
    merged away later, a missed one cannot be recovered."""
    for drift in (0.0, 3.0):
        for seed in range(6):
            X, truth = _blocks(160, 4, drift=drift, seed=seed)
            cuts = seg.detect_transitions(X, [(0, len(X))], "abd", {})
            for boundary in truth:
                assert any(abs(c - boundary) <= 3 for c in cuts), (
                    drift,
                    seed,
                    boundary,
                )


def test_abd_over_segments_on_purpose():
    """Stated behavior: it produces more cuts than there are behaviors, for a merge to clean up."""
    X, truth = _blocks(160, 4, seed=0)
    cuts = seg.detect_transitions(X, [(0, len(X))], "abd", {})
    assert len(cuts) > len(truth)


def test_abd_smoothing_wider_than_the_signal_destroys_boundaries():
    """Pins why the default is small — the failure that picking a larger one reintroduces."""
    found = {}
    for smooth_half in (2, 8):
        hits = 0
        for seed in range(6):
            X, truth = _blocks(160, 4, seed=seed)
            cuts = seg.detect_transitions(
                X, [(0, len(X))], "abd", {"smooth_half": smooth_half}
            )
            hits += sum(any(abs(c - b) <= 3 for c in cuts) for b in truth)
        found[smooth_half] = hits
    assert found[2] > found[8], found


def test_abd_has_a_curve_for_the_diagnostics():
    """A peak-picking detector must explain itself in the diagnostics view like the others do."""
    X, _ = _blocks(160, 4, seed=0)
    curve = seg.transition_curve(X, [(0, len(X))], "abd", {})
    assert curve is not None
    assert len(curve.values) == len(X)


def test_abd_handles_a_run_too_short_to_score():
    assert seg.detect_transitions(np.zeros((2, 4)), [(0, 2)], "abd", {}) == []


@pytest.mark.parametrize("detector", sorted(seg.TRANSITION_DETECTORS))
def test_no_detector_cuts_outside_its_run(detector):
    """Every cut must land inside the run it was found in — a cut at a run edge would make a bout
    that spans a tracking gap."""
    X, _ = _blocks(160, 4, seed=0)
    runs = [(0, 80), (80, 160)]
    cuts = seg.detect_transitions(X, runs, detector, {"penalty": 5.0})
    assert all(0 < c < 160 for c in cuts)
    assert all(c != 80 for c in cuts), "a cut landed exactly on the run boundary"
