"""Per-behavior F1 needs the floor shipped beside it, or a prevalent behavior reads as a win.

The metric is not chance-corrected. Put every bout in one cluster, let majority vote map it to
the commonest behavior, and that behavior gets recall 1.0 and precision equal to its prevalence
— a good-looking number off a fit that learned nothing: a degenerate 2-cluster fit can score
well on a prevalent behavior while its AMI is near zero.

The subtle part, and the reason the obvious implementation is wrong: the naive floor `2p/(1+p)`
sits *below* what a degenerate fit really scores, because
its unassigned frames raise precision. A floor the degenerate answer beats certifies the thing
it exists to catch. So the floor matches the real prediction's coverage.

This is the cluster-eval counterpart of the nulls ``analysis/query_eval`` has shipped since it
learned the same lesson.
"""

import numpy as np

from detail.analysis import cluster_metrics as cm


def _truth(n_common: int, n_rare: int) -> np.ndarray:
    return np.array([0] * n_common + [1] * n_rare, dtype=np.int64)


def test_the_floor_matches_the_analytic_value_at_full_coverage():
    """With everything assigned, the floor is exactly 2p/(1+p) for the dominant behavior."""
    truth = _truth(586, 414)
    pred = np.zeros_like(truth)  # one cluster, full coverage
    floor = cm.one_cluster_floor(truth, pred)

    p = 586 / 1000
    assert floor[0]["f1"] == np.float64(2 * p / (1 + p)).item()
    assert floor[0]["recall"] == 1.0
    assert floor[0]["precision"] == p


def test_a_degenerate_fit_does_not_beat_its_own_floor():
    """The property that makes the floor worth shipping — and that the naive one fails.

    A fit leaving some frames unassigned scores *above* 2p/(1+p) on the dominant behavior. If
    the floor ignored coverage it would sit below that, and the table would read as though the
    degenerate fit had beaten the trivial answer.
    """
    truth = _truth(586, 414)
    pred = np.zeros_like(truth)
    pred[900:] = -1  # 10% unassigned, exactly the shape that inflated the real number

    scored = cm.per_behavior_frame_prf(
        truth, pred, cm.majority_vote_map(truth, pred), noise_label=-1
    )
    floor = cm.one_cluster_floor(truth, pred)

    naive = 2 * 0.586 / 1.586
    assert scored[0]["f1"] > naive, "the setup must reproduce the inflation"
    assert scored[0]["f1"] <= floor[0]["f1"] + 1e-12, (
        "a fit that learned nothing must not score above the floor — a floor it beats "
        "certifies exactly what it exists to catch"
    )


def test_a_real_clustering_beats_the_floor():
    """The floor must not be so generous that a genuine result cannot clear it."""
    truth = _truth(586, 414)
    pred = truth.copy()  # perfect clustering
    scored = cm.per_behavior_frame_prf(
        truth, pred, cm.majority_vote_map(truth, pred), noise_label=-1
    )
    floor = cm.one_cluster_floor(truth, pred)
    assert scored[1]["f1"] > floor[1]["f1"], "a perfect fit must clear the floor"
