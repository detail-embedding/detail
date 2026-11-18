"""Fusing same-label runs never crosses a tracking gap.

**The class:** a later step undoing a cut an earlier step made on purpose. The codebook method
cuts its runs at tracking gaps (``codebook_viterbi._runs_of``: the animal was not observed in
between, so an instance spanning the gap would claim time nobody saw). The module then fused
every two consecutive same-label instances in one recording and subject
(``authored_run.fuse_runs``), with no look at the time between them, and undid the cut. One
labelled behavior on both sides of a 34 s gap became one 39.6 s instance in 6.1 s of observed
time: ``time_fraction`` 6.47, ``n_instances`` 1 instead of 2, and nothing raised.

**How it hid:** each step was right on its own. The method's tests saw two runs, the module's
saw a join across a dropped flicker, and no fixture had a gap with the same label on both sides.

**The rule:** the method says where joining is allowed (``row_block_starts``, the same blocks
its runs and its clips are cut on) and ``fuse_runs`` joins only inside one block.
"""

import numpy as np

from detail.api.authored_run import fuse_runs
from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.dataclasses import ClusterInstance
from detail.core.registries import TEMPORAL_CLUSTER_METHODS

STEP = 17_000_000  # ns between rows, the ~59 rows/s an embed run writes
GAP_NS = 34_000_000_000  # a 34 s tracking dropout
PARAMS = {"window_s": 0.05, "min_run_s": 0.1, "fit_share": 40.0, "codewords": 4}


def _fit_with_a_gap():
    """One pose held for 120 rows, with a 34 s dropout after row 60: one behavior, two blocks."""
    import_temporal_cluster_methods()
    method = TEMPORAL_CLUSTER_METHODS["codebook_viterbi"]()
    rng = np.random.default_rng(0)
    pose = rng.normal(size=8)
    rows = (pose + 0.1 * rng.normal(size=(120, 8))).astype(np.float32)
    t_ns = np.arange(120, dtype=np.int64) * STEP
    t_ns[60:] += GAP_NS
    return method, method.fit([("rec", 0, rows, t_ns)], PARAMS)


def _gap_after(fit) -> int:
    """The fit's last row before the dropout (windowing trims rows, so not 59)."""
    (gap,) = np.nonzero(np.diff(np.asarray(fit.t_ns, dtype=np.int64)) > GAP_NS // 2)
    assert len(gap) == 1
    return int(gap[0])


def _observed_ns(fit) -> int:
    """The time the animal was seen: each block's span, never the gap between two."""
    t_ns = np.asarray(fit.t_ns, dtype=np.int64)
    gap = _gap_after(fit)
    return int(t_ns[gap] - t_ns[0]) + int(t_ns[-1] - t_ns[gap + 1])


def test_one_label_on_both_sides_of_a_gap_stays_two_instances():
    method, fit = _fit_with_a_gap()
    every_leaf_one_group = np.zeros(int(fit.tree.n_nodes), dtype=np.int64)
    found = method.instances_for(fit, every_leaf_one_group, PARAMS)
    assert len(found) == 2  # the method cut at the gap

    fused = fuse_runs(fit, method, found, PARAMS)

    assert len(fused) == 2
    spans = [method.resolve(fit, instance) for instance in fused]
    claimed = sum(span["end_t_ns"] - span["t_ns"] for span in spans)
    assert claimed / _observed_ns(fit) <= 1.0  # a share of time never above 100%


def test_runs_inside_one_block_still_join():
    """The join a dropped flicker needs is kept: the rule narrows it, it does not remove it."""
    method, fit = _fit_with_a_gap()
    gap, last = _gap_after(fit), len(fit.t_ns) - 1
    inside = [ClusterInstance(0, 20, 3), ClusterInstance(25, gap, 3)]
    across = [ClusterInstance(25, gap, 3), ClusterInstance(gap + 1, last, 3)]

    joined = fuse_runs(fit, method, inside, PARAMS)
    kept_apart = fuse_runs(fit, method, across, PARAMS)

    assert [(i.start_row, i.end_row) for i in joined] == [(0, gap)]
    assert [(i.start_row, i.end_row) for i in kept_apart] == [
        (25, gap),
        (gap + 1, last),
    ]
