"""No instance spans time nobody observed — after labelling, not only out of the method.

**The class.** A method cuts its runs at the timeline's gaps (a stretch tracking lost), so that
an instance never claims unseen time as the behavior. A later step that joins runs — here the
module fusing two runs of one label across a dropped flicker, ``authored_run.fuse_runs`` —
undid that cut: it checked only that both runs were in one recording and subject. Two runs of a
label on either side of a gap became one, its span covered the gap, and every time summed from
``instances.csv`` (the BROWSE totals, §23's measures, the coverage line) counted the gap as the
behavior. The coverage line read **123% "in a posture"** on a zebrafish recording tracking lost
a fifth of, because the denominator (``coverage.observed_ns``) leaves gaps out.

**How it hid.** Each half was right and tested alone: ``test_finch.py`` pins that the method's
own runs stop at a gap, and the fusion was written for flickers, which lie inside observed time.
Nothing checked the method's output *after* the module had touched it. The node cache's time
(``member_places``) stops at gaps too, so INTERPRET's numbers stayed right and nothing looked off
until a run with no labels made a whole recording one instance.

**The rule.** The method declares where its runs may not cross (``run_blocks``) and every step
that joins runs joins inside one block. Data-driven over both registries, so a new run-building
method is covered the day it is registered — and one that builds runs without declaring its
blocks fails here rather than fusing across a gap.
"""

import numpy as np
import pytest

from detail.api import authored_run
from detail.cluster_methods import import_pose_cluster_methods
from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.dataclasses import ClusterInstance
from detail.core.protocols import PoseClusterMethod
from detail.core.protocols import TemporalClusterMethod
from detail.core.registries import POSE_CLUSTER_METHODS
from detail.core.registries import TEMPORAL_CLUSTER_METHODS

import_pose_cluster_methods()
import_temporal_cluster_methods()

_EVERY = {**POSE_CLUSTER_METHODS, **TEMPORAL_CLUSTER_METHODS}
RUN_METHODS = sorted(m for m, cls in _EVERY.items() if cls.instances_are_runs)

STEP = 17_000_000  # ~59 rows/s, the cadence a real embedding run has
GAP_AT = 360  # the first row after the gap
GAP_STEPS = 60  # ~1 s nobody observed

# Small fixtures want every row fitted: the shipped shares would sample a handful of them.
_OVERRIDES = {"fit_share": 100.0, "min_run_s": 0.0, "window_s": 0.05, "codewords": 40}


def _params(method) -> dict:
    fields = method.config_model.model_fields
    params = method.config_model().model_dump()
    params.update({k: v for k, v in _OVERRIDES.items() if k in fields})
    return params


def _trajectory(kinds=3, hold=120, repeats=2, dim=8, noise=0.15, seed=0):
    """Poses held for a stretch each, with one gap tracking lost in the middle."""
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(kinds, dim))
    block = np.repeat(anchors, hold, axis=0)
    rows = np.vstack(
        [block + noise * rng.normal(size=block.shape) for _ in range(repeats)]
    )
    t_ns = np.arange(len(rows), dtype=np.int64) * STEP
    t_ns[GAP_AT:] += GAP_STEPS * STEP
    return [("recording_001", 0, rows.astype(np.float32), t_ns)]


def _first_row_after_the_gap(method, fit) -> int:
    """The member row the gap ends at, read off the method's own timestamps: a method that
    windows its rows does not keep the trajectory's row numbers."""
    t_ns = np.asarray(method.member_places(fit).start_ns)
    (after,) = np.nonzero(np.diff(t_ns) > GAP_STEPS * STEP)[0] + 1
    return int(after)


@pytest.fixture(scope="module", params=RUN_METHODS)
def fitted(request):
    method = _EVERY[request.param]()
    params = _params(method)
    return method, method.fit(_trajectory(), params), params


def test_there_are_run_building_methods_to_check():
    assert RUN_METHODS, "no registered method builds runs; this guard checks nothing"


def test_one_label_on_both_sides_of_a_gap_is_two_instances(fitted):
    """Everything under one label is the worst case: every run is one group, so the module
    would fuse the whole recording into one instance covering the gap."""
    method, fit, params = fitted
    groups = np.zeros(method.hierarchy_of(fit).n_leaves, dtype=np.int64)
    found = method.instances_for(fit, groups, params)

    fused = authored_run.fuse_runs(fit, method, found, params)

    after = _first_row_after_the_gap(method, fit)
    spans = [(int(i.start_row), int(i.end_row)) for i in fused]
    assert spans == [(0, after - 1), (after, len(method.member_leaves(fit)) - 1)]


def test_labelled_time_never_exceeds_observed_time(fitted):
    """The coverage line's two numbers, in the fit's own terms: instance time over member time."""
    method, fit, params = fitted
    groups = np.zeros(method.hierarchy_of(fit).n_leaves, dtype=np.int64)
    fused = authored_run.fuse_runs(
        fit, method, method.instances_for(fit, groups, params), params
    )
    t_ns = method.member_places(fit).start_ns
    observed = int(method.member_places(fit).time_ns.sum())

    labelled = sum(int(t_ns[i.end_row]) - int(t_ns[i.start_row]) + STEP for i in fused)

    assert labelled <= observed + len(
        fused
    )  # rounding in the measured period, never a gap


def test_a_flicker_inside_observed_time_is_still_fused(fitted):
    """The fusion's job survives: two runs of one group split by a dropped flicker join."""
    method, fit, params = fitted
    rows, after = len(method.member_leaves(fit)), _first_row_after_the_gap(method, fit)
    pieces = [
        ClusterInstance(0, 99, 0),
        ClusterInstance(103, after - 1, 0),
        ClusterInstance(after, rows - 1, 0),
    ]

    fused = authored_run.fuse_runs(fit, method, pieces, params)

    assert [(i.start_row, i.end_row) for i in fused] == [
        (0, after - 1),
        (after, rows - 1),
    ]


@pytest.mark.parametrize("protocol", [PoseClusterMethod, TemporalClusterMethod])
def test_a_run_building_method_that_declares_no_blocks_is_refused(protocol):
    """The negative case: the protocol's default refuses, so fusing cannot fall back to units."""
    with pytest.raises(NotImplementedError, match="run_blocks"):
        protocol.run_blocks(object(), None, {})
