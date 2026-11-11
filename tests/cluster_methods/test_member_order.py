"""A node's members come back **whole**, ordered, with the method's own distance.

The class: **a cap taken off the wrong end.** `representatives(n)` returned the ``n`` members
nearest the middle of a cluster and threw the rest away — so the surface that shows a cluster
could only ever show its typical half. What tells a user a cluster is *one behavior* is the
opposite half: a grooming cluster whose farthest members are still grooming is one, and one whose
farthest members are rearing is two. A user hit this live, watching a strip whose first clips all
looked alike and then finding one that did not.

The per-cut hooks (`member_order`, `representatives`) are gone (§22); the node card ranks through
`rank_members`, so this pins the same two properties there: the ordering leaves no member out, and
smaller is more typical for every method.

Data-driven over both registries, so a method registered tomorrow is covered the day it lands
rather than the day someone remembers this file.
"""

import types

import numpy as np
import pytest

from detail.cluster_methods import (
    import_pose_cluster_methods,
    import_temporal_cluster_methods,
)
from detail.core.registries import POSE_CLUSTER_METHODS, TEMPORAL_CLUSTER_METHODS

import_pose_cluster_methods()
import_temporal_cluster_methods()

#: More members under the leaf than any cap a surface shows (12 typical, 24 sampled), so a ranking
#: cut to a top-n leaves members out and the coverage test sees it. Measured: with four members a
#: `[:12]` truncation passed every test here.
LABELS = np.tile(np.array([0, 1, 0, 0, 1, 0]), 10)


def _methods():
    return {**POSE_CLUSTER_METHODS, **TEMPORAL_CLUSTER_METHODS}


def test_the_registries_are_populated():
    """A sweep over an empty registry passes by testing nothing."""
    assert len(_methods()) >= 2, sorted(_methods())


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_rank_members_is_declared(method_id):
    """The ordering is the method's job, declared on the class rather than inherited by accident."""
    cls = _methods()[method_id]
    assert "rank_members" in vars(cls), f"{method_id} must declare rank_members"


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_rank_members_returns_every_member(method_id):
    """The whole ordering, not a top-n — the property the head-and-tail split depends on.

    Driven through a stub fit rather than a real one: what is under test is the contract, and a
    real fit per method would make this a slow test of numpy instead.
    """
    order = _rank(_methods()[method_id], LABELS, leaf=0)
    # **Coverage**, not a count. A count has to assume the unit, and the methods do not share
    # one: `dtw_finch` ranks bouts, the pose methods rank single rows, `codebook_viterbi`
    # run-length encodes adjacent rows into runs. What is true of all of them is that the
    # ordering leaves no member out: every member under the leaf falls inside some instance.
    covered = set()
    for instance, _distance in order:
        covered.update(range(int(instance.start_row), int(instance.end_row) + 1))
    expected = set(int(i) for i in np.flatnonzero(LABELS == 0))
    assert covered == expected, "the tail was truncated — the split has nothing to show"


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_the_distance_is_ascending_so_smaller_is_more_typical(method_id):
    """One direction across every method.

    `finch` once ranked by cosine *similarity*, where bigger is more typical — the opposite of the
    others. A surface ordering two methods' members by their raw score would put one of them
    exactly backwards, so the protocol states the direction and this pins it.
    """
    order = _rank(_methods()[method_id], LABELS, leaf=0)
    distances = [d for _instance, d in order]
    assert distances, f"{method_id} ranked nothing"
    assert distances == sorted(distances), f"{method_id} ranks backwards: {distances}"


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_a_node_with_no_members_ranks_to_nothing(method_id):
    """A leaf no member sits in is a lookup that returns nothing, not a raise."""
    assert _rank(_methods()[method_id], np.array([1, 1, 1]), leaf=0) == []


# --- stubs ------------------------------------------------------------------------- #
def _stub_fit(labels: np.ndarray):
    """The smallest object each method's `rank_members` reads; ``labels`` are the leaves."""
    n = len(labels)
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(n, 4))
    sample /= np.linalg.norm(sample, axis=1, keepdims=True)
    dist = np.abs(rng.normal(size=(n, n)))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return types.SimpleNamespace(
        levels=[labels],
        fine_labels=labels,
        sample=sample,
        dist=dist,
        tree=types.SimpleNamespace(n_leaves=2),
        fine_distance=np.abs(rng.normal(size=n)),
        min_run_s=0.0,
        # One row per second, adjacent, so a run-length encoder sees one block and no gaps.
        t_ns=np.arange(n, dtype=np.int64) * 1_000_000_000,
        # One recording, one animal: a run is cut per unit before any gap rule.
        recording_index=np.zeros(n, dtype=np.int64),
        subjects=np.zeros(n, dtype=np.int64),
    )


def _rank(cls, labels: np.ndarray, *, leaf: int):
    return cls().rank_members(_stub_fit(labels), np.array([leaf]))
