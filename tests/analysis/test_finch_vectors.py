"""The vector FINCH path is the dense one, computed without the matrix.

`finch_hierarchy_from_vectors` exists because the dense path is bounded by memory: a precomputed
distance matrix is 1.15 GB at 12,000 rows and impossible at the row counts a frame-level pose fit
wants. It finds neighbours by a blocked scan instead, and recovers the dense path's average linkage
from plain cluster means — for unit vectors, ``mean_{a,b} (1 - x_a·x_b) = 1 - m_A·m_B``, because the
double sum factorizes.

The class this guards: **two mechanisms computing the same named thing.** Each is individually
plausible, and a disagreement surfaces as a different vocabulary rather than an error — nobody gets
a stack trace, they get different clusters. The first attempt here *did* diverge, by renormalizing
those cluster means, which is the obvious thing to do and what a reading of the paper suggests: it
gave 337/81/19 where the dense path gives 285/63/12 on a real 6,000-row fit, agreeing only at level
0 where no averaging has happened yet. So the test asserts the hierarchies are **identical at every
level**, not that they are similar — a weaker assertion would have passed the broken version.

The schema edges (empty, single row, identical rows) are tested for the separate reason that a
clustering rule which raises on a degenerate part is a bug no happy path ever sees.
"""

import numpy as np
import pytest

from detail.analysis import segmentation as seg


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _dense_of(feats: np.ndarray) -> np.ndarray:
    unit = _unit(feats)
    dist = 1.0 - unit @ unit.T
    np.fill_diagonal(dist, 0.0)
    return dist


def _blobs(seed: int, n: int = 150, dim: int = 10, groups: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(groups, dim))
    rows = np.repeat(centers, n // groups, axis=0)
    return rows + 0.3 * rng.normal(size=rows.shape)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_two_paths_produce_the_identical_hierarchy(seed):
    """Every level, not just the first, and identical labels rather than matching counts.

    This is the whole guarantee. It holds because average linkage over cosine distances equals
    ``1 - m_A·m_B`` on plain cluster means, so the vector path recovers the dense path's
    aggregation exactly instead of approximating it. Renormalizing those means — the obvious
    thing, and what the paper's own description suggests — breaks it: measured 337/81/19 against
    the dense path's 285/63/12 on a real 6,000-row fit.
    """
    feats = _blobs(seed)
    vector = seg.finch_hierarchy_from_vectors(feats)
    dense = seg.finch_hierarchy(_dense_of(feats))
    assert len(vector) == len(dense)
    for got, want in zip(vector, dense):
        assert np.array_equal(got, want)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_hierarchy_coarsens_monotonically(seed):
    levels = seg.finch_hierarchy_from_vectors(_blobs(seed))
    counts = [int(lab.max()) + 1 for lab in levels]
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_level_nests_inside_the_next(seed):
    """The property §22 lets a method declare, and the reason names can carry across a granularity
    change. It is true by construction — the recursion composes `labels = step[labels]` — which is
    exactly why it needs a test rather than a comment: a change that broke it would hand a name to
    a cluster that is half something else, and look deliberate."""
    levels = seg.finch_hierarchy_from_vectors(_blobs(seed))
    for fine, coarse in zip(levels, levels[1:]):
        for cluster in range(int(fine.max()) + 1):
            members = coarse[fine == cluster]
            assert len(np.unique(members)) == 1


def test_a_recurring_pose_gets_the_same_id_each_time():
    """The property TW-FINCH cannot deliver, and the reason a pose method must not use it.

    `tests/analysis/test_tw_finch.py` pins the counterpart: its temporal weighting makes two
    occurrences of one posture look dissimilar however alike they are, so an A-B-A-C-A trajectory
    comes back as five contiguous chunks on five labels. For a posture vocabulary that is the wrong
    model — recurrence is the whole point — so the plain rule has to recover it.
    """
    rng = np.random.default_rng(11)
    block = 30
    a, b, c = rng.normal(size=(3, 8))
    order = [a, b, a, c, a]
    traj = np.vstack(
        [np.tile(v, (block, 1)) + 0.05 * rng.normal(size=(block, 8)) for v in order]
    )
    levels = seg.finch_hierarchy_from_vectors(traj)
    usable = [lab for lab in levels if int(lab.max()) + 1 >= 3]
    labels = seg._refine_to(_dense_of(traj), levels, 3) if usable else levels[0]
    assert labels[0] == labels[2 * block] == labels[4 * block]


def test_identical_rows_collapse_rather_than_raising():
    """A degenerate fit — every row the same pose — must return one cluster, not divide by zero
    normalizing a zero-length mean."""
    levels = seg.finch_hierarchy_from_vectors(np.ones((40, 6)))
    assert levels
    assert int(levels[-1].max()) + 1 == 1


def test_the_empty_and_single_row_edges_do_not_raise():
    assert seg.finch_hierarchy_from_vectors(np.zeros((0, 5))) == []
    single = seg.finch_hierarchy_from_vectors(np.ones((1, 5)))
    assert len(single) == 1 and single[0].tolist() == [0]


def test_the_blocked_scan_finds_the_same_neighbours_as_a_full_matrix():
    """`_nearest_by_dot` blocks the scan to keep memory O(N); the block boundary is where an
    off-by-one in the diagonal fill would hide, so it is checked against the whole matrix.
    """
    feats = _unit(_blobs(5, n=120, dim=7))
    dense = 1.0 - feats @ feats.T
    np.fill_diagonal(dense, np.inf)
    expected = np.argmin(dense, axis=1)
    for block in (7, 32, 512):
        assert np.array_equal(seg._nearest_by_dot(feats, block=block), expected)
