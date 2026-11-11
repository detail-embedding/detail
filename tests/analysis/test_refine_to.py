"""``_refine_to``'s incremental average linkage returns exactly the naive algorithm's partition.

The class: **a hot loop rewritten for cost, where the cheap version and the expensive one can
silently disagree.** Nothing fails at the line where they diverge — the refinement still returns a
partition, the fit still completes, the clusterer still scores. The only signature is that the
groups are not the ones the documented rule produces, and no behavioural test would notice because
there is no independent statement of what those groups should be.

`_refine_to` used to rebuild `_average_linkage` from scratch after every merge, which is O(n²k)
*per merge* — free on the few hundred bouts the segment path produces, and unusable on the
thousands of rows a frame-level pose fit needs (a 6,000-row fit ran 18 minutes without finishing,
and 21 seconds after the change). Average linkage is a sum over the product of the cluster sizes,
so a merge adds the two summands: the Lance-Williams update, exact rather than approximate.

The guard is therefore not "does it run" but "is it the same partition", against a reference
implementation of the algorithm being replaced, written out here so the comparison does not depend
on the code under test. Revert the update and these fail.
"""

import numpy as np
import pytest

from detail.analysis import segmentation as seg


def _naive_refine_to(
    dist: np.ndarray, levels: list[np.ndarray], n_clusters: int
) -> np.ndarray:
    """The algorithm `_refine_to` used to be: rebuild the linkage after every merge.

    Kept here rather than in the module because its whole purpose is to be an *independent*
    statement of the answer. If it imported anything from the fast path it would stop being one.
    """
    usable = [lab for lab in levels if int(lab.max()) + 1 >= n_clusters]
    labels = np.array(usable[-1] if usable else levels[0], copy=True)
    while int(labels.max()) + 1 > n_clusters:
        linkage = seg._average_linkage(dist, labels)
        np.fill_diagonal(linkage, np.inf)
        i, j = np.unravel_index(np.argmin(linkage), linkage.shape)
        lo, hi = (int(i), int(j)) if i < j else (int(j), int(i))
        labels[labels == hi] = lo
        labels[labels > hi] -= 1
    return labels


def _cosine_dist(x: np.ndarray) -> np.ndarray:
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    dist = 1.0 - unit @ unit.T
    np.fill_diagonal(dist, 0.0)
    return dist


def _blobs(seed: int, n: int = 180, dim: int = 12, groups: int = 6) -> np.ndarray:
    """Points in a few loose groups — a matrix with real structure to merge, not noise."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(groups, dim))
    rows = np.repeat(centers, n // groups, axis=0)
    return rows + 0.35 * rng.normal(size=rows.shape)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_clusters", [2, 3, 5, 8, 13])
def test_the_fast_update_returns_the_naive_partition(seed, n_clusters):
    dist = _cosine_dist(_blobs(seed))
    levels = seg.finch_hierarchy(dist)
    assert np.array_equal(
        seg._refine_to(dist, levels, n_clusters),
        _naive_refine_to(dist, levels, n_clusters),
    )


def test_it_agrees_on_a_matrix_of_pure_noise_too():
    """Structured input could hide a tie-break difference behind large gaps between merges."""
    rng = np.random.default_rng(7)
    dist = _cosine_dist(rng.normal(size=(120, 8)))
    levels = seg.finch_hierarchy(dist)
    for target in (2, 4, 9):
        assert np.array_equal(
            seg._refine_to(dist, levels, target),
            _naive_refine_to(dist, levels, target),
        )


def test_a_target_at_a_natural_level_returns_that_level_untouched():
    """`_refine_to` starts from the tightest partition with enough clusters, so asking for a
    level's own count merges nothing. The pose method relies on this: a granularity setting that
    lands on a nested rung has to *be* the rung, or names could not carry across it."""
    dist = _cosine_dist(_blobs(3))
    levels = seg.finch_hierarchy(dist)
    for level in levels:
        count = int(level.max()) + 1
        if count > 1:
            assert np.array_equal(seg._refine_to(dist, levels, count), level)


def test_asking_for_more_clusters_than_the_finest_level_has_is_a_no_op():
    """The schema edge: no partition has enough clusters, so `usable` is empty and the finest
    level is returned rather than the loop running against an impossible target."""
    dist = _cosine_dist(_blobs(4))
    levels = seg.finch_hierarchy(dist)
    finest = int(levels[0].max()) + 1
    assert np.array_equal(seg._refine_to(dist, levels, finest + 50), levels[0])
