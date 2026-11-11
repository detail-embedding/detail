"""TW-FINCH and the frame path — what it does, and the two things it cannot do.

The limitations are pinned as tests rather than left as prose because both are easy to
misrepresent: the method segments a recording well, so a casual look suggests it is a
general-purpose clusterer when it is not.
"""

import numpy as np
import pytest

from detail.analysis import segmentation as seg
from detail.analysis.tw_finch import tw_finch_hierarchy, tw_finch_labels


def _recurring(seed=0, block=40, dim=16, noise=0.3):
    """A B A C A — the middle behavior recurs three times, the normal case for an animal."""
    rng = np.random.default_rng(seed)
    pattern = [0, 1, 0, 2, 0]
    truth = np.concatenate([[p] * block for p in pattern])
    centers = rng.normal(0, 1, (3, dim))
    traj = centers[truth] + rng.normal(0, noise, (len(truth), dim))
    times = (np.arange(len(truth)) * 33_000_000).astype(float)
    return traj, times, truth, block


def test_it_finds_where_the_behavior_changes():
    """What it is good at: every true block comes out as one pure, contiguous stretch."""
    traj, times, _truth, block = _recurring()
    labels = tw_finch_labels(traj, times, n_clusters=5)
    for i in range(5):
        stretch = labels[i * block : (i + 1) * block]
        assert len(set(stretch.tolist())) == 1, f"block {i} was split"


def test_a_recurring_behavior_gets_a_different_id_each_time():
    """The limitation that matters here, measured rather than assumed.

    The time weighting makes two occurrences far apart look dissimilar however alike the poses
    are, so a behavior cannot be recognised as the same one twice. For procedural video, where
    each action happens once in order, that is a feature; for an animal it is the wrong model.
    """
    traj, times, _truth, block = _recurring()
    labels = tw_finch_labels(traj, times, n_clusters=5)
    first, second, third = labels[0], labels[2 * block], labels[4 * block]
    assert not (first == second == third), (
        "TW-FINCH recovered a recurring behavior; if this now passes the temporal weighting "
        "changed and the docs claiming it cannot need revisiting"
    )


def test_a_distance_only_rule_does_recover_the_recurrence():
    """The contrast that makes the previous test a property of TW-FINCH, not of the fixture."""
    traj, _times, _truth, block = _recurring()
    unit = traj / np.linalg.norm(traj, axis=1, keepdims=True)
    dist = 1.0 - unit @ unit.T
    labels = seg.cluster_labels(dist, "agglomerative", {"n_clusters": 3})
    assert labels[0] == labels[2 * block] == labels[4 * block]


def test_the_hierarchy_coarsens_and_nests():
    traj, times, _truth, _block = _recurring()
    levels = tw_finch_hierarchy(traj, times)
    counts = [len(set(lab.tolist())) for lab in levels]
    assert counts == sorted(counts, reverse=True), counts
    for fine, coarse in zip(levels, levels[1:]):
        for cluster in set(fine.tolist()):
            assert len(set(coarse[fine == cluster].tolist())) == 1


def test_clusters_are_contiguous_in_time():
    """Its defining property: a label owns one stretch, never two separated ones."""
    traj, times, _truth, _block = _recurring()
    labels = tw_finch_labels(traj, times, n_clusters=5)
    for cluster in set(labels.tolist()):
        where = np.nonzero(labels == cluster)[0]
        assert (np.diff(where) == 1).all(), f"cluster {cluster} is not one stretch"


def test_subsampling_bounds_the_cost_and_still_labels_every_row():
    """max_rows is the visible bound on a quadratic search; every row must still get a label."""
    traj, times, _truth, _block = _recurring(block=200)
    labels = tw_finch_labels(traj, times, n_clusters=5, max_rows=150)
    assert len(labels) == len(traj)
    assert (labels >= 0).all()


def test_single_and_empty_trajectories_do_not_raise():
    assert len(tw_finch_labels(np.zeros((0, 4)), np.zeros(0))) == 0
    assert len(tw_finch_labels(np.zeros((1, 4)), np.zeros(1))) == 1


# --- the frame rule + RLE, as the pipeline uses them ---------------------------------- #
def test_the_frame_rule_keeps_runs_apart():
    """A label may not carry across a tracking gap: the runs are clustered independently."""
    traj, times, _truth, _block = _recurring(block=20)
    runs = [(0, 50), (50, len(traj))]
    labels = seg.FRAME_RULES["tw_finch"](traj, times, runs, n_clusters=3)
    assert set(labels[:50].tolist()).isdisjoint(set(labels[50:].tolist()))


def test_rle_turns_row_labels_into_the_assignment_rows():
    labels = np.array([0, 0, 0, 1, 1, 2])
    frame_idx = np.arange(6)
    t_ns = (np.arange(6) * 1000).astype(np.int64)
    rows = seg.labels_to_bout_rows(labels, frame_idx, t_ns)
    assert [r["syllable_id"] for r in rows] == [0, 1, 2]
    assert [(r["start_frame"], r["end_frame"]) for r in rows] == [
        (0, 2),
        (3, 4),
        (5, 5),
    ]


def test_rle_breaks_a_span_at_a_run_edge():
    """One unbroken label across a gap must still yield two rows: the animal was not observed
    doing the same thing throughout the gap."""
    labels = np.zeros(10, dtype=int)
    rows = seg.labels_to_bout_rows(
        labels,
        np.arange(10),
        (np.arange(10) * 1000).astype(np.int64),
        [(0, 5), (5, 10)],
    )
    assert len(rows) == 2
    assert (rows[0]["start_frame"], rows[0]["end_frame"]) == (0, 4)
    assert (rows[1]["start_frame"], rows[1]["end_frame"]) == (5, 9)


def test_rle_of_nothing_is_nothing():
    assert seg.labels_to_bout_rows(np.zeros(0), np.zeros(0), np.zeros(0)) == []
