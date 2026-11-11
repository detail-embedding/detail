"""Every cluster method carries its merge order, and serves every granularity from it.

Data-driven over both registries rather than written against FINCH, so a method added later is
covered the day it is added. The properties are the ones the naming model above this rests on, and
each is invisible to a behavioural test: a method that rebuilt a partition per request still
returns the right *number* of clusters, still renders, still runs to completion — and the
partitions it returns at two settings need not agree about which cluster a row is in.

``tests/analysis/test_cluster_tree.py`` pins the builder. This file pins the wiring: that each
method stores the tree, rebuilds it when a shard predates it, and no longer reaches for the
refinement it replaced.
"""

import numpy as np
import pytest

from detail.analysis import segmentation
from detail.cluster_methods import (
    import_pose_cluster_methods,
    import_temporal_cluster_methods,
)
from detail.core.dataclasses import ClusterTree
from detail.core.registries import POSE_CLUSTER_METHODS, TEMPORAL_CLUSTER_METHODS

import_pose_cluster_methods()
import_temporal_cluster_methods()

STEP = 33_000_000  # ~30 fps in nanoseconds


def _methods():
    return {**POSE_CLUSTER_METHODS, **TEMPORAL_CLUSTER_METHODS}


# Segmentation settings that actually cut a synthetic trajectory at its cadence. The shipped
# defaults are tuned for a real recording, where a 0.27 s minimum bout is nothing; here it would be
# most of the fixture, and the fit would collapse to one cluster with no setting between its rungs
# — which is the exact state these tests exist to exercise. The same override
# `tests/cluster_methods/test_dtw_finch.py` states, for the same reason.
_BOUT_PARAMS = {
    "window_s": 0.05,
    "min_len_s": 0.1,
    "band_s": 0.05,
    "penalty": 1.0,
    "min_size_s": 0.05,
}


def _trajectories(n_bouts: int = 24, bout_rows: int = 30, dim: int = 8, kinds: int = 6):
    """Two subjects walking repeating *bout kinds* — structure a bout comparison can find.

    Each kind is a smooth path between two anchors, so bouts of one kind resemble each other.
    Independent noise per row would give the pose methods a hierarchy and the bout method a single
    cluster, and a fixture that reaches only two of three registered methods is how a sweep ends up
    skipping the one it was written for.
    """
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(kinds, 2, dim))
    rows = []
    for index in range(n_bouts):
        ramp = np.linspace(0.0, 1.0, bout_rows)[:, None]
        start, end = anchors[index % kinds]
        rows.append(
            start * (1 - ramp) + end * ramp + 0.05 * rng.normal(size=(bout_rows, dim))
        )
    features = np.vstack(rows).astype(np.float32)
    half = len(features) // 2
    return [
        ("rec_a", 0, features[:half], np.arange(half, dtype=np.int64) * STEP),
        (
            "rec_b",
            0,
            features[half:],
            np.arange(len(features) - half, dtype=np.int64) * STEP,
        ),
    ]


def _params(cls) -> dict:
    params = cls.config_model().model_dump()
    return {**params, **{k: v for k, v in _BOUT_PARAMS.items() if k in params}}


@pytest.fixture(scope="module")
def fits():
    """One real fit per registered method. Module-scoped: three fits is the cost of this file."""
    out = {}
    for method_id, cls in _methods().items():
        method = cls()
        out[method_id] = (method, method.fit(_trajectories(), _params(cls)))
    return out


def _refines(fine: np.ndarray, coarse: np.ndarray) -> bool:
    return all(len(np.unique(coarse[fine == c])) == 1 for c in np.unique(fine))


def test_the_registries_are_populated():
    """A sweep over an empty registry passes by testing nothing."""
    assert len(_methods()) >= 2, sorted(_methods())


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_the_method_declares_its_hierarchy(method_id):
    """Declared on the class, not reached for on the fit — the mistake ``granularity_of`` made.

    A surface that draws the tree, names a node or cuts it goes through the protocol. A method
    whose tree is only reachable by knowing the shape of its private fit object is a method the
    module has to special-case, which §22 exists to prevent.
    """
    cls = _methods()[method_id]
    assert "hierarchy_of" in vars(cls), f"{method_id} must declare hierarchy_of"


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_the_tree_carries_the_methods_own_rungs(method_id, fits):
    """The tree and the granularity knob must describe one hierarchy, not two.

    If they drift, the slider offers a rung the tree cannot cut to and the page shows a count
    nothing produced — two numbers on screen that disagree, with nothing saying why.
    """
    method, fit = fits[method_id]
    tree = method.hierarchy_of(fit)
    assert isinstance(tree, ClusterTree)
    assert tree.rungs == method.granularity_of(fit).rungs


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_every_setting_nests_in_every_coarser_one(method_id, fits):
    """Driven through ``labels_at``, the call every surface actually makes.

    Both branches on purpose: a rung returns its stored level and anything else cuts the tree, so
    a test that only probed rungs would pass with the off-rung path still refining. This is the
    property that used to fail — 50 refined 12 and 30 did not, on a real fit.
    """
    method, fit = fits[method_id]
    spec = method.granularity_of(fit)
    probes = sorted({k for k in range(1, min(spec.maximum, 24) + 1)} | set(spec.rungs))
    labels = {k: np.asarray(fit.labels_at(k)) for k in probes}
    for fine in probes:
        for coarse in probes:
            if fine > coarse:
                assert _refines(
                    labels[fine], labels[coarse]
                ), f"{method_id}: {fine} does not refine {coarse}"


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_an_off_rung_setting_never_refines_a_matrix(method_id, fits, monkeypatch):
    """The allocation the tree removes, asserted as the call that made it.

    ``_refine_to`` is what built ``1 - sample @ sample.T`` — 0.29 GB at the default fit size and
    20 GB at the largest the form offers. Patching it to raise is stronger than measuring memory:
    it fails on the *mechanism* rather than on a threshold that a smaller fixture would slip
    under. ``_refine_to`` itself stays, because the ``finch`` cluster rule still uses it.
    """
    method, fit = fits[method_id]
    spec = method.granularity_of(fit)
    off_rung = next(
        (k for k in range(spec.maximum, spec.minimum - 1, -1) if not spec.is_rung(k)),
        None,
    )
    # Asserted, never skipped. A fit whose rungs happen to cover every integer would make this
    # test pass by never reaching the path it is about — the condition it exists to catch, wearing
    # a green tick. If a method ever legitimately has no gap, the fixture is what to change.
    assert (
        off_rung is not None
    ), f"{method_id}'s fit has no setting between rungs, so this test exercises nothing"

    def _refuse(*_args, **_kwargs):
        raise AssertionError(
            "labels_at refined a distance matrix instead of cutting the tree"
        )

    monkeypatch.setattr(segmentation, "_refine_to", _refuse)
    fit._labels.pop(off_rung, None)
    assert len(np.unique(fit.labels_at(off_rung))) == off_rung


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_a_shard_written_before_the_tree_still_opens(method_id, fits, tmp_path):
    """Real experiments hold fits from before this existed; they rebuild rather than refuse.

    Built by writing a real shard and deleting the tree keys, so it is the actual absence a loader
    meets — not a hand-made file that might differ from what ``serialize`` produces in some other
    way and pass for the wrong reason.
    """
    method, fit = fits[method_id]
    path = tmp_path / f"fit{method.shard_suffix}"
    method.serialize(fit, path)
    with np.load(path, allow_pickle=True) as data:
        kept = {k: data[k] for k in data.files if not k.startswith("tree_")}
    assert len(kept) < len(
        np.load(path, allow_pickle=True).files
    ), "nothing was stripped"
    np.savez_compressed(path, **kept)

    old = method.load(path)
    assert old._tree is None, "the stripped shard must exercise the rebuild"
    spec = method.granularity_of(old)
    for k in sorted(set(list(spec.rungs) + [max(spec.minimum, min(7, spec.maximum))])):
        assert np.array_equal(
            np.asarray(old.labels_at(k)), np.asarray(fit.labels_at(k))
        ), f"{method_id}: a rebuilt tree labels {k} differently"


def _node_of_cluster(method, fit, granularity: int) -> dict[int, int]:
    """Each cluster id at ``granularity`` mapped to its tree node, via the module's own join."""
    from detail.api import authored_run

    labels = fit.labels_at(granularity)
    nodes = authored_run.cluster_nodes(method.hierarchy_of(fit), labels, granularity)
    return {cluster: int(node) for cluster, node in enumerate(nodes)}


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_a_partition_by_the_cut_yields_the_cuts_instances(method_id, fits):
    """``instances_for`` over a cut's nodes is ``instances_at`` at that cut, run for run.

    The redesign's floor (§22): labels drive the output through ``instances_for``, and the cut path
    is what every run on disk was written with. If the two disagreed about where a run starts, where
    a gap cuts it or what is too short to keep, relabelling a run would move its instances for a
    reason nobody chose. Checked at every setting the fit offers, rungs and between.
    """
    method, fit = fits[method_id]
    tree = method.hierarchy_of(fit)
    params = _params(type(method))
    for granularity in range(max(fit.granularity.minimum, 1), tree.n_leaves + 1):
        node_of = _node_of_cluster(method, fit, granularity)
        old = [
            (i.start_row, i.end_row, node_of[int(i.cluster_id)])
            for i in method.instances_at(fit, granularity, params)
        ]
        new = [
            (i.start_row, i.end_row, int(i.cluster_id))
            for i in method.instances_for(fit, tree.leaf_nodes_at(granularity), params)
        ]
        assert new == old, f"{method_id} disagrees with its own cut at {granularity}"


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_member_leaves_are_leaves_of_the_tree(method_id, fits):
    """Every member lands on a real leaf, so every node's membership is well defined."""
    method, fit = fits[method_id]
    leaves = method.member_leaves(fit)
    tree = method.hierarchy_of(fit)
    assert leaves.dtype == np.int64
    assert len(leaves) and leaves.min() >= 0 and leaves.max() < tree.n_leaves


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_unlabelled_leaves_still_make_instances(method_id, fits):
    """A partition with nothing labelled is one unlabelled group, and it still covers the run."""
    method, fit = fits[method_id]
    tree = method.hierarchy_of(fit)
    found = method.instances_for(fit, tree.leaf_groups(set()), _params(type(method)))
    assert found and {int(i.cluster_id) for i in found} == {-1}


def test_leaf_groups_resolve_to_the_deepest_labelled_ancestor(fits):
    """A leaf under a labelled child reports the child; under only the parent, the parent."""
    method, fit = next(iter(fits.values()))
    tree = method.hierarchy_of(fit)
    root = tree.n_nodes - 1
    left = int(tree.children[root - tree.n_leaves][0])
    groups = tree.leaf_groups({root, left})
    under_left = set(tree.leaves_under(left).tolist())
    for leaf, group in enumerate(groups):
        assert group == (left if leaf in under_left else root)
        assert group == tree.deepest_named(leaf, {root, left})
    assert set(tree.leaf_groups(set()).tolist()) == {-1}


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_member_places_line_up_with_member_leaves(method_id, fits):
    """One place per member, in the same index space, with measured non-negative time."""
    method, fit = fits[method_id]
    places = method.member_places(fit)
    count = len(method.member_leaves(fit))
    for column in ("recording_index", "subject", "start_ns", "end_ns", "time_ns"):
        values = getattr(places, column)
        assert len(values) == count, column
        assert values.dtype == np.int64, column
    assert (places.time_ns >= 0).all() and places.time_ns.sum() > 0
    assert (places.end_ns >= places.start_ns).all()
    assert places.recording_index.max() < len(places.recording_ids)


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_ranking_a_node_is_ascending_and_nests_in_its_parent(method_id, fits):
    """Every node ranks most typical first, and ranks only members its parent also ranks.

    ``rank_members`` is the node card's one ordering (§22). Ascending is the direction the protocol
    states for every method; nesting is what the tree promises — a child's members are its
    parent's, so a child ranking a row the parent does not would be ranking something that is not
    under the node at all.
    """
    method, fit = fits[method_id]
    tree = method.hierarchy_of(fit)
    root = tree.n_nodes - 1
    root_rows = _covered(method.rank_members(fit, tree.leaves_under(root)))
    assert root_rows, f"{method_id} ranks nothing at the root"
    for merge, (a, b) in enumerate(np.asarray(tree.children, dtype=np.int64)):
        parent = _covered(
            method.rank_members(fit, tree.leaves_under(tree.n_leaves + merge))
        )
        for child in (int(a), int(b)):
            order = method.rank_members(fit, tree.leaves_under(child))
            distances = [d for _, d in order]
            assert distances == sorted(distances), f"{method_id} node {child}"
            assert _covered(order) <= parent, f"{method_id} node {child}"


def _covered(order) -> set[int]:
    """The member indices a ranking's instances span."""
    out: set[int] = set()
    for instance, _distance in order:
        out.update(range(int(instance.start_row), int(instance.end_row) + 1))
    return out


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_a_fit_saved_with_the_retired_layout_still_loads(method_id, fits, tmp_path):
    """A shard an older build wrote carries the 2-D layout fields; loading must ignore them.

    ``projection_points``, ``projection_layout`` and ``coords`` were written by every method's
    ``serialize`` until the interpretation screen stopped drawing a scatter (§22). Fits on disk in
    real experiments still have them, and a run that will not reopen loses its naming work.
    """
    method, fit = fits[method_id]
    path = tmp_path / f"fit{method.shard_suffix}"
    method.serialize(fit, path)
    with np.load(path, allow_pickle=True) as data:
        stored = {key: data[key] for key in data.files}
    assert not {"projection_points", "projection_layout", "coords"} & set(stored)
    np.savez_compressed(
        path,
        **stored,
        projection_points=50000,
        projection_layout="umap",
        coords=np.zeros((len(method.member_leaves(fit)), 3), dtype=np.float64),
    )

    loaded = method.load(path)
    assert np.array_equal(method.member_leaves(loaded), method.member_leaves(fit))
    tree = method.hierarchy_of(loaded)
    root = tree.leaves_under(tree.n_nodes - 1)
    assert method.rank_members(loaded, root) == method.rank_members(fit, root)


@pytest.mark.parametrize("method_id", sorted(_methods()))
def test_every_internal_node_has_a_separation(method_id, fits):
    """Leaves have none (``NaN``); internal nodes whose children have members get a number."""
    method, fit = fits[method_id]
    tree = method.hierarchy_of(fit)
    separations = method.node_separations(fit)
    assert separations.shape == (tree.n_nodes,)
    assert np.isnan(separations[: tree.n_leaves]).all()
    internal = separations[tree.n_leaves :]
    assert np.isfinite(internal).mean() > 0.5
    assert (internal[np.isfinite(internal)] >= 0).all()


def test_frame_time_counts_the_cadence_not_an_advertised_rate():
    """A fit analysed at every third frame counts three frames of time per row.

    Frames at a true 30 fps, analysed every third: each row stands for ~100 ms. Dividing a row
    count by any advertised fps would give 33 ms (or, from a header claiming 25 fps, 40 ms), which
    is the bug this rules out.
    """
    from detail.cluster_methods._common import measured_frame_time

    true_step = 33_333_333
    t_ns = np.arange(0, 30, dtype=np.int64) * 3 * true_step
    time = measured_frame_time(t_ns, [(0, len(t_ns))])
    assert (time == 3 * true_step).all()
    assert int(time.sum()) == int(t_ns[-1] - t_ns[0]) + 3 * true_step


def test_frame_time_never_crosses_a_gap_and_a_lone_frame_takes_the_run_step():
    from detail.cluster_methods._common import measured_frame_time

    t_ns = np.array([0, 10, 20, 1_000, 1_010, 5_000], dtype=np.int64)
    time = measured_frame_time(t_ns, [(0, 3), (3, 5), (5, 6)])
    assert time.tolist() == [10, 10, 10, 10, 10, 10]
