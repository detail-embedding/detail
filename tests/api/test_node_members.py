"""A node's clips and their typicality order (``NodeView.clips``, ``typical_members``), per method.

**The class: a picture budget mistaken for a reachability rule, and then a ranking that lived
only where pictures were.** The node cache once listed members only for nodes over a size floor,
and every other node was ranked on demand from the fit. Now every node lists every clip from the
cache, and the order has to be **the method's own order**, or "most typical first" means one thing
in an export (``rank_members``) and another on screen. Pinned here per method:

- ``codebook_viterbi``: a clip is one leaf run (one action, never joined with its neighbours)
  and its score is its frames' mean decode distance, so at a leaf, where a run is an instance,
  clips and order equal ``rank_members`` exactly; above a leaf a node's clips are its children's,
  each keeping its own score, in that score's order.
- the handle's rows against the view's order, on the same frame method (this was pinned on pose
  ``finch`` until pose clustering was removed).
- ``dtw_finch``: medoid order is ranked on demand from the fit and kept per node; its order is
  ``rank_members``' order.
"""

from __future__ import annotations

import numpy as np
import pytest

from detail.analysis import node_cache as tables_lib
from detail.api.node_view import NodeView
from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.registries import TEMPORAL_CLUSTER_METHODS
from detail.storage import node_cache as store

STEP = 17_000_000


def _codebook_view():
    """A real codebook fit over two recordings, its leaf tables built the way the cache build
    builds them, and the view over those — no executor run, no experiment on disk."""
    import_temporal_cluster_methods()
    method = TEMPORAL_CLUSTER_METHODS["codebook_viterbi"]()
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(3, 8))
    trajectories = []
    for recording in ("recording_001", "recording_002"):
        block = np.repeat(anchors, 90, axis=0)
        rows = np.vstack([block + 0.3 * rng.normal(size=block.shape) for _ in range(2)])
        t_ns = np.arange(len(rows), dtype=np.int64) * STEP
        t_ns[len(rows) // 2 :] += 40 * STEP  # a tracking gap: two continuity blocks
        trajectories.append((recording, 0, rows.astype(np.float32), t_ns))
    params = {"window_s": 0.05, "min_run_s": 0.1, "fit_share": 40.0, "codewords": 30}
    fit = method.fit(trajectories, params)
    return method, fit, _view_for(method, fit, params)


def _view_for(method, fit, params):
    """The view the cache build would give ``fit`` under ``params``."""
    scoring = method.clip_scoring(fit, params)
    tree = method.hierarchy_of(fit)
    tables = tables_lib.leaf_tables(
        method.member_places(fit),
        method.member_leaves(fit),
        tree.n_leaves,
        method.node_separations(fit),
        scoring=scoring,
    )
    cache = store.NodeCache(
        leaves=tables,
        built_from={},
        clips={"kind": scoring.kind, "min_members": scoring.min_members, "join": True},
    )
    return NodeView(cache, tree)


def _keys(clips) -> list[tuple[int, int, int]]:
    return sorted(
        zip(clips.recording.tolist(), clips.subject.tolist(), clips.start_ns.tolist())
    )


def test_codebook_clips_and_their_order_are_rank_members_at_every_leaf():
    method, fit, view = _codebook_view()
    compared = 0
    for node in range(view.tree.n_leaves):
        ranked = method.rank_members(fit, view.leaves_under(node))
        clips = view.clips(node)
        order = clips.order("typical")
        assert [
            (
                fit.recording_ids[fit.recording_index[i.start_row]],
                int(fit.t_ns[i.start_row]),
            )
            for i, _d in ranked
        ] == [
            (clips.recording_ids[clips.recording[o]], int(clips.start_ns[o]))
            for o in order
        ], node
        assert [int(fit.t_ns[i.end_row]) for i, _d in ranked] == [
            int(clips.end_ns[o]) for o in order
        ], node
        assert np.allclose(
            [d for _i, d in ranked], clips.typicality[order], atol=1e-6
        ), node
        compared += len(ranked)
    assert compared > view.tree.n_leaves, "the fixture must give leaves several clips"


def test_a_codebook_node_lists_each_action_once_with_its_own_score():
    """A clip is one action (review round 2, choice B): a node's clips are its children's, none
    twice, each with the score it has at its leaf; the root lists every kept leaf run. Joining
    a node's touching runs (as clips once did) gives the root a handful of long clips and fails
    the root count."""
    _method, _fit, view = _codebook_view()
    t = view._cache.leaves
    by_leaf = {}
    for leaf in range(view.tree.n_leaves):
        clips = view.clips(leaf)
        for key, score in zip(
            zip(
                clips.recording.tolist(),
                clips.subject.tolist(),
                clips.start_ns.tolist(),
            ),
            clips.typicality.tolist(),
        ):
            by_leaf[key] = score
    for node in range(view.tree.n_leaves, view.tree.n_nodes):
        clips = view.clips(node)
        keys = _keys(clips)
        assert len(set(keys)) == len(keys), node
        assert keys == sorted(
            k for c in view.children(node) for k in _keys(view.clips(c))
        )
        order = clips.order("typical")
        mine = list(
            zip(
                clips.recording.tolist(),
                clips.subject.tolist(),
                clips.start_ns.tolist(),
            )
        )
        assert np.allclose(
            clips.typicality[order], [by_leaf[mine[o]] for o in order]
        ), node
        assert list(clips.typicality[order]) == sorted(clips.typicality), node
    root = view.clips(view.root)
    assert len(root) == int((t.run_rows >= view.clip_rule["min_members"]).sum())
    assert len(root) > 4, "two recordings × two blocks: joining would give 4 clips"
    assert len(set(root.recording.tolist())) == 2


def test_least_typical_is_the_ranking_reversed():
    method, fit, view = _codebook_view()
    node = max(
        range(view.tree.n_nodes), key=lambda n: len(view.clips(n)) * (n != view.root)
    )
    clips = view.clips(node)
    assert list(clips.typicality[clips.order("atypical")]) == sorted(
        clips.typicality, reverse=True
    )


@pytest.fixture
def handle(fitted_codebook):
    """A frame method's run (``codebook_viterbi``), whose clip order the cache already holds.
    These two tests ran on a pose-clustering fit until pose clustering was removed."""
    experiment, run_id = fitted_codebook
    return experiment.temporal_clustering_run(run_id)


def _spans(rows):
    return [
        (r["recording_id"], int(r["subject"]), int(r["start_t_ns"]), int(r["end_t_ns"]))
        for r in rows
    ]


def test_a_frame_node_s_rows_come_from_the_cache_without_the_fit(handle, monkeypatch):
    view = handle.node_view()
    monkeypatch.setattr(
        type(handle), "_load_fit", lambda self: pytest.fail("loaded the fit")
    )
    seen = []
    for node in range(view.tree.n_nodes):
        clips = view.clips(node)
        rows = handle.typical_members(node, progress=lambda *a: seen.append(a))
        assert _spans(rows) == _spans(clips.rows(clips.order("typical"))), node
    assert seen == [], "a leaf-additive method costs nothing, so it reports nothing"


def test_a_limit_takes_the_most_typical(handle):
    view = handle.node_view()
    node = next(n for n in range(view.tree.n_nodes) if len(view.clips(n)) > 2)
    every = handle.typical_members(node)
    assert _spans(handle.typical_members(node, 2)) == _spans(every[:2])


def test_a_dtw_node_is_ranked_on_first_ask_and_kept(fitted_temporal, monkeypatch):
    """Medoid order depends on the node, so the first ask loads the fit and ranks (with
    progress); the second answers from memory. The order is ``rank_members``'."""
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    assert view.ranks_on_demand
    node = view.root
    assert np.isnan(
        view.clips(node).typicality
    ).all(), "the cache holds no medoid order"
    seen = []
    clips = run.clips(node, progress=lambda *a: seen.append(a[0]))
    assert set(seen) == {"ranking clips"}
    method, fit = run._method(), run._load_fit()
    resolve = run._member_resolver(method, fit)
    ranked = [
        resolve(i)[2] for i, _d in method.rank_members(fit, view.leaves_under(node))
    ]
    ordered = clips.start_ns[clips.order("typical")].tolist()
    assert ordered[: len(ranked)] == ranked
    monkeypatch.setattr(
        type(run), "_load_fit", lambda self: pytest.fail("ranked the node twice")
    )
    again = run.clips(node, progress=lambda *a: pytest.fail("reported a cached answer"))
    assert np.array_equal(again.typicality, clips.typicality, equal_nan=True)
    assert len(clips) == view.summary(node).members, "a bout is its own clip"


@pytest.mark.parametrize("max_gap_steps", [60, 1])
def test_codebook_clips_follow_an_explicit_gap_threshold(max_gap_steps):
    """``max_gap_s`` decides where ``instances_for`` cuts. Set above the fixture's 40-step gap,
    runs on either side are one instance; set below the frame step, every frame is its own
    block. A leaf's clips must cut where its instances do, not where the default threshold
    would, and above a leaf no clip may cross one of the node's own cuts.
    """
    method, fit, _view = _codebook_view()
    params = {
        "window_s": 0.05,
        "min_run_s": 0.1,
        "fit_share": 40.0,
        "codewords": 30,
        "max_gap_s": max_gap_steps * STEP / 1e9,
    }
    view = _view_for(method, fit, params)
    for node in range(view.tree.n_nodes):
        groups = np.full(view.tree.n_leaves, -1, dtype=np.int64)
        groups[view.leaves_under(node)] = 0
        own = [
            (int(fit.t_ns[i.start_row]), int(fit.t_ns[i.end_row]))
            for i in method.instances_for(fit, groups, params)
            if i.cluster_id == 0
        ]
        clips = view.clips(node)
        spans = sorted(zip(clips.start_ns.tolist(), clips.end_ns.tolist()))
        if node < view.tree.n_leaves:
            assert sorted(own) == spans, node
        for start, end in spans:
            assert any(
                a <= start and end <= b for a, b in own
            ), f"node {node}: a clip crosses a cut of the method's own instances"
