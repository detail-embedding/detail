"""The node cache a clustering fit writes (§22): its schema, its sums, and every node's clips.

Built on the session's temporal fits — ``fitted_temporal`` (``dtw_finch``, whose members are
bouts) and ``fitted_codebook`` (``codebook_viterbi``, whose members are frames) — which write the
cache in their ``caching`` phase, so no test here starts a run of its own. The frame-member tests
ran on a pose-clustering fit until pose clustering was removed. There is no size floor: every node
lists every clip, and the tests below hold each node's clips to the method's own instances of
that node and each clip's typicality to a computation from the full member vectors.
"""

import numpy as np
import pytest
import yaml
import zarr

from detail.analysis import node_cache as tables_lib
from detail.api import cluster_filter
from detail.pipeline import temporal_clustering as temporal_pipeline
from detail.storage import analysis_runs as analysis_store
from detail.storage import node_cache as store
from detail.storage import temporal_runs


@pytest.fixture
def cached(fitted_temporal):
    proj, run_id = fitted_temporal
    run_dir = analysis_store.analysis_dir(proj.dir, run_id)
    state = temporal_runs.read_temporal_state(run_dir)
    method = temporal_pipeline.resolve_method(state.method)
    fit = method.load(temporal_runs.fit_path(run_dir, method.shard_suffix))
    return run_dir, state, method, fit, store.read_node_cache(run_dir)


@pytest.fixture
def cached_frames(fitted_codebook):
    """``cached`` on the ``codebook_viterbi`` fit, whose members are frames and which has
    kinematics: for the tests that recompute a frame method's cache from its fit (a member's
    time, a leaf run's clips and positions)."""
    proj, run_id = fitted_codebook
    run_dir = analysis_store.analysis_dir(proj.dir, run_id)
    state = temporal_runs.read_temporal_state(run_dir)
    method = temporal_pipeline.resolve_method(state.method)
    fit = method.load(temporal_runs.fit_path(run_dir, method.shard_suffix))
    return run_dir, state, method, fit, store.read_node_cache(run_dir)


def test_a_fit_writes_its_node_cache_with_the_declared_schema(cached_frames):
    """Names, dtypes and chunks as §22 tables them; schema_version first; no floor anywhere."""
    run_dir, state, method, fit, cache = cached_frames
    root = zarr.open_group(store.cache_path(run_dir), mode="r")
    assert list(root.attrs)[0] == "schema_version"
    assert root.attrs["schema_version"] == store.SCHEMA_VERSION
    assert (
        root.attrs[store.BUILT_FROM_ATTR]["fit"]
        == temporal_runs.fit_path(run_dir, method.shard_suffix).name
    )
    clips = root.attrs[store.CLIPS_ATTR]
    assert clips["kind"] == "distance" and clips["join"] is True
    assert clips["min_members"] >= 1
    # The floor is gone from every place it was carried: the cache, the record and the state.
    assert "min_cluster_s" not in root.attrs and "pictured" not in root.attrs
    assert "min_cluster_s" not in root.attrs[store.BUILT_FROM_ATTR]
    assert "min_cluster_s" not in analysis_store.read_analysis_yaml(run_dir)["params"]
    on_disk = yaml.safe_load(
        (run_dir / "temporal_clustering.yaml").read_text(encoding="utf-8-sig")
    )
    assert "cache_floor_s" not in on_disk

    tree = method.hierarchy_of(fit)
    leaves, recordings = tree.n_leaves, len(state.source_recording_ids)
    expected = {
        "leaf_time_ns": ((leaves,), np.int64),
        "leaf_members": ((leaves,), np.int64),
        "leaf_rec_time_ns": ((leaves, recordings), np.int64),
        "leaf_rec_pos": (
            (leaves, recordings, tables_lib.POSITION_BINS, tables_lib.POSITION_BINS),
            np.uint32,
        ),
        "separation": ((tree.n_nodes,), np.float32),
        "run_recording": (None, np.int32),
        "run_subject": (None, np.int64),
        "run_start_ns": (None, np.int64),
        "run_end_ns": (None, np.int64),
        "run_leaf": (None, np.int32),
        "run_cx": (None, np.float32),
        "run_cy": (None, np.float32),
        "run_rows": (None, np.int32),
        "run_last_ns": (None, np.int64),
        "run_dist_sum": (None, np.float64),
    }
    for name, (shape, dtype) in expected.items():
        array = root[name]
        assert array.dtype == dtype, name
        if shape is not None:
            assert array.shape == shape, name
    runs = root["run_leaf"].shape[0]
    for name in ("run_cx", "run_rows", "run_last_ns", "run_dist_sum"):
        assert root[name].shape == (runs,), name
        assert root[name].chunks == (65_536,), name
    assert root["leaf_rec_pos"].chunks[1:] == (
        recordings,
        tables_lib.POSITION_BINS,
        tables_lib.POSITION_BINS,
    )
    # Nothing per node and nothing pictured: no member lists, no thumbnails.
    assert not [
        name for name in root.array_keys() if name.startswith(("list_", "thumb_"))
    ]
    assert (
        "run_vec_sum" not in root and "leaf_vec_sum" not in root
    ), "a distance method keeps distances, not vector sums"


def test_every_node_figure_is_the_sum_over_its_own_members(cached):
    """Leaf additivity: a node's cached time, count and occupancy equal a direct computation."""
    _run_dir, _state, method, fit, cache = cached
    tree = method.hierarchy_of(fit)
    places = method.member_places(fit)
    member_leaf = method.member_leaves(fit)
    t = cache.leaves
    for node, under in enumerate(tables_lib.node_leaf_index(tree)):
        members = np.isin(member_leaf, under)
        assert t.leaf_time_ns[under].sum() == places.time_ns[members].sum()
        assert t.leaf_members[under].sum() == members.sum()
        for index in range(len(places.recording_ids)):
            here = members & (places.recording_index == index)
            assert t.leaf_rec_time_ns[under, index].sum() == places.time_ns[here].sum()
    assert t.leaf_rec_pos.sum() > 0, "the fixture has kinematics; positions must land"
    assert t.leaf_rec_pos.sum() <= len(member_leaf)
    assert t.run_rows.sum() == len(member_leaf), "every member is in exactly one run"


def test_member_time_is_the_timestamp_span(cached_frames):
    """The whole tree's member time is the analysed span plus one step, not rows over an fps."""
    _run_dir, _state, method, fit, cache = cached_frames
    places = method.member_places(fit)
    t_ns = np.sort(places.start_ns)
    step = int(np.median(np.diff(t_ns)))
    assert cache.leaves.leaf_time_ns.sum() == int(t_ns[-1] - t_ns[0]) + step


def _view(cache, method, fit):
    from detail.api.node_view import NodeView

    return NodeView(cache, method.hierarchy_of(fit))


def _keys(clips) -> list[tuple[int, int, int]]:
    return sorted(
        zip(clips.recording.tolist(), clips.subject.tolist(), clips.start_ns.tolist())
    )


def test_every_clip_is_one_action_a_leaf_run_never_joined(cached_frames):
    """A clip is one action: one leaf run under the node (review round 2, choice B). So a leaf's
    clips are exactly the runs the method's ``instances_for`` makes of that leaf with everything
    else unlabelled (same continuity blocks, same minimum length); a node's clips are its
    children's, none twice; and the root lists every kept leaf run of the run. Joining touching
    runs under a node (as clips once did) makes the root a few recording-long clips and fails
    the count."""
    _run_dir, state, method, fit, cache = cached_frames
    view = _view(cache, method, fit)
    tree = view.tree
    t_ns = np.asarray(fit.t_ns, dtype=np.int64)
    counts = []
    for node in range(tree.n_nodes):
        clips = view.clips(node)
        keys = _keys(clips)
        assert len(set(keys)) == len(keys), f"node {node} lists one run twice"
        children = view.children(node)
        if children:
            assert keys == sorted(k for c in children for k in _keys(view.clips(c)))
        else:
            groups = np.full(tree.n_leaves, -1, dtype=np.int64)
            groups[node] = 0
            own = [
                i
                for i in method.instances_for(fit, groups, dict(state.method_params))
                if i.cluster_id == 0
            ]
            assert sorted(int(t_ns[i.start_row]) for i in own) == sorted(
                clips.start_ns.tolist()
            ), node
            assert sorted(int(t_ns[i.end_row]) for i in own) == sorted(
                clips.end_ns.tolist()
            ), node
            assert sorted(i.end_row - i.start_row + 1 for i in own) == sorted(
                clips.members.tolist()
            ), node
        counts.append(len(clips))
    kept = int((cache.leaves.run_rows >= int(cache.clips["min_members"])).sum())
    assert len(view.clips(view.root)) == kept
    assert len(view.clips(view.root)) > len(
        set(zip(cache.leaves.run_recording.tolist(), cache.leaves.run_subject.tolist()))
    ), "the fixture's root must hold more actions than units, or joining would pass"
    assert max(counts) > 1, "the fixture must hold nodes with several clips"


def test_a_clip_order_is_typical_atypical_or_a_shuffle_seeded_by_the_node(
    cached_frames,
):
    _run_dir, _state, method, fit, cache = cached_frames
    view = _view(cache, method, fit)
    node = view.root
    clips = view.clips(node)
    typical = clips.typicality[clips.order("typical")]
    assert list(typical) == sorted(typical)
    assert list(clips.typicality[clips.order("atypical")]) == sorted(
        typical, reverse=True
    )
    shuffle = clips.order("random", seed=node)
    assert sorted(shuffle) == list(range(len(clips)))
    assert list(shuffle) == list(view.clips(node).order("random", seed=node))
    with pytest.raises(ValueError, match="unknown sort"):
        clips.order("longest")


def _tiny_view(rec_time: np.ndarray):
    """A NodeView over three leaves ((0, 1) then 2) and the given leaf x recording times."""
    from detail.api.node_view import NodeView
    from detail.core.dataclasses import ClusterTree

    tree = ClusterTree(
        children=np.array([[0, 1], [3, 2]], dtype=np.int32),
        height=np.array([1.0, 2.0]),
        size=np.array([1, 1, 1, 2, 3], dtype=np.int32),
        n_rows=np.array([1, 1, 1, 2, 3], dtype=np.int32),
        leaf_of_row=np.array([0, 1, 2], dtype=np.int32),
        rung_merges=(0,),
    )
    leaves, recordings = rec_time.shape
    tables = tables_lib.LeafTables(
        recording_ids=tuple(f"recording_{i:03d}" for i in range(recordings)),
        leaf_time_ns=rec_time.sum(axis=1).astype(np.int64),
        leaf_members=(rec_time.sum(axis=1) // 1_000_000).astype(np.int64),
        leaf_rec_time_ns=rec_time.astype(np.int64),
        leaf_rec_pos=np.zeros((leaves, recordings, 16, 16), dtype=np.uint32),
        run_recording=np.array([0, 0, 1], dtype=np.int32),
        run_subject=np.zeros(3, dtype=np.int64),
        run_start_ns=np.array([0, 10, 0], dtype=np.int64),
        run_end_ns=np.array([10, 20, 5], dtype=np.int64),
        run_leaf=np.array([0, 1, 2], dtype=np.int32),
        separation=np.array([np.nan, np.nan, np.nan, 0.4, 0.2], dtype=np.float32),
    )
    return NodeView(store.NodeCache(leaves=tables, built_from={}), tree)


def test_a_node_confined_to_one_recording_flags_and_the_whole_run_does_not():
    """No concentration without its floor: the root is the run, so it never flags; a leaf whose
    time is all in one of ten equally used recordings beats any random draw of its size.
    """
    ms = 1_000_000
    rec_time = np.full((3, 10), 100 * ms)
    rec_time[0] = 0
    rec_time[0, 3] = 1_000 * ms
    # Leaf 2 follows the run's own distribution exactly, which is what "spread like the run"
    # means once the run itself leans towards recording 3.
    rec_time[2] = 2 * (rec_time[0] + rec_time[1])
    view = _tiny_view(rec_time)
    groups = {r: r for r in view.recording_ids}
    assert not view.distribution(view.root, groups).flagged
    confined = view.distribution(0, groups)
    assert confined.flagged and confined.top == "recording_003"
    assert confined.ratio > confined.floor
    spread = view.distribution(2, groups)
    assert not spread.flagged, "a leaf spread like the run must not flag"


def test_the_most_specific_label_wins_and_unlabelled_keeps_the_rest():
    ms = 1_000_000
    view = _tiny_view(np.array([[10 * ms, 0], [20 * ms, 0], [0, 30 * ms]]))
    times = view.label_time({"3": "moving", "0": "walking"})
    assert times == {"0": 10 * ms, "3": 20 * ms, "unlabelled": 30 * ms}
    assert sum(times.values()) == view.summary(view.root).time_ns


def test_the_labels_ethogram_merges_runs_that_land_on_one_label():
    view = _tiny_view(np.ones((3, 2), dtype=np.int64) * 1_000_000)
    assert view.labelled_runs({}) == []
    merged = view.labelled_runs({"3": "moving"})
    assert [(r["recording_id"], r["start_ns"], r["end_ns"]) for r in merged] == [
        ("recording_000", 0, 20)
    ]


def test_touching_runs_stay_two_clips():
    """Node 3 holds leaves 0 and 1, whose runs touch (10 → 10): still two clips, one per action.
    The root adds leaf 2, in another recording. A legacy table (no ``run_rows``) still lists
    clips, with no typicality."""
    view = _tiny_view(np.ones((3, 2), dtype=np.int64) * 1_000_000)
    clips = view.clips(3)
    assert list(zip(clips.start_ns, clips.end_ns)) == [(0, 10), (10, 20)]
    root = view.clips(view.root)
    assert list(zip(root.start_ns, root.recording)) == [(0, 0), (10, 0), (0, 1)]
    assert np.isnan(clips.typicality).all()


def test_children_times_add_up_to_their_parent(cached):
    run_dir, _state, method, fit, _cache = cached
    from detail.api.node_view import NodeView

    view = NodeView.open(run_dir, method.hierarchy_of(fit))
    assert view.summary(view.root).share == pytest.approx(1.0)
    for node in range(view.tree.n_leaves, view.tree.n_nodes):
        a, b = view.children(node)
        assert view.summary(node).time_ns == (
            view.summary(a).time_ns + view.summary(b).time_ns
        )


def test_a_rebuild_writes_the_same_cache_again(fitted_codebook):
    """No floor to move: a rebuild reproduces the cache the fit wrote, array for array."""
    proj, run_id = fitted_codebook
    handle = proj.temporal_clustering_run(run_id)
    run_dir = analysis_store.analysis_dir(proj.dir, run_id)
    before = store.read_node_cache(run_dir)
    handle.rebuild_node_cache()
    after = store.read_node_cache(run_dir)
    assert after.clips == before.clips
    for name in ("run_start_ns", "run_rows", "run_last_ns", "leaf_rec_pos"):
        assert np.array_equal(getattr(after.leaves, name), getattr(before.leaves, name))
    assert np.allclose(after.leaves.run_dist_sum, before.leaves.run_dist_sum, atol=1e-5)


# --- runs written with a size floor (legacy) -------------------------------------------


def test_a_run_saved_with_a_floor_still_opens_and_says_to_rebuild(cached, tmp_path):
    """A cache from before every clip was listed: member lists, thumbnails, a ``min_cluster_s``
    attr, and no per-run sums. It opens, its thumbnails are ignored, every node still lists
    clips (without a typicality order), and the view says why a rebuild would help."""
    import shutil

    from detail.api import node_cache as builder
    from detail.api.node_view import NodeView

    run_dir, _state, method, fit, _cache = cached
    copy = tmp_path / "run"
    shutil.copytree(store.cache_path(run_dir), store.cache_path(copy))
    root = zarr.open_group(store.cache_path(copy), mode="a")
    for name in ("run_rows", "run_last_ns", "run_vec_sum", "leaf_vec_sum"):
        del root[name]
    del root.attrs[store.CLIPS_ATTR]
    root.attrs["min_cluster_s"] = 0.5
    root.attrs["pictured"] = {"0": 24, "1": 12, "2": 12}
    root.attrs[store.BUILT_FROM_ATTR] = {
        **root.attrs[store.BUILT_FROM_ATTR],
        "min_cluster_s": 0.5,
    }
    root.create_array("list_node", data=np.zeros(3, np.int32), chunks=(65_536,))
    root.create_array("thumb_bytes", data=np.zeros(10, np.uint8), chunks=(65_536,))
    old = store.read_node_cache(copy)
    assert old.leaves.run_rows is None and old.clips == {}
    view = NodeView(old, method.hierarchy_of(fit))
    current = builder.fit_digest(temporal_runs.fit_path(run_dir, method.shard_suffix))
    assert "before every clip" in view.stale_reason(current)
    clips = view.clips(view.root)
    assert len(clips) and np.isnan(clips.typicality).all()


def test_legacy_floor_keys_are_dropped_on_read(fitted_temporal):
    """The one helper: a state with ``cache_floor_s`` reads and writes back without it, and a run
    started from params carrying ``min_cluster_s`` validates to a config without it."""
    from detail.configs.clustering_legacy import drop_legacy_floor
    from detail.executor.runners import ANALYSIS_MODULES

    proj, run_id = fitted_temporal
    run_dir = analysis_store.analysis_dir(proj.dir, run_id)
    path = run_dir / "temporal_clustering.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    data["cache_floor_s"] = 11.8
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    state = temporal_runs.read_temporal_state(run_dir)
    assert not hasattr(state, "cache_floor_s")
    temporal_runs.write_temporal_state(run_dir, state)
    assert "cache_floor_s" not in path.read_text(encoding="utf-8-sig")

    params = dict(analysis_store.read_analysis_yaml(run_dir)["params"])
    seeded = {
        **params,
        "source_recording_ids": list(state.source_recording_ids),
        "min_cluster_s": 30.0,
    }
    validated = ANALYSIS_MODULES["temporal_clustering"].validate(proj.dir, seeded)
    assert "min_cluster_s" not in validated
    assert drop_legacy_floor({"min_cluster_s": 1, "cache_floor_s": 2, "keep": 3}) == {
        "keep": 3
    }
    assert drop_legacy_floor(None) == {}


# --- positions for runs and clips; filters ------------------------------------------------


def test_a_run_is_where_its_members_were(cached_frames):
    """A run's position is the mean of its members', recomputed the long way from the same
    kinematics the build reads, so a stored position cannot drift from the members it
    describes."""
    from detail.api import node_cache as builder

    run_dir, state, method, fit, cache = cached_frames
    places = method.member_places(fit)
    leaves = method.member_leaves(fit)
    kinematics = builder._kinematics(run_dir.parent.parent, places, state.embed_run_for)
    cx, cy = tables_lib.member_positions(places, kinematics)
    t = cache.leaves
    assert np.isfinite(
        t.run_cx
    ).any(), "the fixture has kinematics; runs must get positions"
    order, heads, tails, _sorted = tables_lib._run_bounds(places, leaves)
    for run in range(min(len(heads), 50)):
        span = order[heads[run] : tails[run] + 1]
        known = np.isfinite(cx[span])
        if known.any():
            assert t.run_cx[run] == pytest.approx(cx[span][known].mean(), rel=1e-5)
            assert t.run_cy[run] == pytest.approx(cy[span][known].mean(), rel=1e-5)


def test_a_cache_from_before_positions_still_opens_without_them(cached, tmp_path):
    """Additive columns: an older cache reads them as NaN, says it has none, and its clips
    match no region."""
    import shutil

    from detail.api.node_view import NodeView

    run_dir, _state, method, fit, _cache = cached
    copy = tmp_path / "run"
    shutil.copytree(store.cache_path(run_dir), store.cache_path(copy))
    root = zarr.open_group(store.cache_path(copy), mode="a")
    for name in ("run_cx", "run_cy"):
        del root[name]
    old = store.read_node_cache(copy)
    assert np.isnan(old.leaves.run_cx).all()
    view = NodeView(old, method.hierarchy_of(fit))
    clips = view.clips(view.root)
    assert np.isnan(clips.cx).all()
    region = cluster_filter.Selection(cells=frozenset({(0, 0)}))
    assert not cluster_filter.clip_matches(region, clips).any()


def _metadata(view):
    """Two recordings would be best; the fixture has one, so give it a value and a number."""
    return {r: {"sex": "F", "weight": "21.5"} for r in view.recording_ids}


def test_match_shares_count_every_member_meeting_the_selection(cached):
    """Exact in members, against a brute-force count over the member rows the build placed:
    a region of the frame, a metadata value, and both at once (AND)."""
    from detail.api import node_cache as builder

    run_dir, state, method, fit, cache = cached
    view = _view(cache, method, fit)
    places = method.member_places(fit)
    leaves = method.member_leaves(fit)
    kinematics = builder._kinematics(run_dir.parent.parent, places, state.embed_run_for)
    cx, cy = tables_lib.member_positions(places, kinematics)
    known = np.isfinite(cx) & np.isfinite(cy)
    bins = tables_lib.POSITION_BINS
    bx = np.clip((np.nan_to_num(cx) * bins).astype(int), 0, bins - 1)
    by = np.clip((np.nan_to_num(cy) * bins).astype(int), 0, bins - 1)
    left_half = frozenset((x, y) for x in range(bins // 2) for y in range(bins))
    metadata = _metadata(view)
    cases = [
        (cluster_filter.Selection(cells=left_half), known & (bx < bins // 2)),
        (cluster_filter.Selection(values={"sex": frozenset({"F"})}), known),
        (cluster_filter.Selection(values={"sex": frozenset({"M"})}), known & False),
        (
            cluster_filter.Selection(ranges={"weight": (20, None)}, cells=left_half),
            known & (bx < bins // 2),
        ),
    ]
    for selection, member_match in cases:
        matched, total = view.match_counts(selection, metadata)
        shares = view.match_shares(selection, metadata)
        for node in range(view.tree.n_nodes):
            under = np.isin(leaves, view.leaves_under(node))
            assert total[node] == (under & known).sum()
            assert matched[node] == (under & member_match).sum()
            if total[node]:
                assert shares[node] == pytest.approx(matched[node] / total[node])
    assert 0 < view.match_shares(cases[0][0], metadata)[view.root] < 1


def test_a_clip_matches_by_its_recording_and_its_mean_position(cached):
    _run_dir, _state, method, fit, cache = cached
    view = _view(cache, method, fit)
    clips = view.clips(view.root)
    bins = tables_lib.POSITION_BINS
    left_half = frozenset((x, y) for x in range(bins // 2) for y in range(bins))
    region = cluster_filter.Selection(cells=left_half)
    expected = np.isfinite(clips.cx) & (np.nan_to_num(clips.cx) * bins < bins // 2)
    assert np.array_equal(cluster_filter.clip_matches(region, clips), expected)
    nobody = cluster_filter.Selection(recording_ids=frozenset())
    assert not cluster_filter.clip_matches(nobody, clips).any()
    assert cluster_filter.clip_matches(cluster_filter.Selection(), clips).all()


def test_a_metadata_condition_without_metadata_or_a_range_on_a_category_is_refused():
    selection = cluster_filter.Selection(values={"sex": frozenset({"F"})})
    with pytest.raises(ValueError, match="needs the experiment's metadata"):
        cluster_filter.recording_mask(selection, ["r1"])
    with pytest.raises(ValueError, match="is a category"):
        cluster_filter.recording_mask(
            cluster_filter.Selection(ranges={"sex": ("A", "B")}),
            ["r1"],
            {"r1": {"sex": "F"}},
        )
    with pytest.raises(ValueError, match="outside"):
        cluster_filter.cell_mask(cluster_filter.Selection(cells=frozenset({(16, 0)})))


def test_or_within_a_column_and_across_columns_and_dates_range():
    metadata = {
        "a": {"sex": "F", "day": "2026-07-02", "dose": "1"},
        "b": {"sex": "M", "day": "2026-07-08", "dose": "10"},
        "c": {"sex": "", "day": "", "dose": "5"},
    }
    ids = ["a", "b", "c"]
    mask = cluster_filter.recording_mask

    def sel(**kw):
        return cluster_filter.Selection(**kw)

    assert list(mask(sel(values={"sex": frozenset({"F", "M"})}), ids, metadata)) == [
        True,
        True,
        False,
    ]
    assert list(mask(sel(values={"sex": frozenset({""})}), ids, metadata)) == [
        False,
        False,
        True,
    ]
    both = sel(values={"sex": frozenset({"F", "M"})}, ranges={"dose": (5, None)})
    assert list(mask(both, ids, metadata)) == [False, True, False]
    days = sel(ranges={"day": ("2026-07-05", None)})
    assert list(mask(days, ids, metadata)) == [False, True, False]
    assert list(mask(sel(recording_ids=frozenset({"c"})), ids, metadata)) == [
        False,
        False,
        True,
    ]


def test_a_span_belongs_to_the_labelled_cluster_holding_most_of_it(
    fitted_temporal,
):
    """`cluster_of_spans` credits a span to the labelled node whose leaf runs overlap it most —
    the attribution AUDIT's precision by cluster rests on, since one fused instance can cross
    two clusters of a label."""
    experiment, run_id = fitted_temporal
    view = experiment.authored_run(run_id).node_view()
    t = view._cache.leaves
    tree = view.tree
    left, right = view.children(view.root)
    i = 0
    span = (
        view.recording_ids[int(t.run_recording[i])],
        int(t.run_subject[i]),
        int(t.run_start_ns[i]),
        int(t.run_end_ns[i]),
    )
    leaf = int(t.run_leaf[i])
    expected = left if leaf in set(tree.leaves_under(left).tolist()) else right
    assert view.cluster_of_spans([span], {left, right}) == [expected]
    assert view.cluster_of_spans([("nowhere", 0, 0, 1)], {left, right}) == [None]


def _cluster_of_spans_reference(view, spans, labelled):
    """One full scan of every run per span — kept as the definition the indexed version must
    agree with."""
    import numpy as np

    t = view._cache.leaves
    owner = view.tree.leaf_groups({int(n) for n in labelled})
    run_owner = owner[t.run_leaf.astype(np.int64)]
    index = {rid: i for i, rid in enumerate(view.recording_ids)}
    out = []
    for recording_id, subject, t0, t1 in spans:
        where = index.get(recording_id)
        if where is None:
            out.append(None)
            continue
        mask = (t.run_recording == where) & (t.run_subject == int(subject or 0))
        overlap = np.minimum(t.run_end_ns, int(t1)) - np.maximum(
            t.run_start_ns, int(t0)
        )
        # Runs are half-open: a run that only touches the span adds nothing, and an instant
        # belongs to the run that contains it.
        if (mask & (overlap > 0)).any():
            mask &= (overlap > 0) & (run_owner >= 0)
        else:
            mask &= (
                (t.run_start_ns <= int(t0))
                & (t.run_end_ns > int(t0))
                & (run_owner >= 0)
            )
        if not mask.any():
            out.append(None)
            continue
        weight = {}
        for node, amount in zip(run_owner[mask], overlap[mask]):
            weight[int(node)] = weight.get(int(node), 0) + int(amount) + 1
        out.append(max(weight, key=weight.get))
    return out


def test_the_indexed_attribution_agrees_with_a_full_scan(fitted_temporal):
    """The audit attributes every instance in a label, so the lookup reads only the runs that can
    overlap a span. Same answers as scanning them all, over spans of every shape: inside one run,
    across several, on an edge, an instant, and outside the data."""
    import numpy as np

    experiment, run_id = fitted_temporal
    view = experiment.authored_run(run_id).node_view()
    t = view._cache.leaves
    rng = np.random.default_rng(3)
    spans = []
    for i in rng.choice(len(t.run_leaf), size=min(60, len(t.run_leaf)), replace=False):
        rec = view.recording_ids[int(t.run_recording[i])]
        subj = int(t.run_subject[i])
        start, end = int(t.run_start_ns[i]), int(t.run_end_ns[i])
        spans += [
            (rec, subj, start, end),
            (rec, subj, start, end + 5_000_000_000),
            (rec, subj, end, end),
            (rec, subj, start - 10**12, start - 10**12 + 1),
        ]
    spans.append(("nowhere", 0, 0, 1))
    left, right = view.children(view.root)
    for labelled in ({left, right}, {left}, set(view.children(left)) | {right}):
        assert view.cluster_of_spans(spans, labelled) == _cluster_of_spans_reference(
            view, spans, labelled
        )


def test_clips_built_from_another_fit_are_reported_stale(cached):
    """§22: a stale cache is reported, never silently read. Name and size, not time: a copied
    experiment keeps its clips."""
    from detail.api import node_cache as builder
    from detail.api.node_view import NodeView

    run_dir, _state, method, fit, cache = cached
    view = NodeView(cache, method.hierarchy_of(fit))
    current = builder.fit_digest(temporal_runs.fit_path(run_dir, method.shard_suffix))
    assert view.stale_reason(current) is None
    assert (
        view.stale_reason({**current, "fit_mtime_ns": 1}) is None
    ), "time is not compared"
    reason = view.stale_reason({**current, "fit_size": current["fit_size"] + 1})
    assert reason and "fit size" in reason


def test_an_instant_on_a_boundary_belongs_to_the_run_that_contains_it(
    fitted_temporal,
):
    """A one-frame bout at the end of one run and the start of the next went to the earlier run:
    both overlapped it by zero, and the tie went first."""
    import numpy as np

    experiment, run_id = fitted_temporal
    view = experiment.authored_run(run_id).node_view()
    units, starts, ends, leaves = view._runs_by_unit()
    left, right = view.children(view.root)
    owner = view.tree.leaf_groups({left, right})[leaves]
    (recording, subject), (lo, hi) = next(
        (key, bounds)
        for key, bounds in units.items()
        if any(
            owner[i] != owner[i + 1] and ends[i] == starts[i + 1]
            for i in range(bounds[0], bounds[1] - 1)
        )
    )
    i = next(
        i
        for i in range(lo, hi - 1)
        if owner[i] != owner[i + 1] and ends[i] == starts[i + 1]
    )
    edge = int(starts[i + 1])
    rid = view.recording_ids[recording]
    got = view.cluster_of_spans([(rid, subject, edge, edge)], {left, right})
    assert got == [int(owner[i + 1])], "the instant went to the run it only touches"
    assert np.all(ends[lo:hi] >= starts[lo:hi])


# --- review fixes ----------------------------------------------------------------------


def test_the_vector_basis_is_the_svd_s_up_to_sign():
    """The Gram-matrix basis spans what the SVD's does, vector by vector up to sign, so every
    projected score is unchanged; it never builds the sample-sized left factor."""
    rng = np.random.default_rng(4)
    sample = rng.normal(size=(500, 20)) * np.linspace(5, 1, 20)
    _u, _s, vt = np.linalg.svd(sample, full_matrices=False)
    basis = tables_lib.vector_basis(sample, dims=10)
    assert basis.shape == (10, 20)
    assert np.allclose(np.abs(basis @ vt[:10].T), np.eye(10), atol=1e-6)
    x, c = rng.normal(size=(3, 20)), rng.normal(size=20)

    def cos(b):
        p, q = x @ b.T, b @ c
        return (p @ q) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q))

    assert np.allclose(cos(basis), cos(vt[:10]))


def test_a_clip_s_position_is_its_own_run_s_and_short_runs_are_dropped():
    """A clip is one run, so its position is that run's mean, never averaged with a touching
    run's; a run below the method's minimum length is no clip at any node."""
    import dataclasses

    view = _tiny_view(np.ones((3, 2), dtype=np.int64) * 1_000_000)
    t = view._cache.leaves
    leaves = dataclasses.replace(
        t,
        run_cx=np.array([0.0, 1.0, 0.5], np.float32),
        run_cy=np.array([0.0, 1.0, 0.5], np.float32),
        run_rows=np.array([10, 10, 5], np.int32),
    )
    from detail.api.node_view import NodeView

    view = NodeView(
        store.NodeCache(
            leaves=leaves, built_from={}, clips={"kind": "none", "min_members": 6}
        ),
        view.tree,
    )
    assert view.clips(3).cx.tolist() == [0.0, 1.0]
    assert view.clips(3).members.tolist() == [10, 10]
    assert len(view.clips(2)) == 0 and len(view.clips(view.root)) == 2


def test_match_counts_never_copies_the_position_table_whole(cached):
    """A region over a minority of cells copies only those cells, and nothing is cast to float:
    the old path cast all of ``leaf_rec_pos`` to float64 and fancy-indexed a copy (~740 MB at
    5,000 leaves x 72 recordings)."""
    import dataclasses
    import tracemalloc

    from detail.api.node_view import NodeView

    _run_dir, _state, method, fit, cache = cached
    leaves = cache.leaves.leaf_rec_pos.shape[0]
    big = np.ones((leaves, 40, 16, 16), dtype=np.uint32)
    tables = dataclasses.replace(
        cache.leaves,
        recording_ids=tuple(f"recording_{i:03d}" for i in range(40)),
        leaf_rec_pos=big,
        leaf_rec_time_ns=np.ones((leaves, 40), np.int64),
    )
    view = NodeView(
        store.NodeCache(leaves=tables, built_from={}), method.hierarchy_of(fit)
    )
    corner = cluster_filter.Selection(
        cells=frozenset((x, y) for x in range(4) for y in range(4))
    )
    view.match_counts(corner)  # the per-recording totals, kept on the view
    tracemalloc.start()
    matched, total = view.match_counts(corner)
    _now, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < big.nbytes / 4, f"peak {peak:,} B against a {big.nbytes:,} B table"
    assert matched[view.root] == leaves * 40 * 16
    assert total[view.root] == leaves * 40 * 256
    everything = cluster_filter.Selection(
        cells=frozenset((x, y) for x in range(16) for y in range(15))
    )
    assert view.match_counts(everything)[0][view.root] == leaves * 40 * 240


def test_a_run_without_a_cache_names_the_rebuild(fitted_temporal):
    """The clips come from the node cache alone: a run fitted before it existed has none, and
    the error says how to build it rather than only that a file is missing."""
    import shutil

    proj, run_id = fitted_temporal
    handle = proj.authored_run(run_id)
    shutil.rmtree(store.cache_path(handle._run_dir()))
    with pytest.raises(
        FileNotFoundError, match="temporal-interpret --run .* --rebuild-cache"
    ):
        handle.typical_members(0)


# --- a cache built under an older rule -----------------------------------------------------


def _rewrite_under_rule_one(run_dir) -> None:
    """Rewrite ``run_dir``'s cache as rule 1 wrote it: no ``rules`` in ``built_from``, and each
    run ending at its last member's end plus its time (``run_last_ns + covered time``).
    """
    import dataclasses

    cache = store.read_node_cache(run_dir)
    t = cache.leaves
    old_end = t.run_last_ns + (t.run_end_ns - t.run_start_ns)
    built = {k: v for k, v in cache.built_from.items() if k != store.RULES_KEY}
    store.write_node_cache(
        run_dir,
        dataclasses.replace(
            cache,
            leaves=dataclasses.replace(t, run_end_ns=old_end),
            built_from=built,
        ),
    )


def test_a_bout_cache_from_the_old_run_end_rule_is_stale_and_a_rebuild_fixes_it(
    fitted_temporal,
):
    """A ``dtw_finch`` cache written before rule 2 opens, reports why it is out of date (the
    banner and **Rebuild clips** read ``view.stale``), and a rebuild through the run handle,
    the path the GUI and ``--rebuild-cache`` take, writes runs that end one step past their last
    frame, with the current rules recorded."""
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    run_dir = run._run_dir()
    fresh = store.read_node_cache(run_dir)
    assert fresh.built_from[store.RULES_KEY] == store.RULES
    assert run.node_view().stale is None

    _rewrite_under_rule_one(run_dir)
    run = experiment.temporal_clustering_run(run_id)
    stale = run.node_view().stale
    assert stale and "bout" in stale

    steps = []
    run.rebuild_node_cache(
        progress=lambda stage, fraction, detail: steps.append(detail)
    )
    assert steps and steps[-1] == "done"
    rebuilt = store.read_node_cache(run_dir)
    assert rebuilt.built_from[store.RULES_KEY] == store.RULES
    assert rebuilt.leaves.run_end_ns.tolist() == fresh.leaves.run_end_ns.tolist()
    assert run.node_view().stale is None


def test_a_frame_cache_from_the_old_rule_is_not_stale(fitted_codebook):
    """A frame starts where it ends, so rule 1 and rule 2 give a frame method the same runs:
    its old cache is not flagged, which would only teach users to ignore the banner."""
    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    run_dir = run._run_dir()
    import dataclasses

    before = store.read_node_cache(run_dir).leaves.run_end_ns.tolist()
    cache = store.read_node_cache(run_dir)
    store.write_node_cache(
        run_dir,
        dataclasses.replace(
            cache,
            built_from={
                k: v for k, v in cache.built_from.items() if k != store.RULES_KEY
            },
        ),
    )
    run = experiment.temporal_clustering_run(run_id)
    assert run.node_view().stale is None
    run.rebuild_node_cache()
    assert store.read_node_cache(run_dir).leaves.run_end_ns.tolist() == before
