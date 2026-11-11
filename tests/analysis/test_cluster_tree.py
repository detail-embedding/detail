"""A cut of the merge tree nests; the refinement it replaces did not.

**The class.** A hierarchy that answers "give me k clusters" by re-deriving an answer each time
has no order, and two answers derived independently need not agree with each other. FINCH's own
partitions nest by construction, so the defect hid everywhere except between them: refinement
started from whichever rung was tightest and merged down, which means the partition at 30 and the
partition at 12 could disagree about which cluster a row belongs to even though 12 is a rung.
Measured on a real fit and recorded at ``api/pose_clustering.py`` (since removed) — *"50
refines 12, and 30 does not refine 12"*. Nothing fails when that happens: every partition is
individually well formed, the counts are right, and a name carried from one setting to another
lands on a cluster that is half something else.

The tree removes the possibility rather than the symptom. Merges are recorded once, constrained to
stay inside one group of the next-coarser partition, so a cut is a **prefix** of one sequence —
and a prefix of a sequence refines every longer prefix, for every pair, by construction.

``test_the_refinement_it_replaces_does_not_nest`` is the negative case: it runs this file's own
nesting assertion against ``_refine_to`` on the same fixtures and requires it to fail. Without it
this suite would pass in two different worlds — the property holds, or the fixtures never
exercised it.
"""

import numpy as np
import pytest

from detail.analysis import segmentation as seg


def _blobs(seed: int, n: int = 400, dim: int = 8, groups: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(groups, dim))
    rows = centres[rng.integers(0, groups, n)] + rng.normal(scale=0.15, size=(n, dim))
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _dense_of(rows: np.ndarray) -> np.ndarray:
    dist = 1.0 - rows @ rows.T
    np.fill_diagonal(dist, 0.0)
    return dist


def _refines(fine: np.ndarray, coarse: np.ndarray) -> bool:
    """Whether every cluster of ``fine`` sits inside exactly one cluster of ``coarse``."""
    return all(len(np.unique(coarse[fine == c])) == 1 for c in np.unique(fine))


def _counts(levels: list[np.ndarray]) -> list[int]:
    return [int(level.max()) + 1 for level in levels]


def _probe_counts(levels: list[np.ndarray]) -> list[int]:
    """Every rung, plus the off-rung settings between them — where the defect lives."""
    counts = _counts(levels)
    return sorted(set(list(range(1, min(counts[0], 40) + 1)) + counts))


SEEDS = (0, 1, 2, 7)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_rung_is_a_cut_of_the_tree(seed):
    """The constraint the tree is built under, stated as its observable consequence.

    If a rung were not reachable by cutting, the tree would be a *different* hierarchy wearing the
    method's rung labels — every granularity the user has already saved would move.
    """
    rows = _blobs(seed)
    levels = seg.finch_hierarchy_from_vectors(rows)
    tree = seg.cluster_tree_from_vectors(rows, levels)
    assert tree.rungs == tuple(
        _counts(levels)
    ), "the tree lost one of the method's partitions"
    for level, count in zip(levels, _counts(levels)):
        cut = tree.cut_at(count)[tree.leaf_of_row]
        assert _refines(cut, level) and _refines(
            level, cut
        ), f"cut at {count} is not the hierarchy's own partition"


@pytest.mark.parametrize("seed", SEEDS)
def test_every_cut_refines_every_coarser_cut(seed):
    """The property a name on a node depends on, over every pair — not only adjacent ones."""
    rows = _blobs(seed)
    levels = seg.finch_hierarchy_from_vectors(rows)
    tree = seg.cluster_tree_from_vectors(rows, levels)
    cuts = {k: tree.cut_at(k) for k in _probe_counts(levels)}
    for fine, fine_labels in cuts.items():
        for coarse, coarse_labels in cuts.items():
            if fine > coarse:
                assert _refines(
                    fine_labels, coarse_labels
                ), f"cut at {fine} does not refine cut at {coarse}"


def test_the_refinement_it_replaces_does_not_nest():
    """Guard the guard: the assertion above must be capable of failing.

    Run against ``_refine_to`` — the path ``labels_at`` used between rungs — on the same fixtures.
    If this ever stops finding violations, either the fixtures stopped reaching the off-rung path
    or ``_refine_to`` changed, and in both cases the test above has stopped proving anything.
    """
    violations = 0
    for seed in SEEDS:
        rows = _blobs(seed)
        levels = seg.finch_hierarchy_from_vectors(rows)
        dense = _dense_of(rows)
        counts = _counts(levels)
        old = {
            k: (
                levels[counts.index(k)]
                if k in counts
                else seg._refine_to(dense, levels, k)
            )
            for k in _probe_counts(levels)
        }
        violations += sum(
            not _refines(old[fine], old[coarse])
            for fine in old
            for coarse in old
            if fine > coarse
        )
    assert violations, (
        "the fixtures no longer reach the defect the tree exists to remove, so the nesting "
        "tests above are passing vacuously"
    )


@pytest.mark.parametrize("seed", (0, 3, 5))
def test_the_vector_and_dense_builders_produce_one_tree(seed):
    """The two paths exist for memory, not for different answers.

    ``finch_hierarchy`` and ``finch_hierarchy_from_vectors`` are pinned as identical; the trees
    built over them have to be too, or a pose fit and a bout fit would disagree about what the
    same hierarchy means.
    """
    rows = _blobs(seed, n=300, groups=10)
    dense = _dense_of(rows)
    vector_tree = seg.cluster_tree_from_vectors(
        rows, seg.finch_hierarchy_from_vectors(rows)
    )
    dense_tree = seg.cluster_tree_from_distances(dense, seg.finch_hierarchy(dense))
    assert np.array_equal(vector_tree.children, dense_tree.children)
    assert np.allclose(vector_tree.height, dense_tree.height)
    assert vector_tree.rung_merges == dense_tree.rung_merges


def test_an_unconstrained_run_is_the_refinement_it_replaces():
    """The merge **rule**, pinned against the reference implementation that already exists.

    Structure is not correctness. A tree built by merging the *farthest* pair at every step still
    lands on every rung and still nests — it passes every other test in this file — and every
    off-rung cut it produces is nonsense. So the rule needs its own assertion, and ``_refine_to``
    is the right thing to assert it against: it is greedy average linkage over the same sums, and
    ``tests/analysis/test_refine_to.py`` already pins *it* against a naive rebuild-every-step
    reference.

    A **single-level** hierarchy is where the two must agree exactly. With one level there is no
    rung to stay inside, so the tail is an unconstrained greedy run and ``_refine_to`` starts from
    that same level — same rule, same input, so the same partition at every count.
    """
    rows = _blobs(4, n=180, groups=9)
    dense = _dense_of(rows)
    flat = [seg._finch_partition(dense)]
    tree = seg.cluster_tree_from_vectors(rows, flat)
    for k in range(1, tree.n_leaves + 1):
        mine = tree.cut_at(k)[tree.leaf_of_row]
        theirs = seg._refine_to(dense, flat, k)
        assert _refines(mine, theirs) and _refines(
            theirs, mine
        ), f"the merge order disagrees with greedy average linkage at {k} clusters"


@pytest.mark.parametrize("seed", SEEDS)
def test_one_greedy_run_never_merges_downhill(seed):
    """Average linkage is reducible, so a single greedy run's heights only climb.

    Asserted on a **single-level** hierarchy, which is the only place the whole sequence is one
    run. Inside a real transition the heights sawtooth, and correctly so: each of the coarser
    level's groups is merged on its own, so a boundary between two groups is a jump back down.
    A renderer has to know that; a test must not paper over it with a "mostly increasing" check,
    which is a proxy and was false the first time I wrote it.
    """
    rows = _blobs(seed, n=180, groups=9)
    dense = _dense_of(rows)
    tree = seg.cluster_tree_from_vectors(rows, [seg._finch_partition(dense)])
    assert len(tree.height) > 2, "the fixture must produce a run worth checking"
    assert np.all(
        np.diff(tree.height) >= 0
    ), "a greedy average-linkage run cannot merge a farther pair before a nearer one"


def test_a_tight_pair_merges_before_a_far_one():
    """The rule in one readable case, with the answer visible in the fixture."""
    tight = np.array([[1.0, 0.0, 0.0], [0.999, 0.045, 0.0]])
    far = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rows = np.vstack([tight, far])
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    tree = seg.cluster_tree_from_vectors(rows, [np.arange(4)])
    first_merge = set(int(node) for node in tree.children[0])
    assert first_merge == {
        0,
        1,
    }, f"the closest pair must merge first, not {sorted(first_merge)}"
    assert tree.height[0] < tree.height[-1]


def test_a_hierarchy_with_one_level_still_reaches_a_root():
    """The unconstrained tail — and the shape a method with no nested partitions would take.

    ``finch_hierarchy`` stops when a step merges nothing, which can leave more than one cluster.
    A tree that stopped there would have no root, so nothing above the last rung could be named.
    """
    rows = _blobs(0, n=60, groups=6)
    flat = [np.repeat(np.arange(6), 10)]
    tree = seg.cluster_tree_from_vectors(rows, flat)
    assert tree.rungs == (6,), "a single level is the only rung"
    assert len(tree.height) == 5, "six leaves need five merges to reach one root"
    assert len(np.unique(tree.cut_at(1))) == 1


def test_an_empty_fit_has_no_tree_and_a_single_row_has_no_merges():
    """The schema edges: zero rows and one row, which a blob fixture never produces."""
    empty = seg.cluster_tree_from_vectors(np.zeros((0, 5)), [])
    assert empty.n_leaves == 0 and len(empty.height) == 0

    one = seg.cluster_tree_from_vectors(np.ones((1, 5)), [np.zeros(1, dtype=np.int64)])
    assert one.n_leaves == 1 and len(one.height) == 0
    assert one.cut_at(1).tolist() == [0]


def test_a_cut_outside_the_range_clamps_rather_than_raises():
    """Asking for more clusters than the fit found is a real request from a slider."""
    rows = _blobs(1)
    levels = seg.finch_hierarchy_from_vectors(rows)
    tree = seg.cluster_tree_from_vectors(rows, levels)
    assert len(np.unique(tree.cut_at(0))) == 1
    assert len(np.unique(tree.cut_at(tree.n_leaves + 500))) == tree.n_leaves


def test_leaves_under_a_node_are_the_leaves_a_name_would_reach():
    """What a node *is*, for the naming model above this: every leaf beneath it."""
    rows = _blobs(2)
    levels = seg.finch_hierarchy_from_vectors(rows)
    tree = seg.cluster_tree_from_vectors(rows, levels)
    root = tree.n_nodes - 1
    assert tree.leaves_under(root).tolist() == list(range(tree.n_leaves))
    assert tree.leaves_under(0).tolist() == [0]

    nodes = tree.nodes_at(4)
    assert len(nodes) == 4
    covered = np.concatenate([tree.leaves_under(node) for node in nodes])
    assert sorted(covered.tolist()) == list(
        range(tree.n_leaves)
    ), "the nodes at a cut must partition the leaves"


def test_sizes_and_row_counts_add_up_the_tree():
    """What a collapsed node draws itself with, so a renderer never has to load its members."""
    rows = _blobs(7)
    levels = seg.finch_hierarchy_from_vectors(rows)
    tree = seg.cluster_tree_from_vectors(rows, levels)
    root = tree.n_nodes - 1
    assert int(tree.size[root]) == tree.n_leaves
    assert int(tree.n_rows[root]) == len(rows)
    for merge, (left, right) in enumerate(tree.children):
        node = tree.n_leaves + merge
        assert tree.size[node] == tree.size[left] + tree.size[right]
        assert tree.n_rows[node] == tree.n_rows[left] + tree.n_rows[right]


def test_parent_map_reads_the_coarser_cluster_off_any_member():
    fine = np.array([0, 0, 1, 1, 2, 2, 3])
    coarse = np.array([0, 0, 0, 0, 1, 1, 1])
    assert seg.parent_map(fine, coarse).tolist() == [0, 0, 1, 1]
    assert seg.parent_map(np.zeros(0, dtype=np.int64), np.zeros(0)).tolist() == []


def test_a_tree_round_trips_through_its_arrays():
    rows = _blobs(0)
    levels = seg.finch_hierarchy_from_vectors(rows)
    tree = seg.cluster_tree_from_vectors(rows, levels)
    restored = seg.tree_from_arrays(seg.tree_arrays(tree))
    assert np.array_equal(restored.children, tree.children)
    assert np.allclose(restored.height, tree.height)
    assert np.array_equal(restored.leaf_of_row, tree.leaf_of_row)
    assert restored.rung_merges == tree.rung_merges
    assert np.array_equal(restored.cut_at(5), tree.cut_at(5))


def test_a_shard_with_no_tree_reads_back_as_none_rather_than_raising():
    """Fits written before the tree existed are on disk; the caller rebuilds from ``None``."""
    assert seg.tree_from_arrays({"sample": np.zeros((2, 2))}) is None
