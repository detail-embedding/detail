"""A label is a stored thing clusters point at, and its cached resolution cannot drift (§22).

**The class.** ``resolved_names`` / ``resolved_paths`` are *derived* values written to disk: each
label and its path, precomputed so a snapshot record, the artifact manifest and the state module
can answer "what is this called" without a fit in hand. Derived state with more than one writer is
how two files come to disagree, and this one disagrees *plausibly*: the run still opens and the
labels are the ones from before the last edit.

So every label edit goes through one door, ``LabelOperations._label_edit``, which folds an older
state, applies the change, refuses it if ``check_labels`` would, and recomputes the resolution. Two
guards: the behavioural ones drive each kind of edit and recompute from the authority (``labels`` +
``node_labels`` and the fit's tree); the source scan holds that every label edit uses the door and
that neither module grows its own copy — a fix applied to one copy and not the other is the normal
failure, and a behavioural test against a pose fixture cannot see the temporal one.
"""

import ast
import pathlib

import pytest

from detail.api import authored_run as authored

MODULES = ("detail/api/temporal_clustering.py",)

#: Every edit that can change what a member is called.
MUTATING = (
    "name_nodes",
    "create_label",
    "label_nodes",
    "rename_label",
    "merge_labels",
    "delete_label",
    "restore_labels",
    "edit_labels",
)


@pytest.fixture(autouse=True)
def _silence_projection(monkeypatch):
    """The §19 rewrite is not what is under test, and the cohort has no embedding run behind it.

    Patched on the **concrete** class, which is the one that overrides ``_project``.
    """
    from detail.api.temporal_clustering import TemporalClusteringRun

    monkeypatch.setattr(TemporalClusteringRun, "_project", lambda self, **_: None)


def _expected(run):
    """The resolution recomputed from the authority, the long way round."""
    state = run.state()
    return authored.label_resolution(run._tree(), state.labels, state.node_labels)


def _a_branch(run) -> int:
    """One of the root's children: a real node with members under it, never a lone leaf."""
    tree = run._tree()
    return int(tree.children[tree.n_nodes - 1 - tree.n_leaves][0])


def _siblings(run) -> tuple[int, int]:
    tree = run._tree()
    left, right = (int(node) for node in tree.children[-1])
    return left, right


def _inner(tree, node: int) -> int:
    """A child of ``node`` — assumes ``node`` is internal, which the root's children are here."""
    return int(tree.children[node - tree.n_leaves][0])


def test_labelling_a_node_leaves_the_cache_agreeing_with_the_authority(authored_cohort):
    run = authored_cohort.run()
    node = _a_branch(run)
    run.name_nodes({node: "rearing"})
    state = run.state()
    names, paths = _expected(run)
    assert names == {"L1": "rearing"}, "the fixture must resolve to something"
    assert state.resolved_names == names and state.resolved_paths == paths
    assert state.labels == {"L1": "rearing"}
    assert state.node_labels == {str(node): "L1"}
    assert state.node_names == {str(node): "rearing"}


def test_removing_a_label_leaves_the_cache_agreeing(authored_cohort):
    """The direction a cache is most likely to keep a stale entry: something was removed."""
    run = authored_cohort.run()
    node = _a_branch(run)
    run.name_nodes({node: "rearing"})
    run.name_nodes({node: ""})
    state = run.state()
    assert state.node_labels == {} and state.labels == {}
    assert state.resolved_names == {} == _expected(run)[0]


def test_a_label_on_the_root_covers_every_member(authored_cohort):
    """One label high up is one behavior: every leaf resolves to the labelled root."""
    run = authored_cohort.run()
    tree = run._tree()
    root = tree.n_nodes - 1
    run.name_nodes({root: "everything"})
    groups = tree.leaf_groups(run.state().named_nodes())
    assert set(groups.tolist()) == {root}


def test_two_sibling_labels_split_their_parent(authored_cohort):
    """Labels on two siblings each keep their own members; nothing takes the other's name."""
    run = authored_cohort.run()
    tree = run._tree()
    left, right = _siblings(run)
    run.name_nodes({left: "a", right: "b"})
    groups = authored.label_groups(tree, run.state())
    assert set(groups[tree.leaves_under(left)].tolist()) == {0}
    assert set(groups[tree.leaves_under(right)].tolist()) == {1}
    assert run.state().resolved_names == {"L1": "a", "L2": "b"}


def test_a_name_that_exists_adds_the_node_to_that_label(authored_cohort):
    """Two clusters given one name are **one label**, never two matching strings: the second
    points at the first's id, and the output groups their members together."""
    run = authored_cohort.run()
    tree = run._tree()
    left, right = _siblings(run)
    run.name_nodes({left: "rearing"})
    run.name_nodes({right: "Rearing "})  # case and spacing are the same name
    state = run.state()
    assert state.labels == {"L1": "rearing"}
    assert state.node_labels == {str(left): "L1", str(right): "L1"}
    groups = authored.label_groups(tree, state)
    assert set(groups[tree.leaves_under(left)].tolist()) == {0}
    assert set(groups[tree.leaves_under(right)].tolist()) == {0}


def test_a_label_whose_clusters_would_sit_under_two_parents_is_refused(authored_cohort):
    """Nesting comes from the tree, so a label's path is the labels above its clusters. Clusters
    under two different parents would give one label two paths, and the edit is refused before it
    is written — never quietly repaired."""
    run = authored_cohort.run()
    tree = run._tree()
    left, right = _siblings(run)
    run.name_nodes({left: "climb", _inner(tree, left): "corner"})
    before = run.state().node_labels
    with pytest.raises(ValueError, match="share one parent label"):
        run.name_nodes({right: "corner"})
    assert run.state().node_labels == before, "the refused edit reached the state"


def test_labelling_a_branch_above_some_of_a_labels_clusters_is_refused(authored_cohort):
    """The same rule reached from the other side: a label given to a branch holding only some
    of another label's clusters would split that label's parents."""
    run = authored_cohort.run()
    tree = run._tree()
    left, right = _siblings(run)
    inner = _inner(tree, left)
    run.name_nodes({inner: "corner", right: "corner"})
    with pytest.raises(ValueError, match="share one parent label"):
        run.name_nodes({left: "climb"})
    assert run.state().labels == {"L1": "corner"}


def test_a_label_inside_its_own_branch_is_refused(authored_cohort):
    run = authored_cohort.run()
    tree = run._tree()
    left, _right = _siblings(run)
    run.name_nodes({left: "rearing"})
    label = run.state().node_labels[str(left)]
    with pytest.raises(ValueError, match="already inside"):
        run.label_nodes({_inner(tree, left): label})


def test_rename_keeps_the_id_and_reaches_every_cluster(authored_cohort):
    run = authored_cohort.run()
    left, right = _siblings(run)
    run.name_nodes({left: "climb", right: "climb"})
    run.rename_label("L1", "climbing")
    state = run.state()
    assert state.labels == {"L1": "climbing"}
    assert state.node_names == {str(left): "climbing", str(right): "climbing"}


def test_renaming_a_sole_cluster_by_name_keeps_its_label_id(authored_cohort):
    """``name_nodes`` on a label's only cluster with a new name is a rename, not a new label —
    so the id, and every instance id carrying it, stays put."""
    run = authored_cohort.run()
    node = _a_branch(run)
    run.name_nodes({node: "rearing"})
    run.name_nodes({node: "rearing up"})
    assert run.state().labels == {"L1": "rearing up"}


def test_merge_moves_every_cluster_and_removes_the_source(authored_cohort):
    run = authored_cohort.run()
    left, right = _siblings(run)
    run.name_nodes({left: "climb", right: "scale"})
    run.merge_labels("L2", "L1")
    state = run.state()
    assert state.labels == {"L1": "climb"}
    assert state.node_labels == {str(left): "L1", str(right): "L1"}


@pytest.mark.parametrize("edit", ["name", "rename", "merge", "remove"])
def test_restoring_either_side_of_an_edit_puts_the_maps_back_by_id(
    authored_cohort, edit
):
    """INTERPRET's ⌘Z: ``label_change`` read either side of an edit, then ``restore_labels``
    writes the *before* side (undo) and the *after* side (redo). By id: undoing a merge brings
    ``L2`` back as ``L2``, which writing its name back would not."""
    run = authored_cohort.run()
    left, right = _siblings(run)
    run.name_nodes({left: "climb", right: "scale"})
    before = run.label_maps()
    {
        "name": lambda: run.name_nodes({left: "rear"}),
        "rename": lambda: run.rename_label("L1", "climbing"),
        "merge": lambda: run.merge_labels("L2", "L1"),
        "remove": lambda: run.name_nodes({right: ""}),
    }[edit]()
    after = run.label_maps()
    change = authored.label_change(before, after)
    assert any(change), "the edit changed something"
    run.restore_labels(*_side(change, 0))
    assert run.label_maps() == before
    run.restore_labels(*_side(change, 1))
    assert run.label_maps() == after


def _side(change, index):
    return tuple({k: v[index] for k, v in part.items()} for part in change)


def test_a_second_label_with_an_existing_name_is_refused(authored_cohort):
    run = authored_cohort.run()
    run.name_nodes({_a_branch(run): "rearing"})
    with pytest.raises(ValueError, match="already a label called"):
        run.create_label("REARING")
    with pytest.raises(ValueError, match="already a label called"):
        run.rename_label(run.create_label("grooming"), "rearing")


def test_the_legacy_map_is_folded_onto_nodes_by_the_next_edit(authored_cohort):
    """A run written under the per-granularity model keeps its names, now as labels on nodes."""
    cohort = authored_cohort
    state = cohort.read_state()
    state.resolved_names = {}
    state.labels, state.node_labels = {}, {}
    state.names_by_granularity = {str(state.granularity): {"0": "rearing"}}
    cohort.write_state(state)

    run = cohort.run()
    assert run.state().name_of(0) == "rearing", "read back before anything is folded"

    run.name_nodes({})  # the next edit, whatever it is
    after = run.state()
    assert after.names_by_granularity == {}, "the legacy map is emptied once folded"
    assert list(after.node_names.values()) == [
        "rearing"
    ], "the name now lives on a node"
    assert after.resolved_names == _expected(run)[0]


def test_per_node_names_from_before_labels_had_ids_read_as_one_label_per_name():
    """Fit-free, so a snapshot of an older run still opens: each distinct name is one label."""
    from detail.storage import temporal_runs

    state = temporal_runs.TemporalClusteringState.from_dict(
        {
            "method": "dtw_finch",
            "track_id": 0,
            "embed_run_id": "emb_01",
            "node_names": {"9": "rearing", "12": "grooming", "14": "rearing"},
        }
    )
    assert state.labels == {"L1": "rearing", "L2": "grooming"}
    assert state.node_labels == {"9": "L1", "12": "L2", "14": "L1"}


def test_a_legacy_name_under_two_parents_is_split_by_the_next_edit(authored_cohort):
    """The fold, which has the tree, repairs what the fit-free read could not see."""
    cohort = authored_cohort
    run = cohort.run()
    tree = run._tree()
    left, right = _siblings(run)
    inner = _inner(tree, left)
    state = cohort.read_state()
    state.labels = {"L1": "climb", "L2": "corner"}
    state.node_labels = {str(left): "L1", str(inner): "L2", str(right): "L2"}
    cohort.write_state(state)

    run.name_nodes({})
    after = run.state()
    assert sorted(after.labels.values()) == ["climb", "corner", "corner (2)"]
    assert after.node_names[str(inner)] != after.node_names[str(right)]


def _legacy(cohort):
    """Rewrite the cohort's state the way a run from before labels lived on nodes stored it."""
    state = cohort.read_state()
    state.resolved_names = {}
    state.labels, state.node_labels = {}, {}
    state.names_by_granularity = {str(state.granularity): {"0": "rearing"}}
    cohort.write_state(state)


def test_a_legacy_runs_names_show_on_read_without_writing_anything(authored_cohort):
    """Review finding: such a run opened unlabelled in INTERPRET and ``--tree`` until its first
    edit. ``node_labels`` folds on a copy, so every surface shows the names, and reading writes
    nothing."""
    _legacy(authored_cohort)
    path = authored_cohort.state_path()
    before = path.read_bytes()
    run = authored_cohort.run()
    labels = run.node_names()
    assert list(labels.values()) == ["rearing"], "the legacy name, on its node"
    assert path.read_bytes() == before, "reading folded nothing on disk"
    run.name_nodes({})
    assert run.node_names() == labels, "the same node the edit folds it onto"


def test_the_first_edit_on_a_legacy_run_records_the_migration(authored_cohort):
    """§22: the fold is its own fact in the history, once."""
    import json

    _legacy(authored_cohort)
    run = authored_cohort.run()
    run.name_nodes({})
    run.name_nodes({})
    history = authored_cohort.history_path()
    entries = [
        json.loads(line)
        for line in history.read_text(encoding="utf-8-sig").splitlines()
        if line
    ]
    migrated = [e for e in entries if "migrated_names" in e]
    assert len(migrated) == 1 and migrated[0]["migrated_names"] == 1


@pytest.mark.parametrize("module", MODULES)
def test_neither_module_grows_its_own_label_edit(module):
    """The label edits live once, in ``LabelOperations``. A module that defined its own
    ``name_nodes`` would edit the labels past the rule check and the resolution."""
    tree = ast.parse(pathlib.Path(module).read_text(encoding="utf-8-sig"))
    own = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in MUTATING
    }
    assert not own, f"{module} defines its own {sorted(own)}"


def _calls(func) -> set[str]:
    return {
        child.func.attr if isinstance(child.func, ast.Attribute) else child.func.id
        for child in ast.walk(func)
        if isinstance(child, ast.Call)
        and isinstance(child.func, (ast.Attribute, ast.Name))
    }


def _methods(source: str, cls: str) -> dict:
    tree = ast.parse(source)
    klass = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
    )
    return {n.name: n for n in klass.body if isinstance(n, ast.FunctionDef)}


def test_every_label_edit_goes_through_the_one_door():
    """The source half: every label edit calls ``_label_edit``, and the door itself folds,
    checks and recomputes the resolution."""
    methods = _methods(
        pathlib.Path("detail/api/authored_run.py").read_text(encoding="utf-8-sig"),
        "LabelOperations",
    )
    for name in MUTATING:
        assert "_label_edit" in _calls(methods[name]), f"{name} skips the door"
    door = _calls(methods["_label_edit"])
    assert {"_reresolve", "check_labels", "label_resolution"} <= door


def test_the_scan_would_catch_an_edit_that_skipped_the_door():
    """Guard the guard: a scan that matches nothing passes forever and says nothing."""
    source = (
        "class LabelOperations:\n"
        "    def name_nodes(self, names):\n"
        "        def mutate(state):\n"
        "            state.node_labels.update(names)\n"
        "            return {}\n"
        "        return self._edit('name nodes', mutate)\n"
    )
    assert "_label_edit" not in _calls(
        _methods(source, "LabelOperations")["name_nodes"]
    )


def test_a_path_is_outermost_first_and_ends_in_the_label(authored_cohort):
    """``label`` is the last element, so a prefix is a roll-up; reversed, a reader taking the
    last element gets the coarsest name, which is plausible and wrong."""
    run = authored_cohort.run()
    tree = run._tree()
    outer = tree.n_nodes - 1
    inner = _inner(tree, outer)
    names = {str(inner): "resting", str(outer): "not moving"}
    assert authored.name_path(tree, inner, names) == ["not moving", "resting"]
    assert authored.name_path(tree, outer, names) == ["not moving"]
    labels = {"L1": "not moving", "L2": "resting"}
    _names, paths = authored.label_resolution(
        tree, labels, {str(outer): "L1", str(inner): "L2"}
    )
    assert paths == {"L1": "not moving", "L2": "not moving/resting"}


def test_an_unlabelled_run_has_no_paths(authored_cohort):
    """No path and no label are the same state: a path defaulting to an id would put a code
    where a name goes, on every row of a fresh run."""
    run = authored_cohort.run()
    assert authored.label_resolution(run._tree(), {}, {}) == ({}, {})


def test_other_is_not_a_label_name(authored_cohort):
    """The labels panel calls a label's own remainder "other": a sub-label of that name was two
    rows with one name, and Rename on it acted on the parent."""
    import pytest

    from detail.api.authored_run import check_labels

    with pytest.raises(ValueError, match="'other' is the name"):
        check_labels({"L1": "Other"}, {}, None)
