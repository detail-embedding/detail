"""Look-alike clusters (``api/cluster_lookalikes.py``): which clusters in different branches of a
run's tree are one behaviour, and one label for them.

The classes guarded here:

- **A group held together by a link nobody can see.** A group chains pair by pair, so every
  member must be there through a saved comparison that looked alike — its newest, so a pair
  compared again and told apart drops out — and the links are part of the group.
- **A comparison that looks alike because of what it left in.** Only comparisons with position
  and facing left out are evidence: kept, the place measures would answer a question the person
  does not ask, and a comparison of the rest of the run is not a pair at all.
- **A label that sweeps in another behaviour, unsaid.** A label covers everything under its
  cluster with no label of its own, and clusters nest. On an_17 the rule "the outer member's
  label covers the inner" climbed the rearing chain to the tree's two halves and proposed one
  name for 75% of the run, walking clusters included — invisible on a test tree whose
  look-alikes did not nest. Every member is listed; one that would also label a cluster outside
  the group says which and how much of the run, and starts unticked.
- **A suggestion that overwrites a person's label.** Labelling a group goes through the label
  edit: a member with another label of its own keeps it, a member the rules refuse is skipped
  with the rule's own reason, and each label made records that it came from a look-alike group.
- **Every pair when a person meant a few.** ``pairs: labelled`` compares each branch top a
  label covers (its own, or the nearest above) against every other, never two sides one label
  covers, never a cluster against its branch; the counts and the warning every surface shows
  before every pair say how many pairs each mode compares.
- **A proposed name that ignores the person.** The proposed name is a label the members
  carry: the one most of them share, the largest member's on a tie; none when no member has one.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import detail.api.cluster_lookalikes as lookalikes
import detail.api.cluster_profile as cluster_profile
from detail.analysis.descriptors import DESCRIPTORS, FrameSlice
from tests.api.test_cluster_profile import (  # noqa: F401 — `profiled` is a fixture
    _apart_from,
    _config,
    profiled,
)

# --------------------------------------------------------------------------- #
# A small tree:      6
#                  /   \
#                 4     5
#                / \   / \
#               0   1 2   3
# --------------------------------------------------------------------------- #
PARENT = [4, 4, 5, 5, 6, 6, -1]


class _View:
    """The parts of a ``NodeView`` the look-alike layer reads."""

    def __init__(self, seconds=None, floor_s=30.0, parent=None):
        self._parent = parent or PARENT
        self.tree = SimpleNamespace(n_nodes=len(self._parent), parent=self._parent)
        self.root = self._parent.index(-1)
        self.min_cluster_s = floor_s
        self._seconds = seconds or {0: 100, 1: 90, 2: 80, 3: 70, 4: 190, 5: 150, 6: 340}

    def children(self, node):
        return [n for n, p in enumerate(self._parent) if p == node]

    def ancestors(self, node):
        out, current = [], self._parent[node]
        while current >= 0:
            out.append(current)
            current = self._parent[current]
        return out[::-1]

    def summary(self, node):
        return SimpleNamespace(time_ns=int(self._seconds[node] * 1e9))


def _evidence(a, b, accuracy, *, created="2026-10-30T00:00:00Z", floor=0.5, margin=0.1):
    return lookalikes.PairEvidence(
        node=a,
        against=b,
        accuracy=accuracy,
        floor=floor,
        margin=margin,
        separated=accuracy - floor >= margin,
        place_accuracy=0.9,
        place_floor=0.5,
        params={},
        created_at=created,
        file=f"profiles/node_{a}/against_{b}/profile.yaml",
    )


def save_comparison(run_dir, a: int, b: int, accuracy: float) -> None:
    """A comparison of ``a`` against ``b`` with place left out, holding the fields
    ``save_profile`` writes that the look-alike layer reads — for a surface test that needs
    groups on disk without profiling."""
    import yaml

    folder = cluster_profile.profile_dir(run_dir, a, b)
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "run_id": "an_01",
        "node": a,
        "compare": "node",
        "comparison_node": b,
        "params": {"compare": "node", "against": b, "leave_out_place": True},
        "created_at": "2026-10-30T00:00:00Z",
        "explanation": {
            "accuracy": accuracy,
            "floor": 0.5,
            "margin": 0.1,
            "separated": accuracy - 0.5 >= 0.1,
        },
        "place_explanation": {"accuracy": 0.9, "floor": 0.5},
    }
    (folder / "profile.yaml").write_text(yaml.safe_dump(record), encoding="utf-8")


# --- which pairs ------------------------------------------------------------------------ #
def test_branch_tops_are_both_halves_of_splits_over_the_floor():
    assert lookalikes.branch_tops(_View()) == [0, 1, 2, 3, 4, 5]
    # #3 under the floor: #5's split only trims a sliver, so #2 and #3 are not tops.
    short = _View({0: 100, 1: 90, 2: 80, 3: 10, 4: 190, 5: 90, 6: 280})
    assert lookalikes.branch_tops(short) == [0, 1, 4, 5]


def test_every_pair_of_branch_tops_but_never_a_cluster_with_its_branch():
    pairs = lookalikes.candidate_pairs(_View())
    assert (0, 4) not in pairs and (2, 5) not in pairs, "one holds the other"
    assert pairs == [
        (0, 1), (0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5),
        (2, 3), (2, 4), (3, 4), (4, 5),
    ]  # fmt: skip


# --- which pairs: labelled ------------------------------------------------------------- #
# The tree above, labelled: label L1 on #4 covers #0 and #1 beneath it.
_ON_4 = {"4": "L1"}


def test_labelled_pairs_keep_a_pair_with_a_labelled_side():
    pairs = lookalikes.labelled_pairs(_View(), _ON_4)
    assert pairs == [
        (0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (2, 4), (3, 4), (4, 5),
    ]  # fmt: skip
    assert (2, 3) not in pairs, "neither side labelled"


def test_labelled_pairs_drop_two_sides_one_label_covers():
    """#0 and #1 sit under one label (on #4): already one behaviour, so not compared."""
    assert (0, 1) not in lookalikes.labelled_pairs(_View(), _ON_4)
    # #0 with a label of its own is another behaviour inside L1, and is compared with #1.
    assert (0, 1) in lookalikes.labelled_pairs(_View(), {**_ON_4, "0": "L2"})
    # One label on two separate clusters: one behaviour, wherever its clusters sit.
    assert (0, 2) not in lookalikes.labelled_pairs(_View(), {"0": "L1", "2": "L1"})


def test_labelled_pairs_never_ask_a_cluster_against_its_branch():
    pairs = lookalikes.labelled_pairs(_View(), {"0": "L1"})
    assert (0, 4) not in pairs, "#4 holds #0"
    assert set(pairs) <= set(lookalikes.candidate_pairs(_View()))
    assert pairs == [(0, 1), (0, 2), (0, 3), (0, 5)]


def test_the_label_covering_a_node_is_its_own_else_the_nearest_above():
    tree = _View().tree
    assert lookalikes.covering_label(tree, 0, _ON_4) == "L1"
    assert lookalikes.covering_label(tree, 0, {**_ON_4, "0": "L2"}) == "L2"
    assert lookalikes.covering_label(tree, 2, _ON_4) is None
    assert lookalikes.labelled_tops(_View(), _ON_4) == [0, 1, 4]
    assert lookalikes.labelled_tops(_View(), {}) == []


def test_pairs_for_each_mode():
    view = _View()
    assert lookalikes.pairs_for(view, _ON_4, "all") == lookalikes.candidate_pairs(view)
    assert lookalikes.pairs_for(view, _ON_4, "labelled") == lookalikes.labelled_pairs(
        view, _ON_4
    )
    with pytest.raises(ValueError, match="unknown pairs 'names'"):
        lookalikes.pairs_for(view, _ON_4, "names")


# --- how many pairs, and about how long ------------------------------------------------- #
def _job(**kw):
    from detail.configs.cluster_lookalikes import ClusterLookalikesConfig

    return ClusterLookalikesConfig(source_run_id="an_01", **kw)


def _timed(a, b, seconds, job):
    from detail.configs.cluster_lookalikes import PROFILE_FIELDS

    return dataclasses.replace(
        _evidence(a, b, 0.9),
        params={name: getattr(job, name) for name in PROFILE_FIELDS},
        elapsed_s=seconds,
    )


def test_the_counts_say_how_many_pairs_each_mode_compares():
    counts = lookalikes.count_pairs(_View(), _ON_4, [], _job())
    assert counts.counts == {"all": 11, "labelled": 9}
    assert counts.n_labelled_tops == 3
    assert counts.skipped_all == counts.skipped_labelled == 0


def test_the_estimate_is_this_runs_own_pairs_and_skips_the_compared_ones():
    job = _job()
    evidence = [_timed(0, 2, 10.0, job), _timed(2, 3, 30.0, job)]
    counts = lookalikes.count_pairs(_View(), _ON_4, evidence, job)
    assert counts.per_pair_s == 20.0 and "median of 2 pairs" in counts.basis
    assert (counts.skipped_all, counts.skipped_labelled) == (2, 1)
    assert counts.seconds("all") == (11 - 2) * 20.0
    assert counts.to_compare("all") == 9 and counts.to_compare("labelled") == 8
    assert "9 of 11 pairs left (2 already compared)" in counts.warning()
    assert counts.seconds("labelled") == (9 - 1) * 20.0
    # Redo compares them again; other settings are not this run's timing.
    again = lookalikes.count_pairs(_View(), _ON_4, evidence, _job(skip_existing=False))
    assert again.skipped_all == 0
    other = lookalikes.count_pairs(_View(), _ON_4, evidence, _job(seed=7))
    assert "time budget" in other.basis


def test_the_warning_names_both_counts_or_that_nothing_is_labelled():
    counts = lookalikes.count_pairs(_View(), _ON_4, [], _job())
    text = counts.warning()
    assert "compute intensive" in text and "11 pairs" in text
    assert "'labelled'" in text and "9 pairs" in text
    bare = lookalikes.count_pairs(_View(), {}, [], _job()).warning()
    assert "11 pairs" in bare and "no cluster is labelled yet" in bare


# --- groups ----------------------------------------------------------------------------- #
def test_look_alike_pairs_chain_into_one_group_with_their_links():
    groups = lookalikes.find_groups(
        _View(),
        [_evidence(0, 2, 0.52), _evidence(2, 1, 0.55), _evidence(0, 3, 0.95)],
    )
    assert len(groups) == 1
    group = groups[0]
    assert group.nodes == (0, 1, 2), "largest first"
    assert {link.pair for link in group.links} == {(0, 2), (1, 2)}
    assert 3 not in group.nodes, "a pair the body measures tell apart is no link"


def test_a_pair_is_read_at_its_newest_comparison():
    older = _evidence(0, 2, 0.52, created="2026-10-29T00:00:00Z")
    newer = _evidence(2, 0, 0.95, created="2026-10-30T00:00:00Z")
    assert lookalikes.find_groups(_View(), [older, newer]) == []
    assert len(lookalikes.find_groups(_View(), [newer, older])) == 0
    assert len(lookalikes.find_groups(_View(), [older])) == 1


def test_a_comparison_that_could_not_be_fitted_is_no_link():
    assert lookalikes.find_groups(_View(), [_evidence(0, 2, float("nan"))]) == []


def test_a_member_that_holds_another_behaviour_says_so_and_starts_unticked():
    # an_17's shape: #5 looks like #0 and holds #2, which does too — and #3, which does not.
    view = _View()
    evidence = [_evidence(0, 5, 0.5), _evidence(0, 2, 0.5)]
    group = lookalikes.find_groups(view, evidence, {})[0]
    assert group.nodes == (5, 0, 2), "every member listed, nested ones too"
    assert set(group.sweeps) == {5}
    (swept,) = group.sweeps[5]
    assert swept.node == 3
    assert swept.share == pytest.approx(70 / 340)
    assert group.ticked == (0, 2), "the member that sweeps #3 in is not ticked"


def test_a_labelled_cluster_is_not_swept_in():
    view = _View()
    evidence = [_evidence(0, 5, 0.5), _evidence(0, 2, 0.5)]
    labelled = lookalikes.find_groups(view, evidence, {3: "walking"})[0]
    assert labelled.sweeps == {}, "#3 keeps its own label"
    assert labelled.ticked == (5, 0), "#2 is inside #5, whose label covers it"


def test_a_swept_cluster_is_said_once_through_its_outermost_and_less_what_is_kept():
    # #6 (the root, as a hand comparison could pair it) holds #4 and #5; #5 holds member #3.
    group = lookalikes.find_groups(_View(), [_evidence(6, 3, 0.5)], {})[0]
    assert [(s.node, s.share) for s in group.sweeps[6]] == [
        (4, pytest.approx(190 / 340)),
        (5, pytest.approx((150 - 70) / 340)),
    ], "#0-#2 are said through #4 and #5; #3 is a member"


def test_a_cluster_under_a_member_inside_a_swept_one_is_still_said():
    # an_17's #1393 ⊃ #1209 "forward walking" ⊃ member #1101 ⊃ #931 "walking forward": #1209's
    # share leaves #1101 out, so #931 said "through #1209" was never said at all.
    #         0
    #       1   2          member #1 holds #3, which holds member #5,
    #     3   4            which holds #7; #4 and #8 carry the group's label
    #   5   6
    # 7   8
    parent = [-1, 0, 0, 1, 1, 3, 3, 5, 5]
    seconds = {0: 900, 1: 500, 2: 400, 3: 300, 4: 200, 5: 200, 6: 100, 7: 120, 8: 80}
    view = _View(seconds, parent=parent)
    labels = {4: "rearing", 8: "rearing"}
    group = lookalikes.find_groups(
        view, [_evidence(2, 1, 0.5), _evidence(2, 5, 0.5)], labels
    )[0]
    assert [(s.node, s.share) for s in group.sweeps[1]] == [
        (7, pytest.approx(120 / 900)),
        (3, pytest.approx((300 - 200) / 900)),
    ]
    assert [s.node for s in group.sweeps[5]] == [7]
    assert group.ticked == (2,)


def test_a_plan_that_sweeps_names_what_and_warns():
    view = _View()
    evidence = [_evidence(0, 5, 0.5), _evidence(0, 2, 0.5)]
    group = lookalikes.find_groups(view, evidence, {})[0]
    plan = lookalikes.plan_join({}, PARENT, [5, 0], "rearing")
    text = lookalikes.plan_text(plan, "rearing", sweeps=group.sweeps)
    assert "#5 — also labels #3 (20.6% of the run)" in text
    assert "Label them first" in text
    clean = lookalikes.plan_join({}, PARENT, list(group.ticked), "rearing")
    assert "Label them first" not in lookalikes.plan_text(
        clean, "rearing", sweeps=group.sweeps
    )


def test_a_label_a_member_carries_is_the_proposed_name():
    view = _View()
    evidence = [_evidence(0, 2, 0.5), _evidence(2, 1, 0.5)]
    group = lookalikes.find_groups(view, evidence, {2: "Rear_to_right"})[0]
    assert group.name == "Rear_to_right" and group.named == "a member's label"
    shared = lookalikes.find_groups(
        view, evidence, {0: "rearing up", 1: "Rearing up", 2: "standing"}
    )[0]
    assert shared.name == "rearing up" and "2 of them" in shared.named
    tie = lookalikes.find_groups(view, evidence, {1: "walking", 2: "standing"})[0]
    assert tie.name == "walking", "#1 is larger than #2"
    unnamed = lookalikes.find_groups(view, evidence, {})[0]
    assert unnamed.name == "" and unnamed.named == "no member has a label"


# --- one label -------------------------------------------------------------------------- #
def test_a_plan_keeps_a_persons_label_and_joins_an_existing_one():
    plan = lookalikes.plan_join(
        {"1": "grooming", "3": "rearing"}, PARENT, [0, 1, 2], "Rearing"
    )
    by_node = {step.node: step for step in plan.steps}
    assert not by_node[1].accepted and "grooming" in by_node[1].reason
    assert by_node[0].accepted and by_node[0].joins == "rearing"
    assert by_node[0].joins_yours
    assert by_node[2].accepted


def test_a_plan_skips_what_the_label_rules_refuse_with_their_reason():
    plan = lookalikes.plan_join({"4": "rearing"}, PARENT, [0, 2], "rearing")
    by_node = {step.node: step for step in plan.steps}
    assert not by_node[0].accepted
    assert "already inside a branch labelled" in by_node[0].reason
    assert by_node[2].accepted


def test_a_plan_with_no_name_labels_nothing():
    plan = lookalikes.plan_join({}, PARENT, [0, 2], "  ")
    assert not plan.accepted and all(s.reason == "no name given" for s in plan.steps)


# --- on a real run ---------------------------------------------------------------------- #
@pytest.fixture
def three(profiled, monkeypatch):
    """The profile fixture's node and two clusters apart from it and from each other, whose
    members all differ from the rest only in where they are."""
    experiment, run, node, _stand_in = profiled
    view = run.node_view()
    first = _apart_from(view, node)
    related = {node, first, *view.ancestors(node), *view.ancestors(first)}
    second = next(
        n
        for n in sorted(
            range(view.tree.n_nodes), key=lambda n: -view.summary(n).members
        )
        if n not in related
        and not ({node, first} & set(view.ancestors(n)))
        and len(view.sample_runs(n, n=10_000, seed=0)) >= 12
    )
    by_unit: dict = {}
    for offset, member in enumerate([node, first, second]):
        for m in view.sample_runs(member, n=100_000, seed=0):
            key = (m["recording_id"], int(m["subject"]))
            by_unit.setdefault(key, []).append((m, 300.0 * (offset + 1)))
    monkeypatch.setattr(
        cluster_profile,
        "_trajectory",
        lambda _e, key, _r, _ts: _placed_track(by_unit.get(key, [])),
    )
    real = cluster_profile.explainers.explain

    def body_alike(X, y, groups, names, **kw):
        # The fixture is one recording, so rows are held out one by one and a cluster's
        # time-adjacent members share frames: any noise in a body measure then "separates"
        # two clusters. The body fit's verdict is stood in (scored at its own floor); the
        # place fit, the save and everything read from it are real.
        e = real(X, y, groups, names, **kw)
        if any(DESCRIPTORS[n].place for n in names):
            return e
        return dataclasses.replace(e, accuracy=e.floor, separated=False)

    monkeypatch.setattr(cluster_profile.explainers, "explain", body_alike)
    return experiment, run, (node, first, second)


def _placed_track(spans):
    """A track at one speed everywhere, each cluster's runs at its own height in the image:
    the clusters differ only in where they are."""
    end = max([int(s["end_t_ns"]) for s, _y in spans] + [0]) + 600_000_000_000
    t_ns = np.arange(0, end, 10_000_000, dtype=np.int64)
    y = np.zeros(len(t_ns))
    for span, height in spans:
        lo, hi = span["start_t_ns"], span["end_t_ns"]
        y[(t_ns >= lo) & (t_ns <= hi)] = height
    x = np.cumsum(np.random.default_rng(len(spans)).uniform(0.5, 1.5, len(t_ns)))
    return FrameSlice(
        frame_idx=np.arange(len(t_ns)),
        t_ns=t_ns,
        stride=1,
        centroid=np.stack([x, y], axis=1),
        area=500.0 * np.random.default_rng(7).uniform(0.9, 1.1, len(t_ns)),
        bbox=np.stack([x - 10, y, x + 10, y + 20.0], axis=1),
    )


def _compare(experiment, run, a, b, **kw):
    return experiment.profile_cluster(
        str(run.id),
        a,
        _config(compare="node", against=b, margin=0.3, **kw).model_dump(),
    )


def test_saved_comparisons_group_and_one_label_is_written_through_the_label_edit(three):
    experiment, run, (a, b, c) = three
    _compare(experiment, run, a, b)
    _compare(experiment, run, c, b)
    groups = run.lookalike_groups()
    assert len(groups) == 1, [g.text() for g in groups]
    assert set(groups[0].nodes) == {a, b, c}

    plan = run.plan_lookalike_label(groups[0].nodes, "rearing")
    assert len(plan.accepted) == 3
    applied = run.label_lookalikes(groups[0].nodes, "rearing")
    assert len(applied.accepted) == 3
    state = run.state()
    assert {state.node_names[str(n)] for n in (a, b, c)} == {"rearing"}
    assert len(state.labels) == 1, "one label for the group"
    source = state.node_label_sources[str(a)]
    assert source["from"] == lookalikes.LOOKALIKES_SOURCE
    assert source["group"] == sorted((a, b, c))

    again = run.label_lookalikes(groups[0].nodes, "rearing")
    assert not again.accepted, "nothing left to label"

    run.rename_label(state.node_labels[str(a)], "rising")
    assert run.state().node_label_sources[str(a)]["group"] == sorted((a, b, c))
    run.name_nodes({a: "grooming"})  # the person chose this cluster's label now
    assert str(a) not in run.state().node_label_sources
    assert str(b) in run.state().node_label_sources


def test_the_state_keeps_where_a_label_came_from():
    """``node_label_sources`` round-trips through the state's YAML shape, and a state written
    before it existed reads as every label typed."""
    from detail.storage import temporal_runs

    state_class = temporal_runs.TemporalClusteringState
    source = {
        "from": lookalikes.LOOKALIKES_SOURCE,
        "group": [5, 9],
        "label": "L1",
        "accepted_at": "2026-10-29T00:00:00Z",
    }
    data = {
        "method": "m",
        "labels": {"L1": "walking"},
        "node_labels": {"5": "L1"},
        "node_label_sources": {"5": source},
    }
    state = state_class.from_dict(data)
    assert state.node_label_sources == {"5": source}
    assert state_class.from_dict(state.to_dict()).node_label_sources == {"5": source}
    data.pop("node_label_sources")
    assert state_class.from_dict(data).node_label_sources == {}


def test_a_comparison_that_kept_place_or_was_against_the_rest_is_not_evidence(three):
    experiment, run, (a, b, _c) = three
    _compare(experiment, run, a, b, leave_out_place=False)
    experiment.profile_cluster(str(run.id), a, _config().model_dump())
    assert lookalikes.read_evidence(experiment.analysis_run_dir(str(run.id))) == []
    assert run.lookalike_groups() == []


def test_the_evidence_carries_the_place_fit(three):
    experiment, run, (a, b, _c) = three
    _compare(experiment, run, a, b)
    (item,) = lookalikes.read_evidence(experiment.analysis_run_dir(str(run.id)))
    assert (item.node, item.against) == (a, b)
    assert item.looks_alike
    assert item.place_accuracy > item.place_floor + 0.3, item.text()
    assert item.file == f"profiles/node_{a}/against_{b}/profile.yaml"
