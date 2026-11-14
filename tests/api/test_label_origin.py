"""A label a difference screen chose carries the screen, and a comparison that scores it says so
(§22 *The difference screen*, §24).

**The class.** *A label whose provenance is unrecorded gets scored by the contrast that selected
it.* The screen ranks every cluster of a run across a metadata column and hands the user a few to
watch; the user names one from the clips. Nothing about that is wrong -- until the same label is
compared across the same column, where its effect is the effect the screen selected it for,
reported as if it had been found independently. The number is plausible, the run is green, and
nothing fails (CLAUDE.md, *labels score a method; they never build one*: the label did not build
the method, but the contrast chose what the label is on). The only defence is the record: the
origin travels with the node's own label (``run_labels.LabelOrigin``, ``label_origins`` on the
clustering state), survives undo and redo by id, goes when the label goes, and a labels comparison
over that column writes the labels it is not independent of into its record (``screened_labels``),
which every surface prints from one function (``comparison_report.screened_note`` /
``screened_sentence``).

The guards: the storage round trip (a state written with origins reads them back, a malformed one
is dropped rather than crashing the run), the edit door (an origin is written with a name, kept
through a rename, dropped with the label, put back by ``restore_labels``), the relation of a node
to a screen's kept clusters (kept / under / above / none), the origin built from a saved screen,
and the labels comparison's record.
"""

import pytest

from detail.api import authored_run as authored
from detail.api.comparison import run_comparison
from detail.storage import run_labels
from detail.storage import temporal_runs


@pytest.fixture(autouse=True)
def _silence_projection(monkeypatch):
    """The §19 rewrite is not under test (``tests/api/test_node_names.py`` says why)."""
    from detail.api.temporal_clustering import TemporalClusteringRun

    monkeypatch.setattr(TemporalClusteringRun, "_project", lambda self, **_: None)


ORIGIN = run_labels.LabelOrigin(
    comparison="cmp_01",
    kept="7",
    relation="kept",
    column="drug",
    screen="drug vs vehicle · paired",
    text="more under drug than vehicle, 2.1× (3 of 4 animal(s) in that direction)",
)


def _a_branch(run) -> int:
    tree = run._tree()
    return int(tree.children[tree.n_nodes - 1 - tree.n_leaves][0])


# --- the stored shape --------------------------------------------------------------------


def test_an_origin_round_trips_through_the_state():
    state = temporal_runs.TemporalClusteringState(
        method="codebook",
        embed_run_id="emb_01",
        track_id=0,
        labels={"L1": "climb"},
        node_labels={"7": "L1"},
        label_origins={"7": ORIGIN.to_dict()},
    )
    data = state.to_dict()
    assert data["label_origins"] == {"7": ORIGIN.to_dict()}
    read = temporal_runs.TemporalClusteringState.from_dict(data)
    assert read.label_origins == {"7": ORIGIN.to_dict()}
    assert run_labels.LabelOrigin.from_dict(read.label_origins["7"]) == ORIGIN


def test_a_malformed_origin_is_dropped_not_fatal():
    """A state written by a later version, or edited by hand, still opens: an origin missing
    what it needs, or with a relation this version does not know, is left out."""
    data = {
        "labels": {"L1": "climb"},
        "node_labels": {"7": "L1", "9": "L1"},
        "label_origins": {
            "7": ORIGIN.to_dict(),
            "9": {"comparison": "cmp_02"},
            "11": {"comparison": "cmp_03", "kept": "2", "relation": "beside"},
            "13": "cmp_04",
        },
    }
    assert run_labels.read_origins(data) == {"7": ORIGIN.to_dict()}
    assert run_labels.LabelOrigin.from_dict(None) is None


# --- the relation of a node to what a screen kept ------------------------------------------

#: A tree in root-first ancestor lists: 6 is the root, 4 = (0, 1), 5 = (2, 3).
ANCESTORS = {0: [6, 4], 1: [6, 4], 2: [6, 5], 3: [6, 5], 4: [6], 5: [6], 6: []}


def _ancestors(node):
    return ANCESTORS[int(node)]


@pytest.mark.parametrize(
    "node, kept, relation",
    [
        (4, [4], "kept"),
        (0, [4], "under"),
        (6, [4], "above"),
        (5, [4], None),
        (2, [4], None),
        (6, [4, 5], "above"),
    ],
)
def test_the_relation_of_a_node_to_the_kept_clusters(node, kept, relation):
    assert run_labels.screen_relation(node, kept, _ancestors) == relation


def test_the_origin_names_the_kept_cluster_the_node_stands_to():
    kept_texts = {4: "more under drug", 5: "less under drug"}
    build = lambda node: run_labels.label_origin(  # noqa: E731
        node,
        comparison="cmp_01",
        kept_texts=kept_texts,
        column="drug",
        screen="drug vs vehicle · paired",
        ancestors_of=_ancestors,
    )
    assert build(4) == run_labels.LabelOrigin(
        "cmp_01", "4", "kept", "drug", "drug vs vehicle · paired", "more under drug"
    )
    assert build(3).kept == "5" and build(3).relation == "under"
    assert build(3).text == "less under drug"
    above = build(6)
    assert above.relation == "above" and above.kept in {"4", "5"}
    assert build(2).relation == "under" and build(2).kept == "5"
    alone = run_labels.label_origin(
        2,
        comparison="cmp_01",
        kept_texts={4: "more under drug"},
        column="drug",
        screen="",
        ancestors_of=_ancestors,
    )
    assert alone is None, "a node the screen did not point at has no origin"


def test_a_node_above_several_kept_clusters_cites_the_nearest():
    """The root above a kept node and that node's kept child cites the child's parent -- the
    one closest to it, so the origin reads as the cluster it is a slice of."""
    deeper = {**ANCESTORS, 7: [6, 4, 0]}
    origin = run_labels.label_origin(
        6,
        comparison="cmp_01",
        kept_texts={7: "", 4: ""},
        column="drug",
        screen="",
        ancestors_of=lambda n: deeper[int(n)],
    )
    assert origin.kept == "4"


# --- the edit door -------------------------------------------------------------------------


def test_an_origin_is_written_with_the_name_and_kept_through_a_rename(authored_cohort):
    run = authored_cohort.run()
    node = _a_branch(run)
    run.name_nodes({node: "climb"}, origins={node: ORIGIN})
    assert run.state().label_origins == {str(node): ORIGIN.to_dict()}
    run.name_nodes(
        {node: "climbing"}
    )  # a rename in place: the cluster is the same cluster
    state = run.state()
    assert state.labels == {"L1": "climbing"}
    assert state.label_origins == {str(node): ORIGIN.to_dict()}
    assert run.label_maps()[3] == {str(node): ORIGIN.to_dict()}


def test_an_origin_goes_with_the_nodes_own_label(authored_cohort):
    run = authored_cohort.run()
    node = _a_branch(run)
    run.name_nodes({node: "climb"}, origins={node: ORIGIN})
    run.name_nodes({node: ""})
    assert run.state().label_origins == {}
    run.name_nodes({node: "climb"}, origins={node: ORIGIN})
    run.delete_label(run.state().node_labels[str(node)])
    assert run.state().label_origins == {}


def test_a_dict_origin_is_accepted_and_a_malformed_one_refused(authored_cohort):
    run = authored_cohort.run()
    node = _a_branch(run)
    run.name_nodes({node: "climb"}, origins={node: ORIGIN.to_dict(), 3: None})
    assert run.state().label_origins == {str(node): ORIGIN.to_dict()}
    with pytest.raises(ValueError, match="not a label origin"):
        run.name_nodes({node: "climb"}, origins={node: {"comparison": "cmp_01"}})


def test_label_nodes_by_id_takes_an_origin_too(authored_cohort):
    run = authored_cohort.run()
    node = _a_branch(run)
    label = run.create_label("climb")
    run.label_nodes({node: label}, origins={node: ORIGIN})
    assert run.state().label_origins == {str(node): ORIGIN.to_dict()}


def test_restoring_either_side_puts_the_origin_back(authored_cohort):
    """INTERPRET's ⌘Z and ⇧⌘Z: a label written from a screen, undone, loses its origin with
    the label; redone, gets it back -- by the same four maps the page restores."""
    run = authored_cohort.run()
    node = _a_branch(run)
    before = run.label_maps()
    run.name_nodes({node: "climb"}, origins={node: ORIGIN})
    after = run.label_maps()
    change = authored.label_change(before, after)
    assert change[3] == {str(node): (None, ORIGIN.to_dict())}
    run.restore_labels(*_side(change, 0))
    assert run.label_maps() == before
    assert run.state().label_origins == {}
    run.restore_labels(*_side(change, 1))
    assert run.label_maps() == after
    assert run.state().label_origins == {str(node): ORIGIN.to_dict()}


def _side(change, index):
    return tuple({k: v[index] for k, v in part.items()} for part in change)


def test_ancestors_of_walks_the_fits_tree(authored_cohort):
    run = authored_cohort.run()
    tree = run._tree()
    root = tree.n_nodes - 1
    assert run.ancestors_of(root) == []
    node = _a_branch(run)
    assert run.ancestors_of(node) == [root]
    assert run.ancestors_of(node) == authored.ancestors(tree, node)


# --- the origin a saved screen gives ----------------------------------------------------


def test_the_origin_from_a_saved_screen_carries_its_sentence(tmp_path, monkeypatch):
    """Built from the saved screen alone -- its record, its ranking's kept items and their
    sentences -- so the CLI's ``--from-screen`` and the dock's ticked screen record the same
    thing. The synthetic fixture clears nothing on its own, so the ranking is stood in for:
    one kept cluster, node 4, with node 0 beneath it and node 2 beside it."""
    from detail.analysis import compare, comparison_report
    from detail.api.comparison import Comparison, label_origin, label_origins
    from tests.api.test_comparison import _node_screen

    exp, view, comp = _node_screen(tmp_path, monkeypatch)
    key = ("time_fraction", "node:4", "", "whole")
    item = compare.Effect(key, 0.74, 4, 4, 3, True)
    ranking = compare.Ranking(
        contrasts=[item],
        floor=0.2,
        draws=16,
        seed=0,
        design=compare.PAIRED,
        levels=("veh", "fent"),
        unit=("animal",),
        n_units=4,
        min_units=2,
        kept=(key,),
    )
    monkeypatch.setattr(Comparison, "ranking", lambda self: ranking)
    record = comp.record()
    origin = label_origin(exp, comp.id, 4, ancestors_of=view.ancestors)
    assert origin == run_labels.LabelOrigin(
        comparison=comp.id,
        kept="4",
        relation="kept",
        column="drug",
        screen="veh vs fent · paired",
        text=comparison_report.suggestion(item, ranking, record),
    )
    assert "2.1×" in origin.text and "3 of 4 animal(s)" in origin.text
    under = label_origin(exp, comp.id, 0, ancestors_of=view.ancestors)
    assert under.relation == "under" and under.kept == "4" and under.text == origin.text
    assert label_origin(exp, comp.id, 2, ancestors_of=view.ancestors) is None

    # Several nodes, one ranking: re-ranking re-reads stats.csv and redraws the null.
    rankings = []
    monkeypatch.setattr(
        Comparison, "ranking", lambda self: rankings.append(1) or ranking
    )
    many = label_origins(exp, comp.id, [4, 0, 2], ancestors_of=view.ancestors)
    assert many == {4: origin, 0: under, 2: None}
    assert len(rankings) == 1


def test_a_labels_comparison_is_not_a_screen_to_label_from(tmp_path):
    from detail.api.comparison import label_origin
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    with pytest.raises(ValueError, match="compares labels, not clusters"):
        label_origin(exp, comp.id, 0, ancestors_of=lambda n: [])


# --- the labels comparison records what it is not independent of ----------------------


def _drugged(authored_cohort):
    from detail.storage import metadata as md

    exp = authored_cohort.experiment
    md.upsert_row(exp.dir, "r1", {"animal": "A", "drug": "veh"})
    md.upsert_row(exp.dir, "r2", {"animal": "B", "drug": "fent"})
    return exp


def test_a_labels_comparison_records_the_labels_a_screen_over_its_column_chose(
    authored_cohort,
):
    """The record carries ``screened_labels`` -- keyed by the label *as measured* (the name in
    the snapshot the comparison read), each with the screen, the kept cluster and the node --
    only for a screen over one of the compared columns. A label chosen over ``sex`` and
    compared over ``drug`` is independent of that contrast, and says nothing."""
    exp = _drugged(authored_cohort)
    run = authored_cohort.run()
    node = _a_branch(run)
    run.name_nodes({node: "climb"}, origins={node: ORIGIN})
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    record = comp.record()
    assert record["screened_labels"] == {
        "climb": [{"node": str(node), **ORIGIN.to_dict()}]
    }
    assert "screened_labels" in comp.record(), "persisted, not only returned"
    other = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["animal"], "aggregate_to": []}
    )
    assert other.record()["screened_labels"] == {}


def test_a_node_screen_records_no_screened_labels(tmp_path, monkeypatch):
    from tests.api.test_comparison import _node_screen

    _exp, _view, comp = _node_screen(tmp_path, monkeypatch)
    assert "screened_labels" not in comp.record()


def test_the_labels_written_from_a_screen_are_listed_from_the_run(
    authored_cohort, monkeypatch
):
    """The screen's page lists what was labelled from it, as the run holds it *now*: a label
    removed since is gone from the list, and a labels comparison lists nothing."""
    import detail.api.comparison as api
    from detail.api.comparison import labels_from_screen
    from tests.analysis.test_node_measures import synthetic_view

    exp = _drugged(authored_cohort)
    view = synthetic_view(recording_ids=("r1", "r2"))
    monkeypatch.setattr(api, "_node_view", lambda experiment, run_id: view)
    screen = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "source": "nodes",
            "group_by": ["drug"],
            "aggregate_to": [],
            "candidate_floor_s": 1.0,
        },
    )
    run = authored_cohort.run()
    node = _a_branch(run)
    origin = run_labels.LabelOrigin(
        screen.id, "4", "under", "drug", "veh vs fent · between"
    )
    run.name_nodes({node: "climb"}, origins={node: origin})
    assert labels_from_screen(exp, screen.id) == [
        {"node": node, "name": "climb", "kept": "4", "relation": "under", "alpha": ""}
    ]
    run.name_nodes({node: ""})
    assert labels_from_screen(exp, screen.id) == []
    labels = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    assert labels_from_screen(exp, labels.id) == []


# --- one wording, every surface ----------------------------------------------------------


def test_the_note_and_the_sentence_read_from_the_record():
    from detail.analysis import comparison_report as report

    record = {
        "group_by": ["drug"],
        "screened_labels": {
            "climb": [{"node": "9", **ORIGIN.to_dict()}],
            "rear": [
                {"node": "3", **ORIGIN.to_dict(), "relation": "under", "kept": "2"},
                {"node": "5", **ORIGIN.to_dict(), "comparison": "cmp_04"},
            ],
        },
    }
    assert report.screened_note(record, "climb") == (
        "chosen from screen cmp_01 (cluster #7 kept)"
    )
    assert report.screened_note(record, "rear") == (
        "chosen from screen cmp_01 (inside cluster #2 kept), cmp_04 (cluster #7 kept)"
    )
    assert report.screened_note(record, "groom") == ""
    assert report.screened_note({}, "climb") == ""
    sentence = report.screened_sentence(record)
    assert sentence.startswith(
        "climb (cmp_01), rear (cmp_01, cmp_04) were chosen from a "
    )
    assert "over drug" in sentence and "not an independent finding" in sentence
    assert report.screened_sentence({"screened_labels": {}}) == ""
    one = report.screened_sentence(
        {"group_by": ["drug"], "screened_labels": {"climb": []}}
    )
    assert one.startswith("climb () was chosen")


def test_a_screened_labels_suggestion_carries_the_note(tmp_path):
    """The ranking's sentence for a screened label ends with the note, so the Overview table,
    ``ranking.csv``, ``summary.md`` and the CLI all say it without a column of their own.
    """
    from detail.analysis import comparison_report as report
    from tests.api.test_comparison import CROSSOVER, _PAIRED, _cohort

    exp = _cohort(tmp_path, CROSSOVER)
    comp = run_comparison(exp, _PAIRED)
    ranking = comp.ranking()
    item = ranking.contrasts[0]
    label = item.measure_key[1]
    plain = report.suggestion(item, ranking, comp.record())
    assert " · chosen from screen" not in plain
    noted = report.suggestion(
        item,
        ranking,
        {
            **comp.record(),
            "screened_labels": {label: [{**ORIGIN.to_dict(), "node": "9"}]},
        },
    )
    assert noted == f"{plain} · chosen from screen cmp_01 (cluster #7 kept)"


# --- the strictness a screen was read at travels with the label (§22) ---------------------


def test_an_origin_keeps_the_level_and_the_numbers_it_was_written_at():
    """The stored shape: the level the screen was read at and the kept cluster's p's, as
    strings like every other field; an origin written before strictness existed reads back
    with them blank, which is the screen's own floor."""
    origin = run_labels.LabelOrigin(
        comparison="cmp_01",
        kept="7",
        relation="kept",
        column="drug",
        alpha="0.2",
        family_p="0.14",
        own_p="0.0039",
        q="0.9",
    )
    assert run_labels.LabelOrigin.from_dict(origin.to_dict()) == origin
    assert run_labels.LabelOrigin.from_dict(ORIGIN.to_dict()).alpha == ""
    stored = {k: v for k, v in origin.to_dict().items() if k not in ("alpha", "q")}
    read = run_labels.LabelOrigin.from_dict(stored)
    assert read.alpha == "" and read.q == "" and read.family_p == "0.14"


def test_label_origin_records_the_level_and_the_kept_clusters_numbers():
    origin = run_labels.label_origin(
        0,
        comparison="cmp_01",
        kept_texts={4: "more under drug"},
        column="drug",
        screen="drug vs vehicle · paired",
        ancestors_of=_ancestors,
        alpha=0.2,
        numbers={4: {"family_p": 0.14, "own_p": 1 / 256, "q": 0.9}},
    )
    assert origin.relation == "under" and origin.kept == "4"
    assert origin.alpha == "0.2"
    assert origin.family_p == "0.14" and origin.own_p == "0.00390625"
    assert origin.q == "0.9"
    at_floor = run_labels.label_origin(
        4,
        comparison="cmp_01",
        kept_texts={4: "more under drug"},
        column="drug",
        screen="",
        ancestors_of=_ancestors,
    )
    assert at_floor.alpha == "" and at_floor.family_p == ""


def test_the_origin_from_a_screen_read_looser_than_its_floor(tmp_path, monkeypatch):
    """At p ≤ 0.2 a cluster the floor hid is kept, so a label on it carries the screen with
    the level and the cluster's own numbers; at the screen's own floor the same node is
    nothing to the screen. ``kept_nodes_of`` reads the same level, so the CLI's refusal and
    the dock's leaves agree with the origin."""
    from detail.analysis import compare
    from detail.api.comparison import Comparison, kept_nodes_of, label_origin
    from tests.api.test_comparison import _node_screen

    exp, view, comp = _node_screen(tmp_path, monkeypatch)
    kept_key = ("time_fraction", "node:4", "", "whole")
    lead_key = ("time_fraction", "node:5", "", "whole")
    kept = compare.Effect(
        kept_key, 0.74, 4, 4, 3, True, family_p=0.02, own_p=0.0625, q=0.1
    )
    lead = compare.Effect(
        lead_key, 0.31, 4, 4, 3, True, family_p=0.2, own_p=0.0625, q=0.1
    )
    ranking = compare.Ranking(
        contrasts=[kept, lead],
        floor=0.5,
        draws=16,
        seed=0,
        floor_distribution=[0.1 * i for i in range(16)],
        design=compare.PAIRED,
        levels=("veh", "fent"),
        unit=("animal",),
        n_units=4,
        min_units=2,
        kept=(kept_key,),
        ancestors={"node:5": set(), "node:4": set()},
    )
    monkeypatch.setattr(Comparison, "ranking", lambda self: ranking)
    assert kept_nodes_of(ranking) == [4] and kept_nodes_of(ranking, 0.2) == [4, 5]
    assert comp.kept_nodes(0.2) == [4, 5]
    assert label_origin(exp, comp.id, 5, ancestors_of=view.ancestors) is None
    origin = label_origin(exp, comp.id, 5, ancestors_of=view.ancestors, alpha=0.2)
    assert origin.relation == "kept" and origin.kept == "5"
    assert origin.alpha == "0.2" and origin.family_p == "0.2"
    assert origin.own_p == "0.0625" and origin.q == "0.1"
    assert "1.4×" in origin.text
    at_floor = label_origin(exp, comp.id, 4, ancestors_of=view.ancestors)
    assert at_floor.alpha == "" and at_floor.family_p == "0.02"
