"""A clustering run's precision from reviewing every clip of a cluster (§19/§22,
``api/clip_review.py``).

A reviewer takes the wrong clips out of a cluster (``take_out_clips``) and calls ``finish_review``,
which writes one census: every clip of the cluster judged, ``no`` for each taken out or moved,
``yes`` for the rest. A cluster's figure is then exact; a label's combines its clusters' by size.
Run on the codebook decode, whose labels hold dozens of clips.

**The class: a judgment keyed by the thing judged, not by what it was judged as.** A census answer
says "this clip is (not) climb". Matched on the clip alone, it answered for whatever label the
clip's cluster carries now — and a newly made label has no ``membership_changed_at`` to set it
aside, so a cluster moved from a reviewed label to a new one showed a complete precision it was
never given. The rule: an answer counts only under the ``label_path`` it was written for; a path
moves only through clusters that kept their label (a rename, a label added above) or a merge, and
the answers move with it (``_carry_review``). Its twin: **an inverse write that stamps "now"**. An
undo is an edit, so it stamped the label as changed and set aside a review of exactly the clusters
the undo put back; ``restore_labels`` restores ``membership_changed_at`` with the clusters.
Swept: ``audit.csv`` answers name the path the reviewer chose, so they were not keyed on the clip
alone — but its ``_carry_answers`` shared ``path_moves`` and carried a moved cluster's answers to
its new label too, fixed by the same kept-label rule.
"""

import time

import numpy as np
import pytest
from click.testing import CliRunner

from detail.api.authored_run import ReadOnlyRunError
from detail.api.clip_edits import clip_ref
from detail.cli import main
from detail.storage import audit_figures
from detail.storage import behavior_audit
from detail.storage import clip_review as review_store
from detail.storage import temporal_runs


def _labelled(experiment, run_id):
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    climb, rear = view.children(view.root)
    run.name_nodes({climb: "climb", rear: "rear"})
    return run, climb, rear


def _keys(rows) -> list[tuple]:
    return [clip_ref(r).key for r in rows]


def test_finishing_writes_one_census_of_every_clip_in_one_append(
    fitted_codebook, monkeypatch
):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    clips = run.node_view().clips(climb).rows()
    run.take_out_clips(clips[:2], "climb")
    run.take_out_clips(clips[2], "climb", "rear")
    appends = []
    real = review_store.AppendLog.append_many
    monkeypatch.setattr(
        review_store.AppendLog,
        "append_many",
        lambda self, rows: appends.append(len(rows)) or real(self, rows),
    )

    row = run.finish_review(climb)

    assert appends == [len(clips)], "one write for the whole census"
    log = review_store.read_review(run._run_dir())
    assert sorted(r.clip.key for r in log) == sorted(_keys(clips))
    no = {r.clip.key for r in log if r.answer == review_store.NO}
    assert no == set(_keys(clips[:3])), "taken out and moved are both no"
    estimate = row["estimate"]
    assert estimate["precision"] == pytest.approx((len(clips) - 3) / len(clips))
    assert estimate["remaining"] == 0 and audit_figures.complete(estimate)
    assert row["node"] == climb and row["size"] == len(clips)


def test_a_later_census_supersedes_the_earlier_one(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    clips = run.node_view().clips(climb).rows()
    run.take_out_clips(clips[0], "climb")
    run.finish_review(climb)
    run.put_back_clips(clips[0])
    assert run.finish_review(climb)["estimate"]["precision"] == 1.0


def test_an_edit_from_another_label_is_not_a_no(fitted_codebook):
    """A take-out made from a label the cluster no longer has does not apply, so it is a yes."""
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    clip = run.node_view().clips(climb).rows()[0]
    run.take_out_clips(clip, "climb")
    run.name_nodes({climb: "rear"})
    assert run.taken_out(), "the edit is still stored, and no longer applies"
    row = run.finish_review(climb)
    assert row["estimate"]["precision"] == 1.0


def test_clusters_of_one_label_combine_by_size(fitted_codebook):
    """The small cluster all yes and the big one with two taken out: the label reads the
    clusters' sizes."""
    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    a, b = view.children(view.children(view.root)[0])
    run.name_nodes({a: "climb", b: "climb"})
    by_node = {c["node"]: c["clips"] for c in run._review_clusters("climb")}
    sizes = {n: len(c) for n, c in by_node.items()}
    small, big = sorted(sizes, key=sizes.get)
    run.take_out_clips(by_node[big][:2], "climb")
    run.finish_review(small)
    partial = run.review_precision("climb")
    assert partial["remaining"] == sizes[big], "one cluster left: not exact yet"
    assert not audit_figures.complete(partial)
    run.finish_review(big)

    estimate = run.review_precision("climb")

    expected = (sizes[small] + sizes[big] - 2) / (sizes[small] + sizes[big])
    assert estimate["precision"] == pytest.approx(expected)
    assert audit_figures.complete(estimate)
    rows = {row["node"]: row for row in run.review_by_cluster("climb")}
    assert rows[small]["estimate"]["precision"] == 1.0
    log = review_store.read_review(run._run_dir())
    assert rows[big]["reviewed_at"] == max(r.drawn_at for r in log), "AUDIT's '· 8 Nov'"
    assert rows[big]["estimate"]["precision"] == pytest.approx(
        (sizes[big] - 2) / sizes[big]
    )


def test_a_parent_combines_its_own_clips_with_its_sub_labels(fitted_codebook):
    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    top = view.children(view.root)[0]
    sub = view.children(top)[0]
    run.name_nodes({top: "climb", sub: "front"})
    clusters = {c["path"]: c for c in run._review_clusters("climb")}
    assert set(clusters) == {"climb", "climb/front"}
    run.take_out_clips(clusters["climb/front"]["clips"], "front")
    run.finish_review(sub)
    run.finish_review(top)
    own, front = len(clusters["climb"]["clips"]), len(clusters["climb/front"]["clips"])
    assert run.review_precision("climb")["precision"] == pytest.approx(
        own / (own + front)
    )
    assert run.review_precision("climb/front")["precision"] == 0.0


def test_a_membership_change_resets_the_figure_and_a_take_out_does_not(fitted_codebook):
    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    a, b = view.children(view.children(view.root)[0])
    run.name_nodes({a: "climb", b: "climb"})
    run.finish_review(a)
    run.take_out_clips(view.clips(b).rows()[-1], "climb")
    rows = {r["node"]: r for r in run.review_by_cluster("climb")}
    assert audit_figures.complete(
        rows[a]["estimate"]
    ), "a take-out is not a membership change"
    time.sleep(1.1)  # the stamps are whole seconds
    run.label_nodes({b: None})  # climb's clusters changed
    assert run.review_precision("climb") is None


def test_a_cluster_moved_to_a_new_label_starts_unreviewed(fitted_codebook):
    """Climb reviewed whole, then its cluster given to a label made for it: the new label has
    no change time to set climb's answers aside, and they answer "is this climb?", not "is this
    scale?". Matched on the clip alone, scale showed climb's complete census."""
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    run.finish_review(climb)
    assert audit_figures.complete(run.review_precision("climb"))

    run.label_nodes({climb: run.create_label("scale")})

    assert run.review_precision("scale") is None
    (row,) = [r for r in run.review_by_cluster() if r["node"] == climb]
    assert row["label"] == "scale"
    assert row["estimate"]["remaining"] == row["size"] and row["reviewed_at"] is None
    run.finish_review(climb)
    assert audit_figures.complete(run.review_precision("scale"))


def test_a_rename_keeps_the_review(fitted_codebook):
    """A rename moves no cluster: the answers are carried to the new path and still count."""
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    clips = run.node_view().clips(climb).rows()
    run.take_out_clips(clips[0], "climb")
    run.finish_review(climb)
    figure = run.review_precision("climb")["precision"]
    label = next(k for k, v in run.state().labels.items() if v == "climb")

    run.rename_label(label, "climbing")

    estimate = run.review_precision("climbing")
    assert audit_figures.complete(estimate)
    assert estimate["precision"] == pytest.approx(figure)


def _side(change, index):
    return tuple({k: v[index] for k, v in part.items()} for part in change)


@pytest.mark.parametrize("undo", [True, False], ids=["undone", "kept"])
def test_undoing_a_misclick_keeps_the_review_and_keeping_it_resets(
    fitted_codebook, undo
):
    """Climb reviewed, then rear's cluster put into climb by mistake and ⌘Z'd: climb's clusters
    are exactly what was reviewed, so the review stands. The undo restores the label's change
    time with its clusters; a restore that stamped "now" set the whole review aside. Kept, the
    same edit is a real change and resets the figure."""
    from detail.api import authored_run as authored

    experiment, run_id = fitted_codebook
    run, climb, rear = _labelled(experiment, run_id)
    run.take_out_clips(run.node_view().clips(climb).rows()[0], "climb")
    run.finish_review(climb)
    reviewed = run.review_precision("climb")
    time.sleep(1.1)  # the stamps are whole seconds
    before = run.label_maps()
    run.name_nodes({rear: "climb"})
    change = authored.label_change(before, run.label_maps())
    assert (
        run.review_precision("climb") is None
    ), "the misclick changed climb's clusters"

    if undo:
        run.restore_labels(*_side(change, 0))
        assert run.label_maps() == before
        again = run.review_precision("climb")
        assert again is not None and audit_figures.complete(again)
        assert again["precision"] == pytest.approx(reviewed["precision"])
        run.restore_labels(*_side(change, 1))  # redo: the change again
    assert run.review_precision("climb") is None


def test_a_restore_without_change_times_is_an_edit_like_any_other(fitted_codebook):
    """A caller that passes no change times (a script) gets the stamp every edit makes."""
    experiment, run_id = fitted_codebook
    run, climb, rear = _labelled(experiment, run_id)
    run.finish_review(climb)
    time.sleep(1.1)
    labels, nodes, _stamps = run.label_maps()
    run.name_nodes({rear: "climb"})
    run.restore_labels(labels, {str(rear): nodes[str(rear)]})
    assert run.review_precision("climb") is None


def test_a_refused_finish_logs_nothing(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    inner = run.node_view().children(climb)[0]
    history = run._run_dir() / temporal_runs.HISTORY_FILENAME
    before = history.read_bytes()
    with pytest.raises(ValueError, match="is not a labelled cluster"):
        run.finish_review(inner)
    assert history.read_bytes() == before
    run.finish_review(climb)
    assert (
        history.read_bytes() != before
    ), "the history is the file finish_review logs to"


def test_an_unlabelled_node_cannot_be_finished(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    inner = run.node_view().children(climb)[0]
    with pytest.raises(ValueError, match="is not a labelled cluster"):
        run.finish_review(inner)
    assert not review_store.review_path(run._run_dir()).is_file()


def test_a_snapshot_reads_its_own_review_and_refuses_to_add_to_it(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    run.finish_review(climb)
    view = run.view(run.pin(reason="finish"))
    run.reopen()
    run.take_out_clips(run.node_view().clips(climb).rows()[0], "climb")
    run.finish_review(climb)
    assert view.review_precision("climb")["precision"] == 1.0
    assert run.review_precision("climb")["precision"] < 1.0
    with pytest.raises(ReadOnlyRunError, match="will not finish_review"):
        view.finish_review(climb)


def test_a_finalized_run_refuses_to_finish(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    run.finalize()
    with pytest.raises(ValueError, match="finalized"):
        run.finish_review(climb)
    assert not review_store.review_path(run._run_dir()).is_file()


def test_the_old_audit_is_left_alone_and_still_read(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    for row in run.audit_sample(path="climb", budget=3):
        run.label_audit(row.instance_id, True)
    audit_bytes = (run._run_dir() / "audit.csv").read_bytes()
    run.finish_review(climb)
    assert (run._run_dir() / "audit.csv").read_bytes() == audit_bytes
    assert run.audit_precision(path="climb")["n"] == 3
    assert audit_figures.complete(run.review_precision("climb"))


def test_the_cli_prints_each_cluster_then_the_earlier_audit(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, rear = _labelled(experiment, run_id)
    for row in run.audit_sample(path="climb", budget=3):
        run.label_audit(row.instance_id, True)
    clips = run.node_view().clips(climb).rows()
    run.finish_review(climb)

    result = CliRunner().invoke(
        main,
        ["analysis", "audit", "--experiment", str(experiment.dir), "--run", run_id],
    )

    assert result.exit_code == 0, result.output + str(result.exception)
    lines = result.output.splitlines()
    assert lines[0].startswith("Clip review")
    assert lines[1] == (
        f"  climb: precision 1.00  (all {len(clips)} audited), every cluster reviewed (1)"
    )
    assert lines[2].split() == [
        f"#{climb}",
        str(len(clips)),
        "clips,",
        "100%",
        "of",
        "the",
        "label",
        "precision",
        "1.00",
        "(all",
        str(len(clips)),
        "audited)",
    ]
    assert lines[3] == "  rear: 0 of 1 clusters reviewed"
    assert lines[4].split()[0] == f"#{rear}" and lines[4].endswith("not reviewed")
    heading = lines.index("Earlier one-by-one audit:")
    assert lines[heading + 1].startswith("climb: precision"), lines
    assert "n=3" in lines[heading + 1]


def test_estimate_weights_are_the_audit_s_own():
    """The review reuses ``behavior_audit.stratified_estimate`` rather than a second copy."""
    groups = {1: (10, [True] * 3), 2: (30, [False] * 3)}
    estimate = behavior_audit.stratified_estimate(groups)
    assert estimate["precision"] == pytest.approx(0.25)
    assert np.isfinite(estimate["interval"][0])
