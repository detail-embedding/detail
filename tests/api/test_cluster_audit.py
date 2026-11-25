"""The audit of a clustering run whose labels nest (§19, §22).

Labels sit on tree nodes, and an instance reports the deepest one above it. The audit asks each
clip one question answered at the deepest level that is true, so a parent label gets an exact
precision from the same judgments that score its sub-labels; the unlabelled remainder is never
drawn. The storage arithmetic is `tests/storage/test_behavior_audit.py`; this is the run handle
and the CLI that read it.

The one-by-one draw and judge are gone (the review of every clip replaced them); a run audited
before keeps what it bought, so these tests seed that earlier audit with `tests/audit_seed.py`
and check what the readers make of it.
"""

import re
import time

from click.testing import CliRunner

from detail.cli import main

from tests.audit_seed import draw, judge


def _nested_artifact(run, n: int = 48) -> None:
    """Rewrite ``run``'s artifact as nested labels plus an unlabelled remainder.

    Written directly, as the other clustering tests here are: what is under test is the audit's
    reading of label paths, and a real fit would make the label mix an accident of the fixture.
    A quarter each: ``rearing`` (its own), ``rearing/supported``, ``grooming``, unlabelled.
    """
    from detail.api.authored_run import UNLABELLED
    from detail.storage import behavior_output as bo

    shapes = [
        ("101", "rearing", "rearing"),
        ("102", "supported", "rearing/supported"),
        ("103", "grooming", "grooming"),
        (UNLABELLED, "", ""),
    ]
    existing = run.artifact()
    bo.write_behavior_output(
        run._run_dir(),
        instances=[
            bo.BehaviorInstance(
                recording_id="recording_001",
                subject=0,
                t_ns=i * 1_000_000,
                end_t_ns=i * 1_000_000 + 500_000,
                start_frame=i,
                end_frame=i,
                source_label=shapes[i % 4][0],
            )
            for i in range(n)
        ],
        coverage=list(existing.coverage),
        manifest={
            **existing.manifest,
            "vocabulary": ["grooming", "rearing", "supported"],
            "names": {node: [name] for node, name, _path in shapes[:3]},
            "label_paths": {node: [path] for node, _name, path in shapes[:3]},
            "behavior_types": {
                "grooming": "state",
                "rearing": "state",
                "supported": "state",
            },
        },
    )


def test_one_batch_scores_the_parent_and_the_child(fitted_temporal):
    """The point of answering at the deepest true level: "rearing, not supported" is a miss
    for the sub-label and a hit for the parent, from the same keystroke."""
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    _nested_artifact(run)

    for row in draw(run, path="rearing/supported", budget=10):
        judge(run, row.instance_id, False, answer="rearing")
    child = run.audit_precision(path="rearing/supported")
    parent = run.audit_precision(path="rearing")
    assert child["n"] == 10 and child["precision"] == 0.0
    assert parent["n"] == 10 and parent["precision"] == 1.0


# --- cluster verdicts (§19) -----------------------------------------------------------------


def _two_clusters(experiment, run_id, name="climb"):
    run = experiment.authored_run(run_id)
    view = run.node_view()
    left, right = view.children(view.root)
    run.name_nodes({left: name, right: name})
    return run, left, right


def test_a_label_edit_stamps_the_label_and_the_labels_above_it(fitted_temporal):
    """Moving clusters in or out of a label resets that label's audit figure — and its parents',
    since a parent's audit covers its sub-labels. Making a label resets nothing (there is no audit
    of it yet), and neither does a rename, which moves no cluster.
    """
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    view = run.node_view()
    left, right = view.children(view.root)
    inner = view.children(left)[0]
    run.name_nodes({left: "climb", inner: "corner"})
    assert run.state().membership_changed_at == {}, (
        "a label just made has no audit to set aside — stamping it said 'edited after the "
        "audit' for every label"
    )
    other = view.children(left)[1]
    run.name_nodes(
        {other: "corner"}
    )  # corner gains a cluster: it and climb above it change
    stamped = dict(run.state().membership_changed_at)
    assert set(stamped) == {"L1", "L2"}

    run.rename_label("L2", "corner climb")
    assert run.state().membership_changed_at == stamped, "a rename reset the figure"

    run.name_nodes(
        {inner: "", other: ""}
    )  # corner's clusters leave it: climb changes too
    after = run.state().membership_changed_at
    assert "L2" not in after and after["L1"] >= stamped["L1"]


def test_the_headline_counts_only_judgments_made_after_an_edit(fitted_temporal):
    """Choosing clusters by how they scored cannot raise the score the choice was made on."""
    experiment, run_id = fitted_temporal
    run, left, right = _two_clusters(experiment, run_id)
    for row in draw(run, path="climb", budget=12):
        judge(run, row.instance_id, True)
    assert run.audit_precision(path="climb")["n"] > 0
    # The edit's stamp must fall after the judgments, which are to the second.
    time.sleep(1.1)
    run.name_nodes({right: ""})
    after = run.audit_precision(path="climb")
    assert after["since"] is not None
    assert (
        after["n"] == 0
    ), "judgments from before the edit still count toward the headline"


def _climb_bouts(run, left: int, right: int, monkeypatch, n: int = 40):
    """Twenty "climb" bouts from each of two clusters, written as separate instances.

    By hand, because in the shared fixture two clusters of one label sit back to back and now
    fuse into one instance — the point of labels having ids — which leaves nothing to tell apart.
    Which cluster each bout came from is stubbed the same way: even bouts from ``left``, odd from
    ``right``. The real attribution is ``NodeView.cluster_of_spans``, tested on its own.
    """
    from detail.api.temporal_clustering import TemporalClusteringRun
    from detail.storage import behavior_output as bo

    existing = run.artifact()
    rows = [
        bo.BehaviorInstance(
            recording_id=existing.coverage[0].recording_id,
            subject=existing.coverage[0].subject,
            t_ns=i * 100_000_000,
            end_t_ns=i * 100_000_000 + 40_000_000,
            start_frame=i,
            end_frame=i,
            source_label="L1",
        )
        for i in range(n)
    ]
    bo.write_behavior_output(
        run._run_dir(),
        instances=rows,
        coverage=list(existing.coverage),
        manifest={
            **existing.manifest,
            "vocabulary": ["climb"],
            "names": {"L1": ["climb"]},
            "label_paths": {"L1": ["climb"]},
            "behavior_types": {"climb": "state"},
        },
    )
    which = {
        r.instance_id: (left if (r.t_ns // 100_000_000) % 2 == 0 else right)
        for r in run.artifact().instances
    }
    monkeypatch.setattr(
        TemporalClusteringRun,
        "_audit_clusters",
        lambda self, rows: {
            r.instance_id: which[r.instance_id] for r in rows if r.instance_id in which
        },
    )
    return which


def test_each_cluster_gets_its_own_interval(fitted_temporal, monkeypatch):
    """Each cluster of the label an earlier audit judged comes back with its own precision and
    interval, size, share and what its clips were answered as — and no verdict: whether a
    cluster is good enough is the reader's call."""
    experiment, run_id = fitted_temporal
    run, left, right = _two_clusters(experiment, run_id)
    which = _climb_bouts(run, left, right, monkeypatch)
    drawn = draw(run, path="climb", budget=16)
    assert {which[r.instance_id] for r in drawn} == {
        left,
        right,
    }, "the seeded audit judged one cluster; the per-cluster figures below are vacuous"
    for row in drawn:
        judge(run, row.instance_id, which[row.instance_id] == left)
    rows = {r["node"]: r for r in run.audit_by_cluster(path="climb")}
    assert set(rows) == {left, right}
    assert rows[left]["estimate"]["precision"] == 1.0
    assert rows[right]["estimate"]["precision"] == 0.0
    assert "verdict" not in rows[left] and "clips_to_decide" not in rows[left]
    assert rows[left]["size"] == 20 and rows[left]["estimate"]["interval"]
    assert rows[right]["answers"].get("not climb", 0) == rows[right]["estimate"]["n"]
    assert abs(sum(r["share"] for r in rows.values()) - 1.0) < 1e-9
    # The headline is the stratified figure: each cluster weighted by its half of the label.
    assert abs(run.audit_precision(path="climb")["precision"] - 0.5) < 1e-9


def test_the_cli_reports_each_clusters_interval_and_its_width(
    fitted_temporal, monkeypatch
):
    from click.testing import CliRunner

    from detail.cli import main

    experiment, run_id = fitted_temporal
    run, left, right = _two_clusters(experiment, run_id)
    which = _climb_bouts(run, left, right, monkeypatch)
    for row in draw(run, path="climb", budget=16):
        judge(run, row.instance_id, which[row.instance_id] == left)
    result = CliRunner().invoke(
        main,
        [
            "analysis", "audit", "--experiment", str(experiment.dir), "--run", run_id,
            "--by-cluster", "--goal-width", "0.5",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert f"#{left}" in result.output and f"#{right}" in result.output
    assert "95% CI" in result.output and "goal 0.50" in result.output
    # The earlier audit's cluster rows ("N instances"), not the clip review's ("N clips").
    cluster_lines = [
        line
        for line in result.output.splitlines()
        if (f"#{left}" in line or f"#{right}" in line) and " instances, " in line
    ]
    assert len(cluster_lines) == 2, result.output
    for line in cluster_lines:
        assert re.search(r"\[CI width \d\.\d\d, goal 0\.50( reached)?\]$", line), line
    assert (
        "more would give" not in result.output
    ), "the width note projects more one-by-one clips, which nothing can draw any more"
    for word in ("holds", "fails", "undecided"):
        assert word not in result.output, "the CLI gave a verdict"


def test_own_only_checks_the_parents_own_clusters_without_its_sub_labels(
    fitted_temporal,
):
    """A parent's own clusters — its "other" — can be read on their own: the figure leaves its
    sub-labels' judgments out, from an audit that judged both."""
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    _nested_artifact(run)  # rearing / rearing/supported / grooming / unlabelled rows
    whole = draw(run, path="rearing", budget=40)
    own = [r for r in whole if r.label_path == "rearing"]
    vacuous = "the seed judged no sub-label; this guards nothing"
    assert own and len(own) < len(whole), vacuous
    for row in whole:
        judge(run, row.instance_id, True, answer=row.label_path)

    assert run.audit_precision(path="rearing")["n"] == len(whole)
    figure = run.audit_precision(path="rearing", own_only=True)
    leaked = "the sub-label's judgments counted toward its parent's own"
    assert figure["n"] == len(own), leaked
    split = run.audit_precision_by_recording(path="rearing", own_only=True)
    assert sum(row["n"] for row in split) == len(own)


def test_an_edited_label_says_how_many_judgments_came_before(
    fitted_temporal, monkeypatch
):
    """The reported text: "not audited yet · edited after the audit" on a parent whose sub-label
    had 66 judged clips. A sub-label's judgments outlive an edit elsewhere in the branch, and the
    headline says how many it set aside."""
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    view = run.node_view()
    left, _right = view.children(view.root)
    inner, other = view.children(left)[:2]
    run.name_nodes({left: "climb", inner: "corner"})
    # Judged yesterday: in the test the edit lands in the same second, and `since` is inclusive.
    monkeypatch.setattr(
        "detail.storage.behavior_audit.utc_timestamp", lambda: "2020-01-01T00:00:00Z"
    )
    for row in draw(run, path="climb/corner", budget=12):
        judge(run, row.instance_id, True, answer="climb/corner")
    monkeypatch.undo()
    # An edit in climb's branch that leaves corner's clips as they were.
    run.name_nodes({other: "wall"})
    after = run.audit_precision(path="climb")
    assert after["since"] is not None
    assert after["n"] == 0 and after["n_before"] > 0, after

    result = CliRunner().invoke(
        main,
        ["analysis", "audit", "--experiment", str(experiment.dir), "--run", run_id],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    climb = next(
        line for line in result.output.splitlines() if line.startswith("climb")
    )
    assert f"0 audited since, {after['n_before']} before" in climb, climb


# --- review fixes: answers survive renames, re-audit after an edit, the history entry ---------


def _nested_labels(run):
    view = run.node_view()
    left, _right = view.children(view.root)
    inner = view.children(left)[0]
    run.name_nodes({left: "climb", inner: "corner"})
    return left, inner


def _audit_corner_as_corner(run) -> int:
    drawn = draw(run, path="climb/corner", budget=12)
    for row in drawn:
        judge(run, row.instance_id, True, answer="climb/corner")
    return len(drawn)


def test_a_rename_keeps_every_recorded_answer(fitted_temporal):
    """The reported bug: answers are stored as label paths, so renaming "climb" left every
    "climb › corner" answer naming a path that no longer existed — each yes read as no.
    """
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    _nested_labels(run)
    assert _audit_corner_as_corner(run)
    before = run.audit_precision(path="climb/corner")["precision"]
    run.rename_label("L1", "rear")
    assert run.audit_precision(path="rear/corner")["precision"] == before == 1.0
    assert run.audit_precision(path="rear")["precision"] == 1.0


def test_renaming_the_sub_label_itself_keeps_its_answers(fitted_temporal):
    """The other half of a rename: the answers name the sub-label's own path. (A merge moves
    clusters, so their instance ids change and the judgments stop joining, as any membership
    change does — that is the documented behaviour, not this bug.)"""
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    _nested_labels(run)
    assert _audit_corner_as_corner(run)
    run.rename_label("L2", "front wall")
    assert run.audit_precision(path="climb/front wall")["precision"] == 1.0


def test_the_cli_reads_every_label_edit_before_writing_any(fitted_temporal):
    """A malformed --rename-label used to fail after --name had already written."""
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    left, _right = run.node_view().children(run.node_view().root)
    result = CliRunner().invoke(
        main,
        [
            "analysis", "temporal-interpret", "--experiment", str(experiment.dir), "--run", run_id,
            "--name-node", f"{left}=climb", "--rename-label", "nonsense",
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert (
        not run.state().node_labels
    ), "the node was labelled before the bad pair was read"


def test_a_failing_label_command_writes_nothing(fitted_temporal):
    """**A promise of all-or-nothing that writes per item.** The command read every pair's
    *syntax* up front, but resolved each label name only when its own edit ran, one write per
    pair — so ``--rename-label groom=g2 --merge-label g2=NOPE`` (no NOPE) failed after the
    rename had landed, and the corrected command then failed on ``groom``. The whole command is
    now one edit (``LabelOperations.edit_labels``): one state write, or none.
    """
    from detail.storage import temporal_runs

    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    left, right = run.node_view().children(run.node_view().root)
    run.name_nodes({left: "groom", right: "walk"})
    state_file = run._run_dir() / temporal_runs.TEMPORAL_FILENAME
    before = state_file.read_bytes()
    result = CliRunner().invoke(
        main,
        [
            "analysis", "temporal-interpret", "--experiment", str(experiment.dir), "--run", run_id,
            "--name-node", f"{left}=rear", "--rename-label", "walk=w2",
            "--merge-label", "w2=NOPE",
        ],
    )  # fmt: skip
    assert result.exit_code != 0, result.output
    assert "no label called 'NOPE'" in result.output
    assert state_file.read_bytes() == before, "part of the failed command was written"


def test_a_label_command_reads_its_own_earlier_edits(fitted_temporal):
    """Names resolve in order within the one edit: a rename's new name can be merged."""
    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    left, right = run.node_view().children(run.node_view().root)
    run.name_nodes({left: "groom", right: "walk"})
    result = CliRunner().invoke(
        main,
        [
            "analysis", "temporal-interpret", "--experiment", str(experiment.dir), "--run", run_id,
            "--rename-label", "walk=w2", "--merge-label", "w2=groom",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert sorted(run.state().labels.values()) == ["groom"]
    assert "renamed 'walk' to 'w2'" in result.output
    assert "merged 'w2' into 'groom'" in result.output


def test_the_cli_label_edits_report_progress(fitted_temporal, monkeypatch):
    from detail.api.authored_run import LabelOperations

    experiment, run_id = fitted_temporal
    run = experiment.authored_run(run_id)
    left, _right = run.node_view().children(run.node_view().root)
    seen = []
    real = LabelOperations.edit_labels

    def spy(self, names, *, progress=None, **kwargs):
        seen.append(progress)
        return real(self, names, progress=progress, **kwargs)

    monkeypatch.setattr(LabelOperations, "edit_labels", spy)
    result = CliRunner().invoke(
        main,
        [
            "analysis", "temporal-interpret", "--experiment", str(experiment.dir), "--run", run_id,
            "--name-node", f"{left}=climb",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen and seen[0] is not None, "the label edit ran without a progress bar"
