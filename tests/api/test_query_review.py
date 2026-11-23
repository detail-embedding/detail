"""A query run's audit: every clip the cutoff keeps, reviewed one recording at a time (§20).

The review writes ``clip_review.csv`` (a census per recording, under the behavior's id) and
removes clips from the behavior's output (``Behavior.removed``). These tests hold the rules
``DESIGN_PLAN.md`` §20 *The audit* states: a figure is exact only where every clip is answered,
the old figure is never read as the one at a lower cutoff, a removal edits the output and no
input, and the review never hands a surface a score.
"""

import shutil

import pytest

from detail.api.authored_run import ReadOnlyRunError
from detail.api.experiment import Experiment
from detail.core.errors import UserInputError
from detail.storage import analysis_runs as analysis_store
from detail.storage import clip_review as review_store
from detail.storage import query_runs
from tests.api.test_query import _searched, _two_behaviors


@pytest.fixture(scope="module")
def _searched_template(tmp_path_factory):
    """One completed two-behavior search over s1 and s2, built once for the module. Both
    behaviors share one example clip, so they have the same candidates: one span is a candidate
    of two behaviors, the case a per-behavior removal and withdrawal exist for."""
    root = tmp_path_factory.mktemp("query_review")
    exp, _run = _searched(root, behaviors=_two_behaviors(["s1", "s2"]))
    return exp.dir


@pytest.fixture
def run(_searched_template, tmp_path):
    """A per-test copy of the searched run."""
    copy = tmp_path / _searched_template.name
    shutil.copytree(_searched_template, copy)
    return Experiment(copy).query_run("an_01")


def _scores(run, behavior: str, recording: str) -> list[float]:
    """The behavior's reviewable candidates in ``recording``, highest score first."""
    pool = run._audit_pool(behavior)
    examples = run._audit_examples(pool)
    return sorted(
        (
            row.score
            for row in pool
            if row.recording_id == recording and row.instance_id not in examples
        ),
        reverse=True,
    )


def _cutoff_keeping(run, k: int, behavior: str = "q01") -> float:
    """A cutoff that keeps exactly ``k`` of s1's reviewable candidates."""
    scores = _scores(run, behavior, "s1")
    assert len(scores) > k and scores[k] < scores[k - 1], "fixture too small to cut"
    return scores[k - 1]


def _keys(rows) -> set:
    return {query_runs.clip_key(row) for row in rows}


def _artifact_keys(run, behavior: str = "q01") -> set:
    return _keys(
        row for row in run.artifact().instances if row.source_label == behavior
    )


def test_finishing_a_recording_writes_one_answer_per_clip_and_an_exact_figure(run):
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    clips = run.review_clips("q01", "s1")
    assert len(clips) == 6
    run.take_out_clips(clips[:2], "q01")
    row = run.finish_review("q01", "s1")

    assert row["exact"] and row["estimate"]["remaining"] == 0
    assert (row["estimate"]["n"], row["estimate"]["n_confirmed"]) == (6, 4)
    assert row["removed"] == 2
    written = review_store.read_review(run._run_dir())
    assert len(written) == 6 and len({r.drawn_at for r in written}) == 1, "one census"
    assert {r.label_path for r in written} == {"q01"}
    assert {r.node for r in written} == {review_store.NO_NODE}
    removed = {
        (c["recording_id"], c["subject"], c["t_ns"], c["end_t_ns"]) for c in clips[:2]
    }
    assert {r.clip.key for r in written if r.answer == "no"} == removed
    assert not removed & _artifact_keys(run), "a removed clip is still an instance"


def test_raising_the_cutoff_keeps_the_figure_exact_and_lowering_it_reports_the_floor(
    run,
):
    """The answered clips are the top of the candidates, not a sample of them: below the
    reviewed range the figure is refused and the exact one is given at the floor."""
    at = _cutoff_keeping(run, 6)
    run.set_cutoff("q01", at)
    clips = run.review_clips("q01", "s1")
    run.take_out_clips(clips[:1], "q01")
    run.finish_review("q01", "s1")
    reviewed = run.review_precision("q01", "s1")
    assert reviewed["exact"] and reviewed["precision"] == pytest.approx(5 / 6)

    run.set_cutoff("q01", _cutoff_keeping(run, 3))
    raised = run.review_precision("q01", "s1")
    assert raised["exact"] and raised["remaining"] == 0

    run.set_cutoff("q01", _cutoff_keeping(run, 9))
    (row,) = [r for r in run.review_by_recording("q01") if r["recording_id"] == "s1"]
    assert not row["exact"] and row["estimate"]["remaining"] == 3
    assert row["reviewed_floor"] == pytest.approx(at)
    lowered = run.review_precision("q01", "s1")
    assert lowered["precision"] is None and "not reviewed" in lowered["reason"]
    assert lowered["at_floor"]["remaining"] == 0
    assert lowered["at_floor"]["precision"] == pytest.approx(5 / 6)


def test_a_removal_drops_the_clip_at_every_cutoff_and_changes_no_input(run):
    """Guard (§20 *A removal edits the output, never an input*). Without the removal the clip
    is an instance at both cutoffs — the baseline that makes the two absences mean something —
    and the search shards, the examples and the non-examples are byte-identical after it.
    """
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    run.set_cutoff("q02", 0.0)
    clip = run.review_clips("q01", "s1")[0]
    key = (clip["recording_id"], clip["subject"], clip["t_ns"], clip["end_t_ns"])
    assert key in _artifact_keys(run)
    run.set_cutoff("q01", 0.0)
    assert key in _artifact_keys(run)

    shards = {
        p: p.read_bytes()
        for p in query_runs.search_dir(run._run_dir()).rglob("*")
        if p.is_file()
    }
    before = run.state().behavior("q01")
    run.take_out_clips(clip, "q01")
    for cutoff in (0.0, _cutoff_keeping(run, 6)):
        run.set_cutoff("q01", cutoff)
        assert key not in _artifact_keys(run), f"removed clip back at cutoff {cutoff}"
    after = run.state().behavior("q01")
    assert (after.examples, after.negatives) == (before.examples, before.negatives)
    assert {p: p.read_bytes() for p in shards} == shards
    assert key in _artifact_keys(run, "q02"), "removing from q01 touched q02"


def test_putting_a_clip_back_withdraws_its_answer_as_that_behavior_only(run):
    cutoff = _cutoff_keeping(run, 6)
    for behavior in ("q01", "q02"):
        run.set_cutoff(behavior, cutoff)
    clip = run.review_clips("q01", "s1")[0]
    for behavior in ("q01", "q02"):
        run.take_out_clips(clip, behavior)
        run.finish_review(behavior, "s1")

    run.put_back_clips(clip, "q01")
    (q01,) = [r for r in run.review_by_recording("q01") if r["recording_id"] == "s1"]
    (q02,) = [r for r in run.review_by_recording("q02") if r["recording_id"] == "s1"]
    assert not q01["exact"] and q01["estimate"]["remaining"] == 1
    assert q02["exact"] and q02["removed"] == 1
    assert run.taken_out("q02") and not run.taken_out("q01")


def test_a_rename_keeps_the_review(run):
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    run.finish_review("q01", "s1")
    run.name_behavior("q01", "climbing")
    (row,) = [
        r for r in run.review_by_recording("climbing") if r["recording_id"] == "s1"
    ]
    assert row["exact"] and row["estimate"]["n"] == 6


def test_the_users_own_clips_are_not_reviewed_and_cannot_be_removed(run):
    run.set_cutoff("q01", 0.0)
    pool = run._audit_pool("q01")
    examples = [row for row in pool if row.instance_id in run._audit_examples(pool)]
    assert (
        examples
    ), "no candidate overlaps the fixture's example clip; this guards nothing"
    shown = {
        (c["recording_id"], c["subject"], c["t_ns"], c["end_t_ns"])
        for c in run.review_clips("q01", examples[0].recording_id)
    }
    assert not _keys(examples) & shown
    with pytest.raises(UserInputError, match="example"):
        run.take_out_clips([query_runs.clip_key(examples[0])], "q01")


def test_a_clip_is_removed_never_moved_and_must_be_a_candidate(run):
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    clip = run.review_clips("q01", "s1")[0]
    with pytest.raises(ValueError, match="removed, not moved"):
        run.take_out_clips(clip, "q01", "q02")
    with pytest.raises(UserInputError, match="not a candidate"):
        run.take_out_clips([("s1", 0, 1, 2)], "q01")
    assert not run.taken_out()


def test_a_behavior_without_a_cutoff_has_nothing_to_review(run):
    with pytest.raises(UserInputError, match="set a cutoff on CALIBRATE first"):
        run.review_clips("q01", "s1")


def test_the_review_order_is_shuffled_stable_and_carries_no_score(run):
    run.set_cutoff("q01", _cutoff_keeping(run, 8))
    clips = run.review_clips("q01", "s1")
    assert clips == run.review_clips("q01", "s1"), "reopening reorders the clips"
    assert all(set(c) == {"recording_id", "subject", "t_ns", "end_t_ns"} for c in clips)
    pool = {query_runs.clip_key(row): row.score for row in run._audit_pool("q01")}
    scores = [
        pool[(c["recording_id"], c["subject"], c["t_ns"], c["end_t_ns"])] for c in clips
    ]
    assert scores != sorted(scores, reverse=True), "the order is the score order"
    assert run.review_clips("q01", "s2") != clips


def test_a_snapshot_reads_its_own_review_and_refuses_to_add_to_it(run):
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    run.finish_review("q01", "s1")
    view = run.view(run.pin(reason="finish"))
    with pytest.raises(ReadOnlyRunError):
        view.finish_review("q01", "s1")
    with pytest.raises(ReadOnlyRunError):
        view.take_out_clips(run.review_clips("q01", "s1")[0], "q01")
    assert view.review_precision("q01", "s1")["exact"]


def test_a_finalized_or_stale_run_refuses_to_finish(run):
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    run._set_stale(True)
    with pytest.raises(analysis_store.ResultsOutOfDateError):
        run.finish_review("q01", "s1")
    run._set_stale(False)
    run.finalize()
    with pytest.raises(ValueError, match="finalized"):
        run.finish_review("q01", "s1")
    assert not review_store.review_path(run._run_dir()).is_file()


def _audit_cli(run, *args) -> str:
    from click.testing import CliRunner

    from detail.cli import main

    result = CliRunner().invoke(
        main,
        [
            "analysis",
            "audit",
            "--experiment",
            str(run._experiment.dir),
            "--run",
            "an_01",
        ]
        + list(args),
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_the_cli_prints_the_review_before_the_earlier_audit(run):
    """Surface parity: `analysis audit` reads the same figures, and a behavior's figure only
    once every recording is exact; below the reviewed range it names the floor instead.
    """
    at = _cutoff_keeping(run, 6)
    run.set_cutoff("q01", at)
    run.take_out_clips(run.review_clips("q01", "s1")[:1], "q01")
    run.finish_review("q01", "s1")

    out = _audit_cli(run, "--behavior", "q01", "--by-recording")
    review, earlier = out.split("Earlier one-by-one audit:")
    assert "1 of 2 recordings reviewed" in review
    assert "every recording reviewed" not in review
    assert "1 removed" in review and "not reviewed" in review

    run.finish_review("q01", "s2")
    out = _audit_cli(run, "--behavior", "q01")
    assert "every recording reviewed" in out.split("Earlier one-by-one audit:")[0]

    run.set_cutoff("q01", _cutoff_keeping(run, 9))
    review = _audit_cli(run, "--behavior", "q01").split("Earlier one-by-one audit:")[0]
    assert "reviewed to" in review and "every recording reviewed" not in review


def test_a_run_whose_method_is_gone_is_refused_by_name_before_anything_is_written(run):
    """A run searched with a method this version no longer has (`dtw`, removed): its search
    cannot be read. Reading the review, and the CLI's `analysis audit`, refuse naming the
    method rather than raising the bare registry error the CLI printed as a traceback; a rename
    refuses before the state is written, where it used to save the name and then fail to
    rebuild. The negative case is the same run with its method back: it reads."""
    from detail.storage import query_runs as store

    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    assert run.review_by_recording("q01")  # reads with its own method
    state = run.state()
    state.method = "dtw"
    store.write_query_state(run._run_dir(), state)
    before = store.query_yaml_path(run._run_dir()).read_bytes()

    with pytest.raises(
        UserInputError, match="'dtw' method, which this version no longer has"
    ):
        type(run)(None, run._experiment, run.id).review_by_recording("q01")
    fresh = type(run)(None, run._experiment, run.id)
    with pytest.raises(UserInputError, match="no longer has"):
        fresh.name_behavior("q01", "climbing")
    assert (
        store.query_yaml_path(run._run_dir()).read_bytes() == before
    ), "written anyway"

    from click.testing import CliRunner

    from detail.cli import main

    result = CliRunner().invoke(
        main,
        [
            "analysis",
            "audit",
            "--experiment",
            str(run._experiment.dir),
            "--run",
            run.id,
        ],
    )
    assert result.exit_code == 1 and "Traceback" not in result.output
    assert "no longer has" in result.output
