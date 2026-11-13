"""Clip checks on a clustering run (§22 *Clip checks*, ``storage/member_checks.py``): "does this
clip fit the cluster?", answered in the clip window before any label exists.

**What is pinned.** A clip's latest answer is its answer (a changed mind supersedes, nothing is
erased); the estimate is fits over fits-plus-doesn't with a Wilson interval, and ``unclear``
counts in neither; one node's checks never leak into another's; a finalized run refuses a
check as it refuses the audit; and the checks build nothing — the node cache and the labels
are byte-identical with or without them.
"""

from __future__ import annotations

import pytest

from detail.storage import member_checks
from detail.storage.member_checks import ClipKey


def _key(i: int, node: int = 7) -> ClipKey:
    return ClipKey(node, "recording_001", 0, i * 1_000_000_000, i * 1_000_000_000 + 500)


def test_the_latest_answer_is_the_clips_answer(tmp_path):
    member_checks.append_check(tmp_path, _key(0), "doesnt")
    member_checks.append_check(tmp_path, _key(0), "fits")
    member_checks.append_check(tmp_path, _key(1), "fits")
    member_checks.append_check(tmp_path, _key(2), "unclear")
    summary = member_checks.summary(tmp_path, 7)
    assert (summary.fits, summary.doesnt, summary.unclear) == (2, 0, 1)
    assert summary.answers[_key(0)] == "fits"
    assert summary.checked == 3
    rows = (tmp_path / member_checks.FILENAME).read_text(encoding="utf-8-sig")
    assert rows.count("doesnt") == 1, "the superseded answer stays in the file"


def test_the_estimate_leaves_unclear_out_and_carries_its_interval(tmp_path):
    for i, verdict in enumerate(["fits"] * 6 + ["doesnt"] * 2 + ["unclear"] * 3):
        member_checks.append_check(tmp_path, _key(i), verdict)
    summary = member_checks.summary(tmp_path, 7)
    assert summary.estimate == pytest.approx(6 / 8)
    low, high = summary.interval
    assert low < 6 / 8 < high and high <= 1.0


def test_one_nodes_checks_are_not_anothers(tmp_path):
    member_checks.append_check(tmp_path, _key(0, node=7), "fits")
    member_checks.append_check(tmp_path, _key(0, node=8), "doesnt")
    assert member_checks.summary(tmp_path, 7).fits == 1
    assert member_checks.summary(tmp_path, 8).doesnt == 1
    empty = member_checks.summary(tmp_path, 9)
    assert empty.checked == 0 and empty.estimate is None and empty.interval is None


def test_an_unknown_verdict_is_refused(tmp_path):
    with pytest.raises(ValueError, match="verdict must be one of"):
        member_checks.append_check(tmp_path, _key(0), "maybe")


def test_a_run_records_checks_and_refuses_them_once_finalized(fitted_temporal):
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    run.check_member(7, "recording_001", 0, 0, 500, "fits")
    assert run.member_checks(7).fits == 1
    with pytest.raises(ValueError, match="verdict must be one of"):
        run.check_member(7, "recording_001", 0, 0, 500, "nope")
    run.finalize()
    with pytest.raises(ValueError):
        run.check_member(7, "recording_001", 0, 0, 500, "doesnt")
    assert run.member_checks(7).fits == 1, "the refused answer was not written"


def test_checks_build_nothing(fitted_temporal):
    """Evidence, not input (CLAUDE.md, *Labels score a method*): the node cache and the run's
    labels are the same with checks as without."""
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    node = view.children(view.root)[0]
    before = (run.node_names(), view.summary(node))
    for verdict in ("fits", "doesnt", "unclear"):
        run.check_member(node, "recording_001", 0, 0, 500, verdict)
    after_run = experiment.temporal_clustering_run(run_id)
    after_view = after_run.node_view()
    assert (after_run.node_names(), after_view.summary(node)) == before
