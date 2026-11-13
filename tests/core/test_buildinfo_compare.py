"""The DETAIL on a cluster against the DETAIL running here: same, behind, ahead, or neither (§6).

**The class: a dependency on a copy of this code that nothing checked was this code.** A remote job
imports the app's modules *on the cluster*, so what matters is whether that checkout holds the
commit the app is running. It did not: Tillicum's predated the keypoints distill, and a resumed
detection run failed ten recordings with a bare ``No module named
'detail.pipeline.keypoint_dataset'``, found by reading the failure rather than by anything that
compared the two copies.

Like ``test_buildinfo``, these point ``buildinfo.source_root`` at a throwaway repository made with
real ``git``: the order of two commits is git's answer, and a stand-in for git would only test the
stand-in. The "remote" report is one this same code made about a commit of the same repository,
because that is exactly what a cluster prints (`buildinfo.report`, run there).
"""

import shlex

import pytest

from detail.core import buildinfo
from tests.core.test_buildinfo import (  # noqa: F401
    _commit,
    _git,
    _isolated_git,
    checkout,
)


def _report(repo, rev="HEAD", **overrides) -> dict:
    """What ``buildinfo.report()`` prints on a machine whose checkout is at ``rev``."""
    report = {
        "sha": _git(repo, "rev-parse", rev),
        "commit": _git(repo, "rev-parse", "--short", rev),
        "branch": "remote-detail",
        "date": "2026-10-06",
        "local_changes": False,
        "root": "/gpfs/home/someone/detail",
    }
    report.update(overrides)
    return report


def _compare(report) -> buildinfo.BuildComparison:
    return buildinfo.compare_with(report, target="tillicum", host="tillicum")


def test_the_same_commit_is_the_same_build(checkout):
    result = _compare(_report(checkout))

    assert result.state == buildinfo.SAME and result.matches
    assert "same build" in result.describe()


def test_a_cluster_on_an_ancestor_is_behind_by_the_commits_between(checkout):
    _commit(checkout, "c.txt", "third change (#3)")
    _commit(checkout, "d.txt", "fourth change (#4)")

    result = _compare(_report(checkout, "HEAD~2"))

    assert (result.state, result.commits) == (buildinfo.BEHIND, 2)
    assert not result.matches


def test_the_update_command_goes_to_the_clusters_own_checkout(checkout):
    _commit(checkout, "c.txt", "third change (#3)")
    folder = "/gpfs/home/some one/detail"  # a path with a space: quoted, not split

    result = _compare(_report(checkout, "HEAD~1", root=folder))

    command = result.update_command()
    assert command == f"ssh tillicum 'git -C {shlex.quote(folder)} pull --ff-only'"
    text = result.describe()
    assert "1 commit(s) behind this app" in text
    assert command in text
    assert (
        "conda" not in text
    )  # the ssh command runs git directly, not in an environment
    # Both short commits are named, so a person can see which two copies these are.
    assert result.remote_commit in text and result.local_commit in text


def test_a_cluster_on_another_branch_is_told_so_because_a_pull_will_not_fix_that(
    checkout,
):
    _commit(checkout, "c.txt", "third change (#3)")

    text = _compare(_report(checkout, "HEAD~1", branch="main")).describe()

    assert "It is on main; this app is on main." not in text  # only when they differ
    result = _compare(_report(checkout, "HEAD~1", branch="old-branch"))
    assert "It is on old-branch; this app is on main." in result.describe()


def test_local_changes_on_the_cluster_warn_that_a_pull_may_be_refused(checkout):
    _commit(checkout, "c.txt", "third change (#3)")

    text = _compare(_report(checkout, "HEAD~1", local_changes=True)).describe()

    assert "local changes" in text and "refused" in text


def test_a_cluster_on_a_descendant_is_ahead_and_the_app_is_what_to_update(checkout):
    _commit(checkout, "c.txt", "third change (#3)")
    ahead = _report(checkout, "HEAD")
    _git(
        checkout, "checkout", "-q", "HEAD~1"
    )  # this app is one commit behind the cluster

    result = _compare(ahead)

    assert (result.state, result.commits) == (buildinfo.AHEAD, 1)
    assert result.update_command() is None  # not the cluster that needs the pull
    assert "Update this app" in result.describe()


def test_a_side_branch_is_diverged(checkout):
    _git(checkout, "checkout", "-q", "-b", "side", "HEAD~1")
    _commit(checkout, "side.txt", "a change only the cluster has")
    side = _report(checkout)
    _git(checkout, "checkout", "-q", "main")

    result = _compare(side)

    assert result.state == buildinfo.DIVERGED
    assert result.update_command() is None


def test_a_commit_this_copy_has_never_seen_is_diverged_not_looked_up(checkout):
    result = _compare(_report(checkout, sha="f" * 40, commit="fffffff"))

    assert result.state == buildinfo.DIVERGED
    assert "fffffff" in result.describe()


@pytest.mark.parametrize(
    "report, reason",
    [
        ({"sha": None, "commit": None}, "not a git checkout"),
        ({}, "not a git checkout"),
    ],
    ids=["cluster-not-git", "empty-report"],
)
def test_a_cluster_whose_build_is_unknown_says_so(checkout, report, reason):
    result = _compare(report)

    assert result.state == buildinfo.UNKNOWN
    assert reason in result.describe()
    assert result.update_command() is None


def test_an_app_that_is_not_a_git_checkout_cannot_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(buildinfo, "source_root", lambda: tmp_path)

    result = _compare({"sha": "a" * 40, "commit": "aaaaaaa"})

    assert result.state == buildinfo.UNKNOWN
    assert "this copy of DETAIL is not a git checkout" in result.describe()


def test_report_is_what_a_cluster_prints_and_compare_reads(checkout):
    report = buildinfo.report()

    assert report["sha"] == _git(checkout, "rev-parse", "HEAD")
    assert report["commit"] == _git(checkout, "rev-parse", "--short", "HEAD")
    assert report["branch"] == "main" and report["local_changes"] is False
    assert report["root"] == str(checkout)
    # Plain data: it crosses ssh as JSON.
    import json

    assert json.loads(json.dumps(report)) == report
    assert _compare(report).state == buildinfo.SAME
