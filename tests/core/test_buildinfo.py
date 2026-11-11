"""The build facts and history are read from a real git checkout, never kept by hand.

Each test points ``buildinfo.source_root`` at a throwaway repository made with real ``git`` —
the facts are git's answers, so a stand-in for git would only test the stand-in. The update
check fetches from a second local repository playing ``origin``, so no network is touched.
"""

import os
import shutil
import subprocess

import pytest

from detail import __version__
from detail.core import buildinfo

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()


def _commit(repo, name: str, subject: str) -> None:
    (repo / name).write_text(subject, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", subject)


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch):
    # The developer's own git config (signing, hooks, a default branch) must not reach a repo
    # the test makes; the identity is what `commit` needs and nothing else.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key, value in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }.items():
        monkeypatch.setenv(key, value)
    buildinfo.build_info.cache_clear()
    yield
    buildinfo.build_info.cache_clear()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "a.txt", "first change (#1)")
    _commit(repo, "b.txt", "second change (#2)")
    monkeypatch.setattr(buildinfo, "source_root", lambda: repo)
    return repo


def test_build_info_names_the_checked_out_commit(checkout):
    info = buildinfo.build_info()
    assert info.version == __version__
    assert info.commit == _git(checkout, "rev-parse", "--short", "HEAD")
    assert info.date == _git(checkout, "log", "-1", "--format=%cs")
    assert info.commit_count == 2
    assert info.branch == "main"
    assert info.local_changes is False
    assert info.label() == f"v{__version__} · {info.date} · {info.commit}"


def test_an_edited_tracked_file_marks_the_build_as_locally_changed(checkout):
    (checkout / "a.txt").write_text("edited", encoding="utf-8")
    info = buildinfo.build_info()
    assert info.local_changes is True
    assert info.label().endswith(f"{info.commit}+")


def test_history_is_the_first_parent_line_newest_first(checkout):
    _git(checkout, "checkout", "-q", "-b", "feature")
    _commit(checkout, "c.txt", "side commit")
    _git(checkout, "checkout", "-q", "main")
    _git(checkout, "merge", "-q", "--no-ff", "feature", "-m", "merge the feature (#3)")
    subjects = [entry.subject for entry in buildinfo.history()]
    # The side commit is reached only through the merge's second parent, so it is not listed.
    assert subjects == [
        "merge the feature (#3)",
        "second change (#2)",
        "first change (#1)",
    ]
    assert [entry.subject for entry in buildinfo.history(limit=1)] == [subjects[0]]


def test_outside_a_git_checkout_the_facts_are_unknown_not_an_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(buildinfo, "source_root", lambda: tmp_path)
    info = buildinfo.build_info()
    assert info.commit is None
    assert info.label() == f"v{__version__}"
    assert buildinfo.history() == []
    assert buildinfo.head_sha() is None
    with pytest.raises(buildinfo.UpdateCheckError, match="not a git checkout"):
        buildinfo.check_for_updates()


def test_update_check_counts_commits_behind_and_ahead_of_origin_main(
    tmp_path, monkeypatch
):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit(origin, "a.txt", "first change (#1)")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    monkeypatch.setattr(buildinfo, "source_root", lambda: clone)

    status = buildinfo.check_for_updates()
    assert (status.behind, status.ahead) == (0, 0)

    _commit(origin, "b.txt", "second change (#2)")
    _commit(origin, "c.txt", "third change (#3)")
    _commit(clone, "local.txt", "an unpushed local commit")
    status = buildinfo.check_for_updates()
    assert (status.behind, status.ahead) == (2, 1)
    assert status.upstream == "origin/main"
    assert status.upstream_date == _git(origin, "log", "-1", "--format=%cs")


def test_update_check_without_the_remote_raises_an_actionable_error(checkout):
    with pytest.raises(buildinfo.UpdateCheckError, match="'origin' remote"):
        buildinfo.check_for_updates()


def test_describe_names_commit_branch_date_and_local_changes():
    info = buildinfo.BuildInfo(
        version="0.1.0",
        commit="abc1234",
        date="2026-09-20",
        commit_count=832,
        branch="main",
    )
    assert info.describe() == "detail v0.1.0 — abc1234 on main, 2026-09-20, commit #832"
    edited = buildinfo.BuildInfo(
        version="0.1.0", commit="abc1234", date="2026-09-20", local_changes=True
    )
    assert (
        edited.describe() == "detail v0.1.0 — abc1234, 2026-09-20 (with local changes)"
    )
    assert "not a git checkout" in buildinfo.BuildInfo(version="0.1.0").describe()


def test_update_status_says_how_far_behind_and_how_to_update():
    behind = buildinfo.UpdateStatus(behind=3, ahead=0, upstream_date="2026-09-22")
    assert behind.describe() == (
        "3 commits behind origin/main (latest 2026-09-22). "
        "To update: git pull, then reinstall."
    )
    one = buildinfo.UpdateStatus(behind=1, ahead=0, upstream_date="2026-09-22")
    assert one.describe().startswith("1 commit behind")
    current = buildinfo.UpdateStatus(behind=0, ahead=2, upstream_date="2026-09-20")
    assert current.describe() == (
        "Up to date with origin/main. This checkout also has 2 commit(s) not on it."
    )


def test_shorter_labels_drop_the_commit_then_the_version_and_keep_the_date():
    info = buildinfo.BuildInfo(
        version="0.1.0", commit="abc1234", date="2026-09-20", local_changes=True
    )
    assert info.labels() == [
        "v0.1.0 · 2026-09-20 · abc1234+",
        "v0.1.0 · 2026-09-20",
        "2026-09-20",
    ]
    assert info.label() == info.labels()[0]
    assert buildinfo.BuildInfo(version="0.1.0").labels() == ["v0.1.0"]
