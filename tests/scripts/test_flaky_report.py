"""Every pytest run is logged, and the report counts a test's failures against the runs that ran it.

**The class: a test that fails now and then, with no record of how often.** Each failure is seen
by one agent in one session, re-run green, and forgotten, so nobody can say whether a test fails
once a month or one run in three, or under what load. ``suite_progress_plugin`` appends every
finished run to one log per clone and ``scripts/flaky_report.py`` reads it back.

What would make it lie, each pinned here:

- **A log per worktree, or per session.** Agents work in worktrees and sessions get a private
  ``TMPDIR`` and ``HOME``; a log under any of them splits the week into pieces nobody collects.
  ``test_every_worktree_of_a_clone_shares_one_log``.
- **Rewriting instead of appending.** ``test_each_run_adds_one_line``.
- **A rate with the wrong denominator.** Counting a failure against runs that never selected the
  test makes a flaky test look rare. ``test_a_rate_counts_only_runs_that_selected_the_test``.
- **An interrupted run counted as a run.** ``test_an_interrupted_run_is_left_out``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts import flaky_report

REPO = Path(__file__).resolve().parents[2]

_TESTS = """
def test_ok(): pass
def test_fails(): assert False
"""


def _run_pytest(tmp_path: Path, log: Path, *args: str) -> subprocess.CompletedProcess:
    if not (tmp_path / "test_sample.py").exists():
        (tmp_path / "test_sample.py").write_text(_TESTS, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "DETAIL_SUITE_PROGRESS_ACTIVE"}
    env["DETAIL_SUITE_PROGRESS"] = str(tmp_path / "progress")
    env["DETAIL_TEST_RUN_LOG"] = str(log)
    env["PYTHONPATH"] = str(REPO)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    cmd += ["-p", "scripts.suite_progress_plugin", *args, "test_sample.py"]
    return subprocess.run(
        cmd, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8"
    )


def _lines(log: Path) -> list[dict]:
    return [
        json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()
    ]


def test_each_run_adds_one_line(tmp_path):
    log = tmp_path / "runs.jsonl"
    _run_pytest(tmp_path, log)
    _run_pytest(tmp_path, log)
    runs = _lines(log)
    assert len(runs) == 2
    run = runs[0]
    assert next(iter(run)) == "schema_version"
    assert run["total"] == 2
    assert run["passed"] == 1
    assert run["failed"] == 1
    assert run["exitstatus"] == 1
    assert run["paths"] == ["test_sample.py"]
    assert [f["id"] for f in run["failures"]] == ["test_sample.py::test_fails"]
    assert set(run["failures"][0]) == {"id", "when", "error", "load", "other_runs"}
    assert run["failures"][0]["when"] == "call"
    assert run["failures"][0]["error"].startswith("assert False")
    assert run["host"]


def test_a_failure_logged_under_xdist_keeps_its_phase_and_crash_line(tmp_path):
    """A week of failures with no reason attached could only be guessed at: the log is read
    long after the scrollback is gone, and under xdist the controller sees each report only
    after it crossed the wire, so this is where the crash line has to survive."""
    log = tmp_path / "runs.jsonl"
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def broken():\n"
        "    raise RuntimeError('the fixture fell over')\n"
        "def test_setup_fails(broken): pass\n",
        encoding="utf-8",
    )
    _run_pytest(tmp_path, log, "-n", "2")
    (failure,) = _lines(log)[0]["failures"]
    assert failure["when"] == "setup"
    assert failure["error"] == "RuntimeError: the fixture fell over"


def test_every_worktree_of_a_clone_shares_one_log(tmp_path, monkeypatch):
    monkeypatch.delenv("DETAIL_TEST_RUN_LOG", raising=False)
    clone, worktree, plain = (
        tmp_path / "clone",
        tmp_path / "worktree",
        tmp_path / "plain",
    )
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", str(clone)], check=True)
    subprocess.run(
        [*git, "-C", str(clone), "commit", "-q", "--allow-empty", "-m", "x"],
        check=True,
    )
    subprocess.run(
        [*git, "-C", str(clone), "worktree", "add", "-q", str(worktree)], check=True
    )
    plain.mkdir()
    from_clone = flaky_report.log_path(clone)
    assert from_clone == flaky_report.log_path(worktree)
    assert from_clone.parent.resolve() == (clone / ".git").resolve()
    assert flaky_report.log_path(plain) is None
    branch, commit = flaky_report.git_head(worktree)
    assert branch == "worktree"
    assert commit is not None and int(commit, 16) >= 0


def _run(paths, failed=(), exitstatus=None, host="a", load=1.0):
    return {
        "host": host,
        "paths": paths,
        "exitstatus": exitstatus if exitstatus is not None else (1 if failed else 0),
        "failures": [{"id": f, "load": load, "other_runs": 0} for f in failed],
    }


def test_a_rate_counts_only_runs_that_selected_the_test():
    flaky = "tests/gui/test_pages.py::test_x"
    runs = [
        _run([], failed=[flaky]),
        _run(["tests/gui"]),
        _run(["tests/gui/test_pages.py::test_x"]),
        _run(["tests/storage"]),
        _run(["tests/storage/test_a.py"]),
    ]
    (row,) = flaky_report.report(runs)
    assert (row["failed"], row["runs"]) == (1, 3)


def test_an_interrupted_run_is_left_out(tmp_path):
    log = tmp_path / "runs.jsonl"
    for run in (_run([], failed=["t::a"]), _run([], exitstatus=2)):
        flaky_report.append(log, run)
    runs = flaky_report.read([log])
    assert len(runs) == 1


def test_several_logs_combine(tmp_path, capsys):
    logs = []
    for host, load in (("mac", 12.0), ("linux", 3.0)):
        log = tmp_path / f"{host}.jsonl"
        flaky_report.append(log, _run([], failed=["t::a"], host=host, load=load))
        flaky_report.append(log, _run([], host=host))
        logs.append(log)
    assert flaky_report.main([str(p) for p in logs]) == 0
    out = capsys.readouterr().out
    assert "4 runs from 2 host(s): linux, mac" in out
    assert "2 / 4" in out
    assert "3.0–12.0" in out


def test_a_row_says_how_many_branches_it_failed_on_and_its_commonest_crash():
    """Without these a week's log cannot tell a flake from a branch's work in progress: on one
    laptop, most failing tests had failed only on the branch that was editing them."""
    flake = "tests/gui/test_pages.py::test_x"
    runs = []
    for branch, error in (("a", "boom"), ("b", "boom"), ("c", "other")):
        run = _run([], failed=[flake])
        run["branch"] = branch
        run["failures"][0]["error"] = error
        runs.append(run)
    runs.append(_run([], failed=[flake]))  # logged before crash lines were kept
    (row,) = flaky_report.report(runs)
    assert row["branches"] == 3
    assert row["error"] == "boom"
    assert "boom" in flaky_report.render(runs, [row])
