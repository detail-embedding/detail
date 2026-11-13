"""A pytest run's progress reaches a file the watcher and the status line can draw.

**The class: a long run a person cannot see into.** A foreground command's output reaches Claude
Code only when it ends, so a four-minute suite looks exactly like a hang, and the user watching
has no way to tell which. ``scripts/suite_progress_plugin.py`` writes each run's counts to a
per-machine directory; ``scripts/suite_progress.py watch`` follows a run and prints a line per
milestone and per failure, which Claude relays into the conversation through the Monitor tool
(the Claude app), and ``statusline`` draws a live bar under the prompt (the terminal).

What would make it lie, each pinned here:

- **Counting in the wrong process.** Under xdist the controller's own session collects nothing
  and every worker sees a fraction; a plugin counting in the workers, or taking the total from
  the controller's session, reports a bar that never fills. ``test_an_xdist_run_counts_every_test``.
- **A run inside a run.** A test that starts pytest would show as a second run, or clobber the
  first run's file. ``test_a_run_started_inside_a_run_stays_quiet``.
- **A directory under ``TMPDIR``.** Sessions here give every run a private ``TMPDIR``, so the
  watcher would read a different directory from the one the run writes to — the failure
  ``with_suite_lock.py`` records for its lock. ``test_the_directory_does_not_follow_tmpdir``.
- **A dead run followed forever.** A killed run never writes ``finished``; the reader drops it by
  process, not by waiting. ``test_a_killed_run_is_dropped``.
- **Following the last run instead of this one.** ``test_the_watcher_skips_a_run_that_had_already_finished``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts import suite_progress
from tests.subprocesses import kill_tree, run_tree, start_tree, wait_tree

REPO = Path(__file__).resolve().parents[2]

_TESTS = """
import time, pytest
def test_a(): time.sleep({sleep})
def test_b(): time.sleep({sleep})
def test_c(): time.sleep({sleep})
def test_fails(): assert False
@pytest.mark.skip(reason="skipped on purpose")
def test_skipped(): pass
"""


def _child_env(progress: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "DETAIL_SUITE_PROGRESS_ACTIVE"}
    env["DETAIL_SUITE_PROGRESS"] = str(progress)
    env["PYTHONPATH"] = str(REPO)
    return env


def _run(tmp_path, *args, sleep=0.0, env=None, wait=True):
    (tmp_path / "test_sample.py").write_text(
        _TESTS.format(sleep=sleep), encoding="utf-8"
    )
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    cmd += ["-p", "scripts.suite_progress_plugin", *args, "test_sample.py"]
    env = env or _child_env(tmp_path / "progress")
    if not wait:
        return start_tree(cmd, cwd=tmp_path, env=env, stdout=subprocess.DEVNULL)
    return run_tree(
        cmd, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8"
    )


def _only_state(progress: Path) -> dict:
    files = list(progress.glob("*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text(encoding="utf-8-sig"))


def test_a_run_counts_every_outcome(tmp_path):
    _run(tmp_path)
    state = _only_state(tmp_path / "progress")
    assert (state["total"], state["done"]) == (5, 5)
    assert (state["passed"], state["failed"], state["skipped"]) == (3, 1, 1)
    assert state["failed_ids"] == ["test_sample.py::test_fails"]
    assert state["exitstatus"] == 1 and state["finished_at"] is not None
    assert state["label"] == tmp_path.name


def test_an_xdist_run_counts_every_test(tmp_path):
    """Two workers, one file, the whole total — counted in the controller."""
    _run(tmp_path, "-n", "2")
    state = _only_state(tmp_path / "progress")
    assert (state["total"], state["done"], state["passed"], state["failed"]) == (
        5,
        5,
        3,
        1,
    )


def test_a_run_started_inside_a_run_stays_quiet(tmp_path):
    env = _child_env(tmp_path / "progress")
    env["DETAIL_SUITE_PROGRESS_ACTIVE"] = "1"
    _run(tmp_path, env=env)
    assert not list((tmp_path / "progress").glob("*.json"))


def test_the_directory_does_not_follow_tmpdir(monkeypatch, tmp_path):
    monkeypatch.delenv("DETAIL_SUITE_PROGRESS", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # gettempdir() caches its first answer; without this the TMPDIR above would never be read,
    # and a directory that did follow it would still pass.
    monkeypatch.setattr(tempfile, "tempdir", None)
    assert tmp_path not in suite_progress.progress_dir().parents


def test_a_killed_run_is_dropped(monkeypatch, tmp_path):
    monkeypatch.setenv("DETAIL_SUITE_PROGRESS", str(tmp_path))
    dead = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    now = time.time()
    live = {
        "pid": os.getpid(),
        "started": now - 5,
        "total": 10,
        "done": 4,
        "finished_at": None,
    }
    suite_progress.write_state(live)
    suite_progress.write_state({**live, "pid": int(dead.stdout), "started": now - 9})
    suite_progress.write_state({**live, "pid": 1, "finished_at": now - 600})
    assert [s["pid"] for s in suite_progress.read_runs(now)] == [os.getpid()]
    assert len(list(tmp_path.glob("*.json"))) == 1
    # A live process does not keep a run on screen forever: on Windows liveness cannot be asked,
    # so age is the only way a killed run leaves.
    later = now + suite_progress.MAX_RUN_S + 1
    assert suite_progress.read_runs(later) == []


def test_the_line_reads_like_pytest(monkeypatch):
    state = {"label": "detail", "started": 100.0, "total": 200, "done": 50, "failed": 1}
    assert suite_progress.render(state, now=110.0) == (
        "detail  pytest [█████░░░░░░░░░░░░░░░]  25% 50/200  ✗1  ETA 30s"
    )
    done = {**state, "done": 200, "passed": 199, "finished_at": 340.0}
    assert (
        suite_progress.render(done, 400.0)
        == "detail  pytest ✗ 1 failed, 199 passed in 4m00s"
    )
    green = {**done, "failed": 0, "passed": 200}
    assert suite_progress.render(green, 400.0) == "detail  pytest ✓ 200 passed in 4m00s"


def test_the_status_line_is_silent_and_never_raises(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DETAIL_SUITE_PROGRESS", str(tmp_path))
    suite_progress.statusline()
    (tmp_path / "123.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "456.json").write_text('{"pid": "x"}', encoding="utf-8")
    suite_progress.statusline()
    assert capsys.readouterr().out == ""


def test_milestones_are_each_step_crossed_once():
    assert suite_progress.milestones_crossed(0, 9, 100, 10) == []
    assert suite_progress.milestones_crossed(0, 25, 100, 10) == [10, 20]
    assert suite_progress.milestones_crossed(25, 100, 100, 10) == [
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    assert suite_progress.milestones_crossed(0, 5, 0, 10) == []


def test_the_watcher_follows_a_run_to_its_end(tmp_path):
    """End to end: the Monitor's view of a real run — milestones, the failure, the result."""
    since = time.time()
    run = _run(tmp_path, sleep=1.0, wait=False)
    try:
        watcher = run_tree(
            [
                sys.executable,
                str(REPO / "scripts" / "suite_progress.py"),
                "watch",
                "--since",
                str(since),
                "--root",
                str(tmp_path),
                "--step",
                "20",
                "--poll",
                "0.1",
                "--every",
                "0",
            ],
            env=_child_env(tmp_path / "progress"),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        wait_tree(run)
    finally:
        kill_tree(run)
    lines = watcher.stdout.splitlines()
    assert watcher.returncode == 1, lines  # the run has a failing test
    assert "  ✗ test_sample.py::test_fails" in lines
    assert any("%" in line and "[" in line for line in lines), lines
    assert lines[-1].endswith(tuple(f"in {s}s" for s in range(1, 10))), lines
    assert "✗ 1 failed, 3 passed" in lines[-1]


def test_the_watcher_skips_a_run_that_had_already_finished(monkeypatch, tmp_path):
    """The run that ended a moment ago is the last one, not the one about to start. The first
    version latched onto it, printed its result and exited while the real suite was running.
    """
    monkeypatch.setenv("DETAIL_SUITE_PROGRESS", str(tmp_path / "progress"))
    now = time.time()
    finished = {
        "pid": os.getpid(),
        "root": str(tmp_path),
        "label": "x",
        "started": now - 3,
        "total": 8,
        "done": 8,
        "passed": 8,
        "failed": 0,
        "finished_at": now - 1,
    }
    suite_progress.write_state(finished)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "scripts" / "suite_progress.py"),
            "watch",
            "--root",
            str(tmp_path),
            "--poll",
            "0.1",
        ],
        env={**os.environ, "DETAIL_SUITE_PROGRESS": str(tmp_path / "progress")},
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    time.sleep(1.0)
    still_waiting = proc.poll() is None
    proc.kill()
    out = proc.communicate()[0]
    assert still_waiting, out
    assert "8 passed" not in out


def test_a_short_run_posts_its_failures_and_result_and_nothing_else(tmp_path):
    """Every run is watched now, not only the suite: a run of a few seconds must not post ten
    milestones in a row. The failure and the result still arrive."""
    since = time.time()
    run = _run(tmp_path, sleep=0.3, wait=False)
    try:
        watcher = run_tree(
            [
                sys.executable,
                str(REPO / "scripts" / "suite_progress.py"),
                "watch",
                "--since",
                str(since),
                "--root",
                str(tmp_path),
                "--poll",
                "0.1",
            ],
            env=_child_env(tmp_path / "progress"),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        wait_tree(run)
    finally:
        kill_tree(run)
    lines = [
        line
        for line in watcher.stdout.splitlines()
        if line != "waiting for pytest to start"
    ]
    assert lines[0] == "  ✗ test_sample.py::test_fails", lines
    assert len(lines) == 2 and "✗ 1 failed, 3 passed" in lines[1], lines


def test_the_bar_prints_on_a_console_that_is_not_utf8(monkeypatch, tmp_path):
    """Windows consoles default to encodings like cp1252, which have no ``█`` or ``✓``. Without
    forcing UTF-8 the status line swallowed the error and showed nothing."""
    monkeypatch.setenv("DETAIL_SUITE_PROGRESS", str(tmp_path))
    now = time.time()
    suite_progress.write_state(
        {
            "pid": os.getpid(),
            "label": "d",
            "started": now - 5,
            "total": 10,
            "done": 4,
            "finished_at": None,
        }
    )
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert suite_progress.main(["statusline"]) == 0
    sys.stdout.flush()
    assert "█" in raw.getvalue().decode("utf-8")
