"""The machine-wide suite lock serialises runs across sessions, and a waiter is told who it waits on.

**The class: a lock that cannot collide is a lock that locks nothing.** The pre-push hook took its
lock at ``${TMPDIR:-/tmp}/detail-suite.lock``. Every session on the shared box gives its runs a
private ``TMPDIR``, so each run took a different file, and the lock serialised nothing between the
sessions it existed for. Nothing failed: each run held a lock, and every run was green. The only
signature was two full suites and a benchmark on the machine at once, which throws the
benchmark away and publishes a test tier as several times slower than it is.

So the path is fixed per machine, and ``test_the_lock_does_not_follow_tmpdir`` pins that. The
other tests pin what makes the lock usable by hand: a waiter is not silent, the command's exit
code is the script's, and the holder note does not outlive the run.

**The second class: a lock that needs a cleanup step to release deadlocks when a run is killed.**
Gates get killed --- ``tmux kill-session`` on a run started against a tree already known to fail,
Ctrl-C, an OOM kill --- and a mutex that releases by deleting its own file skips that delete on a
kill, so every later run on the box waits forever on a holder that does not exist. ``flock(2)``
lives on the descriptor and the kernel drops it when the holder dies, which is why nothing here
times a lock out. That was the one claim in the script's docstring with no test behind it, and it
is the claim that decides whether a killed gate costs a keystroke or the whole machine;
``test_a_killed_run_releases_the_lock`` fails against the delete-to-release mutex and passes
against this one.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "scripts" / "with_suite_lock.py"


def _module():
    spec = importlib.util.spec_from_file_location("with_suite_lock", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(lock: Path, who: str, code: str, *, capture: bool = True) -> subprocess.Popen:
    """Start the script holding ``lock``. ``capture=False`` for a run that will be killed.

    A killed parent cannot be drained with ``communicate``: the command it spawned inherits the
    pipes and outlives it, so the call would block on a process the test means to leave running.
    Killing without draining leaks the pipe instead, which ``filterwarnings = error`` turns into a
    failure. Neither is a lock problem, so a run nobody reads sends its output nowhere.
    """
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "DETAIL_SUITE_LOCK_NOTICE_S": "0.3",
    }
    sink = subprocess.PIPE if capture else subprocess.DEVNULL
    return subprocess.Popen(
        [sys.executable, str(_SCRIPT), "--lock", str(lock), "--who", who, "--",
         sys.executable, "-c", code],
        env=env,
        stdout=sink,
        stderr=sink,
        text=True,
        encoding="utf-8",
    )  # fmt: skip


def test_a_second_run_waits_and_names_who_it_is_waiting_on(tmp_path):
    lock = tmp_path / "suite.lock"
    first = _run(lock, "first-session", "import time; time.sleep(2.5)")
    try:
        deadline = time.monotonic() + 10
        holder = lock.with_name(lock.name + ".holder")
        while not holder.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert holder.exists(), "the first run never took the lock"

        started = time.monotonic()
        second = _run(lock, "second-session", "pass")
        _, err = second.communicate(timeout=30)
        waited = time.monotonic() - started
    finally:
        first.communicate(timeout=30)

    assert second.returncode == 0
    assert waited > 1.0, f"the second run finished in {waited:.1f}s, so it never waited"
    assert (
        "waiting" in err and "first-session" in err
    ), f"a waiter must say who it is waiting on, so a wait is never silent: {err!r}"
    assert not holder.exists(), "the holder note outlived the run"


def test_the_commands_exit_code_is_the_scripts(tmp_path):
    run = _run(tmp_path / "suite.lock", "t", "raise SystemExit(3)")
    run.communicate(timeout=30)
    assert run.returncode == 3


def test_the_lock_does_not_follow_tmpdir(tmp_path, monkeypatch):
    """The regression: a TMPDIR-relative lock is a different file for every session."""
    monkeypatch.delenv("DETAIL_SUITE_LOCK", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    # `tempfile` caches the temp dir on first use, so without this a TMPDIR-following lock would
    # still resolve to whatever this process cached earlier, and the check would pass against
    # the very bug it exists for. It did, until this line.
    monkeypatch.setattr(tempfile, "tempdir", None)
    path = _module().lock_path()
    # No `os.name` skip: one stood here, and it is why main's Windows lock followed TEMP unseen.
    assert tmp_path not in path.parents, (
        f"the lock is at {path}, under this run's TMPDIR, so another session with its own "
        "TMPDIR takes a different lock and nothing is serialised"
    )


def test_the_override_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("DETAIL_SUITE_LOCK", str(tmp_path / "custom.lock"))
    assert _module().lock_path() == tmp_path / "custom.lock"


def test_a_killed_run_releases_the_lock(tmp_path):
    """The class: a lock that needs a cleanup step to release deadlocks when a run is killed.

    A gate is killed for ordinary reasons --- ``tmux kill-session`` on a run started against a
    tree already known to fail, Ctrl-C, an OOM kill. Under a mutex built from "create the file
    exclusively, delete it on the way out", the kill skips the delete and every later run on the
    box waits forever on a holder that no longer exists, until somebody works out which file to
    ``rm``. ``flock(2)`` lives on the open descriptor, so the kernel drops it when the holder dies
    for any reason. This pins that, because it is the property that makes the lock safe to kill
    and it is invisible for as long as nothing is ever killed.

    The command outlives the script here on purpose: ``subprocess`` closes inherited descriptors,
    so the command never holds the lock itself. If it did, killing the script would release
    nothing while the command ran on --- and the box would be locked by a process that no longer
    has a lock holder to name. It is left to exit on its own; re-parented to init, it is not
    reachable by pid from here without a process scan.
    """
    lock = tmp_path / "suite.lock"
    holder_note = lock.with_name(lock.name + ".holder")
    victim = _run(lock, "killed-session", "import time; time.sleep(10)", capture=False)
    try:
        deadline = time.monotonic() + 10
        while not holder_note.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (
            holder_note.exists()
        ), "the run never took the lock, so there is nothing to kill"

        victim.kill()
        victim.wait(timeout=30)
        assert holder_note.exists(), (
            "the killed run still cleaned up after itself, so this no longer exercises the case "
            "where nothing got to run"
        )

        after = _run(lock, "after-the-kill", "pass")
        # A generous timeout, not a speed assertion: the failure being pinned is "waits forever",
        # and a wall-clock bound on the box that also runs the suite would be a flake.
        _, err = after.communicate(timeout=60)
    finally:
        victim.kill()

    assert after.returncode == 0, (
        "a run could not take the lock after the holder was killed, so the kill left a stale "
        f"lock that nothing times out: {err!r}"
    )
    assert "killed-session" not in err, (
        "the holder note left by the killed run was reported as the current holder; a waiter "
        f"must only ever name a live one: {err!r}"
    )
