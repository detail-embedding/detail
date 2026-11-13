"""The hang watchdog stays armed after a test fails (``scripts/watchdog_plugin.py``).

**The class: a watchdog that stands down at the moment it is most needed.** ``faulthandler_timeout``
(pyproject) arms a C-level timer around each test's whole protocol; on expiry it dumps every
thread's stack and exits the worker, so a hang becomes a named crash that xdist reports and runs
past. pytest's own faulthandler plugin cancels that timer in ``pytest_exception_interact``, which
fires on *every* failed setup or call, not only under ``--pdb``. So everything a failed test does
after failing ran unwatched: its fixture teardown, and, when it was the last test its worker ran,
the teardown of every module- and session-scoped fixture. A test that failed *because* a worker,
thread or child process misbehaved is exactly the one whose teardown then waits on it.

**How it hid.** A hang there leaves no dump, no crash and no failure line, so nothing points at
it. It was found on 2026-11-06 while looking for why two gate runs on the Windows box never
finished, as the first explanation for "the watchdog said nothing". It was not that one: that
gate's log did hold a dump (``tests/test_nested_runs_bounded.py`` has it). No hang has been traced
to this gap. It is closed because a hang it hides takes the whole run and the machine lock with it,
and the record it would have left is the only thing that names the test.

**The rule.** After pytest cancels the timer for a failure, it is armed again for the rest of that
test, with the same timeout and the same exit — except under ``--pdb``, the one case the cancel
was written for. ``test_a_failed_test_that_hangs_in_teardown_is_dumped_and_the_run_ends`` is the
guard; ``test_pytest_alone_leaves_that_teardown_unwatched`` pins pytest's behaviour, so the day
pytest stops cancelling, it fails and the plugin can be retired rather than left excusing nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.subprocesses import run_tree

REPO = Path(__file__).resolve().parents[2]

#: The watchdog's deadline in these runs, and how long the teardown would hang without it.
_TIMEOUT_S = 2
_HANG_S = 12

_INI = f"""
[pytest]
faulthandler_timeout = {_TIMEOUT_S}
faulthandler_exit_on_timeout = true
"""

_TESTS = f"""
import time, pytest

@pytest.fixture
def stuck_teardown():
    yield
    time.sleep({_HANG_S})  # a worker the failed test was waiting on, never coming back

def test_fails_then_hangs(stuck_teardown):
    assert False
"""


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    (tmp_path / "pytest.ini").write_text(_INI, encoding="utf-8")
    (tmp_path / "test_sample.py").write_text(_TESTS, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "DETAIL_SUITE_PROGRESS_ACTIVE"}
    env["PYTHONPATH"] = str(REPO)
    return run_tree(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _dumped(proc: subprocess.CompletedProcess) -> bool:
    out = proc.stdout + proc.stderr
    return f"Timeout (0:00:0{_TIMEOUT_S})!" in out and "stuck_teardown" in out


def test_a_failed_test_that_hangs_in_teardown_is_dumped_and_the_run_ends(tmp_path):
    """Under xdist, as the gate runs: the worker is dumped and exits, and the run reports it."""
    proc = _run(tmp_path, "-p", "scripts.watchdog_plugin", "-n", "1")
    assert _dumped(proc), proc.stdout + proc.stderr
    assert (
        "crashed while running 'test_sample.py::test_fails_then_hangs'" in proc.stdout
    )


def test_pytest_alone_leaves_that_teardown_unwatched(tmp_path):
    """Without the plugin the teardown runs its full length with no dump. If this fails, pytest no
    longer cancels the timer on a failure, and ``scripts/watchdog_plugin.py`` can go."""
    proc = _run(tmp_path)
    assert not _dumped(proc), proc.stdout + proc.stderr
    assert "1 failed" in proc.stdout
