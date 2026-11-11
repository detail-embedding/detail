"""Kill-capable, reopen-safe run cancellation (``cancel_run_dir``) + PID record.

These cover the bug behind the GUI cancel feature: a run launched then orphaned
(GUI quit) must still be stoppable by its directory alone — recorded PID + flag +
terminal status, no live handle.
"""

import json
import subprocess
import sys
import time

from detail.executor import local as local_mod
from detail.executor import rundir
from detail.executor.local import _alive, _is_our_worker, cancel_run_dir


def _envelope(run_dir, **extra):
    payload = {"schema_version": 1, "status": "running", "error": None, **extra}
    (run_dir / "job.json").write_text(json.dumps(payload), encoding="utf-8")


def test_set_and_read_pid(tmp_path):
    _envelope(tmp_path)
    rundir.set_pid(tmp_path, 4242)
    assert rundir.read_pid(tmp_path) == 4242


def test_read_pid_none_when_absent(tmp_path):
    _envelope(tmp_path)
    assert rundir.read_pid(tmp_path) is None


def test_cancel_kills_live_worker_and_sets_status(tmp_path):
    # a worker that ignores the cancel flag (like a blocking training call); it
    # carries the run dir in its argv so the PID-reuse identity guard recognizes it
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(tmp_path)],
        **local_mod.new_process_group_kwargs(),
    )
    _envelope(tmp_path, pid=proc.pid)

    cancel_run_dir(tmp_path, grace_s=0.0)  # no grace → force-kill the group

    # the worker terminated (wait() reaps it; raises TimeoutExpired if still alive).
    # We check via the Popen, not _alive: this process is the worker's parent, so a
    # killed-but-unreaped child is a zombie that os.kill(pid, 0) still reports alive.
    proc.wait(timeout=5)
    assert proc.returncode is not None
    assert rundir.read_status(tmp_path).value == "cancelled"
    assert rundir.cancel_requested(tmp_path)


def test_cancel_no_pid_just_marks_cancelled(tmp_path):
    _envelope(tmp_path)  # no pid (worker already gone)
    cancel_run_dir(tmp_path, grace_s=0.0)
    assert rundir.read_status(tmp_path).value == "cancelled"


def test_cancel_preserves_a_clean_terminal_status(tmp_path):
    # a cooperative runner that already stopped on the flag + wrote 'partial' must
    # not be clobbered to 'cancelled'
    _envelope(tmp_path, status="partial")
    cancel_run_dir(tmp_path, grace_s=0.0)
    assert rundir.read_status(tmp_path).value == "partial"
    assert rundir.cancel_requested(tmp_path)  # flag still recorded


def test_cancel_idempotent_on_dead_pid(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()  # already dead
    _envelope(tmp_path, pid=proc.pid)
    cancel_run_dir(tmp_path, grace_s=0.0)  # must not raise on a dead pid
    assert rundir.read_status(tmp_path).value == "cancelled"


def test_kill_tree_windows_uses_taskkill(monkeypatch):
    """The Windows branch shells out to taskkill and must never reference
    ``signal.SIGKILL`` (absent on Windows) — ``force`` selects ``/F``. Forces the
    nt branch so it runs on Linux CI too."""
    calls = []
    monkeypatch.setattr(local_mod.os, "name", "nt")
    monkeypatch.setattr(local_mod.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    local_mod._kill_tree(4242, force=False)
    local_mod._kill_tree(4242, force=True)

    assert calls == [
        ["taskkill", "/T", "/PID", "4242"],
        ["taskkill", "/F", "/T", "/PID", "4242"],
    ]


def test_cancel_does_not_kill_a_recycled_pid(tmp_path):
    # the recorded pid was reused by an unrelated process (its command line does NOT
    # carry our run dir) — cancel must NOT kill it, only mark the run cancelled
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        **local_mod.new_process_group_kwargs(),
    )
    try:
        assert not _is_our_worker(other.pid, tmp_path)
        _envelope(tmp_path, pid=other.pid)
        cancel_run_dir(tmp_path, grace_s=0.0)
        assert other.poll() is None  # the innocent process is untouched
        assert rundir.read_status(tmp_path).value == "cancelled"
    finally:
        other.kill()
        other.wait(timeout=5)
