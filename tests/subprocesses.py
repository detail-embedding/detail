"""Running a pytest inside a test, bounded so it cannot park the run that started it.

A nested ``pytest -n 2`` is a process tree: a controller and its xdist workers, which inherit
the controller's output pipes. ``subprocess.run(cmd, capture_output=True)`` with no timeout waits
for every holder of those pipes to let go, forever. Adding ``timeout=`` does not bound it on
Windows: on expiry ``subprocess.run`` kills the controller and then calls ``communicate()`` with
no timeout to collect the output, so a worker that outlived its controller still holds the pipe
and the wait goes on. Both shapes stood in ``tests/scripts/``.

The bound here is a failsafe, well under ``faulthandler_timeout`` (pyproject, 120s): a nested run
of a few trivial tests finishes in seconds, so reaching it means the tree is not coming back, and
the test fails naming the command instead of the watchdog killing the whole xdist worker. The
whole tree is killed first, because what outlives a kill of the controller alone is exactly what
held the pipe. The class and the guard are in ``tests/test_nested_runs_bounded.py``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

#: Slack for a nested run of a handful of tests. Under the 120s watchdog, so a hang is reported
#: by the test that started it.
TREE_TIMEOUT_S = 90.0
#: How long to wait for the pipes to close once the tree has been killed.
_DRAIN_S = 10.0


def _group_kwargs() -> dict:
    """``executor.local.new_process_group_kwargs``: the child leads its own group, so the tree
    can be killed as one, and killing it on Windows sends no Ctrl-C to this test runner.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(
    proc: subprocess.Popen,
) -> tuple[str | bytes | None, str | bytes | None]:
    """Kill ``proc`` and everything it started, reap it, and return the output it left."""
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=_DRAIN_S,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        return proc.communicate(timeout=_DRAIN_S)
    except subprocess.TimeoutExpired:
        # Something the tree kill missed still holds a pipe. Stop waiting for it rather than
        # close the pipe: on Windows ``communicate``'s reader threads are blocked in a read on
        # it, and closing it waits for them, which is this hang again. They are daemon threads.
        proc.kill()
        proc.wait(timeout=_DRAIN_S)
        return None, None


def start_tree(cmd: list[str], **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` in its own process group; finish it with :func:`wait_tree`."""
    return subprocess.Popen(cmd, **_group_kwargs(), **kwargs)


def wait_tree(
    proc: subprocess.Popen, timeout: float = TREE_TIMEOUT_S
) -> tuple[str | bytes | None, str | bytes | None]:
    """``proc.communicate()``, bounded: on timeout the whole tree is killed and it raises.

    The output the tree wrote before it stalled goes on the exception and to this test's stderr,
    which pytest prints with the failure: it is the one record of how far the nested run got.
    """
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        exc.output, exc.stderr = kill_tree(proc)
        for name, stream in (("stdout", exc.output), ("stderr", exc.stderr)):
            if stream:
                text = (
                    stream
                    if isinstance(stream, str)
                    else stream.decode(errors="replace")
                )
                sys.stderr.write(f"--- the timed-out tree's {name} ---\n{text}\n")
        raise


def run_tree(
    cmd: list[str], *, timeout: float = TREE_TIMEOUT_S, **kwargs
) -> subprocess.CompletedProcess:
    """``subprocess.run`` for a child that starts children of its own.

    Takes ``subprocess.run``'s ``capture_output``; everything else goes to ``Popen``.
    """
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    proc = start_tree(cmd, **kwargs)
    out, err = wait_tree(proc, timeout)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
