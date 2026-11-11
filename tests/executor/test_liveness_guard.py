"""Not knowing whether a process is ours is answered one way, not two.

The class: **two ways of failing to get an answer, given opposite answers.**

``local._is_our_worker`` asks ``ps`` whether a pid's command line still carries this run's
directory — the PID-reuse guard that stops a recycled pid being force-killed, and the check
``live_process_pid`` uses to decide whether a worker is still *working*. It had three outcomes
and treated them as two:

===========================  =========================
``ps`` could not be launched  ``True``  — assume it's ours
``ps`` ran, path absent       ``False`` — genuinely not ours
``ps`` ran and **failed**     ``False`` — treated as an answer
===========================  =========================

The third is not an answer, and ``False`` is the dangerous direction to guess in. Callers read
it as "that process has finished": ``resume_training`` is a re-submit path with no reconcile
behind it, so a transient ``ps`` failure on a loaded box lets a second worker stack onto the same
weights and the same device behind a live first one. That is the incident #280 exists for,
reachable without anything being wrong with the process at all.

There was no timeout either, so a wedged ``ps`` hung whoever asked — and the askers are reconcile
and re-submit.

This was found while chasing intermittent ``retry_run`` failures. Those turned out to be
environmental (leftover processes from killed profiling scripts) and are not reproducible from
the code; this is the latent bug the search turned up, and it is exactly the shape that would
produce them.
"""

import base64
import os
import pathlib
import subprocess

import pytest

from detail.executor import local

RUN_DIR = pathlib.Path("/tmp/a/run_dir")


class _Result:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _command_line(text: str) -> str:
    """``text`` as the platform's lookup prints it: base64 of UTF-8 from PowerShell."""
    if os.name == "nt":
        return base64.b64encode(text.encode("utf-8")).decode("ascii") + "\n"
    return text + "\n"


def test_a_real_answer_is_believed(monkeypatch):
    """The negative case: when ``ps`` does answer, the answer stands. Without this the guard
    could be "always True" and every test above would still pass."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Result(0, _command_line(f"python -m worker {RUN_DIR}")),
    )
    assert local._is_our_worker(1, RUN_DIR)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Result(0, _command_line("python -m something_else")),
    )
    assert not local._is_our_worker(1, RUN_DIR)


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(OSError("no ps"), id="ps-cannot-be-launched"),
        pytest.param(subprocess.TimeoutExpired("ps", 10), id="ps-hangs"),
    ],
)
def test_an_unanswerable_check_assumes_the_process_is_ours(monkeypatch, failure):
    """Conservative, because the cost of the two mistakes is not symmetric: guessing "ours"
    refuses a retry somebody can repeat, guessing "not ours" starts a second process on a
    device the first one is still holding."""

    def _raise(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", _raise)
    assert local._is_our_worker(1, RUN_DIR)


def test_a_failed_ps_is_unanswerable_too_not_a_no(monkeypatch):
    """The bug. ``ps`` exiting non-zero was read as "the path is not in its command line",
    which is a claim it never made — its stdout was empty because it failed."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(1, ""))
    assert local._is_our_worker(1, RUN_DIR), (
        "a failed `ps` was read as a live worker being finished; that is how a re-submit "
        "stacks onto a process still holding the device"
    )


@pytest.mark.skipif(os.name != "nt", reason="the base64 envelope is the Windows lookup")
def test_an_unreadable_windows_answer_is_unanswerable_too(monkeypatch):
    """PowerShell answering with something that is not the base64 it was asked for is the
    same "can't check" as it failing outright — not a "no"."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(0, "not base64!\n"))
    assert local._is_our_worker(1, RUN_DIR)


def test_the_check_is_bounded(monkeypatch):
    """A wedged ``ps`` must not hang reconcile. Asserted on the call, because the only other
    way to test it is to wait for the hang."""
    seen = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return _Result(0, "")

    monkeypatch.setattr(subprocess, "run", _capture)
    local._is_our_worker(1, RUN_DIR)
    assert seen.get("timeout"), "the liveness check can block forever"
