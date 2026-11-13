"""A shared ssh connection that is open and not answering (§6).

**The class: a wait with no bound on a thing that can stop answering.** Every remote command rides
one long-lived ``ControlMaster`` process, and that process can go silent while staying alive (a
stale socket after the computer slept or the network changed; the one that was caught had been up
twenty hours). Each command then sits on it at no CPU, prints nothing, and writes nothing to a log.
Eight recordings of an 80-recording detection stood on "uploading experiment files" for eight
minutes before anyone could tell it from a slow upload; ``ssh -O check`` was the only call that
said so, by not returning.

**It hid because every test gave ssh an answer.** The fakes return a result or raise; none of them
*never return*, which is the only thing this failure does. These make the check itself time out and
assert what the caller then sees: a refusal that names the cause and the fix, **not** the command
being started, and the same exit status (255) every caller already reads as "could not ask".

The shape to remember when adding a remote call: if the thing it waits on can stop answering
without closing, the call needs a bound, and the bound needs to say what to do.
"""

import subprocess
from pathlib import Path

import pytest

from detail.core.dataclasses import JobStatus
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState
from detail.executor.schedulers.slurm import SlurmScheduler

# Fixtures and helpers of the shepherd tests: the same experiment, fake scheduler and transport.
from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeScheduler,
    _install_scheduler,
    experiment,
)

REAL_STALL_CHECK = True  # tests/conftest.py's autouse stand-in stays out of this module


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _fresh_trust():
    ssh_transport._answered_at.clear()
    yield
    ssh_transport._answered_at.clear()


def _ssh(monkeypatch, *, check):
    """``subprocess.run`` where the ``-O check`` call is ``check`` (a result, or an exception to
    raise) and any other ssh call is recorded and answers ``ok``. Returns the call log.
    """
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[1:3] == ["-O", "check"]:
            if isinstance(check, Exception):
                raise check
            return check
        return _Result(0, stdout="ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _hang():
    return subprocess.TimeoutExpired(["ssh", "-O", "check", "tillicum"], 15)


def _commands(calls):
    return [c for c in calls if c[1:3] != ["-O", "check"]]


def test_a_connection_that_does_not_answer_is_refused_and_the_command_never_starts(
    monkeypatch,
):
    calls = _ssh(monkeypatch, check=_hang())

    with pytest.raises(ssh_transport.SshStalledError) as caught:
        ssh_transport.run_remote("tillicum", "mkdir -p /x")

    assert _commands(calls) == []  # waiting on it is the failure this replaces
    assert isinstance(caught.value, ssh_transport.RemoteCommandError)
    assert caught.value.returncode == ssh_transport.SSH_ERROR_EXIT
    message = str(caught.value)
    assert "not responding" in message
    assert "ssh tillicum true" in message  # says what to do, not just what is wrong


def test_a_poll_reads_a_stalled_connection_as_could_not_ask(monkeypatch):
    """``check=False`` callers read the exit status themselves: for them the stall is an ssh
    failure, the same word as a connection that never opened, so the reconnect wait takes it.
    """
    _ssh(monkeypatch, check=_hang())

    result = ssh_transport.run_remote("tillicum", "true", check=False)

    assert ssh_transport.ssh_failed(result)
    assert "not responding" in result.stderr
    assert ssh_transport.is_reachable("tillicum") is False


def test_the_scheduler_poll_reports_unreachable_not_a_failed_job(monkeypatch):
    _ssh(monkeypatch, check=_hang())
    target = type("T", (), {"host": "tillicum"})()

    assert SlurmScheduler().poll_state(target, "123") == JobState.UNREACHABLE
    assert SlurmScheduler().cancel(target, "123") is False


def test_an_answering_connection_runs_the_command_and_is_trusted_for_a_moment(
    monkeypatch,
):
    calls = _ssh(monkeypatch, check=_Result(0, stderr="Master running (pid=1)"))

    ssh_transport.run_remote("tillicum", "echo a")
    ssh_transport.run_remote("tillicum", "echo b")

    assert len(_commands(calls)) == 2
    assert len(calls) - len(_commands(calls)) == 1  # one check, not one per command


def test_no_master_running_is_an_answer_not_a_stall(monkeypatch):
    """Nothing to share (never opened, expired, or an ssh without connection sharing): the
    command connects on its own and works or fails fast, so the check must not stop it.
    """
    calls = _ssh(
        monkeypatch, check=_Result(255, stderr="Control socket connect: No such file")
    )

    ssh_transport.run_remote("tillicum", "echo a")

    assert len(_commands(calls)) == 1


def test_a_machine_with_no_ssh_lets_the_command_report_it(monkeypatch):
    calls = _ssh(monkeypatch, check=FileNotFoundError("ssh"))

    ssh_transport.run_remote("tillicum", "echo a")

    assert len(_commands(calls)) == 1


def test_a_stalled_connection_is_asked_again_not_remembered_as_fine(monkeypatch):
    calls = _ssh(monkeypatch, check=_hang())
    for _ in range(2):
        with pytest.raises(ssh_transport.SshStalledError):
            ssh_transport.run_remote("tillicum", "echo a")

    assert len(calls) == 2  # asked each time, since a stall may have been cleared


def test_the_copy_back_checks_before_it_copies(monkeypatch, tmp_path):
    calls = _ssh(monkeypatch, check=_hang())

    with pytest.raises(ssh_transport.SshStalledError):
        ssh_transport.copy_from_remote("tillicum", "/remote/run", tmp_path / "run")

    assert _commands(calls) == []


def test_a_command_over_its_own_connection_gives_up_on_a_dead_one(monkeypatch):
    """Where there is no shared connection to ask (a fresh connect per command), the bound is
    ssh's own: it gives up on an unanswered connect and on a quiet one."""
    calls = _ssh(monkeypatch, check=_Result(255))

    ssh_transport.run_remote("tillicum", "echo a")

    options = " ".join(_commands(calls)[0])
    assert "ConnectTimeout=20" in options
    assert "ServerAliveInterval=15" in options
    assert "ServerAliveCountMax=3" in options
    assert "BatchMode=yes" in options  # still never prompts


def test_an_upload_that_meets_a_stalled_connection_fails_the_recording_with_the_cause(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))

    def stalled(*args, **kwargs):
        raise ssh_transport.SshStalledError("tillicum")

    monkeypatch.setattr(ssh_transport, "copy_to_remote", stalled)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    error = rundir.read_envelope(run_dir)["error"]
    assert "not responding" in error.splitlines()[0]


# --------------------------------------------------------------------------- #
# the sweep: nothing else starts an ssh the check does not guard
# --------------------------------------------------------------------------- #

#: Files that may name ``"ssh"`` as a process to start, and why they need no check.
_MAY_START_SSH = {
    "executor/ssh_transport.py": "the transport: every other file goes through it",
    "api/remote.py": "`test_remote_target`'s own bounded probe (ConnectTimeout + a deadline)",
    "checks/commands.py": "builds a terminal command a person runs and watches",
}


def test_no_other_module_starts_an_ssh_of_its_own():
    """A remote call written beside the transport has no bound, so it can wait for good."""
    package = Path(ssh_transport.__file__).resolve().parents[1]
    offenders = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        if relative in _MAY_START_SSH:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if '"ssh"' in text and ("subprocess" in text or "Popen" in text):
            offenders.append(relative)
    assert offenders == [], (
        f"{offenders} start an ssh process directly; go through "
        "ssh_transport.run_remote (which checks the shared connection and bounds the "
        "connect), or add the file to _MAY_START_SSH with the reason it needs neither"
    )
