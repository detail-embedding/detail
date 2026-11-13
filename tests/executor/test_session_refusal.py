"""A full ssh connection is not a failed job (§6).

**The class: a refusal that means "try again" recorded as a verdict.** Tillicum lets one ssh
connection carry ten sessions at once (OpenSSH's ``MaxSessions``), and DETAIL's ``max_concurrent_jobs``
jobs share one connection. Each job runs ``mkdir``, ``rsync``, ``sbatch``, a model check and
a poll through it; one session more than the host allows is refused with "Session open refused by
peer". Three recordings of an 80-recording run were lost to a ``mkdir`` that would have worked a
second later, and a fourth, a detection that **finished**, was recorded as failed because the
refusal landed on the read of its result file and "could not ask" was read as "nothing written".
The poll loop had already learned that being unable to ask is not an answer
(`remote_shepherd._Reconnect`); the commands that *do* something, and the decision made when a job
ends, had not.

**It hid because every test gave the fake transport a definite answer.** No test made ssh refuse a
session, so a command that fails on the first refusal and one that retries were both green. The
tests here make the host refuse, count the tries, and check the three things that must not change:
a real login failure still fails at once (retrying it would hang a run on a Duo prompt nobody
sees), a refused command is never run twice (a refusal means it never ran, which is what makes
retrying ``sbatch`` safe), and an answer the cluster *did* give is not second-guessed.
"""

import subprocess

import pytest

from detail.executor import ssh_transport

REFUSED = (
    "mux_client_request_session: session request failed: Session open refused by peer\n"
    "sagolden@tillicum.hyak.uw.edu: Permission denied (gssapi-keyex,gssapi-with-mic,"
    "keyboard-interactive)."
)
LOGIN_FAILED = (
    "sagolden@tillicum.hyak.uw.edu: Permission denied "
    "(gssapi-keyex,gssapi-with-mic,keyboard-interactive)."
)


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def waits(monkeypatch):
    """Run the backoff without waiting, and keep what it would have slept."""
    slept: list[float] = []
    monkeypatch.setattr(ssh_transport, "_sleep", slept.append)
    return slept


def _script(monkeypatch, *results):
    """Make ``subprocess.run`` answer with ``results`` in turn, counting the calls."""
    calls = {"n": 0}
    answers = list(results)

    def fake_run(argv, **kwargs):
        calls["n"] += 1
        return answers.pop(0) if len(answers) > 1 else answers[0]

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_a_command_refused_a_session_is_tried_again_and_succeeds(monkeypatch, waits):
    calls = _script(
        monkeypatch,
        _Result(255, stderr=REFUSED),
        _Result(255, stderr=REFUSED),
        _Result(0, stdout="done\n"),
    )
    result = ssh_transport.run_remote("tillicum", "mkdir -p /x")
    assert result.stdout == "done\n"
    assert calls["n"] == 3
    assert len(waits) == 2 and waits[1] > 0  # it backed off between tries


def test_the_backoff_grows_and_is_capped(monkeypatch, waits):
    _script(monkeypatch, _Result(255, stderr=REFUSED))
    with pytest.raises(ssh_transport.RemoteCommandError):
        ssh_transport.run_remote("tillicum", "mkdir -p /x")
    assert len(waits) == ssh_transport.REFUSED_RETRIES
    assert max(waits) <= ssh_transport.REFUSED_MAX_WAIT_S * 1.5  # jitter allows 1.5x
    assert waits[-1] > waits[0]


def test_a_real_login_failure_is_not_retried(monkeypatch, waits):
    """Also exit 255 and also "Permission denied", but no refusal line: the session was never
    the problem (no live shared connection, an expired Duo login). Retrying would only hold the
    run up for the whole backoff before failing with the same words."""
    calls = _script(monkeypatch, _Result(255, stderr=LOGIN_FAILED))
    with pytest.raises(ssh_transport.RemoteCommandError, match="Permission denied"):
        ssh_transport.run_remote("tillicum", "mkdir -p /x")
    assert calls["n"] == 1 and waits == []


def test_a_remote_commands_own_failure_is_not_retried(monkeypatch, waits):
    calls = _script(monkeypatch, _Result(2, stderr="mkdir: cannot create directory"))
    with pytest.raises(ssh_transport.RemoteCommandError, match="cannot create"):
        ssh_transport.run_remote("tillicum", "mkdir -p /x")
    assert calls["n"] == 1 and waits == []


def test_after_the_last_try_the_refusal_is_raised_not_swallowed(monkeypatch, waits):
    calls = _script(monkeypatch, _Result(255, stderr=REFUSED))
    with pytest.raises(ssh_transport.RemoteCommandError, match="refused by peer"):
        ssh_transport.run_remote("tillicum", "mkdir -p /x", refused_retries=3)
    assert calls["n"] == 4  # the first try and three more


def test_a_caller_that_reads_the_status_itself_gets_the_refusal_after_its_budget(
    monkeypatch, waits
):
    """A poll's ask passes ``check=False`` and a small budget: it must come back with the 255 for
    its own "could not ask" handling, quickly, not wait out the long budget."""
    calls = _script(monkeypatch, _Result(255, stderr=REFUSED))
    result = ssh_transport.run_remote(
        "tillicum",
        "squeue",
        check=False,
        refused_retries=ssh_transport.ASK_REFUSED_RETRIES,
    )
    assert ssh_transport.ssh_failed(result)
    assert calls["n"] == ssh_transport.ASK_REFUSED_RETRIES + 1


def test_the_asks_the_poll_makes_use_the_small_budget(monkeypatch, waits):
    """Every ask on the poll path must pass the short budget, or a full connection stalls the
    cancel check for minutes. Read off what the real call sites pass."""
    from detail.configs.remote import RemoteTargetConfig
    from detail.executor.schedulers.slurm import SlurmScheduler

    seen: list[int] = []

    def fake_run_remote(host, command, **kwargs):
        seen.append(kwargs.get("refused_retries"))
        return _Result(0, stdout="RUNNING\n")

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run_remote)
    target = RemoteTargetConfig(name="t", host="t", remote_workdir="/w")
    SlurmScheduler().poll_state(target, "123")
    SlurmScheduler().cancel(target, "123")
    assert seen == [ssh_transport.ASK_REFUSED_RETRIES] * 2


def test_sbatch_is_retried_and_submitted_once(monkeypatch, waits):
    """A refused session means the command never ran, so retrying ``sbatch`` cannot double-submit:
    exactly one call returned a job id, and that is the one parsed."""
    from detail.configs.remote import RemoteTargetConfig
    from detail.executor.schedulers.slurm import SlurmScheduler

    answers = iter(
        [
            _Result(0),  # write_remote_file
            _Result(255, stderr=REFUSED),  # sbatch, refused: never ran
            _Result(0, stdout="Submitted batch job 347123\n"),  # sbatch, ran
        ]
    )
    commands: list[str] = []

    def fake_run(argv, **kwargs):
        commands.append(argv[-1])
        return next(answers)

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = RemoteTargetConfig(name="t", host="t", remote_workdir="/w")
    job_id = SlurmScheduler().submit_script(target, "#!/bin/bash\n", "/w/run")
    assert job_id == "347123"
    assert sum(c.startswith("sbatch") for c in commands) == 2  # tried twice, ran once


def test_rsync_and_scp_are_retried_too(monkeypatch, waits, tmp_path):
    """The upload is where the three lost recordings failed (its ``mkdir``), and the copy itself
    holds a session for as long as it runs."""
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    runs = {"n": 0}

    def fake_run(argv, **kwargs):
        runs["n"] += 1
        if argv[0] == "rsync" and runs["n"] <= 3:
            return _Result(255, stderr=REFUSED)
        return _Result(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ssh_transport, "_rsync_available", lambda: True)
    ssh_transport.copy_to_remote("tillicum", tmp_path, "/remote/exp")
    assert runs["n"] >= 4  # mkdir, then the rsync refused twice and run once
    assert len(waits) == 2
