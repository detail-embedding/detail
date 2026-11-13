"""What a person reads when the shared ssh session has expired (§6).

**The class: a failure reported in the tool's words, not the cause's.** With the shared session
gone, ssh falls back to a fresh login, which under ``BatchMode`` cannot ask for a password or Duo,
so it fails with ``Permission denied (gssapi-keyex,gssapi-with-mic,keyboard-interactive)``. That
is true and tells a person nothing: 77 recordings of a detection run each failed on it in eight
seconds, and the run table showed ``RemoteCommandError: command exited 255: ssh -o BatchMode=yes
…`` with the cause on a second line nobody read. The fix is one command, and nothing said which.

These tests give the failing ssh/rsync/scp the exact text a real expired session produces and
assert what comes out: the cause and the command first, the original failure kept (callers read its
exit status and ``stderr``), and **nothing else translated**. A remote file's own
``Permission denied``, a refused session on a full connection (which prints the same words after
its refusal line, and is retried instead) and a timeout must keep their own messages, or the
advice becomes the thing that is wrong.
"""

import subprocess

import pytest

from detail.api import remote as api_remote
from detail.core.dataclasses import JobStatus
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState

# Fixtures and helpers of the shepherd tests: the same experiment and fake scheduler.
from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeScheduler,
    _install_scheduler,
    experiment,
)

_REAL_RUN_REMOTE = ssh_transport.run_remote  # before any fixture swaps it for a fake

LOGIN_REFUSED = (
    "sagolden@tillicum.hyak.uw.edu: Permission denied "
    "(gssapi-keyex,gssapi-with-mic,keyboard-interactive).\n"
)
REFUSED_THEN_DENIED = (
    "mux_client_request_session: session request failed: Session open refused by peer\n"
    + LOGIN_REFUSED
)


class _Result:
    def __init__(self, returncode=255, stdout="", stderr=LOGIN_REFUSED):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _answer(monkeypatch, result):
    """Every subprocess.run answers ``result`` (the shared-connection check is stubbed out)."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _first_line(error) -> str:
    return str(error).splitlines()[0]


def test_an_expired_session_says_so_and_names_the_command(monkeypatch):
    _answer(monkeypatch, _Result())

    with pytest.raises(ssh_transport.SshSessionExpiredError) as caught:
        ssh_transport.run_remote("tillicum", "mkdir -p /x")

    first = _first_line(caught.value)
    assert "ssh session to tillicum has expired" in first
    assert (
        "ssh tillicum true" in first
    )  # the one thing to do, on the line a table shows
    assert "Retry" in first


def test_the_original_failure_is_kept_for_the_callers_that_read_it(monkeypatch):
    _answer(monkeypatch, _Result())

    with pytest.raises(ssh_transport.RemoteCommandError) as caught:
        ssh_transport.run_remote("tillicum", "mkdir -p /x")

    error = caught.value
    assert (
        error.returncode == ssh_transport.SSH_ERROR_EXIT
    )  # "could not ask", as before
    assert "Permission denied (gssapi-keyex" in error.stderr
    assert error.command[-2:] == ["tillicum", "mkdir -p /x"]
    assert "Permission denied" in str(
        error
    )  # and it is still in the message, below the advice


def test_a_poll_that_reads_the_status_itself_is_not_given_an_exception(monkeypatch):
    """``check=False`` callers (the scheduler poll) read the exit status; they still get it."""
    _answer(monkeypatch, _Result())

    result = ssh_transport.run_remote("tillicum", "squeue", check=False)

    assert ssh_transport.ssh_failed(result)


@pytest.mark.parametrize(
    "result",
    [
        _Result(1, stderr="mkdir: cannot create directory '/x': Permission denied\n"),
        _Result(255, stderr=REFUSED_THEN_DENIED),
        _Result(
            255, stderr="ssh: connect to host tillicum port 22: Connection refused\n"
        ),
        _Result(2, stderr="cat: /x: No such file or directory\n"),
    ],
    ids=["file-permission", "refused-session", "connection-refused", "remote-error"],
)
def test_nothing_else_is_called_an_expired_session(monkeypatch, result):
    """A remote file's permissions, a full connection and a refused connect keep their words."""
    monkeypatch.setattr(ssh_transport, "REFUSED_RETRIES", 0)
    _answer(monkeypatch, result)

    with pytest.raises(ssh_transport.RemoteCommandError) as caught:
        ssh_transport.run_remote("tillicum", "mkdir -p /x", refused_retries=0)

    assert not isinstance(caught.value, ssh_transport.SshSessionExpiredError)
    assert "has expired" not in str(caught.value)


def test_a_copy_over_an_expired_session_says_so_with_the_hosts_name(
    monkeypatch, tmp_path
):
    """rsync and scp fail the same way; they have no ``host`` argument, so it is read from the
    ``host:path`` they were given."""
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(ssh_transport, "_rsync_available", lambda: True)
    _answer(monkeypatch, _Result(255, stderr=LOGIN_REFUSED))

    with pytest.raises(ssh_transport.SshSessionExpiredError) as caught:
        ssh_transport._run(
            ["rsync", "-az", f"{tmp_path}/", "tillicum:/remote/dir/"], timeout_s=None
        )

    assert "ssh session to tillicum has expired" in _first_line(caught.value)


def test_a_copy_that_names_no_host_still_gives_the_advice(monkeypatch):
    _answer(monkeypatch, _Result(255, stderr=LOGIN_REFUSED))

    with pytest.raises(ssh_transport.SshSessionExpiredError) as caught:
        ssh_transport._run(["rsync", "-az", "somewhere"], timeout_s=None)

    assert "ssh session to the cluster has expired" in _first_line(caught.value)


@pytest.mark.parametrize(
    "argv, host",
    [
        (["rsync", "-az", "/local/", "tillicum:/remote/"], "tillicum"),
        (["rsync", "-az", "tillicum:/remote/", "/local/"], "tillicum"),
        (["scp", "C:\\videos\\a.mp4", "tillicum:/remote/a.mp4"], "tillicum"),  # a drive
        (["rsync", "-az", "--exclude-from=/tmp/x", "/local/"], None),
    ],
    ids=["upload", "download", "windows-drive-skipped", "no-host"],
)
def test_the_host_is_read_from_the_copy_command(argv, host):
    assert ssh_transport._host_in(argv) == host


def test_the_remote_target_test_gives_the_same_advice(monkeypatch):
    """`detail remote test` and the Remote targets page's Test button run their own ssh."""
    from detail.api import remote as api

    monkeypatch.setattr(
        api,
        "get_remote_target",
        lambda name: type("T", (), {"host": "tillicum"})(),
    )
    _answer(monkeypatch, _Result(255, stderr=LOGIN_REFUSED))

    status = api_remote.test_remote_target("tillicum")

    assert not status.ok
    assert "ssh session to tillicum has expired" in status.detail
    assert "ssh tillicum true" in status.detail


def test_a_run_whose_upload_meets_an_expired_session_fails_with_the_advice_first(
    monkeypatch, experiment
):
    """The incident: 77 recordings, each recorded as `RemoteCommandError: command exited 255: …`."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Result())

    def upload(host, local_dir, remote_dir, **kwargs):
        # the first thing the real upload does, over the real transport and the failing ssh
        # (`run_remote` itself is replaced by the shepherd fixture, so the import-time one)
        _REAL_RUN_REMOTE(host, f"mkdir -p {remote_dir}")

    monkeypatch.setattr(ssh_transport, "copy_to_remote", upload)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    first = rundir.read_envelope(run_dir)["error"].splitlines()[0]
    assert "SshSessionExpiredError" in first
    assert "ssh session to tillicum has expired" in first
    assert "ssh tillicum true" in first
