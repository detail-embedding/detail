"""ssh_transport.py — shells out to ssh/scp/rsync; every subprocess call is mocked
here, so nothing in this file needs network access or a real remote host."""

import subprocess

import pytest

from detail.executor import ssh_transport


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_remote_builds_batch_mode_command(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeResult(returncode=0, stdout="ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ssh_transport.run_remote("tillicum", "echo hi")
    assert captured["argv"] == [
        "ssh",
        *ssh_transport.CONNECT_OPTIONS,
        "tillicum",
        "echo hi",
    ]
    assert "BatchMode=yes" in captured["argv"]
    assert result.stdout == "ok\n"


def test_run_remote_passes_stdin_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: (captured.update(kwargs) or _FakeResult()),
    )
    ssh_transport.run_remote("tillicum", "cat > f", stdin_text="hello")
    assert captured["input"] == "hello"


def test_run_remote_raises_on_nonzero_by_default(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _FakeResult(returncode=1, stderr="boom"),
    )
    with pytest.raises(ssh_transport.RemoteCommandError, match="boom"):
        ssh_transport.run_remote("tillicum", "false")


def test_run_remote_check_false_returns_nonzero(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeResult(returncode=1, stderr="boom")
    )
    result = ssh_transport.run_remote("tillicum", "false", check=False)
    assert result.returncode == 1


def test_write_remote_file_pipes_content_and_makes_parent(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["stdin"] = kwargs.get("input")
        return _FakeResult(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ssh_transport.write_remote_file("tillicum", "/scratch/x/run_1/job.json", '{"a": 1}')
    command = captured["argv"][-1]
    assert "mkdir -p" in command and "/scratch/x/run_1" in command
    assert "cat >" in command and "/scratch/x/run_1/job.json" in command
    assert captured["stdin"] == '{"a": 1}'


def test_read_remote_file_returns_stdout(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeResult(returncode=0, stdout="content")
    )
    assert ssh_transport.read_remote_file("tillicum", "/scratch/x") == "content"


def test_copy_to_remote_prefers_rsync_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: "/usr/bin/rsync")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _FakeResult(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    ssh_transport.copy_to_remote("tillicum", tmp_path, "/scratch/exp")
    # first call makes the remote dir (over ssh), second is the rsync itself
    assert calls[0][0] == "ssh"
    assert calls[1][0] == "rsync"
    assert calls[1][1] == "-az"
    assert calls[1][2].endswith("/")
    assert calls[1][3] == "tillicum:/scratch/exp/"


def test_copy_to_remote_falls_back_to_scp_when_rsync_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **k: (calls.append(argv), _FakeResult())[1]
    )
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    ssh_transport.copy_to_remote("tillicum", tmp_path, "/scratch/exp")
    scp_call = calls[1]
    assert scp_call[0] == "scp"
    assert scp_call[1] == "-r"
    assert scp_call[-1] == "tillicum:/scratch/exp/"
    # every top-level child was passed as its own source arg
    assert str(tmp_path / "a.txt") in scp_call
    assert str(tmp_path / "sub") in scp_call


def test_copy_to_remote_scp_is_noop_on_empty_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **k: (calls.append(argv), _FakeResult())[1]
    )
    ssh_transport.copy_to_remote("tillicum", tmp_path, "/scratch/exp")
    assert len(calls) == 1  # only the mkdir -p over ssh; no scp with zero children


def test_copy_to_remote_falls_back_to_scp_when_the_host_has_no_rsync(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: "/usr/bin/rsync")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "rsync":
            return _FakeResult(
                returncode=12, stderr="bash: line 1: rsync: command not found"
            )
        return _FakeResult(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    ssh_transport.copy_to_remote("tillicum", tmp_path, "/scratch/exp")
    assert calls[1][0] == "rsync"
    assert calls[2][0] == "scp"


def test_copy_from_remote_creates_local_dir_and_prefers_rsync(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: "/usr/bin/rsync")
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **k: (calls.append(argv), _FakeResult())[1]
    )
    dest = tmp_path / "local" / "exp"
    ssh_transport.copy_from_remote("tillicum", "/scratch/exp", dest)
    assert dest.is_dir()
    assert calls[0][0] == "rsync"
    assert calls[0][2] == "tillicum:/scratch/exp/"


def test_copy_from_remote_scp_fallback_uses_dot_suffix(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **k: (calls.append(argv), _FakeResult())[1]
    )
    dest = tmp_path / "local" / "exp"
    ssh_transport.copy_from_remote("tillicum", "/scratch/exp", dest)
    assert calls[0][0] == "scp"
    assert calls[0][2] == "tillicum:/scratch/exp/."


# --- byte progress on a single-file copy (#644) ----------------------------------------- #
# A multi-gigabyte upload used to report nothing until it finished. `rsync --progress` prints a
# carriage-return-overwritten line per second; the transport turns those into `on_bytes(n)` calls.
@pytest.mark.parametrize(
    "line, sent",
    [
        ("  1,396,736,000  100%   46.21MB/s    0:00:28", 1_396_736_000),
        ("     32,768   0%    0.00kB/s    0:00:00", 32_768),
        (
            "  1,396,736,000 100%   46.21MB/s    0:00:28 (xfr#4, to-chk=76/80)",
            1_396_736_000,
        ),
        ("        0   0%    0.00kB/s    0:00:00", 0),
        # Not progress. A filename that starts with digits (these videos all do) must not parse.
        ("2024-11-11_09-44-01_BOX5_AM_21_DrugOn_S16.mp4", None),
        ("2024 11 11 12% of the file", None),
        ("sending incremental file list", None),
        ("", None),
    ],
)
def test_parse_rsync_progress(line, sent):
    assert ssh_transport.parse_rsync_progress(line) == sent


class _FakeStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read1(self, _size):
        return self._chunks.pop(0) if self._chunks else b""

    def close(self):
        pass


class _FakeProc:
    def __init__(self, chunks, returncode=0):
        self.stdout = _FakeStdout(chunks)
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        raise AssertionError("a finished process is never killed")


def _rsync_with_output(monkeypatch, chunks, returncode=0):
    """Route `rsync` through a fake Popen emitting ``chunks``; `ssh`/`scp` stay on `run`."""
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: "/usr/bin/rsync")
    popen_calls, run_calls = [], []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda argv, **kw: (popen_calls.append(argv), _FakeProc(chunks, returncode))[1],
    )
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **k: (run_calls.append(argv), _FakeResult())[1]
    )
    return popen_calls, run_calls


def test_copy_file_to_remote_reports_bytes_as_rsync_prints_them(monkeypatch, tmp_path):
    """The reads cut a line in two (the second chunk finishes `1,000,000`), and rsync separates
    updates with `\r`, not a newline."""
    chunks = [
        b"x.mp4\r\n    32,768   0%  0.00kB/s  0:00:00\r  1,00",
        b"0,000  50%  1.00MB/s  0:00:01\r  2,000,000 100%  1.00MB/s  0:00:02 "
        b"(xfr#1, to-chk=0/1)\r\n",
    ]
    popen_calls, _ = _rsync_with_output(monkeypatch, chunks)
    seen = []
    ssh_transport.copy_file_to_remote(
        "tillicum", tmp_path / "x.mp4", "/scratch/x.mp4", on_bytes=seen.append
    )
    assert seen == [32_768, 1_000_000, 2_000_000]
    assert popen_calls[0][:4] == [
        "rsync",
        "-az",
        "--partial-dir=.rsync-partial",
        "--progress",
    ]
    assert popen_calls[0][-1] == "tillicum:/scratch/x.mp4"


def test_a_failed_rsync_with_a_callback_falls_back_to_scp_like_the_plain_run(
    monkeypatch, tmp_path
):
    _, run_calls = _rsync_with_output(
        monkeypatch, [b"rsync: connection unexpectedly closed\n"], returncode=12
    )
    ssh_transport.copy_file_to_remote(
        "tillicum", tmp_path / "x.mp4", "/scratch/x.mp4", on_bytes=lambda n: None
    )
    assert [argv[0] for argv in run_calls] == ["ssh", "scp"]


def test_the_reader_raises_with_rsyncs_own_words_when_it_fails(monkeypatch, tmp_path):
    _rsync_with_output(
        monkeypatch, [b"rsync: write failed: No space left on device\n"], returncode=11
    )
    with pytest.raises(ssh_transport.RemoteCommandError, match="No space left"):
        ssh_transport._rsync_file_reporting(
            "tillicum", tmp_path / "x.mp4", "/scratch/x.mp4", lambda n: None
        )


def test_without_a_callback_or_with_a_deadline_the_plain_run_is_used(
    monkeypatch, tmp_path
):
    """Streaming cannot honour a timeout (a blocked read is not interruptible), and a caller
    that wants no progress gets exactly the old command."""
    popen_calls, run_calls = _rsync_with_output(monkeypatch, [])
    ssh_transport.copy_file_to_remote("tillicum", tmp_path / "x.mp4", "/scratch/x.mp4")
    ssh_transport.copy_file_to_remote(
        "tillicum",
        tmp_path / "x.mp4",
        "/scratch/x.mp4",
        timeout_s=60,
        on_bytes=lambda n: None,
    )
    assert popen_calls == []
    assert [a for a in run_calls if a[0] == "rsync"] == [
        [
            "rsync",
            "-az",
            "--partial-dir=.rsync-partial",
            str(tmp_path / "x.mp4"),
            "tillicum:/scratch/x.mp4",
        ]
    ] * 2


@pytest.mark.parametrize(
    "returncode, reachable",
    [
        (0, True),
        (1, True),  # the remote command failing is not ssh failing
        (127, True),
        (255, False),  # ssh's own "could not connect or authenticate"
    ],
)
def test_is_reachable_tells_ssh_failing_from_a_remote_command_failing(
    monkeypatch, returncode, reachable
):
    monkeypatch.setattr(
        ssh_transport.subprocess,
        "run",
        lambda argv, **k: _FakeResult(returncode=returncode),
    )

    assert ssh_transport.is_reachable("tillicum") is reachable


def test_is_reachable_reads_a_call_that_never_answers_as_unreachable(monkeypatch):
    def hang(argv, **k):
        raise ssh_transport.subprocess.TimeoutExpired(cmd=argv, timeout=k["timeout"])

    monkeypatch.setattr(ssh_transport.subprocess, "run", hang)

    assert ssh_transport.is_reachable("tillicum") is False


def test_an_interrupted_video_copy_resumes_without_leaving_a_truncated_file_in_place(
    monkeypatch, tmp_path
):
    """A stopped background upload is restarted later, and a multi-gigabyte video must continue
    rather than start over (`--partial-dir`). Bare `--partial` would resume too, but by leaving the
    half-sent file at the *final* path, where a job on the cluster takes it for a whole video.
    """
    popen_calls, run_calls = _rsync_with_output(monkeypatch, [])
    ssh_transport.copy_file_to_remote(
        "tillicum", tmp_path / "x.mp4", "/scratch/x.mp4", on_bytes=lambda n: None
    )
    ssh_transport.copy_file_to_remote("tillicum", tmp_path / "x.mp4", "/scratch/x.mp4")

    rsyncs = [popen_calls[0], *[a for a in run_calls if a[0] == "rsync"]]
    assert len(rsyncs) == 2
    for argv in rsyncs:
        assert "--partial-dir=.rsync-partial" in argv
        assert "--partial" not in argv
