"""Shell out to the user's own ssh/scp/rsync (§6) — never a Python SSH library.

A remote target's ``host`` is expected to match a ``Host`` alias in the user's own
``~/.ssh/config``, so whatever authentication is already set up there (keys, an agent,
ControlMaster connection-sharing, a cached Duo session) is what runs these commands.
Nothing here ever sees, stores, or prompts for a credential.

``rsync`` is preferred (incremental, one process for a whole tree) and ``scp`` is the
portable fallback — a Windows dev machine, an actively supported local platform for this
package, has no native ``rsync``.
"""

import contextlib
import os
import posixpath
import random
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from detail.core import atomic

_T = TypeVar("_T")


class RemoteCommandError(RuntimeError):
    """A remote command (or a local ssh/scp/rsync invocation) exited non-zero."""

    def __init__(
        self, command: list[str], returncode: int, stdout: str, stderr: str
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"command exited {returncode}: {' '.join(command)}\n{stderr or stdout}"
        )


#: ``ssh``'s own exit status for "I could not connect or authenticate" (a dropped or expired
#: shared session under ``BatchMode``, a network change). Not the remote command's: ``squeue``
#: and ``cat`` never exit 255, so it means the command was **never run**, and the caller knows
#: nothing about what it would have answered.
SSH_ERROR_EXIT = 255

#: rsync's exit status for "some source files vanished before they could be transferred". The
#: experiment is a live tree — eight shepherds rewrite their envelopes through atomic temps while
#: one uploads — so a temp rsync listed and then found gone is the normal case, not a failure:
#: everything that still exists was sent.
RSYNC_VANISHED_EXIT = 24

#: What a shell on the host says when it has no ``rsync`` to run for the far side of the copy
#: (bash ``rsync: command not found``, sh/dash ``rsync: not found``, zsh ``command not found:
#: rsync``, tcsh ``rsync: Command not found.``). The one rsync failure that means the plain
#: ``scp`` mirror is the only copy available.
_RSYNC_ABSENT = re.compile(
    r"\brsync: (?:command )?not found|command not found: rsync\b", re.IGNORECASE
)

#: How long a poll's single ssh call may take before it is read as "could not ask". A call that
#: hangs (a half-dead connection) otherwise blocks the poll loop for good.
ASK_TIMEOUT_S = 30

#: What ssh and the server say when the shared connection is **full**: OpenSSH's ``MaxSessions``
#: (10 unless the host raises it) caps the commands one connection carries at once, and a request
#: past it is refused with this text. Then ssh falls back to a fresh connection, which under
#: ``BatchMode`` cannot log in (Duo), so the *visible* error is "Permission denied" — the cause is
#: only in these lines. Everything below keys on them, never on the exit status alone, because a
#: genuine login failure also exits 255 and must not be retried.
_REFUSED_MARKERS = (
    "Session open refused by peer",
    "administratively prohibited: open failed",
)

#: How many times a command that was refused a session is tried again, and the longest wait
#: between tries (doubling from 1 s, jittered so eight jobs do not retry in step). A refusal is
#: transient by nature — commands are short and the sessions free up — and **the command never
#: ran**, which is what makes retrying it safe even for ``sbatch``.
REFUSED_RETRIES = 12
REFUSED_MAX_WAIT_S = 15.0

#: The budget for a poll's *ask*: two quick tries, because a poll that waits minutes for a session
#: delays the next cancel check, and the poll loop already has its own visible wait for a cluster
#: it cannot reach (``remote_shepherd._Reconnect``).
ASK_REFUSED_RETRIES = 2

_sleep = time.sleep  # indirection, so a test can run the backoff without waiting

#: How many ssh sessions DETAIL's processes on this machine hold on one host at once, together.
#: Below the host's ``MaxSessions`` (10 by default) so the person's own terminal on the same
#: shared connection still gets in. See :func:`session_slot`.
MAX_SESSIONS = 8

#: How many of those a long transfer (rsync, scp, a streamed remote step) may hold. The rest stay
#: free for the short commands a poll makes, so eight uploads cannot hold up every job's poll and
#: cancel check for as long as they run.
MAX_TRANSFER_SESSIONS = 6

#: How long a command waits before looking for a free session again.
SLOT_WAIT_S = 0.25

#: Where the session slots live: one lock file per slot, per host. Under the home directory rather
#: than the system temp folder, which another user on the machine may own first.
_SLOT_ROOT = Path.home() / ".cache" / "detail" / "ssh-sessions"

_slot_sleep = time.sleep  # indirection, so a test can wait for a slot without waiting


def _try_lock(path: Path) -> int | None:
    """An open descriptor holding an exclusive lock on ``path``, or ``None`` if it is held."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | atomic.O_BINARY, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    os.close(fd)  # releases a flock, and is what frees a slot of a process that dies


@contextlib.contextmanager
def session_slot(host: str | None, *, transfer: bool = False) -> Iterator[None]:
    """Hold one of ``host``'s :data:`MAX_SESSIONS` session slots for the body, waiting for one.

    **Every process asked the shared connection for a session whenever it liked.** A run starts
    one shepherd per recording, up to ``max_concurrent_jobs`` of them, and two runs at once is
    sixteen. Each polls with two commands every five seconds and uploads with long rsyncs, all
    through the one ControlMaster connection, whose host allows ten sessions. The retries in
    :func:`_retry_while_refused` kept commands from failing, but the polls ran out of their small
    budget, so every job read "cannot reach tillicum … log in again" about once a minute for half
    an hour while ssh worked. Counting refusals cannot fix that, because the excess is the sum of
    processes that know nothing of each other. So each command takes a slot first: a lock file,
    shared by every process on this machine, and released by the operating system if the process
    dies holding it.

    A ``transfer`` takes only the first :data:`MAX_TRANSFER_SESSIONS` slots; a short command tries
    the others first, so it rarely sits behind an upload.
    """
    if host is None:
        yield
        return
    folder = _SLOT_ROOT / re.sub(r"[^\w.-]", "_", host)
    folder.mkdir(parents=True, exist_ok=True)
    order = (
        range(MAX_TRANSFER_SESSIONS) if transfer else range(MAX_SESSIONS - 1, -1, -1)
    )
    while True:
        for index in order:
            fd = _try_lock(folder / f"{index}.lock")
            if fd is not None:
                try:
                    yield
                finally:
                    _unlock(fd)
                return
        _slot_sleep(SLOT_WAIT_S * random.uniform(0.5, 1.5))


#: Hosts whose most recent command was refused a session because the shared connection was full,
#: read by a caller that has to say *why* it could not ask (:func:`connection_full`).
_refused_last: set[str] = set()


def connection_full(host: str) -> bool:
    """Whether the last command :func:`run_remote` ran on ``host`` failed only because the shared
    connection had no free session. ssh exits 255 for that as for a dropped connection, but the
    fix differs: a full connection frees itself, and logging in again does nothing for it.
    """
    return host in _refused_last


#: What every ssh call of ours says about a connection it has to *make*: give up on an unanswered
#: connect in 20 s, and notice a connection that goes quiet after about 45 s (three missed
#: heartbeats). Without these a call over a half-dead TCP connection waits on the operating
#: system's own timeouts, which are minutes to hours. They govern the connection **this call**
#: opens; a call that rides a shared ControlMaster connection inherits the master's, which was
#: set by whoever opened it (see :func:`require_responsive`).
CONNECT_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
]

#: How long the shared connection has to answer a check before it is called stalled, and how long
#: an answer is trusted. A check is a message down a local socket to the master process, so a
#: healthy master answers in milliseconds even while it carries a multi-gigabyte upload.
STALL_CHECK_S = 15
_STALL_TRUST_S = 15.0
_answered_at: dict[str, float] = {}


class SshStalledError(RemoteCommandError):
    """The shared ssh connection to a host is open and not answering.

    Exits ``255`` like any ssh that could not reach the host, so every caller that already reads
    that as "could not ask" (:func:`ssh_failed`, the shepherd's reconnect wait) treats it the same.
    """

    def __str__(self) -> str:
        # The message *is* the error: a run's table shows the first line, and the generic
        # "command exited 255: ssh -O check …" would be all it said.
        return self.stderr

    def __init__(self, host: str) -> None:
        super().__init__(
            ["ssh", "-O", "check", host],
            SSH_ERROR_EXIT,
            "",
            f"the shared ssh connection to {host} is not responding (a check of it got no "
            f"answer in {STALL_CHECK_S} s). It went stale, usually after the computer slept or "
            f"the network changed, and every command would wait on it forever. Close it and "
            f"log in again, in a terminal: "
            f"`pkill -f 'ssh: .*\\[mux\\]'` then `ssh {host} true`.",
        )


#: What ssh prints when the **login** is refused: ``Permission denied`` followed by the list of
#: methods it tried, in parentheses. A remote file's permission error is ``Permission denied``
#: without that list, which is what keeps the two apart.
_LOGIN_FAILED = re.compile(r"Permission denied \([a-z-]+(?:,[a-z-]+)*\)")


def login_failed(text: str) -> bool:
    """Whether ``text`` (a failed ssh/rsync/scp's output) says the *login* was refused, not a
    command.

    Under ``BatchMode`` that is what an expired shared session looks like: with no connection to
    share, ssh tries a fresh login, which needs a password or a Duo approval it is told not to
    ask for. A refused *session* on a full connection prints this too, after the refusal line, so
    that case is excluded: its cause is a busy connection, and it is retried, not re-logged-in.
    """
    return bool(_LOGIN_FAILED.search(text)) and not session_refused(text)


def expired_session_advice(host: str | None) -> str:
    """What to do about an expired session, in the person's terms; shared by every surface."""
    target = host or "<your host>"
    return (
        f"your ssh session to {host or 'the cluster'} has expired and DETAIL cannot log in for "
        f"you (it never asks for a password or Duo). In a terminal run `ssh {target} true`, "
        f"complete the login, then Retry. If that still fails, run "
        f"`pkill -f 'ssh: .*\\[mux\\]'` first to drop a stale connection."
    )


class SshSessionExpiredError(RemoteCommandError):
    """The login was refused because the shared session is gone (see :func:`login_failed`).

    The original failure, unchanged (``command``, exit status, ``stderr``), with the cause and
    the fix first. Callers' reading of the exit status (``ssh`` exits ``255``, which they take as
    "could not ask") is not affected: only the text changes.
    """

    def __init__(self, host: str | None, original: RemoteCommandError) -> None:
        super().__init__(
            original.command, original.returncode, original.stdout, original.stderr
        )
        self.host = host
        self._advice = expired_session_advice(host)

    def __str__(self) -> str:
        # First line is the whole message: a run's table shows only that.
        return f"{self._advice}\n{super().__str__()}"


def explain_failure(host: str | None, error: RemoteCommandError) -> RemoteCommandError:
    """``error``, or the expired-session error that says what it means.

    Only a login failure is translated; a remote command's own failure, a refused session and a
    timeout are returned as they are. Judged by the text, not the exit status: ssh exits 255, but
    rsync and scp report the same failure under their own codes.
    """
    if login_failed(f"{error.stderr}\n{error.stdout}"):
        return SshSessionExpiredError(host, error)
    return error


def _host_in(argv: Sequence[str]) -> str | None:
    """The ``host`` of the first ``host:path`` argument of an rsync/scp command, if any.

    Skips a Windows drive (``C:\\...``), whose one-letter "host" is not one.
    """
    for arg in argv[1:]:
        head, colon, _ = arg.partition(":")
        if colon and len(head) > 1 and not re.search(r"[\\/\s]", head):
            return head
    return None


def require_responsive(host: str) -> None:
    """Raise :class:`SshStalledError` when the shared connection to ``host`` does not answer.

    **A connection that is open and silent was waited on for good.** ``ControlMaster`` hands every
    command to one long-lived ssh process, and when that process stops answering (a stale socket
    after a sleep or a network change; one run saw it 20 hours into a connection) each command
    sits on it at no CPU, with no error and nothing in the log. Eight recordings stood on
    "uploading experiment files" for eight minutes. A check is a message to the master, so asking
    it first turns that wait into a refusal that says what to do. Nothing else in a run asks.

    Only *no answer* counts. ``ssh -O check`` answering "no master running", or failing because
    this ssh has no connection sharing at all (Windows' OpenSSH), is an answer: the command that
    follows connects on its own and fails fast or works.
    """
    now = time.monotonic()
    answered = _answered_at.get(host)
    if answered is not None and now - answered < _STALL_TRUST_S:
        return
    try:
        subprocess.run(
            ["ssh", "-O", "check", host], capture_output=True, timeout=STALL_CHECK_S
        )
    except subprocess.TimeoutExpired:
        _answered_at.pop(host, None)
        raise SshStalledError(host) from None
    except FileNotFoundError:
        return  # no ssh here: the command itself says so, in words
    _answered_at[host] = now


def session_refused(text: str) -> bool:
    """Whether ``text`` (a failed ssh/rsync/scp's output) says the host refused a new session."""
    return any(marker in text for marker in _REFUSED_MARKERS)


def _retry_while_refused(attempt: Callable[[], _T], retries: int) -> _T:
    """Run ``attempt`` again, with a jittered backoff, while it fails *only* because the shared
    connection was full.

    **A full connection was recorded as a failed job.** Eight jobs sharing one ssh connection each
    ran ``mkdir``, ``rsync``, ``sbatch`` and a model check through it, one more session than the
    host allows was enough, and the first refusal ended the job — three recordings lost to a
    ``mkdir`` that would have worked a second later. The poll loop had learned not to read "could
    not ask" as an answer (``remote_shepherd._Reconnect``); the commands that *do* something had
    not.

    ``attempt`` either raises :class:`RemoteCommandError` or returns a ``CompletedProcess`` (the
    ``check=False`` callers read the exit status themselves). Anything that is not a refusal —
    a login failure, a remote command's own error — is returned or raised at once.
    """
    delay = 1.0
    for tries in range(retries + 1):
        try:
            result = attempt()
        except RemoteCommandError as exc:
            if tries == retries or not session_refused(f"{exc.stderr}\n{exc.stdout}"):
                raise
        else:
            # Read as a process result by what it carries, not its type: the only thing that
            # comes back here that is not one (a copy's ``None``) carries neither.
            refused = getattr(result, "returncode", 0) != 0 and session_refused(
                f"{getattr(result, 'stderr', '')}\n{getattr(result, 'stdout', '')}"
            )
            if tries == retries or not refused:
                return result
        _sleep(delay * random.uniform(0.5, 1.5))
        delay = min(delay * 2, REFUSED_MAX_WAIT_S)
    raise AssertionError("unreachable")  # the loop returns or raises on its last try


def ssh_failed(result: subprocess.CompletedProcess) -> bool:
    """Whether ``result`` is ssh failing to reach the host rather than a remote command exiting."""
    return result.returncode == SSH_ERROR_EXIT


def is_reachable(host: str, *, timeout_s: float = ASK_TIMEOUT_S) -> bool:
    """Whether a trivial command runs on ``host`` over the existing (or a fresh) ssh session.

    ``BatchMode`` means this never prompts: an expired session is ``False`` until the person logs
    in again (`ssh <host>`, or the Remote targets page's **Open SSH session**), which is exactly
    what a caller waiting to reconnect polls for.
    """
    try:
        result = run_remote(
            host,
            "true",
            check=False,
            timeout_s=timeout_s,
            refused_retries=ASK_REFUSED_RETRIES,
        )
    except subprocess.TimeoutExpired:
        return False
    return not ssh_failed(result)


def run_remote(
    host: str,
    command: str,
    *,
    stdin_text: str | None = None,
    timeout_s: float | None = None,
    check: bool = True,
    refused_retries: int = REFUSED_RETRIES,
) -> subprocess.CompletedProcess:
    """Run ``command`` on ``host`` over ssh (``BatchMode=yes`` — fail fast rather than
    hang on an interactive prompt) and return the completed process. ``stdin_text`` is
    piped to the remote command's stdin, which is how a file's content is written
    remotely without shell-quoting it into the command line (see
    :func:`write_remote_file`).

    A command the host **refused a session for** (the shared connection is full) is tried again
    up to ``refused_retries`` times before it counts as failed — see
    :func:`_retry_while_refused`.

    A shared connection that is open but not answering (:func:`require_responsive`) is an ssh
    failure, exit 255: raised when ``check``, otherwise returned for the caller's own
    :func:`ssh_failed` to read, as for a connection that never opened.

    Raises:
        RemoteCommandError: if ``check`` and the command exits non-zero.
    """
    argv = ["ssh", *CONNECT_OPTIONS, host, command]
    try:
        require_responsive(host)
    except SshStalledError as stalled:
        if check:
            raise
        return subprocess.CompletedProcess(
            argv, stalled.returncode, stalled.stdout, stalled.stderr
        )

    def attempt() -> subprocess.CompletedProcess:
        with session_slot(host):
            return subprocess.run(
                argv,
                input=stdin_text,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )

    _refused_last.discard(
        host
    )  # a timeout below leaves it clear: that is not a full connection
    result = _retry_while_refused(attempt, refused_retries)
    if result.returncode != 0 and session_refused(f"{result.stderr}\n{result.stdout}"):
        _refused_last.add(host)
    if check and result.returncode != 0:
        raise explain_failure(
            host,
            RemoteCommandError(argv, result.returncode, result.stdout, result.stderr),
        )
    return result


def run_remote_lines(
    host: str,
    command: str,
    on_line: Callable[[str], None],
    *,
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Run ``command`` on ``host`` and hand each line of its output to ``on_line`` as it
    arrives — for a remote step that runs for many minutes and reports as it goes (a model
    download), where :func:`run_remote` would say nothing until the end.

    Returns the completed process with every line as ``stdout`` (stderr is merged into it).
    No timeout: the remote step owns its own. Not retried on a refused session, since a step
    that already started cannot be safely started again.
    """
    argv = ["ssh", *CONNECT_OPTIONS, host, command]
    require_responsive(host)
    lines: list[str] = []
    with session_slot(host, transfer=True):
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        try:
            if stdin_text is not None:
                proc.stdin.write(stdin_text)
            proc.stdin.close()
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                lines.append(line)
                on_line(line)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            returncode = proc.wait()
    return subprocess.CompletedProcess(argv, returncode, "\n".join(lines), "")


def write_remote_file(
    host: str, path: str, content: str, *, timeout_s: float | None = None
) -> None:
    """Write ``content`` to ``path`` on ``host``, creating its parent directory.

    ``path`` is a remote (POSIX) path — never joined with :mod:`pathlib`, which would
    use backslashes on a Windows local machine.
    """
    parent = posixpath.dirname(path)
    command = f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(path)}"
    run_remote(host, command, stdin_text=content, timeout_s=timeout_s)


def read_remote_file(
    host: str,
    path: str,
    *,
    timeout_s: float | None = None,
    refused_retries: int = REFUSED_RETRIES,
) -> str:
    """Read the text content of ``path`` on ``host``.

    Raises:
        RemoteCommandError: if the file cannot be read (missing, permission, ...).
    """
    result = run_remote(
        host,
        f"cat {shlex.quote(path)}",
        timeout_s=timeout_s,
        refused_retries=refused_retries,
    )
    return result.stdout


def rsync_available() -> bool:
    """Whether ``rsync`` exists on *this* machine.

    Best-effort: a remote host missing it fails the rsync invocation, and
    ``copy_to_remote`` falls back to ``scp`` on exactly that failure (``copy_from_remote`` on
    any). Public
    because a caller deciding whether jobs may run concurrently has to know whether the upload can
    leave another job's run directory alone, which only the rsync path does.
    """
    return shutil.which("rsync") is not None


_rsync_available = rsync_available


def copy_to_remote(
    host: str,
    local_dir: Path,
    remote_dir: str,
    *,
    timeout_s: float | None = None,
    never_overwrite: Sequence[Path | str] = (),
    never_send: Sequence[str] = (),
) -> None:
    """Mirror the contents of local directory ``local_dir`` into ``remote_dir`` on
    ``host``, creating ``remote_dir`` if needed. Prefers ``rsync -az``; falls back to
    ``scp -r`` (per top-level child, so an empty directory is a no-op either way).

    ``never_overwrite`` names directories (relative to ``local_dir``, or under it) that are sent
    **only when absent on the host**: whatever a directory already holds there is left exactly
    as it is, and what it lacks is added. See :func:`_rsync_mirror` for why a caller wants that.
    Honoured by ``rsync`` only: the ``scp`` fallback has no way to ask "is it there already", so a
    host without rsync gets the plain mirror.

    ``never_send`` names files, by name wherever they are, that the ordinary pass never sends
    (with their ``<name>.*`` atomic-write temps); one inside a ``never_overwrite`` directory still
    arrives with it. Also ``rsync`` only.

    **``scp`` only where rsync does not exist** — on this machine, or on the host
    (:func:`rsync_missing_on_host`). Any other rsync failure is raised. It used to fall back on
    *every* failure, and a host that has rsync fails it for passing reasons (a refused session on
    a full connection, a dropped one): the fallback then re-sent the whole experiment with
    ``scp``, which honours neither list, over a cluster with jobs in flight, and laid each one's
    local ``job.json`` (``backend: remote``) over the envelope its shepherd had written. A
    worker that started inside that window read it and failed in ``resolve_device``.
    """
    run_remote(host, f"mkdir -p {shlex.quote(remote_dir)}", timeout_s=timeout_s)
    if _rsync_available():
        try:
            _rsync_to_remote(
                host,
                local_dir,
                remote_dir,
                timeout_s=timeout_s,
                never_overwrite=never_overwrite,
                never_send=never_send,
            )
            return
        except RemoteCommandError as exc:
            if not rsync_missing_on_host(exc):
                raise
    _scp_to_remote(host, local_dir, remote_dir, timeout_s=timeout_s)


def rsync_missing_on_host(error: RemoteCommandError) -> bool:
    """Whether a failed rsync failed because the host has no ``rsync`` to run."""
    return bool(_RSYNC_ABSENT.search(f"{error.stderr}\n{error.stdout}"))


#: What lets an interrupted single-file copy pick up where it stopped: rsync keeps the part it
#: sent in a hidden directory beside the destination and resumes from it on the next run. A
#: **directory**, not bare ``--partial``, which moves the half-sent file to the *final* path, where
#: a reader would take a truncated video for a finished one. Without either, a stopped upload of a
#: multi-gigabyte video started that file over. ``--partial-dir`` is old enough (rsync 2.6.4) for
#: the lab machines that have no ``--info`` (see :func:`parse_rsync_progress`).
RSYNC_RESUME = ["--partial-dir=.rsync-partial"]


def copy_file_to_remote(
    host: str,
    local_path: Path,
    remote_path: str,
    *,
    timeout_s: float | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> None:
    """Copy one local file to an exact remote path on ``host``, creating its parent
    directory. Prefers ``rsync -az`` (a re-run against an unchanged file is then a
    cheap no-op — this is called on every job submission, not just the first); falls
    back to plain ``scp``.

    Unlike :func:`copy_to_remote`, which mirrors a whole directory (the experiment
    tree), this copies a single file at an exact destination — for a video file
    resolved through a ``video_roots`` alias (§2), which lives outside the experiment
    directory entirely and so is never swept up by that mirror.

    ``on_bytes(n)`` is called with the bytes of **this file** sent so far as ``rsync``
    reports them, so a caller copying many large files can show motion *within* one. It
    is not called for ``scp`` (which reports nothing here), nor when ``timeout_s`` is set:
    a deadline needs the plain run, which cannot be interrupted mid-read.
    """
    remote_dir = posixpath.dirname(remote_path)
    run_remote(host, f"mkdir -p {shlex.quote(remote_dir)}", timeout_s=timeout_s)
    if _rsync_available():
        try:
            if on_bytes is not None and timeout_s is None:
                _retry_while_refused(
                    lambda: _rsync_file_reporting(
                        host, local_path, remote_path, on_bytes
                    ),
                    REFUSED_RETRIES,
                )
            else:
                _run(
                    [
                        "rsync",
                        "-az",
                        *RSYNC_RESUME,
                        str(local_path),
                        f"{host}:{remote_path}",
                    ],
                    timeout_s=timeout_s,
                )
            return
        except RemoteCommandError:
            pass  # remote may lack rsync — fall back
    _run(["scp", str(local_path), f"{host}:{remote_path}"], timeout_s=timeout_s)


_RSYNC_PROGRESS = re.compile(r"^\s*([\d,]+)\s+\d+%")
_PROGRESS_SEPARATORS = re.compile(rb"[\r\n]")


def parse_rsync_progress(line: str) -> int | None:
    """Bytes of the current file sent so far, from one ``rsync --progress`` line, else ``None``.

    The line looks like ``  1,396,736,000  100%   46.21MB/s    0:00:28`` (the filename line
    that precedes it, and the trailing ``(xfr#4, to-chk=76/80)`` summary after it, are not
    progress). ``--progress`` rather than ``--info=progress2``: the older rsync a lab machine
    can carry has no ``--info``, and an unknown option would read as a failed copy and fall
    back to ``scp``, which reports nothing.
    """
    match = _RSYNC_PROGRESS.match(line)
    return int(match.group(1).replace(",", "")) if match else None


def _rsync_file_reporting(
    host: str, local_path: Path, remote_path: str, on_bytes: Callable[[int], None]
) -> None:
    """``rsync -az --progress`` of one file, calling ``on_bytes`` as it reports.

    Raises:
        RemoteCommandError: ``rsync`` exited non-zero (its last output is the message), so the
            caller's ``scp`` fallback sees the same failure the plain run would have given.
    """
    argv = [
        "rsync",
        "-az",
        *RSYNC_RESUME,
        "--progress",
        str(local_path),
        f"{host}:{remote_path}",
    ]
    other: list[str] = []  # what is not progress: the error text if this fails
    pending = b""
    with session_slot(host, transfer=True):
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                *lines, pending = _PROGRESS_SEPARATORS.split(pending + chunk)
                for raw in lines:
                    text = raw.decode("utf-8", errors="replace")
                    sent = parse_rsync_progress(text)
                    if sent is not None:
                        on_bytes(sent)
                    elif text.strip():
                        other = [*other[-9:], text]
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            returncode = proc.wait()
    if returncode != 0:
        raise explain_failure(
            host, RemoteCommandError(argv, returncode, "", "\n".join(other))
        )


def copy_from_remote(
    host: str,
    remote_dir: str,
    local_dir: Path,
    *,
    timeout_s: float | None = None,
    keep_local: Sequence[str] = (),
) -> None:
    """Mirror the contents of ``remote_dir`` on ``host`` into local directory
    ``local_dir``, creating it if needed.

    ``keep_local`` names files directly in ``local_dir`` that the copy never replaces, and never
    even moves. The shepherd's own log is one: it is the shepherd's stdout, and rsync replaces a
    file by renaming a new one over it, so the cluster's stale copy (uploaded when the run began)
    took the name while the shepherd went on writing into the unlinked original. Every line after
    the copy back was lost and the log read as if the run had stopped at "uploading experiment
    files". The run's envelope is the other (#742): the cluster's copy says the job is over, and
    a reader that sees it acts on a run whose data is still arriving.

    rsync excludes them. ``scp`` cannot, so it copies into a staging folder inside ``local_dir``
    (same filesystem, so each file is then a rename, not a second copy) and every file but the
    kept ones is moved into place. The kept files stay where they are throughout: moving them
    aside, as this used to, left the envelope *missing* for the whole copy, and copying them back
    over after would have shown the cluster's terminal status for its length.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    require_responsive(host)
    if _rsync_available():
        try:
            _rsync_from_remote(
                host, remote_dir, local_dir, timeout_s=timeout_s, keep_local=keep_local
            )
            return
        except RemoteCommandError:
            pass
    staging = local_dir / _SCP_STAGING
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        _scp_from_remote(host, remote_dir, staging, timeout_s=timeout_s)
        _move_into_place(staging, local_dir, skip=frozenset(keep_local))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


#: Where the ``scp`` fallback lands a copy back before it is moved into place: hidden, inside the
#: run folder, so the move is a rename on one filesystem.
_SCP_STAGING = ".scp-incoming"


def _move_into_place(staging: Path, local_dir: Path, *, skip: frozenset[str]) -> None:
    """Every file under ``staging`` renamed to the same place under ``local_dir``, except the
    top-level names in ``skip`` (and the staging folder itself, should the copy contain one).
    """
    for src in sorted(staging.rglob("*")):  # a parent sorts before its children
        rel = src.relative_to(staging)
        if rel.parts[0] in skip or rel.parts[0] == _SCP_STAGING:
            continue
        dest = local_dir / rel
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            atomic.replace(src, dest)


def _run(
    argv: list[str], *, timeout_s: float | None, accept: Sequence[int] = (0,)
) -> None:
    def attempt() -> None:
        with session_slot(_host_in(argv), transfer=True):
            result = subprocess.run(
                argv,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        if result.returncode not in accept:
            raise explain_failure(
                _host_in(argv),
                RemoteCommandError(
                    argv, result.returncode, result.stdout, result.stderr
                ),
            )

    _retry_while_refused(attempt, REFUSED_RETRIES)


def mirror_to_remote(
    host: str,
    local_dir: Path,
    remote_dir: str,
    *,
    keep_remote: Sequence[str] = (),
    timeout_s: float | None = None,
) -> None:
    """Make ``remote_dir`` on ``host`` exactly ``local_dir``: send what differs, and remove
    what this machine no longer has — except the files named in ``keep_remote`` (relative to
    the directory), which are neither sent nor removed.

    For a directory a remote job *reads* and this machine owns (a clustering run a Profile &
    describe job profiles): the plain upload never rewrites a run directory the cluster holds,
    and a copy that only adds would leave there what a cancelled job wrote and never brought
    back. rsync only — the ``scp`` fallback cannot remove — so a machine or host without it is
    refused with the reason rather than given a copy that is not exact.

    Raises:
        RuntimeError: rsync is not available here or on the host.
        RemoteCommandError: rsync failed.
    """
    if not _rsync_available():
        raise RuntimeError(
            f"refreshing {Path(local_dir).name} on {host} needs rsync on this machine; "
            "install it (in WSL: sudo apt install rsync)"
        )
    run_remote(host, f"mkdir -p {shlex.quote(remote_dir)}", timeout_s=timeout_s)
    src = str(local_dir)
    if not src.endswith(("/", "\\")):
        src += "/"
    excludes = [f"--exclude={_rsync_pattern(name)}" for name in keep_remote]
    try:
        _run(
            ["rsync", "-az", "--delete", *excludes, src, f"{host}:{remote_dir}/"],
            timeout_s=timeout_s,
            accept=_MIRROR_SUCCESS,
        )
    except RemoteCommandError as exc:
        if rsync_missing_on_host(exc):
            raise RuntimeError(
                f"refreshing {Path(local_dir).name} on {host} needs rsync there"
            ) from exc
        raise


def _rsync_to_remote(
    host, local_dir, remote_dir, *, timeout_s, never_overwrite=(), never_send=()
):
    _rsync_mirror(
        local_dir,
        f"{host}:{remote_dir}/",
        timeout_s=timeout_s,
        never_overwrite=never_overwrite,
        never_send=never_send,
    )


def _rsync_pattern(relative: str) -> str:
    """``relative`` as an rsync exclude pattern that matches that one path and nothing else.

    Anchored with a leading slash (so it is relative to the transfer root, not "anywhere"), and
    with rsync's wildcard characters escaped: an experiment is named by a person, and a folder
    called ``[old] runs`` would otherwise be a character class.
    """
    return f"/{_rsync_escape(relative)}"


def _rsync_escape(text: str) -> str:
    """``text`` with rsync's wildcard characters escaped, so it matches only itself."""
    return "".join(f"\\{c}" if c in "\\*?[" else c for c in text)


#: A mirror of a live tree is complete when everything still there was sent.
_MIRROR_SUCCESS = (0, RSYNC_VANISHED_EXIT)


def _rsync_mirror(
    local_dir, destination: str, *, timeout_s, never_overwrite=(), never_send=()
):
    """``rsync -az`` of ``local_dir``'s contents to ``destination`` (``host:/path/``).

    **With ``never_overwrite``: two passes.** The first copies everything *except* those
    directories. The second sends them with ``--ignore-existing``, so a directory the host has
    already is never rewritten and one it lacks arrives whole.

    *Why a caller wants that:* a job's run directory is the **host's** from the moment it is
    submitted, because the worker on the cluster rewrites ``job.json``, its progress and its
    result records there. The local copy of that directory is older (the last sync back, or the
    initial pending state), and a plain mirror lays the older copy over the newer one. With one
    job in flight nobody notices; with two, the second job's upload resets the first's
    ``job.json`` just as it finishes, its shepherd never sees the terminal status, and a
    completed job is recorded as failed. No clock is consulted (``--update`` would be: it skips
    "newer" files by mtime, and the cluster's clock is not ours).

    **``never_send``: what the list cannot cover.** ``never_overwrite`` is taken when the upload
    starts, and an upload can run for twenty minutes. A run directory created after that — the
    next recording of the same batch — is ordinary to the first pass, which reaches it whenever
    its walk gets there: by then that job's shepherd may have written the cluster's ``job.json``
    (with the compute the cluster resolves), and the first pass lays the local one, ``backend:
    remote``, over it. Four recordings of one keypoints run failed that way. A name in
    ``never_send`` is left out of the first pass wherever it is, so a run's envelope only ever
    arrives through the second pass of the job that listed it.
    """
    src = str(local_dir)
    if not src.endswith(("/", "\\")):
        src += "/"
    root = Path(local_dir).resolve()
    held = []
    for path in never_overwrite:
        path = Path(path)
        relative = path.resolve().relative_to(root) if path.is_absolute() else path
        held.append(relative.as_posix())
    unsent = [
        pattern
        for name in never_send
        for pattern in (_rsync_escape(name), f"{_rsync_escape(name)}.*")
    ]
    if not held and not unsent:
        _run(
            ["rsync", "-az", src, destination],
            timeout_s=timeout_s,
            accept=_MIRROR_SUCCESS,
        )
        return
    excludes = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".rsync-exclude", delete=False
    )
    listed = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".rsync-files", delete=False
    )
    try:
        with excludes, listed:
            excludes.write("".join(f"{_rsync_pattern(h)}/\n" for h in held))
            excludes.write("".join(f"{pattern}\n" for pattern in unsent))
            listed.write("".join(f"{h}/\n" for h in held))
        _run(
            ["rsync", "-az", f"--exclude-from={excludes.name}", src, destination],
            timeout_s=timeout_s,
            accept=_MIRROR_SUCCESS,
        )
        if not held:
            return
        # `-r` explicitly: `--files-from` switches the recursion `-a` implies off.
        _run(
            [
                "rsync",
                "-az",
                "-r",
                "--ignore-existing",
                f"--files-from={listed.name}",
                src,
                destination,
            ],
            timeout_s=timeout_s,
            accept=_MIRROR_SUCCESS,
        )
    finally:
        for temp in (excludes.name, listed.name):
            Path(temp).unlink(missing_ok=True)


def _rsync_from_remote(host, remote_dir, local_dir, *, timeout_s, keep_local=()):
    excludes = [f"--exclude={_rsync_pattern(name)}" for name in keep_local]
    _run(
        ["rsync", "-az", *excludes, f"{host}:{remote_dir}/", f"{local_dir}/"],
        timeout_s=timeout_s,
    )


def _scp_to_remote(host, local_dir, remote_dir, *, timeout_s):
    children = sorted(Path(local_dir).iterdir())
    if not children:
        return
    _run(
        ["scp", "-r", *[str(c) for c in children], f"{host}:{remote_dir}/"],
        timeout_s=timeout_s,
    )


def _scp_from_remote(host, remote_dir, local_dir, *, timeout_s):
    # The trailing "/." copies the remote directory's *contents* rather than the
    # directory itself, matching rsync's trailing-slash semantics above.
    _run(["scp", "-r", f"{host}:{remote_dir}/.", str(local_dir)], timeout_s=timeout_s)
