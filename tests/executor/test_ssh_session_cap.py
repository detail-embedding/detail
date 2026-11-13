"""DETAIL's processes share a host's ssh sessions under one cap (§6).

**The class: a shared, capped resource taken by many processes that do not know of each other.**
Every remote job has its own shepherd process, and a run starts up to ``max_concurrent_jobs`` of
them (two runs at once is sixteen). All of them send their commands down the one ControlMaster
connection, and the host lets that connection carry ten sessions (OpenSSH's ``MaxSessions``).
Each shepherd polls with a ``cat`` and a ``squeue`` every five seconds and uploads with rsyncs
that hold a session for minutes. The refusal retries (``test_session_refusal.py``) kept each
command from failing, but no process could see the others, so the excess was never smaller than
the sum. On 2026-11-03 every job of a 16-shepherd pair of runs logged "cannot reach tillicum over
ssh … log in again" about once a minute for half an hour while ssh worked: the polls ran out of
their small refusal budget, and the waiting message said to log in again because ssh exits 255
for a full connection exactly as for a dropped one.

**It hid because every test ran one command at a time.** The fakes answered at once, so nothing
ever held a session while another asked for one.

**The rule.** Every ssh, rsync and scp that DETAIL starts holds a ``session_slot`` for as long as
it runs: a lock file per slot, shared by every process on the machine and freed by the operating
system if its holder dies. Long transfers may hold only ``MAX_TRANSFER_SESSIONS`` of the
``MAX_SESSIONS`` slots, so polls and cancels still get in during an upload. A wait the connection
could not serve says "full", not "log in again".

**Where the fix stops.** The sweep for argv literals starting ``ssh`` / ``rsync`` / ``scp`` and
for ``subprocess`` calls beside them finds every launch in ``executor/ssh_transport.py`` plus
``api/remote.py::test_remote_target``; all are slotted. ``ssh -O check`` in
``require_responsive`` is deliberately not: it is a message to the master process and opens no
session. ``checks/commands.py::ssh_refresh_argv`` is deliberately not: it opens the person's own
interactive login in a terminal, which is what the two slots under ``MaxSessions`` are left for.
"""

import ast
import contextlib
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import detail
from detail.executor import local, ssh_transport
from detail.executor.remote_shepherd import _Reconnect

REFUSED = (
    "mux_client_request_session: session request failed: Session open refused by peer\n"
    "sagolden@tillicum.hyak.uw.edu: Permission denied (gssapi-keyex,gssapi-with-mic,"
    "keyboard-interactive)."
)


class _Waited(Exception):
    """Raised by a ``_slot_sleep`` stand-in: the caller found no free slot and began to wait."""


def _never_wait(_seconds):
    raise _Waited


def _wait_at_most(limit_s):
    """A ``_slot_sleep`` stand-in that really waits, and raises ``_Waited`` past ``limit_s``."""
    deadline = time.monotonic() + limit_s

    def wait(seconds):
        if time.monotonic() > deadline:
            raise _Waited
        time.sleep(seconds)

    return wait


def _hold(host, count, *, transfer=False):
    stack = contextlib.ExitStack()
    for _ in range(count):
        stack.enter_context(ssh_transport.session_slot(host, transfer=transfer))
    return stack


def _peak_concurrency(monkeypatch, launch, n_threads=20):
    """Run ``launch`` on ``n_threads`` threads against a fake ssh that stays busy for a moment,
    and return the most sessions that were open at once."""
    state = {"open": 0, "peak": 0}
    lock = threading.Lock()

    def fake_run(argv, **kwargs):
        with lock:
            state["open"] += 1
            state["peak"] = max(state["peak"], state["open"])
        time.sleep(0.02)
        with lock:
            state["open"] -= 1
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ssh_transport, "SLOT_WAIT_S", 0.005)
    threads = [threading.Thread(target=launch) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a command never got a session"
    return state["peak"]


def test_commands_never_hold_more_sessions_than_the_cap(monkeypatch):
    peak = _peak_concurrency(
        monkeypatch, lambda: ssh_transport.run_remote("tillicum", "squeue")
    )
    assert 1 < peak <= ssh_transport.MAX_SESSIONS


def test_transfers_never_hold_more_than_their_share(monkeypatch):
    peak = _peak_concurrency(
        monkeypatch,
        lambda: ssh_transport._run(
            ["rsync", "-az", "/local/", "tillicum:/remote/"], timeout_s=None
        ),
    )
    assert 1 < peak <= ssh_transport.MAX_TRANSFER_SESSIONS


def test_a_short_command_gets_in_while_transfers_fill_their_share(monkeypatch):
    """Eight uploads must not hold up a poll or a cancel for as long as they run."""
    monkeypatch.setattr(ssh_transport, "_slot_sleep", _never_wait)
    with _hold("tillicum", ssh_transport.MAX_TRANSFER_SESSIONS, transfer=True):
        with ssh_transport.session_slot("tillicum"):
            pass  # no wait: a slot is kept for it
        with pytest.raises(_Waited):
            with ssh_transport.session_slot("tillicum", transfer=True):
                pass


def test_a_command_waits_when_every_slot_is_taken(monkeypatch):
    monkeypatch.setattr(ssh_transport, "_slot_sleep", _never_wait)
    with _hold("tillicum", ssh_transport.MAX_SESSIONS):
        with pytest.raises(_Waited):
            with ssh_transport.session_slot("tillicum"):
                pass
    with ssh_transport.session_slot("tillicum"):
        pass  # released when the holder left


def test_hosts_have_their_own_slots(monkeypatch):
    monkeypatch.setattr(ssh_transport, "_slot_sleep", _never_wait)
    with _hold("tillicum", ssh_transport.MAX_SESSIONS):
        with ssh_transport.session_slot("klone"):
            pass


_CHILD = """
import contextlib, sys, time
from pathlib import Path
from detail.executor import ssh_transport
ssh_transport._SLOT_ROOT = Path(sys.argv[1])
stack = contextlib.ExitStack()
for _ in range(ssh_transport.MAX_SESSIONS):
    stack.enter_context(ssh_transport.session_slot("tillicum"))
print("held", flush=True)
time.sleep(120)
"""


def test_slots_are_shared_across_processes_and_freed_when_a_holder_dies(monkeypatch):
    """The cap is only worth anything across processes: each shepherd is its own. And a shepherd
    that is killed holding a slot must not take it with it, or every later run waits forever.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(ssh_transport._SLOT_ROOT)],
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path(detail.__file__).parent.parent,
        **local.new_process_group_kwargs(),
    )
    try:
        assert proc.stdout.readline().strip() == "held"
        monkeypatch.setattr(ssh_transport, "_slot_sleep", _never_wait)
        with pytest.raises(_Waited):
            with ssh_transport.session_slot("tillicum"):
                pass
    finally:
        proc.kill()
        proc.wait(timeout=30)
        proc.stdout.close()
    # Windows frees a dead process's byte-range locks after it exits, not as it exits ("the time
    # it takes ... depends upon available system resources", LockFileEx), so a holder killed
    # under a loaded suite can still hold its slot for a moment: 14 full runs failed here from
    # 2026-11-05 to 11-06, never a run of this file alone. Production waits it out by sleeping.
    monkeypatch.setattr(ssh_transport, "_slot_sleep", _wait_at_most(30))
    with ssh_transport.session_slot("tillicum"):
        pass


def test_a_refused_ask_is_reported_as_a_full_connection(monkeypatch):
    monkeypatch.setattr(ssh_transport, "_sleep", lambda s: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **k: subprocess.CompletedProcess(argv, 255, "", REFUSED),
    )
    ssh_transport.run_remote("tillicum", "squeue", check=False, refused_retries=0)
    assert ssh_transport.connection_full("tillicum")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **k: subprocess.CompletedProcess(
            argv, 255, "", "Connection reset"
        ),
    )
    ssh_transport.run_remote("tillicum", "squeue", check=False, refused_retries=0)
    assert not ssh_transport.connection_full("tillicum")


def _target():
    from types import SimpleNamespace

    return SimpleNamespace(name="tillicum", host="tillicum")


def test_the_wait_on_a_full_connection_does_not_say_log_in_again(monkeypatch, tmp_path):
    """The message sent the person to fix a login that was working."""
    published = []

    class _Transfer:
        def phase(self, message):
            published.append(message)

    monkeypatch.setattr(ssh_transport, "connection_full", lambda host: True)
    reconnect = _Reconnect(tmp_path, _target(), _Transfer())
    reconnect.lost(None)
    assert "full" in published[-1] and "log in again" not in published[-1]

    monkeypatch.setattr(ssh_transport, "connection_full", lambda host: False)
    reconnect.lost(None)
    assert "log in again" in published[-1]


# -- every launch holds a slot ---------------------------------------------------------------- #

_SSH_FAMILY = {"ssh", "rsync", "scp"}
_LAUNCHERS = {"run", "Popen", "check_output", "call", "check_call"}

#: ``(file, function)`` launches that deliberately hold no slot, with the reason.
_UNSLOTTED = {
    ("executor/ssh_transport.py", "require_responsive"): (
        "`ssh -O check` is a message to the master process; it opens no session"
    ),
}


def _launches_ssh(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            first = node.elts[0]
            if isinstance(first, ast.Constant) and first.value in _SSH_FAMILY:
                return True
    return False


def _is_slot(with_node: ast.With) -> bool:
    for item in with_node.items:
        call = item.context_expr
        if isinstance(call, ast.Call):
            func = call.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name == "session_slot":
                return True
    return False


def _unslotted_launches() -> set[tuple[str, str]]:
    root = Path(detail.__file__).parent
    found: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        if not _launches_ssh(tree):
            continue
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr in _LAUNCHERS
            ):
                continue
            slotted, function, up = False, "<module>", parents.get(node)
            while up is not None:
                if isinstance(up, ast.With) and _is_slot(up):
                    slotted = True
                if function == "<module>" and isinstance(
                    up, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    function = up.name
                up = parents.get(up)
            if not slotted:
                found.add((path.relative_to(root).as_posix(), function))
    return found


def test_every_ssh_rsync_and_scp_launch_holds_a_session_slot():
    """Fail-closed: a new launch beside an ssh-family argv must take a slot or be listed in
    ``_UNSLOTTED`` with its reason. One launch outside the cap is how the excess comes back.
    """
    unslotted = _unslotted_launches()
    assert unslotted - set(_UNSLOTTED) == set()


def test_every_unslotted_exemption_still_excuses_something():
    assert set(_UNSLOTTED) - _unslotted_launches() == set()


def test_the_scan_finds_an_unslotted_launch(tmp_path, monkeypatch):
    """The guard's own negative case: a module that runs ssh with no slot is reported."""
    package = tmp_path / "detail"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "bad.py").write_text(
        "import subprocess\n\n"
        "def ask(host):\n"
        "    return subprocess.run(['ssh', host, 'true'])\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(detail, "__file__", str(package / "__init__.py"))
    assert _unslotted_launches() == {("bad.py", "ask")}


def test_a_slot_is_a_real_os_lock(tmp_path):
    """Two descriptors on one slot file cannot both hold it, even in one process: the property
    the threads above and the shepherds in separate processes both rely on."""
    first = ssh_transport._try_lock(tmp_path / "0.lock")
    assert first is not None
    try:
        assert ssh_transport._try_lock(tmp_path / "0.lock") is None
    finally:
        ssh_transport._unlock(first)
    second = ssh_transport._try_lock(tmp_path / "0.lock")
    assert second is not None
    ssh_transport._unlock(second)
