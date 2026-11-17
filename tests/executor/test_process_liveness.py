"""Asking whether a process is alive must not send it anything.

**The class: a liveness probe that is a signal on one platform and a query on the other.**
``os.kill(pid, 0)`` is the POSIX idiom — signal 0 is "check, deliver nothing". On Windows
there is no signal 0: ``signal.CTRL_C_EVENT == 0``, so the same line is
``GenerateConsoleCtrlEvent(CTRL_C_EVENT, pid)`` — a Ctrl-C sent to process group ``pid``.

**What it cost on Windows (PR #436).** ``executor/local.py::_alive`` used it,
and with a worker spawned the way real ones are (``new_process_group_kwargs``):

=====================================  ==================  =====
worker                                 ``os.kill(pid,0)``  truth
=====================================  ==================  =====
running                                alive               alive
exited, a ``Popen`` handle still open  **alive**           dead
exited, every handle closed            **alive**           dead
=====================================  ==================  =====

So on Windows no worker was ever seen to die: a reattached run's progress stream never ended
(``test_stream_progress_ends_when_a_reattached_worker_dies`` timed out), a dead run was never
reconciled to failed, and ``cancel_run_dir`` always waited out its hard timeout. Aimed at a
child that shared the caller's console, the Ctrl-C was broadcast to the whole console — the
caller, and the shell that launched it, took a ``KeyboardInterrupt``. The PR that fixed it
first explained it as a stale ``Popen`` handle; the table's last row is what refuted that.

**How it hid.** CI is Linux, where the line is correct. Nothing is wrong at the line where it is
written, and on Windows the symptom is a hang somewhere else.

**The rule.** ``os.kill(<pid>, 0)`` appears only on a path Windows cannot reach — today that is
``_alive``'s POSIX branch, after its ``os.name == "nt"`` early return, which asks
``GetExitCodeProcess`` instead. The scan below enforces the first half; the behaviour tests
enforce the second, and on Windows fail against the old ``_alive``.

Also here, because it is the same pair of processes: ``_is_our_worker`` recognising a worker
whose run dir is not ASCII. That one is the text-decoding class of
``tests/test_encoding_guard.py`` (a subprocess decoded on the locale codec); these are its
behaviour tests on the site where it cost something — a live worker read as "not ours".
"""

import ast
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

from detail.executor import local

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOTS = ("detail", "tests", "scripts")

#: ``(file, enclosing function)`` where ``os.kill(pid, 0)`` is allowed, and why. A claim:
#: ``test_every_allowed_site_still_matches`` fails on one that excuses nothing.
ALLOWED = {
    (
        "detail/executor/local.py",
        "_alive",
    ): "the POSIX branch, after the nt early return",
    (
        "scripts/suite_progress.py",
        "_pid_alive",
    ): "stdlib-only for the status line, so it cannot import executor.local; after the nt return",
}


def kill_zero_calls(tree: ast.AST):
    """``(call, enclosing function name)`` for every ``os.kill(<x>, 0)`` in a module."""
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "os.kill"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == 0
        ):
            continue
        scope, up = "<module>", parents.get(id(node))
        while up is not None:
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = up.name
                break
            up = parents.get(id(up))
        yield node, scope


def _offenders(root: pathlib.Path) -> list[str]:
    out = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for call, scope in kill_zero_calls(tree):
            if (rel, scope) not in ALLOWED:
                out.append(f"{rel}:{call.lineno} in {scope}(): {ast.unparse(call)}")
    return out


@pytest.mark.parametrize("root", ROOTS)
def test_no_liveness_probe_signals_on_windows(root):
    offenders = _offenders(REPO / root)
    assert not offenders, (
        "os.kill(pid, 0) sends Ctrl-C on Windows (signal.CTRL_C_EVENT == 0) and reports a "
        "dead process alive — use executor.local.pid_alive:\n" + "\n".join(offenders)
    )


def test_the_scan_would_catch_one():
    # Guard the guard, in the spellings a person writes.
    source = "\n".join(
        [
            "import os",
            "def check(pid):",
            "    os.kill(pid, 0)",
            "def other(p):",
            "    return os.kill(p.pid, 0)",
            "os.kill(1, 0)",
            "os.kill(pid, signal.SIGTERM)",  # a real signal is not a probe
        ]
    )
    found = sorted(scope for _, scope in kill_zero_calls(ast.parse(source)))
    assert found == ["<module>", "check", "other"]


def test_every_allowed_site_still_matches():
    for (rel, scope), reason in ALLOWED.items():
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8-sig"))
        scopes = {s for _, s in kill_zero_calls(tree)}
        assert scope in scopes, f"{rel}::{scope} no longer probes; drop it ({reason})"


def _is_nt_test(node: ast.AST) -> bool:
    return ast.unparse(node).replace("'", '"') == 'os.name == "nt"'


def unguarded_probes(func: ast.FunctionDef) -> list[int]:
    """Lines of ``os.kill(<x>, 0)`` in ``func`` not preceded by an ``os.name == "nt"`` branch
    that returns — the one position where Windows cannot reach the probe.

    The allowlist names a function, and a function is not a position: deleting `_alive`'s
    Windows branch left the probe in the allowed function and every test green on Linux, the
    only platform CI runs. So the claim the allowlist makes — "after the nt early return" — is
    checked as a claim about where the call sits.
    """
    probes = [call.lineno for call, _ in kill_zero_calls(func)]
    guarded_after = None
    for stmt in func.body:
        if (
            isinstance(stmt, ast.If)
            and _is_nt_test(stmt.test)
            and any(isinstance(n, ast.Return) for n in ast.walk(stmt))
        ):
            guarded_after = stmt.end_lineno
            break
    return [line for line in probes if guarded_after is None or line <= guarded_after]


def test_every_allowed_probe_sits_behind_the_windows_return():
    for (rel, scope), reason in ALLOWED.items():
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8-sig"))
        (func,) = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == scope
        ]
        exposed = unguarded_probes(func)
        assert not exposed, (
            f"{rel}::{scope} is allowed os.kill(pid, 0) because {reason!r}, but the probe at "
            f"line(s) {exposed} is reachable on Windows — the early return in front of it is "
            "gone, and every Linux test still passes"
        )


def test_the_position_check_would_catch_a_missing_windows_branch():
    def _func(src: str) -> ast.FunctionDef:
        return ast.parse(src).body[0]

    guarded = _func(
        "def _alive(pid):\n"
        "    if os.name == 'nt':\n"
        "        return query(pid)\n"
        "    os.kill(pid, 0)\n"
    )
    assert unguarded_probes(guarded) == []
    assert unguarded_probes(_func("def _alive(pid):\n    os.kill(pid, 0)\n")) == [2]
    # A Windows branch that falls through protects nothing.
    falls_through = _func(
        "def _alive(pid):\n"
        "    if os.name == 'nt':\n"
        "        log(pid)\n"
        "    os.kill(pid, 0)\n"
    )
    assert unguarded_probes(falls_through) == [4]


def _sleeper(*extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", *extra],
        **local.new_process_group_kwargs(),
    )


@pytest.fixture
def interrupts():
    """Every SIGINT this process receives while the test runs — the broadcast Ctrl-C."""
    seen: list[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, _frame: seen.append(signum))
    yield seen
    signal.signal(signal.SIGINT, previous)


def test_a_worker_that_exited_reads_dead(interrupts):
    """The bug. On Windows the old probe answered "alive" in both states below."""
    proc = _sleeper()
    try:
        assert local._alive(proc.pid), "the negative case: a running worker is alive"
    finally:
        proc.kill()
        proc.wait()
    assert not local._alive(proc.pid), "exited, but its Popen handle is still open"
    pid = proc.pid
    del proc  # the handle closes with the object
    assert not local.pid_alive(pid), "exited, and nothing holds a handle to it"
    time.sleep(0.2)  # a console control event is delivered on a new thread
    assert not interrupts, "asking whether a process is alive sent this one a Ctrl-C"


def test_this_process_reads_alive():
    assert local._alive(os.getpid())


RUN_DIR_NAMES = ["plain_run"]
if os.name == "nt":
    # POSIX `ps` under a C locale escapes non-ASCII itself, which is not this bug.
    RUN_DIR_NAMES += ["café_run", "run_”quote"]


@pytest.mark.parametrize("name", RUN_DIR_NAMES)
def test_a_live_worker_is_recognised_whatever_its_run_dir_is_called(tmp_path, name):
    """Measured before the fix: ``café_run`` read "not ours" (mojibake never matches) and
    ``run_”quote`` raised ``TypeError`` (a byte cp1252 leaves undefined made ``stdout``
    None)."""
    run_dir = tmp_path / name
    proc = _sleeper(str(run_dir))
    try:
        assert local._is_our_worker(proc.pid, run_dir)
        assert not local._is_our_worker(
            proc.pid, tmp_path / "another_run"
        ), "the negative case: a live process with a different run dir is not ours"
    finally:
        proc.kill()
        proc.wait()


# --------------------------------------------------------------------------- #
# a coordinator this process spawned and never reaped (issue #28, finding 9)
# --------------------------------------------------------------------------- #
def test_an_exited_coordinator_this_process_spawned_reads_dead(tmp_path):
    """**The class: a liveness probe that answers "exists" for a process that has exited.**

    A coordinator is a child of the process that launched it (`spawn_coordinator`). Once it
    dies — an OOM kill, a native crash — it stays a *zombie* until its parent waits on it, and
    ``os.kill(pid, 0)`` succeeds on a zombie. So the launcher's own reattach poll
    (`AttachedRun.progress`) never counted a dead strike, never reconciled, and `wait()` sat in
    its sleep forever while the worker's ``job.json`` already said ``complete``. Any other
    process (a GUI reopen, ``process attach``) saw it dead at once, which is how it hid.

    The rule: for a pid this process spawned, the ``Popen`` handle is the authority —
    ``poll()`` reaps the zombie and says it exited. The fixture is the real path: spawned
    through `spawn_coordinator` (so through `new_process_group_kwargs`), exiting at once, and
    **not** reaped by the test. Against a plain ``os.kill(pid, 0)`` probe this fails.
    """
    pid = local.spawn_coordinator(tmp_path, (), entrypoint="this")
    deadline = time.monotonic() + 10
    # Wait for the child to exit without reaping it: /proc shows a zombie as state Z.
    stat = pathlib.Path(f"/proc/{pid}/stat")
    while time.monotonic() < deadline:
        if (
            os.name == "nt"
            or not stat.exists()
            or stat.read_text(encoding="utf-8-sig").split()[2] == "Z"
        ):
            break
        time.sleep(0.05)
    try:
        assert local.pid_alive(pid) is False
        assert local.coordinator_alive(tmp_path, pid) is False
    finally:
        local.reap_coordinator(pid)
