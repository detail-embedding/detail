"""A pytest started by a test runs through ``tests/subprocesses.py``, bounded and killed as a tree.

**The class: a test waits on a process tree with no bound, or with a bound a grandchild defeats.**
A nested ``pytest -n 2`` is a controller plus xdist workers that inherit its output pipes.
``subprocess.run(cmd, capture_output=True)`` waits for every holder of those pipes to let go;
with no ``timeout`` that is forever, and on Windows a ``timeout`` does not help either, because on
expiry ``subprocess.run`` kills the controller alone and then calls ``communicate()`` with no
timeout, still reading pipes that a surviving worker holds. ``Popen.wait(timeout=…)`` raises and
leaves the tree running behind it.

**How it showed.** On 2026-11-06 a gate run on the Windows box stopped at about 12,110 of 12,161
tests. Its one stack dump is ``tests/scripts/test_suite_progress.py::
test_an_xdist_run_counts_every_test`` in ``subprocess.run`` → ``communicate`` on a nested
``pytest -n 2``: the 120s watchdog killed that xdist worker, xdist reported the test as failed,
and the run never finished. A later gate also failed this test and hung (its log is gone, so how
it failed is not known). Each held the machine lock while it sat, so other sessions queued behind
it, and a PR merged without its gate. Why the run stayed stuck *after* the worker was killed is not
established: reproduced on this box with the full collection, a killed worker is replaced and the
run ends. Taking the trigger away is what this guard does.

**The rule.** A test that starts pytest does it with ``run_tree`` / ``start_tree`` + ``wait_tree``,
which put the child in its own process group, bound every wait under the watchdog, and kill the
whole tree before raising. The failure is then a ``TimeoutExpired`` in the test that started it.

**Where it stops.** The scan keys on a nested pytest argv, the one shape here that starts a tree
of its own. ``tests/scripts/test_with_suite_lock.py`` starts the lock script, which starts one
command: deliberately not, since every wait there is already ``communicate(timeout=30)`` and one
test leaves the command running on purpose. Single processes (``python -c``, ``git``, ``sh``) are
deliberately not: ``subprocess.run(timeout=…)`` does bound a child with no children.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ROOTS = ("tests",)
_STARTS = {"run", "Popen", "call", "check_call", "check_output"}
#: Functions that start a nested pytest today. A scan that finds fewer has stopped matching.
_KNOWN_SITES = 3


def _builds_pytest_argv(node: ast.AST) -> bool:
    for lst in ast.walk(node):
        if isinstance(lst, ast.List):
            values = [e.value for e in lst.elts if isinstance(e, ast.Constant)]
            if any(a == "-m" and b == "pytest" for a, b in zip(values, values[1:])):
                return True
    return False


def _direct_starts(node: ast.AST) -> list[int]:
    return [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _STARTS
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
    ]


def scan(source: str) -> tuple[int, list[int]]:
    """``(functions that start a nested pytest, lines starting one directly)``."""
    sites, bad = 0, []
    for func in ast.walk(ast.parse(source)):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _builds_pytest_argv(func):
                sites += 1
                bad += _direct_starts(func)
    return sites, bad


def test_every_nested_pytest_is_bounded_and_killed_as_a_tree():
    sites, offenders = 0, []
    for root in _ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            found, bad = scan(path.read_text(encoding="utf-8-sig"))
            sites += found
            rel = path.relative_to(REPO).as_posix()
            offenders += [f"{rel}:{line}" for line in bad]
    assert not offenders, (
        "a test starts pytest with a bare subprocess call, which can wait forever on the "
        "nested run's workers (and on Windows a timeout does not stop that). Use "
        f"tests.subprocesses.run_tree / start_tree + wait_tree: {offenders}"
    )
    assert (
        sites >= _KNOWN_SITES
    ), f"found {sites} nested-pytest starts, expected {_KNOWN_SITES}"


def test_the_scan_rejects_a_bare_nested_run():
    bare = (
        "import subprocess, sys\n"
        "def _run(tmp_path):\n"
        "    cmd = [sys.executable, '-m', 'pytest', '-q']\n"
        "    return subprocess.run(cmd, capture_output=True, timeout=60)\n"
    )
    assert scan(bare) == (1, [4])
    bounded = bare.replace("subprocess.run(", "run_tree(")
    assert scan(bounded) == (1, [])
