"""A timeout in a test is a failsafe, never a budget.

**The class.** `thread.wait(3000)` reads as "this should take under three seconds". That is a
claim about the *machine*, not about the code — and on a box where several agent sessions run the
suite at once, it measures whoever else is busy. Worse, `QThread.wait` returns a bool that 25 of
this suite's 27 waits ignored, so a timeout was **silent**: the thread kept running, the next
assertion failed on half-built state, and the message named a symptom three lines from the cause.

Three failures traced to this in one session, each passing alone and failing inside a loaded run:
a 5s scan budget that hit 5.08s, a 3s worker wait that timed out, and a 2s spin loop that ran out
while an off-thread decode was still going. None was a real defect; all three cost a 25-minute
suite run to diagnose.

**The rule.** Wait through `tests/waits.py`, which uses one generous timeout a working thread
never reaches and asserts what did not finish when it does. A test that takes 20ms still takes
20ms — `wait` returns as soon as the thread is done, so a failsafe costs nothing.

A *short* wait is a different thing and stays: `wait(5)` or `wait(30)` means "let a queued
delivery land", not "wait until this is finished". The threshold below is what separates them.
"""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: At or above this, a wait means "until it is done" and must be a failsafe. Below it, the wait is
#: a deliberate yield and is left alone.
COMPLETION_MS = 1000

#: Waits this long or longer are failsafes rather than budgets — long enough that only a hang
#: reaches them. `tests/waits.py` uses 30s.
FAILSAFE_MS = 10_000

#: Files whose long waits are deliberate and not covered by the helper, with why.
ALLOWED: dict[str, str] = {
    "tests/executor/test_local.py": (
        "waits on real subprocesses it spawned, not on a QThread — `waits.finished` takes a "
        "thread, and a process's own timeout is part of what these tests are checking"
    ),
}


def _fixed_waits():
    """Every ``something.wait(<literal ms>)`` under ``tests/``, with where it is."""
    out = []
    for path in sorted(REPO.glob("tests/**/*.py")):
        rel = str(path.relative_to(REPO))
        if rel in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "wait"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, int)
            ):
                out.append((f"{rel}:{node.lineno}", node.args[0].value))
    return out


def test_no_test_waits_on_a_budget():
    """A wait long enough to mean "until it is done" has to be long enough to mean "or it hung"."""
    budgets = [
        (where, ms) for where, ms in _fixed_waits() if COMPLETION_MS <= ms < FAILSAFE_MS
    ]
    assert not budgets, (
        f"{len(budgets)} wait(s) set a timeout that reads as a budget: "
        + "; ".join(f"{where} waits {ms}ms" for where, ms in budgets)
        + f". Under {FAILSAFE_MS}ms this measures the machine, and on a box several agent "
        "sessions share that is whoever else is busy. Use `tests.waits.finished`, which waits "
        "generously and says what did not finish."
    )


def test_the_scan_would_catch_one():
    """Guard the guard: a scan matching nothing passes forever."""
    tree = ast.parse("worker.wait(3000)\n")
    call = tree.body[0].value
    assert getattr(call.func, "attr", "") == "wait"
    assert COMPLETION_MS <= call.args[0].value < FAILSAFE_MS


@pytest.mark.parametrize("path", sorted(ALLOWED))
def test_every_allowance_names_a_file_that_exists(path):
    """An allowance for a file nobody has any more is a claim that has stopped being true."""
    assert (REPO / path).is_file(), f"{path} is allowed a fixed wait and does not exist"
