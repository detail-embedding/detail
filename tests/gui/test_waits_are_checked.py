"""A bounded wait in a test fails where it ran out, not five assertions later.

**The class: a wait whose answer is dropped.** ``drain_until`` used to return False when its
predicate never held, and 12 of its callers (plus the fixtures and module wrappers built on it)
called it as a statement. On an idle box the predicate always held first, so nothing showed.
Inside a full gate at load ~40 on 32 cores, the Keypoints manager's off-thread library read took
longer than the 10 s wait: ``_settle`` returned, the test carried on with ``page._reading`` still
True, the Edit and Export buttons were still disabled, and five tests failed on an empty table,
``None == '1 example set'`` and ``no row sk_01`` — naming the symptom, never the wait. All five
passed alone. Reproduced on ``main`` by making ``_read_library`` sleep 11 s: the same five
failed with the same messages.

The same shape sat in three other spellings: ``QThread.wait(ms)`` as a statement
(``test_busy._drain``, ``test_tools``'s gather worker), ``QThreadPool.waitForDone(ms)`` as a
statement before an assertion (``test_scrubber``), and a local spinner annotated ``-> bool``.

**The rule.** A wait either raises naming what never happened, or its answer is used.
``drain_until`` now raises, quoting the predicate's source; ``raise_on_timeout=False`` is the
explicit opt-out for a caller that asks *whether* something happens, and it must use the answer.
A fixed-time pump — formerly ``drain_until(qapp, lambda: False, ms)``, a wait for nothing whose
timeout was the point — is ``pump_for``. A thread is joined through ``tests.waits.finished``,
a pool drained through ``drain_thread_pool``.

This is not a longer timeout. A wait that runs out still fails; it fails *there*, saying what it
waited for. ``tests/test_timeouts_are_failsafes.py`` is the sibling rule about how long a wait may
be.

**Where the scan stops.** It sees an answer dropped by a statement-level call. It does not see an
answer assigned to a name nobody reads, a local helper that returns a bool without a ``-> bool``
annotation, or a ``.wait(timeout)`` on a receiver whose name says neither *worker* nor *thread* —
``.wait`` is also ``Popen.wait`` (raises on timeout), ``threading.Event.wait`` (the release gates
inside fake workers, whose timeout is the worker's escape hatch rather than the test's check) and
``run.wait()`` (no timeout). Indexed in ``docs/recorded_classes.md``.
"""

import ast
import pathlib

import pytest

from tests.gui.conftest import drain_until
from tests.gui.conftest import pump_for

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Qt waits that answer with a bool and never raise.
_QT_ANSWERING_WAITS = frozenset({"waitForDone", "qWaitFor"})

#: The helpers that raise when their wait runs out, so calling one as a statement drops nothing.
_RAISING = frozenset({"drain_until"})


def _opts_out(call: ast.Call) -> bool:
    return any(
        kw.arg == "raise_on_timeout"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
        for kw in call.keywords
    )


def _callee(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _receiver(call: ast.Call) -> str:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return ""
    target = func.value
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def _answering_helpers(tree: ast.Module) -> set[str]:
    """Module functions annotated ``-> bool`` that do not raise through a raising wait.

    A wrapper whose every ``return`` hands back a raising helper's call (``return
    drain_until(qapp, predicate, timeout_ms)``) raises too, so dropping its answer drops
    nothing; resolved to a fixed point so a wrapper of a wrapper counts.
    """
    defs = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and isinstance(node.returns, ast.Name)
        and node.returns.id == "bool"
    }
    raising = set(_RAISING)
    changed = True
    while changed:
        changed = False
        for name, node in defs.items():
            if name in raising:
                continue
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            if returns and all(
                isinstance(r.value, ast.Call)
                and _callee(r.value) in raising
                and not _opts_out(r.value)
                for r in returns
            ):
                raising.add(name)
                changed = True
    return set(defs) - raising


def _dropped(source: str) -> list[tuple[int, str]]:
    """Every statement-level call in ``source`` that throws a wait's answer away."""
    tree = ast.parse(source)
    answering = _answering_helpers(tree)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        name = _callee(call)
        if _opts_out(call):
            why = f"{name}(..., raise_on_timeout=False)"
        elif name in _QT_ANSWERING_WAITS:
            why = f"{name}(...)"
        elif isinstance(call.func, ast.Name) and name in answering:
            why = f"{name}(...), which returns a bool"
        elif (
            name == "wait"
            and (call.args or call.keywords)
            and any(word in _receiver(call).lower() for word in ("worker", "thread"))
        ):
            why = f"{_receiver(call)}.wait(timeout)"
        else:
            continue
        found.append((node.lineno, why))
    return found


def test_no_test_drops_a_waits_answer():
    """Every wait under ``tests/`` raises when it runs out, or its answer is used."""
    hits = []
    for path in sorted(REPO.glob("tests/**/*.py")):
        rel = path.relative_to(REPO).as_posix()
        for line, why in _dropped(path.read_text(encoding="utf-8-sig")):
            hits.append(f"{rel}:{line} {why}")
    assert not hits, (
        f"{len(hits)} wait(s) drop their answer, so a test that runs out carries on against a "
        "state it never reached and fails later, naming the symptom: "
        + "; ".join(hits)
        + ". Use `drain_until` (it raises), `assert` the answer, `pump_for` for a fixed pump, "
        "`tests.waits.finished` for a thread, `drain_thread_pool` for the pool."
    )


@pytest.mark.parametrize(
    "source",
    [
        "drain_until(qapp, lambda: page.ready, raise_on_timeout=False)\n",
        "QThreadPool.globalInstance().waitForDone(2000)\n",
        "QTest.qWaitFor(lambda: done)\n",
        "matrix._gather_worker.wait(10_000)\n",
        "worker.wait(timeout_ms)\n",
        "def _spin(qapp) -> bool:\n    return False\n\n\n_spin(qapp)\n",
        # A wrapper that opts out answers, so dropping *its* answer is the bug too.
        "def _drain(qapp, p) -> bool:\n"
        "    return drain_until(qapp, p, raise_on_timeout=False)\n\n\n"
        "_drain(qapp, lambda: True)\n",
    ],
)
def test_the_scan_would_catch_a_dropped_answer(source):
    assert _dropped(source), source


@pytest.mark.parametrize(
    "source",
    [
        "drain_until(qapp, lambda: page.ready)\n",
        "assert not drain_until(qapp, lambda: x, 500, raise_on_timeout=False)\n",
        "pump_for(qapp, 300)\n",
        "proc.wait(timeout=5)\n",
        "gate.wait(5)\n",
        "run.wait()\n",
        "def _drain(qapp, p, t=10_000) -> bool:\n"
        "    return drain_until(qapp, p, t)\n\n\n"
        "def _quiet(qapp) -> bool:\n    return _drain(qapp, lambda: True)\n\n\n"
        "_drain(qapp, lambda: True)\n_quiet(qapp)\n",
    ],
)
def test_the_scan_leaves_a_checked_wait_alone(source):
    assert not _dropped(source), source


def test_a_wait_that_runs_out_raises_naming_what_it_waited_for(qapp):
    """The mechanism half: dropping ``drain_until``'s answer can no longer drop anything."""
    ready = False
    with pytest.raises(AssertionError, match=r"never held: .*lambda: ready"):
        drain_until(qapp, lambda: ready, 50)


def test_the_opt_out_answers_instead(qapp):
    assert drain_until(qapp, lambda: False, 30, raise_on_timeout=False) is False
    assert drain_until(qapp, lambda: True, 30, raise_on_timeout=False) is True


def test_pump_for_runs_the_loop_and_waits_on_nothing(qapp):
    from detail.gui.qt import QTimer

    fired = []
    QTimer.singleShot(0, lambda: fired.append(1))
    pump_for(qapp, 30)
    assert fired == [1]
