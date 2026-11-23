"""Waiting for off-thread work in tests, without turning the clock into the assertion.

**A timeout is a failsafe, not a budget.** `QThread.wait(3000)` reads as "this should take under
three seconds", and that is a claim about the machine rather than about the code — on a box
several agent sessions share it measures whoever else is busy. Worse, 25 of the 27 waits in this
suite ignored the returned bool, so a timeout was *silent*: the thread was still running, the next
assertion failed on half-built state, and the message named the symptom rather than the wait.

So: one generous timeout that a working thread never reaches, and an assertion that says what did
not finish when it does. A test that passes in 20ms still passes in 20ms — `wait` returns as soon
as the thread is done.

Three failures traced to fixed budgets in one session, each passing alone and failing inside a
loaded suite: a 5s scan budget hit 5.08s, a 3s worker wait timed out, and a 2s spin loop ran out
while an off-thread decode was still going.
"""

#: Long enough that a thread doing its job is never cut off, short enough that a deadlocked one
#: is still reported rather than hanging the run. The suite's own watchdog (`faulthandler_timeout`,
#: 120s) is the backstop behind this.
WORKER_TIMEOUT_MS = 30_000


def finished(thread, what: str, timeout_ms: int = WORKER_TIMEOUT_MS) -> None:
    """Wait for ``thread``, and fail naming it if it does not finish.

    ``what`` is what the thread was doing, because the failure this replaces read as an unrelated
    assertion three lines later.
    """
    assert thread.wait(timeout_ms), (
        f"{what} did not finish within {timeout_ms / 1000:.0f}s. That is a hang or a deadlock — "
        "this timeout is a failsafe, not a budget, so a slow machine does not reach it."
    )


def until(qapp, predicate, what: str, timeout_ms: int = WORKER_TIMEOUT_MS) -> None:
    """Pump the event loop until ``predicate`` holds, and fail naming it if it never does.

    For work that reports through signals rather than a joinable thread. Bounded by wall clock
    rather than by a spin count: `processEvents` returns immediately when nothing is queued, so N
    bare pumps can burn through in microseconds while the worker is still running — green on an
    idle box and flaky inside a loaded suite.
    """
    from detail.gui.qt import QTest

    waited = 0
    while waited < timeout_ms:
        qapp.processEvents()
        if predicate():
            return
        QTest.qWait(10)
        waited += 10
    qapp.processEvents()
    assert (
        predicate()
    ), f"{what} never happened within {timeout_ms / 1000:.0f}s — a hang, not a slow machine."
