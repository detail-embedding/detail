"""Bounded waits for the thread-based concurrency tests.

``threading.Barrier.wait()`` and ``Thread.join()`` both block forever by default, and the
lock tests are exactly the shape that turns that into a stopped suite: N threads meet at a
barrier and then race for a lock. A thread that raises *before* it reaches the barrier
leaves the other N-1 parties waiting for a party that will never arrive, and the main
thread's ``join()`` then waits on them. Nothing fails — the run simply stops, which reads
as "the machine is slow" rather than as the assertion failure it actually is.

The bounds here are hang detectors, not race windows. These tests rendezvous and finish in
milliseconds, so the timeout is three orders of magnitude of headroom; exceeding it means a
thread is not coming back. It sits well under ``faulthandler_timeout`` (pyproject) so the
failure is a named assertion in one test rather than the whole run being stopped.
"""

import threading

#: Slack for every party to reach a rendezvous, and for a joined thread to finish.
WAIT_TIMEOUT_S = 30.0


def rendezvous(barrier: threading.Barrier) -> None:
    """``barrier.wait()`` that raises instead of parking the run.

    A timeout raises ``BrokenBarrierError`` *and* breaks the barrier, so the parties still
    waiting are released with the same error rather than inheriting the wait — one late
    thread fails the test once instead of hanging every thread behind it.
    """
    barrier.wait(timeout=WAIT_TIMEOUT_S)


def join_all(threads: list[threading.Thread]) -> None:
    """Join every thread, and say which ones never returned instead of blocking on them."""
    for thread in threads:
        thread.join(timeout=WAIT_TIMEOUT_S)
    stuck = [t.name for t in threads if t.is_alive()]
    assert not stuck, f"threads did not finish within {WAIT_TIMEOUT_S}s: {stuck}"
