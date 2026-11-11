"""Reaping a finished run's coordinator is idempotent under concurrent readers.

The class: **a cleanup written as an action on shared state — "remove this one" — reached
from more than one thread.** ``reap_coordinator`` snapshotted the module's list of spawned
coordinators, waited on the matching handle, then ``list.remove``-d it. Each step is right for
one caller; the function has as many callers as a run has readers, and the GUI starts a poller
thread per *opening* of a run page and stops none of them. Two pollers of one run see the same
terminal write in the same 0.25 s poll, both snapshot the handle, both return from ``wait``,
and the second ``remove`` raises ``ValueError: list.remove(x): x not in list`` — out of
``AttachedRun.progress``, on a ``QThread``, which PyQt turns into a process abort. The
sibling ``LocalExecutor._reap`` says "Idempotent" in its first line and is; this one claimed
to be its twin and inherited the wait but not the idempotence.

How it hid: nothing is wrong at the line where it is written, and the race needs two readers
to finish the same run inside one poll interval, so the CLI (one reader) never sees it and a
test that opens one page never sees it. The rule: a cleanup on shared state is a *prune to the
correct state* under a lock, never a *remove of the element I found*; the second caller then
finds nothing to do rather than an element that is gone.

Negative case: with ``_drop_exited_coordinators`` replaced by the original
``_SPAWNED_COORDINATORS.remove(proc)``, ``test_two_readers_reaping_one_run_both_return``
fails with that ``ValueError`` on every run — the barrier makes the interleaving certain,
not likely.
"""

import threading

import pytest

from detail.executor import local
from tests.concurrency import WAIT_TIMEOUT_S, join_all


class _ExitedCoordinator:
    """A coordinator handle whose process has already exited, as the reaper sees one.

    ``wait`` returns at once (the process is gone) but first holds every caller at ``barrier``,
    so the readers leave it together — the interleaving two GUI pollers produce when they
    read the same terminal status in the same poll. ``poll`` reports the return code, which
    is what "exited" means to the prune.
    """

    def __init__(self, pid: int, barrier: threading.Barrier) -> None:
        self.pid = pid
        self.returncode = 0
        self._barrier = barrier

    def wait(self, timeout: float | None = None) -> int:
        self._barrier.wait(timeout=WAIT_TIMEOUT_S)
        return self.returncode

    def poll(self) -> int:
        return self.returncode


@pytest.fixture
def registry(monkeypatch):
    """A private coordinator registry, so the test neither sees nor leaves real handles."""
    procs: list = []
    monkeypatch.setattr(local, "_SPAWNED_COORDINATORS", procs)
    return procs


def _reap_from(n_readers: int, pid: int) -> list[BaseException]:
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            local.reap_coordinator(pid)
        except (
            BaseException
        ) as exc:  # noqa: BLE001 — the test's verdict is the exception
            errors.append(exc)

    threads = [
        threading.Thread(target=reader, name=f"reader-{i}") for i in range(n_readers)
    ]
    for thread in threads:
        thread.start()
    join_all(threads)
    return errors


def test_two_readers_reaping_one_run_both_return(registry):
    """Two progress pollers finish the same run at once: neither raises, the handle is dropped."""
    registry.append(_ExitedCoordinator(pid=4242, barrier=threading.Barrier(2)))

    errors = _reap_from(2, pid=4242)

    assert errors == [], f"a second reader of a finished run raised: {errors!r}"
    assert registry == [], "the exited coordinator is still retained after reaping"


def test_reaping_drops_only_the_exited_handles(registry):
    """The prune is a state, not an action: a live coordinator of another run stays retained."""

    class _Live(_ExitedCoordinator):
        def poll(self):
            return None

    live = _Live(pid=1, barrier=threading.Barrier(1))
    registry.append(live)
    registry.append(_ExitedCoordinator(pid=2, barrier=threading.Barrier(1)))

    assert _reap_from(1, pid=2) == []
    assert registry == [live]


def test_reaping_a_pid_this_process_did_not_spawn_is_a_no_op(registry):
    """A reattached run's coordinator belongs to another process: nothing to wait on or drop."""
    assert _reap_from(1, pid=999) == []
    assert local.reap_coordinator(None) is None
    assert registry == []
