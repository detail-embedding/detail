"""Two threads reaping one run's coordinator must both return, and exactly one waits on it.

**The class: a shared list read, waited on, then mutated by whoever got there first.** A GUI
worker streams each open run page's progress, and when a run goes terminal each calls
`reap_coordinator(pid)`. It snapshotted `_SPAWNED_COORDINATORS`, waited on the handle, and then
called `list.remove`: the second thread's `remove` raised `ValueError: list.remove(x): x not in
list` out of `RunProgressWorker.run`, which ended the stream, with the page showing a run that
was still going. Seen when a Keypoints dinov3 model finished training remotely, the same session
that then crashed natively (the native crash is not established to follow from this; it is why
the Python stack of every thread is now printed on a fatal signal).

What is pinned: the interleaving above, forced (the wait is made slow so both threads are inside
it at once), and that a coordinator still alive after the bounded wait is kept for the exit-time
reap, which is what `test_a_finished_run_leaves_no_coordinator_behind` guards the other side of.
"""

import subprocess
import sys
import threading
import time

import pytest

from detail.executor import local


@pytest.fixture
def handle():
    """A real, already-finished child, registered the way `spawn_coordinator` registers one."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    with local._SPAWNED_LOCK:
        local._SPAWNED_COORDINATORS.append(proc)
    yield proc
    with local._SPAWNED_LOCK:
        if proc in local._SPAWNED_COORDINATORS:
            local._SPAWNED_COORDINATORS.remove(proc)
    subprocess.Popen.wait(proc)  # not the instance attribute a test may have set


def _reap_from_threads(pid: int, count: int = 2) -> list[BaseException]:
    errors: list[BaseException] = []
    start = threading.Barrier(count)

    def reaper() -> None:
        start.wait()
        try:
            local.reap_coordinator(pid)
        except BaseException as exc:  # noqa: BLE001 - a probe: report what escaped
            errors.append(exc)

    threads = [threading.Thread(target=reaper) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    return errors


def test_two_threads_reaping_one_coordinator_both_return(handle):
    real_wait = handle.wait

    def slow_wait(timeout=None):
        time.sleep(0.3)  # long enough that the other thread is past its own snapshot
        return real_wait(timeout=timeout)

    handle.wait = slow_wait

    assert _reap_from_threads(handle.pid) == []
    assert handle not in local._SPAWNED_COORDINATORS


def test_a_coordinator_still_running_after_the_bounded_wait_is_kept(handle):
    def never_exits(timeout=None):
        raise subprocess.TimeoutExpired("coordinator", timeout)

    handle.wait = never_exits

    local.reap_coordinator(handle.pid)

    assert (
        handle in local._SPAWNED_COORDINATORS
    ), "dropped while running: a warning at exit"


def test_reaping_a_pid_this_process_did_not_spawn_is_a_no_op(handle):
    local.reap_coordinator(handle.pid + 100_000)
    local.reap_coordinator(None)
    assert handle in local._SPAWNED_COORDINATORS
