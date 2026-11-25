"""A pytest plugin that ends a run which has stopped making progress, and says what stuck.

Loaded on every run by ``addopts`` in ``pyproject.toml``. Two settings do the per-test half:
``faulthandler_timeout`` (``pyproject.toml``) dumps a stuck test's threads and ends its process, and
``--max-worker-restart=0`` — defaulted here, below — turns a dead xdist worker into a failed run
naming the test it died in. Neither can see a stall that is not inside a test:

- **The controller waiting on work nobody will run.** pytest-xdist 3.8's ``--dist loadfile``
  re-queues *every* file a crashed worker had ever been given, finished ones included. A worker
  handed a finished file is sent an empty list, never reports again, and is never given more —
  so every worker idles and the controller waits forever: one segfault, then a run that
  never ends. Restarts are off for that reason; this is what catches the next stall of
  that shape.
- **A process that will not exit.** A non-daemon thread a test leaked keeps the interpreter
  alive after the summary has printed.

**What it does.** In the controller only (a plain run's single process, or xdist's controller —
never a worker), a daemon thread watches for any sign of progress: a test starting, a report, a
worker collecting. After ``suite_stall_timeout`` seconds with none it writes what it knows to
stderr — the tests started and not finished, each xdist worker's unfinished tests and the queue
left, and every thread's stack — and ends the run with exit status 3 (pytest's own "internal
error"), killing its worker processes on the way out. After the session finishes it allows
``_EXIT_GRACE_S`` for the process to exit and then does the same.

The bound is set above ``faulthandler_timeout`` (120s) so a stuck *test* is always reported by
faulthandler first, with its own stack; this is the net under that one. ``0`` turns it off, and
it stands down under ``--pdb`` / ``--trace``, where a person is the one holding the run.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time

import pytest

#: Seconds after the session finishes before a process that has not exited is ended (or the stall
#: bound, when that is shorter).
_EXIT_GRACE_S = 60.0
#: Set to the pid of the process whose run is being watched. A pytest started *in-process* by a
#: test (``pytest.main``) sees its own pid here and stays out: its watchdog would outlive its
#: session inside the outer run and end the outer process.
_ACTIVE = "DETAIL_SUITE_WATCHDOG_PID"
#: How often the watching thread wakes.
_TICK_S = 1.0
#: Exit status for a run ended by the watchdog — pytest's INTERNAL_ERROR.
_EXIT_STATUS = 3
#: The clock and the sleep the watchdog uses, taken when this plugin loads, before any test runs.
#: A test that patches ``time.monotonic`` or ``time.sleep`` (even through another module's
#: ``time`` attribute) patches them for the whole process, and the serial CI job runs tests in
#: the process this thread lives in: a fake clock set ahead read as 714 s with no progress and
#: ended a run 8 minutes old; a no-op sleep turns this loop into a busy one.
_monotonic = time.monotonic
_sleep = time.sleep


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "suite_stall_timeout",
        "Seconds with no test progress before the run is ended and the stall reported "
        "(scripts/suite_watchdog.py). 0 turns it off.",
        default="300",
    )


class _Watchdog:
    def __init__(self, config: pytest.Config, timeout_s: float) -> None:
        self._config = config
        self._timeout_s = timeout_s
        self._last = _monotonic()
        self._deadline_s = timeout_s
        self._phase = "starting"
        self._running: dict[str, str] = {}
        self._lock = threading.Lock()
        thread = threading.Thread(
            target=self._watch, name="suite-watchdog", daemon=True
        )
        thread.start()

    def _progress(self, phase: str | None = None) -> None:
        with self._lock:
            self._last = _monotonic()
            if phase is not None:
                self._phase = phase

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self._progress("running tests")

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids) -> None:
        self._progress("running tests")

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodeready(self, node) -> None:
        self._progress()

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error) -> None:
        self._progress()

    def pytest_runtest_logstart(self, nodeid: str, location) -> None:
        with self._lock:
            self._running[nodeid] = time.strftime("%H:%M:%S")
        self._progress()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # "???" is the report xdist writes for the test a worker died in.
        if report.when in ("teardown", "???"):
            with self._lock:
                self._running.pop(report.nodeid, None)
        self._progress()

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        with self._lock:
            self._last = _monotonic()
            self._deadline_s = min(_EXIT_GRACE_S, self._timeout_s)
            self._phase = "exiting after the session finished"
            self._running.clear()

    def _watch(self) -> None:
        while True:
            _sleep(_TICK_S)
            with self._lock:
                idle = _monotonic() - self._last
                deadline = self._deadline_s
            if idle >= deadline:
                self._fire(idle)
                return

    def _fire(self, idle: float) -> None:
        out = sys.__stderr__ or sys.stderr
        with self._lock:
            running = dict(self._running)
            phase = self._phase
        lines = [
            "",
            "=" * 78,
            f"suite_watchdog: no progress for {idle:.0f}s while {phase} — ending the run.",
        ]
        if running:
            lines.append("Tests started and not finished:")
            lines += [f"  {nodeid}  (started {at})" for nodeid, at in running.items()]
        lines += _xdist_state(self._config)
        lines.append(
            "Every thread of this (controller) process follows. Raise or disable the bound "
            "with -o suite_stall_timeout=<seconds> (0 = off)."
        )
        lines.append("=" * 78)
        try:
            out.write("\n".join(lines) + "\n")
            out.flush()
            faulthandler.dump_traceback(file=out, all_threads=True)
            out.flush()
        except (OSError, ValueError):
            pass
        _kill_children()
        os._exit(_EXIT_STATUS)


def _xdist_state(config: pytest.Config) -> list[str]:
    """What xdist's controller is still waiting on: per worker, and the queue nobody has taken."""
    dsession = config.pluginmanager.get_plugin("dsession")
    sched = getattr(dsession, "sched", None)
    if sched is None:
        return []
    lines = []
    assigned = getattr(sched, "assigned_work", None)
    if isinstance(assigned, dict):
        for node, work in list(assigned.items()):
            pending = [
                nodeid
                for unit in work.values()
                for nodeid, done in unit.items()
                if not done
            ]
            lines.append(f"xdist worker {node.gateway.id}: {len(pending)} unfinished")
            lines += [f"    {nodeid}" for nodeid in pending[:5]]
    queue = getattr(sched, "workqueue", None)
    if queue:
        lines.append(f"xdist queue: {len(queue)} unit(s) never taken by a worker")
    pending = getattr(sched, "pending", None)
    if isinstance(pending, list) and pending:
        lines.append(f"xdist queue: {len(pending)} test(s) never taken by a worker")
    return lines


def _kill_children() -> None:
    """End this process's children (xdist workers and what they started) before it exits."""
    try:
        import psutil
    except ImportError:
        return
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass


def _timeout_s(config: pytest.Config) -> float:
    raw = config.getini("suite_stall_timeout")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise pytest.UsageError(
            f"suite_stall_timeout must be a number of seconds, got {raw!r}"
        ) from None


def pytest_configure(config: pytest.Config) -> None:
    if hasattr(config, "workerinput"):
        return  # a worker's stuck test is faulthandler_timeout's to report
    # xdist's own default restarts a dead worker (4 per worker), which under `--dist loadfile`
    # is the hang this module exists for. Set here rather than in `addopts`, where it would make
    # `-p no:xdist` an unknown-option error; an explicit `--max-worker-restart` still wins. Runs
    # before xdist reads it: xdist's `pytest_configure` is `trylast`.
    if getattr(config.option, "maxworkerrestart", "") is None:
        config.option.maxworkerrestart = "0"
    if config.getoption("usepdb", False) or config.getoption("trace", False):
        return
    if os.environ.get(_ACTIVE) == str(os.getpid()):
        return
    timeout_s = _timeout_s(config)
    if timeout_s <= 0:
        return
    os.environ[_ACTIVE] = str(os.getpid())
    config.pluginmanager.register(_Watchdog(config, timeout_s), "suite-watchdog")
