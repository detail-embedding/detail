"""A worker thread's object is never destroyed while Qt still counts the thread as running.

The class: **a keep-alive released on the program's idea of "the thread is done" rather than
Qt's.** Destroying a ``QThread`` whose state is Running is a Qt fatal abort, not an exception:

    QThread: Destroyed while thread '' is still running
    Fatal Python error: Aborted
    Current thread ... (most recent call first):
      Garbage-collecting
      File "detail/gui/ui_thread_gc.py", line 61 in collect_due
      File "tests/conftest.py", line 114 in pytest_runtest_protocol

That ended the whole ``gui`` CI job on ``main`` on 2026-11-16, a quarter of the way in, on code
that had passed the same job forty minutes earlier.

**How it hid.** "Running" is wider than "inside ``run()``". Read from Qt 6.12's
``qthread_unix.cpp``: the state becomes Running when ``start()`` returns, before the thread has
entered ``run()``, and stays Running after ``run()`` returns until ``QThreadPrivate::finish``
sets Finishing — only then is ``finished`` emitted, and ``~QThread`` *waits* for a finishing
thread instead of aborting. So there are two gaps in which a thread runs no Python and Qt still
calls it running:

- **before ``run()``** — started, not yet scheduled;
- **after ``run()``** — returned, Qt's exit code not yet reached.

The live montage's picture worker rooted itself in a module set and let go in a ``finally`` at
the end of its own ``run()``: the second gap, every time a montage was destroyed. Its comment
said it copied ``BackgroundCall``'s self-rooting, which lets go on ``finished``; that one
difference was the bug. Destroying a montage stops its worker, and the test suite collects
garbage right after it destroys a test's windows, so the worker's exit and the collection that
frees it were lined up within a millisecond or two on most cluster-page tests (measured: within
5 ms on 63 of 140). The app has the same exposure with a wider aim: closing a cluster window,
then the UI thread's collection timer.

Four more workers had no root at all (the run-progress poller, the model installer, the import
queue's and the preview dialog's), and were safe only while their page happened to be alive.

**The rule.** ``GuiWorker`` holds every started worker from ``start()`` until its ``finished``
is delivered on the UI thread (``workers._UNFINISHED``), and joins the thread before letting go.
No subclass keeps its own root, and none lets go from ``run()``.

Three layers, as in ``test_worker_lifetime.py``:

- the abort itself, in a **child process** (a failure here is a process abort, which must not
  take the suite with it): each gap, and the real picture worker, a few hundred times. Before
  the fix each case aborted within those rounds on every run tried (20 of 20 each);
- the mechanism in-process: held while running, let go once finished, held again by a restart;
- two source scans, so a worker written tomorrow cannot opt out: a ``start`` override that
  skips the base's, and a ``run`` that removes itself from something.
"""

import ast
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from detail.gui import workers
from tests import waits
from tests.gui.test_worker_lifetime import _THE_BASE, _base_names

_GUI = Path(workers.__file__).parent

#: Run in a child process: ``case`` and the number of rounds are its arguments. Each round ends
#: a worker in one of the two gaps, drops the last reference the caller holds, and collects.
_STRESS = """
import gc
import sys
import threading

from detail.gui import live_montage
from detail.gui import workers
from detail.gui.qt import QApplication

case, rounds = sys.argv[1], int(sys.argv[2])
app = QApplication([])


class Quick(workers.GuiWorker):
    def __init__(self):
        super().__init__(receiver=None)
        self.ended = threading.Event()

    def run(self):
        self.ended.set()


class Source:
    def close(self):
        pass


for _ in range(rounds):
    if case == "before_run":
        worker = Quick()
        worker.start()
    elif case == "after_run":
        worker = Quick()
        worker.start()
        while not worker.ended.is_set():
            pass
    else:  # the montage's picture worker, stopped as a destroyed montage stops it
        worker = live_montage._PictureWorker(receiver=None)
        worker.retire(Source())
        worker.stop()
        while worker._live:
            pass
    del worker
    gc.collect()
    app.processEvents()

# Every worker is let go once its thread has finished: nothing is held for good.
for _ in range(2000):
    if not workers._UNFINISHED:
        break
    app.processEvents()
    threading.Event().wait(0.005)
assert not workers._UNFINISHED, len(workers._UNFINISHED)
print("survived", rounds)
"""


#: Rounds per case. Before the fix, 20 of 20 runs of each case aborted within this many;
#: more only costs time, a full collection per round.
_ROUNDS = "500"


@pytest.mark.parametrize("case", ["before_run", "after_run", "picture_worker"])
def test_collecting_a_dropped_worker_never_aborts_the_process(case):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, "-c", _STRESS, case, _ROUNDS],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=240,
    )
    assert result.returncode == 0 and f"survived {_ROUNDS}" in result.stdout, (
        f"{case}: the child ended with {result.returncode}: "
        + result.stderr.strip()[-600:]
    )


class _Gated(workers.GuiWorker):
    """Runs until told to stop, so a test can look while it is running."""

    def __init__(self) -> None:
        super().__init__(receiver=None)
        self.gate = threading.Event()

    def run(self) -> None:
        self.gate.wait(30)


def test_a_started_worker_is_held_until_its_thread_has_finished(qapp):
    worker = _Gated()
    assert worker not in workers._UNFINISHED, "not before it is started"
    worker.start()
    assert worker in workers._UNFINISHED
    worker.gate.set()
    waits.finished(worker, "the gated worker")
    waits.until(qapp, lambda: worker not in workers._UNFINISHED, "the release")
    assert worker.isFinished()


def test_a_worker_started_again_stays_held_by_the_first_runs_release(qapp):
    """The picture worker's shape: it exits when idle and is started again. The first run's
    ``finished`` is delivered after the second ``start``, and must not let go of a thread that
    is running again."""
    worker = _Gated()
    worker.gate.set()
    worker.start()
    worker.wait()  # finished, its release still queued
    worker.gate.clear()
    worker.start()
    qapp.processEvents()  # the first run's release lands
    assert worker.isRunning() and worker in workers._UNFINISHED
    worker.gate.set()
    waits.finished(worker, "the restarted worker")
    waits.until(qapp, lambda: worker not in workers._UNFINISHED, "the second release")


# --------------------------------------------------------------------------- #
# The scans: no worker opts out
# --------------------------------------------------------------------------- #
def _worker_classes(source: str) -> list[ast.ClassDef]:
    """Classes in ``source`` that descend from the base, by name, through one another."""
    classes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)]
    names = {_THE_BASE}
    grew = True
    while grew:
        grew = False
        for node in classes:
            if node.name not in names and names & _base_names(node):
                names.add(node.name)
                grew = True
    return [node for node in classes if node.name in names and node.name != _THE_BASE]


def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def starts_that_skip_the_base(source: str) -> list[str]:
    """Workers whose ``start`` override never calls ``super().start``: not held while running."""
    out = []
    for node in _worker_classes(source):
        start = _method(node, "start")
        if start is None:
            continue
        calls = [
            call
            for call in ast.walk(start)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "start"
            and isinstance(call.func.value, ast.Call)
            and isinstance(call.func.value.func, ast.Name)
            and call.func.value.func.id == "super"
        ]
        if not calls:
            out.append(node.name)
    return out


def runs_that_let_go_of_themselves(source: str) -> list[str]:
    """Workers whose ``run`` removes ``self`` from something (``x.discard(self)``,
    ``x.remove(self)``): a root of their own, released before Qt has finished the thread.
    """
    out = []
    for node in _worker_classes(source):
        run = _method(node, "run")
        if run is None:
            continue
        for call in ast.walk(run):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in ("discard", "remove", "pop")
                and any(isinstance(a, ast.Name) and a.id == "self" for a in call.args)
            ):
                out.append(f"{node.name} (line {call.lineno})")
    return out


def _sources() -> list[tuple[str, str]]:
    return [
        (path.name, path.read_text(encoding="utf-8-sig"))
        for path in sorted(_GUI.rglob("*.py"))
    ]


def test_no_worker_starts_without_being_held():
    offenders = {
        name: found
        for name, source in _sources()
        if (found := starts_that_skip_the_base(source))
    }
    assert not offenders, (
        "a start() that does not call super().start() leaves a running thread held by "
        f"nothing but its caller: {offenders}"
    )


def test_no_worker_lets_go_of_itself_from_run():
    offenders = {
        name: found
        for name, source in _sources()
        if (found := runs_that_let_go_of_themselves(source))
    }
    assert not offenders, (
        "the end of run() is before Qt stops counting the thread as running; GuiWorker lets "
        f"go on `finished`: {offenders}"
    )


def test_the_scans_would_catch_one():
    """Guard the guards: a scan that matches nothing today passes vacuously forever."""
    skipping = (
        f"class Mine({_THE_BASE}):\n"
        "    def start(self):\n"
        "        ROOT.add(self)\n"
        "        QThread.start(self)\n"
    )
    assert starts_that_skip_the_base(skipping) == ["Mine"]
    calling = (
        f"class Mine({_THE_BASE}):\n"
        "    def start(self, *a):\n"
        "        ROOT.add(self)\n"
        "        super().start(*a)\n"
    )
    assert starts_that_skip_the_base(calling) == []
    # The montage's picture worker, as it was — and through a subclass of a subclass.
    letting_go = (
        f"class Mine({_THE_BASE}):\n"
        "    pass\n"
        "class Deeper(Mine):\n"
        "    def run(self):\n"
        "        try:\n"
        "            work()\n"
        "        finally:\n"
        "            _RUNNING.discard(self)\n"
    )
    assert runs_that_let_go_of_themselves(letting_go) == ["Deeper (line 8)"]
    assert runs_that_let_go_of_themselves(calling) == []
    assert (
        runs_that_let_go_of_themselves(
            "class Other:\n    def run(self):\n        x.discard(self)\n"
        )
        == []
    ), "not a worker"
