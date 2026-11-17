"""A run's label writes outlive the page that queued them (``detail/gui/run_writes.py``).

**The class: a queue of writes whose progress depends on the widget that started it.** INTERPRET
and AUDIT each held their write queue on the page and started each next write from the previous
write's ``done``. ``GuiWorker`` drops a delivery whose receiver is gone, which is right for a
result that only updates a widget and wrong for a result that *starts the next write*: the shell
``deleteLater``\\ s the page on every navigation, so leaving it with three labels queued wrote the
first and lost the other two, with three names on screen and no message (issue #24, finding 1).

How it hid: every write path was tested with the page alive, and a page alive to the end drains
its own queue. The rule: a write queue's owner lives as long as the writes do — here one queue
per run, kept by the application, reporting to itself — and a page only *hears* about its writes.
Quitting is the one exit left, so the shell asks first.

The page-level guards drive the real pages over a real fit, slowed so the writes are still queued
when the page goes: ``tests/gui/test_cluster_workspace.py::
test_labels_queued_when_the_page_is_left_still_reach_the_run`` (INTERPRET) and
``tests/gui/test_label_check.py::test_marks_queued_when_the_page_is_left_still_reach_the_run``
(AUDIT). This file holds the queue's own contract and the quit prompt.
"""

import threading

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent  # noqa: E402
from PyQt6.QtGui import QCloseEvent  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

from detail.gui import run_writes  # noqa: E402

from tests.gui.conftest import drain_until  # noqa: E402


class _Run:
    """Just what ``queue_for`` keys on: a run's directory."""

    def __init__(self, run_dir) -> None:
        self._dir = run_dir

    def _run_dir(self):
        return self._dir


def _gone(qapp, widget: QWidget) -> None:
    """Tear ``widget`` down the way the shell does: ``deleteLater``, no page code run."""
    widget.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def test_writes_queued_by_a_page_that_has_gone_still_run_in_order(qapp, tmp_path):
    queue = run_writes.queue_for(_Run(tmp_path))
    page = QWidget()
    gate = threading.Event()
    written: list[int] = []
    heard: list[int] = []

    def write(n):
        def work():
            if n == 0:
                assert gate.wait(10)
            written.append(n)
            return n

        return work

    for n in range(3):
        queue.submit(write(n), owner=page, done=heard.append)
    assert queue.count() == 3 and run_writes.pending() >= 3
    _gone(qapp, page)
    gate.set()
    assert drain_until(qapp, lambda: not queue.busy(), 10_000)
    assert written == [0, 1, 2], "a write was lost or run out of order"
    assert heard == [], "a page that has gone was called back"


def test_one_queue_per_run_directory(qapp, tmp_path):
    a = run_writes.queue_for(_Run(tmp_path / "an_01"))
    assert run_writes.queue_for(_Run(tmp_path / "an_01")) is a
    assert run_writes.queue_for(_Run(tmp_path / "an_02")) is not a


def test_a_failure_drops_the_writes_queued_behind_it_and_tells_the_page_once(
    qapp, tmp_path
):
    queue = run_writes.queue_for(_Run(tmp_path))
    page = QWidget()
    gate = threading.Event()
    written: list[str] = []
    told: list[str] = []

    def fail():
        assert gate.wait(10)
        raise OSError("disk full")

    queue.submit(fail, owner=page, failed=lambda exc: told.append(str(exc)))
    for name in ("b", "c"):
        queue.submit(
            lambda n=name: written.append(n),
            owner=page,
            failed=lambda exc: told.append(str(exc)),
        )
    gate.set()
    assert drain_until(qapp, lambda: not queue.busy(), 10_000)
    assert written == [], "a write made assuming the failed one went ahead"
    assert told == ["disk full"]
    page.deleteLater()


def test_a_failure_after_the_page_has_gone_is_still_shown(qapp, tmp_path, monkeypatch):
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        run_writes,
        "report_orphan_failure",
        lambda title, exc: shown.append((title, str(exc))),
    )
    queue = run_writes.queue_for(_Run(tmp_path))
    page = QWidget()
    gate = threading.Event()

    def fail():
        assert gate.wait(10)
        raise OSError("disk full")

    queue.submit(fail, owner=page, failed=lambda exc: None, title="Label not saved")
    _gone(qapp, page)
    gate.set()
    assert drain_until(qapp, lambda: bool(shown), 10_000)
    assert shown == [("Label not saved", "disk full")]


def test_when_idle_waits_for_every_queued_write(qapp, tmp_path):
    queue = run_writes.queue_for(_Run(tmp_path))
    gate = threading.Event()
    order: list[str] = []
    queue.submit(lambda: (gate.wait(10), order.append("write")), owner=None)
    queue.when_idle(lambda: order.append("then"), owner=None)
    assert order == []
    gate.set()
    assert drain_until(qapp, lambda: order == ["write", "then"], 10_000)
    queue.when_idle(lambda: order.append("at once"), owner=None)
    assert order[-1] == "at once"


# --- quitting ----------------------------------------------------------------------------


def _window(tmp_path, monkeypatch):
    from detail.storage import locations
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return MainWindow()


@pytest.fixture
def held(qapp, tmp_path):
    """One write held open on a run, released (and drained) at teardown."""
    gate = threading.Event()
    queue = run_writes.queue_for(_Run(tmp_path / "held"))
    queue.submit(lambda: gate.wait(10), owner=None)
    yield queue
    gate.set()
    drain_until(qapp, lambda: not queue.busy(), 10_000)


def test_quitting_with_writes_queued_asks_and_cancel_stays(
    qapp, tmp_path, monkeypatch, held
):
    from detail.gui.app import MainWindow

    asked: list[int] = []
    monkeypatch.setattr(
        MainWindow, "_confirm_unwritten", lambda self, n: asked.append(n) or "cancel"
    )
    window = _window(tmp_path, monkeypatch)
    event = QCloseEvent()
    window.closeEvent(event)
    assert asked == [1]
    assert not event.isAccepted()


def test_quit_now_quits_over_the_writes(qapp, tmp_path, monkeypatch, held):
    from detail.gui.app import MainWindow

    monkeypatch.setattr(MainWindow, "_confirm_unwritten", lambda self, n: "quit")
    window = _window(tmp_path, monkeypatch)
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_wait_then_quit_closes_the_window_once_the_writes_land(
    qapp, tmp_path, monkeypatch
):
    from detail.gui.app import MainWindow

    gate = threading.Event()
    queue = run_writes.queue_for(_Run(tmp_path / "waited"))
    queue.submit(lambda: gate.wait(10), owner=None)
    monkeypatch.setattr(MainWindow, "_confirm_unwritten", lambda self, n: "wait")
    window = _window(tmp_path, monkeypatch)
    closed: list[bool] = []
    monkeypatch.setattr(window, "close", lambda: closed.append(True))
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted(), "it quit before the writes landed"
    assert closed == []
    gate.set()
    assert drain_until(qapp, lambda: closed == [True], 10_000)


def test_quitting_with_nothing_queued_does_not_ask(qapp, tmp_path, monkeypatch):
    from detail.gui.app import MainWindow

    def ask(self, n):
        raise AssertionError("asked with nothing queued")

    monkeypatch.setattr(MainWindow, "_confirm_unwritten", ask)
    assert drain_until(qapp, lambda: run_writes.pending() == 0, 10_000)
    window = _window(tmp_path, monkeypatch)
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
