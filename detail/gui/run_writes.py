"""Label writes that outlive the page that queued them (§21, issue #24 finding 1).

A label typed in INTERPRET, or a clip taken out or a cluster finished in AUDIT, shows at once and
is written in the background, one write at a time and in order: each write reprojects the run, and
that can take seconds on a large cohort. The queue used to live on the page. Each next write was
started from the previous write's ``done``, and ``GuiWorker`` drops a delivery whose receiver is
gone — so leaving the page (the shell ``deleteLater``\\ s it on every navigation) dropped the first
write's ``done``, nothing started the second, and every write after the first was lost without a
word. Labelling three clusters quickly and clicking another run left one label on disk.

So the queue belongs to the **application**, keyed by the run's directory, as a node-cache build
does (``node_cache_jobs``): one queue per run, its worker reporting to the queue (its
``receiver``), which lives in this registry for the life of the app. A page only hears about its
own writes, and only while it is alive: ``done`` and ``failed`` callbacks of a page that has gone
are dropped, and the writes themselves carry on. Quitting the app is the one exit that can still
lose them, so the shell asks first (:func:`pending`, ``app.py::closeEvent``).

A page that opens while a run's writes are still landing reads the run as it stands, then reads it
again once the queue is idle (:meth:`RunWriteQueue.when_idle`) — or it would show the run without
the writes that were still in flight.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox

from detail.gui.lifetime import alive
from detail.gui.workers import BackgroundCall


@dataclass
class _Write:
    work: Callable[[], object]
    owner: QObject | None
    done: Callable[[object], None] | None
    failed: Callable[[Exception], None] | None
    title: str


class RunWriteQueue(QObject):
    """One run's writes, run one at a time in the order they were submitted."""

    #: Emitted on the UI thread each time the last queued write has landed (or the queue was
    #: dropped after a failure).
    idle = pyqtSignal()

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self._writes: list[_Write] = []
        self._writer: BackgroundCall | None = None
        self._current: _Write | None = None
        self._when_idle: list[tuple[QObject | None, Callable[[], None]]] = []

    def count(self) -> int:
        """Writes not yet on disk: the one running and every one queued behind it."""
        return len(self._writes) + (1 if self._writer is not None else 0)

    def busy(self) -> bool:
        return self.count() > 0

    def submit(
        self,
        work: Callable[[], object],
        *,
        owner: QObject | None,
        done: Callable[[object], None] | None = None,
        failed: Callable[[Exception], None] | None = None,
        title: str = "Not saved",
    ) -> None:
        """Queue ``work`` behind every write already queued for this run.

        ``done(result)`` / ``failed(exc)`` run on the UI thread, and only while ``owner`` is
        alive; ``work`` runs whatever happens to ``owner``. ``title`` heads the message shown
        when a write fails after its page has gone.
        """
        self._writes.append(_Write(work, owner, done, failed, title))
        self._next()

    def when_idle(self, then: Callable[[], None], *, owner: QObject | None) -> None:
        """Run ``then`` once no write is queued or running — at once if none is."""
        if not self.busy():
            then()
            return
        self._when_idle.append((owner, then))

    def _next(self) -> None:
        if self._writer is not None:
            return
        if not self._writes:
            waiting, self._when_idle = self._when_idle, []
            for owner, then in waiting:
                if owner is None or alive(owner):
                    then()
            self.idle.emit()
            hub().changed.emit()
            return
        write = self._writes.pop(0)
        writer = BackgroundCall(write.work, receiver=self)
        self._writer, self._current = writer, write
        writer.done.connect(self._on_done)
        writer.failed.connect(self._on_failed)
        writer.finished.connect(writer.deleteLater)
        writer.start()

    def _on_done(self, result) -> None:
        write = self._current
        self._writer = self._current = None
        if write is not None and write.done is not None and _owner_alive(write):
            write.done(result)
        self._next()

    def _on_failed(self, exc) -> None:
        """Every write queued behind a failed one was made assuming it: drop them all, and tell
        each page still open once. A failure nobody is left to hear is shown on its own.
        """
        write = self._current
        dropped = [write, *self._writes] if write is not None else list(self._writes)
        self._writer = self._current = None
        self._writes = []
        told: set[int] = set()
        for each in dropped:
            if each.failed is None or not _owner_alive(each) or id(each.owner) in told:
                continue
            told.add(id(each.owner))
            each.failed(exc)
        if write is not None and id(write.owner) not in told:
            report_orphan_failure(write.title, exc)
        self._next()


def _owner_alive(write: _Write) -> bool:
    return write.owner is None or alive(write.owner)


def report_orphan_failure(title: str, exc: Exception) -> None:
    """A write failed after the page that queued it was left: say so on whatever is on screen.

    A module function so a test can replace it; a silent failure here is the loss this module
    exists to prevent.
    """
    QMessageBox.critical(QApplication.activeWindow(), title, str(exc))


class _Hub(QObject):
    """Emits ``changed`` whenever any run's queue goes idle — what the shell's quit waits on."""

    changed = pyqtSignal()


_HUB: _Hub | None = None
_QUEUES: dict[str, RunWriteQueue] = {}


def hub() -> _Hub:
    """The one object the shell connects to; built on first use, on the UI thread."""
    global _HUB
    if _HUB is None:
        _HUB = _Hub()
    return _HUB


def _key(run) -> str:
    run_dir = run._run_dir() if hasattr(run, "_run_dir") else None
    if run_dir is None:
        return f"object:{id(run)}"
    return str(Path(run_dir).resolve()).lower()


def queue_for(run) -> RunWriteQueue:
    """The write queue for ``run``'s directory, made on first use and kept for the app's life."""
    return queue_for_key(_key(run))


def queue_for_key(key: str) -> RunWriteQueue:
    """The write queue named ``key``, for writes that belong to something other than a run —
    an example set's marks (``keypoint_labeling._MarkSaver``)."""
    hub()
    queue = _QUEUES.get(key)
    if queue is None:
        queue = _QUEUES[key] = RunWriteQueue(key)
    return queue


def pending() -> int:
    """Writes not yet on disk across every run — what quitting now would lose."""
    return sum(queue.count() for queue in _QUEUES.values())


def when_all_idle(then: Callable[[], None]) -> None:
    """Run ``then`` once no run has a write left — at once if none has. What *Wait, then quit*
    hangs on: the window closes itself when the last write lands."""
    if not pending():
        then()
        return

    def check() -> None:
        if pending():
            return
        hub().changed.disconnect(check)
        then()

    hub().changed.connect(check)
