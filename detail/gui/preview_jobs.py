"""Video preview builds, queued and run in the background by the application (§3 *Video
preview cache*).

A preview build reads a whole video part and takes minutes per hour of footage, and it is
wanted on three occasions nobody is waiting at a page for: a recording finished importing, an
experiment was opened (on a new machine, after an archive import, or after the cache was
cleared), and a tracking run finished. So the queue belongs to the **application**, like
``node_cache_jobs``: one per open experiment, one build at a time, shown on a permanent bar in
the status bar, and reachable by any page through :func:`current` — a page that needs some
recordings first moves them to the front (:meth:`PreviewQueue.prioritise`).

The queue holds **recording ids**, not tasks: what a recording is missing is asked inside the
worker (``Experiment.build_previews``), so queueing costs the UI thread nothing and a recording
queued twice is built once.
"""

from __future__ import annotations

import time

from detail.core.progress import OperationCancelled, remaining_seconds
from detail.gui.qt import QObject, Signal
from detail.gui.workers import StageProgressWorker, _with_eta


class PreviewQueue(QObject):
    """One experiment's preview builds: what is waiting, what is running, and its progress."""

    progress = Signal(int, str)  # permille of the build in hand, status line
    recording_done = Signal(str)  # a recording's previews are built
    failed = Signal(
        str, str
    )  # recording id, what failed and why (one per failed build)
    idle = Signal()

    def __init__(self, experiment) -> None:
        super().__init__()
        self._experiment = experiment
        self.pending: list[str] = []
        self.current: str | None = None
        self.permille = 0
        self.status = ""
        #: The build in hand's own phase line, without the queue's count or time left — what a
        #: page waiting on some recordings shows under its own bar.
        self.phase = ""
        self._worker: StageProgressWorker | None = None
        self._started = 0.0
        self._closed = False
        #: Recordings queued again while they were the one building: what they are missing
        #: was read when the build started, so a tracking run that finished since would
        #: otherwise get no masks until the experiment was reopened.
        self._again: set[str] = set()

    @property
    def experiment(self):
        return self._experiment

    @property
    def busy(self) -> bool:
        return self.current is not None

    def queue(self, recording_ids=None) -> None:
        """Add recordings to the back of the queue (``None``: every recording) and start."""
        if self._closed:
            return
        ids = (
            list(self._experiment.recording_ids)
            if recording_ids is None
            else list(recording_ids)
        )
        for recording_id in ids:
            if recording_id == self.current:
                self._again.add(recording_id)
            elif recording_id not in self.pending:
                self.pending.append(recording_id)
        self._next()

    def prioritise(self, recording_ids) -> None:
        """Move ``recording_ids`` to the front, queueing any not yet waiting. The build in hand
        finishes first: it is never more than one recording's worth of work."""
        if self._closed:
            return
        front = [rid for rid in recording_ids if rid != self.current]
        self.pending = front + [rid for rid in self.pending if rid not in front]
        self._next()

    def cancel(self) -> None:
        """Drop everything waiting and stop the build in hand (its part is discarded)."""
        self.pending.clear()
        self._again.clear()
        if self._worker is not None:
            self._worker.request_cancel()

    def withdraw(self, recording_ids) -> None:
        """Drop ``recording_ids`` from the queue, and stop the build in hand if it is one of
        them — a page's Cancel, which must not cancel what another page is waiting on.
        """
        ids = set(recording_ids)
        self.pending = [rid for rid in self.pending if rid not in ids]
        if self.current in ids and self._worker is not None:
            self._worker.request_cancel()

    def close(self) -> None:
        """Stop for good — the experiment was switched or the app is closing."""
        self._closed = True
        self.cancel()

    def shutdown(self, timeout_ms: int = 3_000) -> None:
        """:meth:`close`, then wait (bounded) for the build in hand to stop — the app is
        quitting, and a thread still running when Qt tears down takes the process with it.
        """
        worker = self._worker
        self.close()
        if worker is not None:
            worker.wait(timeout_ms)

    def _next(self) -> None:
        if self.current is not None or not self.pending or self._closed:
            if self.current is None and not self.pending:
                self.idle.emit()
            return
        recording_id = self.pending.pop(0)
        self.current = recording_id
        self.permille = 0
        self._started = time.monotonic()
        experiment = self._experiment

        def work(emit):
            return experiment.build_previews([recording_id], progress=emit)

        worker = StageProgressWorker(work, receiver=self)
        self._worker = worker
        worker.tick.connect(self._on_tick)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_tick(self, permille: int, status: str) -> None:
        eta = remaining_seconds(permille / 1000.0, time.monotonic() - self._started)
        waiting = f" · {len(self.pending)} waiting" if self.pending else ""
        self.permille = int(permille)
        self.phase = status
        self.status = _with_eta(f"Previews: {status}{waiting}", eta)
        self.progress.emit(self.permille, self.status)

    def _finish(self) -> str:
        recording_id, self.current, self._worker = self.current, None, None
        if recording_id in self._again:
            self._again.discard(recording_id)
            if recording_id not in self.pending and not self._closed:
                self.pending.append(recording_id)
        return recording_id

    def _on_done(self, report) -> None:
        recording_id = self._finish()
        for task, why in getattr(report, "failed", ()):
            self.failed.emit(
                recording_id, f"part {task.part_index + 1} {task.kind}: {why}"
            )
        self.recording_done.emit(recording_id)
        self._next()

    def _on_failed(self, exc) -> None:
        recording_id = self._finish()
        if not isinstance(exc, OperationCancelled):
            self.failed.emit(recording_id, f"{type(exc).__name__}: {exc}")
        self._next()


_QUEUE: PreviewQueue | None = None


def install(experiment) -> PreviewQueue:
    """Make a fresh queue for ``experiment`` the application's, stopping the previous one.

    The old queue's build in hand is cancelled and waited for (bounded — a cancel lands at its
    next progress report, well under a second), so two builds never run side by side here. The
    cache's build lock would serialize them anyway; waiting keeps the old one from finishing
    work for an experiment nobody has open."""
    global _QUEUE
    if _QUEUE is not None:
        _QUEUE.shutdown()
    _QUEUE = PreviewQueue(experiment)
    return _QUEUE


def current() -> PreviewQueue | None:
    """The open experiment's queue, or ``None`` before one is open."""
    return _QUEUE


def queue_recordings(experiment, recording_ids) -> None:
    """Queue ``recording_ids`` if ``experiment`` is the one whose queue is installed — a page
    for another experiment (or a test with no app) queues nothing."""
    found = _QUEUE
    if found is None or found.experiment is None:
        return
    if found.experiment.dir != experiment.dir:
        return
    found.queue(recording_ids)
