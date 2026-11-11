"""Node-cache builds that outlive the page that started them (§22).

Building a run's node cache reads every recording once and can take minutes. It used to run on a
worker owned by the INTERPRET tab, so leaving the page tore the tab down while the build carried
on with nothing showing it. Coming back opened a fresh tab that found no finished cache and offered
**Build** again: pressing it queued a second build behind the first one's lock, its bar sat at
zero, and the screen read as hung (reported as "building the clusters for existing experiments
has issues when I leave the page and come back").

So a build belongs to the **application**, keyed by the run's directory: one per run, started
here, remembered here until it ends. A tab that opens while one is running attaches to it and
shows its progress from where it is, and cannot start a second.
"""

import time
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from detail.core.progress import remaining_seconds
from detail.gui.workers import StageProgressWorker, _with_eta


class NodeCacheJob(QObject):
    """One build in flight: its last progress, and signals any number of pages can follow."""

    progress = pyqtSignal(int, str)  # permille 0..1000, status
    finished = pyqtSignal()
    failed = pyqtSignal(object)  # the exception

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.permille = 0
        self.status = "starting"
        self._worker: StageProgressWorker | None = None
        self._started = time.monotonic()

    def _on_tick(self, permille: int, status: str) -> None:
        """Every page following this build gets the same status, with an ETA (§6)."""
        eta = remaining_seconds(permille / 1000.0, time.monotonic() - self._started)
        self.permille, self.status = int(permille), _with_eta(status, eta)
        self.progress.emit(self.permille, self.status)


_JOBS: dict[str, NodeCacheJob] = {}


def _key(run_dir) -> str:
    return str(Path(run_dir).resolve()).lower()


def running(run_dir) -> NodeCacheJob | None:
    """The build in flight for ``run_dir``, or ``None``."""
    return _JOBS.get(_key(run_dir))


def start(
    run_dir,
    work: Callable[[Callable[[str, float, str], None]], object],
) -> NodeCacheJob:
    """Start ``work(emit)`` for ``run_dir``, or return the build already running for it.

    The worker reports to the **job** (its ``receiver``), not to a page: the job is what has to
    hear it, and it lives in this registry until the build ends either way, so a delivery always
    has somewhere live to land however many pages came and went.
    """
    existing = running(run_dir)
    if existing is not None:
        return existing
    key = _key(run_dir)
    job = NodeCacheJob(key)
    worker = StageProgressWorker(work, receiver=job)
    job._worker = worker
    worker.tick.connect(job._on_tick)

    def done(_result) -> None:
        _JOBS.pop(key, None)
        job.finished.emit()

    def fail(exc) -> None:
        _JOBS.pop(key, None)
        job.failed.emit(exc)

    worker.done.connect(done)
    worker.failed.connect(fail)
    worker.finished.connect(worker.deleteLater)
    _JOBS[key] = job
    worker.start()
    return job
