"""App-level, non-blocking recording-import queue (the import reframe).

``submit`` creates **all** the bout's recording shells synchronously (so empty
recordings appear ``ready`` and video/stream ones appear ``queued`` at once), then
imports the slow ones on a background thread — the user keeps working and can
queue more. One worker drains sequentially (no disk contention, trivially
race-free on ``metadata.csv``). Queued recordings' source paths live here in memory
(never persisted — Tenet 6) and are surfaced via ``source_for``. All work goes
through ``api.imports`` (Qt-free); this marshals it off the UI thread + re-emits.

Progress is folded here, on the UI thread, into one ``ImportProgress`` per tick: one fold spans
every batch submitted before the queue next goes idle, so a bout queued while another imports
grows the "N of M" instead of starting a second count.
"""

import contextlib
import threading
from pathlib import Path

from detail.api.imports import (
    ImportProgressFold,
    RecordingImportSpec,
    create_shells,
    import_pending,
)
from detail.gui.qt import QObject, Signal
from detail.gui.workers import GuiWorker
from detail.storage import recording as recording_store


class _ImportWorker(GuiWorker):
    item_started = Signal(str)  # recording_id
    item_progress = Signal(str, float, str)  # recording_id, fraction, phase
    item_done = Signal(str, bool, str)  # recording_id, ok, error
    batch_finished = Signal()

    def __init__(
        self, experiment, specs, part_kwargs, cancel: threading.Event, *, receiver
    ) -> None:
        super().__init__(receiver=receiver)
        self._experiment = experiment
        self._specs = specs
        self._part_kwargs = part_kwargs
        self._cancel = cancel

    def run(self) -> None:
        import_pending(
            self._experiment,
            self._specs,
            part_kwargs=self._part_kwargs,
            on_progress=lambda _i, _n, sid, frac, st: self.emit_to_ui(
                self.item_progress, sid, float(frac), st
            ),
            on_item_started=lambda sid: self.emit_to_ui(self.item_started, sid),
            on_item_done=lambda sid, ok, err: self.emit_to_ui(
                self.item_done, sid, ok, err
            ),
            should_cancel=self._cancel.is_set,
        )
        self.emit_to_ui(self.batch_finished)


class ImportQueue(QObject):
    """Owned by the shell (per open experiment). ``submit`` creates shells + enqueues."""

    item_started = Signal(str)
    #: The whole queue, folded (``api.imports.ImportProgress``) — what a bar is drawn from.
    progress = Signal(object)
    item_done = Signal(str, bool, str)  # a recording finished (rebuild the tree)
    idle = Signal()  # the queue drained (no pending, no running worker)

    def __init__(self, experiment) -> None:
        super().__init__()
        self._experiment = experiment
        self._pending: list[RecordingImportSpec] = []
        self._part_kwargs: dict = {}
        self._worker: _ImportWorker | None = None
        self._cancel = threading.Event()
        self._items: dict[str, RecordingImportSpec] = {}  # queued/importing → source
        #: Everything submitted since the queue was last idle; ``None`` while idle.
        self._fold: ImportProgressFold | None = None

    def submit(
        self, specs: list[RecordingImportSpec], part_kwargs: dict | None = None
    ) -> None:
        create_shells(self._experiment, specs)  # the whole bout appears at once
        deferred = [s for s in specs if s.has_deferred_work]
        for spec in deferred:
            self._items[spec.id] = spec
        self._pending.extend(deferred)
        if part_kwargs:
            self._part_kwargs = part_kwargs
        if deferred:
            if self._fold is None:
                self._fold = ImportProgressFold()
            # The bar appears the moment the work is queued, not when the first file opens.
            self.progress.emit(self._fold.add(deferred))
        self._drain()

    @property
    def active(self) -> bool:
        return self._worker is not None or bool(self._pending)

    def source_for(self, recording_id: str) -> str | None:
        """A short display of a queued/importing recording's pending source (from the
        in-memory queue; never persisted). ``None`` once imported/terminal."""
        spec = self._items.get(recording_id)
        if spec is None:
            return None
        if spec.video_parts:
            first = Path(spec.video_parts[0]).name
            extra = len(spec.video_parts) - 1
            return f"{first}  (+{extra})" if extra else first
        if spec.streams:
            return f"{len(spec.streams)} stream(s)"
        return None

    def cancel(self) -> None:
        """Stop the queue: the in-flight recording finishes; remaining queued ones are
        marked ``cancelled`` (kept, tags preserved). The worker's current batch is
        handled by ``import_pending``; not-yet-started batches are marked here.

        The stop is for **what is queued now**. With no worker running there is nothing to
        stop, and the event is left clear: set, it waited for the next ``submit`` and handed
        that batch a stop nobody asked for, so a recording added after a Cancel imported as
        ``cancelled`` with its video never copied (#35). A Cancel enabled by a table read a
        moment before the queue went idle — or the quit path, which decides from run records
        rather than from this queue — reaches here with nothing running.
        """
        if self._worker is not None:
            self._cancel.set()
        for spec in self._pending:  # queued-but-never-handed-to-a-worker
            with contextlib.suppress(FileNotFoundError, ValueError):
                recording_store.set_import_status(
                    self._experiment.get_recording(spec.id).dir, "cancelled"
                )
        self._pending.clear()

    def _drain(self) -> None:
        if self._worker is not None or not self._pending:
            return
        batch, self._pending = self._pending, []
        self._worker = _ImportWorker(
            self._experiment,
            batch,
            dict(self._part_kwargs),
            self._cancel,
            receiver=self,  # the queue outlives every page; it is what re-emits to them
        )
        self._worker.item_started.connect(self._on_item_started)
        self._worker.item_progress.connect(self._on_item_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.start()

    def _on_item_started(self, recording_id: str) -> None:
        self.item_started.emit(recording_id)
        if self._fold is not None:
            self.progress.emit(self._fold.started(recording_id))

    def _on_item_progress(self, recording_id: str, fraction: float, phase: str) -> None:
        if self._fold is not None:
            self.progress.emit(self._fold.progress(recording_id, fraction, phase))

    def _on_item_done(self, recording_id: str, ok: bool, error: str) -> None:
        self._items.pop(recording_id, None)  # no longer a pending source
        self.item_done.emit(recording_id, ok, error)
        if self._fold is not None:
            self.progress.emit(self._fold.finished(recording_id, ok))

    def _on_batch_finished(self) -> None:
        self._worker = None
        # A stop applied to the batch that just ended. `cancel` emptied `_pending` when it was
        # asked, so anything pending now was submitted *after* it and is not stopped by it —
        # leaving those behind a set event stranded them as "queued" for good.
        self._cancel.clear()
        if self._pending:
            self._drain()  # specs queued while this batch ran
        else:
            self._items.clear()  # any cancelled remainder are terminal now
            self._fold = None  # the next submit starts a fresh count and clock
            self.idle.emit()
