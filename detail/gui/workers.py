"""QThread workers + the orchestration logic they wrap.

Long operations never block the UI thread (CLAUDE.md). The orchestration logic
is split into plain functions (testable without Qt) that the QThread workers
call, emitting progress as Qt signals.
"""

from collections.abc import Callable

from detail.core.progress import format_duration, remaining_seconds
from detail.gui.lifetime import alive
from detail.gui.qt import QObject, QThread, Signal


def _with_eta(status: str, eta_s: float | None) -> str:
    """Append ``· ~<duration> left`` to a status line when an ETA is available and
    at least a second remains. One place so every bar renders ETA the same way."""
    if eta_s is not None and round(eta_s) >= 1:
        return f"{status} · ~{format_duration(eta_s)} left"
    return status


#: Workers with an emission in flight, so a caller's scope ending cannot collect them.
#:
#: ``run_busy`` has always rooted its worker in ``busy._active``; a **directly constructed**
#: ``BackgroundCall`` had no such root, and that is a correctness problem rather than a tidiness
#: one. The widget that started one typically holds it as ``self._call`` and connects a closure
#: back to itself, so worker → connection → closure → widget → worker is a cycle with nothing
#: outside it. When the widget goes out of scope the *cyclic* collector breaks that cycle by
#: **clearing the closure** — ``func_clear`` drops ``__defaults__`` and blanks ``__qualname__``.
#:
#: The C++ side of the connection outlives that clear. So a queued ``done`` posted before the
#: collection is later delivered into a cleared function, and a closure written in the ordinary
#: ``lambda result, g=generation:`` idiom is suddenly missing an argument it always supplied
#: itself. The failure surfaces as ``TypeError: () missing 1 required positional argument: 'g'``
#: with **no traceback and no function name** — raised at the C++ boundary, before any Python
#: frame — and lands in whichever test happens to be pumping the event loop, which is why it read
#: as an unrelated flake for weeks.
#:
#: Rooting here rather than at each call site is deliberate: the four sites that had the idiom are
#: not special, and the next one written would have inherited the bug.
_IN_FLIGHT: set = set()

#: Every started worker Qt has not finished with: what keeps a running thread's object alive.
#:
#: **Destroying a QThread whose state is Running is a Qt fatal abort** (``QThread: Destroyed
#: while thread '' is still running``), not an exception, and "Running" is wider than "inside
#: ``run()``": it starts when ``start()`` returns, before the thread has entered ``run()``, and
#: it lasts after ``run()`` returns until Qt's own exit code marks the thread finishing. A
#: worker held only by its page — or one that released its own root at the end of ``run()``, as
#: the montage's picture worker did — is unreachable while Running in one of those two gaps
#: whenever its page is already garbage, and a collection on the UI thread there ends the
#: process. Qt's ``finished`` is emitted *after* the state has left Running (``~QThread`` waits
#: for a finishing thread rather than aborting), so that signal, delivered on the UI thread, is
#: the only safe place to let go: ``tests/gui/test_thread_outlives_its_running.py``.
#:
#: On the base class, so a worker written tomorrow has it: before this there were three
#: per-class copies (``_IN_FLIGHT``'s two users, ``_LIVE_WORKERS``, the montage's ``_RUNNING``)
#: and four workers with none.
_UNFINISHED: set = set()


def _release(worker: "GuiWorker") -> None:
    """``worker``'s thread has finished (delivered on the UI thread): stop holding it.

    A worker started again since — a picture worker woken from idle — is Running: its next
    ``finished`` releases it. ``wait()`` joins a thread still in Qt's exit code, so the object
    is never destroyed mid-exit; one whose C++ side is already gone (``deleteLater`` on the
    same signal, ``busy.py``) has nothing left to protect.
    """
    if alive(worker):
        if worker.isRunning():
            return
        worker.wait()
    _UNFINISHED.discard(worker)


class GuiWorker(QThread):
    """Base for every GUI thread: a result that arrives after its widget is gone is dropped.

    Qt severs a connection when the **receiver QObject** is destroyed, so
    ``worker.done.connect(self._on_done)`` is safe for free. A closure has no receiver —
    ``connect(lambda paths, g=generation: self._apply(g, paths))``, the ordinary idiom for
    carrying a generation — so there is nothing for Qt to sever, and a delivery posted
    before the page was torn down runs against deleted widgets.

    A generation counter cannot stand in for this. Generations express **supersession**
    (a later request owns the view now); destruction is **liveness**, and nothing bumps a
    counter on the way out — the shell tears a page down with ``deleteLater``, which runs
    no page code at all.

    So the receiver is named once, where the thread is created, and every emission goes
    through :meth:`emit_to_ui`: the payload takes one hop through ``self`` — which lives on
    the UI thread — and is dropped there if the receiver's C++ side has gone. ``receiver``
    has no default: a thread that reports to a widget and one that reports to nothing are
    different things, and the difference is invisible at the line where it is written.

    **Self-rooting while Qt counts it as running** (:data:`_UNFINISHED`): held from
    :meth:`start` until its ``finished`` lands on the UI thread, so no caller has to keep a
    running thread alive and none can drop one. Construct it on the UI thread: that is where
    the release is delivered. A subclass that overrides :meth:`start` calls ``super().start``.
    """

    #: ``(bound signal, args)`` — the hop that puts the liveness check on the UI thread.
    #: Emitted from the worker thread; ``self`` lives on the UI thread, so Qt queues it
    #: there and ``_deliver`` re-emits the real signal from the thread widgets live on.
    _delivery = Signal(object, tuple)

    def __init__(self, *, receiver: QObject | None) -> None:
        super().__init__()
        self._receiver = receiver
        self._delivery.connect(self._deliver)
        # A closure, not a bound method: it must still run when this object's C++ side has
        # been deleted on the same signal, to drop the Python side from the set.
        self.finished.connect(lambda: _release(self))

    def start(self, *args, **kwargs) -> None:  # noqa: D102 — QThread's own docstring
        _UNFINISHED.add(self)
        super().start(*args, **kwargs)

    def emit_to_ui(self, signal, *args: object) -> None:
        """Emit ``signal`` on the UI thread — unless ``receiver`` is gone by the time it
        lands, in which case the emission is dropped."""
        self._delivery.emit(signal, args)

    def _deliver(self, signal, args: tuple) -> None:
        # `receiver=None` means "reports to nothing that can be destroyed" — deliver always.
        if self._receiver is not None and not alive(self._receiver):
            return
        signal.emit(*args)


class BackgroundCall(GuiWorker):
    """Run a 0-arg callable off the UI thread; emit its result or the exception.

    For one-shot, no-progress operations that can still be slow (a multi-GB
    experiment export/import) — keeps the shell responsive without bespoke workers.

    **Self-rooting while it runs** (see :data:`_IN_FLIGHT`): the instance keeps itself reachable
    from ``start()`` until it has emitted, so neither it nor the closures connected to it can be
    collected with a delivery still queued. ``receiver`` names the widget the result is *for*
    (:class:`GuiWorker`) — being rooted is what keeps a delivery from arriving corrupted, and
    naming a receiver is what keeps it from arriving at a widget that no longer exists.
    """

    done = Signal(object)  # the callable's return value
    failed = Signal(object)  # the raised exception

    def __init__(self, fn: Callable[[], object], *, receiver: QObject | None) -> None:
        super().__init__(receiver=receiver)
        self._fn = fn

    def start(self, *args, **kwargs) -> None:  # noqa: D102 — QThread's own docstring
        _IN_FLIGHT.add(self)
        # `finished` fires on the worker thread's exit, after `run` has emitted, so the root is
        # held across the whole window in which a delivery can still be queued.
        self.finished.connect(lambda: _IN_FLIGHT.discard(self))
        super().start(*args, **kwargs)

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 — boundary: report to the UI
            self.emit_to_ui(self.failed, exc)
            return
        self.emit_to_ui(self.done, result)


class StageProgressWorker(GuiWorker):
    """Run a synchronous op that drives a ``(stage, fraction, detail)`` progress callback
    off the UI thread — a tool run or an mp4 export (§27). Emits ``tick`` (permille +
    status) so a page can drive a ``QProgressBar``, then ``done`` / ``failed``.

    The synchronous-callback twin of ``RunProgressWorker`` (which polls a run handle's
    ``ProcessingProgress``); a tool runs in-process, so there is no handle to poll."""

    tick = Signal(int, str)  # permille 0..1000, status
    done = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        work: Callable[[Callable[[str, float, str], None]], object],
        *,
        receiver: QObject | None,
    ) -> None:
        super().__init__(receiver=receiver)
        self._work = work
        self._stop = False
        self._cancel = False

    def start(self, *args, **kwargs) -> None:  # noqa: D102 — QThread's own docstring
        # Self-rooting while it runs, as BackgroundCall is (see _IN_FLIGHT): a caller that
        # connects closures and keeps the worker only on its own widget is that cycle.
        _IN_FLIGHT.add(self)
        self.finished.connect(lambda: _IN_FLIGHT.discard(self))
        super().start(*args, **kwargs)

    def request_stop(self) -> None:
        """Ask the work to end early. It is polled, not interrupted: the op finishes the
        frame it is on and returns its partial result, so a stopped run is still openable.
        """
        self._stop = True

    def stop_requested(self) -> bool:
        return self._stop

    def run(self) -> None:
        try:
            result = self._work(self._emit)
        except Exception as exc:  # noqa: BLE001 — boundary: report to the UI
            self.emit_to_ui(self.failed, exc)
            return
        self.emit_to_ui(self.done, result)

    def request_cancel(self) -> None:
        """Abandon the work: its next progress report raises ``OperationCancelled``.

        Not :meth:`request_stop`, which keeps a partial result and is polled by work that
        knows how to finish early (a comparison, a tool run). Cancel needs nothing from the
        work but its reports, so any op that reports is cancellable as it stands.
        """
        self._cancel = True

    def cancel_requested(self) -> bool:
        return self._cancel

    def _emit(self, stage: str, fraction: float, detail: str) -> None:
        if self.cancel_requested():
            from detail.core.progress import OperationCancelled

            raise OperationCancelled(stage)
        status = f"{stage} — {detail}" if detail else stage
        self.emit_to_ui(self.tick, round(fraction * 1000), status)


def _recording_position(done: int, total: int) -> int:
    """1-based index of the recording in flight, for a done-count of ``done``.

    ``done`` counts *finished* recordings, so the live one is ``done + 1`` — except on a
    terminal event, where every recording is finished and ``done == total`` would render
    the nonsensical "recording 2/1". Mirrors ``cli.py::_recording_position``; the two
    surfaces keep their own renderers (tqdm owns its ETA) but must not disagree on this.
    """
    return min(int(done) + 1, int(total) or 1)


def fit_progress_fraction(event) -> tuple[float, str]:
    """Map a fit run's progress event to ``(fraction 0..1, status text)``.

    Duck-types the shapes a fit/analysis handle yields: training
    (``TrainingProgressEvent``: ``epoch`` / ``total_epochs`` + ``metrics``), an
    analysis run (``recordings_done`` / ``recordings_total`` + ``message``), and
    dataset export (``LabelingProgressEvent``: ``frames_done`` / ``frames_total``).
    """
    # A distill event's fraction is stage-*weighted* while its elapsed_s is wall-clock, so
    # the two are not comparable and the generic fallback below must not pair them — doing
    # so is the ~5x ETA under-report this mapping used to reproduce. Its runner supplies the
    # authoritative eta_s; when it has none, show none.
    trust_event_eta_only = hasattr(event, "overall_fraction") or getattr(
        event, "eta_from_runner_only", False
    )
    if getattr(event, "overall_fraction_is_frames", False):
        # A recording-keyed event whose frames_* is already the whole-run fraction (the removed
        # clusterer fit set this; nothing does today). Folding it into recordings_done would
        # double-count: that fit's DTW phase held recordings_done at the total while the
        # fraction climbed, so the bar would read >100%.
        total = event.frames_total or 0
        fraction = event.frames_done / total if total else 0.0
        status = f"{event.recordings_total} recording(s)"
        if event.message:
            status += f" — {event.message}"
    elif hasattr(event, "overall_fraction"):  # distill run (§12): folded label → train
        fraction = event.overall_fraction
        status = f"{event.phase} ({event.stage_index + 1}/{event.stage_count})"
        if event.message:
            status += f" — {event.message}"
    elif hasattr(event, "epoch"):  # training
        total = event.total_epochs or 0
        fraction = event.epoch / total if total else 0.0
        status = f"epoch {event.epoch}/{total}"
        metrics = event.metrics or {}
        if metrics:
            status += "  ·  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        if getattr(event, "message", None):  # e.g. "loading base model"
            status += f" — {event.message}"
    elif hasattr(event, "recordings_total"):  # analysis run (§16)
        stotal = event.recordings_total or 0
        within = event.frames_done / event.frames_total if event.frames_total else 0.0
        fraction = (event.recordings_done + within) / stotal if stotal else 0.0
        # Always show recordings completed/total AND the phase message (not either/or). An event
        # names its own units, as the CLI's `_stream_*` reads it: an export's are *targets*, and
        # "recording 1/2" over a bundle and its metadata CSV is a wrong answer (#916).
        noun = getattr(event, "unit_label", None) or "recording"
        status = f"{noun} {_recording_position(event.recordings_done, stotal)}/{stotal}"
        if event.message:
            status += f" — {event.message}"
    else:  # dataset export (labeling)
        total = event.frames_total or 0
        fraction = event.frames_done / total if total else 0.0
        status = f"frame {event.frames_done}/{total}"
        if getattr(event, "message", None):  # e.g. "loading model"
            status += f" — {event.message}"
    # Whole-operation ETA: a fit event's elapsed_s is measured from the run start,
    # so it pairs with the overall fraction. Prefer a runner-supplied eta_s.
    eta_s = event.eta_s
    if eta_s is None and not trust_event_eta_only:
        eta_s = remaining_seconds(fraction, event.elapsed_s)
    return fraction, _with_eta(status, eta_s)


class FitProgressWorker(GuiWorker):
    """Consume a run handle's ``progress()`` off the UI thread, forwarding each event as a
    Qt signal (the run executes in its own worker process; this only marshals progress to
    the UI).

    Outlives the page that started it (:data:`_UNFINISHED`): its only other Python reference
    lives on that page, and the shell's page swap ``deleteLater``s the page while the thread
    is still polling its run. A late tick cannot reach the dead page either: Qt disconnects a
    receiver automatically when it is destroyed."""

    tick = Signal(int, str)  # permille (0..1000), status text
    #: The raw event itself, for a surface that shows the run's *parts* rather than only the
    #: folded bar — the ``RunProgressWorker.progress`` signal's sibling, and the same reason:
    #: `tick` is a rendering (a fraction and a sentence), and a distill's two stages each need
    #: their own fraction, which is on the event and thrown away by `fit_progress_fraction`
    #: before it reaches the page.
    progress = Signal(object)
    finished_run = Signal(str)  # the run's error message, "" if it succeeded

    def __init__(self, handle, *, receiver: QObject | None) -> None:
        super().__init__(receiver=receiver)
        self._handle = handle

    def run(self) -> None:
        for event in self._handle.progress():
            fraction, status = fit_progress_fraction(event)
            self.emit_to_ui(self.tick, round(fraction * 1000), status)
            self.emit_to_ui(self.progress, event)
        error = self._handle.error if self._handle.status == "failed" else None
        self.emit_to_ui(self.finished_run, error or "")


def processing_progress_fraction(event) -> tuple[float, str]:
    """Map a ``ProcessingProgress`` (a multi-recording run) to ``(fraction 0..1, status)``.

    Folds the active recording's inner per-stage progress into the aggregate so the bar
    advances *within* a recording, not only when one completes:
    ``(recordings_done + inner_fraction) / recordings_total``. A bar that reads only
    ``recordings_done`` sits frozen through a long recording (the recurring progress bug).
    Every surface renders progress through a shared mapping like this — see DESIGN_PLAN
    'Progress reporting'.
    """
    total = event.recordings_total or 0
    within = 0.0
    label = (
        f"recording {_recording_position(event.recordings_done, total)}/{total or 1}"
    )
    inner = event.inner
    if inner is not None:
        frames_total = getattr(inner, "frames_total", 0) or 0
        frames_done = getattr(inner, "frames_done", 0) or 0
        if frames_total:
            within = min(frames_done / frames_total, 1.0)
        label = (
            f"{event.current_recording} "
            f"({_recording_position(event.recordings_done, total)}/{total or 1})"
        )
        message = getattr(inner, "message", None)
        if message:
            label += f" — {message}"
    fraction = (event.recordings_done + within) / total if total else 0.0
    # A whole-run ETA from the elapsed the coordinator publishes — more useful for a
    # durable multi-recording run than a per-recording estimate. The "(N/total)" above
    # always shows recordings completed/left.
    #
    # The ETA divides by the work done *since this coordinator started*, not the
    # run-wide fraction: on a retry (§15) the bar legitimately starts at 29/60, but
    # elapsed_s only ever buys the recordings run from that mark. Feeding it the run-wide
    # fraction infers a 30x throughput and reads "30m left" on a 15h retry.
    at_start = getattr(event, "recordings_at_start", 0) or 0
    own_total = total - at_start
    own_fraction = (
        (event.recordings_done + within - at_start) / own_total
        if own_total > 0
        else 0.0
    )
    label = _with_eta(
        label, remaining_seconds(own_fraction, getattr(event, "elapsed_s", 0.0))
    )
    return fraction, label


def import_progress_fraction(event) -> tuple[float, str]:
    """Map an ``ImportProgress`` (the recording import queue, §3) to ``(fraction, status)``.

    The import queue's twin of ``processing_progress_fraction``: the same fold (finished
    recordings plus each in-flight one's own fraction, which ``ImportProgress.fraction`` owns),
    the same ``(N/total)`` position, the same ETA rendering. Under transcode parallelism several
    recordings import at once, so the line says how many and names the one that started first —
    naming whichever reported last would flicker between them on every tick.
    """
    total = event.recordings_total
    done = event.recordings_done
    if len(event.in_flight) == 1:
        name, _, phase = event.in_flight[0]
        status = (
            f"Importing {name} ({_recording_position(done, total)}/{total or 1})"
            f" — {phase}"
        )
    elif event.in_flight:
        name, _, phase = event.in_flight[0]
        status = (
            f"Importing {len(event.in_flight)} recordings at once"
            f" ({done}/{total} done) — {name}: {phase}"
        )
    else:
        status = f"Importing recordings ({done}/{total} done)"
    if event.failed:
        status += f" · {event.failed} failed"
    return event.fraction, _with_eta(status, event.eta_s)


class RunProgressWorker(GuiWorker):
    """Consume an ``AttachedRun`` handle's ``progress()`` off the UI thread and
    forward each aggregate event as a fractional bar update — folding in the active
    recording's inner per-stage progress (``processing_progress_fraction``) so the bar
    moves within a recording, not only at recording boundaries."""

    tick = Signal(int, str)  # permille (0..1000), status text
    #: The aggregate event itself, for a surface that shows the run's *parts*.
    #:
    #: `tick` is a rendering — a fraction and a sentence — and for a long time it was all the UI
    #: thread ever saw, so the only thing a page could draw was one bar and one line of text.
    #: That is what made a large run a scrolling log: the structure (which recording, how
    #: far into it, what is still queued) existed in the worker and was flattened before it got
    #: out. Both signals, because the two answer different questions and the bar's rendering is
    #: pinned by tests that read its string.
    progress = Signal(object)  # ProcessingProgress
    finished_run = Signal()

    def __init__(self, handle, *, receiver: QObject | None) -> None:
        super().__init__(receiver=receiver)
        self._handle = handle

    def run(self) -> None:
        for event in self._handle.progress():
            fraction, status = processing_progress_fraction(event)
            self.emit_to_ui(self.tick, round(fraction * 1000), status)
            self.emit_to_ui(self.progress, event)
        self.emit_to_ui(self.finished_run)


class InstallModelWorker(GuiWorker):
    """Download a model's weights off the UI thread (§5), forwarding the byte-level
    ``InstallProgress`` stream as a Qt signal so the Models tab can show a bar."""

    progress = Signal(int, int, str)  # bytes_done, bytes_total (-1 if unknown), file
    finished_install = Signal(str)  # "" on success, else the error message

    def __init__(
        self,
        model_id: str,
        variant: str | None = None,
        *,
        receiver: QObject | None,
    ) -> None:
        super().__init__(receiver=receiver)
        self._model_id = model_id
        self._variant = variant  # each size is a separate download (§5)

    def run(self) -> None:
        from detail.api import models as api_models

        def on_progress(p) -> None:
            self.emit_to_ui(
                self.progress,
                p.bytes_done,
                p.bytes_total if p.bytes_total is not None else -1,
                p.current_file or "",
            )

        try:
            api_models.install_model(self._model_id, on_progress, variant=self._variant)
        except Exception as exc:  # noqa: BLE001 — boundary: report to the UI
            self.emit_to_ui(self.finished_install, str(exc))
            return
        self.emit_to_ui(self.finished_install, "")
