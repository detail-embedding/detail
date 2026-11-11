"""Indeterminate "busy" + one-shot progress primitives for the GUI.

The executor bar (``workers.RunProgressWorker``) covers long batch runs and
``FrameScrubber`` covers off-thread frame decode. This module fills the remaining
gap: the *indeterminate* and *one-shot* waits that were previously hand-rolled (a
greyed button, a bare ``QLabel("Loading…")``) or left blocking the UI thread (a
black preview, a frozen dialog).

- :class:`BusyOverlay` — a neutral cover over a widget's content with a centered
  label and an **indeterminate** ``QProgressBar`` (the marquee ``InstallModelWorker``
  already uses, the one sanctioned "animation"). It has a determinate mode so a sized
  one-shot renders through the same widget, and shows only after a short delay so an
  instant op never flashes.
- :func:`run_busy` — run a callable off the UI thread (``BackgroundCall``), cover
  ``target`` with a busy overlay while it runs, then deliver the result or surface the
  error. The single entry point for an indeterminate GUI wait.
- :func:`run_progress` — the determinate twin (``StageProgressWorker``), for a
  one-shot that drives a ``(stage, fraction, detail)`` callback.

The callable runs on a worker thread, so it must do no Qt; the ``on_done`` /
``on_error`` handlers run on the UI thread and may touch widgets.
"""

from collections.abc import Callable
import time
import weakref

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPalette
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.gui.lifetime import alive
from detail.gui.workers import BackgroundCall, StageProgressWorker

SHOW_DELAY_MS = 150  # don't flash an indicator for an op that finishes in a blink


def with_remaining(status: str, fraction: float, elapsed_s: float) -> str:
    """``status`` with the time left appended, when there is enough progress to say.

    `run_progress`'s one rendering of an ETA, kept a function so it is tested apart from a
    widget. Nothing is added before the work has moved, or once it is done.
    """
    from detail.core.progress import format_duration, remaining_seconds

    remaining = remaining_seconds(fraction, elapsed_s)
    if remaining is None:
        return status
    return f"{status} · about {format_duration(remaining)} left"


class BusyOverlay(QWidget):
    """A cover over ``target``'s content: a scrim, a centered label, and a progress
    bar. Indeterminate by default; :meth:`set_fraction` switches it to determinate so
    a sized one-shot uses the same widget. Reveals only after ``SHOW_DELAY_MS``.

    **Covering and revealing are two different moments.** The delay exists so an op that
    finishes in a blink never flashes an indicator — it is about *appearance*. It must not
    also decide when the target stops taking input, which is what it used to do: the overlay
    was hidden for the first ``SHOW_DELAY_MS``, so every :func:`run_busy` caller sat live over
    state its load had not produced yet. So a blocking overlay is shown — and therefore
    hit-tested — from :meth:`start`, and paints nothing until the delay elapses.

    ``blocks_input`` is the one genuine choice here, and it is ``True`` by default because
    that is what "busy" means. :func:`decode_overlay` is the sole opt-out and says why.
    """

    def __init__(
        self, target: QWidget, *, scrim_alpha: int = 205, blocks_input: bool = True
    ) -> None:
        super().__init__(target)
        self._target = target
        self._scrim_alpha = scrim_alpha
        self._blocks_input = blocks_input
        self._revealed = False
        if not blocks_input:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QVBoxLayout(self)
        layout.addStretch()
        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._label)
        self._bar = QProgressBar(self)
        self._bar.setTextVisible(False)
        self._bar.setMaximumWidth(240)
        bar_row = QHBoxLayout()
        bar_row.addStretch()
        bar_row.addWidget(self._bar)
        bar_row.addStretch()
        layout.addLayout(bar_row)
        # Built only by :meth:`offer_cancel`: an overlay that offers no Cancel holds no
        # button, so a widget walk over a page mid-load finds nothing new and live in it.
        self._cancel: QPushButton | None = None
        self._layout = layout
        layout.addStretch()

        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.setInterval(SHOW_DELAY_MS)
        self._delay.timeout.connect(self._reveal)

        self.hide()

    def start(self, label: str) -> None:
        """Begin an indeterminate busy state; reveal after the show delay."""
        # The filter is installed here and removed in `stop`, rather than held from `__init__`
        # for the object's life. An overlay is parented to its target, so the **C++** object and
        # its installed filter outlive the last Python reference to the wrapper; Qt then delivers
        # a resize into a freshly materialized wrapper that never ran `__init__`, and
        # `self._target` does not exist. Same class as the cleared closure in `workers.py`: C++
        # state outliving the Python object it needs. Scoping the filter to the window where
        # something actually holds the overlay closes it.
        if alive(self._target):
            self._target.installEventFilter(self)
        self._label.setText(label)
        self._bar.setRange(0, 0)  # marquee
        self._sync_geometry()
        # Cover now, show later. `paintEvent` draws nothing and the label and bar are hidden
        # until `_reveal`, so this is invisible — but it is a real widget on top of the target,
        # so Qt's hit-testing routes clicks here instead of into a page mid-load.
        self._revealed = False
        self._set_content_visible(False)
        if self._blocks_input:
            self.raise_()
            self.show()
        self._delay.start()

    def _set_content_visible(self, visible: bool) -> None:
        """Show or hide the overlay's own chrome.

        Separate from ``show()``/``hide()`` on the overlay itself: a blocking overlay is
        *shown* the whole time it is active, so hiding the widget is not available as a way to
        stay invisible during the delay. These two children are the only things it draws.
        """
        self._label.setVisible(visible)
        self._bar.setVisible(visible)
        if self._cancel is not None:
            self._cancel.setVisible(visible)

    def offer_cancel(self, on_cancel: Callable[[], None]) -> None:
        """Show a Cancel button under the bar; a click calls ``on_cancel`` once and says
        "Cancelling…" until the work reaches its next progress report and stops."""

        button = QPushButton("Cancel", self)
        button.setAutoDefault(False)

        def clicked() -> None:
            button.setEnabled(False)
            self._label.setText("Cancelling…")
            on_cancel()

        button.clicked.connect(clicked)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button)
        row.addStretch()
        self._layout.insertLayout(self._layout.count() - 1, row)
        self._cancel = button
        button.setVisible(self._revealed)

    def set_fraction(self, fraction: float, text: str = "") -> None:
        """Switch to a determinate bar at ``fraction`` (0..1), updating the label."""
        self._bar.setRange(0, 1000)
        self._bar.setValue(max(0, min(1000, round(fraction * 1000))))
        cancelling = self._cancel is not None and not self._cancel.isEnabled()
        if text and not cancelling:
            self._label.setText(text)

    def stop(self) -> None:
        self._delay.stop()
        if alive(getattr(self, "_target", None)):
            self._target.removeEventFilter(self)
        self._revealed = False
        self.hide()

    def _reveal(self) -> None:
        self._revealed = True
        self._set_content_visible(True)
        self._sync_geometry()
        self.raise_()
        self.show()
        self.update()  # `_revealed` changed what paintEvent draws

    def _sync_geometry(self) -> None:
        self.setGeometry(self._target.rect())

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # `getattr`, not `self._target`: a wrapper Qt materialized for a C++ object whose Python
        # side was collected has no instance state at all, and raising here escapes into the
        # excepthook where it is swallowed. Inert is the right behaviour for an overlay nobody
        # holds any more.
        target = getattr(self, "_target", None)
        if target is not None and obj is target and event.type() == QEvent.Type.Resize:
            self._sync_geometry()
        return super().eventFilter(obj, event)

    def paintEvent(self, event) -> None:
        if not self._revealed:
            return  # covering, not yet showing — the target draws through untouched
        painter = QPainter(self)
        scrim = self.palette().color(QPalette.ColorRole.Window)
        scrim.setAlpha(self._scrim_alpha)
        painter.fillRect(self.rect(), scrim)


def decode_overlay(
    scrubber,
    target: QWidget,
    *,
    label: str = "Decoding frame…",
):
    """Wire a ``FrameScrubber``'s ``busy_changed`` to a light :class:`BusyOverlay` over
    ``target``, so a frame decode (first-load or a slow scrub) shows ``label`` instead
    of a blank pane. Returns the overlay (a child of ``target``, so it lives as long).

    The scrim is light so the previously-shown frame stays faintly visible while the
    next decodes; on first-load (nothing shown yet) the label reads over the blank.

    **A streaming host is asked, not told.** This overlay means "you are waiting for a
    frame", and ``busy_changed`` means "a decode is in flight" — the same thing for a scrub,
    and not the same thing at all for autoplay, where a decode is always in flight and frames
    are landing 25 times a second. A player that cannot quite keep up never drains the queue,
    so ``busy_changed(False)`` never arrives and the scrim latches over frames that are
    arriving normally.

    That used to be a ``suppress_when`` predicate the host passed in, defaulting to "never
    suppress". One of eight embedders passed it, and the seven that did not are all playable —
    so seven surfaces played video under a permanent "Decoding frame…". The one predicate
    anybody wrote was ``return self._scrubber.is_playing()``: a fact about the scrubber this
    function already receives, which makes it derivable rather than a choice, and a parameter
    whose default means "as before" is where a missing fact hides.

    **The one overlay that does not block input**, and the reason is the same one: this is an
    indicator over a *live* view, not a modal wait. A decode is in flight most of the time a
    scrubber is being dragged, so a blocking cover here would eat the very clicks that drive
    the next decode. Nothing behind it reads state the decode produces — the frame arrives or
    it does not — which is exactly what makes it safe, and is not true of a ``run_busy`` load.
    """
    overlay = BusyOverlay(target, scrim_alpha=130, blocks_input=False)

    def _on_busy(busy: bool) -> None:
        if busy and not scrubber.is_playing():
            overlay.start(label)
        else:
            overlay.stop()

    scrubber.busy_changed.connect(_on_busy)
    return overlay


def _reapply_dialog_policy(widget) -> None:
    """Re-apply a `Dialog`'s show-time policies after a callback has changed its content.

    `Dialog.showEvent` makes text selectable, top-justifies the body and refits — for the children
    that exist *at show*. A `run_busy` callback delivers after that, so a dialog that builds its
    labels in the callback (`StreamPreviewDialog._populate`) got none of the three (#563).

    Duck-typed on purpose: `gui/dialog.py` imports enough of the package that importing it here
    would risk a cycle, and the only thing this needs is "does the host carry the policy".
    Walks up because the busy target is often a page or body inside the dialog, not the dialog.
    """
    node = widget
    while node is not None:
        if not alive(node):
            return
        refresh = getattr(node, "refresh_content_policy", None)
        if callable(refresh):
            if node.isVisible():  # not yet shown -> showEvent will do it, and better
                refresh()
            return
        node = node.parent()


def _show_error(target: QWidget, exc: BaseException) -> None:
    parent = target if alive(target) else None
    QMessageBox.critical(parent, "Error", str(exc) or exc.__class__.__name__)


# Threads/overlays are kept here so they outlive the calling scope until they finish;
# each entry is discarded when its worker emits done/failed.
_active: set = set()

# The newest call per (target, key) for callers that pass `supersedes`. Weak on the target, so a
# torn-down widget takes its entries with it.
_latest: "weakref.WeakKeyDictionary[QWidget, dict[str, object]]" = (
    weakref.WeakKeyDictionary()
)


def _claim(target: QWidget, key: str | None) -> Callable[[], bool]:
    """Make this the newest call for ``key`` on ``target``; return "is it still?".

    Without a key every call is current, which is the old behaviour. With one, a call started
    later for the same key makes this one's result a no-op: two overlapping loads otherwise
    deliver in *completion* order, and the slower, older one wins — AUDIT showed a stale
    "nothing to audit" after a cutoff change that way (13 of 86 full runs of one test).
    """
    if key is None:
        return lambda: True
    token = object()
    _latest.setdefault(target, {})[key] = token
    return lambda: _latest.get(target, {}).get(key) is token


def run_busy(
    target: QWidget,
    fn: Callable[[], object],
    on_done: Callable[[object], None],
    *,
    label: str = "Loading…",
    on_error: Callable[[BaseException], None] | None = None,
    supersedes: str | None = None,
) -> BackgroundCall:
    """Run ``fn()`` off the UI thread while a busy overlay covers ``target``.

    Delivers ``fn``'s result to ``on_done`` (UI thread) or its exception to
    ``on_error`` (default: a critical message box). ``target`` is also the worker's
    receiver, so if it is torn down before the call returns the result is dropped rather
    than touching a dead widget (:class:`~detail.gui.workers.GuiWorker`).

    ``supersedes`` names what the call loads. A later call on the same ``target`` with the same
    name drops this one's result and error, so the newest request decides the screen whatever
    order the threads finish in. A method that can run again while its last call is in flight
    passes one (``requirements.a_reload_supersedes_its_last_call``).
    """
    overlay = BusyOverlay(target)
    overlay.start(label)
    worker = BackgroundCall(fn, receiver=target)
    _active.add(worker)
    current = _claim(target, supersedes)

    def clear() -> None:
        _active.discard(worker)
        if alive(overlay):
            overlay.stop()
            overlay.deleteLater()

    def deliver(result: object) -> None:
        clear()
        if not current():
            return
        on_done(result)
        _reapply_dialog_policy(target)

    def fail(exc: BaseException) -> None:
        clear()
        if not current():
            return
        (on_error or (lambda e: _show_error(target, e)))(exc)

    worker.done.connect(deliver)
    worker.failed.connect(fail)
    # `clear` is where the register is normally released, and a torn-down target no longer
    # reaches it — `GuiWorker` drops that delivery. So release on the thread's own exit too,
    # or a call abandoned by a page swap would sit in `_active` for the life of the app.
    # `finished` is queued behind the delivery, so the ordinary path still clears first.
    worker.finished.connect(lambda: _active.discard(worker))
    worker.finished.connect(worker.deleteLater)
    worker.start()
    return worker


def _cancel_with_window(target: QWidget, worker: StageProgressWorker) -> None:
    """Cancel ``worker`` when ``target``'s window is closed or destroyed.

    Reported from the app: Add frames went on copying after its window closed, and opening the
    window again started a second copy beside the first. A dialog closed by Close or Esc
    emits ``finished`` and no close event, so that is the signal for a dialog; ``destroyed``
    covers everything else. ``alive`` because the worker may already be gone when either fires.
    """
    from PyQt6.QtWidgets import QDialog

    def cancel(*_args) -> None:
        if alive(worker):
            worker.request_cancel()

    window = target.window()
    if isinstance(window, QDialog):
        window.finished.connect(cancel)
    window.destroyed.connect(cancel)


def run_progress(
    target: QWidget,
    work: Callable[[Callable[[str, float, str], None]], object],
    on_done: Callable[[object], None],
    *,
    label: str = "Working…",
    on_error: Callable[[BaseException], None] | None = None,
    supersedes: str | None = None,
    on_cancel: Callable[[], None] | None = None,
    eta: bool = False,
) -> StageProgressWorker:
    """The determinate twin of :func:`run_busy` for a sized one-shot.

    ``eta`` adds the time remaining to the label, extrapolated from the fraction. Pass it only
    for work whose fraction is weighed by *time* (an experiment import is). Off by default
    because most of these fractions are a count of steps or a nominal stage weight, and an ETA
    extrapolated from one of those is a confident wrong number (`core.progress.PhaseClock`).

    ``work(emit)`` runs off the UI thread and calls ``emit(stage, fraction, detail)``
    as it progresses; the overlay renders a real bar. Same delivery/teardown contract,
    ``supersedes`` included.

    ``on_cancel`` makes the work **cancellable**: the overlay offers Cancel, and the work
    stops at its next ``emit`` (which raises ``core.progress.OperationCancelled``); then
    ``on_cancel`` runs instead of ``on_error``. Closing ``target``'s window cancels too, so a
    window opened again never runs a second copy beside one nobody is watching. The work must
    report often enough to stop promptly, and leave nothing half-written when it does.
    """
    from detail.core.progress import OperationCancelled

    overlay = BusyOverlay(target)
    overlay.start(label)
    worker = StageProgressWorker(work, receiver=target)
    _active.add(worker)
    current = _claim(target, supersedes)
    if on_cancel is not None:
        overlay.offer_cancel(worker.request_cancel)
        _cancel_with_window(target, worker)

    def clear() -> None:
        _active.discard(worker)
        if alive(overlay):
            overlay.stop()
            overlay.deleteLater()

    started = time.monotonic()

    def tick(permille: int, status: str) -> None:
        if not alive(
            overlay
        ):  # a tick queued behind `clear`, which deleted the overlay
            return
        if eta:
            status = with_remaining(status, permille / 1000, time.monotonic() - started)
        overlay.set_fraction(permille / 1000, status)

    def deliver(result: object) -> None:
        clear()
        if current():
            on_done(result)

    def fail(exc: BaseException) -> None:
        clear()
        if not current():
            return
        if on_cancel is not None and isinstance(exc, OperationCancelled):
            on_cancel()
            return
        (on_error or (lambda e: _show_error(target, e)))(exc)

    worker.tick.connect(tick)
    worker.done.connect(deliver)
    worker.failed.connect(fail)
    worker.finished.connect(lambda: _active.discard(worker))  # see `run_busy`
    worker.finished.connect(worker.deleteLater)
    worker.start()
    return worker
