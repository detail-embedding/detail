"""The cyclic garbage collector runs on the UI thread, and only there.

**The class: Qt objects torn down by whichever thread happened to allocate.** Python's automatic
collector runs on the thread that crosses its allocation threshold — in this app, as often as not
a ``BackgroundCall``, a montage's picture worker or a run-progress poller. A collection *clears*
every object in a garbage cycle before freeing any: a widget's ``__dict__``, and the closures
connected to its signals (``func_clear`` drops their globals). When the C++ widget outlives that —
destroyed in the same pass, or later from the UI thread's queue — Qt still delivers its events and
emits ``destroyed`` into those cleared objects, and from another thread it does so concurrently
with the collection.

Measured on the test suite, 2026-11-10: ``Fatal Python error: Segmentation fault`` with the main
thread in ``busy.py``'s ``cancel`` at ``line ???`` (the closure ``run_progress`` connects to a
window's ``destroyed``), ``EXC_BAD_ACCESS at 0x18`` inside ``_PyEval_EvalFrameDefault`` called from
``PyQtSlotProxy::unislot`` ← ``QObject::destroyed`` ← ``QWidget::~QWidget`` ← ``DeferredDelete`` in
four macOS crash reports, and ``'LiveMontage' object has no attribute 'canvas'`` from an event
filter whose ``__dict__`` had been cleared. On a subset of the suite that crashed a worker
in half its runs, collecting on the UI thread only gave no crashes, at no measurable cost.

:func:`install` turns the automatic collector off and runs the same generational schedule from a
timer on the UI thread instead, so a collection never runs beside the UI thread's own Qt work. The
test suite does the same between tests (``tests/conftest.py``), and destroys each test's windows
before the collector can find them (``tests/gui/conftest.py``).
"""

from __future__ import annotations

import gc
import time

from PyQt6.QtCore import QObject, QTimer

#: How often the UI thread checks whether a collection is due. The automatic collector checks on
#: every allocation; half a second is far below any interval at which garbage costs memory.
_INTERVAL_MS = 500
#: The least time between two full collections.
_FULL_EVERY_S = 10.0
_last_full = float("-inf")


def collect_due() -> int:
    """Run the collection the automatic collector would have run by now, on this thread.

    The oldest generation whose count is over its threshold, as CPython schedules them; nothing
    when none is due. A full (generation 2) collection waits at least :data:`_FULL_EVERY_S`
    after the last one — the job CPython's own long-lived-objects ratio does, which is not
    readable from Python; without it a heap the size of this app's pays a full collection
    every tenth young one. Returns the number of objects collected.
    """
    global _last_full
    counts = gc.get_count()
    thresholds = gc.get_threshold()
    for generation in (2, 1, 0):
        if not thresholds[generation] or counts[generation] <= thresholds[generation]:
            continue
        if generation == 2:
            now = time.monotonic()
            if now - _last_full < _FULL_EVERY_S:
                continue
            _last_full = now
        return gc.collect(generation)
    return 0


def install(parent: QObject) -> QTimer:
    """Collect on the UI thread only: automatic collection off, :func:`collect_due` on a timer.

    ``parent`` (the application) owns the timer, so it runs for the life of the app.
    """
    gc.disable()
    timer = QTimer(parent)
    timer.setInterval(_INTERVAL_MS)
    timer.timeout.connect(collect_due)
    timer.start()
    return timer
