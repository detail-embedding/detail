"""Shared GUI test fixtures, and the autouse one that keeps the suite from inheriting a blindness.

**The class: an exception escaping a Qt slot is reported and swallowed**, so a widget still
constructs and every test passes over the part that never ran. Qt cannot propagate an exception out
of a slot across the C++ boundary, so the binding routes it to ``sys.excepthook`` and carries on. The app's
own excepthook is **right** to do that — a stray error in one handler should not kill a session
someone has work open in — but a test suite that inherits it reports green for a construction that
half-happened.

``_slot_exceptions_are_failures`` is the autouse fixture that turns those into failures for the
duration of a test. It is why the construction corpus
(``tests/gui/test_smoke_construct.py``, ``test_run_preview.py``, ``test_run_detail.py``) means
anything: without it, "the widget built" and "the widget built and every slot it fired worked" are
the same green dot.

Indexed in ``docs/recorded_classes.md`` against
``tests/gui/conftest.py::_slot_exceptions_are_failures``.
"""

import os
import sys

import pytest

# The GUI tier needs the binding. Imported here, a missing one is an error for every GUI test,
# never a skip that leaves the tier reading green (#107).
import detail.gui.qt  # noqa: F401


@pytest.fixture(autouse=True)
def _checks_run_in_process(monkeypatch):
    """Stub ``api_checks.run_checks`` with canned results for every GUI test.

    The readiness panel and the SAM 3 backend page recompute via ``run_checks`` off the UI
    thread (a real subprocess in production). GUI tests must neither spawn that subprocess nor
    run the *real* checks in-process on a background ``QThreadPool`` thread: the real checks
    import ``torch`` / ``huggingface_hub``, and doing that concurrently with the model tests
    (which mock ``huggingface_hub`` / ``requests``) races the import machinery — it surfaces
    elsewhere as ``module 'requests' has no attribute 'exceptions'``. Canned results keep
    construction fast, import-free, and deterministic; tests that need specific signals pass
    them to ``_render`` / ``signals_from`` directly. The real check bodies and the subprocess
    path are covered in tests/checks and tests/api/test_readiness."""
    from detail.api import checks as api_checks
    from detail.core.dataclasses import CheckResult, CheckStatus

    def _canned(**kwargs):
        return [
            CheckResult(info.check_id, CheckStatus.WARN, "")
            for info in api_checks.list_checks()
        ]

    monkeypatch.setattr(api_checks, "run_checks", _canned)


def drain_until(
    qapp, predicate, timeout_ms: int = 10_000, *, raise_on_timeout: bool = True
) -> bool:
    """Spin the event loop until ``predicate`` holds; fail, naming it, if it never does.

    Bounded by wall clock, not by a spin count: ``processEvents`` returns immediately when
    nothing is queued, so N bare pumps can burn through in microseconds while the worker is
    still running — green on an idle box, flaky inside a loaded suite.

    **It raises rather than answers**, because an answer can be dropped. This used to return
    False on a timeout, and 12 of its callers called it as a statement: on a loaded machine the
    wait ran out, the test carried on against a state it never reached, and it failed later on a
    disabled button or an empty table, naming none of this. The class is written up in
    ``tests/gui/test_waits_are_checked.py``, which also scans for the forms that can still drop
    an answer. Pass ``raise_on_timeout=False`` only to ask *whether* something
    happens — ``assert not drain_until(..., raise_on_timeout=False)`` — and use the answer. To
    let the loop run for a fixed time, call ``pump_for``.

    Here rather than private to one module because a *result* that arrives on a thread is not
    a rare shape any more: five files had grown their own copy, and the sixth would have been
    written next to a rail that fills asynchronously.

    Each pump also delivers deferred deletes, which ``processEvents`` leaves queued: a widget a
    rebuild ``deleteLater``-ed is otherwise still a findable child when the predicate is
    checked, and a lookup by text found an old disabled button beside its replacement.
    """
    from detail.gui.qt import QEvent, qwait

    waited = 0
    while waited < timeout_ms:
        qapp.processEvents()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if predicate():
            return True
        qwait(10)
        waited += 10
    if predicate():
        return True
    if raise_on_timeout:
        raise AssertionError(
            f"waited {timeout_ms / 1000:g}s and this never held: {_describe(predicate)}"
        )
    return False


def _describe(predicate) -> str:
    """The predicate's source where it has one (a lambda's line), else its name."""
    import inspect

    try:
        source = inspect.getsource(predicate)
    except (OSError, TypeError):
        return getattr(predicate, "__qualname__", repr(predicate))
    return " ".join(source.split())


def pump_for(qapp, ms: int) -> None:
    """Run the event loop for ``ms``, delivering deferred deletes — waiting on nothing.

    For "let whatever is queued land" and "nothing more arrives in this window", which used
    to be spelled ``drain_until(qapp, lambda: False, ms)``: a wait for something that never
    happens, whose timeout was the point.
    """
    from detail.gui.qt import QEvent, qwait

    waited = 0
    while waited < ms:
        qapp.processEvents()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qwait(10)
        waited += 10
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def export_source(clips=None):
    """A ``montage_export.ExportSource`` for a montage built on its own: ``clips()`` gives the
    grid's export clips (none by default)."""
    from pathlib import Path

    from detail.gui.montage_export import ExportSource

    return ExportSource(
        clips=clips or (lambda: []),
        experiment_dir=lambda: Path("."),
        origin=lambda: None,
        name=lambda: "test",
        empty=lambda: "Nothing here to export.",
    )


def drain_busy_calls(qapp, timeout_ms: int = 5_000) -> int:
    """Deliver every pending ``run_busy`` result, and report how many were outstanding.

    Exposed so the guard can call it directly; ``_no_leaked_busy_calls`` is what makes every
    GUI test use it.
    """
    from detail.gui.qt import qwait

    from detail.gui import busy
    from detail.gui import workers

    def pending() -> int:
        # The **union** of both registers. `workers._IN_FLIGHT` exists to keep a directly
        # constructed `BackgroundCall` from being *collected* mid-flight (#263); this is the
        # other half of the same fact — such a call is also not *drained*, so its result lands in
        # whichever later test next pumps the loop, and several of these callbacks are modal.
        # A union rather than a sum, because `run_busy` starts a `BackgroundCall` too and its
        # worker is in both.
        return len(busy._active | workers._IN_FLIGHT)

    outstanding = pending()
    waited = 0
    while pending() and waited < timeout_ms:
        qapp.processEvents()
        qwait(10)
        waited += 10
    qapp.processEvents()
    return outstanding


def drain_thread_pool(qapp, timeout_ms: int = 5_000) -> int:
    """Let every queued Qt pool task finish and deliver, and report how many were outstanding.

    The other half of the same class ``_no_leaked_busy_calls`` covers. That fixture drains
    ``run_busy``'s workers, which are the ones it knows about; thumbnail decoding goes through
    ``QThreadPool`` instead (``clip_grid``, ``cluster_strip``, ``scrubber``) and nothing waited
    for it. Measured before this existed: **85 of 1416 GUI tests** ended with a pool task still
    running, and the test it would land on was always an unrelated one.

    **Bounded, and pumping.** A hard ``waitForDone()`` would turn a task that never finishes into
    a hung session — the failure this is meant to prevent — so it gives up after ``timeout_ms``
    like the busy drain does. Pumping matters as much as waiting: a task's result is delivered by
    a queued signal, so without ``processEvents`` the work would finish and the *delivery* would
    still cross into the next test.
    """
    from detail.gui.qt import QThreadPool, qwait

    pool = QThreadPool.globalInstance()

    def busy_count() -> int:
        # Active threads only: the pool reports no count of queued tasks. A queued-but-unstarted
        # task is covered anyway: the pool starts it while this loop pumps, and the loop only
        # exits once nothing is active.
        return pool.activeThreadCount()

    outstanding = busy_count()
    waited = 0
    while busy_count() and waited < timeout_ms:
        qapp.processEvents()
        qwait(10)
        waited += 10
    qapp.processEvents()
    return outstanding


# Declared **before** `_no_leaked_busy_calls` on purpose. Autouse teardown is LIFO, so a
# fixture declared later finishes first — and this one's assertion has to run *after* the busy
# drain, because the drain is exactly where a leaked `run_busy` callback fires. Declared after
# it, the excepthook was already restored by then and an exception raised in that window went
# to the real hook and was swallowed: the precise class this fixture exists to catch.
@pytest.fixture(autouse=True)
def _slot_exceptions_are_failures(request, monkeypatch):
    """An exception escaping a Qt slot fails the test that caused it.

    The class: **a slot that raises is reported and swallowed, so the suite cannot see it.**
    The binding routes an exception escaping a slot to ``sys.excepthook`` and then carries on, and
    the app installs a hook that reports rather than dying (``test_fatal_slots.py``) — which is
    right for a user mid-session and blinding for a test. The widget still constructs, the
    assertions still pass, and the only evidence is a traceback on stderr that a green run
    buries.

    It shipped exactly that way: ``AuditTab`` set its default scope before building the list
    that scope selects, `toggled` fired into a slot that reached for a widget which did not
    exist yet, and 44 tests plus the construction smoke passed over a sidebar that was never
    filled. A user found it by opening the tab.

    Opt out with ``@pytest.mark.allow_slot_exceptions`` where the raising *is* the subject.
    """
    if request.node.get_closest_marker("allow_slot_exceptions"):
        yield
        return
    caught: list = []
    original = sys.excepthook
    monkeypatch.setattr(sys, "excepthook", lambda *info: caught.append(info))
    yield
    sys.excepthook = original
    if caught:
        exc_type, exc, tb = caught[0]
        raise AssertionError(
            f"{len(caught)} exception(s) escaped a Qt slot and were swallowed; the first was "
            f"{exc_type.__name__}: {exc}"
        ).with_traceback(tb)


def destroy_new_windows(qapp, before: set[int]) -> int:
    """Delete every top-level widget that is not in ``before``, here, and report how many.

    Exposed so the guard can call it directly; ``_windows_die_with_their_test`` is what makes
    every GUI test use it.
    """
    from detail.gui.qt import QEvent, address, created_by_python

    # Only windows Python made — ones a test built. A top-level Qt made for something else (a
    # completer's popup) is that owner's to delete, and deleting it here is a double free when
    # the owner goes: measured, a segfault in `~QCompleter`.
    #
    # *Made*, not *owned*: PySide6 hands a widget to C++ when some reads return it —
    # `QSplitterHandle.splitter()` does — so every `GripSplitter` a test showed stopped counting
    # as Python's, outlived its test, and broke later tests from the collector (#107).
    doomed = [
        w
        for w in qapp.topLevelWidgets()
        if created_by_python(w) and address(w) not in before
    ]
    for widget in doomed:
        widget.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return len(doomed)


# Declared after `_slot_exceptions_are_failures` and before `_no_leaked_busy_calls`, so its teardown
# runs after the busy drain (a result still in flight is delivered to a live window) and before the
# slot check (an exception a window raises while being destroyed fails this test).
@pytest.fixture(autouse=True)
def _windows_die_with_their_test(qapp):
    """Every window a test opens is destroyed at its end, by Qt, while its Python side is whole.

    **The class: a widget left for the cyclic collector.** A test's window is held by a cycle (its
    closures reach back to it), so dropping it leaves it to Python's cyclic collector, which runs
    whenever an allocation crosses a threshold — mid-way through a later test, on whichever thread
    allocated. The collector *clears* every object in the cycle before any is freed: the window's
    ``__dict__`` and the closures connected to its signals. When freeing one of them then destroys
    the C++ window, Qt still delivers events to its filters and emits ``destroyed`` to its slots —
    into Python objects that no longer have their attributes, and functions that no longer have
    their globals. Measured 2026-11-10: ``Fatal Python error: Segmentation fault`` in
    ``busy.py``'s ``cancel`` at ``line ???`` (the closure ``run_progress`` connects to a window's
    ``destroyed``), ``EXC_BAD_ACCESS at 0x18`` under the earlier binding's ``PyQtSlotProxy::unislot`` ←
    ``QObject::destroyed`` in four macOS crash reports, and ``'LiveMontage' object has no attribute
    'canvas'`` / ``'ClipPaneWindow' object has no attribute '_body_widget'`` raised from an event
    filter — each in a test that never touched that window. The segfault killed an xdist worker,
    and three full runs then hung (``scripts/suite_watchdog.py`` has that half).

    Destroying the windows here — ``deleteLater`` and the deferred delete delivered at once — runs
    their teardown while every closure and attribute still exists, on the UI thread, inside the
    test that made them. What the collector finds afterwards is Python objects whose C++ side is
    already gone, which it can clear without anything calling back. Windows that existed before
    the test (a module- or session-scoped fixture's) are left alone.
    """
    from detail.gui.qt import address

    before = {address(w) for w in qapp.topLevelWidgets()}
    yield
    destroy_new_windows(qapp, before)


# --- which tests raced their own off-thread work (#533) ------------------------------------ #
# Opt-in via `DETAIL_LEAK_LOG=<path>`; a normal run is unchanged. Records two different things the
# drains below already compute and discard:
#
#   before  how much off-thread work was in flight when teardown began. **Not a leak** — the drain
#           then handles it. It is the list of tests whose assertions ran while the screen was
#           still being built, which is the class #554 and #563 came out of: such a test does not
#           fail, it passes *vacuously*, against labels that do not exist yet. Measured at ~300 of
#           ~1,609 gui tests (~18%).
#   after   how much is still outstanding once the bounded drain has given up. That is a real leak,
#           crossing into the next test. Measured at 1 of ~1,609, now 0 — the drains work.
#
# Both numbers have to be read here rather than from the drains' return values, which are the
# *before* snapshot by construction.
#
# **There is deliberately no gate on this**, and no port of napari's `_dangling_*` fixtures: with
# the leak rate at ~0 there is nothing for a gate to catch, and `drain_busy_calls` /
# `drain_thread_pool` already are the mechanism those fixtures would duplicate. What this is for is
# generating the *before* list, which a source scan cannot — "every `.text()` assertion with no
# preceding wait" is 202 hits across 40 files, almost all synchronously-set labels.
#
# An earlier version of this also tried to count raw QThreads outliving their test. Three sensors
# gave 96, 28 and 134 of 1,609; measuring Qt thread liveness from the UI thread of the process under
# test perturbs what it measures, so that half was removed rather than shipped. #533 has the detail.
def _outstanding_now() -> tuple[int, int]:
    from detail.gui.qt import QThreadPool

    from detail.gui import busy, workers

    return (
        len(busy._active | workers._IN_FLIGHT),
        QThreadPool.globalInstance().activeThreadCount(),
    )


def _record_outstanding(nodeid, before, after) -> None:
    import os

    path = os.environ.get("DETAIL_LEAK_LOG")
    if not path:
        return
    busy_after, pool_after = after
    if not (before[0] or before[1] or busy_after or pool_after):
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            f"{nodeid}\tbusy_before={before[0]}\tpool_before={before[1]}"
            f"\tbusy_after={busy_after}\tpool_after={pool_after}\n"
        )


@pytest.fixture(autouse=True)
def _no_leaked_busy_calls(request, qapp, monkeypatch):
    """A background call started in one test must not deliver into the next one.

    The class: **``run_busy`` hands its result to a callback on the UI thread, so a test that
    starts one and returns before the delivery lands leaves that callback queued for whichever
    later test next pumps the event loop.** Several of those callbacks are modal — the metadata
    export ends in ``QMessageBox.information`` — and a modal box in a headless run has nobody to
    dismiss it, so the whole session stops. Not a failure: a stop. On CI that read as a queue
    and burned 1h43m before anyone cancelled it.

    It is invisible to the test that causes it, which passes, and it lands on a test that did
    nothing wrong — ``test_recording_detail_button_rows_share_a_left_edge`` hung because it
    calls ``processEvents()``, which is not a mistake. And it is timing-dependent: the leaking
    test waits for its *output file*, which appears before the ``done`` signal is delivered, so
    on a fast machine the delivery slips in anyway and on a slow one it does not.

    Two halves, because either alone leaves a hole:

    - **Modal dialogs never block.** A GUI test can never *want* a real modal box; it would
      hang. A test that cares what a dialog returns patches it itself, and its ``monkeypatch``
      applies after this one, so it wins.
    - **Pending calls are drained at teardown**, so nothing crosses a test boundary at all.

    The stubs are still in place during that drain: this fixture asks for ``monkeypatch``, so
    it is torn down before ``monkeypatch`` undoes anything.
    """
    from detail.gui.qt import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(QMessageBox, "about", staticmethod(lambda *a, **k: None))
    # `question` answers **No**: it guards destructive actions, so the safe default for a test
    # that did not mean to trigger one is to decline.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    yield
    busy_before = drain_busy_calls(qapp)
    # The pool half of the same class: thumbnail decoding runs on QThreadPool, which the busy
    # drain above knows nothing about. Drained after it, because a busy callback can start one.
    pool_before = drain_thread_pool(qapp)
    _record_outstanding(
        request.node.nodeid, (busy_before, pool_before), _outstanding_now()
    )


@pytest.fixture
def real_machine():
    """Opt out of ``capable_machine`` — this test states the machine itself.

    Request it from any test about *availability*: which models are installed, which
    backends exist, what a picker greys out.
    """
    return True


@pytest.fixture(autouse=True)
def capable_machine(monkeypatch, request):
    """Every GUI test runs as if all weights are downloaded and all backends present.

    The model pickers grey out what cannot run on this machine (`gui/model_availability`),
    so without this a form's contents would depend on the box running the suite — the dev
    machine has no CUDA and no downloaded weights, and CI has neither either. Pinning them
    keeps every *other* GUI test asking its own question instead of the host's.

    A test that requests ``real_machine`` is skipped here and patches what it means; a test
    that patches ``cuda_available`` itself also wins, since its ``monkeypatch`` applies
    after this one.
    """
    if "real_machine" in request.fixturenames:
        return
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: True)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)


@pytest.fixture(scope="session")
def qapp(tmp_path_factory):
    """A headless QApplication for widget tests; a missing binding is an error, not a skip."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from detail.gui.qt import QApplication, QCoreApplication, QSettings, QStandardPaths

    # Isolate QSettings (the guided-layer dismissal store) so tests never read or clobber the
    # developer's real settings.
    QStandardPaths.setTestModeEnabled(True)
    QCoreApplication.setOrganizationName("detail-test")
    QCoreApplication.setApplicationName("detail-test")
    # One store per test process, in its own basetemp. `_forget_view_preferences` clears the
    # store around every test, so any store two processes share lets one wipe another's view
    # state mid-test. Naming the store per xdist worker stopped that inside one run, but not
    # between runs: every concurrent run on the machine — other worktrees, other sessions —
    # has a `gw0`, and test mode puts them all in one directory (macOS's native store ignores
    # test mode altogether). Every `QSettings()` here is the default constructor, which
    # follows the default format, so these two calls move all of them.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path_factory.mktemp("qsettings")),
    )
    app = QApplication.instance() or QApplication([])
    # The app's own style, so a test renders what a user sees. Left at the platform default, a
    # layout test measured macOS's native widgets, and whichever test first called
    # `configure_application` switched the rest of its worker to Fusion — results that depended
    # on test order.
    from detail.gui.style import apply_style

    apply_style(app)
    # The app collects garbage on the UI thread only (`detail.gui.ui_thread_gc`); a test process
    # left on Python's automatic collector segfaulted in GUI teardown when a background thread
    # collected a widget the UI thread was still destroying.
    from detail.gui import ui_thread_gc

    ui_thread_gc.install(app)
    return app


#: Fusion's palette under a dark colour scheme, measured on Windows 11 — the roles a chart reads.
_DARK_ROLES = {
    "Window": "#1e1e1e",
    "Base": "#2d2d2d",
    "AlternateBase": "#002823",
    "WindowText": "#ffffff",
    "Text": "#ffffff",
    "Button": "#3c3c3c",
    "ButtonText": "#ffffff",
}


@pytest.fixture(params=["light", "dark"])
def app_theme(request, qapp):
    """The application drawn in each chart theme; yields that ``ChartTheme``.

    Through the **palette**, because that is the one lever the offscreen test platform honours:
    it ignores ``styleHints().setColorScheme`` and reports the scheme as unknown, and
    ``plot_theme.current`` falls back to the palette's lightness. The yielded theme is asserted,
    so a "dark" run that silently drew light fails here rather than passing as a light test.
    """
    from detail.gui.qt import QColor, QPalette

    from detail.gui import plot_theme

    original = QPalette(qapp.palette())
    if request.param == "dark":
        dark = QPalette(original)
        for role, colour in _DARK_ROLES.items():
            dark.setColor(getattr(QPalette.ColorRole, role), QColor(colour))
        qapp.setPalette(dark)
    theme = plot_theme.current()
    assert theme.name == request.param, "the test platform did not take the theme"
    yield theme
    qapp.setPalette(original)


@pytest.fixture
def stub_yolo_model(monkeypatch):
    """Register a fake open-vocab detection model with size variants and batching.

    Stands in for the removed YOLOE as the multi-variant, batch-capable *pretrained*
    detector the picker tests exercise — SAM3 (the only real built-in left) has
    neither a size axis nor batching. Returns the model id. Scoped to the test via
    ``monkeypatch`` so it never leaks into the real ``MODELS`` catalog.
    """
    from detail.core.dataclasses import FrameResult, TextCue
    from detail.core.protocols import DetectionModel
    from detail.core.registries import MODELS

    class _FakeYOLO(DetectionModel):
        model_id = "fake_yolo"
        architecture = "yolo"
        VARIANTS = {"11s": "a.pt", "11m": "b.pt", "11l": "c.pt"}
        DEFAULT_VARIANT = "11s"
        supported_cue_types = [TextCue]
        supported_backends = ["cpu", "mps", "cuda"]
        supports_batching = True

        def load(self, device):
            pass

        def predict(self, frame, prompt):
            return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=[])

        def predict_batch(self, frames, prompts):
            return [self.predict(f, p) for f, p in zip(frames, prompts)]

    monkeypatch.setitem(MODELS, "fake_yolo", _FakeYOLO)
    return "fake_yolo"


def release_held_input(qapp) -> None:
    """Put Qt's global keyboard-modifier and mouse-button state back to nothing held.

    Exposed so the guard can call it directly; ``_no_input_held_from_an_earlier_test`` is what
    makes every GUI test start from it.
    """
    from detail.gui.qt import QEvent, QTest, QWidget, Qt

    # A release with no modifiers is the one simulated event that clears both halves (a key
    # release clears modifiers only, a move neither). The target is never shown and nothing
    # listens to it, so the event itself lands nowhere.
    target = QWidget()
    QTest.mouseRelease(target, Qt.MouseButton.LeftButton)
    target.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def _no_input_held_from_an_earlier_test(qapp):
    """Every GUI test starts with no modifier key and no mouse button held.

    **The class: Qt's global input state outlives the test that set it.** A key event carrying
    a modifier (``QTest.keyClick(w, key, ShiftModifier)``, or a ``Key_Shift`` press sent with
    ``sendEvent``) and a ``QTest`` mouse press leave ``QGuiApplication.keyboardModifiers()`` /
    ``mouseButtons()`` reading them until the next input event. A call that has no event of its
    own reads that state: ``QTableView.selectRow`` under the default extended selection takes a
    held Shift to mean *extend from the anchor*, and selects nothing.

    Measured over the GUI tier, 18 tests in 8 files end with something held, every one of them
    on purpose (holding Shift is what ``test_live_montage`` and ``test_label_clean`` test). The
    test that paid was whichever file ``--dist loadfile`` ran next on that worker: when it was
    ``test_keypoints_manager.py``, five of its twelve tests failed on disabled Edit/Export
    buttons and "Select a skeleton first." All five passed alone, and the failures were put down
    to machine load. They fail on demand with Shift held before each test, and pass with this
    fixture. Guard: ``tests/gui/test_input_state.py``.
    """
    release_held_input(qapp)
    yield


@pytest.fixture(autouse=True)
def _forget_view_preferences():
    """Start every GUI test from the documented defaults, not from what the last one left.

    ``QSettings`` here is isolated from the developer's real store (see ``qapp``) but not from
    the *rest of the session*: it is one file, shared by every test in the process. So a test
    that folds a result band decides what the next test's workspace opens with, and the failure
    lands on whichever test happens to run next rather than on the one that caused it — green
    file by file, red in the full run, which is the shape CLAUDE.md calls a cross-test leak.

    Scoped to the keys that are *view state* a test might set (``bands/`` folds, ``query/``
    thumbnail size, ``panes/`` every split's widths and open/shut panels (and ``splitters/``, the
    keys they migrate from), ``interpret/`` the tree's height and the montage's masks,
    ``label_check/`` AUDIT's goal width, ``clip_window/`` the clip window's *Advanced* and
    *Blind* ticks). Each test process has its own store (see ``qapp``). Window geometry is left alone: ``test_window_state.py`` owns those keys and
    clears them itself, and clearing them here would silently disarm it.
    """
    from detail.gui.qt import QSettings

    def clear():
        settings = QSettings()
        for group in (
            "bands",
            "query",
            "panes",
            "splitters",
            "interpret",
            "label_check",
            "clip_window",
        ):
            settings.beginGroup(group)
            settings.remove("")
            settings.endGroup()

    clear()
    yield
    clear()


@pytest.fixture
def fitted(fitted_temporal):
    """The cluster-workspace tests' name for the root conftest's temporal fit, as
    ``(experiment, run_id)``.

    Here, not in each of the files that use it: they used to import it from
    ``test_cluster_workspace.py``, and an imported fixture is a separate fixture in every module
    that imports it, so its session-scoped fit was built once per file.
    """
    return fitted_temporal
