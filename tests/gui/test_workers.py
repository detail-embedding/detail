"""The GUI's progress→render mappings, its background workers, and one lifetime class.

Two subjects. The larger is the **one tested `event → render` mapping per surface** that
``DESIGN_PLAN.md`` §6 requires — ``fit_progress_fraction`` and ``processing_progress_fraction``
(+ ``_with_eta``) — including the aggregate fold, where a multi-recording bar folds the inner
fraction rather than stepping at the recording boundary. The rest is the workers themselves and
``BackgroundCall``.

(The previous docstring said "logic; no QApplication needed". Much of this file takes ``qapp``.)

**The class: a Qt object's C++ side outliving the Python state it needs.** Three instances in one
day — a closure the cyclic collector *cleared* while its connection survived; a widget wrapper Qt
re-materialized after the original was collected; and queued deliveries crossing into a later test.

The signature of the first is worth knowing by heart, because it names the wrong culprit:

    TypeError: () missing 1 required positional argument: 'x'

— an **empty function name and no traceback**. ``func_clear`` blanks ``__qualname__`` and drops
``__defaults__``, and the call fails at the C++ boundary before a Python frame exists, so it is
reported against whichever test next pumps the event loop.

So anything Qt holds by C++ pointer — a connection, an installed event filter, a queued call — must
be outlived by the Python object behind it, or scoped to the window where something holds it. The
ordinary ``lambda result, g=generation:`` idiom is exactly the shape that breaks, which is why every
worker here names a ``receiver=``. Covered by
``test_a_worker_with_a_delivery_in_flight_is_not_collected``,
``test_fit_progress_worker_outlives_the_page_that_started_it`` and
``test_pool_work_does_not_cross_a_test_boundary``; ``tests/gui/test_busy.py`` holds the event-filter
half, and ``tests/gui/test_worker_lifetime.py`` the mirror class — Python state outliving the Qt
object it was going to touch.
"""

import threading
import weakref
from dataclasses import dataclass

import pytest

pytest.importorskip("PyQt6")

from detail.gui.workers import (  # noqa: E402
    BackgroundCall,
    fit_progress_fraction,
    import_progress_fraction,
    processing_progress_fraction,
)


def test_fit_progress_fraction_training_event():
    from detail.core.dataclasses import TrainingProgressEvent

    fraction, status = fit_progress_fraction(
        TrainingProgressEvent(3, 100, {"box_map50": 0.85}, 5.0, 9.0)
    )
    assert fraction == pytest.approx(0.03)
    assert "epoch 3/100" in status
    assert "box_map50=0.850" in status


def test_fit_progress_fraction_training_message():
    # #18: a phase message (e.g. the pre-train "loading base model") is shown.
    from detail.core.dataclasses import TrainingProgressEvent

    _, status = fit_progress_fraction(
        TrainingProgressEvent(0, 100, {}, 0.0, None, message="loading base model")
    )
    assert "loading base model" in status


def test_fit_progress_fraction_dataset_export_event():
    from detail.core.dataclasses import LabelingProgressEvent

    fraction, status = fit_progress_fraction(
        LabelingProgressEvent(50, 200, 1, 1, 5.0, None)
    )
    assert fraction == pytest.approx(0.25)
    assert "frame 50/200" in status


def test_fit_progress_fraction_dataset_export_message():
    # #18: labeling's "loading model" phase text reaches the GUI dataset-export bar.
    from detail.core.dataclasses import LabelingProgressEvent

    _, status = fit_progress_fraction(
        LabelingProgressEvent(0, 200, 0, 1, 0.0, None, message="loading model")
    )
    assert "loading model" in status


def test_an_export_run_counts_targets_not_recordings():
    """An export's units are its targets, and the event says so (`unit_label`). The CLI read it
    and the GUI did not, so the page a bundle export now opens on read "recording 1/2" over the
    bundle and its metadata CSV (#916). One whole string, not a containment check: "target 1/2"
    is also inside "subtarget 1/2"."""
    from detail.pipeline.export_run import ExportProgressEvent

    _, status = fit_progress_fraction(
        ExportProgressEvent(0, 2, 500, 1000, 3.0, None, message="hashing · 2 / 4 files")
    )
    assert status == "target 1/2 — hashing · 2 / 4 files"


def test_an_export_run_shows_its_own_eta_or_none():
    """The export's fraction gives every target an equal share, so extrapolating one from it
    reads the whole run's guess: with a bundle and its CSV, the bundle's remaining three times
    over (#916). The runner's own ETA is shown; when it has none (a target just starting), none
    is invented."""
    from detail.pipeline.export_run import ExportProgressEvent

    def status(eta_s):
        return fit_progress_fraction(
            ExportProgressEvent(1, 2, 0, 1000, 60.0, eta_s, message="writing")
        )[1]

    assert (
        status(None) == "target 2/2 — writing"
    ), "an ETA was made up from the fraction"
    assert status(4.0) != status(None), "the runner's own ETA is not shown"


def test_fit_progress_fraction_analysis_within_recording():
    # The analysis bar must advance *within* a long single recording, not jump 0->100:
    # recording 0 of 2, halfway through its rows → overall (0 + 0.5) / 2 = 0.25.
    from detail.pipeline.location_analysis import LocationProgressEvent

    event = LocationProgressEvent(
        recordings_done=0,
        recordings_total=2,
        frames_done=5,
        frames_total=10,
        elapsed_s=1.0,
        eta_s=None,
        message="s1 (1/2)",
    )
    fraction, status = fit_progress_fraction(event)
    assert fraction == pytest.approx(0.25)
    # recordings count AND the phase message are both shown, plus the ETA suffix.
    assert "recording 1/2" in status and "s1 (1/2)" in status
    assert "3s" in status and "left" in status  # 25% in 1s → ~3s


def test_fit_progress_fraction_recordings_only_event():
    # An analysis event that reports only whole recordings (frames_total 0) must map like any
    # §16 analysis event (recordings_*), not fall through and raise on the empty frame pair.
    # It was written on clip query's event, removed with that module; the location event has
    # the same shape.
    from detail.pipeline.location_analysis import LocationProgressEvent

    event = LocationProgressEvent(
        recordings_done=1,
        recordings_total=2,
        frames_done=0,
        frames_total=0,
        elapsed_s=1.0,
        eta_s=None,
        message="searched 1/2 · 7 matches",
    )
    fraction, status = fit_progress_fraction(event)
    assert fraction == pytest.approx(0.5)
    # recordings count AND the message both shown (the count is no longer hidden).
    assert "recording 2/2" in status and "searched 1/2 · 7 matches" in status
    assert "left" in status


def test_fit_progress_fraction_folds_within_recording():
    # An analysis that reports within-recording progress has the bar fold
    # frames_done/frames_total inside a recording: recording 0 of 2, 3/6 done →
    # overall (0 + 0.5) / 2 = 0.25. (Revert the fold and this is 0.) Written on syllable
    # discovery's event, removed with that module; the location module's event has the shape.
    from detail.pipeline.location_analysis import LocationProgressEvent

    event = LocationProgressEvent(
        recordings_done=0,
        recordings_total=2,
        frames_done=3,
        frames_total=6,
        elapsed_s=1.0,
        eta_s=None,
        message="4 syllables so far",
    )
    fraction, status = fit_progress_fraction(event)
    assert fraction == pytest.approx(0.25)
    assert "recording 1/2" in status and "4 syllables so far" in status
    assert "left" in status


@dataclass
class _WholeRunFramesEvent:
    """An event whose ``frames_*`` is already the whole-run fraction (``overall_fraction_is_frames``).

    The removed clusterer fit was the one producer (``docs/removed_capabilities.md``); the
    renderer branch is generic, so it keeps its tests on this stand-in rather than going dark.
    """

    recordings_done: int
    recordings_total: int
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None
    overall_fraction_is_frames: bool = True


def test_fit_progress_fraction_does_not_fold_a_whole_run_fraction():
    # frames_* is ALREADY the whole-run fraction (permille), unlike a within-recording count.
    # Folding it into recordings_done double-counts: a phase can pin recordings_done at the total
    # while the fraction climbs 0.3 -> 0.9, so the bar would read (2 + 0.6) / 2 = 130%. Drop the
    # overall_fraction_is_frames branch and this fails.
    event = _WholeRunFramesEvent(
        recordings_done=2,
        recordings_total=2,
        frames_done=600,
        frames_total=1000,
        elapsed_s=1.0,
        eta_s=None,
        message="comparing 300 bouts",
    )
    fraction, status = fit_progress_fraction(event)
    assert fraction == pytest.approx(0.6)
    assert "2 recording(s)" in status and "comparing 300 bouts" in status
    assert "left" in status


def test_fit_progress_fraction_of_a_whole_run_fraction_never_exceeds_one():
    # The end of the run: every phase reported, fraction pinned at 1.0 (not 1.5).
    fraction, _ = fit_progress_fraction(
        _WholeRunFramesEvent(2, 2, 1000, 1000, 5.0, 0.0, message="done")
    )
    assert fraction == pytest.approx(1.0)


def test_fit_progress_fraction_zero_total_is_safe():
    from detail.core.dataclasses import TrainingProgressEvent

    fraction, _ = fit_progress_fraction(TrainingProgressEvent(0, 0, {}, 0.0, None))
    assert fraction == 0.0


def test_fit_progress_fraction_derives_eta_when_none():
    # eta_s=None on a dataset-export event → the render layer derives a whole-op
    # ETA from fraction + elapsed: 25% done after 30s → ~90s (1m30s) remaining.
    from detail.core.dataclasses import LabelingProgressEvent

    _, status = fit_progress_fraction(LabelingProgressEvent(25, 100, 1, 1, 30.0, None))
    assert "1m30s" in status and "left" in status


def test_fit_progress_fraction_no_eta_before_progress():
    # Nothing done yet → no ETA suffix (can't estimate from 0%).
    from detail.core.dataclasses import LabelingProgressEvent

    _, status = fit_progress_fraction(LabelingProgressEvent(0, 100, 1, 1, 5.0, None))
    assert "left" not in status


def test_fit_progress_fraction_never_derives_a_distill_eta_from_the_weighted_bar():
    """A distill event's fraction is stage-*weighted*; its elapsed_s is wall clock.

    Pairing them reconstructs the ~5x under-report downstream of the fix in the fold: at
    the labeling→training boundary the bar reads 0.45 with ~370s elapsed, which the generic
    fallback would render as "~7m left" on a run with ~104m to go. Its runner owns this
    number now — no eta_s means no ETA.
    """
    from detail.core.dataclasses import DistillProgress

    _, status = fit_progress_fraction(
        DistillProgress(
            phase="training",
            stage_index=1,
            stage_count=2,
            inner_fraction=0.0,
            overall_fraction=0.45,
            elapsed_s=370.0,
            eta_s=None,
            message="starting training",
        )
    )
    assert "left" not in status  # not "7m32s left"

    # A runner-supplied eta_s is still rendered.
    _, status = fit_progress_fraction(
        DistillProgress(
            phase="training",
            stage_index=1,
            stage_count=2,
            inner_fraction=0.14,
            overall_fraction=0.527,
            elapsed_s=1238.0,
            eta_s=5332.0,
            message="epoch 14/100",
        )
    )
    assert "1h28m" in status and "left" in status


def test_processing_progress_fraction_folds_inner_within_recording():
    # A multi-recording processing run's bar must advance *within* a recording (from the
    # active recording's inner frame progress), not only when a whole recording completes:
    # recording 0 of 2, halfway through its frames → (0 + 0.5) / 2 = 0.25.
    from detail.core.dataclasses import (
        DetectionProgressEvent,
        ProcessingProgress,
    )

    inner = DetectionProgressEvent(
        parts_done=0,
        parts_total=1,
        frames_done=5,
        frames_total=10,
        elapsed_s=1.0,
        eta_s=None,
        message="detecting",
    )
    event = ProcessingProgress(
        recordings_done=0, recordings_total=2, current_recording="s1", inner=inner
    )
    fraction, status = processing_progress_fraction(event)
    assert fraction == pytest.approx(0.25)
    assert "s1" in status and "1/2" in status and "detecting" in status


def _import_event(done, total, in_flight, *, failed=0, eta_s=None):
    from detail.api.imports import ImportProgress

    return ImportProgress(
        recordings_done=done,
        recordings_total=total,
        in_flight=tuple(in_flight),
        failed=failed,
        elapsed_s=10.0,
        eta_s=eta_s,
    )


def test_import_progress_fraction_folds_the_recording_in_flight():
    """The import queue's bar moves *within* a recording, and the line says which one of how
    many, what it is doing, and how long is left (§6 rendering invariants)."""
    event = _import_event(2, 4, [("rec_03", 0.5, "Copying — 1.2 / 2.9 GB")], eta_s=95)
    fraction, status = import_progress_fraction(event)
    assert fraction == pytest.approx((2 + 0.5) / 4)
    assert status == ("Importing rec_03 (3/4) — Copying — 1.2 / 2.9 GB · ~1m35s left")


def test_import_progress_fraction_counts_every_recording_in_flight():
    """Transcode parallelism runs several at once: each one's own fraction moves the bar, and
    the line names the one that started first rather than whichever ticked last."""
    event = _import_event(
        1,
        5,
        [("first", 0.5, "Copying"), ("second", 0.25, "Reading timestamps")],
        failed=1,
    )
    fraction, status = import_progress_fraction(event)
    assert fraction == pytest.approx((1 + 0.5 + 0.25) / 5)
    assert status == (
        "Importing 2 recordings at once (1/5 done) — first: Copying · 1 failed"
    )


def test_import_progress_fraction_between_recordings():
    fraction, status = import_progress_fraction(_import_event(3, 4, []))
    assert fraction == pytest.approx(0.75)
    assert status == "Importing recordings (3/4 done)"


def test_processing_progress_fraction_without_inner():
    # Before the first inner event arrives, the bar still reflects completed recordings.
    from detail.core.dataclasses import ProcessingProgress

    fraction, _ = processing_progress_fraction(
        ProcessingProgress(
            recordings_done=1, recordings_total=2, current_recording="s2", inner=None
        )
    )
    assert fraction == pytest.approx(0.5)


def test_processing_progress_fraction_zero_total_is_safe():
    from detail.core.dataclasses import ProcessingProgress

    fraction, _ = processing_progress_fraction(
        ProcessingProgress(
            recordings_done=0, recordings_total=0, current_recording="", inner=None
        )
    )
    assert fraction == 0.0


def test_processing_progress_fraction_appends_whole_run_eta():
    # A WHOLE-RUN ETA from the run-level elapsed the coordinator publishes + the
    # aggregate fraction: recording 0/2 half-done → overall 25% in 30s → ~90s (1m30s).
    from detail.core.dataclasses import DetectionProgressEvent, ProcessingProgress

    inner = DetectionProgressEvent(
        parts_done=0,
        parts_total=1,
        frames_done=5,
        frames_total=10,
        elapsed_s=1.0,  # inner elapsed is ignored for the aggregate ETA
        eta_s=None,
        message="detecting",
    )
    _, status = processing_progress_fraction(
        ProcessingProgress(
            recordings_done=0,
            recordings_total=2,
            current_recording="s1",
            inner=inner,
            elapsed_s=30.0,  # run-level elapsed drives the whole-run ETA
        )
    )
    assert (
        "s1" in status and "1/2" in status
    )  # recordings completed/remaining always shown
    assert "1m30s" in status and "left" in status


def test_processing_progress_fraction_no_eta_before_progress():
    # No aggregate progress yet (0%) → no ETA, but recordings are still shown.
    from detail.core.dataclasses import DetectionProgressEvent, ProcessingProgress

    inner = DetectionProgressEvent(
        parts_done=0,
        parts_total=1,
        frames_done=0,
        frames_total=10,
        elapsed_s=5.0,
        eta_s=None,
        message="loading model",
    )
    _, status = processing_progress_fraction(
        ProcessingProgress(
            recordings_done=0,
            recordings_total=2,
            current_recording="s1",
            inner=inner,
            elapsed_s=5.0,
        )
    )
    assert "1/2" in status and "left" not in status


class _StubHandle:
    def __init__(self, status, error):
        self.status = status
        self.error = error

    def progress(self):
        return iter(())  # no events — go straight to the terminal check


def test_fit_progress_worker_emits_error_on_failure(qapp):
    from detail.gui.workers import FitProgressWorker

    seen = []
    worker = FitProgressWorker(
        _StubHandle("failed", "RuntimeError: boom"), receiver=None
    )
    worker.finished_run.connect(seen.append)
    worker.run()  # synchronous; a direct-connected slot fires inline
    assert seen == ["RuntimeError: boom"]


def test_fit_progress_worker_emits_empty_on_success(qapp):
    from detail.gui.workers import FitProgressWorker

    seen = []
    worker = FitProgressWorker(_StubHandle("complete", None), receiver=None)
    worker.finished_run.connect(seen.append)
    worker.run()
    assert seen == [""]


def test_fit_progress_worker_forwards_the_raw_event(qapp):
    """The signal `DistillRunDetail` draws its stage rows from — asserted on what it *emits*.

    `hasattr(FitProgressWorker, "progress")` is true of a worker that declares the signal and
    never sends anything down it, the same trap `RunProgressWorker`'s sibling test
    (`test_the_worker_forwards_the_event_itself`) guards against — deleting the emit left that
    one green. Drive `run()` and collect what actually came out.
    """
    from detail.core.dataclasses import TrainingProgressEvent
    from detail.gui.workers import FitProgressWorker

    class _Handle:
        status = "complete"
        error = None

        def progress(self):
            yield TrainingProgressEvent(3, 10, {}, 30.0, None)

    sent = []
    worker = FitProgressWorker(_Handle(), receiver=None)
    worker.progress.connect(sent.append)
    worker.run()

    assert sent, (
        "the worker emitted no raw event — a page can then draw one bar and one line, "
        "which is the shape distill's stage rows had no way to move past"
    )
    assert sent[0].epoch == 3


def test_install_model_worker_streams_then_succeeds(qapp, monkeypatch):
    from detail.api import models as api_models
    from detail.core.dataclasses import InstallProgress
    from detail.gui.workers import InstallModelWorker

    def fake_install(model_id, cb, *, variant=None):
        cb(InstallProgress(bytes_done=5, bytes_total=10, current_file="a.pt"))
        cb(InstallProgress(bytes_done=10, bytes_total=None, current_file="a.pt"))

    monkeypatch.setattr(api_models, "install_model", fake_install)
    ticks, done = [], []
    worker = InstallModelWorker("sam3", receiver=None)
    worker.progress.connect(lambda d, t, f: ticks.append((d, t, f)))
    worker.finished_install.connect(done.append)
    worker.run()  # synchronous; direct-connected slots fire inline
    assert ticks == [
        (5, 10, "a.pt"),
        (10, -1, "a.pt"),
    ]  # None total → -1 (indeterminate)
    assert done == [""]


def test_install_model_worker_emits_error(qapp, monkeypatch):
    from detail.api import models as api_models
    from detail.core.dataclasses import ModelReadiness
    from detail.core.protocols import ModelNotReadyError
    from detail.gui.workers import InstallModelWorker

    readiness = ModelReadiness(
        ready=False,
        reason="no_token",
        message="needs token",
        action_url="https://hf.co",
        repo_id=None,
    )

    def fake_install(model_id, cb, *, variant=None):
        raise ModelNotReadyError(readiness)

    monkeypatch.setattr(api_models, "install_model", fake_install)
    done = []
    worker = InstallModelWorker("sam3", receiver=None)
    worker.finished_install.connect(done.append)
    worker.run()
    assert done and "needs token" in done[0]


def test_install_model_worker_installs_the_requested_variant(qapp, monkeypatch):
    """Each size is its own download, so the worker must pass the row's variant —
    without it every Install fetched the model's default variant, and a run
    configured for another size stayed 'not downloaded' no matter how often the
    user clicked Install."""
    from detail.api import models as api_models
    from detail.gui.workers import InstallModelWorker

    seen = {}

    def fake_install(model_id, cb, *, variant=None):
        seen["args"] = (model_id, variant)

    monkeypatch.setattr(api_models, "install_model", fake_install)
    worker = InstallModelWorker("dinov3_convnext_base", "convnext_tiny", receiver=None)
    worker.run()
    assert seen["args"] == ("dinov3_convnext_base", "convnext_tiny")


def test_background_call_runs_off_thread_but_delivers_on_ui_thread(qapp):
    """`BackgroundCall` runs its fn on a worker thread yet delivers `done` on the UI
    thread — so launch.py's bare-closure slots (apply_recordings / QMessageBox) touch
    widgets safely. The `_SyncBackgroundCall` stand-in the other launch tests use runs
    inline and can't exercise this real cross-thread delivery (CLAUDE.md concurrency).
    """
    import threading

    from PyQt6.QtCore import QEventLoop, QTimer

    main_tid = threading.get_ident()
    seen: dict[str, int] = {}
    call = BackgroundCall(lambda: threading.get_ident(), receiver=None)
    loop = QEventLoop()

    def _done(worker_tid: int) -> None:
        seen["worker"] = worker_tid
        seen["deliver"] = threading.get_ident()
        loop.quit()

    call.done.connect(_done)
    call.start()
    QTimer.singleShot(5000, loop.quit)  # safety net; should never fire
    loop.exec()
    call.wait()
    assert seen["worker"] != main_tid  # fn ran OFF the UI thread
    assert seen["deliver"] == main_tid  # done delivered ON the UI thread


def test_processing_progress_eta_measures_only_this_coordinators_work():
    """A retry's bar legitimately starts part-way (29/60 — those recordings really are
    done), but elapsed_s only ever buys the recordings run *since this coordinator
    started*. Dividing the run-wide fraction by it infers a ~30x throughput: after one
    30-minute recording the ETA read "30m left" on a run with 15 hours to go.

    The bar still spans the run; only the ETA's denominator changes.
    """
    from detail.core.dataclasses import ProcessingProgress

    recording_s = 30 * 60
    # 29 already complete; this coordinator has run 1 of the remaining 31.
    event = ProcessingProgress(
        recordings_done=30,
        recordings_total=60,
        current_recording="s30",
        inner=None,
        elapsed_s=recording_s,
        recordings_at_start=29,
    )
    fraction, label = processing_progress_fraction(event)
    assert fraction == pytest.approx(0.5)  # the bar spans the whole run, unchanged
    # 30 recordings left at 30 min each = 15h — not the 30m the run-wide fraction implies.
    assert "15h00m" in label


def test_processing_progress_eta_unchanged_for_a_fresh_run():
    """recordings_at_start=0 (every run before retry existed, and every new one) must
    take exactly the old path: elapsed bought the whole fraction."""
    from detail.core.dataclasses import ProcessingProgress

    recording_s = 30 * 60
    event = ProcessingProgress(
        recordings_done=30,
        recordings_total=60,
        current_recording="s30",
        inner=None,
        elapsed_s=30 * recording_s,  # it really did run all 30
        recordings_at_start=0,
    )
    fraction, label = processing_progress_fraction(event)
    assert fraction == pytest.approx(0.5)
    assert "15h00m" in label  # 30 left at the observed rate


def test_processing_progress_survives_a_record_without_the_retry_key():
    """A progress.json written before recordings_at_start existed must still load."""
    from detail.core.dataclasses import processing_progress_from_dict

    event = processing_progress_from_dict(
        {
            "recordings_done": 1,
            "recordings_total": 2,
            "current_recording": "s",
            "inner": None,
            "elapsed_s": 10.0,
        }
    )
    assert event.recordings_at_start == 0


def test_distill_progress_fraction_maps_phase_and_message():
    from detail.core.dataclasses import DistillProgress

    fraction, status = fit_progress_fraction(
        DistillProgress("training", 1, 2, 0.4, 0.72, 0.5, 5.0, "epoch 2/5")
    )
    assert fraction == pytest.approx(0.72)  # the folded overall fraction drives the bar
    assert "training" in status  # phase
    assert "epoch 2/5" in status  # advancing message


def test_fit_progress_worker_outlives_the_page_that_started_it(qapp):
    """A FitProgressWorker is a parentless QThread whose only Python reference lives on the
    page that started it. The shell deleteLater()s a page on navigation, so that reference
    drops while the thread is still polling — and destroying a *running* QThread is a Qt
    fatal abort, not an exception. It has to hold its own reference until it finishes.
    """
    from detail.gui.workers import _LIVE_WORKERS, FitProgressWorker

    class _Handle:
        status = "complete"
        error = None

        def progress(self):
            return iter(())

    worker = FitProgressWorker(_Handle(), receiver=None)
    worker.start()
    assert worker in _LIVE_WORKERS  # referenced independently of any page

    worker.wait()
    qapp.processEvents()  # let the finished signal land
    assert worker not in _LIVE_WORKERS  # ...and released once it is safe


def test_fit_progress_fraction_never_renders_a_position_past_the_total():
    """A terminal analysis event sets ``recordings_done == recordings_total``, and the
    label used to render ``done + 1`` unclamped — "recording 3/2", or "recording 2/1" on
    a single-recording run. The CLI already clamped this (``cli.py::_recording_position``
    and its comment); the GUI's mapping did not, so every module hit it at 100%."""
    from detail.pipeline.location_analysis import LocationProgressEvent

    for total in (1, 2, 5):
        event = LocationProgressEvent(
            recordings_done=total,  # terminal: every recording finished
            recordings_total=total,
            frames_done=0,
            frames_total=0,
            elapsed_s=1.0,
            eta_s=0.0,
            message="done",
        )
        _fraction, status = fit_progress_fraction(event)
        assert f"recording {total}/{total}" in status, status
        assert f"recording {total + 1}/{total}" not in status


def test_a_worker_with_a_delivery_in_flight_is_not_collected(qapp):
    """The class: **a queued Qt delivery outliving the Python objects it targets.**

    A widget that starts a `BackgroundCall` typically holds it as `self._call` and connects a
    closure back to itself, so worker → connection → closure → widget → worker is a cycle with
    nothing outside it. When the widget's scope ends, the *cyclic* collector breaks the cycle by
    **clearing the closure**: `func_clear` drops `__defaults__` and blanks `__qualname__`.

    The C++ side of the connection survives that. A `done` posted before the collection is later
    delivered into a cleared function, and the ordinary `lambda result, g=generation:` idiom is
    suddenly missing an argument it always supplied itself —

        TypeError: () missing 1 required positional argument: 'g'

    with no function name and **no traceback**, because it is raised at the C++ boundary before
    any Python frame exists. It surfaces in whichever test is pumping the event loop, which is why
    it read for weeks as an unrelated flake in a file that had nothing to do with it.

    `run_busy` never had this: it roots its worker in `busy._active`. A directly constructed
    `BackgroundCall` had no root, so this pins the root on the class — the four call sites that
    used the idiom were not special, and the next one written would have inherited the bug.
    """
    import gc

    from detail.gui.workers import _IN_FLIGHT, BackgroundCall

    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        release.wait(5)
        return "payload"

    class Holder:
        """The shape every call site has: the worker on the widget, a closure back to it."""

        def __init__(self):
            self.seen = []
            self._call = BackgroundCall(work, receiver=None)
            generation = 7
            self._connected = lambda result, g=generation: self.seen.append((g, result))
            self._call.done.connect(self._connected)
            self._call.start()

    holder = Holder()
    assert started.wait(5), "the worker never ran"
    assert holder._call in _IN_FLIGHT, "a running worker must be rooted"

    # A weak reference, so watching the closure does not itself keep it reachable — the whole
    # question is what the collector does when nothing else holds it. Scanning the heap for any
    # cleared function instead would fail on closures other tests left behind, which is a
    # different bug and not this test's business.
    closure = weakref.ref(holder._connected)

    # The caller's scope ends while the delivery is still in flight.
    del holder
    gc.collect()

    survivor = closure()
    assert (
        survivor is not None
    ), "the rooted worker must keep its connected closure alive"
    assert survivor.__defaults__, (
        "the cyclic collector cleared a connected closure while a delivery was queued: "
        "`func_clear` drops __defaults__, so the next delivery arrives one argument short"
    )
    assert survivor.__qualname__, "a cleared function has its qualname blanked"

    release.set()


def test_pool_work_does_not_cross_a_test_boundary(qapp):
    """The class: **a queued Qt delivery outliving the test that started it.**

    `_no_leaked_busy_calls` has always drained `run_busy`'s workers, because one of those
    delivering into a later test once ended in a modal box with nobody to dismiss it and stalled
    CI for 1h43m. Thumbnail decoding does not go through `run_busy` — it goes through
    `QThreadPool` (`clip_grid`, `cluster_strip`, `scrubber`) — and nothing waited for that half.
    Measured before this drain existed: **85 of 1416 GUI tests** ended with a pool task still
    running, and the test it would land on was always an unrelated one.

    Bounded on purpose. A hard `waitForDone()` would turn a task that never finishes into a hung
    session, which is the failure this exists to prevent.
    """
    from PyQt6.QtCore import QRunnable, QThreadPool, QTimer

    from tests.gui.conftest import drain_thread_pool

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class Task(QRunnable):
        def run(self):
            started.set()
            release.wait(5)
            finished.set()

    QThreadPool.globalInstance().start(Task())
    assert started.wait(5), "the pool never ran the task"

    # Release the task from *inside* the drain's own event pumping, not before calling it.
    # `drain_thread_pool` snapshots `activeThreadCount()` on entry and returns that snapshot, so
    # releasing first is a race against it: the task can finish before the snapshot is taken, the
    # drain then correctly reports 0, and this test fails having proved nothing. A zero-timer fires
    # on the drain's first `processEvents()` — after the snapshot, before the wait loop can exit —
    # which is deterministic instead. The race lost roughly 1 full-suite run in 14 once the suite
    # became parallel, because 32-way scheduling widens the window (#557).
    QTimer.singleShot(0, release.set)
    outstanding = drain_thread_pool(qapp)

    assert outstanding >= 1, "the drain must notice work that was in flight"
    assert finished.is_set(), "the drain must wait for the task to finish"
    assert QThreadPool.globalInstance().activeThreadCount() == 0


def test_the_pool_drain_gives_up_rather_than_hanging(qapp):
    """A task that never finishes must not stall the session — the drain is bounded, so a stuck
    decode costs a bounded wait at one teardown instead of the whole run."""
    import time

    from PyQt6.QtCore import QRunnable, QThreadPool

    from tests.gui.conftest import drain_thread_pool

    release = threading.Event()

    class Stuck(QRunnable):
        def run(self):
            release.wait(30)

    QThreadPool.globalInstance().start(Stuck())
    began = time.monotonic()
    drain_thread_pool(qapp, timeout_ms=200)
    elapsed = time.monotonic() - began

    assert elapsed < 5, f"the drain blocked for {elapsed:.1f}s instead of giving up"
    release.set()
    QThreadPool.globalInstance().waitForDone(5_000)
