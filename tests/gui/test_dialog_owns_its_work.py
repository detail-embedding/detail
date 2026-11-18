"""A dialog whose answer comes from background work stays open until that work has answered.

**The class.** A dialog starts work off the UI thread and closes as *Accepted* when the work's
result arrives. Esc, the window's close button and a Cancel wired to ``reject`` close it as
*Rejected* — and nothing stopped them while the work ran, so the dialog answered "cancelled"
and the work went on. Reported in #35: the launch dialogs closed during validation, the page
did not open the run, and ~2 s later the run started anyway (a second attempt started a second
run); Add video part closed on Cancel and the part was added behind a page still showing the old
list. The export dialog, the behavior-label import and the skeleton editor's save had the same
shape, found by the sweep.

**How it hid.** Every test of these dialogs drove ``_run`` / ``_start`` and then the worker's
result — usually with a stand-in worker that runs inline, so no moment existed in which a close
could arrive while the work ran. Nothing ever sent Esc to a dialog with work in flight.

**The rule.** The dialog passes its worker to ``Dialog.hold_open_while``. While it runs, a close
request stops the work (or says why it cannot be stopped) and the dialog closes on the result:
accepted if the work happened, rejected if it stopped. These tests send the real keys to a
shown dialog with a real worker thread held at a gate, so the close lands mid-work.
``requirements.a_dialog_that_runs_work_owns_its_close`` is the structural half, swept over the
package: it names any work whose result accepts a dialog that is not held.
"""

import threading

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from tests import waits  # noqa: E402
from tests.gui.test_launch import _experiment_with_mixed_recordings  # noqa: E402

#: Every way a user closes a dialog that is not its OK button.
_CLOSES = ("escape", "window close", "cancel button")


def _close(dialog, how: str) -> None:
    if how == "escape":
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
    elif how == "window close":
        dialog.close()
    else:
        box = dialog.findChild(QDialogButtonBox)
        box.button(QDialogButtonBox.StandardButton.Cancel).click()


def _shown(qapp, dialog) -> None:
    dialog.resize(640, 640)
    dialog.show()
    qapp.processEvents()


@pytest.fixture(autouse=True)
def _no_message_boxes(monkeypatch):
    shown: list = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: shown.append(a[-1]) or None
    )
    yield
    assert not shown, f"an error was reported: {shown}"


def _launch_ready(dialog, module_id: str) -> None:
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData(module_id))
    dialog._select_all.click()
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {}
    dialog._config.covered_recording_ids = lambda: None


# --- the launch dialogs ------------------------------------------------------ #
@pytest.mark.parametrize("how", _CLOSES)
def test_closing_the_launch_dialog_while_it_validates_starts_no_run(
    qapp, tmp_path, monkeypatch, how
):
    """Finding 1: closed during "Validating N recording(s)…", the dialog said cancelled and
    the run started ~2 s later. Now it stays open, the stop lands on validation's next
    report — before any run is reserved — and the dialog closes rejected with nothing
    started."""
    from detail.gui.launch import LaunchRunDialog

    gate, validating, spawned = threading.Event(), threading.Event(), []

    def run_processing(self, module_id, recordings, config, progress=None):
        progress("Validating", 0.0, "first (1/2)")
        validating.set()
        gate.wait(30)
        progress("Validating", 0.5, "second (2/2)")  # where the API reports next
        spawned.append(module_id)
        return object()

    monkeypatch.setattr(Experiment, "run_processing", run_processing)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    _launch_ready(dialog, "detection")
    _shown(qapp, dialog)
    try:
        dialog._run()
        assert validating.wait(30), "the launch never started validating"
        _close(dialog, how)
        qapp.processEvents()
        assert dialog.isVisible(), "closed as cancelled while its run was being started"
        assert dialog._status.text().startswith("Stopping"), dialog._status.text()
    finally:
        gate.set()
    waits.finished(dialog._launch_call, "the launch worker")
    waits.until(qapp, lambda: not dialog.isVisible(), "the stopped launch closing")
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert spawned == [], "a run was started after the user stopped the launch"
    assert dialog.run_handle is None


def test_a_launch_too_far_along_to_stop_opens_its_run_instead(
    qapp, tmp_path, monkeypatch
):
    """The other honest outcome. `run_analysis` reports nothing, so a stop cannot land: the
    run starts, and the dialog accepts with its handle — the page opens the run — rather
    than closing as though nothing had started."""
    from detail.gui.launch import LaunchAnalysisDialog

    gate, launching = threading.Event(), threading.Event()
    handle = object()

    def run_analysis(self, module_id, config):
        launching.set()
        gate.wait(30)
        return handle

    monkeypatch.setattr(Experiment, "run_analysis", run_analysis)
    dialog = LaunchAnalysisDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._select_all.click()
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {}
    dialog._config.covered_recording_ids = lambda: None
    _shown(qapp, dialog)
    try:
        dialog._run()
        assert launching.wait(30), "the launch never started"
        _close(dialog, "escape")
        qapp.processEvents()
        assert dialog.isVisible(), "closed as cancelled while its run was being started"
    finally:
        gate.set()
    waits.finished(dialog._launch_call, "the launch worker")
    waits.until(qapp, lambda: not dialog.isVisible(), "the launched dialog closing")
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.run_handle is handle


def test_the_launch_dialog_closes_normally_once_nothing_is_running(qapp, tmp_path):
    """The hold is for work in flight only: an idle dialog still closes on Esc."""
    from detail.gui.launch import LaunchRunDialog

    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    _shown(qapp, dialog)
    _close(dialog, "escape")
    assert not dialog.isVisible()
    assert dialog.result() == QDialog.DialogCode.Rejected


# --- Add video part ---------------------------------------------------------- #
def _add_part_dialog(qapp, tmp_path, add_video_part):
    from detail.gui.pages import AddVideoPartDialog

    recording = Experiment.create("p", tmp_path).create_recording("r1")
    recording.add_video_part = add_video_part  # instance override
    dialog = AddVideoPartDialog(recording)
    dialog._path.setText(str(tmp_path / "clip.mp4"))
    _shown(qapp, dialog)
    return dialog


@pytest.mark.parametrize("how", _CLOSES)
def test_closing_add_video_part_mid_copy_stops_it_and_adds_nothing(qapp, tmp_path, how):
    """Finding 8: Cancel and Esc stayed live during the copy, the dialog closed, and the part
    was added afterwards with the page showing the old list. Now a close asks the copy to
    stop (``should_stop``, which removes the partial file) and the dialog closes once it
    has."""
    from detail.storage.video import ImportStopped

    gate, copying, added = threading.Event(), threading.Event(), []

    def add_video_part(path, progress=None, should_stop=None, **kwargs):
        copying.set()
        gate.wait(30)
        if should_stop is not None and should_stop():
            raise ImportStopped()
        added.append(path)

    dialog = _add_part_dialog(qapp, tmp_path, add_video_part)
    try:
        dialog._start()
        assert copying.wait(30), "the copy never started"
        _close(dialog, how)
        qapp.processEvents()
        assert dialog.isVisible(), "closed as cancelled while the part was being added"
        assert dialog._worker.stop_requested(), "the close did not ask the copy to stop"
    finally:
        gate.set()
    waits.finished(dialog._worker, "the add-part worker")
    waits.until(qapp, lambda: not dialog.isVisible(), "the stopped dialog closing")
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert added == []


def test_a_part_added_despite_a_stop_is_accepted_so_the_page_refreshes(qapp, tmp_path):
    """A referenced part copies nothing, so a stop has nothing to interrupt and the part is
    added. The dialog accepts — `_add_part` refreshes the page on Accepted — instead of
    closing as cancelled over a part that now exists."""
    gate, copying, added = threading.Event(), threading.Event(), []

    def add_video_part(path, progress=None, should_stop=None, **kwargs):
        copying.set()
        gate.wait(30)
        added.append(path)  # never asks should_stop

    dialog = _add_part_dialog(qapp, tmp_path, add_video_part)
    try:
        dialog._start()
        assert copying.wait(30), "the import never started"
        _close(dialog, "escape")
        qapp.processEvents()
        assert dialog.isVisible()
    finally:
        gate.set()
    waits.finished(dialog._worker, "the add-part worker")
    waits.until(qapp, lambda: not dialog.isVisible(), "the dialog closing")
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert len(added) == 1


# --- Export ------------------------------------------------------------------ #
def test_escape_while_an_export_is_submitted_waits_and_hands_the_run_up(
    qapp, tmp_path, monkeypatch
):
    """The finding-1 shape in the export dialog (#35 finding 15 asked for it to be checked):
    Esc during submission closed the dialog as cancelled and the export run started anyway,
    with no page shown for it. Submission cannot be stopped part-way, so the dialog waits,
    says so, and accepts with the run."""
    from detail.api import exporters
    from detail.api.exporters import ExportSubject
    from detail.gui import export as gui_export

    proj = Experiment.create("study", tmp_path)
    subject = ExportSubject("experiment", proj.dir, None)
    gate, submitting = threading.Event(), threading.Event()
    handle = object()

    def run_export(self, kind, subject_id, target_ids, parent, **kwargs):
        submitting.set()
        gate.wait(30)
        return handle

    monkeypatch.setattr(Experiment, "run_export", run_export)
    dialog = gui_export.ExportDialog(subject, exporters.targets_for(subject), None)
    dialog._out.setText(str(tmp_path / "out"))
    if not dialog._chosen():
        next(c for c in dialog._include.values() if c.isEnabled()).setChecked(True)
    dialog._estimate = None
    _shown(qapp, dialog)
    try:
        dialog._export()
        assert submitting.wait(30), "the export was never submitted"
        _close(dialog, "escape")
        qapp.processEvents()
        assert dialog.isVisible(), "closed as cancelled while its run was being started"
        assert "Still submitting" in dialog._status.text()
    finally:
        gate.set()
    waits.finished(dialog._submit_call, "the export submission")
    waits.until(qapp, lambda: not dialog.isVisible(), "the export dialog closing")
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.run_handle is handle
