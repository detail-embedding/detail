"""The Export dialog narrows an export to the recordings you pick, and says what it will cost.

The class and the incident are written up in `tests/api/test_export_recordings_scope.py` and
`tests/api/test_export_estimates.py`: an export that wrote one video per recording of a
whole run did not fit the drive it was written to, when one recording was wanted
(issue #710). This is the dialog half, and it pins what a person sees:

- a run of several recordings shows them, all ticked, and the scope the export runs with is
  *everything* until some are unticked (so "all of them" is never recorded as a thirty-name list);
- a page that is showing one recording (`RunResults`, the run page's list) opens the dialog on
  that one, as a cluster workspace opens it on its cluster: the page says what you are looking
  at, and the list is where you widen it;
- the list is greyed, not hidden, while nothing ticked above is written per recording, and a run
  of one recording has no list at all (a control with one entry cannot be operated);
- the estimate follows the ticks, comes off the UI thread, drops an answer to a question that
  has since changed, and, when it says the export will not fit, the **Export** button is off and
  the dialog says why.

Where it stops: the list is offered for a processing run. An analysis run is narrowed by cluster
or selection in the dialog and by recording through the API and `--for-recording`; its recordings
are only known by reading the whole behavior artifact, which the dialog should not do on the UI
thread.
"""

import pytest

from detail.api import exporters
from detail.api.exporters import ExportScope, ExportSubject
from detail.gui import export as gui_export
from detail.gui.qt import QDialogButtonBox
from tests import waits


@pytest.fixture
def two(processed_recording):
    built = processed_recording
    square = built.square(4, 20, 4, 20)
    built.add_recording(
        "recording_002",
        detections=[(i, 0, (0.1, 0.1, 0.5, 0.5), square) for i in range(3)],
        object_ids=[0, 0, 0],
    )
    return built


def _dialog(built, scope=None, **kwargs):
    subject = ExportSubject("run", built.experiment.dir, built.detection_run_id)
    return gui_export.ExportDialog(
        subject, exporters.targets_for(subject), scope, **kwargs
    )


def _settled(qapp, dialog):
    """Ask now, rather than after the pause, and wait for the answer to land."""
    dialog._estimate_timer.stop()
    dialog._estimate = None
    dialog._start_estimate()
    waits.until(qapp, lambda: dialog._estimate is not None, "the export estimate")


def _ok(dialog):
    return dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)


def _names(dialog):
    lst = dialog._recording_list
    return [lst.item(i).text() for i in range(lst.count())]


# --------------------------------------------------------------------------- #
# the list
# --------------------------------------------------------------------------- #
def test_a_run_of_several_recordings_shows_them_all_ticked(qapp, two):
    dialog = _dialog(two)

    assert dialog._recording_list is not None
    assert len(_names(dialog)) == 2
    assert dialog._ticked_recordings() == ["recording_001", "recording_002"]
    assert (
        dialog._final_scope().recording_ids == ()
    ), "all of them is the scope that means everything"
    assert "all 2" in dialog._recordings_count.text()
    dialog.deleteLater()
    qapp.processEvents()


def test_a_page_showing_one_recording_opens_the_dialog_on_it(qapp, two):
    dialog = _dialog(two, ExportScope(recording_ids=("recording_002",)))

    assert dialog._ticked_recordings() == ["recording_002"]
    assert dialog._final_scope().recording_ids == ("recording_002",)
    assert (
        dialog._scope_pick is None
    ), "the list owns this choice; a second control would repeat it"
    assert "1 of 2 selected" in dialog._recordings_count.text()
    dialog.deleteLater()
    qapp.processEvents()


def test_unticking_makes_it_a_real_choice_and_ticking_all_makes_it_everything_again(
    qapp, two
):
    dialog = _dialog(two)

    dialog._recording_list.item(0).setCheckState(gui_export.Qt.CheckState.Unchecked)
    assert dialog._final_scope().recording_ids == ("recording_002",)

    dialog._tick_recordings(True)
    assert dialog._final_scope().recording_ids == ()
    dialog._tick_recordings(False)
    assert dialog._ticked_recordings() == []
    dialog.deleteLater()
    qapp.processEvents()


def test_a_run_of_one_recording_has_no_list(qapp, processed_recording):
    dialog = _dialog(processed_recording)

    assert dialog._recording_list is None
    assert dialog._final_scope().recording_ids == ()
    dialog.deleteLater()
    qapp.processEvents()


def test_the_list_is_greyed_not_hidden_while_nothing_ticked_is_per_recording(qapp, two):
    dialog = _dialog(two)
    assert dialog._recordings_holder.isEnabled()

    for target_id, check in dialog._include.items():
        if check.isChecked():
            check.setChecked(False)  # every default-ticked target here is per recording

    assert not dialog._recordings_holder.isEnabled()
    assert "written per recording" in dialog._recordings_count.text()
    assert not dialog._recordings_holder.isHidden()
    # and a recordings choice is not recorded on an export that does not use it
    dialog._tick_recordings(False)
    (
        dialog._include["performance"].setChecked(True)
        if dialog._include["performance"].isEnabled()
        else None
    )
    assert dialog._final_scope().recording_ids == ()
    dialog.deleteLater()
    qapp.processEvents()


def test_exporting_with_no_recording_ticked_is_refused(
    qapp, two, tmp_path, monkeypatch
):
    warned = []
    monkeypatch.setattr(
        gui_export.QMessageBox, "warning", lambda *a, **k: warned.append(a[2])
    )
    dialog = _dialog(two)
    dialog._out.setText(str(tmp_path / "out"))
    dialog._tick_recordings(False)

    dialog._export()

    assert warned == ["Tick at least one recording."]
    assert dialog.run_handle is None
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


# --------------------------------------------------------------------------- #
# the estimate
# --------------------------------------------------------------------------- #
def test_the_estimate_follows_the_ticks(qapp, two):
    dialog = _dialog(two)
    for target_id, check in dialog._include.items():
        check.setChecked(target_id == "masks")
    _settled(qapp, dialog)
    both = dialog._estimate.bytes
    assert both and both > 0
    assert "about" in dialog._estimate_label.text(), "one target, fully estimated"

    dialog._recording_list.item(0).setCheckState(gui_export.Qt.CheckState.Unchecked)
    dialog._recording_list.item(1).setCheckState(gui_export.Qt.CheckState.Checked)
    _settled(qapp, dialog)

    assert dialog._estimate.bytes < both, "one recording is less than two"
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


def test_a_change_says_it_is_asking_again(qapp, two):
    dialog = _dialog(two)
    _settled(qapp, dialog)
    dialog._tick_recordings(False)
    assert dialog._estimate_label.text() == "Estimating…"
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


def test_a_target_that_cannot_be_sized_says_so_instead_of_showing_a_number(
    qapp, two, monkeypatch
):
    """Here the only ticked target is given an estimator that cannot know. The dialog says it
    cannot be estimated and why, rather than a zero. (The 'at least' total, when only some
    cannot be sized, is pinned in `test_export_estimates`: only one target is ticked here.)
    """
    import dataclasses

    from detail.api.exporters import Estimate

    masks = exporters.target("run", "masks")
    unsized = dataclasses.replace(
        masks, estimate=lambda s, sc, o: Estimate(None, None, "depends on the machine")
    )
    monkeypatch.setattr(
        exporters,
        "_TARGETS",
        [unsized if t is masks else t for t in exporters._TARGETS],
    )
    dialog = _dialog(two)
    _settled(qapp, dialog)

    assert "cannot be estimated" in dialog._estimate_label.text()
    assert "depends on the machine" in dialog._estimate_label.toolTip()
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


def test_the_estimate_says_how_much_room_there_is_where_it_is_going(
    qapp, two, tmp_path
):
    dialog = _dialog(two)
    dialog._out.setText(str(tmp_path / "out"))
    _settled(qapp, dialog)

    assert "free there" in dialog._estimate_label.text()
    # the drive is live (this test's own files, other processes): the same drive, not the same byte
    assert dialog._estimate.free_bytes == pytest.approx(
        exporters.free_bytes_at(tmp_path), abs=64 * 1024 * 1024
    )
    assert _ok(dialog).isEnabled()
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


def test_an_export_that_will_not_fit_is_not_offered_and_says_why(
    qapp, two, tmp_path, monkeypatch
):
    warned = []
    monkeypatch.setattr(
        gui_export.QMessageBox, "warning", lambda *a, **k: warned.append(a[2])
    )
    monkeypatch.setattr(exporters, "free_bytes_at", lambda path: 5)
    dialog = _dialog(two)
    dialog._out.setText(str(tmp_path / "out"))
    _settled(qapp, dialog)

    assert "will not fit" in dialog._estimate_label.text()
    assert not _ok(dialog).isEnabled()
    dialog._export()  # whichever way it is reached, it does not start
    assert dialog.run_handle is None and warned and "is free where" in warned[0]

    monkeypatch.undo()  # another folder, with room
    dialog._out.setText(str(tmp_path / "elsewhere"))
    _settled(qapp, dialog)
    assert _ok(dialog).isEnabled()
    assert "will not fit" not in dialog._estimate_label.text()
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


def test_an_answer_to_a_question_that_has_since_changed_is_dropped(qapp, two):
    dialog = _dialog(two)
    _settled(qapp, dialog)
    shown = dialog._estimate_label.text()

    stale = exporters.ExportEstimate(
        (("masks", exporters.Estimate(1, 1.0)),), free_bytes=0
    )
    dialog._on_estimated(dialog._estimate_generation - 1, stale)

    assert dialog._estimate_label.text() == shown
    assert _ok(dialog).isEnabled(), "a stale 'will not fit' must not switch Export off"
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


def test_nothing_ticked_says_so_instead_of_estimating(qapp, two):
    dialog = _dialog(two)
    for check in dialog._include.values():
        check.setChecked(False)
    dialog._estimate_timer.stop()
    dialog._start_estimate()
    assert dialog._estimate_label.text() == "Tick something to export."
    assert _ok(dialog).isEnabled()
    dialog.deleteLater()
    qapp.processEvents()


def test_the_tooltip_says_how_each_figure_was_reached(qapp, two):
    dialog = _dialog(two)
    _settled(qapp, dialog)
    tip = dialog._estimate_label.toolTip()
    assert "Masks:" in tip and "detections" in tip
    assert "Estimates, not promises" in tip
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    qapp.processEvents()


# --------------------------------------------------------------------------- #
# the page: what it is looking at
# --------------------------------------------------------------------------- #
def _run_results(built):
    from detail.gui.pages import RunResults
    from detail.storage import runs as runs_store

    record = runs_store.read_run(built.experiment.dir, built.detection_run_id)
    return RunResults(built.experiment, record, ui=None)


def test_the_run_pages_list_supplies_the_selected_recording_as_the_scope(qapp, two):
    results = _run_results(two)
    first = results.export_scope()
    assert first == ExportScope(
        recording_ids=("recording_001",)
    ), "it opens on a recording"

    results._table.view.selectRow(1)
    assert results.export_scope() == ExportScope(recording_ids=("recording_002",))
    assert gui_export.scope_of(results) == results.export_scope()
    results.deleteLater()
    qapp.processEvents()


def test_a_run_of_one_recording_has_no_scope_to_supply(qapp, processed_recording):
    results = _run_results(processed_recording)
    assert results.export_scope() is None
    results.deleteLater()
    qapp.processEvents()


def test_the_dialog_opens_on_what_the_page_is_showing(qapp, two):
    results = _run_results(two)
    results._table.view.selectRow(1)

    dialog = _dialog(two, gui_export.scope_of(results))

    assert dialog._ticked_recordings() == ["recording_002"]
    dialog._estimate_timer.stop()
    dialog.deleteLater()
    results.deleteLater()
    qapp.processEvents()
