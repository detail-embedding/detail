"""Metadata page (§11 grouping variables) + no-video recording detail render.

Column layout, fixed at the front of every table: ``recording_id`` (0), the derived
``file`` column (1), then the user's own metadata columns, then the completion
columns (§ "Recording information" in CLAUDE.md / OQ-TBL-3..8). ``_USER_COL_START``
and ``_COMPLETION_COLUMNS`` below are the page's own constants, imported rather than
hard-coded, so a column added to either group doesn't silently desync these tests.
"""

from detail.api.experiment import Experiment
from detail.gui.pages import _COMPLETION_COLUMNS, MetadataPage
from detail.storage import recording as recording_store

#: recording_id + file — every user column starts after these two.
_FIXED_COLS = MetadataPage._USER_COL_START
_N_COMPLETION = len(_COMPLETION_COLUMNS)


def test_metadata_page_edit_then_save_writes_through(qapp, tmp_path):
    from detail.gui.qt import QAbstractItemView

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    experiment.create_recording("recording_002", metadata={"animal_id": "M002"})
    page = MetadataPage(experiment)
    assert page._table.rowCount() == 2
    # recording_id + file + animal_id + 8 completion columns
    assert page._table.columnCount() == _FIXED_COLS + 1 + _N_COMPLETION
    assert page._table.horizontalHeaderItem(0).text() == "recording_id"
    assert page._table.horizontalHeaderItem(1).text() == "file"
    assert page._table.horizontalHeaderItem(2).text() == "animal_id"

    # view-only by default — no edit triggers, no unwritten changes
    assert page._table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    page._table.item(0, 2).setText("staged-but-not-saved")
    assert experiment.recording_metadata["recording_001"]["animal_id"] == "M001"

    # Edit → change → Save writes through
    page._enter_edit()
    assert page._table.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers
    page._table.item(0, 2).setText("M001b")
    page._save()
    assert page._editing is False
    assert experiment.recording_metadata["recording_001"]["animal_id"] == "M001b"


def test_metadata_page_cancel_discards_edits(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    page = MetadataPage(experiment)
    page._enter_edit()
    page._table.item(0, 2).setText("bad")
    page._cancel()
    assert page._editing is False
    assert experiment.recording_metadata["recording_001"]["animal_id"] == "M001"
    assert page._table.item(0, 2).text() == "M001"  # table reverted to the store


def test_metadata_page_copy_paste(qapp, tmp_path):
    from detail.gui.qt import QGuiApplication

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    experiment.create_recording("recording_002", metadata={"animal_id": "M002"})
    page = MetadataPage(experiment)

    # copy works in view mode: selected cell → clipboard
    page._table.setCurrentCell(0, 2)
    page._copy()
    assert QGuiApplication.clipboard().text() == "M001"

    # paste is a no-op outside edit mode
    QGuiApplication.clipboard().setText("nope")
    page._table.setCurrentCell(0, 2)
    page._paste()
    assert page._table.item(0, 2).text() == "M001"

    # in edit mode, a 2×1 TSV block fills down from the current cell...
    page._enter_edit()
    QGuiApplication.clipboard().setText("A\nB")
    page._table.setCurrentCell(0, 2)
    page._paste()
    assert (page._table.item(0, 2).text(), page._table.item(1, 2).text()) == ("A", "B")

    # ...and recording_id (col 0) is never overwritten
    QGuiApplication.clipboard().setText("X")
    page._table.setCurrentCell(0, 0)
    page._paste()
    assert page._table.item(0, 0).text() == "recording_001"

    # ...neither is the derived file column (col 1)
    QGuiApplication.clipboard().setText("X")
    page._table.setCurrentCell(0, 1)
    page._paste()
    assert page._table.item(0, 1).text() == ""

    # ...nor a completion column
    completion_col = page._completion_start()
    QGuiApplication.clipboard().setText("X")
    page._table.setCurrentCell(0, completion_col)
    page._paste()
    assert page._table.item(0, completion_col).text() == ""


def test_metadata_page_export_csv(qapp, tmp_path, monkeypatch):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    out = tmp_path / "template.csv"
    monkeypatch.setattr(
        pages.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
    )
    monkeypatch.setattr(pages.QMessageBox, "information", lambda *a, **k: None)
    page = MetadataPage(experiment)
    page._export_csv()  # writes off-thread now (run_busy)
    from detail.gui.qt import qwait

    from detail.gui import busy

    # Wait for the **delivery**, not for the file. The worker writes the CSV and only then
    # emits `done`, so a loop that stops at `out.exists()` can return with the callback still
    # queued — and that callback is a modal `QMessageBox.information` whose patch this test is
    # about to undo. That leak hung CI for 1h43m. `tests/gui/conftest.py` now drains it at
    # teardown regardless; this waits for the right thing in the first place.
    for _ in range(200):
        if out.exists() and not busy._active:
            break
        qwait(10)
    # The template carries the name and the video file name so the person filling it in
    # can tell the rows apart — an id alone says nothing. Both are keys on the way back,
    # never values (#330). The completion columns are never part of the CSV — they aren't
    # stored data, so a round trip through Import has nowhere to put them back.
    assert out.read_text(encoding="utf-8-sig") == (
        "recording_id,name,file,animal_id\nrecording_001,recording_001,,M001\n"
    )


def test_metadata_page_add_column_commits_on_save(qapp, tmp_path, monkeypatch):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1", metadata={"animal_id": "M001"})
    monkeypatch.setattr(
        pages.QInputDialog, "getText", lambda *a, **k: ("treatment", True)
    )
    page = MetadataPage(experiment)
    page._enter_edit()
    page._add_column()
    # inserted right after animal_id, before the completion columns
    assert page._table.horizontalHeaderItem(3).text() == "treatment"
    assert page._table.horizontalHeaderItem(4).text() == _COMPLETION_COLUMNS[0][0]
    page._table.item(0, 3).setText("drug_A")  # a value in the new column for r1
    page._save()
    assert experiment.metadata_columns == ["animal_id", "treatment"]
    assert experiment.recording_metadata["r1"] == {
        "animal_id": "M001",
        "treatment": "drug_A",
    }


def test_metadata_page_add_column_cancel_discards(qapp, tmp_path, monkeypatch):
    # The reported bug: add a column, then Cancel — the column must NOT persist.
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1", metadata={"animal_id": "M001"})
    monkeypatch.setattr(
        pages.QInputDialog, "getText", lambda *a, **k: ("treatment", True)
    )
    page = MetadataPage(experiment)
    page._enter_edit()
    page._add_column()
    # staged: recording_id + file + animal_id + treatment + completion columns
    assert page._table.columnCount() == _FIXED_COLS + 2 + _N_COMPLETION
    assert experiment.metadata_columns == ["animal_id"]  # store untouched while staging
    page._cancel()
    assert experiment.metadata_columns == ["animal_id"]  # the added column is gone
    assert page._table.columnCount() == _FIXED_COLS + 1 + _N_COMPLETION


def test_metadata_page_add_column_is_gated_behind_edit_mode(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    page = MetadataPage(experiment)
    assert page._add_col_btn.isHidden()  # not a peer of Edit in view mode
    page._enter_edit()
    assert not page._add_col_btn.isHidden()  # appears inside edit mode
    page._cancel()
    assert page._add_col_btn.isHidden()  # hidden again on leaving edit mode


def test_metadata_page_staging_does_not_touch_store_until_save(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1", metadata={"animal_id": "M001"})
    monkeypatch.setattr(pages.QInputDialog, "getText", lambda *a, **k: ("cohort", True))
    page = MetadataPage(experiment)
    page._enter_edit()
    page._table.item(0, 2).setText("EDITED")  # a cell edit
    page._add_column()  # + a column
    # every mutation is staged in the table only — disk is untouched until Save
    assert experiment.metadata_columns == ["animal_id"]
    assert experiment.recording_metadata["r1"] == {"animal_id": "M001"}


def test_metadata_page_delete_column_commits_on_save(qapp, tmp_path, monkeypatch):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1", metadata={"animal_id": "M001"})
    experiment.add_metadata_column("treatment")
    monkeypatch.setattr(
        pages.QInputDialog, "getItem", lambda *a, **k: ("treatment", True)
    )
    page = MetadataPage(experiment)
    page._enter_edit()
    page._delete_column()
    assert experiment.metadata_columns == ["animal_id", "treatment"]  # not yet written
    page._save()
    assert experiment.metadata_columns == ["animal_id"]
    assert experiment.recording_metadata["r1"] == {"animal_id": "M001"}


def test_metadata_page_delete_column_cancel_restores(qapp, tmp_path, monkeypatch):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1", metadata={"animal_id": "M001"})
    monkeypatch.setattr(
        pages.QInputDialog, "getItem", lambda *a, **k: ("animal_id", True)
    )
    page = MetadataPage(experiment)
    page._enter_edit()
    page._delete_column()
    # staged removal → recording_id + file + completion columns
    assert page._table.columnCount() == _FIXED_COLS + _N_COMPLETION
    page._cancel()
    assert experiment.metadata_columns == ["animal_id"]  # restored
    assert page._table.columnCount() == _FIXED_COLS + 1 + _N_COMPLETION


def test_metadata_page_add_delete_edit_cancel_discards_all(qapp, tmp_path, monkeypatch):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1", metadata={"animal_id": "M001"})
    experiment.add_metadata_column("treatment")  # store: animal_id, treatment
    monkeypatch.setattr(pages.QInputDialog, "getText", lambda *a, **k: ("cohort", True))
    monkeypatch.setattr(
        pages.QInputDialog, "getItem", lambda *a, **k: ("treatment", True)
    )
    page = MetadataPage(experiment)
    page._enter_edit()
    page._table.item(0, 2).setText("EDITED")  # edit animal_id
    page._add_column()  # + cohort
    page._delete_column()  # - treatment
    page._cancel()
    # the store is fully unchanged, and the table matches it again
    assert experiment.metadata_columns == ["animal_id", "treatment"]
    assert experiment.recording_metadata["r1"] == {"animal_id": "M001", "treatment": ""}
    # recording_id + file + animal_id + treatment + completion columns
    assert page._table.columnCount() == _FIXED_COLS + 2 + _N_COMPLETION
    assert page._table.item(0, 2).text() == "M001"


def test_metadata_page_add_column_rejects_reserved_and_duplicate(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1")
    experiment.add_metadata_column("animal_id")
    warned = []
    monkeypatch.setattr(pages.QMessageBox, "warning", lambda *a, **k: warned.append(a))
    page = MetadataPage(experiment)
    page._enter_edit()
    monkeypatch.setattr(
        pages.QInputDialog, "getText", lambda *a, **k: ("recording_id", True)
    )
    page._add_column()  # reserved → rejected
    monkeypatch.setattr(
        pages.QInputDialog, "getText", lambda *a, **k: ("animal_id", True)
    )
    page._add_column()  # duplicate → rejected
    # unchanged: recording_id + file + animal_id + completion columns
    assert page._table.columnCount() == _FIXED_COLS + 1 + _N_COMPLETION
    assert len(warned) == 2


def test_metadata_page_delete_column_disabled_when_no_columns(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1")
    page = MetadataPage(experiment)
    page._enter_edit()
    assert not page._delete_col_btn.isEnabled()  # nothing to delete


def test_metadata_page_import_export_are_view_mode_only(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1")
    page = MetadataPage(experiment)
    assert not page._import_btn.isHidden() and not page._export_btn.isHidden()
    page._enter_edit()
    assert page._import_btn.isHidden() and page._export_btn.isHidden()
    assert not page._delete_col_btn.isHidden()  # edit-mode tool appears instead
    page._cancel()
    assert not page._import_btn.isHidden() and not page._export_btn.isHidden()


def test_recording_detail_builds_without_video(qapp, tmp_path):
    # the reframe: a metadata-only recording's detail page renders (no video/streams)
    from detail.gui.pages import RecordingDetailPage

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    page = RecordingDetailPage(
        experiment.get_recording("recording_001"), experiment=experiment
    )
    assert page is not None
    assert "animal_id" in page._meta_edits  # the inline metadata editor is present


def test_metadata_page_import_csv(qapp, tmp_path, monkeypatch):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("s1")
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("recording_id,treatment\ns1,drug_A\n", encoding="utf-8")
    monkeypatch.setattr(
        pages.QFileDialog, "getOpenFileName", lambda *a, **k: (str(csv_path), "")
    )
    monkeypatch.setattr(pages.QMessageBox, "information", lambda *a, **k: None)

    def _no_prompt(*_a, **_k):
        raise AssertionError("a sheet with a recognised key must not ask for one")

    monkeypatch.setattr(pages.QInputDialog, "getItem", _no_prompt)
    page = MetadataPage(experiment)
    page._import_csv()
    assert experiment.recording_metadata["s1"] == {"treatment": "drug_A"}
    # recording_id + file + treatment + completion columns, after rebuild
    assert page._table.columnCount() == _FIXED_COLS + 1 + _N_COMPLETION


def test_metadata_page_import_button_describes_every_key_it_accepts(qapp, tmp_path):
    """The hover is where a person learns which column links a row to a recording, so it
    is built from the importer's own list of headers rather than restating it (#330)."""
    from detail.storage import metadata as metadata_store

    page = MetadataPage(Experiment.create("p", tmp_path))
    tooltip = page._import_btn.toolTip()
    for key in ("recording_id", "name", *metadata_store.VIDEO_KEY_HEADERS):
        assert key in tooltip
    assert "extension optional" in tooltip and "asked" in tooltip


def _sheet_without_a_key(tmp_path, monkeypatch, pick):
    from detail.gui import pages

    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {"files": [{"index": 0, "origin_name": "/rig/m27_day3.mp4"}]}
    recording_store.write_recording(recording.dir, data)
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("Session Clip,dose\nm27_day3.mp4,0.5\n", encoding="utf-8")
    monkeypatch.setattr(
        pages.QFileDialog, "getOpenFileName", lambda *a, **k: (str(csv_path), "")
    )
    monkeypatch.setattr(pages.QMessageBox, "information", lambda *a, **k: None)
    asked = []

    def _get_item(_parent, _title, label, items, *_a, **_k):
        asked.append((label, list(items)))
        return pick

    monkeypatch.setattr(pages.QInputDialog, "getItem", _get_item)
    return experiment, recording, asked


def test_metadata_page_import_asks_which_column_holds_the_video_file(
    qapp, tmp_path, monkeypatch
):
    """No recording_id, name or file-name header: ask, offering the sheet's own columns,
    and key the import on the one picked (#330)."""
    experiment, recording, asked = _sheet_without_a_key(
        tmp_path, monkeypatch, ("Session Clip", True)
    )
    page = MetadataPage(experiment)
    page._import_csv()

    [(label, items)] = asked
    assert items == ["Session Clip", "dose"] and "video file name" in label
    assert experiment.recording_metadata[recording.id] == {"dose": "0.5"}
    assert "dose" in [
        page._table.horizontalHeaderItem(c).text()
        for c in range(page._table.columnCount())
    ]


def test_metadata_page_import_cancelled_at_the_column_prompt_applies_nothing(
    qapp, tmp_path, monkeypatch
):
    experiment, recording, asked = _sheet_without_a_key(
        tmp_path, monkeypatch, ("", False)
    )
    MetadataPage(experiment)._import_csv()
    assert len(asked) == 1
    assert experiment.recording_metadata[recording.id] == {}


# --------------------------------------------------------------------------- #
# The file column and the completion columns (§ "Recording information")
# --------------------------------------------------------------------------- #
def test_metadata_page_file_column_shows_base_name_full_path_in_tooltip(qapp, tmp_path):
    from detail.gui.qt import Qt

    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("r1")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {
        "files": [
            {"origin_name": "/data/rig1/m1_BL_t1_2026.mp4"},
            {"origin_name": "/data/rig1/m1_BL_t1_2026_part2.mp4"},
        ]
    }
    recording_store.write_recording(recording.dir, data)

    page = MetadataPage(experiment)
    item = page._table.item(0, 1)
    assert item.text() == "m1_BL_t1_2026.mp4  +1 more"
    assert item.toolTip() == (
        "/data/rig1/m1_BL_t1_2026.mp4\n/data/rig1/m1_BL_t1_2026_part2.mp4"
    )
    assert not item.flags() & Qt.ItemFlag.ItemIsEditable


def test_metadata_page_file_column_blank_with_no_video(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1")
    page = MetadataPage(experiment)
    assert page._table.item(0, 1).text() == ""


def test_metadata_page_completion_columns_are_unchecked_for_a_fresh_recording(
    qapp, tmp_path
):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1")
    page = MetadataPage(experiment)
    from detail.gui.qt import Qt

    start = page._completion_start()
    assert [c[0] for c in _COMPLETION_COLUMNS] == [
        "Field of view",
        "Distance calibration",
        "ROI",
        "POI",
        "Streams",
        "Detection",
        "Tracking",
        "Embedding",
    ]
    for i, (label, _fn) in enumerate(_COMPLETION_COLUMNS):
        item = page._table.item(0, start + i)
        assert item.checkState() == Qt.CheckState.Unchecked, label
        assert item.toolTip()  # says *why* it's unchecked, not just blank


def test_metadata_page_fov_and_spatial_completion_checks_once_defined(qapp, tmp_path):
    from detail.gui.qt import Qt

    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("r1")
    square = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    recording.define_arena("arena", [(0, square)])
    recording.define_roi("zone", [(0, square)])
    recording.define_point("lever", [(0, [(0.5, 0.5)])])

    page = MetadataPage(experiment)
    start = page._completion_start()
    labels = [label for label, _fn in _COMPLETION_COLUMNS]
    fov_item = page._table.item(0, start + labels.index("Field of view"))
    roi_item = page._table.item(0, start + labels.index("ROI"))
    poi_item = page._table.item(0, start + labels.index("POI"))
    assert fov_item.checkState() == Qt.CheckState.Checked
    assert "1 field of view set" in fov_item.toolTip()
    assert roi_item.checkState() == Qt.CheckState.Checked
    assert "zone" in roi_item.toolTip()
    assert poi_item.checkState() == Qt.CheckState.Checked
    assert "lever" in poi_item.toolTip()


def test_metadata_page_run_completion_requires_a_complete_run(qapp, tmp_path):
    from detail.gui.qt import Qt

    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("r1")
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [{"id": "det_01", "status": "failed"}]
    data["tracking_runs"] = [{"id": "track_01", "status": "complete"}]
    recording_store.write_recording(recording.dir, data)

    page = MetadataPage(experiment)
    start = page._completion_start()
    labels = [label for label, _fn in _COMPLETION_COLUMNS]
    det_item = page._table.item(0, start + labels.index("Detection"))
    track_item = page._table.item(0, start + labels.index("Tracking"))
    embed_item = page._table.item(0, start + labels.index("Embedding"))
    # one attempt, but it failed — not checked
    assert det_item.checkState() == Qt.CheckState.Unchecked
    assert "none complete" in det_item.toolTip()
    assert track_item.checkState() == Qt.CheckState.Checked
    assert "1/1" in track_item.toolTip()
    assert embed_item.checkState() == Qt.CheckState.Unchecked
    assert "No embedding runs" in embed_item.toolTip()


def test_metadata_page_completion_checkbox_cannot_be_toggled(qapp, tmp_path):
    """A completion box reports what the recording has; it is not a control.

    It used to be `ItemIsUserCheckable` with an `itemChanged` handler putting the click
    back. That works and reads badly: Qt toggles the box under the cursor and something
    invisible undoes it, so the app appears to disagree with you and says nothing about
    why. **The check indicator draws from the check state alone** — verified — so dropping
    the flag makes the cell genuinely inert, and the handler goes with it.

    Asserted as the property (the cell is not user-checkable) rather than as the old
    workaround's symptom (a toggle gets reverted). The symptom test passes either way; it
    was satisfied by the revert, so it could not tell an inert cell from a live one being
    corrected.
    """
    from detail.gui.qt import Qt

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1")
    page = MetadataPage(experiment)
    start = page._completion_start()
    item = page._table.item(0, start)
    assert not (item.flags() & Qt.ItemFlag.ItemIsEditable)
    assert not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable), (
        "the box is user-checkable, so Qt will toggle it on click and something has to "
        "undo that — make the cell inert instead"
    )
    assert item.checkState() == Qt.CheckState.Unchecked


def test_metadata_page_completion_columns_are_never_pasted_into(qapp, tmp_path):
    from detail.gui.qt import QGuiApplication, Qt

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1")
    page = MetadataPage(experiment)
    page._enter_edit()
    start = page._completion_start()
    QGuiApplication.clipboard().setText("X")
    page._table.setCurrentCell(0, start)
    page._paste()
    item = page._table.item(0, start)
    assert item.text() == ""
    assert item.checkState() == Qt.CheckState.Unchecked


# --------------------------------------------------------------------------- #
# Double-click a row → that recording (design note item D)
# --------------------------------------------------------------------------- #
def _row(row: int):
    import types

    return types.SimpleNamespace(row=lambda: row)


def test_metadata_page_double_click_opens_the_recording(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    experiment.create_recording("recording_002")
    opened = []
    page = MetadataPage(experiment, on_open=opened.append)
    page._open_row(_row(1))
    assert opened == ["recording_002"]


def test_metadata_page_double_click_does_nothing_in_edit_mode(qapp, tmp_path):
    """In edit mode a double-click means "edit this cell", not "leave the page" —
    the same line `_paste` already draws for clipboard paste."""
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    opened = []
    page = MetadataPage(experiment, on_open=opened.append)
    page._enter_edit()
    page._open_row(_row(0))
    assert opened == []


def test_metadata_page_double_click_is_a_no_op_without_on_open(qapp, tmp_path):
    """The callback is optional (most tests construct the page with none) — a
    double-click must not raise when there's nowhere to go."""
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    page = MetadataPage(experiment)  # no on_open
    page._open_row(_row(0))  # does not raise


def test_metadata_page_shows_the_pointing_hand_cursor_only_when_openable(
    qapp, tmp_path
):
    from detail.gui.qt import Qt

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    with_open = MetadataPage(experiment, on_open=lambda _rid: None)
    without_open = MetadataPage(experiment)
    assert (
        with_open._table.viewport().cursor().shape()
        == Qt.CursorShape.PointingHandCursor
    )
    assert (
        without_open._table.viewport().cursor().shape()
        != Qt.CursorShape.PointingHandCursor
    )


def test_metadata_page_hint_mentions_double_click_only_when_openable(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    with_open = MetadataPage(experiment, on_open=lambda _rid: None)
    without_open = MetadataPage(experiment)
    assert "Double-click a row" in with_open._hint.text()
    assert "Double-click a row" not in without_open._hint.text()
