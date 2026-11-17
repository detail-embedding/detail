"""Run tables navigate: recording run-history → experiment run, run → recording."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QStandardItem, QStandardItemModel  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.dataclasses import ModelReadiness  # noqa: E402
from detail.gui.pages import RunResults  # noqa: E402
from detail.gui.tables import stamp_row_id  # noqa: E402
from detail.storage import runs as runs_store  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

#: `QModelIndex` holds a raw pointer into its model's C++ side; a model built and
#: dropped inside `_row()` gets collected the moment that function returns (nothing else
#: refcounts it), leaving the index it handed back dangling — an access violation on the
#: very next `.model()` call, not a Python exception. Keep every model this makes alive
#: for the life of the test process; each one is a handful of bytes.
_KEEPALIVE: list = []


def _row(row: int, key=None):
    """A real ``QModelIndex`` for row ``row`` of a one-column model, with ``key``
    (default: ``row`` itself) stamped as that row's identity.

    A bare ``types.SimpleNamespace(row=lambda: row)`` was enough while a handler read
    only ``index.row()`` — it no longer is, now that ``pages.row_id`` reads identity off
    the row's own stamped data (``index.model()...``), the fix for a click surviving a
    sort. This builds the real shape instead of a stub that predates it.
    """
    model = QStandardItemModel(row + 1, 1)
    _KEEPALIVE.append(model)
    item = QStandardItem("x")
    stamp_row_id(item, row if key is None else key)
    model.setItem(row, 0, item)
    return model.index(row, 0)


def _recording_with_tracking(proj, sid, processing_run_id):
    recording = proj.create_recording(sid)
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"] = [
        {
            "id": "track_01",
            "status": "complete",
            "processing_run_id": processing_run_id,
            "detection_run_id": "det_01",
            "tracker_id": "anchor_tracker",
            "created_at": "2026-06-09T14:00:00Z",
        }
    ]
    recording_store.write_recording(recording.dir, data)
    return proj.get_recording(sid)


def test_recording_run_history_opens_experiment_run(qapp, tmp_path):
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = _recording_with_tracking(proj, "s1", "run_003")
    opened = []
    page = RecordingDetailPage(recording, on_open_run=opened.append)
    page._open_run_row(_row(0))
    assert opened == ["run_003"]  # navigates to the experiment run that produced it


def test_recording_run_history_is_pinned_to_the_bottom(qapp, tmp_path):
    # Run history moved below the ANALYSIS band (it only matters once runs exist).
    # Reverting the move puts it back above ANALYSIS and this ordering check fails.
    from PyQt6.QtWidgets import QLabel

    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    page = RecordingDetailPage(recording, experiment=proj)

    def _main_index_of(text):
        layout = page.layout()
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w is None:
                continue
            if isinstance(w, QLabel) and w.text() == text:
                return i
            if any(lbl.text() == text for lbl in w.findChildren(QLabel)):
                return i
        return -1

    analysis_index = _main_index_of("ANALYSIS")
    run_history_index = _main_index_of("Run history")
    assert analysis_index >= 0 and run_history_index >= 0
    assert run_history_index > analysis_index


def test_recording_run_history_opens_recording_run_without_experiment_run(
    qapp, tmp_path
):
    # A per-recording run (CLI/API; no experiment record) opens a recording-scoped result
    # view instead of jumping to a experiment run.
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = _recording_with_tracking(
        proj, "s1", None
    )  # ran directly, no experiment run
    opened_experiment, opened_recording = [], []
    page = RecordingDetailPage(
        recording,
        on_open_run=opened_experiment.append,
        on_open_recording_run=lambda *a: opened_recording.append(a),
    )
    page._open_run_row(_row(0))
    assert opened_experiment == []  # no experiment run to jump to
    assert opened_recording == [("s1", "tracking", "track_01")]


def test_add_video_part_dialog_wires_progress(qapp, tmp_path, monkeypatch):
    from detail.gui import pages
    from detail.gui.pages import AddVideoPartDialog
    from detail.gui.workers import StageProgressWorker

    class _SyncWorker(StageProgressWorker):
        def start(
            self,
        ):  # run the import inline so the test is deterministic (no thread)
            self.run()

    monkeypatch.setattr(pages, "StageProgressWorker", _SyncWorker)

    recording = Experiment.create("p", tmp_path).create_recording("r1")
    captured = {}

    def fake_add(path, progress=None, **kwargs):
        captured["path"] = path
        captured["progress"] = progress
        captured["kwargs"] = kwargs
        if progress is not None:
            progress("Copying", 0.5, "1 / 2 GB")  # exercise the tick path

    recording.add_video_part = fake_add  # instance override

    dialog = AddVideoPartDialog(recording)
    dialog._path.setText(str(tmp_path / "clip.mp4"))
    dialog._start()
    # the import ran off the UI thread (StageProgressWorker) with a progress callback
    # wired (long-op rule)
    assert captured["path"] == str(tmp_path / "clip.mp4")
    assert captured["progress"] is not None


def test_reference_controls_params(qapp):
    # Just a checkbox now — no location picker. On → storage=reference (the tool remembers
    # the video's folder itself, so no video_root is passed).
    from detail.gui.pages import _ReferenceControls

    ctrl = _ReferenceControls()
    assert ctrl.reference_params() is None  # off by default
    ctrl._on.setChecked(True)
    assert ctrl.reference_params() == {"storage": "reference"}


def test_add_video_part_dialog_reference_mode(qapp, tmp_path, monkeypatch):
    from detail.gui import pages
    from detail.gui.pages import AddVideoPartDialog
    from detail.gui.workers import StageProgressWorker

    class _SyncWorker(StageProgressWorker):
        def start(self):
            self.run()

    monkeypatch.setattr(pages, "StageProgressWorker", _SyncWorker)
    recording = Experiment.create("p", tmp_path).create_recording("r1")
    captured = {}
    recording.add_video_part = lambda path, progress=None, **kw: captured.update(kw)

    dialog = AddVideoPartDialog(recording)
    dialog._path.setText(str(tmp_path / "clip.mp4"))
    dialog._reference._on.setChecked(True)
    dialog._start()
    assert captured == {"storage": "reference"}  # no location picked — auto


def test_create_dialog_passes_reference_kwargs(qapp, tmp_path):
    from detail.gui.pages import AddRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    submitted = {}
    dialog = AddRecordingsDialog(proj, lambda specs, ds: submitted.update(ds=ds))
    dialog._append_row("s1", "")
    dialog._reference._on.setChecked(True)
    dialog._create()
    assert submitted["ds"] == {"storage": "reference"}


def test_part_location_shows_absolute_paths(qapp, tmp_path, monkeypatch):
    from detail.gui.pages import _part_location
    from detail.storage import video_roots

    copied = {"index": 0, "id": "part_001", "file": "video/part_001.mp4"}
    assert _part_location(tmp_path, copied) == str(tmp_path / "video/part_001.mp4")

    root = tmp_path / "nas"
    monkeypatch.setattr(
        video_roots, "video_root", lambda a: root if a == "nas" else None
    )
    ref = {"index": 0, "source": {"root": "nas", "rel": "day3/m.mp4"}}
    assert _part_location(tmp_path, ref) == f"{root / 'day3' / 'm.mp4'}  (referenced)"

    ref_ghost = {"index": 0, "source": {"root": "ghost", "rel": "m.mp4"}}
    assert "not registered" in _part_location(tmp_path, ref_ghost)


def test_relink_video_dialog_do_relink(qapp, tmp_path):
    from detail.gui.pages import RelinkVideoDialog
    from detail.storage import recording as recording_store

    recording = Experiment.create("p", tmp_path).create_recording("r1")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {
        "storage": "reference",
        "files": [
            {
                "index": 0,
                "id": "part_001",
                "source": {"root": "nas", "rel": "m.mp4"},
                "fingerprint": {"size": 1, "sha256_ends": "x"},
            }
        ],
    }
    recording_store.write_recording(recording.dir, data)
    captured = {}

    def relink(idx, path):
        captured.update(idx=idx, path=path)
        return "nas"

    recording.relink_video_part = relink

    dialog = RelinkVideoDialog(recording)
    assert dialog._part.count() == 1  # the one reference part
    dialog._path.setText(str(tmp_path / "moved.mp4"))
    assert dialog.do_relink() == "nas"  # the registered root reaches the page's notice
    assert captured == {"idx": 0, "path": str(tmp_path / "moved.mp4")}


def test_create_dialog_submits_specs_non_blocking(qapp, tmp_path):
    import cv2
    import numpy as np

    from detail.gui.pages import AddRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    submitted = {}
    dialog = AddRecordingsDialog(
        proj, lambda specs, ds: submitted.update(specs=specs, ds=ds)
    )
    vid = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
    vw.write(np.zeros((16, 16, 3), np.uint8))
    vw.release()
    dialog._append_row("with_video", str(vid))
    dialog._append_row("meta_only", "")  # video optional → metadata-only recording
    dialog._create()
    specs = submitted["specs"]
    # The dialog submits *names*; the id is minted when the shell is created (§11), so a
    # spec leaving the dialog has none yet.
    assert [s.name for s in specs] == ["with_video", "meta_only"]
    assert [s.id for s in specs] == [None, None]
    assert len(specs[0].video_parts) == 1
    assert specs[1].video_parts == []


def test_create_dialog_tags_metadata_at_import(qapp, tmp_path):
    from detail.gui.pages import AddRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    proj.add_metadata_column("treatment")
    submitted = {}
    dialog = AddRecordingsDialog(proj, lambda specs, ds: submitted.update(specs=specs))
    dialog._append_row("s1", "")
    dialog._table.item(0, 2).setText("drug_A")  # column 2 = the 'treatment' column
    dialog._create()
    assert submitted["specs"][0].metadata == {"treatment": "drug_A"}


def test_experiment_data_page_deletes_experiment(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.pages import ExperimentOverviewPage

    proj = Experiment.create("p", tmp_path)
    pdir = proj.dir
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    deleted = []
    page = ExperimentOverviewPage(
        proj, on_open_section=lambda k: None, on_deleted=lambda: deleted.append(True)
    )
    page._delete_experiment()
    assert deleted == [True]  # navigates home via on_deleted
    assert not pdir.exists()  # the experiment is gone


def test_recording_detail_deletes_recording(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    deleted = []
    page = RecordingDetailPage(
        recording, on_deleted=lambda: deleted.append(True), experiment=proj
    )
    page._delete_recording()
    assert deleted == [True]  # navigated back via on_deleted
    assert {s.id for s in proj.recordings} == set()  # recording is gone


def test_recording_detail_refuses_to_delete_a_recording_a_run_is_using(
    qapp, tmp_path, monkeypatch
):
    """Issue #28, finding 6: the refusal reaches the user as a warning naming the run, and
    the page stays — an uncaught ``ValueError`` in a Qt slot would abort the app."""
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.pages import RecordingDetailPage
    from detail.storage import runs as runs_store

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["s1"],
            "config": {},
            "per_recording": {"s1": {"run_id": "det_01", "status": "running"}},
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda parent, title, text: warnings.append(text)
    )
    deleted = []
    page = RecordingDetailPage(
        recording, on_deleted=lambda: deleted.append(True), experiment=proj
    )
    page._delete_recording()
    assert deleted == []
    assert len(warnings) == 1 and "run_001" in warnings[0]
    assert {s.id for s in proj.recordings} == {"s1"}


def _ticked(dialog):
    """The definition names currently ticked in a picker."""
    from detail.gui.pages import _checked_movables

    return [m.name for m in _checked_movables(dialog._items, dialog._movables)]


def _row_for_name(dialog, name):
    from PyQt6.QtCore import Qt

    for row in range(dialog._items.count()):
        i = dialog._items.item(row).data(Qt.ItemDataRole.UserRole)
        if i is not None and dialog._movables[i].name == name:
            return dialog._items.item(row)
    raise AssertionError(f"no row for {name!r}")


def _square_kf():
    return [(0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]


def _scene_button_texts(page):
    from PyQt6.QtWidgets import QPushButton

    texts = []
    for w in page._spatial_widgets:
        if isinstance(w, QPushButton):
            texts.append(w.text())
        texts += [b.text() for b in w.findChildren(QPushButton)]
    return texts


def _scene_label_texts(page):
    from PyQt6.QtWidgets import QLabel

    texts = []
    for w in page._spatial_widgets:
        if isinstance(w, QLabel):
            texts.append(w.text())
        texts += [lbl.text() for lbl in w.findChildren(QLabel)]
    return texts


def test_recording_detail_summarises_every_spatial_kind(qapp, tmp_path):
    """One section prints what four used to, including the kinds with nothing in them.

    The summary is the half of the merge that has to survive it: the page was collapsed
    to a single button, so if a kind stopped being *named* here there would be no
    remaining surface saying the recording has none of it.
    """
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    recording.define_arena("first", _square_kf())
    recording.define_arena("second", _square_kf())
    recording.define_roi("zone", _square_kf())

    page = RecordingDetailPage(recording, experiment=proj)
    labels = " | ".join(_scene_label_texts(page))
    assert "Field of view (2)" in labels
    assert "first" in labels and "second" in labels
    assert "Regions of interest (1)" in labels and "zone" in labels
    # The kinds with nothing defined still get a line saying so.
    assert "Points of interest (0)" in labels
    assert "Distance calibration" in labels
    assert labels.count("(none yet)") == 2


def test_locate_button_enables_only_when_a_referenced_video_is_missing(
    qapp, tmp_path, monkeypatch
):
    import cv2
    import numpy as np

    from detail.gui.pages import RecordingDetailPage
    from detail.storage import locations
    from detail.storage.video import FrameCursor
    from tests.file_handles import file_is_held
    from tests.gui.conftest import drain_busy_calls

    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    src = tmp_path / "external" / "m.mp4"
    src.parent.mkdir()
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    assert vw.isOpened()
    for i in range(5):
        vw.write(np.full((24, 32, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()
    recording.add_video_part(src, storage="reference")  # auto-location, no picker

    # The page decodes frame 0 for its geometry preview, and its reader lets go of the file
    # once idle; shortened so the user's "a moment later" is a moment here too.
    monkeypatch.setattr(FrameCursor, "IDLE_CLOSE_S", 0.05)
    recording = proj.get_recording("s1")
    page = RecordingDetailPage(recording, experiment=proj)
    assert not page._locate_btn.isHidden()  # shown for a reference recording

    # Present video → nothing to locate → the button is greyed (the off-thread scan render).
    page._on_video(recording.verify_videos())
    assert not page._locate_btn.isEnabled()

    # The user moves the external file with the page still open. Asked of the OS rather than
    # left to `unlink`, which succeeds on an open file on Linux and so cannot fail there.
    drain_busy_calls(qapp)
    assert drain_until(qapp, lambda: not file_is_held(src)), (
        "the open recording page still holds its video, so on Windows the user cannot move "
        "the file that Locate exists to find"
    )
    src.unlink()
    page._on_video(recording.verify_videos())
    assert page._locate_btn.isEnabled()


def test_recording_detail_spatial_opens_one_workspace(qapp, tmp_path):
    """Twenty spatial buttons became one way in.

    Counted rather than spot-checked, because the failure this guards is the one the merge
    was for: a second entry point growing back beside the first.
    """
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    recording.define_arena("main", _square_kf())
    recording.define_roi("zone", _square_kf())

    page = RecordingDetailPage(recording, experiment=proj)
    texts = _scene_button_texts(page)
    assert texts.count("Edit scene geometry…") == 1
    # Editing lives behind that one button — no per-kind editor on the page any more.
    for gone in (
        "Define field of view",
        "Set distance calibration",
        "Edit",
        "Delete",
        "Manage regions of interest",
        "Manage points of interest",
    ):
        assert gone not in texts, f"{gone!r} is still on the page"
    # The carry-over actions moved inside the editor, beside the thing they carry.
    for moved in (
        "Copy to other recordings…",
        "Export spatial setup (CSV)",
        "Import regions and points",
    ):
        assert moved not in texts


def test_recording_detail_spatial_row_says_who_reads_each_kind(qapp, tmp_path):
    """The DATA PIPELINE / ANALYSIS split said which kinds change what gets tracked.

    Merging the sections would have thrown that away, so it moved onto the row — it is a
    fact about the kind, not about which heading the widget landed under.
    """
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    page = RecordingDetailPage(recording, experiment=proj)
    labels = _scene_label_texts(page)
    assert "— used by detection & tracking" in labels  # the field of view
    assert labels.count("— analysis only") == 3  # calibration, regions, points


def test_recording_detail_spatial_workspace_needs_a_video(qapp, tmp_path):
    """Drawing needs a frame, so the one button says why it is off rather than going dead.

    The file-based actions stay reachable: they carry normalized polygons and need no
    video at all, which is the whole reason a setup can be copied onto a recording before
    its footage is imported.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    page = RecordingDetailPage(recording, experiment=proj)
    by_text = {b.text(): b for b in page.findChildren(QPushButton)}
    assert not by_text["Edit scene geometry…"].isEnabled()
    assert by_text["Edit scene geometry…"].toolTip() == "Import a video part first"


def test_recording_detail_button_rows_share_a_left_edge(qapp, tmp_path):
    # Putting two buttons in a row means wrapping them in a layout, and a wrapper keeps
    # Qt's default margins unless told otherwise — which indents that row past every other
    # button and label in the page. Assert the left edges line up rather than the margins,
    # so any future wrapper is covered too.
    #
    # "Edit scene geometry…" is deliberately not in this set: it now sits centered under
    # the geometry preview rather than on the left edge with these — see
    # `test_recording_detail_edit_scene_geometry_centered_under_preview`.
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    page = RecordingDetailPage(recording, experiment=proj)
    page.show()
    qapp.processEvents()
    by_text = {b.text(): b for b in page.findChildren(QPushButton)}
    left_edges = {
        text: by_text[text].mapTo(page, by_text[text].rect().topLeft()).x()
        for text in ("Add video part", "Add stream")
    }
    assert len(set(left_edges.values())) == 1, left_edges


def test_recording_detail_edit_scene_geometry_centered_under_preview(qapp, tmp_path):
    """The button moved off the shared left edge on purpose: it now sits centered
    beneath the geometry preview, not on the row `Add video part` / `Add stream` share.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.arena import GeometryPreview
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    page = RecordingDetailPage(recording, experiment=proj)
    page.show()
    qapp.processEvents()
    by_text = {b.text(): b for b in page.findChildren(QPushButton)}
    button = by_text["Edit scene geometry…"]
    preview = page.findChild(GeometryPreview)
    assert preview is not None
    button_center = button.mapTo(page, button.rect().center()).x()
    preview_center = preview.mapTo(page, preview.rect().center()).x()
    assert abs(button_center - preview_center) <= 2, (button_center, preview_center)


def test_recording_detail_preview_follows_an_added_video_part(qapp, tmp_path):
    """Through the page's own Add video part path: the preview leaves its "No video
    imported" placeholder and the editor button enables."""
    from unittest import mock

    from PyQt6.QtWidgets import QDialog, QPushButton

    from detail.gui import pages
    from detail.gui.arena import GeometryPreview
    from tests.gui.test_arena import _video_recording

    video = next(_video_recording(tmp_path / "src").dir.glob("video/part_*.mp4"))
    proj = Experiment.create("p", tmp_path / "dst")
    recording = proj.create_recording("s1")
    page = pages.RecordingDetailPage(recording, experiment=proj)
    page.show()
    qapp.processEvents()

    class _AddsAPart:
        def __init__(self, rec, parent=None):
            self._rec = rec

        def exec(self):
            self._rec.add_video_part(video)
            return QDialog.DialogCode.Accepted

    with mock.patch.object(pages, "AddVideoPartDialog", _AddsAPart):
        page._add_part()

    def _current_preview():
        live = [p for p in page.findChildren(GeometryPreview) if p.isVisible()]
        return live[-1] if live else None

    assert drain_until(
        qapp, lambda: _current_preview() is not None and _current_preview()._openable
    )
    button = next(
        b
        for b in page.findChildren(QPushButton)
        if b.text() == "Edit scene geometry…" and b.isVisible()
    )
    assert button.isEnabled()


def test_recording_detail_calibration_summary_reads_its_value(qapp, tmp_path):
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {"resolution": [1920, 1080], "files": [], "storage": "default"}
    recording_store.write_recording(recording.dir, data)

    page = RecordingDetailPage(recording, experiment=proj)
    assert "(none yet)" in _scene_label_texts(page)

    recording.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    page._rebuild_scene_geometry()
    labels = " | ".join(_scene_label_texts(page))
    assert "4.8 px/mm" in labels and "200 mm reference" in labels


def test_recording_detail_says_a_calibration_no_longer_fits_its_video(qapp, tmp_path):
    """A video replaced by one of another shape leaves a calibration that gives no scale.

    The page used to print whatever number the width ratio produced; reading the scale
    now raises, so the page has to ask first or it does not build at all.
    """
    from detail.gui.pages import RecordingDetailPage
    from detail.gui.pages import _calibration_completion

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {"resolution": [1920, 1080], "files": [], "storage": "default"}
    recording_store.write_recording(recording.dir, data)
    recording.set_pixels_per_mm(5.0)
    data = recording_store.read_recording(recording.dir)
    data["video"]["resolution"] = [1280, 1024]
    recording_store.write_recording(recording.dir, data)

    page = RecordingDetailPage(recording, experiment=proj)
    labels = " | ".join(_scene_label_texts(page))
    assert "unusable" in labels and "1920x1080" in labels
    done, text = _calibration_completion(recording)
    assert not done and "different shape" in text


def test_recording_detail_spatial_summary_shows_an_active_window(qapp, tmp_path):
    """A region scoped to part of the video says so on the page.

    It is the one property of a definition that no picture can show — the preview draws
    frame 0 — so if the page does not print it, nothing does.
    """
    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    recording.define_roi("nest", _square_kf())
    recording.define_roi("post-injection", _square_kf(), active=(5000, 9000))

    page = RecordingDetailPage(recording, experiment=proj)
    labels = " | ".join(_scene_label_texts(page))
    assert "post-injection · 4 verts · frames 5000–9000" in labels
    assert "nest · 4 verts · whole video" in labels


def test_recording_detail_spatial_summary_reads_the_record_once(qapp, tmp_path):
    """The four-kind summary costs one record read, not one per kind.

    Every `Recording` property is its own mtime-cached read and therefore its own `stat`.
    The section prints eight facts off one `recording.yaml`, so without the pin it is
    eight round trips per page build — invisible on a local disk, a frozen window on a
    share. Counted in syscalls, because a wall clock here measures nothing.
    """
    import os

    from detail.gui.pages import RecordingDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    recording.define_arena("main", _square_kf())
    recording.define_roi("zone", _square_kf())
    page = RecordingDetailPage(recording, experiment=proj)

    real_stat = os.stat
    calls = {"n": 0}

    def spy(*a, **k):
        calls["n"] += 1
        return real_stat(*a, **k)

    os.stat = spy
    try:
        page._rebuild_scene_geometry()
    finally:
        os.stat = real_stat

    # Eight properties behind one pinned read. The number is the point: revert the pin in
    # `_rebuild_scene_geometry` and this climbs with the number of kinds printed.
    assert calls["n"] <= 4, f"{calls['n']} stat calls to summarise one recording"


def test_import_spatial_dialog_source_options(qapp, tmp_path):
    from detail.gui.pages import _ImportSpatialDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    proj.create_recording("s2")

    dialog = _ImportSpatialDialog(r1, proj)
    assert dialog._from_rec.isEnabled() and dialog._from_rec.isChecked()

    solo_proj = Experiment.create("solo", tmp_path / "solo")
    solo = _ImportSpatialDialog(solo_proj.create_recording("only"), solo_proj)
    # Nothing to pull from, so the recording option is off and the CSV one is chosen.
    assert not solo._from_rec.isEnabled()
    assert solo._from_csv.isChecked()


def _give_video_filename(recording, filename: str) -> None:
    """Seed just enough of `video.files` for `video_sources` to name a file, without
    writing an actual video — `origin_name` is the pre-fingerprint fallback path."""
    data = recording_store.read_recording(recording.dir)
    data["video"] = {
        "storage": "default",
        "resolution": [64, 48],
        "files": [{"index": 0, "origin_name": filename}],
    }
    recording_store.write_recording(recording.dir, data)


def test_recording_pickers_lead_with_the_name_that_is_unique(qapp, tmp_path):
    """A picker has to tell two recordings apart, and only the name is guaranteed to.

    This led with the video's file name, on the reasoning that it is what a person
    recognises — true until a rig writes every session to the same file name, or several
    recordings reference one clip. Then the picker offers eight rows reading
    ``rig_camera.mp4`` and the choice cannot be made at all. Names are unique by
    construction (§11); the file name comes along when it adds something.
    """
    from detail.gui.pages import _recording_display_name

    proj = Experiment.create("p", tmp_path)
    a = proj.create_recording("s1")
    b = proj.create_recording("s2")
    _give_video_filename(a, "rig_camera.mp4")
    _give_video_filename(b, "rig_camera.mp4")  # one clip, two recordings

    labels = [_recording_display_name(r) for r in proj.recordings]
    assert len(set(labels)) == 2, f"the picker cannot tell them apart: {labels}"
    assert all(r.name in label for r, label in zip(proj.recordings, labels))
    assert all("rig_camera.mp4" in label for label in labels)

    bare = proj.create_recording("s3")  # no video yet
    assert _recording_display_name(bare) == bare.name


def test_import_dialog_lists_definitions_not_categories(qapp, tmp_path):
    """You pick the regions you want, not "regions".

    A category tick is a bet — you tick "Regions of interest" and learn afterwards which
    of them the destination refused. A definition tick is a specific thing, so whatever
    cannot come across is disabled with its reason before the click.
    """
    from detail.gui.pages import _ImportSpatialDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    src = proj.create_recording("s2")
    src.define_arena("rig", _square_kf())
    src.define_roi("nest", _square_kf())
    src.define_roi("feeder", _square_kf())

    dialog = _ImportSpatialDialog(r1, proj)
    assert sorted(_ticked(dialog)) == ["feeder", "nest", "rig"]
    _row_for_name(dialog, "nest").setCheckState(Qt.CheckState.Unchecked)
    assert sorted(_ticked(dialog)) == ["feeder", "rig"]

    dialog.do_import()
    assert sorted(x["name"] for x in r1.rois) == ["feeder"]
    assert [a["name"] for a in r1.arenas] == ["rig"]


def test_import_dialog_disables_a_definition_that_cannot_land(qapp, tmp_path):
    """Two reasons a row is dead, and both are shown rather than discovered afterwards:
    the name is taken here, or it carries an active window and cannot cross at all."""
    from detail.gui.pages import _ImportSpatialDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_roi("nest", _square_kf())
    r1.define_arena("its own", _square_kf())
    src = proj.create_recording("s2")
    src.define_arena("rig", _square_kf())
    src.define_roi("nest", _square_kf())
    src.define_roi("feeder", _square_kf())
    src.define_roi("timed", _square_kf(), active=(500, 900))

    dialog = _ImportSpatialDialog(r1, proj)
    assert _ticked(dialog) == ["feeder"]

    taken = _row_for_name(dialog, "nest")
    assert not (taken.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "already has one with this name" in taken.toolTip()

    windowed = _row_for_name(dialog, "timed")
    assert not (windowed.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "active frame window" in windowed.toolTip()

    fov = _row_for_name(dialog, "rig")
    assert not (fov.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "already has a field of view" in fov.toolTip()


def _square_kf():
    return [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])]


def test_copy_dialog_lists_this_recordings_definitions(qapp, tmp_path):
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    r1.define_roi("nest", _square_kf())
    proj.create_recording("s2")

    dialog = CopyToRecordingsDialog(r1, proj)
    assert sorted(_ticked(dialog)) == ["nest", "rig"]
    assert CopyToRecordingsDialog.has_anything_to_copy(r1)
    assert not CopyToRecordingsDialog.has_anything_to_copy(proj.get_recording("s2"))


def test_copy_dialog_lists_other_recordings_checked_by_default(qapp, tmp_path):
    from PyQt6.QtCore import Qt

    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    proj.create_recording("s2")
    proj.create_recording("s3")

    dialog = CopyToRecordingsDialog(r1, proj)
    assert dialog._targets.count() == 2  # every other recording, not the source
    names = {dialog._targets.item(i).text() for i in range(2)}
    assert names == {"s2", "s3"}
    for i in range(2):
        assert dialog._targets.item(i).checkState() == Qt.CheckState.Checked
    assert set(dialog._checked_target_ids()) == {
        proj.get_recording("s2").id,
        proj.get_recording("s3").id,
    }
    assert dialog._ok.text() == "Copy to 2"


def test_copy_dialog_uncheck_all_clears_every_target(qapp, tmp_path):
    from PyQt6.QtCore import Qt

    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    proj.create_recording("s2")
    proj.create_recording("s3")

    dialog = CopyToRecordingsDialog(r1, proj)
    assert dialog._checked_target_ids() != []

    dialog._all.setChecked(False)

    assert dialog._checked_target_ids() == []
    for i in range(dialog._targets.count()):
        assert dialog._targets.item(i).checkState() == Qt.CheckState.Unchecked
    assert dialog._ok.text() == "Copy"  # back to zero, same as no categories checked


def test_unticking_a_definition_keeps_the_targets_the_user_chose(qapp, tmp_path):
    """Narrowing the batch survives changing your mind about *what* to copy.

    The dialog rebuilds every target row when a definition is ticked, because a row's
    blocked state depends on what is ticked. Rebuilding the *check* state along with it
    silently rearmed every target somebody had turned off — on an action that writes to
    each one.
    """
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    r1.set_rois([{"name": "zone", "keyframes": _square_kf()}])
    for rid in ("s2", "s3", "s4"):
        proj.create_recording(rid)

    dialog = CopyToRecordingsDialog(r1, proj)
    for i in range(dialog._targets.count()):
        item = dialog._targets.item(i)
        if item.data(Qt.ItemDataRole.UserRole) != "s2":
            item.setCheckState(Qt.CheckState.Unchecked)
    assert dialog._checked_target_ids() == ["s2"]

    _row_for_name(dialog, "rig").setCheckState(Qt.CheckState.Unchecked)
    assert dialog._checked_target_ids() == ["s2"]
    _row_for_name(dialog, "rig").setCheckState(Qt.CheckState.Checked)
    assert dialog._checked_target_ids() == ["s2"]


def test_copy_dialog_shows_an_occupied_target_before_the_click(qapp, tmp_path):
    """A recording that already has one keeps it, and the row says so up front.

    The fixture needs an *open* target as well as the occupied one: with every target
    occupied the definition itself is withdrawn, so a single-target fixture would test the
    row-level gate and never reach the row it is about.
    """
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    proj.create_recording("s2").define_arena("its own", _square_kf())
    proj.create_recording("s3")

    dialog = CopyToRecordingsDialog(r1, proj)
    assert _ticked(dialog) == ["rig"]  # s3 can still take it
    occupied = next(
        dialog._targets.item(i)
        for i in range(dialog._targets.count())
        if dialog._targets.item(i).data(Qt.ItemDataRole.UserRole) == "s2"
    )
    assert not (occupied.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "already has: Field of view" in occupied.text()
    assert dialog._checked_target_ids() == ["s3"]


def test_copy_dialog_withdraws_a_definition_nothing_can_take(qapp, tmp_path):
    """Every chosen recording already has one, so ticking it could only produce refusals."""
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    r1.define_roi("nest", _square_kf())
    for rid in ("s2", "s3"):
        other = proj.create_recording(rid)
        other.define_arena("its own", _square_kf())
        other.define_roi("nest", _square_kf())

    dialog = CopyToRecordingsDialog(r1, proj)
    assert _ticked(dialog) == []
    fov = _row_for_name(dialog, "rig")
    assert not (fov.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "every chosen recording already has one" in fov.toolTip()
    roi = _row_for_name(dialog, "nest")
    assert not (roi.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "already has one with this name" in roi.toolTip()
    assert not dialog._ok.isEnabled()


def test_a_partly_occupied_target_still_receives_the_rest(qapp, tmp_path):
    """Blocking is per definition, not per recording."""
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    r1.set_rois([{"name": "zone", "keyframes": _square_kf()}])
    proj.create_recording("s2").define_arena("its own", _square_kf())
    proj.create_recording("s3")

    dialog = CopyToRecordingsDialog(r1, proj)
    item = next(
        dialog._targets.item(i)
        for i in range(dialog._targets.count())
        if dialog._targets.item(i).data(Qt.ItemDataRole.UserRole) == "s2"
    )
    assert item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert "already has: Field of view" in item.text()

    dialog.run(dialog.plan(), None)
    target = proj.get_recording("s2")
    assert [a["name"] for a in target.arenas] == ["its own"]  # kept
    assert [r["name"] for r in target.rois] == ["zone"]  # received


def test_copy_dialog_marks_motion_and_windows_on_the_row(qapp, tmp_path):
    """What the trip changes is said on the definition it changes, not in a notice block
    the row has no connection to."""
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("panning rig", _square_kf() + [(900, _square_kf()[0][1])])
    r1.define_roi("nest", _square_kf())
    r1.define_roi("timed zone", _square_kf(), active=(5000, 9000))
    proj.create_recording("s2")

    dialog = CopyToRecordingsDialog(r1, proj)
    assert "camera motion not copied" in _row_for_name(dialog, "panning rig").text()
    windowed = _row_for_name(dialog, "timed zone")
    assert not (windowed.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "active frame window" in windowed.toolTip()
    assert sorted(_ticked(dialog)) == ["nest", "panning rig"]


def test_copy_dialog_run_copies_checked_categories_to_checked_targets(qapp, tmp_path):
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    r1.set_rois([{"name": "zone", "keyframes": _square_kf()}])
    proj.create_recording("s2")
    proj.create_recording("s3")

    dialog = CopyToRecordingsDialog(r1, proj)
    seen = []
    notes = dialog.run(dialog.plan(), lambda stage, f, detail: seen.append(stage))

    assert notes == []
    for rid in ("s2", "s3"):
        rec = proj.get_recording(rid)
        assert [a["name"] for a in rec.arenas] == ["rig"]
        assert [r["name"] for r in rec.rois] == ["zone"]
    assert set(seen) == {"Field of view", "Regions of interest"}


def test_run_touches_no_widget_so_a_worker_can_call_it(qapp, tmp_path):
    """`run` executes off the UI thread, so every Qt read has to happen in `plan`.

    Asserted by *deleting the widgets* between the two calls rather than by threading: a
    worker touching a dialog the page has since destroyed is the failure this guards, and
    a plan that still reaches back into the list raises here.
    """
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    r1 = proj.create_recording("s1")
    r1.define_arena("rig", _square_kf())
    proj.create_recording("s2")

    dialog = CopyToRecordingsDialog(r1, proj)
    plan = dialog.plan()
    assert plan == ({"fov": ["rig"]}, ["s2"])

    dialog._items.clear()
    dialog._targets.clear()
    dialog._movables = []

    assert dialog.run(plan, None) == []
    assert [a["name"] for a in proj.get_recording("s2").arenas] == ["rig"]


def test_run_detail_recording_row_opens_recording(qapp, tmp_path):
    from detail.gui.pages import RunDetailPage

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("s1")
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "tracking",
            "recording_ids": ["s1"],
            "config": {},
            "status": "failed",  # skip the result view
            "per_recording": {"s1": {"run_id": "track_01", "status": "failed"}},
            "created_at": "2026-06-09T14:00:00Z",
        },
    )
    opened = []
    page = RunDetailPage(proj, "run_001", on_open_recording=opened.append)
    # Double-click still opens the recording, now from the Results list that replaced the
    # separate Per-recording table.
    results = page.findChild(RunResults)
    results._open_recording_row(_row(0, key="s1"))
    assert opened == ["s1"]


def test_the_results_list_scrolls_past_five_rows(qapp, tmp_path, monkeypatch):
    """Reported twice: the list "shows too many instances", pushing the video player or PCA
    scatter under it off a full-screen window. It took the display's own row budget — 20 to 40
    rows — so it is asked here on a display with room for 40, where that budget is what would
    show. Five, because the results now open underneath it rather than in a separate view.
    """
    from detail.gui import tables
    from detail.gui.pages import IndexTable, RunDetailPage

    monkeypatch.setattr(tables, "max_visible_rows", lambda _row_height: 40)
    proj = Experiment.create("p", tmp_path)
    ids = [f"s{i:02d}" for i in range(25)]
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "tracking",
            "recording_ids": ids,
            "config": {},
            "status": "failed",  # skip the result view
            "per_recording": {
                sid: {"run_id": "track_01", "status": "failed"} for sid in ids
            },
            "created_at": "2026-06-09T14:00:00Z",
        },
    )
    page = RunDetailPage(proj, "run_001")
    page.resize(1400, 1600)
    page.show()
    qapp.processEvents()
    try:
        (table,) = [
            t
            for t in page.findChildren(IndexTable)
            if t.view.model().headerData(0, Qt.Orientation.Horizontal) == "Recording"
        ]
        view = table.view
        chrome = view.horizontalHeader().height() + 2 * view.frameWidth()
        rows = (view.height() - chrome) / view.verticalHeader().defaultSectionSize()
        assert rows == pytest.approx(RunResults.ROWS, abs=0.6)
        assert view.verticalScrollBar().maximum() > 0, "the other 20 are not reachable"
    finally:
        page.close()


def test_model_detail_page_shows_recorded_error(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel

    from detail.gui.pages import ModelDetailPage
    from detail.storage import models as mstore

    experiment = Experiment.create("p", tmp_path)
    mdir = mstore.model_dir(experiment.dir, "model_001")
    mdir.mkdir(parents=True)
    mstore.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "status": "failed",
            "error": "RuntimeError: boom",
        },
    )
    page = ModelDetailPage(experiment, "model_001")
    texts = [w.text() for w in page.findChildren(QLabel)]
    assert any("RuntimeError: boom" in t for t in texts)


def test_recordings_page_shows_queued_source_and_cancel(qapp, tmp_path):
    from detail.gui.pages import ExperimentRecordingsPage
    from detail.storage import recording as recording_store

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("s1")
    recording_store.set_import_status(experiment.get_recording("s1").dir, "queued")
    page = ExperimentRecordingsPage(
        experiment,
        on_open=lambda sid: None,
        pending_source=lambda sid: "clip.mp4" if sid == "s1" else None,
        cancel_imports=lambda: None,
    )
    # The rows are read off the UI thread now, so wait for them rather than for the table.
    assert drain_until(qapp, lambda: page._table.view.model().rowCount() == 1)
    model = page._table.view.model()  # columns: Recording, Status, Source, ...
    assert model.index(0, 1).data() == "queued"
    # The pending source is the import queue's in-memory answer, overlaid on the UI thread
    # over what disk says — the worker cannot see the queue.
    assert model.index(0, 2).data() == "clip.mp4"
    assert not page._cancel_btn.isHidden()  # shown (cancel wired)
    assert page._cancel_btn.isEnabled()  # a queued recording is active


def test_recordings_page_always_shows_source(qapp, tmp_path):
    """It used to be dropped on a fully-imported experiment, because the only source we
    had was the import queue's pending path (TRIAL_NOTES #3). Every part now records
    where its footage came from, so the column always has an answer — blank only for a
    recording with no video."""
    from detail.gui.pages import ExperimentRecordingsPage

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("s1")  # ready — no pending import
    page = ExperimentRecordingsPage(experiment, on_open=lambda sid: None)
    assert drain_until(qapp, lambda: page._table.view.model().rowCount() == 1)
    model = page._table.view.model()
    headers = [model.horizontalHeaderItem(c).text() for c in range(model.columnCount())]
    assert headers == [
        "Recording",
        "Status",
        "Source",
        "Parts",
        "Streams",
        "Field of view",
        "Detections",
        "",  # trailing "›" column — the navigable-row cue
    ]
    assert model.index(0, 2).data() == ""  # no video → nothing to say


def test_global_models_install_status_shows_file_and_mb(qapp):
    # #18: the GUI install surface renders the current file + MB (was bar-only,
    # while the CLI already showed them).
    from detail.gui.pages import GlobalModelsPage

    page = GlobalModelsPage()
    page._on_install_progress(5_000_000, 10_000_000, "model.safetensors")
    text = page._install_status.text()
    assert "model.safetensors" in text and "MB" in text and "50%" in text


def test_global_models_lists_and_installs_one_row_per_variant(qapp, monkeypatch):
    """Install state is per size — each DINOv3 variant is its own repo and its own
    download. With one row per model the tab reported the default variant's status
    for every size and Install always fetched the default, so a run configured for
    another size could never be made runnable from the GUI."""
    from detail.gui import pages as gui_pages
    from detail.gui.pages import GlobalModelsPage

    page = GlobalModelsPage()
    model = page._table.model()
    id_col = GlobalModelsPage._COLUMNS.index("Id")
    variant_col = GlobalModelsPage._COLUMNS.index("Size / variant")
    rows = [
        (model.index(r, id_col).data(), model.index(r, variant_col).data())
        for r in range(model.rowCount())
    ]
    assert ("dinov3_convnext_base", "convnext_tiny") in rows
    assert ("dinov3_convnext_base", "convnext_base") in rows
    assert ("sam3", "—") in rows  # no size axis → a single row

    launched = {}

    def fake_worker(model_id, variant, *, receiver):
        launched["args"] = (model_id, variant)
        return _NoopWorker()

    monkeypatch.setattr(gui_pages, "InstallModelWorker", fake_worker)
    page._table.selectRow(rows.index(("dinov3_convnext_base", "convnext_tiny")))
    page._install()
    assert launched["args"] == ("dinov3_convnext_base", "convnext_tiny")


class _NoopWorker:
    """Stand-in for InstallModelWorker: connectable signals, never starts a thread."""

    def __init__(self):
        self.progress = _NoopSignal()
        self.finished_install = _NoopSignal()

    def start(self):
        pass


class _NoopSignal:
    def connect(self, _slot):
        pass


def test_global_models_scan_failure_shows_unknown_not_a_stuck_checking(qapp):
    # If the off-thread is_installed() scan raises, the Status column must read "unknown"
    # rather than hang on "checking…" forever, and Install stays off (we can't verify).
    from detail.gui.pages import GlobalModelsPage

    page = GlobalModelsPage()
    page._on_scan_failed(RuntimeError("cache broke"))
    model = page._table.model()
    status_col = GlobalModelsPage._COLUMNS.index("Status")
    statuses = [model.index(r, status_col).data() for r in range(model.rowCount())]
    assert statuses and all(s == "unknown" for s in statuses)
    assert not page._install_btn.isEnabled()


def test_check_access_routes_not_installed_to_install_not_a_warning(qapp, monkeypatch):
    """A granted gated model that isn't downloaded is the normal pre-install state,
    so Check access must read as information pointing at Install — not the warning
    box every other not-ready reason gets."""
    from detail.gui import pages as gui_pages
    from detail.gui.pages import GlobalModelsPage

    seen = {}
    for kind in ("information", "warning"):
        monkeypatch.setattr(
            gui_pages.QMessageBox,
            kind,
            lambda *a, _k=kind, **k: seen.update(kind=_k, text=a[2]),
        )
    page = GlobalModelsPage()
    page._on_readiness(
        ModelReadiness(
            ready=False,
            reason="not_installed",
            message="facebook/sam3 is not downloaded.",
            action_url=None,
            repo_id="facebook/sam3",
        )
    )
    assert seen["kind"] == "information"
    assert "Install" in seen["text"]


def test_check_access_still_warns_on_a_real_access_problem(qapp, monkeypatch):
    from detail.gui import pages as gui_pages
    from detail.gui.pages import GlobalModelsPage

    seen = {}
    for kind in ("information", "warning"):
        monkeypatch.setattr(
            gui_pages.QMessageBox,
            kind,
            lambda *a, _k=kind, **k: seen.update(kind=_k, text=a[2]),
        )
    page = GlobalModelsPage()
    page._on_readiness(
        ModelReadiness(
            ready=False,
            reason="no_access",
            message="Access to facebook/sam3 is forbidden.",
            action_url="https://huggingface.co/facebook/sam3",
            repo_id="facebook/sam3",
        )
    )
    assert seen["kind"] == "warning"
    assert "huggingface.co/facebook/sam3" in seen["text"]


# --- the carry-over actions, now inside the editor --------------------------- #
def test_scene_geometry_actions_say_why_they_are_off(qapp, tmp_path):
    """Copy/Import/Export moved off the page and into the editor, beside what they carry.

    They keep the rule they had on the page: a control that cannot run explains itself
    rather than sitting inert. Driven as a pair — an experiment with nowhere to copy to,
    then one with somewhere — because either half alone passes with the reason hardcoded.
    """
    from detail.gui.arena import SceneGeometryDialog

    proj = Experiment.create("p", tmp_path)
    only = proj.create_recording("s1")
    only.define_arena("rig", _square_kf())

    alone = SceneGeometryDialog(only, proj)
    assert not alone._copy.isEnabled()
    assert alone._copy.toolTip() == "No other recordings in this experiment"
    assert alone._export.isEnabled()  # there is something to write

    proj.create_recording("s2")
    with_others = SceneGeometryDialog(proj.get_recording("s1"), proj)
    assert with_others._copy.isEnabled()
    assert not with_others._copy.toolTip()


def test_scene_geometry_export_has_nothing_to_write_when_empty(qapp, tmp_path):
    from detail.gui.arena import SceneGeometryDialog

    proj = Experiment.create("p", tmp_path)
    empty = proj.create_recording("s1")
    dialog = SceneGeometryDialog(empty, proj)
    assert not dialog._export.isEnabled()
    assert dialog._export.toolTip() == "Nothing defined on this recording yet"


def test_scene_geometry_export_names_the_experiment_and_recording(
    qapp, tmp_path, monkeypatch
):
    """`Experiment` has no `.id`; reading one was an AttributeError on every export."""
    from PyQt6.QtWidgets import QFileDialog

    from detail.gui.arena import SceneGeometryDialog

    proj = Experiment.create("study3", tmp_path)
    rec = proj.create_recording("s1")
    rec.define_arena("rig", _square_kf())
    dialog = SceneGeometryDialog(rec, proj)

    seen = {}

    def _stub(_parent, _title, default, _filter):
        seen["default"] = default
        return "", ""  # the user cancels; we only care what was offered

    with monkeypatch.context() as patch:
        patch.setattr(QFileDialog, "getSaveFileName", staticmethod(_stub))
        dialog._export_spatial()
    assert "study3_s1_spatial.csv" in seen["default"]


def test_the_two_directions_offer_the_same_definitions(qapp, tmp_path):
    """Copy and Import are one operation with the arrow reversed, so they list the same
    things and block them for the same reasons.

    They are edited at different times by whoever is looking at one of them, which is how
    they drifted apart the first time — Import reached regions and points while Copy
    reached all four, and only one of them flattened or refused anything. Driving both
    over one fixture is what keeps a capability added to either from landing on one.
    """
    from detail.gui.pages import CopyToRecordingsDialog, _ImportSpatialDialog

    proj = Experiment.create("p", tmp_path)
    src = proj.create_recording("s1")
    src.define_arena("rig", _square_kf())
    src.define_roi("nest", _square_kf())
    src.define_roi("timed", _square_kf(), active=(500, 900))
    src.define_point("lever", [(0, [(0.5, 0.5)])])
    dst = proj.create_recording("s2")

    push = CopyToRecordingsDialog(src, proj)
    pull = _ImportSpatialDialog(dst, proj)

    def rows(dialog):
        return [
            (m.kind.id, m.name, bool(m.blocked), m.flattens) for m in dialog._movables
        ]

    assert rows(push) == rows(pull), "the two directions disagree about what can travel"
    # and the one that cannot travel is refused in both, for the timeline reason
    assert [m.name for m in push._movables if m.blocked] == ["timed"]
    assert [m.name for m in pull._movables if m.blocked] == ["timed"]


def test_copy_sends_only_the_definitions_that_are_ticked(qapp, tmp_path):
    """Unticking a region has to actually stop it.

    The selection travels as a name filter through `plan` → `propagate_spatial(names=…)`,
    and dropping that argument copies the whole category while the dialog goes on showing
    the tick you cleared. Nothing fails: the copy succeeds, the report says so, and the
    recording gets a region the user deliberately excluded.
    """
    from detail.gui.pages import CopyToRecordingsDialog

    proj = Experiment.create("p", tmp_path)
    src = proj.create_recording("s1")
    src.define_roi("nest", _square_kf())
    src.define_roi("feeder", _square_kf())
    src.define_point("lever", [(0, [(0.5, 0.5)])])
    proj.create_recording("s2")

    dialog = CopyToRecordingsDialog(src, proj)
    _row_for_name(dialog, "nest").setCheckState(Qt.CheckState.Unchecked)
    _row_for_name(dialog, "lever").setCheckState(Qt.CheckState.Unchecked)

    plan = dialog.plan()
    assert plan[0] == {"roi": ["feeder"]}, plan[0]
    dialog.run(plan, None)

    target = proj.get_recording("s2")
    assert [r["name"] for r in target.rois] == [
        "feeder"
    ], "an unticked region was copied"
    assert target.points == [], "an unticked point was copied"


# --- the run page's Measures table ------------------------------------------- #
def test_the_measures_table_keeps_every_column_draggable(qapp):
    """Handing the slack to **Label** used to relock the count column to `ResizeToContents`, so
    a user could not drag it (#463), and `Stretch` on the label made every column after it
    grow leftward when dragged. The table only exists once `_show_rows` lands, which no
    page-construction sweep reaches, so this drives it directly and drags each border.
    """
    from detail.gui.pages import MeasuresPanel
    from tests.gui.layout_checks import assert_border_follows_the_drag, locked_columns

    panel = MeasuresPanel(experiment=None, run_id="analysis_001")
    panel._show_rows(
        [
            ("recording_001", "0", "rearing", "Mean bout", "1.20", "s", "3"),
            ("recording_002", "0", "grooming", "Bout rate", "0.40", "/min", "7"),
        ]
    )
    panel.resize(1000, 400)
    panel.show()
    qapp.processEvents()
    table = panel._body
    header = table.horizontalHeader()
    assert locked_columns(table) == []
    for column in range(header.count()):
        assert_border_follows_the_drag(table, column, dx=30)


def _measures_rows(n):
    return [
        (f"recording_{i:03d}", "0", "climb", "Median bout length", "1.20", "s", "3")
        for i in range(n)
    ]


def test_the_measures_tables_spare_width_is_shared_by_every_column(qapp):
    """At 1440 wide the slack went to Label alone — 678px, Measure / Value / Unit at their
    content widths, the numbers a gulf away from the names. Every column now gets its content
    and an equal share of what is left."""
    from detail.gui.pages import MeasuresPanel

    panel = MeasuresPanel(experiment=None, run_id="analysis_001")
    panel._show_rows(_measures_rows(2))
    panel.resize(1200, 400)
    panel.show()
    qapp.processEvents()
    try:
        table = panel._body
        header = table.horizontalHeader()
        extra = [
            header.sectionSize(c)
            - max(header.sectionSizeHint(c), table.sizeHintForColumn(c))
            for c in range(header.count())
        ]
        assert min(extra) > 40, f"the spare width was not shared: {extra}"
        assert max(extra) - min(extra) <= header.count(), extra
        assert sum(header.sectionSize(c) for c in range(header.count())) <= (
            table.viewport().width()
        )
    finally:
        panel.hide()


def test_the_measures_table_fills_its_tab(qapp):
    """The table is the tab's whole content: with more rows than fit, it reaches the bottom of
    the tab rather than stopping at the display's row budget — 118px of empty tab under it at
    1440x847."""
    from PyQt6.QtWidgets import QApplication

    from detail.gui.pages import MeasuresPanel
    from detail.gui.tables import top_of_pane

    panel = MeasuresPanel(experiment=None, run_id="analysis_001")
    panel._show_rows(_measures_rows(200))
    pane = top_of_pane(panel)
    screen = QApplication.primaryScreen().availableGeometry().height()
    pane.resize(1200, screen + 400)
    pane.show()
    qapp.processEvents()
    try:
        table = panel._body
        bottom = table.mapTo(pane, table.rect().bottomLeft()).y()
        assert bottom >= pane.height() - 2, (bottom, pane.height())
    finally:
        pane.hide()


def _calibration_rigs(tmp_path):
    """A calibrated 16:9 source, a 16:9 recording and a 5:4 one — two rigs."""
    proj = Experiment.create("p", tmp_path)
    recs = {}
    for rid, res in (
        ("s1", [1920, 1080]),
        ("same", [960, 540]),
        ("other", [1280, 1024]),
    ):
        rec = proj.create_recording(rid)
        data = recording_store.read_recording(rec.dir)
        data["video"] = {"resolution": res, "files": [], "storage": "default"}
        recording_store.write_recording(rec.dir, data)
        recs[rid] = rec
    recs["s1"].set_distance_calibration([(0.5, 0.25), (0.5, 0.75)], 200.0)
    return proj, recs


def test_copy_dialog_greys_out_a_recording_from_another_rig(qapp, tmp_path):
    """A calibration belongs to its rig: a recording whose video is another shape is
    greyed out before the click, with the reason, not refused after it."""
    from PyQt6.QtCore import Qt

    from detail.gui.pages import CopyToRecordingsDialog

    proj, recs = _calibration_rigs(tmp_path)
    dialog = CopyToRecordingsDialog(recs["s1"], proj)
    rows = {
        dialog._targets.item(i).data(Qt.ItemDataRole.UserRole): dialog._targets.item(i)
        for i in range(dialog._targets.count())
    }
    other = rows[recs["other"].id]
    assert not (other.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "different video shape: Distance calibration" in other.text()
    assert "1280x1024" in other.toolTip()
    assert rows[recs["same"].id].flags() & Qt.ItemFlag.ItemIsEnabled
    assert dialog._checked_target_ids() == [recs["same"].id]


def test_copy_dialog_withdraws_a_calibration_no_chosen_recording_can_take(
    qapp, tmp_path
):
    from detail.gui.pages import CopyToRecordingsDialog

    proj, recs = _calibration_rigs(tmp_path)
    recs["same"].set_pixels_per_mm(1.0)  # the only same-rig target is taken
    dialog = CopyToRecordingsDialog(recs["s1"], proj)
    row = _row_for_name(dialog, "Distance calibration")
    assert not (row.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "different video shape" in row.toolTip()


def test_import_dialog_disables_a_calibration_from_another_rig(qapp, tmp_path):
    from detail.gui.pages import _ImportSpatialDialog

    proj, recs = _calibration_rigs(tmp_path)
    dialog = _ImportSpatialDialog(recs["other"], proj)
    dialog._rec_combo.setCurrentIndex(dialog._rec_combo.findData(recs["s1"].id))
    row = _row_for_name(dialog, "Distance calibration")
    assert not (row.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "different video shape" in row.toolTip()

    same = _ImportSpatialDialog(recs["same"], proj)
    same._rec_combo.setCurrentIndex(same._rec_combo.findData(recs["s1"].id))
    assert (
        _row_for_name(same, "Distance calibration").flags() & Qt.ItemFlag.ItemIsEnabled
    )


def _three_part_recording(tmp_path):
    """A recording with three video parts of 60, 70 and 80 frames and no processing runs
    (the only state in which reordering is allowed)."""
    import cv2
    import numpy as np

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    for n_frames in (60, 70, 80):
        vid = tmp_path / f"clip_{n_frames}.mp4"
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
        for _ in range(n_frames):
            vw.write(np.zeros((16, 16, 3), np.uint8))
        vw.release()
        recording.add_video_part(vid)
    return proj, proj.get_recording("s1")


def _part_frame_counts(recording) -> list[int]:
    import cv2

    counts = []
    for part in recording.video_parts:
        cap = cv2.VideoCapture(str(recording.dir / part["file"]))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    return counts


def test_the_video_parts_table_does_not_sort(qapp, tmp_path):
    """Its row order *is* the part order Move up / Move down edits, so a header click
    must not reorder it (#26). Listed in `test_table_sorting.SORT_EXEMPT`."""
    from detail.gui.pages import RecordingDetailPage

    _proj, recording = _three_part_recording(tmp_path)
    page = RecordingDetailPage(recording)
    assert page._parts_table.objectName() == "video_parts"
    assert not page._parts_table.isSortingEnabled()


@pytest.mark.parametrize(
    "delta, expected",
    [(1, [60, 70, 80]), (-1, [60, 80, 70])],
    ids=["move_down_on_the_last_part_does_nothing", "move_up_moves_the_last_part"],
)
def test_move_acts_on_the_part_the_row_shows_even_when_reordered(
    qapp, tmp_path, delta, expected
):
    """Reorder the parts view (as a header click did before the table stopped sorting),
    select the top row — part #2, the last — and press Move down / Move up. The part the
    row shows moves; before the fix the row's *position* was used, so Move down on
    the last part swapped parts 0 and 1 and rewrote both timestamp maps (#26)."""
    from detail.gui.pages import RecordingDetailPage
    from detail.gui.tables import row_id

    _proj, recording = _three_part_recording(tmp_path)
    assert _part_frame_counts(recording) == [60, 70, 80]
    page = RecordingDetailPage(recording)
    table = page._parts_table
    # `sortByColumn` reorders the model whether or not header-click sorting is enabled,
    # so this still exercises the identity read with sorting turned off.
    table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    assert [table.model().index(r, 0).data() for r in range(3)] == ["2", "1", "0"]
    table.selectRow(0)
    assert row_id(table.currentIndex()) == 2

    page._move_part(delta)

    assert _part_frame_counts(recording) == expected
    # The moved part stays selected, wherever it now sits.
    expected_selected = 2 if delta == 1 else 1
    assert row_id(page._parts_table.currentIndex()) == expected_selected
