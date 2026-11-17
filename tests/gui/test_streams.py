"""GUI stream dialogs: add + preview construct and render (Section 3).

**A chosen mapping dropped by the next file** (#35). The CSV band re-applied a saved mapping only
when it had been chosen *before* any file, so picking another file — the very advice the
missing-column message gave — unticked every column while **Reuse a mapping** still named the
mapping, and choosing it again did nothing because the selection had not changed. The band keeps
the mapping until another is chosen and applies it to each file; the stream keeps its name.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QScrollArea  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402

DATA_TXT = """\
Start Date: 02/16/26
Subject: TEST1
MSN: Test_MSN
G:       3.000
H:      10.000
E:
     0:       19.000       11.000      100.000
T:
     0:        0.000        1.500       10.000
"""

PROGRAM_MPC = """\
\\  Event stamps ( array E )
\\   11  = Houselight on
\\   19  = Drug available
\\   100 = Recording termination
\\
\\  Variables
"""


def _recording_with_stream(tmp_path):
    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    src = tmp_path / "_src"
    src.mkdir()
    (src / "subject.txt").write_text(DATA_TXT, encoding="utf-8")
    (src / "task.MPC").write_text(PROGRAM_MPC, encoding="utf-8")
    recording.add_stream(
        "medpc",
        raw_path=src / "subject.txt",
        importer="medpc",
        aux_paths={"program": src / "task.MPC"},
    )
    return recording


BORIS_CSV = """\
Behavior,Behavior type,Time
groom,START,0.10
groom,STOP,0.30
"""


def _select_importer(dialog, importer_id):
    ids = [dialog._importer.itemData(i) for i in range(dialog._importer.count())]
    dialog._importer.setCurrentIndex(ids.index(importer_id))


def _recording_with_video_and_streams(tmp_path):
    """A recording with a video plus one external-clock (MED-PC) and one video-anchored
    (BORIS) stream — the two alignment cases the streams UI must distinguish."""
    import cv2
    import numpy as np

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    vid = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
    for _ in range(30):  # ~1s so the BORIS times (0.1s–0.3s) land inside the video
        vw.write(np.zeros((16, 16, 3), np.uint8))
    vw.release()
    recording.add_video_part(vid)

    src = tmp_path / "_src"
    src.mkdir()
    (src / "subject.txt").write_text(DATA_TXT, encoding="utf-8")
    (src / "task.MPC").write_text(PROGRAM_MPC, encoding="utf-8")
    recording.add_stream(
        "medpc",
        raw_path=src / "subject.txt",
        importer="medpc",
        aux_paths={"program": src / "task.MPC"},
    )
    (src / "labels.csv").write_text(BORIS_CSV, encoding="utf-8")
    recording.add_stream("behavior", raw_path=src / "labels.csv", importer="boris")
    return proj.get_recording("s1")


def test_add_stream_dialog_lists_importer_and_aux(qapp, tmp_path):
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    dialog = AddStreamDialog(recording)
    importer_ids = [
        dialog._importer.itemData(i) for i in range(dialog._importer.count())
    ]
    assert "medpc" in importer_ids
    _select_importer(dialog, "medpc")
    assert list(dialog._aux_pickers) == ["program"]  # MED-PC's .MPC aux input


def test_add_stream_dialog_includes_behavior_importers(qapp, tmp_path):
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    dialog = AddStreamDialog(recording)
    ids = [dialog._importer.itemData(i) for i in range(dialog._importer.count())]
    assert "behavior_csv" in ids and "boris" in ids
    # a behavior importer needs no aux files and suggests the "behavior" name
    _select_importer(dialog, "behavior_csv")
    assert dialog._aux_pickers == {}
    assert dialog._name.text() == "behavior"


def test_add_stream_dialog_includes_generic_csv_importers(qapp, tmp_path):
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    dialog = AddStreamDialog(recording)
    ids = [dialog._importer.itemData(i) for i in range(dialog._importer.count())]
    assert "timeseries_csv" in ids and "events_csv" in ids
    # both are single-file importers (no aux pickers)
    _select_importer(dialog, "timeseries_csv")
    assert dialog._aux_pickers == {}
    _select_importer(dialog, "events_csv")
    assert dialog._aux_pickers == {}


def test_import_behavior_labels_dialog_constructs(qapp, tmp_path):
    from detail.gui.streams import ImportBehaviorLabelsDialog

    proj = Experiment.create("p", tmp_path)
    dialog = ImportBehaviorLabelsDialog(proj)
    ids = [dialog._importer.itemData(i) for i in range(dialog._importer.count())]
    assert ids == ["behavior_csv", "boris"]
    assert dialog._name.text() == "behavior"
    assert not dialog._manifest.path  # nothing selected yet; no worker started
    assert dialog._worker is None


def test_import_behavior_labels_drives_a_bar(qapp, tmp_path, monkeypatch):
    """The API always reported per recording; the dialog dropped it and said "Importing…"
    until the batch ended (#916). The folded fraction moves inside a recording, too."""
    from detail.gui import streams
    from detail.gui.streams import ImportBehaviorLabelsDialog

    proj = Experiment.create("p", tmp_path)
    manifest = tmp_path / "labels.yaml"
    manifest.write_text("schema_version: 1\nlabels: {}\n", encoding="utf-8")

    def fake_import(path, *, name, importer, progress=None):
        assert progress is not None, "the dialog dropped the callback"
        progress(0, 2, "recording_001", 0.5, "running")
        progress(1, 2, "recording_002", 1.0, "done")
        return []

    monkeypatch.setattr(proj, "import_behavior_labels", fake_import)
    emitted = []

    def sync_run_progress(target, work, on_done, **_kw):
        on_done(work(lambda *event: emitted.append(event)))

    monkeypatch.setattr(streams, "run_progress", sync_run_progress)
    monkeypatch.setattr(streams.QMessageBox, "information", lambda *a, **k: None)
    dialog = ImportBehaviorLabelsDialog(proj)
    dialog._manifest.set_path(str(manifest))
    dialog._import()

    assert [round(fraction, 2) for _s, fraction, _d in emitted] == [0.25, 1.0]
    assert "recording_001" in emitted[0][2]


def test_boris_label_names_the_supported_export_flavor(qapp, tmp_path):
    # BORIS exports several formats; we read only the tabular events export, so the
    # picker label must say which — in both surfaces that import it.
    from detail.gui.streams import AddStreamDialog, ImportBehaviorLabelsDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    add = AddStreamDialog(recording)
    ids = [add._importer.itemData(i) for i in range(add._importer.count())]
    boris_label = add._importer.itemText(ids.index("boris"))
    assert "tabular events export" in boris_label

    batch = ImportBehaviorLabelsDialog(proj)
    b_ids = [batch._importer.itemData(i) for i in range(batch._importer.count())]
    assert "tabular events export" in batch._importer.itemText(b_ids.index("boris"))


def test_alignment_buttons_gated_on_the_selected_streams_clock(qapp, tmp_path):
    # A video-anchored stream (BORIS behavior labels) is already aligned, so Sync /
    # Edit-offset must be disabled for it; an external-clock stream (MED-PC) needs a
    # read-time offset, so both are enabled (video present). Guards the unconditional
    # button that crashed on a behavior stream.
    from detail.gui.pages import RecordingDetailPage

    recording = _recording_with_video_and_streams(tmp_path)
    page = RecordingDetailPage(recording)
    assert page._stream_names == ["medpc", "behavior"]

    page._streams_table.selectRow(0)  # MED-PC (stream_seconds)
    page._update_alignment_buttons()
    assert page._sync_btn.isEnabled()
    assert page._edit_offset_btn.isEnabled()

    page._streams_table.selectRow(1)  # BORIS behavior (video_seconds)
    page._update_alignment_buttons()
    assert not page._sync_btn.isEnabled()
    assert not page._edit_offset_btn.isEnabled()
    assert "no alignment needed" in page._sync_btn.toolTip()


def _wait(cond, timeout_ms=2000):
    """Spin the event loop until ``cond()`` — the stream dialogs read off-thread now,
    so their anchor picker / strip / scrubber appear only after the async populate."""
    from PyQt6.QtTest import QTest

    for _ in range(timeout_ms // 10):
        if cond():
            return
        QTest.qWait(10)
    raise AssertionError("stream dialog did not populate off-thread")


def test_sync_stream_dialog_builds_for_both_row_shapes(qapp, tmp_path):
    # The dialog is offered only for external-clock streams, but its anchor picker
    # must tolerate either row shape by index (events: t_ns,code,label; interval:
    # t_ns,end_t_ns,label,…) — the interval unpack is what crashed before.
    from detail.gui.streams import SyncStreamDialog

    recording = _recording_with_video_and_streams(tmp_path)
    medpc = SyncStreamDialog(recording, "medpc")  # 3-tuple events
    _wait(lambda: medpc._anchor.count() > 0)  # events load off-thread
    assert medpc._anchor.count() > 0
    interval = SyncStreamDialog(recording, "behavior")  # 6-tuple intervals, no crash
    _wait(lambda: interval._anchor.count() > 0)
    assert interval._anchor.count() == 1


def test_stream_preview_dialog_no_video(qapp, tmp_path):
    from detail.gui.streams import StreamPreviewDialog

    recording = _recording_with_stream(tmp_path)
    dialog = StreamPreviewDialog(recording, "medpc")
    _wait(lambda: hasattr(dialog, "_strip"))  # off-thread load then populate
    # three distinct lanes, one tick-list each; no video → time slider
    assert dialog._strip._lanes == [
        "Drug available",
        "Houselight on",
        "Recording termination",
    ]
    assert not dialog._has_video
    # scrubbing updates the "now" panel without error
    dialog._on_scrub(0)
    assert "Drug available" in dialog._now.text()


def test_preview_puts_the_video_first_and_keeps_its_controls(qapp, tmp_path):
    """The video is what this window is for, and it goes above the raster.

    Reported from the app: the frame was a ~15px band at the very bottom with no transport at
    all. Two causes, both asserted here — the raster was laid out first and took a *fixed* 320px
    whatever was left for the video, and `PictureWithControls` had a size hint but no floor, so
    a host that under-gave it positioned the controls off the end of the widget, where they are
    clipped away.
    """
    from detail.gui.image_view import ImageView, PictureWithControls
    from detail.gui.streams import StreamPreviewDialog

    recording = _recording_with_video_and_streams(tmp_path)
    dialog = StreamPreviewDialog(recording, "medpc")
    dialog.resize(810, 640)
    dialog.show()
    _wait(lambda: hasattr(dialog, "_block"))
    dialog._layout.activate()
    qapp.processEvents()

    layout = dialog._layout
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    block = dialog._block
    assert isinstance(block, PictureWithControls)
    assert isinstance(dialog._image, ImageView) and dialog._image.zoomable
    strip_scroll = next(w for w in order if isinstance(w, QScrollArea))
    assert order.index(block) < order.index(
        strip_scroll
    ), "the raster is above the video"

    # The controls are inside the block, not positioned past its end and clipped away.
    controls = dialog._scrubber.geometry()
    assert controls.bottom() <= block.height(), (
        f"the transport sits at {controls.bottom()}px in a {block.height()}px block, so it is "
        f"clipped — the picture took the whole widget"
    )
    assert dialog._image.height() >= 200, "the video is a band rather than a picture"
    dialog.close()


def test_stream_preview_strip_click_moves_slider(qapp, tmp_path):
    from detail.gui.streams import StreamPreviewDialog

    recording = _recording_with_stream(tmp_path)
    dialog = StreamPreviewDialog(recording, "medpc")
    _wait(lambda: hasattr(dialog, "_scrubber"))
    dialog._on_strip_click(10_000_000_000)  # the termination event time
    assert dialog._scrubber.frame() > 0


# -- AddStreamDialog: one surface, importer-driven (Section 3) ---------------- #


def _pose_csv(tmp_path, name="pose_001.csv"):
    """A CSV shaped the way a source tool shaped it, so `can_import` claims nothing."""
    p = tmp_path / name
    p.write_text("t_sec,nose_x,nose_y,tail_x\n0,1,2,3\n0.1,4,5,6\n", encoding="utf-8")
    return p


def _tick(band, column):
    from PyQt6.QtWidgets import QCheckBox

    for row in range(band._table.rowCount()):
        if band._table.item(row, 1).text() == column:
            band._table.cellWidget(row, 0).findChild(QCheckBox).click()
            return
    raise AssertionError(f"no row for {column!r}")


def _dialog(tmp_path, qapp):
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    dialog = AddStreamDialog(recording)
    return recording, dialog


def test_an_unrecognized_csv_falls_back_to_the_configurable_importer(qapp, tmp_path):
    """The case this dialog exists for.

    `can_import` recognizes a CSV *already shaped for DETAIL*, so a file that needs a column
    mapping is by definition one it cannot claim. Leaving the picker on whatever came first
    alphabetically would show an empty dialog for exactly the file the mapping is for.
    """
    recording, dialog = _dialog(tmp_path, qapp)
    csv_path = _pose_csv(tmp_path)

    dialog._data.set_path(str(csv_path))

    assert dialog._importer.currentData() == "timeseries_csv"
    assert dialog._band is not None
    assert "No importer recognized" in dialog._sniffed.text()


def test_a_recognized_file_is_claimed_by_its_own_importer(qapp, tmp_path):
    """Guard the guard: the fallback must not be what makes the test above pass."""
    recording, dialog = _dialog(tmp_path, qapp)
    labels = tmp_path / "labels.csv"
    labels.write_text("start_frame,end_frame,label\n1,3,groom\n", encoding="utf-8")

    dialog._data.set_path(str(labels))

    assert dialog._importer.currentData() == "behavior_csv"
    assert "Read as" in dialog._sniffed.text()


def test_the_time_kind_follows_the_time_column(qapp, tmp_path):
    """A kind pinned to one value regardless of the column is silently wrong data: frame
    indices read as seconds import as t_ns = index x 1e9, and nothing complains."""
    recording, dialog = _dialog(tmp_path, qapp)
    p = tmp_path / "f.csv"
    p.write_text("frame,t_sec,v\n0,0.0,1\n1,0.1,2\n", encoding="utf-8")
    dialog._data.set_path(str(p))

    assert dialog._band._time_col.currentText() == "frame"
    assert dialog._band._time_kind.currentData() == "frame"

    dialog._band._time_col.setCurrentText("t_sec")
    assert dialog._band._time_kind.currentData() == "seconds"


def test_ticked_columns_become_channels_of_one_stream(qapp, tmp_path):
    recording, dialog = _dialog(tmp_path, qapp)
    dialog._data.set_path(str(_pose_csv(tmp_path)))
    _tick(dialog._band, "nose_x")
    _tick(dialog._band, "nose_y")
    dialog._name.setText("pose")

    dialog._import()

    assert len(recording.streams) == 1
    assert recording.streams[0]["columns"] == ["t_ns", "nose_x", "nose_y"]


def test_the_offset_is_set_at_import(qapp, tmp_path):
    recording, dialog = _dialog(tmp_path, qapp)
    dialog._data.set_path(str(_pose_csv(tmp_path)))
    _tick(dialog._band, "nose_x")
    dialog._name.setText("pose")
    dialog._offset.setValue(3.48)

    dialog._import()

    assert recording.streams[0]["alignment"]["offset_ns"] == 3_480_000_000


def test_alignment_row_is_gated_on_the_importers_declared_clock(qapp, tmp_path):
    """Read off `clock`, never a list of importer ids, so a new importer inherits it."""
    recording, dialog = _dialog(tmp_path, qapp)

    dialog._importer.setCurrentIndex(dialog._importer.findData("behavior_csv"))
    assert not dialog._offset.isEnabled()  # video_frames: anchored at import
    assert dialog._origin.isEnabled()  # behavior role: the origin choice applies

    dialog._importer.setCurrentIndex(dialog._importer.findData("medpc"))
    assert dialog._offset.isEnabled()  # stream_seconds: MED-PC gains offset-at-import
    assert not dialog._origin.isEnabled()


def test_a_duplicate_header_is_named_before_anything_is_mapped(qapp, tmp_path):
    """The one problem the user can only fix in the source file, so it is said up front."""
    recording, dialog = _dialog(tmp_path, qapp)
    p = tmp_path / "dup.csv"
    p.write_text("t_sec,x,x\n0,1,2\n", encoding="utf-8")

    dialog._data.set_path(str(p))

    assert "more than once" in dialog._band.problem()
    assert not dialog._box.button(dialog._box.StandardButton.Ok).isEnabled()


def test_a_reserved_channel_name_is_refused_in_the_form(qapp, tmp_path):
    recording, dialog = _dialog(tmp_path, qapp)
    dialog._data.set_path(str(_pose_csv(tmp_path)))
    _tick(dialog._band, "nose_x")
    dialog._band._rows["nose_x"]["name"] = "t_ns"
    dialog._band._emit_changed()

    assert "reserved" in dialog._band.problem()


def test_toggling_a_column_keeps_the_other_rows_widgets(qapp, tmp_path):
    """Rebuilding the table inside the checkbox's own signal destroys the checkbox being
    clicked and takes the keyboard focus with it, on every space press."""
    recording, dialog = _dialog(tmp_path, qapp)
    dialog._data.set_path(str(_pose_csv(tmp_path)))
    band = dialog._band
    before = [band._table.cellWidget(r, 0) for r in range(band._table.rowCount())]

    _tick(band, "nose_x")

    after = [band._table.cellWidget(r, 0) for r in range(band._table.rowCount())]
    assert before == after, "the row widgets were rebuilt under the click"


def test_the_file_is_read_once_per_pick(qapp, tmp_path):
    """Picking a file both swaps the band and hands it the path; without a guard the whole
    table is rebuilt twice, and the second rebuild lands under the user's first click.
    """
    recording, dialog = _dialog(tmp_path, qapp)
    csv_path = _pose_csv(tmp_path)
    reads = []
    dialog._data.set_path(str(csv_path))
    band = dialog._band
    original = band._rebuild_rows
    band._rebuild_rows = lambda: (reads.append(1), original())[1]

    band.set_source(str(csv_path))  # the same file again

    assert reads == [], "set_source re-read a file it had already read"


# -- continuous streams reach the alignment + preview surfaces ---------------- #


def _recording_with_continuous_stream(tmp_path, channels=("nose_x", "nose_y")):
    """A video plus a continuous stream — the shape the sync/preview surfaces never had."""
    import cv2
    import numpy as np

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    vid = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
    for _ in range(30):
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    recording.add_video_part(vid)

    src = tmp_path / "trace.csv"
    header = ",".join(["t_sec", *channels])
    rows = "".join(
        f"{i / 30:.4f}," + ",".join(str(i + j) for j, _ in enumerate(channels)) + "\n"
        for i in range(5)
    )
    src.write_text(f"{header}\n{rows}", encoding="utf-8")
    recording.add_stream(
        "trace",
        raw_path=src,
        importer="timeseries_csv",
        config={
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [{"column": c, "name": c} for c in channels],
        },
    )
    return recording


@pytest.mark.parametrize("channels", [("nose_x",), ("nose_x", "nose_y")])
def test_sync_offers_one_anchor_for_a_continuous_stream(qapp, tmp_path, channels):
    """A continuous stream has no events to anchor on.

    Reading `row[2]` as a label listed sensor *values* as anchor names on a multi-channel
    stream (one entry per sample) and raised `IndexError` on a single-channel one — where
    `stream_seconds` means Sync is offered, so it was reachable.
    """
    from detail.gui.streams import SyncStreamDialog

    recording = _recording_with_continuous_stream(tmp_path, channels)
    dialog = SyncStreamDialog(recording, "trace")
    _wait(lambda: dialog._anchor.count() > 0)

    assert dialog._anchor.count() == 1
    assert "stream start" in dialog._anchor.itemText(0)


def test_sync_for_rows_returns_an_offset_without_writing(qapp, tmp_path):
    """The import flow syncs *before* the stream exists, so the dialog hands the number
    back instead of writing it."""
    from detail.gui.streams import SyncStreamDialog

    recording = _recording_with_continuous_stream(tmp_path)
    before = recording.streams[0]["alignment"]["offset_ns"]
    dialog = SyncStreamDialog.for_rows(
        recording, ["t_ns", "v"], [(0, 1.0), (100_000_000, 2.0)]
    )
    # `decode=False`: this test wants frame 3's *timestamp*, not its picture. The default starts a
    # `_DecodeTask` on the global QThreadPool, and the test returns without reading it — the one
    # test in the gui tier (of ~1,609) that still had work outstanding after the teardown drain
    # gave up, so its decode crossed into whatever ran next. `_apply` sets the frame, the slider
    # and `frame_changed` synchronously and only gates `_request` on `decode`, so nothing this
    # test asserts on is lost (#533).
    dialog._scrubber.set_frame(3, decode=False)
    dialog._save()

    assert dialog.offset_ns == recording.frame_t_ns(3)
    assert recording.streams[0]["alignment"]["offset_ns"] == before  # nothing written


def test_preview_shows_a_continuous_streams_samples(qapp, tmp_path):
    """It used to draw one lane per sample, named after that sample's channel value —
    plausible-looking output rather than a failure. The samples are still worth showing:
    when a stream has data against the video is exactly the sync check."""
    from detail.gui.streams import StreamPreviewDialog

    recording = _recording_with_continuous_stream(tmp_path)
    dialog = StreamPreviewDialog(recording, "trace")
    _wait(lambda: hasattr(dialog, "_strip"))  # off-thread load then populate

    assert dialog._strip._lanes == ["samples"]
    assert len(dialog._strip._events["samples"]) == 5  # one tick per sample


def test_stream_preview_refuses_to_read_channels_as_labels(tmp_path):
    """The API half. `-1` is a valid Python index, so the sentinel read the *last* column:
    `labels()` returned ['2.0', '5.0'] for a sensor trace and nothing looked wrong."""
    recording = _recording_with_continuous_stream(tmp_path)
    preview = recording.stream_preview("trace")

    assert preview.has_labels is False
    assert preview.channels == ["nose_x", "nose_y"]
    assert len(preview.times()) == 5
    with pytest.raises(ValueError, match="no 'label' column"):
        preview.labels()


# -- saved setups in the dialog (Section 3) ---------------------------------- #


def test_choosing_a_saved_setup_prefills_the_mapping(qapp, tmp_path):
    """A mapping worked out on one recording, reused on the next — from the dialog."""
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("s1")
    recording = proj.create_recording("s2")
    proj.save_stream_preset(
        "dlc",
        "timeseries_csv",
        {
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [{"column": "nose_y", "name": "ny"}],
        },
    )
    dialog = AddStreamDialog(recording)
    dialog._data.set_path(str(_pose_csv(tmp_path)))

    dialog._preset.setCurrentIndex(dialog._preset.findData("dlc"))

    assert dialog._band._time_col.currentText() == "t_sec"
    assert dialog._band.config()["channels"] == [{"column": "nose_y", "name": "ny"}]
    assert dialog._band.problem() is None


def test_a_saved_setup_names_the_columns_this_csv_lacks(qapp, tmp_path):
    """Importing 'the rest' of a preset silently gives one recording fewer channels than
    the others, and nothing downstream says why it is short."""
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    proj.save_stream_preset(
        "dlc",
        "timeseries_csv",
        {
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [
                {"column": "nose_x", "name": "nose_x"},
                {"column": "elbow", "name": "elbow"},
            ],
        },
    )
    dialog = AddStreamDialog(recording)
    dialog._data.set_path(str(_pose_csv(tmp_path)))  # has no `elbow`

    dialog._preset.setCurrentIndex(dialog._preset.findData("dlc"))

    assert "elbow" in dialog._band.problem()
    assert not dialog._box.button(dialog._box.StandardButton.Ok).isEnabled()


def test_a_setup_chosen_before_a_file_applies_once_one_arrives(qapp, tmp_path):
    """The combo sits above the file row in the flow people actually use second: pick the
    setup you know you want, then browse. It must not be dropped on the floor.

    Since #332 the list is scoped to the selected importer, so picking the importer is the
    first step of this flow rather than something a chosen mapping did silently. What
    follows it is unchanged: the mapping is held and applied once a file arrives.
    """
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    proj.save_stream_preset(
        "dlc",
        "timeseries_csv",
        {
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [{"column": "nose_x", "name": "nose_x"}],
        },
    )
    dialog = AddStreamDialog(recording)

    dialog._importer.setCurrentIndex(dialog._importer.findData("timeseries_csv"))
    dialog._preset.setCurrentIndex(dialog._preset.findData("dlc"))
    dialog._data.set_path(str(_pose_csv(tmp_path)))

    assert dialog._band.config()["channels"] == [{"column": "nose_x", "name": "nose_x"}]


def test_saving_a_setup_from_the_dialog(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog

    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    dialog = AddStreamDialog(recording)
    dialog._data.set_path(str(_pose_csv(tmp_path)))
    _tick(dialog._band, "nose_x")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("dlc", True))

    dialog._save_preset()

    saved = proj.get_stream_preset("dlc")
    assert saved["importer"] == "timeseries_csv"
    assert saved["config"]["channels"] == [{"column": "nose_x", "name": "nose_x"}]
    assert dialog._preset.currentData() == "dlc"  # and it is selected afterwards


def test_the_save_prompt_states_the_name_rule_and_what_the_name_becomes(
    qapp, tmp_path, monkeypatch
):
    """The charset was discoverable only by tripping over it — a name with a space is
    refused after you have typed it (#332). The coupling is stated in the same breath
    because it is the reason the rule cannot simply be relaxed: this name becomes the
    stream's name, and a stream name is a directory name.
    """
    from PyQt6.QtWidgets import QInputDialog

    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    dialog = AddStreamDialog(recording)
    dialog._data.set_path(str(_pose_csv(tmp_path)))
    _tick(dialog._band, "nose_x")

    prompts: list[str] = []

    def capture(_parent, _title, label, *args, **kwargs):
        prompts.append(label)
        return ("dlc", True)

    monkeypatch.setattr(QInputDialog, "getText", capture)
    dialog._save_preset()

    assert prompts, "saving a mapping must ask for a name"
    prompt = prompts[0]
    assert "no spaces" in prompt.lower()
    assert "stream's name" in prompt

    # And the rule the prompt describes is the one actually enforced, so the hint cannot
    # drift away from the charset it is describing.
    with pytest.raises(ValueError):
        proj.save_stream_preset("has space", "timeseries_csv", {})


def test_save_as_is_offered_only_once_there_is_a_mapping_to_save(qapp, tmp_path):
    """Offered on an empty dialog it stores an importer name and no mapping, which reuses
    to exactly what picking that importer by hand already does."""
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    dialog = AddStreamDialog(proj.create_recording("s1"))

    assert not dialog._save_preset_btn.isEnabled()
    assert "Choose a data file" in dialog._save_preset_btn.toolTip()

    dialog._data.set_path(str(_pose_csv(tmp_path)))
    _tick(dialog._band, "nose_x")

    assert dialog._save_preset_btn.isEnabled()


def test_the_reuse_row_reads_as_empty_not_broken(qapp, tmp_path):
    """A live dropdown holding one meaningless entry looks like a list that failed to
    load, and says nothing about how one gets there."""
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")

    assert (
        AddStreamDialog(recording)._preset.itemText(0)
        == "— none saved for this importer yet —"
    )

    proj.save_stream_preset("dlc", "timeseries_csv", {"time_column": "t_sec"})
    reopened = AddStreamDialog(recording)
    reopened._importer.setCurrentIndex(reopened._importer.findData("timeseries_csv"))
    assert [reopened._preset.itemText(i) for i in range(reopened._preset.count())] == [
        "— none —",
        "Time-series CSV (sensor data): dlc",
    ]


def test_a_saved_mapping_is_offered_only_under_its_own_importer(qapp, tmp_path):
    """Both halves as a pair (#332): present under the importer it was saved for, and
    absent under another. Either assertion alone passes with the filter hardcoded the
    other way, which is why they are one test.

    Listing every mapping meant picking one *switched the importer above it* — a row that
    offers a mapping and delivers an importer change.
    """
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    proj.save_stream_preset("ts_map", "timeseries_csv", {"time_column": "t_sec"})
    proj.save_stream_preset("ev_map", "events_csv", {"time_column": "t_sec"})

    dialog = AddStreamDialog(recording)

    def offered() -> set[str]:
        return {
            dialog._preset.itemData(i)
            for i in range(dialog._preset.count())
            if dialog._preset.itemData(i) is not None
        }

    dialog._importer.setCurrentIndex(dialog._importer.findData("timeseries_csv"))
    assert offered() == {"ts_map"}

    dialog._importer.setCurrentIndex(dialog._importer.findData("events_csv"))
    assert offered() == {"ev_map"}

    # No assertion here that applying a mapping leaves the importer alone. The scoping
    # above is what makes that true: every listed mapping already carries the selected
    # importer, so the `setCurrentIndex` this fix removed could only ever have been a
    # no-op once the list is filtered. An assertion that cannot fail is not a guard.


def test_a_chosen_mapping_applies_to_the_next_file_picked(qapp, tmp_path):
    """#35 finding 9. Only a mapping chosen *before* any file was re-applied, so picking
    another file — the advice the missing-column message gives — unticked every column while
    **Reuse a mapping** still named the mapping, and choosing it again did nothing because the
    selection had not changed. The mapping showing is the mapping applied, file after file.
    """
    from detail.gui.streams import AddStreamDialog

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    proj.save_stream_preset(
        "dlc",
        "timeseries_csv",
        {
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [{"column": "elbow", "name": "elbow"}],
        },
    )
    dialog = AddStreamDialog(recording)
    dialog._data.set_path(str(_pose_csv(tmp_path)))  # has no `elbow`
    dialog._preset.setCurrentIndex(dialog._preset.findData("dlc"))
    problem = dialog._band.problem()
    assert problem == (
        "this CSV has no elbow, which the chosen mapping names — pick a file that has it, "
        "or choose another mapping."
    ), "the advice must be something the user can do (there is no row to untick)"

    other = tmp_path / "pose_002.csv"
    other.write_text("t_sec,elbow,nose_x\n0,1,2\n0.1,3,4\n", encoding="utf-8")
    dialog._data.set_path(str(other))

    assert dialog._preset.currentData() == "dlc"
    assert dialog._band.config()["channels"] == [{"column": "elbow", "name": "elbow"}]
    assert dialog._band.problem() is None
    assert dialog._name.text() == "dlc", "the stream keeps the mapping's name"
