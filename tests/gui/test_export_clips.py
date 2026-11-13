"""A clip grid's one **Export clips…** dialog (``gui.export_clips``), opened from **Export ▾**.

Opened the way a user opens it — **Export ▾ → Export clips…** on a ``LiveMontage``, the grid
every page builds, with pick mode for *Picked only* — over
``tests/api/test_clip_views._walking``, a real recording with video, masks and keypoints, so
the preview and every exported file were decoded and drawn, not stubbed. The grid's tiles have
no picture source here: the dialog reads the clips through the montage's ``ExportSource`` and
draws from the source video, as every export does.
"""

import cv2
import pytest
from PIL import Image

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from detail.api import clip_montage as cm  # noqa: E402
from detail.api import montage_export as me  # noqa: E402
from tests.api.test_clip_views import _walking  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

STEP = int(1e9 / 30)
ORIGIN = cm.MontageOrigin(run_id="an_07", node=7, name=None)
_SPANS = ((0, 4), (1, 2), (2, 3))
NO_MODIFIER = Qt.KeyboardModifier.NoModifier


def _grid(qapp, tmp_path, spans=_SPANS, **walking):
    """A montage over ``spans`` of the walking recording, as a page builds one."""
    from detail.api.clip_tiles import TileLook
    from detail.gui import live_montage
    from detail.gui.montage_export import ExportSource

    recording = _walking(tmp_path, **walking)
    clips = [
        cm.MontageClip(i + 1, recording.id, "mouse_07", 0, "emb_01", a * STEP, b * STEP)
        for i, (a, b) in enumerate(spans)
    ]
    experiment_dir = recording.dir.parent.parent
    montage = live_montage.LiveMontage(
        aligned_key="interpret/test_export_aligned",
        look_key=None,
        context_key="interpret/test_export_context",
        export=ExportSource(
            clips=lambda: clips,
            experiment_dir=lambda: experiment_dir,
            origin=lambda: ORIGIN,
            name=lambda: "Cluster #7",
            empty=lambda: "Nothing here.",
            experiment=lambda: None,
        ),
    )
    montage.set_context_s(0.0)
    montage.set_points(None)
    montage.set_look(TileLook(footage=False, masks=True, mask_trail=True))
    montage.set_clips(
        [live_montage.MontageClip(c.end_t_ns - c.start_t_ns, c.rank, "") for c in clips]
    )
    montage._clock.stop()
    montage.resize(900, 600)
    montage.show()
    montage.clips_for_test = clips
    return montage


def _loaded(qapp, dialog):
    assert drain_until(
        qapp,
        lambda: dialog._has_keypoints is not None and dialog.preview_shows(),
        20_000,
    ), "the dialog never read its clips"
    return dialog


def _open(qapp, montage):
    montage.exports.button.actions_by_kind["clips"].trigger()
    dialog = montage.exports._dialog
    dialog.resize(1150, 760)
    return _loaded(qapp, dialog)


def _close(montage):
    if montage.exports._dialog is not None:
        montage.exports._dialog.reject()
    montage.close()
    montage._worker.stop()
    montage.deleteLater()


def test_export_menu_has_the_quick_items_and_the_dialogs(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    texts = [
        a.text() for a in montage.exports.button.menu.actions() if not a.isSeparator()
    ]
    assert texts[:5] == [
        "Pick clips to export…",
        "Video of 3 clips…",
        "GIF of 3 clips…",
        "One file per clip…",
        "CSV list of clips…",
    ]
    assert texts[5:8] == [
        "Export clips…",
        "Export overlay…",
        "New Query from 3 clips…",
    ]
    _close(montage)


def test_it_opens_on_the_defaults(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    assert dialog.ticked_visuals() == [me.CURRENT]
    assert dialog.write() == me.WRITE_BOTH
    assert not dialog.selected_only()
    assert dialog._clips.itemText(0) == "All in the grid (3)"
    assert dialog.formats() == [
        me.ExportFormat("mp4", None, 30.0),
        me.ExportFormat("gif", 640, 12.0),
    ], "a full-size mp4 at the footage's own rate, a 640 px 12 fps GIF"
    assert dialog._mp4_width.currentText().startswith("full size (")
    assert dialog.preview_key() == me.CURRENT
    assert dialog._rows[me.CURRENT][1].text() == "Previewing"
    assert dialog._export.isEnabled()
    assert "= 8 files" in dialog.summary()
    _close(montage)


def test_the_current_rows_name_follows_the_show_menu(qapp, tmp_path):
    from detail.api.clip_tiles import TileLook

    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    assert dialog.current_label() == "Current preview settings  (Mask + Mask trail)"
    montage.set_look(TileLook(footage=True, masks=True, mask_trail=False))
    assert dialog.current_label() == "Current preview settings  (Raw + Mask)"
    assert dialog.plan().looks[0].layers == montage.exports.layers()
    montage.set_look(
        TileLook(footage=True, masks=False, keypoints=True, keypoint_trail=True)
    )
    assert dialog.current_label() == (
        "Current preview settings  (Raw + Keypoints + Keypoint trail)"
    ), "the keypoint trail Show ▾ draws reaches the export"
    _close(montage)


def test_keypoint_rows_are_greyed_with_the_reason_without_keypoints(qapp, tmp_path):
    from detail.gui.export_clips import NO_KEYPOINTS

    montage = _grid(qapp, tmp_path, keypoints=False)
    dialog = _open(qapp, montage)
    for visual in me.VISUALS:
        tick, show = dialog._rows[visual.key]
        assert tick.isEnabled() is (not visual.needs_keypoints), visual.key
        assert show.isEnabled() is (not visual.needs_keypoints), visual.key
        if visual.needs_keypoints:
            assert tick.toolTip() == show.toolTip() == NO_KEYPOINTS
    assert not dialog._points.isEnabled() and dialog._points.toolTip() == NO_KEYPOINTS
    _close(montage)


def test_keypoint_rows_are_usable_with_keypoints(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    tick, show = dialog._rows["keypoints-trail"]
    assert tick.isEnabled() and show.isEnabled()
    assert drain_until(qapp, lambda: dialog._points.isEnabled(), 20_000)
    assert dialog._points.text() == "all 4 points"
    _close(montage)


def test_picked_only_is_greyed_until_clips_are_picked(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    item = dialog._clips.model().item(1)
    assert not item.isEnabled() and dialog._clips.itemText(1) == "Picked only (0)"
    exports = montage.exports
    exports.start_picking()
    montage._clicked(2, NO_MODIFIER)
    montage._clicked(0, NO_MODIFIER)
    assert item.isEnabled() and dialog._clips.itemText(1) == "Picked only (2)"
    dialog.set_selected_only(True)
    assert [c.rank for c in dialog.clips()] == [3, 1], "in pick order"
    exports.stop_picking()
    assert not dialog.selected_only(), "no picks left: back to every clip"
    _close(montage)


def test_the_pick_bars_export_opens_it_preset_to_the_picks(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    exports = montage.exports
    exports.start_picking()
    montage._clicked(1, NO_MODIFIER)
    bar = exports.pick_bar.export
    assert bar.actions_by_kind["clips"].isEnabled()
    bar.actions_by_kind["clips"].trigger()
    dialog = _loaded(qapp, exports._dialog)
    assert dialog.selected_only()
    assert dialog.clips() == [montage.clips_for_test[1]]
    assert "(1 montage + 1 clip) = 4 files" in dialog.summary()
    _close(montage)


def test_the_preview_is_the_grids_first_clip_and_the_context_widens_it(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    first = montage.clips_for_test[0]
    clip, layers, _tile, fps = dialog.preview_shows()
    assert clip == first
    assert dialog._caption.text().startswith("#1 · mouse_07 · 0:00.0")
    assert layers == montage.exports.layers()
    assert fps == 30.0, "the mp4's rate: it is ticked"
    montage.set_context_s(1.0)
    assert drain_until(
        qapp,
        lambda: dialog.preview_shows()[0].end_t_ns == first.end_t_ns + int(1e9),
        20_000,
    ), "the preview plays the clip and its context, as the file will"
    assert [c.end_t_ns for c in dialog.plan().clips] == [
        c.end_t_ns + int(1e9) for c in montage.clips_for_test
    ]
    assert (
        dialog.plan().labels[0].startswith("#1 · mouse_07 · 0:00.0")
    ), "labelled by the clip's own span"
    _close(montage)


def test_the_preview_plays_the_pressed_row_at_the_exports_own_size(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    dialog._rows["raw-outline"][1].click()
    assert dialog._rows["raw-outline"][1].text() == "Previewing"
    assert dialog._rows[me.CURRENT][1].text() == "Preview"
    _clip, layers, tile, fps = dialog.preview_shows()
    assert layers == me.VISUALS_BY_KEY["raw-outline"].layers()
    assert tile == dialog._native[_clip], "full size: the crop's own size"
    dialog._mp4.setChecked(False)
    _clip, _layers, tile, fps = dialog.preview_shows()
    assert (tile, fps) == (640 - 2 * cm.GAP_PX, 12.0)
    assert "the GIF (MP4 is not ticked)" in dialog._plays_as.text()
    assert drain_until(qapp, lambda: dialog._view.pixmap_size() is not None, 20_000)
    _close(montage)


def test_export_is_greyed_with_a_reason_when_nothing_would_be_written(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    dialog._mp4.setChecked(False)
    dialog._gif.setChecked(False)
    assert not dialog._export.isEnabled()
    assert "Tick MP4 or GIF" in dialog._export.toolTip()
    dialog._gif.setChecked(True)
    dialog._rows[me.CURRENT][0].setChecked(False)
    assert not dialog._export.isEnabled()
    assert "at least one visual" in dialog._export.toolTip()
    _close(montage)


def _frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        count = 0
        while cap.read()[0]:
            count += 1
        return count
    finally:
        cap.release()


def test_two_visuals_two_formats_and_both_write_the_planned_tree(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    dialog._rows["raw-outline"][0].setChecked(True)
    dialog._mp4_width.setCurrentIndex(dialog._mp4_width.findData(320))
    dialog._mp4_fps.setValue(15.0)
    dialog._mp4_codec.set_codec("mp4v")
    dialog._gif_width.setValue(96)
    dialog._gif_fps.setValue(10.0)
    out = tmp_path / "exported"
    dialog.set_folder(out)
    assert drain_until(qapp, lambda: "estimating" not in dialog.summary(), 20_000)
    summary = dialog.summary()
    count = int(summary.split(" = ")[1].split()[0])
    planned = sorted(f.relpath for f in dialog.plan().files())
    dialog.export()
    assert drain_until(qapp, lambda: not dialog.is_exporting(), 60_000)
    on_disk = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*"))
    assert on_disk == planned and len(on_disk) == count == 16
    assert {p.split("/")[0] for p in on_disk} == {
        "current_Mask_Mask_trail",
        "Raw_mask_outline",
    }
    assert (
        "Wrote 16 files" in montage.exports.written.text()
    ), "said on the line over the grid"
    for path in out.rglob("*.gif"):
        with Image.open(path) as gif:
            assert gif.size[0] == 96 and gif.info["duration"] == 100
    for path in out.rglob("*.mp4"):
        cap = cv2.VideoCapture(str(path))
        try:
            assert round(cap.get(cv2.CAP_PROP_FPS)) == 15
            assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
        finally:
            cap.release()
        # 15 fps: the montage and clip 1 run 133 ms (3 pictures); clips 2 and 3 are 33 ms.
        want = 3 if path.name.startswith(("montage", "1_")) else 1
        assert _frames(path) == want, path.name
    _close(montage)


def test_an_export_of_the_picks_ends_pick_mode(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    exports = montage.exports
    exports.start_picking()
    montage._clicked(1, NO_MODIFIER)
    exports.pick_bar.export.actions_by_kind["clips"].trigger()
    dialog = _loaded(qapp, exports._dialog)
    dialog._mp4.setChecked(False)
    dialog._gif_width.setValue(64)
    dialog.set_write(me.WRITE_MONTAGE)
    dialog.set_folder(tmp_path / "picked")
    dialog.export()
    assert drain_until(qapp, lambda: not dialog.isVisible(), 60_000)
    assert not exports.is_picking(), "the picks were for that export"
    assert [p.name for p in (tmp_path / "picked").rglob("*.gif")] == ["montage.gif"]
    _close(montage)


def test_cancel_keeps_the_finished_files_and_removes_the_partial(
    qapp, tmp_path, monkeypatch
):
    import time

    from detail.api import montage_export as api
    from detail.gui import export_clips as module

    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    dialog._gif.setChecked(False)
    out = tmp_path / "stopped"
    dialog.set_folder(out)
    worker = []
    real = api.export_clips

    def cancelling(*args, progress, **kwargs):
        def report(stage, fraction, detail):
            if "file 2 of" in detail:
                while not worker:
                    time.sleep(0.01)
                worker[0].request_cancel()
            progress(stage, fraction, detail)

        return real(*args, progress=report, **kwargs)

    monkeypatch.setattr(module.export_api, "export_clips", cancelling)
    worker.append(dialog.export())
    assert drain_until(qapp, lambda: "Stopped" in dialog._status.text(), 60_000)
    assert "kept the 1 finished file of 4" in dialog._status.text()
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*")) == [
        "current_Mask_Mask_trail/montage.mp4"
    ]
    assert dialog.isVisible() and dialog._export.isEnabled(), "it can be run again"
    _close(montage)


def test_the_folder_is_remembered_per_experiment(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    assert dialog.folder().name == "an_07_cluster_7"
    dialog._mp4.setChecked(False)
    dialog.set_write(me.WRITE_MONTAGE)
    dialog.set_folder(tmp_path / "exports" / "first")
    dialog.export()
    assert drain_until(qapp, lambda: not dialog.isVisible(), 60_000)
    again = _open(qapp, montage)
    assert again is not dialog
    assert again.folder() == tmp_path / "exports" / "an_07_cluster_7"
    _close(montage)


# --- New Query from N clips… --------------------------------------------------------------- #
def test_new_query_opens_the_query_launch_with_the_picks_as_examples(
    qapp, tmp_path, monkeypatch
):
    """Through the real launch dialog: the Query form holds one behaviour whose examples are
    the picked clips, their recording is checked, the embedding the clustering read is chosen
    over another the recording also has, the dialog says what is left — and nothing runs
    until Run is pressed."""
    import shutil

    from detail.gui import launch, live_montage
    from detail.gui.montage_export import ExportSource
    from tests.gui.test_behavior_query import _append_run, _embed_video_experiment

    proj = _embed_video_experiment(tmp_path)
    recording = proj.recordings[0]
    shutil.copytree(
        recording.dir / "embeddings" / "emb_01", recording.dir / "embeddings" / "emb_02"
    )
    _append_run(
        recording.dir,
        "embedding_runs",
        {"id": "emb_02", "status": "complete", "tracking_run_id": "track_01"},
    )
    clips = [
        cm.MontageClip(1, recording.id, recording.name, 0, "emb_02", 0, 4000),
        cm.MontageClip(2, recording.id, recording.name, 0, "emb_02", 2000, 4000),
    ]
    montage = live_montage.LiveMontage(
        aligned_key="interpret/test_query_aligned",
        look_key=None,
        context_key="interpret/test_query_context",
        export=ExportSource(
            clips=lambda: clips,
            experiment_dir=lambda: proj.dir,
            origin=lambda: ORIGIN,
            name=lambda: "Cluster #7",
            empty=lambda: "Nothing here.",
            experiment=lambda: proj,
        ),
    )
    montage.set_clips([live_montage.MontageClip(4000, c.rank, "") for c in clips])
    montage._clock.stop()
    exports = montage.exports
    exports.start_picking()
    montage._clicked(1, NO_MODIFIER)
    prefill = exports.query_prefill()
    (behavior,) = prefill["seed"]["params"]["behaviors"]
    assert behavior["name"] == "cluster 7" and prefill["seed"]["module_id"] == "query"
    assert behavior["examples"] == [
        {"recording_id": recording.id, "track_id": 0, "start_t_ns": 2000,
         "end_t_ns": 4000}
    ], "the pick, its own span"  # fmt: skip
    opened = []
    monkeypatch.setattr(
        launch.LaunchAnalysisDialog,
        "exec",
        lambda self: opened.append(self)
        or launch.LaunchAnalysisDialog.DialogCode.Rejected,
    )
    runs_before = sorted(p.name for p in (proj.dir / "analysis").glob("*"))
    action = exports.pick_bar.export.actions_by_kind["query"]
    assert action.text() == "New Query from 1 clips…" and action.isEnabled()
    action.trigger()
    (dialog,) = opened
    assert dialog._module_id() == "query"
    assert dialog._selected_recordings() == [recording.id]
    config = dialog.config_widget().config()
    assert config["embed_run_id"] == "emb_02", "the embedding the clustering read"
    assert [b["examples"] for b in config["behaviors"]] == [behavior["examples"]]
    assert "press Run" in dialog._status.text()
    assert dialog.run_handle is None
    assert sorted(p.name for p in (proj.dir / "analysis").glob("*")) == runs_before
    dialog.deleteLater()
    _close(montage)
