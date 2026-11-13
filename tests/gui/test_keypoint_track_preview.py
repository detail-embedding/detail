"""*Preview tracking…* (§26 *Previewing a track method*): the track form's button, and the window
that runs settings over short windows, plays them side by side and hands one back.

The window is driven end to end over a real recording (``test_keypoint_track_preview``'s moving
box, with a real detect run and a decodable video): the preview runs through
``Experiment.preview_keypoint_tracking`` off the UI thread, the panes show decoded frames with
each setting's points, and the table describes every setting with detection alone first.
"""

import numpy as np
import pytest

from detail.gui import keypoint_track_preview as window
from detail.gui.keypoint_track_preview import KeypointTrackPreviewDialog
from detail.gui.modules.registry import GUI_MODULES
from detail.pipeline import keypoint_track_preview as preview
from tests.gui.conftest import drain_until
from tests.pipeline.test_keypoint_track_preview import moving  # noqa: F401 — fixture
from tests.pipeline.test_keypoint_track_preview import _launch

SPEC = {"identity": {"model_id": "dinov3_match"}}


class _Recording:
    def __init__(self, rid):
        self.id = rid


class _Experiment:
    dir = (
        None  # the form looks a method up through it; the built-in ones need no folder
    )


@pytest.fixture
def form(qapp):
    return GUI_MODULES["keypoint_track"].config_widget()


def test_the_button_says_why_until_it_can_preview(form, monkeypatch):
    assert not form._preview.isEnabled()
    assert form._preview.toolTip() == "Open an experiment first"
    form.on_experiment_set(_Experiment())
    assert form._preview.toolTip() == "Choose a keypoint detect run first"
    monkeypatch.setattr(form._picker, "spec", lambda: SPEC)
    form._sync_tune()  # what choosing a detect run in the picker does
    assert "No selected recording" in form._preview.toolTip()
    form._recordings = [_Recording("recording_001")]
    monkeypatch.setattr(form, "covered_recording_ids", lambda: {"recording_001"})
    form._sync_tune()
    # Unlike Tune, a preview needs no labels and runs either method.
    form._method.setCurrentIndex(form._method.findData("despike_smooth"))
    assert form._preview.isEnabled()


def test_the_window_gets_the_launch_config_and_its_choice_fills_the_form(
    form, monkeypatch
):
    import detail.gui.keypoint_track_preview as window

    opened = {}

    class _Dialog:
        def __init__(self, experiment, config, recordings, parent):
            opened.update(config=config, recordings=recordings)
            self.chosen_config = {**config["config"], "transition_sigma_body": 3.0}

        def exec(self):
            return 1

    monkeypatch.setattr(window, "KeypointTrackPreviewDialog", _Dialog)
    monkeypatch.setattr(form._picker, "spec", lambda: SPEC)
    monkeypatch.setattr(form, "covered_recording_ids", lambda: {"recording_002"})
    form.on_experiment_set(_Experiment())
    form._recordings = [_Recording("recording_001"), _Recording("recording_002")]
    form._recording_names = {"recording_002": "mouse two"}
    form._sync_tune()
    form._open_preview()
    # the form's own launch config, and only the recordings it covers, by name
    assert opened["config"] == form.config() | {"config": opened["config"]["config"]}
    assert opened["config"]["keypoint_detect_spec"] == SPEC
    assert opened["recordings"] == [("recording_002", "mouse two")]
    assert form.config()["config"]["transition_sigma_body"] == 3.0


def test_the_window_previews_plays_the_settings_side_by_side_and_hands_one_back(
    qapp, moving  # noqa: F811 — the imported fixture
):
    it, _recording = moving
    dialog = KeypointTrackPreviewDialog(
        it.experiment, _launch(), [("recording_002", "mouse two")]
    )
    combo, start = dialog._slots[0]
    assert combo.currentData() == "recording_002"
    assert (
        dialog._slots[1][0].currentData() is None
    )  # one recording offered, one ticked
    assert start.specialValueText() == "middle"
    start.setValue(3.0)
    dialog._duration.setValue(2.0)
    # The variants start where PREVIEW_STARTING_VARIANTS puts them, as changes from current.
    assert dialog.variants() == [
        {"transition_sigma_body": 1.0},
        {"transition_sigma_body": 3.0, "smooth": False, "despike": False},
    ]
    dialog._variants[0][0].setChecked(False)
    assert len(dialog.variants()) == 1
    # B alone is still B: the tile's name is its column's, not its place among those ticked
    assert [letter for letter, _c in dialog._lettered_variants()] == ["B"]
    dialog._variants[0][0].setChecked(True)
    assert not dialog._use.isEnabled()
    dialog.show()
    dialog.resize(1100, 900)
    dialog.run_preview()
    assert drain_until(qapp, lambda: dialog._table_model.rowCount() == 4, 60_000)
    names = [s.name for s in dialog._result.settings]
    assert names[:2] == [preview.DETECTION_ONLY, preview.CURRENT]
    # Each row is named as the form names it, so a tile, its variant column and its row of
    # figures read as one.
    rows = [dialog._table_model.item(r, 0).text() for r in range(4)]
    assert rows == [
        "Detection only",
        "Current settings",
        "Variant A: transition 1",
        "Variant B: transition 3, smooth off, despike off",
    ]
    assert "3.0–5.0 s" in dialog._summary.text()
    # One frame, once per setting, each tile with that setting's points drawn on the
    # footage, two tiles to a row in one picture, a rule between them.
    picture, rects, tile_names = dialog._compute(0)
    raw = dialog._frames["recording_002"].read(
        dialog._frames["recording_002"].first_frame
    )
    h, w = raw.shape[:2]
    gap = window._GAP
    assert picture.shape == (2 * h + gap, 2 * w + gap, 3)
    assert rects == [
        (c * (w + gap), r * (h + gap), w, h) for r in range(2) for c in range(2)
    ]
    tiles = [picture[y : y + h, x : x + w] for x, y, _w, _h in rects]
    assert all(not np.array_equal(tile, raw) for tile in tiles)  # points on each
    # Each tile is named as its row is, and the view is handed the names and where each
    # tile lies, so it zooms them together and draws the names at a size the screen reads.
    assert tile_names == rows
    # Frame 100, ten into the window, is a box jump: detection follows it, the current
    # settings call both points missing there, so their tiles differ.
    jump, _rects, _names = dialog._compute(10)
    assert not np.array_equal(jump[:h, :w], jump[:h, w + gap :])
    assert drain_until(qapp, lambda: dialog._view._pixmap is not None)
    assert dialog._view.zoomable and dialog._view.tile_names() == rows
    # the controls sit under the picture they drive (PictureWithControls)
    assert dialog._scrubber.parentWidget() is dialog._player
    dialog._table.selectRow(0)  # detection alone is not a tracking setting
    assert not dialog._use.isEnabled() and "Detection alone" in dialog._use.toolTip()
    dialog._table.selectRow(3)
    assert dialog._use.isEnabled()
    dialog._take()
    assert dialog.chosen_config["transition_sigma_body"] == 3.0
    assert dialog.chosen_config["smooth"] is False
    dialog.close()


def test_the_grid_search_lists_the_front_first_and_plays_the_selected_row(
    qapp, moving  # noqa: F811 — the imported fixture
):
    it, _recording = moving
    dialog = KeypointTrackPreviewDialog(
        it.experiment, _launch(), [("recording_002", "mouse two")]
    )
    dialog._slots[0][1].setValue(3.0)
    dialog._duration.setValue(2.0)
    assert dialog._search.isEnabled()
    dialog.show()
    dialog.resize(1100, 900)
    dialog.search_grid()
    assert drain_until(qapp, lambda: dialog._table_model.rowCount() == 38, 60_000)
    result = dialog._result
    assert result.grid and result.front
    model = dialog._table_model
    rows = [model.item(r, 0).text() for r in range(model.rowCount())]
    assert rows == [dialog.label(s.name) for s in result.ranked()]
    assert rows[:2] == ["Detection only", "Current settings"]
    marked = [model.item(r, 0).text() for r in range(38) if model.item(r, 1).text()]
    assert marked == [n for n in rows if n in result.front]
    assert not dialog._table.isColumnHidden(1) and dialog._front_note.isVisibleTo(
        dialog
    )
    # The first front row is selected, and the picture is three tiles: detection, current,
    # and that row, on the same frame.
    first = result.ranked()[2]
    assert dialog._selected().name == first.name
    assert [s.name for s in dialog._tiles] == [
        preview.DETECTION_ONLY,
        preview.CURRENT,
        first.name,
    ]
    raw = dialog._frames["recording_002"].read(
        dialog._frames["recording_002"].first_frame
    )
    h, w = raw.shape[:2]
    picture, rects, tile_names = dialog._compute(0)
    assert picture.shape == (2 * h + window._GAP, 2 * w + window._GAP, 3)
    assert len(rects) == 3
    assert tile_names == [
        "Detection only",
        "Current settings",
        f"Selected row: {first.name}",
    ]
    # Selecting another row plays that setting instead.
    other = rows.index("transition 0.15, smooth off, despike off")
    dialog._table.selectRow(other)
    assert dialog._tiles[2].name == "transition 0.15, smooth off, despike off"
    # The figures switch keeps the selection.
    dialog._figures_combo.setCurrentIndex(1)
    assert dialog._selected().name == "transition 0.15, smooth off, despike off"
    dialog._take()
    assert dialog.chosen_config["transition_sigma_body"] == 0.15
    dialog.close()


def test_the_grid_search_is_offered_only_for_temporal_decode(qapp):
    dialog = KeypointTrackPreviewDialog(
        _Experiment(), _launch(method="despike_smooth"), [("recording_002", "two")]
    )
    assert not dialog._search.isEnabled()
    assert "temporal_decode" in dialog._search.toolTip()
    dialog.close()
