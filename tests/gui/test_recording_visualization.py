"""INTERPRET's **Visualize recordings…** and the window it opens (§22 "Recording visualization"),
through the real entry point: the button beside Find, the window loading a range off the UI
thread, the cut's numbered list, saving a recipe, and the Export dialog measuring before it writes.
"""

import numpy as np

from detail.configs.recording_visualization import RecordingVisualizationConfig as C
from detail.gui._pydantic_form import set_any_field_value
from tests.gui.conftest import drain_busy_calls, drain_until


def _workspace(experiment, run):
    from detail.gui.cluster_workspace import TEMPORAL_TERMS, ClusterWorkspace

    return ClusterWorkspace(experiment, run, TEMPORAL_TERMS)


def _interpret(qapp, workspace):
    from detail.gui.node_workspace import NodeInterpretTab

    tabs = workspace.findChildren(NodeInterpretTab)
    assert tabs, "the workspace builds INTERPRET"
    return tabs[0]


def test_the_button_sits_directly_right_of_find(qapp, visualizable_run):
    experiment, run = visualizable_run
    ws = _workspace(experiment, run)
    tab = _interpret(qapp, ws)
    row = tab._find._row
    assert row.indexOf(tab._visualize) == row.indexOf(tab._find.find) + 1
    assert tab._visualize.text() == "Visualize recordings…"


def _open(qapp, visualizable_run):
    experiment, run = visualizable_run
    ws = _workspace(experiment, run)
    tab = _interpret(qapp, ws)
    tab._visualize.click()
    window = tab._visualization_window
    assert window is not None and not window.isModal()
    tab._visualize.click()
    assert tab._visualization_window is window, "a second click raises the open window"
    set_any_field_value(C, "start_s", window._range["start_s"], 0.5)
    set_any_field_value(C, "end_s", window._range["end_s"], 2.0)
    window._reload()
    assert drain_until(
        qapp, lambda: window._viz is not None, 60_000
    ), window._status.text()
    return ws, run, window


def test_the_window_loads_a_range_and_draws_it(qapp, visualizable_run):
    _ws, _run, window = _open(qapp, visualizable_run)
    d = window._viz.data
    assert window._scrubber._total == d.n_frames
    assert drain_until(qapp, lambda: window._view.pixmap_size() is not None, 30_000)
    assert window._save_btn.isEnabled() and window._export_btn.isEnabled()
    assert "principal components" in window._status.text()


def test_a_cut_numbers_the_clusters_and_colours_at_most_eight(qapp, visualizable_run):
    _ws, run, window = _open(qapp, visualizable_run)
    window._cut.setChecked(True)
    drain_busy_calls(qapp)
    assert window._numbers.count() >= 1
    ticked = window._ticked()
    assert 1 <= len(ticked) <= 8
    recipe = window.recipe()
    assert recipe.clusters == "cut" and recipe.cut_included == ticked
    assert drain_until(
        qapp, lambda: window._viz.data.names == [f"cluster {n}" for n in ticked], 30_000
    )


def test_save_lists_the_recipe_and_the_run_keeps_it(qapp, visualizable_run):
    _ws, run, window = _open(qapp, visualizable_run)
    window._save()
    assert window._saved.count() == 1
    assert [r["id"] for r in run.visualizations()] == ["viz_001"]
    assert window._open_btn.isEnabled()


def test_export_dialog_measures_before_it_writes(qapp, visualizable_run):
    from detail.gui.recording_visualization import ExportVisualizationDialog

    _ws, run, window = _open(qapp, visualizable_run)
    dialog = ExportVisualizationDialog(
        window._experiment,
        current=(run, window.recipe()),
        themes=["dark", "light"],
        parent=window,
    )
    assert dialog._one.isChecked()
    assert drain_until(
        qapp, lambda: "MB per file" in dialog._estimate.text(), 60_000
    ), dialog._estimate.text()
    assert "2 files" in dialog._estimate.text()
    settings = dialog.export_settings()
    assert settings.themes == ["dark", "light"] and settings.speed == "real-time"
    assert "1 x 1" in dialog._layout_note.text()


def test_the_window_scrubs_to_a_new_frame(qapp, visualizable_run):
    _ws, _run, window = _open(qapp, visualizable_run)
    window._scrubber.set_frame(min(5, window._viz.data.n_frames - 1))
    assert drain_until(
        qapp, lambda: window._scrubber.frame() == min(5, window._viz.data.n_frames - 1)
    )
    assert np.isfinite(window._viz.data.xy).any()
