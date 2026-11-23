"""The Plots view of a location-export result page (§16 "Location plots"), reached the way a user
reaches it: ``AnalysisRunDetailPage`` → its **Plots** card → ``LocationPlotsPanel``.

**Model ↔ control drift.** Every ``LocationPlotView`` field has a control built from the model and
reads back through ``view()`` (``test_every_view_field_has_a_control_and_reads_back``): a field
added to the model without a control, or a control that drops its value, fails there.
"""

import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.configs.location_plots import LocationPlotView
from detail.gui._pydantic_form import set_any_field_value
from tests.gui.conftest import drain_busy_calls, drain_until
from tests.location_fixtures import finished_run, track


def _walk(offset=0.0):
    n = 150
    return track(
        10.0 + offset + np.arange(n) * 2.0, 40.0 + 10 * np.sin(np.arange(n) / 10)
    )


@pytest.fixture
def page(qapp, tmp_path):
    from detail.gui.pages import AnalysisRunDetailPage

    proj = Experiment.create("p", tmp_path)
    finished_run(
        proj,
        {"a": {0: _walk()}, "b": {0: _walk(5)}, "c": {0: _walk(9)}},
        metadata={"a": {"genotype": "wt"}, "b": {"genotype": "mut"}},
    )
    page = AnalysisRunDetailPage(proj, "an_01")
    page.resize(1300, 900)
    yield page
    page.close()
    drain_busy_calls(qapp)


def _open(page) -> None:
    """Show the run's page and open its Plots view, as a user clicks the card."""
    page.show()
    page._cards["Plots"].click()


def _panel(page):
    from detail.gui.location_plots import LocationPlotsPanel

    (panel,) = page.findChildren(LocationPlotsPanel)
    return panel


def _drawn(qapp, panel) -> bool:
    return drain_until(qapp, lambda: panel._picture.pixmap_size() is not None)


def _row_shown(panel, name) -> bool:
    return panel._form.main.isRowVisible(panel._widgets[name])


def test_nothing_is_read_until_the_page_is_shown(qapp, page):
    panel = _panel(page)
    drain_busy_calls(qapp)
    assert panel._plots is None
    _open(page)
    assert _drawn(qapp, panel)
    assert [panel._recordings.item(i).text() for i in range(3)] == ["a", "b", "c"]
    assert len(panel.selected_recordings()) == 3


def test_every_view_field_has_a_control_and_reads_back(qapp, page):
    panel = _panel(page)
    assert set(panel._widgets) == set(LocationPlotView.model_fields)
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "body_hist")
    set_any_field_value(
        LocationPlotView, "body_measure", panel._widgets["body_measure"], "width"
    )
    set_any_field_value(LocationPlotView, "hist_bins", panel._widgets["hist_bins"], 17)
    panel._widgets["elev"].set_value(40.0)
    view = panel.view()
    assert (view.kind, view.body_measure, view.hist_bins, view.elev) == (
        "body_hist", "width", 17, 40.0,
    )  # fmt: skip
    assert view.posture_every_s is None and view.group_by is None


def test_group_columns_come_from_recording_information(qapp, page):
    panel = _panel(page)
    _open(page)
    assert _drawn(qapp, panel)
    combo = panel._widgets["group_by"]
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "(each recording)",
        "genotype",
    ]
    combo.setCurrentIndex(1)
    assert panel.view().group_by == "genotype"


@pytest.mark.parametrize(
    "kind, shown, hidden",
    [
        ("trace", ["source", "color_by", "posture_every_s"], ["body_measure", "bins"]),
        ("heatmap", ["source", "bins"], ["color_by", "posture_every_s", "elev"]),
        ("speed", ["time_bin_s"], ["posture_every_s", "body_measure"]),
        ("trace3d", ["elev", "azim"], ["posture_every_s"]),
        ("body", ["body_measure", "time_bin_s"], ["source", "hist_bins"]),
        ("body_hist", ["body_measure", "hist_bins"], ["time_bin_s", "color_by"]),
    ],
)
def test_rows_follow_the_plot_kind(qapp, page, kind, shown, hidden):
    panel = _panel(page)
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], kind)
    panel._sync_rows()
    assert all(
        _row_shown(panel, name) for name in shown if name not in ("elev", "azim")
    )
    assert not any(
        _row_shown(panel, name) for name in hidden if name not in ("elev", "azim")
    )


def test_a_run_from_before_the_body_hides_the_body_rows(qapp, tmp_path):
    from detail.gui.pages import AnalysisRunDetailPage

    proj = Experiment.create("p", tmp_path)
    finished_run(proj, {"a": {0: _walk()}}, body=False)
    page = AnalysisRunDetailPage(proj, "an_01")
    try:
        _open(page)
        panel = _panel(page)
        assert _drawn(qapp, panel)
        assert not panel._has_body
        assert not _row_shown(panel, "posture_every_s")  # a path plot, but no body
    finally:
        page.close()
        drain_busy_calls(qapp)


def test_a_refused_view_is_shown_on_the_panel_not_raised(qapp, page):
    panel = _panel(page)
    _open(page)
    assert _drawn(qapp, panel)
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "body")
    set_any_field_value(LocationPlotView, "units", panel._widgets["units"], "norm")
    panel._draw()
    assert drain_until(qapp, lambda: "px or mm" in panel._status.text())


def test_unticking_every_recording_says_what_to_do(qapp, page):
    from PyQt6.QtCore import Qt

    panel = _panel(page)
    _open(page)
    assert _drawn(qapp, panel)
    for i in range(panel._recordings.count()):
        panel._recordings.item(i).setCheckState(Qt.CheckState.Unchecked)
    panel._draw()
    assert panel._picture.text() == "Tick at least one recording to plot."


def test_export_writes_the_current_plot(qapp, page, tmp_path):
    panel = _panel(page)
    _open(page)
    assert _drawn(qapp, panel)
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "body")
    panel.export_to(str(tmp_path / "body.svg"))
    assert drain_until(qapp, lambda: (tmp_path / "body.svg").is_file())
    assert drain_until(qapp, lambda: panel._status.text().startswith("Wrote "))
