"""The Plots tab of a location-export result page (§16 "Location plots"), reached the way a user
reaches it: ``AnalysisRunDetailPage`` → its **Results** card (the run's landing page opens a
result view full window) → the module's ``result_widget`` → ``LocationPlotsPanel``.
"""

import pytest

from detail.api.experiment import Experiment
from detail.configs.location_plots import LocationPlotView
from detail.gui._pydantic_form import set_any_field_value
from tests.gui.conftest import drain_busy_calls, drain_until
from tests.location_runs import location_run


@pytest.fixture
def page(qapp, tmp_path):
    from detail.gui.pages import AnalysisRunDetailPage

    proj = Experiment.create("p", tmp_path)
    location_run(
        proj, {f"recording_00{i}": {"pixels_per_mm": 4.0} for i in range(1, 4)}
    )
    proj.set_recording_metadata("recording_001", {"genotype": "wt"})
    proj.set_recording_metadata("recording_002", {"genotype": "mut"})
    page = AnalysisRunDetailPage(proj, "an_01")
    page.resize(1300, 900)
    yield page
    page.close()
    drain_busy_calls(qapp)


def _open(page) -> None:
    """Show the run's page and open its Results view, as a user clicks the card."""
    page.show()
    page._cards["Results"].click()


def _panel(page):
    from detail.gui.location_plots import LocationPlotsPanel

    (panel,) = page.findChildren(LocationPlotsPanel)
    return panel


def _drawn(qapp, panel) -> bool:
    return drain_until(qapp, lambda: panel._picture.pixmap_size() is not None)


def test_nothing_is_read_until_the_page_is_shown(qapp, page):
    panel = _panel(page)
    drain_busy_calls(qapp)
    assert panel._plots is None
    _open(page)
    assert _drawn(qapp, panel)
    assert panel.selected_recordings() == [
        "recording_001",
        "recording_002",
        "recording_003",
    ]


def test_every_view_field_has_a_control_and_reads_back(qapp, page):
    panel = _panel(page)
    assert set(panel._widgets) == set(LocationPlotView.model_fields)
    assert panel.view() == LocationPlotView()


def test_group_columns_come_from_recording_information(qapp, page):
    _open(page)
    panel = _panel(page)
    assert _drawn(qapp, panel)
    combo = panel._widgets["group_by"]
    offered = [combo.itemData(i) for i in range(combo.count())]
    assert offered[0] is None and "genotype" in offered
    assert offered[1:] == panel._plots.group_columns
    combo.setCurrentIndex(offered.index("genotype"))
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "heatmap")
    assert panel.view().group_by == "genotype" and panel.view().kind == "heatmap"


def test_rows_follow_the_plot_kind(qapp, page):
    _open(page)
    panel = _panel(page)
    main = panel._form.main
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "heatmap")
    assert main.isRowVisible(panel._widgets["bins"])
    assert not main.isRowVisible(panel._widgets["elev"])
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "trace3d")
    assert main.isRowVisible(panel._widgets["elev"])
    assert main.isRowVisible(panel._widgets["azim"])
    assert not main.isRowVisible(panel._widgets["bins"])


def test_a_refused_view_is_shown_on_the_panel_not_raised(qapp, page, monkeypatch):
    _open(page)
    panel = _panel(page)
    assert _drawn(qapp, panel)

    def refuse(*_a, **_k):
        raise ValueError("mm needs a distance calibration; not calibrated: x")

    monkeypatch.setattr(panel._plots, "build", refuse)
    panel._draw()
    assert drain_until(qapp, lambda: "not calibrated" in panel._status.text())


def test_unticking_every_recording_says_what_to_do(qapp, page):
    _open(page)
    panel = _panel(page)
    assert _drawn(qapp, panel)
    from detail.gui.recording_filter import RecordingFilter

    (bar,) = panel.findChildren(RecordingFilter)
    bar.clear_shown()
    assert panel.selected_recordings() == []
    assert drain_until(
        qapp, lambda: "Tick at least one recording" in panel._picture.text()
    )


def test_export_writes_the_current_plot(qapp, page, tmp_path):
    _open(page)
    panel = _panel(page)
    assert _drawn(qapp, panel)
    panel.export_to(str(tmp_path / "plot.svg"))
    assert drain_until(qapp, lambda: (tmp_path / "plot.svg").is_file())
    assert drain_until(qapp, lambda: "Wrote" in panel._status.text())


@pytest.fixture
def pair_page(qapp, tmp_path):
    from detail.gui.pages import AnalysisRunDetailPage
    from tests.location_runs import two_animals

    proj = Experiment.create("p", tmp_path)
    location_run(
        proj,
        {
            "recording_001": {
                "rows": two_animals(ids=(0, 1)),
                "subject_labels": {0: "resident", 1: "intruder"},
            },
            "recording_002": {
                "rows": two_animals(ids=(1, 0)),
                "subject_labels": {1: "resident", 0: "intruder"},
            },
        },
    )
    page = AnalysisRunDetailPage(proj, "an_01")
    page.resize(1300, 900)
    yield page
    page.close()
    drain_busy_calls(qapp)


def test_the_animals_list_offers_every_label_and_reads_back_a_choice(qapp, pair_page):
    _open(pair_page)
    panel = _panel(pair_page)
    assert _drawn(qapp, panel)
    listing = panel._widgets["subjects"]
    assert [listing.item(i).text() for i in range(listing.count())] == [
        "intruder",
        "resident",
    ]
    assert panel._form.main.isRowVisible(listing)
    assert panel.view().subjects is None  # every animal ticked is "all"
    from PyQt6.QtCore import Qt

    listing.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert panel.view().subjects == ["resident"]
    listing.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert drain_until(
        qapp, lambda: "Tick at least one animal" in panel._picture.text()
    )


def test_one_animal_everywhere_hides_the_animal_rows(qapp, page):
    _open(page)
    panel = _panel(page)
    assert _drawn(qapp, panel)
    assert not panel._form.main.isRowVisible(panel._widgets["subjects"])
    assert not panel._form.main.isRowVisible(panel._widgets["split_by_subject"])


def test_keypoint_and_pair_rows_appear_only_when_the_run_has_them(qapp, tmp_path):
    from detail.gui.pages import AnalysisRunDetailPage
    from tests.location_runs import circle, keypoint_rows, tracks_rows, two_animals

    proj = Experiment.create("p", tmp_path)
    x, y = circle(40)
    location_run(
        proj,
        {
            "recording_001": {
                "rows": two_animals(n=40),
                "subject_labels": {0: "Intruder", 1: "Resident"},
                "keypoints": keypoint_rows(0, {"nose": (x, y), "tail": (x, y)}),
            }
        },
    )
    page = AnalysisRunDetailPage(proj, "an_01")
    page.resize(1300, 900)
    _open(page)
    panel = _panel(page)
    assert _drawn(qapp, panel)
    main = panel._form.main
    source = panel._widgets["source"]
    assert [source.itemData(i) for i in range(source.count())] == [
        "mask",
        "bbox",
        "nose",
        "tail",
    ]
    assert main.isRowVisible(panel._widgets["points"])  # trace is the opening kind
    assert not main.isRowVisible(panel._widgets["pairs"])
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "distance")
    assert main.isRowVisible(panel._widgets["pairs"])
    assert not main.isRowVisible(panel._widgets["points"])
    pairs = panel._widgets["pairs"]
    assert [pairs.item(i).text() for i in range(pairs.count())] == [
        "Intruder – Resident"
    ]
    assert panel.view().kind == "distance" and panel.view().pairs is None
    page.close()
    drain_busy_calls(qapp)


def test_export_writes_an_interactive_page(qapp, page, tmp_path):
    _open(page)
    panel = _panel(page)
    assert _drawn(qapp, panel)
    panel.export_to(str(tmp_path / "plot.html"))
    assert drain_until(qapp, lambda: (tmp_path / "plot.html").is_file())
