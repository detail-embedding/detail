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


def test_a_newer_draw_stops_the_one_it_replaces(qapp, page, monkeypatch):
    """A control change mid-draw stops that draw at its next panel, not after its last.

    Left to finish, every superseded draw ran to its end: on a 50-recording run each was 2–8 s,
    so ticking recordings off or stepping through *Group by* queued a whole draw per change and
    the picture answered seconds after the last one (``tests/gui/test_busy.py``, class 4).
    The draw reports per panel (``plot_render.figure``), and the newer draw stops it there.
    """
    import threading
    import time

    from detail.report import plot_render

    panel = _panel(page)
    _open(page)
    assert _drawn(qapp, panel)
    real = plot_render._draw_panel
    drawn = []
    started = threading.Event()

    def slow_panel(*args):
        drawn.append(args[2].title)
        started.set()
        time.sleep(0.2)
        real(*args)

    monkeypatch.setattr(plot_render, "_draw_panel", slow_panel)
    panel._draw()
    assert started.wait(10), "the first draw never reached a panel"
    panel._draw()  # a control changed while the first was drawing
    drain_busy_calls(qapp)
    assert drain_until(qapp, lambda: len(drawn) >= 3)
    assert (
        3 < len(drawn) < 6
    ), f"{len(drawn)} panels drawn for two draws of 3: the replaced draw " + (
        "ran to its end" if len(drawn) >= 6 else "never started"
    )


@pytest.mark.parametrize("ratio", [1.0, 1.25])
def test_a_grouped_plot_is_drawn_at_the_size_its_pane_shows_it(
    qapp, page, monkeypatch, ratio
):
    """A plot grouped into one panel was drawn at a fixed 110 dpi, 385×232, and stretched to
    fill the pane. It is now drawn at the pane's size in the screen's own pixels (``ratio`` is a
    display scaled to 125%), with ``ZOOM_HEADROOM`` to zoom into."""
    from detail.report import plot_render

    panel = _panel(page)
    monkeypatch.setattr(panel._picture, "devicePixelRatioF", lambda: ratio)
    _open(page)
    assert _drawn(qapp, panel)
    first = panel._picture.pixmap_size()
    set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "speed")
    panel._widgets["group_by"].setCurrentIndex(1)
    panel._draw()
    assert drain_until(qapp, lambda: panel._picture.pixmap_size() != first)
    drain_until(qapp, lambda: not panel._timer.isActive())
    drain_busy_calls(qapp)
    drawn = panel._picture.pixmap_size()
    pane_w, pane_h = panel.pane_pixels()
    assert (pane_w, pane_h) == (
        panel._picture.width() * ratio,
        panel._picture.height() * ratio,
    )
    assert max(drawn.width() / pane_w, drawn.height() / pane_h) == pytest.approx(
        plot_render.ZOOM_HEADROOM, abs=0.02
    )


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


def test_keypoint_rows_appear_only_when_the_run_has_keypoints(qapp, page, tmp_path):
    from PyQt6.QtCore import Qt

    from detail.gui.pages import AnalysisRunDetailPage
    from tests.location_fixtures import write_keypoints

    _open(page)
    bare = _panel(page)
    assert _drawn(qapp, bare)
    assert not _row_shown(bare, "points")  # this run read no keypoints

    proj = Experiment.create("k", tmp_path / "k")
    walk = _walk()
    ids = finished_run(proj, {"a": {0: walk}})
    write_keypoints(
        proj.analysis_run_dir("an_01") / ids["a"],
        walk,
        {"nose": (5.0, 0.0, None), "tail_base": (-5.0, 0.0, None)},
    )
    other = AnalysisRunDetailPage(proj, "an_01")
    try:
        _open(other)
        panel = _panel(other)
        assert _drawn(qapp, panel)
        source = panel._widgets["source"]
        assert [source.itemText(i) for i in range(source.count())] == [
            "body", "mask", "bbox", "nose", "tail_base",
        ]  # fmt: skip
        assert _row_shown(panel, "points")
        panel._widgets["points"].item(1).setCheckState(Qt.CheckState.Checked)
        assert panel.view().points == ["tail_base"]
        set_any_field_value(LocationPlotView, "kind", panel._widgets["kind"], "heatmap")
        panel._sync_rows()
        assert not _row_shown(panel, "points")  # several points belong on a path
    finally:
        other.close()
        drain_busy_calls(qapp)


def test_export_writes_an_interactive_page(qapp, page, tmp_path):
    from detail.gui.location_plots import EXPORT_FILTERS

    assert "Interactive HTML (*.html)" in EXPORT_FILTERS
    panel = _panel(page)
    _open(page)
    assert _drawn(qapp, panel)
    panel.export_to(str(tmp_path / "paths.html"))
    assert drain_until(qapp, lambda: (tmp_path / "paths.html").is_file())
