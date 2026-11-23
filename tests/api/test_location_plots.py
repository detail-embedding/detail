"""``Experiment.location_plots`` — a location-export run drawn from its own CSVs (§16).

Cases come from the edges of what the plots consume (``schema-edge-tests``): ``tracks.csv`` as the
export writes it (the real runner's output, one row, several animals, no body columns),
``locations.yaml`` (calibrated or not, differing resolutions, none recorded), and the run around
them (a failed recording, a deleted one, a blank or unknown group column, another module's run).
"""

import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.configs.location_plots import LocationPlotView
from detail.storage import location_export
from tests.location_fixtures import finished_run, track

KINDS = ("trace", "heatmap", "speed", "trace3d", "body", "body_hist")


@pytest.fixture
def proj(tmp_path):
    return Experiment.create("p", tmp_path)


def _walk(speed=20.0, seconds=30, **kwargs):
    """A walk along x at ``speed`` mm/s, five rows a second, starting 10 mm in."""
    return track(
        10.0 + np.arange(seconds * 5) * speed / 5.0,
        np.full(seconds * 5, 50.0),
        **kwargs,
    )


def test_plots_read_a_run_the_real_export_wrote(tmp_path):
    """The producer end to end: the location-export runner's own CSVs (one tracked detection,
    every column the export writes) plot with no hand-built file in between."""
    from tests.pipeline.test_location_analysis import (
        _add_tracked_recording,
        _run_directly,
    )

    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    _run_directly(proj, "recording_001")
    plots = proj.location_plots("an_01")
    assert plots.recording_ids == ["recording_001"] and plots.body_measured()
    for kind in KINDS:
        spec = plots.build(LocationPlotView(kind=kind, units="px"))
        assert len(spec.panels) == 1, kind


def test_only_completed_recordings_are_offered(proj):
    ids = finished_run(proj, {"a": {0: _walk()}, "b": {0: _walk()}}, failed=("b",))
    plots = proj.location_plots("an_01")
    assert plots.recording_ids == [ids["a"]]
    assert plots.names() == {ids["a"]: "a"}


def test_a_run_with_nothing_complete_refuses(proj):
    finished_run(proj, {"a": {0: _walk()}}, failed=("a",))
    with pytest.raises(ValueError, match="no completed recording"):
        proj.location_plots("an_01")


def test_another_module_s_run_refuses(proj):
    from detail.storage import analysis_runs

    run_dir = analysis_runs.analysis_dir(proj.dir, "an_02")
    analysis_runs.write_analysis_yaml(run_dir, {"id": "an_02", "module_id": "query"})
    with pytest.raises(ValueError, match="not a location export"):
        proj.location_plots("an_02")


def test_a_recording_with_several_animals_is_named_and_the_rest_plot(proj):
    finished_run(proj, {"one": {0: _walk()}, "two": {0: _walk(), 3: _walk()}})
    spec = proj.location_plots("an_01").build(LocationPlotView())
    assert [p.title for p in spec.panels] == ["one"]
    assert "two: 2 animals (tracks 0, 3)" in spec.note


def test_auto_units_follow_calibration_and_mm_refuses_an_uncalibrated_one(tmp_path):
    calibrated = Experiment.create("c", tmp_path / "c")
    finished_run(calibrated, {"a": {0: _walk()}})
    assert (
        calibrated.location_plots("an_01").build(LocationPlotView()).x.label == "x (mm)"
    )
    bare = Experiment.create("b", tmp_path / "b")
    finished_run(bare, {"a": {0: _walk()}}, pixels_per_mm=None)
    plots = bare.location_plots("an_01")
    assert plots.build(LocationPlotView()).x.label == "x (px)"
    with pytest.raises(ValueError, match="not calibrated: a"):
        plots.build(LocationPlotView(units="mm"))


def test_speed_is_read_in_the_run_s_millimetres(proj):
    finished_run(proj, {"a": {0: _walk(speed=20.0)}})
    spec = proj.location_plots("an_01").build(LocationPlotView(kind="speed"))
    speed = spec.panels[0].lines[0].y
    assert np.nanmedian(speed) == pytest.approx(20.0, rel=0.02)
    assert spec.y.label == "speed (mm/s)"


def test_body_plots_read_the_run_s_body(proj):
    finished_run(proj, {"a": {0: _walk(length=180.0, width=60.0)}})
    plots = proj.location_plots("an_01")
    spec = plots.build(LocationPlotView(kind="body", body_measure="elongation"))
    assert np.nanmedian(spec.panels[0].lines[0].y) == pytest.approx(3.0)
    spec = plots.build(LocationPlotView(kind="body", body_measure="mask_area"))
    assert np.nanmedian(spec.panels[0].lines[0].y) == pytest.approx(
        180.0 * 60.0 * 0.8 * 1.25, rel=1e-3
    )


def test_a_run_made_before_the_body_plots_its_path_but_not_its_body(proj):
    finished_run(proj, {"a": {0: _walk()}}, body=False)
    plots = proj.location_plots("an_01")
    assert not plots.body_measured()
    assert len(plots.build(LocationPlotView()).panels) == 1  # the mask's centre
    with pytest.raises(ValueError, match="a: exported before the body was measured"):
        plots.build(LocationPlotView(kind="body"))


def test_group_by_reads_recording_information_and_keeps_blank_values(proj):
    finished_run(
        proj,
        {"a": {0: _walk()}, "b": {0: _walk()}, "c": {0: _walk()}},
        metadata={
            "a": {"genotype": "wt"},
            "b": {"genotype": "ko"},
            "c": {"genotype": ""},
        },
    )
    spec = proj.location_plots("an_01").build(
        LocationPlotView(kind="heatmap", group_by="genotype")
    )
    assert [p.title for p in spec.panels] == [
        "ko (n=1)",
        "wt (n=1)",
        "(no value) (n=1)",
    ]


def test_an_unknown_group_column_names_the_columns_there_are(proj):
    finished_run(proj, {"a": {0: _walk()}}, metadata={"a": {"genotype": "wt"}})
    with pytest.raises(ValueError, match=r"'strain' \(columns: genotype\)"):
        proj.location_plots("an_01").build(LocationPlotView(group_by="strain"))


def test_a_deleted_recording_still_plots_under_its_id(proj):
    ids = finished_run(proj, {"a": {0: _walk()}})
    proj.delete_recording(ids["a"])
    spec = proj.location_plots("an_01").build(LocationPlotView())
    assert [p.title for p in spec.panels] == [ids["a"]]


def test_differing_resolutions_share_one_axis_range(tmp_path):
    proj = Experiment.create("p", tmp_path)
    finished_run(proj, {"a": {0: _walk()}}, run_id="an_01", frame=(640, 480))
    spec = proj.location_plots("an_01").build(LocationPlotView(units="px"))
    assert spec.x.range == (0.0, 640.0) and spec.y.range == (0.0, 480.0)


def test_a_manifest_without_a_resolution_refuses(proj):
    ids = finished_run(proj, {"a": {0: _walk()}})
    run_dir = proj.analysis_run_dir("an_01") / ids["a"]
    manifest = location_export.read_location_manifest(run_dir)
    manifest.pop("video_resolution")
    import yaml

    (run_dir / location_export.MANIFEST_FILE).write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="records no video_resolution"):
        proj.location_plots("an_01").build(LocationPlotView())


def test_a_second_plot_rereads_nothing(proj, monkeypatch):
    finished_run(proj, {"a": {0: _walk()}})
    plots = proj.location_plots("an_01")
    plots.build(LocationPlotView())
    from detail.analysis import location_plots as lp

    def fail(*_args, **_kwargs):
        raise AssertionError("read again")

    monkeypatch.setattr(lp, "read_tracks", fail)
    plots.build(LocationPlotView(kind="speed"))


def test_build_and_export_report_reading_measuring_drawing_then_writing(proj, tmp_path):
    finished_run(proj, {"a": {0: _walk()}, "b": {0: _walk()}})
    stages = []
    paths = proj.location_plots("an_01").export(
        LocationPlotView(kind="trace"),
        tmp_path / "out" / "path",
        formats=("png", "svg", "pdf"),
        progress=lambda stage, fraction, detail: stages.append((stage, fraction)),
    )
    assert [p.suffix for p in paths] == [".png", ".svg", ".pdf"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    seen = list(dict.fromkeys(s for s, _ in stages))
    assert seen == ["reading", "measuring", "drawing", "writing"]
    # A panel per recording, each reported as it is drawn.
    assert [f for s, f in stages if s == "drawing"] == [0.0, 0.5]
    assert stages[-1] == ("writing", 1.0)


def test_export_writes_an_interactive_page_beside_the_pictures(proj, tmp_path):
    finished_run(proj, {"a": {0: _walk()}})
    paths = proj.location_plots("an_01").export(
        LocationPlotView(kind="trace3d"), tmp_path / "paths", formats=("png", "html")
    )
    assert [p.name for p in paths] == ["paths.png", "paths.html"]
    page = paths[1].read_text(encoding="utf-8-sig")
    assert "<title>Path over time</title>" in page
    assert "<script src=" not in page  # plotly.js is embedded: the page opens offline


def test_an_unknown_export_format_is_refused_before_anything_is_read(proj, tmp_path):
    finished_run(proj, {"a": {0: _walk()}})
    with pytest.raises(ValueError, match="png, svg, pdf, html; not gif"):
        proj.location_plots("an_01").export(
            LocationPlotView(), tmp_path / "x", formats=("gif",)
        )


def test_export_locations_reports_its_rows_and_its_write(tmp_path):
    """``Recording.export_locations`` measures every tracked row; without ``progress`` it sat
    silent for minutes (the long-op rule: CLAUDE.md, ``tests/test_progress_callers.py``).
    """
    from tests.analysis.test_locations import _one_track

    recording = _one_track(tmp_path)
    stages = []
    out = recording.export_locations(
        "track_01",
        tmp_path / "out",
        progress=lambda stage, fraction, detail: stages.append(
            (stage, fraction, detail)
        ),
    )
    assert (out / location_export.TRACKS_FILE).is_file()
    assert stages[0][:2] == ("measuring", 0.0)
    assert ("measuring", 1.0) in [s[:2] for s in stages]
    assert stages[-1][:2] == ("writing", 1.0) and stages[-1][2] == out.name
