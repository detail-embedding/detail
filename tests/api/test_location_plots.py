"""``Experiment.location_plots`` — a location-export run drawn from its own CSVs (§16).

Cases come from the edges of what the plots consume: ``tracks.csv`` as the export writes it (one
row, several subjects, a gap, no rows, any column order), ``locations.yaml`` (calibrated or not,
differing resolutions), and the run around them (a failed recording, a deleted one, a blank or
unknown group column).
"""

import csv

import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.configs.location_plots import LocationPlotView
from detail.storage import analysis_runs as ar
from tests.location_runs import (
    circle,
    location_run,
    tracks_rows,
    two_animals,
    write_recording,
)


@pytest.fixture
def proj(tmp_path):
    return Experiment.create("p", tmp_path)


def test_plots_read_a_run_the_real_export_wrote(tmp_path):
    """The producer end to end: the location-export runner's own CSVs (one tracked detection)
    plot without a hand-built fixture in between."""
    from tests.pipeline.test_location_analysis import (
        _add_tracked_recording,
        _run_directly,
    )

    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    _run_directly(proj, "recording_001")
    plots = proj.location_plots("an_01")
    assert plots.recording_ids == ["recording_001"]
    for kind in ("trace", "heatmap", "speed", "trace3d"):
        spec = plots.build(LocationPlotView(kind=kind))
        assert len(spec.panels) == 1


def test_only_completed_recordings_are_offered(proj):
    location_run(proj, {"recording_001": {}, "recording_002": {"status": "failed"}})
    plots = proj.location_plots("an_01")
    assert plots.recording_ids == ["recording_001"]
    with pytest.raises(ValueError, match="not a completed recording of an_01"):
        plots.build(LocationPlotView(), ["recording_002"])


def test_a_run_with_nothing_complete_refuses(proj):
    location_run(proj, {"recording_001": {"status": "failed"}})
    with pytest.raises(ValueError, match="no completed recording"):
        proj.location_plots("an_01")


def test_another_module_s_run_refuses(proj):
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {"id": "an_01", "module_id": "query_eval", "status": "complete"},
    )
    with pytest.raises(ValueError, match="not a location export"):
        proj.location_plots("an_01")


def test_both_animals_appear_and_labels_resolve_per_recording(proj):
    """Track numbers differ between recordings (the merge put the intruder first in one); a
    label finds the same animal in each."""
    location_run(
        proj,
        {
            "recording_001": {
                "rows": two_animals(ids=(0, 1)),
                "subject_labels": {0: "resident", 1: "intruder"},
            },
            "recording_002": {
                "rows": two_animals(ids=(4, 2)),
                "subject_labels": {4: "resident", 2: "intruder"},
            },
        },
    )
    plots = proj.location_plots("an_01")
    # Read in run order first, as the GUI does, so a map carried over from recording_001
    # would be the one recording_002 sees.
    assert plots.subject_choices() == ["intruder", "resident"]
    assert {s.label: s.object_id for s in plots.subjects("recording_002")} == {
        "resident": 4,
        "intruder": 2,
    }
    spec = plots.build(LocationPlotView(kind="trace"))
    assert [[ln.label for ln in p.lines] for p in spec.panels] == [
        ["intruder", "resident"],
        ["intruder", "resident"],
    ]
    spec = plots.build(LocationPlotView(kind="heatmap", subjects=["intruder"]))
    assert [p.title for p in spec.panels] == [
        "recording_001 · intruder",
        "recording_002 · intruder",
    ]


def test_an_older_export_takes_its_labels_from_the_tracking_run(proj):
    location_run(
        proj,
        {
            "recording_001": {
                "rows": two_animals(),
                "tracking_labels": {0: "white mouse", 1: "black mouse"},
            }
        },
    )
    plots = proj.location_plots("an_01")
    assert plots.subject_choices() == ["black mouse", "white mouse"]


def test_a_run_without_labels_names_animals_by_track(proj):
    location_run(proj, {"recording_001": {"rows": two_animals(ids=(3, 7))}})
    plots = proj.location_plots("an_01")
    assert plots.subject_choices() == ["subject 3", "subject 7"]
    spec = plots.build(LocationPlotView(kind="speed", subjects=["subject 7"]))
    assert [p.title for p in spec.panels] == ["recording_001 · subject 7"]


def test_a_group_counts_animals_and_crosses_with_the_animal_label(proj):
    labels = {0: "resident", 1: "intruder"}
    location_run(
        proj,
        {
            f"recording_00{i}": {"rows": two_animals(), "subject_labels": labels}
            for i in (1, 2, 3)
        },
    )
    for sid, genotype in (("recording_001", "wt"), ("recording_002", "wt")):
        proj.set_recording_metadata(sid, {"genotype": genotype})
    plots = proj.location_plots("an_01")
    spec = plots.build(LocationPlotView(kind="heatmap", group_by="genotype"))
    assert [p.title for p in spec.panels] == ["wt (n=4)", "(no value) (n=2)"]
    spec = plots.build(
        LocationPlotView(kind="heatmap", group_by="genotype", split_by_subject=True)
    )
    assert [p.title for p in spec.panels] == [
        "wt · intruder (n=2)",
        "wt · resident (n=2)",
        "(no value) · intruder (n=1)",
        "(no value) · resident (n=1)",
    ]


def test_an_animal_with_no_rows_is_named_under_the_title(proj):
    location_run(
        proj,
        {
            "recording_001": {
                "rows": tracks_rows(*circle(50), object_id=0),
                "subject_labels": {0: "resident", 1: "intruder"},
            }
        },
    )
    spec = proj.location_plots("an_01").build(LocationPlotView(kind="heatmap"))
    assert [p.title for p in spec.panels] == ["recording_001 · resident"]
    assert spec.note == "Not plotted: recording_001 · intruder: no tracked rows"


def test_a_recording_with_no_tracked_rows_says_so(proj):
    location_run(proj, {"recording_001": {"rows": []}})
    with pytest.raises(ValueError, match="no tracked rows"):
        proj.location_plots("an_01").build(LocationPlotView())


def test_columns_are_read_by_name_not_position(proj):
    run_dir = location_run(proj, {"recording_001": {}})
    path = run_dir / "recording_001" / "tracks.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([list(reversed(r)) for r in rows])
    spec = proj.location_plots("an_01").build(LocationPlotView())
    assert np.nanmax(spec.panels[0].lines[0].x) == pytest.approx(0.8 * 640, rel=0.01)


def test_auto_units_follow_calibration_and_mm_refuses_an_uncalibrated_one(proj):
    location_run(
        proj,
        {
            "recording_001": {"pixels_per_mm": 4.0},
            "recording_002": {"pixels_per_mm": 4.0},
        },
    )
    plots = proj.location_plots("an_01")
    assert plots.build(LocationPlotView()).x.label == "x (mm)"
    proj2 = Experiment.create("q", proj.dir.parent)
    location_run(proj2, {"recording_001": {"pixels_per_mm": 4.0}, "recording_002": {}})
    plots2 = proj2.location_plots("an_01")
    assert plots2.build(LocationPlotView()).x.label == "x (px)"
    with pytest.raises(ValueError, match="not calibrated: recording_002"):
        plots2.build(LocationPlotView(units="mm"))


def test_group_by_reads_recording_information_and_keeps_blank_values(proj):
    location_run(proj, {f"recording_00{i}": {} for i in range(1, 5)})
    proj.set_recording_metadata("recording_001", {"genotype": "wt"})
    proj.set_recording_metadata("recording_002", {"genotype": "wt"})
    proj.set_recording_metadata("recording_003", {"genotype": "mut"})
    plots = proj.location_plots("an_01")
    assert "genotype" in plots.group_columns
    spec = plots.build(LocationPlotView(kind="heatmap", group_by="genotype"))
    assert [p.title for p in spec.panels] == [
        "mut (n=1)",
        "wt (n=2)",
        "(no value) (n=1)",
    ]


def test_an_unknown_group_column_names_the_columns_there_are(proj):
    location_run(proj, {"recording_001": {}})
    proj.set_recording_metadata("recording_001", {"genotype": "wt"})
    with pytest.raises(
        ValueError, match="no Recording Information column 'sex'.*genotype"
    ):
        proj.location_plots("an_01").build(LocationPlotView(group_by="sex"))


def test_a_deleted_recording_still_plots_under_its_id(proj):
    location_run(proj, {"recording_001": {}})
    proj.delete_recording("recording_001")
    spec = proj.location_plots("an_01").build(LocationPlotView())
    assert spec.panels[0].title == "recording_001"


def test_a_second_plot_rereads_nothing(proj, monkeypatch):
    location_run(proj, {"recording_001": {}, "recording_002": {}})
    plots = proj.location_plots("an_01")
    plots.build(LocationPlotView())
    from detail.analysis import location_plots as lp

    def boom(*_a, **_k):
        raise AssertionError("re-read a tracks.csv")

    monkeypatch.setattr(lp, "read_tracks", boom)
    plots.build(LocationPlotView(kind="heatmap", source="bbox"))


def test_build_and_export_report_reading_then_drawing(proj, tmp_path):
    location_run(proj, {"recording_001": {}, "recording_002": {}})
    events = []
    paths = proj.location_plots("an_01").export(
        LocationPlotView(kind="trace3d"),
        tmp_path / "out" / "fig",
        formats=("png", "svg"),
        progress=lambda *e: events.append(e),
    )
    assert [p.name for p in paths] == ["fig.png", "fig.svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    stages = [stage for stage, _, _ in events]
    assert stages.index("reading") < stages.index("drawing") < stages.index("writing")
    reading = [fraction for stage, fraction, _ in events if stage == "reading"]
    assert reading[0] == 0.0 and reading[-1] == 1.0 and len(reading) == 3


def test_differing_resolutions_share_one_axis_range(proj):
    location_run(
        proj,
        {
            "recording_001": {"resolution": (640, 480)},
            "recording_002": {
                "resolution": (1280, 720),
                "rows": tracks_rows(*circle(50), resolution=(1280, 720)),
            },
        },
    )
    spec = proj.location_plots("an_01").build(LocationPlotView())
    assert spec.x.range == (0.0, 1280.0) and spec.y.range == (0.0, 720.0)


def test_a_manifest_without_a_resolution_refuses(proj):
    run_dir = location_run(proj, {"recording_001": {}})
    write_recording(run_dir, "recording_001", tracks_rows(*circle(10)), resolution=None)
    with pytest.raises(ValueError, match="no video_resolution"):
        proj.location_plots("an_01").build(LocationPlotView())


# --------------------------------------------------------------------------- #
# Keypoints and distances (§16 "Location plots", phase 2)
# --------------------------------------------------------------------------- #
def _kp_run(proj, *, hidden=frozenset()):
    from tests.location_runs import keypoint_rows

    x, y = circle(40)
    points = {"nose": (x, y), "tail_base": (x * 0.9, y)}
    location_run(
        proj,
        {
            "recording_001": {
                "rows": tracks_rows(x, y)[:40],
                "keypoints": keypoint_rows(0, points, hidden=hidden),
            }
        },
    )
    return proj.location_plots("an_01")


def test_a_keypoint_is_a_position_and_a_hidden_point_is_a_gap(proj):
    plots = _kp_run(proj, hidden=frozenset({(5, "nose")}))
    assert plots.keypoint_names == ["nose", "tail_base"]
    spec = plots.build(LocationPlotView(kind="trace", source="nose", color_by="time"))
    (line,) = spec.panels[0].lines
    assert np.isnan(line.x).sum() == 1  # frame 5, under the cut-off
    heat = plots.build(LocationPlotView(kind="heatmap", source="tail_base"))
    assert heat.panels[0].heat.grid.sum() == pytest.approx(1.0)


def test_several_points_draw_one_animal_per_panel_coloured_by_point(proj):
    plots = _kp_run(proj)
    spec = plots.build(LocationPlotView(kind="trace", points=["tail_base", "nose"]))
    (panel,) = spec.panels
    assert [(ln.label, ln.identity) for ln in panel.lines] == [
        ("tail_base", 0),
        ("nose", 1),
    ]
    assert spec.legend and spec.color is None


def test_an_unknown_keypoint_or_a_run_without_keypoints_says_so(proj):
    plots = _kp_run(proj)
    with pytest.raises(ValueError, match="no keypoint paw.*nose, tail_base"):
        plots.build(LocationPlotView(source="paw"))
    proj2 = Experiment.create("q", proj.dir.parent)
    location_run(proj2, {"recording_001": {}})
    with pytest.raises(ValueError, match="exported without a keypoint track run"):
        proj2.location_plots("an_01").build(LocationPlotView(source="nose"))


def _three_animal_run(proj, **extra):
    rows = (
        tracks_rows(*circle(30, radius=0.3), object_id=0)
        + tracks_rows(*circle(30, radius=0.1), object_id=1)
        + tracks_rows(
            *circle(20, radius=0.2), object_id=2
        )  # missing the last 10 frames
    )
    labels = {0: "a", 1: "b", 2: "c"}
    location_run(
        proj, {"recording_001": {"rows": rows, "subject_labels": labels, **extra}}
    )
    return proj.location_plots("an_01")


def test_three_animals_make_three_pairs_on_the_frames_both_are_on(proj):
    plots = _three_animal_run(proj)
    assert plots.pair_choices() == ["a – b", "a – c", "b – c"]
    pairs, notes = plots.pairs(LocationPlotView(kind="distance"))
    assert {p.label: len(p.t_s) for p in pairs} == {
        "a – b": 30,
        "a – c": 20,
        "b – c": 20,
    }
    spec = plots.build(LocationPlotView(kind="distance", pairs=["a – c"]))
    assert [ln.label for ln in spec.panels[0].lines] == ["a – c"]
    assert spec.legend and notes == []


def test_a_centroid_distance_matches_the_two_centroids(proj):
    plots = _three_animal_run(proj)
    pairs, _ = plots.pairs(LocationPlotView(kind="distance", units="px"))
    ab = next(p for p in pairs if p.label == "a – b")
    # circles of radius 0.3 and 0.1 around the centre, in phase: 0.2 of the frame apart
    assert ab.dist("norm")[0] == pytest.approx(0.2, abs=1e-5)
    assert ab.dist("px")[0] == pytest.approx(0.2 * 640, abs=0.01)


def test_mask_edge_reads_the_export_or_is_refused(proj):
    from tests.location_runs import pair_rows

    plots = _three_animal_run(proj)
    with pytest.raises(ValueError, match="no mask-edge distances"):
        plots.build(LocationPlotView(kind="distance", distance_by="mask_edge"))
    proj2 = Experiment.create("q", proj.dir.parent)
    plots2 = _three_animal_run(proj2, pairs=pair_rows({(0, 1): [3.0] * 30}))
    pairs, notes = plots2.pairs(
        LocationPlotView(kind="distance", distance_by="mask_edge", units="px")
    )
    assert [p.label for p in pairs] == ["a – b"]
    assert any("a – c: no mask-edge distance" in n for n in notes)
    with pytest.raises(ValueError, match="measured in pixels"):
        pairs[0].dist("norm")


def test_a_group_counts_pairs_and_a_histogram_sums_to_one_per_pair(proj):
    labels = {0: "Intruder", 1: "Resident"}
    location_run(
        proj,
        {
            f"recording_00{i}": {"rows": two_animals(), "subject_labels": labels}
            for i in (1, 2, 3)
        },
    )
    proj.set_recording_metadata("recording_001", {"genotype": "wt"})
    proj.set_recording_metadata("recording_002", {"genotype": "wt"})
    plots = proj.location_plots("an_01")
    spec = plots.build(LocationPlotView(kind="distance_hist", group_by="genotype"))
    labels_shown = [ln.label for ln in spec.panels[0].lines]
    assert labels_shown == ["wt (n=2)", "(no value) (n=1)"]
    per_recording = plots.build(LocationPlotView(kind="distance_hist"))
    for panel in per_recording.panels:
        (line,) = panel.lines
        assert line.step and line.y.sum() == pytest.approx(1.0)


def test_a_single_animal_recording_has_no_pair_and_says_so(proj):
    location_run(proj, {"recording_001": {}, "recording_002": {"rows": two_animals()}})
    spec = proj.location_plots("an_01").build(LocationPlotView(kind="distance"))
    assert [p.title for p in spec.panels] == ["recording_002"]
    assert "recording_001: fewer than two animals, no pair" in spec.note


def test_export_writes_an_interactive_page_beside_the_pictures(proj, tmp_path):
    location_run(proj, {"recording_001": {}})
    paths = proj.location_plots("an_01").export(
        LocationPlotView(kind="trace3d"), tmp_path / "fig", formats=("png", "html")
    )
    assert [p.name for p in paths] == ["fig.png", "fig.html"]
    assert "plotly" in paths[1].read_text(encoding="utf-8-sig")
