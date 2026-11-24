"""Keypoints in a location export, and in its plots (§16 *Keypoints in the export*).

**The lineage refusal.** A keypoint track run assigns its poses to the animals of one tracking
run, and ``object_id`` is a position within that run. An export that read a keypoint run of
another tracking run would put one animal's points on another — numbers that look right and
are not. So the run is refused at launch (``LocationExportModule.validate``), naming the
recording, and again at the compute for a direct caller (``keypoint_table``).
``test_a_keypoint_run_of_another_tracking_run_is_refused_at_launch`` is the guard.

Cases come from the edges of what is consumed: the real stages' ``kt_01`` (two animals, a
point on every frame), a sampled export (its keypoints on its tracks' rows), a point hidden
under the cut-off (a gap in a plot), a run exported without keypoints, and an unknown point.
"""

import csv

import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.configs.location_plots import LocationPlotView
from detail.storage import location_export
from tests.api.test_keypoint_export import keypoint_runs  # noqa: F401 — a fixture
from tests.location_fixtures import finished_run, track, write_keypoints


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def recorded(keypoint_runs):  # noqa: F811 — the fixture imported above
    """``keypoint_runs`` with ``kt_01`` recorded in ``recording.yaml`` as the keypoint track
    stage records it (the shared fixture writes the run, and records only the detect run).
    """
    from detail.storage import recording as recording_store

    it, recording = keypoint_runs
    data = recording_store.read_recording(recording.dir)
    data.setdefault("keypoint_track_runs", []).append(
        {"id": "kt_01", "status": "complete", "detect_run_id": "kd_01"}
    )
    recording_store.write_recording(recording.dir, data)
    return it, recording


# --------------------------------------------------------------------------- #
# The export
# --------------------------------------------------------------------------- #
def test_keypoints_are_written_on_the_rows_of_the_tracks(recorded, tmp_path):
    _it, recording = recorded
    out = recording.export_locations("track_01", tmp_path, keypoint_run_id="kt_01")
    tracks = _rows(out / location_export.TRACKS_FILE)
    points = _rows(out / location_export.KEYPOINTS_FILE)
    manifest = location_export.read_location_manifest(out)
    names = manifest["keypoint_names"]
    assert names and manifest["keypoint_run_id"] == "kt_01"
    assert manifest["files"]["keypoints"] == location_export.KEYPOINTS_FILE
    keys = {(r["frame_idx"], r["object_id"]) for r in tracks}
    # Every tracked frame of every animal carries every point, and nothing else.
    assert {(r["frame_idx"], r["object_id"]) for r in points} == keys
    assert len(points) == len(tracks) * len(names)
    assert list(points[0])[:4] == ["frame_idx", "t_ns", "object_id", "keypoint"]


def test_a_sampled_export_keeps_its_keypoints_on_its_tracks_rows(recorded, tmp_path):
    _it, recording = recorded
    out = recording.export_locations(
        "track_01", tmp_path, keypoint_run_id="kt_01", sample_fps=10.0
    )
    keys = {(r["frame_idx"], r["object_id"]) for r in _rows(out / "tracks.csv")}
    points = {(r["frame_idx"], r["object_id"]) for r in _rows(out / "keypoints.csv")}
    assert points == keys


def test_an_export_without_keypoints_writes_no_keypoint_file(keypoint_runs, tmp_path):
    _it, recording = keypoint_runs
    out = recording.export_locations("track_01", tmp_path)
    assert not (out / location_export.KEYPOINTS_FILE).exists()
    assert "keypoints" not in location_export.read_location_manifest(out)["files"]


def _point_at_another_tracking_run(recording, run_id="kt_01"):
    import yaml

    path = recording.dir / "keypoints" / "track" / run_id / "track.yaml"
    if not path.is_file():
        path = next(recording.dir.rglob(f"{run_id}/track.yaml"))
    meta = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    meta["tracking_run_id"] = "track_09"
    path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def test_a_keypoint_run_of_another_tracking_run_is_refused_at_launch(recorded):
    from detail.executor.runners import ANALYSIS_MODULES

    it, recording = recorded
    module = ANALYSIS_MODULES["location_export"]
    config = {
        "source_recording_ids": ["recording_002"],
        "tracking_run_id": "track_01",
        "keypoint_run_id": "kt_01",
    }
    resolved = module.validate(it.experiment.dir, dict(config))
    assert resolved["keypoint_run_ids"] == {"recording_002": "kt_01"}
    job = module.build_job("an_01", resolved)
    assert job.keypoint_run_ids == {"recording_002": "kt_01"}
    assert job.to_dict()["keypoint_run_ids"] == {"recording_002": "kt_01"}

    _point_at_another_tracking_run(recording)
    with pytest.raises(ValueError, match="recording_002.*'track_09'.*wrong animal"):
        module.validate(it.experiment.dir, dict(config))


def test_the_compute_refuses_it_too_for_a_direct_caller(recorded, tmp_path):
    _it, recording = recorded
    _point_at_another_tracking_run(recording)
    with pytest.raises(ValueError, match="wrong animal"):
        recording.export_locations("track_01", tmp_path, keypoint_run_id="kt_01")


def test_a_missing_keypoint_run_is_named(keypoint_runs, tmp_path):
    from detail.executor.runners import ANALYSIS_MODULES

    it, recording = keypoint_runs
    with pytest.raises(ValueError, match="'kt_07' is missing"):
        recording.export_locations("track_01", tmp_path, keypoint_run_id="kt_07")
    with pytest.raises(
        ValueError, match="recording_002.*no keypoint track run 'kt_07'"
    ):
        ANALYSIS_MODULES["location_export"].validate(
            it.experiment.dir,
            {
                "source_recording_ids": ["recording_002"],
                "tracking_run_id": "track_01",
                "keypoint_run_id": "kt_07",
            },
        )


def test_a_job_from_before_keypoints_reads_none():
    from detail.pipeline.location_analysis import LocationAnalysisJob

    job = LocationAnalysisJob.from_dict(
        {
            "job_id": "an_01",
            "source_recording_ids": ["s1"],
            "tracking_run_ids": {"s1": "track_01"},
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    assert job.keypoint_run_ids == {}


# --------------------------------------------------------------------------- #
# The plots
# --------------------------------------------------------------------------- #
def _walk(n=150):
    return track(10.0 + np.arange(n) * 2.0, 40.0 + 10 * np.sin(np.arange(n) / 10))


@pytest.fixture
def proj(tmp_path):
    return Experiment.create("p", tmp_path)


def _with_points(proj, hidden=None):
    walk = _walk()
    ids = finished_run(proj, {"a": {0: walk}, "b": {0: walk}})
    for sid in ids.values():
        write_keypoints(
            proj.analysis_run_dir("an_01") / sid,
            walk,
            {"nose": (8.0, 0.0, hidden), "tail_base": (-8.0, 0.0, None)},
        )
    return ids


def test_a_keypoint_is_a_position_and_a_hidden_point_is_a_gap(proj):
    hidden = np.zeros(150, dtype=bool)
    hidden[60:70] = True
    _with_points(proj, hidden)
    plots = proj.location_plots("an_01")
    assert plots.keypoint_names == ["nose", "tail_base"]
    spec = plots.build(LocationPlotView(source="nose", units="mm"))
    line = spec.panels[0].lines[0]
    assert np.nanmin(line.x) == pytest.approx(18.0, abs=0.01)  # 10 mm + the 8 mm offset
    assert np.isnan(line.x).sum() >= 10  # the hidden stretch draws as nothing
    # The nose rides the body at a fixed offset, so it moves at the body's speed; the hidden
    # stretch is a gap in its path, not a jump.
    nose = (
        plots.build(LocationPlotView(kind="speed", source="nose")).panels[0].lines[0].y
    )
    body = plots.build(LocationPlotView(kind="speed")).panels[0].lines[0].y
    both = np.isfinite(nose) & np.isfinite(body)
    np.testing.assert_allclose(nose[both], body[both], rtol=0.02)
    # Sample by sample: a hidden point has no speed, and every placed one on either side of
    # it has one. Smoothed through the hidden stretch instead, the NaN spread into its
    # neighbours and the steps on either side were lost with it.
    (trace, *_), _notes = plots.traces(
        LocationPlotView(source="nose"), [plots.recording_ids[0]]
    )
    per_sample = trace.speed(LocationPlotView(), "mm")
    assert np.isnan(per_sample[60:70]).all()
    assert np.isfinite(per_sample[:60]).all() and np.isfinite(per_sample[70:]).all()


def test_several_points_are_one_panel_per_recording_coloured_by_point(proj):
    _with_points(proj)
    spec = proj.location_plots("an_01").build(
        LocationPlotView(points=["tail_base", "nose"], posture_every_s=10.0)
    )
    assert [p.title for p in spec.panels] == ["a", "b"]
    lines = spec.panels[0].lines
    assert [(ln.label, ln.identity) for ln in lines[:2]] == [
        ("tail_base", 0),
        ("nose", 1),
    ]
    assert lines[2].label == "body outline" and lines[2].values is None  # once, in ink
    assert spec.legend and spec.color is None


@pytest.mark.parametrize(
    "view, drawn",
    [
        (
            LocationPlotView(points=["tail_base", "nose"], posture_every_s=10.0),
            ['"name":"tail_base"', '"name":"nose"', '"name":"body outline"'],
        ),
        (LocationPlotView(kind="trace3d", source="nose"), ['"type":"scatter3d"']),
    ],
    ids=["points", "nose-3d"],
)
def test_keypoint_plots_export_as_an_interactive_page_too(proj, tmp_path, view, drawn):
    """The interactive page (#99) and the keypoint plots (#100) were built on two branches,
    each against plots without the other: a keypoint plot's lines, legend and outline are
    drawn by the page as by the pictures."""
    _with_points(proj)
    plots = proj.location_plots("an_01")
    paths = plots.export(view, tmp_path / "kp", formats=("png", "html"))
    assert [p.name for p in paths] == ["kp.png", "kp.html"]
    # Spaces dropped from both: plotly's JSON spacing is not this test's business.
    page = (tmp_path / "kp.html").read_text(encoding="utf-8").replace(" ", "")
    missing = [mark for mark in drawn if mark.replace(" ", "") not in page]
    assert not missing


def test_an_unknown_keypoint_or_a_run_without_keypoints_says_so(proj, tmp_path):
    _with_points(proj)
    with pytest.raises(
        ValueError, match=r"no keypoint ear in this run \(it has nose, tail_base\)"
    ):
        proj.location_plots("an_01").build(LocationPlotView(source="ear"))
    bare = Experiment.create("b", tmp_path / "b")
    finished_run(bare, {"a": {0: _walk()}})
    with pytest.raises(ValueError, match="exported without a keypoint track run"):
        bare.location_plots("an_01").build(LocationPlotView(source="nose"))


def test_a_body_plot_reads_the_body_whatever_the_position(proj):
    _with_points(proj)
    spec = proj.location_plots("an_01").build(
        LocationPlotView(kind="body", source="nose", points=["nose"])
    )
    assert len(spec.panels) == 2
