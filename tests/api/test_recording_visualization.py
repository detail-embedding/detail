"""Recording visualization through the API (§22): reading a recipe's data, drawing a frame at
any tile size, re-colouring, saving recipes, and exporting one or a montage.

The edges come from what the stage consumes (``schema-edge-tests``): a method with no frame
sample (``dtw_finch``, the shared fit) takes the pose-space fallback; an uncalibrated recording
draws location in px; a range that lies outside the recording, a recording outside the run and a
recode that changes the range are refused.
"""

import av
import numpy as np
import pytest
import yaml

from detail.analysis import recording_visualization as rv
from detail.api import recording_visualization as rviz
from detail.configs.recording_visualization import (
    RecordingVisualizationConfig,
    RecordingVisualizationExport,
)

VIZ_W, VIZ_H = 64, 48  # the visualizable_run fixture's video


def _cut(run, k=3):
    tree = run.hierarchy()
    shown = np.sort(
        rv.display_heights(tree.children, tree.height, tree.n_leaves)[tree.n_leaves :]
    )[::-1]
    return float((shown[k - 2] + shown[k - 1]) / 2)


def _recipe(run, **kw):
    base = dict(recording_id="recording_001", start_s=0.5, end_s=3.0)
    base.update(kw)
    return RecordingVisualizationConfig(**base)


def test_load_reads_pose_space_clusters_and_location(visualizable_run):
    _experiment, run = visualizable_run
    cut = _cut(run)
    viz = run.visualization(
        _recipe(run, clusters="cut", cut_height=cut, cut_included=[1, 2])
    )
    d = viz.data
    assert d.size == (VIZ_W, VIZ_H)
    assert d.first > 0 and d.last > d.first and d.n_frames == d.last - d.first
    assert d.xy.shape == (len(d.t_ns), 2)
    inside = np.isfinite(d.xy[d.first : d.last, 0])
    assert inside.mean() > 0.5
    # dtw_finch keeps no frame sample: the documented fallback, and the window says so.
    assert "no frame sample of its own" in d.pose_note
    assert set(np.unique(d.codes[d.first : d.last])) <= {0, 1, rv.GREY, rv.NO_POSE}
    assert d.names == ["cluster 1", "cluster 2"] and d.grey_name == "other"
    # uncalibrated: px, over the video's own extent
    assert d.pos_units == "px" and d.floor == (float(VIZ_W), float(VIZ_H))
    track = d.positions[d.first : d.last]
    assert np.isfinite(track).all()
    assert track[-1, 0] > track[0, 0]  # the square moves left to right


def test_named_without_names_is_all_unnamed(visualizable_run):
    _experiment, run = visualizable_run
    d = run.visualization(_recipe(run, location=False)).data
    assert d.names == [] and d.grey_name == "unnamed" and d.positions is None
    assert set(np.unique(d.codes[d.first : d.last])) <= {rv.GREY, rv.NO_POSE}


@pytest.mark.parametrize("theme, corner", [("dark", 0), ("light", 255)])
@pytest.mark.parametrize("size", [(1920, 1080), (960, 1080), (640, 360)])
def test_a_frame_is_exactly_its_tile_on_its_background(
    visualizable_run, theme, corner, size
):
    _experiment, run = visualizable_run
    viz = run.visualization(_recipe(run))
    pic = viz.frame(viz.data.first + 10, theme, size)
    assert pic.shape == (size[1], size[0], 3) and pic.dtype == np.uint8
    assert tuple(pic[-2, 2]) == (corner,) * 3


def test_recode_recolours_without_reloading_and_keeps_the_range(visualizable_run):
    _experiment, run = visualizable_run
    viz = run.visualization(_recipe(run))
    cut = _cut(run)
    again = rviz.recode(
        viz.data, run, _recipe(run, clusters="cut", cut_height=cut, cut_included=[1])
    )
    assert again.names == ["cluster 1"]
    assert again.xy is viz.data.xy  # the pose space is not read again
    with pytest.raises(ValueError, match="recode keeps the recording and range"):
        rviz.recode(viz.data, run, _recipe(run, start_s=0.6))


def test_a_recipe_outside_the_run_or_the_recording_is_refused(visualizable_run):
    _experiment, run = visualizable_run
    with pytest.raises(ValueError, match="is not one of an_"):
        run.visualization(RecordingVisualizationConfig(recording_id="recording_999"))
    with pytest.raises(ValueError, match="holds no frames"):
        run.visualization(_recipe(run, start_s=900.0, end_s=901.0))


def test_recipes_save_list_and_delete_through_the_run(visualizable_run):
    experiment, run = visualizable_run
    viz_id = run.save_visualization(_recipe(run))
    assert viz_id == "viz_001"
    saved = run.visualizations()
    assert [r["id"] for r in saved] == ["viz_001"] and saved[0]["start_s"] == 0.5
    path = (
        experiment.dir
        / "analysis"
        / run.id
        / "visualizations"
        / "viz_001"
        / "visualization.yaml"
    )
    assert (
        next(iter(yaml.safe_load(path.read_text(encoding="utf-8-sig"))))
        == "schema_version"
    )
    assert run.visualization("viz_001").data.n_frames > 0
    run.delete_visualization("viz_001")
    assert run.visualizations() == []
    with pytest.raises(ValueError, match="is not one of"):
        run.save_visualization(
            RecordingVisualizationConfig(recording_id="recording_999")
        )


def test_data_survives_the_trip_to_a_worker(visualizable_run, tmp_path):
    _experiment, run = visualizable_run
    d = run.visualization(_recipe(run)).data
    d.save(tmp_path / "d.npz")
    back = rviz.VizData.read(tmp_path / "d.npz")
    assert back.size == d.size and back.first == d.first and back.recipe == d.recipe
    np.testing.assert_array_equal(back.codes, d.codes)
    np.testing.assert_allclose(back.positions, d.positions)


def _decoded(path):
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
        return frames, float(c.duration / 1e6)


def test_export_writes_one_exact_file_per_background(visualizable_run, tmp_path):
    _experiment, run = visualizable_run
    viz = run.visualization(_recipe(run, end_s=1.5))
    settings = RecordingVisualizationExport(
        themes=["dark", "light"], width=640, workers=2
    )
    seen = []
    written = viz.export(
        tmp_path / "out.mp4", settings, progress=lambda s, f, d: seen.append(f)
    )
    assert [p.name for p in written] == ["out_dark.mp4", "out_light.mp4"]
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    fps = min(50.0, viz.data.fps)
    expected = int(viz.data.n_frames / viz.data.fps * fps)
    for path, corner in zip(written, (0, 255)):
        frames, duration = _decoded(path)
        assert len(frames) == expected
        assert frames[0].shape == (360, 640, 3)
        assert tuple(frames[len(frames) // 2][-2, 2]) == (corner,) * 3


def test_a_montage_tiles_and_holds_the_shorter_one(visualizable_run, tmp_path):
    _experiment, run = visualizable_run
    long_ = run.visualization(_recipe(run, end_s=2.0))
    short = run.visualization(_recipe(run, end_s=1.0))
    settings = RecordingVisualizationExport(themes=["dark"], width=640, workers=2)
    (path,) = rviz.export_montage([long_, short], tmp_path / "m.mp4", settings)
    frames, _duration = _decoded(path)
    fps = min(50.0, long_.data.fps)
    assert len(frames) == int(long_.data.n_frames / long_.data.fps * fps)

    # The right tile has ended by the last frames and is held: still, where the left tile
    # still moves. Measured as mean change per pixel, since H.264 leaves a few pixels of a still
    # region a few levels off from frame to frame (measured: about 0.001 held, about 0.5 moving).
    def change(side):
        cols = slice(320, None) if side == "right" else slice(0, 320)
        late = [f[:, cols].astype(int) for f in frames[-6:]]
        return max(np.abs(b - a).mean() for a, b in zip(late, late[1:]))

    assert change("right") < 0.05
    assert change("left") > 10 * change("right")


def test_estimate_measures_time_and_size(visualizable_run):
    _experiment, run = visualizable_run
    viz = run.visualization(_recipe(run, end_s=2.0))
    est = viz.estimate(
        RecordingVisualizationExport(themes=["dark", "light"], width=640, workers=2)
    )
    assert est["files"] == 2 and est["frames"] > 0 and est["workers"] >= 1
    assert est["seconds"] > 0 and est["bytes_per_file"] > 0
