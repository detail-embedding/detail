"""A location export's preview and montage (§16 "Preview and montage"): the cluster montage's
engine over clips found through a ``ClipSource`` (tracking run + keypoint track run), one tile
per recording with every chosen animal in its colour.

Built on ``_walking``: two animals walking right in a real five-frame video, tracked, with a
keypoint track run on track 0 — and a location export the real compute wrote from it.
"""

import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.api.run_preview import ClipSource
from detail.configs.location_plots import LocationMontageView
from detail.storage import analysis_runs as ar
from tests.api.test_clip_views import H, W, _walking
from tests.location_runs import tracks_rows, write_recording


@pytest.fixture
def walked(tmp_path):
    """``(experiment, LocationPlots)`` over a location export of the walking recording."""
    recording = _walking(tmp_path)
    experiment = Experiment(recording.dir.parent.parent)
    run_dir = ar.analysis_dir(experiment.dir, "an_01")
    # The walkers' own centroids (10x10 masks at x0 = 4 + 6i, rows 4..14 and 30..40), through
    # the export's writer: the montage reads the run's timing and the recording's own runs.
    rows = []
    for oid, y0 in ((0, 4), (1, 30)):
        xs = np.array([(4 + 6 * i + 5) / W for i in range(5)])
        ys = np.full(5, (y0 + 5) / H)
        rows += tracks_rows(xs, ys, resolution=(W, H), object_id=oid)
    write_recording(run_dir, recording.id, rows, resolution=(W, H))
    ar.write_analysis_yaml(
        run_dir,
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": [recording.id],
            "status": "complete",
            "per_recording": {recording.id: {"status": "complete"}},
            "tracking_run_ids": {recording.id: "track_01"},
            "params": {"keypoint_run_ids": {recording.id: "kt_01"}},
        },
    )
    return experiment, experiment.location_plots("an_01")


def test_a_tile_is_a_recording_with_every_animal_in_its_colour(walked):
    _experiment, plots = walked
    built = plots.montage(LocationMontageView(full_length=True))
    (clip,) = built.montage.clips
    assert clip.subject == 0 and clip.embed_run_id == ""
    source = clip.source
    assert isinstance(source, ClipSource)
    assert (
        source.tracking_run_id == "track_01" and source.keypoint_track_run_id == "kt_01"
    )
    assert source.subject_ids == (0, 1)
    colours = dict(source.subject_colors)
    assert colours[0] != colours[1]
    built.montage.close()


def test_animals_are_chosen_by_label(walked):
    _experiment, plots = walked
    built = plots.montage(LocationMontageView(full_length=True), subjects=["subject 1"])
    assert built.montage.clips[0].source.subject_ids == (1,)
    built.montage.close()
    with pytest.raises(ValueError, match="none of the chosen animals"):
        plots.montage(LocationMontageView(full_length=True), subjects=["subject 9"])


def test_the_window_clamps_to_what_was_tracked_and_a_late_one_is_named(walked):
    _experiment, plots = walked
    built = plots.montage(LocationMontageView(start_s=0.0, length_s=60.0))
    (clip,) = built.montage.clips
    assert clip.start_t_ns == 0 and clip.end_t_ns == int(4 * 1e9 / 30)
    built.montage.close()
    with pytest.raises(ValueError, match="nothing tracked in the window"):
        plots.montage(LocationMontageView(start_s=10.0, length_s=5.0))


def test_each_animal_is_drawn_in_its_own_colour(walked):
    _experiment, plots = walked
    built = plots.montage(LocationMontageView(full_length=True))
    montage = built.montage
    montage.prepare()
    clip = montage.clips[0]
    views = montage._preview(0).clip_views(
        clip.recording_id, clip.key, clip.subject, 2, masks_on_frame=False,
        points_on_frame=False,
    )  # fmt: skip
    colours = [np.asarray(rgb) for _oid, rgb in clip.source.subject_colors]
    for colour in colours:
        assert (views.masks.reshape(-1, 3) == colour).all(axis=1).any()
    montage.close()


@pytest.mark.parametrize("trail", [False, True])
def test_each_animal_is_outlined_in_its_own_colour_on_the_footage(walked, trail):
    """With **Raw** the masks are outlines, drawn at the tile's scale; they were one tint for
    every animal, so a two-animal tile could not say which was which."""
    _experiment, plots = walked
    view = LocationMontageView(full_length=True, full_frame=True, raw=True, mask=True)
    view = view.model_copy(update={"mask_trail": trail})
    built = plots.montage(view)
    montage = built.montage
    montage.prepare()
    clip = montage.clips[0]
    tile = montage.tile(0, montage.frame_at(0, 2 * 10**8), view.layers(), 128)
    for _oid, rgb in clip.source.subject_colors:
        assert (tile.reshape(-1, 3) == np.asarray(rgb)).all(axis=1).any(), rgb
    montage.close()


def test_full_frame_shows_the_whole_frame(walked):
    _experiment, plots = walked
    built = plots.montage(LocationMontageView(full_length=True, full_frame=True))
    montage = built.montage
    montage.prepare()
    montage.tile(0, montage.frame_at(0, 0), LocationMontageView().layers(), 64)
    assert montage._prep(0).crop == (0, 0, 64, 48)
    montage.close()


def test_speed_writes_fewer_frames_of_the_same_rate(walked, tmp_path):
    from PIL import Image

    _experiment, plots = walked
    slow = plots.export_montage(
        LocationMontageView(full_length=True, speed=0.5), tmp_path / "slow.gif"
    )
    fast = plots.export_montage(
        LocationMontageView(full_length=True, speed=4.0), tmp_path / "fast.gif"
    )
    with Image.open(slow) as a, Image.open(fast) as b:
        assert a.n_frames > b.n_frames


def test_the_last_tracing_frame_trails_over_the_whole_clip(
    walked, tmp_path, monkeypatch
):
    from detail.api.run_preview import ClipQueryPreview

    _experiment, plots = walked
    seen = []
    original = ClipQueryPreview.clip_views

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("trail_s"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ClipQueryPreview, "clip_views", spy)
    view = LocationMontageView(full_length=True, keypoints=True, keypoint_trail=True)
    path = plots.export_montage(view, tmp_path / "last.png", last_frame=True)
    assert path.is_file() and path.stat().st_size > 0
    assert seen and seen[-1] == pytest.approx(
        4 / 30
    )  # the clip's own length, not 0.5 s


def test_a_recording_gone_from_the_experiment_is_named(walked):
    experiment, plots = walked
    sid = plots.recording_ids[0]
    experiment.delete_recording(sid)
    with pytest.raises(ValueError, match="no longer in the experiment"):
        experiment.location_plots("an_01").montage(
            LocationMontageView(full_length=True)
        )


def test_no_layer_ticked_is_refused(walked, tmp_path):
    _experiment, plots = walked
    view = LocationMontageView(full_length=True, mask=False, mask_trail=False)
    with pytest.raises(ValueError, match="turn on at least one"):
        plots.export_montage(view, tmp_path / "x.gif")


def test_a_crop_is_recut_when_the_picture_size_changes(walked):
    """A video can decode at another size than its record says (a 1920x1080 file recorded as
    1280x720). A Raw-off tile draws on a canvas of the recorded size, a Raw-on tile on the
    footage; a crop cut in one's pixels landed in a corner of the other until it was re-cut.
    """
    from detail.storage import recording as recording_store

    experiment, plots = walked
    recording = experiment.resolve_recording(plots.recording_ids[0])
    data = recording_store.read_recording(recording.dir)
    data["video"]["resolution"] = [W * 2, H * 2]  # the record says larger than the file
    recording_store.write_recording(recording.dir, data)
    built = plots.montage(LocationMontageView(full_length=True))
    montage = built.montage
    montage.prepare()
    frame = montage.frame_at(0, 0)
    montage.tile(0, frame, LocationMontageView(raw=False).layers(), 64)
    assert montage._prep(0).crop_size == (W * 2, H * 2)
    montage.tile(0, frame, LocationMontageView(raw=True).layers(), 64)
    assert montage._prep(0).crop_size == (W, H)
    montage.close()
