"""The clip popup's strips (``ClipQueryPreview.clip_signals``): speed, mask area, distance to the
nearest animal and the quality flags, over the popup's window of frames.

**Units a person can read.** An uncalibrated recording is measured in body sizes (the square root
of the subject's typical mask area); a calibrated one in mm. A strip in pixels would mean nothing
across two rigs, and the same clip would read differently on two videos of different resolution.

**The window is the frames shown, the gaps included.** A frame the subject is not tracked on is a
NaN in every strip and flagged on the quality strip — never interpolated, because a gap a
reviewer cannot see is a gap they read as behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from detail.api.recording import Recording
from detail.api.run_preview import ClipQueryPreview
from detail.core.skeleton import Skeleton
from detail.storage import keypoint_runs
from detail.storage.keypoint_zarr import KeypointTrackPartWriter
from tests.api.test_clip_views import _two_animals

# _two_animals: frame 0 holds track 0 (a 10x10 px mask at 10..20) and track 1 (a 10x10 mask at
# 40..50, 30..40); frame 2 holds track 0 alone, boxed at (0.5, 0.5, 0.6, 0.6) with no mask.
W, H = 64, 48


def _signals(recording, track_id=0, first=0, last=4):
    preview = ClipQueryPreview(recording.dir.parent.parent)
    return preview.clip_signals(recording.id, "emb_01", track_id, first, last)


def _by_name(signals):
    return {s.name: s for s in signals.signals}


def test_untracked_frames_are_gaps_and_flagged(tmp_path):
    signals = _signals(_two_animals(tmp_path))
    assert signals.frames.tolist() == [0, 1, 2, 3, 4]
    assert signals.unreliable.tolist() == [False, True, False, True, True]
    assert np.isnan(signals.centre[[1, 3, 4]]).all()
    for strip in signals.signals:
        assert np.isnan(strip.values[[3, 4]]).all(), strip.name


def test_an_uncalibrated_recording_is_measured_in_body_sizes(tmp_path):
    strips = _by_name(_signals(_two_animals(tmp_path)))
    # the subject's typical area is its one masked frame's 100 px, so a body size is 10 px
    assert strips["mask area"].values[0] == pytest.approx(1.0)
    assert strips["to nearest animal"].unit == "body sizes"
    apart = np.hypot(45 - 15, 35 - 15) / 10
    assert strips["to nearest animal"].values[0] == pytest.approx(apart, rel=1e-6)
    assert np.isnan(strips["to nearest animal"].values[2]), "alone on frame 2"


def test_speed_spans_the_frames_either_side(tmp_path):
    signals = _signals(_two_animals(tmp_path))
    speed = _by_name(signals)["speed"]
    assert speed.unit == "body sizes/s"
    moved = np.hypot((0.55 * W) - 15, (0.55 * H) - 15) / 10
    seconds = (signals.t_ns[2] - signals.t_ns[0]) / 1e9
    # frames 0..2 reach the tracked frames 0 and 2 on either side, across the gap at 1
    for i in (0, 1, 2):
        assert speed.values[i] == pytest.approx(moved / seconds, rel=1e-6), i
    assert np.isnan(speed.values[4]), "nothing tracked after frame 4"


def test_a_calibrated_recording_is_measured_in_mm(tmp_path):
    recording = _two_animals(tmp_path)
    Recording(recording.dir.parent.parent, recording.id).set_pixels_per_mm(2.0)
    strips = _by_name(_signals(recording))
    assert strips["to nearest animal"].unit == "mm"
    assert strips["speed"].unit == "mm/s"
    apart_px = np.hypot(45 - 15, 35 - 15)
    assert strips["to nearest animal"].values[0] == pytest.approx(apart_px / 2.0)


def test_the_subject_is_whichever_track_the_clip_names(tmp_path):
    signals = _signals(_two_animals(tmp_path), track_id=1)
    assert signals.unreliable.tolist() == [False, True, True, True, True]
    assert signals.centre[0] == pytest.approx([45 / W, 35 / H])


# --- the body's shape, from the keypoints the embedding read --------------------------- #
SPINE = Skeleton(
    names=("nose", "mid", "tail"), edges=(("nose", "mid"), ("mid", "tail"))
)


def _with_spine(recording, bodies, *, definition=True):
    """A keypoint track run ``kt_01`` with track 0's nose, mid and tail on frames 0.. (pixel
    triples), read by the embedding run — which names nose/tail as its heading definition
    unless ``definition`` is False."""
    run_dir = keypoint_runs.track_run_dir(recording.dir, "kt_01")
    keypoint_runs.write_track_yaml(
        run_dir,
        {
            "schema_version": 1,
            "id": "kt_01",
            "created_at": "2026-10-21T00:00:00Z",
            "detect_run_id": "kd_01",
            "tracking_run_id": "track_01",
            "method": "despike_smooth",
            "config": {},
            "skeleton": SPINE.to_dict(),
            "skeleton_digest": SPINE.digest(),
        },
    )
    n = len(bodies)
    path = keypoint_runs.track_part_path(run_dir, 0, "part_001")
    path.parent.mkdir(parents=True, exist_ok=True)
    with KeypointTrackPartWriter(path, SPINE.names) as writer:
        writer.add_rows(
            np.arange(n),
            (np.arange(n) * 1e9 / 30).astype(np.int64),
            np.arange(n, dtype=np.int32),
            (np.asarray(bodies, dtype=np.float32) / np.array([W, H], np.float32)),
            np.full((n, 3), 0.9, np.float32),
            np.zeros((n, 3), np.int8),
        )
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["keypoint_run_id"] = "kt_01"
    if definition:
        meta["align"] = "rotate"
        meta["heading_definition"] = {"front": ["nose"], "back": ["tail"]}
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    return recording


def _body(angle_deg, length=20.0, bend_px=0.0, centre=(32.0, 24.0)):
    """Nose, mid, tail in pixels: a body ``length`` long facing ``angle_deg`` (image degrees),
    its mid pushed ``bend_px`` off the axis."""
    a = np.radians(angle_deg)
    along = np.array([np.cos(a), np.sin(a)])
    across = np.array([-np.sin(a), np.cos(a)])
    c = np.asarray(centre)
    return [c + along * length / 2, c + across * bend_px, c - along * length / 2]


def test_the_keypoints_add_body_length_bend_and_turning(tmp_path):
    bodies = [_body(10.0 * i, length=20.0 if i != 3 else 30.0) for i in range(5)]
    bodies[4] = _body(40.0, bend_px=10.0)
    signals = _signals(_with_spine(_two_animals(tmp_path), bodies))
    strips = _by_name(signals)
    assert [s.name for s in signals.signals][-3:] == ["body length", "bend", "turning"]
    length = strips["body length"].values
    # typical is the median over the whole run (20 px): a stretch to 30 px reads 1.5
    assert length[[0, 1, 2]] == pytest.approx([1.0, 1.0, 1.0], rel=1e-5)
    assert length[3] == pytest.approx(1.5, rel=1e-5)
    assert strips["bend"].values[0] == pytest.approx(0.0, abs=1e-3)
    assert abs(strips["bend"].values[4]) > 40, "the mid pushed 10 px off a 20 px body"
    # turning 10° a frame at 30 fps is 300°/s, clockwise on screen (positive)
    assert strips["turning"].unit == "°/s"
    assert strips["turning"].values[1:4] == pytest.approx([300.0] * 3, rel=1e-3)


def test_without_a_heading_definition_there_are_no_pose_strips(tmp_path):
    recording = _with_spine(_two_animals(tmp_path), [_body(0.0)] * 5, definition=False)
    names = [s.name for s in _signals(recording).signals]
    assert names == ["speed", "mask area", "to nearest animal"]


def test_the_one_saved_definition_that_fits_is_used_and_two_are_not_chosen_between(
    tmp_path,
):
    from detail.storage import heading_definitions

    recording = _with_spine(_two_animals(tmp_path), [_body(0.0)] * 5, definition=False)
    experiment_dir = recording.dir.parent.parent
    heading_definitions.save_definition(experiment_dir, "spine", ["nose"], ["tail"])
    heading_definitions.save_definition(  # names points this skeleton lacks: not offered
        experiment_dir, "ears", ["left_ear"], ["tail_base"]
    )
    assert "bend" in [s.name for s in _signals(recording).signals]

    heading_definitions.save_definition(experiment_dir, "mid_out", ["mid"], ["tail"])
    preview = ClipQueryPreview(experiment_dir)
    views = preview.clip_views(
        recording.id, "emb_01", 0, 0, masks_on_frame=False, points_on_frame=False
    )
    assert "Several saved heading definitions" in views.align_note
    assert "mid_out" in views.align_note and "spine" in views.align_note
    assert "bend" not in [
        s.name for s in preview.clip_signals(recording.id, "emb_01", 0, 0, 4).signals
    ]


def test_a_skeleton_with_no_midline_point_gets_no_bend_strip(tmp_path):
    from tests.api.test_clip_views import _with_heading, _with_keypoints

    recording = _with_heading(
        _with_keypoints(_two_animals(tmp_path))
    )  # nose and tail only
    names = [s.name for s in _signals(recording).signals]
    assert names[-2:] == ["body length", "turning"] and "bend" not in names
