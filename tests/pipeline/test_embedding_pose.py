"""Pose in the embedding (§14, #659 ported onto the §26 keypoint track runs): ``draw`` paints the
points onto the image the model sees, ``append`` writes the body-relative pose block after the
mask's vector. Built on the aligned-embedding fixture, which is the same keypoint track run a
pose reads.
"""

import shutil

import numpy as np
import pytest
import yaml

from detail.pipeline import embedding
from detail.storage.zarr_io import PartZarrReader
from tests.pipeline.test_embedding import (  # noqa: F401  (aligned_recording is a fixture)
    D,
    FACING_RIGHT,
    _aligned_config,
    _keypoint_runs,
    _registered_embedder,
    _setup_multi_part,
    _validate,
    aligned_recording,
)


def _run(recording_dir, module, config, run_id):
    job = module.build_job("recording_001", run_id, config)
    run_dir = recording_dir / "embeddings" / run_id
    run_dir.mkdir(parents=True)
    embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    part = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    meta = yaml.safe_load((run_dir / "embedding.yaml").read_text(encoding="utf-8-sig"))
    return part, meta


def test_draw_paints_the_points_before_placement_and_counts_them(aligned_recording):
    recording_dir = aligned_recording
    with _registered_embedder():
        module, config = _validate(
            recording_dir,
            _aligned_config(align="none", heading=None, pose="draw"),
        )
        types = [s["type"] for s in config["renderer_spec"]]
        assert types == [
            "crop_to_bbox",
            "apply_mask",
            "draw_pose",
            "center_in_square",
            "resize",
        ]
        step = next(s for s in config["renderer_spec"] if s["type"] == "draw_pose")
        assert step["skeleton"]["names"] == ["nose", "tail"]
        assert (
            step["min_score"] == 0.3 == config["pose_min_score"]
        )  # the decode cut-off
        assert config["pose_weight"] is None and config["heading_definition"] is None
        part, meta = _run(recording_dir, module, config, "emb_p1")
    # track 0 is on frames 0, 1 and 3; the keypoint run placed points on 0 and 1 only
    assert part["frame_idx"].tolist() == [0, 1, 3]
    assert part["pose_points"].tolist() == [2, 2, 0]
    assert part["embedding"].shape == (3, D)  # drawing keeps the vector's size
    assert meta["pose"] == "draw" and meta["pose_dim"] is None


def test_append_writes_the_mask_then_the_weighted_pose_block(aligned_recording):
    recording_dir = aligned_recording
    with _registered_embedder():
        module, config = _validate(
            recording_dir,
            _aligned_config(align="none", pose="append", pose_weight=0.5),
        )
        assert "draw_pose" not in [s["type"] for s in config["renderer_spec"]]
        assert config["heading_definition"] == {"front": ["nose"], "back": ["tail"]}
        part, meta = _run(recording_dir, module, config, "emb_p2")
    vectors = part["embedding"]
    assert vectors.shape == (3, D + 4) and meta["output_dim"] == D + 4
    assert (meta["mask_dim"], meta["pose_dim"]) == (D, 4)
    mask, pose = vectors[:, :D], vectors[:, D:]
    # unit length; the fake embeds the image's mean, and frame 0 of the video is black, so
    # its vector is zero and stays zero rather than being divided by nothing
    norms = np.linalg.norm(mask, axis=1)
    assert np.all(np.isclose(norms, 1.0, atol=1e-5) | (norms == 0)) and norms.max() > 0
    # facing right (heading 0): the nose ahead of the centre, the tail behind, on the axis
    root_half = np.sqrt(0.5)
    np.testing.assert_allclose(
        pose[:2], 0.5 * np.array([[root_half, 0, -root_half, 0]] * 2), atol=1e-5
    )
    assert not pose[2].any() and part["pose_points"].tolist() == [2, 2, 0]


def test_pose_is_part_of_the_recipe(aligned_recording):
    recording_dir = aligned_recording
    with _registered_embedder():
        module, drawn = _validate(
            recording_dir, _aligned_config(align="none", heading=None, pose="draw")
        )
        _module, plain = _validate(
            recording_dir,
            _aligned_config(align="none", heading=None, keypoint_run_id=None),
        )
    assert {"pose", "pose_weight", "pose_min_score"} <= set(module.identity_fields)
    record = module.extra_record_fields(drawn)
    assert record["pose"] == "draw" and record["pose_min_score"] == 0.3
    assert module.extra_record_fields(plain)["pose"] == "none"
    assert plain["pose_min_score"] is None  # a setting that did nothing is not recorded


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        (
            {"align": "none", "heading": None, "pose": "draw", "keypoint_run_id": None},
            "reads a keypoint track run",
        ),
        ({"align": "none", "pose": "append", "heading": None}, "needs a heading"),
        ({"align": "none", "pose": "draw"}, "nothing reads it"),
        (
            {"align": "none", "heading": None, "pose": "draw", "every_n_frames": None},
            None,
        ),
    ],
)
def test_pose_validation_names_what_to_fix(aligned_recording, overrides, fragment):
    with _registered_embedder():
        if fragment is None:
            _validate(aligned_recording, _aligned_config(**overrides))
            return
        with pytest.raises(ValueError, match=fragment):
            _validate(aligned_recording, _aligned_config(**overrides))


def test_pose_needs_points_on_every_embedded_frame(aligned_recording):
    """A keypoint run at every 2nd frame cannot give a pose to a run embedding every frame."""
    recording_dir = aligned_recording
    data = yaml.safe_load(
        (recording_dir / "recording.yaml").read_text(encoding="utf-8-sig")
    )
    data["keypoint_detect_runs"][0]["every_n_frames"] = 2
    (recording_dir / "recording.yaml").write_text(
        yaml.dump(data, sort_keys=False), encoding="utf-8"
    )
    with _registered_embedder(), pytest.raises(ValueError, match="multiple of 2"):
        _validate(
            recording_dir, _aligned_config(align="none", heading=None, pose="draw")
        )


def test_a_frame_is_found_by_part_and_frame_together(tmp_path, monkeypatch):
    """Two parts whose frame numbers both start at zero: each part's frames take that part's
    points, never the other part's frame with the same number."""
    from detail.storage import heading_definitions, keypoint_runs
    from detail.storage.keypoint_zarr import KeypointTrackPartWriter

    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0]])
    _keypoint_runs(recording_dir, FACING_RIGHT)  # part_001: nose right of the tail
    heading_definitions.save_definition(tmp_path, "nose_to_tail", ["nose"], ["tail"])
    monkeypatch.setattr(
        "detail.storage.video_integrity.validate_source_videos", lambda *a, **k: []
    )
    second = keypoint_runs.track_part_path(
        keypoint_runs.track_run_dir(recording_dir, "kt_01"), 0, "part_002"
    )
    if second.exists():
        shutil.rmtree(second)
    with KeypointTrackPartWriter(second, ("nose", "tail")) as writer:
        points = np.array([[(0.6, 0.7), (0.8, 0.7)]] * 2, np.float32)  # facing left
        writer.add_rows(
            np.array([0, 1]),
            np.array([0, 1000]),
            np.array([0, 1]),
            points,
            np.full((2, 2), 0.9, np.float32),
            np.zeros((2, 2), np.int8),
        )
    with _registered_embedder():
        _module, config = _validate(
            recording_dir, _aligned_config(align="none", heading=None, pose="draw")
        )
    poses = embedding.poses_for_run(
        recording_dir,
        "track_01",
        [0],
        1,
        embedding._alignment_settings(config),
        (16, 20),
        None,
    )
    np.testing.assert_allclose(
        poses[(0, "part_001")]["xy"][0], [(0.35, 0.3), (0.15, 0.3)], atol=1e-6
    )
    np.testing.assert_allclose(
        poses[(0, "part_002")]["xy"][0], [(0.6, 0.7), (0.8, 0.7)], atol=1e-6
    )


def test_a_drawn_runs_preview_shows_the_points_it_drew(aligned_recording):
    """The run viewer re-renders a run from its own record (``for_embedding_run``): a drawn
    run's frames carry the points, and the same frame without them is the plain mask."""
    from detail.api.run_preview import InputImagePreview
    from detail.storage import recording as recording_store

    recording_dir = aligned_recording
    with _registered_embedder():
        module, config = _validate(
            recording_dir, _aligned_config(align="none", heading=None, pose="draw")
        )
        _part, meta = _run(recording_dir, module, config, "emb_p3")
    # the fixture writes no timestamp map, which a preview seeks by: one per frame, as import
    rel = "video/timestamps/generated/part_001.csv"
    (recording_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    recording_store.write_timestamps_csv(
        recording_dir / rel, np.arange(4, dtype=np.int64) * 1000
    )
    data = recording_store.read_recording(recording_dir)
    data["video"]["files"][0]["timestamps"] = {
        "generated": rel,
        "user": None,
        "active": "generated",
    }
    recording_store.write_recording(recording_dir, data)
    drawn = InputImagePreview.for_embedding_run(recording_dir, meta)
    plain = InputImagePreview(
        recording_dir,
        meta["tracking_run_id"],
        [s for s in meta["renderer_spec"] if s["type"] != "draw_pose"],
    )
    _frame, _rgb, with_points = drawn.input_at(0, 0, 0)
    _frame, _rgb, without = plain.input_at(0, 0, 0)
    assert with_points.shape == without.shape
    assert (with_points != without).any()
    _frame, _rgb, no_points_frame = drawn.input_at(0, 3, 0)  # frame 3: no points placed
    _frame, _rgb, plain_3 = plain.input_at(0, 3, 0)
    assert (no_points_frame == plain_3).all()
