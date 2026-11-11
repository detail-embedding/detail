"""Tests for the render-preview pipeline (Section 14).

Uses a real (tiny) synthetic video so the decode path exercises cv2 properly,
and mock-free fake detection/tracking zarrs so sampling + per-track filtering
are tested against real reader I/O. No GPU / network involved.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.core.dataclasses import RenderPreviewJob
from detail.pipeline import render_preview
from detail.pipeline.render_preview import _sample, run_render_preview
from detail.storage import recording as recording_store
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartWriter,
)

H, W = 24, 32
N_FRAMES = 8

RENDERER = [
    {"type": "init_binary_mask"},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 16},
]


def _mask_counts(frame_idx: int) -> str:
    m = np.zeros((H, W), np.uint8)
    top = (frame_idx * 2) % (H - 4)
    m[top : top + 4, 4:12] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _write_video(path: Path, n_frames: int = N_FRAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()


def _seed_recording_with_run(recording_dir: Path, object_ids: list[int]) -> None:
    """Drop video + detection zarr + tracking assignments + track.yaml + job.json."""
    _write_video(recording_dir / "video" / "part_001.mp4", n_frames=len(object_ids))
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(len(object_ids), dtype=np.int64) * 1000,
    )
    data = recording_store.read_recording(recording_dir)
    data["video"] = {
        "storage": "default",
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    }
    recording_store.write_recording(recording_dir, data)

    det_dir = recording_dir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    with DetectionPartWriter(det_dir / "part_001.zarr") as w:
        for f in range(len(object_ids)):
            w.add_row(
                f, f * 1000, 0, "mouse", 0.9, (0.1, 0.2, 0.3, 0.4), _mask_counts(f)
            )
    (det_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "mask_resolution": [H, W],
                "job": {"type": "detection"},
            }
        ),
        encoding="utf-8",
    )

    track_dir = recording_dir / "tracking" / "track_01"
    (track_dir / "assignments").mkdir(parents=True)
    with AssignmentPartWriter(track_dir / "assignments" / "part_001.zarr") as w:
        for oid in object_ids:
            w.add_object_id(oid)
    (track_dir / "track.yaml").write_text(
        yaml.dump(
            {"schema_version": 1, "id": "track_01", "detection_run_id": "det_01"}
        ),
        encoding="utf-8",
    )


def _job(track_ids=(0,), n=3):
    return RenderPreviewJob(
        tracking_run_id="track_01",
        track_ids=list(track_ids),
        renderer_spec=RENDERER,
        n_sample_frames=n,
    )


# --- _sample --------------------------------------------------------------- #
def test_sample_zero_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        _sample([object(), object()], 0, seed=0)


def test_sample_n_at_or_above_total_returns_all():
    cands = [object(), object(), object()]
    assert _sample(cands, 3, seed=0) == cands
    assert _sample(cands, 99, seed=0) == cands


def test_sample_deterministic_with_seed():
    cands = list(range(20))
    a = _sample(cands, 5, seed=42)
    b = _sample(cands, 5, seed=42)
    assert a == b
    assert len(a) == 5 and len(set(a)) == 5


# --- end-to-end (real video, real zarrs, fake-free) ----------------------- #
def test_returns_one_result_per_sample(recording_dir):
    _seed_recording_with_run(recording_dir, object_ids=[0] * N_FRAMES)
    results = run_render_preview(recording_dir, _job(n=3), seed=0)
    assert len(results) == 3
    for r in results:
        assert r.track_id == 0
        assert r.source_frame.shape == (H, W, 3)
        assert r.source_frame.dtype == np.uint8
        # rendered_image is shaped per the renderer spec (resize to 16x16x3)
        assert r.rendered_image.shape == (16, 16, 3)
        # det_row is the back-reference to the source detection zarr row
        assert isinstance(r.det_row, int) and 0 <= r.det_row < N_FRAMES


def test_filters_to_requested_track_ids(recording_dir):
    # half the rows are track 0, half are track 1; ask only for track 0
    object_ids = [0, 1, 0, 1, 0, 1, 0, 1]
    _seed_recording_with_run(recording_dir, object_ids=object_ids)
    results = run_render_preview(recording_dir, _job(track_ids=(0,), n=4), seed=0)
    assert {r.track_id for r in results} == {0}


def test_n_above_total_returns_all_candidates(recording_dir):
    _seed_recording_with_run(recording_dir, object_ids=[0] * 3)
    results = run_render_preview(recording_dir, _job(n=99), seed=0)
    assert len(results) == 3  # all candidates returned, not 99


def test_no_candidates_for_requested_track_raises(recording_dir):
    _seed_recording_with_run(recording_dir, object_ids=[0] * N_FRAMES)
    with pytest.raises(ValueError, match="no detections found"):
        run_render_preview(recording_dir, _job(track_ids=(99,), n=2), seed=0)


def test_unknown_transform_raises(recording_dir):
    _seed_recording_with_run(recording_dir, object_ids=[0] * N_FRAMES)
    job = _job(n=2)
    job.renderer_spec = [{"type": "bogus_transform"}]
    with pytest.raises(ValueError, match="unknown transform"):
        run_render_preview(recording_dir, job, seed=0)


def test_source_frame_has_mask_overlay(recording_dir):
    # The synthetic video frames are solid uniform color; if draw_mask ran, the
    # source_frame must differ from the raw frame in the masked pixels.
    _seed_recording_with_run(recording_dir, object_ids=[0] * 3)
    results = run_render_preview(recording_dir, _job(n=1), seed=0)
    r = results[0]
    raw_frame_color = (r.frame_idx * 17) % 256
    raw = np.full((H, W, 3), raw_frame_color, dtype=np.uint8)
    # At least some pixel in source_frame should differ from the raw frame
    # (the mask region is non-empty for every frame in this fixture).
    assert not np.array_equal(r.source_frame, raw)


# --- save: preview.json + 2 JPGs per sample ------------------------------- #
def test_save_writes_preview_json_and_two_jpgs_per_sample(recording_dir):
    _seed_recording_with_run(recording_dir, object_ids=[0] * N_FRAMES)
    results = run_render_preview(recording_dir, _job(n=3), save_as="preview_a", seed=0)
    out = recording_dir / "embeddings" / "render_previews" / "preview_a"
    assert (out / "preview.json").is_file()
    meta = json.loads((out / "preview.json").read_text(encoding="utf-8-sig"))
    assert meta["tracking_run_id"] == "track_01"
    assert meta["n_sample_frames"] == 3
    assert meta["seed"] == 0
    assert "0" in meta["track_colors"]
    assert len(meta["sampled_detections"]) == 3

    # two JPGs per sample: *_src.jpg and *_rendered.jpg
    frames_dir = out / "frames"
    jpgs = sorted(frames_dir.glob("*.jpg"))
    assert len(jpgs) == len(results) * 2
    assert sum(1 for p in jpgs if p.name.endswith("_src.jpg")) == len(results)
    assert sum(1 for p in jpgs if p.name.endswith("_rendered.jpg")) == len(results)
    img = cv2.imread(str(jpgs[0]))
    assert img is not None


def test_save_omitted_by_default_no_disk_artifacts(recording_dir):
    _seed_recording_with_run(recording_dir, object_ids=[0] * 4)
    run_render_preview(recording_dir, _job(n=2), seed=0)
    assert not (recording_dir / "embeddings" / "render_previews").exists()


def test_does_not_register_in_recording_yaml(recording_dir):
    _seed_recording_with_run(recording_dir, object_ids=[0] * 4)
    run_render_preview(recording_dir, _job(n=2), save_as="preview_a", seed=0)
    data = yaml.safe_load(
        (recording_dir / "recording.yaml").read_text(encoding="utf-8-sig")
    )
    assert "render_preview_runs" not in data
    assert "embedding_runs" not in data
