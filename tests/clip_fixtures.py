"""Recordings for the clip window's tests, shared by the API, GUI and CLI halves.

A helper module rather than fixtures in a conftest, the way `tests/experiments.py` is: these
build a recording with hand-written detection, tracking, embedding and keypoint stores (no
model), and the test that needs one calls the builder it wants. Six test modules used to
import them as private names from one test file, which the impact table could not see.

Every recording is `tests.api.test_run_preview._recording_with_detection`'s: a 64x48 video of
five frames and a detection run ``det_01``, plus a tracking run ``track_01`` whose rows these
builders write.
"""

from __future__ import annotations

import json
import shutil
import threading

import numpy as np
import yaml
from pycocotools import mask as mask_utils

from detail.core.skeleton import Skeleton
from detail.pipeline import kinematics_backfill
from detail.storage import keypoint_runs
from detail.storage import recording as recording_store
from detail.storage.keypoint_zarr import KeypointTrackPartWriter
from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter

from tests.api.test_run_preview import (
    _INPUT_SPEC,
    _add_embedding,
    _recording_with_detection,
)
from tests.experiments import write_embedding_run

H, W = 48, 64
SUBJECT = (slice(10, 20), slice(10, 20))  # track 0's mask on frame 0
OTHER = (slice(30, 40), slice(40, 50))  # track 1's
SKELETON = Skeleton(names=("nose", "tail"), edges=(("nose", "tail"),))
#: A walker for the trails: a midline nose and tail, and a left and right paw.
WALKER = Skeleton(
    names=("nose", "left_paw", "right_paw", "tail"),
    edges=(("nose", "tail"),),
    pairs=(("left_paw", "right_paw"),),
)


def write_track_yaml(tracking_dir, *, detection_run_id):
    """The ``track.yaml`` a complete tracking run writes, which names its detection run."""
    tracking_dir.mkdir(parents=True, exist_ok=True)
    (tracking_dir / "track.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": tracking_dir.name,
                "detection_run_id": detection_run_id,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def rle(region) -> str:
    """A rectangular region of the 48x64 frame as the ASCII COCO-RLE string the zarr stores."""
    mask = np.zeros((H, W), np.uint8)
    mask[region] = 1
    return mask_utils.encode(np.asfortranarray(mask))["counts"].decode("ascii")


def _tracking_run(recording):
    write_track_yaml(recording.dir / "tracking" / "track_01", detection_run_id="det_01")
    data = recording_store.read_recording(recording.dir)
    data.setdefault("tracking_runs", []).append(
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    )
    recording_store.write_recording(recording.dir, data)


def two_animals(tmp_path):
    """Frame 0 holds two tracked animals with masks (tracks 0 and 1); frame 2 holds track 0.
    The embedding ``emb_01`` covers frames 0 and 2."""
    recording = _recording_with_detection(tmp_path)
    ddir = recording.dir / "detections" / "det_01"
    shutil.rmtree(ddir / "part_001.zarr")
    with DetectionPartWriter(ddir / "part_001.zarr") as writer:
        writer.add_row(
            0, 0, 0, "mouse", 0.9, (10 / W, 10 / H, 20 / W, 20 / H), rle(SUBJECT)
        )
        writer.add_row(
            0, 0, 1, "mouse", 0.8, (40 / W, 30 / H, 50 / W, 40 / H), rle(OTHER)
        )
        writer.add_row(2, int(2e9 / 30), 0, "mouse", 0.7, (0.5, 0.5, 0.6, 0.6), "")
    adir = recording.dir / "tracking" / "track_01" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        for object_id in (0, 1, 0):
            writer.add_object_id(object_id)
    _tracking_run(recording)
    return _add_embedding(recording)


def _name_keypoint_run(recording, run_id="kt_01", *, heading=False):
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["keypoint_run_id"] = run_id
    if heading:
        meta["align"] = "rotate"
        meta["heading_definition"] = {"front": ["nose"], "back": ["tail"]}
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")


def with_keypoints(recording):
    """A keypoint track run ``kt_01`` with both animals' points on frame 0, named by the
    embedding run as the one it read."""
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
            "skeleton": SKELETON.to_dict(),
            "skeleton_digest": SKELETON.digest(),
        },
    )
    for track_id, (nose, tail) in {
        0: ((12 / W, 12 / H), (18 / W, 18 / H)),
        1: ((42 / W, 32 / H), (48 / W, 38 / H)),
    }.items():
        path = keypoint_runs.track_part_path(run_dir, track_id, "part_001")
        path.parent.mkdir(parents=True, exist_ok=True)
        with KeypointTrackPartWriter(path, SKELETON.names) as writer:
            writer.add_rows(
                np.array([0]),
                np.array([0], np.int64),
                np.array([0], np.int32),
                np.array([[nose, tail]], np.float32),
                np.full((1, 2), 0.9, np.float32),
                np.zeros((1, 2), np.int8),
            )
    _name_keypoint_run(recording)
    return recording


def with_heading(recording):
    """The embedding was aligned by nose (front) and tail (back)."""
    _name_keypoint_run(recording, heading=True)
    return recording


def walking(tmp_path, *, keypoints=True, heading=False, kinematics=True):
    """Five frames of two animals walking right, one row apart: track 0's 10x10 mask at
    ``x0 = 4 + 6i`` (rows 4..14) and track 1's at the same x (rows 30..40), every frame. The
    embedding ``emb_01`` covers every frame of track 0, and with ``kinematics`` its §14
    sidecar is written through the real backfill, so the centre trail has rows to read.

    With ``keypoints``, track 0's points walk with it rigidly — ``tail`` (x0, 9), ``nose``
    (x0 + 10, 9), ``left_paw`` (x0 + 5, 5) and ``right_paw`` (x0 + 5, 13) — so each point's
    path is a row of its own, in its own role colour (midline green, left blue, right
    vermillion). ``heading`` adds nose-front / tail-back for the aligned pose.
    """
    recording = _recording_with_detection(tmp_path)
    ddir = recording.dir / "detections" / "det_01"
    shutil.rmtree(ddir / "part_001.zarr")
    with DetectionPartWriter(ddir / "part_001.zarr") as writer:
        for i in range(5):
            x0 = 4 + 6 * i
            for track, y0 in ((0, 4), (1, 30)):
                box = (x0 / W, y0 / H, (x0 + 10) / W, (y0 + 10) / H)
                writer.add_row(
                    i,
                    int(i * 1e9 / 30),
                    track,
                    "mouse",
                    0.9,
                    box,
                    rle((slice(y0, y0 + 10), slice(x0, x0 + 10))),
                )
    adir = recording.dir / "tracking" / "track_01" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        for _ in range(5):
            writer.add_object_id(0)
            writer.add_object_id(1)
    _tracking_run(recording)
    frames = np.arange(5)
    write_embedding_run(
        recording.dir,
        np.eye(5, 4, dtype=np.float32),
        frames,
        (frames * 1e9 / 30).astype(np.int64),
        tracking_run_id="track_01",
        constructor="full_frame",
        renderer_spec=_INPUT_SPEC,
    )
    if kinematics:
        # The backfill decodes masks at the detection run's size, which it reads off the
        # run's envelope; a hand-written run has to write that much of one.
        (ddir / "job.json").write_text(
            json.dumps(
                {"schema_version": 1, "status": "complete", "mask_resolution": [H, W]}
            ),
            encoding="utf-8",
        )
        kinematics_backfill.backfill_run(recording.dir, "emb_01")
    if not keypoints:
        return recording
    run_dir = keypoint_runs.track_run_dir(recording.dir, "kt_01")
    keypoint_runs.write_track_yaml(
        run_dir,
        {
            "schema_version": 1,
            "id": "kt_01",
            "created_at": "2026-10-22T00:00:00Z",
            "detect_run_id": "kd_01",
            "tracking_run_id": "track_01",
            "method": "despike_smooth",
            "config": {},
            "skeleton": WALKER.to_dict(),
            "skeleton_digest": WALKER.digest(),
        },
    )
    path = keypoint_runs.track_part_path(run_dir, 0, "part_001")
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.array(
        [
            [
                [(4 + 6 * i + dx) / W, y / H]
                for dx, y in ((10, 9), (5, 5), (5, 13), (0, 9))
            ]
            for i in range(5)
        ],
        np.float32,
    )
    with KeypointTrackPartWriter(path, WALKER.names) as writer:
        writer.add_rows(
            np.arange(5),
            (np.arange(5) * 1e9 / 30).astype(np.int64),
            np.zeros(5, np.int32),
            points,
            np.full((5, 4), 0.9, np.float32),
            np.zeros((5, 4), np.int8),
        )
    _name_keypoint_run(recording, heading=heading)
    return recording


def count_decodes(monkeypatch) -> list:
    """Every call into the video decode, as ``(seam, thread id)``: the cursor every preview
    surface reads frames through (``FrameCursor.frame``), and any container opened at all
    (``av.open``), so a decode reached by another route is counted too."""
    import av

    from detail.storage.video import FrameCursor

    calls: list = []
    cursor_frame, av_open = FrameCursor.frame, av.open

    def frame(self, frame_idx):
        calls.append(("FrameCursor.frame", threading.get_ident()))
        return cursor_frame(self, frame_idx)

    def opened(*args, **kwargs):
        calls.append(("av.open", threading.get_ident()))
        return av_open(*args, **kwargs)

    monkeypatch.setattr(FrameCursor, "frame", frame)
    monkeypatch.setattr(av, "open", opened)
    return calls
