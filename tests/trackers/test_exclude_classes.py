"""A tracker never takes a left-out class as the subject or merges it (§13 ``exclude_classes``).

**The class.** A detector that sees a thing beside the animal, a tether's cable most of all,
reports it as a detection whose box overlaps the animal's. ``anchor_tracker`` gates fragments
on box overlap alone, so it merged the cable into the subject's mask on a fifth of a tethered
pilot's frames, and on a frame where the cable outscored the animal it would anchor on the
cable. Training a ``cable`` class (§12 *Tether or cable in the video*) only helps if tracking
then leaves that class out, at the one place both trackers read a frame's candidates
(``_common.group_by_frame``).

**The alignment half.** The excluded rows keep their places and are written ``-1``, so each
part's assignments stay parallel to all of its detection rows. A filter that dropped the rows
instead would shift every later row's assignment onto the wrong detection.
"""

import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from detail.core.dataclasses import TrackingJob
from detail.core.registries import TRACKERS
from detail.storage.zarr_io import DetectionPartWriter, PartZarrReader
from detail.trackers import anchor_tracker  # noqa: F401  (registers the tracker)
from detail.trackers import iou_single_tracker  # noqa: F401  (registers the tracker)

MH, MW = 20, 20

MOUSE = (0.2, 0.2, 0.6, 0.6)
CABLE = (
    0.5,
    0.1,
    0.7,
    0.5,
)  # overlaps the mouse's box, as a cable leaving the head does

#: frame → its detections (bbox, confidence, class). Frame 0: the animal with the cable beside
#: it, weak enough to be a merged piece. Frame 1: the cable alone, confident enough to anchor.
FRAMES = [
    (0, [(MOUSE, 0.9, "mouse"), (CABLE, 0.3, "cable")]),
    (1, [(CABLE, 0.9, "cable")]),
    (2, [(MOUSE, 0.9, "mouse")]),
]
CABLE_ROWS = [1, 2]
MOUSE_ROWS = [0, 3]

HYPERPARAMS = {
    "anchor_tracker": {"conf_threshold": 0.5, "secondary_conf": 0.1},
    "iou_single_tracker": {"conf_threshold": 0.5},
}


def _rect(bbox) -> str:
    x1, y1, x2, y2 = bbox
    m = np.zeros((MH, MW), np.uint8)
    m[int(y1 * MH) : int(y2 * MH), int(x1 * MW) : int(x2 * MW)] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _detection_run(tmp_path):
    det = tmp_path / "det"
    det.mkdir()
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [MH, MW]}), encoding="utf-8"
    )
    with DetectionPartWriter(det / "part_001.zarr") as w:
        for frame_idx, dets in FRAMES:
            for d_idx, (bbox, conf, cname) in enumerate(dets):
                w.add_row(
                    frame_idx, frame_idx * 1000, d_idx, cname, conf, bbox, _rect(bbox)
                )
    return det


def _assignments(tmp_path, tracker_id, exclude_classes):
    det = _detection_run(tmp_path)
    out = tmp_path / f"out_{tracker_id}_{len(exclude_classes)}"
    job = TrackingJob(
        job_id="track_01",
        recording_id="recording_001",
        track_id="track_01",
        detection_run_id="det_01",
        tracker_id=tracker_id,
        hyperparams=HYPERPARAMS[tracker_id],
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
        exclude_classes=list(exclude_classes),
    )
    TRACKERS[tracker_id]().track(job, det, out)
    return PartZarrReader(out / "assignments" / "part_001.zarr").read()["object_id"]


@pytest.mark.parametrize("tracker_id", sorted(HYPERPARAMS))
def test_a_left_out_class_is_never_the_subject_or_a_merged_piece(tmp_path, tracker_id):
    assigned = _assignments(tmp_path, tracker_id, ["cable"])
    assert len(assigned) == sum(len(dets) for _, dets in FRAMES)  # parallel to the rows
    assert [int(assigned[i]) for i in CABLE_ROWS] == [-1, -1]
    assert all(int(assigned[i]) >= 0 for i in MOUSE_ROWS)


@pytest.mark.parametrize("tracker_id", sorted(HYPERPARAMS))
def test_without_the_exclusion_the_cable_is_tracked(tmp_path, tracker_id):
    """The case above is live: with nothing left out, some cable row is assigned."""
    assigned = _assignments(tmp_path, tracker_id, [])
    assert any(int(assigned[i]) >= 0 for i in CABLE_ROWS)
