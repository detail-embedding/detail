"""``tracks.zarr``'s ``n_gaps`` counts frames the run looked at and missed, at any stride (#32).

**The class: a count over source frames where only every Nth was processed.** ``n_gaps`` was
``(last - first + 1) - n_present``, over source frames. At ``every_n_frames`` 2 every frame the
run never looked at counted as a miss, so a track that missed 5 frames read 11 — the same
mistake `run_preview.frames_processed` exists to avoid for the GUI's coverage line. §13 defines
the column as "total missed frames within the track span", and §4 says a stride selects which
frames are looked at and that a tracker's ``max_missed`` counts frames the run *processed*: a
frame never looked at was not missed.

**How it hid.** Every tracker fixture ran at stride 1, where the two counts agree. And the
processed frames of a recording with several parts are not every Nth *global* frame: each
part's own frame 0 is processed, so a part that does not start on a multiple of the stride
shifts the grid. Counting ``(last - first) // stride + 1`` gets that wrong, which is why the
multi-part case is here.

**The rule.** `trackers._common.processed_frames_between` is the one count of processed frames
in a span; every tracker's ``n_gaps`` is that minus its present frames. Data-driven over
``TRACKERS``, so a new tracker is held to it the day it registers.
"""

import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

import detail.pipeline.tracking  # noqa: F401 — imports + registers all TRACKERS
from detail.core.dataclasses import TrackingJob
from detail.core.registries import TRACKERS
from detail.storage.zarr_io import DetectionPartWriter, PartZarrReader
from detail.trackers import _common

MH, MW = 20, 20
_BOX = (0.3, 0.3, 0.6, 0.6)


def _rect_rle(bbox):
    x1, y1, x2, y2 = bbox
    m = np.zeros((MH, MW), np.uint8)
    m[int(y1 * MH) : int(y2 * MH), int(x1 * MW) : int(x2 * MW)] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _run(tmp_path, parts: list[list[int]]):
    """A detection run with one subject on each listed (part-local) frame of each part."""
    det = tmp_path / "det"
    det.mkdir()
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [MH, MW]}), encoding="utf-8"
    )
    for index, frames in enumerate(parts, start=1):
        with DetectionPartWriter(det / f"part_{index:03d}.zarr") as w:
            for f in frames:
                w.add_row(f, f * 1000, 0, "mouse", 0.9, _BOX, _rect_rle(_BOX))
    return det


def _track(tmp_path, tracker_id, parts, every_n_frames, offsets=None):
    tracker = TRACKERS[tracker_id]()
    job = TrackingJob(
        job_id="track_01",
        recording_id="recording_001",
        track_id="track_01",
        detection_run_id="det_01",
        tracker_id=tracker_id,
        hyperparams=tracker.validate_hyperparams({}),
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
        every_n_frames=every_n_frames,
    )
    out = tmp_path / "out"
    tracker.track(job, _run(tmp_path, parts), out, part_frame_offsets=offsets)
    return PartZarrReader(out / "tracks.zarr").read()


@pytest.mark.parametrize("tracker_id", sorted(TRACKERS))
def test_n_gaps_counts_processed_frames_missed_at_a_stride(tmp_path, tracker_id):
    """Stride 2 over frames 0-30: 16 frames processed, the subject on 11, so 5 missed. The
    source-frame count said 31 - 11 = 20."""
    missing = {8, 10, 12, 20, 22}
    present = [f for f in range(0, 31, 2) if f not in missing]
    tracks = _track(tmp_path, tracker_id, [present], every_n_frames=2)
    assert tracks["first_frame"].tolist() == [0]
    assert tracks["last_frame"].tolist() == [30]
    assert tracks["n_detections"].tolist() == [11]
    assert tracks["n_gaps"].tolist() == [5]


@pytest.mark.parametrize("tracker_id", sorted(TRACKERS))
def test_n_gaps_follows_each_parts_own_grid(tmp_path, tracker_id):
    """Part 1 is 11 frames, so part 2's frame 0 is global frame 11, and at stride 2 the run
    processed 0, 2, ..., 10 and then 11, 13, 15. Present on 0, 2, 4, 11 and 15: of the 9
    processed frames in 0-15, 4 were missed (6, 8, 10 and 13). Every 2nd global frame would
    say 3; source frames would say 11."""
    tracks = _track(
        tmp_path,
        tracker_id,
        [[0, 2, 4], [0, 4]],
        every_n_frames=2,
        offsets=[0, 11],
    )
    assert tracks["first_frame"].tolist() == [0]
    assert tracks["last_frame"].tolist() == [15]
    assert tracks["n_gaps"].tolist() == [4]


@pytest.mark.parametrize("tracker_id", sorted(TRACKERS))
def test_n_gaps_at_stride_one_is_unchanged(tmp_path, tracker_id):
    """Every frame processed: the two counts agree, so a stride-1 run reads as it always did."""
    tracks = _track(tmp_path, tracker_id, [[0, 1, 5, 6]], every_n_frames=1)
    assert tracks["n_gaps"].tolist() == [3]


def test_processed_frames_between_counts_each_parts_grid():
    count = _common.processed_frames_between
    assert count(0, 30, 2, [0]) == 16
    assert count(0, 15, 2, [0, 11]) == 9
    assert count(3, 9, 4, [0]) == 2  # 4 and 8
    assert count(5, 5, 2, [0]) == 0  # a single unprocessed frame
    assert count(7, 3, 1, [0]) == 0  # an empty span
    assert count(0, 9, 1, None) == 10
    # 13,16,19 · 20,23,26,29 · 30,33,36,39: each part's own frame 0 restarts the grid
    assert count(12, 40, 3, [0, 10, 20, 30]) == 3 + 4 + 4
