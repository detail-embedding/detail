"""Cadence net (the prevention): every registered tracker must stream *intermediate*
progress on a single-part run — a live bar, not one event at the end.

Data-driven over ``TRACKERS`` so a new tracker is covered automatically; this is
the guard for the recurring "no live bar on a single part" bug. Patching the one
shared ``core.progress.PROGRESS_INTERVAL_S`` to 0.0 makes every tracker forward
its in-loop emits deterministically — so a tracker that only emits at part
boundaries (no in-loop emit) fails here.
"""

import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

import detail.pipeline.tracking  # noqa: F401 — imports + registers all TRACKERS
from detail.core import progress as progress_mod
from detail.core.dataclasses import TrackingJob
from detail.core.registries import TRACKERS
from detail.storage.zarr_io import DetectionPartWriter

MH, MW = 20, 20
N_FRAMES = 8


def _rect_rle(bbox):
    x1, y1, x2, y2 = bbox
    m = np.zeros((MH, MW), np.uint8)
    m[int(y1 * MH) : int(y2 * MH), int(x1 * MW) : int(x2 * MW)] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _single_part_run(tmp_path):
    """One part, one contiguous high-conf 'mouse' per frame over N_FRAMES — valid
    input for every tracker (anchor needs a contiguous mask; iou links by bbox)."""
    det = tmp_path / "det"
    det.mkdir()
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [MH, MW]}), encoding="utf-8"
    )
    bbox = (0.3, 0.3, 0.6, 0.6)
    with DetectionPartWriter(det / "part_001.zarr") as w:
        for f in range(N_FRAMES):
            w.add_row(f, f * 1000, 0, "mouse", 0.9, bbox, _rect_rle(bbox))
    return det


def _job(tracker_id, hyperparams):
    return TrackingJob(
        job_id="track_01",
        recording_id="recording_001",
        track_id="track_01",
        detection_run_id="det_01",
        tracker_id=tracker_id,
        hyperparams=hyperparams,
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
    )


@pytest.mark.parametrize("tracker_id", sorted(TRACKERS))
def test_tracker_streams_intermediate_progress(tmp_path, monkeypatch, tracker_id):
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    tracker = TRACKERS[tracker_id]()
    det = _single_part_run(tmp_path)
    job = _job(tracker_id, tracker.validate_hyperparams({}))
    events = []
    tracker.track(job, det, tmp_path / "out", progress_cb=events.append)

    frame_events = [e for e in events if getattr(e, "frames_total", 0)]
    assert frame_events, f"{tracker_id} emitted no frame-progress events"
    total = frame_events[-1].frames_total
    # the live-bar guarantee: an intermediate frames_done below the total appears,
    # not just one terminal event (the recurring single-part "no live bar" bug)
    assert any(0 < e.frames_done < total for e in frame_events), (
        f"{tracker_id} did not stream intermediate progress on a single part "
        f"(frames_done seen: {[e.frames_done for e in frame_events]}, total={total})"
    )
    assert frame_events[-1].frames_done == total
    # #18: the runner populates eta_s (not None) on a mid-run event.
    mid = [e for e in frame_events if 0 < e.frames_done < total]
    assert any(e.eta_s is not None for e in mid), f"{tracker_id} left eta_s unpopulated"
