"""Resume net (Tenet 12 / Section 7): every tracker resumes at part boundaries.

Data-driven over ``TRACKERS`` so a new tracker is covered automatically. Two
guarantees: (1) a run interrupted after a completed part, then re-invoked on the
same run dir, produces output **identical** to an uninterrupted run — via the
``part_handoff.json`` snapshot; (2) re-invoking a *complete* run is idempotent and
never doubles rows (the assignment/summary writers open ``mode="a"``, so a naive
re-run would append).
"""

import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

import detail.pipeline.tracking  # noqa: F401 — imports + registers all TRACKERS
from detail.core.dataclasses import TrackingJob
from detail.core.registries import TRACKERS
from detail.storage.zarr_io import DetectionPartWriter, PartZarrReader

MH, MW = 20, 20
N_PARTS = 3
N_PER_PART = 3
BOX = (0.3, 0.3, 0.6, 0.6)  # fixed box: consecutive-frame IoU 1.0 keeps one track
OFFSETS = [i * N_PER_PART for i in range(N_PARTS)]


class _StopAfter:
    """A stop_event that reports 'set' only after ``n_parts`` top-of-loop checks,
    so the run completes exactly ``n_parts`` parts and then stops (a mid-run
    interrupt at a clean part boundary)."""

    def __init__(self, n_parts: int) -> None:
        self._n = n_parts
        self._checks = 0

    def is_set(self) -> bool:
        self._checks += 1
        return self._checks > self._n


def _rect_rle(bbox) -> str:
    x1, y1, x2, y2 = bbox
    m = np.zeros((MH, MW), np.uint8)
    m[int(y1 * MH) : int(y2 * MH), int(x1 * MW) : int(x2 * MW)] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _multi_part_run(tmp_path):
    """A detection run with N_PARTS parts, each a contiguous high-conf 'rat' per
    frame — valid input for every tracker."""
    det = tmp_path / "det"
    det.mkdir(parents=True)
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [MH, MW]}), encoding="utf-8"
    )
    rle = _rect_rle(BOX)
    for part in range(N_PARTS):
        with DetectionPartWriter(det / f"part_{part + 1:03d}.zarr") as w:
            for f in range(N_PER_PART):
                w.add_row(f, f * 1000, 0, "rat", 0.9, BOX, rle)
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


def _snapshot(out) -> dict:
    """Everything a downstream stage reads: per-part assignments + the summary +
    the run yaml — for an exact identical-output comparison."""
    assigns = {
        p.name: PartZarrReader(p).read()["object_id"].tolist()
        for p in sorted((out / "assignments").glob("part_*.zarr"))
    }
    tracks = PartZarrReader(out / "tracks.zarr").read()
    return {
        "assignments": assigns,
        "tracks": {k: np.asarray(v).tolist() for k, v in tracks.items()},
        "track_yaml": (out / "track.yaml").read_text(encoding="utf-8-sig"),
    }


@pytest.mark.parametrize("tracker_id", sorted(TRACKERS))
def test_resume_matches_uninterrupted_run(tmp_path, tracker_id):
    tracker = TRACKERS[tracker_id]
    params = tracker().validate_hyperparams({})

    baseline_det = _multi_part_run(tmp_path / "a")
    baseline_out = tmp_path / "a" / "track_01"
    tracker().track(
        _job(tracker_id, params), baseline_det, baseline_out, part_frame_offsets=OFFSETS
    )

    # Interrupt after the first part completes, then re-invoke on the same run dir.
    det = _multi_part_run(tmp_path / "b")
    out = tmp_path / "b" / "track_01"
    tracker().track(
        _job(tracker_id, params),
        det,
        out,
        part_frame_offsets=OFFSETS,
        stop_event=_StopAfter(1),
    )
    handoff = out / "part_handoff.json"
    assert handoff.exists()
    assert json.loads(handoff.read_text(encoding="utf-8-sig"))["parts_completed"] == 1
    # only the first part was written before the interrupt
    assert not (out / "assignments" / "part_002.zarr").exists()

    tracker().track(
        _job(tracker_id, params), det, out, part_frame_offsets=OFFSETS
    )  # resume

    assert _snapshot(out) == _snapshot(baseline_out)


@pytest.mark.parametrize("tracker_id", sorted(TRACKERS))
def test_reinvoke_complete_run_is_idempotent(tmp_path, tracker_id):
    tracker = TRACKERS[tracker_id]
    params = tracker().validate_hyperparams({})
    det = _multi_part_run(tmp_path)
    out = tmp_path / "track_01"

    tracker().track(_job(tracker_id, params), det, out, part_frame_offsets=OFFSETS)
    first = _snapshot(out)
    # Re-invoking a finished run must not append duplicate assignment/summary rows.
    tracker().track(_job(tracker_id, params), det, out, part_frame_offsets=OFFSETS)
    assert _snapshot(out) == first
