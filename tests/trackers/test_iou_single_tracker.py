import json

import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.core.dataclasses import PolygonKeyframes, TrackingJob
from detail.core.registries import TRACKERS
from detail.storage.zarr_io import DetectionPartWriter, PartZarrReader
from detail.trackers.iou_single_tracker import IoUSingleTracker

MH, MW = 20, 20  # mask resolution (h, w)


def _encode(m: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _rect(bbox) -> str:
    """A filled rectangle over the (normalized) bbox — one connected component."""
    x1, y1, x2, y2 = bbox
    m = np.zeros((MH, MW), np.uint8)
    m[int(y1 * MH) : int(y2 * MH), int(x1 * MW) : int(x2 * MW)] = 1
    return _encode(m)


def _mask_over(x1n, y1n, x2n, y2n) -> str:
    """A filled rectangle over an explicit normalized region (for a mask whose
    pixels differ from its declared bbox — a reflection)."""
    m = np.zeros((MH, MW), np.uint8)
    m[int(y1n * MH) : int(y2n * MH), int(x1n * MW) : int(x2n * MW)] = 1
    return _encode(m)


def _det(bbox, conf, cname="rat", mask=None):
    return (bbox, float(conf), cname, mask if mask is not None else "")


def _write_part(path, frames):
    """frames: list of (frame_idx, [(_det(...)), ...])."""
    with DetectionPartWriter(path) as w:
        for frame_idx, dets in frames:
            for d_idx, (bbox, conf, cname, mrle) in enumerate(dets):
                w.add_row(frame_idx, frame_idx * 1000, d_idx, cname, conf, bbox, mrle)


def _det_run(tmp_path):
    det = tmp_path / "det"
    det.mkdir()
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [MH, MW]}), encoding="utf-8"
    )
    return det


def _job(track_id="track_01", **hyperparams):
    return TrackingJob(
        job_id=track_id,
        recording_id="recording_001",
        track_id=track_id,
        detection_run_id="det_01",
        tracker_id="iou_single_tracker",
        hyperparams=hyperparams,
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
    )


def _read(path):
    return PartZarrReader(path).read()


def _assign(out, part="part_001.zarr"):
    return _read(out / "assignments" / part)["object_id"].tolist()


# --- registry + hyperparams ------------------------------------------------- #
def test_registered_in_trackers():
    assert TRACKERS["iou_single_tracker"] is IoUSingleTracker


def test_validate_hyperparams_defaults():
    assert IoUSingleTracker().validate_hyperparams({}) == {
        "conf_threshold": 0.05,
        "secondary_conf": 0.05,
        "merge_iou": 0.1,
        "merge_mask_iou": 0.5,
        "max_merge": 0,
        "iou_weight": 1.0,
        "max_missed": 30,
        "require_contiguous": False,
    }


def test_validate_hyperparams_rejects_bool_for_float():
    # bool is an int subclass — a True typo must not silently become 1.0.
    with pytest.raises(ValueError, match="must be float"):
        IoUSingleTracker().validate_hyperparams({"iou_weight": True})


# --- IoU continuity vs raw confidence (the core win) ------------------------ #
def _continuity_scene(det):
    """Subject locks onto A over frames 0-2; frame 3 offers the subject at A
    (lower conf) and a far distractor (higher conf)."""
    a = (0.1, 0.1, 0.3, 0.3)  # subject's established location
    far = (0.7, 0.7, 0.9, 0.9)  # a disjoint distractor
    frames = [(f, [_det(a, 0.9)]) for f in range(3)]
    frames.append((3, [_det(a, 0.6), _det(far, 0.9)]))  # subject, then distractor
    _write_part(det / "part_001.zarr", frames)


def test_iou_continuity_keeps_subject_over_higher_conf_distractor(tmp_path):
    det = _det_run(tmp_path)
    _continuity_scene(det)
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(), det, out)
    assign = _assign(out)
    # frames 0-2: subject; frame 3: [subject-at-A=0, far-distractor=-1]
    assert assign == [0, 0, 0, 0, -1]


def test_without_continuity_the_distractor_steals_the_subject(tmp_path):
    # Revert the fix: iou_weight=0 removes the continuity bonus, so selection is by
    # confidence alone and the higher-conf distractor wins frame 3.
    det = _det_run(tmp_path)
    _continuity_scene(det)
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(iou_weight=0.0), det, out)
    assign = _assign(out)
    assert assign == [0, 0, 0, -1, 0]  # frame 3: [subject-at-A=-1, distractor=0]


# --- coasting through gaps (max_missed) ------------------------------------- #
def _gap_scene(det):
    """Subject at A over frames 0-2, empty frames 3-4, then frame 5 offers the
    subject at A (lower conf) and a far distractor (higher conf)."""
    a = (0.1, 0.1, 0.3, 0.3)
    far = (0.7, 0.7, 0.9, 0.9)
    frames = [(f, [_det(a, 0.9)]) for f in range(3)]
    frames.append((5, [_det(a, 0.6), _det(far, 0.9)]))
    _write_part(det / "part_001.zarr", frames)


def test_coast_through_short_gap_keeps_continuity(tmp_path):
    det = _det_run(tmp_path)
    _gap_scene(det)
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(max_missed=10), det, out)
    # still locked at frame 5 (missed=2 <= 10) -> continuity keeps the subject at A
    assert _assign(out) == [0, 0, 0, 0, -1]


def test_reacquire_by_confidence_after_long_gap(tmp_path):
    det = _det_run(tmp_path)
    _gap_scene(det)
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(max_missed=1), det, out)
    # lock lost by frame 5 (missed=2 > 1) -> re-acquire by conf -> distractor wins
    assert _assign(out) == [0, 0, 0, -1, 0]


def test_zero_detection_frames_age_the_subject(tmp_path):
    # The empty gap is *frames with no rows at all* (§9) — the full-span walk must
    # still age `missed` across them, or the lock never expires. Revert the
    # full-span walk (visit only frames with rows) and this flips to continuity.
    det = _det_run(tmp_path)
    _gap_scene(det)  # frames 3,4 have no rows
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(max_missed=1), det, out)
    assert _assign(out) == [0, 0, 0, -1, 0]  # aged out -> distractor wins by conf


# --- fragment merge + the mask-IoU gate ------------------------------------- #
def test_mask_iou_gate_rejects_reflection(tmp_path):
    det = _det_run(tmp_path)
    subj_box = (0.1, 0.1, 0.5, 0.5)
    subj_mask = _mask_over(0.1, 0.1, 0.3, 0.3)  # pixels in the top-left of the bbox
    refl_box = (0.2, 0.2, 0.6, 0.6)  # bbox overlaps the subject (passes merge_iou)
    refl_mask = _mask_over(0.4, 0.4, 0.6, 0.6)  # but pixels are disjoint (~0 mask IoU)
    _write_part(
        det / "part_001.zarr",
        [
            (
                0,
                [
                    _det(subj_box, 0.9, mask=subj_mask),
                    _det(refl_box, 0.3, mask=refl_mask),
                ],
            )
        ],
    )
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(max_merge=1), det, out)
    assert _assign(out) == [0, -1]  # reflection rejected by the mask-IoU floor


def test_mask_gate_disabled_merges_the_overlapping_bbox(tmp_path):
    # merge_mask_iou=0 disables the mask floor, so the bbox-overlapping reflection
    # is merged — proving the mask gate is what rejected it above.
    det = _det_run(tmp_path)
    subj_box = (0.1, 0.1, 0.5, 0.5)
    refl_box = (0.2, 0.2, 0.6, 0.6)
    _write_part(
        det / "part_001.zarr",
        [
            (
                0,
                [
                    _det(subj_box, 0.9, mask=_mask_over(0.1, 0.1, 0.3, 0.3)),
                    _det(refl_box, 0.3, mask=_mask_over(0.4, 0.4, 0.6, 0.6)),
                ],
            )
        ],
    )
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(max_merge=1, merge_mask_iou=0.0), det, out)
    assert _assign(out) == [0, 0]


def test_max_merge_zero_attaches_no_secondaries(tmp_path):
    # The validated default: no fragment merge, cleanest single silhouette.
    det = _det_run(tmp_path)
    subj = (0.1, 0.1, 0.4, 0.4)
    frag = (0.15, 0.15, 0.45, 0.45)  # overlaps, would merge if max_merge>0
    _write_part(det / "part_001.zarr", [(0, [_det(subj, 0.9), _det(frag, 0.4)])])
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(), det, out)  # max_merge=0 default
    assert _assign(out) == [0, -1]


# --- schema edges ----------------------------------------------------------- #
def test_single_row(tmp_path):
    det = _det_run(tmp_path)
    _write_part(det / "part_001.zarr", [(0, [_det((0.1, 0.1, 0.4, 0.4), 0.9)])])
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(), det, out)
    assert _assign(out) == [0]
    assert _read(out / "tracks.zarr")["track_id"].tolist() == [0]


def test_empty_part_among_parts(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.4, 0.4)
    _write_part(det / "part_001.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    with DetectionPartWriter(det / "part_002.zarr"):  # entirely empty part
        pass
    _write_part(det / "part_003.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(), det, out, part_frame_offsets=[0, 3, 6])
    assert _assign(out, "part_002.zarr") == []
    assert _assign(out, "part_001.zarr") == [0, 0, 0]
    assert _assign(out, "part_003.zarr") == [0, 0, 0]


def test_cross_part_continuity(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.4, 0.4)
    _write_part(det / "part_001.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    _write_part(det / "part_002.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(), det, out, part_frame_offsets=[0, 3])
    assert _assign(out, "part_001.zarr") == [0, 0, 0]
    assert _assign(out, "part_002.zarr") == [0, 0, 0]
    tracks = _read(out / "tracks.zarr")
    assert tracks["track_id"].tolist() == [0]
    assert tracks["n_detections"].tolist() == [6]
    assert tracks["first_frame"].tolist() == [0]
    assert tracks["last_frame"].tolist() == [5]


def test_zero_tracks_raises(tmp_path):
    det = _det_run(tmp_path)
    _write_part(
        det / "part_001.zarr",
        [(f, [_det((0.1, 0.1, 0.4, 0.4), 0.02)]) for f in range(3)],  # all below conf
    )
    with pytest.raises(ValueError, match="zero tracks produced"):
        IoUSingleTracker().track(_job(), det, tmp_path / "track_01")


# --- arena ------------------------------------------------------------------ #
def test_arena_drops_detection_with_center_outside(tmp_path):
    det = _det_run(tmp_path)
    outside = (0.85, 0.85, 0.95, 0.95)  # center (0.9, 0.9)
    inside = (0.2, 0.2, 0.3, 0.3)  # center (0.25, 0.25)
    frames = [(f, [_det(outside, 0.95), _det(inside, 0.9)]) for f in range(3)]
    _write_part(det / "part_001.zarr", frames)
    out = tmp_path / "track_01"
    polygons = {
        "main": PolygonKeyframes(
            n_vertices=4,
            keyframes=[(0, [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)])],
        )
    }
    IoUSingleTracker().track(_job(), det, out, arena_polygons=polygons)
    # the outside (higher-conf) row is filtered before selection -> inside is subject
    assert _assign(out) == [-1, 0, -1, 0, -1, 0]


# --- track.yaml ------------------------------------------------------------- #
def test_track_yaml_contents(tmp_path):
    det = _det_run(tmp_path)
    _write_part(
        det / "part_001.zarr",
        [(f, [_det((0.1, 0.1, 0.4, 0.4), 0.9)]) for f in range(3)],
    )
    out = tmp_path / "track_01"
    IoUSingleTracker().track(_job(), det, out)
    text = (out / "track.yaml").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("schema_version:")
    data = yaml.safe_load(text)
    assert data["id"] == "track_01"
    assert data["tracker_id"] == "iou_single_tracker"
    assert data["status"] == "complete"
    assert data["n_tracks"] == 1
    assert data["hyperparams"]["max_missed"] == 30
