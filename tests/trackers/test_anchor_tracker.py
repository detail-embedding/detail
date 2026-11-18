import json

import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.core.dataclasses import PolygonKeyframes, TrackingJob
from detail.core.registries import TRACKERS
from detail.storage.zarr_io import DetectionPartWriter, PartZarrReader
from detail.trackers.anchor_tracker import AnchorTracker

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


def _frag() -> str:
    """Two disjoint blobs — a non-contiguous mask."""
    m = np.zeros((MH, MW), np.uint8)
    m[2:5, 2:5] = 1
    m[12:15, 12:15] = 1
    return _encode(m)


def _det(bbox, conf, cname="mouse", mask=None):
    """A detection tuple; ``mask`` defaults to a filled-bbox (contiguous) mask."""
    return (bbox, float(conf), cname, mask if mask is not None else _rect(bbox))


def _write_part(path, frames):
    """frames: list of (frame_idx, [(_det(...)), ...])."""
    with DetectionPartWriter(path) as w:
        for frame_idx, dets in frames:
            for d_idx, (bbox, conf, cname, mrle) in enumerate(dets):
                w.add_row(frame_idx, frame_idx * 1000, d_idx, cname, conf, bbox, mrle)


def _det_run(tmp_path):
    """A detection run dir with the mask_resolution envelope the tracker reads."""
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
        tracker_id="anchor_tracker",
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
    assert TRACKERS["anchor_tracker"] is AnchorTracker


def test_validate_hyperparams_defaults_and_override():
    t = AnchorTracker()
    assert t.validate_hyperparams({}) == {
        "num_subjects": 1,
        "conf_threshold": 0.1,
        "secondary_conf": 0.05,
        "iou_threshold": 0.1,
        "max_secondary": 2,
        # Off: on, the gate discards a frame whose only candidates have split masks, and
        # that costs frames on real data (see the schema's own note).
        "require_contiguous": False,
    }
    assert t.validate_hyperparams({"conf_threshold": 0.3})["conf_threshold"] == 0.3
    assert t.validate_hyperparams({"require_contiguous": False})[
        "require_contiguous"
    ] is (False)


def test_validate_hyperparams_rejects_unknown_wrong_type_and_multi_subject():
    t = AnchorTracker()
    with pytest.raises(ValueError, match="unknown hyperparams"):
        t.validate_hyperparams({"nope": 1})
    with pytest.raises(ValueError, match="must be float"):
        t.validate_hyperparams({"conf_threshold": "x"})
    with pytest.raises(ValueError, match="must be bool"):
        t.validate_hyperparams({"require_contiguous": 1})
    with pytest.raises(ValueError, match="num_subjects=1 only"):
        t.validate_hyperparams({"num_subjects": 2})


# --- anchoring -------------------------------------------------------------- #
def test_single_detection_is_anchor(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.4, 0.4)
    _write_part(det / "part_001.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)

    assert _assign(out) == [0, 0, 0]
    tracks = _read(out / "tracks.zarr")
    assert tracks["track_id"].tolist() == [0]
    assert tracks["n_detections"].tolist() == [3]
    assert tracks["first_frame"].tolist() == [0]
    assert tracks["last_frame"].tolist() == [2]


def test_top_confidence_is_anchor_others_dropped(tmp_path):
    """More candidates than num_subjects: only the best (and overlapping
    secondaries) survive; disjoint extras are unassigned even above threshold."""
    det = _det_run(tmp_path)
    anchor = (0.1, 0.1, 0.4, 0.4)
    far = (0.6, 0.6, 0.9, 0.9)  # high conf but disjoint from the anchor
    frames = [(f, [_det(anchor, 0.9), _det(far, 0.8)]) for f in range(3)]
    _write_part(det / "part_001.zarr", frames)
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)

    # per frame: [anchor=0, disjoint-high-conf=-1]
    assert _assign(out) == [0, -1, 0, -1, 0, -1]
    assert _read(out / "tracks.zarr")["n_detections"].tolist() == [3]


def test_secondary_overlap_attached(tmp_path):
    det = _det_run(tmp_path)
    anchor = (0.1, 0.1, 0.4, 0.4)
    overlap = (0.2, 0.2, 0.5, 0.5)  # IoU ~0.29 with anchor
    frames = [(f, [_det(anchor, 0.9), _det(overlap, 0.07)]) for f in range(2)]
    _write_part(det / "part_001.zarr", frames)
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)

    # both rows of each frame are the subject (object_id 0)
    assert _assign(out) == [0, 0, 0, 0]
    tracks = _read(out / "tracks.zarr")
    assert tracks["n_detections"].tolist() == [2]  # frames present, not rows


def test_secondary_below_threshold_or_disjoint_unassigned(tmp_path):
    det = _det_run(tmp_path)
    anchor = (0.1, 0.1, 0.4, 0.4)
    too_low = (0.2, 0.2, 0.5, 0.5)  # overlaps but conf < secondary_conf
    disjoint = (0.6, 0.6, 0.9, 0.9)  # conf ok but no overlap
    _write_part(
        det / "part_001.zarr",
        [(0, [_det(anchor, 0.9), _det(too_low, 0.02), _det(disjoint, 0.6)])],
    )
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)
    assert _assign(out) == [0, -1, -1]


def test_max_secondary_caps_attachments(tmp_path):
    det = _det_run(tmp_path)
    anchor = (0.1, 0.1, 0.4, 0.4)
    s1 = (0.15, 0.15, 0.45, 0.45)
    s2 = (0.2, 0.2, 0.5, 0.5)
    s3 = (0.12, 0.12, 0.42, 0.42)
    _write_part(
        det / "part_001.zarr",
        [(0, [_det(anchor, 0.9), _det(s1, 0.08), _det(s2, 0.07), _det(s3, 0.06)])],
    )
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(max_secondary=2), det, out)
    # anchor + the two highest-confidence overlapping secondaries; the 3rd drops
    assert _assign(out) == [0, 0, 0, -1]


# --- contiguity gate -------------------------------------------------------- #
def test_fragmented_anchor_rejected_next_candidate_wins(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.5, 0.5)
    # row0: highest conf but non-contiguous mask; row1: lower conf, contiguous
    _write_part(
        det / "part_001.zarr",
        [(0, [_det(box, 0.9, mask=_frag()), _det(box, 0.5, mask=_rect(box))])],
    )
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(require_contiguous=True), det, out)
    # fragmented row rejected as anchor AND as secondary; contiguous row is the anchor
    assert _assign(out) == [-1, 0]


def test_require_contiguous_false_keeps_fragment_as_anchor(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.5, 0.5)
    _write_part(
        det / "part_001.zarr",
        [(0, [_det(box, 0.9, mask=_frag()), _det(box, 0.5, mask=_rect(box))])],
    )
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(require_contiguous=False), det, out)
    # without the gate the top-conf fragment is the anchor; the other overlaps -> 0
    assert _assign(out) == [0, 0]


def test_a_clearly_detected_subject_is_never_discarded_for_a_split_mask(tmp_path):
    """The frames a real cohort lost, as a test.

    **The class: a gate that vetoes the whole unit where degrading it would have done.**
    `select_subject` walks candidates in score order and *skips* every non-contiguous one,
    so a frame whose detections are all split gets no subject — `track` does
    `if anchor is None: continue` and the frame is simply gone. Nothing fails; the run
    completes, the record is well-formed, and the number a user reads is merely smaller.

    On real recordings of nearly motionless animals, at this tracker's own defaults,
    frames that **had a detection** — confident, mid-frame, the animal barely moving —
    were discarded. A mouse's mask splits constantly (a tail, a limb, a shadow across the
    body), and a motionless animal holding a pose whose mask stays split loses every frame
    of it.

    So: a confident detection in the middle of the frame keeps its subject, whatever
    shape the mask came back as.
    """
    det = _det_run(tmp_path)
    box = (0.3, 0.3, 0.6, 0.6)  # mid-frame, nowhere near an edge
    _write_part(
        det / "part_001.zarr",
        [(f, [_det(box, 0.93, mask=_frag())]) for f in range(5)],
    )
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)  # the defaults, as a user gets them

    assert _assign(out) == [0] * 5, "a confident, centred subject was thrown away"


def test_all_fragmented_raises_zero_tracks(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.5, 0.5)
    _write_part(
        det / "part_001.zarr", [(f, [_det(box, 0.9, mask=_frag())]) for f in range(3)]
    )
    with pytest.raises(ValueError, match="zero tracks produced"):
        AnchorTracker().track(_job(require_contiguous=True), det, tmp_path / "track_01")


def test_all_below_conf_raises_zero_tracks(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.4, 0.4)
    _write_part(det / "part_001.zarr", [(f, [_det(box, 0.02)]) for f in range(3)])
    with pytest.raises(
        ValueError, match="from 3 detection.*settings may be too strict"
    ):
        AnchorTracker().track(_job(), det, tmp_path / "track_01")


def test_a_detection_run_that_found_nothing_is_named_as_the_cause(tmp_path):
    """Blaming the tracker's settings sent the user to tune the wrong stage (#33)."""
    det = _det_run(tmp_path)
    _write_part(det / "part_001.zarr", [(f, []) for f in range(3)])
    with pytest.raises(
        ValueError, match="the detection run found no detections"
    ) as err:
        AnchorTracker().track(_job(), det, tmp_path / "track_01")
    assert "too strict" not in str(err.value)


# --- schema edges: gaps, empty parts, multi-part ---------------------------- #
def test_zero_detection_frames_are_skipped(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.4, 0.4)
    # frames 2,3,4 have no rows (zero-detection frames leave no rows, §9)
    frames = [(f, [_det(box, 0.9)]) for f in (0, 1, 5, 6)]
    _write_part(det / "part_001.zarr", frames)
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)

    assert _assign(out) == [0, 0, 0, 0]
    tracks = _read(out / "tracks.zarr")
    assert tracks["first_frame"].tolist() == [0]
    assert tracks["last_frame"].tolist() == [6]
    assert tracks["n_detections"].tolist() == [4]
    assert tracks["n_gaps"].tolist() == [3]  # frames 2,3,4 within the span


def test_empty_part_among_parts(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.4, 0.4)
    _write_part(det / "part_001.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    with DetectionPartWriter(det / "part_002.zarr"):  # entirely empty part
        pass
    _write_part(det / "part_003.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)

    assert _assign(out, "part_002.zarr") == []
    assert _assign(out, "part_001.zarr") == [0, 0, 0]
    assert _assign(out, "part_003.zarr") == [0, 0, 0]


def test_cross_part_continuity_single_subject(tmp_path):
    det = _det_run(tmp_path)
    box = (0.1, 0.1, 0.4, 0.4)
    _write_part(det / "part_001.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    _write_part(det / "part_002.zarr", [(f, [_det(box, 0.9)]) for f in range(3)])
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)

    assert _assign(out, "part_001.zarr") == [0, 0, 0]
    assert _assign(out, "part_002.zarr") == [0, 0, 0]
    tracks = _read(out / "tracks.zarr")
    assert tracks["track_id"].tolist() == [0]
    assert tracks["n_detections"].tolist() == [6]
    assert tracks["first_frame"].tolist() == [0]
    assert tracks["last_frame"].tolist() == [5]  # global frame index across parts


# --- arena (Section 10 + 13) ------------------------------------------------ #
def _arena(keyframes):
    return {"main": PolygonKeyframes(n_vertices=4, keyframes=keyframes)}


def test_arena_drops_detection_with_center_outside(tmp_path):
    det = _det_run(tmp_path)
    outside = (0.85, 0.85, 0.95, 0.95)  # center (0.9, 0.9)
    inside = (0.2, 0.2, 0.3, 0.3)  # center (0.25, 0.25)
    frames = [(f, [_det(outside, 0.95), _det(inside, 0.9)]) for f in range(3)]
    _write_part(det / "part_001.zarr", frames)
    out = tmp_path / "track_01"
    polygons = _arena([(0, [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)])])
    AnchorTracker().track(_job(), det, out, arena_polygons=polygons)
    # the outside (higher-conf) row is filtered before anchoring -> inside is the anchor
    assert _assign(out) == [-1, 0, -1, 0, -1, 0]


# --- track.yaml ------------------------------------------------------------- #
def test_track_yaml_counts_rows_and_frames(tmp_path):
    det = _det_run(tmp_path)
    anchor = (0.1, 0.1, 0.4, 0.4)
    overlap = (0.2, 0.2, 0.5, 0.5)
    _write_part(
        det / "part_001.zarr",
        [(f, [_det(anchor, 0.9), _det(overlap, 0.07)]) for f in range(3)],
    )
    out = tmp_path / "track_01"
    AnchorTracker().track(_job(), det, out)

    text = (out / "track.yaml").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("schema_version:")
    data = yaml.safe_load(text)
    assert data["id"] == "track_01"
    assert data["tracker_id"] == "anchor_tracker"
    assert data["subject_labels"] is None
    assert data["status"] == "complete"
    assert data["n_tracks"] == 1
    # 3 frames x (anchor + 1 secondary) = 6 assigned rows; none unassigned
    assert data["n_detections_assigned"] == 6
    assert data["n_detections_unassigned"] == 0
