"""Assignment, the first step of every keypoint track run (§26): pose instance → subject.

The class these guard: identity decided by box overlap, or greedily, or by pairing a tracking
run's assignment to detection rows by position. Each test builds the case where the wrong rule
gives a different answer; the Hungarian test was checked by swapping in a greedy match (it
fails).

A second class, since assignment stopped rasterising: **a few-pixel membership question answered
by decoding and dilating the whole object.** Assignment decoded every subject's full mask and
``cv2.dilate``d it, per subject per frame, to read three pixels — most of a keypoint
track run. ``inside_share`` now evaluates the grow kernel's footprint at the
points against the mask's run lengths (``core.rle``). ``_reference_inside_share`` below is the old
computation kept as the oracle; ``test_inside_share_matches_the_dilated_mask_reference`` holds the
two bit-equal over the masks, radii and points a rasteriser's edges hide, and is shown able to fail
by shifting the footprint one column. ``test_assignment_rasterises_no_mask`` is the count the speed
claim rests on: zero decodes, zero dilates. Swept: every other mask decode in the pipeline needs the
dense array (shape statistics, crops, drawing, export) and stays.
"""

from __future__ import annotations

import json
import tracemalloc

import cv2
import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.core import rle
from detail.pipeline import pose_match
from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter

H, W = 24, 32  # mask resolution (h, w); the recording's frame is 640 x 480


def _block(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _pts(*xy) -> np.ndarray:
    """One instance's points ``(K, 2)`` from (x, y) pairs, at pixel centres."""
    return np.asarray(xy, dtype=np.float64) + 0.5


def _rle(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _run(mask: np.ndarray) -> rle.RunMask:
    """The mask as the assignment reads it: through the stored string and the real parser."""
    return rle.RunMask.from_counts_str(_rle(mask), mask.shape)


def _runs(mask: np.ndarray) -> list[rle.RunMask]:
    return [_run(mask)]


def _dense(runs: list[rle.RunMask]) -> np.ndarray:
    return np.logical_or.reduce([r.to_array() for r in runs])


def _reference_inside_share(
    points_xy: np.ndarray, mask: np.ndarray, radius_px: float
) -> np.ndarray:
    """The share as assignment computed it before ``core.rle``: the whole decoded mask,
    cropped to its box plus the radius, ``cv2.dilate`` with the ellipse kernel, the points
    looked up. Kept verbatim as the oracle :func:`pose_match.inside_share` is held to.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    n = points_xy.shape[0]
    out = np.zeros(n, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    finite = np.isfinite(points_xy).all(axis=-1)
    n_placed = finite.sum(axis=1)
    if n == 0 or not finite.any() or not mask.any():
        return out
    h, w = mask.shape
    r = int(round(max(float(radius_px), 0.0)))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = max(int(rows[0]) - r, 0), min(int(rows[-1]) + r + 1, h)
    x0, x1 = max(int(cols[0]) - r, 0), min(int(cols[-1]) + r + 1, w)
    grown = mask[y0:y1, x0:x1].astype(np.uint8)
    if r > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        grown = cv2.dilate(grown, kernel)
    grown = grown > 0
    xi = np.floor(np.where(finite, points_xy[..., 0], -1.0)).astype(np.int64) - x0
    yi = np.floor(np.where(finite, points_xy[..., 1], -1.0)).astype(np.int64) - y0
    on_grid = (
        finite & (xi >= 0) & (xi < grown.shape[1]) & (yi >= 0) & (yi < grown.shape[0])
    )
    hit = np.zeros_like(finite)
    hit[on_grid] = grown[yi[on_grid], xi[on_grid]]
    placed = n_placed > 0
    out[placed] = hit.sum(axis=1)[placed] / n_placed[placed]
    return out


# --------------------------------------------------------------------------- #
# inside_share
# --------------------------------------------------------------------------- #
def test_inside_share_counts_the_share_of_placed_points():
    mask = _block(20, 20, 5, 15, 5, 15)
    three_in = _pts((6, 6), (7, 7), (8, 8), (1, 1))
    one_in = _pts((6, 6), (1, 1), (2, 2), (18, 18))
    share = pose_match.inside_share(np.stack([three_in, one_in]), _runs(mask), 0.0)
    assert share.dtype == np.float32
    np.testing.assert_allclose(share, [0.75, 0.25])
    assert share[0] >= 0.5 > share[1]


def test_inside_share_ignores_unplaced_points_and_counts_off_grid_outside():
    mask = _block(20, 20, 5, 15, 5, 15)
    inst = _pts((6, 6), (7, 7), (np.nan, np.nan), (-3, 40))
    share = pose_match.inside_share(inst[None], _runs(mask), 0.0)
    np.testing.assert_allclose(share, [2 / 3])


def test_mask_grow_pulls_an_edge_point_inside():
    mask = _block(20, 20, 5, 15, 5, 15)  # columns 5..14
    edge = _pts((16, 10))[None]  # two pixels right of the mask
    assert pose_match.inside_share(edge, _runs(mask), 0.0)[0] == 0.0
    assert pose_match.inside_share(edge, _runs(mask), 2.0)[0] == 1.0


def test_an_instance_with_no_placed_point_scores_zero_and_stays_unassigned():
    mask = _block(20, 20, 5, 15, 5, 15)
    empty = np.full((1, 4, 2), np.nan)
    assert pose_match.inside_share(empty, _runs(mask), 3.0)[0] == 0.0
    sid, inside = pose_match.match_instances(
        empty, np.array([0.9]), {1: _runs(mask)}, {1: 3.0}
    )
    assert sid.tolist() == [-1]
    assert np.isnan(inside[0])


def test_grow_radius_is_a_fraction_of_body_scale():
    assert pose_match.grow_radius_px(40.0, 0.05) == pytest.approx(2.0)
    assert pose_match.grow_radius_px(40.0, 0.0) == 0.0


# --------------------------------------------------------------------------- #
# match_instances
# --------------------------------------------------------------------------- #
def test_two_poses_on_one_subject_the_better_fit_wins_and_the_other_is_unassigned():
    mask = _block(20, 20, 5, 15, 5, 15)
    good = _pts((6, 6), (7, 7), (8, 8), (9, 9))  # 1.0
    fair = _pts((6, 6), (7, 7), (8, 8), (1, 1))  # 0.75
    sid, inside = pose_match.match_instances(
        np.stack([fair, good]), np.array([0.99, 0.1]), {4: _runs(mask)}, {4: 0.0}
    )
    assert sid.tolist() == [-1, 4]
    assert np.isnan(inside[0])
    assert inside[1] == pytest.approx(1.0)
    assert sid.dtype == np.int32 and inside.dtype == np.float32


def test_crossed_scores_are_resolved_globally_not_greedily():
    """Greedy takes the single best pair (pose 0 → A at 1.0) and strands pose 1 (only A fits
    it): total 1.0. The global match gives pose 0 to B (0.9) and pose 1 to A (0.8): 1.7.
    """
    a = _block(10, 40, 0, 10, 0, 20)  # columns 0..19
    b = _block(10, 40, 0, 10, 10, 30)  # columns 10..29: overlaps A on 10..19
    pose0 = _pts(*[(12 + i % 6, 5) for i in range(9)], (2, 5))  # A 10/10, B 9/10
    pose1 = _pts(*[(2 + i, 5) for i in range(8)], (35, 5), (36, 5))  # A 8/10, B 0
    sid, inside = pose_match.match_instances(
        np.stack([pose0, pose1]),
        np.array([0.9, 0.9]),
        {1: _runs(a), 2: _runs(b)},
        {1: 0.0, 2: 0.0},
    )
    assert sid.tolist() == [2, 1]
    np.testing.assert_allclose(inside, [0.9, 0.8], rtol=1e-6)


def test_two_subjects_two_poses_each_to_its_own():
    a = _block(20, 40, 0, 20, 0, 15)
    b = _block(20, 40, 0, 20, 25, 40)
    on_a = _pts((3, 3), (5, 5), (8, 8))
    on_b = _pts((30, 3), (33, 5), (35, 8))
    sid, _ = pose_match.match_instances(
        np.stack([on_b, on_a]),
        np.array([0.5, 0.5]),
        {7: _runs(a), 9: _runs(b)},
        {7: 0.0, 9: 0.0},
    )
    assert sid.tolist() == [9, 7]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_tie_goes_to_the_higher_confidence(order):
    mask = _block(20, 20, 5, 15, 5, 15)
    pose = _pts((6, 6), (7, 7), (8, 8))
    conf = np.array([0.3, 0.9])[list(order)]
    sid, _ = pose_match.match_instances(
        np.stack([pose, pose]), conf, {3: _runs(mask)}, {3: 0.0}
    )
    winner = int(np.argmax(conf))
    assert sid[winner] == 3 and sid[1 - winner] == -1


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_nan_confidence_loses_a_tie(order):
    mask = _block(20, 20, 5, 15, 5, 15)
    pose = _pts((6, 6), (7, 7), (8, 8))
    conf = np.array([np.nan, 0.01])[list(order)]
    sid, _ = pose_match.match_instances(
        np.stack([pose, pose]), conf, {3: _runs(mask)}, {3: 0.0}
    )
    assert sid[int(np.flatnonzero(~np.isnan(conf))[0])] == 3
    assert sid[int(np.flatnonzero(np.isnan(conf))[0])] == -1


def test_below_min_inside_is_never_matched_even_with_no_competition():
    mask = _block(20, 20, 5, 15, 5, 15)
    one_in = _pts((6, 6), (1, 1), (2, 2), (18, 18))  # 0.25
    sid, _ = pose_match.match_instances(
        one_in[None], np.array([0.9]), {1: _runs(mask)}, {1: 0}
    )
    assert sid.tolist() == [-1]
    sid, inside = pose_match.match_instances(
        one_in[None], np.array([0.9]), {1: _runs(mask)}, {1: 0}, min_inside=0.25
    )
    assert sid.tolist() == [1] and inside[0] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def test_normalized_points_are_scaled_to_the_mask_not_the_frame():
    """A mask at detection resolution (24 x 32) against a 640 x 480 frame: a point at frame
    pixel (500, 100) is mask pixel (25, 5). Scaling by the frame instead lands off the grid.
    """
    norm = np.array([[[500 / 640, 100 / 480]]])
    px = pose_match.to_mask_px(norm, (H, W))
    np.testing.assert_allclose(px[0, 0], [25.0, 5.0])
    mask = _block(H, W, 3, 8, 22, 28)
    assert pose_match.inside_share(px, _runs(mask), 0.0)[0] == 1.0
    assert pose_match.inside_share(norm * [640, 480], _runs(mask), 0.0)[0] == 0.0


# --------------------------------------------------------------------------- #
# frame_subjects + assign_part on real zarrs
# --------------------------------------------------------------------------- #
def _write_tracked_part(
    recording_dir, rows, object_ids, with_masks=True, mask_hw=(H, W)
):
    """A detection run ``det_a`` (one part) and tracking run ``trk_a`` over it."""
    det_dir = recording_dir / "detections" / "det_a"
    det_dir.mkdir(parents=True)
    envelope = {"schema_version": 1, "status": "complete"}
    if with_masks:
        envelope["mask_resolution"] = list(mask_hw)
    (det_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")
    with DetectionPartWriter(det_dir / "part_000.zarr") as w:
        for frame, cls, bbox, mask in rows:
            w.add_row(frame, frame * 1000, 0, cls, 0.9, bbox, _rle(mask))
    trk_dir = recording_dir / "tracking" / "trk_a"
    (trk_dir / "assignments").mkdir(parents=True)
    (trk_dir / "track.yaml").write_text(
        yaml.dump({"schema_version": 1, "detection_run_id": "det_a"}, sort_keys=False),
        encoding="utf-8",
    )
    with AssignmentPartWriter(trk_dir / "assignments" / "part_000.zarr") as w:
        for object_id in object_ids:
            w.add_object_id(object_id)


def _box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return (xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H)


LEFT = _block(H, W, 4, 20, 2, 12)
RIGHT = _block(H, W, 4, 20, 18, 30)
RIGHT_TAIL = _block(H, W, 20, 23, 22, 26)  # a secondary detection of track 2
CAT = _block(H, W, 0, 3, 0, 4)


@pytest.fixture
def tracked(recording_dir):
    """Frame 0: a cat (untracked class) before both mice; frame 1: cat only; frame 2: mouse 1
    and mouse 2 as two rows; frame 3: mouse 1 only — its rows past the assignment's end.
    """
    rows = [
        (0, "cat", _box(CAT), CAT),
        (0, "mouse", _box(LEFT), LEFT),
        (0, "mouse", _box(RIGHT), RIGHT),
        (1, "cat", _box(CAT), CAT),
        (2, "mouse", _box(LEFT), LEFT),
        (2, "mouse", _box(RIGHT), RIGHT),
        (2, "mouse", _box(RIGHT_TAIL), RIGHT_TAIL),
        (3, "mouse", _box(LEFT), LEFT),
    ]
    object_ids = [-1, 1, 2, -1, 1, 2, 2]  # shorter than the detection part by one row
    _write_tracked_part(recording_dir, rows, object_ids)
    return recording_dir


def test_frame_subjects_joins_assignment_to_detection_rows_by_row_id(tracked):
    frames = list(pose_match.frame_subjects(tracked, "trk_a", "part_000", 1))
    assert [(f, t) for f, t, _ in frames] == [(0, 0), (2, 2000)]
    _, _, s0 = frames[0]
    assert sorted(s0) == [1, 2]
    np.testing.assert_array_equal(_dense(s0[1].runs), LEFT)
    np.testing.assert_array_equal(_dense(s0[2].runs), RIGHT)
    _, _, s2 = frames[1]
    assert len(s2[2].runs) == 2  # one RunMask per detection row of the subject
    np.testing.assert_array_equal(_dense(s2[2].runs), RIGHT | RIGHT_TAIL)
    assert s2[2].bbox == pytest.approx((18 / W, 4 / H, 30 / W, 23 / H))
    bw, bh = 10 / W * 640, 16 / H * 480
    assert s0[1].scale_px == pytest.approx(np.sqrt(bw * bh))
    assert pose_match.subject_mask_hw(tracked, "trk_a") == (H, W)


def test_frame_subjects_keeps_the_cadence(tracked):
    frames = list(pose_match.frame_subjects(tracked, "trk_a", "part_000", 2))
    assert [f for f, _, _ in frames] == [0, 2]
    assert [
        f for f, _, _ in pose_match.frame_subjects(tracked, "trk_a", "part_000", 3)
    ] == [0]


def test_frame_subjects_without_masks_stands_each_box_in(recording_dir):
    rows = [(0, "mouse", _box(LEFT), LEFT)]
    _write_tracked_part(recording_dir, rows, [5], with_masks=False)
    assert pose_match.subject_mask_hw(recording_dir, "trk_a") == (480, 640)
    [(_, _, subjects)] = pose_match.frame_subjects(
        recording_dir, "trk_a", "part_000", 1
    )
    [run] = subjects[5].runs
    assert run.size == (480, 640)
    mask = run.to_array()
    assert mask[80:400, 40:240].all()
    assert (
        320 * 200 <= mask.sum() <= 322 * 202
    )  # the box, give or take float32 rounding


def test_frame_subjects_refuses_an_unknown_tracking_run(recording_dir):
    with pytest.raises(ValueError, match="no tracking run 'nope'"):
        list(pose_match.frame_subjects(recording_dir, "nope", "part_000", 1))


def _norm(*mask_xy) -> np.ndarray:
    """One instance's normalized full-frame points from mask-pixel centres."""
    return (np.asarray(mask_xy, dtype=np.float64) + 0.5) / [W, H]


def _end_to_end_instances() -> dict:
    return {
        "frame_idx": np.array([0, 0, 0, 1, 2, 3], dtype=np.int32),
        "points": np.stack(
            [
                _norm((20, 8), (24, 12), (28, 16)),  # frame 0 → mouse 2
                _norm((4, 8), (8, 12), (10, 16)),  # frame 0 → mouse 1
                _norm((15, 0), (15, 1), (15, 2)),  # frame 0, between them: nobody
                _norm((1, 1), (2, 1), (3, 1)),  # frame 1: on the cat, no subject
                _norm((24, 21), (23, 22), (25, 21)),  # frame 2: on mouse 2's tail row
                _norm((4, 8), (8, 12), (10, 16)),  # frame 3: past the assignment's end
            ]
        ).astype(np.float32),
        "confidence": np.array([0.8, 0.7, 0.9, 0.9, np.nan, 0.9], dtype=np.float32),
    }


def test_assign_part_end_to_end(tracked):
    instances = _end_to_end_instances()
    subjects = pose_match.frame_subjects(tracked, "trk_a", "part_000", 1)
    sid, inside, per_subject = pose_match.assign_part(
        instances, subjects, (H, W), eligible=None, min_inside=0.5, mask_grow=0.05
    )
    assert sid.tolist() == [2, 1, -1, -1, 2, -1]
    np.testing.assert_allclose(inside[[0, 1, 4]], 1.0)
    assert np.isnan(inside[[2, 3, 5]]).all()
    assert sorted(per_subject) == [1, 2]
    assert [r[:3] for r in per_subject[1]] == [(0, 0, 1), (2, 2000, -1)]
    assert [r[:3] for r in per_subject[2]] == [(0, 0, 0), (2, 2000, 4)]
    assert all(r[3] > 0 for rows in per_subject.values() for r in rows)


def _long_two_subject_part(recording_dir, n_frames=400, mh=240, mw=320):
    """A part of ``n_frames`` frames with two subjects, every frame posed inside subject 1;
    returns the detect instances, the mask size and the bytes every mask together takes.
    """
    # Striped, not solid: a solid block is ~200 runs (~7 KB parsed), so a cache of every
    # frame's parsed runs would sit under the bound and the guard would pass against the
    # class it exists for. Every 8th row makes each mask ~4,000 runs (~130 KB parsed).
    left = np.zeros((mh, mw), dtype=bool)
    left[40:200:8, 20:120] = True
    right = np.zeros((mh, mw), dtype=bool)
    right[40:200:8, 180:300] = True
    rows = []
    for frame in range(n_frames):
        rows += [(frame, "mouse", (0.06, 0.17, 0.38, 0.83), left)]
        rows += [(frame, "mouse", (0.56, 0.17, 0.94, 0.83), right)]
    _write_tracked_part(recording_dir, rows, [1, 2] * n_frames, mask_hw=(mh, mw))
    instances = {
        "frame_idx": np.arange(n_frames, dtype=np.int32),
        "points": np.tile(
            ((np.array([[60, 104], [70, 120], [80, 136]]) + 0.5) / [mw, mh])[
                None
            ],  # stripe rows
            (n_frames, 1, 1),
        ).astype(np.float32),
        "confidence": np.full(n_frames, 0.9, dtype=np.float32),
    }
    every_mask = 2 * n_frames * mh * mw  # bool bytes, one mask per subject-frame
    return instances, (mh, mw), every_mask


def _traced_peak(subjects, instances, mask_hw):
    """``assign_part`` over fresh ``subjects()``: its result and its peak traced allocation.

    One untraced pass first takes the one-off allocations (imports, codec set-up, ~20 MB) out
    of the measured one; without it the fixed code peaks near a quarter-of-every-mask bound.
    Each pass gets its own ``subjects()``, made before tracing starts, so a mask the first pass
    decoded and kept cannot make the measured pass look cheap.
    """
    pose_match.assign_part(instances, subjects(), mask_hw, eligible=None)
    frames = subjects()
    tracemalloc.start()
    try:
        result = pose_match.assign_part(instances, frames, mask_hw, eligible=None)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def test_assignment_holds_one_frames_masks_not_the_parts(recording_dir):
    """Assigning a part keeps about one frame's masks in memory, not every frame's.

    ``frame_subjects`` held each frame's ``SubjectOnFrame`` for the whole part, and a keypoint
    track worker grew ~450 MB/s to the 32 GB the box had on a 30-minute two-animal recording.
    The class, and the guard for a caller that holds every frame, are
    :func:`test_a_held_subject_keeps_no_decoded_mask`.
    """
    instances, mask_hw, every_mask = _long_two_subject_part(recording_dir)

    (sid, _inside, _per_subject), peak = _traced_peak(
        lambda: pose_match.frame_subjects(recording_dir, "trk_a", "part_000", 1),
        instances,
        mask_hw,
    )

    assert (sid == 1).all()  # every frame's masks were read
    assert peak < every_mask / 4, (
        f"assignment peaked at {peak / 1e6:.1f} MB; holding every mask is "
        f"{every_mask / 1e6:.1f} MB"
    )


def test_a_held_subject_keeps_no_decoded_mask(recording_dir):
    """A caller holding every frame of a part still holds about one frame's masks.

    The class: an item that caches the value it decoded, so "decoded only when read" becomes
    "every value read, resident for as long as anyone holds the item". ``SubjectOnFrame`` kept
    its decoded mask: ``frame_subjects`` holding every frame took a keypoint track worker to
    32 GB, and the keypoint dataset sampler, which holds every tracked frame for a whole job,
    kept the mask of every frame it taught (``n_frames`` × subjects, with no upper limit on
    ``n_frames``). A subject now parses its runs on each read and keeps nothing (the fixture is
    striped so a cache of parsed runs breaks the bound too). Swept with it: every
    other mask decode outside the GUI decodes, uses and drops one frame's masks, and the GUI's
    caches hold metadata, not masks.
    """
    instances, mask_hw, every_mask = _long_two_subject_part(recording_dir)
    (sid, _inside, _per_subject), peak = _traced_peak(
        lambda: list(pose_match.frame_subjects(recording_dir, "trk_a", "part_000", 1)),
        instances,
        mask_hw,
    )

    assert (sid == 1).all()  # every held frame's masks were read
    assert peak < every_mask / 4, (
        f"assigning held frames peaked at {peak / 1e6:.1f} MB; holding every mask is "
        f"{every_mask / 1e6:.1f} MB"
    )


def test_assign_part_with_no_instances_lists_every_subject_frame_unassigned(tracked):
    instances = {
        "frame_idx": np.zeros(0, dtype=np.int32),
        "points": np.zeros((0, 3, 2), dtype=np.float32),
        "confidence": np.zeros(0, dtype=np.float32),
    }
    sid, inside, per_subject = pose_match.assign_part(
        instances,
        pose_match.frame_subjects(tracked, "trk_a", "part_000", 1),
        (H, W),
        eligible=None,
    )
    assert sid.shape == (0,) and inside.shape == (0,)
    assert [r[2] for r in per_subject[1]] == [-1, -1]


def test_a_class_filtered_tracking_run_gives_each_subject_its_own_rows(
    processed_recording,
):
    """**The class: pairing by position two arrays a filter made not parallel (§13).** A
    tracker given ``class_name`` assigns only that class's detections, so its ``object_id``
    column is shorter than the detection part and row ``i`` of one is not row ``i`` of the
    other. Paired by position, every subject after the first dropped row lands on the wrong
    frame, box and mask, and keypoint track gives it the wrong animal's pose — a plausible,
    well-formed run. It hid because every fixture tracked one class, where the two are
    parallel; ``zarr_io.object_ids_on_detection_rows`` is the one pairing for a reader of the
    whole part. (Moved here from the detect stage when §26's 2026-10-21 revision moved the
    reading of tracking into the assignment.)

    Frames 0..3 hold a cat, a rat, a cat, a rat; the run tracked only rats, so the subject is
    on frames 1 and 3, on the rat's box.
    """
    it = processed_recording
    sq = it.square
    rat_box = (0.5, 0.5, 0.9, 0.9)
    cat_box = (0.1, 0.1, 0.5, 0.5)
    recording = it.add_recording(
        "recording_002",
        [
            (0, 0, cat_box, sq(4, 20, 4, 20)),
            (1, 0, rat_box, sq(20, 36, 20, 36)),
            (2, 0, cat_box, sq(4, 20, 4, 20)),
            (3, 0, rat_box, sq(20, 36, 20, 36)),
        ],
        [0, 0],  # one row per rat: the tracker never saw the cats
        classes=["cat", "rat", "cat", "rat"],
        class_name="rat",
    )
    frames = list(pose_match.frame_subjects(recording.dir, "track_01", "part_001", 1))
    assert [f for f, _t, _s in frames] == [1, 3]
    for _f, _t, subjects in frames:
        assert subjects[0].bbox == pytest.approx(rat_box)


def test_a_pose_placed_on_a_box_of_another_class_is_never_given_to_a_subject(
    processed_recording,
):
    """**The class: a pose matched by mask alone, whatever box it was placed on.** A
    per-detection keypoint method places a pose in every box the detection run holds; a
    detector that also finds the arena (``plate``) gets a pose placed in the arena's box,
    the whole arena cropped and the animal small inside it. That pose lies on the animal as
    squarely as the animal's own, so the mask match took it whenever it scored higher: 28% of
    the tracked poses on the Hotplate experiment (13,750 of 48,872), the paws lost, while the
    animal's own pose went unused. The tracking run's ``class_name`` decides which boxes'
    poses its subjects may take.

    One frame: a ``plate`` box over the whole field and the rat's box; the run tracked rats.
    Two poses lie on the rat, the plate's more confident. The rat takes its own.
    """
    it = processed_recording
    sq = it.square
    recording = it.add_recording(
        "recording_002",
        [
            (0, 0, (0.0, 0.0, 1.0, 1.0), sq(0, 40, 0, 40)),  # the plate
            (0, 1, (0.5, 0.5, 0.9, 0.9), sq(20, 36, 20, 36)),  # the rat
        ],
        [0],  # the tracker saw the rat's row only
        classes=["plate", "rat"],
        class_name="rat",
    )
    pose = [[0.7, 0.7], [0.75, 0.65], [0.65, 0.75]]
    instances = {
        "frame_idx": np.array([0, 0], dtype=np.int32),
        "points": np.array([pose, pose], dtype=np.float32),
        "confidence": np.array([0.99, 0.5], dtype=np.float32),
        "detection_row": np.array([0, 1], dtype=np.int32),  # on the plate, on the rat
    }
    tracked = pose_match.tracked_class_rows(recording.dir, "track_01", "part_001")
    assert tracked.tolist() == [False, True]
    eligible = pose_match.eligible_instances(instances["detection_row"], tracked)

    def assign(offered):
        return pose_match.assign_part(
            instances,
            pose_match.frame_subjects(recording.dir, "track_01", "part_001", 1),
            pose_match.subject_mask_hw(recording.dir, "track_01"),
            eligible=offered,
        )[0].tolist()

    assert assign(eligible) == [-1, 0], "the rat takes the pose on its own box"
    assert assign(None) == [0, -1], "offered every pose, the plate's wins on confidence"


def test_which_poses_a_tracking_run_offers():
    rows = np.array([-1, 0, 1, 2, 5], dtype=np.int32)
    tracked = np.array([False, True, False])
    # no box (a whole-frame method): offered; a row past the part: not the tracked class
    assert pose_match.eligible_instances(rows, tracked).tolist() == [
        True,
        False,
        True,
        False,
        False,
    ]
    assert pose_match.eligible_instances(rows, None).all(), "every class tracked"


def test_no_stage_offers_every_pose_whatever_its_box():
    """Every caller of ``assign_part`` in ``detail/`` decides eligibility from the tracking
    run's class (``eligible_instances``); ``eligible=None`` is for a test's direct call. A
    stage passing it would give subjects the poses placed on other classes' boxes again.
    """
    import re
    from pathlib import Path

    import detail

    root = Path(detail.__file__).parent
    pattern = re.compile(r"eligible\s*=\s*None\b")
    offenders = [
        f"{path.relative_to(root).as_posix()}:{n}"
        for path in sorted(root.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if pattern.search(line) and "def " not in line
    ]
    assert not offenders, offenders
    assert pattern.search("eligible=None,")  # the check can fire


# --------------------------------------------------------------------------- #
# the run-level share against the rasterised one
# --------------------------------------------------------------------------- #
def _fuzz_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    kind = int(rng.integers(0, 6))
    mask = np.zeros((h, w), dtype=bool)
    if kind == 1:
        mask[:] = True
    elif kind == 2:
        mask[int(rng.integers(0, h)), int(rng.integers(0, w))] = True
    elif kind == 3:
        mask[0, :] = True
        mask[:, -1] = True
    elif kind == 4:
        yy, xx = np.ogrid[:h, :w]
        cy, cx = rng.uniform(-3, h + 3), rng.uniform(-3, w + 3)
        ay, ax = rng.uniform(1, max(2, h / 2)), rng.uniform(1, max(2, w / 2))
        mask = ((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2 <= 1.0
    elif kind == 5:
        mask = rng.random((h, w)) < 0.05
    return mask


_RADII = (0.0, 0.49, 0.5, 0.51, 1.0, 1.5, 2.5, 2.7, 10.6, 40.0)
_SPECIAL_POINTS = (
    (-0.5, 0.5),
    (0.5, -0.5),
    (np.inf, 1.0),
    (-np.inf, 1.0),
    (1e9, 1e9),
    (-1e9, 2.0),
)


def _fuzz_cases(seed: int = 0, n: int = 300):
    """``(points_xy, rows, radius_px)``: grids 1..40 and a few 600x800, 1-3 rows per subject,
    points at pixel centres, integer boundaries, off the grid, NaN, infinite and absurd.
    """
    rng = np.random.default_rng(seed)
    for i in range(n):
        h, w = (
            (600, 800)
            if i % 60 == 0
            else (int(rng.integers(1, 41)), int(rng.integers(1, 41)))
        )
        rows = [_fuzz_mask(rng, h, w) for _ in range(int(rng.integers(1, 4)))]
        n_inst, k = int(rng.integers(0, 5)), int(rng.integers(1, 6))
        pts = rng.random((n_inst, k, 2)) * [w, h] * 1.3 - np.array([w, h]) * 0.15
        kinds = rng.random((n_inst, k))
        pts[kinds < 0.15] = np.nan
        snap = kinds > 0.8
        pts[snap] = np.floor(pts[snap])
        for j in range(min(k, n_inst)):
            if rng.random() < 0.3:
                pts[j, 0] = _SPECIAL_POINTS[int(rng.integers(len(_SPECIAL_POINTS)))]
            if rng.random() < 0.3:
                pts[j, -1] = (w + 0.5, float(rng.integers(0, h)) + 0.5)
        yield pts, rows, float(rng.choice(_RADII))


def _disagreements(cases) -> list:
    """Cases where the run-level share and the rasterised reference differ in any bit."""
    bad = []
    for pts, rows, radius in cases:
        want = _reference_inside_share(pts, np.logical_or.reduce(rows), radius)
        got = pose_match.inside_share(pts, [_run(m) for m in rows], radius)
        if got.dtype != want.dtype or got.shape != want.shape:
            bad.append((pts.shape, rows[0].shape, radius, want, got))
        elif not np.array_equal(got, want):
            bad.append((pts.shape, rows[0].shape, radius, want, got))
    return bad


def test_inside_share_matches_the_dilated_mask_reference():
    """Bit-equal to decode-crop-dilate-index over 300 seeded cases, float32 and all."""
    assert _disagreements(list(_fuzz_cases())) == []


def test_the_reference_fuzz_would_catch_a_shifted_footprint(monkeypatch):
    real = pose_match._footprint_spans

    def shifted(r: int):
        dx, lo, hi = real(r)
        return dx + 1, lo, hi

    monkeypatch.setattr(pose_match, "_footprint_spans", shifted)
    assert len(_disagreements(list(_fuzz_cases(seed=1, n=80)))) > 0


def test_assignment_rasterises_no_mask(tracked, monkeypatch):
    """The count the speed claim rests on: assigning a part decodes no mask and dilates
    nothing, and gives the same answer as the end-to-end case."""

    def rasterised(*_args, **_kwargs):
        raise AssertionError("assignment rasterised a mask")

    monkeypatch.setattr("detail.pipeline.subject_frames.decode_mask", rasterised)
    monkeypatch.setattr("detail.core.utils.decode_mask", rasterised)
    monkeypatch.setattr(pose_match.cv2, "dilate", rasterised)
    sid, _inside, _per_subject = pose_match.assign_part(
        _end_to_end_instances(),
        pose_match.frame_subjects(tracked, "trk_a", "part_000", 1),
        (H, W),
        eligible=None,
        min_inside=0.5,
        mask_grow=0.05,
    )
    assert sid.tolist() == [2, 1, -1, -1, 2, -1]
