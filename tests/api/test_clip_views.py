"""The clip popup's three pictures (``ClipQueryPreview.clip_views``): the footage with optional
marks, every tracked animal's mask on white, and the keypoints the embedding run used on white.

**What is pinned.** The masks pane shows *every* tracked animal, the clip's subject in the mask
tint and the others grey, so a second animal reads as one with the footage gone (the popup is
for telling behaviours apart, and contact is half of that). The pose is **the keypoint run the
embedding read** — ``keypoint_run_id`` on its ``embedding.yaml`` — and an embedding with none
says so rather than drawing nothing or reaching for some other run. The footage's marks are
toggles: off, the frame is the decoded frame, byte for byte.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.api.run_preview import ClipQueryPreview
from detail.core.skeleton import Skeleton
from detail.core.visualize import MASK_TINT
from detail.storage import keypoint_runs
from detail.storage import recording as recording_store
from detail.storage.keypoint_zarr import KeypointTrackPartWriter
from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter
from tests.api.test_run_preview import (
    _add_embedding,
    _recording_with_detection,
    _write_track_yaml,
)

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


def _rle(region) -> str:
    mask = np.zeros((H, W), np.uint8)
    mask[region] = 1
    return mask_utils.encode(np.asfortranarray(mask))["counts"].decode("ascii")


def _two_animals(tmp_path):
    """Frame 0 holds two tracked animals with masks (tracks 0 and 1); frame 2 holds track 0."""
    recording = _recording_with_detection(tmp_path)
    ddir = recording.dir / "detections" / "det_01"
    shutil.rmtree(ddir / "part_001.zarr")
    with DetectionPartWriter(ddir / "part_001.zarr") as writer:
        writer.add_row(
            0, 0, 0, "mouse", 0.9, (10 / W, 10 / H, 20 / W, 20 / H), _rle(SUBJECT)
        )
        writer.add_row(
            0, 0, 1, "mouse", 0.8, (40 / W, 30 / H, 50 / W, 40 / H), _rle(OTHER)
        )
        writer.add_row(2, int(2e9 / 30), 0, "mouse", 0.7, (0.5, 0.5, 0.6, 0.6), "")
    adir = recording.dir / "tracking" / "track_01" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        for object_id in (0, 1, 0):
            writer.add_object_id(object_id)
    _write_track_yaml(
        recording.dir / "tracking" / "track_01", detection_run_id="det_01"
    )
    data = recording_store.read_recording(recording.dir)
    data.setdefault("tracking_runs", []).append(
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    )
    recording_store.write_recording(recording.dir, data)
    return _add_embedding(recording)


def _with_keypoints(recording):
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
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["keypoint_run_id"] = "kt_01"
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    return recording


def _views(recording, *, track_id=0, frame=0, masks=False, points=False):
    preview = ClipQueryPreview(recording.dir.parent.parent)
    return preview.clip_views(
        recording.id,
        "emb_01",
        track_id,
        frame,
        masks_on_frame=masks,
        points_on_frame=points,
    )


def test_the_masks_pane_shows_every_animal_on_white_the_subject_tinted(tmp_path):
    recording = _two_animals(tmp_path)
    views = _views(recording)
    masks = views.masks
    assert masks.shape == (H, W, 3) and masks.dtype == np.uint8
    assert (masks[0, 0] == 255).all(), "the background is white"
    # sampled off the line drawn between the two animals' centres
    assert tuple(masks[11, 18]) == tuple(MASK_TINT), "the clip's animal, tinted"
    other = masks[38, 48]
    assert other[0] == other[1] == other[2] < 255, "the other animal, grey"
    # the subject is whichever track the clip names
    assert tuple(_views(recording, track_id=1).masks[38, 48]) == tuple(MASK_TINT)


def test_the_footage_is_untouched_unless_a_mark_is_asked_for(tmp_path):
    recording = _with_keypoints(_two_animals(tmp_path))
    plain = _views(recording)
    decoded = ClipQueryPreview(recording.dir.parent.parent).frame_plain(
        recording.id, "emb_01", 0, 0
    )[0]
    np.testing.assert_array_equal(plain.frame, decoded)
    masked = _views(recording, masks=True)
    assert (masked.frame[15, 15] != decoded[15, 15]).any()
    assert (masked.frame[35, 45] == decoded[35, 45]).all(), "only the subject's mask"
    pointed = _views(recording, points=True)
    assert (pointed.frame != decoded).any()
    assert (
        pointed.frame[30:41, 40:51] == decoded[30:41, 40:51]
    ).all(), "only the subject's points go on the footage"


def test_the_pose_pane_is_the_embeddings_keypoint_run_on_white(tmp_path):
    recording = _with_keypoints(_two_animals(tmp_path))
    views = _views(recording)
    assert views.pose_note is None
    pose = views.pose
    assert pose.shape == (H, W, 3) and (pose[0, 0] == 255).all()
    assert (pose[10:21, 10:21] != 255).any(), "the subject's points"
    assert (pose[30:41, 40:51] != 255).any(), "the other animal's points, too"
    # a frame the run placed nothing on is a blank pose, not an error
    assert (_views(recording, frame=2).pose == 255).all()


def _walking(tmp_path, *, keypoints=True, heading=False):
    """Five frames of two animals walking right, one row apart: track 0's 10x10 mask at
    ``x0 = 4 + 6i`` (rows 4..14) and track 1's at the same x (rows 30..40), every frame.

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
                    _rle((slice(y0, y0 + 10), slice(x0, x0 + 10))),
                )
    adir = recording.dir / "tracking" / "track_01" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        for _ in range(5):
            writer.add_object_id(0)
            writer.add_object_id(1)
    _write_track_yaml(
        recording.dir / "tracking" / "track_01", detection_run_id="det_01"
    )
    data = recording_store.read_recording(recording.dir)
    data.setdefault("tracking_runs", []).append(
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    )
    recording_store.write_recording(recording.dir, data)
    recording = _add_embedding(recording)
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
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["keypoint_run_id"] = "kt_01"
    if heading:
        meta["align"] = "rotate"
        meta["heading_definition"] = {"front": ["nose"], "back": ["tail"]}
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    return recording


def _walked(recording, **kwargs):
    """The last frame (4) of a ``_walking`` recording, nothing on the footage but ``kwargs``."""
    preview = ClipQueryPreview(recording.dir.parent.parent)
    return preview.clip_views(
        recording.id,
        "emb_01",
        0,
        4,
        masks_on_frame=False,
        points_on_frame=False,
        **kwargs,
    )


def _changed(after, before, row, xs=slice(12, 30)):
    """Whether row ``row`` (± 1) over ``xs`` differs — where a point's path lies."""
    return (after[row - 1 : row + 2, xs] != before[row - 1 : row + 2, xs]).any()


def _ink(picture, before, row, xs=slice(12, 30)):
    """The mean colour of the pixels a trail changed along ``row``."""
    band = picture[row - 1 : row + 2, xs].reshape(-1, 3).astype(float)
    moved = (picture[row - 1 : row + 2, xs] != before[row - 1 : row + 2, xs]).any(
        axis=2
    )
    return band[moved.reshape(-1)].mean(axis=0)


def test_each_keypoint_trails_in_its_own_colour_on_the_footage_and_the_pose(tmp_path):
    """Trail is per keypoint now, not one line through the box centre: each point's path, in
    the role colour ``draw_keypoints`` gives it, on the footage and on the Keypoints pane.
    """
    recording = _walking(tmp_path)
    plain, traced = _walked(recording), _walked(recording, trail=True)
    assert "keypoint" in traced.trail_note
    for picture, before in ((traced.frame, plain.frame), (traced.pose, plain.pose)):
        for row in (5, 9, 13):  # left paw, nose and tail, right paw
            assert _changed(picture, before, row), f"no trail along row {row}"
        left, right = _ink(picture, before, 5), _ink(picture, before, 13)
        assert left[2] > left[0], "the left paw's trail is the left colour (blue)"
        assert (
            right[0] > right[2]
        ), "the right paw's trail is the right colour (vermillion)"
        midline = _ink(picture, before, 9)
        assert midline[1] > midline[0], "a midline point's trail is green"
    np.testing.assert_array_equal(traced.masks, plain.masks)
    assert [*traced.skeleton.names] == [*WALKER.names], "for the points picker"


def test_only_the_chosen_points_trail(tmp_path):
    recording = _walking(tmp_path)
    plain = _walked(recording)
    paw = _walked(recording, trail=True, trail_points=["left_paw"])
    for picture, before in ((paw.frame, plain.frame), (paw.pose, plain.pose)):
        assert _changed(picture, before, 5), "the chosen point trails"
        assert not _changed(picture, before, 13), "an unchosen point does not"
        assert not _changed(picture, before, 9)


def test_body_aligned_trails_are_paths_relative_to_the_body(tmp_path):
    """Aligned, each past frame is turned by its own axis: an animal walking without moving
    any point relative to its body leaves no trail on the pose — while the footage, which
    shows it crossing the arena, does."""
    recording = _walking(tmp_path, heading=True)
    still = _walked(recording, pose_aligned=True)
    walked = _walked(recording, pose_aligned=True, trail=True)
    assert walked.align_note is None
    np.testing.assert_array_equal(walked.pose, still.pose)
    assert _changed(walked.frame, still.frame, 13)


def test_without_keypoints_the_trail_is_the_box_centre_on_the_footage(tmp_path):
    recording = _walking(tmp_path, keypoints=False)
    plain, traced = _walked(recording), _walked(recording, trail=True)
    assert traced.pose is None and traced.skeleton is None
    assert "box" in traced.trail_note
    assert _changed(traced.frame, plain.frame, 9), "the centre walked along row 9"
    assert not _changed(traced.frame, plain.frame, 40), "one line, not one per point"
    ink = _ink(traced.frame, plain.frame, 9)
    assert ink.min() > plain.frame[9, 20].max(), "white over the footage"
    np.testing.assert_array_equal(traced.masks, plain.masks)


def _whiteness_lost(masks, y, x):
    return 255 * 3 - int(masks[y, x].astype(int).sum())


def test_the_mask_trail_is_the_subjects_past_masks_fading_with_age(tmp_path):
    """The masks pane's onion skin: the subject's masks from the frames before, under the
    current one, each fainter than the one after it — and nobody else's."""
    recording = _walking(tmp_path, keypoints=False)
    plain, skin = _walked(recording), _walked(recording, mask_trail=True)
    # Track 0 is at x 28..38 now; x 22..27 was only frame 3's, 16..21 frame 2's (newer than
    # frame 1's there), 10..15 frame 1's and 4..9 frame 0's.
    ages = [_whiteness_lost(skin.masks, 9, x) for x in (25, 19, 13, 6)]
    assert all(_whiteness_lost(plain.masks, 9, x) == 0 for x in (25, 19, 13, 6))
    assert ages[0] > ages[1] > ages[2] > ages[3] > 0, f"not fading with age: {ages}"
    tint = skin.masks[9, 25].astype(int)
    assert tint[0] > tint[2], "in the subject's tint"
    # Track 1 walked the same way on rows 30..40: its past is not drawn.
    np.testing.assert_array_equal(skin.masks[30:41, 0:27], plain.masks[30:41, 0:27])
    np.testing.assert_array_equal(skin.frame, plain.frame)  # the footage is untouched


def test_trails_decode_no_extra_video_frame(tmp_path, monkeypatch):
    """The past masks come from the detection rows and the past points from the keypoint
    run: one clip_views call decodes exactly the one frame it shows."""
    from detail.storage.video import FrameCursor

    recording = _walking(tmp_path)
    decoded = []
    original = FrameCursor.frame

    def counting(self, frame_idx):
        decoded.append(frame_idx)
        return original(self, frame_idx)

    monkeypatch.setattr(FrameCursor, "frame", counting)
    _walked(recording, trail=True, mask_trail=True)
    assert decoded == [4]


def test_the_mask_trail_cache_is_bounded(tmp_path):
    """Kept between frames for playback, but never more than its capacity."""
    from detail.api.run_preview import _MaskHistory

    recording = _walking(tmp_path, keypoints=False)
    preview = ClipQueryPreview(recording.dir.parent.parent)
    _walked(recording, mask_trail=True)  # builds the scrub context
    history = _MaskHistory()
    history.capacity = 2
    scrub, _t = preview._scrub_ctx(recording.id, "emb_01")
    layers = history.layers(scrub, 0, 0, [0, 1, 2, 3], (H, W))
    assert all(layer is not None for layer in layers)
    assert len(history._kept) == 2


def test_an_embedding_that_read_no_keypoints_says_so(tmp_path):
    recording = _two_animals(tmp_path)
    views = _views(recording, points=True)
    assert views.pose is None
    assert "used no keypoints" in views.pose_note
    preview = ClipQueryPreview(recording.dir.parent.parent)
    assert "used no keypoints" in preview.clip_has_points(recording.id, "emb_01")


@pytest.mark.parametrize("missing", ["kt_09"])
def test_a_keypoint_run_that_cannot_be_read_says_which(tmp_path, missing):
    recording = _with_keypoints(_two_animals(tmp_path))
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["keypoint_run_id"] = missing
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    views = _views(recording)
    assert views.pose is None and missing in views.pose_note


def test_the_masks_pane_draws_the_arena_the_region_and_the_nearest_animal(tmp_path):
    """§10's shapes on the masks pane: the field of view and each region outlined, the line
    from the clip's animal to the nearest other one, and a region it is in named."""
    from detail.api.recording import Recording

    recording = _two_animals(tmp_path)
    handle = Recording(recording.dir.parent.parent, recording.id)
    plain = _views(recording).masks
    corner = [(0.0, 0.0), (0.45, 0.0), (0.45, 0.6), (0.0, 0.6)]
    handle.set_rois([{"name": "left corner", "keyframes": [(0, corner)]}])
    drawn = _views(recording).masks
    left_edge = drawn[0:8, 28]  # the region's right edge, x = 0.45 * 64, above the line
    assert (left_edge != 255).any() and (plain[0:8, 28] == 255).all()
    assert (
        drawn[2:10, 0:30] != plain[2:10, 0:30]
    ).any(), "the region it is in is named"
    # the line between the two centres, (15, 15) to (45, 35)
    assert (plain[25, 30] != 255).any() or (drawn[25, 30] != 255).any()


def _with_heading(recording):
    """The embedding was aligned by nose (front) and tail (back)."""
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["align"] = "rotate"
    meta["heading_definition"] = {"front": ["nose"], "back": ["tail"]}
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    return recording


def _aligned(recording):
    preview = ClipQueryPreview(recording.dir.parent.parent)
    return preview.clip_views(
        recording.id,
        "emb_01",
        0,
        0,
        masks_on_frame=False,
        points_on_frame=False,
        pose_aligned=True,
    )


def test_the_aligned_pose_is_the_subject_alone_facing_up_at_the_centre(tmp_path):
    recording = _with_heading(_with_keypoints(_two_animals(tmp_path)))
    views = _aligned(recording)
    assert views.align_note is None
    inked = np.argwhere((views.pose < 250).any(axis=2))  # (row, col)
    # The subject faced down-right; aligned, its 0.4 x H body runs straight up the middle
    # column, nose above tail — and the other animal (at 42..48, 32..38) is not drawn.
    dot = 5  # a keypoint's dot reaches this far past its point
    assert inked[:, 1].mean() == pytest.approx(W / 2, abs=1)
    assert np.abs(inked[:, 1] - W / 2).max() <= dot
    nose_y, tail_y = H / 2 - 0.2 * H, H / 2 + 0.2 * H
    assert inked[:, 0].min() == pytest.approx(nose_y, abs=dot)
    assert inked[:, 0].max() == pytest.approx(tail_y, abs=dot)
    unaligned = _views(recording).pose
    assert (unaligned[38, 48] < 250).any(), "unaligned, the other animal is drawn"


def test_without_a_heading_definition_the_pose_is_not_aligned_and_says_why(tmp_path):
    recording = _with_keypoints(_two_animals(tmp_path))
    views = _aligned(recording)
    assert "No heading definition" in views.align_note
    assert np.array_equal(views.pose, _views(recording).pose), "drawn as usual"


def test_a_frame_without_the_subjects_axis_is_blank_but_for_a_line(tmp_path):
    recording = _with_heading(_with_keypoints(_two_animals(tmp_path)))
    preview = ClipQueryPreview(recording.dir.parent.parent)
    views = preview.clip_views(
        recording.id,
        "emb_01",
        0,
        2,  # track 0 is tracked on frame 2 but has no keypoints there
        masks_on_frame=False,
        points_on_frame=False,
        pose_aligned=True,
    )
    inked = (views.pose < 250).any(axis=2)
    assert inked.any() and not inked[: H // 4].any(), "a line of text mid-pane, no pose"


# --- without the footage ----------------------------------------------------------------- #
def count_decodes(monkeypatch) -> list:
    """Every call into the video decode, as ``(seam, thread id)``: the cursor every preview
    surface reads frames through (``FrameCursor.frame``), and any container opened for reading
    (``av.open``), so a decode reached by another route is counted too. An H.264 export opens
    its output with ``av.open(path, "w")``, which writes and decodes nothing."""
    import threading

    import av

    from detail.storage.video import FrameCursor

    calls: list = []
    cursor_frame, av_open = FrameCursor.frame, av.open

    def frame(self, frame_idx):
        calls.append(("FrameCursor.frame", threading.get_ident()))
        return cursor_frame(self, frame_idx)

    def opened(*args, **kwargs):
        if (args[1] if len(args) > 1 else kwargs.get("mode", "r")) != "w":
            calls.append(("av.open", threading.get_ident()))
        return av_open(*args, **kwargs)

    monkeypatch.setattr(FrameCursor, "frame", frame)
    monkeypatch.setattr(av, "open", opened)
    return calls


def _both_ways(recording, monkeypatch, frame, **kwargs):
    """``(decoded, blank, decodes)``: the views drawn from the footage and without it, each
    by a fresh preview, and the decodes the second one made."""
    root = recording.dir.parent.parent
    decoded = ClipQueryPreview(root).clip_views(
        recording.id, "emb_01", 0, frame, masks_on_frame=False,
        points_on_frame=False, **kwargs,
    )  # fmt: skip
    calls = count_decodes(monkeypatch)
    blank = ClipQueryPreview(root).clip_views(
        recording.id, "emb_01", 0, frame, masks_on_frame=True,
        points_on_frame=True, footage=False, **kwargs,
    )  # fmt: skip
    monkeypatch.undo()
    return decoded, blank, list(calls)


@pytest.mark.parametrize(
    "build, kwargs",
    [
        ("two_animals", {}),
        ("walking", {}),
        ("walking", {"trail": True, "mask_trail": True}),
        ("walking", {"trail": True, "trail_points": ["nose"], "mask_trail": True}),
        ("heading", {"trail": True, "pose_aligned": True}),
        ("no_keypoints", {"trail": True, "mask_trail": True}),
    ],
)
def test_without_the_footage_the_panes_are_the_decoded_ones_and_nothing_is_decoded(
    tmp_path, monkeypatch, build, kwargs
):
    """``footage=False`` is for a caller that shows only the masks and the keypoints: it
    must draw them pixel for pixel as the decode does, at the same frame and time, and never
    open the video. The frame is white, and nothing is drawn on it even when asked."""
    recording = {
        "two_animals": lambda: _with_keypoints(_two_animals(tmp_path)),
        "walking": lambda: _walking(tmp_path),
        "heading": lambda: _walking(tmp_path, heading=True),
        "no_keypoints": lambda: _walking(tmp_path, keypoints=False),
    }[build]()
    frames = (0, 2) if build == "two_animals" else range(5)
    for frame in frames:
        decoded, blank, decodes = _both_ways(recording, monkeypatch, frame, **kwargs)
        assert decodes == [], f"frame {frame} decoded video: {decodes}"
        np.testing.assert_array_equal(blank.masks, decoded.masks)
        if decoded.pose is None:
            assert blank.pose is None
        else:
            np.testing.assert_array_equal(blank.pose, decoded.pose)
        assert (blank.frame == 255).all() and blank.frame.shape == decoded.frame.shape
        assert (blank.actual, blank.t_ns) == (decoded.actual, decoded.t_ns)
        for field in ("pose_note", "align_note", "trail_note", "skeleton"):
            assert getattr(blank, field) == getattr(decoded, field), field


def test_a_recording_without_a_resolution_is_decoded_as_before(tmp_path, monkeypatch):
    """The frame's size comes from ``recording.yaml``; without one, the footage is decoded."""
    recording = _walking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    del data["video"]["resolution"]
    recording_store.write_recording(recording.dir, data)
    decoded, blank, decodes = _both_ways(recording, monkeypatch, 4, mask_trail=True)
    assert decodes, "no size to draw on, so the frame is decoded"
    np.testing.assert_array_equal(blank.masks, decoded.masks)


# --- the onion skin's cost --------------------------------------------------------------- #
def _whole_frame_onion_skin(canvas, layers, colour):
    """The onion skin as it was first written: one float32 blend over the whole frame. Kept as
    the reference the bounded blend must equal byte for byte."""
    from detail.api.run_preview import _MASK_TRAIL_ALPHA

    weight = np.zeros(canvas.shape[:2], dtype=np.float32)
    n = len(layers)
    for age, layer in enumerate(reversed(layers)):
        if layer is None:
            continue
        y0, x0, crop = layer
        alpha = np.float32(_MASK_TRAIL_ALPHA * (1.0 - age / n))
        region = weight[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]]
        np.maximum(region, np.where(crop, alpha, np.float32(0.0)), out=region)
    if not weight.any():
        return canvas
    tint = np.asarray(colour, dtype=np.float32)
    share = weight[..., None]
    blended = canvas.astype(np.float32) * (1.0 - share) + tint * share
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _trail_layers(rng, h, w, n):
    """``n`` random past masks as ``(y0, x0, crop)``, some frames without one."""
    layers = []
    for _ in range(n):
        if rng.random() < 0.2:
            layers.append(None)
            continue
        ch, cw = rng.integers(1, h // 3), rng.integers(1, w // 3)
        y0, x0 = rng.integers(0, h - ch), rng.integers(0, w - cw)
        layers.append((int(y0), int(x0), rng.random((ch, cw)) < 0.6))
    return layers


@pytest.mark.parametrize("seed", range(6))
def test_the_onion_skin_is_the_whole_frame_blend_byte_for_byte(seed):
    from detail.core.visualize import draw_onion_skin as _draw_onion_skin

    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    layers = _trail_layers(rng, 90, 160, int(rng.integers(1, 12)))
    before = canvas.copy()
    np.testing.assert_array_equal(
        _draw_onion_skin(canvas, layers, MASK_TINT),
        _whole_frame_onion_skin(canvas, layers, MASK_TINT),
    )
    np.testing.assert_array_equal(
        canvas, before
    )  # the canvas handed in is not drawn on


class _Watched(np.ndarray):
    """A canvas that records the shape of every float copy taken of it or its slices."""

    shapes: list = []

    def astype(self, dtype, *args, **kwargs):
        if np.dtype(dtype).kind == "f":
            _Watched.shapes.append(self.shape)
        return super().astype(dtype, *args, **kwargs)


def test_the_onion_skin_blends_only_where_the_trail_is():
    """A trail is a few small masks on a large frame: the blend covers their bounding box,
    not the frame. The float copy it makes is that box, 30x40 here, on a 1080p canvas — where
    the whole-frame blend copied all 1920x1080 pixels for every picture drawn."""
    from detail.core.visualize import draw_onion_skin as _draw_onion_skin

    canvas = np.full((1080, 1920, 3), 255, np.uint8).view(_Watched)
    crop = np.ones((20, 30), bool)
    _Watched.shapes = []
    out = _draw_onion_skin(
        canvas, [(500, 900, crop), None, (510, 910, crop)], MASK_TINT
    )
    assert _Watched.shapes, "the blend took no float copy: re-derive this guard"
    assert max(s[0] * s[1] for s in _Watched.shapes) == 30 * 40
    assert (np.asarray(out)[:500] == 255).all() and (np.asarray(out)[530:] == 255).all()
