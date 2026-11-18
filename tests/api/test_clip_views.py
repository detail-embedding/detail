"""The clip window's three pictures (``api.clip_views`` drawn from ``ClipQueryPreview.clip_frame``):
the footage with optional marks, every tracked animal's mask on white, and the keypoints the
embedding run used on white.

**What is pinned.** The masks pane shows *every* tracked animal, the clip's subject in the mask
tint and the others grey, so a second animal reads as one with the footage gone (the window is
for telling behaviours apart, and contact is half of that). The pose is **the keypoint run the
embedding read** — ``keypoint_run_id`` on its ``embedding.yaml`` — and an embedding with none
says so rather than drawing nothing or reaching for some other run. The footage's marks are
ticks: off, the frame is the decoded frame, byte for byte. A read without the footage is
``rgb=None``, never a white canvas that looks like one.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from detail.api import clip_views
from detail.api.clip_views import ClipViewSettings, MaskHistory
from detail.api.run_preview import ClipQueryPreview
from detail.core.errors import UserInputError
from detail.core.visualize import MASK_TINT
from detail.storage import recording as recording_store

from tests.clip_fixtures import (
    H,
    W,
    WALKER,
    count_decodes,
    two_animals,
    walking,
    with_heading,
    with_keypoints,
)

PLAIN = ClipViewSettings(masks_on_frame=False, points_on_frame=False)


def _preview(recording):
    return ClipQueryPreview(recording.dir.parent.parent)


def _views(recording, *, track_id=0, frame=0, masks=False, points=False, **kwargs):
    settings = ClipViewSettings(masks_on_frame=masks, points_on_frame=points, **kwargs)
    return _preview(recording).clip_views(
        recording.id, "emb_01", track_id, frame, settings
    )


def test_the_masks_pane_shows_every_animal_on_white_the_subject_tinted(tmp_path):
    recording = two_animals(tmp_path)
    views = _views(recording)
    masks = views.masks
    assert masks.shape == (H, W, 3) and masks.dtype == np.uint8
    assert (masks[0, 0] == 255).all(), "the background is white"
    assert tuple(masks[11, 18]) == tuple(MASK_TINT), "the clip's animal, tinted"
    other = masks[38, 48]
    assert other[0] == other[1] == other[2] < 255, "the other animal, grey"
    # the subject is whichever track the clip names
    assert tuple(_views(recording, track_id=1).masks[38, 48]) == tuple(MASK_TINT)


def test_the_footage_is_untouched_unless_a_mark_is_asked_for(tmp_path):
    recording = with_keypoints(two_animals(tmp_path))
    plain = _views(recording)
    decoded = _preview(recording).frame_plain(recording.id, "emb_01", 0, 0)[0]
    np.testing.assert_array_equal(plain.frame, decoded)
    masked = _views(recording, masks=True)
    assert (masked.frame[15, 15] != decoded[15, 15]).any()
    assert (masked.frame[35, 45] == decoded[35, 45]).all(), "only the subject's mask"
    pointed = _views(recording, points=True)
    assert (pointed.frame != decoded).any()
    assert (
        pointed.frame[30:41, 40:51] == decoded[30:41, 40:51]
    ).all(), "only the subject's points go on the footage"


def test_the_footage_tint_is_draw_masks_default_blend(tmp_path):
    """The subject's mask on the footage is `visualize.draw_mask`'s default blend — the one
    the clip renders use — so a clip looks the same wherever it is shown. (An old window
    tinted at 0.35 beside the since-deleted node cards at 0.5.)"""
    recording = two_animals(tmp_path)
    decoded = _preview(recording).frame_plain(recording.id, "emb_01", 0, 0)[0]
    masked = _views(recording, masks=True)
    expected = 0.5 * decoded[15, 15].astype(float) + 0.5 * np.asarray(MASK_TINT)
    np.testing.assert_allclose(masked.frame[15, 15].astype(float), expected, atol=1.5)


def test_the_pose_pane_is_the_embeddings_keypoint_run_on_white(tmp_path):
    recording = with_keypoints(two_animals(tmp_path))
    views = _views(recording)
    assert views.pose_note is None
    pose = views.pose
    assert pose.shape == (H, W, 3) and (pose[0, 0] == 255).all()
    assert (pose[10:21, 10:21] != 255).any(), "the subject's points"
    assert (pose[30:41, 40:51] != 255).any(), "the other animal's points, too"
    # a frame the run placed nothing on is a blank pose, not an error
    assert (_views(recording, frame=2).pose == 255).all()


def _walked(recording, **kwargs):
    """The last frame (4) of a ``walking`` recording, nothing on the footage but ``kwargs``."""
    settings = ClipViewSettings(masks_on_frame=False, points_on_frame=False, **kwargs)
    return _preview(recording).clip_views(recording.id, "emb_01", 0, 4, settings)


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
    """Each point's path, in the role colour ``draw_keypoints`` gives it, on the footage and
    on the Keypoints pane."""
    recording = walking(tmp_path)
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
    recording = walking(tmp_path)
    plain = _walked(recording)
    paw = _walked(recording, trail=True, trail_points=("left_paw",))
    for picture, before in ((paw.frame, plain.frame), (paw.pose, plain.pose)):
        assert _changed(picture, before, 5), "the chosen point trails"
        assert not _changed(picture, before, 13), "an unchosen point does not"
        assert not _changed(picture, before, 9)


def test_body_aligned_trails_are_paths_relative_to_the_body(tmp_path):
    """Aligned, each past frame is turned by its own axis: an animal walking without moving
    any point relative to its body leaves no trail on the pose — while the footage, which
    shows it crossing the arena, does."""
    recording = walking(tmp_path, heading=True)
    still = _walked(recording, pose_aligned=True)
    walked = _walked(recording, pose_aligned=True, trail=True)
    assert walked.align_note is None
    np.testing.assert_array_equal(walked.pose, still.pose)
    assert _changed(walked.frame, still.frame, 13)


def test_without_keypoints_the_trail_is_the_animals_centre_on_the_footage(tmp_path):
    """No keypoint run: the trail is the subject's centroid from the kinematics sidecar
    (`frame_slices.from_kinematics`), one line, in white, on the footage only."""
    recording = walking(tmp_path, keypoints=False)
    plain, traced = _walked(recording), _walked(recording, trail=True)
    assert traced.pose is None and traced.skeleton is None
    assert "centre" in traced.trail_note
    assert _changed(traced.frame, plain.frame, 9), "the centre walked along row 9"
    assert not _changed(traced.frame, plain.frame, 40), "one line, not one per point"
    ink = _ink(traced.frame, plain.frame, 9)
    assert ink.min() > plain.frame[9, 20].max(), "white over the footage"
    np.testing.assert_array_equal(traced.masks, plain.masks)


def test_without_keypoints_or_kinematics_there_is_no_trail_and_the_note_says_so(
    tmp_path,
):
    recording = walking(tmp_path, keypoints=False, kinematics=False)
    plain, traced = _walked(recording), _walked(recording, trail=True)
    np.testing.assert_array_equal(traced.frame, plain.frame)
    assert "backfill-kinematics" in traced.trail_note


def _whiteness_lost(masks, y, x):
    return 255 * 3 - int(masks[y, x].astype(int).sum())


def test_the_mask_trail_is_the_subjects_past_masks_fading_with_age(tmp_path):
    """The masks pane's onion skin: the subject's masks from the frames before, under the
    current one, each fainter than the one after it — and nobody else's."""
    recording = walking(tmp_path, keypoints=False)
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
    np.testing.assert_array_equal(
        skin.frame, plain.frame
    )  # no mask on the footage: no skin


def test_the_mask_trail_goes_under_the_mask_wherever_the_mask_is_drawn(tmp_path):
    """One rule instead of a second flag: with the subject's mask on the footage, its onion
    skin is there too."""
    recording = walking(tmp_path, keypoints=False)
    masked = _views(recording, frame=4, masks=True)
    skinned = _views(recording, frame=4, masks=True, mask_trail=True)
    assert (skinned.frame[9, 25] != masked.frame[9, 25]).any(), "frame 3's mask, faded"
    np.testing.assert_array_equal(skinned.frame[30:41], masked.frame[30:41])


def test_trails_decode_no_extra_video_frame(tmp_path, monkeypatch):
    """The past masks come from the detection rows and the past points from the keypoint
    run: one clip_views call decodes exactly the one frame it shows."""
    from detail.storage.video import FrameCursor

    recording = walking(tmp_path)
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
    recording = walking(tmp_path, keypoints=False)
    preview = _preview(recording)
    _walked(recording, mask_trail=True)  # builds the scrub context
    history = MaskHistory(capacity=2)
    scrub, _t = preview._scrub_ctx(recording.id, "emb_01")
    layers = history.layers(scrub, 0, 0, [0, 1, 2, 3], (H, W))
    assert all(layer is not None for layer in layers)
    assert history.held == 2


def test_an_embedding_that_read_no_keypoints_says_so(tmp_path):
    recording = two_animals(tmp_path)
    views = _views(recording, points=True)
    assert views.pose is None
    assert "used no keypoints" in views.pose_note
    assert "used no keypoints" in _preview(recording).clip_has_points(
        recording.id, "emb_01"
    )


def test_a_keypoint_run_that_cannot_be_read_says_which(tmp_path):
    recording = with_keypoints(two_animals(tmp_path))
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["keypoint_run_id"] = "kt_09"
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    views = _views(recording)
    assert views.pose is None and "kt_09" in views.pose_note


def _aligned(recording, frame=0):
    settings = ClipViewSettings(
        masks_on_frame=False, points_on_frame=False, pose_aligned=True
    )
    return _preview(recording).clip_views(recording.id, "emb_01", 0, frame, settings)


def test_the_aligned_pose_is_the_subject_alone_facing_up_at_the_centre(tmp_path):
    recording = with_heading(with_keypoints(two_animals(tmp_path)))
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
    recording = with_keypoints(two_animals(tmp_path))
    views = _aligned(recording)
    assert "No heading definition" in views.align_note
    assert np.array_equal(views.pose, _views(recording).pose), "drawn as usual"


def test_a_frame_without_the_subjects_axis_is_blank_but_for_a_line(tmp_path):
    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    # track 0 is tracked on frame 2 but has no keypoints there
    views = _aligned(recording, frame=2)
    inked = (views.pose < 250).any(axis=2)
    assert inked.any() and not inked[: H // 4].any(), "a line of text mid-pane, no pose"


# --- without the footage ----------------------------------------------------------------- #
def _both_ways(recording, monkeypatch, frame, **kwargs):
    """``(decoded, blank, decodes)``: the views drawn from the footage and without it, each
    by a fresh preview, and the decodes the second one made."""
    root = recording.dir.parent.parent
    decoded = ClipQueryPreview(root).clip_views(
        recording.id, "emb_01", 0, frame,
        ClipViewSettings(masks_on_frame=False, points_on_frame=False, **kwargs),
    )  # fmt: skip
    calls = count_decodes(monkeypatch)
    blank = ClipQueryPreview(root).clip_views(
        recording.id, "emb_01", 0, frame,
        ClipViewSettings(masks_on_frame=True, points_on_frame=True, **kwargs),
        footage=False,
    )  # fmt: skip
    monkeypatch.undo()
    return decoded, blank, list(calls)


@pytest.mark.parametrize(
    "build, kwargs",
    [
        ("two_animals", {}),
        ("walking", {}),
        ("walking", {"trail": True, "mask_trail": True}),
        ("walking", {"trail": True, "trail_points": ("nose",), "mask_trail": True}),
        ("heading", {"trail": True, "pose_aligned": True}),
        ("no_keypoints", {"trail": True, "mask_trail": True}),
    ],
)
def test_without_the_footage_the_panes_are_the_decoded_ones_and_nothing_is_decoded(
    tmp_path, monkeypatch, build, kwargs
):
    """``footage=False`` is for a caller that shows only the masks and the keypoints: it
    must draw them pixel for pixel as the decode does, at the same frame and time, and never
    open the video. There is no footage picture at all — not a white one."""
    recording = {
        "two_animals": lambda: with_keypoints(two_animals(tmp_path)),
        "walking": lambda: walking(tmp_path),
        "heading": lambda: walking(tmp_path, heading=True),
        "no_keypoints": lambda: walking(tmp_path, keypoints=False),
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
        assert blank.frame is None
        assert (blank.actual, blank.t_ns) == (decoded.actual, decoded.t_ns)
        for field in ("pose_note", "align_note", "trail_note", "skeleton"):
            assert getattr(blank, field) == getattr(decoded, field), field


def test_without_footage_there_is_no_frame_to_draw_on(tmp_path):
    """A read without the footage holds no picture of it, and drawing one is an error — a
    blank screen becomes a traceback rather than a white pane that looks like a frame.
    """
    recording = walking(tmp_path)
    frame = _preview(recording).clip_frame(
        recording.id, "emb_01", 0, 4, PLAIN, footage=False
    )
    assert frame.rgb is None and frame.frame_hw == (H, W)
    with pytest.raises(ValueError, match="no footage was read"):
        clip_views.draw_footage(frame, PLAIN, 0)
    assert clip_views.draw_masks_pane(frame, PLAIN, 0).shape == (H, W, 3)


def test_without_a_recorded_resolution_the_api_refuses_rather_than_decoding(
    tmp_path, monkeypatch
):
    """The panes' size comes from ``recording.yaml``; without one the API has no size to
    draw at and says so — it never decodes the footage behind a caller's back. The window
    that asked decides (it decodes, and says so)."""
    recording = walking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    del data["video"]["resolution"]
    recording_store.write_recording(recording.dir, data)
    calls = count_decodes(monkeypatch)
    with pytest.raises(ValueError, match="video.resolution"):
        _preview(recording).clip_views(
            recording.id, "emb_01", 0, 4, PLAIN, footage=False
        )
    assert calls == []
    assert (
        _preview(recording).clip_views(recording.id, "emb_01", 0, 4, PLAIN) is not None
    )


# --- the pane exports -------------------------------------------------------------------- #
def test_pane_picture_and_title():
    views = clip_views.ClipViews(
        frame=None, masks=np.zeros((2, 2, 3), np.uint8), pose=None, pose_note="none",
        actual=0, t_ns=0,
    )  # fmt: skip
    assert clip_views.pane_picture(views, clip_views.MASKS) is views.masks
    assert clip_views.pane_picture(views, clip_views.KEYPOINTS) is None
    with pytest.raises(ValueError, match="pane must be one of"):
        clip_views.pane_picture(views, "footage")
    assert clip_views.pane_title(clip_views.MASKS) == "Masks"
    assert clip_views.pane_title(clip_views.KEYPOINTS, "clip 3") == "Keypoints · clip 3"


def test_pane_frames_sample_real_time_or_take_every_frame():
    t_ns = (np.arange(10) * 1e9 / 30).astype(np.int64)
    assert clip_views.pane_frames(t_ns, 2, 6, None) == [2, 3, 4, 5, 6]
    assert clip_views.pane_frames(t_ns, 2, 6, 15.0) == [2, 4, 6]
    assert clip_views.pane_frames(t_ns, 4, 4, 15.0) == [4]


def test_pane_frames_clamp_a_span_past_the_end_and_refuse_one_that_starts_there():
    """The montage export clamps a span past the video's end; the pane and view exports
    raised a bare ``IndexError`` from deep in the frame reader instead (#33)."""
    t_ns = (np.arange(5) * 1e9 / 30).astype(np.int64)
    assert clip_views.pane_frames(t_ns, 3, 40, None) == [3, 4]
    assert clip_views.pane_frames(t_ns, 3, 40, 30.0) == [3, 4]
    with pytest.raises(
        UserInputError, match=r"frame 5 is past the end .* \(5 frames\)"
    ):
        clip_views.pane_frames(t_ns, 5, 40, None)


def test_a_pane_export_decodes_no_video_and_writes_what_the_pane_shows(
    tmp_path, monkeypatch
):
    from PIL import Image

    recording = walking(tmp_path, keypoints=False)
    calls = count_decodes(monkeypatch)
    out = clip_views.export_clip_pane(
        recording.dir.parent.parent,
        recording_id=recording.id,
        embed_run_id="emb_01",
        track_id=0,
        first=0,
        last=4,
        pane=clip_views.MASKS,
        settings=ClipViewSettings(mask_trail=True),
        path=tmp_path / "masks.gif",
        max_width=W,
        fps=30.0,  # every frame; a GIF alone would sample at its default 12
    )
    assert calls == []
    with Image.open(out) as gif:
        assert gif.n_frames == 5 and gif.size == (W, H)


def test_a_keypoint_pane_export_of_a_run_with_none_is_refused(tmp_path):
    recording = walking(tmp_path, keypoints=False)
    with pytest.raises(ValueError, match="no keypoints to export"):
        clip_views.export_clip_pane(
            recording.dir.parent.parent,
            recording_id=recording.id,
            embed_run_id="emb_01",
            track_id=0,
            first=0,
            last=4,
            pane=clip_views.KEYPOINTS,
            settings=PLAIN,
            path=tmp_path / "pose.mp4",
        )


def _export(recording, tmp_path, pane, name, **kwargs):
    return clip_views.export_clip_pane(
        recording.dir.parent.parent,
        recording_id=recording.id,
        embed_run_id="emb_01",
        track_id=0,
        first=kwargs.pop("first", 0),
        last=kwargs.pop("last", 4),
        pane=pane,
        settings=kwargs.pop("settings", PLAIN),
        path=tmp_path / name,
        **kwargs,
    )


def test_a_pane_export_past_the_video_end_writes_what_there_is(tmp_path):
    """An mp4 with no ``fps`` takes every frame of the span, the path that indexed past the
    end; it writes the frames there are."""
    import cv2

    recording = walking(tmp_path, keypoints=False)  # 5 frames
    out = _export(recording, tmp_path, clip_views.MASKS, "m.mp4", first=3, last=40)
    cap = cv2.VideoCapture(str(out))
    try:
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == 2


def test_an_unknown_pane_is_refused(tmp_path):
    recording = walking(tmp_path, keypoints=False)
    with pytest.raises(ValueError, match="pane must be one of"):
        _export(recording, tmp_path, "footage", "x.mp4")


def test_an_mp4_pane_export_keeps_every_frame_at_full_size(tmp_path):
    """An mp4 is the file to keep: every frame, the footage's own size and rate."""
    import cv2

    recording = walking(tmp_path, keypoints=False)
    out = _export(recording, tmp_path, clip_views.MASKS, "masks.mp4")
    cap = cv2.VideoCapture(str(out))
    try:
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert size == (W, H) and count == 5


def test_the_keypoints_pane_export_is_the_pose_on_white(tmp_path):
    """The file is what the enlarged pane shows: the pose picture, not a re-drawing."""
    from PIL import Image

    recording = walking(tmp_path)
    out = _export(
        recording, tmp_path, clip_views.KEYPOINTS, "pose.gif",
        first=4, last=4, max_width=W, fps=30.0,
    )  # fmt: skip
    with Image.open(out) as gif:
        written = np.asarray(gif.convert("RGB"))
    pose = (
        _preview(recording)
        .clip_views(recording.id, "emb_01", 0, 4, PLAIN, footage=False)
        .pose
    )
    # A GIF quantizes colours: compare where the ink is, not the bytes.
    inked = (written < 250).any(axis=2) == (pose < 250).any(axis=2)
    assert inked.mean() > 0.98
