"""The **Aligned** orientation (``api.clip_views``' turn), on every surface that draws a clip.

One turn — :func:`clip_views.turn_matrix` — used by the clip window (``ClipQueryPreview.
clip_views``), the live montage's tiles (``previews.PreviewClip.set_pose`` via
``clip_tiles.ClipTiles``) and the exports (``clip_montage.ClusterMontage``). What is pinned: the
body ends up pointing **up** (the clip window's Align pose direction), about its middle, at the
picture's centre; the footage, the masks and the keypoints turn together; a frame with no body axis
is drawn as recorded and marked, never blank; a run with no keypoints or no heading definition
says why, in the clip window's words.

**A side view is shown as its embedding drew it.** The class this guards: a surface that aligns a
clip by the body axis, ignoring how its embedding aligned. A ``mirror`` (side-view) embedding
mirrors the animal to face right and never turns it, yet every surface here once turned it
nose-up, because the one turn read the axis and not the run. It hid three more ways, each pinned
below:

- **Recomputing the flag instead of reading it.** The first fix re-derived each frame's flip from
  the keypoints (``pipeline.embedding.headings_for_run``) at a step of one frame, at the video's
  size and by the heading definition as it reads today. The embedding flipped at its own
  ``every_n_frames`` (the mirror hold runs over the frames it sampled, so another step gives
  other flags), at the masks' size, by the definition it was given, and it stored the answer:
  ``align_mirrored`` on every embedded row (§14). The recomputation could disagree with the
  embedding, raised on every frame once the definition no longer fit the skeleton, and ran over
  the whole recording on the first frame of any clip, Aligned or not. ``ClipPoints.flips`` reads
  the stored column, held between embedded frames at the run's step.
- **Asking the keypoints first.** A frame the embedding carried a heading across (its nose lost
  for a moment) was flipped by the run, but each surface checked for this frame's body axis
  before it looked at the flip, and drew it unflipped and marked *no axis*. A side view now reads
  only its flip (:func:`clip_views.mirror_as_embedded`).
- **Align pose.** The clip window's pose pane, scaled and centred, still turned a side view
  nose-up beside mirrored footage. It is now mirrored where the run mirrored it, never turned
  (``body_axis.side_view_points``).

A side view is also drawn as the surface draws it as recorded, its columns reversed where
flipped: never resampled, and never placed on each frame's body-axis middle, which made the live
tiles bob with every jitter of the keypoints while the exports held still.
"""

from __future__ import annotations

import numpy as np
import pytest

from detail.api import clip_views
from detail.api.clip_views import ClipViewSettings
from detail.api.run_preview import ClipQueryPreview
from detail.analysis.body_axis import BodyAxis
from detail.core.visualize import MASK_TINT
from tests.clip_fixtures import H
from tests.clip_fixtures import W
from tests.clip_fixtures import two_animals
from tests.clip_fixtures import with_heading
from tests.clip_fixtures import with_keypoints
from tests.clip_fixtures import with_side_heading


def _views(recording, frame=0, **kwargs):
    settings = ClipViewSettings(masks_on_frame=False, points_on_frame=False, **kwargs)
    return ClipQueryPreview(recording.dir.parent.parent).clip_views(
        recording.id, "emb_01", 0, frame, settings
    )


def _tinted(picture) -> np.ndarray:
    return np.abs(picture.astype(int) - np.array(MASK_TINT)).sum(axis=2) < 30


# --- the turn itself ------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "front, back",
    [
        ((10, 10), (20, 20)),
        ((30, 5), (5, 30)),
        ((5, 40), (5, 10)),
        ((50, 20), (10, 20)),
    ],
)
def test_the_turn_puts_the_front_straight_above_the_back_at_the_centre(front, back):
    out = (64, 48)
    matrix = clip_views.turn_matrix(np.array(front), np.array(back), 48, out)
    moved = np.array([front, back], dtype=float) @ matrix[:, :2].T + matrix[:, 2]
    (fx, fy), (bx, by) = moved
    assert fx == pytest.approx(32) and bx == pytest.approx(32), "on one vertical line"
    assert fy < by, "the front is above the back: the body points up"
    assert (fy + by) / 2 == pytest.approx(24), "its middle at the centre"
    length = np.hypot(*np.subtract(front, back))
    assert by - fy == pytest.approx(length), "a side of the output's height: no scaling"


@pytest.mark.parametrize("flip", [0.0, 1.0, float("nan")])
def test_a_side_view_picture_is_the_recorded_one_reversed_where_flipped(flip):
    picture = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    shown = clip_views.mirror_as_embedded(picture, flip)
    if np.isnan(flip):
        assert (
            shown is None
        ), "no embedded row: the caller draws it as recorded and marks it"
    else:
        expected = picture[:, ::-1] if flip else picture
        assert np.array_equal(shown, expected), "exact: no resampling, never turned"


def test_align_pose_places_a_side_view_unturned_and_mirrored_where_flipped():
    axis = BodyAxis((0,), (1,), (), (W, H), None, 10.0)
    # Nose up-left of its tail, as if rearing while facing left.
    xy = np.array([[[20 / W, 10 / H], [26 / W, 18 / H]]] * 3)
    scores = np.full((3, 2), 0.9)
    placed = clip_views.aligned_window(
        xy, scores, axis, (W, H), flips=np.array([1.0, 0.0, np.nan])
    )
    nose, tail = placed[0] * [W, H]
    assert (
        nose[0] > tail[0] and nose[1] < tail[1]
    ), "mirrored: rearing, now facing right"
    assert (nose + tail) / 2 == pytest.approx([W / 2, H / 2]), "centred on the body"
    nose, tail = placed[1] * [W, H]
    assert nose[0] < tail[0] and nose[1] < tail[1], "unflipped: as it stood, not turned"
    assert np.isnan(placed[2]).all(), "no embedded row: not placed"
    nose, tail = clip_views.aligned_window(xy, scores, axis, (W, H))[0] * [W, H]
    assert nose[0] == pytest.approx(tail[0]), "an overhead run is turned up"


def test_a_frame_with_no_axis_is_marked_not_blanked():
    picture = np.full((48, 64, 3), 200, np.uint8)
    marked = clip_views.mark_no_axis(picture)
    assert (marked[2, 2] == 40).all(), "the mark's box in the top-left corner"
    assert (marked[40, 60] == 200).all(), "the rest is the picture as recorded"


# --- the clip window ---------------------------------------------------------------------- #
def test_aligned_turns_the_footage_the_masks_and_the_pose_together(tmp_path):
    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    plain = _views(recording)
    turned = _views(recording, aligned=True)
    # The subject faced up-left (nose 12,12 over tail 18,18): its 10-pixel mask, centred on
    # (15, 15), is turned 45° about that point and put at the picture's centre.
    assert _tinted(plain.masks)[15, 15] and not _tinted(plain.masks)[24, 32]
    assert _tinted(turned.masks)[24, 32], "the subject's middle at the centre"
    assert not _tinted(turned.masks)[12, 12], "nothing left where it was"

    def height(picture):
        rows = np.argwhere(_tinted(picture))[:, 0]
        return rows.max() - rows.min()

    # A square turned by 45° is a diamond: taller than the square it was (its diagonal).
    assert height(turned.masks) >= height(plain.masks) + 2
    expected = clip_views.turn_picture(
        _views(recording).frame,
        clip_views.turn_matrix(
            np.array([12.0 / W * W, 12.0 / H * H]),
            np.array([18.0 / W * W, 18.0 / H * H]),
            H,
            (W, H),
        ),
        (W, H),
        (0, 0, 0),
    )
    assert np.array_equal(turned.frame, expected), "the footage, by the same turn"
    assert not np.array_equal(turned.pose, plain.pose), "the pose pane turns with it"


def test_a_frame_without_the_subjects_axis_plays_as_recorded_and_says_so(tmp_path):
    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    # Track 0 is tracked on frame 2 but has no keypoints there.
    plain = _views(recording, frame=2)
    turned = _views(recording, frame=2, aligned=True)
    assert (turned.frame[2, 2] == 40).all(), "marked"
    assert np.array_equal(turned.frame[30:], plain.frame[30:]), "as recorded below it"
    assert np.array_equal(turned.masks, plain.masks)


def _side_frame(recording, frame, **kwargs):
    preview = ClipQueryPreview(recording.dir.parent.parent)
    return preview.clip_frame(
        recording.id, "emb_01", 0, frame, ClipViewSettings(**kwargs)
    )


def test_a_side_view_clip_is_flipped_as_its_embedding_flipped_it_not_turned_up(
    tmp_path,
):
    recording = with_side_heading(with_keypoints(two_animals(tmp_path)))
    # Nose (12, 12) over tail (18, 18): facing up-left, so the embedding mirrored it.
    assert _side_frame(recording, 0, aligned=True).flip == 1.0
    plain, aligned = _views(recording), _views(recording, aligned=True)
    # The picture as recorded, mirrored: not turned, and not moved onto the body's middle
    # (which would move it with every jitter of the keypoints).
    assert np.array_equal(aligned.masks, plain.masks[:, ::-1])
    assert np.array_equal(aligned.frame, plain.frame[:, ::-1])


def test_the_flip_shown_is_the_one_the_embedding_stored_not_one_recomputed(tmp_path):
    # Frame 0 faces left, so a flip recomputed from its keypoints would mirror it; the run
    # stored that it did not (another step, frame size or definition can make it so).
    recording = with_side_heading(
        with_keypoints(two_animals(tmp_path)), {0: False, 2: True}
    )
    assert _side_frame(recording, 0, aligned=True).flip == 0.0
    assert np.array_equal(
        _views(recording, aligned=True).masks, _views(recording).masks
    ), "drawn as the embedding drew it: unflipped"


def test_a_side_view_frame_with_no_keypoints_is_flipped_by_its_stored_flag(tmp_path):
    # Frame 2 has no keypoints, but the run carried a heading across it and mirrored it.
    recording = with_side_heading(with_keypoints(two_animals(tmp_path)))
    plain = _views(recording, frame=2)
    aligned = _views(recording, frame=2, aligned=True)
    # Mirrored, not marked: a side view never asks the keypoints.
    assert np.array_equal(aligned.frame, plain.frame[:, ::-1])
    assert np.array_equal(aligned.masks, plain.masks[:, ::-1])


def test_a_side_view_frame_its_embedding_has_no_row_for_plays_as_recorded(tmp_path):
    recording = with_side_heading(with_keypoints(two_animals(tmp_path)), {0: True})
    frame = _side_frame(recording, 2, aligned=True)
    assert frame.flip is not None and np.isnan(frame.flip)
    plain = _views(recording, frame=2)
    aligned = _views(recording, frame=2, aligned=True)
    assert (aligned.frame[2, 2] == 40).all(), "marked"
    assert np.array_equal(aligned.frame[20:], plain.frame[20:]), "and drawn as recorded"
    assert np.array_equal(aligned.masks, plain.masks)


@pytest.mark.parametrize("every_n_frames", [1, 2])
def test_between_embedded_frames_a_frame_holds_the_flag_of_the_one_it_follows(
    tmp_path, every_n_frames
):
    recording = with_side_heading(
        with_keypoints(two_animals(tmp_path)),
        {0: True, 2: False},
        every_n_frames=every_n_frames,
    )
    points = ClipQueryPreview(recording.dir.parent.parent).clip_points(
        recording.id, "emb_01"
    )
    nan = np.nan
    if every_n_frames == 1:
        expected = [
            1.0,
            nan,
            0.0,
            nan,
            nan,
        ]  # every frame was sampled; 1 and 3 have no row
    else:
        expected = [1.0, 1.0, 0.0, 0.0, nan]  # 1 and 3 were never sampled; 4 has no row
    np.testing.assert_array_equal(points.flips(0, "part_001", range(5)), expected)
    np.testing.assert_array_equal(points.flip_window(0, [5], 1, 3), expected[1:4])


def test_a_side_view_whose_definition_no_longer_fits_plays_as_recorded(tmp_path):
    import yaml

    recording = with_side_heading(with_keypoints(two_animals(tmp_path)))
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["heading_definition"] = {"front": ["snout"], "back": ["tail"]}
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    plain, aligned = _views(recording), _views(recording, aligned=True)
    assert "does not fit" in aligned.align_note
    assert np.array_equal(aligned.masks, plain.masks), "greyed, so as recorded"


def test_a_side_view_reads_its_flips_only_for_what_draws_one(tmp_path, monkeypatch):
    recording = with_side_heading(with_keypoints(two_animals(tmp_path)))
    asked = []
    read = clip_views.ClipPoints.flips

    def flips(self, *args):
        asked.append(args)
        return read(self, *args)

    monkeypatch.setattr(clip_views.ClipPoints, "flips", flips)
    assert _side_frame(recording, 0).flip is None
    assert _side_frame(recording, 0, trail=True).flip is None
    assert not asked, "as recorded, nothing reads a side view's flips"
    assert _side_frame(recording, 0, aligned=True).flip == 1.0
    assert _side_frame(recording, 0, pose_aligned=True).flip == 1.0
    assert len(asked) == 2


def test_align_pose_on_a_side_view_faces_the_way_the_embedding_drew_it(tmp_path):
    def pose(name, flips):
        recording = with_side_heading(
            with_keypoints(two_animals(tmp_path / name)), flips
        )
        settings = ClipViewSettings(
            masks_on_frame=False, points_on_frame=False, pose_aligned=True
        )
        preview = ClipQueryPreview(recording.dir.parent.parent)
        return preview.clip_views(recording.id, "emb_01", 0, 0, settings).pose

    def lean(picture):
        """Mean column of the ink above the middle row less that below it: > 0 when the
        nose (above) is right of the tail."""
        rows, cols = np.nonzero((picture < 250).any(axis=2))
        return cols[rows < H // 2].mean() - cols[rows > H // 2].mean()

    # Nose (12, 12) up-left of tail (18, 18): turned up, it would not lean at all.
    assert lean(pose("mirrored", {0: True})) > 6, "mirrored: up and to the right"
    assert lean(pose("as_drawn", {0: False})) < -6, "as the run drew it: up and left"


def test_an_overhead_run_has_no_flip(tmp_path):
    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    assert _side_frame(recording, 0, aligned=True).flip is None


def test_without_a_heading_definition_aligned_changes_nothing_and_says_why(tmp_path):
    recording = with_keypoints(two_animals(tmp_path))
    turned = _views(recording, aligned=True)
    assert np.array_equal(turned.frame, _views(recording).frame), "unmarked, untouched"
    assert "No heading definition" in turned.align_note
    root = recording.dir.parent.parent
    points_note, align_note = clip_views.pose_notes(root, {recording.id: "emb_01"})
    assert points_note is None and "No heading definition" in align_note


def test_the_run_level_notes_say_what_is_missing(tmp_path):
    recording = two_animals(tmp_path)
    root = recording.dir.parent.parent
    points_note, align_note = clip_views.pose_notes(root, {recording.id: "emb_01"})
    assert "used no keypoints" in points_note and "used no keypoints" in align_note
    with_heading(with_keypoints(recording))
    assert clip_views.pose_notes(root, {recording.id: "emb_01"}) == (None, None)
    assert clip_views.alignment_note(root, {}) is not None


# --- the exports of the clip window's view ------------------------------------------------- #
def test_the_view_picture_is_the_footage_or_the_footage_with_its_panes(tmp_path):
    recording = with_keypoints(two_animals(tmp_path))
    views = _views(recording)
    alone = clip_views.view_picture(views, panes=False)
    assert np.array_equal(alone, views.frame)
    whole = clip_views.view_picture(views, panes=True)
    assert whole.shape[0] == H and whole.shape[1] > W
    assert np.array_equal(whole[:, :W], views.frame)
    assert (whole[:, W : W + clip_views.VIEW_GAP_PX] == 255).all(), "the gap"


def test_this_frame_as_png_is_what_the_window_shows(tmp_path):
    from PIL import Image

    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    settings = ClipViewSettings(masks_on_frame=True, aligned=True)
    path = clip_views.write_clip_frame(
        recording.dir.parent.parent,
        recording_id=recording.id,
        embed_run_id="emb_01",
        track_id=0,
        frame=0,
        settings=settings,
        panes=True,
        path=tmp_path / "frame",
    )
    assert path.suffix == ".png"
    written = np.asarray(Image.open(path).convert("RGB"))
    views = ClipQueryPreview(recording.dir.parent.parent).clip_views(
        recording.id, "emb_01", 0, 0, settings
    )
    assert np.array_equal(written, clip_views.view_picture(views, panes=True))


def test_a_view_export_writes_every_frame_of_the_span(tmp_path):
    import cv2

    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    out = clip_views.export_clip_view(
        recording.dir.parent.parent,
        recording_id=recording.id,
        embed_run_id="emb_01",
        track_id=0,
        first=0,
        last=2,
        settings=ClipViewSettings(aligned=True),
        panes=False,
        path=tmp_path / "view.mp4",
    )
    cap = cv2.VideoCapture(str(out))
    try:
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == 3, "every frame of [first, last], the context included"


# --- the live montage's tiles ------------------------------------------------------------ #
def _tile_shape(tile) -> tuple[float, float, int, int]:
    """``(centre row, centre col, height, width)`` of a tile's tinted (filled) animal."""
    rows, cols = np.nonzero(_tinted(tile))
    return rows.mean(), cols.mean(), np.ptp(rows) + 1, np.ptp(cols) + 1


def _ends_facing_right(clip) -> np.ndarray:
    """Per tile frame of ``test_previews``' tracked walker: its mask's middle, with the body
    axis pointing right (the walker's 12 × 10 box at ``x0 = 2 + frame // 4`` of a 40 px frame).
    """
    src = clip.source_frames()
    centre_x = (2 + src // 4 + 5) / 40.0
    centre_y = np.full(len(src), 10 / 40.0)
    front = np.stack([centre_x + 0.1, centre_y], axis=1)
    back = np.stack([centre_x - 0.1, centre_y], axis=1)
    return np.stack([front, back], axis=1)


def test_an_aligned_tile_turns_the_animal_up_about_its_middle(
    processed_recording, write_vfr_video
):
    from detail.api import previews
    from tests.api.test_previews import _timestamps, _tracked_recording

    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    with reader.clip(0, int(t[10]), int(t[40]), 64, footage=False, trail=1) as clip:
        as_recorded = clip.frame_at(0)
        clip.set_pose(axes=_ends_facing_right(clip))
        aligned = clip.frame_at(0)
        _r, _c, tall, wide = _tile_shape(as_recorded)
        row, col, tall_now, wide_now = _tile_shape(aligned)
        assert tall > wide and wide_now > tall_now, "a quarter turn: right became up"
        assert row == pytest.approx(32, abs=1.5) and col == pytest.approx(32, abs=1.5)
        none = np.full_like(_ends_facing_right(clip), np.nan)
        clip.set_pose(axes=none)
        marked = clip.frame_at(0)
        assert (marked[1, 1] == 40).all(), "no axis: marked"
        assert np.array_equal(marked[40:], as_recorded[40:]), "and drawn as recorded"


def test_a_side_view_tile_is_the_recorded_tile_mirrored_and_marked_without_a_row(
    processed_recording, write_vfr_video
):
    from detail.api import previews
    from tests.api.test_previews import _timestamps, _tracked_recording

    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    with reader.clip(0, int(t[10]), int(t[40]), 64, footage=False, trail=1) as clip:
        as_recorded = clip.frame_at(0)
        n = len(clip)
        clip.set_pose(flips=np.ones(n))
        assert np.array_equal(clip.frame_at(0), as_recorded[:, ::-1]), "exact, unturned"
        clip.set_pose(flips=np.zeros(n))
        assert np.array_equal(clip.frame_at(0), as_recorded), "unflipped: as recorded"
        clip.set_pose(flips=np.full(n, np.nan))
        marked = clip.frame_at(0)
        assert (marked[1, 1] == 40).all(), "no embedded row: marked"
        assert np.array_equal(marked[40:], as_recorded[40:]), "and drawn as recorded"
        with pytest.raises(ValueError, match="pass one"):
            clip.set_pose(axes=_ends_facing_right(clip), flips=np.ones(n))


def test_the_tile_source_hands_a_side_view_tile_its_runs_flips(
    processed_recording, write_vfr_video, monkeypatch
):
    from types import SimpleNamespace

    from detail.api.clip_tiles import TileLook
    from tests.api.test_clip_tiles import _tiles
    from tests.api.test_previews import _timestamps, _tracked_recording

    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[10]), int(t[40]))])

    def rows_at(track_id, stem, frames):
        x = (2 + np.asarray(frames) // 4 + 5) / 40.0
        points = np.stack(
            [np.stack([x + 0.1, np.full(len(x), 0.25)], 1),
             np.stack([x - 0.1, np.full(len(x), 0.25)], 1)], axis=1,
        )  # fmt: skip
        return points, np.full(points.shape[:2], 0.9)

    asked = []

    def flips(track_id, stem, frames):
        asked.append((track_id, stem))
        return np.ones(len(frames))

    points = SimpleNamespace(
        run=object(),
        skeleton=None,
        rows_at=rows_at,
        flips=flips,
        axis=lambda track: BodyAxis((0,), (1,), (), (40, 40), None, 10.0),
    )
    monkeypatch.setattr(tiles._catalog, "points", lambda rid: points)
    monkeypatch.setattr(tiles._catalog, "part_stem", lambda rid, part: "part_001")
    plain = tiles.frame(0, 0, 64, True, look=TileLook(footage=False))
    aligned = tiles.frame(0, 0, 64, True, look=TileLook(footage=False, aligned=True))
    assert asked == [(0, "part_001")], "the side-view flips are asked for once per clip"
    assert np.array_equal(aligned, plain[:, ::-1]), "the recorded tile, mirrored"
    tiles.close()


def test_the_tile_source_reads_the_keypoints_once_and_turns_its_tiles(
    processed_recording, write_vfr_video, monkeypatch
):
    from types import SimpleNamespace

    from detail.analysis.body_axis import BodyAxis
    from detail.api import clip_tiles
    from detail.api.clip_tiles import TileLook
    from tests.api.test_clip_tiles import _tiles
    from tests.api.test_previews import _timestamps, _tracked_recording

    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[10]), int(t[40]))])
    asked = []

    def rows_at(track_id, stem, frames):
        asked.append((track_id, stem, tuple(frames)))
        x = (2 + np.asarray(frames) // 4 + 5) / 40.0
        points = np.stack(
            [np.stack([x + 0.1, np.full(len(x), 0.25)], 1),
             np.stack([x - 0.1, np.full(len(x), 0.25)], 1)], axis=1,
        )  # fmt: skip
        return points, np.full(points.shape[:2], 0.9)

    points = SimpleNamespace(
        run=object(),
        skeleton=None,
        rows_at=rows_at,
        flips=lambda track_id, stem, frames: None,
        axis=lambda track: BodyAxis((0,), (1,), (), (40, 40), None, 10.0),
    )
    monkeypatch.setattr(tiles._catalog, "points", lambda rid: points)
    monkeypatch.setattr(tiles._catalog, "part_stem", lambda rid, part: "part_001")
    plain = tiles.frame(0, 0, 64, True, look=TileLook(footage=False))
    turned = tiles.frame(0, 0, 64, True, look=TileLook(footage=False, aligned=True))
    tiles.frame(0, 1, 64, True, look=TileLook(footage=False, aligned=True))
    assert len(asked) == 1, "the clip's keypoints are read once, when it is opened"
    _r, _c, tall, wide = _tile_shape(plain)
    row, col, tall_now, wide_now = _tile_shape(turned)
    assert tall > wide and wide_now > tall_now
    assert row == pytest.approx(32, abs=1.5) and col == pytest.approx(32, abs=1.5)
    assert clip_tiles.tile_axes(*rows_at(0, "", [8])[0:2], points.axis(0)).shape == (
        1,
        2,
        2,
    )
    tiles.close()


# --- the exports ------------------------------------------------------------------------- #
def _montage(recording):
    from detail.api.clip_montage import ClusterMontage, MontageClip

    clip = MontageClip(
        rank=1,
        recording_id=recording.id,
        recording_name="r",
        subject=0,
        embed_run_id="emb_01",
        start_t_ns=0,
        end_t_ns=int(2e9 / 30),
    )
    montage = ClusterMontage(recording.dir.parent.parent, [clip], workers=1)
    montage.prepare()
    return montage


def test_an_aligned_export_tile_is_turned_and_a_frame_without_an_axis_is_marked(
    tmp_path,
):
    from detail.api.clip_montage import MontageLayers

    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    montage = _montage(recording)
    try:
        plain = montage.tile(0, 0, MontageLayers(mask=True), 48)
        turned = montage.tile(0, 0, MontageLayers(mask=True, aligned=True), 48)
        assert not np.array_equal(plain, turned)
        row, col, _tall, _wide = _tile_shape(turned)
        assert row == pytest.approx(24, abs=1.5) and col == pytest.approx(24, abs=1.5)
        marked = montage.tile(0, 2, MontageLayers(mask=True, aligned=True), 48)
        assert (marked[1, 1] == 40).all()
        raw = montage.tile(0, 0, MontageLayers(raw=True, mask=True, aligned=True), 48)
        assert _tinted(raw).any(), "the outline is placed by the same turn"
    finally:
        montage.close()


def test_a_side_view_export_tile_is_the_recorded_tile_mirrored(tmp_path):
    from detail.api.clip_montage import MontageLayers

    recording = with_side_heading(with_keypoints(two_animals(tmp_path)))
    montage = _montage(recording)
    try:
        for frame in (0, 2):  # frame 2 has no keypoints; its stored flag still flips it
            for raw in (False, True):
                plain = montage.tile(0, frame, MontageLayers(raw=raw, mask=True), 48)
                aligned = montage.tile(
                    0, frame, MontageLayers(raw=raw, mask=True, aligned=True), 48
                )
                assert np.array_equal(aligned, plain[:, ::-1]), (frame, raw)
    finally:
        montage.close()


def test_a_side_view_export_tile_with_no_embedded_row_is_marked(tmp_path):
    from detail.api.clip_montage import MontageLayers

    recording = with_side_heading(with_keypoints(two_animals(tmp_path)), {0: True})
    montage = _montage(recording)
    try:
        plain = montage.tile(0, 2, MontageLayers(mask=True), 48)
        marked = montage.tile(0, 2, MontageLayers(mask=True, aligned=True), 48)
        assert (marked[1, 1] == 40).all()
        assert np.array_equal(marked[24:], plain[24:]), "and drawn as recorded"
    finally:
        montage.close()


def test_context_widens_each_clip_and_keeps_its_rank():
    from detail.api.clip_montage import MontageClip, with_context

    clip = MontageClip(1, "r", "r", 0, "e", 500_000_000, 2_000_000_000)
    (wide,) = with_context([clip], 1.0)
    assert (wide.start_t_ns, wide.end_t_ns, wide.rank) == (0, 3_000_000_000, 1)
    assert with_context([clip], 0) == [clip]
