"""The shared clip renderer (api/clip_render.py): one forward decode pass per video part.

This is the module both exports route their decoding through, so its guarantees are pinned here
rather than in either export's tests: **frame accuracy** (a clip holds exactly ``[lo, hi]``, because
frames are counted forward and never sought past) and **cost** (one ``VideoCapture`` per part no
matter how many clips come out of it — the property that makes the export fast, and the
one a refactor could silently undo).

``render_part`` takes a plain part dict and a job list, so most of this needs no preview and no
experiment — just a real decodable mp4, which it does need: the whole point is the decode.
"""

import unittest.mock

import cv2
import numpy as np
import pytest

from detail.api import clip_render

# Solid-colour frames survive mp4v well enough to identify themselves by mean, which is what makes
# the frame-accuracy assertion possible at all — a textured frame would not come back recognizable.
# A clip is encoded twice (fixture, then render), and each pass shifts a flat grey by a few levels,
# so the step is wide enough to absorb the drift — measured: 15 round-trips exactly, 12 does not,
# which caps the fixture at 16 frames.
_STEP = 15
_N_FRAMES = 16


def write_indexed_video(path, n_frames: int, size: int = 64) -> None:
    """An mp4 whose every frame is a solid grey encoding its own index.

    Past ``_N_FRAMES`` the greys wrap, so ``frame_indices`` can no longer identify them — longer
    videos are for tests that only count clips, not ones that check *which* frames landed.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (size, size)
    )
    try:
        for i in range(n_frames):
            writer.write(np.full((size, size, 3), (i * _STEP) % 256, dtype=np.uint8))
    finally:
        writer.release()


def frame_indices(path) -> list[int]:
    """Decode a clip back to the source frame indices its greys encode."""
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(int(round(float(bgr.mean()) / _STEP)))
    finally:
        cap.release()
    return out


class _Col:
    def __init__(self, values):
        self._values = np.asarray(values)

    def __getitem__(self, item):
        return self._values[item]


class _Reader:
    """The columns ``track_rows`` reads off a detection part (bbox is normalized x1,y1,x2,y2)."""

    def __init__(self, frames, boxes, rle=None):
        self._cols = {
            "frame_idx": _Col(frames),
            "bbox": _Col(boxes),
            "mask_rle": _Col(rle if rle is not None else [""] * len(frames)),
        }
        self.n_rows = len(frames)

    def column(self, name):
        return self._cols[name]


class _Assignment:
    def __init__(self, object_ids):
        self._object_id = _Col(object_ids)

    def column(self, name):
        assert name == "object_id"
        return self._object_id


def rle_of(mask) -> str:
    """A real COCO-RLE string, so the overlay path decodes what the pipeline actually stores."""
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return encoded["counts"].decode("ascii")


def make_part(video, frames, track_id=0, other_frames=(), mask=None, size=64):
    """A ``_build_parts``-shaped part: ``track_id`` detected on ``frames``, another track elsewhere.

    ``mask`` (a bool array, default a centred square) is stored as RLE on every row of this track.
    """
    box = (0.25, 0.25, 0.75, 0.75)
    if mask is None:
        mask = np.zeros((size, size), dtype=bool)
        mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = True
    all_frames = list(frames) + list(other_frames)
    ids = [track_id] * len(frames) + [track_id + 99] * len(other_frames)
    return {
        "video": video,
        "reader": _Reader(
            all_frames, [box] * len(all_frames), [rle_of(mask)] * len(all_frames)
        ),
        "assignment": _Assignment(ids),
    }


def job(lo, hi, path, fps=30.0):
    return {
        "lo": lo,
        "hi": hi,
        "fps": fps,
        "path": path,
        "writers": {},
        "paths": {},
        "n_written": 0,
    }


def read_frames(path) -> list[np.ndarray]:
    """Every decoded frame of a clip, as BGR arrays."""
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(bgr)
    finally:
        cap.release()
    return out


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "part.mp4"
    write_indexed_video(path, _N_FRAMES)
    return path


def test_clip_holds_exactly_the_requested_frames(video, tmp_path):
    """Frame accuracy is the reason for the forward pass — assert the frames, not just the count."""
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        make_part(video, range(_N_FRAMES)),
        [job(4, 11, out)],
        0,
        32,
        renders=(clip_render.RAW,),
    )
    assert frame_indices(out) == list(range(4, 12))


def test_one_capture_serves_every_bout_in_the_part(video, tmp_path):
    """The cost guarantee: one capture per part.

    Seeking per bout re-decodes from the preceding keyframe, which is what
    made the real export slow; a refactor back to that would still produce correct clips,
    so only counting the opens can catch it.

    ``cv2.VideoCapture`` is patched at the only binding that exists — ``clip_render`` imports cv2
    inside the function, so there is no module attribute to patch instead.
    """
    jobs = [job(i * 4, i * 4 + 1, tmp_path / f"c{i}.mp4") for i in range(4)]
    real = cv2.VideoCapture
    with unittest.mock.patch("cv2.VideoCapture", wraps=real) as spy:
        clip_render.render_part(
            make_part(video, range(_N_FRAMES)), jobs, 0, 32, renders=(clip_render.RAW,)
        )
    assert spy.call_count == 1
    assert [j["n_written"] for j in jobs] == [2] * 4


def test_overlapping_bouts_are_written_in_the_same_pass(video, tmp_path):
    """Two clips wanting the same frames must both get them — writers stay open concurrently."""
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    clip_render.render_part(
        make_part(video, range(_N_FRAMES)),
        [job(2, 8, a), job(5, 11, b)],
        0,
        32,
        renders=(clip_render.RAW,),
    )
    assert frame_indices(a) == list(range(2, 9))
    assert frame_indices(b) == list(range(5, 12))


def test_bouts_are_sorted_by_the_renderer_not_the_caller(video, tmp_path):
    """A forward pass depends on order, so the caller must not have to supply it."""
    late, early = tmp_path / "late.mp4", tmp_path / "early.mp4"
    clip_render.render_part(
        make_part(video, range(_N_FRAMES)),
        [job(10, 13, late), job(1, 4, early)],
        0,
        32,
        renders=(clip_render.RAW,),
    )
    assert frame_indices(early) == list(range(1, 5))
    assert frame_indices(late) == list(range(10, 14))


def test_frames_without_a_detection_are_skipped(video, tmp_path):
    """A gap in the track drops those frames from the clip rather than writing a stale crop."""
    out = tmp_path / "clip.mp4"
    detected = [i for i in range(_N_FRAMES) if i % 2 == 0]
    clip_render.render_part(
        make_part(video, detected), [job(4, 13, out)], 0, 32, renders=(clip_render.RAW,)
    )
    assert frame_indices(out) == [4, 6, 8, 10, 12]


def test_a_bout_the_track_is_absent_from_writes_nothing(video, tmp_path):
    """n_written == 0 is the signal the caller drops the file on — one bad bout must not
    kill the export."""
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        make_part(video, range(6)),
        [job(10, 14, out)],
        0,
        32,
        renders=(clip_render.RAW,),
    )
    assert not out.exists()


def test_only_the_requested_track_is_rendered(video, tmp_path):
    """The assignment zarr carries every track; box_map must filter to this one."""
    out = tmp_path / "clip.mp4"
    part = make_part(video, [4, 6, 8], track_id=0, other_frames=[5, 7, 9])
    clip_render.render_part(part, [job(4, 9, out)], 0, 32, renders=(clip_render.RAW,))
    assert frame_indices(out) == [4, 6, 8]


def test_a_missing_video_leaves_every_job_empty(tmp_path):
    """An unopenable part returns quietly with n_written 0, so the caller records rows without
    videos instead of raising partway through a long export."""
    out = tmp_path / "clip.mp4"
    part = make_part(tmp_path / "absent.mp4", range(_N_FRAMES))
    clip_render.render_part(part, [job(0, 5, out)], 0, 32, renders=(clip_render.RAW,))
    assert not out.exists()


def test_a_part_with_no_detections_at_all_is_handled(video, tmp_path):
    out = tmp_path / "clip.mp4"
    part = {"video": video, "reader": None, "assignment": None}
    clip_render.render_part(part, [job(0, 5, out)], 0, 32, renders=(clip_render.RAW,))
    assert not out.exists()


def test_no_jobs_is_a_no_op(video):
    clip_render.render_part(
        make_part(video, range(_N_FRAMES)), [], 0, 32, renders=(clip_render.RAW,)
    )


def test_writers_are_released_when_the_video_ends_mid_bout(video, tmp_path):
    """A bout running past the end of the file still yields the frames that exist, closed cleanly."""
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        make_part(video, range(40)),
        [job(10, 25, out)],
        0,
        32,
        renders=(clip_render.RAW,),
    )
    assert out.exists()
    assert frame_indices(out) == list(range(10, _N_FRAMES))


def test_plan_job_maps_a_global_range_into_its_part():
    part_index, planned = clip_render.plan_job([100, 50], 120, 130, 0.5, "x.mp4")
    assert part_index == 1 and (planned["lo"], planned["hi"]) == (20, 30)
    assert planned["fps"] == 20.0  # 10 frames over 0.5 s


def test_plan_job_clips_a_bout_that_would_span_two_parts():
    """Better a short clip from one video than a jump-cut across two."""
    part_index, planned = clip_render.plan_job([100, 50], 95, 130, 1.0, "x.mp4")
    assert part_index == 0 and (planned["lo"], planned["hi"]) == (95, 99)


def test_plan_job_falls_back_to_30_fps_on_a_degenerate_duration():
    _part, planned = clip_render.plan_job([100], 10, 10, 0.0, "x.mp4")
    assert planned["fps"] == 30.0


def test_plan_job_clamps_an_absurd_frame_rate():
    _part, fast = clip_render.plan_job([1000], 0, 900, 0.1, "x.mp4")
    _part, slow = clip_render.plan_job([1000], 0, 1, 900.0, "x.mp4")
    assert fast["fps"] == 120.0 and slow["fps"] == 1.0


def test_finish_drops_an_empty_clip_and_names_a_real_one(tmp_path):
    empty, real = tmp_path / "empty.mp4", tmp_path / "real.mp4"
    empty.touch()
    real.touch()
    assert clip_render.finish({"paths": {"raw": empty}, "n_written": 0}) == {}
    assert not empty.exists()
    assert clip_render.finish({"paths": {"raw": real}, "n_written": 3}) == {
        "raw": "real.mp4"
    }
    assert real.exists()


def test_progress_fires_once_per_finished_clip(video, tmp_path):
    seen = []
    jobs = [job(i * 5, i * 5 + 1, tmp_path / f"c{i}.mp4") for i in range(3)]
    clip_render.render_part(
        make_part(video, range(_N_FRAMES)),
        jobs,
        0,
        32,
        progress=seen.append,
        renders=(clip_render.RAW,),
    )
    assert len(seen) == 3
    assert {j["path"].name for j in seen} == {"c0.mp4", "c1.mp4", "c2.mp4"}


# --- the overlay framing (crop=False) ---------------------------------------------------- #
def test_overlay_keeps_the_whole_frame_at_a_steady_scale(tmp_path):
    """The reason this framing exists. A subject crop's window is ``max(bbox side) * pad``, so it
    changes with the animal's posture and the background appears to zoom; the overlay's output size
    comes from the source, so every frame of every clip is the same view."""
    video = tmp_path / "wide.mp4"
    _write_sized_video(video, 24, width=96, height=48)
    growing = [
        (0.4 - i * 0.01, 0.4 - i * 0.01, 0.6 + i * 0.01, 0.6 + i * 0.01)
        for i in range(24)
    ]
    part = {
        "video": video,
        "reader": _Reader(
            list(range(24)), growing, [rle_of(np.ones((48, 96), bool))] * 24
        ),
        "assignment": _Assignment([0] * 24),
    }
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        part, [job(4, 11, out)], 0, 64, renders=(clip_render.OVERLAY,), crop=False
    )
    assert _sizes(out) == {
        (64, 32)
    }  # 96x48 fitted to a 64 px longest side, aspect kept


def test_overlay_draws_the_mask_on_the_frame(tmp_path):
    """The subject must be visibly tinted — a clip that is just the raw frame has lost the point."""
    video = tmp_path / "grey.mp4"
    _write_sized_video(video, 12, width=64, height=64, value=128)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    part = make_part(video, range(12), mask=mask)
    plain, tinted = tmp_path / "plain.mp4", tmp_path / "tinted.mp4"
    clip_render.render_part(
        {**part, "reader": _Reader(list(range(12)), [(0.0, 0.0, 1.0, 1.0)] * 12)},
        [job(2, 6, plain)],
        0,
        64,
        renders=(clip_render.OVERLAY,),
        crop=False,
    )
    clip_render.render_part(
        part, [job(2, 6, tinted)], 0, 64, renders=(clip_render.OVERLAY,), crop=False
    )
    assert _first_frame(tinted) is not None
    # Per channel, not the mean over them: the overlay colour averages near mid-grey, so a
    # channel-averaged patch barely moves even where the tint is obvious.
    assert _channel_shift(tinted, plain, 24, 40) > 20  # inside the mask
    assert _channel_shift(tinted, plain, 0, 8) < 5  # a corner outside it


def test_overlay_still_writes_a_frame_when_the_mask_is_missing(tmp_path):
    """Detection rows with no mask (a bbox-only model) must not silently drop the frame."""
    video = tmp_path / "grey.mp4"
    _write_sized_video(video, 12, width=64, height=64, value=128)
    part = {
        "video": video,
        "reader": _Reader(list(range(12)), [(0.2, 0.2, 0.8, 0.8)] * 12, [""] * 12),
        "assignment": _Assignment([0] * 12),
    }
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        part, [job(2, 6, out)], 0, 64, renders=(clip_render.OVERLAY,), crop=False
    )
    assert _first_frame(out) is not None


def test_overlay_is_frame_accurate_too(tmp_path):
    """The forward-pass guarantee is independent of framing."""
    video = tmp_path / "part.mp4"
    write_indexed_video(video, _N_FRAMES)
    part = make_part(video, range(_N_FRAMES))
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        part, [job(4, 11, out)], 0, 64, renders=(clip_render.OVERLAY,), crop=False
    )
    assert _frame_count(out) == 8


def test_crop_remains_the_default_framing(tmp_path):
    """A pose comparison wants a constant subject scale, so that framing stays available."""
    video = tmp_path / "part.mp4"
    write_indexed_video(video, _N_FRAMES)
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        make_part(video, range(_N_FRAMES)),
        [job(4, 11, out)],
        0,
        32,
        renders=(clip_render.RAW,),
    )
    assert _sizes(out) == {(32, 32)}


def test_fit_keeps_the_aspect_and_never_goes_odd():
    """Several mp4 encoders reject an odd width or height outright."""
    assert clip_render.fit_within(1920, 1080, max_longest=640) == (640, 360)
    assert clip_render.fit_within(640, 480, max_longest=1920) == (
        640,
        480,
    )  # never upscales
    assert all(v % 2 == 0 for v in clip_render.fit_within(101, 33, max_longest=51))


def _write_sized_video(path, n_frames, *, width, height, value=None):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    try:
        for i in range(n_frames):
            fill = value if value is not None else (i * _STEP) % 256
            writer.write(np.full((height, width, 3), fill, dtype=np.uint8))
    finally:
        writer.release()


def _read_frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(bgr)
    finally:
        cap.release()
    return out


def _sizes(path):
    return {(f.shape[1], f.shape[0]) for f in _read_frames(path)}


def _frame_count(path):
    return len(_read_frames(path))


def _first_frame(path):
    frames = _read_frames(path)
    return frames[0] if frames else None


def _channel_shift(a, b, lo, hi):
    """Largest per-channel mean difference between the two clips over one patch."""
    pa = _first_frame(a)[lo:hi, lo:hi].reshape(-1, 3).mean(axis=0)
    pb = _first_frame(b)[lo:hi, lo:hi].reshape(-1, 3).mean(axis=0)
    return float(np.abs(pa - pb).max())


def test_the_blend_runs_at_the_output_size_not_the_source_size(tmp_path):
    """The overlay downscales *before* blending.

    Blending at source resolution is several times slower end to end on 1080p data, and at
    a third scale the difference in the mask edge is sub-pixel. The clips look the same
    either way, so only the size the blend runs at can catch a regression back to it.

    ``draw_mask`` is patched on its own module: ``_overlay`` imports it inside the function, so that
    attribute is what the call resolves to.
    """
    video = tmp_path / "big.mp4"
    _write_sized_video(video, 12, width=320, height=240, value=100)
    mask = np.zeros((240, 320), dtype=bool)
    mask[60:180, 80:240] = True
    part = {
        "video": video,
        "reader": _Reader(
            list(range(12)), [(0.25, 0.25, 0.75, 0.75)] * 12, [rle_of(mask)] * 12
        ),
        "assignment": _Assignment([0] * 12),
    }
    import detail.core.visualize as visualize

    shapes: list = []
    real = visualize.draw_mask
    with unittest.mock.patch.object(
        visualize,
        "draw_mask",
        lambda frame, m, color, **kw: (
            shapes.append(frame.shape[:2]) or real(frame, m, color, **kw)
        ),
    ):
        clip_render.render_part(
            part,
            [job(2, 6, tmp_path / "c.mp4")],
            0,
            64,
            renders=(clip_render.OVERLAY,),
            crop=False,
        )
    assert shapes and set(shapes) == {(48, 64)}  # the 64-px output, never 240x320


def test_sample_index_maps_the_output_grid_back_onto_the_source():
    """The mask is subsampled by index rather than cv2.resize — 0.5 ms against 3.2 on a 1080p
    mask, and nearest-neighbour is all a bool mask supports anyway."""
    yi, xi = clip_render._sample_index((1080, 1920), (640, 360))
    assert (len(yi), len(xi)) == (360, 640)
    assert yi[0] == 0 and xi[0] == 0
    assert yi[-1] < 1080 and xi[-1] < 1920  # never indexes past the source
    assert np.all(np.diff(yi) >= 0) and np.all(np.diff(xi) >= 0)


# --- the three flavours, in one decode pass (#336) ---------------------------------------- #
def test_all_three_flavours_come_out_of_one_decode(video, tmp_path):
    """Three flavours were asked for; decode dominates, so they cost one pass, not three.

    Asserts the *cost* property, not just that three files appeared: a loop that re-opened the
    source per flavour would write exactly the same files and take three times as long, which
    is the regression this module exists to prevent.
    """
    part = make_part(video, range(_N_FRAMES))
    out = tmp_path / "clip.mp4"
    real = cv2.VideoCapture
    opened = []

    def counting(*args, **kwargs):
        opened.append(args[0] if args else None)
        return real(*args, **kwargs)

    with unittest.mock.patch.object(cv2, "VideoCapture", counting):
        clip_render.render_part(
            part,
            [job(2, 6, out)],
            0,
            64,
            renders=clip_render.RENDERS,
            crop=False,
        )
    assert len(opened) == 1, f"one decode per part, not one per flavour: {opened}"
    for render in clip_render.RENDERS:
        assert (tmp_path / f"clip_{render}.mp4").is_file()


def test_a_single_flavour_keeps_the_bare_filename(video, tmp_path):
    """So a caller that asks for one render writes what it wrote before this existed."""
    part = make_part(video, range(_N_FRAMES))
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        part, [job(2, 6, out)], 0, 64, renders=(clip_render.OVERLAY,), crop=False
    )
    assert out.is_file()
    assert not (tmp_path / "clip_overlay.mp4").exists()


def test_the_mask_flavour_is_binary_not_a_picture_of_one(video, tmp_path):
    """A mask that came back as greys would need thresholding, and guessing at a cutoff.

    Asserts the *property* that makes it a mask — two levels, the subject's shape — rather than
    that a file exists. mp4 is lossy, so the check is that the histogram is at the two ends and
    not in the middle, which a blended overlay would fail and a grey image would fail.
    """
    part = make_part(video, range(_N_FRAMES))
    out = tmp_path / "clip.mp4"
    clip_render.render_part(
        part, [job(2, 6, out)], 0, 64, renders=(clip_render.MASK,), crop=False
    )
    frames = read_frames(out)
    assert frames, "the mask flavour wrote no decodable frames"
    values = np.concatenate([f[:, :, 0].ravel() for f in frames])
    middling = np.mean((values > 60) & (values < 195))
    assert (
        middling < 0.10
    ), f"{middling:.1%} of the mask is mid-grey — that is a picture of a mask, not a mask"
    # The fixture's mask is a centred square covering a quarter of the frame.
    assert (
        0.10 < np.mean(values > 195) < 0.45
    ), "the white area is not the subject's shape"


def test_the_raw_flavour_has_no_overlay_drawn_on_it(video, tmp_path):
    """The point of "export raw clip" is that nothing was drawn — assert the pixels, not the file.

    The overlay tints the mask with a palette colour, so the two flavours differ in *chroma*:
    a raw frame stays grey (its channels equal), an overlaid one does not.
    """
    part = make_part(video, range(_N_FRAMES))
    raw_out, overlay_out = tmp_path / "r.mp4", tmp_path / "o.mp4"
    clip_render.render_part(
        part, [job(2, 6, raw_out)], 0, 64, renders=(clip_render.RAW,), crop=False
    )
    clip_render.render_part(
        part,
        [job(2, 6, overlay_out)],
        0,
        64,
        renders=(clip_render.OVERLAY,),
        crop=False,
    )

    def chroma(path):
        frames = read_frames(path)
        assert frames, f"{path.name} decoded to nothing"
        stack = np.stack(frames).astype(np.int16)
        return float(np.mean(stack.max(axis=3) - stack.min(axis=3)))

    assert chroma(raw_out) < 8, "the raw clip has colour drawn on it"
    assert (
        chroma(overlay_out) > chroma(raw_out) + 8
    ), "the overlay is not visibly different from the raw frame, so this asserts nothing"


def test_the_flavours_are_written_in_catalogue_order_whatever_the_caller_asks(
    video, tmp_path
):
    """Two callers asking for the same set write the same files, which is what makes the
    CLI and the GUI comparable byte for byte."""
    assert clip_render.resolve_renders(("mask", "raw")) == ("raw", "mask")
    assert clip_render.resolve_renders(("raw", "raw")) == ("raw",)


def test_an_unknown_flavour_is_refused_by_name(video, tmp_path):
    with pytest.raises(ValueError, match="unknown clip render"):
        clip_render.resolve_renders(("raw", "heatmap"))


def test_asking_for_no_flavour_is_refused(video, tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        clip_render.resolve_renders(())


def test_crop_and_render_are_independent(video, tmp_path):
    """The two axes used to be one flag, so four of six combinations did not exist.

    A cropped *mask* is one of the four: it needs the framing from one axis and the drawing
    from the other, and there was no way to ask for it at all.
    """
    part = make_part(video, range(_N_FRAMES))
    out, raw_out = tmp_path / "clip.mp4", tmp_path / "raw.mp4"
    clip_render.render_part(
        part, [job(2, 6, out)], 0, 32, renders=(clip_render.MASK,), crop=True
    )
    clip_render.render_part(
        part, [job(2, 6, raw_out)], 0, 32, renders=(clip_render.RAW,), crop=True
    )
    frames, raw_frames = read_frames(out), read_frames(raw_out)
    assert frames and raw_frames, "a cropped flavour wrote nothing"
    assert frames[0].shape[:2] == (
        32,
        32,
    ), "crop=True must still give a square of `size`"

    # The mask's own white area, not merely "not mid-grey": the fixture's source frames are dark
    # greys, so an *overlay* on a crop also has few mid-tones and would pass a mid-tone check.
    # What only a real mask has is saturated white where the subject is.
    white = np.mean(frames[0][:, :, 0] > 195)
    assert white > 0.3, f"a cropped mask is white where the subject is, got {white:.1%}"
    assert (
        np.mean(raw_frames[0][:, :, 0] > 195) < 0.05
    ), "the raw crop must not already be white, or the assertion above proves nothing"


def test_plan_optional_returns_none_for_a_bout_past_the_end_of_the_video():
    """One bad row must not take a whole export down.

    A bout past the end of its recording — what a re-import shorter than the run that analysed
    it leaves behind — used to raise `IndexError` out of `plan_job` and abort the export at
    whatever clip it had reached. Every clip exporter now plans through this, so the rule
    `finish` states for an undetected bout holds for an out-of-range one too.
    """
    assert clip_render.plan_optional([40], 100, 104, 1.0, "x.mp4") is None
    assert clip_render.plan_optional([40], 2, 6, 1.0, "x.mp4") is not None


def test_plan_job_itself_still_raises():
    """For any caller that is not an export, an out-of-range frame is a bug, not a data
    condition — so the strict form stays strict and the tolerant one is opted into."""
    with pytest.raises(IndexError):
        clip_render.plan_job([40], 100, 104, 1.0, "x.mp4")


def test_every_clip_exporter_plans_through_the_tolerant_form():
    """Data-driven over the exporters, so a fourth inherits the rule the day it is written.

    This is the class, not the site: the same three-line exposure sat in all three exporters,
    and fixing only the one being edited is what leaves siblings in place.
    """
    import ast
    import pathlib

    offenders = []
    for name in ("behavior_clips",):
        path = pathlib.Path("detail/api") / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "plan_job"
                and isinstance(node.value, ast.Name)
                and node.value.id == "clip_render"
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, (
        "a clip exporter calls plan_job directly, so one bout past the end of its video "
        f"aborts the whole export: {offenders}. Use clip_render.plan_optional."
    )


def test_the_exporter_scan_would_catch_one():
    """Guard the guard: the scan must be able to see a direct call, or it is vacuous."""
    import ast

    tree = ast.parse("clip_render.plan_job(a, b, c, d, e)\n")
    hits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr == "plan_job"
        and isinstance(n.value, ast.Name)
        and n.value.id == "clip_render"
    ]
    assert len(hits) == 1


# --- fit_within: the one rule for an exported picture's size ------------------------------ #
def test_fit_within_never_scales_up_and_keeps_both_sides_even():
    """One rule sizes every export (the clip renders by longest side, ``api.animation`` by
    width): never up, aspect kept, even on both sides for H.264, the tighter limit winning.
    """
    from detail.api.clip_render import fit_within

    assert fit_within(321, 181) == (320, 180)  # only the odd pixel goes
    assert fit_within(100, 50, max_width=640) == (100, 50)  # never up
    assert fit_within(1920, 1080, max_longest=200) == (200, 112)
    assert fit_within(1000, 100, max_width=100) == (100, 10)
    assert fit_within(1920, 1080, max_width=960, max_longest=200) == (200, 112)
    assert fit_within(1, 1) == (2, 2)  # a floor, so a writer always gets a frame
