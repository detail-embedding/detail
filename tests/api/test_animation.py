"""The one writer behind the enlarged panes' and the clip montage's exports (``api.animation``).

What is pinned is what a person asked for, read back out of the file: a GIF's width and frame
rate (the dialog's two questions) and an mp4's rate and size. A size or rate that is accepted and
then silently not used is the failure an "it wrote a file" test cannot see.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from detail.api import animation
from detail.core.progress import OperationCancelled


def _frames(n=6, width=320, height=180):
    out = []
    for i in range(n):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, : (i + 1) * width // n] = (200, 40, 40)
        out.append(frame)
    return out


def test_a_gif_has_the_width_and_frame_rate_asked_for(tmp_path):
    path = animation.write_animation(
        tmp_path / "a.gif", iter(_frames()), fps=12, count=6, max_width=160
    )
    with Image.open(path) as gif:
        assert gif.size == (160, 90), "scaled to the width asked, aspect kept"
        assert gif.n_frames == 6
        # GIF stores hundredths of a second: 12 fps is 83 ms, stored as 80.
        assert gif.info["duration"] in (80, 90)


def test_a_narrower_picture_is_never_scaled_up(tmp_path):
    path = animation.write_animation(
        tmp_path / "a.gif", iter(_frames(width=100, height=50)), fps=5, count=6,
        max_width=640,
    )  # fmt: skip
    with Image.open(path) as gif:
        assert gif.size == (100, 50)
        assert gif.info["duration"] == 200


def test_an_mp4_keeps_its_rate_and_size(tmp_path):
    path = animation.write_animation(
        tmp_path / "a.mp4", iter(_frames(width=321, height=181)), fps=25, count=6
    )
    cap = cv2.VideoCapture(str(path))
    try:
        assert round(cap.get(cv2.CAP_PROP_FPS)) == 25
        # Even dimensions, which H.264 needs: the odd pixel is cropped, not scaled.
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 180
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == 6


def test_progress_moves_frame_by_frame(tmp_path):
    seen = []
    animation.write_animation(
        tmp_path / "a.gif",
        iter(_frames()),
        fps=10,
        count=6,
        progress=lambda stage, fraction, detail: seen.append(fraction),
    )
    inner = [f for f in seen if 0 < f < 1]
    assert len(inner) >= 4, f"the bar did not move between ends: {seen}"
    assert seen[-1] == 1.0


def test_a_cancelled_export_leaves_no_file(tmp_path):
    def progress(stage, fraction, detail):
        if fraction > 0.4:
            raise OperationCancelled(stage)

    with pytest.raises(OperationCancelled):
        animation.write_animation(
            tmp_path / "a.mp4", iter(_frames()), fps=10, count=6, progress=progress
        )
    assert not (tmp_path / "a.mp4").exists()


@pytest.mark.parametrize("name", ["a.mp4", "a.gif"])
def test_a_cancel_leaves_the_file_it_would_have_replaced(tmp_path, name):
    """The user agreed to replace ``a.gif`` with a finished one, not to lose it (#29): the new
    one is written beside it and moved over it only once complete."""
    path = animation.write_animation(tmp_path / name, iter(_frames()), fps=10, count=6)
    before = path.read_bytes()

    def progress(stage, fraction, detail):
        if fraction > 0.4:
            raise OperationCancelled(stage)

    with pytest.raises(OperationCancelled):
        animation.write_animation(
            path, iter(_frames(width=160)), fps=10, count=6, progress=progress
        )
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [name], "no partial left"


def test_an_unknown_suffix_is_refused(tmp_path):
    with pytest.raises(ValueError, match=".mp4 or .gif"):
        animation.write_animation(tmp_path / "a.avi", iter(_frames()), fps=10, count=6)


def test_nothing_to_write_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        animation.write_animation(tmp_path / "a.gif", iter([]), fps=10, count=0)
    assert not (tmp_path / "a.gif").exists()


def test_a_rate_samples_real_time():
    """A 12 fps GIF of a one-second span holds 13 pictures, the last at most a second in."""
    offsets = animation.sample_offsets_ns(1_000_000_000, 12)
    assert len(offsets) == 13 and offsets[0] == 0 and offsets[-1] <= 1_000_000_000
    assert animation.sample_offsets_ns(0, 12) == [0]
    with pytest.raises(ValueError):
        animation.sample_offsets_ns(10, 0)


# --- one palette per GIF ---------------------------------------------------------------- #
def _decoded(path):
    with Image.open(path) as gif:
        out = []
        for k in range(gif.n_frames):
            gif.seek(k)
            out.append(np.asarray(gif.convert("RGB")).astype(int))
    return out


def test_a_gif_chooses_its_palette_once_not_per_frame(tmp_path, monkeypatch):
    """A palette is chosen once per GIF; each frame is only mapped onto it.

    Pillow's per-frame median cut was nearly all of a montage GIF's export time.
    """
    chosen = []
    real = Image.Image.quantize

    def quantize(self, *a, **k):
        if k.get("palette") is None:
            chosen.append(self.size)
        return real(self, *a, **k)

    monkeypatch.setattr(Image.Image, "quantize", quantize)
    animation.write_animation(tmp_path / "a.gif", iter(_frames()), fps=12, count=6)
    assert len(chosen) == 1, f"a palette was chosen {len(chosen)} times"


def test_few_colours_are_kept_exactly(tmp_path):
    """Masks and keypoints on white are a handful of colours: 256 or fewer across the whole GIF
    come back exactly as drawn."""
    frames = _frames()
    frames[3][:20, :20] = (17, 201, 99)
    path = animation.write_animation(tmp_path / "a.gif", iter(frames), fps=10, count=6)
    for drawn, read in zip(frames, _decoded(path)):
        assert np.array_equal(read, drawn)


def test_footage_stays_within_the_quantization_tolerance(tmp_path):
    """Thousands of colours a frame, a scene with something moving through it (what a montage
    pass or a pane is): what the GIF reads back is the picture to within a few levels — not
    posterized, and no frame mapped onto colours it does not have.

    One palette for the whole GIF is a trade: a sequence whose colours change from frame to
    frame (a cut between two scenes) is the case a palette per frame does better, and the reason
    this asserts on one scene.
    """
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:120, 0:160]
    scene = np.stack([xx + 40, yy + 60, (xx + yy) // 2 + 30], axis=2)
    frames = []
    for i in range(5):
        frame = scene + rng.integers(-6, 7, scene.shape)
        frame[30:60, 20 + 20 * i : 50 + 20 * i] = (230, 160, 20)  # the animal, walking
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    assert len(np.unique(frames[0].reshape(-1, 3), axis=0)) > 2000
    path = animation.write_animation(tmp_path / "a.gif", iter(frames), fps=10, count=5)
    for drawn, read in zip(frames, _decoded(path)):
        error = np.abs(read - drawn.astype(int))
        assert error.mean() < 4.0 and np.percentile(error, 99.9) < 24, error.mean()
