"""Write a sequence of RGB pictures as an mp4 or a GIF: the one writer behind the clip
window's enlarged-pane exports and the cluster montage's (§22 *Clip montage*).

**Two formats, one call.** An mp4 is the file to keep or to step through frame by frame; a GIF is
the file to drop into a slide or a message, where a video player is not on offer. They differ in
what is worth asking: an mp4 keeps every frame at the size it was drawn (H.264 through cv2, the
``storage.video.open_video_writer`` every clip export uses), while a GIF is 256 colours a frame and
grows with every pixel, so it is written at a width and a frame rate the person chooses —
:data:`GIF_DEFAULT_WIDTH` and :data:`GIF_DEFAULT_FPS` unless they say otherwise. Pillow writes it;
nothing here shells out to an FFmpeg executable (Tenet 15).

Frames are pulled one at a time from an iterator and reported through ``progress(stage, fraction,
detail)`` as they are written, so an export of a long clip shows a bar that moves.

**One palette per GIF** (:func:`gif_palette`), chosen once from the frames themselves rather than
once per frame. Pillow's per-frame median cut is slow on footage and slower the larger
the frame; one palette each made a folder of per-clip GIFs several times faster. Read
back against the RGB frames of a montage GIF, an octree over the whole pass leaves fewer
badly-off pixels, and with every frame on one palette a pixel that did not change
encodes as unchanged, which made the file several times smaller.
Side by side the two cannot be
told apart. Frames with 256 colours or fewer between them get those colours exactly. The trade is a
sequence whose colours change from frame to frame (a cut between scenes), where a palette per frame
fits better; a montage pass or a pane is one scene. A GIF holds its frames until the palette is
chosen (it held them all before too, as palette images, since Pillow writes a GIF in one call);
past :data:`GIF_PALETTE_BYTES` of frames the palette is chosen from those, and the rest are mapped
as they come.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from detail.core import atomic

#: What the GIF dialog opens with: the user's answer of 2026-10-22 ("ask each time", 640 / 12).
GIF_DEFAULT_WIDTH = 640
GIF_DEFAULT_FPS = 12.0
#: The bounds the GIF dialog offers, so a typo cannot ask for a 40,000-pixel animation.
GIF_WIDTH_RANGE = (64, 3840)
GIF_FPS_RANGE = (1.0, 50.0)

#: The most RGB bytes a GIF holds back to choose its palette from (see the module docstring): a
#: 640-pixel montage GIF's whole pass is ~53 MB; a pass larger than this takes its palette from
#: its first frames.
GIF_PALETTE_BYTES = 256 * 1024 * 1024
#: The most pixels the palette is chosen from; more frames are sampled on a sparser grid.
GIF_PALETTE_PIXELS = 16_000_000

MP4 = "mp4"
GIF = "gif"
FORMATS = (MP4, GIF)

Progress = Callable[[str, float, str], None] | None


def format_of(path: Path | str) -> str:
    """``"mp4"`` or ``"gif"`` from a path's suffix.

    Raises:
        ValueError: any other suffix.
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix not in FORMATS:
        raise ValueError(
            f"{Path(path).name}: an animation is written as .mp4 or .gif, not {suffix!r}"
        )
    return suffix


def sample_offsets_ns(duration_ns: int, fps: float) -> list[int]:
    """Offsets from a span's start at ``fps``, the first at 0 and none past ``duration_ns``.

    What a GIF at a chosen rate samples: one picture per ``1/fps`` of real time, so a 12 fps GIF
    of 30 fps footage keeps real time by showing every 2.5th frame rather than playing slower.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    period = 1e9 / float(fps)
    # Integer-nanosecond frame times fall a nanosecond short of a rational period (five frames
    # at 30 fps span 133,333,332 ns, not 2 x 66,666,666.7), and the last sample must not be
    # lost to that: a millionth of a period of slack.
    count = int(np.floor(max(int(duration_ns), 0) / period + 1e-6)) + 1
    return [int(round(i * period)) for i in range(count)]


def _fit_frame(rgb: np.ndarray, max_width: int | None) -> np.ndarray:
    """``rgb`` at the size :func:`detail.api.clip_render.fit_within` gives it — scaled down to
    ``max_width`` (aspect kept, never up) and even on both sides, which H.264 needs. When only
    an odd pixel is in the way the frame is cropped rather than resampled: invisible, lossless,
    and the same picture the writer got before one rule sized every export.
    """
    from detail.api.clip_render import fit_within

    height, width = rgb.shape[:2]
    target_w, target_h = fit_within(width, height, max_width=max_width)
    if (target_w, target_h) == (width, height):
        return rgb
    if width - target_w <= 1 and height - target_h <= 1:
        return np.ascontiguousarray(rgb[:target_h, :target_w])
    import cv2

    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def write_animation(
    path: Path | str,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
    count: int,
    max_width: int | None = None,
    progress: Progress = None,
) -> Path:
    """Write ``frames`` (RGB ``uint8``, all one size) to ``path`` as its suffix says.

    ``count`` is how many ``frames`` will yield, for the bar. ``max_width`` scales every frame
    down to that width first. The frames go to a hidden ``.partial`` sibling that is moved over
    ``path`` only once it is complete: a failed or cancelled export leaves nothing that looks
    finished, and leaves a file already at ``path`` exactly as it was. Writing in place and
    deleting ``path`` on a cancel deleted the earlier good copy the user had agreed to replace
    only with a finished one (#29).

    Raises:
        ValueError: an unknown suffix, a non-positive ``fps``, or no frames at all.
    """
    path = Path(path)
    kind = format_of(path)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    path.parent.mkdir(parents=True, exist_ok=True)

    def report(done: int, detail: str | None = None) -> None:
        if progress is not None:
            progress(
                "writing",
                min(done / max(count, 1), 1.0),
                detail or f"frame {done:,} of {count:,}",
            )

    report(0)
    partial = partial_path(path)
    try:
        if kind == GIF:
            written = _write_gif(partial, frames, fps, max_width, report)
        else:
            written = _write_mp4(partial, frames, fps, max_width, report)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if not written:
        partial.unlink(missing_ok=True)
        raise ValueError("nothing to write: the sequence had no frames")
    atomic.replace(partial, path)
    return path


def partial_path(path: Path) -> Path:
    """Where ``path`` is written until it is finished: hidden, same folder, same suffix (the
    suffix is what picks the encoder)."""
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def _write_mp4(path, frames, fps, max_width, report) -> int:
    from detail.core.visualize import rgb_to_bgr
    from detail.storage.video import open_video_writer

    writer = None
    written = 0
    try:
        for rgb in frames:
            rgb = _fit_frame(np.ascontiguousarray(rgb, dtype=np.uint8), max_width)
            if writer is None:
                writer, _codec = open_video_writer(
                    path, fps, int(rgb.shape[1]), int(rgb.shape[0])
                )
            writer.write(rgb_to_bgr(rgb))
            written += 1
            report(written)
    finally:
        if writer is not None:
            writer.release()
    return written


def gif_palette(frames: list[np.ndarray]):
    """One 256-colour palette (a Pillow ``P`` image) for ``frames`` (RGB ``uint8``, one size).

    Their colours exactly when there are 256 or fewer of them between them; otherwise an octree
    over every frame, sampled on a sparser grid past :data:`GIF_PALETTE_PIXELS`. Not dithered:
    a montage tile is small, and dither noise reads as fur that is not there.
    """
    from PIL import Image

    pixels = sum(f.shape[0] * f.shape[1] for f in frames)
    step = max(1, int(np.ceil(np.sqrt(pixels / GIF_PALETTE_PIXELS))))
    sample = Image.fromarray(
        np.ascontiguousarray(
            np.concatenate([f[::step, ::step] for f in frames], axis=0)
        ),
        "RGB",
    )
    if sample.getcolors(256) is not None:
        # Median cut gives every colour a box of its own when there are no more than 256.
        method = Image.Quantize.MEDIANCUT
    else:
        method = Image.Quantize.FASTOCTREE
    return sample.quantize(colors=256, method=method, dither=Image.Dither.NONE)


def _write_gif(path, frames, fps, max_width, report) -> int:
    from PIL import Image

    held: list[np.ndarray] = []  # frames waiting for the palette
    held_bytes = 0
    palette = None
    pictures = []

    def on_palette(rgb: np.ndarray):
        return Image.fromarray(rgb, "RGB").quantize(
            palette=palette, dither=Image.Dither.NONE
        )

    for rgb in frames:
        rgb = _fit_frame(np.ascontiguousarray(rgb, dtype=np.uint8), max_width)
        if palette is None:
            held.append(rgb)
            held_bytes += rgb.nbytes
            if held_bytes >= GIF_PALETTE_BYTES:
                palette = gif_palette(held)
                pictures += [on_palette(f) for f in held]
                held = []
        else:
            pictures.append(on_palette(rgb))
        report(len(pictures) + len(held))
    if pictures or held:
        # The frames are all in; choosing the palette and compressing are what is left.
        report(len(pictures) + len(held), "choosing colours and saving the GIF")
    if held:
        palette = gif_palette(held)
        pictures += [on_palette(f) for f in held]
    if not pictures:
        return 0
    # GIF stores a delay in hundredths of a second and Pillow *truncates* the milliseconds it is
    # given, so 15 fps (66.7 ms) would be written as 60 ms and play 11% fast. Round to the
    # nearest hundredth here instead.
    pictures[0].save(
        path,
        save_all=True,
        append_images=pictures[1:],
        duration=max(int(round(100.0 / float(fps))), 1) * 10,
        loop=0,
        disposal=1,
    )
    return len(pictures)
