"""One of the clip window's side panes — the masks alone, or the keypoints alone — over a span of
frames, at full resolution, written as an mp4 or a GIF (§22 *Enlarged panes*).

The clip window draws three pictures of one frame from one decode (``ClipQueryPreview.clip_views``):
the footage, the masks on white and the keypoints on white. Its side panes are a half-height
thumbnail of the last two; clicking one opens it large, and this is what that window writes. The
pictures are ``clip_views``'s own, at the footage's resolution, drawn with whatever the clip window
has ticked (:class:`ClipViewSettings`) — so the file is what was on screen, not a re-drawing of it.
Both are stored data (the detection run's masks, the keypoint run's points), so neither the export
nor the enlarged window decodes the video (``clip_views(footage=False)``).

The window's own **Export ▾** writes the whole of what it shows the same way
(:func:`export_clip_view`, :func:`write_clip_frame`): the footage alone, or the footage with
its two panes beside it (``clip_views.view_picture``), every tick and the orientation included.

**Which frames:** the ones the enlarged window's bar spans — the clip, plus the context either
side when the clip window's **Context** is on. An mp4 keeps every one at the footage's own rate; a
GIF samples real time at its chosen rate (``api.animation``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.api.animation import (
    GIF,
    GIF_DEFAULT_FPS,
    GIF_DEFAULT_WIDTH,
    format_of,
    frame_at_offset,
    sample_offsets_ns,
    write_animation,
)
from detail.core.video_codecs import DEFAULT_VIDEO_CODEC

MASKS = "masks"
KEYPOINTS = "keypoints"
PANES = (MASKS, KEYPOINTS)

Progress = Callable[[str, float, str], None] | None


@dataclass(frozen=True)
class ClipViewSettings:
    """The clip window's view ticks, as plain values: what ``clip_views`` draws with."""

    masks_on_frame: bool = True
    points_on_frame: bool = False
    trail: bool = False
    #: The keypoints that trail, by name; ``None`` is every point.
    trail_points: tuple[str, ...] | None = None
    mask_trail: bool = False
    pose_aligned: bool = False
    #: The window's **Aligned** orientation: the whole view turned by the body axis.
    aligned: bool = False

    def views_kwargs(self) -> dict:
        """The keyword arguments ``ClipQueryPreview.clip_views`` takes for these settings."""
        return {
            "masks_on_frame": self.masks_on_frame,
            "points_on_frame": self.points_on_frame,
            "trail": self.trail,
            "trail_points": (
                None if self.trail_points is None else list(self.trail_points)
            ),
            "mask_trail": self.mask_trail,
            "pose_aligned": self.pose_aligned,
            "aligned": self.aligned,
        }


def pane_picture(views, pane: str) -> np.ndarray | None:
    """``pane``'s picture from one ``ClipViews`` — ``None`` for keypoints on a run with none.

    Raises:
        ValueError: ``pane`` is not one of :data:`PANES`.
    """
    if pane == MASKS:
        return views.masks
    if pane == KEYPOINTS:
        return views.pose
    raise ValueError(f"pane must be one of {PANES}, not {pane!r}")


def pane_frames(
    t_ns: np.ndarray, first: int, last: int, fps: float | None
) -> list[int]:
    """The global frames an export of ``[first, last]`` writes: every one when ``fps`` is
    ``None``, else one per ``1/fps`` of real time (``t_ns`` is the recording's frame times).
    """
    first, last = int(first), int(max(last, first))
    if fps is None:
        return list(range(first, last + 1))
    span = np.asarray(t_ns[first : last + 1], dtype=np.int64)
    return [
        first + frame_at_offset(span, offset)
        for offset in sample_offsets_ns(int(span[-1] - span[0]), fps)
    ]


def export_clip_pane(
    experiment_dir: Path | str,
    *,
    recording_id: str,
    embed_run_id: str,
    track_id,
    first: int,
    last: int,
    pane: str,
    settings: ClipViewSettings,
    path: Path | str,
    max_width: int | None = None,
    fps: float | None = None,
    progress: Progress = None,
    preview=None,
    codec: str = DEFAULT_VIDEO_CODEC,
) -> Path:
    """Write ``pane`` over global frames ``[first, last]`` of one clip to ``path``.

    An mp4 keeps every frame at the recording's rate and full size unless ``max_width`` /
    ``fps`` say otherwise, encoded as ``codec`` (``core.video_codecs.VIDEO_CODECS``); a GIF defaults to :data:`GIF_DEFAULT_WIDTH` wide at
    :data:`GIF_DEFAULT_FPS`. ``preview`` (a ``ClipQueryPreview``) is reused when given.

    Raises:
        ValueError: an unknown pane or suffix, or keypoints asked of a run with none.
    """
    from detail.api.run_preview import ClipQueryPreview

    if pane not in PANES:
        raise ValueError(f"pane must be one of {PANES}, not {pane!r}")
    kind = format_of(path)
    if kind == GIF:
        fps = GIF_DEFAULT_FPS if fps is None else fps
        max_width = GIF_DEFAULT_WIDTH if max_width is None else max_width
    preview = preview or ClipQueryPreview(experiment_dir)
    if pane == KEYPOINTS:
        note = preview.clip_has_points(recording_id, embed_run_id)
        if note is not None:
            raise ValueError(f"no keypoints to export: {note}")
    t_ns = preview.timestamps(recording_id, embed_run_id)
    rate = (
        float(fps)
        if fps is not None
        else preview.frame_rate(recording_id, embed_run_id)
    )
    frames = pane_frames(t_ns, first, last, fps)

    def pictures():
        for frame in frames:
            views = preview.clip_views(
                recording_id,
                embed_run_id,
                track_id,
                frame,
                footage=False,
                **settings.views_kwargs(),
            )
            if views is None:
                continue  # an undecodable frame is skipped, as the player skips it
            picture = pane_picture(views, pane)
            if picture is not None:
                yield picture

    return write_animation(
        path,
        pictures(),
        fps=rate,
        count=len(frames),
        max_width=max_width,
        progress=progress,
        codec=codec,
    )


def pane_title(pane: str, label: str = "") -> str:
    """The enlarged window's title: ``Masks · <clip>``."""
    name = "Masks" if pane == MASKS else "Keypoints"
    return f"{name} · {label}" if label else name


def export_clip_view(
    experiment_dir: Path | str,
    *,
    recording_id: str,
    embed_run_id: str,
    track_id,
    first: int,
    last: int,
    settings: ClipViewSettings,
    panes: bool,
    path: Path | str,
    max_width: int | None = None,
    fps: float | None = None,
    progress: Progress = None,
    preview=None,
    codec: str = DEFAULT_VIDEO_CODEC,
) -> Path:
    """Write the clip window's view over global frames ``[first, last]`` — the clip and the
    context its bar spans — to ``path`` (``.mp4`` / ``.gif``), drawn with the window's
    ``settings`` (marks, trails, **Aligned**) and laid out as it is: the footage alone, or with
    its two panes (``panes``; ``clip_views.view_picture``).

    An mp4 keeps every frame at the recording's rate and full size unless ``max_width`` /
    ``fps`` say otherwise, encoded as ``codec``; a GIF defaults to :data:`GIF_DEFAULT_WIDTH`
    wide at :data:`GIF_DEFAULT_FPS`. Decodes the footage one frame at a time, in order.

    Raises:
        ValueError: an unknown suffix.
    """
    from detail.api.clip_views import view_picture
    from detail.api.run_preview import ClipQueryPreview

    kind = format_of(path)
    if kind == GIF:
        fps = GIF_DEFAULT_FPS if fps is None else fps
        max_width = GIF_DEFAULT_WIDTH if max_width is None else max_width
    preview = preview or ClipQueryPreview(experiment_dir)
    t_ns = preview.timestamps(recording_id, embed_run_id)
    rate = (
        float(fps)
        if fps is not None
        else preview.frame_rate(recording_id, embed_run_id)
    )
    frames = pane_frames(t_ns, first, last, fps)

    def pictures():
        for frame in frames:
            views = preview.clip_views(
                recording_id, embed_run_id, track_id, frame, **settings.views_kwargs()
            )
            if views is None:
                continue  # an undecodable frame is skipped, as the player skips it
            yield view_picture(views, panes=panes)

    return write_animation(
        path,
        pictures(),
        fps=rate,
        count=len(frames),
        max_width=max_width,
        progress=progress,
        codec=codec,
    )


def write_clip_frame(
    experiment_dir: Path | str,
    *,
    recording_id: str,
    embed_run_id: str,
    track_id,
    frame: int,
    settings: ClipViewSettings,
    panes: bool,
    path: Path | str,
    preview=None,
) -> Path:
    """Write global frame ``frame`` as the clip window shows it to ``path``, a PNG (the
    suffix added when missing).

    Raises:
        ValueError: the frame cannot be read.
    """
    from PIL import Image

    from detail.api.clip_views import view_picture
    from detail.api.run_preview import ClipQueryPreview

    path = Path(path)
    if path.suffix.lower() != ".png":
        path = path.with_name(f"{path.name}.png")
    preview = preview or ClipQueryPreview(experiment_dir)
    views = preview.clip_views(
        recording_id, embed_run_id, track_id, int(frame), **settings.views_kwargs()
    )
    if views is None:
        raise ValueError(f"frame {frame} of {recording_id} could not be read")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(view_picture(views, panes=panes), "RGB").save(path)
    return path
