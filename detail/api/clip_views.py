"""The clip window's pictures (§22): one frame of a clip as the footage, the masks on white and
the keypoints on white, drawn from **one** read.

**Reading and drawing are separate here, on purpose.** :class:`ClipFrame` is everything one frame's
pictures are made of — the decoded footage (or none), the frame's detections with their masks, the
keypoints placed on it, the trails read for it — and ``ClipQueryPreview.clip_frame`` is what fills
one. The three drawers below are pure functions over a `ClipFrame` and a `ClipViewSettings`: no
I/O, no caches, no Qt. So a surface that wants a different picture (the montage's cropped tile, a
pane export, a figure) draws it from the same read rather than asking for another, and the way a
mask is tinted or a trail fades is written once.

**One settings object.** The clip window's ticks are `ClipViewSettings`; the montage builds the
same object from its layers. The port this replaced took eight booleans, two of which were the
same expression at the only call site that set both.

**No footage is ``rgb=None``, not a white canvas.** A caller that shows only the panes (the
enlarged pane, a Raw-off montage tile) reads no video; the frame it did not read is absent, and
`draw_footage` raises rather than tinting a blank — a surface that paints the footage by mistake
sees an error, not a white picture that looks like an overexposed frame.

The mask tint on the footage is `visualize.draw_mask`'s default alpha — the one the node cards
and the clip renders use — so a clip looks the same wherever it is shown.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import threading

import cv2
import numpy as np

from detail.analysis import body_axis
from detail.analysis.body_axis import BodyAxis
from detail.api.animation import (
    GIF,
    GIF_DEFAULT_FPS,
    GIF_DEFAULT_WIDTH,
    format_of,
    sample_offsets_ns,
    write_animation,
)
from detail.core import visualize
from detail.core.dataclasses import Detection
from detail.core.errors import UserInputError
from detail.core.lru import CountLRU
from detail.core.skeleton import Skeleton
from detail.core.visualize import MASK_TINT, draw_keypoints, draw_mask

#: How far back the clip window's trails reach: the point trails, the box-centre trail and the
#: masks pane's onion skin all show this much of the animal's past.
CLIP_TRAIL_S = 0.5
#: The masks pane: an animal that is not the clip's subject, on white.
OTHER_ANIMAL = (150, 150, 150)
#: The pose pane draws the skeleton's edges dark: ``draw_keypoints``'s default is near-white.
POSE_EDGE = (110, 110, 110)
#: The trail drawn without keypoints: the animal's centre, in white.
CENTRE_TRAIL = (255, 255, 255)
#: The share of the pose pane's height the animal's typical body length fills, body-aligned.
ALIGNED_LENGTH = 0.4
#: How far an integer-nanosecond sample time may fall short of the frame it means — a
#: microsecond, three orders of magnitude under any frame period.
SAMPLE_SLACK_NS = 1_000

MASKS = "masks"
KEYPOINTS = "keypoints"
PANES = (MASKS, KEYPOINTS)

Progress = Callable[[str, float, str], None] | None


# --- what a picture is made of ----------------------------------------------------------- #
@dataclass(frozen=True)
class ClipViewSettings:
    """The clip window's view ticks, as plain values — the one object every picture is drawn
    with. ``trail_points`` names the keypoints that trail; ``None`` is every point."""

    masks_on_frame: bool = True
    points_on_frame: bool = False
    trail: bool = False
    trail_points: tuple[str, ...] | None = None
    mask_trail: bool = False
    pose_aligned: bool = False
    #: **Aligned** orientation: every picture — the footage with what is drawn on it, the masks
    #: pane, the pose pane — turned about the animal's middle so its body points up, frame by
    #: frame (:func:`turn_matrix`); a side-view (``mirror``) run's pictures are flipped where
    #: its embedding flipped them instead (:func:`aligned_matrix`). A frame with no body axis
    #: is drawn as recorded and marked
    #: (:func:`mark_no_axis`). ``pose_aligned`` is the pose pane's own, older alignment (scaled
    #: and centred too); with it on, the pose pane is not turned a second time.
    aligned: bool = False

    @property
    def trail_s(self) -> float:
        """How far back a reader looks for this picture: nothing without a trail ticked."""
        return CLIP_TRAIL_S if (self.trail or self.mask_trail) else 0.0


@dataclass(frozen=True)
class PointTrail:
    """Where the subject's keypoints have just been: ``xy`` ``(n, K, 2)`` normalized, oldest
    first, ``scores`` ``(n, K)``, a colour per point by its role, and which points were asked
    for (``kept``, ``(K,)`` bool)."""

    xy: np.ndarray
    scores: np.ndarray
    colours: tuple[tuple[int, int, int], ...]
    kept: np.ndarray


@dataclass(frozen=True)
class ClipFrame:
    """One frame of a clip, read: everything its pictures are drawn from and nothing drawn.

    ``rgb`` is the decoded footage, or ``None`` when the caller asked for none; ``frame_hw`` is
    the picture size either way. ``detections`` carry their masks decoded at that size.
    ``actual`` is the part-local frame the read landed on, ``shown`` the same frame
    recording-global, ``t_ns`` its time. ``placed`` is each subject's keypoints on this frame;
    ``skeleton`` is theirs, or ``None`` with ``pose_note`` saying why the run has none. ``axis``
    is the subject's body axis for aligning (``None`` with ``align_note``). The trails are read
    only when the settings asked for them: ``point_trail`` with keypoints, ``centre_trail``
    (``(n, 2)`` normalized, oldest first, a NaN row across a gap) without, ``mask_history`` the
    subject's past masks for the onion skin.
    """

    rgb: np.ndarray | None
    frame_hw: tuple[int, int]
    detections: list[Detection]
    actual: int
    shown: int
    t_ns: int
    placed: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    skeleton: Skeleton | None = None
    pose_note: str | None = None
    axis: BodyAxis | None = None
    align_note: str | None = None
    point_trail: PointTrail | None = None
    centre_trail: np.ndarray | None = None
    mask_history: list | None = None
    #: How a **side-view** embedding (``align: mirror``) drew this frame: ``1.0`` flipped,
    #: ``0.0`` not, NaN where it has no heading here; ``None`` for any other run, whose
    #: Aligned picture is turned (:func:`aligned_matrix`).
    flip: float | None = None

    def subject(self, track_id: int) -> Detection | None:
        """The clip's animal's detection on this frame, or ``None`` when it is not here."""
        return next((d for d in self.detections if _is_subject(d, track_id)), None)

    def subject_mask(self, track_id: int) -> np.ndarray | None:
        found = self.subject(track_id)
        return None if found is None or found.mask is None else found.mask


@dataclass(frozen=True)
class ClipViews:
    """One frame as the clip window shows it: the footage (``None`` when none was read), the
    masks on white, and the keypoints on white — ``pose`` ``None`` with ``pose_note`` when the
    run has none."""

    frame: np.ndarray | None
    masks: np.ndarray
    pose: np.ndarray | None
    pose_note: str | None
    actual: int
    t_ns: int
    align_note: str | None = None
    skeleton: Skeleton | None = None
    trail_note: str | None = None


def _is_subject(det: Detection, track_id: int) -> bool:
    return det.object_id is not None and int(det.object_id) == int(track_id)


# --- the drawers: pure over a ClipFrame -------------------------------------------------- #
def draw_footage(
    frame: ClipFrame, settings: ClipViewSettings, track_id: int
) -> np.ndarray:
    """The footage with the subject marked as ``settings`` say: its mask trail under its mask
    (the onion skin goes wherever the mask is drawn), its mask in the tint, its keypoints'
    trail, its keypoints. Returns a new frame.

    Raises:
        ValueError: no footage was read for this frame (``clip_frame(footage=False)``).
    """
    if frame.rgb is None:
        raise ValueError(
            "no footage was read for this frame (clip_frame(footage=False)); draw the masks "
            "or the pose pane, or read the frame with footage"
        )
    out = frame.rgb
    if settings.masks_on_frame:
        if settings.mask_trail and frame.mask_history:
            out = visualize.draw_onion_skin(out, frame.mask_history, MASK_TINT)
        mask = frame.subject_mask(track_id)
        if mask is not None:
            out = draw_mask(out, mask, MASK_TINT)
    if settings.trail:
        if frame.point_trail is not None:
            out = _draw_point_trail(out, frame.point_trail)
        elif frame.centre_trail is not None:
            out = visualize.draw_fading_paths(
                out, frame.centre_trail[:, None, :], [CENTRE_TRAIL]
            )
    if settings.points_on_frame and frame.skeleton is not None:
        own = frame.placed.get(int(track_id))
        if own is not None:
            out = draw_keypoints(out, own[0], frame.skeleton, scores=own[1])
    return out if out is not frame.rgb else out.copy()


def draw_masks_pane(
    frame: ClipFrame, settings: ClipViewSettings, track_id: int
) -> np.ndarray:
    """Every tracked animal's mask alone on white, the subject in the tint and the others grey
    (so a second animal reads as one with the footage gone), the subject's onion skin under
    them when ``settings.mask_trail``. Returns a new picture."""
    canvas = np.full((*frame.frame_hw, 3), 255, dtype=np.uint8)
    if settings.mask_trail and frame.mask_history:
        canvas = visualize.draw_onion_skin(canvas, frame.mask_history, MASK_TINT)
    for det in sorted(frame.detections, key=lambda d: _is_subject(d, track_id)):
        if det.mask is not None:
            colour = MASK_TINT if _is_subject(det, track_id) else OTHER_ANIMAL
            canvas = draw_mask(canvas, det.mask, colour, alpha=1.0)
    return canvas


def draw_pose_pane(
    frame: ClipFrame, settings: ClipViewSettings, track_id: int
) -> np.ndarray | None:
    """Every subject's keypoints alone on white, the clip's animal in its role colours and the
    others greyed; body-aligned (the animal alone, axis up, its middle at the centre) when
    ``settings.pose_aligned`` and the frame has an axis. ``None`` when the run has no
    keypoints. Returns a new picture."""
    if frame.skeleton is None:
        return None
    pose = np.full((*frame.frame_hw, 3), 255, dtype=np.uint8)
    aligned = settings.pose_aligned and frame.axis is not None
    if settings.trail and frame.point_trail is not None:
        trail = frame.point_trail
        if aligned:
            trail = PointTrail(
                aligned_window(trail.xy, trail.scores, frame.axis, _wh(frame)),
                trail.scores,
                trail.colours,
                trail.kept,
            )
        pose = _draw_point_trail(pose, trail)
    placed = dict(frame.placed)
    if aligned:
        pose = draw_aligned_pose(
            pose, frame.axis, placed.get(int(track_id)), frame.skeleton
        )
        placed = {}
    for subject_id in sorted(placed, key=lambda t: t == int(track_id)):
        xy, scores = placed[subject_id]
        own = subject_id == int(track_id)
        pose = draw_keypoints(
            pose,
            xy,
            frame.skeleton,
            scores=scores if own else np.zeros_like(scores),
            edge_color=POSE_EDGE,
        )
    return pose


def trail_note(frame: ClipFrame, settings: ClipViewSettings) -> str | None:
    """What ``settings.trail`` draws for this clip, or why it draws nothing — for the tick's
    tooltip. ``None`` when no trail is asked for."""
    if not settings.trail:
        return None
    if frame.point_trail is not None:
        return (
            "Each ticked keypoint's path over the last half second, in its colour, on the "
            "footage and the Keypoints pane."
        )
    if frame.centre_trail is not None:
        return (
            "This run has no keypoints, so the trail is the centre of the animal over the "
            "last half second, on the footage."
        )
    return (
        "This run has no keypoints and no kinematics, so there is no trail to draw: run "
        "`detail recording backfill-kinematics` for a centre trail."
    )


def _draw_point_trail(picture: np.ndarray, trail: PointTrail) -> np.ndarray:
    kept = np.asarray(trail.kept, dtype=bool)
    if not kept.any():
        return picture
    colours = [c for c, k in zip(trail.colours, kept) if k]
    return visualize.draw_fading_paths(picture, trail.xy[:, kept], colours)


def _wh(frame: ClipFrame) -> tuple[int, int]:
    return int(frame.frame_hw[1]), int(frame.frame_hw[0])


def aligned_window(
    xy: np.ndarray, scores: np.ndarray, axis: BodyAxis, canvas_wh: tuple[int, int]
) -> np.ndarray:
    """``xy`` (``(n, K, 2)`` normalized) with each frame turned by **its own** body axis, the
    way :func:`draw_aligned_pose` turns the frame shown — NaN where a frame has no axis.

    So a trail on the body-aligned pose is the path each point took *relative to the body*:
    a paw stepping, a head turning, not the animal crossing the arena (which the footage's
    trail shows, and which aligning is there to remove)."""
    wh = axis.frame_wh if min(axis.frame_wh) > 0 else canvas_wh
    out = np.full_like(xy, np.nan)
    if axis.typical_length_px <= 0:
        return out
    front_xy, back_xy = body_axis.row_axis_ends(
        xy, scores, list(axis.front), list(axis.back), wh, axis.cutoff
    )
    scale = ALIGNED_LENGTH * wh[1] / axis.typical_length_px
    for i in range(len(xy)):
        if np.isfinite(front_xy[i]).all() and np.isfinite(back_xy[i]).all():
            out[i] = body_axis.aligned_points(xy[i], front_xy[i], back_xy[i], wh, scale)
    return out


def draw_aligned_pose(
    canvas: np.ndarray,
    axis: BodyAxis,
    subject: tuple[np.ndarray, np.ndarray] | None,
    skeleton: Skeleton,
) -> np.ndarray:
    """The clip's animal on ``canvas``, its body axis up; a line of text instead when this
    frame gives no axis. ``subject`` is its ``(points, scores)`` on the frame, or ``None``.
    """
    h, w = canvas.shape[:2]
    # Measured in the recording's pixels, as the typical length was; drawn normalized.
    wh = axis.frame_wh if min(axis.frame_wh) > 0 else (w, h)
    if subject is not None and axis.typical_length_px > 0:
        xy, scores = subject
        front_xy, back_xy = body_axis.row_axis_ends(
            xy[None], scores[None], list(axis.front), list(axis.back), wh, axis.cutoff
        )
        if np.isfinite(front_xy).all() and np.isfinite(back_xy).all():
            turned = body_axis.aligned_points(
                xy,
                front_xy[0],
                back_xy[0],
                wh,
                ALIGNED_LENGTH * wh[1] / axis.typical_length_px,
            )
            return draw_keypoints(
                canvas, turned, skeleton, scores=scores, edge_color=POSE_EDGE
            )
    out = canvas.copy()
    cv2.putText(
        out,
        NO_AXIS_LINE,
        (8, max(h // 2, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(h / 600.0, 0.4),
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    return out


# --- the aligned orientation: one turn, every picture --------------------------------------- #
#: The pose pane's line on a frame with no axis.
NO_AXIS_LINE = "no body axis on this frame"
#: The mark on a picture drawn as recorded because its frame gives no body axis.
NO_AXIS_MARK = "no axis"


def axis_ends(
    points: np.ndarray, scores: np.ndarray, axis: BodyAxis, picture_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(front, back)`` — the two ends of the body axis, in ``picture_wh``'s pixels, from one
    frame's normalized ``points`` ``(K, 2)`` and ``scores`` ``(K,)``; ``None`` when either end
    has no usable point (the heading definition's sets, at the run's cutoff).

    In the picture's own pixels, so the angle is right whatever size the picture is drawn at:
    the full frame, a 480-pixel preview proxy, a tile's crop."""
    front, back = body_axis.row_axis_ends(
        np.asarray(points, dtype=np.float64)[None],
        np.asarray(scores, dtype=np.float64)[None],
        list(axis.front),
        list(axis.back),
        (int(picture_wh[0]), int(picture_wh[1])),
        axis.cutoff,
    )
    front, back = front[0], back[0]
    if not (np.isfinite(front).all() and np.isfinite(back).all()):
        return None
    if np.allclose(front, back):
        return None  # no length, so no direction
    return front, back


def turn_matrix(
    front: np.ndarray, back: np.ndarray, side: float, out_wh: tuple[int, int]
) -> np.ndarray:
    """The ``2 × 3`` affine (``cv2.warpAffine``'s forward map) that turns a picture about the
    body's middle so the axis from ``back`` to ``front`` points straight up — the direction the
    clip window's **Align pose** faces — puts that middle at the centre of an ``out_wh``
    picture, and scales a ``side``-pixel span to the output's height.

    The montage passes its crop's side (a tile shows as much around the animal as it does as
    recorded); the clip window passes the frame's height, which turns without scaling.
    """
    front = np.asarray(front, dtype=np.float64)
    back = np.asarray(back, dtype=np.float64)
    centre = (front + back) / 2.0
    dx, dy = front - back
    angle = -np.pi / 2 - np.arctan2(
        dy, dx
    )  # onto (0, -1): screen up, image y points down
    scale = float(out_wh[1]) / max(float(side), 1e-6)
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    linear = np.array([[c, -s], [s, c]])
    shift = np.array([out_wh[0] / 2.0, out_wh[1] / 2.0]) - linear @ centre
    return np.hstack([linear, shift[:, None]])


def mirror_matrix(
    front: np.ndarray,
    back: np.ndarray,
    side: float,
    out_wh: tuple[int, int],
    flipped: bool,
) -> np.ndarray:
    """The ``2 × 3`` affine of a **side-view** run's Aligned picture: no turn, flipped left to
    right about the body's middle where ``flipped`` (the embedding's own flip, §14 ``mirror``),
    that middle at the centre of an ``out_wh`` picture, a ``side``-pixel span scaled to the
    output's height — :func:`turn_matrix`'s placement without its rotation."""
    front = np.asarray(front, dtype=np.float64)
    back = np.asarray(back, dtype=np.float64)
    centre = (front + back) / 2.0
    scale = float(out_wh[1]) / max(float(side), 1e-6)
    linear = np.array([[-scale if flipped else scale, 0.0], [0.0, scale]])
    shift = np.array([out_wh[0] / 2.0, out_wh[1] / 2.0]) - linear @ centre
    return np.hstack([linear, shift[:, None]])


def aligned_matrix(
    front: np.ndarray,
    back: np.ndarray,
    side: float,
    out_wh: tuple[int, int],
    flip: float | None,
) -> np.ndarray | None:
    """The **Aligned** picture's affine for one frame, the way the clip's embedding aligned:
    :func:`turn_matrix` (body up) for an overhead or unaligned run (``flip`` ``None``), and
    :func:`mirror_matrix` for a side-view one (``flip`` the embedding's flag), so a side view
    stays a side view, as the embedding saw it. ``None`` where a side-view run has no heading
    on this frame (``flip`` NaN): the frame is drawn as recorded and marked, never guessed.

    Every surface that draws a clip Aligned (the clip window, the live tiles, the exports)
    takes its matrix from here, so none of them turns a side view upright."""
    if flip is None:
        return turn_matrix(front, back, side, out_wh)
    if not np.isfinite(flip):
        return None
    return mirror_matrix(front, back, side, out_wh, bool(flip))


def turn_picture(
    picture: np.ndarray, matrix: np.ndarray, out_wh: tuple[int, int], fill
) -> np.ndarray:
    """``picture`` through ``matrix`` into an ``out_wh`` picture; ``fill`` where the turned
    frame does not reach (black for footage, white for the panes)."""
    return cv2.warpAffine(
        picture,
        matrix,
        (int(out_wh[0]), int(out_wh[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(v) for v in fill),
    )


def turn_mask(
    mask: np.ndarray, matrix: np.ndarray, out_wh: tuple[int, int], *, at=(0, 0)
) -> np.ndarray | None:
    """A bool ``mask`` through ``matrix``, or ``None`` when nothing of it lands in the output.
    ``at`` is ``(y0, x0)``: where a cropped mask (a trail layer) sits in the picture."""
    y0, x0 = int(at[0]), int(at[1])
    moved = matrix.copy()
    moved[:, 2] += moved[:, :2] @ np.array([x0, y0], dtype=np.float64)
    turned = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        moved,
        (int(out_wh[0]), int(out_wh[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    out = turned > 127
    return out if out.any() else None


def turn_points(
    points: np.ndarray, matrix: np.ndarray, picture_wh, out_wh
) -> np.ndarray:
    """Normalized ``(K, 2)`` points of a ``picture_wh`` picture, through ``matrix``, as
    normalized points of the ``out_wh`` output (NaN stays NaN)."""
    px = np.asarray(points, dtype=np.float64) * np.array(picture_wh, dtype=np.float64)
    moved = px @ matrix[:, :2].T + matrix[:, 2]
    return moved / np.array(out_wh, dtype=np.float64)


def mark_no_axis(picture: np.ndarray) -> np.ndarray:
    """``picture`` with a small "no axis" box in its top-left corner: drawn as recorded
    because this frame gives no body axis — said, rather than left blank or turned wrongly.
    """
    out = picture.copy()
    h = out.shape[0]
    scale = max(h / 700.0, 0.32)
    thickness = 1
    (tw, th), base = cv2.getTextSize(
        NO_AXIS_MARK, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    pad = max(int(round(3 * scale / 0.32)), 2)
    cv2.rectangle(
        out, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (40, 40, 40), cv2.FILLED
    )
    cv2.putText(
        out,
        NO_AXIS_MARK,
        (pad, pad + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )
    return out


def frame_turn(
    frame: "ClipFrame", track_id: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """The axis ends of the clip's animal on ``frame`` in the frame's pixels, or ``None``
    when the run has no axis for it (no keypoints, no heading definition, no points here).
    """
    if frame.axis is None or frame.skeleton is None:
        return None
    own = frame.placed.get(int(track_id))
    if own is None:
        return None
    return axis_ends(own[0], own[1], frame.axis, _wh(frame))


def turn_views(
    frame: "ClipFrame",
    settings: ClipViewSettings,
    track_id: int,
    footage: np.ndarray | None,
    masks: np.ndarray,
    pose: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, bool]:
    """The three pictures in the **Aligned** orientation (``settings.aligned``): each turned
    by the one :func:`aligned_matrix` of this frame's axis, at the frame's own size. Returns
    ``(footage, masks, pose, turned)``; ``turned`` is False on a frame with no axis, whose
    pictures come back as recorded with the footage (or, without it, the masks pane) marked.

    A run with no axis at all (``frame.axis`` None: no keypoints or no heading definition)
    comes back untouched and unmarked — the switch that asks for this is greyed with the reason
    there, and marking every frame would only repeat it."""
    if not settings.aligned or frame.axis is None:
        return footage, masks, pose, False
    ends = frame_turn(frame, track_id)
    if ends is None:
        if footage is not None:
            return mark_no_axis(footage), masks, pose, False
        return footage, mark_no_axis(masks), pose, False
    wh = _wh(frame)
    matrix = aligned_matrix(ends[0], ends[1], wh[1], wh, frame.flip)
    if matrix is None:
        if footage is not None:
            return mark_no_axis(footage), masks, pose, False
        return footage, mark_no_axis(masks), pose, False
    if footage is not None:
        footage = turn_picture(footage, matrix, wh, (0, 0, 0))
    masks = turn_picture(masks, matrix, wh, (255, 255, 255))
    if pose is not None and not settings.pose_aligned:
        pose = turn_picture(pose, matrix, wh, (255, 255, 255))
    return footage, masks, pose, True


def pose_notes(
    experiment_dir: Path | str, embeds: dict
) -> tuple[str | None, str | None]:
    """``(keypoints, aligned)``: ``None`` each when the clips of these
    ``{recording_id: embed_run_id}`` can have their keypoints drawn / be drawn **Aligned**,
    else why not, in the clip window's words (``ClipPoints.note`` / ``axis_note``): an
    embedding that read no keypoints, a keypoint run that cannot be read, no heading definition
    that fits, several that do. The first recording that cannot answers for all: a montage
    mixes them. Reads each recording's embedding record and keypoint run — call it off the UI
    thread."""
    from detail.api.run_preview import _embedding_meta
    from detail.storage import experiment as experiment_store

    if not embeds:
        nothing = "There are no clips here."
        return nothing, nothing
    points_note = align_note = None
    for recording_id, embed_run_id in sorted(embeds.items()):
        recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
        try:
            meta = _embedding_meta(recording_dir, embed_run_id)
        except (ValueError, FileNotFoundError) as exc:
            why = f"The embedding run of {recording_id} cannot be read: {exc}"
            return why, why
        points = ClipPoints.of(Path(experiment_dir), recording_dir, meta, embed_run_id)
        points_note = points_note or points.note
        align_note = align_note or points.axis_note
        if points_note and align_note:
            break
    return points_note, align_note


def alignment_note(experiment_dir: Path | str, embeds: dict) -> str | None:
    """``None`` when these clips can be drawn **Aligned**, else why not
    (:func:`pose_notes`' second half)."""
    return pose_notes(experiment_dir, embeds)[1]


# --- the keypoints a run read ------------------------------------------------------------ #
class ClipPoints:
    """The keypoint run an embedding run read, held for drawing: each (subject, part) series
    is read whole the first time a frame of it is asked for, so playback looks a frame up
    rather than reopening a store per frame.

    It also holds the **heading definition** the body's axis is measured by (which points are
    the front and which the back): the embedding run's own when it was aligned, else the
    experiment's one saved definition that fits this skeleton. Several that fit is a choice
    this window does not make for you, so it uses none and ``axis_note`` says why.

    A **side-view** embedding (``align: mirror``) also hands over ``mirror``: ``(recording_dir,
    tracking_run_id, alignment settings)``, from which :meth:`flips` reads the flip the
    embedding made on each frame, so the Aligned picture flips where it flipped rather than
    turning the animal upright.
    """

    def __init__(
        self,
        run,
        note: str | None,
        part_stems: list[str],
        *,
        frame_wh: tuple[int, int] = (0, 0),
        definition: dict | None = None,
        axis_note: str | None = None,
        cutoff: float | None = None,
        mirror: tuple[Path, str, dict] | None = None,
    ) -> None:
        self.run = run
        self.note = note
        self.part_stems = part_stems
        self.frame_wh = (int(frame_wh[0]), int(frame_wh[1]))
        self.definition = definition
        self.axis_note = note if run is None else axis_note
        self.cutoff = cutoff
        self._mirror = mirror
        self._series: dict[tuple[int, str], tuple | None] = {}
        self._typical: dict[int, float] = {}
        self._flips: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._lock = threading.Lock()

    @property
    def skeleton(self) -> Skeleton | None:
        return None if self.run is None else self.run.skeleton

    @property
    def axis_columns(self) -> tuple[list[int], list[int], list[int]] | None:
        """``(front, back, middle)`` skeleton columns, or None without a definition."""
        if self.run is None or self.definition is None:
            return None
        names = tuple(self.skeleton.names)
        front, back = body_axis.definition_columns(self.definition, names)
        return front, back, body_axis.midline_columns(self.skeleton, self.definition)

    def axis(self, track_id: int) -> BodyAxis | None:
        """The subject's body axis for the curves and the aligned pose, or ``None`` without a
        heading definition (``axis_note`` says why)."""
        columns = self.axis_columns
        if columns is None:
            return None
        front, back, middle = columns
        return BodyAxis(
            front=tuple(front),
            back=tuple(back),
            middle=tuple(middle),
            frame_wh=self.frame_wh,
            cutoff=self.cutoff,
            typical_length_px=self.typical_length(int(track_id)),
        )

    @classmethod
    def of(
        cls, experiment_dir: Path, recording_dir: Path, meta: dict, embed_run_id: str
    ) -> "ClipPoints":
        """For one embedding run, from its record ``meta`` (``embedding.yaml``)."""
        from detail.storage import recording as recording_store
        from detail.storage.keypoint_runs import KeypointTrackRun

        data = recording_store.read_recording(recording_dir)
        stems = [
            recording_store.part_stem(f) for f in recording_store.video_files(data)
        ]
        resolution = (data.get("video") or {}).get("resolution") or (0, 0)
        run_id = str(meta.get("keypoint_run_id") or "")
        if not run_id:
            return cls(
                None,
                f"Embedding run {embed_run_id} used no keypoints, so there is no pose to "
                "show. An embedding aligned by, or drawing, a keypoint track run carries one.",
                stems,
            )
        try:
            run = KeypointTrackRun(recording_dir, run_id)
        except (FileNotFoundError, ValueError) as exc:
            return cls(
                None,
                f"The keypoint run embedding run {embed_run_id} used ({run_id}) cannot be "
                f"read: {exc}",
                stems,
            )
        definition, axis_note = heading_definition_for(
            experiment_dir, meta, tuple(run.skeleton.names)
        )
        cutoff = meta.get("heading_cutoff")
        if cutoff is None:
            cutoff = meta.get("pose_min_score")
        mirror = None
        if meta.get("align") == "mirror" and meta.get("tracking_run_id"):
            from detail.pipeline.embedding import _alignment_settings

            mirror = (
                Path(recording_dir),
                str(meta["tracking_run_id"]),
                _alignment_settings(meta),
            )
        return cls(
            run,
            None,
            stems,
            frame_wh=(int(resolution[0]), int(resolution[1])),
            definition=definition,
            axis_note=axis_note,
            cutoff=None if cutoff is None else float(cutoff),
            mirror=mirror,
        )

    def flips(self, track_id: int, part_stem: str, frames) -> np.ndarray | None:
        """``(n,)`` float, how a **side-view** embedding drew one subject on part-local
        ``frames`` of one part: ``1.0`` flipped, ``0.0`` not, NaN where it has no heading
        (a gap longer than its *Max gap*, before the first or after the last heading). ``None``
        for any other run: its Aligned picture is turned, not flipped (:func:`aligned_matrix`).

        The flags are `pipeline.embedding.headings_for_run`'s at every frame with the run's own
        settings, the one computation the embedding flips by, read whole once per subject.
        """
        if self._mirror is None:
            return None
        frames = np.asarray(frames, dtype=np.int64)
        out = np.full(len(frames), np.nan, dtype=np.float64)
        known = self._flip_series(int(track_id)).get(part_stem)
        if known is None or not len(known[0]) or not len(frames):
            return out
        at = np.clip(np.searchsorted(known[0], frames), 0, len(known[0]) - 1)
        hit = known[0][at] == frames
        out[hit] = known[1][at[hit]]
        return out

    def _flip_series(self, track_id: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            found = self._flips.get(track_id)
        if found is not None:
            return found
        from detail.analysis.body_axis import SOURCE_NONE
        from detail.executor import rundir
        from detail.pipeline.embedding import headings_for_run
        from detail.storage.user_yaml import read_user_yaml

        recording_dir, tracking_run_id, settings = self._mirror
        # A heading's angle is the same at any scale of the frame, so the recording's own
        # size serves; the masks' size is read only when the recording records none.
        mask_w, mask_h = self.frame_wh
        if min(self.frame_wh) <= 0:
            track_meta = read_user_yaml(
                recording_dir / "tracking" / tracking_run_id / "track.yaml"
            )
            detection_dir = (
                recording_dir / "detections" / track_meta["detection_run_id"]
            )
            mask_h, mask_w = rundir.read_envelope(detection_dir)["mask_resolution"]
        per_part = headings_for_run(
            recording_dir,
            tracking_run_id,
            [track_id],
            1,
            settings,
            (int(mask_h), int(mask_w)),
        )
        series = {}
        for (_track, part_id), headings in per_part.items():
            frames = np.asarray(headings["frame_idx"], dtype=np.int64)
            flag = np.where(
                np.asarray(headings["source"]) == SOURCE_NONE,
                np.nan,
                np.asarray(headings["mirrored"], dtype=np.float64),
            )
            order = np.argsort(frames, kind="stable")
            series[part_id] = (frames[order], flag[order])
        with self._lock:
            self._flips[track_id] = series
        return series

    def window(
        self, track_id: int, part_frame_counts: list[int], first: int, last: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(points (n, K, 2), scores (n, K))`` for one subject over the inclusive global
        frames ``[first, last]``, NaN on a frame it has no keypoints."""
        n = int(last) - int(first) + 1
        k = len(self.skeleton.names)
        points = np.full((n, k, 2), np.nan, dtype=np.float64)
        scores = np.full((n, k), np.nan, dtype=np.float64)
        start = 0
        for stem, count in zip(self.part_stems, part_frame_counts):
            lo, hi = max(int(first), start), min(int(last), start + int(count) - 1)
            if lo <= hi:
                series = self._load(int(track_id), stem)
                if series is not None:
                    frames, pts, scs = series
                    keep = (frames >= lo - start) & (frames <= hi - start)
                    at = frames[keep].astype(np.int64) + start - int(first)
                    points[at] = pts[keep]
                    scores[at] = scs[keep]
            start += int(count)
        return points, scores

    def rows_at(
        self, track_id: int, part_stem: str, frames
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(points (n, K, 2), scores (n, K))`` for one subject on part-local ``frames`` of
        one part, NaN on a frame it has no keypoints — a montage tile's frames, which are a
        proxy's samples of the part rather than a run of global frames."""
        frames = np.asarray(frames, dtype=np.int64)
        k = len(self.skeleton.names)
        points = np.full((len(frames), k, 2), np.nan, dtype=np.float64)
        scores = np.full((len(frames), k), np.nan, dtype=np.float64)
        series = self._load(int(track_id), part_stem)
        if series is None or not len(series[0]) or not len(frames):
            return points, scores
        known, pts, scs = series
        at = np.clip(np.searchsorted(known, frames), 0, len(known) - 1)
        hit = known[at] == frames
        points[hit] = pts[at[hit]]
        scores[hit] = scs[at[hit]]
        return points, scores

    def typical_length(self, track_id: int) -> float:
        """The subject's median body-axis length in pixels over the whole run (0 when it has
        none): what a body length is relative to, and what the aligned pose is scaled by, so
        neither changes with the window."""
        columns = self.axis_columns
        if columns is None:
            return 0.0
        with self._lock:
            found = self._typical.get(int(track_id))
        if found is not None:
            return found
        front, back, _middle = columns
        lengths = []
        for stem in self.part_stems:
            series = self._load(int(track_id), stem)
            if series is not None:
                lengths.append(
                    body_axis.row_headings(
                        series[1], series[2], front, back, self.frame_wh, self.cutoff
                    )[1]
                )
        every = np.concatenate(lengths) if lengths else np.zeros(0)
        every = every[np.isfinite(every) & (every > 0)]
        typical = float(np.median(every)) if every.size else 0.0
        with self._lock:
            self._typical[int(track_id)] = typical
        return typical

    def at(
        self, part_id: str, frame_idx: int
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """``{subject: (points (K, 2) normalized, scores (K,))}`` on one frame of a part."""
        out = {}
        for track_id in self.run.track_ids:
            series = self._load(int(track_id), part_id)
            if series is None:
                continue
            frames, points, scores = series
            i = int(np.searchsorted(frames, frame_idx))
            if i < len(frames) and int(frames[i]) == int(frame_idx):
                out[int(track_id)] = (points[i], scores[i])
        return out

    def _load(self, track_id: int, part_id: str):
        key = (track_id, part_id)
        with self._lock:
            if key not in self._series:
                if part_id in self.run.part_ids(track_id):
                    rows = self.run.rows(track_id, part_id)
                    self._series[key] = (
                        np.asarray(rows["frame_idx"]),
                        np.asarray(rows["points"], dtype=np.float32),
                        np.asarray(rows["point_scores"], dtype=np.float32),
                    )
                else:
                    self._series[key] = None
            return self._series[key]


def heading_definition_for(
    experiment_dir: Path, meta: dict, names: tuple[str, ...]
) -> tuple[dict | None, str | None]:
    """``(definition, None)`` or ``(None, why there is none)`` for the clip window's body axis:
    the aligned embedding's own, else the experiment's single saved definition that fits.
    """
    from detail.storage import heading_definitions

    own = meta.get("heading_definition")
    if own:
        try:
            return body_axis.validate_definition(dict(own), names), None
        except ValueError as exc:
            return None, f"The embedding's heading definition does not fit: {exc}"
    fitting = [
        d
        for d in heading_definitions.list_definitions(experiment_dir)
        if heading_definitions.fits(d, names) is None
    ]
    if len(fitting) == 1:
        return body_axis.validate_definition(fitting[0], names), None
    if not fitting:
        return None, (
            "No heading definition says which keypoints are the front and the back. Make "
            "one with Define... beside Heading on the embedding form (it is saved for the "
            "experiment) to measure the body's length, bend and turning and to align the "
            "pose."
        )
    return None, (
        "Several saved heading definitions fit this skeleton "
        f"({', '.join(sorted(str(d.get('name')) for d in fitting))}) and the embedding was "
        "not aligned by one, so none is chosen for you."
    )


# --- the subject's past masks, for the onion skin ---------------------------------------- #
_MISSING = object()


class MaskHistory:
    """The clip's animal's masks on the frames before the one shown, for the onion skin —
    **read from the detection run's rows, never by decoding video**
    (``DetectionPreview.subject_masks``), cropped to the mask, and kept for the frames
    playback asks for again. Bounded: a `core.lru.CountLRU` of ``capacity`` frames, so a long
    playback holds a few hundred small crops rather than a frame-sized array per frame.
    """

    def __init__(self, capacity: int = 256) -> None:
        self._kept: CountLRU = CountLRU(capacity)

    @property
    def held(self) -> int:
        return len(self._kept)

    def layers(
        self,
        preview,
        part_index: int,
        track_id: int,
        local_frames: list[int],
        size: tuple[int, int],
    ) -> list:
        """One ``(y0, x0, crop)`` or None per frame of ``local_frames`` (one part's)."""
        key = (int(part_index), int(track_id), tuple(int(v) for v in size))
        found: dict[int, object] = {}
        for frame in local_frames:
            hit = self._kept.get((key, frame), _MISSING)
            if hit is not _MISSING:
                found[frame] = hit
        missing = [f for f in local_frames if f not in found]
        if missing:
            read = preview.subject_masks(part_index, int(track_id), missing, size)
            for frame in missing:
                layer = _crop_to_mask(read.get(frame))
                found[frame] = layer
                self._kept.put((key, frame), layer)
        return [found[frame] for frame in local_frames]


def _crop_to_mask(mask: np.ndarray | None) -> tuple[int, int, np.ndarray] | None:
    """``(y0, x0, crop)`` of a bool mask's extent, or None for no mask. Two axis reductions
    rather than ``np.nonzero``, which lists every pixel of the animal to find four numbers.
    """
    if mask is None:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask[rows[0] : rows[-1] + 1].any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return int(y0), int(x0), mask[y0:y1, x0:x1].copy()


# --- small shared helpers ---------------------------------------------------------------- #
def source_frame_of(
    part_frame_counts: Sequence[int], part_index: int, local: int
) -> int:
    """A part-local frame as the recording-global frame §10's keyframes are numbered in."""
    return int(sum(part_frame_counts[:part_index])) + int(local)


def blind_aliases(recording_ids: Sequence[str]) -> dict[str, str]:
    """``{recording_id: "A"}``, lettered in the order the recordings first appear (then
    ``AA``, ``AB`` … past 26) — the clip window's Blind mode names, shared by the montage.
    """
    out: dict[str, str] = {}
    for recording_id in recording_ids:
        if recording_id in out:
            continue
        n = len(out)
        letters = ""
        while True:
            letters = chr(ord("A") + n % 26) + letters
            n = n // 26 - 1
            if n < 0:
                break
        out[recording_id] = letters
    return out


# --- a side pane, exported --------------------------------------------------------------- #
def pane_picture(views: ClipViews, pane: str) -> np.ndarray | None:
    """``pane``'s picture from one ``ClipViews`` — ``None`` for keypoints on a run with none.

    Raises:
        ValueError: ``pane`` is not one of :data:`PANES`.
    """
    if pane == MASKS:
        return views.masks
    if pane == KEYPOINTS:
        return views.pose
    raise ValueError(f"pane must be one of {PANES}, not {pane!r}")


def pane_title(pane: str, label: str = "") -> str:
    """The enlarged window's title: ``Masks · <clip>``."""
    name = "Masks" if pane == MASKS else "Keypoints"
    return f"{name} · {label}" if label else name


def pane_frames(
    t_ns: np.ndarray, first: int, last: int, fps: float | None
) -> list[int]:
    """The global frames an export of ``[first, last]`` writes: every one when ``fps`` is
    ``None``, else one per ``1/fps`` of real time (``t_ns`` is the recording's frame times) —
    the frame at or just before each sample time, with :data:`SAMPLE_SLACK_NS` of slack: an
    integer-nanosecond sample meant for a frame can fall a nanosecond short of that frame's
    own integer time, and without the slack a GIF at half the footage rate showed the frame
    *before* every other one it meant to.

    A ``last`` past the video's end is clamped to its last frame, as the montage export
    clamps the same span (``ClusterMontage.prepare``).

    Raises:
        UserInputError: ``first`` is past the video's end, so the span holds no frame.
    """
    n_frames = len(t_ns)
    if int(first) >= n_frames:
        raise UserInputError(
            f"frame {int(first)} is past the end of the video ({n_frames} frames); "
            f"the last frame is {n_frames - 1}"
        )
    first, last = int(first), int(min(max(last, first), n_frames - 1))
    if fps is None:
        return list(range(first, last + 1))
    span = np.asarray(t_ns[first : last + 1], dtype=np.int64)
    out = []
    for offset in sample_offsets_ns(int(span[-1] - span[0]), fps):
        position = (
            int(
                np.searchsorted(
                    span, int(span[0]) + offset + SAMPLE_SLACK_NS, side="right"
                )
            )
            - 1
        )
        out.append(first + min(max(position, 0), len(span) - 1))
    return out


def require_track(
    experiment_dir: Path | str, recording_id: str, embed_run_id: str, track_id: int
) -> None:
    """Refuse a ``track_id`` the embedding run did not embed in this recording.

    A drawing of a subject that is not there is a file of blank frames, written and reported
    as an export (#31). The GUI offers only tracks that exist; a script or the CLI can name
    any number.

    Raises:
        UserInputError: naming the tracks the run has.
    """
    from detail.api.run_preview import embedding_track_ids
    from detail.storage import experiment as experiment_store

    tracks = embedding_track_ids(
        experiment_store.recording_path(experiment_dir, recording_id), embed_run_id
    )
    if int(track_id) not in tracks:
        raise UserInputError(
            f"track {int(track_id)} is not in embedding run {embed_run_id!r} on "
            f"recording {recording_id!r}; its tracks: "
            f"{', '.join(str(t) for t in sorted(tracks)) or 'none'}"
        )


def export_clip_pane(
    experiment_dir: Path | str,
    *,
    recording_id: str,
    embed_run_id: str,
    track_id: int,
    first: int,
    last: int,
    pane: str,
    settings: ClipViewSettings,
    path: Path | str,
    max_width: int | None = None,
    fps: float | None = None,
    progress: Progress = None,
    preview=None,
) -> Path:
    """Write ``pane`` (masks or keypoints, on white, at the footage's size) over global frames
    ``[first, last]`` of one clip to ``path`` — what the enlarged pane shows, drawn with the
    clip window's ``settings``, and **no video decoded** (the masks and points are stored
    data).

    An mp4 keeps every frame at the recording's rate and full size unless ``max_width`` /
    ``fps`` say otherwise; a GIF defaults to :data:`GIF_DEFAULT_WIDTH` wide at
    :data:`GIF_DEFAULT_FPS`. ``preview`` (a ``ClipQueryPreview``) is reused when given.

    Raises:
        ValueError: an unknown pane or suffix, or keypoints asked of a run with none.
        UserInputError: a ``track_id`` the run does not have (:func:`require_track`).
    """
    from detail.api.run_preview import ClipQueryPreview

    if pane not in PANES:
        raise ValueError(f"pane must be one of {PANES}, not {pane!r}")
    require_track(experiment_dir, recording_id, embed_run_id, track_id)
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
                recording_id, embed_run_id, track_id, frame, settings, footage=False
            )
            if views is None:
                continue  # an unreadable frame is skipped, as the player skips it
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
    )


# --- the clip window's view, exported as shown ------------------------------------------- #
#: Pixels between the footage and the side column, and between the side panes — the window's.
VIEW_GAP_PX = 6


def view_picture(views: ClipViews, *, panes: bool) -> np.ndarray:
    """What the clip window shows of one frame, as one picture: the footage alone (``panes``
    False — the plain player), or the footage with the masks pane over the keypoints pane
    beside it at half its height (the Advanced player's layout, ``gui/clip_panes.py``). A run
    with no keypoints leaves the keypoints pane white, as the window leaves it empty.

    Raises:
        ValueError: ``views`` was read without the footage.
    """
    if views.frame is None:
        raise ValueError("this view was read without the footage; read it with footage")
    footage = views.frame
    if not panes:
        return footage
    h, w = footage.shape[:2]
    cell_h = max((h - VIEW_GAP_PX) // 2, 1)
    cell_w = max(int(round(cell_h * w / max(h, 1))), 1)
    out = np.full((h, w + VIEW_GAP_PX + cell_w, 3), 255, dtype=np.uint8)
    out[:, :w] = footage
    x = w + VIEW_GAP_PX
    for row, pane in enumerate((views.masks, views.pose)):
        if pane is None:
            continue
        small = cv2.resize(pane, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        y = row * (cell_h + VIEW_GAP_PX)
        out[y : y + cell_h, x : x + cell_w] = small
    return out


def export_clip_view(
    experiment_dir: Path | str,
    *,
    recording_id: str,
    embed_run_id: str,
    track_id: int,
    first: int,
    last: int,
    settings: ClipViewSettings,
    panes: bool,
    path: Path | str,
    max_width: int | None = None,
    fps: float | None = None,
    progress: Progress = None,
    preview=None,
) -> Path:
    """Write the clip window's view over global frames ``[first, last]`` — the clip and the
    context the window spans — to ``path`` (``.mp4`` / ``.gif``), drawn with the window's
    ``settings`` (marks, trails, **Aligned**) and laid out as it is (:func:`view_picture`).

    An mp4 keeps every frame at the recording's rate and full size unless ``max_width`` /
    ``fps`` say otherwise; a GIF defaults to :data:`GIF_DEFAULT_WIDTH` wide at
    :data:`GIF_DEFAULT_FPS`. Decodes the footage, one frame at a time, in order.

    Raises:
        ValueError: an unknown suffix.
    """
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
                recording_id, embed_run_id, track_id, frame, settings
            )
            if views is None:
                continue  # an unreadable frame is skipped, as the player skips it
            yield view_picture(views, panes=panes)

    return write_animation(
        path,
        pictures(),
        fps=rate,
        count=len(frames),
        max_width=max_width,
        progress=progress,
    )


def write_clip_frame(
    experiment_dir: Path | str,
    *,
    recording_id: str,
    embed_run_id: str,
    track_id: int,
    frame: int,
    settings: ClipViewSettings,
    panes: bool,
    path: Path | str,
    preview=None,
) -> Path:
    """Write global frame ``frame`` as the clip window shows it to ``path`` as a PNG.

    Raises:
        ValueError: the frame cannot be read.
    """
    from PIL import Image

    from detail.api.run_preview import ClipQueryPreview

    path = Path(path)
    if path.suffix.lower() != ".png":
        path = path.with_name(f"{path.name}.png")
    preview = preview or ClipQueryPreview(experiment_dir)
    views = preview.clip_views(
        recording_id, embed_run_id, track_id, int(frame), settings
    )
    if views is None:
        raise ValueError(f"frame {frame} of {recording_id} could not be read")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(view_picture(views, panes=panes), "RGB").save(path)
    return path
