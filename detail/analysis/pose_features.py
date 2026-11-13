"""A subject's pose as numbers an embedding can carry (§14 "Pose in the embedding").

Pure numpy, and the only place the pose block is computed: the embedding run builds it per
track and part before the workers start, so a preview of the same frames reads the same
numbers.

**The body-relative frame.** Every frame's points are moved to the body's own frame before they
become numbers: centred on the mean of the visible points, turned so the heading (the run's
heading definition, ``analysis/body_axis.py``) points along +x, and divided by the track's body
length — the heading series' typical axis length, one number per track, so a curled body is
shorter *in the numbers*, which is the point. A mouse in the corner facing up and the same pose in
the middle facing right then give the same numbers, and a big animal and a small one do too. In
``mirror`` nothing is turned: the animal is only flipped where the run flips its image (the
hold-based ``mirror_flags``), for the reason alignment does not turn it (§14: a free rotation
flips on foreshortened frames).

**The vector.** ``(x, y)`` of every keypoint in that frame, in skeleton order, scaled to unit
length. A point that is not visible (not placed, or scoring below the floor) is put at the body's
centre (0, 0) and counted out of ``n_visible``, so a frame that is mostly guesses can be told from
one that is not. Unit length makes the pose block the same size whatever the animal does, so the
pose weight is the only thing that sets how much it counts against the mask's own vector (which
the worker also scales to unit length): both are compared by cosine everywhere downstream, so
"weight 1" gives them equal say.

Ported from the pre-§26 keypoints stage (#659), where the axis was a skeleton's front and back
point; on main it is the heading definition the aligned embedding already uses.
"""

from __future__ import annotations

import numpy as np

#: How the pose block is turned, by the run's ``align``: ``mirror`` flips, anything else turns.
POSE_FRAMES = ("rotate", "mirror")


def visible_points(
    xy: np.ndarray, score: np.ndarray, min_score: float | None
) -> np.ndarray:
    """Which points count as visible: placed, and (with a floor) scoring at least it."""
    placed = np.isfinite(np.asarray(xy, dtype=np.float64)).all(axis=-1)
    if min_score is None:
        return placed
    return placed & ~(np.asarray(score, dtype=np.float64) < float(min_score))


def pose_block(
    xy: np.ndarray,
    score: np.ndarray,
    *,
    deg: np.ndarray,
    mirrored: np.ndarray,
    body_length_px: float | None,
    frame_wh: tuple[int, int],
    min_score: float | None,
    frame_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """The pose block for each frame: ``(vectors (N, 2K) float32, n_visible (N,) uint8)``.

    ``xy`` is ``(N, K, 2)`` normalized full-frame points (NaN = not placed), ``score`` ``(N, K)``,
    ``deg`` the frame's heading in image degrees (NaN = none), ``mirrored`` whether the run flips
    that frame. A frame with fewer than two visible points, or (in ``rotate``) no heading, gets a
    zero vector and the count of points it had — never a made-up one.

    Raises:
        ValueError: ``frame_mode`` is not ``rotate`` or ``mirror``, or there is no body length
            (the heading's points were never both placed on this track).
    """
    if frame_mode not in POSE_FRAMES:
        raise ValueError(f"frame_mode must be rotate or mirror, got {frame_mode!r}")
    if not body_length_px or not np.isfinite(body_length_px):
        raise ValueError(
            "the pose numbers are in body lengths, and this track has none: the heading's "
            "front and back points are never both placed on it"
        )
    xy = np.asarray(xy, dtype=np.float64)
    n, k = xy.shape[0], xy.shape[1]
    vectors = np.zeros((n, 2 * k), dtype=np.float32)
    if n == 0:
        return vectors, np.zeros(0, dtype=np.uint8)
    vis = visible_points(xy, score, min_score)
    n_visible = vis.sum(axis=1).astype(np.uint8)
    deg = np.asarray(deg, dtype=np.float64)
    usable = vis.sum(axis=1) >= 2
    if frame_mode == "rotate":
        usable &= np.isfinite(deg)
    if not usable.any():
        return vectors, n_visible
    w, h = frame_wh
    px = xy * np.array([float(w), float(h)])
    count = np.maximum(vis.sum(axis=1), 1)[:, None]  # a row with none is not usable
    centre = np.where(vis[..., None], px, 0.0).sum(axis=1) / count
    rel = np.where(vis[..., None], px - centre[:, None, :], 0.0)
    if frame_mode == "rotate":
        theta = np.radians(np.where(usable, deg, 0.0))
        cos, sin = np.cos(theta), np.sin(theta)
        # Turn by -theta so the heading lies along +x (image coordinates: y points down).
        x = rel[..., 0] * cos[:, None] + rel[..., 1] * sin[:, None]
        y = -rel[..., 0] * sin[:, None] + rel[..., 1] * cos[:, None]
    else:
        flip = np.where(np.asarray(mirrored, dtype=bool), -1.0, 1.0)
        x, y = rel[..., 0] * flip[:, None], rel[..., 1]
    block = np.stack([x, y], axis=-1) / float(body_length_px)
    block = np.where(vis[..., None], block, 0.0).reshape(n, 2 * k)
    norm = np.linalg.norm(block, axis=1, keepdims=True)
    unit = np.divide(block, norm, out=np.zeros_like(block), where=norm > 0)
    vectors[usable] = unit[usable].astype(np.float32)
    return vectors, n_visible
