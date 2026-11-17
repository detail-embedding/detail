"""Where a subject was on one frame — the primitives the pipeline and the research surface share.

Split out of ``analysis/dynamics.py`` rather than imported from it. That module's own docstring
states the invariant: *"Nothing in the pipeline imports this module — it is a research surface,
and keeping it that way is what makes it safe to change."* The embedding pass needs
:func:`frame_geometry`, so the shared half moves down here and ``dynamics`` re-exports
:func:`mask_centroid` for its existing callers. The research surface stays free to change.

Pure over arrays: numpy only, no zarr, no paths, no Qt.
"""

from collections import namedtuple

import numpy as np


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Center of mass ``(x, y)`` of a ``(H, W)`` bool mask; ``(nan, nan)`` when empty.

    This is the mask's *mass* center, unlike the bbox center (geometric): the two diverge when
    the animal's shape changes without its box moving, which is what ``centroid_divergence``
    picks up.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    return (float(xs.mean()), float(ys.mean()))


def mask_centre_norm(mask: np.ndarray) -> tuple[float, float]:
    """Centre of mass ``(x, y)`` of a ``(H, W)`` bool mask as a fraction of its width and
    height; ``(nan, nan)`` when empty. **The** position of a mask, for every caller.

    ``+ 0.5`` because :func:`mask_centroid` averages cell *indices* and cell ``i`` covers
    ``[i, i+1)``: its centre is ``i + 0.5``. Without it the position is half a cell up and
    left, which stops it being resolution-independent (the same region of a 20-cell and a
    200-cell mask comes back as 0.225 and 0.2475) and put the location export's mask centre
    half a cell away from the kinematics one for the same frame (#32).
    """
    mx, my = mask_centroid(mask)
    height, width = mask.shape[:2]
    if not np.isfinite(mx) or height <= 0 or width <= 0:
        return (float("nan"), float("nan"))
    return ((mx + 0.5) / width, (my + 0.5) / height)


#: One frame's subject geometry, all normalized to the frame (§14 ``kinematics/``).
FrameGeometry = namedtuple(
    "FrameGeometry", ["cx_norm", "cy_norm", "mask_area_norm", "bbox_norm"]
)

#: What a frame with nothing to describe reports. Public because the embedding worker writes it
#: for a row the renderer never reached, keeping the kinematics file row-aligned with the
#: embedding one.
NO_GEOMETRY = FrameGeometry(
    float("nan"), float("nan"), float("nan"), np.full(4, np.nan, dtype=np.float32)
)


def frame_geometry(mask: np.ndarray | None, bboxes, frame_hw) -> FrameGeometry:
    """Where the subject was on one frame, from its merged mask and its detections' boxes.

    ``bboxes`` are the frame's **normalized xyxy** rows for this subject — several when the
    segmentation split the animal into fragments, which are OR-merged into ``mask`` upstream. The
    box returned is their **union**, the same quantity ``render_preview._Candidate`` computes;
    taking the highest-confidence one instead would describe the largest fragment and call it the
    animal.

    The centroid is the mask's center of mass, which is what a person means by "where it was". It
    falls back to the union box's center when there is a box but no usable mask, and to NaN when
    there is neither. **NaN, never zero** — zero is the top-left corner, a perfectly plausible
    position, so a zero filler is indistinguishable from a real reading on the one screen this
    exists to make trustworthy.

    ``frame_hw`` is ``(height, width)``, matching ``decode_mask``'s size argument, so a caller
    that has one has the other.
    """
    boxes = np.asarray(list(bboxes), dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return NO_GEOMETRY
    union = np.array(
        [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()],
        dtype=np.float32,
    )
    height, width = int(frame_hw[0]), int(frame_hw[1])
    area = float("nan")
    cx = cy = float("nan")
    if mask is not None and height > 0 and width > 0:
        pixels = int(np.count_nonzero(mask))
        area = pixels / float(height * width)
        if pixels:
            # The cell-centre rule (``+ 0.5``) lives in `mask_centre_norm`, so the location
            # export cannot drift from it again.
            cx, cy = mask_centre_norm(mask)
    if not np.isfinite(cx):
        cx, cy = float((union[0] + union[2]) / 2), float((union[1] + union[3]) / 2)
    return FrameGeometry(float(cx), float(cy), area, union)
