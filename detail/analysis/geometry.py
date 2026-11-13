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

from detail.analysis import mask_shape


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


#: One frame's subject geometry, all normalized to the frame (§14 ``kinematics/``). ``shape``
#: is the outline (``mask_shape.FrameShape``), NaN where there was no mask.
FrameGeometry = namedtuple(
    "FrameGeometry",
    ["cx_norm", "cy_norm", "mask_area_norm", "bbox_norm", "shape"],
    defaults=(mask_shape.NO_SHAPE,),
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
            mx, my = mask_centroid(mask)
            # +0.5 because `mask_centroid` averages pixel *indices* and pixel ``i`` covers
            # ``[i, i+1)`` — its centre is ``i + 0.5``. Without it the normalized position is
            # short by half a pixel, which is negligible on one recording and stops the value
            # being resolution-independent: the same region of a 20px-wide frame and a
            # 200px-wide one would come back as 0.225 and 0.2475, and the whole reason nothing
            # here is in pixels is that two recordings have to be comparable.
            cx, cy = (mx + 0.5) / width, (my + 0.5) / height
    if not np.isfinite(cx):
        cx, cy = float((union[0] + union[2]) / 2), float((union[1] + union[3]) / 2)
    shape = mask_shape.frame_shape(mask, frame_hw)
    return FrameGeometry(float(cx), float(cy), area, union, shape)
