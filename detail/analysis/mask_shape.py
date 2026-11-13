"""The shape of a subject's mask on one frame — the outline measures a cluster profile reads (§14).

``geometry.frame_geometry`` says *where* the subject was (centroid, area, box). This says what
shape it was: how high it reached and how wide it stood on the floor, how long against how wide,
which way its long axis pointed, how filled-in and how round the outline is, and how much sticks
out of it. Computed where the merged mask is already in hand — the embed pass and the kinematics
backfill — and written to the same ``kinematics/`` sidecar, so nothing decodes a mask twice.

**Normalized, like everything in that sidecar.** Vertical extents are fractions of the frame's
height; lengths are fractions of ``sqrt(height × width)``, so a length means the same on a
non-square frame in any direction and a loader recovers pixels with one multiplication. Angles
are degrees in image coordinates (y down), the long axis in ``[0, 180)``. Ratios are unitless.

**NaN, never zero, when there is no mask.** Zero is a real shape (a point) and a real position.

OpenCV for the contour, hull and convexity defects — the one place a mask's outline is traced;
``geometry.py`` stays numpy-only for the research surface that shares it.
"""

from collections import namedtuple

import numpy as np

#: The kinematics columns this writes, in order — the sidecar's schema is built from it.
FIELDS = (
    "mask_top_norm",
    "mask_bottom_norm",
    "floor_width_norm",
    "major_axis_norm",
    "minor_axis_norm",
    "orientation_deg",
    "solidity",
    "circularity",
    "protrusions",
    "protrusion_depth_norm",
)

FrameShape = namedtuple("FrameShape", FIELDS)

NO_SHAPE = FrameShape(*([float("nan")] * len(FIELDS)))

#: The bottom share of the mask's own height counted as "on the floor".
FLOOR_BAND = 0.1
#: A convexity defect counts as a protrusion when it is at least this deep, as a share of the
#: mask's long axis — a tail or a raised limb, not the ripple of a fur outline.
PROTRUSION_DEPTH = 0.15


def frame_shape(mask: np.ndarray | None, frame_hw) -> FrameShape:
    """The outline measures of one frame's merged ``(H, W)`` bool mask, or :data:`NO_SHAPE`."""
    if mask is None:
        return NO_SHAPE
    height, width = int(frame_hw[0]), int(frame_hw[1])
    ys, xs = np.nonzero(mask)
    if len(xs) < 3 or height <= 0 or width <= 0:
        return NO_SHAPE
    import cv2

    scale = float(np.sqrt(height * width))
    top, bottom = int(ys.min()), int(ys.max())
    # Pixel i covers [i, i+1): the top edge is at `top`, the bottom edge at `bottom + 1`.
    top_norm, bottom_norm = top / height, (bottom + 1) / height
    band = max(1, int(round((bottom - top + 1) * FLOOR_BAND)))
    on_floor = xs[ys > bottom - band]
    floor_width = (int(on_floor.max()) - int(on_floor.min()) + 1) / width

    # Second moments: the equivalent ellipse's full axis lengths are 4·sqrt(eigenvalue).
    coords = np.stack([xs, ys]).astype(np.float64)
    cov = np.cov(coords)
    values, vectors = np.linalg.eigh(cov)
    minor_len, major_len = 4.0 * np.sqrt(np.maximum(values, 0.0))
    vx, vy = vectors[:, 1]
    orientation = float(np.degrees(np.arctan2(vy, vx)) % 180.0)

    # The outline, on the mask's own box so the trace is as cheap as the mask is small.
    x0, y0 = int(xs.min()), top
    crop = np.ascontiguousarray(
        mask[y0 : bottom + 1, x0 : int(xs.max()) + 1], dtype=np.uint8
    )
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = float(len(xs))
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    points = np.concatenate([c.reshape(-1, 2) for c in contours]) if contours else None
    hull_area = (
        float(cv2.contourArea(cv2.convexHull(points))) if points is not None else 0.0
    )
    solidity = min(area / hull_area, 1.0) if hull_area > 0 else float("nan")
    circularity = (
        min(4.0 * np.pi * area / perimeter**2, 1.0) if perimeter > 0 else float("nan")
    )
    protrusions, deepest = 0, 0.0
    largest = max(contours, key=cv2.contourArea) if contours else None
    if largest is not None and len(largest) >= 4:
        hull_idx = cv2.convexHull(largest, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 3:
            try:
                defects = cv2.convexityDefects(largest, hull_idx)
            except cv2.error:  # a self-intersecting hull index order on a tiny contour
                defects = None
            if defects is not None:
                depths = defects.reshape(-1, 4)[:, 3].astype(np.float64) / 256.0
                deep = depths >= PROTRUSION_DEPTH * major_len
                protrusions = int(deep.sum())
                deepest = float(depths.max()) if len(depths) else 0.0
    return FrameShape(
        top_norm,
        bottom_norm,
        float(floor_width),
        float(major_len / scale),
        float(minor_len / scale),
        orientation,
        float(solidity),
        float(circularity),
        float(protrusions),
        float(deepest / scale),
    )
