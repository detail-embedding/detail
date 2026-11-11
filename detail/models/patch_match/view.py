"""The view: an axis-aligned affine map from frame pixels into the square crop a backbone sees.

A square around the tracked box — the longer side × (1 + 2 · ``CROP_PAD``) — rendered to
``crop_px`` pixels, black outside the frame (§26 "Patch matching", step 1). Every animal is
resized to the same crop, so a patch is a fixed fraction of body size whatever the rig. The
view also carries the backbone's patch size, so a grid cell maps back to frame pixels without a
module constant: the patch is the backbone's, not the engine's.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Padding either side of the tracked box's longer side (the research's ``CROP_PAD``).
CROP_PAD = 0.15
#: The crop the research measured on (ViT-B; §26 *Open questions* 4).
DEFAULT_CROP_PX = 448


@dataclass(frozen=True)
class View:
    """``u = (x - ox) * s``, ``v = (y - oy) * s``; the rendered image is ``size × size`` and a
    grid cell is ``patch`` rendered pixels."""

    ox: float
    oy: float
    s: float
    size: int
    patch: int

    @property
    def grid(self) -> int:
        return self.size // self.patch

    @property
    def patch_px(self) -> float:
        """One patch in source (frame) pixels."""
        return self.patch / self.s

    def to_view(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) * self.s, (y - self.oy) * self.s

    def to_source(self, u: float, v: float) -> tuple[float, float]:
        return u / self.s + self.ox, v / self.s + self.oy

    def cell_to_source(self, gy: float, gx: float) -> tuple[float, float]:
        """A (possibly fractional) grid cell's centre, in source pixels."""
        return self.to_source((gx + 0.5) * self.patch, (gy + 0.5) * self.patch)

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.s, 0.0, -self.ox * self.s], [0.0, self.s, -self.oy * self.s]],
            dtype=np.float64,
        )


def make_view(
    bbox_norm: tuple[float, float, float, float],
    frame_wh: tuple[int, int],
    crop_px: int,
    patch: int,
) -> View:
    if crop_px % patch:
        raise ValueError(
            f"crop_px must be a multiple of the patch size {patch}; got {crop_px}"
        )
    w, h = frame_wh
    x0, y0, x1, y1 = bbox_norm
    side = max((x1 - x0) * w, (y1 - y0) * h, 1.0) * (1 + 2 * CROP_PAD)
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    return View(cx - side / 2, cy - side / 2, crop_px / side, int(crop_px), int(patch))


def render(frame: np.ndarray, view: View) -> np.ndarray:
    """The view as RGB uint8; outside the frame is black."""
    return cv2.warpAffine(
        frame,
        view.matrix(),
        (view.size, view.size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def body_scale(
    bbox_norm: tuple[float, float, float, float], frame_wh: tuple[int, int]
) -> float:
    """``sqrt(w · h)`` of the tracked box in pixels — the decode's length unit (not a label)."""
    w, h = frame_wh
    x0, y0, x1, y1 = bbox_norm
    return float(np.sqrt(max((x1 - x0) * w, 1.0) * max((y1 - y0) * h, 1.0)))


def allowed_cells(
    mask: np.ndarray | None, view: View, frame_wh: tuple[int, int]
) -> np.ndarray | None:
    """Grid cells touching the subject mask dilated by one patch; ``None`` = no constraint.

    The mask never changes what the backbone sees — it only forbids cells in the score maps.
    """
    if mask is None or not mask.any():
        return None
    w, h = frame_wh
    if mask.shape != (h, w):
        mask = (
            cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            > 0
        )
    warped = cv2.warpAffine(
        mask.astype(np.uint8),
        view.matrix(),
        (view.size, view.size),
        flags=cv2.INTER_NEAREST,
    )
    p = view.patch
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p + 1, 2 * p + 1))
    dilated = cv2.dilate(warped, kernel) > 0
    g = view.grid
    cells = dilated[: g * p, : g * p].reshape(g, p, g, p)
    return cells.any(axis=(1, 3))
