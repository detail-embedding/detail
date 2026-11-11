"""Where the subject was on one frame — the rules the spatial map is built on.

Two of these guard findings that motivated capturing geometry in the embedding pass at all,
rather than reconstructing it later from ``det_row``:

- a frame's detections are **unioned**, not represented by the highest-confidence one, so a
  fragmented mask reports the animal rather than its largest piece;
- a frame with nothing to describe reports **NaN, not zero** — zero is the top-left corner, a
  perfectly plausible position, so a zero filler is indistinguishable from a real reading.

Both are plausible-wrong rather than crashes, on the one screen that exists to be trusted.
"""

import numpy as np
import pytest

from detail.analysis import geometry


def _mask(shape, rows, cols) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[rows[0] : rows[1], cols[0] : cols[1]] = True
    return m


def test_the_centroid_is_the_masks_centre_of_mass_normalized():
    # 4 rows x 6 cols at rows 0-3, cols 2-7 of a 16x20 frame.
    mask = _mask((16, 20), (0, 4), (2, 8))
    got = geometry.frame_geometry(mask, [(0.1, 0.2, 0.3, 0.4)], (16, 20))
    assert got.cx_norm == 5.0 / 20  # pixel i covers [i, i+1), so its centre is i+0.5
    assert got.cy_norm == 2.0 / 16
    assert got.mask_area_norm == 24 / 320


def test_the_box_is_the_union_over_the_frames_detections():
    """The animal, not its largest fragment.

    A frame may carry several detections of one subject (the detector split it), OR-merged
    downstream into one silhouette. ``det_row`` names only the highest-confidence one, which is
    why geometry taken from it later would describe a piece and call it the animal.
    """
    boxes = [(0.10, 0.20, 0.30, 0.40), (0.25, 0.05, 0.60, 0.35)]
    got = geometry.frame_geometry(None, boxes, (16, 20))
    assert tuple(got.bbox_norm) == pytest.approx((0.10, 0.05, 0.60, 0.40))
    # And the fallback centroid is the union's centre — not either box's.
    assert got.cx_norm == pytest.approx((0.10 + 0.60) / 2)
    assert got.cy_norm == pytest.approx((0.05 + 0.40) / 2)


def test_an_empty_mask_falls_back_to_the_box_rather_than_reporting_nothing():
    # A detection with no mask pixels still says where the animal was.
    got = geometry.frame_geometry(
        np.zeros((16, 20), dtype=bool), [(0.2, 0.4, 0.6, 0.8)], (16, 20)
    )
    assert got.cx_norm == pytest.approx(0.4)
    assert got.cy_norm == pytest.approx(0.6)
    assert (
        got.mask_area_norm == 0.0
    )  # measured as empty, which is not the same as unknown


def test_no_detection_is_nan_not_zero():
    got = geometry.frame_geometry(None, [], (16, 20))
    assert np.isnan(got.cx_norm) and np.isnan(got.cy_norm)
    assert np.isnan(got.mask_area_norm)
    assert np.isnan(got.bbox_norm).all()
    # The property that matters, stated as itself: nothing here reads as a position.
    assert got.cx_norm != 0.0 or np.isnan(got.cx_norm)


def test_the_mask_centroid_beats_the_box_centre_when_both_exist():
    """They differ whenever the animal's shape is off-centre in its box, which is most of the
    time — and the mask's is what a person means by "where it was"."""
    mask = _mask((16, 20), (0, 2), (0, 2))  # hard up in the corner
    box = (0.0, 0.0, 1.0, 1.0)  # a box covering the whole frame
    got = geometry.frame_geometry(mask, [box], (16, 20))
    assert got.cx_norm < 0.2 and got.cy_norm < 0.2  # the mask's, not the box's 0.5/0.5


def test_geometry_is_normalized_so_two_resolutions_agree():
    """The reason nothing here is in pixels: a value has to be comparable across recordings of
    different resolutions, and across a downsampled copy of one."""
    small = geometry.frame_geometry(
        _mask((16, 20), (0, 4), (2, 8)), [(0, 0, 1, 1)], (16, 20)
    )
    big = geometry.frame_geometry(
        _mask((160, 200), (0, 40), (20, 80)), [(0, 0, 1, 1)], (160, 200)
    )
    assert abs(small.cx_norm - big.cx_norm) < 1e-6
    assert abs(small.cy_norm - big.cy_norm) < 1e-6
    assert abs(small.mask_area_norm - big.mask_area_norm) < 1e-6
