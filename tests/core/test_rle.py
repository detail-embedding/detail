"""`core.rle`: a COCO RLE string read as runs, exact against pycocotools and against a dense
cumsum (its module docstring is the write-up). The class the pose-match change depends on is a
few-pixel membership question answered by rasterising the whole mask; this file makes the
run-level answer trustworthy on the edges a rasteriser never sees: a mask starting with an on
pixel (a zero-length first run), duplicate run ends, an index equal to the pixel total,
multi-chunk run lengths and negative deltas in the string, and a box clipped at the grid.
"""

from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from detail.core import rle
from detail.core.utils import decode_mask


def _encode(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _corpus(rng: np.random.Generator) -> list[np.ndarray]:
    masks = [np.zeros((9, 13), bool), np.ones((9, 13), bool), np.zeros((1, 1), bool)]
    for y, x in ((0, 0), (0, 12), (8, 0), (8, 12)):
        m = np.zeros((9, 13), bool)
        m[y, x] = True
        masks.append(m)
    strip = np.zeros((9, 13), bool)
    strip[:, 0] = True
    strip[8, :] = True
    masks.append(strip)
    for _ in range(40):
        h, w = int(rng.integers(1, 48)), int(rng.integers(1, 48))
        masks.append(rng.random((h, w)) < rng.choice([0.02, 0.3, 0.7]))
    big = np.zeros((2000, 2000), bool)
    big[1000, 1000] = True  # one run of two million pixels: five 5-bit chunks
    masks.append(big)
    salt = rng.random((2000, 2000)) < 0.001
    masks.append(salt)
    return masks


def test_runs_decode_to_the_pycocotools_mask():
    rng = np.random.default_rng(7)
    saw_continuation = saw_negative_delta = False
    for mask in _corpus(rng):
        s = _encode(mask)
        counts = rle.rle_counts(s)
        saw_continuation |= any((ord(c) - 48) & 0x20 for c in s)
        if len(counts) > 3:
            saw_negative_delta |= bool((counts[3:] - counts[1:-2] < 0).any())
        run = rle.RunMask.from_counts_str(s, mask.shape)
        np.testing.assert_array_equal(run.to_array(), decode_mask(s, mask.shape))
        assert run.n_ones == int(mask.sum())
    assert saw_continuation, "corpus never exercised a multi-chunk run length"
    assert saw_negative_delta, "corpus never exercised a negative delta"


def test_ones_before_and_any_in_columns_count_exactly():
    rng = np.random.default_rng(11)
    leading_on = np.zeros((6, 5), bool)
    leading_on[0, 0] = True  # the string starts with a zero-length off run
    masks = [leading_on, np.ones((3, 4), bool)] + [
        rng.random((int(rng.integers(1, 30)), int(rng.integers(1, 30)))) < 0.4
        for _ in range(30)
    ]
    for mask in masks:
        h, w = mask.shape
        run = rle.RunMask.from_counts_str(_encode(mask), (h, w))
        cum = np.concatenate(([0], np.cumsum(mask.T.ravel())))
        every = np.arange(h * w + 1)
        np.testing.assert_array_equal(run.ones_before(every), cum)
        x = rng.integers(0, w, 200)
        y0 = rng.integers(0, h + 1, 200)
        y1 = rng.integers(0, h + 1, 200)
        want = np.array([mask[a:b, c].any() for c, a, b in zip(x, y0, y1)])
        np.testing.assert_array_equal(run.any_in_columns(x, y0, y1), want)


@pytest.mark.parametrize(
    "box",
    [
        (2, 3, 7, 9),  # inside
        (-4, -2, 3, 5),  # clipped at the origin
        (5, 4, 40, 50),  # clipped at the far edges
        (0, 0, 11, 13),  # the whole grid
        (3, 0, 6, 13),  # full height: adjacent columns' off runs merge
        (6, 2, 6, 9),  # degenerate width
        (8, 9, 4, 3),  # inverted
        (20, 20, 30, 30),  # entirely outside
    ],
)
def test_from_box_matches_a_filled_array(box):
    h, w = 13, 11
    x0, y0, x1, y1 = box
    want = np.zeros((h, w), bool)
    want[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = True
    run = rle.RunMask.from_box(x0, y0, x1, y1, (h, w))
    np.testing.assert_array_equal(run.to_array(), want)
    assert run.n_ones == int(want.sum())


def test_an_empty_string_is_an_empty_mask():
    assert rle.rle_counts("").shape == (0,)
    run = rle.RunMask.from_counts_str("", (5, 7))
    assert run.n_ones == 0 and run.size == (5, 7)
    assert not run.to_array().any()
    assert not run.any_in_columns(np.array([3]), np.array([0]), np.array([5]))[0]


def test_the_rle_check_would_catch_a_corrupt_run():
    mask = np.zeros((20, 30), bool)
    mask[5:15, 8:22] = True
    s = _encode(mask)
    with pytest.raises(ValueError, match="RLE"):
        rle.RunMask.from_counts_str(s[:-1], mask.shape)
    flipped = s[:3] + chr(ord(s[3]) ^ 0x01) + s[4:]
    with pytest.raises(ValueError, match="RLE"):
        rle.RunMask.from_counts_str(flipped, mask.shape)
    with pytest.raises(ValueError, match="RLE runs cover"):
        rle.RunMask(np.array([5, 5]), (3, 3))
