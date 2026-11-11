"""COCO run-length masks read as runs, not pixels (§26 *Assignment*).

A detection's mask is stored as a pycocotools compressed RLE string (§9). Two kinds of reader
want different things from it. A drawer, a crop or a shape statistic wants the ``(h, w)``
array, which is ``core.utils.decode_mask``. The keypoint track assignment wants to know
whether a handful of pixels around a pose's points are on the mask, and for that the array
is the wrong unit: an hour of one animal decoded 480k pixels 96,000 times to test 3 points
each time. :class:`RunMask` answers that question from the run lengths. ``ones_before(i)``
counts the mask's pixels before linear index ``i`` (column-major, as the format stores them)
by one binary search over the runs' ends, so "any pixel on in column ``x``, rows
``[y0, y1)``" is two searches whatever the frame size.

Pure numpy; this module imports neither cv2 nor pycocotools. The string format is
pycocotools' ``rleToString``: each run length is a sequence of 5-bit chunks, low bits first,
each chunk ``+ 48`` as a printable byte, ``0x20`` set on every chunk but the last, and
``0x10`` on the last chunk meaning the value is negative; from the fourth run on (index
``> 2``) the stored value is the difference from the run two before it. Runs are
column-major, alternate off / on, and start with an off run, which may be empty.
"""

from __future__ import annotations

import numpy as np

# A run length needs at most this many 5-bit chunks before the shift below would leave
# int64; a longer sequence is a corrupt string, not a mask.
_MAX_CHUNKS = 12


def rle_counts(counts: str) -> np.ndarray:
    """Run lengths (``int64``) of a pycocotools compressed RLE string; ``""`` → empty.

    Raises:
        ValueError: The string is not a compressed RLE (a byte outside the printable range,
            a value cut off at the end, or a run that does not fit an ``int64``).
    """
    if not counts:
        return np.zeros(0, dtype=np.int64)
    code = np.frombuffer(counts.encode("ascii"), dtype=np.uint8).astype(np.int64) - 48
    if code.min() < 0 or code.max() > 63:
        raise ValueError("RLE string holds a byte outside the compressed range")
    bits = code & 0x1F
    last = (code & 0x20) == 0
    if not last[-1]:
        raise ValueError("RLE string ends inside a run length")
    first = np.concatenate(([True], last[:-1]))
    start = np.flatnonzero(first)
    chunk_in_value = np.arange(len(code)) - start[np.cumsum(first) - 1]
    if chunk_in_value.max() >= _MAX_CHUNKS:
        raise ValueError("RLE string holds a run length that does not fit an int64")
    partial = np.cumsum(bits << (5 * chunk_in_value))
    ends_at = np.flatnonzero(last)
    raw = partial[ends_at] - np.concatenate(([0], partial[ends_at[:-1]]))
    negative = (bits[ends_at] & 0x10) != 0
    raw[negative] -= np.int64(1) << (5 * (chunk_in_value[ends_at][negative] + 1))
    out = raw.copy()
    if len(out) > 3:
        out[3::2] = out[1] + np.cumsum(raw[3::2])
    if len(out) > 4:
        out[4::2] = out[2] + np.cumsum(raw[4::2])
    return out


class RunMask:
    """A ``(h, w)`` bool mask held as its runs, counting exactly over pixel-index ranges.

    Pixel ``(y, x)`` has linear index ``x * h + y``. Runs alternate off / on from an off run.
    """

    __slots__ = (
        "size",
        "counts",
        "n_ones",
        "_ends",
        "_starts",
        "_ones_before",
        "_is_one",
    )

    def __init__(self, counts: np.ndarray, size: tuple[int, int]) -> None:
        h, w = int(size[0]), int(size[1])
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        total = int(counts.sum()) if len(counts) else 0
        if (len(counts) and int(counts.min()) < 0) or total != h * w:
            raise ValueError(
                f"RLE runs cover {total} pixels of a {h}x{w} mask, not {h * w}"
            )
        n = len(counts)
        self.size = (h, w)
        self.counts = counts
        ends = np.cumsum(counts)
        is_one = (np.arange(n) & 1) == 1
        ones = np.where(is_one, counts, 0)
        self.n_ones = int(ones.sum())
        # One sentinel run past the end, so an index equal to h*w resolves to "after
        # everything" rather than off the arrays.
        self._ends = ends
        self._starts = np.concatenate((ends - counts, [h * w]))
        self._is_one = np.concatenate((is_one, [False]))
        self._ones_before = np.concatenate(([0], np.cumsum(ones)))

    @classmethod
    def from_counts_str(cls, counts: str, size: tuple[int, int]) -> "RunMask":
        """From the zarr's ``mask_rle`` string; ``""`` is an all-off mask, as ``decode_mask``."""
        h, w = int(size[0]), int(size[1])
        if not counts:
            return cls(np.array([h * w], dtype=np.int64), (h, w))
        return cls(rle_counts(counts), (h, w))

    @classmethod
    def from_box(
        cls, x0: int, y0: int, x1: int, y1: int, size: tuple[int, int]
    ) -> "RunMask":
        """The filled box ``[x0, x1) × [y0, y1)`` clipped to the grid; empty when degenerate."""
        h, w = int(size[0]), int(size[1])
        x0, x1 = max(int(x0), 0), min(max(int(x1), 0), w)
        y0, y1 = max(int(y0), 0), min(max(int(y1), 0), h)
        if x0 >= x1 or y0 >= y1:
            return cls(np.array([h * w], dtype=np.int64), (h, w))
        n_cols = x1 - x0
        counts = np.empty(2 * n_cols + 1, dtype=np.int64)
        counts[0] = x0 * h + y0
        counts[1::2] = y1 - y0
        counts[2::2] = (h - y1) + y0
        counts[-1] = (h - y1) + (w - x1) * h
        return cls(counts, (h, w))

    def ones_before(self, index: np.ndarray) -> np.ndarray:
        """On pixels with linear index ``< index``, for ``0 <= index <= h * w`` (vectorised)."""
        index = np.asarray(index, dtype=np.int64)
        run = np.searchsorted(self._ends, index, side="right")
        partial = np.where(self._is_one[run], index - self._starts[run], 0)
        return self._ones_before[run] + partial

    def any_in_columns(
        self, x: np.ndarray, y0: np.ndarray, y1: np.ndarray
    ) -> np.ndarray:
        """Whether column ``x`` has an on pixel in rows ``[y0, y1)``; inputs on the grid."""
        base = np.asarray(x, dtype=np.int64) * self.size[0]
        return (self.ones_before(base + y1) - self.ones_before(base + y0)) > 0

    def to_array(self) -> np.ndarray:
        """The ``(h, w)`` bool array: tests and debugging, never the hot path."""
        h, w = self.size
        flat = np.repeat(self._is_one[:-1], self.counts)
        return flat.reshape(w, h).T
