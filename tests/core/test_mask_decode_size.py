"""A mask RLE decoded at a size other than the one it was encoded at.

**The class.** A detection run encodes each mask at the size its frames decoded at, and
records it as ``mask_resolution`` (§9). A COCO RLE string carries no size of its own: it is a
list of run lengths read column by column, so pycocotools decodes it at whatever ``(h, w)`` it
is handed and never complains. Handed the size of the frame a mask is about to be drawn on, or
the recording's ``video.resolution``, it lays another grid's runs down this one's columns and
the animal comes out as stripes across the frame. Nothing raised, so it was found by eye, on a
recording whose file decoded larger than its detection had, in the clip window, the cluster
montage and every clip export (detail-embedding/detail-dev#970).

**How it hid.** Every fixture wrote its masks at the size of its own footage, so the size a
reader passed and the size the mask had were always the same number, by two routes.

**The rule.** ``core.utils.decode_mask(counts, size, out=)``: ``size`` is the size the mask was
encoded at (``executor.rundir.mask_resolution``, per part); a caller that wants it on a frame
of another size passes that as ``out``. A decode at a size the runs cannot cover raises,
naming the rule, at every one of the ~30 sites that decode a mask — the readers that drew
stripes and any later one alike. A part dict (``run_preview._build_part``) carries its
encoded size as ``mask_hw`` for the readers that only see the footage.

The other way two sizes met: a part added at another resolution than the recording's, whose
masks every reader decoded at the run's (its first part's) size. §3 says a recording's parts
share one resolution, a bundle import checked it, and adding a part did not; now both ask
``storage/video_integrity.another_parts_shape``
(``tests/api/test_recording.py::test_a_part_of_another_shape_is_refused_before_anything_is_written``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from detail.core.utils import decode_mask, mask_boxes
from detail.executor import rundir


def _encode(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _square(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return mask


def test_pycocotools_alone_decodes_at_any_size_without_a_word():
    """Why the check is needed: the library turns another grid's runs into stripes."""
    counts = _encode(_square(24, 32))
    striped = mask_utils.decode({"counts": counts.encode("ascii"), "size": [48, 64]})
    assert striped.shape == (48, 64)
    assert not np.array_equal(striped.astype(bool), _square(48, 64))


def test_a_mask_decoded_at_another_size_than_its_own_is_refused():
    counts = _encode(_square(24, 32))
    with pytest.raises(ValueError, match="mask_resolution"):
        decode_mask(counts, (48, 64))


def test_a_mask_decodes_at_its_own_size_then_resizes_to_out():
    counts = _encode(_square(24, 32))
    np.testing.assert_array_equal(decode_mask(counts, (24, 32)), _square(24, 32))
    np.testing.assert_array_equal(
        decode_mask(counts, (24, 32), out=(48, 64)), _square(48, 64)
    )
    shrunk = decode_mask(counts, (24, 32), out=(12, 16))
    assert shrunk.shape == (12, 16) and shrunk.dtype == bool
    np.testing.assert_array_equal(shrunk, _square(12, 16))


def test_an_empty_mask_is_empty_at_the_size_asked_for():
    assert decode_mask("", (24, 32)).shape == (24, 32)
    blank = decode_mask("", (24, 32), out=(48, 64))
    assert blank.shape == (48, 64) and not blank.any()


def test_boxes_from_runs_refuse_another_size_too():
    counts = [_encode(_square(24, 32)), ""]
    np.testing.assert_allclose(mask_boxes(counts, (24, 32))[0], [8, 6, 16, 12])
    with pytest.raises(ValueError, match="mask_resolution"):
        mask_boxes(counts, (48, 64))


def _envelope(run_dir, **payload) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.json").write_text(
        json.dumps({"schema_version": 1, "status": "complete", **payload}),
        encoding="utf-8",
    )


def test_a_part_reads_its_own_encoded_size_before_the_runs(tmp_path):
    """The run's ``mask_resolution`` is its first part's; a part records its own."""
    run = tmp_path / "det_01"
    _envelope(
        run,
        mask_resolution=[48, 64],
        parts=[
            {"id": "part_001", "mask_resolution": [48, 64]},
            {"id": "part_002", "mask_resolution": [24, 32]},
        ],
    )
    assert rundir.mask_resolution(run, "part_002") == (24, 32)
    assert rundir.mask_resolution(run, "part_001") == (48, 64)
    assert rundir.mask_resolution(run, "part_009") == (
        48,
        64,
    )  # unknown part: the run's
    assert rundir.mask_resolution(run) == (48, 64)


def test_a_run_without_a_size_has_none(tmp_path):
    assert rundir.mask_resolution(tmp_path / "absent", "part_001") is None
    run = tmp_path / "det_02"
    _envelope(
        run, mask_resolution=None, parts=[{"id": "part_001", "mask_resolution": None}]
    )
    assert rundir.mask_resolution(run, "part_001") is None
