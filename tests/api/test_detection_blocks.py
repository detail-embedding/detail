"""A span's detections are read once per block, not once per frame.

**The class: a per-item read of a store whose cost is per call.** Every column read is its own
zarr call, and zarr v3's async layer charges about 2 ms for one however few rows it returns.
`DetectionPreview._detections_at` makes five per frame (bbox, class, confidence, object id, and
the mask RLE when asked), which is right for a scrub, one frame, now, and ruinous for a caller that
walks the recording. Measured on a real 87,187-frame recording: 7.7 ms a frame without masks and
11.8 with, more than decoding the frame, drawing on it and encoding it together. The keypoints
viewer's mask-plus-points video was 16 minutes where the points alone took about a minute.

`DetectionBlocks` reads the columns of a block of frames once and slices a frame out of them
(0.22 ms a frame on that recording, 51x, the same detections down to the mask pixels). These pin
the two halves of keeping it that way:

- **the answer is the per-frame answer**, for every frame, with and without masks, for a detection
  and for a tracking run (whose unassigned rows are dropped), at a block of one, of two and of
  more than the recording, and when frames are asked for out of order;
- **the reads stay per block**: a recording that fits in one block is one read of the columns,
  whatever its frame count. A caller that quietly goes back to asking per frame fails here, where
  every answer is still correct and only the cost has moved.

Where the fix stops (`fix-the-class` step 3.5): the viewer's own per-frame read (`frame`,
`frames_range`) is **deliberately not** changed, because a scrub wants one frame and a block read
for it would read a thousand; `frames_range` still reads per frame for the clip player's short
spans. `export_targets._tracking_underlay` is the one sequential walker of a whole recording, and
`tests/api/test_export_keypoints_preview.py` pins that it goes through the blocks.
"""

import numpy as np
import pytest

from detail.api import run_preview
from tests.api.test_run_preview import _add_tracking, _recording_with_detection


def _same(a, b):
    return len(a) == len(b) and all(
        x.bbox == y.bbox
        and x.object_id == y.object_id
        and x.class_name == y.class_name
        and x.confidence == y.confidence
        and ((x.mask is None and y.mask is None) or np.array_equal(x.mask, y.mask))
        for x, y in zip(a, b)
    )


@pytest.mark.parametrize("block", [1, 2, 1024])
@pytest.mark.parametrize("with_masks", [False, True])
@pytest.mark.parametrize("tracked", [False, True])
def test_blocks_give_the_detections_the_per_frame_read_gives(
    tmp_path, block, with_masks, tracked
):
    """Every frame, including the ones with no rows and the tracker's unassigned one."""
    recording = _recording_with_detection(tmp_path)
    if tracked:
        preview = _add_tracking(recording).tracking_preview("track_01")
    else:
        preview = recording.detection_preview("det_01")
    size = (48, 64) if with_masks else None
    blocks = preview.detection_blocks(0, size, block=block)
    part = preview._parts[0]

    for frame in range(5):
        assert _same(blocks.at(frame), preview._detections_at(part, frame, size)), frame
    # something was compared: frame 0 has rows, and with masks one of them has pixels
    assert len(blocks.at(0)) == (1 if tracked else 2)
    if with_masks:
        assert blocks.at(0)[0].mask[15, 15]


def test_blocks_answer_a_frame_asked_out_of_order(tmp_path):
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    blocks = preview.detection_blocks(0, None, block=2)
    part = preview._parts[0]
    for frame in (4, 0, 2, 2, 1, 3, 0):
        assert _same(blocks.at(frame), preview._detections_at(part, frame, None))


def _count_reads(monkeypatch):
    reads = []
    real = run_preview.DetectionPreview._read_rows

    def counting(self, part, lo, hi, *, with_masks):
        reads.append((lo, hi))
        return real(self, part, lo, hi, with_masks=with_masks)

    monkeypatch.setattr(run_preview.DetectionPreview, "_read_rows", counting)
    return reads


def test_blocks_read_the_columns_once_per_block_not_once_per_frame(
    tmp_path, monkeypatch
):
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    reads = _count_reads(monkeypatch)

    blocks = preview.detection_blocks(0, None, block=1024)
    for frame in range(5):
        blocks.at(frame)
    assert len(reads) == 1, f"five frames in one block made {len(reads)} reads: {reads}"

    reads.clear()
    small = preview.detection_blocks(0, None, block=2)
    for frame in range(5):
        small.at(frame)
    # frames 0-1 hold rows, 2-3 hold one, 4 holds none: a block with no rows reads nothing
    assert len(reads) == 2, reads


def test_a_block_with_no_detections_answers_empty(tmp_path):
    """Frame 4 sits alone in the last block and has no rows, so there is nothing to read and
    nothing to fail on."""
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    blocks = preview.detection_blocks(0, (48, 64), block=2)
    assert blocks.at(4) == []
    assert blocks.at(3) == []
