"""Movement inside the mask (``pipeline/motion_backfill.py``) and the two measures on it.

Guards:

- **The change is against the frame just before, or nothing.** The first frame of a part has no
  previous frame and is NaN; a frame whose previous was not decoded must not be compared with an
  older one.
- **Same rows, same order as the embedding part**, so the loader joins it by position.
- **Ends and middle are the long axis's thirds**: change painted on one end reads at the ends.
"""

import numpy as np
import pytest

from detail.analysis.descriptors import DESCRIPTORS, FrameSlice
from detail.pipeline import motion_backfill
from detail.storage.zarr_io import PartZarrReader
from tests.pipeline.test_kinematics_backfill import _run_embedding
from tests.pipeline.test_embedding import _setup  # noqa: F401


def test_every_embedded_frame_gets_its_change_since_the_frame_before(tmp_path):
    # The fixture's frames are uniform and brighten by 7 levels a frame.
    recording_dir = _setup(tmp_path, object_ids=[0, 0, -1, 0])
    run_dir = _run_embedding(recording_dir)
    part = run_dir / "track_0" / "part_001.zarr"
    stages = []
    written = motion_backfill.backfill_run(
        recording_dir, "emb_01", progress=lambda *a: stages.append(a)
    )
    assert written == 1 and stages[-1][1] == 1.0
    rows = PartZarrReader(motion_backfill.motion_part_path(part)).read()
    embedded = PartZarrReader(part).read()
    assert rows["frame_idx"].tolist() == embedded["frame_idx"].tolist() == [0, 1, 3]
    inside = rows["motion_inside"]
    assert np.isnan(inside[0]), "frame 0 has no frame before it"
    assert inside[1:] == pytest.approx([7 / 255, 7 / 255], abs=0.01)
    assert motion_backfill.missing_parts(recording_dir, "emb_01") == []
    assert (
        motion_backfill.backfill_run(recording_dir, "emb_01") == 0
    ), "already measured"


def test_change_at_one_end_reads_at_the_ends():
    mask = np.zeros((40, 90), bool)
    mask[10:30, 0:90] = True  # long axis along x
    diff = np.zeros(mask.shape, np.float32)
    diff[:, 0:20] = 0.5  # the left end moved
    inside, ends, middle = motion_backfill.mask_motion(diff, mask)
    assert ends == pytest.approx(0.5 * 20 / 60, abs=0.01)  # 20 of the ends' 60 columns
    assert middle == pytest.approx(0.0)
    assert inside == pytest.approx(0.5 * 20 / 90, abs=0.01)
    assert all(np.isnan(v) for v in motion_backfill.mask_motion(diff, mask & False))


def _slice(inside, ends, middle):
    n = len(inside)
    return FrameSlice(
        frame_idx=np.arange(n),
        t_ns=np.arange(n, dtype=np.int64),
        motion={
            "inside": np.asarray(inside, float),
            "ends": np.asarray(ends, float),
            "middle": np.asarray(middle, float),
        },
    )


def test_the_measures_read_the_motion_and_ignore_gaps():
    s = _slice([np.nan, 0.2, 0.4], [np.nan, 0.3, 0.3], [np.nan, 0.1, 0.1])
    assert DESCRIPTORS["inner_motion"].fn(s) == pytest.approx(0.3)
    assert DESCRIPTORS["end_motion_share"].fn(s) == pytest.approx(0.75)
    bare = FrameSlice(frame_idx=np.arange(3), t_ns=np.arange(3, dtype=np.int64))
    assert np.isnan(DESCRIPTORS["inner_motion"].fn(bare))
