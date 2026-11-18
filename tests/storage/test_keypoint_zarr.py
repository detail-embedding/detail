"""The keypoint stores (§26 "Run directory structure"): the detect part's instance schema —
columns, dtypes, chunks, the ``layout`` attr, the skeleton in the attrs, the method's extras
under ``extras/`` — its frame-boundary flush and its reopen, the refusal of the per-track
layout, the track part's ``instance_row`` and the assignment part."""

import numpy as np
import pytest

from detail.core.dataclasses import FrameResult
from detail.storage import zarr_store
from detail.storage.keypoint_zarr import (
    EXTRAS_PREFIX,
    KeypointAssignmentPartReader,
    KeypointAssignmentPartWriter,
    KeypointDetectPartReader,
    KeypointDetectPartWriter,
    LegacyKeypointLayoutError,
    detect_columns,
    extras_spec,
)

NAMES = ("nose", "left_ear", "right_ear")
K, C = 3, 5


def _result(frame_idx: int, n: int = 2, with_extras: bool = True) -> FrameResult:
    rng = np.random.default_rng(frame_idx)
    extras = {}
    if with_extras:
        extras = {
            "ref_index": rng.integers(0, 9, (n, K, C)).astype(np.int16),
            "mirrored": rng.random((n, K)).astype(np.float32),
        }
    return FrameResult(
        frame_idx=frame_idx,
        t_ns=frame_idx * 1000,
        n=n,
        bbox=rng.random((n, 4)).astype(np.float32),
        confidence=rng.random(n).astype(np.float32),
        detection_row=np.arange(n, dtype=np.int32) + 10 * frame_idx,
        points=rng.random((n, K, 2)).astype(np.float32),
        point_scores=rng.random((n, K)).astype(np.float32),
        candidates=rng.random((n, K, C, 2)).astype(np.float32),
        candidate_scores=rng.random((n, K, C)).astype(np.float32),
        extras=extras,
    )


def _write(writer, result: FrameResult) -> None:
    for row in range(result.n):
        writer.add_instance(result, row)


def test_detect_part_schema_is_one_row_per_instance(tmp_path):
    """Column names, dtypes and chunk sizes exactly as §26 lists them (Tenet 16), the layout,
    the skeleton and the candidate count in the attrs, the extras nested under ``extras/``;
    zero, one and several instances per frame."""
    path = tmp_path / "part_001.zarr"
    spec = extras_spec(_result(0))
    assert spec == {"ref_index": ("int16", (K, C)), "mirrored": ("float32", (K,))}
    with KeypointDetectPartWriter(path, NAMES, C, spec) as writer:
        _write(writer, _result(0, n=2))
        _write(writer, _result(1, n=0))  # a frame with no instance writes nothing
        _write(writer, _result(2, n=1))
        _write(writer, _result(3, n=3))
    reader = KeypointDetectPartReader(path)
    attrs = reader._root.attrs
    assert reader.schema_version == 1 and attrs["layout"] == "instances"
    assert reader.skeleton_names == NAMES and reader.n_candidates == C
    assert reader.n_rows == 6
    expected = {
        "frame_idx": ("int32", (10000,)),
        "t_ns": ("int64", (10000,)),
        "bbox": ("float32", (10000, 4)),
        "confidence": ("float32", (10000,)),
        "detection_row": ("int32", (10000,)),
        "points": ("float32", (10000, K, 2)),
        "point_scores": ("float32", (10000, K)),
        "candidates": ("float32", (10000, K, C, 2)),
        "candidate_scores": ("float32", (10000, K, C)),
        f"{EXTRAS_PREFIX}mirrored": ("float32", (10000, K)),
        f"{EXTRAS_PREFIX}ref_index": ("int16", (10000, K, C)),
    }
    assert [name for name, *_ in detect_columns(K, C, spec)] == list(expected)
    assert set(reader.column_names) | {
        EXTRAS_PREFIX + n for n in reader.extras_names
    } == set(expected)
    for name, (dtype, chunks) in expected.items():
        column = reader.column(name)
        assert str(column.dtype) == dtype, name
        assert tuple(column.chunks) == chunks, name
    rows = reader.read_all()
    assert rows["frame_idx"].tolist() == [0, 0, 2, 3, 3, 3]
    assert rows["t_ns"].tolist() == [0, 0, 2000, 3000, 3000, 3000]
    assert rows["detection_row"].tolist() == [0, 1, 20, 30, 31, 32]
    np.testing.assert_array_equal(rows["bbox"][3:6], _result(3, n=3).bbox)
    np.testing.assert_array_equal(rows["confidence"][:2], _result(0).confidence)
    np.testing.assert_array_equal(rows["points"][3:6], _result(3, n=3).points)
    np.testing.assert_array_equal(
        rows[f"{EXTRAS_PREFIX}ref_index"][2:3], _result(2, n=1).extras["ref_index"]
    )


def test_a_method_without_box_confidence_or_detection_stores_nan_and_minus_one(
    tmp_path,
):
    """A row whose result leaves ``bbox`` / ``confidence`` / ``detection_row`` unset reads
    as "none": NaN box, NaN confidence, ``detection_row = −1``. One candidate means no
    candidate columns at all, not a K×1 array a reader has to special-case."""
    path = tmp_path / "part_001.zarr"
    result = FrameResult(
        frame_idx=4,
        t_ns=40,
        n=1,
        points=np.zeros((1, K, 2), np.float32),
        point_scores=np.zeros((1, K), np.float32),
    )
    with KeypointDetectPartWriter(path, NAMES, 1) as writer:
        writer.add_instance(result, 0)
    reader = KeypointDetectPartReader(path)
    rows = reader.read_all()
    assert np.isnan(rows["bbox"]).all() and np.isnan(rows["confidence"]).all()
    assert rows["detection_row"].tolist() == [-1]
    assert "candidates" not in reader.column_names
    assert "candidate_scores" not in reader.column_names
    assert reader.extras_names == () and reader.n_candidates == 1


def test_the_writer_flushes_only_where_a_new_frame_begins(tmp_path):
    """Tenet 12: with a buffer of 2 and three instances on one frame, the disk never holds
    part of a frame — so the resume point is always a whole frame."""
    path = tmp_path / "part_001.zarr"
    seen: list[list[int]] = []
    with KeypointDetectPartWriter(path, NAMES, 1, buffer_size=2) as writer:
        for frame, n in ((0, 1), (1, 3), (2, 2), (3, 1)):
            result = _result(frame, n=n, with_extras=False)
            for row in range(n):
                writer.add_instance(result, row)
                on_disk = KeypointDetectPartReader(path).column("frame_idx")[:]
                seen.append(on_disk.tolist())
    for frames in seen:
        if frames:
            last = frames[-1]
            total = {0: 1, 1: 3, 2: 2, 3: 1}[last]
            assert (
                frames.count(last) == total
            ), frames  # the last frame on disk is whole
    assert KeypointDetectPartReader(path).column("frame_idx")[:].tolist() == [
        0,
        1,
        1,
        1,
        2,
        2,
        3,
    ]


def test_a_reopened_store_is_cut_to_the_common_length_and_appends(tmp_path):
    path = tmp_path / "part_001.zarr"
    spec = extras_spec(_result(0))
    with KeypointDetectPartWriter(path, NAMES, C, spec) as writer:
        _write(writer, _result(0))
        _write(writer, _result(1))
    # a flush torn across columns: frame_idx grew, the rest did not
    root = zarr_store.open_group(path, mode="a")
    root["frame_idx"].resize((6,))
    root["frame_idx"][4:] = [2, 2]
    with KeypointDetectPartWriter(path, NAMES, C, spec) as writer:
        assert writer.rows_written == 4  # cut back to the common length
        _write(writer, _result(2))
    rows = KeypointDetectPartReader(path).read_all()
    assert rows["frame_idx"].tolist() == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError, match="never changes its skeleton"):
        KeypointDetectPartWriter(path, ("a", "b", "c"), C, spec)
    with pytest.raises(ValueError, match="cannot be read"):
        KeypointDetectPartReader(path).check_skeleton(("a", "b", "c"))


def test_a_store_in_the_per_track_layout_is_refused(tmp_path):
    """A store written before 2026-10-21 has a skeleton and no ``layout``: never read, never
    appended to, and the message says what to do."""
    path = tmp_path / "track_0" / "part_001.zarr"
    root = zarr_store.open_group(path, mode="a")
    root.attrs["schema_version"] = 1
    root.create_array("frame_idx", shape=(0,), chunks=(10000,), dtype="int32")
    root.attrs["skeleton_names"] = list(NAMES)
    with pytest.raises(LegacyKeypointLayoutError, match="re-run keypoint detection"):
        KeypointDetectPartReader(path)
    with pytest.raises(LegacyKeypointLayoutError, match="re-run keypoint detection"):
        KeypointDetectPartWriter(path, NAMES, 1)


def test_track_part_schema_and_skeleton_check(tmp_path):
    """The track part (§26): ``instance_row`` (−1 = no instance assigned), the final point,
    its score and ``changed`` per keypoint; the skeleton in the attrs, checked on read.
    """
    from detail.storage.keypoint_zarr import (
        KeypointTrackPartReader,
        KeypointTrackPartWriter,
        track_columns,
    )

    path = tmp_path / "track_0" / "part_001.zarr"
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    points = rng.random((4, K, 2)).astype(np.float32)
    with KeypointTrackPartWriter(path, NAMES) as writer:
        writer.add_rows(
            np.arange(4),
            np.arange(4) * 1000,
            np.array([0, -1, 2, 3]),
            points,
            np.full((4, K), 0.5),
            np.zeros((4, K)),
        )
    reader = KeypointTrackPartReader(path)
    assert reader.schema_version == 1 and reader.skeleton_names == NAMES
    expected = {
        "frame_idx": ("int32", (10000,)),
        "t_ns": ("int64", (10000,)),
        "instance_row": ("int32", (10000,)),
        "points": ("float32", (10000, K, 2)),
        "point_scores": ("float32", (10000, K)),
        "changed": ("int8", (10000, K)),
    }
    assert set(reader.column_names) == set(expected)
    assert [name for name, *_ in track_columns(K)] == list(expected)
    for name, (dtype, chunks) in expected.items():
        assert str(reader.column(name).dtype) == dtype, name
        assert tuple(reader.column(name).chunks) == chunks, name
    rows = reader.read()
    np.testing.assert_array_equal(rows["points"], points)
    assert rows["instance_row"].tolist() == [0, -1, 2, 3]
    with pytest.raises(ValueError, match="never changes its skeleton"):
        KeypointTrackPartWriter(path, ("a", "b", "c"))
    with pytest.raises(ValueError, match="cannot be read"):
        reader.check_skeleton(("a", "b", "c"))


def test_assignment_part_schema(tmp_path):
    """``assignments/<part>.zarr`` (§26): a subject per detect row, −1 / NaN unassigned."""
    path = tmp_path / "assignments" / "part_001.zarr"
    with KeypointAssignmentPartWriter(path) as writer:
        writer.add(0, 0.9)
        writer.add(-1, float("nan"))
        writer.add(3, 0.5)
    reader = KeypointAssignmentPartReader(path)
    assert reader.schema_version == 1
    expected = {"subject_id": ("int32", (10000,)), "inside": ("float32", (10000,))}
    assert set(reader.column_names) == set(expected)
    for name, (dtype, chunks) in expected.items():
        assert str(reader.column(name).dtype) == dtype, name
        assert tuple(reader.column(name).chunks) == chunks, name
    rows = reader.read_all()
    assert rows["subject_id"].tolist() == [0, -1, 3]
    assert rows["inside"][0] == pytest.approx(0.9) and np.isnan(rows["inside"][1])
