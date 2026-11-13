"""The per-part zarr writers and readers: schema, growth, resume, and the share retry.

Most of this file is the Tenet 16 schema pattern — columns, dtypes and chunk sizes asserted
explicitly for each writer. One section is a recorded class's guard, the behavioural half of
`tests/storage/test_zarr_seam.py`: *a hazard mitigated at one call site rather than at the
seam every call site passes through.* zarr commits each ``zarr.json`` and chunk with its own
``tmp.replace(dst)``, which on a network share intermittently raises a transient
``PermissionError``. The seam scan proves every group is opened through
``storage/zarr_store.py``; it cannot prove that store actually retries. The
"Network-share replace retry" tests below do — a store that retried once, or not at all,
passes the scan and fails here. The seam guard's docstring is the full write-up.
"""

import numpy as np
import pytest
import zarr

from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartReader,
    DetectionPartWriter,
    EmbeddingPartWriter,
    PartZarrReader,
    TracksSummaryWriter,
    UNSEEN,
    object_ids_on_detection_rows,
    resolve_assignment_rows,
)


def _rows_for_frame(frame_idx, n_det):
    """Deterministic detection rows for a frame."""
    rows = []
    for d in range(n_det):
        rows.append(
            dict(
                frame_idx=frame_idx,
                t_ns=frame_idx * 33_333_333,
                detection_idx=d,
                class_name="mouse" if d == 0 else "pellet",
                confidence=round(0.5 + 0.1 * d, 2),
                bbox=(0.1 * d, 0.2, 0.3, 0.4),
                mask_rle=f"rle_{frame_idx}_{d}",
            )
        )
    return rows


def _write_frames(writer, frames, n_det_fn):
    for f in frames:
        for row in _rows_for_frame(f, n_det_fn(f)):
            writer.add_row(**row)


def _always_two(_frame):
    return 2


def _read(path):
    return DetectionPartReader(path).read()


def _assert_columns_equal(a, b):
    for name in a:
        if a[name].dtype == object or str(a[name].dtype).startswith("str"):
            assert [str(x) for x in a[name]] == [str(x) for x in b[name]], name
        else:
            assert np.array_equal(a[name], b[name]), name


def test_schema_dtypes_chunks(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as w:
        _write_frames(w, range(5), _always_two)

    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["schema_version"] == 1
    assert set(root.array_keys()) == {
        "frame_idx",
        "t_ns",
        "detection_idx",
        "class_name",
        "confidence",
        "bbox",
        "mask_rle",
    }
    assert root["frame_idx"].dtype == np.int32
    assert root["t_ns"].dtype == np.int64
    assert root["detection_idx"].dtype == np.int32
    assert root["confidence"].dtype == np.float32
    assert root["bbox"].dtype == np.float32
    assert root["bbox"].shape[1] == 4
    assert root["frame_idx"].chunks == (10000,)
    assert root["mask_rle"].chunks == (1000,)


def test_data_round_trip(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as w:
        _write_frames(w, range(4), _always_two)

    data = _read(path)
    assert data["frame_idx"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert data["detection_idx"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert [str(x) for x in data["class_name"]] == ["mouse", "pellet"] * 4
    assert [str(x) for x in data["mask_rle"]][:2] == ["rle_0_0", "rle_0_1"]
    assert data["t_ns"][2] == 1 * 33_333_333
    assert data["bbox"].shape == (8, 4)
    assert np.allclose(data["bbox"][1], [0.1, 0.2, 0.3, 0.4])


def test_buffer_flush_multiple_flushes_preserve_all_rows(tmp_path):
    path = tmp_path / "part_001.zarr"
    # small buffer forces many frame-boundary flushes
    with DetectionPartWriter(path, buffer_size=3) as w:
        _write_frames(w, range(20), _always_two)

    data = _read(path)
    assert data["frame_idx"].shape[0] == 40  # 20 frames * 2 detections
    assert data["frame_idx"].tolist() == [f for f in range(20) for _ in range(2)]


def test_zero_detection_frames_write_no_rows(tmp_path):
    path = tmp_path / "part_001.zarr"

    def n_det(frame):
        return frame % 3  # frames 0, 3, 6, ... produce 0 rows

    with DetectionPartWriter(path, buffer_size=4) as w:
        _write_frames(w, range(9), n_det)

    expected_rows = sum(f % 3 for f in range(9))
    reader = DetectionPartReader(path)
    assert reader.n_rows == expected_rows
    # frame 0 (0 dets) absent; first row belongs to frame 1
    assert int(reader.column("frame_idx")[0]) == 1


def test_resume_is_identical_to_uninterrupted(tmp_path):
    ref_path = tmp_path / "ref.zarr"
    with DetectionPartWriter(ref_path) as w:
        _write_frames(w, range(50), _always_two)
    reference = _read(ref_path)

    # interrupted run: write frames 0..29, flush, then abandon (simulated crash)
    int_path = tmp_path / "int.zarr"
    w_a = DetectionPartWriter(int_path)
    _write_frames(w_a, range(30), _always_two)
    w_a.flush()
    del w_a  # crash — no clean close, buffer already flushed at a frame boundary

    # resume from the next unprocessed frame
    last = DetectionPartReader(int_path).last_frame_idx
    assert last == 29
    with DetectionPartWriter(int_path) as w_b:
        _write_frames(w_b, range(last + 1, 50), _always_two)

    _assert_columns_equal(reference, _read(int_path))


def test_torn_flush_is_repaired_on_open(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as w:
        _write_frames(w, range(10), _always_two)  # 20 rows

    # simulate a crash mid-flush: one column got more rows than the others
    root = zarr.open_group(str(path), mode="a")
    root["frame_idx"].resize((22,))
    assert root["frame_idx"].shape[0] != root["t_ns"].shape[0]

    # reopening the writer truncates every column to the common length
    DetectionPartWriter(path)
    reader = DetectionPartReader(path)
    assert reader.n_rows == 20
    assert reader.column("frame_idx").shape[0] == 20


def test_a_torn_store_reads_consistently_without_a_writer_reopening_it(tmp_path):
    """A read-only reader has no write access to repair a torn store the way
    ``test_torn_flush_is_repaired_on_open`` shows a writer does — but every caller
    that cross-references two columns by row index (a tracker masking by class,
    frame grouping against ``bbox``) already assumes they agree. Without this, a
    reader crashes exactly the way `detail.trackers._common.filter_by_class` did in
    practice: ``IndexError: boolean index did not match indexed array``, because its
    mask (built from ``class_name``) and another column (``bbox``) disagreed in
    length — with no crash message pointing at the real cause, a torn store from an
    interrupted remote/multi-GPU detection run."""
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as w:
        _write_frames(w, range(10), _always_two)  # 20 rows

    # simulate a crash mid-flush: one column got more rows than the others, and
    # nothing ever reopens a *writer* on this store again (a reader is all that's
    # left, exactly the tracking-run shape).
    root = zarr.open_group(str(path), mode="a")
    root["frame_idx"].resize((22,))
    assert root["frame_idx"].shape[0] != root["t_ns"].shape[0]

    reader = DetectionPartReader(path)
    assert reader.n_rows == 20
    data = reader.read()
    assert {len(v) for v in data.values()} == {20}
    assert reader.column("frame_idx").shape[0] == 22  # column() stays lazy/raw


def test_resume_appends_after_consistent_repair(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as w:
        _write_frames(w, range(10), _always_two)

    with DetectionPartWriter(path) as w:
        _write_frames(w, range(10, 15), _always_two)

    data = _read(path)
    assert data["frame_idx"].shape[0] == 30
    assert data["frame_idx"].tolist() == [f for f in range(15) for _ in range(2)]


def test_detection_writer_schema_matches_fixture(sample_detection_zarr, tmp_path):
    # Guard against drift between the hand-rolled conftest fixture and the writer.
    writer_path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(writer_path) as w:
        _write_frames(w, range(3), _always_two)

    fixture = zarr.open_group(str(sample_detection_zarr), mode="r")
    written = zarr.open_group(str(writer_path), mode="r")
    assert set(written.array_keys()) == set(fixture.array_keys())
    for name in fixture.array_keys():
        assert written[name].dtype == fixture[name].dtype, name
        assert written[name].chunks == fixture[name].chunks, name


def test_empty_part_reads_zero_rows(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path):  # closed without any add_row
        pass

    reader = DetectionPartReader(path)
    assert reader.n_rows == 0
    assert reader.last_frame_idx is None
    assert reader.read()["frame_idx"].tolist() == []
    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["schema_version"] == 1
    assert "mask_rle" in set(root.array_keys())


def test_single_row_round_trip(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as w:
        w.add_row(7, 7000, 0, "mouse", 0.9, (0.1, 0.2, 0.3, 0.4), "rle")
    reader = DetectionPartReader(path)
    assert reader.n_rows == 1
    assert reader.last_frame_idx == 7


# --- assignment zarr (Section 13) ------------------------------------------- #
def test_assignment_writer_schema_and_round_trip(sample_assignment_zarr, tmp_path):
    # distinct path from the sample_assignment_zarr fixture
    path = tmp_path / "assign_out.zarr"
    with AssignmentPartWriter(path, buffer_size=3) as w:
        for oid in [0, 1, -1, 0, 1, 1, -1]:
            w.add_object_id(oid)

    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["schema_version"] == 1
    assert root["object_id"].dtype == np.int32
    assert root["object_id"].chunks == (10000,)
    assert root["object_id"][:].tolist() == [0, 1, -1, 0, 1, 1, -1]
    # matches the conftest fixture schema
    fixture = zarr.open_group(str(sample_assignment_zarr), mode="r")
    assert root["object_id"].dtype == fixture["object_id"].dtype
    assert root["object_id"].chunks == fixture["object_id"].chunks


def test_assignment_resume_is_identical(tmp_path):
    ids = [i % 3 - 1 for i in range(40)]  # mix of -1, 0, 1
    ref = tmp_path / "ref.zarr"
    with AssignmentPartWriter(ref) as w:
        for oid in ids:
            w.add_object_id(oid)

    part = tmp_path / "int.zarr"
    w_a = AssignmentPartWriter(part, buffer_size=5)
    for oid in ids[:20]:
        w_a.add_object_id(oid)
    w_a.flush()
    del w_a  # crash
    resumed = AssignmentPartWriter(part)
    for oid in ids[resumed.rows_written :]:
        resumed.add_object_id(oid)
    resumed.close()

    assert PartZarrReader(part).read()["object_id"].tolist() == ids
    assert PartZarrReader(ref).read()["object_id"].tolist() == ids


def _detection_reader_two_classes(tmp_path, n_frames=4):
    """2 rows/frame: ``mouse`` at index 0, ``pellet`` at index 1 (`_rows_for_frame`)."""
    path = tmp_path / "det.zarr"
    with DetectionPartWriter(path) as writer:
        _write_frames(writer, range(n_frames), _always_two)
    return DetectionPartReader(path)


def test_resolve_assignment_rows_filters_by_class(tmp_path):
    reader = _detection_reader_two_classes(tmp_path)
    assign_path = tmp_path / "assign.zarr"
    with AssignmentPartWriter(assign_path) as w:
        for _ in range(4):  # one row per "mouse" detection (the real filtered count)
            w.add_object_id(0)
    assignment = PartZarrReader(assign_path)

    raw_rows = resolve_assignment_rows(reader, assignment, "mouse")

    assert raw_rows.tolist() == [0, 2, 4, 6]  # "mouse" is always detection_idx 0


def test_resolve_assignment_rows_no_filter_is_identity(tmp_path):
    reader = _detection_reader_two_classes(tmp_path)
    assign_path = tmp_path / "assign.zarr"
    with AssignmentPartWriter(assign_path) as w:
        for _ in range(8):
            w.add_object_id(0)
    assignment = PartZarrReader(assign_path)

    assert resolve_assignment_rows(reader, assignment, None).tolist() == list(range(8))


def test_resolve_assignment_rows_shrinks_for_a_torn_store(tmp_path):
    """A crash-torn store: the assignment itself is short. Not an error — the
    existing recovery (module docstring) bounds to the shorter side."""
    reader = _detection_reader_two_classes(tmp_path)
    assign_path = tmp_path / "assign.zarr"
    with AssignmentPartWriter(assign_path) as w:
        for _ in range(2):  # fewer rows than the 4 "mouse" detections
            w.add_object_id(0)
    assignment = PartZarrReader(assign_path)

    assert resolve_assignment_rows(reader, assignment, "mouse").tolist() == [0, 2]


def test_resolve_assignment_rows_rejects_a_class_name_the_tracker_never_used(tmp_path):
    """The recorded class: a `class_name` that is not what the tracker actually
    filtered by (e.g. a track_merge run's joined record label, not its track.yaml's
    real `None`) makes the filtered row count collapse below what the assignment
    holds — this must raise, not silently resolve to an empty/wrong pairing
    (`detail/api/run_preview.py::TrackingPreview` is the consumer this guards)."""
    reader = _detection_reader_two_classes(tmp_path)
    assign_path = tmp_path / "assign.zarr"
    with AssignmentPartWriter(assign_path) as w:
        for _ in range(4):  # the real, unfiltered assignment: one per "mouse" row
            w.add_object_id(0)
    assignment = PartZarrReader(assign_path)

    with pytest.raises(ValueError, match="not the filter this run's tracker actually"):
        resolve_assignment_rows(reader, assignment, "black mouse+white mouse")


# --- embedding zarr (Section 14) -------------------------------------------- #
def test_embedding_writer_schema_and_round_trip(tmp_path):
    path = tmp_path / "track_0" / "part_001.zarr"
    path.parent.mkdir(parents=True)
    D = 8
    with EmbeddingPartWriter(path, output_dim=D, buffer_size=4) as w:
        for f in range(10):
            w.add_row(
                frame_idx=f,
                t_ns=f * 33_333_333,
                det_row=f * 2,
                embedding=np.full(D, f, dtype=np.float32),
            )

    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["schema_version"] == 1
    assert set(root.array_keys()) == {"frame_idx", "t_ns", "det_row", "embedding"}
    assert root["embedding"].dtype == np.float32
    assert root["embedding"].shape == (10, D)
    assert root["embedding"].chunks == (1000, D)
    assert root["det_row"][:].tolist() == [f * 2 for f in range(10)]
    assert np.array_equal(root["embedding"][3], np.full(D, 3, dtype=np.float32))


def test_embedding_resume_rejects_dim_mismatch(tmp_path):
    path = tmp_path / "track_0" / "part_001.zarr"
    path.parent.mkdir(parents=True)
    with EmbeddingPartWriter(path, output_dim=8) as w:
        w.add_row(0, 0, 0, np.zeros(8, dtype=np.float32))

    with pytest.raises(ValueError, match="trailing shape"):
        EmbeddingPartWriter(path, output_dim=16)


# --- tracks summary zarr (Section 13) --------------------------------------- #
def test_tracks_summary_writer(tmp_path):
    path = tmp_path / "tracks.zarr"
    with TracksSummaryWriter(path) as w:
        w.add_track(0, "mouse", 0, 100, 95, 5, 0.88)
        w.add_track(1, "mouse", 10, 120, 100, 10, 0.81)

    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["schema_version"] == 1
    assert set(root.array_keys()) == {
        "track_id",
        "class_name",
        "first_frame",
        "last_frame",
        "n_detections",
        "n_gaps",
        "mean_conf",
    }
    assert root["track_id"][:].tolist() == [0, 1]
    assert [str(x) for x in root["class_name"][:]] == ["mouse", "mouse"]
    assert root["mean_conf"].dtype == np.float32


# --------------------------------------------------------------------------- #
# Network-share replace retry (Windows/SMB): zarr commits every zarr.json and
# chunk via its own tmp.replace(dst), which intermittently raises a transient
# PermissionError (WinError 5/32) on a network share and bypasses core.atomic.
# The per-part writers route zarr's write path through core.atomic.retry_on_lock.
# --------------------------------------------------------------------------- #
def _fail_zarr_json_puts(monkeypatch, remaining: dict) -> None:
    """Patch zarr's LocalStore._put to raise a transient PermissionError on the
    first ``remaining['n']`` metadata (zarr.json) writes, then delegate. Chunk
    writes are left alone so only the metadata replace is exercised."""
    import zarr.storage._local as zl

    real_put = zl._put

    def flaky_put(path, value, exclusive=False):
        if remaining["n"] > 0 and str(path).endswith("zarr.json"):
            remaining["n"] -= 1
            raise PermissionError(5, "Access is denied")
        return real_put(path, value, exclusive=exclusive)

    monkeypatch.setattr(zl, "_put", flaky_put)
    monkeypatch.setattr("detail.core.atomic.RETRY_DELAY_S", 0.0)


def test_append_retries_transient_share_permission_error(tmp_path, monkeypatch):
    remaining = {"n": 0}
    _fail_zarr_json_puts(monkeypatch, remaining)

    path = tmp_path / "e.zarr"
    with EmbeddingPartWriter(path, output_dim=3, buffer_size=1) as w:
        w.add_row(0, 0, 0, np.array([1, 2, 3], np.float32))
        # Arm the transient failures so the next flush's resize replace hits them.
        remaining["n"] = 3
        w.add_row(1, 10, 0, np.array([4, 5, 6], np.float32))

    assert remaining["n"] == 0  # the retries actually consumed the transient locks
    data = PartZarrReader(path).read()
    assert data["frame_idx"].tolist() == [0, 1]
    assert data["embedding"].shape == (2, 3)
    assert data["embedding"][1].tolist() == [4.0, 5.0, 6.0]


def test_writer_gives_up_after_exhausting_share_retries(tmp_path, monkeypatch):
    remaining = {"n": 10_000}  # never recovers
    _fail_zarr_json_puts(monkeypatch, remaining)

    with pytest.raises(PermissionError):
        with EmbeddingPartWriter(tmp_path / "e.zarr", output_dim=2):
            pass


def test_object_ids_land_on_the_detection_rows_the_tracker_saw(tmp_path):
    """A class-filtered assignment is laid onto the part's own rows: the pellets it never saw
    read UNSEEN, the mice their tracks — never the assignment's raw row ``i`` on row ``i``.
    """
    reader = _detection_reader_two_classes(tmp_path)
    assign_path = tmp_path / "assign.zarr"
    with AssignmentPartWriter(assign_path) as w:
        for oid in (3, -1, 3, 4):  # one per mouse, which is every even detection row
            w.add_object_id(oid)

    aligned = object_ids_on_detection_rows(reader, PartZarrReader(assign_path), "mouse")

    assert aligned.tolist() == [3, UNSEEN, -1, UNSEEN, 3, UNSEEN, 4, UNSEEN]


def test_object_ids_without_a_filter_are_the_assignment_and_a_torn_tail_is_unseen(
    tmp_path,
):
    reader = _detection_reader_two_classes(tmp_path)
    assign_path = tmp_path / "assign.zarr"
    with AssignmentPartWriter(assign_path) as w:
        for oid in range(6):  # a crash left the last two rows unassigned
            w.add_object_id(oid)

    aligned = object_ids_on_detection_rows(reader, PartZarrReader(assign_path), None)

    assert aligned.tolist() == [0, 1, 2, 3, 4, 5, UNSEEN, UNSEEN]
