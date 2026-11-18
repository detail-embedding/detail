"""Stream storage: canonical CSV, manifest, alignment (Section 3)."""

import pytest

from detail.storage import recording as recording_store
from detail.storage import streams as streams_store

COLUMNS = ["t_ns", "code", "label"]


def _manifest(name="medpc", offset_ns=0):
    return {
        "name": name,
        "type": "events",
        "columns": COLUMNS,
        "importer": "medpc",
        "alignment": {"offset_ns": offset_ns, "active": "offset"},
        "raw": [],
        "files": [{"index": 0, "file": f"streams/{name}/canonical/part_001.csv"}],
        "metadata": {},
    }


def _write_canonical(recording_dir, name, rows):
    out_dir = (
        streams_store.stream_dir(recording_dir, name) / streams_store.CANONICAL_DIRNAME
    )
    return streams_store.write_canonical_csv(out_dir, COLUMNS, rows)


# -- canonical round-trip --------------------------------------------------- #
def test_canonical_round_trip_quotes_commas(recording_dir):
    rows = [
        (0, 19, "Houselight on"),
        (2_000_000_000, 25, "Time-limit reached, non-contingent reward delivery"),
    ]
    _write_canonical(recording_dir, "medpc", rows)
    columns, read = streams_store.read_canonical(recording_dir, "medpc")
    assert columns == COLUMNS
    # t_ns and code come back as int; the comma label survives CSV quoting
    assert read == rows


def test_canonical_empty_stream(recording_dir):
    """Schema edge: a stream with zero events is a valid header-only CSV."""
    _write_canonical(recording_dir, "empty", [])
    columns, read = streams_store.read_canonical(recording_dir, "empty")
    assert columns == COLUMNS
    assert read == []


def test_canonical_single_event(recording_dir):
    """Schema edge: a single-row stream."""
    _write_canonical(recording_dir, "one", [(5, 1, "press")])
    _columns, read = streams_store.read_canonical(recording_dir, "one")
    assert read == [(5, 1, "press")]


def test_read_canonical_missing_raises(recording_dir):
    with pytest.raises(FileNotFoundError):
        streams_store.read_canonical(recording_dir, "nope")


# -- manifest list / append / delete ---------------------------------------- #
def test_append_list_delete(recording_dir):
    assert streams_store.list_streams(recording_dir) == []
    streams_store.append_stream_to_recording_yaml(recording_dir, _manifest("a"))
    streams_store.append_stream_to_recording_yaml(recording_dir, _manifest("b"))
    names = [s["name"] for s in streams_store.list_streams(recording_dir)]
    assert names == ["a", "b"]

    # delete prunes the entry (and its dir, if present)
    sdir = streams_store.stream_dir(recording_dir, "a")
    (sdir / "canonical").mkdir(parents=True)
    (sdir / "canonical" / "part_001.csv").write_text(
        "t_ns,code,label\n", encoding="utf-8"
    )
    streams_store.delete_stream(recording_dir, "a")
    assert [s["name"] for s in streams_store.list_streams(recording_dir)] == ["b"]
    assert not sdir.exists()


def test_delete_missing_is_idempotent(recording_dir):
    streams_store.delete_stream(recording_dir, "ghost")  # no raise


def test_append_duplicate_name_raises(recording_dir):
    streams_store.append_stream_to_recording_yaml(recording_dir, _manifest("dup"))
    with pytest.raises(ValueError, match="already exists"):
        streams_store.append_stream_to_recording_yaml(recording_dir, _manifest("dup"))


# -- name validation -------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["", "with space", "a/b", "name.csv", "café"])
def test_validate_stream_name_rejects(bad):
    with pytest.raises(ValueError):
        streams_store.validate_stream_name(bad)


@pytest.mark.parametrize("good", ["medpc", "heart_rate", "accel-2", "S1"])
def test_validate_stream_name_accepts(good):
    streams_store.validate_stream_name(good)  # no raise


# -- alignment -------------------------------------------------------------- #
def test_set_stream_offset(recording_dir):
    streams_store.append_stream_to_recording_yaml(recording_dir, _manifest("s"))
    streams_store.set_stream_offset(recording_dir, "s", -5_000_000_000)
    entry = streams_store.list_streams(recording_dir)[0]
    assert entry["alignment"]["offset_ns"] == -5_000_000_000
    assert entry["alignment"]["active"] == "offset"
    # persisted to disk
    on_disk = recording_store.read_recording(recording_dir)["streams"][0]
    assert on_disk["alignment"]["offset_ns"] == -5_000_000_000


def test_set_stream_offset_unknown_raises(recording_dir):
    with pytest.raises(ValueError, match="unknown stream"):
        streams_store.set_stream_offset(recording_dir, "nope", 0)


def test_apply_alignment_offset():
    assert streams_store.apply_alignment(10, _manifest(offset_ns=0)) == 10
    assert streams_store.apply_alignment(10, _manifest(offset_ns=5)) == 15
    # negative offset (stream started before the video)
    assert (
        streams_store.apply_alignment(
            10_000_000_000, _manifest(offset_ns=-5_000_000_000)
        )
        == 5_000_000_000
    )


def test_apply_alignment_correction_table_not_implemented():
    manifest = {"alignment": {"offset_ns": 0, "active": "table"}}
    with pytest.raises(NotImplementedError):
        streams_store.apply_alignment(0, manifest)


# -- typed read + shape + role (behavior interval streams, Section 3) -------- #
_BEHAVIOR_COLUMNS = [
    "t_ns",
    "end_t_ns",
    "label",
    "subject",
    "start_frame",
    "end_frame",
]
_BEHAVIOR_DTYPES = {
    "t_ns": "int64",
    "end_t_ns": "int64",
    "label": "str",
    "subject": "int32",
    "start_frame": "int32",
    "end_frame": "int32",
}


def test_typed_read_parses_by_dtype(recording_dir):
    out_dir = (
        streams_store.stream_dir(recording_dir, "gt") / streams_store.CANONICAL_DIRNAME
    )
    rows = [(100, 300, "groom", 1, 1, 3), (400, 500, "rear", None, 4, 5)]
    streams_store.write_canonical_csv(out_dir, _BEHAVIOR_COLUMNS, rows)
    columns, read = streams_store.read_canonical(
        recording_dir, "gt", dtypes=_BEHAVIOR_DTYPES
    )
    assert columns == _BEHAVIOR_COLUMNS
    # end_t_ns and *_frame come back int; an empty subject cell is None
    assert read == [(100, 300, "groom", 1, 1, 3), (400, 500, "rear", None, 4, 5)]


def test_float_dtype_round_trips(recording_dir):
    """A continuous stream's float value channels survive write -> typed read."""
    columns = ["t_ns", "bpm", "temp"]
    dtypes = {"t_ns": "int64", "bpm": "float", "temp": "float64"}
    out_dir = (
        streams_store.stream_dir(recording_dir, "sig") / streams_store.CANONICAL_DIRNAME
    )
    rows = [(0, 72.5, 36.6), (500_000_000, 73.0, None)]
    streams_store.write_canonical_csv(out_dir, columns, rows)
    read_columns, read = streams_store.read_canonical(
        recording_dir, "sig", dtypes=dtypes
    )
    assert read_columns == columns
    # floats parse back as float; an empty cell is None
    assert read == [(0, 72.5, 36.6), (500_000_000, 73.0, None)]
    assert isinstance(read[0][1], float)


def test_default_dtypes_backcompat():
    dt = streams_store.default_dtypes(
        ["t_ns", "end_t_ns", "code", "label", "start_frame"]
    )
    assert dt["t_ns"] == "int64"
    assert dt["end_t_ns"] == "int64"
    assert dt["code"] == "int64"
    assert dt["start_frame"] == "int64"
    assert dt["label"] == "str"


def test_write_canonical_is_atomic(recording_dir):
    out_dir = (
        streams_store.stream_dir(recording_dir, "a") / streams_store.CANONICAL_DIRNAME
    )
    streams_store.write_canonical_csv(out_dir, COLUMNS, [(0, 1, "x")])
    # no leftover temp file beside the canonical CSV
    assert not (out_dir / (streams_store.CANONICAL_FILENAME + ".tmp")).exists()
    assert (out_dir / streams_store.CANONICAL_FILENAME).is_file()


def test_stream_shape_prefers_shape_then_type():
    assert streams_store.stream_shape({"shape": "interval"}) == "interval"
    assert streams_store.stream_shape({"type": "events"}) == "event"
    assert streams_store.stream_shape({"type": "sampled"}) == "continuous"
    assert streams_store.stream_shape({}) == "event"


def test_streams_by_role(recording_dir):
    streams_store.append_stream_to_recording_yaml(
        recording_dir, {**_manifest("medpc"), "role": None}
    )
    behavior = {
        "name": "gt",
        "type": "events",
        "shape": "interval",
        "role": "behavior",
        "columns": _BEHAVIOR_COLUMNS,
        "importer": "behavior_csv",
        "alignment": {"offset_ns": 0, "active": "offset"},
        "raw": [],
        "files": [{"index": 0, "file": "streams/gt/canonical/part_001.csv"}],
        "metadata": {},
    }
    streams_store.append_stream_to_recording_yaml(recording_dir, behavior)
    found = streams_store.streams_by_role(recording_dir, "behavior")
    assert [s["name"] for s in found] == ["gt"]


@pytest.mark.parametrize("clock", ["video_frames", "video_seconds"])
def test_set_stream_offset_refuses_a_video_anchored_stream(recording_dir, clock):
    """The storage door every surface reaches (API, CLI, GUI) refuses it, so none has to
    remember to. Stored, the preview drew the labels late while scoring read them unshifted.
    """
    manifest = {**_manifest("behavior"), "clock": clock, "role": "behavior"}
    streams_store.append_stream_to_recording_yaml(recording_dir, manifest)
    with pytest.raises(ValueError, match="already the video's own"):
        streams_store.set_stream_offset(recording_dir, "behavior", 5_000_000_000)
    on_disk = recording_store.read_recording(recording_dir)["streams"][0]
    assert on_disk["alignment"]["offset_ns"] == 0


def test_set_stream_offset_still_aligns_an_external_clock(recording_dir):
    manifest = {**_manifest("medpc"), "clock": "stream_seconds"}
    streams_store.append_stream_to_recording_yaml(recording_dir, manifest)
    streams_store.set_stream_offset(recording_dir, "medpc", 3_480_000_000)
    assert streams_store.stream_offset_ns(
        streams_store.list_streams(recording_dir)[0]
    ) == (3_480_000_000)


def test_stream_offset_ns_reads_the_alignment_block_only():
    """A top-level ``offset_ns`` is not where the offset lives; reading it reported 0."""
    assert streams_store.stream_offset_ns(_manifest(offset_ns=7)) == 7
    assert streams_store.stream_offset_ns({"offset_ns": 7}) == 0
    assert streams_store.stream_offset_ns({"alignment": None}) == 0


def test_the_offset_refusal_is_a_user_input_error(recording_dir):
    from detail.core.errors import UserInputError

    manifest = {**_manifest("behavior"), "clock": "video_frames", "role": "behavior"}
    streams_store.append_stream_to_recording_yaml(recording_dir, manifest)
    with pytest.raises(UserInputError):
        streams_store.set_stream_offset(recording_dir, "behavior", 5_000_000_000)
