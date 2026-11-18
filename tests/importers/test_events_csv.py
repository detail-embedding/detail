"""Generic timestamped-events CSV importer (Section 3).

Covers the happy path with and without a ``code`` column, each named time column,
``can_import`` sniffing, and every strict-validation reject path.
"""

import pytest

from detail.importers.events_csv import EventsCsvImporter


def _write(tmp_path, text, name="events.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_declares_shape_role_and_clock():
    imp = EventsCsvImporter()
    assert imp.importer_id == "events_csv"
    assert imp.display_name == "Events CSV"
    assert imp.stream_type == "events"
    assert imp.role is None
    assert imp.aux_inputs == []


def test_seconds_clock_with_code(tmp_path):
    p = _write(tmp_path, "seconds,code,label\n1.0,5,press\n2.0,,release\n")
    result = EventsCsvImporter().import_stream(p, tmp_path, {})
    assert result.type == "events"
    assert result.shape == "event"
    assert result.clock == "stream_seconds"
    assert result.columns == ["t_ns", "code", "label"]
    assert result.frame_columns is None
    assert result.dtypes == {"t_ns": "int64", "code": "int64", "label": "str"}
    # a missing code cell is None
    assert result.rows == [
        (1_000_000_000, 5, "press"),
        (2_000_000_000, None, "release"),
    ]


def test_without_code_column(tmp_path):
    p = _write(tmp_path, "seconds,label\n1.0,press\n")
    result = EventsCsvImporter().import_stream(p, tmp_path, {})
    assert result.columns == ["t_ns", "code", "label"]
    assert result.rows == [(1_000_000_000, None, "press")]


def test_t_ns_clock_passes_through(tmp_path):
    p = _write(tmp_path, "t_ns,code,label\n5,1,press\n")
    result = EventsCsvImporter().import_stream(p, tmp_path, {})
    assert result.clock == "stream_seconds"
    assert result.rows == [(5, 1, "press")]


def test_frame_clock_leaves_t_ns_unresolved(tmp_path):
    p = _write(tmp_path, "frame,label\n10,press\n")
    result = EventsCsvImporter().import_stream(p, tmp_path, {})
    assert result.clock == "video_frames"
    assert result.columns == ["t_ns", "code", "label", "frame"]
    assert result.frame_columns == {"t_ns": "frame"}
    assert result.dtypes["frame"] == "int32"
    # t_ns is None (the import layer resolves it from the recording's map)
    assert result.rows == [(None, None, "press", 10)]


def test_can_import_accepts_and_rejects(tmp_path):
    good = _write(tmp_path, "seconds,label\n1.0,press\n", "g.csv")
    continuous = _write(tmp_path, "seconds,bpm\n0.0,72\n", "c.csv")
    no_time = _write(tmp_path, "code,label\n5,press\n", "n.csv")
    imp = EventsCsvImporter()
    assert imp.can_import(good)
    assert not imp.can_import(continuous)  # no label column -> continuous, not events
    assert not imp.can_import(no_time)


def test_no_label_column_suggests_timeseries_importer(tmp_path):
    p = _write(tmp_path, "seconds,code\n1.0,5\n")
    with pytest.raises(ValueError, match="Time-series CSV importer"):
        EventsCsvImporter().import_stream(p, tmp_path, {})


def test_no_time_column_raises(tmp_path):
    p = _write(tmp_path, "code,label\n5,press\n")
    with pytest.raises(ValueError, match="needs a time column named one of"):
        EventsCsvImporter().import_stream(p, tmp_path, {})


def test_non_integer_code_raises(tmp_path):
    p = _write(tmp_path, "seconds,code,label\n1.0,x,press\n")
    with pytest.raises(ValueError, match="column 'code'.*not an integer"):
        EventsCsvImporter().import_stream(p, tmp_path, {})


def test_non_integer_frame_raises(tmp_path):
    p = _write(tmp_path, "frame,label\nx,press\n")
    with pytest.raises(ValueError, match="not an integer"):
        EventsCsvImporter().import_stream(p, tmp_path, {})


def test_empty_label_raises(tmp_path):
    p = _write(tmp_path, "seconds,label\n1.0,\n")
    with pytest.raises(ValueError, match="empty label"):
        EventsCsvImporter().import_stream(p, tmp_path, {})


def test_header_only_raises(tmp_path):
    p = _write(tmp_path, "seconds,label\n")
    with pytest.raises(ValueError, match="no data rows"):
        EventsCsvImporter().import_stream(p, tmp_path, {})


# -- times beyond int64; the clock the file resolves to (#30 findings 8, 10) ---------------- #
def test_epoch_milliseconds_in_a_seconds_column_are_refused_with_the_row(tmp_path):
    """Stored, it was t_ns 1788000000000000000000 and the preview died on OverflowError."""
    p = _write(tmp_path, "seconds,label\n1.0,cue\n1788000000000,cue\n")
    with pytest.raises(ValueError) as err:
        EventsCsvImporter().import_stream(p, tmp_path, {})
    assert str(err.value).startswith(
        "row 3, column 'seconds': 1788000000000.0 seconds is too large to be a time"
    )


def test_a_t_ns_beyond_int64_is_refused(tmp_path):
    p = _write(tmp_path, "t_ns,label\n9223372036854775808,cue\n")
    with pytest.raises(
        ValueError, match=r"row 2, column 't_ns': .* beyond the int64 range"
    ):
        EventsCsvImporter().import_stream(p, tmp_path, {})


@pytest.mark.parametrize(
    "header,clock",
    [
        ("frame,label", "video_frames"),
        ("seconds,label", "stream_seconds"),
        ("t_ns,label", "stream_seconds"),
    ],
)
def test_resolve_clock_follows_the_time_column(tmp_path, header, clock):
    p = _write(tmp_path, f"{header}\n1,cue\n")
    importer = EventsCsvImporter()
    assert importer.resolve_clock(p) == clock
    assert (
        importer.import_stream(p, tmp_path, {}).clock == clock
    )  # and agrees with import
