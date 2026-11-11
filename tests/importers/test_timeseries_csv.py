"""Generic continuous-signal CSV importer (Section 3).

Covers the three named time columns (``seconds`` / ``frame`` / ``t_ns``), the
value-channel layout, ``sample_rate_hz`` derivation, ``can_import`` sniffing, and
every strict-validation reject path.
"""

import pytest

from detail.importers.timeseries_csv import TimeseriesCsvImporter


def _write(tmp_path, text, name="signal.csv"):
    p = tmp_path / name
    # Explicit: these fixtures include a BOM, which the locale codec on Windows
    # (cp1252) cannot encode — and the importer decodes utf-8-sig regardless of OS.
    p.write_text(text, encoding="utf-8")
    return p


def test_declares_shape_role_and_clock():
    imp = TimeseriesCsvImporter()
    assert imp.importer_id == "timeseries_csv"
    assert imp.display_name == "Time-series CSV (sensor data)"
    assert imp.stream_type == "sampled"
    assert imp.role is None
    assert imp.aux_inputs == []


def test_seconds_clock_parses(tmp_path):
    p = _write(tmp_path, "seconds,bpm,temp\n0.0,72,36.5\n0.5,73,36.6\n")
    result = TimeseriesCsvImporter().import_stream(p, tmp_path, {})
    assert result.type == "sampled"
    assert result.shape == "continuous"
    assert result.clock == "stream_seconds"
    assert result.columns == ["t_ns", "bpm", "temp"]
    assert result.frame_columns is None
    assert result.dtypes == {"t_ns": "int64", "bpm": "float", "temp": "float"}
    assert result.rows == [(0, 72.0, 36.5), (500_000_000, 73.0, 36.6)]
    # median inter-sample dt is 0.5s -> 2 Hz
    assert result.sample_rate_hz == 2.0


def test_t_ns_clock_passes_through(tmp_path):
    p = _write(tmp_path, "t_ns,bpm\n0,72\n1000000000,73\n")
    result = TimeseriesCsvImporter().import_stream(p, tmp_path, {})
    assert result.clock == "stream_seconds"
    assert result.columns == ["t_ns", "bpm"]
    assert result.rows == [(0, 72.0), (1_000_000_000, 73.0)]
    assert result.sample_rate_hz == 1.0


def test_frame_clock_leaves_t_ns_unresolved(tmp_path):
    p = _write(tmp_path, "frame,bpm\n0,72\n1,73\n")
    result = TimeseriesCsvImporter().import_stream(p, tmp_path, {})
    assert result.clock == "video_frames"
    assert result.columns == ["t_ns", "frame", "bpm"]
    assert result.frame_columns == {"t_ns": "frame"}
    assert result.dtypes == {"t_ns": "int64", "frame": "int32", "bpm": "float"}
    # t_ns is None (the import layer resolves it from the recording's map)
    assert result.rows == [(None, 0, 72.0), (None, 1, 73.0)]
    # a frame-clocked stream has no import-time sample rate
    assert result.sample_rate_hz is None


def test_case_insensitive_time_column_and_bom(tmp_path):
    p = _write(tmp_path, "﻿Seconds,bpm\n0.0,72\n1.0,73\n")
    result = TimeseriesCsvImporter().import_stream(p, tmp_path, {})
    assert result.columns == ["t_ns", "bpm"]
    assert result.rows == [(0, 72.0), (1_000_000_000, 73.0)]


def test_single_row_has_no_sample_rate(tmp_path):
    p = _write(tmp_path, "seconds,bpm\n0.0,72\n")
    result = TimeseriesCsvImporter().import_stream(p, tmp_path, {})
    assert result.rows == [(0, 72.0)]
    assert result.sample_rate_hz is None


def test_can_import_accepts_and_rejects(tmp_path):
    good = _write(tmp_path, "seconds,bpm\n0.0,72\n", "g.csv")
    events = _write(tmp_path, "seconds,label\n0.0,press\n", "e.csv")
    no_time = _write(tmp_path, "a,b\n1,2\n", "n.csv")
    non_numeric = _write(tmp_path, "seconds,note\n0.0,hello\n", "x.csv")
    imp = TimeseriesCsvImporter()
    assert imp.can_import(good)
    assert not imp.can_import(events)  # has a label column -> events, not continuous
    assert not imp.can_import(no_time)
    assert not imp.can_import(non_numeric)


def test_no_time_column_raises(tmp_path):
    p = _write(tmp_path, "a,bpm\n1,72\n")
    with pytest.raises(ValueError, match="needs a time column named one of"):
        TimeseriesCsvImporter().import_stream(p, tmp_path, {})


def test_two_time_columns_raise(tmp_path):
    p = _write(tmp_path, "seconds,t_ns,bpm\n0.0,0,72\n")
    with pytest.raises(ValueError, match="more than one time column"):
        TimeseriesCsvImporter().import_stream(p, tmp_path, {})


def test_empty_file_raises(tmp_path):
    p = _write(tmp_path, "")
    with pytest.raises(ValueError, match="empty"):
        TimeseriesCsvImporter().import_stream(p, tmp_path, {})


def test_header_only_raises(tmp_path):
    p = _write(tmp_path, "seconds,bpm\n")
    with pytest.raises(ValueError, match="no data rows"):
        TimeseriesCsvImporter().import_stream(p, tmp_path, {})


def test_no_value_columns_raises(tmp_path):
    p = _write(tmp_path, "seconds\n0.0\n1.0\n")
    with pytest.raises(ValueError, match="no value columns"):
        TimeseriesCsvImporter().import_stream(p, tmp_path, {})


def test_non_numeric_value_cell_raises(tmp_path):
    p = _write(tmp_path, "seconds,bpm\n0.0,72\n1.0,oops\n")
    with pytest.raises(ValueError, match="row 3, column 'bpm'.*not a number"):
        TimeseriesCsvImporter().import_stream(p, tmp_path, {})


def test_label_column_suggests_events_importer(tmp_path):
    p = _write(tmp_path, "seconds,bpm,label\n0.0,72,press\n")
    with pytest.raises(ValueError, match="Events CSV importer"):
        TimeseriesCsvImporter().import_stream(p, tmp_path, {})
