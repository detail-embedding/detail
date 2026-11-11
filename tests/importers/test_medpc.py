"""MED-PC importer parsing + canonical conversion (Section 3)."""

import pytest

from detail.importers.medpc import (
    MedPCImporter,
    parse_event_map,
    parse_medpc_data,
)

# A minimal MED-PC data dump: header, scalars (G, H), and parallel E/T arrays.
# Events: two at t=0 (simultaneous), one at 1.5s, one at 2.0s (comma label),
# a termination at 10s, and a trailing sentinel pair that must be dropped.
DATA_TXT = """\
Start Date: 02/16/26
End Date: 02/16/26
Subject: TEST1
Experiment: UnitTest
Box: 4
Start Time: 14:00:38
End Time: 14:10:00
MSN: Test_MSN
G:       5.000
H:      10.000
E:
     0:       19.000       11.000        2.000       25.000      100.000
     5:     -987.987
T:
     0:        0.000        0.000        1.500        2.000       10.000
     5:     -987.987
"""

PROGRAM_MPC = """\
\\  Some preamble comment
\\
\\  Event stamps ( array E )
\\   2   = Left Lev press
\\   11  = Houselight on
\\   19  = Timeout ends and/or drug available at START
\\   25  = Time-limit reached, non-contingent reward delivery
\\   100 = Recording termination
\\
\\  Variables
\\   A(0)  = Dummy variable for menu display
"""


def _write_files(tmp_path, data=DATA_TXT, program=PROGRAM_MPC):
    data_path = tmp_path / "subject.txt"
    prog_path = tmp_path / "task.MPC"
    data_path.write_text(data, encoding="utf-8")
    prog_path.write_text(program, encoding="utf-8")
    return data_path, prog_path


def test_parse_medpc_data_header_scalars_arrays():
    header, scalars, arrays = parse_medpc_data(DATA_TXT)
    assert header["Subject"] == "TEST1"
    assert header["MSN"] == "Test_MSN"
    assert header["Start Time"] == "14:00:38"  # value's own colons preserved
    assert scalars["G"] == 5.0
    assert scalars["H"] == 10.0
    assert arrays["E"] == [19.0, 11.0, 2.0, 25.0, 100.0, -987.987]
    assert arrays["T"] == [0.0, 0.0, 1.5, 2.0, 10.0, -987.987]


def test_parse_event_map():
    event_map = parse_event_map(PROGRAM_MPC)
    assert event_map[2] == "Left Lev press"
    assert event_map[100] == "Recording termination"
    # the comma in the label is preserved verbatim
    assert event_map[25] == "Time-limit reached, non-contingent reward delivery"
    # parsing stops at the next section — Variables (A(0) = ...) is not an event
    assert set(event_map) == {2, 11, 19, 25, 100}


def test_parse_event_map_missing_block_raises():
    with pytest.raises(ValueError, match="Event stamps"):
        parse_event_map("\\  No event block here\n\\  Variables\n")


def test_import_stream_rows_and_sentinel_filtering(tmp_path):
    data_path, prog_path = _write_files(tmp_path)
    result = MedPCImporter().import_stream(data_path, tmp_path, {"program": prog_path})
    assert result.type == "events"
    assert result.columns == ["t_ns", "code", "label"]
    # the sentinel (-987.987) pair is dropped → 5 rows, not 6
    assert result.rows == [
        (0, 19, "Timeout ends and/or drug available at START"),
        (0, 11, "Houselight on"),
        (1_500_000_000, 2, "Left Lev press"),
        (2_000_000_000, 25, "Time-limit reached, non-contingent reward delivery"),
        (10_000_000_000, 100, "Recording termination"),
    ]


def test_import_stream_metadata(tmp_path):
    data_path, prog_path = _write_files(tmp_path)
    result = MedPCImporter().import_stream(data_path, tmp_path, {"program": prog_path})
    assert result.metadata["subject"] == "TEST1"
    assert result.metadata["msn"] == "Test_MSN"
    # the full code→label map is kept for legend / re-mapping, keyed by str
    assert result.metadata["event_map"]["100"] == "Recording termination"


def test_import_stream_unknown_code_falls_back(tmp_path):
    # code 7 is in the data but absent from the program's event map
    data = DATA_TXT.replace("100.000", "7.000")
    data_path, prog_path = _write_files(tmp_path, data=data)
    result = MedPCImporter().import_stream(data_path, tmp_path, {"program": prog_path})
    assert result.rows[-1] == (10_000_000_000, 7, "event_7")


def test_import_stream_missing_program_raises(tmp_path):
    data_path, _ = _write_files(tmp_path)
    with pytest.raises(ValueError, match="program"):
        MedPCImporter().import_stream(data_path, tmp_path, {})


def test_import_stream_no_event_array_raises(tmp_path):
    # a data file with no E/T arrays
    data = "Start Date: 02/16/26\nMSN: x\nG:       0.000\n"
    data_path, prog_path = _write_files(tmp_path, data=data)
    with pytest.raises(ValueError, match="E .events. and T .times. arrays"):
        MedPCImporter().import_stream(data_path, tmp_path, {"program": prog_path})


def test_can_import(tmp_path):
    data_path, _ = _write_files(tmp_path)
    importer = MedPCImporter()
    assert importer.can_import(data_path) is True
    other = tmp_path / "other.txt"
    other.write_text("not a med-pc file\n", encoding="utf-8")
    assert importer.can_import(other) is False
