"""MED-PC importer parsing + canonical conversion (Section 3).

**The class (#30): a file holding several sessions is flattened into one recording's stream.**
MED-PC can save every box of a day into one data file, one ``Start Date:`` block each. The parser
kept a dict per array letter and overwrote it on every block, so the last box won: a recording got
another animal's events and that animal's ``Subject:`` in its metadata, with nothing said. The
same file layout also hid two smaller wrongs: unsealed array space printed as ``0.000`` became
``event_0`` rows at ``t = 0``, and a program logging ``T`` in ticks imported every time 100x late.

**The rule.** A several-session file is refused until ``MedpcSessionConfig`` picks one (never
merged, never cut to the last block), and the refusal lists every session; a trailing
``(code 0, time 0)`` run is dropped and counted; a ``T`` past ``H`` is refused. The BORIS twin of
this class (observations and subjects) is ``tests/importers/test_boris.py``.
"""

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


# -- several sessions in one file, unused slots, ticks (#30 findings 3, 6, 12) --------------- #
def _session(subject, box, codes, times, h=10.0, start="14:00:38"):
    def row(values):
        return "".join(f"{v:>13.3f}" for v in values)

    return (
        f"\n\nStart Date: 02/16/26\nSubject: {subject}\nBox: {box}\n"
        f"Start Time: {start}\nMSN: Test_MSN\nH: {h:>10.3f}\n"
        f"E:\n     0:{row(codes)}\nT:\n     0:{row(times)}\n"
    )


_TWO_BOXES = (
    "File: C:\\MED-PC\\Data\\day.txt\n"
    + _session("RAT1", "1", [2, 100], [1.0, 10.0])
    + _session("RAT2", "2", [2, 100], [0.5, 3.0], start="14:00:41")
)


def _import(tmp_path, data, config=None):
    data_path, prog_path = _write_files(tmp_path, data=data)
    return MedPCImporter().import_stream(
        data_path, tmp_path, {"program": prog_path}, config
    )


def test_a_file_of_several_sessions_is_refused_naming_each(tmp_path):
    """Keeping the last block gave a recording another animal's events, with no word."""
    with pytest.raises(ValueError) as err:
        _import(tmp_path, _TWO_BOXES)
    assert str(err.value) == (
        "this MED-PC file holds 2 sessions (1: Subject RAT1, box 1, 02/16/26 14:00:38; "
        "2: Subject RAT2, box 2, 02/16/26 14:00:41). A recording is one session, so choose "
        "one by subject, box or session number; they are never merged"
    )


@pytest.mark.parametrize(
    "config,subject,rows",
    [
        ({"subject": "RAT1"}, "RAT1", [(1_000_000_000, 2), (10_000_000_000, 100)]),
        ({"box": "2"}, "RAT2", [(500_000_000, 2), (3_000_000_000, 100)]),
        ({"session": 1}, "RAT1", [(1_000_000_000, 2), (10_000_000_000, 100)]),
    ],
)
def test_a_selector_imports_exactly_the_chosen_session(tmp_path, config, subject, rows):
    result = _import(tmp_path, _TWO_BOXES, config)
    assert [(t, code) for t, code, _label in result.rows] == rows
    assert result.metadata["subject"] == subject
    assert result.metadata["sessions_in_file"] == 2
    assert result.metadata["session"] == (1 if subject == "RAT1" else 2)


def test_a_selector_matching_nothing_is_refused(tmp_path):
    with pytest.raises(
        ValueError, match=r"no session in this MED-PC file matches \{'box': '9'\}"
    ):
        _import(tmp_path, _TWO_BOXES, {"box": "9"})


def test_a_one_session_file_imports_unchanged_with_no_selector(tmp_path):
    result = _import(tmp_path, DATA_TXT)
    assert len(result.rows) == 5
    assert "session" not in result.metadata  # nothing was chosen


def test_unused_zero_slots_after_the_last_event_are_dropped_and_counted(tmp_path):
    result = _import(tmp_path, _session("RAT1", "1", [2, 100, 0, 0], [1.0, 2.0, 0, 0]))
    assert result.rows == [
        (1_000_000_000, 2, "Left Lev press"),
        (2_000_000_000, 100, "Recording termination"),
    ]
    assert result.metadata["notes"] == [
        "dropped 2 unused (code 0, time 0) slot(s) after the last event"
    ]


def test_a_code_zero_event_between_events_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"E\(1\) is event code 0 at 2s"):
        _import(tmp_path, _session("RAT1", "1", [2, 0, 100], [1.0, 2.0, 3.0]))


def test_times_past_the_session_length_are_refused_as_ticks(tmp_path):
    """T logged in 10 ms ticks imported 100x too late, silently."""
    with pytest.raises(
        ValueError, match=r"T\(1\) is 1000, past the session length H = 10s"
    ):
        _import(tmp_path, _session("RAT1", "1", [2, 100], [100, 1000], h=10.0))


def test_a_time_beyond_int64_is_refused_naming_the_slot(tmp_path):
    with pytest.raises(ValueError, match=r"MED-PC T\(0\): .* too large to be a time"):
        _import(
            tmp_path, "Start Date: x\nMSN: x\nE:\n     0:  2.000\nT:\n     0:  1e19\n"
        )


def test_list_sessions_describes_each_block_as_the_refusal_does(tmp_path):
    from detail.importers.medpc import list_sessions

    data_path, _ = _write_files(tmp_path, data=_TWO_BOXES)
    assert [s["label"] for s in list_sessions(data_path)] == [
        "1: Subject RAT1, box 1, 02/16/26 14:00:38",
        "2: Subject RAT2, box 2, 02/16/26 14:00:41",
    ]
