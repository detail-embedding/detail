"""Behavior-label interval-CSV importer (Section 3)."""

import pytest

from detail.importers.behavior import BehaviorCsvImporter

_COLUMNS = ["t_ns", "end_t_ns", "label", "subject", "start_frame", "end_frame"]


def _write(tmp_path, text, name="labels.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_declares_role_and_clock():
    imp = BehaviorCsvImporter()
    assert imp.importer_id == "behavior_csv"
    assert imp.role == "behavior"
    assert imp.clock == "video_frames"


def test_can_import_accepts_and_rejects(tmp_path):
    good = _write(tmp_path, "start_frame,end_frame,label\n10,20,groom\n", "g.csv")
    bad = _write(tmp_path, "a,b,c\n1,2,3\n", "b.csv")
    assert BehaviorCsvImporter().can_import(good)
    assert not BehaviorCsvImporter().can_import(bad)


def test_import_stream_parses(tmp_path):
    p = _write(
        tmp_path,
        "start_frame,end_frame,label,subject\n10,20,groom,1\n30,40,rear,\n",
    )
    result = BehaviorCsvImporter().import_stream(p, tmp_path, {})
    assert result.shape == "interval"
    assert result.clock == "video_frames"
    assert result.columns == _COLUMNS
    assert result.frame_columns == {"t_ns": "start_frame", "end_t_ns": "end_frame"}
    assert result.dtypes["subject"] == "int32"
    # rows are in frame space (t_ns unresolved); subject empty -> None
    assert result.rows == [
        (None, None, "groom", 1, 10, 20),
        (None, None, "rear", None, 30, 40),
    ]


def test_same_label_overlaps_left_untouched(tmp_path):
    p = _write(
        tmp_path,
        "start_frame,end_frame,label\n10,50,groom\n30,70,groom\n",
    )
    rows = BehaviorCsvImporter().import_stream(p, tmp_path, {}).rows
    assert len(rows) == 2  # both overlapping bouts preserved, no merge


def test_missing_required_column(tmp_path):
    p = _write(tmp_path, "start_frame,end_frame\n10,20\n")
    with pytest.raises(ValueError, match="missing required column"):
        BehaviorCsvImporter().import_stream(p, tmp_path, {})


def test_non_integer_frame(tmp_path):
    p = _write(tmp_path, "start_frame,end_frame,label\nx,20,groom\n")
    with pytest.raises(ValueError, match="not an integer"):
        BehaviorCsvImporter().import_stream(p, tmp_path, {})


def test_empty_label(tmp_path):
    p = _write(tmp_path, "start_frame,end_frame,label\n10,20,\n")
    with pytest.raises(ValueError, match="empty label"):
        BehaviorCsvImporter().import_stream(p, tmp_path, {})


def test_end_before_start(tmp_path):
    p = _write(tmp_path, "start_frame,end_frame,label\n30,10,groom\n")
    with pytest.raises(ValueError, match="before start_frame"):
        BehaviorCsvImporter().import_stream(p, tmp_path, {})


def test_metadata_covered_and_types(tmp_path):
    p = _write(
        tmp_path,
        "start_frame,end_frame,label\n10,20,groom\n30,30,bite\n40,50,groom\n",
    )
    md = BehaviorCsvImporter().import_stream(p, tmp_path, {}).metadata
    assert md["covered_behaviors"] == ["bite", "groom"]
    # groom has duration -> state; bite is all zero-length -> point
    assert md["behavior_types"] == {"groom": "state", "bite": "point"}
