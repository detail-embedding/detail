"""Tests for the clip-query result CSV schema (detail/storage/clip_query_csv.py)."""

import csv

import pytest

from detail.storage import clip_query_csv as cc


def _rows():
    return [
        cc.ClipRow("query", 0, "s1", 0, "emb_01", 1000, 6000, 30, 180, 0.0),
        cc.ClipRow("match", 1, "s2", 0, "emb_02", 9000, 14000, 270, 420, 0.031),
    ]


def test_round_trip(tmp_path):
    path = tmp_path / "matches.csv"
    cc.write_clip_csv(path, _rows())
    back = cc.read_clip_csv(path)
    assert back == _rows()


def test_length_is_derived_from_frame_span():
    row = cc.ClipRow("match", 1, "s1", 0, "emb_01", 0, 10, 100, 149, 0.1)
    assert row.length == 50


def test_t_ns_columns_are_written(tmp_path):
    path = tmp_path / "matches.csv"
    cc.write_clip_csv(path, _rows())
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert "start_t_ns" in header and "end_t_ns" in header
    assert header[0] == "query_id" and "query_name" in header and "kind" in header


def test_kind_column_distinguishes_query_and_match(tmp_path):
    path = tmp_path / "matches.csv"
    cc.write_clip_csv(path, _rows())
    back = cc.read_clip_csv(path)
    assert [r.kind for r in back] == ["query", "match"]


def test_write_is_atomic_no_tmp_left(tmp_path):
    path = tmp_path / "matches.csv"
    cc.write_clip_csv(path, _rows())
    assert path.exists()
    assert not (tmp_path / "matches.csv.tmp").exists()


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "rank", "score"])  # missing identifying/span columns
        w.writerow(["match", 1, 0.1])
    with pytest.raises(ValueError, match="missing columns"):
        cc.read_clip_csv(path)


def test_query_columns_round_trip(tmp_path):
    path = tmp_path / "matches.csv"
    rows = [
        cc.ClipRow(
            "query",
            0,
            "s1",
            0,
            "emb_01",
            1000,
            6000,
            30,
            180,
            0.0,
            query_id="q02",
            query_name="groom",
        ),
        cc.ClipRow(
            "match",
            1,
            "s2",
            0,
            "emb_01",
            9000,
            14000,
            270,
            420,
            0.03,
            query_id="q02",
            query_name="groom",
        ),
    ]
    cc.write_clip_csv(path, rows)
    back = cc.read_clip_csv(path)
    assert [(r.query_id, r.query_name) for r in back] == [("q02", "groom")] * 2


def test_example_idx_round_trips(tmp_path):
    # A non-zero example_idx (which example clip a match best matched) survives write→read,
    # and the column is in the header (a bare default-0 row would not prove either).
    path = tmp_path / "matches.csv"
    row = cc.ClipRow(
        "match", 1, "s2", 0, "emb_01", 9000, 14000, 270, 420, 0.03, example_idx=2
    )
    cc.write_clip_csv(path, [row])
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert "example_idx" in header
    assert cc.read_clip_csv(path)[0].example_idx == 2


def test_pre_multi_query_csv_reads_as_one_query(tmp_path):
    # a CSV written before the query columns existed still reads — one unnamed query.
    path = tmp_path / "old.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "kind",
                "rank",
                "recording_id",
                "track_id",
                "embed_run_id",
                "start_t_ns",
                "end_t_ns",
                "start_frame",
                "end_frame",
                "length",
                "score",
            ]
        )
        w.writerow(["match", 1, "s1", 0, "emb_01", 1000, 6000, 30, 180, 151, 0.05])
    rows = cc.read_clip_csv(path)
    assert len(rows) == 1
    assert rows[0].query_id == "q01" and rows[0].query_name == "query"


def test_import_prior_matches_as_examples(tmp_path):
    # a match row from a prior run is a valid query example (has subject + span).
    path = tmp_path / "prev.csv"
    cc.write_clip_csv(path, _rows())
    rows = cc.read_clip_csv(path)
    match = next(r for r in rows if r.kind == "match")
    spec = {
        "recording_id": match.recording_id,
        "track_id": match.track_id,
        "start_frame": match.start_frame,
        "end_frame": match.end_frame,
    }
    assert spec["recording_id"] == "s2" and spec["start_frame"] == 270


def test_extra_columns_follow_the_canonical_ones_and_read_back_unchanged(tmp_path):
    """A writer with more to say per clip (the clip montage's list) appends columns after the
    canonical header; the reader ignores them, so the file is still a clip-query CSV."""
    import csv

    from detail.storage.clip_query_csv import (
        COLUMNS,
        ClipRow,
        read_clip_csv,
        write_clip_csv,
    )

    rows = [
        ClipRow("query", 0, "rec_a", 0, "emb_01", 0, 10, 0, 1, 0.0, query_name="rear"),
        ClipRow("query", 0, "rec_b", 1, "emb_01", 20, 30, 2, 3, 0.0, query_name="rear"),
    ]
    path = tmp_path / "list.csv"
    write_clip_csv(
        path, rows, extra={"rank_typical": [3, 1], "recording_name": ["A", "B"]}
    )
    with open(path, newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    assert header == COLUMNS + ["rank_typical", "recording_name"]
    back = read_clip_csv(path)
    assert [(r.recording_id, r.query_name) for r in back] == [
        ("rec_a", "rear"),
        ("rec_b", "rear"),
    ]
    with open(path, newline="", encoding="utf-8-sig") as handle:
        listed = list(csv.DictReader(handle))
    assert [r["rank_typical"] for r in listed] == ["3", "1"]
    with pytest.raises(ValueError, match="already a clip-query column"):
        write_clip_csv(path, rows, extra={"rank": [1, 2]})
    with pytest.raises(ValueError, match="2 rows"):
        write_clip_csv(path, rows, extra={"node": [7]})
