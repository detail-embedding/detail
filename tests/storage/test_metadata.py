"""Grouping-variable storage — metadata.csv (Section 11)."""

import csv

import pytest

from detail.storage import metadata as md
from detail.storage import experiment as experiment_store


def test_read_absent_is_empty(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    assert md.read_metadata(proj) == {}
    assert md.columns(proj) == []


def test_upsert_creates_row_with_recording_id_first(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "recording_001", {"animal_id": "M001", "treatment": "drug_A"})
    # header order: recording_id first, then columns in insertion order
    with open(md.metadata_path(proj), newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert header == ["recording_id", "animal_id", "treatment"]
    assert md.read_metadata(proj) == {
        "recording_001": {"animal_id": "M001", "treatment": "drug_A"}
    }
    assert md.columns(proj) == ["animal_id", "treatment"]


def test_upsert_unions_new_columns_and_merges(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "recording_001", {"animal_id": "M001"})
    md.upsert_row(proj, "recording_002", {"treatment": "vehicle"})  # new column
    md.upsert_row(proj, "recording_001", {"treatment": "drug_A"})  # merge into existing
    table = md.read_metadata(proj)
    assert table["recording_001"] == {"animal_id": "M001", "treatment": "drug_A"}
    assert table["recording_002"] == {"animal_id": "", "treatment": "vehicle"}
    assert md.columns(proj) == ["animal_id", "treatment"]
    assert list(table) == ["recording_001", "recording_002"]  # row order preserved


def test_empty_upsert_seeds_a_key_row(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "recording_001")  # how create_recording seeds the row
    assert md.read_metadata(proj) == {"recording_001": {}}


def test_remove_row_idempotent(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "recording_001", {"animal_id": "M001"})
    md.remove_row(proj, "recording_001")
    assert md.read_metadata(proj) == {}
    md.remove_row(proj, "recording_001")  # absent row → no error
    md.remove_row(proj, "never")  # absent file path also fine


def test_add_column(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "recording_001", {"animal_id": "M001"})
    md.add_column(proj, "group")
    assert md.columns(proj) == ["animal_id", "group"]
    assert md.read_metadata(proj)["recording_001"] == {"animal_id": "M001", "group": ""}


def test_add_column_rejects_reserved(tmp_path):
    import pytest

    proj = experiment_store.create_experiment(tmp_path, "p")
    with pytest.raises(ValueError, match="invalid metadata column"):
        md.add_column(proj, "recording_id")


# --- bulk CSV import (§11) --------------------------------------------------- #
def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_load_import_csv_parses(tmp_path):
    p = tmp_path / "m.csv"
    _write_csv(
        p,
        ["recording_id", "animal_id", "treatment"],
        [["s1", "M001", "drug_A"], ["s2", "M002", ""]],
    )
    key, cols, rows = md.load_import_csv(p)
    assert key == "recording_id"
    assert cols == ["animal_id", "treatment"]
    assert rows == [
        ("s1", {"animal_id": "M001", "treatment": "drug_A"}),
        ("s2", {"animal_id": "M002", "treatment": ""}),
    ]


def test_load_import_csv_with_no_key_column_names_its_columns(tmp_path):
    """The refusal carries the header, so a surface can ask which column is the key
    instead of leaving the person to guess what the importer wanted (#330)."""
    p = tmp_path / "m.csv"
    _write_csv(p, ["animal_id", "Session Clip"], [["M001", "m27.mp4"]])
    with pytest.raises(md.MissingKeyColumnError, match="Session Clip") as caught:
        md.load_import_csv(p)
    assert caught.value.header == ["animal_id", "Session Clip"]
    assert "recording_id" in str(caught.value) and "video" in str(caught.value)


@pytest.mark.parametrize(
    "header, expected",
    [
        (["recording_id", "name", "file", "dose"], "recording_id"),  # an export
        (["name", "file", "dose"], "name"),
        (["dose", "Video File", "file"], "Video File"),  # first alias in file order
        (["Recording ID", "dose"], "Recording ID"),
        (["dose", "FILENAME"], "FILENAME"),
        (["dose", "notes"], None),
    ],
)
def test_detect_key_column_prefers_id_then_name_then_a_video_file(header, expected):
    assert md.detect_key_column(header) == expected


@pytest.mark.parametrize(
    "column, kind",
    [
        ("recording_id", "recording_id"),
        ("Name", "name"),
        ("video", "file"),
        ("Session Clip", "file"),  # a column the person named: a video file name
    ],
)
def test_key_kind(column, kind):
    assert md.key_kind(column) == kind


def test_load_import_csv_keys_on_a_named_column_and_drops_the_derived_ones(tmp_path):
    """An explicit key is matched as a video file name; `name` and `file` riding along
    from an export are never stored as values, whichever column keys the import."""
    p = tmp_path / "m.csv"
    _write_csv(
        p,
        ["Session Clip", "name", "file", "dose"],
        [["m27.mp4", "m27", "m27.mp4", "0.5"]],
    )
    kind, cols, rows = md.load_import_csv(p, key_column="Session Clip")
    assert (kind, cols) == ("file", ["dose"])
    assert rows == [("m27.mp4", {"dose": "0.5"})]


def test_load_import_csv_refuses_a_key_column_it_does_not_have(tmp_path):
    p = tmp_path / "m.csv"
    _write_csv(p, ["video", "dose"], [["m27.mp4", "0.5"]])
    with pytest.raises(ValueError, match="no column 'clip'.*video, dose"):
        md.load_import_csv(p, key_column="clip")


def test_load_import_csv_rejects_duplicate_headers(tmp_path):
    import pytest

    p = tmp_path / "m.csv"
    _write_csv(p, ["recording_id", "x", "x"], [["s1", "a", "b"]])
    with pytest.raises(ValueError, match="duplicate column"):
        md.load_import_csv(p)


def test_validate_import_unknown_recording_is_error():
    rows = [("s1", {"treatment": "a"}), ("ghost", {"treatment": "b"})]
    errors, _ = md.validate_import(["treatment"], rows, ["s1"])
    assert any("unknown recording 'ghost'" in e for e in errors)


def test_validate_import_duplicate_and_empty_ids():
    rows = [("s1", {"t": "a"}), ("s1", {"t": "b"}), ("", {"t": "c"})]
    errors, _ = md.validate_import(["t"], rows, ["s1"])
    assert any("duplicate recording_id" in e for e in errors)
    assert any("empty recording_id" in e for e in errors)


def test_validate_import_no_data_columns_warns():
    errors, warnings = md.validate_import([], [("s1", {})], ["s1"])
    assert errors == [] and any("no data columns" in w for w in warnings)


def test_apply_import_merges_skips_blanks_adds_columns(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "s1", {"animal_id": "M001", "treatment": "old"})
    # blank treatment is skipped (keeps "old"); new 'group' column added
    md.apply_import(
        proj, ["treatment", "group"], [("s1", {"treatment": "", "group": "ctrl"})]
    )
    assert md.read_metadata(proj)["s1"] == {
        "animal_id": "M001",
        "treatment": "old",
        "group": "ctrl",
    }
    assert "group" in md.columns(proj)


# --- CSV export / template (§11) --------------------------------------------- #
def test_export_metadata_writes_template(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "s1", {"animal_id": "M001", "treatment": "drug_A"})
    md.upsert_row(proj, "s2", {"animal_id": "M002"})  # blank treatment
    out = tmp_path / "template.csv"
    md.export_metadata(proj, out)
    # recording_id first, every column, one row per recording, blanks empty
    assert out.read_text(encoding="utf-8-sig") == (
        "recording_id,animal_id,treatment\ns1,M001,drug_A\ns2,M002,\n"
    )
    # round-trips back through the importer
    key, cols, rows = md.load_import_csv(out)
    assert (key, cols) == ("recording_id", ["animal_id", "treatment"])
    assert rows[0] == ("s1", {"animal_id": "M001", "treatment": "drug_A"})


def test_export_metadata_empty_is_header_only(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    out = tmp_path / "template.csv"
    md.export_metadata(proj, out)
    assert out.read_text(encoding="utf-8-sig") == "recording_id\n"


def test_export_metadata_writes_the_derived_columns_it_is_given(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "s1", {"dose": "0.5"})
    md.upsert_row(proj, "s2")
    out = tmp_path / "template.csv"
    md.export_metadata(
        proj, out, names={"s1": "m27", "s2": "m28"}, files={"s1": "m27.mp4"}
    )
    assert out.read_text(encoding="utf-8-sig") == (
        "recording_id,name,file,dose\ns1,m27,m27.mp4,0.5\ns2,m28,,\n"
    )


# --- delete column + bulk replace (§11) -------------------------------------- #
def test_delete_column_removes_column_and_values(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "s1", {"animal_id": "M001", "treatment": "drug_A"})
    md.upsert_row(proj, "s2", {"animal_id": "M002", "treatment": "vehicle"})
    md.delete_column(proj, "treatment")
    assert md.columns(proj) == ["animal_id"]
    assert md.read_metadata(proj) == {
        "s1": {"animal_id": "M001"},
        "s2": {"animal_id": "M002"},
    }


def test_delete_column_absent_is_noop(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "s1", {"animal_id": "M001"})
    md.delete_column(proj, "nope")  # no error, no change
    assert md.columns(proj) == ["animal_id"]


def test_delete_column_rejects_recording_id(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    with pytest.raises(ValueError, match="reserved column"):
        md.delete_column(proj, "recording_id")


def test_write_metadata_replaces_whole_table(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    md.upsert_row(proj, "s1", {"animal_id": "M001", "treatment": "old"})
    # replace: drop treatment, add cohort, change values — one write
    md.write_metadata(
        proj, ["animal_id", "cohort"], {"s1": {"animal_id": "M1", "cohort": "A"}}
    )
    assert md.columns(proj) == ["animal_id", "cohort"]
    assert md.read_metadata(proj) == {"s1": {"animal_id": "M1", "cohort": "A"}}


# --- column types (the cluster view's filters pick a widget from these) -------------------


def test_a_column_is_a_number_a_date_a_time_or_a_category():
    rows = {
        "r1": {"dose": "1.5", "day": "2026-07-02", "sex": "F", "time": "14:00:21"},
        "r2": {"dose": "10", "day": "2026-07-08T17:23", "sex": "M", "time": "17:23:19"},
    }
    assert md.infer_column_types(rows) == {
        "dose": "number",
        "day": "date",
        "sex": "category",
        "time": "time",  # a bare time names no day: a time of day, not a date
    }


def test_a_clock_time_is_a_time_of_day_and_ranges_like_a_number():
    """A recording's start time of day (``14:00:21``) sorts and ranges by seconds after
    midnight; anything that is not a clock time keeps the column a category."""
    from detail.api import cluster_filter

    assert md.parse_time("14:00:21") == 14 * 3600 + 21
    assert md.parse_time(" 9:05 ") == 9 * 3600 + 5 * 60
    assert md.parse_time("14:00:21.5") == pytest.approx(14 * 3600 + 21.5)
    for bad in ("25:00", "14:60", "14:00:60", "2026-07-02T14:00", "14", "2pm", ""):
        assert md.parse_time(bad) is None, bad
    rows = {
        "a": {"time": "09:30"},
        "b": {"time": "14:00:21"},
        "c": {"time": "17:23:19"},
        "d": {"time": ""},
    }
    assert md.infer_column_types(rows) == {"time": "time"}
    assert md.infer_column_types({"a": {"t": "14:00"}, "b": {"t": "late"}}) == {
        "t": "category"
    }
    mask = cluster_filter.recording_mask(
        cluster_filter.Selection(ranges={"time": ("10:00", "17:23:19")}),
        ["a", "b", "c", "d"],
        rows,
    )
    assert list(mask) == [False, True, True, False], "a blank time is in no range"


def test_blank_cells_are_ignored_and_one_odd_value_makes_a_category():
    rows = {
        "r1": {"dose": "", "day": "2026-07-02", "mixed": "3", "empty": ""},
        "r2": {"dose": "5", "day": " ", "mixed": "three", "empty": ""},
        "r3": {"dose": "nan", "day": "2026-07-09", "mixed": "4"},
    }
    types = md.infer_column_types(rows)
    assert types["day"] == "date", "a blank does not break a date column"
    assert types["mixed"] == "category", "one value that is not a number"
    assert types["empty"] == "category", "nothing to range over"
    assert types["dose"] == "category", "nan is not a number a range can hold"
    assert md.infer_column_types({"r": {"year": "2026"}}) == {"year": "number"}
    assert "recording_id" not in md.infer_column_types({"r": {"recording_id": "r"}})


def test_a_date_with_an_offset_compares_with_one_without():
    """An aware and a naive datetime raise TypeError against each other, so a column mixing
    ``2026-07-02T10:00+02:00`` with ``2026-07-03`` broke every range on it. Offsets convert to
    naive UTC."""
    from detail.api import cluster_filter

    aware = md.parse_date("2026-07-02T10:00+02:00")
    assert aware.tzinfo is None and aware.hour == 8
    rows = {"a": {"day": "2026-07-02T10:00+02:00"}, "b": {"day": "2026-07-03"}}
    assert md.infer_column_types(rows) == {"day": "date"}
    mask = cluster_filter.recording_mask(
        cluster_filter.Selection(ranges={"day": ("2026-07-02T09:00", None)}),
        ["a", "b"],
        rows,
    )
    assert list(mask) == [False, True], "08:00 UTC is before the naive 09:00 bound"
