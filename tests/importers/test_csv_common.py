"""The opener for a CSV a user hands in (``importers/_csv_common.read_user_csv``)."""

import csv

import pytest

from detail.core.errors import UserInputError
from detail.importers import _csv_common


def test_rows_read_as_dict_reader_reads_them(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("﻿a,b,c\n1,2,3\n\n4,5\n6,7,8,9\n", encoding="utf-8")
    header, rows = _csv_common.read_user_csv(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        expected = list(csv.DictReader(f))
    assert header == ["a", "b", "c"]
    assert [dict(r) for r in rows] == [dict(r) for r in expected]
    assert [(r.line, r.width, r.expected) for r in rows] == [
        (2, 3, 3),
        (4, 2, 3),
        (5, 4, 3),
    ]


def test_an_empty_file_has_no_header(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"")
    assert _csv_common.read_user_csv(path) == ([], [])
    with pytest.raises(UserInputError, match="CSV is empty"):
        _csv_common.read_rows(path)


def test_a_file_in_another_encoding_says_to_save_as_utf8(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes("name\nJosé\n".encode("cp1252"))
    with pytest.raises(UserInputError) as caught:
        _csv_common.read_user_csv(path)
    assert str(caught.value) == (
        f"{path} is not UTF-8 text. If it is a spreadsheet, save it from Excel with "
        "Save As → CSV UTF-8 (Comma delimited)"
    )


def test_a_csv_the_parser_refuses_is_named(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text(
        'a\n"' + "x" * (csv.field_size_limit() + 1) + '"\n', encoding="utf-8"
    )
    with pytest.raises(UserInputError, match="is not a CSV text file"):
        _csv_common.read_user_csv(path)


def test_a_cell_names_the_file_row_and_column(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("x,y\n1,2\n3\n4,z\n", encoding="utf-8")
    _, rows = _csv_common.read_user_csv(path)
    assert _csv_common.int_cell(rows[0], "y") == 2
    assert _csv_common.float_cell(rows[0], "x") == 1.0
    with pytest.raises(UserInputError) as short:
        _csv_common.float_cell(rows[1], "y")
    assert str(short.value) == f"{path} row 3: expected 2 cells, found 1 (no 'y')"
    with pytest.raises(UserInputError) as bad:
        _csv_common.int_cell(rows[2], "y")
    assert str(bad.value) == f"{path} row 4, column 'y': 'z' is not an integer"
