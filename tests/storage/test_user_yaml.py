"""The reader for YAML a user hands in (``detail/storage/user_yaml.py``)."""

import pytest

from detail.core.errors import UserInputError
from detail.storage import dirscan
from detail.storage.user_yaml import parse_user_yaml, read_user_yaml, require_mapping


def test_a_valid_file_reads_as_yaml(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("﻿name: x\nitems: [1, 2]\n", encoding="utf-8")
    assert read_user_yaml(path) == {"name": "x", "items": [1, 2]}


def test_an_empty_file_reads_as_none(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("", encoding="utf-8")
    assert read_user_yaml(path) is None


def test_a_syntax_error_names_the_file_line_and_column(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("names: [a, b\nedges: []\n", encoding="utf-8")
    with pytest.raises(UserInputError) as caught:
        read_user_yaml(path)
    assert str(caught.value) == (
        f"{path} is not valid YAML (line 2, column 6): expected ',' or ']', but got ':'"
    )


def test_bytes_that_are_not_text_say_so(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    with pytest.raises(UserInputError, match=r"a\.yaml is not a text file$"):
        read_user_yaml(path)
    with pytest.raises(UserInputError, match=r"^member is not a text file$"):
        parse_user_yaml(b"\xff\xfe\x00", "member")


def test_a_missing_file_is_still_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_user_yaml(tmp_path / "nope.yaml")


def test_a_listing_still_skips_a_record_the_reader_refuses():
    assert issubclass(UserInputError, dirscan.UNREADABLE_RECORD_ERRORS)


@pytest.mark.parametrize("data, found", [(None, "empty"), ([1], "a list")])
def test_require_mapping_names_what_it_found(data, found):
    with pytest.raises(UserInputError) as caught:
        require_mapping(data, "m.yaml", "a model record")
    assert str(caught.value) == (
        "m.yaml is not a model record: expected a YAML mapping (key: value lines), "
        f"found {found}"
    )
    assert require_mapping({"a": 1}, "m.yaml", "x") == {"a": 1}
