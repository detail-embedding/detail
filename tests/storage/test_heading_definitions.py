"""Saved heading definitions (§14): the stream-preset pattern, one file per experiment."""

import pytest
import yaml

from detail.storage import heading_definitions as hd


def test_no_file_lists_nothing(tmp_path):
    assert hd.list_definitions(tmp_path) == []


def test_save_then_read_back_schema_version_first(tmp_path):
    saved = hd.save_definition(tmp_path, "nose_to_tail", ["nose"], ["tail_base"])
    assert saved == {"name": "nose_to_tail", "front": ["nose"], "back": ["tail_base"]}
    assert hd.list_definitions(tmp_path) == [saved]
    assert hd.get_definition(tmp_path, "nose_to_tail") == saved
    text = (tmp_path / "heading_definitions.yaml").read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_a_repeat_name_needs_overwrite_and_a_bad_set_is_refused(tmp_path):
    hd.save_definition(tmp_path, "h", ["nose"], ["tail_base"])
    with pytest.raises(ValueError, match="already exists"):
        hd.save_definition(tmp_path, "h", ["nose"], ["tail_base"])
    hd.save_definition(
        tmp_path, "h", ["nose", "left_ear"], ["tail_base"], overwrite=True
    )
    assert hd.get_definition(tmp_path, "h")["front"] == ["nose", "left_ear"]
    with pytest.raises(ValueError, match="both front and back"):
        hd.save_definition(tmp_path, "x", ["nose"], ["nose"])
    with pytest.raises(ValueError, match="at least one 'back'"):
        hd.save_definition(tmp_path, "x", ["nose"], [])
    with pytest.raises(ValueError, match="letters, digits"):
        hd.save_definition(tmp_path, "bad name", ["nose"], ["tail"])


def test_unknown_name_names_what_is_there_and_delete(tmp_path):
    hd.save_definition(tmp_path, "a", ["nose"], ["tail"])
    with pytest.raises(
        ValueError, match=r"no heading definition named 'b' \(have: a\)"
    ):
        hd.get_definition(tmp_path, "b")
    hd.delete_definition(tmp_path, "a")
    assert hd.list_definitions(tmp_path) == []
    with pytest.raises(ValueError, match="no heading definition named 'a'"):
        hd.delete_definition(tmp_path, "a")


def test_export_import_round_trip_and_rename(tmp_path):
    hd.save_definition(tmp_path, "h", ["nose"], ["tail_base"])
    out = tmp_path / "h.yaml"
    hd.export_definition(tmp_path, "h", out)
    data = yaml.safe_load(out.read_text(encoding="utf-8-sig"))
    assert data == {
        "schema_version": 1,
        "name": "h",
        "front": ["nose"],
        "back": ["tail_base"],
    }
    other = tmp_path / "other"
    other.mkdir()
    assert hd.import_definition(other, out) == "h"
    assert hd.import_definition(other, out, name="h2") == "h2"
    with pytest.raises(ValueError, match="already exists"):
        hd.import_definition(other, out)
    assert hd.import_definition(other, out, overwrite=True) == "h"
    with pytest.raises(FileNotFoundError):
        hd.import_definition(other, tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 2\nname: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        hd.import_definition(other, bad)


def test_fits_says_why_a_definition_cannot_be_used():
    definition = {"name": "h", "front": ["nose"], "back": ["hip"]}
    assert hd.fits(definition, ("nose", "hip")) is None
    assert "skeleton lacks: ['hip']" in hd.fits(definition, ("nose", "tail"))
