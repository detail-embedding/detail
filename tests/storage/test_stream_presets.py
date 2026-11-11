"""Saved stream-import presets: the experiment-scoped store (Section 3)."""

import pytest
import yaml

from detail.storage import stream_presets as presets

CONFIG = {
    "time_column": "t_sec",
    "time_kind": "seconds",
    "channels": [{"column": "nose_x", "name": "nose_x"}],
}


def test_no_file_lists_nothing(tmp_path):
    assert presets.list_presets(tmp_path) == []


def test_save_then_read_back(tmp_path):
    presets.save_preset(tmp_path, "dlc", "timeseries_csv", CONFIG)

    entry = presets.get_preset(tmp_path, "dlc")
    assert entry["importer"] == "timeseries_csv"
    assert entry["config"] == CONFIG
    assert (
        "offset_ns" not in entry
    )  # alignment is per recording; absent unless asked for


def test_schema_version_is_the_first_key(tmp_path):
    presets.save_preset(tmp_path, "dlc", "timeseries_csv", CONFIG)

    text = (tmp_path / presets.PRESETS_FILENAME).read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1")
    assert yaml.safe_load(text)["schema_version"] == 1


def test_an_offset_is_kept_when_explicitly_given(tmp_path):
    """The deterministic-rig case: one trigger starts camera and logger together."""
    presets.save_preset(tmp_path, "rig", "timeseries_csv", CONFIG, offset_ns=5)

    assert presets.get_preset(tmp_path, "rig")["offset_ns"] == 5


def test_a_repeat_name_needs_overwrite(tmp_path):
    presets.save_preset(tmp_path, "dlc", "timeseries_csv", CONFIG)

    with pytest.raises(ValueError, match="already exists"):
        presets.save_preset(tmp_path, "dlc", "timeseries_csv", CONFIG)

    presets.save_preset(tmp_path, "dlc", "medpc", None, overwrite=True)
    assert presets.get_preset(tmp_path, "dlc")["importer"] == "medpc"
    assert len(presets.list_presets(tmp_path)) == 1


def test_unknown_name_names_what_is_there(tmp_path):
    """An error that lists the alternatives, since the name came from someone's memory."""
    presets.save_preset(tmp_path, "dlc", "timeseries_csv", CONFIG)

    with pytest.raises(ValueError, match="have: dlc"):
        presets.get_preset(tmp_path, "nope")


@pytest.mark.parametrize("name", ["", "has space", "dots.here", "slash/es"])
def test_an_unsafe_name_is_refused(tmp_path, name):
    with pytest.raises(ValueError):
        presets.save_preset(tmp_path, name, "timeseries_csv", CONFIG)


def test_delete(tmp_path):
    presets.save_preset(tmp_path, "a", "timeseries_csv", CONFIG)
    presets.save_preset(tmp_path, "b", "timeseries_csv", CONFIG)

    presets.delete_preset(tmp_path, "a")

    assert [p["name"] for p in presets.list_presets(tmp_path)] == ["b"]
    with pytest.raises(ValueError, match="no stream preset"):
        presets.delete_preset(tmp_path, "a")


def test_export_import_round_trip(tmp_path):
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    source.mkdir()
    dest.mkdir()
    presets.save_preset(source, "dlc", "timeseries_csv", CONFIG)

    presets.export_preset(source, "dlc", tmp_path / "dlc.yaml")
    name = presets.import_preset(dest, tmp_path / "dlc.yaml")

    assert name == "dlc"
    assert presets.get_preset(dest, "dlc")["config"] == CONFIG


def test_import_can_rename(tmp_path):
    presets.save_preset(tmp_path, "dlc", "timeseries_csv", CONFIG)
    presets.export_preset(tmp_path, "dlc", tmp_path / "dlc.yaml")

    assert presets.import_preset(tmp_path, tmp_path / "dlc.yaml", name="dlc2") == "dlc2"


@pytest.mark.parametrize(
    "payload, match",
    [
        ("just a string\n", "not a mapping"),
        ("schema_version: 2\nname: x\nimporter: y\n", "schema_version"),
        ("schema_version: 1\nimporter: y\n", "no 'name'"),
        ("schema_version: 1\nname: x\n", "no 'importer'"),
        (
            "schema_version: 1\nname: x\nimporter: y\nconfig: 3\n",
            "non-mapping 'config'",
        ),
    ],
)
def test_a_malformed_preset_file_is_refused(tmp_path, payload, match):
    path = tmp_path / "bad.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        presets.import_preset(tmp_path, path)


def test_import_of_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        presets.import_preset(tmp_path, tmp_path / "nope.yaml")


def test_the_write_is_atomic_and_leaves_no_temp(tmp_path):
    """A concurrently-read file is written tmp-then-replace, never truncated in place."""
    presets.save_preset(tmp_path, "a", "timeseries_csv", CONFIG)
    presets.save_preset(tmp_path, "b", "timeseries_csv", CONFIG)

    leftovers = list(tmp_path.glob("*.tmp"))
    assert not leftovers, leftovers
    assert len(presets.list_presets(tmp_path)) == 2
