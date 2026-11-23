"""Tests for the location-export config model (detail/configs/location_export.py)."""

import pytest

from detail.configs.location_export import LocationExportConfig
from detail.configs.location_export import option_names
from detail.configs.location_export import options_from


def test_round_trips():
    # model_dump() reproduces the stored dict shape validate() returns into the record.
    d = {
        "source_recording_ids": ["s1", "s2"],
        "tracking_run_id": "track_01",
        "keypoint_run_id": "kt_01",
        "sample_fps": 5.0,
        "remove_thin_parts": True,
        "thin_part_scale": 0.2,
    }
    assert LocationExportConfig.model_validate(d).model_dump() == d


def test_defaults_are_empty_and_none():
    # An empty config is well-formed at the model level; validate() supplies the
    # "needs at least one recording / a tracking run" messages (experiment-aware).
    cfg = LocationExportConfig.model_validate({})
    # Unset, the export measures every tracked frame of the whole mask: what it did before
    # the options existed, so an older job or record reads as what actually ran.
    assert cfg.model_dump() == {
        "source_recording_ids": [],
        "tracking_run_id": None,
        "keypoint_run_id": None,
        "sample_fps": None,
        "remove_thin_parts": False,
        "thin_part_scale": 0.12,
    }


def test_extra_keys_ignored():
    cfg = LocationExportConfig.model_validate(
        {"source_recording_ids": ["s1"], "tracking_run_id": "track_01", "extra": "x"}
    )
    assert "extra" not in cfg.model_dump()


def test_field_metadata_is_the_single_source():
    fields = LocationExportConfig.model_fields
    assert fields["tracking_run_id"].description
    assert fields["source_recording_ids"].json_schema_extra["label"] == "Recordings"


def test_the_options_are_the_fields_that_are_not_the_selection():
    # read off the model, in its order: what every surface enumerates instead of listing
    assert option_names() == ("sample_fps", "remove_thin_parts", "thin_part_scale")
    selection = {
        name
        for name, field in LocationExportConfig.model_fields.items()
        if (field.json_schema_extra or {}).get("scope") == "selection"
    }
    assert selection == {"source_recording_ids", "tracking_run_id", "keypoint_run_id"}
    assert set(option_names()) | selection == set(LocationExportConfig.model_fields)


def test_options_are_read_out_of_a_larger_config_and_checked():
    run_config = {
        "source_recording_ids": ["s1"],
        "tracking_spec": {"tracker_id": "anchor_tracker"},  # not a field of this model
        "sample_fps": 5,
        "remove_thin_parts": True,
    }
    options = options_from(run_config)
    assert (options.sample_fps, options.remove_thin_parts, options.thin_part_scale) == (
        5.0,
        True,
        0.12,
    )
    assert options_from({}).sample_fps is None


@pytest.mark.parametrize(
    "bad",
    [
        {"sample_fps": 0},
        {"sample_fps": -5.0},
        {"thin_part_scale": 0.0},
        {"thin_part_scale": 0.51},
    ],
)
def test_an_option_outside_its_bounds_is_refused_and_named(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        options_from(bad)
