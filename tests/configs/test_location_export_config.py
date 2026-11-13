"""Tests for the location-export config model (detail/configs/location_export.py)."""

from detail.configs.location_export import LocationExportConfig


def test_round_trips():
    # model_dump() reproduces the stored dict shape validate() returns into the record.
    d = {
        "source_recording_ids": ["s1", "s2"],
        "tracking_run_id": "track_01",
        "keypoint_run_id": "kt_01",
        "pair_edge_distance": True,
    }
    assert LocationExportConfig.model_validate(d).model_dump() == d


def test_defaults_are_empty_and_none():
    # An empty config is well-formed at the model level; validate() supplies the
    # "needs at least one recording / a tracking run" messages (experiment-aware).
    cfg = LocationExportConfig.model_validate({})
    assert cfg.model_dump() == {
        "source_recording_ids": [],
        "tracking_run_id": None,
        "keypoint_run_id": None,
        "pair_edge_distance": False,
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
