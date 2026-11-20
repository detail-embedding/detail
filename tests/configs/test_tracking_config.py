"""Tests for the tracking config model (detail/configs/tracking.py)."""

import pytest
from pydantic import ValidationError

import detail.pipeline.tracking  # noqa: F401 — populates TRACKERS
from detail.configs.tracking import DEFAULT_TRACKER, TrackingConfig
from detail.core.registries import TRACKERS


def test_round_trips():
    d = {
        "detection_run_id": "det_01",
        "tracker_id": "anchor_tracker",
        "arena_ids": ["arena_01"],
        "arena_name": "rig A camera",
        "hyperparams": {"iou_threshold": 0.3},
        "every_n_frames": 2,
        "exclude_classes": ["cable"],
    }
    assert TrackingConfig.model_validate(d).model_dump() == d


def test_defaults():
    cfg = TrackingConfig.model_validate(
        {"detection_run_id": "det_01", "tracker_id": "anchor_tracker"}
    )
    assert cfg.model_dump() == {
        "detection_run_id": "det_01",
        "tracker_id": "anchor_tracker",
        "arena_ids": None,
        "arena_name": None,
        "hyperparams": None,
        "every_n_frames": None,
        "exclude_classes": None,  # validate() resolves it from the detection model
    }


def test_requires_a_detection_run():
    with pytest.raises(ValidationError, match="detection_run_id"):
        TrackingConfig.model_validate({"tracker_id": "anchor_tracker"})


def test_a_config_that_names_no_tracker_gets_the_default():
    """The tracker is a *setting* with a sensible answer, not a selection only the user knows.

    It used to be required, so each surface supplied its own — the CLI hardcoded
    ``iou_tracker`` and the GUI opened on whichever id sorted first. Naming it here is what
    makes those one fact; `DEFAULT_TRACKER` is what they all read.
    """
    cfg = TrackingConfig.model_validate({"detection_run_id": "det_01"})
    assert cfg.tracker_id == DEFAULT_TRACKER
    assert TrackingConfig.model_fields["tracker_id"].default == DEFAULT_TRACKER
    assert DEFAULT_TRACKER in TRACKERS, "the default names no registered tracker"


def test_accepts_none_arena_and_hyperparams():
    # validate() feeds these leniently — arena_ids may be None/absent/list and hyperparams a
    # dict/None; the tracker owns hyperparam validation, so the model must not reject them.
    cfg = TrackingConfig.model_validate(
        {
            "detection_run_id": "det_01",
            "tracker_id": "anchor_tracker",
            "arena_ids": None,
            "hyperparams": None,
        }
    )
    assert cfg.arena_ids is None and cfg.hyperparams is None


def test_field_metadata_is_the_single_source():
    fields = TrackingConfig.model_fields
    assert fields["tracker_id"].description
    assert fields["detection_run_id"].json_schema_extra["label"] == "Detection run"
