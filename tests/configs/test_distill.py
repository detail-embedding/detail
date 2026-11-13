"""Tests for the distill config model (detail/configs/distill.py)."""

import pytest
from pydantic import ValidationError

from detail.configs.distill import DistillConfig


def _labeling(**over):
    d = {
        "model_id": "sam3",
        "model_variant": None,
        "n_frames": 200,
        "sampling": "stratified",
        "logit_threshold": 0.5,
        "min_contour_area": 100,
        "val_split": 0.15,
        "background_share": 0.1,
    }
    d.update(over)
    return d


def _training(**over):
    d = {"epochs": 100, "batch_size": 32, "imgsz": 640, "patience": 100}
    d.update(over)
    return d


def test_valid_config_round_trips():
    d = {
        "name": "mouse_fast",
        "base_model": "yolo11s-seg.pt",
        "trainer_id": "ultralytics_yolo",
        "stage_videos": True,
        "labeling": _labeling(),
        "training": _training(),
    }
    assert DistillConfig.model_validate(d).model_dump() == d


def test_defaults():
    cfg = DistillConfig.model_validate(
        {"name": "x", "labeling": _labeling(n_frames=10)}
    )
    assert cfg.stage_videos is False, "100+ GB of cluster storage is not a default"
    assert cfg.base_model == "yolo11s-seg.pt"
    assert cfg.trainer_id == "ultralytics_yolo"
    assert cfg.training.epochs == 100  # nested default via default_factory
    assert cfg.labeling.sampling == "stratified"


def test_name_required():
    with pytest.raises(ValidationError, match="name"):
        DistillConfig.model_validate({"labeling": _labeling(n_frames=10)})


def test_rejects_empty_name():
    with pytest.raises(ValidationError, match="at least 1 character"):
        DistillConfig.model_validate({"name": "", "labeling": _labeling(n_frames=10)})


def test_forbids_unknown_top_level_key():
    with pytest.raises(ValidationError, match="bogus"):
        DistillConfig.model_validate(
            {"name": "x", "labeling": _labeling(n_frames=10), "bogus": 1}
        )


def test_nested_labeling_bound_propagates():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        DistillConfig.model_validate({"name": "x", "labeling": _labeling(n_frames=0)})


def test_nested_training_bound_propagates():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        DistillConfig.model_validate(
            {
                "name": "x",
                "labeling": _labeling(n_frames=10),
                "training": _training(epochs=0),
            }
        )


def test_field_metadata_is_the_single_source():
    fields = DistillConfig.model_fields
    for name in ("name", "base_model", "trainer_id"):
        assert fields[name].description
        assert fields[name].json_schema_extra["label"]
    # the nested stage configs carry their own field metadata, unchanged
    assert (
        DistillConfig.model_fields["labeling"]
        .annotation.model_fields["n_frames"]
        .description
    )
