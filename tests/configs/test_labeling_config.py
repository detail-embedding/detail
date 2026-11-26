"""Tests for the labeling config model (detail/configs/labeling.py)."""

import pytest
from pydantic import ValidationError

from detail.configs.labeling import LabelingConfig


def test_valid_config_round_trips():
    d = {
        "model_id": "fake_yolo",
        "model_variant": "11l",
        "n_frames": 200,
        "sampling": "uniform",
        "logit_threshold": 0.4,
        "min_contour_area": 50,
        "val_split": 0.2,
        "background_share": 0.25,
        "negative_prompt": "cable",
    }
    assert LabelingConfig.model_validate(d).model_dump() == d


def test_defaults():
    cfg = LabelingConfig.model_validate({"n_frames": 10})
    assert cfg.model_id == "sam3"  # default keeps the pre-migration behavior
    assert cfg.model_variant is None
    assert cfg.sampling == "stratified"
    assert cfg.logit_threshold == 0.5
    assert cfg.min_contour_area == 100
    assert cfg.val_split == 0.15
    assert cfg.background_share == 0.1
    assert cfg.negative_prompt is None


@pytest.mark.parametrize("share", [-0.1, 0.51])
def test_rejects_background_share_out_of_bounds(share):
    with pytest.raises(ValidationError, match="background_share"):
        LabelingConfig.model_validate({"n_frames": 10, "background_share": share})


def test_n_frames_required():
    with pytest.raises(ValidationError, match="n_frames"):
        LabelingConfig.model_validate({})


def test_rejects_n_frames_below_one():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        LabelingConfig.model_validate({"n_frames": 0})


def test_rejects_bad_sampling():
    with pytest.raises(ValidationError, match="uniform"):
        LabelingConfig.model_validate({"n_frames": 10, "sampling": "middle-out"})


def test_rejects_val_split_at_or_above_one():
    with pytest.raises(ValidationError, match="less than 1"):
        LabelingConfig.model_validate({"n_frames": 10, "val_split": 1.0})


def test_field_metadata_is_the_single_source():
    fields = LabelingConfig.model_fields
    for name in (
        "model_id",
        "model_variant",
        "n_frames",
        "sampling",
        "logit_threshold",
        "min_contour_area",
        "val_split",
    ):
        assert fields[name].description
        assert fields[name].json_schema_extra["label"]
    # the threshold label is model-neutral now (was "SAM3 threshold")
    assert (
        fields["logit_threshold"].json_schema_extra["label"] == "Confidence threshold"
    )
    assert "SAM3" not in fields["logit_threshold"].description


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_negative_prompt_is_none(blank):
    cfg = LabelingConfig.model_validate({"n_frames": 10, "negative_prompt": blank})
    assert cfg.negative_prompt is None


def test_a_negative_prompt_is_trimmed():
    cfg = LabelingConfig.model_validate({"n_frames": 10, "negative_prompt": " cable "})
    assert cfg.negative_prompt == "cable"
