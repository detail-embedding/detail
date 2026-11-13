"""Tests for the detection config model (detail/configs/detection.py)."""

import pytest
from pydantic import ValidationError

from detail.configs.detection import DetectionConfig


def test_round_trips():
    d = {
        "model_id": "sam3",
        "model_variant": None,
        "conf_threshold": 0.3,
        "class_conf_thresholds": {"black mouse": 0.2, "white mouse": 0.4},
        "batch_size": 2,
        "every_n_frames": 4,
        "teacher_window_s": 300.0,
    }
    assert DetectionConfig.model_validate(d).model_dump() == d


def test_defaults():
    cfg = DetectionConfig.model_validate({"model_id": "sam3"})
    assert cfg.model_dump() == {
        "model_id": "sam3",
        "model_variant": None,
        "conf_threshold": 0.5,
        "class_conf_thresholds": {},  # no class-level overrides
        "batch_size": None,  # unset → validate resolves per model (4 YOLO / 1 SAM3)
        "every_n_frames": 1,  # every frame
        "teacher_window_s": None,  # the whole recording: not a keypoint teacher run
    }


def test_rejects_conf_out_of_range():
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        DetectionConfig.model_validate({"model_id": "sam3", "conf_threshold": 1.5})
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        DetectionConfig.model_validate({"model_id": "sam3", "conf_threshold": -0.1})


def test_class_conf_thresholds_defaults_empty():
    cfg = DetectionConfig.model_validate({"model_id": "sam3"})
    assert cfg.class_conf_thresholds == {}


def test_rejects_batch_size_below_1():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        DetectionConfig.model_validate({"model_id": "sam3", "batch_size": 0})


def test_requires_model_id():
    with pytest.raises(ValidationError, match="model_id"):
        DetectionConfig.model_validate({"conf_threshold": 0.5})


def test_extra_keys_ignored():
    # prompt / compute / weights_source / processing_run_id ride alongside in the config dict;
    # the model ignores them (validate() preserves them via {**config, **model_dump()}).
    cfg = DetectionConfig.model_validate(
        {
            "model_id": "sam3",
            "prompt": {"targets": []},
            "weights_source": "trained",
            "processing_run_id": "run_01",
        }
    )
    assert set(cfg.model_dump()) == {
        "model_id",
        "model_variant",
        "conf_threshold",
        "class_conf_thresholds",
        "batch_size",
        "every_n_frames",
        "teacher_window_s",
    }


def test_field_metadata_is_the_single_source():
    fields = DetectionConfig.model_fields
    assert fields["conf_threshold"].description
    assert fields["conf_threshold"].json_schema_extra["label"] == "Confidence threshold"


@pytest.mark.parametrize("window", [0, -5.0])
def test_rejects_a_teacher_window_that_is_not_positive(window):
    with pytest.raises(ValidationError, match="greater than 0"):
        DetectionConfig.model_validate({"model_id": "sam3", "teacher_window_s": window})
