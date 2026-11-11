"""Tests for the embedding config model (detail/configs/embedding.py)."""

import pytest
from pydantic import ValidationError

from detail.configs.embedding import EmbeddingConfig


def test_round_trips():
    d = {
        "tracking_run_id": "track_01",
        "all_tracks": False,
        "model_id": "dinov3_convnext_base",
        "model_variant": None,
        "readout": "gap",
        "constructor": "full_frame",
        "input_size": 384,
        "resize": 224,
        "canvas_scope": "shared",
        "every_n_frames": 2,
        "batch_size": 16,
        "align": "mirror",
        "keypoint_run_id": "kt_01",
        "heading": "nose_to_tail",
        "max_gap_s": 0.25,
        "mirror_hold_s": 2.0,
        "facing_camera_ratio": 0.4,
    }
    assert EmbeddingConfig.model_validate(d).model_dump() == d


def test_defaults():
    cfg = EmbeddingConfig.model_validate(
        {"tracking_run_id": "track_01", "model_id": "m"}
    )
    assert cfg.model_dump() == {
        "tracking_run_id": "track_01",
        "all_tracks": False,
        "model_id": "m",
        "model_variant": None,
        "readout": None,
        "constructor": "masked_color_centered_contrast",
        "input_size": None,
        "resize": None,
        # The canvas scope defaults to per_recording — the pre-existing behaviour, so
        # adding it changed no run. The background has no scope: always one per run.
        "canvas_scope": "per_recording",
        "every_n_frames": None,
        "batch_size": 32,
        # §14 alignment: off, and every setting at its default
        "align": "none",
        "keypoint_run_id": None,
        "heading": None,
        "max_gap_s": 0.5,
        "mirror_hold_s": 1.0,
        "facing_camera_ratio": 0.5,
    }


def test_rejects_batch_size_out_of_range():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        EmbeddingConfig.model_validate(
            {"tracking_run_id": "t", "model_id": "m", "batch_size": 0}
        )
    with pytest.raises(ValidationError, match="less than or equal to 1024"):
        EmbeddingConfig.model_validate(
            {"tracking_run_id": "t", "model_id": "m", "batch_size": 2048}
        )


def test_requires_tracking_and_defaults_the_model():
    with pytest.raises(ValidationError, match="tracking_run_id"):
        EmbeddingConfig.model_validate({"model_id": "m"})
    # One default per stage (DINOv3 ViT-L at its own 224 px), read by every surface.
    cfg = EmbeddingConfig.model_validate({"tracking_run_id": "t"})
    assert cfg.model_id == "dinov3_vit"
    assert (
        cfg.model_variant is None
    )  # the model's own default size, resolved in validate
    assert cfg.input_size is None  # the model's own input size
    assert cfg.constructor == "masked_color_centered_contrast"


def test_extra_keys_ignored():
    # track_ids / renderer_spec / compute ride alongside in the config dict; the model
    # ignores them (validate() resolves + returns them via the enriched config).
    cfg = EmbeddingConfig.model_validate(
        {
            "tracking_run_id": "t",
            "model_id": "m",
            "track_ids": [0, 1],
            "renderer_spec": [{"type": "resize"}],
            "compute": "x",
        }
    )
    assert set(cfg.model_dump()) == {
        "tracking_run_id",
        "all_tracks",
        "model_id",
        "model_variant",
        "readout",
        "constructor",
        "input_size",
        "resize",
        "canvas_scope",
        "every_n_frames",
        "batch_size",
        "align",
        "keypoint_run_id",
        "heading",
        "max_gap_s",
        "mirror_hold_s",
        "facing_camera_ratio",
    }


def test_field_metadata_is_the_single_source():
    fields = EmbeddingConfig.model_fields
    assert fields["batch_size"].description
    assert fields["batch_size"].json_schema_extra["label"] == "Batch size"
