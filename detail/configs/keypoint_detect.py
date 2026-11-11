"""Keypoint detect config model (§26) — the typed shape of a ``keypoint_detect`` run.

The model is a **subset** of the run's config: the nested ``compute`` target, the resolved
example-set copy, the skeleton and its digest, the backbone and the matching constants are
enrichment ``KeypointDetectModule.validate`` writes back (Tenet 9) and stay outside it, so
``validate`` returns the enriched config rather than merging a partial ``model_dump()``.
"""

from typing import Literal

from pydantic import BaseModel, Field

#: The crop the research measured on (§26 *Open questions* 4); a multiple of the ViT patch.
DEFAULT_CROP_PX = 448
#: Ultralytics' own default box confidence (§26 *TrainedYoloPose*): the one threshold applied
#: at run time, and only to a method that ``reports_confidence``.
DEFAULT_MIN_POSE_CONFIDENCE = 0.25


class KeypointDetectConfig(BaseModel):
    detection_run_id: str = Field(
        ...,
        description=(
            "Detection run to place keypoints on (e.g. det_01). A method that works per "
            "detection crops each of its detections (and reads their masks if it uses them); "
            "a whole-frame method uses its frames. Keypoints are stored per frame with no "
            "subject: keypoint track gives each pose to a tracked animal."
        ),
        json_schema_extra={"label": "Detection run"},
    )
    model_id: str = Field(
        ...,
        description=(
            "The keypoint method, by registered id (dinov3_match). No default: which method "
            "fits is a fact about the animal and the camera, and a default is where a wrong "
            "one hides."
        ),
        json_schema_extra={"label": "Method"},
    )
    model_variant: str | None = Field(
        None,
        description=(
            "The method's size variant; omit for its default (dinov3_match: vit_large). "
            "For a method on a shared backbone this is the backbone's size."
        ),
        json_schema_extra={"label": "Size / variant"},
    )
    example_set_id: str | None = Field(
        None,
        description=(
            "The example set the method learns from (ex_NN): the marked frames of the "
            "skeleton to place. Required by a method that needs examples; refused by one "
            "that does not."
        ),
        json_schema_extra={"label": "Example set"},
    )
    min_pose_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Keep a pose only if the method's confidence in the whole animal is at least "
            "this. Only for a method that reports one (a trained pose model); omit for "
            f"{DEFAULT_MIN_POSE_CONFIDENCE}. Lower keeps more faint or partly hidden animals "
            "and more false ones. Poses below it are not stored, so changing it means "
            "running again."
        ),
        json_schema_extra={"label": "Minimum pose confidence"},
    )
    decode: Literal["skeleton", "per_point"] | None = Field(
        None,
        description=(
            "How a frame's candidates become points. 'skeleton' chooses jointly: the "
            "midline as a tree with edge lengths learned from the examples, left/right "
            "pairs as two distinct places (the default when the skeleton has edges). "
            "'per_point' takes each point's best candidate on its own."
        ),
        json_schema_extra={"label": "Decode"},
    )
    side_rule: Literal["auto", "midline", "hard", "soft", "off"] | None = Field(
        None,
        description=(
            "How a left/right pair is kept apart under the skeleton decode. 'auto' decides "
            "per pair from the example clicks (the default when the skeleton has a pair): "
            "'midline' puts each member on its own side of the midline (right from above), "
            "'hard' keeps their relative order across the body (right from the side). "
            "'soft' only penalises a swap; 'off' ignores sides (the default without pairs)."
        ),
        json_schema_extra={"label": "Left/right rule"},
    )
    crop_px: int | None = Field(
        None,
        ge=64,
        le=1792,
        description=(
            "Side of the square crop around each detection's box the method matches in, in "
            f"pixels; a multiple of the backbone's patch size (16). Omit for {DEFAULT_CROP_PX}. "
            "Every animal is resized to this crop, so a patch is a fixed fraction of body "
            "size whatever the rig."
        ),
        json_schema_extra={"label": "Crop size (px)"},
    )
    every_n_frames: int | None = Field(
        None,
        ge=1,
        description=(
            "Use every Nth frame of the video. Omit to use every frame the detection run "
            "holds. Counted in source frames, so it must be a multiple of what detection "
            "used."
        ),
        json_schema_extra={"label": "Use every"},
    )
    batch_size: int = Field(
        16,
        ge=1,
        le=256,
        description=(
            "Frames per forward pass. Lower it if the method runs out of GPU memory; it "
            "never changes a result."
        ),
        json_schema_extra={"label": "Batch size"},
    )
