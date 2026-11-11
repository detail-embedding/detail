"""Detection config model (§7) — the single source for its scalar parameter (confidence) and the
typed shape of a detection run. The nested prompt / compute-target inputs stay their own dataclasses
(assembled + validated in ``DetectionModule.validate``, which resolves the model and checks the
prompt against it), so they are not modelled here; ``weights_source`` is resolved there too.
"""

from pydantic import BaseModel, Field

# Default frames-per-forward-pass for a batch-capable model (YOLO) when the user
# leaves batch_size unset. Non-batching models (SAM3) default to 1.
BATCHABLE_DEFAULT_BATCH_SIZE = 4


class DetectionConfig(BaseModel):
    model_id: str = Field(
        ...,
        description="Detection model to run: a built-in, experiment-trained, or imported model id.",
        json_schema_extra={"label": "Model"},
    )
    model_variant: str | None = Field(
        None,
        description="Model size/variant, if the model has size options; omit for the default.",
        json_schema_extra={"label": "Size / variant"},
    )
    conf_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence to keep (0–1).",
        json_schema_extra={"label": "Confidence threshold"},
    )
    batch_size: int | None = Field(
        None,
        ge=1,
        description=(
            "Frames per GPU forward pass. Omit to use the model's default (4 for "
            "batch-capable YOLO models, 1 for SAM3); bounded by GPU memory."
        ),
        json_schema_extra={"advanced": True, "label": "Batch size", "ui_max": 512},
    )
    every_n_frames: int = Field(
        1,
        ge=1,
        description=(
            "Use every Nth frame of the video: 1 detects every frame, 4 detects frames 0, 4, "
            "8, … Cost falls roughly in proportion. Tracking and embedding count the same "
            "source frames, so they can only go coarser by a multiple of this."
        ),
        json_schema_extra={"label": "Use every"},
    )
