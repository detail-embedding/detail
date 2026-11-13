"""Detection config model (§7) — the single source for its scalar parameters (confidence,
batch size, frame stride) and the typed shape of a detection run. The nested prompt /
compute-target inputs stay their own dataclasses (assembled + validated in
``DetectionModule.validate``, which resolves the model and checks the prompt against it), so they
are not modelled here; ``weights_source`` is resolved there too. ``class_conf_thresholds`` is a
partial exception: it is a plain dict, so it round-trips through ``model_validate`` like any other
field, but which keys are valid depends on the resolved model's classes — that check, like the
prompt's, stays in ``validate()``.
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
    # Keyed by class name rather than modelled per-key (the valid keys are whichever
    # classes the *selected* model actually has, known only at validate() time, not at
    # schema-definition time) — a plain dict still round-trips through model_validate
    # cleanly, unlike prompt/compute, which need their own dataclass construction. Each
    # value shares conf_threshold's own [0, 1] bound, checked in DetectionModule.validate()
    # since pydantic has no per-value Field bound for a dict here; an unknown class name is
    # also validate()'s job, since it needs the resolved model to know what's valid.
    class_conf_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-class confidence overrides, keyed by class name. A class with no entry "
            "here uses conf_threshold. Each value is bounded the same way (0–1)."
        ),
        json_schema_extra={"label": "Per-class confidence overrides", "advanced": True},
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
    teacher_window_s: float | None = Field(
        None,
        gt=0,
        description=(
            "Detect only the first this-many seconds of each recording, for a keypoint "
            "teacher run (§26). A run with a window is a teacher run: tracking and keypoints "
            "inherit it, and nothing else accepts it as input. Omit for the whole recording."
        ),
        json_schema_extra={"label": "Teacher window", "unit": "s", "advanced": True},
    )
