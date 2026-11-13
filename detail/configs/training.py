"""Training config model (§12) — the single source for a trainer's hyperparameters: each
parameter's type, bounds, default, help, and label. Unlike the tracking model (which types the run
envelope and leaves tracker hyperparameters to ``HYPERPARAMS_SCHEMA``), the trainer's knobs are
modelled here directly: the trainer sets ``config_model``, and the GUI form, CLI ``--help``, and
``validate_hyperparams`` all read this one source.

Per-trainer: each trainer declares its own model (there is one trainer today). ``extra="forbid"``
so an unknown hyperparameter is a hard error, matching the old dict schema's rejection.
"""

from pydantic import BaseModel, ConfigDict, Field


class UltralyticsYOLOConfig(BaseModel):
    """Hyperparameters for ``UltralyticsYOLOTrainer.train`` (Ultralytics ``model.train``)."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(
        100,
        ge=1,
        description="Maximum training epochs.",
        json_schema_extra={"label": "Epochs"},
    )
    batch_size: int = Field(
        8,
        ge=1,
        description=(
            "Images per training batch. The default fits an 8 GB GPU: measured on an RTX 2070 "
            "SUPER at imgsz 640 with yolo11s-seg, 8 used 3.0 GB and 32 needed 9.3 GB, about "
            "0.26 GB per image. Raise it on a larger GPU; a batch that does not fit is slow "
            "on Windows (it spills into shared memory) and stops with out-of-memory elsewhere."
        ),
        json_schema_extra={"label": "Batch size"},
    )
    imgsz: int = Field(
        640,
        ge=32,
        description="Training image size (square).",
        json_schema_extra={"label": "Image size", "unit": "px"},
    )
    patience: int = Field(
        100,
        ge=0,
        description="Early stopping: stop if no improvement for this many epochs (0 disables).",
        json_schema_extra={"label": "Patience"},
    )
