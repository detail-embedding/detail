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
        32,
        ge=1,
        description="Images per training batch.",
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
