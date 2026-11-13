"""Distill config model (§12) — the unified "distill a model" run's parameters: label a
frame sample with a prompt-capable model, then train a closed-set YOLO detector on those labels.

It **composes** the two existing stage configs rather than re-homing their fields (§6.5):
``labeling`` is a :class:`LabelingConfig` and ``training`` is a :class:`UltralyticsYOLOConfig`,
so each stage's bounds/help/labels stay in one place. The experiment-aware inputs — ``classes``,
``source_recording_ids``, and the two per-stage compute targets — are ``Experiment.distill``
arguments (they need recording data or are nested dataclasses), the same split ``label_dataset``
uses for ``classes``/``compute``.
"""

from pydantic import BaseModel, ConfigDict, Field

from detail.configs.labeling import LabelingConfig
from detail.configs.training import UltralyticsYOLOConfig

# Frames the Distill dialog opens at, for a detector or a keypoint model. ``n_frames`` stays
# required in both labeling configs, so the API and CLI still make the caller choose.
DEFAULT_N_FRAMES = 1000


class DistillConfig(BaseModel):
    """The scalar knobs of a distill run. ``extra="forbid"`` so an unknown key is a hard error,
    matching the two nested stage configs."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Distill run name; seeds the dataset and trained-model names.",
        json_schema_extra={"label": "Name"},
    )
    base_model: str = Field(
        "yolo11s-seg.pt",
        description="The student: a pretrained YOLO-seg to fine-tune (larger = more accurate, slower).",
        json_schema_extra={"label": "Student model"},
    )
    trainer_id: str = Field(
        "ultralytics_yolo",
        description="Training framework that fits the student on the labeled dataset.",
        json_schema_extra={"label": "Training framework"},
    )
    stage_videos: bool = Field(
        False,
        description=(
            "Copy every source recording's video to the remote cluster in the background while "
            "labeling and training run, so the detection run that follows finds it already "
            "there. Opt-in: it can be 100+ GB. Needs a remote labeling or training target."
        ),
        json_schema_extra={"label": "Upload videos in the background"},
    )
    labeling: LabelingConfig = Field(
        ...,
        description="Labeling-stage parameters (prompt-capable model, frames to label, sampling).",
    )
    training: UltralyticsYOLOConfig = Field(
        default_factory=UltralyticsYOLOConfig,
        description="Training-stage hyperparameters (epochs, batch size, image size, patience).",
    )
