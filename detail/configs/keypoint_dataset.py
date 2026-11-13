"""Keypoint dataset config model (§12 *Distill keypoints*): the flat parameters of the label
stage that turns a teacher's placements into a YOLO-pose training set. The teacher is either a
finished ``keypoint_detect`` run per recording, or the patch method itself (a keypoint model
with an example set) run on the sampled frames.

The teacher (its runs per recording, or the method and its example set) and the dataset's name
are experiment-aware inputs of ``Experiment.keypoint_dataset`` and stay outside the model, as
``classes`` and ``source_recording_ids`` stay outside ``LabelingConfig`` (§6.5). The score
cut-off is resolved from the teacher's method when left unset and written back (Tenet 9).
"""

from typing import Literal

from pydantic import BaseModel, Field


class KeypointDatasetConfig(BaseModel):
    """Sample ``n_frames`` from the teacher's rows (shared across recordings by row count, one
    draw per equal time slice), decode each exactly, and write one YOLO-pose label line per
    tracked animal: its box, then every skeleton point as ``x y v``."""

    n_frames: int = Field(
        ...,
        ge=1,
        description=(
            "Frames to label across the chosen recordings, shared out by each recording's "
            "tracked rows, one draw per equal time slice."
        ),
        json_schema_extra={"label": "Frames to label"},
    )
    val_split: float = Field(
        0.15,
        ge=0.0,
        lt=1.0,
        description="Fraction of labeled frames held out for validation, per recording.",
        json_schema_extra={"advanced": True, "label": "Validation split"},
    )
    score_cutoff: float | None = Field(
        None,
        ge=0.0,
        description=(
            "A point placed with a score below this is taught as not visible (v = 0); the box "
            "is still taught. Omit for the teacher method's own default cut-off, recorded on "
            "the dataset."
        ),
        json_schema_extra={"label": "Score cut-off"},
    )
    exclude_audit_s: float = Field(
        1.0,
        ge=0.0,
        description=(
            "Never use an audit frame, nor any frame within this many seconds of one, so the "
            "labels that score a method are never what builds it."
        ),
        json_schema_extra={
            "advanced": True,
            "label": "Keep clear of audit frames",
            "unit": "s",
        },
    )
    every_n_frames: int = Field(
        1,
        ge=1,
        description=(
            "Consider only every n-th teacher row as a candidate, so neighbouring frames of "
            "one posture are not drawn together."
        ),
        json_schema_extra={"advanced": True, "label": "Candidate stride"},
    )
    # What the student is trained on, and so what it is run on (§12 *Distill keypoints*).
    student_input: Literal["whole_frame", "crop"] = Field(
        "whole_frame",
        description=(
            "What the student sees. whole_frame: the whole video frame, one label per "
            "tracked animal, so one pass per frame serves every animal. crop: a square crop "
            "around each tracked animal, the animal centred, every animal reaching into it "
            "labelled; one pass per animal, for overlapping animals or a small animal in a "
            "wide view. The student is run the way it was trained."
        ),
        json_schema_extra={"label": "Student input"},
    )
    student_crop_px: int = Field(
        448,
        ge=64,
        le=1280,
        description=(
            "crop only: the side of the square crop, in pixels. The crop is the tracked box's "
            "longer side plus a margin, resized to this, so every animal is the same size to "
            "the student."
        ),
        json_schema_extra={"advanced": True, "label": "Crop size", "unit": "px"},
    )
