"""Labeling config model (§12) — the single source for the flat scalar parameters' metadata +
bounds for the *label a frame sample with a prompt-capable model* dataset source. Experiment-aware and
nested inputs (``classes``, ``source_recording_ids``, ``compute``, ``name``) stay in
``Experiment.label_dataset`` since they need recording data or are nested dataclasses.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from detail.core.dataclasses import LabelingClass


class LabelingConfig(BaseModel):
    """Sample ``n_frames`` across the selected recordings (proportional to recording length) and label
    them with a prompt-capable model into a YOLO-seg dataset. ``sampling`` picks how frames are
    drawn within each video."""

    model_id: str = Field(
        "sam3",
        description="The teacher: a prompt-capable model that labels the frames (e.g. sam3).",
        json_schema_extra={"label": "Teacher model"},
    )
    model_variant: str | None = Field(
        None,
        description="Teacher size/variant, if the model has size options; omit for the default.",
        json_schema_extra={"label": "Teacher size"},
    )
    n_frames: int = Field(
        ...,
        ge=1,
        description="Total frames to sample and label, allocated proportionally to recording length.",
        json_schema_extra={"label": "Frames to label"},
    )
    sampling: Literal["uniform", "stratified"] = Field(
        "stratified",
        description="Within each video: 'stratified' spreads picks evenly across time (one draw "
        "per equal-width bin); 'uniform' is fully random.",
        json_schema_extra={"advanced": True, "label": "Frame sampling"},
    )
    logit_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for keeping a detection.",
        json_schema_extra={"advanced": True, "label": "Confidence threshold"},
    )
    min_contour_area: int = Field(
        100,
        ge=0,
        description="Drop mask contours smaller than this many pixels.",
        json_schema_extra={"advanced": True, "label": "Min contour area", "unit": "px"},
    )
    val_split: float = Field(
        0.15,
        ge=0.0,
        lt=1.0,
        description="Fraction of labeled frames held out for validation (stratified per recording).",
        # The GUI's box stops at 0.9, a practical cap: past it there is hardly a training set.
        # Declared here so all three labeling dialogs agree; the validator keeps `lt=1.0`.
        json_schema_extra={
            "advanced": True,
            "label": "Validation split",
            "ui_max": 0.9,
        },
    )
    background_share: float = Field(
        0.1,
        ge=0.0,
        le=0.5,
        description="Largest share of the dataset that may be empty background. A background "
        "image is one where the teacher found none of the classes; it is kept with an empty "
        "label so the trained detector learns what nothing there looks like. 0 keeps none.",
        json_schema_extra={"advanced": True, "label": "Background share"},
    )
    negative_prompt: str | None = Field(
        None,
        description="Something in the video that touches the animal but is not it, such as a "
        "tether, cable or patch cord. The teacher labels it as a class of its own, so the "
        "trained detector learns to tell it apart from your classes and never reports it. "
        "Leave empty for none. For a tethered animal, `cable` works with SAM 3 where "
        "`tether` finds nothing. Check it with Test prompts: it must not take part of the "
        "animal, such as a tail.",
        json_schema_extra={"advanced": True, "label": "Negative prompt"},
    )

    @field_validator("negative_prompt", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        """An empty or whitespace-only prompt is no negative prompt; any other is trimmed."""
        if isinstance(value, str):
            return value.strip() or None
        return value


def with_negative_prompt(
    classes: list[LabelingClass], negative_prompt: str | None
) -> tuple[list[LabelingClass], list[str]]:
    """``classes`` plus the negative prompt's class, and the negative class names (§12
    *Negative prompt*).

    The negative class is named after its prompt and takes the next class id after the
    user's, so the user's classes keep their ids and a trained model's output classes stay
    the first ones.

    Args:
        classes: The user's classes, the ones a trained model reports.
        negative_prompt: The text prompt for the thing to learn apart, or None or blank
            for none.

    Returns:
        The classes to label (the user's, then the negative class) and the negative class
        names. The user's classes unchanged and ``[]`` when there is no negative prompt.

    Raises:
        ValueError: a class is already named or prompted like the negative prompt, so the
            same thing would be labelled both as a class and as not one.
    """
    prompt = (negative_prompt or "").strip()
    if not prompt:
        return list(classes), []
    for c in classes:
        if prompt.lower() in (c.name.strip().lower(), c.prompt.strip().lower()):
            raise ValueError(
                f"the negative prompt {prompt!r} is also class {c.name!r}; a class is "
                "labelled as what you want detected, and the negative prompt as what you "
                "do not, so change one of them"
            )
    next_id = max((c.class_id for c in classes), default=-1) + 1
    negative = LabelingClass(class_id=next_id, name=prompt, prompt=prompt)
    return [*classes, negative], [prompt]
