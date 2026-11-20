"""Labeling config model (§12) — the single source for the flat scalar parameters' metadata +
bounds for the *label a frame sample with a prompt-capable model* dataset source. Experiment-aware and
nested inputs (``classes``, ``source_recording_ids``, ``compute``, ``name``) stay in
``Experiment.label_dataset`` since they need recording data or are nested dataclasses.
"""

from typing import Literal

from pydantic import BaseModel, Field

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
    tether: bool = Field(
        False,
        description="The video shows a tether, cable or patch cord attached to the animal. "
        "Adds a second class, `cable` (prompt: cable), beside your classes, so the trained "
        "detector learns the cable as a thing of its own instead of as a faint piece of the "
        "animal. The dataset and the model record it as an auxiliary class, and tracking on "
        "that model leaves it out by default, so a cable is never the subject or merged into "
        "it. Works for any species.",
        json_schema_extra={"advanced": True, "label": "Tether or cable in the video"},
    )


#: The class a tethered video adds (§12 *Tether or cable in the video*). The prompt is "cable",
#: never "tether": on a tethered-mouse pilot SAM 3 found nothing at all for "tether" (0 of 10
#: frames) and found the cable on 9 of 10 frames for "cable".
TETHER_CLASS = "cable"
TETHER_PROMPT = "cable"


def with_tether(
    classes: list[LabelingClass], tether: bool
) -> tuple[list[LabelingClass], list[str]]:
    """``classes`` plus the tether class when ``tether`` is set, and the auxiliary class names.

    Returns the user's classes unchanged and ``[]`` when ``tether`` is off.

    Raises:
        ValueError: ``tether`` is set and a class is already named ``cable`` or already
            prompts ``cable``: the option would add the same class twice.
    """
    if not tether:
        return list(classes), []
    for c in classes:
        if c.name.strip().lower() == TETHER_CLASS or (
            c.prompt.strip().lower() == TETHER_PROMPT
        ):
            raise ValueError(
                f"class {c.name!r} is already the cable; turn off Tether or cable in the "
                "video, or rename that class, so the cable is labelled once"
            )
    next_id = max((c.class_id for c in classes), default=-1) + 1
    cable = LabelingClass(class_id=next_id, name=TETHER_CLASS, prompt=TETHER_PROMPT)
    return [*classes, cable], [TETHER_CLASS]
