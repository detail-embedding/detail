"""Tracking config model (§13) — the typed shape of a tracking run plus the metadata each surface
reads. The tracker-specific ``hyperparams`` stay a tracker-owned dict: each tracker declares and
validates its own set via ``HYPERPARAMS_SCHEMA`` / ``validate_hyperparams`` (the per-tracker single
source), so the model types the field but leaves its checks — and the ``arena_ids`` normalization —
to ``TrackingModule.validate``, which returns the enriched config."""

from pydantic import BaseModel, Field

#: The tracker a run gets when nobody picks one. Named here rather than written into each
#: surface: the CLI's option default, the GUI's opening selection and the API's model default
#: are one fact, and three copies of it drift. They had already drifted — the CLI defaulted to
#: ``iou_tracker`` while the GUI opened on whichever id sorted first.
DEFAULT_TRACKER = "anchor_tracker"


class TrackingConfig(BaseModel):
    detection_run_id: str = Field(
        ...,
        description="Detection run to track: its per-frame detections are linked into tracks "
        "(e.g. det_01).",
        json_schema_extra={"label": "Detection run"},
    )
    tracker_id: str = Field(
        DEFAULT_TRACKER,
        description="Tracking algorithm to run, by registered tracker id. The default keeps "
        "one subject per frame: it takes the highest-confidence detection and attaches "
        "overlapping fragments of the same animal to it, which is the shape of every "
        "single-animal recording. Recordings with more than one subject are not supported yet.",
        json_schema_extra={"label": "Tracker"},
    )
    arena_name: str | None = Field(
        None,
        description=(
            "Restrict tracking to inside the field of view with this name. A name rather "
            "than an id, because an arena id is only unique within one recording: ids are "
            "minted per recording, so the same id is a different polygon on the recording "
            "next to it. `validate()` resolves this to each recording's own `arena_ids`."
        ),
        json_schema_extra={"label": "Field of view"},
    )
    arena_ids: list[str] | None = Field(
        None,
        description=(
            "The resolved field of view: this recording's own arena id (at most one). "
            "Normally derived from `arena_name`; set it directly to name one particular "
            "recording's arena. Stored under the kind `arena`, which is what the on-disk "
            "paths still call it."
        ),
        json_schema_extra={"label": "Field of view (resolved)"},
    )
    class_name: str | None = Field(
        None,
        description=(
            "Restrict tracking to detections of this class only, by the label the "
            "detection run stored on them (e.g. `black mouse`). Omit to link every "
            "class's detections together, which is right for a single-class detection "
            "run; a multi-class run (two animals told apart by detection instead of by "
            "tracking) needs one tracking run per class to get one clean track each. "
            "`validate()` checks the name against the classes the detection run actually "
            "has."
        ),
        json_schema_extra={"label": "Class"},
    )
    every_n_frames: int | None = Field(
        None,
        ge=1,
        description=(
            "Use every Nth frame of the video. Omit to link every frame detection produced. "
            "Counted in source frames, so it must be a multiple of what detection used. "
            "Going coarser here also thins what embedding can read, and makes linking harder: "
            "the subject moves further between the frames being compared."
        ),
        json_schema_extra={"label": "Use every"},
    )
    hyperparams: dict | None = Field(
        None,
        description="Tracker-specific hyperparameters; each tracker defines and validates its own "
        "set.",
        json_schema_extra={"label": "Hyperparameters"},
    )
