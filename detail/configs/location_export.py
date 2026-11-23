"""Location-export config model (§16) — the typed shape of a location-export run.

Two kinds of field. The selection (the recordings and the tracking run) is checked in
``LocationExportModule.validate``, which needs the experiment. The rest are choices about what
is measured and how often, and this model is the single source for their bounds, defaults and
help: the compute (``analysis/locations.py``), the CLI options and the launch form all read it.
"""

from pydantic import BaseModel, Field


class LocationExportConfig(BaseModel):
    source_recording_ids: list[str] = Field(
        default_factory=list,
        description="Recordings whose tracks to export (the analysis run's selection).",
        json_schema_extra={"scope": "selection", "label": "Recordings"},
    )
    tracking_run_id: str | None = Field(
        None,
        description="Completed tracking run to read tracks from, common to every selected "
        "recording (e.g. track_01).",
        json_schema_extra={"scope": "selection", "label": "Tracking run"},
    )
    keypoint_run_id: str | None = Field(
        None,
        description="A completed keypoint track run to export beside the tracks "
        "(keypoints.csv: where each skeleton point was, on the rows tracks.csv has), common to every selected "
        "recording (e.g. kt_01). Its poses must have been assigned to the tracking run this "
        "export reads. Unset: no keypoints.",
        json_schema_extra={"scope": "selection", "label": "Keypoint track run"},
    )
    sample_fps: float | None = Field(
        None,
        gt=0.0,
        description="Rows to write per second for each subject. Leave unset to write every "
        "tracked frame. When set, the tracked frame nearest each tick of a clock at this "
        "rate is kept and nothing is interpolated, so a stretch with no tracking has no "
        "rows. A few per second is enough for distance, speed and body shape, and is many "
        "times faster than every frame.",
        json_schema_extra={"label": "Rows per second", "ui_max": 240.0},
    )
    remove_thin_parts: bool = Field(
        False,
        description="Leave thin parts (a tail, a limb, an antenna) out of the body before "
        "its centre, area, length and width are measured. Recommended for rats and mice, "
        "whose tail otherwise moves the centre and dominates the length. Leave off for an "
        "animal whose thin parts are its body, such as a fish or a snake.",
        json_schema_extra={"label": "Remove thin parts"},
    )
    thin_part_scale: float = Field(
        0.12,
        gt=0.0,
        le=0.5,
        description="How thick a part must be to count as body when thin parts are "
        "removed: the smallest thickness kept, as a share of the square root of the "
        "subject's own typical mask area. Sized from the animal, so it does not depend on "
        "the distance calibration. 0.12 removes a rat's tail in top-down and side views.",
        json_schema_extra={"advanced": True, "label": "Thin part scale"},
    )


def option_names() -> tuple[str, ...]:
    """The fields that are choices about the measurement rather than the selection, in the
    order a surface shows them. Read off the model, so a field added to it reaches every
    surface that sets or records the options without being listed a second time."""
    return tuple(
        name
        for name, field in LocationExportConfig.model_fields.items()
        if (field.json_schema_extra or {}).get("scope") != "selection"
    )


def options_from(values: dict) -> LocationExportConfig:
    """The validated options in ``values``, which may also hold other keys of a run config.

    Raises:
        ValueError: An option outside its bounds (pydantic's message names the field).
    """
    return LocationExportConfig.model_validate(
        {name: values[name] for name in option_names() if name in values}
    )
