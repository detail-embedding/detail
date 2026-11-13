"""Location-export config model (§16) — the typed shape of a location-export run. Both fields are
experiment references (a recording selection + a tracking run); their existence and completeness are
checked in ``LocationExportModule.validate``, which needs the experiment, so the model carries no
bounds — only the types + the metadata each surface (CLI help, docs) reads."""

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
        description="A keypoint track run (kt_NN) to also export each animal's points from, "
        "as keypoints.csv. It must come from the same tracking run. Unset: no keypoints.",
        json_schema_extra={"scope": "selection", "label": "Keypoint track run"},
    )
    pair_edge_distance: bool = Field(
        False,
        description="Also measure the nearest distance between two tracks' masks (0 when "
        "they touch) in pair_relations.csv. Slower: it compares the masks on every frame.",
        json_schema_extra={"label": "Distance between mask edges", "advanced": True},
    )
