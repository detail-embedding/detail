"""Track-merge config model (§13) — the typed shape of a track-merge run.

``source_specs`` is a list of upstream *tracking* run specs (the same spec shape
``spec_groups``/``resolve_spec_in_recording`` already produce for every other upstream
picker), not a plain field a generic form can render — ``TrackMergeModule.validate``
resolves each to this recording's own tracking run id, checks they share one detection
run and use pairwise-distinct classes, and writes the resolved ``source_run_ids`` /
``detection_run_id`` back onto the config (Tenet 9)."""

from pydantic import BaseModel, Field


class TrackMergeConfig(BaseModel):
    source_specs: list[dict] = Field(
        ...,
        min_length=2,
        description="Two or more existing tracking runs to combine into one multi-subject "
        "run: one per animal, each already a class-filtered single-subject track (§13). "
        "Every source must be built from the same detection run and use a distinct class; "
        "the merged run gets one object_id per source, matching its own class's rows.",
        json_schema_extra={"label": "Tracks to merge"},
    )
