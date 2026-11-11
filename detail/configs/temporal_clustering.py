"""Temporal-clustering module parameters (§22) — what a run is *given*, not how it clusters.

The module's half. The method's own knobs live in ``configs/temporal_dtw_finch.py``, split the way
the removed ``configs/pose_clustering.py`` split from ``configs/pose_finch.py``. The split is
sharper here than it was for pose: **segmentation belongs to the method**, so the window, the
detector, the minimum bout length and the DTW band are all on the method's side. A module that
carried them would be carrying one grouping rule's internals on every other rule's behalf.

The GUI form (`gui/_pydantic_form.py`) and the CLI's ``--help`` read this model's field metadata,
so a parameter's range or default is written once here rather than hand-copied into a widget and an
option decorator (§6.5).
"""

from pydantic import BaseModel, Field


class TemporalClusteringConfig(BaseModel):
    """What a temporal-clustering fit is given, before any authored granularity is chosen."""

    source_recording_ids: list[str] = Field(
        default_factory=list,
        description="Recordings to fit the vocabulary on. Every recording's bouts are pooled into "
        "one vocabulary, so a cluster common in one treatment and absent in another is directly "
        "comparable.",
        json_schema_extra={"scope": "selection", "label": "Recordings"},
    )
    embed_run_id: str | None = Field(
        None,
        description="Embedding run to read pose trajectories from, by id (e.g. emb_02): that id "
        "in every selected recording. Left unset, each recording's own run with the one set of "
        "embedding settings they all have is used, whatever its id, and the choice is refused when "
        "they share several: a vocabulary silently fitted on whichever sorted first would score "
        "differently tomorrow.",
        json_schema_extra={"scope": "selection", "label": "Embedding run"},
    )
    track_id: int = Field(
        0,
        ge=0,
        description="Track (subject) to cluster, in each recording.",
        json_schema_extra={"label": "Track"},
    )
    method: str = Field(
        "codebook_viterbi",
        description="Clustering method. codebook_viterbi builds a codebook of poses from a "
        "sample of each recording and decodes the whole trajectory against it; its cost grows "
        "in proportion to the video pooled. dtw_finch cuts each trajectory into bouts, compares every pair with banded DTW "
        "and groups them with FINCH; it draws slightly better boundaries, but its cost grows with "
        "the square of the bout count. The two tie on vocabulary quality. A registry key rather "
        "than a hardcoded call, so a further method needs no change here.",
        json_schema_extra={"label": "Method"},
    )
