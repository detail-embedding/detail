"""The DTW+FINCH temporal method's own parameters (§22) — the method's half of the config.

Split from ``configs/temporal_clustering.py`` the way the removed ``configs/pose_finch.py`` split
from ``configs/pose_clustering.py``, and for the same reason: **segmentation belongs to the
method**. Where a bout starts and ends, how short is too short to be one, and how much warp the
comparison tolerates are decisions this grouping rule is entitled to make differently from the next
one. A knob that lived on the module would be one the module had to understand on every method's
behalf.

Defaults are the ones internal evaluations measured, and each carries its evidence in its help.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TemporalDtwFinchConfig(BaseModel):
    """What the DTW+FINCH method needs, beyond the trajectories the module hands it."""

    window_s: float = Field(
        0.53,
        ge=0.0,
        description="Sliding-window length (seconds) the pose trajectory is pooled over before "
        "segmentation, which smooths per-frame jitter. About 0.53 s is the tuned 8-frame "
        "window at a 15 fps embed rate; the rows it resolves to depend on your embed cadence.",
        json_schema_extra={"label": "Window", "unit": "s"},
    )
    pool: Literal["mean", "max", "flatten"] = Field(
        "mean",
        description="How each window is pooled into one feature: mean (average pose), max "
        "(per-dimension peak, so a brief distinctive pose is not averaged away) or flatten "
        "(concatenate the frames).",
        json_schema_extra={"advanced": "Segmentation", "label": "Window pooling"},
    )
    feat_stride: int = Field(
        1,
        ge=1,
        description="Step (embedded frames) between successive window positions. Raise it to "
        "subsample windows, which is coarser and faster.",
        json_schema_extra={"advanced": "Segmentation", "label": "Window stride"},
    )
    detector: Literal["binseg", "novelty", "speed", "abd"] = Field(
        "binseg",
        description="Transition detector that cuts the trajectory into bouts. binseg is the "
        "default on measurement: it cuts better than novelty at the same rule and vocabulary "
        "size. novelty is a windowed cosine discrepancy, speed takes cosine-step peaks, and abd "
        "deliberately over-cuts for a rule that merges the pieces back.",
        json_schema_extra={"label": "Detector"},
    )
    # The four detectors' own knobs. Named to match `ClusterFitConfig`'s, because
    # `bout_segmentation.detector_params` assembles the per-detector dict by attribute and this
    # model is passed to it — one assembly rule rather than a second copy that could disagree
    # about what "penalty" means.
    penalty: float = Field(
        40.0,
        ge=0.0,
        description="binseg stop threshold. Lower cuts more, giving more and shorter bouts.",
        json_schema_extra={"advanced": "Segmentation", "label": "binseg penalty"},
    )
    percentile: float = Field(
        96.0,
        ge=0.0,
        le=100.0,
        description="novelty/speed peak threshold percentile. Lower cuts more.",
        json_schema_extra={"advanced": "Segmentation", "label": "Peak percentile"},
    )
    novelty_half_s: float = Field(
        0.14,
        gt=0.0,
        description="novelty comparison window half-width, in seconds.",
        json_schema_extra={
            "advanced": "Segmentation",
            "label": "Novelty window",
            "unit": "s",
        },
    )
    abd_smooth_half_s: float = Field(
        0.034,
        ge=0.0,
        description="abd: how long either side is averaged before looking for a change. Keep it "
        "small, because smoothing wider than the pose's own correlation length blurs away the "
        "transitions it is meant to sharpen.",
        json_schema_extra={
            "advanced": "Segmentation",
            "label": "abd smoothing",
            "unit": "s",
        },
    )
    abd_nms_window_s: float = Field(
        0.085,
        gt=0.0,
        description="abd: how far apart in time two cuts must be. A cut survives only if it is "
        "the strongest change within this long either side.",
        json_schema_extra={
            "advanced": "Segmentation",
            "label": "abd spacing",
            "unit": "s",
        },
    )
    min_size_s: float = Field(
        0.10,
        gt=0.0,
        description="Minimum time between successive cuts, in seconds, for every detector.",
        json_schema_extra={
            "advanced": "Segmentation",
            "label": "Min cut spacing",
            "unit": "s",
        },
    )
    max_gap_s: float | None = Field(
        None,
        gt=0.0,
        description="Break the trajectory into continuous runs wherever tracking is missing for "
        "longer than this. Omit to derive it from the embed cadence, which is what makes a "
        "tracking gap a boundary rather than a bout that spans one.",
        json_schema_extra={
            "advanced": "Segmentation",
            "label": "Max tracking gap",
            "unit": "s",
        },
    )
    min_len_s: float = Field(
        0.27,
        ge=0.0,
        description="Drop bouts shorter than this (seconds) before clustering. This is why the "
        "method needs no separate minimum-run floor the way a frame-labeling method does: a bout "
        "is already an extent, and one too short to be a behavior never reaches the comparison. "
        "About 0.27 s is the tuned 4-frame minimum at a 15 fps embed rate.",
        json_schema_extra={"label": "Min bout length", "unit": "s"},
    )
    band_s: float = Field(
        0.27,
        ge=0.0,
        description="DTW warp tolerance when comparing bouts (seconds; 0 = rigid). This is what "
        "lets a slow and a fast performance of the same behavior match. Larger merges more bouts "
        "into fewer, coarser clusters.",
        json_schema_extra={"label": "DTW band", "unit": "s"},
    )
    max_segments: int | None = Field(
        None,
        ge=2,
        description="Cap the bouts compared against each other, evenly subsampled. The DTW "
        "comparison is O(bouts²) and is the whole cost of a fit, so this is the only bound on "
        "it. A capped fit is stated on the interpretation screen, because it is identical to a "
        "full one in every other respect. Omit to compare every bout.",
        json_schema_extra={
            "advanced": "Clustering",
            "label": "Max bouts",
            "ui_max": 1_000_000,
        },
    )
