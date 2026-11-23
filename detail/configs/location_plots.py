"""Location-plot view parameters — how a finished location export is *drawn*, not what it
computed (DESIGN_PLAN §16 "Location plots").

The ``configs/display.py`` kind: each knob still has choices, bounds, a default and help, so it
lives in a model the GUI controls and the CLI flags both read. Not part of any run's config and
not recorded in a run manifest — two people plotting one run two ways are looking at one run.

The path's smoothing and jump rule are a comparison's own (``configs/comparison.py``), read
from there rather than written again, so a speed plot and a comparison's speed measure one path.
"""

from typing import Literal

from pydantic import BaseModel, Field

from detail.configs.comparison import ComparisonConfig

PlotKind = Literal["trace", "heatmap", "speed", "trace3d", "body", "body_hist"]
BodyMeasure = Literal["length", "width", "area", "elongation", "mask_area"]

_COMPARISON = ComparisonConfig.model_fields


class LocationPlotView(BaseModel):
    """What a location plot shows and how."""

    kind: PlotKind = Field(
        "trace",
        description="trace: the 2-D path. heatmap: where the animal spent its time. speed: "
        "speed over time. trace3d: the path against time, in 3-D. body: a measure of the "
        "body's shape or size over time. body_hist: how often the body measured each value.",
        json_schema_extra={"label": "Plot"},
    )
    source: str = Field(
        "body",
        description="The animal's position: 'body' (the centre of its body, without the "
        "parts the export left out, such as a tail; on an export made before the body was "
        "measured, the mask's centre, which is the same thing), 'mask' (the centre of the "
        "whole mask), 'bbox' (the bounding box's centre), or a keypoint's name (e.g. "
        "'nose') when the export read a keypoint track run.",
        json_schema_extra={"label": "Position"},
    )
    points: list[str] | None = Field(
        None,
        description="Several keypoints to draw on a path plot, each in its own colour. "
        "Unset: the one Position.",
        json_schema_extra={"label": "Keypoints"},
    )
    body_measure: BodyMeasure = Field(
        "length",
        description="What a body plot shows: the body's length or width (along and across "
        "its long axis), its area, its elongation (length ÷ width), or the area of the whole "
        "mask, tail and all.",
        json_schema_extra={"label": "Body measure"},
    )
    units: Literal["mm", "px", "norm"] | None = Field(
        None,
        description="Auto: mm when every plotted recording is calibrated, otherwise "
        "pixels. Plots by group pool positions in normalized coordinates (0-1 of the "
        "frame), the only frame shared across videos.",
        json_schema_extra={"label": "Units"},
    )
    color_by: Literal["time", "speed"] = Field(
        "time",
        description="What colours a path: elapsed time, or speed on the smoothed path.",
        json_schema_extra={"label": "Colour by"},
    )
    group_by: str | None = Field(
        None,
        description="A Recording Information column to group recordings by (one panel per "
        "group, each animal counting once). Unset: one panel per recording.",
        json_schema_extra={"label": "Group by"},
    )
    posture_every_s: float | None = Field(
        None,
        gt=0.0,
        le=3600.0,
        description="On a 2-D path, draw the body's outline every this many seconds: an "
        "ellipse as long and as wide as the body, along its long axis. Unset: none.",
        json_schema_extra={"label": "Body outline every (s)", "ui_step": 1.0},
    )
    bins: int = Field(
        40,
        ge=5,
        le=200,
        description="Heat-map cells along the frame's longer side.",
        json_schema_extra={"label": "Heat-map bins"},
    )
    hist_bins: int = Field(
        40,
        ge=5,
        le=200,
        description="Bins of a body histogram, from the smallest value plotted to the "
        "largest.",
        json_schema_extra={"label": "Histogram bins"},
    )
    time_bin_s: float = Field(
        1.0,
        ge=0.04,
        le=600.0,
        description="Width of one time bin on a speed or body plot over time, in seconds.",
        json_schema_extra={"label": "Time bin (s)", "ui_step": 0.5},
    )
    smooth_s: float = Field(
        _COMPARISON["location_smooth_s"].default,
        gt=0.0,
        le=60.0,
        description="Seconds the position is averaged over before speed is measured, as a "
        "comparison measures it. Without it, frame-to-frame jitter reads as movement.",
        json_schema_extra={"label": "Path smoothing (s)", "advanced": True},
    )
    jump_speed: float = Field(
        _COMPARISON["location_jump_speed"].default,
        gt=0.0,
        le=1000.0,
        description="A raw step faster than this many of the animal's own body lengths per "
        "second is a tracking jump, not movement: the path is broken there, drawn and "
        "measured, rather than joined. As a comparison breaks it.",
        json_schema_extra={"label": "Jump above (body lengths/s)", "advanced": True},
    )
    speed_clip_pct: float = Field(
        95.0,
        ge=50.0,
        le=100.0,
        description="Top of the speed colour scale, as a percentile of every plotted speed, "
        "so one fast outlier does not wash out the rest.",
        json_schema_extra={"label": "Speed colour cap (%)", "advanced": True},
    )
    elev: float = Field(
        22.0,
        ge=-90.0,
        le=90.0,
        description="3-D camera elevation in degrees. Every 3-D panel shares it.",
        json_schema_extra={"label": "Elevation", "ui_step": 1.0},
    )
    azim: float = Field(
        -60.0,
        ge=-180.0,
        le=180.0,
        description="3-D camera rotation in degrees. Every 3-D panel shares it.",
        json_schema_extra={"label": "Rotation", "ui_step": 1.0},
    )
    max_points: int = Field(
        20000,
        ge=500,
        le=1_000_000,
        description="Most samples drawn per path, evenly spaced. Speeds and histograms use "
        "every sample; this only thins the drawing.",
        json_schema_extra={"label": "Drawn points", "advanced": True},
    )
