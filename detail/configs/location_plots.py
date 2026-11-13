"""Location-plot view parameters — how a finished location export is *drawn*, not what it
computed (DESIGN_PLAN §16 "Location plots").

The ``configs/display.py`` kind: each knob still has choices, bounds, a default and help, so it
lives in a model the GUI controls and the CLI help both read. Not part of any run's config and
not recorded in a run manifest — two people plotting one run two ways are looking at one run.
"""

from typing import Literal

from pydantic import BaseModel, Field

PlotKind = Literal["trace", "heatmap", "speed", "trace3d", "distance", "distance_hist"]


class LocationPlotView(BaseModel):
    """What a location plot shows and how."""

    kind: PlotKind = Field(
        "trace",
        description="trace: the 2-D path. heatmap: where the animal spent its time. speed: "
        "speed over time. trace3d: the path against time, in 3-D. distance: the distance "
        "between two tracks over time. distance_hist: how often they were at each distance.",
        json_schema_extra={"label": "Plot"},
    )
    source: str = Field(
        "mask",
        description="The animal's position: 'mask' (the mask's centroid, truer to its shape), "
        "'bbox' (the bounding box's), or a keypoint's name (e.g. 'nose'), when the export "
        "read a keypoint track run.",
        json_schema_extra={"label": "Position"},
    )
    points: list[str] | None = Field(
        None,
        description="Several keypoints to draw on a path plot, each in its own colour (one "
        "animal per panel). Unset: the one Position.",
        json_schema_extra={"label": "Keypoints"},
    )
    units: Literal["mm", "px", "norm"] | None = Field(
        None,
        description="Auto: mm when every plotted recording is calibrated, otherwise "
        "pixels. Plots by group pool positions in normalized coordinates (0-1 of the "
        "frame), the only frame shared across videos.",
        json_schema_extra={"label": "Units"},
    )
    color_by: Literal["auto", "time", "speed", "subject"] = Field(
        "auto",
        description="What colours a path: elapsed time, speed, or which animal it is. Auto: "
        "by animal when a panel shows more than one, otherwise by time.",
        json_schema_extra={"label": "Colour by"},
    )
    subjects: list[str] | None = Field(
        None,
        description="Which animals to plot, by their label (a track merge names each one, "
        "e.g. 'white mouse'; a run without labels calls them 'subject 0', 'subject 1', …). "
        "A label is matched in each recording, whatever its track number there. Unset: "
        "every animal.",
        json_schema_extra={"label": "Animals"},
    )
    group_by: str | None = Field(
        None,
        description="A Recording Information column to group recordings by (one panel per "
        "group). Unset: one panel per recording.",
        json_schema_extra={"label": "Group by"},
    )
    split_by_subject: bool = Field(
        False,
        description="Also group by animal label, alone or crossed with Group by (e.g. "
        "genotype × resident / intruder).",
        json_schema_extra={"label": "Group by animal"},
    )
    pairs: list[str] | None = Field(
        None,
        description="Which pairs of tracks a distance plot shows, by their labels joined "
        "with ' – ' (e.g. 'mouse – object'), matched in each recording. Unset: every "
        "pair.",
        json_schema_extra={"label": "Pairs"},
    )
    distance_by: Literal["mask_centroid", "bbox_centroid", "mask_edge"] = Field(
        "mask_centroid",
        description="What a distance between two tracks measures: between their mask "
        "centroids, their box centroids, or the nearest edges of their masks (0 when they "
        "touch; only on an export that measured it).",
        json_schema_extra={"label": "Distance between"},
    )
    hist_bins: int = Field(
        40,
        ge=5,
        le=200,
        description="Bins of a distance histogram, from 0 to the largest distance plotted.",
        json_schema_extra={"label": "Histogram bins"},
    )
    bins: int = Field(
        40,
        ge=5,
        le=200,
        description="Heat-map cells along the frame's longer side.",
        json_schema_extra={"label": "Heat-map bins"},
    )
    time_bin_s: float = Field(
        1.0,
        ge=0.04,
        le=600.0,
        description="Width of one time bin on a speed-over-time plot, in seconds.",
        json_schema_extra={"label": "Speed bin (s)", "ui_step": 0.5},
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
        description="Most samples drawn per path, evenly spaced. Statistics use every "
        "sample; this only thins the drawing.",
        json_schema_extra={"label": "Drawn points", "advanced": True},
    )


class LocationMontageView(BaseModel):
    """How a location export's recordings play side by side (§16 "Preview and montage")."""

    start_s: float = Field(
        0.0,
        ge=0.0,
        le=86_400.0,
        description="Where the window starts, in seconds of each recording.",
        json_schema_extra={"label": "Start (s)", "ui_step": 1.0},
    )
    length_s: float = Field(
        30.0,
        ge=0.5,
        le=86_400.0,
        description="How long the window is, in seconds; every recording plays the same "
        "stretch, in sync.",
        json_schema_extra={"label": "Length (s)", "ui_step": 5.0},
    )
    full_length: bool = Field(
        False,
        description="Play each recording from its first to its last tracked frame instead "
        "of the window, so they end at different times.",
        json_schema_extra={"label": "Full length"},
    )
    full_frame: bool = Field(
        False,
        description="Show the whole frame instead of cropping to the animals.",
        json_schema_extra={"label": "Full frame"},
    )
    speed: float = Field(
        1.0,
        ge=0.25,
        le=4.0,
        description="Play and export faster (2 = twice real time) or slower (0.5).",
        json_schema_extra={"label": "Speed", "ui_step": 0.25},
    )
    raw: bool = Field(
        False,
        description="The video footage under the drawing.",
        json_schema_extra={"label": "Raw"},
    )
    mask: bool = Field(
        True,
        description="Each animal's mask, in its colour.",
        json_schema_extra={"label": "Mask"},
    )
    keypoints: bool = Field(
        False,
        description="Each animal's keypoints, when the export read a keypoint track run.",
        json_schema_extra={"label": "Keypoints"},
    )
    mask_trail: bool = Field(
        True,
        description="Fading copies of each mask over the last half second.",
        json_schema_extra={"label": "Mask trail"},
    )
    keypoint_trail: bool = Field(
        False,
        description="Each animal's path over the last half second, in its colour: its "
        "keypoints, or the centre of its box without keypoints.",
        json_schema_extra={"label": "Keypoint trail"},
    )
    trail_points: list[str] | None = Field(
        None,
        description="Which keypoints leave a trail. Unset: every point.",
        json_schema_extra={"label": "Trail points"},
    )

    def layers(self):
        """The montage engine's ``MontageLayers`` for these ticks."""
        from detail.api.clip_montage import MontageLayers

        return MontageLayers(
            raw=self.raw,
            mask=self.mask,
            keypoints=self.keypoints,
            mask_trail=self.mask_trail and self.mask,
            keypoint_trail=self.keypoint_trail and self.keypoints,
            trail_points=(
                None if self.trail_points is None else tuple(self.trail_points)
            ),
        )
