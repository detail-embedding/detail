"""Recording visualization recipes — one recording (or one range of it) played beside its path
through pose space and its location (DESIGN_PLAN §22 "Recording visualization").

The ``configs/display.py`` kind: how a finished clustering run is *drawn*, not what it computed.
Each knob has its bounds, default and help here, so the window's controls, the CLI's help and the
saved recipe all read one source. A saved recipe is exactly this model's dump.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

#: The most clusters a cut draws in colour: the categorical palette's length (the dataviz rule:
#: a ninth series folds into "other", never a generated hue).
MAX_COLOURED = 8

Theme = Literal["dark", "light"]
ClusterMode = Literal["named", "cut"]
ExportSpeed = Literal["real-time", "every-frame", "half-speed"]

#: The output rate a slide plays smoothly: a faster source is thinned to it ("real-time").
SLIDE_FPS = 50.0


def speed_timing(speed: str, source_fps: float) -> tuple[float, float]:
    """``(output fps, playback speed)`` for an export ``speed`` of a ``source_fps`` video."""
    fps = float(source_fps)
    if speed == "every-frame":
        return fps, 1.0
    if speed == "half-speed":
        return min(SLIDE_FPS, fps), 0.5
    return min(SLIDE_FPS, fps), 1.0


class RecordingVisualizationConfig(BaseModel):
    """One visualization: which recording, which stretch, which clusters, and what is drawn."""

    recording_id: str = Field(
        ...,
        description="The recording to show, one of the clustering run's own recordings.",
        json_schema_extra={"label": "Recording"},
    )
    track_id: int = Field(
        0,
        ge=0,
        description="The animal (track) to follow, as the clustering run embedded it.",
        json_schema_extra={"label": "Animal"},
    )
    start_s: float | None = Field(
        None,
        ge=0.0,
        description="Where the visualization starts, in seconds of the recording. Leave both "
        "start and end empty for the full recording.",
        json_schema_extra={"label": "From (s)", "ui_step": 1.0},
    )
    end_s: float | None = Field(
        None,
        gt=0.0,
        description="Where it ends, in seconds of the recording.",
        json_schema_extra={"label": "To (s)", "ui_step": 1.0},
    )
    clusters: ClusterMode = Field(
        "named",
        description="Which clusters are coloured: the run's named clusters, or the numbered "
        "clusters of a cut across its tree.",
        json_schema_extra={"label": "Clusters"},
    )
    cut_height: float | None = Field(
        None,
        ge=0.0,
        description="Where the cut crosses the tree (merge height). Every branch crossing it "
        "is a numbered cluster.",
        json_schema_extra={"label": "Cut at"},
    )
    cut_included: list[int] = Field(
        default_factory=list,
        description=f"The cut's cluster numbers drawn in colour (at most {MAX_COLOURED}); "
        "the rest are drawn grey, as other.",
        json_schema_extra={"label": "Coloured clusters"},
    )
    location: bool = Field(
        True,
        description="Add Location Export's path over time: where the animal was on the "
        "floor over the last seconds, time running up.",
        json_schema_extra={"label": "Location panel"},
    )
    trail_s: float = Field(
        2.0,
        ge=0.2,
        le=10.0,
        description="How far back the bright trail in pose space and the outlines reach.",
        json_schema_extra={"label": "Trail (s)", "ui_step": 0.5},
    )
    ghost_s: float = Field(
        8.0,
        ge=0.0,
        le=60.0,
        description="How far back the faint path in pose space reaches.",
        json_schema_extra={"label": "Faint path (s)", "ui_step": 1.0},
    )
    ethogram_half_s: float = Field(
        10.0,
        ge=1.0,
        le=120.0,
        description="How much of the ethogram shows either side of now.",
        json_schema_extra={"label": "Ethogram span (± s)", "ui_step": 1.0},
    )
    location_window_s: float = Field(
        10.0,
        ge=1.0,
        le=120.0,
        description="How much of the past the location panel shows.",
        json_schema_extra={"label": "Location window (s)", "ui_step": 1.0},
    )

    @model_validator(mode="after")
    def _consistent(self) -> "RecordingVisualizationConfig":
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError(
                "give both start_s and end_s for a range, or neither for the full recording"
            )
        if self.start_s is not None and self.end_s <= self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) must be after start_s ({self.start_s})"
            )
        if self.clusters == "cut":
            if self.cut_height is None:
                raise ValueError(
                    "clusters 'cut' needs cut_height: where the cut crosses the tree"
                )
            if len(self.cut_included) > MAX_COLOURED:
                raise ValueError(
                    f"at most {MAX_COLOURED} clusters can be coloured, got "
                    f"{len(self.cut_included)}: the rest are drawn grey, as other"
                )
            if any(n < 1 for n in self.cut_included):
                raise ValueError("cluster numbers start at 1")
        elif self.cut_included or self.cut_height is not None:
            raise ValueError(
                "cut_height and cut_included belong to clusters 'cut'; clear them for 'named'"
            )
        return self


class RecordingVisualizationExport(BaseModel):
    """How a visualization (or a montage of them) is written."""

    themes: list[Theme] = Field(
        default_factory=lambda: ["dark"],
        min_length=1,
        description="Background: dark (black), light (white), or both, one file each.",
        json_schema_extra={"label": "Background"},
    )
    speed: ExportSpeed = Field(
        "real-time",
        description="real-time: as fast as it happened, at up to 50 fps (a faster video is "
        "thinned to that); every-frame: real time with every frame, at the video's own rate; "
        "half-speed: half as fast, at up to 50 fps.",
        json_schema_extra={"label": "Speed"},
    )
    width: int = Field(
        1920,
        ge=640,
        le=3840,
        description="The video's width in pixels; the height keeps 16:9.",
        json_schema_extra={"label": "Width (px)", "ui_step": 320},
    )
    workers: int | None = Field(
        None,
        ge=1,
        le=64,
        description="Processes rendering frames at once. Empty: one per core but two.",
        json_schema_extra={"label": "Workers"},
    )

    @property
    def height(self) -> int:
        return int(round(self.width * 9 / 16)) // 2 * 2
