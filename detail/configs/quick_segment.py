"""Quick-segment tool config model (§27) — the single source for its scalar parameters
(mode, frame count) and the typed shape of a quick-segment run. The non-scalar
inputs (the video paths, the ``Prompt`` list, and the ``ComputeTarget``) stay their own
types, assembled by each surface (GUI form / CLI flags) and passed alongside this config,
exactly as detection does with its prompt / compute (see ``configs/detection.py``).

There is deliberately **no confidence parameter here.** A quick-segment run keeps every
proposal the model found, and the confidence a result is *shown* at is chosen afterwards —
``configs/display.ConfidenceView``, a slider in the GUI and ``--min-conf`` in the CLI.
Baking a threshold into the run would throw away the rows the slider exists to show.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QuickSegmentConfig(BaseModel):
    mode: Literal["sample", "consecutive"] = Field(
        "sample",
        description=(
            "sample: spread the frames evenly across each clip. The default, because a "
            "still animal or an empty opening makes the first N frames show nothing. "
            "consecutive: read straight through from frame 0, which is what an overlay "
            "video needs."
        ),
        json_schema_extra={"label": "Frames"},
    )
    n_frames: int = Field(
        50,
        ge=1,
        description=(
            "How many frames to segment per clip. Ignored when 'whole video' is set. "
            "Every frame is run through the model, so this is what the run costs."
        ),
        json_schema_extra={"label": "Frame count", "ui_max": 100_000},
    )
    whole_video: bool = Field(
        False,
        description=(
            "Segment every frame of each clip (consecutive mode only). No cap, so a long "
            "recording can run for hours; the run is stoppable."
        ),
        json_schema_extra={"label": "Whole video"},
    )

    @model_validator(mode="after")
    def _whole_video_is_consecutive_only(self) -> "QuickSegmentConfig":
        # Sampling every frame is just a consecutive sweep, so the combination has no
        # meaning — reject it rather than silently picking one.
        if self.whole_video and self.mode != "consecutive":
            raise ValueError("whole_video applies to mode='consecutive' only")
        return self
