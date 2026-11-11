"""Parameters for the ``soft_knn`` query method (§20).

Every one is a **multiplier or a unitless quantity**, and the two that describe time
(smoothing, the length cap) are multiples of the example clip's own duration rather than
seconds. That is not tidiness: a temporal window written in rows means a different duration on
every recording (``tests/configs/test_unit_conventions.py`` fails one), and a window written in
seconds would need retuning for every behavior, while a window expressed against the
exemplar does not.

Deliberately **no threshold**, like every §20 method: the cutoff is authored afterwards, per
behavior, and reprojects the artifact with no compute.
"""

from pydantic import BaseModel, Field


class SoftKnnQueryConfig(BaseModel):
    """Per-frame soft k-NN scoring, with instances formed from the peaks."""

    smoothing_mult: float = Field(
        0.35,
        gt=0.0,
        le=2.0,
        description=(
            "Smoothing window, as a multiple of the example clip's length. The raw per-frame "
            "score flickers, so it is smoothed before instances are formed. "
            "Smaller keeps brief instances separate; larger merges them."
        ),
        json_schema_extra={"label": "Smoothing", "ui_step": 0.05},
    )
    extent_fraction: float = Field(
        0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Where an instance ends, as a fraction of its own peak score. Each match grows "
            "outwards from its peak while the score stays above this fraction of it, so the "
            "extent follows the behavior instead of the example's length. Lower returns "
            "longer instances. At 0.5 the median match came out 2.51 s against a 2.75 s "
            "ground truth."
        ),
        json_schema_extra={"label": "Extent fraction", "ui_step": 0.05},
    )
    duration_max_mult: float = Field(
        4.0,
        ge=1.0,
        description=(
            "Longest match to return, as a multiple of the example clip's length. A hard cap "
            "on the growth above, so a long flat stretch of moderate score cannot become one "
            "enormous instance. It also sets how much context a split search needs, so a very "
            "large value makes chunking a long recording more expensive."
        ),
        json_schema_extra={"label": "Max duration", "ui_max": 20.0, "ui_step": 0.5},
    )
    k: int = Field(
        20,
        ge=1,
        description=(
            "Neighbours voting on each frame. Each frame's score is a weighted vote over its "
            "`k` nearest labelled frames: the example frames plus the drawn background."
        ),
        json_schema_extra={"label": "Neighbours (k)", "advanced": True, "ui_max": 500},
    )
    tau: float = Field(
        0.07,
        gt=0.0,
        le=1.0,
        description=(
            "Vote temperature. Lower weights the closest neighbours more sharply, making the "
            "score more decisive and more sensitive to a single close match; higher averages "
            "over the `k` more evenly."
        ),
        json_schema_extra={"label": "Temperature", "advanced": True, "ui_step": 0.01},
    )
