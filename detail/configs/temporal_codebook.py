"""The frame-codebook temporal method's own parameters (§22) — the method's half of the config.

Split from ``configs/temporal_clustering.py`` the way ``configs/temporal_dtw_finch.py`` is, and for
the same reason: how a trajectory becomes labeled stretches is the method's business.

Where the sibling method's knobs are about *segmenting* — where a bout starts, how long is long
enough, how much warp to tolerate — these are about the **codebook** and the **decode**, because
this method never segments. It labels every frame and lets the runs fall out.

Defaults carry their evidence. Internal evaluations are behind the window and the run floor, and
behind `fit_share` and `codewords`, which replaced a single `fit_rows` total that did both jobs (#573).
"""

from pydantic import BaseModel, Field


class TemporalCodebookConfig(BaseModel):
    """What the codebook+decode method needs, beyond the trajectories the module hands it."""

    window_s: float = Field(
        0.53,
        ge=0.0,
        description="Sliding-window length (seconds) the trajectory is pooled over before the "
        "codebook is built, which smooths per-frame jitter. About 0.53 s is the tuned "
        "8-frame window at a 15 fps embed rate; the rows it resolves to depend on your cadence.",
        json_schema_extra={"label": "Window", "unit": "s"},
    )
    fit_share: float = Field(
        5.0,
        ge=0.5,
        le=100.0,
        description="Percent of each recording's frames the codebook is built from, evenly "
        "spaced. Every recording is sampled at the same density, so a behavior one animal shows "
        "briefly is as findable in a large study as in a small one; every frame is labelled "
        "afterwards whatever this is. It does not set how many codewords there are, so raising "
        "it costs the codebook fit, not the decode.",
        json_schema_extra={
            "label": "Share of each recording",
            "unit": "%",
            "ui_step": 0.5,
        },
    )
    codewords: int = Field(
        700,
        ge=2,
        le=5000,
        description="How many codewords the sample is grouped into: the finest setting the "
        "vocabulary offers, and the decode's states. Every pooled frame is decoded at a cost "
        "roughly the square of this, so it is the cost knob: halving it roughly quarters the "
        "decode. Lower it first if a fit is taking too long. A sample with fewer "
        "distinct rows than this uses one codeword per row.",
        json_schema_extra={"label": "Codewords", "ui_max": 5000, "ui_step": 50},
    )
    min_run_s: float = Field(
        0.27,
        ge=0.0,
        description="Runs shorter than this are dropped: they do not become instances, and "
        "their frames stay unassigned (they are not merged into a neighbour). A frame labeling "
        "flickers even after the decode, and without a floor every flicker becomes a one-frame "
        "behavior in the output. About 0.27 s is "
        "the tuned 4-frame minimum at a 15 fps embed rate.",
        json_schema_extra={"label": "Minimum run", "unit": "s"},
    )
    max_gap_s: float | None = Field(
        None,
        gt=0.0,
        description="Break the trajectory into continuous runs wherever tracking is missing for "
        "longer than this; the decode restarts at each. Omit to derive it from the embed cadence. "
        "A gap is an absence of evidence, not a transition between behaviors, and a decode that "
        "ran across one would invent a transition where the animal was simply not seen.",
        json_schema_extra={
            "advanced": "Decode",
            "label": "Max tracking gap",
            "unit": "s",
        },
    )
    seed: int = Field(
        0,
        ge=0,
        description="Seed for the k-means grouping. The same seed on the same recordings gives "
        "the same vocabulary; the decode is deterministic.",
        json_schema_extra={"label": "Seed", "advanced": True},
    )
