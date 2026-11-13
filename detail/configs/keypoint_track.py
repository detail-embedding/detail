"""Keypoint track config models (§26 "Track methods").

``KeypointTrackConfig`` is the stage's: the detect run to read, the tracking run whose subjects
its poses are given to and the two match settings (§26 *Assignment* — policy defaults, not
measured), and the method. Each **method**
has its own model — ``TemporalDecodeConfig``, ``DespikeSmoothConfig`` — and **every one of its
fields is identity**, filter constants included: a run is "the same settings" only if every
number that moved a point matches. The method models are flat (``despike_window_s`` rather than
a nested ``despike:`` group) so each knob is one documented field with one ``?``; ``track.yaml``
records the same flat dict under ``config:``.

Every window is a duration in seconds, resolved to rows at the recording's cadence: a frame
count is a different duration at every frame rate (``tests/configs/test_unit_conventions.py``).
The transition and penalty defaults are **policy**, not measured (§26 *Open questions* 6).
"""

from typing import Literal

from pydantic import BaseModel, Field

#: The method a detect run gets when nobody picks one is the *detect* method's own
#: ``default_track_method`` (dinov3_match: temporal_decode; the student: despike_smooth), so
#: there is no stage-level default here.
METHOD_IDS = ("temporal_decode", "despike_smooth")

#: Overrides how many recordings a local keypoint track run works on at once.
PARALLELISM_ENV = "DETAIL_MAX_KEYPOINT_TRACK_PARALLELISM"


class _Filters(BaseModel):
    """The two switchable filters both methods end with (§26 "Track methods"), and which of
    the skeleton's points the method may move at all."""

    track_points: list[str] | None = Field(
        None,
        description=(
            "The skeleton points this run settles over time; every other point keeps the "
            "detect run's own placement, untouched by the path and the filters. Unset: every "
            "point. Any of the detect run's point names, in any combination; a left/right "
            "pair with only one of its points here is not swapped, since settling left from "
            "right needs both."
        ),
        json_schema_extra={"label": "Points to track", "ui": "skeleton_points"},
    )

    despike: bool = Field(
        True,
        description=(
            "Replace a point that jumps away from its neighbours' median position by that "
            "median (a Hampel filter per point across rows). Steady motion moves the median "
            "with it and is left alone; a jump shorter than half the window is removed."
        ),
        json_schema_extra={"label": "Despike"},
    )
    despike_window_s: float = Field(
        0.1,
        gt=0,
        le=5.0,
        description=(
            "How far either side of a row the despike's median reaches, in seconds; resolved "
            "to rows at the recording's cadence."
        ),
        json_schema_extra={"label": "Despike window (s)", "unit": "s"},
    )
    despike_k: float = Field(
        3.0,
        gt=0,
        le=20.0,
        description=(
            "How many scaled median deviations from the window's median a point may sit "
            "before it counts as a spike."
        ),
        json_schema_extra={"label": "Despike threshold (MADs)"},
    )
    despike_floor: float = Field(
        0.03,
        ge=0,
        le=1.0,
        description=(
            "The deviation's lower bound, as a fraction of body scale, so a still animal's "
            "sub-pixel jitter is never read as a spike."
        ),
        json_schema_extra={"label": "Despike floor (body fraction)"},
    )
    smooth: bool = Field(
        True,
        description=(
            "Replace each point by a local quadratic fitted to its neighbours in time, "
            "Gaussian-weighted. It removes frame-to-frame jitter and keeps turning points a "
            "plain average would flatten; an absent point stays absent."
        ),
        json_schema_extra={"label": "Smooth"},
    )
    smooth_window_s: float = Field(
        0.133,
        gt=0,
        le=5.0,
        description=(
            "The smoothing weights' standard deviation, in seconds, measured in time so a "
            "gap or an every-n run is weighted by real distance."
        ),
        json_schema_extra={"label": "Smooth window (s)", "unit": "s"},
    )
    smooth_kernel: Literal["local_quadratic"] = Field(
        "local_quadratic",
        description="The local fit: a quadratic, the only kernel offered.",
        json_schema_extra={"label": "Smooth kernel"},
    )
    smooth_truncate: float = Field(
        3.0,
        gt=0,
        le=10.0,
        description="How many smooth windows either side the weights reach.",
        json_schema_extra={"label": "Smooth reach (windows)"},
    )
    smooth_min_points: int = Field(
        5,
        ge=3,
        le=50,
        description=(
            "How many weighted neighbours (the point itself included) a quadratic needs; "
            "with fewer, the point takes the weighted mean."
        ),
        json_schema_extra={"label": "Smooth minimum neighbours"},
    )


class TemporalDecodeConfig(_Filters):
    """``temporal_decode``: a path through the detect run's candidates, then the filters."""

    transition_sigma_body: float = Field(
        0.15,
        gt=0,
        le=5.0,
        description=(
            "Expected per-row move of a point, as a fraction of the subject's body scale; the "
            "transition cost is a Gaussian on the move at that scale, widened by the square "
            "root of the time elapsed, so a gap costs what its duration earns."
        ),
        json_schema_extra={"label": "Transition scale (body fraction)"},
    )
    missing_penalty: float = Field(
        2.0,
        ge=0,
        le=100.0,
        description=(
            "What the path pays, in score units, to call a point missing on a row rather than "
            "take a candidate; higher keeps weak candidates, lower drops them."
        ),
        json_schema_extra={"label": "Missing penalty"},
    )
    mirror_weight: float = Field(
        1.0,
        ge=0,
        le=100.0,
        description=(
            "What a left/right pair pays to switch between its normal and swapped reading; "
            "the mirrored-minus-unmirrored score the detect run stored is the evidence, so a "
            "swap must hold over frames to win and never flickers."
        ),
        json_schema_extra={"label": "Mirror switch cost"},
    )
    decode_cutoff: float | None = Field(
        None,
        description=(
            "The score below which a candidate is not offered to the path. Omit for the "
            "method's default cut-off (none for dinov3_match). This is the decode cut-off and "
            "part of the run's identity; what the viewer shows is a separate display cut-off."
        ),
        json_schema_extra={"label": "Decode cut-off"},
    )


class DespikeSmoothConfig(_Filters):
    """``despike_smooth``: the filters alone, on the detect run's best point per row."""


class KeypointTrackConfig(BaseModel):
    detect_run_id: str = Field(
        ...,
        description=(
            "Keypoint detect run to settle over time (e.g. kd_01); its candidates, scores and "
            "skeleton are what the method reads."
        ),
        json_schema_extra={"label": "Keypoint detect run"},
    )
    tracking_run_id: str | None = Field(
        None,
        description=(
            "Tracking run whose subjects the poses are given to (e.g. track_01); it must be "
            "on the detect run's detection run. Omit when exactly one such run is complete; "
            "with several, name one."
        ),
        json_schema_extra={"label": "Tracking run"},
    )
    match_min_inside: float = Field(
        0.5,
        gt=0,
        le=1,
        description=(
            "The least share of a pose's placed points that must fall inside a subject's "
            "(grown) mask for the pose to be given to that subject. A policy default, not "
            "a measured one."
        ),
        json_schema_extra={"label": "Match: share inside mask", "advanced": True},
    )
    match_mask_grow: float = Field(
        0.05,
        ge=0,
        description=(
            "How far each subject's mask is grown before the match, as a fraction of its "
            "body scale, so a nose or tail tip the segmentation clipped still counts as "
            "inside. A policy default, not a measured one."
        ),
        json_schema_extra={
            "label": "Match: mask grow (body fraction)",
            "advanced": True,
        },
    )
    method: str | None = Field(
        None,
        description=(
            "The track method: temporal_decode (a path through the stored candidates, then "
            "the filters) or despike_smooth (the filters on the best point). Omit for the "
            "detect run's method's own default."
        ),
        json_schema_extra={"label": "Method"},
    )


METHOD_CONFIGS: dict[str, type[BaseModel]] = {
    "temporal_decode": TemporalDecodeConfig,
    "despike_smooth": DespikeSmoothConfig,
}

#: The settings *Tune tracking* tries for ``temporal_decode`` (§26 *Tuning a track method*):
#: every combination of these values, the rest of the config held at what the user has set.
#: They are the knobs a study of track settings on hand-labelled hot-plate video found to move
#: accuracy, at values spanning its range: the transition scale (0.15, the default, is far too
#: small for 25 fps oblique video), smoothing off / short / default, and despike. The missing penalty and the
#: mirror cost moved nothing there and are left where the user put them. Each combination is
#: validated against ``TemporalDecodeConfig`` by ``tests/configs/test_keypoint_track.py``.
TUNE_GRID: dict[str, tuple] = {
    "transition_sigma_body": (0.15, 0.5, 1.0, 2.0, 3.0, 5.0),
    "smooth_window_s": (None, 0.05, 0.133),  # None: smoothing off
    "despike": (False, True),
}
TUNE_METHOD = "temporal_decode"


class KeypointTrackPreviewWindow(BaseModel):
    """*Preview tracking*'s stretch of one recording (§26 *Previewing a track method*)."""

    start_s: float | None = Field(
        None,
        ge=0,
        description=(
            "Where the preview starts, in seconds from the start of the recording. Omit for "
            "the middle of the recording. A window that would cross into the next video "
            "part stops at the end of the part it starts in."
        ),
        json_schema_extra={"label": "Start (s)", "unit": "s"},
    )
    duration_s: float = Field(
        30.0,
        gt=0,
        le=120.0,
        description=(
            "How long each preview is, in seconds. Tracking runs over these seconds alone "
            "(and a short margin either side, so the path settles as it would in a full "
            "run), which is what keeps a preview to seconds rather than minutes."
        ),
        json_schema_extra={"label": "Duration (s)", "unit": "s"},
    )


#: How many recordings a preview covers at most, and how many the window ticks by default.
PREVIEW_MAX_RECORDINGS = 3
PREVIEW_DEFAULT_RECORDINGS = 2
#: How many settings a preview tries beside the current ones and detection alone.
PREVIEW_MAX_VARIANTS = 2
#: Seconds decoded either side of the window and then dropped: the path is a whole-series
#: decode, so a window cut at its edges would settle differently there than a full run does.
PREVIEW_MARGIN_S = 2.0
#: The method settings a preview variant changes from the current ones: the knobs found to
#: move accuracy on hand-labelled video (``TUNE_GRID``'s), each read off
#: the method's own model; a method without one (despike_smooth has no transition) skips it.
PREVIEW_VARIANT_FIELDS = (
    "transition_sigma_body",
    "smooth",
    "smooth_window_s",
    "despike",
)
#: Where the two variants start, per method, before the user edits them: for temporal_decode
#: a wider transition, and the best of that study's 28 settings (transition 3, filters off).
#: Starting points to look at, not recommendations.
PREVIEW_STARTING_VARIANTS: dict[str, tuple[dict, ...]] = {
    "temporal_decode": (
        {"transition_sigma_body": 1.0},
        {"transition_sigma_body": 3.0, "smooth": False, "despike": False},
    ),
    "despike_smooth": ({"smooth": False}, {"smooth": False, "despike": False}),
}


def tune_settings() -> list[dict]:
    """Every ``TUNE_GRID`` combination as config overrides, in grid order: smoothing off is
    ``smooth: False`` (its window left as set), on is ``smooth: True`` plus the window.
    """
    out = []
    for sigma in TUNE_GRID["transition_sigma_body"]:
        for window in TUNE_GRID["smooth_window_s"]:
            for despike in TUNE_GRID["despike"]:
                setting = {"transition_sigma_body": sigma, "despike": despike}
                if window is None:
                    setting["smooth"] = False
                else:
                    setting.update(smooth=True, smooth_window_s=window)
                out.append(setting)
    return out
