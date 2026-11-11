"""Single source for tracker hyperparameter labels, help, and bounds.

Tracker knobs are not pydantic config fields — each tracker declares a
``HYPERPARAMS_SCHEMA`` of ``{name: (type, default)}``, which carries the type and
default and nothing else. This module supplies the rest for every knob any tracker
exposes, so the generated Tracking doc page (``configs/describe.py``), the GUI form
(``gui/modules/tracking.py``), and the field ``?`` all read one definition.

The bounds are here because the GUI used to guess them as
``min(0, default)..max(1, default)``, which is right for a confidence and wrong for
everything else: ``iou_weight`` defaults to 1.0, so the guess pinned its box at 1.0 and
the form could not express a value the tracker accepts (``validate_hyperparams`` checks
types, not ranges — the CLI and API take 2.0 happily).

Every name in any tracker's ``HYPERPARAMS_SCHEMA`` must have an entry here — the
tracker-params doc test asserts it, so a new knob without a definition fails CI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerParam:
    """What a tracker knob is, beyond its type and default.

    ``minimum`` / ``maximum`` bound the form's spin box; ``step`` is its arrow increment
    (defaults to 1 for a whole number, 0.05 for a fraction). ``help`` follows the user-doc
    voice: plain, one or two sentences, no em dashes.
    """

    label: str
    help: str
    minimum: float = 0.0
    maximum: float = 1.0
    step: float | None = None


TRACKER_HYPERPARAM_DOCS: dict[str, TrackerParam] = {
    "num_subjects": TrackerParam(
        "Number of subjects",
        "How many subjects (animals) to track. The anchor tracker currently "
        "supports a single subject.",
        minimum=1,
        maximum=16,
    ),
    "conf_threshold": TrackerParam(
        "Min detection confidence",
        "Lowest detector confidence a detection needs to be a subject candidate. "
        "Detections below it are ignored when forming tracks.",
    ),
    "secondary_conf": TrackerParam(
        "Secondary confidence",
        "Lowest confidence for a secondary (fragment) detection to attach to the "
        "subject. Must be at most the main detection confidence.",
    ),
    "iou_threshold": TrackerParam(
        "Fragment overlap",
        "Smallest box overlap (intersection over union) a lower-confidence detection "
        "needs with the subject to be merged into it as a fragment of the same animal.",
    ),
    "max_secondary": TrackerParam(
        "Max secondary detections",
        "Most lower-confidence secondary detections merged into the subject on a "
        "single frame.",
        maximum=100,
    ),
    "require_contiguous": TrackerParam(
        "Require contiguous mask",
        "Only accept a subject whose mask is a single connected region. On, a frame "
        "whose detections all have a split mask gets no subject at all, however "
        "confident the detector was. Usually leave this off.",
    ),
    "merge_iou": TrackerParam(
        "Merge box IoU",
        "Box-overlap pre-filter for merging fragments: a candidate whose box "
        "overlaps the subject less than this is not merged.",
    ),
    "merge_mask_iou": TrackerParam(
        "Merge mask IoU",
        "Mask-overlap floor for merging a fragment into the subject. It rejects a "
        "reflection whose box overlaps but whose mask does not. 0 disables the "
        "mask gate.",
    ),
    "max_merge": TrackerParam(
        "Max fragments merged",
        "Most detection fragments merged into one subject per frame. 0 keeps the "
        "single cleanest mask and merges nothing.",
        maximum=20,
    ),
    "iou_weight": TrackerParam(
        "Continuity weight",
        "How strongly a candidate's overlap with the subject's previous box is "
        "favored when scoring matches. Higher keeps the track on the same animal "
        "through crowding.",
        maximum=10.0,
        step=0.1,
    ),
    "max_missed": TrackerParam(
        "Max missed frames",
        "Frames the subject may be absent, with no matching detection, before the "
        "track ends and must be re-acquired.",
        maximum=100_000,
    ),
    "max_missed_frames": TrackerParam(
        "Max missed frames",
        "Frames a track survives without a matching detection before it ends.",
        maximum=100_000,
    ),
}


def tracker_param(name: str) -> TrackerParam:
    """The definition for a hyperparameter, or a title-cased stand-in for an unknown one."""
    return TRACKER_HYPERPARAM_DOCS.get(
        name, TrackerParam(name.replace("_", " ").capitalize(), "")
    )


def tracker_param_label(name: str) -> str:
    """The display label for a hyperparameter, falling back to a title-cased name."""
    return tracker_param(name).label
