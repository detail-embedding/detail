"""The four spatial-setup categories, as one table (§10).

A field of view / distance calibration / ROI / POI drawn on one recording commonly applies
to a whole rig's worth of them (issue #342 — otherwise one setting is pulled into
every other recording one at a time). Copying them the other way is one operation with four
subjects, and every surface needs the same four facts about each: what to call it, whether
a recording has one, whether a *target* can receive one, and how to hand it over.

**Why a table rather than four of everything.** The alternative — and the shape this
started as — is four `Experiment.propagate_*` methods, a four-row tuple in the GUI dialog,
and four CLI commands, which is three hand-maintained copies of one list. They are
complete on the day they are written; afterwards a fifth kind, or a change to what "already
has one" means, is three edits in three files and nothing fails when it is two.
`Experiment._JOB_RUN_KINDS` is the pattern this follows.

**Copying never replaces.** A target that already has a definition of the category is
refused, with the reason, *before* the copy is offered — not silently overwritten. That is
the whole safety model: the operation has no undo and touches many recordings at once, so
it is additive by construction and a correction is a deliberate delete on the recording
that needs it. `occupied()` is the half that makes it true, and it is asked once per target
so the dialog can grey the row out ahead of the click rather than report a failure after.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only, imports no storage
    from detail.api.recording import Recording


@dataclass(frozen=True)
class SpatialCategory:
    """One copyable piece of a recording's spatial setup."""

    key: str
    #: What a person calls it, in the dialog, the CLI's output and the skip report.
    label: str
    #: The `storage.spatial` kind, or `None` for the calibration (which has no polygons).
    kind: str | None
    #: Whether `recording` has one of these to copy *from*.
    has_data: Callable[["Recording"], bool]
    #: Why `recording` cannot *receive* one, or `None`. Never "it would be overwritten" —
    #: having one at all is the refusal.
    #:
    #: Only the **kind-grained** kinds answer this. A field of view and a calibration are
    #: refused whole, because a second one is not a richer setup: two arenas break the
    #: batch tracking picker, which intersects arena *ids* across recordings, and there is
    #: only one pixel scale. Regions and points are sets by design, so their collisions are
    #: judged per name in `Recording.receive_spatial` — refusing a whole set because one
    #: name clashes would kill "pull one more zone from a template".
    occupied: Callable[["Recording"], str | None]
    #: Why `source`'s cannot go to `target` whatever `target` already holds, or `None`.
    #: Only the calibration answers: it belongs to the rig it was measured on, and a
    #: video of another shape is another rig. Asked before the copy, like `occupied`, so
    #: a picker greys the recording out instead of reporting a refusal after the click.
    unfit: Callable[["Recording", "Recording"], str | None]
    #: Hand `source`'s over to `target`, optionally narrowed to `names`; returns
    #: `(added, flattened, refused)` — what landed, whose keyframe motion was dropped, and
    #: what could not travel at all. `names` is how a picker offers *definitions* rather
    #: than whole categories, so every tick is a specific thing that will or will not go.
    give: Callable[
        ["Recording", "Recording"],
        tuple[list[str], list[str], list[tuple[str, str]]],
    ]


def _fov_occupied(recording: "Recording") -> str | None:
    """One sentence, shared with the pull direction — see
    `Recording.field_of_view_refusal`. Naming the runs is the difference between "delete
    it" and "delete these two runs, then delete it", and the second is what the user
    actually has to do."""
    return recording.field_of_view_refusal()


def _give_kind(
    kind: str,
) -> Callable[..., tuple[list[str], list[str], list[tuple[str, str]]]]:
    def give(source: "Recording", target: "Recording", names: list[str] | None = None):
        return target.propagate_spatial_from(source, kind, names)

    return give


def _give_calibration(
    source: "Recording", target: "Recording", names: list[str] | None = None
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    # Nothing to flatten: a calibration is two points, a distance and the resolution it
    # was measured at, with no frame index anywhere. A target of another shape was
    # screened out by `unfit`; the write refuses it again for callers that skip that.
    target.import_calibration_from(source)
    return ["distance calibration"], [], []


SPATIAL_CATEGORIES: tuple[SpatialCategory, ...] = (
    SpatialCategory(
        key="fov",
        label="Field of view",
        kind="arena",
        has_data=lambda r: bool(r.arenas),
        occupied=_fov_occupied,
        unfit=lambda _s, _t: None,
        give=_give_kind("arena"),
    ),
    SpatialCategory(
        key="calibration",
        label="Distance calibration",
        kind=None,
        has_data=lambda r: r.distance_calibration is not None,
        occupied=lambda r: (
            "already has a distance calibration — delete the existing one first"
            if r.distance_calibration is not None
            else None
        ),
        unfit=lambda s, t: t.calibration_refusal_for(s.distance_calibration),
        give=_give_calibration,
    ),
    SpatialCategory(
        key="roi",
        label="Regions of interest",
        kind="roi",
        has_data=lambda r: bool(r.rois),
        occupied=lambda _r: None,  # judged per definition — see collision_grain
        unfit=lambda _s, _t: None,  # normalized polygons fit any frame
        give=_give_kind("roi"),
    ),
    SpatialCategory(
        key="point",
        label="Points of interest",
        kind="point",
        has_data=lambda r: bool(r.points),
        occupied=lambda _r: None,  # judged per definition — see collision_grain
        unfit=lambda _s, _t: None,  # normalized polygons fit any frame
        give=_give_kind("point"),
    ),
)

CATEGORIES_BY_KEY: dict[str, SpatialCategory] = {c.key: c for c in SPATIAL_CATEGORIES}


def category(key: str) -> SpatialCategory:
    """The category named ``key``.

    Raises:
        ValueError: unknown key.
    """
    try:
        return CATEGORIES_BY_KEY[key]
    except KeyError:
        raise ValueError(
            f"unknown spatial category {key!r}; expected one of "
            f"{sorted(CATEGORIES_BY_KEY)}"
        ) from None


@dataclass(frozen=True)
class PropagationResult:
    """What one category's copy did, per target and per source definition.

    Four lists rather than a count, because every one of them is something the user has to
    be told: which recordings took it, which refused and why, and — the two that are easy
    to forget — what the trip *changed* about the definitions themselves. A copy that
    silently flattened a moving polygon or left an ROI behind is a copy that did not do
    what the button said.
    """

    copied: list[str]
    skipped: list[tuple[str, str]]
    flattened: list[str]
    refused: list[tuple[str, str]]
