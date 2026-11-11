"""What a cluster *is*, as numbers — the quantitative half of naming one.

A cluster of bouts is interpretable from its clips only up to a point. Naming clusters by eye
from their clips is no better than eyeballing a UMAP of them; the information component of that
judgement had not been automated. These are that information component: a handful of numbers per member,
so a surface can say this cluster is slow and compact where the run as a whole is fast and long.

**A descriptor is a function over a set of frames, for a part.** That one sentence is the whole
design, and it is what makes the same code describe three different things:

    L0  frame      the raw signals — geometry, and the embedding
    L1  pose       a cluster of frames
    L2  bout       a run of frames
    L3  behavior   a cluster of bouts   (= a set of frames, via its bouts)

A pose cluster, a bout and a behavior cluster are all *sets of frames*, so only the grouping
differs. That is the building block from pose to behavior: not similar metrics at two levels, the
same ones.

**The part axis is the other one, and it is why this takes a `FrameSlice` rather than a file.**
Today every slice is ``part="body"`` — one detection, one mask, one centroid. When a keypoint
layer or a second detection per subject arrives, a part is *more slices*, not a wider schema, and
a descriptor written today keeps working unchanged. A descriptor declares how many parts it
needs: ``parts=1`` fans out over whatever parts exist (so ``speed`` becomes head speed and tail
speed for free), and a future ``head_tail_angle`` declares two.

**Not `dynamics.py`, and this is not a style choice.** That module ends its own docstring with
*"Nothing in the pipeline imports this module — it is a research surface, and keeping it that way
is what makes it safe to change."* The pipeline needs these formulas, so they live here, pure
(numpy only: no zarr, no paths, no Qt) and importable by anything. ``dynamics`` imports them, so
the research video and the interpretation panel cannot drift apart about what "mask speed" means
— ``analysis/geometry.py`` was split out of ``dynamics`` for the same reason.

**NaN is missing data, never zero.** A step across a tracking gap is undefined: consecutive rows
are not consecutive in time, so the change between them describes nothing. Every aggregate here
ignores NaN rather than filling it, and a descriptor with nothing left to average returns NaN —
which a surface renders as absent. Filling would invent behaviour, quietly, in the direction of
"nothing happened".
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Rows are adjacent in time iff their ``frame_idx`` differ by exactly this.
DEFAULT_STRIDE = 1


@dataclass
class FrameSlice:
    """The signals for one unit's frames — a bout, a cluster's members, a whole recording.

    Pixel units for geometry, matching ``dynamics.DynamicsInputs``: a normalized speed is
    anisotropic (x and y scale differently on a non-square frame) and incomparable to a
    mask-pixel one, so a loader rescales by ``(width, height)`` before building this.

    Every array is indexed the same way — row ``i`` is one embedded frame — and any of the
    optional ones may be ``None`` when its source is unavailable. A descriptor that needs a
    missing one returns ``NaN`` rather than raising: a run embedded before the kinematics
    sidecar existed has no mask geometry, and that is a reason to show fewer numbers, not to
    fail opening the run.
    """

    frame_idx: np.ndarray
    t_ns: np.ndarray
    stride: int = DEFAULT_STRIDE
    part: str = "body"
    embeddings: np.ndarray | None = None
    centroid: np.ndarray | None = None  # (N, 2) px
    area: np.ndarray | None = None  # (N,) px
    bbox: np.ndarray | None = None  # (N, 4) xyxy px
    #: (N,) the *posture* holding at each frame, from a pose-clustering run over the same
    #: subject — ``-1`` where that run labelled nothing. The bridge from pose to behavior: a
    #: bout is a sequence of postures, and what it is made of is more interpretable than any
    #: scalar about it. Absent unless a pose run covers this unit, which is ordinary.
    labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frame_idx)

    @property
    def position(self) -> np.ndarray | None:
        """Where the subject is, from the best source this slice has.

        The mask's centre of mass when there is one, the box's geometric centre otherwise.
        Interchangeable **for this purpose and not for others**: how far something travelled and
        what shape its path took are the same question answered from either, differing by
        sub-pixel amounts on a rigid subject. Its *speed* is not — mass-centre motion carries
        deformation that box motion does not, which is why ``mask_speed`` and ``bbox_speed``
        stay two descriptors here as they are two metrics in ``dynamics``.

        So a run with no mask geometry still answers "where did it go", and gains the two
        mass-centre descriptors later without any of this changing.
        """
        if self.centroid is not None:
            return np.asarray(self.centroid, dtype=np.float64)
        if self.bbox is not None:
            return bbox_center(self.bbox)
        return None


@dataclass(frozen=True)
class Descriptor:
    """One named quantity, and what it needs to be computed."""

    name: str
    label: str
    unit: str
    fn: Callable[[FrameSlice], float]
    parts: int = 1
    #: Which grouping levels it means anything at. Duration means nothing for a single frame,
    #: so a pose surface must not offer it — and the surface asks rather than hardcoding a list.
    levels: tuple[str, ...] = ("bout", "cluster")
    #: Which optional `FrameSlice` fields it reads, so a surface can say *why* a number is
    #: absent instead of showing a blank cell.
    needs: tuple[str, ...] = ()
    #: Whether it measures the **span itself** rather than what happened inside it. Such a
    #: descriptor cannot be compared against spans the tool chose: a background window is as long
    #: as we made it, so "above 100% of them" is a fact about the sampler, not the animal.
    measures_span: bool = False


DESCRIPTORS: dict[str, Descriptor] = {}


def register_descriptor(
    name: str,
    *,
    label: str,
    unit: str,
    parts: int = 1,
    levels: tuple[str, ...] = ("bout", "cluster"),
    needs: tuple[str, ...] = (),
    measures_span: bool = False,
) -> Callable:
    """Declare a descriptor. The registry is the list every surface renders from.

    Hand-listing descriptors in a panel is the defect this avoids — the shape CLAUDE.md records
    as "a shared surface summarizing a module-shaped fact by hardcoding one module's shape".
    """

    def deco(fn: Callable[[FrameSlice], float]) -> Callable[[FrameSlice], float]:
        if name in DESCRIPTORS:
            raise ValueError(f"descriptor {name!r} is already registered")
        DESCRIPTORS[name] = Descriptor(
            name=name,
            label=label,
            unit=unit,
            fn=fn,
            parts=parts,
            levels=levels,
            needs=needs,
            measures_span=measures_span,
        )
        return fn

    return deco


# --- shared shape helpers, imported by `dynamics` so the two cannot drift -------------- #
def adjacent(frame_idx: np.ndarray, stride: int) -> np.ndarray:
    """(N-1,) bool: is row i+1 the next embedded frame after row i (no tracking gap)?"""
    return np.diff(np.asarray(frame_idx, dtype=np.int64)) == stride


def step_curve(values: np.ndarray, frame_idx: np.ndarray, stride: int) -> np.ndarray:
    """Per-row change of ``values`` ((N,) or (N, k)), NaN at row 0 and across gaps.

    The step at row ``i`` is the change *into* ``i`` (from ``i-1``), so it aligns with the frame
    where the change is visible. Row 0 has no predecessor.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    delta = values[1:] - values[:-1]
    step = np.abs(delta) if delta.ndim == 1 else np.linalg.norm(delta, axis=1)
    out[1:] = np.where(adjacent(frame_idx, stride), step, np.nan)
    return out


def cosine_step(x: np.ndarray, frame_idx: np.ndarray, stride: int) -> np.ndarray:
    """(N,) ``1 - cos(x_i-1, x_i)``, NaN at row 0 and across gaps."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    # Epsilon-normalized rather than guarded on a positive norm, matching the implementation
    # this was lifted from verbatim: a move that changes a number is not a move. A zero-length
    # embedding therefore reads as maximally changed rather than undefined, which is the
    # behaviour the segmentation video has always had.
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    cos = np.sum(xn[:-1] * xn[1:], axis=1)
    out[1:] = np.where(adjacent(frame_idx, stride), 1.0 - cos, np.nan)
    return out


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """(N, 2) geometric centre of each box."""
    bbox = np.asarray(bbox, dtype=np.float64)
    return np.stack(
        [0.5 * (bbox[:, 0] + bbox[:, 2]), 0.5 * (bbox[:, 1] + bbox[:, 3])], axis=1
    )


def _mean(values) -> float:
    """The mean of what is defined, or NaN when nothing is.

    ``np.nanmean`` of an all-NaN slice warns and returns NaN; under ``filterwarnings = error``
    that warning is a test failure, and an all-NaN slice is ordinary here — a one-frame bout has
    no steps at all.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = values[~np.isnan(values)]
    return float(defined.mean()) if len(defined) else float("nan")


def _peak(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    defined = values[~np.isnan(values)]
    return float(defined.max()) if len(defined) else float("nan")


# --- locomotion ------------------------------------------------------------------------ #
@register_descriptor("mask_speed", label="speed", unit="px/frame", needs=("centroid",))
def _mask_speed(s: FrameSlice) -> float:
    """How fast the silhouette's centre of mass moves, averaged over the unit."""
    if s.centroid is None:
        return float("nan")
    return _mean(step_curve(s.centroid, s.frame_idx, s.stride))


@register_descriptor(
    "mask_speed_peak", label="peak speed", unit="px/frame", needs=("centroid",)
)
def _mask_speed_peak(s: FrameSlice) -> float:
    """The fastest single step. A bout with one lunge in it reads as slow on the mean."""
    if s.centroid is None:
        return float("nan")
    return _peak(step_curve(s.centroid, s.frame_idx, s.stride))


@register_descriptor(
    "net_displacement", label="net displacement", unit="px", needs=("position",)
)
def _net_displacement(s: FrameSlice) -> float:
    """Straight-line distance from where the unit started to where it ended."""
    points = s.position
    if points is None or len(s) < 2:
        return float("nan")
    finite = points[~np.isnan(points).any(axis=1)]
    if len(finite) < 2:
        return float("nan")
    return float(np.linalg.norm(finite[-1] - finite[0]))


@register_descriptor("tortuosity", label="tortuosity", unit="0–1", needs=("position",))
def _tortuosity(s: FrameSlice) -> float:
    """Net displacement over path length: 1 is a straight line, 0 is going nowhere.

    The one descriptor here that separates *travelling* from *circling* from *staying put*, and
    it needs no orientation reference to do it — a walk across the arena and a tight circle can
    have identical mean speeds and opposite tortuosity.
    """
    if s.position is None or len(s) < 2:
        return float("nan")
    steps = step_curve(s.position, s.frame_idx, s.stride)
    path = np.nansum(steps)
    net = _net_displacement(s)
    if not np.isfinite(net) or path <= 0:
        return float("nan")
    return float(min(1.0, net / path))


@register_descriptor("bbox_speed", label="box speed", unit="px/frame", needs=("bbox",))
def _bbox_speed(s: FrameSlice) -> float:
    """How fast the detection box's centre moves.

    Kept beside ``mask_speed`` rather than folded into it, exactly as ``dynamics`` keeps both:
    the box tracks where the animal *is*, the mass centre also moves when it changes shape, and
    the difference between them is a signal rather than noise. A run with no mask geometry
    answers this one and not the other.
    """
    if s.bbox is None:
        return float("nan")
    return _mean(step_curve(bbox_center(s.bbox), s.frame_idx, s.stride))


# --- shape ----------------------------------------------------------------------------- #
@register_descriptor("mask_area", label="area", unit="px", needs=("area",))
def _mask_area(s: FrameSlice) -> float:
    """Silhouette footprint — it shrinks as an animal rears up out of the floor plane."""
    if s.area is None:
        return float("nan")
    return _mean(s.area)


@register_descriptor(
    "mask_area_step", label="area change", unit="px/frame", needs=("area",)
)
def _mask_area_step(s: FrameSlice) -> float:
    """How fast the footprint is changing — rearing and stretching show here."""
    if s.area is None:
        return float("nan")
    return _mean(
        step_curve(np.asarray(s.area, dtype=np.float64), s.frame_idx, s.stride)
    )


@register_descriptor("bbox_aspect", label="aspect", unit="h/w", needs=("bbox",))
def _bbox_aspect(s: FrameSlice) -> float:
    """Box elongation — the posture axis speed cannot see, and where rears sit high."""
    if s.bbox is None:
        return float("nan")
    bbox = np.asarray(s.bbox, dtype=np.float64)
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    return _mean(np.where(w > 0, h / np.where(w > 0, w, 1.0), np.nan))


@register_descriptor(
    "centroid_divergence",
    label="mass offset",
    unit="px",
    needs=("centroid", "bbox"),
)
def _centroid_divergence(s: FrameSlice) -> float:
    """Distance between the mask's centre of mass and its box's geometric centre.

    A posture signal that needs no motion at all: the mass shifts while the box does not, which
    is what a hunched or reaching animal looks like from above.
    """
    if s.centroid is None or s.bbox is None:
        return float("nan")
    offset = np.asarray(s.centroid, dtype=np.float64) - bbox_center(s.bbox)
    return _mean(np.linalg.norm(offset, axis=1))


# --- change ---------------------------------------------------------------------------- #
@register_descriptor(
    "pose_step", label="appearance change", unit="1-cos", needs=("embeddings",)
)
def _pose_step(s: FrameSlice) -> float:
    """How fast the animal's *appearance* is changing, frame to frame.

    The most interpretable number here and the cheapest, because it is the signal the clustering
    itself runs on — so it cannot be irrelevant to why these members grouped. Read against
    ``mask_speed`` it separates deforming from translating: high change with near-zero speed is
    grooming or scratching; high both is walking. That contrast is what optical flow would
    otherwise provide, available without a decode pass and without an orientation reference.
    """
    if s.embeddings is None:
        return float("nan")
    return _mean(cosine_step(s.embeddings, s.frame_idx, s.stride))


# --- composition: what a bout is made of, in postures ----------------------------------- #
def _labelled(s: FrameSlice) -> np.ndarray:
    """The frames of this unit a pose run actually labelled."""
    if s.labels is None:
        return np.empty(0, dtype=np.int64)
    labels = np.asarray(s.labels, dtype=np.int64)
    return labels[labels >= 0]


@register_descriptor("n_postures", label="postures", unit="count", needs=("labels",))
def _n_postures(s: FrameSlice) -> float:
    """How many distinct postures the unit passes through.

    The first number of the pose→behavior bridge. A bout holding one posture throughout is a
    static thing — a freeze, a hold; a bout cycling several is a movement made of parts. Neither
    is legible from a mean of anything.
    """
    labelled = _labelled(s)
    return float(len(np.unique(labelled))) if len(labelled) else float("nan")


@register_descriptor(
    "dominant_posture_share", label="main posture", unit="0–1", needs=("labels",)
)
def _dominant_posture_share(s: FrameSlice) -> float:
    """The share of labelled frames spent in the single most common posture.

    Near 1 is a behavior that *is* one posture; near 1/n is one that moves through several
    evenly. Read beside ``n_postures`` it separates "holds a rear" from "alternates two".
    """
    labelled = _labelled(s)
    if not len(labelled):
        return float("nan")
    _values, counts = np.unique(labelled, return_counts=True)
    return float(counts.max() / len(labelled))


@register_descriptor(
    "posture_switch_rate", label="posture switches", unit="/s", needs=("labels",)
)
def _posture_switch_rate(s: FrameSlice) -> float:
    """How often the posture changes, per second.

    Grooming cycles a small set quickly; walking holds one for a stride. A rate rather than a
    count, so a long bout and a short one are comparable — the count alone would rank every long
    bout as busier than every short one.

    A **lower bound**: an unlabelled stretch may hide changes, so this counts the ones it can
    see. Reported per second of *labelled* time, since dividing by a span the pose run only
    partly covered would report a busy bout as calm in proportion to how much was missing.
    """
    if s.labels is None or len(s) < 2:
        return float("nan")
    labels = np.asarray(s.labels, dtype=np.int64)
    t = np.asarray(s.t_ns, dtype=np.float64)
    labelled = labels >= 0
    if np.count_nonzero(labelled) < 2:
        return float("nan")

    # Over the labelled frames **in sequence**, skipping the unlabelled ones between them —
    # and over the labelled *time*, not the bout's whole span.
    #
    # The first version required both frames of a pair to be labelled, so a switch across a gap
    # was not counted. That reported two postures and zero switches for a
    # bout whose labels ran `1111......0000000` — the posture plainly changed, across the gap.
    # Zero is the wrong answer there; one is a lower bound and a true one, since a gap may hide
    # further changes but cannot hide the fact that the ends differ.
    kept = labels[labelled]
    seconds = float((t[labelled][-1] - t[labelled][0]) / 1e9)
    if seconds <= 0:
        return float("nan")
    return float(int(np.count_nonzero(kept[:-1] != kept[1:])) / seconds)


# --- duration -------------------------------------------------------------------------- #
@register_descriptor(
    "duration_s",
    label="duration",
    unit="s",
    levels=("bout", "cluster"),
    measures_span=True,
)
def _duration_s(s: FrameSlice) -> float:
    """How long the unit lasted. Meaningless for a single frame, which is why `levels` says so."""
    if len(s) < 2:
        return float("nan")
    t = np.asarray(s.t_ns, dtype=np.float64)
    return float((t[-1] - t[0]) / 1e9)


# --- the surface's entry point --------------------------------------------------------- #
def describe(slice_: FrameSlice, *, level: str = "bout", names=None) -> dict:
    """``{name: value}`` for every descriptor defined at ``level``.

    ``names`` narrows the set; ``None`` means all of them. A descriptor whose inputs are missing
    returns ``NaN`` rather than being omitted, so a caller sees the same keys whatever the run
    can answer — a missing key and an undefined value read very differently in a table.
    """
    wanted = DESCRIPTORS if names is None else {n: DESCRIPTORS[n] for n in names}
    return {
        name: float(d.fn(slice_)) for name, d in wanted.items() if level in d.levels
    }


def available(slice_: FrameSlice, *, level: str = "bout") -> list[str]:
    """The descriptors this slice can actually answer, so a surface shows only those."""
    return [
        name
        for name, d in DESCRIPTORS.items()
        if level in d.levels
        and all(getattr(slice_, need, None) is not None for need in d.needs)
    ]
