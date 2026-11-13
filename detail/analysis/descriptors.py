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
    #: The mask's outline per row (``analysis/mask_shape``), in pixels / degrees / ratios:
    #: ``top``, ``bottom``, ``floor_width``, ``major``, ``minor``, ``orientation``,
    #: ``solidity``, ``circularity``, ``protrusions``, ``protrusion_depth``. Absent for a run whose
    #: sidecar predates the outline columns (``recording backfill-kinematics`` adds them).
    shape: dict | None = None
    #: The animal's typical length in pixels over its **whole** track — one number, so a bout's
    #: heights and widths are in body lengths and a big animal and a small one compare. ``None``
    #: when nothing could measure it.
    body_length: float | None = None
    #: The recording's resolved camera view (§10): ``top_down``, ``side``, ``angled_down``,
    #: ``other``, or ``None`` when not set. A descriptor that only means something from one view
    #: declares it (``Descriptor.views``) and is out of scope elsewhere.
    camera_view: str | None = None
    #: The keypoints per row, by skeleton name: ``{name: (N, 2) px}``, NaN where the point was
    #: not placed or scored under the run's cut-off. ``None`` with no keypoint track run.
    points: dict | None = None
    #: The keypoint track run the points came from (``kt_NN``), so a surface can say so.
    points_run: str | None = None
    #: Movement inside the mask per row (``pipeline/motion_backfill``): ``inside``, ``ends`` and
    #: ``middle``, each a mean change in brightness (0-1) since the previous video frame. Absent
    #: until ``recording backfill-motion`` has run for the embedding run.
    motion: dict | None = None

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
    #: The camera views it means anything from, or ``None`` for any view. How high the mask
    #: reaches is a fact about the animal from the side and about perspective from above.
    views: tuple[str, ...] | None = None
    #: Whether it says *where* the animal is or *which way it faces* rather than what its body
    #: does. Rearing toward the left and toward the back are one behaviour to most people, so a
    #: cluster profile can leave these out (``ClusterProfileConfig.leave_out_place``).
    place: bool = False


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
    views: tuple[str, ...] | None = None,
    place: bool = False,
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
            views=views,
            place=place,
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
    "pose_step",
    label="appearance change",
    unit="1-cos",
    needs=("embeddings",),
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


@register_descriptor(
    "n_postures",
    label="postures",
    unit="count",
    needs=("labels",),
)
def _n_postures(s: FrameSlice) -> float:
    """How many distinct postures the unit passes through.

    The first number of the pose→behavior bridge. A bout holding one posture throughout is a
    static thing — a freeze, a hold; a bout cycling several is a movement made of parts. Neither
    is legible from a mean of anything.
    """
    labelled = _labelled(s)
    return float(len(np.unique(labelled))) if len(labelled) else float("nan")


@register_descriptor(
    "dominant_posture_share",
    label="main posture",
    unit="0–1",
    needs=("labels",),
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
    "posture_switch_rate",
    label="posture switches",
    unit="/s",
    needs=("labels",),
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


# --- the mask's outline (``analysis/mask_shape``, via the kinematics sidecar) ------------ #
#: Below this speed, in body lengths a second, a step counts as still.
STILL_BODY_LENGTHS_PER_S = 0.25
#: The rhythm measure needs this many seconds of bout and samples of at least this rate (Hz),
#: or a "rhythm" is just the sampling.
RHYTHM_MIN_S = 1.0
RHYTHM_MIN_RATE_HZ = 6.0
#: The band a repetitive movement (licking, scratching, grooming strokes) sits in, Hz.
RHYTHM_BAND_HZ = (2.0, 12.0)

SIDE = ("side",)


def _outline(s: FrameSlice, key: str) -> np.ndarray | None:
    if s.shape is None or key not in s.shape:
        return None
    return np.asarray(s.shape[key], dtype=np.float64)


def _per_body(values: np.ndarray | None, s: FrameSlice) -> np.ndarray | None:
    if values is None or not s.body_length or not np.isfinite(s.body_length):
        return None
    return values / float(s.body_length)


def _heights(s: FrameSlice) -> np.ndarray | None:
    top, bottom = _outline(s, "top"), _outline(s, "bottom")
    if top is None or bottom is None:
        return None
    return _per_body(bottom - top, s)


def _rates(s: FrameSlice, values: np.ndarray) -> np.ndarray:
    """``values`` (one per step) per second, NaN across a gap or a zero time step."""
    dt = np.diff(np.asarray(s.t_ns, dtype=np.float64)) / 1e9
    ok = adjacent(s.frame_idx, s.stride) & (dt > 0)
    return np.where(ok, values / np.where(dt > 0, dt, 1.0), np.nan)


@register_descriptor(
    "mask_height",
    label="mask height",
    unit="body lengths",
    needs=("shape", "body_length"),
    views=SIDE,
)
def _mask_height(s: FrameSlice) -> float:
    """How tall the mask stands, top to bottom, in body lengths, from a side camera. Rearing
    is the obvious reader: an animal on its hind legs is taller than it is long."""
    heights = _heights(s)
    return float("nan") if heights is None else _mean(heights)


@register_descriptor(
    "mask_height_peak",
    label="peak mask height",
    unit="body lengths",
    needs=("shape", "body_length"),
    views=SIDE,
)
def _mask_height_peak(s: FrameSlice) -> float:
    heights = _heights(s)
    return float("nan") if heights is None else _peak(heights)


@register_descriptor(
    "floor_contact",
    label="floor contact",
    unit="body lengths",
    needs=("shape", "body_length"),
    views=SIDE,
)
def _floor_contact(s: FrameSlice) -> float:
    """How wide the mask's bottom edge is where it meets the floor, from a side camera:
    narrow when upright on two feet, wide on all fours."""
    widths = _per_body(_outline(s, "floor_width"), s)
    return float("nan") if widths is None else _mean(widths)


@register_descriptor("elongation", label="elongation", unit="ratio", needs=("shape",))
def _elongation(s: FrameSlice) -> float:
    """Long axis over short axis: stretched out against hunched or curled."""
    major, minor = _outline(s, "major"), _outline(s, "minor")
    if major is None or minor is None:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        return _mean(np.where(minor > 0, major / minor, np.nan))


@register_descriptor("solidity", label="solidity", unit="0-1", needs=("shape",))
def _solidity(s: FrameSlice) -> float:
    """How filled-in the outline is: 1 for a convex blob, lower as limbs, a tail or gaps open
    notches in it."""
    values = _outline(s, "solidity")
    return float("nan") if values is None else _mean(values)


@register_descriptor("roundness", label="roundness", unit="0-1", needs=("shape",))
def _roundness(s: FrameSlice) -> float:
    """How close the outline is to a circle (4 pi area over perimeter squared): huddled or
    curled up."""
    values = _outline(s, "circularity")
    return float("nan") if values is None else _mean(values)


@register_descriptor("protrusions", label="protrusions", unit="count", needs=("shape",))
def _protrusions(s: FrameSlice) -> float:
    """How many parts stick out of the outline, deeper than a ripple of fur."""
    values = _outline(s, "protrusions")
    return float("nan") if values is None else _mean(values)


@register_descriptor(
    "axis_turning", label="body-axis turning", unit="deg/s", needs=("shape",)
)
def _axis_turning(s: FrameSlice) -> float:
    """How fast the mask's long axis turns. Without keypoints the axis has no head, so this is
    turning without direction: which end is which never counts as a turn."""
    angle = _outline(s, "orientation")
    if angle is None or len(angle) < 2:
        return float("nan")
    step = np.abs(np.diff(angle))
    return _mean(_rates(s, np.minimum(step, 180.0 - step)))


@register_descriptor(
    "stillness", label="stillness", unit="share", needs=("centroid", "body_length")
)
def _stillness(s: FrameSlice) -> float:
    """The share of the bout's steps where the body moves under a quarter body length a
    second: freezing and resting high, walking low."""
    if s.centroid is None or len(s) < 2 or not s.body_length:
        return float("nan")
    xy = np.asarray(s.centroid, dtype=np.float64)
    speed = _rates(s, np.hypot(*np.diff(xy, axis=0).T)) / float(s.body_length)
    finite = speed[np.isfinite(speed)]
    if not len(finite):
        return float("nan")
    return float(np.mean(finite < STILL_BODY_LENGTHS_PER_S))


@register_descriptor("rhythm", label="rhythm", unit="share", needs=("area",))
def _rhythm(s: FrameSlice) -> float:
    """How much of the mask area's change is one regular back-and-forth between 2 and 12 Hz
    (licking, scratching, grooming strokes): the share of the detrended signal's power at the
    strongest frequency of that band. NaN for a bout under a second, or one sampled too
    slowly to see the band (an embedding cadence that skips frames lowers the rate)."""
    if s.area is None or len(s) < 8:
        return float("nan")
    area = np.asarray(s.area, dtype=np.float64)
    t = np.asarray(s.t_ns, dtype=np.float64) / 1e9
    duration = float(t[-1] - t[0])
    if duration < RHYTHM_MIN_S or not np.isfinite(area).all():
        return float("nan")
    rate = (len(t) - 1) / duration
    if rate < RHYTHM_MIN_RATE_HZ:
        return float("nan")
    grid = np.linspace(t[0], t[-1], len(t))
    signal = np.interp(grid, t, area)
    signal = signal - np.polyval(np.polyfit(grid, signal, 1), grid)
    power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=duration / (len(signal) - 1))
    total = float(power[1:].sum())
    band = (freqs >= RHYTHM_BAND_HZ[0]) & (freqs <= min(RHYTHM_BAND_HZ[1], rate / 2))
    if total <= 0 or not band.any():
        return float("nan")
    return float(power[band].max() / total)


# --- three keypoints: nose, mid back, tailbase ---------------------------------------- #
#: The names a skeleton gives each of the three points these read, matched without case,
#: spaces, hyphens or underscores. A skeleton that names none of a role's spellings simply
#: does not offer the descriptors that need it.
POINT_ALIASES = {
    "nose": ("nose", "snout", "nosetip"),
    "mid_back": (
        "midback",
        "back",
        "spine",
        "spinemid",
        "midspine",
        "centre",
        "center",
    ),
    "tailbase": ("tailbase", "tailroot", "baseoftail", "tailstart"),
}


def _plain(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def point_named(points: dict | None, role: str) -> np.ndarray | None:
    """The ``(N, 2)`` px column of the point playing ``role`` (``nose``, ``mid_back``,
    ``tailbase``), or ``None`` when the skeleton has no point by any of its names."""
    if not points:
        return None
    spellings = POINT_ALIASES[role]
    for name, column in points.items():
        if _plain(name) in spellings:
            return np.asarray(column, dtype=np.float64)
    return None


def _angle_deg(v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]))


def _wrap(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


@register_descriptor(
    "nose_height",
    label="nose height",
    unit="body lengths",
    needs=("points", "body_length"),
    views=SIDE,
)
def _nose_height(s: FrameSlice) -> float:
    """How far the nose is above the tailbase, in body lengths, from a side camera: head up
    and rearing high, level walking near zero, head down below it."""
    nose, tail = point_named(s.points, "nose"), point_named(s.points, "tailbase")
    if nose is None or tail is None or not s.body_length:
        return float("nan")
    return _mean((tail[:, 1] - nose[:, 1]) / float(s.body_length))  # y grows downward


@register_descriptor("bend", label="bend", unit="deg", needs=("points",))
def _bend(s: FrameSlice) -> float:
    """How far the back is from straight at the mid back: the angle nose, mid back, tailbase,
    as degrees off 180. Stretched near 0, curled (grooming, turning) high."""
    nose = point_named(s.points, "nose")
    mid = point_named(s.points, "mid_back")
    tail = point_named(s.points, "tailbase")
    if nose is None or mid is None or tail is None:
        return float("nan")
    a, b = nose - mid, tail - mid
    cos = np.sum(a * b, axis=1) / (np.hypot(*a.T) * np.hypot(*b.T))
    with np.errstate(invalid="ignore"):
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return _mean(180.0 - angle)


@register_descriptor(
    "heading_turning", label="heading turning", unit="deg/s", needs=("points",)
)
def _heading_turning(s: FrameSlice) -> float:
    """How fast the animal turns, from tailbase toward nose: a true heading, so unlike the
    mask's axis it knows which way is forward."""
    nose, tail = point_named(s.points, "nose"), point_named(s.points, "tailbase")
    if nose is None or tail is None or len(s) < 2:
        return float("nan")
    heading = _angle_deg(nose - tail)
    return _mean(_rates(s, np.abs(_wrap(np.diff(heading)))))


@register_descriptor(
    "head_scanning",
    label="head scanning",
    unit="body lengths/s",
    needs=("points", "body_length"),
)
def _head_scanning(s: FrameSlice) -> float:
    """How much the nose moves *relative to the mid back*, in body lengths a second: sniffing
    and scanning with the body still reads high where walking, which moves both, does not.
    """
    nose, mid = point_named(s.points, "nose"), point_named(s.points, "mid_back")
    if nose is None or mid is None or len(s) < 2 or not s.body_length:
        return float("nan")
    relative = nose - mid
    return _mean(
        _rates(s, np.hypot(*np.diff(relative, axis=0).T)) / float(s.body_length)
    )


# --- where it is and which way it faces (``Descriptor.place``) ------------------------- #
def _facing(s: FrameSlice) -> np.ndarray | None:
    """The unit vector of the circular mean of the tailbase-to-nose direction in the image, or
    ``None`` when there is no such direction (no points, or the directions cancel)."""
    nose, tail = point_named(s.points, "nose"), point_named(s.points, "tailbase")
    if nose is None or tail is None:
        return None
    v = nose - tail
    length = np.hypot(*v.T)
    keep = np.isfinite(length) & (length > 0)
    if not keep.any():
        return None
    mean = (v[keep] / length[keep, None]).mean(axis=0)
    size = float(np.hypot(*mean))
    return mean / size if size > 1e-9 else None


@register_descriptor(
    "facing_right", label="facing right", unit="-1 to 1", needs=("points",), place=True
)
def _facing_right(s: FrameSlice) -> float:
    """Which way the animal faces across the image: 1 toward the right edge, -1 toward the
    left, 0 straight up or down it. With ``facing_down`` it is the facing direction as two
    numbers, because an angle's -180 and 180 are the same way and a classifier would not know.
    """
    facing = _facing(s)
    return float("nan") if facing is None else float(facing[0])


@register_descriptor(
    "facing_down", label="facing down", unit="-1 to 1", needs=("points",), place=True
)
def _facing_down(s: FrameSlice) -> float:
    """Which way the animal faces down the image: 1 toward the bottom edge, -1 toward the top.
    From the side that is mostly nose down or up; from above, toward or away from the camera's
    lower edge."""
    facing = _facing(s)
    return float("nan") if facing is None else float(facing[1])


def _where(s: FrameSlice, axis: int) -> float:
    position = s.position
    return float("nan") if position is None else _mean(position[:, axis])


@register_descriptor(
    "position_x", label="position across", unit="px", needs=("position",), place=True
)
def _position_x(s: FrameSlice) -> float:
    """Where the animal is across the image (its mean centre, in pixels from the left edge).
    Pixels, not the arena: two cameras that frame the chamber differently disagree."""
    return _where(s, 0)


@register_descriptor(
    "position_y", label="position down", unit="px", needs=("position",), place=True
)
def _position_y(s: FrameSlice) -> float:
    """Where the animal is down the image (its mean centre, in pixels from the top edge)."""
    return _where(s, 1)


# --- movement inside the mask (``pipeline/motion_backfill``) --------------------------- #
@register_descriptor(
    "inner_motion", label="motion inside the mask", unit="0-1", needs=("motion",)
)
def _inner_motion(s: FrameSlice) -> float:
    """How much the pixels inside the outline change from frame to frame, even while the
    outline holds still: grooming, licking and scratching read high where freezing reads
    near zero."""
    values = np.asarray(s.motion["inside"], dtype=np.float64) if s.motion else None
    return float("nan") if values is None else _mean(values)


@register_descriptor(
    "end_motion_share", label="motion at the ends", unit="share", needs=("motion",)
)
def _end_motion_share(s: FrameSlice) -> float:
    """How much of the inner motion is at the body's ends (outer thirds of its long axis)
    rather than its middle: face-washing and paw-licking toward 1, flank scratching toward 0.
    """
    if not s.motion:
        return float("nan")
    ends = _mean(np.asarray(s.motion["ends"], dtype=np.float64))
    middle = _mean(np.asarray(s.motion["middle"], dtype=np.float64))
    total = ends + middle
    return float(ends / total) if np.isfinite(total) and total > 0 else float("nan")


# --- the surface's entry point --------------------------------------------------------- #
def in_scope(d: Descriptor, slice_: FrameSlice) -> bool:
    """Whether ``d`` means anything for this slice, from this camera view. A view-gated
    descriptor is out of scope where the view is not set.
    """
    return d.views is None or slice_.camera_view in d.views


def describe(slice_: FrameSlice, *, level: str = "bout", names=None) -> dict:
    """``{name: value}`` for every descriptor defined at ``level``.

    ``names`` narrows the set; ``None`` means every one that is meaningful for this slice. A
    descriptor whose inputs are missing returns ``NaN`` rather than
    being omitted, so a caller sees the same keys whatever the run can answer — a missing key and
    an undefined value read very differently in a table.
    """
    wanted = (
        {n: d for n, d in DESCRIPTORS.items() if in_scope(d, slice_)}
        if names is None
        else {n: DESCRIPTORS[n] for n in names}
    )
    return {
        name: float(d.fn(slice_)) for name, d in wanted.items() if level in d.levels
    }


def available(slice_: FrameSlice, *, level: str = "bout") -> list[str]:
    """The descriptors this slice can actually answer, so a surface shows only those."""
    return [
        name
        for name, d in DESCRIPTORS.items()
        if level in d.levels
        and in_scope(d, slice_)
        and all(getattr(slice_, need, None) is not None for need in d.needs)
    ]
