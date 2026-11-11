"""The heading a keypoint track run gives each frame, and what an aligned embedding does
with it (§14 "Egocentric alignment").

A **heading definition** is two sets of skeleton point names, ``front`` and ``back``. On a
row the heading is ``atan2(dy, dx)`` in **pixels** (normalized x and y are not the same
length on a non-square frame; y down, 0° = facing +x) from the mean of the back set to the
mean of the front set, over the points that are present and at or above the track run's
decode cut-off. A set rather than a pair, so one missing point moves the mean a little
instead of losing the heading; a single pair is the one-point case.

Pure numpy, and the only place the heading is computed: the embedding worker, the canvas
pre-pass, the launch form's coverage count and every preview of an aligned run call these
functions, so what a preview shows is what the run embeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALIGN_MODES = ("none", "rotate", "mirror")

#: ``heading_source`` values on the embedding part (§14): the row's own heading, one
#: interpolated across a gap, and (never written, used in counts) no heading at all.
SOURCE_OWN = 0
SOURCE_INTERPOLATED = 1
SOURCE_NONE = -1


def validate_definition(definition: dict, names: tuple[str, ...] | None = None) -> dict:
    """``{front: [...], back: [...]}``, each a non-empty list of distinct names, the two
    disjoint, and (when ``names`` is given) every name in the skeleton.

    Returns the definition with the two lists as lists of ``str``.

    Raises:
        ValueError: naming what is wrong — and the missing points, when a skeleton is given.
    """
    if not isinstance(definition, dict):
        raise ValueError("a heading definition is a mapping with 'front' and 'back'")
    out: dict = {}
    for side in ("front", "back"):
        value = definition.get(side)
        if isinstance(value, str):
            value = [value]
        if not value:
            raise ValueError(f"a heading definition needs at least one '{side}' point")
        points = [str(v) for v in value]
        if len(set(points)) != len(points):
            raise ValueError(f"'{side}' lists a point twice: {points}")
        out[side] = points
    shared = sorted(set(out["front"]) & set(out["back"]))
    if shared:
        raise ValueError(f"a point cannot be both front and back: {shared}")
    if names is not None:
        missing = [p for p in out["front"] + out["back"] if p not in names]
        if missing:
            raise ValueError(
                f"the heading definition names points this skeleton lacks: {missing}; "
                f"it has {list(names)}"
            )
    return out


@dataclass(frozen=True)
class HeadingSeries:
    """One track's heading on the rows where it was defined, in time order."""

    t_ns: np.ndarray  # (N,) int64 ascending, the defined rows
    deg: np.ndarray  # (N,) float64, image degrees in (-180, 180]
    n_rows: int  # rows the track had (defined or not)
    typical_length_px: float  # the 95th percentile axis length, the facing-camera scale

    @property
    def n_defined(self) -> int:
        return int(self.t_ns.size)


@dataclass(frozen=True)
class BodyAxis:
    """Where a skeleton's body axis is, as columns of its points: the ``front`` and ``back``
    sets of a heading definition, the ``middle`` points between them where a bend shows
    (:func:`midline_columns`), the frame size the normalized points are on, the track run's
    decode cut-off, and the track's typical axis length in pixels — what a body length is
    relative to, so 1.0 is the animal's usual length whatever the camera's distance.

    Plain data, so the functions over it stay pure: a reader fills one from a run's heading
    definition and skeleton, and the clip window's curves and pictures take it as given.
    """

    front: tuple[int, ...]
    back: tuple[int, ...]
    middle: tuple[int, ...]
    frame_wh: tuple[int, int]
    cutoff: float | None
    typical_length_px: float


def row_headings(
    points: np.ndarray,
    scores: np.ndarray,
    front: list[int],
    back: list[int],
    frame_wh: tuple[int, int],
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the heading in image degrees and the axis length in pixels, NaN where one
    set has no usable point (present and at or above ``cutoff``)."""
    front_xy, back_xy = row_axis_ends(points, scores, front, back, frame_wh, cutoff)
    dx = front_xy[:, 0] - back_xy[:, 0]
    dy = front_xy[:, 1] - back_xy[:, 1]
    length = np.hypot(dx, dy)
    deg = np.degrees(np.arctan2(dy, dx))
    deg[~np.isfinite(length)] = np.nan
    return deg, length


def _usable_px(
    points: np.ndarray,
    scores: np.ndarray,
    frame_wh: tuple[int, int],
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(points in pixels (M, K, 2), usable (M, K))``: present, and at or above
    ``cutoff``."""
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    usable = np.isfinite(points).all(axis=2)
    if cutoff is not None:
        usable &= np.nan_to_num(scores, nan=-np.inf) >= float(cutoff)
    w, h = (float(frame_wh[0]), float(frame_wh[1]))
    return points * np.array([w, h]), usable


def row_axis_ends(
    points: np.ndarray,
    scores: np.ndarray,
    front: list[int],
    back: list[int],
    frame_wh: tuple[int, int],
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the mean of the usable ``front`` points and of the usable ``back`` points, in
    pixels ``(M, 2)`` each, NaN where a set has none. The two ends of the body axis that
    :func:`row_headings` measures."""
    px, usable = _usable_px(points, scores, frame_wh, cutoff)

    def _mean(columns: list[int]) -> np.ndarray:
        sub = px[:, columns, :]  # (M, S, 2)
        ok = usable[:, columns]  # (M, S)
        count = ok.sum(axis=1)
        total = np.where(ok[..., None], sub, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count[:, None]
        mean[count == 0] = np.nan
        return mean

    return _mean(front), _mean(back)


def midline_columns(skeleton, definition: dict) -> list[int]:
    """The skeleton's points on the body's midline between the axis ends: every point with no
    left/right partner that is in neither set of the (validated) ``definition``. These are
    where a bend shows; ears and paws, which come in pairs, are not."""
    paired = {name for pair in skeleton.pairs for name in pair}
    ends = set(definition["front"]) | set(definition["back"])
    return [
        i
        for i, name in enumerate(skeleton.names)
        if name not in paired and name not in ends
    ]


def row_bends(
    points: np.ndarray,
    scores: np.ndarray,
    front: list[int],
    back: list[int],
    middle: list[int],
    frame_wh: tuple[int, int],
    cutoff: float | None,
) -> np.ndarray:
    """Per row: how far the body is bent, in degrees, NaN where it cannot be told.

    The axis runs from the back end to the front end (:func:`row_axis_ends`); of the usable
    ``middle`` points, the one furthest from that line is the body's elbow, and the bend is the
    turn the body makes there — the angle between back → elbow and elbow → front. 0° is a
    straight body; a mouse curled nose to tail approaches 180°. The sign is the turn's
    direction on screen: positive clockwise (image y points down). No usable middle point,
    or no axis, is NaN.
    """
    front_xy, back_xy = row_axis_ends(points, scores, front, back, frame_wh, cutoff)
    out = np.full(len(front_xy), np.nan)
    if not middle:
        return out
    px, usable = _usable_px(points, scores, frame_wh, cutoff)
    mid = px[:, middle, :]  # (M, S, 2)
    ok = usable[:, middle]
    axis = front_xy - back_xy  # (M, 2)
    length = np.hypot(axis[:, 0], axis[:, 1])
    rel = mid - back_xy[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        off_axis = np.abs(
            axis[:, None, 0] * rel[..., 1] - axis[:, None, 1] * rel[..., 0]
        )
        off_axis = off_axis / length[:, None]
    off_axis = np.where(ok, off_axis, -np.inf)
    has = ok.any(axis=1) & np.isfinite(length) & (length > 0)
    rows = np.flatnonzero(has)
    if not rows.size:
        return out
    elbow = mid[rows, np.argmax(off_axis[rows], axis=1), :]
    first = elbow - back_xy[rows]
    second = front_xy[rows] - elbow
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = (first * second).sum(axis=1)
    out[rows] = np.degrees(np.arctan2(cross, dot))
    return out


def turning_rate(deg: np.ndarray, t_ns: np.ndarray, *, half: int = 2) -> np.ndarray:
    """Degrees per second at each row, between the outermost rows with a heading within
    ``half`` either side of it, the short way round the circle (positive clockwise on screen);
    NaN without a heading on one side. Reaching a few rows out rather than one: a single
    row's keypoint jitter is not a turn."""
    deg = np.asarray(deg, dtype=np.float64)
    t_ns = np.asarray(t_ns, dtype=np.int64)
    n = len(deg)
    out = np.full(n, np.nan)
    known = np.isfinite(deg)
    for i in range(n):
        before = [j for j in range(max(i - half, 0), i + 1) if known[j]]
        after = [j for j in range(i, min(i + half, n - 1) + 1) if known[j]]
        if not before or not after or before[0] == after[-1]:
            continue
        a, b = before[0], after[-1]
        dt = (int(t_ns[b]) - int(t_ns[a])) / 1e9
        if dt > 0:
            out[i] = ((float(deg[b]) - float(deg[a]) + 180.0) % 360.0 - 180.0) / dt
    return out


def body_shape_series(
    points: np.ndarray, scores: np.ndarray, t_ns: np.ndarray, axis: BodyAxis
) -> dict[str, np.ndarray]:
    """The body's shape over rows: ``points`` ``(n, K, 2)`` normalized and ``scores`` ``(n, K)``
    per row (NaN where the subject has no keypoints), ``t_ns`` each row's time.

    Returns ``length`` (the axis length over ``axis.typical_length_px``, so 1.0 is the animal's
    usual length), ``turning`` (:func:`turning_rate`, °/s), and ``bend`` (:func:`row_bends`, °)
    only when the skeleton has a midline point between the axis ends — a two-point axis
    cannot bend, and an all-NaN curve would say it never did. NaN wherever the axis is
    undefined on that row.
    """
    front, back = list(axis.front), list(axis.back)
    deg, length = row_headings(points, scores, front, back, axis.frame_wh, axis.cutoff)
    has_axis = np.isfinite(length) & (length > 0)
    relative = np.full(len(length), np.nan)
    if axis.typical_length_px > 0:
        relative[has_axis] = length[has_axis] / float(axis.typical_length_px)
    out = {"length": relative}
    if axis.middle:
        out["bend"] = row_bends(
            points, scores, front, back, list(axis.middle), axis.frame_wh, axis.cutoff
        )
    out["turning"] = turning_rate(np.where(has_axis, deg, np.nan), t_ns)
    return out


def aligned_points(
    points: np.ndarray,
    front_xy: np.ndarray,
    back_xy: np.ndarray,
    frame_wh: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """One row's ``(K, 2)`` normalized points turned so the body axis points straight up, its
    middle at the frame's centre, ``scale`` pixels drawn per pixel measured — the pose as if
    the camera followed the animal. ``front_xy`` / ``back_xy`` are the axis ends in pixels
    (:func:`row_axis_ends`). Returns normalized points in the same frame."""
    w, h = (float(frame_wh[0]), float(frame_wh[1]))
    px = np.asarray(points, dtype=np.float64) * np.array([w, h])
    centre = (np.asarray(front_xy) + np.asarray(back_xy)) / 2.0
    dx, dy = np.asarray(front_xy) - np.asarray(back_xy)
    # Rotate the axis onto (0, -1): screen up, since image y points down.
    angle = -np.pi / 2 - np.arctan2(dy, dx)
    c, s = np.cos(angle), np.sin(angle)
    rel = px - centre
    turned = np.stack(
        [c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], axis=1
    )
    return (turned * float(scale) + np.array([w / 2.0, h / 2.0])) / np.array([w, h])


def heading_series(
    t_ns: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    front: list[int],
    back: list[int],
    frame_wh: tuple[int, int],
    *,
    cutoff: float | None,
    facing_camera_ratio: float,
    typical_length_px: float | None = None,
) -> HeadingSeries:
    """The defined rows of one track: a heading is undefined where one set has no usable
    point, where the axis has no length, or where it is shorter than
    ``facing_camera_ratio`` × the track's typical length (its 95th percentile over the
    rows given, or ``typical_length_px`` when the caller measured it over the whole run).
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    deg, length = row_headings(points, scores, front, back, frame_wh, cutoff)
    has_axis = np.isfinite(length) & (length > 0)
    if typical_length_px is None:
        typical = float(np.percentile(length[has_axis], 95)) if has_axis.any() else 0.0
    else:
        typical = float(typical_length_px)
    defined = has_axis & (length >= float(facing_camera_ratio) * typical)
    order = np.argsort(t_ns[defined], kind="stable")
    return HeadingSeries(
        t_ns=t_ns[defined][order],
        deg=deg[defined][order],
        n_rows=int(len(t_ns)),
        typical_length_px=typical,
    )


def _wrap(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


def heading_at(
    series: HeadingSeries, t_ns: np.ndarray, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """The heading at each time, and where it came from.

    Returns ``(deg (N,) float32, source (N,) int8)``: a time that is a defined row takes
    that row's heading (``SOURCE_OWN``); a time between two defined rows no further apart
    than ``max_gap_s`` takes the interpolation between them, the short way round the circle
    (``SOURCE_INTERPOLATED``); any other time — a longer gap, or before the first / after the
    last defined row — is NaN with ``SOURCE_NONE``, and the embedding writes no row for it.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    deg = np.full(t_ns.shape, np.nan, dtype=np.float64)
    source = np.full(t_ns.shape, SOURCE_NONE, dtype=np.int8)
    if series.t_ns.size == 0 or t_ns.size == 0:
        return deg.astype(np.float32), source
    known_t, known_deg = series.t_ns, series.deg
    after = np.searchsorted(known_t, t_ns, side="left")  # first known >= t
    exact = (after < known_t.size) & (
        known_t[np.minimum(after, known_t.size - 1)] == t_ns
    )
    deg[exact] = known_deg[after[exact]]
    source[exact] = SOURCE_OWN
    between = ~exact & (after > 0) & (after < known_t.size)
    if between.any():
        lo = after[between] - 1
        hi = after[between]
        t_lo, t_hi = known_t[lo], known_t[hi]
        # an unbounded gap (the canvas pre-pass measures every frame that could ever
        # receive a heading) interpolates across any gap between defined rows
        gap_ok = (
            np.ones(t_lo.shape, dtype=bool)
            if not np.isfinite(max_gap_s)
            else (t_hi - t_lo) <= int(round(float(max_gap_s) * 1e9))
        )
        rows = np.flatnonzero(between)[gap_ok]
        lo, hi, t_lo, t_hi = lo[gap_ok], hi[gap_ok], t_lo[gap_ok], t_hi[gap_ok]
        frac = (t_ns[rows] - t_lo) / np.maximum(t_hi - t_lo, 1)
        delta = _wrap(known_deg[hi] - known_deg[lo])
        deg[rows] = _wrap(known_deg[lo] + frac * delta)
        source[rows] = SOURCE_INTERPOLATED
    return deg.astype(np.float32), source


def mirror_flags(deg: np.ndarray, t_ns: np.ndarray, hold_s: float) -> np.ndarray:
    """For ``mirror``: flip left-right where the animal faces left (``cos < 0``), with the
    hold looking **both ways** (§14): the defined frames split into runs facing one way, and
    a run that lasts at least ``hold_s`` sets the side from its **first** frame, so no frame
    of a genuine turn is embedded facing the wrong way; a shorter run is a flicker at ±90°
    and inherits the side before it. Frames without a heading are never mirrored.

    A run's duration is measured to the next run's start (a one-frame flicker lasts one frame
    period, not zero); the last run is measured to its own last frame.
    """
    deg = np.asarray(deg, dtype=np.float64)
    t_ns = np.asarray(t_ns, dtype=np.int64)
    out = np.zeros(deg.shape, dtype=bool)
    defined = np.isfinite(deg)
    if not defined.any():
        return out
    faces_left = np.cos(np.radians(np.where(defined, deg, 0.0))) < 0
    hold_ns = int(round(float(hold_s) * 1e9))
    rows = np.flatnonzero(defined).tolist()
    runs: list[tuple[int, int, bool]] = []  # (first row, last row, side)
    for i in rows:
        side = bool(faces_left[i])
        if runs and runs[-1][2] == side:
            runs[-1] = (runs[-1][0], i, side)
        else:
            runs.append((i, i, side))
    durations = [
        (
            int(t_ns[runs[k + 1][0]] - t_ns[first])
            if k + 1 < len(runs)
            else int(t_ns[last] - t_ns[first])
        )
        for k, (first, last, _side) in enumerate(runs)
    ]
    held = [d >= hold_ns for d in durations]
    # the opening side: the first run that holds; a series of nothing but flickers keeps
    # its first frame's side
    state = next((runs[k][2] for k in range(len(runs)) if held[k]), runs[0][2])
    for k, (first, last, side) in enumerate(runs):
        if held[k]:
            state = side
        for i in rows:
            if first <= i <= last:
                out[i] = state
    return out


def definition_columns(
    definition: dict, names: tuple[str, ...]
) -> tuple[list[int], list[int]]:
    """The skeleton columns a validated definition's two sets name."""
    index = {name: i for i, name in enumerate(names)}
    return (
        [index[p] for p in definition["front"]],
        [index[p] for p in definition["back"]],
    )


def track_series(
    run,
    track_id: int,
    definition: dict,
    frame_wh: tuple[int, int],
    *,
    cutoff: float | None,
    facing_camera_ratio: float,
    part_ids: list[str] | None = None,
) -> dict[str, HeadingSeries]:
    """One `HeadingSeries` per part of a track of a `KeypointTrackRun` (or detect run),
    the typical axis length measured over the **whole run** first, so a part where the
    animal faced the camera throughout is judged against the run, not against itself."""
    names = tuple(run.skeleton.names)
    front, back = definition_columns(validate_definition(definition, names), names)
    parts = run.part_ids(track_id) if part_ids is None else list(part_ids)
    rows = {part_id: run.rows(track_id, part_id) for part_id in parts}
    stacked = [
        row_headings(r["points"], r["point_scores"], front, back, frame_wh, cutoff)[1]
        for r in rows.values()
    ]
    all_lengths = np.concatenate(stacked) if stacked else np.zeros(0, dtype=np.float64)
    has_axis = np.isfinite(all_lengths) & (all_lengths > 0)
    typical = float(np.percentile(all_lengths[has_axis], 95)) if has_axis.any() else 0.0
    return {
        part_id: heading_series(
            r["t_ns"],
            r["points"],
            r["point_scores"],
            front,
            back,
            frame_wh,
            cutoff=cutoff,
            facing_camera_ratio=facing_camera_ratio,
            typical_length_px=typical,
        )
        for part_id, r in rows.items()
    }


def coverage_counts(source: np.ndarray) -> dict[str, int]:
    """``{frames, own, interpolated, dropped}`` over a `heading_at` ``source`` array."""
    source = np.asarray(source)
    return {
        "frames": int(source.size),
        "own": int((source == SOURCE_OWN).sum()),
        "interpolated": int((source == SOURCE_INTERPOLATED).sum()),
        "dropped": int((source == SOURCE_NONE).sum()),
    }


def longest_gap_s(t_ns: np.ndarray, source: np.ndarray) -> float:
    """The longest stretch without a heading, in seconds (0 when none): the time between
    the frames with a heading that bracket a run of dropped frames, so a single dropped
    frame reads as one frame period rather than zero. A run at either edge of the series
    is measured from its outer dropped frame."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    dropped = np.asarray(source) == SOURCE_NONE
    n = int(dropped.size)
    if n == 0 or not dropped.any():
        return 0.0
    longest = 0
    start = None
    for i, flag in enumerate(dropped.tolist()):
        if flag and start is None:
            start = i
        if (not flag or i == n - 1) and start is not None:
            end = i if flag else i - 1
            before = start - 1 if start > 0 else start
            after = end + 1 if end + 1 < n else end
            longest = max(longest, int(t_ns[after] - t_ns[before]))
            start = None
    return longest / 1e9
