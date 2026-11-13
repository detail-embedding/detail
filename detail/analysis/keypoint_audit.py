"""Scoring one keypoint track run against an audit set's labels (§26 "Auditing a keypoint track
run").

**The only reader of the labels.** They score the run; nothing here feeds a method. The way a
label could have built the run — a frame the run recorded as an example frame, or a frame a
distilled student's dataset used — is excluded by the caller (``excluded``) and counted, never
scored.

The definitions are the research scorer's, so a figure means what it meant in the studies: body
size ``s = sqrt(w_px * h_px)`` of the frame's tracked box; error in pixels and as a fraction of
``s``; PCK at 0.05 and 0.1 of ``s``; a visible label the run did not place (or placed below its
display cut-off) is **missed** — it counts against PCK and is left out of the error medians; a
not-visible label the run placed is a **false placement**; left/right pairs sort into correct /
swapped / collapsed / off; the heading error is the angle between the labeled and the predicted
heading (a heading definition's front and back sets, §14).

**Every figure ships with its floor and its interval** (the headline-metric rule). The floor is
"the box alone": each keypoint at its mean box-relative position over the *other* labeled frames
(leave-one-out), scored over **the same cells the run's figure used** — a run that leaves out its
hard frames is not compared against a floor over all of them. A pair's floor is the
random-assignment rate. A proportion carries a 95% Wilson interval with ``n`` = the frames
contributing — conservative where a group pools several points per frame, because points on one
frame are not independent, and silent about frames from one video being alike. A median carries
a seeded frame bootstrap. Below ``MIN_N_FOR_ESTIMATE`` frames the value, its interval and its
floor are withheld (None) and only ``n`` is reported: 1.00 from three frames renders like 1.00
from fifty while supporting nothing.

Pure numpy; the run's rows arrive already read (``pipeline/keypoint_audit.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from detail.core.examples import ExampleSet
from detail.storage.behavior_audit import MIN_N_FOR_ESTIMATE, wilson_interval

PCK_RADII = (0.05, 0.10)
HEADING_WITHIN_DEG = 15.0
HEADING_FLIPPED_DEG = 90.0
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 0
PAIR_OUTCOMES = ("correct", "swapped", "collapsed", "off")
GROUPS = ("all", "paired", "unpaired")
#: What the intervals do not account for; the report states it (§26 *Figures*).
INTERVAL_NOTE = (
    "Intervals: a proportion carries a 95% Wilson interval with n = the frames contributing "
    "(conservative where a group pools several points per frame); a median a seeded "
    f"{BOOTSTRAP_DRAWS}-draw frame bootstrap. Neither accounts for frames from one video "
    f"being alike. Below {MIN_N_FOR_ESTIMATE} frames a figure is withheld."
)


@dataclass(frozen=True)
class RunPart:
    """One (track, part) of a run's rows: normalized points, scores, and which points the run
    placed at or above its display cut-off (``visible``)."""

    t_ns: np.ndarray  # (M,) int64
    points: np.ndarray  # (M, K, 2) normalized [0, 1], NaN where absent
    scores: np.ndarray  # (M, K)
    visible: np.ndarray  # (M, K) bool


@dataclass(frozen=True)
class RunRows:
    """One recording's audited run, as the scorer needs it."""

    run_id: str
    tracking_run_id: str
    names: tuple[str, ...]  # the run skeleton's names, column order
    parts: dict[tuple[int, str], RunPart]  # (track_id, part_id) -> rows


@dataclass
class AuditScores:
    """What a run scored against an audit set — derived, never stored (§26)."""

    coverage: dict[str, int]
    rows: list[
        dict
    ]  # one per figure: scope, name, metric, value, ci_low, ci_high, n, floor
    frames: list[dict]  # one per scored (frame, keypoint)
    name_map: dict[str, str | None]  # audit name -> run name
    unmapped: list[str] = field(default_factory=list)  # audit keypoints not in the run

    def row(self, scope: str, name: str, metric: str) -> dict | None:
        return next(
            (
                r
                for r in self.rows
                if r["scope"] == scope and r["name"] == name and r["metric"] == metric
            ),
            None,
        )


def run_names(runs: dict[str, RunRows]) -> tuple[str, ...]:
    """The keypoint names every recording's run places. Raises ``ValueError`` when they
    disagree, or when there is no run at all."""
    if not runs:
        raise ValueError(
            "no keypoint track run to audit: none of the audit set's recordings has one"
        )
    names = {r.names for r in runs.values()}
    if len(names) > 1:
        raise ValueError(
            "the runs being audited place different keypoints in different recordings "
            f"({sorted(names)}); audit one method's runs at a time"
        )
    return next(iter(names))


def score(
    audit: ExampleSet,
    runs: dict[str, RunRows],
    name_map: dict[str, str | None],
    excluded: set[tuple],
    frame_wh: dict[str, tuple[int, int]],
    *,
    heading: dict | None = None,
    heading_name: str = "heading",
) -> AuditScores:
    """Score ``runs`` (recording id -> that recording's run) against ``audit``'s labels.

    ``name_map`` maps each audit keypoint to the run keypoint it is scored against (None = not
    in this run). ``excluded`` holds the ``(recording_id, part_id, t_ns, track_id)`` keys of
    frames that helped build the run. ``frame_wh`` is each recording's frame size in pixels.
    ``heading`` is ``{front: [...], back: [...]}`` in **audit** names, or None for no heading
    figures.

    Raises:
        ValueError: the runs disagree on their skeleton, or the map names a keypoint the run
            does not place.
    """
    placed_names = list(run_names(runs))
    bad = sorted(
        v for v in name_map.values() if v is not None and v not in placed_names
    )
    if bad:
        raise ValueError(
            f"the name map points at {bad}, which the run does not place; it places "
            f"{placed_names}"
        )
    names = list(audit.skeleton.names)
    mapped = [name_map.get(n) is not None for n in names]
    columns = [
        placed_names.index(name_map[n]) if m else -1 for n, m in zip(names, mapped)
    ]

    coverage = {"excluded": 0, "uncovered": 0, "scored": 0, "edited_after_report": 0}
    coverage.update(audit.counts())
    scored = []
    for i, frame in enumerate(audit.frames):
        if not audit.labeled(i):
            continue
        if any(m.extra.get("edited_after_report") for m in frame.marks):
            coverage["edited_after_report"] += 1
        if frame.key() in excluded:
            coverage["excluded"] += 1
            continue
        row = _run_row(runs.get(frame.recording_id), frame)
        if row is None:
            coverage["uncovered"] += 1
            continue
        scored.append((frame, frame_wh[frame.recording_id], *row))
    coverage["scored"] = len(scored)
    cells = _cells(scored, names, columns)
    rows = _rows(cells, audit, names, mapped)
    if heading is not None:
        rows += _heading_rows(cells, names, heading, heading_name)
    return AuditScores(
        coverage=coverage,
        rows=rows,
        frames=_frame_rows(cells, scored, names, columns, placed_names, mapped),
        name_map={n: name_map.get(n) for n in names},
        unmapped=[n for n, m in zip(names, mapped) if not m],
    )


# --------------------------------------------------------------------------- #
# Matching a labeled frame to the run's row
# --------------------------------------------------------------------------- #
def _run_row(run: RunRows | None, frame):
    """``(points (K, 2) normalized, visible (K,), score (K,))`` of the run's row for this frame,
    or None when the run does not cover it: no run in the recording, the run read another
    tracking run, no rows for the track in that part, or no row at the frame's time."""
    if run is None or run.tracking_run_id != frame.tracking_run_id:
        return None
    part = run.parts.get((int(frame.track_id), frame.part_id))
    if part is None:
        return None
    hits = np.flatnonzero(np.asarray(part.t_ns) == int(frame.t_ns))
    if not hits.size:
        return None
    j = int(hits[0])
    return (
        np.asarray(part.points[j], float),
        np.asarray(part.visible[j], bool),
        np.asarray(part.scores[j], float),
    )


# --------------------------------------------------------------------------- #
# The per-cell arrays every figure is computed from
# --------------------------------------------------------------------------- #
def _cells(scored, names, columns) -> dict:
    """Arrays over (scored frame, audit keypoint): labels, predictions, the floor, errors."""
    n, k = len(scored), len(names)
    label = np.full((n, k, 2), np.nan)
    label_visible = np.zeros((n, k), bool)
    label_decided = np.zeros((n, k), bool)
    pred = np.full((n, k, 2), np.nan)
    placed = np.zeros((n, k), bool)
    score = np.full((n, k), np.nan)
    box = np.ones((n, 4))  # x0, y0, w, h in pixels
    body = np.ones(n)
    recording = []
    for i, (frame, (w, h), xy, ok, sc) in enumerate(scored):
        x0, y0, x1, y1 = frame.bbox
        box[i] = (x0 * w, y0 * h, max((x1 - x0) * w, 1.0), max((y1 - y0) * h, 1.0))
        body[i] = np.sqrt(box[i, 2] * box[i, 3])
        recording.append(frame.recording_id)
        for c, name in enumerate(names):
            mark = frame.mark_for(name)
            label_decided[i, c] = mark is not None
            if mark is not None and mark.visible and mark.x is not None:
                label_visible[i, c] = True
                label[i, c] = (mark.x * w, mark.y * h)
            j = columns[c]
            if j >= 0:
                score[i, c] = sc[j]
                if ok[j] and np.isfinite(xy[j]).all():
                    placed[i, c] = True
                    pred[i, c] = (xy[j, 0] * w, xy[j, 1] * h)
    floor = _floor(label, label_visible, box)
    error = np.linalg.norm(pred - label, axis=2)
    floor_error = np.linalg.norm(floor - label, axis=2)
    return {
        "label": label,
        "visible": label_visible,
        "decided": label_decided,
        "pred": pred,
        "placed": placed,
        "score": score,
        "floor": floor,
        "floor_placed": np.isfinite(floor).all(axis=2),
        "error": error,
        "error_body": error / body[:, None],
        "floor_error": floor_error,
        "floor_error_body": floor_error / body[:, None],
        "body": body,
        "recording": np.asarray(recording, dtype=object),
    }


def _floor(label: np.ndarray, label_visible: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Each keypoint at its mean box-relative position over the *other* visible labels of it
    (leave-one-out, so a frame's own label never places its floor); NaN with no other label.
    """
    origin, size = box[:, None, :2], box[:, None, 2:]
    rel = np.where(label_visible[..., None], (label - origin) / size, 0.0)
    total = rel.sum(axis=0)  # (K, 2)
    count = label_visible.sum(axis=0).astype(float)  # (K,)
    others = count[None, :] - label_visible
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total[None] - rel) / np.where(others > 0, others, np.nan)[..., None]
    return origin + mean * size


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _row(scope, name, metric, value, interval, n, floor=None) -> dict:
    withheld = n < MIN_N_FOR_ESTIMATE
    return {
        "scope": scope,
        "name": name,
        "metric": metric,
        "value": None if withheld or value is None else float(value),
        "ci_low": None if withheld or interval is None else float(interval[0]),
        "ci_high": None if withheld or interval is None else float(interval[1]),
        "n": int(n),
        "floor": None if withheld or floor is None else float(floor),
    }


def _rate(hits: np.ndarray, frames: np.ndarray):
    """Share of ``hits`` over its cells, its Wilson interval at n = frames contributing, n."""
    n_frames = len(np.unique(frames))
    if not hits.size:
        return None, None, 0
    p = float(np.mean(hits))
    return p, wilson_interval(p * n_frames, n_frames), n_frames


def _share(hits: np.ndarray) -> float | None:
    return float(np.mean(hits)) if hits.size else None


def median_with_interval(values: np.ndarray, frames: np.ndarray):
    """Median, its seeded frame-bootstrap 95% interval (``BOOTSTRAP_DRAWS`` resamples of the
    contributing frames, each carrying all its cells), and the frames contributing."""
    values = np.asarray(values, float)
    if not values.size:
        return None, None, 0
    uniq, inv = np.unique(frames, return_inverse=True)
    m = len(uniq)
    order = np.argsort(values, kind="stable")
    v, owner = values[order], inv[order]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for start in range(0, BOOTSTRAP_DRAWS, 200):
        b = min(200, BOOTSTRAP_DRAWS - start)
        counts = np.stack(
            [np.bincount(rng.integers(0, m, m), minlength=m) for _ in range(b)]
        )
        cum = np.cumsum(counts[:, owner], axis=1).astype(float)
        half = cum[:, -1:] / 2.0
        draws.append(v[np.minimum((cum < half).sum(axis=1), len(v) - 1)])
    lo, hi = np.percentile(np.concatenate(draws), [2.5, 97.5])
    return float(np.median(values)), (float(lo), float(hi)), m


def _finite_median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _accuracy_rows(scope: str, name: str, cells: dict, mask: np.ndarray) -> list[dict]:
    """The accuracy figures over the (frame, keypoint) cells ``mask`` selects, each floor over
    the cells its own figure used."""
    frame_ids = np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape)
    vis = mask & cells["visible"]
    hidden = mask & cells["decided"] & ~cells["visible"]
    placed = vis & cells["placed"]
    rows = []
    for radius in PCK_RADII:
        hit = cells["error_body"][vis] <= radius  # NaN error (not placed) is a miss
        floor_hit = cells["floor_error_body"][vis] <= radius
        p, ci, n = _rate(hit, frame_ids[vis])
        rows.append(
            _row(
                scope,
                name,
                f"pck{int(round(radius * 100)):02d}",
                p,
                ci,
                n,
                _share(floor_hit),
            )
        )
    for metric, err, floor_err in (
        ("error_px_median", "error", "floor_error"),
        ("error_body_median", "error_body", "floor_error_body"),
    ):
        med, ci, n = median_with_interval(cells[err][placed], frame_ids[placed])
        floor = _finite_median(cells[floor_err][placed])
        rows.append(_row(scope, name, metric, med, ci, n, floor))
    p, ci, n = _rate(~cells["placed"][vis], frame_ids[vis])
    rows.append(
        _row(scope, name, "missed", p, ci, n, _share(~cells["floor_placed"][vis]))
    )
    p, ci, n = _rate(cells["placed"][hidden], frame_ids[hidden])
    rows.append(
        _row(
            scope,
            name,
            "false_placement",
            p,
            ci,
            n,
            _share(cells["floor_placed"][hidden]),
        )
    )
    return rows


def _rows(cells: dict, audit: ExampleSet, names: list[str], mapped: list[bool]) -> list:
    n = cells["visible"].shape[0]
    k = len(names)
    paired = {name for pair in audit.skeleton.pairs for name in pair}
    keep_mapped = np.asarray(mapped, bool)
    groups = {
        "all": np.ones(k, bool),
        "paired": np.asarray([nm in paired for nm in names], bool),
        "unpaired": np.asarray([nm not in paired for nm in names], bool),
    }
    rows = []
    for group in GROUPS:
        keep = groups[group] & keep_mapped
        if keep.any():
            mask = np.broadcast_to(keep[None, :], (n, k))
            rows += _accuracy_rows("group", group, cells, mask)
    for c, name in enumerate(names):
        if mapped[c]:
            mask = np.zeros((n, k), bool)
            mask[:, c] = True
            rows += _accuracy_rows("keypoint", name, cells, mask)
    mapped_mask = np.broadcast_to(keep_mapped[None, :], (n, k))
    for recording_id in sorted(set(cells["recording"])):
        mask = mapped_mask & (cells["recording"] == recording_id)[:, None]
        rows += _accuracy_rows("recording", recording_id, cells, mask)
    rows += _pair_rows(cells, audit, names, mapped)
    return rows


def pair_outcome(pa, pb, ga, gb, body: float) -> str:
    """What happened to a left/right pair on one frame (the research scorer's rule): ``off``
    when either prediction is farther than 0.1 body from both labels; otherwise each prediction
    takes the side it is nearer, giving ``correct``, ``swapped`` or ``collapsed``."""

    def d(p, g) -> float:
        return float(np.hypot(p[0] - g[0], p[1] - g[1]))

    radius = 0.10 * body
    if min(d(pa, ga), d(pa, gb)) > radius or min(d(pb, ga), d(pb, gb)) > radius:
        return "off"
    side_a = "a" if d(pa, ga) <= d(pa, gb) else "b"
    side_b = "a" if d(pb, ga) <= d(pb, gb) else "b"
    if side_a == side_b:
        return "collapsed"
    return "correct" if side_a == "a" else "swapped"


def _pair_rows(cells, audit, names, mapped) -> list[dict]:
    """Per left/right pair, over the frames where both are labeled visible and both placed.
    The floor of ``correct`` and ``swapped`` is the random-assignment rate: half the frames
    where the two predictions landed one on each label."""
    rows = []
    index = {name: c for c, name in enumerate(names)}
    for left, right in audit.skeleton.pairs:
        ia, ib = index[left], index[right]
        if not (mapped[ia] and mapped[ib]):
            continue
        both = (
            cells["visible"][:, ia]
            & cells["visible"][:, ib]
            & cells["placed"][:, ia]
            & cells["placed"][:, ib]
        )
        frames = np.flatnonzero(both)
        outcomes = np.asarray(
            [
                pair_outcome(
                    cells["pred"][i, ia],
                    cells["pred"][i, ib],
                    cells["label"][i, ia],
                    cells["label"][i, ib],
                    cells["body"][i],
                )
                for i in frames
            ],
            dtype=object,
        )
        resolved = (
            float(np.isin(outcomes, ("correct", "swapped")).mean()) / 2.0
            if frames.size
            else None
        )
        for outcome in PAIR_OUTCOMES:
            p, ci, n = _rate(outcomes == outcome, frames)
            floor = resolved if outcome in ("correct", "swapped") else None
            rows.append(_row("pair", f"{left}|{right}", outcome, p, ci, n, floor))
    return rows


def _set_mean(xy: np.ndarray, usable: np.ndarray, columns: list[int]) -> np.ndarray:
    """Per frame, the mean pixel position of the usable points of ``columns``; NaN where none."""
    sub = xy[:, columns, :]
    ok = usable[:, columns]
    count = ok.sum(axis=1)
    total = np.where(ok[..., None], sub, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count[:, None]
    mean[count == 0] = np.nan
    return mean


def _heading_deg(xy: np.ndarray, usable: np.ndarray, front, back) -> np.ndarray:
    """Image degrees from the back set's mean to the front set's (``analysis.body_axis``'s
    definition, in pixels); NaN where a set has no usable point."""
    f = _set_mean(xy, usable, front)
    b = _set_mean(xy, usable, back)
    return np.degrees(np.arctan2(f[:, 1] - b[:, 1], f[:, 0] - b[:, 0]))


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _heading_rows(cells, names, heading, heading_name) -> list[dict]:
    """The heading error over the frames where both the labeled and the predicted heading are
    defined; the floor's over the same frames, from the box-alone positions."""
    index = {name: c for c, name in enumerate(names)}
    front = [index[p] for p in heading["front"]]
    back = [index[p] for p in heading["back"]]
    labeled = _heading_deg(cells["label"], cells["visible"], front, back)
    predicted = _heading_deg(cells["pred"], cells["placed"], front, back)
    floor = _heading_deg(cells["floor"], cells["floor_placed"], front, back)
    both = np.isfinite(labeled) & np.isfinite(predicted)
    frames = np.flatnonzero(both)
    angle = _angle_between(predicted, labeled)[both]
    floor_angle = _angle_between(floor, labeled)[both]
    floor_angle = floor_angle[np.isfinite(floor_angle)]
    med, ci, n = median_with_interval(angle, frames)
    rows = [
        _row(
            "heading",
            heading_name,
            "error_deg_median",
            med,
            ci,
            n,
            _finite_median(floor_angle),
        )
    ]
    for metric, hit, floor_hit in (
        (
            "within_15deg",
            angle <= HEADING_WITHIN_DEG,
            floor_angle <= HEADING_WITHIN_DEG,
        ),
        ("flipped", angle > HEADING_FLIPPED_DEG, floor_angle > HEADING_FLIPPED_DEG),
    ):
        p, ci, n = _rate(hit, frames)
        rows.append(_row("heading", heading_name, metric, p, ci, n, _share(floor_hit)))
    return rows


# --------------------------------------------------------------------------- #
# The per-frame table
# --------------------------------------------------------------------------- #
def _num(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _frame_rows(cells, scored, names, columns, placed_names, mapped) -> list[dict]:
    out = []
    for i, (frame, *_rest) in enumerate(scored):
        edited = any(m.extra.get("edited_after_report") for m in frame.marks)
        for c, name in enumerate(names):
            if not mapped[c]:
                continue
            vis, ok = bool(cells["visible"][i, c]), bool(cells["placed"][i, c])
            out.append(
                {
                    "recording_id": frame.recording_id,
                    "track_id": int(frame.track_id),
                    "part_id": frame.part_id,
                    "frame_idx": int(frame.frame_idx),
                    "t_ns": int(frame.t_ns),
                    "keypoint": name,
                    "run_keypoint": placed_names[columns[c]],
                    "label_visible": vis,
                    "label_x_px": _num(cells["label"][i, c, 0]) if vis else None,
                    "label_y_px": _num(cells["label"][i, c, 1]) if vis else None,
                    "placed": ok,
                    "pred_x_px": _num(cells["pred"][i, c, 0]) if ok else None,
                    "pred_y_px": _num(cells["pred"][i, c, 1]) if ok else None,
                    "score": _num(cells["score"][i, c]),
                    "error_px": _num(cells["error"][i, c]) if vis and ok else None,
                    "error_body": (
                        _num(cells["error_body"][i, c]) if vis and ok else None
                    ),
                    "body_px": float(cells["body"][i]),
                    "floor_error_body": (
                        _num(cells["floor_error_body"][i, c]) if vis else None
                    ),
                    "edited_after_report": edited,
                }
            )
    return out
