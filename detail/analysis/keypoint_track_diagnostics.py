"""What a keypoint track looks like over time, without labels (§26 *Previewing a track method*).

*Preview tracking* shows a few settings side by side, and these figures sit under the clips. They
**describe; on their own they do not rank.** The motion figures move in the direction smoothing
pushes them, so the setting that smooths hardest "wins" each of them while its points drift off
the animal: a lower median move or a steadier skeleton is a trade, not an improvement. Which
setting is right is read off the clips (or, with labels, off *Tune tracking*). That is why the
table always carries **detection alone** as its first row — the floor every column is read
against.

**Agreement** is the counterweight: it falls when a setting leaves the points detection was most
sure of, which is what over-smoothing and a too-tight transition both do (on 25 fps oblique
video the transition default lost a third of the paws detection had placed). Detection alone
agrees with itself by construction, so its 100% is a ceiling, not a score. The grid search
(:func:`pareto_front`) marks the settings no other setting beats on agreement, placed, jumps and
swaps together: a **provisional** rule, not yet checked against labels.

Every figure is over the points the run tracks (``tracked``), so a point left as detected moves
none of them:

- **placed** — the share of subject-frames × points with a point placed.
- **median move** — the median distance a point moves from one row to the next, as a share of
  the subject's body scale. Real motion is in it as well as jitter: compare settings on the same
  clip, never across clips.
- **jumps / min** — moves of more than :data:`JUMP_BODY` body lengths in one row, per minute of
  subject time: at 30 fps that is 15 body lengths a second, faster than a mouse runs, so nearly
  always a point that left the body part and came back.
- **swaps / min** — rows where a left/right pair (both tracked) reads more naturally crossed than
  straight from the row before, per minute: a left/right flip.
- **stretch** — how much each skeleton edge between two tracked points changes length (its
  coefficient of variation, length over body scale), the median over edges. A rigid edge (ear to
  ear) should barely change; a point that wanders stretches its edges.
- **agreement** — of the cells where detection was confident (a placed point scoring at least
  that point's median detection score over the subject's rows), the share where the setting's
  point is within :data:`AGREE_BODY` body lengths of detection's.

Consecutive rows are rows at most one and a half rows apart in time, so a gap (the subject not
detected, or an every-n run) is never read as one long move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from detail.core.skeleton import Skeleton

#: A move longer than this many body lengths in one row counts as a jump.
JUMP_BODY = 0.5
#: A pair reads crossed when the crossed reading costs less than this share of the straight one.
SWAP_RATIO = 0.5
#: An edge needs this many rows with both points placed for its stretch to count.
MIN_EDGE_ROWS = 10
#: A point within this many body lengths of a confident detection agrees with it (the audit's
#: PCK@0.1 radius).
AGREE_BODY = 0.1
#: The figures, in the order every surface shows them, with how each reads.
METRICS = (
    "placed",
    "median_move",
    "jumps_per_min",
    "swaps_per_min",
    "stretch",
    "agreement",
)
METRIC_LABELS = {
    "placed": "Placed",
    "median_move": "Median move (body / row)",
    "jumps_per_min": "Jumps / min",
    "swaps_per_min": "Left/right swaps / min",
    "stretch": "Skeleton stretch",
    "agreement": "Agreement",
}
#: The figures that read as a share (every surface shows them as a percentage); the rest
#: are counts per minute.
SHARES = ("placed", "median_move", "stretch", "agreement")
#: The grid search's provisional front: each figure, and whether higher is better.
FRONT_METRICS = {
    "agreement": True,
    "placed": True,
    "jumps_per_min": False,
    "swaps_per_min": False,
}
_NS_PER_S = 1_000_000_000


@dataclass
class Tally:
    """The raw counts behind the figures, so a pooled figure is computed over the pooled rows
    rather than averaged from per-recording figures."""

    cells: int = 0
    placed: int = 0
    moves: list[np.ndarray] = field(default_factory=list)
    jumps: int = 0
    swaps: int = 0
    pair_steps: int = 0
    minutes: float = 0.0
    edge_lengths: dict[tuple[int, int], list[np.ndarray]] = field(default_factory=dict)
    confident: int = 0
    agreed: int = 0

    def add(self, other: "Tally") -> None:
        self.cells += other.cells
        self.placed += other.placed
        self.moves.extend(other.moves)
        self.jumps += other.jumps
        self.swaps += other.swaps
        self.pair_steps += other.pair_steps
        self.minutes += other.minutes
        self.confident += other.confident
        self.agreed += other.agreed
        for edge, lengths in other.edge_lengths.items():
            self.edge_lengths.setdefault(edge, []).extend(lengths)

    def figures(self) -> dict[str, float | None]:
        """``{metric: value}``; None where there is nothing to measure (no consecutive rows,
        no pair, no edge with enough rows)."""
        moves = np.concatenate(self.moves) if self.moves else np.zeros(0)
        stretches = []
        for lengths in self.edge_lengths.values():
            joined = np.concatenate(lengths)
            if len(joined) >= MIN_EDGE_ROWS and joined.mean() > 0:
                stretches.append(float(joined.std() / joined.mean()))
        return {
            "placed": self.placed / self.cells if self.cells else None,
            "median_move": float(np.median(moves)) if len(moves) else None,
            "jumps_per_min": self.jumps / self.minutes if self.minutes else None,
            "swaps_per_min": (
                self.swaps / self.minutes if self.minutes and self.pair_steps else None
            ),
            "stretch": float(np.median(stretches)) if stretches else None,
            "agreement": self.agreed / self.confident if self.confident else None,
        }


def tally(
    points_px: np.ndarray,
    t_ns: np.ndarray,
    scale_px: np.ndarray,
    skeleton: Skeleton,
    tracked: np.ndarray,
    rows_per_second: float,
    *,
    detected_px: np.ndarray,
    detected_scores: np.ndarray,
) -> Tally:
    """One subject's rows: ``points_px`` ``(M, K, 2)`` in frame pixels (NaN: not placed),
    ``t_ns`` ``(M,)``, ``scale_px`` ``(M,)`` its body scale per row, ``tracked`` ``(K,)`` bool;
    ``detected_px`` / ``detected_scores`` the same rows as detection placed and scored them,
    which agreement is measured against.
    """
    points_px = np.asarray(points_px, dtype=np.float64)
    t_ns = np.asarray(t_ns, dtype=np.int64)
    scale = np.maximum(np.asarray(scale_px, dtype=np.float64), 1e-6)
    tracked = np.asarray(tracked, dtype=bool)
    cols = np.flatnonzero(tracked)
    out = Tally()
    m = len(t_ns)
    finite = np.isfinite(points_px).all(axis=2)  # (M, K)
    out.cells = int(m * len(cols))
    out.placed = int(finite[:, cols].sum())
    out.confident, out.agreed = _agreement(
        points_px, detected_px, detected_scores, scale, cols
    )
    if m < 2 or not len(cols):
        return out
    step_ns = _NS_PER_S / float(rows_per_second)
    steps = np.flatnonzero(np.diff(t_ns) <= 1.5 * step_ns)  # row i -> i + 1 consecutive
    out.minutes = len(steps) / float(rows_per_second) / 60.0
    if len(steps):
        before, after = points_px[steps][:, cols], points_px[steps + 1][:, cols]
        both = finite[steps][:, cols] & finite[steps + 1][:, cols]
        dist = np.linalg.norm(after - before, axis=2) / scale[steps + 1][:, None]
        moves = dist[both]
        out.moves.append(moves)
        out.jumps = int((moves > JUMP_BODY).sum())
        for left, right in skeleton.pair_indices():
            if not (tracked[left] and tracked[right]):
                continue
            ok = (
                finite[steps, left]
                & finite[steps, right]
                & finite[steps + 1, left]
                & finite[steps + 1, right]
            )
            lb, rb = points_px[steps, left], points_px[steps, right]
            la, ra = points_px[steps + 1, left], points_px[steps + 1, right]
            straight = np.linalg.norm(la - lb, axis=1) + np.linalg.norm(ra - rb, axis=1)
            crossed = np.linalg.norm(la - rb, axis=1) + np.linalg.norm(ra - lb, axis=1)
            out.pair_steps += int(ok.sum())
            out.swaps += int((ok & (crossed < SWAP_RATIO * straight)).sum())
    index = {name: i for i, name in enumerate(skeleton.names)}
    for a, b in skeleton.edges:
        i, j = index[a], index[b]
        if not (tracked[i] and tracked[j]):
            continue
        both = finite[:, i] & finite[:, j]
        length = np.linalg.norm(points_px[:, i] - points_px[:, j], axis=1) / scale
        out.edge_lengths.setdefault((i, j), []).append(length[both])
    return out


def _agreement(points_px, detected_px, detected_scores, scale, cols) -> tuple[int, int]:
    """``(confident, agreed)``: detection's confident cells over ``cols`` (a placed point
    scoring at least its point's median placed score over these rows) and how many of them
    ``points_px`` places within :data:`AGREE_BODY` body lengths."""
    detected_px = np.asarray(detected_px, dtype=np.float64)[:, cols]
    detected_scores = np.asarray(detected_scores, dtype=np.float64)[:, cols]
    placed = np.isfinite(detected_px).all(axis=2) & np.isfinite(detected_scores)
    confident = np.zeros_like(placed)
    for k in range(placed.shape[1]):
        if placed[:, k].any():
            cut = np.median(detected_scores[placed[:, k], k])
            confident[:, k] = placed[:, k] & (detected_scores[:, k] >= cut)
    dist = np.linalg.norm(points_px[:, cols] - detected_px, axis=2) / scale[:, None]
    agreed = confident & np.isfinite(dist) & (dist <= AGREE_BODY)
    return int(confident.sum()), int(agreed.sum())


def pareto_front(figures: dict[str, dict[str, float | None]]) -> list[str]:
    """The names in ``figures`` (``{name: {metric: value}}``) that no other name beats on
    the :data:`FRONT_METRICS` figures, in ``figures``' order. A figure is compared only where
    every name has one (no left/right pair tracked: no swaps to compare); one name beats
    another when it is at least as good on each figure and better on one."""
    used = [
        (metric, higher)
        for metric, higher in FRONT_METRICS.items()
        if all(f.get(metric) is not None for f in figures.values())
    ]
    signed = {
        name: [f[metric] if higher else -f[metric] for metric, higher in used]
        for name, f in figures.items()
    }

    def beats(a: list[float], b: list[float]) -> bool:
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    return [
        name
        for name, mine in signed.items()
        if not any(beats(other, mine) for o, other in signed.items() if o != name)
    ]
