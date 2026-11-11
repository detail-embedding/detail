"""From similarity maps to placed points: candidates, and the two decodes (§26 "Patch matching",
steps 4 and 5). Private to the engine — a second backbone or a second readout changes one piece.

**Candidates.** The top local maxima of each map (true peaks, so a runner-up is a competing
*place* and not the shoulder of the best peak) and, for pair members, the top cells under greedy
suppression; each refined to sub-patch position by a soft-argmax over its 3 × 3 neighbourhood
(``REFINE_TAU``), then mapped back to frame pixels through the view.

**Decode.** ``per_point`` takes each target's best candidate. ``skeleton`` chooses jointly: the
midline as a tree with edge lengths learned from the clicks, pairs as two distinct peaks at least
a patch apart, and each pair held to its **side rule**:

- ``hard`` (the research's): "left" never lands on the side of "right" the clicks put it — the
  pair's *relative* order across the local body axis. Both may still sit on one side of the
  midline, which is right from the side, where a near and a far ear both overlap the body.
- ``midline``: each member on **its own side of the midline** — the polyline through the
  skeleton's unpaired points, judged at the segment nearest the point. Wrong from the side;
  what "left" means from above.
- ``auto`` (the default): per pair, from the clicks — ``midline`` when every example that shows
  the pair puts its members on opposite sides of the midline, else ``hard`` when the examples
  agree on the relative order, else no rule. ``soft`` keeps everything and subtracts a penalty;
  ``off`` ignores sides.

A rule is tried against each member's 5 best candidates, then (only if no combination keeps it)
their 10 best, then ``midline`` relaxes to ``hard``, and only then to the penalty.

**No front, no back.** Which image side "left" falls on is **learned from the clicks** (every
example that shows a pair must agree, else the pair gets no rule), and the local direction a
rule needs is derived from the **edges**: the midline polyline is the longest chain through
unpaired points, oriented by column order, and a pair's parent is the unpaired point both
members join. Any fixed direction works because the learned sign cancels it — so a skeleton
carries no head/tail marking (§26 "Skeletons").

The constants are the research's, fixed before the labels they were scored on, set on ViT-B and
never re-tuned for ViT-L (§26 *Open questions* 5); they are recorded on every run (``MATCHING``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from detail.core.skeleton import Skeleton

DECODES = ("skeleton", "per_point")
SIDE_RULES = ("auto", "midline", "hard", "soft", "off")

# Soft-argmax temperature over the 3x3 neighbourhood, in cosine units — chosen on the research's
# label-free geometry check (median error 0.25 patches; spread between estimators ~0.5 px).
REFINE_TAU = 0.01
# Skeleton-decode constants, fixed before any skeleton-dataset label was scored, never tuned on
# those labels:
SKEL_TOPK = 5  # candidate peaks per keypoint; the global maximum is always the first
# When no combination of a pair's SKEL_TOPK candidates keeps its side rule, the rule is tried
# again over this many before relaxing (one overhead mouse run, 300 frames: the relative rule was
# satisfiable at 5 candidates in 93-97% of frames, at 10 in 99-100%).
SKEL_TOPK_WIDE = 10
SKEL_TAU = 0.05  # cosine similarity per unit of evidence
SKEL_LEN_TOL = 0.25  # an edge's spread is at least this fraction of its mean length
SKEL_LEN_FLOOR = 0.02  # ... and at least this fraction of body scale
SKEL_SIDE_PENALTY = (
    4.0  # evidence a pair loses for "left" on the body's right (soft rule)
)

#: What a run records under ``matching:`` (§26): the constants that moved a point.
MATCHING = {
    "refine_tau": REFINE_TAU,
    "skel_tau": SKEL_TAU,
    "topk": SKEL_TOPK,
    "topk_wide": SKEL_TOPK_WIDE,
    "len_tol": SKEL_LEN_TOL,
    "len_floor": SKEL_LEN_FLOOR,
    "side_penalty": SKEL_SIDE_PENALTY,
}


# --------------------------------------------------------------------------- #
# The skeleton as the decode reads it
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecodePlan:
    """Names, edges and left/right pairs, plus each pair's ``(parent, axis point, direction)``.

    ``parent`` is the unpaired point both members of a pair attach to; ``axis point`` is the
    parent's unpaired neighbour toward the midline's start, so ``axis - parent`` is the local
    body axis. A parent with no neighbour that way uses one the other way, with
    ``direction = -1`` reversing the axis. The orientation is a convention, not anatomy: the
    side a rule enforces is learned from the clicks under the same convention, so it cancels.
    """

    names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    pairs: tuple[tuple[str, str], ...]  # (left, right)
    parents: dict = field(default_factory=dict)
    #: The midline the ``midline`` rule judges sides against: the longest chain through points
    #: in no pair, along the edges, oriented by column order. Empty when there is none.
    midline_path: tuple[str, ...] = ()

    @classmethod
    def from_skeleton(cls, skeleton: Skeleton) -> "DecodePlan":
        names = skeleton.names
        pairs = tuple(skeleton.pairs)
        paired = {k for p in pairs for k in p}
        path = _midline_path(names, skeleton.edges, paired)
        return cls(
            names=names,
            edges=skeleton.edges,
            pairs=pairs,
            parents=_pair_parents(names, skeleton.edges, pairs, path),
            midline_path=path,
        )

    @property
    def paired(self) -> set[str]:
        return {k for p in self.pairs for k in p}


def _unpaired_graph(edges, paired) -> dict[str, list[str]]:
    nbrs: dict[str, list[str]] = {}
    for a, b in edges:
        if a in paired or b in paired:
            continue
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    return nbrs


def _farthest(start: str, nbrs: dict, order: dict) -> tuple[str, dict]:
    """BFS from ``start``: the farthest node (ties by column order) and every node's parent."""
    prev = {start: None}
    depth = {start: 0}
    queue = [start]
    while queue:
        n = queue.pop(0)
        for m in sorted(nbrs.get(n, ()), key=order.__getitem__):
            if m not in prev:
                prev[m] = n
                depth[m] = depth[n] + 1
                queue.append(m)
    far = min(prev, key=lambda n: (-depth[n], order[n]))
    return far, prev


def _midline_path(names, edges, paired) -> tuple[str, ...]:
    """The longest chain through unpaired points along the edges (the tree's diameter, by two
    searches), from the end with the lower column index. Empty below two points."""
    order = {n: i for i, n in enumerate(names)}
    nbrs = _unpaired_graph(edges, paired)
    best: tuple[str, ...] = ()
    seen: set[str] = set()
    for start in sorted(nbrs, key=order.__getitem__):
        if start in seen:
            continue
        end_a, prev = _farthest(start, nbrs, order)
        seen.update(prev)
        end_b, prev = _farthest(end_a, nbrs, order)
        path = [end_b]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        if order[path[0]] > order[path[-1]]:
            path.reverse()
        if len(path) > len(best) or (
            len(path) == len(best) and best and order[path[0]] < order[best[0]]
        ):
            best = tuple(path)
    return best if len(best) >= 2 else ()


def _pair_parents(names, edges, pairs, midline_path) -> dict:
    paired = {k for p in pairs for k in p}
    nbrs: dict[str, set] = {}
    for a, b in edges:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    order = {n: i for i, n in enumerate(names)}
    # "toward the start" is fewer unpaired edges from the midline's first point; a point off
    # the midline's component falls back to column order.
    hops: dict[str, int] = {}
    if midline_path:
        hops[midline_path[0]] = 0
        queue = [midline_path[0]]
        while queue:
            n = queue.pop(0)
            for m in nbrs.get(n, ()):
                if m not in paired and m not in hops:
                    hops[m] = hops[n] + 1
                    queue.append(m)

    def rank(n: str) -> tuple[float, int]:
        return (hops.get(n, float("inf")), order[n])

    out = {}
    for left, right in pairs:
        common = (nbrs.get(left, set()) & nbrs.get(right, set())) - paired
        if not common:
            continue
        parent = min(common, key=order.__getitem__)
        mid = [n for n in nbrs.get(parent, set()) if n not in paired]
        head = [n for n in mid if rank(n) < rank(parent)]
        tail = [n for n in mid if rank(n) > rank(parent)]
        if head:
            out[(left, right)] = (parent, max(head, key=rank), +1)
        elif tail:
            out[(left, right)] = (parent, min(tail, key=rank), -1)
    return out


def midline_side(point, path_xy: list) -> int:
    """+1 / -1: which side of the midline polyline ``point`` is on, judged at the nearest
    segment, each segment pointing toward the path's start (the convention ``side_of`` uses).
    0 when the midline has fewer than two placed points, or the point is on it."""
    if len(path_xy) < 2:
        return 0
    p = np.asarray(point, dtype=float)
    best, best_d = 0, np.inf
    for front_pt, back_pt in zip(path_xy[:-1], path_xy[1:]):
        a, b = np.asarray(back_pt, float), np.asarray(front_pt, float)
        d = b - a
        length2 = float(d @ d)
        t = 0.0 if length2 == 0 else float(np.clip((p - a) @ d / length2, 0.0, 1.0))
        dist = float(np.hypot(*(p - (a + t * d))))
        if dist < best_d:
            c = _cross(d, p - a)
            best, best_d = (0 if c == 0 else (1 if c > 0 else -1)), dist
    return best


def _cross(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def side_of(left_xy, right_xy, parent_xy, axis_xy, direction: int) -> int:
    """+1 / -1: which side of the local body axis the left member is on, relative to the right."""
    axis = (np.asarray(axis_xy, float) - np.asarray(parent_xy, float)) * direction
    c = _cross(axis, np.asarray(left_xy, float) - np.asarray(right_xy, float))
    return 0 if c == 0 else (1 if c > 0 else -1)


@dataclass
class SkeletonStats:
    """Learned from the clicks: ``(a, b) -> (mean, sd)`` of each edge's length as a fraction of
    body scale; ``(left, right) -> +1/-1``, the side "left" falls on across the local axis
    (``sides``) and of the midline (``midline_sides``); and the rule each pair is held to.
    """

    edges: dict = field(default_factory=dict)
    sides: dict = field(default_factory=dict)
    midline_sides: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)  # pair -> "midline" | "hard" | "soft"


def learn_skeleton(
    clicks: list, plan: DecodePlan, side_rule: str = "auto"
) -> SkeletonStats:
    """Edge lengths, left/right sides and each pair's side rule from the examples' clicks.

    ``clicks`` is one ``({target: (x_px, y_px)}, body_scale_px)`` per example frame, visible
    points only. A side is set only when every example that shows the pair agrees on its sign:
    enforcing a coin flip would add swaps rather than remove them. ``auto`` holds a pair to
    ``midline`` when every example showing it puts the members on opposite sides of the
    midline (consistently), else to ``hard`` when the relative sign is consistent; an explicit
    ``midline`` uses the relative sign for a pair the examples do not show straddling.
    """
    if side_rule not in SIDE_RULES:
        raise ValueError(
            f"unknown side_rule {side_rule!r}; expected one of {SIDE_RULES}"
        )
    stats = SkeletonStats()
    lengths: dict = {}
    for pts, scale in clicks:
        for a, b in plan.edges:
            if a in pts and b in pts:
                d = np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) / scale
                lengths.setdefault((a, b), []).append(d)
    for edge, ls in lengths.items():
        mu = float(np.mean(ls))
        sd = float(np.std(ls)) if len(ls) > 1 else 0.0
        stats.edges[edge] = (mu, max(sd, SKEL_LEN_TOL * mu, SKEL_LEN_FLOOR))
    for pair, (parent, axis_kp, direction) in plan.parents.items():
        signs = [
            side_of(pts[pair[0]], pts[pair[1]], pts[parent], pts[axis_kp], direction)
            for pts, _ in clicks
            if all(k in pts for k in (*pair, parent, axis_kp))
        ]
        signs = [s for s in signs if s != 0]
        if signs and all(s == signs[0] for s in signs):
            stats.sides[pair] = signs[0]
    for pair in plan.pairs:
        own = []
        for pts, _ in clicks:
            path = [pts[n] for n in plan.midline_path if n in pts]
            if pair[0] in pts and pair[1] in pts and len(path) >= 2:
                sl = midline_side(pts[pair[0]], path)
                sr = midline_side(pts[pair[1]], path)
                own.append(sl if sl != 0 and sr == -sl else 0)
        if own and all(o != 0 and o == own[0] for o in own):
            stats.midline_sides[pair] = own[0]
    for pair in plan.pairs:
        if side_rule == "off":
            continue
        if side_rule == "auto":
            if pair in stats.midline_sides:
                stats.rules[pair] = "midline"
            elif pair in stats.sides:
                stats.rules[pair] = "hard"
        elif side_rule == "midline" and plan.midline_path:
            if pair not in stats.midline_sides and pair in stats.sides:
                stats.midline_sides[pair] = stats.sides[pair]
            if pair in stats.midline_sides:
                stats.rules[pair] = "midline"
        elif pair in stats.sides:  # "hard" / "soft", or "midline" with no midline
            stats.rules[pair] = "hard" if side_rule == "midline" else side_rule
    return stats


# --------------------------------------------------------------------------- #
# Score maps -> candidates
# --------------------------------------------------------------------------- #
def refine_peak(s: np.ndarray, gy: int, gx: int) -> tuple[float, float]:
    """Soft-argmax over the finite cells of the 3x3 neighbourhood of ``(gy, gx)``."""
    gh, gw = s.shape
    ys = slice(max(0, gy - 1), min(gh, gy + 2))
    xs = slice(max(0, gx - 1), min(gw, gx + 2))
    window = s[ys, xs]
    finite = np.isfinite(window)
    w = np.where(
        finite,
        np.exp((np.where(finite, window, 0) - s[gy, gx]) / REFINE_TAU),
        0.0,
    )
    yy, xx = np.mgrid[ys, xs]
    total = w.sum()
    return float((w * yy).sum() / total), float((w * xx).sum() / total)


def topk_peaks(score: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """The ``k`` highest local maxima (a finite cell >= each of its 8 neighbours).

    A runner-up is then a competing *place*, never the shoulder of the best peak — so the
    skeleton decode can move a midline point to a rival but cannot slide it along its own peak.
    """
    s = np.where(np.isfinite(score), score, -np.inf)
    pad = np.pad(s, 1, constant_values=-np.inf)
    h, w = s.shape
    neigh = np.max(
        [
            pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
        ],
        axis=0,
    )
    ys, xs = np.nonzero(np.isfinite(s) & (s >= neigh))
    order = np.argsort(s[ys, xs])[::-1][:k]
    return [(int(ys[i]), int(xs[i]), float(s[ys[i], xs[i]])) for i in order]


def topk_cells(
    score: np.ndarray, k: int, min_sep: float = 1.5
) -> list[tuple[int, int, float]]:
    """Greedy suppression: the best cells at least ``min_sep`` cells apart.

    Kept for pair members, where two close candidates let a blob that merged two ears be split.
    """
    order = np.argsort(np.where(np.isfinite(score), score, -np.inf).ravel())[::-1]
    picked: list[tuple[int, int, float]] = []
    for idx in order:
        if len(picked) >= k:
            break
        gy, gx = divmod(int(idx), score.shape[1])
        if not np.isfinite(score[gy, gx]):
            break
        if any((gy - py) ** 2 + (gx - px) ** 2 < min_sep**2 for py, px, _ in picked):
            continue
        picked.append((gy, gx, float(score[gy, gx])))
    return picked


def candidates(maps: dict, view) -> dict:
    """``{"nms": ..., "peaks": ...}``, each ``{target: [((x_px, y_px), score, (gy, gx)), ...]}``
    best first — the refined position, the cosine, and the cell it came from.

    A target whose map has no finite cell (nothing inside the mask) has no entry: no point.
    """
    out = {}
    # nms runs wide so a pair's side rule can widen its search (SKEL_TOPK_WIDE); every other
    # use reads its first SKEL_TOPK, exactly the research's candidates.
    for kind, pick, k_top in (
        ("nms", topk_cells, SKEL_TOPK_WIDE),
        ("peaks", topk_peaks, SKEL_TOPK),
    ):
        cands = {}
        for k, s in maps.items():
            if not np.isfinite(s).any():
                continue
            found = []
            for gy, gx, score in pick(s, k_top):
                ry, rx = refine_peak(s, gy, gx)
                found.append((view.cell_to_source(ry, rx), score, (gy, gx)))
            if found:
                cands[k] = found
        out[kind] = cands
    return out


def candidate_pool(cands: dict, plan: DecodePlan) -> dict:
    """What each target chooses among (and the detect stage stores): true peaks for a midline
    point, greedy-suppression cells for a pair member — the skeleton decode's own pool.
    """
    paired = plan.paired
    nms, peaks = cands["nms"], cands["peaks"]
    return {k: (peaks.get(k, c) if k not in paired else c) for k, c in nms.items()}


# --------------------------------------------------------------------------- #
# Decodes
# --------------------------------------------------------------------------- #
def decode_per_point(cands: dict) -> dict:
    """``{target: ((x_px, y_px), score, (gy, gx))}`` — each target's best candidate."""
    return {k: c[0] for k, c in cands["nms"].items()}


def _edge_term(stats: SkeletonStats, a: str, b: str, pa, pb, scale: float) -> float:
    st = stats.edges.get((a, b)) or stats.edges.get((b, a))
    if st is None:
        return 0.0
    z = (np.hypot(pa[0] - pb[0], pa[1] - pb[1]) / scale - st[0]) / st[1]
    return -0.5 * float(z * z)


def _unary(c) -> float:
    return c[1] / SKEL_TAU


def _decode_midline(cands: dict, mid: list[str], edges, stats, scale: float) -> dict:
    """Exact max-sum over each connected component of the midline tree."""
    nbrs = {k: [] for k in mid}
    for a, b in edges:
        if a in nbrs and b in nbrs:
            nbrs[a].append(b)
            nbrs[b].append(a)
    chosen, seen = {}, set()
    for root in mid:
        if root in seen:
            continue
        order, parent, stack = [], {root: None}, [root]
        while stack:
            n = stack.pop()
            seen.add(n)
            order.append(n)
            for m in nbrs[n]:
                if m not in parent:
                    parent[m] = n
                    stack.append(m)
        msg, best_child = {}, {}
        for n in reversed(order):  # leaves first
            total = np.array([_unary(c) for c in cands[n]])
            for m in nbrs[n]:
                if parent.get(m) != n:
                    continue
                folded = np.array(
                    [
                        [
                            msg[m][j]
                            + _edge_term(
                                stats, n, m, cands[n][i][0], cands[m][j][0], scale
                            )
                            for j in range(len(cands[m]))
                        ]
                        for i in range(len(cands[n]))
                    ]
                )
                total = total + folded.max(axis=1)
                for i in range(len(cands[n])):
                    best_child[(m, i)] = int(np.argmax(folded[i]))
            msg[n] = total
        pick = {root: int(np.argmax(msg[root]))}
        for n in order:  # root first, so every parent is picked before its children
            for m in nbrs[n]:
                if parent.get(m) == n:
                    pick[m] = best_child[(m, pick[n])]
        for n, i in pick.items():
            chosen[n] = cands[n][i]
    return chosen


#: The searches a pair's rule makes, in order, until one finds a combination that keeps it:
#: ``(constraint, candidates per member)``. ``penalty`` keeps everything and subtracts the soft
#: side penalty from a relative-order breach — the last resort, and the whole of ``soft``.
_PASSES = {
    "midline": (
        ("midline", SKEL_TOPK),
        ("midline", SKEL_TOPK_WIDE),
        ("hard", SKEL_TOPK_WIDE),
        ("penalty", SKEL_TOPK_WIDE),
    ),
    "hard": (
        ("hard", SKEL_TOPK),
        ("hard", SKEL_TOPK_WIDE),
        ("penalty", SKEL_TOPK_WIDE),
    ),
    "soft": (("penalty", SKEL_TOPK),),
    None: (("none", SKEL_TOPK),),
}


def decode_skeleton(
    cands: dict,
    plan: DecodePlan,
    stats: SkeletonStats,
    scale_px: float,
    patch_px: float,
) -> dict:
    """``{target: ((x_px, y_px), score, (gy, gx))}``, chosen jointly (the research's
    ``skeleton_peaks_*``).

    Midline points choose among their map's true local maxima; pair members among the
    greedy-suppression candidates. Each pair is decoded given the midline: both members'
    similarity, their edges to the parent, at least one patch apart (two places, never one), and
    the pair's side rule (``stats.rules``) — searched as ``_PASSES`` lists, widening and then
    relaxing only when nothing keeps it. With every skeleton term flat this is the per-point
    decode.
    """
    paired = plan.paired
    pool = candidate_pool(cands, plan)
    mid = [k for k in plan.names if k in pool and k not in paired]
    chosen = _decode_midline(
        {k: c[:SKEL_TOPK] for k, c in pool.items()}, mid, plan.edges, stats, scale_px
    )
    path_xy = [chosen[n][0] for n in plan.midline_path if n in chosen]
    for left, right in plan.pairs:
        cl, cr = pool.get(left, []), pool.get(right, [])
        info = plan.parents.get((left, right))
        par = chosen.get(info[0]) if info else None
        ax = chosen.get(info[1]) if info else None

        def anchor(k, c, info=info, par=par):
            if par is None:
                return 0.0
            return _edge_term(stats, k, info[0], c[0], par[0], scale_px)

        if not cl or not cr:
            for k, c in ((left, cl), (right, cr)):
                if c:
                    chosen[k] = max(
                        c[:SKEL_TOPK], key=lambda x, k=k: _unary(x) + anchor(k, x)
                    )
            continue
        pair = (left, right)
        rule = stats.rules.get(pair)
        want = stats.sides.get(pair)
        own = stats.midline_sides.get(pair)
        has_axis = want is not None and par is not None and ax is not None
        # Each candidate's side of the midline, once — not once per combination per pass.
        if rule == "midline" and len(path_xy) >= 2:
            side_l = [midline_side(a[0], path_xy) for a in cl]
            side_r = [midline_side(b[0], path_xy) for b in cr]
        else:
            side_l = side_r = None

        def breaks(kind, i, j):
            if kind == "midline":
                return side_l is None or not (side_l[i] == own and side_r[j] == -own)
            if kind in ("hard", "penalty") and has_axis:
                got = side_of(cl[i][0], cr[j][0], par[0], ax[0], info[2])
                return bool(got) and got != want
            return False

        best = None
        for kind, k_top in _PASSES[rule]:
            best_score = -np.inf
            for i, a in enumerate(cl[:k_top]):
                for j, b in enumerate(cr[:k_top]):
                    if np.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1]) < patch_px:
                        continue  # one place cannot be both sides
                    score = _unary(a) + _unary(b) + anchor(left, a) + anchor(right, b)
                    if breaks(kind, i, j):
                        if kind != "penalty":
                            continue
                        score -= SKEL_SIDE_PENALTY
                    if score > best_score:
                        best, best_score = (a, b), score
            if best is not None:
                break
        if best is None:  # every combination collapsed onto one place
            best = (cl[0], cr[0])
        chosen[left], chosen[right] = best
    for (
        k,
        c,
    ) in pool.items():  # anything the skeleton does not reach keeps its best peak
        chosen.setdefault(k, c[0])
    return chosen
