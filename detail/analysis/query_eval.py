"""Pure retrieval scoring for the behavior-query eval (§16).

Arrays/spans in, metrics out — no I/O, no experiment. A **span** is an inclusive
``(start_t_ns, end_t_ns)`` pair (Tenet 8); the query engine returns predicted spans
best-first (lowest score = best). Ground truth is the behavior's remaining intervals
(after held-out exemplars). Matching is greedy one-to-one by rank: **state** behaviors
match by temporal IoU, **point** behaviors (zero-length GT) by a ``t_ns`` tolerance
window. Metrics are COCO-style all-point AP plus precision/recall/best-F1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Span = tuple[int, int]


def _overlap_ns(a: Span, b: Span) -> int:
    """Inclusive temporal overlap of two spans in ns (<= 0 when disjoint)."""
    return min(a[1], b[1]) - max(a[0], b[0])


def temporal_iou(a: Span, b: Span) -> float:
    """Temporal IoU of two ``(start_t_ns, end_t_ns)`` spans (0 when disjoint)."""
    inter = _overlap_ns(a, b)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def seeded_order(n: int, seed: int) -> list[int]:
    """A deterministic permutation of ``range(n)`` — reproducible across runs."""
    return [int(i) for i in np.random.default_rng(seed).permutation(n)]


@dataclass
class MatchResult:
    """One prediction's match outcome (aligned with ``preds`` order).

    ``matched_gt`` indexes into the prediction's group's GT list when it is a TP,
    else ``None``. ``iou`` is set for a **state** hit, ``offset_ns`` for a **point**
    hit; the other stays ``None`` (both ``None`` on a false positive). The per-clip
    detail the eval persists (``predictions.csv``) is built from these.
    """

    is_tp: bool
    matched_gt: int | None
    iou: float | None
    offset_ns: int | None


def match_ranked_detailed(
    preds: list[tuple],
    gt_by_group: dict,
    *,
    mode: str,
    iou_threshold: float = 0.5,
    window_ns: int = 0,
) -> list[MatchResult]:
    """Greedy one-to-one match, returning per-prediction detail (which GT + quality).

    Same matching as ``match_ranked`` (each GT hit at most once, best-first), but it
    records *which* GT each TP consumed and the match quality (IoU for state, offset
    for point) so the eval can persist per-clip outcomes and mark unmatched GT as FN.
    """
    matched = {g: [False] * len(v) for g, v in gt_by_group.items()}
    out: list[MatchResult] = []
    for group, start, end, _score in preds:
        gts = gt_by_group.get(group, [])
        used = matched.get(group, [])
        if mode == "state":
            best_j, best = -1, 0.0
            for j, span in enumerate(gts):
                if used[j]:
                    continue
                iou = temporal_iou((start, end), span)
                if iou > best:
                    best, best_j = iou, j
            hit = best_j >= 0 and best >= iou_threshold
            if hit:
                used[best_j] = True
                out.append(MatchResult(True, best_j, best, None))
            else:
                out.append(MatchResult(False, None, None, None))
        else:  # point
            mid = (start + end) // 2
            best_j, best = -1, window_ns + 1
            for j, point in enumerate(gts):
                if used[j]:
                    continue
                dist = abs(mid - point)
                if dist < best:
                    best, best_j = dist, j
            hit = best_j >= 0 and best <= window_ns
            if hit:
                used[best_j] = True
                out.append(MatchResult(True, best_j, None, int(best)))
            else:
                out.append(MatchResult(False, None, None, None))
    return out


def match_ranked(
    preds: list[tuple],
    gt_by_group: dict,
    *,
    mode: str,
    iou_threshold: float = 0.5,
    window_ns: int = 0,
) -> list[bool]:
    """Greedy one-to-one match of globally-ranked predictions to per-group GT.

    ``preds`` are ``(group, start_t_ns, end_t_ns, score)`` best-first (lowest score
    first); the *group* is the subject (``(recording_id, track_id)``) so a prediction
    only matches GT in its own recording. ``gt_by_group`` is ``{group: [(start,end)]}``
    for ``mode="state"`` (temporal IoU) or ``{group: [point_t_ns]}`` for
    ``mode="point"`` (a ``±window_ns`` tolerance on the prediction midpoint). Returns
    a per-prediction TP flag aligned with ``preds`` (one-to-one, so each GT is hit
    at most once).
    """
    return [
        r.is_tp
        for r in match_ranked_detailed(
            preds,
            gt_by_group,
            mode=mode,
            iou_threshold=iou_threshold,
            window_ns=window_ns,
        )
    ]


def average_precision(
    tp_flags: list[bool], n_gt: int
) -> tuple[float, list[tuple[float, float]]]:
    """COCO-style all-point AP from a ranked TP/FP list + the raw PR points.

    Returns ``(ap, [(recall, precision), ...])``. ``ap`` is the area under the
    precision-interpolated (monotonic) PR step curve; 0 when there is no GT.
    """
    if n_gt == 0:
        return 0.0, []
    tp = fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for flag in tp_flags:
        if flag:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    envelope = list(precisions)
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])
    ap, prev_r = 0.0, 0.0
    for i, recall in enumerate(recalls):
        if recall > prev_r:
            ap += (recall - prev_r) * envelope[i]
            prev_r = recall
    return ap, list(zip(recalls, precisions))


# --- nulls: is a number real, or just the shape of the answers? ---------------------- #
# A metric with nothing to compare it against can't be read. A permissive matcher that
# covers a third of the recording scores a high recall whatever it retrieves — the
# recall that turned out to be no better than its null. So every
# headline ships with a deliberately-uninformative version of itself:
#   shifted-time null    — the same spans slid to the wrong places (destroys alignment,
#                          keeps count/length/coverage) → is recall better than coverage?
#   shuffled-rank null   — the same spans and outcomes in random order → do the scores
#                          carry any ranking information?
# `lift` is actual − null: the part that is real.

# Overlap levels the per-level breakdown reports for **state** behaviors. A constant, not
# a config knob: these are for reading the shape of a result, not for tuning.
OVERLAP_LEVELS = (0.1, 0.25, 0.5)
NULL_DRAWS = 20


@dataclass
class OverlapMetrics:
    """One behavior's span-level scores at one overlap level, each with its null."""

    iou: float
    n_gt: int
    n_pred: int
    tp: int
    recall: float
    recall_null: float
    recall_lift: float
    precision: float
    ap: float
    ap_null: float
    ap_lift: float


def _shift_spans(preds: list[tuple], bounds_by_group: dict, shift: int) -> list[tuple]:
    """Every prediction slid ``shift`` ns later within its own group's timeline, wrapping
    at the end. Lengths, count and total coverage are preserved — only the alignment with
    ground truth is destroyed."""
    out = []
    for group, start, end, score in preds:
        t_min, t_max = bounds_by_group.get(group, (start, end))
        span = max(t_max - t_min, 1)
        moved = (start - t_min + shift) % span + t_min
        out.append((group, moved, moved + (end - start), score))
    return out


def overlap_metrics(
    preds: list[tuple],
    gt_by_group: dict,
    bounds_by_group: dict,
    *,
    mode: str,
    levels: Sequence[float],
    window_ns: int = 0,
    n_null: int = NULL_DRAWS,
    seed: int = 0,
) -> list[OverlapMetrics]:
    """Recall / precision / AP at each overlap level, each with its null and lift.

    ``preds`` are ``(group, start_t_ns, end_t_ns, score)`` **best-first** (the same shape
    ``match_ranked`` takes); ``bounds_by_group`` gives each subject's ``(t_min, t_max)``
    for the shifted-time null, so a prediction wraps within its own recording rather than
    a pooled range. For ``mode="point"`` the level is not used for matching (the
    ``window_ns`` tolerance is), so pass a single level.
    """
    rng = np.random.default_rng(seed)
    n_gt = sum(len(v) for v in gt_by_group.values())
    out: list[OverlapMetrics] = []
    for level in levels:

        def _flags(rows: list[tuple], iou: float = level) -> list[bool]:
            return [
                r.is_tp
                for r in match_ranked_detailed(
                    rows,
                    gt_by_group,
                    mode=mode,
                    iou_threshold=iou,
                    window_ns=window_ns,
                )
            ]

        flags = _flags(preds)
        tp = sum(1 for f in flags if f)
        ap, _points = average_precision(flags, n_gt)
        recall = tp / n_gt if n_gt else 0.0
        precision = tp / len(flags) if flags else 0.0
        null_recalls, null_aps = [], []
        for _ in range(n_null):
            shift = int(rng.integers(1, max(2, _timeline_span(bounds_by_group))))
            shifted = _flags(_shift_spans(preds, bounds_by_group, shift))
            null_recalls.append((sum(1 for f in shifted if f) / n_gt) if n_gt else 0.0)
            shuffled = [flags[i] for i in rng.permutation(len(flags))]
            null_aps.append(average_precision(shuffled, n_gt)[0])
        recall_null = _mean(null_recalls)
        ap_null = _mean(null_aps)
        out.append(
            OverlapMetrics(
                iou=float(level),
                n_gt=n_gt,
                n_pred=len(flags),
                tp=tp,
                recall=recall,
                recall_null=recall_null,
                recall_lift=recall - recall_null,
                precision=precision,
                ap=ap,
                ap_null=ap_null,
                ap_lift=ap - ap_null,
            )
        )
    return out


def _timeline_span(bounds_by_group: dict) -> int:
    """The widest group timeline — the range random shift offsets are drawn from."""
    spans = [t_max - t_min for t_min, t_max in bounds_by_group.values()]
    return max(spans) if spans else 1


def aggregate_overlap(per_fold: list[list[OverlapMetrics]]) -> list[OverlapMetrics]:
    """Macro-average per-level metrics across folds — the same mean the headline uses, so
    a per-level row and the ``metrics.csv`` row at the same level agree."""
    by_level: dict[float, list[OverlapMetrics]] = {}
    for fold in per_fold:
        for m in fold:
            by_level.setdefault(m.iou, []).append(m)
    out = []
    for level in sorted(by_level):
        rows = by_level[level]
        out.append(
            OverlapMetrics(
                iou=level,
                n_gt=round(_mean([m.n_gt for m in rows])),
                n_pred=round(_mean([m.n_pred for m in rows])),
                tp=round(_mean([m.tp for m in rows])),
                recall=_mean([m.recall for m in rows]),
                recall_null=_mean([m.recall_null for m in rows]),
                recall_lift=_mean([m.recall_lift for m in rows]),
                precision=_mean([m.precision for m in rows]),
                ap=_mean([m.ap for m in rows]),
                ap_null=_mean([m.ap_null for m in rows]),
                ap_lift=_mean([m.ap_lift for m in rows]),
            )
        )
    return out


@dataclass
class EvalMetrics:
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    ap: float
    precision: float
    recall: float
    best_f1: float
    best_f1_threshold: float
    pr_points: list[tuple[float, float]]


@dataclass
class AggregateMetrics:
    """Macro aggregate of a behavior's per-fold metrics (§16 cross-validation).

    Point estimates (``ap`` … ``best_f1_threshold``) are the **mean** across folds;
    ``*_std`` are the population standard deviations of the key metrics (0 for a single
    fold, so a ``folds=1`` run reduces to the old single-split numbers). Counts are the
    rounded per-fold means, with ``fp``/``fn`` derived from ``tp`` + the totals so the
    row stays internally consistent (``tp+fp==n_pred``, ``tp+fn==n_gt``)."""

    n_folds: int
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    ap: float
    precision: float
    recall: float
    best_f1: float
    best_f1_threshold: float
    ap_std: float
    precision_std: float
    recall_std: float
    best_f1_std: float


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pop_std(xs: list[float]) -> float:
    """Population standard deviation (0 for fewer than two values)."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def aggregate_folds(per_fold: list[EvalMetrics]) -> AggregateMetrics:
    """Fold each fold's ``EvalMetrics`` into a macro mean±std summary (§16)."""
    ap = [m.ap for m in per_fold]
    precision = [m.precision for m in per_fold]
    recall = [m.recall for m in per_fold]
    best_f1 = [m.best_f1 for m in per_fold]
    # Derive fp/fn from tp + the totals so independent rounding can't produce a
    # self-contradictory row (tp+fp==n_pred, tp+fn==n_gt); folds=1 stays exact.
    n_gt = round(_mean([m.n_gt for m in per_fold]))
    n_pred = round(_mean([m.n_pred for m in per_fold]))
    tp = round(_mean([m.tp for m in per_fold]))
    return AggregateMetrics(
        n_folds=len(per_fold),
        n_gt=n_gt,
        n_pred=n_pred,
        tp=tp,
        fp=n_pred - tp,
        fn=n_gt - tp,
        ap=_mean(ap),
        precision=_mean(precision),
        recall=_mean(recall),
        best_f1=_mean(best_f1),
        best_f1_threshold=_mean([m.best_f1_threshold for m in per_fold]),
        ap_std=_pop_std(ap),
        precision_std=_pop_std(precision),
        recall_std=_pop_std(recall),
        best_f1_std=_pop_std(best_f1),
    )


def perframe_metrics(scores: np.ndarray, labels: np.ndarray) -> EvalMetrics:
    """``EvalMetrics`` for a per-frame ranked list (scores + boolean positive labels).

    The vectorized twin of :func:`summarize` for the per-frame classification path, where
    the ranked list is *every scored frame* (tens of thousands) rather than a handful of
    predicted spans. Ranks frames by score (higher = more positive) and integrates the same
    COCO all-point AP. Two things differ from a naive rank sweep, both deliberate:

    - **Ties are grouped.** A per-frame classifier saturates (``soft_knn`` votes to exactly
      0 or 1; ``frame_scores`` hits the cosine ceiling), so exact-tie groups are large and
      positives cluster temporally — an order-dependent AP would swing with where a positive
      lands inside its tie block. One PR point is taken per *distinct* score, counting every
      frame at that score together, so the AP is order-independent (matching ``sklearn``).
    - **The confusion block is at the best-F1 operating point**, not predict-all. The full
      ranking's end always has ``recall==1``/``fn==0`` (every positive appears somewhere), so
      reporting there would print a degenerate "perfect recall" in the tables. Instead
      ``precision``/``recall``/``tp``/``fp``/``fn``/``n_pred`` all describe the one threshold
      where F1 peaks — the same meaning the windowed path's ``summarize`` reports. ``ap`` and
      the PR curve stay threshold-free. ``pr_points`` is thinned to at most 512 points.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_all = int(scores.shape[0])
    n_gt = int(labels.sum())
    if n_all == 0 or n_gt == 0:
        return EvalMetrics(n_gt, n_all, 0, n_all, n_gt, 0.0, 0.0, 0.0, 0.0, 0.0, [])
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    lab = labels[order]
    tp_cum = np.cumsum(lab)
    fp_cum = np.cumsum(~lab)
    # One PR point per distinct score, with the cumulative counts at the END of that score's
    # group (``last`` = last index of each group), so tied frames are counted together.
    last = np.concatenate((np.nonzero(np.diff(s))[0], [n_all - 1]))
    tp = tp_cum[last].astype(np.float64)
    fp = fp_cum[last].astype(np.float64)
    thr = s[last]
    precision = tp / (tp + fp)
    recall = tp / n_gt
    # All-point interpolation: envelope = running max of precision scanned from the right.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    dr = np.diff(np.concatenate(([0.0], recall)))  # recall increment per distinct score
    ap = float((dr * envelope).sum())
    denom = precision + recall
    # Masked divide: a leading false positive gives precision=recall=0, so divide only
    # where the denominator is positive (a bare ``a/denom`` warns on 0/0 → fails the suite).
    f1 = np.zeros_like(denom)
    np.divide(2 * precision * recall, denom, out=f1, where=denom > 0)
    best = int(np.argmax(f1))
    tp_best, fp_best = int(tp[best]), int(fp[best])
    n_thr = len(thr)
    if n_thr > 512:  # thin the stored curve; the figure needs ~hundreds of points
        sel = np.unique(np.linspace(0, n_thr - 1, 512).round().astype(int))
    else:
        sel = np.arange(n_thr)
    pr_points = list(zip(recall[sel].tolist(), precision[sel].tolist()))
    return EvalMetrics(
        n_gt=n_gt,
        n_pred=tp_best
        + fp_best,  # frames flagged at the operating point (tp+fp==n_pred)
        tp=tp_best,
        fp=fp_best,
        fn=n_gt - tp_best,
        ap=ap,
        precision=float(precision[best]),
        recall=float(recall[best]),
        best_f1=float(f1[best]),
        best_f1_threshold=float(thr[best]),
        pr_points=pr_points,
    )


def perframe_null(
    per_subject: list[tuple],
    metrics: EvalMetrics,
    level: float,
    *,
    n_null: int = NULL_DRAWS,
    seed: int = 0,
) -> OverlapMetrics:
    """A per-frame ranking's floor: the same scores, slid so they point at the wrong frames.

    ``per_subject`` is ``[(scores, labels), ...]`` — one entry per searched subject, in
    ``t_ns`` order.

    The floor has to destroy the *alignment* between score and label while keeping everything
    else about the score series, which is what the span path's ``_shift_spans`` does for
    predictions. A circular shift per subject does exactly that here: same values, same
    distribution, same autocorrelation, wrong times.

    The obvious analytic alternative — expected AP of an i.i.d. permutation, i.e. the
    prevalence — is **wrong for this data and wrong in the optimistic direction.** These
    embeddings are strongly autocorrelated and the behaviors are contiguous bouts, so a score
    that is merely smooth in time clears a prevalence floor while carrying no signal at all:
    a zero-signal smoothed scorer can beat prevalence by a wide margin. A floor the
    degenerate answer beats is worse than no floor (the principle
    ``cluster_metrics.one_cluster_floor`` already applies on the clustering side).

    Emitted through ``OverlapMetrics`` so ``null_at`` and every CSV surface read the floor by
    one code path; ``level`` is the configured ``iou_threshold``, carried purely as that
    lookup key — a per-frame score has no overlap level.
    """
    rng = np.random.default_rng(seed)
    usable = [(np.asarray(sc), np.asarray(lb, dtype=bool)) for sc, lb in per_subject]
    usable = [(sc, lb) for sc, lb in usable if sc.size]
    aps: list[float] = []
    recalls: list[float] = []
    for _ in range(n_null if usable else 0):
        shifted, labels = [], []
        for sc, lb in usable:
            # A shift of 0 or len would re-align it; draw strictly inside.
            shifted.append(
                np.roll(sc, int(rng.integers(1, sc.size)) if sc.size > 1 else 0)
            )
            labels.append(lb)
        drawn = perframe_metrics(np.concatenate(shifted), np.concatenate(labels))
        aps.append(drawn.ap)
        recalls.append(drawn.recall)
    ap_null = float(np.mean(aps)) if aps else 0.0
    recall_null = float(np.mean(recalls)) if recalls else 0.0
    return OverlapMetrics(
        iou=level,
        n_gt=metrics.n_gt,
        n_pred=metrics.n_pred,
        tp=metrics.tp,
        recall=metrics.recall,
        recall_null=recall_null,
        recall_lift=metrics.recall - recall_null,
        precision=metrics.precision,
        ap=metrics.ap,
        ap_null=ap_null,
        ap_lift=metrics.ap - ap_null,
    )


def summarize(tp_flags: list[bool], pred_scores: list[float], n_gt: int) -> EvalMetrics:
    """Fold a ranked TP/FP list (+ its scores) into ``EvalMetrics``.

    Matching is one-to-one, so ``fn = n_gt - tp``. ``best_f1_threshold`` is the
    prediction score at the rank maximizing F1 (the sweep is over the score cutoff).
    """
    n_pred = len(tp_flags)
    tp = sum(1 for f in tp_flags if f)
    fp = n_pred - tp
    fn = max(0, n_gt - tp)
    ap, pr_points = average_precision(tp_flags, n_gt)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    best_f1, best_thr = 0.0, 0.0
    c_tp = c_fp = 0
    for i, flag in enumerate(tp_flags):
        if flag:
            c_tp += 1
        else:
            c_fp += 1
        p = c_tp / (c_tp + c_fp)
        r = c_tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, pred_scores[i]
    return EvalMetrics(
        n_gt, n_pred, tp, fp, fn, ap, precision, recall, best_f1, best_thr, pr_points
    )
