"""Behavior-query eval — orchestration + the synchronous scripting entry (§16).

``evaluate_query(experiment_dir, config)`` runs the behavior-query engine (the pure
``analysis/clip_query.query_tracks``) with **held-out** ground-truth exemplars and scores
the ranked matches against the remaining ground-truth intervals (``analysis/query_eval``),
one result per behavior. It was the no-executor twin of the removed ``clip_query.query_clips``;
the CLI and a scripting loop both call it, and a later recorded ``AnalysisModule`` will reuse
the same core. Ground truth is the `behavior`-role stream (§3): a recording is eligible for a
behavior only if it declared that behavior (``is_annotated_for``); matching is by the behavior's
point/state type.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np

from detail.analysis import behavior_labels
from detail.analysis import clip_query as core
from detail.analysis import query_eval as scoring
from detail.analysis import retrieval
from detail.configs.query_eval import _PERFRAME_METHODS, QueryEvalConfig
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.utils import check_track_is_unambiguous, utc_timestamp
from detail.executor.runners import AnalysisModule, register_analysis_module
from detail.storage import analysis_runs as analysis_store
from detail.storage import query_eval_csv

_PERMISSIVE_THRESHOLD = 2.0  # max cosine distance — nothing filtered (§16)
_NS_PER_SECOND = 1_000_000_000

_log = logging.getLogger(__name__)


@dataclass
class BehaviorEval:
    behavior: str
    behavior_type: str
    n_recordings: int
    # Aggregate (mean±std) across the run's cross-validation folds; None when skipped.
    metrics: scoring.AggregateMetrics | None
    skipped: str | None = None
    # Per-fold metrics (index = fold) behind the aggregate — written to folds.csv.
    fold_metrics: list[scoring.EvalMetrics] = field(default_factory=list)
    # Scores at each overlap level, each beside its null — written to overlap.csv. The
    # headline in ``metrics`` is one level; this is the shape across levels, which is
    # where a matcher that finds the right moments at the wrong extent shows up.
    overlap: list[scoring.OverlapMetrics] = field(default_factory=list)
    # Per-clip detail across all folds (empty for a skipped behavior) the recorded run
    # persists as predictions.csv / ground_truth.csv and the GUI reads for the explorer.
    predictions: list[query_eval_csv.PredictionRow] = field(default_factory=list)
    ground_truth: list[query_eval_csv.GroundTruthRow] = field(default_factory=list)
    # Fold-0 per-frame similarity for the display recording (recording_id, t_ns, score) — the fixed
    # split the similarity figure uses. None when skipped/cancelled.
    similarity: tuple[str, np.ndarray, np.ndarray] | None = None


def _fold_similarity(db, subject, queries, cfg, max_gap_ns):
    """The display subject's per-frame similarity to the fold's exemplars (for the figure).

    The raw ``frame_scores`` cosine curve (higher = more example-like) — a reference the
    similarity figure overlays for the windowed matchers, whose own cost is per window not
    per frame. The per-frame methods persist their actual score curve instead. ``None`` if
    the subject isn't loaded.
    """
    if subject not in db:
        return None
    emb, _frame_idx, t_ns = db[subject]
    query_embs = [core._resolve_example(db, q, max_gap_ns)[1] for q in queries]
    if not query_embs:
        return None
    score, _ = retrieval.frame_scores(emb, query_embs, fusion=cfg.fusion)
    return (subject[0], np.asarray(t_ns), np.asarray(score, dtype=np.float32))


def _interval_mask(
    t_ns: np.ndarray, intervals: list[tuple[int, int]], btype: str, window_ns: int
) -> np.ndarray:
    """Boolean per-frame mask: which rows fall inside the ground-truth ``intervals``.

    State behaviors mark a frame when its ``t_ns`` lies in a scored ``[start, end]``; point
    behaviors mark it within ``±window_ns`` of the point (the interval's start), mirroring
    the span path's point-mode match. This is the per-frame ground-truth label.
    """
    mask = np.zeros(len(t_ns), dtype=bool)
    t = np.asarray(t_ns, dtype=np.int64)
    if btype == "point":
        for start, _end in intervals:
            mask |= np.abs(t - start) <= window_ns
    else:
        for start, end in intervals:
            mask |= (t >= start) & (t <= end)
    return mask


# The per-frame score crosses the operating threshold frequently, so raw same-outcome runs
# number in the thousands and render as an unreadable hatch. Bridging same-outcome runs that
# sit within this gap collapses the flicker into readable segments; it never crosses the other
# outcome (within a GT interval every above-threshold frame is a TP, between GT every one is an
# FP), so the green-on-GT / red-off-GT picture is unchanged. Readability only.
_RUN_BRIDGE_NS = 300_000_000  # 0.3 s


def _bool_runs(mask: np.ndarray, t_ns: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of ``mask`` as inclusive ``(start_idx, end_idx)`` pairs,
    with runs whose time gap is ``<= _RUN_BRIDGE_NS`` merged (see the note above)."""
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    runs: list[tuple[int, int]] = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        if runs and t_ns[a] - t_ns[runs[-1][1]] <= _RUN_BRIDGE_NS:
            runs[-1] = (runs[-1][0], b)  # bridge a brief same-outcome dip
        else:
            runs.append((a, b))
    return runs


def _perframe_detail_rows(
    behavior,
    btype,
    subject,
    t_ns,
    score,
    scored_intervals,
    exemplar_intervals,
    threshold,
    window_ns,
    fold=0,
):
    """Fold-0 display-subject detail for the matches/similarity figures (§16).

    Mirrors the per-frame AP's confusion at the best-F1 operating point (``threshold``): a
    frame is **predicted positive** when its score is ``>= threshold`` and **actually
    positive** under the *same* ``_interval_mask`` the metric labels with, so an
    above-threshold frame is a TP where it lands on a scored GT interval and an FP where it
    does not. Emitted as contiguous same-outcome runs (green/red spans), plus the GT
    intervals (scored, matched/fn) and the held-out exemplars. Drawing whole above-threshold
    *spans* as one TP (the old approach) painted a weak classifier that flags everything as a
    solid green bar; coloring per frame against the GT mask instead shows it honestly (mostly
    red off the GT). Visualization only; the AP is threshold-free and unchanged.
    """
    sid, tid = subject
    t = np.asarray(t_ns, dtype=np.int64)
    sc = np.asarray(score, dtype=np.float64)
    gt_mask = _interval_mask(
        t, scored_intervals, btype, window_ns
    )  # the AP's positive labels
    predicted = sc >= threshold

    predictions: list[query_eval_csv.PredictionRow] = []
    for outcome, mask in (("tp", predicted & gt_mask), ("fp", predicted & ~gt_mask)):
        for a, b in _bool_runs(mask, t):
            mid = int((t[a] + t[b]) // 2)
            predictions.append(
                query_eval_csv.PredictionRow(
                    behavior=behavior,
                    recording_id=sid,
                    track_id=tid,
                    rank=len(predictions) + 1,
                    outcome=outcome,
                    score=float(sc[a : b + 1].max()),
                    start_t_ns=int(t[a]),
                    end_t_ns=int(t[b]),
                    iou=None,  # per-frame runs are not span-IoU matched
                    offset_ns=(
                        min(abs(mid - iv[0]) for iv in scored_intervals)
                        if outcome == "tp" and btype == "point" and scored_intervals
                        else None
                    ),
                    matched_gt_start_t_ns=None,
                    matched_gt_end_t_ns=None,
                    fold=fold,
                )
            )
    # A scored interval is "matched" when it contains at least one predicted-positive frame.
    matched: set[int] = {
        gi
        for gi, iv in enumerate(scored_intervals)
        if (predicted & _interval_mask(t, [iv], btype, window_ns)).any()
    }
    ground_truth: list[query_eval_csv.GroundTruthRow] = [
        query_eval_csv.GroundTruthRow(
            behavior=behavior,
            recording_id=sid,
            track_id=tid,
            start_t_ns=s,
            end_t_ns=e,
            role="scored",
            outcome="matched" if gi in matched else "fn",
            fold=fold,
        )
        for gi, (s, e) in enumerate(scored_intervals)
    ]
    ground_truth += [
        query_eval_csv.GroundTruthRow(
            behavior=behavior,
            recording_id=sid,
            track_id=tid,
            start_t_ns=s,
            end_t_ns=e,
            role="exemplar",
            outcome="",
            fold=fold,
        )
        for (s, e) in exemplar_intervals
    ]
    return predictions, ground_truth


def _perframe_fold(
    behavior,
    db,
    search,
    exemplars,
    remaining,
    holdout,
    cfg,
    btype,
    window_ns,
    max_gap_ns,
):
    """One cross-validation fold scored as a per-frame classification (§16 per-frame AP).

    Scores every trajectory frame of the searched subjects with the chosen per-frame method,
    labels each by ground-truth membership (state: inside a ``remaining`` interval; point:
    within ``±point_window`` of a ``remaining`` point), and folds the ranked list into
    ``EvalMetrics`` via the shared per-frame Average Precision. The **exemplar** frames are the
    positive index; the **holdout** frames (the exemplars, plus any instance held out of scoring
    without being queried) are the only frames excluded from the scored test set, so a positive
    can never match itself and an unqueried held-out instance is never counted as a negative —
    it is a true positive the run has simply chosen not to score. ``soft_knn``'s sampled
    background negatives are the negative index but stay in the test set (a true negative that
    self-ranks low). Subjects
    are streamed (one trajectory resident at a time), not all materialized. Returns
    ``(EvalMetrics, null, similarity, predictions, ground_truth)`` where ``null`` is the
    per-frame floor (``scoring.perframe_null``, a circular-shift null) in the span path's shape,
    similarity is the display subject's ``(recording_id, t_ns, score)`` curve (or ``None``)
    and the detail rows are the display subject's GT + operating-point matches for the figures.
    """
    empty_metrics = scoring.perframe_metrics(np.array([]), np.array([], dtype=bool))
    empty = (
        empty_metrics,
        scoring.perframe_null([], empty_metrics, cfg.iou_threshold),
        None,
        [],
        [],
    )
    queries = [
        {"recording_id": sid, "track_id": tid, "start_t_ns": s, "end_t_ns": e}
        for ((sid, tid), s, e) in exemplars
    ]
    pos_examples: list[np.ndarray] = []
    for q in queries:
        try:
            pos_examples.append(core._resolve_example(db, q, max_gap_ns)[1])
        except (KeyError, ValueError):
            continue
    if not pos_examples:
        return empty

    remaining_by_subject: dict = {}
    for subject, s, e in remaining:
        remaining_by_subject.setdefault(subject, []).append((s, e))
    # Keyed on the *holdout*, not the exemplars: everything removed from the scored set leaves
    # the test set together, or an instance held out but not queried would rank as a negative.
    holdout_by_subject: dict = {}
    for subject, s, e in holdout:
        holdout_by_subject.setdefault(subject, []).append((s, e))

    pos_frames = np.vstack(pos_examples)
    display = search[0] if search else None
    dim = pos_frames.shape[1]

    def _load(subject):
        """Load one subject and its per-frame masks. Streams (one trajectory resident)."""
        emb, _frame_idx, t_ns = db[subject]
        t_ns = np.asarray(t_ns)
        pos_mask = _interval_mask(
            t_ns, remaining_by_subject.get(subject, []), btype, window_ns
        )
        excl_mask = _interval_mask(
            t_ns, holdout_by_subject.get(subject, []), btype, window_ns
        )
        return np.asarray(emb, dtype=np.float32), t_ns, pos_mask, excl_mask

    # soft_knn negative index: sample `n_negatives` background frames uniformly across all
    # subjects by a streaming reservoir (seeded), so the full trajectories are never all
    # resident at once — the span path streams too, and materializing all of them was a
    # memory regression. The negatives are the neighbor INDEX only; they stay in the scored
    # test set (a background frame that is also a sampled negative is a legitimate true
    # negative and self-ranks low — excluding them let a large `n_negatives` fake AP=1).
    neg_frames = np.empty((0, dim), dtype=np.float32)
    n_skipped = 0
    loaded_subjects: list = (
        []
    )  # subjects that read OK, in stream order (reused for scoring)
    rng = np.random.default_rng(cfg.seed)
    n_neg = int(cfg.n_negatives)
    buf_emb = buf_key = None  # the running reservoir: the n_neg largest-key bg rows
    for subject in search:
        try:
            emb, _t, pos_mask, excl_mask = _load(subject)
        except (KeyError, ValueError, FileNotFoundError):
            n_skipped += 1
            _log.warning(
                "query-eval %r: subject %r unreadable, skipped", behavior, subject
            )
            continue
        loaded_subjects.append(subject)
        # `unlabeled` is what a user's session has: everything not held out, including the
        # target behavior they are still looking for. `labeled_clean` removes the positives
        # too, which needs the ground truth — it scores a retrieval no user can reproduce,
        # and is kept only so runs recorded before this field existed can still be read.
        if cfg.negative_pool == "unlabeled":
            bg = emb[~excl_mask]
        else:
            bg = emb[~pos_mask & ~excl_mask]
        if not len(bg):
            continue
        keys = rng.random(len(bg))
        buf_emb = bg if buf_emb is None else np.vstack([buf_emb, bg])
        buf_key = keys if buf_key is None else np.concatenate([buf_key, keys])
        if len(buf_key) > n_neg:  # keep the n_neg largest keys = uniform sample
            keep = np.argpartition(-buf_key, n_neg - 1)[:n_neg]
            buf_emb, buf_key = buf_emb[keep], buf_key[keep]
    if buf_emb is not None:
        neg_frames = np.asarray(buf_emb, dtype=np.float32)

    # Scoring pass: stream each loaded subject, keep only its per-frame score+label slices.
    scores_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    similarity = None
    for subject in loaded_subjects:
        try:
            emb, t_ns, pos_mask, excl_mask = _load(subject)
        except (KeyError, ValueError, FileNotFoundError):
            n_skipped += 1
            _log.warning(
                "query-eval %r: subject %r unreadable, skipped", behavior, subject
            )
            continue
        score = retrieval.soft_knn_frame_scores(
            emb, pos_frames, neg_frames, k=cfg.k, tau=cfg.tau
        )
        test = ~excl_mask  # only the held-out exemplar frames leave the test set
        scores_parts.append(score[test])
        labels_parts.append(pos_mask[test])
        if similarity is None and subject == display:
            similarity = (subject[0], t_ns, np.asarray(score, dtype=np.float32))
    if not scores_parts:
        if n_skipped:
            _log.warning(
                "query-eval %r: all %d subject(s) unreadable — behavior not scored",
                behavior,
                n_skipped,
            )
        return empty

    scores = np.concatenate(scores_parts)
    labels = np.concatenate(labels_parts)
    metrics = scoring.perframe_metrics(scores, labels)
    # Per subject, not the concatenation: the null circularly shifts each subject's score
    # series within itself, so a shift can never slide one recording's scores onto another's
    # labels.
    null = scoring.perframe_null(
        list(zip(scores_parts, labels_parts)),
        metrics,
        cfg.iou_threshold,
        seed=cfg.seed,
    )
    predictions: list[query_eval_csv.PredictionRow] = []
    ground_truth: list[query_eval_csv.GroundTruthRow] = []
    if similarity is not None:  # the display subject loaded → detail for the figures
        _sid, disp_t_ns, disp_score = similarity
        predictions, ground_truth = _perframe_detail_rows(
            behavior,
            btype,
            display,
            disp_t_ns,
            disp_score,
            remaining_by_subject.get(display, []),
            holdout_by_subject.get(display, []),
            metrics.best_f1_threshold,
            window_ns,
            fold=0,
        )
    return metrics, null, similarity, predictions, ground_truth


def evaluate_query(
    experiment_dir: Path | str,
    config: dict,
    progress: Callable[[str, float, str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> list[BehaviorEval]:
    """Evaluate one embedding run + query config against behavior ground truth.

    Returns one ``BehaviorEval`` per requested behavior (``metrics=None`` +
    ``skipped`` when a behavior has too little ground truth or no eligible recording).

    The query engine loads each track's embeddings (from zarr, often over a network
    share) and runs a sliding-window similarity search per behavior — long enough that
    a silent run is indistinguishable from a hang. ``progress`` (``(stage, fraction,
    detail)`` — the CLI's ``_StageProgress`` shape) reports the phase (``loading`` →
    ``querying`` → ``scoring``) and overall fraction across the requested behaviors.

    Raises:
        ValueError: On a malformed config, an unknown/incomparable embedding run,
            or no recordings to evaluate.
    """
    from detail.api import run_preview
    from detail.api.recording import Recording
    from detail.storage import experiment as experiment_store

    cfg = QueryEvalConfig.model_validate(config)  # incl. the LOO ⇒ n_exemplars=1 rule
    experiment_root = Path(experiment_dir)
    if not cfg.behaviors:
        raise ValueError("at least one behavior is required")
    recording_ids = list(
        cfg.source_recording_ids
    ) or experiment_store.discover_recordings(experiment_root)
    if not recording_ids:
        raise ValueError("no recordings found to evaluate")

    recordings = {sid: Recording(experiment_root, sid) for sid in recording_ids}
    databases = run_preview.LazyTrajectories(
        experiment_root, cfg.embed_run_id
    )  # cached per behavior below
    band_ns = round(cfg.band_s * _NS_PER_SECOND)
    max_gap_ns = (
        round(cfg.trajectory_split_gap_s * _NS_PER_SECOND)
        if cfg.trajectory_split_gap_s is not None
        else None
    )
    window_ns = round(cfg.point_window_s * _NS_PER_SECOND)

    n_behaviors = len(cfg.behaviors)

    def report(stage: str, done: float, detail: str) -> None:
        # `done` is a fractional behavior index (0..n_behaviors) → overall fraction.
        if progress is not None:
            progress(stage, min(done / n_behaviors, 1.0), detail)

    results: list[BehaviorEval] = []
    for i, behavior in enumerate(cfg.behaviors):
        if stop_check is not None and stop_check():
            break
        report("loading", i, f"{behavior} · finding eligible recordings")
        eligible = [
            sid
            for sid, sess in recordings.items()
            if sess.is_annotated_for(behavior, cfg.stream_name)
            and cfg.track_id in _track_ids(run_preview, sess, cfg.embed_run_id)
        ]
        btype = (
            recordings[eligible[0]]
            .behavior_types(cfg.stream_name)
            .get(behavior, "state")
            if eligible
            else "state"
        )
        if not eligible:
            results.append(
                BehaviorEval(behavior, btype, 0, None, "no eligible recordings")
            )
            continue
        run_preview.validate_comparability(
            [(recordings[sid].dir, cfg.embed_run_id) for sid in eligible]
        )
        # Cross-validation reuses the same searched subjects across every fold; the lazy
        # store reloads a trajectory on each access, so cache it for a multi-fold run —
        # scoped to *this* behavior so the cache never holds every behavior's subjects at
        # once (the OOM the lazy store avoids). A single split keeps the lazy path.
        db = _CachingTrajectories(databases) if cfg.folds != 1 else databases
        pooled = [
            ((sid, cfg.track_id), s, e)
            for sid in eligible
            for (s, e) in _behavior_intervals(
                recordings[sid], behavior, cfg.stream_name
            )
        ]
        n_hold = cfg.n_holdout if cfg.n_holdout is not None else cfg.n_exemplars
        if len(pooled) <= n_hold:
            results.append(
                BehaviorEval(
                    behavior,
                    btype,
                    len(eligible),
                    None,
                    f"only {len(pooled)} ground-truth interval(s) <= "
                    + (
                        f"n_holdout={n_hold}"
                        if cfg.n_holdout is not None
                        else f"n_exemplars={n_hold}"
                    ),
                )
            )
            continue
        report("loading", i + 0.3, f"{behavior} · loading trajectory + selecting folds")
        folds, k_max = _make_folds(pooled, db, cfg, max_gap_ns)
        if not folds:
            results.append(
                BehaviorEval(
                    behavior,
                    btype,
                    len(eligible),
                    None,
                    "could not resolve enough exemplars within continuous trajectory",
                )
            )
            continue
        if cfg.folds > 0 and cfg.folds > k_max:
            _log.warning(
                "behavior %r: %d folds requested but only %d fit n_exemplars=%d over "
                "the resolvable instances; using %d",
                behavior,
                cfg.folds,
                k_max,
                cfg.n_exemplars,
                k_max,
            )

        search = [(sid, cfg.track_id) for sid in eligible]
        k = len(folds)
        fold_metrics: list[scoring.EvalMetrics] = []
        fold_overlap: list[list[scoring.OverlapMetrics]] = []
        predictions: list[query_eval_csv.PredictionRow] = []
        ground_truth: list[query_eval_csv.GroundTruthRow] = []
        similarity = None
        for j, (exemplars, remaining, holdout) in enumerate(folds):
            if stop_check is not None and stop_check():
                break
            queries = [
                {"recording_id": sid, "track_id": tid, "start_t_ns": s, "end_t_ns": e}
                for ((sid, tid), s, e) in exemplars
            ]

            def query_progress(
                done, total, found, phase, _i=i, _j=j, _k=k, _b=behavior
            ):
                # Fold the query's per-subject progress into (fold × subject) within the
                # behavior's querying band, so even a single-fold single-recording run
                # moves here — not just at phase boundaries. ``done`` is fractional, so
                # this now moves *within* a recording too, not only between recordings.
                within = (_j + (done / total if total else 1.0)) / _k
                report(
                    "querying",
                    _i + 0.45 + 0.44 * within,
                    f"{_b} · fold {_j + 1}/{_k} · {phase} · {found} matches",
                )

            report(
                "querying",
                i + 0.45 + 0.44 * (j / k),
                f"{behavior} · fold {j + 1}/{k}",
            )
            # Per-frame classification path (soft_knn): score every
            # frame, rank, integrate a per-frame AP — no span retrieval, no overlap/IoU.
            if cfg.method in _PERFRAME_METHODS:
                fold_fm, fold_null, fold_sim, fold_preds, fold_gt = _perframe_fold(
                    behavior,
                    db,
                    search,
                    exemplars,
                    remaining,
                    holdout,
                    cfg,
                    btype,
                    window_ns,
                    max_gap_ns,
                )
                if stop_check is not None and stop_check():
                    break  # interrupted → discard this partial fold, don't score it
                fold_metrics.append(fold_fm)
                fold_overlap.append([fold_null])
                if j == 0:  # fold 0 is the fixed split the figures read
                    similarity = fold_sim
                    predictions.extend(fold_preds)
                    ground_truth.extend(fold_gt)
                continue
            _examples, matches = core.query_tracks(
                db,
                queries,
                search,
                fusion=cfg.fusion,
                band_ns=band_ns,
                threshold=_PERMISSIVE_THRESHOLD,
                max_results=cfg.max_results,
                trajectory_split_gap_ns=max_gap_ns,
                method=cfg.method,
                method_params={
                    "keyframe_select": cfg.keyframe_select,
                    "n_keyframes": cfg.n_keyframes,
                },
                progress_cb=query_progress,
                stop_check=stop_check,
            )
            if stop_check is not None and stop_check():
                break  # interrupted mid-query → discard this partial fold, don't score it

            if j == 0 and search:  # the fixed split the figures use
                similarity = _fold_similarity(db, search[0], queries, cfg, max_gap_ns)

            gt_by_group: dict = {}
            gt_spans_by_group: dict = {}
            for subject, s, e in remaining:
                gt_by_group.setdefault(subject, []).append(
                    s if btype == "point" else (s, e)
                )
                gt_spans_by_group.setdefault(subject, []).append((s, e))
            preds = [
                ((m.recording_id, m.track_id), m.start_t_ns, m.end_t_ns, m.score)
                for m in matches
            ]
            outcomes = scoring.match_ranked_detailed(
                preds,
                gt_by_group,
                mode=btype,
                iou_threshold=cfg.iou_threshold,
                window_ns=window_ns,
            )
            fold_metrics.append(
                scoring.summarize(
                    [o.is_tp for o in outcomes],
                    [m.score for m in matches],
                    len(remaining),
                )
            )
            # The shifted-time null wraps each prediction inside its **own** subject's
            # recording, so it needs that subject's span — cheap here (the trajectory is
            # already loaded and cached across folds).
            fold_overlap.append(
                scoring.overlap_metrics(
                    preds,
                    gt_by_group,
                    _timeline_bounds(db, gt_by_group),
                    mode=btype,
                    # The configured level is always among them, so the headline row in
                    # metrics.csv can read its null straight out of this table.
                    levels=(
                        [cfg.iou_threshold]
                        if btype == "point"
                        else sorted({*scoring.OVERLAP_LEVELS, cfg.iou_threshold})
                    ),
                    window_ns=window_ns,
                    seed=cfg.seed,
                )
            )
            fold_preds, fold_gt = _detail_rows(
                behavior, matches, outcomes, gt_spans_by_group, exemplars, fold=j
            )
            predictions.extend(fold_preds)
            ground_truth.extend(fold_gt)

        if not fold_metrics:  # cancelled before any fold completed — nothing to score
            continue
        aggregate = scoring.aggregate_folds(fold_metrics)
        if aggregate.n_gt == 0:
            # Every fold resolved zero scorable ground truth (e.g. all searched subjects'
            # trajectories were unreadable) — report a skip, not a real "scored 0" row.
            results.append(
                BehaviorEval(
                    behavior, btype, len(eligible), None, "no scorable ground truth"
                )
            )
            continue
        report(
            "scoring", i + 0.9, f"{behavior} · aggregating {len(fold_metrics)} fold(s)"
        )
        results.append(
            BehaviorEval(
                behavior,
                btype,
                len(eligible),
                aggregate,
                fold_metrics=fold_metrics,
                overlap=scoring.aggregate_overlap(fold_overlap),
                predictions=predictions,
                ground_truth=ground_truth,
                similarity=similarity,
            )
        )
    report("scoring", n_behaviors, "done")
    return results


def _timeline_bounds(databases, gt_by_group: dict) -> dict:
    """``{subject: (t_min, t_max)}`` for each scored subject — the range the shifted-time
    null wraps a prediction within. A subject whose trajectory is unreadable is skipped;
    ``overlap_metrics`` then falls back to the prediction's own span (no shift)."""
    bounds = {}
    for subject in gt_by_group:
        try:
            t_ns = databases[subject][2]
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if len(t_ns):
            bounds[subject] = (int(t_ns[0]), int(t_ns[-1]))
    return bounds


def _track_ids(run_preview, recording, embed_run_id: str) -> list[int]:
    """Track ids embedded in ``embed_run_id`` for a recording ([] if the run is absent)."""
    try:
        return run_preview.embedding_track_ids(recording.dir, embed_run_id)
    except (FileNotFoundError, ValueError):
        return []


def _behavior_intervals(
    recording, behavior: str, stream_name: str | None = None
) -> list[tuple[int, int]]:
    """The ``(start_t_ns, end_t_ns)`` intervals labeled ``behavior`` in a recording.

    ``stream_name`` selects the ground-truth stream; omitted, it falls back to the
    recording's first behavior stream. It must be the same stream the eligibility
    check read, or a recording could be judged annotated off one stream and scored
    against another's intervals.
    """
    name = behavior_labels.resolve_behavior_stream(recording, stream_name)
    if name is None:
        return []
    rows = recording.stream_events(name)
    return [(r[0], r[1]) for r in rows if r[2] == behavior]


class _CachingTrajectories:
    """Memoize a ``LazyTrajectories`` so a multi-fold run loads each searched subject's
    trajectory once and reuses it across folds (rather than reloading per fold). A drop-in
    ``Mapping`` — ``query_tracks`` and ``_make_folds`` need no change. Memory is the
    subjects touched by the run; a single split keeps the un-cached lazy store."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cache: dict = {}

    def __contains__(self, subject) -> bool:
        return subject in self._inner

    def __getitem__(self, subject):
        if subject not in self._cache:
            self._cache[subject] = self._inner[subject]
        return self._cache[subject]


def _make_folds(pooled, databases, cfg, max_gap_ns):
    """Partition the **resolvable** instances (seeded) into cross-validation folds.

    Each fold takes a distinct ``n_holdout``-chunk of a seeded permutation (standard k-fold)
    out of the scored set; the first ``n_exemplars`` of that chunk become the query. Its GT is
    every *other* pooled instance (including unresolvable ones, which stay scorable as misses).
    ``folds=0`` = leave-one-out (all chunks); ``folds=k`` = the first ``k`` (capped at
    ``floor(N_resolvable / n_holdout)``). Returns ``(folds, k_max)`` where each fold is
    ``(exemplars, remaining, holdout)``.

    ``n_holdout`` defaults to ``n_exemplars``, which is the original behaviour: the query *is*
    the holdout. Setting it larger fixes the scored set across a sweep over ``n_exemplars`` —
    without it the positive set shrinks as the query grows, and a curve drawn against n measures
    its own denominator. The unused remainder of the chunk is scored neither as a positive nor as
    a negative; it leaves the test set entirely (see ``_perframe_fold``'s ``excl_mask``).
    """
    order = scoring.seeded_order(len(pooled), cfg.seed)
    # Resolvability needs only a subject's timeline, and every instance of a subject shares
    # it — so load each *distinct* subject once, not once per instance. On the uncached
    # folds=1 lazy store the per-instance load reloaded the full (~400MB) trajectory once
    # per GT interval (84x for an 84-instance behavior): the reload storm that made a single
    # split unusably slow. The memo also skips the emb matrix entirely for the check.
    t_ns_by_subject: dict = {}

    def _subject_t_ns(subject):
        if subject not in t_ns_by_subject:
            try:
                t_ns_by_subject[subject] = databases[subject][2]
            except (KeyError, ValueError, FileNotFoundError):
                t_ns_by_subject[subject] = None
        return t_ns_by_subject[subject]

    resolvable_idx = [
        idx
        for idx in order
        if _resolvable(
            _subject_t_ns(pooled[idx][0]), pooled[idx][1], pooled[idx][2], max_gap_ns
        )
    ]
    n_ex = cfg.n_exemplars
    n_hold = cfg.n_holdout if cfg.n_holdout is not None else n_ex
    k_max = len(resolvable_idx) // n_hold
    if k_max == 0:
        return [], 0
    k = k_max if cfg.folds == 0 else min(cfg.folds, k_max)
    folds = []
    for j in range(k):
        # Exclude the holdout from ``remaining`` **by index**, not by value: two identical
        # GT intervals (a duplicated annotation) must not collapse — holding out one copy
        # would otherwise drop the other from scoring and vary n_gt across folds.
        idxs = resolvable_idx[j * n_hold : (j + 1) * n_hold]
        chosen = set(idxs)
        holdout = [pooled[i] for i in idxs]
        exemplars = holdout[:n_ex]
        remaining = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        folds.append((exemplars, remaining, holdout))
    return folds, k_max


def _resolvable(t_ns, start_t_ns, end_t_ns, max_gap_ns) -> bool:
    """Whether a span has embedded rows lying within one continuous trajectory run.

    ``t_ns`` is the subject's timeline (loaded once per subject by ``_make_folds``), or
    ``None`` when the subject's trajectory is missing/unreadable.
    """
    if t_ns is None:
        return False
    try:
        lo, hi = core.span_to_rows(t_ns, start_t_ns, end_t_ns)
    except ValueError:
        return False
    gap = max_gap_ns if max_gap_ns is not None else core.default_max_gap_ns(t_ns)
    return any(rs <= lo and hi < re for rs, re in core.segment_runs(t_ns, gap))


def _detail_rows(behavior, matches, outcomes, gt_spans_by_group, exemplars, fold=0):
    """Per-clip prediction + ground-truth rows for one behavior fold (persisted detail).

    Predictions are the ranked matches tagged TP/FP (with the consumed GT span + IoU/
    offset on a hit); ground truth is every scored interval marked matched/FN plus the
    held-out exemplar spans (``role="exemplar"``, unscored). All rows carry ``fold``.
    """
    matched_keys: set[tuple] = set()
    predictions: list[query_eval_csv.PredictionRow] = []
    for rank, (m, o) in enumerate(zip(matches, outcomes), start=1):
        group = (m.recording_id, m.track_id)
        gt_start = gt_end = None
        if o.is_tp and o.matched_gt is not None:
            gt_start, gt_end = gt_spans_by_group[group][o.matched_gt]
            matched_keys.add((group, o.matched_gt))
        predictions.append(
            query_eval_csv.PredictionRow(
                behavior=behavior,
                recording_id=m.recording_id,
                track_id=m.track_id,
                rank=rank,
                outcome="tp" if o.is_tp else "fp",
                score=m.score,
                start_t_ns=m.start_t_ns,
                end_t_ns=m.end_t_ns,
                iou=o.iou,
                offset_ns=o.offset_ns,
                matched_gt_start_t_ns=gt_start,
                matched_gt_end_t_ns=gt_end,
                fold=fold,
            )
        )
    ground_truth: list[query_eval_csv.GroundTruthRow] = []
    for group, spans in gt_spans_by_group.items():
        for idx, (s, e) in enumerate(spans):
            ground_truth.append(
                query_eval_csv.GroundTruthRow(
                    behavior=behavior,
                    recording_id=group[0],
                    track_id=group[1],
                    start_t_ns=s,
                    end_t_ns=e,
                    role="scored",
                    outcome="matched" if (group, idx) in matched_keys else "fn",
                    fold=fold,
                )
            )
    for subject, s, e in exemplars:
        ground_truth.append(
            query_eval_csv.GroundTruthRow(
                behavior=behavior,
                recording_id=subject[0],
                track_id=subject[1],
                start_t_ns=s,
                end_t_ns=e,
                role="exemplar",
                outcome="",
                fold=fold,
            )
        )
    return predictions, ground_truth


# --------------------------------------------------------------------------- #
# Recorded analysis module (§16) — the executor-backed, durable twin of the
# synchronous ``evaluate_query`` above (two surfaces, one core, as the removed
# ``clip_query`` module had). It writes metrics + per-clip detail into ``analysis/<run_id>/`` and
# streams progress; the CLI's ``--out`` sweep stays the lightweight scripting path.
# --------------------------------------------------------------------------- #
@dataclass
class QueryEvalJob:
    """One behavior-query eval run: an embed run + behaviors + the scored config.

    Every scored config key must be a field here: the recorded run rebuilds its config
    from ``_job_config(job)``, so anything missing silently reverts to the model default
    — which is how a run configured with one matcher quietly scored with another.
    """

    job_id: str
    source_recording_ids: list[str]
    embed_run_id: str
    behaviors: list[str]
    stream_name: str | None
    track_id: int
    fusion: str
    method: str
    keyframe_select: str
    n_keyframes: int
    band_s: float
    k: int
    tau: float
    n_negatives: int
    negative_pool: str
    trajectory_split_gap_s: float | None
    max_results: int | None
    n_exemplars: int
    n_holdout: int | None
    seed: int
    iou_threshold: float
    point_window_s: float
    folds: int
    created_at: str

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_recording_ids": list(self.source_recording_ids),
            "embed_run_id": self.embed_run_id,
            "behaviors": list(self.behaviors),
            "stream_name": self.stream_name,
            "track_id": self.track_id,
            "fusion": self.fusion,
            "method": self.method,
            "keyframe_select": self.keyframe_select,
            "n_keyframes": self.n_keyframes,
            "band_s": self.band_s,
            "k": self.k,
            "tau": self.tau,
            "n_negatives": self.n_negatives,
            "negative_pool": self.negative_pool,
            "trajectory_split_gap_s": self.trajectory_split_gap_s,
            "max_results": self.max_results,
            "n_exemplars": self.n_exemplars,
            "n_holdout": self.n_holdout,
            "seed": self.seed,
            "iou_threshold": self.iou_threshold,
            "point_window_s": self.point_window_s,
            "folds": self.folds,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueryEvalJob":
        defaults = QueryEvalConfig.model_fields
        split_gap = d.get("trajectory_split_gap_s", d.get("max_gap_s"))  # renamed
        return cls(
            job_id=d["job_id"],
            source_recording_ids=list(d["source_recording_ids"]),
            embed_run_id=d["embed_run_id"],
            behaviors=list(d["behaviors"]),
            # A job recorded before the selector existed scored the first behavior
            # stream, which is what None means.
            stream_name=d.get("stream_name"),
            track_id=int(d["track_id"]),
            fusion=d["fusion"],
            # A job recorded before the matcher was selectable ran DTW, so its defaults
            # are the right reading of an older run.
            method=d.get("method", defaults["method"].default),
            keyframe_select=d.get(
                "keyframe_select", defaults["keyframe_select"].default
            ),
            n_keyframes=int(d.get("n_keyframes", defaults["n_keyframes"].default)),
            band_s=float(d["band_s"]),
            k=int(d.get("k", defaults["k"].default)),
            tau=float(d.get("tau", defaults["tau"].default)),
            n_negatives=int(d.get("n_negatives", defaults["n_negatives"].default)),
            # A job recorded before the pool was selectable sampled the clean one.
            negative_pool=d.get("negative_pool", "labeled_clean"),
            trajectory_split_gap_s=split_gap,
            max_results=d["max_results"],
            n_exemplars=int(d["n_exemplars"]),
            # A job recorded before the holdout existed held out exactly its exemplars, which is
            # what None means.
            n_holdout=(int(d["n_holdout"]) if d.get("n_holdout") is not None else None),
            seed=int(d["seed"]),
            iou_threshold=float(d["iou_threshold"]),
            point_window_s=float(d["point_window_s"]),
            folds=int(d.get("folds", 1)),  # compat: a pre-folds job is a single split
            created_at=d["created_at"],
        )


@dataclass
class QueryEvalProgressEvent:
    """Progress for an eval run (§16 analysis convention the GUI ``fit_progress_fraction``
    reads). ``recordings_done``/``recordings_total`` count *behaviors*; ``frames_*`` carry the
    within-behavior fraction (a resolution, not shown as text) so the bar moves during a
    single behavior's query. ``message`` is the live phase + per-behavior detail."""

    recordings_done: int
    recordings_total: int
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


_WITHIN_RESOLUTION = 1000  # frames_total units for the within-behavior fraction


_JOB_ONLY_KEYS = ("job_id", "created_at")


def _job_config(job: QueryEvalJob) -> dict:
    """The ``evaluate_query`` config dict reconstructed from a serialized job.

    Derived from ``to_dict`` rather than hand-listed: a key missing here does not fail,
    it silently falls back to the model default — which is how a recorded run configured
    with one matcher went on scoring with another.
    """
    return {k: v for k, v in job.to_dict().items() if k not in _JOB_ONLY_KEYS}


def null_at(overlap: list[scoring.OverlapMetrics], iou: float):
    """The per-level row matching ``iou`` (the configured level is always present)."""
    for row in overlap:
        if row.iou == iou:
            return row
    return None


def _metric_row(job: QueryEvalJob, be: BehaviorEval) -> query_eval_csv.MetricRow:
    m = be.metrics
    at = null_at(be.overlap, job.iou_threshold)
    return query_eval_csv.MetricRow(
        embed_run_id=job.embed_run_id,
        behavior=be.behavior,
        behavior_type=be.behavior_type,
        n_recordings=be.n_recordings,
        fusion=job.fusion,
        method=job.method,
        band_s=job.band_s,
        iou_threshold=job.iou_threshold,
        point_window_s=job.point_window_s,
        n_exemplars=job.n_exemplars,
        n_holdout=job.n_holdout,
        negative_pool=job.negative_pool,
        seed=job.seed,
        trajectory_split_gap_s=job.trajectory_split_gap_s,
        max_results=job.max_results,
        recall_null=at.recall_null if at else None,
        recall_lift=at.recall_lift if at else None,
        ap_null=at.ap_null if at else None,
        ap_lift=at.ap_lift if at else None,
        n_gt=m.n_gt,
        n_pred=m.n_pred,
        tp=m.tp,
        fp=m.fp,
        fn=m.fn,
        ap=m.ap,
        precision=m.precision,
        recall=m.recall,
        best_f1=m.best_f1,
        best_f1_threshold=m.best_f1_threshold,
        n_folds=m.n_folds,
        ap_std=m.ap_std,
        precision_std=m.precision_std,
        recall_std=m.recall_std,
        best_f1_std=m.best_f1_std,
    )


def _fold_rows(be: BehaviorEval) -> list[query_eval_csv.FoldMetricRow]:
    """Per-fold metric rows (for folds.csv) from a behavior's fold metrics."""
    return [
        query_eval_csv.FoldMetricRow(
            behavior=be.behavior,
            fold=j,
            n_gt=m.n_gt,
            n_pred=m.n_pred,
            tp=m.tp,
            fp=m.fp,
            fn=m.fn,
            ap=m.ap,
            precision=m.precision,
            recall=m.recall,
            best_f1=m.best_f1,
            best_f1_threshold=m.best_f1_threshold,
        )
        for j, m in enumerate(be.fold_metrics)
    ]


def _overlap_rows(be: BehaviorEval) -> list[query_eval_csv.OverlapRow]:
    """Per-overlap-level rows (for overlap.csv), each headline beside its null."""
    return [
        query_eval_csv.OverlapRow(
            behavior=be.behavior,
            behavior_type=be.behavior_type,
            iou=m.iou,
            n_gt=m.n_gt,
            n_pred=m.n_pred,
            tp=m.tp,
            recall=m.recall,
            recall_null=m.recall_null,
            recall_lift=m.recall_lift,
            precision=m.precision,
            ap=m.ap,
            ap_null=m.ap_null,
            ap_lift=m.ap_lift,
        )
        for m in be.overlap
    ]


@register_analysis_module
class QueryEvalModule(AnalysisModule):
    module_id = "query_eval"
    config_model = QueryEvalConfig
    id_prefix = "an"
    requires = ["embedding", "behavior"]
    selects = ("embedding", "embed_run_id")
    job_class = QueryEvalJob
    # ``evaluate_query``'s own docstring names these, in order: loading -> querying -> scoring.
    phases = ("loading", "querying", "scoring")
    progress_event_class = QueryEvalProgressEvent
    # frames_total is a per-mille fraction
    progress_counts_frames = False

    def validate(self, experiment_dir, config: dict) -> dict:
        from detail.api.run_preview import (
            embedding_track_ids,
            validate_comparability,
        )
        from detail.api.recording import Recording
        from detail.storage import experiment as experiment_store

        cfg = QueryEvalConfig.model_validate(config)  # bounds/shape only
        experiment_dir = Path(experiment_dir)
        if not cfg.behaviors:
            raise ValueError("behavior-query eval needs at least one behavior")
        recording_ids = list(
            cfg.source_recording_ids
        ) or experiment_store.discover_recordings(experiment_dir)
        if cfg.stream_name is not None:
            # A name that matches nothing would otherwise make every recording
            # ineligible, surfacing as "no recording is annotated" — which points at
            # the labels rather than at the typo that caused it.
            carriers = {
                sid: [
                    s.get("name")
                    for s in Recording(experiment_dir, sid).streams_by_role("behavior")
                ]
                for sid in recording_ids
            }
            if not any(cfg.stream_name in names for names in carriers.values()):
                available = sorted({n for names in carriers.values() for n in names})
                raise ValueError(
                    f"no selected recording has a behavior stream named "
                    f"{cfg.stream_name!r}; available: {available or '(none)'}"
                )
        # The search set is every requested recording that is both annotated for one of
        # the behaviors and carries the embed run at the search track — resolved here so
        # the pre-written record's ``recording_ids`` is the real universe (not empty).
        eligible: list[str] = []
        for sid in recording_ids:
            sess = Recording(experiment_dir, sid)
            if not any(
                sess.is_annotated_for(b, cfg.stream_name) for b in cfg.behaviors
            ):
                continue
            try:
                tracks = embedding_track_ids(
                    experiment_store.recording_path(experiment_dir, sid),
                    cfg.embed_run_id,
                )
            except (FileNotFoundError, ValueError):
                continue
            if cfg.track_id in tracks:
                # Retrieval is scored across every eligible recording, so an integer that
                # names a different animal in each would pool one animal's hits with
                # another's and report a single number for both.
                check_track_is_unambiguous(
                    cfg.track_id, sid, tracks, run_id=cfg.embed_run_id
                )
                eligible.append(sid)
        if not eligible:
            raise ValueError(
                "no requested recording is annotated for the behavior(s) and has embed "
                f"run {cfg.embed_run_id!r} at track {cfg.track_id}"
            )
        # Per behavior, not only for the set: one misspelt `--behavior` beside a real one
        # would otherwise be scored as "skipped — no eligible recordings" and exit 0.
        searched = [Recording(experiment_dir, sid) for sid in eligible]
        unscored = [
            b
            for b in cfg.behaviors
            if not any(sess.is_annotated_for(b, cfg.stream_name) for sess in searched)
        ]
        if unscored:
            available = sorted(
                {
                    b
                    for sess in searched
                    for b in sess.covered_behaviors(cfg.stream_name)
                }
            )
            raise ValueError(
                f"no recording with embed run {cfg.embed_run_id!r} at track "
                f"{cfg.track_id} is annotated for {', '.join(map(repr, unscored))}; "
                f"behaviors annotated there: {', '.join(available) or '(none)'}"
            )
        validate_comparability(
            [
                (experiment_store.recording_path(experiment_dir, sid), cfg.embed_run_id)
                for sid in eligible
            ]
        )
        out = dict(config)
        out["source_recording_ids"] = eligible
        return out

    def build_job(self, run_id: str, config: dict) -> QueryEvalJob:
        # Every job field is carried from the validated config by name, so adding a
        # config parameter cannot leave it behind here (the bug that made a recorded
        # run ignore its matcher). The model supplies bounds, defaults and legacy keys.
        cfg = QueryEvalConfig.model_validate(config)
        carried = {f.name for f in dataclass_fields(QueryEvalJob)} - set(_JOB_ONLY_KEYS)
        return QueryEvalJob(
            job_id=run_id,
            created_at=utc_timestamp(),
            **{name: getattr(cfg, name) for name in carried},
        )

    def run_dir_for(self, experiment_root, job):
        return analysis_store.analysis_dir(experiment_root, job.job_id)

    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        artifacts = [
            {"path": "metrics.csv", "kind": "query_eval_metrics", "label": "metrics"},
            {
                "path": "folds.csv",
                "kind": "query_eval_folds",
                "label": "per-fold metrics",
            },
            {
                "path": "overlap.csv",
                "kind": "query_eval_overlap",
                "label": "scores vs nulls by overlap",
            },
            {
                "path": "predictions.csv",
                "kind": "query_eval_predictions",
                "label": "predictions",
            },
            {
                "path": "ground_truth.csv",
                "kind": "query_eval_ground_truth",
                "label": "ground truth",
            },
        ]
        for behavior in config.get("behaviors", []):
            artifacts.append(
                {
                    "path": f"pr/{behavior}/",
                    "kind": "query_eval_pr",
                    "label": f"PR · {behavior}",
                }
            )
        return artifacts

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run_dir = Path(run_dir)
        experiment_root = run_dir.parent.parent  # <experiment>/analysis/<run_id>/
        emit = ThrottledProgress(progress_cb)
        started = time.monotonic()
        n_behaviors = max(len(job.behaviors), 1)
        last_msg = None

        def on_report(stage: str, fraction: float, detail: str) -> None:
            nonlocal last_msg
            elapsed = time.monotonic() - started
            per = fraction * n_behaviors  # 0..n_behaviors
            done = min(int(per), n_behaviors - 1)
            within = per - done
            # `stage` is one of `self.phases` (loading / querying / scoring); prefixing it onto
            # `detail` is what lets a run page know which declared phase a message belongs to —
            # `detail` alone (a behavior/fold name) carries none of the three phase words.
            message = f"{stage} · {detail}" if detail else stage
            # Force a *new* message past the throttle so each phase / behavior / fold shows
            # the moment it starts (not up to PROGRESS_INTERVAL_S — plus a job.json network
            # round-trip — later); an unchanged message stays throttled. query-eval's
            # messages are paced by real work (per behavior / fold / subject), never floods.
            force = message != last_msg
            last_msg = message
            emit(
                QueryEvalProgressEvent(
                    recordings_done=done,
                    recordings_total=n_behaviors,
                    frames_done=round(within * _WITHIN_RESOLUTION),
                    frames_total=_WITHIN_RESOLUTION,
                    elapsed_s=elapsed,
                    eta_s=remaining_seconds(fraction, elapsed),
                    message=message,
                ),
                force=force,
            )

        stop_check = stop_event.is_set if stop_event is not None else None
        # An informative first message (not "starting"): resolving recordings + loading the
        # embeddings from zarr (a network share here) is the slow opening, so name it.
        on_report("loading", 0.0, f"loading embeddings for {n_behaviors} behavior(s)…")
        results = evaluate_query(
            experiment_root, _job_config(job), progress=on_report, stop_check=stop_check
        )

        query_eval_csv.write_metrics_csv(
            run_dir / "metrics.csv",
            [_metric_row(job, be) for be in results if be.metrics is not None],
        )
        query_eval_csv.write_folds_csv(
            run_dir / "folds.csv",
            [row for be in results for row in _fold_rows(be)],
        )
        query_eval_csv.write_overlap_csv(
            run_dir / "overlap.csv",
            [row for be in results for row in _overlap_rows(be)],
        )
        query_eval_csv.write_predictions_csv(
            run_dir / "predictions.csv",
            [row for be in results for row in be.predictions],
        )
        query_eval_csv.write_ground_truth_csv(
            run_dir / "ground_truth.csv",
            [row for be in results for row in be.ground_truth],
        )
        # One PR curve per fold, under pr/<behavior>/fold_NN.csv (the GUI overlays them).
        for be in results:
            for j, fm in enumerate(be.fold_metrics):
                query_eval_csv.write_pr_points(
                    run_dir / "pr" / be.behavior / f"fold_{j:02d}.csv", fm.pr_points
                )
        # The fixed-split similarity curve per behavior (the export's similarity figure reads it).
        for be in results:
            if be.similarity is not None:
                recording_id, t_ns, score = be.similarity
                query_eval_csv.write_similarity(
                    run_dir / "similarity" / f"{be.behavior}.npz",
                    recording_id,
                    t_ns,
                    score,
                )

        scored = [be.behavior for be in results if be.metrics is not None]
        # Terminal tick at exactly 1.0: last behavior complete (within=1) rather than
        # recordings_done=n_behaviors, which would render "recording n+1/n".
        emit(
            QueryEvalProgressEvent(
                recordings_done=max(n_behaviors - 1, 0),
                recordings_total=n_behaviors,
                frames_done=_WITHIN_RESOLUTION,
                frames_total=_WITHIN_RESOLUTION,
                elapsed_s=time.monotonic() - started,
                eta_s=0.0,
                message="done",
            ),
            force=True,
        )

        interrupted = stop_event is not None and stop_event.is_set()
        record = analysis_store.read_analysis_yaml(run_dir)
        record["embed_run_id"] = job.embed_run_id
        record["track_id"] = job.track_id
        # The whole scored config, so **Refine** re-opens the launch on the settings the
        # run actually used (a param missing here silently reverts to a default).
        record["params"] = {
            k: v
            for k, v in _job_config(job).items()
            if k
            not in ("source_recording_ids", "embed_run_id", "behaviors", "track_id")
        }
        record["behaviors"] = [
            {
                "behavior": be.behavior,
                "behavior_type": be.behavior_type,
                "n_recordings": be.n_recordings,
                "n_folds": be.metrics.n_folds if be.metrics is not None else 0,
                "skipped": be.skipped,
            }
            for be in results
        ]
        record["columns"] = list(query_eval_csv.COLUMNS)
        record["artifacts"] = self.output_manifest(job.job_id, {"behaviors": scored})
        record["status"] = "partial" if interrupted else "complete"
        analysis_store.write_analysis_yaml(run_dir, record)
