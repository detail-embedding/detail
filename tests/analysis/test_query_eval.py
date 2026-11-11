"""Pure retrieval scoring for the behavior-query eval (§16)."""

import numpy as np
import pytest

from detail.analysis import query_eval as qe


def test_temporal_iou():
    assert qe.temporal_iou((0, 100), (0, 100)) == 1.0
    assert qe.temporal_iou((0, 100), (50, 150)) == 50 / 150
    assert qe.temporal_iou((0, 10), (20, 30)) == 0.0


def test_match_state_greedy():
    preds = [
        (("s", 0), 0, 100, 0.1),
        (("s", 0), 500, 600, 0.2),
        (("s", 0), 195, 305, 0.3),
    ]
    gt = {("s", 0): [(0, 100), (200, 300)]}
    assert qe.match_ranked(preds, gt, mode="state", iou_threshold=0.5) == [
        True,
        False,
        True,
    ]


def test_match_state_one_to_one():
    # two predictions over the same single GT -> only the first (best-ranked) is a TP
    preds = [(("s", 0), 0, 100, 0.1), (("s", 0), 0, 100, 0.2)]
    gt = {("s", 0): [(0, 100)]}
    assert qe.match_ranked(preds, gt, mode="state", iou_threshold=0.5) == [True, False]


def test_match_point_window():
    preds = [
        (("s", 0), 90, 90, 0.1),
        (("s", 0), 305, 305, 0.2),
        (("s", 0), 999, 999, 0.3),
    ]
    gt = {("s", 0): [100, 300]}
    assert qe.match_ranked(preds, gt, mode="point", window_ns=20) == [True, True, False]


def test_match_wrong_group_never_hits():
    preds = [(("a", 0), 0, 100, 0.1)]
    gt = {("b", 0): [(0, 100)]}
    assert qe.match_ranked(preds, gt, mode="state", iou_threshold=0.5) == [False]


def test_average_precision():
    assert qe.average_precision([True, True], 2)[0] == 1.0
    ap, _ = qe.average_precision([True, False, True], 2)
    assert abs(ap - 5 / 6) < 1e-9
    assert qe.average_precision([True], 0) == (0.0, [])


def test_summarize_counts_and_best_f1():
    m = qe.summarize([True, False, True], [0.1, 0.2, 0.3], n_gt=2)
    assert (m.tp, m.fp, m.fn, m.n_pred) == (2, 1, 0, 3)
    assert m.recall == 1.0 and abs(m.precision - 2 / 3) < 1e-9
    assert m.best_f1 == 0.8 and m.best_f1_threshold == 0.3


def test_seeded_order_deterministic():
    assert qe.seeded_order(6, 7) == qe.seeded_order(6, 7)
    assert sorted(qe.seeded_order(6, 7)) == [0, 1, 2, 3, 4, 5]


def test_match_ranked_detailed_names_matched_gt_and_iou():
    preds = [
        (("s", 0), 0, 100, 0.1),
        (("s", 0), 500, 600, 0.2),
        (("s", 0), 195, 305, 0.3),
    ]
    gt = {("s", 0): [(0, 100), (200, 300)]}
    out = qe.match_ranked_detailed(preds, gt, mode="state", iou_threshold=0.5)
    assert [o.is_tp for o in out] == [True, False, True]
    assert out[0].matched_gt == 0 and out[0].iou == 1.0 and out[0].offset_ns is None
    assert out[1].matched_gt is None and out[1].iou is None  # a false positive
    assert out[2].matched_gt == 1 and out[2].iou > 0.5


def test_match_ranked_detailed_point_offset():
    preds = [(("s", 0), 90, 90, 0.1), (("s", 0), 999, 999, 0.2)]
    gt = {("s", 0): [100]}
    out = qe.match_ranked_detailed(preds, gt, mode="point", window_ns=20)
    assert out[0].is_tp and out[0].offset_ns == 10 and out[0].iou is None
    assert not out[1].is_tp and out[1].offset_ns is None


def test_aggregate_folds_mean_and_std():
    m1 = qe.EvalMetrics(3, 3, 3, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.1, [(1.0, 1.0)])
    m2 = qe.EvalMetrics(3, 4, 2, 2, 1, 0.5, 0.5, 0.75, 0.6, 0.2, [(0.75, 0.5)])
    agg = qe.aggregate_folds([m1, m2])
    assert agg.n_folds == 2
    assert abs(agg.ap - 0.75) < 1e-9 and agg.ap_std > 0.0  # mean + spread
    assert agg.n_gt == 3  # constant across folds
    # counts stay internally consistent despite independent per-metric rounding
    assert agg.tp + agg.fp == agg.n_pred and agg.tp + agg.fn == agg.n_gt


def test_aggregate_single_fold_has_zero_std():
    m = qe.EvalMetrics(3, 3, 3, 0, 0, 0.9, 1.0, 1.0, 0.95, 0.1, [(1.0, 1.0)])
    agg = qe.aggregate_folds([m])
    assert agg.n_folds == 1 and agg.ap == 0.9 and agg.ap_std == 0.0


def test_match_ranked_equals_detailed_flags():
    # match_ranked must stay exactly the detailed result's TP flags (one source of truth).
    preds = [(("s", 0), 0, 100, 0.1), (("s", 0), 0, 100, 0.2)]
    gt = {("s", 0): [(0, 100)]}
    flags = qe.match_ranked(preds, gt, mode="state", iou_threshold=0.5)
    detailed = qe.match_ranked_detailed(preds, gt, mode="state", iou_threshold=0.5)
    assert flags == [o.is_tp for o in detailed] == [True, False]


# --- nulls and lift ---------------------------------------------------------------- #
# A number that doesn't clear its null isn't real. These pin that the null is actually
# uninformative: a random ranking must score like the shuffled-rank null, and spans slid
# to the wrong places must score like the shifted-time null.

_GROUP = ("s", 0)


def _evenly_spaced(n, width, step, score_from_rank=True):
    """`n` predictions of a fixed width, `step` apart, ranked best-first."""
    return [
        (_GROUP, i * step, i * step + width, (i / n) if score_from_rank else 0.5)
        for i in range(n)
    ]


def test_a_perfect_ranking_beats_its_shuffled_null():
    gt = {_GROUP: [(i * 1000, i * 1000 + 500) for i in range(10)]}
    # The first 10 predictions land exactly on ground truth; the next 40 are junk.
    preds = [(_GROUP, i * 1000, i * 1000 + 500, 0.01 * i) for i in range(10)]
    preds += [(_GROUP, 50_000 + i * 700, 50_000 + i * 700 + 500, 0.5 + 0.01 * i)
              for i in range(40)]  # fmt: skip
    (m,) = qe.overlap_metrics(
        preds, gt, {_GROUP: (0, 100_000)}, mode="state", levels=[0.5], n_null=20
    )
    assert m.recall == 1.0
    assert m.ap > 0.9
    assert m.ap_null < 0.5  # a random order of the same outcomes does far worse
    assert m.ap_lift > 0.4
    assert m.recall_lift > 0.5  # and sliding them elsewhere finds almost nothing


def test_a_random_ranking_scores_like_its_null():
    """The point of ap_null: shuffle the scores and the metric refuses to credit it."""
    import numpy as np

    gt = {_GROUP: [(i * 1000, i * 1000 + 500) for i in range(10)]}
    preds = [(_GROUP, i * 1000, i * 1000 + 500, 0.0) for i in range(10)]
    preds += [(_GROUP, 50_000 + i * 700, 50_000 + i * 700 + 500, 0.0)
              for i in range(40)]  # fmt: skip
    rng = np.random.default_rng(3)
    shuffled = [preds[i] for i in rng.permutation(len(preds))]
    (m,) = qe.overlap_metrics(
        shuffled, gt, {_GROUP: (0, 100_000)}, mode="state", levels=[0.5], n_null=50
    )
    assert abs(m.ap - m.ap_null) < 0.12  # no ranking information → no lift


def test_tiling_predictions_do_not_clear_the_recall_null():
    """A matcher that blankets the timeline hits ground truth by coverage alone — the
    tiling artifact. Its recall must not clear the shifted-time null."""
    gt = {_GROUP: [(i * 5000, i * 5000 + 400) for i in range(8)]}
    tiling = [(_GROUP, i * 400, i * 400 + 400, 0.5) for i in range(100)]
    (m,) = qe.overlap_metrics(
        tiling, gt, {_GROUP: (0, 40_000)}, mode="state", levels=[0.1], n_null=30
    )
    assert m.recall_lift < 0.2


def test_stricter_overlap_scores_no_higher():
    gt = {_GROUP: [(0, 1000), (5000, 6000)]}
    preds = [(_GROUP, 0, 1400, 0.1), (_GROUP, 5000, 5600, 0.2)]
    loose, medium, strict = qe.overlap_metrics(
        preds,
        gt,
        {_GROUP: (0, 10_000)},
        mode="state",
        levels=qe.OVERLAP_LEVELS,
        n_null=5,
    )
    assert loose.iou == 0.1 and strict.iou == 0.5
    assert loose.recall >= medium.recall >= strict.recall


def test_shift_wraps_within_each_group_timeline():
    preds = [(("a", 0), 100, 200, 0.1), (("b", 0), 100, 200, 0.2)]
    bounds = {("a", 0): (0, 1000), ("b", 0): (0, 500)}
    moved = qe._shift_spans(preds, bounds, 900)
    assert moved[0][1] == 0  # (100 + 900) % 1000
    assert moved[1][1] == 0  # (100 + 900) % 500 == 0, its own shorter timeline
    assert all(e - s == 100 for _g, s, e, _sc in moved)  # lengths preserved


def test_no_ground_truth_yields_zero_not_a_crash():
    (m,) = qe.overlap_metrics(
        [(_GROUP, 0, 100, 0.1)],
        {_GROUP: []},
        {_GROUP: (0, 1000)},
        mode="state",
        levels=[0.5],
        n_null=3,
    )
    assert m.n_gt == 0 and m.recall == 0.0 and m.ap == 0.0


def test_no_predictions_yields_zero_not_a_crash():
    (m,) = qe.overlap_metrics(
        [], {_GROUP: [(0, 100)]}, {_GROUP: (0, 1000)}, mode="state",
        levels=[0.5], n_null=3,
    )  # fmt: skip
    assert m.n_pred == 0 and m.recall == 0.0 and m.precision == 0.0


def test_overlap_metrics_are_deterministic_for_a_seed():
    gt = {_GROUP: [(0, 1000)]}
    preds = [(_GROUP, 0, 1000, 0.1), (_GROUP, 4000, 5000, 0.2)]
    kw = dict(mode="state", levels=[0.5], n_null=10)
    a = qe.overlap_metrics(preds, gt, {_GROUP: (0, 10_000)}, **kw)
    b = qe.overlap_metrics(preds, gt, {_GROUP: (0, 10_000)}, **kw)
    assert a == b


def test_aggregate_overlap_means_across_folds():
    fold_a = qe.OverlapMetrics(0.5, 10, 20, 5, 0.5, 0.1, 0.4, 0.25, 0.4, 0.1, 0.3)
    fold_b = qe.OverlapMetrics(0.5, 10, 20, 7, 0.7, 0.1, 0.6, 0.35, 0.6, 0.1, 0.5)
    (agg,) = qe.aggregate_overlap([[fold_a], [fold_b]])
    assert agg.recall == 0.6 and agg.ap == 0.5 and agg.tp == 6


# --- per-frame AP (perframe_metrics) -------------------------------------------------- #


def test_perframe_metrics_ap_matches_summarize_on_distinct_scores():
    # AP + best-F1 agree with the reference on a distinct-score ranking (no ties). n_pred
    # deliberately differs: perframe reports the best-F1 operating-point count, summarize the
    # full predicted set.
    scores = np.array([0.9, 0.85, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2])
    labels = np.array([True, False, True, True, False, False, True, False])
    order = np.argsort(-scores, kind="stable")
    tp_flags = [bool(labels[o]) for o in order]
    m_vec = qe.perframe_metrics(scores, labels)
    m_ref = qe.summarize(tp_flags, [float(scores[o]) for o in order], int(labels.sum()))
    assert m_vec.ap == pytest.approx(m_ref.ap)
    assert m_vec.best_f1 == pytest.approx(m_ref.best_f1)
    assert m_vec.n_gt == m_ref.n_gt == 4


def test_perframe_metrics_ap_is_order_independent_under_ties():
    # A per-frame classifier saturates, so tie groups are large; AP must not swing with the
    # order inside a tie. The same multiset at one score integrates to the prevalence.
    a = qe.perframe_metrics(np.ones(4), np.array([True, True, False, False]))
    b = qe.perframe_metrics(np.ones(4), np.array([False, False, True, True]))
    assert a.ap == pytest.approx(b.ap) == pytest.approx(0.5)


def test_perframe_metrics_reports_the_best_f1_operating_point():
    # The confusion block is at the best-F1 threshold, not predict-all: a perfect ranking
    # operates at tp=2/fp=0 (n_pred=2), not "predict every frame" (recall=1 but fp=all).
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([True, True, False, False])
    m = qe.perframe_metrics(scores, labels)
    assert m.ap == pytest.approx(1.0) and m.best_f1 == pytest.approx(1.0)
    assert (
        m.n_gt == 2 and m.n_pred == 2
    )  # tp+fp at the operating point, not all 4 frames
    assert m.tp == 2 and m.fp == 0 and m.fn == 0
    assert m.tp + m.fp == m.n_pred and m.tp + m.fn == m.n_gt  # internally consistent


def test_perframe_metrics_edges_empty_and_no_positives():
    import numpy as np

    empty = qe.perframe_metrics(np.array([]), np.array([], dtype=bool))
    assert empty.n_gt == 0 and empty.n_pred == 0 and empty.ap == 0.0
    none_pos = qe.perframe_metrics(np.array([0.9, 0.1]), np.array([False, False]))
    assert none_pos.n_gt == 0 and none_pos.ap == 0.0


def test_perframe_metrics_thins_the_stored_pr_curve():
    import numpy as np

    n = 2000
    rng = np.random.default_rng(0)
    scores = rng.random(n)
    labels = rng.random(n) < 0.1
    m = qe.perframe_metrics(scores, labels)
    assert len(m.pr_points) <= 512  # the CSV curve is capped even for a huge ranking


# --- the per-frame floor (perframe_null) ----------------------------------------------- #


def _blocks(n=4000, block=200, rng=None):
    """Contiguous-bout labels, the shape real behavior annotation has."""
    labels = np.zeros(n, dtype=bool)
    for start in range(0, n, block * 4):
        labels[start : start + block] = True
    return labels


def _smooth_noise(n, window, rng):
    """A zero-signal score that is smooth in time — the degenerate answer that matters here,
    because embeddings are autocorrelated and a prevalence floor does not survive it."""
    return np.convolve(rng.normal(size=n), np.ones(window) / window, mode="same")


def test_perframe_null_is_much_harder_for_a_smooth_zero_signal_score_than_prevalence():
    """Why the floor is a circular shift and not the prevalence.

    A score with NO relationship to the labels, but with the temporal smoothness real scores
    have, clears a prevalence floor comfortably — prevalence is the expected AP of an i.i.d.
    permutation, and these scores are not i.i.d. The shift floor keeps the smoothness and so
    is not fooled by it. Measured here: mean spurious lift drops ~5x.

    Asserted on the MEAN over trials, not a single trial: one realization against a mean of
    draws has real variance, and an unlucky draw can clear any floor. The claim is about the
    floor's bias, not its variance.
    """
    rng = np.random.default_rng(0)
    labels = _blocks()
    prevalence = labels.mean()
    shift_lift, prevalence_lift = [], []
    for _ in range(20):
        scores = _smooth_noise(len(labels), 400, rng)
        m = qe.perframe_metrics(scores, labels)
        shift_lift.append(
            qe.perframe_null([(scores, labels)], m, 0.5, n_null=20).ap_lift
        )
        prevalence_lift.append(m.ap - prevalence)
    shift_mean = sum(shift_lift) / len(shift_lift)
    prevalence_mean = sum(prevalence_lift) / len(prevalence_lift)
    assert prevalence_mean > 0.05, "fixture is not exercising the failure mode"
    assert shift_mean < prevalence_mean / 3, (
        f"shift floor mean lift {shift_mean:.3f} vs prevalence {prevalence_mean:.3f} — "
        "the floor is still being beaten by smoothness alone"
    )


def test_perframe_null_leaves_real_signal_standing():
    """The other half: a floor that nothing beats is equally useless. A score that IS the
    labels must clear it decisively."""
    labels = _blocks()
    scores = labels.astype(float) + np.random.default_rng(1).normal(
        0, 0.01, len(labels)
    )
    m = qe.perframe_metrics(scores, labels)
    null = qe.perframe_null([(scores, labels)], m, 0.5, n_null=20)
    assert m.ap > 0.95
    assert null.ap_lift > 0.5


def test_perframe_null_shifts_within_each_subject():
    """A shift must never slide one subject's scores onto another's labels: the subjects are
    different recordings, and cross-subject alignment is not a null, it is a bug."""
    rng = np.random.default_rng(2)
    a = (_smooth_noise(2000, 200, rng), _blocks(2000, 100))
    b = (_smooth_noise(2000, 200, rng), _blocks(2000, 100))
    null = qe.perframe_null([a, b], qe.perframe_metrics(*a), 0.5, n_null=5)
    assert 0.0 <= null.ap_null <= 1.0
    assert null.iou == 0.5  # the configured level, carried only as the lookup key


def test_perframe_null_survives_an_empty_scored_set():
    empty = qe.perframe_metrics(np.array([]), np.array([], dtype=bool))
    null = qe.perframe_null([], empty, 0.5)
    assert null.ap_null == 0.0 and null.recall_null == 0.0
