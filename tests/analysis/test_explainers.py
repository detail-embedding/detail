"""Explainers (``analysis/explainers.py``): a held-out score with its floor, and the gate.

The class: **a confident explanation of nothing.** Every classifier finds features to lean on;
on noise it still ranks them. So every explainer must score itself on held-out recordings beside
a shuffled-membership floor, and noise must come back not separated whichever explainer runs.
Rows with gaps are dropped and counted, never filled.

**A held-out score taken on a fold that cannot be balanced.** Folds split by recording, and a
recording without the cluster in it is one class on its held-out side, where "balanced"
accuracy is plain accuracy on that class: a model that learned only *not this recording*
passes. A cluster drawn wholly from one recording was scored only on such folds and read 83%
"on held-out recordings" on the Two-animal demo, beside a floor that shuffled the cluster over
every recording and so sat near 50%, separated, with an empty permutation chart (which skipped
exactly those folds). Such a fold is now unscored in the score and its floor alike, the floor is
scored fold for fold on the score's own folds, and with none left there is no score and
``Explanation.unscored`` says why.
"""

import numpy as np
import pytest

from detail.analysis import explainers


def _data(seed=0, n=240, planted=True):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"rec{i}" for i in range(6)], n // 6)
    y = rng.random(n) < 0.5
    height = rng.normal(0.6, 0.2, n) + (0.8 * y if planted else 0)
    speed = rng.normal(1.9, 0.5, n) - (1.5 * y if planted else 0)
    noise = rng.normal(0, 1, n)
    return np.c_[height, speed, noise], y, groups, ["height", "speed", "noise"]


FAST = ["logistic_regression", "decision_tree", "random_forest_permutation"]


@pytest.mark.parametrize("name", FAST)
def test_a_planted_difference_separates_with_the_right_direction(name):
    X, y, groups, names = _data()
    e = explainers.explain(X, y, groups, names, explainer=name)
    assert e.separated and e.held_out == "recording"
    assert e.accuracy > e.floor + e.margin
    by_name = {i.name: i for i in e.importances}
    assert by_name["height"].direction == 1 and by_name["speed"].direction == -1
    assert e.importances[0].name in {"height", "speed"}


@pytest.mark.parametrize("name", FAST)
def test_noise_is_not_separated(name):
    X, y, groups, names = _data(planted=False)
    e = explainers.explain(X, y, groups, names, explainer=name)
    assert not e.separated, (e.accuracy, e.floor)


def test_a_mostly_missing_measure_is_dropped_and_named_and_gaps_drop_rows():
    X, y, groups, names = _data()
    gappy = np.where(np.arange(len(y)) % 3 == 0, 1.0, np.nan)
    X = np.c_[X, gappy]
    X[0, 0] = np.nan
    e = explainers.explain(X, y, groups, [*names, "gappy"], explainer="decision_tree")
    assert "gappy" in e.measures_dropped and "missing on" in e.measures_dropped["gappy"]
    assert e.rows_dropped == 1
    assert "gappy" not in e.measures_used


def test_one_recording_is_held_out_by_row_and_says_so():
    X, y, _groups, names = _data()
    e = explainers.explain(
        X, y, np.array(["only"] * len(y)), names, explainer="decision_tree"
    )
    assert e.held_out.startswith("row")


def _confined(in_cluster_recordings: int, seed=0, per_recording=60):
    """Four recordings; the cluster's rows only in the first ``in_cluster_recordings``, the
    comparison's in all four. ``light`` names the recording, as a camera's lighting does, so a
    model can learn "not one of those recordings" and nothing about the cluster."""
    rng = np.random.default_rng(seed)
    groups, y, light = [], [], []
    for r in range(4):
        member = rng.random(per_recording) < (0.5 if r < in_cluster_recordings else 0.0)
        groups += [f"rec{r}"] * per_recording
        y += member.tolist()
        light += (r + rng.normal(0, 0.05, per_recording)).tolist()
    y = np.array(y)
    noise = rng.normal(0, 1, (len(y), 2))
    return np.c_[light, noise], y, np.array(groups), ["light", "a", "b"]


@pytest.mark.parametrize("name", FAST)
def test_a_cluster_from_one_recording_has_no_held_out_score(name):
    """A held-out recording without the cluster in it is one class, and its "balanced"
    accuracy was plain accuracy on that class: a model that learned the recording, not the
    cluster, scored 83% "on held-out recordings" on the Two-animal demo, against a floor that
    shuffled the cluster across every recording and so sat near 50%. The permutation chart
    beside it was empty, because it skipped exactly those folds."""
    X, y, groups, names = _confined(1)
    e = explainers.explain(X, y, groups, names, explainer=name)
    assert not np.isfinite(e.accuracy) and not e.separated, (e.accuracy, e.floor)
    assert e.unscored == explainers.NO_SCORABLE_FOLD
    assert e.fold_accuracy == [] and e.fold_floor == []


def test_the_floor_is_scored_on_the_folds_the_cluster_is_scored_on():
    """Cluster in two of four recordings: the other two folds hold one class, so neither the
    score nor its floor counts them, and the two lists pair fold for fold."""
    X, y, groups, names = _confined(2)
    e = explainers.explain(X, y, groups, names, explainer="logistic_regression")
    assert len(e.fold_accuracy) == len(e.fold_floor) == 2
    assert np.isfinite(e.accuracy) and e.unscored == ""


def test_too_few_rows_fits_nothing_and_says_why():
    X, y, groups, names = _data(n=12)
    e = explainers.explain(X, y, groups, names, explainer="decision_tree")
    assert not e.separated and not e.importances
    assert "too few complete rows" in e.note


def test_rules_read_as_conditions_on_labels():
    X, y, groups, names = _data()
    e = explainers.explain(
        X, y, groups, names, explainer="decision_tree", labels={"height": "mask height"}
    )
    assert e.rules and all(
        r.precision >= explainers.RULE_MIN_PRECISION for r in e.rules
    )
    assert any("mask height" in r.text for r in e.rules)


def test_the_default_falls_back_without_shap_and_says_so(monkeypatch):
    real = explainers.installed.is_importable
    monkeypatch.setattr(
        explainers.installed,
        "is_importable",
        lambda m: False if m == "shap" else real(m),
    )
    chosen, note = explainers.resolve(explainers.DEFAULT_EXPLAINER)
    assert chosen.name == explainers.FALLBACK_EXPLAINER and "shap" in note
    with pytest.raises(ValueError, match="unknown explainer"):
        explainers.resolve("nope")


def test_shap_explains_when_installed():
    pytest.importorskip("shap")
    X, y, groups, names = _data()
    e = explainers.explain(X, y, groups, names, explainer="random_forest_shap")
    assert e.explainer == "random_forest_shap" and e.separated
    by_name = {i.name: i for i in e.importances}
    assert by_name["height"].direction == 1 and by_name["speed"].direction == -1
