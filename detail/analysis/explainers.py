"""Which measurements tell a cluster apart — several at once, and how far to trust the answer.

``descriptor_ranking.rank_selection`` asks one measurement at a time how far a cluster sits from
the rest (Cliff's delta). That is the quick view, and it cannot see two measurements that only
separate the cluster *together* — high **and** still, say, where neither alone is unusual. An
explainer fits a small classifier to the cluster's rows against a comparison set's rows and
reports which measurements it leaned on, in which direction, and how well it did.

**The score comes first, with its floor.** A classifier always finds *some* features to lean on,
so an importance chart from a model at chance is a confident-looking explanation of nothing. Every
explainer here scores itself by **balanced accuracy on held-out recordings** — the folds split by
recording, so it is always scored on animals it has not seen and cannot pass by learning one
animal's quirks — beside a **floor**: the same fit on the same rows with the membership shuffled.
``separated`` is the gate, and a surface keeps the chart folded behind it when it is False. That is
a finding in itself: the cluster is defined by something these measurements cannot see.

**Rows with a gap are dropped, and counted.** A measurement missing on most rows is dropped whole
(and named); a row still missing one of the rest is dropped and counted. Never filled: a filled
value is invented behaviour, the same rule ``descriptors.py`` keeps (NaN is missing, never zero).
Dropping rather than relying on a library's own missing-value handling also keeps the four
explainers on the same rows, so their answers can be compared.

**Pluggable.** An explainer is a registered :class:`Explainer`: a name, a label, the importable
modules it needs, and one ``fit``. ``random_forest_shap`` is the default when ``shap`` is
installed; without it the forest falls back to permutation importance and says so
(:func:`resolve`). scikit-learn and shap are imported inside the fits only (the ``analysis``
extra), so this module imports with neither.

Pure numpy in, dataclasses out: no paths, no Qt, no zarr.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from detail.core import installed

#: A measurement missing on more than this share of rows is dropped rather than costing rows.
MAX_MISSING_SHARE = 0.5
#: How far above its floor a held-out score must sit to count as separating (balanced accuracy,
#: 0–1). A default, saved with every result, and settable wherever a profile is asked for.
DEFAULT_MARGIN = 0.1
#: Shuffles averaged into the floor — one shuffle is a noisy floor.
FLOOR_SHUFFLES = 3
#: The fewest rows a side may have before nothing is fitted at all.
MIN_ROWS_PER_SIDE = 10
#: Depth of the decision tree read out as plain rules.
RULE_DEPTH = 3
#: A rule is kept only when the leaf it ends in is mostly the cluster.
RULE_MIN_PRECISION = 0.6


@dataclass(frozen=True)
class Importance:
    """One measurement's weight in the fit, and which way it pushes.

    ``direction`` is +1 when higher values push toward the cluster, -1 when they push away, 0
    when there is no consistent direction. ``cluster_median`` / ``comparison_median`` are in
    the measurement's own units, so a surface can show "1.4 vs 0.6".
    """

    name: str
    value: float
    direction: int
    cluster_median: float
    comparison_median: float


@dataclass(frozen=True)
class Rule:
    """A readable path through a shallow decision tree: ``conditions`` all hold → the cluster.

    ``precision`` is how often a row meeting them is the cluster; ``coverage`` how much of the
    cluster meets them."""

    conditions: tuple[str, ...]
    precision: float
    coverage: float

    @property
    def text(self) -> str:
        return " AND ".join(self.conditions)


@dataclass
class Explanation:
    """What one explainer found. ``accuracy`` and ``floor`` are mean balanced accuracy over the
    held-out folds (0–1); ``separated`` is whether ``accuracy`` clears ``floor`` by ``margin``.
    """

    explainer: str
    accuracy: float
    floor: float
    margin: float
    separated: bool
    importances: list[Importance]
    rules: list[Rule]
    n_cluster: int
    n_comparison: int
    held_out: str
    measures_used: list[str]
    measures_dropped: dict[str, str] = field(default_factory=dict)
    rows_dropped: int = 0
    note: str = ""
    fold_accuracy: list[float] = field(default_factory=list)
    fold_floor: list[float] = field(default_factory=list)
    #: Why there is no held-out score although there were rows enough to fit
    #: (:data:`NO_SCORABLE_FOLD`), else ``""``. Apart from ``note``, which says how the fit was
    #: made (an explainer's fallback) and must survive beside it.
    unscored: str = ""


@dataclass(frozen=True)
class Explainer:
    name: str
    label: str
    needs: tuple[str, ...]
    fit: Callable[..., Explanation]
    description: str = ""

    def available(self) -> bool:
        return all(installed.is_importable(module) for module in self.needs)

    def missing(self) -> list[str]:
        return [m for m in self.needs if not installed.is_importable(m)]


EXPLAINERS: dict[str, Explainer] = {}


def register_explainer(
    name: str, *, label: str, needs: tuple[str, ...], description: str = ""
) -> Callable:
    """Declare an explainer. The registry is the list every surface offers."""

    def deco(fn: Callable[..., Explanation]) -> Callable[..., Explanation]:
        if name in EXPLAINERS:
            raise ValueError(f"explainer {name!r} is already registered")
        EXPLAINERS[name] = Explainer(name, label, needs, fn, description)
        return fn

    return deco


DEFAULT_EXPLAINER = "random_forest_shap"
FALLBACK_EXPLAINER = "random_forest_permutation"


def resolve(name: str) -> tuple[Explainer, str]:
    """The explainer to run for ``name``, and a note when it is not the one asked for.

    Only the default falls back (to permutation importance, which needs scikit-learn alone);
    an explainer asked for by name that cannot run is refused, so a script never gets a
    different method than it named without being told.

    Raises:
        ValueError: If ``name`` is unknown, or names an explainer whose modules are missing.
    """
    if name not in EXPLAINERS:
        raise ValueError(
            f"unknown explainer {name!r}; expected one of {', '.join(EXPLAINERS)}"
        )
    chosen = EXPLAINERS[name]
    if chosen.available():
        return chosen, ""
    if name == DEFAULT_EXPLAINER and EXPLAINERS[FALLBACK_EXPLAINER].available():
        return (
            EXPLAINERS[FALLBACK_EXPLAINER],
            f"{', '.join(chosen.missing())} is not installed, so the forest's "
            "importances are permutation importance, not SHAP",
        )
    raise ValueError(
        f"explainer {name!r} needs {', '.join(chosen.missing())}, which is not installed "
        "(pip install 'detail[analysis]')"
    )


# --------------------------------------------------------------------------- #
# Shared preparation and scoring
# --------------------------------------------------------------------------- #
@dataclass
class _Prepared:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    names: list[str]
    dropped: dict[str, str]
    rows_dropped: int


def prepare(X: np.ndarray, y: np.ndarray, groups, names: list[str]) -> _Prepared:
    """Drop mostly-missing measurements (named) and then rows with any gap left (counted)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    groups = np.asarray(groups, dtype=object)
    if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] != len(names):
        raise ValueError(
            f"X is {X.shape}, expected ({len(y)}, {len(names)}) to match y and names"
        )
    dropped: dict[str, str] = {}
    keep_cols = []
    for j, name in enumerate(names):
        column = X[:, j]
        missing = ~np.isfinite(column)
        if missing.mean() > MAX_MISSING_SHARE:
            dropped[name] = f"missing on {int(missing.sum())} of {len(column)} rows"
            continue
        finite = column[~missing]
        if len(finite) and np.ptp(finite) == 0:
            dropped[name] = "the same value on every row"
            continue
        keep_cols.append(j)
    X = X[:, keep_cols]
    kept_names = [names[j] for j in keep_cols]
    rows = np.isfinite(X).all(axis=1) if X.shape[1] else np.zeros(len(y), dtype=bool)
    return _Prepared(
        X[rows], y[rows], groups[rows], kept_names, dropped, int((~rows).sum())
    )


def _folds(y: np.ndarray, groups: np.ndarray, seed: int):
    """Held-out folds by recording when there are at least two; by row otherwise (and say so)."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    n_groups = len(set(groups.tolist()))
    if n_groups >= 2:
        splitter = GroupKFold(n_splits=min(5, n_groups))
        return list(splitter.split(np.zeros(len(y)), y, groups)), "recording"
    n_splits = max(2, min(5, int(min(y.sum(), (~y).sum()))))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y)), "row (only one recording)"


#: Why no held-out score exists when every fold has one class on a side.
NO_SCORABLE_FOLD = (
    "no held-out recording has both the cluster and the rest to score on — the cluster's "
    "members are all from one recording, so how well the measures find it on a recording "
    "they have not seen cannot be measured"
)


def _one_class(y: np.ndarray) -> bool:
    return bool(y.all() or (~y).all())


def _balanced(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tpr = float(np.mean(y_pred[y_true]))
    tnr = float(np.mean(~y_pred[~y_true]))
    return (tpr + tnr) / 2


def _cross_validated(make_model, X, y, folds) -> list[float | None]:
    """Held-out balanced accuracy per fold, ``None`` for a fold that cannot be scored: one
    class on its training side (nothing to learn) or on its held-out side (nothing to
    balance — its plain accuracy is one class's hit rate, which a model that has learned
    only "not this recording" passes)."""
    scores: list[float | None] = []
    for train, test in folds:
        if _one_class(y[train]) or _one_class(y[test]):
            scores.append(None)
            continue
        model = make_model()
        model.fit(X[train], y[train])
        scores.append(_balanced(y[test], model.predict(X[test]).astype(bool)))
    return scores


def _score(make_model, X, y, folds, seed) -> tuple[list[float], list[float]]:
    """Held-out balanced accuracy on each fold the real membership can be scored on, and the
    shuffled-membership floor **on those same folds**.

    Fold by fold, not by position: a shuffle spreads the cluster over every recording, so it
    can score a fold the real membership cannot, and averaging that in set the floor of
    a cluster from one recording at ~50% beside a "held-out" score that was one class's hit
    rate.
    """
    real = _cross_validated(make_model, X, y, folds)
    rng = np.random.default_rng(seed)
    floors = [
        _cross_validated(make_model, X, rng.permutation(y), folds)
        for _ in range(FLOOR_SHUFFLES)
    ]
    scored, floor = [], []
    for i, value in enumerate(real):
        shuffled = [f[i] for f in floors if f[i] is not None]
        if value is None or not shuffled:
            continue
        scored.append(value)
        floor.append(float(np.mean(shuffled)))
    return scored, floor


def _medians(X, y, j) -> tuple[float, float]:
    return float(np.median(X[y, j])), float(np.median(X[~y, j]))


def _direction_from_medians(cluster: float, comparison: float) -> int:
    if cluster > comparison:
        return 1
    if cluster < comparison:
        return -1
    return 0


def _rules(X, y, names, labels: dict[str, str], seed: int) -> list[Rule]:
    """The cluster's leaves of a depth-:data:`RULE_DEPTH` tree, read as conditions."""
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(
        max_depth=RULE_DEPTH,
        min_samples_leaf=max(5, len(y) // 20),
        class_weight="balanced",
        random_state=seed,
    ).fit(X, y)
    t = tree.tree_
    n_cluster = max(int(y.sum()), 1)
    leaf_of = tree.apply(X)
    out: list[Rule] = []

    def walk(node: int, conditions: list[str]) -> None:
        if t.children_left[node] == -1:
            rows = leaf_of == node
            if not rows.any():
                return
            precision = float(np.mean(y[rows]))
            if precision < RULE_MIN_PRECISION:
                return
            coverage = float(np.sum(y[rows]) / n_cluster)
            out.append(Rule(tuple(conditions), precision, coverage))
            return
        name = names[t.feature[node]]
        label = labels.get(name, name)
        threshold = float(t.threshold[node])
        walk(t.children_left[node], [*conditions, f"{label} <= {threshold:.3g}"])
        walk(t.children_right[node], [*conditions, f"{label} > {threshold:.3g}"])

    walk(0, [])
    out.sort(key=lambda r: r.precision * r.coverage, reverse=True)
    return out[:3]


def _explain(
    name: str,
    make_model,
    importances_of,
    X,
    y,
    groups,
    names: list[str],
    *,
    labels: dict[str, str] | None,
    seed: int,
    margin: float,
    note: str = "",
) -> Explanation:
    prepared = prepare(X, y, groups, names)
    X, y, groups = prepared.X, prepared.y, prepared.groups
    n_cluster, n_comparison = int(y.sum()), int((~y).sum())
    base = dict(
        explainer=name,
        margin=margin,
        n_cluster=n_cluster,
        n_comparison=n_comparison,
        measures_used=prepared.names,
        measures_dropped=prepared.dropped,
        rows_dropped=prepared.rows_dropped,
    )
    if min(n_cluster, n_comparison) < MIN_ROWS_PER_SIDE or not prepared.names:
        return Explanation(
            accuracy=float("nan"),
            floor=float("nan"),
            separated=False,
            importances=[],
            rules=[],
            held_out="",
            note=(
                f"too few complete rows to fit ({n_cluster} in the cluster, "
                f"{n_comparison} to compare, {len(prepared.names)} measures); "
                f"each side needs at least {MIN_ROWS_PER_SIDE}"
            ),
            **base,
        )
    folds, held_out = _folds(y, groups, seed)
    real, floor = _score(make_model, X, y, folds, seed)
    accuracy = float(np.mean(real)) if real else float("nan")
    floor_mean = float(np.mean(floor)) if floor else float("nan")
    unscored = "" if real else NO_SCORABLE_FOLD
    separated = bool(np.isfinite(accuracy) and accuracy - floor_mean >= margin)
    model = make_model()
    model.fit(X, y)
    raw = importances_of(model, X, y, folds, seed)
    importances = []
    for j, measure in enumerate(prepared.names):
        value, direction = raw[j]
        cluster_median, comparison_median = _medians(X, y, j)
        if direction is None:
            direction = _direction_from_medians(cluster_median, comparison_median)
        importances.append(
            Importance(
                measure,
                float(value),
                int(direction),
                cluster_median,
                comparison_median,
            )
        )
    importances.sort(key=lambda i: i.value, reverse=True)
    return Explanation(
        accuracy=accuracy,
        floor=floor_mean,
        separated=separated,
        importances=importances,
        rules=_rules(X, y, prepared.names, labels or {}, seed),
        held_out=held_out,
        note=note,
        fold_accuracy=[float(v) for v in real],
        fold_floor=[float(v) for v in floor],
        unscored=unscored,
        **base,
    )


def _forest(seed: int):
    from sklearn.ensemble import RandomForestClassifier

    # 100 trees and every core: the forest is fitted once per fold, three times more for the
    # shuffled floor, and once on everything — the fit has to sit inside a 30 s profile.
    return lambda: RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _shap_importances(model, X, y, folds, seed):
    """Mean |SHAP| per measurement for the cluster class, and its direction: the sign of how
    the measurement's value and its SHAP value move together."""
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # newer shap: (rows, features, classes)
        values = values[:, :, 1]
    out = []
    for j in range(X.shape[1]):
        column, contribution = X[:, j], values[:, j]
        if np.ptp(column) == 0 or np.ptp(contribution) == 0:
            direction = 0
        else:
            direction = int(np.sign(np.corrcoef(column, contribution)[0, 1]))
        out.append((float(np.mean(np.abs(contribution))), direction))
    return out


def _permutation_importances(model, X, y, folds, seed):
    """Held-out permutation importance (drop in balanced accuracy), averaged over the folds;
    direction from the medians."""
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    totals = np.zeros(X.shape[1])
    n = 0
    for train, test in folds:
        if y[train].all() or (~y[train]).all() or y[test].all() or (~y[test]).all():
            continue
        fitted = clone(model).fit(X[train], y[train])
        result = permutation_importance(
            fitted,
            X[test],
            y[test],
            scoring="balanced_accuracy",
            n_repeats=3,
            random_state=seed,
        )
        totals += np.maximum(result.importances_mean, 0.0)
        n += 1
    means = totals / max(n, 1)
    return [(float(v), None) for v in means]


@register_explainer(
    "random_forest_shap",
    label="Random forest + SHAP",
    needs=("sklearn", "shap"),
    description="The default. A forest of 100 trees, explained by mean |SHAP| per measure.",
)
def random_forest_shap(
    X, y, groups, names, *, labels=None, seed=0, margin=DEFAULT_MARGIN, note=""
) -> Explanation:
    return _explain(
        "random_forest_shap",
        _forest(seed),
        _shap_importances,
        X,
        y,
        groups,
        names,
        labels=labels,
        seed=seed,
        margin=margin,
        note=note,
    )


@register_explainer(
    "random_forest_permutation",
    label="Random forest + permutation importance",
    needs=("sklearn",),
    description="The same forest, scored by how much held-out accuracy drops when a "
    "measure is shuffled. Used when shap is not installed.",
)
def random_forest_permutation(
    X, y, groups, names, *, labels=None, seed=0, margin=DEFAULT_MARGIN, note=""
) -> Explanation:
    return _explain(
        "random_forest_permutation",
        _forest(seed),
        _permutation_importances,
        X,
        y,
        groups,
        names,
        labels=labels,
        seed=seed,
        margin=margin,
        note=note,
    )


def _logistic(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )


def _coefficients(model, X, y, folds, seed):
    coefs = model[-1].coef_[0]
    return [(float(abs(c)), int(np.sign(c))) for c in coefs]


@register_explainer(
    "logistic_regression",
    label="Logistic regression",
    needs=("sklearn",),
    description="A straight-line check on standardized measures. When it agrees with the "
    "forest, the story does not depend on the model.",
)
def logistic_regression(
    X, y, groups, names, *, labels=None, seed=0, margin=DEFAULT_MARGIN, note=""
) -> Explanation:
    return _explain(
        "logistic_regression",
        _logistic(seed),
        _coefficients,
        X,
        y,
        groups,
        names,
        labels=labels,
        seed=seed,
        margin=margin,
        note=note,
    )


def _tree(seed: int):
    from sklearn.tree import DecisionTreeClassifier

    return lambda: DecisionTreeClassifier(
        max_depth=RULE_DEPTH, class_weight="balanced", random_state=seed
    )


def _tree_importances(model, X, y, folds, seed):
    return [(float(v), None) for v in model.feature_importances_]


@register_explainer(
    "decision_tree",
    label="Decision tree only",
    needs=("sklearn",),
    description="One shallow tree: the plain rules, scored on their own. The fastest.",
)
def decision_tree(
    X, y, groups, names, *, labels=None, seed=0, margin=DEFAULT_MARGIN, note=""
) -> Explanation:
    return _explain(
        "decision_tree",
        _tree(seed),
        _tree_importances,
        X,
        y,
        groups,
        names,
        labels=labels,
        seed=seed,
        margin=margin,
        note=note,
    )


def explain(
    X,
    y,
    groups,
    names: list[str],
    *,
    explainer: str = DEFAULT_EXPLAINER,
    labels: dict[str, str] | None = None,
    seed: int = 0,
    margin: float = DEFAULT_MARGIN,
) -> Explanation:
    """Fit ``explainer`` (resolved, so the default may fall back — the result's ``note`` and
    ``explainer`` then say which ran) to the cluster rows (``y`` True) against the rest.

    Raises:
        ValueError: As :func:`resolve`.
    """
    chosen, note = resolve(explainer)
    return chosen.fit(
        X, y, groups, names, labels=labels, seed=seed, margin=margin, note=note
    )
