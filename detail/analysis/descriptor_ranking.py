"""Which measurements are worth looking at — for a run, and for one thing inside it.

`analysis/descriptors.py` computes a value per instance. This ranks those values, because a
panel of nine equally-weighted plots is a panel with no answer in it: measured on a real
36-cluster run, one measurement carried 0.66 of the between-cluster variance and the next
carried 0.21, and the median cluster differed non-trivially on **2 of 9**.

Pure numpy. No zarr, no paths, no Qt — the same contract as `descriptors.py`, so every statistic
here is testable over arrays and reachable from a surface that is not the GUI.

**Two rankings, because there are two questions and they disagree.**

- :func:`separation` asks *of the run*: how much of this measurement's variance does the
  partition explain? Constant across selections — it is a property of the clustering, not of
  whichever cluster is open — so it answers "which axes matter here at all".
- :func:`rank_selection` asks *of one cluster*: how far is it from everything else? On the same
  run, `duration` ranked last for the whole run and was a real effect on several individual
  clusters.

**Why Cliff's delta and not a percentile.** The obvious statistic — what fraction of the run sits
below this cluster's median — is not tie-safe, and two of these measurements are small integer
counts with a dominant mode. `100 * mean(rest < median)` reported the **0th percentile for all 36
clusters** on `postures`, which is both wrong and unfalsifiable by eye: an extreme rank looks like
a finding. :func:`cliffs_delta` splits ties at half weight, so a measurement whose median *is* the
mode lands near zero, where it belongs. :func:`midrank_percentile` is the same correction applied
to the number the table prints, so the two columns cannot disagree.

**Why not `compare.py`'s ranking.** §24 ranks measures by ``|log ratio|`` of two groups' centres
(`compare.Contrast.magnitude`) and that is right there: it obeys the recorded rule that a ranking
needs a unit-free scale, and §24 deliberately fits no variance model. It is the wrong statistic
*here* for two reasons — it is undefined at zero or negative values, and `tortuosity` reaches
zero; and it compares centres only, so a cluster shifted a little with heavy overlap ranks level
with one that is cleanly separated. Separation is the entire question a person naming a cluster is
asking. Both modules obey the same lesson; they are not the same measurement.

**A statistic is withheld when the data could not have produced a different answer** — but only
the statistic that could not. :func:`separation` asks how much of a measurement's variance a
*partition* explains, which needs two groups, so :func:`separation_withheld` says why it is
missing rather than leaving a blank (a blank reads as "nothing differed"). That is a fact about
the run line and nothing else.

:func:`rank_selection` is separate, deliberately. It needs a non-empty *complement*, which is not
the same condition — and withholding on the wrong one made the band dead where it mattered most:
four of five real query runs have a single behavior, so comparing it against the other behaviors
left nothing and the panel drew nothing at all. Comparing it against the recording it was found in
is the more useful question anyway ("faster than most of what this animal does"). So a caller may
supply background rows carrying **no group**, spelled ``None``: they count in the comparison and
stay out of the partition.
"""

from dataclasses import dataclass

import numpy as np

#: Cliff's conventional medium-effect line. Below it, a difference is real but not worth a plot;
#: above it the measurement goes on the shortlist. Named here rather than written into a surface,
#: because it decides how many plots a panel draws and is the first thing anyone will want to tune.
STRONG = 0.33


@dataclass(frozen=True)
class Ranked:
    """One measurement's standing for one selection.

    Carried as an object rather than a tuple because the table, the caption and the drawing order
    all read the same three facts, and three surfaces destructuring three tuples is how a column
    ends up under the wrong header.
    """

    name: str
    value: float
    #: Cliff's delta against everything not in the selection: -1 entirely below, +1 entirely above.
    delta: float
    #: Where the selection's median sits in the rest of the run, 0-100, ties split.
    percentile: float

    @property
    def strong(self) -> bool:
        return abs(self.delta) >= STRONG

    @property
    def above(self) -> bool:
        return self.delta > 0


def _finite(values) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    return out[np.isfinite(out)]


def cliffs_delta(a, b) -> float:
    """``P(a > b) - P(a < b)`` — how far ``a`` sits from ``b``, on a scale with no units.

    ``-1`` when every value of ``a`` is below every value of ``b``, ``+1`` when every one is
    above, ``0`` when the two are indistinguishable. Computed from midranks rather than by
    counting pairs, so it costs a sort rather than ``len(a) * len(b)`` comparisons — the panel
    runs this over every measurement on every selection change.

    Ties count half, which is the whole reason this is here rather than a percentile: on a
    measurement that is a small integer count, most pairs *are* ties.

    ``NaN`` when either side is empty, which is a real state — a descriptor that could not be
    computed for any member of this cluster has no standing, and zero would say it had one.
    """
    a, b = _finite(a), _finite(b)
    if not len(a) or not len(b):
        return float("nan")
    both = np.concatenate([a, b])
    # Midranks: every member of a tied run takes the mean of the ranks that run spans, which is
    # two searchsorted lookups over the sorted values rather than a Python pass per tie group.
    ordered = np.sort(both)
    first = np.searchsorted(ordered, both, side="left")
    last = np.searchsorted(ordered, both, side="right")
    ranks = 0.5 * (first + last + 1)
    rank_sum = ranks[: len(a)].sum()
    # The Mann-Whitney U of `a` over `b`, rescaled from [0, 1] to [-1, +1].
    u = rank_sum - len(a) * (len(a) + 1) / 2.0
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def midrank_percentile(value: float, rest) -> float:
    """Where ``value`` sits in ``rest``, 0-100, with ties split at half weight.

    The readable form of :func:`cliffs_delta`, and corrected the same way for the same reason: on
    a discrete measurement whose median is also its mode, ``mean(rest < value)`` counts almost
    nothing and prints "above 0% of bouts" beside a delta of ``+0.01``.
    """
    rest = _finite(rest)
    if not len(rest) or not np.isfinite(value):
        return float("nan")
    below = float((rest < value).mean())
    tied = float((rest == value).mean())
    return 100.0 * (below + 0.5 * tied)


def _labeled(labels) -> np.ndarray:
    """Which rows belong to a group. A ``None`` label is background — measured, drawn and
    compared against, but not part of any partition."""
    labels = np.asarray(labels, dtype=object)
    return np.array([label is not None for label in labels.tolist()], dtype=bool)


def separation(values, labels) -> float:
    """The share of a measurement's variance the partition explains, 0-1.

    A property of the *run*: it does not move when the selection does, which is what makes it the
    honest answer to "which measurements is this clustering actually about". ``NaN`` when there is
    nothing to explain — fewer than two groups, or every value identical.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    ok = np.isfinite(values)
    values, labels = values[ok], labels[ok]
    groups = np.unique(labels)
    if len(values) < 2 or len(groups) < 2:
        return float("nan")
    grand = values.mean()
    total = float(((values - grand) ** 2).sum())
    if total == 0:
        return float("nan")
    between = 0.0
    for group in groups:
        rows = labels == group
        between += rows.sum() * (values[rows].mean() - grand) ** 2
    return float(between / total)


def separation_withheld(labels, noun: str = "group") -> str | None:
    """Why the **run-level** ranking has no answer, or ``None`` when it does.

    Separation is a property of a partition, so one group has none — there is nothing for the
    grouping to explain. Reported as a sentence rather than a blank, which reads as "nothing
    differed".

    It says nothing about whether a *selection* can be ranked: that needs a non-empty complement,
    which background rows supply even when the run has one group.
    """
    labels = np.asarray(labels, dtype=object)
    groups = {str(label) for label in labels[_labeled(labels)].tolist()}
    if len(groups) >= 2:
        return None
    only = next(iter(groups), "")
    named = f" ({only})" if only else ""
    return (
        f"This run has one {noun}{named}, so there is nothing for the grouping to explain. "
        f"The ranking below is against the rest of the recording."
    )


def rank_selection(values_by_name: dict, labels, selected) -> list[Ranked]:
    """Every measurement, most distinctive first, for the members carrying ``selected``.

    ``values_by_name[name][i]`` is row ``i``'s value and ``labels[i]`` its group, so the caller
    measures the run once and every selection after that is a mask. A row whose label is ``None``
    is background: part of what ``selected`` is compared *against*, never selected itself.

    A measurement with no finite value on either side is dropped rather than ranked ``NaN``: it
    has no standing to report, and sorting it anywhere is an invention.
    """
    labels = np.asarray(labels, dtype=object)
    mine = np.array(
        [str(label) == str(selected) for label in labels.tolist()], dtype=bool
    )
    out: list[Ranked] = []
    for name, values in values_by_name.items():
        values = np.asarray(values, dtype=np.float64)
        here, rest = _finite(values[mine]), _finite(values[~mine])
        if not len(here) or not len(rest):
            continue
        median = float(np.median(here))
        out.append(
            Ranked(
                name=name,
                value=median,
                delta=cliffs_delta(here, rest),
                percentile=midrank_percentile(median, rest),
            )
        )
    out.sort(key=lambda r: -abs(r.delta))
    return out


def rank_run(values_by_name: dict, labels) -> list[tuple[str, float]]:
    """Every measurement by how much the partition separates it, most first.

    The run-level companion to :func:`rank_selection`. Measurements it cannot score are dropped,
    for the same reason.
    """
    labels = np.asarray(labels, dtype=object)
    keep = _labeled(labels)
    scored = [
        (name, separation(np.asarray(values, dtype=np.float64)[keep], labels[keep]))
        for name, values in values_by_name.items()
    ]
    scored = [(name, value) for name, value in scored if np.isfinite(value)]
    scored.sort(key=lambda pair: -pair[1])
    return scored
