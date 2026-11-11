"""Comparison — group statistics across recording metadata (§24).

The question both research tasks exist to serve: *does this differ between these groups?* Takes the
§23 measures table, joins each row's recording to its grouping variables (§11), collapses through a
declared aggregation path, and summarizes by whatever columns the question needs.

Pure functions — no executor, no Qt, no disk. The record and the export tree sit above this.

**Scope is descriptive, deliberately.** Effect sizes, spreads, and every observation. No tests, no
multiple-comparison corrections, no models: behavioral counts are overdispersed, sessions within an
animal are not independent, and getting inference right needs choices this tool has no business
making on a user's behalf. What it owes them instead is long-form rows with their factors attached,
which is exactly what a mixed model in R or statsmodels wants as input.

**The aggregation path is the whole ballgame.** One animal contributes four sessions and another
one. Pool recordings and the first animal votes four times, so the group mean tilts toward whichever
animal happens to have more sessions; average within animal first and each votes once, at a much
smaller n. Both are legitimate and they answer different questions — so the path is a **required
argument with no default**, ``[]`` is a choice rather than an absence, and it rides on every output
row so a table or a figure can never be read without knowing what one point is.

**Absence is not zero, one layer up.** §23 emits explicit zeros for vocabulary labels *inside* a
recording's coverage and nothing at all outside it, so a recording missing from the measures table
was never analyzed. It must not quietly shrink a group's n — hence ``coverage`` beside ``n`` on
every group summary.
"""

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# The group-by keys that are not metadata columns. ``subject`` and ``window_id`` come off the
# measures row itself; everything else is looked up in metadata.csv.
ROW_KEYS = ("subject", "window_id")

# What identifies the thing being measured, independent of who was measured. A pairwise measure
# keys on both labels (§23), which needs no branch here — the key is a tuple either way.
#
# ``window_id`` belongs here, not with the group keys: "rearing rate in the first third" is a
# different measurement from "rearing rate in the last third", the same way "rate" differs from
# "total_time". Left out, a group's mean averages the first third of every session together with
# the last third — silently answering the opposite of what a windowed comparison was asked. It
# moves to the group key when the caller groups **by** window, which is how "early versus late"
# becomes the same operation as "drug A versus B".
MEASURE_KEYS = ("measure", "label", "label_to", "window_id")

STATS_COLUMNS = [
    "group",
    "observation",
    "aggregate_to",
    "measure",
    "label",
    "label_to",
    "window_id",
    "value",
    "n_underlying",
]


class ComparisonError(ValueError):
    """A comparison asked for something the data or the metadata cannot answer."""


@dataclass(frozen=True)
class Observation:
    """One unit of analysis after the aggregation path has been applied.

    ``n_underlying`` is how many measure rows were averaged into ``value`` — the sample-size trail
    that says whether an animal-level point rests on four sessions or one. Losing it is how an
    unbalanced design becomes invisible.
    """

    key: tuple
    group: tuple
    measure_key: tuple
    value: float
    n_underlying: int


@dataclass(frozen=True)
class GroupSummary:
    """One group's numbers for one measure, carrying what makes them readable.

    ``values`` is kept whole rather than reduced to a centre and a spread: at the cohort sizes this
    project works at, a bar and an error bar hide the entire dataset, and the points *are* the
    result.
    """

    group: tuple
    measure_key: tuple
    values: list[float]
    coverage: int

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.values) if self.values else None

    @property
    def median(self) -> float | None:
        return statistics.median(self.values) if self.values else None

    @property
    def sd(self) -> float | None:
        """Population sd, and ``None`` at n = 1 — not zero, which reads as "no variation observed"
        when the truth is "variation not observable"."""
        return statistics.pstdev(self.values) if len(self.values) > 1 else None


def _factors(row: dict, metadata: dict) -> dict:
    """A measure row's grouping variables: its recording's metadata, plus the row's own keys.

    ``subject`` and ``window_id`` are group-by keys like any other even though they come off the
    row rather than out of ``metadata.csv`` — which is what makes "first third versus last third"
    the same operation as "drug A versus B" instead of a separate feature.
    """
    out = dict(metadata.get(row["recording_id"], {}))
    out["recording_id"] = row["recording_id"]
    for key in ROW_KEYS:
        out[key] = row[key]
    return out


def _require_columns(names: Sequence[str], metadata: dict, what: str) -> None:
    """Fail on an unknown grouping column, naming what is available.

    A silent miss here groups everything into one bucket labelled with an empty string, which
    renders as a perfectly plausible single-group comparison.
    """
    known = {"recording_id", *ROW_KEYS}
    for row in metadata.values():
        known |= set(row)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ComparisonError(
            f"{what} names {unknown}, which is not a metadata column or a row key. "
            f"Available: {sorted(known)}"
        )


def measure_key_columns(group_by: Sequence[str]) -> tuple[str, ...]:
    """The ``MEASURE_KEYS`` still identifying the measurement, given what is grouped on."""
    return tuple(k for k in MEASURE_KEYS if k not in group_by)


def observations(
    rows: Sequence[dict],
    metadata: dict,
    *,
    group_by: Sequence[str],
    aggregate_to: Sequence[str],
    recordings: Sequence[str] | None = None,
    predicate: Callable[[dict], bool] | None = None,
) -> list[Observation]:
    """Measure rows → one observation per unit of analysis.

    ``aggregate_to`` is the path: the columns to collapse through **before** anything is grouped.
    ``[]`` leaves each measure row its own observation. It has no default here on purpose — the
    caller states it, because a tool that picks silently picks wrong roughly half the time and the
    output looks identical either way.

    ``recordings`` and ``predicate`` filter, in that order. A row whose value is blank (a measure
    that declined to answer — a point behavior's duration, §23) is dropped rather than coerced,
    since a coerced blank enters a group mean as a real observation of zero.
    """
    _require_columns(group_by, metadata, "group_by")
    _require_columns(aggregate_to, metadata, "aggregate_to")
    keep = set(recordings) if recordings is not None else None

    buckets: dict[tuple, list[float]] = {}
    identity: dict[tuple, tuple] = {}
    for row in rows:
        if keep is not None and row["recording_id"] not in keep:
            continue
        if row["value"] == "" or row["value"] is None:
            continue
        factors = _factors(row, metadata)
        if predicate is not None and not predicate(factors):
            continue
        group = tuple(str(factors.get(k, "")) for k in group_by)
        # The observation's identity. With a path, it is the path's columns. With an **empty**
        # path it is the measure row itself — ``(recording, subject, window)`` — because "[] means
        # no aggregation" has to mean one observation per row, not one observation per group. An
        # empty tuple would collapse the whole group into a single point and report n = 1 for a
        # cohort of any size, which looks like a legitimately small study.
        unit = (
            tuple(str(factors.get(k, "")) for k in aggregate_to)
            if aggregate_to
            else (row["recording_id"], str(row["subject"]), row["window_id"])
        )
        measure_key = tuple(row[k] for k in measure_key_columns(group_by))
        key = (group, unit, measure_key)
        buckets.setdefault(key, []).append(float(row["value"]))
        identity[key] = unit

    out: list[Observation] = []
    for (group, unit, measure_key), values in buckets.items():
        out.append(
            Observation(
                key=unit,
                group=group,
                measure_key=measure_key,
                value=statistics.fmean(values),
                n_underlying=len(values),
            )
        )
    return out


def coverage_by_group(obs) -> dict[tuple, int]:
    """How many distinct observations each group contributed, across **all** its measures.

    Beside a summary's ``n`` — the values *that measure* returned — this says how many units the
    group actually has, so ``coverage > n`` means the measure declined for some of them: a point
    behavior's duration, a rate in a window with no observed time (§23 emits no value rather than
    a zero, and a zero would enter a group mean as a real observation of "never happened").

    **What it cannot see, and no longer has to.** It is derived from the observations that
    arrived, so a unit that never reached the table at all is missing from both ``n`` and
    ``coverage`` identically — the column could not detect the case its first docstring claimed
    it existed for. That case is now refused at the door instead: `run_comparison` raises when
    the recording filter names anything the artifact does not cover, rather than narrowing to the
    intersection. Eligible and arrived agree by construction, which is what makes the reading
    above true.

    Shared, because every surface that renders a group table has to arrive at the same number —
    one that recomputes it from what it happens to have loaded reports ``coverage == n``
    everywhere, which is the reading this column exists to make impossible.
    """
    groups: dict[tuple, int] = {}
    for row in obs:
        groups.setdefault(row.group, 0)
    for group in groups:
        groups[group] = len({o.key for o in obs if o.group == group})
    return groups


def summarize(
    obs: Sequence[Observation], *, coverage_by_group: dict | None = None
) -> list[GroupSummary]:
    """Observations → one summary per ``(group, measure)``.

    ``coverage_by_group`` is how many units the group has across all measures — see that
    function. Without it ``coverage`` is recomputed per measure and equals ``n`` everywhere, so a
    measure that declined for half a group reads exactly like one that answered for all of it
    (§23's absence-versus-zero rule, one layer up).
    """
    buckets: dict[tuple, list[float]] = {}
    for row in obs:
        buckets.setdefault((row.group, row.measure_key), []).append(row.value)
    coverage_by_group = coverage_by_group or {}
    return [
        GroupSummary(
            group=group,
            measure_key=measure_key,
            values=sorted(values),
            coverage=coverage_by_group.get(group, len(values)),
        )
        for (group, measure_key), values in buckets.items()
    ]


@dataclass(frozen=True)
class Contrast:
    """One measure's difference between two groups.

    Raw difference and ratio, **not** a standardized effect: dividing by a pooled spread is a
    variance model, and a variance model is the inference §24 declines to do. The ratio is ``None``
    rather than infinite when the reference centre is zero — "infinitely more common" is not a
    finding, it is a division.
    """

    measure_key: tuple
    a: GroupSummary
    b: GroupSummary
    difference: float | None
    ratio: float | None

    @property
    def magnitude(self) -> float | None:
        """``|log ratio|`` — how big the change is, on a scale with no units.

        This is what measures are **ranked** by. A raw difference cannot order a set of measures
        that do not share a unit: a real comparison put a 45-count change (rearing, 92 → 47)
        beside a 2.5e11-nanosecond one on a single linear axis, and the count rendered as a bar
        of zero width. Nanoseconds win every such contest, whatever the biology says.

        Logarithmic rather than percent because a ranking must not depend on which group is
        written first. 92 → 47 and 47 → 92 are the same size of change and must rank together;
        as percentages they are −49% and +96%. Percent is what gets *displayed* — see
        :attr:`percent_change`.

        It is not a standardized effect: nothing is divided by a spread, so §24's refusal to fit
        a variance model stands.

        ``0.0`` when both centres are zero — no change, and a real case (`unnamed_time` is zero
        throughout a query run). ``None`` when exactly one centre is zero, which is
        :attr:`unbounded`, or when a centre is missing.
        """
        lo, hi = self.a.mean, self.b.mean
        if lo is None or hi is None:
            return None
        if lo == 0 and hi == 0:
            return 0.0
        if lo <= 0 or hi <= 0:
            return None
        return abs(math.log(hi / lo))

    @property
    def unbounded(self) -> bool:
        """Whether the fold change is undefined because one side is zero and the other is not.

        A behavior that appears or vanishes entirely is the largest relative change there is, and
        it has no finite magnitude to rank or to test. Reported as itself rather than dropped or
        given an invented number.
        """
        lo, hi = self.a.mean, self.b.mean
        if lo is None or hi is None:
            return False
        return (lo == 0) != (hi == 0)

    @property
    def percent_change(self) -> float | None:
        """``(b - a) / a`` as a percentage — the readable form of :attr:`magnitude`."""
        lo, hi = self.a.mean, self.b.mean
        if lo is None or hi is None or lo == 0:
            return None
        return (hi - lo) / lo * 100.0


def _rank_key(item: Contrast) -> float:
    """Sort key: unbounded changes first, then by magnitude, then the incomputable."""
    if item.unbounded:
        return float("inf")
    return item.magnitude if item.magnitude is not None else -1.0


def contrast(a: GroupSummary, b: GroupSummary) -> Contrast:
    """``b`` against ``a`` — difference and ratio of the central values."""
    if a.measure_key != b.measure_key:
        raise ComparisonError(
            f"cannot contrast different measures: {a.measure_key} vs {b.measure_key}"
        )
    lo, hi = a.mean, b.mean
    difference = None if lo is None or hi is None else hi - lo
    ratio = None if not lo or hi is None else hi / lo
    return Contrast(a.measure_key, a, b, difference, ratio)


@dataclass(frozen=True)
class Ranking:
    """Measures ranked by how much they differ between two groups, **with the floor they must
    clear**.

    ``floor`` is the **95th percentile** of the differences the same procedure produces from
    shuffled group labels. It is not decoration: with tens of clusters and a handful of animals,
    ranking by difference *always* produces a most-different cluster whether or not anything is
    there — and this screen gets run on every comparison, which is the condition under which a
    spurious top hit becomes a finding.

    **A percentile rather than the maximum, and the reason is not taste.** Every permutation
    distribution contains the observed labelling — shuffle enough times and one draw reproduces it
    — so the maximum is always at least the real effect and nothing can ever clear it. The floor
    would be unfalsifiable by construction.

    **A small cohort cannot clear this floor at all, and that is the honest answer.** With three
    observations per group there are only twenty distinct splits, so the true labelling accounts for
    10% of them and no difference reaches the 95th percentile however large it is. That is not a
    defect of the floor; it is the floor reporting that this cohort cannot distinguish any effect
    from chance, which a bare ranking would never say.

    Constructed only by :func:`rank_by_difference`, which computes both in one pass — so a ranking
    cannot exist without its floor **or without saying why it has none**. ``floor`` is ``None``
    where the permutation had nothing to move (one observation per group: every relabelling leaves
    every fold change unchanged, so the null's maximum *is* the observed maximum and `>` can never
    be satisfied), and :attr:`no_floor_reason` carries the sentence every surface prints. A blank
    column reads as "nothing differed" when the truth is "this design cannot tell".
    """

    contrasts: list[Contrast]
    floor: float | None
    draws: int
    seed: int
    floor_distribution: list[float] = field(default_factory=list)
    #: Why there is no floor, when there is none — so a surface can say what happened instead of
    #: leaving a blank that reads as "nothing differed".
    no_floor_reason: str | None = None

    def clears_floor(self, item: Contrast) -> bool:
        """Whether this measure's change exceeds the floor.

        ``False`` also when there is no floor and when the fold change is unbounded — both are
        "not shown to clear", not "shown not to". A surface must read ``no_floor_reason`` rather
        than render this as a verdict.
        """
        # `unbounded` is *implied* by `magnitude is None` -- one side zero and the other not
        # takes the `lo <= 0 or hi <= 0` branch. It is named anyway because it is the case a
        # reader comes here about, and because the converse does not hold: a missing centre is
        # also `None` and is not unbounded. Do not read a test that mutates it away as coverage.
        if self.floor is None or item.unbounded or item.magnitude is None:
            return False
        return item.magnitude > self.floor


def _largest_magnitude(obs: Sequence[Observation], groups: tuple) -> float:
    """The biggest ``|log ratio|`` over all measures, for one labelling.

    **The max across measures is the multiple-comparisons correction, not an oversight.** Rank
    twenty measures and one of them is the largest whether or not anything is there; taking the
    largest under shuffling too is what makes the floor a family-wise threshold rather than a
    per-measure one. That is worth stating because it looks like a bug and removing it would
    quietly drop the correction.

    On ``|log ratio|`` rather than the raw difference, because the raw version made the floor a
    *nanosecond* quantity that a count measure could never reach at any sample size. Unit-free
    ranking needs a unit-free null, or the floor is not the floor of the thing being ranked.

    A measure whose fold change is unbounded under this labelling contributes nothing — there is
    no finite value to take a maximum with.
    """
    by_key: dict[tuple, dict[tuple, list[float]]] = {}
    for row in obs:
        by_key.setdefault(row.measure_key, {}).setdefault(row.group, []).append(
            row.value
        )
    best = 0.0
    for per_group in by_key.values():
        a, b = per_group.get(groups[0]), per_group.get(groups[1])
        if not a or not b:
            continue
        lo, hi = statistics.fmean(a), statistics.fmean(b)
        if lo <= 0 or hi <= 0:
            continue
        best = max(best, abs(math.log(hi / lo)))
    return best


def _vacuous_permutation(obs: Sequence[Observation], groups: tuple) -> str | None:
    """Why relabelling cannot test anything here, or ``None`` when it can.

    :class:`Ranking` states the principle — a floor that the observed value cannot exceed "would
    be unfalsifiable by construction" — but nothing checked for the data shape that guarantees it.

    **The shape is one observation per group.** A measure with a single value on each side has a
    magnitude that a swap leaves unchanged, so no relabelling can produce a larger one; the null's
    maximum *is* the observed maximum, and `clears_floor` compares with `>`. On a real
    two-recording comparison this came out as a floor equal to the observed maximum to six
    significant figures, with 0 of 19 measures clearing it. Not unlikely — impossible.

    Checked on the observation counts rather than on the resulting distribution, because the
    distribution alone cannot tell the two cases apart: a clean six-per-group separation also
    produces only a handful of distinct null values, and there the floor works perfectly well
    (the observed 3.91 sits well above a null that tops out at 1.52). What distinguishes them is
    whether the permutation had anything to move.

    This is arithmetic on the data's shape, not a judgement about the study.
    """
    per_measure: dict[tuple, dict[tuple, int]] = {}
    for row in obs:
        counts = per_measure.setdefault(row.measure_key, {})
        counts[row.group] = counts.get(row.group, 0) + 1
    if not per_measure:
        return "there are no observations to relabel"
    movable = any(
        counts.get(groups[0], 0) > 1 or counts.get(groups[1], 0) > 1
        for counts in per_measure.values()
    )
    if movable:
        return None
    return (
        "every measure has at most one observation per group, so relabelling cannot change any "
        "measure's fold change and no measure could exceed the floor"
    )


def rank_by_difference(
    obs: Sequence[Observation],
    groups: tuple,
    *,
    draws: int = 200,
    seed: int = 0,
) -> Ranking:
    """Rank every measure by ``|log ratio|`` between two groups, and compute the floor beside it.

    **Not by raw difference**, which the name still carries for the callers and records that use
    it: a set of measures that do not share a unit has no common linear scale, and ranking by
    difference put a 45-count change beside a 2.5e11-nanosecond one and drew the count as a bar of
    zero width. See :attr:`Contrast.magnitude`.

    The floor is the 95th percentile of the shuffled magnitudes rather than their maximum — see
    :class:`Ranking` for why the maximum is unfalsifiable — and is ``None``, with a reason, where
    the permutation could not have produced a different answer.

    **The permutation shuffles whole observations, which is the point.** An observation is one unit
    of the declared aggregation path — an animal, when the path aggregates to animal — so shuffling
    here preserves exactly the structure the path exists to impose. Shuffling the underlying
    *recordings* instead would break an animal into four independent draws, inflate the effective
    sample size, and make the floor far easier to clear than the real ranking had to work for.
    """
    import random

    if len(groups) != 2:
        raise ComparisonError(f"a contrast needs exactly two groups, got {groups}")
    summaries = {(s.group, s.measure_key): s for s in summarize(obs)}
    contrasts = []
    for measure_key in sorted({row.measure_key for row in obs}):
        a = summaries.get((groups[0], measure_key))
        b = summaries.get((groups[1], measure_key))
        if a is None or b is None:
            continue
        contrasts.append(contrast(a, b))
    contrasts.sort(key=_rank_key, reverse=True)

    rng = random.Random(seed)
    labels = [row.group for row in obs]
    distribution: list[float] = []
    for _ in range(draws):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        relabelled = [
            Observation(o.key, g, o.measure_key, o.value, o.n_underlying)
            for o, g in zip(obs, shuffled)
        ]
        distribution.append(_largest_magnitude(relabelled, groups))
    ordered = sorted(distribution)
    # A floor is reported only when the permutation had something to move. Withheld with its
    # reason rather than drawn as a threshold nothing could reach.
    reason = _vacuous_permutation(obs, groups)
    floor = (
        None
        if reason or not ordered
        else ordered[min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)]
    )
    return Ranking(contrasts, floor, draws, seed, ordered, reason)


def stats_rows(
    obs: Sequence[Observation],
    *,
    aggregate_to: Sequence[str],
    group_by: Sequence[str] = (),
) -> list[dict]:
    """Long-form rows — one per observation, with its factors — the source of truth an export
    reads and a notebook reshapes.

    ``aggregate_to`` is stamped on **every** row. A table or a figure built from these is
    unreadable without it: "one point is one animal" and "one point is one session" are different
    claims about the same picture, and nothing else on the row distinguishes them.
    """
    path = "|".join(aggregate_to)
    columns = measure_key_columns(group_by)
    out = []
    for row in obs:
        keyed = dict(zip(columns, row.measure_key))
        out.append(
            {
                "group": "|".join(row.group),
                "observation": "|".join(row.key),
                "aggregate_to": path,
                "measure": keyed.get("measure", ""),
                "label": keyed.get("label", ""),
                "label_to": keyed.get("label_to", ""),
                "window_id": keyed.get("window_id", ""),
                "value": row.value,
                "n_underlying": row.n_underlying,
            }
        )
    return out
