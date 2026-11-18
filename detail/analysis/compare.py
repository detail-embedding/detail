"""Comparison — group statistics across recording metadata (§24).

The question both research tasks exist to serve: *does this differ between these groups?* Takes the
§23 measures table, joins each row's recording to its grouping variables (§11), collapses through a
declared aggregation path, and summarizes by whatever columns the question needs.

Pure functions — no executor, no Qt, no disk. The record and the export tree sit above this.

**Scope is descriptive, deliberately.** Effect sizes, spreads, and every observation. No models:
behavioral counts are overdispersed, sessions within an animal are not independent, and getting
inference right needs choices this tool has no business making on a user's behalf. What it owes
them instead is long-form rows with their factors attached, which is exactly what a mixed model in
R or statsmodels wants as input. The one inference it does make is the permutation floor a ranking
ships with (:class:`Ranking`), and the numbers that floor already implies: each item's
**family-wise p** (the share of the null's draws whose largest effect reached it — the floor read
as a p), its **own p** (the same share against its own null, ignoring the rest of the family) and
the Benjamini–Hochberg **q** over the family's own p's, so a cohort too small to clear a
family-wise floor can still say which items are consistent on their own.

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

**The design is resolved from the chosen column, never configured** (§24 *Designs*). A column
constant within each unit is compared *between* units; a column that varies within units is a
*paired* contrast of two of its levels, a *trend* over its numeric values or a *profile* over three
or more named ones (the largest difference between any two), and the permutation behind the floor
is the one that design allows — sign flips or within-unit shuffles, never a relabelling across
units that would break the pairing the design exists to use. The user chooses the variables;
:func:`resolve_design` says what they imply, in one sentence every surface prints.

**A within-unit ranking scores by evidence, not by size** (§24 *The statistic*). The number a
measure is ranked by, and the floor is taken over, is its **moderated t**: the mean per-unit
effect over its spread, ``mean / ((sd + s0) / √n)``, with ``s0`` the family's median spread so a
measure whose units happen to agree exactly cannot own the floor with a zero denominator. The
effect itself (the mean log ratio, the mean ρ) stays what every surface prints; the t is what
decides the order and the verdict. On the effect alone the family-wise floor belongs to the
sparsest measures — a cluster a few animals barely touch swings ±4 log units under any sign — and
a cluster eight of eight animals agree on at 12× sat under a 16.8× floor. ``statistic`` on
:func:`rank_by_design` and on the record chooses; a record written before the field existed
ranks on the effect, as it always did.
"""

import dataclasses
import math
import statistics
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from detail.analysis.cohort import cell
from detail.analysis.cohort import left_out
from detail.analysis.cohort import missing

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
    "label_display",
    "label_is_remainder",
    "window_id",
    "value",
    "n_underlying",
    "denominator_ns",
]

#: The designs a comparison resolves to (§24 *Designs*). The values are what the record stores.
BETWEEN, PAIRED, TREND, PROFILE = "between", "paired", "trend", "profile"
DESIGNS = (BETWEEN, PAIRED, TREND, PROFILE)

#: A sign-flip null with this many units or fewer is enumerated exactly (2^n configurations)
#: rather than sampled: at eight units the sampled 95th percentile moved three nodes in and out
#: of a kept set with its seed, and 1,024 configurations cost less than 1,000 draws.
EXACT_FLIPS_MAX_UNITS = 10

#: The measure whose per-unit presence decides whether a unit counts toward ``min_units``.
PRESENCE_MEASURE = "total_time"

#: The floor a ranking ships with is the null's 95th percentile: a family-wise level of 5%.
FLOOR_ALPHA = 0.05
#: The strictness levels a screen is read at (§22 *The difference screen*): the family-wise
#: level the floor is taken at, :data:`FLOOR_ALPHA` among them. A view over a saved ranking,
#: never a run parameter — what the record kept is kept at :data:`FLOOR_ALPHA`.
STRICTNESS_LEVELS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)

#: What a within-unit ranking scores each measure by (§24 *The statistic*): the mean per-unit
#: effect studentized — the moderated t, ``mean / ((sd + s0) / √n)``, ``s0`` the family's
#: median spread — or the mean effect itself. The config's default is the t; a record written
#: before the field existed carries neither and ranks on the effect. A between design has no
#: per-unit effects to studentize and ranks on the effect whatever was asked.
STATISTIC_T = "moderated_t"
STATISTIC_EFFECT = "effect"
STATISTICS = (STATISTIC_T, STATISTIC_EFFECT)

#: How close, relative to the observed value, a null draw must come to count as reaching it.
#: A null recomputes the observed configuration by other arithmetic (a matrix product where the
#: effect is a ``nansum``, a fresh ``fmean`` where the contrast read a summary's mean), so the
#: draw that *is* the observed labelling can land a rounding step short of it. Compared exactly,
#: that draw reached nothing: an enumerated null's own p came out 0, which it cannot be, and the
#: observed value cleared a floor that was itself. 100 machine epsilons, scipy's
#: ``permutation_test`` convention; read only through :func:`_reaches`.
_TIE_RTOL = 100 * float(np.finfo(float).eps)
#: The share of a sum of squares below which ``Σx² − n·mean²`` is rounding and not a spread.
#: The two terms are equal when every unit has the same effect, and their difference then
#: comes out a few ulps of ``Σx²`` either side of zero rather than zero: a positive one read
#: as a spread near 1e-8, became the family's ``s0``, and a family in which no measure's units
#: disagree was given a floor and t's in the hundreds of millions where it should have had
#: neither (187 of 500 such families). 1e-12 is a spread a millionth of the effects' size,
#: far above the rounding (~1e-16 per term) and far below any spread a cohort can show; read
#: only in :func:`_spread`.
_SPREAD_RTOL = 1e-12


class ComparisonError(ValueError):
    """A comparison asked for something the data or the metadata cannot answer."""


@dataclass(frozen=True)
class Observation:
    """One unit of analysis after the aggregation path has been applied.

    ``n_underlying`` is how many measure rows were averaged into ``value`` — the sample-size trail
    that says whether an animal-level point rests on four sessions or one. Losing it is how an
    unbalanced design becomes invisible.

    ``denominator_ns`` is the mean observed time behind those rows, where the measure had one. It
    is what converts a pseudo-count declared in seconds into the measure's own unit (a share of
    time, a rate), so a within-unit log ratio can be taken over a node that is absent at one level.
    """

    key: tuple
    group: tuple
    measure_key: tuple
    value: float
    n_underlying: int
    denominator_ns: float | None = None


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

    Each metadata value is read through :func:`cohort.cell` — stripped, ``""`` when missing —
    the one reading the design, the left-out rule and every preview apply, so a padded
    ``"drug "`` is the level ``drug`` everywhere and a ``NaN`` is no value everywhere.
    """
    values = metadata.get(row["recording_id"], {})
    out = {column: cell(values, column) for column in values}
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


def _blank(value) -> bool:
    return value == "" or value is None


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
    denominators: dict[tuple, list[float]] = {}
    identity: dict[tuple, tuple] = {}
    for row in rows:
        if keep is not None and row["recording_id"] not in keep:
            continue
        if _blank(row["value"]):
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
        denominator = row.get("denominator_ns")
        if not _blank(denominator):
            denominators.setdefault(key, []).append(float(denominator))
        identity[key] = unit

    out: list[Observation] = []
    for (group, unit, measure_key), values in buckets.items():
        denominator = denominators.get((group, unit, measure_key))
        out.append(
            Observation(
                key=unit,
                group=group,
                measure_key=measure_key,
                value=statistics.fmean(values),
                n_underlying=len(values),
                denominator_ns=statistics.fmean(denominator) if denominator else None,
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

    ``epsilon`` is the pseudo-count added to both centres before the ratio is taken, in the
    measure's own unit. Zero for a label comparison, where a behavior that appears or vanishes is
    reported as itself (:attr:`unbounded`); positive over the node source, where a sparse node
    absent at one level is a sparsity artefact rather than a finding and must have a finite
    magnitude so the family-wise floor is not owned by it (§24 *Designs*).
    """

    measure_key: tuple
    a: GroupSummary
    b: GroupSummary
    difference: float | None
    ratio: float | None
    epsilon: float = 0.0
    #: The share of the null's draws whose *largest* magnitude reached this one — the floor
    #: read as a p — and the share whose magnitude *for this measure* did (its own null); the
    #: Benjamini–Hochberg q over the family's own p's. ``None`` where the magnitude is.
    family_p: float | None = None
    own_p: float | None = None
    q: float | None = None

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
        lo, hi = lo + self.epsilon, hi + self.epsilon
        if lo == 0 and hi == 0:
            return 0.0
        if lo <= 0 or hi <= 0:
            return None
        return abs(math.log(hi / lo))

    @property
    def effect(self) -> float | None:
        """The signed log ratio, ``b`` over ``a`` — what :attr:`magnitude` is the size of."""
        lo, hi = self.a.mean, self.b.mean
        if lo is None or hi is None:
            return None
        lo, hi = lo + self.epsilon, hi + self.epsilon
        if lo <= 0 or hi <= 0:
            return 0.0 if lo == hi else None
        return math.log(hi / lo)

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
        lo, hi = lo + self.epsilon, hi + self.epsilon
        return (lo == 0) != (hi == 0)

    @property
    def percent_change(self) -> float | None:
        """``(b - a) / a`` as a percentage — the readable form of :attr:`magnitude`."""
        lo, hi = self.a.mean, self.b.mean
        if lo is None or hi is None or lo == 0:
            return None
        return (hi - lo) / lo * 100.0

    @property
    def pair(self) -> tuple[str, str]:
        """The two groups the contrast is between, ``(a, b)`` read as *b against a*. Over two
        groups it is the same for every item; over more it is the pair this measure differs
        most between (§24 *Designs*)."""
        return ("|".join(self.a.group), "|".join(self.b.group))

    @property
    def statistic(self) -> float | None:
        """A between contrast has no per-unit effects to studentize: never a t."""
        return None

    @property
    def score(self) -> float | None:
        """What the item is ranked by and the floor is compared with: its magnitude."""
        return self.magnitude


@dataclass(frozen=True)
class Effect:
    """One measure's within-unit effect under a paired or trend design (§24 *Designs*).

    ``effect`` is the **mean of the per-unit effects** — a log ratio of the unit's level means
    for a paired contrast, a Spearman ρ over the unit's levels for a trend, the largest of the
    pairwise log ratios for a profile — never a ratio of group means, which would let one
    animal's four sessions outvote another's one and would permit the across-unit relabelling
    the design forbids. What makes it readable rides with it: how many units had an effect, how
    many were *present*, how many share the sign, and whether the sign survives dropping any
    one unit.

    ``statistic`` is that mean studentized — the moderated t, ``mean / ((sd + s0) / √n)`` over
    the same per-unit effects (§24 *The statistic*) — when the ranking scores by evidence
    (:data:`STATISTIC_T`), and ``None`` when it scores by the effect itself. :attr:`score` is
    whichever the ranking used: what :meth:`Ranking.clears_floor` compares with the floor and
    the order is by; the effect is what a surface prints beside it.

    ``effect`` is ``None`` where the item is undefined — fewer than ``min_units`` present units —
    which is "not shown to differ", not "shown not to", and ranks last rather than large.

    ``pair`` is the two levels the effect is between, ``(a, b)`` read as *b against a*: the
    paired contrast's levels, or the pair a profile's measure differs most between. ``None`` for
    a trend and for an undefined profile item.

    ``family_p`` is the share of the null's draws whose largest |effect| over the family reached
    this one (the floor read as a p: it clears the 5% floor when ``family_p`` is at most about
    0.05, to the null's resolution), ``own_p`` the share whose effect *for this measure* did —
    its own null, which says whether the units agree on it without asking it to beat a thousand
    siblings — and ``q`` the Benjamini–Hochberg false-discovery rate over the family's own p's.
    All ``None`` where the effect is undefined. Over an enumerated null they are exact.
    """

    measure_key: tuple
    effect: float | None
    n_units: int
    n_present: int
    consistency: int
    loo_sign_stable: bool
    a: GroupSummary | None = None
    b: GroupSummary | None = None
    present_a: int | None = None
    present_b: int | None = None
    pair: tuple[str, str] | None = None
    family_p: float | None = None
    own_p: float | None = None
    q: float | None = None
    statistic: float | None = None

    @property
    def magnitude(self) -> float | None:
        return None if self.effect is None else abs(self.effect)

    @property
    def score(self) -> float | None:
        """``|statistic|`` on a studentized ranking, else :attr:`magnitude`: the size the
        floor is compared with."""
        return self.magnitude if self.statistic is None else abs(self.statistic)

    @property
    def unbounded(self) -> bool:
        return False

    @property
    def percent_change(self) -> float | None:
        """A paired log ratio as a percentage; a trend has no percent."""
        if self.effect is None or self.a is None:
            return None
        return (math.exp(self.effect) - 1.0) * 100.0


def _rank_key(item) -> float:
    """Sort key: unbounded changes first, then by score, then the incomputable."""
    if item.unbounded:
        return float("inf")
    return item.score if item.score is not None else -1.0


def contrast(a: GroupSummary, b: GroupSummary, *, epsilon: float = 0.0) -> Contrast:
    """``b`` against ``a`` — difference and ratio of the central values."""
    if a.measure_key != b.measure_key:
        raise ComparisonError(
            f"cannot contrast different measures: {a.measure_key} vs {b.measure_key}"
        )
    lo, hi = a.mean, b.mean
    difference = None if lo is None or hi is None else hi - lo
    ratio = None if not lo or hi is None else hi / lo
    return Contrast(a.measure_key, a, b, difference, ratio, epsilon)


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

    Constructed only by :func:`rank_by_difference` and :func:`rank_by_design`, which compute both
    in one pass — so a ranking cannot exist without its floor **or without saying why it has
    none**. ``floor`` is ``None`` where the permutation had nothing to move (one observation per
    group: every relabelling leaves every fold change unchanged, so the null's maximum *is* the
    observed maximum and `>` can never be satisfied), and :attr:`no_floor_reason` carries the
    sentence every surface prints. A blank column reads as "nothing differed" when the truth is
    "this design cannot tell".

    ``design`` says which permutation produced the floor (§24 *Designs*); ``contrasts`` holds
    :class:`Contrast` items for a between design and :class:`Effect` items otherwise. ``kept`` is
    the set after nesting suppression (:func:`suppress_nesting`), ``None`` where none was applied.
    ``level_values`` is every level (or group) the ranking ran over, in the order the record
    prints them; :attr:`profile` says whether each item's effect is the largest difference
    between any two of them, with the pair on the item.

    **The floor at another strictness is a lookup, not a re-run.** ``floor_distribution`` is the
    whole null, so :meth:`floor_at` reads it at any family-wise level and :meth:`kept_at` is the
    kept set there — the same peak pick over the items clearing *that* floor, with the
    ``ancestors`` the suppression used. At :data:`FLOOR_ALPHA` they are ``floor`` and ``kept``
    exactly; a screen read looser or stricter (§22) is a view over this ranking, and what the
    record says it kept does not move.

    ``statistic`` names what the items are scored by and the floor is in the unit of
    (:data:`STATISTICS`): the moderated t, with ``s0`` the family's median per-unit spread the t
    was moderated by, or the effect itself (``s0`` ``None``). ``item.score`` is that number on
    each item; the floor is compared with nothing else.
    """

    contrasts: list
    floor: float | None
    draws: int
    seed: int
    floor_distribution: list[float] = field(default_factory=list)
    #: Why there is no floor, when there is none — so a surface can say what happened instead of
    #: leaving a blank that reads as "nothing differed".
    no_floor_reason: str | None = None
    design: str = BETWEEN
    levels: tuple[str, str] | None = None
    unit: tuple[str, ...] = ()
    n_units: int = 0
    min_units: int = 0
    #: The floor enumerated every configuration of its null rather than sampling ``draws``.
    exact: bool = False
    kept: tuple[tuple, ...] | None = None
    level_values: tuple[str, ...] = ()
    #: The hierarchy the nesting suppression used (label → the labels above it), so the kept set
    #: at another strictness is picked the same way; ``None`` where none was applied.
    ancestors: Mapping[str, Collection[str]] | None = field(default=None, compare=False)
    #: What the items are scored by (:data:`STATISTICS`), and the spread the t was moderated
    #: by, ``None`` on a ranking by the effect.
    statistic: str = STATISTIC_EFFECT
    s0: float | None = None

    @property
    def profile(self) -> bool:
        """Whether each item is the largest difference between any two of three or more levels
        (§24 *Designs*): a profile design, or a between design over more than two groups. The
        pair then rides on the item (``item.pair``), and the effect is a size whose sign belongs
        to that pair rather than to the measure."""
        return self.design == PROFILE or (
            self.design == BETWEEN and len(self.level_values) > 2
        )

    def clears_floor(self, item) -> bool:
        """Whether this measure's score — its t, or its magnitude — exceeds the floor.

        ``False`` also when there is no floor and when the fold change is unbounded — both are
        "not shown to clear", not "shown not to". A surface must read ``no_floor_reason`` rather
        than render this as a verdict.
        """
        # `unbounded` is *implied* by `score is None` -- one side zero and the other not
        # takes the `lo <= 0 or hi <= 0` branch. It is named anyway because it is the case a
        # reader comes here about, and because the converse does not hold: a missing centre is
        # also `None` and is not unbounded. Do not read a test that mutates it away as coverage.
        if self.floor is None or item.unbounded or item.score is None:
            return False
        return not _reaches(self.floor, item.score)

    def clearing(self) -> list:
        """The items that clear the floor, in rank order."""
        return [item for item in self.contrasts if self.clears_floor(item)]

    def kept_items(self) -> list:
        """What the screen shows as its findings: the kept set after nesting suppression, or
        every clearing item where no suppression was applied. Rank order either way."""
        if self.kept is None:
            return self.clearing()
        kept = set(self.kept)
        return [item for item in self.contrasts if item.measure_key in kept]

    def floor_at(self, alpha: float) -> float | None:
        """The floor at the family-wise level ``alpha`` — the null's ``1 - alpha`` quantile, by
        the same rule as ``floor`` (which it equals at :data:`FLOOR_ALPHA`). ``None`` where there
        is no floor."""
        if self.floor is None:
            return None
        if not self.floor_distribution:
            # A ranking built with its floor and no null behind it: the floor is known at
            # its own level and nowhere else.
            return self.floor if abs(float(alpha) - FLOOR_ALPHA) < 1e-9 else None
        return _percentile_floor(self.floor_distribution, alpha)

    def clears_at(self, item, alpha: float) -> bool:
        """Whether ``item`` clears the floor at the family-wise level ``alpha``: its
        ``family_p`` is at most ``alpha``. ``family_p`` is the smallest level at which the item
        clears :meth:`floor_at`, so this agrees with the item's ``score`` (its t, or its
        magnitude) clearing ``floor_at(alpha)`` at every level, and with :meth:`clears_floor`
        at :data:`FLOOR_ALPHA`; a hover's p and a slider's threshold cannot disagree.
        ``False`` with no floor, or no p."""
        if self.floor is None:
            return False
        p = getattr(item, "family_p", None)
        return p is not None and p <= float(alpha)

    def clearing_at(self, alpha: float) -> list:
        """The items that clear the floor at ``alpha``, in rank order."""
        return [item for item in self.contrasts if self.clears_at(item, alpha)]

    def kept_at(self, alpha: float) -> list:
        """The findings at the family-wise level ``alpha``: the items clearing that floor,
        nesting suppressed the way ``kept`` was (every one of them where it was not). At
        :data:`FLOOR_ALPHA` this is :meth:`kept_items`."""
        if abs(float(alpha) - FLOOR_ALPHA) < 1e-9:
            return self.kept_items()
        clearing = self.clearing_at(alpha)
        if self.kept is None:
            return clearing
        return _peak_pick(clearing, self.ancestors or {})


def _magnitudes(
    keys: Sequence[tuple],
    labels: Sequence[tuple],
    values: Sequence[float],
    groups: tuple,
    epsilon: Mapping[tuple, float],
) -> dict[tuple, float]:
    """Each measure's ``|log ratio|`` for one labelling — the ranked statistic, so the null is
    the null of the thing ranked. A measure whose fold change is unbounded under this labelling
    is left out: there is no finite value for it.

    On ``|log ratio|`` rather than the raw difference, because the raw version made the floor a
    *nanosecond* quantity that a count measure could never reach at any sample size. Unit-free
    ranking needs a unit-free null, or the floor is not the floor of the thing being ranked.

    Over more than two groups the measure's value is the largest ``|log ratio|`` between any two
    of them — the ranked statistic (§24 *Designs*) — so the floor pays for choosing the pair.
    """
    by_key: dict[tuple, dict[tuple, list[float]]] = {}
    for measure_key, group, value in zip(keys, labels, values):
        by_key.setdefault(measure_key, {}).setdefault(group, []).append(value)
    out: dict[tuple, float] = {}
    for measure_key, per_group in by_key.items():
        eps = epsilon.get(measure_key, 0.0)
        centres = [
            statistics.fmean(per_group[g]) + eps for g in groups if per_group.get(g)
        ]
        positive = [c for c in centres if c > 0]
        if len(positive) < 2:
            continue
        out[measure_key] = math.log(max(positive) / min(positive))
    return out


def _largest_magnitude(magnitudes: Mapping[tuple, float]) -> float:
    """The biggest ``|log ratio|`` over all measures, for one labelling.

    **The max across measures is the multiple-comparisons correction, not an oversight.** Rank
    twenty measures and one of them is the largest whether or not anything is there; taking the
    largest under shuffling too is what makes the floor a family-wise threshold rather than a
    per-measure one. That is worth stating because it looks like a bug and removing it would
    quietly drop the correction. The per-measure values it is taken over are each measure's
    own null, kept for its ``own_p``.
    """
    return max(magnitudes.values(), default=0.0)


def _vacuous_permutation(obs: Sequence[Observation], groups: tuple) -> str | None:
    """Why relabelling cannot test anything here, or ``None`` when it can.

    :class:`Ranking` states the principle — a floor that the observed value cannot exceed "would
    be unfalsifiable by construction" — but nothing checked for the data shape that guarantees it.

    **The shape is one observation per group.** A measure with a single value on each side has a
    magnitude that a swap leaves unchanged, so no relabelling can produce a larger one; the null's
    maximum *is* the observed maximum, and `clears_floor` asks for more than a tie. On a real
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
        counts.get(g, 0) > 1 for counts in per_measure.values() for g in groups
    )
    if movable:
        return None
    return (
        "every measure has at most one observation per group, so relabelling cannot change any "
        "measure's fold change and no measure could exceed the floor"
    )


def _percentile_floor(
    ordered: Sequence[float], alpha: float = FLOOR_ALPHA
) -> float | None:
    """The ``1 - alpha`` quantile of a sorted null distribution (the 95th percentile by
    default), by the one convention every ranking uses — so a between floor and a paired floor
    read the same quantile of their nulls, and a floor read at another strictness is the same
    rule at another level.
    """
    if not ordered:
        return None
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"a family-wise level is a fraction in (0, 1), got {alpha!r}")
    return ordered[_floor_index(len(ordered), alpha)]


def _floor_index(n: int, alpha: float) -> int:
    """Where the floor at ``alpha`` sits in a sorted null of ``n`` draws."""
    position = int(round((1.0 - alpha) * (n - 1)))
    return min(max(position, 0), n - 1)


def _reaches(draw, observed: float):
    """Whether a null draw (a float or an array of them) reaches ``observed``, allowing for the
    rounding step by which a recomputed observed configuration can fall short of it
    (:data:`_TIE_RTOL`). The one comparison every p and every floor verdict reads, so the own p,
    the family-wise p and :meth:`Ranking.clears_floor` agree on what a tie is. ``NaN`` reaches
    nothing."""
    return draw >= observed - abs(observed) * _TIE_RTOL


def _family_p(ordered: Sequence[float], magnitude: float | None) -> float | None:
    """The smallest family-wise level at which ``magnitude`` clears the floor of a sorted
    null: the share of the null's draws whose largest effect reaches it, read through the
    same percentile rule as the floor. Defined that way, rather than as the bare count over
    the draws, so ``family_p <= alpha`` and ``magnitude`` clearing ``floor_at(alpha)``
    agree at every level — the rule rounds its position in the null, and a bare count sits one draw off it
    at some levels, which would put a cluster's hover and the slider's threshold at odds.
    ``1`` for an effect no draw falls short of; ``None`` for no magnitude or no null."""
    if magnitude is None or not ordered:
        return None
    n = len(ordered)
    below = int(np.sum(~_reaches(np.asarray(ordered, dtype=float), magnitude)))
    if below == 0:
        return 1.0
    if n == 1:
        return 0.0

    def clears(alpha: float) -> bool:
        return not _reaches(ordered[_floor_index(n, alpha)], magnitude)

    # The rule's position crosses `below - 1` where (1 - alpha)(n - 1) passes below - 0.5;
    # the round there is the one case the arithmetic cannot settle, so the estimate is
    # nudged by a float step either way until it sits on the boundary.
    p = min(max(1.0 - (below - 0.5) / (n - 1), 0.0), 1.0)
    while p < 1.0 and not clears(p):
        p = math.nextafter(p, 1.0)
    while p > 0.0 and clears(math.nextafter(p, 0.0)):
        p = math.nextafter(p, 0.0)
    return p


def _bh(p_values: Sequence[float | None]) -> list[float | None]:
    """Benjamini–Hochberg q over the defined p's: the smallest false-discovery rate at which
    each item is among the discoveries. ``None`` stays ``None``."""
    by_p = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    out: list[float | None] = [None] * len(p_values)
    n = len(by_p)
    running = 1.0
    for rank in range(n, 0, -1):
        p, i = by_p[rank - 1]
        running = min(running, p * n / rank)
        out[i] = running
    return out


def _share_reaching(
    null: np.ndarray, magnitude: float | None, exact: bool
) -> float | None:
    """The share of one item's own null draws reaching ``magnitude`` (unsorted; NaN where a
    draw had no finite value for it, which reaches nothing). Exact where the null was
    enumerated: the observed configuration is one of its draws, so the share is never zero
    (that draw counts through :func:`_reaches`, which allows for its rounding).
    Over a sample the observed labelling counts as one more draw, the standard convention for
    a Monte Carlo p, so the smallest reportable value is ``1 / (draws + 1)``."""
    if magnitude is None or not len(null):
        return None
    reached = int(np.sum(_reaches(null, magnitude)))
    if exact:
        return reached / len(null)
    return (reached + 1) / (len(null) + 1)


def _with_p(
    items: list, ordered: Sequence[float], own: Sequence[float | None], exact: bool
) -> list:
    """``items`` with their family-wise p (against ``ordered``, the null of the family's
    largest), their own p (``own[i]``, already a share) and the BH q over the own p's.
    """
    family = [_family_p(ordered, item.score) for item in items]
    q = _bh(list(own))
    return [
        dataclasses.replace(item, family_p=f, own_p=o, q=qq)
        for item, f, o, qq in zip(items, family, own, q)
    ]


def _epsilon_by_measure(
    obs: Sequence[Observation], pseudo: Callable[[tuple, float | None], float] | None
) -> dict[tuple, float]:
    """One ε per measure key: the pseudo-count in that measure's unit, averaged over the
    observations' denominators (they differ by a recording's length, and a contrast of group
    means has one ε for both sides)."""
    if pseudo is None:
        return {}
    per_key: dict[tuple, list[float]] = {}
    for row in obs:
        per_key.setdefault(row.measure_key, []).append(
            float(pseudo(row.measure_key, row.denominator_ns))
        )
    return {key: statistics.fmean(values) for key, values in per_key.items()}


def _unit_groups(obs: Sequence[Observation]) -> dict[tuple, tuple]:
    """Each unit's group in a between design, where a unit is in exactly one group.

    The between null permutes groups among units, so a unit carrying two groups has no single
    label to permute. The aggregation path cannot produce one today; this states the assumption
    instead of letting the shuffle pick one of the two silently.
    """
    out: dict[tuple, tuple] = {}
    for row in obs:
        seen = out.setdefault(row.key, row.group)
        if seen != row.group:
            raise ComparisonError(
                f"unit {row.key!r} is in two groups, {seen!r} and {row.group!r}; a between "
                "comparison needs each unit in one group — group by a column that is constant "
                "within the unit, or aggregate to a finer unit"
            )
    return out


def rank_by_difference(
    obs: Sequence[Observation],
    groups: tuple,
    *,
    draws: int = 200,
    seed: int = 0,
    pseudo: Callable[[tuple, float | None], float] | None = None,
) -> Ranking:
    """Rank every measure by ``|log ratio|`` between two groups, and compute the floor beside it.

    Over three or more groups each measure's item is the **largest contrast between any two of
    them** (§24 *Designs*), read as the later group against the earlier, with the pair on the
    item (:attr:`Contrast.pair`); the null takes the same largest contrast under each shuffle.
    Over two groups that is the one contrast there is, so nothing already saved changes.

    **Not by raw difference**, which the name still carries for the callers and records that use
    it: a set of measures that do not share a unit has no common linear scale, and ranking by
    difference put a 45-count change beside a 2.5e11-nanosecond one and drew the count as a bar of
    zero width. See :attr:`Contrast.magnitude`.

    The floor is the 95th percentile of the shuffled magnitudes rather than their maximum — see
    :class:`Ranking` for why the maximum is unfalsifiable — and is ``None``, with a reason, where
    the permutation could not have produced a different answer.

    **The permutation shuffles the group labels among units, once per draw, which is the point.**
    A unit is one key of the declared aggregation path — an animal, when the path aggregates to
    animal — and the design randomised units, so the null does too (§24 *Designs*: "group shuffled
    among units"). Each draw is one unit → group map applied to every measure's observations: a
    unit's measures move together, which is what the family-wise maximum assumes, and every
    measure keeps its real group sizes (less any unit missing that measure). Only units in
    ``groups`` take part; an observation of any other group is not in the contrast and stays out
    of the null. Shuffling the underlying *recordings* instead would break an animal into four
    independent draws, and shuffling each (unit, measure) observation on its own would let a
    measure be handed 0 to 8 "drug" animals in a 4-against-4 design and make a measure's own p
    move when unrelated measures are added — the guard is
    ``tests/analysis/test_compare_ranking.py``'s unit-shuffle tests.

    ``pseudo(measure_key, denominator_ns)`` is the pseudo-count in the measure's unit, added to
    both centres before the ratio — the node source's ε (§24 *Designs*). ``None`` keeps a label's
    appearance unbounded, as it always was.
    """
    import random

    if len(groups) < 2:
        raise ComparisonError(f"a contrast needs at least two groups, got {groups}")
    epsilon = _epsilon_by_measure(obs, pseudo)
    summaries = {(s.group, s.measure_key): s for s in summarize(obs)}
    contrasts = []
    for measure_key in sorted({row.measure_key for row in obs}):
        present = [
            s
            for s in (summaries.get((g, measure_key)) for g in groups)
            if s is not None
        ]
        if len(present) < 2:
            continue
        eps = epsilon.get(measure_key, 0.0)
        candidates = [
            contrast(a, b, epsilon=eps)
            for i, a in enumerate(present)
            for b in present[i + 1 :]
        ]
        contrasts.append(max(candidates, key=_rank_key))
    contrasts.sort(key=_rank_key, reverse=True)

    rng = random.Random(seed)
    # One label per unit, shuffled among the units of the contrasted groups and applied to all of
    # a unit's observations — never one label per (unit, measure) row.
    unit_group = _unit_groups(obs)
    units = sorted(u for u, g in unit_group.items() if g in groups)
    unit_index = {u: i for i, u in enumerate(units)}
    labels = [unit_group[u] for u in units]
    in_null = [row for row in obs if row.key in unit_index]
    row_unit = [unit_index[row.key] for row in in_null]
    keys = [row.measure_key for row in in_null]
    values = [row.value for row in in_null]
    distribution: list[float] = []
    # Each measure's own null beside the family's largest: what its own p is read from.
    own_null = np.full((len(contrasts), draws), np.nan)
    for d in range(draws):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        row_labels = [shuffled[u] for u in row_unit]
        magnitudes = _magnitudes(keys, row_labels, values, groups, epsilon)
        distribution.append(_largest_magnitude(magnitudes))
        for n, item in enumerate(contrasts):
            own_null[n, d] = magnitudes.get(item.measure_key, np.nan)
    ordered = sorted(distribution)
    # A floor is reported only when the permutation had something to move. Withheld with its
    # reason rather than drawn as a threshold nothing could reach.
    reason = _vacuous_permutation(obs, groups)
    floor = None if reason else _percentile_floor(ordered)
    if not reason:
        contrasts = _with_p(
            contrasts,
            ordered,
            [
                _share_reaching(own_null[n], item.magnitude, False)
                for n, item in enumerate(contrasts)
            ],
            False,
        )
    return Ranking(
        contrasts,
        floor,
        draws,
        seed,
        ordered,
        reason,
        design=BETWEEN,
        levels=(
            ("|".join(groups[0]), "|".join(groups[1]))
            if contrasts and len(groups) == 2
            else None
        ),
        n_units=len({o.key for o in obs}),
        level_values=tuple("|".join(g) for g in groups),
    )


# --------------------------------------------------------------------------------------------- #
# designs (§24 *Designs*)
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Design:
    """What the chosen column implies, resolved once and printed everywhere.

    ``kind`` is one of :data:`DESIGNS`; ``column`` is the one grouping column of a within-unit
    design; ``levels`` is the paired contrast's ``(a, b)``, read as *b against a*, and ``None``
    for a trend or a profile, which run over every value the column holds; ``unit`` is the
    aggregation path; ``reason`` is the sentence a surface shows, which is also the record's
    ``design_reason``.
    """

    kind: str
    column: str | None
    levels: tuple[str, str] | None
    unit: tuple[str, ...]
    reason: str

    @property
    def within(self) -> bool:
        return self.kind != BETWEEN


def _is_number(text: str) -> bool:
    """A finite number. ``nan`` and ``inf`` parse under ``float`` and are not levels of a dose."""
    if missing(text):
        return False
    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError):
        return False


def _level_sort_key(value: str):
    """Numeric levels sort as numbers (0, 25, 50, 75), the rest as text."""
    return (0, float(value), "") if _is_number(value) else (1, 0.0, value)


def _unit_word(unit: Sequence[str]) -> str:
    return "|".join(unit) if unit else "recording"


def _design_from_pairs(
    pairs: Iterable[tuple[tuple, str]],
    *,
    column: str | None,
    unit: Sequence[str],
    levels: Sequence[str] | None,
    n_columns: int,
) -> Design:
    """The resolution rule itself, over ``(unit key, column value)`` pairs (§24 *Designs*)."""
    unit = tuple(unit)
    word = _unit_word(unit)
    by_unit: dict[tuple, set[str]] = {}
    for key, value in pairs:
        if not missing(value):
            by_unit.setdefault(key, set()).add(value)
    distinct = sorted(
        {v for values in by_unit.values() for v in values}, key=_level_sort_key
    )
    chosen: tuple[str, str] | None = None
    if levels:
        chosen = tuple(str(v).strip() for v in levels)
        if n_columns != 1:
            raise ComparisonError(
                f"levels {list(chosen)} name values of one column, and {n_columns} columns "
                f"define the groups. Group by one column to contrast two of its levels."
            )
        if len(chosen) != 2 or chosen[0] == chosen[1]:
            raise ComparisonError(
                f"levels must be two distinct values of '{column}', got {list(chosen)}"
            )
        unknown = [v for v in chosen if v not in distinct]
        if unknown:
            raise ComparisonError(
                f"{unknown} is not a value of '{column}' in these recordings. "
                f"Its values: {distinct}"
            )

    if not unit:
        return Design(
            BETWEEN,
            column,
            chosen,
            unit,
            "each recording is one observation, so the groups are compared across recordings",
        )
    varies = any(len(values) > 1 for values in by_unit.values())
    if column is None:
        return Design(
            BETWEEN,
            None,
            chosen,
            unit,
            f"{n_columns} columns define the groups, which are compared across {word}s",
        )
    if not varies:
        what = f"{chosen[0]} and {chosen[1]}" if chosen else "its groups"
        return Design(
            BETWEEN,
            column,
            chosen,
            unit,
            f"'{column}' is constant within each {word}, so {what} are compared across "
            f"{word}s",
        )

    numeric = bool(distinct) and all(_is_number(v) for v in distinct)
    if chosen is None and numeric and len(distinct) >= 3:
        return Design(
            TREND,
            column,
            None,
            unit,
            f"'{column}' is numeric and varies within each {word}: a trend over "
            f"{', '.join(distinct)}, measured within each {word}",
        )
    if chosen is None:
        if len(distinct) == 2:
            chosen = (distinct[0], distinct[1])
        else:
            return Design(
                PROFILE,
                column,
                None,
                unit,
                f"'{column}' varies within each {word} over {len(distinct)} values "
                f"({', '.join(distinct)}): a profile, the largest difference between any "
                f"two of them, measured within each {word}",
            )
    both = sum(1 for values in by_unit.values() if set(chosen) <= values)
    return Design(
        PAIRED,
        column,
        chosen,
        unit,
        f"'{column}' varies within each {word}: {chosen[1]} against {chosen[0]}, paired "
        f"within {word} ({both} with both)",
    )


def resolve_design(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    *,
    group_by: Sequence[str],
    aggregate_to: Sequence[str],
    levels: Sequence[str] | None = None,
) -> Design:
    """The design a comparison over these recordings will run, from ``metadata.csv`` alone.

    What the launch dialog previews before the run and the CLI prints after it, so the two cannot
    disagree: one function, one sentence. A row key (``subject``, ``window_id``) is not in the
    metadata, so a comparison grouped on one resolves at run time from the measure rows
    (:func:`resolve_design_from_rows`) and previews as "decided when it runs".
    """
    group_by = list(group_by)
    column = group_by[0] if len(group_by) == 1 else None
    pairs = []
    for recording in recordings:
        values = metadata.get(recording, {})
        row = {name: cell(values, name) for name in values}
        row["recording_id"] = recording
        key = tuple(row.get(k, "") for k in aggregate_to)
        value = row.get(column, "") if column else ""
        pairs.append((key, value))
    return _design_from_pairs(
        pairs, column=column, unit=aggregate_to, levels=levels, n_columns=len(group_by)
    )


def resolve_design_from_rows(
    rows: Sequence[dict],
    metadata: Mapping[str, Mapping[str, str]],
    *,
    group_by: Sequence[str],
    aggregate_to: Sequence[str],
    levels: Sequence[str] | None = None,
) -> Design:
    """The same rule over the measure rows, where ``subject`` and ``window_id`` exist."""
    group_by = list(group_by)
    column = group_by[0] if len(group_by) == 1 else None
    seen: set[tuple] = set()
    pairs = []
    for row in rows:
        factors = _factors(row, metadata)
        key = tuple(str(factors.get(k, "")) for k in aggregate_to)
        value = str(factors.get(column, "")) if column else ""
        if (key, value, row["recording_id"]) in seen:
            continue
        seen.add((key, value, row["recording_id"]))
        pairs.append((key, value))
    return _design_from_pairs(
        pairs, column=column, unit=aggregate_to, levels=levels, n_columns=len(group_by)
    )


def _levels_needed(kind: str) -> int:
    """How many levels a unit must hold for the design's null to move it: both of a paired
    contrast's, two of a profile's (a shuffle within one level is the identity), three of a
    trend's (a Spearman over fewer is undefined, and :func:`rank_by_design` skips the unit).
    """
    return 3 if kind == TREND else 2


def _levels_held(design: Design) -> str:
    """What a unit has to hold to move under ``design``'s null, in the reason's words."""
    if design.kind == PAIRED and design.levels:
        return f"both {design.levels[0]} and {design.levels[1]}"
    if design.kind == TREND:
        return f"three or more values of '{design.column}'"
    return f"two or more values of '{design.column}'"


def _too_few_units_reason(design: Design, movable: Sequence[tuple]) -> str:
    """Why a within-unit null cannot move: fewer than two units hold the levels it flips or
    shuffles. ``movable`` is the units that do, so the one there is can be named. One sentence
    for :func:`rank_by_design` after measuring and :func:`no_floor_before_measuring` before.
    """
    word = _unit_word(design.unit)
    who = f"only one {word} ({'|'.join(movable[0])})" if movable else f"no {word}"
    return (
        f"{who} has {_levels_held(design)}, so a within-{word} null has nothing to move "
        f"and no floor can be computed"
    )


@dataclass(frozen=True)
class NoFloor:
    """Why a comparison over these recordings can have no floor, known before measuring (§24).

    ``reason`` is the sentence the ranking carries as ``no_floor_reason`` once measured, and
    ``remedy`` what would change the answer, in the data's terms ("Add a second animal that has
    both veh and fent."). A surface appends what else the user could do.
    """

    reason: str
    remedy: str


def no_floor_before_measuring(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    *,
    group_by: Sequence[str],
    aggregate_to: Sequence[str],
    levels: Sequence[str] | None = None,
    one_row_per_recording: bool = False,
) -> NoFloor | None:
    """Why no floor could be computed over these recordings, from ``metadata.csv`` alone, or
    ``None`` when measuring could still give one.

    Three shapes are certain before anything is measured, because they are counts of
    recordings and their metadata: a within-unit design where fewer than two units hold the
    levels its null moves (:func:`_too_few_units_reason`, the sentence :func:`rank_by_design`
    carries afterwards); a between design where every group holds at most one unit, so
    relabelling changes nothing (:func:`_vacuous_permutation`'s shape); and fewer than two
    groups to contrast, which ranks nothing at all. The other no-floor reasons (no measure
    present in enough units, no spread to divide by) need the measurements and are not asked.

    **Never a false refusal.** Every count here is an upper bound on what the ranking will have,
    since a unit can lose a level to a blank measure and never gain one, so a reason here implies
    the same reason after measuring. The guard is
    ``tests/analysis/test_no_floor_before_measuring.py``.

    ``one_row_per_recording`` says each recording yields one observation per measure (whole
    windows, one subject), which is when a between design with an empty aggregation path can be
    counted: under windows one recording per group is many observations, and the null can move.
    A grouping or aggregation column the metadata does not carry (``subject``, ``window_id``) is
    decided when the comparison runs, and returns ``None``. A recording with no value for a
    grouping column is left out before counting, as the run leaves it out (§24).
    """
    group_by, aggregate_to = list(group_by), list(aggregate_to)
    rows: dict[str, dict[str, str]] = {}
    for recording in recordings:
        values = metadata.get(recording, {})
        row = {name: cell(values, name) for name in values}
        row["recording_id"] = recording
        rows[recording] = row
    carried: set[str] = set()
    for row in rows.values():
        carried |= set(row)
    if not rows or any(c not in carried for c in [*group_by, *aggregate_to]):
        return None
    outside = set(left_out(metadata, recordings, group_by))
    kept = [r for r in recordings if r not in outside]
    design = resolve_design(
        metadata, kept, group_by=group_by, aggregate_to=aggregate_to, levels=levels
    )
    word = _unit_word(design.unit)

    def unit_of(recording: str) -> tuple:
        return tuple(rows[recording][k] for k in aggregate_to)

    if design.within:
        wanted = set(design.levels) if design.levels else None
        held: dict[tuple, set[str]] = {}
        for recording in kept:
            value = rows[recording][design.column]
            if wanted is not None and value not in wanted:
                continue
            if design.kind == TREND and not _is_number(value):
                continue
            held.setdefault(unit_of(recording), set()).add(value)
        needed = _levels_needed(design.kind)
        movable = sorted(
            u for u, levels_held in held.items() if len(levels_held) >= needed
        )
        if len(movable) >= 2:
            return None
        article = "a second" if movable else "an"
        return NoFloor(
            _too_few_units_reason(design, movable),
            f"Add {article} {word} that has {_levels_held(design)}.",
        )

    wanted = set(design.levels) if design.levels else None
    groups: dict[tuple, set[tuple]] = {}
    for recording in kept:
        group = tuple(rows[recording][k] for k in group_by)
        if wanted is not None and group[0] not in wanted:
            continue
        unit = unit_of(recording) if aggregate_to else (recording,)
        groups.setdefault(group, set()).add(unit)
    if len(groups) < 2:
        if not groups:
            what = "no recording has a value for every grouping column"
        elif len(group_by) == 1:
            what = (
                f"these recordings hold one value of '{group_by[0]}' "
                f"({next(iter(groups))[0]})"
            )
        else:
            what = f"these recordings form one group ({'|'.join(next(iter(groups)))})"
        remedy = (
            f"Choose recordings with two values of '{group_by[0]}'."
            if len(group_by) == 1
            else "Choose recordings that form two groups."
        )
        return NoFloor(f"{what}, so there is nothing to contrast", remedy)
    if not aggregate_to and not one_row_per_recording:
        return None
    if any(len(units) > 1 for units in groups.values()):
        return None
    return NoFloor(
        f"every group holds at most one {word}, so relabelling cannot change any measure "
        f"and no floor can be computed",
        f"Add a second {word} to a group.",
    )


def _loo(per_unit: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Whether the sign survives leaving any one unit out, per measure."""
    valid = ~np.isnan(per_unit)
    n = valid.sum(axis=1)
    total = np.nansum(per_unit, axis=1)
    z = np.nan_to_num(per_unit)
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (total[:, None] - z) / (n[:, None] - valid)
    loo = np.where(valid & (n[:, None] > 1), loo, np.nan)
    sign = np.sign(effect)
    out = np.zeros(len(effect), dtype=bool)
    for i, row in enumerate(loo):
        kept = row[~np.isnan(row)]
        out[i] = bool(len(kept)) and bool(np.all(np.sign(kept) == sign[i]))
    return out


def _sign_patterns(n_units: int, draws: int, rng) -> tuple[np.ndarray, bool]:
    """``(units, configurations)`` of ±1: every configuration when there are few enough units
    to enumerate, a sample of ``draws`` otherwise."""
    if n_units <= EXACT_FLIPS_MAX_UNITS:
        bits = (np.arange(2**n_units)[:, None] >> np.arange(n_units)) & 1
        return (bits * 2 - 1).T.astype(float), True
    return rng.choice([-1.0, 1.0], size=(n_units, draws)), False


def _spearman_rows(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Spearman ρ of every row of ``X`` against ``d``; 0 where a row is constant."""
    from scipy.stats import rankdata

    dr = rankdata(d)
    dc = dr - dr.mean()
    dn = np.linalg.norm(dc)
    R = rankdata(X, axis=1)
    Rc = R - R.mean(axis=1, keepdims=True)
    rn = np.linalg.norm(Rc, axis=1)
    safe = np.where(rn > 0, rn, 1.0)
    r = (Rc @ dc) / (safe * dn)
    r[rn == 0] = 0.0
    return r


def rank_by_design(
    obs: Sequence[Observation],
    design: Design,
    *,
    draws: int = 200,
    seed: int = 0,
    min_units: int = 1,
    presence_ns: float = 0.0,
    pseudo: Callable[[tuple, float | None], float] | None = None,
    ranking_measure: str | None = None,
    statistic: str = STATISTIC_EFFECT,
) -> Ranking:
    """Rank every measure by its within-unit effect, with the floor the design's own null gives.

    ``obs`` is the full observation list; ``ranking_measure`` names the family actually ranked
    (the node source ranks on ``time_fraction`` and shows the rest beside it), and the
    ``total_time`` observations decide presence. ``min_units`` is the resolved count of *present*
    units below which an effect is undefined. ``pseudo(measure_key, denominator_ns)`` is the paired
    ratio's ε in the measure's unit.

    The null **respects the design**: a paired contrast flips each unit's sign (every configuration
    when the cohort has ``EXACT_FLIPS_MAX_UNITS`` units or fewer, so the floor is deterministic),
    and a trend or a profile shuffles each unit's levels within that unit. The family-wise floor
    is the 95th percentile of the largest |score| over the ranked family, as for
    :func:`rank_by_difference`.

    ``statistic`` is what each measure is scored by (:data:`STATISTICS`, §24 *The statistic*).
    :data:`STATISTIC_T` studentizes the mean per-unit effect: ``t = mean / ((sd + s0) / √n)``
    over the units with an effect, ``sd`` their sample spread (``0`` for a single unit) and
    ``s0`` the median ``sd`` over the family's defined measures, fixed from the observed
    configuration and applied to every draw of the null. Under the sign flips the sum of squares
    is invariant, so each draw's spread follows from its mean, and the observed configuration
    stays one of the enumerated draws. :data:`STATISTIC_EFFECT` scores by the mean itself, the
    rule every record written before the field existed is read by. The floor, every p and the
    order are in the score's unit; the effect is printed beside it either way.

    A **profile** (three or more named levels) gives every pair of levels the paired contrast and
    ranks each measure by the largest score among the pairs defined for it, with the pair on the
    item. A pair is defined where ``min_units`` units are present at either of its levels, as a
    paired contrast is; a measure with no defined pair is undefined. The null takes the same
    largest pair under each within-unit shuffle, so the floor already pays for choosing the pair
    and no second correction is applied. Its null is sampled: ``(k!)^n`` configurations cannot be
    enumerated at any realistic size. Under the t the ``s0`` is the median spread over every
    defined pair of every measure, since the pair is chosen by the t it moderates.
    """
    if design.kind == BETWEEN:
        raise ComparisonError("a between design is ranked by rank_by_difference")
    if statistic not in STATISTICS:
        raise ComparisonError(
            f"statistic must be one of {', '.join(STATISTICS)}, got {statistic!r}"
        )
    studentized = statistic == STATISTIC_T
    rng = np.random.default_rng(seed)
    units = sorted({o.key for o in obs})
    unit_index = {u: i for i, u in enumerate(units)}
    if design.kind == PAIRED:
        level_values = list(design.levels or ())
    elif design.kind == PROFILE:
        level_values = sorted(
            {o.group[0] for o in obs if o.group and not missing(o.group[0])},
            key=_level_sort_key,
        )
    else:
        level_values = sorted(
            {o.group[0] for o in obs if o.group and _is_number(o.group[0])}, key=float
        )
    level_index = {(lvl,): j for j, lvl in enumerate(level_values)}
    ranked = [
        o
        for o in obs
        if (ranking_measure is None or o.measure_key[0] == ranking_measure)
        and o.group in level_index
    ]
    keys = sorted({o.measure_key for o in ranked})
    key_index = {k: i for i, k in enumerate(keys)}
    N, U, L = len(keys), len(units), len(level_values)
    X = np.full((N, U, L), np.nan)
    Den = np.full((N, U, L), np.nan)
    for o in ranked:
        n, u, j = key_index[o.measure_key], unit_index[o.key], level_index[o.group]
        X[n, u, j] = o.value
        if o.denominator_ns is not None:
            Den[n, u, j] = o.denominator_ns

    # Presence per (measure, unit, level), from the total_time observations of the same
    # measurement; without them, a unit with a value counts as present.
    present_level = np.zeros((N, U, L), dtype=bool)
    time_of = {
        (o.measure_key[1:], o.key, o.group): o.value
        for o in obs
        if o.measure_key[0] == PRESENCE_MEASURE and o.group in level_index
    }
    if time_of:
        for key, n in key_index.items():
            for u, unit in enumerate(units):
                for group, j in level_index.items():
                    value = time_of.get((key[1:], unit, group))
                    present_level[n, u, j] = value is not None and value >= presence_ns
    else:
        present_level = ~np.isnan(X)
    present = present_level.any(axis=2)

    exact = False
    present_a = present_b = None
    pair_of: list[tuple[str, str] | None] = [None] * N
    if design.kind == PAIRED:
        EA = np.zeros((N, U))
        EB = np.zeros((N, U))
        if pseudo is not None:
            for key, n in key_index.items():
                for u in range(U):
                    EA[n, u] = pseudo(key, _finite_or_none(Den[n, u, 0]))
                    EB[n, u] = pseudo(key, _finite_or_none(Den[n, u, 1]))
        A = X[:, :, 0] + EA
        B = X[:, :, 1] + EB
        with np.errstate(invalid="ignore", divide="ignore"):
            D = np.log(B / A)
        D = np.where(np.isnan(A) | np.isnan(B) | (A <= 0) | (B <= 0), np.nan, D)
        valid = ~np.isnan(D)
        n_units = valid.sum(axis=1)
        n_present = (valid & present).sum(axis=1)
        enough = n_present >= max(int(min_units), 1)
        S, exact = _sign_patterns(U, draws, rng)
        null_sum = np.nan_to_num(D) @ S
        null_count = np.broadcast_to(n_units[:, None], null_sum.shape)
        # A sign flip leaves the sum of squares where it was, so each draw's spread follows
        # from its mean (`_spread`), and the observed configuration is still one of the draws.
        null_sq = np.broadcast_to(np.nansum(D**2, axis=1)[:, None], null_sum.shape)
        present_a = (valid & present_level[:, :, 0]).sum(axis=1)
        present_b = (valid & present_level[:, :, 1]).sum(axis=1)
        per_unit = D
    elif design.kind == PROFILE:
        E = np.zeros((N, U, L))
        if pseudo is not None:
            for key, n in key_index.items():
                for u in range(U):
                    for j in range(L):
                        E[n, u, j] = pseudo(key, _finite_or_none(Den[n, u, j]))
        shifted = X + E
        with np.errstate(invalid="ignore", divide="ignore"):
            LX = np.where(np.isnan(shifted) | (shifted <= 0), np.nan, np.log(shifted))
        # Every pair of levels, (i, j) with i before j in level order, read as j against i.
        I_, J_ = np.triu_indices(L, k=1)
        D = LX[:, :, J_] - LX[:, :, I_]
        valid_p = ~np.isnan(D)
        present_p = present_level[:, :, I_] | present_level[:, :, J_]
        n_units_p = valid_p.sum(axis=1)
        n_present_p = (valid_p & present_p).sum(axis=1)
        enough_p = n_present_p >= max(int(min_units), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_p = np.where(
                enough_p, np.nansum(D, axis=1) / np.maximum(n_units_p, 1), np.nan
            )
        if studentized:
            # One s0 over every defined pair of every measure: the pair is chosen by the t
            # it moderates, so it cannot be read from the chosen pairs alone.
            sd_p = _spread(np.nansum(D**2, axis=1), mean_p, n_units_p)
            s0 = _s0(sd_p[enough_p])
            score_p = _studentize(mean_p, sd_p, n_units_p, s0)
        else:
            s0, score_p = None, mean_p
        # The measure's pair: the largest |score| among its defined pairs. An undefined pair
        # is below every defined one, and a measure with none is undefined, never large.
        size = np.where(np.isnan(score_p), -1.0, np.abs(score_p))
        rows = np.arange(N)
        if N and len(I_):
            best = size.argmax(axis=1)
            defined = size[rows, best] >= 0
        else:
            best = np.zeros(N, dtype=int)
            defined = np.zeros(N, dtype=bool)
        if len(I_):
            per_unit = D[rows, :, best]
            valid = valid_p[rows, :, best]
            n_units = n_units_p[rows, best]
            n_present = np.where(
                defined, n_present_p[rows, best], n_present_p.max(axis=1)
            )
            enough = enough_p[rows, best]
            score = score_p[rows, best]
            present_a = (valid & present_level[rows, :, I_[best]]).sum(axis=1)
            present_b = (valid & present_level[rows, :, J_[best]]).sum(axis=1)
        else:
            per_unit = np.full((N, U), np.nan)
            valid = np.zeros((N, U), dtype=bool)
            n_units = n_present = np.zeros(N, dtype=int)
            enough = np.zeros(N, dtype=bool)
            score = np.full(N, np.nan)
            present_a = present_b = np.zeros(N, dtype=int)
        pair_of = [
            (level_values[I_[b]], level_values[J_[b]]) if d else None
            for b, d in zip(best, defined)
        ]
        # The null: each unit's levels shuffled within it (the levels it has data for), and
        # the same largest pair taken each time, so the floor pays for choosing the pair.
        have = ~np.isnan(X).all(axis=0) if N else np.zeros((U, L), dtype=bool)
        null_score = np.zeros((N, draws))
        for d in range(draws):
            perm = np.tile(np.arange(L), (U, 1))
            for u in range(U):
                idx = np.flatnonzero(have[u])
                if len(idx) > 1:
                    perm[u, idx] = rng.permutation(idx)
            LXp = np.take_along_axis(LX, np.broadcast_to(perm[None], (N, U, L)), axis=2)
            Dp = LXp[:, :, J_] - LXp[:, :, I_]
            count = (~np.isnan(Dp)).sum(axis=1)
            moved = enough_p & (count > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mp = np.where(moved, np.nansum(Dp, axis=1) / np.maximum(count, 1), 0.0)
            if studentized and s0 is not None:
                sd_d = _spread(np.nansum(Dp**2, axis=1), mp, count)
                sp = np.where(moved, _studentize(mp, sd_d, count, s0), 0.0)
            elif studentized:
                sp = np.zeros_like(mp)
            else:
                sp = mp
            if sp.size:
                null_score[:, d] = np.abs(sp).max(axis=1)
    else:
        rho = np.full((N, U), np.nan)
        null_sum = np.zeros((N, draws))
        null_sq = np.zeros((N, draws))
        null_count = np.zeros((N, draws))
        levels_f = np.array([float(v) for v in level_values])
        for u in range(U):
            Xu = X[:, u, :]
            have = ~np.isnan(Xu).all(axis=0)
            if have.sum() < 3:
                continue
            d = levels_f[have]
            block = Xu[:, have]
            valid_u = ~np.isnan(block).any(axis=1)
            r = _spearman_rows(np.nan_to_num(block), d)
            rho[:, u] = np.where(valid_u, r, np.nan)
            shuffled = np.stack([rng.permutation(d) for _ in range(draws)])
            rnull = np.stack(
                [_spearman_rows(np.nan_to_num(block), s) for s in shuffled], axis=1
            )
            rnull[~valid_u] = 0.0
            null_sum += rnull
            null_sq += rnull**2
            null_count += valid_u[:, None]
        valid = ~np.isnan(rho)
        n_units = valid.sum(axis=1)
        n_present = (valid & present).sum(axis=1)
        enough = n_present >= max(int(min_units), 1)
        per_unit = rho

    with np.errstate(invalid="ignore", divide="ignore"):
        effect = np.where(
            enough, np.nansum(per_unit, axis=1) / np.maximum(n_units, 1), np.nan
        )
    if design.kind != PROFILE:
        moved = enough[:, None] & (null_count > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            null_effect = np.where(moved, null_sum / np.maximum(null_count, 1), 0.0)
        if studentized:
            sd = _spread(np.nansum(per_unit**2, axis=1), effect, n_units)
            s0 = _s0(sd[enough])
            score = _studentize(effect, sd, n_units, s0)
            if s0 is None:
                null_score = np.zeros_like(null_effect)
            else:
                null_sd = _spread(null_sq, null_effect, null_count)
                null_score = np.where(
                    moved, _studentize(null_effect, null_sd, null_count, s0), 0.0
                )
        else:
            s0, score, null_score = None, effect, null_effect
    sign = np.sign(effect)
    consistency = (valid & (np.sign(per_unit) == sign[:, None]) & (per_unit != 0)).sum(
        axis=1
    )
    loo_stable = _loo(per_unit, effect) if N else np.zeros(0, dtype=bool)
    null_max = np.abs(null_score).max(axis=0) if N else np.zeros(null_score.shape[1])
    ordered = sorted(float(v) for v in null_max)

    # The units the null can move: those with data at the levels it flips or shuffles. A unit
    # at one level of a paired contrast has no per-unit effect and every flip of it is the
    # identity, so two units with one of them so is one unit: its floor was the observed
    # maximum, which nothing can exceed (`_vacuous_permutation` is the between shape).
    needed = _levels_needed(design.kind)
    movable = [
        units[u]
        for u in range(U)
        if N and int((~np.isnan(X[:, u, :])).any(axis=0).sum()) >= needed
    ]
    if not N:
        reason = "there are no observations to rank"
    elif len(movable) < 2:
        reason = _too_few_units_reason(design, movable)
    elif not enough.any():
        reason = (
            f"no measure is present in {max(int(min_units), 1)} or more "
            f"{_unit_word(design.unit)}s, so every effect is undefined"
        )
    elif studentized and s0 is None:
        reason = (
            f"no measure's {_unit_word(design.unit)}s disagree with each other, so the "
            f"moderated t has no spread to divide by"
        )
    else:
        reason = None
    floor = None if reason else _percentile_floor(ordered)

    summaries = {(s.group, s.measure_key): s for s in summarize(ranked)}
    items = []
    for key, n in key_index.items():
        a = b = None
        pair = pair_of[n]
        if design.kind == PAIRED:
            pair = (level_values[0], level_values[1])
        if pair is not None:
            a = summaries.get(((pair[0],), key))
            b = summaries.get(((pair[1],), key))
        value = float(effect[n])
        t = float(score[n]) if studentized else float("nan")
        items.append(
            Effect(
                measure_key=key,
                effect=None if math.isnan(value) else value,
                n_units=int(n_units[n]),
                n_present=int(n_present[n]),
                consistency=int(consistency[n]),
                loo_sign_stable=bool(loo_stable[n]),
                a=a,
                b=b,
                present_a=None if present_a is None else int(present_a[n]),
                present_b=None if present_b is None else int(present_b[n]),
                pair=pair,
                statistic=None if math.isnan(t) else t,
            )
        )
    if not reason:
        # Each item's own null is the same draws read for that item alone; exact where the
        # flips were enumerated, since the observed configuration is then one of them.
        size = np.abs(score)
        own = [
            (
                None
                if math.isnan(size[n])
                else _share_reaching(np.abs(null_score[n]), float(size[n]), exact)
            )
            for n in range(N)
        ]
        items = _with_p(items, ordered, own, exact)
    items.sort(key=_rank_key, reverse=True)
    return Ranking(
        items,
        floor,
        len(ordered) if exact else draws,
        seed,
        ordered,
        reason,
        design=design.kind,
        levels=tuple(level_values[:2]) if design.kind == PAIRED else None,
        unit=tuple(design.unit),
        n_units=U,
        min_units=int(min_units),
        exact=exact,
        level_values=tuple(level_values),
        statistic=statistic,
        s0=s0,
    )


def _spread(ss: np.ndarray, mean: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The sample spread of per-unit effects from their sum of squares, mean and count:
    ``sqrt((Σx² − n·mean²) / (n − 1))``, ``0`` for fewer than two units (one unit has no
    spread to report, and its t is its effect over ``s0``). The sum of squares does not move
    under a sign flip, so a null draw's spread follows from its mean alone.

    **Exactly ``0`` when the units agree.** The difference of the two terms is then rounding,
    a few ulps of ``Σx²`` and as often positive as negative, so one within
    :data:`_SPREAD_RTOL` of the sum of squares is no spread: :func:`_s0` asks ``> 0``, and a
    rounding residue answered yes.
    """
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = ss - n * mean**2
        residual = np.where(residual <= ss * _SPREAD_RTOL, 0.0, residual)
        var = residual / np.maximum(n - 1.0, 1.0)
    return np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), 0.0)


def _s0(spreads: np.ndarray) -> float | None:
    """The moderation constant: the median of the family's positive spreads, fixed from the
    observed configuration (SAM's fudge factor, chosen the simple way). ``None`` when no
    measure's units disagree at all — then there is nothing to divide by, and the ranking
    says so instead of inventing a number."""
    positive = spreads[np.isfinite(spreads) & (spreads > 0)]
    return float(np.median(positive)) if positive.size else None


def _studentize(
    mean: np.ndarray, sd: np.ndarray, n: np.ndarray, s0: float | None
) -> np.ndarray:
    """The moderated t, ``mean / ((sd + s0) / √n)``; ``NaN`` wherever the mean is undefined,
    no unit had an effect, or there is no ``s0``."""
    if s0 is None:
        return np.full(np.shape(mean), np.nan)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = mean * np.sqrt(n) / (sd + s0)
    return np.where(n > 0, t, np.nan)


def _finite_or_none(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _label_of(item) -> str:
    """The label a ranked item is about — position 1 of the measure key, after the measure."""
    return str(item.measure_key[1]) if len(item.measure_key) > 1 else ""


def _peak_pick(items: Sequence, ancestors: Mapping[str, Collection[str]], limit=None):
    """Keep the best, drop anything nested in a kept one, repeat — in the order given."""
    kept: list = []
    kept_labels: list[str] = []
    for item in items:
        label = _label_of(item)
        above = set(ancestors.get(label, ()))
        if any(k in above or label in set(ancestors.get(k, ())) for k in kept_labels):
            continue
        kept.append(item)
        kept_labels.append(label)
        if limit is not None and len(kept) >= limit:
            break
    return kept


def suppress_nesting(
    ranking: Ranking, ancestors: Mapping[str, Collection[str]]
) -> Ranking:
    """Peak picking over the items that clear (§24 *The node source*).

    ``ancestors`` maps a label to the labels above it in the hierarchy. A node and its parent
    carry nearly the same time, so the clearing set is full of slices of one behaviour; keeping
    the best and dropping its ancestors and descendants leaves one item per behaviour.
    """
    kept = _peak_pick(ranking.clearing(), ancestors)
    return dataclasses.replace(
        ranking, kept=tuple(item.measure_key for item in kept), ancestors=ancestors
    )


def below_floor(
    ranking: Ranking, ancestors: Mapping[str, Collection[str]] | None, n: int
) -> list:
    """The best-ranked items that did not clear, nesting suppressed among themselves — the
    "show more" list under a ranking's findings."""
    rest = [
        item
        for item in ranking.contrasts
        if not ranking.clears_floor(item) and item.magnitude is not None
    ]
    if ancestors is None:
        return rest[:n]
    return _peak_pick(rest, ancestors, limit=n)


def stats_rows(
    obs: Sequence[Observation],
    *,
    aggregate_to: Sequence[str],
    remainders: Collection[str],
    group_by: Sequence[str] = (),
) -> list[dict]:
    """Long-form rows — one per observation, with its factors — the source of truth an export
    reads and a notebook reshapes.

    ``aggregate_to`` is stamped on **every** row. A table or a figure built from these is
    unreadable without it: "one point is one animal" and "one point is one session" are different
    claims about the same picture, and nothing else on the row distinguishes them.

    ``remainders`` are the labels whose rows count only a parent's leftover
    (``measures.remainder_labels``, stored on the record as ``remainder_labels``): ``label``
    stays the plain name, the join key, and ``label_display`` / ``label_is_remainder`` say so.
    Required, because a default of "none" is exactly the silent miss this column exists for.
    """
    from detail.analysis.measures import display_label

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
                "label_display": display_label(keyed.get("label", ""), remainders),
                "label_is_remainder": bool(keyed.get("label"))
                and keyed.get("label") in remainders,
                "window_id": keyed.get("window_id", ""),
                "value": row.value,
                "n_underlying": row.n_underlying,
                "denominator_ns": (
                    "" if row.denominator_ns is None else row.denominator_ns
                ),
            }
        )
    return out
