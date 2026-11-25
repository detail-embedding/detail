"""Measures — the §19 behavior artifact turned into numbers (§23).

How much of a behavior, how often, how long, in what order. Pure functions: artifact in, tidy
table out, with no executor, no Qt, and **no knowledge of which module produced the instances**.
Every producer therefore inherits every measure, and a measure added later applies to every
artifact already on disk.

**Everything is a windowed question and the answer is exact.** The whole recording is simply the
window covering everything (``analysis/windows.py``), so there is no separate per-recording path.
The artifact stores intervals, so no binning happens anywhere — binning is a choice of where to
*evaluate*, and it never changes what an answer is.

**The framework applies the boundary rule; a measure only computes.** An instance straddling a
window edge has to be handled, and the right handling differs per measure — but getting it wrong
does not fail, it produces plausible numbers that do not add up. So the rule is declared once per
measure and applied here, rather than re-derived in ten functions:

- **onset** — the instance counts if it *starts* in the window, and its **full** extent is used.
  Keeps counts additive across adjacent windows and keeps a duration honest. Measuring duration by
  overlap instead would truncate every bout at the edge and report bouts shorter than they were.
- **overlap** — the instance contributes the part of itself inside the window. Keeps total time
  additive.
- **pair** — needs two consecutive instances, so a pair counts when both are in the window.

**An instance is measured half-open, ``[t_ns, end_t_ns + period)``**, even though §19 *stores*
``end_t_ns`` inclusive. Inclusive storage is right — an instance is a thing with a first and last
frame, and it matches the §3 behavior stream so an artifact re-ingests — but measuring it that way
computes ``end_t_ns - t_ns``, which is **zero for a single-frame bout** and short by one period for
every other. A bout over frames 10–12 covers three frames, so it lasts three periods. Denominators
are frame-period counts too (``frame_view.observed_ns_in``), which is what makes ``time_fraction``
mean anything.

**Counting goes through ``behavior_output.bouts()``.** ``instances.csv`` is one row per
``(bout, label)``, so a cluster named twice appears twice over one span; the §19 audit counted rows
and reported precision 0.667 where the truth was 0.5. Anything that counts collapses first.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median, pstdev

from detail.analysis.clip_query import nominal_period_ns
from detail.analysis.windows import Window
from detail.storage import behavior_output, frame_view

ONSET = "onset"
OVERLAP = "overlap"
PAIR = "pair"
RULES = frozenset({ONSET, OVERLAP, PAIR})

# The one place a label's name is absent but its rows exist: a discovered cluster nobody named.
UNNAMED = ""

#: A manifest condition: what it means, and how to read it off an artifact's manifest.
#:
#: Each entry answers "could this artifact *ever* produce a meaningful value", which is a
#: structural question, not a question about the data. That distinction is what keeps two
#: comparisons of the same run offering the same measures.
CONDITIONS: dict[str, tuple[str, "Callable[[dict], bool]"]] = {
    "exclusive": (
        "the behaviors can overlap, so 'what came next' has no answer",
        lambda manifest: manifest.get("overlap") == "exclusive",
    ),
    "unnamed_labels": (
        "every behavior in this run is named, so there is no unnamed time to measure",
        # A clustering producer writes an empty list for its `unlabelled` source label
        # (`pipeline/temporal_clustering._manifest`); a query producer names every behavior
        # by construction, so no entry is ever empty. Keyed on the fact rather than on
        # `producer.module_id`, which would need editing here for every producer added later.
        # The real producer's output is what guards this, not a hand-written manifest: the
        # predicate was once true of a fixture and of no run on disk
        # (`tests/analysis/test_measure_naming.py`).
        lambda manifest: any(
            not names for names in (manifest.get("names") or {}).values()
        ),
    ),
}


def unmet_condition(spec: "MeasureSpec", manifest: dict) -> str | None:
    """Why ``spec`` cannot be measured on this artifact, or ``None`` when it can.

    The one answer every surface reads. Without it the pickers offered all eleven measures for
    every artifact and the run path quietly dropped the ones that could not apply — so a user
    comparing a query run got a `Time in unnamed clusters` row of zeros, and `Transitions`
    silently produced nothing at all.
    """
    for condition in spec.requires:
        meaning, holds = CONDITIONS[condition]
        if not holds(manifest):
            return meaning
    return None


def applicable_measures(manifest: dict) -> list[str]:
    """The measures worth computing for this artifact, in registry order."""
    return [
        measure_id
        for measure_id, spec in MEASURES.items()
        if unmet_condition(spec, manifest) is None
    ]


COLUMNS = [
    "recording_id",
    "subject",
    "label",
    "label_to",
    "label_display",
    "label_is_remainder",
    "window_id",
    "window_start_t_ns",
    "window_end_t_ns",
    "measure",
    "value",
    "denominator_ns",
    "n",
]

_NS_PER_S = 1_000_000_000

#: What a parent label's row is called: it counts only the time none of its children took.
REMAINDER_SUFFIX = " (other)"


def remainder_labels(manifest: dict) -> frozenset[str]:
    """The labels whose rows count only a **remainder**: every label with another label nested
    under it (its ``label_paths`` path a proper prefix of another label's).

    Labels nest (§22) and an instance carries its *deepest* label, so a parent's row counts only
    what none of its children took. **Owner decision (#24): the numbers stay, the row says so** —
    ``label`` keeps the plain name, because it is the join key a comparison groups on
    (``compare.MEASURE_KEYS``) and a user joins to ``instances.csv`` on; ``label_display`` reads
    "<label> (other)" and ``label_is_remainder`` marks the row. Decided here, once, from the
    manifest, so the measures table and a comparison's ``stats.csv`` cannot disagree. A producer
    with no hierarchy (a query run) writes no ``label_paths`` and has no remainders.
    """
    from detail.storage import behavior_output

    names = behavior_output._names_map(manifest)
    paths = behavior_output._names_map(manifest, key="label_paths")
    path_of: dict[str, str] = {}
    for source, labels in names.items():
        here = paths.get(source) or []
        for i, label in enumerate(labels):
            path_of[label] = (here[i] if i < len(here) else "") or label
    every = set(path_of.values())
    return frozenset(
        label
        for label, path in path_of.items()
        if any(other.startswith(path + "/") for other in every)
    )


def display_label(label: str, remainders) -> str:
    """``label`` as a person reads it: a parent's remainder row is "<label> (other)"."""
    return f"{label}{REMAINDER_SUFFIX}" if label and label in remainders else label


class MeasureError(ValueError):
    """A measure asked for something the artifact cannot answer."""


@dataclass(frozen=True)
class MeasureSpec:
    """One registered measure and everything the framework needs to drive it.

    ``rule`` decides which instances reach ``fn`` (see the module docstring) — the single most
    consequential field, and the one whose mistakes are invisible.

    ``additive`` marks a measure whose values over adjacent windows must sum to the value over
    their union. That is the guard which catches a wrong ``rule``, so it is declared rather than
    inferred.

    ``duration_shaped`` marks a measure **undefined for a point behavior**: a point is an instant,
    and half-open measurement would hand it exactly one period — an artifact of frame quantization
    dressed as a duration, so those return ``None``. It is not a statement about the value's units:
    ``time_fraction`` carries the flag and returns a ratio. ``unit`` is what says what a value *is*.

    ``unit`` is how a surface renders a number. Without it a table of measures is a column of
    unlabelled floats mixing nanoseconds, counts, ratios and per-second rates — and the reader
    cannot tell which is which. ``"ns"`` is a duration, ``"1/s"`` a rate, ``"fraction"`` a
    0-to-1 ratio, and ``""`` a plain count.

    ``scope`` is ``label`` for almost everything. ``unit`` gives the measure the whole ordered bout
    sequence, which a transition needs because "the label that came next" is a fact about the
    sequence rather than about one label's instances.

    ``fans_out`` marks a measure returning ``{label_to: value}`` rather than one number, which the
    emitter turns into one row per pair.

    ``requires`` names the manifest conditions the measure needs. It replaced a hardcoded
    ``requires_exclusive`` flag because the same problem had a second instance nobody had
    declared: ``unnamed_time`` counts time in clusters a user has not named, and a *query*
    artifact has no unnamed clusters by construction — so a comparison of a query run carried a
    row of zeros that could not say anything. A condition names the fact rather than the
    producer, so a module added later inherits the right answer without editing this file.

    ``label`` and ``description`` are what a *person* reads. Without them ``measure_id`` was the
    only text a measure had, so every surface printed it and a comparison table read
    ``n_instances__rearing__whole`` — a user looking at one said the page had "too many ``_``".
    The id stays the identity: it is the value in ``stats.csv``, the CLI's ``--measure``
    argument, and the filename of every figure and table, none of which may drift. The label is
    for rendering only. This follows ``Tool``/``ModuleUI``/``SweepUI``, each of which pairs an
    id with a display ``label`` (``core/protocols.py``, ``gui/modules/registry.py``).
    """

    measure_id: str
    rule: str
    fn: Callable
    label: str = ""
    description: str = ""
    unit: str = ""
    additive: bool = False
    duration_shaped: bool = False
    scope: str = "label"
    fans_out: bool = False
    #: Manifest conditions this measure needs, by name (see ``CONDITIONS``). Declared rather
    #: than checked inside the computation, so a surface can say *why* a measure is missing
    #: instead of a user finding a column of zeros. `DESIGN_PLAN` §23 specifies this shape.
    requires: tuple[str, ...] = ()
    needs_denominator: bool = False
    #: What a figure's axis says after the behavior's name, in the unit a reader sees:
    #: ``Rearing (% time)``, ``Rearing (bouts)``. Distinct from ``label`` because an axis
    #: names the thing and its unit, and "Share of time (%)" names neither the behavior nor
    #: what the number is of. Empty for a measure that is not drawn (``fans_out``).
    axis: str = ""


MEASURES: dict[str, MeasureSpec] = {}


def spec_of(measure_id: str):
    """What a surface needs to print a measure, whatever source it belongs to.

    A behavior measure's :class:`MeasureSpec`, a location measure's
    ``location_measures.LocationMeasure``, or ``None`` for an id neither registry has. Both
    carry ``label``, ``description``, ``unit`` and ``fans_out``, which is all a table, an axis
    or a pseudo-count asks. Every renderer goes through here: one that read :data:`MEASURES`
    directly printed a location measure by its raw id, with no unit.
    """
    spec = MEASURES.get(measure_id)
    if spec is not None:
        return spec
    from detail.analysis import location_measures

    return location_measures.MEASURES.get(measure_id)


def register_measure(measure_id: str, **kwargs) -> Callable[[Callable], Callable]:
    """Register ``fn(selection, ctx) -> value`` under ``measure_id``.

    Additive by registration, like the other registries in this package
    (``segmentation.TRANSITION_DETECTORS``, ``segmentation.CLUSTER_RULES``): a new measure
    reaches every artifact and every surface at once.
    """

    def deco(fn: Callable) -> Callable:
        if kwargs.get("rule") not in RULES:
            raise MeasureError(f"{measure_id}: rule must be one of {sorted(RULES)}")
        # Required, not defaulted to the id: a measure that falls back to its id renders as one
        # everywhere and nobody notices until a user reads the table.
        for field in ("label", "description"):
            if not (kwargs.get(field) or "").strip():
                raise MeasureError(f"{measure_id}: a {field} is required")
        MEASURES[measure_id] = MeasureSpec(measure_id, fn=fn, **kwargs)
        return fn

    return deco


@dataclass(frozen=True)
class Context:
    """What a measure gets besides its instances."""

    window: Window
    period_ns: int
    denominator_ns: int | None
    #: The unit's whole time-ordered bout list. A `scope="unit"` measure is handed it as its
    #: pool anyway; what this is *for* is a **label**-scoped measure that needs the sequence
    #: around its own bouts — `transitions` looks up each bout's successor here.
    unit_bouts: Sequence
    behavior_types: dict = field(default_factory=dict)

    @property
    def denominator_s(self) -> float:
        if not self.denominator_ns:
            raise MeasureError("no observed time in this window")
        return self.denominator_ns / _NS_PER_S


@dataclass(frozen=True)
class Selection:
    """The instances a rule admitted, with their measured spans and window overlaps.

    ``spans`` and ``overlaps`` are parallel to ``instances`` and precomputed here so that no
    measure re-derives the half-open span — the arithmetic that is wrong by one period when done
    from the stored inclusive bounds.
    """

    instances: list
    spans: list[tuple[int, int]]
    overlaps: list[int]
    pairs: list[tuple]

    @property
    def n(self) -> int:
        return len(self.pairs) if self.pairs else len(self.instances)


def measured_span(instance, period_ns: int) -> tuple[int, int]:
    """An instance's half-open span, ``[t_ns, end_t_ns + period)`` — see the module docstring."""
    return int(instance.t_ns), int(instance.end_t_ns) + int(period_ns)


def select(instances: Sequence, rule: str, ctx: Context) -> Selection:
    """Apply ``rule`` to ``instances`` for ``ctx.window``.

    ``instances`` must already be bouts (``behavior_output.bouts()``) and time-ordered; both are
    the caller's job because both are shared across every measure of a unit.
    """
    spans = [measured_span(row, ctx.period_ns) for row in instances]
    if rule == ONSET:
        keep = [
            i
            for i, row in enumerate(instances)
            if ctx.window.contains_instant(row.t_ns)
        ]
    elif rule in (OVERLAP, PAIR):
        keep = [i for i, s in enumerate(spans) if ctx.window.overlap_ns(*s) > 0]
        if rule == PAIR:
            keep = [i for i in keep if ctx.window.contains_instant(instances[i].t_ns)]
    else:  # pragma: no cover - register_measure rejects an unknown rule
        raise MeasureError(f"unknown rule {rule!r}")
    chosen = [instances[i] for i in keep]
    chosen_spans = [spans[i] for i in keep]
    pairs = list(zip(chosen, chosen[1:])) if rule == PAIR else []
    return Selection(
        instances=chosen,
        spans=chosen_spans,
        overlaps=[ctx.window.overlap_ns(*s) for s in chosen_spans],
        pairs=pairs,
    )


# --- the measures ---------------------------------------------------------------------- #


@register_measure(
    "n_instances",
    rule=ONSET,
    additive=True,
    label="Bouts",
    description="How many separate bouts of the behavior occurred.",
    axis="bouts",
)
def _n_instances(sel: Selection, ctx: Context) -> float:
    return float(len(sel.instances))


@register_measure(
    "rate",
    rule=ONSET,
    unit="1/s",
    needs_denominator=True,
    label="Rate",
    description="Bouts per second of observed time — not per second of window length.",
    axis="bouts/s",
)
def _rate(sel: Selection, ctx: Context) -> float:
    """Onsets per second of **observed** time — never per second of window length."""
    return len(sel.instances) / ctx.denominator_s


@register_measure(
    "total_time",
    rule=OVERLAP,
    unit="ns",
    additive=True,
    duration_shaped=True,
    label="Total time",
    description="How much time the behavior occupies in total.",
    axis="s in total",
)
def _total_time(sel: Selection, ctx: Context) -> float:
    return float(sum(sel.overlaps))


@register_measure(
    "time_fraction",
    rule=OVERLAP,
    unit="fraction",
    duration_shaped=True,
    label="Share of time",
    description="What proportion of the observed time the behavior occupies.",
    needs_denominator=True,
    axis="% time",
)
def _time_fraction(sel: Selection, ctx: Context) -> float:
    return sum(sel.overlaps) / ctx.denominator_ns


def _durations(sel: Selection) -> list[int]:
    """Full extents of the admitted instances — never the windowed overlap.

    An onset-rule measure sees the whole instance even where it runs past the window's end;
    truncating here is what would report bouts shorter than they were.
    """
    return [hi - lo for lo, hi in sel.spans]


@register_measure(
    "duration_mean",
    rule=ONSET,
    unit="ns",
    duration_shaped=True,
    label="Mean bout length",
    description="How long a typical bout lasts.",
    axis="s per bout, mean",
)
def _duration_mean(sel: Selection, ctx: Context) -> float | None:
    d = _durations(sel)
    return sum(d) / len(d) if d else None


@register_measure(
    "duration_median",
    rule=ONSET,
    unit="ns",
    duration_shaped=True,
    label="Median bout length",
    description="The middle bout length, unmoved by a few very long or short bouts.",
    axis="s per bout, median",
)
def _duration_median(sel: Selection, ctx: Context) -> float | None:
    d = _durations(sel)
    return float(median(d)) if d else None


@register_measure(
    "duration_sd",
    rule=ONSET,
    unit="ns",
    duration_shaped=True,
    label="Bout length spread",
    description="How variable the bout lengths are (population sd).",
    axis="s, bout length sd",
)
def _duration_sd(sel: Selection, ctx: Context) -> float | None:
    """Population sd, not sample: these are the bouts that occurred, not a draw from a larger set."""
    d = _durations(sel)
    return float(pstdev(d)) if len(d) > 1 else None


@register_measure(
    "latency_to_first",
    rule=ONSET,
    unit="ns",
    duration_shaped=True,
    label="Time to first bout",
    description="How long from the start of the window until the behavior first occurs.",
    axis="s to first bout",
)
def _latency_to_first(sel: Selection, ctx: Context) -> float | None:
    """Time from the window's start to the first onset. ``None`` when the behavior never occurs —
    not zero, and not the window length, both of which are real answers to different questions.
    """
    if not sel.instances:
        return None
    return float(min(row.t_ns for row in sel.instances) - ctx.window.start_t_ns)


@register_measure(
    "inter_bout_interval",
    rule=PAIR,
    unit="ns",
    duration_shaped=True,
    label="Gap between bouts",
    description="The mean pause between consecutive bouts — the gap, not the bout.",
    axis="s between bouts",
)
def _inter_bout_interval(sel: Selection, ctx: Context) -> float | None:
    """Mean gap between consecutive bouts of one label — the pause, not the bout."""
    gaps = [
        int(nxt.t_ns) - (int(prev.end_t_ns) + ctx.period_ns) for prev, nxt in sel.pairs
    ]
    return sum(gaps) / len(gaps) if gaps else None


@register_measure(
    "transitions",
    rule=PAIR,
    fans_out=True,
    requires=("exclusive",),
    label="Transitions",
    description="How often this behavior is followed by another.",
)
def _transitions(sel: Selection, ctx: Context) -> dict:
    """Counts of ``label -> label_to``: how often *this* behavior's bouts are followed by each
    other behavior. §23's measure table: "one row per ordered pair; ``label`` is the from,
    ``label_to`` the to".

    **Label-scoped, and reading `ctx.unit_bouts`.** It used to be ``scope="unit"``, which hands a
    measure the whole unit and no from-label — so it bound ``prev`` in its loop, never read it,
    and counted *destinations*: every from-label's transitions summed into one row keyed with a
    blank from. R→G and G→R were the same number. `unit_bouts` is exactly the input a successor
    lookup needs and nothing read it; this is what it is for.

    ``requires=("exclusive",)`` because once two labels can be active at once, "next" is
    undefined — the measure is withheld rather than answered with whichever row happened to sort
    first.

    A bout's successor is looked up in the *unit's* sequence, not in the window's selection, so a
    transition out of the last bout before a window edge is still counted — once, in the window
    holding the bout it leaves. That is what keeps the measure additive across adjacent windows.
    """
    order = {row.instance_id: i for i, row in enumerate(ctx.unit_bouts)}
    out: dict[str, int] = {}
    for row in sel.instances:
        index = order.get(row.instance_id)
        if index is None or index + 1 >= len(ctx.unit_bouts):
            continue
        destination = ctx.unit_bouts[index + 1].label or UNNAMED
        out[destination] = out.get(destination, 0) + 1
    return {k: float(v) for k, v in out.items()}


@register_measure(
    "unnamed_time",
    rule=OVERLAP,
    unit="ns",
    scope="unit",
    duration_shaped=True,
    requires=("unnamed_labels",),
    label="Time in unnamed clusters",
    description="Observed time in real clusters that have not been named yet.",
    axis="s in unnamed clusters",
)
def _unnamed_time(sel: Selection, ctx: Context) -> float:
    """Observed time in real clusters the user has not named yet.

    Reported separately from noise on purpose (§23): "31% is in clusters you have not named" and
    "12% is noise" are different facts about a run's completeness, and one "unattributed" bucket
    hides which one you are looking at.
    """
    total = 0
    for row, span in zip(sel.instances, sel.spans):
        if not row.label:
            total += ctx.window.overlap_ns(*span)
    return float(total)


# ``noise_time`` is deliberately absent. §22 gives noise a *reserved* ``source_label``, but neither
# §19 nor ``storage/behavior_output.py`` fixes what that value is, and no producer emits one yet.
# Choosing it here would mean a reader inventing a convention the writer never agreed to — and the
# guard would be unexercised, which this project has just relearned is a guard that passes forever.
# The measure lands with the first producer that has noise; pose clustering (§22) was to be that
# producer until it was removed, and neither temporal method leaves bouts unassigned.


# --- the emitter ----------------------------------------------------------------------- #


def _period_ns(view, coverage) -> int:
    """The subject's frame period.

    From the frame view's own axis where there is one. Otherwise from coverage, which knows how
    much time it observed and over how many rows — the mean observed frame duration, which is the
    honest stand-in and needs no guessing about cadence.
    """
    if view is not None and view.n_frames > 1:
        return int(nominal_period_ns(view.t_ns[:]))
    if coverage.n_observed:
        return max(int(coverage.observed_ns // coverage.n_observed), 1)
    return 1


def _denominator_ns(view, coverage, window: Window) -> int | None:
    """Observed time inside ``window``.

    With a frame view this is exact (``observed_ns_in``). Without one it is answerable **only** for
    the whole-coverage window, where ``coverage.observed_ns`` is exactly right — and refused for
    any sub-window, because ``coverage`` records how much was unobserved and not *where*. Silently
    substituting a whole-recording denominator into a sub-window inflates or deflates every rate in
    proportion to where the gaps happen to fall, which is the specific error the frame view exists
    to prevent.
    """
    if view is not None:
        return frame_view.observed_ns_in(view, window.start_t_ns, window.end_t_ns)
    if (window.start_t_ns, window.end_t_ns) == (coverage.start_t_ns, coverage.end_t_ns):
        return int(coverage.observed_ns)
    raise MeasureError(
        f"a windowed denominator needs the frame view: {coverage.unit} has none, so only the "
        f"whole-coverage window can be measured. coverage.observed_ns records how much time was "
        f"observed, not where the gaps fall."
    )


def _row(
    recording_id,
    subject,
    label,
    label_to,
    window,
    measure_id,
    value,
    denom,
    n,
    remainders,
):
    return {
        "recording_id": recording_id,
        "subject": "" if subject is None else subject,
        "label": label,
        "label_to": label_to,
        "label_display": display_label(label, remainders),
        "label_is_remainder": bool(label) and label in remainders,
        "window_id": window.window_id,
        "window_start_t_ns": window.start_t_ns,
        "window_end_t_ns": window.end_t_ns,
        "measure": measure_id,
        "value": "" if value is None else value,
        "denominator_ns": "" if denom is None else denom,
        "n": n,
    }


def measure_unit(
    instances: Sequence,
    coverage,
    manifest: dict,
    windows: Sequence[Window],
    *,
    view=None,
    measures: Sequence[str] | None = None,
) -> list[dict]:
    """Tidy rows for one ``(recording, subject)``. The pure core — nothing here reads a disk.

    ``instances`` are this unit's rows straight from the artifact; they are collapsed to bouts and
    time-ordered here, once, because every measure of the unit needs the same collapse and the
    consumers that reimplemented it are the ones that got it wrong.

    **The label universe is the manifest's ``vocabulary``, not the labels present.** A vocabulary
    label with no instances inside coverage is a **real zero** and belongs in the table; leaving it
    out makes an absence indistinguishable from a recording nobody analyzed, which is the
    distinction §19's coverage exists to preserve.
    """
    # "All of them" means all the ones this artifact can answer. Expanding to the whole registry
    # is what put a `Time in unnamed clusters` row of zeros on every query comparison.
    chosen = list(measures) if measures else applicable_measures(manifest)
    unknown = sorted(set(chosen) - set(MEASURES))
    if unknown:
        raise MeasureError(f"unknown measure(s): {unknown}. Known: {sorted(MEASURES)}")

    rows_in = list(instances)
    # Collapse **per label**, not globally. ``bouts()`` keeps the first row of each
    # ``instance_id``, so collapsing first and filtering after loses every label but one — a
    # cluster the user named twice reports zero instances under its second name, which is what
    # this order was written the wrong way round to begin with and a test caught. Within one label
    # a bout appears once anyway, so the per-label collapse is a defensive no-op that leaves the
    # fan-out intact; the global collapse is still what a ``scope="unit"`` measure wants — one
    # entry per bout, whatever it was named.
    by_label: dict[str, list] = {}
    for row in rows_in:
        by_label.setdefault(row.label, []).append(row)
    for label, group in by_label.items():
        by_label[label] = sorted(
            behavior_output.bouts(group), key=lambda r: int(r.t_ns)
        )
    bouts = sorted(behavior_output.bouts(rows_in), key=lambda r: int(r.t_ns))
    period = _period_ns(view, coverage)
    types = manifest.get("behavior_types") or {}
    vocabulary = sorted(manifest.get("vocabulary") or [])
    remainders = remainder_labels(manifest)

    rows: list[dict] = []
    for window in windows:
        denom = _denominator_ns(view, coverage, window)
        ctx = Context(
            window=window,
            period_ns=period,
            denominator_ns=denom,
            unit_bouts=bouts,
            behavior_types=types,
        )
        for measure_id in chosen:
            spec = MEASURES[measure_id]
            # Not redundant with the expansion above: that one decides what "all of them"
            # means, this one catches a measure a caller *named* explicitly — `--measure
            # transitions` on an overlapping artifact. Dropping it is the declared behaviour
            # (`pipeline/query.py` says choosing `overlap: allowed` "costs §23's transitions").
            if unmet_condition(spec, manifest) is not None:
                continue
            if spec.needs_denominator and not denom:
                # A window with no observed time is **no measurement**, not a zero rate. Emitting
                # 0.0 would enter a group mean as a real observation of "never happened".
                continue
            labels = [None] if spec.scope == "unit" else vocabulary
            for label in labels:
                pool = bouts if spec.scope == "unit" else by_label.get(label, [])
                sel = select(pool, spec.rule, ctx)
                if spec.duration_shaped and label and types.get(label) == "point":
                    # A point behavior is an instant; half-open measurement would hand it exactly
                    # one period, which is frame quantization dressed as a duration.
                    rows.append(
                        _row(
                            coverage.recording_id,
                            coverage.subject,
                            label,
                            "",
                            window,
                            measure_id,
                            None,
                            denom,
                            sel.n,
                            remainders,
                        )
                    )
                    continue
                value = spec.fn(sel, ctx)
                if spec.fans_out:
                    for to_label, count in sorted((value or {}).items()):
                        rows.append(
                            _row(
                                coverage.recording_id,
                                coverage.subject,
                                UNNAMED if label is None else label,
                                to_label,
                                window,
                                measure_id,
                                count,
                                denom,
                                sel.n,
                                remainders,
                            )
                        )
                    continue
                rows.append(
                    _row(
                        coverage.recording_id,
                        coverage.subject,
                        UNNAMED if label is None else label,
                        "",
                        window,
                        measure_id,
                        value,
                        denom,
                        sel.n,
                        remainders,
                    )
                )
    return rows


def measure_run(
    run_dir: Path | str,
    *,
    windows_for: Callable[[object], Sequence[Window]] | None = None,
    measures: Sequence[str] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Tidy rows for every unit in a behavior-output run — the thin loader over the pure core.

    ``windows_for(coverage_row)`` builds that unit's windows, defaulting to the whole coverage
    span. It takes the coverage row rather than a fixed list because a normalized or absolute
    window set is per recording: recordings differ in length, which is the whole reason
    ``windows.normalized`` exists.

    Reads each unit's frame view lazily and drops it before the next, so peak memory stays
    proportional to one subject rather than a cohort (a 100-hour unit is 21.6M rows).

    ``progress_cb(units_done, units_total)`` reports per unit. This is the whole pass a caller
    waits on — before it existed, ``run_comparison`` emitted ``measuring 0.0`` and then
    ``measuring 1.0`` with a cohort's worth of work in between, so the bar sat frozen for the
    entire run and read as a hang. ``stop_check()`` is polled at the same boundary and returns
    the rows measured so far rather than raising: a stopped run still has usable numbers.
    """
    run_dir = Path(run_dir)
    artifact = behavior_output.read_behavior_output(run_dir)
    by_unit: dict[tuple, list] = {}
    for row in artifact.instances:
        by_unit.setdefault((row.recording_id, row.subject), []).append(row)

    rows: list[dict] = []
    total = len(artifact.coverage)
    for done, coverage in enumerate(artifact.coverage):
        if stop_check is not None and stop_check():
            break
        if progress_cb is not None:
            progress_cb(done, total)
        try:
            view = frame_view.read_frame_view(
                run_dir, coverage.recording_id, coverage.subject
            )
        except (FileNotFoundError, KeyError):
            view = None
        windows = (
            windows_for(coverage)
            if windows_for
            else [Window("whole", coverage.start_t_ns, coverage.end_t_ns)]
        )
        rows.extend(
            measure_unit(
                by_unit.get(coverage.unit, []),
                coverage,
                artifact.manifest,
                windows,
                view=view,
                measures=measures,
            )
        )
    if progress_cb is not None:
        progress_cb(total, total)
    return rows


def write_measures(path: Path | str, rows: list[dict]) -> Path:
    """Write the tidy table, atomically, through the shared long-form writer."""
    from detail.report.tables import write_tidy_csv

    write_tidy_csv(path, rows, COLUMNS)
    return Path(path)
