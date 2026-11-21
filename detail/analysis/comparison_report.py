"""The export tree for a §24 comparison: tables, figures, summary, manifest.

Reads a saved comparison's record and its ``stats.csv`` and writes the publication artifacts. Pure
re-render — no recompute — so re-exporting is deterministic and byte-identical (§18.5 L1).

**Its own entry point.** The removed ``EXPORTERS`` registry dispatched on
``record["member_module"]``, a *benchmark* concept: a benchmark's members were eval runs and the key
selected which module's figures to draw. A comparison has no members, so routing it through would
have meant inventing a fake member module to satisfy a lookup. The `report/` primitives underneath
are shared exactly as they are.

Domain figures live with the module and never in ``report/`` (§18.4) — the same rule that keeps
query-eval's PR curves out of the shared package.

**matplotlib is imported where a figure is drawn, not at module scope.** Reading `stats.csv`,
formatting a table and naming a measurement need none of it, and this module is imported by the
GUI's comparison page — which put a 160 ms plotting-stack import into app startup for surfaces
that never draw anything. `detail/report/figures.py` is legitimately a matplotlib module; this
one is not.
"""

import dataclasses
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Callable, Mapping

from detail.analysis import compare, provenance
from detail.analysis import windows as windows_lib
from detail.core.utils import safe_stem
from detail.report import exporter as exporter_lib
from detail.report import tables as tablib
from detail.report.style import DEFAULT_STYLE, FigureStyle

STATS_FILENAME = "stats.csv"
NODES_FILENAME = "nodes.csv"
RANKING_FILENAME = "ranking.csv"
#: The statistics behind every result that cleared, one row per measure and contrast.
SIGNIFICANT_FILENAME = "significant.csv"
#: One row per unit and measurement, one column per group: the shape a person pastes into
#: Prism or R.
UNITS_FILENAME = "units.csv"
#: The per-unit effects a within-unit ranking averaged, one row per unit and measurement.
UNIT_EFFECTS_FILENAME = "unit_effects.csv"
#: A paragraph for the Methods, generated from the record so it cannot disagree with what ran.
METHODS_FILENAME = "methods.txt"
#: The figure formats every surface writes unless told otherwise: a PDF with editable type for
#: a manuscript, an SVG rewritten for editing by hand, and the PNG the GUI shows.
DEFAULT_FORMATS = ("pdf", "svg", "png")
#: The composite figure's stem: the chosen measurements as one row or grid of panels.
PANELS_STEM = "panels"
#: How many panels the composite holds at most; past it the per-measurement figures stand alone.
PANELS_MAX = 12
TABLE_COLUMNS = ["group", "n", "coverage", "mean", "median", "sd"]
SIGNIFICANT_COLUMNS = [
    "rank",
    "measure",
    "label",
    "label_to",
    "window_id",
    "measurement",
    "contrast",
    "pair_a",
    "pair_b",
    "a_mean",
    "a_sem",
    "a_n",
    "b_mean",
    "b_sem",
    "b_n",
    "unit",
    "effect",
    "change",
    "statistic",
    "p",
    "p_kind",
    "family_p",
    "own_p",
    "q",
    "n_units",
    "in_direction",
]
UNITS_COLUMNS_FIXED = [
    "measure",
    "label",
    "label_to",
    "window_id",
    "measurement",
    "unit",
    "observation",
]
UNIT_EFFECTS_COLUMNS = [
    "measure",
    "label",
    "label_to",
    "window_id",
    "measurement",
    "observation",
    "effect",
    "effect_is",
    "pair_a",
    "pair_b",
]
RANKING_COLUMNS = [
    "rank",
    "measure",
    "label",
    "label_to",
    "label_display",
    "label_is_remainder",
    "window_id",
    "effect",
    "magnitude",
    "statistic",
    "floor",
    "clears",
    "kept",
    "family_p",
    "own_p",
    "q",
    "n_units",
    "n_present",
    "consistency",
    "loo_sign_stable",
    "present_a",
    "present_b",
    "a_mean",
    "b_mean",
    "pair_a",
    "pair_b",
    "suggestion",
]
#: Over the node source, how many below-floor items (nesting suppressed) the per-measurement
#: tables and figures cover beside the kept set (§24 *The node source*).
SHOWN_BELOW_FLOOR = 10


#: The unit of a measure that is one quantity divided by another of the same kind (a body's
#: length over its width). It is named, because the report reads ``""`` as a plain count,
#: whose pseudo-count is one bout; and it is printed as nothing, because "2.4 ratio" is not
#: how anyone writes a ratio.
RATIO_UNIT = "ratio"


def measure_unit(measure_key: tuple) -> str:
    """The unit a measure's numbers are in, or ``""`` when it is a plain count.

    ``compare.py`` deliberately knows nothing about the measure registry — it works on whatever
    rows it is handed. Rendering is where the unit has to come back, or a reader gets a column of
    unlabelled floats mixing nanoseconds, counts and ratios, which is what every comparison
    surface shipped: a mean duration printed as ``4956702127.6595745``.
    """
    from detail.analysis.measures import spec_of

    spec = spec_of(measure_key[0] if measure_key else "")
    return spec.unit if spec is not None else ""


def shown_value(value, unit: str) -> tuple[str, str]:
    """``(text, unit)`` as a person reads them: nanoseconds become seconds.

    The one rule, so the figures, ``summary.md``, the per-measure tables and both GUI panels
    cannot disagree about what a number means. Lifted out of the §23 measures panel, which was the
    only surface in the app that had it — and which then kept its copy for a while, so a
    `time_fraction` read `0.317 fraction` there and `31.7 %` on the comparison page. "One rule"
    is only true while there is one function; `MeasuresPanel` calls this one.
    """
    if value is None or value == "":
        return "", unit
    if unit == "ns":
        return f"{float(value) / 1e9:.3g}", "s"
    if unit == "fraction":
        return f"{float(value) * 100:.3g}", "%"
    if unit == RATIO_UNIT:
        return f"{float(value):.4g}", ""
    return f"{float(value):.4g}", unit


def remainders_of(record: dict) -> frozenset[str]:
    """The labels a comparison's rows count only a parent's leftover of (#24), as recorded.

    Empty for a record written before the field existed and for a node comparison, whose labels
    are nodes."""
    return frozenset(record.get("remainder_labels") or ())


def measure_title(measure_key: tuple, remainders: Collection[str]) -> str:
    """What a *person* reads for one measurement: ``Bouts · rearing · whole recording``.

    The twin of :func:`_measure_stem`, which keeps producing the id form. The split is the point:
    a stem names a file and a `stats.csv` cell and must never drift, while a table row and a
    figure title are read by someone who did not write the registry. Printing the stem in both
    places is what put `n_instances__rearing__whole` on screen.

    Only the measure id maps to a label — the remaining parts of the key are the user's own
    behavior names, a transition's destination, and the window, all already human. A parent
    label whose rows count only its leftover reads "<label> (other)" (``remainders``, from
    :func:`remainders_of`; required, because leaving it out shows the leftover under the plain
    name, which reads as all of the parent).
    """
    from detail.analysis.measures import display_label, spec_of

    if not measure_key:
        return "measure"
    spec = spec_of(measure_key[0])
    parts = [spec.label if spec is not None else measure_key[0]]
    parts += [
        _readable_label(display_label(p, remainders) if i < 2 else p)
        for i, p in enumerate(measure_key[1:])
        if p and p != "whole"
    ]
    return "  ·  ".join(parts)


def _readable_label(label) -> str:
    """A node's label reads ``node 1192``, the id form INTERPRET shows; the rest is the user's."""
    from detail.analysis import node_measures

    node = node_measures.node_of(label)
    return f"node {node}" if node is not None else str(label)


def measure_axis_label(measure_key: tuple) -> str:
    """``total_time (s)`` — the measure with the unit a reader will actually see."""
    from detail.analysis.measures import spec_of

    unit = measure_unit(measure_key)
    shown = {"ns": "s", "fraction": "%", RATIO_UNIT: ""}.get(unit, unit)
    spec = spec_of(measure_key[0]) if measure_key else None
    name = spec.label if spec is not None else (measure_key[0] if measure_key else "")
    return f"{name} ({shown})" if shown else name


def panel_label(measure_key: tuple, remainders: Collection[str]) -> str:
    """What a figure's y axis says: the thing and its unit, ``Rearing (% time)``,
    ``node 1354 (bouts/s)``, ``Distance (m)`` (detail#76).

    A behavior measure names the behavior and says what of it is measured in the unit a
    reader sees (``MeasureSpec.axis``); a location measure is the thing itself. Falls back to
    :func:`measure_axis_label` for a measure that declares no axis.
    """
    from detail.analysis.measures import display_label, spec_of

    if not measure_key:
        return "measure"
    spec = spec_of(measure_key[0])
    axis = getattr(spec, "axis", "") if spec is not None else ""
    if not axis:
        return measure_axis_label(measure_key)
    label = measure_key[1] if len(measure_key) > 1 else ""
    if not label:
        # A location measure's axis is the thing itself; a behavior measure's is what of
        # the behavior is measured, and with no behavior to name it stands behind its label.
        return axis if axis.startswith(spec.label) else f"{spec.label} ({axis})"
    name = _readable_label(display_label(label, remainders))
    if len(measure_key) > 2 and measure_key[2]:
        name = f"{name} \u2192 {_readable_label(measure_key[2])}"
    # An axis starts with a capital, whatever case the behavior was typed in.
    return f"{name[:1].upper()}{name[1:]} ({axis})"


def _read_stats(comparison_dir: Path) -> list[dict]:
    import csv

    path = comparison_dir / STATS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no {STATS_FILENAME} in {comparison_dir}: a comparison writes its observations when "
            f"it runs, and the export re-renders them rather than recomputing."
        )
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _observations(rows: list[dict]) -> list[compare.Observation]:
    """Rebuild observations from the stored rows — the export never re-measures."""
    return [
        compare.Observation(
            key=tuple(row["observation"].split("|")) if row["observation"] else (),
            group=tuple(row["group"].split("|")) if row["group"] else (),
            measure_key=(
                row["measure"],
                row["label"],
                row["label_to"],
                row["window_id"],
            ),
            value=float(row["value"]),
            n_underlying=int(row["n_underlying"] or 1),
            # Absent from a stats.csv written before the column existed: those records are
            # between designs, which never read it.
            denominator_ns=(
                float(row["denominator_ns"]) if row.get("denominator_ns") else None
            ),
        )
        for row in rows
    ]


def read_hierarchy(comparison_dir: Path) -> dict[str, set[str]] | None:
    """The node source's ``nodes.csv`` as ``{label: ancestor labels}``, or ``None`` for a flat
    source — what nesting suppression reads back."""
    import csv

    from detail.analysis import node_measures

    path = Path(comparison_dir) / NODES_FILENAME
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return node_measures.ancestors_from_rows(list(csv.DictReader(handle)))


def pseudo_count(seconds: float) -> Callable[[tuple, float | None], float]:
    """``pseudo(measure_key, denominator_ns)``: ``seconds`` of time in the measure's own unit.

    A share of time needs the observation's denominator to say what one second is; without one
    it contributes nothing rather than a guess. A count's pseudo-count is one bout and a rate's
    one bout over the denominator, whatever ``seconds`` is: there is no bout per second. Both
    are off at ``seconds == 0``, so ``pseudo_count_s: 0`` ranks the raw ratios it records.
    """
    from detail.analysis.measures import spec_of

    def pseudo(measure_key: tuple, denominator_ns: float | None) -> float:
        spec = spec_of(measure_key[0] if measure_key else "")
        unit = spec.unit if spec is not None else ""
        if unit == "ns":
            return float(seconds) * 1e9
        if unit == "fraction":
            return float(seconds) * 1e9 / denominator_ns if denominator_ns else 0.0
        if unit == "1/s":
            return 1e9 / denominator_ns if denominator_ns and seconds > 0 else 0.0
        if unit == "":
            return 1.0 if seconds > 0 else 0.0
        return 0.0

    return pseudo


def rank_record(obs, record: dict, ancestors=None):
    """The one record → ranking mapping (§22, §24): the design the record resolved, over the
    observations it stored, with its floor — what the detail page, the export and a live screen
    all read, so none of them can disagree.

    ``None`` where there is nothing to rank: a between design with fewer than two groups. A
    record written before designs existed is a between design with no pseudo-count, so it ranks
    exactly as it always did over two groups; over more, each item is now the largest contrast
    between any two of them (§24 *Designs*) where it had no ranking at all.
    """
    design_kind = record.get("design") or compare.BETWEEN
    levels = tuple(record["levels"]) if record.get("levels") else None
    ranking_measure = record.get("ranking_measure")
    family = [
        o for o in obs if ranking_measure is None or o.measure_key[0] == ranking_measure
    ]
    draws = int(record.get("draws", 200))
    seed = int(record.get("seed", 0))
    if design_kind == compare.BETWEEN:
        groups = sorted({o.group for o in family})
        if levels is not None:
            groups = [g for g in groups if "|".join(g) in levels]
        if len(groups) < 2:
            return None
        # ε only over the node source: a label's appearance stays unbounded, as it always was.
        pseudo = (
            pseudo_count(float(record.get("pseudo_count_s", 0.0)))
            if record.get("source") == "nodes"
            else None
        )
        ranking = compare.rank_by_difference(
            family, tuple(groups), draws=draws, seed=seed, pseudo=pseudo
        )
    else:
        design = compare.Design(
            kind=design_kind,
            column=record.get("design_column"),
            levels=levels,
            unit=tuple(record.get("aggregate_to") or ()),
            reason=record.get("design_reason") or "",
        )
        ranking = compare.rank_by_design(
            obs,
            design,
            draws=draws,
            seed=seed,
            min_units=int(record.get("min_units") or 1),
            presence_ns=float(record.get("presence_s", 0.0)) * 1e9,
            pseudo=pseudo_count(float(record.get("pseudo_count_s", 0.0))),
            ranking_measure=ranking_measure,
            # The resolved statistic (§24 *The statistic*); a record written before the
            # field existed ranks on the effect, as it did when it was written.
            statistic=record.get("ranking_statistic") or compare.STATISTIC_EFFECT,
        )
    if ancestors is not None and record.get("suppress_nesting", True):
        ranking = compare.suppress_nesting(ranking, ancestors)
    if record.get("verdict") == compare.VERDICT_OWN:
        ranking = by_own_test(ranking)
    return ranking


def by_own_test(ranking):
    """``ranking`` read by each measure's own test (:data:`compare.VERDICT_OWN`): the same
    items and the same numbers, ordered by their own p, smallest first, and a finding where
    that p is at most :data:`compare.FLOOR_ALPHA`.

    The order is by evidence rather than by size because the sizes are not comparable: a
    distance that halves and a body that shortens by a twentieth are in different units, and
    the larger number is not the surer finding. A measure with no p goes last.
    """

    def order(item):
        p = getattr(item, "own_p", None)
        return (p is None, p if p is not None else 0.0, -(item.magnitude or 0.0))

    return dataclasses.replace(
        ranking,
        verdict=compare.VERDICT_OWN,
        contrasts=sorted(ranking.contrasts, key=order),
    )


def left_out_sentence(record: dict, names: Mapping[str, str] | None = None) -> str:
    """The recordings a comparison left out for having no value in a grouping column (§24),
    in one wording for the summary, the detail page, the launch preview and `comparison run`.
    Empty when none was. ``names`` shows a recording's name in place of its id where a
    surface has them."""
    ids = list(record.get("left_out") or [])
    if not ids:
        return ""
    columns = [c for c in (record.get("group_by") or []) if c not in compare.ROW_KEYS]
    where = " / ".join(f"'{c}'" for c in columns) or "a grouping column"
    shown = ", ".join((names or {}).get(i, i) for i in ids)
    return f"{len(ids)} recording(s) left out, no value for {where}: {shown}"


def _unit_word(record: dict) -> str:
    return "|".join(record.get("aggregate_to") or []) or "recording"


def screened_note(record: dict, label: str) -> str:
    """Why this label's effect is not an independent finding, when a difference screen over
    the compared column chose its cluster (``screened_labels``, §24): "chosen from screen
    cmp_01 (cluster #1375 kept)". Empty for every other label. One wording, for the table, the
    summary and the CLI alike."""
    origins = (record.get("screened_labels") or {}).get(label) or []
    if not origins:
        return ""
    parts = []
    for origin in origins:
        relation = origin.get("relation", "")
        where = f"cluster #{origin.get('kept', '?')}"
        if relation == "under":
            where = f"inside {where}"
        elif relation == "above":
            where = f"above {where}"
        parts.append(
            f"{origin.get('comparison', '?')} ({where} kept{kept_at_text(origin)})"
        )
    return "chosen from screen " + ", ".join(parts)


def kept_at_text(origin: Mapping[str, str]) -> str:
    """`` at p ≤ 0.2`` when the screen was read looser or stricter than its own floor when the
    label was written (``LabelOrigin.alpha``, §22); ``""`` at the floor, or for an origin
    written before strictness existed."""
    alpha = str(origin.get("alpha") or "")
    if not alpha:
        return ""
    try:
        value = float(alpha)
    except ValueError:
        return ""
    if abs(value - compare.FLOOR_ALPHA) < 1e-9:
        return ""
    return f" at p ≤ {value:g}"


def screened_sentence(record: dict) -> str:
    """The header's line for a labels comparison some of whose labels a screen over the same
    column chose: "rearing was chosen from screen cmp_01 over this column, so its effect here
    is not an independent finding." Empty when none was."""
    screened = record.get("screened_labels") or {}
    if not screened:
        return ""
    names = sorted(screened)
    cited = ", ".join(
        f"{name} ({', '.join(sorted({o.get('comparison', '?') for o in screened[name]}))})"
        for name in names
    )
    column = ", ".join(record.get("group_by") or []) or "this column"
    verb = "was" if len(names) == 1 else "were"
    return (
        f"{cited} {verb} chosen from a difference screen over {column}, so the effect here "
        f"is not an independent finding: the screen selected the cluster this comparison "
        f"scores."
    )


def suggestion(item, ranking, record: dict) -> str:
    """What the item says, in one sentence a user can act on — "where to look", never a result
    (§22). Generated from the design, so the same effect reads the same everywhere. A label a
    screen over this column chose says so after its sentence (:func:`screened_note`)."""
    text = _suggestion(item, ranking, record)
    note = screened_note(record, str(item.measure_key[1]))
    return f"{text} · {note}" if note else text


def _suggestion(item, ranking, record: dict) -> str:
    unit = _unit_word(record)
    if isinstance(item, compare.Effect):
        if item.effect is None:
            return (
                f"undefined: present in {item.n_present} {unit}(s), fewer than the "
                f"{ranking.min_units} required"
            )
        more = "more" if item.effect > 0 else "less"
        direction = f"{item.consistency} of {item.n_units} {unit}(s) in that direction"
        if ranking.design == compare.TREND:
            column = record.get("design_column") or "the column"
            return f"{more} with {column} (mean ρ {item.effect:+.2f}; {direction})"
        a, b = item.pair or ranking.levels or ("a", "b")
        text = f"{more} under {b} than {a}, {math.exp(abs(item.effect)):.1f}× ({direction})"
        # A profile's pair is the item's finding, so the sentence names it first (§24).
        if ranking.design == compare.PROFILE:
            return f"differs most between {a} and {b}: {text}"
        return text
    a, b = item.pair
    if item.unbounded:
        text = "appeared" if (item.a.mean or 0) == 0 else "vanished"
        return (
            f"differs most between {a} and {b}: {text} in {b}"
            if ranking.profile
            else text
        )
    if item.magnitude is None:
        return "—"
    which = b if (item.effect or 0) > 0 else a
    text = f"{math.exp(item.magnitude):.1f}× more in {which} ({unit} means)"
    if ranking.profile:
        return f"differs most between {a} and {b}: {text}"
    return text


def _floor_cell(ranking, item) -> str:
    return "yes" if ranking.clears_floor(item) else "no"


def _own_p_cell(item) -> str:
    """The item's own p with the false-discovery rate over the family after it:
    ``0.002 · q 0.01``."""
    own = p_text(getattr(item, "own_p", None))
    q = p_text(getattr(item, "q", None))
    if not own:
        return "—"
    return f"{own} · q {q}" if q else own


#: The ranking tables' header for :func:`_own_p_cell`.
OWN_P_HEADER = "p (own · q)"


def _verdict_headers(ranking) -> list[str]:
    """The two columns that say whether an item is a finding and by what p (§24 *The
    verdict*), named for the rule the ranking uses."""
    if ranking.verdict == compare.VERDICT_OWN:
        return [f"p ≤ {compare.FLOOR_ALPHA:g}", OWN_P_HEADER]
    return ["clears floor", P_HEADER]


def _verdict_cells(ranking, item) -> list[str]:
    if ranking.verdict == compare.VERDICT_OWN:
        return [_floor_cell(ranking, item), _own_p_cell(item)]
    return [_floor_cell(ranking, item), _p_cell(item)]


def _t_cell(item) -> str:
    """The moderated t as a cell: ``+8.8``, ``—`` for none."""
    t = getattr(item, "statistic", None)
    return "—" if t is None else f"{t:+.1f}"


def _studentized(ranking) -> bool:
    return ranking.statistic == compare.STATISTIC_T


def p_text(p: float | None) -> str:
    """A p or q as a person reads it: ``<0.001``, ``0.0039``, ``0.053``, ``0.5``; ``""`` for
    none. Two significant figures: enough to tell ``0.053`` from the slider's ``0.05``, and
    no more than a null of a few hundred draws can support."""
    if p is None:
        return ""
    return "<0.001" if p < 0.001 else f"{p:.2g}"


def _p_cell(item) -> str:
    """The family-wise p with the item's own p after it: ``0.05 · own <0.01``. The first is
    the floor read as a p; the second says whether the units agree on this item by itself.
    """
    family = p_text(getattr(item, "family_p", None))
    own = p_text(getattr(item, "own_p", None))
    if not family and not own:
        return "—"
    return f"{family} · own {own}" if own else family


#: The ranking tables' header for :func:`_p_cell`.
P_HEADER = "p (family · own)"


def _kept_cell(ranking, item) -> str:
    """Over a hierarchical family: whether a clearing item is a finding or a slice of one."""
    if item.measure_key in set(ranking.kept):
        return "yes"
    return "nested" if ranking.clears_floor(item) else ""


def ranking_rows(
    ranking, record: dict, items=None
) -> tuple[list[str], list[list[str]]]:
    """The ranking as a table — headers and one row of cells per item — the shape the detail
    page's Overview, ``summary.md`` and ``ranking.csv`` all render from, so the three cannot
    disagree about what a column means (one tested mapping per surface, CLAUDE.md)."""
    items = list(ranking.contrasts) if items is None else list(items)
    with_floor = ranking.floor is not None
    if ranking.profile:
        # Each row is between its own pair, so the pair is a column and the value columns
        # cannot carry a group's name (§24 *Designs*).
        within = ranking.design != compare.BETWEEN
        headers = ["measure", "between", "a", "b", "change (a \u2192 b)"]
        headers += ["t"] if _studentized(ranking) else []
        headers += ["in that direction", "present"] if within else []
        headers += _verdict_headers(ranking) if with_floor else []
        headers += ["kept"] if ranking.kept is not None else []
        headers += ["suggestion"]
        rows = []
        for item in items:
            unit = measure_unit(item.measure_key)
            a_text, shown_unit = shown_value(
                None if item.a is None else item.a.mean, unit
            )
            b_text, _ = shown_value(None if item.b is None else item.b.mean, unit)
            suffix = f" {shown_unit}" if shown_unit else ""
            pair = item.pair
            cells = [
                measure_title(item.measure_key, remainders_of(record)),
                " \u2192 ".join(pair) if pair else "\u2014",
                f"{a_text}{suffix}" if a_text else "\u2014",
                f"{b_text}{suffix}" if b_text else "\u2014",
                _change_text(item),
            ]
            cells += [_t_cell(item)] if _studentized(ranking) else []
            if within:
                cells += [
                    f"{item.consistency} of {item.n_units}",
                    f"{item.present_a} / {item.present_b}",
                ]
            cells += _verdict_cells(ranking, item) if with_floor else []
            cells += [_kept_cell(ranking, item)] if ranking.kept is not None else []
            cells += [suggestion(item, ranking, record)]
            rows.append(cells)
        return headers, rows
    if ranking.design == compare.TREND:
        headers = ["measure", "mean ρ"]
        headers += ["t"] if _studentized(ranking) else []
        headers += ["in that direction", "present"]
        headers += _verdict_headers(ranking) if with_floor else []
        headers += ["kept"] if ranking.kept is not None else []
        headers += ["suggestion"]
        rows = []
        for item in items:
            cells = [
                measure_title(item.measure_key, remainders_of(record)),
                "—" if item.effect is None else f"{item.effect:+.2f}",
            ]
            cells += [_t_cell(item)] if _studentized(ranking) else []
            cells += [
                f"{item.consistency} of {item.n_units}",
                str(item.n_present),
            ]
            cells += _verdict_cells(ranking, item) if with_floor else []
            cells += [_kept_cell(ranking, item)] if ranking.kept is not None else []
            cells += [suggestion(item, ranking, record)]
            rows.append(cells)
        return headers, rows
    names = group_headers(ranking)
    headers = ["measure", *names, f"change ({names[0]} \u2192 {names[1]})"]
    headers += ["t"] if _studentized(ranking) else []
    if ranking.design == compare.PAIRED:
        headers += ["in that direction", "present"]
    headers += _verdict_headers(ranking) if with_floor else []
    headers += ["kept"] if ranking.kept is not None else []
    headers += ["suggestion"]
    rows = []
    for item in items:
        unit = measure_unit(item.measure_key)
        a_text, shown_unit = shown_value(None if item.a is None else item.a.mean, unit)
        b_text, _ = shown_value(None if item.b is None else item.b.mean, unit)
        suffix = f" {shown_unit}" if shown_unit else ""
        cells = [
            measure_title(item.measure_key, remainders_of(record)),
            f"{a_text}{suffix}" if a_text else "—",
            f"{b_text}{suffix}" if b_text else "—",
            _change_text(item),
        ]
        cells += [_t_cell(item)] if _studentized(ranking) else []
        if ranking.design == compare.PAIRED:
            cells += [
                f"{item.consistency} of {item.n_units}",
                f"{item.present_a} / {item.present_b}",
            ]
        cells += _verdict_cells(ranking, item) if with_floor else []
        cells += [_kept_cell(ranking, item)] if ranking.kept is not None else []
        cells += [suggestion(item, ranking, record)]
        rows.append(cells)
    return headers, rows


def _number_or_blank(value: float | None):
    return "" if value is None else value


def ranking_csv_rows(ranking, record: dict) -> list[dict]:
    """Every ranked item with its verdict — ``ranking.csv``, complete where the tables and
    figures over a large family are the shown set."""
    kept = set(ranking.kept) if ranking.kept is not None else None
    # A parent label's row counts only its leftover (#24): `label` stays the join key, and
    # the two columns say so, as stats.csv's do.
    from detail.analysis.measures import display_label

    remainders = remainders_of(record)
    out = []
    for rank, item in enumerate(ranking.contrasts, start=1):
        keyed = dict(
            zip(("measure", "label", "label_to", "window_id"), item.measure_key)
        )
        effect = getattr(item, "effect", None)
        is_effect = isinstance(item, compare.Effect)
        out.append(
            {
                "rank": rank,
                "measure": keyed.get("measure", ""),
                "label": keyed.get("label", ""),
                "label_to": keyed.get("label_to", ""),
                "label_display": display_label(keyed.get("label", ""), remainders),
                "label_is_remainder": bool(keyed.get("label"))
                and keyed.get("label") in remainders,
                "window_id": keyed.get("window_id", ""),
                "effect": "" if effect is None else effect,
                "magnitude": "" if item.magnitude is None else item.magnitude,
                # The moderated t where the ranking scores by it (§24 *The statistic*): the
                # floor is in its unit, and `clears` is `|statistic|` against `floor`.
                "statistic": _number_or_blank(item.statistic),
                # No floor where each measure is read by its own test: the family's
                # floor is not what `clears` was decided against, and printing it says it was.
                "floor": (
                    ""
                    if ranking.floor is None or ranking.verdict == compare.VERDICT_OWN
                    else ranking.floor
                ),
                "clears": ranking.clears_floor(item),
                "kept": (
                    ranking.clears_floor(item)
                    if kept is None
                    else item.measure_key in kept
                ),
                # The floor read as a p, the item's own p, and the BH q over the family's own
                # p's (§24): what a reader thresholds at a strictness of their own.
                "family_p": _number_or_blank(item.family_p),
                "own_p": _number_or_blank(item.own_p),
                "q": _number_or_blank(item.q),
                "n_units": item.n_units if is_effect else ranking.n_units,
                "n_present": item.n_present if is_effect else "",
                "consistency": item.consistency if is_effect else "",
                "loo_sign_stable": item.loo_sign_stable if is_effect else "",
                "present_a": (
                    "" if not is_effect or item.present_a is None else item.present_a
                ),
                "present_b": (
                    "" if not is_effect or item.present_b is None else item.present_b
                ),
                "a_mean": "" if item.a is None or item.a.mean is None else item.a.mean,
                "b_mean": "" if item.b is None or item.b.mean is None else item.b.mean,
                # The two levels (or groups) the effect is between — the item's own pair over
                # a profile, the design's over a paired or two-group between (§24 *Designs*).
                "pair_a": "" if item.pair is None else item.pair[0],
                "pair_b": "" if item.pair is None else item.pair[1],
                "suggestion": suggestion(item, ranking, record),
            }
        )
    return out


def shown_items(ranking, ancestors) -> list:
    """What a large family's tables and figures cover: the kept set and the first
    ``SHOWN_BELOW_FLOOR`` below the floor, nesting suppressed (§24 *The node source*).
    """
    return ranking.kept_items() + compare.below_floor(
        ranking, ancestors, SHOWN_BELOW_FLOOR
    )


def _SILENT(_stage: str, _fraction: float, _detail: str) -> None:
    """The default progress sink: an export driven from a script reports to nobody."""


def _by_measure(summaries) -> dict[tuple, list]:
    out: dict[tuple, list] = defaultdict(list)
    for summary in summaries:
        out[summary.measure_key].append(summary)
    return out


def _measure_stem(measure_key: tuple) -> str:
    """A filename stem naming the measurement, so a file copied out of its folder still says
    what it is: ``rate__rearing`` rather than ``figure_3``.

    The behavior name in the middle is the user's own text, so it is sanitized: a `/` in it made
    this a path into a directory that does not exist and aborted the export, and a `.` was cut off
    by the `.with_suffix` that follows, merging two measurements into one file.
    """
    parts = [p for p in measure_key if p]
    return "__".join(safe_stem(p.replace(" ", "-")) for p in parts) or "measure"


def _write_tables(out_dir: Path, by_measure: dict) -> list[Path]:
    written: list[Path] = []
    for measure_key, summaries in sorted(by_measure.items()):
        rows = [
            {
                "group": "|".join(s.group),
                "n": s.n,
                "coverage": s.coverage,
                "mean": "" if s.mean is None else round(s.mean, 6),
                "median": "" if s.median is None else round(s.median, 6),
                # None at n=1 rather than 0.0, which reads as "no variation observed" when the
                # truth is "variation not observable" (§24).
                "sd": "" if s.sd is None else round(s.sd, 6),
            }
            for s in sorted(summaries, key=lambda s: s.group)
        ]
        stem = out_dir / _measure_stem(measure_key)
        tablib.write_tidy_csv(stem.with_suffix(".csv"), rows, TABLE_COLUMNS)
        written.append(stem.with_suffix(".csv"))
    return written


def observation_note(record: dict) -> str:
    """What one point is, in one line — the same sentence every output carries."""
    path = record.get("aggregate_to") or []
    return "one observation is: " + ("|".join(path) if path else "one per recording")


def _caption(ax, note: str, style: FigureStyle) -> None:
    """Print what one point is, under the axes.

    §24: *"The path is printed on every output — every table, every figure caption, `summary.md`,
    and `stats.csv` as a column. **A plot that does not say what one point is cannot be read.**"*
    It reached `stats.csv` and `summary.md` and no figure — so every chart this tool has ever
    exported was missing the one line that says whether a dot is a recording or an animal, which
    is the difference between two answers to the same question.
    """
    from detail.report import figures as figlib

    # Above the axes, not below. A time course has rotated window labels along the bottom and
    # the note landed on top of them; `show_title` is off for publication, so the top strip is
    # free. In the axes' own coordinates rather than the figure's, so constrained layout makes
    # room for it instead of the tight crop being the only thing that keeps it on the page.
    figlib.caption(ax, note, style=style)


#: The measure key without its window — what one figure is about.
def _fans_out(measure_key: tuple) -> bool:
    """Whether this measure answers with one value **per ordered pair of behaviors**.

    Such a measure is a matrix, not a series, and it is not drawn. `transitions` emits one row
    per ordered pair, so a vocabulary of N behaviors produces up to N x N of them — 400 for a
    20-syllable clusterer, each a two-point dot plot of one cell. That is precisely the blow-up
    collapsing the window out of a figure's identity just cured, arriving from the other
    direction, and a per-cell dot plot is not the figure anyone wants for a transition matrix.

    Nothing is hidden: the pairs are in `stats.csv`, in the per-measure tables, in the contrast
    grid and in the Measures tab, all one row per pair. What a transition matrix should *look*
    like is a §24 design question that has not been answered, and drawing 400 wrong answers is
    not a way of answering it.
    """
    from detail.analysis.measures import spec_of

    spec = spec_of(measure_key[0] if measure_key else "")
    return bool(spec is not None and spec.fans_out)


def _measurement(measure_key: tuple) -> tuple:
    return tuple(measure_key[:3])


def _scale_for(measure_key: tuple) -> float:
    """Nanoseconds are plotted as seconds and fractions as percent. An axis reading 0-5 with
    "(s)" on it is the same information as one reading 0-5e9 with nothing, minus the arithmetic.
    """
    return {"ns": 1e9, "fraction": 0.01}.get(measure_unit(measure_key), 1.0)


def _write_grid_tables(out_dir: Path, record: dict, by_measure: dict) -> list[Path]:
    """The contrast as one grid — measures down, groups across — in LaTeX, Markdown and CSV.

    §24 lists `tables.py`'s "LaTeX/Markdown/CSV" among its outputs and §24's exporter only ever
    called `write_tidy_csv`, so the three renderers shipped, were used by both (since removed)
    benchmark exporters, and were unreachable from a comparison. A user asked for "tables as latex";
    it turned out to be wiring rather than building.

    Values are rendered in their unit (nanoseconds as seconds), because a `\num{4956702127.7}`
    in a manuscript is not what anybody meant.
    """
    rows = []
    for measure_key, summaries in sorted(by_measure.items()):
        unit = measure_unit(measure_key)
        scale = _scale_for(measure_key)
        shown_unit = shown_value(0.0, unit)[1]
        name = measure_title(measure_key, remainders_of(record))
        rows += [
            {
                "measure": f"{name} ({shown_unit})" if shown_unit else name,
                "group": "|".join(summary.group),
                "value": None if summary.mean is None else summary.mean / scale,
                "sd": None if summary.sd is None else summary.sd / scale,
            }
            for summary in sorted(summaries, key=lambda s: s.group)
        ]
    if not rows:
        return []

    path = record.get("aggregate_to") or []
    grid = tablib.pivot_to_grid(
        [r for r in rows if r["value"] is not None],
        row_key="measure",
        col_key="group",
        value_key="value",
        std_key="sd",
        row_header="measure",
        # §24: the path reaches every output. A table without it cannot be read either.
        caption=(
            f"Comparison {record.get('id', '')} across "
            f"{', '.join(record.get('group_by') or [])}. One observation is "
            f"{'|'.join(path) if path else 'one per recording'}."
        ),
        label=f"tab:{record.get('id', 'comparison')}",
        # Nothing here is "better"; bolding a maximum would assert a direction §24 declines.
        bold_best=False,
    )
    written = []
    for suffix, render in (
        (".tex", tablib.to_latex),
        (".md", tablib.to_markdown),
    ):
        target = out_dir / f"comparison{suffix}"
        target.write_text(render(grid), encoding="utf-8")
        written.append(target)
    csv_path = out_dir / "comparison.csv"
    tablib.write_grid_csv(csv_path, grid)
    written.append(csv_path)
    return written


def p_symbol(ranking) -> str:
    """Which P a figure writes (detail#76): ``P_FWE`` where a measure had to clear the
    family's floor, having been found by screening many; ``P`` where it was tested on its
    own."""
    return "P" if ranking.verdict == compare.VERDICT_OWN else "P_FWE"


def item_p(ranking, item) -> tuple[float | None, float | None]:
    """``(p, bound)``: the P the ranking judged ``item`` by, and the bound it is reported as
    where a shuffled test could not resolve lower (``1 / draws``), else ``None``.

    A shuffled family-wise P of zero means no draw reached the observed statistic, which is
    not a P of zero: it is below the test's resolution, and is written ``< 1/draws``.
    """
    p = item.own_p if ranking.verdict == compare.VERDICT_OWN else item.family_p
    if p is None:
        return None, None
    if p <= 0.0 and not ranking.exact:
        return p, 1.0 / max(int(ranking.draws), 1)
    return p, None


def p_label(p: float | None, symbol: str = "P", *, bound: float | None = None) -> str:
    """``P = 0.004``, ``P < 0.001``, ``P_FWE < 0.005``: a P as a figure writes it.

    Three decimals below 0.01 and two above, ``< 0.001`` below that, and the bound form where
    a shuffled test could not resolve lower. Two decimals would print a significant 0.045 as
    the threshold itself, so that case takes three. ``symbol`` is ``P`` or ``P_FWE``; the P is
    set in italic math text, which the SVG rewrite keeps as one text object.
    """
    head, _, sub = symbol.partition("_")
    name = rf"$\mathit{{{head}}}$"
    if sub:
        name = rf"$\mathit{{{head}}}_{{\mathrm{{{sub}}}}}$"
    if bound is not None:
        return f"{name} < {bound:.2g}"
    if p is None:
        return ""
    if p < 0.001:
        return f"{name} < 0.001"
    crowded = p < compare.FLOOR_ALPHA and f"{p:.2f}" == f"{compare.FLOOR_ALPHA:.2f}"
    return f"{name} = {p:.3f}" if p < 0.01 or crowded else f"{name} = {p:.2f}"


def group_columns(groups: Collection[tuple]) -> tuple[list[int], list[int]]:
    """The positions of a group tuple that vary across ``groups``, split into the ordered
    ones (every value a number: a dose, a day) and the named ones (a sex, a strain). A
    position every group shares says nothing about a group and is in neither."""
    width = max((len(g) for g in groups), default=0)
    ordered: list[int] = []
    named: list[int] = []
    for i in range(width):
        values = {g[i] for g in groups if i < len(g)}
        if len(values) < 2:
            continue
        (ordered if all(compare.is_number(v) for v in values) else named).append(i)
    return ordered, named


def _group_sort_key(group: tuple, ordered: list[int], named: list[int]) -> tuple:
    """Ordered columns first, then named, then every part: in a sex × dose design the two
    sexes at one dose sit side by side (user, 2026-11-20), as they would on paper."""
    return (
        tuple(compare.level_sort_key(group[i]) for i in ordered),
        tuple(compare.level_sort_key(group[i]) for i in named),
        tuple(compare.level_sort_key(part) for part in group),
    )


def group_order(record: dict, ranking, groups: Collection[tuple]) -> list[tuple]:
    """The groups in the order a figure draws them (detail#76): the levels a user named,
    reference first; else the order a within-unit design resolved (numeric levels in numeric
    order); else by the ordered columns first (numeric, in numeric order), then the named
    ones, so the groups of a sex × dose design interleave by dose. Never plain string order,
    which drew doses 10, 25, 5 and every F before every M. A between design's level list is
    that string order, so it is not taken as the design's."""
    joined = {"|".join(g): g for g in groups}
    preferred = list(record.get("levels") or ())
    if not preferred and ranking is not None and ranking.design != compare.BETWEEN:
        preferred = list(ranking.level_values)
    first = [joined[name] for name in preferred if name in joined]
    ordered, named = group_columns(groups)
    rest = sorted(
        (g for g in groups if g not in first),
        key=lambda g: _group_sort_key(g, ordered, named),
    )
    return first + rest


def _all_groups(record: dict, ranking, summaries) -> list[tuple]:
    """Every group the comparison holds, in drawing order, so a colour is fixed by what a
    group is and a figure over a subset of them repaints nothing."""
    groups = {s.group for s in summaries}
    for name in record.get("groups") or ():
        groups.add(tuple(str(name).split("|")))
    return group_order(record, ranking, groups)


def group_colors(record: dict, ranking, groups: list[tuple], style) -> dict[tuple, str]:
    """``{group: colour}`` by what the group is, not where it sits (detail#76).

    The reference level (the one an effect is read against: the first) is grey; the levels of
    an ordered variable take one hue from light to dark, checked monotone; two levels are grey
    and the hue; three or more named values take distinct hues. In a design of a named column
    and an ordered one (sex × dose) each named value takes one hue and the ordered value sets
    its lightness, the same ramp a dose alone takes, so every dose of one sex is one family
    of colour (user, 2026-11-20); there is no grey, since two groups at the reference dose
    would otherwise be told apart by nothing. A missing group leaves the others' colours where
    they were, because the map is over every group of the record.
    """
    from detail.report import figures as figlib

    chart = style.chart
    grey = chart.text_muted
    hues = theme_hues(style)
    hue = hues[0]
    lighten = style.theme == "dark"
    names = ["|".join(g) for g in groups]
    ordered = (ranking is not None and ranking.design == compare.TREND) or (
        len(groups) > 2 and all(compare.is_number(n) for n in names)
    )
    if not groups:
        return {}
    ordered_cols, named_cols = group_columns(groups)
    if ordered_cols and named_cols:
        named_values = sorted(
            {tuple(g[i] for i in named_cols) for g in groups},
            key=lambda v: tuple(compare.level_sort_key(part) for part in v),
        )
        ordered_values = sorted(
            {tuple(g[i] for i in ordered_cols) for g in groups},
            key=lambda v: tuple(compare.level_sort_key(part) for part in v),
        )
        ramps = {
            value: figlib.one_hue_ramp(
                hues[k % len(hues)], len(ordered_values), lighten=lighten
            )
            for k, value in enumerate(named_values)
        }
        return {
            g: ramps[tuple(g[i] for i in named_cols)][
                ordered_values.index(tuple(g[i] for i in ordered_cols))
            ]
            for g in groups
        }
    if ordered:
        colors = [grey, *figlib.one_hue_ramp(hue, len(groups) - 1, lighten=lighten)]
    elif len(groups) <= 2:
        colors = [grey, hue][: len(groups)]
    else:
        colors = [hues[i % len(hues)] for i in range(len(groups))]
    return dict(zip(groups, colors))


def theme_hues(style) -> list[str]:
    """The identity hues a comparison figure takes, in order, for the style's surface: the
    palette from blue on white, and from orange on the dark surface, where orange reads
    better than blue and the rest follow it round (user, 2026-11-20). The first is the one
    hue an ordered variable's ramp and a two-level contrast take."""
    chart = style.chart
    order = list(chart.categorical)
    if style.theme == "dark":
        start = order.index(chart.hue("orange"))
        order = order[start:] + order[:start]
    return order


def _units_by_group(obs, measurement: tuple, groups: list[tuple], scale: float):
    """Per group, ``[(unit, value)]`` for one single-window measurement, in shown units."""
    out: dict[tuple, list[tuple[str, float]]] = {g: [] for g in groups}
    for row in obs:
        if _measurement(row.measure_key) != measurement or row.group not in out:
            continue
        out[row.group].append(("|".join(row.key), row.value / scale))
    return [sorted(out[g]) for g in groups]


def _contrast_item(ranking, measure_key: tuple):
    """The ranked item for a full measure key, or ``None``."""
    if ranking is None:
        return None
    for item in ranking.contrasts:
        if item.measure_key == measure_key:
            return item
    return None


def _bracket_text(ranking, item, record: dict) -> str:
    """The P as the plot writes it; a test against an ordered variable names the variable,
    never "trend", which a reader takes to mean "not quite significant"."""
    p, bound = item_p(ranking, item)
    label = p_label(p, p_symbol(ranking), bound=bound)
    if ranking.design == compare.TREND:
        return f"{record.get('design_column') or 'trend'}: {label}"
    return label


def _bracket_span(ranking, item, drawn: list[tuple]) -> tuple[int, int] | None:
    """Which drawn groups the item's P brackets: every level of a trend, else its pair."""
    names = ["|".join(g) for g in drawn]
    if ranking.design == compare.TREND:
        return (0, len(drawn) - 1) if len(drawn) > 1 else None
    pair = item.pair if getattr(item, "pair", None) else ranking.levels
    if pair is None or pair[0] not in names or pair[1] not in names:
        return None
    a, b = names.index(pair[0]), names.index(pair[1])
    return (min(a, b), max(a, b))


def group_axis_label(record: dict) -> str:
    """What a group tick is made of, as the x axis says it (user, 2026-11-20): the grouping
    columns joined as the ticks are (*Sex | fentanyl_dose* over *F|0*), one column's name
    alone when there is one."""
    return " | ".join(str(c) for c in record.get("group_by") or ())


def draw_measurement(
    ax, measurement: tuple, obs, record: dict, ranking, *, style, colors
) -> bool:
    """One measurement into ``ax`` (detail#76): bars of the mean with s.e.m., every unit a
    dot, the same unit joined across groups, and its P bracketed when the ranking judged it
    to differ. Returns whether a bracket was drawn.

    ``colors`` is the record-wide ``{group: colour}``; the groups drawn are those with an
    observation, in :func:`group_order`. A ratio's bars start at 1, and the baseline is a
    labelled tick.
    """
    from detail.report import figures as figlib

    unit = measure_unit(measurement)
    scale = _scale_for(measurement)
    baseline = 1.0 if unit == RATIO_UNIT else 0.0
    present = {row.group for row in obs if _measurement(row.measure_key) == measurement}
    drawn = group_order(record, ranking, present)
    observations = _units_by_group(obs, measurement, drawn, scale)
    figlib.bars_with_units(
        ax,
        ["|".join(g) for g in drawn],
        observations,
        colors=[colors.get(g, style.chart.neutral) for g in drawn],
        style=style,
        ylabel=panel_label(measurement, remainders_of(record)),
        baseline=baseline,
    )
    ax.set_xlabel(group_axis_label(record))
    top = max((v for group in observations for _, v in group), default=baseline)
    window = next(
        (
            row.measure_key[3]
            for row in obs
            if _measurement(row.measure_key) == measurement and len(row.measure_key) > 3
        ),
        "whole",
    )
    item = _contrast_item(ranking, (*measurement, window))
    span = None
    if item is not None and ranking.clears_floor(item):
        span = _bracket_span(ranking, item, drawn)
    y_bar, y_text = figlib.fit_baseline_axis(
        ax, baseline, top, annotated=span is not None
    )
    if span is not None:
        figlib.bracket(
            ax,
            span[0],
            span[1],
            _bracket_text(ranking, item, record),
            y_bar=y_bar,
            y_text=y_text,
            style=style,
        )
    return span is not None


def _measurements(by_measure: dict) -> list[tuple[tuple, dict[str, list]]]:
    """Each drawn measurement with its summaries per window, in registry order."""
    grouped: dict[tuple, dict[str, list]] = {}
    for measure_key, summaries in by_measure.items():
        if _fans_out(measure_key):
            continue
        window = measure_key[3] if len(measure_key) > 3 else ""
        grouped.setdefault(_measurement(measure_key), {})[window] = summaries
    return sorted(grouped.items(), key=lambda kv: measurement_order(kv[0]))


def measurement_order(measurement: tuple):
    """Sort key: registry order for the measure, then the label; the same row sits in the
    same place in every comparison."""
    from detail.analysis.measures import MEASURES

    ids = list(MEASURES)
    name = measurement[0] if measurement else ""
    index = ids.index(name) if name in ids else len(ids)
    return (index, tuple(str(p) for p in measurement[1:]))


def _single_window(per_window: dict) -> bool:
    return len(per_window) == 1


def _measure_figures(
    out_dir: Path,
    by_measure: dict,
    obs,
    record: dict,
    ranking,
    note: str,
    formats,
    style,
    report=_SILENT,
) -> list[Path]:
    """One figure per *measurement* — bars with every unit across groups, or a curve across
    windows.

    **The window is not part of a figure's identity.** It used to be: `_by_measure` bucketed on
    the whole `measure_key`, which ends in `window_id`, so a sliding-window comparison wrote one
    two-point dot plot per window — hundreds of figures for a sliding window over one
    recording. Collapsing the window out removes the blow-up and produces the missing curve in
    the same change, because a measurement over many windows *is* the curve.

    **One figure, not two** (detail#76). The dot plot and the paired-lines figure it used to
    draw beside it are one panel now: bars of the mean with s.e.m., every unit a dot, and the
    same unit joined across the groups it appears in, with its P on the plot when the ranking
    judged it to differ.
    """
    from detail.report import figures as figlib

    written: list[Path] = []
    ordered_measurements = _measurements(by_measure)
    colors = group_colors(
        record, ranking, _all_groups(record, ranking, obs_summaries(obs)), style
    )
    for index, (measurement, per_window) in enumerate(ordered_measurements):
        report("figures", index / len(ordered_measurements), _measure_stem(measurement))
        windows = sorted(per_window, key=windows_lib.window_sort_key)
        scale = _scale_for(measurement)
        with figlib.styled(style):
            import matplotlib.pyplot as plt

            if _single_window(per_window):
                n_groups = len({s.group for s in per_window[windows[0]]})
                fig, ax = plt.subplots(
                    figsize=(min(3.5, max(1.9, 1.0 + 0.42 * n_groups)), 2.3)
                )
                draw_measurement(
                    ax, measurement, obs, record, ranking, style=style, colors=colors
                )
            else:
                fig, ax = plt.subplots()
                groups = group_order(
                    record,
                    ranking,
                    {s.group for windowed in per_window.values() for s in windowed},
                )
                series = []
                for group in groups:
                    values = []
                    for window in windows:
                        match = [s for s in per_window[window] if s.group == group]
                        centre = match[0].mean if match else None
                        values.append(None if centre is None else centre / scale)
                    series.append(("|".join(group), values))
                figlib.group_curves(
                    ax,
                    windows,
                    series,
                    style=style,
                    xlabel="window",
                    ylabel=panel_label(measurement, remainders_of(record)),
                )
                for line, group in zip(ax.lines, groups):
                    line.set_color(colors.get(group, line.get_color()))
                if len(series) > 1:
                    figlib.place_legend(ax, style=style)
            _caption(ax, note, style)
            written += figlib.save_figure(
                fig, out_dir / _measure_stem(measurement), formats, editable_svg=True
            )
    return written


def obs_summaries(obs):
    """Group summaries over ``obs``, for the group list alone."""
    return compare.summarize(obs)


def panel_measurements(by_measure: dict, record: dict, ranking) -> list[tuple]:
    """The measurements the composite figure holds (detail#76): over the node source the
    clusters the screen kept, in rank order; otherwise every single-window measurement, in
    registry order; at most :data:`PANELS_MAX`."""
    single = [
        m for m, per_window in _measurements(by_measure) if _single_window(per_window)
    ]
    if record.get("source") == "nodes" and ranking is not None:
        kept = [item.measure_key[1] for item in ranking.kept_items()]
        single = sorted(
            (m for m in single if m[1] in kept),
            key=lambda m: (kept.index(m[1]), measurement_order(m)),
        )
    return single[:PANELS_MAX]


def _panels_figure(
    out_dir: Path,
    by_measure: dict,
    obs,
    record: dict,
    ranking,
    note: str,
    formats,
    style,
) -> list[Path]:
    """The chosen measurements as one figure (detail#76): a row or grid of panels at a double
    column's width, one legend, panel letters. The common case, three locomotion measures or
    the clusters that cleared, is one export rather than files to assemble by hand."""
    from matplotlib.patches import Patch

    from detail.report import figures as figlib

    panels = panel_measurements(by_measure, record, ranking)
    if len(panels) < 2:
        return []
    groups = _all_groups(record, ranking, obs_summaries(obs))
    colors = group_colors(record, ranking, groups, style)
    n = len(panels)
    cols = n if n <= 4 else (3 if n in (5, 6, 9) else 4)
    rows = -(-n // cols)
    with figlib.styled(style):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(style.double_col_width_in, 1.75 * rows + 0.45),
            squeeze=False,
        )
        flat = [ax for row in axes for ax in row]
        for ax in flat[n:]:
            ax.set_visible(False)
        for ax, measurement in zip(flat, panels):
            draw_measurement(
                ax, measurement, obs, record, ranking, style=style, colors=colors
            )
        handles = [
            Patch(
                facecolor=figlib.tint(colors[g], towards=style.chart.surface),
                edgecolor=colors[g],
                linewidth=0.7,
                label="|".join(g),
            )
            for g in groups
        ]
        fig.legend(
            handles=handles,
            loc="outside upper right",
            ncol=len(handles),
            frameon=False,
            title=group_axis_label(record),
            title_fontsize=style.type.tick,
            fontsize=style.type.tick,
            handlelength=1.0,
            handleheight=0.8,
            columnspacing=0.9,
            handletextpad=0.4,
        )
        fig.suptitle(
            note,
            x=0.0,
            ha="left",
            fontsize=style.type.annotation,
            color=style.chart.text_muted,
            fontweight="normal",
        )
        # Last: the letters are placed from the laid-out panels, labels included.
        figlib.panel_letters_outside(fig, flat[:n], style=style)
        return figlib.save_figure(
            fig, out_dir / PANELS_STEM, formats, editable_svg=True
        )


def _ranking_figure(
    out_dir: Path, ranking, note: str, formats, style, remainders: Collection[str]
) -> list[Path]:
    """Draw the difference ranking. Takes a ``compare.Ranking``, which cannot exist without its
    floor — so "no ranking without a floor" is structural rather than remembered (§24).
    """
    from detail.report import figures as figlib

    if not ranking.contrasts:
        return []
    if ranking.verdict == compare.VERDICT_OWN:
        # Bars of size across measures in different units, against a floor none of them was
        # judged by, would be a chart of the wrong question. The table carries the verdicts.
        return []
    # Unbounded changes have no bar to draw — a fold change from zero is off every axis — so they
    # are listed in `summary.md` and left out of the chart rather than given an invented height.
    drawn = [c for c in ranking.contrasts if c.score is not None]
    if not drawn:
        return []
    # A family of a thousand nodes is not a bar chart: the bars are what cleared the floor
    # and the first few below it, as the screen shows them; never more than `_MAX_DRAWN_BARS`.
    clearing = [c for c in drawn if ranking.clears_floor(c)]
    drawn = (clearing + [c for c in drawn if c not in clearing][:SHOWN_BELOW_FLOOR])[
        :_MAX_DRAWN_BARS
    ]
    labels, xlabel = _ranking_labels(drawn, ranking, remainders)
    with figlib.styled(style):
        import matplotlib.pyplot as plt

        # Tall enough for its bars: at a fixed height forty labels overlapped into a smear.
        fig, ax = plt.subplots(
            figsize=(style.single_col_width_in, 0.9 + _BAR_ROW_IN * len(drawn))
        )
        # The bars are the score the floor is in the unit of: the moderated t on a
        # studentized ranking, the |effect| otherwise (\u00a724 *The statistic*).
        figlib.ranked_bars_with_floor(
            ax,
            labels,
            [c.score for c in drawn],
            ranking.floor,
            style=style,
            xlabel=xlabel,
            color=theme_hues(style)[0],
        )
        if ranking.design != compare.TREND and not _studentized(ranking):
            _label_fold_change(ax, drawn)
        # As the figure's title line rather than `_caption`'s strip above the axes, which
        # the floor's "chance" label also sits in: the two collided wherever the floor fell
        # under the note. Constrained layout keeps a suptitle clear of the axes' annotations.
        fig.suptitle(
            note,
            x=0.0,
            ha="left",
            fontsize=style.type.annotation,
            color=style.chart.text_muted,
            fontweight="normal",
        )
        return figlib.save_figure(
            fig, out_dir / "difference_ranking", formats, editable_svg=True
        )


#: Bars the ranking figure draws at most — the top of the ranking, which is what it is for.
_MAX_DRAWN_BARS = 40
#: The ranking figure's height per bar, in inches: a 7 pt label with air either side.
_BAR_ROW_IN = 0.16


def _ranking_labels(
    drawn: list, ranking, remainders: Collection[str]
) -> tuple[list[str], str]:
    """The bars' labels and the axis label. Over one measure (a node screen is always one)
    the measure's name is said once, on the axis, and each bar is its label alone: forty
    bars each starting "Share of time · " said nothing forty times."""
    xlabel = _ranking_xlabel(ranking)
    measures = {c.measure_key[0] for c in drawn}
    if len(measures) != 1:
        return [measure_title(c.measure_key, remainders) for c in drawn], xlabel
    from detail.analysis.measures import display_label, spec_of

    (measure,) = measures
    spec = spec_of(measure)
    name = spec.label if spec is not None else measure
    labels = [
        "  ·  ".join(
            _readable_label(display_label(p, remainders) if i < 2 else p)
            for i, p in enumerate(c.measure_key[1:])
            if p and p != "whole"
        )
        or name
        for c in drawn
    ]
    return labels, f"{name}: {xlabel}"


def _ranking_xlabel(ranking) -> str:
    """The ranking figure's axis: what the bars are, in the unit the floor is in."""
    if _studentized(ranking):
        return "|moderated t| within unit (mean ÷ ((sd + s0) / √n))"
    if ranking.design == compare.TREND:
        return "|mean Spearman ρ| within unit"
    if ranking.design == compare.PAIRED:
        return "|fold change| within unit (log scale)"
    if ranking.design == compare.PROFILE:
        return "|fold change| of the most different pair of levels, within unit (log scale)"
    if ranking.profile:
        return "|fold change| of the most different pair of groups (log scale)"
    return "|fold change| between groups (log scale)"


def _label_fold_change(ax, drawn) -> None:
    """Relabel the log-ratio axis as fold change, which is what a reader thinks in.

    The bars are ``|log ratio|`` because that is the scale on which a doubling and a halving are
    the same size. Nobody reads log ratios, so the ticks say 1x / 2x / 4x.
    """
    import math

    top = max((c.magnitude for c in drawn), default=0.0)
    ticks, k = [0.0], 1
    while ticks[-1] < top and k < 12:
        ticks.append(math.log(2**k))
        k += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{2**i:g}x" for i in range(len(ticks))])


def _direction(ranking) -> str:
    """`veh \u2192 fent` — which way the change column runs.

    The groups are sorted, not baseline-first: nothing in a comparison declares a reference, and
    inventing one would be a claim the user did not make. Naming the direction is the only
    reading that cannot be got backwards.
    """
    return " \u2192 ".join(tablib.md_cell(g) for g in group_headers(ranking))


def group_headers(ranking) -> list[str]:
    """The two group labels, as column headers — taken from the contrast the cells come from.

    Not from ``record["groups"]``. That list is joined-then-sorted, this is sorted-then-joined, and
    the two disagree whenever one group's first component is a prefix of another's — which put
    every mean under the other group's name on the GUI Overview. Reading the headers off the same
    contrast as the numbers makes them impossible to get out of step, and keeps records written
    before that ordering was fixed readable (tests/analysis/test_group_ordering.py).
    """
    first = ranking.contrasts[0]
    if first.a is not None and first.b is not None:
        return ["|".join(first.a.group), "|".join(first.b.group)]
    return list(ranking.levels or ("a", "b"))


def _change_text(item) -> str:
    """How the change reads: a percent, or what actually happened when there is no percent."""
    if item.unbounded:
        return "appeared" if (item.a.mean or 0) == 0 else "vanished"
    percent = item.percent_change
    if percent is None:
        return "—"
    return f"{percent:+.0f}%"


#: How many ranked items `summary.md` lists beyond the kept set over a large family.
_SUMMARY_ROWS = 20


def statistic_sentence(ranking) -> str:
    """What the ranking scores by, in one line every surface prints beside the floor (\u00a724
    *The statistic*): the moderated t with its ``s0``, or the effect. Empty for a between
    design, which has only the effect to rank on, and for a ranking read by each measure's
    own test, which is ordered by that test's p and says so in :func:`floor_sentence`.
    """
    if ranking.design == compare.BETWEEN or ranking.verdict == compare.VERDICT_OWN:
        return ""
    unit = "|".join(ranking.unit) or "unit"
    if ranking.statistic == compare.STATISTIC_T:
        s0 = "" if ranking.s0 is None else f", s0 = {ranking.s0:.3g}"
        return (
            f"ranked on the moderated t: each measure's mean per-{unit} effect over its "
            f"spread across {unit}s plus the family's median spread (t = mean \u00f7 "
            f"((sd + s0) / \u221an){s0})"
        )
    return f"ranked on the effect: each measure's mean per-{unit} effect"


def statistic_note(record: Mapping) -> str:
    """The record's resolved statistic as a short line (``ranked on the moderated t`` /
    ``ranked on the effect``) for the surfaces that print the record before ranking it: the
    detail page's header and ``comparison run``. Empty for a between design, and for a
    record written before the field existed, which ranks on the effect as it always did.
    """
    if record.get("design") in (None, compare.BETWEEN):
        return ""
    if record.get("verdict") == compare.VERDICT_OWN:
        return "ordered by each measure's own test, smallest p first"
    statistic = record.get("ranking_statistic")
    if statistic == compare.STATISTIC_T:
        return "ranked on the moderated t: the mean per-unit effect over its spread"
    if statistic == compare.STATISTIC_EFFECT:
        return "ranked on the effect: the mean per-unit effect"
    return ""


def floor_sentence(ranking) -> str:
    """How the floor was made, in one line every surface can print."""
    how = (
        f"every one of the {ranking.draws} sign-flip configurations"
        if ranking.exact
        else f"{ranking.draws} shuffled draws"
    )
    if ranking.verdict == compare.VERDICT_OWN:
        what = {
            compare.TREND: "its mean ρ was at least as far from zero, each unit's levels "
            "shuffled within it",
            compare.PAIRED: "its mean change was at least as large",
            compare.PROFILE: "its largest change between two levels was at least as large, "
            "each unit's levels shuffled within it",
        }.get(ranking.design, "its change between groups was at least as large")
        return (
            f"Each measure is tested on its own: its p is the share of {how} in which "
            f"{what}. No floor is shared between measures, which are in different units; "
            f"q is the false-discovery rate over the **{len(ranking.contrasts)}** tested."
        )
    if ranking.statistic == compare.STATISTIC_T:
        within = (
            ", each unit's levels shuffled within it"
            if ranking.design != compare.PAIRED
            else ""
        )
        pairs = (
            " and the largest difference between any two levels taken each time"
            if ranking.design == compare.PROFILE
            else ""
        )
        return (
            f"Floor (95th percentile of the largest |moderated t| over {how}{within}"
            f"{pairs}): **{ranking.floor:.3g}**."
        )
    if ranking.design == compare.TREND:
        return (
            f"Floor (95th percentile of the largest |mean \u03c1| over {how}, each unit's "
            f"levels shuffled within it): **{ranking.floor:.3g}**."
        )
    if ranking.design == compare.PAIRED:
        return (
            f"Floor (95th percentile of the largest |effect| over {how}): "
            f"**{ranking.floor:.4g}** as a fold change ({math.exp(ranking.floor):.3g}x)."
        )
    if ranking.design == compare.PROFILE:
        return (
            f"Floor (95th percentile of the largest |effect| over {how}, each unit's levels "
            f"shuffled within it and the largest difference between any two levels taken "
            f"each time): **{ranking.floor:.4g}** as a fold change "
            f"({math.exp(ranking.floor):.3g}x)."
        )
    pairs = (
        ", the largest difference between any two groups taken each time"
        if ranking.profile
        else ""
    )
    return (
        f"Floor (95th percentile of {how} of the group labels{pairs}): "
        f"**{ranking.floor:.4g}** as a fold change ({math.exp(ranking.floor):.3g}x). "
        f"Measures are ranked by |fold change|, which orders counts, durations and "
        f"fractions on one scale."
    )


def _summary_ranking_lines(record: dict, ranking) -> list[str]:
    lines = ["## Largest changes", ""]
    if ranking.floor is None:
        # Stated, not left blank. A missing floor and a floor nothing cleared look identical
        # on the page, and they are different facts.
        lines += [
            f"**No floor computed.** {ranking.no_floor_reason}. The changes below are "
            "reported as measured; this comparison does not say how large a change "
            "the design's own null alone would have produced.",
            "",
        ]
    else:
        lines += [floor_sentence(ranking)]
        if statistic_sentence(ranking):
            lines += [
                statistic_sentence(ranking)[0].upper()
                + statistic_sentence(ranking)[1:]
                + "."
            ]
        lines += [""]
    if ranking.design != compare.BETWEEN:
        lines += [
            f"An effect is undefined below {ranking.min_units} present "
            f"{_unit_word(record)}(s) of {ranking.n_units}.",
            "",
        ]
    kept = ranking.kept_items()
    if ranking.kept is not None:
        lines += [
            f"**{len(kept)} kept** after nesting suppression, of "
            f"{len(ranking.clearing())} clearing the floor.",
            "",
        ]
    shown = kept + [i for i in ranking.contrasts if i not in kept][:_SUMMARY_ROWS]
    headers, rows = ranking_rows(ranking, record, shown)
    lines.append("| " + " | ".join(tablib.md_cell(h) for h in headers) + " |")
    lines.append("|" + " --- |" * len(headers))
    for cells in rows:
        lines.append("| " + " | ".join(tablib.md_cell(c) for c in cells) + " |")
    if len(shown) < len(ranking.contrasts):
        lines.append("")
        lines.append(
            f"{len(ranking.contrasts) - len(shown)} more in `ranking.csv`, every candidate "
            f"with its effect, floor and verdict."
        )
    lines.append("")
    return lines


def _sem(values) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _measurement_cells(item, record: dict) -> dict:
    keyed = dict(zip(("measure", "label", "label_to", "window_id"), item.measure_key))
    return {
        "measure": keyed.get("measure", ""),
        "label": keyed.get("label", ""),
        "label_to": keyed.get("label_to", ""),
        "window_id": keyed.get("window_id", ""),
        "measurement": measure_title(item.measure_key, remainders_of(record)),
    }


def _contrast_text(ranking, item, record: dict) -> str:
    if ranking.design == compare.TREND:
        return f"trend over {record.get('design_column') or ''}".strip()
    pair = item.pair if getattr(item, "pair", None) else ranking.levels
    return f"{pair[1]} vs {pair[0]}" if pair else ""


def significant_rows(ranking, record: dict) -> list[dict]:
    """One row per result that cleared (detail#76): the group means and s.e.m., the effect,
    the statistic, the P it was judged by and which P that is, the others beside it, n, and
    how many units went the effect's way. ``ranking.csv`` holds every item; this is what a
    paper cites, beside the figures."""
    rows = []
    rank_of = {item.measure_key: i for i, item in enumerate(ranking.contrasts, start=1)}
    for item in ranking.kept_items():
        unit = measure_unit(item.measure_key)
        scale = _scale_for(item.measure_key)
        shown_unit = shown_value(0.0, unit)[1]
        a, b = item.a, item.b
        p, bound = item_p(ranking, item)
        is_effect = isinstance(item, compare.Effect)
        if ranking.design == compare.TREND and item.effect is not None:
            change = f"rho {item.effect:+.2f}"
        else:
            change = _change_text(item)
        rows.append(
            {
                "rank": rank_of[item.measure_key],
                **_measurement_cells(item, record),
                "contrast": _contrast_text(ranking, item, record),
                "pair_a": "" if item.pair is None else item.pair[0],
                "pair_b": "" if item.pair is None else item.pair[1],
                "a_mean": "" if a is None or a.mean is None else a.mean / scale,
                "a_sem": (
                    ""
                    if a is None or _sem(a.values) is None
                    else _sem(a.values) / scale
                ),
                "a_n": "" if a is None else a.n,
                "b_mean": "" if b is None or b.mean is None else b.mean / scale,
                "b_sem": (
                    ""
                    if b is None or _sem(b.values) is None
                    else _sem(b.values) / scale
                ),
                "b_n": "" if b is None else b.n,
                "unit": shown_unit,
                "effect": _number_or_blank(getattr(item, "effect", None)),
                "change": change,
                "statistic": _number_or_blank(item.statistic),
                "p": "" if p is None else p,
                "p_kind": (
                    "own" if ranking.verdict == compare.VERDICT_OWN else "family-wise"
                )
                + ("" if bound is None else f" (below 1/{ranking.draws})"),
                "family_p": _number_or_blank(item.family_p),
                "own_p": _number_or_blank(item.own_p),
                "q": _number_or_blank(item.q),
                "n_units": item.n_units if is_effect else ranking.n_units,
                "in_direction": item.consistency if is_effect else "",
            }
        )
    return rows


def units_rows(obs, record: dict, groups: list[tuple]) -> tuple[list[str], list[dict]]:
    """``(columns, rows)``: one row per unit and measurement with a column per group, in the
    unit a reader sees (detail#76). The long ``stats.csv`` reshaped, because this is the
    shape Prism and a spreadsheet take."""
    names = ["|".join(g) for g in groups]
    columns = UNITS_COLUMNS_FIXED + names
    cells: dict[tuple, dict] = {}
    for row in obs:
        key = (row.measure_key, row.key)
        if key not in cells:
            keyed = dict(
                zip(("measure", "label", "label_to", "window_id"), row.measure_key)
            )
            unit = measure_unit(row.measure_key)
            cells[key] = {
                "measure": keyed.get("measure", ""),
                "label": keyed.get("label", ""),
                "label_to": keyed.get("label_to", ""),
                "window_id": keyed.get("window_id", ""),
                "measurement": measure_title(row.measure_key, remainders_of(record)),
                "unit": shown_value(0.0, unit)[1],
                "observation": "|".join(row.key),
                **{name: "" for name in names},
            }
        cells[key]["|".join(row.group)] = row.value / _scale_for(row.measure_key)
    ordered = sorted(cells, key=lambda k: (measurement_order(k[0][:3]), k[0], k[1]))
    return columns, [cells[k] for k in ordered]


def unit_effect_rows(ranking, record: dict) -> list[dict]:
    """The per-unit effects a within-unit ranking averaged (detail#76), one row per unit and
    measurement; empty for a between design, whose effect is a contrast of group means.
    """
    rows = []
    if ranking is None:
        return rows
    for item in ranking.contrasts:
        if not isinstance(item, compare.Effect) or not item.unit_effects:
            continue
        effect_is = (
            "Spearman rho over the unit's levels"
            if ranking.design == compare.TREND
            else "log ratio of the unit's level means, b over a"
        )
        for unit, value in item.unit_effects:
            rows.append(
                {
                    **_measurement_cells(item, record),
                    "observation": unit,
                    "effect": value,
                    "effect_is": effect_is,
                    "pair_a": "" if item.pair is None else item.pair[0],
                    "pair_b": "" if item.pair is None else item.pair[1],
                }
            )
    return rows


def methods_sentence(record: dict, ranking) -> str:
    """A paragraph for a Methods section, from the record and its ranking (detail#76): which
    test, what was shuffled, how many draws or that it was exact, which P is reported and at
    what level, the correction beside it, and n. Generated, so it cannot disagree with what
    ran."""
    unit = _unit_word(record)
    column = record.get("design_column") or ", ".join(record.get("group_by") or [])
    groups = record.get("groups") or []
    n_obs = record.get("n_observations", 0)
    head = (
        f"Each {unit} is one observation ({n_obs} observations over "
        f"{len(groups)} group(s) of {column})."
    )
    if ranking is None or not ranking.contrasts:
        return f"{head} Fewer than two groups were compared, so nothing was contrasted."
    n = ranking.n_units
    family = len(ranking.contrasts)
    if ranking.design == compare.BETWEEN:
        largest = (
            " The effect is the largest log ratio between any two groups."
            if ranking.profile
            else ""
        )
        design = (
            f"Groups were compared between {unit}s (n = {n}): each measure's effect is the "
            f"log ratio of the group means.{largest}"
        )
        null = f"{ranking.draws} shuffles of the group labels among the {unit}s"
    elif ranking.design == compare.PAIRED:
        a, b = ranking.levels or ("a", "b")
        design = (
            f"{b} was compared with {a} within each {unit} (n = {n}): each measure's "
            f"effect is the mean over {unit}s of the log ratio of the {unit}'s {b} value to "
            f"its {a} value."
        )
        null = (
            f"every one of the {ranking.draws} sign-flip configurations (an exact test)"
            if ranking.exact
            else f"{ranking.draws} random sign flips of the per-{unit} effects"
        )
    elif ranking.design == compare.TREND:
        design = (
            f"The relationship to {column} was tested within each {unit} (n = {n}): each "
            f"measure's effect is the mean over {unit}s of Spearman's rank correlation "
            f"between the {unit}'s values and {column}."
        )
        null = f"{ranking.draws} shuffles of each {unit}'s {column} values within it"
    else:
        design = (
            f"{column} was compared within each {unit} (n = {n}) over its "
            f"{len(ranking.level_values)} values: each measure's effect is the largest mean "
            f"log ratio between any two of them, the pair chosen by the same statistic the "
            f"null uses."
        )
        null = f"{ranking.draws} shuffles of each {unit}'s {column} values within it"
    statistic = ""
    if ranking.statistic == compare.STATISTIC_T:
        s0 = "" if ranking.s0 is None else f" (s0 = {ranking.s0:.3g})"
        statistic = (
            f" Measures were scored by a moderated t: the mean effect over its spread "
            f"across {unit}s plus the family's median spread{s0}."
        )
    if ranking.floor is None:
        verdict = (
            f" No null distribution could be computed ({ranking.no_floor_reason}); the "
            f"effects are reported as measured, without a test."
        )
    elif ranking.verdict == compare.VERDICT_OWN:
        verdict = (
            f" Each of the {family} measures was tested on its own against {null}: P is the "
            f"share of draws in which the measure's statistic reached the observed one, and "
            f"a measure is reported as differing at P <= {compare.FLOOR_ALPHA:g}, with a "
            f"Benjamini-Hochberg false-discovery rate (q) over the {family} measures beside it."
        )
    else:
        verdict = (
            f" The null distribution was {null}. P_FWE is family-wise: the share of draws in "
            f"which the largest statistic over the {family} measures reached the observed one, "
            f"and a measure is reported as differing at P_FWE <= {compare.FLOOR_ALPHA:g}. "
            f"Each measure's own P and a Benjamini-Hochberg false-discovery rate (q) over the "
            f"family are reported beside it."
        )
    floor = ""
    if ranking.floor is not None and ranking.exact:
        floor = (
            f" The smallest P an exact test of {n} {unit}s can report is "
            f"{2.0 / ranking.draws:.3g}."
        )
    elif ranking.floor is not None:
        floor = f" A P below 1/{ranking.draws} is reported as that bound."
    return f"{head} {design}{statistic}{verdict}{floor}"


def shown_observations(record: dict, obs, ranking, ancestors):
    """The observations the per-measurement tables and figures cover: over the node source
    the kept nodes and the first few below the floor (§24 *The node source*); every
    observation otherwise."""
    if record.get("source") != "nodes" or ranking is None:
        return list(obs)
    labels = {item.measure_key[1] for item in shown_items(ranking, ancestors)}
    return [o for o in obs if o.measure_key[1] in labels]


def write_result_tables(
    out_dir: Path, record: dict, obs, ranking, *, shown=None
) -> list[Path]:
    """Write the result tables beside a comparison (detail#76): ``ranking.csv`` (every
    item), ``significant.csv`` (what cleared), ``units.csv`` (wide, over ``shown`` or every
    observation), ``unit_effects.csv`` and ``methods.txt``. Written by the run and again by
    every export, so a script reads a result without drawing anything."""
    out_dir = Path(out_dir)
    written = []
    shown = list(obs) if shown is None else list(shown)
    if ranking is not None:
        path = out_dir / RANKING_FILENAME
        tablib.write_tidy_csv(path, ranking_csv_rows(ranking, record), RANKING_COLUMNS)
        written.append(path)
        path = out_dir / SIGNIFICANT_FILENAME
        tablib.write_tidy_csv(
            path, significant_rows(ranking, record), SIGNIFICANT_COLUMNS
        )
        written.append(path)
        path = out_dir / UNIT_EFFECTS_FILENAME
        tablib.write_tidy_csv(
            path, unit_effect_rows(ranking, record), UNIT_EFFECTS_COLUMNS
        )
        written.append(path)
    columns, rows = units_rows(
        shown, record, _all_groups(record, ranking, obs_summaries(shown))
    )
    path = out_dir / UNITS_FILENAME
    tablib.write_tidy_csv(path, rows, columns)
    written.append(path)
    path = out_dir / METHODS_FILENAME
    path.write_text(methods_sentence(record, ranking) + "\n", encoding="utf-8")
    written.append(path)
    return written


def _write_summary(path: Path, record: dict, by_measure: dict, ranking) -> None:
    path_of = (
        "|".join(record.get("aggregate_to") or []) or "(none — one point per recording)"
    )
    lines = [
        f"# Comparison {record.get('id', '')}",
        "",
        f"- **grouped by**: {', '.join(record.get('group_by') or [])}",
        # Printed on the summary as well as every row, because "one point is one animal" and "one
        # point is one session" are different claims about the same numbers (§24).
        f"- **one observation is**: {path_of}",
        f"- **source artifact**: {record.get('source_run_id', '')}"
        + (
            " (every node of its node cache)" if record.get("source") == "nodes" else ""
        ),
        f"- **observations**: {record.get('n_observations', 0)}",
    ]
    if record.get("design"):
        lines.append(
            f"- **design**: {record['design']} — {record.get('design_reason', '')}"
        )
    if left_out_sentence(record):
        lines.append(f"- **left out**: {left_out_sentence(record)}")
    if screened_sentence(record):
        lines.append(f"- **not independent**: {screened_sentence(record)}")
    lines += [
        "",
        "Descriptive only — no models. `stats.csv` is long form, one row per observation with "
        "its factors, for inferential work elsewhere; `units.csv` is the same, one row per "
        "unit with a column per group; `significant.csv` holds the statistics behind every "
        "result that cleared.",
        "",
        "## Methods",
        "",
        methods_sentence(record, ranking),
        "",
    ]
    if ranking is not None and ranking.contrasts:
        lines += _summary_ranking_lines(record, ranking)
    lines += [
        "## Groups",
        "",
        "| measure | group | n | coverage | mean | median | sd |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for measure_key, summaries in sorted(by_measure.items()):
        unit = measure_unit(measure_key)
        for summary in sorted(summaries, key=lambda s: s.group):
            cells = [
                shown_value(v, unit)[0]
                for v in (summary.mean, summary.median, summary.sd)
            ]
            shown_unit = shown_value(summary.mean, unit)[1]
            name = measure_title(measure_key, remainders_of(record))
            name = f"{name} ({shown_unit})" if shown_unit else name
            lines.append(
                f"| {name} | {tablib.md_cell('|'.join(summary.group))} | {summary.n} | "
                f"{summary.coverage} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path, record: dict, artifacts: list[Path], out_dir: Path, theme: str
) -> dict:
    manifest = {
        "comparison": record.get("id"),
        # Which surface the figures were drawn on. Light unless asked for; recorded because a
        # dark figure dropped into a paper is the one thing about it nothing else says.
        "figure_theme": theme,
        "source_run_id": record.get("source_run_id"),
        "group_by": record.get("group_by"),
        "aggregate_to": record.get("aggregate_to"),
        "provenance": record.get("provenance"),
        "artifacts": {
            p.relative_to(out_dir).as_posix(): provenance.sha256_file(p)
            for p in sorted(set(artifacts))
            if p.is_file()
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def export_comparison(
    experiment_dir: Path,
    record: dict,
    out_dir: Path,
    *,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    style: FigureStyle = DEFAULT_STYLE,
    progress: Callable[[str, float, str], None] | None = None,
) -> dict:
    """Write the full export tree for a saved comparison; return the manifest.

    ``progress(stage, fraction, detail)`` reports through it. Every surface offered a
    *determinate* bar for this and nothing ever drove it, so the bar sat at 0 for the whole
    render and then jumped to done — which is the "silent long op" failure with an extra step,
    because a bar that does not move reads as a hang more loudly than no bar at all. It is
    minutes on a comparison with many measurements: one matplotlib figure per measurement, saved
    in each requested format.

    Counting figures rather than weighing them is right *here*, unlike a loop over spans of very
    different sizes: every figure is one render of a handful of points.
    """
    from detail.report import figures as figlib
    from detail.storage import comparisons as comparisons_store

    figlib.check_formats(tuple(formats))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = comparisons_store.comparison_dir(experiment_dir, record["id"])
    obs = _observations(_read_stats(source))
    ancestors = read_hierarchy(source)
    # With `coverage_by_group`, or the exported column reports `coverage == n` for every group —
    # the exact reading it exists to refuse. The GUI table had the same bug.
    summaries = compare.summarize(obs, coverage_by_group=compare.coverage_by_group(obs))
    note = observation_note(record)
    ranking = rank_record(obs, record, ancestors)
    # The shown set (§24 *The node source*): the per-measurement tables and figures over
    # the kept nodes and the first few below the floor; `ranking.csv` carries the rest.
    figure_obs = shown_observations(record, obs, ranking, ancestors)
    if len(figure_obs) != len(obs):
        labels = {o.measure_key[1] for o in figure_obs}
        summaries = [s for s in summaries if s.measure_key[1] in labels]
    by_measure = _by_measure(summaries)

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    previous = exporter_lib.previous_artifacts(manifest_path)

    # Keep the export tree self-contained: when exporting somewhere other than the comparison's
    # own directory, the observations go with it, or the tree is a set of figures whose source
    # numbers live somewhere else.
    stats = out_dir / STATS_FILENAME
    if stats.resolve() != (source / STATS_FILENAME).resolve():
        stats.write_text(
            (source / STATS_FILENAME).read_text(encoding="utf-8-sig"), encoding="utf-8"
        )

    report = progress or _SILENT
    artifacts: list[Path] = [stats]
    report("tables", 0.0, "")
    artifacts += _write_tables(tables_dir, by_measure)
    artifacts += _write_grid_tables(tables_dir, record, by_measure)
    report("tables", 1.0, f"{len(artifacts) - 1} table(s)")
    artifacts += _measure_figures(
        figures_dir,
        by_measure,
        figure_obs,
        record,
        ranking,
        note,
        formats,
        style,
        report,
    )
    report("figures", 1.0, "")
    report("panels", 0.0, "")
    artifacts += _panels_figure(
        figures_dir, by_measure, figure_obs, record, ranking, note, formats, style
    )
    report("panels", 1.0, "")
    report("ranking", 0.0, "")
    if ranking is not None:
        artifacts += _ranking_figure(
            figures_dir, ranking, note, formats, style, remainders_of(record)
        )
    artifacts += write_result_tables(out_dir, record, obs, ranking, shown=figure_obs)
    report("ranking", 1.0, "")
    report("summary", 0.0, "")
    summary = out_dir / "summary.md"
    _write_summary(summary, record, by_measure, ranking)
    artifacts.append(summary)
    report("summary", 1.0, "")
    manifest = _write_manifest(manifest_path, record, artifacts, out_dir, style.theme)
    exporter_lib.drop_superseded(out_dir, previous, set(manifest["artifacts"]))
    return manifest
