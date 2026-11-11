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

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

from detail.analysis import compare, provenance
from detail.analysis import windows as windows_lib
from detail.core.utils import safe_stem
from detail.report import exporter as exporter_lib
from detail.report import tables as tablib
from detail.report.style import DEFAULT_STYLE, FigureStyle

STATS_FILENAME = "stats.csv"
TABLE_COLUMNS = ["group", "n", "coverage", "mean", "median", "sd"]


def measure_unit(measure_key: tuple) -> str:
    """The unit a measure's numbers are in, or ``""`` when it is a plain count.

    ``compare.py`` deliberately knows nothing about the measure registry — it works on whatever
    rows it is handed. Rendering is where the unit has to come back, or a reader gets a column of
    unlabelled floats mixing nanoseconds, counts and ratios, which is what every comparison
    surface shipped: a mean duration printed as ``4956702127.6595745``.
    """
    from detail.analysis.measures import MEASURES

    spec = MEASURES.get(measure_key[0] if measure_key else "")
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
    return f"{float(value):.4g}", unit


def measure_title(measure_key: tuple) -> str:
    """What a *person* reads for one measurement: ``Bouts · rearing · whole recording``.

    The twin of :func:`_measure_stem`, which keeps producing the id form. The split is the point:
    a stem names a file and a `stats.csv` cell and must never drift, while a table row and a
    figure title are read by someone who did not write the registry. Printing the stem in both
    places is what put `n_instances__rearing__whole` on screen.

    Only the measure id maps to a label — the remaining parts of the key are the user's own
    behavior names, a transition's destination, and the window, all already human.
    """
    from detail.analysis.measures import MEASURES

    if not measure_key:
        return "measure"
    spec = MEASURES.get(measure_key[0])
    parts = [spec.label if spec is not None else measure_key[0]]
    parts += [str(p) for p in measure_key[1:] if p and p != "whole"]
    return "  ·  ".join(parts)


def measure_axis_label(measure_key: tuple) -> str:
    """``total_time (s)`` — the measure with the unit a reader will actually see."""
    from detail.analysis.measures import MEASURES

    unit = measure_unit(measure_key)
    shown = {"ns": "s", "fraction": "%"}.get(unit, unit)
    spec = MEASURES.get(measure_key[0]) if measure_key else None
    name = spec.label if spec is not None else (measure_key[0] if measure_key else "")
    return f"{name} ({shown})" if shown else name


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
        )
        for row in rows
    ]


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
    from detail.analysis.measures import MEASURES

    spec = MEASURES.get(measure_key[0] if measure_key else "")
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
        name = measure_title(measure_key)
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


def _measure_figures(
    out_dir: Path, by_measure: dict, note: str, formats, style, report=_SILENT
) -> list[Path]:
    """One figure per *measurement* — dots across groups, or a curve across windows.

    **The window is not part of a figure's identity.** It used to be: `_by_measure` bucketed on
    the whole `measure_key`, which ends in `window_id`, so a sliding-window comparison wrote one
    two-point dot plot per window — hundreds of figures for a sliding window over one
    recording, where §24 asks for "dot plots, time-resolved
    curves, the difference ranking". Collapsing the window out removes the blow-up and produces
    the missing curve in the same change, because a measurement over many windows *is* the curve.
    """
    from detail.report import figures as figlib

    written: list[Path] = []
    grouped: dict[tuple, dict[str, list]] = {}
    for measure_key, summaries in by_measure.items():
        if _fans_out(measure_key):
            continue
        window = measure_key[3] if len(measure_key) > 3 else ""
        grouped.setdefault(_measurement(measure_key), {})[window] = summaries

    ordered_measurements = sorted(grouped.items())
    for index, (measurement, per_window) in enumerate(ordered_measurements):
        report("figures", index / len(ordered_measurements), _measure_stem(measurement))
        windows = sorted(per_window, key=windows_lib.window_sort_key)
        scale = _scale_for(measurement)
        with figlib.styled(style):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            if len(windows) == 1:
                ordered = sorted(per_window[windows[0]], key=lambda s: s.group)
                figlib.observation_dots(
                    ax,
                    ["|".join(s.group) for s in ordered],
                    [[v / scale for v in s.values] for s in ordered],
                    style=style,
                    ylabel=measure_axis_label(measurement),
                )
            else:
                groups = sorted(
                    {s.group for windowed in per_window.values() for s in windowed}
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
                    ylabel=measure_axis_label(measurement),
                )
            _caption(ax, note, style)
            written += figlib.save_figure(
                fig, out_dir / _measure_stem(measurement), formats
            )
    return written


def _paired_figures(
    out_dir: Path, obs, note: str, formats, style, report=_SILENT
) -> list[Path]:
    """A line per subject, when the same subject appears in more than one group.

    Fires on the structure rather than on a setting: if no observation key spans two groups there
    is nothing to pair and nothing is drawn. That is the crossover shape
    `analysis/cohort.infer_subject_column` already detects on the selecting side.

    Reads observations, not `GroupSummary` — `compare.summarize` buckets by group and sorts the
    values, dropping `Observation.key`, so the summaries cannot say which point belongs to which
    animal. That is exactly the identity a paired plot is made of.
    """
    from detail.report import figures as figlib

    written: list[Path] = []
    by_window: dict[tuple, dict[str, dict[tuple, dict[tuple, float]]]] = {}
    for row in obs:
        window = row.measure_key[3] if len(row.measure_key) > 3 else ""
        by_window.setdefault(_measurement(row.measure_key), {}).setdefault(
            window, {}
        ).setdefault(row.key, {})[row.group] = row.value

    ordered = sorted(by_window.items())
    for index, (measurement, per_window) in enumerate(ordered):
        report("paired figures", index / len(ordered), _measure_stem(measurement))
        # A pairing has one line per subject; across many windows there is one line per subject
        # *per window*, which is a different figure and a design question §24 has not answered.
        # Keying on the measurement alone silently answered it the worst way: every window wrote
        # into the same slot, the last one survived, and it was captioned as the whole
        # measurement. The time-resolved reading is the curve `_measure_figures` draws.
        if len(per_window) != 1:
            continue
        per_subject = next(iter(per_window.values()))
        groups = sorted({g for spans in per_subject.values() for g in spans})
        spanning = {key: spans for key, spans in per_subject.items() if len(spans) > 1}
        if len(groups) < 2 or not spanning:
            continue
        scale = _scale_for(measurement)
        pairs = [
            (
                "|".join(key),
                [
                    None if group not in spans else spans[group] / scale
                    for group in groups
                ],
            )
            for key, spans in sorted(spanning.items())
        ]
        with figlib.styled(style):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            figlib.paired_lines(
                ax,
                ["|".join(g) for g in groups],
                pairs,
                style=style,
                ylabel=measure_axis_label(measurement),
            )
            _caption(ax, note, style)
            written += figlib.save_figure(
                fig, out_dir / f"{_measure_stem(measurement)}__paired", formats
            )
    return written


def _ranking_figure(out_dir: Path, ranking, note: str, formats, style) -> list[Path]:
    """Draw the difference ranking. Takes a ``compare.Ranking``, which cannot exist without its
    floor — so "no ranking without a floor" is structural rather than remembered (§24).
    """
    from detail.report import figures as figlib

    if not ranking.contrasts:
        return []
    # Unbounded changes have no bar to draw — a fold change from zero is off every axis — so they
    # are listed in `summary.md` and left out of the chart rather than given an invented height.
    drawn = [c for c in ranking.contrasts if c.magnitude is not None]
    if not drawn:
        return []
    with figlib.styled(style):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        figlib.ranked_bars_with_floor(
            ax,
            [measure_title(c.measure_key) for c in drawn],
            [c.magnitude for c in drawn],
            ranking.floor,
            style=style,
            xlabel="|fold change| between groups (log scale)",
        )
        _label_fold_change(ax, drawn)
        _caption(ax, note, style)
        return figlib.save_figure(fig, out_dir / "difference_ranking", formats)


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
    return ["|".join(first.a.group), "|".join(first.b.group)]


def _change_text(item) -> str:
    """How the change reads: a percent, or what actually happened when there is no percent."""
    if item.unbounded:
        return "appeared" if (item.a.mean or 0) == 0 else "vanished"
    percent = item.percent_change
    if percent is None:
        return "—"
    return f"{percent:+.0f}%"


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
        f"- **source artifact**: {record.get('source_run_id', '')}",
        f"- **observations**: {record.get('n_observations', 0)}",
        "",
        "Descriptive only — no tests, no corrections, no models. `stats.csv` is long form, one row "
        "per observation with its factors, for inferential work elsewhere.",
        "",
    ]
    if ranking is not None and ranking.contrasts:
        lines += ["## Largest changes", ""]
        if ranking.floor is None:
            # Stated, not left blank. A missing floor and a floor nothing cleared look identical
            # on the page, and they are different facts.
            lines += [
                f"**No floor computed.** {ranking.no_floor_reason}. The changes below are "
                "reported as measured; this comparison does not say how large a change "
                "relabelling alone would have produced.",
                "",
                "| measure | "
                + " | ".join(tablib.md_cell(g) for g in group_headers(ranking))
                + f" | change ({_direction(ranking)}) |",
                "| --- | --- | --- | --- |",
            ]
        else:
            lines += [
                f"Floor (95th percentile of {ranking.draws} shuffled-label draws): "
                f"**{ranking.floor:.4g}** as a fold change "
                f"({math.exp(ranking.floor):.3g}x). Measures are ranked by |fold change|, "
                "which orders counts, durations and fractions on one scale.",
                "",
                "| measure | "
                + " | ".join(tablib.md_cell(g) for g in group_headers(ranking))
                + f" | change ({_direction(ranking)}) | clears floor |",
                "| --- | --- | --- | --- | --- |",
            ]
        for item in ranking.contrasts:
            unit = measure_unit(item.measure_key)
            a_text, shown_unit = shown_value(item.a.mean, unit)
            b_text, _ = shown_value(item.b.mean, unit)
            suffix = f" {shown_unit}" if shown_unit else ""
            change = _change_text(item)
            row = (
                f"| {measure_title(item.measure_key)} | {a_text}{suffix} | "
                f"{b_text}{suffix} | {change} |"
            )
            if ranking.floor is not None:
                row += " yes |" if ranking.clears_floor(item) else " no |"
            lines.append(row)
        lines.append("")
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
            name = measure_title(measure_key)
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
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
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
    from detail.storage import comparisons as comparisons_store

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = comparisons_store.comparison_dir(experiment_dir, record["id"])
    obs = _observations(_read_stats(source))
    # With `coverage_by_group`, or the exported column reports `coverage == n` for every group —
    # the exact reading it exists to refuse. The GUI table had the same bug.
    summaries = compare.summarize(obs, coverage_by_group=compare.coverage_by_group(obs))
    by_measure = _by_measure(summaries)

    note = observation_note(record)
    groups = sorted({o.group for o in obs})
    ranking = (
        compare.rank_by_difference(
            obs,
            (groups[0], groups[1]),
            draws=int(record.get("draws", 200)),
            seed=int(record.get("seed", 0)),
        )
        if len(groups) == 2
        else None
    )

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
    artifacts += _measure_figures(figures_dir, by_measure, note, formats, style, report)
    report("figures", 1.0, "")
    artifacts += _paired_figures(figures_dir, obs, note, formats, style, report)
    report("paired figures", 1.0, "")
    if ranking is not None:
        report("ranking", 0.0, "")
        artifacts += _ranking_figure(figures_dir, ranking, note, formats, style)
        report("ranking", 1.0, "")
    report("summary", 0.0, "")
    summary = out_dir / "summary.md"
    _write_summary(summary, record, by_measure, ranking)
    artifacts.append(summary)
    report("summary", 1.0, "")
    manifest = _write_manifest(manifest_path, record, artifacts, out_dir, style.theme)
    exporter_lib.drop_superseded(out_dir, previous, set(manifest["artifacts"]))
    return manifest
