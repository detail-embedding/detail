"""Figures for a §19 behavior artifact — the pictures a result screen shows, as files (#336).

The rule is that any analysis you show should be exportable, and a result screen is half
picture. Exporting the ethogram's rows and not the ethogram is the same gap one level down as
exporting a clip without the bout behind it.

**Drawn in matplotlib, not grabbed off the widget.** A `QWidget.grab()` is a screenshot: screen
resolution, no vector output, and impossible from the CLI — where the export is actually headed,
because the destination is a collaborator's machine or a GLM. `report/figures.py` already exists
for exactly this (five analysis modules draw through it), is `Agg`-backed, and emits PNG and SVG
from one figure. So these are real figures that render with no display attached.

One function per visual, each taking the artifact and returning a matplotlib figure, so the
export target stays a thin call and the drawing is testable without Qt.
"""

from __future__ import annotations

from detail.core import chart_theme
from detail.report import figures
from detail.report.style import DEFAULT_STYLE, FigureStyle

#: Seconds per nanosecond — every artifact time is `t_ns` (Tenet 8), every axis is seconds.
_NS = 1e9


def _label_colors(labels, style: FigureStyle = DEFAULT_STYLE) -> dict[str, str]:
    """A stable colour per label, assigned in sorted order.

    Sorted rather than first-seen: a figure of recording B must colour "rearing" the same as
    a figure of recording A, and first-seen order depends on which bout happens to come first.
    The assignment is `chart_theme.assign_identities`, the one the GUI's ethogram uses too.
    """
    return chart_theme.assign_identities(labels, style.chart)


def ethogram(
    instances, coverage, *, title: str = "", style: FigureStyle = DEFAULT_STYLE
):
    """One lane per recording, each bout a block coloured by its behavior.

    The picture the interpretation screen shows, as a figure: what happened, when, in which
    recording. Recordings run down the y axis so a cohort reads as a block — the thing you are
    looking for is usually one animal's lane differing from the rest.

    Args:
        instances: `BehaviorInstance` rows, already narrowed to whatever scope was asked for.
        coverage: `CoverageRow` rows, which set each lane's full extent — a lane drawn only to
            its last bout would silently shorten a recording that stopped behaving.
        title: figure title; omitted when empty.
        style: the figure style — light unless the export asked for dark.

    Returns:
        A matplotlib figure. The caller saves and closes it (`figures.save_figure`).
    """
    import matplotlib.pyplot as plt

    recordings = sorted(
        {row.recording_id for row in instances} | {c.recording_id for c in coverage}
    )
    names = sorted({row.label or row.source_label for row in instances})
    colors = _label_colors(names, style)
    observed = {
        c.recording_id: (c.start_t_ns / _NS, c.end_t_ns / _NS) for c in coverage
    }
    rows = [
        (
            recording,
            [
                (r.t_ns / _NS, r.end_t_ns / _NS, r.label or r.source_label)
                for r in instances
                if r.recording_id == recording
            ],
        )
        for recording in recordings
    ]

    with figures.styled(style):
        fig, ax = plt.subplots(
            figsize=(
                style.double_col_width_in,
                max(1.4, 0.32 * len(recordings) + 0.9),
            )
        )
        # First recording at the top, like the screen.
        figures.lanes(ax, rows, colors, extents=observed, style=style)
        if title:
            ax.set_title(title)
    return fig


def time_by_behavior(instances, *, title: str = "", style: FigureStyle = DEFAULT_STYLE):
    """How much time each behavior accounts for — the bar chart beside the ethogram.

    Built on **bouts**, not rows: a multi-label cluster contributes one row per label over the
    same span, so summing rows would double-count the time (`behavior_output.bouts`). Each bar
    wears its behavior's colour because the same export draws that behavior as a lane in that
    colour — identity, not rank.
    """
    import matplotlib.pyplot as plt

    from detail.storage.behavior_output import bouts

    totals: dict[str, float] = {}
    for row in bouts(list(instances)):
        name = row.label or row.source_label
        totals[name] = totals.get(name, 0.0) + (row.end_t_ns - row.t_ns) / _NS
    order = sorted(totals, key=lambda n: totals[n], reverse=True)
    colors = _label_colors(list(totals), style)

    with figures.styled(style):
        fig, ax = plt.subplots(
            figsize=(style.single_col_width_in * 1.5, max(1.6, 0.26 * len(order) + 1.0))
        )
        # `bars(horizontal=True)` draws the first label at the top, so the longest leads.
        figures.bars(
            ax,
            order,
            [totals[n] for n in order],
            horizontal=True,
            colors=[colors[n] for n in order],
            xlabel="total time (s)",
            style=style,
        )
        if title:
            ax.set_title(title)
    return fig


def bout_durations(instances, *, title: str = "", style: FigureStyle = DEFAULT_STYLE):
    """The spread of bout lengths per behavior — what a mean hides.

    A behavior whose bouts are all 0.4s and one whose bouts run 0.2–30s have the same mean and
    are not the same thing, which is the reason this is a distribution rather than a number.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    from detail.storage.behavior_output import bouts

    by_name: dict[str, list[float]] = {}
    for row in bouts(list(instances)):
        name = row.label or row.source_label
        by_name.setdefault(name, []).append((row.end_t_ns - row.t_ns) / _NS)
    order = sorted(by_name)
    colors = _label_colors(order, style)
    marks = style.marks

    with figures.styled(style):
        fig, ax = plt.subplots(
            figsize=(style.single_col_width_in * 1.5, max(1.8, 0.26 * len(order) + 1.2))
        )
        for index, name in enumerate(reversed(order)):
            values = by_name[name]
            # No alpha: a behavior is the same colour in every figure of this export, and
            # translucency makes it a different colour wherever points overlap. Overlap is
            # shown by jitter and the surface ring instead, which do not lie about the hue.
            jitter = [index + 0.10 * ((i % 5) - 2) / 2 for i in range(len(values))]
            ax.scatter(
                values,
                jitter,
                s=(marks.marker_size * 0.8) ** 2,
                color=colors[name],
                edgecolors=style.chart.surface,
                linewidths=marks.marker_ring,
                zorder=2,
            )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(list(reversed(order)))
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("bout duration (s)")
        # Durations are heavy-tailed: a linear axis puts a thousand half-second bouts in one
        # pixel and renders perfectly while saying nothing (CLAUDE.md's recorded class).
        # A log axis only where the spread earns it. Durations *are* heavy-tailed in real
        # data — a linear axis puts a thousand half-second bouts in one pixel — but on a
        # narrow spread it buys nothing and costs readability, crowding "0.9" against "1.0".
        # Twenty-fold is where the tail starts to matter.
        positive = [v for values in by_name.values() for v in values if v > 0]
        if positive and max(positive) / min(positive) > 20:
            ax.set_xscale("log")
            # Plain numbers, not `2 x 10^-1`. This axis is read by somebody asking how long a
            # behavior lasts, and a duration in scientific notation has to be decoded before it
            # can be judged — the same reason `enableAutoSIPrefix(False)` is the rule for the
            # Qt plots (gui_design.md §3.1).
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(left=0)
        if title:
            ax.set_title(title)
    return fig
