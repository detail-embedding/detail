"""Module-agnostic styled matplotlib primitives — the chart design language, on paper.

The generic building blocks a module's domain figures compose: a style context, a multi-format
save, and one function per chart form this project draws — points with a summary, distributions,
bars, curves with and without a band, step curves, histograms, heatmaps, scatter, and the lanes of
an ethogram — plus the furniture every figure shares (the legend, a caption, a reference line,
panel letters). Rendered headless (Agg). No domain knowledge here.

Every colour, size and weight comes from ``FigureStyle`` (``report/style.py``), which reads
``core/chart_theme.py``. A primitive never spells a colour: a hard-coded ``#444`` is how a median
came to be one grey here and another in the GUI. The rules the forms follow — when a violin is
allowed, why a summary is ink, where the legend goes — are ``docs/chart_design.md``.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detail.core.errors import UserInputError
from detail.report.style import (
    DEFAULT_STYLE,
    SAVE_PARAMS,
    FigureStyle,
    diverging_cmap,
    rc_params,
    sequential_cmap,
)

#: A group with fewer observations than this is drawn as its points and a median, never as a
#: violin or a box. At the cohort sizes this project works at — three animals a group, sometimes
#: two — a density estimate is a shape invented from a handful of numbers, and the points *are*
#: the result (§24).
VIOLIN_MIN_N = 10

#: Past this many observations a group's points are no longer drawn over its violin: they stop
#: being countable and start hiding the shape they sit on.
POINTS_OVER_VIOLIN_MAX = 120

#: Past this many points a scatter becomes a density. "Each dot is one" and "more colour means more"
#: are different claims about the same marks, so the switch is visible in the legend.
SCATTER_DENSITY_ABOVE = 5000

#: The golden-ratio fraction: successive multiples mod 1 spread evenly over [0, 1) without
#: clumping, which gives a jitter that is deterministic (re-export is byte-identical, §18.5 L1)
#: and never lands two neighbours on one spot.
_GOLDEN = 0.6180339887498949


@contextmanager
def styled(style: FigureStyle = DEFAULT_STYLE):
    """Apply the design language's rcParams for the duration of the block."""
    with plt.rc_context(rc_params(style)):
        yield


# Strip the embedded creation date so re-rendering the same figure is reproducible (the PDF
# /CreationDate and PNG "Creation Time" otherwise change every run). Combined with a fixed
# ``svg.hashsalt`` (style.rc_params) this makes figure output deterministic.
_NO_DATE_METADATA = {
    "pdf": {"CreationDate": None},
    "png": {"Creation Time": None},
    "svg": {"Date": None},
}


def check_formats(formats: tuple[str, ...]) -> None:
    """Refuse a figure format matplotlib cannot write, before anything is drawn.

    Raises:
        UserInputError: a format matplotlib has no writer for; the message lists them.
    """
    from matplotlib.backend_bases import FigureCanvasBase

    supported = FigureCanvasBase.get_supported_filetypes()
    unknown = [fmt for fmt in formats if fmt not in supported]
    if unknown:
        raise UserInputError(
            f"cannot write figures as {', '.join(unknown)}; choose from "
            f"{', '.join(sorted(supported))}"
        )


def save_figure(
    fig, out_stem: Path | str, formats: tuple[str, ...] = ("pdf", "svg", "png")
) -> list[Path]:
    """Save ``fig`` as ``<out_stem>.<fmt>`` for each format, close it, and return the paths.

    Self-sufficient: it does not rely on being called inside ``styled``. The save-time settings
    (editable fonts, stable SVG ids, the tight crop, the surface behind the figure) are applied
    here, and colliding category labels are turned before the first format is written.
    """
    stem = Path(out_stem)
    written: list[Path] = []
    with plt.rc_context(SAVE_PARAMS):
        _turn_colliding_tick_labels(fig)
        for fmt in formats:
            path = stem.with_suffix(f".{fmt}")
            fig.savefig(
                path,
                format=fmt,
                metadata=_NO_DATE_METADATA.get(fmt),
                bbox_inches="tight",
                pad_inches=0.04,
                facecolor=fig.get_facecolor(),
            )
            written.append(path)
    plt.close(fig)
    return written


#: The angles a colliding row of category labels is turned to, smallest first.
_TURN_ANGLES = (30, 45, 60, 90)


def _turn_colliding_tick_labels(fig) -> None:
    """Rotate an axis's x tick labels when any two of them overlap — by as little as works.

    Category names are as long as the user made them, so no fixed rotation is right: "Vehicle" and
    "Drug" sit level, "Drug+Antagonist" beside them runs into its neighbour. Turned labels
    are parallel lines of text, one per tick, and they clear each other when the perpendicular
    distance between neighbours — tick spacing x sin(angle) — is at least the text's height; the
    smallest angle in :data:`_TURN_ANGLES` that achieves it is used. Measured on the rendered
    labels, so the rule is "never overlap" rather than "rotate past N characters".
    """
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    gap = 2.0 * fig.dpi / 72.0
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        labels = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        if len(labels) < 2 or any(t.get_rotation() for t in labels):
            continue
        boxes = sorted(
            (t.get_window_extent(renderer) for t in labels), key=lambda b: b.x0
        )
        if not any(a.x1 + gap > b.x0 for a, b in zip(boxes, boxes[1:])):
            continue
        angle = turn_angle(boxes, gap)
        for t in labels:
            t.set_rotation(angle)
            t.set_horizontalalignment("center" if angle == 90 else "right")
            t.set_rotation_mode("anchor")


def turn_angle(level_boxes, gap: float = 0.0) -> int:
    """The smallest angle at which labels measured level as ``level_boxes`` stop overlapping."""
    import math

    centres = sorted((b.x0 + b.x1) / 2 for b in level_boxes)
    spacing = min(b - a for a, b in zip(centres, centres[1:]))
    height = max(b.height for b in level_boxes) + gap
    return next(
        (a for a in _TURN_ANGLES if spacing * math.sin(math.radians(a)) >= height),
        _TURN_ANGLES[-1],
    )


def small_multiples(
    n: int,
    *,
    width: float,
    height_per: float,
    sharex: bool = True,
    sharey: bool = False,
    style: FigureStyle = DEFAULT_STYLE,
):
    """A vertical stack of ``n`` axes (optionally sharing x/y); returns ``(fig, [ax, ...])``."""
    with styled(style):
        fig, axes = plt.subplots(
            n,
            1,
            figsize=(width, height_per * n),
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
    return fig, [row[0] for row in axes]


def grid_multiples(
    n: int,
    *,
    cols: int,
    width: float,
    height_per: float,
    sharex: bool = True,
    sharey: bool = False,
    style: FigureStyle = DEFAULT_STYLE,
):
    """``n`` axes laid out in a ``cols``-wide grid; returns ``(fig, [ax, ...])`` row-major.

    The counterpart to ``small_multiples`` for panels that are compared side by side rather
    than read as a stacked timeline. Any trailing cell the grid over-allocates is hidden, so
    the caller gets exactly ``n`` axes back.
    """
    if n < 1 or cols < 1:
        raise ValueError(f"n and cols must be >= 1, got n={n}, cols={cols}")
    rows = -(-n // cols)
    with styled(style):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(width, height_per * rows),
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


# --------------------------------------------------------------------------------------------- #
# Furniture every figure shares
# --------------------------------------------------------------------------------------------- #


def place_legend(ax, *, style: FigureStyle = DEFAULT_STYLE, handles=None, title=None):
    """The house legend: outside the plot area, right of it, top-aligned, no frame.

    Outside because a legend inside the axes sits on whatever data happens to be in its corner —
    measured on the paired-lines primitive, where it covered two subjects' lines. Right rather
    than above because the strip above the axes carries the caption (``caption``). Constrained
    layout makes room for it, so nothing is clipped.
    """
    kwargs = {} if handles is None else {"handles": handles}
    placed = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title=title,
        alignment="left",
        **kwargs,
    )
    # The surface ring separates overlapping points on the plot; on a legend key it cuts the line
    # either side of the marker, and the key reads as a dashed series.
    for handle in placed.legend_handles:
        if hasattr(handle, "set_markeredgewidth"):
            handle.set_markeredgewidth(0)
    return placed


def caption(ax, note: str, *, style: FigureStyle = DEFAULT_STYLE) -> None:
    """A muted one-line note above the axes, left-aligned — what one point *is*, say (§24)."""
    if not note:
        return
    ax.annotate(
        note,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=style.type.annotation,
        color=style.chart.text_muted,
        annotation_clip=False,
    )


def reference_line(
    ax,
    value: float,
    *,
    axis: str = "y",
    label: str = "",
    style: FigureStyle = DEFAULT_STYLE,
) -> None:
    """A line the data is read against — chance, a null floor, a cutoff.

    Dashed and in the reference ink, so it is never mistaken for a series; labelled in muted text
    rather than in red, because "chance" is a fact about the axis, not an alarm. The label sits
    *outside* the plot area — past the right edge for a horizontal line, above the top for a
    vertical one — because inside it lands on whatever data is at the line's end.
    """
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(
        value,
        color=style.chart.reference,
        lw=style.marks.reference_width,
        linestyle=(0, (4, 2.5)),
        zorder=1.5,
    )
    if not label:
        return
    if axis == "y":
        ax.annotate(
            label,
            xy=(1.0, value),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=style.type.annotation,
            color=style.chart.text_muted,
            annotation_clip=False,
        )
    else:
        ax.annotate(
            label,
            xy=(value, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style.type.annotation,
            color=style.chart.text_muted,
            annotation_clip=False,
        )


def panel_letters(axes, *, style: FigureStyle = DEFAULT_STYLE, letters=None) -> None:
    """Bold ``A``, ``B``, … in the top-left corner of each panel of a multi-panel figure."""
    letters = letters or [chr(ord("A") + i) for i in range(len(axes))]
    for ax, letter in zip(axes, letters):
        ax.annotate(
            letter,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-22, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=style.type.panel_letter,
            fontweight="bold",
            color=style.chart.text,
            annotation_clip=False,
        )


def _label_axes(ax, xlabel: str = "", ylabel: str = "") -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _spread(n: int, half_width: float) -> np.ndarray:
    """Deterministic horizontal offsets for ``n`` points within ``±half_width``."""
    if n <= 1:
        return np.zeros(max(n, 0), dtype=float)
    if n <= 12:
        # Few enough to lay out evenly: every point visible, none on top of another.
        return np.linspace(-half_width, half_width, n)
    return ((np.arange(n) * _GOLDEN) % 1.0 - 0.5) * 2 * half_width


def _points(ax, x, y, color: str, style: FigureStyle, *, size_scale: float = 1.0):
    """Observations as filled markers with a surface-coloured ring (the overlap separator)."""
    marks = style.marks
    ax.plot(
        x,
        y,
        "o",
        linestyle="none",
        color=color,
        markersize=marks.marker_size * size_scale,
        markeredgecolor=style.chart.surface,
        markeredgewidth=marks.marker_ring,
        alpha=marks.point_alpha,
        zorder=3,
    )


def _summary_tick(ax, center: float, value: float, half_width: float, style) -> None:
    """A group's summary as a short horizontal bar in ink, drawn over its points."""
    ax.plot(
        [center - half_width, center + half_width],
        [value, value],
        "-",
        color=style.chart.summary,
        lw=style.marks.summary_width,
        solid_capstyle="butt",
        zorder=4,
    )


# --------------------------------------------------------------------------------------------- #
# Group comparisons
# --------------------------------------------------------------------------------------------- #


def grouped_dots(
    ax,
    group_labels: list[str],
    series: list[tuple[str, list[float], list[float] | None]],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    ylabel: str = "",
    as_bars: bool = False,
) -> None:
    """Grouped dot (or bar) chart with optional error bars.

    ``series`` is ``[(label, values, errors_or_None), ...]``; each ``values``/``errors`` lists one
    entry per group. Series take the identity colours in order; groups sit along x. Error bars are
    ink, not the series colour — a bar drawn in its own hue disappears into the dot it belongs to.
    """
    n_groups = len(group_labels)
    n_series = len(series)
    centers = np.arange(n_groups, dtype=float)
    span = style.marks.bar_fraction if as_bars else 0.6
    width = span / max(n_series, 1)
    for s, (label, values, errors) in enumerate(series):
        offset = -span / 2 + width * (s + 0.5)
        x = centers + offset
        color = style.color(s)
        if as_bars:
            ax.bar(
                x,
                values,
                width=width,
                color=color,
                edgecolor=style.chart.surface,
                linewidth=style.marks.marker_ring * 2,
                label=label,
                zorder=2,
            )
            if errors is not None:
                ax.errorbar(
                    x,
                    values,
                    yerr=errors,
                    fmt="none",
                    ecolor=style.chart.summary,
                    lw=style.marks.summary_width * 0.8,
                    zorder=3,
                )
        else:
            if errors is not None:
                ax.errorbar(
                    x,
                    values,
                    yerr=errors,
                    fmt="none",
                    ecolor=style.chart.summary,
                    lw=style.marks.summary_width * 0.8,
                    zorder=2,
                )
            ax.plot(
                x,
                values,
                "o",
                linestyle="none",
                color=color,
                markeredgecolor=style.chart.surface,
                markeredgewidth=style.marks.marker_ring,
                label=label,
                zorder=3,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.6, n_groups - 0.4)
    if as_bars:
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    _label_axes(ax, ylabel=ylabel)
    if n_series > 1:
        place_legend(ax, style=style)


def observation_dots(
    ax,
    group_labels: list[str],
    observations: list[list[float]],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    ylabel: str = "",
    jitter: float = 0.12,
) -> None:
    """Every observation as a point, with the group's median drawn across them (§24).

    A **sibling** of :func:`grouped_dots`, not a mode of it. That one takes a single value and an
    optional error bar per group, which was the right shape for the removed benchmark leaderboard
    and is the wrong one here: at the cohort sizes this project works at — three animals a group,
    sometimes two — a centre and a spread hide the entire dataset, and the points *are* the result.
    §18's benchmark exporters (since removed) depended on ``grouped_dots`` keeping its signature, so
    this was added beside it.

    Points are spread horizontally by a small deterministic offset rather than a random jitter, so
    a re-render is byte-identical (§18.5 L1) and two points never land exactly on top of each other
    and read as one. The median is ink, not the group's colour, so it stays visible across the
    points it summarises.
    """
    centers = np.arange(len(group_labels), dtype=float)
    for index, (center, values) in enumerate(zip(centers, observations)):
        if not values:
            continue
        values = np.asarray(values, dtype=float)
        _points(
            ax, center + _spread(len(values), jitter), values, style.color(index), style
        )
        _summary_tick(ax, center, float(np.median(values)), 0.26, style)
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.5, len(group_labels) - 0.5)
    _label_axes(ax, ylabel=ylabel)


def group_distribution(
    ax,
    group_labels: list[str],
    observations: list[list[float]],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    ylabel: str = "",
) -> str:
    """A distribution per group: a violin with its points when there are enough, else the points.

    Returns which form it drew — ``"violin"`` or ``"points"`` — so a caller can say so. The choice
    is made for the whole chart, not per group: a violin beside a column of three dots invites a
    comparison of shapes when one of them has no shape.

    A violin is a kernel density estimate, and it is honest only when there is a distribution to
    estimate. Below :data:`VIOLIN_MIN_N` in any group the chart is :func:`observation_dots`
    instead. Above it, the body is the group's colour as a wash, the IQR is a thick ink bar, the
    median a surface-coloured dot on it, and the observations ride on top until there are too many
    to count (:data:`POINTS_OVER_VIOLIN_MAX`).
    """
    sizes = [len(v) for v in observations if len(v)]
    if not sizes or min(sizes) < VIOLIN_MIN_N:
        observation_dots(ax, group_labels, observations, style=style, ylabel=ylabel)
        return "points"
    marks = style.marks
    centers = np.arange(len(group_labels), dtype=float)
    for index, (center, values) in enumerate(zip(centers, observations)):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2 or np.ptp(values) == 0:
            _points(ax, [center] * len(values), values, style.color(index), style)
            continue
        color = style.color(index)
        parts = ax.violinplot(
            [values],
            positions=[center],
            widths=0.7,
            showextrema=False,
            showmedians=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(marks.fill_alpha)
            body.set_edgecolor(color)
            body.set_linewidth(marks.thin_line_width * 0.8)
            body.set_zorder(1)
        if len(values) <= POINTS_OVER_VIOLIN_MAX:
            _points(
                ax,
                center + _spread(len(values), 0.08),
                values,
                color,
                style,
                size_scale=0.55,
            )
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.plot(
            [center, center],
            [q1, q3],
            "-",
            color=style.chart.summary,
            lw=marks.summary_width * 2.5,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.plot(
            [center],
            [median],
            "o",
            color=style.chart.surface,
            markersize=marks.marker_size * 0.8,
            markeredgecolor=style.chart.summary,
            markeredgewidth=marks.marker_ring,
            zorder=5,
        )
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.6, len(group_labels) - 0.4)
    _label_axes(ax, ylabel=ylabel)
    return "violin"


def group_curves(
    ax,
    x_labels: list[str],
    series: list[tuple[str, list[float | None]]],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """One line per group across an ordered axis — a measure becoming a curve (§24).

    The sibling of :func:`observation_dots` for the case where a measurement was taken in more
    than one window. Without it, §24 drew one dot plot *per window*: a 30-minute recording with a
    5-minute sliding window produced 540 two-point figures where the answer is a handful of
    curves, and "did it change over the session?" — the question windows exist to ask — could not
    be read off any of them.

    ``series`` is ``[(group_label, [value per x])]``; a ``None`` is a gap and is not drawn
    across, because joining over a window where the behavior never occurred would invent a
    measurement.
    """
    positions = list(range(len(x_labels)))
    for index, (label, values) in enumerate(series):
        xs = [x for x, v in zip(positions, values) if v is not None]
        ys = [v for v in values if v is not None]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=style.marks.marker_size * 0.8,
            color=style.color(index),
            label=label,
        )
    ax.set_xticks(positions)
    # Every window id on a 30-window axis is unreadable; thin them and keep the ends, which are
    # the two a reader looks for.
    step = max(1, len(x_labels) // 8)
    ax.set_xticklabels(
        [
            name if (i % step == 0 or i == len(x_labels) - 1) else ""
            for i, name in enumerate(x_labels)
        ],
        rotation=45 if len(x_labels) > 6 else 0,
        ha="right" if len(x_labels) > 6 else "center",
    )
    _label_axes(ax, xlabel, ylabel)
    if len(series) > 1:
        place_legend(ax, style=style)


def paired_lines(
    ax,
    group_labels: list[str],
    pairs: list[tuple[str, list[float | None]]],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    ylabel: str = "",
) -> None:
    """One line per subject across the groups — a crossover design drawn as one (§24).

    When the same animal appears in both groups, two independent clouds of dots throw away the
    pairing that makes the design powerful: a consistent within-animal shift can look like heavy
    overlap between groups. A line per subject shows the direction each one moved, which is the
    question the design was built to ask.

    ``pairs`` is ``[(subject_label, [value per group])]``. A subject missing from a group is a
    gap and is not joined across — a line drawn through a group the animal was never in would
    invent a measurement.
    """
    positions = list(range(len(group_labels)))
    for index, (label, values) in enumerate(pairs):
        xs = [x for x, v in zip(positions, values) if v is not None]
        ys = [v for v in values if v is not None]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            lw=style.marks.thin_line_width,
            color=style.color(index),
            label=label,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.35, len(group_labels) - 0.65)
    _label_axes(ax, ylabel=ylabel)
    # A legend per subject is unreadable past a handful; the lines carry the message either way.
    if 1 < len(pairs) <= 8:
        place_legend(ax, style=style)


def ranked_bars_with_floor(
    ax,
    labels: list[str],
    values: list[float],
    floor: float | None,
    *,
    style: FigureStyle = DEFAULT_STYLE,
    xlabel: str = "",
) -> None:
    """Horizontal bars ranked by magnitude, with the null floor drawn across them (§24).

    ``floor`` is what the same ranking procedure yields on shuffled group labels. Drawing it is not
    decoration: with tens of measures and a handful of animals, ranking by difference *always*
    produces a top hit whether or not anything is there, and this screen runs on every comparison.
    A bar that does not cross the line has not been shown to differ from chance.

    One colour for every bar: the bars are one series, and a hue per rank spent the identity
    channel saying nothing. ``None`` is drawn as an explicit annotation rather than omitted, so a
    figure with no floor says so instead of looking like a figure whose bars all cleared it.
    """
    positions = np.arange(len(labels), dtype=float)
    ax.barh(
        positions,
        values,
        height=style.marks.bar_fraction,
        color=style.color(0),
        edgecolor=style.chart.surface,
        linewidth=0,
        zorder=2,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(left=min(0.0, ax.get_xlim()[0]))
    if floor is None:
        ax.annotate(
            "no floor computed",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            fontsize=style.type.annotation,
            color=style.chart.text_muted,
        )
    else:
        reference_line(ax, floor, axis="x", label="chance", style=style)
    _label_axes(ax, xlabel=xlabel)


def bars(
    ax,
    labels: list[str],
    values: list[float],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    horizontal: bool = False,
    colors: list[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """One bar per category, grown from a zero baseline.

    One series is one colour (slot 0). ``colors`` is for the case where each bar *is* an identity
    that appears elsewhere in the same export — a behavior's bar in the colour of its lane in the
    ethogram — never for colouring bars by their rank.
    """
    positions = np.arange(len(labels), dtype=float)
    fill = colors or style.color(0)
    common = dict(color=fill, edgecolor=style.chart.surface, linewidth=0, zorder=2)
    if horizontal:
        ax.barh(positions, values, height=style.marks.bar_fraction, **common)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlim(left=0.0)
    else:
        ax.bar(positions, values, width=style.marks.bar_fraction, **common)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(bottom=0.0)
    _label_axes(ax, xlabel, ylabel)


# --------------------------------------------------------------------------------------------- #
# Curves, distributions, matrices, scatter
# --------------------------------------------------------------------------------------------- #


def overlaid_step(
    ax,
    curves: list[tuple[str, list[float], list[float]]],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = True,
) -> None:
    """Overlay step curves (e.g. precision-recall), one per series, identity-coloured.

    ``curves`` is ``[(label, xs, ys), ...]``. Set ``legend=False`` to draw a single shared legend
    at the figure level instead of one per axes.
    """
    for i, (label, xs, ys) in enumerate(curves):
        ax.step(xs, ys, where="post", color=style.color(i), label=label)
    _label_axes(ax, xlabel, ylabel)
    if legend and len(curves) > 1:
        place_legend(ax, style=style)


def line_with_band(
    ax,
    x,
    y,
    *,
    lower=None,
    upper=None,
    label: str = "",
    index: int = 0,
    style: FigureStyle = DEFAULT_STYLE,
) -> None:
    """A curve in identity colour ``index``, with its uncertainty as a wash of the same hue.

    The band is a fill at the theme's fill opacity with no outline — a band drawn with edges reads
    as two more curves.
    """
    color = style.color(index)
    if lower is not None and upper is not None:
        ax.fill_between(
            x, lower, upper, color=color, alpha=style.marks.fill_alpha, lw=0, zorder=1
        )
    ax.plot(x, y, color=color, label=label or None, zorder=2)


def histogram(
    ax,
    values,
    *,
    bins=30,
    style: FigureStyle = DEFAULT_STYLE,
    index: int = 0,
    xlabel: str = "",
    ylabel: str = "count",
    range_=None,
) -> None:
    """A histogram in one identity colour, bins separated by a hairline of the surface."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(
        values,
        bins=bins,
        range=range_,
        color=style.color(index),
        edgecolor=style.chart.surface,
        linewidth=0.4,
        zorder=2,
    )
    ax.set_ylim(bottom=0.0)
    _label_axes(ax, xlabel, ylabel)


def heatmap(
    ax,
    grid,
    *,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    style: FigureStyle = DEFAULT_STYLE,
    diverging: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "",
    annotate: bool = False,
    fmt: str = "{:.2f}",
):
    """A matrix as colour: the one-hue sequential scale, or the diverging one centred on zero.

    ``annotate`` writes each cell's value in whichever ink clears the cell's own colour, never in a
    series colour. Returns the colorbar so a caller can adjust its ticks.
    """
    grid = np.asarray(grid, dtype=float)
    if diverging:
        bound = float(np.nanmax(np.abs(grid))) if grid.size else 1.0
        vmin, vmax = (-bound, bound) if vmin is None and vmax is None else (vmin, vmax)
    cmap = diverging_cmap(style) if diverging else sequential_cmap(style)
    image = ax.imshow(
        grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        # Level; `save_figure` turns them only if they actually collide.
        ax.set_xticklabels(col_labels)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=style.marks.tick_length * 0.6)
    if colorbar_label:
        bar.set_label(colorbar_label)
    if annotate:
        from detail.core.color_checks import relative_luminance

        normed = image.norm
        for (row, col), value in np.ndenumerate(grid):
            if not np.isfinite(value):
                continue
            rgba = cmap(normed(value))
            cell = "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in rgba[:3]))
            dark_cell = relative_luminance(cell) < 0.35
            ink = "#ffffff" if dark_cell else "#1a1a1a"
            ax.text(
                col,
                row,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=style.type.annotation,
                color=ink,
            )
    return bar


def scatter_groups(
    ax,
    groups: list[tuple[str, list[float], list[float]]],
    *,
    style: FigureStyle = DEFAULT_STYLE,
    xlabel: str = "",
    ylabel: str = "",
    neutral_first: bool = False,
) -> None:
    """Points, one identity colour per group, each with a surface ring so overlaps separate.

    Scatter can put any two groups side by side, so only the first three identity colours pass the
    all-pairs check; past three groups the colours still differ but a reader needs the legend.
    ``neutral_first`` draws the first group in the theme's neutral — the "everything else" cloud a
    selection is shown against. A single group past :data:`SCATTER_DENSITY_ABOVE` points is drawn
    as a hexbin density in the sequential scale.
    """
    marks = style.marks
    total = sum(len(xs) for _, xs, _ in groups)
    if len(groups) == 1 and total > SCATTER_DENSITY_ABOVE:
        _, xs, ys = groups[0]
        ax.hexbin(xs, ys, gridsize=60, cmap=sequential_cmap(style), mincnt=1, lw=0)
        _label_axes(ax, xlabel, ylabel)
        return
    size = (marks.marker_size * (0.6 if total > 500 else 1.0)) ** 2
    for index, (label, xs, ys) in enumerate(groups):
        if neutral_first and index == 0:
            color = style.chart.neutral
        else:
            color = style.color(index - 1 if neutral_first else index)
        ax.scatter(
            xs,
            ys,
            s=size,
            color=color,
            edgecolors=style.chart.surface,
            linewidths=marks.marker_ring * (0.5 if total > 500 else 1.0),
            alpha=marks.point_alpha,
            label=label,
            zorder=2 if (neutral_first and index == 0) else 3,
        )
    _label_axes(ax, xlabel, ylabel)
    if len(groups) > 1:
        place_legend(ax, style=style)


# --------------------------------------------------------------------------------------------- #
# Timelines
# --------------------------------------------------------------------------------------------- #


def lanes(
    ax,
    rows: list[tuple[str, list[tuple[float, float, str]]]],
    colors: dict[str, str],
    *,
    extents: dict[str, tuple[float, float]] | None = None,
    style: FigureStyle = DEFAULT_STYLE,
    xlabel: str = "time (s)",
    legend_max: int = 12,
) -> None:
    """An ethogram: one lane per row, each span a block coloured by its category.

    ``rows`` is ``[(lane_label, [(start, end, category), ...])]`` drawn top to bottom in the order
    given. ``extents`` gives a lane's full observed range, drawn as a neutral track under its
    spans, so a recording with few bouts still reads as a recording and not a short one. Lanes are
    separated by air, not by rules; the legend lists the categories once, capped at
    ``legend_max`` so a sixty-cluster vocabulary does not draw a legend taller than the figure.
    """
    from matplotlib.patches import Patch

    height = 0.62
    for index, (label, spans) in enumerate(rows):
        y = len(rows) - 1 - index
        extent = (extents or {}).get(label)
        if extent is not None:
            ax.broken_barh(
                [(extent[0], extent[1] - extent[0])],
                (y - height / 2, height),
                facecolors=style.chart.grid,
                edgecolor="none",
                zorder=1,
            )
        by_category: dict[str, list[tuple[float, float]]] = {}
        for start, end, category in spans:
            by_category.setdefault(category, []).append((start, end - start))
        for category in sorted(by_category):
            ax.broken_barh(
                by_category[category],
                (y - height / 2, height),
                facecolors=colors[category],
                edgecolor="none",
                zorder=2,
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _label_axes(ax, xlabel=xlabel)
    if colors and len(colors) <= legend_max:
        handles = [
            Patch(facecolor=color, edgecolor="none", label=name)
            for name, color in sorted(colors.items())
        ]
        place_legend(ax, style=style, handles=handles)
