"""Styled figure primitives build, save in every format, and wear the design language.

**The classes these guard** (``docs/recorded_classes.md``) are invisible to a test that checks a
figure came back, which is what this file used to be:

- **A figure saved after its style closed loses it.** matplotlib reads fonts, ``pdf.fonttype`` and
  ``svg.hashsalt`` when it *saves*, not when it builds. The behavior exports are built in one
  place and saved in another, outside the ``styled`` block, so every ethogram written before
  ``save_figure`` carried its own settings had Type-3 (uneditable) PDF text, a fallback face, and
  SVG element ids that changed on every save — the last of which also broke §18.5's byte-identical
  re-export for those files. ``test_a_figure_saved_after_its_style_closed_keeps_it`` saves exactly
  that way, in both themes.
- **Text in a series colour, and text too small to print.** A label that wears its series' hue is
  illegible for the light slots and repeats what the mark beside it already says; a size computed
  as ``base_fontsize - 3`` landed at 6pt. Every primitive is asked, in both themes, for ink-only
  text and the print floor (``core/chart_theme.PRINT_TYPE``).
- **Category labels run together.** Group names are as long as users make them; the fixture here
  is proved colliding before the save-time turn is asked to clear it, by the geometry of rotated
  text rather than by overlapping boxes, which rotated text always has.

Each check has a negative case — text in a series colour, a Type-3 save, a colliding fixture — so
a check that stops seeing its failure fails too. The palettes themselves are
``tests/core/test_chart_theme.py``; a plot on screen is ``tests/gui/test_plot_theme.py``.
"""

from detail.report import figures
from detail.report.style import DEFAULT_STYLE, FigureStyle, rc_params


def test_rc_params_embeds_editable_vector_text():
    rc = rc_params(FigureStyle(dpi=200))
    assert rc["savefig.dpi"] == 200
    assert rc["pdf.fonttype"] == 42  # editable text in vector output
    assert rc["svg.fonttype"] == "none"
    assert rc["axes.spines.top"] is False


def test_identity_colours_come_from_the_theme_and_wrap_only_past_the_extension():
    style = FigureStyle()
    assert style.color(0) == style.palette[0] == style.chart.categorical[0]
    assert (
        style.color(6) == style.chart.extended[0]
    )  # the seventh is not the first again
    assert (
        style.color(len(style.palette)) == style.palette[0]
    )  # past 24, a documented wrap


def test_small_multiples_shape():
    fig, axes = figures.small_multiples(3, width=7.0, height_per=1.2)
    assert len(axes) == 3
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_save_figure_writes_each_format(tmp_path):
    fig, axes = figures.small_multiples(1, width=4.0, height_per=1.5)
    axes[0].axvspan(1, 3)
    paths = figures.save_figure(fig, tmp_path / "fig", formats=("pdf", "svg", "png"))
    assert {p.suffix for p in paths} == {".pdf", ".svg", ".png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_grouped_dots_with_errorbars_builds():
    fig, axes = figures.small_multiples(1, width=5.0, height_per=3.0)
    figures.grouped_dots(
        axes[0],
        ["climb", "groom"],
        [
            ("silhouette", [0.42, 0.31], [0.03, 0.04]),
            ("context_crop", [0.55, 0.29], [0.04, 0.05]),
        ],
        ylabel="AP",
    )
    import matplotlib.pyplot as plt

    assert axes[0].get_ylabel() == "AP"
    plt.close(fig)


def test_overlaid_step_builds_pr_style():
    fig, axes = figures.small_multiples(1, width=4.0, height_per=3.0)
    figures.overlaid_step(
        axes[0],
        [("silhouette", [0.0, 0.5, 1.0], [1.0, 0.6, 0.4])],
        xlabel="Recall",
        ylabel="Precision",
    )
    import matplotlib.pyplot as plt

    assert axes[0].get_xlabel() == "Recall"
    plt.close(fig)


def test_styled_context_applies_font_size():
    import matplotlib.pyplot as plt

    with figures.styled(FigureStyle(base_fontsize=13.0)):
        # `base_fontsize` is the axis-label size; ticks and legend scale with it.
        assert plt.rcParams["axes.labelsize"] == 13.0
        assert plt.rcParams["xtick.labelsize"] == 7.0 * 13.0 / 8.0


def test_grid_multiples_returns_exactly_n_axes_and_hides_the_rest():
    import matplotlib.pyplot as plt

    fig, axes = figures.grid_multiples(3, cols=2, width=7.0, height_per=2.0)
    assert len(axes) == 3  # 2x2 grid, one cell over-allocated
    assert all(ax.get_visible() for ax in axes)
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(hidden) == 1
    plt.close(fig)


def test_grid_multiples_rejects_a_degenerate_grid():
    import pytest

    with pytest.raises(ValueError, match="must be >= 1"):
        figures.grid_multiples(0, cols=2, width=7.0, height_per=2.0)
    with pytest.raises(ValueError, match="must be >= 1"):
        figures.grid_multiples(4, cols=0, width=7.0, height_per=2.0)


# --------------------------------------------------------------------------------------------- #
# The design language, rendered — asked of every primitive, in both themes.
#
# What these guard is invisible to a test that checks a figure came back: text drawn in a series
# colour, a label that collides with its neighbour, a size below what a journal will print, a
# figure that loses its fonts because it was saved after its style closed. Each was found in the
# figures this package wrote before the design language existed.
# --------------------------------------------------------------------------------------------- #

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from detail.core import chart_theme  # noqa: E402

_THEMES = pytest.mark.parametrize("theme", ["light", "dark"])
_GROUPS = ["Saline", "Drug A", "Drug A+Antagonist"]


def _draw_primitives(style):
    """``name -> draw(ax)`` for every primitive, on small deterministic data."""
    rng = np.random.default_rng(0)
    return {
        "observation_dots": lambda ax: figures.observation_dots(
            ax, _GROUPS, [[1.0, 2.0, 3.0], [2.0, 3.5], [1.5, 2.5, 2.0]], style=style
        ),
        "group_distribution": lambda ax: figures.group_distribution(
            ax, _GROUPS, [list(rng.gamma(3, 1, 40)) for _ in _GROUPS], style=style
        ),
        "grouped_bars": lambda ax: figures.grouped_dots(
            ax,
            ["a", "b"],
            [("x", [1, 2], [0.1, 0.2]), ("y", [2, 1], None)],
            style=style,
            as_bars=True,
        ),
        "group_curves": lambda ax: figures.group_curves(
            ax,
            ["w0", "w1", "w2"],
            [("g1", [1, 2, 3]), ("g2", [2, None, 1])],
            style=style,
        ),
        "paired_lines": lambda ax: figures.paired_lines(
            ax, ["pre", "post"], [("m1", [1, 2]), ("m2", [2, 1])], style=style
        ),
        "ranked_bars_with_floor": lambda ax: figures.ranked_bars_with_floor(
            ax, ["speed", "rearing"], [0.9, 0.3], 0.5, style=style
        ),
        "overlaid_step": lambda ax: figures.overlaid_step(
            ax, [("a", [0, 1], [1, 0]), ("b", [0, 1], [0.8, 0.1])], style=style
        ),
        "line_with_band": lambda ax: figures.line_with_band(
            ax, [0, 1, 2], [1, 2, 3], lower=[0, 1, 2], upper=[2, 3, 4], style=style
        ),
        "histogram": lambda ax: figures.histogram(
            ax, rng.normal(0, 1, 200), style=style
        ),
        "heatmap": lambda ax: figures.heatmap(
            ax, rng.random((3, 3)), style=style, annotate=True
        ),
        "scatter_groups": lambda ax: figures.scatter_groups(
            ax,
            [("all", [0, 1, 2], [0, 1, 0]), ("sel", [1], [1])],
            style=style,
            neutral_first=True,
        ),
        "lanes": lambda ax: figures.lanes(
            ax,
            [("r1", [(0.0, 2.0, "rear")]), ("r2", [(1.0, 3.0, "groom")])],
            {"rear": style.color(0), "groom": style.color(1)},
            extents={"r1": (0.0, 5.0), "r2": (0.0, 5.0)},
            style=style,
        ),
    }


def _free_texts(fig) -> list[Text]:
    """Every visible, non-empty text object the figure will draw."""
    fig.draw_without_rendering()
    return [
        t
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and t.axes is not None
    ]


@_THEMES
@pytest.mark.parametrize("primitive", sorted(_draw_primitives(DEFAULT_STYLE)))
def test_every_primitive_wears_the_theme_and_prints_at_a_readable_size(
    theme, primitive
):
    """Text wears ink, never a series colour; nothing drops below the print floor."""
    from matplotlib.colors import to_hex

    style = FigureStyle(theme=theme)
    with figures.styled(style):
        fig, ax = plt.subplots()
        _draw_primitives(style)[primitive](ax)
    try:
        assert to_hex(fig.get_facecolor()) == style.chart.surface
        series = {c.lower() for c in style.palette}
        for text in _free_texts(fig):
            colour = to_hex(text.get_color()).lower()
            assert colour not in series, (
                f"{primitive}: {text.get_text()!r} is drawn in a series colour ({colour}); "
                "text wears ink and the mark beside it carries the identity"
            )
            assert text.get_fontsize() >= chart_theme.PRINT_TYPE.floor - 1e-6, (
                f"{primitive}: {text.get_text()!r} at {text.get_fontsize()}pt is below the "
                f"{chart_theme.PRINT_TYPE.floor}pt print floor"
            )
    finally:
        plt.close(fig)


def test_the_ink_check_would_catch_text_in_a_series_colour():
    from matplotlib.colors import to_hex

    style = FigureStyle()
    with figures.styled(style):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "rearing", color=style.color(0))
    try:
        series = {c.lower() for c in style.palette}
        assert any(to_hex(t.get_color()).lower() in series for t in _free_texts(fig))
    finally:
        plt.close(fig)


def _level_boxes(ax):
    renderer = ax.figure.canvas.get_renderer()
    return sorted(
        (
            t.get_window_extent(renderer)
            for t in ax.get_xticklabels()
            if t.get_visible() and t.get_text()
        ),
        key=lambda b: b.x0,
    )


def test_long_category_labels_are_turned_rather_than_run_together():
    """Measured on the rendered labels: "never overlap", not "rotate past N characters".

    Turned labels are parallel lines of text, so they are asked the geometric question — is the
    perpendicular distance between neighbours at least a line's height — rather than whether their
    axis-aligned boxes touch, which rotated text always does and which says nothing about glyphs.
    """
    import math

    style = FigureStyle()
    labels = ["Saline control", "Drug A, high dose", "Drug A + antagonist", "Vehicle"]
    with figures.styled(style):
        fig, ax = plt.subplots(figsize=(style.single_col_width_in, 2.4))
        figures.observation_dots(ax, labels, [[1.0, 2.0]] * 4, style=style)
    fig.draw_without_rendering()
    level = _level_boxes(ax)
    assert any(
        a.overlaps(b) for a, b in zip(level, level[1:])
    ), "the fixture must start out colliding, or proves nothing"
    figures._turn_colliding_tick_labels(fig)
    angle = ax.get_xticklabels()[0].get_rotation()
    centres = [(b.x0 + b.x1) / 2 for b in level]
    spacing = min(b - a for a, b in zip(centres, centres[1:]))
    height = max(b.height for b in level)
    apart = spacing * math.sin(math.radians(angle))
    assert apart >= height, (
        f"turned to {angle} degrees, neighbours are {apart:.0f}px apart for {height:.0f}px of "
        "text"
    )
    assert angle == figures.turn_angle(level, 2.0 * fig.dpi / 72.0)
    plt.close(fig)


def test_the_turn_is_the_smallest_that_clears():
    """A row that clears at 30 degrees is not stood on its end."""
    from matplotlib.transforms import Bbox

    def row(spacing, height, n=3):
        return [
            Bbox.from_bounds(i * spacing, 0, spacing * 1.5, height) for i in range(n)
        ]

    assert figures.turn_angle(row(100, 40)) == 30  # 100 x sin 30 = 50 >= 40
    assert figures.turn_angle(row(60, 40)) == 45  # 30 < 40 <= 42
    assert figures.turn_angle(row(20, 40)) == 90  # nothing clears; the steepest is used


def test_short_labels_are_left_level():
    style = FigureStyle()
    with figures.styled(style):
        fig, ax = plt.subplots()
        figures.observation_dots(ax, ["a", "b"], [[1.0], [2.0]], style=style)
    figures._turn_colliding_tick_labels(fig)
    assert all(t.get_rotation() == 0 for t in ax.get_xticklabels())
    plt.close(fig)


def test_a_violin_needs_a_distribution_to_estimate():
    """Below VIOLIN_MIN_N in any group the points are the result, and a violin would invent one."""
    style = FigureStyle()
    few = [[1.0, 2.0, 3.0]] * 2
    many = [list(np.linspace(0, 1, figures.VIOLIN_MIN_N))] * 2
    for observations, form in ((few, "points"), (many, "violin")):
        with figures.styled(style):
            fig, ax = plt.subplots()
            drawn = figures.group_distribution(
                ax, ["a", "b"], observations, style=style
            )
        assert drawn == form
        plt.close(fig)
    with figures.styled(style):
        fig, ax = plt.subplots()
        drawn = figures.group_distribution(
            ax, ["a", "b"], [few[0], many[0]], style=style
        )
    assert (
        drawn == "points"
    ), "one group too small to estimate draws the whole chart as points"
    plt.close(fig)


@_THEMES
def test_a_figure_saved_after_its_style_closed_keeps_it(tmp_path, theme):
    """The behavior exports build a figure in one place and save it in another — outside the
    ``styled`` block — and wrote Type-3 fonts, a fallback face and SVGs whose element ids changed
    on every save. The save now carries its own settings; this saves exactly that way.
    """
    from matplotlib import font_manager

    from detail.report import behavior_figures as bf
    from detail.storage import behavior_output as bo

    ns = 1_000_000_000
    rows = [bo.BehaviorInstance("r1", 0, 1 * ns, 2 * ns, 30, 60, "c00", "rearing")]
    coverage = [bo.CoverageRow("r1", 0, 0, 60 * ns, 60 * ns, 1800)]
    style = FigureStyle(theme=theme)
    for run in ("one", "two"):
        fig = bf.ethogram(rows, coverage, style=style)  # built inside its style...
        figures.save_figure(
            fig, tmp_path / run, formats=("pdf", "svg")
        )  # ...saved outside

    pdf = (tmp_path / "one.pdf").read_bytes()
    assert b"/Type3" not in pdf, "text was embedded as Type-3 glyphs — not editable"
    svg = (tmp_path / "one.svg").read_text(encoding="utf-8-sig")
    face = font_manager.FontProperties(family=list(chart_theme.PRINT_TYPE.family))
    expected = font_manager.get_font(font_manager.findfont(face)).family_name
    assert expected in svg, f"the house face {expected!r} is not in the saved SVG"
    assert (tmp_path / "one.svg").read_bytes() == (tmp_path / "two.svg").read_bytes()


def test_the_type3_check_would_catch_a_type3_save(tmp_path):
    with figures.styled():
        fig, ax = plt.subplots()
        ax.set_xlabel("time (s)")
    with plt.rc_context({"pdf.fonttype": 3}):
        fig.savefig(tmp_path / "t3.pdf")
    plt.close(fig)
    assert b"/Type3" in (tmp_path / "t3.pdf").read_bytes()


@_THEMES
def test_a_saved_figure_is_on_its_themes_surface(tmp_path, theme):
    import matplotlib.image as mpimg

    style = FigureStyle(theme=theme)
    with figures.styled(style):
        fig, ax = plt.subplots()
        figures.histogram(ax, [1, 2, 2, 3], style=style)
    (path,) = figures.save_figure(fig, tmp_path / "h", formats=("png",))
    corner = np.round(mpimg.imread(path)[0, 0, :3] * 255).astype(int)
    assert tuple(int(c) for c in corner) == chart_theme.rgb(style.chart.surface)


def test_an_unknown_theme_is_refused_when_the_style_is_made():
    with pytest.raises(ValueError, match="chart theme must be one of"):
        FigureStyle(theme="sepia")
    assert FigureStyle().with_theme("dark").chart is chart_theme.DARK
