"""The matplotlib adapter for the chart design language (module-agnostic).

Every value here comes from ``core/chart_theme.py`` — colours, font sizes, line weights, mark
sizes — so an exported figure and the same chart on screen are one design. This module only
translates them into matplotlib's vocabulary: rcParams, colormaps, and the few per-call values a
primitive needs (``FigureStyle.type`` / ``.marks``).

Export figures carry no embedded title (the caption lives in the LaTeX/Markdown around them),
render vector-first, and are **light unless asked otherwise** — a figure is headed for paper. A
``dark`` style is the same design on the dark surface, for slides.
"""

import functools
from dataclasses import dataclass, replace

from detail.core import chart_theme
from detail.core.chart_theme import ChartTheme, Marks, Type


@dataclass(frozen=True)
class FigureStyle:
    """How an export figure looks. ``show_title`` is off for publication (caption lives outside).

    ``base_fontsize`` is the axis-label size in points; every other size scales with it, so a
    caller who wants a larger figure for a poster changes one number.
    """

    theme: str = "light"
    base_fontsize: float = chart_theme.PRINT_TYPE.label
    dpi: int = 300
    single_col_width_in: float = chart_theme.SINGLE_COLUMN_IN
    double_col_width_in: float = chart_theme.DOUBLE_COLUMN_IN
    # Full-video timeline figures (matches / similarity) render wide so a whole ~30-min recording
    # can be zoomed to per-event detail; at `dpi` this is ~12000 px across.
    timeline_width_in: float = 40.0
    show_title: bool = False

    def __post_init__(self) -> None:
        chart_theme.theme(self.theme)  # refuse an unknown theme at construction

    @property
    def chart(self) -> ChartTheme:
        """The theme's colours."""
        return chart_theme.theme(self.theme)

    @property
    def palette(self) -> tuple[str, ...]:
        """Identity colours in order: the six validated slots, then the frozen extension."""
        return self.chart.categorical + self.chart.extended

    @property
    def type(self) -> Type:
        """Font sizes, scaled so ``label`` is ``base_fontsize``. Never below the floor."""
        base = chart_theme.PRINT_TYPE
        scale = self.base_fontsize / base.label
        return replace(
            base,
            tick=max(base.floor, base.tick * scale),
            label=max(base.floor, base.label * scale),
            annotation=max(base.floor, base.annotation * scale),
            panel_letter=max(base.floor, base.panel_letter * scale),
        )

    @property
    def marks(self) -> Marks:
        """Line weights and mark sizes, in points."""
        return chart_theme.PRINT_MARKS

    def color(self, index: int) -> str:
        """The identity colour for a series index (see ``ChartTheme.identity``)."""
        return self.chart.identity(index)

    def with_theme(self, theme: str) -> "FigureStyle":
        """This style on another surface — the one knob an export's ``theme`` option turns."""
        return replace(self, theme=theme)


DEFAULT_STYLE = FigureStyle()

#: The rcParams matplotlib reads when a figure is **saved** rather than when it is built, and
#: which do not depend on the theme. ``figures.save_figure`` applies them itself, because a
#: figure is often saved after the ``styled`` block that built it has closed — every behavior
#: export was, and wrote Type-3 fonts (uneditable text) and SVGs with random element ids.
SAVE_PARAMS: dict = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "detail",  # stable element ids -> reproducible SVG output
}


def save_params() -> dict:
    """:data:`SAVE_PARAMS` plus the math-text faces, which matplotlib reads when a figure is
    drawn. An italic *P* in an annotation is math text, and without these it is set in
    Computer Modern beside the house face; with them it is the house face in italic, so a
    rewritten SVG (``svg_edit``) is one text object in one family."""
    family = installed_families()[0]
    return {
        **SAVE_PARAMS,
        "mathtext.default": "regular",
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
    }


@functools.cache
def installed_families() -> tuple[str, ...]:
    """The house faces this machine has, in preference order; DejaVu Sans when it has none.

    Filtered once rather than handed to matplotlib whole, because matplotlib logs a warning for
    every family in the list it cannot find — on every figure — and a CLI export printed
    "findfont: Font family 'Liberation Sans' not found" beside its progress bar. DejaVu Sans ships
    with matplotlib, so the list is never empty.
    """
    from matplotlib import font_manager

    names = {entry.name for entry in font_manager.fontManager.ttflist}
    found = tuple(f for f in chart_theme.PRINT_TYPE.family if f in names)
    return found or ("DejaVu Sans",)


def rc_params(style: FigureStyle = DEFAULT_STYLE) -> dict:
    """matplotlib rcParams for a figure in this style.

    Embeds editable text in vector output (``pdf/ps fonttype 42``, ``svg.fonttype none``) so a
    figure's labels stay selectable/editable in a vector editor. Constrained layout is on for
    every figure: without it a y label ran into the neighbouring panel and a legend sat on the
    data (measured on the primitives before this module existed).
    """
    from cycler import cycler

    colors = style.chart
    sizes = style.type
    marks = style.marks
    return {
        # Output
        "figure.dpi": style.dpi,
        "savefig.dpi": style.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.figsize": (style.single_col_width_in, 2.6),
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        **save_params(),
        # Surfaces and ink
        "figure.facecolor": colors.surface,
        "savefig.facecolor": colors.surface,
        "axes.facecolor": colors.surface,
        "axes.edgecolor": colors.axis,
        "axes.labelcolor": colors.text,
        "axes.titlecolor": colors.text,
        "text.color": colors.text,
        "xtick.color": colors.axis,
        "ytick.color": colors.axis,
        "xtick.labelcolor": colors.text_secondary,
        "ytick.labelcolor": colors.text_secondary,
        "legend.labelcolor": colors.text_secondary,
        "grid.color": colors.grid,
        "axes.prop_cycle": cycler(color=list(colors.categorical)),
        "patch.edgecolor": colors.surface,
        "patch.force_edgecolor": False,
        "scatter.edgecolors": colors.surface,
        # Type. The concrete family list, not the generic "sans-serif": a text object stores its
        # family when it is made but resolves a *generic* one when it is drawn, so a figure saved
        # after its style context closed fell back to DejaVu — every exported ethogram did.
        "font.family": list(installed_families()),
        "font.sans-serif": list(installed_families()),
        "font.size": sizes.tick,
        "axes.titlesize": sizes.label,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": sizes.label,
        "xtick.labelsize": sizes.tick,
        "ytick.labelsize": sizes.tick,
        "legend.fontsize": sizes.tick,
        "legend.title_fontsize": sizes.tick,
        "figure.titlesize": sizes.label,
        # Axes: left and bottom only, ticks out, no grid unless a primitive asks for one
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": marks.axis_width,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.labelpad": 3.0,
        "axes.titlepad": 5.0,
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.05,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": marks.tick_length,
        "ytick.major.size": marks.tick_length,
        "xtick.major.width": marks.tick_width,
        "ytick.major.width": marks.tick_width,
        "xtick.minor.size": marks.tick_length * 0.6,
        "ytick.minor.size": marks.tick_length * 0.6,
        "xtick.minor.width": marks.tick_width * 0.75,
        "ytick.minor.width": marks.tick_width * 0.75,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        # Marks
        "lines.linewidth": marks.line_width,
        "lines.markersize": marks.marker_size,
        "lines.markeredgewidth": marks.marker_ring,
        "lines.markeredgecolor": colors.surface,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "butt",
        "errorbar.capsize": 0,
        # Legend: frameless, compact
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.35,
        "legend.columnspacing": 1.2,
    }


def sequential_cmap(style: FigureStyle = DEFAULT_STYLE):
    """A magnitude colormap from the theme's one-hue ramp (low end near the surface)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"detail_sequential_{style.theme}", list(style.chart.sequential)
    )


def diverging_cmap(style: FigureStyle = DEFAULT_STYLE):
    """A signed-difference colormap, blue ↔ grey ↔ vermillion. Centre it on the neutral value."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"detail_diverging_{style.theme}", list(style.chart.diverging)
    )
