"""The chart design language — every colour, weight and size a chart is drawn with.

One module, three adapters. Charts in this package are drawn three ways: matplotlib for exported
figures (``report/style.py``), pyqtgraph for live plots (``gui/plot_theme.py``) and QPainter for
the hand-drawn lanes and maps. Each used to carry its own idea of a chart — a white background
hard-coded at eleven sites, ``#444`` for a median in one file and ``(55, 62, 72)`` in another, and
two copies of Okabe-Ito **in different orders**, so the first cluster was orange on screen and
blue in its exported ethogram. The adapters now read the values here and nothing else, so a chart
on screen and the same chart on paper are one design.

Pure data: no Qt (Tenet 3), no matplotlib, no numpy. The reasoning behind each choice — the
measurements that picked the palette order, why the dark lightness band is wider than the
reference method's, why summaries are ink and never a series colour — is ``docs/chart_design.md``.

**Colour does one job at a time.** ``categorical`` is identity (which group, which behavior);
``sequential`` is magnitude; ``diverging`` is a signed difference from a neutral middle;
``neutral`` is "everything else" — the run-wide spread behind a selection, the unselected
clusters, the Other bucket. Status (ready / warning / error) is not a chart colour and stays in
``gui/status_color.py``. Text is never drawn in a series colour: a label wears ink, and the
coloured mark beside it carries the identity.

**Two themes, each validated on its own surface.** ``LIGHT`` is the publication default — every
export is light unless the caller asks for ``dark``. ``DARK`` is the same hues stepped for a dark
surface, not an automatic inversion; ``tests/core/test_chart_theme.py`` runs both through
``core/color_checks.py``.
"""

from dataclasses import dataclass

from detail.core.color_checks import hex_to_rgb

# --------------------------------------------------------------------------------------------- #
# Identity colours
# --------------------------------------------------------------------------------------------- #

#: Okabe-Ito minus black and yellow, in the order that passes every gate in both themes. Black is
#: the ink colour, so a series drawn in it reads as an axis or an annotation; yellow is 1.3:1 on
#: white and turns olive once it is stepped into the lightness band. The order is the CVD-safety
#: mechanism — neighbours are what lines and bars put side by side — and was chosen by search:
#: worst adjacent ΔE 11.4 (protan/deutan) and 16.4 (normal) in both themes, and the first three
#: pass all pairs, which is what a scatter needs. The order this replaced (blue, orange, green,
#: purple, sky, vermillion) measured 7.6, below the target of 8.
_OKABE_ITO_LIGHT: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange (amber)
    "#009E73",  # bluish green
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

#: What each categorical slot *is*, in slot order — for the charts whose categories carry a meaning
#: of their own (a true positive is green, a false positive vermillion) and must keep it however
#: the slots are ordered. Ask ``ChartTheme.hue(name)``; never index the palette for a meaning.
HUE_NAMES: tuple[str, ...] = ("blue", "orange", "green", "sky", "vermillion", "purple")

#: The same six on the dark surface. Canonical Okabe-Ito except blue: ``#0072B2`` is 2.9:1 on the
#: dark surface, and lifting it is bounded above by sky blue, which is the same hue separated only
#: by lightness. ``#1F81C2`` clears 3:1 and keeps the pair ΔE 15.5 apart.
_OKABE_ITO_DARK: tuple[str, ...] = (
    "#1F81C2",
    "#E69F00",
    "#009E73",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
)

#: Identities past the sixth — a clustering run names thirty behaviors, and each needs a colour
#: that stays put. Generated once by greedy max-min OKLab distance from the six above and from the
#: theme's neutrals, then frozen here, so the first six never move when a seventh arrives (the
#: HSV fallback this replaces recoloured *every* cluster the moment a run had seven). Worst pair
#: across all 24 is ΔE 9.3 in both themes: distinct at a glance, **not** colour-blind safe, so a
#: chart that reaches this list must name its marks some other way too (a label, a selection).
_EXTENDED_LIGHT: tuple[str, ...] = (
    "#6ecc49",
    "#8540bd",
    "#ba1f58",
    "#8c7a2b",
    "#9b8bfb",
    "#ea82f4",
    "#2bccb4",
    "#fc7c69",
    "#317d01",
    "#965a88",
    "#96a85e",
    "#c54cae",
    "#6872f2",
    "#48a515",
    "#99523f",
    "#c69fdf",
    "#bf884f",
    "#8079bc",
)
_EXTENDED_DARK: tuple[str, ...] = (
    "#6ecc49",
    "#9450ce",
    "#cc3466",
    "#8c7a2b",
    "#9b8bfb",
    "#ea82f4",
    "#2bccb4",
    "#fc7c69",
    "#348008",
    "#995d8b",
    "#96a85e",
    "#c54cae",
    "#6872f2",
    "#48a515",
    "#a65e4a",
    "#c69fdf",
    "#bf884f",
    "#8079bc",
)


#: The identity slots a behavior label never takes: those within :data:`LABEL_POLE_FLOOR`
#: (ΔE, normal vision) of either pole of the diverging scale in either theme. The same set in
#: both themes, because a label's slot is its colour in each. Blue, amber, sky and vermillion
#: of the six, and eight of the extension; twelve remain. Written out, and recomputed from the
#: palettes by ``tests/core/test_chart_theme.py``, so a palette edit that moves a colour next
#: to a pole fails there instead of quietly colouring a label "less".
LABEL_RESERVED_SLOTS: frozenset[int] = frozenset(
    {0, 1, 3, 4, 9, 10, 13, 16, 18, 21, 22, 23}
)
#: How far (ΔE, OKLab ×100, normal vision) a label's colour must stay from both poles.
LABEL_POLE_FLOOR = 15.0


@dataclass(frozen=True)
class ChartTheme:
    """Every colour a chart uses, for one surface.

    ``surface`` is what the plot area is filled with. In the GUI it is one step lighter than the
    window in both themes — white on Fusion's ``#f0f0f0``, ``#262626`` on its ``#1e1e1e`` — so a
    plot reads as a card on the page in either, and it is the surface every colour below was
    checked against.
    """

    name: str
    surface: str
    #: Titles and axis labels.
    text: str
    #: Tick labels and legend entries.
    text_secondary: str
    #: Annotations that should recede: ``n = 4``, "no floor computed".
    text_muted: str
    #: Spines and ticks.
    axis: str
    #: Gridlines, when a chart has them. Hairline and recessive; never dashed.
    grid: str
    #: Everything that is not the subject: the run-wide spread behind a selection, unselected
    #: clusters, the Other bucket.
    neutral: str
    #: A reference the data is read against — chance, a null floor, a cutoff. Dashed, so it is
    #: never mistaken for a series.
    reference: str
    #: Summaries drawn over the data — a median, a mean, an error bar. Ink, never a series
    #: colour: a summary in the group's own hue vanishes into the group's points.
    summary: str
    #: A control drawn on a plot — the cutoff a person drags. Not data and not a reference, so it
    #: has its own colour, the same on every plot that carries one; before this the pedigree's cut
    #: was orange and the calibration curve's cutoff pyqtgraph's default pale yellow.
    handle: str
    categorical: tuple[str, ...]
    extended: tuple[str, ...]
    #: Magnitude, low → high. The low end sits near the surface, so zero reads as "nothing".
    #: No hue, lightness alone: it was a blue ramp, and the tree it shades also carries the
    #: diverging scale's blue ("less") as a frame on the clusters a difference screen kept —
    #: a blue frame on a blue fill did not read, and blue meant two things (2026-11-16).
    sequential: tuple[str, ...]
    #: The location plots' colours, low → high: a path coloured by time or speed, an occupancy
    #: heat map. Inferno on dark (black to pale yellow), inferno reversed on light (pale
    #: yellow to black): hue *and* lightness move, monotonically, and the low end fades into
    #: the surface in both, so a cell an animal barely visited reads as almost nothing and a
    #: hotspot stands out. The grey ``sequential`` ramp read as black and white on a busy
    #: picture of paths. Viridis on light was tried: it put its strongest colour, a dark
    #: purple, on the least-visited cells and framed the arena's edges. An empty cell is
    #: masked, never drawn in the low end; a line skips the steps too close to the surface
    #: (``report/style.line_stops``). Not for a share where zero must read as nothing on the
    #: cluster tree: that stays ``sequential``.
    colormap: tuple[str, ...]
    #: Signed difference, negative pole → neutral middle → positive pole (blue ↔ yellow).
    #: The poles differ in lightness as well as hue: blue against yellow is the contrast a
    #: red-green colour-blind reader keeps, and the earlier blue ↔ vermillion pair sat at one
    #: lightness, so only its hue said which way (``tests/core/test_chart_theme.py``).
    diverging: tuple[str, ...]

    def identity(self, index: int) -> str:
        """The colour for identity ``index`` — the six validated slots, then the frozen extension.

        Past the end it wraps, and that is a documented limit rather than a design: at more than
        24 identities no colour scheme keeps them apart, and the chart must already be naming its
        marks another way.
        """
        colors = self.categorical + self.extended
        return colors[index % len(colors)]

    def label_identity(self, index: int) -> str:
        """The colour for a behavior label's slot ``index``: the identity colours less the
        slots reserved for the diverging scale (:data:`LABEL_RESERVED_SLOTS`), in order.

        **Blue and yellow mean less and more, and nothing else.** A label coloured like
        either pole reads as a direction on a cluster that has none, and the palette's first
        two slots are blue and amber (user, 2026-11-16). Every surface that colours a
        behavior by its label asks here — the cluster tree and its label bar, BROWSE's list
        and lanes, the label check, Find clusters' bands, and the exported behavior figures
        (:func:`assign_identities`) — so a label is one colour everywhere and never a
        pole's. Wraps past the end, the documented limit :meth:`identity` has.
        """
        colors = self.label_identities()
        return colors[index % len(colors)]

    def label_identities(self) -> tuple[str, ...]:
        """Every colour a label can take, in slot order."""
        colors = self.categorical + self.extended
        return tuple(c for i, c in enumerate(colors) if i not in LABEL_RESERVED_SLOTS)

    def identities(self, n: int) -> list[str]:
        """The first ``n`` identity colours, in order."""
        return [self.identity(i) for i in range(n)]

    def hue(self, name: str) -> str:
        """The categorical colour called ``name`` (see :data:`HUE_NAMES`), in this theme's step.

        Raises:
            ValueError: If ``name`` is not one of :data:`HUE_NAMES`.
        """
        try:
            return self.categorical[HUE_NAMES.index(name)]
        except ValueError:
            raise ValueError(
                f"hue must be one of {list(HUE_NAMES)}, got {name!r}"
            ) from None


#: inferno, nine even stops (``matplotlib.colormaps["inferno"](i / 8)``): the dark theme's
#: ``colormap``, and reversed the light theme's.
_INFERNO = (
    "#000004",
    "#210c4a",
    "#57106e",
    "#8a226a",
    "#bc3754",
    "#e45a31",
    "#f98e09",
    "#f9cb35",
    "#fcffa4",
)

LIGHT = ChartTheme(
    name="light",
    surface="#ffffff",
    text="#1a1a1a",
    text_secondary="#404040",
    text_muted="#707070",
    axis="#333333",
    grid="#e6e6e6",
    neutral="#cccccc",
    reference="#5c5c5c",
    summary="#1a1a1a",
    handle="#D55E00",
    categorical=_OKABE_ITO_LIGHT,
    extended=_EXTENDED_LIGHT,
    colormap=_INFERNO[::-1],
    sequential=(
        "#dedede",
        "#c4c4c4",
        "#ababab",
        "#929292",
        "#7a7a7a",
        "#636363",
        "#4d4d4d",
        "#383838",
        "#242424",
    ),
    diverging=(
        "#0b4c92",
        "#4a74ab",
        "#809dc3",
        "#b7c6da",
        "#f0f0f0",
        "#e9d9c0",
        "#e2c18f",
        "#daa95a",
        "#ce9113",
    ),
)

DARK = ChartTheme(
    name="dark",
    surface="#262626",
    text="#ededed",
    text_secondary="#c4c4c4",
    text_muted="#9a9a9a",
    axis="#c8c8c8",
    grid="#3a3a3a",
    neutral="#5a5a5a",
    reference="#a8a8a8",
    summary="#ededed",
    handle="#D55E00",
    categorical=_OKABE_ITO_DARK,
    extended=_EXTENDED_DARK,
    colormap=_INFERNO,
    sequential=(
        "#333333",
        "#484848",
        "#5d5d5d",
        "#747474",
        "#8c8c8c",
        "#a4a4a4",
        "#bebebe",
        "#d7d7d7",
        "#f2f2f2",
    ),
    diverging=(
        "#62a2ff",
        "#5686cd",
        "#4d6b98",
        "#435167",
        "#383838",
        "#655c41",
        "#968244",
        "#c9a940",
        "#ffd337",
    ),
)

THEMES: dict[str, ChartTheme] = {"light": LIGHT, "dark": DARK}


def theme(name: str) -> ChartTheme:
    """The theme called ``name``.

    Raises:
        ValueError: If ``name`` is not ``"light"`` or ``"dark"``.
    """
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(
            f"chart theme must be one of {sorted(THEMES)}, got {name!r}"
        ) from None


def rgb(color: str) -> tuple[int, int, int]:
    """A theme colour as an ``(r, g, b)`` triple, for the adapters that want integers."""
    return hex_to_rgb(color)


def assign_identities(labels, chart_theme: ChartTheme = LIGHT) -> dict[str, str]:
    """A stable colour per label: sorted, then the label colours in order
    (:meth:`ChartTheme.label_identity`: the identity colours less the two the diverging scale
    keeps for less and more).

    Sorted rather than first-seen, so "rearing" is the same colour in recording A's figure and
    recording B's, on screen and on paper — the one assignment every surface that colours a
    behavior by name goes through.
    """
    return {
        label: chart_theme.label_identity(i)
        for i, label in enumerate(sorted(set(labels)))
    }


# --------------------------------------------------------------------------------------------- #
# Weights and sizes
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Type:
    """Font sizes for one medium, in that medium's unit (pt on paper, pt on screen)."""

    family: tuple[str, ...]
    #: Tick labels, legend entries.
    tick: float
    #: Axis labels and panel titles.
    label: float
    #: Annotations that recede.
    annotation: float
    #: The bold ``A`` / ``B`` in the corner of a multi-panel figure.
    panel_letter: float
    #: Nothing is ever drawn smaller than this.
    floor: float


@dataclass(frozen=True)
class Marks:
    """Line weights and mark sizes for one medium (pt on paper, px on screen)."""

    axis_width: float
    tick_length: float
    tick_width: float
    #: A data line: a curve, a trajectory, a step function.
    line_width: float
    #: A line that is one of many — a per-subject line under a group summary.
    thin_line_width: float
    reference_width: float
    #: Error bars and summary ticks (a median across a group's points).
    summary_width: float
    #: Marker diameter.
    marker_size: float
    #: The surface-coloured ring round a marker, so overlapping points stay separable.
    marker_ring: float
    #: A bar takes this fraction of its slot; the rest is air.
    bar_fraction: float
    #: Opacity of a filled region drawn under data — a confidence band, a violin body.
    fill_alpha: float
    #: Opacity of individual observations drawn over a summary.
    point_alpha: float


#: Journal-standard final sizes. Nature and Cell both ask for 5-7 pt at the printed size with a
#: 5 pt minimum; this sits at the readable end, with 6 pt as the floor.
PRINT_TYPE = Type(
    family=("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"),
    tick=7.0,
    label=8.0,
    annotation=6.5,
    panel_letter=10.0,
    floor=6.0,
)

PRINT_MARKS = Marks(
    axis_width=0.8,
    tick_length=3.0,
    tick_width=0.8,
    line_width=1.5,
    thin_line_width=0.9,
    reference_width=0.8,
    summary_width=1.2,
    marker_size=4.5,
    marker_ring=0.6,
    bar_fraction=0.6,
    fill_alpha=0.25,
    point_alpha=0.9,
)

#: On screen a font size is relative to the application's own (``tick`` and ``label`` are offsets
#: from it); the family is the system UI font and is not set here.
SCREEN_TYPE = Type(
    family=(),
    tick=0.0,
    label=0.0,
    annotation=-1.0,
    panel_letter=2.0,
    floor=8.0,
)

SCREEN_MARKS = Marks(
    axis_width=1.0,
    tick_length=5.0,
    tick_width=1.0,
    line_width=2.0,
    thin_line_width=1.25,
    reference_width=1.0,
    summary_width=2.0,
    marker_size=8.0,
    marker_ring=1.5,
    bar_fraction=0.6,
    fill_alpha=0.25,
    point_alpha=0.9,
)

#: Printed figure widths, in inches: Nature's 89 mm / 136 mm / 183 mm columns (Cell's are within
#: 5 mm of each).
SINGLE_COLUMN_IN = 3.5
ONE_AND_HALF_COLUMN_IN = 5.35
DOUBLE_COLUMN_IN = 7.2
