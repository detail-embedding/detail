"""A figure as plain data — arrays, labels and colour *roles*, no plotting library.

A builder (``analysis/location_plots.build_plot``) describes a figure once; a renderer draws it.
``report/plot_render.py`` draws it with matplotlib (PNG / SVG / PDF, and the GUI's picture), and
an HTML renderer reads the same spec, so nothing about a figure lives only in one library.

Colours are roles, resolved against ``core/chart_theme.py`` by the renderer: a ``Line`` either
carries per-sample ``values`` read through the spec's ``ColorScale`` (the sequential ramp) or an
``identity`` index into the theme's categorical colours. One spec therefore renders light or dark.
NaN in a line's coordinates breaks the line there (a tracking gap is not drawn as a jump).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Axis:
    """One axis: its label (unit included) and, when panels share it, its range."""

    label: str
    range: tuple[float, float] | None = None


@dataclass(frozen=True)
class ColorScale:
    """How a line's ``values`` (or a heat map's cells) map to colour. ``ramp`` is a theme role."""

    label: str
    vmin: float
    vmax: float
    ramp: str = "sequential"


@dataclass(frozen=True)
class Line:
    """A path or a curve. ``z`` makes it 3-D. ``values`` colours it per sample through the
    spec's scale; otherwise ``identity`` picks a categorical colour (None = the theme's ink).
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    values: np.ndarray | None = None
    identity: int | None = None
    label: str = ""
    #: Drawn as steps centred on each x (a histogram's bars as one outline).
    step: bool = False


@dataclass(frozen=True)
class Band:
    """An uncertainty band around a curve, in the curve's identity colour."""

    x: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    identity: int = 0
    step: bool = False


@dataclass(frozen=True)
class Heat:
    """A grid of cell values, ``grid[row, col]`` over ``y_edges`` × ``x_edges``."""

    grid: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray


@dataclass(frozen=True)
class Panel:
    """One panel of a figure: a title, and the marks drawn in it."""

    title: str
    lines: list[Line] = field(default_factory=list)
    bands: list[Band] = field(default_factory=list)
    heat: Heat | None = None
    note: str = ""


@dataclass(frozen=True)
class Camera:
    """The one 3-D camera every panel shares, in degrees."""

    elev: float
    azim: float


@dataclass(frozen=True)
class PlotSpec:
    """A whole figure: panels sharing axes, one colour scale, one camera.

    ``aspect`` is the y/x data-unit ratio to hold (1.0 for a position plot; None lets the panel
    fill its box). ``invert_y`` puts y = 0 at the top, as in an image. ``legend`` is True when
    lines carry identities a reader has to tell apart: one entry per distinct ``(identity,
    label)`` across every panel, so an animal in several panels is named once.
    """

    kind: str
    title: str
    panels: list[Panel]
    x: Axis
    y: Axis
    z: Axis | None = None
    color: ColorScale | None = None
    camera: Camera | None = None
    aspect: float | None = None
    invert_y: bool = False
    legend: bool = False
    #: A figure-wide line under the title — what was left out, say.
    note: str = ""

    @property
    def three_d(self) -> bool:
        """True when the panels are 3-D (every line carries ``z``)."""
        return self.z is not None
