"""Plot vocabulary — an axis carries a quantity, and a quantity has a domain.

Three things live here: ``bound_axes``; ``DescriptorStrip``, the one repeated object the
Measurements band is built from; and ``cluster_lane``, the strip that answers "and where do the
*other* clusters sit". A panel of ten hand-rolled pyqtgraph widgets is ten places to forget the
theme, the domain, the SI-prefix suppression and the range set after the items — so the row is
written once, here, and repeated per measurement. Colours and weights are the chart theme's
(``gui/plot_theme.py``): the run's spread is the neutral, this selection's members the first
identity colour.

That claim was false for a while and is worth keeping true. #301 rewrote the strip as a private
class in ``gui/measurements.py`` and left this one orphaned, with this docstring still calling it
"the one repeated object" — two implementations of one row, one of them unreachable, and the
prose asserting the opposite. ``tests/gui/test_plot_bounds.py`` now checks that the strip has a
caller, because a shared object nobody shares is indistinguishable from a deleted one until the
two copies disagree about what a number means.

``pyqtgraph`` gives every plot free pan and zoom, so a view that is never told otherwise will
happily scroll to confidence −4 and precision 3. Neither number exists. Nothing fails when
somebody does it — the plot renders, the axis is labelled, the ticks are drawn — and the only
way to find out the coordinate you just read is meaningless is to already know the quantity's
domain, which is the thing the axis was supposed to tell you.

So the domain is declared where the plot is built, and ``None`` is how "genuinely unbounded"
is spelled: an argument someone passed rather than a call someone forgot, which is the whole
difference between a projection axis and an oversight (``tests/gui/test_plot_bounds.py``).

**And the axis is drawn.** The strip used to hide both of them, so nine rows stacked nine
different percentile-clipped scales with nothing on screen to say what any of them was, and two
dots at the same horizontal position shared nothing. Reported as *"I'm confused by the x axis
for all the plots"*, which is what an absent axis reads as.
"""

import numpy as np

from detail.gui import plot_theme
from detail.gui.qt import QSizePolicy, QVBoxLayout, QWidget, Signal


def bound_axes(plot, *, x, y) -> None:
    """Stop the view leaving the domain of the quantity on each axis.

    Args:
        plot: A ``pyqtgraph.PlotWidget``.
        x: ``(low, high)`` for the bottom axis, either end ``None`` for open on that side, or
            the whole axis ``None`` for a quantity with no domain (projection units).
        y: The same, for the left axis.
    """
    view = plot.getViewBox()
    limits = {}
    for axis, domain in (("x", x), ("y", y)):
        low, high = (None, None) if domain is None else domain
        # Slack only where a span is closed at both ends, and only then because a symbol
        # sitting exactly on the boundary would otherwise be drawn half outside the frame.
        # A half-open axis — a count, a distance — takes its limit exactly: points at zero
        # sit on the baseline, which is what every plotting library does with a count.
        pad = 0.02 * (high - low) if low is not None and high is not None else 0.0
        if low is not None:
            limits[f"{axis}Min"] = low - pad
        if high is not None:
            limits[f"{axis}Max"] = high + pad
    if limits:
        view.setLimits(**limits)


# --- one measurement, one row ------------------------------------------------------------ #
#: Past this many members a scatter is a smear, and a smear cannot be clicked. Raised from 120
#: once the marks began stacking: at :data:`BINS` columns and a row six lines tall a column holds
#: roughly a dozen dots before they touch, so the row saturates around here — and 120 was cutting
#: off clicking at sizes that read perfectly well (a 127-match behavior drew as a density).
DENSITY_ABOVE = 500

#: The axis spans this percentile range of the run's values rather than its extremes. These
#: quantities are heavy-tailed and a full-range axis renders every value as a single mark at the
#: left edge — measured on a real recording, on duration and net displacement both. The tails are
#: not discarded; they are clamped to the edge, so a value past the range appears at the end.
PERCENTILE_LOW, PERCENTILE_HIGH = 1.0, 99.0

#: Bins for the background spread, and for the members once they are a density.
BINS = 48

#: How much of the axis a dot occupies, so a hundred members read as a distribution and eight
#: read as eight.
DOT_PX = 7

#: Opacity of the members' density, so the run's spread reads *through* it rather than being
#: hidden by the thing it exists to be compared against.
_DENSITY_ALPHA = 170

#: A row narrower than this is a rule with marks on it rather than a distribution. ``QFormLayout``
#: hands its field column the widget's *minimum* width and a ``PlotWidget`` asks for almost
#: nothing, so without a floor nine rows rendered 70px wide beside an empty video pane.
MIN_ROW_WIDTH_PX = 320

#: A row is a height *range*, never one number: a fixed height is a floor and a ceiling at once,
#: so a short pane cannot squeeze the row and crops it instead — and what a plot loses off the
#: bottom is its x axis (``tests/gui/test_pane_floors.py``).
MIN_ROW_PX, MAX_ROW_PX = 40, 260

#: The cluster lane's height — a floor and a ceiling, not one number. It holds a tick and no
#: axis, so it looks like the one plot a fixed height could not hurt; but a pane short enough to
#: squeeze everything else would crop this instead of shrinking it, and a cropped tick is a tick
#: pointing at nothing. Two calls, like every other plot in the package.
LANE_MIN_PX, LANE_MAX_PX = 22, 44

#: Units that are not units — a range, a formula, a bare tally. They belong on an axis, where
#: they say what the numbers mean; they never follow a value, where ``0.15 0–1`` is what one
#: format string produces.
NOT_A_UNIT = frozenset({"0–1", "0-1", "1-cos", "count", ""})


def fmt_value(value: float, unit: str) -> str:
    """A number and its unit, spaced the way that unit is written.

    ``0 /s`` and ``1 count`` are what one format string produces, and neither is how a rate or a
    plain count is written. The one formatter for these numbers: a second copy diverges silently,
    which is how the same measure came to read ``0.317 fraction`` on one page and ``31.7 %`` on
    another (``tests/gui/test_run_detail.py``).
    """
    if not np.isfinite(value):
        return "—"
    if unit == "count":
        return f"{value:.0f}"
    text = f"{value:.3g}"
    if unit in NOT_A_UNIT:
        return text
    return f"{text}{unit}" if unit.startswith("/") else f"{text} {unit}"


def axis_title(label: str, unit: str) -> str:
    """What the bottom axis is called — the measurement, and what its numbers are in."""
    return f"{label} ({unit})" if unit and unit != "count" else label


def robust_range(pool) -> tuple[float, float]:
    """An axis that shows the data rather than its outliers."""
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    if not len(pool):
        return 0.0, 1.0
    low, high = (
        float(v) for v in np.percentile(pool, [PERCENTILE_LOW, PERCENTILE_HIGH])
    )
    if high - low <= 0:
        # Every value identical, or a range collapsed by the clip: a domain of zero width has no
        # inside, so the axis is given one and the marks land in the middle. A full-width bar
        # would claim a spread that is not there.
        centre = float(np.median(pool))
        return centre - 0.5, centre + 0.5
    return low, high


def swarm(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """A y for each value, stacking the ones that share a column.

    Height carries count. On one line, three hundred members and thirty are the same picture —
    which is what a pose cluster's row looked like, and why this exists.
    """
    if not len(values):
        return values
    column = np.clip(((values - low) / (high - low) * BINS).astype(int), 0, BINS - 1)
    y = np.zeros(len(values), dtype=np.float64)
    seen: dict[int, int] = {}
    for i, c in enumerate(column):
        y[i] = seen.get(int(c), 0)
        seen[int(c)] = y[i] + 1
    tallest = max(y.max(), 1.0)
    return 0.12 + 0.62 * (y / tallest)


class DescriptorStrip(QWidget):
    """One measurement: the run's spread behind, this selection's members on top of it.

    A table of numbers answers "what is this cluster's speed"; it does not answer "is that a lot",
    which is the question somebody naming a cluster is actually asking. The spread behind the dots
    answers it without a second plot and without leaving the row.

    **The dots are the members, not a summary of them.** An outlier is a dot away from its
    fellows, and clicking it is how the number becomes a clip — which is the point of putting the
    two on one surface rather than side by side. Past :data:`DENSITY_ABOVE` there is nothing to
    click at, so the dots become a density and :attr:`is_dense` says so, because "each dot is one"
    and "taller means more" are different claims about the same marks.

    **One unit of observation.** Every mark on this plot is one member. An earlier version drew a
    tick per *cluster* on the same axis, and the report was "I'm confused by the median vertical
    bars" — a mark that silently means something else is not fixed by a caption. That picture is
    :func:`cluster_lane`, which is its own plot with its own label.
    """

    picked = Signal(int)  #: index into the member list handed to :meth:`show_values`

    def __init__(self, label: str, unit: str, parent=None) -> None:
        super().__init__(parent)
        self.label = label
        self.unit = unit
        self._points = None
        self._dense = False

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.plot = plot_theme.themed_plot()
        self.plot.hideAxis("left")
        self.plot.getAxis("bottom").setLabel(axis_title(label, unit))
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.getViewBox().setYRange(-1.0, 1.0, padding=0)
        self.plot.setMinimumWidth(MIN_ROW_WIDTH_PX)
        self.plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        box.addWidget(self.plot)
        self.set_row_height(MIN_ROW_PX * 2)

    def set_row_height(self, px: int) -> None:
        """A range, never one number — see :data:`MIN_ROW_PX`."""
        px = int(min(MAX_ROW_PX, max(MIN_ROW_PX, px)))
        self.plot.setMinimumHeight(int(px * 0.6))
        self.plot.setMaximumHeight(px)

    @property
    def is_dense(self) -> bool:
        """Whether this row drew a density rather than individually clickable points."""
        return self._dense

    def show_values(self, run_wide, members) -> None:
        """Draw ``run_wide`` as the background spread and ``members`` on top of it.

        Raw values both; the strip scales them, because the alternative is every caller
        normalizing and one of them eventually doing it differently. A ``NaN`` is dropped rather
        than drawn at zero — the descriptor said it could not be computed, and a mark on the
        baseline would say it was computed and came out low.
        """
        import pyqtgraph as pg

        theme = plot_theme.current()
        self.plot.clear()
        self._points = None
        self._dense = False
        run_values = _finite(run_wide)
        member_values = _finite(members)
        if not len(run_values) and not len(member_values):
            return
        pool = (
            np.concatenate([run_values, member_values])
            if len(run_values)
            else member_values
        )
        low, high = robust_range(pool)

        if len(run_values):
            counts, edges = np.histogram(
                np.clip(run_values, low, high), bins=BINS, range=(low, high)
            )
            if counts.max():
                self.plot.addItem(
                    pg.BarGraphItem(
                        x=0.5 * (edges[:-1] + edges[1:]),
                        height=0.95 * counts / counts.max(),
                        y0=-1.0,
                        width=(high - low) / BINS,
                        brush=plot_theme.brush(theme.neutral),
                        pen=None,
                    )
                )

        if len(member_values):
            clipped = np.clip(member_values, low, high)
            self._dense = len(clipped) > DENSITY_ABOVE
            if self._dense:
                counts, edges = np.histogram(clipped, bins=BINS, range=(low, high))
                self.plot.addItem(
                    pg.BarGraphItem(
                        x=0.5 * (edges[:-1] + edges[1:]),
                        height=0.7 * counts / max(counts.max(), 1),
                        y0=0.1,
                        width=(high - low) / BINS,
                        brush=plot_theme.brush(theme.identity(0), _DENSITY_ALPHA),
                        pen=None,
                    )
                )
            else:
                self._points = pg.ScatterPlotItem(
                    x=clipped,
                    y=swarm(clipped, low, high),
                    size=DOT_PX,
                    brush=plot_theme.brush(theme.identity(0)),
                    # The surface ring: stacked members stay separable where they touch.
                    pen=plot_theme.pen(theme.surface, width=1),
                    hoverable=True,
                    data=np.arange(len(clipped)),
                )
                self._points.sigClicked.connect(self._on_clicked)
                self.plot.addItem(self._points)

        # After the items, never before: a range set first does not survive them being added,
        # which is how a splitter drag used to reset a plot's view (`gui_design` §3).
        self.plot.setXRange(low, high, padding=0.02)
        bound_axes(self.plot, x=(low, high), y=(-1.0, 1.0))
        self.plot.getViewBox().disableAutoRange()
        self.x_range = (low, high)

    def _on_clicked(self, _item, points) -> None:
        if len(points):
            self.picked.emit(int(points[0].data()))


def cluster_lane(medians, own: float, low: float, high: float, parent=None):
    """One tick per group, at its median, with ``own`` picked out — its **own** plot.

    Not a layer on :class:`DescriptorStrip`: a mark here means a whole cluster and a mark there
    means one bout, and an axis carrying both without saying so is what "I'm confused by the
    median vertical bars" describes. Stacked above the strip and given the same ``low, high``, it
    shares the x axis — both hide the left axis, so the two plotting areas start at the same x
    and a tick sits over its own data (``gui_design``'s stacked-plot rule).
    """
    import pyqtgraph as pg

    theme = plot_theme.current()
    plot = plot_theme.themed_plot(parent=parent)
    plot.hideAxis("left")
    plot.hideAxis("bottom")
    plot.setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.setMinimumHeight(LANE_MIN_PX)
    plot.setMaximumHeight(LANE_MAX_PX)
    plot.setMinimumWidth(MIN_ROW_WIDTH_PX)
    for median in _finite(medians):
        plot.addItem(
            pg.PlotCurveItem(
                x=[float(np.clip(median, low, high))] * 2,
                y=[0.0, 1.0],
                pen=plot_theme.pen(theme.neutral, width=1),
            )
        )
    if np.isfinite(own):
        plot.addItem(
            pg.PlotCurveItem(
                x=[float(np.clip(own, low, high))] * 2,
                y=[-0.45, 1.0],
                pen=plot_theme.pen(theme.identity(0), width=3),
            )
        )
    plot.getViewBox().setYRange(-0.45, 1.0, padding=0)
    plot.setXRange(low, high, padding=0.02)
    bound_axes(plot, x=(low, high), y=(-0.45, 1.0))
    plot.getViewBox().disableAutoRange()
    return plot


def _finite(values) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    return values[np.isfinite(values)]
