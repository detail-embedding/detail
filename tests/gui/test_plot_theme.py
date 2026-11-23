"""A plot follows the application's chart theme — and follows it when the theme changes.

**The class:** a plot styled at its construction site. Every plot in the package was built with
``setBackground("w")``, so in a dark window each one was a lit panel with grey tick labels on it;
the colours of its marks were spelled at the call site too, and two of them had already drifted
from the curve they claimed to match. ``gui/plot_theme.py`` is the one place a plot is styled
(``requirements.a_plot_takes_its_colours_from_the_chart_theme`` keeps it that way); this file
asks what a themed plot actually looks like, in both themes, and across a live switch.

Driven through the application **palette** (the ``app_theme`` fixture), because the offscreen
test platform ignores a colour-scheme request.
"""

import pytest

pytest.importorskip("pyqtgraph")

from detail.core.chart_theme import DARK, LIGHT, SCREEN_MARKS  # noqa: E402
from detail.gui import plot_theme  # noqa: E402
from detail.gui.qt import QColor, QPalette


def _shown(qapp, plot):
    plot.resize(400, 240)
    plot.show()
    qapp.processEvents()
    return plot


def test_a_themed_plot_wears_its_theme(qapp, app_theme):
    plot = _shown(qapp, plot_theme.themed_plot())
    plot.setLabel("bottom", "speed (cm/s)")
    try:
        assert plot.backgroundBrush().color().name() == app_theme.surface
        axis = plot.getAxis("bottom")
        assert axis.pen().color().name() == app_theme.axis
        assert axis.textPen().color().name() == app_theme.text_secondary
        # The title in primary ink, set *after* the theme — the order every site uses.
        assert axis.labelStyle["color"] == app_theme.text
        assert axis.style["tickLength"] == int(SCREEN_MARKS.tick_length) > 0  # outward
        assert axis.autoSIPrefix is False
    finally:
        plot.close()


def test_a_themed_plot_follows_a_switch_while_it_is_open(qapp):
    """A system light/dark flip reaches the app as a new palette; an open plot re-themes."""
    original = QPalette(qapp.palette())
    plot = _shown(qapp, plot_theme.themed_plot())
    plot.setLabel("left", "precision")
    assert plot.backgroundBrush().color().name() == LIGHT.surface
    try:
        dark = QPalette(original)
        dark.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
        qapp.setPalette(dark)
        qapp.processEvents()
        assert plot_theme.current() is DARK
        assert plot.backgroundBrush().color().name() == DARK.surface
        axis = plot.getAxis("left")
        assert axis.labelStyle["color"] == DARK.text
        assert axis.labelText == "precision", "re-theming must keep the title it had"
    finally:
        qapp.setPalette(original)
        qapp.processEvents()
        plot.close()
    assert plot.backgroundBrush().color().name() == LIGHT.surface


def test_building_a_plot_survives_pyqtgraphs_own_palette_event(qapp, monkeypatch):
    """pyqtgraph's constructor delivers a palette change before its plot item exists; a handler
    that re-themed on it raised ``AttributeError: getAxis`` out of a Qt event, once per plot.
    """
    import sys

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    plot = _shown(qapp, plot_theme.themed_plot())
    plot.close()
    assert errors == []


def test_current_reads_the_palette_when_the_platform_names_no_scheme(qapp):
    """The fallback every test here stands on, asked directly."""
    original = QPalette(qapp.palette())
    try:
        assert plot_theme.current() is LIGHT
        dark = QPalette(original)
        dark.setColor(QPalette.ColorRole.Window, QColor("#202020"))
        qapp.setPalette(dark)
        assert plot_theme.current() is DARK
    finally:
        qapp.setPalette(original)


def test_a_legend_needs_its_corner_named(qapp):
    """Only the site knows where its data is not; a default corner is where a curve goes."""
    plot = plot_theme.themed_plot()
    with pytest.raises(TypeError):
        plot_theme.add_legend(plot)  # noqa — the missing keyword is the point
    legend = plot_theme.add_legend(plot, corner="bottom-right")
    assert legend is not None
    plot.close()
