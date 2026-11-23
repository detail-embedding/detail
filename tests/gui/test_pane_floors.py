"""A plot is given a range to live in, never a single number.

The class: **one number serving as a floor and a ceiling at once.** ``setFixedHeight(n)`` says
both "never smaller" and "never larger", so a container shorter than ``n`` has no way to make the
widget fit — it keeps its height and Qt clips it at the parent's edge. For a plot that means the
bottom of the drawing goes, and the bottom of a plot is its x-axis and every tick label on it.
Nothing errors; the plot is there, correctly drawn, with its axis cut off.

Measured on the query calibration pane at a laptop size: a 444px dial column in a 334px viewport,
cropped by 110px — of which **304px was two ``setFixedHeight`` calls**, the plot's 240 and its
caption's 64, neither of which could give up a pixel. That is the "the plots are cut off" report.

pyqtgraph imposes no such floor of its own — ``PlotWidget.minimumSizeHint()`` is 68x68 — and it
degrades gracefully, because its axis chrome is a *constant* ~40px whatever the height: measured
at 240/160/120/90px the data area is 200/120/80/50. So a plot given a **range** shrinks its data
area and keeps its axis; a plot given a fixed height loses the axis and keeps the data area.

There is no companion scan for raw-pixel ``setMinimumHeight`` here, deliberately. The ones left in
the package are on image and canvas previews (``streams``, ``arena``, ``run_preview``), where the
number comes from an aspect ratio rather than from text, and ``sizing.lines`` would be the wrong
unit for them. The typographic rule belongs where a size is holding *text* or a graph.
"""

import ast
import pathlib

import pytest

pytest.importorskip("PyQt6")

_GUI = pathlib.Path(__file__).resolve().parents[2] / "detail" / "gui"

#: How a plot is spelled in this package. Since #492 every plot is built by
#: `plot_theme.themed_plot()`; the pyqtgraph names stay for a site that bypasses it. The scan
#: matched nothing for months after that change, because this set still held only the pyqtgraph
#: names — `test_the_scan_sees_the_plots_the_package_builds` is the floor that makes it say so.
_PLOT_FACTORIES = {"themed_plot", "PlotWidget", "GraphicsLayoutWidget"}

#: Plot construction sites the scan recognised when the floor was set (14 on 2026-09-29; 7 after
#: the hidden capabilities' GUIs were removed on 2026-11-10; 4 once query eval's explorer and the
#: query AUDIT tab's plots went, 2026-11-23). Set at or below the count so a plot that is removed
#: does not fail it, and far above zero so a factory rename does.
_PLOTS_SEEN_FLOOR = 3


def _target_name(node) -> str | None:
    """``self._plot`` -> ``_plot``; ``plot`` -> ``plot``. The attribute is enough to key on."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _plots_built(tree: ast.AST) -> set[str]:
    """Names bound to the result of a plot factory in ``tree``."""
    plots: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        made = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if made not in _PLOT_FACTORIES:
            continue
        for target in node.targets:
            name = _target_name(target)
            if name:
                plots.add(name)
    return plots


def _fixed_height_plots(source: str) -> list[str]:
    """Names bound to a plot in ``source`` that also get ``setFixedHeight``."""
    tree = ast.parse(source)
    plots = _plots_built(tree)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "setFixedHeight":
            continue
        name = _target_name(node.func.value)
        if name in plots:
            offenders.append(name)
    return offenders


def test_no_plot_is_pinned_to_a_single_height():
    """The scan. A plot needs a floor *and* a ceiling, which is two calls, not one."""
    found = []
    for path in sorted(_GUI.rglob("*.py")):
        for name in _fixed_height_plots(path.read_text(encoding="utf-8-sig")):
            found.append(f"{path.relative_to(_GUI.parent.parent)}: {name}")
    assert not found, (
        "a plot pinned to one height cannot be squeezed, only cropped — and what crops is its "
        f"x-axis. Use setMinimumHeight + setMaximumHeight (in `sizing.lines`): {found}"
    )


def test_the_scan_would_catch_one():
    """Guard the guard: a scan that matches nothing today passes vacuously forever."""
    assert _fixed_height_plots(
        "import pyqtgraph as pg\n"
        "class W:\n"
        "    def __init__(self):\n"
        "        self._plot = pg.PlotWidget()\n"
        "        self._plot.setFixedHeight(240)\n"
    ) == ["_plot"]


def test_the_scan_would_catch_one_built_the_way_the_package_builds_plots():
    """The spelling every plot in the package uses. The case above alone let the scan go blind:
    it kept passing on a pyqtgraph name nothing in `detail/gui` constructs any more."""
    assert _fixed_height_plots(
        "from detail.gui import plot_theme\n"
        "class W:\n"
        "    def __init__(self):\n"
        "        self._plot = plot_theme.themed_plot()\n"
        "        self._plot.setFixedHeight(240)\n"
    ) == ["_plot"]


def test_the_scan_sees_the_plots_the_package_builds():
    """A scan over plots that recognises no plot passes in a package full of them."""
    seen = sum(
        len(_plots_built(ast.parse(path.read_text(encoding="utf-8-sig"))))
        for path in sorted(_GUI.rglob("*.py"))
    )
    assert seen >= _PLOTS_SEEN_FLOOR, (
        f"the scan recognises {seen} plot construction sites in detail/gui — a factory was "
        f"renamed or added that `_PLOT_FACTORIES` does not name, so it is checking nothing"
    )


def test_the_scan_leaves_a_ranged_plot_alone():
    """The negative case: the shape the fix takes must not be flagged."""
    assert (
        _fixed_height_plots(
            "import pyqtgraph as pg\n"
            "class W:\n"
            "    def __init__(self):\n"
            "        self._plot = pg.PlotWidget()\n"
            "        self._plot.setMinimumHeight(128)\n"
            "        self._plot.setMaximumHeight(272)\n"
        )
        == []
    )


def test_the_scan_leaves_a_fixed_height_non_plot_alone():
    """A painted strip sized to its own row count is a legitimate fixed height — this class is
    about plots, whose bottom edge carries an axis."""
    assert (
        _fixed_height_plots(
            "class Strip:\n"
            "    def __init__(self, rows):\n"
            "        self.setFixedHeight(max(len(rows), 1) * 20)\n"
        )
        == []
    )


# --- the behaviour the scan stands in for ------------------------------------------- #


def test_the_query_dial_column_fits_the_pane_it_is_given(qapp, monkeypatch):
    """The reported bug, as a number: 444px of column in a 334px viewport.

    Asserted on the floor rather than on a screenshot, because "is the axis visible" is what a
    person checks and "does the column's minimum exceed its pane" is the same question in the
    form that fails loudly.
    """
    import detail.gui.busy as busy
    import detail.gui.modules.analysis_query as aq

    monkeypatch.setattr(busy, "run_busy", lambda t, f, d, **k: d(f()))
    monkeypatch.setattr(aq, "run_busy", lambda t, f, d, **k: d(f()))

    class _Experiment:
        dir = pathlib.Path(".")

        def recording_names(self):
            return {}

    class _Run:
        editable = True
        status = "open"
        id = "an_00"
        label = "rear"

        def __getattr__(self, name):
            raise RuntimeError("stub")

    from PyQt6.QtWidgets import QApplication

    tab = aq._CalibrateTab(_Run(), _Experiment())
    tab.resize(1400, 620)
    tab.show()
    QApplication.processEvents()
    tab.resize(1400, 620)
    QApplication.processEvents()

    column = tab._dial_scroll.widget()
    viewport = tab._dial_scroll.viewport().height()
    assert column.minimumSizeHint().height() <= viewport, (
        f"the dial column's floor is {column.minimumSizeHint().height()}px in a {viewport}px "
        "pane, so its scroll area crops the graph — which takes the x-axis with it"
    )


def test_a_plot_squeezed_below_its_preferred_height_shrinks_rather_than_cropping(qapp):
    """The property that makes a range a fix: the widget gets smaller, so its axis stays inside
    it. A fixed-height plot in the same host keeps its height and is clipped by the parent.
    """
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

    pg = pytest.importorskip("pyqtgraph")
    from detail.gui.sizing import lines

    host = QWidget()
    box = QVBoxLayout(host)
    plot = pg.PlotWidget()
    plot.setMinimumHeight(lines(8))
    plot.setMaximumHeight(lines(17))
    box.addWidget(plot)
    host.resize(400, lines(10))
    host.show()
    QApplication.processEvents()
    host.resize(400, lines(10))
    QApplication.processEvents()

    assert plot.height() <= host.height(), (
        f"the plot is {plot.height()}px inside a {host.height()}px host — the overflow is "
        "clipped, and the bottom of a plot is its axis"
    )
    assert plot.height() >= lines(8), "it must not shrink past the floor it declared"
