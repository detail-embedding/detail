"""An axis carries a quantity, and a quantity has a domain.

The class: **a plot lets the view — and any handle dragged on it — leave the domain of the
quantity it shows.** pyqtgraph gives every ``PlotWidget`` free pan and zoom, so a view that is
never told otherwise scrolls happily to confidence -4 and precision 3. Neither number exists.

Nothing fails. The plot renders, the axis stays labelled, the ticks are drawn at the coordinates
you scrolled to, and the only way to know the value you just read off it is meaningless is to
already know the quantity's domain — which is what the axis was there to tell you. All seven of
this package's plots shipped this way.

Two of them carry a **draggable line**, where the same defect has a worse ending: the confidence
cutoff on the calibration curve is committed to run state, and ``api.query.set_cutoff`` refuses
anything outside ``[0, 1]``. So an unbounded drag turned a gesture into an error dialog, with the
API check quietly doing the whole job of keeping a bad number off disk.

The fix is ``detail/gui/plots.py::bound_axes`` at every construction site, and pyqtgraph's own
``InfiniteLine.setBounds`` on every movable line. ``None`` is how "genuinely unbounded" is
spelled — a principal component has no domain and panning the projection is the point — so the
scan below can tell a deliberate choice from a forgotten call, which is the only thing that makes
a fail-closed scan liveable.
"""

import ast
import pathlib
import re

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from detail.gui.plots import bound_axes  # noqa: E402

_GUI = pathlib.Path(__file__).resolve().parents[2] / "detail" / "gui"


def _sources() -> dict[pathlib.Path, str]:
    return {py: py.read_text(encoding="utf-8-sig") for py in sorted(_GUI.rglob("*.py"))}


#: The ways this package builds a plot. ``themed_plot`` is the one every site uses since the chart
#: theme (``gui/plot_theme.py``); the bare constructor is kept because a scan that only knows the
#: new spelling would pass every file that went back to the old one. ``plot_theme.py`` itself is
#: where the constructor lives, not a plot.
_BUILDS_A_PLOT = ("PlotWidget()", "themed_plot(")


def _name(node) -> str | None:
    """``self._plot`` -> ``_plot``; ``plot`` -> ``plot``: the handle a plot is kept under."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _unbounded_plots(text: str) -> list[str]:
    """Plots built in ``text`` that are never handed to ``bound_axes`` — per plot, not per file.

    The first version asked whether the *file* contained ``bound_axes(`` at all, so one call
    excused every plot beside it: deleting the audit strip's call while the curve's remained
    stayed green. Each plot is matched to a call on its own handle instead.
    """
    tree = ast.parse(text)
    factories = {s.rstrip("()(") for s in _BUILDS_A_PLOT}
    built, bounded = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            made = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if made in factories:
                built += [n for n in map(_name, node.targets) if n]
        if (
            isinstance(node, ast.Call)
            and _name(node.func) == "bound_axes"
            and node.args
            and _name(node.args[0])
        ):
            bounded.add(_name(node.args[0]))
    return [name for name in built if name not in bounded]


def _builds_a_plot_without_a_domain(text: str) -> bool:
    return bool(_unbounded_plots(text))


def test_every_plot_declares_the_domain_of_its_axes():
    missing = [
        py.relative_to(_GUI.parent).as_posix()
        for py, text in _sources().items()
        # `plot_theme.py` is where the constructor lives; `requirements.py` spells a plot only
        # inside a counterexample string, the thing a rule must reject — neither is a plot.
        if py.name not in ("plot_theme.py", "requirements.py")
        and _builds_a_plot_without_a_domain(text)
    ]
    assert not missing, (
        "these build a plot without saying what its axes can show, so the view can be "
        f"scrolled outside the quantity's domain: {missing}. Call bound_axes(plot, x=…, y=…), "
        "passing None for an axis that genuinely has no domain."
    )


def test_every_draggable_line_declares_its_domain():
    missing = [
        py.relative_to(_GUI.parent).as_posix()
        for py, text in _sources().items()
        if re.search(r"InfiniteLine\([^)]*movable=True", text)
        and "setBounds(" not in text
    ]
    assert not missing, (
        "a movable line whose position is a bounded quantity must carry setBounds(), or it "
        f"can be dragged to a value the quantity cannot take: {missing}"
    )


def _imports_strip(source: str) -> bool:
    """Whether this module imports the shared strip by name."""
    try:
        tree = ast.parse(source)
    except (
        SyntaxError
    ):  # pragma: no cover - the package has to parse to be imported at all
        return False
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "detail.gui.plots"
        for alias in node.names
        if alias.name == "DescriptorStrip"
    }
    # An import beside a local class of the same shape is the reimplementation, not a user —
    # and naming the import `_Local` frees the real name for the copy. So neither the class's
    # own name nor the alias may be defined here.
    local = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    return bool(imported) and not (local & (imported | {"DescriptorStrip"}))


def test_the_shared_strip_is_actually_shared():
    """`plots.py` says its strip is "the one repeated object" the band is built from. That was
    false for a while: #301 rewrote it as a private class in `gui/measurements.py` and left this
    one orphaned, with the docstring still making the claim — two implementations of one row, one
    unreachable, and nothing failing until the two disagreed about what a number meant.

    A shared object nobody shares is indistinguishable from a deleted one, so the guard is that
    somebody outside `plots.py` imports it.
    """
    # An *import*, parsed — not a substring. A text scan is satisfied by a comment naming the
    # class, and by `import DescriptorStrip as _Local` beside a local reimplementation, which is
    # the shape being guarded against. These imports are parenthesised over several lines, so a
    # line-based regex misses them too.
    users = [
        py.relative_to(_GUI.parent).as_posix()
        for py, text in _sources().items()
        if py.name != "plots.py" and _imports_strip(text)
    ]
    assert users, (
        "DescriptorStrip has no caller — either wire it up or delete it, but do not leave a "
        "second copy of the row in the module that draws it."
    )


def test_the_shared_strip_scan_would_catch_a_reimplementation():
    """Guard the guard. A comment naming the class, or a local class of the same shape, must not
    count as sharing — which a substring scan cannot tell apart."""
    assert _imports_strip("from detail.gui.plots import DescriptorStrip\n")
    assert _imports_strip(
        "from detail.gui.plots import (\n    bound_axes,\n    DescriptorStrip,\n)\n"
    )
    assert not _imports_strip(
        "# a copy of DescriptorStrip\nclass DescriptorStrip: ...\n"
    )
    assert not _imports_strip("from detail.gui.plots import bound_axes\n")
    # The spelling this scan's own comment names, which it used to pass: an aliased import
    # sitting beside a local copy under the real name.
    assert not _imports_strip(
        "from detail.gui.plots import DescriptorStrip as _Local\n"
        "class DescriptorStrip: ...\n"
    )


def test_the_scan_would_catch_a_plot_that_declares_nothing():
    """Guard the guard: a scan that matches nothing passes vacuously forever — and every site
    moved from ``pg.PlotWidget()`` to ``plot_theme.themed_plot()`` in one change, which is exactly
    when a scan that knew only the old spelling would have stopped seeing any plot at all.
    """
    for build in ("pg.PlotWidget()", "plot_theme.themed_plot()"):
        source = f"def build():\n    plot = {build}\n"
        assert _builds_a_plot_without_a_domain(source), build
        assert not _builds_a_plot_without_a_domain(
            source + "    bound_axes(plot, x=None, y=None)\n"
        )


def test_the_scan_would_catch_one_plot_left_unbounded_beside_a_bounded_one():
    """The shape that got through: two plots in one file, only one of them bounded."""
    source = (
        "def build(self):\n"
        "    self._plot = plot_theme.themed_plot()\n"
        "    bound_axes(self._plot, x=(0.0, 1.0), y=(0.0, 1.0))\n"
        "    self._strip = plot_theme.themed_plot()\n"
    )
    assert _unbounded_plots(source) == ["_strip"]


def test_the_scan_would_catch_an_unbounded_movable_line():
    source = "line = pg.InfiniteLine(pos=0.5, angle=90, movable=True)\n"
    assert re.search(r"InfiniteLine\([^)]*movable=True", source)
    assert "setBounds(" not in source


# -- what bound_axes actually does to a view ---------------------------------------- #


def _plot(qapp, **domains):
    import pyqtgraph as pg

    plot = pg.PlotWidget()
    bound_axes(plot, **domains)
    plot.resize(500, 200)
    plot.show()
    QApplication.processEvents()
    return plot


def _wrench(plot):
    """Zoom hard and pan hard — what a wheel and a drag do, without a mouse."""
    view = plot.getViewBox()
    view.scaleBy((8.0, 8.0))
    view.translateBy((5.0, 5.0))
    QApplication.processEvents()
    return view.viewRange()


def test_a_closed_domain_holds_through_zoom_and_pan(qapp):
    plot = _plot(qapp, x=(0.0, 1.0), y=(0.0, 1.0))
    (x_low, x_high), (y_low, y_high) = _wrench(plot)
    # 2% slack so a point sitting on the boundary is drawn whole, and not a pixel more.
    assert -0.021 <= x_low and x_high <= 1.021
    assert -0.021 <= y_low and y_high <= 1.021


def test_a_half_open_domain_holds_its_floor_and_leaves_the_top_free(qapp):
    """A count has a floor and no ceiling — the bound says exactly that."""
    plot = _plot(qapp, x=(0.0, None), y=(0.0, None))
    (x_low, x_high), (y_low, _) = _wrench(plot)
    assert x_low >= 0.0 and y_low >= 0.0
    assert x_high > 1.0, "an open end must stay open"


def test_a_declared_unbounded_axis_is_left_alone(qapp):
    """The PCA scatter: no domain, and panning it is the point."""
    plot = _plot(qapp, x=None, y=None)
    (x_low, x_high), _ = _wrench(plot)
    assert x_high - x_low > 5.0, "an undeclared domain must not be clamped"


def test_the_calibration_cutoff_cannot_be_dragged_out_of_range(qapp):
    import pyqtgraph as pg

    line = pg.InfiniteLine(pos=0.5, angle=90, movable=True)
    line.setBounds([0.0, 1.0])
    line.setValue(-2.0)
    assert line.value() == 0.0
    line.setValue(4.0)
    assert line.value() == 1.0
