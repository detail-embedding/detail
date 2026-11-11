"""A page child that asks to expand is given room to expand.

The class: **a page adds every child with stretch 0, so a widget that declares it wants to fill
the content area can never get any.** Pages top-justify and end with `addStretch()`, which is
right for stacked rows of small controls — but it means a fill element (a splitter, a canvas, a
3-D scatter) collapses to its size hint while the stretch beneath it takes the rest of the
window.

Nothing fails when this happens. The page renders, the widget is present, it is simply
letterbox-thin with empty space below it, and the only symptom is somebody saying "the plot is
too small". It recurred often enough to be worth a mechanism rather than another per-page tweak.

``gui_design`` §3.1's *Pages top-justify* already names the *opposite* failure — a page with
small content, no fill child and no trailing stretch — and this is its mirror,
which had neither rule nor guard. `pages.add_body_widget` is the mechanism; this is its guard,
including the negative case, since a rule that cannot fail passes vacuously forever.
"""

from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from detail.gui.pages import add_body_widget


def _expanding() -> QWidget:
    widget = QWidget()
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return widget


def test_a_widget_that_asks_to_expand_gets_a_stretch(qapp):
    body = QVBoxLayout()
    add_body_widget(body, _expanding())
    assert body.stretch(0) == 1


def test_an_ordinary_row_keeps_the_top_justified_behaviour(qapp):
    """The negative case. Giving *everything* stretch would spread a column of small controls
    down the page — the failure the top-justify rule exists to prevent — so a widget that does
    not ask must not be given."""
    body = QVBoxLayout()
    add_body_widget(body, QLabel("a row"))
    assert body.stretch(0) == 0


def test_the_two_kinds_coexist_in_one_body(qapp):
    """A real page is mixed: a heading, a fill element, some buttons. Only the fill element
    should take the slack."""
    body = QVBoxLayout()
    for widget in (QLabel("heading"), _expanding(), QLabel("footer")):
        add_body_widget(body, widget)
    assert [body.stretch(i) for i in range(3)] == [0, 1, 0]


def test_it_reads_the_vertical_policy_not_the_horizontal_one(qapp):
    """A page body is a column, so only the vertical policy is about who takes the slack.

    Worth pinning: a widget that expands horizontally and is fixed vertically is extremely
    common (any full-width row), and giving those a stretch would spread every form down the
    page — the exact failure the top-justify rule prevents.
    """
    body = QVBoxLayout()
    wide = QWidget()
    wide.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    add_body_widget(body, wide)
    assert body.stretch(0) == 0


def test_a_widget_that_ignores_its_hint_counts_as_wanting_the_space(qapp):
    """``Ignored`` means "fill me", and this predicate has to know that.

    Qt spells "give me more than my hint" two ways. ``Expanding`` and ``MinimumExpanding`` carry
    ``ExpandFlag``; ``Ignored`` does **not** — measured, its value is ``Grow | Shrink | Ignore``
    — and yet Qt documents it as *"the sizeHint() is ignored; the widget will get as much space
    as possible"*. Same intent, different route.

    ``expands_vertically``'s docstring named ``Ignored`` among the policies it covered and the
    test for it was the flag it does not have. Nothing noticed while no widget used the policy;
    the first one that did — the video pane, which must not take its size from the pixmap it was
    handed — was read as a label spreading down its pane, and two unrelated guards failed.
    """
    from PyQt6.QtWidgets import QSizePolicy, QWidget

    from detail.gui.sizing import expands_vertically

    widget = QWidget()
    for policy in (
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.MinimumExpanding,
        QSizePolicy.Policy.Ignored,
    ):
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, policy)
        assert expands_vertically(
            widget
        ), f"{policy.name} asks for the space and was refused it"

    for policy in (
        QSizePolicy.Policy.Fixed,
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Maximum,
    ):
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, policy)
        assert not expands_vertically(
            widget
        ), f"{policy.name} does not ask for the space"


def _fit_page(qapp, child: QWidget):
    """A real `_FitDetailPage` (the analysis, dataset, model, distill and export pages'
    base) whose body is a heading and ``child``, laid out at a window's size."""
    from detail.gui.pages import _FitDetailPage

    class _Page(_FitDetailPage):
        def _record(self):
            return {"status": "complete"}

        def _render(self, record):
            self._add(QLabel("Results"))
            self._add(child)

    page = _Page("Analysis: an_03")
    page.resize(900, 1000)
    page.show()
    qapp.processEvents()
    return page


def test_a_fit_page_gives_its_fill_child_the_window_not_half_of_it(qapp):
    """The analysis run page sat a clustering run's INTERPRET in the top half of the window:
    the fill child had stretch 1 and so did the page's own trailing spacer, and they split it.
    """
    child = _expanding()
    page = _fit_page(qapp, child)
    assert child.height() > 0.8 * page.height(), (child.height(), page.height())
    page.close()


def test_a_fit_page_with_no_fill_child_still_top_justifies(qapp):
    """The negative case: small rows stay small, the spacer under them takes the room."""
    row = QLabel("a row")
    page = _fit_page(qapp, row)
    assert row.height() < 0.2 * page.height()
    page.close()
