"""Panes pack at the top, the same as pages do.

The class is `gui_design` §3's top-justify rule, which was written for a content-area **page**
and guarded only on pages. `assert_top_justified` already recurses into every descendant, so a
splitter half and a tab body were always in scope for the *check* — what was missing was any
corpus containing one. Its two call sites were the eleven pages in
`test_smoke_construct.py`, none of which holds a splitter or a tab widget, and `AuditTab`,
whose three panes were therefore the only ones measured anywhere in the package, and only
because `findChildren` happened to reach them.

Twelve splitters and ten tab bodies had never been looked at. Two were spreading: a
`quick_segment` header laid out at 175px against a 15px hint and growing with the window, and a
`cluster_workspace` ethogram title at 69px. Neither shows up in a screenshot as obviously wrong
— it reads as generous spacing — which is why it needs measuring rather than looking.
"""

import pytest

pytest.importorskip("PyQt6")

from tests.gui.layout_checks import assert_panes_top_justify, owns_a_pane  # noqa: E402

# The cluster workspace's panes are asserted in `test_cluster_workspace.py`, where the fit
# fixture that builds one lives; `quick_segment`'s sampled grid in `test_tools.py`
# (`test_the_sampled_grid_packs_its_rows_at_the_top`) — the corpus is wherever a surface can be
# built, not one module. This said `test_quick_segment_ui.py` for months, a file that never
# existed, and the grid's stretch could be deleted with every test green.


def _grid(qapp, *, beside_a_tall_cell: bool):
    from PyQt6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

    host = QWidget()
    grid = QGridLayout(host)
    for row in range(3):
        grid.addWidget(QLabel(f"row {row}"), row, 0)
        if beside_a_tall_cell:
            cell = QWidget()  # a thumbnail: a floor, and it takes what the row is given
            cell.setMinimumHeight(200)
            cell.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
            )
            grid.addWidget(cell, row, 1)
    return host


def test_a_grid_label_beside_a_tall_cell_is_not_spread(qapp):
    """A grid row is as tall as its tallest cell, so a row label beside a 200px thumbnail is
    200px by construction — the grid's form of a label in a horizontal form row."""
    from tests.gui.layout_checks import spread_labels

    assert spread_labels(_grid(qapp, beside_a_tall_cell=True), size=(600, 660)) == []


def test_the_grid_excuse_would_catch_a_grid_of_labels_alone(qapp):
    """The excuse is keyed on a sibling that insists on the height, not on being in a grid:
    a grid of labels with no stretch row still spreads its slack, and must still be caught.
    """
    from tests.gui.layout_checks import spread_labels

    grown = spread_labels(_grid(qapp, beside_a_tall_cell=False), size=(600, 1000))
    assert [text for text, _hint, _h in grown] == ["row 0", "row 1", "row 2"]
