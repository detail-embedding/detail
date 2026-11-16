"""INTERPRET's Find clusters line: ids typed, each ringed in its own colour (§22).

What a user types is parsed the way the app writes an id ("#1321") or a list of them; the tree
zooms to the smallest cluster holding every id; and each id keeps its colour while it is listed,
so taking one out never repaints the rest (the chip is how a ring is read back, and a chip that
changed colour under a ring that did not would name the wrong cluster).
"""

import pytest
from PyQt6.QtGui import QColor

from detail.gui import plot_theme
from detail.gui.cluster_find import FindClustersBar, common_cluster, parse_entries


class _View:
    """Root 10 over 8 and 9; 8 over 6 and 7; 6 over leaves 0, 1; 7 over 2, 3; 9 over 4, 5."""

    _children = {10: (8, 9), 8: (6, 7), 9: (4, 5), 6: (0, 1), 7: (2, 3)}
    root = 10

    def ancestors(self, node: int) -> list[int]:
        for parent, kids in self._children.items():
            if node in kids:
                return [*self.ancestors(parent), parent]
        return []


@pytest.mark.parametrize(
    ("text", "entries"),
    [
        ("1321", [1321]),
        ("1321, 1384", [1321, 1384]),
        ("#1321 #1384;902", [1321, 1384, 902]),
        ("  1321,,  1321 \n 5 ", [1321, 5]),
        # Anything that is not all ids is one label, its spaces kept to one.
        ("12, climb, rear   up", [12, "climb", "rear up"]),
        ("climb, Climb, CLIMB", ["climb"]),
        ("-4, 3.5", ["-4", "3.5"]),
        # A digit to str.isdigit that int() refuses: a label, not a ValueError in the slot.
        ("7, ², #³", [7, "²", "#³"]),
        ("", []),
    ],
)
def test_ids_and_labels_are_read_as_the_app_writes_them(text, entries):
    assert parse_entries(text) == entries


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [
        ([0], 0),
        ([0, 1], 6),
        ([0, 3], 8),
        ([0, 5], 10),
        ([2, 7], 7),
        ([6, 1, 3], 8),
        ([10, 4], 10),
        ([], None),
    ],
)
def test_the_zoom_is_the_smallest_cluster_holding_every_id(nodes, expected):
    assert common_cluster(_View(), nodes) == expected


#: The labels the bar's lookup knows, largest cluster first, as the page orders them.
_LABELS = {"climb": [8, 3], "walk": [5]}


@pytest.fixture
def bar(qapp):
    widget = FindClustersBar(unit="behavior", units="behaviors")
    widget.set_lookup(
        lambda entry: (
            _LABELS.get(entry.casefold(), [])
            if isinstance(entry, str)
            else ([entry] if 0 <= entry <= 10 else [])
        )
    )
    widget.resize(900, 60)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def _enter(bar, text: str) -> None:
    bar.field.setText(text)
    bar.submit()


def test_each_id_gets_its_own_colour_and_a_chip_in_it(bar):
    seen = []
    bar.found_changed.connect(seen.append)
    _enter(bar, "3, 7")
    assert list(bar.found()) == [3, 7]
    colours = [bar.found()[3].name(), bar.found()[7].name()]
    assert colours == [plot_theme.identity(0).name(), plot_theme.identity(1).name()]
    assert [chip.entry for chip in bar.chips()] == [3, 7]
    assert [chip.colour.name() for chip in bar.chips()] == colours
    assert seen[-1] == bar.found()
    assert bar.field.text() == "", "the field empties once its ids are listed"


def test_removing_one_id_keeps_every_other_colour(bar):
    _enter(bar, "3, 7, 9")
    before = {n: c.name() for n, c in bar.found().items()}
    bar.chips()[0].remove.click()
    assert list(bar.found()) == [7, 9]
    assert {n: c.name() for n, c in bar.found().items()} == {
        7: before[7],
        9: before[9],
    }
    _enter(bar, "1")
    assert bar.found()[1].name() == before[3], "a new id takes the first free colour"


def test_an_id_the_run_does_not_have_is_said_and_not_ringed(bar):
    _enter(bar, "4, 99")
    assert list(bar.found()) == [4]
    missing = [chip for chip in bar.chips() if chip.entry == 99][0]
    assert missing.colour is None
    assert "not in this run" in missing.name.text()
    assert not missing.name.isEnabled() and missing.name.toolTip()


def test_a_label_rings_every_cluster_it_names_in_one_colour(bar):
    _enter(bar, "4, Climb")
    found = bar.found()
    assert list(found) == [4, 8, 3]
    assert found[8].name() == found[3].name() == plot_theme.identity(1).name()
    label = bar.chips()[1]
    assert label.entry == "Climb" and label.nodes == [8, 3]
    assert label.name.text() == "Climb · 2", "a label on several clusters says how many"
    assert "#8" in label.name.toolTip() and "#3" in label.name.toolTip()


def test_a_band_never_takes_a_labels_colour(bar):
    """A labelled block is filled in its label's colour; a band in the same colour round it
    vanished into the fill. Reserved slots are skipped, and an entry whose colour a new label
    takes moves to a free one."""
    bar.set_reserved({0, 1})
    _enter(bar, "4, 6")
    assert [c.name() for c in bar.found().values()] == [
        plot_theme.identity(2).name(),
        plot_theme.identity(3).name(),
    ]
    bar.set_reserved({0, 1, 2})
    bar.refresh()
    assert bar.found()[6].name() == plot_theme.identity(3).name(), "6 keeps its colour"
    assert (
        bar.found()[4].name() == plot_theme.identity(4).name()
    ), "4 moves off the label's"
    assert bar.chips()[0].colour.name() == bar.found()[4].name()


def test_a_label_no_cluster_has_is_said_and_not_ringed(bar):
    _enter(bar, "run")
    [chip] = bar.chips()
    assert chip.colour is None and bar.found() == {}
    assert chip.name.text() == "run · no behavior has this label"
    assert not chip.name.isEnabled() and chip.name.toolTip()


def test_a_label_change_re_rings_without_zooming(bar):
    """``refresh`` (the page's labels changed) emits ``rings_changed``, never
    ``found_changed``: the page zooms on the latter, and a label landing never moves the tree.
    """
    _enter(bar, "walk")
    zooms, rings = [], []
    bar.found_changed.connect(zooms.append)
    bar.rings_changed.connect(rings.append)
    _LABELS["walk"] = [5, 2]
    try:
        bar.refresh()
    finally:
        _LABELS["walk"] = [5]
    assert zooms == [] and list(rings[-1]) == [5, 2]


def test_the_line_speaks_the_pages_word_for_a_cluster(bar):
    _enter(bar, "4, 99, run")
    assert bar.field.placeholderText().startswith("Find behaviors:")
    assert "behavior" in bar.field.toolTip() and "cluster" not in bar.field.toolTip()
    unknown = [chip for chip in bar.chips() if chip.entry == "run"][0]
    assert "behavior" in unknown.name.text() and "behavior" in unknown.name.toolTip()
    missing = [chip for chip in bar.chips() if chip.entry == 99][0]
    assert "behavior" in missing.name.toolTip()


def test_clicking_a_chip_picks_its_cluster_and_a_labels_largest(bar):
    picked = []
    bar.picked.connect(picked.append)
    _enter(bar, "4, 6, climb")
    bar.chips()[1].name.click()
    bar.chips()[2].name.click()
    assert picked == [6, 8]


def test_clear_empties_the_list_and_shows_only_while_ids_are_listed(bar):
    """Hidden, not greyed, with nothing listed: the idle zoom row carries the field alone."""
    assert bar.clear.isHidden()
    _enter(bar, "4, 6")
    assert not bar.clear.isHidden() and bar.clear.toolTip()
    seen = []
    bar.found_changed.connect(seen.append)
    bar.clear.click()
    assert bar.entries() == [] and bar.chips() == [] and seen == [{}]
    assert bar.clear.isHidden()


def test_a_new_tree_rechecks_which_ids_it_has(bar):
    _enter(bar, "4, 9")
    bar.set_lookup(
        lambda entry: [entry] if isinstance(entry, int) and entry < 5 else []
    )
    assert list(bar.found()) == [4]
    assert [chip.colour is None for chip in bar.chips()] == [False, True]
    assert isinstance(bar.found()[4], QColor)
