"""INTERPRET's Find clusters box: one id or label searched, its clusters ringed (§22).

What a user types is one entry, an id as the app writes it ("#1321") or else a label; a label on
several clusters rings them all in one colour, and the tree zooms to the smallest cluster
holding them. A new search replaces the last. The ring's colour is never a label's, which fills
the block it rings.
"""

import pytest
from PyQt6.QtGui import QColor

from detail.gui import plot_theme
from detail.gui.cluster_find import FindClustersBar, common_cluster, parse_entry


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
    ("text", "entry"),
    [
        ("1321", 1321),
        ("  #1321 ", 1321),
        ("climb", "climb"),
        ("rear   up", "rear up"),
        # One search: commas and spaces are part of a label, not separators.
        ("1321, 1384", "1321, 1384"),
        ("1321 1384", "1321 1384"),
        ("-4", "-4"),
        # A digit to str.isdigit that int() refuses: a label, not a ValueError in the slot.
        ("²", "²"),
        ("   ", None),
    ],
)
def test_one_id_or_one_label_is_read_as_the_app_writes_it(text, entry):
    assert parse_entry(text) == entry


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


def _lookup(entry):
    if isinstance(entry, str):
        return list(_LABELS.get(entry.casefold(), []))
    return [entry] if 0 <= entry <= 10 else []


@pytest.fixture
def bar(qapp):
    widget = FindClustersBar(unit="behavior", units="behaviors")
    widget.set_lookup(_lookup)
    widget.resize(900, 60)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def _enter(bar, text: str) -> None:
    bar.field.setText(text)
    bar.submit()


def test_an_id_is_ringed_in_the_first_colour_and_its_chip_says_it(bar):
    seen = []
    bar.found_changed.connect(seen.append)
    _enter(bar, "#7")
    assert bar.entry() == 7
    assert {n: c.name() for n, c in bar.found().items()} == {
        7: plot_theme.label_identity(0).name()
    }
    assert bar.chip().name.text() == "#7"
    assert bar.chip().colour.name() == plot_theme.label_identity(0).name()
    assert seen[-1] == bar.found()
    assert bar.field.text() == "", "the field empties once searched"


def test_a_new_search_replaces_the_last_one(bar):
    """One search at a time (user, 2026-11-16): the last one's rings go."""
    _enter(bar, "climb")
    _enter(bar, "7")
    assert bar.entry() == 7 and list(bar.found()) == [7]
    assert bar.chip().entry == 7
    _enter(bar, "   ")
    assert bar.entry() == 7, "a search of nothing keeps the last"


def test_a_label_rings_every_cluster_it_names_in_one_colour(bar):
    _enter(bar, "Climb")
    found = bar.found()
    assert list(found) == [8, 3]
    assert found[8].name() == found[3].name()
    chip = bar.chip()
    assert chip.entry == "Climb" and chip.nodes == [8, 3]
    assert chip.name.text() == "Climb · 2", "a label on several clusters says how many"
    assert "#8" in chip.name.toolTip() and "#3" in chip.name.toolTip()


def test_a_band_never_takes_a_labels_colour(bar):
    """A labelled block is filled in its label's colour; a band in the same colour round it
    vanished into the fill. Reserved slots are skipped, and a new label that takes the
    search's colour moves it."""
    bar.set_reserved({0, 1})
    _enter(bar, "4")
    assert bar.found()[4].name() == plot_theme.label_identity(2).name()
    bar.set_reserved({0, 1, 2})
    bar.refresh()
    assert bar.found()[4].name() == plot_theme.label_identity(3).name()
    assert bar.chip().colour.name() == bar.found()[4].name()


def test_an_id_the_run_does_not_have_is_said_and_not_ringed(bar):
    _enter(bar, "99")
    assert bar.found() == {}
    chip = bar.chip()
    assert chip.colour is None and chip.name.text() == "#99 · not in this run"
    assert not chip.name.isEnabled() and chip.name.toolTip()


def test_a_label_no_cluster_has_is_said_and_not_ringed(bar):
    _enter(bar, "run")
    chip = bar.chip()
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


def test_the_box_speaks_the_pages_word_for_a_cluster(bar):
    assert bar.field.placeholderText().startswith("Find behaviors:")
    assert "behavior" in bar.field.toolTip() and "cluster" not in bar.field.toolTip()
    _enter(bar, "run")
    assert "behavior" in bar.chip().name.text()
    assert "behavior" in bar.chip().name.toolTip()


def test_clicking_the_chip_picks_its_cluster_and_a_labels_largest(bar):
    picked = []
    bar.picked.connect(picked.append)
    _enter(bar, "6")
    bar.chip().name.click()
    _enter(bar, "climb")
    bar.chip().name.click()
    assert picked == [6, 8]


def test_clear_ends_the_search_and_shows_only_during_one(bar):
    """Hidden, not greyed, with no search: the idle zoom row carries the field alone."""
    assert bar.clear.isHidden()
    _enter(bar, "4")
    assert not bar.clear.isHidden() and bar.clear.toolTip()
    seen = []
    bar.found_changed.connect(seen.append)
    bar.clear.click()
    assert bar.entry() is None and bar.chip() is None and seen == [{}]
    assert bar.clear.isHidden()


def test_a_new_tree_rechecks_what_the_search_names(bar):
    _enter(bar, "9")
    bar.set_lookup(
        lambda entry: [entry] if isinstance(entry, int) and entry < 5 else []
    )
    assert bar.found() == {} and bar.chip().colour is None
    assert isinstance(plot_theme.identity(0), QColor)
