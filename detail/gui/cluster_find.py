"""INTERPRET's **Find clusters** box: type a cluster id or a label, see it ringed in the tree
(§22).

On screen it uses the page's word for a cluster: "Find behaviors" on a temporal run.

A real fit has hundreds of clusters, and the id a user carries in from somewhere else (a window's
title, a comparison's row, a note: "look at #1321") names a block that can be one pixel wide. A
label ("climb") names every cluster the user gave it. This box takes one or the other, rings its
clusters' blocks in the tree in one colour, and zooms the tree onto **the smallest cluster that
holds all of them**, so they are on screen at once and as wide as they can be. A single cluster
zooms right onto it.

**One search at a time** (user, 2026-11-16). What is typed is one entry: an id ("1321",
"#1321") or else a label, matched to each cluster's own label whole and ignoring case — commas
and spaces included, so any label can be typed. A new search replaces the last; **Clear** ends
it and puts back the view from before the first search (the page keeps that view).

**A chip, in the ring's colour.** The chip is how the ring is read back: the colour square beside
"#1321" or "climb". A label on several clusters says how many ("climb · 3"). The colour is the
first ``plot_theme.label_identity`` slot **not a label's** (``set_reserved``): the tree fills a labelled
block in its label's colour, from the same order, so a band in that colour round it vanished
into its fill — and a label search always lands on labelled blocks.

**An entry the run does not have is said to be.** Its chip reads "#99999 · not in this run" or
"climb · no behavior has this label", has no colour and no ring, and leaves the zoom alone.
Labelling a cluster while its label is searched re-reads it (``refresh``): the rings follow, and
the tree does not move, as it never does when a label lands.

**Nothing here selects, filters or labels.** Clicking the chip selects its cluster (a label's
largest), as clicking its block would; the rings are only rings. A re-fit, which makes every id
name a different cluster, clears the search.

The parsing and the common cluster are plain functions (``parse_entry``, ``common_cluster``),
so they are tested without a widget.
"""

import html
import re

from detail.gui import plot_theme
from detail.gui.flow import FlowLayout
from detail.gui.qt import (
    QColor,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.sizing import set_available, text_width

#: Object names, so a walk over a page's controls can tell these from the label field's
#: (both are an unnamed line edit otherwise) — the snapshot check allows them by name, since
#: finding reads the tree and writes nothing.
FIELD_NAME = "find_clusters_field"
FIND_NAME = "find_clusters_find"
CHIP_NAME = "find_clusters_chip"
CLEAR_NAME = "find_clusters_clear"

#: The chip's button: its text plus this, so the macOS bezel does not crowd the text.
CHIP_PAD_PX = 20

#: One id, as the app writes it or bare. ``\d`` would take "²", which int() refuses.
_ID = re.compile(r"#?([0-9]+)")

#: What is searched: a cluster id, or a label as typed.
Entry = int | str


def find_tooltip(unit: str) -> str:
    """What the field and **Find** say about themselves, in the page's word for a cluster."""
    return (
        f"Type a {unit} id (1321) or a label (climb) and press Enter: its {unit}s are ringed "
        f"in the tree, and the tree zooms to the smallest {unit} holding them. A new search "
        "replaces the last; Clear goes back to the view you had. Click the result to open its "
        "clips."
    )


def parse_entry(text: str) -> Entry | None:
    """What was typed: an id (``int``) if it is one, as the app writes it ("#1321") or bare;
    otherwise the label, its runs of spaces kept to one; ``None`` for nothing."""
    words = text.split()
    if not words:
        return None
    label = " ".join(words)
    match = _ID.fullmatch(label)
    return int(match.group(1)) if match else label


def common_cluster(view, nodes) -> int | None:
    """The smallest cluster of ``view``'s tree that holds every node in ``nodes``: the node
    itself for one, their deepest shared ancestor for several, ``None`` for none."""
    nodes = [int(n) for n in nodes]
    if not nodes:
        return None
    shared = [*view.ancestors(nodes[0]), nodes[0]]
    for node in nodes[1:]:
        path = [*view.ancestors(node), node]
        depth = 0
        while (
            depth < len(shared) and depth < len(path) and shared[depth] == path[depth]
        ):
            depth += 1
        shared = shared[:depth]
    return shared[-1] if shared else int(view.root)


def chip_text(entry: Entry, nodes: list[int], unit: str) -> str:
    """What the chip reads: the id or label, with a count for a label on several clusters, or
    why it rings nothing."""
    name = f"#{entry}" if isinstance(entry, int) else entry
    if not nodes:
        if isinstance(entry, int):
            return f"{name} · not in this run"
        return f"{name} · no {unit} has this label"
    return f"{name} · {len(nodes)}" if len(nodes) > 1 else name


class FindChip(QWidget):
    """The search's result: its colour square and its name, a button that selects its cluster.
    An entry the run does not have says so and selects nothing.
    """

    picked = Signal(int)

    def __init__(
        self,
        entry: Entry,
        nodes: list[int],
        colour: QColor | None,
        *,
        unit: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.entry = entry
        self.nodes = list(nodes)
        self.colour = colour
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.swatch = QLabel()
        self.swatch.setTextFormat(Qt.TextFormat.RichText)
        self.swatch.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        if colour is not None:
            self.swatch.setText(
                f'<span style="color:{QColor(colour).name()}">&#9632;</span>'
            )
        else:
            self.swatch.setText(html.escape("·"))
        row.addWidget(self.swatch)
        text = chip_text(entry, self.nodes, unit)
        self.name = QPushButton(text)
        self.name.setObjectName(CHIP_NAME)
        # As wide as its text, not the style's 80px button floor: it shares the zoom row.
        self.name.setFixedWidth(text_width(text) + CHIP_PAD_PX)
        self.name.clicked.connect(lambda: self.picked.emit(self.nodes[0]))
        row.addWidget(self.name)
        if not self.nodes:
            reason = (
                f"This run's tree has no {unit} with this id"
                if isinstance(entry, int)
                else f"No {unit} in this run has this label"
            )
            set_available(self.name, False, reason)
        elif len(self.nodes) > 1:
            ids = ", ".join(f"#{n}" for n in self.nodes)
            self.name.setToolTip(
                f"{len(self.nodes)} {unit}s have this label ({ids}); click to open the "
                "largest one's clips"
            )
        else:
            self.name.setToolTip(f"Open {unit} #{self.nodes[0]}'s clips")


class FindClustersBar(QWidget):
    """In INTERPRET's zoom row, between the label key and the zoom buttons: a field, the
    search's chip, and **Clear**.

    ``found_changed`` carries ``{node: colour}`` for every cluster the search rings when the
    user searched or cleared, or the tree changed (``set_lookup``): the page rings them and
    zooms. ``rings_changed`` carries the same after ``refresh`` — the labels changed, and the
    tree must not move. ``picked`` is the chip clicked. ``unit`` / ``units`` are the page's
    words for a cluster (``WorkspaceTerms``).
    """

    found_changed = Signal(dict)
    rings_changed = Signal(dict)
    picked = Signal(int)

    def __init__(self, *, unit: str, units: str, parent=None) -> None:
        super().__init__(parent)
        self._unit = unit
        self._entry: Entry | None = None
        self._nodes: list[int] = []
        #: Colour slots the page's labels fill blocks with: never a band's.
        self._reserved: frozenset[int] = frozenset()
        self._nodes_of = lambda entry: [entry] if isinstance(entry, int) else []
        self._chip: FindChip | None = None
        self._row = FlowLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        # No caption of its own: the box shares the tree's zoom row (user, 2026-11-16: no
        # added height), so the field's placeholder says what it is.
        placeholder = f"Find {units}: an id or a label"
        self.field = QLineEdit()
        self.field.setPlaceholderText(placeholder)
        self.field.setObjectName(FIELD_NAME)
        self.field.setToolTip(find_tooltip(unit))
        self.field.setClearButtonEnabled(True)
        self.field.setMinimumWidth(text_width(placeholder) + 40)
        self.field.returnPressed.connect(self.submit)
        self._row.addWidget(self.field)
        self.find = QPushButton("Find")
        self.find.setObjectName(FIND_NAME)
        self.find.setToolTip(find_tooltip(unit))
        self.find.clicked.connect(self.submit)
        self._row.addWidget(self.find)
        self.clear = QPushButton("Clear")
        self.clear.setObjectName(CLEAR_NAME)
        self.clear.setToolTip("End the search and go back to the view you had")
        self.clear.clicked.connect(self.clear_all)
        self._row.addWidget(self.clear)
        self._sync_clear()

    # --- inputs --------------------------------------------------------------------------

    def set_lookup(self, nodes_of) -> None:
        """``nodes_of(entry) -> [node, ...]``: the clusters an id or label names in the run's
        tree, largest first; empty when it names none. A new tree: re-reads the search, and
        zooms."""
        self._nodes_of = nodes_of
        self._rebuild(self.found_changed)

    def set_reserved(self, slots) -> None:
        """The ``plot_theme.label_identity`` slots labels are using; takes effect at the next
        rebuild (the page calls ``refresh`` next)."""
        self._reserved = frozenset(int(s) for s in slots)

    def refresh(self) -> None:
        """The labels changed: re-read which clusters the search names. Rings, never zooms."""
        if self._entry is not None:
            self._rebuild(self.rings_changed)

    def search(self, entry: Entry) -> None:
        """Search for ``entry`` (an id or a label), replacing any search before it."""
        self._entry = entry
        self._rebuild(self.found_changed)

    def submit(self) -> None:
        """The field's text as a search; the field empties. Nothing typed changes nothing."""
        entry = parse_entry(self.field.text())
        self.field.clear()
        if entry is not None:
            self.search(entry)

    def clear_all(self) -> None:
        """End the search, and empty the field."""
        self.field.clear()
        if self._entry is not None:
            self._entry = None
            self._rebuild(self.found_changed)

    # --- outputs -------------------------------------------------------------------------

    def entry(self) -> Entry | None:
        """What is searched for, found or not; ``None`` with no search."""
        return self._entry

    def found(self) -> dict[int, QColor]:
        """``{node: colour}`` for every cluster the search names, largest first."""
        colour = plot_theme.label_identity(self._slot())
        return {node: colour for node in self._nodes}

    def chip(self) -> FindChip | None:
        return self._chip

    # --- internals -----------------------------------------------------------------------

    def _slot(self) -> int:
        """The first colour slot the labels are not using."""
        slot = 0
        while slot in self._reserved:
            slot += 1
        return slot

    def _rebuild(self, signal) -> None:
        self._nodes = (
            [] if self._entry is None else [int(n) for n in self._nodes_of(self._entry)]
        )
        if self._chip is not None:
            self._row.removeWidget(self._chip)
            self._chip.setParent(None)
            self._chip.deleteLater()
            self._chip = None
        if self._entry is not None:
            colour = plot_theme.label_identity(self._slot()) if self._nodes else None
            self._chip = FindChip(
                self._entry, self._nodes, colour, unit=self._unit, parent=self
            )
            self._chip.picked.connect(self.picked)
            # After Find, before Clear.
            self._row.insertWidget(self._row.indexOf(self.find) + 1, self._chip)
        self._sync_clear()
        signal.emit(self.found())

    def _sync_clear(self) -> None:
        """**Clear** shows only during a search: an idle zoom row carries the field alone."""
        self.clear.setVisible(self._entry is not None)
