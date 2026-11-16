"""INTERPRET's **Find clusters** box: type cluster ids or labels, see each one ringed in the
tree (§22).

On screen it uses the page's word for a cluster: "Find behaviors" on a temporal run.

A real fit has hundreds of clusters, and the id a user carries in from somewhere else (a window's
title, a comparison's row, a note: "look at #1321 and #1384") names a block that can be one
pixel wide. A label ("climb") names every cluster the user gave it. This box takes either, gives
each entry a colour, rings its clusters' blocks in the tree in it, and zooms the tree onto **the
smallest cluster that holds every one of them**, so all of them are on screen at once and as wide
as they can be. One cluster zooms right onto it.

**What is typed.** Commas and semicolons separate entries. An entry made only of ids ("1321",
"#1321", "1321 1384") is those ids; anything else is a label, matched to each cluster's own label
whole and ignoring case, so a label with a space in it ("rear up") is one entry.

**A chip per entry, in the ring's colour.** The chip is how a ring is read back: the same colour
square beside "#1321" or "climb". A label on several clusters is one chip and one colour, and
says how many ("climb · 3"). A colour stays with its entry while it is in the list, so removing
one never repaints the others; a new entry takes the first colour not in use
(``plot_theme.identity``, the chart theme's categorical order) **and not a label's**
(``set_reserved``): the tree fills a labelled block in its label's colour, from the same order,
so a band in that colour round it vanished into its fill — and a label search always lands on
labelled blocks. A label added later that takes an entry's colour moves that entry, chip and
band together, to a free one.

**An entry the run does not have is said to be, not dropped.** Its chip reads "#99999 · not in
this run" or "climb · no behavior has this label", has no colour and no ring, and leaves the
zoom alone. Labelling a cluster while it is being found re-reads the labels (``refresh``): the
rings follow, and the tree does not move, as it never does when a label lands.

**Nothing here selects, filters or labels.** Clicking a chip selects its cluster (a label's
largest), as clicking its block would; the rings are only rings. The list is the page's for as
long as the page is open; a re-fit, which makes every id name a different cluster, clears it.

The parsing and the common cluster are plain functions (``parse_entries``, ``common_cluster``),
so they are tested without a widget.
"""

import html
import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from detail.gui import plot_theme
from detail.gui.flow import FlowLayout
from detail.gui.sizing import set_available, text_width

#: Object names, so a walk over a page's controls can tell these from the label field's
#: (both are an unnamed line edit otherwise) — the snapshot check allows them by name, since
#: finding reads the tree and writes nothing.
FIELD_NAME = "find_clusters_field"
FIND_NAME = "find_clusters_find"
CHIP_NAME = "find_clusters_chip"
CHIP_REMOVE_NAME = "find_clusters_remove"
CLEAR_NAME = "find_clusters_clear"

#: A chip's name button: its text plus this, so the macOS bezel does not crowd the text.
CHIP_PAD_PX = 20
#: A chip's remove button, "×" (as the readiness banner's dismiss, `gui/pages.py`).
REMOVE_PX = 24

#: What separates entries in the field. Not whitespace: a label may hold a space.
_ENTRY_SEPARATORS = re.compile(r"[,;]+")
#: One id, as the app writes it or bare. ``\d`` would take "²", which int() refuses.
_ID = re.compile(r"#?([0-9]+)")

#: An entry in the list: a cluster id, or a label as typed.
Entry = int | str


def find_tooltip(unit: str) -> str:
    """What the field and **Find** say about themselves, in the page's word for a cluster."""
    return (
        f"Type {unit} ids or labels, separated by commas (1321, 1384, climb), and press Enter: "
        f"each is ringed in the tree in its own colour, and the tree zooms to the smallest "
        f"{unit} holding all of them. Click one to open its clips."
    )


def parse_entries(text: str) -> list[Entry]:
    """The entries typed, in order, each once: ids as ``int``, labels as ``str``.

    An entry is what lies between commas or semicolons. One whose every word is an id ("1321",
    "#1321 1384") is those ids; any other is one label, its spaces kept to one ("rear  up" is
    "rear up"). A label repeated in another case is the same entry.
    """
    out: list[Entry] = []
    labels: set[str] = set()
    for part in _ENTRY_SEPARATORS.split(text):
        words = part.split()
        if not words:
            continue
        matches = [_ID.fullmatch(word) for word in words]
        if all(matches):
            for match in matches:
                node = int(match.group(1))
                if node not in out:
                    out.append(node)
            continue
        label = " ".join(words)
        if label.casefold() not in labels:
            labels.add(label.casefold())
            out.append(label)
    return out


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
    """What a chip reads: the id or label, with a count for a label on several clusters, or
    why it rings nothing."""
    name = f"#{entry}" if isinstance(entry, int) else entry
    if not nodes:
        if isinstance(entry, int):
            return f"{name} · not in this run"
        return f"{name} · no {unit} has this label"
    return f"{name} · {len(nodes)}" if len(nodes) > 1 else name


class FindChip(QWidget):
    """One entry in the list: its colour square, its name (a button that selects its cluster),
    and **×**. An entry the run does not have says so and selects nothing.
    """

    picked = pyqtSignal(int)
    removed = pyqtSignal(object)

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
        # As wide as its text, not the style's 80px button floor: the chips share the tree's
        # zoom row, so three must fit beside the field.
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
        self.remove = QPushButton("×")
        self.remove.setObjectName(CHIP_REMOVE_NAME)
        self.remove.setFixedWidth(REMOVE_PX)
        self.remove.setToolTip(f"Take {text.split(' · ')[0]} out of the list")
        self.remove.clicked.connect(lambda: self.removed.emit(self.entry))
        row.addWidget(self.remove)


class FindClustersBar(QWidget):
    """In INTERPRET's zoom row, between the label key and the zoom buttons: a field, the
    entries as chips, and **Clear**.

    ``found_changed`` carries ``{node: colour}`` for every cluster the list rings, in the order
    entered, when the user changed the list or the tree changed (``set_lookup``): the page rings
    them and zooms. ``rings_changed`` carries the same after ``refresh`` — the labels changed,
    and the tree must not move. ``picked`` is a chip clicked. ``unit`` / ``units`` are the
    page's words for a cluster (``WorkspaceTerms``).
    """

    found_changed = pyqtSignal(dict)
    rings_changed = pyqtSignal(dict)
    picked = pyqtSignal(int)

    def __init__(self, *, unit: str, units: str, parent=None) -> None:
        super().__init__(parent)
        self._unit = unit
        self._entries: list[Entry] = []
        self._slots: dict[Entry, int] = {}
        #: Colour slots the page's labels fill blocks with: never a band's.
        self._reserved: frozenset[int] = frozenset()
        self._nodes_of = lambda entry: [entry] if isinstance(entry, int) else []
        self._resolved: dict[Entry, list[int]] = {}
        self._chips: list[FindChip] = []
        self._row = FlowLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        # No caption of its own: the box shares the tree's zoom row (user, 2026-11-16: no
        # added height), so the field's placeholder says what it is.
        placeholder = f"Find {units}: ids or labels, e.g. 1321, climb"
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
        self.clear.clicked.connect(self.clear_all)
        self._row.addWidget(self.clear)
        self._sync_clear()

    # --- inputs --------------------------------------------------------------------------

    def set_lookup(self, nodes_of) -> None:
        """``nodes_of(entry) -> [node, ...]``: the clusters an id or label names in the run's
        tree, largest first; empty when it names none. A new tree: re-reads the list, and zooms.
        """
        self._nodes_of = nodes_of
        self._rebuild(self.found_changed)

    def set_reserved(self, slots) -> None:
        """The ``plot_theme.identity`` slots labels are using; takes effect at the next
        rebuild (the page calls ``refresh`` next)."""
        self._reserved = frozenset(int(s) for s in slots)

    def refresh(self) -> None:
        """The labels changed: re-read which clusters each entry names. Rings, never zooms."""
        if self._entries:
            self._rebuild(self.rings_changed)

    def add(self, entries) -> None:
        """Append ``entries`` (ids or labels, those not listed already) and say the result."""
        for entry in entries:
            if self._index(entry) is None:
                self._entries.append(entry)
        self._rebuild(self.found_changed)

    def submit(self) -> None:
        """The field's text, parsed: its entries join the list, and the field empties."""
        entries = parse_entries(self.field.text())
        self.field.clear()
        if entries:
            self.add(entries)

    def remove(self, entry: Entry) -> None:
        index = self._index(entry)
        if index is not None:
            del self._entries[index]
            self._rebuild(self.found_changed)

    def clear_all(self) -> None:
        """Empty the list, and the field with it."""
        self.field.clear()
        if self._entries:
            self._entries = []
            self._rebuild(self.found_changed)

    # --- outputs -------------------------------------------------------------------------

    def entries(self) -> list[Entry]:
        """Every entry in the list, the run's and not, in the order entered."""
        return list(self._entries)

    def found(self) -> dict[int, QColor]:
        """``{node: colour}`` for every cluster the list names, in the order entered; a node
        two entries name keeps the first one's colour."""
        out: dict[int, QColor] = {}
        for entry in self._entries:
            for node in self._resolved.get(entry, []):
                out.setdefault(node, plot_theme.identity(self._slots[entry]))
        return out

    def chips(self) -> list[FindChip]:
        return list(self._chips)

    # --- internals -----------------------------------------------------------------------

    def _index(self, entry: Entry) -> int | None:
        """Where ``entry`` is listed: an id by value, a label ignoring case."""
        for index, listed in enumerate(self._entries):
            if isinstance(entry, int) and isinstance(listed, int) and entry == listed:
                return index
            if (
                isinstance(entry, str)
                and isinstance(listed, str)
                and entry.casefold() == listed.casefold()
            ):
                return index
        return None

    def _assign_slots(self) -> None:
        """Keep each listed entry's colour while it names a cluster; give a newly found one
        the first colour not in use."""
        named = [e for e in self._entries if self._resolved[e]]
        self._slots = {
            e: s
            for e, s in self._slots.items()
            if e in named and s not in self._reserved
        }
        for entry in named:
            if entry not in self._slots:
                used = set(self._slots.values()) | self._reserved
                slot = 0
                while slot in used:
                    slot += 1
                self._slots[entry] = slot

    def _rebuild(self, signal) -> None:
        self._resolved = {e: [int(n) for n in self._nodes_of(e)] for e in self._entries}
        self._assign_slots()
        for chip in self._chips:
            self._row.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._chips = []
        # Chips go after Find and before Clear, in the order entered.
        at = self._row.indexOf(self.find) + 1
        for entry in self._entries:
            nodes = self._resolved[entry]
            colour = plot_theme.identity(self._slots[entry]) if nodes else None
            chip = FindChip(entry, nodes, colour, unit=self._unit, parent=self)
            chip.picked.connect(self.picked)
            chip.removed.connect(self.remove)
            self._row.insertWidget(at, chip)
            at += 1
            self._chips.append(chip)
        self._sync_clear()
        signal.emit(self.found())

    def _sync_clear(self) -> None:
        """**Clear** shows only while entries are listed: an idle zoom row carries the field
        alone."""
        self.clear.setVisible(bool(self._entries))
        self.clear.setToolTip("Take everything out of the list")
