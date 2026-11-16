"""INTERPRET's **Find clusters** line: type cluster ids, see each one ringed in the tree (§22).

On screen it uses the page's word for a cluster: "Find behaviors" on a temporal run.

A real fit has hundreds of clusters, and the id a user carries in from somewhere else (a window's
title, a comparison's row, a note: "look at #1321 and #1384") names a block that can be one
pixel wide. This line takes those ids, gives each one a colour, rings each cluster's block in
the tree in it, and zooms the tree onto **the smallest cluster that holds every one of them**,
so all of them are on screen at once and as wide as they can be. One id zooms right onto it.

**A chip per id, in the ring's colour.** The chip is how a ring is read back: the same colour
square beside "#1321". A colour stays with its id while it is in the list, so removing one id
never repaints the others; a new id takes the first colour not in use (``plot_theme.identity``,
the chart theme's categorical order).

**An id the run does not have is said to be, not dropped.** Its chip reads "#99999 · not in this
run", has no colour and no ring, and leaves the zoom alone. Text that is not an id at all
("abc") is said once on the line and never becomes a chip.

**Nothing here selects, filters or labels.** Clicking a chip's id selects that cluster, as
clicking its block would; the rings are only rings. The list is the page's for as long as the
page is open; a re-fit, which makes every id name a different cluster, clears it.

The parsing and the common cluster are plain functions (``parse_ids``, ``common_cluster``), so
they are tested without a widget.
"""

import html
import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from detail.gui import plot_theme
from detail.gui import status_color
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

#: What separates ids in the field: commas, semicolons and whitespace, in any mix.
_SEPARATORS = re.compile(r"[\s,;]+")


def find_tooltip(unit: str) -> str:
    """What the field and **Find** say about themselves, in the page's word for a cluster."""
    return (
        f"Type one or more {unit} ids (1321, 1384) and press Enter: each is ringed in the tree "
        f"in its own colour, and the tree zooms to the smallest {unit} holding all of them. "
        "Click an id to open its clips."
    )


def parse_ids(text: str) -> tuple[list[int], list[str]]:
    """``(ids, rejected)`` from what was typed: ids in the order given, each once.

    An id may carry the ``#`` the app writes before one ("#1321"). Anything else that is not a
    non-negative whole number is returned in ``rejected``, as typed.
    """
    ids: list[int] = []
    rejected: list[str] = []
    for token in _SEPARATORS.split(text.strip()):
        if not token:
            continue
        digits = token[1:] if token.startswith("#") else token
        if not digits.isdigit():
            rejected.append(token)
            continue
        node = int(digits)
        if node not in ids:
            ids.append(node)
    return ids, rejected


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


class FindChip(QWidget):
    """One id in the list: its colour square, the id (a button that selects the cluster), and
    **Remove**. An id the run does not have reads "not in this run" and selects nothing.
    """

    picked = pyqtSignal(int)
    removed = pyqtSignal(int)

    def __init__(
        self, node: int, colour: QColor | None, *, unit: str, parent=None
    ) -> None:
        super().__init__(parent)
        self.node = int(node)
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
        self.name = QPushButton(f"#{self.node}")
        self.name.setObjectName(CHIP_NAME)
        self.name.clicked.connect(lambda: self.picked.emit(self.node))
        row.addWidget(self.name)
        if colour is None:
            self.name.setText(f"#{self.node} · not in this run")
            set_available(
                self.name, False, f"This run's tree has no {unit} with this id"
            )
        else:
            self.name.setToolTip(f"Open {unit} #{self.node}'s clips")
        self.remove = QPushButton("Remove")
        self.remove.setObjectName(CHIP_REMOVE_NAME)
        self.remove.setToolTip(f"Take #{self.node} out of the list")
        self.remove.clicked.connect(lambda: self.removed.emit(self.node))
        row.addWidget(self.remove)


class FindClustersBar(QWidget):
    """The line above INTERPRET's tree: a field, the ids entered as chips, and **Clear**.

    ``found_changed`` carries ``{node: colour}`` for every id the run has, in the order entered;
    the page rings them and zooms. ``picked`` is a chip's id clicked. ``unit`` / ``units`` are
    the page's words for a cluster (``WorkspaceTerms``): a temporal run's line says "Find
    behaviors", and a default would have it say "clusters" on a screen that never does.
    """

    found_changed = pyqtSignal(dict)
    picked = pyqtSignal(int)

    def __init__(self, *, unit: str, units: str, parent=None) -> None:
        super().__init__(parent)
        self._unit = unit
        self._ids: list[int] = []
        self._slots: dict[int, int] = {}
        self._valid = lambda _node: True
        self._chips: list[FindChip] = []
        self._row = FlowLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self.caption = QLabel(f"Find {units}")
        self.caption.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._row.addWidget(self.caption)
        self.field = QLineEdit()
        self.field.setPlaceholderText("ids, e.g. 1321, 1384")
        self.field.setObjectName(FIELD_NAME)
        self.field.setToolTip(find_tooltip(unit))
        self.field.setClearButtonEnabled(True)
        self.field.setMinimumWidth(text_width("ids, e.g. 1321, 1384") + 40)
        self.field.returnPressed.connect(self.submit)
        self._row.addWidget(self.field)
        self.find = QPushButton("Find")
        self.find.setObjectName(FIND_NAME)
        self.find.setToolTip(find_tooltip(unit))
        self.find.clicked.connect(self.submit)
        self._row.addWidget(self.find)
        self.note = QLabel("")
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_color.set_label_color(self.note, status_color.WARN)
        self.note.setVisible(False)
        self._row.addWidget(self.note)
        self._row.addStretch(1)
        self.clear = QPushButton("Clear")
        self.clear.setObjectName(CLEAR_NAME)
        self.clear.clicked.connect(self.clear_all)
        self._row.addWidget(self.clear)
        self._sync_clear()

    # --- inputs --------------------------------------------------------------------------

    def set_valid(self, valid) -> None:
        """``valid(node) -> bool``: whether the run's tree has that id. Re-checks the list."""
        self._valid = valid
        self._rebuild()

    def add_ids(self, ids) -> None:
        """Append ``ids`` (those not listed already) and say the result."""
        for node in ids:
            node = int(node)
            if node in self._ids:
                continue
            self._ids.append(node)
        self._rebuild()

    def submit(self) -> None:
        """The field's text, parsed: its ids join the list, and the field empties."""
        ids, rejected = parse_ids(self.field.text())
        if rejected:
            self.note.setText(
                f"Not a {self._unit} id: "
                + ", ".join(rejected[:3])
                + (" …" if len(rejected) > 3 else "")
            )
        self.note.setVisible(bool(rejected))
        self.field.clear()
        if ids:
            self.add_ids(ids)

    def remove_id(self, node: int) -> None:
        if int(node) in self._ids:
            self._ids.remove(int(node))
            self._rebuild()

    def clear_all(self) -> None:
        """Empty the list, and the field and its note with it."""
        self.field.clear()
        self.note.setVisible(False)
        if self._ids:
            self._ids = []
            self._rebuild()

    # --- outputs -------------------------------------------------------------------------

    def ids(self) -> list[int]:
        """Every id in the list, the run's and not, in the order entered."""
        return list(self._ids)

    def found(self) -> dict[int, QColor]:
        """``{node: colour}`` for the ids the run has, in the order entered."""
        return {node: self._colour(node) for node in self._ids if self._valid(node)}

    def chips(self) -> list[FindChip]:
        return list(self._chips)

    # --- internals -----------------------------------------------------------------------

    def _colour(self, node: int) -> QColor:
        return plot_theme.identity(self._slots[node])

    def _assign_slots(self) -> None:
        """Keep each listed valid id's colour; give a new one the first colour not in use."""
        valid = [n for n in self._ids if self._valid(n)]
        self._slots = {n: s for n, s in self._slots.items() if n in valid}
        for node in valid:
            if node not in self._slots:
                used = set(self._slots.values())
                slot = 0
                while slot in used:
                    slot += 1
                self._slots[node] = slot

    def _rebuild(self) -> None:
        self._assign_slots()
        for chip in self._chips:
            self._row.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._chips = []
        # Chips go after the note and before the stretch, in the order entered.
        at = self._row.indexOf(self.note) + 1
        for node in self._ids:
            colour = self._colour(node) if self._valid(node) else None
            chip = FindChip(node, colour, unit=self._unit, parent=self)
            chip.picked.connect(self.picked)
            chip.removed.connect(self.remove_id)
            self._row.insertWidget(at, chip)
            at += 1
            self._chips.append(chip)
        self._sync_clear()
        self.found_changed.emit(self.found())

    def _sync_clear(self) -> None:
        set_available(self.clear, bool(self._ids), "No cluster ids are listed")
        if self._ids:
            self.clear.setToolTip("Take every id out of the list")
