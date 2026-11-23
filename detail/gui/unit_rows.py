"""The rows themselves: name, bar, status — one per part of a run.

Split from `run_units.py` so the derivation is testable without a `QApplication`: what a run is
made of is arithmetic over a record, and only the drawing needs Qt.

**This is redrawn about twice a second for the life of a run**, which is what shapes it. Two
things follow, and both were bugs before they were rules:

- A row is **updated in place** when only its numbers changed. Tearing down sixty widgets and
  building sixty more twice a second is the cost this package has already measured once
  (on the query confidence strip, since removed), and it throws away the expander
  state every time besides.
- When the shape *does* change, the old widgets are **hidden before** `deleteLater`. Deletion is
  deferred to the event loop, so a row removed from the layout is still drawn at its old
  geometry until that runs — which put two rows of text on top of each other on the first real
  render of this widget.
"""

from __future__ import annotations

from detail.gui.qt import (
    QColor,
    QGridLayout,
    QLabel,
    QPalette,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QWidget,
    Qt,
    disconnect_if_connected,
)
from detail.gui.run_units import DONE, FAILED, QUEUED, Unit, group_units
from detail.gui.sizing import lines, mono_font
from detail.gui.status_color import ERROR, set_label_color

#: How tall a row is — **every** part of it, not just the bar.
#:
#: In lines rather than pixels because a user who has asked for larger text gets larger rows; a
#: pixel constant is correct on the screen it was picked on and nowhere else.
#:
#: Applied to the label, the button, the bar and the status alike, because a row whose height is
#: the *tallest thing in it* is a row that changes height when a different thing is in it. A
#: `QToolButton` wants 28px where a `QLabel` wants 14, so a count row was twice the height of a
#: recording row — and every redraw passed through a state with the button shown, which made the
#: whole page below jump and come back one frame later. Measured off a screen recording: a 7px
#: hop on every single update, for the life of the run.
ROW_LINES = 1.4

#: The marker in front of a name. Text, not icons — the package has no icon set.
MARKS = {DONE: "✓", "running": "▶", QUEUED: " ", FAILED: "✕"}

#: Width of the name and status columns, in characters of the monospace face.
#:
#: **Fixed, and that is the whole point.** These columns hold text that changes on every update —
#: `running · 0%` becomes `detecting · 47% · ~19m00s left` becomes `~1h55m left` — and a column
#: sized to its content resizes with it. The bar column takes the remainder, so the bar's right
#: edge then moves on every tick: measured 659 → 652 → 529 → 558px across four consecutive
#: updates of one run. It reads as the bar twitching, and it is the reason a progress display
#: has to be laid out in fixed columns rather than fitted ones.
#:
#: In characters rather than pixels so they follow the interface font, and generous enough for
#: the longest status any surface writes; anything longer elides rather than pushing.
#:
#: The fixed width and the elision each hold the layout still on their own — reverting either
#: alone leaves it steady, and only removing both brings the twitch back. They are kept as a
#: pair because they answer different halves: the width makes the geometry deterministic
#: whatever the text, and the elision is what stops a long recording name being sliced without
#: a tooltip to recover it.
NAME_CHARS = 22
STATUS_CHARS = 32


def _column_px(chars: int) -> int:
    from detail.gui.qt import QFontMetrics

    return QFontMetrics(mono_font()).horizontalAdvance("0" * chars)


def _elided(label, text: str) -> None:
    """Set text that cannot widen its column.

    A fixed-width label still reports the full string through `sizeHint`, and a grid will happily
    hand it more than the minimum — so the text itself is shortened to what fits.
    """
    from detail.gui.qt import QFontMetrics

    metrics = QFontMetrics(label.font())
    label.setText(
        metrics.elidedText(text, Qt.TextElideMode.ElideRight, label.width() or 1)
    )
    label.setToolTip(text if label.text() != text else "")


def row_px() -> int:
    """The one height every widget in a row is pinned to."""
    return round(lines(ROW_LINES))


def _muted(widget) -> None:
    """Grey a label that is reference rather than news."""
    pal = widget.palette()
    colour = pal.color(QPalette.ColorRole.WindowText)
    pal.setColor(
        QPalette.ColorRole.WindowText,
        QColor(colour.red(), colour.green(), colour.blue(), 150),
    )
    widget.setPalette(pal)


def _plain(widget) -> None:
    widget.setPalette(widget.style().standardPalette())


def style_bar(bar: QProgressBar, unit: Unit) -> None:
    """Set a bar's value and the fill colour its state calls for.

    Only two states set a colour, and both are the package's existing rule rather than
    decoration: **failed is red**, which is what red is for here, and **done is the window's own
    text colour at reduced alpha**, which reads as spent rather than as another live thing. A
    running row is left at the system highlight — the default is the right answer, and
    overriding it would be this app inventing a palette every other widget has avoided.
    """
    if unit.state == FAILED:
        # A failed unit draws a *full* bar, not an empty one: empty reads as "not started", and
        # the row a person most needs to see would be the least visible thing on the page.
        bar.setValue(1000)
    else:
        bar.setValue(round(min(max(unit.fraction, 0.0), 1.0) * 1000))
    pal = bar.style().standardPalette()
    if unit.state == FAILED:
        # The app's one error colour rather than this file's own red: a second red is a second
        # thing a reader has to learn means the same as the first.
        pal.setColor(QPalette.ColorRole.Highlight, ERROR)
    elif unit.state == DONE:
        base = pal.color(QPalette.ColorRole.WindowText)
        pal.setColor(
            QPalette.ColorRole.Highlight,
            QColor(base.red(), base.green(), base.blue(), 110),
        )
    bar.setPalette(pal)


class _Row:
    """The three widgets of one row, kept together so they can be reused."""

    def __init__(self, grid: QGridLayout, parent: QWidget, row: int) -> None:
        # Parented at construction, never reparented afterwards: a widget built with no parent
        # is a top-level window, and macOS gives each one native backing (this package measured
        # seconds for a few hundred of them).
        self.name = QLabel(parent)
        self.name.setFont(mono_font())
        self.button = QToolButton(parent)
        self.button.setAutoRaise(True)
        self.button.setFont(mono_font())
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bar = QProgressBar(parent)
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(row_px())
        self.bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.status = QLabel(parent)
        self.status.setFont(mono_font())
        # Left-aligned, so the text starts immediately after the bar it describes. Right-aligned
        # against a fixed column, a short status ("29m00s") floats at the far edge with 200px of
        # nothing between it and its own bar, reading as though it belongs to neither.
        self.status.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        name_px, status_px = _column_px(NAME_CHARS), _column_px(STATUS_CHARS)
        self.name.setFixedWidth(name_px)
        self.status.setFixedWidth(status_px)
        for w in (self.name, self.button, self.status):
            w.setFixedHeight(row_px())
        grid.addWidget(self.name, row, 0)
        # The button sizes to its own text and is aligned left in the cell, rather than filling
        # the column: a `QToolButton` centres its label, so a full-width one put "▸ 7 done" 90px
        # in while every recording row started at the margin — two kinds of row in one list,
        # indented differently for no reason a reader could see.
        grid.addWidget(self.button, row, 0, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self.bar, row, 1)
        grid.addWidget(self.status, row, 2)

    def widgets(self):
        return (self.name, self.button, self.bar, self.status)

    def set_visible(
        self, visible: bool, *, name: bool = True, button: bool = True
    ) -> None:
        """Show or hide the row, saying in one pass which of the two name widgets is the one."""
        self.name.setVisible(visible and name)
        self.button.setVisible(visible and button)
        self.bar.setVisible(visible)
        self.status.setVisible(visible)


class UnitRows(QWidget):
    """A grid of unit rows, grouping the reference ones once there are enough to need it.

    ``toggled`` is not a signal: expanding a count row is a view state of this widget and
    nothing outside it needs to know, so it is kept here and survives a redraw.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(4)
        self._grid.setColumnStretch(1, 1)
        self._expanded: set[str] = set()
        self._units: list[Unit] = []
        self._rows: list[_Row] = []

    def set_units(self, units: list[Unit]) -> None:
        """Redraw from a list of units. Called on every progress event, so it reuses rows."""
        self._units = list(units)
        self._render()

    def _visible_units(self) -> list[tuple[Unit, int]]:
        """The rows to draw, each with its indent — groups, plus the members of open ones."""
        out: list[tuple[Unit, int]] = []
        for unit in group_units(self._units):
            out.append((unit, 0))
            if unit.is_group and unit.name in self._expanded:
                out.extend((member, 1) for member in unit.members)
        return out

    def _render(self) -> None:
        rows = self._visible_units()
        while len(self._rows) < len(rows):
            self._rows.append(_Row(self._grid, self, len(self._rows)))
        for row, (unit, indent) in zip(self._rows, rows):
            self._fill(row, unit, indent)
            # Never show what is about to be hidden. `set_visible(True)` used to reveal the
            # button on every row and the next line put it away again — a state the widget passes
            # through on each of ~7,000 redraws in a long run, and Qt is entitled to lay out and
            # paint in it. The heights above make it harmless; not entering it makes it moot.
            row.set_visible(True, name=not unit.is_group, button=unit.is_group)
        for row in self._rows[len(rows) :]:
            # Hidden, not destroyed: this widget is redrawn twice a second and a run's shape
            # changes at every recording boundary, so the spare rows are worth keeping.
            row.set_visible(False)

    def _fill(self, row: _Row, unit: Unit, indent: int) -> None:
        if unit.is_group:
            open_ = unit.name in self._expanded
            row.button.setText(("▾ " if open_ else "▸ ") + unit.name)
            disconnect_if_connected(row.button.clicked)  # a row built just now has none
            row.button.clicked.connect(
                lambda _=False, name=unit.name: self._toggle(name)
            )
        else:
            _elided(
                row.name, f"{'    ' * indent}{MARKS.get(unit.state, ' ')} {unit.name}"
            )
            _muted(row.name) if unit.state == QUEUED else _plain(row.name)

        style_bar(row.bar, unit)
        _elided(row.status, unit.status)
        if unit.state in (QUEUED, DONE):
            _muted(row.status)
        elif unit.state == FAILED:
            set_label_color(row.status, ERROR)
        else:
            _plain(row.status)

    def _toggle(self, name: str) -> None:
        if name in self._expanded:
            self._expanded.discard(name)
        else:
            self._expanded.add(name)
        self._render()
