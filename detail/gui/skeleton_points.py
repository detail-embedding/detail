"""A choice of a skeleton's points: one checkbox per point — the clip window's **Trail points**
(§22), and the montage's.

It knows nothing of a body plan: the names come from whatever skeleton the window hands it, in
column order. Every point starts ticked; :meth:`SkeletonPointsPicker.value` is ``None`` while
they all are ("every point"), else the ticked names. A value set before the skeleton is known is
held, reported back as it was set, and applied when the skeleton arrives; a choice made on one
skeleton carries to the next for the names they share.

:class:`SkeletonPointsButton` is the picker behind a button, in a menu, so a control row stays
one line. It is the one place that shape is written: the port this replaced had the same
``QMenu`` + ``QWidgetAction`` block in three windows, each with its own "k of N points" label and
its own greying.
"""

from __future__ import annotations

from detail.gui.qt import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    Qt,
    Signal,
)

_COLUMNS = 4
_UNSET = object()
#: The note under the ticks: no skeleton yet, none ticked, every point ticked, some ticked.
_EMPTY = "No keypoints in this clip, so no point leaves a trail."
_NONE = "Tick at least one point to draw its trail."
_EVERY = "Every point leaves a trail."
_SOME = "Only the ticked points leave a trail."
#: The button's tooltip while there is a skeleton to choose from.
POINTS_TIP = (
    "Choose which keypoints leave a trail: every point, or only some (the paws, say)."
)


class SkeletonPointsPicker(QWidget):
    """Tick the points to include; ``changed`` fires on every tick."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._names: list[str] = []
        self._boxes: dict[str, QCheckBox] = {}
        self._pending = _UNSET  # a value set before the skeleton was known
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        host = QWidget()
        self._grid = QGridLayout(host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(host)
        self._note = QLabel(_EMPTY)
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

    # -- the skeleton ------------------------------------------------------------- #
    def set_skeleton(self, names) -> None:
        """Show ``names`` (column order), keeping the ticks of the names it already knew."""
        if self._pending is not _UNSET:
            wanted = None if self._pending is None else set(self._pending)
            self._pending = _UNSET
        else:
            unticked = {n for n, box in self._boxes.items() if not box.isChecked()}
            wanted = {str(n) for n in names} - unticked
        for box in self._boxes.values():
            self._grid.removeWidget(box)
            box.deleteLater()
        self._names = [str(n) for n in names]
        self._boxes = {}
        for i, name in enumerate(self._names):
            box = QCheckBox(name)
            box.setChecked(wanted is None or name in wanted)
            box.toggled.connect(self._ticked)
            self._grid.addWidget(box, i // _COLUMNS, i % _COLUMNS)
            self._boxes[name] = box
        self._sync_note()

    def names(self) -> list[str]:
        return list(self._names)

    # -- the value ---------------------------------------------------------------- #
    def value(self) -> list[str] | None:
        """``None`` when every point is ticked, else the ticked names in column order
        (possibly empty). Before a skeleton: what was set."""
        if not self._boxes:
            return None if self._pending is _UNSET else self._pending
        ticked = [n for n in self._names if self._boxes[n].isChecked()]
        return None if len(ticked) == len(self._names) else ticked

    def set_value(self, value: list[str] | None) -> None:
        """Tick exactly ``value`` (``None``: every point)."""
        if not self._boxes:
            self._pending = None if value is None else list(value)
            self.changed.emit()  # what `value()` reports changed, skeleton or not
            return
        wanted = None if value is None else set(value)
        for name, box in self._boxes.items():
            box.blockSignals(True)
            box.setChecked(wanted is None or name in wanted)
            box.blockSignals(False)
        self._sync_note()
        self.changed.emit()

    def _ticked(self, _checked: bool) -> None:
        self._sync_note()
        self.changed.emit()

    def _sync_note(self) -> None:
        if not self._boxes:
            self._note.setText(_EMPTY)
            return
        ticked = sum(1 for box in self._boxes.values() if box.isChecked())
        if not ticked:
            self._note.setText(_NONE)
        elif ticked == len(self._boxes):
            self._note.setText(_EVERY)
        else:
            self._note.setText(_SOME)


class SkeletonPointsButton(QPushButton):
    """A :class:`SkeletonPointsPicker` behind a button, dropped down as a menu.

    ``changed`` is the picker's; :meth:`value` and :meth:`set_skeleton` reach through to it.
    :meth:`set_reason` greys the button with why there is nothing to choose (a run with no
    keypoints), and clears the greying with ``None`` — the one way the button is disabled, so
    it always says why.
    """

    changed = Signal()

    def __init__(
        self, text: str = "Trail points", parent: QWidget | None = None
    ) -> None:
        super().__init__(text, parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.picker = SkeletonPointsPicker()
        self.picker.changed.connect(self.changed)
        menu = QMenu(self)
        holder = QWidgetAction(menu)
        holder.setDefaultWidget(self.picker)
        menu.addAction(holder)
        self.setMenu(menu)
        self.set_reason(None)

    def value(self) -> list[str] | None:
        return self.picker.value()

    def set_skeleton(self, names) -> None:
        self.picker.set_skeleton(names)

    def set_reason(self, reason: str | None) -> None:
        """Grey the button with ``reason``, or enable it (``None``) with the usual tip."""
        self.setEnabled(reason is None)
        self.setToolTip(POINTS_TIP if reason is None else reason)
