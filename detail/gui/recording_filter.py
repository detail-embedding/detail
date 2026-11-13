"""A recording picker that shows what a recording *is*, and filters on it.

A recording's name is the handle (§11), and in a cohort of eighty it is a poor one to choose by:
``2024-09-27_12-41-09_BOX1_PM_9_DrugOn_S20`` says when and where, and nothing about the strain,
the dose or the brain state the run is being made *for*. Those are the **Recording information**
columns, and they were one page away from every picker that asked for recordings.

:class:`RecordingFilter` attaches to an existing recording list and gives it two things:

* every row shows its recording's information after its name (a delegate paints it, so no
  picker's own item text, ``_ELIGIBLE_ROLE`` or "(needs tracking)" suffix is touched), and
* a filter row above the list: free text over the name and every value, plus one drop-down for
  each column that holds a small set of values, so "CD1 and DrugOn" is two clicks.

**A filter hides rows; it never changes the selection.** A recording checked and then filtered
out is still checked, and the count says so ("12 selected, 4 hidden by the filter") — otherwise a
filter would silently shrink the very set a run is about to use. *Select all* / *Clear* act on
the rows shown (:func:`check_shown`, :func:`uncheck_shown`).

Rows are identified by the recording id in ``UserRole`` (``gui/recording_items.py``); a row
without one has no information to show and is hidden by any column filter.
"""

from collections.abc import Iterable

from PyQt6.QtCore import QEvent, QRect, Qt, QTimer
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from detail.gui.status_color import MUTED, set_label_color

ID_ROLE = Qt.ItemDataRole.UserRole

#: A column with more distinct values than this has no drop-down (an animal id is unique to a
#: recording, so a drop-down of it is the list again); it is still searched by the text box.
MAX_FILTER_VALUES = 12

#: How many drop-downs sit on one row of the filter.
_COMBOS_PER_ROW = 3

_BLANK = "(blank)"
_NO_VALUE = "–"  # an en dash: this recording has nothing in that column
_SEPARATOR = " · "


def _id_of(item: QListWidgetItem, role=ID_ROLE) -> str | None:
    value = item.data(role)
    return value if isinstance(value, str) else None


def _has_box(item: QListWidgetItem) -> bool:
    """Whether the row shows a tick box. Not the ``ItemIsUserCheckable`` flag, which every
    ``QListWidgetItem`` carries by default: a box is drawn only once a check state is set.
    """
    return item.data(Qt.ItemDataRole.CheckStateRole) is not None


def _is_picked(item: QListWidgetItem) -> bool:
    """Whether the row is chosen: ticked if it has a box, selected if it does not."""
    if _has_box(item):
        return item.checkState() == Qt.CheckState.Checked
    return item.isSelected()


def check_shown(listwidget: QListWidget) -> None:
    """Check every *enabled* row the filter is showing; hidden rows keep their state.

    Signals are blocked so the caller can push one config update after, the way
    ``launch._check_all_eligible`` always did.
    """
    listwidget.blockSignals(True)
    for row in range(listwidget.count()):
        item = listwidget.item(row)
        if listwidget.isRowHidden(row):
            continue
        if item.flags() & Qt.ItemFlag.ItemIsEnabled:
            item.setCheckState(Qt.CheckState.Checked)
    listwidget.blockSignals(False)


def uncheck_shown(listwidget: QListWidget) -> None:
    """Uncheck every row the filter is showing. Hidden rows are left as they are."""
    listwidget.blockSignals(True)
    for row in range(listwidget.count()):
        if not listwidget.isRowHidden(row):
            listwidget.item(row).setCheckState(Qt.CheckState.Unchecked)
    listwidget.blockSignals(False)


class _InformationDelegate(QStyledItemDelegate):
    """The default row, with the recording's information after its name in the muted colour."""

    def __init__(self, source: "RecordingFilter", parent=None) -> None:
        super().__init__(parent)
        self._source = source

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        text = self._source.information_text(index.data(self._source.id_role))
        if not text:
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget
        )
        metrics = opt.fontMetrics
        # One column for every row, set by the longest name (capped), so the values line up.
        left = text_rect.left() + self._source.name_column(option.rect.width())
        room = text_rect.right() - left
        if room < 40:
            return  # the name filled the row; the tooltip still carries the rest
        painter.save()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        role = (
            QPalette.ColorRole.HighlightedText
            if selected
            else QPalette.ColorRole.PlaceholderText
        )
        painter.setPen(option.palette.color(role))
        painter.drawText(
            QRect(left, text_rect.top(), room, text_rect.height()),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            metrics.elidedText(text, Qt.TextElideMode.ElideRight, room),
        )
        painter.restore()

    def helpEvent(self, event, view, option, index) -> bool:
        if event.type() == QEvent.Type.ToolTip and not index.data(
            Qt.ItemDataRole.ToolTipRole
        ):
            tip = self._source.information_tooltip(index.data(self._source.id_role))
            if tip:
                QToolTip.showText(event.globalPos(), tip, view)
                return True
        return super().helpEvent(event, view, option, index)


class RecordingFilter(QWidget):
    """The filter row for one recording list; also the one that paints the list's information.

    Put it in the layout directly above the list::

        self._recordings = QListWidget()
        layout.addWidget(RecordingFilter(self._recordings, experiment))
        layout.addWidget(self._recordings)

    It follows the list: rows added, removed or rebuilt (a picker that rebuilds on a module
    change) are picked up on their own, and the drop-downs keep what was chosen when the same
    values are still there.

    Args:
        listwidget: the recording list. Rows carry the recording id in ``UserRole``.
        experiment: where the Recording information comes from (``recording_metadata`` and
            ``metadata_columns``); :meth:`set_experiment` swaps it for a picker with an
            experiment chooser.
        select_buttons: add *Select shown* / *Clear shown* beside the count, for a list whose
            dialog has no buttons of its own.
        id_role: the item role holding the recording id, for a picker whose ``UserRole`` is
            already spoken for.
    """

    def __init__(
        self,
        listwidget: QListWidget,
        experiment,
        *,
        select_buttons: bool = False,
        id_role=ID_ROLE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("recording_filter")
        self._list = listwidget
        self.id_role = id_role
        self._widest_name = 0
        self._columns: list[str] = []
        self._information: dict[str, dict[str, str]] = {}
        self._combos: dict[str, QComboBox] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._text = QLineEdit()
        self._text.setObjectName("recording_filter_text")
        self._text.setPlaceholderText("Filter by name or any column…")
        self._text.setClearButtonEnabled(True)
        self._text.textChanged.connect(self._filter_rows)
        outer.addWidget(self._text)
        self._combo_grid = QGridLayout()
        self._combo_grid.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._combo_grid)
        self._legend = QLabel("")
        self._legend.setWordWrap(True)
        set_label_color(self._legend, MUTED)
        outer.addWidget(self._legend)
        status_row = QHBoxLayout()
        self._count = QLabel("")
        set_label_color(self._count, MUTED)
        status_row.addWidget(self._count, 1)
        self._reset = QPushButton("Clear filter")
        self._reset.clicked.connect(self.clear_filter)
        status_row.addWidget(self._reset)
        if select_buttons:
            select_shown = QPushButton("Select shown")
            select_shown.clicked.connect(self.select_shown)
            status_row.addWidget(select_shown)
            clear_shown = QPushButton("Clear shown")
            clear_shown.clicked.connect(self.clear_shown)
            status_row.addWidget(clear_shown)
        outer.addLayout(status_row)

        self._delegate = _InformationDelegate(self, listwidget)
        listwidget.setItemDelegate(self._delegate)
        # Model signals, not the widget's: a picker blocks the *widget's* signals while it
        # rebuilds rows or checks them all, and the model's still say what changed.
        model = listwidget.model()
        for signal in (
            model.rowsInserted,
            model.rowsRemoved,
            model.modelReset,
            model.layoutChanged,
        ):
            signal.connect(self._rows_changed)
        model.dataChanged.connect(self._state_changed)
        listwidget.itemSelectionChanged.connect(self._state_changed)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self.refresh)
        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(0)
        self._count_timer.timeout.connect(self._update_count)

        self.set_experiment(experiment)

    # --- what the experiment knows ------------------------------------------ #
    def set_experiment(self, experiment) -> None:
        """Read the Recording information of ``experiment`` (``None`` for none) and re-filter."""
        if experiment is None:
            self._columns, self._information = [], {}
        else:
            self._columns = list(experiment.metadata_columns)
            self._information = experiment.recording_metadata
        self.refresh()

    def information_text(self, recording_id) -> str:
        """One line of a recording's values, in column order (a dash where it has none)."""
        if not self._columns or not isinstance(recording_id, str):
            return ""
        values = self._information.get(recording_id, {})
        cells = [values.get(column) or _NO_VALUE for column in self._columns]
        return _SEPARATOR.join(cells) if any(c != _NO_VALUE for c in cells) else ""

    def information_tooltip(self, recording_id) -> str:
        if not isinstance(recording_id, str):
            return ""
        values = self._information.get(recording_id, {})
        return "\n".join(
            f"{column}: {values[column]}"
            for column in self._columns
            if values.get(column)
        )

    # --- the rows -------------------------------------------------------------- #
    def _row_ids(self) -> list[str | None]:
        return [
            _id_of(self._list.item(row), self.id_role)
            for row in range(self._list.count())
        ]

    def _rows_changed(self, *_args) -> None:
        self._refresh_timer.start()

    def _state_changed(self, *_args) -> None:
        self._count_timer.start()

    def _distinct_values(self, ids: Iterable[str | None]) -> dict[str, list[str]]:
        """Per column, the values the listed recordings hold (a column worth a drop-down)."""
        found: dict[str, set[str]] = {column: set() for column in self._columns}
        for recording_id in ids:
            values = self._information.get(recording_id or "", {})
            for column in self._columns:
                found[column].add(values.get(column) or "")
        usable = {}
        for column, values in found.items():
            if 2 <= len(values) <= MAX_FILTER_VALUES:
                usable[column] = sorted(values, key=_natural_key)
        return usable

    def refresh(self) -> None:
        """Rebuild the drop-downs from the rows now in the list, then re-apply the filter."""
        usable = self._distinct_values(self._row_ids())
        kept = {
            column: combo.currentData()
            for column, combo in self._combos.items()
            if combo.currentData() is not None
        }
        for combo in self._combos.values():
            self._combo_grid.removeWidget(combo)
            combo.deleteLater()
        self._combos = {}
        for index, (column, values) in enumerate(usable.items()):
            combo = QComboBox()
            combo.setObjectName(f"recording_filter_{column}")
            combo.addItem(f"{column}: all", None)
            for value in values:
                combo.addItem(f"{column}: {value or _BLANK}", value)
            if column in kept and kept[column] in values:
                combo.setCurrentIndex(combo.findData(kept[column]))
            combo.currentIndexChanged.connect(self._filter_rows)
            self._combos[column] = combo
            self._combo_grid.addWidget(
                combo, index // _COMBOS_PER_ROW, index % _COMBOS_PER_ROW
            )
        self._legend.setText(
            "Shown after each name: " + _SEPARATOR.join(self._columns)
            if self._columns
            else ""
        )
        self._legend.setVisible(bool(self._columns))
        self._filter_rows()

    # --- filtering ------------------------------------------------------------- #
    def filter_active(self) -> bool:
        return bool(self._text.text().strip()) or any(
            combo.currentData() is not None for combo in self._combos.values()
        )

    def clear_filter(self) -> None:
        self._text.clear()
        for combo in self._combos.values():
            combo.setCurrentIndex(0)

    def matches(self, item: QListWidgetItem) -> bool:
        """Whether the row passes the text box and every drop-down."""
        recording_id = _id_of(item, self.id_role)
        values = self._information.get(recording_id or "", {})
        for column, combo in self._combos.items():
            wanted = combo.currentData()
            if wanted is not None and (
                recording_id is None or (values.get(column) or "") != wanted
            ):
                return False
        terms = self._text.text().lower().split()
        if not terms:
            return True
        haystack = " ".join([item.text(), *values.values()]).lower()
        return all(term in haystack for term in terms)

    def _filter_rows(self, *_args) -> None:
        for row in range(self._list.count()):
            self._list.setRowHidden(row, not self.matches(self._list.item(row)))
        self._reset.setVisible(self.filter_active())
        self._update_count()

    def _measure_name_column(self) -> None:
        metrics = self._list.fontMetrics()
        self._widest_name = max(
            (
                metrics.horizontalAdvance(self._list.item(row).text())
                for row in range(self._list.count())
            ),
            default=0,
        )
        self._list.viewport().update()

    def name_column(self, row_width: int) -> int:
        """Where the information column starts, from the start of a row's text.

        After the longest name, so the values line up, but capped at 60% of the row so a very
        long name cannot push them out of sight.
        """
        return min(self._widest_name, int(row_width * 0.6)) + 16

    def _update_count(self) -> None:
        self._measure_name_column()
        total = self._list.count()
        shown = picked = hidden_picked = 0
        for row in range(total):
            item = self._list.item(row)
            hidden = self._list.isRowHidden(row)
            shown += not hidden
            if _is_picked(item):
                picked += 1
                hidden_picked += hidden
        text = (
            f"{shown} of {total} recordings shown"
            if self.filter_active()
            else f"{total} recordings"
        )
        text += f" · {picked} selected"
        if hidden_picked:
            text += f" ({hidden_picked} hidden by the filter)"
        self._count.setText(text)

    # --- selecting what is shown (for a list whose dialog has no buttons) -------- #
    def select_shown(self) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if self._list.isRowHidden(row) or not item.flags() & (
                Qt.ItemFlag.ItemIsEnabled
            ):
                continue
            if _has_box(item):
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setSelected(True)

    def clear_shown(self) -> None:
        for row in range(self._list.count()):
            if self._list.isRowHidden(row):
                continue
            item = self._list.item(row)
            if _has_box(item):
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setSelected(False)


class _FilteredList(QWidget):
    """A recording list with its filter above it, as one widget to drop into a form row."""

    def __init__(self, listwidget: QListWidget, experiment, **options) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.filter = RecordingFilter(listwidget, experiment, **options)
        layout.addWidget(self.filter)
        layout.addWidget(listwidget)


def with_filter(listwidget: QListWidget, experiment, **options) -> "_FilteredList":
    """``listwidget`` under a :class:`RecordingFilter`, as one widget (``.filter`` is the bar).

    For a form row, where the list is passed to ``addRow`` and there is no layout of its own to
    put the bar into. ``options`` are :class:`RecordingFilter`'s.
    """
    return _FilteredList(listwidget, experiment, **options)


def _natural_key(value: str) -> tuple[int, float, str]:
    """Numbers in numeric order (``9`` before ``10``), then words; a blank last."""
    if not value:
        return (2, 0.0, "")
    try:
        return (0, float(value), value)
    except ValueError:
        return (1, 0.0, value.lower())
