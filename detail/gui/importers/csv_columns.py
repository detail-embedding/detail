"""The CSV importer's config band (Section 3): pick the time column and the channels.

Shows the file's **actual header** with a first-value preview beside each column, so the user
maps a CSV shaped however its source tool shaped it without editing the file. The mapping is
``CsvStreamConfig``; the time-kind combo comes from the model's own ``choices``
(``choice_combo``), so its option list has exactly one source.

Ticked columns become **channels of one stream**, not one stream each. The name beside a
column is therefore the *channel*'s name, and the stream's own name is the dialog's, one
level up.
"""

import re

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detail.configs.stream_csv import CsvStreamConfig
from detail.gui._pydantic_form import choice_combo
from detail.gui.importers.registry import ConfigBand, register_importer_ui
from detail.gui.sizing import lines, mono_font
from detail.gui.tables import (
    enable_sorting,
    fill_width,
    make_columns_resizable,
    refill_table_widget,
)
from detail.gui.status_color import ERROR, MUTED, set_label_color
from detail.importers import _csv_common

#: Column names the canonical stream schema owns, mirrored from the importer so the band can
#: say so before the import refuses it.
_RESERVED = ("t_ns", "frame")

_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

#: How tall the channel table stands. Enough to read a pose file's first bodyparts without
#: the dialog growing past a laptop screen; it scrolls beyond that.
_TABLE_LINES = 10

#: What a time column's name says it holds, for seeding the kind combo. The same table the
#: importer sniffs with, plus the two spellings a source tool most often uses for seconds.
#: A guess the user can see and change, never one applied silently: the combo shows the
#: result, and ``auto`` stays available for "work it out at import".
_NAME_HINTS = {
    "frame": "frame",
    "frame_idx": "frame",
    "frame_index": "frame",
    "t_ns": "t_ns",
    "seconds": "seconds",
    "time": "seconds",
    "t_sec": "seconds",
    "time_s": "seconds",
    "timestamp": "utc",
}


def _default_channel_name(column: str) -> str:
    """A path-safe channel name from a CSV header cell."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", column.strip()).strip("_")
    return cleaned or "channel"


class CsvColumnBand(ConfigBand):
    """Time column + kind + a checkable channel table, read from the chosen CSV."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list[str] = []
        self._preview: dict[str, str] = {}
        self._rows: dict[str, dict] = {}  # column -> {"checked": bool, "name": str}
        self._error: str | None = None
        self._source: str | None = None
        self._missing: list[str] = []
        self._pending: dict | None = None  # a preset chosen before a file was

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self._time_col = QComboBox()
        self._time_col.currentIndexChanged.connect(self._on_time_column_changed)
        form.addRow("Time column:", self._time_col)
        self._time_kind = choice_combo(CsvStreamConfig, "time_kind")
        self._time_kind.currentIndexChanged.connect(self._emit_changed)
        form.addRow("Time is:", self._time_kind)
        layout.addLayout(form)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        set_label_color(self._hint, MUTED)
        layout.addWidget(self._hint)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["", "Column", "First value", "Channel"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        make_columns_resizable(self._table)
        fill_width(self._table)
        # Safe to sort: every handler here closes over the column *name*, never a row
        # position (`_on_toggle` scans for it by text), and Qt moves a `setCellWidget`
        # widget along with its row on sort (checked directly — not documented either way).
        enable_sorting(self._table)
        self._table.setMinimumHeight(lines(_TABLE_LINES))
        layout.addWidget(self._table, 1)

        self._empty = QLabel("(choose a data file above to list its columns)")
        self._empty.setWordWrap(True)
        set_label_color(self._empty, MUTED)
        layout.addWidget(self._empty)
        self._table.hide()
        self._update_hint()

    # -- ConfigBand ---------------------------------------------------------- #

    def set_source(self, path: str) -> None:
        """Read the header + first row of ``path``; an unreadable file becomes the problem.

        Repeats are ignored. Picking a file both swaps the band (which seeds it with the
        path) and hands it to the band, so the same file would otherwise be read twice and
        the whole table rebuilt under the user's first click.
        """
        if path == self._source:
            return
        self._source = path
        self._error = None
        self._missing = []
        try:
            header, first = _csv_common.sniff_header_and_first(path)
        except OSError as exc:  # a path that vanished between picking and reading
            header, first, self._error = [], None, str(exc)
        if not self._error and not header:
            self._error = f"no header row in {path}"
        repeated = _repeated(header)
        if repeated:
            # Named here as well as refused at import: this is the one problem the user can
            # only fix in the source file, so saying it before they map anything saves the
            # whole trip. Every DeepLabCut export lands here.
            self._error = (
                f"this CSV names {', '.join(repeated)} more than once. Column names must "
                "be unique; a multi-row header needs flattening to one row first."
            )
            header = []
        self._columns = header
        self._preview = dict(zip(header, first)) if first else {}
        self._rows = {
            c: {"checked": False, "name": _default_channel_name(c)} for c in header
        }
        self._time_col.blockSignals(True)
        self._time_col.clear()
        self._time_col.addItems(header)
        self._time_col.blockSignals(False)
        self._seed_time_kind()
        self._rebuild_rows()
        if self._pending is not None:
            self.set_config(self._pending)

    def config(self) -> dict | None:
        if not self._columns:
            return None
        return {
            "time_column": self._time_col.currentText(),
            "time_kind": self._time_kind.currentData(),
            "channels": [
                {"column": c, "name": self._rows[c]["name"].strip()}
                for c in self._checked()
            ],
        }

    def set_config(self, config: dict) -> None:
        """Apply a saved mapping to the loaded file, reporting what this CSV doesn't have.

        Starts from a clean slate so a preset always gives the same state whatever was
        ticked before, and names the missing columns rather than quietly importing fewer
        channels than the preset says.
        """
        self._missing = []
        if not self._columns:
            self._pending = dict(config)  # no file yet; apply it when one arrives
            return
        self._pending = None
        self._rows = {
            c: {"checked": False, "name": _default_channel_name(c)}
            for c in self._columns
        }
        time_column = config.get("time_column")
        if time_column in self._columns:
            self._time_col.blockSignals(True)
            self._time_col.setCurrentText(time_column)
            self._time_col.blockSignals(False)
        elif time_column:
            self._missing.append(time_column)
        index = self._time_kind.findData(config.get("time_kind", "auto"))
        if index >= 0:
            self._time_kind.blockSignals(True)
            self._time_kind.setCurrentIndex(index)
            self._time_kind.blockSignals(False)
        for channel in config.get("channels") or []:
            column = channel.get("column")
            if column in self._columns:
                self._rows[column] = {
                    "checked": True,
                    "name": channel.get("name") or column,
                }
            else:
                self._missing.append(column)
        self._rebuild_rows()

    def problem(self) -> str | None:
        if self._error:
            return self._error
        if self._missing:
            # Not a warning to one side: importing "the rest" of a preset silently gives a
            # stream with fewer channels than every other recording's, and nothing later
            # says why one recording is short.
            return (
                f"this CSV has no {', '.join(self._missing)} (named by the saved setup) — "
                "untick it from the setup, or pick a file that has it."
            )
        if not self._columns:
            return None  # no file yet; the dialog says so
        checked = self._checked()
        if not checked:
            return "Tick at least one column to bring in as a channel."
        for column in checked:
            name = self._rows[column]["name"].strip()
            if not name:
                return f"{column}: the channel needs a name."
            if not _VALID_NAME.match(name):
                return (
                    f"{column}: a channel name takes letters, digits, '-' and '_' only."
                )
            if name in _RESERVED:
                return (
                    f"{column}: {name!r} is reserved by the stream schema; "
                    "give the channel another name."
                )
        names = [self._rows[c]["name"].strip() for c in checked]
        for name in names:
            if names.count(name) > 1:
                return f"two channels are both named {name!r}."
        return None

    def suggested_name(self) -> str:
        """One ticked channel names the stream after itself; several do not.

        A stream called ``nose_x`` holding four channels would be actively misleading, so the
        suggestion is only made where it is unambiguous.
        """
        checked = self._checked()
        return self._rows[checked[0]]["name"].strip() if len(checked) == 1 else ""

    # -- internals ----------------------------------------------------------- #

    def _seed_time_kind(self) -> None:
        """Preselect the kind the time column's name implies, leaving it editable."""
        key = self._time_col.currentText().strip().lower()
        index = self._time_kind.findData(_NAME_HINTS.get(key, "auto"))
        if index >= 0:
            self._time_kind.blockSignals(True)
            self._time_kind.setCurrentIndex(index)
            self._time_kind.blockSignals(False)
        self._update_hint()

    def _on_time_column_changed(self) -> None:
        self._seed_time_kind()
        self._rebuild_rows()
        # The time column decides the clock (a frame index is the video's), and the dialog
        # gates its alignment row on that, so a change here is a change it must hear about.
        self.changed.emit()

    def _update_hint(self) -> None:
        kind = self._time_kind.currentData()
        self._hint.setText(_KIND_HINTS.get(kind, ""))

    def _checked(self) -> list[str]:
        time_col = self._time_col.currentText()
        return [c for c in self._columns if c != time_col and self._rows[c]["checked"]]

    def _rebuild_rows(self) -> None:
        time_col = self._time_col.currentText()
        visible = [c for c in self._columns if c != time_col]
        self._table.setVisible(bool(visible))
        self._empty.setVisible(not visible)

        def _fill() -> None:
            self._table.setRowCount(len(visible))
            for row, column in enumerate(visible):
                state = self._rows[column]
                self._table.setCellWidget(
                    row, 0, self._checkbox(column, state["checked"])
                )
                for col_index, text in (
                    (1, column),
                    (2, self._preview.get(column, "")),
                ):
                    item = QTableWidgetItem(text)
                    item.setFont(mono_font())
                    self._table.setItem(row, col_index, item)
                field = QLineEdit(state["name"], parent=self._table)
                field.setFont(mono_font())
                field.setEnabled(state["checked"])
                field.textChanged.connect(
                    lambda text, c=column: self._on_name_changed(c, text)
                )
                self._table.setCellWidget(row, 3, field)

        # `QTableWidget` resorts itself on every `setItem` while sorting is enabled, which
        # would scramble this fill loop's row-by-row writes.
        refill_table_widget(self._table, _fill)
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(0, lines(1))
        self._emit_changed()

    def _checkbox(self, column: str, checked: bool) -> QWidget:
        box = QCheckBox(parent=self._table)
        box.setChecked(checked)
        box.toggled.connect(lambda on, c=column: self._on_toggle(c, on))
        holder = QWidget(parent=self._table)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(box)
        row.addStretch(1)
        return holder

    def _on_toggle(self, column: str, checked: bool) -> None:
        """Enable that row's name field in place.

        Rebuilding the table here would destroy the checkbox mid-signal and take the
        keyboard focus with it, on every space press.
        """
        self._rows[column]["checked"] = checked
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 1)
            if item is not None and item.text() == column:
                field = self._table.cellWidget(row, 3)
                if field is not None:
                    field.setEnabled(checked)
                break
        self._emit_changed()

    def _on_name_changed(self, column: str, text: str) -> None:
        # No rebuild: it would take the focus out of the field being typed in.
        self._rows[column]["name"] = text
        self._emit_changed()

    def _emit_changed(self) -> None:
        self._update_hint()
        self.changed.emit()


#: What each time kind means, in the words a user needs to pick between them.
_KIND_HINTS = {
    "auto": "Work it out from the time column's name, and refuse the import if the name "
    "doesn't say.",
    "frame": "The video's own frame index, counting from 0. Aligned at import.",
    "seconds": "Seconds since this stream's own start.",
    "t_ns": "Nanoseconds since this stream's own start.",
    "utc": "A wall-clock timestamp: ISO 8601 (2026-09-04T14:00:38Z) or Unix epoch seconds. "
    "The first row becomes t=0.",
}


@register_importer_ui
class CsvColumnsUI:
    importer_id = "timeseries_csv"

    def config_band(self, parent=None) -> ConfigBand:
        return CsvColumnBand(parent)


def _repeated(header: list[str]) -> list[str]:
    """Header names appearing more than once, in first-seen order."""
    seen: set[str] = set()
    repeated: list[str] = []
    for field in header:
        if field in seen and field not in repeated:
            repeated.append(field)
        seen.add(field)
    return repeated
