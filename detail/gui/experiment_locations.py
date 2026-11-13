"""The folders experiments live in (§2): pick one, add one, change or remove one.

An experiment location is a folder on this machine — the default one in Documents, or one
the user added on an external drive — that experiments are created in and listed from. The
choice lives in `~/.config/detail/storage.yaml` (the ``storage:`` aliases); no experiment file
holds the path, so a drive that comes back under another letter is one "Change folder…" away.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from detail.gui.dialog import Dialog
from detail.gui.sizing import mono_font
from detail.storage import locations


def location_label(location: locations.ExperimentLocation) -> str:
    """How a location reads in a picker: its name and folder, and whether it is there."""
    name = "Default folder" if location.is_default else location.name
    text = f"{name} — {location.root.as_posix()}"
    return text if location.connected else f"{text} (not connected)"


def not_connected_notice(found: list[locations.ExperimentLocation]) -> str:
    """One line naming each location whose folder is missing, or ``""`` when none is."""
    missing = [loc for loc in found if not loc.connected]
    if not missing:
        return ""
    names = ", ".join(f"{loc.name} ({loc.root.as_posix()})" for loc in missing)
    return (
        f"Not connected: {names}. Experiments there are listed when the drive is "
        "plugged in."
    )


class LocationPicker(QWidget):
    """Where a new experiment goes: the locations on this machine, plus Add location…."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        row.addWidget(self._combo, 1)
        add = QPushButton("Add location…")
        add.setToolTip("Use another folder — on an external drive, say")
        add.clicked.connect(self._add)
        row.addWidget(add)
        self.reload()

    def reload(self, select: str | None = None) -> None:
        """Re-read the locations, keeping ``select`` (or the default) chosen."""
        self._combo.clear()
        for location in locations.experiment_locations():
            self._combo.addItem(location_label(location), location.name)
            if not location.connected:
                # Listed so the user sees the drive is known and absent, not forgotten.
                item = self._combo.model().item(self._combo.count() - 1)
                item.setEnabled(False)
        index = self._combo.findData(select or locations.DEFAULT_LOCATION)
        self._combo.setCurrentIndex(max(index, 0))

    def location(self) -> str:
        """The chosen location's name."""
        return self._combo.currentData()

    def _add(self) -> None:
        dialog = AddLocationDialog(self)
        if dialog.exec() == Dialog.DialogCode.Accepted and dialog.added is not None:
            self.reload(select=dialog.added.name)


class AddLocationDialog(Dialog):
    """Name a folder as an experiment location, or give an existing one its new folder."""

    def __init__(self, parent=None, *, existing: str | None = None) -> None:
        super().__init__(parent)
        self.added: locations.ExperimentLocation | None = None
        self._existing = existing
        self.setWindowTitle("Change folder" if existing else "Add location")
        layout = self.body()
        intro = QLabel(
            f"Choose the folder that now holds the experiments of {existing!r}."
            if existing
            else "Choose a folder on any drive — an external hard drive, say. An "
            "experiment created there keeps its recordings and results in that folder, "
            "and every experiment already in it is listed."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self._folder = QLineEdit()
        self._folder.setFont(mono_font())
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self._folder, 1)
        folder_row.addWidget(choose)
        form.addRow("Folder:", folder_row)
        self._name = QLineEdit(existing or "")
        self._name.setReadOnly(existing is not None)
        self._name.setPlaceholderText("e.g. Lab drive")
        form.addRow("Name:", self._name)
        layout.addLayout(form)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Change" if existing else "Add"
        )
        box.accepted.connect(self._on_accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Experiment location", self._folder.text()
        )
        if not folder:
            return
        self._folder.setText(folder)
        if not self._name.text().strip():
            self._name.setText(Path(folder).name or Path(folder).anchor.strip(":\\/"))

    def _on_accept(self) -> None:
        if not self._folder.text().strip():
            QMessageBox.warning(self, self.windowTitle(), "Choose a folder.")
            return
        try:
            self.added = locations.add_location(
                self._name.text(), self._folder.text().strip()
            )
        except (ValueError, NotADirectoryError) as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.accept()


class ExperimentLocationsDialog(Dialog):
    """Every experiment location on this machine, with Add / Change folder / Remove."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        #: Whether anything was added, re-pointed or removed — the list behind needs a refresh.
        self.changed = False
        self.setWindowTitle("Experiment locations")
        layout = self.body()
        intro = QLabel(
            "The folders your experiments are kept in. Removing one only stops it being "
            "listed here; nothing on the drive is deleted, and adding it again brings its "
            "experiments back."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        add = QPushButton("Add location…")
        add.clicked.connect(self._add)
        add_row = QHBoxLayout()
        add_row.addWidget(add)
        add_row.addStretch()
        layout.addLayout(add_row)
        self._rows = QWidget()
        self._grid = QGridLayout(self._rows)
        self._grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._rows)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        self._fill()

    def _fill(self) -> None:
        while self._grid.count():
            widget = self._grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        for row, location in enumerate(locations.experiment_locations()):
            name = QLabel("Default folder" if location.is_default else location.name)
            font = name.font()
            font.setBold(True)
            name.setFont(font)
            self._grid.addWidget(name, row, 0)
            folder = QLabel(location.root.as_posix())
            folder.setFont(mono_font())
            self._grid.addWidget(folder, row, 1)
            self._grid.addWidget(
                QLabel("connected" if location.connected else "not connected"), row, 2
            )
            if location.is_default:
                continue
            change = QPushButton("Change folder…")
            change.clicked.connect(lambda _c=False, n=location.name: self._change(n))
            self._grid.addWidget(change, row, 3)
            remove = QPushButton("Remove")
            remove.clicked.connect(lambda _c=False, n=location.name: self._remove(n))
            self._grid.addWidget(remove, row, 4)
        self._grid.setColumnStretch(1, 1)

    def _add(self) -> None:
        dialog = AddLocationDialog(self)
        if dialog.exec() == Dialog.DialogCode.Accepted:
            self.changed = True
            self._fill()

    def _change(self, name: str) -> None:
        dialog = AddLocationDialog(self, existing=name)
        if dialog.exec() == Dialog.DialogCode.Accepted:
            self.changed = True
            self._fill()

    def _remove(self, name: str) -> None:
        # No confirmation: removing forgets a name and deletes nothing, and adding the folder
        # again undoes it (`a_confirmation_guards_something_the_user_cannot_undo`).
        try:
            locations.remove_location(name)
        except ValueError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.changed = True
        self._fill()


class ChooseLocationDialog(Dialog):
    """Which location an imported experiment goes into, when there is more than one."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import experiment")
        layout = self.body()
        form = QFormLayout()
        self._picker = LocationPicker()
        form.addRow("Import into:", self._picker)
        layout.addLayout(form)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def location(self) -> str:
        return self._picker.location()
