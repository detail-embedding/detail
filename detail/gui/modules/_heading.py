"""The heading-definition picker (§14 "Egocentric alignment"), one shared widget.

A heading definition is two sets of skeleton point names, front and back; the embedding form
chooses one to align on. It became one shared widget while the keypoint audit (since removed)
scored the same one, so the two forms could not drift — the shape ``modules/_common.py`` and
``modules/_every_n.py`` give to the other shared form rows: a combo over the saved definitions
(``heading_definitions.yaml``, the stream-preset pattern) plus **Define…**, definitions that do not
fit the run's skeleton greyed with the reason. Leading underscore: not a module UI, so
``modules/__init__.py`` never imports it and the launch dialog never lists it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QWidget,
)

from detail.gui.dialog import Dialog
from detail.storage import heading_definitions

NONE_TEXT = "(choose a heading definition)"


class HeadingDefinitionDialog(Dialog):
    """Define a heading: a name, and the skeleton's points sorted into front, back or
    neither. With no skeleton on hand the points are typed, comma-separated."""

    def __init__(self, experiment, skeleton_names=None, parent=None) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._names = list(skeleton_names or [])
        self.saved: dict | None = None
        self.setWindowTitle("Define a heading")
        layout = self.body()
        form = QFormLayout()
        self._name = QLineEdit()
        self._name.setPlaceholderText("nose_to_tail")
        self._name.setToolTip(
            "Letters, digits, '-' and '_'. How a run names the heading."
        )
        form.addRow("Name:", self._name)
        self._front_list = self._back_list = None
        self._front_text = self._back_text = None
        if self._names:
            self._front_list = self._point_list()
            self._back_list = self._point_list()
            form.addRow("Front points:", self._front_list)
            form.addRow("Back points:", self._back_list)
        else:
            self._front_text = QLineEdit()
            self._front_text.setPlaceholderText("nose, left_ear, right_ear")
            self._back_text = QLineEdit()
            self._back_text.setPlaceholderText("tail_base")
            form.addRow("Front points:", self._front_text)
            form.addRow("Back points:", self._back_text)
        note = QLabel(
            "The heading points from the mean of the back points to the mean of the front "
            "points. A set rather than a pair: one missing point moves the mean a little "
            "instead of losing the heading."
        )
        note.setWordWrap(True)
        form.addRow("", note)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

    def _point_list(self) -> QListWidget:
        widget = QListWidget()
        widget.setToolTip("Tick the points in this set.")
        for name in self._names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            widget.addItem(item)
        return widget

    @staticmethod
    def _checked(widget: QListWidget) -> list[str]:
        return [
            widget.item(i).text()
            for i in range(widget.count())
            if widget.item(i).checkState() == Qt.CheckState.Checked
        ]

    @staticmethod
    def _typed(widget: QLineEdit) -> list[str]:
        return [p.strip() for p in widget.text().split(",") if p.strip()]

    def points(self) -> tuple[list[str], list[str]]:
        if self._front_list is not None:
            return self._checked(self._front_list), self._checked(self._back_list)
        return self._typed(self._front_text), self._typed(self._back_text)

    def _save(self) -> None:
        front, back = self.points()
        try:
            self.saved = self._experiment.save_heading_definition(
                self._name.text().strip(), front, back
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Define a heading", str(exc))
            return
        self.accept()


class HeadingPicker(QWidget):
    """A combo over the experiment's saved heading definitions plus **Define…**.

    ``set_skeleton_names`` greys every definition the skeleton cannot satisfy, with the
    missing points in the row's tooltip, and clears a greyed choice; ``set_experiment``
    (re)reads the saved list. ``changed`` fires when the chosen name changes.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._experiment = None
        self._skeleton_names: list[str] | None = None
        self._definitions: list[dict] = []
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setToolTip(
            "A saved heading definition: the points whose mean is the front and the points "
            "whose mean is the back. Greyed when the keypoint run's skeleton lacks one of "
            "its points."
        )
        self.combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.define = QPushButton("Define…")
        self.define.setToolTip("Save a new heading definition for this experiment.")
        self.define.clicked.connect(self._define)
        self.define.setEnabled(False)
        row.addWidget(self.combo, 1)
        row.addWidget(self.define)
        self._fill()

    # -- inputs ---------------------------------------------------------------- #
    def set_experiment(self, experiment) -> None:
        self._experiment = experiment
        self.define.setEnabled(experiment is not None)
        self._definitions = (
            list(experiment.heading_definitions()) if experiment is not None else []
        )
        self._fill()

    def set_skeleton_names(self, names) -> None:
        """The skeleton the chosen keypoint run follows; ``None`` greys nothing."""
        self._skeleton_names = None if names is None else [str(n) for n in names]
        self._fill()

    # -- outputs --------------------------------------------------------------- #
    def name(self) -> str | None:
        data = self.combo.currentData()
        return data if isinstance(data, str) else None

    def set_name(self, name: str | None) -> None:
        index = self.combo.findData(name) if name else 0
        self.combo.setCurrentIndex(index if index >= 0 else 0)

    def reason(self) -> str | None:
        """Why the chosen definition cannot be used, or ``None``."""
        name = self.name()
        if name is None:
            return None
        definition = next((d for d in self._definitions if d["name"] == name), None)
        if definition is None or self._skeleton_names is None:
            return None
        return heading_definitions.fits(definition, self._skeleton_names)

    # -- internals ------------------------------------------------------------- #
    def _fill(self) -> None:
        previous = self.name()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(NONE_TEXT, None)
        for definition in self._definitions:
            label = (
                f"{definition['name']}  ({', '.join(definition['front'])} ← "
                f"{', '.join(definition['back'])})"
            )
            self.combo.addItem(label, definition["name"])
            index = self.combo.count() - 1
            reason = (
                heading_definitions.fits(definition, self._skeleton_names)
                if self._skeleton_names is not None
                else None
            )
            item = self.combo.model().item(index)
            if reason is not None:
                item.setEnabled(False)
                self.combo.setItemText(index, f"{label} — {reason}")
            self.combo.setItemData(index, reason or "", Qt.ItemDataRole.ToolTipRole)
        self.combo.blockSignals(False)
        self.set_name(previous)
        if self.reason() is not None:
            self.combo.setCurrentIndex(0)
        self.changed.emit()

    def _define(self) -> None:
        if self._experiment is None:
            return
        dialog = HeadingDefinitionDialog(self._experiment, self._skeleton_names, self)
        if dialog.exec() and dialog.saved is not None:
            self._definitions = list(self._experiment.heading_definitions())
            self._fill()
            self.set_name(dialog.saved["name"])
