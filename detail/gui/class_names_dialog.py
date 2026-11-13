"""**Class names…** — what each detection class is called in this experiment (§1 *Class names*).

One row per class: the stored name its runs recorded, and a field for the name every surface
shows. Saved together on *Save* (``Experiment.rename_classes``), so two classes can trade names.
An empty field is the stored name. Nothing is re-run: the name is a label over the stored one.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from detail.gui.dialog import Dialog
from detail.gui.status_color import MUTED, set_label_color


class ClassNamesDialog(Dialog):
    """Edit the experiment's class names. ``class_names`` is ``Experiment.class_names()``, read by
    the caller off the UI thread (the roster reads every run record)."""

    def __init__(self, experiment, class_names: dict[str, str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Class names")
        self._experiment = experiment
        body = self.body()
        note = QLabel(
            "Name each detection class as this experiment calls it — Resident and "
            "Intruder, say. Every screen shows the name, and either name is accepted where a "
            "class is asked for. Runs keep the name they were detected with, so nothing is "
            "re-run; exported files and rendered videos carry that one."
        )
        note.setWordWrap(True)
        body.addWidget(note)
        self._fields: dict[str, QLineEdit] = {}
        if not class_names:
            empty = QLabel(
                "No detection classes yet: they appear once a detection run or a trained "
                "detector has them."
            )
            empty.setWordWrap(True)
            set_label_color(empty, MUTED)
            body.addWidget(empty)
        holder = QWidget()
        form = QFormLayout(holder)
        for stored, name in class_names.items():
            field = QLineEdit("" if name == stored else name)
            field.setPlaceholderText(stored)
            field.setToolTip(
                f"What the class detected as {stored!r} is called; empty keeps {stored!r}"
            )
            form.addRow(f"{stored}:", field)
            self._fields[stored] = field
        body.addWidget(holder)
        body.addStretch()
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def value(self) -> dict[str, str | None]:
        """``{stored: name}`` as typed — ``None`` for an empty field (the stored name)."""
        return {
            stored: field.text().strip() or None
            for stored, field in self._fields.items()
        }

    def _save(self) -> None:
        try:
            self._experiment.rename_classes(self.value())
        except KeyError as exc:
            # A KeyError's str() is its repr, so its message is args[0], not str().
            QMessageBox.warning(self, "Class names", str(exc.args[0]))
            return
        except ValueError as exc:
            QMessageBox.warning(self, "Class names", str(exc))
            return
        self.accept()
