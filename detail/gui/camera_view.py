"""The camera-view picker: one dropdown + a note box, shared by every surface that sets one.

Three surfaces set a camera view (§10) — the New experiment dialog and the Recordings page set
the experiment's default, Scene geometry sets a recording's own — so the vocabulary, the order,
the "Other needs a description" rule and the note box's visibility live here once. The words
come from ``core.camera_view.VIEWS``; nothing here spells a view.

**The first option is a fact the caller supplies, not a default.** On the experiment it means
"no default yet"; on a recording it means "inherit", and says what that inherits — "Experiment
default (Side view)" — so choosing it is an informed choice rather than a blank. A label written
here would be right for one of the two and wrong for the other, so ``unset_label`` is required.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QWidget

from detail.core import camera_view

#: The note box's placeholder — what to write, not a sample answer.
_NOTE_HINT = "Describe the view, e.g. 45° from above"


class CameraViewPicker(QWidget):
    """A camera-view dropdown, with a description box shown only for **Other**.

    ``value()`` is ``(view, note)`` with ``view`` ``None`` for the first option; it is not
    validated — the caller's setter is, and its ``ValueError`` names the problem.
    """

    #: Emitted when the chosen view changes, and when the note is finished being edited.
    changed = pyqtSignal()

    def __init__(self, *, unset_label: str, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        self._combo.addItem(unset_label, None)
        for view, label in camera_view.VIEWS.items():
            self._combo.addItem(label, view)
        self._combo.setToolTip(
            "Where the camera looks at the animal from. Measures that only make sense "
            "from one view (how high the animal reaches, from the side) read this."
        )
        self._note = QLineEdit()
        self._note.setPlaceholderText(_NOTE_HINT)
        self._last_note = ""
        row.addWidget(self._combo)
        row.addWidget(self._note, 1)
        # Takes the slack while the note is hidden, so the dropdown keeps its own width
        # instead of stretching across the row.
        row.addStretch(0)
        self._row = row
        self._combo.currentIndexChanged.connect(self._on_view_changed)
        self._note.editingFinished.connect(self._on_note_finished)
        self._sync_note()

    def value(self) -> tuple[str | None, str | None]:
        view = self._combo.currentData()
        note = self._note.text().strip() if view == "other" else ""
        return view, note or None

    def set_value(self, view: str | None, note: str | None = None) -> None:
        """Show ``view`` / ``note`` without emitting ``changed`` — loading is not editing."""
        self.blockSignals(True)
        try:
            self._combo.setCurrentIndex(max(self._combo.findData(view), 0))
            self._note.setText(note or "")
            self._sync_note()
        finally:
            self.blockSignals(False)
        self._last_note = self._note.text()

    def set_unset_label(self, text: str) -> None:
        """Re-word the first option — a recording's "Experiment default (…)" follows the
        default when it changes."""
        self._combo.setItemText(0, text)

    def _sync_note(self) -> None:
        shown = self._combo.currentData() == "other"
        self._note.setVisible(shown)
        self._row.setStretch(2, 0 if shown else 1)

    def _on_view_changed(self, _index: int) -> None:
        self._sync_note()
        # Choosing Other with an empty note is not a value yet; the note's own
        # editingFinished emits once there is something to save.
        if self._combo.currentData() == "other" and not self._note.text().strip():
            self._note.setFocus()
            return
        self.changed.emit()

    def _on_note_finished(self) -> None:
        # editingFinished also fires on focus loss with nothing typed; only a real edit
        # is a change.
        if self._note.text() == self._last_note:
            return
        self._last_note = self._note.text()
        self.changed.emit()


def inherit_label(experiment_view: camera_view.CameraView) -> str:
    """The first option on a recording's picker: what choosing it inherits."""
    return f"Experiment default ({experiment_view.label})"
