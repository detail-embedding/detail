"""The example-set picker (§26 *Example sets follow the model that needs them*).

An example set is the frames on which a user clicked a skeleton's points: the input of a
keypoint model that ``needs_examples``. It has no page of its own; wherever such a model is
chosen (the keypoint detect form, *Test keypoints…*, *Distill new model* for a keypoint model), this
one widget appears::

    Skeleton:     [rat overhead ▾]                     (the library, presets included)
    Example set:  [overhead rats ▾]  8 / 8 frames      (this experiment's sets on it)
                  [New example set…]  [Mark frames…]  [Delete set…]

A set belongs to a skeleton by its ``skeleton_ref`` (the library entry it was made from) or,
for a set made before the library existed, by the digest of the skeleton it holds. A set that
matches neither is listed under *(no library skeleton)*, so nothing an experiment holds is
hidden. Everything is read off the UI thread.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from detail.api import skeletons as library
from detail.core.examples import ExampleSet
from detail.gui.dialog import Dialog
from detail.gui.sizing import lines, set_available
from detail.gui.workers import BackgroundCall

#: The skeleton choice for sets whose skeleton is in no library entry.
UNLINKED = "(no library skeleton)"


def skeleton_of(es: ExampleSet, entries: list[dict]) -> str | None:
    """The library id ``es`` belongs to: its ``skeleton_ref`` when that entry holds the set's
    skeleton, else the first entry holding it (by digest), else None.

    The digest is checked before the ref's id is trusted: a library id means something only
    on the machine that minted it, so a set from elsewhere (or one whose entry was deleted and
    its id reused) can name an entry holding a different skeleton.
    """
    ref = (es.skeleton_ref or {}).get("id")
    digest = es.skeleton.digest()
    holders = [e["id"] for e in entries if e["skeleton"].digest() == digest]
    if ref in holders:
        return ref
    return holders[0] if holders else None


def skeleton_label(entry: dict) -> str:
    """How a library skeleton reads in a dropdown: a preset's name already says its size."""
    if entry["builtin"]:
        return f"{entry['name']} · preset"
    return f"{entry['name']} · {len(entry['skeleton'])} points"


def fill_skeleton_combo(combo: QComboBox, entries: list[dict]) -> None:
    """Every library skeleton in ``combo``, its id as the item data."""
    for entry in entries:
        combo.addItem(skeleton_label(entry), entry["id"])


class NewExampleSetDialog(Dialog):
    """Name a new example set and choose its skeleton (§26): what **New example set…** opens.
    The skeleton opens on the one chosen in the picker; it can still change in the marking
    window until the first mark."""

    def __init__(
        self,
        skeleton_entries: list[dict],
        chosen_skeleton_id: str | None,
        default_set_name: str,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New example set")
        self.resize(lines(36), lines(8))
        form = QFormLayout()
        self._name = QLineEdit(default_set_name)
        self._name.textEdited.connect(lambda _t: self._sync())
        form.addRow("Name:", self._name)
        self._skeleton = QComboBox()
        fill_skeleton_combo(self._skeleton, skeleton_entries)
        index = self._skeleton.findData(chosen_skeleton_id)
        self._skeleton.setCurrentIndex(max(index, 0))
        form.addRow("Skeleton:", self._skeleton)
        self.body().addLayout(form)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Create and mark frames")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        self._sync()

    def name(self) -> str:
        return self._name.text().strip()

    def skeleton_id(self) -> str | None:
        return self._skeleton.currentData()

    def _sync(self) -> None:
        set_available(self._ok, bool(self.name()), "Give the example set a name.")


def _read(experiment) -> tuple[list[dict], list[ExampleSet]]:
    return library.skeletons(), experiment.example_sets("example")


class ExampleSetPicker(QWidget):
    """Choose a skeleton and one of this experiment's example sets on it, or make one.

    ``changed`` fires when the chosen set changes (including to none). :meth:`set_id` is the
    chosen set.
    """

    changed = pyqtSignal()

    def __init__(self, experiment=None, *, parent=None) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._entries: list[dict] = []
        self._sets: list[ExampleSet] = []
        self._reading = False
        self._pending_select: str | None = None
        self._last: str | None = None

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self._skeleton = QComboBox()
        self._skeleton.activated.connect(lambda _i: self._fill_sets(keep=None))
        form.addRow("Skeleton:", self._skeleton)
        set_row = QHBoxLayout()
        self._set = QComboBox()
        self._set.currentIndexChanged.connect(lambda _i: self._set_chosen())
        set_row.addWidget(self._set, 1)
        self._counts = QLabel("")
        set_row.addWidget(self._counts)
        form.addRow("Example set:", set_row)
        buttons = QHBoxLayout()
        self._new = QPushButton("New example set…")
        self._new.clicked.connect(self._create)
        buttons.addWidget(self._new)
        self._mark = QPushButton("Mark frames…")
        self._mark.clicked.connect(self._mark_frames)
        buttons.addWidget(self._mark)
        self._delete = QPushButton("Delete set…")
        self._delete.clicked.connect(self._delete_set)
        buttons.addWidget(self._delete)
        buttons.addStretch()
        form.addRow("", buttons)
        self._sync()
        if experiment is not None:
            self.refresh()

    # -- public -------------------------------------------------------------- #
    def set_experiment(self, experiment) -> None:
        """Show ``experiment``'s sets (re-read off the UI thread)."""
        self._experiment = experiment
        self.refresh()

    def set_id(self) -> str | None:
        """The chosen example set, or None."""
        return self._set.currentData() or None

    def skeleton_id(self) -> str | None:
        """The chosen library skeleton, or None (none chosen, or the unlinked sets)."""
        return self._skeleton.currentData() or None

    def select(self, set_id: str) -> None:
        """Choose ``set_id`` (and its skeleton) once the sets are read."""
        if self._reading or not self._sets:
            self._pending_select = set_id
            return
        es = next((s for s in self._sets if s.id == set_id), None)
        if es is None:
            return
        owner = skeleton_of(es, self._entries) or ""
        index = self._skeleton.findData(owner)
        if index >= 0:
            self._skeleton.setCurrentIndex(index)
        self._fill_sets(keep=set_id)

    def refresh(self, select: str | None = None) -> None:
        """Re-read the library and the experiment's sets off the UI thread."""
        if select is not None:
            self._pending_select = select
        elif self.set_id() is not None and self._pending_select is None:
            self._pending_select = self.set_id()
        if self._experiment is None:
            return
        experiment = self._experiment
        self._reading = True
        self._sync()
        worker = BackgroundCall(lambda: _read(experiment), receiver=self)
        worker.done.connect(self._loaded)
        worker.failed.connect(self._read_failed)
        worker.finished.connect(worker.deleteLater)
        self._reader = worker
        worker.start()

    # -- reading ------------------------------------------------------------------ #
    def _read_failed(self, exc: BaseException) -> None:
        self._reading = False
        self._counts.setText(f"(could not read: {exc})")
        self._sync()

    def _loaded(self, result) -> None:
        self._reading = False
        entries, sets = result
        self._entries = list(entries)
        self._sets = list(sets)
        keep_skeleton = self.skeleton_id()
        self._skeleton.blockSignals(True)
        self._skeleton.clear()
        fill_skeleton_combo(self._skeleton, self._entries)
        if any(skeleton_of(es, self._entries) is None for es in self._sets):
            self._skeleton.addItem(UNLINKED, "")
        self._skeleton.blockSignals(False)
        pending, self._pending_select = self._pending_select, None
        if pending is not None and any(s.id == pending for s in self._sets):
            self.select(pending)
            return
        if keep_skeleton is None and self._sets:
            # Open on the skeleton the experiment's first set uses, not on a preset no set
            # was made from.
            keep_skeleton = skeleton_of(self._sets[0], self._entries) or ""
        index = (
            self._skeleton.findData(keep_skeleton) if keep_skeleton is not None else -1
        )
        self._skeleton.setCurrentIndex(max(index, 0))
        self._fill_sets(keep=None)

    def _sets_on(self, skeleton_id: str | None) -> list[ExampleSet]:
        if skeleton_id is None:
            return []
        return [
            s
            for s in self._sets
            if (skeleton_of(s, self._entries) or "") == skeleton_id
        ]

    def _fill_sets(self, *, keep: str | None) -> None:
        skeleton_id = self._skeleton.currentData()
        shown = self._sets_on(skeleton_id)
        self._set.blockSignals(True)
        self._set.clear()
        for es in shown:
            self._set.addItem(f"{es.name} ({es.id})", es.id)
        index = self._set.findData(keep) if keep is not None else -1
        if index < 0 and self._last is not None:
            index = self._set.findData(self._last)
        self._set.setCurrentIndex(max(index, 0) if shown else -1)
        self._set.blockSignals(False)
        self._set_chosen()

    def _set_chosen(self) -> None:
        es = next((s for s in self._sets if s.id == self.set_id()), None)
        if es is None:
            self._counts.setText("")
        else:
            counts = es.counts()
            self._counts.setText(f"{counts['labeled']} / {counts['drawn']} frames")
        self._sync()
        if self.set_id() != self._last:
            self._last = self.set_id()
            self.changed.emit()

    def _sync(self) -> None:
        if self._experiment is None:
            reason = "Open an experiment first."
        elif self._reading:
            reason = "Reading the example sets…"
        else:
            reason = None
        new_reason = reason
        set_available(self._new, new_reason is None, new_reason or "")
        chosen_reason = reason or (
            None if self.set_id() is not None else "Choose an example set first."
        )
        set_available(self._mark, chosen_reason is None, chosen_reason or "")
        set_available(self._delete, chosen_reason is None, chosen_reason or "")
        if self._set.count() == 0 and not self._reading:
            self._set.setPlaceholderText("(no example sets on this skeleton yet)")

    # -- actions ------------------------------------------------------------------ #
    def _default_name(self) -> str:
        taken = {s.name for s in self._sets}
        n = len(taken) + 1
        while f"example set {n}" in taken:
            n += 1
        return f"example set {n}"

    def _create(self) -> None:
        if self._experiment is None or not self._entries:
            return
        dialog = NewExampleSetDialog(
            self._entries, self.skeleton_id(), self._default_name(), parent=self
        )
        if dialog.exec() != Dialog.DialogCode.Accepted or not dialog.skeleton_id():
            return
        try:
            made = self._experiment.create_example_set(
                dialog.name(), dialog.skeleton_id()
            )
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "New example set", str(exc))
            return
        from detail.gui.keypoint_labeling import open_labeling

        open_labeling(self, self._experiment, made.id)
        self.refresh(select=made.id)

    def _mark_frames(self) -> None:
        set_id = self.set_id()
        if set_id is None or self._experiment is None:
            return
        from detail.gui.keypoint_labeling import open_labeling

        open_labeling(self, self._experiment, set_id)
        self.refresh(select=set_id)

    def _delete_set(self) -> None:
        set_id = self.set_id()
        if set_id is None or self._experiment is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete example set",
            f"Delete the example set {self._set.currentText()}? Its frames and marks are "
            "removed; a keypoint run that lists it refuses the delete.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_example_set(set_id)
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.warning(self, "Delete example set", str(exc))
            return
        self._last = None
        self._pending_select = None
        self.refresh()
