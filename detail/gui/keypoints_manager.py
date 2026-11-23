"""The Keypoints manager (§26 *The Keypoints manager*): the top bar's home for skeletons.

A skeleton is defined once, in the machine-wide library (``api/skeletons.py``), and copied by
whatever uses it. :class:`KeypointsManagerPage` is the library as a table, the shape of the
app's other list pages: **+ New skeleton** and **Import…** above it, **Edit…**, **Export…** and
**Delete** below, double-click to edit. Drawing happens in :class:`SkeletonEditorDialog`, the
skeleton editor (``gui/skeleton_editor.py``) in a window.

**What may change.** A skeleton nothing uses is edited and deleted in place. One with
**dependents** (an example set, a keypoint run, a dataset or a trained model in a known
experiment, found by a live search) keeps its points: the dialog's **Save** becomes **Save as
new**, which writes the edit as a new skeleton, and **Delete** is greyed with what uses it. Its
name may still change. A preset is always frozen. The API refuses the same things
(``update_skeleton`` / ``delete_skeleton``), so the screen's state is a courtesy, not the guard.

The library and the dependents are read off the UI thread (one ``BackgroundCall``), and the
writes that re-run the dependents search go through ``run_busy``.
"""

from __future__ import annotations

from pathlib import Path

from detail.api import skeletons as library
from detail.core.skeleton import Skeleton
from detail.gui.busy import run_busy
from detail.gui.dialog import Dialog
from detail.gui.pages import IndexTable, _heading, _index_caption
from detail.gui.qt import (
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from detail.gui.sizing import lines, set_available
from detail.gui.skeleton_editor import SkeletonEditor
from detail.gui.tables import row_id
from detail.gui.workers import BackgroundCall

_EMPTY_TEXT = "(no skeletons yet: + New skeleton draws one)"


def _read_library() -> tuple[list[dict], dict[str, list[dict]]]:
    """Off the UI thread: every skeleton, and what uses each (a workspace scan)."""
    return library.skeletons(), library.dependents_by_skeleton()


def used_by_text(found: list[dict] | None) -> str:
    """A *Used by* cell: ``2 example sets, 1 keypoint detect run``, or a dash."""
    if found is None:
        return "…"
    if not found:
        return "—"
    counts: dict[str, int] = {}
    for dep in found:
        counts[dep["kind"]] = counts.get(dep["kind"], 0) + 1
    return ", ".join(f"{n} {kind}{'s' if n > 1 else ''}" for kind, n in counts.items())


def describe_dependents(found: list[dict]) -> str:
    """``example set in lab a, keypoint detect run in lab b and 2 more``."""
    shown = [f"{d['kind']} in {d['experiment']}" for d in found[:3]]
    more = f" and {len(found) - 3} more" if len(found) > 3 else ""
    return ", ".join(shown) + more


def _free_name(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    n = 2
    while f"{name} {n}" in taken:
        n += 1
    return f"{name} {n}"


class SkeletonEditorDialog(Dialog):
    """Draw one skeleton: a new one (``skeleton_entry`` None, an empty canvas) or a library or
    preset entry. ``skeleton_dependents`` is what uses it (already read by the page), which decides whether
    **Save** writes in place or as a new skeleton. :meth:`saved_id` is the skeleton written,
    or None. **Cancel** with edits asks first.
    """

    def __init__(
        self,
        skeleton_entry: dict | None,
        skeleton_dependents: list[dict],
        taken_skeleton_names: set[str],
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        entry = skeleton_entry
        self._entry = entry
        self._dependents = list(skeleton_dependents)
        self._taken = set(taken_skeleton_names)
        self._saved: str | None = None
        self._saving = False
        if entry is None:
            n = 1
            while f"skeleton {n}" in self._taken:
                n += 1
            self._base_name = f"skeleton {n}"
            self._base = Skeleton(names=())
            self.setWindowTitle("New skeleton")
        else:
            self._base_name = entry["name"]
            self._base = entry["skeleton"]
            self.setWindowTitle(f"Edit skeleton: {entry['name']}")
        self.resize(lines(60), lines(44))

        body = self.body()
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name = QLineEdit(self._base_name)
        self._name.textEdited.connect(lambda _t: self._sync())
        name_row.addWidget(self._name, 1)
        body.addLayout(name_row)
        self._editor = SkeletonEditor()
        self._editor.load(self._base)
        self._editor.edited.connect(self._sync)
        body.addWidget(self._editor, 1)
        self._status = QLabel(self._status_text())
        self._status.setWordWrap(True)
        body.addWidget(self._status)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._save = box.button(QDialogButtonBox.StandardButton.Save)
        box.accepted.connect(self.save)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        self._sync()

    # -- public -------------------------------------------------------------- #
    def editor(self) -> SkeletonEditor:
        return self._editor

    def saved_id(self) -> str | None:
        return self._saved

    def save_mode(self) -> str:
        """``new`` (not yet in the library), ``update`` or ``as_new``."""
        if self._entry is None:
            return "new"
        if self._entry["builtin"]:
            return "as_new"
        if self._structure_changed() and self._dependents:
            return "as_new"
        return "update"

    def dirty(self) -> bool:
        return self._name.text().strip() != self._base_name or self._structure_changed()

    def problem(self) -> str | None:
        """Why the skeleton cannot be saved yet, or None."""
        if not self._name.text().strip():
            return "Give the skeleton a name."
        return self._editor.problem()

    def save(self) -> None:
        """Write the skeleton (off the UI thread: an update re-runs the dependents search)
        and close on success."""
        if self.problem() or self._saving:
            return
        mode = self.save_mode()
        name = self._name.text().strip()
        sk = self._editor.skeleton()
        if mode == "as_new" and name == self._base_name:
            name = _free_name(name, self._taken)
        source_id = self._entry["id"] if self._entry is not None else None
        changed = self._structure_changed()

        def write() -> str:
            if mode == "new":
                return library.create_skeleton(name, sk)["id"]
            if mode == "as_new":
                return library.save_as_new(source_id, name, sk)["id"]
            return library.update_skeleton(
                source_id, name=name, skeleton=sk if changed else None
            )["id"]

        def done(saved_id) -> None:
            self.let_go()
            self._saving = False
            self._saved = saved_id
            self.accept()

        def failed(exc: BaseException) -> None:
            self.let_go()
            self._saving = False
            QMessageBox.warning(self, "Save skeleton", str(exc))

        self._saving = True
        worker = run_busy(
            self, write, done, label="Saving the skeleton…", on_error=failed
        )
        # A save cannot be taken back half-written; closing waits for it, so the library the
        # caller re-reads on Accepted is never one the dialog closed in front of.
        self.hold_open_while(worker, self._still_saving)

    def _still_saving(self) -> None:
        self._status.setText(
            "Still saving the skeleton — this closes when it is written."
        )

    def reject(self) -> None:
        if self._saving:
            super().reject()  # held while the save runs (`hold_open_while`)
            return
        if self.dirty():
            answer = QMessageBox.question(
                self,
                "Unsaved changes",
                "Discard the changes to this skeleton?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                return
        super().reject()

    # -- state ------------------------------------------------------------------- #
    def _structure_changed(self) -> bool:
        return self._editor.skeleton().to_dict() != self._base.to_dict()

    def _status_text(self) -> str:
        if self._entry is None:
            return "A new skeleton: click the canvas to add points."
        if self._entry["builtin"]:
            return "A built-in preset: an edit is saved as a new skeleton."
        if self._dependents:
            return (
                f"Used by {describe_dependents(self._dependents)}. Its points are fixed "
                "while anything uses it: an edit is saved as a new skeleton. Its name can "
                "still change."
            )
        return "Nothing uses this skeleton yet: an edit changes it in place."

    def _sync(self) -> None:
        self._save.setText("Save as new" if self.save_mode() == "as_new" else "Save")
        problem = self.problem()
        pending = self.dirty() or self._entry is None
        set_available(
            self._save, problem is None and pending, problem or "Nothing changed."
        )


class KeypointsManagerPage(QWidget):
    """The skeleton library as a table (§26)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: dict[str, dict] = {}
        self._order: list[str] = []
        self._deps: dict[str, list[dict]] | None = None
        self._reading = False  # rows in flight: the empty notice is not yet a claim
        self._pending_select: str | None = None
        self._dialog: SkeletonEditorDialog | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Keypoints manager"))
        intro = QLabel(
            "A skeleton is the points a keypoint model finds, the lines that join them and "
            "which points are a left/right pair. Skeletons are shared by every experiment on "
            "this machine; whatever uses one keeps its own copy."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        top = QHBoxLayout()
        self._new = QPushButton("+ New skeleton")
        self._new.clicked.connect(self._new_skeleton)
        top.addWidget(self._new)
        self._import = QPushButton("Import…")
        self._import.clicked.connect(self._import_file)
        top.addWidget(self._import)
        top.addStretch()
        layout.addLayout(top)
        self._caption = _index_caption("Reading…", rows=None)
        layout.addWidget(self._caption)
        self._table = IndexTable(
            ["Name", "Kind", "Points", "Used by"], empty_text="(reading…)"
        )
        self._table.view.doubleClicked.connect(lambda idx: self._edit(row_id(idx)))
        self._table.view.selectionModel().selectionChanged.connect(
            lambda *_a: self._sync()
        )
        layout.addWidget(self._table)
        below = QHBoxLayout()
        self._edit_button = QPushButton("Edit…")
        self._edit_button.clicked.connect(lambda: self._edit(self.selected_id()))
        below.addWidget(self._edit_button)
        self._export = QPushButton("Export…")
        self._export.clicked.connect(self._export_selected)
        below.addWidget(self._export)
        self._delete = QPushButton("Delete")
        self._delete.clicked.connect(self._delete_selected)
        below.addWidget(self._delete)
        below.addStretch()
        layout.addLayout(below)
        layout.addStretch()
        self._sync()
        self.refresh()

    # -- reading ------------------------------------------------------------------ #
    def refresh(self, select: str | None = None) -> None:
        """Re-read the library and its dependents off the UI thread."""
        if select is not None:
            self._pending_select = select
        self._reading = True
        self._sync()
        worker = BackgroundCall(_read_library, receiver=self)
        worker.done.connect(self._loaded)
        worker.failed.connect(self._read_failed)
        worker.finished.connect(worker.deleteLater)
        self._reader = worker
        worker.start()

    def _read_failed(self, exc: BaseException) -> None:
        self._reading = False
        self._caption.setText(f"(could not read the library: {exc})")
        self._sync()

    def _loaded(self, result) -> None:
        self._reading = False
        entries, deps = result
        keep = self._pending_select or self.selected_id()
        self._pending_select = None
        self._entries = {e["id"]: e for e in entries}
        self._order = [e["id"] for e in entries]
        self._deps = dict(deps)
        rows = [
            (
                e["name"],
                "preset" if e["builtin"] else "library",
                len(e["skeleton"]),
                used_by_text(self._deps.get(e["id"], [])),
            )
            for e in entries
        ]
        self._table.set_rows(rows, ids=self._order)
        self._table.set_empty_text(_EMPTY_TEXT)
        self._caption.setText(f"{len(rows)} skeleton(s)")
        if keep in self._entries:
            self.select(keep)
        self._sync()

    # -- selection ------------------------------------------------------------------ #
    def select(self, skeleton_id: str) -> None:
        """Select ``skeleton_id``'s row (by identity, whatever the sort)."""
        model = self._table.view.model()
        for r in range(model.rowCount()):
            if row_id(model.index(r, 0)) == skeleton_id:
                self._table.view.selectRow(r)
                return

    def selected_id(self) -> str | None:
        view = self._table.view
        index = view.currentIndex()
        if not index.isValid() or not view.selectionModel().hasSelection():
            return None
        return row_id(index)

    def _dependents(self, skeleton_id: str | None) -> list[dict] | None:
        if skeleton_id is None or self._deps is None:
            return None
        return self._deps.get(skeleton_id, [])

    def _edit_reason(self, skeleton_id: str | None) -> str | None:
        if skeleton_id is None:
            return "Select a skeleton first."
        if self._reading or self._deps is None:
            return "Checking what uses each skeleton…"
        return None

    def _delete_reason(self, skeleton_id: str | None) -> str | None:
        reason = self._edit_reason(skeleton_id)
        if reason:
            return reason
        entry = self._entries.get(skeleton_id)
        if entry is not None and entry["builtin"]:
            return "A built-in preset cannot be deleted."
        found = self._dependents(skeleton_id)
        if found:
            return f"Used by {describe_dependents(found)}; delete those first."
        return None

    def _sync(self) -> None:
        chosen = self.selected_id()
        reason = self._edit_reason(chosen)
        set_available(self._edit_button, reason is None, reason or "")
        set_available(self._export, chosen is not None, "Select a skeleton first.")
        reason = self._delete_reason(chosen)
        set_available(self._delete, reason is None, reason or "")
        set_available(self._new, not self._reading, "Checking what uses each skeleton…")

    # -- actions ------------------------------------------------------------------- #
    def _open_editor(self, entry: dict | None) -> None:
        deps = self._dependents(entry["id"]) if entry is not None else []
        dialog = SkeletonEditorDialog(
            entry,
            deps or [],
            {e["name"] for e in self._entries.values()},
            parent=self,
        )
        self._dialog = dialog
        if dialog.exec() == Dialog.DialogCode.Accepted and dialog.saved_id():
            self.refresh(select=dialog.saved_id())

    def _new_skeleton(self) -> None:
        if self._reading:
            return
        self._open_editor(None)

    def _edit(self, skeleton_id: str | None) -> None:
        if self._edit_reason(skeleton_id) is not None:
            return
        self._open_editor(self._entries[skeleton_id])

    def _import_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import skeleton", "", "Skeleton (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            made = library.import_skeleton(Path(path))
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.warning(self, "Import skeleton", str(exc))
            return
        self.refresh(select=made["id"])

    def _export_selected(self) -> None:
        skeleton_id = self.selected_id()
        if skeleton_id is None:
            return
        entry = self._entries[skeleton_id]
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export skeleton", f"{entry['name']}.yaml", "Skeleton (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            # The save dialog has already asked about replacing an existing file.
            library.export_skeleton(skeleton_id, Path(path), overwrite=True)
        except OSError as exc:
            QMessageBox.warning(self, "Export skeleton", str(exc))

    def _delete_selected(self) -> None:
        skeleton_id = self.selected_id()
        if self._delete_reason(skeleton_id) is not None:
            return
        answer = QMessageBox.question(
            self,
            "Delete skeleton",
            f"Delete the skeleton {self._entries[skeleton_id]['name']!r}? The delete is "
            "refused if anything has started using it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        run_busy(
            self,
            lambda: library.delete_skeleton(skeleton_id),
            lambda _result: self.refresh(),
            label="Deleting the skeleton…",
            on_error=lambda exc: QMessageBox.warning(self, "Delete skeleton", str(exc)),
        )
