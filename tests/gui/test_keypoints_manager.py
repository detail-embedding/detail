"""The Keypoints manager (§26): the skeleton library as a table, and the skeleton editor
dialog that New and Edit… open.

The canvas gestures are driven as real mouse events (a click adds, a drag moves, a right click
removes, Shift+click joins); every save goes to the real library on disk, and dependents are
real example sets in a real experiment under the patched experiments root."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from detail.api import skeletons as library  # noqa: E402
from detail.api.experiment import Experiment  # noqa: E402
from detail.core.skeleton import PRESETS, Skeleton  # noqa: E402
from detail.gui import keypoints_manager as mod  # noqa: E402
from detail.gui import skeleton_editor  # noqa: E402
from detail.gui.dialog import Dialog  # noqa: E402
from detail.gui.tables import row_id  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

TWO = Skeleton(names=("a", "b"), layout=((0.2, 0.5), (0.8, 0.5)))
LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
NONE, SHIFT = Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier


@pytest.fixture
def root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(locations, "experiments_root", lambda: workspace)
    return workspace


def _settle(qapp, page):
    drain_until(qapp, lambda: not page._reading)
    qapp.processEvents()


def _page(qapp):
    page = mod.KeypointsManagerPage(experiment=None)
    page.resize(1000, 600)
    page.show()
    _settle(qapp, page)
    return page


def _dialog(qapp, entry=None, dependents=(), taken=()):
    dialog = mod.SkeletonEditorDialog(
        entry, list(dependents), set(taken), experiment=None
    )
    dialog.show()
    qapp.processEvents()
    return dialog


def _at(canvas, u, v):
    rect = canvas._image_rect()
    return QPointF(rect.x() + u * rect.width(), rect.y() + v * rect.height())


def _mouse(canvas, kind, pos, button, modifiers=NONE, held=None):
    event = QMouseEvent(
        kind, pos, button, held if held is not None else button, modifiers
    )
    {
        QEvent.Type.MouseButtonPress: canvas.mousePressEvent,
        QEvent.Type.MouseMove: canvas.mouseMoveEvent,
        QEvent.Type.MouseButtonRelease: canvas.mouseReleaseEvent,
    }[kind](event)


def _click(canvas, u, v, button=LEFT, modifiers=NONE):
    pos = _at(canvas, u, v)
    _mouse(canvas, QEvent.Type.MouseButtonPress, pos, button, modifiers)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, pos, button, modifiers)


def _drag(canvas, u0, v0, u1, v1, button=LEFT):
    _mouse(canvas, QEvent.Type.MouseButtonPress, _at(canvas, u0, v0), button)
    for t in (0.3, 0.7, 1.0):
        pos = _at(canvas, u0 + (u1 - u0) * t, v0 + (v1 - v0) * t)
        _mouse(canvas, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, held=button)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, _at(canvas, u1, v1), button)


def _answer(monkeypatch, button):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: button)


def _save(qapp, dialog):
    dialog._save.click()
    drain_until(qapp, lambda: dialog.saved_id() is not None)
    return dialog.saved_id()


def _choose(qapp, page, skeleton_id):
    view = page._table.view
    model = view.model()
    for r in range(model.rowCount()):
        if row_id(model.index(r, 0)) == skeleton_id:
            view.selectRow(r)
            qapp.processEvents()
            return
    raise AssertionError(f"no row {skeleton_id}")


# --- the page: a table ------------------------------------------------------------- #
def test_the_page_is_a_table_of_presets_then_the_library(qapp, root):
    made = library.create_skeleton("two", TWO)
    page = _page(qapp)
    model = page._table.view.model()
    names = [model.index(r, 0).data() for r in range(model.rowCount())]
    assert names == [label for label, _sk in PRESETS.values()] + ["two"]
    kinds = [model.index(r, 1).data() for r in range(model.rowCount())]
    assert kinds == ["preset"] * len(PRESETS) + ["library"]
    assert page.selected_id() is None, "opens on the table, not on an editor"
    assert not page.findChildren(
        skeleton_editor.SkeletonCanvas
    ), "no canvas on the page"
    for button in (page._edit_button, page._export, page._delete):
        assert not button.isEnabled() and "Select a skeleton" in button.toolTip()
    _choose(qapp, page, "quadruped_side_5")
    assert page._edit_button.isEnabled() and page._export.isEnabled()
    assert not page._delete.isEnabled() and "preset" in page._delete.toolTip()
    _choose(qapp, page, made["id"])
    assert page._delete.isEnabled()


def test_new_and_edit_open_the_editor_and_a_save_refreshes_the_table(
    qapp, root, monkeypatch
):
    made = library.create_skeleton("two", TWO)
    page = _page(qapp)
    opened = []

    def fake_exec(dialog):
        opened.append((dialog._entry["id"] if dialog._entry else None, dialog))
        return Dialog.DialogCode.Rejected

    monkeypatch.setattr(mod.SkeletonEditorDialog, "exec", fake_exec)
    page._new.click()
    assert opened[-1][0] is None
    assert opened[-1][1].editor().skeleton().names == (), "New opens empty"
    _choose(qapp, page, made["id"])
    page._edit_button.click()
    assert opened[-1][0] == made["id"]
    index = page._table.view.model().index(len(PRESETS), 0)
    page._table.view.doubleClicked.emit(index)
    assert opened[-1][0] == made["id"], "double-click is Edit…"

    def saving_exec(dialog):
        dialog._saved = made["id"]
        return Dialog.DialogCode.Accepted

    monkeypatch.setattr(mod.SkeletonEditorDialog, "exec", saving_exec)
    page._edit_button.click()
    assert page._reading, "a save re-reads the library"
    _settle(qapp, page)
    assert page.selected_id() == made["id"]


def test_a_used_skeleton_cannot_be_deleted_and_says_by_what(qapp, root):
    made = library.create_skeleton("two", TWO)
    exp = Experiment.create("lab", root)
    exp.create_example_set("overhead", made["id"])
    page = _page(qapp)
    assert page._table.view.model().index(len(PRESETS), 3).data() == "1 example set"
    _choose(qapp, page, made["id"])
    assert not page._delete.isEnabled()
    assert "example set in lab; delete those first" in page._delete.toolTip()


def test_delete_asks_then_removes_an_unused_skeleton(qapp, root, monkeypatch):
    made = library.create_skeleton("two", TWO)
    page = _page(qapp)
    _choose(qapp, page, made["id"])
    _answer(monkeypatch, QMessageBox.StandardButton.No)
    page._delete.click()
    assert library.skeleton(made["id"])
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    page._delete.click()
    drain_until(qapp, lambda: page._reading)
    _settle(qapp, page)
    assert [e for e in library.skeletons() if not e["builtin"]] == []
    assert page._table.view.model().rowCount() == len(PRESETS)


def test_export_replaces_the_file_the_save_dialog_confirmed(
    qapp, root, tmp_path, monkeypatch
):
    page = _page(qapp)
    _choose(qapp, page, "quadruped_side_5")
    out = tmp_path / "out.yaml"
    out.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
    )
    page._export.click()
    assert out.read_text(encoding="utf-8-sig").startswith("schema_version: 1\n")


# --- the dialog: drawing -------------------------------------------------------------- #
def test_a_new_skeleton_is_drawn_on_an_empty_plain_canvas_then_saved(qapp, root):
    dialog = _dialog(qapp, taken={"skeleton 1"})
    editor = dialog.editor()
    assert editor.skeleton().names == () and dialog._name.text() == "skeleton 2"
    assert not dialog._save.isEnabled(), "an empty skeleton cannot be saved"
    pixel = editor.canvas()._pixmap.toImage().pixelColor(5, 5)
    assert pixel.red() == skeleton_editor.BLANK[0, 0, 0], "no video frame behind it"
    canvas = editor.canvas()
    _click(canvas, 0.5, 0.2)
    _click(canvas, 0.5, 0.8)
    assert editor._names == ["point_1", "point_2"]
    editor._point.setText("tail")
    editor._point.editingFinished.emit()
    _click(canvas, 0.5, 0.2)
    _click(canvas, 0.5, 0.8, modifiers=SHIFT)
    dialog._name.setText("my rats")
    dialog._name.textEdited.emit("my rats")
    assert dialog._save.text() == "Save" and dialog._save.isEnabled()

    saved = _save(qapp, dialog)

    sk = library.skeleton(saved)["skeleton"]
    assert library.skeleton(saved)["name"] == "my rats"
    assert sk.names == ("point_1", "tail") and sk.edges == (("point_1", "tail"),)
    assert [round(v, 2) for v in sk.layout[0]] == [0.5, 0.2], "the layout as drawn"


def test_drag_moves_right_click_removes_shift_click_unjoins(qapp, root):
    made = library.create_skeleton("two", TWO)
    dialog = _dialog(qapp, library.skeleton(made["id"]))
    assert not dialog._save.isEnabled() and "Nothing changed" in dialog._save.toolTip()
    editor = dialog.editor()
    canvas = editor.canvas()
    _drag(canvas, 0.2, 0.5, 0.3, 0.7)
    assert [round(v, 2) for v in editor._layout[0]] == [0.3, 0.7]
    assert dialog._save.text() == "Save" and dialog._save.isEnabled()
    _click(canvas, 0.3, 0.7)
    _click(canvas, 0.8, 0.5, modifiers=SHIFT)
    assert editor._edges == [("a", "b")]
    _click(canvas, 0.8, 0.5, modifiers=SHIFT)
    assert editor._edges == []
    _click(canvas, 0.8, 0.5, button=RIGHT)
    assert editor._names == ["a"]
    saved = _save(qapp, dialog)
    assert saved == made["id"], "nothing uses it: saved in place"
    assert library.skeleton(made["id"])["skeleton"].names == ("a",)


def test_pair_and_side_from_the_point_row(qapp, root):
    made = library.create_skeleton("two", TWO)
    dialog = _dialog(qapp, library.skeleton(made["id"]))
    editor = dialog.editor()
    _click(editor.canvas(), 0.2, 0.5)
    index = editor._pair.findData("b")
    editor._pair.setCurrentIndex(index)
    editor._pair.activated.emit(index)
    assert editor._pairs == [("a", "b")] and editor._side.isEnabled()
    editor._side.setCurrentIndex(1)
    editor._side.activated.emit(1)
    assert editor._pairs == [("b", "a")]


# --- the dialog: what may change ------------------------------------------------------ #
def test_a_used_skeleton_saves_as_new_but_renames_in_place(qapp, root):
    made = library.create_skeleton("two", TWO)
    used = [{"kind": "example set", "experiment": "lab"}]
    dialog = _dialog(qapp, library.skeleton(made["id"]), used, {"two"})
    assert "Used by example set in lab" in dialog._status.text()
    dialog._name.setText("two renamed")
    dialog._name.textEdited.emit("two renamed")
    assert dialog._save.text() == "Save", "a name is not identity"
    _drag(dialog.editor().canvas(), 0.2, 0.5, 0.3, 0.7)
    assert dialog._save.text() == "Save as new"

    saved = _save(qapp, dialog)

    assert saved != made["id"]
    assert library.skeleton(saved)["derived_from"] == made["id"]
    assert library.skeleton(made["id"])["skeleton"].layout == TWO.layout, "untouched"


def test_editing_a_preset_saves_as_new_under_a_free_name(qapp, root):
    entry = library.skeleton("midline_3")
    dialog = _dialog(qapp, entry, [], {entry["name"]})
    assert dialog._save.text() == "Save as new"
    x, y = dialog.editor()._layout[0]
    _drag(dialog.editor().canvas(), x, y, 0.1, 0.1)
    saved = _save(qapp, dialog)
    made = library.skeleton(saved)
    assert made["derived_from"] == "midline_3"
    assert made["name"] == f"{PRESETS['midline_3'][0]} 2"


def test_cancel_with_edits_asks(qapp, root, monkeypatch):
    made = library.create_skeleton("two", TWO)
    dialog = _dialog(qapp, library.skeleton(made["id"]))
    _drag(dialog.editor().canvas(), 0.2, 0.5, 0.3, 0.5)
    _answer(monkeypatch, QMessageBox.StandardButton.Cancel)
    dialog.reject()
    assert dialog.isVisible(), "Cancel kept the window"
    _answer(monkeypatch, QMessageBox.StandardButton.Discard)
    dialog.reject()
    assert not dialog.isVisible()
    assert library.skeleton(made["id"])["skeleton"].layout == TWO.layout, "not written"


def test_the_used_by_cell_counts_each_kind():
    found = [
        {"kind": "example set"},
        {"kind": "example set"},
        {"kind": "trained model"},
    ]
    assert mod.used_by_text(found) == "2 example sets, 1 trained model"
    assert mod.used_by_text([]) == "—" and mod.used_by_text(None) == "…"
