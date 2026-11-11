"""The Notes box (`gui/notes.py`, §15): on every entry page, autosaved, kept across rebuilds.

The class this guards is the one `wire-every-surface` exists for: **a capability added to the
page being worked on and to none of the others.** So the placement guard is driven from
`storage.notes.KINDS` and from the shell's own dispatcher rather than from a list of pages
somebody remembered — a new entry kind with no page placing its notes fails, and so does a new
rail leaf whose kind has no notes.
"""

import ast
import inspect
import re
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QPalette  # noqa: E402
from PyQt6.QtWidgets import QLabel, QPushButton  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import app as gui_app  # noqa: E402
from detail.gui import comparison as gui_comparison  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.gui.collapse import CollapsibleBand  # noqa: E402
from detail.gui import notes as gui_notes  # noqa: E402
from detail.gui.notes import NotesBox  # noqa: E402
from detail.storage import datasets as datasets_store  # noqa: E402
from detail.storage import notes as notes_store  # noqa: E402
from detail.storage import runs as runs_store  # noqa: E402

#: The page that shows each entry kind.
PAGE_FOR_KIND = {
    "experiment": pages.ExperimentOverviewPage,
    "recording": pages.RecordingDetailPage,
    "run": pages.RunDetailPage,
    "analysis_run": pages.AnalysisRunDetailPage,
    "dataset": pages.DatasetDetailPage,
    "model": pages.ModelDetailPage,
    "distill": pages.DistillRunDetail,
    "export_run": pages.ExportRunDetailPage,
    "comparison": gui_comparison.ComparisonDetailPage,
}

_PLACERS = {"NotesBox", "place_notes", "_add_notes"}


def _kinds_placed_by(cls) -> set[str]:
    """Every string literal handed to a Notes placer anywhere in ``cls``'s own source."""
    tree = ast.parse(inspect.getsource(cls).lstrip())
    kinds = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in _PLACERS:
            kinds |= {
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            }
    return kinds


def test_every_notes_kind_has_a_page():
    assert set(PAGE_FOR_KIND) == set(notes_store.KINDS)


@pytest.mark.parametrize("kind", sorted(PAGE_FOR_KIND))
def test_every_entry_page_places_its_notes(kind):
    assert kind in _kinds_placed_by(PAGE_FOR_KIND[kind])


def test_every_rail_entry_kind_has_notes():
    """A rail leaf that opens one entry — `experiment, <id> = payload` in `_navigate` — is an
    entry the user can be on, so it is a notes kind."""
    source = inspect.getsource(gui_app.MainWindow._navigate)
    leaves = set(re.findall(r'kind == "(\w+)":\n\s+experiment, \w+ = payload', source))
    assert leaves, "the dispatcher's shape changed; this scan no longer reads it"
    assert leaves | {"experiment"} <= set(notes_store.KINDS)


# --------------------------------------------------------------------------- #
# behaviour
# --------------------------------------------------------------------------- #
def _experiment_with_run(tmp_path, status="complete") -> Experiment:
    experiment = Experiment.create("p", tmp_path)
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": [],
            "per_recording": {},
            "config": {},
            "status": status,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    return experiment


def _notes_file(experiment, kind, entry_id="") -> Path:
    return (
        notes_store.entry_dir(experiment.dir, kind, entry_id)
        / notes_store.NOTES_FILENAME
    )


def test_the_run_page_puts_notes_between_its_details_and_its_results(qapp, tmp_path):
    experiment = _experiment_with_run(tmp_path)
    experiment.set_notes("run", "run_001", "kept from last time")
    page = pages.RunDetailPage(experiment, "run_001")
    body = page._body
    order = [body.itemAt(i).widget() for i in range(body.count())]
    details = next(w for w in order if isinstance(w, CollapsibleBand))
    results = next(w for w in order if isinstance(w, QLabel) and w.text() == "Results")
    assert order.index(details) < order.index(page._notes_box) < order.index(results)
    assert page.findChild(NotesBox).text() == "kept from last time"


def test_a_page_carries_no_notes_button_of_its_own(qapp, tmp_path):
    """The fold control is the shell's (round 2): nothing on the page says "Notes ▸"."""
    experiment = _experiment_with_run(tmp_path)
    for page in (
        pages.RunDetailPage(experiment, "run_001"),
        pages.ExperimentOverviewPage(experiment, lambda _k: None),
    ):
        texts = [b.text() for b in page.findChildren(QPushButton)]
        assert not [t for t in texts if t.startswith("Notes")], texts


def test_folded_notes_cost_the_run_page_nothing(qapp, tmp_path):
    """Folded, the box is hidden outright, so the page is as tall as it was before notes
    existed — what `test_run_page_fits` needs."""
    experiment = _experiment_with_run(tmp_path)
    page = pages.RunDetailPage(experiment, "run_001")
    page.resize(1200, 600)
    page.show()
    qapp.processEvents()
    assert not page._notes_box.isVisible()
    folded = page.sizeHint().height()
    page._notes_box.set_open(True)
    qapp.processEvents()
    assert page._notes_box.isVisible()
    assert page.sizeHint().height() > folded
    page._build()  # the box survives a rebuild, still open
    qapp.processEvents()
    assert page._notes_box.isVisible()


@pytest.fixture
def shell(qapp, tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path / "root")
    window = gui_app.MainWindow()
    window.resize(1400, 900)
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def _show(qapp, shell, page):
    shell._show(page, "page")
    for _ in range(3):
        qapp.processEvents()
    return page


def test_the_shell_notes_button_opens_the_notes_only_on_the_page_you_are_on(
    qapp, tmp_path, shell
):
    """Folded by default on every page; opening the notes on one page leaves another page
    folded, and the same page opened afresh is folded again — nothing is remembered, across
    pages or sessions (user, review round 4: "I don't want the notes visibly to propagate
    across pages"; it replaced one app-wide `bands/notes`)."""
    experiment = _experiment_with_run(tmp_path)
    button = shell._notes_button
    keys_before = set(QSettings().allKeys())
    run = _show(qapp, shell, pages.RunDetailPage(experiment, "run_001"))
    assert button.isVisible() and button.text() == "Notes"
    assert not button.isChecked() and not run._notes_box.is_open()

    button.click()
    assert run._notes_box.is_open() and run._notes_box.isVisible()
    assert button.isChecked()
    overview = _show(
        qapp, shell, pages.ExperimentOverviewPage(experiment, lambda _k: None)
    )
    box = overview.findChild(NotesBox)
    assert not box.is_open() and not box.isVisible(), "opened on another page"
    assert not button.isChecked(), "the button reads the page it is on"

    button.click()
    assert box.is_open() and button.isChecked()
    again = _show(qapp, shell, pages.RunDetailPage(experiment, "run_001"))
    assert not again._notes_box.is_open(), "a fresh page opens folded"
    assert not button.isChecked()
    assert set(QSettings().allKeys()) == keys_before, "open or folded was remembered"


def test_a_fresh_box_opens_folded_whatever_an_old_build_remembered(qapp, tmp_path):
    """`bands/notes` was the app-wide setting; a value an earlier build left must not open a
    box now."""
    QSettings().setValue("bands/notes", "1")
    experiment = _experiment_with_run(tmp_path)
    assert not NotesBox(experiment, "run", "run_001").is_open()


def test_the_shell_button_hides_on_a_page_with_no_notes(qapp, tmp_path, shell):
    experiment = _experiment_with_run(tmp_path)
    _show(qapp, shell, pages.RunDetailPage(experiment, "run_001"))
    assert shell._notes_button.isVisible()
    shell._show(QLabel("no notes here"), "elsewhere")
    qapp.processEvents()
    assert not shell._notes_button.isVisible()


def test_the_shell_button_binds_a_box_built_after_the_page_was_shown(
    qapp, tmp_path, shell
):
    """A body that loads off the UI thread places its box after `_show` ran."""
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    experiment = _experiment_with_run(tmp_path)
    page = QWidget()
    layout = QVBoxLayout(page)
    _show(qapp, shell, page)
    assert not shell._notes_button.isVisible()
    layout.addWidget(NotesBox(experiment, "run", "run_001"))
    for _ in range(3):
        qapp.processEvents()
    assert shell._notes_button.isVisible()


def test_the_dot_says_a_note_exists_and_goes_when_it_is_cleared(qapp, tmp_path, shell):
    """A folded box still says whether anything is under it: ● on the shell's button after a
    save, gone once the note is emptied, and there when a page opens with a note."""
    experiment = _experiment_with_run(tmp_path)
    page = _show(qapp, shell, pages.RunDetailPage(experiment, "run_001"))
    button, box = shell._notes_button, page._notes_box
    assert not box.has_note() and gui_notes.HAS_NOTE_MARK not in button.text()
    box.editor.setPlainText("worth a second look")
    box.save()
    assert box.has_note()
    assert button.text() == f"Notes {gui_notes.HAS_NOTE_MARK}"
    box.editor.setPlainText("   ")
    box.save()
    assert not box.has_note() and button.text() == "Notes"

    experiment.set_notes("experiment", "", "pilot cohort")
    _show(qapp, shell, pages.ExperimentOverviewPage(experiment, lambda _k: None))
    assert button.text() == f"Notes {gui_notes.HAS_NOTE_MARK}"
    assert not button.isChecked()


def test_an_empty_box_says_what_it_is_for_in_grey(qapp, tmp_path):
    """The placeholder is the only thing in the box saying what it is for, so it is the short
    one the user asked for, drawn in the palette's placeholder colour rather than as text.
    """
    experiment = _experiment_with_run(tmp_path)
    editor = NotesBox(experiment, "run", "run_001").editor
    assert editor.toPlainText() == ""
    assert editor.placeholderText() == "Add notes here (optional)"
    palette = editor.palette()
    assert palette.color(QPalette.ColorRole.PlaceholderText) != palette.color(
        QPalette.ColorRole.Text
    )


def test_typing_starts_the_autosave_and_save_writes_the_file(qapp, tmp_path):
    experiment = _experiment_with_run(tmp_path)
    box = NotesBox(experiment, "run", "run_001")
    box.editor.setPlainText("first thought")
    assert box._timer.isActive()
    assert box.status.text() == "Editing…"
    box._timer.timeout.emit()
    assert _notes_file(experiment, "run", "run_001").read_text(
        encoding="utf-8-sig"
    ) == ("first thought")
    assert box.status.text().startswith("Saved")


def test_hiding_the_page_saves_what_was_typed(qapp, tmp_path):
    """Navigating away hides the page before it is discarded; the note must not wait for a
    timer that will never fire."""
    experiment = _experiment_with_run(tmp_path)
    page = pages.RunDetailPage(experiment, "run_001")
    page.show()
    qapp.processEvents()
    page.findChild(NotesBox).set_open(True)  # nobody types into a folded box
    page.findChild(NotesBox).editor.setPlainText("typed, then clicked away")
    page.hide()
    assert experiment.notes("run", "run_001") == "typed, then clicked away"


def test_a_rebuild_keeps_the_box_and_what_is_in_it(qapp, tmp_path):
    """A run page rebuilds as recordings finish; the box someone is typing in survives it."""
    experiment = _experiment_with_run(tmp_path, status="running")
    page = pages.RunDetailPage(experiment, "run_001")
    box = page.findChild(NotesBox)
    box.editor.setPlainText("mid-sentence")
    page._build()
    assert page.findChildren(NotesBox) == [box]
    assert page._notes_box is box
    assert box.text() == "mid-sentence"


def test_the_fit_page_base_places_and_keeps_the_box(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    directory = datasets_store.dataset_dir(experiment.dir, "dataset_001")
    directory.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        directory, {"id": "dataset_001", "name": "d", "status": "complete"}
    )
    experiment.set_notes("dataset", "dataset_001", "frames from box 1-7")
    page = pages.DatasetDetailPage(experiment, "dataset_001")
    box = page.findChild(NotesBox)
    assert box.text() == "frames from box 1-7"
    page._build()
    assert page.findChildren(NotesBox) == [box]


def test_a_save_after_the_entry_was_deleted_says_so(qapp, tmp_path):
    experiment = _experiment_with_run(tmp_path)
    box = NotesBox(experiment, "run", "run_001")
    runs_store.delete_run(experiment.dir, "run_001")
    box.editor.setPlainText("too late")
    box.save()
    assert "no longer exists" in box.status.text()
    assert not runs_store.run_dir(experiment.dir, "run_001").exists()
