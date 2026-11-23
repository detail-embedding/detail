"""Leaving a page must not throw away what you typed into it.

A page with a Save button loses its edits the moment you click something else, and the
metadata table is where that hurt: fill a column, click the next recording, gone. The
shell asks before leaving (`gui/editing.py`), and the two guards at the bottom are what
stop the *next* editable page from rediscovering the bug.
"""

import re
from pathlib import Path

import pytest

import detail
from detail.api.experiment import Experiment
from detail.gui import pages as gui_pages
from detail.gui.editing import confirm_leave
from detail.gui.qt import QMessageBox, QPushButton, QWidget
from detail.storage import locations


class _Page(QWidget):
    """A minimal editable surface."""

    def __init__(self, summary=None):
        super().__init__()
        self.summary = summary
        self.saved = False
        self.discarded = False
        self.save_error = None

    def unsaved_summary(self):
        return self.summary

    def save_edits(self):
        if self.save_error:
            raise ValueError(self.save_error)
        self.saved = True

    def discard_edits(self):
        self.discarded = True


def _answer(monkeypatch, button):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: button))


def _warned(monkeypatch):
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: seen.append(a))
    )
    return seen


# --------------------------------------------------------------------------- #
# confirm_leave
# --------------------------------------------------------------------------- #
def test_a_clean_page_passes_straight_through(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: pytest.fail("asked about a clean page")),
    )
    assert confirm_leave(None, _Page(summary=None)) is True
    assert confirm_leave(None, QWidget()) is True  # not editable at all
    assert confirm_leave(None, None) is True  # nothing shown yet


def test_save_commits_and_lets_you_leave(qapp, monkeypatch):
    _answer(monkeypatch, QMessageBox.StandardButton.Save)
    page = _Page("2 changed cells")
    assert confirm_leave(None, page) is True
    assert page.saved and not page.discarded


def test_discard_drops_the_edits_and_lets_you_leave(qapp, monkeypatch):
    _answer(monkeypatch, QMessageBox.StandardButton.Discard)
    page = _Page("2 changed cells")
    assert confirm_leave(None, page) is True
    assert page.discarded and not page.saved


def test_cancel_keeps_you_where_you_are(qapp, monkeypatch):
    _answer(monkeypatch, QMessageBox.StandardButton.Cancel)
    page = _Page("2 changed cells")
    assert confirm_leave(None, page) is False
    assert not page.saved and not page.discarded


def test_a_failing_save_stays_put(qapp, monkeypatch):
    """Proceeding would discard the very edits the user asked to keep."""
    _answer(monkeypatch, QMessageBox.StandardButton.Save)
    warned = _warned(monkeypatch)
    page = _Page("a renamed recording")
    page.save_error = "another recording is already named 'zebra'"
    assert confirm_leave(None, page) is False
    assert warned  # and it said why


# --------------------------------------------------------------------------- #
# The adopters report real pending work
# --------------------------------------------------------------------------- #
def test_metadata_page_reports_only_genuine_changes(qapp, tmp_path):
    """Compared against the store, not a dirty flag: entering edit mode and changing
    nothing is not unsaved work, and a cell typed back to its old value isn't either."""
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("m27_day3")
    experiment.add_metadata_column("dose")
    page = gui_pages.MetadataPage(experiment)

    assert page.unsaved_summary() is None  # view mode
    page._enter_edit()
    assert page.unsaved_summary() is None  # editing, nothing typed

    page._table.item(0, 2).setText("0.5")  # col 2: recording_id, file, then dose
    assert "1 changed cell" in page.unsaved_summary()
    page._table.item(0, 2).setText("")  # back to what the store holds
    assert page.unsaved_summary() is None

    page._table.item(0, 2).setText("0.5")
    page.save_edits()
    assert experiment.recording_metadata["recording_000"]["dose"] == "0.5"
    assert page.unsaved_summary() is None


def test_metadata_page_discard_restores_the_stored_values(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("m27_day3")
    experiment.add_metadata_column("dose")
    page = gui_pages.MetadataPage(experiment)
    page._enter_edit()
    page._table.item(0, 2).setText("999")
    page.discard_edits()
    assert page.unsaved_summary() is None
    assert experiment.recording_metadata["recording_000"].get("dose", "") == ""


def test_recording_page_reports_a_pending_rename(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27_day3")
    page = gui_pages.RecordingDetailPage(recording, experiment=experiment)

    assert page.unsaved_summary() is None
    page._name.setText("zebra")
    assert "the name" in page.unsaved_summary()
    page.save_edits()
    assert experiment.get_recording(recording.id).name == "zebra"
    assert page.unsaved_summary() is None


def test_experiment_overview_reports_a_pending_description(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    page = gui_pages.ExperimentOverviewPage(experiment, on_open_section=lambda _k: None)
    assert page.unsaved_summary() is None
    page._desc.setText("a new description")
    assert page.unsaved_summary() is not None
    page.save_edits()
    assert Experiment.open(experiment.dir).description == "a new description"


# --------------------------------------------------------------------------- #
# Guard 1: every navigation entry point asks
# --------------------------------------------------------------------------- #
_APP = Path(detail.__file__).parent / "gui" / "app.py"

# Methods that put a different page on screen. Each must ask before it does — the check
# has to be *early*, because by the time `_show` runs the next page is built and the rail
# selection has moved, so Cancel could no longer leave you where you were.
_ENTRY_POINTS = (
    "_on_tree_click",
    "_switch_mode",
    "_go_to_node",
    "_go_to_section",
    "_go_to_tool",
    "_open_experiment",
    "closeEvent",
)


def _method_body(source: str, name: str) -> str:
    start = source.index(f"    def {name}(")
    rest = source[start + 1 :]
    end = rest.find("\n    def ")
    return rest if end < 0 else rest[:end]


def test_every_navigation_entry_point_asks_before_leaving():
    source = _APP.read_text(encoding="utf-8-sig")
    missing = [
        name
        for name in _ENTRY_POINTS
        if "_may_leave()" not in _method_body(source, name)
    ]
    assert not missing, (
        "these navigate away without asking about unsaved edits, so a filled-in table is "
        f"lost by clicking elsewhere: {missing}. Call self._may_leave() first."
    )


def test_the_entry_point_list_is_not_stale():
    source = _APP.read_text(encoding="utf-8-sig")
    absent = [name for name in _ENTRY_POINTS if f"    def {name}(" not in source]
    assert not absent, f"listed entry points that no longer exist: {absent}"


# --------------------------------------------------------------------------- #
# Guard 2: a page with a Save control implements the protocol
# --------------------------------------------------------------------------- #
_PAGES = Path(detail.__file__).parent / "gui" / "pages.py"

# Content pages that own a Save-like control but hold nothing a user typed — the token
# page writes a credential straight through, with nothing staged to lose.
ALLOWED_WITHOUT_PROTOCOL = {"HuggingFacePage"}


def _classes_with_a_save_button(source: str) -> set[str]:
    found = set()
    current = None
    for line in source.splitlines():
        match = re.match(r"class (\w+)\(", line)
        if match:
            current = match.group(1)
        if current and re.search(r'QPushButton\("Save[^"]*"\)', line):
            found.add(current)
    return found


def test_a_page_with_a_save_button_can_be_asked_about_unsaved_edits():
    source = _PAGES.read_text(encoding="utf-8-sig")
    offenders = sorted(
        name
        for name in _classes_with_a_save_button(source) - ALLOWED_WITHOUT_PROTOCOL
        if f"class {name}(" in source
        and "def unsaved_summary" not in _class_body(source, name)
    )
    assert not offenders, (
        "these hold edits behind a Save button but the shell cannot ask about them, so "
        f"navigating away drops the edits silently: {offenders}. Implement "
        "unsaved_summary / save_edits / discard_edits (detail/gui/editing.py)."
    )


def _class_body(source: str, name: str) -> str:
    start = source.index(f"class {name}(")
    rest = source[start:]
    end = rest.find("\nclass ")
    return rest if end < 0 else rest[:end]


def test_the_save_button_scan_would_catch_one():
    """Guard the guard: a scan that matches nothing passes vacuously forever."""
    sample = 'class Thing(QWidget):\n    save = QPushButton("Save name")\n'
    assert _classes_with_a_save_button(sample) == {"Thing"}


# --------------------------------------------------------------------------- #
# Guard 3: a page whose data was deleted must not brick the shell
# --------------------------------------------------------------------------- #
class _RaisingPage(QWidget):
    """A page whose own data went away — its pending-check raises, like the overview's
    ``self._experiment.name`` does once the experiment directory is gone."""

    def __init__(self):
        super().__init__()
        self.asked = False

    def unsaved_summary(self):
        self.asked = True
        raise FileNotFoundError("no config.yaml in /gone")

    def save_edits(self):
        raise AssertionError("must not be asked to save a page whose data is gone")

    def discard_edits(self):
        raise AssertionError("must not be asked to discard a page whose data is gone")


def test_confirm_leave_fails_open_when_the_pending_check_raises(qapp, monkeypatch):
    """Leaving must win over a page that can no longer answer.

    Every navigation entry point calls this first, so an exception escaping here leaves
    the dead page on screen and makes the window permanently unnavigable.
    """
    prompted = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: prompted.append(a) or QMessageBox.StandardButton.Cancel
        ),
    )
    page = _RaisingPage()
    assert confirm_leave(None, page) is True
    assert page.asked  # it really did ask — the fail-open is in confirm_leave
    assert not prompted  # and it did not prompt about edits that cannot exist


def test_deleting_an_experiment_leaves_a_usable_window(qapp, tmp_path, monkeypatch):
    """The reported Windows bug, driven through the real click path.

    Delete used to raise out of ``_on_experiment_deleted`` → ``_switch_mode`` →
    ``_may_leave`` → the dead overview's ``unsaved_summary``, which left the stale page as
    ``_content.widget(0)`` forever — so *every* later navigation raised too. Asserting only
    that the delete returns would miss that second half, so this asserts where you land.
    """
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    from detail.gui.app import MainWindow

    experiment = Experiment.create("Test2", tmp_path)
    window = MainWindow()
    window._open_experiment(experiment)
    page = window._content.widget(0)
    assert isinstance(page, gui_pages.ExperimentOverviewPage)

    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    delete = next(
        b for b in page.findChildren(QPushButton) if b.text() == "Delete experiment"
    )
    delete.click()  # the real button, not the private handler

    assert not experiment.dir.exists()  # the delete itself still happens
    assert window._experiment is None
    landed = window._content.widget(0)
    assert isinstance(landed, gui_pages.HomePage)  # we went somewhere, not nowhere
    assert not isinstance(landed, gui_pages.ExperimentOverviewPage)
    assert not window._tree.isVisible()  # the rail dropped the deleted experiment

    # ...and the window still navigates afterwards — the "all tabs non-responsive" half.
    window._switch_mode("config")
    assert window._mode == "config"
