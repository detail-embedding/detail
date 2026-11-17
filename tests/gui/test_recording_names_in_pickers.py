"""A recording picker shows the name; the selection it returns is still ids.

§11: "A recording's id is the tool's, its name is the user's … never display one where a
name will do." PR #174 fixed this for *model* ids in run labels. The sweep found the same
shape on recordings across nine more surfaces, including five list pickers that used the id
as the row's **display text** and read it back with `item.text()` — so showing a name there
would have silently broken selection unless the id moved to `UserRole` at the same time.

The Recordings table shows only names, so a list of `recording_000…` is unreadable without
opening each recording's own page — which is exactly the cross-referencing §11 exists to
prevent, and it bites hardest when picking fit vs held-out recordings for a clusterer.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QListWidget  # noqa: E402

import detail  # noqa: E402
from detail.api.experiment import Experiment  # noqa: E402
from detail.gui.recording_items import (  # noqa: E402
    fill_recording_list,
    selected_recording_ids,
)


@pytest.fixture
def named_experiment(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    for rid, name in (
        ("recording_001", "Fish tank A"),
        ("recording_002", "Fish tank B"),
    ):
        experiment.create_recording(rid).set_name(name)
    return experiment


# --------------------------------------------------------------------------- #
# the helper: name shown, id returned
# --------------------------------------------------------------------------- #
def test_a_picker_shows_names_and_returns_ids(qapp, named_experiment):
    widget = QListWidget()
    fill_recording_list(widget, named_experiment.recordings)

    shown = [widget.item(i).text() for i in range(widget.count())]
    assert shown == ["Fish tank A", "Fish tank B"]

    widget.item(1).setSelected(True)
    assert selected_recording_ids(widget) == ["recording_002"]  # ids, not labels


def test_refilling_does_not_duplicate_rows(qapp, named_experiment):
    widget = QListWidget()
    fill_recording_list(widget, named_experiment.recordings)
    fill_recording_list(widget, named_experiment.recordings)
    assert widget.count() == 2


# --------------------------------------------------------------------------- #
# the guard: no picker may go back to labelling with an id
# --------------------------------------------------------------------------- #
_GUI = Path(detail.__file__).parent / "gui"
_BAD_SNIPPETS = (
    "QListWidgetItem(recording.id",
    "addItem(recording.id",
    "addItem(recording_id, recording_id)",
)


def test_no_picker_labels_a_row_with_a_recording_id():
    offenders = []
    for path in sorted(_GUI.rglob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        for snippet in _BAD_SNIPPETS:
            if snippet in text:
                offenders.append(f"{path.relative_to(_GUI)}: {snippet}")
    assert not offenders, (
        "a picker must show the recording's name and keep the id as the row's data (§11); "
        f"these label a row with the id: {offenders}. Use gui/recording_items.py."
    )


def test_the_scan_would_catch_one():
    """Guard the guard — a scan that matches nothing passes vacuously forever."""
    sample = "            QListWidgetItem(recording.id, self._recordings)"
    assert any(snippet in sample for snippet in _BAD_SNIPPETS)


# --------------------------------------------------------------------------- #
# messages and path lines, not only pickers (#35)
# --------------------------------------------------------------------------- #
def _capture_question(monkeypatch, answer):
    """Stand in for `QMessageBox.question` (a modal that would block the test), keeping the
    text it was asked to show."""
    from PyQt6.QtWidgets import QMessageBox

    asked: list[str] = []

    def question(_parent, _title, text, *args, **kwargs):
        asked.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


def test_the_delete_confirmation_names_the_recording(
    qapp, named_experiment, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.pages import RecordingDetailPage

    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.No)
    recording = named_experiment.get_recording("recording_001")
    page = RecordingDetailPage(recording, experiment=named_experiment)
    page._delete_recording()
    assert asked and "'Fish tank A'" in asked[0], asked
    assert "recording_001" not in asked[0], asked
    assert named_experiment.get_recording("recording_001")  # "No" deleted nothing


def test_the_launch_skip_list_names_the_recordings(qapp, named_experiment, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.launch import _covered_subset

    class _CoversOnlyTheFirst:
        def covered_recording_ids(self):
            return {"recording_001"}

    asked = _capture_question(monkeypatch, QMessageBox.StandardButton.Yes)
    names = named_experiment.recording_names()
    kept = _covered_subset(
        _CoversOnlyTheFirst(),
        ["recording_001", "recording_002"],
        None,
        "Launch run",
        names=names,
    )
    assert kept == ["recording_001"]  # still ids: that is what the launch takes
    assert "will be skipped:\nFish tank B\n" in asked[0], asked
    assert "recording_002" not in asked[0], asked


def test_the_path_line_names_the_recording(qapp, named_experiment, monkeypatch):
    """The shell's path line over a recording page, and over one of its runs. Driven through
    the shell's own openers with a stand-in for the window around them: a whole `MainWindow`
    holding a recording page aborts at teardown in this harness (on `main` too), which is a
    separate matter from what is shown."""
    import types

    from detail.gui.app import MainWindow

    shown: list[str] = []
    shell = types.SimpleNamespace(
        _experiment=named_experiment,
        _show=lambda widget, context: (shown.append(context), widget.deleteLater()),
        _set_page_export=lambda *a: None,
        _open_run=lambda *a: None,
        _on_recording_deleted=lambda *a: None,
        _open_recording_run=lambda *a: None,
    )
    MainWindow._open_recording(shell, "recording_001")
    # A run page needs a run on disk; what is checked here is the line above it.
    from PyQt6.QtWidgets import QWidget

    from detail.gui import pages

    monkeypatch.setattr(pages, "RecordingRunDetailPage", lambda *a, **k: QWidget())
    MainWindow._open_recording_run(shell, "recording_001", "detection", "det_001")
    assert shown == [
        f"{named_experiment.name}  ›  Fish tank A",
        f"{named_experiment.name}  ›  Fish tank A  ›  det_001",
    ]
