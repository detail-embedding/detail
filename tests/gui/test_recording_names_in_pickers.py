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
