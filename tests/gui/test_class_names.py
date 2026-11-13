"""Class names in the GUI (§1 *Class names*): the dialog that sets them, reached from the
Recordings page, and the class pickers that show them while keeping the stored name as data.

The pickers are the part that goes quietly wrong: a picker showing the name but handing the
name on as its value would record whatever the class was called that day, and one handing on
the stored name but showing it would be the screen the user asked to change. So each test
reads both the text and the data.
"""

import json

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QPushButton  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import pages as gui_pages  # noqa: E402
from detail.gui.class_example_sets import ClassExampleSets  # noqa: E402
from detail.gui.class_names_dialog import ClassNamesDialog  # noqa: E402
from detail.gui.modules._common import arena_items  # noqa: E402
from detail.gui.modules._common import detected_class_groups  # noqa: E402
from detail.gui.modules._common import spec_groups  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402
from detail.storage import runs as runs_store  # noqa: E402
from detail.storage.zarr_io import DetectionPartWriter  # noqa: E402


@pytest.fixture
def experiment(tmp_path):
    """A two-class detection run on one recording, its experiment record, and one tracking
    run filtered to the white mouse — with the white mouse renamed Resident."""
    experiment = Experiment.create("cn", tmp_path)
    recording = experiment.create_recording("recording_001")
    det = recording.dir / "detections" / "det_01"
    det.mkdir(parents=True)
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    with DetectionPartWriter(det / "part_001.zarr") as writer:
        writer.add_row(0, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        writer.add_row(0, 0, 1, "white mouse", 0.8, (0.6, 0.6, 0.7, 0.7), "")
    data = recording_store.read_recording(recording.dir)
    data.setdefault("detection_runs", []).append(
        {
            "id": "det_01",
            "status": "complete",
            "output": "detections/det_01/",
            "model_id": "sam3",
        }
    )
    data.setdefault("tracking_runs", []).append(
        {
            "id": "track_01",
            "status": "complete",
            "output": "tracking/track_01/",
            "detection_run_id": "det_01",
            "tracker_id": "anchor_tracker",
            "class_name": "white mouse",
        }
    )
    recording_store.write_recording(recording.dir, data)
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "config": {
                "model_id": "m",
                "class_conf_thresholds": {"black mouse": 0.5, "white mouse": 0.5},
            },
            "status": "complete",
        },
    )
    experiment.rename_class("white mouse", "Resident")
    return experiment


def test_the_dialog_saves_every_row_together(qapp, experiment):
    dialog = ClassNamesDialog(experiment, experiment.class_names())
    assert dialog._fields["white mouse"].text() == "Resident"
    assert dialog._fields["black mouse"].text() == ""
    assert dialog._fields["black mouse"].placeholderText() == "black mouse"
    dialog._fields["black mouse"].setText("Resident")
    dialog._fields["white mouse"].setText("Intruder")
    dialog._save()
    assert experiment.renamed_classes() == {
        "black mouse": "Resident",
        "white mouse": "Intruder",
    }


def test_a_refused_save_keeps_the_dialog_open_and_the_file_unchanged(
    qapp, experiment, monkeypatch
):
    warned = []
    monkeypatch.setattr(
        "detail.gui.class_names_dialog.QMessageBox.warning",
        lambda *args: warned.append(args[2]),
    )
    dialog = ClassNamesDialog(experiment, experiment.class_names())
    dialog._fields["white mouse"].setText("black mouse")
    dialog._save()
    assert warned and "'black mouse' is already a class" in warned[0]
    assert experiment.renamed_classes() == {"white mouse": "Resident"}
    assert dialog.result() == 0


def test_the_recordings_page_offers_class_names(qapp, experiment):
    page = gui_pages.ExperimentRecordingsPage(experiment, lambda _rid: None)
    buttons = [b.text() for b in page.findChildren(QPushButton)]
    assert "Class names…" in buttons


def test_class_groups_show_the_name_and_hand_on_the_stored_one(experiment):
    recordings = experiment.recordings
    [detection] = spec_groups(recordings, "detection")
    groups = detected_class_groups(recordings, detection["spec"])
    items = {data: text for text, data in arena_items(groups, 1)}
    assert items["white mouse"].startswith("Resident")
    assert items["black mouse"].startswith("black mouse")


def test_a_tracking_run_label_names_the_class_as_the_experiment_does(experiment):
    labels = [g["label"] for g in spec_groups(experiment.recordings, "tracking")]
    assert any("Resident" in label for label in labels), labels
    assert not any("white mouse" in label for label in labels), labels


def test_per_class_rows_are_headed_by_the_name_and_keyed_by_the_stored_one(qapp):
    sets = ClassExampleSets()
    sets.set_classes(["black mouse", "white mouse"], labels={"white mouse": "Resident"})
    ticks = {name: tick.text() for name, (tick, _picker) in sets._rows.items()}
    assert ticks == {"black mouse": "black mouse", "white mouse": "Resident"}
    assert sets.classes() == ["black mouse", "white mouse"]
    assert "Resident" in sets.problem()
