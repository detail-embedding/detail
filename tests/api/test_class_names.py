"""Class names through the API (§1 *Class names*): the roster read off records, a rename
persisted in ``config.yaml``, and a class named by its name reaching a run as its stored name.

The run seam is the one that matters: a run's record must hold the stored name whatever the
user typed, or two runs of the same settings get two identities.
"""

import json

import pytest
import yaml

from detail.api.experiment import Experiment
from detail.api.run_spec import _processing_module
from detail.storage import models as models_store
from detail.storage import recording as recording_store
from detail.storage import runs as runs_store
from detail.storage.zarr_io import DetectionPartWriter


@pytest.fixture
def experiment(tmp_path):
    """One recording with a finished two-class detection run, and the experiment-level run
    record that lists the classes it was prompted for."""
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
    recording_store.write_recording(recording.dir, data)
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "config": {
                "model_id": "sam3",
                "prompt": {
                    "targets": [
                        {"class_name": "black mouse", "cue_type": "text", "cue": "b"},
                        {"class_name": "white mouse", "cue_type": "text", "cue": "w"},
                    ]
                },
            },
            "per_recording": {},
            "status": "complete",
        },
    )
    return experiment


def _config(experiment) -> dict:
    return yaml.safe_load(
        (experiment.dir / "config.yaml").read_text(encoding="utf-8-sig")
    )


def test_the_roster_is_read_off_run_records_and_trained_models(experiment):
    mdir = models_store.model_dir(experiment.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "name": "m", "classes": [{"id": 0, "name": "plate"}]},
    )
    assert experiment.classes() == ["black mouse", "plate", "white mouse"]


def test_a_rename_is_persisted_after_schema_version_and_shown_everywhere(experiment):
    experiment.rename_class("white mouse", " Resident ")
    config = _config(experiment)
    assert next(iter(config)) == "schema_version"
    assert config["class_names"] == {"white mouse": "Resident"}
    assert experiment.class_names() == {
        "black mouse": "black mouse",
        "white mouse": "Resident",
    }
    assert experiment.class_label("white mouse") == "Resident"
    assert experiment.renamed_classes() == {"white mouse": "Resident"}


def test_renaming_to_nothing_or_the_stored_name_removes_the_entry(experiment):
    experiment.rename_class("white mouse", "Resident")
    experiment.rename_class("white mouse", None)
    assert "class_names" not in _config(experiment)
    experiment.rename_class("white mouse", "Resident")
    experiment.rename_class("white mouse", "white mouse")
    assert "class_names" not in _config(experiment)


def test_two_classes_can_trade_names_in_one_rename(experiment):
    experiment.rename_classes({"black mouse": "A", "white mouse": "B"})
    experiment.rename_classes({"black mouse": "B", "white mouse": "A"})
    assert experiment.renamed_classes() == {"black mouse": "B", "white mouse": "A"}


def test_a_refused_rename_leaves_the_file_as_it_was(experiment):
    before = (experiment.dir / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match="'black mouse' is already a class"):
        experiment.rename_class("white mouse", "black mouse")
    with pytest.raises(KeyError, match="no class 'rat'"):
        experiment.rename_class("rat", "Rat")
    assert (experiment.dir / "config.yaml").read_bytes() == before


def test_resolve_class_takes_either_name(experiment):
    experiment.rename_class("white mouse", "Resident")
    assert experiment.resolve_class("Resident") == "white mouse"
    assert experiment.resolve_class("white mouse") == "white mouse"
    with pytest.raises(KeyError, match="no class 'rat'.*Resident"):
        experiment.resolve_class("rat")


def test_a_run_given_the_name_validates_and_records_the_stored_name(experiment):
    experiment.rename_class("white mouse", "Resident")
    module = _processing_module("tracking")
    config, validated = experiment._validate_processing(
        module,
        ["recording_001"],
        {
            "detection_run_id": "det_01",
            "tracker_id": "iou_single_tracker",
            "class_name": "Resident",
        },
    )
    assert config["class_name"] == "white mouse"
    assert validated["recording_001"]["class_name"] == "white mouse"
