"""Tracking resolves ``exclude_classes`` from the detection model and writes it down (§13).

**The class.** The tether option (§12) records ``cable`` as the model's auxiliary class; that
fact is useless unless the run that tracks the model's detections reads it. Omitted, the
setting resolves to the model's ``auxiliary_classes`` in ``validate()`` and is written back
(Tenet 9), so the run record says which classes it left out. Given, a name the detection run
does not have is refused, not silently matched against nothing.
"""

import pytest

from detail.pipeline.tracking import TrackingModule
from detail.storage import models as models_store


def _detection(model_id="model_001", weights_source="trained", prompt_classes=()):
    return {
        "detection_runs": [
            {
                "id": "det_01",
                "status": "complete",
                "output": "detections/det_01/",
                "model_id": model_id,
                "weights_source": weights_source,
                "prompt_classes": list(prompt_classes),
            }
        ]
    }


@pytest.fixture
def tethered(recording_dir, experiment_dir):
    """A complete detection run from a model whose ``model.yaml`` names ``cable`` auxiliary."""
    assert recording_dir.parent.parent == experiment_dir  # the root validate() derives
    (recording_dir / "detections" / "det_01" / "part_000.zarr").mkdir(parents=True)
    mdir = models_store.model_dir(experiment_dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "status": "complete",
            "classes": [{"id": 0, "name": "mouse"}, {"id": 1, "name": "cable"}],
            "auxiliary_classes": ["cable"],
        },
    )
    return recording_dir


def _validate(recording_dir, recording_data, **config):
    return TrackingModule().validate(
        recording_dir, recording_data, {"detection_run_id": "det_01", **config}
    )


def test_omitted_it_resolves_to_the_models_auxiliary_classes(tethered):
    config = _validate(tethered, _detection())
    assert config["exclude_classes"] == ["cable"]
    assert TrackingModule().extra_record_fields(config)["exclude_classes"] == ["cable"]
    job = TrackingModule().build_job("recording_001", "track_01", config)
    assert job.exclude_classes == ["cable"]


def test_an_empty_list_tracks_every_class(tethered):
    assert (
        _validate(tethered, _detection(), exclude_classes=[])["exclude_classes"] == []
    )


def test_a_built_in_model_leaves_out_nothing_by_default(tethered):
    data = _detection("sam3", "pretrained", ["mouse"])
    assert _validate(tethered, data)["exclude_classes"] == []


def test_a_class_the_detection_run_does_not_have_is_refused(tethered):
    with pytest.raises(ValueError, match="has no class named 'pellet'"):
        _validate(tethered, _detection(), exclude_classes=["pellet"])


def test_leaving_out_every_class_is_refused(tethered):
    with pytest.raises(ValueError, match="leaves nothing to track"):
        _validate(tethered, _detection(), exclude_classes=["mouse", "cable"])


def test_a_prompted_runs_classes_are_checked_too(tethered):
    data = _detection("sam3", "pretrained", ["mouse", "cable"])
    assert _validate(tethered, data, exclude_classes=["cable"])["exclude_classes"] == [
        "cable"
    ]
    with pytest.raises(ValueError, match="has no class named 'tail'"):
        _validate(tethered, data, exclude_classes=["tail"])


def test_a_run_recorded_before_the_field_reads_as_leaving_out_nothing():
    assert TrackingModule.identity_defaults["exclude_classes"] == []
    assert "exclude_classes" in TrackingModule.identity_fields
