"""``Experiment.tune_tracking`` and finding a tuning again (§12 *Tune detection and tracking
settings*): what launching refuses, where a tuning lives, and how the detection and tracking
forms find the newest one from either source."""

import json
import types

import pytest
from pydantic import ValidationError

from detail.api import tuning as tuning_api
from detail.api.experiment import Experiment
from detail.storage import models as models_store
from detail.storage import recording as recording_store
from detail.storage import tunings as tunings_store


def _model(experiment, model_id="model_001", **over):
    record = {
        "id": model_id,
        "name": "student",
        "status": "complete",
        "task": "segment",
        "trainer_id": "ultralytics_yolo",
        **over,
    }
    directory = models_store.model_dir(experiment.dir, model_id)
    directory.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(directory, record)
    return model_id


class _Detector:
    supported_backends = ["cpu"]

    def readiness(self):
        return types.SimpleNamespace(ready=True, message="ok")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path / "exp")
    proj.add_recording("no video")
    monkeypatch.setattr(
        "detail.models.resolve.resolve_detection_model", lambda *a, **k: _Detector()
    )
    return proj


def _tuning(experiment, model_id, tuning_id, *, status="complete", created, **over):
    directory = tunings_store.tuning_dir(experiment.dir, model_id, tuning_id)
    record = {
        "id": tuning_id,
        "model_id": model_id,
        "status": status,
        "created_at": created,
        "finished_at": created,
        "source": {"kind": "standalone"},
        "pick": {
            "setting": {
                "detection_conf": 0.05,
                "conf_threshold": 0.1,
                "secondary_conf": 0.05,
                "max_secondary": 4,
            },
            "keep_defaults": False,
            "reason": "lowest mean jumpy share",
        },
        "warnings": [],
        **over,
    }
    tunings_store.write_tuning(directory, record)
    (directory / "job.json").write_text(
        json.dumps({"schema_version": 1, "status": status, "job": {}, "error": None}),
        encoding="utf-8",
    )
    return record


# --------------------------------------------------------------------- launching


def test_a_keypoint_model_is_refused(experiment):
    model_id = _model(experiment, task="pose")
    with pytest.raises(ValueError, match="places keypoints"):
        experiment.tune_tracking(model_id)


def test_an_unfinished_model_is_refused(experiment):
    model_id = _model(experiment, status="running")
    with pytest.raises(ValueError, match="finished training"):
        experiment.tune_tracking(model_id)


def test_an_unknown_model_is_refused(experiment):
    with pytest.raises(FileNotFoundError):
        experiment.tune_tracking("model_404")


@pytest.mark.parametrize("seconds", [5.0, 4000.0])
def test_the_sample_size_is_bounded(experiment, seconds):
    model_id = _model(experiment)
    with pytest.raises(ValidationError, match="sample_seconds"):
        experiment.tune_tracking(model_id, sample_seconds=seconds)


def test_a_named_recording_without_video_is_refused_by_name(experiment):
    model_id = _model(experiment)
    recording = experiment.recordings[0]
    with pytest.raises(ValueError, match="'no video' has no video"):
        experiment.tune_tracking(model_id, [recording.id])


def test_no_recording_with_video_is_refused(experiment):
    with pytest.raises(ValueError, match="at least one recording with video"):
        experiment.tune_tracking(_model(experiment))


def test_a_model_already_being_tuned_is_refused(experiment, monkeypatch):
    model_id = _model(experiment)
    monkeypatch.setattr(
        experiment, "_sampling_recordings", lambda ids: ["recording_001"]
    )
    monkeypatch.setattr(
        experiment,
        "get_recording",
        lambda rid: types.SimpleNamespace(dir=None, name="a"),
    )
    monkeypatch.setattr(
        recording_store, "read_recording", lambda d: {"video": {"files": [{}]}}
    )
    _tuning(experiment, model_id, "tune_001", status="running", created="2026-11-19")
    with pytest.raises(ValueError, match="already being tuned"):
        experiment.tune_tracking(model_id)


def test_launching_submits_a_job_in_the_models_tuning_folder(experiment, monkeypatch):
    model_id = _model(experiment)
    monkeypatch.setattr(
        experiment, "_sampling_recordings", lambda ids: ["recording_001"]
    )
    monkeypatch.setattr(
        experiment,
        "get_recording",
        lambda rid: types.SimpleNamespace(dir=None, name="a"),
    )
    monkeypatch.setattr(
        recording_store, "read_recording", lambda d: {"video": {"files": [{}]}}
    )
    submitted = []
    monkeypatch.setattr(
        "detail.executor.local.LocalExecutor.submit",
        lambda self, job: submitted.append(job) or job.job_id,
    )
    run = experiment.tune_tracking(model_id, sample_seconds=120)
    (job,) = submitted
    assert run.id == job.job_id == "tune_001"
    assert job.sample_seconds == 120.0
    assert job.source == {"kind": "standalone"}
    assert job.compute.backend == "cpu"
    assert tunings_store.tuning_dir(experiment.dir, model_id, "tune_001").is_dir()


# --------------------------------------------------------------------- finding it


def test_the_newest_complete_tuning_is_offered_and_says_which(experiment):
    model_id = _model(experiment)
    _tuning(
        experiment,
        model_id,
        "tune_001",
        created="2026-11-18T10:00:00Z",
        source={"kind": "distill", "distill_id": "distill_003"},
    )
    _tuning(experiment, model_id, "tune_002", created="2026-11-19T10:00:00Z")
    _tuning(experiment, model_id, "tune_003", status="failed", created="2026-11-20")
    found = experiment.tuned_settings(model_id)
    assert found.tuning_id == "tune_002"
    assert found.n_complete == 2
    assert found.describe() == (
        "tune_002 (from a standalone tune, 2026-11-19T10:00:00Z), the newest of 2"
    )
    assert found.flags() == (
        "--conf 0.05",
        "--conf-threshold 0.1 --secondary-conf 0.05 --max-secondary 4",
    )
    assert [r["id"] for r in experiment.tunings(model_id)] == [
        "tune_003",
        "tune_002",
        "tune_001",
    ]


def test_a_distill_tuning_names_its_distill(experiment):
    model_id = _model(experiment)
    _tuning(
        experiment,
        model_id,
        "tune_001",
        created="2026-11-18",
        source={"kind": "distill", "distill_id": "distill_003"},
    )
    found = experiment.tuned_settings(model_id)
    assert found.describe() == "tune_001 (from distill distill_003, 2026-11-18)"


def test_an_untuned_model_offers_nothing(experiment):
    assert experiment.tuned_settings(_model(experiment)) is None


def test_a_worker_that_died_is_folded_in_from_its_job(experiment):
    model_id = _model(experiment)
    _tuning(experiment, model_id, "tune_001", status="running", created="2026-11-19")
    directory = tunings_store.tuning_dir(experiment.dir, model_id, "tune_001")
    (directory / "job.json").write_text(
        json.dumps({"schema_version": 1, "status": "failed", "error": "killed"}),
        encoding="utf-8",
    )
    (record,) = experiment.tunings(model_id)
    assert record["status"] == "failed"
    assert record["error"] == "killed"


def _detection(conf, weights_source="trained"):
    return {
        "id": "det_01",
        "model_id": "model_001",
        "weights_source": weights_source,
        "conf_threshold": conf,
    }


def test_a_tracking_form_learns_when_its_detection_kept_too_little(experiment):
    model_id = _model(experiment)
    _tuning(experiment, model_id, "tune_001", created="2026-11-19")
    found = experiment.tuned_settings_for_detection(_detection(0.2))
    assert found.detection_conf_used == 0.2
    assert found.needs_new_detection
    assert not experiment.tuned_settings_for_detection(
        _detection(0.05)
    ).needs_new_detection


def test_a_detection_with_a_model_this_experiment_did_not_train_offers_nothing(
    experiment,
):
    model_id = _model(experiment)
    _tuning(experiment, model_id, "tune_001", created="2026-11-19")
    assert (
        experiment.tuned_settings_for_detection(_detection(0.2, "pretrained")) is None
    )
    assert experiment.tuned_settings_for_detection({"conf_threshold": 0.2}) is None


def test_source_label_reads_both_kinds():
    assert tuning_api.source_label({"source": {"kind": "standalone"}}) == (
        "a standalone tune"
    )
    assert tuning_api.source_label(
        {"source": {"kind": "distill", "distill_id": "distill_009"}}
    ) == ("distill distill_009")
