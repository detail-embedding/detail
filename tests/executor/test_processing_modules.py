"""Tests for the ProcessingModule contract + registry (§6.5)."""

from dataclasses import dataclass

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import Experiment
from detail.core.dataclasses import _JOB_TAGS, _JOB_TYPES
from detail.executor.runners import (
    JOB_RUNNERS,
    PROCESSING_MODULES,
    ProcessingModule,
    import_runner_modules,
    register_processing_module,
)


@dataclass
class _DummyJob:
    job_id: str
    recording_id: str

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "recording_id": self.recording_id}

    @classmethod
    def from_dict(cls, d: dict) -> "_DummyJob":
        return cls(**d)


def _dummy_module_cls(module_id: str = "_dummy_test") -> type:
    class _DummyModule(ProcessingModule):
        id_prefix = "dum"
        runs_key = "_dummy_runs"
        upstream = "video"
        job_class = _DummyJob
        progress_event_class = None

        def validate(self, recording_dir, recording_data, config):
            return config

        def build_job(self, recording_id, run_id, config):
            return _DummyJob(job_id=run_id, recording_id=recording_id)

        def runner(self, job, run_dir, progress_cb, stop_event):
            pass

        def run_dir_for(self, experiment_root, job):
            return experiment_root

        def output_path(self, run_id):
            return f"_dummy/{run_id}/"

        def estimate_output_bytes(self, recording_dir, recording_data, config):
            return 0

    _DummyModule.module_id = module_id
    return _DummyModule


def test_first_party_modules_registered():
    import_runner_modules(PIPELINE_RUNNERS)
    expected = {
        "detection": ("det", "detection_runs", "video"),
        "tracking": ("track", "tracking_runs", "detection"),
        "embedding": ("emb", "embedding_runs", "tracking"),
    }
    for module_id, (prefix, runs_key, upstream) in expected.items():
        module = PROCESSING_MODULES[module_id]
        assert module.id_prefix == prefix
        assert module.runs_key == runs_key
        assert module.upstream == upstream
        assert module_id in JOB_RUNNERS
        assert _JOB_TYPES[module_id] is module.job_class


def test_register_processing_module_wires_all_seams():
    cls = _dummy_module_cls()
    try:
        register_processing_module(cls)
        assert isinstance(PROCESSING_MODULES["_dummy_test"], cls)
        assert "_dummy_test" in JOB_RUNNERS
        assert _JOB_TYPES["_dummy_test"] is _DummyJob
    finally:
        PROCESSING_MODULES.pop("_dummy_test", None)
        JOB_RUNNERS.pop("_dummy_test", None)
        _JOB_TYPES.pop("_dummy_test", None)
        _JOB_TAGS.pop(_DummyJob, None)


def test_register_processing_raises_on_duplicate_id():
    register_processing_module(_dummy_module_cls())
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_processing_module(_dummy_module_cls())  # same module_id
    finally:
        PROCESSING_MODULES.pop("_dummy_test", None)
        JOB_RUNNERS.pop("_dummy_test", None)
        _JOB_TYPES.pop("_dummy_test", None)
        _JOB_TAGS.pop(_DummyJob, None)


def test_register_processing_raises_on_id_taken_by_an_analysis_module():
    import_runner_modules(PIPELINE_RUNNERS)  # registers detection/... + location_export
    with pytest.raises(ValueError, match="already registered"):
        register_processing_module(_dummy_module_cls("location_export"))  # cross-family


def test_base_run_record_has_standard_keys():
    module = _dummy_module_cls()()
    record = module.run_record("dum_01", {}, "pending")
    assert record["id"] == "dum_01"
    assert record["status"] == "pending"
    assert record["output"] == "_dummy/dum_01/"
    assert record.get("created_at")  # dated for run labels
    assert "processing_run_id" not in record  # only set when batch-launched


def test_run_record_carries_processing_run_id_when_batch_launched():
    module = _dummy_module_cls()()
    record = module.run_record("dum_01", {"processing_run_id": "run_007"}, "pending")
    assert record["processing_run_id"] == "run_007"


def test_run_rejects_unknown_module_id(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    recording = proj.create_recording("recording_001")
    with pytest.raises(ValueError, match="unknown module_id"):
        recording.run("nope", {})
