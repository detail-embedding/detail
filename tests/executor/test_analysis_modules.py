"""Tests for the AnalysisModule contract + registry (§16)."""

from dataclasses import dataclass

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.core.dataclasses import _JOB_TAGS, _JOB_TYPES
from detail.executor.runners import (
    ANALYSIS_MODULES,
    JOB_RUNNERS,
    AnalysisModule,
    import_runner_modules,
    register_analysis_module,
)


@dataclass
class _DummyAnalysisJob:
    job_id: str
    config: dict

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "config": self.config}

    @classmethod
    def from_dict(cls, d: dict) -> "_DummyAnalysisJob":
        return cls(**d)


def _dummy_analysis_cls(module_id: str = "_dummy_analysis") -> type:
    class _DummyAnalysis(AnalysisModule):
        id_prefix = "an"
        requires = ["tracking"]
        job_class = _DummyAnalysisJob
        progress_event_class = None

        def validate(self, experiment_dir, config):
            return config

        def build_job(self, run_id, config):
            return _DummyAnalysisJob(job_id=run_id, config=dict(config))

        def runner(self, job, run_dir, progress_cb, stop_event):
            pass

        def run_dir_for(self, experiment_root, job):
            return experiment_root

        def output_manifest(self, run_id, config):
            return []

    _DummyAnalysis.module_id = module_id
    return _DummyAnalysis


def test_register_analysis_module_wires_all_seams():
    cls = _dummy_analysis_cls()
    try:
        register_analysis_module(cls)
        assert isinstance(ANALYSIS_MODULES["_dummy_analysis"], cls)
        assert "_dummy_analysis" in JOB_RUNNERS  # shared dispatch table
        assert _JOB_TYPES["_dummy_analysis"] is _DummyAnalysisJob
    finally:
        ANALYSIS_MODULES.pop("_dummy_analysis", None)
        JOB_RUNNERS.pop("_dummy_analysis", None)
        _JOB_TYPES.pop("_dummy_analysis", None)
        _JOB_TAGS.pop(_DummyAnalysisJob, None)


def test_register_raises_on_duplicate_id_within_family():
    register_analysis_module(_dummy_analysis_cls())
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_analysis_module(_dummy_analysis_cls())  # same module_id
    finally:
        ANALYSIS_MODULES.pop("_dummy_analysis", None)
        JOB_RUNNERS.pop("_dummy_analysis", None)
        _JOB_TYPES.pop("_dummy_analysis", None)
        _JOB_TAGS.pop(_DummyAnalysisJob, None)


def test_register_raises_on_id_taken_by_a_processing_module():
    import_runner_modules(PIPELINE_RUNNERS)  # registers detection/tracking/...
    with pytest.raises(ValueError, match="already registered"):
        register_analysis_module(_dummy_analysis_cls("detection"))  # cross-family clash
