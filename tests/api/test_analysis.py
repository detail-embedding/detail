"""Experiment-scoped analysis runs (§16) — the framework backbone, end-to-end.

Uses a fake analysis module written to a temp file on ``sys.path`` (the same
harness as ``test_run_detection_end_to_end``) so the run executes through a real
worker subprocess: register → run_analysis → submit → subprocess → analysis/<id>/
→ discover → delete.
"""

import json
import os
import sys

import pytest

import detail.api.experiment as api_experiment
from detail.api.experiment import Experiment
from detail.storage import analysis_runs as ar
from detail.core.dataclasses import _JOB_TAGS, _JOB_TYPES
from detail.executor.runners import ANALYSIS_MODULES, JOB_RUNNERS

FAKE_ANALYSIS_SOURCE = """
from dataclasses import dataclass
from pathlib import Path

from detail.executor.runners import AnalysisModule, register_analysis_module
from detail.storage import analysis_runs as ar


@dataclass
class _FakeAnalysisJob:
    job_id: str
    config: dict

    def to_dict(self):
        return {"job_id": self.job_id, "config": self.config}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@register_analysis_module
class _FakeAnalysis(AnalysisModule):
    module_id = "fake_analysis"
    id_prefix = "an"
    requires = []
    job_class = _FakeAnalysisJob
    progress_event_class = None

    def validate(self, experiment_dir, config):
        return config

    def build_job(self, run_id, config):
        return _FakeAnalysisJob(job_id=run_id, config=dict(config))

    def runner(self, job, run_dir, progress_cb, stop_event):
        run_dir = Path(run_dir)
        for sid in job.config.get("source_recording_ids", []):
            (run_dir / sid).mkdir(parents=True, exist_ok=True)
            (run_dir / sid / "result.txt").write_text("ok")
        rec = ar.read_analysis_yaml(run_dir)
        rec["artifacts"] = self.output_manifest(job.job_id, job.config)
        ar.write_analysis_yaml(run_dir, rec)

    def run_dir_for(self, experiment_root, job):
        return ar.analysis_dir(experiment_root, job.job_id)

    def output_manifest(self, run_id, config):
        return [
            {"path": f"{sid}/result.txt", "kind": "text", "label": sid}
            for sid in config.get("source_recording_ids", [])
        ]
"""


@pytest.fixture
def fake_analysis(tmp_path, monkeypatch):
    """Make a fake analysis module importable by both this process and the worker
    subprocess, and clean the global registries afterward."""
    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    (mod_dir / "fake_analysis_mod.py").write_text(
        FAKE_ANALYSIS_SOURCE, encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(mod_dir))
    monkeypatch.setenv(
        "PYTHONPATH", str(mod_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    monkeypatch.setattr(
        api_experiment,
        "_PIPELINE_RUNNERS",
        api_experiment._PIPELINE_RUNNERS + ("fake_analysis_mod",),
    )
    try:
        yield
    finally:  # the parent process registered "fake_analysis"; undo it fully so the
        # next test re-imports + re-registers (drop the cached module too)
        ANALYSIS_MODULES.pop("fake_analysis", None)
        JOB_RUNNERS.pop("fake_analysis", None)
        cls = _JOB_TYPES.pop("fake_analysis", None)
        _JOB_TAGS.pop(cls, None)
        sys.modules.pop("fake_analysis_mod", None)


def test_run_analysis_end_to_end(tmp_path, fake_analysis):
    proj = Experiment.create("proj", tmp_path)
    proj.create_recording("recording_001")

    run = proj.run_analysis(
        "fake_analysis", {"source_recording_ids": ["recording_001"]}
    )
    run.wait()
    assert run.status == "complete"

    refs = proj.analysis_runs  # discovered by scanning analysis/
    assert [r.id for r in refs] == ["an_01"]
    assert refs[0].module_id == "fake_analysis"
    assert refs[0].recording_ids == ["recording_001"]

    rec = proj.analysis_record("an_01")
    assert rec["status"] == "complete"
    assert rec["artifacts"][0]["path"] == "recording_001/result.txt"  # relative path
    assert (proj.dir / "analysis" / "an_01" / "recording_001" / "result.txt").is_file()


def test_run_analysis_batches_over_recordings(tmp_path, fake_analysis):
    proj = Experiment.create("proj", tmp_path)
    for sid in ("recording_001", "recording_002"):
        proj.create_recording(sid)
    run = proj.run_analysis(
        "fake_analysis", {"source_recording_ids": ["recording_001", "recording_002"]}
    )
    run.wait()
    assert run.status == "complete"
    for sid in ("recording_001", "recording_002"):
        assert (proj.dir / "analysis" / "an_01" / sid / "result.txt").is_file()


def test_run_analysis_rejects_unknown_module(tmp_path, fake_analysis):
    proj = Experiment.create("proj", tmp_path)
    with pytest.raises(ValueError, match="unknown analysis module"):
        proj.run_analysis("nope", {})


def test_delete_analysis_after_complete(tmp_path, fake_analysis):
    proj = Experiment.create("proj", tmp_path)
    proj.create_recording("recording_001")
    run = proj.run_analysis(
        "fake_analysis", {"source_recording_ids": ["recording_001"]}
    )
    run.wait()
    proj.delete_analysis("an_01")  # completed → deletable
    assert proj.analysis_runs == []


def test_delete_analysis_blocks_while_active(tmp_path):
    # A still-running record must not be deletable (synthesize it directly — no
    # need for a hanging worker).
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("proj", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {"id": "an_01", "module_id": "x", "status": "running"},
    )
    with pytest.raises(ValueError, match="still active"):
        proj.delete_analysis("an_01")


class _Inner:
    """Minimal executor-handle stub for AnalysisRun.wait().

    ``error`` is part of the shape ``ExecutorRunHandle`` has always had: a reconcile records
    *why* a run failed, not only that it did, so a stub without it is a stub of the wrong thing.
    """

    def __init__(self, status, error=None):
        self.status = status
        self.error = error

    def wait(self):
        return self


def test_analysis_wait_keeps_runner_partial(tmp_path):
    # wait() must not clobber the runner's terminal status with the executor's
    # "complete": a partial run (some recordings failed) stays partial.
    from detail.api.experiment import AnalysisRun
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("proj", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    ar.write_analysis_yaml(
        adir, {"id": "an_01", "module_id": "location_export", "status": "partial"}
    )

    AnalysisRun(_Inner("complete"), proj, "an_01").wait()
    assert ar.read_analysis_yaml(adir)["status"] == "partial"  # not overwritten


def test_analysis_wait_fixes_up_crashed_run(tmp_path):
    # If the runner crashed before finalizing (record left "running"), wait()
    # reconciles to the executor's terminal status.
    from detail.api.experiment import AnalysisRun
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("proj", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    ar.write_analysis_yaml(
        adir, {"id": "an_01", "module_id": "location_export", "status": "running"}
    )

    AnalysisRun(_Inner("failed"), proj, "an_01").wait()
    assert ar.read_analysis_yaml(adir)["status"] == "failed"  # fixed up from running


# --- reattach: a reopened page must stream a run it did not launch (§16) ----- #
def _running_analysis(proj, run_id="an_001", *, pid=None, progress=None):
    """An analysis run dir the way the executor leaves one mid-run — record, job
    envelope, optional worker pid and last progress — without launching a worker."""
    from detail.core.dataclasses import JobStatus
    from detail.executor import rundir

    adir = ar.analysis_dir(proj.dir, run_id)
    adir.mkdir(parents=True, exist_ok=True)
    ar.write_analysis_yaml(
        adir,
        {
            "id": run_id,
            "module_id": "fake_analysis",
            "recording_ids": [],
            "status": "running",
            "created_at": "2026-08-01T10:00:00Z",
        },
    )
    envelope = {
        "schema_version": 1,
        "status": JobStatus.RUNNING.value,
        "job": {"job_id": run_id, "config": {}},
        "progress": progress,
        "error": None,
    }
    if pid is not None:
        envelope["pid"] = pid
    (adir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")
    return adir


def test_attach_active_analysis_run_only_for_an_active_run(tmp_path):
    proj = Experiment.create("p", tmp_path)
    adir = _running_analysis(proj)
    assert proj.attach_active_analysis_run("an_001") is not None

    ar.set_analysis_status(adir, "complete")
    assert proj.attach_active_analysis_run("an_001") is None  # static record, no bar
    assert proj.attach_active_analysis_run("an_404") is None  # absent, not an error


def test_attached_analysis_run_streams_a_run_it_did_not_launch(tmp_path):
    """The reported bug's mechanism.

    The launch-time handle wraps a LocalExecutor whose run dirs live in memory, populated
    by submit — so a reopened page could not stream a run this process didn't start. This
    polls the run dir instead, so a *fresh* Experiment streams it to completion.
    """
    from detail.core.dataclasses import (
        LabelingProgressEvent,
        progress_event_to_dict,
    )
    from detail.executor import rundir

    proj = Experiment.create("p", tmp_path)
    event = LabelingProgressEvent(3, 10, 0, 1, 5.0, 12.0, "working")
    adir = _running_analysis(proj, progress=progress_event_to_dict(event))

    handle = Experiment.open(proj.dir).attach_analysis_run("an_001")  # a fresh process
    handle._poll = 0.0
    streamed = []
    for ev in handle.progress():
        streamed.append(ev)
        if len(streamed) == 1:  # the worker finishes while we watch
            rundir.set_status(adir, rundir.JobStatus.COMPLETE)

    assert [e.frames_done for e in streamed] == [3]  # the on-disk event, not a guess
    assert streamed[0].message == "working"
    assert handle.status == "complete"


def test_attached_analysis_run_reconciles_a_killed_worker(tmp_path):
    """A killed worker used to leave the record reading "running" forever — which also
    made it undeletable, because delete_analysis refuses an active run."""
    proj = Experiment.create("p", tmp_path)
    adir = _running_analysis(proj, pid=_dead_pid())

    handle = proj.attach_analysis_run("an_001")
    handle._poll = 0.0
    handle.wait()

    assert handle.status == "failed"
    assert ar.read_analysis_yaml(adir)["status"] == "failed"
    proj.delete_analysis("an_001")  # no longer blocked by a phantom active run


def test_reconcile_analysis_runs_fixes_a_killed_run_on_open(tmp_path):
    proj = Experiment.create("p", tmp_path)
    adir = _running_analysis(proj, pid=_dead_pid())

    proj.reconcile_analysis_runs()  # what the GUI calls when it opens an experiment

    record = ar.read_analysis_yaml(adir)
    assert record["status"] == "failed"
    assert record.get("error")  # says why, not just that


def test_reconcile_analysis_runs_leaves_a_live_run_alone(tmp_path):
    proj = Experiment.create("p", tmp_path)
    adir = _running_analysis(proj, pid=os.getpid())  # this process is very much alive

    proj.reconcile_analysis_runs()

    assert ar.read_analysis_yaml(adir)["status"] == "running"


def _dead_pid() -> int:
    """A pid that is certainly not running (spawn one and reap it)."""
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid
