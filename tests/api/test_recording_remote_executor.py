"""Recording.run()'s executor choice (§6): RemoteExecutor when compute.backend is
"remote", LocalExecutor otherwise — via executor_for(), the one shared choice point
every processing-module-shaped caller submits through (detection/tracking/embedding
here; label_dataset/train_model in tests/api/test_experiment_remote_executor.py).
``module.validate``/``build_job`` are stubbed so this tests only the executor choice,
not detection's own validation.
"""

from detail.api.experiment import Experiment
from detail.core.dataclasses import ComputeTarget, DetectionJob, JobStatus, Prompt
from detail.executor import remote as remote_module
from detail.executor.runners import PROCESSING_MODULES


class _FakeExecutor:
    """Records how it was constructed and satisfies just enough of Executor for
    Recording.run() to finish (submit + status; ExecutorRunHandle wraps the rest)."""

    log: list[tuple[str, tuple, dict]] = []

    def __init__(self, experiment_root, *args, runner_modules=(), **kwargs):
        type(self).log.append((type(self).__name__, args, kwargs))
        self._experiment_root = experiment_root

    def submit(self, job) -> str:
        self._job = job
        return job.job_id

    def status(self, job_id: str) -> JobStatus:
        return JobStatus.COMPLETE

    def error(self, job_id: str):
        return None

    def cancel(self, job_id: str) -> None:
        pass

    def stream_progress(self, job_id: str):
        return iter(())


class _FakeLocalExecutor(_FakeExecutor):
    log: list[tuple[str, tuple, dict]] = []


class _FakeRemoteExecutor(_FakeExecutor):
    log: list[tuple[str, tuple, dict]] = []


def _fake_validate(recording_dir, recording_data, config):
    return config


def _fake_build_job(recording_id, run_id, config):
    return DetectionJob(
        job_id=run_id,
        recording_id=recording_id,
        model_id="fake_model",
        prompt=Prompt(targets=[]),
        conf_threshold=0.5,
        compute=config["compute"],
        created_at="2026-01-01T00:00:00Z",
    )


def _stub_module_and_executors(monkeypatch):
    module = PROCESSING_MODULES["detection"]
    monkeypatch.setattr(module, "validate", _fake_validate)
    monkeypatch.setattr(module, "build_job", _fake_build_job)
    _FakeLocalExecutor.log = []
    _FakeRemoteExecutor.log = []
    # executor_for() resolves these as its own module globals, so patching them here
    # reaches every caller (Recording.run(), label_dataset, train_model) through the
    # one shared choice point rather than the callers' own (now nonexistent) imports.
    monkeypatch.setattr(remote_module, "LocalExecutor", _FakeLocalExecutor)
    monkeypatch.setattr(remote_module, "RemoteExecutor", _FakeRemoteExecutor)


def test_run_uses_local_executor_by_default(tmp_path, monkeypatch):
    _stub_module_and_executors(monkeypatch)
    proj = Experiment.create("proj", tmp_path)
    recording = proj.create_recording("recording_001")

    recording.run(
        "detection",
        {"model_id": "fake_model", "compute": ComputeTarget("cpu", None, None, None)},
    )

    assert len(_FakeLocalExecutor.log) == 1
    assert _FakeRemoteExecutor.log == []


def test_run_uses_remote_executor_for_remote_backend(tmp_path, monkeypatch):
    _stub_module_and_executors(monkeypatch)
    proj = Experiment.create("proj", tmp_path)
    recording = proj.create_recording("recording_001")

    recording.run(
        "detection",
        {
            "model_id": "fake_model",
            "compute": ComputeTarget(
                "remote", None, None, None, remote_target="tillicum"
            ),
        },
    )

    assert _FakeLocalExecutor.log == []
    assert len(_FakeRemoteExecutor.log) == 1
    name, args, kwargs = _FakeRemoteExecutor.log[0]
    assert args == ("tillicum",)


def test_run_uses_local_executor_when_compute_is_absent(tmp_path, monkeypatch):
    """A module with no compute field at all (e.g. one that never sets it) must not
    crash the (compute is not None and ...) check."""
    _stub_module_and_executors(monkeypatch)

    def _build_job_no_compute(recording_id, run_id, config):
        return DetectionJob(
            job_id=run_id,
            recording_id=recording_id,
            model_id="fake_model",
            prompt=Prompt(targets=[]),
            conf_threshold=0.5,
            compute=ComputeTarget("cpu", None, None, None),
            created_at="2026-01-01T00:00:00Z",
        )

    monkeypatch.setattr(
        PROCESSING_MODULES["detection"], "build_job", _build_job_no_compute
    )
    proj = Experiment.create("proj", tmp_path)
    recording = proj.create_recording("recording_001")

    recording.run("detection", {"model_id": "fake_model"})  # no "compute" key at all
    assert len(_FakeLocalExecutor.log) == 1
