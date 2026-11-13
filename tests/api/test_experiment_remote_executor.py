"""Experiment.label_dataset()/train_model()'s executor choice (§6, §12): both already
carried a compute field, but hardcoded LocalExecutor regardless of it -- found live
running a real distillation-shaped workload against Tillicum. executor_for() (the
same choice point Recording.run() uses, see test_recording_remote_executor.py) fixes
that; these tests are the distill side of that same guard.
"""

import contextlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import (
    ComputeTarget,
    LabelingClass,
    ModelReadiness,
    TextCue,
)
from detail.core.protocols import DetectionModel, Trainer
from detail.core.registries import MODELS, TRAINERS, register_trainer
from detail.executor import remote as remote_module
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import remote_targets

H, W = 24, 32


class _FakeSAM3(DetectionModel):
    model_id = "sam3"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu", "cuda"]

    def load(self, device):
        pass

    def predict(self, frame, prompt):
        raise NotImplementedError  # never reached: these tests stop at submit()

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


class _CpuOnlySAM3(_FakeSAM3):
    model_id = "cpu_only_sam3"
    supported_backends = ["cpu"]


class _NotInstalledSAM3(_FakeSAM3):
    """A model this machine has not installed — readiness() reports it, unlike
    _FakeSAM3, which inherits the base class's always-ready default."""

    model_id = "not_installed_sam3"

    def readiness(self):
        return ModelReadiness(
            ready=False,
            reason="not_installed",
            message="not_installed_sam3 weights are not installed on this machine",
            action_url=None,
            repo_id=None,
        )


class _StubTrainer(Trainer):
    trainer_id = "stub_trainer"
    model_family = "yolo"
    supported_cue_types = []
    supported_backends = ["cpu", "cuda"]

    def is_ready(self):
        return True

    def prepare_dataset(self, dataset_dir, work_dir):
        raise NotImplementedError

    def train(self, *a, **k):
        raise NotImplementedError

    def resume(self, *a, **k):
        raise NotImplementedError

    def checkpoint_path(self, model_dir):
        p = Path(model_dir) / "run" / "weights" / "last.pt"
        return p if p.is_file() else None

    def best_weights_path(self, model_dir):
        p = Path(model_dir) / "run" / "weights" / "best.pt"
        return p if p.is_file() else None

    def resolved_settings(self, model_dir):
        return {}

    def final_metrics(self, model_dir):
        return {}

    def metric_history(self, model_dir):
        return []

    def validate_hyperparams(self, hyperparams):
        return dict(hyperparams or {})


class _CpuOnlyTrainer(_StubTrainer):
    trainer_id = "cpu_only_trainer"
    supported_backends = ["cpu"]


class _NotInstalledTrainer(_StubTrainer):
    """A trainer framework this machine has not installed — is_ready() reports it,
    unlike _StubTrainer, which is always ready."""

    trainer_id = "not_installed_trainer"

    def is_ready(self):
        return False


@contextlib.contextmanager
def _registered_model(model_id, cls):
    MODELS[model_id] = cls
    try:
        yield
    finally:
        MODELS.pop(model_id, None)


@contextlib.contextmanager
def _registered_trainer(cls):
    register_trainer(cls.trainer_id)(cls)
    try:
        yield
    finally:
        TRAINERS.pop(cls.trainer_id, None)


def _write_video(path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for f in range(n_frames):
        vw.write(np.full((H, W, 3), (f * 7) % 250 + 1, np.uint8))
    vw.release()


def _api_experiment(tmp_path, n_frames=20):
    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("recording_001")
    _write_video(recording.dir / "video" / "part_001.mp4", n_frames)
    data = recording_store.read_recording(recording.dir)
    ts_csv = recording.dir / "video" / "timestamps" / "generated" / "part_001.csv"
    ts_csv.parent.mkdir(parents=True, exist_ok=True)
    recording_store.write_timestamps_csv(
        ts_csv, np.arange(n_frames, dtype=np.int64) * 1000
    )
    data["video"] = {
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ]
    }
    recording_store.write_recording(recording.dir, data)
    return proj


def _dataset(proj, dataset_id="dataset_001"):
    from detail.storage import datasets as datasets_store

    ddir = datasets_store.dataset_dir(proj.dir, dataset_id)
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        ddir,
        {
            "id": dataset_id,
            "name": "d",
            "status": "complete",
            "classes": [{"id": 0, "name": "mouse"}],
        },
    )
    datasets_store.write_data_yaml(ddir, {0: "mouse"})
    return proj


def _register_remote_target(monkeypatch, tmp_path, name="tillicum"):
    cfg_dir = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg_dir / "remote_targets.yaml"
    )
    remote_targets.register_remote_target(
        RemoteTargetConfig(name=name, host=name, remote_workdir="/scratch/x")
    )


class _FakeExecutor:
    log: list = []

    def __init__(self, experiment_root, *args, runner_modules=(), **kwargs):
        type(self).log.append((type(self).__name__, args, kwargs))

    def submit(self, job) -> str:
        return job.job_id

    def status(self, job_id):
        from detail.core.dataclasses import JobStatus

        return JobStatus.COMPLETE

    def error(self, job_id):
        return None

    def cancel(self, job_id) -> None:
        pass

    def stream_progress(self, job_id):
        return iter(())


class _FakeLocalExecutor(_FakeExecutor):
    log: list = []


class _FakeRemoteExecutor(_FakeExecutor):
    log: list = []


def _stub_executors(monkeypatch):
    _FakeLocalExecutor.log = []
    _FakeRemoteExecutor.log = []
    monkeypatch.setattr(remote_module, "LocalExecutor", _FakeLocalExecutor)
    monkeypatch.setattr(remote_module, "RemoteExecutor", _FakeRemoteExecutor)


# --------------------------------------------------------------------------- #
# label_dataset() — executor choice
# --------------------------------------------------------------------------- #
def test_label_dataset_uses_local_executor_by_default(tmp_path, monkeypatch):
    proj = _api_experiment(tmp_path)
    _stub_executors(monkeypatch)
    with _registered_model("sam3", _FakeSAM3):
        proj.label_dataset(
            "d",
            [LabelingClass(0, "mouse", "a mouse")],
            n_frames=4,
            compute=ComputeTarget("cpu", None, None, None),
        )
    assert _FakeLocalExecutor.log and not _FakeRemoteExecutor.log


def test_label_dataset_uses_remote_executor_for_remote_backend(tmp_path, monkeypatch):
    proj = _api_experiment(tmp_path)
    _stub_executors(monkeypatch)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_model("sam3", _FakeSAM3):
        proj.label_dataset(
            "d",
            [LabelingClass(0, "mouse", "a mouse")],
            n_frames=4,
            compute=ComputeTarget("remote", None, None, None, remote_target="tillicum"),
        )
    assert _FakeRemoteExecutor.log and not _FakeLocalExecutor.log
    assert _FakeRemoteExecutor.log[0][1] == ("tillicum",)


def test_label_dataset_remote_requires_a_remote_target(tmp_path):
    proj = _api_experiment(tmp_path)
    with _registered_model("sam3", _FakeSAM3):
        with pytest.raises(ValueError, match="no remote_target was given"):
            proj.label_dataset(
                "d",
                [LabelingClass(0, "mouse", "a mouse")],
                n_frames=4,
                compute=ComputeTarget("remote", None, None, None),
            )


def test_label_dataset_remote_rejects_unregistered_target(tmp_path, monkeypatch):
    proj = _api_experiment(tmp_path)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_model("sam3", _FakeSAM3):
        with pytest.raises(ValueError, match="no remote target registered"):
            proj.label_dataset(
                "d",
                [LabelingClass(0, "mouse", "a mouse")],
                n_frames=4,
                compute=ComputeTarget("remote", None, None, None, remote_target="nope"),
            )


def test_label_dataset_remote_rejects_cpu_only_model(tmp_path, monkeypatch):
    proj = _api_experiment(tmp_path)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_model("cpu_only_sam3", _CpuOnlySAM3):
        with pytest.raises(ValueError, match="does not support the 'remote' backend"):
            proj.label_dataset(
                "d",
                [LabelingClass(0, "mouse", "a mouse")],
                n_frames=4,
                model_id="cpu_only_sam3",
                compute=ComputeTarget(
                    "remote", None, None, None, remote_target="tillicum"
                ),
            )


def test_label_dataset_local_rejects_a_not_installed_model(tmp_path):
    proj = _api_experiment(tmp_path)
    with _registered_model("not_installed_sam3", _NotInstalledSAM3):
        with pytest.raises(ValueError, match="not_installed_sam3 is not ready"):
            proj.label_dataset(
                "d",
                [LabelingClass(0, "mouse", "a mouse")],
                n_frames=4,
                model_id="not_installed_sam3",
                compute=ComputeTarget("cpu", None, None, None),
            )


def test_label_dataset_remote_skips_the_local_model_access_check(tmp_path, monkeypatch):
    """The model is not installed *here*, but the shepherd's own readiness probe
    checks it on the cluster before the sbatch submission (remote_shepherd.py) --
    gating this machine's submission on its own disk/network access would demand
    an install for a job that never loads the model here. Found live: a remote
    embedding job hit the identical bug (Experiment.check_model_access called
    unconditionally); label_dataset had the same shape and no test caught it."""
    proj = _api_experiment(tmp_path)
    _stub_executors(monkeypatch)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_model("not_installed_sam3", _NotInstalledSAM3):
        proj.label_dataset(
            "d",
            [LabelingClass(0, "mouse", "a mouse")],
            n_frames=4,
            model_id="not_installed_sam3",
            compute=ComputeTarget("remote", None, None, None, remote_target="tillicum"),
        )
    assert _FakeRemoteExecutor.log and not _FakeLocalExecutor.log


# --------------------------------------------------------------------------- #
# train_model() — executor choice
# --------------------------------------------------------------------------- #
def test_train_model_uses_local_executor_by_default(tmp_path, monkeypatch):
    proj = _dataset(_api_experiment(tmp_path))
    _stub_executors(monkeypatch)
    with _registered_trainer(_StubTrainer):
        proj.train_model(
            "m",
            "dataset_001",
            "stub_trainer",
            "yolo11s-seg.pt",
            compute=ComputeTarget("cpu", None, None, None),
        )
    assert _FakeLocalExecutor.log and not _FakeRemoteExecutor.log


def test_train_model_uses_remote_executor_for_remote_backend(tmp_path, monkeypatch):
    proj = _dataset(_api_experiment(tmp_path))
    _stub_executors(monkeypatch)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_trainer(_StubTrainer):
        proj.train_model(
            "m",
            "dataset_001",
            "stub_trainer",
            "yolo11s-seg.pt",
            compute=ComputeTarget("remote", None, None, None, remote_target="tillicum"),
        )
    assert _FakeRemoteExecutor.log and not _FakeLocalExecutor.log
    assert _FakeRemoteExecutor.log[0][1] == ("tillicum",)


def test_train_model_remote_requires_a_remote_target(tmp_path):
    proj = _dataset(_api_experiment(tmp_path))
    with _registered_trainer(_StubTrainer):
        with pytest.raises(ValueError, match="no remote_target was given"):
            proj.train_model(
                "m",
                "dataset_001",
                "stub_trainer",
                "yolo11s-seg.pt",
                compute=ComputeTarget("remote", None, None, None),
            )


def test_train_model_remote_rejects_unregistered_target(tmp_path, monkeypatch):
    proj = _dataset(_api_experiment(tmp_path))
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_trainer(_StubTrainer):
        with pytest.raises(ValueError, match="no remote target registered"):
            proj.train_model(
                "m",
                "dataset_001",
                "stub_trainer",
                "yolo11s-seg.pt",
                compute=ComputeTarget("remote", None, None, None, remote_target="nope"),
            )


def test_train_model_remote_rejects_cpu_only_trainer(tmp_path, monkeypatch):
    proj = _dataset(_api_experiment(tmp_path))
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_trainer(_CpuOnlyTrainer):
        with pytest.raises(ValueError, match="does not support the 'remote' backend"):
            proj.train_model(
                "m",
                "dataset_001",
                "cpu_only_trainer",
                "yolo11s-seg.pt",
                compute=ComputeTarget(
                    "remote", None, None, None, remote_target="tillicum"
                ),
            )


def test_train_model_local_rejects_a_not_ready_trainer(tmp_path):
    proj = _dataset(_api_experiment(tmp_path))
    with _registered_trainer(_NotInstalledTrainer):
        with pytest.raises(ValueError, match="not_installed_trainer' is not ready"):
            proj.train_model(
                "m",
                "dataset_001",
                "not_installed_trainer",
                "yolo11s-seg.pt",
                compute=ComputeTarget("cpu", None, None, None),
            )


def test_train_model_remote_skips_the_local_trainer_readiness_check(
    tmp_path, monkeypatch
):
    """The training framework is not importable *here*, but training happens
    entirely on the cluster for a remote job -- gating this machine's submission
    on its own backend availability would demand the framework be installed for a
    job that never trains here. Same shape as label_dataset's local-access skip
    and detection's/embedding's local-readiness skip; found by inspection once
    those were fixed, not live, but the class is identical."""
    proj = _dataset(_api_experiment(tmp_path))
    _stub_executors(monkeypatch)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    with _registered_trainer(_NotInstalledTrainer):
        proj.train_model(
            "m",
            "dataset_001",
            "not_installed_trainer",
            "yolo11s-seg.pt",
            compute=ComputeTarget("remote", None, None, None, remote_target="tillicum"),
        )
    assert _FakeRemoteExecutor.log and not _FakeLocalExecutor.log


# --------------------------------------------------------------------------- #
# resume_training() — re-dispatches through the original job's own compute
# --------------------------------------------------------------------------- #
def test_resume_training_uses_remote_executor_when_the_original_job_was_remote(
    tmp_path, monkeypatch
):
    from detail.core.dataclasses import TrainingJob
    from detail.executor import rundir
    from detail.storage import models as models_store

    proj = _dataset(_api_experiment(tmp_path))
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    mdir = models_store.model_dir(proj.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "trainer_id": "stub_trainer",
            "dataset_id": "dataset_001",
            "base_model": "yolo11s-seg.pt",
            "status": "failed",
        },
    )
    job = TrainingJob(
        job_id="model_001",
        model_id="model_001",
        dataset_id="dataset_001",
        trainer_id="stub_trainer",
        base_model="yolo11s-seg.pt",
        hyperparams={},
        compute=ComputeTarget("remote", None, None, None, remote_target="tillicum"),
        created_at="2026-06-09T00:00:00Z",
    )
    rundir.init_run_dir(mdir, job)
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"fake-last")
    _stub_executors(monkeypatch)
    with _registered_trainer(_StubTrainer):
        proj.resume_training("model_001")
    assert _FakeRemoteExecutor.log and not _FakeLocalExecutor.log
    assert _FakeRemoteExecutor.log[0][1] == ("tillicum",)
