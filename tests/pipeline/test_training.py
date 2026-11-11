"""Training runner + Experiment.train_model (§12), via a stub Trainer."""

import contextlib
from pathlib import Path

import pytest

from detail.api.experiment import Experiment
from detail.core.dataclasses import ComputeTarget, TrainingJob
from detail.core.protocols import Trainer
from detail.core.registries import TRAINERS, register_trainer
from detail.executor import rundir
from detail.pipeline import training
from detail.storage import datasets as datasets_store
from detail.storage import models as models_store
from detail.storage import experiment as experiment_store


class StubTrainer(Trainer):
    trainer_id = "stub_trainer"
    model_family = "yolo"
    supported_cue_types = []
    supported_backends = ["cpu"]

    def is_ready(self):
        return True

    def prepare_dataset(self, dataset_dir, work_dir):
        return Path(dataset_dir) / "data.yaml"

    def train(
        self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
    ):
        weights = Path(model_dir) / "run" / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"fake-best")
        (weights / "last.pt").write_bytes(b"fake-last")

    def resume(
        self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
    ):
        self.train(job, dataset_dir, model_dir, device, progress_cb, stop_event)

    def checkpoint_path(self, model_dir):
        p = Path(model_dir) / "run" / "weights" / "last.pt"
        return p if p.is_file() else None

    def best_weights_path(self, model_dir):
        p = Path(model_dir) / "run" / "weights" / "best.pt"
        return p if p.is_file() else None

    def final_metrics(self, model_dir):
        return {"box_map50": 0.9}

    def metric_history(self, model_dir):
        return [{"epoch": 1, "box_map50": 0.5}, {"epoch": 2, "box_map50": 0.9}]

    def validate_hyperparams(self, hyperparams):
        if "bad" in hyperparams:
            raise ValueError("unknown hyperparam 'bad'")
        return {"epochs": int(hyperparams.get("epochs", 10))}

    def resolved_settings(self, model_dir):
        return {"backend": "stub", "version": "1.2.3", "args": {"epochs": 10}}


@contextlib.contextmanager
def _registered_stub():
    register_trainer("stub_trainer")(StubTrainer)
    try:
        yield
    finally:
        TRAINERS.pop("stub_trainer", None)


def _dataset(proj, dataset_id="dataset_001", status="complete"):
    ddir = datasets_store.dataset_dir(proj, dataset_id)
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        ddir,
        {
            "id": dataset_id,
            "name": "d",
            "status": status,
            "classes": [{"id": 0, "name": "mouse"}],
        },
    )
    datasets_store.write_data_yaml(ddir, {0: "mouse"})
    return proj


def _job(model_id="model_001", trainer_id="stub_trainer"):
    return TrainingJob(
        job_id=model_id,
        model_id=model_id,
        dataset_id="dataset_001",
        trainer_id=trainer_id,
        base_model="yolo11s-seg.pt",
        hyperparams={"epochs": 10},
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-06-09T00:00:00Z",
    )


# --- runner ----------------------------------------------------------------- #
def test_training_runner_writes_model_yaml(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "mouse_v1",
            "trainer_id": "stub_trainer",
            "dataset_id": "dataset_001",
            "base_model": "yolo11s-seg.pt",
            "status": "running",
        },
    )
    with _registered_stub():
        training.run(_job(), mdir, progress_cb=lambda e: None, stop_event=None)

    rec = models_store.read_model_yaml(mdir)
    assert rec["status"] == "complete"
    assert rec["name"] == "mouse_v1"  # preserved from the initial record
    assert rec["weights"] == {  # recorded relative to the model dir (Tenet 6)
        "best": "run/weights/best.pt",
        "last": "run/weights/last.pt",
    }
    assert rec["metrics"] == {"box_map50": 0.9}
    assert rec["classes"] == [{"id": 0, "name": "mouse"}]
    assert rec["preview"] is None  # the stub dataset has no val frames → no preview
    # What trained it: the settings asked for, and what the backend resolved, with its
    # version — the readers that showed `{}` for every model read these.
    assert rec["hyperparams"] == {"epochs": 10}
    assert rec["trained_with"] == {
        "backend": "stub",
        "version": "1.2.3",
        "args": {"epochs": 10},
    }


def test_train_model_records_the_resolved_hyperparams_at_submit(tmp_path, monkeypatch):
    """The live training curve and `train status` read the epoch count off model.yaml while
    the run is going, so the resolved hyperparameters are there before the first epoch.
    """
    import detail.api.experiment as api_experiment

    class _NoSubmit:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, job):
            pass

    monkeypatch.setattr(api_experiment, "LocalExecutor", _NoSubmit)
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    with _registered_stub():
        run = Experiment(proj).train_model(
            "m", "dataset_001", "stub_trainer", "yolo11s-seg.pt", hyperparams={}
        )
    rec = models_store.read_model_yaml(models_store.model_dir(proj, run.id))
    assert rec["hyperparams"] == {"epochs": 10}  # the trainer's default, filled in
    assert rec["status"] == "running"


def _stub_trained_yolo(monkeypatch, detections=()):
    """Replace the trained-model loader so preview capture runs no real inference
    (the only-mock-inference rule) and pulls in no ultralytics."""
    from detail.core.dataclasses import FrameResult

    class _StubModel:
        def __init__(self, weights, classes):
            self.classes = classes

        def load(self, device):
            pass

        def predict(self, frame_bgr, prompt):
            return FrameResult.from_detections(
                frame_idx=0, t_ns=0, detections=list(detections)
            )

    monkeypatch.setattr("detail.pipeline.training.TrainedYOLO", _StubModel)


def _write_val_frame(dataset_dir):
    import cv2
    import numpy as np

    val = Path(dataset_dir) / "images" / "val"
    val.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(val / "frame.jpg"), np.zeros((16, 24, 3), dtype=np.uint8))


def test_capture_preview_writes_overlay(tmp_path, monkeypatch):
    ddir = tmp_path / "ds"
    _write_val_frame(ddir)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _stub_trained_yolo(monkeypatch)
    rel = training._capture_preview(
        ddir, tmp_path / "best.pt", [{"id": 0, "name": "mouse"}], run_dir, "cpu", None
    )
    assert rel == "preview.png"
    assert (run_dir / "preview.png").is_file()


def test_capture_preview_no_val_frames_returns_none(tmp_path, monkeypatch):
    ddir = tmp_path / "ds"
    (ddir / "images" / "val").mkdir(parents=True)  # exists but empty
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _stub_trained_yolo(monkeypatch)
    assert (
        training._capture_preview(ddir, tmp_path / "b.pt", [], run_dir, "cpu", None)
        is None
    )


def test_capture_preview_swallows_inference_error(tmp_path, monkeypatch):
    # Isolation boundary: a preview failure must never fail an otherwise-valid model.
    ddir = tmp_path / "ds"
    _write_val_frame(ddir)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class _Boom:
        def __init__(self, weights, classes):
            pass

        def load(self, device):
            raise RuntimeError("gpu vanished")

        def predict(self, frame_bgr, prompt):
            raise AssertionError("unreachable")

    monkeypatch.setattr("detail.pipeline.training.TrainedYOLO", _Boom)
    assert (
        training._capture_preview(ddir, tmp_path / "b.pt", [], run_dir, "cpu", None)
        is None
    )
    assert not (run_dir / "preview.png").exists()


def test_training_runner_captures_preview(tmp_path, monkeypatch):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    _write_val_frame(datasets_store.dataset_dir(proj, "dataset_001"))
    _stub_trained_yolo(monkeypatch)
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "running"})
    events = []
    with _registered_stub():
        training.run(_job(), mdir, progress_cb=events.append, stop_event=None)
    rec = models_store.read_model_yaml(mdir)
    assert (
        rec["preview"] == "preview.png"
    )  # recorded relative to the model dir (Tenet 6)
    assert (mdir / "preview.png").is_file()
    # the ~1–3 s tail step names itself so the bar isn't silent (long-op rule)
    assert any(getattr(e, "message", None) == "writing preview" for e in events)


def test_training_runner_unknown_trainer(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "running"})
    with pytest.raises(ValueError, match="unknown trainer_id"):
        training.run(
            _job(trainer_id="nope"), mdir, progress_cb=lambda e: None, stop_event=None
        )
    # the crash must not leave the model stuck at "running" (else it's undeletable)
    assert models_store.read_model_yaml(mdir)["status"] == "failed"


def test_training_runner_marks_failed_when_train_raises(tmp_path):
    class _RaisingTrainer(StubTrainer):
        def train(
            self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
        ):
            raise RuntimeError("boom")

    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "running"})
    register_trainer("raising_trainer")(_RaisingTrainer)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            training.run(
                _job(trainer_id="raising_trainer"),
                mdir,
                progress_cb=lambda e: None,
                stop_event=None,
            )
    finally:
        TRAINERS.pop("raising_trainer", None)
    record = models_store.read_model_yaml(mdir)
    assert record["status"] == "failed"
    assert "boom" in record["error"]  # the message is recorded for the GUI


# --- train_model validation (Tenet 10) -------------------------------------- #
def test_train_model_missing_dataset(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    with _registered_stub():
        with pytest.raises(FileNotFoundError, match="not found"):
            Experiment(proj).train_model(
                "m", "dataset_001", "stub_trainer", "yolo11s-seg.pt"
            )


def test_train_model_rejects_incomplete_dataset(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"), status="running")
    with _registered_stub():
        with pytest.raises(ValueError, match="not complete"):
            Experiment(proj).train_model(
                "m", "dataset_001", "stub_trainer", "yolo11s-seg.pt"
            )


def test_train_model_unknown_trainer(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    with pytest.raises(ValueError, match="unknown trainer_id"):
        Experiment(proj).train_model("m", "dataset_001", "nope", "yolo11s-seg.pt")


def test_train_model_unsupported_backend(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    with _registered_stub():
        with pytest.raises(ValueError, match="does not support backend"):
            Experiment(proj).train_model(
                "m",
                "dataset_001",
                "stub_trainer",
                "yolo11s-seg.pt",
                compute=ComputeTarget("cuda", [0], None, None),
            )


def test_train_model_bad_hyperparams(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    with _registered_stub():
        with pytest.raises(ValueError, match="unknown hyperparam"):
            # Pin to a backend the stub supports so the hyperparam check is what
            # fires — otherwise on a CUDA box the backend check preempts it.
            Experiment(proj).train_model(
                "m",
                "dataset_001",
                "stub_trainer",
                "yolo11s-seg.pt",
                {"bad": 1},
                compute=ComputeTarget("cpu", None, None, None),
            )


# --- resume: the runner picks train vs resume from disk (P1a) ---------------- #
class RecordingTrainer(StubTrainer):
    """Records whether the runner chose ``train`` or ``resume``."""

    calls: list = []

    def train(
        self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
    ):
        RecordingTrainer.calls.append("train")
        StubTrainer.train(self, job, dataset_dir, model_dir, device)

    def resume(
        self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
    ):
        RecordingTrainer.calls.append("resume")
        StubTrainer.train(
            self, job, dataset_dir, model_dir, device
        )  # produce best/last


@contextlib.contextmanager
def _registered(name, cls):
    register_trainer(name)(cls)
    try:
        yield
    finally:
        TRAINERS.pop(name, None)


def _running_model_dir(proj, checkpoint: bool) -> Path:
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    if checkpoint:  # a prior (interrupted) run left last.pt behind
        weights = mdir / "run" / "weights"
        weights.mkdir(parents=True)
        (weights / "last.pt").write_bytes(b"prior")
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "status": "running",
            "trainer_id": "recording_trainer",
            "dataset_id": "dataset_001",
        },
    )
    return mdir


def test_runner_trains_when_no_checkpoint(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = _running_model_dir(proj, checkpoint=False)
    RecordingTrainer.calls = []
    with _registered("recording_trainer", RecordingTrainer):
        training.run(
            _job(trainer_id="recording_trainer"),
            mdir,
            progress_cb=lambda e: None,
            stop_event=None,
        )
    assert RecordingTrainer.calls == ["train"]  # fresh run dir → train from scratch


def test_runner_resumes_when_checkpoint_exists(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = _running_model_dir(proj, checkpoint=True)
    RecordingTrainer.calls = []
    with _registered("recording_trainer", RecordingTrainer):
        training.run(
            _job(trainer_id="recording_trainer"),
            mdir,
            progress_cb=lambda e: None,
            stop_event=None,
        )
    assert RecordingTrainer.calls == ["resume"]  # last.pt present → resume
    assert models_store.read_model_yaml(mdir)["status"] == "complete"


# --- Experiment.resume_training (P1a) ------------------------------------------- #
def test_resume_training_resets_status_and_resubmits(tmp_path, monkeypatch):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"prior")
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "status": "cancelled", "trainer_id": "stub_trainer"}
    )
    rundir.init_run_dir(mdir, _job())  # the original TrainingJob, for read_job
    submitted = []
    monkeypatch.setattr(
        "detail.api.experiment.LocalExecutor.submit",
        lambda self, job: submitted.append(job) or job.model_id,
    )
    with _registered_stub():
        run = Experiment(proj).resume_training("model_001")
    assert run is not None
    assert submitted and submitted[0].model_id == "model_001"  # re-submitted
    assert models_store.read_model_yaml(mdir)["status"] == "running"  # status reset


def test_resume_training_rejects_complete_model(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "complete"})
    with pytest.raises(ValueError, match="can be resumed"):
        Experiment(proj).resume_training("model_001")


def test_resume_training_rejects_without_checkpoint(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "status": "cancelled", "trainer_id": "stub_trainer"}
    )
    rundir.init_run_dir(mdir, _job())  # a job, but no last.pt on disk
    with _registered_stub():
        with pytest.raises(ValueError, match="no checkpoint"):
            Experiment(proj).resume_training("model_001")


# --- auto-detect default backend + cuda preflight (P1b) ---------------------- #
class CudaStubTrainer(StubTrainer):
    supported_backends = ["cpu", "cuda"]


class MpsStubTrainer(StubTrainer):
    supported_backends = ["cpu", "mps"]


def test_train_model_defaults_to_cuda_when_available(tmp_path, monkeypatch):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    checked = []
    monkeypatch.setattr(
        "detail.api.experiment.check_cuda_indices",
        lambda idx: checked.append(list(idx)),
    )
    submitted = []
    monkeypatch.setattr(
        "detail.api.experiment.LocalExecutor.submit",
        lambda self, job: submitted.append(job) or job.model_id,
    )
    with _registered("cuda_stub", CudaStubTrainer):
        Experiment(proj).train_model("m", "dataset_001", "cuda_stub", "yolo11s-seg.pt")
    assert submitted[0].compute.backend == "cuda"  # not CPU by default
    assert submitted[0].compute.device_indices == [0]
    assert checked == [[0]]  # preflight validated the requested device


def test_train_model_defaults_to_mps_when_only_mps(tmp_path, monkeypatch):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    submitted = []
    monkeypatch.setattr(
        "detail.api.experiment.LocalExecutor.submit",
        lambda self, job: submitted.append(job) or job.model_id,
    )
    with _registered("mps_stub", MpsStubTrainer):
        Experiment(proj).train_model("m", "dataset_001", "mps_stub", "yolo11s-seg.pt")
    assert submitted[0].compute.backend == "mps"  # best supported accelerator
    assert submitted[0].compute.device_indices is None  # device index is cuda-only


def test_train_model_defaults_to_cpu_without_accelerator(tmp_path, monkeypatch):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: False)
    submitted = []
    monkeypatch.setattr(
        "detail.api.experiment.LocalExecutor.submit",
        lambda self, job: submitted.append(job) or job.model_id,
    )
    # a cuda-capable trainer with no accelerator present must still fall back to cpu
    with _registered("cuda_stub", CudaStubTrainer):
        Experiment(proj).train_model("m", "dataset_001", "cuda_stub", "yolo11s-seg.pt")
    assert submitted[0].compute.backend == "cpu"


# --- best-epoch metrics (P1c) ------------------------------------------------ #
def test_final_metrics_reports_best_and_final_epoch(tmp_path):
    from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer

    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50(M)\n"
        "1,0.30,0.20\n"
        "2,0.70,0.60\n"  # best epoch (max box mAP50)
        "3,0.50,0.40\n",  # final epoch — worse than best
        encoding="utf-8",
    )
    m = UltralyticsYOLOTrainer().final_metrics(tmp_path)
    assert m["final_epoch"] == 3 and m["box_map50"] == 0.50
    assert m["best_epoch"] == 2
    assert m["best_box_map50"] == 0.70 and m["best_mask_map50"] == 0.60


def test_final_metrics_ignores_malformed_cells(tmp_path):
    from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer

    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50(M)\n"
        "1,0.30,0.20\n"
        "2,nan(gpu),0.55\n",  # a malformed cell must not crash metric extraction (A15)
        encoding="utf-8",
    )
    m = UltralyticsYOLOTrainer().final_metrics(tmp_path)  # must not raise
    assert m["final_epoch"] == 2 and m["mask_map50"] == 0.55
    assert "box_map50" not in m  # malformed box cell skipped, not fatal
    assert m["best_epoch"] == 1 and m["best_box_map50"] == 0.30


# --- resume_training: stale cancel marker + validation branches (A1, A16, D7) --- #
def test_resume_training_clears_stale_cancel_marker(tmp_path, monkeypatch):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"prior")
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "status": "cancelled", "trainer_id": "stub_trainer"}
    )
    rundir.init_run_dir(mdir, _job())
    rundir.request_cancel(mdir)  # the marker a cancelled run leaves behind
    assert rundir.cancel_requested(mdir)  # precondition
    monkeypatch.setattr(
        "detail.api.experiment.LocalExecutor.submit", lambda self, job: job.model_id
    )
    with _registered_stub():
        Experiment(proj).resume_training("model_001")
    assert not rundir.cancel_requested(mdir)  # cleared → resumed worker won't re-cancel


def test_resume_training_rejects_running_model(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "running"})
    with pytest.raises(ValueError, match="can be resumed"):
        Experiment(proj).resume_training("model_001")


def test_resume_training_rejects_unknown_trainer(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"prior")
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "status": "failed", "trainer_id": "gone"}
    )
    rundir.init_run_dir(mdir, _job(trainer_id="gone"))
    with pytest.raises(ValueError, match="unknown trainer"):
        Experiment(proj).resume_training("model_001")


def test_resume_training_missing_job_json(tmp_path):
    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "status": "failed", "trainer_id": "stub_trainer"}
    )  # no job.json written
    with pytest.raises(ValueError, match="no recorded training job"):
        Experiment(proj).resume_training("model_001")


# --- the series beside the summary ------------------------------------------- #
# `final_metrics` answers *how good is this model*; a user deciding whether to stop training is
# asking *is it still getting better*, which no single number can answer. The two read one file
# and were one method for as long as nobody asked the second question.


def _results_csv(model_dir, rows):
    run = model_dir / "run"
    run.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,metrics/mAP50(B),metrics/mAP50(M)"]
    lines += [f"{e},{b},{m}" for e, b, m in rows]
    (run / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_metric_history_is_every_epoch_oldest_first(tmp_path):
    from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer

    _results_csv(tmp_path, [(1, 0.40, 0.35), (2, 0.61, 0.55), (3, 0.58, 0.52)])

    history = UltralyticsYOLOTrainer().metric_history(tmp_path)

    assert [row["epoch"] for row in history] == [1, 2, 3], history
    assert history[1]["box_map50"] == pytest.approx(0.61)
    assert history[1]["mask_map50"] == pytest.approx(0.55)


def test_metric_history_and_final_metrics_agree_on_the_best_epoch(tmp_path):
    """They read one file, so they must not disagree about which epoch won — the failure that
    would put one number in the caption and another in the record."""
    from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer

    _results_csv(tmp_path, [(1, 0.40, 0.35), (2, 0.91, 0.85), (3, 0.58, 0.52)])
    trainer = UltralyticsYOLOTrainer()

    summary = trainer.final_metrics(tmp_path)
    history = trainer.metric_history(tmp_path)
    peak = max(history, key=lambda row: row["box_map50"])

    assert summary["best_epoch"] == peak["epoch"] == 2
    assert summary["best_box_map50"] == pytest.approx(peak["box_map50"])
    assert summary["final_epoch"] == 3, "the summary's final is still the last row"


def test_a_half_written_tail_row_is_skipped_not_plotted_at_zero(tmp_path):
    """This file is appended to while the run is live and we do not own its writer, so a torn
    last line is the ordinary case rather than a corruption. A row whose epoch does not parse
    must be dropped — plotted at epoch 0 it would drag the curve back to the origin on roughly
    every other tick."""
    from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer

    _results_csv(tmp_path, [(1, 0.40, 0.35), (2, 0.61, 0.55)])
    results = tmp_path / "run" / "results.csv"
    with open(results, "a", encoding="utf-8") as f:
        f.write(",0.63,")  # the epoch column never landed

    history = UltralyticsYOLOTrainer().metric_history(tmp_path)

    assert [row["epoch"] for row in history] == [1, 2], history


def test_no_results_file_is_an_empty_history_not_an_error(tmp_path):
    from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer

    assert UltralyticsYOLOTrainer().metric_history(tmp_path) == []


def test_every_trainer_must_supply_a_history(tmp_path):
    """Abstract, not defaulted, and that is the decision.

    A default returning `[]` spells "as before" — the exact shape that left the Activity band
    dead on five pages — and a trainer added later would silently draw no curve with nothing
    failing. Making it abstract costs four test stubs and buys the guarantee.
    """
    import inspect

    from detail.core.protocols import Trainer

    assert (
        "metric_history" in Trainer.__abstractmethods__
    ), "a new trainer can now be written with no per-epoch history and nothing will say so"
    assert inspect.isabstract(Trainer)


def test_the_api_returns_an_empty_history_when_the_trainer_is_not_installed(tmp_path):
    """Reading a finished run's record must not require the backend that produced it.

    A model trained on another machine and imported here has no Ultralytics behind it, and the
    page still has to open. An empty curve is the honest rendering of a history we cannot read;
    an exception would take the whole model page with it.
    """
    from detail.api.experiment import Experiment
    from detail.storage import models as models_store

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "complete", "trainer_id": "not_installed_here"},
    )

    assert exp.model_metric_history("model_001") == []


def test_the_api_reads_the_history_through_the_record_s_trainer(tmp_path):
    from detail.api.experiment import Experiment
    from detail.storage import models as models_store

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "running", "trainer_id": "ultralytics_yolo"},
    )
    _results_csv(mdir, [(1, 0.40, 0.35), (2, 0.61, 0.55)])

    history = exp.model_metric_history("model_001")

    assert [row["epoch"] for row in history] == [1, 2], history
