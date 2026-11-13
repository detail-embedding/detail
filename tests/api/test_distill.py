"""Tests for the durable distill run — label → train (§12)."""

import json
import types

import pytest

from detail.api.experiment import DistillOrchestrator, Experiment, _stage_metrics
from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import (
    ComputeTarget,
    LabelingClass,
    LabelingProgressEvent,
    TrainingProgressEvent,
)
from detail.core.registries import TRAINERS
from detail.core.utils import default_compute
from detail.storage import distill_runs as distill_store
from detail.storage import remote_targets


# --- fakes / helpers -------------------------------------------------------- #
class _FakeRun:
    """Stand-in for the DatasetRun / ModelRun handle the orchestrator drives."""

    def __init__(self, id, events, status="complete", error=None):
        self.id = id
        self._events = events
        self.status = status
        self.error = error

    def progress(self):
        return iter(self._events)

    def wait(self):
        return self


class _FakeTrainer:
    supported_backends = ["cpu"]

    def is_ready(self):
        return True

    def validate_hyperparams(self, hyperparams):
        return dict(hyperparams)


def _compute():
    return default_compute(["cpu"]).to_dict()


def _config(**over):
    cfg = {
        "name": "m",
        "base_model": "yolo11s-seg.pt",
        "trainer_id": "faketrainer",
        "classes": [{"class_id": 0, "name": "mouse", "prompt": "a mouse"}],
        "source_recording_ids": ["recording_001"],
        "labeling": {
            "model_id": "sam3",
            "model_variant": None,
            "n_frames": 4,
            "sampling": "stratified",
            "logit_threshold": 0.5,
            "min_contour_area": 100,
            "val_split": 0.15,
        },
        "training": {"epochs": 5, "batch_size": 2, "imgsz": 64, "patience": 5},
        "label_compute": _compute(),
        "train_compute": _compute(),
    }
    cfg.update(over)
    return cfg


def _reserve(
    experiment, run_id="distill_001", stages=None, status="running", resume=False
):
    record = {
        "id": run_id,
        "name": "m",
        "status": status,
        "created_at": "2026-01-01T00:00:00Z",
        "config": _config(),
        "stages": stages
        or {
            "label": {"status": "pending", "dataset_id": None},
            "train": {"status": "pending", "model_id": None},
        },
        "resume": resume,
    }
    distill_store.write_run(experiment.dir, run_id, record)
    return record


def _write_job_json(run_dir, status, pid=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": 1, "status": status, "job": {}, "error": None}
    if pid is not None:
        envelope["pid"] = pid
    (run_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")


# --- up-front validation (fail-fast, no models needed) ---------------------- #
def test_distill_requires_a_class(tmp_path):
    proj = Experiment.create("p", tmp_path)
    with pytest.raises(ValueError, match="at least one class"):
        proj.distill("m", classes=[], n_frames=4)


def test_distill_rejects_duplicate_prompts(tmp_path):
    proj = Experiment.create("p", tmp_path)
    classes = [LabelingClass(0, "a", "same"), LabelingClass(1, "b", "same")]
    with pytest.raises(ValueError, match="share a text prompt"):
        proj.distill("m", classes=classes, n_frames=4)


def test_distill_rejects_non_prompt_capable_label_model(tmp_path):
    proj = Experiment.create("p", tmp_path)
    with pytest.raises(ValueError, match="cannot label"):
        proj.distill(
            "m",
            classes=[LabelingClass(0, "mouse", "a mouse")],
            n_frames=4,
            label_model_id="not_a_prompt_model",
        )


# --- reserve + spawn (durable-run glue; validation stubbed, no models) ------ #
def _pass_validation(proj, monkeypatch):
    """Stub the model/trainer/frame gates so ``distill`` reaches reserve+spawn."""
    proj.create_recording("recording_001")
    import detail.storage.recording as rec_store

    monkeypatch.setattr("detail.api.models.prompt_capable_models", lambda: ["sam3"])
    monkeypatch.setattr("detail.api.models.supported_backends", lambda mid: ["cpu"])
    monkeypatch.setattr(
        proj,
        "check_model_access",
        lambda mid: types.SimpleNamespace(ready=True, message="ok"),
    )
    monkeypatch.setitem(TRAINERS, "faketrainer", _FakeTrainer)
    monkeypatch.setattr(
        rec_store,
        "read_recording",
        lambda d: {"name": "recording_001", "video": {"files": ["video/part_001.mp4"]}},
    )
    monkeypatch.setattr(
        rec_store, "load_active_timestamps", lambda d, data, i: list(range(100))
    )


def test_distill_reserves_record_then_spawns_the_distill_coordinator(
    tmp_path, monkeypatch
):
    proj = Experiment.create("p", tmp_path)
    _pass_validation(proj, monkeypatch)
    captured = {}

    def _fake_spawn(run_dir, runner_modules, entrypoint="detail.executor.coordinator"):
        captured["entrypoint"] = entrypoint
        return 4321

    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", _fake_spawn)
    proj.distill(
        "m",
        classes=[LabelingClass(0, "mouse", "a mouse")],
        n_frames=2,
        source_recording_ids=["recording_001"],
        trainer_id="faketrainer",
    )
    assert captured["entrypoint"] == "detail.executor.distill_coordinator"
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "running"
    assert rec["stages"]["label"]["status"] == "pending"
    assert rec["stages"]["train"]["status"] == "pending"
    assert rec["config"]["classes"][0]["prompt"] == "a mouse"


def _register_remote_target(monkeypatch, tmp_path, name="tillicum"):
    cfg_dir = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg_dir / "remote_targets.yaml"
    )
    remote_targets.register_remote_target(
        RemoteTargetConfig(name=name, host=name, remote_workdir="/scratch/x")
    )


def test_distill_remote_label_compute_skips_the_local_model_access_check(
    tmp_path, monkeypatch
):
    """The label model is not accessible *here*, but a remote job never loads it
    here — the shepherd's own readiness probe checks it on the cluster before the
    sbatch submission. distill() validates this union up front (Tenet 10), and its
    own check_model_access call was unconditional even though label_dataset's
    identical call had already been fixed to skip for remote — found by
    inspection, same class."""
    proj = Experiment.create("p", tmp_path)
    _pass_validation(proj, monkeypatch)
    monkeypatch.setattr(
        proj,
        "check_model_access",
        lambda mid: types.SimpleNamespace(ready=False, message="not installed"),
    )
    # A cuda-capable model, so effective_backend("remote") -> "cuda" is supported —
    # isolating the local-access skip from the (separately tested) backend check.
    monkeypatch.setattr(
        "detail.api.models.supported_backends", lambda mid: ["cpu", "cuda"]
    )
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    monkeypatch.setattr(
        "detail.api.experiment.spawn_coordinator", lambda run_dir, mods, **k: 4321
    )
    proj.distill(
        "m",
        classes=[LabelingClass(0, "mouse", "a mouse")],
        n_frames=2,
        source_recording_ids=["recording_001"],
        trainer_id="faketrainer",
        label_compute=ComputeTarget(
            "remote", None, None, None, remote_target="tillicum"
        ),
    )
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "running"
    assert rec["config"]["label_compute"]["backend"] == "remote"


def test_distill_remote_train_compute_skips_the_local_trainer_readiness_check(
    tmp_path, monkeypatch
):
    """Same class, the training side: the trainer framework is not importable
    *here*, but training happens entirely on the cluster for a remote job."""

    class _NotReadyTrainer(_FakeTrainer):
        # cuda-capable, so effective_backend("remote") -> "cuda" is supported —
        # isolating the local-readiness skip from the (separately tested) backend
        # check.
        supported_backends = ["cpu", "cuda"]

        def is_ready(self):
            return False

    proj = Experiment.create("p", tmp_path)
    _pass_validation(proj, monkeypatch)
    monkeypatch.setitem(TRAINERS, "faketrainer", _NotReadyTrainer)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    monkeypatch.setattr(
        "detail.api.experiment.spawn_coordinator", lambda run_dir, mods, **k: 4321
    )
    proj.distill(
        "m",
        classes=[LabelingClass(0, "mouse", "a mouse")],
        n_frames=2,
        source_recording_ids=["recording_001"],
        trainer_id="faketrainer",
        train_compute=ComputeTarget(
            "remote", None, None, None, remote_target="tillicum"
        ),
    )
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "running"
    assert rec["config"]["train_compute"]["backend"] == "remote"
    assert distill_store.read_coordinator_pid(proj.dir, "distill_001") == 4321


# --- the opt-in background video upload (stage_videos) ------------------------------------- #
def _distill_kwargs():
    return dict(
        classes=[LabelingClass(0, "mouse", "a mouse")],
        n_frames=2,
        source_recording_ids=["recording_001"],
        trainer_id="faketrainer",
    )


def _remote(name="tillicum"):
    return ComputeTarget("remote", None, None, None, remote_target=name)


def _stageable(proj, monkeypatch, tmp_path):
    """Validation stubbed, a cuda-capable label model and trainer, and a registered target."""
    _pass_validation(proj, monkeypatch)
    monkeypatch.setattr(
        "detail.api.models.supported_backends", lambda mid: ["cpu", "cuda"]
    )

    class _CudaTrainer(_FakeTrainer):
        supported_backends = ["cpu", "cuda"]

    monkeypatch.setitem(TRAINERS, "faketrainer", _CudaTrainer)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    _register_remote_target(monkeypatch, tmp_path, name="hyak")
    monkeypatch.setattr(
        "detail.api.experiment.spawn_coordinator", lambda run_dir, mods, **k: 4321
    )


def test_stage_videos_needs_a_remote_stage_to_upload_to(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    _pass_validation(proj, monkeypatch)

    with pytest.raises(ValueError, match="neither the labeling nor the training"):
        proj.distill("m", stage_videos=True, **_distill_kwargs())

    assert distill_store.list_runs(proj.dir) == [], "refused before reserving a run"


@pytest.mark.parametrize(
    "label, train, staged_to",
    [
        (
            _remote("hyak"),
            _remote("tillicum"),
            "tillicum",
        ),  # detection follows training
        (_remote("hyak"), None, "hyak"),  # training local: the labeling target
        (None, _remote("tillicum"), "tillicum"),
    ],
)
def test_the_upload_goes_where_the_trained_model_will_run(
    tmp_path, monkeypatch, label, train, staged_to
):
    proj = Experiment.create("p", tmp_path)
    _stageable(proj, monkeypatch, tmp_path)

    proj.distill(
        "m",
        label_compute=label,
        train_compute=train,
        stage_videos=True,
        **_distill_kwargs(),
    )

    config = distill_store.read_run(proj.dir, "distill_001")["config"]
    assert config["stage_videos"] is True
    assert config["stage_video_target"] == staged_to


def test_a_distill_stages_nothing_unless_asked(tmp_path, monkeypatch):
    """100+ GB of the cluster's storage is not a default."""
    proj = Experiment.create("p", tmp_path)
    _stageable(proj, monkeypatch, tmp_path)

    proj.distill("m", train_compute=_remote(), **_distill_kwargs())

    config = distill_store.read_run(proj.dir, "distill_001")["config"]
    assert config["stage_videos"] is False and config["stage_video_target"] is None


def test_the_orchestrator_starts_the_upload_before_labeling(tmp_path, monkeypatch):
    from detail.executor import video_staging

    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    record["config"].update(stage_videos=True, stage_video_target="tillicum")
    order = []
    monkeypatch.setattr(
        video_staging, "start", lambda *a: order.append(("stage", a)) or {}
    )
    monkeypatch.setattr(
        proj,
        "label_dataset",
        lambda **kw: order.append(("label", None))
        or _FakeRun("dataset_001", [], status="failed", error="stop here"),
    )

    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()

    assert [step for step, _ in order] == ["stage", "label"]
    assert order[0][1] == (proj.dir, "tillicum", ["recording_001"])


def test_the_orchestrator_stages_nothing_when_not_asked(tmp_path, monkeypatch):
    from detail.executor import video_staging

    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    monkeypatch.setattr(
        video_staging, "start", lambda *a: pytest.fail("staged without being asked")
    )
    monkeypatch.setattr(
        proj,
        "label_dataset",
        lambda **kw: _FakeRun("dataset_001", [], status="failed", error="stop"),
    )

    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()


def test_an_upload_that_cannot_start_never_fails_the_distill(
    tmp_path, monkeypatch, capsys
):
    """It is an optimisation: the detection run uploads the videos itself if it has to."""
    from detail.executor import video_staging

    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    record["config"].update(stage_videos=True, stage_video_target="tillicum")

    def cannot_spawn(*_a):
        raise OSError("too many open files")

    monkeypatch.setattr(video_staging, "start", cannot_spawn)
    monkeypatch.setattr(
        proj,
        "label_dataset",
        lambda **kw: _FakeRun("dataset_001", [], status="failed", error="stop here"),
    )

    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()

    rec = distill_store.read_run(proj.dir, "distill_001")
    assert (
        "labeling failed" in rec["error"]
    ), "reached labeling: the upload did not stop it"
    assert "could not start the background video upload" in capsys.readouterr().out


def test_video_staging_reports_each_upload_and_marks_a_dead_one_failed(
    tmp_path, monkeypatch
):
    from detail.executor import local, video_staging

    proj = Experiment.create("p", tmp_path)
    live = video_staging._initial_state("tillicum", ["recording_001"])
    live.update(pid=1, files_done=3, files_total=8)
    gone = video_staging._initial_state("hyak", ["recording_001"])
    gone.update(pid=2)
    video_staging._write_state(proj.dir, "tillicum", live)
    video_staging._write_state(proj.dir, "hyak", gone)
    monkeypatch.setattr(local, "coordinator_alive", lambda root, pid: pid == 1)

    states = {s["target"]: s for s in proj.video_staging()}

    assert states["tillicum"]["status"] == "running"
    assert states["tillicum"]["files_done"] == 3
    assert (
        states["hyak"]["status"] == "failed"
    ), "a record cannot say running of a dead process"
    assert "no longer running" in states["hyak"]["error"]


def _staging_launches(monkeypatch):
    """``video_staging.start`` replaced by a recorder, so no process is spawned."""
    from detail.executor import video_staging

    launched = []
    monkeypatch.setattr(
        video_staging,
        "start",
        lambda root, target, ids: launched.append((target, list(ids)))
        or {"target": target, "status": "running", "pid": 1},
    )
    return launched


def test_start_video_staging_defaults_to_every_recording(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    proj.create_recording("recording_002")
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    launched = _staging_launches(monkeypatch)

    state = proj.start_video_staging("tillicum")

    assert launched == [("tillicum", ["recording_001", "recording_002"])]
    assert state["status"] == "running"


def test_a_restart_takes_up_the_recordings_of_the_last_upload(tmp_path, monkeypatch):
    """Resuming a stopped upload means *that* upload, not whatever the experiment holds now."""
    from detail.executor import video_staging

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    proj.create_recording("recording_002")
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    previous = video_staging._initial_state("tillicum", ["recording_002"])
    previous["status"] = "cancelled"
    video_staging._write_state(proj.dir, "tillicum", previous)
    launched = _staging_launches(monkeypatch)

    proj.start_video_staging("tillicum")
    proj.start_video_staging("tillicum", ["recording_001"])

    assert launched == [
        ("tillicum", ["recording_002"]),
        ("tillicum", ["recording_001"]),
    ], "explicit recordings win"


def test_start_video_staging_refuses_an_unregistered_target(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    launched = _staging_launches(monkeypatch)

    with pytest.raises(ValueError, match="no remote target registered as 'nope'"):
        proj.start_video_staging("nope")

    assert launched == []


def test_start_video_staging_refuses_an_unknown_recording(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    _register_remote_target(monkeypatch, tmp_path, name="tillicum")
    launched = _staging_launches(monkeypatch)

    with pytest.raises(FileNotFoundError, match="recording_404"):
        proj.start_video_staging("tillicum", ["recording_404"])

    assert launched == []


def test_stop_video_staging_stops_one_target_or_all(tmp_path, monkeypatch):
    from detail.executor import local, video_staging

    proj = Experiment.create("p", tmp_path)
    for name, pid in (("tillicum", 1), ("hyak", 2)):
        state = video_staging._initial_state(name, ["recording_001"])
        state.update(pid=pid)
        video_staging._write_state(proj.dir, name, state)
    monkeypatch.setattr(local, "coordinator_alive", lambda root, pid: True)
    monkeypatch.setattr(local, "kill_coordinator", lambda root, pid: None)

    assert proj.stop_video_staging("hyak") == ["hyak"]
    assert proj.stop_video_staging() == ["tillicum"], "hyak was already stopped"
    assert {s["status"] for s in proj.video_staging()} == {"cancelled"}


def test_distill_spawn_failure_marks_run_failed(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    _pass_validation(proj, monkeypatch)

    def _boom(*_a, **_k):
        raise OSError("cannot fork")

    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", _boom)
    with pytest.raises(OSError):
        proj.distill(
            "m",
            classes=[LabelingClass(0, "mouse", "a mouse")],
            n_frames=2,
            source_recording_ids=["recording_001"],
            trainer_id="faketrainer",
        )
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "failed"
    assert "coordinator failed to launch" in (rec.get("error") or "")


# --- orchestrator: fold + phase order + failure + resume -------------------- #
def test_orchestrator_folds_progress_and_advances_phases(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    label_events = [
        LabelingProgressEvent(1, 4, 0, 1, 0.1, 5.0, "labeling frames"),
        LabelingProgressEvent(4, 4, 1, 1, 0.2, 0.0, None),
    ]
    train_events = [
        TrainingProgressEvent(1, 5, {}, 0.1, 4.0, None),
        TrainingProgressEvent(5, 5, {"box_map50": 0.9}, 0.5, 0.0, None),
    ]
    monkeypatch.setattr(
        proj, "label_dataset", lambda **kw: _FakeRun("dataset_001", label_events)
    )
    monkeypatch.setattr(
        proj, "train_model", lambda **kw: _FakeRun("model_001", train_events)
    )

    emitted = []
    DistillOrchestrator(proj, "distill_001", record, emitted.append).run()

    # Phases never interleave: stage_index is non-decreasing (labeling then training).
    assert [e.stage_index for e in emitted] == sorted(e.stage_index for e in emitted)
    messages = [e.message for e in emitted]
    assert "starting labeling" in messages  # forced phase-boundary emit
    assert "starting training" in messages
    assert messages.index("starting labeling") < messages.index("starting training")
    assert len(set(messages)) >= 3  # not frozen on one label
    # a real intermediate fraction with an ETA
    assert any(0.0 < e.overall_fraction < 1.0 and e.eta_s is not None for e in emitted)
    assert emitted[-1].overall_fraction == pytest.approx(1.0)

    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["label"] == {"status": "complete", "dataset_id": "dataset_001"}
    assert rec["stages"]["train"] == {"status": "complete", "model_id": "model_001"}


def test_orchestrator_fails_the_run_when_a_stage_fails(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    trained = []
    monkeypatch.setattr(
        proj,
        "label_dataset",
        lambda **kw: _FakeRun("dataset_001", [], status="failed", error="boom"),
    )
    monkeypatch.setattr(proj, "train_model", lambda **kw: trained.append(kw))

    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()

    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "failed"
    assert "labeling failed" in rec["error"]
    assert not trained  # training never started


def test_orchestrator_resume_skips_a_complete_dataset(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_007"},
            "train": {"status": "pending", "model_id": None},
        },
        resume=True,
    )

    def _label(**kw):
        raise AssertionError("labeling should be skipped on resume")

    train_calls = []

    def _train(**kw):
        train_calls.append(kw)
        return _FakeRun("model_002", [TrainingProgressEvent(5, 5, {}, 0.1, 0.0, None)])

    monkeypatch.setattr(proj, "label_dataset", _label)
    monkeypatch.setattr(proj, "train_model", _train)

    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()

    assert train_calls and train_calls[0]["dataset_id"] == "dataset_007"
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["train"]["model_id"] == "model_002"


# --- reconcile on open (SIGKILL backstop) ----------------------------------- #
def test_reconcile_distill_adopts_terminal_stage_status(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "running", "model_id": "model_001"},
        },
    )
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    _write_job_json(proj.dir / "models" / "model_001", "complete")
    proj.reconcile_distill_runs()
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["train"]["status"] == "complete"


def test_reconcile_distill_fails_dead_orphan(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    monkeypatch.setattr("detail.api.experiment.worker_alive", lambda _rd: False)
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "running", "model_id": "model_001"},
        },
    )
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    _write_job_json(proj.dir / "models" / "model_001", "running", pid=999999)
    proj.reconcile_distill_runs()
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "failed"
    assert rec["stages"]["train"]["status"] == "failed"
    assert "interrupted" in rec["stages"]["train"]["error"]


def test_reconcile_distill_leaves_live_coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr("detail.api.experiment.coordinator_alive", lambda _rd, _p: True)
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        stages={
            "label": {"status": "running", "dataset_id": "dataset_001"},
            "train": {"status": "pending", "model_id": None},
        },
    )
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    proj.reconcile_distill_runs()
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "running"


# --- resume guards ---------------------------------------------------------- #
def test_resume_distill_refuses_a_running_run(tmp_path, monkeypatch):
    monkeypatch.setattr("detail.api.experiment.coordinator_alive", lambda _rd, _p: True)
    proj = Experiment.create("p", tmp_path)
    _reserve(proj)
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    with pytest.raises(ValueError, match="still running"):
        proj.resume_distill("distill_001")


def test_resume_distill_refuses_a_complete_run(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        status="complete",
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "complete", "model_id": "model_001"},
        },
    )
    with pytest.raises(ValueError, match="already complete"):
        proj.resume_distill("distill_001")


def test_delete_distill_missing_raises(tmp_path):
    proj = Experiment.create("p", tmp_path)
    with pytest.raises(FileNotFoundError):
        proj.delete_distill("distill_999")


# --- the ETA is the running stage's, not the weighted bar's ------------------ #
class _Clock:
    """A scripted monotonic clock, so an ETA assertion is exact instead of timing-dependent."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        """A no-op. **The script owns this clock; a poll must not add to it.**

        This used to do ``self.t += seconds``, with a comment saying it was never called on this
        path. It is called -- just not on a quiet box. `experiment.py`'s poll loops
        (`time.sleep(self._poll)`, three of them) only iterate when a run is not already finished,
        and the fake runs here finish immediately, so on an idle machine the comment was true. Under
        `-n auto` the loops do iterate, each one advancing this clock by a poll interval, and the
        scripted 1238s read as **1281s** -- 43 seconds of invented time (#574).

        That is why both ETA tests flaked at roughly 1 run in 14, and why checking "is sleep called?"
        in isolation said no. The clock is a script: only `_TimedRun.progress` moves it.
        """


class _TimedRun:
    """A fake run whose events advance the clock, the way a real stage's do."""

    def __init__(self, id, steps, clock, status="complete", error=None):
        self.id = id
        self._steps = steps  # (seconds_to_advance, event)
        self._clock = clock
        self.status = status
        self.error = error

    def progress(self):
        for advance, event in self._steps:
            self._clock.t += advance
            yield event

    def wait(self):
        return self


def _distill_at_epoch_14(proj, monkeypatch, *, train_eta):
    """Drive the orchestrator to the reported screenshot: ~370s of labeling, then epoch
    14 of 100 at ~62s per epoch. Returns the emitted events."""
    clock = _Clock()
    monkeypatch.setattr("detail.api.experiment.time", clock)
    record = _reserve(proj)
    label = _TimedRun(
        "dataset_001",
        [(370.0, LabelingProgressEvent(4, 4, 1, 1, 370.0, 0.0, None))],
        clock,
    )
    train = _TimedRun(
        "model_001",
        [(14 * 62.0, TrainingProgressEvent(14, 100, {}, 868.0, train_eta, None))],
        clock,
    )
    monkeypatch.setattr(proj, "label_dataset", lambda **kw: label)
    monkeypatch.setattr(proj, "train_model", lambda **kw: train)
    emitted = []
    DistillOrchestrator(proj, "distill_001", record, emitted.append).run()
    return emitted


def test_training_eta_tracks_the_real_remaining_time(tmp_path, monkeypatch):
    """The reported bug, with its own numbers.

    At epoch 14/100 with ~62s epochs, ~86 epochs (~89 min) remain. The fold used to feed
    the *nominal* stage weights into a time extrapolation — bar 0.45 + 0.55·0.14 = 0.527,
    ETA = 1238·(1−0.527)/0.527 ≈ 1111s — far short of the time actually left.
    """
    proj = Experiment.create("p", tmp_path)
    # The trainer computes an exact per-epoch estimate; the fold used to discard it.
    emitted = _distill_at_epoch_14(proj, monkeypatch, train_eta=5332.0)
    at_epoch = [e for e in emitted if e.phase == "training" and e.inner_fraction > 0]
    assert at_epoch, "no training progress was emitted"
    eta = at_epoch[-1].eta_s

    assert eta == pytest.approx(86 * 62.0, rel=0.02)  # ~89 min, the truth
    assert eta > 4000  # and emphatically not the old ~1111s (~18m31s)
    # The whole-run clock is still reported as elapsed — it is the ETA that must not use it.
    assert at_epoch[-1].elapsed_s == pytest.approx(1238.0)


def test_training_eta_falls_back_to_the_stage_clock_not_the_run_clock(
    tmp_path, monkeypatch
):
    """A runner that supplies no eta_s still must not be timed against the whole run.

    Same instant as above, so the two estimates are directly comparable: over the stage's
    own clock 868·(1−0.14)/0.14 ≈ 5332s (right); over the run clock with the weighted
    fraction 1238·(1−0.527)/0.527 ≈ 1111s (the bug).
    """
    proj = Experiment.create("p", tmp_path)
    emitted = _distill_at_epoch_14(proj, monkeypatch, train_eta=None)
    at_epoch = [e for e in emitted if e.phase == "training" and e.inner_fraction > 0]
    eta = at_epoch[-1].eta_s

    # **Which clock**, asserted as a comparison rather than a tolerance. That is the whole claim,
    # and the two candidates are 4.8x apart, so this can never be a close call:
    over_stage_clock = 868 * (1 - 0.14) / 0.14  # ~5332s, right
    over_run_clock = 1238 * (1 - 0.527) / 0.527  # ~1111s, the bug
    assert abs(eta - over_stage_clock) < abs(eta - over_run_clock), (
        f"eta {eta:.0f}s is nearer the run clock ({over_run_clock:.0f}s) than the stage clock "
        f"({over_stage_clock:.0f}s) — the fold is timing against the whole run again"
    )

    # And a sanity bound. A tight tolerance here was flaky under `-n auto` (#574), for a
    # cause not resolved: the sibling test above pins `elapsed_s` exactly and does not flake,
    # so the clock itself is faithful. An ETA is an estimate; the comparison above is what
    # this test asserts.
    assert eta == pytest.approx(over_stage_clock, rel=0.01)
    assert eta > 4000


def test_the_bar_still_uses_the_stage_weights(tmp_path, monkeypatch):
    """The weighted fold is deliberately kept for the *bar* — only the ETA stopped using
    it. Pinned so a later reader doesn't 'fix' the fraction to match the ETA."""
    proj = Experiment.create("p", tmp_path)
    emitted = _distill_at_epoch_14(proj, monkeypatch, train_eta=5332.0)
    at_epoch = [e for e in emitted if e.phase == "training" and e.inner_fraction > 0]
    assert at_epoch[-1].overall_fraction == pytest.approx(0.45 + 0.55 * 0.14)
    assert at_epoch[-1].inner_fraction == pytest.approx(0.14)


def test_a_resumed_distill_keeps_its_prior_elapsed(tmp_path, monkeypatch):
    """The bar keeps the prior attempt's progress on resume, so the clock has to keep its
    elapsed too — otherwise elapsed and the bar describe different runs (the distill twin of
    ProcessingProgress.recordings_at_start)."""
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    record["resume"] = True
    distill_store.write_progress(
        proj.dir,
        record["id"],
        {
            "phase": "training",
            "stage_index": 1,
            "stage_count": 2,
            "inner_fraction": 0.4,
            "overall_fraction": 0.67,
            "elapsed_s": 3600.0,  # an hour already spent
            "eta_s": None,
            "message": "epoch 40/100",
        },
    )

    clock = _Clock()
    monkeypatch.setattr("detail.api.experiment.time", clock)
    orchestrator = DistillOrchestrator(proj, record["id"], record, lambda e: None)

    # the clock is seeded back by the prior elapsed, so "now" is already an hour in
    assert orchestrator._start == pytest.approx(-3600.0)


def test_a_fresh_distill_starts_its_clock_at_zero(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    clock = _Clock()
    monkeypatch.setattr("detail.api.experiment.time", clock)
    orchestrator = DistillOrchestrator(proj, record["id"], record, lambda e: None)
    assert orchestrator._start == pytest.approx(0.0)  # the identity for a fresh run


# --- stage metrics rollup (§ run location, duration, cost) ------------------ #
def test_stage_metrics_picks_only_the_present_keys():
    record = {
        "id": "dataset_001",
        "compute": {"backend": "cpu"},
        "duration_s": 12.5,
        "estimated_cost_usd": 0.02,
        "status": "complete",  # not one of the rolled-up keys
    }
    assert _stage_metrics(record) == {
        "compute": {"backend": "cpu"},
        "duration_s": 12.5,
        "estimated_cost_usd": 0.02,
    }


def test_stage_metrics_omits_missing_keys_rather_than_inventing_none():
    assert _stage_metrics({"id": "dataset_001", "status": "running"}) == {}


def test_distill_stage_dicts_carry_their_own_compute_and_duration(
    tmp_path, monkeypatch
):
    """The label/train stages roll up whatever set_dataset_status/set_model_status
    already merged into dataset.yaml/model.yaml — read back, not recomputed."""
    from detail.storage import datasets as datasets_store
    from detail.storage import models as models_store

    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    label_events = [LabelingProgressEvent(4, 4, 1, 1, 0.2, 0.0, None)]
    train_events = [TrainingProgressEvent(5, 5, {"box_map50": 0.9}, 0.5, 0.0, None)]

    def _fake_label_dataset(**kw):
        ddir = datasets_store.dataset_dir(proj.dir, "dataset_001")
        ddir.mkdir(parents=True, exist_ok=True)
        datasets_store.write_dataset_yaml(
            ddir,
            {
                "id": "dataset_001",
                "status": "complete",
                "compute": {"backend": "cpu"},
                "duration_s": 42.0,
            },
        )
        return _FakeRun("dataset_001", label_events)

    def _fake_train_model(**kw):
        mdir = models_store.model_dir(proj.dir, "model_001")
        mdir.mkdir(parents=True, exist_ok=True)
        models_store.write_model_yaml(
            mdir,
            {
                "id": "model_001",
                "status": "complete",
                "compute": {"backend": "remote", "remote_target": "tillicum"},
                "duration_s": 3600.0,
                "estimated_cost_usd": 1.5,
            },
        )
        return _FakeRun("model_001", train_events)

    monkeypatch.setattr(proj, "label_dataset", _fake_label_dataset)
    monkeypatch.setattr(proj, "train_model", _fake_train_model)

    emitted = []
    DistillOrchestrator(proj, "distill_001", record, emitted.append).run()

    final = distill_store.read_run(proj.dir, "distill_001")
    assert final["stages"]["label"]["compute"] == {"backend": "cpu"}
    assert final["stages"]["label"]["duration_s"] == 42.0
    assert "estimated_cost_usd" not in final["stages"]["label"]
    assert final["stages"]["train"]["estimated_cost_usd"] == 1.5
    assert final["total_estimated_cost_usd"] == 1.5  # only train stage had a cost
    assert final["duration_s"] is not None and final["duration_s"] >= 0
