"""The keypoint distill (§12 *Distill keypoints*): resolving the teacher runs, the three-stage
record, the orchestrator's register stage, and resume / cancel over it.

The stage bodies are faked the way ``tests/api/test_distill.py`` fakes them (an external
training run is the one legitimate mock); the teacher detect run, the records and the stores
are real. One real executor run per module: ``test_keypoint_dataset_runs_the_label_stage``.
"""

import pytest

from detail.api import keypoint_distill
from detail.api.experiment import DistillOrchestrator, Experiment
from detail.core.dataclasses import LabelingProgressEvent, TrainingProgressEvent
from detail.core.registries import TRAINERS
from detail.core.skeleton import Skeleton
from detail.core.utils import default_compute
from detail.storage import datasets as datasets_store
from detail.storage import distill_runs as distill_store
from detail.storage import models as models_store
from detail.storage import recording as recording_store
from tests.api.test_distill import _FakeRun, _FakeTrainer
from tests.pipeline.test_keypoint_dataset import (  # noqa: F401 — fixtures
    patch_teacher,
    taught,
)
from tests.pipeline.test_keypoint_detect import (
    tracked,
)  # noqa: F401 — `taught` needs it

SK = Skeleton(names=("nose", "tail_base"), edges=(("nose", "tail_base"),))


def _keypoint_record(
    proj, run_id="distill_001", stages=None, status="running", resume=False
):
    record = {
        "id": run_id,
        "name": "rats",
        "kind": "keypoints",
        "status": status,
        "created_at": "2026-10-08T00:00:00Z",
        "config": {
            "name": "rats",
            "base_model": "yolo11s-pose.pt",
            "trainer_id": "faketrainer",
            "teacher": {
                "kind": "keypoint_detect",
                "detect_runs": {"recording_002": "kd_01"},
                "tracking_runs": {"recording_002": "track_01"},
                "match": {"min_inside": 0.5, "mask_grow": 0.05},
            },
            "keypoint_dataset": {
                "n_frames": 4,
                "val_split": 0.0,
                "score_cutoff": 0.4,
                "exclude_audit_s": 1.0,
                "every_n_frames": 1,
            },
            "training": {"epochs": 2},
            "train_compute": default_compute(["cpu"]).to_dict(),
        },
        "stages": stages
        or {
            "label": {"status": "pending", "dataset_id": None},
            "train": {"status": "pending", "model_id": None},
            "register": {"status": "pending"},
        },
        "resume": resume,
    }
    distill_store.write_run(proj.dir, run_id, record)
    return record


def _student(proj, model_id="model_001", with_weights=True):
    mdir = models_store.model_dir(proj.dir, model_id)
    (mdir / "run" / "weights").mkdir(parents=True, exist_ok=True)
    if with_weights:
        (mdir / "run" / "weights" / "best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": "rats",
            "status": "complete",
            "task": "pose",
            "weights": {"best": "run/weights/best.pt"},
        },
    )
    return mdir


# --- resolving the teacher ---------------------------------------------------------- #
def test_the_only_complete_track_run_is_the_teacher_by_name_or_id(taught):
    it, recording = taught
    proj = it.experiment
    assert keypoint_distill.resolve_detect_runs(proj, ["recording_002"]) == {
        "recording_002": "kd_01"
    }
    name = proj.recording_names()["recording_002"]
    assert keypoint_distill.resolve_detect_runs(proj, {name: "kd_01"}) == {
        "recording_002": "kd_01"
    }
    with pytest.raises(ValueError, match="no complete keypoint detect run 'kd_09'"):
        keypoint_distill.resolve_detect_runs(proj, {"recording_002": "kd_09"})
    with pytest.raises(ValueError, match="at least one keypoint detect run"):
        keypoint_distill.resolve_detect_runs(proj, [])
    with pytest.raises(KeyError):
        keypoint_distill.resolve_detect_runs(proj, ["nope"])


def test_two_complete_runs_must_be_named(taught):
    it, recording = taught
    data = recording_store.read_recording(recording.dir)
    data["keypoint_detect_runs"].append(
        {**data["keypoint_detect_runs"][0], "id": "kd_02"}
    )
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="2 keypoint detect runs .* say which one"):
        keypoint_distill.resolve_detect_runs(it.experiment, ["recording_002"])
    assert keypoint_distill.resolve_detect_runs(
        it.experiment, {"recording_002": "kd_02"}
    ) == {"recording_002": "kd_02"}


RESOLVED = {
    "kind": "keypoint_detect",
    "detect_runs": {"recording_002": "kd_01"},
    "tracking_runs": {"recording_002": "track_01"},
    "match": {"min_inside": 0.5, "mask_grow": 0.05},
}


def test_the_teacher_assigns_to_the_one_tracking_run_on_its_detection_run(taught):
    """§12 (revised 2026-10-21): a detect run's poses belong to no subject, so the teacher
    carries the tracking run they are assigned to — resolved as keypoint track resolves it:
    the one complete tracking run on the detect run's detection run, or the one named;
    several and none named is refused, naming them; never "the first"."""
    it, recording = taught
    proj = it.experiment
    teacher, compute = keypoint_distill.resolve_teacher(proj, ["recording_002"])
    assert teacher == RESOLVED and compute is None
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"][0]["class_name"] = "Intruder"
    data["tracking_runs"].append(
        {
            "id": "track_02",
            "status": "complete",
            "detection_run_id": "det_01",
            "class_name": "Resident",
        }
    )
    data["tracking_runs"].append(
        {"id": "track_03", "status": "complete", "detection_run_id": "det_09"}
    )
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(
        ValueError,
        match=r"2 complete tracking runs .*track_01 \(Intruder\), track_02 \(Resident\);"
        r".*Merge tracks",
    ):
        keypoint_distill.resolve_teacher(proj, ["recording_002"])
    named, _ = keypoint_distill.resolve_teacher(
        proj, ["recording_002"], tracking={"recording_002": "track_02"}
    )
    assert named["tracking_runs"] == {"recording_002": "track_02"}
    with pytest.raises(ValueError, match="no complete tracking run 'track_03' on"):
        keypoint_distill.resolve_teacher(
            proj, ["recording_002"], tracking={"recording_002": "track_03"}
        )
    info = keypoint_distill.teacher_info(proj, "recording_002", "kd_01")
    assert info["detection_run_id"] == "det_01"
    assert info["tracking_run_ids"] == ["track_01", "track_02"]
    data["tracking_runs"] = [data["tracking_runs"][2]]
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="no complete tracking run on detection run"):
        keypoint_distill.resolve_teacher(proj, ["recording_002"])


def test_a_method_without_a_default_cutoff_needs_one(taught):
    it, _recording = taught
    # the fake teacher's ScoreSpec sets no default cut-off
    with pytest.raises(ValueError, match="sets no default score cut-off"):
        keypoint_distill.plan_keypoint_dataset(
            it.experiment, "rats", ["recording_002"], 4
        )
    cfg, teacher, cutoff, _compute = keypoint_distill.plan_keypoint_dataset(
        it.experiment, "rats", ["recording_002"], 4, score_cutoff=0.3
    )
    assert (cfg.n_frames, teacher, cutoff) == (4, RESOLVED, 0.3)
    assert _compute is None, "a track-run teacher runs nothing"
    with pytest.raises(ValueError, match="needs a name"):
        keypoint_distill.plan_keypoint_dataset(it.experiment, " ", ["recording_002"], 4)


def test_keypoint_dataset_runs_the_label_stage(taught):
    """The one real executor run of this module: the dataset the API hands back is the
    runner's, complete, with its pose task and its teachers recorded."""
    it, _recording = taught
    run = it.experiment.keypoint_dataset(
        "rats", ["recording_002"], 2, val_split=0.0, score_cutoff=0.4
    )
    run.wait()
    assert run.status == "complete", run.error
    record = it.experiment.dataset_record(run.id)
    assert record["task"] == "pose" and record["status"] == "complete"
    assert record["source"] == {
        "kind": "keypoint_detect",
        "detect_runs": {"recording_002": "kd_01"},
        "tracking_runs": {"recording_002": "track_01"},
        "match": {"min_inside": 0.5, "mask_grow": 0.05},
        "example_sets": record["source"]["example_sets"],
    }
    assert record["keypoint_dataset"]["score_cutoff"] == 0.4
    assert record["n_train"] == 2


# --- launching the durable run ---------------------------------------------------- #
def test_distill_keypoints_reserves_a_three_stage_record_then_spawns(
    taught, monkeypatch
):
    it, _recording = taught
    proj = it.experiment
    monkeypatch.setitem(TRAINERS, "faketrainer", _FakeTrainer)
    captured = {}

    def _fake_spawn(run_dir, runner_modules, entrypoint="detail.executor.coordinator"):
        captured["entrypoint"] = entrypoint
        return 4321

    monkeypatch.setattr("detail.executor.local.spawn_coordinator", _fake_spawn)
    handle = proj.distill_keypoints(
        "rats", ["recording_002"], 4, score_cutoff=0.4, trainer_id="faketrainer"
    )
    assert handle.id == "distill_001"
    assert captured["entrypoint"] == "detail.executor.distill_coordinator"
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["kind"] == "keypoints" and rec["status"] == "running"
    assert set(rec["stages"]) == {"label", "train", "register"}
    assert all(s["status"] == "pending" for s in rec["stages"].values())
    assert rec["config"]["teacher"] == RESOLVED
    assert rec["config"]["keypoint_dataset"]["score_cutoff"] == 0.4
    assert rec["config"]["base_model"] == keypoint_distill.DEFAULT_POSE_BASE
    assert distill_store.read_coordinator_pid(proj.dir, "distill_001") == 4321


def test_distill_keypoints_refuses_a_base_that_is_not_pose(taught, monkeypatch):
    it, _recording = taught
    monkeypatch.setitem(TRAINERS, "faketrainer", _FakeTrainer)
    with pytest.raises(ValueError, match="not a pose checkpoint"):
        it.experiment.distill_keypoints(
            "rats",
            ["recording_002"],
            4,
            score_cutoff=0.4,
            base_model="yolo11s-seg.pt",
            trainer_id="faketrainer",
        )
    assert not distill_store.list_runs(it.experiment.dir), "refused before reserving"


# --- the orchestrator's third stage --------------------------------------------- #
def test_orchestrator_labels_trains_then_registers(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _keypoint_record(proj)
    label_events = [LabelingProgressEvent(1, 4, 0, 1, 0.1, 5.0, "decoding frames")]
    train_events = [TrainingProgressEvent(2, 2, {"pose_map50": 0.8}, 0.5, 0.0, None)]
    calls = {}

    def _dataset(**kw):
        calls["dataset"] = kw
        return _FakeRun("dataset_001", label_events)

    def _register(experiment, run_id, rec):
        calls["register"] = (run_id, rec["stages"]["train"]["model_id"])
        return {"name": "rats distill_001"}

    monkeypatch.setattr(proj, "keypoint_dataset", _dataset)
    monkeypatch.setattr(
        proj, "train_model", lambda **kw: _FakeRun("model_001", train_events)
    )
    monkeypatch.setattr(keypoint_distill, "register_student", _register)

    emitted = []
    DistillOrchestrator(proj, "distill_001", record, emitted.append).run()

    assert calls["dataset"]["detect_runs"] == {"recording_002": "kd_01"}
    # the tracking run resolved at launch is replayed, never re-resolved
    assert calls["dataset"]["tracking"] == {"recording_002": "track_01"}
    assert calls["dataset"]["score_cutoff"] == 0.4 and calls["dataset"]["n_frames"] == 4
    assert calls["register"] == ("distill_001", "model_001")
    phases = [e.phase for e in emitted]
    assert (
        phases.index("labeling")
        < phases.index("training")
        < phases.index("registering")
    )
    assert {e.stage_count for e in emitted} == {3}
    assert [e.stage_index for e in emitted] == sorted(e.stage_index for e in emitted)
    assert emitted[-1].overall_fraction == pytest.approx(1.0)
    assert emitted[-1].message == "registered rats distill_001"
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["register"] == {"status": "complete", "model_id": "model_001"}


def test_a_failed_register_fails_the_run_and_a_resume_does_not_retrain(
    tmp_path, monkeypatch
):
    proj = Experiment.create("p", tmp_path)
    record = _keypoint_record(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "pending", "model_id": None},
            "register": {"status": "pending"},
        },
        resume=True,
    )
    monkeypatch.setattr(
        proj,
        "keypoint_dataset",
        lambda **kw: pytest.fail("labeling should be reused on resume"),
    )
    monkeypatch.setattr(
        proj,
        "train_model",
        lambda **kw: _FakeRun(
            "model_001", [TrainingProgressEvent(2, 2, {}, 0.1, 0.0, None)]
        ),
    )

    def _boom(experiment, run_id, rec):
        raise ValueError("model 'model_001' has no best weights to register")

    monkeypatch.setattr(keypoint_distill, "register_student", _boom)
    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "failed" and "register failed" in rec["error"]
    assert rec["stages"]["train"] == {"status": "complete", "model_id": "model_001"}
    assert rec["stages"]["register"]["status"] == "failed"

    # A resume keeps the trained student and only registers it.
    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", lambda *a, **k: 1)
    proj.resume_distill("distill_001")
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["resume"] is True and rec["status"] == "running"
    monkeypatch.setattr(
        proj, "train_model", lambda **kw: pytest.fail("a complete model is reused")
    )
    monkeypatch.setattr(
        keypoint_distill, "register_student", lambda e, r, rec: {"name": "rats"}
    )
    emitted = []
    DistillOrchestrator(proj, "distill_001", rec, emitted.append).run()
    assert "reused trained model" in [e.message for e in emitted]
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["register"] == {"status": "complete", "model_id": "model_001"}


def test_resume_refuses_a_registered_run_and_cancel_marks_a_live_register(
    tmp_path, monkeypatch
):
    proj = Experiment.create("p", tmp_path)
    _keypoint_record(
        proj,
        status="complete",
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "complete", "model_id": "model_001"},
            "register": {"status": "complete", "model_id": "model_001"},
        },
    )
    with pytest.raises(ValueError, match="already complete"):
        proj.resume_distill("distill_001")

    _keypoint_record(
        proj,
        run_id="distill_002",
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "complete", "model_id": "model_001"},
            "register": {"status": "running"},
        },
    )
    monkeypatch.setattr("detail.api.experiment.kill_coordinator", lambda *a, **k: None)
    proj.cancel_distill("distill_002")
    rec = distill_store.read_run(proj.dir, "distill_002")
    assert rec["status"] == "cancelled"
    assert rec["stages"]["register"]["status"] == "cancelled"
    assert rec["stages"]["train"]["status"] == "complete", "the student is kept"


# --- register_student ------------------------------------------------------------- #
def test_register_student_promotes_the_trained_model(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _student(proj)
    ddir = datasets_store.dataset_dir(proj.dir, "dataset_001")
    ddir.mkdir(parents=True, exist_ok=True)
    datasets_store.write_dataset_yaml(
        ddir, {"id": "dataset_001", "task": "pose", "skeleton": SK.to_dict()}
    )
    record = _keypoint_record(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "complete", "model_id": "model_001"},
            "register": {"status": "running"},
        },
    )
    model = keypoint_distill.register_student(proj, "distill_001", record)
    on_disk = models_store.read_model_yaml(
        models_store.model_dir(proj.dir, "model_001")
    )
    assert on_disk == model
    assert on_disk["name"] == "rats distill_001"
    assert on_disk["role"] == "keypoints" and on_disk["task"] == "pose"
    assert on_disk["model_family"] == "yolo_pose"
    assert len(on_disk["weights_sha256"]) == 64
    assert on_disk["teacher"] == {
        "kind": "keypoint_detect",
        "distill_run_id": "distill_001",
        "dataset_id": "dataset_001",
        "detect_runs": {"recording_002": "kd_01"},
        "tracking_runs": {"recording_002": "track_01"},
    }
    assert Skeleton.from_dict(on_disk["skeleton"]) == SK
    # idempotent: registering again changes nothing
    assert keypoint_distill.register_student(proj, "distill_001", record) == on_disk


def test_register_student_refuses_a_model_without_weights(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _student(proj, with_weights=False)
    record = _keypoint_record(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "complete", "model_id": "model_001"},
            "register": {"status": "running"},
        },
    )
    with pytest.raises(ValueError, match="no best weights"):
        keypoint_distill.register_student(proj, "distill_001", record)
    record["stages"]["train"]["model_id"] = None
    with pytest.raises(ValueError, match="recorded no model"):
        keypoint_distill.register_student(proj, "distill_001", record)


# --- the patch method's settings reach the distill, as they reach keypoints test ---------- #
def test_the_distill_teacher_takes_and_records_the_methods_settings(patch_teacher):
    """`keypoints test` honoured decode / side rule / crop size and the distill did not, so a
    cut-off judged with them was applied by a teacher labelling without them. The settings are
    resolved into the teacher the dataset records; a detect-run teacher refuses them."""
    it, _recording, set_id, _seen = patch_teacher
    _cfg, teacher, _cutoff, _compute = keypoint_distill.plan_keypoint_dataset(
        it.experiment,
        "rats",
        n_frames=4,
        score_cutoff=0.4,
        example_set=set_id,
        tracking=["recording_002"],
        model_id="fake_learner",
        settings={"decode": "per_point", "crop_px": 96},
    )
    assert teacher["settings"]["decode"] == "per_point"
    assert teacher["settings"]["crop_px"] == 96
    with pytest.raises(ValueError, match="method settings are for the patch-method"):
        keypoint_distill.plan_keypoint_dataset(
            it.experiment,
            "rats",
            {"recording_002": None},
            n_frames=4,
            score_cutoff=0.4,
            settings={"crop_px": 96},
        )


def test_a_distill_patch_teacher_takes_one_set_per_class(taught, monkeypatch):
    """§26 *One example set per detection class*: the class map reaches the patch teacher's
    resolver as it reaches keypoint detect's, and one set and a map at once is refused.
    """
    it, _recording = taught
    handed = []

    def resolve(experiment, example_set, tracking, *rest):
        handed.append((example_set, rest[-1]))
        return {"kind": "model", "model_id": "dinov3_match"}, None

    monkeypatch.setattr(keypoint_distill, "_resolve_patch_teacher", resolve)
    sets = {"Resident": "ex_02", "Intruder": "ex_03"}
    keypoint_distill.plan_keypoint_dataset(
        it.experiment,
        "pair",
        n_frames=4,
        score_cutoff=0.4,
        class_example_sets=sets,
        tracking=["recording_002"],
    )
    assert handed == [(None, sets)]
    with pytest.raises(ValueError, match="give one teacher"):
        keypoint_distill.plan_keypoint_dataset(
            it.experiment,
            "pair",
            n_frames=4,
            score_cutoff=0.4,
            example_set="ex_01",
            class_example_sets=sets,
            tracking=["recording_002"],
        )


def test_a_per_class_teacher_is_recorded_on_the_student(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _student(proj)
    ddir = datasets_store.dataset_dir(proj.dir, "dataset_001")
    ddir.mkdir(parents=True, exist_ok=True)
    datasets_store.write_dataset_yaml(
        ddir, {"id": "dataset_001", "task": "pose", "skeleton": SK.to_dict()}
    )
    record = _keypoint_record(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "complete", "model_id": "model_001"},
            "register": {"status": "running"},
        },
    )
    record["config"]["teacher"] = {
        "kind": "model",
        "model_id": "dinov3_match",
        "example_set": None,
        "example_sets": {
            "Resident": {"id": "ex_02", "digest": "a", "marks": []},
            "Intruder": {"id": "ex_03", "digest": "b", "marks": []},
        },
        "tracking_runs": {"recording_002": "track_03"},
    }
    model = keypoint_distill.register_student(proj, "distill_001", record)
    assert model["teacher"]["example_sets"] == {
        "Resident": {"id": "ex_02", "digest": "a"},
        "Intruder": {"id": "ex_03", "digest": "b"},
    }


@pytest.mark.parametrize(
    "command, method",
    [
        (
            ["keypoints", "distill", "--name", "fast", "--n-frames", "4"],
            "distill_keypoints",
        ),
        (
            ["dataset", "keypoints", "--name", "fast", "--n-frames", "4"],
            "keypoint_dataset",
        ),
    ],
    ids=["keypoints distill", "dataset keypoints"],
)
def test_the_cli_distill_takes_the_same_settings_as_keypoints_test(
    patch_teacher, monkeypatch, command, method
):
    from click.testing import CliRunner

    from detail.cli import main

    it, _recording, set_id, _seen = patch_teacher
    seen = {}
    monkeypatch.setattr(
        Experiment,
        method,
        lambda self, *args, **kwargs: seen.update(kwargs)
        or (_ for _ in ()).throw(ValueError("stop here")),
    )
    base = command + ["--experiment", str(it.experiment.dir), "--score-cutoff", "0.4"]
    patch = [
        "--teacher",
        "patch",
        "--example-set",
        set_id,
        "--tracking-run",
        "recording_002:track_01",
        "--model",
        "fake_learner",
    ]
    result = CliRunner().invoke(
        main, base + patch + ["--crop-px", "96", "--side-rule", "midline"]
    )
    assert "stop here" in result.output, result.output
    assert seen["settings"] == {"crop_px": 96, "side_rule": "midline"}

    seen.clear()
    CliRunner().invoke(main, base + patch)
    assert "settings" not in seen, "nothing typed, nothing sent: the method's defaults"

    result = CliRunner().invoke(
        main,
        base
        + ["--teacher", "detect", "--recording", "recording_002", "--crop-px", "96"],
    )
    assert result.exit_code != 0 and "are for --teacher patch" in result.output


# --- training the student on a remote cluster (#718) ---------------------------- #
class _NotReadyHereTrainer(_FakeTrainer):
    """The training framework is absent on this machine; it trains as cuda on a cluster."""

    supported_backends = ["cpu", "cuda"]

    def is_ready(self):
        return False


def _register_remote_target(monkeypatch, tmp_path, name="tillicum"):
    from detail.configs.remote import RemoteTargetConfig
    from detail.storage import remote_targets

    cfg_dir = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg_dir / "remote_targets.yaml"
    )
    remote_targets.register_remote_target(
        RemoteTargetConfig(name=name, host=name, remote_workdir="/scratch/x")
    )


def _remote(target="tillicum"):
    from detail.core.dataclasses import ComputeTarget

    return ComputeTarget("remote", None, None, None, remote_target=target)


def test_a_remote_student_launches_without_the_trainer_installed_here(
    taught, monkeypatch, tmp_path
):
    """The dataset is cut here, but the student trains entirely on the cluster, so neither
    this machine's training framework nor ``"remote"`` missing from the trainer's backends
    may refuse it — the gap that kept keypoint students off Tillicum."""
    it, _recording = taught
    monkeypatch.setitem(TRAINERS, "faketrainer", _NotReadyHereTrainer)
    _register_remote_target(monkeypatch, tmp_path)
    monkeypatch.setattr("detail.executor.local.spawn_coordinator", lambda *a, **k: 7)
    it.experiment.distill_keypoints(
        "rats",
        ["recording_002"],
        4,
        score_cutoff=0.4,
        trainer_id="faketrainer",
        train_compute=_remote(),
    )
    record = distill_store.read_run(it.experiment.dir, "distill_001")
    assert record["config"]["train_compute"]["backend"] == "remote"
    assert record["config"]["train_compute"]["remote_target"] == "tillicum"


def test_a_remote_student_needs_a_registered_target(taught, monkeypatch, tmp_path):
    it, _recording = taught
    monkeypatch.setitem(TRAINERS, "faketrainer", _NotReadyHereTrainer)
    _register_remote_target(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="hyak"):
        it.experiment.distill_keypoints(
            "rats",
            ["recording_002"],
            4,
            score_cutoff=0.4,
            trainer_id="faketrainer",
            train_compute=_remote("hyak"),
        )
    assert distill_store.list_runs(it.experiment.dir) == []  # nothing reserved


def test_a_local_student_still_needs_the_trainer_installed(taught, monkeypatch):
    from detail.core.dataclasses import ComputeTarget

    it, _recording = taught
    monkeypatch.setitem(TRAINERS, "faketrainer", _NotReadyHereTrainer)
    with pytest.raises(ValueError, match="is not ready"):
        it.experiment.distill_keypoints(
            "rats",
            ["recording_002"],
            4,
            score_cutoff=0.4,
            trainer_id="faketrainer",
            train_compute=ComputeTarget("cpu", None, None, None),
        )
