from detail.storage import models as mstore
from detail.storage import experiment as experiment_store


def _training_job(compute):
    from detail.core.dataclasses import TrainingJob

    return TrainingJob(
        job_id="model_001",
        model_id="model_001",
        dataset_id="dataset_001",
        trainer_id="ultralytics_yolo",
        base_model="yolo11s-seg.pt",
        hyperparams={},
        compute=compute,
        created_at="2026-01-01T00:00:00Z",
    )


def test_model_yaml_roundtrip_schema_first(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    mdir = mstore.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    mstore.write_model_yaml(mdir, {"id": "model_001", "name": "m", "status": "running"})
    assert (mdir / "model.yaml").read_text(encoding="utf-8-sig").splitlines()[
        0
    ] == "schema_version: 1"
    rec = mstore.read_model_yaml(mdir)
    assert rec["id"] == "model_001"
    mstore.set_model_status(mdir, "complete")
    assert mstore.read_model_yaml(mdir)["status"] == "complete"


def test_set_model_status_records_error(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    mdir = mstore.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    mstore.write_model_yaml(mdir, {"id": "model_001", "status": "running"})
    mstore.set_model_status(mdir, "failed", error="RuntimeError: boom")
    rec = mstore.read_model_yaml(mdir)
    assert rec["status"] == "failed"
    assert rec["error"] == "RuntimeError: boom"


def test_set_model_status_merges_duration_from_the_jobs_own_envelope(tmp_path):
    from detail.core.dataclasses import ComputeTarget, JobStatus
    from detail.executor import rundir

    proj = experiment_store.create_experiment(tmp_path, "p")
    mdir = mstore.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    mstore.write_model_yaml(mdir, {"id": "model_001", "status": "running"})
    job = _training_job(compute=ComputeTarget("cuda", None, None, None))
    rundir.init_run_dir(mdir, job)
    rundir.set_status(mdir, JobStatus.RUNNING)
    rundir.set_status(mdir, JobStatus.COMPLETE)

    mstore.set_model_status(mdir, "complete")

    rec = mstore.read_model_yaml(mdir)
    assert rec["duration_s"] is not None
    assert rec["compute"]["backend"] == "cuda"
    assert "estimated_cost_usd" not in rec


def test_discover_and_next_model_id(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    assert mstore.discover_models(proj) == []
    assert mstore.next_model_id(proj) == "model_001"
    for mid in ("model_001", "model_002"):
        d = mstore.model_dir(proj, mid)
        d.mkdir(parents=True)
        mstore.write_model_yaml(d, {"id": mid})
    assert mstore.discover_models(proj) == ["model_001", "model_002"]
    assert mstore.next_model_id(proj) == "model_003"


def test_delete_model_removes_dir_idempotent(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    d = mstore.model_dir(proj, "model_001")
    d.mkdir(parents=True)
    mstore.write_model_yaml(d, {"id": "model_001"})
    mstore.delete_model(proj, "model_001")
    assert not d.exists()
    mstore.delete_model(proj, "model_001")  # missing dir is a no-op
