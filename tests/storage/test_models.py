from detail.storage import models as mstore
from detail.storage import experiment as experiment_store


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
