import pytest

from detail.storage import analysis_runs as ar
from detail.storage import experiment as experiment_store


def test_analysis_yaml_roundtrip_schema_first(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    adir = ar.analysis_dir(proj, "an_01")
    ar.write_analysis_yaml(
        adir, {"id": "an_01", "module_id": "location_export", "status": "running"}
    )
    text = (adir / "analysis.yaml").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] == "schema_version: 1"  # Tenet 7
    rec = ar.read_analysis_yaml(adir)
    assert rec["schema_version"] == 1
    assert rec["id"] == "an_01" and rec["module_id"] == "location_export"
    ar.set_analysis_status(adir, "complete")
    assert ar.read_analysis_yaml(adir)["status"] == "complete"


def test_set_status_records_error(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    adir = ar.analysis_dir(proj, "an_01")
    ar.write_analysis_yaml(adir, {"id": "an_01", "status": "running"})
    ar.set_analysis_status(adir, "failed", error="ValueError: nope")
    rec = ar.read_analysis_yaml(adir)
    assert rec["status"] == "failed" and rec["error"] == "ValueError: nope"


def test_write_is_atomic_no_tmp_left(tmp_path):
    # Atomic write must leave no sibling *.tmp and never a torn/empty file.
    proj = experiment_store.create_experiment(tmp_path, "p")
    adir = ar.analysis_dir(proj, "an_01")
    ar.write_analysis_yaml(adir, {"id": "an_01", "status": "running"})
    assert not (adir / "analysis.yaml.tmp").exists()
    assert ar.read_analysis_yaml(adir)["id"] == "an_01"  # complete, parseable


def test_read_missing_raises(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    with pytest.raises(FileNotFoundError, match="analysis.yaml"):
        ar.read_analysis_yaml(ar.analysis_dir(proj, "an_99"))


def test_discover_and_next_id(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    assert ar.discover_analyses(proj) == []  # nothing yet
    assert ar.next_analysis_id(proj) == "an_01"
    for rid in ("an_01", "an_02"):
        ar.write_analysis_yaml(ar.analysis_dir(proj, rid), {"id": rid})
    assert ar.discover_analyses(proj) == ["an_01", "an_02"]
    assert ar.next_analysis_id(proj) == "an_03"  # one past the highest


def test_delete_is_idempotent(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    ar.write_analysis_yaml(ar.analysis_dir(proj, "an_01"), {"id": "an_01"})
    ar.delete_analysis(proj, "an_01")
    assert ar.discover_analyses(proj) == []
    ar.delete_analysis(proj, "an_01")  # missing dir → no-op, no raise
