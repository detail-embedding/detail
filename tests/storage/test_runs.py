"""Tests for experiment-level processing-run records (§15)."""

import pytest

from detail.storage import experiment as experiment_store
from detail.storage import runs as runs_store


def test_next_run_id_monotonic(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    assert runs_store.next_run_id(proj) == "run_001"
    runs_store.write_run(proj, "run_001", {"id": "run_001", "status": "complete"})
    assert runs_store.next_run_id(proj) == "run_002"


def test_write_run_forces_schema_version_first(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    runs_store.write_run(
        proj, "run_001", {"id": "run_001", "module_id": "downsampling", "status": "x"}
    )
    rec = runs_store.read_run(proj, "run_001")
    assert rec["schema_version"] == 1
    assert rec["module_id"] == "downsampling"


def test_discover_and_list_runs_sorted(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    assert runs_store.discover_runs(proj) == []
    runs_store.write_run(proj, "run_002", {"id": "run_002", "status": "complete"})
    runs_store.write_run(proj, "run_001", {"id": "run_001", "status": "complete"})
    assert runs_store.discover_runs(proj) == ["run_001", "run_002"]
    assert [r["id"] for r in runs_store.list_runs(proj)] == ["run_001", "run_002"]


def test_read_run_missing_raises(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    with pytest.raises(FileNotFoundError):
        runs_store.read_run(proj, "run_999")
