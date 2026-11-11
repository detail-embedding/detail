"""Tests for experiment-level distill-run records (§12)."""

import pytest

from detail.storage import distill_runs as distill_store
from detail.storage import experiment as experiment_store


def test_next_distill_id_monotonic(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    assert distill_store.next_distill_id(proj) == "distill_001"
    distill_store.write_run(
        proj, "distill_001", {"id": "distill_001", "status": "complete"}
    )
    assert distill_store.next_distill_id(proj) == "distill_002"


def test_write_run_forces_schema_version_first(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    distill_store.write_run(
        proj, "distill_001", {"id": "distill_001", "name": "m", "status": "running"}
    )
    rec = distill_store.read_run(proj, "distill_001")
    assert rec["schema_version"] == 1
    assert rec["name"] == "m"


def test_discover_and_list_runs_sorted(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    assert distill_store.discover_runs(proj) == []
    distill_store.write_run(proj, "distill_002", {"id": "distill_002", "status": "x"})
    distill_store.write_run(proj, "distill_001", {"id": "distill_001", "status": "x"})
    assert distill_store.discover_runs(proj) == ["distill_001", "distill_002"]
    assert [r["id"] for r in distill_store.list_runs(proj)] == [
        "distill_001",
        "distill_002",
    ]


def test_read_run_missing_raises(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    with pytest.raises(FileNotFoundError):
        distill_store.read_run(proj, "distill_999")


def test_delete_run_prunes_only_the_index(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    distill_store.write_run(proj, "distill_001", {"id": "distill_001", "status": "x"})
    distill_store.delete_run(proj, "distill_001")
    assert distill_store.discover_runs(proj) == []
    distill_store.delete_run(proj, "distill_001")  # idempotent


def test_progress_round_trip_and_absent_is_none(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    assert distill_store.read_progress(proj, "distill_001") is None
    distill_store.write_progress(proj, "distill_001", {"phase": "training", "x": 1})
    assert distill_store.read_progress(proj, "distill_001") == {
        "phase": "training",
        "x": 1,
    }


def test_coordinator_pid_round_trip(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    assert distill_store.read_coordinator_pid(proj, "distill_001") is None
    distill_store.set_coordinator_pid(proj, "distill_001", 4321)
    assert distill_store.read_coordinator_pid(proj, "distill_001") == 4321


def test_cancel_sentinel(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "proj")
    assert not distill_store.cancel_requested(proj, "distill_001")
    distill_store.request_cancel(proj, "distill_001")
    assert distill_store.cancel_requested(proj, "distill_001")
    distill_store.clear_cancel(proj, "distill_001")
    assert not distill_store.cancel_requested(proj, "distill_001")
