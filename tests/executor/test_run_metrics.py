"""run_metrics.completion_fields — assembling a job's timing/compute/cost from its
job.json envelope + (for a remote job) its target's configured rate."""

import pytest

from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    JobStatus,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.executor import rundir, run_metrics
from detail.storage import locations, remote_targets as remote_targets_store


@pytest.fixture(autouse=True)
def _isolated_remote_targets(tmp_path, monkeypatch):
    config_dir = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets_store, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(
        remote_targets_store, "REMOTE_TARGETS_PATH", config_dir / "remote_targets.yaml"
    )


def _job(compute):
    return DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="sam3",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=compute,
        created_at="2026-01-01T00:00:00Z",
    )


def _completed_run_dir(tmp_path, compute, seconds_running=3600):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rundir.init_run_dir(run_dir, _job(compute))
    rundir.set_status(run_dir, JobStatus.RUNNING)
    env = rundir.read_envelope(run_dir)
    # Backdate started_at so duration is exact and test-independent of real wall time.
    from datetime import datetime, timedelta, timezone

    started = datetime.now(timezone.utc) - timedelta(seconds=seconds_running)
    env["started_at"] = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    rundir._atomic_write(rundir._job_path(run_dir), env)
    rundir.set_status(run_dir, JobStatus.COMPLETE)
    return run_dir


def test_completion_fields_local_run_has_no_cost_key(tmp_path):
    run_dir = _completed_run_dir(tmp_path, ComputeTarget("cuda", None, None, None))
    fields = run_metrics.completion_fields(run_dir, {"backend": "cuda"})
    assert fields["duration_s"] == pytest.approx(3600, abs=2)
    assert "estimated_cost_usd" not in fields


def test_completion_fields_remote_run_with_a_configured_rate_gets_a_cost(tmp_path):
    remote_targets_store.register_remote_target(
        RemoteTargetConfig(
            name="tillicum",
            host="tillicum",
            remote_workdir="/scratch/x",
            gpus_per_job=2,
            cost_per_gpu_hour=0.50,
        )
    )
    compute = ComputeTarget("remote", None, None, None, remote_target="tillicum")
    run_dir = _completed_run_dir(tmp_path, compute)
    fields = run_metrics.completion_fields(run_dir, compute.to_dict())
    # 2 gpus * 1 hour * $0.50/gpu-hour = $1.00
    assert fields["estimated_cost_usd"] == pytest.approx(1.0, rel=0.01)


def test_completion_fields_remote_run_with_no_configured_rate_has_no_cost(tmp_path):
    remote_targets_store.register_remote_target(
        RemoteTargetConfig(
            name="tillicum", host="tillicum", remote_workdir="/scratch/x"
        )
    )
    compute = ComputeTarget("remote", None, None, None, remote_target="tillicum")
    run_dir = _completed_run_dir(tmp_path, compute)
    fields = run_metrics.completion_fields(run_dir, compute.to_dict())
    assert "estimated_cost_usd" not in fields
    assert fields["duration_s"] is not None


def test_completion_fields_remote_run_whose_target_no_longer_exists_has_no_cost(
    tmp_path,
):
    """A run's target may be removed/renamed after it completed — the cost estimate
    is simply unavailable, not a crash."""
    compute = ComputeTarget("remote", None, None, None, remote_target="gone")
    run_dir = _completed_run_dir(tmp_path, compute)
    fields = run_metrics.completion_fields(run_dir, compute.to_dict())
    assert "estimated_cost_usd" not in fields
    assert fields["duration_s"] is not None


def test_completion_fields_missing_run_dir_returns_empty_dict_not_a_raise(tmp_path):
    assert run_metrics.completion_fields(tmp_path / "nope", None) == {}


def test_completion_fields_corrupt_envelope_returns_empty_dict_not_a_raise(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "job.json").write_text("{not json", encoding="utf-8")
    assert run_metrics.completion_fields(run_dir, None) == {}
