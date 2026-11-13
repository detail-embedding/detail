"""A remote job's cost is what it ran for, not what it waited for (§6, § run cost).

**The class: a quantity billed on a clock that starts before the thing it bills.** The estimate
was ``gpus × (completed − started) × rate``, with ``started`` stamped when the shepherd submitted the
job. That was fine while Tillicum had free GPUs, and wrong the day the queue backed up: a recording
that waited five hours and ran for twenty minutes showed five hours and twenty minutes of GPU
time. The cluster worker already stamps its own start and finish, and the shepherd now keeps them
(``worker_started_at`` / ``worker_completed_at``); the cost reads those.

**Three eras of record have to keep meaning what they meant**, so each is a test: a record with
no worker stamps at all (written before this, or a local run) bills ``duration_s`` as it always
did; a record whose worker stamps are present and ``None`` is a job that never reached a GPU and
cost nothing; and one with only a start (the worker was killed) runs until the job ended here.
"""

import pytest

from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import ComputeTarget, JobStatus
from detail.executor import rundir, run_metrics
from detail.storage import remote_targets as remote_targets_store

# the shared fixtures and helpers of the metrics tests: an isolated target store and a run dir
from tests.executor.test_run_metrics import (  # noqa: F401
    _isolated_remote_targets,
    _job,
)

HOUR = 3600


@pytest.fixture
def tillicum():
    remote_targets_store.register_remote_target(
        RemoteTargetConfig(
            name="tillicum",
            host="tillicum",
            remote_workdir="/scratch/x",
            gpus_per_job=2,
            cost_per_gpu_hour=0.50,
        )
    )
    return ComputeTarget("remote", None, None, None, remote_target="tillicum")


def _finished(tmp_path, compute, **stamps):
    """A terminal remote run dir: submitted at 10:00, ended at 15:20 (a 5 h 20 min span), with
    ``stamps`` (the worker's clock) merged into the envelope as the shepherd leaves them.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rundir.init_run_dir(run_dir, _job(compute))
    envelope = rundir.read_envelope(run_dir)
    envelope.update(
        status=JobStatus.COMPLETE.value,
        started_at="2026-10-06T10:00:00Z",
        completed_at="2026-10-06T15:20:00Z",
        **stamps,
    )
    rundir._atomic_write(rundir._job_path(run_dir), envelope)
    return run_dir


def test_a_job_is_billed_for_its_running_time_not_its_queue_time(tmp_path, tillicum):
    run_dir = _finished(
        tmp_path,
        tillicum,
        worker_started_at="2026-10-06T15:00:00Z",  # waited five hours, ran twenty minutes
        worker_completed_at="2026-10-06T15:20:00Z",
    )
    fields = run_metrics.completion_fields(run_dir, tillicum.to_dict())
    assert fields["duration_s"] == pytest.approx(
        5 * HOUR + 20 * 60
    )  # what it took, as before
    assert fields["runtime_s"] == pytest.approx(20 * 60)
    # 2 GPUs x 20 minutes x $0.50/GPU-hour
    assert fields["estimated_cost_usd"] == pytest.approx(2 * (20 / 60) * 0.50)


def test_a_record_with_no_worker_clock_is_billed_as_it_always_was(tmp_path, tillicum):
    """A run from before the worker's clock was kept: the only time it has is submission to the
    end, and billing that is what its cost has always been."""
    run_dir = _finished(tmp_path, tillicum)
    fields = run_metrics.completion_fields(run_dir, tillicum.to_dict())
    assert "runtime_s" not in fields
    assert fields["estimated_cost_usd"] == pytest.approx(2 * (5 + 20 / 60) * 0.50)


def test_a_job_that_never_reached_a_gpu_costs_nothing(tmp_path, tillicum):
    run_dir = _finished(
        tmp_path, tillicum, worker_started_at=None, worker_completed_at=None
    )
    fields = run_metrics.completion_fields(run_dir, tillicum.to_dict())
    assert fields["runtime_s"] == 0.0
    assert fields["estimated_cost_usd"] == 0.0
    assert fields["duration_s"] > 5 * HOUR  # still reports how long it was held


def test_a_worker_killed_after_it_started_runs_until_the_job_ended(tmp_path, tillicum):
    run_dir = _finished(
        tmp_path,
        tillicum,
        worker_started_at="2026-10-06T14:20:00Z",
        worker_completed_at=None,  # never wrote a finish
    )
    fields = run_metrics.completion_fields(run_dir, tillicum.to_dict())
    assert fields["runtime_s"] == pytest.approx(HOUR)  # 14:20 to the job's end at 15:20


def test_a_job_sized_below_its_target_is_billed_for_the_gpus_it_held(
    tmp_path, tillicum
):
    """A module may ask for fewer GPUs than the target's (``remote_resources``: Find
    look-alikes holds one on a target sized for two); the shepherd records what it asked for,
    and the cost bills that, not the target's count."""
    run_dir = _finished(
        tmp_path,
        tillicum,
        worker_started_at="2026-10-06T15:00:00Z",
        worker_completed_at="2026-10-06T15:20:00Z",
        remote_gpus=1,
    )
    fields = run_metrics.completion_fields(run_dir, tillicum.to_dict())
    assert fields["estimated_cost_usd"] == pytest.approx(1 * (20 / 60) * 0.50)


def test_a_local_run_has_no_runtime_and_no_cost(tmp_path):
    run_dir = _finished(tmp_path, ComputeTarget("cuda", None, None, None))
    fields = run_metrics.completion_fields(run_dir, {"backend": "cuda"})
    assert "runtime_s" not in fields and "estimated_cost_usd" not in fields
