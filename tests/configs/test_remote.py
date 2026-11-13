import pytest
from pydantic import ValidationError

from detail.configs.remote import RemoteTargetConfig


def _config(**overrides):
    fields = dict(
        name="tillicum",
        host="tillicum",
        remote_workdir="/gscratch/scrubbed/uwnetid/detail_runs",
    )
    fields.update(overrides)
    return RemoteTargetConfig(**fields)


def test_defaults():
    config = _config()
    assert config.scheduler == "slurm"
    assert config.gpus_per_job == 1
    assert config.cpus_per_task == 4
    assert config.mem_gb == 32
    assert config.walltime == "24:00:00", "the longest a Tillicum job may run"
    assert config.max_concurrent_jobs == 8
    assert config.modules == []
    assert config.partition is None
    assert config.account is None
    assert config.conda_env is None
    assert config.gpu_type is None


def test_full_roundtrip_via_dict():
    config = _config(
        partition="gpu-a40",
        account="mylab",
        conda_env="detail-env",
        modules=["cuda/12.4", "gcc/13.2"],
        gpu_type="h200",
        gpus_per_job=2,
        cpus_per_task=8,
        mem_gb=64,
        walltime="1-00:00:00",
    )
    dumped = config.model_dump()
    assert RemoteTargetConfig(**dumped) == config


@pytest.mark.parametrize(
    "bad_name", ["", "Tillicum", "_tillicum", "-tillicum", "til licum", "til/cum"]
)
def test_invalid_name_rejected(bad_name):
    with pytest.raises(ValidationError):
        _config(name=bad_name)


def test_unsupported_scheduler_rejected():
    with pytest.raises(ValidationError, match="unsupported scheduler"):
        _config(scheduler="pbs")


@pytest.mark.parametrize(
    "field", ["gpus_per_job", "cpus_per_task", "mem_gb", "max_concurrent_jobs"]
)
def test_non_positive_resource_rejected(field):
    with pytest.raises(ValidationError):
        _config(**{field: 0})


def test_concurrent_jobs_has_an_upper_bound():
    """A typo of 800 would open that many ssh sessions through one connection."""
    assert _config(max_concurrent_jobs=64).max_concurrent_jobs == 64
    with pytest.raises(ValidationError):
        _config(max_concurrent_jobs=65)


def test_a_target_saved_before_the_field_existed_loads_with_the_default_and_keeps_its_walltime():
    """`remote_targets.yaml` stores every field of a target, so a saved one keeps the `04:00:00`
    it was written with and gains `max_concurrent_jobs` at the default: the walltime change
    reaches new targets, the concurrency default reaches every target."""
    saved = dict(
        name="tillicum",
        host="tillicum",
        remote_workdir="/scratch/x",
        walltime="04:00:00",
        gpus_per_job=1,
    )

    config = RemoteTargetConfig(**saved)

    assert config.walltime == "04:00:00"
    assert config.max_concurrent_jobs == 8
