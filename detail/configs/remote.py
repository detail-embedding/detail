"""Remote (Slurm) target config (§6) — the typed shape of a machine-global cluster profile.

A ``RemoteTargetConfig`` is not a per-run job config (it never reaches ``configs/describe.py``'s
processing/analysis registry) — it is a machine-global profile, stored like ``credentials.yaml``/
``storage.yaml`` (``storage/remote_targets.py``), that a run's ``ComputeTarget`` names by id. It
is still a pydantic model for validation and labels, but — unlike a module run config — its GUI
dialog (``gui/pages.py::_RemoteTargetDialog``) and its CLI command (``detail remote add``) are
each a hand-written form, not built through the generic ``gui/_pydantic_form.py``/``describe.py``
machinery: a new field here needs explicit wiring in both.
"""

import re

from pydantic import BaseModel, Field, field_validator

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class RemoteTargetConfig(BaseModel):
    name: str = Field(
        ...,
        description="Short identifier for this cluster profile (e.g. `tillicum`). Referenced by "
        "a run's `ComputeTarget.remote_target`.",
        json_schema_extra={"label": "Name"},
    )
    host: str = Field(
        ...,
        description="SSH host to submit through. Should match a `Host` alias in `~/.ssh/config` "
        "so keys, an agent, ControlMaster connection-sharing, and a cached Duo session are "
        "already handled — DETAIL never stores or prompts for credentials itself.",
        json_schema_extra={"label": "SSH host"},
    )
    remote_workdir: str = Field(
        ...,
        description="Scratch directory on the cluster DETAIL stages job data under (e.g. "
        "`/gscratch/scrubbed/<netid>/detail_runs`). Treated as disposable: local storage stays "
        "authoritative and outputs are synced back after each job.",
        json_schema_extra={"label": "Remote work directory"},
    )
    scheduler: str = Field(
        "slurm",
        description="Job scheduler on this cluster. Only `slurm` is implemented today.",
        json_schema_extra={"label": "Scheduler"},
    )
    partition: str | None = Field(
        None,
        description="Slurm partition to submit into (`sbatch --partition`). Leave unset to use "
        "the cluster's default.",
        json_schema_extra={"label": "Partition"},
    )
    account: str | None = Field(
        None,
        description="Slurm account/allocation to charge (`sbatch --account`).",
        json_schema_extra={"label": "Account"},
    )
    conda_env: str | None = Field(
        None,
        description="Name of the conda environment on the cluster with `detail` installed. "
        "Activated before the worker runs.",
        json_schema_extra={"label": "Conda environment"},
    )
    modules: list[str] = Field(
        default_factory=list,
        description="LMOD modules to load before running (e.g. `cuda/12.4`), in order.",
        json_schema_extra={"label": "Modules to load"},
    )
    gpu_type: str | None = Field(
        None,
        description="GPU type to request, if the cluster's Slurm config distinguishes types "
        "(`--gres=gpu:<type>:<n>`). Leave unset to request any available GPU.",
        json_schema_extra={"label": "GPU type"},
    )
    gpus_per_job: int = Field(
        1,
        ge=1,
        description="GPUs to request per job. Also the number of on-cluster device indices the "
        "worker is handed (`ComputeTarget(backend='cuda', device_indices=[0..gpus_per_job-1])`).",
        json_schema_extra={"label": "GPUs per job"},
    )
    cpus_per_task: int = Field(
        4,
        ge=1,
        description="CPU cores to request (`sbatch --cpus-per-task`).",
        json_schema_extra={"label": "CPUs per task"},
    )
    mem_gb: int = Field(
        32,
        ge=1,
        description="Memory to request, in GB (`sbatch --mem`).",
        json_schema_extra={"label": "Memory (GB)"},
    )
    walltime: str = Field(
        "24:00:00",
        description="Walltime limit in Slurm's `HH:MM:SS` (or `D-HH:MM:SS`) format "
        "(`sbatch --time`). Slurm kills a job at this limit, so the default is the longest "
        "a Tillicum job may run (24 hours). A cluster whose limit is lower rejects the "
        "submission with an error that names it; set this to that limit.",
        json_schema_extra={"label": "Walltime"},
    )
    max_concurrent_jobs: int = Field(
        8,
        ge=1,
        le=64,
        description="How many recordings of one run may be on the cluster at once, each its "
        "own Slurm job of `gpus_per_job` GPUs (so the GPUs a run uses are this × "
        "`gpus_per_job`). The default is 8, set by the connection rather than the cluster: "
        "every in-flight job keeps ssh sessions open through the one shared connection, and "
        "OpenSSH allows 10 per connection unless the host raises it, past which new sessions "
        "are refused and the runs wait and retry. Jobs beyond what the cluster has free just "
        "queue. Needs `rsync` on this machine (without it, runs go one at a time).",
        json_schema_extra={"label": "Concurrent jobs"},
    )
    cost_per_gpu_hour: float | None = Field(
        None,
        ge=0,
        description="Estimated $/GPU-hour this target bills, for a live cost estimate on "
        "its runs (estimated cost = GPUs the job held × running hours × this rate: "
        "`gpus_per_job`, or fewer for a job that asks for fewer). Leave "
        "unset to show no cost estimate for runs on this target.",
        json_schema_extra={"label": "Cost per GPU-hour (USD)"},
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                f"remote target name {v!r} must start with a lowercase letter or digit and "
                "contain only lowercase letters, digits, '-', '_'"
            )
        return v

    @field_validator("scheduler")
    @classmethod
    def _validate_scheduler(cls, v: str) -> str:
        if v != "slurm":
            raise ValueError(
                f"unsupported scheduler {v!r}; only 'slurm' is implemented"
            )
        return v
