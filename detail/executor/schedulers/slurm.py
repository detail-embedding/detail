"""SlurmScheduler (§6) — sbatch script rendering, squeue/sacct polling, scancel.

Tillicum and Hyak are both Slurm clusters, so this is the only :class:`JobScheduler`
implementation today; a future non-Slurm cluster is a second one, not a change here.
"""

import posixpath
import re
import shlex
import subprocess

from detail.configs.remote import RemoteTargetConfig
from detail.executor import ssh_transport
from detail.executor.runners import RUNNER_MODULES_ENV
from detail.executor.schedulers.base import JobScheduler, JobState

_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")

#: ``squeue -h -o %T`` state letters/words -> JobState. Anything not listed (a state
#: this map hasn't seen yet) is treated as RUNNING rather than raising, since the job
#: is — by definition, still being reported by squeue — not finished.
_SQUEUE_STATE_MAP = {
    "PENDING": JobState.PENDING,
    "CONFIGURING": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "SUSPENDED": JobState.RUNNING,
}

#: ``sacct -n -o State`` values -> JobState: **every** state sacct documents, because one this map
#: lacks reads as ``FAILED``. sacct is asked whenever squeue gives no state, which is not only a
#: job that has left the queue: a controller too busy to answer (squeue exits 1) does it to a
#: running job, and sacct says ``RUNNING``. With the live states missing, one such poll recorded
#: a job failed 30 s into a run Slurm completed 12 minutes later.
_SACCT_STATE_MAP = {
    "PENDING": JobState.PENDING,
    "REQUEUED": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "RESIZING": JobState.RUNNING,
    "SUSPENDED": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "CANCELLED": JobState.CANCELLED,
    "TIMEOUT": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "PREEMPTED": JobState.FAILED,
    "BOOT_FAIL": JobState.FAILED,
    "DEADLINE": JobState.FAILED,
    "REVOKED": JobState.FAILED,
}

#: Every job state ``sacct`` can report (``man sacct``, JOB STATE CODES), for the guard that the
#: map above leaves none to the ``FAILED`` default.
SACCT_STATES = (
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "COMPLETING",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PENDING",
    "PREEMPTED",
    "REQUEUED",
    "RESIZING",
    "REVOKED",
    "RUNNING",
    "SUSPENDED",
    "TIMEOUT",
)

SBATCH_SCRIPT_FILENAME = "job.sbatch"


def _sbatch_quote(value: str) -> str:
    """Quote a value embedded in a ``#SBATCH --option=value`` directive line.

    A ``#SBATCH`` line is parsed by Slurm's own tokenizer at submission time, not by a
    shell — ``shlex.quote`` (single-quoting) is for the worker-invocation lines below,
    which really do run through bash. Slurm's directive parser instead honors double
    quotes, so an unquoted value containing a space — an experiment name a user typed,
    e.g. "Open Field B", mirrored verbatim into ``remote_run_dir`` — splits into a
    second, invalid directive: ``sbatch: error: Invalid directive found in batch
    script``. Found live, submitting a real distill job against that experiment."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _activation_lines(target: RemoteTargetConfig) -> list[str]:
    """Module loads + conda activation, in order — the environment a submitted
    sbatch job runs under. Shared with :func:`activation_prefix`, which reproduces
    the same environment for a one-off ssh command (a pre-submission check) that
    must see what the job will see."""
    lines = [f"module load {module}" for module in target.modules]
    if target.conda_env:
        lines.append(f"conda activate {target.conda_env}")
    return lines


def activation_prefix(target: RemoteTargetConfig) -> str:
    """Shell prefix (``"module load ... && conda activate ... && "``, or ``""`` if
    the target needs neither) that puts a one-off ssh command in the same
    environment a submitted sbatch job runs under."""
    return "".join(f"{line} && " for line in _activation_lines(target))


def render_sbatch_script(
    target: RemoteTargetConfig, job_id: str, remote_run_dir: str, runner_modules: str
) -> str:
    """The sbatch script for one job: requests the profile's resources, loads its
    modules/conda env, and runs the same worker entrypoint a local job uses
    (``python -m detail.executor.worker <remote_run_dir>``) — unmodified, unaware it
    is running on a compute node instead of the user's machine.

    ``runner_modules`` is the comma-joined ``DETAIL_RUNNER_MODULES`` value the local
    executor resolved (``executor/runners.py``): the remote worker is a fresh shell on
    a compute node, so nothing of the shepherd's own environment reaches it unless this
    script exports it explicitly. Without this line the worker's registry is empty and
    every job type fails with "no runner registered" — required, not defaulted, so a
    caller can't silently reintroduce that."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=detail-{job_id}",
        f"#SBATCH --output={_sbatch_quote(f'{remote_run_dir}/slurm-%j.out')}",
    ]
    if target.partition:
        lines.append(f"#SBATCH --partition={target.partition}")
    if target.account:
        lines.append(f"#SBATCH --account={target.account}")
    gres = "gpu"
    if target.gpu_type:
        gres += f":{target.gpu_type}"
    gres += f":{target.gpus_per_job}"
    lines.append(f"#SBATCH --gres={gres}")
    lines.append(f"#SBATCH --cpus-per-task={target.cpus_per_task}")
    lines.append(f"#SBATCH --mem={target.mem_gb}G")
    lines.append(f"#SBATCH --time={target.walltime}")
    lines.append("set -e")
    lines.extend(_activation_lines(target))
    lines.append(f"export {RUNNER_MODULES_ENV}={shlex.quote(runner_modules)}")
    lines.append(f"python -m detail.executor.worker {shlex.quote(remote_run_dir)}")
    return "\n".join(lines) + "\n"


class SlurmScheduler(JobScheduler):
    def submit_script(
        self, target: RemoteTargetConfig, script: str, remote_run_dir: str
    ) -> str:
        script_path = posixpath.join(remote_run_dir, SBATCH_SCRIPT_FILENAME)
        ssh_transport.write_remote_file(target.host, script_path, script)
        result = ssh_transport.run_remote(
            target.host, f"sbatch {shlex.quote(script_path)}"
        )
        match = _SBATCH_JOB_ID_RE.search(result.stdout)
        if match is None:
            raise ValueError(
                f"could not parse a job id from sbatch output: {result.stdout!r}"
            )
        return match.group(1)

    def poll_state(self, target: RemoteTargetConfig, remote_job_id: str) -> JobState:
        """``UNREACHABLE`` when ssh could not run the question at all.

        **Not asking is not an answer.** Both queries below used to fall through to the last
        line when they came back empty, and an ssh that never connected is empty: a dropped
        session read as "neither squeue nor sacct know this job", which is ``FAILED``, so a
        healthy multi-hour job was recorded as failed because the laptop slept. The two ways of
        being empty are told apart by who produced the exit status (``ssh_transport.ssh_failed``).
        """
        try:
            squeue = ssh_transport.run_remote(
                target.host,
                f"squeue -j {shlex.quote(remote_job_id)} -h -o %T",
                check=False,
                timeout_s=ssh_transport.ASK_TIMEOUT_S,
                refused_retries=ssh_transport.ASK_REFUSED_RETRIES,
            )
        except subprocess.TimeoutExpired:
            return JobState.UNREACHABLE
        if ssh_transport.ssh_failed(squeue):
            return JobState.UNREACHABLE
        line = (
            squeue.stdout.strip().splitlines()[0].strip()
            if squeue.stdout.strip()
            else ""
        )
        if squeue.returncode == 0 and line:
            return _SQUEUE_STATE_MAP.get(line.upper(), JobState.RUNNING)
        # No longer in squeue — it has left the queue (completed, failed, cancelled,
        # or ran outside squeue's window); sacct is the historical record.
        try:
            sacct = ssh_transport.run_remote(
                target.host,
                f"sacct -j {shlex.quote(remote_job_id)} -n -o State -P",
                check=False,
                timeout_s=ssh_transport.ASK_TIMEOUT_S,
                refused_retries=ssh_transport.ASK_REFUSED_RETRIES,
            )
        except subprocess.TimeoutExpired:
            return JobState.UNREACHABLE
        if ssh_transport.ssh_failed(sacct):
            return JobState.UNREACHABLE
        for raw in sacct.stdout.splitlines():
            # sacct emits one line per job step (main step, "batch", "extern", ...) and
            # a state can carry a suffix ("CANCELLED by 12345") — only the first word
            # of the first non-empty line is the state itself.
            state = raw.strip().split()[0] if raw.strip() else ""
            if state:
                return _SACCT_STATE_MAP.get(state, JobState.FAILED)
        # Neither squeue nor sacct know this job id — treat as failed rather than loop
        # forever on a job the scheduler has no record of at all.
        return JobState.FAILED

    def cancel(self, target: RemoteTargetConfig, remote_job_id: str) -> bool:
        """``False`` when ssh could not deliver the ``scancel``, so the caller can ask again.

        A cancel sent over a dead connection used to read as sent, and the job ran on.
        """
        try:
            result = ssh_transport.run_remote(
                target.host,
                f"scancel {shlex.quote(remote_job_id)}",
                check=False,
                timeout_s=ssh_transport.ASK_TIMEOUT_S,
                refused_retries=ssh_transport.ASK_REFUSED_RETRIES,
            )
        except subprocess.TimeoutExpired:
            return False
        return not ssh_transport.ssh_failed(result)
