"""RemoteExecutor (§6) — runs jobs on a configured remote Slurm cluster.

Submitting spawns an ordinary **local** subprocess — a shepherd
(``detail.executor.remote_shepherd``) — using exactly :class:`LocalExecutor`'s own
subprocess/pid/kill machinery (``new_process_group_kwargs``, the run dir's
``job.json``/pid envelope). Everything downstream of ``submit`` — ``status``,
``stream_progress``, ``error``, and every existing reattach/liveness check built on
``executor/local.py`` (``pid_alive``, ``worker_alive``, ``_reconcile_run``) — is
inherited unchanged from :class:`LocalExecutor`: it only ever looks at the local run
dir, which the shepherd mirrors remote state into. The shepherd, not the remote node,
is what every liveness check watches; a killed shepherd reads exactly like a killed
local worker.

``cancel`` is :class:`LocalExecutor`'s, through ``cancel_run_dir``, which issues the
``scancel`` directly (:func:`cancel_scheduler_job`) before stopping the shepherd — so
cancellation depends neither on the shepherd noticing the cooperative flag before it is
killed, nor on a live handle.
"""

import os
import subprocess
import sys
from pathlib import Path

from detail.core.dataclasses import ComputeTarget, Job, job_type_tag
from detail.executor import rundir
from detail.executor.local import LocalExecutor, new_process_group_kwargs
from detail.executor.runners import JOB_RUNNERS, RUNNER_MODULES_ENV
from detail.executor.schedulers.slurm import SlurmScheduler
from detail.storage import remote_targets as remote_targets_store

SHEPHERD_LOG_FILENAME = rundir.SHEPHERD_LOG_FILENAME

_SCHEDULERS = {"slurm": SlurmScheduler()}


def executor_for(
    experiment_root: Path | str,
    compute: ComputeTarget | None,
    runner_modules: tuple[str, ...] = (),
) -> LocalExecutor:
    """The :class:`Executor` ``compute`` selects — the one choice point every
    ``ProcessingModule``-shaped caller shares (``Recording.run()``,
    ``Experiment.label_dataset()``, ``Experiment.train_model()``/``resume_training()``).
    A caller does not need its own ``if compute.backend == "remote"``: ``RemoteExecutor``
    is a drop-in ``LocalExecutor`` (§6), so nothing past this call needs to know which
    one it got. Extracted once four sites needed the identical dispatch, per this
    project's own rule against N parallel copies of one shape."""
    if compute is not None and compute.backend == "remote":
        return RemoteExecutor(
            experiment_root, compute.remote_target, runner_modules=runner_modules
        )
    return LocalExecutor(experiment_root, runner_modules=runner_modules)


class RemoteExecutor(LocalExecutor):
    """Runs jobs on the registered remote target named ``target``."""

    def __init__(
        self,
        experiment_root: Path | str,
        target: str,
        runner_modules: tuple[str, ...] = (),
        poll_interval: float = 0.25,
    ) -> None:
        super().__init__(
            experiment_root, runner_modules=runner_modules, poll_interval=poll_interval
        )
        self._target = target

    def submit(self, job: Job) -> str:
        tag = job_type_tag(job)
        spec = JOB_RUNNERS.get(tag)
        if spec is None:
            raise ValueError(f"no runner registered for job type {tag!r}")
        run_dir = Path(spec.run_dir_for(self._experiment_root, job))
        run_dir.mkdir(parents=True, exist_ok=True)
        rundir.init_run_dir(run_dir, job)
        env = {**os.environ, RUNNER_MODULES_ENV: ",".join(self._runner_modules)}
        with rundir.open_log(run_dir, SHEPHERD_LOG_FILENAME) as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "detail.executor.remote_shepherd",
                    str(run_dir),
                    self._target,
                    str(self._experiment_root),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                **new_process_group_kwargs(),
            )
        rundir.set_pid(run_dir, proc.pid)
        self._runs[job.job_id] = run_dir
        self._procs[job.job_id] = proc
        return job.job_id

    def cancel(self, job_id: str) -> None:
        # `cancel_run_dir` sends the scancel itself (`cancel_scheduler_job`), so a run cancelled
        # by its directory alone — after the GUI restarted — reaches the cluster too.
        super().cancel(job_id)


def cancel_scheduler_job(run_dir: Path | str) -> bool | None:
    """Cancel the cluster job a remote run's envelope names; ``None`` when there is none.

    ``True`` when the scheduler took the cancel, ``False`` when it could not be asked (the
    target is unreachable or no longer registered here) — then the run's shepherd, which reads
    the cancel flag at every poll and asks again until it is delivered, is the one to send it.
    """
    envelope = rundir.read_envelope(Path(run_dir))
    remote_job_id = envelope.get("remote_job_id")
    compute = (envelope.get("job") or {}).get("compute") or {}
    target_name = compute.get("remote_target") if isinstance(compute, dict) else None
    if remote_job_id is None or target_name is None:
        return None
    try:
        target = remote_targets_store.get_remote_target(target_name)
    except (KeyError, ValueError):
        return False
    return _SCHEDULERS[target.scheduler].cancel(target, remote_job_id) is not False
