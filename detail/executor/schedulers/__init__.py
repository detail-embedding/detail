"""Scheduler adapters for :class:`detail.executor.remote.RemoteExecutor` (§6).

Slurm-specific logic (script rendering, ``squeue``/``sacct`` parsing, ``scancel``) is
isolated behind :class:`~detail.executor.schedulers.base.JobScheduler` so a second
scheduler (PBS, a plain SSH background process, ...) is a new implementation, not a
rewrite of the executor above it.
"""
