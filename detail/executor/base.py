"""Executor and run-handle protocols (Section 6).

The executor separates *what* a job does from *where* it runs. Callers submit a
serializable ``Job`` and listen for ``ProgressEvent``s; they never call model
code directly. Concrete executors (``LocalExecutor``, later ``CloudExecutor``)
implement ``Executor``; ``RunHandle`` is the user-facing handle returned by the
Python API's ``run_*`` methods (Section 15).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from detail.core.dataclasses import Job, JobStatus, ProgressEvent


class Executor(ABC):
    """Runs jobs and reports progress, abstracting over local vs cloud compute."""

    @abstractmethod
    def submit(self, job: Job) -> str:
        """Submit a job for execution and return its job id."""

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Current status of a submitted job."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Request cancellation of a running job."""

    @abstractmethod
    def stream_progress(self, job_id: str) -> Iterator[ProgressEvent]:
        """Yield progress events until the job reaches a terminal status."""

    @abstractmethod
    def run_dir(self, job_id: str) -> Path:
        """This job's own run directory — where its ``job.json`` envelope lives.

        The one place a job's ``started_at``/``completed_at`` (stamped by
        ``rundir.set_status``) can be read back from, for a run family's
        ``run_metrics.completion_fields`` call once the job is terminal."""

    def error(self, job_id: str) -> str | None:
        """The error message of a failed job, or None. Default None."""
        return None


class RunHandle(ABC):
    """User-facing handle for a submitted run (Section 15)."""

    id: str
    status: str  # mirrors JobStatus values

    @property
    def error(self) -> str | None:
        """The run's error message if it failed (terminal), else None."""
        return None

    @property
    def owns_worker(self) -> bool:
        """Whether **this process** spawned the worker behind the run.

        The question a surface actually asks is "if I give up now, do I abandon a process I
        started?" — and only the handle knows. A launched run's worker is in this process's care,
        so walking away from it orphans it; a *reattached* run's worker belongs to whoever
        launched it, and detaching is the whole point of being able to reattach.

        The CLI's ``Ctrl-C`` handling is derived from this rather than passed in. It was a
        parameter on two of the three progress drivers, defaulting to "cancel" — so it reached
        the two callers somebody was working on, and the five ``attach`` commands whose
        ``--help`` promises *"Ctrl-C detaches; it does not cancel the run"* all used the third
        driver and all cancelled. That is this repo's recorded class: an opt-in whose default
        means "as before" reaches the surface being edited and no other.

        ``True`` here because it is the **safe** default. A new attached handle that forgets to
        override cancels a run the user was told it would not, which is loud and immediate; a new
        launch handle that forgot would orphan a worker silently, which is worse.
        """
        return True

    @abstractmethod
    def progress(self) -> Iterator[ProgressEvent]:
        """Yield progress events as the job runs; returns immediately when complete."""

    @abstractmethod
    def wait(self) -> "RunHandle":
        """Block until the run is complete or failed. Returns self for chaining."""

    @abstractmethod
    def cancel(self) -> None:
        """Request cancellation of the run."""
