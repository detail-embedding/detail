"""JobScheduler protocol (§6) — the seam a non-Slurm scheduler implements."""

from abc import ABC, abstractmethod
from enum import Enum

from detail.configs.remote import RemoteTargetConfig


class JobState(str, Enum):
    """Scheduler-reported state of a submitted remote job, independent of any
    particular scheduler's vocabulary (Slurm's ``PENDING``/``COMPLETED``/...)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: The scheduler **could not be asked** (the ssh session is down, or the call timed out), so
    #: nothing is known about the job: not that it failed, not that it is gone. Never terminal;
    #: a caller waits and asks again.
    UNREACHABLE = "unreachable"


#: States :meth:`JobScheduler.poll_state` never transitions out of.
TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})


class JobScheduler(ABC):
    """Submits and tracks jobs on a remote target's job scheduler."""

    @abstractmethod
    def submit_script(
        self, target: RemoteTargetConfig, script: str, remote_run_dir: str
    ) -> str:
        """Write and submit the rendered job ``script`` under ``remote_run_dir`` on
        ``target``; return the scheduler's own job id (e.g. a Slurm job id)."""

    @abstractmethod
    def poll_state(self, target: RemoteTargetConfig, remote_job_id: str) -> JobState:
        """Current scheduler-reported state of ``remote_job_id`` on ``target``.

        :attr:`JobState.UNREACHABLE` when the scheduler could not be asked, which is a different
        answer from every terminal one and must not be folded into ``FAILED``.
        """

    @abstractmethod
    def cancel(self, target: RemoteTargetConfig, remote_job_id: str) -> bool | None:
        """Cancel ``remote_job_id`` on ``target``. A no-op if it is already terminal.

        ``False`` means the request could not be delivered (the scheduler was unreachable) and
        should be sent again; anything else, including ``None``, means it went.
        """
