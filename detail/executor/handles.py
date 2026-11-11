"""Concrete run handle wrapping an executor and a job id (Section 15).

Returned by the Python API's ``run_*`` methods. Backed by any ``Executor``, so
the same handle works for local and (later) cloud execution.
"""

from collections.abc import Iterator

from detail.core.dataclasses import ProgressEvent
from detail.executor.base import Executor, RunHandle


class ExecutorRunHandle(RunHandle):
    """A RunHandle that delegates to an ``Executor`` for one submitted job."""

    def __init__(self, executor: Executor, job_id: str) -> None:
        self._executor = executor
        self.id = job_id

    @property
    def status(self) -> str:
        return self._executor.status(self.id).value

    @property
    def error(self) -> str | None:
        return self._executor.error(self.id)

    def progress(self) -> Iterator[ProgressEvent]:
        return self._executor.stream_progress(self.id)

    def wait(self) -> "RunHandle":
        for _ in self._executor.stream_progress(self.id):
            pass
        return self

    def cancel(self) -> None:
        self._executor.cancel(self.id)
