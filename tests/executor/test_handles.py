from detail.core.dataclasses import DetectionProgressEvent, JobStatus
from detail.executor.base import Executor
from detail.executor.handles import ExecutorRunHandle


class FakeExecutor(Executor):
    def __init__(self, events, error=None):
        self._events = events
        self._status = JobStatus.RUNNING
        self._error = error
        self.cancelled = []

    def submit(self, job):
        return "job_1"

    def status(self, job_id):
        return self._status

    def error(self, job_id):
        return self._error

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def stream_progress(self, job_id):
        for event in self._events:
            yield event
        self._status = JobStatus.COMPLETE


def _events():
    return [
        DetectionProgressEvent(1, 2, 50, 100, 1.0, 1.0),
        DetectionProgressEvent(2, 2, 100, 100, 2.0, 0.0),
    ]


def test_handle_exposes_id_and_status():
    ex = FakeExecutor(_events())
    handle = ExecutorRunHandle(ex, "job_1")
    assert handle.id == "job_1"
    assert handle.status == "running"


def test_progress_delegates_to_executor():
    ex = FakeExecutor(_events())
    handle = ExecutorRunHandle(ex, "job_1")
    assert list(handle.progress()) == _events()


def test_wait_blocks_until_terminal_and_returns_self():
    ex = FakeExecutor(_events())
    handle = ExecutorRunHandle(ex, "job_1")
    assert handle.wait() is handle
    assert handle.status == "complete"


def test_cancel_delegates_to_executor():
    ex = FakeExecutor(_events())
    handle = ExecutorRunHandle(ex, "job_1")
    handle.cancel()
    assert ex.cancelled == ["job_1"]


def test_error_delegates_to_executor():
    handle = ExecutorRunHandle(FakeExecutor(_events(), error="boom"), "job_1")
    assert handle.error == "boom"
    assert ExecutorRunHandle(FakeExecutor(_events()), "job_1").error is None
