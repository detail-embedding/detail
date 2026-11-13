import pytest

from detail.executor.base import Executor, RunHandle


@pytest.mark.parametrize("abc", [Executor, RunHandle])
def test_abstract_base_cannot_instantiate(abc):
    with pytest.raises(TypeError):
        abc()


def test_complete_executor_instantiates():
    class FakeExecutor(Executor):
        def submit(self, job):
            return "job_1"

        def status(self, job_id):
            return None

        def cancel(self, job_id):
            pass

        def stream_progress(self, job_id):
            return iter(())

        def run_dir(self, job_id):
            return None

    ex = FakeExecutor()
    assert ex.submit(None) == "job_1"
    assert list(ex.stream_progress("job_1")) == []


def test_complete_run_handle_instantiates():
    class FakeHandle(RunHandle):
        id = "run_1"
        status = "complete"

        def progress(self):
            return iter(())

        def wait(self):
            return self

        def cancel(self):
            pass

    h = FakeHandle()
    assert h.wait() is h
    assert h.id == "run_1"


def test_incomplete_executor_cannot_instantiate():
    class PartialExecutor(Executor):
        def submit(self, job):
            return "x"

        # status/cancel/stream_progress missing

    with pytest.raises(TypeError):
        PartialExecutor()
