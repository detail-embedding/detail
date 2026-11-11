import contextlib
import os
import textwrap
import threading
import time
from pathlib import Path

import pytest

from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    DetectionProgressEvent,
    JobStatus,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.executor import rundir
from detail.executor.local import LocalExecutor
from detail.executor.runners import JOB_RUNNERS, register_runner
from detail.executor.worker import run_job
from tests.concurrency import join_all

_TERMINAL = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.PARTIAL}


def _detection_job(job_id="det_01"):
    return DetectionJob(
        job_id=job_id,
        recording_id="recording_001",
        model_id="sam3",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


def _run_dir_for(experiment_root, job):
    return Path(experiment_root) / "runs" / job.job_id


@contextlib.contextmanager
def _detection_runner_override():
    """Save the real 'detection' runner, then restore it on exit — the runner
    registry is process-global and the real module may already be registered."""
    prior = JOB_RUNNERS.get("detection")
    try:
        yield
    finally:
        if prior is not None:
            JOB_RUNNERS["detection"] = prior
        else:
            JOB_RUNNERS.pop("detection", None)


@contextlib.contextmanager
def _registered(run_fn):
    with _detection_runner_override():
        register_runner("detection", run_dir_for=_run_dir_for)(run_fn)
        yield


def _init(tmp_path, job_id="det_01"):
    run_dir = tmp_path / "runs" / job_id
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _detection_job(job_id))
    return run_dir


# --- in-process run_job ----------------------------------------------------- #
def test_run_job_success_records_progress_and_complete(tmp_path):
    run_dir = _init(tmp_path)

    def runner(job, rd, progress_cb, stop_event):
        progress_cb(DetectionProgressEvent(1, 2, 50, 100, 1.0, 1.0))
        progress_cb(DetectionProgressEvent(2, 2, 100, 100, 2.0, 0.0))

    with _registered(runner):
        assert run_job(run_dir) == JobStatus.COMPLETE
    assert rundir.read_status(run_dir) == JobStatus.COMPLETE
    assert rundir.read_progress(run_dir).parts_done == 2


def test_run_job_failure_records_error(tmp_path):
    run_dir = _init(tmp_path)

    def runner(job, rd, progress_cb, stop_event):
        raise RuntimeError("boom")

    with _registered(runner):
        assert run_job(run_dir) == JobStatus.FAILED
    env = rundir.read_envelope(run_dir)
    assert env["status"] == "failed"
    assert "boom" in env["error"]


def test_run_job_keyboard_interrupt_records_cancelled(tmp_path):
    run_dir = _init(tmp_path)

    def runner(job, rd, progress_cb, stop_event):
        raise KeyboardInterrupt  # Ctrl-C reaches the worker mid-run

    with _registered(runner):
        assert run_job(run_dir) == JobStatus.CANCELLED
    # terminal (not in ACTIVE_STATUSES), so it won't block the next run
    assert rundir.read_status(run_dir) == JobStatus.CANCELLED


def test_run_job_cancel_marks_partial_and_sets_stop_event(tmp_path):
    run_dir = _init(tmp_path)

    def runner(job, rd, progress_cb, stop_event):
        rundir.request_cancel(rd)  # a cancel arrives
        assert stop_event.wait(3.0)  # the watcher propagates it

    with _registered(runner):
        assert run_job(run_dir) == JobStatus.PARTIAL
    assert rundir.read_status(run_dir) == JobStatus.PARTIAL


def test_run_job_stop_and_keep_stops_the_runner_and_lands_complete(tmp_path):
    """The twin of the cancel case, and the difference is the whole point: a stop-and-keep is
    a finish the user chose, so the runner stops the same way and the run is **complete**.
    It shared cancel's marker until #486, and every stopped run landed partial."""
    run_dir = _init(tmp_path)

    def runner(job, rd, progress_cb, stop_event):
        rundir.request_stop(rd)
        assert stop_event.wait(
            3.0
        )  # the watcher propagates a stop exactly like a cancel

    with _registered(runner):
        assert run_job(run_dir) == JobStatus.COMPLETE
    assert rundir.read_status(run_dir) == JobStatus.COMPLETE


def test_run_job_no_runner_fails(tmp_path):
    run_dir = _init(tmp_path)
    with _detection_runner_override():
        JOB_RUNNERS.pop("detection", None)
        assert run_job(run_dir) == JobStatus.FAILED
        assert "no runner" in rundir.read_envelope(run_dir)["error"]


# --- LocalExecutor over a real subprocess ----------------------------------- #
def _write_runner_module(mod_dir: Path, name: str, body: str) -> None:
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / f"{name}.py").write_text(
        "from pathlib import Path\n"
        "from detail.core.dataclasses import DetectionProgressEvent\n"
        "from detail.executor import rundir\n"
        "from detail.executor.runners import register_runner\n\n"
        "def _run_dir_for(experiment_root, job):\n"
        "    return Path(experiment_root) / 'runs' / job.job_id\n\n"
        f"@register_runner('detection', run_dir_for=_run_dir_for)\n"
        f"def run(job, run_dir, progress_cb, stop_event):\n{textwrap.indent(body, '    ')}\n",
        encoding="utf-8",
    )


@contextlib.contextmanager
def _runner_module(tmp_path, monkeypatch, name, body):
    mod_dir = tmp_path / "pkg"
    _write_runner_module(mod_dir, name, body)
    monkeypatch.syspath_prepend(str(mod_dir))  # parent (executor) can import it
    monkeypatch.setenv(
        "PYTHONPATH", str(mod_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )  # worker subprocess can import it
    with _detection_runner_override():
        yield


def _wait_terminal(ex, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ex.status(job_id) in _TERMINAL:
            return ex.status(job_id)
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} never reached terminal (last={ex.status(job_id)})"
    )


def test_local_executor_runs_job_to_completion(tmp_path, monkeypatch):
    body = "progress_cb(DetectionProgressEvent(1, 1, 10, 10, 0.1, 0.0))"
    with _runner_module(tmp_path, monkeypatch, "dummy_ok", body):
        ex = LocalExecutor(tmp_path, runner_modules=("dummy_ok",), poll_interval=0.02)
        job_id = ex.submit(_detection_job())
        events = list(ex.stream_progress(job_id))
        assert ex.status(job_id) == JobStatus.COMPLETE
        assert events[-1].parts_done == 1


def test_local_executor_cancel_marks_partial(tmp_path, monkeypatch):
    body = "stop_event.wait(30)"  # block until cancelled
    with _runner_module(tmp_path, monkeypatch, "dummy_wait", body):
        ex = LocalExecutor(tmp_path, runner_modules=("dummy_wait",), poll_interval=0.02)
        job_id = ex.submit(_detection_job())
        ex.cancel(job_id)
        assert _wait_terminal(ex, job_id) == JobStatus.PARTIAL


def test_local_executor_submit_without_runner_raises(tmp_path):
    with _detection_runner_override():
        JOB_RUNNERS.pop("detection", None)
        ex = LocalExecutor(tmp_path, poll_interval=0.02)
        with pytest.raises(ValueError, match="no runner registered"):
            ex.submit(_detection_job())


def test_stream_progress_ends_when_a_reattached_worker_dies(tmp_path, monkeypatch):
    """A handle with no live ``Popen`` must not poll forever.

    ``stream_progress`` left the loop on a terminal status or on its own ``Popen``
    reporting an exit. A handle this process did not spawn — a reattach (§15), or any
    handle built outside the launcher — holds no ``Popen``, so the second exit could never
    fire and a terminal status became the only one. A worker killed before it wrote one
    left the caller polling a process that was already gone.

    The class: **a wait whose only exit is a message the other side may never send.** It is
    invisible to a behavioural test because nothing is wrong at the line where it is
    written and nothing raises — the run simply never ends, which reads as a slow machine.
    So the assertion here is that the call *returns*, not that it returns something.
    """
    body = "stop_event.wait(30)"  # park the worker so it never writes a terminal status
    with _runner_module(tmp_path, monkeypatch, "dummy_wait", body):
        ex = LocalExecutor(tmp_path, runner_modules=("dummy_wait",), poll_interval=0.02)
        job_id = ex.submit(_detection_job())
        run_dir = ex._runs[job_id]
        for _ in range(
            200
        ):  # the pid is written by submit; wait for the worker to be up
            if rundir.read_pid(run_dir) is not None:
                break
            time.sleep(0.02)
        assert rundir.read_pid(run_dir) is not None

        # Reattach: drop the Popen the way a fresh process would never have had one, then
        # kill the worker so nothing will ever write the status the loop is waiting for.
        proc = ex._procs.pop(job_id)
        proc.kill()
        proc.wait()

        drained: list = []
        # daemon: without the fix this thread never returns, and a non-daemon one would
        # then wedge interpreter shutdown on top of the failure it is there to report.
        poller = threading.Thread(
            target=lambda: drained.extend(ex.stream_progress(job_id)), daemon=True
        )
        poller.start()
        join_all([poller])  # names the hang instead of parking the run in it
        assert ex.status(job_id) == JobStatus.FAILED
        assert "without completing" in (ex.error(job_id) or "")


def test_stream_progress_keeps_waiting_on_a_reattached_worker_that_is_alive(
    tmp_path, monkeypatch
):
    """Guard the guard: the liveness check must not call a *live* worker dead.

    A check that concluded "no Popen, therefore gone" would end every reattached run the
    moment it was watched — passing the test above for the wrong reason, and turning §15's
    durable runs into instant failures.
    """
    body = "progress_cb(DetectionProgressEvent(1, 1, 10, 10, 0.1, 0.0))"
    with _runner_module(tmp_path, monkeypatch, "dummy_ok", body):
        ex = LocalExecutor(tmp_path, runner_modules=("dummy_ok",), poll_interval=0.02)
        job_id = ex.submit(_detection_job())
        # Reattach: no handle on a worker that is running fine. Hold the Popen so this
        # test reaps what it took away from the executor — dropping it unreaped raises a
        # ResourceWarning, which `filterwarnings = error` turns into a failure.
        proc = ex._procs.pop(job_id)
        try:
            events = list(ex.stream_progress(job_id))
            assert ex.status(job_id) == JobStatus.COMPLETE
            assert events[-1].parts_done == 1
        finally:
            proc.wait(timeout=20)


# --- the worker's own output survives the launching console (#279) ----------- #
def test_worker_stdout_and_stderr_land_in_the_run_dir_log(tmp_path, monkeypatch):
    """A spawned worker's output is captured to ``run_dir/worker.log``.

    Without the redirect the worker inherits whatever stdout the launcher had, so the
    backend's own log — the one artifact naming where a stalled stage stopped — lives
    only in the launching terminal's scrollback and dies with the window. That is what
    left #279 undiagnosable. Assert the bytes are on disk, not that the call returned.
    """
    body = (
        "import sys\n"
        "print('MARKER-ON-STDOUT', flush=True)\n"
        "sys.stderr.write('MARKER-ON-STDERR\\n')\n"
        "sys.stderr.flush()\n"
    )
    with _runner_module(tmp_path, monkeypatch, "dummy_log", body):
        ex = LocalExecutor(tmp_path, runner_modules=("dummy_log",), poll_interval=0.02)
        job_id = ex.submit(_detection_job())
        assert _wait_terminal(ex, job_id) == JobStatus.COMPLETE

    log = (tmp_path / "runs" / job_id / "worker.log").read_text(encoding="utf-8-sig")
    assert "MARKER-ON-STDOUT" in log
    assert "MARKER-ON-STDERR" in log  # stderr is folded into the same file


def test_a_crashing_worker_leaves_its_traceback_in_the_log(tmp_path, monkeypatch):
    """The failure case is the one the log exists for. ``job.json``'s ``error`` carries
    only the exception's ``repr``; the traceback that says *where* is on stderr, and
    without the redirect it is discarded exactly when it is needed."""
    body = "raise RuntimeError('boom-in-the-worker')"
    with _runner_module(tmp_path, monkeypatch, "dummy_boom", body):
        ex = LocalExecutor(tmp_path, runner_modules=("dummy_boom",), poll_interval=0.02)
        job_id = ex.submit(_detection_job())
        assert _wait_terminal(ex, job_id) == JobStatus.FAILED

    log = (tmp_path / "runs" / job_id / "worker.log").read_text(encoding="utf-8-sig")
    assert "boom-in-the-worker" in log


# --- what the job ran on, and how much memory it used ----------------------- #
@pytest.mark.parametrize("outcome", ["complete", "failed"])
def test_run_job_records_its_machine_and_peak_memory(tmp_path, outcome):
    """The worker is where the work happens, so it is the one that can say what it ran on and
    how much memory it took — a failed job included, since "it ran out of memory" is the
    question a failure most often raises."""
    from detail.storage import run_facts

    run_dir = _init(tmp_path)

    def runner(job, rd, progress_cb, stop_event):
        if outcome == "failed":
            raise RuntimeError("boom")

    with _registered(runner):
        run_job(run_dir)
    machine = run_facts.read_machine(run_dir)
    resources = run_facts.read_resources(run_dir)
    assert machine is not None and machine["os"]
    assert resources is not None and resources["peak_ram_bytes"] > 0


def test_a_failed_facts_write_cannot_fail_the_job(tmp_path, monkeypatch):
    """A report *about* the job is bookkeeping; a write that raises must not reach the
    worker's isolation boundary and be recorded as the job failing (#289's class)."""
    from detail.storage import state_yaml

    def broken(*_a, **_k):
        raise PermissionError("share went away")

    monkeypatch.setattr(state_yaml, "write_state_yaml", broken)
    run_dir = _init(tmp_path)

    def runner(job, rd, progress_cb, stop_event):
        pass

    with _registered(runner):
        assert run_job(run_dir) == JobStatus.COMPLETE
    assert rundir.read_status(run_dir) == JobStatus.COMPLETE
