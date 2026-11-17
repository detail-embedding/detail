"""A cancel is recorded as resumable only by a runner that can resume.

**The class.** ``partial`` means "interrupted, resumable" (``JobStatus.PARTIAL``, Tenet 12, and
``docs/cli.md``'s run statuses). The worker recorded ``partial`` for **every** runner that
returned normally after a cancel, whether or not the runner had a resume. Detection has none, so
a one-part detection run whose frames all finished during the cancel's grace period landed
``partial``, and tracking then refused it as "not complete" with nothing that could finish it
(#28 finding 4). The same cancel landed ``failed`` instead when the stop reached a part and the
runner raised, so one user action gave two different statuses depending on timing, and neither
was ``cancelled``.

**How it hid.** Nothing is wrong where it is written: the worker's mapping is correct for a
runner with a resume (keypoint detect), and detection's module docstring says "stop is
failure, never partial", which is what its multi-GPU path does. The inline path never told the
part to stop (only the multi-GPU poll loop touched ``stop.flag``), so the run simply finished.

**The rule.** Every processing module declares ``resumes`` (no default, like
``progress_counts_frames``). Registration turns it into the runner's ``cancel_status``: a
module that cannot resume records a cancel as ``cancelled``, whether the runner returned or
raised because the stop reached it. Runners registered outside ``register_processing_module``
keep the older mapping (see ``detail/executor/runners.py::RunnerSpec``). The detection half
(the stop reaching the inline loop) is
``tests/pipeline/test_detection.py::test_a_cancel_mid_part_reaches_the_inline_loop_and_records_cancelled``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    JobStatus,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.executor import rundir
from detail.executor.runners import (
    JOB_RUNNERS,
    PROCESSING_MODULES,
    RunnerSpec,
    import_runner_modules,
)
from detail.executor.worker import run_job

import_runner_modules(PIPELINE_RUNNERS)


def _declared_resumes(module) -> bool | None:
    for klass in type(module).__mro__:
        if "resumes" in vars(klass):
            return vars(klass)["resumes"]
    return None


@pytest.mark.parametrize("module_id", sorted(PROCESSING_MODULES))
def test_every_processing_module_says_whether_it_resumes(module_id):
    module = PROCESSING_MODULES[module_id]
    assert isinstance(_declared_resumes(module), bool), (
        f"{module_id} does not declare `resumes`: say whether a cancelled run of it can be "
        "resumed, so the worker knows whether a cancel is `partial` or `cancelled`"
    )


@pytest.mark.parametrize("module_id", sorted(PROCESSING_MODULES))
def test_a_module_that_cannot_resume_records_a_cancel_as_cancelled(module_id):
    module = PROCESSING_MODULES[module_id]
    if module.batch:
        pytest.skip("a batch module runs in the coordinator, not through the worker")
    expected = JobStatus.PARTIAL if module.resumes else JobStatus.CANCELLED
    assert JOB_RUNNERS[module_id].cancel_status == expected


def test_detection_cannot_resume():
    # The module docstring promises it, and the docs tell users `partial` is resumable.
    assert PROCESSING_MODULES["detection"].resumes is False


@contextlib.contextmanager
def _runner(run_fn, cancel_status):
    prior = JOB_RUNNERS.get("detection")
    JOB_RUNNERS["detection"] = RunnerSpec(
        run=run_fn,
        run_dir_for=lambda root, job: Path(root) / job.job_id,
        cancel_status=cancel_status,
    )
    try:
        yield
    finally:
        JOB_RUNNERS["detection"] = prior


def _init(tmp_path):
    run_dir = tmp_path / "det_01"
    run_dir.mkdir()
    rundir.init_run_dir(
        run_dir,
        DetectionJob(
            job_id="det_01",
            recording_id="recording_001",
            model_id="sam3",
            prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
            conf_threshold=0.5,
            compute=ComputeTarget("cpu", None, None, None),
            created_at="2026-01-01T00:00:00Z",
        ),
    )
    return run_dir


def _returns_after_cancel(job, rd, progress_cb, stop_event):
    rundir.request_cancel(rd)
    assert stop_event.wait(3.0)


def _raises_after_cancel(job, rd, progress_cb, stop_event):
    rundir.request_cancel(rd)
    assert stop_event.wait(3.0)
    raise RuntimeError("detection failed on 1 part(s): stop.flag detected")


@pytest.mark.parametrize("runner", [_returns_after_cancel, _raises_after_cancel])
def test_a_cancel_of_a_runner_without_resume_is_cancelled_however_it_ends(
    tmp_path, runner
):
    run_dir = _init(tmp_path)
    with _runner(runner, JobStatus.CANCELLED):
        assert run_job(run_dir) == JobStatus.CANCELLED
    assert rundir.read_status(run_dir) == JobStatus.CANCELLED


def test_a_runner_with_resume_still_records_a_cancel_partial(tmp_path):
    # The negative case: the mapping is per runner, not a blanket rewrite of `partial`.
    run_dir = _init(tmp_path)
    with _runner(_returns_after_cancel, JobStatus.PARTIAL):
        assert run_job(run_dir) == JobStatus.PARTIAL


def test_a_failure_with_no_cancel_is_still_failed(tmp_path):
    def boom(job, rd, progress_cb, stop_event):
        raise RuntimeError("boom")

    run_dir = _init(tmp_path)
    with _runner(boom, JobStatus.CANCELLED):
        assert run_job(run_dir) == JobStatus.FAILED
