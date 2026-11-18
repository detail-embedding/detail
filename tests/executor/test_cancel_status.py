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
runner with a resume, and detection's module docstring says "stop is failure, never partial",
which is what its multi-GPU path does. The inline path never told the
part to stop (only the multi-GPU poll loop touched ``stop.flag``), so the run simply finished.

**The rule.** Every processing module declares ``resumes`` (no default, like
``progress_counts_frames``). Registration turns it into the runner's ``cancel_status``: a
module that cannot resume records a cancel as ``cancelled``, whether the runner returned or
raised because the stop reached it. Runners registered outside ``register_processing_module``
keep the older mapping (see ``detail/executor/runners.py::RunnerSpec``). The detection half
(the stop reaching the inline loop) is
``tests/pipeline/test_detection.py::test_a_cancel_mid_part_reaches_the_inline_loop_and_records_cancelled``.

**No processing module resumes.** Keypoint detect declared ``resumes = True`` and kept a skip
path past the last frame written, so its cancels landed ``partial``. Nothing could reach that
path: the only restart, ``Experiment.retry_run``, re-runs each incomplete recording whole under
a fresh stage run id (Tenet 12), so the runner always started on an empty store. A module's
``resumes`` is therefore ``False`` until a surface hands a stopped run back to its own run
directory; change ``test_no_processing_module_claims_a_resume`` in the same change that builds
one.
"""

from __future__ import annotations

import contextlib
import dataclasses
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
from detail.core.keypoints import KeypointDetectJob
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


@pytest.mark.parametrize("module_id", sorted(PROCESSING_MODULES))
def test_no_processing_module_claims_a_resume(module_id):
    # `retry_run` is the only restart and always mints a fresh stage run id, so no runner is
    # ever handed a run directory it already wrote to; `partial` would promise a resume that
    # no surface reaches (detection, #28; keypoint detect after it).
    assert PROCESSING_MODULES[module_id].resumes is False, (
        f"{module_id} declares `resumes = True`, but no surface resumes a processing run: "
        "`Experiment.retry_run` re-runs the recording under a fresh run id"
    )


def test_a_cancelled_keypoint_detect_run_is_cancelled_not_partial(tmp_path):
    # Through the worker with keypoint detect's registered spec, only its body swapped: the
    # real runner returns after a stop, which is the case that landed `partial`.
    spec = JOB_RUNNERS["keypoint_detect"]
    run_dir = tmp_path / "kd_01"
    run_dir.mkdir()
    rundir.init_run_dir(
        run_dir,
        KeypointDetectJob(
            job_id="kd_01",
            recording_id="recording_001",
            detect_run_id="kd_01",
            detection_run_id="det_01",
            min_pose_confidence=None,
            model_id="dinov3_match",
            model_variant="vit_base",
            backbone={"id": "dinov3_vit", "variant": "vit_base"},
            example_set={"id": "ex_01", "digest": "abc", "frames": [], "marks": []},
            skeleton={"names": ["nose", "tail"], "edges": [["nose", "tail"]]},
            n_candidates=5,
            every_n_frames=1,
            crop_px=448,
            decode="skeleton",
            side_rule="off",
            matching={"topk": 5},
            batch_size=16,
            compute=ComputeTarget("cpu", None, None, None),
            created_at="2026-01-01T00:00:00Z",
        ),
    )
    JOB_RUNNERS["keypoint_detect"] = dataclasses.replace(
        spec, run=_returns_after_cancel
    )
    try:
        assert run_job(run_dir) == JobStatus.CANCELLED
    finally:
        JOB_RUNNERS["keypoint_detect"] = spec
    assert rundir.read_status(run_dir) == JobStatus.CANCELLED


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
