"""Keypoints on a cluster (§26 x §6): what the shepherd has to know about one more job family.

**The class: a job family taught to the executor at several independent sites, one missed.**
Running a stage remotely is not one switch. The shepherd keeps a per-family phase event
(`_phase_event`) and a per-family model-readiness probe (`_model_readiness_probe`); the pipeline
module's `validate` has its own remote branch (skip this machine's weights and backend, name a
registered target); the CLI and the GUI each decide whether to offer `--backend remote`. A stage
that is wired at some of them *works* — it fails only on a cluster, only after the upload, with a
message about something else. Keypoints shipped with none of them wired and no failing test,
because every one of those functions answers "no" for a family it does not know, and "no" is a
legitimate answer for tracking. The guard below enumerates the job families rather than the
sites somebody remembered: every job that carries a compute target must be known to both
shepherd hooks, or be listed with the reason it is not.

It held through the §26 rebuild, which replaced the one ``keypoints`` family with
``keypoint_detect`` / ``keypoint_track`` / ``keypoint_dataset``: the enumeration found the new
families the day they were registered, where a list of sites would have kept naming the old one.
(A second class once lived here — a reference set claimed only on the cluster's copy. Example
sets are recorded by digest and never claimed, so that class cannot be written any more.)
"""

import contextlib
import dataclasses
import io
import json

import pytest

from detail.core import dataclasses as job_types
from detail.core.dataclasses import ComputeTarget, ModelReadiness
from detail.core.keypoints import KeypointDetectJob, KeypointProgressEvent
from detail.core.registries import KEYPOINT_MODELS
from detail.executor import remote_shepherd, rundir
from tests.pipeline.test_keypoint_detect import _FakeKeypointModel

#: Families whose ``model_id`` names a model *to create*, not one to check on the cluster.
_NO_READINESS_PROBE = {
    "training": "its model_id is the model this job creates; there is nothing installed to check"
}


def _compute_job_families():
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)  # the keypoint families register on import
    for cls, tag in sorted(job_types._JOB_TAGS.items(), key=lambda kv: kv[1]):
        if "compute" in {f.name for f in dataclasses.fields(cls)}:
            yield cls, tag


def _bare(cls, **attrs):
    """A job of ``cls`` with only ``attrs`` set: the shepherd's per-family hooks dispatch on the
    type (and read ``model_id``), so building a valid one per family is not needed to ask them.
    """
    job = object.__new__(cls)
    for name, value in attrs.items():
        object.__setattr__(job, name, value)
    return job


def test_the_enumeration_reaches_the_keypoint_families():
    tags = {tag for _cls, tag in _compute_job_families()}
    assert {"keypoint_detect", "keypoint_track", "keypoint_dataset"} <= tags, tags


@pytest.mark.parametrize("cls,tag", list(_compute_job_families()))
def test_every_job_that_takes_a_compute_target_is_known_to_the_shepherd(cls, tag):
    event = remote_shepherd._phase_event(
        _bare(cls), message="m", elapsed_s=0.0, eta_s=None
    )
    assert event is not None, (
        f"{cls.__name__} carries a compute target but remote_shepherd._phase_event has no "
        "event for it, so a remote run of it publishes no phase before the worker starts"
    )
    if "model_id" in {f.name for f in dataclasses.fields(cls)}:
        script = remote_shepherd._model_readiness_probe(
            _bare(cls, model_id="x", model_variant=None), "/scratch/exp", ""
        )
        if tag in _NO_READINESS_PROBE:
            assert script is None, f"{tag}: {_NO_READINESS_PROBE[tag]}"
        else:
            assert script is not None, (
                f"{cls.__name__} names a model but remote_shepherd._model_readiness_probe has "
                "no branch for it, so the cluster's install is never checked before the submit"
            )


def _job(**overrides) -> KeypointDetectJob:
    fields = dict(
        job_id="kd_01",
        recording_id="recording_001",
        detect_run_id="kd_01",
        detection_run_id="det_01",
        min_pose_confidence=None,
        model_id="fake_remote_kp",
        model_variant=None,
        backbone=None,
        example_set=None,
        skeleton={"names": ["nose", "tail"], "edges": [], "pairs": []},
        n_candidates=1,
        every_n_frames=1,
        crop_px=None,
        decode=None,
        side_rule=None,
        matching=None,
        batch_size=32,
        compute=ComputeTarget("remote", None, None, None, "tillicum"),
        created_at="2026-10-06T00:00:00Z",
    )
    fields.update(overrides)
    return KeypointDetectJob(**fields)


def test_a_keypoint_detect_phase_event_survives_the_envelope(tmp_path):
    job = _job()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rundir.init_run_dir(run_dir, job)
    event = remote_shepherd._phase_event(
        job, message="uploading experiment files", elapsed_s=12.0, eta_s=None
    )
    assert isinstance(event, KeypointProgressEvent)
    assert event.frames_total == 0  # no fraction invented before the stage starts
    rundir.set_progress(run_dir, event)
    assert rundir.read_progress(run_dir) == event


class _FakeRemoteKeypoints(_FakeKeypointModel):
    model_id = "fake_remote_kp"
    supported_backends = ["cuda"]

    def readiness(self):
        return ModelReadiness(
            ready=False,
            reason="not_installed",
            message="fake keypoint weights are not installed on this host",
            action_url=None,
            repo_id=None,
        )


def test_the_keypoint_detect_probe_resolves_through_the_keypoint_registry(
    monkeypatch,
):
    """Executed, not just compiled: a wrong-registry lookup fails at runtime, not at parse time
    (the embedding probe's original bug compiled fine)."""
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_remote_kp", _FakeRemoteKeypoints)
    script = remote_shepherd._model_readiness_probe(_job(), "/scratch/x/exp1", "")
    assert "resolve_detection_model" not in script
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(script, "<probe>", "exec"), {})
    payload = json.loads(buf.getvalue().strip())
    assert payload["ready"] is False and payload["reason"] == "not_installed"
