"""The remote shepherd imports the runner modules before it decodes its job (§6).

**The class: a decode that needs an import the process never made.** A job type is decodable only
once the module that registers it has been imported. Detection, tracking, embedding and the rest
are registered by ``core.dataclasses`` itself; ``keypoint_detect`` (``keypoints`` before §26) registers on import of its own module. The local worker imports its runner
modules before it reads the job; the remote shepherd, a second entry point into the same decode,
read the job first, so every remote keypoints run failed in under a second with
``ValueError: unknown job type: 'keypoints'`` (a keypoints run worked locally with CUDA, which is
why nothing caught it). Detection and embedding worked remotely the whole time, which is what made
it look like keypoints' own problem.

**It hid because the shepherd tests only ever ran detection jobs**, and a test that decodes a job
inside the test process has every module already imported by the test session. So these tests do
the decode where the failure lives: the order is pinned inside ``run()``, and the registration
half is checked in a **fresh interpreter**, the way the shepherd subprocess starts.
"""

import json
import os
import subprocess
import sys

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.runners import RUNNER_MODULES_ENV

# Fixtures and helpers of the shepherd tests: the same experiment and fake transport.
from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeScheduler,
    _install_scheduler,
    experiment,
)
from detail.executor.schedulers.base import JobState


def test_run_imports_the_runner_modules_before_it_reads_the_job(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )
    monkeypatch.setenv(
        RUNNER_MODULES_ENV, "detail.pipeline.keypoint_detect,detail.models.sam3"
    )
    events = []
    monkeypatch.setattr(
        remote_shepherd,
        "import_runner_modules",
        lambda modules: events.append(("import", list(modules))),
    )
    read_job = rundir.read_job
    monkeypatch.setattr(
        rundir,
        "read_job",
        lambda directory: (events.append("read_job"), read_job(directory))[1],
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert events[0] == (
        "import",
        ["detail.pipeline.keypoint_detect", "detail.models.sam3"],
    )  # the executor's list, in this process, first
    assert "read_job" in events and events.index("read_job") > 0


def test_a_shepherd_started_with_no_module_list_still_runs(monkeypatch, experiment):
    """An empty or absent list is not an error (a test, a hand-run shepherd)."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )
    monkeypatch.delenv(RUNNER_MODULES_ENV, raising=False)

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert (
        rundir.read_envelope(run_dir)["status"] != "pending"
    )  # it got past the decode


def test_in_a_fresh_interpreter_keypoint_detect_is_unknown_until_its_module_is_imported():
    """The failure and the cure, in a process that starts the way the shepherd does."""
    code = (
        "import json, os\n"
        "from detail.core import dataclasses as d\n"
        "before = sorted(d._JOB_TYPES)\n"
        "from detail.executor.runners import RUNNER_MODULES_ENV, import_runner_modules\n"
        "import_runner_modules([m for m in os.environ[RUNNER_MODULES_ENV].split(',') if m])\n"
        "print(json.dumps({'before': before, 'after': sorted(d._JOB_TYPES)}))\n"
    )
    env = {**os.environ, RUNNER_MODULES_ENV: ",".join(PIPELINE_RUNNERS)}

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=True,
    )
    types = json.loads(result.stdout.strip().splitlines()[-1])

    assert (
        "keypoint_detect" not in types["before"]
    )  # a bare decode cannot read a keypoint_detect job
    assert "keypoint_detect" in types["after"]  # the executor's module list can
    late = sorted(set(types["after"]) - set(types["before"]))
    assert {"keypoint_detect"} <= set(late), late
