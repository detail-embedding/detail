"""A run is not recorded complete until its outputs are on disk.

The class and the incident are written up in ``tests/core/test_damaged_files.py``; this is the
write side. A detection run was copied back from a cluster (rsync, 1,062 files), recorded
**complete**, and a shutdown shortly after left every file the copy had created zero-filled
or empty, under a ``job.json`` that read complete. ``job.json`` survived only because
``atomic.replace`` flushes before it renames; the chunks and metadata the copy created were never
flushed by anything. Nothing in ``detail/executor`` called ``fsync``.

The fix is one line on the one seam: ``rundir.set_status`` is "the only place any job's status
ever changes", so ``COMPLETE`` flushes the run directory first. That covers

- a local run (``worker.py`` ends in ``set_status(COMPLETE)``), and
- a remote one: the shepherd copies the results back and *then* calls ``set_status`` on the
  local run directory (``remote_shepherd.py``), so the copy-back is flushed without the transport
  knowing anything about it.

Where the fix stops (``fix-the-class`` step 3.5): only ``COMPLETE`` flushes. A failed or
cancelled run's partial files are not promised to anyone, and flushing them would cost a ``fsync``
per file for outputs a retry replaces. The cost, once per run, is a ``fsync`` per output file.
"""

import json

import pytest

from detail.core import atomic
from detail.core.dataclasses import JobStatus
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState

from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeScheduler,
    _install_scheduler,
    experiment,
)
from tests.executor.test_rundir import _detection_job
from tests.executor.test_settled_end import COMPLETE, _reads


@pytest.fixture
def flushed(monkeypatch):
    """The paths ``flush_tree`` was asked to make durable, in order, with the run's own events."""
    events = []
    real = atomic._flush_to_disk

    def spy(path):
        events.append(("flush", str(path)))
        real(path)

    monkeypatch.setattr(atomic, "_flush_to_disk", spy)
    return events


def _outputs(run_dir):
    """What a stage leaves in its run directory besides the envelope."""
    for relative in (
        "part_001.zarr/zarr.json",
        "part_001.zarr/bbox/c/0/0",
        "activity.csv",
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"output " + relative.encode())


def _flushed_names(events):
    return {
        path.replace("\\", "/").rsplit("/", 1)[-1]
        for kind, path in events
        if kind == "flush"
    }


def test_completing_a_run_flushes_its_outputs(tmp_path, flushed):
    rundir.init_run_dir(tmp_path, _detection_job())
    _outputs(tmp_path)

    rundir.set_status(tmp_path, JobStatus.COMPLETE)

    assert {"zarr.json", "0", "activity.csv"} <= _flushed_names(flushed)


@pytest.mark.parametrize(
    "status", [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED]
)
def test_only_complete_pays_for_the_flush(tmp_path, flushed, status):
    rundir.init_run_dir(tmp_path, _detection_job())
    _outputs(tmp_path)

    rundir.set_status(tmp_path, status)

    assert "activity.csv" not in _flushed_names(flushed)


def test_the_flush_comes_before_the_record_says_complete(
    tmp_path, monkeypatch, flushed
):
    """The order is the point: a record that says complete over files still in the cache is the
    bug. The envelope write is the moment the run reads complete."""
    rundir.init_run_dir(tmp_path, _detection_job())
    _outputs(tmp_path)
    real_write = rundir._atomic_write

    def write(path, envelope):
        if envelope.get("status") == "complete":
            flushed.append(("complete-written", str(path)))
        real_write(path, envelope)

    monkeypatch.setattr(rundir, "_atomic_write", write)

    rundir.set_status(tmp_path, JobStatus.COMPLETE)

    kinds = [kind for kind, _ in flushed]
    assert "complete-written" in kinds
    # writing the envelope flushes its own temp file after this point; what has to precede it
    # is every output
    before = flushed[: kinds.index("complete-written")]
    assert {"zarr.json", "0", "activity.csv"} <= _flushed_names(before)


def test_a_remote_runs_copy_back_is_flushed_before_it_is_complete(
    monkeypatch, experiment, flushed
):
    """The incident's path, through the real shepherd: results arrive by ``copy_from_remote`` and
    nothing flushes them but the run completing. Every file the copy created is flushed.
    """
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, COMPLETE)

    def copy_back(host, remote_dir, local_dir, **kwargs):
        flushed.append(("copy", str(local_dir)))
        _outputs(local_dir)
        (local_dir / "job.json").write_text(
            json.dumps({"schema_version": 1, "status": "complete", "error": None}),
            encoding="utf-8",
        )

    monkeypatch.setattr(ssh_transport, "copy_from_remote", copy_back)

    assert (
        remote_shepherd.run(run_dir, "tillicum", experiment_root) == JobStatus.COMPLETE
    )

    kinds = [kind for kind, _ in flushed]
    assert "copy" in kinds
    after_copy = {
        path.replace("\\", "/").rsplit("/", 1)[-1]
        for kind, path in flushed[kinds.index("copy") :]
        if kind == "flush"
    }
    assert {"zarr.json", "0", "activity.csv"} <= after_copy


def test_a_failed_copy_back_flushes_nothing_it_did_not_bring(
    monkeypatch, experiment, flushed
):
    """A run whose results never arrived is failed, not complete, so it is not flushed."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, COMPLETE)

    def refused(host, remote_dir, local_dir, **kwargs):
        raise ssh_transport.RemoteCommandError(["rsync"], 255, "", "refused")

    monkeypatch.setattr(ssh_transport, "copy_from_remote", refused)

    assert remote_shepherd.run(run_dir, "tillicum", experiment_root) == JobStatus.FAILED
    assert "activity.csv" not in _flushed_names(flushed)
