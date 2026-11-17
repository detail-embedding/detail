"""A run's recorded status says what the work did — whoever stopped watching it.

**The class (issue #28).** *A run's status computed from part of the evidence, or written only
by the process that happened to be watching, or changed under a live run.* The roll-up half is
guarded in ``tests/api/test_run_lifecycle_parity.py``; this file holds the other shapes, all at
record level (no executor run):

- **Adopted only by a watcher.** ``_RecordedRun.wait()`` was the only writer of a one-recording
  run's terminal status into ``recording.yaml``. Ctrl-C (``cancel()`` then abort), a closed
  terminal, or a script that never waited left the entry ``pending`` forever, and the
  recording refused every later run of the stage. Now ``cancel()`` adopts too, and
  ``Experiment.reconcile_recording_runs`` — driven from ``PROCESSING_MODULES`` — adopts on
  every open, from the GUI and from every CLI command (``cli._open_experiment``).
- **The plan changed under a live run.** ``delete_recording`` deleted a recording a live
  processing run was working on (orphaning its worker and wiping the run record), and left a
  deleted id in a run's ``recording_ids`` so ``process retry`` could never succeed.
  ``run_processing`` ran a recording named twice twice, leaving an unindexed output and a
  ``failed`` batch.
- **An interrupted import never marked.** ``import_data`` caught ``Exception``, so a Ctrl-C
  (``KeyboardInterrupt``) left the recording ``importing`` and the rest ``queued`` until a GUI
  happened to open the experiment.

Each test here was run against the unfixed code and failed; the PR for #28 lists how.
"""

import json
import subprocess
import sys

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import Experiment
from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
from detail.storage import recording as recording_store
from detail.storage import runs as runs_store

import_runner_modules(PIPELINE_RUNNERS)

PER_RECORDING = sorted(
    m for m, mod in PROCESSING_MODULES.items() if not getattr(mod, "batch", False)
)


def test_the_per_recording_registry_is_populated():
    """Guard the guard: the parametrized tests below are vacuous if this is empty."""
    assert len(PER_RECORDING) >= 4


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _pending_entry(proj, module_id, job_status, run_id="x_01"):
    """A recording whose ``recording.yaml`` says ``pending`` while ``job.json`` says
    ``job_status`` and the worker is gone — a run nobody waited on."""
    module = PROCESSING_MODULES[module_id]
    recording = proj.create_recording("rec")
    data = recording_store.read_recording(recording.dir)
    data.setdefault(module.runs_key, []).append({"id": run_id, "status": "pending"})
    recording_store.write_recording(recording.dir, data)
    out = recording.dir / module.output_path(run_id).rstrip("/")
    out.mkdir(parents=True)
    (out / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": job_status,
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": None,
                "pid": _dead_pid(),
            }
        ),
        encoding="utf-8",
    )
    return recording, module


def _entry_status(recording, module, run_id="x_01"):
    data = recording_store.read_recording(recording.dir)
    return next(e for e in data[module.runs_key] if e["id"] == run_id)["status"]


@pytest.mark.parametrize("job_status", ["cancelled", "complete", "running"])
@pytest.mark.parametrize("module_id", PER_RECORDING)
def test_a_run_nobody_waited_on_is_adopted_on_open(tmp_path, module_id, job_status):
    proj = Experiment.create("p", tmp_path)
    recording, module = _pending_entry(proj, module_id, job_status)

    proj.reconcile_recording_runs()

    # A job still "running" with a dead worker is a crash: failed, like every reconciler.
    want = "failed" if job_status == "running" else job_status
    assert _entry_status(recording, module) == want


def test_a_live_worker_is_left_running(tmp_path, monkeypatch):
    """A job still running under a live worker is not adopted — only evidence of an end is."""
    import detail.api.experiment as exp_mod

    proj = Experiment.create("p", tmp_path)
    recording, module = _pending_entry(proj, "detection", "running")
    monkeypatch.setattr(exp_mod, "worker_alive", lambda run_dir: True)

    proj.reconcile_recording_runs()

    assert _entry_status(recording, module) == "pending"


def test_every_cli_command_adopts_on_open(tmp_path):
    """The CLI's one door to an experiment runs the per-recording reconcile."""
    from detail import cli

    proj = Experiment.create("p", tmp_path)
    recording, module = _pending_entry(proj, "detection", "cancelled")

    cli._open_experiment(proj.dir)

    assert _entry_status(recording, module) == "cancelled"


def test_cancel_adopts_without_a_wait(tmp_path):
    """The CLI's Ctrl-C path: ``cancel()`` and then abort — no ``wait()`` follows."""
    from detail.api.recording import _RecordedRun

    proj = Experiment.create("p", tmp_path)
    recording, module = _pending_entry(proj, "detection", "running")

    class _Inner:  # the executor handle: a cancel lands the worker on "cancelled"
        status = "running"

        def cancel(self):
            self.status = "cancelled"

    _RecordedRun(_Inner(), recording, "x_01", module.runs_key).cancel()

    assert _entry_status(recording, module) == "cancelled"


# --------------------------------------------------------------------------- #
# the plan does not change under a live run
# --------------------------------------------------------------------------- #
def _run(proj, run_id, ids, per, status):
    runs_store.write_run(
        proj.dir,
        run_id,
        {
            "id": run_id,
            "module_id": "detection",
            "recording_ids": ids,
            "config": {},
            "per_recording": per,
            "status": status,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )


def test_a_recording_a_live_run_is_using_cannot_be_deleted(tmp_path):
    proj = Experiment.create("p", tmp_path)
    a, b = (proj.create_recording(n).id for n in ("a", "b"))
    # Running on `a`; its coordinator pid not yet recorded reads as "just launched".
    _run(
        proj,
        "run_001",
        [a, b],
        {a: {"run_id": "det_01", "status": "running"}},
        "running",
    )

    with pytest.raises(ValueError, match="run_001"):
        proj.delete_recording(a)
    with pytest.raises(ValueError, match="run_001"):
        proj.delete_recording(b)  # planned, not yet started: still the run's

    assert (proj.dir / "runs" / "run_001").is_dir()
    assert {r.id for r in proj.recordings} == {a, b}


def test_a_run_whose_coordinator_is_gone_does_not_block_a_delete(tmp_path):
    proj = Experiment.create("p", tmp_path)
    a, b = (proj.create_recording(n).id for n in ("a", "b"))
    _run(proj, "run_001", [a, b], {a: {"run_id": None, "status": "running"}}, "running")
    runs_store.set_coordinator_pid(proj.dir, "run_001", _dead_pid())

    proj.delete_recording(a)  # reconciled to failed first, so it no longer blocks

    assert [r.id for r in proj.recordings] == [b]


def test_a_recording_with_its_own_live_run_cannot_be_deleted(tmp_path):
    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("a")
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [
        {"id": "det_01", "status": "pending"}
    ]  # launch in progress
    recording_store.write_recording(recording.dir, data)

    with pytest.raises(ValueError, match="det_01"):
        proj.delete_recording(recording.id)


def test_a_deleted_recording_leaves_every_run_plan(tmp_path):
    proj = Experiment.create("p", tmp_path)
    a, b, c = (proj.create_recording(n).id for n in ("a", "b", "c"))
    # `a` was never run (cancelled before launch): only in the plan, not per_recording.
    _run(
        proj,
        "run_001",
        [a, b, c],
        {b: {"run_id": "det_01", "status": "complete"}},
        "cancelled",
    )
    proj.delete_recording(a)
    proj.delete_recording(b)

    record = runs_store.read_run(proj.dir, "run_001")
    assert record["recording_ids"] == [c]
    assert record["per_recording"] == {}


def test_the_same_recording_named_twice_runs_once(tmp_path, monkeypatch):
    import detail.api.experiment as exp_mod

    proj = Experiment.create("p", tmp_path)
    a, b = (proj.create_recording(n).id for n in ("a", "b"))
    module = PROCESSING_MODULES["tracking"]
    monkeypatch.setattr(module, "validate", lambda sdir, data, config: config)
    monkeypatch.setattr(exp_mod, "spawn_coordinator", lambda *a, **k: _dead_pid())

    run = proj.run_processing("tracking", [a, b, a], {})

    assert runs_store.read_run(proj.dir, run.id)["recording_ids"] == [a, b]


# --------------------------------------------------------------------------- #
# an interrupted import is marked on the way out
# --------------------------------------------------------------------------- #
def test_a_ctrl_c_during_an_import_leaves_nothing_queued_or_importing(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from detail.api import imports
    from detail.api.recording import Recording

    proj = Experiment.create("p", tmp_path)
    specs = [
        imports.RecordingImportSpec(name=n, video_parts=[Path("/nowhere.mp4")])
        for n in ("a", "b")
    ]

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Recording, "add_video_part", interrupted)
    monkeypatch.setattr(imports, "transcode_parallelism", lambda: 1)
    with pytest.raises(KeyboardInterrupt):
        imports.import_specs(proj, specs)

    assert [r.import_status for r in proj.recordings] == ["cancelled", "cancelled"]


def test_an_active_distill_run_cannot_be_deleted(tmp_path):
    """The same shape one family over: a live coordinator's record deleted under it."""
    from detail.storage import distill_runs as distill_store

    proj = Experiment.create("p", tmp_path)
    distill_store.write_run(
        proj.dir,
        "distill_001",
        {"id": "distill_001", "status": "running", "stages": {}, "config": {}},
    )  # no coordinator pid yet: a run just launched

    with pytest.raises(ValueError, match="still active"):
        proj.delete_distill("distill_001")
    assert distill_store.read_run(proj.dir, "distill_001")["status"] == "running"
