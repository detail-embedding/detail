"""A run kind must have the whole lifecycle, not most of it.

CLAUDE.md already states the rule — "a lesson learned in one module has to reach the next
one". These are two places it wasn't followed:

- `BatchProcessingModule` swept `output_path` in with the per-recording *job* seams it
  disables, so deleting a `derive` run raised `NotImplementedError` out of a Qt slot and
  the Qt binding of the time aborted the whole app. `output_path` is not a job seam: it says where a recording's
  output lives, and a batch module writes one per recording.
- `reconcile_runs` / `reconcile_analysis_runs` ran on experiment open while dataset,
  model and clusterer runs had no equivalent — so a SIGKILLed worker left a record reading
  `running` forever, which every delete method then refuses.

Both guards are registry-driven, so a fifth run kind or a second batch module inherits them.
"""

import json
import subprocess
import sys

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import Experiment
from detail.executor.runners import PROCESSING_MODULES, import_runner_modules

# At *import* time, not in a fixture: @parametrize is evaluated during collection, so a
# registry populated by a fixture yields an empty parameter set and pytest skips the test —
# a registry guard that silently never runs is worse than no guard at all.
import_runner_modules(PIPELINE_RUNNERS)


def test_the_module_registry_is_populated():
    """Guard the guard: the parametrized tests below are vacuous if this is empty."""
    assert len(PROCESSING_MODULES) >= 4


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# --------------------------------------------------------------------------- #
# every processing module can say where a recording's output lives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module_id", sorted(PROCESSING_MODULES))
def test_every_module_answers_output_path(module_id):
    """Batch or not. `Experiment.delete_run` needs it for *every* module, and a batch
    module that cannot answer it took the app down at delete time."""
    module = PROCESSING_MODULES[module_id]
    path = module.output_path("x_01")
    assert path.endswith("/") and "x_01" in path


def test_a_batch_module_is_still_denied_the_job_seams():
    """The narrowing must not go too far: the per-recording *job* seams stay dead, since
    calling one on a batch module really is a wiring bug."""
    from detail.executor.runners import BatchProcessingModule

    class _Batch(BatchProcessingModule):
        module_id = "_batch_probe"

        def output_path(self, run_id):
            return f"probe/{run_id}/"

        def validate_batch(self, experiment_root, recording_ids, config):
            return config

        def run_batch(self, experiment, recording_ids, config, emit_progress, cancel):
            return "x_01"

    module = _Batch()
    for seam in ("validate", "build_job", "runner", "run_dir_for"):
        with pytest.raises(NotImplementedError):
            getattr(module, seam)("anything")


def test_deleting_a_batch_module_run_removes_its_outputs(tmp_path):
    """End to end on real files: the crash was in this path.

    The record is a ``derive`` run's, and the ``derive`` module has since been removed
    (``docs/removed_capabilities.md``). The records it wrote are still in experiments, so
    this is now also the check that one can be listed and deleted by a version that cannot
    make another: its output is an ordinary ``emb_NN``, found through the embedding module.
    """
    from detail.executor.runners import PROCESSING_MODULES

    assert "derive" not in PROCESSING_MODULES
    from detail.storage import recording as recording_store
    from detail.storage import runs as runs_store

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("recording_001")
    out = recording.dir / "embeddings" / "emb_02"
    out.mkdir(parents=True)
    (out / "embedding.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    data = recording_store.read_recording(recording.dir)
    data["embedding_runs"] = [{"id": "emb_02", "status": "complete"}]
    recording_store.write_recording(recording.dir, data)
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "derive",
            "recording_ids": ["recording_001"],
            "per_recording": {
                "recording_001": {"run_id": "emb_02", "status": "complete"}
            },
            "config": {},
            "status": "complete",
            "created_at": "2026-08-01T10:00:00Z",
        },
    )

    assert [r["module_id"] for r in proj.processing_runs] == ["derive"]
    proj.delete_run("run_001")  # used to raise NotImplementedError

    assert not out.exists()
    assert [r["id"] for r in proj.processing_runs] == []


# --------------------------------------------------------------------------- #
# every single-job run kind gets the SIGKILL backstop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", sorted(Experiment._JOB_RUN_KINDS))
def test_one_unreadable_record_does_not_make_the_experiment_unopenable(tmp_path, kind):
    """**The class.** *A per-item sweep with no per-item isolation.*

    `reconcile_job_runs` runs on every experiment open and caught only `FileNotFoundError`, so
    any record it could not *parse* propagated out of `_open_experiment` — and the project could
    no longer be opened at all. There is then no way back to the file from inside the app, which
    is what turns a bad byte into a dead project.

    Reported exactly that way: a record written with a `!!python/tuple` in it (the bug in the
    commit before this one) meant "now I can't even open a project". The tuple was one cause;
    a torn write, a hand-edit, or a field a future version stores are others, and the reconciler
    cannot tell them apart or fix any of them.

    `api/active.py::_safe` already makes this trade one level up and says why. This sweep never
    learned it.
    """
    proj = Experiment.create("p", tmp_path)
    run_dir_for, _read, _set, _discover = Experiment._JOB_RUN_KINDS[kind]
    _stranded(proj, kind, "x_01")  # a healthy record beside it
    bad_dir = run_dir_for(proj.dir, "x_02")
    bad_dir.mkdir(parents=True, exist_ok=True)
    marker = next(p for p in bad_dir.parent.glob("x_01/*.yaml"))
    (bad_dir / marker.name).write_text(
        "schema_version: 1\nid: x_02\noptions:\n  renders: !!python/tuple\n  - overlay\n",
        encoding="utf-8",
    )

    proj.reconcile_job_runs()  # must not raise — this is what opening an experiment calls

    # And the healthy neighbour is still reconciled: skipping the bad one must not skip the rest.
    assert _read(run_dir_for(proj.dir, "x_01")).get("status") is not None


def _stranded(proj, kind, run_id="x_01"):
    """A record reading `running` whose worker is gone — a SIGKILLed job."""
    run_dir_for, _read, _set, _discover = Experiment._JOB_RUN_KINDS[kind]
    writers = {
        "analysis": "write_analysis_yaml",
        "dataset": "write_dataset_yaml",
        "model": "write_model_yaml",
        "export": "write_export_yaml",
    }
    from detail.storage import analysis_runs, datasets, export_runs, models

    modules = {
        "analysis": analysis_runs,
        "dataset": datasets,
        "model": models,
        "export": export_runs,
    }
    run_dir = run_dir_for(proj.dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    getattr(modules[kind], writers[kind])(run_dir, {"id": run_id, "status": "running"})
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": None,
                "pid": _dead_pid(),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


@pytest.mark.parametrize("kind", sorted(Experiment._JOB_RUN_KINDS))
def test_a_killed_worker_of_any_kind_reconciles_on_open(tmp_path, kind):
    proj = Experiment.create("p", tmp_path)
    run_dir = _stranded(proj, kind)
    _dir_for, read_record, _set, _discover = Experiment._JOB_RUN_KINDS[kind]

    proj.reconcile_job_runs()  # what the GUI calls when it opens an experiment

    record = read_record(run_dir)
    assert record["status"] == "failed"
    assert record.get("error")  # says why, not just that


@pytest.mark.parametrize("kind", sorted(Experiment._JOB_RUN_KINDS))
def test_reconcile_leaves_a_live_worker_of_any_kind_alone(tmp_path, kind):
    import os

    proj = Experiment.create("p", tmp_path)
    run_dir = _stranded(proj, kind)
    envelope = json.loads((run_dir / "job.json").read_text(encoding="utf-8-sig"))
    envelope["pid"] = os.getpid()  # this process is very much alive
    (run_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")
    _dir_for, read_record, _set, _discover = Experiment._JOB_RUN_KINDS[kind]

    proj.reconcile_job_runs()

    assert read_record(run_dir)["status"] == "running"


def test_the_gui_reconciles_every_record_kind_on_open():
    """Source-level, so a fourth reconciler added later is noticed here."""
    from pathlib import Path

    import detail

    source = (Path(detail.__file__).parent / "gui" / "app.py").read_text(
        encoding="utf-8-sig"
    )
    for call in (
        "reconcile_imports()",
        "reconcile_runs()",
        "reconcile_job_runs()",
        "reconcile_recording_runs()",
    ):
        assert call in source, f"experiment open must call {call}"


# --------------------------------------------------------------------------- #
# a runner's optimistic status does not outlive the job that wrote it
# --------------------------------------------------------------------------- #
# **The class.** *A status written before the work that justifies it, with the reconciler
# keyed on the status it was supposed to have.* An authored analysis run (§21) writes
# `status: open` into `analysis.yaml` *before* projecting its artifact, deliberately — an
# open run is supposed to have one. Every reconciler in the package then read
# `status in (None, "running")`, on the stated assumption that "a crash leaves running".
# True for eight analysis modules; silently inapplicable to the two that write a
# terminal-looking status mid-run, which are the ones that need it.
#
# The damage was not a wrong label. `job.json` said `failed`, `analysis.yaml` said `open`,
# and every surface reads the record: the run listed beside healthy ones, the GUI opened its
# interpretation workspace, and the workspace's first act on a run with no granularity is to
# commit one — which reprojects, which re-ran the whole artifact write that had just
# killed the run, unattended, on the share it died on.
#
# The rule is `core/dataclasses.py::reconciled_status`: a record may claim an outcome the
# envelope has not reached, but never a *success* the envelope contradicts.
from detail.core.dataclasses import (  # noqa: E402
    ACTIVE_STATUSES,
    AUTHORED_STATUSES,
    OUTPUT_STATUSES,
)
from detail.executor.runners import ANALYSIS_MODULES  # noqa: E402


def _with_envelope(proj, kind, record_status, job_status, run_id="x_01"):
    """A run whose record says one thing and whose job envelope says another."""
    run_dir = _stranded(proj, kind, run_id)
    _dir_for, _read, set_status, _discover = Experiment._JOB_RUN_KINDS[kind]
    set_status(run_dir, record_status)
    envelope = json.loads((run_dir / "job.json").read_text(encoding="utf-8-sig"))
    envelope["status"] = job_status
    envelope["error"] = "PermissionError: [WinError 32] the artifact write died"
    (run_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("kind", sorted(Experiment._JOB_RUN_KINDS))
@pytest.mark.parametrize(
    "record_status", sorted(set(ACTIVE_STATUSES + OUTPUT_STATUSES + AUTHORED_STATUSES))
)
def test_a_failed_job_overrides_every_non_failure_record_status(
    tmp_path, kind, record_status
):
    """Whatever the runner claimed, a failed envelope wins — and says why."""
    proj = Experiment.create("p", tmp_path)
    run_dir = _with_envelope(proj, kind, record_status, "failed")
    _dir_for, read_record, _set, _discover = Experiment._JOB_RUN_KINDS[kind]

    proj.reconcile_job_runs()

    record = read_record(run_dir)
    assert record["status"] == "failed"
    assert "WinError 32" in str(record.get("error")), "it carries the envelope's reason"


@pytest.mark.parametrize("record_status", sorted(AUTHORED_STATUSES))
def test_a_complete_job_leaves_an_authored_record_open(tmp_path, record_status):
    """The other half, and the reason the rule is asymmetric rather than "envelope wins".

    An authored run whose compute finished perfectly is still `open`: §21 makes finishing a
    *user's* statement. A reconciler that adopted the envelope unconditionally would close
    every authored run the moment its fit landed.
    """
    proj = Experiment.create("p", tmp_path)
    run_dir = _with_envelope(proj, "analysis", record_status, "complete")

    proj.reconcile_job_runs()

    from detail.storage import analysis_runs

    assert analysis_runs.read_analysis_yaml(run_dir)["status"] == record_status


@pytest.mark.parametrize("module_id", sorted(ANALYSIS_MODULES))
def test_no_analysis_module_reports_a_non_failed_status_after_a_failed_job(
    tmp_path, module_id
):
    """Registry-parity: a module added tomorrow is covered the day it is added.

    Each module's runner is simulated at the point it has written its record and then died —
    which for the authored ones is a *terminal-looking* record, and that is the whole defect.
    Asserted through `Experiment.analysis_runs`, the listing every surface is built on, not
    through the reconciler that was just called.
    """
    from detail.storage import analysis_runs

    module = ANALYSIS_MODULES[module_id]
    mid_run = "open" if getattr(module, "mutable", False) else "complete"
    proj = Experiment.create("p", tmp_path)
    run_dir = analysis_runs.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_runs.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": module_id, "status": mid_run}
    )
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "job": {"job_id": "an_01", "config": {}},
                "progress": None,
                "error": "PermissionError: [WinError 32]",
            }
        ),
        encoding="utf-8",
    )

    proj.reconcile_analysis_runs()

    ref = next(r for r in proj.analysis_runs if r.id == "an_01")
    assert ref.status == "failed", (
        f"{module_id} reports {ref.status!r} for a run whose job failed — a surface reading "
        "this offers a dead run's results"
    )


# --------------------------------------------------------------------------- #
# a processing run's status is rolled up over its plan (issue #28, finding 1)
# --------------------------------------------------------------------------- #
#
# **The class: a roll-up over the entries present instead of over the plan.** A processing
# run's `per_recording` gains an entry only when the coordinator launches that recording, so a
# run cancelled after its first recording holds one entry, and `all(s == "complete")` over it
# is true. `cancel_run` rolled up that way (and also killed the coordinator, removing the only
# other writer), so the run read `complete` and a dataset could be built from one of three
# recordings. `_finalize` guarded it with a length check and `_reconcile_run` by filling the
# missing entries — three copies, one of them wrong. The rule: one function,
# `roll_up_run_status`, over `recording_ids`; the scan below keeps it the only one.
def _three_recording_run(tmp_path, module_id):
    from detail.storage import runs as runs_store

    proj = Experiment.create("p", tmp_path)
    ids = [proj.create_recording(f"rec{i}").id for i in range(3)]
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": module_id,
            "recording_ids": ids,
            "config": {},
            "per_recording": {ids[0]: {"run_id": "x_01", "status": "complete"}},
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    return proj, ids


@pytest.mark.parametrize("module_id", sorted(PROCESSING_MODULES))
def test_a_cancel_between_recordings_is_cancelled_and_lists_the_unrun(
    tmp_path, module_id
):
    from detail.storage import runs as runs_store

    proj, ids = _three_recording_run(tmp_path, module_id)
    proj.cancel_run("run_001")

    record = runs_store.read_run(proj.dir, "run_001")
    assert record["status"] == "cancelled"
    assert record["per_recording"][ids[0]]["status"] == "complete"
    for sid in ids[1:]:
        assert record["per_recording"][sid]["status"] == "cancelled"
        assert record["per_recording"][sid]["run_id"] is None


@pytest.mark.parametrize("module_id", sorted(PROCESSING_MODULES))
def test_a_dead_coordinator_with_unlaunched_recordings_reconciles_failed(
    tmp_path, module_id
):
    from detail.storage import runs as runs_store

    proj, ids = _three_recording_run(tmp_path, module_id)
    runs_store.set_coordinator_pid(proj.dir, "run_001", _dead_pid())
    proj._reconcile_run(runs_store.read_run(proj.dir, "run_001"))

    record = runs_store.read_run(proj.dir, "run_001")
    assert record["status"] == "failed"
    assert [record["per_recording"][s]["status"] for s in ids] == [
        "complete",
        "failed",
        "failed",
    ]


def _complete_rollups(source: str) -> list[int]:
    """Lines holding an ``all(<x> == "complete" | COMPLETE …)`` roll-up."""
    import ast

    found = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "all"):
            continue
        text = ast.unparse(node)
        if "== 'complete'" in text or "COMPLETE" in text:
            found.append(node.lineno)
    return found


def test_the_run_status_roll_up_lives_in_one_function():
    import ast
    import pathlib

    import detail.api.experiment as experiment_module

    source = pathlib.Path(experiment_module.__file__).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    owners = set()
    for line in _complete_rollups(source):
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.lineno <= line <= fn.end_lineno:
                owners.add(fn.name)
    assert owners == {"roll_up_run_status"}, owners


def test_the_roll_up_scan_would_catch_a_second_copy():
    assert _complete_rollups('ok = all(s == "complete" for s in statuses)\n') == [1]
    assert _complete_rollups("ok = all(s == JobStatus.COMPLETE.value for s in xs)\n")
    assert _complete_rollups("ok = all(s == 'failed' for s in xs)\n") == []
