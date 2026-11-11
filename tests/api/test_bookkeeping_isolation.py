"""A failure in *reporting on* work is never attributed to the work itself.

**The class.** A run's coordinator does two different kinds of thing: it drives the work, and it
writes files *about* the work — the aggregate ``runs/<id>/progress.json``, the run record, the
per-job envelope it polls. In ``RunOrchestrator._run_recording`` both lived inside one ``try``
under one ``except Exception``, so a transient Windows ``PermissionError`` on the coordinator's
own progress rename was recorded as *the recording* failing (#289).

That is bad on its own and it is not the worst of it. The handler sits **before** ``run.wait()``,
which is the only call that copies a job's terminal status into ``recording.yaml``. So one failed
bookkeeping write:

1. marked a healthy recording ``failed`` with a file-permission error as the reason;
2. **orphaned its worker** — nothing cancelled or waited on the subprocess, so it ran on,
   finished successfully outside the run's tracking, and competed for the single GPU with the
   recording the coordinator had moved on to (measurably slowing the batch);
3. left ``recording.yaml`` non-terminal forever, which blocks tracking, embedding, export *and*
   retry (``pipeline/detection.py``'s validate refuses while an entry is active).

**Why retries were not the answer, and why this file exists.** ``core/atomic.py`` has retried
``os.replace`` twenty times at 10 ms since well before the incident, and the error still escaped.
``tests/core/test_atomic.py::test_retry_on_lock_reraises_after_exhausting`` proves the suite
*knows* a bounded retry can be exhausted — and every one of the ~21 ``PermissionError``
injections in this repo is at the storage leaf, asserting the retry rides it out. Nobody had ever
asked what the layer above does when it does not. Hardening an operation says nothing about the
caller's contract when the hardening runs out.

So these tests inject at the **seam**, not the leaf, and assert the property rather than the
shape: *a bookkeeping failure changes no work outcome and skips no status adoption.* The
distinction the whole change rests on is in
:func:`test_a_real_launch_failure_still_fails_the_recording` — a fix that simply swallowed
everything would pass every other test here.
"""

import json

import pytest

from detail.api.experiment import Experiment, RunOrchestrator
from detail.api.recording import Recording
from detail.core.dataclasses import DetectionProgressEvent
from detail.storage import runs as runs_store

#: What Windows raised in #289, verbatim enough to be recognisable in a failure message.
WIN_ACCESS_DENIED = PermissionError(5, "Access is denied")


def _event(done: int, total: int) -> DetectionProgressEvent:
    """A well-formed worker event. Built through the real dataclass, not a stub: a fake that
    does not satisfy the schema fails these tests for its own reasons and hides theirs.
    """
    return DetectionProgressEvent(
        parts_done=0,
        parts_total=1,
        frames_done=done,
        frames_total=total,
        elapsed_s=0.0,
        eta_s=None,
        message="detecting",
    )


class _FakeRun:
    """A per-recording job handle that behaves like a healthy worker.

    ``progress()`` yields real events (so the coordinator's emit is exercised) and ``wait()``
    records that it ran — the call whose absence orphans a worker in production.
    """

    def __init__(self, run_id: str = "det_01", events: int = 3) -> None:
        self.id = run_id
        self.status = "complete"
        self.error = None
        self.waited = False
        self._events = events

    def progress(self):
        for i in range(self._events):
            yield _event(i + 1, self._events)

    def wait(self):
        self.waited = True
        return self


def _raise_on(nth: int, exc: BaseException = WIN_ACCESS_DENIED):
    """A callable that raises ``exc`` on its ``nth`` call (1-based) and returns None otherwise."""
    seen = {"n": 0}

    def call(*_args, **_kwargs):
        seen["n"] += 1
        if seen["n"] == nth:
            raise exc

    call.calls = seen
    return call


@pytest.fixture
def orchestrated(tmp_path, monkeypatch):
    """An experiment + a `RunOrchestrator` over one recording, with a fake worker.

    Built in-process (no coordinator subprocess): `RunOrchestrator` is directly constructible
    and `_run_recording` directly callable, the idiom in
    `test_experiment.py::test_coordinator_restores_typed_config_before_recording_run`.
    """

    def build(emit=lambda _p: None, run=None):
        experiment = Experiment.create("p", tmp_path)
        experiment.create_recording("recording_001")
        handle = _FakeRun() if run is None else run
        monkeypatch.setattr(Recording, "run", lambda self, mid, cfg: handle)
        orchestrator = RunOrchestrator(
            experiment,
            "run_001",
            "detection",
            ["recording_001"],
            {},
            emit,
            "2026-01-01T00:00:00Z",
        )
        return experiment, orchestrator, handle

    return build


def _entry(experiment, sid: str = "recording_001") -> dict:
    return runs_store.read_run(experiment.dir, "run_001")["per_recording"][sid]


# --------------------------------------------------------------------------- #
# the property: bookkeeping cannot fail the work, and cannot skip adoption
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nth", [1, 2, 3])
def test_a_failed_progress_write_does_not_fail_the_recording(orchestrated, nth):
    """#289 exactly: the coordinator's own `progress.json` rename raises mid-run.

    Parametrised over *which* emit fails, because the first one happens before any worker
    event and the later ones mid-stream — different lines, one consequence.
    """
    emit = _raise_on(nth)
    experiment, orchestrator, handle = orchestrated(emit=emit)

    orchestrator._run_recording("recording_001")  # the boundary records, never raises

    assert emit.calls["n"] >= nth, "the emit under test was never reached"
    entry = _entry(experiment)
    assert entry["status"] == "complete", (
        "a failed progress write was recorded as the recording failing: "
        f"{entry.get('error')!r}"
    )
    assert handle.waited, (
        "run.wait() was skipped — the worker is orphaned and recording.yaml never gets "
        "its terminal status"
    )


def test_a_failed_run_record_write_does_not_fail_the_recording(
    orchestrated, monkeypatch
):
    """The other bookkeeping write in the same `try`, one line earlier.

    `_write_record` persists the run's `per_recording` map right after launch. It is the same
    kind of thing as the progress file — a statement *about* the work — and it was inside the
    same handler, so it could fail a recording whose worker had just started cleanly.
    """
    experiment, orchestrator, handle = orchestrated()
    real = runs_store.write_run
    calls = {"n": 0}

    def flaky(experiment_dir, run_id, record):
        calls["n"] += 1
        if calls["n"] == 1:  # the write inside _run_recording, before the progress loop
            raise WIN_ACCESS_DENIED
        return real(experiment_dir, run_id, record)

    monkeypatch.setattr(runs_store, "write_run", flaky)

    orchestrator._run_recording("recording_001")

    assert calls["n"] > 1, "the flaky write was never reached"
    assert _entry(experiment)["status"] == "complete"
    assert handle.waited


def test_a_failed_progress_read_does_not_fail_the_recording(orchestrated):
    """The third line in that `try`, and the one that is a **read**.

    `run.progress()` polls the worker's `job.json` every tick. Reads hit the same Windows
    sharing violation as writes (`rundir.read_envelope` retries through `core.atomic` for exactly
    that), so an exhausted read is a third way for a healthy recording to be recorded failed.
    """

    class _FlakyProgress(_FakeRun):
        def progress(self):
            yield _event(1, 3)
            raise WIN_ACCESS_DENIED

    experiment, orchestrator, handle = orchestrated(run=_FlakyProgress())

    orchestrator._run_recording("recording_001")

    assert _entry(experiment)["status"] == "complete"
    assert handle.waited


def test_a_failing_wait_does_not_take_down_the_run(orchestrated):
    """`wait()` is I/O too, and it sits *outside* the launch handler.

    The adoption inside `_RecordedRun.wait` is guarded, but the drain it wraps can still raise
    once its retries are exhausted — and by then we are past the `except` that covers the
    launch, so an unguarded call would escape `_run_recording` altogether and fail the whole
    batch through the coordinator's backstop.

    Written because deleting the guard broke nothing: every fake here had a `wait()` that
    could not fail, so the one line protecting the run was covered by nothing.
    """

    class _FlakyWait(_FakeRun):
        def wait(self):
            self.waited = True
            raise WIN_ACCESS_DENIED

    experiment, orchestrator, handle = orchestrated(run=_FlakyWait())

    orchestrator._run_recording("recording_001")  # must not raise

    assert handle.waited
    assert (
        _entry(experiment)["status"] == "complete"
    ), "the job's own status still stands"


def test_the_worker_is_waited_on_before_the_batch_moves_on(orchestrated):
    """The orphan half, stated as its own property.

    In #289 the coordinator moved to the next recording while the abandoned worker kept the
    GPU — two detection processes on one 2070, and the whole batch slowed down. Waiting is
    what makes the run sequential again, so it is asserted separately from the status.
    """
    experiment, orchestrator, handle = orchestrated(emit=_raise_on(1))

    orchestrator._run_recording("recording_001")

    assert handle.waited


# --------------------------------------------------------------------------- #
# guard the guard
# --------------------------------------------------------------------------- #
# the adoption guarantee itself, at the mechanism rather than one of its callers
# --------------------------------------------------------------------------- #
def test_wait_adopts_the_real_status_even_when_the_drain_fails(tmp_path):
    """``_RecordedRun.wait`` is the **only** writer of a job's terminal status into
    ``recording.yaml``, and every downstream gate reads that copy rather than ``job.json``.

    So the adoption may not live in the happy path. The drain it wraps is a poll of the
    envelope — file I/O that can fail for reasons that say nothing about the job — and a
    recording left non-terminal is unusable *and* un-retryable, which is how #289's one
    transient error became permanent.

    Tested here rather than through the orchestrator because this is the mechanism the CLI,
    the GUI, distill and annotation all inherit. Written after deleting the `finally` broke
    nothing: the guarantee was a claim in a docstring.
    """
    from detail.api.recording import _RecordedRun

    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("recording_001")
    recording._reserve_run(
        "detection_runs", "det", lambda rid: {"id": rid, "status": "running"}
    )

    class _FlakyInner:
        status = "complete"
        error = None

        def wait(self):
            raise WIN_ACCESS_DENIED

    with pytest.raises(PermissionError):  # the caller still hears about it...
        _RecordedRun(_FlakyInner(), recording, "det_01", "detection_runs").wait()

    # ...and the recording is terminal anyway, so nothing downstream is blocked.
    entry = recording.detection_runs[0]
    assert entry["status"] == "complete"


def test_a_real_launch_failure_still_fails_the_recording(orchestrated, monkeypatch):
    """The distinction everything above rests on.

    A fix that swallowed *every* exception would pass every other test in this file. The work's
    own failure — here, the job never launching — must still be recorded as the recording
    failing, with the reason, so the batch continues and the user is told (D5).
    """
    experiment, orchestrator, _ = orchestrated()
    monkeypatch.setattr(
        Recording,
        "run",
        lambda self, mid, cfg: (_ for _ in ()).throw(RuntimeError("no gpu")),
    )

    orchestrator._run_recording("recording_001")

    entry = _entry(experiment)
    assert entry["status"] == "failed"
    assert "no gpu" in entry["error"]


def test_a_failed_job_is_still_recorded_as_failed(orchestrated):
    """The other real outcome: the worker ran and failed. Its status and reason survive."""
    handle = _FakeRun()
    handle.status = "failed"
    handle.error = "CUDA out of memory"
    experiment, orchestrator, _ = orchestrated(run=handle)

    orchestrator._run_recording("recording_001")

    entry = _entry(experiment)
    assert entry["status"] == "failed"
    assert entry["error"] == "CUDA out of memory"


# --------------------------------------------------------------------------- #
# the same property one level up: a batch module's progress
# --------------------------------------------------------------------------- #
def test_a_failed_progress_write_does_not_fail_a_whole_batch(tmp_path, monkeypatch):
    """`_run_batch`'s blast radius is every recording in the run, not one.

    A batch module reports through the coordinator's `on_progress`, which emits. That call was
    inside the same kind of handler, so a single failed progress write marked every
    recording of a run failed — strictly worse than the incident that was reported.
    """
    from detail.executor.runners import PROCESSING_MODULES

    experiment = Experiment.create("p", tmp_path)
    for sid in ("recording_001", "recording_002"):
        experiment.create_recording(sid)

    class _BatchModule:
        batch = True

        def run_batch(self, _experiment, _sids, _config, on_progress, _cancelled):
            on_progress("exporting", 0.5, "recording_001")  # the emit that raises
            return "ds_01"

        def recording_parallelism(self):
            return 1

    monkeypatch.setitem(PROCESSING_MODULES, "fake_batch", _BatchModule())
    orchestrator = RunOrchestrator(
        experiment,
        "run_001",
        "fake_batch",
        ["recording_001", "recording_002"],
        {},
        _raise_on(1),
        "2026-01-01T00:00:00Z",
    )

    orchestrator.run()

    record = runs_store.read_run(experiment.dir, "run_001")
    statuses = {s: e["status"] for s, e in record["per_recording"].items()}
    assert statuses == {"recording_001": "complete", "recording_002": "complete"}, (
        "one failed progress write failed the whole batch: "
        f"{json.dumps(record['per_recording'])}"
    )


# --------------------------------------------------------------------------- #
# the same class, three sites the original sweep did not reach
#
# #334 fixed `_run_recording`, and `tests/test_bookkeeping_boundaries.py` looks for a broad,
# non-re-raising handler whose try body *directly names* one of eight functions. So it cannot
# see a handler that is narrow (`except _StageFailed`), one that re-raises, or one that reaches
# the write a level down inside a helper. These three are each of those shapes, and each has
# the same signature: the statement that **records success** sat inside the handler that stamps
# failure, so a transient file error on it became a verdict on finished work.
# --------------------------------------------------------------------------- #
def test_a_failed_success_write_does_not_fail_finished_training(tmp_path, monkeypatch):
    """Hours of GPU work, `best.pt` on disk, and the record said `failed`.

    `_train`'s last statement was `write_model_yaml`, inside `run()`'s try, whose `except`
    stamps `set_model_status(..., "failed")`. A `PermissionError` on that one write therefore
    marked a complete model failed — and a failed model is unreachable from the catalog, with
    the weights still sitting there.
    """
    from detail.pipeline import training

    statuses: list[tuple] = []
    monkeypatch.setattr(
        training.models_store,
        "set_model_status",
        lambda *a, **k: statuses.append((a, k)),
    )
    monkeypatch.setattr(
        training.models_store,
        "write_model_yaml",
        lambda *a, **k: (_ for _ in ()).throw(WIN_ACCESS_DENIED),
    )
    monkeypatch.setattr(
        training, "_train", lambda *a, **k: {"status": "complete", "id": "m1"}
    )

    with pytest.raises(PermissionError):
        training.run(object(), tmp_path, None, None)

    assert not statuses, (
        "a transient failure on the write that *records* success was turned into a "
        f"set_model_status call: {statuses}"
    )


def test_a_real_training_failure_is_still_recorded_as_failed(tmp_path, monkeypatch):
    """The distinction the fix rests on, in this file's established shape.

    Moving the success write out of the handler must not stop a *genuine* training failure
    being recorded. A change that merely gutted the handler would pass the test above and fail
    this one.
    """
    from detail.pipeline import training

    statuses: list[tuple] = []
    monkeypatch.setattr(
        training.models_store,
        "set_model_status",
        lambda *a, **k: statuses.append((a, k)),
    )
    monkeypatch.setattr(
        training,
        "_train",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuda ran out of memory")),
    )

    with pytest.raises(RuntimeError, match="cuda ran out of memory"):
        training.run(object(), tmp_path, None, None)

    assert any(
        "failed" in repr(call) for call in statuses
    ), f"a real training failure must still be recorded as failed: {statuses}"


def test_a_failed_ready_write_does_not_fail_a_finished_import(tmp_path, monkeypatch):
    """Video transcoded, streams imported — and the recording said `failed`.

    `set_import_status(..., "ready")` was the last statement inside the try whose
    `except Exception` writes `failed`. And `recording.yaml` is the contended file by
    construction: the GUI thread reads it while the import worker rewrites it, which is the
    documented reason its writer is atomic in the first place.

    Driven through the real `import_data` on a real experiment — no storage mock, per the
    house rule — with a spec that has no parts and no streams, so the only write left is the
    terminal one.
    """
    from detail.api import imports as imports_api

    experiment = Experiment.create("p", tmp_path)
    spec = imports_api.RecordingImportSpec(name="r1")
    imports_api.create_shell(experiment, spec)

    real = imports_api.recording_store.set_import_status
    written: list[str] = []

    def _status(recording_dir, status):
        written.append(status)
        if status == "ready":
            raise WIN_ACCESS_DENIED
        return real(recording_dir, status)

    monkeypatch.setattr(imports_api.recording_store, "set_import_status", _status)

    with pytest.raises(PermissionError):
        imports_api.import_data(experiment, spec)

    assert "failed" not in written, (
        "the transient failure on the `ready` write was recorded as the import failing: "
        f"{written}"
    )


def test_a_real_import_failure_is_still_recorded_as_failed(tmp_path, monkeypatch):
    """The negative case for the import site: a genuine failure still marks it failed."""
    from detail.api import imports as imports_api

    experiment = Experiment.create("p", tmp_path)
    spec = imports_api.RecordingImportSpec(name="r1")
    imports_api.create_shell(experiment, spec)
    spec.streams = [
        imports_api.StreamImportSpec(
            name="s", importer="nope", raw=tmp_path / "missing.csv"
        )
    ]

    real = imports_api.recording_store.set_import_status
    written: list[str] = []

    def _status(recording_dir, status):
        written.append(status)
        return real(recording_dir, status)

    monkeypatch.setattr(imports_api.recording_store, "set_import_status", _status)

    with pytest.raises(Exception):
        imports_api.import_data(experiment, spec)

    assert "failed" in written, f"a real import failure must be recorded: {written}"


def test_a_failed_stage_write_does_not_fail_a_distill_run(tmp_path, monkeypatch):
    """`_set_stage` is a note about where the run got to, and it was raw.

    #334 gave `DistillOrchestrator` the `emit` half and not the record half. Worse than the
    processing case: `run()`'s handler is narrowly `except _StageFailed`, so a
    `PermissionError` here is not even recorded as a distill failure — it escapes to the
    coordinator's backstop and becomes "coordinator error" over a run whose training
    succeeded.
    """
    from detail.api import experiment as experiment_api

    orchestrator = object.__new__(experiment_api.DistillOrchestrator)
    orchestrator.id = "distill_001"
    orchestrator._stages = {}
    orchestrator._record = {}
    orchestrator._status = "running"
    orchestrator._error = None
    orchestrator._experiment = type("_E", (), {"dir": tmp_path})()
    monkeypatch.setattr(
        experiment_api.distill_store,
        "write_run",
        lambda *a, **k: (_ for _ in ()).throw(WIN_ACCESS_DENIED),
    )

    # Must not raise: the stage boundary reports on work that already succeeded.
    orchestrator._set_stage("train", "complete", model_id="m1")
    assert orchestrator._stages["train"]["status"] == "complete"


def test_the_distill_terminal_write_still_raises(tmp_path, monkeypatch):
    """And the terminal write is deliberately *not* best-effort.

    Dropping it would leave the record without the status the run actually reached, which is a
    different failure from the one above — so `_finalize` still raises, and the coordinator
    contract for that is an open question rather than something this change guessed at.
    """
    from detail.api import experiment as experiment_api

    orchestrator = object.__new__(experiment_api.DistillOrchestrator)
    orchestrator.id = "distill_001"
    orchestrator._stages = {}
    orchestrator._record = {}
    orchestrator._status = "running"
    orchestrator._error = None
    orchestrator._experiment = type("_E", (), {"dir": tmp_path})()
    monkeypatch.setattr(
        experiment_api.distill_store,
        "write_run",
        lambda *a, **k: (_ for _ in ()).throw(WIN_ACCESS_DENIED),
    )

    with pytest.raises(PermissionError):
        orchestrator._finalize("complete")


def test_the_write_that_records_success_is_not_inside_the_failure_handler():
    """The half a behavioural test cannot see here, and I got this wrong first time.

    The fix is two changes: ``_train`` stops writing the record, and ``run`` writes it outside
    the ``except`` that stamps ``failed``. A test that stubs ``_train`` — which is what you must
    do to avoid a real GPU trainer — exercises only the second, so moving the write back inside
    ``_train`` leaves it **passing**. It did; that is why this is here.

    So the first half is pinned structurally, which is what the class calls for anyway: nothing
    is wrong at the line where the write is written, and the symptom needs a transient file
    error on a network share to appear at all.
    """
    import ast
    import inspect

    from detail.pipeline import training

    tree = ast.parse(inspect.getsource(training))
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert {"run", "_train"} <= set(functions), sorted(functions)

    def writes_record(node) -> bool:
        return any(
            isinstance(child, ast.Call)
            and getattr(child.func, "attr", "") == "write_model_yaml"
            for child in ast.walk(node)
        )

    assert not writes_record(functions["_train"]), (
        "`_train` writes the model record, which puts that write inside `run`'s `except` — so "
        "a transient error on the write that *records* a finished training stamps it `failed`. "
        "Return the record and let `run` write it after the handler."
    )
    assert writes_record(
        functions["run"]
    ), "`run` must be the one that writes the record"

    # And it is *after* the try, not inside it: a call in the `try` body or a handler would be
    # the same bug one level up.
    try_blocks = [n for n in ast.walk(functions["run"]) if isinstance(n, ast.Try)]
    assert try_blocks, "`run` no longer has the handler this rule is about"
    assert not any(
        writes_record(block) for block in try_blocks
    ), "the record write is inside `run`'s try/except — it must follow it"
