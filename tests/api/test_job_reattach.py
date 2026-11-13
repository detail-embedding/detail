"""Reattaching to a single-job run — analysis, dataset, model, export (§12/§16).

A GUI page that opens a run it did not launch has to stream it: the launch-time handles
wrap a ``LocalExecutor`` whose run dirs live in **memory**, populated by ``submit``, so a
reopened page could not see a run this process did not start. ``AttachedJobRun`` polls the
run dir's ``job.json`` instead.

The tests are data-driven over ``Experiment._JOB_RUN_KINDS`` so a fifth run kind is covered
the day it is added, rather than the day someone remembers to add a test.
"""

import json
import os
import subprocess
import sys

import pytest

from detail.api.experiment import Experiment
from detail.storage import analysis_runs as analysis_store
from detail.storage import datasets as datasets_store
from detail.storage import export_runs as export_runs_store
from detail.storage import models as models_store

# kind -> (run_dir_for, write_record, read_record)
_WRITERS = {
    "analysis": (
        analysis_store.analysis_dir,
        analysis_store.write_analysis_yaml,
        analysis_store.read_analysis_yaml,
    ),
    "dataset": (
        datasets_store.dataset_dir,
        datasets_store.write_dataset_yaml,
        datasets_store.read_dataset_yaml,
    ),
    "model": (
        models_store.model_dir,
        models_store.write_model_yaml,
        models_store.read_model_yaml,
    ),
    "export": (
        export_runs_store.export_dir,
        export_runs_store.write_export_yaml,
        export_runs_store.read_export_yaml,
    ),
}


def test_every_job_run_kind_has_a_writer_here():
    """Guard the guard: a new kind in the table must show up in these tests."""
    assert set(_WRITERS) == set(Experiment._JOB_RUN_KINDS)


def _running(proj, kind, run_id="x_01", *, pid=None, status="running"):
    """A run left mid-flight: its own record plus the executor's job envelope."""
    run_dir_for, write_record, _read = _WRITERS[kind]
    run_dir = run_dir_for(proj.dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_record(run_dir, {"id": run_id, "status": status})
    envelope = {
        "schema_version": 1,
        "status": status,
        "job": {"job_id": run_id, "config": {}},
        "progress": None,
        "error": None,
    }
    if pid is not None:
        envelope["pid"] = pid
    (run_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")
    return run_dir


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.mark.parametrize("kind", sorted(_WRITERS))
def test_attach_active_only_for_an_active_run(tmp_path, kind):
    proj = Experiment.create("p", tmp_path)
    run_dir = _running(proj, kind)
    assert proj.attach_active_job_run(kind, "x_01") is not None

    _, write_record, read_record = _WRITERS[kind]
    record = read_record(run_dir)
    record["status"] = "complete"
    write_record(run_dir, record)
    assert proj.attach_active_job_run(kind, "x_01") is None  # static record, no bar
    assert proj.attach_active_job_run(kind, "nope") is None  # absent, not an error


@pytest.mark.parametrize("kind", sorted(_WRITERS))
def test_attached_run_streams_a_run_it_did_not_launch(tmp_path, kind):
    from detail.core.dataclasses import LabelingProgressEvent
    from detail.executor import rundir

    proj = Experiment.create("p", tmp_path)
    run_dir = _running(proj, kind)
    rundir.set_progress(
        run_dir, LabelingProgressEvent(3, 10, 0, 1, 5.0, 12.0, "working")
    )

    handle = Experiment.open(proj.dir).attach_job_run(kind, "x_01")  # a fresh process
    handle._poll = 0.0
    streamed = []
    for event in handle.progress():
        streamed.append(event)
        if len(streamed) == 1:  # the worker finishes while we watch
            rundir.set_status(run_dir, rundir.JobStatus.COMPLETE)

    assert [e.frames_done for e in streamed] == [3]  # the on-disk event, not a guess
    assert streamed[0].message == "working"
    assert handle.status == "complete"


@pytest.mark.parametrize("kind", sorted(_WRITERS))
def test_attached_run_reconciles_a_killed_worker(tmp_path, kind):
    """A killed worker must not leave the record reading "running" forever."""
    proj = Experiment.create("p", tmp_path)
    run_dir = _running(proj, kind, pid=_dead_pid())

    handle = proj.attach_job_run(kind, "x_01")
    handle._poll = 0.0
    handle.wait()

    _, _write, read_record = _WRITERS[kind]
    assert handle.status == "failed"
    assert read_record(run_dir)["status"] == "failed"


@pytest.mark.parametrize("kind", sorted(_WRITERS))
def test_attached_run_leaves_a_live_worker_alone(tmp_path, kind):
    """The other half: a running worker must not be reconciled out from under itself.

    Reversing the pid check would fail this run rather than let it finish — the kind of
    bug that only shows up as runs mysteriously marked failed.
    """
    from detail.core.dataclasses import LabelingProgressEvent
    from detail.executor import rundir

    proj = Experiment.create("p", tmp_path)
    run_dir = _running(proj, kind, pid=os.getpid())  # this process is very much alive
    rundir.set_progress(
        run_dir, LabelingProgressEvent(1, 10, 0, 1, 1.0, 9.0, "working")
    )

    handle = proj.attach_job_run(kind, "x_01")
    handle._poll = 0.0
    for _event in handle.progress():  # the live worker completes normally
        rundir.set_status(run_dir, rundir.JobStatus.COMPLETE)

    _, _write, read_record = _WRITERS[kind]
    assert handle.status == "complete"
    assert read_record(run_dir)["status"] == "complete"  # not failed by the pid guard


def test_reattaching_registers_the_event_types_it_will_have_to_decode(tmp_path):
    """A reattach in a **bare** process must decode an event whose tag a runner registered.

    **The class.** *A registry that the test suite always populates and a real process never
    does.* A progress event is rebuilt from `job.json` by tag, and every tag outside
    `core/dataclasses.py` — every analysis module's, the export's — is registered as a side
    effect of importing the module that defines it. `attach_job_run` did not import them, so
    `detail analysis attach an_15` on a real run died with ``unknown progress event type:
    'temporal_clustering'``. Shipped, and invisible: `conftest`'s autouse `_isolate_registries`
    imports every runner into every test, so the registry is full in here and empty out there.

    Which is why this runs in a **subprocess**. Asserting it in-process would pass against the
    reverted fix, because this process has already imported everything.
    """
    proj = Experiment.create("p", tmp_path)
    run_dir = _running(proj, "analysis", status="running", pid=os.getpid())
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "job": {"job_id": "x_01", "config": {}},
                # The tag a *module* registers — the one a bare process has never heard of.
                "progress": {"type": "temporal_clustering", "recordings_done": 1},
                "error": None,
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    script = (
        "from detail.api.experiment import Experiment\n"
        f"h = Experiment.open({str(proj.dir)!r}).attach_job_run('analysis', 'x_01')\n"
        "h._poll = 0.0\n"
        "next(iter(h.progress()), None)\n"  # decodes the event, or raises
        "print('decoded')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert (
        "decoded" in done.stdout
    ), f"a bare process could not decode a reattached run's progress event:\n{done.stderr}"


@pytest.mark.parametrize(
    "kind,method",
    [
        ("analysis", "attach_active_analysis_run"),
        ("dataset", "attach_active_dataset_run"),
        ("model", "attach_active_model_run"),
        ("export", "attach_active_export_run"),
    ],
)
def test_named_wrappers_reach_the_shared_body(tmp_path, kind, method):
    """Each kind has a discoverable typed accessor, not just the generic one."""
    proj = Experiment.create("p", tmp_path)
    _running(proj, kind)
    assert getattr(proj, method)("x_01") is not None


# --------------------------------------------------------------------------- #
# whose worker is it — the fact Ctrl-C is derived from
# --------------------------------------------------------------------------- #
def _attach_methods():
    """Every ``Experiment.attach_*`` method, discovered rather than listed.

    By prefix so a seventh reattachable family is covered the day it is written. The
    ``attach_active_*`` pair is included: it returns the same handle for a live run and ``None``
    for a terminal one, so it answers the same question on the path that matters.
    """
    return sorted(
        name
        for name in dir(Experiment)
        if name.startswith("attach_") and callable(getattr(Experiment, name))
    )


def test_every_attach_method_returns_a_handle_that_does_not_own_its_worker(
    tmp_path, monkeypatch
):
    """The fact the CLI's ``Ctrl-C`` is derived from, asserted over every family at once.

    Five ``attach`` commands have printed *"Ctrl-C detaches; it does not cancel the run"* in
    their own ``--help`` since they were written, and all five streamed through the one driver
    that cancelled — because the flag carrying that behaviour was a parameter on the other two
    drivers, defaulting to "as before". `RunHandle.owns_worker` replaces the parameter, so the
    promise is kept by construction; this is the half that says the handles answer correctly.
    """
    proj = Experiment.create("p", tmp_path)
    checked = []
    for kind in Experiment._JOB_RUN_KINDS:
        _running(proj, kind, f"{kind}_01")
        handle = proj.attach_job_run(kind, f"{kind}_01")
        assert handle.owns_worker is False, (
            f"a reattached {kind} run claims to own its worker, so Ctrl-C would cancel it — "
            f"which is the opposite of what `detail {kind} attach --help` promises"
        )
        checked.append(kind)
    assert checked, "the sweep ran over nothing"


def test_the_processing_and_distill_handles_agree(tmp_path):
    """The two families outside ``_JOB_RUN_KINDS``, which have their own attached classes and
    their own ``attach`` commands. Named here so deleting either override fails something.
    """
    from detail.api.experiment import AttachedDistillRun, AttachedRun

    proj = Experiment.create("p", tmp_path)
    assert AttachedRun(proj, "run_001").owns_worker is False
    assert AttachedDistillRun(proj, "dst_01").owns_worker is False


def test_a_launched_run_does_own_its_worker(tmp_path):
    """The other half, and the one that makes the rule a rule.

    If every handle detached, Ctrl-C on a run this CLI just started would leave the worker
    running in its own process group with nothing watching it — the orphan the cancel exists to
    prevent. The base class says ``True`` for exactly this reason.
    """
    from detail.executor.base import RunHandle
    from detail.executor.handles import ExecutorRunHandle

    assert RunHandle.owns_worker.fget(object()) is True
    assert ExecutorRunHandle(object(), "run_01").owns_worker is True


#: Commands that promise Ctrl-C detaches **without** being an ``attach``. Two entries, one
#: reason, and each is a claim rather than an opt-out: the test below checks the named launcher
#: really does hand back a handle that does not own its worker, so an entry that stops being
#: true fails.
#:
#: `distill` is the exception because its *API* hands back an `AttachedDistillRun` — launching a
#: distill detaches you from it by design, which is the right call for the longest thing this app
#: does; `keypoints distill` (§12 *Distill keypoints*) is the same run with a different label
#: stage, on the same coordinator, and hands back the same handle. Note which way this cuts:
#: `export run` briefly carried the same promise, on the reasoning that #354 made exports
#: durable, and it was wrong. Durability is what makes walking away possible; it is not what
#: makes it the expected reading of an interrupt on a command you are watching. Every other
#: ``run`` command cancels.
_DETACHING_NON_ATTACH = {
    "distill run": ("distill", "AttachedDistillRun"),
    "keypoints distill": ("distill_keypoints", "AttachedDistillRun"),
    # A keypoint setup (§26, #717) is a distill run of its own kind, launched and continued
    # through the same coordinator; both halves stream an AttachedDistillRun.
    "keypoints setup start": ("start_keypoint_setup", "AttachedDistillRun"),
    "keypoints setup continue": ("continue_keypoint_setup", "AttachedDistillRun"),
}

#: The exact sentences. Substring-sniffing for the *word* "detach" reads a docstring that
#: explains why it does **not** detach as a promise that it does — which happened, to this file,
#: to the docstring written for this guard.
_DETACHES = "Ctrl-C detaches"
_CANCELS = "Ctrl-C cancels"


def _commands_claiming_something_about_interrupts():
    """``{path: "detaches"|"cancels"}`` for every leaf command whose help mentions Ctrl-C."""
    import click

    from detail.cli import main

    found = {}

    def walk(group, path):
        for name, command in getattr(group, "commands", {}).items():
            if isinstance(command, click.Group):
                walk(command, path + [name])
                continue
            # Whitespace-collapsed: a docstring wraps wherever the line ran out, so the phrase
            # is really "Ctrl-C\n    detaches" and a literal check passes or fails on where
            # black happened to break the line.
            text = " ".join((command.help or "").split())
            if _DETACHES in text:
                found[" ".join(path + [name])] = "detaches"
            elif _CANCELS in text:
                found[" ".join(path + [name])] = "cancels"

    walk(main, [])
    return found


def test_every_attach_command_promises_to_detach():
    """The documented half, and the reason the bug lasted: the promise was correct, prominent,
    and unimplementable at the call site. Swept over the CLI's own command tree, so a new
    ``attach`` inherits the obligation rather than needing to be remembered."""
    claims = _commands_claiming_something_about_interrupts()
    attaches = [p for p in claims if p.endswith(" attach")]
    assert len(attaches) >= 6, f"only found {len(attaches)} attach commands: {attaches}"
    for path in attaches:
        assert claims[path] == "detaches", (
            f"`detail {path} --help` does not say Ctrl-C detaches — a reattached run's worker "
            f"is not this process's to cancel"
        )


def test_only_reattachment_promises_to_detach():
    """The invariant `_interrupted` implements, asserted over the prose that advertises it.

    This is the guard that would have caught it: `detail export run` shipped saying **Ctrl-C
    detaches** while launching its own worker, so the one thing it promised was the one thing it
    could not do. A command that promises to detach and then cancels is worse than one that says
    nothing, because the user acted on the sentence.
    """
    claims = _commands_claiming_something_about_interrupts()
    detaching = {p for p, claim in claims.items() if claim == "detaches"}
    unexplained = detaching - {p for p in detaching if p.endswith(" attach")}
    assert unexplained <= set(_DETACHING_NON_ATTACH), (
        f"{sorted(unexplained - set(_DETACHING_NON_ATTACH))} promises Ctrl-C detaches but does "
        f"not reattach, so it launches its own worker and cancels it — either the help is wrong "
        f"or the command belongs in _DETACHING_NON_ATTACH with its reason"
    )


def test_the_detaching_exception_still_earns_its_place():
    """Guard the guard: an allowlist entry is a claim about the code, checked here.

    `distill run` may promise to detach only for as long as `Experiment.distill` really returns a
    handle that does not own its worker. The day it returns a launch handle, the promise becomes
    the `export run` bug and this fails rather than excusing it.
    """
    import typing

    for path, (method, expected) in _DETACHING_NON_ATTACH.items():
        assert path in _commands_claiming_something_about_interrupts(), (
            f"_DETACHING_NON_ATTACH names `detail {path}`, which no longer says anything about "
            f"Ctrl-C — a stale exemption excuses nothing and hides the next one"
        )
        returns = typing.get_type_hints(
            getattr(Experiment, method), include_extras=False
        ).get("return")
        name = getattr(returns, "__name__", str(returns))
        assert name == expected, (
            f"Experiment.{method} returns {name}, not {expected} — `detail {path}` promises "
            f"Ctrl-C detaches and would now cancel"
        )
        assert getattr(returns, "owns_worker").fget(object()) is False
