"""Stopping training keeps the model; cancelling does not. The difference is the kill.

**The bug this exists for, which nearly shipped as a rename.** Training already honoured a stop
flag, and an interrupted run already wrote its best weights into the record — so "Stop and keep"
looked like renaming the Cancel button. It is not, because of what sits between the flag and the
code that keeps anything:

    Cancel  -> flag set -> trainer.stop seen at the NEXT EPOCH END (minutes)
                        -> 3s grace expires -> SIGTERM -> SIGKILL
                        -> pipeline/training.py's post-loop block never runs
                        -> weights.best / metrics / model_family never written

Ultralytics has usually written `best.pt` by then, so the file survives and **nothing points at
it**. The weights are kept and the model is lost, which is the worst version: a directory that
looks complete and a record that cannot load.

So stop and cancel are different calls, and the whole difference is that one never kills.

**The second half: a finish the user chose, recorded as an interruption (#486).** The two calls
still shared one *marker* — `request_stop` wrote Cancel's flag — so neither the worker nor the
training runner could tell which was asked, and every stopped model landed `partial`: the status
for an interrupted, resumable run. But a stopped run has *returned*, and Ultralytics finalizes a
run that returns, stripping `last.pt` of its epoch and optimizer. `partial` offered **Resume
training** on that checkpoint, and Ultralytics 8.4.120 does not refuse one: it starts a new
training from its defaults, on its `coco8-seg.yaml` sample, into its own runs directory — seen
three times on one machine, each model's record then reading `complete`.

How it hid: the intended behaviour was written down twice — `Experiment.stop_model` says the
status "stays ... `complete`, because the run did finish, early", and the distill page's stop says
there is "nothing to resume afterward" — while `DESIGN_PLAN.md` specified `partial`, and the code
followed the spec. The rule now: **stop-and-keep has its own marker and ends `complete` with
`stopped_at_epoch`; only a killed or crashed run is resumable
(`RESUMABLE_TRAINING_STATUSES`); and the trainer refuses a finalized checkpoint rather than let
the backend substitute a different run.** The tests below that bite: the stopped run's record,
the cancel counterexample, the legacy-`partial` refusal, and `resume` never reaching `train()`.
Two more pin the other surfaces: the worker's terminal status
(`tests/executor/test_local.py::test_run_job_stop_and_keep_stops_the_runner_and_lands_complete`)
and the model page offering no Resume on a finished model
(`tests/gui/test_fit.py::test_a_model_that_stopped_cleanly_offers_no_resume`).
"""

from __future__ import annotations

import pytest

from detail.executor import local, rundir


def test_stop_asks_and_does_not_kill(tmp_path, monkeypatch):
    """The property, stated as the absence of the kill.

    Asserted by making the kill loud rather than by timing: a test that waited to see whether a
    process died would be slow *and* would pass on a machine where it happened not to.
    """
    killed = []
    monkeypatch.setattr(local, "_kill_tree", lambda pid, force: killed.append(force))

    local.request_stop(tmp_path)

    assert rundir.stop_requested(tmp_path), "stop did not ask the run to stop"
    assert not rundir.cancel_requested(tmp_path), (
        "stop raised Cancel's marker — the worker then cannot tell a chosen finish from an "
        "interruption, and the model lands partial with Resume offered on it (#486)"
    )
    assert (
        not killed
    ), "stop killed the worker — then the run cannot write its own record"


def test_cancel_still_kills(tmp_path, monkeypatch):
    """The pair, and the reason stop had to be a second function rather than a flag on the first.

    Without this, `request_stop` could be implemented by weakening `cancel_run_dir` and every
    test above would still pass — while Cancel quietly stopped being able to stop anything.
    """
    killed = []
    monkeypatch.setattr(local, "_kill_tree", lambda pid, force: killed.append(force))
    monkeypatch.setattr(local, "_is_our_worker", lambda pid, run_dir: True)
    monkeypatch.setattr(local, "_alive", lambda pid: True)
    rundir.init_run_dir(tmp_path, _job())
    rundir.set_pid(tmp_path, 424242)

    local.cancel_run_dir(tmp_path, grace_s=0.0)

    assert killed, "cancel did not kill; a blocking trainer would never stop"


def _job():
    from detail.core.dataclasses import ComputeTarget, TrainingJob

    return TrainingJob(
        job_id="m1",
        model_id="model_001",
        dataset_id="ds_1",
        base_model="yolo11n.pt",
        trainer_id="ultralytics_yolo",
        hyperparams={"epochs": 1},
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-09-10T12:00:00Z",
    )


def test_stop_is_a_no_op_on_a_finished_run(tmp_path):
    """Pressing it after the run landed must not leave a cancel flag lying in the directory for
    the *next* attempt to find — a resume would see it and stop immediately."""
    from detail.api.experiment import Experiment
    from detail.storage import models as models_store

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "complete"})

    exp.stop_model("model_001")

    assert not rundir.stop_requested(mdir) and not rundir.cancel_requested(mdir)


def test_stop_on_a_missing_model_says_so(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p", tmp_path)
    with pytest.raises(FileNotFoundError):
        exp.stop_model("model_404")


def test_the_trainer_sees_the_flag_at_an_epoch_boundary(tmp_path):
    """What makes the cooperative path work at all: Ultralytics' callback sets `trainer.stop`,
    and `model.train()` then returns **normally** — so control reaches the block that writes the
    record. If it raised or exited instead, stop-and-keep would keep nothing and look identical.
    """
    import threading

    from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer

    stop_event = threading.Event()
    stop_event.set()

    class _FakeModel:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, name, fn):
            self.callbacks.setdefault(name, []).append(fn)

    class _FakeTrainer:
        epoch = 3
        metrics = {"metrics/mAP50(B)": 0.84}
        stop = False

    model = _FakeModel()
    UltralyticsYOLOTrainer()._wire_callbacks(model, 10, None, stop_event)
    for fn in model.callbacks.get("on_fit_epoch_end", []) or model.callbacks.get(
        "on_train_epoch_end", []
    ):
        fn(_FakeTrainer)

    assert _FakeTrainer.stop is True, (
        "the stop flag never reached the trainer, so a cooperative stop would wait forever "
        "and the user would press Cancel — losing the model"
    )


def test_stop_is_offered_on_every_surface_that_can_start_a_training():
    """Surface parity, which is the rule that made this a three-file change.

    The GUI grew the button first, because that is where someone watches a curve and decides it
    has plateaued. But `train run` in a terminal reaches the same runs, and interrupting it is
    the kill — so a CLI user had *only* the option that throws the model away. A capability is
    exposed in every surface a user reaches it through, or the gap is recorded with its reason.
    """
    import ast
    import pathlib

    import detail.cli as cli_mod
    import detail.gui.pages as pages_mod
    from detail.api.experiment import Experiment

    assert hasattr(Experiment, "stop_model"), "the API lost it"

    tree = ast.parse(pathlib.Path(cli_mod.__file__).read_text(encoding="utf-8-sig"))
    train_commands = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            if getattr(dec.func, "attr", "") != "command":
                continue
            if getattr(dec.func.value, "id", "") != "train":
                continue
            train_commands |= {a.value for a in dec.args if isinstance(a, ast.Constant)}
    assert (
        "stop" in train_commands
    ), f"the CLI can start a training and not stop one; it has {sorted(train_commands)}"

    gui = pathlib.Path(pages_mod.__file__).read_text(encoding="utf-8-sig")
    assert "Stop and keep" in gui, "the GUI lost the button"
    assert "stop_model" in gui, "the GUI button no longer calls the API"


def test_the_cli_stop_asks_and_does_not_kill(tmp_path, monkeypatch):
    """Through the real entry point. The property is the same one the API test states, asserted
    where a user actually reaches it — a command that quietly called `cancel_model` would look
    identical from the outside and lose the model every time."""
    from click.testing import CliRunner

    from detail.api.experiment import Experiment
    from detail.cli import main
    from detail.executor import local, rundir
    from detail.storage import models as models_store

    killed = []
    monkeypatch.setattr(local, "_kill_tree", lambda pid, force: killed.append(force))

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "running"})

    result = CliRunner().invoke(
        main, ["train", "stop", "model_001", "--experiment", str(exp.dir)]
    )

    assert result.exit_code == 0, result.output
    assert rundir.stop_requested(mdir), "the command did not ask the run to stop"
    assert not killed, "the CLI killed the worker, so its record is never written"
    assert (
        "best weights" in result.output
    ), f"the command does not say what it kept: {result.output!r}"


def test_train_status_shows_the_best_epoch_so_far(tmp_path):
    """The terminal's half of the curve.

    `detail train stop` was added without this, which left a CLI user able to stop a training and
    with nothing to decide it on — the GUI grew a plot for exactly that question and a terminal
    cannot draw one. The parity that matters is not the widget, it is the fact.
    """
    from click.testing import CliRunner

    from detail.api.experiment import Experiment
    from detail.cli import main
    from detail.storage import models as models_store

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "status": "running",
            "trainer_id": "ultralytics_yolo",
            "hyperparams": {"epochs": 60},
        },
    )
    run = mdir / "run"
    run.mkdir()
    rows = ["epoch,metrics/mAP50(B),metrics/mAP50(M)"]
    rows += [f"{e},{0.80 - 0.004 * abs(e - 31):.4f},0.7" for e in range(1, 39)]
    (run / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        main, ["train", "status", "model_001", "--experiment", str(exp.dir)]
    )

    assert result.exit_code == 0, result.output
    assert "0.800" in result.output, result.output
    assert (
        "epoch 31" in result.output
    ), f"the peak epoch is not named: {result.output!r}"
    assert (
        "7 epochs" in result.output
    ), f"the gap since the peak is missing: {result.output!r}"
    assert "38 of 60" in result.output, f"where the run is up to: {result.output!r}"


def test_train_status_on_a_model_with_no_history_says_nothing_about_epochs(tmp_path):
    """The counterexample: an imported model must not grow a best-so-far line reading 0.000."""
    from click.testing import CliRunner

    from detail.api.experiment import Experiment
    from detail.cli import main
    from detail.storage import models as models_store

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "complete", "trainer_id": "ultralytics_yolo"},
    )

    result = CliRunner().invoke(
        main, ["train", "status", "model_001", "--experiment", str(exp.dir)]
    )

    assert result.exit_code == 0, result.output
    assert "best so far" not in result.output, result.output


# --------------------------------------------------------------------------- #
# A stopped model is finished (#486)
# --------------------------------------------------------------------------- #
# Stop and keep ended every run `partial`, the status for an interruption. Ultralytics has
# already *finalized* a stopped run — best weights kept, `last.pt` stripped of epoch and
# optimizer — so the model was finished and the record said otherwise, and `partial` is what
# offered **Resume training**. Given a stripped checkpoint, Ultralytics 8.4.120 does not fail a
# resume: it trains a new model from its own defaults, on its `coco8-seg.yaml` sample. That
# happened three times on one machine, and each time the model's record then read `complete`.
#
# The mechanism was one shared marker: `request_stop` wrote Cancel's flag, so neither the worker
# nor the training runner could tell which the user asked for. It was documented as intended in
# `Experiment.stop_model` ("the record's status stays ... complete, because the run did finish,
# early") and in the distill page's stop, and implemented in neither.


def _trained(tmp_path, *, marker, final_metrics=None):
    """Run the training runner to the end of a stopped run, with `marker` in its run dir."""
    import threading

    from detail.pipeline import training
    from detail.storage import experiment as experiment_store
    from detail.storage import models as models_store
    from tests.pipeline.test_training import StubTrainer
    from tests.pipeline.test_training import _dataset, _job, _registered_stub

    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "running"})
    rundir.init_run_dir(mdir, _job())
    if marker == "stop":
        rundir.request_stop(mdir)
    elif marker == "cancel":
        rundir.request_cancel(mdir)
    stop_event = threading.Event()
    stop_event.set()  # the worker's watcher sets it for either marker
    with _registered_stub():
        if final_metrics is not None:
            StubTrainer.final_metrics = lambda self, d: dict(final_metrics)
        try:
            training.run(
                _job(), mdir, progress_cb=lambda e: None, stop_event=stop_event
            )
        finally:
            StubTrainer.final_metrics = lambda self, d: {"box_map50": 0.9}
    return models_store.read_model_yaml(mdir)


def test_a_stopped_training_is_complete_and_says_where_it_stopped(tmp_path):
    record = _trained(tmp_path, marker="stop", final_metrics={"final_epoch": 37})
    assert record["status"] == "complete", (
        "a stop-and-keep landed as an interruption; the model is finished and will be "
        "offered Resume on a finalized checkpoint"
    )
    assert (
        record["stopped_at_epoch"] == 37
    ), "complete without the epoch hides that a 5-epoch model did not run its course"


def test_a_cancel_caught_cleanly_is_still_partial(tmp_path):
    """The counterexample: the cancel marker must not be read as a chosen finish."""
    record = _trained(tmp_path, marker="cancel")
    assert record["status"] == "partial"
    assert "stopped_at_epoch" not in record


def test_an_unexplained_stop_keeps_the_conservative_reading(tmp_path):
    """A stop event with neither marker behind it is not evidence the user chose to finish."""
    record = _trained(tmp_path, marker=None)
    assert record["status"] == "partial"


def test_resume_is_refused_on_a_model_that_stopped_cleanly(tmp_path):
    """Records already on disk from before the fix are `partial` with a stripped checkpoint.
    Resume must refuse them before a worker is spawned, and say what to do instead."""
    from detail.api.experiment import Experiment
    from detail.storage import experiment as experiment_store
    from detail.storage import models as models_store
    from tests.pipeline.test_training import _dataset, _job, _registered_stub

    proj = _dataset(experiment_store.create_experiment(tmp_path, "p"))
    mdir = models_store.model_dir(proj, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"stripped")
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "status": "partial", "trainer_id": "stub_trainer"}
    )
    rundir.init_run_dir(mdir, _job())
    with _registered_stub(), pytest.raises(ValueError, match="stopped cleanly"):
        Experiment(proj).resume_training("model_001")


@pytest.mark.parametrize(
    "ckpt, resumable",
    [
        (
            {"epoch": 36, "optimizer": {"state": {}}},
            True,
        ),  # a killed run's per-epoch save
        (
            {"epoch": -1, "optimizer": None},
            False,
        ),  # finalized: what Stop and keep leaves
        ({"epoch": 12, "optimizer": None}, False),
        ({"epoch": None, "optimizer": {"state": {}}}, False),
        (None, False),
    ],
)
def test_the_trainer_knows_a_finalized_checkpoint(ckpt, resumable):
    from detail.trainers.ultralytics_yolo import _resumable

    assert _resumable(ckpt) is resumable


def test_the_trainer_refuses_to_resume_a_finalized_checkpoint(tmp_path, monkeypatch):
    """Through `resume` itself, with Ultralytics' `YOLO` replaced (the only-mock-inference
    rule): given the checkpoint Stop and keep leaves, it must raise and never reach
    `train(resume=True)` — which is where Ultralytics substitutes a fresh run on COCO8.
    """
    import sys
    import types

    from detail.trainers import ultralytics_yolo
    from tests.pipeline.test_training import _job

    trained = []

    class _FakeYOLO:
        def __init__(self, path):
            self.ckpt = {"epoch": -1, "optimizer": None}

        def train(self, **kwargs):
            trained.append(kwargs)

        def add_callback(self, *args):
            pass

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(ultralytics_yolo, "import_ultralytics", lambda: None)
    monkeypatch.setattr(ultralytics_yolo, "_weights_dir", lambda: tmp_path / "cache")
    mdir = tmp_path / "model_001"
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"stripped")
    ddir = tmp_path / "ds"
    ddir.mkdir()
    (ddir / "data.yaml").write_text("path: .\nnames: {0: rat}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finished training checkpoint"):
        ultralytics_yolo.UltralyticsYOLOTrainer().resume(
            _job(trainer_id="ultralytics_yolo"), ddir, mdir, "cpu"
        )
    assert (
        not trained
    ), "resume reached Ultralytics' train() with a finalized checkpoint"
