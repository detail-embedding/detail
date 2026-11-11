"""Distilling a keypoint model (§12 *Distill keypoints*): **Distill new model** with *Model
type: Keypoints* lists the finished track runs and launches ``Experiment.distill_keypoints``;
the Models page and the Model manager reach it through one button; the run page draws the
third stage."""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QLabel, QPushButton

from detail.api.experiment import Experiment
from detail.gui import fit as fit_gui
from detail.gui import keypoint_distill as kd_gui  # noqa: F401 — sync_busy patches it
from detail.gui import run_units
from detail.gui.run_units import DONE, QUEUED, RUNNING
from detail.storage import distill_runs as distill_store
from tests.pipeline.test_keypoint_dataset import (  # noqa: F401 — fixtures
    patch_teacher,
    taught,
)
from tests.pipeline.test_keypoint_detect import (
    tracked,
)  # noqa: F401 — `taught` needs it


@pytest.fixture
def sync_busy(monkeypatch):
    """Run the dialog's off-thread read inline, so the list is filled when the test looks."""
    monkeypatch.setattr(
        kd_gui, "run_busy", lambda target, fn, on_done, **k: on_done(fn())
    )


def _keypoint_dialog(experiment, teacher="keypoint_detect"):
    dialog = fit_gui.DistillDialog(experiment, model_type="keypoints")
    form = dialog.keypoint_form()
    form._kind.setCurrentIndex(form._kind.findData(teacher))
    return dialog, form


def _keypoint_record(exp, register_status="pending", status="running"):
    distill_store.write_run(
        exp.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "rats",
            "kind": "keypoints",
            "status": status,
            "created_at": "2026-10-08T00:00:00Z",
            "config": {},
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "complete", "model_id": "model_001"},
                "register": {"status": register_status},
            },
            "resume": False,
        },
    )


def test_the_dialog_lists_the_finished_detect_runs_and_launches(
    qapp, taught, sync_busy, monkeypatch
):
    it, _recording = taught
    proj = it.experiment
    captured = {}

    def fake_distill(name, detect_runs, n_frames, **options):
        captured.update(
            name=name, detect_runs=detect_runs, n_frames=n_frames, **options
        )
        return object()

    monkeypatch.setattr(proj, "distill_keypoints", fake_distill)
    dialog, form = _keypoint_dialog(proj)
    assert dialog.model_type() == "keypoints" and form.isVisibleTo(dialog)
    assert form._runs.count() == 1
    item = form._runs.item(0)
    assert item.checkState() == Qt.CheckState.Checked, "the only run is pre-ticked"
    assert "kd_01" in item.text()
    assert (
        form._runs.height() < 6 * form._runs.sizeHintForRow(0) + 20
    ), "sized to its rows"
    # The patch method sets no default cut-off: the form says so up front and holds Distill.
    assert not dialog._ok.isEnabled()
    assert "no default" in dialog._ok.toolTip()
    assert "heatmap peak, 0 to 1" in form._cutoff_note.text()
    assert "fake_kp has no default" in form._cutoff_note.text()
    assert form._score_cutoff.isVisibleTo(dialog), "not tucked under Advanced"
    # the training knobs come from the trainer's config model, the label knobs from ours
    assert set(form._hparams) == {"epochs", "batch_size", "imgsz", "patience"}
    assert set(form._fields) == {
        "n_frames",
        "val_split",
        "exclude_audit_s",
        "every_n_frames",
    }

    dialog._name.setText("fast rats")
    form._fields["n_frames"].setValue(40)
    form._score_cutoff.setValue(0.4)
    assert dialog._ok.isEnabled(), "a cut-off releases Distill"
    form._hparams["epochs"].setValue(3)
    dialog._run()
    assert captured["name"] == "fast rats"
    assert captured["detect_runs"] == {"recording_002": "kd_01"}
    assert captured["n_frames"] == 40
    assert captured["score_cutoff"] == pytest.approx(0.4)
    assert captured["hyperparams"]["epochs"] == 3
    assert captured["base_model"] == "yolo11s-pose.pt"
    assert captured["train_compute"] is not None
    assert dialog.run_handle is not None


def test_the_dialog_needs_a_name_and_a_ticked_run(qapp, taught, sync_busy, monkeypatch):
    it, _recording = taught
    launched = []
    monkeypatch.setattr(
        it.experiment, "distill_keypoints", lambda *a, **k: launched.append(a)
    )
    monkeypatch.setattr(fit_gui.QMessageBox, "warning", lambda *a, **k: None)
    dialog, form = _keypoint_dialog(it.experiment)
    dialog._run()  # no name
    dialog._name.setText("s")
    form._runs.item(0).setCheckState(Qt.CheckState.Unchecked)
    dialog._run()  # nothing ticked
    assert not launched and dialog.run_handle is None


def test_the_dialog_says_when_there_is_nothing_to_teach_from(
    qapp, tmp_path, sync_busy, monkeypatch
):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = Experiment.create("p", tmp_path)
    exp.create_recording("recording_001")
    dialog, form = _keypoint_dialog(exp)
    assert form._runs.count() == 0
    assert not dialog._ok.isEnabled()
    assert "finished keypoint detect run" in dialog._ok.toolTip()
    assert "Run keypoint detect" in form._runs_note.text()
    dialog._type.setCurrentIndex(dialog._type.findData("detector"))
    assert dialog._ok.isEnabled(), "a detector does not need a keypoint detect run"
    assert not form.isVisibleTo(dialog)


def test_one_button_makes_either_kind_of_model(qapp, tmp_path, sync_busy, monkeypatch):
    from detail.gui import pages as pages_mod
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = Experiment.create("p", tmp_path)
    launched = []
    page = pages_mod.ModelsPage(
        exp, on_open=lambda *_: None, on_distill_launched=launched.append
    )
    buttons = {b.text(): b for b in page.findChildren(QPushButton)}
    assert "＋ Distill new model" in buttons
    assert not any("keypoint" in t.lower() for t in buttons), "no second distill button"
    offered = []

    def fake_exec(self):
        offered.append([self._type.itemData(i) for i in range(self._type.count())])
        self.run_handle = "handle"
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(fit_gui.DistillDialog, "exec", fake_exec)
    buttons["＋ Distill new model"].click()
    assert launched == ["handle"]
    assert offered == [["detector", "keypoints"]]


def test_a_keypoint_model_says_so_in_the_models_table(qapp, tmp_path):
    from detail.gui import pages as pages_mod

    exp = Experiment.create("p", tmp_path)
    for model_id, task in (("model_001", "detect"), ("model_002", "pose")):
        directory = exp.dir / "models" / model_id
        directory.mkdir(parents=True)
        (directory / "model.yaml").write_text(
            f"schema_version: 1\nid: {model_id}\nname: {model_id}\nstatus: complete\n"
            f"trainer_id: ultralytics_yolo\ndataset_id: dataset_001\ntask: {task}\n",
            encoding="utf-8",
        )
    ids, rows = pages_mod.experiment_model_rows(exp)
    kind = pages_mod.EXPERIMENT_MODEL_COLUMNS.index("Kind")
    assert {i: r[kind] for i, r in zip(ids, rows)} == {
        "model_001": "detector",
        "model_002": "keypoints",
    }


def test_the_run_page_shows_the_register_stage(qapp, tmp_path, monkeypatch):
    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = Experiment.create("p", tmp_path)
    _keypoint_record(exp, register_status="running")

    page = DistillRunDetail(exp, "distill_001")

    drawn = page._panel._units_drawn()
    assert [u.name for u in drawn] == [
        "Label frames",
        "Train model",
        "Register student",
    ]
    assert [u.state for u in drawn] == [DONE, DONE, RUNNING]
    labels = [w.text() for w in page.findChildren(QLabel)]
    assert any(t.startswith("Register:") for t in labels)
    assert any("running" in t and "role keypoints" in t for t in labels)


def test_the_form_opens_at_the_shared_frame_default(qapp, tmp_path, sync_busy):
    from detail.configs.distill import DEFAULT_N_FRAMES

    form = kd_gui.KeypointDistillForm(Experiment.create("p", tmp_path))
    assert form._fields["n_frames"].value() == DEFAULT_N_FRAMES == 1000


def test_a_detector_distill_still_draws_two_rows():
    record = {
        "stages": {
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "pending", "model_id": None},
        }
    }
    units = run_units.distill_units(record)
    assert [u.name for u in units] == ["Label frames", "Train model"]
    assert [u.state for u in units] == [DONE, QUEUED]
    keypoint = {
        "stages": {**record["stages"], "register": {"status": "pending"}},
    }
    assert [u.name for u in run_units.distill_units(keypoint)][-1] == "Register student"


def test_a_teacher_row_names_its_example_set_and_a_default_needs_no_cut_off(
    qapp, tmp_path, sync_busy
):
    form = kd_gui.KeypointDistillForm(Experiment.create("p", tmp_path))
    form._kind.setCurrentIndex(form._kind.findData("keypoint_detect"))
    teacher = {
        "method": "dinov3_match",
        "default_cutoff": None,
        "meaning": "cosine similarity",
        "low": -1.0,
        "high": 1.0,
        "example_set": {"id": "ex_01", "name": "overhead rats"},
        "detection_run_id": "det_01",
        "tracking_run_ids": ["track_01"],
    }
    row = {
        "recording_id": "recording_001",
        "recording_name": "cage a",
        "run_id": "kd_01",
        "method": "temporal_decode",
        "teacher": teacher,
    }
    form._show_runs([row])
    assert form._runs.item(0).text() == (
        'cage a · kd_01 · temporal_decode · examples: ex_01 "overhead rats" · '
        "subjects from track_01"
    )
    assert "cosine similarity, -1 to 1" in form._cutoff_note.text()
    assert form.blocker() == "Set a score cut-off: dinov3_match has no default"
    form._score_cutoff.setValue(0.5)
    assert form.blocker() is None

    student = dict(teacher, method="model_003", default_cutoff=0.5, example_set=None)
    form._show_runs([dict(row, teacher=student)])
    assert form.blocker() is None, "a teacher with a default needs no cut-off"
    assert form._score_cutoff.specialValueText() == "the teacher's default (0.5)"
    form._score_cutoff.setValue(form._score_cutoff.minimum())
    assert "Left as the teacher's default: 0.5" in form._cutoff_note.text()
    form._runs.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert form.blocker() == "Tick at least one keypoint detect run"


@pytest.mark.parametrize(
    "tracking, row_text, fragment",
    [
        ([], "no tracking run on det_01", "has no finished tracking run"),
        (
            ["track_01", "track_02"],
            "2 tracking runs on det_01",
            "--tracking-run",
        ),
    ],
)
def test_a_detect_run_teacher_needs_one_tracking_run_on_its_detection_run(
    qapp, tmp_path, sync_busy, tracking, row_text, fragment
):
    """§26 *Assignment*: a detect run's poses are unassigned, and the teacher gives them to
    the one tracking run on its detection run. The row says which; a run with none or
    several is held before launch with the reason, never resolved by list order."""
    form = kd_gui.KeypointDistillForm(Experiment.create("p", tmp_path))
    form._kind.setCurrentIndex(form._kind.findData("keypoint_detect"))
    teacher = {
        "method": "dinov3_match",
        "default_cutoff": 0.5,
        "meaning": "cosine similarity",
        "low": -1.0,
        "high": 1.0,
        "example_set": None,
        "detection_run_id": "det_01",
        "tracking_run_ids": tracking,
    }
    form._show_runs(
        [
            {
                "recording_id": "recording_001",
                "recording_name": "cage a",
                "run_id": "kd_01",
                "method": "dinov3_match",
                "teacher": teacher,
            }
        ]
    )
    assert form._runs.item(0).text().endswith(row_text)
    assert fragment in form.blocker()


# --- the patch-method teacher ---------------------------------------------------------- #
def _settled(qapp, form):
    from tests.gui.conftest import drain_until

    drain_until(qapp, lambda: not form._examples._reading and not form._reading)
    qapp.processEvents()


def test_the_patch_method_is_the_default_teacher_and_holds_distill_until_complete(
    qapp, patch_teacher, sync_busy, monkeypatch
):
    it, _recording, set_id, _seen = patch_teacher
    proj = it.experiment
    captured = {}
    monkeypatch.setattr(
        proj, "distill_keypoints", lambda *a, **k: captured.update(args=a, **k) or "run"
    )
    dialog = fit_gui.DistillDialog(proj, model_type="keypoints")
    form = dialog.keypoint_form()
    dialog.show()
    _settled(qapp, form)

    assert form.teacher_kind() == "model", "the patch method needs no keypoint runs"
    assert form._examples.isVisibleTo(dialog) and not form._runs.isVisibleTo(dialog)
    form._method.setCurrentIndex(form._method.findData("fake_learner"))
    # The fixture's two recordings each carry their own tracking run; choose the one that
    # covers the recording the example set and the tracked boxes live on.
    index = next(
        i
        for i, group in enumerate(form._tracking._groups)
        if "recording_002" in group["recordings"]
    )
    form._tracking.combo.setCurrentIndex(index)
    assert "1/2 recordings" in form._tracking.combo.currentText()
    assert form._examples.set_id() == set_id
    assert form.blocker() == "Set a score cut-off: fake_learner has no default"
    assert "heatmap peak, 0 to 1" in form._cutoff_note.text()
    form._score_cutoff.setValue(0.4)
    assert form.blocker() is None and dialog._ok.isEnabled()

    dialog._name.setText("fast rats")
    dialog._run()
    assert captured["example_set"] == set_id
    assert captured["tracking"] == {"recording_002": "track_01"}
    assert captured["model_id"] == "fake_learner"
    assert captured["score_cutoff"] == pytest.approx(0.4)
    assert captured["label_compute"].backend == "cpu"
    assert "detect_runs" not in captured and dialog.run_handle == "run"

    form._kind.setCurrentIndex(form._kind.findData("keypoint_detect"))
    assert form._runs.isVisibleTo(dialog) and not form._examples.isVisibleTo(dialog)
    assert form.blocker() == "Needs a finished keypoint detect run"


def test_the_patch_teacher_says_what_is_missing(qapp, tmp_path, sync_busy, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    form = kd_gui.KeypointDistillForm(Experiment.create("p", tmp_path))
    _settled(qapp, form)
    assert form.blocker() == (
        "Choose an example set, or make one with New example set…"
    )


def test_the_patch_teacher_opens_on_an_installed_size_and_refuses_one_that_is_not(
    qapp, tmp_path, sync_busy, monkeypatch
):
    """Reported from the app: only `vit_base` installed, the form opened on `vit_large` (the
    default) and *Test cut-off…* failed inside its window. The size row now opens on what is
    installed, greys the rest, and a greyed size is refused before anything runs."""
    from detail.api import models as api_models
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    installed = {("dinov3_match", "vit_base")}
    monkeypatch.setattr(
        api_models,
        "is_installed",
        lambda model_id, variant=None: (model_id, variant) in installed,
    )
    form = kd_gui.KeypointDistillForm(Experiment.create("p", tmp_path))
    _settled(qapp, form)
    form._method.setCurrentIndex(form._method.findData("dinov3_match"))
    assert form._variant.currentData() == "vit_base", "opened on what is installed"
    assert form.blocker().startswith("Choose an example set"), "the size is fine"

    form._variant.setCurrentIndex(form._variant.findData("vit_large"))
    reason = (
        "dinov3_match vit_large is not installed: choose an installed size, or install "
        "it in the Model manager"
    )
    assert form.blocker() == reason
    assert not form._test.isEnabled() and form._test.toolTip() == reason

    installed.clear()
    form._sync_method()
    assert form.blocker().startswith("dinov3_match needs dinov3_vit")
    assert form.blocker().endswith(": install it in the Model manager")
