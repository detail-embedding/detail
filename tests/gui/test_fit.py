"""Fit/train dialogs + the trained-model (closed-set) detection toggle (§12).

**A default selection that includes what the operation refuses** (#35). "No recordings named"
meant every recording, so one metadata-only recording — which Add recordings allows — refused
the whole labeling or distill job, and named itself by id. It means every recording *with
video* now, resolved once in ``Experiment._sampling_recordings`` so the CLI and GUI agree; a
recording that is *named* and has no video is still refused, by its name. The GUI pickers grey
such a recording and say how many selecting nothing uses.

**And one range for one field**: two labeling dialogs capped ``val_split`` at 0.9 by hand while
Generate dataset went to the model's bound; the cap is the field's ``ui_max`` now
(``test_every_labeling_dialog_offers_one_validation_split_range``).
"""

import pytest

from detail.api.experiment import Experiment
from detail.storage import datasets as datasets_store
from detail.storage import models as models_store
from detail.storage import runs as runs_store


def _experiment_with_run(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["s1"],
            "status": "complete",
            "per_recording": {},
            "config": {},
        },
    )
    return experiment


def _complete_dataset(experiment, dataset_id="dataset_001"):
    ddir = datasets_store.dataset_dir(experiment.dir, dataset_id)
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        ddir,
        {"id": dataset_id, "name": "d", "status": "complete", "n_train": 8, "n_val": 2},
    )


def _complete_model(experiment, model_id="model_001", classes=None):
    mdir = models_store.model_dir(experiment.dir, model_id)
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": "m",
            "status": "complete",
            "model_family": "yolo",
            "classes": classes or [{"id": 0, "name": "mouse"}],
            "weights": {"best": "run/best.pt", "last": None},
        },
    )


def _model(experiment, status, model_id="model_001", **extra):
    mdir = models_store.model_dir(experiment.dir, model_id)
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": "m",
            "status": status,
            "trainer_id": "ultralytics_yolo",
            "dataset_id": "dataset_001",
            **extra,
        },
    )


# --- GenerateDatasetDialog -------------------------------------------------- #
def test_generate_dataset_dialog_lists_runs_and_launches(qapp, tmp_path, monkeypatch):
    from detail.gui.fit import GenerateDatasetDialog

    experiment = _experiment_with_run(tmp_path)
    captured = {}

    def fake_generate(name, source_run_id, **kw):
        captured.update(name=name, source=source_run_id, **kw)
        return object()

    monkeypatch.setattr(experiment, "generate_dataset", fake_generate)
    dialog = GenerateDatasetDialog(experiment)
    assert dialog._source.findData("run_001") >= 0  # the complete run is offered
    dialog._name.setText("mouse_set")
    dialog._mode.setCurrentIndex(dialog._mode.findData("every_n"))
    dialog._every_n.setValue(5)
    dialog._val_split.setValue(0.2)
    dialog._min_area.setValue(50)
    dialog._run()
    assert captured["name"] == "mouse_set"
    assert captured["source"] == "run_001"
    assert captured["frame_selection"] == {"mode": "every_n", "n": 5}
    assert captured["val_split"] == pytest.approx(0.2)
    assert captured["min_contour_area"] == 50
    assert dialog.run_handle is not None


def test_generate_dataset_dialog_preselects_source(qapp, tmp_path):
    from detail.gui.fit import GenerateDatasetDialog

    experiment = _experiment_with_run(tmp_path)
    dialog = GenerateDatasetDialog(experiment, source_run_id="run_001")
    assert dialog._source.currentData() == "run_001"


def test_generate_dataset_dialog_requires_name(qapp, tmp_path, monkeypatch):
    from detail.gui import fit as fitmod

    experiment = _experiment_with_run(tmp_path)
    monkeypatch.setattr(fitmod.QMessageBox, "warning", lambda *a, **k: None)
    dialog = fitmod.GenerateDatasetDialog(experiment)
    dialog._name.setText("")  # missing
    dialog._run()
    assert dialog.run_handle is None  # not launched


# --- GenerateLabelingDialog ------------------------------------------------- #
def test_labeling_dialog_lists_recordings_and_launches(
    qapp, tmp_path, monkeypatch, stub_yolo_model
):
    from detail.gui.fit import GenerateLabelingDialog

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    experiment.create_recording("recording_002")
    captured = {}

    def fake_label(name, classes, **kw):
        captured.update(name=name, classes=classes, **kw)
        return object()

    monkeypatch.setattr(experiment, "label_dataset", fake_label)
    dialog = GenerateLabelingDialog(experiment)
    assert dialog._recordings.count() == 2  # both recordings offered
    # model picker: prompt-capable models (sam3 + the multi-variant stub; other tests may
    # leak extra registered models into the process, so assert membership, not exact contents)
    models = [dialog._model.itemText(i) for i in range(dialog._model.count())]
    assert "sam3" in models and stub_yolo_model in models
    assert dialog._model.currentText() == "sam3"  # defaults to the config default
    assert dialog._variant.currentData() is None and not dialog._variant.isEnabled()
    dialog._name.setText("mouse_set")
    dialog._class_rows[0][0].setText("mouse")
    dialog._class_rows[0][1].setText("a mouse")
    dialog._n_frames.setValue(200)
    dialog._sampling.setCurrentText("uniform")
    dialog._model.setCurrentText(
        stub_yolo_model
    )  # switch model → variant combo repopulates
    assert dialog._variant.isEnabled()
    dialog._variant.setCurrentText("11l")
    dialog._run()
    assert captured["name"] == "mouse_set"
    assert [c.name for c in captured["classes"]] == ["mouse"]
    assert captured["classes"][0].prompt == "a mouse"
    assert captured["n_frames"] == 200
    assert captured["sampling"] == "uniform"
    assert captured["source_recording_ids"] is None  # none selected → all
    assert captured["model_id"] == stub_yolo_model
    assert captured["model_variant"] == "11l"
    assert dialog.run_handle is not None


def test_labeling_dialog_requires_a_class(qapp, tmp_path, monkeypatch):
    from detail.gui import fit as fitmod

    experiment = Experiment.create("p", tmp_path)
    monkeypatch.setattr(fitmod.QMessageBox, "warning", lambda *a, **k: None)
    dialog = fitmod.GenerateLabelingDialog(experiment)
    dialog._name.setText("x")  # name set, but no class filled in
    dialog._run()
    assert dialog.run_handle is None  # not launched


def test_labeling_prompt_test_seeds_from_classes(qapp, tmp_path, stub_yolo_model):
    from detail.gui.fit import GenerateLabelingDialog
    from detail.gui.preview import PromptTestDialog

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    experiment.create_recording("recording_002")
    dialog = GenerateLabelingDialog(experiment)
    dialog._compute.backend.setCurrentText("cpu")
    dialog._class_rows[0][0].setText("mouse")
    dialog._class_rows[0][1].setText("a mouse")
    dialog._add_class_row()
    dialog._class_rows[1][0].setText("cage")
    dialog._class_rows[1][1].setText("a cage")
    dialog._model.setCurrentText(
        stub_yolo_model
    )  # prompt-test follows the selected model
    dialog._variant.setCurrentText("11m")

    preview = dialog._build_prompt_test()
    assert isinstance(preview, PromptTestDialog)
    assert preview._recording.count() == 2  # no selection → all recordings offered
    assert preview._model_id == stub_yolo_model and preview._model_variant == "11m"
    # Each class's text prompt seeds one candidate line.
    assert [p.targets[0].cue.text for p in preview._prompt_list()] == [
        "a mouse",
        "a cage",
    ]


def _with_video(recording) -> None:
    """Give ``recording`` a video part entry, as an import would have."""
    from detail.storage import recording as recording_store

    data = recording_store.read_recording(recording.dir)
    data["video"] = {"files": [{"index": 0, "file": "video/part_001.mp4"}]}
    recording_store.write_recording(recording.dir, data)


def test_labeling_prompt_test_uses_selected_recordings(qapp, tmp_path):
    from detail.gui.fit import GenerateLabelingDialog

    experiment = Experiment.create("p", tmp_path)
    # With video: the picker greys a recording without any (nothing to sample), and a greyed
    # row cannot be selected.
    _with_video(experiment.create_recording("recording_001"))
    _with_video(experiment.create_recording("recording_002"))
    dialog = GenerateLabelingDialog(experiment)
    dialog._compute.backend.setCurrentText("cpu")
    dialog._class_rows[0][0].setText("mouse")
    dialog._class_rows[0][1].setText("a mouse")
    dialog._recordings.item(0).setSelected(True)  # preview only the chosen recording

    preview = dialog._build_prompt_test()
    assert preview._recording.count() == 1
    assert preview._recording.itemText(0) == "recording_001"


def test_labeling_prompt_test_needs_a_class(qapp, tmp_path, monkeypatch):
    from detail.gui import fit as fitmod

    experiment = Experiment.create("p", tmp_path)
    monkeypatch.setattr(fitmod.QMessageBox, "information", lambda *a, **k: None)
    dialog = fitmod.GenerateLabelingDialog(experiment)
    assert dialog._build_prompt_test() is None  # nothing to test without a prompt


# --- TrainModelDialog ------------------------------------------------------- #
def test_train_model_dialog_renders_hyperparams_and_launches(
    qapp, tmp_path, monkeypatch
):
    from detail.gui.fit import TrainModelDialog

    experiment = _experiment_with_run(tmp_path)
    _complete_dataset(experiment)
    captured = {}

    def fake_train(
        name, dataset_id, trainer_id, base_model, hyperparams=None, compute=None
    ):
        captured.update(
            name=name,
            dataset_id=dataset_id,
            trainer_id=trainer_id,
            base_model=base_model,
            hyperparams=hyperparams,
        )
        return object()

    monkeypatch.setattr(experiment, "train_model", fake_train)
    dialog = TrainModelDialog(experiment)
    assert dialog._dataset.findData("dataset_001") >= 0
    assert dialog._trainer.findData("ultralytics_yolo") >= 0
    # hyperparams rendered from the trainer's pydantic config_model (§12 migration)
    assert set(dialog._hparams) == {"epochs", "batch_size", "imgsz", "patience"}
    # patience is ge=0 in the model, so the form can now reach 0 (disable early stop)
    assert dialog._hparams["patience"].minimum() == 0
    dialog._name.setText("mouse_yolo")
    dialog._hparams["epochs"].setValue(7)
    dialog._hparams["patience"].setValue(20)
    dialog._run()
    assert captured["dataset_id"] == "dataset_001"
    assert captured["trainer_id"] == "ultralytics_yolo"
    assert captured["hyperparams"]["epochs"] == 7
    assert captured["hyperparams"]["patience"] == 20
    assert dialog.run_handle is not None


def test_train_model_dialog_no_dataset_blocks_launch(qapp, tmp_path, monkeypatch):
    from detail.gui import fit as fitmod

    experiment = _experiment_with_run(tmp_path)  # no complete datasets
    monkeypatch.setattr(fitmod.QMessageBox, "warning", lambda *a, **k: None)
    dialog = fitmod.TrainModelDialog(experiment)
    dialog._name.setText("m")
    dialog._run()
    assert dialog.run_handle is None


# --- Trained-model (closed-set) detection toggle ---------------------------- #
def test_detection_config_local_lists_trained_model(qapp, tmp_path):
    from detail.core.dataclasses import Prompt
    from detail.gui.modules.detection import _DetectionConfig

    experiment = _experiment_with_run(tmp_path)
    _complete_model(
        experiment, classes=[{"id": 0, "name": "mouse"}, {"id": 1, "name": "pellet"}]
    )
    widget = _DetectionConfig()
    widget.on_experiment_set(experiment)
    # the experiment-trained model_001 sits in the one model list, beside the built-ins
    ids = [
        widget._model.itemData(i)["id"]
        for i in range(widget._model.count())
        if widget._model.itemData(i)
    ]
    assert "model_001" in ids
    widget._model.setCurrentIndex(ids.index("model_001"))
    assert "mouse" in widget._classes.text() and "pellet" in widget._classes.text()
    cfg = widget.config()
    assert cfg["model_id"] == "model_001"
    assert "weights_source" not in cfg  # the validator resolves the source from the id
    assert isinstance(cfg["prompt"], Prompt) and cfg["prompt"].targets == []
    assert widget.validation_error() is None
    # #16: a local model previews too — the button stays enabled, renamed.
    assert widget._test_btn.isEnabled()
    assert widget._test_btn.text() == "Test detection"


def test_detection_config_with_nothing_runnable_is_invalid(
    qapp, tmp_path, monkeypatch, real_machine
):
    """No trained or imported models, and the built-in needs a GPU this machine lacks —
    so the form says so and refuses the run, instead of offering an unrunnable default.
    """
    from detail.gui.modules.detection import _DetectionConfig

    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    experiment = _experiment_with_run(tmp_path)  # no trained or imported models
    widget = _DetectionConfig()
    widget.on_experiment_set(experiment)
    assert widget.config()["model_id"] is None
    assert widget.validation_error() == (
        "No detection models available. Install models via Model manager."
    )


def test_detection_config_pretrained_default_unchanged(qapp, tmp_path):
    from detail.gui.modules.detection import _DetectionConfig

    widget = _DetectionConfig()
    cfg = widget.config()
    assert cfg["model_id"] == "sam3"
    assert "weights_source" not in cfg  # resolved by the validator from the id
    assert cfg["prompt"].targets  # non-empty prompt


# --- detail-page delete buttons (§1.1 principle 6) -------------------------- #
def test_dataset_detail_delete_button_calls_api_and_navigates(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import pages
    from detail.gui.pages import DatasetDetailPage

    experiment = _experiment_with_run(tmp_path)
    _complete_dataset(experiment)
    called, navigated = {}, {}
    monkeypatch.setattr(
        experiment, "delete_dataset", lambda did: called.setdefault("id", did)
    )
    monkeypatch.setattr(
        pages.QMessageBox,
        "question",
        lambda *a, **k: pages.QMessageBox.StandardButton.Yes,
    )
    page = DatasetDetailPage(
        experiment, "dataset_001", on_deleted=lambda: navigated.setdefault("nav", True)
    )
    page._delete()
    assert called["id"] == "dataset_001"
    assert navigated.get("nav") is True


def test_model_detail_delete_button_calls_api_and_navigates(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import pages
    from detail.gui.pages import ModelDetailPage

    experiment = _experiment_with_run(tmp_path)
    _complete_model(experiment)
    called, navigated = {}, {}
    monkeypatch.setattr(
        experiment, "delete_model", lambda mid: called.setdefault("id", mid)
    )
    monkeypatch.setattr(
        pages.QMessageBox,
        "question",
        lambda *a, **k: pages.QMessageBox.StandardButton.Yes,
    )
    page = ModelDetailPage(
        experiment, "model_001", on_deleted=lambda: navigated.setdefault("nav", True)
    )
    page._delete()
    assert called["id"] == "model_001"
    assert navigated.get("nav") is True


def test_model_detail_resume_button_calls_api(qapp, tmp_path, monkeypatch):
    from detail.gui.qt import QPushButton

    from detail.gui.pages import ModelDetailPage

    experiment = _experiment_with_run(tmp_path)
    _model(experiment, "cancelled")  # killed mid-run: its checkpoint can resume
    called = {}

    def fake_resume(mid):
        called["id"] = mid
        return object()

    monkeypatch.setattr(experiment, "resume_training", fake_resume)
    page = ModelDetailPage(experiment, "model_001")
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Resume training" in labels  # shown for a resumable (cancelled) model
    monkeypatch.setattr(page, "_start_live", lambda run: called.setdefault("live", run))
    page._resume()
    assert called["id"] == "model_001"
    assert "live" in called  # the resumed run is streamed on the page


@pytest.mark.parametrize(
    "status, extra",
    [
        ("complete", {"stopped_at_epoch": 37}),  # Stop and keep, since #486
        ("partial", {}),  # stopped cleanly before #486: finished, checkpoint finalized
    ],
)
def test_a_model_that_stopped_cleanly_offers_no_resume(qapp, tmp_path, status, extra):
    """Resume on a finalized checkpoint trained a new model on Ultralytics' COCO8 sample
    (#486), so a model whose run *returned* — Stop and keep, or a legacy `partial` — must not
    offer it. A Stop-and-keep model also says it stopped early."""
    from detail.gui.qt import QLabel, QPushButton

    from detail.gui.pages import ModelDetailPage

    experiment = _experiment_with_run(tmp_path)
    _model(experiment, status, **extra)
    page = ModelDetailPage(experiment, "model_001")
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Resume training" not in labels, f"Resume offered on a {status!r} model"
    texts = " ".join(label.text() for label in page.findChildren(QLabel))
    if extra:
        assert "at epoch 37" in texts, "a stopped model does not say where it stopped"


def test_train_model_dialog_backend_defaults_to_cuda_when_available(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import fit as fitmod

    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    experiment = _experiment_with_run(tmp_path)
    _complete_dataset(experiment)
    dialog = fitmod.TrainModelDialog(experiment)
    assert dialog._compute.backend.currentText() == "cuda"  # a GPU box defaults to GPU


def test_train_model_dialog_backend_defaults_to_mps_without_cuda(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import fit as fitmod

    # no NVIDIA GPU but an Apple accelerator: default to mps, not cpu (the trainer
    # supports it), never to a cuda that isn't there.
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    experiment = _experiment_with_run(tmp_path)
    _complete_dataset(experiment)
    dialog = fitmod.TrainModelDialog(experiment)
    assert dialog._compute.backend.currentText() == "mps"


def test_label_dataset_dialog_backend_defaults_to_best_available(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import fit as fitmod

    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    experiment = _experiment_with_run(tmp_path)
    dialog = fitmod.GenerateLabelingDialog(experiment)
    assert (
        dialog._compute.backend.currentText() == "mps"
    )  # best device, not a hardcoded cuda


def test_dataset_detail_hides_delete_while_active(qapp, tmp_path):
    from detail.gui.qt import QPushButton

    from detail.gui.pages import DatasetDetailPage
    from detail.storage import datasets as datasets_store

    experiment = _experiment_with_run(tmp_path)
    ddir = datasets_store.dataset_dir(experiment.dir, "dataset_001")
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(ddir, {"id": "dataset_001", "status": "running"})
    page = DatasetDetailPage(experiment, "dataset_001")
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Delete dataset" not in labels  # no delete while still generating


# --- DistillDialog ---------------------------------------------------------- #
def test_distill_dialog_renders_and_launches(qapp, tmp_path, monkeypatch):
    from detail.configs.distill import DEFAULT_N_FRAMES
    from detail.gui.fit import DistillDialog
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    experiment = Experiment.create("p", tmp_path)
    captured = {}

    def fake_distill(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(experiment, "distill", fake_distill)
    dialog = DistillDialog(experiment)
    assert dialog._n_frames.value() == DEFAULT_N_FRAMES == 1000
    assert dialog._trainer.findData("ultralytics_yolo") >= 0
    # training hyperparams come from the trainer's pydantic config_model (§12)
    assert set(dialog._hparams) == {"epochs", "batch_size", "imgsz", "patience"}
    dialog._name.setText("mouse_fast")
    dialog._class_rows[0][0].setText("mouse")
    dialog._class_rows[0][1].setText("a mouse")
    dialog._n_frames.setValue(120)
    dialog._hparams["epochs"].setValue(7)
    dialog._run()
    assert captured["name"] == "mouse_fast"
    assert [c.name for c in captured["classes"]] == ["mouse"]
    assert captured["n_frames"] == 120
    assert captured["hyperparams"]["epochs"] == 7
    # compute is per stage (labeling is CUDA-only; training is separate)
    assert captured["label_compute"] is not None
    assert captured["train_compute"] is not None
    assert dialog.run_handle is not None


@pytest.mark.parametrize("dialog_name", ["GenerateLabelingDialog", "DistillDialog"])
def test_labeling_dialogs_pass_every_labeling_field(
    qapp, tmp_path, monkeypatch, dialog_name
):
    """Both dialogs that launch a labeling stage hand every LabelingConfig field to the API,
    each at the model's default when untouched — driven from the model, so a field added to it
    and left off a dialog fails here instead of silently running at the API default."""
    from detail.configs.labeling import LabelingConfig
    from detail.gui import fit
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    experiment = Experiment.create("p", tmp_path)
    captured = {}

    def fake(name, *args, **kwargs):
        captured.update(kwargs)
        return object()

    distill = dialog_name == "DistillDialog"
    monkeypatch.setattr(experiment, "distill" if distill else "label_dataset", fake)
    dialog = getattr(fit, dialog_name)(experiment)
    dialog._name.setText("d")
    dialog._class_rows[0][0].setText("mouse")
    dialog._class_rows[0][1].setText("a mouse")
    dialog._run()
    for field, info in LabelingConfig.model_fields.items():
        key = f"label_{field}" if distill and field.startswith("model_") else field
        assert key in captured, f"{dialog_name} does not pass {field}"
        if not info.is_required() and field not in ("model_id", "model_variant"):
            want = info.default
            assert captured[key] == (
                pytest.approx(want) if isinstance(want, float) else want
            ), field


def test_distill_dialog_requires_a_class(qapp, tmp_path, monkeypatch):
    from detail.gui import fit as fitmod
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    experiment = Experiment.create("p", tmp_path)
    monkeypatch.setattr(fitmod.QMessageBox, "warning", lambda *a, **k: None)
    dialog = fitmod.DistillDialog(experiment)
    dialog._name.setText("m")  # name set, but no class rows filled
    dialog._run()
    assert dialog.run_handle is None


def test_model_detail_promote_calls_promote_model(qapp, tmp_path, monkeypatch):
    from detail.gui import pages as pagesmod

    experiment = _experiment_with_run(tmp_path)
    _complete_model(experiment)
    called = {}

    def fake_promote(model_id, name=None):
        called["model_id"] = model_id
        called["name"] = name
        return "mouse_glob"

    monkeypatch.setattr(experiment, "promote_model", fake_promote)
    monkeypatch.setattr(
        pagesmod.QInputDialog, "getText", lambda *a, **k: ("shared", True)
    )
    # run the (off-thread) promote synchronously for the test
    monkeypatch.setattr(
        pagesmod, "run_busy", lambda target, fn, on_done, **k: on_done(fn())
    )
    monkeypatch.setattr(pagesmod.QMessageBox, "information", lambda *a, **k: None)
    page = pagesmod.ModelDetailPage(experiment, "model_001")
    page._promote()
    assert called == {"model_id": "model_001", "name": "shared"}


def test_distill_run_detail_opens_trained_model(qapp, tmp_path):
    from detail.gui.qt import QPushButton

    from detail.gui.pages import DistillRunDetail
    from detail.storage import distill_runs as distill_store

    experiment = _experiment_with_run(tmp_path)
    distill_store.write_run(
        experiment.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "m",
            "status": "complete",
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "complete", "model_id": "model_001"},
            },
        },
    )
    opened = {}
    page = DistillRunDetail(
        experiment,
        "distill_001",
        on_open_model=lambda mid: opened.setdefault("model_id", mid),
    )
    labels = {b.text(): b for b in page.findChildren(QPushButton)}
    assert (
        "Open trained model →" in labels
    )  # reaches Promote to global… on the model page
    labels["Open trained model →"].click()
    assert opened["model_id"] == "model_001"


def test_distill_run_detail_hides_open_model_before_training_completes(qapp, tmp_path):
    from detail.gui.qt import QPushButton

    from detail.gui.pages import DistillRunDetail
    from detail.storage import distill_runs as distill_store

    experiment = _experiment_with_run(tmp_path)
    distill_store.write_run(
        experiment.dir,
        "distill_002",
        {
            "id": "distill_002",
            "name": "m",
            "status": "running",
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "running", "model_id": "model_002"},
            },
        },
    )
    page = DistillRunDetail(experiment, "distill_002", on_open_model=lambda _id: None)
    labels = {b.text() for b in page.findChildren(QPushButton)}
    assert "Open trained model →" not in labels  # train hasn't produced a model yet


def test_distill_dialog_picks_experiment_from_workspace(qapp, tmp_path, monkeypatch):
    from detail.gui.fit import DistillDialog
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = Experiment.create("study", tmp_path)
    exp.create_recording("rec_1")
    dialog = DistillDialog()  # nothing pre-selected — picks from the workspace
    assert dialog._experiment is not None and dialog._experiment.dir == exp.dir
    assert (
        dialog._recordings.count() == 1
    )  # the selected experiment's recordings populate


def test_the_sampling_pickers_say_what_none_selected_uses(qapp, tmp_path):
    """#35 finding 5, the GUI half. "Recordings (none = all)" was a promise the API broke on
    the first recording without video. Both pickers grey such a recording, say why, and the
    row label says how many recordings selecting nothing uses."""
    from detail.gui.qt import QLabel, Qt

    from detail.gui.fit import DistillDialog, GenerateLabelingDialog

    experiment = Experiment.create("p", tmp_path)
    _with_video(experiment.add_recording("cage a"))
    experiment.add_recording("notes only")

    for dialog in (GenerateLabelingDialog(experiment), DistillDialog(experiment)):
        rows = [dialog._recordings.item(i) for i in range(dialog._recordings.count())]
        assert [r.text() for r in rows] == ["cage a", "notes only   (no video)"]
        assert [bool(r.flags() & Qt.ItemFlag.ItemIsEnabled) for r in rows] == [
            True,
            False,
        ]
        texts = [w.text() for w in dialog.findChildren(QLabel)]
        assert "Recordings (none = all 1 with video):" in texts


def test_every_labeling_dialog_offers_one_validation_split_range(qapp, tmp_path):
    """#35 finding 6: two dialogs capped `val_split` at 0.9 by hand while Generate dataset went
    to the model's bound. The cap is the field's `ui_max` now, so all three agree, and each
    opens on the model's default."""
    from detail.configs.labeling import LabelingConfig
    from detail.gui.fit import (
        DistillDialog,
        GenerateDatasetDialog,
        GenerateLabelingDialog,
    )

    experiment = _experiment_with_run(tmp_path)
    dialogs = [  # held: a dialog garbage-collected takes its spin box with it
        GenerateDatasetDialog(experiment),
        GenerateLabelingDialog(experiment),
        DistillDialog(experiment),
    ]
    spins = [d._val_split for d in dialogs]
    assert {(s.minimum(), s.maximum()) for s in spins} == {(0.0, 0.9)}
    default = LabelingConfig.model_fields["val_split"].default
    assert [s.value() for s in spins] == [pytest.approx(default)] * 3
