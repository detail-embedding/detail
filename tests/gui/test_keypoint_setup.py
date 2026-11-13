"""**Set up keypoints…** (§26, #717): the dialog that launches a keypoint setup, the Models
page button that opens it, and the step a waiting setup's page offers.

The launch itself is ``tests/api/test_keypoint_setup.py``'s; these pin what the person sees and
what the surfaces hand the API.
"""

from __future__ import annotations

from detail.api.experiment import Experiment

_PLAN = [
    {
        "acquisition_setup": "box 1",
        "recording_id": "recording_001",
        "name": "box1_day1",
        "duration_s": 1800.0,
        "windowed": True,
        "recordings": ["recording_001", "recording_003"],
        "durations": {"recording_001": 1800.0, "recording_003": 240.0},
    },
    {
        "acquisition_setup": "top camera",
        "recording_id": "recording_002",
        "name": "top",
        "duration_s": 600.0,
        "windowed": True,
        "recordings": ["recording_002"],
        "durations": {"recording_002": 600.0},
    },
]

_DETECTOR = {
    "label": "rats  ·  model_003  ·  trained",
    "id": "model_003",
    "kind": "trained",
    "prompt_capable": False,
    "classes": ["rat", "hand"],
    "variants": [],
}

_NAMES = {
    "recording_001": "box1_day1",
    "recording_002": "top",
    "recording_003": "box1_day4",
}


def _setup_dialog(qapp, tmp_path, monkeypatch, plan=_PLAN):
    from detail.gui.keypoint_setup import KeypointSetupDialog
    from tests.gui.conftest import drain_busy_calls

    experiment = Experiment.create("p", tmp_path)
    older = {**_DETECTOR, "id": "model_001", "label": "older", "classes": ["mouse"]}
    imported = {**_DETECTOR, "id": "imp", "label": "imp", "kind": "imported"}
    monkeypatch.setattr(
        "detail.gui.modules.detection.detection_model_entries",
        lambda exp: [
            {**_DETECTOR, "id": "sam3", "prompt_capable": True},
            older,
            _DETECTOR,
            imported,
        ],
    )
    if isinstance(plan, Exception):

        def raising(*args, **kwargs):
            raise plan

        monkeypatch.setattr(experiment, "keypoint_teacher_plan", raising)
    else:
        monkeypatch.setattr(experiment, "keypoint_teacher_plan", lambda *a: plan)
    monkeypatch.setattr(experiment, "recording_names", lambda: dict(_NAMES))
    dialog = KeypointSetupDialog(experiment)
    assert not dialog._start.isEnabled()  # nothing to start until the plan is read
    drain_busy_calls(qapp)
    return experiment, dialog


def _close(qapp, *widgets):
    for widget in widgets:
        widget.deleteLater()
    qapp.processEvents()


def test_keypoint_setup_offers_one_teacher_per_setup_and_launches(
    qapp, tmp_path, monkeypatch
):
    experiment, dialog = _setup_dialog(qapp, tmp_path, monkeypatch)
    # Only closed-set detectors, opening on the newest one distilled here, and its classes.
    offered = [
        dialog._detector.itemData(i)["id"] for i in range(dialog._detector.count())
    ]
    assert offered == ["model_001", "model_003", "imp"]
    assert dialog._detector.currentData()["id"] == "model_003"
    classes = [dialog._class.itemText(i) for i in range(dialog._class.count())]
    assert classes == ["rat", "hand"]
    # Named after the animal, until the person names it.
    assert dialog._name.text() == "rat keypoints"
    dialog._class.setCurrentIndex(1)
    assert dialog._name.text() == "hand keypoints"
    dialog._class.setCurrentIndex(0)
    assert not dialog._have_examples.isChecked()  # by default the run waits for a set
    assert list(dialog._teachers) == ["box 1", "top camera"]
    assert dialog._spans["box 1"].text() == "first 5 of 30 min"
    dialog._teachers["box 1"].setCurrentIndex(1)  # the 4-minute recording: used whole
    assert dialog._spans["box 1"].text() == "all 4.0 min"
    # The student sees the whole frame unless asked otherwise (main's default, §12).
    assert dialog._student_input.currentData() == "whole_frame"
    assert not dialog._fields["student_crop_px"].isEnabled()
    assert dialog._start.isEnabled(), dialog.blocker()

    captured = {}

    def fake(name, detector_model_id, class_name, **kwargs):
        captured.update(name=name, detector=detector_model_id, cls=class_name, **kwargs)
        return object()

    monkeypatch.setattr(experiment, "start_keypoint_setup", fake)
    dialog._name.setText("rat pose")
    dialog._name.textEdited.emit("rat pose")  # what typing sends; setText does not
    dialog._class.setCurrentIndex(1)
    dialog._class.setCurrentIndex(0)
    assert dialog._name.text() == "rat pose"  # a typed name is kept
    dialog._minutes.setValue(4)
    dialog._run()
    assert (captured["name"], captured["detector"], captured["cls"]) == (
        "rat pose",
        "model_003",
        "rat",
    )
    assert captured["teachers"] == {
        "box 1": "recording_003",
        "top camera": "recording_002",
    }
    assert captured["window_s"] == 240.0
    assert captured["keypoints_model_id"] == "dinov3_match"
    assert captured["example_set"] is None
    assert captured["student_input"] == "whole_frame"
    assert captured["n_frames"] == 300
    assert dialog.run_handle is not None
    _close(qapp, dialog)


def test_keypoint_setup_given_an_example_set_does_not_wait(qapp, tmp_path, monkeypatch):
    from tests.gui.conftest import drain_busy_calls

    experiment, dialog = _setup_dialog(qapp, tmp_path, monkeypatch)
    made = experiment.create_example_set("rat examples", "quadruped_side_5")
    dialog._examples.refresh(select=made.id)
    drain_busy_calls(qapp)
    assert dialog._form.main.isRowVisible(dialog._examples) is False
    dialog._have_examples.setChecked(True)
    assert dialog._form.main.isRowVisible(dialog._examples)
    assert dialog._examples.set_id() == made.id
    # Going straight through means the cut-off is chosen now: dinov3_match has no default.
    assert not dialog._start.isEnabled()
    assert "score cut-off" in dialog._start.toolTip()
    dialog._score_cutoff.setValue(0.4)
    assert dialog._start.isEnabled(), dialog.blocker()
    captured = {}
    monkeypatch.setattr(
        experiment,
        "start_keypoint_setup",
        lambda *a, **kw: captured.update(kw) or object(),
    )
    dialog._run()
    assert captured["example_set"] == made.id
    assert captured["score_cutoff"] == 0.4
    _close(qapp, dialog)


def test_keypoint_setup_says_which_recordings_need_a_setup(qapp, tmp_path, monkeypatch):
    _experiment, dialog = _setup_dialog(
        qapp,
        tmp_path,
        monkeypatch,
        plan=ValueError(
            "2 recording(s) have no acquisition_setup: 'a', 'b'. Fill it in"
        ),
    )
    assert "no acquisition_setup: 'a', 'b'" in dialog._plan_error.text()
    assert not dialog._start.isEnabled()
    assert "no acquisition_setup" in dialog._start.toolTip()
    _close(qapp, dialog)


def test_the_models_page_opens_the_keypoint_setup_dialog(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QPushButton

    from detail.gui import keypoint_setup
    from detail.gui import pages as gui_pages

    experiment = Experiment.create("p", tmp_path)
    opened, launched = [], []
    monkeypatch.setattr(
        keypoint_setup.KeypointSetupDialog,
        "exec",
        lambda self: opened.append(self) or QDialog.DialogCode.Accepted,
    )
    page = gui_pages.ModelsPage(
        experiment, lambda _id: None, on_distill_launched=launched.append
    )
    (button,) = [
        b for b in page.findChildren(QPushButton) if b.text() == "Set up keypoints…"
    ]
    button.click()
    assert opened and launched == [None]
    _close(qapp, *opened, page)


def _waiting_setup(experiment, method="dinov3_match"):
    from detail.storage import distill_runs as distill_store

    distill_store.write_run(
        experiment.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "rat pose",
            "kind": "keypoint_setup",
            "status": "awaiting_references",
            "config": {
                "kind": "keypoint_setup",
                "keypoints_model_id": method,
                "keypoint_dataset": {"score_cutoff": None},
            },
            "stages": {
                "detect": {"status": "complete"},
                "track": {
                    "status": "complete",
                    "run_ids": {"recording_001": "trk_01"},
                },
                "examples": {"status": "awaiting_references"},
                "label": {"status": "pending"},
                "train": {"status": "pending"},
                "register": {"status": "pending"},
            },
        },
    )


def _buttons(page):
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QPushButton

    # A rebuild only deleteLater's the old buttons, and processEvents does not run that.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    return {b.text(): b for b in page.findChildren(QPushButton)}


def test_a_waiting_setup_asks_for_an_example_set_then_continues(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import pages as gui_pages
    from tests.gui.conftest import drain_busy_calls

    experiment = Experiment.create("p", tmp_path)
    _waiting_setup(experiment)
    page = gui_pages.DistillRunDetail(experiment, "distill_001")
    drain_busy_calls(qapp)
    buttons = _buttons(page)
    assert {"New example set…", "Continue setup"} <= set(buttons)
    assert {"Cancel setup", "Delete distill run"} <= set(buttons)
    assert "Resume distill" not in buttons  # it is waiting, not stopped
    # Marking happens on the teacher's tracks: the picker hands them to the labeler.
    assert page._examples._sources == {"recording_001": "trk_01"}
    assert not buttons["Continue setup"].isEnabled()  # no set chosen yet
    assert "example set" in buttons["Continue setup"].toolTip()

    made = experiment.create_example_set("rat examples", "quadruped_side_5")
    page._examples.refresh(select=made.id)
    drain_busy_calls(qapp)
    # dinov3_match sets no default cut-off (§26 open question 1): never guessed (Tenet 9).
    assert not buttons["Continue setup"].isEnabled()
    assert "dinov3_match has no default" in buttons["Continue setup"].toolTip()
    assert buttons["Test cut-off…"].isEnabled()
    page._setup_cutoff.setValue(0.4)
    assert buttons["Continue setup"].isEnabled()
    continued, live = [], []
    monkeypatch.setattr(
        experiment,
        "continue_keypoint_setup",
        lambda run_id, example_set=None, score_cutoff=None, class_example_sets=None: (
            continued.append((run_id, example_set, score_cutoff))
        )
        or "handle",
    )
    monkeypatch.setattr(page, "_start_live", live.append)
    buttons["Continue setup"].click()
    assert continued == [("distill_001", made.id, 0.4)] and live == ["handle"]
    _close(qapp, page)


def test_a_waiting_setup_says_what_its_example_set_lacks(qapp, tmp_path, monkeypatch):
    from detail.gui import pages as gui_pages
    from tests.gui.conftest import drain_busy_calls

    experiment = Experiment.create("p", tmp_path)
    _waiting_setup(experiment)
    made = experiment.create_example_set("rat examples", "quadruped_side_5")
    page = gui_pages.DistillRunDetail(experiment, "distill_001")
    page._examples.refresh(select=made.id)
    drain_busy_calls(qapp)
    page._setup_cutoff.setValue(0.4)
    told = []
    monkeypatch.setattr(
        gui_pages.QMessageBox, "warning", lambda _p, _t, text: told.append(text)
    )
    _buttons(page)["Continue setup"].click()  # the real API: a set with no frames
    assert told and "has no frames" in told[0]
    _close(qapp, page)


def test_a_waiting_setup_draws_its_example_stage(qapp):
    from detail.gui import run_units

    record = {
        "status": "awaiting_references",
        "stages": {
            "detect": {"status": "complete"},
            "track": {"status": "complete"},
            "examples": {"status": "awaiting_references"},
            "label": {"status": "pending"},
            "train": {"status": "pending"},
            "register": {"status": "pending"},
        },
    }
    units = run_units.distill_units(record)
    assert [u.name for u in units] == [
        "Detect",
        "Track",
        "Mark examples",
        "Label frames",
        "Train model",
        "Register student",
    ]
    assert units[2].status == "waiting for your example set"


# --- two or more animals (§26 *One example set per detection class*) --------------------- #
def _pick(qapp, picker, set_id):
    """Choose a set made after the picker was built: it lists sets as of its last read."""
    from tests.gui.conftest import drain_until

    picker.refresh(select=set_id)
    assert drain_until(qapp, lambda: not picker._reading), "the sets never loaded"
    qapp.processEvents()
    assert picker.set_id() == set_id


def test_two_animals_are_ticked_and_launched_as_class_names(
    qapp, tmp_path, monkeypatch
):
    """Ticked, *Two or more animals* replaces the class picker with one tick per detector
    class; Start hands ``class_names`` (never one ``class_name``) and, with sets already
    marked, one per animal."""
    from tests.gui.conftest import drain_busy_calls, drain_until

    experiment, dialog = _setup_dialog(qapp, tmp_path, monkeypatch)
    dialog.show()
    assert dialog._several.isEnabled()  # the detector finds rat and hand
    assert not dialog._form.main.isRowVisible(dialog._animals_box)
    dialog._several.setChecked(True)
    assert dialog._form.main.isRowVisible(dialog._animals_box)
    assert not dialog._form.main.isRowVisible(dialog._class)
    assert dialog.animals() == ["rat", "hand"]
    assert dialog._name.text() == "rat and hand keypoints"
    dialog._animal_ticks["hand"].setChecked(False)
    assert "Tick two animals" in dialog.blocker()
    dialog._animal_ticks["hand"].setChecked(True)
    assert dialog.blocker() is None, dialog.blocker()

    captured = {}

    def fake(name, detector_model_id, class_name=None, **kwargs):
        captured.update(cls=class_name, **kwargs)
        return object()

    monkeypatch.setattr(experiment, "start_keypoint_setup", fake)
    dialog._run()
    assert captured["cls"] is None and captured["class_names"] == ["rat", "hand"]
    assert captured["class_example_sets"] is None  # it waits for them

    # With sets already marked: one picker per animal, no ticks, and the map is handed on.
    white = experiment.create_example_set("white", "quadruped_side_5")
    black = experiment.create_example_set("black", "quadruped_side_5")
    dialog._have_examples.setChecked(True)
    assert dialog._form.main.isRowVisible(dialog._class_sets)
    assert not dialog._form.main.isRowVisible(dialog._examples)
    rows = dialog._class_sets._rows
    assert list(rows) == ["rat", "hand"]
    assert not any(tick.isVisibleTo(dialog) for tick, _p in rows.values())
    for _tick, picker in rows.values():
        assert drain_until(qapp, lambda p=picker: not p._reading)
    drain_busy_calls(qapp)
    _pick(qapp, rows["rat"][1], white.id)
    assert "Choose an example set for hand" in dialog.blocker()
    _pick(qapp, rows["hand"][1], black.id)
    dialog._score_cutoff.setValue(0.4)
    assert dialog.blocker() is None, dialog.blocker()
    dialog._run()
    assert captured["class_example_sets"] == {"rat": white.id, "hand": black.id}
    assert captured["example_set"] is None
    _close(qapp, dialog)


def test_a_waiting_two_animal_setup_asks_for_a_set_each_then_continues(
    qapp, tmp_path, monkeypatch
):
    """The teacher tracks a person marks on are the **merged** ones; each animal gets its own
    picker, and Continue hands the per-class map."""
    from detail.gui import pages as gui_pages
    from detail.storage import distill_runs as distill_store
    from tests.gui.conftest import drain_busy_calls, drain_until

    experiment = Experiment.create("p", tmp_path)
    _waiting_setup(experiment)
    record = distill_store.read_run(experiment.dir, "distill_001")
    record["config"]["class_names"] = ["rat", "hand"]
    record["stages"]["track"] = {
        "status": "complete",
        "run_ids_by_class": {
            "rat": {"recording_001": "trk_01"},
            "hand": {"recording_001": "trk_02"},
        },
    }
    record["stages"]["merge"] = {
        "status": "complete",
        "run_ids": {"recording_001": "trk_03"},
    }
    distill_store.write_run(experiment.dir, "distill_001", record)
    page = gui_pages.DistillRunDetail(experiment, "distill_001")
    drain_busy_calls(qapp)
    buttons = _buttons(page)
    rows = page._class_sets._rows
    assert list(rows) == ["rat", "hand"] and page._examples.isHidden()
    assert rows["rat"][1]._sources == {"recording_001": "trk_03"}  # the merged tracks
    assert "Choose an example set for rat, hand" in buttons["Continue setup"].toolTip()
    white = experiment.create_example_set("white", "quadruped_side_5")
    black = experiment.create_example_set("black", "quadruped_side_5")
    for _tick, picker in rows.values():
        assert drain_until(qapp, lambda p=picker: not p._reading)
    _pick(qapp, rows["rat"][1], white.id)
    _pick(qapp, rows["hand"][1], black.id)
    page._setup_cutoff.setValue(0.4)
    assert buttons["Continue setup"].isEnabled(), buttons["Continue setup"].toolTip()
    continued, live = [], []
    monkeypatch.setattr(
        experiment,
        "continue_keypoint_setup",
        lambda run_id, example_set=None, score_cutoff=None, class_example_sets=None: (
            continued.append((run_id, example_set, score_cutoff, class_example_sets))
        )
        or "handle",
    )
    monkeypatch.setattr(page, "_start_live", live.append)
    buttons["Continue setup"].click()
    assert continued == [
        ("distill_001", None, 0.4, {"rat": white.id, "hand": black.id})
    ]
    _close(qapp, page)


def test_a_two_animal_setup_draws_its_merge_stage(qapp):
    from detail.gui import run_units

    record = {
        "status": "running",
        "stages": {
            stage: {"status": "pending"}
            for stage in ("detect", "track", "merge", "examples", "label", "train")
        },
    }
    names = [u.name for u in run_units.distill_units(record)]
    assert names[:4] == ["Detect", "Track", "Merge tracks", "Mark examples"]
