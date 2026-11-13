"""The camera-view picker and the three surfaces that set a view (§10).

Guards the two ways a setting goes quietly wrong in a combined save: **not touching it is not a
write** (Scene geometry must not turn an inheriting recording into one that pins today's
default when only a region changed), and **a surface shows the fact it was told** — a
recording's first option names what it inherits, so "Experiment default" is never a blank.
"""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui.camera_view import CameraViewPicker  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment.create("cv", tmp_path)
    exp.set_camera_view("side")
    return exp


def _choose(picker, view):
    picker._combo.setCurrentIndex(picker._combo.findData(view))


def test_loading_a_value_emits_nothing(qapp):
    picker = CameraViewPicker(unset_label="Not set")
    seen = []
    picker.changed.connect(lambda: seen.append(1))
    picker.set_value("other", "mirror behind")
    assert seen == []
    assert picker.value() == ("other", "mirror behind")


def test_the_note_shows_only_for_other(qapp):
    picker = CameraViewPicker(unset_label="Not set")
    picker.show()
    picker.set_value("side")
    assert not picker._note.isVisible()
    picker.set_value("other", "x")
    assert picker._note.isVisible()


def test_a_note_left_behind_is_not_part_of_a_named_view(qapp):
    picker = CameraViewPicker(unset_label="Not set")
    picker.set_value("other", "stale")
    _choose(picker, "side")
    assert picker.value() == ("side", None)


def test_choosing_other_waits_for_its_note(qapp):
    picker = CameraViewPicker(unset_label="Not set")
    seen = []
    picker.changed.connect(lambda: seen.append(1))
    _choose(picker, "other")
    assert seen == []
    picker._note.setText("45° from above")
    picker._note.editingFinished.emit()
    assert seen == [1]


def test_scene_geometry_names_what_a_recording_inherits(qapp, experiment):
    from detail.gui.arena import SceneGeometryDialog

    rec = experiment.create_recording("r1")
    dialog = SceneGeometryDialog(rec, experiment)
    assert dialog._camera_view._combo.itemText(0) == "Experiment default (Side view)"
    assert dialog._camera_view.value() == (None, None)


def test_scene_geometry_saves_the_view_and_untouched_is_no_write(qapp, experiment):
    from detail.gui.arena import SceneGeometryDialog

    rec = experiment.create_recording("r1")
    dialog = SceneGeometryDialog(rec, experiment)
    assert dialog._dirty() == []
    dialog._save_all()  # nothing touched: the recording keeps inheriting
    assert "camera_view" not in recording_store.read_recording(rec.dir)
    _choose(dialog._camera_view, "top_down")
    assert dialog._dirty() == ["camera_view"]
    dialog._save_all()
    assert rec.camera_view.view == "top_down"
    assert rec.camera_view.source == "recording"
    assert dialog._dirty() == []


def test_scene_geometry_shows_a_recordings_own_view(qapp, experiment):
    from detail.gui.arena import SceneGeometryDialog

    rec = experiment.create_recording("r1")
    rec.set_camera_view("other", "mirror behind")
    dialog = SceneGeometryDialog(rec, experiment)
    assert dialog._camera_view.value() == ("other", "mirror behind")
    assert dialog._dirty() == []


def test_scene_geometry_refuses_other_without_a_note(qapp, experiment, monkeypatch):
    from detail.gui.arena import SceneGeometryDialog

    rec = experiment.create_recording("r1")
    dialog = SceneGeometryDialog(rec, experiment)
    refused = []
    monkeypatch.setattr(dialog, "_refuse", lambda _t, message: refused.append(message))
    _choose(dialog._camera_view, "other")
    dialog._save_all()
    assert refused and "needs a description" in refused[0]
    assert rec.camera_view.source == "experiment"


def test_the_recordings_page_sets_the_default(qapp, experiment):
    from detail.gui.pages import ExperimentRecordingsPage

    from tests.gui.conftest import drain_until

    page = ExperimentRecordingsPage(experiment, on_open=lambda _id: None)
    assert drain_until(qapp, page._camera_view.isEnabled, 10_000), "never loaded"
    assert page._camera_view.value() == ("side", None)
    _choose(page._camera_view, "top_down")
    assert experiment.camera_view.view == "top_down"


def test_the_recordings_table_marks_an_inherited_view(experiment):
    from detail.gui.pages import ExperimentRecordingsPage, _recording_rows

    experiment.create_recording("a")
    experiment.create_recording("b").set_camera_view("top_down")
    _ids, rows, _active = _recording_rows(experiment)
    column = ExperimentRecordingsPage._HEADERS.index("Camera view")
    assert [row[column] for row in rows] == ["Side view (default)", "Top down"]


def test_new_experiment_dialog_refuses_other_without_a_note(qapp, monkeypatch):
    from detail.gui import pages

    dialog = pages.NewExperimentDialog()
    dialog._name.setText("x")
    warned = []
    monkeypatch.setattr(pages.QMessageBox, "warning", lambda *a: warned.append(a[-1]))
    _choose(dialog._camera_view, "other")
    dialog._on_accept()
    assert warned and "needs a description" in warned[0]
    dialog._camera_view._note.setText("overhead, tilted")
    dialog._on_accept()
    assert dialog.camera_view() == ("other", "overhead, tilted")
