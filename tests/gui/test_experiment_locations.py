"""Experiment locations in the GUI (§2): where New experiment saves, the Experiments list
across every location, an unplugged drive, import into a chosen location, and the
Experiment locations dialog."""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import experiment_locations  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests.gui.test_export import _sync_run_progress  # noqa: E402

Accepted = pages.QDialog.DialogCode.Accepted
Rejected = pages.QDialog.DialogCode.Rejected


def _lab_drive(tmp_path, name="Lab drive"):
    folder = tmp_path / name.replace(" ", "_")
    folder.mkdir()
    locations.add_location(name, folder)
    return folder.resolve()


def _choose(picker, name: str) -> None:
    picker._combo.setCurrentIndex(picker._combo.findData(name))


def test_save_in_lists_every_location_and_greys_an_unplugged_one(qapp, tmp_path):
    _lab_drive(tmp_path)
    gone = _lab_drive(tmp_path, "Old drive")
    gone.rmdir()

    dialog = pages.NewExperimentDialog()
    combo = dialog._location._combo

    assert [combo.itemData(i) for i in range(combo.count())] == [
        "default",
        "Lab drive",
        "Old drive",
    ]
    assert combo.itemText(0).startswith("Default folder — ")
    assert combo.itemText(2).endswith("(not connected)")
    enabled = [combo.model().item(i).isEnabled() for i in range(combo.count())]
    assert enabled == [True, True, False]
    assert dialog.location() == "default"  # nothing changes for someone who never looks


def test_new_experiment_is_created_in_the_location_chosen(qapp, tmp_path, monkeypatch):
    drive = _lab_drive(tmp_path)

    def _fill_and_accept(dialog):
        dialog._name.setText("study")
        _choose(dialog._location, "Lab drive")
        return Accepted

    monkeypatch.setattr(pages.NewExperimentDialog, "exec", _fill_and_accept)
    opened = []
    page = pages.HomePage(opened.append)
    page._new_experiment()

    assert [e.dir for e in opened] == [drive / "study"]
    assert (drive / "study" / "config.yaml").is_file()


def test_add_location_from_the_new_experiment_dialog_selects_it(
    qapp, tmp_path, monkeypatch
):
    folder = tmp_path / "usb"
    folder.mkdir()

    def _fill_and_accept(dialog):
        dialog._folder.setText(str(folder))
        dialog._name.setText("USB drive")
        dialog._on_accept()
        return dialog.result()

    monkeypatch.setattr(
        experiment_locations.AddLocationDialog, "exec", _fill_and_accept
    )
    dialog = pages.NewExperimentDialog()
    dialog._location._add()

    assert dialog.location() == "USB drive"
    assert locations.location_root("USB drive") == folder.resolve()


def test_add_location_dialog_shows_why_it_refused(qapp, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(
        experiment_locations.QMessageBox, "warning", lambda *a: warned.append(a[-1])
    )
    dialog = experiment_locations.AddLocationDialog()
    dialog._on_accept()
    dialog._folder.setText(str(tmp_path / "missing"))
    dialog._name.setText("Lab drive")
    dialog._on_accept()

    assert warned[0] == "Choose a folder."
    assert "not an existing folder" in warned[1]
    assert dialog.added is None and dialog.result() != Accepted


def test_the_list_spans_locations_and_names_an_unplugged_drive(qapp, tmp_path):
    Experiment.create("home_study")
    drive = _lab_drive(tmp_path)
    Experiment.create("drive_study", location="Lab drive")
    gone = _lab_drive(tmp_path, "Old drive")
    gone.rmdir()

    page = pages.HomePage(lambda _e: None)

    assert [e.dir for e in page._experiments] == [
        locations.experiments_root() / "home_study",
        drive / "drive_study",
    ]
    tags = [
        [lab.text() for lab in card.findChildren(pages.QLabel)] for card in page._cards
    ]
    assert "· Default folder" in tags[0]
    assert "· Lab drive" in tags[1]
    assert not page._not_connected.isHidden()
    assert "Old drive" in page._not_connected.text()
    assert "plugged in" in page._not_connected.text()


def test_one_location_shows_no_tag_and_no_notice(qapp):
    Experiment.create("home_study")
    page = pages.HomePage(lambda _e: None)
    texts = [lab.text() for lab in page._cards[0].findChildren(pages.QLabel)]
    assert not any(text.startswith("· ") for text in texts)
    assert page._not_connected.isHidden()


def test_import_asks_which_location_when_there_is_more_than_one(
    qapp, tmp_path, monkeypatch
):
    drive = _lab_drive(tmp_path)
    archive = Experiment.create("study", tmp_path / "src").export(tmp_path / "s.tar")
    monkeypatch.setattr(
        pages.QFileDialog, "getOpenFileName", lambda *a, **k: (str(archive), "")
    )
    _sync_run_progress(monkeypatch)  # the import runs behind a bar (#917)

    def _pick_drive(dialog):
        _choose(dialog._picker, "Lab drive")
        return Accepted

    monkeypatch.setattr(experiment_locations.ChooseLocationDialog, "exec", _pick_drive)
    opened = []
    page = pages.HomePage(opened.append)
    page._import_experiment()

    assert [e.dir for e in opened] == [drive / "study"]


def test_import_with_one_location_asks_nothing(qapp, tmp_path, monkeypatch):
    archive = Experiment.create("study", tmp_path / "src").export(tmp_path / "s.tar")
    monkeypatch.setattr(
        pages.QFileDialog, "getOpenFileName", lambda *a, **k: (str(archive), "")
    )
    _sync_run_progress(monkeypatch)  # the import runs behind a bar (#917)

    def _never(_dialog):
        raise AssertionError("one location: nothing to choose")

    monkeypatch.setattr(experiment_locations.ChooseLocationDialog, "exec", _never)
    opened = []
    pages.HomePage(opened.append)._import_experiment()

    assert [e.dir for e in opened] == [locations.experiments_root() / "study"]


def test_locations_dialog_changes_folder_and_removes(qapp, tmp_path, monkeypatch):
    _lab_drive(tmp_path)
    moved = tmp_path / "moved"
    moved.mkdir()

    def _repoint(dialog):
        assert dialog._name.isReadOnly() and dialog._name.text() == "Lab drive"
        dialog._folder.setText(str(moved))
        dialog._on_accept()
        return dialog.result()

    monkeypatch.setattr(experiment_locations.AddLocationDialog, "exec", _repoint)
    dialog = experiment_locations.ExperimentLocationsDialog()
    buttons = {b.text() for b in dialog.findChildren(pages.QPushButton)}
    assert {"Add location…", "Change folder…", "Remove"} <= buttons

    dialog._change("Lab drive")
    assert locations.location_root("Lab drive") == moved.resolve()
    dialog._remove("Lab drive")

    assert dialog.changed
    assert [loc.name for loc in locations.experiment_locations()] == ["default"]
    assert moved.is_dir()  # removing forgets; it never deletes
