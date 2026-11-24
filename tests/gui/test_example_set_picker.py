"""The example-set picker (§26 *Example sets follow the model that needs them*): a skeleton,
this experiment's sets on it, and the three actions. Real sets on disk; the labeling screen
is the one thing stood in for, since it needs a tracked video."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from detail.api import skeletons as library  # noqa: E402
from detail.api.experiment import Experiment  # noqa: E402
from detail.core.skeleton import Skeleton  # noqa: E402
from detail.gui import example_set_picker as mod  # noqa: E402
from detail.gui import keypoint_labeling  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

TWO = Skeleton(names=("a", "b"))


@pytest.fixture
def proj(tmp_path):
    return Experiment.create("p", tmp_path)


def _picker(qapp, proj):
    picker = mod.ExampleSetPicker(proj)
    picker.resize(700, 160)
    picker.show()
    drain_until(qapp, lambda: not picker._reading)
    qapp.processEvents()
    return picker


def _skeleton(qapp, picker, skeleton_id):
    index = picker._skeleton.findData(skeleton_id)
    assert index >= 0, skeleton_id
    picker._skeleton.setCurrentIndex(index)
    picker._skeleton.activated.emit(index)
    qapp.processEvents()


def test_sets_are_listed_under_the_skeleton_they_were_made_from(qapp, proj):
    mine = library.create_skeleton("two", TWO)
    side = proj.create_example_set("side rats", "midline_2")
    two = proj.create_example_set("two rats", mine["id"])
    loose = proj.create_example_set("loose", Skeleton(names=("x", "y", "z")))
    picker = _picker(qapp, proj)

    assert picker.skeleton_id() == "midline_2", "opens on the first set's skeleton"
    assert picker.set_id() == side.id and "0 / 0 frames" in picker._counts.text()
    _skeleton(qapp, picker, mine["id"])
    assert picker.set_id() == two.id
    _skeleton(qapp, picker, "")
    assert picker._skeleton.currentText() == mod.UNLINKED
    assert picker.set_id() == loose.id, "a set on no library skeleton is still offered"
    _skeleton(qapp, picker, "midline_3")
    assert picker.set_id() is None and not picker._mark.isEnabled()
    assert picker._new.isEnabled(), "a set can be made on any library skeleton"


def test_a_set_made_before_the_library_matches_by_its_skeleton(qapp, proj):
    from detail.storage import example_sets as store

    made = proj.create_example_set("old", "midline_2")
    current = store.read_set(proj.dir, made.id)
    current.skeleton_ref = None
    store.write_set(proj.dir, current)
    picker = _picker(qapp, proj)
    assert picker.skeleton_id() == "midline_2" and picker.set_id() == made.id


def test_new_names_a_set_then_opens_the_labeling_screen(qapp, proj, monkeypatch):
    opened = []
    monkeypatch.setattr(
        keypoint_labeling,
        "open_labeling",
        lambda _parent, _exp, set_id, **_k: opened.append(set_id) or True,
    )
    offered = []

    def choose(dialog):
        offered.append(dialog.skeleton_id())
        dialog._name.setText("overhead rats")
        dialog._skeleton.setCurrentIndex(dialog._skeleton.findData("midline_3"))
        return mod.Dialog.DialogCode.Accepted

    monkeypatch.setattr(mod.NewExampleSetDialog, "exec", choose)
    picker = _picker(qapp, proj)
    _skeleton(qapp, picker, "midline_2")
    changed = []
    picker.changed.connect(lambda: changed.append(picker.set_id()))
    picker._new.click()
    drain_until(qapp, lambda: not picker._reading and picker.set_id() is not None)
    made = proj.example_sets()[0]
    assert made.name == "overhead rats" and made.skeleton_ref["id"] == "midline_3"
    assert offered == ["midline_2"], "the window opens on the picker's skeleton"
    assert opened == [made.id] and picker.set_id() == made.id
    assert picker.skeleton_id() == "midline_3", "the picker follows the new set"
    assert changed[-1] == made.id

    picker._mark.click()
    assert opened == [made.id, made.id], "Mark frames… opens the same screen"


def test_delete_asks_and_a_set_a_run_lists_is_refused(qapp, proj, monkeypatch):
    made = proj.create_example_set("side", "midline_2")
    picker = _picker(qapp, proj)
    from detail.storage import example_sets as store

    monkeypatch.setattr(
        store, "runs_using", lambda _d, _s: [("recording_001", "kd_01")]
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda _p, _t, text: warned.append(text)
    )
    picker._delete.click()
    assert warned and "kd_01" in warned[0]
    assert proj.example_sets()[0].id == made.id

    monkeypatch.setattr(store, "runs_using", lambda _d, _s: [])
    picker._delete.click()
    drain_until(qapp, lambda: not picker._reading)
    assert proj.example_sets() == [] and picker.set_id() is None


def test_with_no_experiment_everything_says_why(qapp):
    picker = mod.ExampleSetPicker(None)
    assert not picker._new.isEnabled() and "Open an experiment" in picker._new.toolTip()
    assert picker.set_id() is None


def test_the_new_example_set_window_needs_a_name(qapp):
    entries = library.skeletons()
    dialog = mod.NewExampleSetDialog(entries, "midline_3", "example set 1")
    assert dialog.skeleton_id() == "midline_3" and dialog.name() == "example set 1"
    assert [dialog._skeleton.itemData(i) for i in range(dialog._skeleton.count())] == [
        e["id"] for e in entries
    ]
    dialog._name.setText("  ")
    dialog._name.textEdited.emit("  ")
    assert not dialog._ok.isEnabled() and "name" in dialog._ok.toolTip()


def test_a_set_is_filed_under_its_ref_only_when_that_entry_holds_its_skeleton(proj):
    """A library id means something only on the machine that minted it. A set whose
    ``skeleton_ref.id`` names an entry here holding a different skeleton (a set from another
    machine, or an id freed and reused) is filed by its digest, not under that entry."""
    from detail.core.examples import ExampleSet

    three = Skeleton(names=("a", "b", "c"))
    theirs = library.create_skeleton("fly 3pt", three)
    mine = library.create_skeleton("two", TWO)
    entries = library.skeletons()
    foreign = ExampleSet(
        id="ex_01",
        purpose="example",
        created_at="2026-01-01T00:00:00Z",
        name="from elsewhere",
        skeleton=TWO,
        skeleton_ref={"id": theirs["id"], "name": "rat", "digest": TWO.digest()},
    )
    assert mod.skeleton_of(foreign, entries) == mine["id"]
    nowhere = ExampleSet(
        id="ex_02",
        purpose="example",
        created_at="2026-01-01T00:00:00Z",
        name="nothing here",
        skeleton=Skeleton(names=("x", "y")),
        skeleton_ref={"id": theirs["id"]},
    )
    assert mod.skeleton_of(nowhere, entries) is None
    own = ExampleSet(
        id="ex_03",
        purpose="example",
        created_at="2026-01-01T00:00:00Z",
        name="own",
        skeleton=three,
        skeleton_ref={"id": theirs["id"]},
    )
    assert mod.skeleton_of(own, entries) == theirs["id"]
