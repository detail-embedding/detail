"""The skeleton library's API (§26 *The Keypoints manager*): presets as read-only entries,
copy-on-use with ``skeleton_ref``, and a skeleton frozen while anything in a known experiment
uses it (found by a live search, so deleting the user frees it again)."""

import pytest
import yaml

from detail.api import skeletons as library
from detail.api.experiment import Experiment
from detail.core.skeleton import PRESETS, Skeleton
from detail.storage import locations

SK = Skeleton(names=("nose", "tail"), edges=(("nose", "tail"),))
MOVED = Skeleton(names=("nose", "tail", "ear"), edges=(("nose", "tail"),))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The experiments the app knows: whatever is under this folder."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(locations, "experiments_root", lambda: workspace)
    return workspace


def test_presets_come_first_and_are_read_only(root):
    entries = library.skeletons()
    assert [e["id"] for e in entries] == list(PRESETS)
    assert all(e["builtin"] for e in entries)
    made = library.create_skeleton("mine", SK)
    assert [e["id"] for e in library.skeletons()][-1] == made["id"]
    with pytest.raises(ValueError, match="built-in preset and cannot change"):
        library.update_skeleton("quadruped_side_5", name="x")
    with pytest.raises(ValueError, match="built-in preset and cannot be deleted"):
        library.delete_skeleton("quadruped_side_5")


def test_create_from_a_preset_a_library_id_a_dict_or_a_file(root, tmp_path):
    assert library.create_skeleton("p", "quadruped_side_5")["skeleton"] == (
        PRESETS["quadruped_side_5"][1]
    )
    first = library.create_skeleton("a", SK)
    assert library.create_skeleton("b", first["id"])["skeleton"] == SK
    assert library.create_skeleton("c", SK.to_dict())["skeleton"] == SK
    path = tmp_path / "sk.yaml"
    path.write_text(yaml.dump({"skeleton": SK.to_dict()}), encoding="utf-8")
    assert library.create_skeleton("d", path)["skeleton"] == SK
    with pytest.raises(
        ValueError, match="no skeleton preset, library skeleton or file"
    ):
        library.create_skeleton("e", "nope")


def test_ref_is_id_name_and_digest(root):
    made = library.create_skeleton("rats", SK)
    assert library.ref(made["id"]) == {
        "id": made["id"],
        "name": "rats",
        "digest": SK.digest(),
    }


def test_a_skeleton_nothing_uses_is_edited_and_deleted_in_place(root):
    made = library.create_skeleton("rats", SK)
    updated = library.update_skeleton(made["id"], name="mice", skeleton=MOVED)
    assert updated["name"] == "mice" and updated["skeleton"] == MOVED
    assert library.skeleton(made["id"])["skeleton"] == MOVED, "written"
    library.delete_skeleton(made["id"])
    with pytest.raises(FileNotFoundError):
        library.skeleton(made["id"])


def test_dependents_freeze_the_points_until_they_are_deleted(root):
    made = library.create_skeleton("rats", SK)
    a = Experiment.create("lab a", root)
    b = Experiment.create("lab b", root)
    set_a = a.create_example_set("overhead", made["id"])
    b.create_example_set("side", made["id"])
    a.create_example_set("unrelated", "quadruped_side_5")

    found = library.dependents(made["id"])
    assert [(d["experiment"], d["kind"]) for d in found] == [
        ("lab a", "example set"),
        ("lab b", "example set"),
    ]
    assert found[0]["record"] == f"examples/{set_a.id}/examples.yaml"
    assert set(library.dependents_by_skeleton()) == {made["id"], "quadruped_side_5"}

    with pytest.raises(
        ValueError, match="used by example set .*save the change as a new"
    ):
        library.update_skeleton(made["id"], skeleton=MOVED)
    with pytest.raises(ValueError, match="used by example set .*delete those first"):
        library.delete_skeleton(made["id"])
    assert (
        library.update_skeleton(made["id"], name="rats v1")["name"] == "rats v1"
    ), "a name is not identity: it changes whatever uses the skeleton"
    same = library.update_skeleton(made["id"], skeleton=SK)
    assert same["skeleton"] == SK, "an unchanged skeleton is not a change"

    edited = library.save_as_new(made["id"], "rats v2", MOVED)
    assert edited["derived_from"] == made["id"] and edited["skeleton"] == MOVED
    assert library.skeleton(made["id"])["skeleton"] == SK, "the original is untouched"

    for exp in (a, b):
        for es in exp.example_sets():
            exp.delete_example_set(es.id)
    assert library.dependents(made["id"]) == []
    library.update_skeleton(made["id"], skeleton=MOVED)
    library.delete_skeleton(made["id"])


def test_every_record_kind_that_copies_a_skeleton_is_searched(root):
    made = library.create_skeleton("rats", SK)
    exp = Experiment.create("lab", root)
    ref = {"skeleton_ref": library.ref(made["id"])}
    for rel in (
        "recordings/recording_001/keypoint_detect/kd_01/detect.yaml",
        "recordings/recording_001/keypoint_track/kt_01/track.yaml",
        "datasets/dataset_01/dataset.yaml",
        "models/model_01/model.yaml",
    ):
        path = exp.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump({"schema_version": 1, **ref}), encoding="utf-8")
    kinds = [d["kind"] for d in library.dependents(made["id"])]
    assert kinds == [
        "keypoint detect run",
        "keypoint track run",
        "keypoint dataset",
        "trained model",
    ]


def test_duplicate_export_and_import(root, tmp_path):
    copy = library.duplicate_skeleton("quadruped_side_5")
    assert copy["name"] == f"{PRESETS['quadruped_side_5'][0]} copy"
    assert copy["derived_from"] == "quadruped_side_5"
    out = tmp_path / "shared.yaml"
    library.export_skeleton(copy["id"], out)
    assert out.read_text(encoding="utf-8-sig").startswith("schema_version: 1\n")
    with pytest.raises(FileExistsError, match="already exists"):
        library.export_skeleton(copy["id"], out)
    library.export_skeleton("quadruped_overhead_12", out, overwrite=True)
    back = library.import_skeleton(out)
    assert back["name"] == PRESETS["quadruped_overhead_12"][0]
    assert back["skeleton"] == PRESETS["quadruped_overhead_12"][1]
    bare = tmp_path / "bare.yaml"
    bare.write_text(yaml.dump(SK.to_dict()), encoding="utf-8")
    assert library.import_skeleton(bare)["name"] == "bare"
    with pytest.raises(FileNotFoundError, match="no skeleton file"):
        library.import_skeleton(tmp_path / "missing.yaml")


def test_resolve_skeleton_and_its_source_name(root, tmp_path):
    # The public reader the model import uses for its skeleton= argument (§12).
    made = library.create_skeleton("rats", SK)
    assert library.resolve_skeleton(made["id"]) == SK
    assert library.skeleton_source_name(made["id"]) == "rats"
    assert library.skeleton_source_name("midline_3") == PRESETS["midline_3"][0]
    named = tmp_path / "rat_overhead.yaml"
    named.write_text(
        yaml.safe_dump({"name": "rat overhead", "skeleton": SK.to_dict()}),
        encoding="utf-8",
    )
    assert library.resolve_skeleton(named) == SK
    assert library.skeleton_source_name(named) == "rat overhead"
    bare = tmp_path / "bare.yaml"
    bare.write_text(yaml.safe_dump(SK.to_dict()), encoding="utf-8")
    assert library.skeleton_source_name(bare) == "bare"
    assert library.skeleton_source_name(SK) is None
    with pytest.raises(
        ValueError, match="no skeleton preset, library skeleton or file"
    ):
        library.resolve_skeleton("nope")
