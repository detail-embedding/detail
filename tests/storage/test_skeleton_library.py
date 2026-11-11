"""The machine-wide skeleton library's files (§26 *The Keypoints manager*): one
``skeleton.yaml`` per ``sk_NN`` directory under the model cache, written atomically."""

import pytest
import yaml

from detail.core.skeleton import Skeleton
from detail.models.hf_access import model_cache_dir
from detail.storage import skeleton_library as store

SK = Skeleton(
    names=("nose", "tail"), edges=(("nose", "tail"),), layout=((0.5, 0.2), (0.5, 0.8))
)


def test_create_writes_one_record_beside_the_model_cache():
    made = store.create("rats", SK)
    assert made["id"] == "sk_01"
    path = model_cache_dir() / "skeletons" / "sk_01" / "skeleton.yaml"
    text = path.read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1\n"), "schema_version is the first key"
    data = yaml.safe_load(text)
    assert list(data) == [
        "schema_version",
        "id",
        "name",
        "created_at",
        "skeleton",
        "digest",
    ]
    assert Skeleton.from_dict(data["skeleton"]) == SK
    assert data["skeleton"]["layout"] == [[0.5, 0.2], [0.5, 0.8]]
    assert data["digest"] == SK.digest()
    assert not list(path.parent.glob("*.tmp")), "no torn temp file left behind"


def test_ids_count_up_and_derived_from_is_kept():
    first = store.create("a", SK)
    second = store.create("b", SK, derived_from=first["id"])
    assert second["id"] == "sk_02" and second["derived_from"] == "sk_01"
    assert store.discover() == ["sk_01", "sk_02"]
    assert [r["name"] for r in store.list_records()] == ["a", "b"]


def test_an_invalid_skeleton_or_empty_name_writes_nothing():
    with pytest.raises(ValueError, match="needs a name"):
        store.create("  ", SK)
    with pytest.raises(ValueError):
        store.create("bad", Skeleton(names=("a", "a")))
    assert store.discover() == []


def test_read_names_what_exists_and_a_torn_file_costs_only_its_row():
    store.create("a", SK)
    with pytest.raises(FileNotFoundError, match=r"no skeleton 'sk_09'.*\['sk_01'\]"):
        store.read("sk_09")
    torn = store.library_dir() / "sk_02"
    torn.mkdir()
    (torn / "skeleton.yaml").write_text("{unclosed: [", encoding="utf-8")
    assert [r["id"] for r in store.list_records()] == ["sk_01"]


def test_remove_deletes_the_directory():
    made = store.create("a", SK)
    store.remove(made["id"])
    assert store.discover() == []
    store.remove(made["id"])  # idempotent


# --- a skeleton that arrives with a model (§12 *Keypoint models in the catalog*) ---- #
OTHER = Skeleton(names=("nose", "tail", "ear"), edges=(("nose", "tail"),))


def test_find_by_digest_matches_a_preset_or_an_entry_and_nothing_else():
    from detail.core.skeleton import PRESETS

    label, preset = PRESETS["midline_3"]
    assert store.find_by_digest(preset.digest()) == {
        "id": "midline_3",
        "name": label,
        "digest": preset.digest(),
    }
    assert store.find_by_digest(SK.digest()) is None
    made = store.create("rats", SK)
    found = store.find_by_digest(SK.digest())
    assert found == {"id": made["id"], "name": "rats", "digest": SK.digest()}


def test_unique_name_suffixes_a_taken_name_and_counts_presets():
    from detail.core.skeleton import PRESETS

    assert store.unique_name("rats") == "rats"
    store.create("rats", SK)
    assert store.unique_name("rats") == "rats (2)"
    store.create("rats (2)", OTHER)
    assert store.unique_name("rats") == "rats (3)"
    preset_label = PRESETS["midline_3"][0]
    assert store.unique_name(preset_label) == f"{preset_label} (2)"


def test_ensure_in_library_reuses_on_digest_and_adds_under_a_free_name():
    first = store.ensure_in_library(SK, "rats")
    assert first["name"] == "rats" and first["digest"] == SK.digest()
    # the same skeleton again, under any name: the same entry, no second one
    assert store.ensure_in_library(SK, "something else") == first
    assert store.discover() == [first["id"]]
    # a different skeleton under the taken name: a new entry, the old one untouched
    before = store.read(first["id"])
    second = store.ensure_in_library(OTHER, "rats")
    assert second["name"] == "rats (2)" and second["id"] != first["id"]
    assert store.read(first["id"]) == before


def test_ensure_in_library_reuses_a_preset_without_adding_an_entry():
    from detail.core.skeleton import PRESETS

    ref = store.ensure_in_library(PRESETS["quadruped_side_5"][1], "anything")
    assert ref["id"] == "quadruped_side_5"
    assert store.discover() == []
