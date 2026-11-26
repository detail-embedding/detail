"""Saved recording-visualization recipes on disk (§22): beside their run, schema_version first,
written atomically, numbered without reuse, no absolute path in them."""

import pytest
import yaml

from detail.storage import recording_visualizations as store

RECIPE = {
    "recording_id": "recording_001",
    "track_id": 0,
    "start_s": 1.0,
    "end_s": 2.0,
    "clusters": "named",
    "cut_height": None,
    "cut_included": [],
    "location": True,
}


def test_a_recipe_round_trips_with_schema_version_first(experiment_dir):
    path = store.write_recipe(experiment_dir, "an_01", "viz_001", RECIPE)
    assert (
        path
        == experiment_dir
        / "analysis"
        / "an_01"
        / "visualizations"
        / "viz_001"
        / "visualization.yaml"
    )
    text = path.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] == "schema_version: 1"
    assert not any(tag in text for tag in ("/Users/", "/home/", "C:\\"))
    data = store.read_recipe(experiment_dir, "an_01", "viz_001")
    assert data["id"] == "viz_001" and data["run_id"] == "an_01"
    assert {k: data[k] for k in RECIPE} == RECIPE
    assert not (path.parent / (path.name + ".tmp")).exists()


def test_ids_count_up_and_are_listed_in_order(experiment_dir):
    assert store.list_recipes(experiment_dir, "an_01") == []
    for _ in range(3):
        viz_id = store.claim_id(experiment_dir, "an_01")
        store.write_recipe(experiment_dir, "an_01", viz_id, RECIPE)
    assert store.list_recipes(experiment_dir, "an_01") == [
        "viz_001",
        "viz_002",
        "viz_003",
    ]
    store.delete_recipe(experiment_dir, "an_01", "viz_002")
    assert store.list_recipes(experiment_dir, "an_01") == ["viz_001", "viz_003"]
    newest = store.claim_id(experiment_dir, "an_01")
    assert newest == "viz_004"
    store.write_recipe(experiment_dir, "an_01", newest, RECIPE)
    store.delete_recipe(experiment_dir, "an_01", newest)
    # the newest one deleted: its number is still not handed out again
    assert store.claim_id(experiment_dir, "an_01") == "viz_005"


def test_a_missing_recipe_says_which(experiment_dir):
    with pytest.raises(
        FileNotFoundError, match="an_01 has no saved visualization viz_009"
    ):
        store.read_recipe(experiment_dir, "an_01", "viz_009")
    with pytest.raises(FileNotFoundError, match="viz_009"):
        store.delete_recipe(experiment_dir, "an_01", "viz_009")


def test_a_file_of_another_version_is_refused(experiment_dir):
    path = store.recipe_path(experiment_dir, "an_01", "viz_001")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.dump({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(
        ValueError, match="not a visualization recipe this version reads"
    ):
        store.read_recipe(experiment_dir, "an_01", "viz_001")
