"""Tests for the machine-local storage locations / managed workspace (§2)."""

import pytest
import yaml

from detail.storage import locations
from detail.storage import experiment as experiment_store


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point storage.yaml + the default root at tmp so tests never touch ~/."""
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "DetailExperiments")


def test_experiments_root_defaults_and_creates(tmp_path):
    root = locations.experiments_root()
    assert root == tmp_path / "DetailExperiments"
    assert root.is_dir()


def test_list_experiment_dirs_ignores_non_experiments(tmp_path):
    assert locations.list_experiment_dirs() == []
    root = locations.experiments_root()
    experiment_store.create_experiment(root, "p1")
    experiment_store.create_experiment(root, "p2")
    (root / "not_a_experiment").mkdir()  # no config.yaml → ignored
    assert [p.name for p in locations.list_experiment_dirs()] == ["p1", "p2"]


def test_storage_yaml_overrides_root(tmp_path):
    custom = tmp_path / "custom"
    locations.STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(
        yaml.safe_dump(
            {"storage": {"default": {"type": "local", "root": str(custom)}}}
        ),
        encoding="utf-8",
    )
    assert locations.experiments_root() == custom


def test_configured_model_cache_none_when_unset(tmp_path):
    assert locations.configured_model_cache() is None
    locations.STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(
        yaml.safe_dump({"storage": {"default": {"root": str(tmp_path / "p")}}}),
        encoding="utf-8",
    )
    assert locations.configured_model_cache() is None  # root set, model_cache absent


def test_configured_model_cache_from_storage_yaml(tmp_path):
    cache = tmp_path / "shared" / "models"
    locations.STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(
        yaml.safe_dump({"storage": {"default": {"model_cache": str(cache)}}}),
        encoding="utf-8",
    )
    assert locations.configured_model_cache() == cache
