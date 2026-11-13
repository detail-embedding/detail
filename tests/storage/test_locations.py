"""Tests for the machine-local storage locations / managed workspace (§2).

**The class the guard at the bottom pins: a consumer that finds experiments by reading the
default folder instead of every experiment location.** Before locations, ``experiments_root()``
*was* the set of experiments, and five consumers listed or placed experiments through it —
the Experiments page, the distill dialog's picker, the readiness probe, a skeleton's
dependents, import. Each would keep working after a second location existed, and simply not
see an experiment on the external drive: a plausible, shorter list, and nothing fails. The
rule: experiments are listed through ``list_experiment_dirs()`` and placed through
``location_root()``; ``experiments_root()`` is read directly only for the machine-global
trees that live beside the default location's experiments (``tools/``, ``roi_exports/``).
"""

import ast
from pathlib import Path

import pytest
import yaml

from detail.storage import locations
from detail.storage import experiment as experiment_store


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point storage.yaml + the default root at tmp so tests never touch ~/."""
    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "DetailExperiments")


def _drive(tmp_path, name="drive"):
    """A folder standing in for an external drive."""
    folder = tmp_path / name / "DetailExperiments"
    folder.mkdir(parents=True)
    return folder


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


# --- experiment locations ----------------------------------------------------------- #
def test_the_default_location_is_always_first_and_connected(tmp_path):
    found = locations.experiment_locations()
    assert [(loc.name, loc.root, loc.connected) for loc in found] == [
        ("default", tmp_path / "DetailExperiments", True)
    ]
    assert found[0].is_default


def test_an_added_location_lists_its_experiments_after_the_default_ones(tmp_path):
    experiment_store.create_experiment(locations.experiments_root(), "home_study")
    drive = _drive(tmp_path)
    experiment_store.create_experiment(drive, "drive_study")

    added = locations.add_location("Lab drive", drive)

    assert added == locations.ExperimentLocation("Lab drive", drive.resolve(), True)
    assert [loc.name for loc in locations.experiment_locations()] == [
        "default",
        "Lab drive",
    ]
    assert [p.name for p in locations.list_experiment_dirs()] == [
        "home_study",
        "drive_study",
    ]
    assert locations.location_root("Lab drive") == drive.resolve()


def test_an_unplugged_location_is_listed_as_not_connected_and_never_created(tmp_path):
    drive = _drive(tmp_path)
    experiment_store.create_experiment(drive, "drive_study")
    locations.add_location("Lab drive", drive)
    unplugged = tmp_path / "drive_unplugged"
    (tmp_path / "drive").rename(unplugged)

    found = locations.experiment_locations()

    assert [(loc.name, loc.connected) for loc in found] == [
        ("default", True),
        ("Lab drive", False),
    ]
    assert locations.list_experiment_dirs() == []
    assert not drive.exists()  # listing it must not make an empty folder in its place
    with pytest.raises(locations.LocationNotConnectedError, match="not connected"):
        locations.location_root("Lab drive")


def test_location_root_names_the_known_locations_for_an_unknown_one():
    with pytest.raises(
        ValueError, match="no experiment location named 'nope'.*default"
    ):
        locations.location_root("nope")


def test_adding_a_known_name_repoints_it_and_keeps_its_other_keys(tmp_path):
    first, second = _drive(tmp_path, "e"), _drive(tmp_path, "f")
    locations.add_location("Lab drive", first)
    data = yaml.safe_load(locations.STORAGE_PATH.read_text(encoding="utf-8-sig"))
    data["storage"]["Lab drive"]["volume_label"] = "LaCie"
    data["video_roots"] = {"lab_nas": str(tmp_path)}
    locations.STORAGE_PATH.write_text(yaml.safe_dump(data), encoding="utf-8")

    locations.add_location("Lab drive", second)

    data = yaml.safe_load(locations.STORAGE_PATH.read_text(encoding="utf-8-sig"))
    assert data["storage"]["Lab drive"] == {
        "type": "local",
        "root": second.resolve().as_posix(),
        "volume_label": "LaCie",
    }
    assert data["video_roots"] == {"lab_nas": str(tmp_path)}  # other writers' keys kept


@pytest.mark.parametrize(
    "name, match",
    [
        ("", "must start with a letter or digit"),
        ("../up", "must start with a letter or digit"),
        ("a/b", "must start with a letter or digit"),
        ("default", "built-in location"),
        ("Default", "built-in location"),
    ],
)
def test_add_location_refuses_a_bad_name(tmp_path, name, match):
    with pytest.raises(ValueError, match=match):
        locations.add_location(name, _drive(tmp_path))
    assert not locations.STORAGE_PATH.exists()


def test_add_location_refuses_a_folder_that_does_not_exist(tmp_path):
    with pytest.raises(NotADirectoryError, match="not an existing folder"):
        locations.add_location("Lab drive", tmp_path / "missing")


def test_add_location_refuses_a_folder_already_registered_under_another_name(tmp_path):
    drive = _drive(tmp_path)
    locations.add_location("Lab drive", drive)
    with pytest.raises(ValueError, match="already the experiment location 'Lab drive'"):
        locations.add_location("Other", drive)
    with pytest.raises(ValueError, match="already the experiment location 'default'"):
        locations.add_location("Home", locations.experiments_root())


def test_remove_location_forgets_the_name_and_deletes_nothing(tmp_path):
    drive = _drive(tmp_path)
    experiment_store.create_experiment(drive, "drive_study")
    locations.add_location("Lab drive", drive)

    locations.remove_location("Lab drive")

    assert [loc.name for loc in locations.experiment_locations()] == ["default"]
    assert (drive / "drive_study" / "config.yaml").is_file()


@pytest.mark.parametrize(
    "name, match",
    [("default", "cannot be removed"), ("nope", "no experiment location named")],
)
def test_remove_location_refuses_the_default_and_unknown_names(name, match):
    with pytest.raises(ValueError, match=match):
        locations.remove_location(name)


def test_location_of_names_the_location_an_experiment_sits_in(tmp_path):
    drive = _drive(tmp_path)
    locations.add_location("Lab drive", drive)
    home = experiment_store.create_experiment(locations.experiments_root(), "a")
    away = experiment_store.create_experiment(drive, "b")
    elsewhere = experiment_store.create_experiment(tmp_path / "loose", "c")

    assert locations.location_of(home) == "default"
    assert locations.location_of(away) == "Lab drive"
    assert locations.location_of(elsewhere) is None
    found = locations.experiment_locations()
    assert locations.location_of(away, among=found) == "Lab drive"


def test_a_storage_alias_that_is_not_a_local_folder_is_not_a_location(tmp_path):
    locations.STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(
        yaml.safe_dump(
            {"storage": {"s3_archive": {"type": "s3", "root": "s3://bucket/detail"}}}
        ),
        encoding="utf-8",
    )
    assert [loc.name for loc in locations.experiment_locations()] == ["default"]
    with pytest.raises(ValueError, match="no experiment location named"):
        locations.remove_location("s3_archive")


# --- guard: nothing lists experiments from the default folder alone --------------------- #
#: Modules that may call ``experiments_root()`` directly, and why. Each is a claim:
#: `test_every_direct_reader_is_still_one` fails on an entry that no longer calls it.
_READS_THE_DEFAULT_ROOT = {
    "storage/locations.py": "defines it, and lists the default location through it",
    "storage/spatial.py": (
        "roi_exports/ is a machine-global tree beside the default location's experiments, "
        "not an experiment"
    ),
    "storage/tool_runs.py": (
        "tools/ holds machine-global tool runs beside the default location's experiments, "
        "not an experiment"
    ),
}


def _direct_readers(package: Path) -> set[str]:
    readers = set()
    for source in sorted(package.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            func = getattr(node, "func", None) if isinstance(node, ast.Call) else None
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "experiments_root":
                readers.add(source.relative_to(package).as_posix())
    return readers


def _package() -> Path:
    return Path(locations.__file__).parents[1]


def test_nothing_finds_experiments_in_the_default_folder_alone():
    unexpected = sorted(_direct_readers(_package()) - set(_READS_THE_DEFAULT_ROOT))
    assert not unexpected, (
        f"{unexpected} call experiments_root() directly. That is the default location only — "
        "list experiments with locations.list_experiment_dirs() and place one with "
        "locations.location_root(name), or an experiment on an external drive is silently "
        "missing. A machine-global tree beside the experiments goes in _READS_THE_DEFAULT_ROOT "
        "with its reason."
    )


def test_every_direct_reader_is_still_one():
    stale = sorted(set(_READS_THE_DEFAULT_ROOT) - _direct_readers(_package()))
    assert not stale, f"{stale} no longer call experiments_root(); drop the entry"


def test_the_scan_sees_a_listing_through_the_default_root(tmp_path):
    package = tmp_path / "detail"
    (package / "api").mkdir(parents=True)
    (package / "api" / "lister.py").write_text(
        "from detail.storage import locations\n\n"
        "def every_experiment():\n"
        "    return list(locations.experiments_root().iterdir())\n",
        encoding="utf-8",
    )
    assert _direct_readers(package) == {"api/lister.py"}
