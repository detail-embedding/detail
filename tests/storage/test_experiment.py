import pytest
import yaml

from detail.storage import experiment


def test_create_experiment_builds_structure(tmp_path):
    proj = experiment.create_experiment(
        tmp_path, "my_experiment", created_at="2026-01-01T00:00:00Z"
    )
    assert proj == tmp_path / "my_experiment"
    assert (proj / "config.yaml").is_file()
    for sub in ("recordings", "analysis", "models", "datasets"):
        assert (proj / sub).is_dir()

    config = experiment.read_config(proj)
    assert config["name"] == "my_experiment"
    assert config["created_at"] == "2026-01-01T00:00:00Z"
    assert config["schema_version"] == 1


def test_config_schema_version_is_first_key(tmp_path):
    proj = experiment.create_experiment(tmp_path, "p")
    first_line = (proj / "config.yaml").read_text(encoding="utf-8-sig").splitlines()[0]
    assert first_line.startswith("schema_version:")


def test_create_experiment_default_created_at_is_utc_z(tmp_path):
    proj = experiment.create_experiment(tmp_path, "p")
    created = experiment.read_config(proj)["created_at"]
    assert created.endswith("Z") and "T" in created


def test_create_experiment_twice_raises(tmp_path):
    experiment.create_experiment(tmp_path, "p")
    with pytest.raises(FileExistsError, match="already exists"):
        experiment.create_experiment(tmp_path, "p")


def test_create_experiment_empty_name_raises(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        experiment.create_experiment(tmp_path, "")


def test_read_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        experiment.read_config(tmp_path)


def test_read_config_wrong_schema_version_raises(tmp_path):
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"schema_version": 2, "name": "p"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version"):
        experiment.read_config(tmp_path)


def test_read_config_missing_name_raises(tmp_path):
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"schema_version": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="name"):
        experiment.read_config(tmp_path)


def test_read_config_non_mapping_raises(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "- a\n- b\n", encoding="utf-8"
    )  # a YAML list, not a mapping
    with pytest.raises(ValueError, match="not a mapping"):
        experiment.read_config(tmp_path)


def test_write_config_forces_schema_version_first(tmp_path):
    # input dict deliberately puts schema_version last / wrong
    experiment.write_config(
        tmp_path, {"name": "p", "created_at": "x", "schema_version": 9}
    )
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("schema_version: 1")
    assert experiment.read_config(tmp_path)["schema_version"] == 1


def test_a_failed_write_config_leaves_the_old_file_and_no_temp(tmp_path, monkeypatch):
    # config.yaml has several writers, so its temp name is per-writer — and a per-writer name
    # that is not cleaned up on failure is litter an experiment bundle would carry.
    experiment.write_config(tmp_path, {"name": "before"})

    def exhausted(_src, _dst):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(experiment.atomic, "replace", exhausted)
    with pytest.raises(PermissionError):
        experiment.write_config(tmp_path, {"name": "after"})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml"]
    assert (
        yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8-sig"))[
            "name"
        ]
        == "before"
    )


def test_discover_recordings(experiment_dir):
    assert experiment.discover_recordings(experiment_dir) == []
    for sid in ("recording_002", "recording_001"):
        d = experiment.recording_path(experiment_dir, sid)
        d.mkdir()
        (d / "recording.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    # a dir without recording.yaml is ignored
    (experiment.recordings_dir(experiment_dir) / "not_a_recording").mkdir()
    assert experiment.discover_recordings(experiment_dir) == [
        "recording_001",
        "recording_002",
    ]


def test_discover_recordings_uses_conftest_fixture(recording_dir):
    # recording_dir fixture creates recordings/recording_001/recording.yaml under experiment_dir
    proj = recording_dir.parent.parent
    assert experiment.discover_recordings(proj) == ["recording_001"]


def _proj(tmp_path):
    experiment.create_experiment(tmp_path, "p")
    return tmp_path / "p"


def test_path_helpers(tmp_path):
    assert experiment.recordings_dir(tmp_path) == tmp_path / "recordings"
    assert experiment.models_dir(tmp_path) == tmp_path / "models"
    assert experiment.recording_path(tmp_path, "s1") == tmp_path / "recordings" / "s1"


def test_experiment_path_rejects_absolute(tmp_path):
    assert experiment.experiment_path(tmp_path, "detections/det_01") == (
        tmp_path / "detections" / "det_01"
    )
    # Both flavors must be rejected on every OS: a POSIX-absolute path is not
    # absolute under Windows rules and vice versa, so the native check alone
    # would let the foreign form through (Tenet 6).
    with pytest.raises(ValueError, match="absolute"):
        experiment.experiment_path(tmp_path, "/Users/someone/data")
    with pytest.raises(ValueError, match="absolute"):
        experiment.experiment_path(tmp_path, "C:\\Users\\someone\\data")


def test_set_metadata_updates_name_and_description(tmp_path):
    proj = experiment.create_experiment(tmp_path, "proj")
    experiment.set_metadata(proj, name="renamed", description="a cohort")
    config = experiment.read_config(proj)
    assert config["name"] == "renamed"
    assert config["description"] == "a cohort"
    assert proj.name == "proj"  # the folder is not renamed


def test_set_metadata_only_description_keeps_name(tmp_path):
    proj = experiment.create_experiment(tmp_path, "proj")
    experiment.set_metadata(proj, description="just the desc")
    config = experiment.read_config(proj)
    assert config["name"] == "proj"
    assert config["description"] == "just the desc"


def test_set_metadata_empty_name_raises(tmp_path):
    proj = experiment.create_experiment(tmp_path, "proj")
    with pytest.raises(ValueError, match="non-empty"):
        experiment.set_metadata(proj, name="")
