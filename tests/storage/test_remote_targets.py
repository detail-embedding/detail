import pytest

from detail.configs.remote import RemoteTargetConfig
from detail.storage import remote_targets


@pytest.fixture
def target_paths(tmp_path, monkeypatch):
    """Point the remote-targets file at a temp dir."""
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg / "remote_targets.yaml"
    )
    return cfg


def _config(name="tillicum", **overrides):
    fields = dict(
        name=name,
        host="tillicum",
        remote_workdir="/gscratch/scrubbed/uwnetid/detail_runs",
    )
    fields.update(overrides)
    return RemoteTargetConfig(**fields)


def test_list_empty_when_no_file(target_paths):
    assert remote_targets.list_remote_targets() == {}


def test_register_then_list_round_trips(target_paths):
    config = _config(partition="gpu-a40", modules=["cuda/12.4"])
    remote_targets.register_remote_target(config)
    targets = remote_targets.list_remote_targets()
    assert targets == {"tillicum": config}


def test_get_missing_raises_keyerror(target_paths):
    with pytest.raises(KeyError):
        remote_targets.get_remote_target("nope")


def test_get_returns_registered(target_paths):
    config = _config()
    remote_targets.register_remote_target(config)
    assert remote_targets.get_remote_target("tillicum") == config


def test_register_re_points_existing_name(target_paths):
    remote_targets.register_remote_target(_config(host="old-host"))
    remote_targets.register_remote_target(_config(host="new-host"))
    targets = remote_targets.list_remote_targets()
    assert len(targets) == 1
    assert targets["tillicum"].host == "new-host"


def test_register_multiple_targets(target_paths):
    remote_targets.register_remote_target(_config(name="tillicum"))
    remote_targets.register_remote_target(
        _config(name="hyak", host="klone.hyak.uw.edu")
    )
    targets = remote_targets.list_remote_targets()
    assert set(targets) == {"tillicum", "hyak"}


def test_remove_deletes_entry(target_paths):
    remote_targets.register_remote_target(_config())
    remote_targets.remove_remote_target("tillicum")
    assert remote_targets.list_remote_targets() == {}


def test_remove_missing_raises_keyerror(target_paths):
    with pytest.raises(KeyError):
        remote_targets.remove_remote_target("nope")


def test_file_has_schema_version_first(target_paths):
    remote_targets.register_remote_target(_config())
    text = remote_targets.REMOTE_TARGETS_PATH.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0] == "schema_version: 1"


def test_non_mapping_file_raises(target_paths):
    target_paths.mkdir(parents=True, exist_ok=True)
    (target_paths / "remote_targets.yaml").write_text(
        "just a string\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must contain a mapping"):
        remote_targets.list_remote_targets()
