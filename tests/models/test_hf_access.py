import types

import huggingface_hub
import pytest
import yaml
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from detail.core.protocols import ModelNotReadyError
from detail.models import hf_access
from detail.storage import credentials
from detail.storage import locations

REPO = "facebook/sam3"


def _http_error(status: int) -> HfHubHTTPError:
    # auth_check raises an HfHubHTTPError carrying a response with a status code.
    # Build it without __init__ (signature varies by hf_hub version) and attach a
    # minimal response — _check_access branches on the status, not the class.
    err = HfHubHTTPError.__new__(HfHubHTTPError)
    err.response = types.SimpleNamespace(status_code=status)
    return err


@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr(hf_access, "is_installed", lambda *a, **k: False)


@pytest.fixture
def cached(monkeypatch):
    monkeypatch.setattr(hf_access, "is_installed", lambda *a, **k: True)


def _token(monkeypatch, value):
    monkeypatch.setattr(credentials, "get_hf_token", lambda: value)


# --- readiness states ------------------------------------------------------- #
def test_installed_is_ready(cached):
    r = hf_access.readiness(REPO, gated=True)
    assert r.ready and r.reason == "ok"


def test_ungated_not_installed_is_not_ready(no_cache):
    """Ungated needs no auth, but "no auth needed" is not "the weights are here":
    load() will not fetch them, so the honest verdict is not_installed."""
    r = hf_access.readiness(REPO, gated=False)
    assert not r.ready
    assert r.reason == "not_installed"


def test_not_installed_message_names_the_variants_install_command(no_cache):
    """The message must name the command that installs *this repo*.

    A varianted model resolves a different repo per size, so a message that names
    only the model id sends the user to a command that installs the default variant
    and leaves the requested one missing — the run then fails again, identically.
    """
    r = hf_access.readiness(
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        gated=False,
        install_cmd="detail model install dinov3_convnext_base --variant convnext_tiny",
    )
    assert r.reason == "not_installed"
    assert "--variant convnext_tiny" in r.message


def test_mixin_install_command_names_the_resolved_variant(monkeypatch):
    from detail.models.dinov3 import DINOv3ConvNextBase

    tiny = DINOv3ConvNextBase("convnext_tiny")
    assert (
        tiny.install_command()
        == "detail model install dinov3_convnext_base --variant convnext_tiny"
    )
    # A variant-less model has one download, so no --variant to name.
    from detail.models.sam3 import SAM3

    assert SAM3().install_command() == "detail model install sam3"


def test_no_token(no_cache, monkeypatch):
    _token(monkeypatch, None)
    r = hf_access.readiness(REPO, gated=True)
    assert not r.ready
    assert r.reason == "no_token"
    assert r.action_url == hf_access.TOKEN_SETTINGS_URL


def test_access_granted_but_not_installed(no_cache, monkeypatch):
    """Access is not posrecording. A granted gated repo whose weights are absent is
    not_installed — ready would send the run to a load() that never downloads."""
    _token(monkeypatch, "tok")
    monkeypatch.setattr(huggingface_hub, "auth_check", lambda repo_id, token=None: None)
    r = hf_access.readiness(REPO, gated=True)
    assert not r.ready
    assert r.reason == "not_installed"
    assert "model install" in r.message


def test_no_access(no_cache, monkeypatch):
    _token(monkeypatch, "tok")

    def _raise(repo_id, token=None):
        raise _http_error(403)

    monkeypatch.setattr(huggingface_hub, "auth_check", _raise)
    r = hf_access.readiness(REPO, gated=True)
    assert not r.ready
    assert r.reason == "no_access"
    assert r.action_url == hf_access.gate_url(REPO)


def test_bad_token(no_cache, monkeypatch):
    _token(monkeypatch, "tok")

    def _raise(repo_id, token=None):
        raise _http_error(401)

    monkeypatch.setattr(huggingface_hub, "auth_check", _raise)
    r = hf_access.readiness(REPO, gated=True)
    assert not r.ready and r.reason == "bad_token"


def test_offline(no_cache, monkeypatch):
    _token(monkeypatch, "tok")

    def _raise(repo_id, token=None):
        raise ConnectionError("no network")  # builtin; an OSError subclass

    monkeypatch.setattr(huggingface_hub, "auth_check", _raise)
    r = hf_access.readiness(REPO, gated=True)
    assert not r.ready
    assert r.reason == "offline"
    assert r.action_url is None


def test_unexpected_http_status_reraises(no_cache, monkeypatch):
    _token(monkeypatch, "tok")

    def _raise(repo_id, token=None):
        raise _http_error(500)

    monkeypatch.setattr(huggingface_hub, "auth_check", _raise)
    with pytest.raises(HfHubHTTPError):
        hf_access.readiness(REPO, gated=True)


# --- is_installed / offline readiness --------------------------------------- #
def test_readiness_is_ready_with_zero_network_when_installed(
    fake_hf_snapshot, monkeypatch
):
    """Guards the cache-first contract going forward: with the weights under
    model_cache_dir(), readiness() must consult nothing remote — not even the
    credential store, which it has no reason to read for installed weights.

    Note this does NOT reproduce the original incident (most of a batch killed by a
    transient Hub 504): readiness()'s guard was always correct, and the old is_cached()
    short-circuits this same fixture. The bug was that load() wrote the weights to a
    *different* cache, so the guard never saw them — that regression is pinned by
    test_load_reads_our_cache_and_never_fetches (dinov3) and
    test_load_passes_an_explicit_checkpoint_from_our_cache (sam3), which fail on the
    old code. What this test adds is a floor: no unconditional network call may creep
    back into readiness().
    """
    files = ["config.json", "sam3.pt"]
    fake_hf_snapshot(REPO, files=files)

    def _boom(*a, **k):
        raise AssertionError("readiness() consulted the network for installed weights")

    monkeypatch.setattr(huggingface_hub, "auth_check", _boom)
    monkeypatch.setattr(credentials, "get_hf_token", _boom)

    r = hf_access.readiness(REPO, gated=True, files=files)
    assert r.ready and r.reason == "ok"


def test_is_installed_false_when_a_declared_file_is_missing(fake_hf_snapshot):
    """The distinction is_cached cannot make: snapshot_download(local_files_only)
    returns the folder as soon as the *directory* exists and ignores allow_patterns
    entirely, so a partial download reads as cached. Pin both verdicts here — if
    is_installed is ever re-backed onto is_cached, this fails."""
    fake_hf_snapshot(REPO, files=["config.json"])  # sam3.pt never finished

    assert hf_access.is_cached(REPO, ["config.json", "sam3.pt"]) is True
    assert hf_access.is_installed(REPO, ["config.json", "sam3.pt"]) is False
    assert hf_access.is_installed(REPO, ["config.json"]) is True


def test_is_installed_makes_no_network_call(fake_hf_snapshot, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("is_installed() is a pure disk check")

    monkeypatch.setattr(huggingface_hub, "auth_check", _boom)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    fake_hf_snapshot(REPO, files=["sam3.pt"])
    assert hf_access.is_installed(REPO, ["sam3.pt"]) is True


def test_is_installed_false_in_empty_cache():
    assert hf_access.is_installed(REPO, ["sam3.pt"]) is False


# --- local_file ------------------------------------------------------------- #
def test_local_file_resolves_under_the_model_cache(fake_hf_snapshot):
    snapshot = fake_hf_snapshot(REPO, files=["sam3.pt"])
    assert hf_access.local_file(REPO, "sam3.pt") == snapshot / "sam3.pt"


def test_local_file_raises_actionable_when_absent(fake_hf_snapshot):
    fake_hf_snapshot(REPO, files=["config.json"])
    with pytest.raises(FileNotFoundError) as exc:
        hf_access.local_file(REPO, "sam3.pt")
    assert "model install" in str(exc.value)


# --- is_cached -------------------------------------------------------------- #
def test_is_cached_true(monkeypatch):
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: "/some/path"
    )
    assert hf_access.is_cached(REPO) is True


def test_is_cached_false(monkeypatch):
    def _raise(*a, **k):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _raise)
    assert hf_access.is_cached(REPO) is False


# --- install ---------------------------------------------------------------- #
def test_install_short_circuits_when_installed(cached, monkeypatch):
    called = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: called.append(k)
    )
    hf_access.install(REPO, gated=True)
    assert called == []


def test_install_raises_when_not_ready(no_cache, monkeypatch):
    _token(monkeypatch, None)
    with pytest.raises(ModelNotReadyError) as exc:
        hf_access.install(REPO, gated=True)
    assert exc.value.action_url == hf_access.TOKEN_SETTINGS_URL


def test_install_downloads_when_access_granted_but_not_installed(no_cache, monkeypatch):
    """not_installed is the one not-ready reason install() must act on rather than
    raise — the state every first install starts from."""
    _token(monkeypatch, "tok")
    monkeypatch.setattr(huggingface_hub, "auth_check", lambda repo_id, token=None: None)
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: calls.append((a, k)) or "/path",
    )
    hf_access.install(REPO, gated=True, files=["weights.pt"])
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (REPO,)
    assert kwargs["cache_dir"] == str(hf_access.model_cache_dir())
    assert kwargs["token"] == "tok"
    assert kwargs["allow_patterns"] == ["weights.pt"]


# --- model cache dir -------------------------------------------------------- #
def _isolate_storage_yaml(tmp_path, monkeypatch):
    """Point storage.yaml at a nonexistent tmp path so _read() returns {}."""
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")


def test_model_cache_dir_default(tmp_path, monkeypatch):
    monkeypatch.delenv(hf_access.MODEL_DIR_ENV, raising=False)
    _isolate_storage_yaml(tmp_path, monkeypatch)
    assert hf_access.model_cache_dir().parts[-3:] == (".cache", "detail", "models")


def test_model_cache_dir_env_override(tmp_path, monkeypatch):
    _isolate_storage_yaml(tmp_path, monkeypatch)
    monkeypatch.setenv(hf_access.MODEL_DIR_ENV, str(tmp_path / "weights"))
    assert hf_access.model_cache_dir() == tmp_path / "weights"


def test_model_cache_dir_from_storage_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv(hf_access.MODEL_DIR_ENV, raising=False)
    _isolate_storage_yaml(tmp_path, monkeypatch)
    custom = tmp_path / "shared" / "models"
    locations.STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(
        yaml.safe_dump({"storage": {"default": {"model_cache": str(custom)}}}),
        encoding="utf-8",
    )
    assert hf_access.model_cache_dir() == custom


def test_model_cache_dir_env_beats_storage_yaml(tmp_path, monkeypatch):
    """The env var wins over storage.yaml (explicit per-process override)."""
    _isolate_storage_yaml(tmp_path, monkeypatch)
    locations.STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(
        yaml.safe_dump(
            {"storage": {"default": {"model_cache": str(tmp_path / "from_yaml")}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(hf_access.MODEL_DIR_ENV, str(tmp_path / "from_env"))
    assert hf_access.model_cache_dir() == tmp_path / "from_env"


# --- HFHubModel mixin ------------------------------------------------------- #
def test_gate_url():
    assert hf_access.gate_url(REPO) == "https://huggingface.co/facebook/sam3"


def test_mixin_delegates(monkeypatch):
    recorded = {}

    def _readiness(repo_id, *, gated, files=None, install_cmd=None):
        recorded["readiness"] = (repo_id, gated, files)
        return "READINESS"

    def _install(repo_id, *, gated, files=None, progress_cb=None, install_cmd=None):
        recorded["install"] = (repo_id, gated, files)

    monkeypatch.setattr(hf_access, "readiness", _readiness)
    monkeypatch.setattr(hf_access, "install", _install)

    class M(hf_access.HFHubModel):
        hf_repo = REPO
        gated = True
        hf_files = ["w.pt"]

    m = M()
    assert m.readiness() == "READINESS"
    m.install()
    assert recorded["readiness"] == (REPO, True, ["w.pt"])
    assert recorded["install"] == (REPO, True, ["w.pt"])
