import os

import huggingface_hub
import pytest
from huggingface_hub.errors import HfHubHTTPError

from detail.storage import credentials


@pytest.fixture
def cred_paths(tmp_path, monkeypatch):
    """Point the credential file at a temp dir and clear the env var."""
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(credentials, "CONFIG_DIR", cfg)
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", cfg / "credentials.yaml")
    monkeypatch.delenv(credentials.HF_ENV_VAR, raising=False)
    return cfg


def _write_file(cred_paths, text):
    cred_paths.mkdir(parents=True, exist_ok=True)
    (cred_paths / "credentials.yaml").write_text(text, encoding="utf-8")


def test_env_var_wins_over_file(cred_paths, monkeypatch):
    _write_file(cred_paths, "huggingface: filetok\n")
    monkeypatch.setenv(credentials.HF_ENV_VAR, "envtok")
    assert credentials.get_hf_token() == "envtok"


def test_file_used_when_no_env(cred_paths, monkeypatch):
    _write_file(cred_paths, "huggingface: filetok\n")
    # login fallback must not be consulted when the file provides the token
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "logintok")
    assert credentials.get_hf_token() == "filetok"


def test_login_fallback_when_no_env_or_file(cred_paths, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "logintok")
    assert credentials.get_hf_token() == "logintok"


def test_none_when_nothing_available(cred_paths, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert credentials.get_hf_token() is None


def test_non_mapping_file_raises(cred_paths):
    _write_file(cred_paths, "just a string\n")
    with pytest.raises(ValueError, match="must contain a mapping"):
        credentials.get_hf_token()


def test_set_hf_token_validates_writes_and_chmods(cred_paths, monkeypatch):
    class _Api:
        def whoami(self, token):
            assert token == "hf_good"
            return {"name": "alice"}

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    assert credentials.set_hf_token("hf_good") == "alice"
    path = credentials.CREDENTIALS_PATH
    assert "hf_good" in path.read_text(encoding="utf-8-sig")
    # Windows does not enforce POSIX mode bits, so chmod(0o600) is a no-op there;
    # only assert the 0o600 hardening where it is meaningful.
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600
    # round-trips through get_hf_token
    assert credentials.get_hf_token() == "hf_good"


def test_clear_hf_token_removes_stored_token(cred_paths, monkeypatch):
    class _Api:
        def whoami(self, token):
            return {"name": "alice"}

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)  # no login fallback
    credentials.set_hf_token("hf_good")
    assert credentials.get_hf_token() == "hf_good"
    credentials.clear_hf_token()
    assert credentials.get_hf_token() is None
    credentials.clear_hf_token()  # idempotent — no error when nothing is stored


def test_set_hf_token_invalid_raises(cred_paths, monkeypatch):
    class _Api:
        def whoami(self, token):
            # Built without __init__: HfHubHTTPError's signature varies by hf_hub
            # version (newer requires a keyword-only `response`); only type matters.
            raise HfHubHTTPError.__new__(HfHubHTTPError)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    with pytest.raises(ValueError, match="invalid or could not be validated"):
        credentials.set_hf_token("hf_bad")
