"""Credential resolution and storage (Section 1, Tenet 14).

The Hugging Face token (and future provider secrets) live in the machine-local
credentials file or environment — never in a experiment file. ``huggingface_hub``
is imported lazily so importing this module pulls in no backend; the login
fallback degrades to ``None`` when it is absent.

Resolution order for the HF token:
    1. ``DETAIL_HF_TOKEN`` (env)
    2. ``~/.config/detail/credentials.yaml`` (key ``huggingface``)
    3. the user's existing Hugging Face login (``huggingface_hub.get_token()``)
"""

import os
from pathlib import Path

import yaml

from detail.core import atomic

# Same machine config dir as storage.yaml, and honours the same ``DETAIL_CONFIG_DIR``
# override (see storage/locations.py) so a worker — or a test — never writes a token into
# the real one.
CONFIG_DIR = Path(
    os.environ.get("DETAIL_CONFIG_DIR") or Path.home() / ".config" / "detail"
)
CREDENTIALS_PATH = CONFIG_DIR / "credentials.yaml"
HF_ENV_VAR = "DETAIL_HF_TOKEN"
_HF_KEY = "huggingface"


def get_hf_token() -> str | None:
    """Resolve the HF token by the documented source order, or ``None``."""
    env = os.environ.get(HF_ENV_VAR)
    if env:
        return env
    stored = _read_credentials().get(_HF_KEY)
    if stored:
        return stored
    return _hf_login_token()


def set_hf_token(token: str) -> str:
    """Validate ``token`` against the HF API, store it (0600), return the account name.

    Raises:
        ValueError: if the token is invalid or could not be validated.
    """
    name = _validate_hf_token(token)
    data = _read_credentials()
    data[_HF_KEY] = token
    _write_credentials(data)
    return name


def clear_hf_token() -> None:
    """Remove the stored Hugging Face token (a no-op if none is stored).

    Only clears the credentials file; a token supplied via the ``DETAIL_HF_TOKEN``
    environment variable takes precedence and cannot be removed here.
    """
    data = _read_credentials()
    if _HF_KEY in data:
        del data[_HF_KEY]
        _write_credentials(data)


def _read_credentials() -> dict:
    if not CREDENTIALS_PATH.is_file():
        return {}
    data = yaml.safe_load(atomic.read_text(CREDENTIALS_PATH))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{CREDENTIALS_PATH} must contain a mapping, got {type(data).__name__}"
        )
    return data


def _write_credentials(data: dict) -> None:
    # Atomic: the GUI writes this (off a worker thread) while a readiness probe may be
    # reading it on another thread — a truncating write could hand the reader an empty
    # file. Write a sibling tmp, then replace it in — through `core.atomic`, which rides
    # out the sharing violation a concurrent reader provokes on Windows.
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CREDENTIALS_PATH.with_name(CREDENTIALS_PATH.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    tmp.chmod(0o600)
    atomic.replace(tmp, CREDENTIALS_PATH)


def _hf_login_token() -> str | None:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def _validate_hf_token(token: str) -> str:
    try:
        import huggingface_hub
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to set the HF token; install it "
            "(e.g. `pip install -e '.[torch]'`)."
        ) from e
    try:
        info = huggingface_hub.HfApi().whoami(token=token)
    except huggingface_hub.errors.HfHubHTTPError as e:
        raise ValueError(
            "Hugging Face token is invalid or could not be validated"
        ) from e
    return info["name"]
