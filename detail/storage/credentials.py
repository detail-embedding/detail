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

from detail.storage import machine_file

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
    with machine_file.editing(CREDENTIALS_PATH, mode=0o600) as data:
        data[_HF_KEY] = token
    return name


def clear_hf_token() -> None:
    """Remove the stored Hugging Face token (a no-op if none is stored).

    Only clears the credentials file; a token supplied via the ``DETAIL_HF_TOKEN``
    environment variable takes precedence and cannot be removed here.
    """
    path = CREDENTIALS_PATH  # once: the check and the edit are about one file
    if _HF_KEY not in machine_file.read(path):
        return  # nothing stored: no file to write
    with machine_file.editing(path, mode=0o600) as data:
        data.pop(_HF_KEY, None)


def _read_credentials() -> dict:
    return machine_file.read(CREDENTIALS_PATH)


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
        # The server's reason (a 401's "Invalid credentials", a 429, an outage) is the only
        # thing that tells a bad token from a bad moment; never drop it.
        said = " ".join(str(e).split())
        raise ValueError(
            "Hugging Face token is invalid or could not be validated"
            + (f": {said}" if said else "")
        ) from e
    return info["name"]
