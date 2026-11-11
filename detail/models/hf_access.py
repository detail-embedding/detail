"""Readiness and install for Hugging Face Hub models (Section 5).

The cache-first readiness tree and the ``HFHubModel`` mixin live here so every HF
model (SAM3, DINOv3, ...) shares one implementation. ``huggingface_hub`` is
imported lazily inside functions: importing this module pulls in no backend
(Tenet 13). Weights cache to ``~/.cache/detail/models/`` (the ``auto_download``
row of the §5 weight-sourcing table).

``model_cache_dir()`` is the single owner of that location, and **every** call site
passes it explicitly (``cache_dir=``, or a path resolved via ``local_file``).
Steering the location through a process env var (``HF_HUB_CACHE``, ``HF_HOME``,
``TRANSFORMERS_CACHE``) is prohibited: ``huggingface_hub`` snapshots those into
module constants at *import* time, so an ``os.environ`` write after the first
``import huggingface_hub`` is a silent no-op — and ``readiness()`` imports it
(``_require_hf``) before ``load()`` ever runs, so "set it earlier" is not a fix.
Getting this wrong splits the cache in two and falsifies ``is_installed()``
forever, which makes every run consult the network (§5).
"""

import os
from collections.abc import Callable
from pathlib import Path

from detail.core.dataclasses import InstallProgress, ModelReadiness
from detail.core.protocols import ModelNotReadyError
from detail.storage import credentials
from detail.storage import locations

_DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "detail" / "models"
MODEL_DIR_ENV = "DETAIL_MODEL_DIR"
TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"


def model_cache_dir() -> Path:
    """Where auto-downloaded weights live, shared across experiments.

    Resolution order: the ``DETAIL_MODEL_DIR`` env var, then the ``default``
    alias's ``model_cache`` in ``~/.config/detail/storage.yaml`` (§2), then
    the ``~/.cache/detail/models/`` default. The resolved path is passed to
    the backend explicitly (``cache_dir=``) — never steered via ``HF_HUB_CACHE``.
    """
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser()
    configured = locations.configured_model_cache()
    if configured is not None:
        return configured
    return _DEFAULT_MODEL_CACHE


def gate_url(repo_id: str) -> str:
    """The repo page that hosts the gated-access request form."""
    return f"https://huggingface.co/{repo_id}"


def is_cached(repo_id: str, files: list[str] | None = None) -> bool:
    """True if the repo's snapshot folder is in the local cache. No network.

    Weaker than it looks, and weaker than ``is_installed``: on the
    ``local_files_only`` path ``snapshot_download`` returns the snapshot folder as
    soon as the *directory* exists and ignores ``allow_patterns`` entirely ("we
    can't check if all the files are actually there" — its own source). A
    half-finished download therefore reports True here. Prefer ``is_installed``.
    """
    hf = _require_hf()
    try:
        hf.snapshot_download(
            repo_id,
            cache_dir=str(model_cache_dir()),
            local_files_only=True,
            allow_patterns=files,
        )
        return True
    except (hf.errors.LocalEntryNotFoundError, FileNotFoundError):
        return False


def is_installed(repo_id: str, files: list[str] | None = None) -> bool:
    """True if every file in ``files`` is present in the local cache. No network.

    ``files`` is the model's manifest (``hf_files``) — load-bearing for correctness,
    not a download-size optimization: it is what this verifies *and* what
    ``install`` downloads, so the two agree by construction. Without a manifest we
    can only fall back to ``is_cached``'s snapshot-folder check, which reports True
    for a partial download.
    """
    hf = _require_hf()
    if not files:
        return is_cached(repo_id)
    cache_dir = str(model_cache_dir())
    # try_to_load_from_cache returns the path (str), None when absent, or the
    # _CACHED_NO_EXIST sentinel — only a str means "the file is here".
    return all(
        isinstance(hf.try_to_load_from_cache(repo_id, name, cache_dir=cache_dir), str)
        for name in files
    )


def local_file(repo_id: str, filename: str) -> Path:
    """The cached path of one file, for a loader that takes an explicit path. No network.

    Raises rather than downloading: ``load()`` runs inside a pool worker subprocess
    with no channel to report a multi-GB fetch, so installing there is invisible by
    construction (§5).
    """
    hf = _require_hf()
    hit = hf.try_to_load_from_cache(repo_id, filename, cache_dir=str(model_cache_dir()))
    if not isinstance(hit, str):
        raise FileNotFoundError(
            f"{filename} for {repo_id} is not in the local model cache "
            f"({model_cache_dir()}). Install the model first — `detail model "
            f"install <model_id>`, or Install on the GUI's Models tab. A run will "
            f"not download it."
        )
    return Path(hit)


def readiness(
    repo_id: str,
    *,
    gated: bool,
    files: list[str] | None = None,
    install_cmd: str | None = None,
) -> ModelReadiness:
    """Cache-first status (Section 5). Network is consulted only when not installed.

    ``install_cmd`` is the exact command that installs *this* repo — a varianted
    model resolves a different repo per size, so a message naming only the model id
    sends the user to a command that installs a different variant.
    """
    if is_installed(repo_id, files):
        return _ok(repo_id)
    if not gated:
        # Ungated: no auth needed, the weights are simply not here yet.
        return _not_installed(repo_id, install_cmd)
    token = credentials.get_hf_token()
    if token is None:
        return ModelReadiness(
            ready=False,
            reason="no_token",
            message=(
                f"A Hugging Face token is required to download {repo_id}. "
                f"Create one at {TOKEN_SETTINGS_URL} and call set_hf_token()."
            ),
            action_url=TOKEN_SETTINGS_URL,
            repo_id=repo_id,
        )
    return _check_access(repo_id, token, install_cmd)


def install(
    repo_id: str,
    *,
    gated: bool,
    files: list[str] | None = None,
    progress_cb: Callable[[InstallProgress], None] | None = None,
    install_cmd: str | None = None,
) -> None:
    """Download the repo's weights to the cache. Idempotent; raises if not ready.

    The only function that may download (§5).
    """
    status = readiness(repo_id, gated=gated, files=files, install_cmd=install_cmd)
    if status.ready:
        return  # readiness is "ok" only when is_installed() — nothing to fetch
    # "not_installed" is the one not-ready reason install() exists to resolve;
    # treating it as fatal (the obvious `if not status.ready: raise`) would make
    # install() incapable of ever installing anything.
    if status.reason != "not_installed":
        raise ModelNotReadyError(status)
    hf = _require_hf()
    kwargs: dict = {}
    if progress_cb is not None:
        kwargs["tqdm_class"] = _make_tqdm_class(progress_cb)
    hf.snapshot_download(
        repo_id,
        cache_dir=str(model_cache_dir()),
        token=credentials.get_hf_token(),
        allow_patterns=files,
        **kwargs,
    )


class HFHubModel:
    """Mixin: readiness/install for a model whose weights live in an HF repo.

    Concrete models set ``hf_repo`` (and optionally ``gated`` / ``hf_files``) and
    inherit the cache-first readiness + download. Compose with a protocol ABC,
    e.g. ``class SAM3(HFHubModel, DetectionModel): ...`` — the mixin's methods take
    precedence over the protocol defaults.
    """

    hf_repo: str
    gated: bool = True
    hf_files: list[str] | None = None
    # Models whose weights sit behind ONE access agreement share a gate id. Meta gates
    # the DINOv3 collection as a family, so accepting once covers every backbone and
    # size — listing a request link per model would send a user to fill in the same form
    # repeatedly. ``None`` means the gate is the repo itself (SAM 3).
    hf_gate: str | None = None

    def readiness(self) -> ModelReadiness:
        return readiness(
            self.hf_repo,
            gated=self.gated,
            files=self.hf_files,
            install_cmd=self.install_command(),
        )

    def install(
        self, progress_cb: Callable[[InstallProgress], None] | None = None
    ) -> None:
        install(
            self.hf_repo,
            gated=self.gated,
            files=self.hf_files,
            progress_cb=progress_cb,
            install_cmd=self.install_command(),
        )

    def install_command(self) -> str:
        """The CLI command that installs *this instance's* weights.

        Each size variant is its own repo and its own download, so a message that
        names only the model id points at whichever variant is the default —
        following it leaves the requested one still missing (§5).
        """
        model_id = getattr(self, "model_id", None)
        if not model_id:
            return "detail model install <model_id>"
        variant = getattr(self, "variant", None)
        if not variant:
            return f"detail model install {model_id}"
        return f"detail model install {model_id} --variant {variant}"

    def is_installed(self) -> bool:
        return is_installed(self.hf_repo, self.hf_files)

    def is_ready(self) -> bool:
        return self.readiness().ready

    def local_file(self, filename: str) -> Path:
        """Cached path of one of this model's files — for a loader taking a path."""
        return local_file(self.hf_repo, filename)

    def weights_dir(self) -> Path:
        """Local snapshot path for the cached weights (call after install)."""
        hf = _require_hf()
        return Path(
            hf.snapshot_download(
                self.hf_repo,
                cache_dir=str(model_cache_dir()),
                local_files_only=True,
                allow_patterns=self.hf_files,
            )
        )


def _check_access(
    repo_id: str, token: str, install_cmd: str | None = None
) -> ModelReadiness:
    hf = _require_hf()
    try:
        hf.auth_check(repo_id, token=token)
    except hf.errors.HfHubHTTPError as e:
        # auth_check got an HTTP response. Classify by status code, not exception
        # class: across hf_hub versions this is sometimes the base HfHubHTTPError
        # (1.x) and sometimes a GatedRepoError / RepositoryNotFoundError subclass.
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 403:
            return ModelReadiness(
                ready=False,
                reason="no_access",
                message=(
                    f"Access to {repo_id} is forbidden (HTTP 403). Check that you have "
                    f"requested access at {gate_url(repo_id)}, and that your token allows "
                    f"gated repos — for a fine-grained token, enable 'Read access to "
                    f"public gated repos' at {TOKEN_SETTINGS_URL}."
                ),
                action_url=gate_url(repo_id),
                repo_id=repo_id,
            )
        if status in (401, 404):
            return ModelReadiness(
                ready=False,
                reason="bad_token",
                message=(
                    f"The Hugging Face token is invalid or cannot access {repo_id} "
                    f"(HTTP {status})."
                ),
                action_url=TOKEN_SETTINGS_URL,
                repo_id=repo_id,
            )
        raise  # unexpected HTTP status — surface it rather than mislabel
    except OSError:
        # A genuine connection failure (no network). requests' connection errors and
        # the builtin ConnectionError are OSError subclasses; HfHubHTTPError is caught
        # above, so this branch is only reached for real I/O failures (Tenet 17:
        # graceful, never hang). Avoids a hard dependency on `requests`.
        return ModelReadiness(
            ready=False,
            reason="offline",
            message=(
                f"{repo_id} is not downloaded and there is no network connection "
                "to fetch it. Connect and try again."
            ),
            action_url=None,
            repo_id=repo_id,
        )
    # Access is granted — but that is not the same as having the weights, and
    # load() will not fetch them. Only is_installed() earns "ok".
    return _not_installed(repo_id, install_cmd)


def _ok(repo_id: str) -> ModelReadiness:
    return ModelReadiness(
        ready=True, reason="ok", message="", action_url=None, repo_id=repo_id
    )


def _not_installed(repo_id: str, install_cmd: str | None = None) -> ModelReadiness:
    command = install_cmd or "detail model install <model_id>"
    return ModelReadiness(
        ready=False,
        reason="not_installed",
        message=(
            f"{repo_id} is not downloaded. Install it first — `{command}`, or "
            f"Install on the GUI's Models tab (each size is its own download). "
            f"A run will not fetch it."
        ),
        action_url=None,
        repo_id=repo_id,
    )


def _make_tqdm_class(progress_cb: Callable[[InstallProgress], None]):
    from huggingface_hub.utils import tqdm as hf_tqdm

    class _ForwardingTqdm(hf_tqdm):
        def update(self, n=1):
            displayed = super().update(n)
            # Forward byte-unit bars only; hf also runs a file-count bar.
            if getattr(self, "unit", None) == "B":
                progress_cb(
                    InstallProgress(
                        bytes_done=int(self.n),
                        bytes_total=int(self.total) if self.total else None,
                        current_file=self.desc or None,
                    )
                )
            return displayed

    return _ForwardingTqdm


def _require_hf():
    try:
        import huggingface_hub
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face models; install it "
            "(e.g. `pip install -e '.[torch]'`)."
        ) from e
    return huggingface_hub
