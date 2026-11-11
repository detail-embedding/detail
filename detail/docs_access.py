"""Offline documentation access.

Resolve the docs source tree, build the MkDocs Material site into a per-user
cache once per version, and hand out URLs for the in-app / CLI "open docs"
surfaces. The build enables Material's ``offline`` plugin (via the
``DETAIL_DOCS_OFFLINE`` env toggle in ``mkdocs.yml``) so the site works over
``file://`` with no server — help works with no network.

Qt-free by construction (the hard rule): this drives progress through the shared
``(stage, fraction, detail)`` callback the CLI bar and the GUI worker both
consume, and builds via a subprocess — no backend or Qt import.
"""

from __future__ import annotations

import functools
import http.server
import os
import socketserver
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import detail

DOCS_SOURCE_ENV = (
    "DETAIL_DOCS_DIR"  # override the docs source tree (contains mkdocs.yml)
)
DOCS_CACHE_ENV = "DETAIL_DOCS_CACHE"  # override the built-site cache root
DOCS_URL_ENV = "DETAIL_DOCS_URL"  # a hosted docs base URL (once the site is published)

_PACKAGE_ROOT = Path(detail.__file__).resolve().parent
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "detail" / "docs"

ProgressFn = Callable[[str, float, str], None]


class DocsUnavailableError(RuntimeError):
    """The docs source tree (mkdocs.yml) can't be found, so the site can't be built."""


class DocsBuildError(RuntimeError):
    """``mkdocs build`` failed or produced no site."""


def docs_source_dir() -> Path:
    """The directory containing ``mkdocs.yml`` — the docs source tree.

    Resolution order: ``$DETAIL_DOCS_DIR`` → the repo root of an editable/source
    install → a copy bundled under the package (wheels). Raises if none has a
    ``mkdocs.yml``.
    """
    candidates: list[Path] = []
    env = os.environ.get(DOCS_SOURCE_ENV)
    if env:
        candidates.append(Path(env))
    candidates.append(_PACKAGE_ROOT.parent)  # editable / source checkout
    candidates.append(_PACKAGE_ROOT / "_docs")  # bundled fallback (wheels)
    for candidate in candidates:
        if (candidate / "mkdocs.yml").is_file():
            return candidate
    raise DocsUnavailableError(
        "Could not locate the documentation source (mkdocs.yml). Install from a "
        "source checkout (pip install -e .) or set $DETAIL_DOCS_DIR to the repo root."
    )


def docs_cache_dir() -> Path:
    """Per-user, per-version cache root for the built site (mirrors ``model_cache_dir``)."""
    root = Path(os.environ.get(DOCS_CACHE_ENV) or _DEFAULT_CACHE_ROOT)
    return root / detail.__version__


def built_site_dir() -> Path:
    """Where the built HTML site lives for this version."""
    return docs_cache_dir() / "site"


def is_site_built() -> bool:
    """Pure disk check — the built site's entry page exists for this version."""
    return (built_site_dir() / "index.html").is_file()


def _newest_source_mtime(source: Path) -> float:
    """Newest mtime across the files a build depends on — doc content + the Python that
    generates pages. Bounded roots (never the whole repo / .venv)."""
    files: list[Path] = []
    docs = source / "docs"
    if docs.is_dir():
        files += [p for p in docs.rglob("*") if p.is_file()]
    config = source / "mkdocs.yml"
    if config.is_file():
        files.append(config)
    scripts = source / "scripts"
    if scripts.is_dir():
        files += list(scripts.glob("gen_docs_*.py"))
        # The mkdocs hook decides which prose a hidden capability publishes, so editing it
        # changes the built site as surely as editing a page does. It does not match the
        # generator glob above, and without this the in-app docs serve a stale site.
        files += list(scripts.glob("docs_*_hook.py"))
    package = (
        source / "detail"
    )  # config models etc. the generators read (editable installs)
    if package.is_dir():
        files += [p for p in package.rglob("*.py")]
    return max((p.stat().st_mtime for p in files), default=0.0)


def needs_build() -> bool:
    """Whether the cached site must be (re)built: missing, or older than a doc source.

    The mtime check makes an **editable/dev** install rebuild whenever the docs (or the code
    that generates them) change — the version key alone would serve a stale site. A released
    install whose source doesn't change stays on the version cache.
    """
    if not is_site_built():
        return True
    try:
        source = docs_source_dir()
    except DocsUnavailableError:
        return False  # no source to rebuild from — keep whatever is cached
    marker = built_site_dir() / "index.html"
    return _newest_source_mtime(source) > marker.stat().st_mtime


def ensure_site_built(progress: ProgressFn | None = None, force: bool = False) -> Path:
    """Build the docs site into the cache and return its directory.

    A no-op when the cached site is present **and** newer than every doc source, unless
    ``force``. Drives the ``(stage, fraction, detail)`` callback throughout — the build is
    multi-second, so a caller (the CLI bar, the GUI worker) can show live progress.
    """
    site = built_site_dir()
    if not force and not needs_build():
        if progress is not None:
            progress("Documentation", 1.0, "ready")
        return site
    source = docs_source_dir()  # raises DocsUnavailableError if missing
    site.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress("Building documentation", 0.03, "starting")
    env = dict(os.environ, DETAIL_DOCS_OFFLINE="true")
    fraction = 0.05
    tail: list[str] = []
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--clean",
            "-f",
            str(source / "mkdocs.yml"),
            "-d",
            str(site),
        ],
        cwd=str(source),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:  # mkdocs prints per-page/plugin; creep toward 0.95
            stripped = line.rstrip()
            tail.append(stripped)
            del tail[:-20]  # keep only the last lines for an error message
            fraction = min(0.95, fraction + 0.02)
            if progress is not None:
                progress("Building documentation", fraction, stripped[:80])
        code = proc.wait()
    if code != 0 or not is_site_built():
        raise DocsBuildError(
            "mkdocs build failed (exit "
            f"{code}). Last output:\n" + "\n".join(tail[-10:])
        )
    if progress is not None:
        progress("Building documentation", 1.0, "done")
    return site


def help_url(page: str = "index", anchor: str | None = None) -> str:
    """A URL for a docs ``page`` (a nav path without extension, e.g. ``reference/parameters``).

    Hosted (``$DETAIL_DOCS_URL`` set) → the published site. Offline (default) → a
    ``file://`` URL into the built site (build first via :func:`ensure_site_built`).
    ``use_directory_urls`` is off, so pages are flat ``<page>.html``. Note a ``file://``
    ``#anchor`` is dropped by the OS file-opener — use :func:`local_url` to deep-link.
    """
    hosted = os.environ.get(DOCS_URL_ENV)
    if hosted:
        base = hosted.rstrip("/")
        url = base if page == "index" else f"{base}/{page}"
        return f"{url}#{anchor}" if anchor else url
    url = (built_site_dir() / f"{page}.html").as_uri()
    return f"{url}#{anchor}" if anchor else url


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # don't spam stderr per request
        pass


_server_base: str | None = None
_server_lock = threading.Lock()


def _local_server_base() -> str:
    """Base URL of a background HTTP server for the built site (started once, daemon thread).

    Serving over ``http://127.0.0.1`` rather than ``file://`` so the browser honors an
    ``#anchor`` — the OS file-opener drops the fragment when opening a local ``.html``.
    """
    global _server_base
    with _server_lock:
        if _server_base is None:
            handler = functools.partial(_QuietHandler, directory=str(built_site_dir()))
            httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
            httpd.daemon_threads = True
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            _server_base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return _server_base


def local_url(page: str = "index", anchor: str | None = None) -> str:
    """A browser URL that honors ``#anchor``: the hosted site when ``$DETAIL_DOCS_URL`` is set,
    else a ``http://127.0.0.1`` URL into the built site. Build the site first."""
    hosted = os.environ.get(DOCS_URL_ENV)
    if hosted:
        base = hosted.rstrip("/")
        url = base if page == "index" else f"{base}/{page}"
        return f"{url}#{anchor}" if anchor else url
    try:
        url = f"{_local_server_base()}/{page}.html"
    except OSError:  # can't bind loopback — fall back (the OS may drop the #anchor)
        return help_url(page, anchor)
    return f"{url}#{anchor}" if anchor else url
