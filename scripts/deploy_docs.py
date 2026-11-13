"""Build the docs site and upload it to the lab's gated Cloudflare Pages host.

    python scripts/deploy_docs.py            # build, check the gate, upload
    python scripts/deploy_docs.py --check    # only check the gate

The site is readable by the lab only: Cloudflare Access sits in front of every address the
project serves, and an upload is refused unless a signed-out request to both the production
address and an invented preview address is redirected to the Access login. Both, because every
upload also gets its own ``<hash>.detail-docs.pages.dev`` preview address. On 2026-11-10 the
production address was gated while the preview wildcard was missing, which a check of production
alone would have passed. A new Pages project serves the open internet until its Access
application covers it; this check is the only thing standing between a misconfigured gate and
publishing the guides.

The copy is for sharing the guides with people outside DETAIL. In-app help never uses it: the Docs
tab, ``?`` links and ``detail docs`` build the site locally (``detail/docs_access.py``), and
``DETAIL_DOCS_URL`` stays unset (``docs/docs_design.md`` §8).

Run it in an interactive terminal: Wrangler refuses to authenticate when stdin is not a TTY,
even with a valid login on disk. Needs Node (``npx``) and a ``wrangler login`` to the account
that owns the project. On Windows run it from WSL, where Node is installed.

Temporary while the repo is private; retired once the docs have a public host.
"""

from __future__ import annotations

import argparse
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = "detail-docs"
PRODUCTION_HOST = f"{PROJECT}.pages.dev"
ACCESS_LOGIN_SUFFIX = ".cloudflareaccess.com"
WRANGLER = "wrangler@4"

_REPO = Path(__file__).resolve().parents[1]


class GateOpenError(RuntimeError):
    """A signed-out request reached the site instead of the Access login."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs) -> None:
        return None


def is_gated(status: int, location: str | None) -> bool:
    """Whether a signed-out response is Access's redirect to its login page."""
    if status not in (301, 302, 303, 307, 308) or not location:
        return False
    host = urlsplit(location).hostname or ""
    return host.endswith(ACCESS_LOGIN_SUFFIX)


def _probe(url: str) -> tuple[int, str | None]:
    opener = urllib.request.build_opener(_NoRedirect)
    # Cloudflare answers urllib's default ``Python-urllib`` agent with a bare 403 before
    # Access ever sees it, which would read as "not gated" on a site that is.
    request = urllib.request.Request(url, headers={"User-Agent": "detail-gate-check"})
    try:
        with opener.open(request, timeout=20) as response:
            return response.status, response.headers.get("Location")
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Location")


def gate_urls() -> list[str]:
    """Production, plus a preview address no upload has used, so the wildcard is checked."""
    preview = f"gatecheck{secrets.token_hex(4)}.{PRODUCTION_HOST}"
    return [f"https://{PRODUCTION_HOST}/", f"https://{preview}/"]


def check_gate() -> None:
    """Raise unless every address answers a signed-out request with the Access login."""
    for url in gate_urls():
        try:
            status, location = _probe(url)
        except urllib.error.URLError as error:
            # Unverifiable is not gated. Windows' TLS stack has failed the handshake on
            # preview addresses that WSL's verifies, so run this from WSL there.
            raise GateOpenError(f"could not reach {url}: {error.reason}") from error
        if not is_gated(status, location):
            raise GateOpenError(
                f"{url} answered {status} without redirecting to Cloudflare Access. "
                f"Add both {PRODUCTION_HOST} and *.{PRODUCTION_HOST} to the Access "
                "application before uploading."
            )
        print(f"gated: {url}")


def build(site: Path) -> None:
    """``mkdocs build --strict`` into ``site`` (hosted build: the offline plugin stays off)."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--clean",
            "-d",
            str(site),
        ],
        cwd=_REPO,
        check=True,
    )


def upload(site: Path) -> None:
    """Upload ``site`` as the project's production deployment."""
    subprocess.run(
        [
            "npx",
            "-y",
            WRANGLER,
            "pages",
            "deploy",
            str(site),
            "--project-name",
            PROJECT,
            "--branch",
            "main",
            "--commit-dirty=true",
        ],
        cwd=_REPO,
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="only check the gate; build nothing"
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=_REPO / "build" / "docs-site",
        help="build output directory (default: build/docs-site)",
    )
    args = parser.parse_args(argv)
    try:
        check_gate()
    except GateOpenError as error:
        print(f"refusing to upload: {error}", file=sys.stderr)
        return 1
    if args.check:
        return 0
    build(args.site)
    upload(args.site)
    try:
        check_gate()
    except GateOpenError as error:
        print(f"UPLOADED BUT NOT GATED — fix Access now: {error}", file=sys.stderr)
        return 1
    print(f"live: https://{PRODUCTION_HOST}/user-guides/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
