"""Offline docs access (PR1): source resolution, cache, guarded build, URL seam."""

import os
import subprocess

import pytest

import detail
from detail import docs_access


def test_docs_source_dir_finds_mkdocs_in_source_checkout():
    # editable/source install: the repo root (package parent) has mkdocs.yml.
    source = docs_access.docs_source_dir()
    assert (source / "mkdocs.yml").is_file()


def test_docs_source_dir_env_override(tmp_path, monkeypatch):
    (tmp_path / "mkdocs.yml").write_text("site_name: x\n", encoding="utf-8")
    monkeypatch.setenv(docs_access.DOCS_SOURCE_ENV, str(tmp_path))
    assert docs_access.docs_source_dir() == tmp_path


def test_docs_source_dir_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(docs_access.DOCS_SOURCE_ENV, raising=False)
    monkeypatch.setattr(docs_access, "_PACKAGE_ROOT", tmp_path / "pkg")
    with pytest.raises(docs_access.DocsUnavailableError, match="mkdocs.yml"):
        docs_access.docs_source_dir()


def test_docs_cache_dir_is_versioned_and_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv(docs_access.DOCS_CACHE_ENV, str(tmp_path))
    assert docs_access.docs_cache_dir() == tmp_path / detail.__version__


def test_help_url_offline_points_into_built_site(tmp_path, monkeypatch):
    monkeypatch.delenv(docs_access.DOCS_URL_ENV, raising=False)
    monkeypatch.setattr(docs_access, "built_site_dir", lambda: tmp_path)
    (tmp_path / "concepts.html").write_text("<html></html>", encoding="utf-8")
    expected = (tmp_path / "concepts.html").as_uri()
    assert docs_access.help_url("concepts") == expected
    assert (
        docs_access.help_url("concepts", "field-of-view") == f"{expected}#field-of-view"
    )
    assert docs_access.help_url() == (tmp_path / "index.html").as_uri()


def test_help_url_hosted_when_env_set(monkeypatch):
    monkeypatch.setenv(docs_access.DOCS_URL_ENV, "https://docs.example.com/")
    assert docs_access.help_url("cli") == "https://docs.example.com/cli"
    assert docs_access.help_url("cli", "run") == "https://docs.example.com/cli#run"
    assert docs_access.help_url() == "https://docs.example.com"


def test_ensure_site_built_is_guarded_when_fresh(monkeypatch):
    # A no-op when the site exists and is up to date — must not shell out to mkdocs again.
    monkeypatch.setattr(docs_access, "needs_build", lambda: False)

    def _boom(*_args, **_kwargs):
        raise AssertionError("mkdocs must not run when the cached site is fresh")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    events = []
    docs_access.ensure_site_built(progress=lambda s, f, d: events.append((s, f, d)))
    assert events and events[-1][1] == 1.0


def test_needs_build_rebuilds_when_a_doc_source_is_newer(tmp_path, monkeypatch):
    # The version key alone would serve a stale site; a doc edit must trigger a rebuild.
    source = tmp_path / "src"
    (source / "docs").mkdir(parents=True)
    (source / "mkdocs.yml").write_text("site_name: x\n", encoding="utf-8")
    doc = source / "docs" / "concepts.md"
    doc.write_text("# hi\n", encoding="utf-8")
    monkeypatch.setattr(docs_access, "docs_source_dir", lambda: source)
    monkeypatch.setenv(docs_access.DOCS_CACHE_ENV, str(tmp_path / "cache"))
    marker = docs_access.built_site_dir() / "index.html"
    marker.parent.mkdir(parents=True)
    marker.write_text("<html></html>", encoding="utf-8")

    os.utime(marker, (doc.stat().st_mtime - 10, doc.stat().st_mtime - 10))
    assert docs_access.needs_build()  # site older than the doc → rebuild
    os.utime(marker, (doc.stat().st_mtime + 10, doc.stat().st_mtime + 10))
    assert not docs_access.needs_build()  # site newer than every source → fresh


def test_ensure_site_built_builds_a_browsable_offline_site(tmp_path, monkeypatch):
    # End-to-end: a real mkdocs build lands a flat-URL site whose pages resolve
    # via help_url, and progress moves before completion (long-op cadence).
    monkeypatch.setenv(docs_access.DOCS_CACHE_ENV, str(tmp_path))
    monkeypatch.delenv(docs_access.DOCS_URL_ENV, raising=False)
    events = []
    site = docs_access.ensure_site_built(
        progress=lambda s, f, d: events.append((s, f, d))
    )
    assert docs_access.is_site_built()
    assert (site / "index.html").is_file()
    # flat URLs (use_directory_urls: false) so file:// links resolve
    assert (site / "concepts.html").is_file()
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    # url2pathname, not unquote(urlparse(...).path): as_uri() yields file:///C:/... on
    # Windows, whose .path is "/C:/..." — not a filesystem path.
    target = url2pathname(urlparse(docs_access.help_url("concepts")).path)
    assert (site / "concepts.html").samefile(target)
    assert any(0.0 < f < 1.0 for _, f, _ in events)  # intermediate movement
    assert events[-1][1] == 1.0


def test_local_url_hosted_when_env_set(monkeypatch):
    monkeypatch.setenv(docs_access.DOCS_URL_ENV, "https://docs.example.com/")
    assert (
        docs_access.local_url("gui", "metadata")
        == "https://docs.example.com/gui#metadata"
    )


def test_local_url_serves_the_site_over_http_with_a_working_anchor(
    tmp_path, monkeypatch
):
    # A file:// #anchor is dropped by the OS opener; local_url serves over http so the browser
    # honors it. Assert the URL is http and the page is actually reachable + carries the id.
    import urllib.request

    monkeypatch.delenv(docs_access.DOCS_URL_ENV, raising=False)
    monkeypatch.setattr(docs_access, "_server_base", None)
    monkeypatch.setenv(docs_access.DOCS_CACHE_ENV, str(tmp_path))
    site = docs_access.built_site_dir()
    site.mkdir(parents=True)
    (site / "gui.html").write_text(
        "<html><h2 id='metadata'>Metadata</h2></html>", encoding="utf-8"
    )

    url = docs_access.local_url("gui", "metadata")
    assert url.startswith("http://127.0.0.1:")
    assert url.endswith("/gui.html#metadata")
    body = urllib.request.urlopen(url.split("#")[0], timeout=5).read().decode()
    assert "id='metadata'" in body
