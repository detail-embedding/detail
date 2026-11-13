"""Tests for ``scripts/deploy_docs.py``'s refusal to upload past an open gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy_docs.py"
_LOGIN = "https://team.cloudflareaccess.com/cdn-cgi/access/login/detail-docs.pages.dev"


def _load():
    spec = importlib.util.spec_from_file_location("deploy_docs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def deploy():
    return _load()


@pytest.mark.parametrize(
    ("status", "location", "gated"),
    [
        (302, _LOGIN, True),
        (200, None, False),
        (522, None, False),
        (302, "https://detail-docs.pages.dev/index.html", False),
        (302, "https://cloudflareaccess.com.attacker.example/login", False),
        (302, None, False),
    ],
)
def test_is_gated_only_for_a_redirect_to_access(deploy, status, location, gated):
    assert deploy.is_gated(status, location) is gated


def test_gate_urls_include_a_fresh_preview_address(deploy):
    first, second = deploy.gate_urls(), deploy.gate_urls()
    assert first[0] == "https://detail-docs.pages.dev/"
    assert first[1].endswith(".detail-docs.pages.dev/")
    assert first[1] != second[1]


def test_upload_refused_when_only_production_is_gated(deploy, monkeypatch):
    def probe(url):
        if url.startswith("https://detail-docs.pages.dev/"):
            return 302, _LOGIN
        return 200, None

    monkeypatch.setattr(deploy, "_probe", probe)
    uploads = []
    monkeypatch.setattr(deploy, "build", lambda site: None)
    monkeypatch.setattr(deploy, "upload", uploads.append)
    assert deploy.main([]) == 1
    assert uploads == []


def test_upload_refused_when_an_address_cannot_be_reached(deploy, monkeypatch):
    def probe(url):
        raise deploy.urllib.error.URLError("certificate verify failed")

    monkeypatch.setattr(deploy, "_probe", probe)
    uploads = []
    monkeypatch.setattr(deploy, "build", lambda site: None)
    monkeypatch.setattr(deploy, "upload", uploads.append)
    assert deploy.main([]) == 1
    assert uploads == []


def test_upload_runs_when_every_address_is_gated(deploy, monkeypatch, tmp_path):
    monkeypatch.setattr(deploy, "_probe", lambda url: (302, _LOGIN))
    uploads = []
    monkeypatch.setattr(deploy, "build", lambda site: None)
    monkeypatch.setattr(deploy, "upload", uploads.append)
    assert deploy.main(["--site", str(tmp_path)]) == 0
    assert uploads == [tmp_path]
