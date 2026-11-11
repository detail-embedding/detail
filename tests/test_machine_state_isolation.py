"""No test reaches this machine's real state, even from a thread that outlives it.

**The class: per-test isolation that a background thread outlives.** The suite isolates the
model cache, the video preview cache, the built docs site and ``~/.config/detail`` with
per-test ``monkeypatch`` fixtures (``tests/conftest.py``). A GUI page starts work on a thread;
when that thread runs after its test, the monkeypatch has already reverted, and the path it
resolves is the developer's real one. Nothing fails: the write succeeds, somewhere it should
not. Found twice: the model cache on a network share faulted the process, and a GUI test's
temporary video path turned up in a developer's real ``~/.cache/detail/previews/index.json``
(a page's preview lookup writes the cache index).

**The rule.** Every piece of machine-local state has a session-wide floor below its per-test
isolation (``conftest._machine_state_floor``), so what a stray thread sees after revert is
still inside the pytest basetemp. QSettings is isolated separately by the ``qapp`` fixture
(``tests/gui/conftest.py``), which points the store at the basetemp for the whole session.

**The guard** reverts this test's own isolation, which is exactly what a stray thread sees,
and asks every resolver where it points now.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from detail import docs_access
from detail.models import hf_access
from detail.storage import credentials
from detail.storage import locations


def _machine_state_paths() -> dict[str, pathlib.Path]:
    """Where each piece of machine-local state resolves right now."""
    return {
        "preview cache": locations.preview_cache_dir(),
        "model cache": hf_access.model_cache_dir(),
        "docs cache": docs_access.docs_cache_dir(),
        "config dir": locations.CONFIG_DIR,
        "storage.yaml": locations.STORAGE_PATH,
        "credentials": credentials.CREDENTIALS_PATH,
    }


def _escapes(
    paths: dict[str, pathlib.Path], basetemp: pathlib.Path
) -> dict[str, pathlib.Path]:
    """The entries that resolve outside the pytest basetemp."""
    root = basetemp.resolve()
    return {
        name: path
        for name, path in paths.items()
        if not pathlib.Path(path).expanduser().resolve().is_relative_to(root)
    }


def test_machine_state_stays_in_the_basetemp_after_a_test_reverts(
    monkeypatch, tmp_path_factory
):
    inside = _escapes(_machine_state_paths(), tmp_path_factory.getbasetemp())
    assert (
        not inside
    ), f"isolated per test, yet resolving outside the basetemp: {inside}"
    monkeypatch.undo()  # what a thread that outlives its test sees
    escaped = _escapes(_machine_state_paths(), tmp_path_factory.getbasetemp())
    assert not escaped, (
        "After this test's own isolation reverts, these resolve to the machine's real state, "
        f"so a background thread that outlives a test writes there: {escaped}. Give each a "
        "session-wide floor in tests/conftest.py::_machine_state_floor."
    )


def test_the_isolation_check_would_catch_the_real_preview_cache(tmp_path_factory):
    """Negative case: the real default location is outside the basetemp."""
    real = {"preview cache": pathlib.Path.home() / ".cache" / "detail" / "previews"}
    assert _escapes(real, tmp_path_factory.getbasetemp()) == real


def test_the_check_would_catch_a_missing_floor(monkeypatch, tmp_path_factory):
    """Negative case through the resolver: with no env var and no storage.yaml entry, the
    preview cache resolves to its real default, and the check names it."""
    monkeypatch.delenv(locations.PREVIEW_DIR_ENV, raising=False)
    monkeypatch.setattr(locations, "read_storage", lambda: {})
    escaped = _escapes(_machine_state_paths(), tmp_path_factory.getbasetemp())
    assert "preview cache" in escaped


@pytest.mark.parametrize("name", ["DETAIL_MODEL_DIR", "DETAIL_PREVIEW_DIR"])
def test_every_per_test_isolation_has_a_floor(name, monkeypatch):
    """Each env var a per-test fixture sets is still set once that fixture reverts."""
    monkeypatch.undo()
    assert os.environ.get(name), f"{name} has no session floor in tests/conftest.py"
