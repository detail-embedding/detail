"""No test reaches the developer's real machine config — at any fixture scope.

The class: **a per-test isolation fixture, which every fixture wider than a test runs
outside of.** ``~/.config/detail/storage.yaml`` holds the machine's video-root registry, and
video import adds the imported file's folder to it. ``tests/conftest.py`` redirected that file
with a function-scoped autouse fixture, and pytest sets those up per *test* — after the
module- and session-scoped fixtures the test requests. Three GUI files build their experiment
once per module, by importing a video, so each run wrote its temp dirs into the real file:
~30 ``conformance0_N`` / ``settings0_N`` / ``fits0_N`` roots under ``pytest-of-<user>``
accumulated on one machine, refilling after it had been cleaned by hand.

It hid because nothing fails: the import succeeds, the root resolves, and the test reads
its own experiment. The only symptom is in a file outside the repo that no test reads, and
the litter points at temp dirs pytest deletes three runs later.

The rule: **machine-global state is isolated at session scope too** — the floor
(``_machine_config_floor``) is what a wide fixture, a spawned worker and a thread outliving
its test all see. The tests here ask the question from each scope a fixture can have, and
check the paths rather than the file, so they can fail without touching real state.
``conftest.py::pytest_sessionfinish`` is the backstop for a route these do not cover: it fails
the run if a real config file changed to name this run's basetemp.

**Which modules are checked is found, not listed.** Every module in ``detail/`` that reads
``DETAIL_CONFIG_DIR`` captures the config dir at import, so each is a copy the conftest has to
redirect. ``remote_targets`` was a third one, which nothing redirected centrally — every test
that touched it patched it by hand, the shape the storage fixture had replaced. A hand-written
list here would have missed it the same way.
"""

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_READS_THE_OVERRIDE = re.compile(r"""os\.environ\.get\(\s*["']DETAIL_CONFIG_DIR["']""")


def _config_modules() -> list[str]:
    """Every ``detail`` module that resolves the config dir itself from the env override."""
    return sorted(
        ".".join(path.relative_to(_REPO).with_suffix("").parts)
        for path in (_REPO / "detail").rglob("*.py")
        if _READS_THE_OVERRIDE.search(path.read_text(encoding="utf-8-sig"))
    )


def _config_paths() -> dict[str, str]:
    """The env override plus every config-dir constant each such module captured at import."""
    paths = {"env DETAIL_CONFIG_DIR": os.environ.get("DETAIL_CONFIG_DIR", "")}
    for name in _config_modules():
        module = importlib.import_module(name)
        for attr, value in vars(module).items():
            if isinstance(value, Path) and (
                attr.endswith("CONFIG_DIR") or attr.endswith("_PATH")
            ):
                paths[f"{name}.{attr}"] = str(value)
    return paths


def test_the_module_discovery_finds_the_known_config_readers():
    found = set(_config_modules())
    assert {"detail.storage.locations", "detail.storage.credentials"} <= found
    assert set(_config_paths()) >= {
        "detail.storage.locations.STORAGE_PATH",
        "detail.storage.credentials.CREDENTIALS_PATH",
    }


def _outside(paths: dict[str, str], basetemp: Path) -> dict[str, str]:
    """The paths not under ``basetemp`` — each one a route to the real machine config."""
    root = basetemp.resolve()
    return {
        name: path
        for name, path in paths.items()
        if not path or not Path(path).resolve().is_relative_to(root)
    }


@pytest.fixture(scope="session")
def seen_by_a_session_fixture():
    return _config_paths()


@pytest.fixture(scope="module")
def seen_by_a_module_fixture():
    return _config_paths()


# Requested as arguments, never through `request.getfixturevalue`: that builds the fixture
# *inside* the test, after the per-test isolation is in place, and so passes with the floor
# removed — measured, the first time this file was run.
def test_a_session_fixture_meets_a_throwaway_machine_config(
    seen_by_a_session_fixture, tmp_path_factory
):
    seen = seen_by_a_session_fixture
    assert _outside(seen, tmp_path_factory.getbasetemp()) == {}


def test_a_module_fixture_meets_a_throwaway_machine_config(
    seen_by_a_module_fixture, tmp_path_factory
):
    seen = seen_by_a_module_fixture
    assert _outside(seen, tmp_path_factory.getbasetemp()) == {}


def test_a_test_meets_its_own_machine_config(tmp_path_factory):
    assert _outside(_config_paths(), tmp_path_factory.getbasetemp()) == {}


def test_a_test_gets_a_config_dir_distinct_from_the_wide_fixtures(
    seen_by_a_module_fixture,
):
    assert _config_paths()["detail.storage.locations.CONFIG_DIR"] != (
        seen_by_a_module_fixture["detail.storage.locations.CONFIG_DIR"]
    )


def test_the_scope_check_would_catch_the_real_config(tmp_path):
    basetemp = tmp_path / "basetemp"
    real = {"locations.STORAGE_PATH": str(tmp_path / "home" / "storage.yaml")}
    unset = {"env DETAIL_CONFIG_DIR": ""}
    inside = {"locations.CONFIG_DIR": str(basetemp / "machine_config_floor0")}
    assert _outside(real, basetemp) == real
    assert _outside(unset, basetemp) == unset
    assert _outside(inside, basetemp) == {}


def _machine_config():
    """``tests/machine_config.py``, loaded the way ``conftest.py`` loads it."""
    name = "detail_test_machine_config"
    if name not in sys.modules:
        path = Path(__file__).parent / "machine_config.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_BASETEMP = Path("C:/Users/u/AppData/Local/Temp/pytest-of-u/pytest-2249")
_BEFORE = {"storage.yaml": b"video_roots:\n  first30s: C:/data/first30s\n"}


@pytest.mark.parametrize(
    "line",
    [
        b"  conformance0: C:/Users/u/AppData/Local/Temp/pytest-of-u/pytest-2249/"
        b"popen-gw3/conformance0\n",
        rb"  conformance0: C:\Users\u\AppData\Local\Temp\pytest-of-u\pytest-2249"
        rb"\conformance0",
    ],
    ids=["posix", "native"],
)
def test_the_session_backstop_would_catch_a_root_under_this_run(line):
    after = {"storage.yaml": _BEFORE["storage.yaml"] + line}
    assert _machine_config().litter(_BEFORE, after, _BASETEMP) == ["storage.yaml"]


def test_the_session_backstop_lets_the_developer_edit_their_config_mid_run():
    old = {
        "storage.yaml": b"video_roots:\n  fits0: C:/Users/u/AppData/Local/Temp/"
        b"pytest-of-u/pytest-2249/popen-gw6/fits0\n"
    }
    # A root the developer added in their own app, beside a previous run's litter whose
    # basetemp merely starts with this run's name.
    edited = {"storage.yaml": old["storage.yaml"] + b"  box8: D:/videos/box8\n"}
    this_run = Path("C:/Users/u/AppData/Local/Temp/pytest-of-u/pytest-22")
    assert _machine_config().litter(old, edited, this_run) == []
    assert _machine_config().litter(old, old, _BASETEMP) == []
