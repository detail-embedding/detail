"""No test reaches the developer's real experiments workspace — at any fixture scope.

The class: **a fallback to a real per-user location, reached because the config that would
override it is isolated and empty.** ``locations.experiments_root()`` returns the
``storage.default.root`` of ``storage.yaml``, else ``DEFAULT_ROOT`` —
``~/Documents/DetailExperiments`` — and creates it. ``tests/conftest.py`` isolates
``storage.yaml`` for every test, so every test's file names no root and the fallback is the
real folder. About 120 test sites patched ``experiments_root`` by hand; the ones that did not
were never told. Run under a probe wrapping ``experiments_root`` (full suite, 2026-10-19,
Windows box), **56 tests in 18 files made 82 calls that resolved the developer's real
workspace** — the Home page's ``Experiment.list_all``, the distill dialog's experiment picker,
the readiness probe's ``has_experiment``, the keypoint library's skeleton dependents, the tool
run list, the active-processes poll — 10 of them in *teardown*, from a GUI thread outliving
its test. One wrote: the scene-geometry export's default path creates ``roi_exports/``
under the root, and the real folder holds an empty ``roi_exports/`` created at 00:06 one night.

It hid because nothing fails: reading a real workspace returns real experiments, every page
builds, and a test that asserts on what it listed was only ever right on a machine whose
workspace happened to suit it. On a machine with no workspace yet, the suite creates one.

The rule: **a per-user location is isolated where its fallback is resolved, not where its
config is** — and, like the machine config, at session scope as well as per test
(``_experiments_root_floor`` beside ``_isolate_experiments_root``), because a function-scoped
autouse fixture is set up after every wider fixture a test requests and reverts before a
stray thread is done. ``DEFAULT_ROOT`` is patched rather than ``experiments_root``, so the
function itself still runs everywhere.

Where it stops: a **spawned worker** re-imports ``locations`` and gets the real
``DEFAULT_ROOT`` — there is no environment override for it, unlike ``DETAIL_CONFIG_DIR``.
None reaches it today: every caller (``Experiment.create`` / ``list_all`` /
``import_archive``, ``readiness``, ``skeletons``, ``tool_runs``, ``spatial``) runs in-process,
and the probe saw no call outside the test process's call stacks.
"""

from pathlib import Path

import pytest

from detail.storage import locations


def _roots() -> dict[str, str]:
    return {
        "locations.DEFAULT_ROOT": str(locations.DEFAULT_ROOT),
        "locations.experiments_root()": str(locations.experiments_root()),
    }


def _outside(paths: dict[str, str], basetemp: Path) -> dict[str, str]:
    """The paths not under ``basetemp`` — each one the developer's real workspace."""
    root = basetemp.resolve()
    return {
        name: path
        for name, path in paths.items()
        if not path or not Path(path).resolve().is_relative_to(root)
    }


@pytest.fixture(scope="session")
def seen_by_a_session_fixture():
    return _roots()


@pytest.fixture(scope="module")
def seen_by_a_module_fixture():
    return _roots()


# Requested as arguments, never through `request.getfixturevalue`: that builds the fixture
# *inside* the test, after the per-test isolation is in place, and passes with the floor
# removed (`tests/test_machine_config_isolation.py` measured exactly that).
def test_a_session_fixture_meets_a_throwaway_workspace(
    seen_by_a_session_fixture, tmp_path_factory
):
    seen = seen_by_a_session_fixture
    assert _outside(seen, tmp_path_factory.getbasetemp()) == {}


def test_a_module_fixture_meets_a_throwaway_workspace(
    seen_by_a_module_fixture, tmp_path_factory
):
    seen = seen_by_a_module_fixture
    assert _outside(seen, tmp_path_factory.getbasetemp()) == {}


def test_a_test_meets_its_own_workspace(tmp_path_factory):
    assert _outside(_roots(), tmp_path_factory.getbasetemp()) == {}


def test_a_test_gets_a_workspace_distinct_from_the_wide_fixtures(
    seen_by_a_module_fixture,
):
    assert _roots()["locations.DEFAULT_ROOT"] != (
        seen_by_a_module_fixture["locations.DEFAULT_ROOT"]
    )


def test_an_experiment_created_in_one_test_is_not_listed_in_the_next(tmp_path_factory):
    """The per-test half's reason to exist: the managed workspace is shared state."""
    from detail.api.experiment import Experiment

    # Refuse before writing: with the isolation broken, this create lands in the real one.
    assert _outside(_roots(), tmp_path_factory.getbasetemp()) == {}
    assert Experiment.list_all() == []
    Experiment.create("left_behind")
    assert [e.dir.name for e in Experiment.list_all()] == ["left_behind"]


def test_and_the_next_test_does_not_see_it():
    """Runs after the one above (same file, same worker under ``--dist loadfile``)."""
    from detail.api.experiment import Experiment

    assert Experiment.list_all() == []


def test_the_scope_check_would_catch_the_real_workspace(tmp_path):
    basetemp = tmp_path / "basetemp"
    real = {"locations.DEFAULT_ROOT": str(tmp_path / "Documents" / "DetailExperiments")}
    unset = {"locations.experiments_root()": ""}
    inside = {"locations.DEFAULT_ROOT": str(basetemp / "experiments_root_floor0")}
    assert _outside(real, basetemp) == real
    assert _outside(unset, basetemp) == unset
    assert _outside(inside, basetemp) == {}
