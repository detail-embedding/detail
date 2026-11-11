"""A wide-scoped fixture is shared through a conftest, never imported from a test module.

**The class: a fixture imported into another module is a separate fixture there.** pytest
registers a fixture per module it is *found* in, and an import puts it in the importing module's
namespace, so pytest finds it again and treats it as that module's own. For a function-scoped
fixture nothing changes: it runs per test either way. For a **session- or module-scoped** one it
means one instance per importing module, so the expensive thing the scope exists to share is
built again in every file that imported it.

**What it cost.** `tests/gui/test_cluster_workspace.py` defined a session-scoped `_fit_prototype`
holding a ~30s pose-clustering fit, and `test_typicality_split.py` and `test_workspace_parity.py`
imported it. That was three fits a run: the three slowest setups in the gui tier, each ~30s,
where one was intended. Nothing failed. Every test passed, and the scope said "session" while
the behaviour was "per file". The shared fixture now lives in `tests/conftest.py`, next to
the pose-clustering fit template, which was built in #534 for the same reason (both went with
pose clustering; the temporal templates there follow the rule).

**The rule.** A session- or module-scoped fixture that more than one test module needs goes in a
`conftest.py`. Importing function-scoped helpers or fixtures between test modules is fine.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

_WIDE = ("session", "module", "package")


def _wide_fixtures(tree: ast.AST) -> set[str]:
    """Top-level functions in ``tree`` decorated as a session/module/package-scoped fixture."""
    wide = set()
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            text = ast.unparse(decorator)
            if "fixture" in text and any(
                f'"{s}"' in text or f"'{s}'" in text for s in _WIDE
            ):
                wide.add(node.name)
    return wide


def _module_name(path: pathlib.Path) -> str:
    return path.relative_to(REPO).with_suffix("").as_posix().replace("/", ".")


def imported_wide_fixtures(modules: dict[str, str]) -> list[str]:
    """``"importer <- module.fixture"`` for each wide fixture imported across test modules.

    ``modules`` maps a dotted module name to its source, so the negative case can pass literals.
    """
    trees = {name: ast.parse(source) for name, source in modules.items()}
    wide = {name: _wide_fixtures(tree) for name, tree in trees.items()}
    found = []
    for importer, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in wide:
                for alias in node.names:
                    if alias.name in wide[node.module]:
                        found.append(f"{importer} <- {node.module}.{alias.name}")
    return found


def test_no_wide_fixture_is_imported_from_another_test_module():
    modules = {
        _module_name(path): path.read_text(encoding="utf-8-sig")
        for path in sorted(TESTS.rglob("test_*.py"))
    }
    assert len(modules) > 100, "the scan read too few test modules to mean anything"
    found = imported_wide_fixtures(modules)
    assert not found, (
        "a session/module-scoped fixture is imported into another test module, where pytest "
        "builds it again — once per importing file. Move it to a conftest.py: "
        + ", ".join(found)
    )


def test_the_scan_would_catch_the_imported_fit():
    """Guard the guard, on the shape that cost three fits a run."""
    defining = (
        "import pytest\n"
        '@pytest.fixture(scope="session")\n'
        "def _fit_prototype():\n"
        "    return {}\n"
        "@pytest.fixture\n"
        "def per_test():\n"
        "    return 1\n"
    )
    modules = {
        "tests.gui.test_a": defining,
        "tests.gui.test_b": "from tests.gui.test_a import _fit_prototype, per_test\n",
    }
    assert imported_wide_fixtures(modules) == [
        "tests.gui.test_b <- tests.gui.test_a._fit_prototype"
    ]
