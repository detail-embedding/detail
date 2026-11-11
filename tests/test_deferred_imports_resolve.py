"""A name imported inside a function must exist in the module it is imported from.

**The class.** An import inside a function body is not run when the module is imported, so a
name that does not exist there fails only when a user reaches that function. Import checks, the
construction tests and lint all pass.

**The instance.** The clip window's *Export GIF…* imported ``GIF`` from
``detail.gui.animation_export``, which never defined it; the action raised ``ImportError`` the
first time anyone chose it. *Export video…* beside it worked only because that module happened
to import ``MP4`` for its own use.

**The guard.** Every ``from detail.<module> import <name>`` inside a function under ``detail/``
is resolved: the module imports, and the name is an attribute of it or a submodule.
"""

import ast
import importlib
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def _unresolved(source: str, where: str) -> list[str]:
    """Deferred ``from detail... import name`` statements in ``source`` that cannot resolve."""
    missing = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if node.module.split(".")[0] != "detail":
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if hasattr(module, alias.name):
                    continue
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    missing.append(
                        f"{where}:{node.lineno}: {node.module} has no {alias.name}"
                    )
    return sorted(set(missing))


def test_every_deferred_import_names_something_that_exists():
    missing = []
    for path in sorted((REPO / "detail").rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        missing += _unresolved(source, path.relative_to(REPO).as_posix())
    assert not missing, "\n".join(missing)


def test_the_scan_rejects_a_name_the_module_does_not_have():
    source = (
        "def export_gif():\n"
        "    from detail.gui.animation_export import GIF, ask_save_path\n"
    )
    assert _unresolved(source, "x.py") == [
        "x.py:2: detail.gui.animation_export has no GIF"
    ]


def test_the_scan_accepts_a_submodule_and_ignores_top_level_imports():
    source = (
        "from detail.gui.animation_export import GIF\n"
        "def f():\n"
        "    from detail.api import animation\n"
    )
    assert _unresolved(source, "x.py") == []
