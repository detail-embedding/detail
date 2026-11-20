"""A name read from a ``detail`` module must exist in that module.

**The class.** A name looked up in another module inside a function body — imported there, or
read off a module object as ``module.NAME`` — is not resolved when the module is imported, so a
name that does not exist fails only when a user reaches that function. Import checks, the
construction tests and lint all pass.

**The instances.** The clip window's *Export GIF…* imported ``GIF`` from
``detail.gui.animation_export``, which never defined it; the action raised ``ImportError`` the
first time anyone chose it. The montage's *Export GIF…* (INTERPRET) then read
``animation_export.GIF`` off the module object — the same missing name, an ``AttributeError``
this time, which the import-only scan could not see. *Export video…* beside each worked only
because that module happened to import ``MP4`` for its own use. ``GIF`` and ``MP4`` are
``detail.api.animation``'s.

**The guard.** Every ``from detail.<module> import <name>`` inside a function under ``detail/``
is resolved: the module imports, and the name is an attribute of it or a submodule. And every
``alias.NAME`` read, where ``alias`` is a ``detail`` module bound by an import (``from detail.x
import module [as alias]`` or ``import detail.x.module as alias``), names an attribute or
submodule of it — whether the import is deferred or at module level.
"""

import ast
import importlib
import pathlib
import types

REPO = pathlib.Path(__file__).resolve().parent.parent


def _resolves(module: types.ModuleType, name: str) -> bool:
    if hasattr(module, name):
        return True
    try:
        importlib.import_module(f"{module.__name__}.{name}")
    except ImportError:
        return False
    return True


def _is_detail(module_name: str | None) -> bool:
    return bool(module_name) and module_name.split(".")[0] == "detail"


def _module_bindings(scope: ast.AST) -> dict[str, types.ModuleType]:
    """Names bound to a ``detail`` module by the imports directly in ``scope``'s body."""
    bound = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and _is_detail(alias.name):
                    bound[alias.asname] = importlib.import_module(alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if not _is_detail(node.module):
                continue
            for alias in node.names:
                value = getattr(importlib.import_module(node.module), alias.name, None)
                if isinstance(value, types.ModuleType):
                    bound[alias.asname or alias.name] = value
    return bound


def _rebound(scope: ast.AST, name: str) -> bool:
    """Whether ``scope`` also binds ``name`` to something other than an import."""
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Store):
                return True
        elif isinstance(node, ast.arg) and node.arg == name:
            return True
    return False


def _unresolved(source: str, where: str) -> list[str]:
    """``detail`` names in ``source`` that cannot resolve: deferred ``from detail... import
    name`` statements, and ``module.NAME`` reads off an imported ``detail`` module."""
    tree = ast.parse(source)
    missing = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            if not _is_detail(node.module):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if not _resolves(module, alias.name):
                    missing.append(
                        f"{where}:{node.lineno}: {node.module} has no {alias.name}"
                    )
    scopes = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for scope in scopes:
        for name, module in _module_bindings(scope).items():
            if _rebound(scope, name):
                continue
            for node in ast.walk(scope):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == name
                    and isinstance(node.ctx, ast.Load)
                    and not _resolves(module, node.attr)
                ):
                    missing.append(
                        f"{where}:{node.lineno}: {module.__name__} has no {node.attr}"
                    )
    return sorted(set(missing))


def test_every_detail_name_read_from_a_module_exists():
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


def test_the_scan_rejects_an_attribute_the_module_does_not_have():
    source = (
        "def export(self):\n"
        "    from detail.gui import animation_export as ax\n"
        "    return ax.ask_save_path(None, 'Export GIF', 'x.gif', ax.GIF)\n"
    )
    assert _unresolved(source, "x.py") == [
        "x.py:3: detail.gui.animation_export has no GIF"
    ]


def test_the_scan_rejects_an_attribute_off_a_top_level_module_import():
    source = (
        "import detail.gui.animation_export as ax\n"
        "def export():\n"
        "    return ax.GIF\n"
    )
    assert _unresolved(source, "x.py") == [
        "x.py:3: detail.gui.animation_export has no GIF"
    ]


def test_the_scan_accepts_a_submodule_and_ignores_top_level_imports():
    source = (
        "from detail.gui.animation_export import GIF\n"
        "def f():\n"
        "    from detail.api import animation\n"
        "    return animation.GIF\n"
    )
    assert _unresolved(source, "x.py") == []


def test_the_scan_skips_a_name_the_scope_rebinds():
    source = (
        "def f(ax):\n"
        "    from detail.gui import animation_export as ax\n"
        "    return ax.GIF\n"
    )
    assert _unresolved(source, "x.py") == []
