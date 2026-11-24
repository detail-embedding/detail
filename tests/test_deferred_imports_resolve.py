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

**Scopes are lexical.** An alias is checked in the function, lambda or class whose body imports
it and in the scopes nested inside, and dropped wherever a scope rebinds the name (a parameter,
an assignment, a ``def``). A scope's own bindings never leak into its siblings: one function's
``ax`` parameter does not hide a module-level ``ax`` alias from the others, and two functions
that bind ``m`` to different modules are each checked against their own. A ``from detail.x
import submodule`` binding imports the submodule rather than relying on something else having
loaded it first, so the verdict does not depend on which files the sweep read before.
"""

import ast
import importlib
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parent.parent

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _resolves(module: types.ModuleType, name: str) -> bool:
    if hasattr(module, name):
        return True
    try:
        importlib.import_module(f"{module.__name__}.{name}")
    except ImportError:
        return False
    return True


def _as_module(parent: types.ModuleType, name: str) -> types.ModuleType | None:
    """``parent.name`` if it is a module, importing it when it is a submodule not yet loaded."""
    if hasattr(parent, name):
        value = getattr(parent, name)
        return value if isinstance(value, types.ModuleType) else None
    try:
        return importlib.import_module(f"{parent.__name__}.{name}")
    except ImportError:
        return None


def _is_detail(module_name: str | None) -> bool:
    return bool(module_name) and module_name.split(".")[0] == "detail"


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope``'s own body: a nested function, lambda or class is listed, not
    entered, since what it binds is its own."""
    own = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        own.append(node)
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))
    return own


def _module_bindings(own: list[ast.AST]) -> dict[str, types.ModuleType]:
    """Names the imports in ``own`` bind to a ``detail`` module."""
    bound = {}
    for node in own:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and _is_detail(alias.name):
                    bound[alias.asname] = importlib.import_module(alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if not _is_detail(node.module):
                continue
            parent = importlib.import_module(node.module)
            for alias in node.names:
                module = _as_module(parent, alias.name)
                if module is not None:
                    bound[alias.asname or alias.name] = module
    return bound


def _rebinds(own: list[ast.AST]) -> set[str]:
    """Names ``own`` binds to something other than an import."""
    names = set()
    for node in own:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _unresolved_reads(
    scope: ast.AST, visible: dict[str, types.ModuleType], where: str
) -> list[str]:
    """``alias.NAME`` reads in ``scope`` and the scopes inside it that do not resolve."""
    own = _own_nodes(scope)
    here = {**visible, **_module_bindings(own)}
    for name in _rebinds(own):
        here.pop(name, None)
    # A class body's names are not visible to its methods.
    inherited = visible if isinstance(scope, ast.ClassDef) else here
    missing = []
    for node in own:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in here
            and isinstance(node.ctx, ast.Load)
            and not _resolves(here[node.value.id], node.attr)
        ):
            module = here[node.value.id]
            missing.append(
                f"{where}:{node.lineno}: {module.__name__} has no {node.attr}"
            )
        if isinstance(node, _SCOPES):
            missing += _unresolved_reads(node, inherited, where)
    return missing


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
    missing += _unresolved_reads(tree, {}, where)
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


def test_the_scan_rejects_a_top_level_submodule_import_nothing_loaded_yet(monkeypatch):
    # Unload the submodule, as if no file the sweep read earlier had imported it: the
    # ``from detail.gui import animation_export`` binding must import it, not skip it.
    package = importlib.import_module("detail.gui")
    monkeypatch.delitem(sys.modules, "detail.gui.animation_export", raising=False)
    monkeypatch.delattr(package, "animation_export", raising=False)
    source = (
        "from detail.gui import animation_export as ax\n"
        "def export():\n"
        "    return ax.GIF\n"
    )
    assert _unresolved(source, "x.py") == [
        "x.py:3: detail.gui.animation_export has no GIF"
    ]


def test_the_scan_rejects_a_top_level_alias_another_function_shadows():
    source = (
        "import detail.gui.animation_export as ax\n"
        "def plot(ax):\n"
        "    ax.plot()\n"
        "def export():\n"
        "    return ax.GIF\n"
    )
    assert _unresolved(source, "x.py") == [
        "x.py:5: detail.gui.animation_export has no GIF"
    ]


def test_the_scan_accepts_a_submodule_and_ignores_top_level_imports():
    source = (
        "from detail.gui.animation_export import GIF\n"
        "def f():\n"
        "    from detail.api import animation\n"
        "    return animation.GIF\n"
    )
    assert _unresolved(source, "x.py") == []


def test_the_scan_checks_each_function_against_its_own_alias():
    source = (
        "def a():\n"
        "    from detail.api import animation as m\n"
        "    return m.GIF\n"
        "def b():\n"
        "    from detail.gui import animation_export as m\n"
        "    return m.ask_save_path\n"
    )
    assert _unresolved(source, "x.py") == []


def test_the_scan_skips_a_name_the_scope_rebinds():
    source = (
        "def f(ax):\n"
        "    from detail.gui import animation_export as ax\n"
        "    return ax.GIF\n"
    )
    assert _unresolved(source, "x.py") == []
