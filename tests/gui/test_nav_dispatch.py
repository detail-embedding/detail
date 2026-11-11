"""Every ``self.<method>()`` the app shell calls actually exists on the shell.

The class: **a rename that moved a definition and left behind a caller on a path nothing
runs.** Python resolves an attribute when the line executes, so a dispatcher branch that
names a method by hand is checked only when that branch is taken, and a branch guarded by a
hidden surface (§25) is taken by nobody.

The instance this was written for shipped in #210, the §18 rename. ``MainWindow._navigate``'s
``"benchmark"`` branch kept calling ``self._open_comparison(benchmark_id)`` after the method
became ``_open_benchmark``, so clicking a benchmark in the rail raised ``AttributeError``.
Three things hid it at once:

1. The section was hidden in that build, so the branch was reachable only under
   ``DETAIL_SHOW_ALL`` -- which is exactly how the research workflow reached it. (Benchmarks
   have since been removed.)
2. The sibling path *worked*. ``_open_benchmarks`` handed ``BenchmarksPage`` a reference to
   ``self._open_benchmark``, so opening a benchmark by double-clicking a table row was fine;
   only the rail leaf was dead. A bug that half-works reads as working.
3. The construction smoke (``test_smoke_construct.py``) builds every page, which catches a
   dead ``signal.connect`` at construction time. This one is not a connect and not
   construction: it is a plain call inside an ``elif``, invisible until dispatch.

**Why a source scan and not a click-through.** Driving the rail would need a fixture carrying
one node of every kind, and would still skip the hidden kinds -- the ones with the least
coverage and, for that reason, the most drift. Reading the calls out of the source covers
every branch whether or not anything can reach it, costs nothing, and is data-driven over the
class, so a section added later is covered the day its branch is written.

Resolution is against the real class, never a list of names parsed out of the file: that way
inherited Qt methods (``setWindowTitle``, ``statusBar``) resolve on their own and this guard
needs no allow list to fall out of date.

The second scan below is that one's mirror, and the other half of "a capability with no way
in": **an opener nobody calls.** The first scan asks whether every name the shell calls
exists; this one asks whether every destination the shell can *build* is named from anywhere.
Both failures are silent for the same reason -- a page that nothing opens raises nothing,
renders nothing, and fails no test. ``MainWindow._open_global_models`` was the instance: it
built a standalone Model installs page, carried a docstring calling itself "kept as a direct
opener", and was referenced by no line in ``detail/``, while ``_open_model_manager`` wired the
same readiness banner and ``_go_to_node`` routed that step to it. ``test_no_orphan_functions``
cannot see it -- that guard skips ``_``-prefixed privates, which every opener here is.

**References, not calls.** An opener is at least as often *handed* to a page as invoked --
``ExperimentAnalysisPage(..., self._open_analysis_run, ...)`` is how a table row drills in --
so a scan for ``self._x(...)`` would read every one of those as dead. The edge is any mention
of the name.
"""

import ast
from pathlib import Path

import pytest

from detail.gui import app as gui_app

APP_SOURCE = Path(gui_app.__file__)


def _class_nodes(source: str) -> list[ast.ClassDef]:
    return [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)]


def _assigned_attrs(node: ast.ClassDef) -> set[str]:
    """``self.x = ...`` anywhere in the class -- an instance attribute, not a method, so it is
    absent from the class object and has to be collected from the source."""
    return {
        target.attr
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }


def _self_calls(node: ast.ClassDef) -> dict[str, int]:
    """``self.x(...)`` call sites, mapped to the line of the first one."""
    calls: dict[str, int] = {}
    for stmt in ast.walk(node):
        if (
            isinstance(stmt, ast.Call)
            and isinstance(stmt.func, ast.Attribute)
            and isinstance(stmt.func.value, ast.Name)
            and stmt.func.value.id == "self"
        ):
            calls.setdefault(stmt.func.attr, stmt.lineno)
    return calls


def _unresolved(node: ast.ClassDef, cls: type) -> dict[str, int]:
    assigned = _assigned_attrs(node)
    return {
        name: line
        for name, line in _self_calls(node).items()
        if not hasattr(cls, name) and name not in assigned
    }


@pytest.mark.parametrize(
    "class_name",
    [n.name for n in _class_nodes(APP_SOURCE.read_text(encoding="utf-8-sig"))],
)
def test_every_self_call_in_the_shell_resolves(class_name):
    node = next(
        n
        for n in _class_nodes(APP_SOURCE.read_text(encoding="utf-8-sig"))
        if n.name == class_name
    )
    unresolved = _unresolved(node, getattr(gui_app, class_name))
    assert not unresolved, (
        f"{class_name} calls {sorted(unresolved)} on itself, and no such attribute exists "
        f"(first call at app.py:{min(unresolved.values())}). A dispatcher branch resolves its "
        f"method only when that branch runs, so this raises AttributeError for whoever reaches "
        f"it -- and a hidden section (§25) means that may be nobody until a user sets "
        f"DETAIL_SHOW_ALL."
    )


#: An opener with no caller, and the reason it is allowed to have none. Empty on purpose: the
#: one entry this list would have held was deleted instead, which is the remedy the failure
#: message names first. An entry here is a claim that a destination is deliberately buildable
#: and deliberately unreachable -- rare enough to need saying out loud.
_OPENERS_WITH_NO_CALLER: dict[str, str] = {}


def _self_refs(node: ast.ClassDef) -> set[str]:
    """Every ``self.x`` mentioned in the class, called or merely passed along."""
    return {
        stmt.attr
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Attribute)
        and isinstance(stmt.value, ast.Name)
        and stmt.value.id == "self"
    }


def _openers(node: ast.ClassDef) -> dict[str, int]:
    """``_open_*`` / ``_go_to_*`` methods defined on the class, mapped to their line."""
    return {
        stmt.name: stmt.lineno
        for stmt in node.body
        if isinstance(stmt, ast.FunctionDef)
        and stmt.name.startswith(("_open_", "_go_to"))
    }


def _uncalled(node: ast.ClassDef) -> dict[str, int]:
    """Openers named by no *other* method -- a destination with no way in."""
    uncalled = {}
    for name, line in _openers(node).items():
        others = [m for m in node.body if getattr(m, "name", None) != name]
        if not any(name in _self_refs(m) for m in others if isinstance(m, ast.AST)):
            uncalled[name] = line
    return uncalled


def test_every_opener_in_the_shell_is_reachable():
    node = next(
        n
        for n in _class_nodes(APP_SOURCE.read_text(encoding="utf-8-sig"))
        if n.name == "MainWindow"
    )
    uncalled = {
        name: line
        for name, line in _uncalled(node).items()
        if name not in _OPENERS_WITH_NO_CALLER
    }
    assert not uncalled, (
        f"MainWindow defines {sorted(uncalled)} and nothing names them "
        f"(app.py:{min(uncalled.values())}). An opener nobody calls builds a page no user can "
        f"reach: nothing raises, nothing renders, and no test fails. Delete it, wire it to a "
        f"rail node or a landing button, or add it to _OPENERS_WITH_NO_CALLER with the reason."
    )


def test_the_scan_would_catch_an_opener_nobody_calls():
    """Guard the guard, on the property that actually matters here: the scan must count a
    reference that is *passed* as reaching the opener, or it would report every drill-in
    callback in the shell as dead and be turned off within the week."""
    source = (
        "class Shell:\n"
        "    def _switch(self):\n"
        "        self._open_reached()\n"
        "    def _land(self):\n"
        "        Page(on_open=self._open_handed_over)\n"
        "    def _open_reached(self):\n"
        "        pass\n"
        "    def _open_handed_over(self):\n"
        "        pass\n"
        "    def _open_stranded(self):\n"
        "        pass\n"
    )
    uncalled = _uncalled(_class_nodes(source)[0])
    assert set(uncalled) == {"_open_stranded"}, uncalled


def test_the_scan_would_catch_a_renamed_method():
    """Guard the guard. A scan that matches nothing passes forever, and this one would go
    quiet the moment ``_self_calls`` stopped recognising a call shape."""
    source = (
        "class Shell:\n"
        "    def _dispatch(self, kind):\n"
        "        if kind == 'a':\n"
        "            self._open_thing()\n"
        "        else:\n"
        "            self._renamed_away()\n"
        "    def _open_thing(self):\n"
        "        pass\n"
    )
    node = _class_nodes(source)[0]

    class Shell:
        def _dispatch(self, kind):
            pass

        def _open_thing(self):
            pass

    unresolved = _unresolved(node, Shell)
    assert set(unresolved) == {"_renamed_away"}, unresolved
