"""A page that names itself with a kind is reachable by a node of that kind.

The class: **a way in is not a way back.** A screen can be perfectly reachable on the way
into a scope and unreachable from inside it, and nothing distinguishes the two states except
trying to return. `ExperimentOverviewPage` was shown once by `_open_experiment` on the way
into a project and had no rail node (#340/#331), so clicking any section lost it — and with
it *Export experiment*, *Delete experiment* and the name/description edit. The reporter found
the export, navigated away, and could not get back: *"this only shows up when you click into
the experiments. There is no way to get back to it."*

**Why the obvious sweep does not work, and this one does.** The issue asked for "sweep every
page class the app can construct and assert it is reachable". Written that way the guard
**passes on the bug it exists for**: `ExperimentOverviewPage` *is* constructed by
`_open_experiment`, which *is* called from `_show_experiments_home`, so a reachability walk
from the banner finds it and reports success. Measured before writing this — the naive form
is green on the broken tree. A guard that cannot fail on its own motivating case is the
fixture hazard this repo keeps paying for.

What separates the two states is *scope*. So the property asserted here is the one the shell
already encodes: a page that calls `_set_page_export(kind)` or `_set_page_help(kind)` is
**claiming to be the page for that kind**, and the rail is what carries kinds. If the shell
names a kind no node carries, that page is somewhere you cannot be sent — you can only arrive
by accident and never return.

**The rail is read, not listed.** `_rail_kinds` builds a real project rail and reads the
kinds off its nodes. Listing them here instead would make the guard circular: it would go on
passing with the Overview node deleted, because the expected set would still contain the kind
nobody could reach. Delete the node and this fails, which is the whole point.

Three-quarters of the kinds in `_BANNER_SCOPED` are landings and Configuration drill-ins,
which are legitimately not rail nodes (`docs/gui_design.md` §3.1: a banner mode without a rail
is a full-width page, and sub-navigation within one is drill-in). Each says so.

**The third shape: a page whose only door is a delete.** The Model manager rework (#136)
dropped the one link to the experiment's trained-models index, and `ModelsPage` stayed
reachable through exactly one path — `_on_model_deleted`, which lands on the page listing what
was just deleted. A user found it by deleting a model. Neither sweep could see it: the page
names no kind, so the first one never asks about it, and `test_nav_dispatch`'s orphan-opener
scan passes because `_open_models` *is* called. `DatasetsPage` and the Distill-runs page were in
the same state. So the second half of this file drives every `_on_*_deleted` handler on a
real window and asserts it lands on a page some project-rail node also opens — read off the
real rail, for the reason given above.
"""

import ast
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import app as gui_app  # noqa: E402
from detail.gui import sections as gui_sections  # noqa: E402

pytestmark = pytest.mark.guard

APP_SOURCE = Path(gui_app.__file__)

#: The kinds the shell names that are reached from the **banner** or a drill-in rather than
#: from a rail node, and why each is. Claims, not exemptions: `test_no_stale_banner_scope`
#: fails on an entry the shell no longer names.
_BANNER_SCOPED = {
    "experiments_group": "the Experiments landing — a banner mode, and the rail's absence is "
    "what says you are not in a project yet",
    "active_processes": "the Active processes landing — a banner mode, full width, no rail",
    "config_group": "the Configuration checklist — a banner mode whose steps drill in",
    "model_manager_group": "the Model manager landing — a banner mode, full width",
    "keypoints_manager_group": "the Keypoints manager (§26) — a banner mode, full width: "
    "skeletons are machine-wide, so no experiment's rail holds them",
    "tools_group": "the Quick tools landing — a banner mode; its rail carries tools, not this",
    "documentation": "the Docs screen — a banner mode",
    "gpu_cuda": "a Configuration drill-in, shown full width from the checklist",
    "sam3_backend": "a Configuration drill-in, shown full width from the checklist",
    "hugging_face": "a Configuration drill-in, shown full width from the checklist",
    "remote_targets": "a Configuration drill-in (Remote targets), reached by its own "
    "Open -> on the Configuration landing, not the checklist -- zero registered targets "
    "is a normal setup, not an incomplete one, so it isn't a checklist step either",
    "distill": "a distill run's page, a drill-in from the Distill runs table on the Models "
    "page — which a rail node opens, so the way back is Models, then the run's row",
}

#: The two calls by which a page declares which kind it is showing.
_SELF_DECLARING = ("_set_page_export", "_set_page_help")


def _declared_kinds() -> dict[str, int]:
    """``kind -> line`` for every literal the shell passes to one of those two calls.

    Non-literal arguments are skipped deliberately: `_navigate` forwards the node's own kind
    (`self._set_page_help(kind)`), which is carried by a rail node by construction — it came
    off one. Only a hand-written string can name a kind the rail does not have.
    """
    found: dict[str, int] = {}
    for node in ast.walk(ast.parse(APP_SOURCE.read_text(encoding="utf-8-sig"))):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SELF_DECLARING
            and node.args
        ):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.setdefault(first.value, node.lineno)
    return found


def _rail_kinds(experiment) -> set[str]:
    """Every kind the project rail can carry, read off a real rail.

    Top-level nodes are built; a section's children are not (they are read off the disk when
    the folder opens, #316), so the child kinds come from the same declaration the fill uses.
    """
    window = gui_app.MainWindow()
    try:
        window._build_project_rail(experiment)
        tree = window._tree
        top = {
            tree.topLevelItem(i).data(0, gui_app._ROLE)[0]
            for i in range(tree.topLevelItemCount())
        }
    finally:
        window.deleteLater()
    children = {s.child_kind for s in gui_sections.sections() if s.child_kind}
    return top | children


@pytest.fixture
def experiment(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    return proj


def test_every_kind_the_shell_names_is_carried_by_a_node(qapp, experiment):
    carried = _rail_kinds(experiment)
    stranded = {
        kind: line
        for kind, line in _declared_kinds().items()
        if kind not in carried and kind not in _BANNER_SCOPED
    }
    assert not stranded, (
        f"the shell shows a page as {sorted(stranded)} and no rail node carries that kind "
        f"(app.py:{min(stranded.values())}). That page can be arrived at but not returned to: "
        f"its Export control and its ? go with it the moment you click anything else, and "
        f"nothing fails. Give the kind a rail node, or add it to _BANNER_SCOPED with the "
        f"reason it is reached from the banner instead."
    )


def test_no_stale_banner_scope():
    """An exemption that excuses nothing is how a list stops meaning anything."""
    stale = sorted(set(_BANNER_SCOPED) - set(_declared_kinds()))
    assert not stale, f"_BANNER_SCOPED names kinds the shell no longer shows: {stale}"


def test_the_sweep_would_catch_a_page_with_no_way_back(qapp, experiment, monkeypatch):
    """Guard the guard, and on the property that matters: the sweep has to fail when a kind
    the shell names stops being carried by a node.

    Driven by removing the node rather than by inventing a fake kind — a made-up kind is
    absent from the rail by construction and would prove nothing about whether the rail is
    really being read. This removes the **Overview** node, which is exactly the state
    `origin/main` was in, and the sweep must notice.
    """
    real = gui_app.MainWindow._experiment_section_items

    def without_overview(self, exp):
        return [
            item
            for item in real(self, exp)
            if item.data(0, gui_app._ROLE)[0] != "experiment"
        ]

    monkeypatch.setattr(
        gui_app.MainWindow, "_experiment_section_items", without_overview
    )
    with pytest.raises(AssertionError, match="no rail node carries that kind"):
        test_every_kind_the_shell_names_is_carried_by_a_node(qapp, experiment)


# --------------------------------------------------------------------------- #
# The second door: where a delete puts you
# --------------------------------------------------------------------------- #
#: The after-delete handlers that do not land on a project-rail page, and why. Claims, not
#: exemptions: `test_no_stale_landing_claim` fails on one that has since become ordinary.
_LANDS_OFF_THE_RAIL = {
    "_on_experiment_deleted": "the experiment is gone, so there is no project rail to be on — "
    "it returns to the Experiments landing, a banner mode",
    "_on_tool_run_deleted": "a Quick tools run; it lands on its tool's page, a node of the "
    "tools rail rather than the project rail",
}


def _after_delete_handlers() -> list[str]:
    return sorted(
        name
        for name in vars(gui_app.MainWindow)
        if name.startswith("_on_") and name.endswith("_deleted")
    )


def _landings(qapp, experiment, handlers, monkeypatch) -> tuple[dict, set]:
    """``(handler -> page class it shows, page classes the project rail opens)``."""
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: experiment.dir.parent)
    window = gui_app.MainWindow()
    try:
        window._open_experiment(experiment)
        tree = window._tree
        rail_pages = set()
        for i in range(tree.topLevelItemCount()):
            window._navigate(tree.topLevelItem(i))
            rail_pages.add(type(window._content.currentWidget()))
        landed = {}
        for name in handlers:
            getattr(window, name)()
            landed[name] = type(window._content.currentWidget())
    finally:
        window.deleteLater()
    return landed, rail_pages


def test_a_delete_lands_somewhere_the_rail_can_take_you(qapp, experiment, monkeypatch):
    """Deleting a thing puts you on the page that lists its kind — and that page must be one
    a rail node opens, or the delete is the only way there.

    The instance: the Model manager rework (#136) dropped the one link to the experiment's
    trained-models index, and `ModelsPage` stayed reachable through exactly one door —
    `_on_model_deleted`. The user found it by deleting a model. It was invisible to the sweep
    above, which only sees pages that *name* their kind, and `_open_models` named none; and to
    `test_nav_dispatch`'s orphan-opener scan, because the opener *was* called — by the delete.
    """
    handlers = [h for h in _after_delete_handlers() if h not in _LANDS_OFF_THE_RAIL]
    landed, rail_pages = _landings(qapp, experiment, handlers, monkeypatch)
    stranded = {h: cls.__name__ for h, cls in landed.items() if cls not in rail_pages}
    assert not stranded, (
        f"after a delete the shell shows {stranded}, and no project-rail node opens that "
        f"page — so deleting something is the only way to get there. Give the section a "
        f"node in gui/sections.py, or land the delete on the page that lists it."
    )


def test_no_stale_landing_claim():
    """An entry must still be true: its handler exists."""
    missing = sorted(set(_LANDS_OFF_THE_RAIL) - set(_after_delete_handlers()))
    assert not missing, f"_LANDS_OFF_THE_RAIL names handlers that are gone: {missing}"


def test_the_landing_sweep_would_catch_a_section_with_no_node(
    qapp, experiment, monkeypatch
):
    """Guard the guard on its motivating state: remove the Models section from the rail —
    `origin/main` before this fix — and the sweep must name `_on_model_deleted`."""
    real = gui_sections.sections
    monkeypatch.setattr(
        gui_sections,
        "sections",
        lambda: tuple(s for s in real() if s.kind != "models"),
    )
    with pytest.raises(AssertionError, match="_on_model_deleted"):
        test_a_delete_lands_somewhere_the_rail_can_take_you(
            qapp, experiment, monkeypatch
        )
