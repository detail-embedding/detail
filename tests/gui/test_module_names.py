"""A module has one display name, and every surface reads it.

**The class: two surfaces naming the same thing, with nothing comparing them.** A module's
docs page took its title from ``module_id.replace("_", " ").capitalize()`` while the launch
picker took it from a hand-written ``ModuleUI.label``. The two agreed for most modules, which
is why the disagreement went unseen: the docs page for ``query`` was titled **"Query"** while
the whole app called it **"Behavior scan"**.

Nothing failed. ``tests/test_docs_modules_nav.py`` compares the nav to the generated pages by
**path** (``modules/<id>.md``) and never looks at either label, and the nav's own human label
is hand-written a third time in ``mkdocs.yml``. Three names for one module, no comparison —
"each failing differently and neither loudly", as that file's docstring puts it about its own
class.

The rule: ``configs.describe.module_display_name`` is the single source, ``ModuleUI.label``
reads it, and the mkdocs nav label matches. Guarded below in all three directions.
"""

import re
from pathlib import Path

import pytest

from detail.configs.describe import module_display_name, surfaced_config_models

_MKDOCS = Path(__file__).resolve().parents[2] / "mkdocs.yml"


def _nav_labels() -> dict[str, str]:
    """``module_id -> the nav's human label`` for every Modules entry."""
    pattern = re.compile(r"^\s*- (.+): modules/([A-Za-z0-9_]+)\.md\s*$", re.MULTILINE)
    return {
        mid: label
        for label, mid in pattern.findall(_MKDOCS.read_text(encoding="utf-8-sig"))
    }


def test_every_module_ui_label_is_the_shared_display_name(qapp):
    """The launch picker and the docs page cannot disagree, because there is one string.

    Revert ``ModuleUI.label`` to a per-class attribute and this passes only while every
    hand-written copy happens to match — which is the state that shipped "Query".
    """
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES, GUI_MODULES

    for registry in (GUI_MODULES, ANALYSIS_GUI_MODULES):
        for module_id, ui in registry.items():
            assert ui.label == module_display_name(module_id), (
                f"{module_id}: the picker says {ui.label!r} and the docs page would be "
                f"titled {module_display_name(module_id)!r}"
            )


def test_the_nav_label_matches_the_page_title(qapp):
    """``mkdocs.yml``'s Modules nav writes the name a third time, by hand. A nav entry
    reading one thing over a page titled another is invisible to every existing guard.

    Scoped to ids a user can *launch* — the ones with a ``ModuleUI``, whose label is the
    thing the nav has to agree with. The nav also lists **methods and trainers** (``soft_knn``,
    ``dtw_finch``, ``ultralytics_yolo``…) whose labels deliberately carry a category the id
    cannot ("Query method - soft k-NN"); they have no picker entry to disagree with, so pinning
    them here would only forbid a useful nav label.
    """
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES, GUI_MODULES

    launchable = set(GUI_MODULES) | set(ANALYSIS_GUI_MODULES)
    nav = _nav_labels()
    mismatched = {
        module_id: (nav[module_id], module_display_name(module_id))
        for module_id, _model in surfaced_config_models()
        if module_id in nav
        and module_id in launchable
        and nav[module_id] != module_display_name(module_id)
    }
    assert not mismatched, f"nav label != page title: {mismatched}"


@pytest.mark.parametrize(
    "module_id, expected",
    [
        ("temporal_clustering", "Behavior clustering"),
        ("query", "Behavior scan"),
        ("pose_clustering", "Pose clustering"),  # the fallback still works
    ],
)
def test_the_names_people_actually_read(module_id, expected):
    """The three that matter, stated rather than derived — ``temporal_clustering`` because
    the screen it opens has called everything a *behavior* since #278 while the module was
    still called "Temporal clustering", and ``query`` because its page was titled "Query".
    """
    assert module_display_name(module_id) == expected
