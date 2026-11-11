"""Doc↔tool map coverage + build-level anchor resolution (PR 2).

The map (``detail/help.py``) must cover every field, module, screen, and wired `?`, and each
target must resolve to a real anchor — checked against the source and, definitively, against the
built site's HTML. A new field / screen / module with no doc, or a `?` pointing nowhere, fails CI.
"""

import re
from pathlib import Path

import pytest

import detail
from detail import docs_access
from detail import help as help_mod
from detail.configs.describe import (
    describe_config,
    iter_config_models,
    surfaced_config_models,
)

_GUI = Path(detail.__file__).parent / "gui"


def _navigate_kinds() -> set[str]:
    """Every navigable sidebar kind, scanned from app.py::_navigate's ``kind == "..."`` chain.
    Underscore-prefixed kinds (``_placeholder``) are structural folders, not pages."""
    src = (_GUI / "app.py").read_text(encoding="utf-8-sig")
    return set(re.findall(r'kind == "([a-z][a-z_]*)"', src))


def _help_button_calls() -> list[tuple[str, str]]:
    """``(rel:line, key)`` for every ``help_button("<key>")`` call under detail/gui/."""
    pattern = re.compile(r'help_button\(\s*["\']([^"\']+)["\']')
    calls = []
    for py in _GUI.rglob("*.py"):
        for lineno, line in enumerate(
            py.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            for match in pattern.finditer(line):
                rel = py.relative_to(_GUI.parent.parent).as_posix()
                calls.append((f"{rel}:{lineno}", match.group(1)))
    return calls


def test_every_field_resolves_to_its_module_page_anchor():
    missing = [
        f"{module_id}.{p['name']}"
        for module_id, model in iter_config_models()
        for p in describe_config(model)
        if not help_mod.anchor_exists(
            *_target(help_mod.field_topic(module_id, p["name"]))
        )
    ]
    assert not missing, "fields with no module-page anchor: " + ", ".join(missing)


def test_field_topic_bogus_field_does_not_resolve():
    # The guard must bite, or field coverage proves nothing.
    assert not help_mod.anchor_exists(
        *_target(help_mod.field_topic("detection", "nope"))
    )


# The Documentation screen *is* the docs (its own "Open full docs" button), so it
# intentionally has no page-top "Info" ? and no PAGE_DOCS entry.
_NO_PAGE_DOC_KINDS = {"documentation"}


def test_every_navigable_screen_has_a_resolving_page_doc():
    missing = []
    for kind in sorted(_navigate_kinds() - _NO_PAGE_DOC_KINDS):
        topic = help_mod.PAGE_DOCS.get(kind)
        if topic is None:
            missing.append(f"{kind} (no PAGE_DOCS entry)")
        elif not help_mod.anchor_exists(topic.page, topic.anchor):
            missing.append(f"{kind} -> {topic.page}#{topic.anchor}")
    assert not missing, "navigable screens with no resolving doc: " + ", ".join(missing)


def test_every_wired_help_button_topic_resolves():
    offenders = []
    for where, key in _help_button_calls():
        topic = help_mod.HELP.get(key)
        if topic is None:
            offenders.append(f"{where}: help_button('{key}') not in HELP")
        elif not help_mod.anchor_exists(topic.page, topic.anchor):
            offenders.append(f"{where}: '{key}' -> {topic.page}#{topic.anchor}")
    assert not offenders, "unresolved help_button topics:\n" + "\n".join(offenders)


def _field_help_calls() -> list[tuple[str, str, str]]:
    """``(rel:line, module_id, field)`` for every ``field_label/field_help_button("m","f",...)``."""
    pattern = re.compile(
        r"""field_(?:label|help_button)\(\s*["']([^"']+)["']\s*,\s*["']([^"']+)["']"""
    )
    calls = []
    for py in _GUI.rglob("*.py"):
        text = py.read_text(encoding="utf-8-sig")
        rel = py.relative_to(_GUI.parent.parent).as_posix()
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            calls.append((f"{rel}:{line}", match.group(1), match.group(2)))
    return calls


def test_every_wired_field_help_resolves():
    offenders = []
    for where, module_id, field in _field_help_calls():
        topic = help_mod.field_topic(module_id, field)
        if not help_mod.anchor_exists(topic.page, topic.anchor):
            offenders.append(
                f"{where}: {module_id}.{field} -> {topic.page}#{topic.anchor}"
            )
    assert not offenders, "unresolved field ? topics:\n" + "\n".join(offenders)


def _target(topic: help_mod.HelpTopic) -> tuple[str, str | None]:
    return topic.page, topic.anchor


def _all_mapped_targets() -> set[tuple[str, str | None]]:
    """Every doc target the help map can deep-link to, for the built-site check below.

    Modules come from ``surfaced_config_models``: a module this build hides (§25) publishes no
    ``modules/<id>.html``, so its targets are not resolvable *and* not reachable — its config
    form, where the `?` buttons live, is never built — the module is absent from every launch
    list, and Refine is withheld on its finished runs.

    Its `?` buttons stay wired in source on purpose (§25 keeps hidden code in the tree, tested
    and switchable back on), so scanning call sites would only re-report that. The one way to
    build such a form is ``DETAIL_SHOW_ALL``, where a deliberately revealed surface deep-linking
    to an unpublished page is a developer's problem, not a user's.
    """
    targets = {_target(t) for t in help_mod.HELP.values()}
    targets |= {_target(t) for t in help_mod.PAGE_DOCS.values()}
    for module_id, model in surfaced_config_models():
        targets.add(_target(help_mod.module_topic(module_id)))
        for p in describe_config(model):
            targets.add(_target(help_mod.field_topic(module_id, p["name"])))
    return targets


@pytest.fixture(scope="session")
def built_site(tmp_path_factory):
    """Build the docs site once into a throwaway cache for the whole session (~4 s)."""
    cache = tmp_path_factory.mktemp("docs-cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(docs_access.DOCS_CACHE_ENV, str(cache))
        mp.delenv(docs_access.DOCS_URL_ENV, raising=False)
        yield docs_access.ensure_site_built()


def test_every_mapped_anchor_is_a_real_id_in_the_built_site(built_site):
    # The definitive deep-link check: what file://<page>.html#<anchor> actually jumps to.
    missing = []
    for page, anchor in sorted(_all_mapped_targets(), key=lambda t: (t[0], t[1] or "")):
        html = built_site / f"{page}.html"
        if not html.is_file():
            missing.append(f"{page} (page missing)")
        elif anchor is not None and f'id="{anchor}"' not in html.read_text(
            encoding="utf-8-sig"
        ):
            missing.append(f"{page}#{anchor}")
    assert not missing, "mapped anchors not in the built site: " + ", ".join(missing)
