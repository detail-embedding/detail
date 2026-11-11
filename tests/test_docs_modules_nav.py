"""The Modules nav lists exactly the module pages the build generates (§25).

The class: **a static nav and a generated page set that drift apart**, in either direction, each
failing differently and neither loudly.

- A nav entry with **no page** breaks ``mkdocs build --strict``, but only at build time and with an
  error that names the missing file rather than the module that stopped being generated.
- A page with **no nav entry** is worse, because nothing fails at all: mkdocs publishes every page
  under ``docs_dir`` whether the nav mentions it or not, so a module this build deliberately hides
  keeps a live, reachable reference page. That is precisely the failure §25's docs enumeration
  point exists to prevent, and it is invisible from the terminal.

Generated pages are not on disk, so ``tests/test_docs_nav.py`` — which scans ``docs/`` — cannot see
them. This is the guard for the generated half.

If `mkdocs-literate-nav` is ever added, generating the nav removes this failure mode outright and
this test becomes redundant. Until then the two are declared separately and compared here.
"""

import re
from pathlib import Path

from detail.configs.describe import iter_config_models, surfaced_config_models

_MKDOCS = Path(__file__).resolve().parent.parent / "mkdocs.yml"


def _nav_module_pages() -> set[str]:
    return set(
        re.findall(
            r"modules/([A-Za-z0-9_]+)\.md", _MKDOCS.read_text(encoding="utf-8-sig")
        )
    )


def test_the_nav_lists_exactly_the_generated_module_pages():
    generated = {module_id for module_id, _ in surfaced_config_models()}
    nav = _nav_module_pages()

    missing_page = sorted(nav - generated)
    assert not missing_page, (
        f"mkdocs.yml lists {missing_page} under Modules, but the generator no longer emits "
        f"those pages — `mkdocs build --strict` will fail. If they were hidden by §25, drop "
        f"their nav lines too."
    )
    orphaned = sorted(generated - nav)
    assert not orphaned, (
        f"the generator emits pages for {orphaned} that mkdocs.yml does not list. They are still "
        f"published and reachable by URL, so nothing fails — add the nav lines."
    )


def test_a_hidden_module_would_lose_its_page(monkeypatch):
    """Guard the guard, and pin the mechanism.

    With nothing currently hidden the filter is a no-op, so this hides a live module and asserts
    both halves: the page stops being generated, and the nav check then objects to the stale line.
    """
    from detail import surfaces

    victim = sorted(module_id for module_id, _ in iter_config_models())[0]
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", modules=(victim,))
    )

    assert victim not in {m for m, _ in surfaced_config_models()}
    if victim in _nav_module_pages():
        import pytest

        with pytest.raises(AssertionError, match="mkdocs.yml lists"):
            test_the_nav_lists_exactly_the_generated_module_pages()


def test_the_docs_seam_is_filtered_but_the_registry_is_not():
    """Hidden code stays tested. ``iter_config_models`` feeds ``detail help``, the unit-convention
    scan and the advanced-form guard, so filtering it would stop guarding a hidden module's
    parameters — and switching that module back on would be a surprise, not a toggle."""
    from detail import surfaces

    everything = {module_id for module_id, _ in iter_config_models()}
    hidden = {m for e in surfaces.HIDDEN.values() if e.in_effect for m in e.modules}
    assert hidden <= everything or not hidden


def test_a_module_with_no_registry_declares_the_surface_that_launches_it():
    """The class: **a module id that is not its own way in.**

    A registry member is reached by its id, so asking ``is_surfaced(module_id)`` is the whole
    question. The handful with no registry of their own are reached *only* through somebody
    else's CLI group or GUI section — and the docs filter asked about the id alone. So
    ``cluster_fit`` and ``cluster_apply``, launched solely by the ``clusterer`` group and the
    ``clusterers`` section (both hidden, neither id named in §25's entry, which could hold only
    one ``module``), answered True and published a parameter page each to the user site for a
    capability nobody can open.

    Two halves, both needed: the entry can now name every module of a capability, and the
    filter derives the rest from the way in, so the next one does not have to be remembered.
    """
    from detail.configs.describe import LAUNCHED_BY, _UNREGISTERED_CONFIG_MODELS

    assert (
        _UNREGISTERED_CONFIG_MODELS
    ), "the table is empty — this guard would pass vacuously"
    for module_id, _model, surface in _UNREGISTERED_CONFIG_MODELS:
        assert surface, f"{module_id} names no launch surface"
        assert LAUNCHED_BY[module_id] == surface


def test_a_hidden_launch_surface_takes_its_module_page_with_it(monkeypatch):
    """Guard the guard, on the derivation rather than the declaration.

    Hide a *launch surface* and leave every module id surfaced. The page must still go — the
    half that had never been true: before this, hiding the ``clusterer`` group left
    ``cluster_fit`` and ``cluster_apply`` generating, navigable and published.
    """
    from detail import surfaces
    from detail.configs.describe import LAUNCHED_BY, surfaced_config_models

    # Any module whose door is not its own id. It was `cluster_fit`, removed since.
    victim_module, victim_surface = next(
        (m, door) for m, door in sorted(LAUNCHED_BY.items()) if m != door
    )
    # A table with only the *surface* hidden, never the module id, so the assertion below can
    # only pass by way of the derivation. `_index()` is built per call, so this takes effect.
    monkeypatch.setattr(
        surfaces, "HIDDEN", {"probe": surfaces.Hidden("test", cli=(victim_surface,))}
    )
    assert surfaces.is_surfaced(
        victim_module
    ), "the module id itself must stay surfaced here, or the test proves nothing"
    assert not surfaces.is_surfaced(victim_surface)
    assert victim_module not in {m for m, _ in surfaced_config_models()}
