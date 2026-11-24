"""Prose written for the docs is prose a reader can reach.

The class: **a single-source description module with no generator behind it.** A
``*_docs.py`` module under ``detail/`` exists to be *the* place a method describes itself, and the
value only lands when a ``scripts/gen_docs_*.py`` script renders it into a page and the page is in
the nav. Miss the last step and the module still imports, its own anti-drift tests still pass, and
the text is unreachable from the app, the site and the CLI alike.

How it hid: ``analysis/cluster_methods_docs.py`` had a full description of every clustering rule,
a test pinning it against the config Literals, and a docstring saying "the same shape as
``analysis/query_methods_docs.py``" (a module since removed) — and no generator. Its answer to
"what is the cluster hierarchy derived from" had been written for months and had never been on a
screen. Nothing failed, because nothing about an unread function is broken; the reviewer asking the
question in a design meeting was the only signal, and the answer to give them was already in the
repo.

The rule: a ``*_docs.py`` module anywhere under ``detail/`` is imported by a generator script, and
that script writes a page the nav carries. The scan is over the modules that exist, so a new one is
covered the day it is added rather than the day somebody remembers this file. It walks the whole
package rather than ``detail/analysis/``, where these modules first lived: the two there went with
the pages they fed, and the one left (``transforms/constructor_docs.py``) sat outside the scan.

Sibling of ``tests/test_docs_nav.py``, which covers the pages that *are* on disk; generated pages
are invisible to that scan by construction, which is exactly why they need this one.
"""

import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "detail"
SCRIPTS = REPO / "scripts"
MKDOCS_YML = REPO / "mkdocs.yml"


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores mkdocs' custom tags (``!ENV``, ``!!python/...``)."""


_TolerantLoader.add_multi_constructor(None, lambda loader, suffix, node: None)


def _config() -> dict:
    return yaml.load(MKDOCS_YML.read_text(encoding="utf-8-sig"), Loader=_TolerantLoader)


def _nav_targets(nav) -> set[str]:
    """Every markdown path the nav points at, at any depth."""
    out: set[str] = set()
    if isinstance(nav, str):
        out.add(nav)
    elif isinstance(nav, list):
        for item in nav:
            out |= _nav_targets(item)
    elif isinstance(nav, dict):
        for value in nav.values():
            out |= _nav_targets(value)
    return out


def _generator_scripts() -> list[pathlib.Path]:
    """The generator scripts mkdocs actually runs — not whatever is lying in ``scripts/``.

    Read from ``gen-files`` rather than globbed, because a script on disk that the config does not
    list is exactly as unreachable as no script at all.
    """
    for plugin in _config().get("plugins") or []:
        if isinstance(plugin, dict) and "gen-files" in plugin:
            listed = (plugin["gen-files"] or {}).get("scripts") or []
            return [REPO / path for path in listed]
    raise AssertionError("mkdocs.yml has no gen-files plugin")


def _docs_modules(package: pathlib.Path = PACKAGE) -> list[pathlib.Path]:
    return sorted(package.rglob("*_docs.py"))


def _generators_of(
    module: pathlib.Path, scripts: list[pathlib.Path]
) -> list[pathlib.Path]:
    """The generator scripts that import ``module``, read off their source."""
    return [s for s in scripts if module.stem in s.read_text(encoding="utf-8-sig")]


def _written_pages(script: pathlib.Path) -> set[str]:
    """The page paths a generator writes, read off its ``mkdocs_gen_files.open`` calls."""
    text = script.read_text(encoding="utf-8-sig")
    return set(
        re.findall(r"""mkdocs_gen_files\.open\(\s*f?["']([^"']+\.md)["']""", text)
    )


def test_at_least_one_docs_module_exists():
    """Guard the guard: an empty glob would make every parametrised case below vacuous."""
    assert (
        _docs_modules()
    ), "no *_docs.py modules under detail/ — this scan proves nothing"


@pytest.mark.parametrize("module", _docs_modules(), ids=lambda p: p.stem)
def test_every_docs_module_is_rendered_by_a_generator(module):
    """The module's text reaches a page, and the page reaches the nav."""
    users = _generators_of(module, _generator_scripts())
    assert users, (
        f"{module.relative_to(REPO).as_posix()} describes something for the docs and no "
        "generator in mkdocs.yml's gen-files list imports it, so nothing it says is reachable by "
        "a user. "
        "Add a scripts/gen_docs_*.py that renders it, and put the page in the nav."
    )

    nav = _nav_targets(_config().get("nav"))
    pages = {page for script in users for page in _written_pages(script)}
    assert pages, f"{users[0].name} imports {module.stem} but writes no page"
    assert pages & nav, (
        f"{module.stem} is rendered into {sorted(pages)}, and the nav carries none of them — "
        "mkdocs would publish an orphan page nobody can navigate to."
    )


def test_the_scan_would_catch_an_unrendered_module(tmp_path):
    """The negative case. A scan that matches everything today passes vacuously forever.

    The orphan is planted in a copy of the package's shape under ``tmp_path``, never in
    ``detail/``: a file created and deleted in the source tree mid-run is seen coming and
    going by every other test that walks ``detail/`` in parallel (``tests/source_tree.py``).
    """
    orphan = tmp_path / "zz_orphan_docs.py"
    orphan.write_text('"""Temporary, for the guard\'s own test."""\n', encoding="utf-8")
    assert orphan in _docs_modules(tmp_path), "the glob did not pick the orphan up"
    assert not _generators_of(
        orphan, _generator_scripts()
    ), "an unrendered module was not detected as unrendered"


def test_the_behavior_clustering_page_answers_what_the_hierarchy_is_built_on():
    """The question that surfaced this gap, asked in review and previously unanswerable on screen.

    Asserted on the intro's content rather than on the page existing: a page that renders the
    tables and never says what the tree is derived from would satisfy every test above and leave
    the reviewer's question exactly where it was. It was the Clustering methods page's intro until
    that page, which described only the removed clusterer's rules, went with it; the answer moved
    to the behavior clustering page, which is the hierarchy users now reach.
    """
    intro = (
        REPO / "docs" / "modules" / "_intros" / "temporal_clustering.md"
    ).read_text(encoding="utf-8-sig")
    assert "nearest neighbour" in intro
    assert "distance between bouts" in intro
    # It also has to rule things *out* — "derived from distance" is only informative beside the
    # things a reader would otherwise assume it used.
    assert "how often" in intro and "where in the arena" in intro
