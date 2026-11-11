"""Every markdown file in ``docs/`` is either in the site nav or excluded.

The class: **an internal, agent-facing document dropped into a published docs
tree without being declared internal.** mkdocs publishes every file under
``docs_dir`` that ``exclude_docs`` does not name, whether or not the nav
mentions it — so an internal note lands on the user site, and the first link it
carries to a path outside ``docs/`` (a sibling folder at the repo root) fails
``mkdocs build --strict``.

How it hid: nothing is wrong at the line where it is written — the file is
valid markdown and the site builds locally until someone adds a cross-tree
link. mkdocs reports the nav omission at INFO, not WARNING, so the only signal
is the unrelated-looking broken-link warning that appears commits later.
``docs/clustering_experiments.md`` and ``docs/behavior_clustering_ideas.md``
both sat published for weeks; the first one to grow links out of ``docs/`` took
the docs workflow red.

The rule: a new file under ``docs/`` is a **user page in the nav** or an
**internal doc in ``exclude_docs``** (``docs/docs_design.md`` §1). There is no
third state. Pages generated at build time by ``mkdocs-gen-files`` are not on
disk, so this scan does not see them and needs no exemption.
"""

import fnmatch
import pathlib
import re

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
MKDOCS_YML = REPO / "mkdocs.yml"


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores mkdocs' custom tags (``!ENV``, ``!!python/...``)."""


# Registered on the subclass only, so the global SafeLoader keeps rejecting them.
_TolerantLoader.add_multi_constructor(None, lambda loader, suffix, node: None)


def _nav_paths(nav: object) -> set[str]:
    """Every doc path reachable from a mkdocs ``nav`` tree."""
    if isinstance(nav, str):
        return {nav}
    if isinstance(nav, list):
        return set().union(*(_nav_paths(item) for item in nav))
    if isinstance(nav, dict):
        return set().union(*(_nav_paths(value) for value in nav.values()))
    return set()


def _is_excluded(rel: str, patterns: list[str]) -> bool:
    """Whether ``exclude_docs`` covers ``rel`` (gitignore-style, as mkdocs reads it)."""
    name = pathlib.PurePosixPath(rel).name
    return any(
        fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) for pat in patterns
    )


def _undeclared(
    rel_paths: list[str], nav_paths: set[str], excludes: list[str]
) -> list[str]:
    """The docs paths mkdocs would publish without anyone having said to."""
    return [
        rel
        for rel in rel_paths
        if rel not in nav_paths and not _is_excluded(rel, excludes)
    ]


def _config() -> dict:
    return yaml.load(MKDOCS_YML.read_text(encoding="utf-8-sig"), Loader=_TolerantLoader)


def test_every_docs_page_is_navigated_or_excluded():
    config = _config()
    docs_dir = REPO / config["docs_dir"]
    excludes = [
        line.strip()
        for line in config.get("exclude_docs", "").splitlines()
        if line.strip()
    ]
    rel_paths = sorted(
        p.relative_to(docs_dir).as_posix() for p in docs_dir.rglob("*.md")
    )

    offenders = _undeclared(rel_paths, _nav_paths(config["nav"]), excludes)

    assert not offenders, (
        "these docs/ pages are neither in mkdocs.yml's nav nor in exclude_docs, so "
        "mkdocs publishes them to the user site and any link they carry outside "
        "docs/ fails `mkdocs build --strict`. Add each to the nav (a user page) or "
        "to exclude_docs (an internal, agent-facing doc — docs/docs_design.md §1):\n"
        + "\n".join(offenders)
    )


def test_the_scan_would_catch_an_undeclared_page():
    """Guard the guard: the check must not pass vacuously."""
    nav_paths = {"index.md"}
    excludes = ["TODO.md", "modules/_intros/*"]
    rel_paths = ["index.md", "TODO.md", "modules/_intros/detection.md", "notes.md"]

    assert _undeclared(rel_paths, nav_paths, excludes) == ["notes.md"]


def _nav_sections(nav: list) -> dict[str, list[str]]:
    """``{section title: [page paths]}`` for each top-level group in the nav."""
    return {
        title: sorted(_nav_paths(value))
        for item in nav
        if isinstance(item, dict)
        for title, value in item.items()
    }


def _published_links(docs_dir: pathlib.Path, nav_paths: set[str]) -> set[str]:
    """Every in-site page some published page links to, as docs-relative posix paths.

    Only pages that are themselves in the nav are read: a link from an excluded,
    agent-facing note is not a way in for a reader.
    """
    found: set[str] = set()
    for rel in sorted(nav_paths):
        page = docs_dir / rel
        if not page.exists():  # generated at build time by mkdocs-gen-files
            continue
        text = page.read_text(encoding="utf-8-sig")
        for target in re.findall(r"]\(([^)]+)\)", text):
            target = target.split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (page.parent / target).resolve()
            try:
                found.add(resolved.relative_to(docs_dir.resolve()).as_posix())
            except ValueError:  # outside docs/ — a repo file, not a site page
                pass
    return found


def _unreachable_sections(
    sections: dict[str, list[str]], linked: set[str], exempt: set[str]
) -> list[str]:
    """Nav sections no published page links into."""
    return [
        title
        for title, members in sections.items()
        if title not in exempt and not (set(members) & linked)
    ]


# The landing page is what the site opens on, so nothing needs to link to it.
_NAV_LINK_EXEMPT = {"Home"}


def test_every_nav_section_is_linked_from_a_published_page():
    """A nav section nothing links to is invisible to anyone not reading the sidebar.

    The class: **a page declared in the nav and reachable only from the nav.** It
    builds, ``exclude_docs`` does not hide it, every link on it resolves, and
    ``mkdocs build --strict`` is green — so the check above passes while the page
    stays unreachable for a reader arriving at the README or the landing page, and
    for anyone reading the docs on GitHub, where there is no sidebar at all.

    How it hid: reachability is a property of the *other* pages, so nothing is wrong
    on the page itself. The whole five-page **Workflows** section sat this way from
    the day it was written — the only reference to it anywhere in the tree was in
    ``docs/docs_design.md``, which ``exclude_docs`` keeps off the site — and so did
    **Known issues**, the page ``docs/bug_reports.md`` tells a reporter to read first.

    The rule: a top-level nav section is linked from the prose of at least one
    published page. ``Home`` is exempt; it is where the reader lands.
    """
    config = _config()
    docs_dir = REPO / config["docs_dir"]
    nav_paths = _nav_paths(config["nav"])
    linked = _published_links(docs_dir, nav_paths)

    offenders = _unreachable_sections(
        _nav_sections(config["nav"]), linked, _NAV_LINK_EXEMPT
    )

    assert not offenders, (
        "no published page links into these mkdocs.yml nav sections, so they are "
        "reachable only from the site sidebar — invisible from the README, the "
        "landing page, and GitHub. Link each from the prose of a page that leads "
        "to it:\n" + "\n".join(offenders)
    )


def test_the_section_link_scan_would_catch_an_unreachable_one():
    """Guard the guard: the check must not pass vacuously."""
    sections = {"Home": ["index.md"], "Guides": ["gui.md"], "Workflows": ["w/one.md"]}

    assert _unreachable_sections(sections, {"gui.md"}, _NAV_LINK_EXEMPT) == [
        "Workflows"
    ]
    # A section counts as reached when any one of its pages is linked.
    assert (
        _unreachable_sections(sections, {"gui.md", "w/one.md"}, _NAV_LINK_EXEMPT) == []
    )
