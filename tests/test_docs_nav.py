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


DOCS_DESIGN = REPO / "docs" / "docs_design.md"
_SECTION_TABLE_HEADER = "| Section | What goes in it |"


def _defined_sections(design: str) -> list[str]:
    """The section names in the §4 table of ``docs/docs_design.md``, in order."""
    lines = design.splitlines()
    starts = [
        i for i, line in enumerate(lines) if line.startswith(_SECTION_TABLE_HEADER)
    ]
    if len(starts) != 1:
        raise ValueError(
            f"expected one table starting {_SECTION_TABLE_HEADER!r} in "
            f"docs/docs_design.md, found {len(starts)}"
        )
    names = []
    for line in lines[starts[0] + 2 :]:  # skip the header and its rule
        if not line.startswith("|"):
            break
        names.append(line.split("|")[1].strip())
    return names


def _section_mismatch(nav_names: list[str], defined: list[str]) -> list[str]:
    """What differs between the nav's top level and the defined sections."""
    problems = [
        f"nav section {name!r} has no row in the docs_design.md §4 table"
        for name in nav_names
        if name not in defined
    ]
    problems += [
        f"docs_design.md §4 defines {name!r}, which is not a nav section"
        for name in defined
        if name not in nav_names
    ]
    if not problems and nav_names != defined:
        problems.append(f"order differs: nav {nav_names}, table {defined}")
    return problems


def test_every_nav_section_is_one_the_design_defines():
    """The nav's top level is exactly the sections ``docs_design.md`` §4 defines.

    The class: **a docs section with no written rule for what belongs in it.** A section
    is added, or grows, by whoever needs somewhere to put a page; with no definition, the
    next page about the same task lands in a different section.

    How it hid: each page was fine alone. The site ended up with three sections that
    explained how to do things (the Workflows task pages, the "Guides" to each surface,
    and the user guides), three of five Workflows pages repeating two user guide chapters,
    and one of them naming a screen the app had since renamed. Nothing failed, because no
    check compares one page with another.

    The rule: the table in ``docs/docs_design.md`` §4 lists every top-level section with
    what goes in it and the test a page must pass. A section is defined there before it
    is in the nav, and a task a user does in the app is a step in a user guide chapter,
    never a page of its own.
    """
    nav_names = list(_nav_sections(_config()["nav"]))
    defined = _defined_sections(DOCS_DESIGN.read_text(encoding="utf-8-sig"))

    problems = _section_mismatch(nav_names, defined)

    assert not problems, (
        "mkdocs.yml's top-level nav and the section table in docs/docs_design.md §4 "
        "disagree. Define a section in the table (what goes in it, the test for a "
        "page) before adding it to the nav; a task belongs in a user guide chapter, "
        "not in a new section:\n" + "\n".join(problems)
    )


def test_the_section_check_would_catch_an_undefined_or_missing_section():
    """Guard the guard: the comparison must not pass vacuously."""
    defined = ["Home", "User guides", "Manual"]

    assert _section_mismatch(["Home", "User guides", "Manual"], defined) == []
    assert _section_mismatch(
        ["Home", "User guides", "Manual", "Workflows"], defined
    ) == ["nav section 'Workflows' has no row in the docs_design.md §4 table"]
    assert _section_mismatch(["Home", "User guides"], defined) == [
        "docs_design.md §4 defines 'Manual', which is not a nav section"
    ]
    assert _section_mismatch(["User guides", "Home", "Manual"], defined)
    # The table parser reads real rows and stops at the table's end.
    table = "\n".join(
        [
            _SECTION_TABLE_HEADER + " x |",
            "| --- | --- | --- |",
            "| Home | a | b |",
            "",
            "| Not | a | row |",
        ]
    )
    assert _defined_sections(table) == ["Home"]
