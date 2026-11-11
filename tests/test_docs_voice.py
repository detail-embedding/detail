"""The user-facing docs are held to the house voice in ``docs/docs_design.md`` §5.

The class: **a style rule that is written down, quoted in review, and enforced by
nothing.** §5 has banned the em dash since it was written, and gave the reason: it "reads
as agent-written". §5 also states, in the present tense, that "the CI word check (§9)
enforces the banned column" — while §9 lists that check as *(planned)* and no such test
existed. Measured before this file: **430 em dashes across 306 lines** of user-facing
prose, 227 of them the appositive form the rule names.

How it hid: nothing in the suite read the docs as prose. ``mkdocs build --strict``
resolves links, ``test_docs_nav`` checks declaration, ``test_docs_map`` checks anchors.
Voice is a property of the sentences, so every existing check passes over a page written
in a register the style guide forbids.

The rule, in two halves:

- **A banned term does not appear in a user-facing page.** Fail-closed, and the list is
  parsed out of §5's own table so adding a row arms the check rather than needing a second
  copy here.
- **Em dashes only decrease.** A ratchet rather than a flat ban, because 386 remain in two
  large pages and a flat ban would have to be introduced with an allowlist of 386 entries,
  which is not a rule anybody reads. ``_EM_DASH_CEILING`` is the whole inventory; a file
  absent from it must be at zero, so a *new* page starts clean and cannot quietly add one.
  Lower a number as prose is rewritten. Raising one is the thing to argue about in review.

The page set is derived, not listed: the nav, plus the ``*-intro.md`` files whose prose is
inlined into generated pages. A page added tomorrow is covered the day it is added.
"""

import pathlib
import re

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
MKDOCS_YML = REPO / "mkdocs.yml"
DOCS = REPO / "docs"
STYLE_GUIDE = DOCS / "docs_design.md"

EM_DASH = "—"

#: Every user-facing page that still carries em dashes, and how many. Only ever lower a
#: number. A file not listed here is checked at zero.
#:
#: **It is empty, and the intent is that it stays empty.** The ratchet landed holding 386
#: across eight pages and they were cleared in the same branch, so this is now a flat ban
#: wearing a ratchet's clothes. Kept as a table rather than deleted because the next page
#: to arrive with prose carried over from an internal spec will need one row for one
#: commit, and an entry with a number beats a guard somebody turns off.
_EM_DASH_CEILING: dict[str, int] = {}

#: Rows of §5's table this check does not enforce yet, each with why. An entry here is a
#: decision that is still open, not a term that is allowed: both need a call from a person
#: before a sweep, which is exactly what an unexplained skip would hide.
_NOT_ENFORCED = {
    "eval": (
        "the module is *named* 'Behavior-query eval' on every surface, so the 12 hits are "
        "mostly a UI string rather than prose. Enforcing the row means renaming the module "
        "or exempting its name, which is a decision, not a sweep."
    ),
    "scattering the literal name in prose": (
        "23 literals. The row calls the product name provisional and asks that a rename be "
        "one change; doing that needs the rename to be real first."
    ),
    "(calling it a recording)": "a parenthetical gloss, not a term a scan can match.",
    "(used interchangeably)": "a parenthetical gloss, not a term a scan can match.",
    EM_DASH: "has its own ratchet above, which counts rather than forbids.",
}


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores mkdocs' custom tags (``!ENV``, ``!!python/...``)."""


_TolerantLoader.add_multi_constructor(None, lambda loader, suffix, node: None)


def _nav_paths(nav: object) -> set[str]:
    if isinstance(nav, str):
        return {nav}
    if isinstance(nav, list):
        return set().union(*(_nav_paths(item) for item in nav))
    if isinstance(nav, dict):
        return set().union(*(_nav_paths(value) for value in nav.values()))
    return set()


def user_pages() -> list[str]:
    """Every user-facing markdown file on disk, docs-relative.

    The nav, plus the intros: those are in ``exclude_docs`` as standalone pages, but
    ``scripts/gen_docs_*.py`` inlines their prose into pages the site publishes, so they
    are read by users and belong under the same rules.
    """
    config = yaml.load(
        MKDOCS_YML.read_text(encoding="utf-8-sig"), Loader=_TolerantLoader
    )
    nav = {rel for rel in _nav_paths(config["nav"]) if (DOCS / rel).exists()}
    intros = {p.relative_to(DOCS).as_posix() for p in DOCS.glob("*-intro.md")}
    intros |= {
        p.relative_to(DOCS).as_posix()
        for p in (DOCS / "modules" / "_intros").glob("*.md")
    }
    return sorted(nav | intros)


def banned_terms() -> dict[str, str]:
    """``{term: why}`` from §5's word-list table, minus the rows spelled out above."""
    table = re.search(
        r"\| Use \| Not \| Why \|\n\|[-| ]+\|\n((?:\|.*\n)+)",
        STYLE_GUIDE.read_text(encoding="utf-8-sig"),
    )
    assert table, "docs_design.md §5's word-list table did not parse"
    terms = {}
    for row in table.group(1).strip().splitlines():
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) < 3:
            continue
        term = cells[1].strip('"').strip("`")
        if term and term not in _NOT_ENFORCED:
            terms[term] = cells[2]
    return terms


def _offenders(pages: dict[str, str], term: str) -> list[str]:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    return [rel for rel, text in pages.items() if pattern.search(text)]


def _over_ceiling(pages: dict[str, str], ceiling: dict[str, int]) -> dict[str, tuple]:
    return {
        rel: (text.count(EM_DASH), ceiling.get(rel, 0))
        for rel, text in pages.items()
        if text.count(EM_DASH) > ceiling.get(rel, 0)
    }


def _pages() -> dict[str, str]:
    return {rel: (DOCS / rel).read_text(encoding="utf-8-sig") for rel in user_pages()}


def test_no_banned_term_appears_in_a_user_facing_page():
    pages = _pages()
    terms = banned_terms()

    # Guard the guard: a table that stopped parsing would pass over zero terms.
    assert (
        len(terms) >= 6
    ), f"only {len(terms)} enforceable rows parsed: {sorted(terms)}"

    found = {term: _offenders(pages, term) for term in terms}
    hits = {term: files for term, files in found.items() if files}

    assert not hits, (
        "docs/docs_design.md §5's word-list bans these, and they are in user-facing "
        "pages:\n"
        + "\n".join(
            f"  {term!r} ({terms[term]}): {', '.join(f)}" for term, f in hits.items()
        )
    )


def test_em_dashes_only_decrease():
    pages = _pages()

    over = _over_ceiling(pages, _EM_DASH_CEILING)

    assert not over, (
        "docs/docs_design.md §5 bans the em dash in user-facing prose, because it reads "
        "as agent-written. These pages carry more than _EM_DASH_CEILING allows (found, "
        "allowed):\n"
        + "\n".join(
            f"  {rel}: {found} > {allowed}" for rel, (found, allowed) in over.items()
        )
        + "\nUse a comma, a colon, parentheses, or two sentences. Lower the ceiling as you "
        "go; raising one needs a reason in review."
    )


def test_the_ceiling_has_no_entry_that_excuses_nothing():
    """An allowlist entry that is no longer needed is one nobody re-checks.

    The repo holds ``FILE_TIER``, ``UNIT_EXEMPTIONS`` and ``requirements._CONFIRMATIONS``
    to this standard. A ceiling left above the real count would let the next edit put em
    dashes back without failing anything.
    """
    pages = _pages()

    slack = {
        rel: (pages[rel].count(EM_DASH), allowed)
        for rel, allowed in _EM_DASH_CEILING.items()
        if rel in pages and pages[rel].count(EM_DASH) < allowed
    }

    assert not slack, (
        "these _EM_DASH_CEILING entries are above the real count, so they allow em dashes "
        "to come back silently. Lower each to what the file now has (found, allowed):\n"
        + "\n".join(
            f"  {rel}: {found} < {allowed}" for rel, (found, allowed) in slack.items()
        )
    )


def test_the_ceiling_lists_no_page_that_does_not_exist():
    pages = set(user_pages())

    stale = sorted(rel for rel in _EM_DASH_CEILING if rel not in pages)

    assert not stale, f"_EM_DASH_CEILING names pages that are not user-facing: {stale}"


def test_the_voice_scans_would_catch_one():
    """Guard the guard, on the spellings a person actually writes.

    Eleven of this repo's source-scanning guards shipped with no negative case, including
    the one ``CLAUDE.md`` calls "the idiom". Each case below is a way the check could pass
    while the prose is wrong.
    """
    # An em dash anywhere in a file with no ceiling entry.
    assert _over_ceiling({"new-page.md": f"a {EM_DASH} b"}, {}) == {
        "new-page.md": (1, 0)
    }
    # One over an existing ceiling, not just the first one.
    assert _over_ceiling({"gui.md": EM_DASH * 5}, {"gui.md": 4}) == {"gui.md": (5, 4)}
    # At the ceiling exactly is fine; under it is caught by the slack test, not this one.
    assert _over_ceiling({"gui.md": EM_DASH * 4}, {"gui.md": 4}) == {}

    # A banned term in running prose, inside a link, and capitalised.
    assert _offenders({"a.md": "this is a classic footgun here"}, "footgun") == ["a.md"]
    assert _offenders({"b.md": "[the footgun](x.md) explains"}, "footgun") == ["b.md"]
    assert _offenders({"c.md": "Footgun, at the start"}, "footgun") == ["c.md"]
    assert _offenders({"d.md": "nothing to see"}, "footgun") == []


def test_the_style_guide_does_not_claim_a_check_it_lacks():
    """§5 said the word check "enforces" the banned column while §9 called it planned.

    A rule believed and quoted in review, true of nothing in the tree, is this repo's own
    recorded class (window geometry, sortable headers). This file is the check; the prose
    has to agree with which half of it exists.
    """
    text = STYLE_GUIDE.read_text(encoding="utf-8-sig")

    section_9 = text[text.index("## 9.") :]
    entry = re.search(r"\*\*(?:Voice|Word) check\*\*.*", section_9)

    assert entry, "docs_design.md §9 no longer lists the voice check at all"
    assert "*(planned)*" not in entry.group(0), (
        "§9 still calls the voice check planned, but tests/test_docs_voice.py runs it. "
        "Say what is enforced (the banned column, and an em-dash ratchet) and what is not."
    )
    assert "test_docs_voice" in entry.group(0), (
        "§9's voice-check entry should name the file that implements it, so a reader can "
        "see which half is live"
    )
