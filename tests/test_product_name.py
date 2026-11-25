"""The user-facing docs spell the tool's name Detail, never DETAIL.

**The class: a spelling the paper settled that the docs drift back from.** The name is not an
acronym, and the paper writes it Detail. The docs said DETAIL in 285 places across 25 files
when that was decided, so any page copied from an older one, or written from memory, brings
the capitals back. ``tests/test_docs_voice.py`` cannot hold this rule: its banned-term scan
ignores case, and the rule is only about case.

The scope is what a user reads: every user-facing docs page (the same set
``test_docs_voice.user_pages`` derives), the README, CONTRIBUTING, the issue templates and the
site name. The app's own text and the code are deliberately out of scope for now; they keep
DETAIL until someone decides otherwise.

``test_the_scan_catches_the_capitals_and_nothing_else`` shows the scan fails on the capitals
and passes on everything that merely contains them.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_docs_voice import user_pages

REPO = Path(__file__).resolve().parents[1]

_CAPITALS = re.compile(r"\bDETAIL\b")

#: User-facing files outside ``docs/``.
_OTHER_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    ".github/bug_report_body.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
)


def _hits(text: str) -> list[int]:
    """Line numbers where the name is written in capitals."""
    return [n for n, line in enumerate(text.splitlines(), 1) if _CAPITALS.search(line)]


def _files() -> list[str]:
    return [f"docs/{rel}" for rel in user_pages()] + list(_OTHER_FILES)


def test_the_scan_catches_the_capitals_and_nothing_else() -> None:
    assert _hits("Open DETAIL.\nDETAIL's window") == [1, 2]
    assert _hits("detail, Details, Detail, DETAIL_HOME, detail-embedding") == []


def test_every_file_the_scan_reads_exists() -> None:
    missing = [rel for rel in _files() if not (REPO / rel).is_file()]
    assert not missing, f"listed but not on disk: {missing}"


def test_the_user_facing_docs_write_detail_not_detail_in_capitals() -> None:
    offenders = {
        rel: lines
        for rel in _files()
        if (lines := _hits((REPO / rel).read_text(encoding="utf-8-sig")))
    }
    assert not offenders, (
        "the name is Detail, not an acronym (docs/docs_design.md §5); "
        f"DETAIL found at {offenders}"
    )


def test_the_site_name_is_detail() -> None:
    config = (REPO / "mkdocs.yml").read_text(encoding="utf-8-sig")
    match = re.search(r"^site_name:\s*(.+?)\s*$", config, re.MULTILINE)
    assert match, "mkdocs.yml has no site_name"
    assert match.group(1) == "Detail"
