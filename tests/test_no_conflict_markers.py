"""No tracked text file carries an unresolved merge-conflict marker.

**The class this exists for.** A conflict marker is committed content that reads as prose. Nothing
imports it, nothing parses it, no test constructs it — so it survives review, survives CI, and
survives every later edit to the same file, because the next author is looking at their own hunk.

The instance: ``<<<<<<< HEAD`` / ``=======`` / ``>>>>>>> origin/main`` sat in the middle of
``CLAUDE.md``'s recorded-classes table for **four commits**, having arrived with a squash merge.
``CLAUDE.md`` is the file every agent loads at the start of every session, so the damage is not one
broken page — it is that the rulebook contained a torn seam and the two halves of the table read as
one list. Two of the four intervening commits edited that very file, mine among them.

**And the table already had a guard.** ``tests/test_recorded_classes.py`` walks the same rows and
checks that every cited ``tests/...py`` path exists. It passed throughout, because a conflict marker
is not a table row — it does not match the row pattern, so the sweep steps over it. That is the
recorded shape *assert the property, not a proxy for it*: "every citation resolves" is a proxy for
"the table is intact", and the two come apart exactly here.

Repo-wide rather than scoped to ``CLAUDE.md``, because the file it happened in is not the property.
A marker in a source file usually breaks a parse and is found in seconds; in Markdown, YAML comments
or a docstring it is silent, and those are the files a merge is most likely to conflict in.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Anchored at line start, which is where git writes them. A line *about* conflict markers — this
#: file's own docstring, a doc explaining a merge — indents or quotes them and is not matched.
MARKER = re.compile(r"^(<{7}|={7}|>{7})(\s|$)", re.MULTILINE)

#: Extensions where a marker is silent: it neither breaks a parse nor shows up as a syntax error.
#: Source files are scanned too, but these are the ones the class actually lives in.
TEXT_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".cfg", ".txt", ".json"}


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, encoding="utf-8", cwd=REPO
    ).stdout
    return [REPO / name for name in out.split("\0") if name]


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _scannable() -> list[Path]:
    return [
        path
        for path in _tracked_files()
        if path.suffix in TEXT_SUFFIXES
        and path.is_file()
        and path != Path(__file__).resolve()  # this file names the markers on purpose
    ]


@needs_git
def test_there_are_files_to_scan():
    """Guard the guard: a listing that returns nothing passes the sweep below forever."""
    found = _scannable()
    assert (
        len(found) > 300
    ), f"only {len(found)} tracked text files — the listing has drifted"


@needs_git
def test_no_tracked_file_has_a_conflict_marker():
    offenders = []
    for path in _scannable():
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue  # not text after all
        for match in MARKER.finditer(text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(REPO)}:{line}: {match.group(1)}")
    assert not offenders, (
        "unresolved merge-conflict markers are committed. Resolve the conflict — usually both "
        "halves are wanted and only the three marker lines go:\n  "
        + "\n  ".join(offenders)
    )


@needs_git
def test_a_planted_marker_would_be_caught():
    """The negative case, so the sweep cannot pass vacuously."""
    assert MARKER.search("<<<<<<< HEAD\n")
    assert MARKER.search("a\n=======\nb\n")
    assert MARKER.search(">>>>>>> origin/main\n")
    # ...and prose *about* a marker does not trip it
    assert not MARKER.search("the line `<<<<<<< HEAD` is what git writes\n")
    assert not MARKER.search("    ======= indented, so it is a rule, not a conflict\n")
