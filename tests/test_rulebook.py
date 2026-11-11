"""`CLAUDE.md` stays a rulebook plus a router, and the router resolves.

**The class: an always-loaded file that grows back.** `CLAUDE.md` is read into every session, so
every line in it is paid for on every task — and adherence falls as it grows, which is a cost with
no error message. Measured over its history: **143 commits touched it, 122 of them grew it, and two
ever shrank it.** It went from 263 lines to 1,751 in three and a half months, accelerating: +201
lines in July, +356 in August, +630 in the first twelve days of September. Nothing objected, because
a long file reads exactly like a short one from inside a session.

Two rows of `docs/recorded_classes.md` are that growth's own failures — rules written into this file,
believed, quoted in review, and true of nothing in the package. That is what an unchecked prose copy
becomes.

**The rule.** Procedures live in `.claude/skills/`, reference material in `docs/`, and `CLAUDE.md`
holds the invariants plus one line per topic saying where the rest is. A ceiling is the only check
that cannot be argued with, and the router has to resolve or the split has simply hidden things.

**Why a line count rather than a content check.** The failure is *volume*: any individual paragraph
is defensible, and there is no wording to match. The ceiling is a ratchet — lower it as more moves
out; raising it is the visible act of deciding this file should be longer.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
RULEBOOK = REPO / "CLAUDE.md"
SKILLS = REPO / ".claude" / "skills"

#: A ratchet, like `detail/gui/requirements.py`'s `BACKLOG_CEILING`. Lowered as more moves out of the
#: always-loaded file; raised only as a deliberate, visible decision. The file was 1,751 lines when
#: this was written, and the four changes that brought it here are in the git history. Lowered
#: from 800 when the file was rewritten to be the public copy as well: the history behind each
#: rule moved to the guard that enforces it and to `docs/recorded_classes.md`.
LINE_CEILING = 420

#: Skill names the router is allowed not to mention, with the reason. An entry that excuses nothing
#: fails, the standard every allowlist in this repo is held to.
ROUTER_EXEMPT: dict[str, str] = {}

_SKILL_REF = re.compile(r"`([a-z][a-z0-9-]*)`")


def _router() -> str:
    """The *Start here* section, which carries the router table."""
    lines = RULEBOOK.read_text(encoding="utf-8-sig").split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## Start here"))
    end = next(j for j in range(start + 1, len(lines)) if lines[j].startswith("## "))
    return "\n".join(lines[start:end])


def _skill_dirs() -> list[pathlib.Path]:
    return sorted(p for p in SKILLS.iterdir() if p.is_dir())


def test_the_rulebook_stays_under_its_ceiling() -> None:
    lines = RULEBOOK.read_text(encoding="utf-8-sig").split("\n")
    assert len(lines) <= LINE_CEILING, (
        f"CLAUDE.md is {len(lines)} lines, past the {LINE_CEILING}-line ceiling. It is loaded into "
        "every session, so this is paid on every task. Move a procedure to .claude/skills/ or "
        "reference material to docs/ and leave a router line — or lower the bar deliberately by "
        "raising this number, which is the point of it being here."
    )


def test_every_skill_is_reachable_from_the_router() -> None:
    """A skill nobody is told about is a capability with no way in.

    Its description still loads every session, so an unrouted skill costs context and is found only
    by luck. This is the repo's own recorded class — a capability whose only entry point reached 2 of
    9 modules — applied to the mechanism this split depends on.
    """
    router = _router()
    unrouted = [
        d.name
        for d in _skill_dirs()
        if d.name not in router and d.name not in ROUTER_EXEMPT
    ]
    assert not unrouted, (
        f"these skills are not named in CLAUDE.md's router: {unrouted}. Add a row saying when to "
        "read each, or an allowlist entry here with the reason it is deliberately unrouted."
    )
    stale = [name for name in ROUTER_EXEMPT if not (SKILLS / name).is_dir()]
    assert not stale, f"ROUTER_EXEMPT names skills that no longer exist: {stale}"


def test_every_skill_declares_a_name_and_a_trigger() -> None:
    """Frontmatter is how a skill is chosen, so a missing description is a skill that never fires."""
    broken = []
    for d in _skill_dirs():
        skill = d / "SKILL.md"
        if not skill.is_file():
            broken.append(f"{d.name}: no SKILL.md")
            continue
        text = skill.read_text(encoding="utf-8-sig")
        if not text.startswith("---\n"):
            broken.append(f"{d.name}: no frontmatter")
            continue
        front = text.split("---", 2)[1]
        if f"name: {d.name}" not in front:
            broken.append(f"{d.name}: frontmatter name does not match the directory")
        if "description:" not in front:
            broken.append(f"{d.name}: no description, so it can never be chosen")
        elif "Use when" not in front:
            broken.append(f"{d.name}: description does not say when to use it")
    assert not broken, "skills with unusable frontmatter:\n  " + "\n  ".join(broken)


def test_every_path_the_router_names_exists() -> None:
    """A router pointer into a tree that moves, checked by nobody — one level up from the citations.

    Nothing else checks these. `mkdocs build --strict` only follows links between *published* pages,
    and the internal docs this router points at are all in `exclude_docs`.
    """
    router = _router()
    paths = set(re.findall(r"`((?:docs|research|detail|tests|scripts)/[^`]+)`", router))
    assert (
        paths
    ), "the router names no repository paths; the reader is broken, not the router"
    missing = sorted(p for p in paths if not (REPO / p.split("#")[0]).exists())
    assert not missing, (
        f"CLAUDE.md's router points at paths that do not exist: {missing}. A pointer is the only "
        "route from the rulebook to the thing it moved out, and a stale one reads exactly like a "
        "live one."
    )


def test_the_ceiling_would_catch_a_regrown_file() -> None:
    """Guard the guard: the check must reject a file past the ceiling."""
    assert LINE_CEILING < 1751, (
        "the ceiling is at or above the size CLAUDE.md had before it was split, so it would have "
        "passed against the very state it exists to prevent"
    )
    grown = ["a line"] * (LINE_CEILING + 1)
    assert len(grown) > LINE_CEILING, "the comparison itself is inverted"


def test_the_router_check_would_catch_an_unrouted_skill() -> None:
    """Guard the guard: a skill absent from the router must be reported."""
    router = _router()
    invented = "a-skill-nobody-mentioned"
    assert invented not in router, "pick a name the router really does not contain"
    unrouted = [
        name for name in (invented,) if name not in router and name not in ROUTER_EXEMPT
    ]
    assert unrouted == [invented], (
        "the router check did not flag a skill the router never mentions, so it would not flag a "
        "real one either"
    )


#: Flags the rulebook's own commands carry -> the distribution that supplies them. A flag
#: added to those commands without its plugin in the `dev` extra reads identically and fails
#: on the next clone, so the mapping is the claim this pair of tests checks.
_FLAG_PLUGINS = {"-n auto": "pytest-xdist", "--cov": "pytest-cov"}


def _dev_extra() -> str:
    """The `dev = [...]` block of `pyproject.toml`, comments and all."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8-sig")
    start = text.index("dev = [")
    return text[start : text.index("]", start)]


def test_every_flag_the_rulebook_documents_is_installed_by_the_dev_extra() -> None:
    """A documented command has to work from `pip install -e ".[dev]"` and nothing else.

    **The class: a command that only runs on the machine it was written on.** `-n` entered
    CLAUDE.md while `pytest-xdist` was installed by hand to measure whether parallelism was
    worth adopting (#532). Had it stayed hand-installed, the rulebook would have read exactly
    the same and the next clone would have met `unrecognized arguments: -n`. The failure has
    no symptom on the box that wrote it, which is the whole reason it needs a check rather
    than a habit.
    """
    rulebook = RULEBOOK.read_text(encoding="utf-8-sig")
    extra = _dev_extra()
    missing = {
        flag: plugin
        for flag, plugin in _FLAG_PLUGINS.items()
        if flag in rulebook and plugin not in extra
    }
    assert not missing, (
        "CLAUDE.md documents commands whose plugins a fresh clone would not have: "
        + ", ".join(
            f"{flag!r} needs {plugin}" for flag, plugin in sorted(missing.items())
        )
        + ". Add them to the `dev` extra in pyproject.toml, or stop documenting the flag."
    )


def test_the_flag_check_would_catch_a_plugin_missing_from_the_extra() -> None:
    """Guard the guard: a documented flag whose plugin is absent must be reported."""
    extra = _dev_extra()
    invented = "pytest-a-plugin-nobody-depends-on"
    assert invented not in extra, "pick a distribution the extra really does not name"
    missing = {
        flag: plugin
        for flag, plugin in {"-n auto": invented}.items()
        if flag in RULEBOOK.read_text(encoding="utf-8-sig") and plugin not in extra
    }
    assert missing == {"-n auto": invented}, (
        "the check did not flag a documented flag whose plugin is missing from the dev "
        "extra, so it would not flag a real one either"
    )
