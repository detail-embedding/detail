"""Check that a pull request's author has agreed to the DETAIL CLA, and that nobody signed for them.

Run by `.github/workflows/cla.yml` on every pull request. Agreement is a row with the author's
GitHub username in `.github/cla-signatures.md` (`CLA.md` §8). Finding that row is not enough on
its own: a row anyone can type proves nothing. So, for an author outside the organization:

- the pull request may add only its author's own row, never someone else's;
- it may not change or remove a row that is already on the base branch;
- every commit in it that touches the signature file must be by the author's GitHub account.

Organization members (author association ``OWNER`` or ``MEMBER``) may edit any row: they fill in
the authors' rows and correct typos, and a second maintainer reviews their pull requests anyway.

The commit check reads the GitHub account a commit is linked to through its email. A commit can
claim someone else's email, so it is a check against honest mistakes (a maintainer adding the
line for a contributor), not against forgery; the forgery guard is that the row must name the
account that opened the pull request.

Usage (the workflow's form):

    python3 scripts/check_cla.py --head HEAD_FILE --base BASE_FILE --author LOGIN
        --association ASSOC --commit-authors FILE

``--base`` may name a missing file (the branch the file is first added on). ``--commit-authors``
holds one line per pull-request commit that touches the signature file: the GitHub login it is
linked to, or ``-`` when its email is linked to no account.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

SIGNATURES = ".github/cla-signatures.md"

#: Accounts that never sign: they open pull requests on the project's behalf.
BOTS = frozenset({"dependabot[bot]", "github-actions[bot]", "pre-commit-ci[bot]"})

#: Author associations that may edit any row (GitHub's `author_association`).
TRUSTED = frozenset({"OWNER", "MEMBER"})

_HEADER = "github username"

_UNLINKED = "(an email linked to no GitHub account)"


@dataclass(frozen=True)
class PullRequest:
    """What the check needs to know about one pull request."""

    author: str
    association: str
    base_rows: dict[str, str]
    head_rows: dict[str, str]
    commit_authors: tuple[str, ...]


def parse_rows(text: str) -> dict[str, str]:
    """Map each signer's username (case-folded) to its whole row, cells trimmed.

    Raises:
        ValueError: if a username is listed twice.
    """
    rows: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        username = cells[0]
        if not username or username.lower() == _HEADER or set(username) <= set("-: "):
            continue
        key = username.casefold()
        if key in rows:
            raise ValueError(f"{username} is listed more than once in {SIGNATURES}.")
        rows[key] = " | ".join(cells)
    return rows


def problems(pr: PullRequest) -> list[str]:
    """Return what is wrong with this pull request's CLA signature; empty when nothing is."""
    author = pr.author.casefold()
    if author in BOTS:
        return []
    found: list[str] = []
    if author not in pr.head_rows:
        found.append(
            f"No CLA signature for GitHub user '{pr.author}'. Read CLA.md, then add a line "
            f"for '{pr.author}' to {SIGNATURES} in this pull request, in a commit you make "
            "yourself (CONTRIBUTING.md, 'Sign the contributor license agreement')."
        )
    if pr.association.upper() in TRUSTED:
        return found

    changed = sorted(
        user
        for user, row in pr.base_rows.items()
        if pr.head_rows.get(user) != row
    )
    if changed:
        found.append(
            f"This pull request changes or removes existing signatures ({', '.join(changed)}). "
            "Only the project's maintainers can change a signature that is already recorded."
        )
    added = set(pr.head_rows) - set(pr.base_rows)
    others = sorted(added - {author})
    if others:
        found.append(
            f"This pull request adds signatures for other people ({', '.join(others)}). "
            "Each person adds their own, in their own pull request."
        )
    if author in added:
        strangers = sorted(
            {
                _UNLINKED if login == "-" else login
                for login in pr.commit_authors
                if login.casefold() != author
            }
        )
        if not pr.commit_authors:
            found.append(
                f"Could not find the commit that adds the signature for '{pr.author}'. "
                "Re-run this check; if it fails again, ask a maintainer."
            )
        elif strangers:
            found.append(
                f"The signature for '{pr.author}' must be added in a commit by '{pr.author}'s "
                f"GitHub account, but {SIGNATURES} is changed by commits from: "
                f"{', '.join(strangers)}. Commit it yourself, with an email linked to your "
                "GitHub account."
            )
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the check from the workflow; print each problem as a GitHub error annotation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--association", required=True)
    parser.add_argument("--commit-authors", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.head.is_file():
        print(f"::error::{SIGNATURES} is missing from this branch. Merge main into it.")
        return 1
    try:
        head_rows = parse_rows(args.head.read_text(encoding="utf-8-sig"))
        base_text = (
            args.base.read_text(encoding="utf-8-sig") if args.base.is_file() else ""
        )
        base_rows = parse_rows(base_text)
    except ValueError as error:
        print(f"::error::{error}")
        return 1
    commit_authors = tuple(
        args.commit_authors.read_text(encoding="utf-8-sig").splitlines()
    )
    found = problems(
        PullRequest(
            author=args.author,
            association=args.association,
            base_rows=base_rows,
            head_rows=head_rows,
            commit_authors=commit_authors,
        )
    )
    for problem in found:
        print(f"::error::{problem}")
    if not found:
        print(f"{args.author} has agreed to the CLA.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
