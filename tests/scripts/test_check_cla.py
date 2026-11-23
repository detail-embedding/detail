"""A CLA signature counts only when the pull request's author added it for themselves.

**The class: a signature row that proves nothing.** The signature file is plain text in the
repository, so any pull request can type a row. Looking for the author's username alone passes a
row someone else added for them, a row an outsider added for a third person, and an edit to a
row that was already recorded. ``scripts/check_cla.py`` refuses each; each is pinned here:

- **Signing for someone else.** ``test_an_outsider_cannot_add_someone_elses_row``.
- **Rewriting a recorded signature.** ``test_an_outsider_cannot_change_or_remove_an_existing_row``.
- **A row committed by another account.** ``test_the_row_must_be_committed_by_the_author``,
  ``test_a_commit_linked_to_no_account_is_refused``.
- **Failing open when the commits cannot be read.**
  ``test_no_commit_found_for_a_new_row_is_refused``.
- **The workflow not running the rules at all** (the script left out of the sparse checkout, an
  argument dropped). ``test_the_workflow_runs_the_script_with_every_input``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import check_cla

REPO = Path(__file__).resolve().parents[2]

_TABLE = """# CLA signatures

| GitHub username | Full name | Date (YYYY-MM-DD) | CLA version |
|---|---|---|---|
{rows}
"""

_ALICE = "| alice | Alice A | 2026-11-01 | 1.0 |"
_BOB = "| bob | Bob B | 2026-11-02 | 1.0 |"
_CAROL = "| carol | Carol C | 2026-11-03 | 1.0 |"


def _rows(*lines: str) -> dict[str, str]:
    return check_cla.parse_rows(_TABLE.format(rows="\n".join(lines)))


def _problems(
    author: str,
    base: tuple[str, ...],
    head: tuple[str, ...],
    commit_authors: tuple[str, ...] = (),
    association: str = "CONTRIBUTOR",
) -> list[str]:
    return check_cla.problems(
        check_cla.PullRequest(
            author=author,
            association=association,
            base_rows=_rows(*base),
            head_rows=_rows(*head),
            commit_authors=commit_authors,
        )
    )


def test_an_outsider_adding_their_own_row_in_their_own_commit_passes() -> None:
    assert _problems("carol", (_ALICE,), (_ALICE, _CAROL), ("carol",)) == []


def test_a_signer_from_an_earlier_pull_request_passes_without_touching_the_file() -> (
    None
):
    assert _problems("carol", (_ALICE, _CAROL), (_ALICE, _CAROL)) == []


def test_an_author_with_no_row_fails_and_is_named() -> None:
    found = _problems("carol", (_ALICE,), (_ALICE,))
    assert len(found) == 1
    assert "No CLA signature for GitHub user 'carol'" in found[0]


def test_usernames_match_regardless_of_case() -> None:
    assert _problems("Carol", (_ALICE,), (_ALICE, _CAROL), ("carol",)) == []


def test_a_bot_needs_no_row() -> None:
    assert _problems("dependabot[bot]", (_ALICE,), (_ALICE,)) == []


def test_an_outsider_cannot_add_someone_elses_row() -> None:
    found = _problems("carol", (_ALICE,), (_ALICE, _BOB, _CAROL), ("carol",))
    assert len(found) == 1
    assert "adds signatures for other people (bob)" in found[0]


def test_an_outsider_cannot_add_only_someone_elses_row() -> None:
    found = _problems("carol", (_ALICE,), (_ALICE, _BOB), ("carol",))
    assert any("No CLA signature for GitHub user 'carol'" in p for p in found)
    assert any("adds signatures for other people (bob)" in p for p in found)


@pytest.mark.parametrize(
    "head",
    [
        ("| alice | Mallory | 2026-11-01 | 1.0 |", _CAROL),
        (_CAROL,),
    ],
    ids=["changed", "removed"],
)
def test_an_outsider_cannot_change_or_remove_an_existing_row(
    head: tuple[str, ...],
) -> None:
    found = _problems("carol", (_ALICE,), head, ("carol",))
    assert len(found) == 1
    assert "changes or removes existing signatures (alice)" in found[0]


def test_the_row_must_be_committed_by_the_author() -> None:
    found = _problems("carol", (_ALICE,), (_ALICE, _CAROL), ("carol", "alice"))
    assert len(found) == 1
    assert "must be added in a commit by 'carol'" in found[0]
    assert "alice" in found[0]


def test_a_commit_linked_to_no_account_is_refused() -> None:
    found = _problems("carol", (_ALICE,), (_ALICE, _CAROL), ("-",))
    assert len(found) == 1
    assert "an email linked to no GitHub account" in found[0]


def test_no_commit_found_for_a_new_row_is_refused() -> None:
    found = _problems("carol", (_ALICE,), (_ALICE, _CAROL), ())
    assert len(found) == 1
    assert "Could not find the commit" in found[0]


@pytest.mark.parametrize("association", ["OWNER", "MEMBER"])
def test_a_member_may_edit_any_row(association: str) -> None:
    found = _problems(
        "alice",
        ("| alice | [full name] | [date] | 1.0 |", _BOB),
        (_ALICE, "| bob | Bob B | 2026-11-09 | 1.0 |", _CAROL),
        ("bob",),
        association=association,
    )
    assert found == []


def test_a_member_still_needs_their_own_row() -> None:
    found = _problems("alice", (_BOB,), (_BOB,), association="MEMBER")
    assert len(found) == 1
    assert "No CLA signature for GitHub user 'alice'" in found[0]


def test_a_username_listed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="(?i)alice is listed more than once"):
        _rows(_ALICE, "| Alice | Alice again | 2026-11-05 | 1.0 |")


def test_the_real_signature_file_parses() -> None:
    rows = check_cla.parse_rows(
        (REPO / check_cla.SIGNATURES).read_text(encoding="utf-8-sig")
    )
    assert {"mlurbur", "drcoffey", "samagolden"} <= set(rows)
    assert "github username" not in rows


def _run(tmp_path: Path, base: str | None, head: str, authors: str, **kw: str) -> int:
    (tmp_path / "head.md").write_text(head, encoding="utf-8")
    if base is not None:
        (tmp_path / "base.md").write_text(base, encoding="utf-8")
    (tmp_path / "authors").write_text(authors, encoding="utf-8")
    return check_cla.main(
        [
            "--head",
            str(tmp_path / "head.md"),
            "--base",
            str(tmp_path / "base.md"),
            "--author",
            kw.get("author", "carol"),
            "--association",
            kw.get("association", "CONTRIBUTOR"),
            "--commit-authors",
            str(tmp_path / "authors"),
        ]
    )


def test_main_passes_and_says_so(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    base = _TABLE.format(rows=_ALICE)
    head = _TABLE.format(rows=f"{_ALICE}\n{_CAROL}")
    assert _run(tmp_path, base, head, "carol\n") == 0
    assert "carol has agreed to the CLA." in capsys.readouterr().out


def test_main_prints_each_problem_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    head = _TABLE.format(rows=f"{_ALICE}\n{_BOB}")
    assert _run(tmp_path, _TABLE.format(rows=_ALICE), head, "carol\n") == 1
    errors = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("::error::")
    ]
    assert len(errors) == 2


def test_main_with_no_base_file_treats_every_row_as_new(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    head = _TABLE.format(rows=f"{_ALICE}\n{_BOB}")
    assert _run(tmp_path, None, head, "alice\n", author="alice") == 1
    assert "adds signatures for other people (bob)" in capsys.readouterr().out
    assert (
        _run(tmp_path, None, head, "alice\n", author="alice", association="MEMBER") == 0
    )


def test_main_refuses_a_duplicate_row(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    head = _TABLE.format(rows=f"{_ALICE}\n{_ALICE}")
    assert (
        _run(tmp_path, None, head, "alice\n", author="alice", association="MEMBER") == 1
    )
    assert "::error::alice is listed more than once" in capsys.readouterr().out


def test_the_workflow_runs_the_script_with_every_input() -> None:
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/cla.yml").read_text(encoding="utf-8-sig")
    )
    steps = workflow["jobs"]["check"]["steps"]
    sparse = steps[0]["with"]["sparse-checkout"].split()
    assert "scripts/check_cla.py" in sparse
    assert check_cla.SIGNATURES in sparse
    run = next(step["run"] for step in steps if "check_cla.py" in step.get("run", ""))
    for flag in ("--head", "--base", "--author", "--association", "--commit-authors"):
        assert flag in run
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "read"}
