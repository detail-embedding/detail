"""The pre-commit config formats the way CI checks, and installs the commit stage only.

**The class: one tool pinned in three places.** black is pinned in `pyproject.toml`'s `dev` extra
(what a contributor installs), in `.github/workflows/ci.yml` (what `black --check` runs), and in
`.pre-commit-config.yaml` (what formats a commit). black changes its output between releases, so
if any two drift, a commit is formatted one way and checked another. Nothing fails where the pin
is written; CI turns red on files the author never touched. `19c4b253` ("two files that drifted
while Actions was down") is what format drift already costs when one side is not running.

**And a second class: a hook installed for the wrong event costs every time and protects once.**
The push stage ran the whole suite, and a branch is pushed many times --- on creation, per commit,
after merging `main` in --- and merged once: 64 merges on 2026-09-29 against several times as many
pushes. So it is gone, and `test_only_the_commit_stage_is_installed` keeps it gone, because
`pre-commit install` with the older default hook types would quietly bring it back. The suite
belongs at merge, where git offers no hook at all; the mechanism there is a required status check,
which returns with Actions.

Until then nothing automated runs it, so the remaining risk is the opposite one: a clone where
`pre-commit install` was never run formats nothing and is not told.
`test_the_installed_hook_is_the_one_the_config_asks_for` is that check --- git never writes
`.git/hooks` on clone, deliberately, so no config can do it for you.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent


def _pyproject_pin(text: str) -> str | None:
    match = re.search(r'"black==([^"]+)"', text)
    return match.group(1) if match else None


def _ci_pin(text: str) -> str | None:
    match = re.search(r"pip install black==(\S+)", text)
    return match.group(1) if match else None


def _pre_commit_pin(config: dict) -> str | None:
    for repo in config.get("repos", []):
        if any(hook.get("id") == "black" for hook in repo.get("hooks", [])):
            return str(repo.get("rev"))
    return None


def _config() -> dict:
    return yaml.safe_load(
        (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8-sig")
    )


def test_black_is_pinned_to_one_version_everywhere():
    pins = {
        "pyproject.toml": _pyproject_pin(
            (REPO / "pyproject.toml").read_text(encoding="utf-8-sig")
        ),
        ".github/workflows/ci.yml": _ci_pin(
            (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8-sig")
        ),
        ".pre-commit-config.yaml": _pre_commit_pin(_config()),
    }
    assert (
        None not in pins.values()
    ), f"a pin could not be read, so nothing is compared: {pins}"
    assert len(set(pins.values())) == 1, (
        f"black is pinned differently in different places: {pins}. A commit would be "
        "formatted by one version and checked by another."
    )


def test_the_pin_readers_would_catch_a_drift():
    """Guard the guard: each reader finds a pin in the spelling its file uses, and a mismatch
    is visible to the comparison above."""
    assert _pyproject_pin('    "black==26.5.1",  # pin') == "26.5.1"
    assert _ci_pin("      - run: pip install black==26.1.0\n") == "26.1.0"
    assert (
        _pre_commit_pin({"repos": [{"rev": "25.1.0", "hooks": [{"id": "black"}]}]})
        == "25.1.0"
    )
    assert (
        _pre_commit_pin({"repos": [{"rev": "v6.0.0", "hooks": [{"id": "x"}]}]}) is None
    )


def test_only_the_commit_stage_is_installed():
    """No pre-push hook, deliberately — and a guard so it cannot come back unnoticed.

    A hook that runs the suite on every push gates the wrong event: a branch is pushed many times
    and merged once (2026-09-29: 64 merges, several times as many pushes), so it charges minutes
    repeatedly for one decision, and under `scripts/with_suite_lock.py` those runs queue across
    sessions. The suite belongs at merge, and the mechanism there is a required status check on the
    PR — Actions, currently disabled. Git offers a pre-push hook and no merge hook, so push time was
    never the right moment, only the reachable one.

    Until Actions is back this is a documented discipline (CLAUDE.md "Before you merge", and the
    `ship-a-change` skill) rather than a hook. Re-adding a pre-push stage is a decision, so it should
    fail here first.
    """
    config = _config()
    assert config.get("default_install_hook_types") == ["pre-commit"], (
        "`pre-commit install` should install the commit stage and nothing else; a pre-push hook "
        "gates the wrong event (see this test's docstring)"
    )
    hooks = [hook for repo in config["repos"] for hook in repo.get("hooks", [])]
    push = [hook["id"] for hook in hooks if "pre-push" in hook.get("stages", [])]
    assert not push, f"these hooks still run at pre-push: {push}"


def test_the_installed_hook_is_the_one_the_config_asks_for():
    """The config is inert until someone runs `pre-commit install`, and git cannot do it for you.

    Hooks live in `.git/hooks`, which no clone, fetch or checkout ever writes — deliberately, since a
    repo that could install its own hooks would run code before you read it. So this is one manual
    command per clone, and nothing but a check like this notices when it was not run. Skipped under
    CI, where hooks are neither installed nor wanted.
    """
    if os.environ.get("CI"):
        pytest.skip("CI does not install git hooks")
    # Resolved without shelling out to git, so this file stays in the `unit` tier: a
    # `subprocess.run` here is exactly what `tests/tiers.py`'s signature scan flags, and an
    # exemption for the convenience of one call would be an entry excusing nothing.
    #
    # A plain clone has `.git/` as a directory. A **worktree** has `.git` as a *file* reading
    # `gitdir: <common>/worktrees/<name>`, and hooks live in the common dir, not the per-worktree
    # one — which is why installing once covers every worktree of a clone.
    git = REPO / ".git"
    if git.is_dir():
        hooks = git / "hooks"
    elif git.is_file():
        gitdir = pathlib.Path(
            git.read_text(encoding="utf-8-sig").split("gitdir:", 1)[1].strip()
        )
        hooks = gitdir.parent.parent / "hooks"
    else:
        pytest.skip("not a git checkout")
    installed = hooks / "pre-commit"
    assert installed.is_file(), (
        "the pre-commit hook is not installed in this clone, so nothing checks formatting before a "
        "commit — two formatting breaks reached main on 2026-09-29/30 this way. Run:\n"
        "    pre-commit install\n"
        "It covers every worktree of the clone. If it refuses because `core.hooksPath` is set, run "
        "`git config --unset core.hooksPath` first."
    )
