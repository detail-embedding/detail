"""Which build of detail is running, and how far it trails ``origin/main``.

Read from git at run time rather than kept by hand: a user installs with ``git clone`` +
``pip install -e`` and updates with ``git pull``, so the checkout *is* the build, and a
hand-kept version number would need an edit per PR that dozens of parallel branches would all
collide on. ``__version__`` stays the coarse release name; the commit and its date are what say
whether a copy is current.

Every read is best-effort: outside a git checkout, or with no ``git`` on PATH, the facts come
back ``None`` rather than raising — the app still runs, it just cannot say which commit it is.
Only :func:`check_for_updates` touches the network, and only when a user asks.
"""

from __future__ import annotations

import functools
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_REMOTE = "origin"
UPSTREAM_BRANCH = "main"
_FIELD_SEP = "\x1f"
_LOCAL_TIMEOUT_S = 10
_FETCH_TIMEOUT_S = 60


@dataclass(frozen=True)
class BuildInfo:
    """The running build: the release name plus the commit it was checked out at."""

    version: str
    commit: str | None = None  # short SHA
    date: str | None = None  # commit date, YYYY-MM-DD
    commit_count: int | None = None  # commits reachable from HEAD
    branch: str | None = None
    local_changes: bool = False

    def label(self) -> str:
        """One line for the top bar: ``v0.1.0 · 2026-09-20 · 033eef0``."""
        return self.labels()[0]

    def labels(self) -> list[str]:
        """The label, then shorter forms for less room: the commit goes first, then the
        version, and the date — the "is my copy current?" part — goes last."""
        version = f"v{self.version}"
        if not self.date:
            return [version]
        full = f"{version} · {self.date}"
        if self.commit:
            full += f" · {self.commit}" + ("+" if self.local_changes else "")
        return list(dict.fromkeys([full, f"{version} · {self.date}", self.date]))

    def describe(self) -> str:
        """The full build in a sentence — what the GUI panel and ``detail version`` print."""
        if self.commit is None:
            return f"detail v{self.version} — not a git checkout, so the exact build is unknown."
        where = f" on {self.branch}" if self.branch else ""
        count = f", commit #{self.commit_count}" if self.commit_count else ""
        text = f"detail v{self.version} — {self.commit}{where}, {self.date}{count}"
        if self.local_changes:
            text += " (with local changes)"
        return text


@dataclass(frozen=True)
class HistoryEntry:
    """One commit on the checked-out branch's first-parent line — for main, one merged PR."""

    commit: str
    date: str
    subject: str


@dataclass(frozen=True)
class UpdateStatus:
    """How the checkout compares with ``origin/main`` after a fetch."""

    behind: int
    ahead: int
    upstream_date: str | None
    upstream: str = f"{UPSTREAM_REMOTE}/{UPSTREAM_BRANCH}"

    def describe(self) -> str:
        """What the check found, with the step to take when behind."""
        if self.behind == 0:
            text = f"Up to date with {self.upstream}."
        else:
            commits = "commit" if self.behind == 1 else "commits"
            text = (
                f"{self.behind} {commits} behind {self.upstream}"
                f" (latest {self.upstream_date}). To update: git pull, then reinstall."
            )
        if self.ahead:
            text += f" This checkout also has {self.ahead} commit(s) not on it."
        return text


class UpdateCheckError(RuntimeError):
    """The update check could not reach or read the upstream branch."""


def source_root() -> Path:
    """The directory holding the ``detail`` package's checkout (its parent)."""
    return Path(__file__).resolve().parents[2]


def _git(*args: str, timeout: float = _LOCAL_TIMEOUT_S) -> str | None:
    """``git`` in the source tree; its stdout, or ``None`` if git is absent or it failed."""
    kwargs: dict = {}
    if os.name == "nt":
        # The GUI may run without a console; don't flash one per git call.
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        out = subprocess.run(
            ["git", "-C", str(source_root()), *args],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            # A fetch against a private remote must fail, not wait on a password prompt.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def head_sha() -> str | None:
    """Full SHA of the checked-out commit, or ``None`` outside a git checkout."""
    return _git("rev-parse", "HEAD")


@functools.cache
def build_info() -> BuildInfo:
    """The running build. Several local git calls — run it off the UI thread.

    Cached: the code in this process is what was imported at launch, so a ``git pull`` made
    while the app runs must not relabel it.
    """
    from detail import __version__

    head = _git("log", "-1", f"--format=%h{_FIELD_SEP}%cs")
    if not head:
        return BuildInfo(version=__version__)
    commit, date = head.split(_FIELD_SEP)
    count = _git("rev-list", "--count", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return BuildInfo(
        version=__version__,
        commit=commit,
        date=date,
        commit_count=int(count) if count and count.isdigit() else None,
        branch=None if branch in (None, "HEAD") else branch,
        local_changes=bool(status),
    )


def report() -> dict:
    """This build as plain data: what a cluster prints when asked which DETAIL it has.

    The full SHA, not the short one, because it is compared; the checkout's folder because the
    command that updates it is run there.
    """
    info = build_info()
    return {
        "sha": head_sha(),
        "commit": info.commit,
        "branch": info.branch,
        "date": info.date,
        "local_changes": info.local_changes,
        "root": str(source_root()),
    }


SAME = "same"
BEHIND = "behind"  # the other copy is older than this one
AHEAD = "ahead"  # the other copy has commits this one lacks
DIVERGED = (
    "diverged"  # neither contains the other, or its commit is not one this copy has
)
UNKNOWN = "unknown"  # one side is not a git checkout, or could not be asked


@dataclass(frozen=True)
class BuildComparison:
    """How the DETAIL on another machine (a remote target) compares with the one running here.

    Congruence is of the **commit**: a job imports this app's modules on the cluster, so a cluster
    on any other commit is running code the app did not write, and the failure when it is older
    (a module the app lists that the cluster does not have) lands on every recording at once.
    """

    state: str
    target: str
    host: str
    local_commit: str | None = None
    remote_commit: str | None = None
    local_branch: str | None = None
    remote_branch: str | None = None
    remote_root: str | None = None
    remote_changes: bool = False
    commits: int | None = None  # how far behind or ahead, when that is known
    reason: str = ""  # why it is UNKNOWN

    @property
    def matches(self) -> bool:
        return self.state == SAME

    def update_command(self) -> str | None:
        """The command that brings the other copy up to date, when it is the one that is behind."""
        if self.state != BEHIND or not self.remote_root:
            return None
        return (
            f"ssh {self.host} 'git -C {shlex.quote(self.remote_root)} pull --ff-only'"
        )

    def describe(self) -> str:
        """What the comparison found, with the step to take."""
        target = repr(self.target)
        there = self.remote_commit or "an unknown commit"
        here = self.local_commit or "an unknown commit"
        branches = ""
        if (
            self.remote_branch
            and self.local_branch
            and self.remote_branch != self.local_branch
        ):
            branches = (
                f" It is on {self.remote_branch}; this app is on {self.local_branch}."
            )
        changes = (
            " The checkout there has local changes, so a pull may be refused."
            if self.remote_changes
            else ""
        )
        if self.state == SAME:
            return f"DETAIL on {target} is the same build as this app ({here})."
        if self.state == BEHIND:
            how = "" if self.commits is None else f"{self.commits} commit(s) "
            command = self.update_command()
            fix = (
                f" Update it with `detail remote update {shlex.quote(self.target)}` "
                f"(or Update DETAIL… on the Remote targets page), or by hand: {command}"
                if command
                else ""
            )
            return (
                f"DETAIL on {target} is {how}behind this app (it has {there}, this app "
                f"has {here}), so a run that needs what is new will fail there.{branches}"
                f"{changes}{fix}"
            )
        if self.state == AHEAD:
            how = "" if self.commits is None else f"{self.commits} commit(s) "
            return (
                f"DETAIL on {target} is ahead of this app by {how}(it has {there}, this "
                f"app has {here}). Update this app (git pull) rather than the cluster."
                f"{branches}"
            )
        if self.state == DIVERGED:
            return (
                f"DETAIL on {target} has a different history from this app (it has {there}, "
                f"this app has {here}); {there} is not a commit this copy has, or neither "
                f"contains the other.{branches}{changes}"
            )
        return f"Could not compare DETAIL on {target} with this app: {self.reason}"


def compare_with(remote: dict, *, target: str, host: str) -> BuildComparison:
    """Compare a build another machine reported (:func:`report`) with this one.

    Uses this checkout's own git for the ordering, so it never fetches and never touches the
    other machine: a commit this copy does not have is reported as such, not looked up.
    """
    base = {
        "target": target,
        "host": host,
        "remote_commit": remote.get("commit"),
        "remote_branch": remote.get("branch"),
        "remote_root": remote.get("root"),
        "remote_changes": bool(remote.get("local_changes")),
    }
    local_sha = head_sha()
    info = build_info()
    base.update(local_commit=info.commit, local_branch=info.branch)
    remote_sha = remote.get("sha")
    if local_sha is None:
        return BuildComparison(
            UNKNOWN, **base, reason="this copy of DETAIL is not a git checkout"
        )
    if not remote_sha:
        return BuildComparison(
            UNKNOWN,
            **base,
            reason=f"the DETAIL on {target!r} is not a git checkout, so its build is unknown",
        )
    if remote_sha == local_sha:
        return BuildComparison(SAME, **base)
    # `_git` is "" on success and None on failure, so an ancestor is `is not None`. A commit this
    # copy has never seen fails both ancestor questions, and so lands on DIVERGED below.
    if _git("merge-base", "--is-ancestor", remote_sha, local_sha) is not None:
        count = _git("rev-list", "--count", f"{remote_sha}..{local_sha}")
        return BuildComparison(
            BEHIND, **base, commits=int(count) if count and count.isdigit() else None
        )
    if _git("merge-base", "--is-ancestor", local_sha, remote_sha) is not None:
        count = _git("rev-list", "--count", f"{local_sha}..{remote_sha}")
        return BuildComparison(
            AHEAD, **base, commits=int(count) if count and count.isdigit() else None
        )
    return BuildComparison(DIVERGED, **base)


def history(limit: int = 200) -> list[HistoryEntry]:
    """The newest ``limit`` first-parent commits of HEAD, newest first; empty outside git.

    First-parent so that on main each entry is one merged PR, with nothing to maintain by hand.
    The subject is the PR title for a squash merge, and ``Merge pull request #N from
    owner/branch`` for a GitHub merge commit, which carries the title on a later line.
    """
    out = _git(
        "log",
        "--first-parent",
        f"--max-count={limit}",
        f"--format=%h{_FIELD_SEP}%cs{_FIELD_SEP}%s",
        "HEAD",
    )
    entries = []
    for line in (out or "").splitlines():
        commit, date, subject = line.split(_FIELD_SEP, 2)
        entries.append(HistoryEntry(commit=commit, date=date, subject=subject))
    return entries


def check_for_updates() -> UpdateStatus:
    """Fetch ``origin/main`` and count how far the checkout is behind and ahead of it.

    The one network call here; run only on a user's request, off the UI thread.

    Raises:
        UpdateCheckError: not a git checkout, or the fetch failed (offline, no such remote,
            or a remote that needs credentials).
    """
    if head_sha() is None:
        raise UpdateCheckError(
            "This copy of detail is not a git checkout, so it cannot check for updates."
        )
    upstream = f"{UPSTREAM_REMOTE}/{UPSTREAM_BRANCH}"
    fetched = _git(
        "fetch", "--quiet", UPSTREAM_REMOTE, UPSTREAM_BRANCH, timeout=_FETCH_TIMEOUT_S
    )
    if fetched is None:
        raise UpdateCheckError(
            f"Could not fetch {upstream} — check the network connection and that the "
            f"'{UPSTREAM_REMOTE}' remote exists."
        )
    counts = _git("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
    if not counts:
        raise UpdateCheckError(f"Could not compare this checkout with {upstream}.")
    ahead, behind = (int(n) for n in counts.split())
    return UpdateStatus(
        behind=behind,
        ahead=ahead,
        upstream_date=_git("log", "-1", "--format=%cs", upstream),
    )
