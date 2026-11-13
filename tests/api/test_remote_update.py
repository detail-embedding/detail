"""Updating the DETAIL on a cluster: only ever a fast-forward, and only when it is behind.

`Test connection` already says when a cluster is on another commit than this app and prints the
command to fix it. This is that command made a button (and `detail remote update`), which means
the app now **changes something on the user's cluster account**. What these pin is the part that
makes that safe to offer:

- the cluster-side step is the real script, run here against **real git** (an origin, and a clone
  that is behind it): it fast-forwards the checkout, and when git refuses (the checkout has commits
  of its own, or a local change in the way) the checkout is exactly as it was and git's own words
  come back;
- it is attempted only from *behind*: the same build needs nothing, and ahead / diverged / unknown
  are not a fast-forward from here, so nothing runs on the cluster for them;
- `ok` means the cluster now matches this app, not that a command exited 0: a pull that moves
  nothing, or lands on another commit, says why and is `ok=False`;
- the script holds no mutation but `pull --ff-only` (a source guard, so a later edit cannot make
  the update destructive without a test failing).

Not covered: ssh itself and the cluster's environment activation. The script is executed locally in
place of `remote_python`, the way `test_remote_build` runs the build snippet.
"""

import subprocess
import sys

import pytest

from detail.api import remote as api_remote
from detail.configs.remote import RemoteTargetConfig
from detail.core import buildinfo

pytestmark = pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="needs git",
)

_ID = [
    "-c",
    "user.name=t",
    "-c",
    "user.email=t@example.com",
    "-c",
    "commit.gpgsign=false",
]


def _git(cwd, *args):
    done = subprocess.run(
        ["git", *_ID, *args],
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return done.stdout.strip()


def _commit(repo, name, text="x\n", message=None):
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message or f"add {name}")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def cluster(tmp_path):
    """``(origin, checkout)``: a checkout on the cluster, one commit behind its origin."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit(origin, "a.txt")
    checkout = tmp_path / "cluster" / "detail"
    _git(tmp_path, "clone", "-q", str(origin), str(checkout))
    _commit(origin, "b.txt")
    return origin, checkout


def _comparison(state, checkout, **kwargs):
    return buildinfo.BuildComparison(
        state,
        target="tillicum",
        host="tillicum",
        local_commit="aaaaaaaa",
        remote_commit="bbbbbbbb",
        remote_root=str(checkout),
        **kwargs,
    )


@pytest.fixture
def wired(monkeypatch):
    """The target, the two comparisons it returns (before, then after), and the script run here."""
    config = RemoteTargetConfig(
        name="tillicum", host="tillicum", remote_workdir="/scratch/x"
    )
    monkeypatch.setattr(api_remote, "get_remote_target", lambda name: config)
    ran = []
    answers = []

    def run_here(target, script, failed, *, timeout_s=None):
        ran.append(script)
        done = subprocess.run(
            [sys.executable, "-"],
            input=script,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if done.returncode != 0:
            raise RuntimeError(f"{failed}: {done.stderr.strip().splitlines()[-1]}")
        return done

    monkeypatch.setattr(api_remote, "remote_python", run_here)

    def check(name, timeout_s=15):
        return answers.pop(0)

    monkeypatch.setattr(api_remote, "check_remote_build", check)
    return ran, answers


# --------------------------------------------------------------------------- #
# the cluster-side step, on real git
# --------------------------------------------------------------------------- #
def test_a_checkout_that_is_behind_is_fast_forwarded_and_verified(cluster, wired):
    origin, checkout = cluster
    ran, answers = wired
    stages = []
    answers += [
        _comparison(buildinfo.BEHIND, checkout, commits=1),
        _comparison(buildinfo.SAME, checkout),
    ]

    result = api_remote.update_remote_build(
        "tillicum", progress=lambda *a: stages.append(a[0])
    )

    assert result.ok, result.detail
    assert _git(checkout, "rev-parse", "HEAD") == _git(origin, "rev-parse", "HEAD")
    assert "Updated DETAIL on 'tillicum'" in result.detail
    assert "now the same build as this app" in result.detail
    assert stages == ["Checking", "Updating", "Verifying"]
    assert len(ran) == 1


def test_a_supplied_comparison_is_not_asked_for_again(cluster, wired):
    origin, checkout = cluster
    _ran, answers = wired
    answers.append(_comparison(buildinfo.SAME, checkout))  # only the verification

    stages = []
    result = api_remote.update_remote_build(
        "tillicum",
        before=_comparison(buildinfo.BEHIND, checkout),
        progress=lambda *a: stages.append(a[0]),
    )

    assert result.ok
    assert stages == ["Updating", "Verifying"], "the caller had already checked"


def test_when_git_refuses_the_checkout_is_exactly_as_it_was(cluster, wired):
    """A checkout with a commit of its own is not a fast-forward. Nothing may move."""
    origin, checkout = cluster
    ran, answers = wired
    mine = _commit(checkout, "local.txt", message="a commit made on the cluster")
    answers.append(_comparison(buildinfo.BEHIND, checkout))

    result = api_remote.update_remote_build("tillicum")

    assert not result.ok
    assert "git refused the update on 'tillicum', which is unchanged" in result.detail
    assert result.output, "git's own words come back"
    assert _git(checkout, "rev-parse", "HEAD") == mine
    assert (checkout / "local.txt").exists()
    assert not (checkout / "b.txt").exists(), "nothing was merged in"


def test_a_local_change_that_would_be_overwritten_is_left_alone(cluster, wired):
    origin, checkout = cluster
    _commit(origin, "a.txt", "changed upstream\n")
    (checkout / "a.txt").write_text("edited on the cluster\n", encoding="utf-8")
    _ran, answers = wired
    answers.append(_comparison(buildinfo.BEHIND, checkout, remote_changes=True))
    before = _git(checkout, "rev-parse", "HEAD")

    result = api_remote.update_remote_build("tillicum")

    assert not result.ok and "git refused" in result.detail
    assert _git(checkout, "rev-parse", "HEAD") == before
    assert (checkout / "a.txt").read_text(
        encoding="utf-8-sig"
    ) == "edited on the cluster\n"


def test_a_pull_that_moves_nothing_says_the_apps_commits_are_not_there(cluster, wired):
    """The cluster pulls from its own remote, which need not have what this app has."""
    origin, checkout = cluster
    _git(checkout, "pull", "-q", "--ff-only")  # already at its remote's tip
    _ran, answers = wired
    answers += [
        _comparison(buildinfo.BEHIND, checkout, commits=3),
        _comparison(buildinfo.BEHIND, checkout, commits=3),
    ]

    result = api_remote.update_remote_build("tillicum")

    assert not result.ok
    assert "Nothing new to pull" in result.detail
    assert "probably not pushed" in result.detail


def test_a_pull_that_lands_on_another_commit_is_not_called_a_success(cluster, wired):
    origin, checkout = cluster
    _ran, answers = wired
    answers += [
        _comparison(buildinfo.BEHIND, checkout),
        _comparison(buildinfo.AHEAD, checkout, commits=2),
    ]

    result = api_remote.update_remote_build("tillicum")

    assert not result.ok
    assert "still not the same build as this app" in result.detail


def test_a_changed_pyproject_says_to_reinstall_and_does_not(cluster, wired):
    origin, checkout = cluster
    _commit(origin, "pyproject.toml", "[project]\nname='x'\n")
    _ran, answers = wired
    answers += [
        _comparison(buildinfo.BEHIND, checkout),
        _comparison(buildinfo.SAME, checkout),
    ]

    result = api_remote.update_remote_build("tillicum")

    assert result.ok and result.reinstall
    assert "pip install -e ." in result.detail


# --------------------------------------------------------------------------- #
# only from behind
# --------------------------------------------------------------------------- #
def test_the_same_build_needs_nothing_and_touches_nothing(cluster, wired):
    _origin, checkout = cluster
    ran, answers = wired
    answers.append(_comparison(buildinfo.SAME, checkout))

    result = api_remote.update_remote_build("tillicum")

    assert result.ok and "Nothing to update" in result.detail
    assert ran == [], "no command ran on the cluster"


@pytest.mark.parametrize(
    "state", [buildinfo.AHEAD, buildinfo.DIVERGED, buildinfo.UNKNOWN]
)
def test_nothing_runs_on_the_cluster_unless_it_is_behind(cluster, wired, state):
    _origin, checkout = cluster
    ran, answers = wired
    answers.append(_comparison(state, checkout, reason="it could not be read"))

    result = api_remote.update_remote_build("tillicum")

    assert not result.ok and result.detail.startswith("Not updated.")
    assert ran == []


def test_a_target_that_did_not_say_where_its_checkout_is_is_not_guessed_at(
    cluster, wired
):
    _origin, checkout = cluster
    ran, answers = wired
    behind = _comparison(buildinfo.BEHIND, checkout)
    answers.append(
        buildinfo.BuildComparison(
            buildinfo.BEHIND, target="tillicum", host="tillicum", remote_root=None
        )
    )

    result = api_remote.update_remote_build("tillicum")

    assert not result.ok and "did not say where" in result.detail
    assert ran == [] and behind.remote_root


def test_a_failure_to_run_on_the_cluster_is_a_result_not_an_exception(
    cluster, wired, monkeypatch
):
    _origin, checkout = cluster
    _ran, answers = wired
    answers.append(_comparison(buildinfo.BEHIND, checkout))

    def broken(target, script, failed, *, timeout_s=None):
        raise RuntimeError(f"{failed}: Permission denied (publickey)\nmore")

    monkeypatch.setattr(api_remote, "remote_python", broken)

    result = api_remote.update_remote_build("tillicum")

    assert not result.ok
    assert result.detail == (
        "could not update DETAIL on 'tillicum': Permission denied (publickey)"
    )


def test_an_unregistered_target_is_refused(monkeypatch):
    def missing(name):
        raise KeyError(name)

    monkeypatch.setattr(api_remote, "get_remote_target", missing)
    with pytest.raises(KeyError):
        api_remote.update_remote_build("nowhere")


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #
def test_the_cluster_side_script_changes_nothing_but_a_fast_forward_pull():
    """What this adds to a user's cluster account is `git pull --ff-only` and reads. A later edit
    that reaches for anything that rewrites history or discards work fails here first.
    """
    script = api_remote._PULL_SCRIPT
    assert '"pull", "--ff-only"' in script
    for word in (
        "--force",
        "--rebase",
        "--no-ff",
        '"reset"',
        '"clean"',
        '"checkout"',
        '"stash"',
        '"merge"',
        '"fetch"',
        '"rm"',
        "rmtree",
        "os.remove",
        "pip",
    ):
        assert word not in script, f"the update script must not use {word}"
    git_verbs = {
        line.split('"')[1]
        for line in script.splitlines()
        if "git(" in line and '"' in line
    }
    assert git_verbs <= {"rev-parse", "pull", "diff"}, git_verbs
