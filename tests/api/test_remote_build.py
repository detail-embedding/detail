"""``check_remote_build`` and what ``test_remote_target`` does with it (§6).

The comparison itself (same / behind / ahead / diverged, and the command that fixes it) is
``tests/core/test_buildinfo_compare.py``. This file is the part that asks a *cluster*: what is sent
there, what is made of every kind of answer, and that a cluster running another build is a warning
beside a working connection and never a failed one.

**It never raises for the cluster.** "Could not tell" is a result of the check: a Test connection
button that threw because the cluster's DETAIL was too old to answer would hide the connection
that the same press had just proven.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import detail
from detail.api import remote as api_remote
from detail.configs.remote import RemoteTargetConfig
from detail.core import buildinfo
from detail.executor import ssh_transport
from detail.storage import remote_targets

BANNER = "Miniforge (conda) has been loaded.\nWARNING: Login nodes are CPU ONLY.\n"
OLD_SHA = "1" * 40


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg / "remote_targets.yaml"
    )
    api_remote.add_remote_target(
        RemoteTargetConfig(
            name="tillicum", host="tillicum", remote_workdir="/scratch/x"
        )
    )
    # What the app itself is: a fixed build, whatever checkout the tests run from.
    monkeypatch.setattr(buildinfo, "head_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        buildinfo,
        "build_info",
        lambda: buildinfo.BuildInfo(version="0.1.0", commit="aaaaaaa", branch="main"),
    )


class _Process:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _cluster(monkeypatch, answer):
    """The ssh surface: records what was sent, answers with ``answer`` (or calls it)."""
    sent = []

    def run_remote(host, command, *, stdin_text=None, **kwargs):
        sent.append((host, command, stdin_text, kwargs))
        return answer(stdin_text) if callable(answer) else answer

    monkeypatch.setattr(ssh_transport, "run_remote", run_remote)
    return sent


def _report(sha, **overrides) -> str:
    report = {
        "sha": sha,
        "commit": sha[:7],
        "branch": "remote-detail",
        "date": "2026-10-06",
        "local_changes": False,
        "root": "/gpfs/home/someone/detail",
    }
    report.update(overrides)
    return (
        BANNER + json.dumps(report) + "\n"
    )  # the banner first, as a real login prints


def test_the_cluster_is_asked_for_its_own_build_in_the_targets_environment(monkeypatch):
    sent = _cluster(monkeypatch, _Process(0, _report("a" * 40)))

    api_remote.check_remote_build("tillicum")

    ((host, command, script, kwargs),) = sent
    assert host == "tillicum" and command.endswith("python -P -")
    assert "buildinfo.report()" in script
    assert (
        kwargs["timeout_s"] == api_remote.DEFAULT_TEST_TIMEOUT_S
    )  # bounded, not forever


def test_a_banner_before_the_answer_is_ignored(monkeypatch):
    _cluster(monkeypatch, _Process(0, _report("a" * 40)))

    assert api_remote.check_remote_build("tillicum").state == buildinfo.SAME


def test_a_cluster_on_another_commit_is_not_the_same_build(monkeypatch):
    _cluster(monkeypatch, _Process(0, _report(OLD_SHA)))

    result = api_remote.check_remote_build("tillicum")

    assert result.state != buildinfo.SAME and not result.matches
    assert result.remote_commit == "1111111"
    assert (result.target, result.host) == ("tillicum", "tillicum")


@pytest.mark.parametrize(
    "answer, reason",
    [
        (
            _Process(
                1,
                "",
                "ModuleNotFoundError: No module named 'detail.core.buildinfo'\n"
                + BANNER,
            ),
            "older than this app",  # too old to even report: said as that, with the fix
        ),
        (_Process(1, "", "conda: command not found"), "conda: command not found"),
        (_Process(255, "", "Connection reset"), "Connection reset"),
        (_Process(0, BANNER + "not json\n"), "not a build report"),
        (_Process(0, BANNER + "[1, 2]\n"), "not a build report"),
        (_Process(0, ""), "not a build report"),
    ],
    ids=[
        "cluster-too-old",
        "no-conda",
        "ssh-error",
        "not-json",
        "not-an-object",
        "empty",
    ],
)
def test_a_cluster_that_cannot_answer_is_unknown_with_the_reason(
    monkeypatch, answer, reason
):
    _cluster(monkeypatch, answer)

    result = api_remote.check_remote_build("tillicum")

    assert result.state == buildinfo.UNKNOWN
    assert reason in result.reason
    assert reason in result.describe()


def test_a_cluster_that_does_not_answer_in_time_is_unknown(monkeypatch):
    def hang(stdin_text):
        raise subprocess.TimeoutExpired("ssh", 15)

    _cluster(monkeypatch, hang)

    result = api_remote.check_remote_build("tillicum", timeout_s=15)

    assert result.state == buildinfo.UNKNOWN
    assert "did not answer within 15s" in result.reason


def test_an_unregistered_target_is_a_keyerror_like_every_other_call():
    with pytest.raises(KeyError):
        api_remote.check_remote_build("nope")


# --- inside Test connection ---------------------------------------------------- #
def _reachable(monkeypatch):
    """The plain-ssh connectivity probe: reachable, sbatch found."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: _Process(0, "DETAIL_SBATCH_OK\n"),
    )


def test_a_working_connection_carries_the_comparison(monkeypatch):
    _reachable(monkeypatch)
    _cluster(monkeypatch, _Process(0, _report("a" * 40)))

    status = api_remote.test_remote_target("tillicum")

    assert status.ok
    assert status.build is not None and status.build.matches


def test_a_cluster_on_another_build_is_a_warning_not_a_failed_connection(monkeypatch):
    _reachable(monkeypatch)
    _cluster(monkeypatch, _Process(0, _report(OLD_SHA)))

    status = api_remote.test_remote_target("tillicum")

    assert status.ok  # reachable, sbatch found: the connection test passed
    assert not status.build.matches


def test_a_cluster_too_old_to_answer_does_not_fail_the_connection_test(monkeypatch):
    _reachable(monkeypatch)
    _cluster(
        monkeypatch,
        _Process(1, "", "ModuleNotFoundError: No module named 'detail.core.buildinfo'"),
    )

    status = api_remote.test_remote_target("tillicum")

    assert status.ok and status.build.state == buildinfo.UNKNOWN


def test_nothing_is_asked_of_a_cluster_that_could_not_be_reached(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Process(255, "", "Connection refused")
    )
    sent = _cluster(monkeypatch, _Process(0, _report("a" * 40)))

    status = api_remote.test_remote_target("tillicum")

    assert not status.ok and status.build is None
    assert sent == []  # the connection's own failure is the whole answer


# --- the script itself --------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_script_the_cluster_runs_makes_a_report_this_app_can_read(monkeypatch):
    """Run the exact snippet that is sent, here, as the cluster would: against the real DETAIL
    package, in a fresh interpreter. A snippet that names something that does not exist would
    pass every test above, which only hand it a finished answer."""
    monkeypatch.undo()  # the real buildinfo, not this file's fixed build
    root = Path(detail.__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("not run from a git checkout")
    sent = []

    def run_remote(host, command, *, stdin_text=None, **kwargs):
        sent.append(stdin_text)
        done = subprocess.run(
            [sys.executable, "-"],
            input=stdin_text,
            capture_output=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONPATH": str(root)},
            cwd=root,
        )
        return done

    config = RemoteTargetConfig(
        name="tillicum", host="tillicum", remote_workdir="/scratch/x"
    )
    monkeypatch.setattr(api_remote, "get_remote_target", lambda name: config)
    monkeypatch.setattr(ssh_transport, "run_remote", run_remote)

    result = api_remote.check_remote_build("tillicum")

    assert sent, "the snippet was never run"
    assert result.state == buildinfo.SAME, result.describe()
