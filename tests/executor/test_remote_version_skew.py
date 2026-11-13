"""A cluster whose DETAIL is older than the app, said as that (§6).

**The class: a failure whose cause is a version mismatch, reported as a raw import error.** The
app's readiness probe imports every runner module the app lists, so a module a newer release added
(`detail.pipeline.keypoint_dataset`, with the keypoints distill) fails the probe on a cluster that
was last updated before it, for every recording of a run at once. A resumed 77-recording detection
run showed ten red rows reading "could not check model readiness on remote target 'tillicum'",
with the real line, ``ModuleNotFoundError: No module named 'detail.pipeline.keypoint_dataset'``,
past the edge of the table and under a conda login banner. Nothing in it said the cluster needed
updating, and the fix is one ``git pull`` there.

**It was one cause at three sites.** The video-root registration, the imported-model lookup and the
readiness probe each ran ``python -`` on the cluster and each built its own message from raw
stderr, so any of them could show it. They now share one runner (`executor/remote_python.py`), and a source
scan fails if a second place starts building that command, because a second place is where the
mismatch would read as a bare traceback again.

Not covered, deliberately: a *worker* that dies on the cluster before it writes a status ends the
run as "remote job ended (failed) without a terminal status". That needs the remote log tail, a
different mechanism with its own cost; for any job that carries a model this probe imports the
same modules first and gets there before submission.
"""

import pathlib
import re

import pytest

from detail.core.dataclasses import JobStatus
from detail.executor import remote_python, remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState
from tests.executor.test_imported_model_upload import (
    BANNER,
    MODEL_ID,
    _job as _imported_job,
    _library_model,
)
from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeCompletedProcess,
    _FakeScheduler,
    _install_scheduler,
    _job,
    _register_local_video_root,
    _target,
    _write_recording_with_video,
    experiment,
)

MISSING = "detail.pipeline.keypoint_dataset"
# The order the real cluster printed it: the exception first, the login banner after, so the
# *last* line is not the cause.
SKEW_OUTPUT = f"ModuleNotFoundError: No module named '{MISSING}'\n{BANNER}"


def _fail_with(monkeypatch, stderr):
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr=stderr),
    )


def _video_roots(monkeypatch, tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"x")
    _register_local_video_root(monkeypatch, tmp_path, "Cropped Videos", video_dir)
    experiment_root = tmp_path / "exp1"
    _write_recording_with_video(
        experiment_root / "recordings" / "recording_001",
        [{"index": 0, "source": {"root": "Cropped Videos", "rel": "a.mp4"}}],
    )
    monkeypatch.setattr(ssh_transport, "copy_file_to_remote", lambda *a, **k: None)
    return experiment_root


def _register_video_root(monkeypatch, tmp_path):
    root = _video_roots(monkeypatch, tmp_path)
    remote_shepherd._sync_video_roots(_target(), root, _job())


def _find_imported_models(monkeypatch, tmp_path):
    _library_model()
    remote_shepherd._sync_catalog_model(_target(), _imported_job())


def _check_readiness(monkeypatch, tmp_path):
    remote_shepherd._check_model_readiness(
        _target(), _job(), "/scratch/exp1", "detail.pipeline.detection"
    )


#: Every place the app runs Python on the cluster, and what it was doing when it failed.
SITES = {
    "video-root": (
        _register_video_root,
        "could not register video root 'Cropped Videos'",
    ),
    "imported-model": (_find_imported_models, "could not find where 'tillicum'"),
    "readiness": (_check_readiness, "could not check model readiness on remote target"),
}


@pytest.mark.parametrize("site", SITES)
def test_a_cluster_with_an_older_detail_says_so_at_every_site(
    monkeypatch, tmp_path, site
):
    run, doing = SITES[site]
    _fail_with(monkeypatch, SKEW_OUTPUT)

    with pytest.raises(RuntimeError) as raised:
        run(monkeypatch, tmp_path)

    first, *rest = str(raised.value).splitlines()
    # The first line is all a run's table shows, so it carries the whole message.
    assert first.startswith(doing)
    assert "older than this app" in first
    assert MISSING in first
    assert "git pull --ff-only" in first and "'tillicum'" in first
    # The cluster's own output is kept after it for whoever wants the detail.
    assert f"No module named '{MISSING}'" in "\n".join(rest)
    assert "Miniforge" in "\n".join(rest)


def test_a_name_the_cluster_lacks_is_the_same_mismatch(monkeypatch, tmp_path):
    _fail_with(
        monkeypatch,
        "ImportError: cannot import name 'new_helper' from 'detail.core.utils' "
        "(/home/x/detail/detail/core/utils.py)\n",
    )

    with pytest.raises(RuntimeError, match="older than this app") as raised:
        _check_readiness(monkeypatch, tmp_path)

    assert "detail.core.utils.new_helper" in str(raised.value).splitlines()[0]


@pytest.mark.parametrize(
    "stderr",
    [
        # DETAIL is not installed in that environment at all: the target's activation is wrong,
        # which "update it" would send someone away from.
        "ModuleNotFoundError: No module named 'detail'\n",
        # Somebody else's missing package is not a version mismatch.
        "ModuleNotFoundError: No module named 'torch'\n",
        "disk quota exceeded\n",
    ],
    ids=["detail-absent", "other-package", "unrelated"],
)
@pytest.mark.parametrize("site", SITES)
def test_any_other_failure_keeps_its_own_cause(monkeypatch, tmp_path, site, stderr):
    run, doing = SITES[site]
    _fail_with(monkeypatch, stderr)

    with pytest.raises(RuntimeError) as raised:
        run(monkeypatch, tmp_path)

    message = str(raised.value)
    assert message.startswith(doing)
    assert stderr.strip() in message
    assert "older than this app" not in message


@pytest.mark.parametrize(
    "output, expected",
    [
        (f"ModuleNotFoundError: No module named '{MISSING}'", MISSING),
        ("ModuleNotFoundError: No module named 'detail.a'", "detail.a"),
        (
            "ImportError: cannot import name 'f' from 'detail.core.utils' (x.py)",
            "detail.core.utils.f",
        ),
        ("ModuleNotFoundError: No module named 'detail'", None),
        ("ModuleNotFoundError: No module named 'detailed.x'", None),
        ("ImportError: cannot import name 'f' from 'torch.nn'", None),
        ("", None),
    ],
)
def test_remote_version_skew_names_only_what_detail_lacks(output, expected):
    assert remote_python.remote_version_skew(output) == expected


def test_a_run_that_fails_the_probe_ends_with_the_message_in_its_table(
    monkeypatch, experiment
):
    """Through ``run``, the way the resumed detection run met it: the envelope's error is what the
    Active-processes row and the run's recording table show."""
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    _fail_with(monkeypatch, SKEW_OUTPUT)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    error = rundir.read_envelope(run_dir)["error"]
    assert "older than this app" in error.splitlines()[0]
    assert not scheduler.submitted  # failed before an sbatch job was ever queued


def test_python_is_only_ever_run_on_the_cluster_through_one_function():
    """The guard that makes the three sites one: a `python -` command built anywhere else would
    report its own failures, and be the place a version mismatch reads as a bare traceback.

    The command is `interpreter.remote_python_command("-")`; `tests/core/test_interpreter.py`
    fails on a remote `python` started any other way, so finding its callers finds them all.
    """
    package = pathlib.Path(remote_shepherd.__file__).parents[1]
    found = []
    for path in sorted(package.rglob("*.py")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        for number, line in enumerate(lines):
            if 'remote_python_command("-")' not in line:
                continue
            owner = next(
                (
                    re.match(r"\s*def (\w+)", earlier).group(1)
                    for earlier in reversed(lines[:number])
                    if re.match(r"\s*def (\w+)", earlier)
                ),
                None,
            )
            found.append((path.relative_to(package).as_posix(), owner))

    assert found == [("executor/remote_python.py", "remote_python")]
