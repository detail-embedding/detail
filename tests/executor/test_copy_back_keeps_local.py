"""The copy back from the cluster must not replace a file that belongs to this machine.

**The class: a mirror that replaces a file somebody still has open.** A remote run's shepherd writes
its log to ``<run dir>/shepherd.log``, which is the shepherd process's stdout, opened by its parent
(``RemoteExecutor.submit``). The upload carries the run directory to the cluster when the run
starts, log and all, so the cluster holds a copy frozen at "uploading experiment files". When the
job ends, the shepherd copies the run directory back with ``rsync -a``. rsync replaces a file by
writing a temporary one and renaming it over the name, so the cluster's stale log took the name
while the shepherd went on writing into the original, now unlinked. Every line after the copy back
(the sync, the final outcome, a traceback) went nowhere, and every log of keypoints ``run_005``
read as if its run had stopped two lines in.

**The rule.** ``copy_from_remote(keep_local=…)`` names files the copy never replaces, and the
shepherd passes its own log and the run's envelope (#742, `test_status_after_data.py`). rsync
excludes them (anchored, so only the run directory's own file); the ``scp`` fallback cannot, so it
copies into a staging folder and moves everything else into place. The kept files are never
moved: moving them aside, as the fallback once did, left the envelope *missing* for the whole
copy, which a reader polling the run's status sees.

**These run real rsync** against a directory standing in for the cluster, with the host stripped
from the arguments, because the failure is rsync's rename-over semantics meeting an open file, which
an argument-list test cannot show. They skip where rsync is not installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from detail.executor import rundir, ssh_transport

needs_rsync = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="rsync is not installed"
)

HOST = "tillicum"
LOG = rundir.SHEPHERD_LOG_FILENAME


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A local run directory whose log is open for appending, and the cluster's copy of the run."""
    local, cluster = tmp_path / "local", tmp_path / "cluster"
    local.mkdir()
    cluster.mkdir()
    (cluster / LOG).write_text("uploading experiment files\n", encoding="utf-8")
    (cluster / "result.txt").write_text("from the worker", encoding="utf-8")
    monkeypatch.setattr(ssh_transport, "require_responsive", lambda host: None)
    handle = open(local / LOG, "a", encoding="utf-8")  # the shepherd's stdout
    handle.write("uploading experiment files\nsubmitting the job\nrunning\n")
    handle.flush()
    yield local, cluster, handle
    handle.close()


def _real_rsync(monkeypatch, cluster: Path) -> None:
    def run_locally(argv, *, timeout_s):
        argv = [a.replace(f"{HOST}:", "", 1) for a in argv]
        subprocess.run(argv, check=True, capture_output=True)

    monkeypatch.setattr(ssh_transport, "_rsync_available", lambda: True)
    monkeypatch.setattr(ssh_transport, "_run", run_locally)


def _copy_back_then_log(local, cluster, handle, **kwargs) -> str:
    ssh_transport.copy_from_remote(HOST, str(cluster), local, **kwargs)
    handle.write("syncing results back\ncomplete\n")
    handle.flush()
    assert (local / "result.txt").read_text(encoding="utf-8-sig") == "from the worker"
    return (local / LOG).read_text(encoding="utf-8-sig")


@needs_rsync
def test_the_shepherds_log_keeps_every_line_through_the_copy_back(run, monkeypatch):
    local, cluster, handle = run
    _real_rsync(monkeypatch, cluster)

    text = _copy_back_then_log(local, cluster, handle, keep_local=[LOG])

    assert text.splitlines() == [
        "uploading experiment files",
        "submitting the job",
        "running",
        "syncing results back",
        "complete",
    ]


@needs_rsync
def test_without_keep_local_the_log_is_the_clusters_and_later_lines_are_lost(
    run, monkeypatch
):
    """The control, so the test above is shown to be able to fail."""
    local, cluster, handle = run
    _real_rsync(monkeypatch, cluster)

    text = _copy_back_then_log(local, cluster, handle)

    assert text == "uploading experiment files\n"


def _scp_copying(monkeypatch, watch: Path | None = None, seen: list | None = None):
    """``_scp_from_remote`` as a local copy; with ``watch``, what that file read while it ran."""

    def scp(host, remote_dir, local_dir, *, timeout_s):
        if watch is not None:
            seen.append(
                watch.read_text(encoding="utf-8-sig") if watch.exists() else None
            )
        shutil.copytree(remote_dir, local_dir, dirs_exist_ok=True)

    monkeypatch.setattr(ssh_transport, "_rsync_available", lambda: False)
    monkeypatch.setattr(ssh_transport, "_scp_from_remote", scp)


def test_the_scp_fallback_keeps_the_kept_file_open_handle_and_all(run, monkeypatch):
    local, cluster, handle = run
    _scp_copying(monkeypatch)

    text = _copy_back_then_log(local, cluster, handle, keep_local=[LOG])

    assert text.splitlines()[-2:] == ["syncing results back", "complete"]
    assert "running" in text.splitlines()
    assert {p.name for p in local.iterdir()} == {
        LOG,
        "result.txt",
    }, "nothing is left aside or staged"


def test_the_scp_fallback_never_moves_a_kept_file_out_of_its_place(run, monkeypatch):
    """The envelope is read by everything polling the run. Absent for the length of the copy
    (the old move-aside) or holding the cluster's terminal status (a copy-then-restore), a
    reader decides on a run whose data is still arriving (#742)."""
    local, cluster, handle = run
    envelope = local / rundir.JOB_FILENAME
    envelope.write_text('{"status": "running"}', encoding="utf-8")
    (cluster / rundir.JOB_FILENAME).write_text(
        '{"status": "complete"}', encoding="utf-8"
    )
    seen: list = []
    _scp_copying(monkeypatch, watch=envelope, seen=seen)

    ssh_transport.copy_from_remote(
        HOST, str(cluster), local, keep_local=[LOG, rundir.JOB_FILENAME]
    )

    assert seen == [
        '{"status": "running"}'
    ], "the envelope was in place while it copied"
    assert envelope.read_text(encoding="utf-8-sig") == '{"status": "running"}'
    assert (local / "result.txt").read_text(encoding="utf-8-sig") == "from the worker"


@needs_rsync
def test_rsync_never_brings_the_clusters_envelope(run, monkeypatch):
    local, cluster, _handle = run
    _real_rsync(monkeypatch, cluster)
    (local / rundir.JOB_FILENAME).write_text('{"status": "running"}', encoding="utf-8")
    (cluster / rundir.JOB_FILENAME).write_text(
        '{"status": "complete"}', encoding="utf-8"
    )

    ssh_transport.copy_from_remote(
        HOST, str(cluster), local, keep_local=[LOG, rundir.JOB_FILENAME]
    )

    assert (local / rundir.JOB_FILENAME).read_text(encoding="utf-8-sig") == (
        '{"status": "running"}'
    )


@needs_rsync
def test_a_refreshed_input_is_mirrored_exactly_and_its_envelope_kept(
    tmp_path, monkeypatch
):
    """``mirror_to_remote`` (a run a remote job reads, re-sent before it starts): what this
    machine removed is removed there too — a profile a cancelled job placed on the cluster's
    copy would otherwise make the next job skip its cluster — and the cluster's own envelope
    is neither replaced nor removed."""
    local, cluster = tmp_path / "an_10", tmp_path / "cluster_an_10"
    (local / "profiles" / "node_3").mkdir(parents=True)
    (local / "profiles" / "node_3" / "profile.yaml").write_text(
        "here", encoding="utf-8"
    )
    (local / "job.json").write_text("local envelope", encoding="utf-8")
    (cluster / "profiles" / "node_9").mkdir(parents=True)
    (cluster / "profiles" / "node_9" / "profile.yaml").write_text(
        "stale", encoding="utf-8"
    )
    (cluster / "job.json").write_text("cluster envelope", encoding="utf-8")
    monkeypatch.setattr(ssh_transport, "run_remote", lambda *a, **k: None)
    monkeypatch.setattr(ssh_transport, "_rsync_available", lambda: True)
    monkeypatch.setattr(
        ssh_transport,
        "_run",
        lambda argv, **k: subprocess.run(
            [a.replace(f"{HOST}:", "", 1) for a in argv], check=True
        ),
    )

    ssh_transport.mirror_to_remote(
        HOST, local, str(cluster), keep_remote=[rundir.JOB_FILENAME]
    )

    assert (cluster / "profiles" / "node_3" / "profile.yaml").read_text(
        encoding="utf-8-sig"
    ) == "here"
    assert not (cluster / "profiles" / "node_9").exists()
    assert (cluster / "job.json").read_text(encoding="utf-8-sig") == "cluster envelope"


def test_a_mirror_without_rsync_is_refused_rather_than_inexact(tmp_path, monkeypatch):
    monkeypatch.setattr(ssh_transport, "_rsync_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs rsync"):
        ssh_transport.mirror_to_remote(HOST, tmp_path, "/remote/an_10")
