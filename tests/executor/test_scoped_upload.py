"""The upload to the cluster must not lay an older local copy over a run the cluster owns.

**The class guarded: a mirror that is correct for one writer and destructive for two.** The shepherd
mirrors the whole experiment to the cluster before it submits a job, with a plain `rsync -az`. A
job's run directory belongs to the *cluster* from the moment it is submitted: the worker there
rewrites `job.json`, its progress and its result records. The local copy is older (the last sync
back, or the initial pending state), and the mirror lays it over the newer one. With one job in
flight at a time nothing is wrong, which is how it stayed. With two, the second job's upload resets
the first's `job.json` just as it finishes, the first shepherd never reads a terminal status,
Slurm says the job completed, and a completed job is recorded `failed` ("remote job ended without
a terminal status"). Nothing fails at the line where it is written.

**The second half: a list is a snapshot.** The run directories to protect are found when the upload
starts, and an upload can run for twenty minutes. A run reserved after that — the next recording of
the same batch — is ordinary to the first pass, and when rsync's walk reaches it, that job's own
shepherd may already have written the cluster's `job.json` with the compute the cluster resolved.
The first pass laid the local one over it, and four recordings of one keypoints run failed with
`backend 'remote' not supported`. So the first pass never sends a `job.json` (nor its atomic-write
temps) anywhere: a run's envelope only arrives through the second pass of the job that listed it.

**The third half: a fallback that drops the guarantees it was asked for.** Both rules above are
rsync's. `copy_to_remote` fell back to `scp -r` on *any* rsync failure, and `scp` honours neither,
on the premise that rsync fails only on a host that lacks it. Tillicum has rsync and can fail it for
passing reasons: its shared connection holds ten sessions and eight shepherds poll through it, so
refused sessions are routine. (Which failure it was is not recorded: the fallback swallowed rsync's
error.) In run_011 (2026-10-09) one recording's upload took 27 minutes against the usual 45
seconds, the whole experiment re-sent over a cluster with jobs in flight, laying each one's local
`job.json` (`backend: remote`) over the envelope its shepherd had written. `recording_097`'s
shepherd wrote `backend: cuda` at 01:58:03Z; its worker started at 01:59:16Z (Slurm's clock) and
read an envelope already `running` with the *local* shepherd's 01:58:06Z start stamp, which only
the local file carried, and failed `ValueError: backend 'remote' not supported` in
`resolve_device`. The cluster's run directories carried `scp`'s signature, every file stamped at
copy time rather than with its source's mtime. Now `scp` is taken only where rsync does not exist (this machine, or a
host whose shell says `rsync: command not found` in any of its spellings) and every other failure
is raised; exit 24 (source files vanished mid-transfer, the atomic temps of the other shepherds'
envelopes) is a complete mirror, since what still existed was sent.

**These run real rsync** against a directory standing in for the cluster, because the fix is a
pair of rsync option interactions (`--exclude-from`, then `--files-from` with `--ignore-existing`,
and `-r` which `--files-from` silently turns off) that a test of the argument list cannot vouch
for. They skip where rsync is not installed (a bare Windows box); the argument-list tests at the end
run everywhere.

**Why not `--update`.** It skips files that are *newer on the host*, which sounds right and reads
the cluster's clock against ours: when the cluster's is a few seconds behind, the job that matters
(one that has just finished) looks older than the local copy and is overwritten anyway. The
rule here consults no clock: a run directory the host has is never rewritten.
"""

import os
import subprocess
import shutil
import time
from pathlib import Path

import pytest

from detail.executor import rundir, ssh_transport

needs_rsync = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="rsync is not installed"
)

RUN = "recordings/rec_001/detect"


def _write(root: Path, relative: str, text: str, *, age_s: int = 0) -> Path:
    """Write a file, optionally dated ``age_s`` seconds in the past.

    rsync skips a file whose size and modification time match, so a fixture that writes "v1" and
    "v2" in the same second is a copy rsync is right to leave alone. The stale side is made older
    *and* a different length, so a failure is the code's and not the fixture's.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if age_s:
        then = time.time() - age_s
        os.utime(path, (then, then))
    return path


@pytest.fixture
def sides(tmp_path):
    """A local experiment and the cluster's copy of it, one real directory each.

    The cluster has run ``det_01`` as its worker left it (finished, with a result record only the
    worker wrote). The local copy of that run is the stale one: still ``running``. ``det_02`` is a
    new job that exists only locally, and ``recording.yaml`` has moved on since the last upload.
    """
    local, cluster = tmp_path / "local", tmp_path / "cluster"
    _write(local, f"{RUN}/det_01/job.json", "running")
    _write(local, f"{RUN}/det_01/detect.yaml", "pending")
    _write(local, f"{RUN}/det_02/job.json", "pending")
    _write(local, f"{RUN}/det_02/tracks.zarr/0.0", "chunk")
    _write(local, "recordings/rec_001/recording.yaml", "version two")
    _write(local, "recordings/rec_001/notes.txt", "a new note")
    # The cluster's copy of the finished run is dated an hour **before** the local one: its clock
    # is behind ours. A rule that reads modification times would overwrite exactly this.
    _write(cluster, f"{RUN}/det_01/job.json", "complete", age_s=3600)
    _write(cluster, f"{RUN}/det_01/detect.yaml", "written by the worker", age_s=3600)
    _write(cluster, f"{RUN}/det_01/masks.zarr/0.0", "result", age_s=3600)
    _write(cluster, "recordings/rec_001/recording.yaml", "v1", age_s=3600)
    return local, cluster


def _mirror(local: Path, cluster: Path, **kwargs) -> None:
    ssh_transport._rsync_mirror(local, f"{cluster}/", timeout_s=None, **kwargs)


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8-sig")


@needs_rsync
def test_a_run_the_cluster_already_has_is_not_overwritten_by_an_older_local_copy(sides):
    """The incident: job A has finished on the cluster, and job B's upload must not reset it."""
    local, cluster = sides

    _mirror(local, cluster, never_overwrite=rundir.discover_run_dirs(local))

    assert _read(cluster, f"{RUN}/det_01/job.json") == "complete"
    assert _read(cluster, f"{RUN}/det_01/detect.yaml") == "written by the worker"
    assert (
        cluster / RUN / "det_01/masks.zarr/0.0"
    ).is_file(), "a worker-only file survived"


@needs_rsync
def test_the_plain_mirror_is_what_reset_it(sides):
    """The control, so the test above is shown to be able to fail."""
    local, cluster = sides

    _mirror(local, cluster)

    assert _read(cluster, f"{RUN}/det_01/job.json") == "running"


@needs_rsync
def test_a_run_the_cluster_lacks_arrives_whole(sides):
    """This job's own run directory is new to the cluster, results folders and all."""
    local, cluster = sides

    _mirror(local, cluster, never_overwrite=rundir.discover_run_dirs(local))

    assert _read(cluster, f"{RUN}/det_02/job.json") == "pending"
    assert _read(cluster, f"{RUN}/det_02/tracks.zarr/0.0") == "chunk", "-r was dropped"


@needs_rsync
def test_everything_that_is_not_a_run_still_updates(sides):
    local, cluster = sides

    _mirror(local, cluster, never_overwrite=rundir.discover_run_dirs(local))

    assert _read(cluster, "recordings/rec_001/recording.yaml") == "version two"
    assert _read(cluster, "recordings/rec_001/notes.txt") == "a new note"


@needs_rsync
def test_a_run_directory_the_cluster_has_still_gets_files_it_lacks(sides):
    """`--ignore-existing` leaves what is there alone and adds what is not: a retry that kept
    extracted frames locally sends the frames the first attempt did not finish."""
    local, cluster = sides
    _write(local, f"{RUN}/det_01/extra.txt", "only here")

    _mirror(local, cluster, never_overwrite=rundir.discover_run_dirs(local))

    assert _read(cluster, f"{RUN}/det_01/extra.txt") == "only here"
    assert _read(cluster, f"{RUN}/det_01/job.json") == "complete"


@needs_rsync
def test_a_name_with_rsyncs_wildcard_characters_protects_only_itself(tmp_path):
    """`rec [1]` unescaped is a character class that also matches `rec 1`: the protection would
    land on a folder that is not a run, and the one that is would be overwritten."""
    local, cluster = tmp_path / "local", tmp_path / "cluster"
    _write(local, "recordings/rec [1]/detect/det_01/job.json", "running")
    _write(local, "recordings/rec 1/notes.txt", "a newer, longer note")
    _write(cluster, "recordings/rec [1]/detect/det_01/job.json", "complete", age_s=3600)
    _write(cluster, "recordings/rec 1/notes.txt", "old", age_s=3600)

    _mirror(local, cluster, never_overwrite=rundir.discover_run_dirs(local))

    assert _read(cluster, "recordings/rec [1]/detect/det_01/job.json") == "complete"
    assert _read(cluster, "recordings/rec 1/notes.txt") == "a newer, longer note"


def _late_run(tmp_path):
    """A run directory the upload's list missed: the batch's next recording, reserved after the
    list was taken, whose shepherd has since written the cluster's envelope (with the compute the
    cluster resolved). The local envelope still says ``remote``."""
    local, cluster = tmp_path / "local", tmp_path / "cluster"
    _write(local, f"{RUN}/det_01/job.json", "pending")
    _write(
        local, "recordings/rec_002/kp/kp_01/job.json", "backend remote, as submitted"
    )
    _write(local, "recordings/rec_002/kp/kp_01/job.json.k3x9.tmp", "half-written")
    _write(cluster, "recordings/rec_002/kp/kp_01/job.json", "cuda", age_s=3600)
    return local, cluster, [local / RUN / "det_01"]


@needs_rsync
def test_a_run_made_after_the_list_keeps_the_envelope_its_shepherd_wrote(tmp_path):
    """Four recordings of one keypoints run failed with `backend 'remote' not supported`: their
    envelopes were laid over by another job's twenty-minute upload whose list predated them.
    """
    local, cluster, held = _late_run(tmp_path)

    _mirror(local, cluster, never_overwrite=held, never_send=[rundir.JOB_FILENAME])

    assert _read(cluster, "recordings/rec_002/kp/kp_01/job.json") == "cuda"
    assert not (cluster / "recordings/rec_002/kp/kp_01/job.json.k3x9.tmp").exists()
    assert (
        _read(cluster, f"{RUN}/det_01/job.json") == "pending"
    ), "a listed run still arrives"


@needs_rsync
def test_without_never_send_the_late_run_is_overwritten(tmp_path):
    """The control, so the test above is shown to be able to fail."""
    local, cluster, held = _late_run(tmp_path)

    _mirror(local, cluster, never_overwrite=held)

    assert (
        _read(cluster, "recordings/rec_002/kp/kp_01/job.json")
        == "backend remote, as submitted"
    )


@needs_rsync
def test_nothing_to_protect_is_the_ordinary_mirror(tmp_path):
    local, cluster = tmp_path / "local", tmp_path / "cluster"
    _write(local, "experiment.yaml", "version two")
    _write(cluster, "experiment.yaml", "v1", age_s=3600)

    _mirror(local, cluster, never_overwrite=[])

    assert _read(cluster, "experiment.yaml") == "version two"


# --- finding the runs ------------------------------------------------------------------------ #
def test_discover_run_dirs_finds_every_kind_of_run_and_stops_at_each(tmp_path):
    for relative in (
        "recordings/rec_001/detect/det_01",
        "recordings/rec_001/embed/emb_01",
        "datasets/dataset_001",
        "models/model_001",
    ):
        _write(tmp_path, f"{relative}/job.json", "{}")
    _write(tmp_path, "recordings/rec_001/detect/det_01/inner/job.json", "{}")
    _write(tmp_path, "recordings/rec_001/recording.yaml", "x")
    _write(tmp_path, "recordings/rec_001/video/part_001.mp4", "x")

    found = [
        p.relative_to(tmp_path).as_posix() for p in rundir.discover_run_dirs(tmp_path)
    ]

    assert found == [
        "datasets/dataset_001",
        "models/model_001",
        "recordings/rec_001/detect/det_01",
        "recordings/rec_001/embed/emb_01",
    ], "a folder inside a run is part of that run, and a plain folder is not one"


def test_discover_run_dirs_does_not_walk_into_a_run_or_a_zarr_store(
    tmp_path, monkeypatch
):
    """A detection's zarr is hundreds of thousands of chunk files: finding the runs must not
    cost more than the upload it is for."""
    _write(tmp_path, "recordings/rec_001/detect/det_01/job.json", "{}")
    _write(tmp_path, "recordings/rec_001/detect/det_01/masks.zarr/0.0", "x")
    _write(tmp_path, "recordings/rec_001/stray.zarr/job.json", "{}")
    listed = []
    real = rundir.os.scandir
    monkeypatch.setattr(
        rundir.os, "scandir", lambda p: listed.append(Path(p).name) or real(p)
    )

    found = rundir.discover_run_dirs(tmp_path)

    assert [p.name for p in found] == ["det_01"]
    assert "masks.zarr" not in listed and "stray.zarr" not in listed


def test_discover_run_dirs_of_an_empty_or_missing_experiment_is_empty(tmp_path):
    assert rundir.discover_run_dirs(tmp_path) == []
    assert rundir.discover_run_dirs(tmp_path / "nope") == []


# --- the argument lists (run everywhere) ----------------------------------------------------- #
def _record_runs(monkeypatch):
    calls, contents = [], []

    def fake_run(argv, *, timeout_s, accept=(0,)):
        calls.append(argv)
        for arg in argv:
            if arg.startswith(("--exclude-from=", "--files-from=")):
                contents.append(
                    Path(arg.split("=", 1)[1]).read_text(encoding="utf-8-sig")
                )

    monkeypatch.setattr(ssh_transport, "_run", fake_run)
    return calls, contents


def test_two_passes_protect_first_and_add_second(tmp_path, monkeypatch):
    calls, contents = _record_runs(monkeypatch)
    run = tmp_path / "recordings" / "rec 1" / "detect" / "det_01"
    run.mkdir(parents=True)

    ssh_transport._rsync_mirror(
        tmp_path, "tillicum:/work/exp/", timeout_s=None, never_overwrite=[run]
    )

    first, second = calls
    assert first[:2] == ["rsync", "-az"] and first[2].startswith("--exclude-from=")
    assert "--ignore-existing" not in first, "the first pass is the ordinary mirror"
    assert second[:4] == ["rsync", "-az", "-r", "--ignore-existing"]
    assert second[4].startswith("--files-from=")
    assert first[-1] == second[-1] == "tillicum:/work/exp/"
    assert contents == [
        "/recordings/rec 1/detect/det_01/\n",
        "recordings/rec 1/detect/det_01/\n",
    ]


def test_its_scratch_files_are_removed(tmp_path, monkeypatch):
    calls, _ = _record_runs(monkeypatch)
    (tmp_path / "r").mkdir()

    ssh_transport._rsync_mirror(
        tmp_path, "h:/w/", timeout_s=None, never_overwrite=[tmp_path / "r"]
    )

    temps = [
        a.split("=", 1)[1]
        for argv in calls
        for a in argv
        if a.startswith(("--exclude-from=", "--files-from="))
    ]
    assert temps and not [t for t in temps if Path(t).exists()]


def test_without_anything_to_protect_it_is_the_single_ordinary_command(
    tmp_path, monkeypatch
):
    calls, _ = _record_runs(monkeypatch)

    ssh_transport._rsync_mirror(tmp_path, "h:/w/", timeout_s=None)

    assert calls == [["rsync", "-az", f"{tmp_path}/", "h:/w/"]]


def test_a_host_without_rsync_gets_the_plain_mirror(tmp_path, monkeypatch):
    """scp cannot ask whether a file is already there, so the limit is stated, not hidden."""
    monkeypatch.setattr(ssh_transport.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(
        ssh_transport, "_run", lambda argv, *, timeout_s: calls.append(argv[0])
    )
    monkeypatch.setattr(ssh_transport, "run_remote", lambda *a, **k: None)
    (tmp_path / "x").mkdir()

    ssh_transport.copy_to_remote(
        "tillicum", tmp_path, "/work/exp", never_overwrite=[tmp_path / "x"]
    )

    assert calls == ["scp"]


# --- the fallback keeps the guarantees, or is not taken (run everywhere) ---------------------- #
_NO_RSYNC_SPELLINGS = [
    "bash: line 1: rsync: command not found",
    "sh: 1: rsync: not found",
    "zsh:1: command not found: rsync",
    "rsync: Command not found.",
]

_PASSING_FAILURES = [
    "mux_client_request_session: session request failed: Session open refused by peer\n"
    "kex_exchange_identification: read: Connection reset by peer",
    "Connection reset by 198.48.92.25 port 22\n"
    "rsync: connection unexpectedly closed (0 bytes received so far) [sender]",
    "rsync error: some files/attrs were not transferred (see previous errors) (code 23)",
    'rsync: [sender] link_stat "/exp/not found.txt" failed: No such file or directory (2)',
]


class _Result:
    def __init__(self, returncode=0, stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, "", stderr


def _host_answering(monkeypatch, rsync_returncode: int, rsync_stderr: str) -> list:
    """Every ``rsync`` exits ``rsync_returncode`` with ``rsync_stderr``; ssh and scp succeed."""
    programs = []

    def fake_run(argv, **kwargs):
        programs.append(argv[0])
        if argv[0] == "rsync":
            return _Result(rsync_returncode, rsync_stderr)
        return _Result(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ssh_transport, "_rsync_available", lambda: True)
    monkeypatch.setattr(ssh_transport, "require_responsive", lambda host: None)
    return programs


def _upload_with_a_run_in_flight(tmp_path: Path) -> None:
    """The shepherd's own upload: a run directory to leave alone and the envelope never sent."""
    run = tmp_path / RUN / "det_01"
    _write(tmp_path, f"{RUN}/det_01/job.json", "backend: remote")
    ssh_transport.copy_to_remote(
        "tillicum",
        tmp_path,
        "/work/exp",
        never_overwrite=[run],
        never_send=[rundir.JOB_FILENAME],
    )


@pytest.mark.parametrize("stderr", _PASSING_FAILURES)
def test_a_failed_rsync_on_a_host_that_has_rsync_is_raised_not_resent_with_scp(
    monkeypatch, tmp_path, stderr
):
    """The incident: a refused or dropped session failed the rsync, and the fallback re-sent the
    whole experiment with ``scp``, which laid each in-flight job's local ``job.json`` over the
    envelope its shepherd wrote on the cluster."""
    monkeypatch.setattr(ssh_transport, "_sleep", lambda s: None)
    programs = _host_answering(monkeypatch, 12, stderr)

    with pytest.raises(ssh_transport.RemoteCommandError):
        _upload_with_a_run_in_flight(tmp_path)

    assert "rsync" in programs
    assert "scp" not in programs


@pytest.mark.parametrize("stderr", _NO_RSYNC_SPELLINGS)
def test_a_host_with_no_rsync_still_gets_the_scp_mirror(monkeypatch, tmp_path, stderr):
    programs = _host_answering(
        monkeypatch,
        12,
        f"{stderr}\nrsync: connection unexpectedly closed (0 bytes received so far) "
        "[sender]",
    )

    _upload_with_a_run_in_flight(tmp_path)

    assert programs[-1] == "scp"


@pytest.mark.parametrize("stderr", _NO_RSYNC_SPELLINGS)
def test_each_shells_way_of_saying_it_has_no_rsync_is_recognised(stderr):
    error = ssh_transport.RemoteCommandError(["rsync"], 12, "", stderr)
    assert ssh_transport.rsync_missing_on_host(error)


@pytest.mark.parametrize("stderr", _PASSING_FAILURES)
def test_the_no_rsync_check_would_catch_a_passing_failure_read_as_missing(stderr):
    error = ssh_transport.RemoteCommandError(["rsync"], 12, "", stderr)
    assert not ssh_transport.rsync_missing_on_host(error)


def test_files_that_vanished_mid_upload_complete_the_mirror(monkeypatch, tmp_path):
    """Eight shepherds rewrite their envelopes through atomic temps while one uploads, so a temp
    rsync listed and then found gone (exit 24) is the ordinary case: both passes still run, and
    nothing falls back."""
    programs = _host_answering(
        monkeypatch,
        ssh_transport.RSYNC_VANISHED_EXIT,
        'file has vanished: "/exp/runs/run_011/progress.json.tmp"',
    )

    _upload_with_a_run_in_flight(tmp_path)

    assert programs.count("rsync") == 2
    assert "scp" not in programs
