"""The opt-in background video upload (``executor/video_staging.py``).

**The class guarded first: a state file inside a tree that gets mirrored both ways is rewritten by
the stale copy of itself.** The shepherd mirrors the whole experiment directory to the cluster, and
used to mirror the whole remote one back over the local one when a job ended. A staging state file
kept in the experiment would be uploaded mid-run and copied back on top of the live one, rewinding a
running upload's progress and its pid to whatever they said when the job was submitted. Nothing
fails: the upload keeps going and the status just goes backwards. The copy back is now the job's run
folder only (``tests/executor/test_remote_shepherd.py``), which closes that path on its own;
``staging_dir`` is still asserted to sit **outside** the experiment tree, so the upload mirror never
carries a half-written state file either and nothing depends on the copy back staying narrow.

The rest is the contract the shepherd leans on: a staging process records honest progress and a
terminal status (including a failure it caught), ``start`` is idempotent while one is alive,
``wait_for_active`` waits only for a live run that covers the caller's recordings, and a failed or
stopped run never blocks a job (it uploads what is left itself).
"""

import os
import subprocess
from types import SimpleNamespace

import pytest

from detail.core import progress as core_progress
from detail.executor import local, remote_shepherd, video_staging


def _state(root, target="tillicum", **over):
    state = video_staging._initial_state(target, ["recording_001"])
    state.update(pid=4242, **over)
    video_staging._write_state(root, target, state)
    return state


@pytest.fixture
def alive(monkeypatch):
    """Whether ``local.coordinator_alive`` says the staging process is running."""
    flag = {"alive": True}
    monkeypatch.setattr(local, "coordinator_alive", lambda *_a: flag["alive"])
    return flag


def test_state_lives_outside_the_experiment_tree(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    state_dir = video_staging.staging_dir(root)

    assert root.resolve() not in [state_dir, *state_dir.parents]
    assert video_staging.state_path(root, "tillicum").parent == state_dir


def test_two_experiments_with_the_same_name_do_not_share_state(tmp_path):
    first, second = tmp_path / "a" / "exp", tmp_path / "b" / "exp"

    assert video_staging.staging_dir(first) != video_staging.staging_dir(second)


def test_nothing_recorded_reads_as_none_and_empty(tmp_path):
    assert video_staging.read_state(tmp_path / "exp", "tillicum") is None
    assert video_staging.all_states(tmp_path / "exp") == []


# --- the staging process itself ---------------------------------------------- #
def _fake_target(monkeypatch):
    from detail.storage import remote_targets

    monkeypatch.setattr(remote_targets, "get_remote_target", lambda name: name)


def test_main_records_progress_then_completes(tmp_path, monkeypatch):
    root = tmp_path / "exp"
    _fake_target(monkeypatch)
    monkeypatch.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)
    ticks = iter(range(0, 1000, 10))
    monkeypatch.setattr(
        video_staging,
        "time",
        SimpleNamespace(monotonic=lambda: float(next(ticks)), sleep=lambda *_: None),
    )
    seen = []

    def fake_upload(target, experiment_root, job, progress):
        assert job.source_recording_ids == ["recording_001", "recording_002"]
        progress.videos(0, 2, 0, 300, force=True)
        seen.append(video_staging.read_state(root, "tillicum"))
        progress.videos(1, 2, 100, 300, 50)
        seen.append(video_staging.read_state(root, "tillicum"))

    monkeypatch.setattr(remote_shepherd, "_sync_video_roots", fake_upload)
    video_staging._write_state(
        root,
        "tillicum",
        video_staging._initial_state("tillicum", ["recording_001", "recording_002"]),
    )

    code = video_staging.main([str(root), "tillicum", "recording_001", "recording_002"])

    assert code == 0
    first, mid = seen
    assert first["files_total"] == 2 and first["bytes_total"] == 300
    assert (mid["files_done"], mid["bytes_done"], mid["bytes_current"]) == (1, 100, 50)
    assert mid["eta_s"] is not None and mid["eta_s"] > 0, "no measured ETA mid-upload"
    assert mid["message"].startswith("uploading videos · 1/2 files")
    final = video_staging.read_state(root, "tillicum")
    assert final["status"] == "complete" and final["error"] is None
    assert final["pid"] == os.getpid() and final["eta_s"] is None


def test_main_records_a_failed_upload_instead_of_raising(tmp_path, monkeypatch, capsys):
    root = tmp_path / "exp"
    _fake_target(monkeypatch)

    def fake_upload(*_a):
        raise RuntimeError("disk quota exceeded")

    monkeypatch.setattr(remote_shepherd, "_sync_video_roots", fake_upload)

    code = video_staging.main([str(root), "tillicum", "recording_001"])

    assert code == 1
    final = video_staging.read_state(root, "tillicum")
    assert final["status"] == "failed"
    assert "disk quota exceeded" in final["error"]
    assert "disk quota exceeded" in capsys.readouterr().err, "the log must say why"


def test_main_needs_a_target_and_a_recording(tmp_path):
    with pytest.raises(SystemExit, match="usage"):
        video_staging.main([str(tmp_path), "tillicum"])


# --- starting, stopping --------------------------------------------------------- #
def _fake_popen(monkeypatch):
    launched = []

    class _Proc:
        def __init__(self, argv, **kwargs):
            launched.append((argv, kwargs))
            self.pid = 4242 + len(launched)

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _Proc)
    monkeypatch.setattr(video_staging, "_SPAWNED", [])
    return launched


def test_start_launches_a_detached_process_and_records_its_pid(
    tmp_path, monkeypatch, alive
):
    root = tmp_path / "exp"
    launched = _fake_popen(monkeypatch)

    state = video_staging.start(root, "tillicum", ["recording_001", "recording_002"])

    ((argv, kwargs),) = launched
    assert argv[1:3] == ["-m", "detail.executor.video_staging"]
    assert argv[3:] == [
        str(root.resolve()),
        "tillicum",
        "recording_001",
        "recording_002",
    ]
    assert "creationflags" in kwargs or "start_new_session" in kwargs, "not detached"
    assert state["pid"] == 4243 and state["status"] == "running"
    assert video_staging.read_state(root, "tillicum")["pid"] == 4243


def test_start_is_idempotent_while_one_is_running(tmp_path, monkeypatch, alive):
    root = tmp_path / "exp"
    launched = _fake_popen(monkeypatch)
    first = video_staging.start(root, "tillicum", ["recording_001"])

    again = video_staging.start(root, "tillicum", ["recording_001"])

    assert len(launched) == 1, "a second upload would copy the same bytes twice at once"
    assert again["pid"] == first["pid"]


def test_start_replaces_a_run_whose_process_is_gone(tmp_path, monkeypatch, alive):
    root = tmp_path / "exp"
    launched = _fake_popen(monkeypatch)
    video_staging.start(root, "tillicum", ["recording_001"])
    alive["alive"] = False

    video_staging.start(root, "tillicum", ["recording_001"])

    assert len(launched) == 2


def test_stop_kills_the_process_and_records_it_cancelled(tmp_path, monkeypatch, alive):
    root = tmp_path / "exp"
    _state(root)
    killed = []
    monkeypatch.setattr(local, "kill_coordinator", lambda r, pid: killed.append(pid))

    assert video_staging.stop(root, "tillicum") is True

    assert killed == [4242]
    assert video_staging.read_state(root, "tillicum")["status"] == "cancelled"


def test_stop_with_nothing_running_is_a_noop(tmp_path, monkeypatch, alive):
    root = tmp_path / "exp"
    _state(root, status="complete")
    monkeypatch.setattr(
        local, "kill_coordinator", lambda *_a: pytest.fail("killed a finished run")
    )

    assert video_staging.stop(root, "tillicum") is False
    assert video_staging.stop(root, "never_staged") is False


# --- the wait the shepherd does ---------------------------------------------------- #
def _wait(root, ids, *, ticks, alive, cancelled=lambda: False):
    """Run ``wait_for_active``; ``ticks`` are applied one per poll to the recorded state."""
    waited_on = []
    queue = list(ticks)

    def sleep(_s):
        update = queue.pop(0)
        if update == "died":
            alive["alive"] = False
        else:
            state = video_staging.read_state(root, "tillicum")
            video_staging._write_state(root, "tillicum", {**state, **update})

    result = video_staging.wait_for_active(
        root,
        "tillicum",
        ids,
        waited_on.append,
        sleep=sleep,
        cancelled=cancelled,
    )
    return result, waited_on


def test_wait_returns_none_at_once_when_nothing_is_staging(tmp_path, alive):
    result, waited_on = _wait(
        tmp_path / "exp", ["recording_001"], ticks=[], alive=alive
    )

    assert result is None and waited_on == []


def test_wait_follows_a_live_upload_to_its_end(tmp_path, alive):
    root = tmp_path / "exp"
    _state(root, files_done=0, files_total=2, message="uploading videos · 0/2 files")

    result, waited_on = _wait(
        root,
        ["recording_001"],
        ticks=[{"files_done": 1}, {"files_done": 2, "status": "complete"}],
        alive=alive,
    )

    assert [s["files_done"] for s in waited_on] == [0, 1]
    assert result["status"] == "complete", "the caller needs the run's final status"


def test_wait_does_not_wait_for_an_upload_of_other_recordings(tmp_path, alive):
    root = tmp_path / "exp"
    _state(root)

    result, waited_on = _wait(root, ["recording_009"], ticks=[], alive=alive)

    assert result is None and waited_on == []


def test_wait_does_not_wait_on_a_process_that_died_mid_upload(tmp_path, alive):
    """The record still says ``running`` after a kill -9; waiting on it would hang the job forever."""
    root = tmp_path / "exp"
    _state(root)
    alive["alive"] = False

    result, waited_on = _wait(root, ["recording_001"], ticks=[], alive=alive)

    assert result is None and waited_on == []


def test_wait_stops_waiting_when_the_process_dies_while_waiting(tmp_path, alive):
    root = tmp_path / "exp"
    _state(root)

    result, waited_on = _wait(root, ["recording_001"], ticks=["died"], alive=alive)

    assert len(waited_on) == 1
    assert (
        result["status"] == "running"
    ), "the stale record is what the waiter reports on"


def test_the_staging_process_never_waits_on_itself(tmp_path, alive):
    root = tmp_path / "exp"
    _state(root)
    state = video_staging.read_state(root, "tillicum")
    video_staging._write_state(root, "tillicum", {**state, "pid": os.getpid()})

    result, _ = _wait(root, ["recording_001"], ticks=[], alive=alive)

    assert result is None


def test_a_cancelled_job_does_not_wait_out_the_upload(tmp_path, alive):
    root = tmp_path / "exp"
    _state(root)

    result, waited_on = _wait(
        root, ["recording_001"], ticks=[], alive=alive, cancelled=lambda: True
    )

    assert result["status"] == "running" and waited_on == []


def test_a_cancel_with_no_upload_to_wait_on_changes_nothing(tmp_path, alive):
    """The caller reads a non-``None`` return as "I was waiting and got cancelled"; a cancel
    that arrives when nothing is staging must not read as that."""
    result, _ = _wait(
        tmp_path / "exp",
        ["recording_001"],
        ticks=[],
        alive=alive,
        cancelled=lambda: True,
    )

    assert result is None
