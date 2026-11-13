"""A remote run keeps ``max_concurrent_jobs`` recordings on the cluster at once.

**The class guarded: a rule written for one local GPU applied to a cluster.** `recording_parallelism`
is 1 for detection and embedding so two jobs do not fight over one local GPU, and
`RunOrchestrator.run` used it for *every* run, so 80 recordings sent to Tillicum went one at a
time while Slurm had 48 GPUs free: each recording is its own Slurm job and needs nothing the
others hold. The rule is now the target's (`RemoteTargetConfig.max_concurrent_jobs`) for a remote
run, and the module's for a local one.

**It also had a second, quieter half.** `_orchestrate_parallel` had never run with a real module
(nothing overrides `recording_parallelism`), and the progress model it feeds assumes *one* current
recording: `ProcessingProgress.inner` names whichever recording reported last, so a bar over eight
jobs jumped between them and read a fraction that was a fraction of one. `live` carries every
recording in flight and the bar is `(done + Σ live fractions) / total`
(`dataclasses.processing_inflight_fraction`). Publishing is serialized so a thread that lost a race
cannot write an older snapshot over a newer one.

**These tests prove concurrency rather than assume it.** Each fake job parks until the expected
number are in flight together (a rendezvous with a timeout), so "three at once" is observed, not
inferred from a counter that a lucky schedule could satisfy.
"""

import logging
import threading

import pytest

from detail.api.experiment import Experiment, RunOrchestrator
from detail.api.recording import Recording
from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import (
    ComputeTarget,
    DetectionProgressEvent,
    processing_progress_from_dict,
    processing_progress_to_dict,
)
from detail.executor import ssh_transport
from detail.storage import remote_targets
from detail.storage import runs as runs_store


def _event(done: int, total: int) -> DetectionProgressEvent:
    return DetectionProgressEvent(
        parts_done=0,
        parts_total=1,
        frames_done=done,
        frames_total=total,
        elapsed_s=0.0,
        eta_s=None,
        message="detecting",
    )


class _Fleet:
    """The fake cluster: hands out job handles and watches how many are in flight at once."""

    def __init__(self, rendezvous: int, fail: frozenset[str] = frozenset()) -> None:
        self.rendezvous = rendezvous
        self.fail = fail
        self.in_flight = 0
        self.peak = 0
        self.launched: list[str] = []
        self._lock = threading.Lock()
        self._all_here = threading.Event()
        self.on_launch = lambda sid: None

    def start(self, sid: str) -> "_Job":
        with self._lock:
            self.launched.append(sid)
        self.on_launch(sid)
        return _Job(self, sid)

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            if self.in_flight >= self.rendezvous:
                self._all_here.set()
        # Park until the expected number are in flight together; a tail with fewer than that
        # left cannot meet it, so the wait times out and carries on rather than hanging.
        self._all_here.wait(timeout=0.5)

    def leave(self) -> None:
        with self._lock:
            self.in_flight -= 1


class _Job:
    def __init__(self, fleet: _Fleet, sid: str) -> None:
        self._fleet = fleet
        self.id = f"det_{sid[-3:]}"
        self.status = "failed" if sid in fleet.fail else "complete"
        self.error = "CUDA out of memory" if sid in fleet.fail else None
        self._sid = sid

    def progress(self):
        # Report once, *then* wait for the others: the bar can only carry a recording that has
        # reported, and a job that finished before its neighbours had started would never be
        # in flight with them as far as the bar is concerned.
        try:
            yield _event(1, 3)
            self._fleet.enter()
            yield _event(2, 3)
            yield _event(3, 3)
        finally:
            self._fleet.leave()

    def wait(self):
        return self


def _experiment(tmp_path, monkeypatch, n=7, *, max_jobs=3):
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg / "remote_targets.yaml"
    )
    remote_targets.register_remote_target(
        RemoteTargetConfig(
            name="tillicum",
            host="tillicum",
            remote_workdir="/scratch/x",
            max_concurrent_jobs=max_jobs,
        )
    )
    experiment = Experiment.create("p", tmp_path)
    ids = [f"recording_{i:03d}" for i in range(1, n + 1)]
    for sid in ids:
        experiment.create_recording(sid)
    return experiment, ids


def _orchestrate(experiment, ids, fleet, *, compute, emitted=None):
    monkeypatch_run = lambda self, mid, cfg: fleet.start(self.id)  # noqa: E731
    Recording.run = monkeypatch_run
    orchestrator = RunOrchestrator(
        experiment,
        "run_001",
        "detection",
        ids,
        {"compute": compute},
        (lambda p: emitted.append(p)) if emitted is not None else (lambda _p: None),
        "2026-01-01T00:00:00Z",
    )
    orchestrator.run()
    return orchestrator


@pytest.fixture(autouse=True)
def _restore_recording_run():
    original = Recording.run
    yield
    Recording.run = original


@pytest.fixture(autouse=True)
def _rsync_present(monkeypatch):
    """Whether this box has rsync is a fact about the box (a bare Windows machine does not), and
    the orchestrator rightly goes serial without it, so the tests state it rather than inherit it.
    The one test about its absence overrides this."""
    monkeypatch.setattr(ssh_transport, "rsync_available", lambda: True)


REMOTE = ComputeTarget("remote", None, None, None, remote_target="tillicum")


def _record(experiment) -> dict:
    return runs_store.read_run(experiment.dir, "run_001")


def test_a_remote_run_keeps_the_targets_number_of_recordings_in_flight(
    tmp_path, monkeypatch
):
    experiment, ids = _experiment(tmp_path, monkeypatch, n=7, max_jobs=3)
    fleet = _Fleet(rendezvous=3)

    _orchestrate(experiment, ids, fleet, compute=REMOTE)

    assert fleet.peak == 3, "observed three in flight together, and never more"
    record = _record(experiment)
    assert record["status"] == "complete"
    assert record["parallelism"] == 3
    assert sorted(fleet.launched) == ids


def test_more_slots_than_recordings_is_capped_at_the_recordings(tmp_path, monkeypatch):
    experiment, ids = _experiment(tmp_path, monkeypatch, n=2, max_jobs=8)
    fleet = _Fleet(rendezvous=2)

    _orchestrate(experiment, ids, fleet, compute=REMOTE)

    assert fleet.peak == 2
    assert _record(experiment)["parallelism"] == 2


@pytest.mark.parametrize(
    "compute, max_jobs",
    [
        (None, 3),  # a local run follows the module, which says 1 for detection
        (ComputeTarget("cpu", None, None, None), 3),
        (REMOTE, 1),  # a target set to one is serial, as before
    ],
)
def test_everything_else_stays_one_at_a_time(tmp_path, monkeypatch, compute, max_jobs):
    experiment, ids = _experiment(tmp_path, monkeypatch, n=4, max_jobs=max_jobs)
    fleet = _Fleet(rendezvous=2)

    _orchestrate(experiment, ids, fleet, compute=compute)

    assert fleet.peak == 1
    assert _record(experiment)["parallelism"] == 1


def test_without_rsync_it_goes_one_at_a_time_and_says_why(
    tmp_path, monkeypatch, caplog
):
    """Concurrent jobs rely on the upload leaving other jobs' run directories alone, which only
    the rsync path can do; the scp fallback would let one job's upload reset another's.
    """
    experiment, ids = _experiment(tmp_path, monkeypatch, n=4, max_jobs=4)
    monkeypatch.setattr(ssh_transport, "rsync_available", lambda: False)
    fleet = _Fleet(rendezvous=2)

    with caplog.at_level(logging.WARNING):
        _orchestrate(experiment, ids, fleet, compute=REMOTE)

    assert fleet.peak == 1
    assert "rsync is not installed" in caplog.text and "tillicum" in caplog.text
    assert _record(experiment)["parallelism"] == 1


def test_a_target_that_is_not_registered_is_left_to_the_run_to_report(
    tmp_path, monkeypatch
):
    experiment, ids = _experiment(tmp_path, monkeypatch, n=3)
    fleet = _Fleet(rendezvous=2)
    nowhere = ComputeTarget("remote", None, None, None, remote_target="nowhere")

    _orchestrate(experiment, ids, fleet, compute=nowhere)

    assert fleet.peak == 1


def test_the_published_bar_carries_every_recording_in_flight(tmp_path, monkeypatch):
    experiment, ids = _experiment(tmp_path, monkeypatch, n=6, max_jobs=3)
    fleet = _Fleet(rendezvous=3)
    emitted = []

    _orchestrate(experiment, ids, fleet, compute=REMOTE, emitted=emitted)

    widest = max(emitted, key=lambda p: len(p.live))
    assert len(widest.live) == 3 and len(set(widest.live)) == 3
    assert set(widest.live) <= set(ids)
    assert emitted[-1].live == {}, "nothing is in flight once the run is over"
    assert emitted[-1].recordings_done == 6


def test_the_published_count_of_finished_recordings_never_goes_backwards(
    tmp_path, monkeypatch
):
    """Snapshot and write are one step under a lock. Taken apart, a thread that lost the race
    publishes its older `recordings_done` over a newer one, and the bar steps back."""
    experiment, ids = _experiment(tmp_path, monkeypatch, n=12, max_jobs=6)
    fleet = _Fleet(rendezvous=2)
    emitted = []

    _orchestrate(experiment, ids, fleet, compute=REMOTE, emitted=emitted)

    done = [p.recordings_done for p in emitted]
    assert done == sorted(done), done


def test_every_snapshot_is_written_while_the_publish_lock_is_held(
    tmp_path, monkeypatch
):
    """The ordering test above can only pass by luck on a quiet machine; this is the property
    itself. A snapshot taken under the lock and written after releasing it is exactly the race:
    another thread's newer one can be written in between, and this older one lands on top.
    """
    experiment, ids = _experiment(tmp_path, monkeypatch, n=4, max_jobs=2)
    fleet = _Fleet(rendezvous=2)
    unlocked_writes = []
    holder = {"calls": 0}

    def sink(_progress):
        # The first publish is `run()`'s own, before any thread exists: a bar for a reattaching
        # client to find. Nothing can race it, so it takes no lock.
        if holder["calls"] and not holder["orchestrator"]._emit_lock.locked():
            unlocked_writes.append(_progress)
        holder["calls"] += 1

    Recording.run = lambda self, mid, cfg: fleet.start(self.id)
    orchestrator = RunOrchestrator(
        experiment,
        "run_001",
        "detection",
        ids,
        {"compute": REMOTE},
        sink,
        "2026-01-01T00:00:00Z",
    )
    holder["orchestrator"] = orchestrator
    orchestrator.run()

    assert unlocked_writes == []


def test_one_failed_recording_does_not_stop_the_others(tmp_path, monkeypatch):
    experiment, ids = _experiment(tmp_path, monkeypatch, n=6, max_jobs=3)
    fleet = _Fleet(rendezvous=2, fail=frozenset({ids[1]}))

    _orchestrate(experiment, ids, fleet, compute=REMOTE)

    per = _record(experiment)["per_recording"]
    assert {s: e["status"] for s, e in per.items()} == {
        sid: ("failed" if sid == ids[1] else "complete") for sid in ids
    }
    assert per[ids[1]]["error"] == "CUDA out of memory"
    assert _record(experiment)["status"] == "failed"


def test_a_cancel_stops_launching_further_recordings(tmp_path, monkeypatch):
    experiment, ids = _experiment(tmp_path, monkeypatch, n=12, max_jobs=2)
    fleet = _Fleet(rendezvous=1)
    holder = {}

    def cancel_after_three(sid):
        if len(fleet.launched) == 3:
            holder["orchestrator"]._cancelled.set()

    fleet.on_launch = cancel_after_three
    Recording.run = lambda self, mid, cfg: fleet.start(self.id)
    orchestrator = RunOrchestrator(
        experiment,
        "run_001",
        "detection",
        ids,
        {"compute": REMOTE},
        lambda _p: None,
        "2026-01-01T00:00:00Z",
    )
    holder["orchestrator"] = orchestrator

    orchestrator.run()

    assert len(fleet.launched) < len(ids), "it kept launching after the cancel"
    assert _record(experiment)["status"] == "cancelled"


def test_live_events_survive_the_progress_file_round_trip(tmp_path):
    from detail.core.dataclasses import ProcessingProgress

    original = ProcessingProgress(
        2,
        8,
        "recording_003",
        _event(1, 3),
        12.0,
        0,
        live={"recording_003": _event(1, 3), "recording_004": _event(2, 3)},
    )

    restored = processing_progress_from_dict(processing_progress_to_dict(original))

    assert restored == original


def test_inflight_fraction_is_one_recordings_own_for_a_serial_run_and_the_sum_for_several():
    from types import SimpleNamespace

    from detail.core.dataclasses import ProcessingProgress, processing_inflight_fraction

    serial = ProcessingProgress(0, 4, "r1", _event(1, 2))
    live_of_one = ProcessingProgress(
        0, 4, "r1", _event(1, 2), live={"r1": _event(1, 2)}
    )
    eight = ProcessingProgress(
        0, 20, "r1", _event(1, 2), live={f"r{i}": _event(1, 2) for i in range(8)}
    )
    nothing_yet = ProcessingProgress(0, 4, "r1", None)
    no_total_yet = ProcessingProgress(0, 4, "r1", _event(0, 0))
    a_stub_from_before_live_existed = SimpleNamespace(inner=_event(1, 4))

    assert processing_inflight_fraction(serial) == 0.5
    assert processing_inflight_fraction(live_of_one) == 0.5
    assert (
        processing_inflight_fraction(eight) == 4.0
    ), "eight at half way is four recordings"
    assert processing_inflight_fraction(nothing_yet) == 0.0
    assert processing_inflight_fraction(no_total_yet) == 0.0
    assert processing_inflight_fraction(a_stub_from_before_live_existed) == 0.25


def test_a_progress_file_written_before_live_existed_still_loads():
    """Every reattaching reader meets records like this: runs finished last month."""
    old = {
        "recordings_done": 1,
        "recordings_total": 4,
        "current_recording": "recording_002",
        "inner": None,
        "elapsed_s": 5.0,
        "recordings_at_start": 0,
    }

    assert processing_progress_from_dict(old).live == {}
