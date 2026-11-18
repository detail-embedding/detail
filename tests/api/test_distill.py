"""Tests for the durable distill run — label → train (§12).

**A default selection that includes what the operation refuses** (#35). "No recordings named"
meant every recording, so one metadata-only recording — which Add recordings allows — refused
the whole labeling or distill job, and named itself by id. It means every recording *with
video* now, resolved once in ``Experiment._sampling_recordings`` so the CLI and GUI agree; a
recording that is *named* and has no video is still refused, by its name. The GUI pickers grey
such a recording and say how many selecting nothing uses.
"""

import json
import types

import pytest

from detail.api.experiment import DistillOrchestrator, Experiment
from detail.core.dataclasses import (
    LabelingClass,
    LabelingProgressEvent,
    TrainingProgressEvent,
)
from detail.core.registries import TRAINERS
from detail.core.utils import default_compute
from detail.storage import distill_runs as distill_store


# --- fakes / helpers -------------------------------------------------------- #
class _FakeRun:
    """Stand-in for the DatasetRun / ModelRun handle the orchestrator drives."""

    def __init__(self, id, events, status="complete", error=None):
        self.id = id
        self._events = events
        self.status = status
        self.error = error

    def progress(self):
        return iter(self._events)

    def wait(self):
        return self


class _FakeTrainer:
    supported_backends = ["cpu"]

    def is_ready(self):
        return True

    def validate_hyperparams(self, hyperparams):
        return dict(hyperparams)


def _compute():
    return default_compute(["cpu"]).to_dict()


def _config(**over):
    cfg = {
        "name": "m",
        "base_model": "yolo11s-seg.pt",
        "trainer_id": "faketrainer",
        "classes": [{"class_id": 0, "name": "mouse", "prompt": "a mouse"}],
        "source_recording_ids": ["recording_001"],
        "labeling": {
            "model_id": "sam3",
            "model_variant": None,
            "n_frames": 4,
            "sampling": "stratified",
            "logit_threshold": 0.5,
            "min_contour_area": 100,
            "val_split": 0.15,
        },
        "training": {"epochs": 5, "batch_size": 2, "imgsz": 64, "patience": 5},
        "label_compute": _compute(),
        "train_compute": _compute(),
    }
    cfg.update(over)
    return cfg


def _reserve(
    experiment, run_id="distill_001", stages=None, status="running", resume=False
):
    record = {
        "id": run_id,
        "name": "m",
        "status": status,
        "created_at": "2026-01-01T00:00:00Z",
        "config": _config(),
        "stages": stages
        or {
            "label": {"status": "pending", "dataset_id": None},
            "train": {"status": "pending", "model_id": None},
        },
        "resume": resume,
    }
    distill_store.write_run(experiment.dir, run_id, record)
    return record


def _write_job_json(run_dir, status, pid=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": 1, "status": status, "job": {}, "error": None}
    if pid is not None:
        envelope["pid"] = pid
    (run_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")


# --- up-front validation (fail-fast, no models needed) ---------------------- #
def test_distill_requires_a_class(tmp_path):
    proj = Experiment.create("p", tmp_path)
    with pytest.raises(ValueError, match="at least one class"):
        proj.distill("m", classes=[], n_frames=4)


def test_distill_rejects_duplicate_prompts(tmp_path):
    proj = Experiment.create("p", tmp_path)
    classes = [LabelingClass(0, "a", "same"), LabelingClass(1, "b", "same")]
    with pytest.raises(ValueError, match="share a text prompt"):
        proj.distill("m", classes=classes, n_frames=4)


def test_distill_rejects_non_prompt_capable_label_model(tmp_path):
    proj = Experiment.create("p", tmp_path)
    with pytest.raises(ValueError, match="cannot label"):
        proj.distill(
            "m",
            classes=[LabelingClass(0, "mouse", "a mouse")],
            n_frames=4,
            label_model_id="not_a_prompt_model",
        )


# --- reserve + spawn (durable-run glue; validation stubbed, no models) ------ #
def _pass_validation(proj, monkeypatch):
    """Stub the model/trainer/frame gates so ``distill`` reaches reserve+spawn."""
    proj.create_recording("recording_001")
    import detail.storage.recording as rec_store

    monkeypatch.setattr("detail.api.models.prompt_capable_models", lambda: ["sam3"])
    monkeypatch.setattr("detail.api.models.supported_backends", lambda mid: ["cpu"])
    monkeypatch.setattr(
        proj,
        "check_model_access",
        lambda mid: types.SimpleNamespace(ready=True, message="ok"),
    )
    monkeypatch.setitem(TRAINERS, "faketrainer", _FakeTrainer)
    monkeypatch.setattr(
        rec_store,
        "read_recording",
        lambda d: {"name": "recording_001", "video": {"files": ["video/part_001.mp4"]}},
    )
    monkeypatch.setattr(
        rec_store, "load_active_timestamps", lambda d, data, i: list(range(100))
    )


def test_distill_reserves_record_then_spawns_the_distill_coordinator(
    tmp_path, monkeypatch
):
    proj = Experiment.create("p", tmp_path)
    _pass_validation(proj, monkeypatch)
    captured = {}

    def _fake_spawn(run_dir, runner_modules, entrypoint="detail.executor.coordinator"):
        captured["entrypoint"] = entrypoint
        return 4321

    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", _fake_spawn)
    proj.distill(
        "m",
        classes=[LabelingClass(0, "mouse", "a mouse")],
        n_frames=2,
        source_recording_ids=["recording_001"],
        trainer_id="faketrainer",
    )
    assert captured["entrypoint"] == "detail.executor.distill_coordinator"
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "running"
    assert rec["stages"]["label"]["status"] == "pending"
    assert rec["stages"]["train"]["status"] == "pending"
    assert rec["config"]["classes"][0]["prompt"] == "a mouse"
    assert distill_store.read_coordinator_pid(proj.dir, "distill_001") == 4321


def test_distill_spawn_failure_marks_run_failed(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    _pass_validation(proj, monkeypatch)

    def _boom(*_a, **_k):
        raise OSError("cannot fork")

    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", _boom)
    with pytest.raises(OSError):
        proj.distill(
            "m",
            classes=[LabelingClass(0, "mouse", "a mouse")],
            n_frames=2,
            source_recording_ids=["recording_001"],
            trainer_id="faketrainer",
        )
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "failed"
    assert "coordinator failed to launch" in (rec.get("error") or "")


# --- orchestrator: fold + phase order + failure + resume -------------------- #
def test_orchestrator_folds_progress_and_advances_phases(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    label_events = [
        LabelingProgressEvent(1, 4, 0, 1, 0.1, 5.0, "labeling frames"),
        LabelingProgressEvent(4, 4, 1, 1, 0.2, 0.0, None),
    ]
    train_events = [
        TrainingProgressEvent(1, 5, {}, 0.1, 4.0, None),
        TrainingProgressEvent(5, 5, {"box_map50": 0.9}, 0.5, 0.0, None),
    ]
    monkeypatch.setattr(
        proj, "label_dataset", lambda **kw: _FakeRun("dataset_001", label_events)
    )
    monkeypatch.setattr(
        proj, "train_model", lambda **kw: _FakeRun("model_001", train_events)
    )

    emitted = []
    DistillOrchestrator(proj, "distill_001", record, emitted.append).run()

    # Phases never interleave: stage_index is non-decreasing (labeling then training).
    assert [e.stage_index for e in emitted] == sorted(e.stage_index for e in emitted)
    messages = [e.message for e in emitted]
    assert "starting labeling" in messages  # forced phase-boundary emit
    assert "starting training" in messages
    assert messages.index("starting labeling") < messages.index("starting training")
    assert len(set(messages)) >= 3  # not frozen on one label
    # a real intermediate fraction with an ETA
    assert any(0.0 < e.overall_fraction < 1.0 and e.eta_s is not None for e in emitted)
    assert emitted[-1].overall_fraction == pytest.approx(1.0)

    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["label"] == {"status": "complete", "dataset_id": "dataset_001"}
    assert rec["stages"]["train"] == {"status": "complete", "model_id": "model_001"}


def test_orchestrator_fails_the_run_when_a_stage_fails(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    trained = []
    monkeypatch.setattr(
        proj,
        "label_dataset",
        lambda **kw: _FakeRun("dataset_001", [], status="failed", error="boom"),
    )
    monkeypatch.setattr(proj, "train_model", lambda **kw: trained.append(kw))

    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()

    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "failed"
    assert "labeling failed" in rec["error"]
    assert not trained  # training never started


def test_orchestrator_resume_skips_a_complete_dataset(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_007"},
            "train": {"status": "pending", "model_id": None},
        },
        resume=True,
    )

    def _label(**kw):
        raise AssertionError("labeling should be skipped on resume")

    train_calls = []

    def _train(**kw):
        train_calls.append(kw)
        return _FakeRun("model_002", [TrainingProgressEvent(5, 5, {}, 0.1, 0.0, None)])

    monkeypatch.setattr(proj, "label_dataset", _label)
    monkeypatch.setattr(proj, "train_model", _train)

    DistillOrchestrator(proj, "distill_001", record, lambda _p: None).run()

    assert train_calls and train_calls[0]["dataset_id"] == "dataset_007"
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["train"]["model_id"] == "model_002"


# --- reconcile on open (SIGKILL backstop) ----------------------------------- #
def test_reconcile_distill_adopts_terminal_stage_status(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "running", "model_id": "model_001"},
        },
    )
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    _write_job_json(proj.dir / "models" / "model_001", "complete")
    proj.reconcile_distill_runs()
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "complete"
    assert rec["stages"]["train"]["status"] == "complete"


def test_reconcile_distill_fails_dead_orphan(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    monkeypatch.setattr("detail.api.experiment.worker_alive", lambda _rd: False)
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "running", "model_id": "model_001"},
        },
    )
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    _write_job_json(proj.dir / "models" / "model_001", "running", pid=999999)
    proj.reconcile_distill_runs()
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "failed"
    assert rec["stages"]["train"]["status"] == "failed"
    assert "interrupted" in rec["stages"]["train"]["error"]


def test_reconcile_distill_leaves_live_coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr("detail.api.experiment.coordinator_alive", lambda _rd, _p: True)
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        stages={
            "label": {"status": "running", "dataset_id": "dataset_001"},
            "train": {"status": "pending", "model_id": None},
        },
    )
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    proj.reconcile_distill_runs()
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["status"] == "running"


# --- resume guards ---------------------------------------------------------- #
def test_resume_distill_refuses_a_running_run(tmp_path, monkeypatch):
    monkeypatch.setattr("detail.api.experiment.coordinator_alive", lambda _rd, _p: True)
    proj = Experiment.create("p", tmp_path)
    _reserve(proj)
    distill_store.set_coordinator_pid(proj.dir, "distill_001", 12345)
    with pytest.raises(ValueError, match="still running"):
        proj.resume_distill("distill_001")


def test_resume_distill_refuses_a_complete_run(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _reserve(
        proj,
        status="complete",
        stages={
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "complete", "model_id": "model_001"},
        },
    )
    with pytest.raises(ValueError, match="already complete"):
        proj.resume_distill("distill_001")


def test_delete_distill_missing_raises(tmp_path):
    proj = Experiment.create("p", tmp_path)
    with pytest.raises(FileNotFoundError):
        proj.delete_distill("distill_999")


# --- the ETA is the running stage's, not the weighted bar's ------------------ #
class _Clock:
    """A scripted monotonic clock, so an ETA assertion is exact instead of timing-dependent."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        """A no-op. **The script owns this clock; a poll must not add to it.**

        This used to do ``self.t += seconds``, with a comment saying it was never called on this
        path. It is called -- just not on a quiet box. `experiment.py`'s poll loops
        (`time.sleep(self._poll)`, three of them) only iterate when a run is not already finished,
        and the fake runs here finish immediately, so on an idle machine the comment was true. Under
        `-n auto` the loops do iterate, each one advancing this clock by a poll interval, and the
        scripted 1238s read as **1281s** -- 43 seconds of invented time (#574).

        That is why both ETA tests flaked at roughly 1 run in 14, and why checking "is sleep called?"
        in isolation said no. The clock is a script: only `_TimedRun.progress` moves it.
        """


class _TimedRun:
    """A fake run whose events advance the clock, the way a real stage's do."""

    def __init__(self, id, steps, clock, status="complete", error=None):
        self.id = id
        self._steps = steps  # (seconds_to_advance, event)
        self._clock = clock
        self.status = status
        self.error = error

    def progress(self):
        for advance, event in self._steps:
            self._clock.t += advance
            yield event

    def wait(self):
        return self


def _distill_at_epoch_14(proj, monkeypatch, *, train_eta):
    """Drive the orchestrator to the reported screenshot: ~370s of labeling, then epoch
    14 of 100 at ~62s per epoch. Returns the emitted events."""
    clock = _Clock()
    monkeypatch.setattr("detail.api.experiment.time", clock)
    record = _reserve(proj)
    label = _TimedRun(
        "dataset_001",
        [(370.0, LabelingProgressEvent(4, 4, 1, 1, 370.0, 0.0, None))],
        clock,
    )
    train = _TimedRun(
        "model_001",
        [(14 * 62.0, TrainingProgressEvent(14, 100, {}, 868.0, train_eta, None))],
        clock,
    )
    monkeypatch.setattr(proj, "label_dataset", lambda **kw: label)
    monkeypatch.setattr(proj, "train_model", lambda **kw: train)
    emitted = []
    DistillOrchestrator(proj, "distill_001", record, emitted.append).run()
    return emitted


def test_training_eta_tracks_the_real_remaining_time(tmp_path, monkeypatch):
    """The reported bug, with its own numbers.

    At epoch 14/100 with ~62s epochs, ~86 epochs (~89 min) remain. The fold used to feed
    the *nominal* stage weights into a time extrapolation — bar 0.45 + 0.55·0.14 = 0.527,
    ETA = 1238·(1−0.527)/0.527 ≈ 1111s — far short of the time actually left.
    """
    proj = Experiment.create("p", tmp_path)
    # The trainer computes an exact per-epoch estimate; the fold used to discard it.
    emitted = _distill_at_epoch_14(proj, monkeypatch, train_eta=5332.0)
    at_epoch = [e for e in emitted if e.phase == "training" and e.inner_fraction > 0]
    assert at_epoch, "no training progress was emitted"
    eta = at_epoch[-1].eta_s

    assert eta == pytest.approx(86 * 62.0, rel=0.02)  # ~89 min, the truth
    assert eta > 4000  # and emphatically not the old ~1111s (~18m31s)
    # The whole-run clock is still reported as elapsed — it is the ETA that must not use it.
    assert at_epoch[-1].elapsed_s == pytest.approx(1238.0)


def test_training_eta_falls_back_to_the_stage_clock_not_the_run_clock(
    tmp_path, monkeypatch
):
    """A runner that supplies no eta_s still must not be timed against the whole run.

    Same instant as above, so the two estimates are directly comparable: over the stage's
    own clock 868·(1−0.14)/0.14 ≈ 5332s (right); over the run clock with the weighted
    fraction 1238·(1−0.527)/0.527 ≈ 1111s (the bug).
    """
    proj = Experiment.create("p", tmp_path)
    emitted = _distill_at_epoch_14(proj, monkeypatch, train_eta=None)
    at_epoch = [e for e in emitted if e.phase == "training" and e.inner_fraction > 0]
    eta = at_epoch[-1].eta_s

    # **Which clock**, asserted as a comparison rather than a tolerance. That is the whole claim,
    # and the two candidates are 4.8x apart, so this can never be a close call:
    over_stage_clock = 868 * (1 - 0.14) / 0.14  # ~5332s, right
    over_run_clock = 1238 * (1 - 0.527) / 0.527  # ~1111s, the bug
    assert abs(eta - over_stage_clock) < abs(eta - over_run_clock), (
        f"eta {eta:.0f}s is nearer the run clock ({over_run_clock:.0f}s) than the stage clock "
        f"({over_stage_clock:.0f}s) — the fold is timing against the whole run again"
    )

    # And a sanity bound. A tight tolerance here was flaky under `-n auto` (#574), for a
    # cause not resolved: the sibling test above pins `elapsed_s` exactly and does not flake,
    # so the clock itself is faithful. An ETA is an estimate; the comparison above is what
    # this test asserts.
    assert eta == pytest.approx(over_stage_clock, rel=0.01)
    assert eta > 4000


def test_the_bar_still_uses_the_stage_weights(tmp_path, monkeypatch):
    """The weighted fold is deliberately kept for the *bar* — only the ETA stopped using
    it. Pinned so a later reader doesn't 'fix' the fraction to match the ETA."""
    proj = Experiment.create("p", tmp_path)
    emitted = _distill_at_epoch_14(proj, monkeypatch, train_eta=5332.0)
    at_epoch = [e for e in emitted if e.phase == "training" and e.inner_fraction > 0]
    assert at_epoch[-1].overall_fraction == pytest.approx(0.45 + 0.55 * 0.14)
    assert at_epoch[-1].inner_fraction == pytest.approx(0.14)


def test_a_resumed_distill_keeps_its_prior_elapsed(tmp_path, monkeypatch):
    """The bar keeps the prior attempt's progress on resume, so the clock has to keep its
    elapsed too — otherwise elapsed and the bar describe different runs (the distill twin of
    ProcessingProgress.recordings_at_start)."""
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    record["resume"] = True
    distill_store.write_progress(
        proj.dir,
        record["id"],
        {
            "phase": "training",
            "stage_index": 1,
            "stage_count": 2,
            "inner_fraction": 0.4,
            "overall_fraction": 0.67,
            "elapsed_s": 3600.0,  # an hour already spent
            "eta_s": None,
            "message": "epoch 40/100",
        },
    )

    clock = _Clock()
    monkeypatch.setattr("detail.api.experiment.time", clock)
    orchestrator = DistillOrchestrator(proj, record["id"], record, lambda e: None)

    # the clock is seeded back by the prior elapsed, so "now" is already an hour in
    assert orchestrator._start == pytest.approx(-3600.0)


def test_a_fresh_distill_starts_its_clock_at_zero(tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    record = _reserve(proj)
    clock = _Clock()
    monkeypatch.setattr("detail.api.experiment.time", clock)
    orchestrator = DistillOrchestrator(proj, record["id"], record, lambda e: None)
    assert orchestrator._start == pytest.approx(0.0)  # the identity for a fresh run


def test_distill_with_no_recordings_named_samples_those_with_video(
    tmp_path, monkeypatch
):
    """#35 finding 5, the Distill twin of `label_dataset`'s: "none = all" means every recording
    with video, so a metadata-only recording no longer refuses the whole distill."""
    from pathlib import Path

    import detail.storage.recording as rec_store

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_002")  # metadata only
    _pass_validation(proj, monkeypatch)  # recording_001, with video
    monkeypatch.setattr(
        rec_store,
        "read_recording",
        lambda d: {
            "name": Path(d).name,
            "video": {
                "files": (
                    ["video/part_001.mp4"] if Path(d).name == "recording_001" else []
                )
            },
        },
    )
    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", lambda *a, **k: 4321)
    proj.distill(
        "m",
        classes=[LabelingClass(0, "mouse", "a mouse")],
        n_frames=2,
        trainer_id="faketrainer",
    )
    rec = distill_store.read_run(proj.dir, "distill_001")
    assert rec["config"]["source_recording_ids"] == ["recording_001"]
