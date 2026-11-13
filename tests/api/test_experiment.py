import contextlib
import json
import threading

import pytest

from detail.api.experiment import Experiment
from detail.configs.tracking import DEFAULT_TRACKER, TrackingConfig
from detail.core.dataclasses import (
    ComputeTarget,
    FrameResult,
    ModelReadiness,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.core.protocols import DetectionModel, ModelNotReadyError
from detail.core.registries import MODELS, register_model
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import runs as runs_store
from detail.storage.metadata import ACQUISITION_SETUP_KEY as SETUP


def _fake_model(readiness: ModelReadiness):
    class _Fake(DetectionModel):
        model_id = "fake_dl"
        supported_cue_types = [TextCue]
        supported_backends = ["cpu"]
        installed = False

        def load(self, device):
            pass

        def readiness(self):
            return readiness

        def install(self, progress_cb=None):
            if not self.readiness().ready:
                raise ModelNotReadyError(self.readiness())
            type(self).installed = True

        def predict(self, frame, prompt):
            return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=[])

        def predict_batch(self, frames, prompts):
            return []

    return _Fake


@contextlib.contextmanager
def _registered(model_cls):
    register_model("fake_dl")(model_cls)
    try:
        yield model_cls
    finally:
        MODELS.pop("fake_dl", None)


_READY = ModelReadiness(True, "ok", "", None, "facebook/sam3")
_BLOCKED = ModelReadiness(
    False, "no_token", "need a token", "https://huggingface.co/settings/tokens", None
)


def test_check_model_access_returns_readiness(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with _registered(_fake_model(_BLOCKED)):
        r = proj.check_model_access("fake_dl")
    assert r.reason == "no_token"
    assert r.action_url.endswith("/settings/tokens")


def test_install_model_delegates_when_ready(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with _registered(_fake_model(_READY)) as cls:
        proj.install_model("fake_dl")
        assert cls.installed is True


def test_install_model_raises_when_blocked(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with _registered(_fake_model(_BLOCKED)):
        with pytest.raises(ModelNotReadyError) as exc:
            proj.install_model("fake_dl")
    assert exc.value.action_url.endswith("/settings/tokens")


def test_unknown_model_raises(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with pytest.raises(ValueError, match="unknown model_id"):
        proj.check_model_access("nope")


def test_set_hf_token_delegates(tmp_path, monkeypatch):
    from detail.storage import credentials

    monkeypatch.setattr(credentials, "set_hf_token", lambda token: "alice")
    proj = Experiment.create("proj", tmp_path)
    assert proj.set_hf_token("hf_x") == "alice"


def test_create_and_open_experiment(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    assert proj.dir == tmp_path / "proj"
    assert (proj.dir / "config.yaml").is_file()
    for sub in ("recordings", "analysis", "models", "datasets"):
        assert (proj.dir / sub).is_dir()

    reopened = Experiment.open(proj.dir)
    assert reopened.dir == proj.dir


def test_create_get_and_list_recordings(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    s = proj.create_recording("recording_001")
    assert s.id == "recording_001"
    assert s.name == "recording_001"
    assert (s.dir / "recording.yaml").is_file()

    got = proj.get_recording("recording_001")
    assert got.id == "recording_001"
    assert [x.id for x in proj.recordings] == ["recording_001"]


def test_name_description_and_edit(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    assert proj.name == "proj"
    assert proj.description == ""
    proj.set_metadata(name="renamed", description="cohort X")
    assert proj.name == "renamed"
    assert proj.description == "cohort X"
    assert proj.dir.name == "proj"  # display name decoupled from the folder


def test_list_all_and_create_in_managed_workspace(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "DetailExperiments")

    assert Experiment.list_all() == []
    experiment = Experiment.create("alpha")  # parent_dir defaults to the managed root
    assert experiment.dir == tmp_path / "DetailExperiments" / "alpha"
    assert [p.name for p in Experiment.list_all()] == ["alpha"]


def test_modified_at_is_recent(tmp_path):
    import time

    experiment = Experiment.create("proj", tmp_path)
    assert 0 < experiment.modified_at <= time.time() + 1


# --- run_processing (experiment-level batch over a recording selection, §15) ----- #
def _video_recording(experiment, sid, n_frames=10, w=32, h=24, fps=30.0):
    """Add a recording with one synthetic video part (no PyAV, no model)."""
    import cv2
    import numpy as np

    from detail.storage import recording as recording_store

    experiment.create_recording(sid)
    sdir = experiment.get_recording(sid).dir
    vid = sdir / "video" / "part_001.mp4"
    vid.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        vw.write(np.full((h, w, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()
    cap = cv2.VideoCapture(str(vid))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    recording_store.write_timestamps_csv(
        sdir / "video/timestamps/generated/part_001.csv",
        np.arange(n, dtype=np.int64) * int(1e9 / fps),
    )
    data = recording_store.read_recording(sdir)
    data["video"] = {
        "storage": "default",
        "fps_container": fps,
        "fps_actual": fps,
        "fps_source": "container",
        "resolution": [w, h],
        "total_duration_s": n / fps,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    }
    recording_store.write_recording(sdir, data)
    _detection_run(sdir, n_frames=n)


def _detection_run(sdir, *, n_frames=10, run_id="det_01"):
    """A completed detection run: one steady box per frame, written straight to zarr.

    The §15 orchestration tests need a real upstream that costs nothing — no model, no
    decode — so tracking can be the module under test.
    """
    import numpy as np

    from detail.storage import recording as recording_store
    from detail.storage.zarr_io import DetectionPartWriter

    det_dir = sdir / "detections" / run_id
    det_dir.mkdir(parents=True, exist_ok=True)
    with DetectionPartWriter(det_dir / "part_001.zarr") as writer:
        for frame in range(n_frames):
            writer.add_row(
                frame, frame * 1000, 0, "mouse", 0.9, (0.4, 0.4, 0.6, 0.6), ""
            )
    (det_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "mask_resolution": [24, 32],
                "job": {"type": "detection", "every_n_frames": 1},
            }
        ),
        encoding="utf-8",
    )
    data = recording_store.read_recording(sdir)
    data["detection_runs"] = [
        {
            "id": run_id,
            "status": "complete",
            "output": f"detections/{run_id}/",
            "model_id": "fake",
            "conf_threshold": 0.5,
            "every_n_frames": 1,
        }
    ]
    recording_store.write_recording(sdir, data)


TRACKING_CONFIG = {"detection_run_id": "det_01", "tracker_id": "anchor_tracker"}


def test_run_processing_unknown_module(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    proj.create_recording("recording_001")
    with pytest.raises(ValueError, match="unknown module_id"):
        proj.run_processing("nope", ["recording_001"], {})


def test_run_processing_empty_recordings(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        proj.run_processing("tracking", [], dict(TRACKING_CONFIG))


def test_run_processing_validates_upfront(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    proj.create_recording("recording_001")  # no detection run to track
    with pytest.raises(ValueError, match="not found in recording"):
        proj.run_processing("tracking", ["recording_001"], dict(TRACKING_CONFIG))
    assert proj.processing_runs == []  # nothing recorded before validation (Tenet 10)


def test_run_processing_batch(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    _video_recording(proj, "recording_002")
    run = proj.run_processing(
        "tracking", ["recording_001", "recording_002"], dict(TRACKING_CONFIG)
    )
    run.wait()
    assert run.status == "complete"

    runs = proj.processing_runs
    assert len(runs) == 1
    rec = runs[0]
    assert rec["module_id"] == "tracking"
    assert rec["recording_ids"] == ["recording_001", "recording_002"]
    assert set(rec["per_recording"]) == {"recording_001", "recording_002"}
    assert all(v["status"] == "complete" for v in rec["per_recording"].values())
    for sid in ("recording_001", "recording_002"):
        rid = rec["per_recording"][sid]["run_id"]
        assert (proj.get_recording(sid).dir / "tracking" / rid / "tracks.zarr").is_dir()
        # the per-recording run records the experiment run that launched it + a date,
        # so the GUI can map run_NNN <-> the per-recording id (run labels)
        stage = {r["id"]: r for r in proj.get_recording(sid).tracking_runs}[rid]
        assert stage["processing_run_id"] == run.id
        assert stage.get("created_at")


def test_a_finished_run_leaves_no_coordinator_behind(tmp_path):
    """The class (#543): **a detached child whose handle is dropped while it still runs.** A
    coordinator writes the run's terminal status and then exits, so `wait()` returns while the
    process is on its way out — and its handle was only pruned at the *next* spawn. The last run
    of a session left a live coordinator at interpreter exit: `ResourceWarning: subprocess … is
    still running`, fired inside `Popen.__del__` where `filterwarnings = error` cannot see it,
    so the run exited 0 with five of them per parallel slow tier.

    Asserted on the retained handle rather than on the process: whether the coordinator has
    exited by the time the test looks is a race, while "nothing still holds it" is not.
    """
    from detail.executor import local
    from detail.storage import runs as runs_store

    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    run = proj.run_processing("tracking", ["recording_001"], dict(TRACKING_CONFIG))
    run.wait()
    pid = runs_store.read_coordinator_pid(proj.dir, run.id)

    assert pid is not None, "the run recorded no coordinator, so this checks nothing"
    held = [p for p in local._SPAWNED_COORDINATORS if p.pid == pid]
    assert not held, (
        f"the finished run's coordinator (pid {pid}) is still held unreaped, so it is dropped "
        "while running when the process exits"
    )


def test_a_batch_run_records_the_defaults_it_was_launched_with(tmp_path):
    """The stored config names every setting, including the ones nobody typed.

    This record is what **Start from** re-seeds a form with and what ``retry_run``
    re-validates, so a defaulted key left out of it does not mean "unset" — it means
    *whatever the default is on the day it is read back*. Changing a default would then
    re-run an existing run differently, and nothing would say so.

    Unlike the per-recording record, this one keeps the caller's **unresolved** config: a
    batch spans recordings whose arenas and detection runs are each their own, so only the
    model's own defaults are filled in, never one recording's resolution.
    """
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    run = proj.run_processing(
        "tracking", ["recording_001"], {"detection_run_id": "det_01"}
    )
    run.wait()

    stored = proj.processing_runs[0]["config"]
    assert stored["tracker_id"] == DEFAULT_TRACKER
    missing = [
        name
        for name, field in TrackingConfig.model_fields.items()
        if not field.is_required() and name not in stored
    ]
    assert not missing, f"settings the record does not name: {missing}"


def test_run_processing_parallel(tmp_path, monkeypatch):
    # force parallelism > 1 so the orchestrator uses the concurrent pool; every
    # recording must still complete and the record must preserve input order.
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "3")
    proj = Experiment.create("proj", tmp_path)
    sids = ["recording_001", "recording_002", "recording_003"]
    for sid in sids:
        _video_recording(proj, sid)
    run = proj.run_processing("tracking", sids, dict(TRACKING_CONFIG))
    run.wait()
    assert run.status == "complete"
    rec = proj.processing_runs[0]
    assert rec["recording_ids"] == sids  # input order preserved despite concurrency
    assert all(v["status"] == "complete" for v in rec["per_recording"].values())
    for sid in sids:
        rid = rec["per_recording"][sid]["run_id"]
        assert (proj.get_recording(sid).dir / "tracking" / rid / "tracks.zarr").is_dir()
    # § run location, duration, cost: the aggregate record gets its own wall-clock
    # duration, and tracking (no ComputeTarget) shows as local — a real run through
    # the full RunOrchestrator, not a synthetic envelope.
    assert rec["duration_s"] is not None and rec["duration_s"] >= 0
    assert rec["compute"] is None


def test_typed_config_restores_flattened_typed_fields():
    # A run record stores config as JSON — _jsonable flattens compute/prompt to
    # dicts. The coordinator restores the typed objects before Recording.run (its
    # inverse). Regression: a dict compute -> "dict object has no attribute backend".
    from detail.api.experiment import _jsonable, _typed_config
    from detail.core.dataclasses import (
        ComputeTarget,
        Prompt,
        PromptTarget,
        TextCue,
    )

    original = {
        "model_id": "sam3",
        "compute": ComputeTarget("cuda", [0], None, None),
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "conf_threshold": 0.5,
        "hyperparams": {"iou": 0.5},  # a genuine plain-dict field — must stay a dict
    }
    restored = _typed_config(_jsonable(original))
    assert isinstance(restored["compute"], ComputeTarget)
    assert restored["compute"].device_indices == [0]
    assert isinstance(restored["prompt"], Prompt)
    assert restored["prompt"].targets[0].class_name == "mouse"
    # non-typed fields pass through untouched
    assert restored["model_id"] == "sam3"
    assert restored["conf_threshold"] == 0.5
    assert restored["hyperparams"] == {"iou": 0.5}

    # safe on a config that never round-tripped: absent keys and already-typed
    # values are left alone (no double from_dict).
    assert _typed_config({}) == {}
    typed = {"compute": ComputeTarget("cpu", None, None, None)}
    assert _typed_config(typed)["compute"] is typed["compute"]


def test_coordinator_restores_typed_config_before_recording_run(tmp_path, monkeypatch):
    # The detached coordinator builds a RunOrchestrator from the flattened run
    # record and re-runs each recording. The config reaching Recording.run must carry
    # typed compute/prompt — detection's validator reads compute.backend /
    # prompt.targets. Regression (durable coordinator): it received dicts and
    # raised "dict object has no attribute backend".
    from detail.api.experiment import RunOrchestrator, _jsonable
    from detail.api.recording import Recording
    from detail.core.dataclasses import (
        ComputeTarget,
        Prompt,
        PromptTarget,
        TextCue,
    )

    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")

    flattened = _jsonable(
        {
            "model_id": "fake_dl",
            "compute": ComputeTarget("cpu", None, None, None),
            "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
            "conf_threshold": 0.5,
        }
    )
    assert isinstance(flattened["compute"], dict)  # the record really is plain JSON
    assert isinstance(flattened["prompt"], dict)

    orch = RunOrchestrator(
        proj, "run_001", "detection", ["recording_001"], flattened, lambda p: None, ""
    )

    captured = {}

    def _capture(self, module_id, config):
        captured["config"] = config
        raise RuntimeError("stop before spawning a worker")

    monkeypatch.setattr(Recording, "run", _capture)
    orch._run_recording("recording_001")  # isolation boundary records, does not raise

    cfg = captured["config"]
    assert isinstance(cfg["compute"], ComputeTarget)
    assert cfg["compute"].backend == "cpu"
    assert isinstance(cfg["prompt"], Prompt)
    assert cfg["prompt"].targets[0].class_name == "mouse"


def _spec_det(rid, **over):
    """A detection run record with the identity fields a spec needs (no zarr)."""
    return {
        "id": rid,
        "status": "complete",
        "output": f"detections/{rid}/",
        "created_at": "t",
        "model_id": "sam3",
        "model_variant": "large",
        "weights_source": "pretrained",
        "conf_threshold": 0.25,
        "class_conf_thresholds_fingerprint": "ccf",
        "prompt_fingerprint": "fp",
        "prompt_classes": ["mouse"],
        **over,
    }


def _recording_with_detections(proj, sid, det_records):
    from detail.storage import recording as recording_store

    proj.create_recording(sid)
    sdir = proj.get_recording(sid).dir
    data = recording_store.read_recording(sdir)
    data["detection_runs"] = det_records
    recording_store.write_recording(sdir, data)


def test_coordinator_resolves_detection_spec_per_recording(tmp_path, monkeypatch):
    """The durable path: a batch stores the detection SPEC, and the coordinator re-runs
    each recording resolving it to *that recording's own* detection run — even at divergent
    ordinals — without mutating the shared config (§6.5). Regression guard for the
    per-recording isolation (``dict(self._config)``) the whole mechanism rests on."""
    from pathlib import Path

    from detail.api import run_spec
    from detail.api.experiment import RunOrchestrator, _jsonable
    from detail.executor.runners import PROCESSING_MODULES

    proj = Experiment.create("p", tmp_path)
    # Same "mouse @ 0.25" kind lands at det_02 in s1 (det_01 is a different-conf
    # throwaway) and at det_01 in s2 — the exact divergent-ordinal case.
    _recording_with_detections(
        proj, "s1", [_spec_det("det_01", conf_threshold=0.9), _spec_det("det_02")]
    )
    _recording_with_detections(proj, "s2", [_spec_det("det_01")])
    spec = run_spec.spec_from_run(proj.get_recording("s2"), "detection", "det_01")
    config = _jsonable(
        {
            "detection_spec": spec,
            "tracker_id": "anchor_tracker",
            "hyperparams": {},
            "arena_ids": None,
        }
    )

    captured = {}

    def _capture(self, recording_dir, data, cfg):  # runs *after* resolve_upstream_spec
        captured[Path(recording_dir).name] = cfg.get("detection_run_id")
        raise RuntimeError("stop before build_job")

    monkeypatch.setattr(type(PROCESSING_MODULES["tracking"]), "validate", _capture)
    orch = RunOrchestrator(
        proj, "run_001", "tracking", ["s1", "s2"], config, lambda p: None, ""
    )
    orch._run_recording("s1")  # isolation boundary: records, does not raise
    orch._run_recording("s2")

    assert captured == {"s1": "det_02", "s2": "det_01"}  # each resolved its own ordinal
    # the shared config still carries the spec — never mutated across recordings
    assert "detection_spec" in orch._config
    assert "detection_run_id" not in orch._config


def test_coordinator_isolates_recording_missing_the_spec(tmp_path):
    """A recording with no run matching the spec (e.g. its upstream deleted mid-flight)
    is recorded ``failed`` at the per-recording isolation boundary, not crashed (§6.5).
    """
    from detail.api import run_spec
    from detail.api.experiment import RunOrchestrator, _jsonable

    proj = Experiment.create("p", tmp_path)
    _recording_with_detections(proj, "s2", [])  # no detection runs → spec won't resolve
    spec = run_spec.spec_of({}, "detection", _spec_det("det_01"))
    config = _jsonable(
        {
            "detection_spec": spec,
            "tracker_id": "anchor_tracker",
            "hyperparams": {},
            "arena_ids": None,
        }
    )
    orch = RunOrchestrator(
        proj, "run_001", "tracking", ["s2"], config, lambda p: None, ""
    )
    orch._run_recording("s2")
    assert orch._per_recording["s2"]["status"] == "failed"
    assert "matching the selected spec" in orch._per_recording["s2"]["error"]


def test_run_processing_aborts_when_a_recording_lacks_the_spec(tmp_path):
    """CLI/API batch (no GUI narrowing): the up-front pre-validate loop fails the whole
    run if any selected recording has no matching upstream — before any coordinator is
    spawned (§6.5)."""
    from detail.api import run_spec

    proj = Experiment.create("p", tmp_path)
    _recording_with_detections(proj, "s1", [_spec_det("det_01")])  # has the kind
    _recording_with_detections(
        proj, "s2", [_spec_det("det_01", conf_threshold=0.9)]
    )  # different kind
    spec = run_spec.spec_from_run(proj.get_recording("s1"), "detection", "det_01")
    with pytest.raises(ValueError, match="matching the selected spec"):
        proj.run_processing(
            "tracking",
            ["s1", "s2"],
            {
                "detection_spec": spec,
                "tracker_id": "anchor_tracker",
                "hyperparams": {},
                "arena_ids": None,
            },
        )


def test_processing_progress_serializer_round_trip():
    from detail.core.dataclasses import (
        ProcessingProgress,
        TrackingProgressEvent,
        processing_progress_from_dict,
        processing_progress_to_dict,
    )

    inner = TrackingProgressEvent(
        parts_done=0,
        parts_total=1,
        frames_done=3,
        frames_total=10,
        n_active_tracks=2,
        elapsed_s=1.5,
        eta_s=3.5,
        message="tracking",
    )
    p = ProcessingProgress(1, 3, "recording_002", inner, elapsed_s=42.0)
    back = processing_progress_from_dict(processing_progress_to_dict(p))
    assert (back.recordings_done, back.recordings_total, back.current_recording) == (
        1,
        3,
        "recording_002",
    )
    assert back.elapsed_s == 42.0
    assert back.inner.frames_done == 3 and back.inner.message == "tracking"
    # inner=None round-trips too (before the first recording's first event)
    none_back = processing_progress_from_dict(
        processing_progress_to_dict(ProcessingProgress(0, 3, "s1", None, 0.0))
    )
    assert none_back.inner is None


def test_runs_store_progress_round_trip(tmp_path):
    proj = Experiment.create("p", tmp_path)
    assert runs_store.read_progress(proj.dir, "run_001") is None  # absent yet
    runs_store.write_progress(proj.dir, "run_001", {"recordings_done": 2, "x": [1, 2]})
    assert runs_store.read_progress(proj.dir, "run_001") == {
        "recordings_done": 2,
        "x": [1, 2],
    }
    assert runs_store.read_coordinator_pid(proj.dir, "run_001") is None
    runs_store.set_coordinator_pid(proj.dir, "run_001", 4321)
    assert runs_store.read_coordinator_pid(proj.dir, "run_001") == 4321


def test_run_processing_survives_launcher_exit(tmp_path):
    # Durability: a launcher process starts the run and exits WITHOUT waiting; the
    # detached coordinator must still finish it, and a fresh process reattaches.
    import subprocess as sp
    import sys as _sys
    import textwrap

    proj = Experiment.create("proj", tmp_path)
    for sid in ("recording_001", "recording_002"):
        _video_recording(proj, sid)
    launcher = textwrap.dedent(f"""
        from detail.api.experiment import Experiment
        p = Experiment.open({str(proj.dir)!r})
        p.run_processing(
            "tracking", ["recording_001", "recording_002"],
            {dict(TRACKING_CONFIG)},
        )  # return immediately — do not wait; the coordinator is detached
        """)
    sp.run([_sys.executable, "-c", launcher], check=True, timeout=60)

    runs = proj.processing_runs
    assert len(runs) == 1
    handle = proj.attach_run(runs[0]["id"])  # a fresh process reattaching to the run
    handle.wait()
    assert handle.status == "complete"
    for sid in ("recording_001", "recording_002"):
        rid = proj.processing_runs[0]["per_recording"][sid]["run_id"]
        assert (proj.get_recording(sid).dir / "tracking" / rid / "tracks.zarr").is_dir()


def test_run_processing_spawn_failure_marks_run_failed(tmp_path, monkeypatch):
    # If the coordinator can't be spawned, the reserved record must be marked failed,
    # not left stuck "running" — reconcile can't recover a run with no coordinator pid.
    #
    # A failed spawn also leaves exactly the reservation, with nothing from a coordinator on
    # top, so it is where `compute` is checked: the run page opens on the reserved record and
    # builds "Where" / the live cost from it, and a remote run read "local" all its life when
    # only the coordinator wrote it.
    def _boom(*_a, **_k):
        raise OSError("cannot fork")

    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", _boom)
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    compute = ComputeTarget("cuda", [0], None, None)
    with pytest.raises(OSError):
        proj.run_processing(
            "tracking", ["recording_001"], {**TRACKING_CONFIG, "compute": compute}
        )
    rec = proj.processing_runs[0]
    assert rec["status"] == "failed"
    assert "coordinator failed to launch" in (rec.get("error") or "")
    assert rec["compute"] == compute.to_dict()


def test_cancel_run_kills_coordinator(tmp_path):
    # cancel_run must kill the detached coordinator so it can't launch more recordings.
    import subprocess as sp
    import sys as _sys
    import time as _t

    from detail.executor.local import new_process_group_kwargs

    proj = Experiment.create("proj", tmp_path)
    proj.create_recording("recording_001")
    run_dir = runs_store.run_dir(proj.dir, "run_001")
    run_dir.mkdir(parents=True, exist_ok=True)
    # A stand-in coordinator that sleeps, carrying the run-dir path in its argv so the
    # PID-reuse guard (_is_our_worker) matches and it is killable.
    stub = sp.Popen(
        [_sys.executable, "-c", f"import time; time.sleep(30)  # {run_dir}"],
        **new_process_group_kwargs(),
    )
    try:
        rd = proj.get_recording("recording_001").dir / "tracking" / "track_01"
        _write_job_json(rd, "running")
        _write_run_record(
            proj.dir,
            "run_001",
            "tracking",
            {"recording_001": {"run_id": "track_01", "status": "running"}},
            coordinator_pid=stub.pid,
        )
        proj.cancel_run("run_001")
        deadline = _t.monotonic() + 5
        # poll() reaps the terminated stub (the test is its parent, so it would
        # otherwise linger as a zombie); returncode not-None == it was killed.
        while stub.poll() is None and _t.monotonic() < deadline:
            _t.sleep(0.05)
        assert stub.poll() is not None  # the coordinator was killed
        assert runs_store.read_run(proj.dir, "run_001")["status"] == "cancelled"
    finally:
        stub.kill()
        stub.wait()


def test_cancel_run_cancels_all_inflight_recordings(tmp_path):
    # cancel_run must reach EVERY in-flight recording (the old bug cancelled only one),
    # via the on-disk per-recording run dirs — no live handle needed. Record-level so
    # it doesn't depend on the detached coordinator's timing.
    from detail.executor import rundir

    experiment = Experiment.create("proj", tmp_path)
    sids = ["recording_001", "recording_002", "recording_003"]
    per_recording = {}
    for sid in sids:
        experiment.create_recording(sid)
        rd = experiment.get_recording(sid).dir / "tracking" / f"track_{sid}"
        _write_job_json(rd, "running")  # in-flight, no live worker
        per_recording[sid] = {"run_id": f"track_{sid}", "status": "running"}
    _write_run_record(experiment.dir, "run_001", "tracking", per_recording)

    experiment.cancel_run("run_001")

    rec = runs_store.read_run(experiment.dir, "run_001")
    assert rec["status"] == "cancelled"
    for sid in sids:  # every recording was cancelled, not just the first
        assert rec["per_recording"][sid]["status"] == "cancelled"
        rd = experiment.get_recording(sid).dir / "tracking" / f"track_{sid}"
        assert rundir.read_status(rd).value == "cancelled"


def test_delete_run_removes_outputs_and_record(tmp_path):
    from detail.storage import runs as runs_store

    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    _video_recording(proj, "recording_002")
    run = proj.run_processing(
        "tracking", ["recording_001", "recording_002"], dict(TRACKING_CONFIG)
    )
    run.wait()
    rec = proj.processing_runs[0]
    run_dirs = [
        proj.get_recording(sid).dir / "tracking" / info["run_id"]
        for sid, info in rec["per_recording"].items()
    ]
    assert all(d.is_dir() for d in run_dirs)

    proj.delete_run(run.id)

    assert proj.processing_runs == []
    assert runs_store.discover_runs(proj.dir) == []
    assert all(not d.exists() for d in run_dirs)  # per-recording outputs gone
    for sid in ("recording_001", "recording_002"):
        assert proj.get_recording(sid).tracking_runs == []  # recording.yaml entry gone


def test_delete_run_unknown_raises(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with pytest.raises(FileNotFoundError):
        proj.delete_run("run_999")


def test_delete_run_blocked_by_dependent(tmp_path):
    from detail.storage import recording as recording_store

    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    run = proj.run_processing("tracking", ["recording_001"], dict(TRACKING_CONFIG))
    run.wait()
    rec = proj.processing_runs[0]
    track_id = rec["per_recording"]["recording_001"]["run_id"]

    # an embedding run that builds on this tracking run blocks its deletion
    sdir = proj.get_recording("recording_001").dir
    data = recording_store.read_recording(sdir)
    data.setdefault("embedding_runs", []).append(
        {"id": "emb_01", "status": "complete", "tracking_run_id": track_id}
    )
    recording_store.write_recording(sdir, data)

    with pytest.raises(ValueError, match="delete its dependents first"):
        proj.delete_run(run.id)
    assert proj.processing_runs  # nothing removed — the run record is still there
    assert (sdir / "tracking" / track_id).is_dir()  # outputs untouched


# --- delete_recording (Tenet 5; the cohesive delete strategy, gui_design §1.1) - #
def test_delete_recording_removes_it(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("s1")
    proj.create_recording("s2")
    proj.delete_recording("s1")
    assert {s.id for s in proj.recordings} == {"s2"}


def test_delete_recording_missing_raises(tmp_path):
    proj = Experiment.create("p", tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        proj.delete_recording("nope")


def _index_run(proj, run_id, recordings):
    from detail.storage import runs as runs_store

    runs_store.write_run(
        proj.dir,
        run_id,
        {
            "id": run_id,
            "module_id": "tracking",
            "recording_ids": list(recordings),
            "config": {},
            "status": "complete",
            "per_recording": {
                sid: {"run_id": "ds_01", "status": "complete"} for sid in recordings
            },
            "created_at": "2026-06-09T00:00:00Z",
        },
    )


def test_delete_recording_prunes_from_processing_run(tmp_path):
    from detail.storage import runs as runs_store

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("s1")
    proj.create_recording("s2")
    _index_run(proj, "run_001", ["s1", "s2"])
    proj.delete_recording("s1")
    rec = runs_store.read_run(proj.dir, "run_001")
    assert set(rec["per_recording"]) == {"s2"}  # s1 pruned, run kept for s2


def test_delete_recording_removes_emptied_processing_run(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("s1")
    _index_run(proj, "run_001", ["s1"])
    proj.delete_recording("s1")
    assert proj.processing_runs == []  # run indexed only s1 -> record removed


def test_delete_experiment_removes_dir(tmp_path):
    proj = Experiment.create("p", tmp_path)
    pdir = proj.dir
    proj.create_recording("s1")  # not empty
    assert pdir.is_dir()
    proj.delete()
    assert not pdir.exists()  # the whole experiment is gone


def test_delete_experiment_missing_raises(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.delete()
    with pytest.raises(FileNotFoundError, match="not found"):
        proj.delete()  # already gone


# --- dataset_record / model_record (detail-view accessors, §12) ------------- #
def test_dataset_record_returns_full_yaml(tmp_path):
    from detail.storage import datasets as datasets_store

    experiment = Experiment.create("p", tmp_path)
    ddir = datasets_store.dataset_dir(experiment.dir, "dataset_001")
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        ddir,
        {
            "id": "dataset_001",
            "name": "d",
            "status": "complete",
            "n_train": 8,
            "n_val": 2,
            "classes": [{"id": 0, "name": "mouse"}],
            "source": {"kind": "run_export", "run_id": "run_001"},
        },
    )
    rec = experiment.dataset_record("dataset_001")
    assert rec["id"] == "dataset_001"
    assert rec["classes"] == [{"id": 0, "name": "mouse"}]
    assert rec["source"]["run_id"] == "run_001"


def test_model_record_returns_full_yaml(tmp_path):
    from detail.storage import models as models_store

    experiment = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(experiment.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "status": "complete",
            "model_family": "yolo",
            "classes": [{"id": 0, "name": "mouse"}],
            "weights": {"best": "run/best.pt", "last": None},
            "metrics": {"box_map50": 0.9},
        },
    )
    rec = experiment.model_record("model_001")
    assert rec["model_family"] == "yolo"
    assert rec["weights"]["best"] == "run/best.pt"
    assert rec["metrics"] == {"box_map50": 0.9}


def test_dataset_record_missing_raises(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    with pytest.raises(FileNotFoundError):
        experiment.dataset_record("dataset_404")


# --- delete_dataset / delete_model (active-work-only policy, §1.1) ----------- #
def _write_dataset(experiment, status="complete", dataset_id="dataset_001"):
    from detail.storage import datasets as datasets_store

    ddir = datasets_store.dataset_dir(experiment.dir, dataset_id)
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(ddir, {"id": dataset_id, "status": status})
    return ddir


def _write_model(experiment, status="complete", model_id="model_001", dataset_id=None):
    from detail.storage import models as models_store

    mdir = models_store.model_dir(experiment.dir, model_id)
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": model_id, "status": status, "dataset_id": dataset_id}
    )
    return mdir


def test_delete_dataset_removes_and_missing_raises(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    ddir = _write_dataset(experiment)
    experiment.delete_dataset("dataset_001")
    assert not ddir.exists()
    with pytest.raises(FileNotFoundError):
        experiment.delete_dataset("dataset_404")


def test_delete_dataset_blocks_while_active(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    ddir = _write_dataset(experiment, status="running")
    with pytest.raises(ValueError, match="still being generated"):
        experiment.delete_dataset("dataset_001")
    assert ddir.exists()  # nothing removed


def test_delete_dataset_active_model_blocks_completed_does_not(tmp_path):
    from detail.storage import models as models_store

    experiment = Experiment.create("p", tmp_path)
    ddir = _write_dataset(experiment)
    mdir = _write_model(experiment, status="running", dataset_id="dataset_001")
    with pytest.raises(ValueError, match="model is training"):
        experiment.delete_dataset("dataset_001")
    # once that model finishes, the dataset deletes — its lineage just dangles
    models_store.set_model_status(mdir, "complete")
    experiment.delete_dataset("dataset_001")
    assert not ddir.exists()


def test_delete_model_removes_and_missing_raises(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    mdir = _write_model(experiment)
    experiment.delete_model("model_001")
    assert not mdir.exists()
    with pytest.raises(FileNotFoundError):
        experiment.delete_model("model_404")


def test_delete_model_blocks_while_active(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    mdir = _write_model(experiment, status="running")
    with pytest.raises(ValueError, match="still training"):
        experiment.delete_model("model_001")
    assert mdir.exists()


def test_delete_model_active_detection_run_blocks_completed_does_not(tmp_path):
    from detail.storage import runs as runs_store

    experiment = Experiment.create("p", tmp_path)
    mdir = _write_model(experiment)
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "status": "running",
            "config": {"weights_source": "trained", "model_id": "model_001"},
            "per_recording": {},
        },
    )
    with pytest.raises(ValueError, match="detection run is using it"):
        experiment.delete_model("model_001")
    # once that run completes, the model deletes — the run keeps its own masks
    rec = runs_store.read_run(experiment.dir, "run_001")
    rec["status"] = "complete"
    runs_store.write_run(experiment.dir, "run_001", rec)
    experiment.delete_model("model_001")
    assert not mdir.exists()


# --- recording metadata / grouping variables (§11) ---------------------------- #
def test_create_recording_seeds_metadata_row(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    assert experiment.recording_metadata == {"recording_001": {SETUP: ""}}


def test_create_recording_with_metadata(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording(
        "recording_001", metadata={"animal_id": "M001", "treatment": "drug_A"}
    )
    assert experiment.recording_metadata["recording_001"] == {
        SETUP: "",
        "animal_id": "M001",
        "treatment": "drug_A",
    }
    assert experiment.metadata_columns == [SETUP, "animal_id", "treatment"]


def test_set_recording_metadata_merges(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    experiment.set_recording_metadata("recording_001", {"treatment": "drug_A"})
    assert experiment.recording_metadata["recording_001"] == {
        SETUP: "",
        "animal_id": "M001",
        "treatment": "drug_A",
    }


def test_delete_recording_prunes_metadata_row(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    experiment.create_recording("recording_002", metadata={"animal_id": "M002"})
    experiment.delete_recording("recording_001")
    assert set(experiment.recording_metadata) == {"recording_002"}


def test_delete_metadata_column_removes_it_everywhere(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("r1", metadata={"animal_id": "M001", "treatment": "d"})
    experiment.delete_metadata_column("treatment")
    assert experiment.metadata_columns == [SETUP, "animal_id"]
    assert experiment.recording_metadata["r1"] == {SETUP: "", "animal_id": "M001"}


def test_set_metadata_table_replaces_whole_table(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording(
        "r1", metadata={"animal_id": "M001", "treatment": "old"}
    )
    # bulk replace: drop treatment, add cohort, in one write
    experiment.set_metadata_table(
        ["animal_id", "cohort"], {"r1": {"animal_id": "M1", "cohort": "A"}}
    )
    assert experiment.metadata_columns == [SETUP, "animal_id", "cohort"]
    assert experiment.recording_metadata["r1"] == {
        SETUP: "",
        "animal_id": "M1",
        "cohort": "A",
    }


def test_metadata_only_recording_is_well_formed(tmp_path):
    # the reframe: a recording with no video is valid and discoverable
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001", metadata={"animal_id": "M001"})
    recording = experiment.get_recording("recording_001")
    assert recording.video_parts == []
    assert recording.video_resolution is None
    assert [s.id for s in experiment.recordings] == ["recording_001"]


def test_reconcile_imports_flips_orphaned(tmp_path):
    from detail.storage import recording as recording_store

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("queued_one")
    experiment.create_recording("importing_one")
    experiment.create_recording("ready_one")
    recording_store.set_import_status(
        experiment.get_recording("queued_one").dir, "queued"
    )
    recording_store.set_import_status(
        experiment.get_recording("importing_one").dir, "importing"
    )
    # ready_one defaults to ready
    experiment.reconcile_imports()
    assert experiment.get_recording("queued_one").import_status == "failed"
    assert experiment.get_recording("importing_one").import_status == "failed"
    assert experiment.get_recording("ready_one").import_status == "ready"  # untouched


def _write_job_json(run_dir, status, pid=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": 1, "status": status, "job": {}, "error": None}
    if pid is not None:
        envelope["pid"] = pid
    (run_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")


def _write_run_record(
    experiment_dir,
    run_id,
    module_id,
    per_recording,
    status="running",
    coordinator_pid=None,
    config=None,
):
    runs_store.write_run(
        experiment_dir,
        run_id,
        {
            "id": run_id,
            "module_id": module_id,
            "recording_ids": list(per_recording),
            "config": dict(config or {}),
            "per_recording": per_recording,
            "status": status,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    # Reconcile keys liveness on the coordinator pid; without one it treats the run
    # as just-launched and leaves it running.
    if coordinator_pid is not None:
        runs_store.set_coordinator_pid(experiment_dir, run_id, coordinator_pid)


def test_reconcile_runs_adopts_terminal_job_status(tmp_path, monkeypatch):
    # coordinator gone, but a recording worker finished (job.json=complete) after
    # close -> recovered, not blind-failed.
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    sdir = experiment.get_recording("recording_001").dir
    _write_job_json(sdir / "tracking" / "track_01", "complete")
    _write_run_record(
        experiment.dir,
        "run_001",
        "tracking",
        {"recording_001": {"run_id": "track_01", "status": "running"}},
        coordinator_pid=12345,
    )
    experiment.reconcile_runs()
    rec = runs_store.read_run(experiment.dir, "run_001")
    assert rec["status"] == "complete"
    assert rec["per_recording"]["recording_001"]["status"] == "complete"


def test_reconcile_runs_fails_dead_orphan(tmp_path, monkeypatch):
    # coordinator gone and the recording worker is provably gone too -> failed.
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    monkeypatch.setattr("detail.api.experiment.worker_alive", lambda _rd: False)
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    sdir = experiment.get_recording("recording_001").dir
    _write_job_json(sdir / "tracking" / "track_01", "running", pid=999999)
    _write_run_record(
        experiment.dir,
        "run_001",
        "tracking",
        {"recording_001": {"run_id": "track_01", "status": "running"}},
        coordinator_pid=12345,
    )
    experiment.reconcile_runs()
    rec = runs_store.read_run(experiment.dir, "run_001")
    assert rec["status"] == "failed"
    assert rec["per_recording"]["recording_001"]["status"] == "failed"
    assert "interrupted" in rec["per_recording"]["recording_001"]["error"]


def test_reconcile_runs_leaves_live_coordinator(tmp_path, monkeypatch):
    # the coordinator is still driving the run (survived the app close) -> running.
    monkeypatch.setattr("detail.api.experiment.coordinator_alive", lambda _rd, _p: True)
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    sdir = experiment.get_recording("recording_001").dir
    _write_job_json(sdir / "tracking" / "track_01", "running", pid=999999)
    _write_run_record(
        experiment.dir,
        "run_001",
        "tracking",
        {"recording_001": {"run_id": "track_01", "status": "running"}},
        coordinator_pid=12345,
    )
    experiment.reconcile_runs()
    rec = runs_store.read_run(experiment.dir, "run_001")
    assert rec["status"] == "running"
    assert rec["per_recording"]["recording_001"]["status"] == "running"


def test_reconcile_runs_fails_unlaunched_recordings(tmp_path, monkeypatch):
    # coordinator died before reaching a recording -> that recording can never run now.
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    experiment = Experiment.create("p", tmp_path)
    for sid in ("recording_001", "recording_002"):
        experiment.create_recording(sid)
    sdir = experiment.get_recording("recording_001").dir
    _write_job_json(sdir / "tracking" / "track_01", "complete")
    # only recording_001 was launched; recording_002 never appears in per_recording
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "tracking",
            "recording_ids": ["recording_001", "recording_002"],
            "config": {},
            "per_recording": {
                "recording_001": {"run_id": "track_01", "status": "running"}
            },
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    runs_store.set_coordinator_pid(experiment.dir, "run_001", 12345)
    experiment.reconcile_runs()
    rec = runs_store.read_run(experiment.dir, "run_001")
    assert rec["status"] == "failed"
    assert (
        rec["per_recording"]["recording_001"]["status"] == "complete"
    )  # its work survived
    assert rec["per_recording"]["recording_002"]["status"] == "failed"  # never launched


def test_reconcile_runs_ignores_recordless(tmp_path):
    # a "running" record with no readable job.json (synthetic/partial) is left
    # untouched — reconcile acts only on positive on-disk evidence.
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    _write_run_record(
        experiment.dir,
        "run_001",
        "tracking",
        {"recording_001": {"run_id": "track_01", "status": "running"}},
    )
    experiment.reconcile_runs()  # no job.json at tracking/track_01/
    rec = runs_store.read_run(experiment.dir, "run_001")
    assert rec["status"] == "running"
    assert rec["per_recording"]["recording_001"]["status"] == "running"


def test_reconcile_runs_partial_batch_not_marked_complete(tmp_path, monkeypatch):
    # A batch closed mid-run: recording_002 never launched (absent from per_recording). The dead
    # coordinator will never run it, so a complete recording_001 must NOT make the run
    # "complete" — it's a failed (incomplete) run (A5).
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    experiment.create_recording("recording_002")
    sdir = experiment.get_recording("recording_001").dir
    _write_job_json(sdir / "tracking" / "track_01", "complete")
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "tracking",
            "recording_ids": ["recording_001", "recording_002"],  # both planned...
            "config": {},
            "per_recording": {  # ...but only recording_001 ever launched
                "recording_001": {"run_id": "track_01", "status": "running"}
            },
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    runs_store.set_coordinator_pid(experiment.dir, "run_001", 12345)
    experiment.reconcile_runs()
    rec = runs_store.read_run(experiment.dir, "run_001")
    assert rec["status"] == "failed"  # not "complete" — recording_002 never ran
    assert rec["per_recording"]["recording_001"]["status"] == "complete"


# --- metadata CSV import (§11) ---------------------------------------------- #
def test_import_metadata_csv_merges_and_skips_blanks(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording(
        "s1", metadata={"animal_id": "M001", "treatment": "old"}
    )
    experiment.create_recording("s2")
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "recording_id,treatment,group\ns1,,exp\ns2,vehicle,ctrl\n", encoding="utf-8"
    )
    errors, warnings, applied = experiment.import_metadata_csv(csv_path)
    assert errors == [] and applied == 2
    table = experiment.recording_metadata
    assert table["s1"]["treatment"] == "old"  # blank in CSV skipped → kept
    assert table["s1"]["group"] == "exp"  # new column set
    assert table["s2"]["treatment"] == "vehicle" and table["s2"]["group"] == "ctrl"


def test_import_metadata_csv_keys_on_the_name(tmp_path):
    """The workflow the id change would otherwise have broken: a spreadsheet is keyed on
    what a person calls the recording, and ids are no longer that."""
    experiment = Experiment.create("p", tmp_path)
    first = experiment.add_recording("m27_day3")
    second = experiment.add_recording("m27_day4")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text(
        "name,day,dose\nm27_day3,3,0.5\nm27_day4,4,0.5\n", encoding="utf-8"
    )

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert errors == [] and applied == 2
    table = experiment.recording_metadata
    assert table[first.id] == {SETUP: "", "day": "3", "dose": "0.5"}
    assert table[second.id]["day"] == "4"
    assert "name" not in experiment.metadata_columns  # a key, never a stored column


def test_import_metadata_csv_reports_every_unknown_name(tmp_path):
    """All problems upfront (Tenet 10) — not one failure per run of the importer."""
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("m27_day3")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text("name,day\nm27_day3,3\nghost,1\nphantom,2\n", encoding="utf-8")

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert applied == 0  # nothing applied on any error
    assert len(errors) == 2 and "ghost" in errors[0] and "phantom" in errors[1]


def test_import_metadata_csv_prefers_the_id_when_a_csv_has_both(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27_day3")
    csv_path = tmp_path / "design.csv"
    # A stale name column beside the id: the id is the identity, so it wins.
    csv_path.write_text(
        f"recording_id,name,day\n{recording.id},whatever,3\n", encoding="utf-8"
    )

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert errors == [] and applied == 1
    assert experiment.recording_metadata[recording.id] == {SETUP: "", "day": "3"}


def test_export_metadata_carries_the_name_and_file_for_the_person_filling_it_in(
    tmp_path,
):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27_day3")
    _give_video_files(recording, "/data/rig1/m27_day3.mp4")
    experiment.add_metadata_column("dose")
    out = tmp_path / "template.csv"

    experiment.export_metadata_csv(out)

    header, row = out.read_text(encoding="utf-8-sig").splitlines()
    assert header == "recording_id,name,file,acquisition_setup,dose"
    assert row.startswith(f"{recording.id},m27_day3,m27_day3.mp4,")
    # ...and the filled sheet comes back in, keyed on the id
    out.write_text(
        f"{header}\n{recording.id},m27_day3,m27_day3.mp4,rig 1,0.5\n",
        encoding="utf-8",
    )
    errors, _warnings, applied = experiment.import_metadata_csv(out)
    assert errors == [] and applied == 1
    assert experiment.recording_metadata[recording.id] == {
        SETUP: "rig 1",
        "dose": "0.5",
    }
    assert "file" not in experiment.metadata_columns  # derived, never stored


def _give_video_files(recording, *paths: str) -> None:
    """Seed just enough of `video.files` for `video_sources` to name each part, without
    writing a video — `origin_name` is the pre-fingerprint fallback path."""
    data = recording_store.read_recording(recording.dir)
    data["video"] = {
        "files": [{"index": i, "origin_name": p} for i, p in enumerate(paths)]
    }
    recording_store.write_recording(recording.dir, data)


def _file_keyed_experiment(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    recordings = {}
    for name, path in (
        ("m27_day3", "/data/rig1/m27_day3.mp4"),
        ("m27_day4", "/data/rig1/m27_day4.mp4"),
        ("m28_day3", "/data/rig2/M28_Day3.AVI"),
    ):
        recordings[name] = experiment.add_recording(name)
        _give_video_files(recordings[name], path)
    return experiment, recordings


def test_import_metadata_csv_keys_on_the_video_file_name(tmp_path):
    """#330: the key a sheet made before import carries is the video's file name — the
    id is minted at import, and the name is right only once someone renamed everything.
    Base name, any case, extension optional, a pasted path from either platform."""
    experiment, recordings = _file_keyed_experiment(tmp_path)
    # Renamed away from the file name: the file key must not lean on the name.
    recordings["m27_day4"].set_name("second session")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text(
        "Video,dose\n"
        "M27_DAY3.MP4,0.5\n"
        "m27_day4,1.0\n"
        "D:\\raw\\rig2\\m28_day3.avi,2.0\n",
        encoding="utf-8",
    )

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert errors == [] and applied == 3
    table = experiment.recording_metadata
    assert [table[recordings[n].id]["dose"] for n in recordings] == [
        "0.5",
        "1.0",
        "2.0",
    ]
    assert experiment.metadata_columns == [SETUP, "dose"]  # the key is never stored


def test_import_metadata_csv_matches_any_video_part_of_a_recording(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27")
    _give_video_files(recording, "/data/m27_a.mp4", "/data/m27_b.mp4")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text("file,dose\nm27_b.mp4,0.5\n", encoding="utf-8")

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert errors == [] and applied == 1
    assert experiment.recording_metadata[recording.id] == {SETUP: "", "dose": "0.5"}


def test_import_metadata_csv_refuses_a_file_name_shared_across_folders(tmp_path):
    """Two rigs writing the same file name is the ordinary case, not a corner: the
    import names both recordings and applies nothing rather than picking one."""
    experiment = Experiment.create("p", tmp_path)
    for name, path in (("rig1_m27", "/rig1/m27.mp4"), ("rig2_m27", "/rig2/m27.mp4")):
        _give_video_files(experiment.add_recording(name), path)
    _give_video_files(experiment.add_recording("m28"), "/rig1/m28.mp4")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text("file,dose\nm28.mp4,1\nm27.mp4,0.5\n", encoding="utf-8")

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert applied == 0
    assert errors == [
        "video file 'm27.mp4' matches several recordings: 'rig1_m27', 'rig2_m27'"
    ]
    assert all(row == {SETUP: ""} for row in experiment.recording_metadata.values())


def test_import_metadata_csv_reports_every_unmatched_file_name(tmp_path):
    experiment, _recordings = _file_keyed_experiment(tmp_path)
    csv_path = tmp_path / "design.csv"
    # An extension, when given, has to be the right one.
    csv_path.write_text(
        "file,dose\nm27_day3.mp4,0.5\nghost.mp4,1\nm27_day4.avi,2\n,3\n",
        encoding="utf-8",
    )

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert applied == 0
    assert errors == [
        "unknown video file 'ghost.mp4' — not in this experiment",
        "unknown video file 'm27_day4.avi' — not in this experiment",
        "a row has an empty video file",
    ]


def test_import_metadata_csv_refuses_two_rows_for_one_recording(tmp_path):
    """Both parts of one recording listed as rows: which dose is the recording's? The
    import asks rather than letting the last row win silently."""
    experiment = Experiment.create("p", tmp_path)
    _give_video_files(experiment.add_recording("m27"), "/d/m27_a.mp4", "/d/m27_b.mp4")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text("file,dose\nm27_a,0.5\nm27_b,1\n", encoding="utf-8")

    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)

    assert applied == 0
    assert errors == [
        "video files 'm27_a' and 'm27_b' both match recording 'm27' — "
        "one row per recording"
    ]


def test_import_metadata_csv_keys_on_a_column_the_caller_names(tmp_path):
    """A sheet with no recognised key is refused with its columns listed; naming the one
    holding the video file name is the way through, which is what the GUI asks for."""
    experiment, recordings = _file_keyed_experiment(tmp_path)
    csv_path = tmp_path / "design.csv"
    csv_path.write_text("Session Clip,dose\nm27_day3.mp4,0.5\n", encoding="utf-8")

    assert experiment.metadata_csv_key(csv_path) == (None, ["Session Clip", "dose"])
    errors, _warnings, applied = experiment.import_metadata_csv(csv_path)
    assert applied == 0 and len(errors) == 1
    assert "no column to match rows to recordings" in errors[0]
    assert "Session Clip, dose" in errors[0]

    errors, _warnings, applied = experiment.import_metadata_csv(
        csv_path, key_column="Session Clip"
    )
    assert errors == [] and applied == 1
    assert experiment.recording_metadata[recordings["m27_day3"].id] == {
        SETUP: "",
        "dose": "0.5",
    }


def test_import_metadata_csv_refuses_a_named_key_column_it_lacks(tmp_path):
    experiment, _recordings = _file_keyed_experiment(tmp_path)
    csv_path = tmp_path / "design.csv"
    csv_path.write_text("file,dose\nm27_day3.mp4,0.5\n", encoding="utf-8")

    errors, _warnings, applied = experiment.import_metadata_csv(
        csv_path, key_column="clip"
    )

    assert applied == 0 and errors == [
        "CSV has no column 'clip' — its columns are: file, dose"
    ]


def test_import_metadata_csv_detects_the_key_it_would_use(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    csv_path = tmp_path / "design.csv"
    csv_path.write_text("dose,Video File\n", encoding="utf-8")
    assert experiment.metadata_csv_key(csv_path) == (
        "Video File",
        ["dose", "Video File"],
    )


def test_import_metadata_csv_unknown_recording_applies_nothing(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("s1", metadata={"animal_id": "M001"})
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "recording_id,treatment\ns1,drug_A\nghost,x\n", encoding="utf-8"
    )
    errors, warnings, applied = experiment.import_metadata_csv(csv_path)
    assert applied == 0 and any("ghost" in e for e in errors)
    assert experiment.recording_metadata["s1"] == {
        SETUP: "",
        "animal_id": "M001",
    }  # atomic — unchanged


def test_import_metadata_csv_dry_run_applies_nothing(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("s1")
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("recording_id,treatment\ns1,drug_A\n", encoding="utf-8")
    errors, warnings, applied = experiment.import_metadata_csv(csv_path, dry_run=True)
    assert errors == [] and applied == 0
    assert experiment.recording_metadata["s1"] == {SETUP: ""}  # validated, not applied


# --- experiment export / import (§15) ------------------------------------------ #
def _managed_ws(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "ws")
    return tmp_path / "ws"


def test_export_import_archive_roundtrip(tmp_path, monkeypatch):
    import shutil

    ws = _managed_ws(tmp_path, monkeypatch)
    proj = Experiment.create("study")  # lands in ws/study
    proj.create_recording("s1", metadata={"animal_id": "M001"})
    archive = proj.export(tmp_path / "study.tar")
    shutil.rmtree(proj.dir)  # simulate moving to another machine
    imported = Experiment.import_archive(archive)
    assert imported.dir == ws / "study"
    assert imported.recording_metadata["s1"] == {SETUP: "", "animal_id": "M001"}


def test_import_archive_collision_and_name(tmp_path, monkeypatch):
    ws = _managed_ws(tmp_path, monkeypatch)
    proj = Experiment.create("study")
    archive = proj.export(tmp_path / "s.tar")
    with pytest.raises(FileExistsError):
        Experiment.import_archive(archive)  # study already exists
    Experiment.import_archive(archive, replace=True)  # ok
    other = Experiment.import_archive(archive, name="study_copy")
    assert other.dir == ws / "study_copy"


# --- experiment locations (§2) ------------------------------------------------------ #
def _lab_drive(tmp_path):
    """An added experiment location standing in for an external drive."""
    from detail.storage import locations

    folder = tmp_path / "lab_drive"
    folder.mkdir()
    locations.add_location("Lab drive", folder)
    return folder.resolve()


def test_create_in_a_location_and_list_all_spans_every_location(tmp_path):
    drive = _lab_drive(tmp_path)

    home = Experiment.create("home_study")
    away = Experiment.create("drive_study", location="Lab drive")

    assert away.dir == drive / "drive_study"
    assert (home.location, away.location) == ("default", "Lab drive")
    assert [e.dir for e in Experiment.list_all()] == [home.dir, away.dir]
    assert Experiment.create("loose", tmp_path / "elsewhere").location is None


def test_create_refuses_parent_dir_and_location_together(tmp_path):
    _lab_drive(tmp_path)
    with pytest.raises(ValueError, match="parent_dir or location, not both"):
        Experiment.create("x", tmp_path, location="Lab drive")


def test_create_refuses_a_name_another_location_holds(tmp_path):
    _lab_drive(tmp_path)
    Experiment.create("study", location="Lab drive")
    with pytest.raises(FileExistsError, match="already exists in location 'Lab drive'"):
        Experiment.create("study")


def test_create_in_an_unplugged_location_says_so(tmp_path):
    from detail.storage import locations

    drive = _lab_drive(tmp_path)
    drive.rmdir()
    with pytest.raises(locations.LocationNotConnectedError, match="not connected"):
        Experiment.create("study", location="Lab drive")
    assert not drive.exists()


def test_import_archive_into_a_location(tmp_path):
    import shutil

    from detail.storage import locations

    ws = locations.experiments_root()
    drive = _lab_drive(tmp_path)
    proj = Experiment.create("study")
    archive = proj.export(tmp_path / "study.tar")

    with pytest.raises(FileExistsError, match="already exists in location 'default'"):
        Experiment.import_archive(archive, location="Lab drive")
    shutil.rmtree(ws / "study")
    imported = Experiment.import_archive(archive, location="Lab drive")

    assert imported.dir == drive / "study"
    assert imported.location == "Lab drive"


def test_archive_marks_config_and_unarchive_clears_it(tmp_path):
    proj = Experiment.create("study", tmp_path)
    assert not proj.archived
    proj.archive()
    assert proj.archived
    config = experiment_store.read_config(proj.dir)
    assert next(iter(config)) == "schema_version"  # still the literal first key
    assert config["archived"] is True
    proj.archive()  # idempotent
    assert proj.archived
    proj.unarchive()
    assert not proj.archived
    # Absent, not `archived: false`: an unarchived experiment reads as one never archived.
    assert "archived" not in experiment_store.read_config(proj.dir)


def test_list_all_includes_archived_experiments(tmp_path, monkeypatch):
    _managed_ws(tmp_path, monkeypatch)
    kept = Experiment.create("kept")
    shelved = Experiment.create("shelved")
    shelved.archive()
    listed = {e.dir: e.archived for e in Experiment.list_all()}
    assert listed == {kept.dir: False, shelved.dir: True}


def test_import_archive_of_an_archived_experiment_lands_unarchived(
    tmp_path, monkeypatch
):
    import shutil

    _managed_ws(tmp_path, monkeypatch)
    proj = Experiment.create("study")
    proj.archive()
    archive = proj.export(tmp_path / "study.tar")
    shutil.rmtree(proj.dir)
    assert not Experiment.import_archive(archive).archived


# --- cancelling active runs (kill the worker + mark cancelled) -------------- #
#: Every stand-in worker this module has spawned, so none of them outlives its test.
_SLEEPERS: list = []


@pytest.fixture(autouse=True)
def _reap_sleepers():
    """Kill any stand-in worker a test leaves behind.

    They sleep for sixty seconds, and several of these tests rely on the *code under test* to
    kill them — so a test that fails leaves a live process holding a run-dir path for a minute
    afterwards. That is debris one failure hands to the next, and it is exactly the shape that
    turns a single flake into the bursts of four and six seen here. A test's failure has to cost
    the next test nothing.
    """
    yield
    while _SLEEPERS:
        proc = _SLEEPERS.pop()
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)  # reap it, so it does not linger as a zombie either


def _sleeper(run_dir):
    """A stand-in worker carrying ``run_dir`` in its argv (like the real worker)
    so the cancel's PID-reuse identity guard recognizes it. Spawned in its own
    process group (like the real executor) so taskkill-ing it on Windows can't
    deliver a console Ctrl-C to this test process."""
    import subprocess
    import sys

    from detail.executor.local import new_process_group_kwargs

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(run_dir)],
        **new_process_group_kwargs(),
    )
    _SLEEPERS.append(proc)
    return proc


def _job_json(run_dir, pid):
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.json").write_text(
        json.dumps(
            {"schema_version": 1, "status": "running", "pid": pid, "error": None}
        ),
        encoding="utf-8",
    )


def test_cancel_run_kills_worker_and_marks_cancelled(tmp_path):
    from detail.storage import runs as runs_store
    from detail.storage import recording as recording_store

    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    proc = _sleeper(sess.dir / "detections" / "det_01")
    _job_json(sess.dir / "detections" / "det_01", proc.pid)  # the worker run dir
    data = recording_store.read_recording(sess.dir)
    data.setdefault("detection_runs", []).append(
        {"id": "det_01", "status": "running", "output": "detections/det_01/"}
    )
    recording_store.write_recording(sess.dir, data)
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "per_recording": {
                "recording_001": {"run_id": "det_01", "status": "running"}
            },
            "config": {},
            "status": "running",
            "created_at": "x",
        },
    )

    proj.cancel_run("run_001")

    proc.wait(timeout=5)  # worker killed (else TimeoutExpired fails the test)
    assert proc.returncode is not None
    rec = runs_store.read_run(proj.dir, "run_001")
    assert rec["status"] == "cancelled"
    assert rec["per_recording"]["recording_001"]["status"] == "cancelled"
    entry = next(
        r
        for r in recording_store.read_recording(sess.dir)["detection_runs"]
        if r["id"] == "det_01"
    )
    assert entry["status"] == "cancelled"


def test_cancel_run_noop_when_terminal(tmp_path):
    from detail.storage import runs as runs_store

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "per_recording": {
                "recording_001": {"run_id": "det_01", "status": "complete"}
            },
            "config": {},
            "status": "complete",
            "created_at": "x",
        },
    )
    proj.cancel_run("run_001")  # already terminal → untouched
    assert runs_store.read_run(proj.dir, "run_001")["status"] == "complete"


def test_cancel_model_kills_worker_and_marks_cancelled(tmp_path):
    from detail.storage import models as models_store

    proj = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(proj.dir, "model_001")
    mdir.mkdir(parents=True, exist_ok=True)
    proc = _sleeper(mdir)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "trainer_id": "ultralytics_yolo",
            "dataset_id": "d",
            "base_model": "x",
            "status": "running",
        },
    )
    _job_json(mdir, proc.pid)  # models/<id>/ is the run dir

    proj.cancel_model("model_001")

    proc.wait(timeout=5)
    assert proc.returncode is not None
    assert models_store.read_model_yaml(mdir)["status"] == "cancelled"


def test_run_processing_honors_external_cancel_sentinel(tmp_path):
    """Experiment.cancel_run's on-disk sentinel stops a *live* orchestrator before it
    runs the next recording — the dir-based cancel can't touch the in-process thread,
    so without the sentinel the batch would continue and overwrite the status."""
    from detail.storage import runs as runs_store

    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    _video_recording(proj, "recording_002")
    # pre-set the sentinel for the id this batch gets (fresh experiment → run_001)
    runs_store.request_cancel(proj.dir, "run_001")
    run = proj.run_processing(
        "tracking", ["recording_001", "recording_002"], dict(TRACKING_CONFIG)
    )
    run.wait()

    assert run.status == "cancelled"
    rec = runs_store.read_run(proj.dir, "run_001")
    assert rec["status"] == "cancelled"
    assert rec["per_recording"] == {}  # broke at the first loop-top check; nothing ran


def test_cancel_run_keeps_complete_status(tmp_path):
    """A worker that finished after the app closed leaves job.json 'complete' but a
    stale 'running' record; cancel must reflect complete, not relabel it cancelled
    (which would hide the results view)."""
    import json

    from detail.storage import runs as runs_store
    from detail.storage import recording as recording_store

    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    det_dir = sess.dir / "detections" / "det_01"
    det_dir.mkdir(parents=True, exist_ok=True)
    (det_dir / "job.json").write_text(  # finished worker, no pid, terminal status
        json.dumps({"schema_version": 1, "status": "complete", "error": None}),
        encoding="utf-8",
    )
    data = recording_store.read_recording(sess.dir)
    data.setdefault("detection_runs", []).append(
        {"id": "det_01", "status": "running", "output": "detections/det_01/"}
    )
    recording_store.write_recording(sess.dir, data)
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "per_recording": {
                "recording_001": {"run_id": "det_01", "status": "running"}
            },
            "config": {},
            "status": "running",
            "created_at": "x",
        },
    )

    proj.cancel_run("run_001")

    rec = runs_store.read_run(proj.dir, "run_001")
    assert rec["status"] == "complete"  # not relabeled cancelled
    assert rec["per_recording"]["recording_001"]["status"] == "complete"
    entry = next(
        r
        for r in recording_store.read_recording(sess.dir)["detection_runs"]
        if r["id"] == "det_01"
    )
    assert entry["status"] == "complete"


# --- retry_run (gap-fill a terminal run's incomplete recordings, §15) ---------- #
def _failed_tracking_run(proj, run_id="run_001"):
    """A tracking run where recording_002 'failed' and never ran.


    Shaped like the incident: one recording complete with real output, the other holding
    no stage run id at all (its worker never launched), so a retry must run it whole.
    """
    _video_recording(proj, "recording_001")
    _video_recording(proj, "recording_002")
    run = proj.run_processing("tracking", ["recording_001"], dict(TRACKING_CONFIG))
    run.wait()
    done = proj.processing_runs[0]["per_recording"]["recording_001"]
    runs_store.write_run(
        proj.dir,
        run_id,
        {
            "id": run_id,
            "module_id": "tracking",
            "recording_ids": ["recording_001", "recording_002"],
            "config": dict(TRACKING_CONFIG),
            "per_recording": {
                "recording_001": dict(done),
                "recording_002": {
                    "run_id": None,
                    "status": "failed",
                    "error": "HfHubHTTPError: 504",
                },
            },
            "status": "failed",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    return run_id


def test_retry_run_fills_the_same_record_and_rolls_up_complete(tmp_path):
    """The whole point: gap-fill, don't mint a new run. The record stays one logical
    result set and reaches `complete` once the hole is filled."""
    proj = Experiment.create("proj", tmp_path)
    run_id = _failed_tracking_run(proj)

    handle = proj.retry_run(run_id)
    handle.wait()

    assert handle.id == run_id  # same run, not a new one
    assert [r["id"] for r in proj.processing_runs] == [run_id]
    rec = runs_store.read_run(proj.dir, run_id)
    assert rec["status"] == "complete"
    per = rec["per_recording"]
    assert all(v["status"] == "complete" for v in per.values())
    assert "error" not in per["recording_002"]  # the stale reason is cleared
    # the retried recording ran for real, and the completed one was not re-run
    rid = per["recording_002"]["run_id"]
    assert (
        proj.get_recording("recording_002").dir / "tracking" / rid / "tracks.zarr"
    ).is_dir()


def test_retry_run_does_not_rerun_completed_recordings(tmp_path):
    """The coordinator never read per_recording back, so it always started from an empty
    map — a retry would re-run every recording, redoing the 29 completed hours of the
    incident this exists to fix. Revert the seeding and recording_001 gets a second
    stage run id.

    Note the roll-up survives that bug (re-running everything still ends `complete`),
    so only this test catches it — seeding and pending-selection share one argument, so
    _finalize can't see a short count.
    """
    proj = Experiment.create("proj", tmp_path)
    run_id = _failed_tracking_run(proj)
    before = runs_store.read_run(proj.dir, run_id)["per_recording"]["recording_001"]

    proj.retry_run(run_id).wait()

    after = runs_store.read_run(proj.dir, run_id)["per_recording"]["recording_001"]
    assert after["run_id"] == before["run_id"]  # untouched — not re-run under a new id
    assert len(proj.get_recording("recording_001").tracking_runs) == 1


def test_retry_run_refuses_when_nothing_is_incomplete(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    run = proj.run_processing("tracking", ["recording_001"], dict(TRACKING_CONFIG))
    run.wait()
    with pytest.raises(ValueError, match="nothing to retry"):
        proj.retry_run(run.id)


def test_retry_run_refuses_a_running_run(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    _write_run_record(
        proj.dir,
        "run_001",
        "tracking",
        {"recording_001": {"run_id": "track_01", "status": "running"}},
        status="running",
    )  # no coordinator pid → reconcile leaves it running (just-launched)
    with pytest.raises(ValueError, match="still running"):
        proj.retry_run("run_001")


def test_retry_run_unknown_id_raises(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with pytest.raises(FileNotFoundError):
        proj.retry_run("run_404")


def test_retry_run_clears_the_stale_cancel_marker(tmp_path):
    """A cancelled run's sentinel outlives its coordinator. Leave it and the retry's
    fresh coordinator stops on its first recording and re-records cancelled — the
    resume_training lesson at the run level, which had no clearer at all."""
    proj = Experiment.create("proj", tmp_path)
    run_id = _failed_tracking_run(proj)
    runs_store.request_cancel(proj.dir, run_id)  # as Experiment.cancel_run leaves it
    assert runs_store.cancel_requested(proj.dir, run_id)

    proj.retry_run(run_id).wait()

    assert not runs_store.cancel_requested(proj.dir, run_id)
    assert runs_store.read_run(proj.dir, run_id)["status"] == "complete"


def test_retry_run_seeds_progress_so_the_bar_starts_where_the_run_stands(tmp_path):
    """The aggregate bar must resume at 1/2, not restart at 0/2: the run is a whole,
    and a client reattaching before the coordinator's first emit reads this file."""
    proj = Experiment.create("proj", tmp_path)
    run_id = _failed_tracking_run(proj)

    # Not awaited before the reads — inspect the seeded standing, before the coordinator's
    # first emit overwrites it.
    run = proj.retry_run(run_id)

    progress = runs_store.read_progress(proj.dir, run_id)
    assert progress["recordings_done"] == 1  # the already-complete recording
    assert progress["recordings_total"] == 2
    assert progress["current_recording"] == "recording_002"  # first pending, not [0]
    # Awaited after them, so the coordinator does not outlive the test that started it and
    # write into a temp dir pytest is cleaning up (#543).
    run.wait()


def test_retry_run_restores_typed_config_before_validating(tmp_path):
    """The record stores _jsonable(config), so a retry validates flattened *dicts*
    unless it restores them first — detection's validate reads `compute.backend`, which
    a dict doesn't have, so the AttributeError would surface as a bogus 'stale config'
    rejection. run_processing never hits this: it validates the live typed config
    before ever writing the record.

    Uses detection because downsampling's config is plain JSON and cannot show it.
    """
    from detail.api.experiment import _jsonable

    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    config = {
        "model_id": "fake_dl",
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "compute": ComputeTarget("cpu", None, None, None),
        "conf_threshold": 0.5,
    }
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "config": _jsonable(config),  # exactly what a real record holds
            "per_recording": {
                "recording_001": {"run_id": None, "status": "failed", "error": "boom"}
            },
            "status": "failed",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    with _registered(_fake_model(ModelReadiness(True, "ok", "", None, None))):
        # Reaching the spawn at all proves validation ran against restored objects;
        # without the restore it raises "cannot retry ... AttributeError".
        handle = proj.retry_run("run_001")
        handle.wait()
    assert runs_store.read_run(proj.dir, "run_001")["per_recording"]["recording_001"][
        "status"
    ] in (
        "complete",
        "failed",
    )  # it ran; the fake model's output is not the point


def test_retry_run_reconciles_a_dead_coordinator_first(tmp_path, monkeypatch):
    """A SIGKILLed coordinator leaves the record 'running' forever and no CLI path calls
    reconcile — gating on a terminal status alone would reject a retryable run."""
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    _write_run_record(
        proj.dir,
        "run_001",
        "tracking",
        {"recording_001": {"run_id": "track_01", "status": "running"}},
        status="running",
        coordinator_pid=424242,
        config=dict(TRACKING_CONFIG),
    )
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    monkeypatch.setattr("detail.api.experiment.worker_alive", lambda _rd: False)

    # Reconcile folds the dead run to failed, so the retry proceeds instead of raising.
    handle = proj.retry_run("run_001")
    handle.wait()
    assert runs_store.read_run(proj.dir, "run_001")["status"] == "complete"


def test_retry_run_names_every_invalid_recording(tmp_path):
    """Fail fast, naming all of them, so one pass fixes everything — not first-error."""
    proj = Experiment.create("proj", tmp_path)
    run_id = _failed_tracking_run(proj)
    rec = runs_store.read_run(proj.dir, run_id)
    rec["per_recording"]["recording_003"] = {"run_id": None, "status": "failed"}
    rec["recording_ids"].append("recording_003")  # a recording that no longer exists
    rec["per_recording"]["recording_002"]["status"] = "failed"
    runs_store.write_run(proj.dir, run_id, rec)
    # break recording_002 too: no detection run left to track
    import shutil

    from detail.storage import recording as recording_store

    sdir = proj.get_recording("recording_002").dir
    shutil.rmtree(sdir / "detections")
    data = recording_store.read_recording(sdir)
    data["detection_runs"] = []
    recording_store.write_recording(sdir, data)

    with pytest.raises(ValueError) as exc:
        proj.retry_run(run_id)
    message = str(exc.value)
    assert "recording_002" in message and "recording_003" in message


def test_retry_run_survives_launcher_exit(tmp_path):
    """§15 durability for the retry path: the process that calls retry_run may exit
    immediately — the re-spawned coordinator is detached and must still finish, and any
    fresh process can reattach to it. Mirrors test_run_processing_survives_launcher_exit.
    """
    import subprocess as sp
    import sys as _sys
    import textwrap

    proj = Experiment.create("proj", tmp_path)
    run_id = _failed_tracking_run(proj)
    launcher = textwrap.dedent(f"""
        from detail.api.experiment import Experiment
        p = Experiment.open({str(proj.dir)!r})
        p.retry_run({run_id!r})  # return immediately — the coordinator is detached
        """)
    sp.run([_sys.executable, "-c", launcher], check=True, timeout=60)

    handle = proj.attach_run(run_id)  # a fresh process reattaching to the retry
    handle.wait()
    assert handle.status == "complete"
    rid = runs_store.read_run(proj.dir, run_id)["per_recording"]["recording_002"][
        "run_id"
    ]
    assert (
        proj.get_recording("recording_002").dir / "tracking" / rid / "tracks.zarr"
    ).is_dir()


# --- preview image (the Experiments list's identifying still) ------------------ #
def _preview_video(path, n_frames: int, size: int = 64) -> None:
    """An mp4 whose every frame is a solid grey encoding its own index, so a test can
    tell *which* frame a preview came from."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (size, size)
    )
    try:
        for i in range(n_frames):
            writer.write(np.full((size, size, 3), i * 8, dtype=np.uint8))
    finally:
        writer.release()


def test_preview_image_comes_from_the_middle_of_the_video(tmp_path):
    import cv2

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("r1")
    _preview_video(tmp_path / "clip.mp4", n_frames=20)
    recording.add_video_part(tmp_path / "clip.mp4")

    path = proj.preview_image()
    assert path is not None and path.is_file() and path.name == "preview.jpg"
    # Frame i is grey i*8, so the midpoint (frame 10) is ~80 — the opening frames are
    # often a blank lead-in, which is the whole reason not to take frame 0.
    grey = int(cv2.imread(str(path))[:, :, 0].mean())
    assert abs(grey - 10 * 8) <= 8, f"expected the middle frame, got grey {grey}"


def test_preview_image_is_cached_and_refreshes_when_the_experiment_changes(tmp_path):
    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("r1")
    _preview_video(tmp_path / "clip.mp4", n_frames=12)
    recording.add_video_part(tmp_path / "clip.mp4")

    first = proj.preview_image()
    stamp = first.stat().st_mtime_ns
    assert proj.preview_image().stat().st_mtime_ns == stamp  # a read, not a re-decode

    # Adding a recording touches the experiment, so the still is regenerated: listing a
    # workspace must not show a picture of a state that no longer exists.
    import os

    os.utime(first, (0, 0))
    assert proj.preview_image().stat().st_mtime_ns != 0


def test_preview_image_is_none_when_there_is_nothing_to_show(tmp_path):
    # No recordings at all, and a recording with no video: both render a placeholder
    # rather than failing — Home lists whatever is in the workspace.
    empty = Experiment.create("empty", tmp_path)
    assert empty.preview_image() is None

    metadata_only = Experiment.create("meta", tmp_path)
    metadata_only.create_recording("r1")  # a valid recording with no video
    assert metadata_only.preview_image() is None


def test_model_names_maps_ids_to_the_names_people_gave_them(tmp_path):
    """The model twin of recording_names: records key on the id (permanent), users read
    the name (§11). Labels fall back to the id for anything absent — a built-in model,
    whose id is its name, or a model deleted since the run."""
    from detail.storage import models as models_store

    proj = Experiment.create("p", tmp_path)
    assert proj.model_names() == {}

    for model_id, name in (("model_001", "Fish_YOLO"), ("model_002", "")):
        mdir = models_store.model_dir(proj.dir, model_id)
        mdir.mkdir(parents=True, exist_ok=True)
        models_store.write_model_yaml(
            mdir, {"id": model_id, "name": name, "status": "complete"}
        )

    names = proj.model_names()
    assert names["model_001"] == "Fish_YOLO"
    assert "model_002" not in names  # an unnamed model contributes nothing to render


# --- a re-submit asks the process, not the record (#280) -------------------- #
def _training_job_record():
    from detail.core.dataclasses import ComputeTarget, TrainingJob

    return TrainingJob(
        job_id="model_001",
        model_id="model_001",
        dataset_id="dataset_001",
        trainer_id="ultralytics_yolo",
        base_model="yolo11s-seg.pt",
        hyperparams={"epochs": 2},
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-07-10T00:00:00Z",
    )


def test_resume_training_refuses_while_the_previous_worker_is_alive(tmp_path):
    """A terminal record does not prove the process is gone.

    A run's status is written by the process that runs it, so one force-killed mid-flight
    never writes a terminal status — and reconcile only inspects records that are still
    *active*. ``partial`` / ``cancelled`` / ``failed`` is therefore compatible with that
    worker being alive and still holding the GPU. #280 resumed into exactly that and hung:
    `taskkill /T` on the *trainer* kills the subtree below it, leaving the worker above.
    """
    from detail.executor import rundir
    from detail.storage import models as models_store

    proj = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(proj.dir, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"prior")
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "failed", "trainer_id": "ultralytics_yolo"},
    )
    rundir.init_run_dir(mdir, _training_job_record())
    proc = _sleeper(mdir)  # the worker that outlived the kill, run_dir in its argv
    try:
        rundir.set_pid(mdir, proc.pid)
        with pytest.raises(ValueError, match="still has a training process running"):
            proj.resume_training("model_001")
    finally:
        proc.kill()
        proc.wait()


def test_resume_training_proceeds_once_that_worker_is_gone(tmp_path, monkeypatch):
    """The negative case: the guard must not refuse an ordinary resume. Without this a
    guard that simply always raised would pass the test above."""
    from detail.executor import rundir
    from detail.storage import models as models_store

    proj = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(proj.dir, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"prior")
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "failed", "trainer_id": "ultralytics_yolo"},
    )
    rundir.init_run_dir(mdir, _training_job_record())
    proc = _sleeper(mdir)
    proc.kill()
    proc.wait()  # the same pid, now dead
    rundir.set_pid(mdir, proc.pid)
    monkeypatch.setattr(
        "detail.api.experiment.LocalExecutor.submit", lambda self, job: job.model_id
    )
    proj.resume_training("model_001")  # must not raise
    assert models_store.read_model_yaml(mdir)["status"] == "running"


def test_resume_training_allows_a_worker_that_already_reported_its_outcome(tmp_path):
    """A live pid is not work. The exit window must not be refused.

    A run's pid stays alive for a moment after it records its outcome, and a
    coordinator's pid file outlives ``run.wait()`` outright — so a guard built on
    liveness alone refuses legitimate retries whenever the box is loaded enough for the
    shutdown to still be in flight. It did: this check first shipped comparing only
    ``_alive(pid)`` and turned two ``retry_run`` tests red on a full-suite run while
    passing in isolation. The envelope's own terminal status is what says "finished".
    """
    from detail.core.dataclasses import JobStatus
    from detail.executor import rundir
    from detail.executor.local import live_process_pid
    from detail.storage import models as models_store

    proj = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(proj.dir, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "last.pt").write_bytes(b"prior")
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "failed", "trainer_id": "ultralytics_yolo"},
    )
    rundir.init_run_dir(mdir, _training_job_record())
    proc = _sleeper(mdir)  # still alive, still carrying run_dir in its argv
    try:
        rundir.set_pid(mdir, proc.pid)
        rundir.set_status(mdir, JobStatus.FAILED)  # ...but it recorded its outcome
        assert live_process_pid(mdir) is None  # so it is not "working"
    finally:
        proc.kill()
        proc.wait()


def test_a_lost_terminal_record_write_is_recovered_by_the_reconciler(
    tmp_path, monkeypatch
):
    """Why `_finalize` may swallow its write: nothing is lost.

    `_finalize` runs in `run()`'s `finally`, outside every handler, so raising there does not
    merely lose the record — it escapes into `executor/coordinator.py`'s backstop and is
    written down as `failed: coordinator error` over a run whose every recording completed.

    Swallowing it leaves the record at `running`, which is still in `ACTIVE_STATUSES`, so the
    reconciler picks it up and recomputes the status from each recording's own `job.json` —
    the evidence, not the index. This is that path, driven end to end: a record left exactly
    as a failed terminal write leaves it, reconciled back to `complete`.

    That backstop already has to survive a SIGKILL; a lost write is strictly less damage.
    """
    monkeypatch.setattr(
        "detail.api.experiment.coordinator_alive", lambda _rd, _p: False
    )
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    sdir = experiment.get_recording("recording_001").dir
    _write_job_json(sdir / "tracking" / "track_01", "complete")
    # Exactly the state a failed `_finalize` write leaves behind: the last *boundary* write
    # landed (the recording is complete in the index) and the terminal one did not, so the
    # run itself is still "running".
    _write_run_record(
        experiment.dir,
        "run_001",
        "tracking",
        {"recording_001": {"run_id": "track_01", "status": "complete"}},
        coordinator_pid=12345,
    )
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {**runs_store.read_run(experiment.dir, "run_001"), "status": "running"},
    )

    experiment.reconcile_runs()

    record = runs_store.read_run(experiment.dir, "run_001")
    assert record["status"] == "complete", (
        "a run whose recordings all completed must reconcile to complete — this is the "
        "property that makes swallowing the terminal write safe"
    )
    assert "coordinator error" not in str(record.get("error", ""))


def test_a_failed_terminal_write_does_not_escape_to_the_coordinator(
    tmp_path, monkeypatch
):
    """And the swallow itself: `_finalize` must not raise out of `run()`'s `finally`.

    The negative case is the test above — if `_finalize` simply stopped recording anything,
    that one fails, because the reconciler would have no complete recordings to roll up.
    """
    from detail.api.experiment import RunOrchestrator

    experiment = Experiment.create("p", tmp_path)
    orchestrator = object.__new__(RunOrchestrator)
    orchestrator.id = "run_001"
    orchestrator._experiment = experiment
    orchestrator._module_id = "tracking"
    orchestrator._per_recording = {"recording_001": {"status": "complete"}}
    orchestrator._cancelled = threading.Event()
    orchestrator._status = "running"
    monkeypatch.setattr(
        RunOrchestrator,
        "_write_record",
        lambda _self: (_ for _ in ()).throw(PermissionError(5, "Access is denied")),
    )

    orchestrator._finalize(1)  # must not raise

    assert (
        orchestrator._status == "complete"
    ), "the status is still computed — only its persistence is best-effort"


@pytest.mark.parametrize(
    "task, base, fragment",
    [
        ("pose", "yolo11s-seg.pt", "train it from a YOLO-pose base"),
        (None, "yolo11n-pose.pt", "holds detection labels"),
    ],
)
def test_a_base_model_that_cannot_learn_the_labels_is_refused(task, base, fragment):
    from detail.api.experiment import check_base_model_task

    with pytest.raises(ValueError, match=fragment):
        check_base_model_task("dataset_001", task, base)
    check_base_model_task("dataset_001", "pose", "yolo11n-pose.pt")  # a match is fine
    check_base_model_task("dataset_001", None, "yolo11s-seg.pt")
