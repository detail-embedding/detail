"""The tuning runner (§12 *Tune detection and tracking settings*): windows, one detection
pass, the real tracker swept over it, and the record it leaves.

The model's inference is the one thing stubbed (CLAUDE.md: the only legitimate mock): the
video, the timestamp maps, the detection writer, ``anchor_tracker`` and the metrics are real.
"""

import threading
from fractions import Fraction

import numpy as np
import pytest

from detail.analysis import track_quality
from detail.api.experiment import Experiment
from detail.core.dataclasses import ComputeTarget, Detection, FrameResult
from detail.pipeline import tune_tracking as tt
from detail.storage import tunings as tunings_store

W, H = 64, 48
FPS = 10
N_FRAMES = 200


def _write_video(path, n_frames=N_FRAMES):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=Fraction(FPS, 1))
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            img = np.full((H, W, 3), (i * 7) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


def _box_mask(r0, r1, c0, c1):
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _bbox(r0, r1, c0, c1):
    return (c0 / W, r0 / H, c1 / W, r1 / H)


BODY = dict(mask=_box_mask(10, 30, 10, 50), bbox=_bbox(10, 30, 10, 50))
# The second piece: its box overlaps the body's (IoU ~0.2), its pixels mostly do not.
PIECE = dict(mask=_box_mask(15, 25, 50, 60), bbox=_bbox(15, 25, 30, 60))
OTHER = dict(mask=_box_mask(35, 45, 52, 62), bbox=_bbox(35, 45, 52, 62))


def _det(part, conf, name="subject"):
    return Detection(
        mask=part["mask"],
        bbox=part["bbox"],
        class_name=name,
        confidence=conf,
        object_id=None,
    )


class SplitSubject:
    """A model that finds the subject in two pieces whose confidences swap every frame:
    the top one alone flips between a large and a small mask, the two merged are steady.
    """

    supported_backends = ["cpu"]
    supports_batching = True

    def __init__(self, scene="split"):
        self.scene = scene
        self.seen = 0
        self.conf = None

    def load(self, device):
        self.device = device

    def set_conf_threshold(self, threshold):
        self.conf = threshold

    def predict_batch(self, frames, prompts):
        out = []
        for frame in frames:
            assert frame.shape[:2] == (H, W)
            i = self.seen
            self.seen += 1
            if self.scene == "two":
                dets = [_det(BODY, 0.9), _det(OTHER, 0.85)]
            elif i % 2:
                dets = [_det(BODY, 0.9), _det(PIECE, 0.4)]
            else:
                dets = [_det(PIECE, 0.8), _det(BODY, 0.3)]
            out.append(FrameResult.from_detections(i, 0, dets))
        return out


def _experiment(root):
    experiment = Experiment.create("p", root / "exp")
    recording = experiment.add_recording("subject a")
    recording.add_video_part(_write_video(root / "a.mp4"))
    return experiment, recording


@pytest.fixture
def tuned_experiment(tmp_path):
    return _experiment(tmp_path)


def _run(experiment, recording, model, *, sample_seconds=10.0, stop=None, events=None):
    job = tt.TuneTrackingJob(
        job_id="tune_001",
        model_id="model_001",
        weights_source="trained",
        model_variant=None,
        source_recording_ids=[recording.id],
        sample_seconds=sample_seconds,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-11-19T00:00:00Z",
        source={"kind": "standalone"},
    )
    run_dir = tt.run_dir_for(experiment.dir, job)
    run_dir.mkdir(parents=True)
    tt.run(job, run_dir, (events.append if events is not None else None), stop)
    return run_dir, tunings_store.read_tuning(run_dir)


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(
        "detail.pipeline.tune_tracking.resolve_detection_model", lambda *a, **k: model
    )


@pytest.fixture(scope="module")
def split_run(tmp_path_factory):
    """One full sweep of the split-subject scene, shared: a sweep is 43 tracker runs."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("detail.core.progress.PROGRESS_INTERVAL_S", 0.0)
        model = SplitSubject()
        _patch_model(mp, model)
        experiment, recording = _experiment(tmp_path_factory.mktemp("split"))
        events: list = []
        run_dir, record = _run(experiment, recording, model, events=events)
    return model, recording, run_dir, record, events


@pytest.fixture(scope="module")
def two_run(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        _patch_model(mp, SplitSubject("two"))
        experiment, recording = _experiment(tmp_path_factory.mktemp("two"))
        return _run(experiment, recording, None)[1]


# --------------------------------------------------------------------------- windows


def test_a_short_recording_is_taken_whole():
    t = [np.arange(30) * 100_000_000]  # 3 s at 10 fps
    assert tt.plan_windows(t, 300.0) == [tt.Window(0, 0, 30)]


def test_windows_spread_evenly_and_stay_inside_their_part():
    rate_ns = 10_000_000  # 100 fps
    parts = [np.arange(6000) * rate_ns, np.arange(6000) * rate_ns]
    windows = tt.plan_windows(parts, 50.0)  # 5 windows of 10 s = 1000 rows
    assert len(windows) == 5
    assert all(w.n_frames == 1000 for w in windows)
    assert {w.part_index for w in windows} == {0, 1}
    for w in windows:
        assert 0 <= w.start < w.stop <= 6000
    assert windows == sorted(windows, key=lambda w: (w.part_index, w.start))


def test_the_window_length_is_a_duration_not_a_row_count():
    slow = tt.plan_windows([np.arange(100_000) * 40_000_000], 50.0)  # 25 fps
    fast = tt.plan_windows([np.arange(100_000) * 10_000_000], 50.0)  # 100 fps
    assert slow[0].n_frames * 4 == fast[0].n_frames


# --------------------------------------------------------------------------- the run


def test_the_sweep_puts_a_split_subject_back_together(split_run):
    model, recording, run_dir, record, _ = split_run
    assert record["status"] == "complete"
    assert model.conf == track_quality.sweep_detection_conf() == 0.05
    assert model.seen == N_FRAMES // 2  # 5 windows x 2 s at 10 fps
    assert len(record["table"]) == len(track_quality.grid())
    by = {
        (r["conf_threshold"], r["secondary_conf"], r["max_secondary"]): r
        for r in record["table"]
    }
    default = by[(0.5, 0.5, 2)]
    assert default["is_default"]
    assert default["per_recording"][recording.id]["jumpy_pct"] > 30
    pick = record["pick"]
    assert not pick["keep_defaults"]
    assert pick["setting"] == {
        "detection_conf": 0.25,
        "conf_threshold": 0.7,
        "secondary_conf": 0.25,
        "max_secondary": 2,
    }
    chosen = by[(0.7, 0.25, 2)]["per_recording"][recording.id]
    assert chosen["jumpy_pct"] == 0
    assert chosen["tracked_pct"] == 100
    assert chosen["merged_pct"] == 100
    assert chosen["median_area"] == 900
    assert not (run_dir / "sample").exists()  # the scratch detections are gone


def test_the_record_is_schema_versioned_and_says_who_made_it(split_run):
    _, _, run_dir, record, _ = split_run
    text = (run_dir / "tuning.yaml").read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1\n")
    assert record["source"] == {"kind": "standalone"}
    assert record["config"]["sample_seconds"] == 10.0
    assert record["recordings"][0]["sampled_frames"] == N_FRAMES // 2
    assert record["recordings"][0]["frame_rate"] == pytest.approx(FPS)


def test_two_subjects_are_never_merged_into_one(two_run):
    assert two_run["recordings"][0]["multi_subject_share"] == 1.0
    assert two_run["pick"]["setting"]["max_secondary"] == 0
    assert any("more than one subject" in w for w in two_run["warnings"])


def test_progress_moves_inside_one_window_and_names_its_phases(split_run):
    events = split_run[4]
    detecting = [e for e in events if e.phase == "detecting"]
    sweeping = [e for e in events if e.phase == "sweeping"]
    assert any(0 < e.frames_done < e.frames_total for e in detecting)
    assert any(0 < e.frames_done < e.frames_total for e in sweeping)
    assert events[0].message == "loading model"
    assert events[-1].message == "Done"
    assert events.index(detecting[-1]) < events.index(sweeping[0])
    assert all(
        e.eta_s is not None for e in events if 0 < e.frames_done < e.frames_total
    )


def test_the_sweep_eta_is_timed_on_the_sweep_alone(split_run):
    """The sweep is not paced like detection: its ETA divides the sweep's progress by the
    sweep's own elapsed time (``PhaseClock``), never the run's (the PR #171 class)."""
    events = split_run[4]
    sweeping = [e for e in events if e.phase == "sweeping"]
    began = sweeping[0].elapsed_s
    assert began > 0.0
    for e in sweeping:
        if 0 < e.frames_done < e.frames_total:
            spent = e.eta_s * e.frames_done / (e.frames_total - e.frames_done)
            assert spent <= e.elapsed_s - began + 1e-6


def test_a_stop_leaves_the_record_cancelled(tuned_experiment, monkeypatch):
    experiment, recording = tuned_experiment
    _patch_model(monkeypatch, SplitSubject())
    stop = threading.Event()
    stop.set()
    run_dir, record = _run(experiment, recording, None, stop=stop)
    assert record["status"] == "cancelled"
    assert not (run_dir / "sample").exists()


def test_a_crash_leaves_the_record_failed(tuned_experiment, monkeypatch):
    experiment, recording = tuned_experiment

    class Broken(SplitSubject):
        def predict_batch(self, frames, prompts):
            raise RuntimeError("out of memory")

    _patch_model(monkeypatch, Broken())
    with pytest.raises(RuntimeError, match="out of memory"):
        _run(experiment, recording, None)
    record = tunings_store.read_tuning(
        tunings_store.tuning_dir(experiment.dir, "model_001", "tune_001")
    )
    assert record["status"] == "failed"
    assert "out of memory" in record["error"]
