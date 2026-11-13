"""Tests for the prompt-test pipeline (Section 8).

Uses a mocked DetectionModel so no GPU / network is involved. Verifies sampling,
per-prompt result shape, save layout, and reproducibility under a seed.
"""

import contextlib
import json

import cv2
import numpy as np
import pytest
import yaml

from detail.configs.display import KEEP_ALL_PROPOSALS
from detail.core.dataclasses import (
    ComputeTarget,
    Detection,
    FrameResult,
    ModelReadiness,
    Prompt,
    PromptTarget,
    PromptTestJob,
    TextCue,
)
from detail.core.protocols import DetectionModel
from detail.core.registries import (
    MODELS,
    TRAINED_LOADERS,
    register_model,
    register_trained_loader,
)
from detail.pipeline.prompt_test import _sample_frames, run_prompt_test
from detail.storage import models as models_store
from detail.storage import recording as recording_store

H, W = 24, 32


class _FakeDetector(DetectionModel):
    model_id = "fake_pt_detector"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]
    last_set_conf_threshold: float | None = None

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def set_conf_threshold(self, threshold):
        type(self).last_set_conf_threshold = threshold

    def predict(self, frame, prompt):
        # one detection per call, with a small bbox in the upper-left and a mask
        # the size of the frame (so the overlay code has something to draw)
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return FrameResult.from_detections(
            0, 0, [Detection(mask, (0.1, 0.1, 0.3, 0.3), "mouse", 0.9, None)]
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


@contextlib.contextmanager
def _registered_fake():
    register_model("fake_pt_detector")(_FakeDetector)
    try:
        yield
    finally:
        MODELS.pop("fake_pt_detector", None)


class _FakeClosedSet(_FakeDetector):
    """Trained (closed-set) model: no cues; one predict returns every class."""

    supported_cue_types = []

    def __init__(self, weights_path=None, classes=None):
        self._classes = list(classes or [])

    @property
    def classes(self):
        return list(self._classes)

    def readiness(self):
        return ModelReadiness(
            ready=True, reason="ok", message="", action_url=None, repo_id=None
        )

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 2, : w // 2] = True
        return FrameResult.from_detections(
            0,
            0,
            [
                Detection(mask, (0.1, 0.1, 0.3, 0.3), "mouse", 0.9, None),
                Detection(mask, (0.4, 0.4, 0.6, 0.6), "cage", 0.8, None),
            ],
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


@contextlib.contextmanager
def _registered_closed_set_loader():
    register_trained_loader("fake_closed")(_FakeClosedSet)
    try:
        yield
    finally:
        TRAINED_LOADERS.pop("fake_closed", None)


def _write_trained_model(experiment_root, classes=("mouse", "cage")):
    mdir = models_store.model_dir(experiment_root, "model_001")
    mdir.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "status": "complete",
            "model_family": "fake_closed",
            "classes": [{"id": i, "name": n} for i, n in enumerate(classes)],
            "weights": {"best": "best.pt", "last": None},
        },
    )


def _write_video(path, n_frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()


def _count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _seed_recording(recording_dir, *, n_parts=1, n_frames=8):
    """Drop n_parts video files into recording_dir with synthetic timestamps."""
    files = []
    cumulative = 0
    for i in range(n_parts):
        part_name = f"part_{i + 1:03d}"
        _write_video(recording_dir / "video" / f"{part_name}.mp4", n_frames)
        actual_n = _count_frames(recording_dir / "video" / f"{part_name}.mp4")
        ts_rel = f"video/timestamps/generated/{part_name}.csv"
        recording_store.write_timestamps_csv(
            recording_dir / ts_rel,
            np.arange(cumulative, cumulative + actual_n, dtype=np.int64) * 1000,
        )
        cumulative += actual_n
        files.append(
            {
                "index": i,
                "file": f"video/{part_name}.mp4",
                "timestamps": {
                    "generated": ts_rel,
                    "user": None,
                    "active": "generated",
                },
            }
        )
    data = recording_store.read_recording(recording_dir)
    data["video"] = {"storage": "default", "files": files}
    recording_store.write_recording(recording_dir, data)
    return cumulative  # total frames in recording


def _job(prompts, *, n=2):
    return PromptTestJob(
        recording_id="recording_001",
        model_id="fake_pt_detector",
        prompts=prompts,
        n_sample_frames=n,
        compute=ComputeTarget("cpu", None, None, None),
    )


def _one_prompt():
    return Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))])


# --- _sample_frames --------------------------------------------------------- #
def test_sample_frames_returns_requested_count():
    samples = _sample_frames([10, 10, 10], 5, seed=0)
    assert len(samples) == 5


def test_sample_frames_no_duplicates_within_part():
    samples = _sample_frames([10, 10, 10], 15, seed=0)
    assert len(set(samples)) == 15  # unique (part, frame) pairs


def test_sample_frames_seed_is_deterministic():
    a = _sample_frames([10, 10, 10], 5, seed=42)
    b = _sample_frames([10, 10, 10], 5, seed=42)
    assert a == b


def test_sample_frames_raises_when_n_exceeds_total():
    with pytest.raises(ValueError, match="exceeds total"):
        _sample_frames([5], 10, seed=0)


def test_sample_frames_spreads_across_parts():
    samples = _sample_frames([4, 4, 4], 9, seed=0)
    parts_hit = {p for p, _ in samples}
    assert parts_hit == {0, 1, 2}


def test_sample_frames_draws_only_inside_windows():
    # Five frames in the windows, five asked for: every one of them, and nothing else.
    samples = _sample_frames([10, 10, 10], 5, seed=0, windows=[(7, 10), (0, 0), (0, 2)])
    assert samples == [(0, 7), (0, 8), (0, 9), (2, 0), (2, 1)]


# --- time range ------------------------------------------------------------- #
def test_time_range_samples_only_frames_inside_it(recording_dir):
    _seed_recording(recording_dir, n_parts=2, n_frames=8)  # frame g sits at g * 1000 ns
    job = _job([_one_prompt()], n=4)
    job.t_start_ns, job.t_end_ns = 6000, 12000  # frames 6, 7 of part 0; 8-11 of part 1
    with _registered_fake():
        results = run_prompt_test(recording_dir, job, seed=0)
    assert len(results) == 4
    assert all(6000 <= r.t_ns < 12000 for r in results)


def test_time_range_open_end_runs_to_the_end(recording_dir):
    _seed_recording(recording_dir, n_parts=2, n_frames=8)
    job = _job([_one_prompt()], n=3)
    job.t_start_ns = 13000  # the last three frames
    with _registered_fake():
        results = run_prompt_test(recording_dir, job, seed=0)
    assert sorted(r.t_ns for r in results) == [13000, 14000, 15000]


def test_time_range_too_short_for_the_sample_raises(recording_dir):
    _seed_recording(recording_dir, n_frames=8)
    job = _job([_one_prompt()], n=3)
    job.t_start_ns, job.t_end_ns = 2000, 4000
    with _registered_fake():
        with pytest.raises(ValueError, match="holds 2 frame.*fewer than the 3"):
            run_prompt_test(recording_dir, job, seed=0)


# --- end-to-end (mocked model) --------------------------------------------- #
def test_returns_one_result_per_sampled_frame(recording_dir):
    _seed_recording(recording_dir, n_frames=8)
    with _registered_fake():
        results = run_prompt_test(
            recording_dir, _job([_one_prompt(), _one_prompt()], n=3), seed=0
        )
    assert len(results) == 3
    for r in results:
        assert set(r.results) == {"prompt_0", "prompt_1"}
        assert r.frame_image.shape == (H, W, 3)
        # each prompt produces a FrameResult with one Detection (per the fake)
        assert len(r.results["prompt_0"].detections()) == 1
        assert r.results["prompt_0"].frame_idx == r.frame_idx
        assert r.results["prompt_0"].t_ns == r.t_ns


def test_per_prompt_labels_are_sequential(recording_dir):
    _seed_recording(recording_dir, n_frames=4)
    prompts = [_one_prompt(), _one_prompt(), _one_prompt()]
    with _registered_fake():
        results = run_prompt_test(recording_dir, _job(prompts, n=2), seed=0)
    assert list(results[0].results) == ["prompt_0", "prompt_1", "prompt_2"]


def test_prompt_test_runs_the_model_wide_open(recording_dir):
    """SAM3 drops proposals inside its own processor before returning them, so a
    threshold pushed in here would destroy the low-confidence rows the display slider
    exists to show. The preview must ask for everything."""
    _seed_recording(recording_dir, n_frames=4)
    _FakeDetector.last_set_conf_threshold = None
    with _registered_fake():
        run_prompt_test(recording_dir, _job([_one_prompt()], n=2), seed=0)
    assert _FakeDetector.last_set_conf_threshold == KEEP_ALL_PROPOSALS


def test_results_keep_every_proposal_for_the_caller_to_filter(recording_dir):
    """Nothing is dropped on the way out: every detection comes back with its own
    confidence, and choosing a threshold is the reader's job."""
    _seed_recording(recording_dir, n_frames=4)
    with _registered_fake():
        results = run_prompt_test(recording_dir, _job([_one_prompt()], n=2), seed=0)
    for r in results:
        dets = r.results["prompt_0"].detections()
        assert dets  # the fake returns one detection at 0.9
        assert all(
            d.confidence == pytest.approx(0.9) for d in dets
        )  # float32 rows (§5)
        # The filtering a view does is a plain comparison over what came back.
        assert [d for d in dets if d.confidence >= 0.99] == []


def test_on_result_streams_and_stops_accumulating(recording_dir):
    """The memory guard: a caller that takes results one at a time gets them all, and
    the returned list stays empty so full-resolution masks are not held for every
    sampled frame at once."""
    _seed_recording(recording_dir, n_frames=6)
    seen = []
    with _registered_fake():
        returned = run_prompt_test(
            recording_dir,
            _job([_one_prompt()], n=3),
            seed=0,
            on_result=seen.append,
        )
    assert len(seen) == 3
    assert returned == []
    assert all(r.results["prompt_0"].detections() for r in seen)


def test_on_result_with_save_as_is_refused(recording_dir):
    _seed_recording(recording_dir, n_frames=4)
    with _registered_fake():
        with pytest.raises(ValueError, match="one or the other"):
            run_prompt_test(
                recording_dir,
                _job([_one_prompt()], n=2),
                save_as="x",
                on_result=lambda _r: None,
            )


def test_no_video_parts_raises(recording_dir):
    # The conftest fixture pre-populates one video file entry — clear it so the
    # "no video parts" path is reached.
    data = recording_store.read_recording(recording_dir)
    data["video"] = {"storage": "default", "files": []}
    recording_store.write_recording(recording_dir, data)
    with _registered_fake():
        with pytest.raises(ValueError, match="no video parts"):
            run_prompt_test(recording_dir, _job([_one_prompt()], n=1))


def test_unknown_model_id_raises(recording_dir):
    _seed_recording(recording_dir, n_frames=4)
    job = _job([_one_prompt()], n=1)
    job.model_id = "no_such_model"
    with pytest.raises(ValueError, match="unknown model"):
        run_prompt_test(recording_dir, job)


# --- closed-set (trained model) preview (#16) ------------------------------- #
def test_closed_set_splits_one_prediction_into_per_class_columns(recording_dir):
    _seed_recording(recording_dir, n_frames=6)
    experiment_root = recording_dir.parents[1]
    with _registered_closed_set_loader():
        _write_trained_model(experiment_root)
        job = PromptTestJob(
            recording_id="recording_001",
            model_id="model_001",  # resolves to the trained loader, not MODELS
            prompts=[
                Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
                Prompt(targets=[PromptTarget("cage", TextCue("cage"))]),
            ],
            n_sample_frames=2,
            compute=ComputeTarget("cpu", None, None, None),
        )
        results = run_prompt_test(recording_dir, job)
    assert results
    for r in results:  # each column filtered to its own class
        assert set(r.results) == {"prompt_0", "prompt_1"}
        assert [d.class_name for d in r.results["prompt_0"].detections()] == ["mouse"]
        assert [d.class_name for d in r.results["prompt_1"].detections()] == ["cage"]


def test_closed_set_save_writes_class_columns(recording_dir):
    _seed_recording(recording_dir, n_frames=6)
    with _registered_closed_set_loader():
        _write_trained_model(recording_dir.parents[1])
        job = PromptTestJob(
            recording_id="recording_001",
            model_id="model_001",
            prompts=[
                Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
                Prompt(targets=[PromptTarget("cage", TextCue("cage"))]),
            ],
            n_sample_frames=2,
            compute=ComputeTarget("cpu", None, None, None),
        )
        run_prompt_test(recording_dir, job, save_as="td01")
    out = recording_dir / "detections" / "prompt_tests" / "td01"
    meta = json.loads((out / "test.json").read_text(encoding="utf-8-sig"))
    # the grid header renders each column via its prompt's class_name.
    names = [p["prompt"]["targets"][0]["class_name"] for p in meta["prompts"]]
    assert names == ["mouse", "cage"]
    assert list((out / "frames").glob("*.jpg"))  # an overlay per (frame, column)


# --- save_as: test.json + overlay JPGs -------------------------------------- #
def test_save_writes_test_json_and_overlay_jpgs(recording_dir):
    _seed_recording(recording_dir, n_frames=8)
    with _registered_fake():
        results = run_prompt_test(
            recording_dir,
            _job([_one_prompt(), _one_prompt()], n=3),
            save_as="test_01",
            seed=0,
        )
    out = recording_dir / "detections" / "prompt_tests" / "test_01"
    assert (out / "test.json").is_file()
    test_json = json.loads((out / "test.json").read_text(encoding="utf-8-sig"))
    assert test_json["model_id"] == "fake_pt_detector"
    assert test_json["n_sample_frames"] == 3
    assert len(test_json["sampled_frames"]) == 3
    assert [p["label"] for p in test_json["prompts"]] == ["prompt_0", "prompt_1"]
    # color mapping per prompt, drawn from the Okabe-Ito palette
    assert set(test_json["prompt_colors"]) == {"prompt_0", "prompt_1"}
    for rgb in test_json["prompt_colors"].values():
        assert len(rgb) == 3 and all(0 <= c <= 255 for c in rgb)
    # different prompts get different colors (palette has > 1 entry; seeded)
    assert (
        test_json["prompt_colors"]["prompt_0"] != test_json["prompt_colors"]["prompt_1"]
    )

    # one JPG per (sample frame, prompt) pair
    frames_dir = out / "frames"
    jpgs = sorted(frames_dir.glob("*.jpg"))
    assert len(jpgs) == len(results) * 2
    # each JPG is readable and matches the source frame resolution
    img = cv2.imread(str(jpgs[0]))
    assert img is not None
    assert img.shape == (H, W, 3)


def test_save_omitted_by_default_no_disk_artifacts(recording_dir):
    _seed_recording(recording_dir, n_frames=4)
    with _registered_fake():
        run_prompt_test(recording_dir, _job([_one_prompt()], n=2), seed=0)
    assert not (recording_dir / "detections" / "prompt_tests").exists()


def test_does_not_register_in_recording_yaml(recording_dir):
    _seed_recording(recording_dir, n_frames=4)
    with _registered_fake():
        run_prompt_test(
            recording_dir, _job([_one_prompt()], n=2), save_as="test_01", seed=0
        )
    data = yaml.safe_load(
        (recording_dir / "recording.yaml").read_text(encoding="utf-8-sig")
    )
    assert "prompt_test_runs" not in data
    assert "detection_runs" not in data
