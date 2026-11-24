import contextlib
import time

import cv2
import numpy as np
import pytest
from pycocotools import mask as mask_utils

from detail.core.dataclasses import (
    ComputeTarget,
    Detection,
    DetectionJob,
    FrameResult,
    JobStatus,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.core.protocols import DetectionModel
from detail.core.registries import MODELS, TRAINED_LOADERS, register_model
from detail.executor import rundir
from detail.pipeline import detection
from detail.storage import models as models_store
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage.zarr_io import PartZarrReader

H, W = 24, 32


class _FakeDetector(DetectionModel):
    model_id = "fake_detector"
    supported_cue_types = [TextCue]
    # Accept both so the multi-GPU coordinator tests (which need backend=cuda
    # to take the pool path) and the original cpu tests share one fake.
    supported_backends = ["cpu", "cuda"]
    last_set_conf_threshold: float | None = None

    def load(self, device):
        self.device = device

    def is_ready(self):
        return True

    def set_conf_threshold(self, threshold):
        type(self).last_set_conf_threshold = threshold

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return FrameResult.from_detections(
            frame_idx=0,
            t_ns=0,
            detections=[
                Detection(
                    mask=mask,
                    bbox=(0.25, 0.25, 0.75, 0.75),
                    class_name="mouse",
                    confidence=0.9,
                    object_id=None,
                )
            ],
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


class _BatchDetector(_FakeDetector):
    model_id = "batch_detector"
    supports_batching = True


@contextlib.contextmanager
def _registered_model():
    register_model("fake_detector")(_FakeDetector)
    try:
        yield
    finally:
        MODELS.pop("fake_detector", None)


def _write_video(path, n_frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((H, W, 3), (i * 17) % 256, dtype=np.uint8))
    writer.release()


def _count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _setup(tmp_path):
    proj_dir = experiment_store.create_experiment(tmp_path, "proj")
    recording_store.create_recording(proj_dir, "recording_001")
    recording_dir = experiment_store.recording_path(proj_dir, "recording_001")

    _write_video(recording_dir / "video" / "part_001.mp4", n_frames=5)
    n_frames = _count_frames(recording_dir / "video" / "part_001.mp4")
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(n_frames, dtype=np.int64) * 1000,
    )
    data = recording_store.read_recording(recording_dir)
    data["video"] = {
        "storage": "default",
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
    recording_store.write_recording(recording_dir, data)
    return recording_dir, n_frames


def _job():
    return DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="fake_detector",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


def test_part_frame_counts_uses_timestamp_map_not_cv2(tmp_path):
    # The upfront progress denominator comes from the authoritative timestamp map
    # (Tenet 8), not cv2's CAP_PROP_FRAME_COUNT. A map longer than the decodable
    # video makes the two disagree — a reversion to the cv2 count would return the
    # shorter video length, and the live bar would never hit 100%. (Only the count is
    # asked here: running detection on that pair is a FrameCountMismatchError.)
    recording_dir, n_frames = _setup(tmp_path)
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(n_frames + 5, dtype=np.int64) * 1000,
    )
    data = recording_store.read_recording(recording_dir)
    files = recording_store.video_files(data)
    counts = detection._part_frame_counts(recording_dir, data, files)
    assert counts == {"part_001": n_frames + 5}


def test_detection_runner_writes_zarr_and_payload(tmp_path):
    recording_dir, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _job())

    with _registered_model():
        detection.run(_job(), run_dir, progress_cb=lambda e: None, stop_event=None)

    data = PartZarrReader(run_dir / "part_001.zarr").read()
    assert data["frame_idx"].tolist() == list(range(n_frames))
    assert data["t_ns"].tolist() == [i * 1000 for i in range(n_frames)]
    assert data["detection_idx"].tolist() == [0] * n_frames
    assert [str(x) for x in data["class_name"]] == ["mouse"] * n_frames
    assert np.allclose(data["bbox"][0], [0.25, 0.25, 0.75, 0.75])

    # mask round-trips through COCO RLE
    counts = str(data["mask_rle"][0]).encode("ascii")
    decoded = mask_utils.decode({"counts": counts, "size": [H, W]})
    assert decoded[H // 2, W // 2] == 1
    assert decoded[0, 0] == 0

    env = rundir.read_envelope(run_dir)
    assert env["mask_resolution"] == [H, W]
    assert env["parts"][0]["frames_done"] == n_frames
    assert env["parts"][0]["status"] == "complete"


def test_a_recording_whose_parts_differ_in_size_fails_detection_naming_both(tmp_path):
    """Every door a part comes in by refuses one of another size, so this recording is one
    made before those checks. Left to complete, its run reached every stage that reads
    masks, which decodes and places them at the run's one size: the odd part's masks as
    stripes, its positions in other pixels than the arena and the calibration. It fails here
    instead, and a failed run is read by nothing downstream."""
    from detail.core.errors import UserInputError

    recording_dir, n_frames = _setup(tmp_path)
    big = recording_dir / "video" / "part_002.mp4"
    writer = cv2.VideoWriter(
        str(big), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W * 2, H * 2)
    )
    for i in range(n_frames):
        writer.write(np.full((H * 2, W * 2, 3), i * 17, dtype=np.uint8))
    writer.release()
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_002.csv",
        (np.arange(_count_frames(big), dtype=np.int64) + n_frames) * 1000,
    )
    data = recording_store.read_recording(recording_dir)
    second = dict(data["video"]["files"][0])
    second.update(
        index=1,
        file="video/part_002.mp4",
        timestamps={
            "generated": "video/timestamps/generated/part_002.csv",
            "user": None,
            "active": "generated",
        },
    )
    data["video"]["files"].append(second)
    recording_store.write_recording(recording_dir, data)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _job())

    with (
        _registered_model(),
        pytest.raises(
            UserInputError,
            match=rf"part_001 is {W}×{H} and part_002 is {W * 2}×{H * 2}.*remove part_002",
        ),
    ):
        detection.run(_job(), run_dir, progress_cb=lambda e: None, stop_event=None)
    sizes = [p["mask_resolution"] for p in rundir.read_envelope(run_dir)["parts"]]
    assert sizes == [[H, W], [H * 2, W * 2]]  # what each part was read at, recorded


def test_detection_inline_emits_loading_message_and_multiple_events(tmp_path):
    recording_dir, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _job())
    events = []
    with _registered_model():
        detection.run(_job(), run_dir, progress_cb=events.append, stop_event=None)
    # the inline path used to emit a single event at the end with no message;
    # now it announces the load up front and the bar has a real denominator.
    assert events[0].message == "loading model"
    assert events[0].frames_done == 0 and events[0].frames_total > 0
    assert len(events) >= 2
    assert events[-1].frames_done == n_frames


def test_detection_runner_pushes_conf_threshold_to_model(tmp_path):
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    job = _job()
    job.conf_threshold = 0.37
    rundir.init_run_dir(run_dir, job)
    _FakeDetector.last_set_conf_threshold = None
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    assert _FakeDetector.last_set_conf_threshold == 0.37


def test_detection_conf_threshold_filters(tmp_path):
    recording_dir, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    job = _job()
    job.conf_threshold = 0.95  # above the fake's 0.9
    rundir.init_run_dir(run_dir, job)

    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    assert PartZarrReader(run_dir / "part_001.zarr").n_rows == 0


def test_detection_unknown_model_raises(tmp_path):
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    job = _job()
    job.model_id = "nope"
    rundir.init_run_dir(run_dir, job)

    with pytest.raises(ValueError, match="unknown model_id"):
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)


# --- frame stride (§4) ----------------------------------------------------- #
def test_detection_stride_writes_only_the_frames_it_processed(tmp_path):
    # A strided run keeps *source* frame numbering — rows land on 0, 2, 4, and each
    # carries that source frame's own t_ns. Renumbering (0, 1, 2) would silently
    # misalign every downstream stage, which all join on these two columns.
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    job = _job()
    job.every_n_frames = 2
    rundir.init_run_dir(run_dir, job)

    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    data = PartZarrReader(run_dir / "part_001.zarr").read()
    assert data["frame_idx"].tolist() == [0, 2, 4]
    assert data["t_ns"].tolist() == [0, 2000, 4000]


def test_detection_stride_1_is_every_frame(tmp_path):
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_02"
    run_dir.mkdir(parents=True)
    job = _job()
    rundir.init_run_dir(run_dir, job)

    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    data = PartZarrReader(run_dir / "part_001.zarr").read()
    assert data["frame_idx"].tolist() == [0, 1, 2, 3, 4]


def test_detection_stride_records_its_cadence(tmp_path):
    # The stride has to reach the record: it is what a downstream stage validates
    # against, and what the picker labels. A run that ran thinned and recorded 1
    # would let embedding ask for frames that were never detected.
    recording_dir, _ = _setup(tmp_path)
    data = recording_store.read_recording(recording_dir)
    module = detection.DetectionModule()
    assert (
        module.extra_record_fields({"model_id": "m", "every_n_frames": 4})[
            "every_n_frames"
        ]
        == 4
    )
    assert module.extra_record_fields({"model_id": "m"})["every_n_frames"] == 1
    assert "every_n_frames" in module.identity_fields
    del data


# --- multi-GPU coordinator (§7) ------------------------------------------- #
# These tests stub ProcessPoolExecutor with a synchronous fake so we exercise
# the round-robin assignment, progress aggregation, stop semantics, and
# failure propagation without spawning real subprocesses or needing CUDA.

from concurrent.futures import Future

from detail.pipeline import _detection_worker as _dw


def _completed_future(result):
    """Real concurrent.futures.Future, pre-marked done with a result.
    Needed so concurrent.futures.wait() sees the right internals."""
    fut = Future()
    fut.set_result(result)
    return fut


class _SyncPool:
    """Fake ProcessPoolExecutor that runs submitted callables synchronously
    in the test process. Captures initializer args so tests can assert on
    GPU pinning behavior."""

    def __init__(self, max_workers=1, mp_context=None, initializer=None, initargs=()):
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs
        # Run the initializer once (in-process) so worker state mirrors what
        # a real pool's worker would see.
        if initializer is not None:
            # Translate the real initializer into in-process state population
            # so pool_run_part finds the model. We bypass real CUDA pinning
            # since this is a unit test.
            (
                gpu_idx,
                experiment_root,
                model_id,
                weights_source,
                model_variant,
                prompt_dict,
                conf_threshold,
                batch_size,
                _runners,
            ) = initargs
            from detail.core.dataclasses import Prompt
            from detail.models.resolve import resolve_detection_model

            model = resolve_detection_model(
                experiment_root, model_id, weights_source, model_variant
            )
            model.load("cpu")
            model.set_conf_threshold(conf_threshold)
            _dw._WORKER_STATE["model"] = model
            _dw._WORKER_STATE["prompt"] = Prompt.from_dict(prompt_dict)
            _dw._WORKER_STATE["conf_threshold"] = conf_threshold
            _dw._WORKER_STATE["batch_size"] = batch_size
            _dw._WORKER_STATE["gpu_idx"] = gpu_idx

    def submit(self, fn, *args, **kwargs):
        return _completed_future(fn(*args, **kwargs))

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _multi_gpu_job(device_indices):
    return DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="fake_detector",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("cuda", list(device_indices), None, None),
        created_at="2026-01-01T00:00:00Z",
    )


def _patch_pool(monkeypatch):
    """Swap ProcessPoolExecutor + the spawn mp context so the coordinator
    runs entirely in-process."""
    monkeypatch.setattr(detection, "ProcessPoolExecutor", _SyncPool)

    class _NoopCtx:
        pass

    import multiprocessing as mp_mod

    monkeypatch.setattr(mp_mod, "get_context", lambda name: _NoopCtx())


def test_multi_gpu_short_circuits_to_inline_for_one_device(tmp_path, monkeypatch):
    """compute.device_indices=[0] → inline path; pool never instantiated."""
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    pool_constructed = []

    class _Sentinel(_SyncPool):
        def __init__(self, *a, **kw):
            pool_constructed.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(detection, "ProcessPoolExecutor", _Sentinel)
    job = _multi_gpu_job([0])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    assert pool_constructed == []  # inline path took over


def test_multi_gpu_cpu_backend_short_circuits_to_inline(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    pool_constructed = []

    class _Sentinel(_SyncPool):
        def __init__(self, *a, **kw):
            pool_constructed.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(detection, "ProcessPoolExecutor", _Sentinel)
    job = _job()  # cpu backend
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    assert pool_constructed == []


def test_multi_gpu_constructs_one_pool_per_gpu(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    gpu_args = []

    class _Recorder(_SyncPool):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            gpu_args.append(kw.get("initargs", ())[0])

    monkeypatch.setattr(detection, "ProcessPoolExecutor", _Recorder)
    monkeypatch.setattr(
        __import__("multiprocessing"), "get_context", lambda name: object()
    )
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    assert gpu_args == [0, 1]  # one pool per GPU, each initargs[0] is the gpu_idx


def test_multi_gpu_writes_per_part_zarrs(tmp_path, monkeypatch):
    """Two-part recording with two GPUs: both per-part zarrs exist + readable."""
    recording_dir, _ = _setup(tmp_path)
    # add a second video part
    _write_video(recording_dir / "video" / "part_002.mp4", n_frames=4)
    n2 = _count_frames(recording_dir / "video" / "part_002.mp4")
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_002.csv",
        np.arange(n2, dtype=np.int64) * 1000,
    )
    data = recording_store.read_recording(recording_dir)
    data["video"]["files"].append(
        {
            "index": 1,
            "file": "video/part_002.mp4",
            "timestamps": {
                "generated": "video/timestamps/generated/part_002.csv",
                "user": None,
                "active": "generated",
            },
        }
    )
    recording_store.write_recording(recording_dir, data)

    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    # both per-part zarrs exist + schema-correct
    for part_id in ("part_001.zarr", "part_002.zarr"):
        zarr_path = run_dir / part_id
        assert zarr_path.is_dir()
        rows = PartZarrReader(zarr_path).read()
        assert rows["frame_idx"].shape[0] > 0
        assert rows["class_name"].shape[0] == rows["frame_idx"].shape[0]


def test_multi_gpu_more_parts_than_gpus(tmp_path, monkeypatch):
    """4 parts on 2 GPUs: all 4 processed."""
    recording_dir, _ = _setup(tmp_path)
    for i in range(2, 5):  # parts 2, 3, 4
        _write_video(recording_dir / "video" / f"part_{i:03d}.mp4", n_frames=4)
        n = _count_frames(recording_dir / "video" / f"part_{i:03d}.mp4")
        recording_store.write_timestamps_csv(
            recording_dir / "video" / "timestamps" / "generated" / f"part_{i:03d}.csv",
            np.arange(n, dtype=np.int64) * 1000,
        )
    data = recording_store.read_recording(recording_dir)
    for i in range(2, 5):
        data["video"]["files"].append(
            {
                "index": i - 1,
                "file": f"video/part_{i:03d}.mp4",
                "timestamps": {
                    "generated": f"video/timestamps/generated/part_{i:03d}.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        )
    recording_store.write_recording(recording_dir, data)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    env = rundir.read_envelope(run_dir)
    part_ids = sorted(p["id"] for p in env["parts"])
    assert part_ids == ["part_001", "part_002", "part_003", "part_004"]


def test_multi_gpu_one_part_runs_on_one_gpu(tmp_path, monkeypatch):
    """1 part, 2 GPUs requested: completes without error (extra GPU idle)."""
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    env = rundir.read_envelope(run_dir)
    assert len(env["parts"]) == 1
    assert env["parts"][0]["status"] == "complete"


def test_multi_gpu_mask_resolution_captured_once(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    env = rundir.read_envelope(run_dir)
    assert env["mask_resolution"] == [H, W]


def test_multi_gpu_progress_aggregated_across_parts(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    _write_video(recording_dir / "video" / "part_002.mp4", n_frames=4)
    n2 = _count_frames(recording_dir / "video" / "part_002.mp4")
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_002.csv",
        np.arange(n2, dtype=np.int64) * 1000,
    )
    data = recording_store.read_recording(recording_dir)
    data["video"]["files"].append(
        {
            "index": 1,
            "file": "video/part_002.mp4",
            "timestamps": {
                "generated": "video/timestamps/generated/part_002.csv",
                "user": None,
                "active": "generated",
            },
        }
    )
    recording_store.write_recording(recording_dir, data)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    events: list = []
    with _registered_model():
        detection.run(job, run_dir, progress_cb=events.append, stop_event=None)
    # at least one event with parts_total=2 emitted
    assert events
    assert events[-1].parts_total == 2


def test_multi_gpu_worker_failure_raises_run_failed(tmp_path, monkeypatch):
    """If pool_run_part returns a failed status for a part, the coordinator
    raises after aggregation (so the executor envelope sees status=failed)."""
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    # Make pool_run_part return a failed dict regardless of input
    def fake_pool_run_part(args):
        return {
            "id": "part_001",
            "file": args["part"]["file"],
            "gpu": 0,
            "status": "failed",
            "frames_done": 0,
            "frames_total": 0,
            "mask_resolution": None,
            "dropped_frames": [],
            "error": "simulated worker failure",
        }

    monkeypatch.setattr(_dw, "pool_run_part", fake_pool_run_part)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        with pytest.raises(RuntimeError, match="detection failed"):
            detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    # job.json still got the parts payload with status=failed
    env = rundir.read_envelope(run_dir)
    assert env["parts"][0]["status"] == "failed"


# --- real-subprocess integration (spawn pool, CPU-only fake model) -------- #
_FAKE_CUDA_MODEL_SOURCE = """
import numpy as np

from detail.core.dataclasses import Detection, FrameResult, TextCue
from detail.core.protocols import DetectionModel
from detail.core.registries import register_model


@register_model("fake_cuda_detector")
class FakeCudaDetector(DetectionModel):
    model_id = "fake_cuda_detector"
    supported_cue_types = [TextCue]
    # Declares cuda so the coordinator picks the multi-GPU path, but the
    # load/predict implementation never actually touches torch.cuda — works
    # on CI without GPUs.
    supported_backends = ["cuda"]

    def load(self, device):
        self.device = device

    def is_ready(self):
        return True

    def set_conf_threshold(self, threshold):
        pass

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return FrameResult.from_detections(
            0, 0, [Detection(mask, (0.25, 0.25, 0.75, 0.75), "mouse", 0.9, None)]
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]
"""


def test_multi_gpu_real_spawn_pool(tmp_path, monkeypatch):
    """Spawns the actual ProcessPoolExecutor with device_indices=[0, 1] using
    a CPU-only fake model. Verifies the spawn-context plumbing + initializer
    + cross-process pickle work end-to-end on CI without real GPUs."""
    import os

    recording_dir, n_frames = _setup(tmp_path)
    # second part so both pool workers have something to do
    _write_video(recording_dir / "video" / "part_002.mp4", n_frames=4)
    n2 = _count_frames(recording_dir / "video" / "part_002.mp4")
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_002.csv",
        np.arange(n2, dtype=np.int64) * 1000,
    )
    data = recording_store.read_recording(recording_dir)
    data["video"]["files"].append(
        {
            "index": 1,
            "file": "video/part_002.mp4",
            "timestamps": {
                "generated": "video/timestamps/generated/part_002.csv",
                "user": None,
                "active": "generated",
            },
        }
    )
    recording_store.write_recording(recording_dir, data)

    # write the fake model module + put it on the path the spawn workers see
    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    (mod_dir / "fake_cuda_model_mod.py").write_text(
        _FAKE_CUDA_MODEL_SOURCE, encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(mod_dir))
    monkeypatch.setenv(
        "PYTHONPATH", str(mod_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    # make the spawn workers register the fake on init
    from detail.api import _runner_modules

    monkeypatch.setattr(
        _runner_modules,
        "PIPELINE_RUNNERS",
        _runner_modules.PIPELINE_RUNNERS + ("fake_cuda_model_mod",),
    )

    # also register in this process so the coordinator's pre-spawn model lookup passes
    import importlib

    fake_mod = importlib.import_module("fake_cuda_model_mod")

    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    job = DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="fake_cuda_detector",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("cuda", [0, 1], None, None),
        created_at="2026-01-01T00:00:00Z",
    )
    rundir.init_run_dir(run_dir, job)

    try:
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    finally:
        MODELS.pop("fake_cuda_detector", None)

    env = rundir.read_envelope(run_dir)
    assert len(env["parts"]) == 2
    assert all(p["status"] == "complete" for p in env["parts"])
    # both per-part zarrs readable + populated
    assert PartZarrReader(run_dir / "part_001.zarr").n_rows == n_frames
    assert PartZarrReader(run_dir / "part_002.zarr").n_rows == n2


def test_multi_gpu_stop_event_marks_run_failed(tmp_path, monkeypatch):
    """Stop event mid-run → coordinator touches stop.flag, workers should see
    it (but our SyncPool runs synchronously so we simulate the failed return
    directly). Verifies the coordinator escalates failed parts to a raised
    RuntimeError = run status=failed via the executor envelope."""
    recording_dir, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    class _StopEvent:
        def is_set(self):
            return True

    def fake_pool_run_part(args):
        # Mimic a worker that saw stop.flag at its first poll boundary.
        return {
            "id": "part_001",
            "file": args["part"]["file"],
            "gpu": 0,
            "status": "failed",
            "frames_done": 1,
            "frames_total": 5,
            "mask_resolution": [H, W],
            "dropped_frames": [],
            "error": "stop.flag detected during detection",
        }

    monkeypatch.setattr(_dw, "pool_run_part", fake_pool_run_part)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        with pytest.raises(RuntimeError, match="detection failed"):
            detection.run(
                job, run_dir, progress_cb=lambda e: None, stop_event=_StopEvent()
            )
    # coordinator touched the stop flag for any in-flight workers
    assert (run_dir / "stop.flag").exists()


# --------------------------------------------------------------------------- #
# validate(): closed-set / open-vocab prompt rules + trained-model resolution
# --------------------------------------------------------------------------- #
class _ClosedSetModel(DetectionModel):
    model_id = "closed_set_stub"
    supported_cue_types = []  # closed-set: detects its own classes, no prompt
    supported_backends = ["cpu"]

    def load(self, device): ...

    def predict(self, frame, prompt):
        raise NotImplementedError

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


class _TrainedLoaderStub(DetectionModel):
    model_id = "trained_loader_stub"
    supported_cue_types = []
    supported_backends = ["cpu"]

    def __init__(self, weights_path, classes):
        super().__init__(None)
        self.weights_path = weights_path
        self.classes = classes

    def load(self, device): ...

    def predict(self, frame, prompt):
        raise NotImplementedError

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


def _cpu():
    return ComputeTarget("cpu", None, None, None)


def test_validate_makes_no_network_calls_for_an_installed_model(
    tmp_path, monkeypatch, fake_hf_snapshot
):
    """An installed model's validate() makes no network call.

    The regression where one Hub 504 failed most of a batch, at the seam that caused it.

    validate() runs once per recording up front and again per recording inside the
    coordinator, so a network call here is multiplied across the whole batch and
    When a Hub 504 landed hours into a long embedding run, every remaining recording
    died in milliseconds. An installed model
    must resolve entirely from disk.
    """
    import huggingface_hub

    from detail.models.sam3 import SAM3

    recording_dir, _ = _setup(tmp_path)
    fake_hf_snapshot(SAM3.hf_repo, files=SAM3.hf_files)
    # SAM3 readiness now also checks the sam3 *package* imports (is_importable, no
    # network) — pin it present so this test exercises the installed-weights path.
    monkeypatch.setattr("detail.models.sam3.is_importable", lambda m: True)

    def _boom(*a, **k):
        raise AssertionError("validate() reached the network for installed weights")

    monkeypatch.setattr(huggingface_hub, "auth_check", _boom)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)

    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "sam3",
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "compute": ComputeTarget(
            "cuda", None, None, None
        ),  # SAM3 is CUDA-only; None indices skip the cuda preflight
    }
    assert detection.DetectionModule().validate(recording_dir, data, config) is config


def test_validate_rejects_unsupported_backend(tmp_path):
    # SAM3 is CUDA-only — a cpu/mps compute fails at launch with a clear message,
    # not deep in the worker (the trim + this validate check).
    recording_dir, _ = _setup(tmp_path)
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "sam3",
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "compute": _cpu(),
    }
    with pytest.raises(ValueError, match="does not support the 'cpu' backend"):
        detection.DetectionModule().validate(recording_dir, data, config)


def test_validate_closed_set_accepts_empty_prompt(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    monkeypatch.setitem(MODELS, "closed_set_stub", _ClosedSetModel)
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "closed_set_stub",
        "prompt": Prompt(targets=[]),
        "compute": _cpu(),
    }
    assert detection.DetectionModule().validate(recording_dir, data, config) is config


def test_validate_closed_set_rejects_prompt_cues(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    monkeypatch.setitem(MODELS, "closed_set_stub", _ClosedSetModel)
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "closed_set_stub",
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "compute": _cpu(),
    }
    with pytest.raises(ValueError, match="closed-set"):
        detection.DetectionModule().validate(recording_dir, data, config)


def test_validate_open_vocab_requires_a_prompt(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    monkeypatch.setitem(MODELS, "fake_detector", _FakeDetector)  # open-vocab
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "fake_detector",
        "prompt": Prompt(targets=[]),
        "compute": _cpu(),
    }
    with pytest.raises(ValueError, match="at least one prompt target"):
        detection.DetectionModule().validate(recording_dir, data, config)


def test_validate_rejects_batch_on_non_batching_model(tmp_path, monkeypatch):
    # _FakeDetector inherits supports_batching=False → batch_size>1 must be rejected.
    recording_dir, _ = _setup(tmp_path)
    monkeypatch.setitem(MODELS, "fake_detector", _FakeDetector)
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "fake_detector",
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "compute": _cpu(),
        "batch_size": 2,
    }
    with pytest.raises(ValueError, match="does not support batched inference"):
        detection.DetectionModule().validate(recording_dir, data, config)


def test_validate_defaults_batch_to_one_for_non_batching_model(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    monkeypatch.setitem(
        MODELS, "fake_detector", _FakeDetector
    )  # supports_batching=False
    data = recording_store.read_recording(recording_dir)
    config = {  # batch_size unset → resolves to 1
        "model_id": "fake_detector",
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "compute": _cpu(),
    }
    out = detection.DetectionModule().validate(recording_dir, data, config)
    assert out["batch_size"] == 1


def test_validate_defaults_batch_to_four_for_batching_model(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    monkeypatch.setitem(
        MODELS, "batch_detector", _BatchDetector
    )  # supports_batching=True
    data = recording_store.read_recording(recording_dir)
    config = {  # batch_size unset → resolves to 4 (batch-capable default)
        "model_id": "batch_detector",
        "prompt": Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        "compute": _cpu(),
    }
    out = detection.DetectionModule().validate(recording_dir, data, config)
    assert out["batch_size"] == 4


def test_detection_batched_end_to_end(tmp_path):
    # batch_size>1 flows through run() and produces the same per-frame rows.
    recording_dir, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    job = DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="fake_detector",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-01-01T00:00:00Z",
        batch_size=4,
    )
    rundir.init_run_dir(run_dir, job)
    with _registered_model():
        detection.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    rows = PartZarrReader(run_dir / "part_001.zarr").read()
    assert rows["frame_idx"].shape[0] == n_frames  # one detection/frame at batch_size=4
    assert rows["t_ns"].tolist() == [i * 1000 for i in range(n_frames)]


def test_validate_resolves_trained_model(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    proj = recording_dir.parents[1]
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _TrainedLoaderStub)
    mdir = models_store.model_dir(proj, "model_001")
    (mdir / "run").mkdir(parents=True)
    (mdir / "run" / "best.pt").write_bytes(b"w")
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "model_family": "stubfam",
            "status": "complete",
            "classes": [{"id": 0, "name": "mouse"}],
            "weights": {"best": "run/best.pt", "last": None},
        },
    )
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "model_001",
        "weights_source": "trained",
        "prompt": Prompt(targets=[]),
        "compute": _cpu(),
    }
    assert detection.DetectionModule().validate(recording_dir, data, config) is config


def test_validate_trained_model_must_be_complete(tmp_path, monkeypatch):
    recording_dir, _ = _setup(tmp_path)
    proj = recording_dir.parents[1]
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _TrainedLoaderStub)
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "model_family": "stubfam", "status": "running"},
    )
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "model_001",
        "weights_source": "trained",
        "prompt": Prompt(targets=[]),
        "compute": _cpu(),
    }
    with pytest.raises(ValueError, match="not complete"):
        detection.DetectionModule().validate(recording_dir, data, config)


def test_validate_asks_the_install_whether_it_runs_on_the_backend(
    tmp_path, monkeypatch
):
    """``supported_backends`` is the class's claim; ``Model.check_backend`` is this
    install's (§5). A model that supports cpu in principle but cannot run on it on this
    machine refuses at validate, before the run starts, with the model's own reason."""

    class _CpuRefusing(_ClosedSetModel):
        model_id = "cpu_refusing_stub"

        def check_backend(self, backend):
            raise ValueError(
                f"this install of {self.model_id} cannot run on {backend}: torch was "
                "built without it"
            )

    recording_dir, _ = _setup(tmp_path)
    monkeypatch.setitem(MODELS, "cpu_refusing_stub", _CpuRefusing)
    data = recording_store.read_recording(recording_dir)
    config = {
        "model_id": "cpu_refusing_stub",
        "prompt": Prompt(targets=[]),
        "compute": _cpu(),
    }
    with pytest.raises(
        ValueError, match="cannot run on cpu: torch was built without it"
    ):
        detection.DetectionModule().validate(recording_dir, data, config)


def test_a_cancel_mid_part_reaches_the_inline_loop_and_records_cancelled(
    tmp_path, monkeypatch
):
    """#28 finding 4. The inline (single-device) path touched ``stop.flag`` only between
    parts, so on a one-part recording a cancel never reached the frame loop: the part ran to
    its end and the worker recorded ``partial`` for a stage with no resume. This drives the
    real worker (``run_job``) and the real runner, cancels while frame 2 of the only part is
    being detected, and holds that frame until the flag appears."""
    from detail.executor.worker import run_job

    monkeypatch.setattr(detection, "_PROGRESS_POLL_INTERVAL_S", 0.02)
    recording_dir, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _job())
    stop_flag = run_dir / "stop.flag"
    calls = {"n": 0}

    class _CancelledMidPart(_FakeDetector):
        def predict(self, frame, prompt):
            calls["n"] += 1
            if calls["n"] == 2:
                rundir.request_cancel(run_dir)
                deadline = time.monotonic() + 5.0
                while not stop_flag.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
            return super().predict(frame, prompt)

    register_model("fake_detector")(_CancelledMidPart)
    try:
        status = run_job(run_dir)
    finally:
        MODELS.pop("fake_detector", None)
    assert stop_flag.exists(), "the cancel never reached the part being detected"
    assert calls["n"] < n_frames, "the part ran to its end after the cancel"
    assert status == JobStatus.CANCELLED
    assert rundir.read_status(run_dir) == JobStatus.CANCELLED
