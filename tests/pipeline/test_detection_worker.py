"""Unit tests for pipeline/_detection_worker.run_part.

Calls the worker function directly (no subprocess) with fake models so we can
exercise the per-frame loop, progress JSON writes, stop.flag polling, error
capture, and the shape of the returned status dict.

Class guarded here: **a sequential pass that pairs cv2's i-th frame with ``map[i]``
while the two count differently.** A surplus frame was written with ``t_ns = -1`` and a
shortfall was recorded as frames dropped at the end — but an edit-list trimmed file
loses its frames at the *start*, so every row of the part was misplaced and the record
said only "5 dropped". The rule (``run_part``'s docstring): a surplus raises
``FrameCountMismatchError``; a shortfall is a dropped tail only when the file's packets
agree with the map, and raises otherwise.
"""

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from detail.core.dataclasses import (
    Detection,
    FrameResult,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.pipeline import _detection_worker
from detail.storage import recording as recording_store
from detail.storage import video as video_store
from detail.storage.zarr_io import PartZarrReader

H, W = 24, 32


def _one_detection(frame):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return FrameResult.from_detections(
        0, 0, [Detection(mask, (0.25, 0.25, 0.75, 0.75), "mouse", 0.9, None)]
    )


class _FakeModel:
    """Returns one detection per frame (small bbox, real mask)."""

    last_predict_count = 0

    def predict(self, frame, prompt):
        type(self).last_predict_count += 1
        return _one_detection(frame)

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


class _RaisingModel:
    def predict(self, frame, prompt):
        raise RuntimeError("simulated model OOM")

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


class _BatchRecordingModel:
    """Records the size of each predict_batch call; one detection per frame."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def predict(self, frame, prompt):
        return self.predict_batch([frame], [prompt])[0]

    def predict_batch(self, frames, prompts):
        self.batch_sizes.append(len(frames))
        return [_one_detection(f) for f in frames]


def _write_video(path: Path, n_frames: int = 5) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _setup(tmp_path):
    """Make a minimal recording_dir + write one video part + per-part timestamps CSV."""
    proj_dir = tmp_path / "proj"
    recording_dir = proj_dir / "recordings" / "recording_001"
    recording_dir.mkdir(parents=True)
    video_path = recording_dir / "video" / "part_001.mp4"
    n_frames = _write_video(video_path, n_frames=8)
    ts_csv = recording_dir / "video" / "timestamps" / "generated" / "part_001.csv"
    ts_csv.parent.mkdir(parents=True, exist_ok=True)
    recording_store.write_timestamps_csv(
        ts_csv, np.arange(n_frames, dtype=np.int64) * 1000
    )
    (recording_dir / "recording.yaml").write_text(
        "schema_version: 1\n" "id: recording_001\n" "name: recording_001\n",
        encoding="utf-8",
    )
    data = {
        "id": "recording_001",
        "video": {
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
            ]
        },
    }
    return recording_dir, data, n_frames


def _prompt():
    return Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))])


def _part_dict():
    return {"index": 0, "file": "video/part_001.mp4"}


# --- happy path ------------------------------------------------------------ #
def test_run_part_writes_zarr_and_returns_status(tmp_path):
    recording_dir, data, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    out_zarr = run_dir / "part_001.zarr"
    progress_json = run_dir / "parts" / "part_001.json"

    result = _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=out_zarr,
        progress_json_path=progress_json,
        stop_flag_path=run_dir / "stop.flag",
        model=_FakeModel(),
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=0,
    )

    assert result["id"] == "part_001"
    assert result["status"] == "complete"
    assert result["frames_done"] == n_frames
    assert result["frames_total"] == n_frames
    assert result["gpu"] == 0
    assert result["mask_resolution"] == [H, W]
    assert result["error"] is None
    assert result["dropped_frames"] == []

    # zarr is readable and has n_frames rows (one detection per frame)
    rows = PartZarrReader(out_zarr).read()
    assert rows["frame_idx"].shape[0] == n_frames
    assert rows["t_ns"].tolist() == [i * 1000 for i in range(n_frames)]


def _run(recording_dir, data, run_dir, part=None):
    return _detection_worker.run_part(
        part=part or _part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=run_dir / "part_001.zarr",
        progress_json_path=None,
        stop_flag_path=None,
        model=_FakeModel(),
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=0,
    )


def test_frames_total_and_dropped_come_from_timestamp_map(tmp_path, monkeypatch):
    # frames_total / dropped_frames derive from the authoritative timestamp map
    # (Tenet 8), not cv2's CAP_PROP_FRAME_COUNT. Here the file's packets agree with
    # the map but cv2 stops early — a damaged tail — so the worker reports the map
    # length and the genuinely-missing tail as dropped.
    recording_dir, data, n_frames = _setup(tmp_path)
    ts_csv = recording_dir / "video" / "timestamps" / "generated" / "part_001.csv"
    expected_total = n_frames + 5
    recording_store.write_timestamps_csv(
        ts_csv, np.arange(expected_total, dtype=np.int64) * 1000
    )
    monkeypatch.setattr(video_store, "shown_frame_count", lambda _p: expected_total)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    result = _run(recording_dir, data, run_dir)

    assert result["frames_total"] == expected_total
    assert result["frames_done"] == n_frames
    assert result["dropped_frames"] == list(range(n_frames, expected_total))


# --- the map and cv2 disagree on the frame count --------------------------- #
# Class: a sequential pass that pairs cv2's i-th frame with map[i] while the two
# count differently. Every row lands at the wrong time; the old handling wrote
# t_ns = -1 for a surplus and called a shortfall "dropped at the end" — on an
# edit-list file the missing frames are at the *start*, so the whole part shifted.
def test_a_map_longer_than_the_video_is_an_error_not_a_dropped_tail(tmp_path):
    recording_dir, data, n_frames = _setup(tmp_path)
    ts_csv = recording_dir / "video" / "timestamps" / "generated" / "part_001.csv"
    recording_store.write_timestamps_csv(
        ts_csv, np.arange(n_frames + 5, dtype=np.int64) * 1000
    )
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    with pytest.raises(video_store.FrameCountMismatchError) as err:
        _run(recording_dir, data, run_dir)
    message = str(err.value)
    assert "part_001.mp4" in message
    assert f"has {n_frames + 5} frames" in message
    assert f"decodes to {n_frames}" in message
    assert "Re-import" in message


def test_a_map_shorter_than_the_video_is_an_error_not_t_ns_minus_one(tmp_path):
    recording_dir, data, n_frames = _setup(tmp_path)
    ts_csv = recording_dir / "video" / "timestamps" / "generated" / "part_001.csv"
    recording_store.write_timestamps_csv(
        ts_csv, np.arange(n_frames - 2, dtype=np.int64) * 1000
    )
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    with pytest.raises(video_store.FrameCountMismatchError, match=r"more than 6"):
        _run(recording_dir, data, run_dir)


def test_an_edit_list_trimmed_part_is_placed_by_the_frames_it_shows(
    tmp_path, write_vfr_video
):
    """The real container case, through the importer's own map: rows line up with
    the map from frame 0 and nothing is dropped. A map that still counts the trimmed
    frames (what the importer wrote before) is refused rather than shifted."""
    recording_dir, data, _ = _setup(tmp_path)
    video_path = recording_dir / "video" / "part_001.mp4"
    write_vfr_video(video_path, start_ms=-50, b_frames=2)
    ts_csv = recording_dir / "video" / "timestamps" / "generated" / "part_001.csv"
    t_ns = video_store.generate_timestamps(video_path, 50.0)
    assert len(t_ns) == 55
    recording_store.write_timestamps_csv(ts_csv, t_ns)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    result = _run(recording_dir, data, run_dir)

    assert result["status"] == "complete"
    assert result["dropped_frames"] == []
    rows = PartZarrReader(run_dir / "part_001.zarr").read()
    assert rows["t_ns"].tolist() == t_ns.tolist()

    stale = np.concatenate([np.arange(-5, 0, dtype=np.int64) * 10_000_000, t_ns])
    recording_store.write_timestamps_csv(ts_csv, stale)  # counted the trimmed 5
    rerun = recording_dir / "detections" / "det_02"
    rerun.mkdir(parents=True)
    with pytest.raises(video_store.FrameCountMismatchError, match="packets say 55"):
        _run(recording_dir, data, rerun)


def test_run_part_batches_frames(tmp_path):
    # batch_size>1: frames are consumed in bounded batches (not one predict per
    # frame), and the output is identical — same row count + t_ns.
    recording_dir, data, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    out_zarr = run_dir / "part_001.zarr"
    model = _BatchRecordingModel()

    result = _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=out_zarr,
        progress_json_path=None,
        stop_flag_path=None,
        model=model,
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=0,
        batch_size=4,
    )

    assert result["status"] == "complete"
    assert result["frames_done"] == n_frames
    rows = PartZarrReader(out_zarr).read()
    assert rows["frame_idx"].shape[0] == n_frames  # one detection per frame, unchanged
    assert rows["t_ns"].tolist() == [i * 1000 for i in range(n_frames)]
    # genuinely batched: every frame accounted for, in multi-frame calls, fewer
    # calls than frames, none exceeding batch_size
    assert sum(model.batch_sizes) == n_frames
    assert max(model.batch_sizes) > 1
    assert len(model.batch_sizes) < n_frames
    assert all(b <= 4 for b in model.batch_sizes)


def test_run_part_writes_final_progress_json(tmp_path):
    recording_dir, data, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    progress_json = run_dir / "parts" / "part_001.json"

    _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=run_dir / "part_001.zarr",
        progress_json_path=progress_json,
        stop_flag_path=run_dir / "stop.flag",
        model=_FakeModel(),
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=None,
    )
    snapshot = json.loads(progress_json.read_text(encoding="utf-8-sig"))
    assert snapshot["status"] == "complete"
    assert snapshot["frames_done"] == n_frames
    assert snapshot["frames_total"] == n_frames


def test_run_part_works_without_progress_json(tmp_path):
    """progress_json_path=None should just skip the snapshot writes."""
    recording_dir, data, n_frames = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    result = _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=run_dir / "part_001.zarr",
        progress_json_path=None,
        stop_flag_path=None,
        model=_FakeModel(),
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=None,
    )
    assert result["status"] == "complete"


def test_run_part_low_conf_filtered(tmp_path):
    recording_dir, data, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=run_dir / "part_001.zarr",
        progress_json_path=None,
        stop_flag_path=None,
        model=_FakeModel(),
        prompt=_prompt(),
        conf_threshold=0.95,  # above the fake's 0.9
        gpu_idx=None,
    )
    rows = PartZarrReader(run_dir / "part_001.zarr").read()
    assert rows["frame_idx"].shape[0] == 0


# --- stop.flag ------------------------------------------------------------- #
def test_run_part_stops_when_stop_flag_set_before_invocation(tmp_path):
    """If stop.flag exists before the first forward pass, the worker exits with
    status=failed before processing the part."""
    recording_dir, data, _ = _setup(tmp_path)
    _write_video(recording_dir / "video" / "part_001.mp4", n_frames=250)
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(250, dtype=np.int64) * 1000,
    )
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    stop_flag = run_dir / "stop.flag"
    stop_flag.touch()

    result = _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=run_dir / "part_001.zarr",
        progress_json_path=run_dir / "parts" / "part_001.json",
        stop_flag_path=stop_flag,
        model=_FakeModel(),
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=None,
    )
    assert result["status"] == "failed"
    assert "stop.flag" in result["error"]
    assert result["frames_done"] < 250  # didn't process all frames


# --- model exception is *not* caught by run_part directly ----------------- #
def test_run_part_propagates_model_exception(tmp_path):
    """run_part itself doesn't swallow exceptions — that's pool_run_part's job."""
    recording_dir, data, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="simulated model OOM"):
        _detection_worker.run_part(
            part=_part_dict(),
            part_index=0,
            parts_total=1,
            recording_dir=recording_dir,
            recording_data=data,
            out_zarr_path=run_dir / "part_001.zarr",
            progress_json_path=None,
            stop_flag_path=None,
            model=_RaisingModel(),
            prompt=_prompt(),
            conf_threshold=0.5,
            gpu_idx=None,
        )


def test_pool_run_part_captures_model_exception(tmp_path, monkeypatch):
    """pool_run_part catches anything raised by run_part and reports it via
    status=failed + error string, so one bad part doesn't take down the pool."""
    recording_dir, data, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)

    # stash the model + prompt where pool_run_part looks
    monkeypatch.setitem(_detection_worker._WORKER_STATE, "model", _RaisingModel())
    monkeypatch.setitem(_detection_worker._WORKER_STATE, "prompt", _prompt())
    monkeypatch.setitem(_detection_worker._WORKER_STATE, "conf_threshold", 0.5)
    monkeypatch.setitem(_detection_worker._WORKER_STATE, "gpu_idx", 0)

    result = _detection_worker.pool_run_part(
        {
            "part": _part_dict(),
            "part_index": 0,
            "parts_total": 1,
            "recording_dir": str(recording_dir),
            "recording_data": data,
            "out_zarr_path": str(run_dir / "part_001.zarr"),
            "progress_json_path": str(run_dir / "parts" / "part_001.json"),
            "stop_flag_path": str(run_dir / "stop.flag"),
        }
    )
    assert result["status"] == "failed"
    assert "simulated model OOM" in result["error"]
    assert result["gpu"] == 0


# --- atomic progress JSON writes ------------------------------------------ #
def test_progress_json_atomic_under_concurrent_reads(tmp_path):
    """Reader sees either the old or the new JSON, never torn."""
    target = tmp_path / "p.json"
    target.write_text(
        json.dumps({"frames_done": 1, "frames_total": 10, "status": "running"}),
        encoding="utf-8",
    )
    # overwrite atomically — reader during this should see one valid JSON or the other
    _detection_worker._write_progress_json(
        target, frames_done=5, frames_total=10, status="running"
    )
    snapshot = _detection_worker.read_progress_json(target)
    assert snapshot in (
        {"frames_done": 1, "frames_total": 10, "status": "running"},
        {"frames_done": 5, "frames_total": 10, "status": "running"},
    )


def test_read_progress_json_missing_returns_none(tmp_path):
    assert _detection_worker.read_progress_json(tmp_path / "nope.json") is None


# --- pool_initializer device format regression --------------------------- #
def test_pool_initializer_loads_with_ultralytics_index_format(monkeypatch):
    """Regression: pool_initializer must pass ``"0"`` (Ultralytics-style index)
    to model.load, not ``"cuda:0"`` — backends like SAM3 prefix ``"cuda:"``
    themselves and would otherwise build ``"cuda:cuda:0"``.
    """
    from detail.core.dataclasses import Prompt, PromptTarget, TextCue
    from detail.core.protocols import DetectionModel
    from detail.core.registries import MODELS, register_model

    received_devices: list[str] = []

    class _RecordingModel(DetectionModel):
        model_id = "pool_init_probe"
        supported_cue_types = [TextCue]
        supported_backends = ["cuda"]

        def load(self, device):
            received_devices.append(device)

        def set_conf_threshold(self, t):
            pass

        def predict(self, frame, prompt):
            raise NotImplementedError

        def predict_batch(self, frames, prompts):
            raise NotImplementedError

    register_model("pool_init_probe")(_RecordingModel)
    try:
        _detection_worker.pool_initializer(
            gpu_idx=0,
            experiment_root=".",  # unused for a pretrained (registry) model
            model_id="pool_init_probe",
            weights_source="pretrained",
            model_variant=None,
            prompt_dict=Prompt(targets=[PromptTarget("x", TextCue("x"))]).to_dict(),
            conf_threshold=0.5,
            batch_size=1,
            pipeline_runner_modules=(),
        )
    finally:
        MODELS.pop("pool_init_probe", None)

    # Critical assertion: the device string must NOT contain "cuda:" — that's
    # the backend's job to add.
    assert received_devices == ["0"]
    assert "cuda:" not in received_devices[0]


# --- batching edge coverage (#28) ------------------------------------------ #
class _EvenEmptyModel:
    """Zero detections on even frame indices, one on odd — to test a batch that spans the
    empty/non-empty boundary. Frames are processed in order, so the counter tracks frame_idx.
    """

    def __init__(self):
        self._i = 0

    def predict(self, frame, prompt):
        return self.predict_batch([frame], [prompt])[0]

    def predict_batch(self, frames, prompts):
        out = []
        for f in frames:
            out.append(
                _one_detection(f)
                if self._i % 2
                else FrameResult.from_detections(0, 0, [])
            )
            self._i += 1
        return out


class _OOMModel:
    def predict(self, frame, prompt):
        return self.predict_batch([frame], [prompt])[0]

    def predict_batch(self, frames, prompts):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _run_part(recording_dir, data, out_zarr, model, batch_size, stop_flag=None):
    return _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=out_zarr,
        progress_json_path=None,
        stop_flag_path=stop_flag,
        model=model,
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=0,
        batch_size=batch_size,
    )


def test_run_part_batch_with_zero_detection_frames(tmp_path):
    # A batch that MIXES zero-detection and detected frames: rows stay paired to the right
    # frame_idx / t_ns (§9 schema edge — zero-detection frames have no rows).
    recording_dir, data, n_frames = _setup(tmp_path)  # 8 frames
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    out_zarr = run_dir / "part_001.zarr"
    result = _run_part(recording_dir, data, out_zarr, _EvenEmptyModel(), batch_size=3)
    assert result["status"] == "complete" and result["frames_done"] == n_frames
    rows = PartZarrReader(out_zarr).read()
    assert rows["frame_idx"].tolist() == [
        1,
        3,
        5,
        7,
    ]  # only odd frames, correctly paired
    assert rows["t_ns"].tolist() == [1000, 3000, 5000, 7000]


def test_run_part_batch_larger_than_video(tmp_path):
    # batch_size > frame count: the whole part is written by the trailing flush only.
    recording_dir, data, n_frames = _setup(tmp_path)  # 8 frames
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    out_zarr = run_dir / "part_001.zarr"
    model = _BatchRecordingModel()
    result = _run_part(recording_dir, data, out_zarr, model, batch_size=100)
    assert result["status"] == "complete" and result["frames_done"] == n_frames
    assert PartZarrReader(out_zarr).read()["frame_idx"].shape[0] == n_frames
    assert model.batch_sizes == [
        n_frames
    ]  # exactly one trailing flush of the whole part


def test_run_part_oom_gets_actionable_hint(tmp_path):
    # A batch OOM is rewrapped with an actionable "lower the batch size" hint (not swallowed).
    recording_dir, data, _ = _setup(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="lower the batch size"):
        _run_part(
            recording_dir, data, run_dir / "part_001.zarr", _OOMModel(), batch_size=8
        )


def test_run_part_stop_mid_batch_frames_done_matches_rows(tmp_path):
    # On a stop with frames buffered mid-batch, frames_done must equal rows written (A13):
    # the buffer is flushed before breaking, so the honest-partial-run guarantee holds.
    recording_dir, data, _ = _setup(tmp_path)
    _write_video(recording_dir / "video" / "part_001.mp4", n_frames=25)
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(25, dtype=np.int64) * 1000,
    )
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    stop_flag = run_dir / "stop.flag"
    stop_flag.touch()
    out_zarr = run_dir / "part_001.zarr"
    result = _run_part(
        recording_dir, data, out_zarr, _FakeModel(), batch_size=3, stop_flag=stop_flag
    )
    assert result["status"] == "failed"
    rows = PartZarrReader(out_zarr).read()
    assert rows["frame_idx"].shape[0] == result["frames_done"]  # no buffered overcount


# --- the four stages (§7 *Execution*) --------------------------------------- #
# Class guarded: **a stage that overlaps its CPU work with the GPU writes rows that differ
# from the serial loop's** (reordered, dropped, or computed from another batch's frames) or
# leaves a thread running after the part. The reference is the same model with its split
# hidden, which runs every half on one thread in order.


def _frame_value(frame) -> int:
    return int(frame[0, 0, 0])


class _SplitModel:
    """Opts into the three-part split; each frame's detection encodes that frame's pixels,
    so a batch handed to the wrong stage call writes the wrong rows."""

    def __init__(self):
        self.threads: dict[str, set] = {"prepare": set(), "run": set(), "finish": set()}
        self.batches: list[int] = []

    def predict_batch(self, frames, prompts):
        return self.finish_batch(self.run_prepared(self.prepare_batch(frames, prompts)))

    def prepare_batch(self, frames, prompts):
        self.threads["prepare"].add(threading.current_thread().name)
        return [_frame_value(f) for f in frames], [f.shape for f in frames]

    def run_prepared(self, prepared):
        self.threads["run"].add(threading.current_thread().name)
        values, shapes = prepared
        self.batches.append(len(values))
        return list(zip(values, shapes))

    def finish_batch(self, raw):
        self.threads["finish"].add(threading.current_thread().name)
        out = []
        for value, (h, w, _) in raw:
            mask = np.zeros((h, w), dtype=bool)
            mask[: 1 + value % h, : 1 + value % w] = True
            frac = value / 255.0
            out.append(
                FrameResult.from_detections(
                    0, 0, [Detection(mask, (0.0, 0.0, frac, frac), "mouse", 0.9, None)]
                )
            )
        return out


class _Whole:
    """The same model with its split hidden: the worker runs ``predict_batch`` whole."""

    def __init__(self, model):
        self._model = model

    def predict_batch(self, frames, prompts):
        return self._model.predict_batch(frames, prompts)


def _long_part(tmp_path, n_frames=25):
    recording_dir, data, _ = _setup(tmp_path)
    n = _write_video(recording_dir / "video" / "part_001.mp4", n_frames=n_frames)
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(n, dtype=np.int64) * 1000,
    )
    return recording_dir, data, n


def _run_stages(recording_dir, data, out_zarr, model, batch_size, stride=1, stop=None):
    return _detection_worker.run_part(
        part=_part_dict(),
        part_index=0,
        parts_total=1,
        recording_dir=recording_dir,
        recording_data=data,
        out_zarr_path=out_zarr,
        progress_json_path=None,
        stop_flag_path=stop,
        model=model,
        prompt=_prompt(),
        conf_threshold=0.5,
        gpu_idx=0,
        batch_size=batch_size,
        every_n_frames=stride,
    )


def _stage_threads():
    return sorted(t.name for t in threading.enumerate() if "lookahead" in t.name)


def _serial_rows(video: Path, batch_size: int, stride: int) -> dict:
    """The rows a one-thread loop writes: read the video in order, ``predict_batch`` each
    batch, encode each mask. Built without the worker, so a fault in the worker's stages
    cannot also be in the reference."""
    cap = cv2.VideoCapture(str(video))
    frames, kept = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
            kept.append(i)
        i += 1
    cap.release()
    model = _SplitModel()
    results = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        results += model.predict_batch(batch, [_prompt()] * len(batch))
    rows = {"frame_idx": [], "t_ns": [], "bbox": [], "mask_rle": [], "confidence": []}
    for frame_idx, result in zip(kept, results, strict=True):
        for det in result.detections():
            rows["frame_idx"].append(frame_idx)
            rows["t_ns"].append(frame_idx * 1000)
            rows["bbox"].append(det.bbox)
            rows["mask_rle"].append(_detection_worker._encode_mask(det.mask))
            rows["confidence"].append(det.confidence)
    return rows


def _assert_rows(path: Path, want: dict) -> None:
    got = PartZarrReader(path).read()
    assert got["frame_idx"].tolist() == want["frame_idx"]
    assert got["t_ns"].tolist() == want["t_ns"]
    assert got["mask_rle"].tolist() == want["mask_rle"]
    np.testing.assert_allclose(got["bbox"], np.asarray(want["bbox"]), rtol=0, atol=0)
    np.testing.assert_allclose(got["confidence"], want["confidence"], rtol=0, atol=0)


@pytest.mark.parametrize("batch_size,stride", [(1, 1), (3, 1), (4, 2), (100, 1)])
def test_rows_are_the_serial_rows_and_each_half_has_its_thread(
    tmp_path, batch_size, stride
):
    recording_dir, data, n = _long_part(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    want_rows = _serial_rows(
        recording_dir / "video" / "part_001.mp4", batch_size, stride
    )
    split = _SplitModel()
    got = _run_stages(
        recording_dir, data, run_dir / "split.zarr", split, batch_size, stride
    )
    whole = _run_stages(
        recording_dir,
        data,
        run_dir / "whole.zarr",
        _Whole(_SplitModel()),
        batch_size,
        stride,
    )

    assert got == whole and got["status"] == "complete" and got["frames_done"] == n
    _assert_rows(run_dir / "split.zarr", want_rows)
    _assert_rows(run_dir / "whole.zarr", want_rows)  # a model that does not split
    assert want_rows["frame_idx"] == list(range(0, n, stride))
    assert sum(split.batches) == len(range(0, n, stride))
    assert max(split.batches) <= batch_size
    assert split.threads["run"] == {threading.current_thread().name}
    assert split.threads["prepare"] == {"lookahead-producer"}
    assert len(split.threads["finish"]) == 1
    assert next(iter(split.threads["finish"])).startswith("lookahead-worker")
    assert _stage_threads() == []


def test_a_stop_writes_the_computed_batches_and_counts_only_them(tmp_path):
    recording_dir, data, _ = _long_part(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    stop_flag = run_dir / "stop.flag"

    class _StopAfterSecond(_SplitModel):
        def run_prepared(self, prepared):
            out = super().run_prepared(prepared)
            if len(self.batches) == 2:
                stop_flag.touch()
            return out

    model = _StopAfterSecond()
    result = _run_stages(
        recording_dir, data, run_dir / "part_001.zarr", model, 3, stop=stop_flag
    )
    assert result["status"] == "failed" and "stop.flag" in result["error"]
    assert model.batches == [3, 3]  # nothing past the batch that saw the flag
    rows = PartZarrReader(run_dir / "part_001.zarr").read()
    assert rows["frame_idx"].tolist() == [0, 1, 2, 3, 4, 5]
    assert result["frames_done"] == 6
    assert result["dropped_frames"] == []
    assert _stage_threads() == []


def test_a_write_error_fails_the_part_after_the_batches_before_it(
    tmp_path, monkeypatch
):
    recording_dir, data, _ = _long_part(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    encode = _detection_worker._encode_mask
    calls = []

    def _encode_then_fail(mask):
        calls.append(1)
        if len(calls) == 7:  # the first row of the third batch of 3
            raise OSError("disk full")
        return encode(mask)

    monkeypatch.setattr(_detection_worker, "_encode_mask", _encode_then_fail)
    with pytest.raises(OSError, match="disk full"):
        _run_stages(recording_dir, data, run_dir / "part_001.zarr", _SplitModel(), 3)
    rows = PartZarrReader(run_dir / "part_001.zarr").read()
    assert rows["frame_idx"].tolist() == [0, 1, 2, 3, 4, 5]
    assert _stage_threads() == []


class _FailingSplit(_SplitModel):
    def __init__(self, half):
        super().__init__()
        self._half = half

    def prepare_batch(self, frames, prompts):
        if self._half == "prepare" and self.batches:
            raise ValueError("prepare failed")
        return super().prepare_batch(frames, prompts)

    def run_prepared(self, prepared):
        if self._half == "run" and len(self.batches) == 1:
            raise RuntimeError("forward failed")
        return super().run_prepared(prepared)


@pytest.mark.parametrize(
    "half,match", [("prepare", "prepare failed"), ("run", "forward failed")]
)
def test_no_stage_thread_outlives_a_failed_part(tmp_path, half, match):
    recording_dir, data, _ = _long_part(tmp_path)
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    with pytest.raises((ValueError, RuntimeError), match=match):
        _run_stages(
            recording_dir, data, run_dir / "part_001.zarr", _FailingSplit(half), 3
        )
    assert _stage_threads() == []


def test_a_decode_error_reaches_the_caller_after_the_batches_before_it(tmp_path):
    # The map is shorter than the video: the decode thread raises the mismatch at the
    # frame past the map's end; every whole batch before it is on disk, and the error is
    # the caller's.
    recording_dir, data, _ = _long_part(tmp_path)
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(10, dtype=np.int64) * 1000,
    )
    run_dir = recording_dir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    with pytest.raises(video_store.FrameCountMismatchError):
        _run_stages(recording_dir, data, run_dir / "part_001.zarr", _SplitModel(), 3)
    rows = PartZarrReader(run_dir / "part_001.zarr").read()
    assert rows["frame_idx"].tolist() == list(range(9))
    assert _stage_threads() == []
