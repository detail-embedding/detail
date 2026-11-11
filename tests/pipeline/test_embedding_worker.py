"""Unit tests for pipeline/_embedding_worker.

Calls ``embed_one`` directly (no subprocess) with fake models so we can
exercise the per-batch loop, progress JSON writes, stop.flag polling, error
capture, and the shape of the returned status dict. ``pool_initializer`` and
``pool_run_one`` are tested via patches — the real subprocess path is covered
by the integration tests in tests/pipeline/test_embedding.py.
"""

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from pycocotools import mask as mask_utils

from detail.core.protocols import EmbeddingModel
from detail.core.registries import EMBEDDING_MODELS, register_embedding_model
from detail.pipeline import _embedding_worker
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartWriter,
    PartZarrReader,
)
from detail.transforms import basic as _basic  # noqa: F401 — registers transforms
from detail.transforms.renderer import MaskRenderer

H, W, D = 16, 20, 8
RENDERER_SPEC = [
    {"type": "init_binary_mask"},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 32},
]


class _FakeEmbedder(EmbeddingModel):
    """Embeds each image as a (D,) vector of its mean pixel value."""

    model_id = "worker_fake_embedder"
    output_dim = D
    supported_backends = ["cpu"]

    def __init__(self, variant=None):
        super().__init__(variant)
        self.batch_calls = []

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def embed(self, image):
        return np.full(D, float(np.asarray(image).mean()), dtype=np.float32)

    def embed_batch(self, images):
        items = [np.asarray(img) for img in images]
        self.batch_calls.append(len(items))
        return np.stack(
            [np.full(D, float(im.mean()), dtype=np.float32) for im in items], axis=0
        )


class _RaisingEmbedder(_FakeEmbedder):
    model_id = "raising_embedder"

    def embed_batch(self, images):
        raise RuntimeError("simulated embed_batch OOM")


def _write_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((32, 32, 3), (i * 7) % 250 + 1, dtype=np.uint8))
    vw.release()


def _mask_counts(f: int) -> str:
    m = np.zeros((H, W), np.uint8)
    top = f % (H - 4)
    m[top : top + 4, 2:8] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _setup(tmp_path: Path, object_ids: list[int]) -> dict:
    """Build a minimal det zarr + assignment zarr + video for one part."""
    video = tmp_path / "v.mp4"
    _write_video(video, n_frames=len(object_ids))

    det_path = tmp_path / "det" / "part_001.zarr"
    det_path.parent.mkdir(parents=True)
    with DetectionPartWriter(det_path) as w:
        for f in range(len(object_ids)):
            w.add_row(
                f, f * 1000, 0, "mouse", 0.9, (0.1, 0.2, 0.3, 0.4), _mask_counts(f)
            )

    assign_path = tmp_path / "track" / "assignments" / "part_001.zarr"
    assign_path.parent.mkdir(parents=True)
    with AssignmentPartWriter(assign_path) as w:
        for oid in object_ids:
            w.add_object_id(oid)
    return {
        "video": video,
        "det_zarr": det_path,
        "assignment_zarr": assign_path,
    }


def _embed_one(tmp_path, *, model, fixtures, **overrides):
    """Default kwargs + per-test overrides."""
    progress = tmp_path / "progress.json"
    stop_flag = tmp_path / "stop.flag"
    out_zarr = tmp_path / "out" / "track_0" / "part_001.zarr"
    renderer = MaskRenderer.from_spec(RENDERER_SPEC)
    kwargs = dict(
        track_id=0,
        part_index=0,
        part_id="part_001",
        video_path=fixtures["video"],
        det_zarr_path=fixtures["det_zarr"],
        assignment_zarr_path=fixtures["assignment_zarr"],
        mask_size=(H, W),
        out_zarr_path=out_zarr,
        progress_json_path=progress,
        stop_flag_path=stop_flag,
        model=model,
        renderer=renderer,
        output_dim=D,
        batch_size=2,
        gpu_idx=None,
    )
    kwargs.update(overrides)
    return _embedding_worker.embed_one(**kwargs), progress, stop_flag, out_zarr


# --------------------------------------------------------------------------- #
# embed_one
# --------------------------------------------------------------------------- #
def test_embed_one_happy_path(tmp_path):
    """Three rows, batch_size=2 → one batch of 2 + one of 1, all written."""
    fixtures = _setup(tmp_path, object_ids=[0, 0, 0])
    model = _FakeEmbedder()
    result, progress_path, _, out_zarr = _embed_one(
        tmp_path, model=model, fixtures=fixtures
    )

    assert result["status"] == "complete"
    assert result["track_id"] == 0
    assert result["part_id"] == "part_001"
    assert result["frames_done"] == 3
    assert result["frames_total"] == 3
    assert result["gpu"] is None
    assert result["output_dim"] == D
    assert result["error"] is None

    emb = PartZarrReader(out_zarr).read()
    assert emb["embedding"].shape == (3, D)
    assert emb["det_row"].tolist() == [0, 1, 2]
    assert emb["frame_idx"].tolist() == [0, 1, 2]
    assert model.batch_calls == [2, 1]

    # Final progress JSON reflects the completion status.
    snap = json.loads(progress_path.read_text(encoding="utf-8-sig"))
    assert snap["status"] == "complete"
    assert snap["frames_done"] == 3
    assert snap["frames_total"] == 3
    assert snap["track_id"] == 0
    assert snap["part_id"] == "part_001"


def test_embed_one_unions_frame_bboxes(tmp_path):
    """Two detections on one frame → their union bbox reaches the renderer, so the
    centered constructors crop the subject and not the frame center (§14)."""
    video = tmp_path / "v.mp4"
    _write_video(video, n_frames=1)
    det_path = tmp_path / "det" / "part_001.zarr"
    det_path.parent.mkdir(parents=True)
    with DetectionPartWriter(det_path) as w:
        w.add_row(0, 0, 0, "m", 0.9, (0.1, 0.2, 0.5, 0.6), _mask_counts(0))
        w.add_row(0, 0, 1, "m", 0.8, (0.3, 0.1, 0.7, 0.4), _mask_counts(0))
    assign_path = tmp_path / "track" / "assignments" / "part_001.zarr"
    assign_path.parent.mkdir(parents=True)
    with AssignmentPartWriter(assign_path) as w:
        w.add_object_id(0)
        w.add_object_id(0)

    seen: list = []

    class _SpyRenderer:
        def render(self, ctx):
            seen.append(ctx.bbox)
            return np.zeros((4, 4, 3), np.uint8)

    _embedding_worker.embed_one(
        track_id=0,
        part_index=0,
        part_id="p1",
        video_path=video,
        det_zarr_path=det_path,
        assignment_zarr_path=assign_path,
        mask_size=(H, W),
        out_zarr_path=tmp_path / "out" / "e.zarr",
        progress_json_path=None,
        stop_flag_path=None,
        model=_FakeEmbedder(),
        renderer=_SpyRenderer(),
        output_dim=D,
        batch_size=2,
        gpu_idx=None,
    )
    # union = (min .1,.3 ; min .2,.1 ; max .5,.7 ; max .6,.4)
    assert len(seen) == 1
    assert seen[0] == pytest.approx((0.1, 0.1, 0.7, 0.6), abs=1e-6)


def test_embed_one_mask_only_renders_off_thread(tmp_path):
    """A mask-only (needs_frame False) renderer renders rows on a thread pool ahead
    of the GPU consumer; output stays in frame order (§14)."""
    import threading

    fixtures = _setup(tmp_path, object_ids=[0, 0, 0, 0])
    main = threading.get_ident()
    threads: list = []

    class _ThreadSpyRenderer:
        needs_frame = False

        def render(self, ctx):
            threads.append(threading.get_ident())
            return np.zeros((4, 4, 3), np.uint8)

    result, _, _, out_zarr = _embed_one(
        tmp_path,
        model=_FakeEmbedder(),
        fixtures=fixtures,
        renderer=_ThreadSpyRenderer(),
    )
    assert result["status"] == "complete"
    assert result["frames_done"] == 4
    emb = PartZarrReader(out_zarr).read()
    assert emb["frame_idx"].tolist() == [0, 1, 2, 3]  # order preserved
    # rendered on worker threads, not the calling thread
    assert threads and all(t != main for t in threads)


def test_embed_one_filters_to_track_id(tmp_path):
    """object_ids=[0, 1, 0, 2, 0] → track 0 keeps rows 0,2,4 only."""
    fixtures = _setup(tmp_path, object_ids=[0, 1, 0, 2, 0])
    model = _FakeEmbedder()
    result, _, _, out_zarr = _embed_one(tmp_path, model=model, fixtures=fixtures)

    assert result["status"] == "complete"
    assert result["frames_done"] == 3
    emb = PartZarrReader(out_zarr).read()
    assert emb["det_row"].tolist() == [0, 2, 4]
    assert emb["frame_idx"].tolist() == [0, 2, 4]


def test_embed_one_empty_track_writes_empty_zarr(tmp_path):
    """No rows belong to this track — no embed_batch call, empty zarr, complete."""
    fixtures = _setup(tmp_path, object_ids=[-1, -1, -1])
    model = _FakeEmbedder()
    result, _, _, out_zarr = _embed_one(tmp_path, model=model, fixtures=fixtures)

    assert result["status"] == "complete"
    assert result["frames_done"] == 0
    assert result["frames_total"] == 0
    assert model.batch_calls == []
    emb = PartZarrReader(out_zarr).read()
    assert emb["embedding"].shape == (0, D)


def test_embed_one_stop_flag_mid_run(tmp_path):
    """stop.flag set after first batch → second batch isn't computed, status=failed."""
    fixtures = _setup(tmp_path, object_ids=[0, 0, 0, 0])
    stop_flag_path = tmp_path / "stop.flag"

    class _StopAfterFirst(_FakeEmbedder):
        def embed_batch(self, images):
            out = super().embed_batch(images)
            stop_flag_path.touch()
            return out

    model = _StopAfterFirst()
    result, _, _, out_zarr = _embed_one(
        tmp_path,
        model=model,
        fixtures=fixtures,
        stop_flag_path=stop_flag_path,
        batch_size=2,
    )

    assert result["status"] == "failed"
    assert "stop.flag" in (result["error"] or "")
    # First batch (2 rows) made it to disk before the stop check on iteration 2.
    assert result["frames_done"] == 2
    emb = PartZarrReader(out_zarr).read()
    assert emb["embedding"].shape == (2, D)
    assert model.batch_calls == [2]


def test_embed_one_captures_model_exception(tmp_path):
    """An exception inside embed_batch is caught → status=failed, error carries it."""
    fixtures = _setup(tmp_path, object_ids=[0, 0, 0])
    model = _RaisingEmbedder()
    result, _, _, out_zarr = _embed_one(tmp_path, model=model, fixtures=fixtures)

    assert result["status"] == "failed"
    assert result["error"] is not None
    assert "RuntimeError" in result["error"]
    assert "simulated embed_batch OOM" in result["error"]
    # Writer was opened (zarr exists) but nothing got written before the raise.
    emb = PartZarrReader(out_zarr).read()
    assert emb["embedding"].shape == (0, D)


def test_embed_one_progress_json_updates_per_batch(tmp_path):
    """Each batch flushes a progress JSON snapshot — confirms running cadence."""
    fixtures = _setup(tmp_path, object_ids=[0, 0, 0, 0])
    snapshots: list[dict] = []

    real_writer = _embedding_worker._write_progress_json

    def _spy(path, **kwargs):
        real_writer(path, **kwargs)
        snapshots.append(dict(kwargs))

    with patch.object(_embedding_worker, "_write_progress_json", side_effect=_spy):
        _embed_one(tmp_path, model=_FakeEmbedder(), fixtures=fixtures, batch_size=2)

    # 1 initial "running" (before loop) + 2 batch flushes + 1 final terminal
    assert len(snapshots) >= 3
    assert snapshots[0]["frames_done"] == 0
    assert snapshots[-1]["status"] == "complete"
    assert snapshots[-1]["frames_done"] == 4


# --------------------------------------------------------------------------- #
# pool_initializer + pool_run_one
# --------------------------------------------------------------------------- #
def test_pool_initializer_loads_with_ultralytics_index_format(monkeypatch):
    """Regression: ``pool_initializer`` must call ``model.load("0")`` — backends
    prefix ``cuda:`` themselves and would otherwise build ``cuda:cuda:0``
    (the bug we hit in detection)."""
    loads: list[str] = []

    class _SpyEmbedder(_FakeEmbedder):
        model_id = "spy_embedder"

        def load(self, device):
            loads.append(device)

    register_embedding_model("spy_embedder")(_SpyEmbedder)
    try:
        # Reset worker state so the assertions are about this call.
        _embedding_worker._WORKER_STATE.clear()
        _embedding_worker.pool_initializer(
            gpu_idx=7,
            model_id="spy_embedder",
            model_variant=None,
            renderer_spec=RENDERER_SPEC,
            pipeline_runner_modules=(),
        )
    finally:
        EMBEDDING_MODELS.pop("spy_embedder", None)

    assert loads == ["0"]
    assert all("cuda:" not in d for d in loads)
    assert _embedding_worker._WORKER_STATE["gpu_idx"] == 7
    assert _embedding_worker._WORKER_STATE["output_dim"] == D


def test_pool_initializer_sets_cuda_visible_devices(monkeypatch):
    """CUDA_VISIBLE_DEVICES must be set before any torch import in the worker.
    We can't easily assert ordering from outside; we at least confirm the env
    var is set to the requested index."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class _NoopEmbedder(_FakeEmbedder):
        model_id = "noop_embedder"

        def load(self, device):
            pass

    register_embedding_model("noop_embedder")(_NoopEmbedder)
    try:
        _embedding_worker._WORKER_STATE.clear()
        _embedding_worker.pool_initializer(
            gpu_idx=3,
            model_id="noop_embedder",
            model_variant=None,
            renderer_spec=RENDERER_SPEC,
            pipeline_runner_modules=(),
        )
    finally:
        EMBEDDING_MODELS.pop("noop_embedder", None)

    import os

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3"


def test_pool_run_one_catches_initializer_state_missing(tmp_path):
    """pool_run_one with empty WORKER_STATE → fails cleanly (KeyError caught)."""
    fixtures = _setup(tmp_path, object_ids=[0, 0])
    _embedding_worker._WORKER_STATE.clear()
    args = {
        "track_id": 0,
        "part_index": 0,
        "part_id": "part_001",
        "video_path": str(fixtures["video"]),
        "det_zarr_path": str(fixtures["det_zarr"]),
        "assignment_zarr_path": str(fixtures["assignment_zarr"]),
        "mask_size": [H, W],
        "out_zarr_path": str(tmp_path / "out" / "track_0" / "part_001.zarr"),
        "progress_json_path": str(tmp_path / "progress.json"),
        "stop_flag_path": str(tmp_path / "stop.flag"),
        "batch_size": 2,
    }
    result = _embedding_worker.pool_run_one(args)
    assert result["status"] == "failed"
    assert result["error"] is not None
    assert "KeyError" in result["error"]
    assert result["track_id"] == 0
    assert result["part_id"] == "part_001"


def test_pool_run_one_propagates_state_to_embed_one(tmp_path):
    """With WORKER_STATE populated, pool_run_one's result matches a direct embed_one."""
    fixtures = _setup(tmp_path, object_ids=[0, 0, 0])
    model = _FakeEmbedder()
    renderer = MaskRenderer.from_spec(RENDERER_SPEC)

    _embedding_worker._WORKER_STATE.clear()
    _embedding_worker._WORKER_STATE.update(
        {
            "model": model,
            "renderer": renderer,
            "output_dim": D,
            "gpu_idx": 2,
        }
    )
    try:
        args = {
            "track_id": 0,
            "part_index": 0,
            "part_id": "part_001",
            "video_path": str(fixtures["video"]),
            "det_zarr_path": str(fixtures["det_zarr"]),
            "assignment_zarr_path": str(fixtures["assignment_zarr"]),
            "mask_size": [H, W],
            "out_zarr_path": str(tmp_path / "out" / "track_0" / "part_001.zarr"),
            "progress_json_path": str(tmp_path / "progress.json"),
            "stop_flag_path": str(tmp_path / "stop.flag"),
            "batch_size": 2,
        }
        result = _embedding_worker.pool_run_one(args)
    finally:
        _embedding_worker._WORKER_STATE.clear()

    assert result["status"] == "complete"
    assert result["frames_done"] == 3
    assert result["gpu"] == 2
    assert result["output_dim"] == D


# --------------------------------------------------------------------------- #
# frame stride (§4)
# --------------------------------------------------------------------------- #
def test_embed_one_stride_embeds_only_its_own_cadence(tmp_path):
    """A stride thins which rows are embedded and keeps their *source* frame index.

    Renumbering them 0, 1, 2 would break every downstream join — the trajectory is read
    back against the detection rows by frame.
    """
    fixtures = _setup(tmp_path, object_ids=[0] * 8)
    result, _, _, out_zarr = _embed_one(
        tmp_path, model=_FakeEmbedder(), fixtures=fixtures, every_n_frames=4
    )

    assert result["status"] == "complete"
    assert result["frames_done"] == result["frames_total"] == 2  # frames 0 and 4
    rows = PartZarrReader(out_zarr).read()
    assert [int(f) for f in rows["frame_idx"]] == [0, 4]
    assert [int(t) for t in rows["t_ns"]] == [0, 4000]  # each frame's own time


def test_embed_one_default_stride_embeds_every_tracked_frame(tmp_path):
    fixtures = _setup(tmp_path, object_ids=[0] * 5)
    result, _, _, out_zarr = _embed_one(
        tmp_path, model=_FakeEmbedder(), fixtures=fixtures
    )
    rows = PartZarrReader(out_zarr).read()
    assert [int(f) for f in rows["frame_idx"]] == [0, 1, 2, 3, 4]
    assert result["frames_total"] == 5


# --------------------------------------------------------------------------- #
# The pipelined render path is the serial one (§14 *Execution*, #874)
# --------------------------------------------------------------------------- #
# The worker decodes on one thread, renders on a pool and consumes on a third. Nothing about a
# frame may change: publication runs compare embeddings made before and after this path existed.
# The guard renders a textured video through both paths with every renderer family and asserts
# the rows are byte-identical. Scrambling the pool's output order, dropping a frame, or handing
# a row another row's frame all fail it (watched).

IH, IW = 48, 64  # frame and mask share a resolution, as production detection emits


def _write_noise_video(path: Path, n_frames: int) -> None:
    """Per-frame random texture, so a frame handed to the wrong row cannot pass."""
    rng = np.random.default_rng(7)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (IW, IH))
    assert vw.isOpened()
    for _ in range(n_frames):
        vw.write(rng.integers(0, 256, (IH, IW, 3), dtype=np.uint8))
    vw.release()


def _blob_at(f: int, k: int) -> tuple[int, int]:
    return 4 + (f * 3 + k * 11) % (IH - 16), 4 + (f * 5 + k * 17) % (IW - 20)


def _blob_rle(f: int, k: int) -> str:
    m = np.zeros((IH, IW), np.uint8)
    y, x = _blob_at(f, k)
    m[y : y + 10, x : x + 14] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _blobs_bbox(f: int, n: int) -> tuple[float, float, float, float]:
    """The union box of frame ``f``'s blobs, normalized xyxy — what detection would emit, so the
    masked-colour crops hold the subject's real pixels rather than only background."""
    corners = [_blob_at(f, k) for k in range(n)]
    return (
        min(x for _y, x in corners) / IW,
        min(y for y, _x in corners) / IH,
        max(x + 14 for _y, x in corners) / IW,
        max(y + 10 for y, _x in corners) / IH,
    )


def _identity_dataset(video: Path, renderer):
    from detail.pipeline.embedding import DetectionMaskDataset

    # Gaps between rows (the walk grabs past them) and frames with two detections
    # (anchor secondaries, OR-merged): the two shapes the walk and the merge must survive.
    frames = [f for f in range(60) if f % 4 != 1]
    rles = [
        [_blob_rle(f, 0)] + ([_blob_rle(f, 1)] if f % 5 == 0 else []) for f in frames
    ]
    boxes = [_blobs_bbox(f, len(r)) for f, r in zip(frames, rles)]
    return DetectionMaskDataset(
        video_path=video,
        frame_indices=np.asarray(frames),
        t_ns=np.asarray([f * 33_333_333 for f in frames]),
        det_rows=np.arange(len(frames)),
        mask_rles_per_frame=rles,
        bboxes=boxes,
        mask_size=(IH, IW),
        renderer=renderer,
        track_id=0,
    )


def _identity_specs() -> dict[str, list[dict]]:
    """Step lists the render must reproduce, keyed by the recipe that produced them.

    Three are built by a recipe a user can pick. The other four are written out as the step
    lists a saved run records: their recipes were removed from ``CONSTRUCTORS``, the steps
    were not, and a run made with one is still rendered from them.
    """
    from detail.transforms import constructors as C

    resize = {"type": "resize", "size": 32}
    return {
        "full_frame": C.full_frame(resize=32),
        "masked_color_centered": C.masked_color_centered(canvas_px=40, resize=32),
        "masked_color_centered_noise": [
            {"type": "crop_to_bbox"},
            {"type": "apply_mask", "background": 0},
            {"type": "center_in_square", "size": 40},
            resize,
            {"type": "fill_background_noise", "seed": 1234},
        ],
        "context_crop": [
            {"type": "init_frame"},
            {"type": "crop_to_bbox", "padding": 1.0},
            {"type": "pad_to_square"},
            resize,
        ],
        "bbox_crop_color": [
            {"type": "init_frame"},
            {"type": "crop_to_bbox"},
            {"type": "center_in_square", "size": 40},
            resize,
        ],
        "silhouette": [
            {"type": "init_binary_mask"},
            {"type": "pad_to_square"},
            resize,
        ],
        "silhouette_centered": C.silhouette_centered(canvas_px=40, resize=32),
    }


@pytest.mark.parametrize("batch_size", [1, 7, 32])
@pytest.mark.parametrize("constructor", sorted(_identity_specs()))
def test_pipelined_rows_are_the_serial_rows(
    tmp_path, monkeypatch, constructor, batch_size
):
    video = tmp_path / "v.mp4"
    _write_noise_video(video, 60)
    spec = _identity_specs()[constructor]
    serial_ds = _identity_dataset(video, MaskRenderer.from_spec(spec))
    serial = list(serial_ds)

    monkeypatch.setattr(_embedding_worker, "_render_workers", lambda: 6)
    piped_ds = _identity_dataset(video, MaskRenderer.from_spec(spec))
    rows = []
    for images, f_idx, t_ns, det_rows in _embedding_worker._render_batches(
        piped_ds, batch_size
    ):
        assert len(images) <= batch_size
        rows.extend(zip(images, f_idx, t_ns, det_rows))

    assert len(rows) == len(serial) == 45
    for want, (image, f, t, r) in zip(serial, rows):
        assert (f, t, r) == (want["frame_idx"], want["t_ns"], want["det_row"])
        assert image.dtype == want["image"].dtype
        np.testing.assert_array_equal(image, want["image"])
    # What kinematics is written from. A FrameGeometry carries a bbox array, so compare fields.
    assert sorted(piped_ds.geometry) == sorted(serial_ds.geometry)
    for frame, want in serial_ds.geometry.items():
        got = piped_ds.geometry[frame]
        assert (got.cx_norm, got.cy_norm, got.mask_area_norm) == (
            want.cx_norm,
            want.cy_norm,
            want.mask_area_norm,
        )
        np.testing.assert_array_equal(got.bbox_norm, want.bbox_norm)


def test_the_walk_grabs_past_gap_frames_and_reads_only_kept_ones(tmp_path, monkeypatch):
    """A skipped frame is decoded (H.264 needs it) but never converted: grab, not read."""
    from detail.pipeline import embedding as embedding_module

    video = tmp_path / "v.mp4"
    _write_noise_video(video, 60)
    calls = {"grab": 0, "read": 0}
    real = cv2.VideoCapture

    class _Counting:
        def __init__(self, *a):
            self._cap = real(*a)

        def grab(self):
            calls["grab"] += 1
            return self._cap.grab()

        def read(self):
            calls["read"] += 1
            return self._cap.read()

        def release(self):
            self._cap.release()

    monkeypatch.setattr(embedding_module.cv2, "VideoCapture", _Counting)
    dataset = _identity_dataset(
        video, MaskRenderer.from_spec(_identity_specs()["full_frame"])
    )
    kept = len(dataset)
    list(_embedding_worker._render_batches(dataset, 8))
    last = int(dataset.frame_indices[-1])
    assert calls["read"] == kept
    assert calls["grab"] == last + 1 - kept


def test_a_video_that_ends_early_still_fails_the_unit(tmp_path):
    """The walk's error crosses the decode thread and is raised where the serial loop raised
    it — after the rows before it — rather than lost on a background thread."""
    from detail.pipeline.embedding import DetectionMaskDataset

    video = tmp_path / "v.mp4"
    _write_noise_video(video, 10)
    dataset = DetectionMaskDataset(
        video_path=video,
        frame_indices=np.asarray([2, 5, 40]),
        t_ns=np.asarray([2, 5, 40]),
        det_rows=np.asarray([0, 1, 2]),
        mask_rles_per_frame=[[_blob_rle(f, 0)] for f in (2, 5, 40)],
        mask_size=(IH, IW),
        renderer=MaskRenderer.from_spec(_identity_specs()["full_frame"]),
        track_id=0,
    )
    seen = []
    with pytest.raises(RuntimeError, match="ended at frame"):
        for _images, f_idx, _t, _r in _embedding_worker._render_batches(dataset, 1):
            seen.extend(f_idx)
    assert seen == [2, 5]


def test_the_render_pool_shares_cores_with_its_sibling_gpu_workers(monkeypatch):
    from detail.core import cpu

    monkeypatch.setattr(cpu, "usable_cores", lambda: 16)
    monkeypatch.setitem(_embedding_worker._WORKER_STATE, "share", 3)
    assert _embedding_worker._render_workers() == cpu.thread_workers(share=3) == 4


# --------------------------------------------------------------------------- #
# Prep, inference and write each on their own thread (§14 *Execution*)
# --------------------------------------------------------------------------- #
# The model's CPU half (``prepare_batch``) runs a batch ahead on the prep thread, the inference
# thread runs only ``embed_prepared``, and one writer thread puts the rows on disk. What reaches
# the zarr must be what a serial ``embed_batch`` over the same batches would write. Watched
# failing: a writer pool of two threads with jitter (rows reordered), a batch dropped between
# inference and write, and the prep stage handed the next batch's images.

_FRAME_SPEC = [
    {"type": "init_frame"},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 8},
]


class _SplitEmbedder(_FakeEmbedder):
    """Opts into the split; each image's vector is its own pixels, so a row written from the
    wrong frame, or out of order, cannot match."""

    model_id = "split_embedder"

    def __init__(self, variant=None):
        super().__init__(variant)
        self.threads: dict[str, set] = {"prepare": set(), "embed": set()}

    def prepare_batch(self, images):
        import threading

        self.threads["prepare"].add(threading.current_thread().name)
        return np.stack([np.asarray(im, dtype=np.float32) for im in images])

    def embed_prepared(self, prepared):
        import threading

        self.threads["embed"].add(threading.current_thread().name)
        self.batch_calls.append(len(prepared))
        return prepared.reshape(len(prepared), -1)[:, :D] + np.arange(D)

    def embed_batch(self, images):
        return self.embed_prepared(self.prepare_batch(images))


def _lookahead_threads():
    import threading

    return [t for t in threading.enumerate() if t.name.startswith("lookahead")]


def _wait_for_threads(before, timeout=2.0):
    import time

    deadline = time.monotonic() + timeout
    while _lookahead_threads() != before and time.monotonic() < deadline:
        time.sleep(0.01)
    return _lookahead_threads()


@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_rows_are_what_a_serial_embed_batch_writes_and_each_stage_has_its_thread(
    tmp_path, monkeypatch, batch_size
):
    import threading

    from detail.pipeline.embedding import DetectionMaskDataset

    fixtures = _setup(tmp_path, object_ids=[0, -1, 0, 0, 0, -1, 0, 0, 0, 0])
    write_threads: set = set()
    real_progress = _embedding_worker._write_progress_json

    def _spy(path, **kwargs):
        if kwargs.get("status") == "running" and kwargs.get("frames_done", 0) > 0:
            write_threads.add(threading.current_thread().name)
        return real_progress(path, **kwargs)

    monkeypatch.setattr(_embedding_worker, "_write_progress_json", _spy)
    model = _SplitEmbedder()
    result, _, _, out_zarr = _embed_one(
        tmp_path,
        model=model,
        fixtures=fixtures,
        renderer=MaskRenderer.from_spec(_FRAME_SPEC),
        batch_size=batch_size,
    )
    assert result["status"] == "complete", result["error"]

    # The serial reference: the same rendered rows, batched the same way, through embed_batch.
    det = _embedding_worker.DetectionPartReader(fixtures["det_zarr"]).read()
    kept = [0, 2, 3, 4, 6, 7, 8, 9]
    serial = list(
        DetectionMaskDataset(
            video_path=fixtures["video"],
            frame_indices=np.asarray(kept),
            t_ns=np.asarray([int(det["t_ns"][f]) for f in kept]),
            det_rows=np.asarray(kept),
            mask_rles_per_frame=[[str(det["mask_rle"][f])] for f in kept],
            bboxes=[tuple(float(v) for v in det["bbox"][f]) for f in kept],
            mask_size=(H, W),
            renderer=MaskRenderer.from_spec(_FRAME_SPEC),
            track_id=0,
        )
    )
    reference = _SplitEmbedder()
    want = np.concatenate(
        [
            reference.embed_batch([r["image"] for r in serial[i : i + batch_size]])
            for i in range(0, len(serial), batch_size)
        ]
    ).astype(np.float32)

    emb = PartZarrReader(out_zarr).read()
    assert emb["frame_idx"].tolist() == kept
    np.testing.assert_array_equal(np.asarray(emb["embedding"]), want)
    assert result["frames_done"] == len(kept)

    # Names, not idents: an ident is reused once its thread exits, and the prep thread can
    # finish before the writer starts.
    caller = threading.current_thread().name
    assert model.threads["embed"] == {caller}  # inference stays on the calling thread
    assert model.threads["prepare"] == {"lookahead-producer"}
    assert write_threads and all(
        n.startswith("lookahead-worker") for n in write_threads
    )


def test_a_write_error_fails_the_unit_after_the_batches_before_it(
    tmp_path, monkeypatch
):
    from detail.storage import zarr_io

    fixtures = _setup(tmp_path, object_ids=[0] * 8)
    real_add = zarr_io.EmbeddingPartWriter.add_row
    calls = {"n": 0}

    def _add_row(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 5:  # the first row of the third batch of two
            raise OSError("disk full")
        return real_add(self, *a, **k)

    monkeypatch.setattr(zarr_io.EmbeddingPartWriter, "add_row", _add_row)
    before = _lookahead_threads()
    result, _, _, out_zarr = _embed_one(
        tmp_path, model=_SplitEmbedder(), fixtures=fixtures, batch_size=2
    )
    assert result["status"] == "failed"
    assert "OSError: disk full" in result["error"]
    assert result["frames_done"] == 4
    assert PartZarrReader(out_zarr).read()["frame_idx"].tolist() == [0, 1, 2, 3]
    assert _wait_for_threads(before) == before


@pytest.mark.parametrize("outcome", ["complete", "stop", "model_error"])
def test_no_stage_thread_outlives_the_unit(tmp_path, outcome):
    fixtures = _setup(tmp_path, object_ids=[0] * 9)
    stop_flag_path = tmp_path / "stop.flag"

    class _Model(_SplitEmbedder):
        def embed_prepared(self, prepared):
            if outcome == "model_error" and self.batch_calls:
                raise RuntimeError("simulated OOM")
            out = super().embed_prepared(prepared)
            if outcome == "stop":
                stop_flag_path.touch()
            return out

    before = _lookahead_threads()
    result, _, _, _ = _embed_one(
        tmp_path,
        model=_Model(),
        fixtures=fixtures,
        stop_flag_path=stop_flag_path,
        batch_size=2,
    )
    assert result["status"] == ("complete" if outcome == "complete" else "failed")
    if outcome != "complete":
        assert result["frames_done"] == 2  # the one batch computed before is written
    assert _wait_for_threads(before) == before
