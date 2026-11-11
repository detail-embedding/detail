"""Per-unit embedding worker — extracted from ``pipeline/embedding.py``.

One pure-ish function: ``embed_one`` embeds the rows of one ``(track_id,
part_index)`` unit with a **pre-loaded** model + renderer. Callable from the
coordinator both **inline** (single device path — no subprocess) and from a
**ProcessPoolExecutor worker** (multi-GPU path — model loaded once per worker
via the ``pool_initializer`` here).

Cross-process channel is file-based (no queues, no managers):
- Per-unit progress JSON snapshots: ``<run_dir>/parts/track_<tid>_<part_id>.json``,
  written atomically (tmp + ``os.replace``) so the coordinator's polling reader
  never sees torn writes.
- Stop signal: ``<run_dir>/stop.flag``. Workers poll between batches and exit
  early with ``status="failed"`` when present (stop = fail for embedding, same
  as detection — embedding has no resume support, so partial would be
  misleading).

The function is intentionally synchronous and returns a fully-populated status
dict — exception capture happens at the caller in the multi-GPU path so one
worker failing doesn't crash the pool.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from detail.analysis import geometry
from detail.core import atomic
from detail.core import cpu
from detail.core import lookahead
from detail.core import progress as core_progress
from detail.storage.zarr_io import (
    DetectionPartReader,
    EmbeddingPartWriter,
    KinematicsPartWriter,
    PartZarrReader,
    kinematics_part_path,
)

PROGRESS_DIRNAME = "parts"
STOP_FLAG_NAME = "stop.flag"

# Populated by pool_initializer in the multi-GPU path; not used inline.
_WORKER_STATE: dict = {}


def progress_json_name(track_id: int, part_id: str) -> str:
    return f"track_{int(track_id)}_{part_id}.json"


def _render_workers() -> int:
    """Threads that render rows ahead of the GPU: the core rule (``core/cpu.py``), split among
    the per-GPU workers a multi-GPU run started on this machine (``pool_initializer`` records
    how many); the inline path has the machine to itself."""
    return cpu.thread_workers(share=_WORKER_STATE.get("share", 1))


def _render_batches(dataset, batch_size: int):
    """Yield ``(images, frame_idxs, t_ns, det_rows)`` batches in frame order, rendered ahead of
    the GPU consumer (§14 *Execution*).

    Three stages run at once, so the GPU stops waiting on the CPU:

    - **decode** — a pixel-reading renderer's frames come from one thread walking the video
      forward (``DetectionMaskDataset.walk_frames``, Tenet 11), ``decode_depth`` frames ahead;
      a mask-only renderer opens no video and starts at the next stage;
    - **render** — each row's mask decode, geometry and render on a thread pool, a window of
      ``3 * batch_size`` rows ahead, handed back in frame order;
    - **inference** — the caller's loop, on the calling thread.

    Every row is rendered by the same ``_render_row`` the serial ``iter(dataset)`` calls, on the
    same frame, in the same order, so the rows are byte-identical to it
    (``test_embedding_worker.py`` holds them to that). Frames held at once are at most
    ``decode_depth + window``: O(batch_size), never O(video).
    """
    n = len(dataset)
    if n == 0:
        return
    workers = _render_workers()
    window = batch_size * 3
    if getattr(dataset.renderer, "needs_frame", True):
        frames = lookahead.background_iter(dataset.walk_frames(), depth=2 * workers)

        def render(item):
            i, target, frame = item
            return dataset._render_row(i, target, frame)

    else:
        targets = dataset.frame_indices.tolist()
        frames = range(n)

        def render(i):
            return dataset._render_row(i, targets[i], None)

    rows = lookahead.ordered_map(render, frames, workers=workers, window=window)
    try:
        batch: list[dict] = []
        for row in rows:
            batch.append(row)
            if len(batch) == batch_size:
                yield _as_batch(batch)
                batch = []
        if batch:
            yield _as_batch(batch)
    finally:
        rows.close()  # stop the pool and the decode thread with the loop that started them


def _as_batch(rows: list[dict]) -> tuple[list, list, list, list]:
    return (
        [r["image"] for r in rows],
        [r["frame_idx"] for r in rows],
        [r["t_ns"] for r in rows],
        [r["det_row"] for r in rows],
    )


def _as_float32(embeddings) -> np.ndarray:
    """A model's ``(B, D)`` output as float32 numpy, whether it returned a tensor or an array."""
    if hasattr(embeddings, "detach"):
        return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(embeddings, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Per-unit execution (the function both paths call)
# --------------------------------------------------------------------------- #
def embed_one(
    *,
    track_id: int,
    part_index: int,
    part_id: str,
    video_path: Path,
    det_zarr_path: Path,
    assignment_zarr_path: Path,
    mask_size: tuple[int, int],
    out_zarr_path: Path,
    progress_json_path: Path | None,
    stop_flag_path: Path | None,
    model,
    renderer,
    output_dim: int,
    batch_size: int,
    gpu_idx: int | None,
    every_n_frames: int = 1,
    heading: dict | None = None,
) -> dict:
    """Embed one (track_id, part_index) unit with a pre-loaded model + renderer.

    Returns a status dict::

        {track_id, part_index, part_id, gpu, status, frames_done,
         frames_total, error}

    ``status`` is ``"complete"`` on a clean run, ``"failed"`` on stop.flag
    detection or any caught exception. ``frames_done`` reflects rows written
    to the embedding zarr.

    ``heading`` (§14 alignment) is the unit's per-frame heading as ``_build_units``
    computed it — ``{frame_idx, deg, source, mirrored}`` lists over the track's frames at
    this cadence. A frame whose ``source`` is ``SOURCE_NONE`` has no heading within the gap
    and is **not embedded**; the rest carry ``heading_deg`` / ``align_mirrored`` into the
    render context and onto the part's alignment columns.
    """
    # Imports here so the module stays torch-free at import time (Tenet 13).
    import torch

    from detail.pipeline.embedding import DetectionMaskDataset

    det = DetectionPartReader(Path(det_zarr_path)).read()
    object_id = PartZarrReader(Path(assignment_zarr_path)).read()["object_id"]
    keep_idx = np.where(np.asarray(object_id) == int(track_id))[0]
    # Group the track's detections by frame: a frame may hold more than one
    # detection (e.g. anchor_tracker secondaries), OR-merged into one subject
    # image downstream — so one embedding row per frame. The representative row
    # (highest confidence) is recorded as det_row for traceability.
    frame_col = np.asarray(det["frame_idx"])
    t_ns_col = np.asarray(det["t_ns"])
    conf_col = np.asarray(det["confidence"])
    by_frame: dict[int, list[int]] = {}
    for row in keep_idx.tolist():
        by_frame.setdefault(int(frame_col[row]), []).append(row)
    # Thin to this run's cadence. The predicate is on the *source* frame index, so it is
    # the same rule at every stage and composes with whatever detection already skipped —
    # a stride that is a multiple of detection's (validated) selects a subset of these rows.
    frames = [f for f in sorted(by_frame) if f % every_n_frames == 0]
    aux_per_frame: list[dict] | None = None
    if heading is not None:
        from detail.analysis.body_axis import SOURCE_NONE

        lookup = {
            int(f): (float(d), int(s), bool(m))
            for f, d, s, m in zip(
                heading["frame_idx"],
                heading["deg"],
                heading["source"],
                heading["mirrored"],
            )
        }
        kept = [
            f for f in frames if f in lookup and lookup[f][1] != SOURCE_NONE
        ]  # no heading within the gap: not embedded (§14)
        aux_per_frame = [
            {
                "heading_deg": lookup[f][0],
                "heading_source": lookup[f][1],
                "align_mirrored": lookup[f][2],
            }
            for f in kept
        ]
        frames = kept
    rep_rows = [max(by_frame[f], key=lambda r: float(conf_col[r])) for f in frames]
    frame_indices = np.asarray(frames, dtype=np.int64)
    frame_t_ns = np.asarray([int(t_ns_col[r]) for r in rep_rows], dtype=np.int64)
    mask_rles_per_frame = [
        [str(det["mask_rle"][r]) for r in by_frame[f]] for f in frames
    ]
    # Union bbox over the frame's detections (matching render_preview) so the
    # centered constructors crop to the subject, not the frame center (§14).
    bbox_col = np.asarray(det["bbox"])
    bbox_per_frame = [
        (
            min(float(bbox_col[r][0]) for r in by_frame[f]),
            min(float(bbox_col[r][1]) for r in by_frame[f]),
            max(float(bbox_col[r][2]) for r in by_frame[f]),
            max(float(bbox_col[r][3]) for r in by_frame[f]),
        )
        for f in frames
    ]
    frames_total = int(len(frames))

    out_zarr_path = Path(out_zarr_path)
    out_zarr_path.parent.mkdir(parents=True, exist_ok=True)

    status = "complete"
    error: str | None = None
    frames_done = 0

    if progress_json_path is not None:
        _write_progress_json(
            Path(progress_json_path),
            track_id=track_id,
            part_id=part_id,
            frames_done=0,
            frames_total=frames_total,
            status="running",
        )

    try:
        dataset = DetectionMaskDataset(
            video_path=Path(video_path),
            frame_indices=frame_indices,
            t_ns=frame_t_ns,
            det_rows=np.asarray(rep_rows, dtype=np.int64),
            mask_rles_per_frame=mask_rles_per_frame,
            bboxes=bbox_per_frame,
            mask_size=mask_size,
            renderer=renderer,
            track_id=track_id,
            aux_per_frame=aux_per_frame,
        )
        aux_by_frame = {int(f): aux for f, aux in zip(frames, aux_per_frame or [])}
        stopped = False
        written_so_far = 0  # touched only by the writer thread

        def prepared_batches():
            """Prep stage (its own thread): the model's CPU half on each rendered batch."""
            batches = _render_batches(dataset, batch_size)
            try:
                for images, f_idx, t_ns_b, rows in batches:
                    yield model.prepare_batch(images), f_idx, t_ns_b, rows
            finally:
                batches.close()  # the decode thread and render pool stop with this stage

        def computed_batches(prepared):
            """Inference stage (the calling thread): the device half, nothing else."""
            nonlocal stopped
            for inputs, f_idx, t_ns_b, rows in prepared:
                if stop_flag_path is not None and Path(stop_flag_path).exists():
                    stopped = True
                    return
                with torch.inference_mode():
                    arr = _as_float32(model.embed_prepared(inputs))
                yield arr, f_idx, t_ns_b, rows

        def write_batch(item) -> int:
            """Writer stage (one thread, so rows reach disk in order)."""
            nonlocal written_so_far
            arr, f_idx, t_ns_b, rows = item
            for b in range(arr.shape[0]):
                aux = aux_by_frame.get(int(f_idx[b]))
                writer.add_row(
                    int(f_idx[b]),
                    int(t_ns_b[b]),
                    int(rows[b]),
                    arr[b],
                    **(
                        {
                            "heading_deg": aux["heading_deg"],
                            "heading_source": aux["heading_source"],
                            "align_mirrored": aux["align_mirrored"],
                        }
                        if aux is not None
                        else {}
                    ),
                )
                # Same rows, same order, written from the geometry the dataset collected
                # while it had the merged mask in hand. A frame the renderer never reached
                # has no entry, and writes NaN rather than being silently dropped — the two
                # files stay row-aligned.
                geom = dataset.geometry.get(int(f_idx[b]), geometry.NO_GEOMETRY)
                kinematics.add_row(
                    int(f_idx[b]),
                    int(t_ns_b[b]),
                    geom.cx_norm,
                    geom.cy_norm,
                    geom.mask_area_norm,
                    geom.bbox_norm,
                )
            written_so_far += arr.shape[0]
            if progress_json_path is not None:
                _write_progress_json(
                    Path(progress_json_path),
                    track_id=track_id,
                    part_id=part_id,
                    frames_done=written_so_far,
                    frames_total=frames_total,
                    status="running",
                )
            return int(arr.shape[0])

        # decode → render (``_render_batches``) → prep → inference → write, each stage a
        # bounded step ahead of the next (§14 *Execution*). The writers are opened and closed
        # here, on the calling thread; the writer pool has joined before they close.
        prepared = lookahead.background_iter(prepared_batches(), depth=2)
        try:
            with (
                EmbeddingPartWriter(
                    out_zarr_path, output_dim=output_dim, aligned=heading is not None
                ) as writer,
                KinematicsPartWriter(kinematics_part_path(out_zarr_path)) as kinematics,
            ):
                written = lookahead.ordered_map(
                    write_batch, computed_batches(prepared), workers=1, window=2
                )
                try:
                    for n in written:
                        frames_done += n
                finally:
                    written.close()
        finally:
            prepared.close()
        if stopped:
            status = "failed"
            error = "stop.flag detected during embedding"
    except Exception as e:
        status = "failed"
        error = f"{type(e).__name__}: {e}"

    result = {
        "track_id": int(track_id),
        "part_index": int(part_index),
        "part_id": part_id,
        "gpu": gpu_idx,
        "status": status,
        "frames_done": int(frames_done),
        "frames_total": frames_total,
        "output_dim": int(output_dim),
        "error": error,
    }
    if progress_json_path is not None:
        _write_progress_json(
            Path(progress_json_path),
            track_id=track_id,
            part_id=part_id,
            frames_done=frames_done,
            frames_total=frames_total,
            status=status,
        )
    return result


# --------------------------------------------------------------------------- #
# Pool initializer (multi-GPU path only)
# --------------------------------------------------------------------------- #
def pool_initializer(
    gpu_idx: int,
    model_id: str,
    model_variant: str | None,
    renderer_spec: list[dict],
    pipeline_runner_modules: tuple[str, ...],
    readout: str | None = None,
    input_size: int | None = None,
    share: int = 1,
) -> None:
    """Pin this worker to ``gpu_idx``, load the model + renderer once. Module-
    level globals are safe in a spawn-mode pool — each worker has isolated
    state, and tasks dispatched to the same worker reuse it.

    Must run **before any torch import** so ``CUDA_VISIBLE_DEVICES`` takes
    effect — that's why we set the env var first and then import inside.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    from detail.core.registries import EMBEDDING_MODELS, construct_embedding_model
    from detail.executor.runners import import_runner_modules
    from detail.transforms import (
        basic as _basic,
    )  # noqa: F401 — registers transforms
    from detail.transforms.renderer import MaskRenderer

    import_runner_modules(pipeline_runner_modules)
    model_cls = EMBEDDING_MODELS[model_id]
    model = construct_embedding_model(
        model_cls, model_variant, readout=readout, input_size=input_size
    )
    # ``model.load`` expects the Ultralytics-style index string (``"0"``), not a
    # ``"cuda:N"`` device string — backends prefix ``cuda:`` themselves and
    # would otherwise build ``cuda:cuda:0``. With CUDA_VISIBLE_DEVICES set to
    # the assigned GPU, the only visible device is index 0 from the child's
    # perspective.
    model.load("0")
    _WORKER_STATE["model"] = model
    _WORKER_STATE["renderer"] = MaskRenderer.from_spec(renderer_spec)
    _WORKER_STATE["output_dim"] = int(model.output_dim)
    _WORKER_STATE["gpu_idx"] = int(gpu_idx)
    _WORKER_STATE["share"] = max(1, int(share))  # sibling GPU workers on this machine


def pool_run_one(args: dict) -> dict:
    """Pool task entry — pulls model/renderer from worker-local state, calls
    ``embed_one``, and catches any exception so one bad unit doesn't crash the
    pool. Returns the same status dict shape with ``status="failed"`` on
    exception and the exception string in ``error``.
    """
    try:
        return embed_one(
            track_id=args["track_id"],
            part_index=args["part_index"],
            part_id=args["part_id"],
            video_path=Path(args["video_path"]),
            det_zarr_path=Path(args["det_zarr_path"]),
            assignment_zarr_path=Path(args["assignment_zarr_path"]),
            mask_size=tuple(args["mask_size"]),
            out_zarr_path=Path(args["out_zarr_path"]),
            progress_json_path=(
                Path(args["progress_json_path"])
                if args.get("progress_json_path")
                else None
            ),
            stop_flag_path=(
                Path(args["stop_flag_path"]) if args.get("stop_flag_path") else None
            ),
            model=_WORKER_STATE["model"],
            renderer=_WORKER_STATE["renderer"],
            output_dim=_WORKER_STATE["output_dim"],
            batch_size=int(args["batch_size"]),
            every_n_frames=int(args.get("every_n_frames", 1) or 1),
            gpu_idx=_WORKER_STATE.get("gpu_idx"),
            heading=args.get("heading"),
        )
    except Exception as e:
        return {
            "track_id": int(args["track_id"]),
            "part_index": int(args["part_index"]),
            "part_id": args["part_id"],
            "gpu": _WORKER_STATE.get("gpu_idx"),
            "status": "failed",
            "frames_done": 0,
            "frames_total": 0,
            "output_dim": int(_WORKER_STATE.get("output_dim", 0)),
            "error": f"{type(e).__name__}: {e}",
        }


# --------------------------------------------------------------------------- #
# Atomic progress JSON
# --------------------------------------------------------------------------- #
@core_progress.best_effort("unit progress")
def _write_progress_json(
    path: Path,
    *,
    track_id: int,
    part_id: str,
    frames_done: int,
    frames_total: int,
    status: str,
) -> None:
    """Atomic write so the coordinator's polling reader never sees torn JSON.

    **Cannot raise** — the embedding twin of the detection worker's writer, and found by
    sweeping for progress writers rather than by reading the module #289 was reported against.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "track_id": int(track_id),
                "part_id": part_id,
                "frames_done": int(frames_done),
                "frames_total": int(frames_total),
                "status": status,
                "ts": time.time(),
            }
        ),
        encoding="utf-8",
    )
    atomic.replace(tmp, path)


def read_progress_json(path: Path) -> dict | None:
    """Best-effort read — returns None if missing or transiently unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None
