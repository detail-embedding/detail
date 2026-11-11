"""Per-part detection worker — extracted from ``pipeline/detection.py``.

One pure-ish function: ``run_part`` runs detection on one video part with a
**pre-loaded** model. Callable from the coordinator both **inline** (single-GPU /
CPU / MPS path — no subprocess) and from a **ProcessPoolExecutor worker** (multi-
GPU path — model loaded once per worker via the ``pool_initializer`` here).

Cross-process communication is file-based (no queues, no managers):
- Per-part progress JSON snapshots: ``<run_dir>/parts/<part_id>.json``,
  written atomically (tmp + ``os.replace``) so the coordinator's polling reader
  never sees torn writes.
- Stop signal: ``<run_dir>/stop.flag``. Workers poll before each forward pass
  and exit early with ``status="failed"`` when present (per the design decision:
  stop = fail for detection).

The function is intentionally synchronous and returns a fully-populated status
dict — exception capture happens at the caller in the multi-GPU path so one
worker failing doesn't crash the pool.
"""

import functools
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from detail.core import atomic
from detail.core import lookahead
from detail.core import progress as core_progress
from detail.core.dataclasses import Prompt
from detail.core.protocols import DetectionModel
from detail.storage import recording as recording_store
from detail.storage import video as video_store
from detail.storage.zarr_io import DetectionPartWriter

_NO_BBOX = (0.0, 0.0, 0.0, 0.0)
# Source frames between progress JSON snapshots: one each time a written batch crosses a
# multiple of this. Tuned for ~0.5s update cadence at typical detection throughput (30
# fps source video → snapshot every ~0.7s); higher values feel laggy on short clips.
_PROGRESS_EVERY = 20
PROGRESS_DIRNAME = "parts"
STOP_FLAG_NAME = "stop.flag"

# Populated by pool_initializer in the multi-GPU path; not used inline.
_WORKER_STATE: dict = {}


# --------------------------------------------------------------------------- #
# Per-part execution (the function both paths call)
# --------------------------------------------------------------------------- #
def run_part(
    *,
    part: dict,
    part_index: int,
    parts_total: int,
    recording_dir: Path,
    recording_data: dict,
    out_zarr_path: Path,
    progress_json_path: Path | None,
    stop_flag_path: Path | None,
    model,
    prompt: Prompt,
    conf_threshold: float,
    gpu_idx: int | None,
    batch_size: int = 1,
    every_n_frames: int = 1,
) -> dict:
    """Process one video part with a pre-loaded model.

    Returns a parts_status dict matching the §7 ``job.json`` shape::

        {id, file, gpu, status, frames_done, frames_total, mask_resolution,
         dropped_frames, error}

    ``status`` is ``"complete"`` on a clean run, ``"failed"`` on stop.flag
    detection or any caught exception. ``frames_done`` / ``frames_total`` count
    **source** frames walked, not rows written: with ``every_n_frames > 1`` the walk is
    still every frame (demux is sequential — Tenet 11) and is what the run's wall clock
    tracks, so the bar stays honest at any stride.

    **Four stages overlap** (§7 *Execution*): one thread walks the video, one runs the
    model's CPU half (``DetectionModel.prepare_batch``) a batch or two ahead, this thread runs
    only the device half, and one writer thread runs the host half, the mask encoding and
    the zarr rows, in order. The stop flag is checked before each forward pass; the batches
    already computed are written, and ``frames_done`` on a stop is the source frames walked
    through the last batch written, so it never counts a frame with no rows on disk.

    **The timestamp map and cv2 must agree on the frame count** (Tenet 8). Row
    ``frame_idx`` is placed at ``map[frame_idx]``, so a map of a different length
    misplaces every row, and the old handling hid it: a surplus frame got
    ``t_ns = -1`` and a shortfall was recorded as frames dropped at the end — when an
    edit-list trimmed file actually loses its frames at the *start*, shifting the whole
    part. The rule:

    - **cv2 returns more frames than the map has** → ``FrameCountMismatchError``. No
      reading of that is a dropped frame.
    - **cv2 stops before the map ends** → the file's packets are recounted
      (``video.shown_frame_count``, a demux pass paid only here). If they agree with
      the map, the frames exist in the file and cv2 could not decode them — a damaged
      or truncated tail — so they are recorded in ``dropped_frames`` as before. If
      they disagree, the map was built from a different reading of this file and
      ``FrameCountMismatchError`` is raised. When the packets cannot be trusted for a
      count (the map came from a full decode), the shortfall stays a dropped tail:
      there is nothing cheaper to compare against.

    The error is raised rather than returned so the caller's isolation boundary
    records it as the part's ``error`` — a failed part, not a quiet one.

    Raises:
        FrameCountMismatchError: As above, naming the file and both counts.
    """
    src_path = recording_store.resolve_part_video_path(recording_dir, part)
    tsmap = recording_store.load_active_timestamps(
        recording_dir, recording_data, part_index
    )
    # The authoritative frame count is the timestamp map (Tenet 8) — the same
    # count the read loop bounds t_ns lookups by. Not cv2's CAP_PROP_FRAME_COUNT
    # (duration × advertised fps), which over-reports when a file's header fps
    # differs from its true PTS rate and would invent tail dropped_frames.
    in_total = len(tsmap)
    walk: dict = {"frame_idx": 0, "dropped": [], "mask_resolution": None}
    prepare, run_prepared, finish = (
        _model_half(model, name)
        for name in ("prepare_batch", "run_prepared", "finish_batch")
    )
    stopped = False
    reported = 0  # touched only by the writer thread

    def prepared_batches():
        """Prep stage (its own thread): the model's CPU half, a batch or two ahead."""
        batches = lookahead.background_iter(
            _frame_batches(src_path, tsmap, in_total, batch_size, every_n_frames, walk),
            depth=2,
        )
        try:
            for frames, meta, walked in batches:
                yield prepare(frames, [prompt] * len(frames)), meta, walked
        finally:
            batches.close()  # the decode thread stops with this stage

    def computed_batches(prepared):
        """Inference stage (the calling thread): the device half, nothing else."""
        nonlocal stopped
        for inputs, meta, walked in prepared:
            if stop_flag_path is not None and stop_flag_path.exists():
                stopped = True
                return
            try:
                raw = run_prepared(inputs)
            except RuntimeError as exc:  # surface a GPU OOM with an actionable hint
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"detection ran out of GPU memory at batch_size="
                        f"{batch_size} (frame resolution {walk['mask_resolution']}) — "
                        f"lower the batch size"
                    ) from exc
                raise
            yield raw, meta, walked

    def write_batch(item) -> int:
        """Writer stage (one thread, so rows reach disk in order): the model's host half,
        mask encoding and the zarr rows."""
        nonlocal reported
        raw, meta, walked = item
        for (frame_idx, t_ns), result in zip(meta, finish(raw)):
            detection_idx = 0
            for det in result.detections():
                if det.confidence < conf_threshold:
                    continue
                writer.add_row(
                    frame_idx,
                    t_ns,
                    detection_idx,
                    det.class_name,
                    float(det.confidence),
                    tuple(det.bbox) if det.bbox is not None else _NO_BBOX,
                    _encode_mask(det.mask),
                )
                detection_idx += 1
        if progress_json_path is not None and (
            walked // _PROGRESS_EVERY > reported // _PROGRESS_EVERY
        ):
            _write_progress_json(
                progress_json_path,
                frames_done=walked,
                frames_total=in_total,
                status="running",
            )
            reported = walked
        return walked

    # decode → prep → inference → write, each a bounded step ahead of the next (§7
    # *Execution*). The writer is opened and closed here, on the calling thread, after the
    # writer pool has joined.
    written_through = 0
    prepared = lookahead.background_iter(prepared_batches(), depth=2)
    try:
        with DetectionPartWriter(out_zarr_path) as writer:
            written = lookahead.ordered_map(
                write_batch, computed_batches(prepared), workers=1, window=2
            )
            try:
                for walked in written:
                    written_through = walked
            finally:
                written.close()
    finally:
        prepared.close()

    status = "failed" if stopped else "complete"
    frame_idx = written_through if stopped else walk["frame_idx"]
    result = {
        "id": recording_store.part_stem(part),
        "file": part.get("file") or recording_store.part_stem(part),
        "gpu": gpu_idx,
        "status": status,
        "frames_done": frame_idx,
        "frames_total": in_total,
        "mask_resolution": walk["mask_resolution"],
        "dropped_frames": [] if stopped else walk["dropped"],
        "error": "stop.flag detected during detection" if stopped else None,
    }
    if progress_json_path is not None:
        _write_progress_json(
            progress_json_path,
            frames_done=frame_idx,
            frames_total=in_total,
            status=status,
        )
    return result


def _frame_batches(src_path, tsmap, in_total, batch_size, every_n_frames, walk):
    """The part's frames in batches of up to ``batch_size``, from one forward walk.

    Demuxes every frame (sequential, no seek — Tenet 11) but pays the BGR convert only on the
    frames the stride keeps; skipped frames are still walked, so ``frame_idx`` stays a *source*
    frame index. Yields ``(frames, [(frame_idx, t_ns)], walked)``, ``walked`` being the source
    frames walked through that batch. Records ``mask_resolution``, the final ``frame_idx`` and
    the ``dropped`` tail in ``walk``; ``run_part``'s docstring has the frame-count rule.
    """
    cap = cv2.VideoCapture(str(src_path))
    frame_idx = 0
    frames: list = []
    meta: list[tuple[int, int]] = []
    try:
        while cap.grab():
            if frame_idx >= in_total:
                raise _mismatch(src_path, in_total, f"more than {in_total}")
            if frame_idx % every_n_frames:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if walk["mask_resolution"] is None:
                walk["mask_resolution"] = [int(frame.shape[0]), int(frame.shape[1])]
            frames.append(frame)
            meta.append((frame_idx, int(tsmap.t_ns(frame_idx))))
            frame_idx += 1
            if len(frames) >= batch_size:
                yield frames, meta, frame_idx
                frames, meta = [], []
        walk["frame_idx"] = frame_idx
        if frames:  # the trailing partial batch
            yield frames, meta, frame_idx
        walk["dropped"] = _dropped_tail(src_path, frame_idx, in_total)
    finally:
        cap.release()


def _model_half(model, name: str):
    """``model``'s half of ``predict_batch`` (§5), or the protocol's default for a model that
    does not split it, which runs the whole call on the inference thread."""
    half = getattr(model, name, None)
    if half is not None:
        return half
    return functools.partial(getattr(DetectionModel, name), model)


def _mismatch(
    src_path: Path, map_count: int, cv2_count: str
) -> video_store.FrameCountMismatchError:
    return video_store.FrameCountMismatchError(
        f"{src_path.name}: its timestamp map has {map_count} frames but the video "
        f"decodes to {cv2_count}, so every detection in this part would be placed at "
        "the wrong time. The map was made from a different reading of this file (an "
        "import from before edit-list trimming was handled, or a user map for another "
        "file). Re-import the video so its map is regenerated from the frames it shows."
    )


def _dropped_tail(src_path: Path, frame_idx: int, in_total: int) -> list[int]:
    """The frames cv2 could not read at the end of a part, or a mismatch error.

    See ``run_part``'s docstring for the rule this applies.
    """
    if frame_idx >= in_total:
        return []
    shown = video_store.shown_frame_count(src_path)
    if shown is not None and shown != in_total:
        raise _mismatch(src_path, in_total, f"{frame_idx} (its packets say {shown})")
    return list(range(frame_idx, in_total))


# --------------------------------------------------------------------------- #
# Pool initializer (multi-GPU path only)
# --------------------------------------------------------------------------- #
def pool_initializer(
    gpu_idx: int,
    experiment_root: str,
    model_id: str,
    weights_source: str,
    model_variant: str | None,
    prompt_dict: dict,
    conf_threshold: float,
    batch_size: int,
    pipeline_runner_modules: tuple[str, ...],
) -> None:
    """Pin this worker to ``gpu_idx``, load the model + prompt once. Module-
    level globals are safe in a spawn-mode pool — each worker has isolated
    state, and tasks dispatched to the same worker reuse it.

    Must run **before any torch import** so ``CUDA_VISIBLE_DEVICES`` takes
    effect — that's why we set the env var first and then import inside.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    from detail.executor.runners import import_runner_modules
    from detail.models.resolve import resolve_detection_model

    import_runner_modules(pipeline_runner_modules)  # populates the model registries
    model = resolve_detection_model(
        experiment_root, model_id, weights_source, model_variant
    )
    # ``model.load`` expects the Ultralytics-style index string that
    # ``resolve_device`` produces (``"0"``, ``"0,1"``), not a ``"cuda:N"`` device
    # string — backends like SAM3 prefix ``cuda:`` themselves and would otherwise
    # build ``cuda:cuda:0``. With CUDA_VISIBLE_DEVICES set to the assigned GPU,
    # the only visible device is index 0 from the child's perspective.
    model.load("0")
    model.set_conf_threshold(conf_threshold)
    _WORKER_STATE["model"] = model
    _WORKER_STATE["prompt"] = Prompt.from_dict(prompt_dict)
    _WORKER_STATE["conf_threshold"] = conf_threshold
    _WORKER_STATE["batch_size"] = batch_size
    _WORKER_STATE["gpu_idx"] = gpu_idx


def pool_run_part(args: dict) -> dict:
    """Pool task entry — pulls model/prompt from worker-local state, calls
    ``run_part``, and catches any exception so one bad part doesn't crash the
    pool. Returns the same status dict shape with ``status="failed"`` on
    exception and the exception string in ``error``.
    """
    try:
        return run_part(
            part=args["part"],
            part_index=args["part_index"],
            parts_total=args["parts_total"],
            recording_dir=Path(args["recording_dir"]),
            recording_data=args["recording_data"],
            every_n_frames=int(args.get("every_n_frames") or 1),
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
            prompt=_WORKER_STATE["prompt"],
            conf_threshold=_WORKER_STATE["conf_threshold"],
            gpu_idx=_WORKER_STATE.get("gpu_idx"),
            batch_size=_WORKER_STATE.get("batch_size", 1),
        )
    except Exception as e:
        # Capture-and-return so the pool keeps running other parts.
        return {
            "id": recording_store.part_stem(args["part"]),
            "file": args["part"].get("file") or recording_store.part_stem(args["part"]),
            "gpu": _WORKER_STATE.get("gpu_idx"),
            "status": "failed",
            "frames_done": 0,
            "frames_total": 0,
            "mask_resolution": None,
            "dropped_frames": [],
            "error": f"{type(e).__name__}: {e}",
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _encode_mask(mask) -> str:
    if mask is None:
        return ""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


@core_progress.best_effort("part progress")
def _write_progress_json(
    path: Path, *, frames_done: int, frames_total: int, status: str
) -> None:
    """Atomic write so the coordinator's polling reader never sees torn JSON.

    **Cannot raise.** Its reader below has always been best-effort ("returns None if missing or
    transiently unreadable"); the write side was not, and it is called from inside ``run_part``,
    whose caller records a raised exception as *the part* failing with 0 frames. Same asymmetry
    that produced #289 one level up: the fragility was understood at the read and not at the
    write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {"frames_done": frames_done, "frames_total": frames_total, "status": status}
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
