"""Detection stage runner — coordinator over one or more video parts (Sections 7, 9).

Two execution paths share one per-part function (``_detection_worker.run_part``):

- **Inline** (single device): load model once in this process, run each part
  sequentially. Zero subprocess overhead.
- **Multi-GPU** (``compute.backend == "cuda"`` and ``len(device_indices) > 1``):
  ``ProcessPoolExecutor(spawn)`` sized to ``len(device_indices)``; each worker
  is pinned to one GPU via ``CUDA_VISIBLE_DEVICES`` before any torch import,
  loads the model once at startup, then chews through parts as they're
  scheduled round-robin per §7.

Cross-process channel is file-based: workers write per-part progress JSONs
under ``<run_dir>/parts/<part_id>.json``; the coordinator polls these between
``DetectionProgressEvent`` emissions. Stop = the coordinator touches
``<run_dir>/stop.flag``; workers poll it and exit early with ``status="failed"``.
Per the design decision, **a stop is never partial** — detection has no resume
support, so partial would be misleading. A stopped part is failed; under a cancel the worker
records the run ``cancelled`` (``DetectionModule.resumes = False``).
"""

import hashlib
import json
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from detail.configs.detection import (
    BATCHABLE_DEFAULT_BATCH_SIZE,
    DetectionConfig,
)
from detail.core.dataclasses import (
    ACTIVE_STATUSES,
    DetectionJob,
    DetectionProgressEvent,
)
from detail.core.progress import PROGRESS_INTERVAL_S, remaining_seconds
from detail.core.protocols import ModelNotReadyError
from detail.core.utils import check_cuda_indices, resolve_device, utc_timestamp
from detail.executor import rundir
from detail.executor.runners import (
    ProcessingModule,
    import_runner_modules,
    register_processing_module,
)
from detail.models.resolve import (
    resolve_detection_model,
    resolve_detection_model_auto,
)
from detail.pipeline import _detection_worker
from detail.storage import experiment, recording as recording_store
from detail.storage import video_integrity

# Detection runs a multiprocess pool and emits by *polling* the worker progress
# files at the shared live-bar cadence (§6) — it doesn't use the in-loop
# ThrottledProgress helper (a worker.join(timeout) loop is already incremental).
_PROGRESS_POLL_INTERVAL_S = PROGRESS_INTERVAL_S


def _part_frame_counts(
    recording_dir: Path,
    data: dict,
    files: list[dict],
) -> dict[str, int]:
    """Expected frame count per source video, from the authoritative timestamp
    map (Tenet 8) — the same count the worker bounds its loop by. Lets the
    coordinator report the correct ``frames_total`` denominator from the very
    first progress event (without it, every part shows ``0/0`` until the worker
    writes its first JSON snapshot, which on a long model warmup looks like a
    stall). Not cv2's ``CAP_PROP_FRAME_COUNT`` (duration × advertised fps), which
    over-reports when a file's header fps differs from its true PTS rate — that
    inflated the denominator so the bar never reached 100% and invented tail
    ``dropped_frames``."""
    counts: dict[str, int] = {}
    for part_index, part in enumerate(files):
        tsmap = recording_store.load_active_timestamps(recording_dir, data, part_index)
        counts[recording_store.part_stem(part)] = len(tsmap)
    return counts


def detection_run_dir(experiment_root, job: DetectionJob) -> Path:
    """Run directory for a detection job: ``recordings/<id>/detections/<job_id>/``."""
    return (
        experiment.recording_path(experiment_root, job.recording_id)
        / "detections"
        / job.job_id
    )


def prompt_fingerprint(prompt) -> str:
    """Stable short digest of a detection prompt — its target classes and cues — so
    two detection runs that differ only in prompt get distinct run specs (§6.5). The
    prompt *is* the class definition for open-set models, so it must be part of the
    detection run's identity. ``None`` (closed-set models take no prompt) hashes to a
    fixed marker so all closed-set runs of a model share the same prompt component."""
    payload = (
        "none"
        if prompt is None
        else json.dumps(prompt.to_dict(), sort_keys=True, separators=(",", ":"))
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def prompt_classes(prompt) -> list[str]:
    """Sorted target class names of a detection prompt — the human-readable half of
    the prompt identity, so the run-spec picker can tell two runs of the same model +
    confidence apart by *what* they detected (the fingerprint alone is opaque). Empty
    for a closed-set model (no prompt; its own id/classes distinguish it)."""
    if prompt is None:
        return []
    return sorted({target.class_name for target in prompt.targets})


def run(job: DetectionJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    recording_dir = run_dir.parent.parent
    data = recording_store.read_recording(recording_dir)
    files = recording_store.video_files(data)

    progress_dir = run_dir / _detection_worker.PROGRESS_DIRNAME
    progress_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = run_dir / _detection_worker.STOP_FLAG_NAME

    device_indices = _resolve_device_indices(job)
    if _is_multi_gpu(job, device_indices):
        parts_status = _run_multi_gpu(
            job=job,
            run_dir=run_dir,
            recording_dir=recording_dir,
            data=data,
            files=files,
            device_indices=device_indices,
            progress_dir=progress_dir,
            stop_flag=stop_flag,
            progress_cb=progress_cb,
            stop_event=stop_event,
        )
    else:
        parts_status = _run_inline(
            job=job,
            run_dir=run_dir,
            recording_dir=recording_dir,
            data=data,
            files=files,
            progress_dir=progress_dir,
            stop_flag=stop_flag,
            progress_cb=progress_cb,
            stop_event=stop_event,
        )

    mask_resolution = _first_mask_resolution(parts_status)
    rundir.update_payload(run_dir, mask_resolution=mask_resolution, parts=parts_status)
    _raise_if_any_failed(parts_status)


# --------------------------------------------------------------------------- #
# Inline (single-device) path
# --------------------------------------------------------------------------- #
def _run_inline(
    *,
    job,
    run_dir,
    recording_dir,
    data,
    files,
    progress_dir,
    stop_flag,
    progress_cb,
    stop_event,
) -> list[dict]:
    part_totals = _part_frame_counts(recording_dir, data, files)
    start = time.monotonic()
    # The bar appears at 0% immediately and sits there with "loading model" while
    # the (slow) weight load runs, instead of showing nothing until the first frame.
    if progress_cb is not None:
        progress_cb(
            _aggregate_event(
                _read_live_parts_status(files, progress_dir, part_totals),
                len(files),
                start,
                partial_running=True,
                message="loading model",
            )
        )

    model = resolve_detection_model(
        Path(recording_dir).parents[1],
        job.model_id,
        job.weights_source,
        job.model_variant,
    )
    model.load(resolve_device(job.compute, model.supported_backends))
    model.set_conf_threshold(job.conf_threshold)

    parts_status: list[dict] = []
    for part_index, part in enumerate(files):
        if stop_event is not None and stop_event.is_set():
            stop_flag.touch()
        out_zarr = run_dir / (recording_store.part_stem(part) + ".zarr")
        progress_json = progress_dir / (recording_store.part_stem(part) + ".json")
        # Run the part in a thread so the coordinator can poll the worker's
        # per-frame JSON snapshots and stream live progress (the multi-GPU path
        # gets this from the process pool; inline would otherwise emit only once
        # per completed part — invisible for a single-part recording).
        result_holder: dict[str, dict] = {}

        def _work(
            part=part, part_index=part_index, out_zarr=out_zarr, pj=progress_json
        ):
            try:
                result_holder["result"] = _detection_worker.run_part(
                    part=part,
                    part_index=part_index,
                    parts_total=len(files),
                    recording_dir=recording_dir,
                    recording_data=data,
                    every_n_frames=job.every_n_frames,
                    out_zarr_path=out_zarr,
                    progress_json_path=pj,
                    stop_flag_path=stop_flag,
                    model=model,
                    prompt=job.prompt,
                    conf_threshold=job.conf_threshold,
                    gpu_idx=None,
                    batch_size=job.batch_size,
                )
            except Exception as exc:  # isolation boundary: a worker crash (e.g. a batch
                # OOM) becomes a failed part, not a KeyError that crashes the coordinator.
                result_holder["result"] = {
                    "id": recording_store.part_stem(part),
                    "file": part.get("file") or recording_store.part_stem(part),
                    "gpu": None,
                    "status": "failed",
                    "frames_done": 0,
                    "frames_total": 0,
                    "mask_resolution": None,
                    "dropped_frames": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

        worker = threading.Thread(target=_work)
        worker.start()
        while worker.is_alive():
            worker.join(_PROGRESS_POLL_INTERVAL_S)
            # The part polls the flag, not the event: touch it the moment a stop arrives, as
            # the multi-GPU loop does. Checking only between parts never reached a one-part
            # recording's frame loop, so a cancel ran the whole part to its end (#28).
            if (
                stop_event is not None
                and stop_event.is_set()
                and not stop_flag.exists()
            ):
                stop_flag.touch()
            if progress_cb is not None:
                progress_cb(
                    _aggregate_event(
                        _read_live_parts_status(files, progress_dir, part_totals),
                        len(files),
                        start,
                        partial_running=True,
                    )
                )
        parts_status.append(result_holder["result"])
        if progress_cb is not None:
            progress_cb(
                _aggregate_event(parts_status, len(files), start, partial_running=False)
            )
    return parts_status


# --------------------------------------------------------------------------- #
# Multi-GPU pool path
# --------------------------------------------------------------------------- #
def _run_multi_gpu(
    *,
    job,
    run_dir,
    recording_dir,
    data,
    files,
    device_indices,
    progress_dir,
    stop_flag,
    progress_cb,
    stop_event,
) -> list[dict]:
    # Resolve the model in *this* process before spawning workers; gives a clean
    # error instead of a cryptic pool failure.
    experiment_root = Path(recording_dir).parents[1]
    resolve_detection_model(
        experiment_root, job.model_id, job.weights_source, job.model_variant
    )

    from detail.api._runner_modules import PIPELINE_RUNNERS

    n_workers = len(device_indices)
    # Round-robin: part i goes to GPU device_indices[i % n_workers].
    # ProcessPoolExecutor with one worker per GPU + a per-worker initializer.
    # Workers stay alive across multiple parts (model loads once per worker).
    mp_ctx = mp.get_context("spawn")
    parts_status: list[dict] = []
    start = time.monotonic()

    # We need one initializer per worker process but ProcessPoolExecutor's
    # initializer is shared across all workers — we can't pin different GPUs
    # to different workers that way. Workaround: spawn N separate single-worker
    # pools, each with its own initializer, and round-robin parts across them.
    pools: list[ProcessPoolExecutor] = []
    try:
        for gpu_idx in device_indices:
            pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp_ctx,
                initializer=_detection_worker.pool_initializer,
                initargs=(
                    int(gpu_idx),
                    str(experiment_root),
                    job.model_id,
                    job.weights_source,
                    job.model_variant,
                    job.prompt.to_dict(),
                    job.conf_threshold,
                    job.batch_size,
                    PIPELINE_RUNNERS,
                ),
            )
            pools.append(pool)

        futures = {}
        for part_index, part in enumerate(files):
            assigned_gpu_pool = part_index % n_workers
            out_zarr = run_dir / (recording_store.part_stem(part) + ".zarr")
            progress_json = progress_dir / (recording_store.part_stem(part) + ".json")
            args = {
                "part": part,
                "part_index": part_index,
                "parts_total": len(files),
                "recording_dir": str(recording_dir),
                "recording_data": data,
                "every_n_frames": job.every_n_frames,
                "out_zarr_path": str(out_zarr),
                "progress_json_path": str(progress_json),
                "stop_flag_path": str(stop_flag),
            }
            fut = pools[assigned_gpu_pool].submit(_detection_worker.pool_run_part, args)
            futures[fut] = part_index

        pending = set(futures)
        part_totals = _part_frame_counts(recording_dir, data, files)
        if progress_cb is not None:
            progress_cb(
                _aggregate_event(
                    _read_live_parts_status(files, progress_dir, part_totals),
                    len(files),
                    start,
                    partial_running=True,
                    message="loading model",
                )
            )
        while pending:
            if (
                stop_event is not None
                and stop_event.is_set()
                and not stop_flag.exists()
            ):
                stop_flag.touch()
            done, pending = _wait_for_progress(
                pending, files, progress_dir, part_totals, progress_cb, start
            )
            for fut in done:
                parts_status.append(fut.result())
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)

    parts_status.sort(key=lambda r: r["id"])
    return parts_status


def _wait_for_progress(pending, files, progress_dir, part_totals, progress_cb, start):
    """Wait up to one poll-interval for any future to complete; emit progress
    aggregated from the per-part JSON files in the meantime."""
    from concurrent.futures import wait, FIRST_COMPLETED

    done, still_pending = wait(
        pending, timeout=_PROGRESS_POLL_INTERVAL_S, return_when=FIRST_COMPLETED
    )
    if progress_cb is not None:
        live_status = _read_live_parts_status(files, progress_dir, part_totals)
        progress_cb(
            _aggregate_event(live_status, len(files), start, partial_running=True)
        )
    return done, still_pending


def _read_live_parts_status(
    files, progress_dir: Path, part_totals: dict[str, int]
) -> list[dict]:
    """Per-part snapshot. ``frames_total`` always comes from the upfront probe
    in ``part_totals`` (so the denominator is honest from event 1, even when no
    worker has written a JSON snapshot yet)."""
    out = []
    for part in files:
        stem = recording_store.part_stem(part)
        total = part_totals.get(stem, 0)
        snapshot = _detection_worker.read_progress_json(progress_dir / f"{stem}.json")
        if snapshot is None:
            out.append(
                {
                    "id": stem,
                    "file": part.get("file") or recording_store.part_stem(part),
                    "status": "pending",
                    "frames_done": 0,
                    "frames_total": total,
                }
            )
        else:
            out.append(
                {
                    "id": stem,
                    "file": part.get("file") or recording_store.part_stem(part),
                    "status": snapshot.get("status", "running"),
                    "frames_done": snapshot.get("frames_done", 0),
                    "frames_total": total,
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Decisions + helpers
# --------------------------------------------------------------------------- #
def _resolve_device_indices(job: DetectionJob) -> list[int]:
    if job.compute.backend != "cuda":
        return []
    indices = job.compute.device_indices
    if indices is None or len(indices) == 0:
        # device_indices=None means GPU 0 alone (§5): spreading a run over every
        # available GPU when none are named is not built, so the inline single-device
        # path takes it, and CUDA device-count probing stays with the validator.
        return [0]
    return list(indices)


def _is_multi_gpu(job: DetectionJob, device_indices: list[int]) -> bool:
    return job.compute.backend == "cuda" and len(device_indices) > 1


def _aggregate_event(
    parts_status: list[dict],
    parts_total: int,
    start: float,
    *,
    partial_running: bool,
    message: str | None = None,
) -> DetectionProgressEvent:
    parts_done = sum(1 for p in parts_status if p.get("status") == "complete")
    frames_done = sum(int(p.get("frames_done", 0)) for p in parts_status)
    frames_total = sum(int(p.get("frames_total", 0)) for p in parts_status)
    elapsed = time.monotonic() - start
    return DetectionProgressEvent(
        parts_done=parts_done,
        parts_total=parts_total,
        frames_done=frames_done,
        frames_total=frames_total,
        elapsed_s=elapsed,
        eta_s=(
            remaining_seconds(frames_done / frames_total, elapsed)
            if frames_total
            else None
        ),
        message=message,
    )


def _first_mask_resolution(parts_status: list[dict]) -> list[int] | None:
    for r in parts_status:
        mr = r.get("mask_resolution")
        if mr is not None:
            return list(mr)
    return None


def _raise_if_any_failed(parts_status: list[dict]) -> None:
    """Detection treats any failed part (stop, OOM, model exception) as a
    failed run — surfaces to the executor envelope as ``status="failed"``."""
    failed = [r for r in parts_status if r.get("status") == "failed"]
    if failed:
        names = ", ".join(f"{r['id']}({r.get('error')})" for r in failed)
        raise RuntimeError(f"detection failed on {len(failed)} part(s): {names}")


def _validate_source_files(recording_dir, data: dict) -> None:
    """Check the recording has video parts and every part file is present.
    Shared shape with prompt-test validation (§8)."""
    recording_id = Path(recording_dir).name
    if not data.get("video", {}).get("files", []):
        raise ValueError(f"recording {recording_id!r} has no video parts")
    video_integrity.raise_on_video_issues(
        video_integrity.validate_source_videos(recording_dir, data)
    )


@register_processing_module
class DetectionModule(ProcessingModule):
    """Detection stage (§7) as a processing module (§6.5)."""

    module_id = "detection"
    config_model = DetectionConfig
    id_prefix = "det"
    runs_key = "detection_runs"
    upstream = "video"
    job_class = DetectionJob
    progress_event_class = DetectionProgressEvent
    # frames_total is the timestamp map's frames, one per frame read
    progress_counts_frames = True
    resumes = False  # detection has no resume: a re-run starts the recording over
    identity_fields = (
        "model_id",
        "model_variant",
        "weights_source",
        "conf_threshold",
        "every_n_frames",
        "prompt_fingerprint",
        "prompt_classes",
    )
    identity_defaults = {"every_n_frames": 1}
    lineage_ref = None  # roots at the recording's video — no upstream run spec

    def recompute_identity_from_job(self, job) -> dict:
        return {
            "prompt_fingerprint": prompt_fingerprint(job.prompt),
            "prompt_classes": prompt_classes(job.prompt),
        }

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run(job, run_dir, progress_cb, stop_event)

    def run_dir_for(self, experiment_root, job):
        return detection_run_dir(experiment_root, job)

    def output_path(self, run_id: str) -> str:
        return f"detections/{run_id}/"

    def validate(self, recording_dir, recording_data: dict, config: dict) -> dict:
        # The model covers a subset (the scalar conf + typed ids); it enforces the conf
        # bound + shape. The nested prompt / compute and the resolved weights_source stay
        # outside it, so validate keeps its own enrichment and returns the enriched config.
        DetectionConfig.model_validate(config)
        model_id = config["model_id"]
        prompt = config["prompt"]
        compute = config.get("compute")
        # Resolve the id without being told its source (pretrained / experiment-trained
        # / machine-imported); constructing also validates the id + record. Record
        # the discovered source so build_job + the worker re-resolve the same model.
        model, weights_source = resolve_detection_model_auto(
            Path(recording_dir).parents[1], model_id, config.get("model_variant")
        )
        config["weights_source"] = weights_source
        # The *resolved* variant, beside the resolved weights source. Left unset it resolves to
        # `variant or DEFAULT_VARIANT or next(iter(VARIANTS))` (protocols.py) — dict insertion
        # order — and `model_variant` is a run **identity** field, so two runs recorded as
        # `None` would compare as the same spec across a code change that reordered VARIANTS.
        config["model_variant"] = model.variant
        if compute is not None and compute.backend not in model.supported_backends:
            raise ValueError(
                f"model {model_id!r} does not support the {compute.backend!r} backend; "
                f"it requires one of {list(model.supported_backends)} "
                "(a CUDA-only model like SAM3 needs an NVIDIA GPU)."
            )
        if compute is not None:
            model.check_backend(compute.backend)  # this *install* on this backend (§5)
        batch_size = config.get("batch_size")
        if batch_size is None:  # unset -> the model-appropriate default
            config["batch_size"] = (
                BATCHABLE_DEFAULT_BATCH_SIZE if model.supports_batching else 1
            )
        elif batch_size > 1 and not model.supports_batching:
            raise ValueError(
                f"model {model_id!r} does not support batched inference; "
                "set batch size to 1"
            )
        if compute is not None and compute.backend == "cuda" and compute.device_indices:
            check_cuda_indices(compute.device_indices)
        supported = list(model.supported_cue_types)
        if not supported:  # closed-set: detects its trained classes, takes no prompt
            if prompt.targets:
                raise ValueError(
                    f"model {model_id!r} is closed-set and takes no prompt; it "
                    "detects its trained classes — submit an empty prompt"
                )
        elif not prompt.targets:
            raise ValueError(f"model {model_id!r} requires at least one prompt target")
        else:
            unsupported = {
                type(t.cue).__name__
                for t in prompt.targets
                if type(t.cue) not in supported
            }
            if unsupported:
                raise ValueError(
                    f"model {model_id!r} does not support cue types "
                    f"{sorted(unsupported)}; "
                    f"supported: {[c.__name__ for c in supported]}"
                )
        status = model.readiness()
        if not status.ready:
            raise ModelNotReadyError(status)
        _validate_source_files(recording_dir, recording_data)
        for run_entry in recording_data.get("detection_runs", []):
            if run_entry.get("status") in ACTIVE_STATUSES:
                raise ValueError(
                    f"detection run {run_entry['id']!r} is already active on this recording"
                )
        return config

    def build_job(self, recording_id: str, run_id: str, config: dict) -> DetectionJob:
        return DetectionJob(
            job_id=run_id,
            recording_id=recording_id,
            every_n_frames=int(config.get("every_n_frames") or 1),
            model_id=config["model_id"],
            model_variant=config.get("model_variant"),
            weights_source=config.get("weights_source", "pretrained"),
            prompt=config["prompt"],
            conf_threshold=config.get("conf_threshold", 0.5),
            batch_size=config.get("batch_size") or 1,
            compute=config["compute"],
            created_at=utc_timestamp(),
        )

    def extra_record_fields(self, config: dict) -> dict:
        return {
            "model_id": config["model_id"],
            "model_variant": config.get("model_variant"),
            "weights_source": config.get("weights_source", "pretrained"),
            "conf_threshold": config.get("conf_threshold", 0.5),
            "prompt_fingerprint": prompt_fingerprint(config.get("prompt")),
            "prompt_classes": prompt_classes(config.get("prompt")),
            "batch_size": config.get("batch_size") or 1,
            "every_n_frames": int(config.get("every_n_frames") or 1),
        }
