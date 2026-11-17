"""``keypoint_detect`` (§26): place a skeleton's points on the animals in every frame.

The first of the two keypoint stages. It reads a **detection run** — each detection's box, and
its mask for a method that reads them — hands each frame to a ``KeypointModel`` (§26
"KeypointModel protocol"), and writes the raw result once, **one row per pose instance, no
subject**: ``bbox``, ``confidence``, ``detection_row``, ``points``, the method's own
``point_scores``, and its top ``candidates`` so the track stage can choose among them over
time. Identity is not this stage's: ``keypoint_track`` assigns instances to a tracking run's
subjects. Nothing here is ever rewritten.

Per part, **one** forward-sequential ``cv2.VideoCapture`` walk (Tenet 11). A per-detection
method visits the frames that have detections at the run's cadence and is given every
detection on the frame; a whole-frame method visits every frame at the cadence. Frames come
from the detection part's rows or the timestamp map, never ``CAP_PROP_FRAME_COUNT``. Frames are
batched in a bounded buffer of ``batch_size``; the writer flushes only on frame boundaries.
Progress names each phase and forces an emit at every boundary (§6): ``counting frames`` →
``loading model`` → ``placing keypoints``.

Resume (§26): per part, after the last frame written — the store's columns are cut to their
common length, which is a flush and therefore a frame boundary, and the walk reads forward past
it. A stop keeps what was written and records coverage; the executor's status is the only
status.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path

import cv2
import numpy as np

from detail.configs.keypoint_detect import (
    DEFAULT_MIN_POSE_CONFIDENCE,
    KeypointDetectConfig,
)
from detail.core import lookahead
from detail.core.examples import ExampleFrame, ExampleSet
from detail.core.keypoints import (
    InstanceFrame,
    KeypointDetectJob,
    KeypointProgressEvent,
)
from detail.core.progress import remaining_seconds
from detail.core.protocols import KeypointModel
from detail.core.protocols import ModelNotReadyError
from detail.core.registries import KEYPOINT_MODELS
from detail.core.skeleton import Skeleton
from detail.core.utils import (
    check_cuda_indices,
    decode_mask,
    resolve_device,
    resolve_downstream_every_n,
    utc_timestamp,
)
from detail.core.visualize import bgr_to_rgb
from detail.executor import rundir
from detail.executor.runners import ProcessingModule, register_processing_module
from detail.models.resolve import resolve_keypoint_model
from detail.storage import example_sets
from detail.storage import experiment as experiment_store
from detail.storage import keypoint_runs
from detail.storage import recording as recording_store
from detail.storage import video_integrity
from detail.storage.keypoint_zarr import (
    KeypointDetectPartReader,
    KeypointDetectPartWriter,
    extras_spec,
)
from detail.storage.zarr_io import DetectionPartReader

#: The method's settings a config may pass through to the model's constructor.
#: A patch method's own settings, resolved per run and recorded (detect and distill alike).
METHOD_SETTINGS = ("decode", "side_rule", "crop_px")


def keypoint_detect_run_dir(experiment_root, job: KeypointDetectJob) -> Path:
    """``recordings/<id>/keypoint_detect/<kd_NN>/``."""
    return keypoint_runs.detect_run_dir(
        experiment_store.recording_path(experiment_root, job.recording_id),
        job.detect_run_id,
    )


# --------------------------------------------------------------------------- #
# The frames a part hands the model
# --------------------------------------------------------------------------- #
def part_frames(
    det: dict | None,
    timestamps: np.ndarray,
    every_n_frames: int,
    *,
    per_detection: bool,
) -> list[tuple[int, int, list[int]]]:
    """``(frame_idx, t_ns, detection rows)`` for every frame the method visits in a part, in
    frame order.

    A per-detection method visits the frames with at least one detection at the cadence, each
    with its rows (global row indices within the detection part); a whole-frame method visits
    every frame at the cadence of the timestamp map, with no rows. ``det`` holds the detection
    part's ``frame_idx`` and ``t_ns``; None when the part has no detection store.
    """
    if not per_detection:
        return [
            (int(f), int(timestamps[f]), [])
            for f in range(0, len(timestamps), every_n_frames)
        ]
    if det is None:
        return []
    frame_col = np.asarray(det["frame_idx"])
    t_col = np.asarray(det["t_ns"])
    by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(frame_col.tolist()):
        if frame % every_n_frames == 0:
            by_frame.setdefault(int(frame), []).append(row)
    return [
        (frame, int(t_col[rows[0]]), rows) for frame, rows in sorted(by_frame.items())
    ]


def iter_instance_frames(
    video_path: Path | str,
    frames: list[tuple[int, int, list[int]]],
    bbox: np.ndarray | None,
    mask_rle: np.ndarray | None,
    mask_size: tuple[int, int] | None,
):
    """Yield one ``InstanceFrame`` per entry of ``frames`` from one forward walk of the part's
    video (Tenet 11). Each detection row's mask is decoded only when ``mask_rle`` is given
    (the method reads masks and the detection run kept them)."""
    if not frames:
        return
    cap = cv2.VideoCapture(str(video_path))
    try:
        cur = 0
        for target, t_ns, rows in frames:
            while cur < target:
                ok = cap.grab()
                if not ok:
                    raise RuntimeError(
                        f"video {video_path} ended at frame {cur} before reaching frame "
                        f"{target}"
                    )
                cur += 1
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {target} from {video_path}")
            cur += 1
            masks = None
            if mask_rle is not None and mask_size is not None:
                masks = {r: decode_mask(str(mask_rle[r]), mask_size) for r in rows}
            yield InstanceFrame(
                frame_idx=int(target),
                t_ns=int(t_ns),
                frame=bgr_to_rgb(bgr),
                boxes={r: tuple(float(v) for v in bbox[r]) for r in rows},
                masks=masks,
            )
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def snapshot_examples(experiment_dir: Path, examples: ExampleSet) -> dict:
    """A run's own copy of an example set (Tenet 9): ``id``, ``name``, ``digest``, the frame
    ``images`` and every frame's ``marks``. The images are write-once and named by their
    content (``example_sets.add_frame_image``), so the names pin the pictures."""
    hashes = example_sets.image_hashes(experiment_dir, examples)
    return {
        "id": examples.id,
        "name": examples.name,
        "digest": examples.digest(hashes),
        "frames": [f.image for f in examples.frames],
        "marks": [f.to_dict() for f in examples.frames],
    }


def examples_from_snapshot(
    example_set: dict, skeleton: dict, skeleton_ref: dict | None = None
) -> ExampleSet:
    """The :class:`ExampleSet` a run's snapshot describes, on the skeleton the run recorded."""
    return ExampleSet(
        id=example_set["id"],
        name=example_set.get("name", example_set["id"]),
        purpose="example",
        created_at="",
        skeleton=Skeleton.from_dict(skeleton),
        frames=[ExampleFrame.from_dict(f) for f in example_set["marks"]],
        skeleton_ref=None if skeleton_ref is None else dict(skeleton_ref),
    )


def _launch_snapshot(config: dict) -> ExampleSet | None:
    """The example set this run's launch already copied into ``config``, or None.

    A processing run snapshots the set once, in ``resolve_shared_config``, and stores it in
    the run record's config; the coordinator then validates each recording **from that
    record** when its turn comes. Reading the live set again there would let an edit made
    mid-batch reach the later recordings, so one ``run_NNN`` would hold several settings.
    """
    snapshot = config.get("example_set")
    if not isinstance(snapshot, dict) or config.get("skeleton") is None:
        return None
    if snapshot.get("id") != config.get("example_set_id"):
        return None
    return examples_from_snapshot(
        snapshot, config["skeleton"], config.get("skeleton_ref")
    )


def build_method(
    experiment_dir: Path,
    model_id: str,
    model_variant: str | None,
    settings: dict,
    example_set: dict | None,
    skeleton: dict,
    weights_source: str | None = None,
):
    """A keypoint method built from a run's **own copy** of its inputs (the ``example_set``
    snapshot :func:`resolve_method` wrote, the resolved ``settings``, the recorded
    ``weights_source``) — never from the live example set, which the user may have edited
    since the run was configured, and never by searching for a student again. Shared by the
    detect stage and the distill label stage, so both build the method the same way."""
    kwargs: dict = {name: settings.get(name) for name in METHOD_SETTINGS}
    if example_set is not None:
        examples = examples_from_snapshot(example_set, skeleton)
        kwargs["examples"] = examples
        kwargs["example_images"] = example_sets.load_images(experiment_dir, examples)
    return resolve_keypoint_model(
        model_id,
        model_variant,
        experiment_dir=experiment_dir,
        weights_source=weights_source,
        **kwargs,
    )


def resolve_method(experiment_dir: Path, config: dict, compute) -> dict:
    """Resolve a keypoint method for a run and write what was resolved back into ``config``
    (Tenet 9): the variant, ``weights_source`` (where a student's weights live), the
    skeleton and its digest, the **example-set snapshot** (id,
    name, digest, frames, marks) and ``skeleton_ref``, the backbone, the candidate count, the
    method settings (``decode`` / ``side_rule`` / ``crop_px``) and ``matching``. Reads
    ``model_id``, ``model_variant``, ``example_set_id`` and the settings from ``config``.

    Shared by keypoint detect's ``validate`` and the distill label stage's plan, so a patch
    method is checked (set runnable, weights installed, backend supported) and recorded the
    same way wherever it runs.

    Raises:
        ValueError: An unknown method, a missing or unwanted example set, an unrunnable set,
            an unsupported backend.
        ModelNotReadyError: The method's weights are not installed.
    """
    model_id = config["model_id"]
    # A keypoint student — this experiment's (§12 *Distill keypoints*) or the catalog's (§12
    # *Keypoint models in the catalog*) — is not a registered method; the resolver
    # constructs it from its record, and here only its class-level facts are needed.
    from detail.models.resolve import (
        keypoint_method_class,
        resolve_keypoint_weights_source,
    )

    model_cls = keypoint_method_class(experiment_dir, model_id)
    if model_cls is None:
        raise ValueError(
            f"unknown keypoint method {model_id!r}; available: "
            f"{sorted(KEYPOINT_MODELS)}, this experiment's keypoint students and the "
            "catalog's keypoint models"
        )
    # Resolved and written back (Tenet 9): an experiment student and a catalog model can
    # share an id, so the worker must build the one validated here, not search again.
    config["weights_source"] = resolve_keypoint_weights_source(experiment_dir, model_id)
    example_set_id = config.get("example_set_id")
    build: dict = {name: config.get(name) for name in METHOD_SETTINGS}
    examples = None
    if model_cls.needs_examples:
        if not example_set_id:
            raise ValueError(
                f"method {model_id!r} learns from an example set; choose one "
                "(example_set_id)"
            )
        examples = _launch_snapshot(config)
        if examples is None:
            try:
                examples = example_sets.read_set(experiment_dir, example_set_id)
            except FileNotFoundError as exc:
                raise ValueError(str(exc)) from None
        examples.check_runnable()  # purpose, skeleton, every point marked visible
        build["examples"] = examples
        build["example_images"] = example_sets.load_images(experiment_dir, examples)
    elif example_set_id:
        raise ValueError(
            f"method {model_id!r} takes no example set; leave example_set_id unset"
        )
    model = resolve_keypoint_model(
        model_id,
        config.get("model_variant"),
        experiment_dir=experiment_dir,
        weights_source=config["weights_source"],
        **build,
    )
    config["model_variant"] = model.variant
    status = model.readiness()
    if not status.ready:
        raise ModelNotReadyError(status)
    if compute.backend not in model.supported_backends:
        raise ValueError(
            f"method {model_id!r} does not support backend {compute.backend!r}; "
            f"supports {model.supported_backends}"
        )
    model.check_backend(compute.backend)
    if compute.backend == "cuda" and compute.device_indices:
        check_cuda_indices(compute.device_indices)

    skeleton = model.skeleton()
    skeleton.validate()
    config["skeleton"] = skeleton.to_dict()
    config["skeleton_digest"] = skeleton.digest()
    if examples is not None:
        config["example_set"] = snapshot_examples(experiment_dir, examples)
        config["example_set_digest"] = config["example_set"]["digest"]
        config["skeleton_ref"] = (
            None if examples.skeleton_ref is None else dict(examples.skeleton_ref)
        )
    else:
        config["example_set"] = None
        config["example_set_digest"] = None
        config["skeleton_ref"] = None
    shares = getattr(model_cls, "shares_weights_with", None)
    config["backbone"] = {"id": shares, "variant": model.weight} if shares else None
    config["backbone_id"] = shares
    config["backbone_variant"] = model.weight if shares else None
    config["n_candidates"] = int(model_cls.n_candidates)
    settings = getattr(model, "settings", {}) or {}
    for name in METHOD_SETTINGS:
        config[name] = settings.get(name, config.get(name))
    matching = getattr(model, "matching", None)
    config["matching"] = dict(matching) if matching else None
    return config


def _build_model(job: KeypointDetectJob, experiment_dir: Path):
    return build_method(
        experiment_dir,
        job.model_id,
        job.model_variant,
        {name: getattr(job, name) for name in METHOD_SETTINGS},
        job.example_set,
        job.skeleton,
        job.weights_source,
    )


def resolve_min_pose_confidence(model_cls, model_id: str, value: float | None):
    """``min_pose_confidence`` for a method (§26 *Visibility*): the default for a method that
    ``reports_confidence``, None for one that does not — refused when set there, since a
    threshold the method cannot apply would be recorded as identity and change nothing.

    Raises:
        ValueError: A threshold set for a method that reports no confidence.
    """
    if getattr(model_cls, "reports_confidence", False):
        return float(DEFAULT_MIN_POSE_CONFIDENCE if value is None else value)
    if value is not None:
        raise ValueError(
            f"method {model_id!r} reports no pose confidence; leave min_pose_confidence "
            "unset (it applies only to a method that does, such as a trained pose model)"
        )
    return None


class _Progress:
    """One place the stage's events are shaped (§6): frames across the whole run, the ETA
    from the work done since this process started, and the phase message."""

    def __init__(
        self, progress_cb, parts_total: int, frames_total: int, done_at_start: int
    ):
        self._cb = progress_cb
        self.parts_total = parts_total
        self.frames_total = frames_total
        self.frames_done = done_at_start
        self._done_at_start = done_at_start
        self._start = time.monotonic()
        self.parts_done = 0

    def emit(self, message: str | None = None) -> None:
        if self._cb is None:
            return
        elapsed = time.monotonic() - self._start
        own_total = max(self.frames_total - self._done_at_start, 0)
        own_fraction = (
            (self.frames_done - self._done_at_start) / own_total if own_total else 1.0
        )
        self._cb(
            KeypointProgressEvent(
                track_id=None,
                parts_done=self.parts_done,
                parts_total=self.parts_total,
                frames_done=self.frames_done,
                frames_total=self.frames_total,
                elapsed_s=elapsed,
                eta_s=(
                    0.0
                    if not own_total or own_fraction >= 1.0
                    else remaining_seconds(own_fraction, elapsed)
                ),
                message=message,
            )
        )


def _read_det(path: Path, *, with_masks: bool) -> dict | None:
    """The detection part's columns the walk needs; ``mask_rle`` only when masks are read."""
    if not path.is_dir():
        return None
    reader = DetectionPartReader(path)
    names = ["frame_idx", "t_ns", "bbox"] + (["mask_rle"] if with_masks else [])
    return {name: np.asarray(reader.column(name)[:]) for name in names}


def _part_coverage(path: Path, frames_walked: int) -> dict:
    """``{frames, frames_with_instance, instances}`` for one part, read off its store."""
    frame_idx = np.zeros(0, np.int32)
    if path.is_dir():
        frame_idx = np.asarray(KeypointDetectPartReader(path).column("frame_idx")[:])
    return {
        "frames": int(frames_walked),
        "frames_with_instance": int(len(np.unique(frame_idx))),
        "instances": int(len(frame_idx)),
    }


def run(job: KeypointDetectJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    recording_dir = run_dir.parent.parent
    experiment_dir = recording_dir.parent.parent
    recording_data = recording_store.read_recording(recording_dir)
    detection_dir = recording_dir / "detections" / job.detection_run_id
    envelope = rundir.read_envelope(detection_dir)
    mask_size = (
        tuple(int(v) for v in envelope["mask_resolution"])
        if envelope.get("mask_resolution")
        else None
    )
    files = recording_store.video_files(recording_data)
    skeleton = Skeleton.from_dict(job.skeleton)
    names = tuple(skeleton.names)
    model = _build_model(job, experiment_dir)
    per_detection = bool(type(model).per_detection)
    needs_mask = per_detection and bool(getattr(model, "needs_mask", False))

    # One event before the count, so a long count is not silence (§6).
    _Progress(progress_cb, len(files), 0, 0).emit("counting frames")
    plan: list[dict] = []
    frames_total = 0
    done_at_start = 0
    for part_index, part in enumerate(files):
        part_id = recording_store.part_stem(part)
        if per_detection:
            det = _read_det(detection_dir / f"{part_id}.zarr", with_masks=False)
            timestamps = np.zeros(0, np.int64)
        else:
            det = None
            timestamps = recording_store.load_active_timestamps(
                recording_dir, recording_data, part_index
            ).values
        frames = part_frames(
            det, timestamps, job.every_n_frames, per_detection=per_detection
        )
        store = keypoint_runs.part_path(run_dir, part_id)
        last = keypoint_runs.last_frame_written(store)
        skip = 0 if last is None else sum(1 for f, _t, _r in frames if f <= last)
        frames_total += len(frames)
        done_at_start += skip
        plan.append(
            {
                "part_index": part_index,
                "part_id": part_id,
                "store": store,
                "det_path": detection_dir / f"{part_id}.zarr",
                "video_path": recording_store.resolve_part_video_path(
                    recording_dir, part
                ),
                "frames": frames,
                "skip": skip,
                "walked": skip,
            }
        )

    keypoint_runs.write_detect_yaml(
        run_dir, keypoint_runs.detect_record(job, skeleton.digest())
    )
    progress = _Progress(progress_cb, len(plan), frames_total, done_at_start)
    progress.emit("loading model")
    model.load(resolve_device(job.compute, model.supported_backends))
    if getattr(model, "reports_confidence", False):
        model.set_min_pose_confidence(float(job.min_pose_confidence))
    keypoint_runs.update_detect_yaml(run_dir, resolved=model.resolved())

    halves = tuple(
        _model_half(model, name)
        for name in ("prepare_batch", "run_prepared", "finish_batch")
    )
    stopped = False
    for part in plan:
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        todo = part["frames"][part["skip"] :]
        det = None
        if per_detection and todo:
            det = _read_det(part["det_path"], with_masks=needs_mask)
        frame_iter = iter_instance_frames(
            part["video_path"],
            todo,
            None if det is None else det["bbox"],
            None if det is None or not needs_mask else det["mask_rle"],
            mask_size,
        )
        stopped = _place_part(
            halves, model, frame_iter, part, names, job, progress, stop_event
        )
        if stopped:
            break
        progress.parts_done = part["part_index"] + 1
        progress.emit("placing keypoints")

    coverage = {
        part["part_id"]: _part_coverage(part["store"], part["walked"]) for part in plan
    }
    elapsed = time.monotonic() - progress._start
    walked_now = progress.frames_done - done_at_start
    keypoint_runs.update_detect_yaml(
        run_dir,
        coverage=coverage,
        timing={
            "frames": int(walked_now),
            "seconds": round(elapsed, 3),
            "frames_per_s": round(walked_now / elapsed, 2) if elapsed > 0 else None,
        },
    )
    progress.emit("stopped" if stopped else "complete")


def _place_part(halves, model, frame_iter, part, names, job, progress, stop_event):
    """Place keypoints on one part's frames; True when the run was stopped.

    **Four stages overlap** (§26 *Execution*): one thread walks the video and decodes masks
    (``frame_iter``), one runs the method's CPU half (``KeypointModel.prepare_batch``) a batch
    or two ahead, this thread runs only the device half, and one writer thread runs the host
    half and appends every instance in frame order. The stop event is checked before each
    forward pass; the batches already computed are written, so the store still ends on a
    frame boundary. The part's store is opened on its first instance (the method's extras fix
    its columns) and closed here, after the writer has joined.
    """
    prepare, run_prepared, finish = halves
    stopped = False
    writer: KeypointDetectPartWriter | None = None  # touched only by the writer thread

    def prepared_batches():
        """Prep stage (its own thread): the method's CPU half, a batch or two ahead."""
        batches = lookahead.background_iter(
            _batches(frame_iter, job.batch_size), depth=2
        )
        try:
            for batch in batches:
                yield prepare(batch), len(batch)
        finally:
            batches.close()  # the decode thread stops with this stage

    def computed_batches(prepared):
        """Inference stage (the calling thread): the device half, nothing else."""
        nonlocal stopped
        for inputs, size in prepared:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                return
            yield run_prepared(inputs), size

    def write_batch(item) -> None:
        """Writer stage (one thread, so instances reach the store in frame order)."""
        nonlocal writer
        raw, size = item
        results = finish(raw)
        if len(results) != size:
            raise RuntimeError(
                f"{type(model).__name__}.predict_batch returned {len(results)} results "
                f"for {size} frames"
            )
        for result in results:
            for row in range(result.n):
                if writer is None:
                    writer = KeypointDetectPartWriter(
                        part["store"], names, job.n_candidates, extras_spec(result)
                    )
                writer.add_instance(result, row)
        part["walked"] += size
        progress.frames_done += size
        progress.emit("placing keypoints")

    prepared = lookahead.background_iter(prepared_batches(), depth=2)
    try:
        written = lookahead.ordered_map(
            write_batch, computed_batches(prepared), workers=1, window=2
        )
        try:
            for _ in written:
                pass
        finally:
            written.close()
    finally:
        prepared.close()
        if writer is not None:
            writer.close()
    return stopped


def _batches(frames, size: int):
    """``frames`` in lists of up to ``size``: a bounded buffer (Tenet 11)."""
    batch: list = []
    for frame in frames:
        batch.append(frame)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _model_half(model, name: str):
    """``model``'s half of ``predict_batch`` (§26), or the protocol's default for a method
    that does not split it, which runs the whole call on the inference thread."""
    half = getattr(model, name, None)
    if half is not None:
        return half
    return functools.partial(getattr(KeypointModel, name), model)


# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #
@register_processing_module
class KeypointDetectModule(ProcessingModule):
    module_id = "keypoint_detect"
    id_prefix = "kd"
    runs_key = "keypoint_detect_runs"
    upstream = "detection"
    # Met after tracking, not beside it: keypoint track needs a tracking run anyway, and one
    # straight line (detection → tracking → keypoint detect → keypoint track) is what a user
    # follows. The API and CLI still allow running it before tracking.
    offered_after = ("tracking",)
    job_class = KeypointDetectJob
    progress_event_class = KeypointProgressEvent
    # frames_total is the frames the model ran on (detected ones, for a per-detection model)
    progress_counts_frames = True
    resumes = True  # resumes per part after the last frame written (§26)
    config_model = KeypointDetectConfig
    identity_fields = (
        "model_id",
        "model_variant",
        "weights_source",
        "backbone_id",
        "backbone_variant",
        "skeleton_digest",
        "example_set_digest",
        "n_candidates",
        "every_n_frames",
        "crop_px",
        "decode",
        "side_rule",
        "matching",
        "min_pose_confidence",
    )
    lineage_ref = ("detection", "detection_run_id")

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run(job, run_dir, progress_cb, stop_event)

    def run_dir_for(self, experiment_root, job):
        return keypoint_detect_run_dir(experiment_root, job)

    def output_path(self, run_id: str) -> str:
        return f"keypoint_detect/{run_id}/"

    def resolve_shared_config(
        self,
        experiment_dir,
        recording_ids: list[str],
        config: dict,
        on_recording=None,
    ) -> dict:
        """Copy the example set into ``config`` **once, at launch** (Tenet 9).

        Every recording of the batch is validated again by the coordinator when its turn
        comes, from the config the run record stores. Without this copy each turn read the
        live set, so a frame marked while the batch ran reached the later recordings: one run
        held two ``example_set_digest`` values and every "each recording's own match" lookup
        downstream found nothing past the first recording. The copy is the existing snapshot
        keys (``example_set``, ``skeleton``, ``skeleton_ref``), which :func:`resolve_method`
        then uses instead of the live set. A set that is missing or not runnable is left for
        ``validate`` to refuse, with its own message.
        """
        set_id = config.get("example_set_id")
        if not set_id:
            return config
        try:
            examples = example_sets.read_set(experiment_dir, set_id)
        except FileNotFoundError:
            return config
        config["example_set"] = snapshot_examples(Path(experiment_dir), examples)
        config["skeleton"] = examples.skeleton.to_dict()
        config["skeleton_ref"] = (
            None if examples.skeleton_ref is None else dict(examples.skeleton_ref)
        )
        return config

    def validate(self, recording_dir, recording_data: dict, config: dict) -> dict:
        """Every pre-run check of §26 "Validation checks", resolving each defaulted input
        and writing it back (Tenet 9). The model covers a subset of the config, so this
        returns the enriched config rather than merging a partial dump."""
        KeypointDetectConfig.model_validate(config)
        recording_dir = Path(recording_dir)
        recording_id = recording_dir.name
        experiment_dir = recording_dir.parent.parent
        detection_run_id = config["detection_run_id"]
        model_id = config["model_id"]
        compute = config["compute"]

        det_entry = {r["id"]: r for r in recording_data.get("detection_runs", [])}.get(
            detection_run_id
        )
        if det_entry is None:
            raise ValueError(
                f"detection run {detection_run_id!r} not found in recording "
                f"{recording_id!r}"
            )
        if det_entry.get("status") != "complete":
            raise ValueError(
                f"detection run {detection_run_id!r} is {det_entry.get('status')!r}, "
                "not complete"
            )
        recording_store.raise_on_legacy_downsampling(det_entry)
        config["every_n_frames"] = resolve_downstream_every_n(
            config.get("every_n_frames"), int(det_entry.get("every_n_frames") or 1)
        )
        video_integrity.raise_on_video_issues(
            video_integrity.validate_source_videos(recording_dir, recording_data)
        )

        resolve_method(experiment_dir, config, compute)
        from detail.models.resolve import keypoint_method_class

        config["min_pose_confidence"] = resolve_min_pose_confidence(
            keypoint_method_class(experiment_dir, model_id),
            model_id,
            config.get("min_pose_confidence"),
        )
        if config.get("batch_size") is None:
            config["batch_size"] = KeypointDetectConfig.model_fields[
                "batch_size"
            ].default
        return config

    def build_job(
        self, recording_id: str, run_id: str, config: dict
    ) -> KeypointDetectJob:
        return KeypointDetectJob(
            job_id=run_id,
            recording_id=recording_id,
            detect_run_id=run_id,
            detection_run_id=config["detection_run_id"],
            min_pose_confidence=config.get("min_pose_confidence"),
            model_id=config["model_id"],
            model_variant=config.get("model_variant"),
            weights_source=config.get("weights_source"),
            backbone=config.get("backbone"),
            example_set=config.get("example_set"),
            skeleton=dict(config["skeleton"]),
            n_candidates=int(config["n_candidates"]),
            every_n_frames=int(config.get("every_n_frames") or 1),
            crop_px=config.get("crop_px"),
            decode=config.get("decode"),
            side_rule=config.get("side_rule"),
            matching=config.get("matching"),
            batch_size=int(config.get("batch_size") or 16),
            compute=config["compute"],
            created_at=utc_timestamp(),
            skeleton_ref=config.get("skeleton_ref"),
        )

    def extra_record_fields(self, config: dict) -> dict:
        return {
            "detection_run_id": config["detection_run_id"],
            "min_pose_confidence": config.get("min_pose_confidence"),
            "model_id": config["model_id"],
            "model_variant": config.get("model_variant"),
            "weights_source": config.get("weights_source"),
            "backbone_id": config.get("backbone_id"),
            "backbone_variant": config.get("backbone_variant"),
            "example_set_id": config.get("example_set_id"),
            "example_set_digest": config.get("example_set_digest"),
            "skeleton_digest": config.get("skeleton_digest"),
            "n_candidates": config.get("n_candidates"),
            "every_n_frames": int(config.get("every_n_frames") or 1),
            "crop_px": config.get("crop_px"),
            "decode": config.get("decode"),
            "side_rule": config.get("side_rule"),
            "matching": config.get("matching"),
        }
