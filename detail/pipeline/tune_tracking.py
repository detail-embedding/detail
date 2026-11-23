"""Tune a detector's detection and tracking settings on a sample of its recordings (§12 *Tune
detection and tracking settings*).

One job, three phases, in ``models/<model_id>/tuning/<tuning_id>/``:

1. **Detect** — read a few windows of each recording (``configs.tuning.SAMPLE_WINDOWS`` spread
   across its timestamp map, ``sample_seconds`` in all) and run the model over them once, at
   the lowest confidence any setting needs. Each part is walked **sequentially** from its first
   frame (Tenet 11): frames outside a window are grabbed and not decoded into the batch, and the
   walk stops after the last window. Each window is written as one part of a scratch detection
   run in the normal schema (``DetectionPartWriter``), so the tracker reads it as it reads any.
2. **Sweep** — run the real ``anchor_tracker`` once per setting of
   ``analysis.track_quality.grid()`` over those detections. A threshold above the detection
   confidence keeps exactly what detecting at that threshold would have kept, so one detection
   pass serves every setting.
3. **Measure and pick** — ``analysis.track_quality`` turns each setting's assignments into the
   label-free numbers and picks one. ``tuning.yaml`` gets the whole table; the scratch
   detections and assignments are removed.

No label is read anywhere: the method sees the model, the recordings' video and the grid. The
arena (field of view) is not applied — the sample measures the model, not one recording's
restriction — and the docs say so.
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from detail.analysis import track_quality
from detail.configs.detection import BATCHABLE_DEFAULT_BATCH_SIZE, DetectionConfig
from detail.configs.tuning import SAMPLE_WINDOWS
from detail.core import atomic
from detail.core.dataclasses import (
    ComputeTarget,
    JobStatus,
    Prompt,
    TrackingJob,
    register_job_type,
    register_progress_event_type,
)
from detail.core.progress import PhaseClock, ThrottledProgress
from detail.core.utils import resolve_device, utc_timestamp
from detail.executor.runners import register_runner
from detail.models.resolve import resolve_detection_model
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import tunings as tunings_store
from detail.storage.zarr_io import DetectionPartReader, DetectionPartWriter
from detail.storage.zarr_io import PartZarrReader
from detail.trackers.anchor_tracker import AnchorTracker

JOB_TYPE = "tune_tracking"
TRACKER_ID = "anchor_tracker"
_SCRATCH = "sample"
_NO_BBOX = (0.0, 0.0, 0.0, 0.0)


@dataclass
class TuneTrackingJob:
    job_id: str  # the tuning id, ``tune_NNN``
    model_id: str
    weights_source: str
    model_variant: str | None
    #: The recordings sampled. Named as labeling's is, because that is the field a remote
    #: run's shepherd reads to send a job's videos to the cluster (§6).
    source_recording_ids: list[str]
    sample_seconds: float
    compute: ComputeTarget
    created_at: str
    #: Who asked: ``{"kind": "distill", "distill_id": ...}`` or ``{"kind": "standalone"}``.
    source: dict = field(default_factory=lambda: {"kind": "standalone"})

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "weights_source": self.weights_source,
            "model_variant": self.model_variant,
            "source_recording_ids": list(self.source_recording_ids),
            "sample_seconds": self.sample_seconds,
            "compute": self.compute.to_dict(),
            "created_at": self.created_at,
            "source": dict(self.source),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TuneTrackingJob":
        return cls(
            job_id=d["job_id"],
            model_id=d["model_id"],
            weights_source=d["weights_source"],
            model_variant=d.get("model_variant"),
            source_recording_ids=list(d["source_recording_ids"]),
            sample_seconds=float(d["sample_seconds"]),
            compute=ComputeTarget.from_dict(d["compute"]),
            created_at=d["created_at"],
            source=dict(d.get("source") or {"kind": "standalone"}),
        )


@dataclass
class TuneProgressEvent:
    """``phase`` is ``"detecting"`` or ``"sweeping"``. ``frames_done``/``frames_total`` are the
    standard bar pair every surface already renders: frames while detecting, then
    (setting, recording) pairs while sweeping — a new total, so a bar restarts its clock.
    ``eta_s`` is the running phase's own (``PhaseClock``)."""

    phase: str
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


register_job_type(JOB_TYPE, TuneTrackingJob)
register_progress_event_type(JOB_TYPE, TuneProgressEvent)


@dataclass(frozen=True)
class Window:
    """Frames ``[start, stop)`` of part ``part_index`` (its play index)."""

    part_index: int
    start: int
    stop: int

    @property
    def n_frames(self) -> int:
        return self.stop - self.start


def frame_rate(parts_t_ns: list[np.ndarray]) -> float:
    """Frames per second from the timestamp maps' median step (0 when there is none)."""
    steps = [np.diff(np.asarray(t, dtype=np.int64)) for t in parts_t_ns if len(t) > 1]
    steps = [s[s > 0] for s in steps]
    pooled = np.concatenate(steps) if steps else np.zeros(0)
    return float(1e9 / np.median(pooled)) if len(pooled) else 0.0


def plan_windows(
    parts_t_ns: list[np.ndarray], sample_seconds: float, n_windows: int = SAMPLE_WINDOWS
) -> list[Window]:
    """``n_windows`` windows of ``sample_seconds / n_windows`` each, centred at even steps
    through the recording's frames and clipped to the part they fall in, in play order. A
    recording no longer than the sample is taken whole, one window per part."""
    counts = [len(t) for t in parts_t_ns]
    total = sum(counts)
    if total == 0:
        return []
    rate = frame_rate(parts_t_ns) or 1.0
    rows = max(1, int(round(sample_seconds / n_windows * rate)))
    if rows * n_windows >= total:
        return [Window(i, 0, n) for i, n in enumerate(counts) if n]
    starts = np.cumsum([0] + counts[:-1])
    windows: list[Window] = []
    for k in range(n_windows):
        centre = int((k + 0.5) * total / n_windows)
        part = int(np.searchsorted(starts, centre, side="right") - 1)
        local = centre - int(starts[part])
        lo = max(0, local - rows // 2)
        hi = min(counts[part], lo + rows)
        lo = max(0, hi - rows)
        windows.append(Window(part, lo, hi))
    windows.sort(key=lambda w: (w.part_index, w.start))
    merged: list[Window] = []
    for w in windows:  # two windows of a short part may overlap: join them
        if (
            merged
            and merged[-1].part_index == w.part_index
            and w.start < merged[-1].stop
        ):
            merged[-1] = Window(
                w.part_index, merged[-1].start, max(merged[-1].stop, w.stop)
            )
        else:
            merged.append(w)
    return merged


def run_dir_for(experiment_root, job: TuneTrackingJob) -> Path:
    return tunings_store.tuning_dir(experiment_root, job.model_id, job.job_id)


@register_runner(JOB_TYPE, run_dir_for=run_dir_for, cancel_status=JobStatus.CANCELLED)
def run(job: TuneTrackingJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    record = _initial_record(job)
    tunings_store.write_tuning(run_dir, record)
    try:
        finished = _tune(job, run_dir, record, progress_cb, stop_event)
    except Exception as exc:
        # tuning.yaml is what every surface reads; a crash must not leave it "running".
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        tunings_store.write_tuning(run_dir, record)
        raise
    finally:
        shutil.rmtree(run_dir / _SCRATCH, ignore_errors=True)
    if not finished:
        record["status"] = "cancelled"
        tunings_store.write_tuning(run_dir, record)


def _initial_record(job: TuneTrackingJob) -> dict:
    return {
        "id": job.job_id,
        "model_id": job.model_id,
        "status": "running",
        "created_at": job.created_at,
        "source": dict(job.source),
        "config": {
            "sample_seconds": job.sample_seconds,
            "windows": SAMPLE_WINDOWS,
            "recording_ids": list(job.source_recording_ids),
            "tracker_id": TRACKER_ID,
            "detection_conf": track_quality.sweep_detection_conf(),
        },
    }


def default_setting() -> track_quality.Setting:
    """What a user gets without tuning: the detection form's default confidence, then the
    tracker's default hyperparameters, as the stored detections allow them."""
    schema = AnchorTracker.HYPERPARAMS_SCHEMA
    return track_quality.effective_setting(
        DetectionConfig.model_fields["conf_threshold"].default,
        schema["conf_threshold"][1],
        schema["secondary_conf"][1],
        schema["max_secondary"][1],
    )


@dataclass
class _Sampled:
    """One recording's sample: its windows, their scratch parts (one per window, in order)
    and each part's columns with the rows' mask areas, read once for every setting."""

    recording_id: str
    name: str
    rate: float
    windows: list[Window]
    det_dir: Path
    parts: list[Path]
    columns: list[dict]
    areas: list[np.ndarray]


def _tune(job, run_dir, record, progress_cb, stop_event) -> bool:
    root = run_dir.parents[3]  # <root>/models/<model>/tuning/<id>
    emit = ThrottledProgress(progress_cb)
    start = time.monotonic()
    clock = PhaseClock(start)
    stopped = lambda: stop_event is not None and stop_event.is_set()  # noqa: E731

    def event(phase, done, total, message=None):
        now = time.monotonic()
        return TuneProgressEvent(
            phase=phase,
            frames_done=done,
            frames_total=total,
            elapsed_s=now - start,
            # Each phase is timed on its own clock: the sweep is not paced like detection.
            eta_s=clock.eta(phase, done / total if total else 0.0, now),
            message=message,
        )

    plans = []
    for recording_id in job.source_recording_ids:
        rec_dir = experiment_store.recording_path(root, recording_id)
        data = recording_store.read_recording(rec_dir)
        files = recording_store.video_files(data)
        maps = [
            recording_store.load_active_timestamps(rec_dir, data, i).values
            for i in range(len(files))
        ]
        plans.append(
            (
                recording_id,
                data.get("name") or recording_id,
                rec_dir,
                data,
                files,
                maps,
                plan_windows(maps, job.sample_seconds),
            )
        )
    total_frames = sum(w.n_frames for p in plans for w in p[6])
    emit(event("detecting", 0, total_frames, "loading model"), force=True)
    model = resolve_detection_model(
        root, job.model_id, job.weights_source, job.model_variant
    )
    model.load(resolve_device(job.compute, model.supported_backends))
    conf = track_quality.sweep_detection_conf()
    model.set_conf_threshold(conf)
    batch_size = BATCHABLE_DEFAULT_BATCH_SIZE if model.supports_batching else 1

    sampled: list[_Sampled] = []
    done = 0
    multi: dict[str, float] = {}
    for recording_id, name, rec_dir, data, files, maps, windows in plans:
        det_dir = run_dir / _SCRATCH / recording_id / "detections" / _SCRATCH
        det_dir.mkdir(parents=True, exist_ok=True)
        resolution = None
        for part_index, part in enumerate(files):
            mine = [w for w in windows if w.part_index == part_index]
            if not mine:
                continue
            path = recording_store.resolve_part_video_path(rec_dir, part)
            for k, frames_seen, res in _detect_part(
                model,
                path,
                mine,
                maps[part_index],
                conf,
                batch_size,
                det_dir,
                windows,
                stopped,
            ):
                resolution = resolution or res
                done += frames_seen
                emit(
                    event(
                        "detecting",
                        done,
                        total_frames,
                        f"detecting {name}, window {k + 1}/{len(windows)}",
                    )
                )
            if stopped():
                return False
        _write_envelope(det_dir, resolution)
        parts = [_part_path(det_dir, k) for k in range(len(windows))]
        columns = [DetectionPartReader(p).read() for p in parts]
        size = (resolution or [0, 0])[:2]
        multi[recording_id] = _multi_share(columns)
        sampled.append(
            _Sampled(
                recording_id,
                name,
                frame_rate(maps),
                windows,
                det_dir,
                parts,
                columns,
                [track_quality.rle_areas(c["mask_rle"], size) for c in columns],
            )
        )
    emit(event("detecting", done, total_frames, "detected the sample"), force=True)

    settings = track_quality.grid()
    total_pairs = len(settings) * len(sampled)
    results: dict = {s: {} for s in settings}
    pair = 0
    emit(event("sweeping", 0, total_pairs, "sweeping settings"), force=True)
    for i, setting in enumerate(settings):
        for s in sampled:
            if stopped():
                return False
            results[setting][s.recording_id] = _measure(s, setting, i, run_dir)
            pair += 1
            emit(
                event(
                    "sweeping",
                    pair,
                    total_pairs,
                    f"setting {i + 1}/{len(settings)}: {setting.label()} on {s.name}",
                )
            )

    looks_multi = [
        s.name
        for s in sampled
        if multi[s.recording_id] > track_quality.MULTI_SUBJECT_SHARE
    ]
    default = default_setting()
    chosen = track_quality.pick(results, default, allow_merging=not looks_multi)
    warnings = list(chosen.warnings)
    if looks_multi:
        warnings.append(
            "two confident detections far apart on most frames of "
            + ", ".join(looks_multi)
            + ": these look like more than one subject, which the tracker does not "
            "follow. Only settings that merge nothing were considered, so two subjects are "
            "never joined into one"
        )
    record.update(
        status="complete",
        finished_at=utc_timestamp(),
        recordings=[
            {
                "recording_id": s.recording_id,
                "name": s.name,
                "frame_rate": round(s.rate, 3),
                "sampled_frames": sum(w.n_frames for w in s.windows),
                "windows": [[w.part_index, w.start, w.stop] for w in s.windows],
                "multi_subject_share": round(multi[s.recording_id], 4),
            }
            for s in sampled
        ],
        default=default.to_dict(),
        table=[
            {
                **setting.to_dict(),
                "is_default": setting == default,
                "mean_jumpy_pct": round(track_quality.mean_jumpy(per), 3),
                "per_recording": {r: q.to_dict() for r, q in per.items()},
            }
            for setting, per in results.items()
        ],
        pick={
            "setting": chosen.setting.to_dict() if chosen.setting else None,
            "keep_defaults": chosen.keep_defaults,
            "reason": chosen.reason,
        },
        warnings=warnings,
    )
    tunings_store.write_tuning(run_dir, record)
    emit(event("sweeping", total_pairs, total_pairs, "Done"), force=True)
    return True


def _detect_part(
    model, path, windows, t_ns, conf, batch_size, det_dir, all_windows, stopped
):
    """Walk one part from its first frame, detecting inside ``windows`` only. Yields
    ``(window index, frames detected since last yield, mask resolution)`` per batch."""
    cap = cv2.VideoCapture(str(path))
    prompt = Prompt(targets=[])
    try:
        frame_idx = 0
        for window in windows:
            k = all_windows.index(window)
            out = _part_path(det_dir, k)
            with DetectionPartWriter(out) as writer:
                while frame_idx < window.start:
                    if not cap.grab():
                        raise ValueError(
                            f"{path.name}: the video ended at frame {frame_idx}, before "
                            f"the {len(t_ns)} frames its timestamp map lists"
                        )
                    frame_idx += 1
                batch, meta = [], []
                while frame_idx < window.stop:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(
                            f"{path.name}: the video ended at frame {frame_idx}, before "
                            f"the {len(t_ns)} frames its timestamp map lists"
                        )
                    batch.append(frame)
                    meta.append((frame_idx - window.start, int(t_ns[frame_idx])))
                    frame_idx += 1
                    if len(batch) >= batch_size or frame_idx == window.stop:
                        _write_batch(model, batch, meta, prompt, conf, writer)
                        yield k, len(batch), [int(frame.shape[0]), int(frame.shape[1])]
                        batch, meta = [], []
                        if stopped():
                            return
    finally:
        cap.release()


def _write_batch(model, frames, meta, prompt, conf, writer) -> None:
    for (local_idx, t), result in zip(
        meta, model.predict_batch(frames, [prompt] * len(frames))
    ):
        detection_idx = 0
        for det in result.detections():
            if det.confidence < conf:
                continue
            writer.add_row(
                local_idx,
                t,
                detection_idx,
                det.class_name,
                float(det.confidence),
                tuple(det.bbox) if det.bbox is not None else _NO_BBOX,
                _encode(det.mask),
            )
            detection_idx += 1


def _encode(mask) -> str:
    if mask is None:
        return ""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def _write_envelope(det_dir: Path, resolution) -> None:
    """The one envelope key the tracker reads from a detection run (its mask size)."""
    path = det_dir / "job.json"
    tmp = det_dir / "job.json.tmp"
    tmp.write_text(
        json.dumps({"schema_version": 1, "mask_resolution": resolution or [0, 0]}),
        encoding="utf-8",
    )
    atomic.replace(tmp, path)


def _part_path(det_dir: Path, window_index: int) -> Path:
    return det_dir / f"part_{window_index + 1:03d}.zarr"


def _multi_share(columns: list[dict]) -> float:
    shares, weights = [], []
    for cols in columns:
        if not len(cols["frame_idx"]):
            continue
        shares.append(
            track_quality.multi_subject_share(
                cols["frame_idx"], cols["confidence"], cols["bbox"]
            )
        )
        weights.append(len(np.unique(cols["frame_idx"])))
    return float(np.average(shares, weights=weights)) if shares else 0.0


def _measure(s: _Sampled, setting, index: int, run_dir: Path) -> track_quality.Quality:
    out = run_dir / _SCRATCH / s.recording_id / "tracking" / f"s{index:02d}"
    job = TrackingJob(
        job_id=f"s{index:02d}",
        recording_id=s.recording_id,
        track_id=f"s{index:02d}",
        detection_run_id=_SCRATCH,
        tracker_id=TRACKER_ID,
        hyperparams=setting.hyperparams(),
        arena_ids=[],
        created_at=utc_timestamp(),
    )
    offsets = [int(v) for v in np.cumsum([0] + [w.n_frames for w in s.windows[:-1]])]
    try:
        AnchorTracker().track(
            job, s.det_dir, out, arena_polygons=None, part_frame_offsets=offsets
        )
    except ValueError:
        pass  # no subject on any frame at this setting: its assignments are all -1
    size = tuple(json_mask_resolution(s.det_dir))
    tables = []
    for window, part, cols, areas in zip(s.windows, s.parts, s.columns, s.areas):
        assigned = out / "assignments" / part.name
        object_id = (
            PartZarrReader(assigned).read()["object_id"]
            if assigned.exists()
            else np.full(len(cols["frame_idx"]), -1)
        )
        tables.append(
            track_quality.frame_table(
                window.n_frames,
                cols["frame_idx"],
                cols["confidence"],
                cols["mask_rle"],
                object_id,
                size,
                areas=areas,
            )
        )
    shutil.rmtree(out, ignore_errors=True)
    return track_quality.quality(tables, s.rate)


def json_mask_resolution(det_dir: Path) -> list[int]:
    return json.loads(atomic.read_text(det_dir / "job.json"))["mask_resolution"]
