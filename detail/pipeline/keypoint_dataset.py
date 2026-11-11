"""The label stage of a keypoint distill (§12 *Distill keypoints*): sampled frames labelled
by a teacher, written as a YOLO-pose training set.

**Two teachers, both raw** (``KeypointDatasetJob.teacher_kind``; no smoothing over time, the
user's decision 2026-10-09). A finished ``keypoint_detect`` run: its stored instances. Or **the
patch method itself** (``kind: model``, ``dinov3_match`` with an example set) run on the sampled
frames, the way SAM 3 labels frames for the detector distill: built once from the run's own copy
of the example set (``keypoint_detect.build_method``), on the job's compute target, given each
sampled frame's **detections** (boxes and masks) as the detect stage gives them. Either way the
instances are **unassigned** until §26 *Assignment*'s function (``pose_match.assign_part``, the
one keypoint track runs) gives each to at most one subject of the teacher's tracking run, by the
share of its points inside the subject's grown mask; an assigned instance's placement on that
frame alone is the label (revised 2026-10-21). Either way the example set's own clicks join
them: the hand labels the teacher was built from, which are better labels than anything the
method placed. Both are the user's input; neither is an audit label, and **audit frames, and
every frame within ``exclude_audit_s`` of one, are never used**: labels score a method, they
never build one. The dataset records every frame it used so a later audit of the student can
exclude them.

The student sees the **whole frame** (one label line per tracked animal: the tracking box, then
each skeleton point as ``x y v``), so a frame is used only when every detection in it is
assigned to a track (no animal in a training image goes unlabelled) and, for a detect-run
teacher, the detect run placed something on it (a frame it never visited teaches nothing). A
tracked animal no instance was assigned to is still taught with its box and every point
``v = 0``; an instance no subject took is taught to nobody. ``v = 2`` where the teacher placed
the point with a score at or above the resolved cut-off, ``v = 0`` otherwise. ``flip_idx`` comes
from the skeleton's symmetry pairs, so Ultralytics' flip augmentation does not teach left/right
swaps.

Frames are shared across recordings by candidate count, one draw per equal time slice, seeded
by the dataset id, and each is decoded **exactly** (``read_frame_at`` with the part's timestamp
map — the Tenet 11 labeling-sampler exception: a sampled frame's index is a label, not a join
key, so it may seek). Only the drawn frames' instances are read and assigned. Progress reports
every phase (§6).
"""

from __future__ import annotations

import contextlib
import random
import time
import zlib
from pathlib import Path

import numpy as np

from detail.core import atomic
from detail.core.dataclasses import LabelingProgressEvent, register_job_type
from detail.core.keypoints import InstanceFrame, KeypointDatasetJob
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.skeleton import Skeleton
from detail.core.skeleton import flip_index
from detail.core.visualize import bgr_to_rgb, rgb_to_bgr
from detail.executor.runners import register_runner
from detail.pipeline._dataset_common import StagedExamples
from detail.pipeline import pose_match
from detail.pipeline import subject_frames
from detail.storage import datasets as datasets_store
from detail.storage import example_sets
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import KeypointDetectRun
from detail.storage.video import read_frame_at
from detail.core.utils import resolve_device
from detail.storage.zarr_io import DetectionPartReader, PartZarrReader

#: The one class every tracked animal is: a pose student places points, it does not classify.
SUBJECT_CLASS = "subject"
_SOURCE_TEACHER = "teacher"
_SOURCE_EXAMPLE = "example"

register_job_type("keypoint_dataset", KeypointDatasetJob)


def run_dir_for(experiment_root, job: KeypointDatasetJob) -> Path:
    return datasets_store.dataset_dir(experiment_root, job.dataset_id)


@register_runner("keypoint_dataset", run_dir_for=run_dir_for)
def run(job: KeypointDatasetJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    try:
        _label(job, run_dir, progress_cb, stop_event)
    except Exception as exc:
        # The dataset's status is authoritative for every reader; a crash must not leave it
        # stuck at "running" (the labeling stage's rule).
        with contextlib.suppress(FileNotFoundError):
            datasets_store.set_dataset_status(
                run_dir, "failed", error=f"{type(exc).__name__}: {exc}"
            )
        raise


def label_line(
    bbox, points: np.ndarray, scores: np.ndarray, cutoff: float, class_id: int = 0
) -> str:
    """One YOLO-pose line: ``class cx cy w h`` then ``x y v`` per point, all normalized.
    ``v = 2`` where the point is placed with a score at or above ``cutoff``, else ``v = 0``
    with the position zeroed (Ultralytics' convention for an unlabelled point)."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    parts = [
        str(class_id),
        f"{(x0 + x1) / 2:.6f}",
        f"{(y0 + y1) / 2:.6f}",
        f"{x1 - x0:.6f}",
        f"{y1 - y0:.6f}",
    ]
    for (x, y), score in zip(
        np.asarray(points, dtype=float), np.asarray(scores, dtype=float)
    ):
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(score) and score >= cutoff:
            parts.append(f"{min(max(x, 0.0), 1.0):.6f} {min(max(y, 0.0), 1.0):.6f} 2")
        else:
            parts.append("0 0 0")
    return " ".join(parts)


def _audit_times(experiment_root: Path) -> dict[tuple[str, str], np.ndarray]:
    """``(recording_id, part_id) -> sorted t_ns`` of every audit set's frames."""
    out: dict[tuple[str, str], list[int]] = {}
    for es in example_sets.list_sets(experiment_root, purpose="audit"):
        for frame in es.frames:
            out.setdefault((frame.recording_id, frame.part_id), []).append(
                int(frame.t_ns)
            )
    return {key: np.asarray(sorted(v), dtype=np.int64) for key, v in out.items()}


def _near_audit(times: np.ndarray | None, t_ns: int, window_ns: int) -> bool:
    if times is None or not len(times):
        return False
    i = int(np.searchsorted(times, t_ns))
    for j in (i - 1, i):
        if 0 <= j < len(times) and abs(int(times[j]) - int(t_ns)) <= window_ns:
            return True
    return False


def _complete_frames(recording_dir: Path, tracking_run_id: str) -> dict[str, set[int]]:
    """Per part, the frames on which **every** detection is assigned to a track: an
    unassigned detection is an animal the student would see unlabelled."""
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    import yaml

    meta = yaml.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / meta["detection_run_id"]
    out: dict[str, set[int]] = {}
    for det_path in sorted(detection_dir.glob("part_*.zarr")):
        assignment_path = tracking_dir / "assignments" / det_path.name
        if not assignment_path.is_dir():
            continue
        frame_idx = np.asarray(DetectionPartReader(det_path).column("frame_idx")[:])
        object_id = np.asarray(PartZarrReader(assignment_path).read()["object_id"])
        frames = set(int(f) for f in np.unique(frame_idx))
        frames -= set(int(f) for f in np.unique(frame_idx[object_id < 0]))
        out[det_path.stem] = frames
    return out


def _match(teacher: dict) -> dict:
    """The assignment's settings (§26 *Assignment*) the teacher was resolved with; a job
    written before they were recorded takes the defaults they were then."""
    match = dict(teacher.get("match") or {})
    return {
        "min_inside": float(match.get("min_inside", pose_match.DEFAULT_MIN_INSIDE)),
        "mask_grow": float(match.get("mask_grow", pose_match.DEFAULT_MASK_GROW)),
    }


def _tracked_frames(
    recording_dir: Path, tracking_run_id: str
) -> tuple[dict[tuple[str, int], tuple[int, dict]], dict[str, set[int]]]:
    """Every frame a subject of the tracking run is on — ``(part_id, frame_idx) -> (t_ns,
    {track_id: SubjectOnFrame})``, masks decoded only when read — and the complete frames.
    """
    complete = _complete_frames(recording_dir, tracking_run_id)
    frames: dict[tuple[str, int], tuple[int, dict]] = {}
    for part_id in sorted(complete):
        for frame_idx, t_ns, subjects in pose_match.frame_subjects(
            recording_dir, tracking_run_id, part_id, 1
        ):
            frames[(part_id, int(frame_idx))] = (int(t_ns), subjects)
    return frames, complete


def _gather(experiment_root: Path, job: KeypointDatasetJob, progress=None) -> dict:
    """Everything the sampler needs, per recording, read once: the tracked frames (each
    subject's box, its mask decoded only for a frame that is taught), the complete frames,
    the audit times and the example frames the detect run learned from. Neither teacher's
    placements are read here: which frames qualify needs only whether the detect run placed
    anything on them (``frame_idx``), and :func:`_teach` assigns the drawn frames' instances.
    A patch-method teacher has no instances yet: :func:`_gather_model` gathers the frames it
    will label. ``progress(stage, fraction, detail)`` hears each recording as it is read.
    """
    if job.teacher_kind == "model":
        return _gather_model(experiment_root, job, progress)
    audit = _audit_times(experiment_root)
    window_ns = int(round(job.exclude_audit_s * 1e9))
    skeleton: Skeleton | None = None
    sources: dict[str, dict] = {}
    excluded = {"audit": 0, "unassigned": 0, "no_instance": 0}
    runs = job.teacher.get("detect_runs") or {}
    tracking_runs = job.teacher.get("tracking_runs") or {}
    for i, (recording_id, detect_run_id) in enumerate(runs.items()):
        _reading(progress, i, len(runs), recording_id)
        recording_dir = experiment_store.recording_path(experiment_root, recording_id)
        run = KeypointDetectRun(recording_dir, detect_run_id)
        if skeleton is None:
            skeleton = run.skeleton
        elif run.skeleton.digest() != skeleton.digest():
            raise ValueError(
                f"keypoint detect run {detect_run_id!r} of {recording_id!r} has a different "
                "skeleton from the others; a dataset teaches one skeleton"
            )
        tracking_run_id = tracking_runs.get(recording_id)
        if not tracking_run_id:
            raise ValueError(
                f"the teacher names no tracking run for {recording_id!r}: a detect run's "
                "instances belong to no subject until they are assigned to a tracking "
                "run's (§26 *Assignment*); resolve the teacher again"
            )
        frames, complete = _tracked_frames(recording_dir, tracking_run_id)
        placed = {
            part_id: set(np.unique(run.reader(part_id).column("frame_idx")[:]).tolist())
            for part_id in run.part_ids()
        }
        candidates = []
        for (part_id, frame_idx), (t_ns, subjects) in sorted(frames.items()):
            if frame_idx not in complete.get(part_id, set()):
                excluded["unassigned"] += 1
                continue
            if frame_idx not in placed.get(part_id, set()):
                # the detect run placed nothing here (off its cadence, or past where it
                # stopped): a frame it never visited is no teacher's label
                excluded["no_instance"] += 1
                continue
            if _near_audit(audit.get((recording_id, part_id)), t_ns, window_ns):
                excluded["audit"] += 1
                continue
            if frame_idx % job.every_n_frames:
                continue
            candidates.append(
                {
                    "recording_id": recording_id,
                    "part_id": part_id,
                    "frame_idx": frame_idx,
                    "t_ns": t_ns,
                    "subjects": subjects,
                }
            )
        sources[recording_id] = {
            "recording_dir": recording_dir,
            "detect_run_id": detect_run_id,
            "detect_run": run,
            "detect_meta": run.meta,
            "tracking_run_id": tracking_run_id,
            "mask_hw": pose_match.subject_mask_hw(recording_dir, tracking_run_id),
            "example_set": run.meta.get("example_set"),
            "candidates": candidates,
            "boxes": _boxes(frames),
            "complete": complete,
        }
    if skeleton is None:
        raise ValueError("a keypoint dataset needs at least one teacher run")
    return {
        "skeleton": skeleton,
        "sources": sources,
        "excluded": excluded,
        "match": _match(job.teacher),
    }


def _boxes(frames: dict) -> dict[tuple[str, int], dict[int, tuple]]:
    """``(part_id, frame_idx) -> {track_id: bbox}`` — what an example frame is checked
    against (no other animal on it)."""
    return {
        key: {int(tid): subject.bbox for tid, subject in subjects.items()}
        for key, (_t_ns, subjects) in frames.items()
    }


def _reading(progress, i: int, n: int, recording_id: str) -> None:
    if progress is not None:
        progress(
            "reading tracked frames", i / max(n, 1), f"{i + 1}/{n} · {recording_id}"
        )


def _gather_model(
    experiment_root: Path, job: KeypointDatasetJob, progress=None
) -> dict:
    """The patch-method teacher's frames: per recording, every frame on which **every**
    detection is assigned to a track (so no animal in a training image goes unlabelled), minus
    audit frames and off-cadence frames, each with its tracked subjects and its detection
    rows (what the method is given, §26 revision 2026-10-21)."""
    teacher = job.teacher or {}
    skeleton = Skeleton.from_dict(teacher["skeleton"])
    audit = _audit_times(experiment_root)
    window_ns = int(round(job.exclude_audit_s * 1e9))
    excluded = {"audit": 0, "unassigned": 0, "undecodable": 0}
    sources: dict[str, dict] = {}
    runs = teacher["tracking_runs"]
    for i, (recording_id, tracking_run_id) in enumerate(runs.items()):
        _reading(progress, i, len(runs), recording_id)
        recording_dir = experiment_store.recording_path(experiment_root, recording_id)
        frames, complete = _tracked_frames(recording_dir, tracking_run_id)
        candidates = []
        for (part_id, frame_idx), (t_ns, subjects) in sorted(frames.items()):
            if frame_idx not in complete.get(part_id, set()):
                excluded["unassigned"] += 1
                continue
            if _near_audit(audit.get((recording_id, part_id)), t_ns, window_ns):
                excluded["audit"] += 1
                continue
            if frame_idx % job.every_n_frames:
                continue
            candidates.append(
                {
                    "recording_id": recording_id,
                    "part_id": part_id,
                    "frame_idx": frame_idx,
                    "t_ns": t_ns,
                    "subjects": subjects,
                }
            )
        detection_dir = pose_match._detection_dir(recording_dir, tracking_run_id)
        sources[recording_id] = {
            "recording_dir": recording_dir,
            "tracking_run_id": tracking_run_id,
            "detection_dir": detection_dir,
            "mask_hw": pose_match.subject_mask_hw(recording_dir, tracking_run_id),
            "mask_size": subject_frames._mask_size(detection_dir),
            "example_set": teacher.get("example_set"),
            "candidates": candidates,
            "boxes": _boxes(frames),
            "complete": complete,
            "detections": {},  # part_id -> the part's columns, read when first taught
        }
    if not sources:
        raise ValueError("a keypoint dataset needs at least one recording to label")
    return {
        "skeleton": skeleton,
        "sources": sources,
        "excluded": excluded,
        "match": _match(teacher),
    }


def _assign(instances: dict, cand: dict, mask_hw, match: dict, k: int) -> dict:
    """One frame's instances given to its subjects by §26 *Assignment*'s function — the one
    keypoint track runs — and the frame's label rows: ``{track_id: (bbox, points, scores,
    candidates, candidate_scores)}``. A subject no instance was assigned to is still an
    animal on the frame: it is taught with its box and every point not visible, never left
    unlabelled. An instance no subject took is taught to nobody."""
    subject_id, _inside, per_subject = pose_match.assign_part(
        instances,
        [(cand["frame_idx"], cand["t_ns"], cand["subjects"])],
        mask_hw,
        min_inside=match["min_inside"],
        mask_grow=match["mask_grow"],
    )
    cands = instances.get("candidates")
    cand_scores = instances.get("candidate_scores")
    tracks = {}
    for track_id, subject in cand["subjects"].items():
        row = per_subject[int(track_id)][0][2]
        if row < 0:
            tracks[int(track_id)] = (
                subject.bbox,
                np.full((k, 2), np.nan),
                np.full(k, np.nan),
                None,
                None,
            )
            continue
        tracks[int(track_id)] = (
            subject.bbox,
            np.asarray(instances["points"][row]),
            np.asarray(instances["point_scores"][row]),
            None if cands is None else np.asarray(cands[row]),
            None if cand_scores is None else np.asarray(cand_scores[row]),
        )
    return tracks


def _teach(gathered: dict, chosen: list[dict]) -> None:
    """A detect-run teacher's labels for the drawn frames: each frame's stored instances
    (``KeypointDetectRun.at``) assigned to the tracking run's subjects (:func:`_assign`),
    written into each candidate's ``tracks``. Only the drawn frames are read and assigned.
    """
    k = len(gathered["skeleton"].names)
    for cand in chosen:
        if "tracks" in cand:
            continue
        source = gathered["sources"][cand["recording_id"]]
        instances = source["detect_run"].at(cand["part_id"], cand["frame_idx"])
        cand["tracks"] = _assign(
            instances, cand, source["mask_hw"], gathered["match"], k
        )


def _share(counts: list[int], n: int) -> list[int]:
    """``n`` over sources with ``counts`` candidates, proportional, never more than a source
    has; the largest remainders take the rounding."""
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)
    n = min(n, total)
    raw = [c * n / total for c in counts]
    take = [min(c, int(r)) for c, r in zip(counts, raw)]
    order = sorted(range(len(counts)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    while sum(take) < n:
        for i in order:
            if sum(take) >= n:
                break
            if take[i] < counts[i]:
                take[i] += 1
    return take


def _slices(candidates: list[dict], k: int, rng: random.Random) -> list[dict]:
    """``k`` candidates in time order, one drawn at random from each of ``k`` equal slices."""
    if k <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c["part_id"], c["frame_idx"]))
    edges = np.linspace(0, len(ordered), k + 1).astype(int)
    return [
        ordered[rng.randrange(lo, hi)]
        for lo, hi in zip(edges[:-1], edges[1:])
        if hi > lo
    ]


def _example_frames(
    gathered: dict, skeleton: Skeleton, experiment_root: Path
) -> list[dict]:
    """The detect runs' example frames, as labeled frames: the user's own clicks. A frame
    is used only when the tracking says its subject was the only animal on it."""
    out = []
    seen: set[tuple] = set()
    for recording_id, source in gathered["sources"].items():
        example_set = source.get("example_set") or {}
        set_id = example_set.get("id")
        for mark_frame in example_set.get("marks") or []:
            if mark_frame.get("recording_id") != recording_id:
                continue
            key = (recording_id, mark_frame["part_id"], int(mark_frame["frame_idx"]))
            if key in seen:
                continue
            present = source["boxes"].get(
                (mark_frame["part_id"], int(mark_frame["frame_idx"]))
            )
            if not present or set(present) != {int(mark_frame["track_id"])}:
                continue  # another animal would go unlabelled
            k = len(skeleton.names)
            points = np.full((k, 2), np.nan)
            scores = np.full(k, np.nan)
            for mark in mark_frame.get("marks") or []:
                if (
                    mark.get("target") in skeleton.names
                    and mark.get("visible", True)
                    and mark.get("x") is not None
                ):
                    j = skeleton.index(mark["target"])
                    points[j] = (float(mark["x"]), float(mark["y"]))
                    scores[j] = 1.0
            seen.add(key)
            out.append(
                {
                    "recording_id": recording_id,
                    "part_id": mark_frame["part_id"],
                    "frame_idx": int(mark_frame["frame_idx"]),
                    "t_ns": int(mark_frame["t_ns"]),
                    "set_id": set_id,
                    "image": mark_frame["image"],
                    "tracks": {
                        int(mark_frame["track_id"]): (
                            tuple(mark_frame["bbox"]),
                            points,
                            scores,
                            None,
                            None,
                        )
                    },
                    "source": _SOURCE_EXAMPLE,
                }
            )
    return out


def _decode(source: dict, part_id: str, frame_idx: int) -> np.ndarray | None:
    """The exact frame as BGR (what ``StagedExamples`` writes), or None when the decoder lands
    anywhere else."""
    recording_dir = source["recording_dir"]
    data = recording_store.read_recording(recording_dir)
    files = recording_store.video_files(data)
    index = next(
        (i for i, f in enumerate(files) if recording_store.part_stem(f) == part_id),
        None,
    )
    if index is None:
        return None
    tsmap = recording_store.load_active_timestamps_cached(recording_dir, data, index)
    decoded = read_frame_at(
        recording_store.resolve_part_video_path(recording_dir, files[index]),
        frame_idx,
        timestamps=tsmap.values,
    )
    if decoded is None or decoded[0] != frame_idx:
        return None
    return decoded[1]


def _label(job: KeypointDatasetJob, run_dir: Path, progress_cb, stop_event) -> None:
    experiment_root = run_dir.parent.parent
    start = time.monotonic()
    emit = ThrottledProgress(progress_cb)
    by_model = job.teacher_kind == "model"
    emit(
        LabelingProgressEvent(
            0,
            1,
            0,
            1,
            0.0,
            None,
            "reading tracked frames" if by_model else "reading teacher rows",
        ),
        force=True,
    )
    gathered = _gather(experiment_root, job)
    skeleton: Skeleton = gathered["skeleton"]
    sources = gathered["sources"]

    rng = random.Random(zlib.crc32(job.dataset_id.encode()))
    recording_ids = list(sources)
    counts = [len(sources[r]["candidates"]) for r in recording_ids]
    chosen: list[dict] = []
    for recording_id, k in zip(recording_ids, _share(counts, job.n_frames)):
        for cand in _slices(sources[recording_id]["candidates"], k, rng):
            chosen.append({**cand, "source": _SOURCE_TEACHER})
    examples = _example_frames(gathered, skeleton, experiment_root)
    chosen_keys = {(c["recording_id"], c["part_id"], c["frame_idx"]) for c in chosen}
    chosen.extend(
        e
        for e in examples
        if (e["recording_id"], e["part_id"], e["frame_idx"]) not in chosen_keys
    )
    if not chosen:
        raise ValueError(
            "no frame can be labeled: every tracked frame is unassigned, has no teacher "
            f"instance, or is within {job.exclude_audit_s} s of an audit frame "
            f"({gathered['excluded']})"
        )
    if not by_model:
        emit(
            LabelingProgressEvent(0, 1, 0, 1, 0.0, None, "assigning teacher instances"),
            force=True,
        )
        _teach(gathered, [c for c in chosen if c["source"] == _SOURCE_TEACHER])

    # The split is drawn once the frames are written, over the ones that were (an undecodable
    # or stopped-before frame never takes a validation slot): `_dataset_common`'s rule.
    staged = StagedExamples(run_dir)
    total = len(chosen)
    done = 0
    used: list[dict] = []
    interrupted = False
    excluded = gathered["excluded"]
    excluded.setdefault("undecodable", 0)
    raw: list[tuple[int, int, tuple]] = (
        []
    )  # (frame index in `used`, track, teacher output)
    phase = "labeling frames" if by_model else "decoding frames"

    def _write(cand: dict, frame: np.ndarray) -> None:
        lines = [
            label_line(track[0], track[1], track[2], job.score_cutoff)
            for _tid, track in sorted(cand["tracks"].items())
        ]
        for tid, track in sorted(cand["tracks"].items()):
            raw.append((len(used), int(tid), track))
        stem = f"{cand['recording_id']}_{cand['part_id']}_{cand['frame_idx']:07d}"
        staged.add(
            cand["recording_id"],
            (cand["recording_id"], cand["part_id"], cand["frame_idx"]),
            stem,
            frame,
            lines,
        )
        used.append(
            {
                "recording_id": cand["recording_id"],
                "part_id": cand["part_id"],
                "frame_idx": int(cand["frame_idx"]),
                "t_ns": int(cand["t_ns"]),
                "track_ids": sorted(int(t) for t in cand["tracks"]),
                "source": cand["source"],
            }
        )

    def _progress(force: bool = False) -> None:
        elapsed = time.monotonic() - start
        emit(
            LabelingProgressEvent(
                done,
                total,
                1 if used else 0,
                1,
                elapsed,
                remaining_seconds(done / total, elapsed) if total else None,
                phase,
            ),
            force=force,
        )

    teacher = None
    if by_model:
        teacher = _ModelTeacher(job, experiment_root, sources, emit)
    # The phase boundary is always shown (§6): past the throttle, before the first frame.
    _progress(force=True)
    for cand in sorted(
        chosen, key=lambda c: (c["recording_id"], c["part_id"], c["frame_idx"])
    ):
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            break
        source = sources[cand["recording_id"]]
        if cand["source"] == _SOURCE_EXAMPLE:
            rgb = example_sets.load_image(
                experiment_root, cand["set_id"], cand["image"]
            )
            frame = rgb_to_bgr(rgb)
        else:
            frame = _decode(source, cand["part_id"], cand["frame_idx"])
        if frame is None:
            done += 1
            excluded["undecodable"] += 1
            _progress()
            continue
        if teacher is not None and cand["source"] == _SOURCE_TEACHER:
            # Labelled in batches: the frame waits until the batch is full.
            for labelled, labelled_frame in teacher.add(cand, frame):
                done += 1
                _write(labelled, labelled_frame)
                _progress()
            continue
        done += 1
        _write(cand, frame)
        _progress()
    if teacher is not None and not interrupted:
        for labelled, labelled_frame in teacher.flush():
            done += 1
            _write(labelled, labelled_frame)
            _progress()
    elapsed = time.monotonic() - start
    emit(
        LabelingProgressEvent(
            done if interrupted else total,
            total,
            1 if used else 0,
            1,
            elapsed,
            None,
            "complete" if not interrupted else "stopped",
        ),
        force=True,
    )
    split = staged.commit(job.val_split, job.dataset_id)
    for frame in used:
        frame["split"] = split[
            (frame["recording_id"], frame["part_id"], frame["frame_idx"])
        ]
    n_val = sum(1 for frame in used if frame["split"] == "val")
    _write_teacher_output(run_dir, skeleton, raw)
    _finalize(
        run_dir, job, gathered, skeleton, used, len(used) - n_val, n_val, interrupted
    )


class _ModelTeacher:
    """The patch method as a teacher: built once from the job's own copy of the example set,
    loaded on the job's compute target, then fed the sampled frames in batches. Each frame is
    given as the detect stage gives it (§26 revision 2026-10-21): every **detection** on it
    (its box, and its mask for a method that reads masks), or nothing for a whole-frame
    method. The instances it places are then given to the frame's tracked subjects by §26
    *Assignment*'s function (:func:`_assign`), as a detect-run teacher's stored ones are.
    """

    def __init__(self, job: KeypointDatasetJob, experiment_root, sources, emit) -> None:
        from detail.configs.keypoint_detect import KeypointDetectConfig
        from detail.pipeline.keypoint_detect import build_method

        teacher = job.teacher or {}
        emit(
            LabelingProgressEvent(
                0, 1, 0, 1, 0.0, None, f"loading {teacher['model_id']}"
            ),
            force=True,
        )
        self._sources = sources
        self._match = _match(teacher)
        self._k = len(teacher["skeleton"]["names"])
        self._model = build_method(
            Path(experiment_root),
            teacher["model_id"],
            teacher.get("model_variant"),
            teacher.get("settings") or {},
            teacher.get("example_set"),
            teacher["skeleton"],
        )
        self.device = resolve_device(job.compute, self._model.supported_backends)
        self._model.load(self.device)
        self._per_detection = bool(getattr(type(self._model), "per_detection", True))
        self._needs_mask = self._per_detection and bool(
            getattr(self._model, "needs_mask", False)
        )
        self._batch_size = int(KeypointDetectConfig.model_fields["batch_size"].default)
        self._pending: list[tuple[dict, np.ndarray]] = []

    def add(self, cand: dict, frame_bgr: np.ndarray):
        self._pending.append((cand, frame_bgr))
        if len(self._pending) >= self._batch_size:
            return self.flush()
        return []

    def flush(self) -> list[tuple[dict, np.ndarray]]:
        batch, self._pending = self._pending, []
        if not batch:
            return []
        frames = [self._frame(cand, bgr) for cand, bgr in batch]
        results = self._model.predict_batch(frames)
        out = []
        for (cand, bgr), result in zip(batch, results):
            source = self._sources[cand["recording_id"]]
            tracks = _assign(
                _instances(result), cand, source["mask_hw"], self._match, self._k
            )
            out.append(({**cand, "tracks": tracks}, bgr))
        return out

    def _frame(self, cand: dict, bgr: np.ndarray) -> InstanceFrame:
        """What the detect stage would hand the method for this frame."""
        boxes: dict[int, tuple] = {}
        masks = None
        if self._per_detection:
            source = self._sources[cand["recording_id"]]
            det = _part_detections(source, cand["part_id"])
            rows = det["rows_on"].get(int(cand["frame_idx"]), [])
            boxes = {r: tuple(float(v) for v in det["bbox"][r]) for r in rows}
            if self._needs_mask and source["mask_size"] is not None:
                masks = {
                    r: subject_frames.subject_mask(det, [r], source["mask_size"])
                    for r in rows
                }
        return InstanceFrame(
            frame_idx=int(cand["frame_idx"]),
            t_ns=int(cand["t_ns"]),
            frame=bgr_to_rgb(bgr),
            boxes=boxes,
            masks=masks,
        )


def _part_detections(source: dict, part_id: str) -> dict:
    """A detection part's columns (``mask_rle`` read lazily) and its rows per frame, read
    the first time a frame of the part is taught."""
    cache = source["detections"]
    if part_id not in cache:
        det = subject_frames._part_columns(
            DetectionPartReader(source["detection_dir"] / f"{part_id}.zarr")
        )
        rows_on: dict[int, list[int]] = {}
        for row, frame in enumerate(np.asarray(det["frame_idx"]).tolist()):
            rows_on.setdefault(int(frame), []).append(row)
        cache[part_id] = {**det, "rows_on": rows_on}
    return cache[part_id]


def _instances(result) -> dict:
    """A ``FrameResult``'s instances as the rows :func:`_assign` reads (a detect part's
    columns, for one frame)."""
    n = int(result.n)
    out = {
        "frame_idx": np.full(n, int(result.frame_idx), np.int64),
        "points": np.asarray(result.points, np.float32).reshape(n, -1, 2),
        "point_scores": np.asarray(result.point_scores, np.float32).reshape(n, -1),
        "confidence": (
            np.full(n, np.nan, np.float32)
            if result.confidence is None
            else np.asarray(result.confidence, np.float32)
        ),
    }
    if getattr(result, "candidates", None) is not None:
        out["candidates"] = np.asarray(result.candidates)
        out["candidate_scores"] = np.asarray(result.candidate_scores)
    return out


def _finalize(
    run_dir, job, gathered, skeleton, used, n_train, n_val, interrupted
) -> None:
    datasets_store.write_data_yaml(
        run_dir,
        {0: SUBJECT_CLASS},
        kpt_shape=(len(skeleton.names), 3),
        flip_idx=flip_index(skeleton),
    )
    sources = gathered["sources"]
    datasets_store.write_dataset_yaml(
        run_dir,
        {
            "id": job.dataset_id,
            "name": job.name,
            "created_at": job.created_at,
            "task": "pose",
            "source": _source_record(job, sources),
            # The library skeleton the teacher's example set copied (§26): what the Keypoints
            # manager's dependents search reads, so a used skeleton stays frozen.
            "skeleton_ref": (job.teacher or {}).get("skeleton_ref"),
            "skeleton": skeleton.to_dict(),
            "skeleton_digest": skeleton.digest(),
            "classes": [{"id": 0, "name": SUBJECT_CLASS}],
            "keypoint_dataset": {
                "n_frames": int(job.n_frames),
                "val_split": float(job.val_split),
                "score_cutoff": float(job.score_cutoff),
                "exclude_audit_s": float(job.exclude_audit_s),
                "every_n_frames": int(job.every_n_frames),
            },
            "excluded": dict(gathered["excluded"]),
            # Asked for, and labelled from the teacher: fewer when the tracked frames ran out
            # or some could not be decoded exactly (counted in `excluded`), never silently.
            "n_requested": int(job.n_frames),
            "n_from_teacher": sum(1 for f in used if f["source"] == _SOURCE_TEACHER),
            "frames": used,
            "n_train": n_train,
            "n_val": n_val,
            "status": "partial" if interrupted else "complete",
        },
    )


def _source_record(job: KeypointDatasetJob, sources: dict) -> dict:
    """What taught the dataset, in ``dataset.yaml → source``: the ``keypoint_detect`` runs
    (with the tracking runs their instances were assigned to and the example sets behind them),
    or the patch method with the example set it was built from and the tracking runs whose
    subjects its instances were assigned to; either way the assignment's settings."""
    if job.teacher_kind == "model":
        teacher = job.teacher or {}
        example_set = teacher.get("example_set") or {}
        return {
            "kind": "model",
            "model_id": teacher["model_id"],
            "model_variant": teacher.get("model_variant"),
            "settings": dict(teacher.get("settings") or {}),
            "example_set": {
                "id": example_set.get("id"),
                "digest": example_set.get("digest"),
            },
            "tracking_runs": dict(teacher["tracking_runs"]),
            "match": _match(teacher),
        }
    return {
        "kind": "keypoint_detect",
        "detect_runs": {r: s["detect_run_id"] for r, s in sources.items()},
        # the subjects each detect run's instances were assigned to (§26 *Assignment*)
        "tracking_runs": {r: s["tracking_run_id"] for r, s in sources.items()},
        "match": _match(job.teacher or {}),
        "example_sets": {
            r: {
                "id": s["example_set"]["id"],
                "digest": s["example_set"].get("digest"),
            }
            for r, s in sources.items()
            if s.get("example_set")
        },
    }


#: Beside ``dataset.yaml``: what the teacher gave for every labelled (frame, track), before the
#: cut-off. Recorded so a cut-off can be re-applied later without running the teacher again.
TEACHER_OUTPUT = "teacher.npz"


def _write_teacher_output(run_dir: Path, skeleton: Skeleton, raw: list) -> None:
    """``teacher.npz``: one row per labelled (frame, track), ``frame`` indexing
    ``dataset.yaml → frames``; ``points (R, K, 2)`` and ``point_scores (R, K)`` as the teacher
    placed them, and ``candidates (R, K, C, 2)`` / ``candidate_scores (R, K, C)`` (``C`` 0 when
    the teacher kept none; NaN where a row has fewer). An example frame's clicks score 1.
    """
    k = len(skeleton.names)
    c = max(
        (int(np.asarray(t[3]).shape[1]) for _f, _tid, t in raw if t[3] is not None),
        default=0,
    )
    n = len(raw)
    points = np.full((n, k, 2), np.nan, np.float32)
    scores = np.full((n, k), np.nan, np.float32)
    cands = np.full((n, k, c, 2), np.nan, np.float32)
    cand_scores = np.full((n, k, c), np.nan, np.float32)
    frame = np.zeros(n, np.int32)
    track = np.zeros(n, np.int32)
    for row, (index, tid, t) in enumerate(raw):
        frame[row], track[row] = index, tid
        points[row] = np.asarray(t[1], np.float32).reshape(k, 2)
        scores[row] = np.asarray(t[2], np.float32).reshape(k)
        if t[3] is not None and c:
            have = np.asarray(t[3], np.float32)
            cands[row, :, : have.shape[1]] = have
            cand_scores[row, :, : have.shape[1]] = np.asarray(t[4], np.float32)
    tmp = run_dir / (TEACHER_OUTPUT + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(
            handle,
            frame=frame,
            track_id=track,
            points=points,
            point_scores=scores,
            candidates=cands,
            candidate_scores=cand_scores,
        )
    atomic.replace(tmp, run_dir / TEACHER_OUTPUT)
