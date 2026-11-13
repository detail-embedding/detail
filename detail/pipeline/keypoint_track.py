"""``keypoint_track`` (§26 "Track methods"): a detect run's pose instances given to a tracking
run's subjects, then settled over time into final points — one CPU core, re-run whole. On this
machine a batch's recordings run side by side, one per core for half the cores
(``local_parallelism``); on a registered cluster (§6) each is its own job.

**Assignment comes first, whatever method placed the instances** (§26 *Assignment*,
``pipeline/pose_match.py``): per part, each instance goes to at most one subject by the share of
its points inside the subject's grown mask, written to ``assignments/<part>.zarr`` parallel to
the detect rows. Each subject's series then has a row for every frame it is tracked at the
cadence — the assigned instance's columns, or NaN with ``instance_row`` −1 — so a method sees a
gap in time as a gap, never as a neighbour's pose.

A **track method** owns its filtering: nothing filters in the detect stage, and no filter
applies to a method that did not declare it (#626 despiked and smoothed every method's output
with one setting, measured on one method). Two methods:

- ``temporal_decode`` (``needs_candidates``): per point, a path through the stored candidates
  plus a *missing* state — emission the candidate's score, transition a Gaussian on the pixel
  move scaled by body size and by the rows elapsed, so a gap costs what its duration earns.
  Left/right is settled over the path: a pair's ``mirrored`` score (the detect run's extras)
  weighs against its unmirrored one through one two-state path per pair, so a swap must hold
  over frames to win. Then the two filters below, each switchable.
- ``despike_smooth``: the filters alone, on the best point — the student's default.

Both run per (track, part), in pixels (normalized x and y are not isotropic on a non-square
frame), with every window a duration in seconds resolved to rows at the recording's cadence.
``changed`` records how each point came to be: 0 as detected, 1 another candidate than the
detect stage's best, 2 moved by a filter. Raw detect output is never rewritten: a different
filter or decode is a new track run over the same detect run.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np

from detail.analysis import keypoint_filters as filters
from detail.configs.keypoint_track import (
    PARALLELISM_ENV,
    DespikeSmoothConfig,
    KeypointTrackConfig,
    TemporalDecodeConfig,
)
from detail.core.dataclasses import ComputeTarget
from detail.core.keypoints import KeypointProgressEvent, KeypointTrackJob, TrackPart
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.protocols import KeypointTrackMethod
from detail.core.registries import (
    KEYPOINT_TRACK_METHODS,
    register_keypoint_track_method,
)
from detail.core.skeleton import Skeleton
from detail.core.utils import utc_timestamp
from detail.executor.runners import ProcessingModule, register_processing_module
from detail.pipeline import teacher_window
from detail.pipeline.pose_match import (
    assign_part,
    eligible_instances,
    frame_subjects,
    subject_mask_hw,
    tracked_class_rows,
)
from detail.storage import experiment as experiment_store
from detail.storage import keypoint_runs
from detail.storage import recording as recording_store
from detail.storage import remote_targets
from detail.storage.keypoint_runs import KeypointDetectRun
from detail.storage.keypoint_zarr import (
    KeypointAssignmentPartWriter,
    KeypointTrackPartWriter,
)

_NS_PER_S = 1_000_000_000
#: The stage runs numpy on one core and loads no model, so a device to pick has no meaning.
_BACKENDS = ("cpu", "remote")
_MAX_LOCAL_PARALLELISM = 8


def keypoint_track_run_dir(experiment_root, job: KeypointTrackJob) -> Path:
    """``recordings/<id>/keypoint_track/<kt_NN>/``."""
    return keypoint_runs.track_run_dir(
        experiment_store.recording_path(experiment_root, job.recording_id),
        job.track_run_id,
    )


# --------------------------------------------------------------------------- #
# The filters both methods end with
# --------------------------------------------------------------------------- #
def tracked_columns(skeleton: Skeleton, config) -> np.ndarray:
    """``(K,)`` bool: the skeleton's points the config's ``track_points`` names, in column
    order; every point when it is unset. Names only — nothing here knows a body plan."""
    chosen = getattr(config, "track_points", None)
    if chosen is None:
        return np.ones(len(skeleton.names), dtype=bool)
    wanted = set(chosen)
    return np.asarray([name in wanted for name in skeleton.names], dtype=bool)


def resolve_track_points(
    skeleton: Skeleton, track_points: list[str] | None
) -> list[str] | None:
    """``track_points`` as the run records it (Tenet 9): in the skeleton's column order,
    without repeats, and ``None`` when it names every point, so two configs that track the
    same points are the same run.

    Raises:
        ValueError: A name the skeleton does not have, or an empty list (nothing to track:
            leave it unset to track every point).
    """
    if track_points is None:
        return None
    names = list(skeleton.names)
    unknown = sorted(set(track_points) - set(names))
    if unknown:
        raise ValueError(
            f"track_points names {unknown}, which the detect run's skeleton does not have; "
            f"its points are {names}"
        )
    if not track_points:
        raise ValueError(
            "track_points is empty, so nothing would be tracked; leave it unset to track "
            "every point, or name the points to track"
        )
    chosen = [n for n in names if n in set(track_points)]
    return None if len(chosen) == len(names) else chosen


def apply_filters(
    points_px: np.ndarray,
    t_ns: np.ndarray,
    scale_px: np.ndarray,
    config,
    rows_per_second: float,
    tracked: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Despike then smooth, each when the config enables it: ``(points_px, moved)`` with
    ``moved`` ``(M, K)`` bool where a filter moved a point. Absence is never filled. A point
    outside ``tracked`` (``(K,)`` bool; None: all) comes back as it went in: both filters are
    per point, so filtering every column and restoring the untracked ones is exact."""
    original = np.asarray(points_px, dtype=np.float64)
    out = original.copy()
    moved = np.zeros(out.shape[:2], dtype=bool)
    if config.despike:
        half = filters.rows_for_seconds(config.despike_window_s, rows_per_second)
        out, replaced = filters.hampel_despike(
            out,
            scale_px,
            half_window=half,
            k=config.despike_k,
            floor=config.despike_floor,
        )
        moved |= replaced
    if config.smooth:
        smoothed = filters.smooth_local_quadratic(
            out,
            t_ns,
            config.smooth_window_s,
            truncate=config.smooth_truncate,
            min_points=config.smooth_min_points,
        )
        present = np.isfinite(out).all(axis=2)
        shift = np.where(present, np.abs(smoothed - out).max(axis=2), 0.0)
        moved |= present & (shift > 1e-6)
        out = smoothed
    if tracked is not None and not tracked.all():
        out[:, ~tracked] = original[:, ~tracked]
        moved[:, ~tracked] = False
    return out, moved


def _to_px(points: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * np.asarray(frame_wh, dtype=np.float64)


def _to_norm(points_px: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
    return (points_px / np.asarray(frame_wh, dtype=np.float64)).astype(np.float32)


def _as_detected(points: np.ndarray, rows: dict, tracked: np.ndarray) -> np.ndarray:
    """``points`` (normalized) with every untracked point the detect row's own value, bit for
    bit: a trip through pixels and back can move the last float32 digit, and "kept as
    detected" should be exactly that."""
    if tracked.all():
        return points
    out = np.array(points, dtype=np.float32, copy=True)
    out[:, ~tracked] = np.asarray(rows["points"], dtype=np.float32)[:, ~tracked]
    return out


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
@register_keypoint_track_method("despike_smooth")
class DespikeSmooth(KeypointTrackMethod):
    """The filters alone, on the detect run's best point per row."""

    method_id = "despike_smooth"
    config_model = DespikeSmoothConfig
    needs_candidates = False

    def track(
        self,
        rows,
        skeleton,
        config,
        *,
        rows_per_second,
        frame_wh,
        body_scale_px,
        phase=None,
    ) -> TrackPart:
        points = _to_px(rows["points"], frame_wh)
        scores = np.asarray(rows["point_scores"], dtype=np.float32)
        if phase is not None:
            phase("filtering")
        tracked = tracked_columns(skeleton, config)
        filtered, moved = apply_filters(
            points, rows["t_ns"], body_scale_px, config, rows_per_second, tracked
        )
        changed = np.where(moved, 2, 0).astype(np.int8)
        return TrackPart(
            points=_as_detected(_to_norm(filtered, frame_wh), rows, tracked),
            point_scores=scores,
            changed=changed,
        )


@register_keypoint_track_method("temporal_decode")
class TemporalDecode(KeypointTrackMethod):
    """A path through the stored candidates, left/right settled over time, then the filters."""

    method_id = "temporal_decode"
    config_model = TemporalDecodeConfig
    needs_candidates = True

    def track(
        self,
        rows,
        skeleton,
        config,
        *,
        rows_per_second,
        frame_wh,
        body_scale_px,
        phase=None,
    ) -> TrackPart:
        # A row no instance was assigned to is NaN throughout, so every candidate is absent
        # there and the path can only be *missing* — the gap is the missing state, not a fill.
        if phase is not None:
            phase("decoding")
        points, scores, changed = self.decode(
            rows,
            skeleton,
            config,
            rows_per_second=rows_per_second,
            frame_wh=frame_wh,
            body_scale_px=body_scale_px,
        )
        if phase is not None:
            phase("filtering")
        return self.finish(
            rows,
            points,
            scores,
            changed,
            config,
            rows_per_second,
            frame_wh,
            body_scale_px,
            tracked_columns(skeleton, config),
        )

    @staticmethod
    def finish(
        rows,
        points,
        scores,
        changed,
        config,
        rows_per_second,
        frame_wh,
        body_scale_px,
        tracked,
    ) -> TrackPart:
        """The filters on a decoded path, on the ``tracked`` points only. Separate from
        ``decode`` because the path depends on the transition, missing and mirror settings
        (and the points tracked) alone: tuning (``keypoint_track_tune``) decodes once per path
        setting and filters that path once per filter setting."""
        t_ns = np.asarray(rows["t_ns"], dtype=np.int64)
        filtered, moved = apply_filters(
            points, t_ns, body_scale_px, config, rows_per_second, tracked
        )
        changed = np.where(moved, 2, changed).astype(np.int8)
        return TrackPart(
            points=_as_detected(_to_norm(filtered, frame_wh), rows, tracked),
            point_scores=scores.astype(np.float32),
            changed=changed,
        )

    def decode(
        self, rows, skeleton, config, *, rows_per_second, frame_wh, body_scale_px
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The path through ``rows``' candidates with left/right settled: ``(points_px (M, K,
        2), scores (M, K), changed (M, K) int8)``, before any filter. A point outside the
        config's ``track_points`` is the detect stage's own placement and score, unchanged; a
        pair is settled left from right only when both its points are tracked."""
        tracked = tracked_columns(skeleton, config)
        t_ns = np.asarray(rows["t_ns"], dtype=np.int64)
        best = _to_px(rows["points"], frame_wh)  # (M, K, 2) the detect stage's choice
        cand = _to_px(rows["candidates"], frame_wh)  # (M, K, C, 2)
        cand_scores = np.asarray(
            rows["candidate_scores"], dtype=np.float64
        )  # (M, K, C)
        mirrored = rows.get("extras/mirrored")
        m, k, c = cand_scores.shape
        offered = cand_scores.copy()
        if config.decode_cutoff is not None:
            offered[offered < config.decode_cutoff] = np.nan
        offered[~np.isfinite(cand).all(axis=3)] = np.nan
        dt_rows = (
            np.diff(t_ns) / _NS_PER_S * rows_per_second
        )  # rows elapsed between rows
        sigma = (
            config.transition_sigma_body
            * np.asarray(body_scale_px, dtype=np.float64)[1:]
            * np.sqrt(np.maximum(dt_rows, 1.0))
        )  # (M-1,)
        chosen = np.full((m, k), -1, dtype=np.int64)
        for j in np.flatnonzero(tracked):
            cj = cand[:, j]  # (M, C, 2)

            def cost(i, cj=cj):
                d = np.linalg.norm(cj[i][None, :, :] - cj[i - 1][:, None, :], axis=2)
                return 0.5 * (d / max(float(sigma[i - 1]), 1e-6)) ** 2

            chosen[:, j] = filters.viterbi_path(
                offered[:, j], cost, config.missing_penalty
            )
        rows_idx = np.arange(m)[:, None]
        cols = np.arange(k)[None, :]
        picked = np.where(chosen >= 0, chosen, 0)
        points = np.where(
            (chosen >= 0)[..., None], cand[rows_idx, cols, picked], np.nan
        )  # (M, K, 2)
        scores = np.where(chosen >= 0, cand_scores[rows_idx, cols, picked], np.nan)
        untracked = ~tracked
        points[:, untracked] = best[:, untracked]
        scores[:, untracked] = np.asarray(rows["point_scores"], dtype=np.float64)[
            :, untracked
        ]
        # Left/right over the path: evidence for "swapped" is how much better the examples
        # mirrored (labels swapped) matched than the examples as given, per pair.
        if mirrored is not None:
            mirrored = np.asarray(mirrored, dtype=np.float64)
            for left, right in skeleton.pair_indices():
                if not (tracked[left] and tracked[right]):
                    continue  # a pair is settled as a pair, or not at all
                pair = np.stack([scores[:, left], scores[:, right]])  # (2, M)
                n_present = np.isfinite(pair).sum(axis=0)
                as_given = (
                    np.where(  # a row with neither point placed carries no evidence
                        n_present > 0,
                        np.nansum(pair, axis=0) / np.maximum(n_present, 1),
                        np.nan,
                    )
                )
                evidence = (mirrored[:, left] + mirrored[:, right]) / 2 - as_given
                swapped = filters.two_state_path(evidence, config.mirror_weight)
                if swapped.any():
                    points[swapped, left], points[swapped, right] = (
                        points[swapped, right].copy(),
                        points[swapped, left].copy(),
                    )
                    scores[swapped, left], scores[swapped, right] = (
                        scores[swapped, right].copy(),
                        scores[swapped, left].copy(),
                    )
                    chosen[swapped, left] = chosen[swapped, right] = (
                        -2
                    )  # marks "changed"
        # 1 where the path took another candidate than the detect stage's own point
        other = np.isfinite(points).all(axis=2) & (
            ~np.isfinite(best).all(axis=2)
            | (np.linalg.norm(np.nan_to_num(points - best), axis=2) > 1e-6)
        )
        return points, scores, np.where(other, 1, 0).astype(np.int8)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
#: The detect columns a subject's series carries, beside ``extras/<name>``: what a method
#: reads (§26 "Track methods").
_SERIES_COLUMNS = ("points", "point_scores", "candidates", "candidate_scores")


def _empty_instances(k: int, n_candidates: int) -> dict[str, np.ndarray]:
    """A part the detect run wrote no store for: no instance, so every subject-frame there
    is a gap. The extras are unknown without a store, and a method treats them as absent.
    """
    out = {
        "frame_idx": np.zeros(0, np.int32),
        "confidence": np.zeros(0, np.float32),
        "points": np.zeros((0, k, 2), np.float32),
        "point_scores": np.zeros((0, k), np.float32),
    }
    if n_candidates > 1:
        out["candidates"] = np.zeros((0, k, n_candidates, 2), np.float32)
        out["candidate_scores"] = np.zeros((0, k, n_candidates), np.float32)
    return out


def subject_series(
    instances: dict, entries: list[tuple[int, int, int, float]]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One subject's rows for its method: a row per subject-frame (``assign_part``'s
    ``entries``), the assigned instance's columns gathered by ``instance_row``, and NaN (−1 in
    an integer extra) where none was assigned. Returns the rows and the body scale per row.
    """
    frame_idx = np.asarray([e[0] for e in entries], dtype=np.int32)
    t_ns = np.asarray([e[1] for e in entries], dtype=np.int64)
    instance_row = np.asarray([e[2] for e in entries], dtype=np.int32)
    scale = np.asarray([e[3] for e in entries], dtype=np.float64)
    have = instance_row >= 0
    rows: dict[str, np.ndarray] = {
        "frame_idx": frame_idx,
        "t_ns": t_ns,
        "instance_row": instance_row,
    }
    names = [c for c in _SERIES_COLUMNS if c in instances]
    names += sorted(c for c in instances if c.startswith("extras/"))
    for name in names:
        source = np.asarray(instances[name])
        fill = -1 if np.issubdtype(source.dtype, np.integer) else np.nan
        column = np.full((len(entries),) + source.shape[1:], fill, dtype=source.dtype)
        column[have] = source[instance_row[have]]
        rows[name] = column
    return rows, scale


def _count_subject_frames(
    recording_dir: Path, tracking_run_id: str, part_ids: list[str], every_n: int
) -> dict[str, int]:
    """Subject-frames per part — the run's unit of work; no mask is decoded to count them."""
    return {
        part_id: sum(
            len(subjects)
            for _f, _t, subjects in frame_subjects(
                recording_dir, tracking_run_id, part_id, every_n
            )
        )
        for part_id in part_ids
    }


def _tracking_part_ids(recording_dir: Path, tracking_run_id: str) -> list[str]:
    folder = recording_dir / "tracking" / tracking_run_id / "assignments"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("part_*.zarr"))


def rate_and_size(
    recording_dir: Path, detect: KeypointDetectRun
) -> tuple[float, tuple[int, int]]:
    """``(rows_per_second, frame_wh)`` a method's windows and pixels resolve at: the detect
    run's cadence over the recording's rate, and the frame size."""
    data = recording_store.read_recording(Path(recording_dir))
    every_n = int(detect.meta.get("every_n_frames") or 1)
    frame_wh = (
        int(data["video"]["resolution"][0]),
        int(data["video"]["resolution"][1]),
    )
    return float(data["video"]["fps_actual"]) / every_n, frame_wh


def subject_rows(
    recording_dir: Path,
    detect: KeypointDetectRun,
    tracking_run_id: str,
    part_id: str,
    match: dict,
    *,
    window_ns: tuple[int, int] | None = None,
    on_frame=None,
) -> dict[int, tuple[dict[str, np.ndarray], np.ndarray]]:
    """``{track_id: (rows, body_scale_px)}`` for one part: the detect run's instances given to
    the tracking run's subjects by *Assignment*, exactly as the stage gives them, without
    writing anything. The one reader a tuner (``keypoint_track_tune``) and a preview
    (``keypoint_track_preview``) share with the stage, so a tried setting meets the rows the
    run will.

    ``window_ns`` (``[start, end)`` in ``t_ns``) keeps only the frames inside it: a frame's
    assignment reads that frame alone, so a window's rows are the full run's rows there.
    ``on_frame(n_subjects)`` hears each frame as it is assigned."""
    recording_dir = Path(recording_dir)
    every_n = int(detect.meta.get("every_n_frames") or 1)
    instances = (
        detect.rows(part_id)
        if part_id in set(detect.part_ids())
        else _empty_instances(len(detect.skeleton.names), detect.n_candidates)
    )
    frames = frame_subjects(recording_dir, tracking_run_id, part_id, every_n)
    if window_ns is not None:
        frames = _within(frames, window_ns)
    if on_frame is not None:
        frames = _heard(frames, on_frame)
    _subject_id, _inside, per_subject = assign_part(
        instances,
        frames,
        subject_mask_hw(recording_dir, tracking_run_id),
        eligible=_eligible(instances, recording_dir, tracking_run_id, part_id),
        min_inside=match["min_inside"],
        mask_grow=match["mask_grow"],
    )
    return {
        int(track_id): subject_series(instances, entries)
        for track_id, entries in per_subject.items()
    }


def _within(frames, window_ns: tuple[int, int]):
    lo, hi = window_ns
    for frame in frames:
        if lo <= frame[1] < hi:
            yield frame


def _heard(frames, on_frame):
    for frame in frames:
        yield frame
        on_frame(len(frame[2]))


def window_subject_frames(
    recording_dir: Path,
    detect: KeypointDetectRun,
    tracking_run_id: str,
    part_id: str,
    window_ns: tuple[int, int],
) -> int:
    """How many subject-frames :func:`subject_rows` assigns over ``window_ns``: a preview's
    unit of work, counted without decoding a mask."""
    every_n = int(detect.meta.get("every_n_frames") or 1)
    return sum(
        len(subjects)
        for _f, _t, subjects in _within(
            frame_subjects(recording_dir, tracking_run_id, part_id, every_n),
            window_ns,
        )
    )


def _eligible(
    instances: dict, recording_dir: Path, tracking_run_id: str, part_id: str
) -> np.ndarray:
    """The instances the tracking run's subjects may take: none placed on a box of a class
    the run does not follow (``pose_match.eligible_instances``)."""
    n = len(instances["frame_idx"])
    return eligible_instances(
        instances.get("detection_row", np.full(n, -1, np.int32)),
        tracked_class_rows(recording_dir, tracking_run_id, part_id),
    )


def run(job: KeypointTrackJob, run_dir, progress_cb, stop_event) -> None:
    """§26 track runner: per part, *Assignment* then each subject's method; CPU, no resume."""
    run_dir = Path(run_dir)
    recording_dir = run_dir.parent.parent
    detect = KeypointDetectRun(recording_dir, job.detect_run_id)
    skeleton = Skeleton.from_dict(job.skeleton)
    k = len(skeleton.names)
    every_n = int(detect.meta.get("every_n_frames") or 1)
    rows_per_second, frame_wh = rate_and_size(recording_dir, detect)
    method = KEYPOINT_TRACK_METHODS[job.method]()
    config = method.config_model.model_validate(job.config)
    match = dict(job.match)
    keypoint_runs.write_track_yaml(
        run_dir, keypoint_runs.track_record(job, skeleton.digest())
    )

    detect_parts = set(detect.part_ids())
    part_ids = sorted(
        detect_parts | set(_tracking_part_ids(recording_dir, job.tracking_run_id))
    )
    start = time.monotonic()
    # Two passes over every subject-frame — assigning, then the method — so the bar weighs
    # them alike; the count is known before the first part. ``done`` / ``steps`` count both
    # passes; the event reports subject-frames (``done // 2`` of ``steps // 2``), because
    # ``frames_total`` is read as a frame count — the performance report divides it by
    # seconds — and twice the frames there is twice the rate (§6 Progress reporting).
    state = {"done": 0, "steps": 0, "parts_done": 0, "track": None}
    throttled = ThrottledProgress(progress_cb)

    def emit(message: str, *, force: bool = True) -> None:
        elapsed = time.monotonic() - start
        steps = state["steps"]
        fraction = state["done"] / steps if steps else 0.0
        throttled(
            KeypointProgressEvent(
                track_id=state["track"],
                parts_done=state["parts_done"],
                parts_total=len(part_ids),
                frames_done=state["done"] // 2,
                frames_total=steps // 2,
                elapsed_s=elapsed,
                eta_s=remaining_seconds(fraction, elapsed),
                message=message,
            ),
            force=force,
        )

    emit("assigning")  # before the count, so a long count is not silence
    per_part = _count_subject_frames(
        recording_dir, job.tracking_run_id, part_ids, every_n
    )
    state["steps"] = 2 * sum(per_part.values())
    mask_hw = subject_mask_hw(recording_dir, job.tracking_run_id)

    assignment: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    stopped = False
    for part_id in part_ids:
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        state["track"] = None
        emit("assigning")
        instances = (
            detect.rows(part_id)
            if part_id in detect_parts
            else _empty_instances(k, detect.n_candidates)
        )

        def counted(frames):
            for frame in frames:
                yield frame
                state["done"] += len(frame[2])
                emit("assigning", force=False)

        subject_id, inside, per_subject = assign_part(
            instances,
            counted(
                frame_subjects(recording_dir, job.tracking_run_id, part_id, every_n)
            ),
            mask_hw,
            eligible=_eligible(instances, recording_dir, job.tracking_run_id, part_id),
            min_inside=match["min_inside"],
            mask_grow=match["mask_grow"],
        )
        if part_id in detect_parts:
            _write_assignment(run_dir, part_id, subject_id, inside)
        n_assigned = int((subject_id >= 0).sum())
        assignment[part_id] = {
            "instances": int(len(subject_id)),
            "assigned": n_assigned,
            "unassigned": int(len(subject_id)) - n_assigned,
        }
        for track_id in sorted(per_subject):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            state["track"] = int(track_id)
            rows, scale = subject_series(instances, per_subject[track_id])
            part = method.track(
                rows,
                skeleton,
                config,
                rows_per_second=rows_per_second,
                frame_wh=frame_wh,
                body_scale_px=scale,
                phase=emit,
            )
            path = keypoint_runs.track_part_path(run_dir, track_id, part_id)
            if path.exists():  # no resume: the part is re-run whole
                shutil.rmtree(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with KeypointTrackPartWriter(path, tuple(skeleton.names)) as writer:
                writer.add_rows(
                    rows["frame_idx"],
                    rows["t_ns"],
                    rows["instance_row"],
                    part.points,
                    part.point_scores,
                    part.changed,
                )
            coverage.setdefault(f"track_{track_id}", {})[part_id] = {
                "placed": int((rows["instance_row"] >= 0).sum()),
                "total": int(len(rows["frame_idx"])),
            }
            state["done"] += len(rows["frame_idx"])
            emit("filtering")
        if stopped:
            break
        state["parts_done"] += 1
    elapsed = time.monotonic() - start
    keypoint_runs.update_track_yaml(
        run_dir,
        assignment=assignment,
        coverage=coverage,
        timing={"frames": state["done"] // 2, "seconds": round(elapsed, 3)},
    )
    state["track"] = None
    emit("stopped" if stopped else "complete")


def _write_assignment(
    run_dir: Path, part_id: str, subject_id: np.ndarray, inside: np.ndarray
) -> None:
    """``assignments/<part>.zarr``, parallel to the detect part's rows; re-run whole."""
    path = keypoint_runs.assignment_path(run_dir, part_id)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with KeypointAssignmentPartWriter(path) as writer:
        for sid, share in zip(subject_id.tolist(), inside.tolist()):
            writer.add(sid, share)


# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #
@register_processing_module
class KeypointTrackModule(ProcessingModule):
    module_id = "keypoint_track"
    id_prefix = "kt"
    runs_key = "keypoint_track_runs"
    upstream = "keypoint_detect"
    job_class = KeypointTrackJob
    progress_event_class = KeypointProgressEvent
    # frames_total is subject-frames; both passes are folded into one count
    progress_counts_frames = True
    config_model = KeypointTrackConfig
    # The tracking run reaches identity as its spec (``tracking_identity``), never as its id:
    # ``track_01`` is a per-recording counter, so two recordings' ``track_01`` with different
    # settings would merge (embedding's ``keypoint_track_identity`` is the same rule).
    identity_fields = ("tracking_identity", "match", "method", "config")
    lineage_ref = ("keypoint_detect", "detect_run_id")
    # §6.5 auxiliary input: the subjects the poses are assigned to (§26 *Assignment*). Makes a
    # keypoint track run a dependent of its tracking run, and resolves a batch
    # ``tracking_spec`` per recording.
    aux_refs = (("tracking", "tracking_run_id"),)
    # Offered from a keypoint detect run only: from a tracking run it would be a button with
    # nothing to run until a keypoint detect run exists.
    offered_after = ("keypoint_detect",)

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run(job, run_dir, progress_cb, stop_event)

    def run_dir_for(self, experiment_root, job):
        return keypoint_track_run_dir(experiment_root, job)

    def remote_resources(self, job) -> dict:
        """One GPU, the fewest Tillicum accepts: the work is one CPU core per recording, and
        the node's CPUs and memory scale down with it."""
        return {"gpus": 1}

    def recording_parallelism(self) -> int:
        """Recordings of a local run in flight at once: each is its own one-core worker."""
        return local_parallelism()

    def output_path(self, run_id: str) -> str:
        return f"keypoint_track/{run_id}/"

    def estimate_output_bytes(
        self, recording_dir, recording_data: dict, config: dict
    ) -> int:
        """At most its detect run's size (#743): a subject's track part has a row per
        subject-frame and fewer columns than an instance (no candidates, no extras), so the
        detect run on disk bounds it.
        """
        from detail.core import disk_space

        detect_run_id = config.get("detect_run_id")
        if recording_dir is None or not detect_run_id:
            return 0  # a bare registry probe; a launch estimates after `validate`
        return disk_space.dir_bytes(
            keypoint_runs.detect_run_dir(Path(recording_dir), detect_run_id)
        )

    def validate(self, recording_dir, recording_data: dict, config: dict) -> dict:
        """§26 "Validation checks" for the track stage; the tracking run, the match, the
        method and its every setting are resolved and written back, so the record names what
        ran (Tenet 9)."""
        model = KeypointTrackConfig.model_validate(config)
        recording_dir = Path(recording_dir)
        config["compute"] = _resolve_compute(config.get("compute"))
        detect_run_id = config["detect_run_id"]
        entry = {
            r["id"]: r for r in recording_data.get("keypoint_detect_runs", [])
        }.get(detect_run_id)
        if entry is None:
            raise ValueError(
                f"keypoint detect run {detect_run_id!r} not found in recording "
                f"{recording_dir.name!r}"
            )
        if entry.get("status") != "complete":
            raise ValueError(
                f"keypoint detect run {detect_run_id!r} is {entry.get('status')!r}, not "
                "complete"
            )
        # Inherited, never chosen: a track over a teacher detect run is a teacher run (§26).
        config[teacher_window.WINDOW_KEY] = teacher_window.window_of(entry)
        try:
            detect = KeypointDetectRun(recording_dir, detect_run_id)
        except FileNotFoundError as exc:
            raise ValueError(f"keypoint detect run {detect_run_id!r}: {exc}") from None
        for part_id in detect.part_ids():
            detect.reader(part_id)  # refuses a store in the per-track layout
        skeleton = detect.skeleton
        skeleton.validate()
        tracking = _resolve_tracking_run(
            recording_dir, recording_data, detect, model.tracking_run_id
        )
        config["tracking_run_id"] = tracking["id"]
        from detail.api import run_spec

        config["tracking_identity"] = run_spec.spec_of(
            recording_data, "tracking", tracking, recording_dir
        )
        config["match_min_inside"] = float(model.match_min_inside)
        config["match_mask_grow"] = float(model.match_mask_grow)
        config["match"] = {
            "min_inside": float(model.match_min_inside),
            "mask_grow": float(model.match_mask_grow),
        }
        method_id = config.get("method")
        if method_id is None:
            from detail.models.resolve import keypoint_method_class

            model_cls = keypoint_method_class(
                recording_dir.parent.parent, detect.meta.get("model_id")
            )
            method_id = getattr(model_cls, "default_track_method", None)
            if method_id is None:
                raise ValueError(
                    f"the detect run's method {detect.meta.get('model_id')!r} names no "
                    f"default track method; choose one of {sorted(KEYPOINT_TRACK_METHODS)}"
                )
        method_cls = KEYPOINT_TRACK_METHODS.get(method_id)
        if method_cls is None:
            raise ValueError(
                f"unknown track method {method_id!r}; available: "
                f"{sorted(KEYPOINT_TRACK_METHODS)}"
            )
        if method_cls.needs_candidates and detect.n_candidates <= 1:
            fits = sorted(
                m
                for m, cls in KEYPOINT_TRACK_METHODS.items()
                if not cls.needs_candidates
            )
            raise ValueError(
                f"track method {method_id!r} chooses among candidates, but detect run "
                f"{detect_run_id!r} stored one per point (n_candidates = "
                f"{detect.n_candidates}); use one of {fits}"
            )
        method_config = method_cls.config_model.model_validate(
            config.get("config") or {}
        )
        config["config"] = method_config.model_dump()
        config["config"]["track_points"] = resolve_track_points(
            skeleton, method_config.track_points
        )
        if not recording_data.get("video", {}).get("fps_actual"):
            raise ValueError(
                "the recording has no fps_actual, so a window in seconds cannot be resolved "
                "to rows; import the video's timestamps first"
            )
        config["method"] = method_id
        config["skeleton"] = skeleton.to_dict()
        config["skeleton_digest"] = skeleton.digest()
        # carried on from the detect run, so the skeleton library still sees this run
        ref = detect.meta.get("skeleton_ref")
        config["skeleton_ref"] = None if ref is None else dict(ref)
        return config

    def build_job(
        self, recording_id: str, run_id: str, config: dict
    ) -> KeypointTrackJob:
        return KeypointTrackJob(
            job_id=run_id,
            recording_id=recording_id,
            track_run_id=run_id,
            detect_run_id=config["detect_run_id"],
            tracking_run_id=config["tracking_run_id"],
            match=dict(config["match"]),
            method=config["method"],
            config=dict(config["config"]),
            skeleton=dict(config["skeleton"]),
            created_at=utc_timestamp(),
            skeleton_ref=config.get("skeleton_ref"),
            compute=config["compute"],
        )

    def extra_record_fields(self, config: dict) -> dict:
        match = config.get("match") or {
            "min_inside": config.get("match_min_inside"),
            "mask_grow": config.get("match_mask_grow"),
        }
        compute = config.get("compute")
        return {
            "detect_run_id": config["detect_run_id"],
            "tracking_run_id": config.get("tracking_run_id"),
            "tracking_identity": config.get("tracking_identity"),
            "match": dict(match),
            "method": config.get("method"),
            "config": dict(config.get("config") or {}),
            "skeleton_digest": config.get("skeleton_digest"),
            teacher_window.WINDOW_KEY: config.get(teacher_window.WINDOW_KEY),
            # Known at reservation time — where it will run, for the Where column.
            "compute": compute.to_dict() if compute is not None else None,
        }


def local_parallelism() -> int:
    """How many recordings a local keypoint track run works on at once: ``PARALLELISM_ENV``
    when set, else half this machine's cores, capped.

    Measured 2026-11-03 on a 16-core WSL box, 8 recordings of Agression_BMAD_Full (116k
    subject-frames), min of 2: 202 s at 1, 114 s at 2, 64 s at 4, 50 s at 8, ~380 MB per
    worker. At 8 the wall is about the slowest recording's own time."""
    raw = os.environ.get(PARALLELISM_ENV)
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, min(_MAX_LOCAL_PARALLELISM, (os.cpu_count() or 1) // 2))


def _resolve_compute(compute: ComputeTarget | None) -> ComputeTarget:
    """Where the run goes, written back (Tenet 9): unset is this machine's CPU; a remote
    target must be registered here. A GPU backend is refused rather than read as the CPU, so a
    caller who asked for one learns the stage cannot use it."""
    if compute is None:
        return ComputeTarget("cpu", None, None, None)
    if compute.backend == "remote":
        remote_targets.require_registered(compute)
        return compute
    if compute.backend != "cpu":
        raise ValueError(
            f"keypoint track runs on the CPU and loads no model, so backend "
            f"{compute.backend!r} has nothing to run on; use one of {list(_BACKENDS)}"
        )
    return compute


def _resolve_tracking_run(
    recording_dir: Path, recording_data: dict, detect, tracking_run_id: str | None
) -> dict:
    """The tracking run whose subjects the poses go to: the one named, or **the one**
    complete run on the detect run's detection run. Several are refused, naming them —
    "the first" would make the same command assign differently tomorrow (Tenet 9)."""
    runs = list(recording_data.get("tracking_runs", []))
    detection_run_id = detect.detection_run_id
    if tracking_run_id is not None:
        entry = next((r for r in runs if r.get("id") == tracking_run_id), None)
        if entry is None:
            raise ValueError(
                f"tracking run {tracking_run_id!r} not found in recording "
                f"{recording_dir.name!r}"
            )
        if entry.get("status") != "complete":
            raise ValueError(
                f"tracking run {tracking_run_id!r} is {entry.get('status')!r}, not complete"
            )
        if entry.get("detection_run_id") != detection_run_id:
            raise ValueError(
                f"tracking run {tracking_run_id!r} is on detection run "
                f"{entry.get('detection_run_id')!r}, but keypoint detect run "
                f"{detect.run_id!r} is on {detection_run_id!r}; choose a tracking run on "
                f"{detection_run_id!r}"
            )
        return entry
    fits = [
        r
        for r in runs
        if r.get("status") == "complete"
        and r.get("detection_run_id") == detection_run_id
    ]
    if not fits:
        raise ValueError(
            f"no complete tracking run on detection run {detection_run_id!r} (the one "
            f"keypoint detect run {detect.run_id!r} read); run tracking on it first"
        )
    if len(fits) > 1:
        raise ValueError(
            f"{len(fits)} complete tracking runs are on detection run "
            f"{detection_run_id!r}: {describe_tracking_choice(fits)}"
        )
    return fits[0]


def describe_tracking_choice(entries: list[dict]) -> str:
    """Several tracking runs a pose could be given to, as a refusal names them: each by
    what it follows, then which to choose. Runs that each follow one class give the poses
    to that animal alone, so a person tracking animals by class is told the merge is what
    gives them to every animal — and which run that is, when it exists."""
    entries = sorted(entries, key=lambda r: str(r["id"]))
    ids = ", ".join(
        f"{r['id']} ({r['class_name']})" if r.get("class_name") else str(r["id"])
        for r in entries
    )
    merged = [str(r["id"]) for r in entries if r.get("tracker_id") == "track_merge"]
    by_class = {
        str(r["class_name"])
        for r in entries
        if r.get("class_name") and r.get("tracker_id") != "track_merge"
    }
    if merged:
        return (
            f"{ids}; choose one as the tracking run — {', '.join(merged)} merges "
            "several animals' tracks and gives the poses to each of them"
        )
    if len(by_class) > 1:
        return (
            f"{ids}; each follows one class, so the poses would go to that animal "
            "alone — to give them to every animal, combine these with Merge tracks "
            "(track_merge) and choose the merged run as the tracking run"
        )
    return f"{ids}; choose one as the tracking run"
