"""Preview keypoint tracking (§26 *Previewing a track method*): a few settings over a few
seconds of a few recordings, side by side, in seconds rather than a full run's minutes.

**Where a track run's time goes, and why a preview is shaped this way.** Measured on a 16-minute
two-animal recording (Agression_BMAD_Full recording_008, kd_02): *Assignment* — decoding each
subject's mask and matching the poses to it — is 2.5–5 ms per subject-frame; the method's path and
filters together are about 0.5 ms. Assignment is 83–91% of the run and depends on no method
setting. So a preview assigns each window **once** and runs every setting on those rows; the path
is decoded once per path setting and filtered once per setting, as *Tune tracking* does.

**A preview is the stage over a window, not an imitation of it.** The rows come from the stage's
own reader (``keypoint_track.subject_rows``, which *Tune tracking* shares) and the settings run
through the stage's own method; each recording's config is the stage's own ``validate``'s, so the
tracking run, the method and ``track_points`` resolve exactly as a launch would. The path is a
whole-series decode, so the window is decoded with :data:`PREVIEW_MARGIN_S` either side and then
cut back: inside the window the points are the full run's up to the path settling differently
near a cut that far away. Over a whole part with no window, they are the full run's
(``tests/pipeline/test_keypoint_track_preview.py``).

**What a preview reports is description, not a ranking** (``analysis.keypoint_track_diagnostics``):
no motion figure can say which setting is right, because smoothing improves each of them while it
pulls points off the animal. Detection alone is always the first row — the floor each figure is
read against. Derived, never stored.

**The grid search** (``grid=True``) is the same run over every ``TUNE_GRID`` setting, the grid
*Tune tracking* scores against labels, and marks the **front**: the settings no other setting
beats on agreement with confident detections, placed, jumps and swaps together
(``diagnostics.pareto_front``). It is a provisional rule, not yet checked against labels, and
every surface says so. Assignment is still once per window, and the path is decoded once per
transition scale, so the 36 settings cost little more than the four of a preview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.analysis import keypoint_track_diagnostics as diagnostics
from detail.configs.keypoint_track import (
    PREVIEW_MARGIN_S,
    PREVIEW_MAX_RECORDINGS,
    PREVIEW_MAX_VARIANTS,
    PREVIEW_VARIANT_FIELDS,
    TUNE_METHOD,
    KeypointTrackPreviewWindow,
    tune_settings,
)
from detail.core.progress import ThrottledProgress
from detail.core.registries import KEYPOINT_TRACK_METHODS
from detail.core.skeleton import Skeleton
from detail.pipeline import keypoint_track
from detail.pipeline.keypoint_track_tune import path_key
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import KeypointDetectRun

DETECTION_ONLY = "detection only"
CURRENT = "current settings"
#: The pooled figures' key beside each recording's own.
ALL = "all"
_NS_PER_S = 1_000_000_000


@dataclass
class PreviewSource:
    """One recording to preview: its directory, the stage config ``KeypointTrackModule.validate``
    resolved for it, and the window asked for."""

    recording_id: str
    recording_dir: Path
    config: dict
    window: KeypointTrackPreviewWindow


@dataclass
class PreviewWindow:
    """Where a preview ran on one recording: ``[start_ns, end_ns)`` of one part."""

    recording_id: str
    part_id: str
    part_index: int
    start_ns: int
    end_ns: int
    detect_run_id: str
    tracking_run_id: str
    rows_per_second: float
    frame_wh: tuple[int, int]

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / _NS_PER_S


@dataclass
class PreviewTrack:
    """One subject under one setting, inside the window: ``points`` ``(M, K, 2)`` normalized
    full-frame (NaN: not placed), ``scores`` ``(M, K)``."""

    frame_idx: np.ndarray
    t_ns: np.ndarray
    points: np.ndarray
    scores: np.ndarray


@dataclass
class PreviewSetting:
    """One column of the preview: ``overrides`` on the current settings (None: detection
    alone, no tracking), each recording's subjects, and the figures per recording and
    :data:`ALL` (``{metric: value}``, ``analysis.keypoint_track_diagnostics.METRICS``).
    """

    name: str
    overrides: dict | None
    tracks: dict[str, dict[int, PreviewTrack]] = field(default_factory=dict)
    figures: dict[str, dict[str, float | None]] = field(default_factory=dict)


@dataclass
class PreviewResult:
    """What a preview found; derived, never stored."""

    method: str
    base_config: (
        dict  # the current settings, as the first recording's validate resolved them
    )
    skeleton: Skeleton
    tracked: tuple[str, ...]  # the points the figures cover
    windows: list[PreviewWindow]
    settings: list[PreviewSetting]
    grid: bool = False  # a grid search: every TUNE_GRID setting, and the front marked
    front: list[str] = field(default_factory=list)  # the front's names, grid order

    def setting(self, name: str) -> PreviewSetting:
        return next(s for s in self.settings if s.name == name)

    def ranked(self) -> list[PreviewSetting]:
        """Detection alone and the current settings, then the front by agreement (best
        first, grid order on a tie), then the rest in grid order: the order a grid search is
        shown in. A preview's settings as they ran."""
        if not self.grid:
            return list(self.settings)
        fixed, rest = self.settings[:2], self.settings[2:]
        on = [s for s in rest if s.name in self.front]
        on.sort(key=lambda s: -(s.figures[ALL].get("agreement") or 0.0))
        return fixed + on + [s for s in rest if s.name not in self.front]

    def config_of(self, name: str) -> dict:
        """The full method config a setting ran with (the current settings for detection
        alone, which ran none)."""
        return {**self.base_config, **(self.setting(name).overrides or {})}


def variant_name(overrides: dict) -> str:
    """``transition 3, smooth off, despike off`` — the variant fields it sets, in field order."""
    parts = []
    for name in PREVIEW_VARIANT_FIELDS:
        if name not in overrides:
            continue
        value = overrides[name]
        if name == "transition_sigma_body":
            parts.append(f"transition {value:g}")
        elif name == "smooth":
            parts.append("smooth on" if value else "smooth off")
        elif name == "smooth_window_s":
            parts.append(f"smooth window {value:g} s")
        elif name == "despike":
            parts.append("despike on" if value else "despike off")
    return ", ".join(parts) or "same as current"


def resolve_window(
    recording_dir: Path, recording: dict, window: KeypointTrackPreviewWindow
) -> tuple[int, str, int, int]:
    """``(part_index, part_id, start_ns, end_ns)``: the part ``window.start_s`` falls in (the
    middle of the recording when it is None), cut at that part's end.

    Raises:
        ValueError: The start is past the end of the recording's video.
    """
    parts = recording_store.video_files(recording)
    spans = []
    for index, part in enumerate(parts):
        t_ns = recording_store.load_active_timestamps_cached(
            recording_dir, recording, index
        ).values
        if len(t_ns):
            spans.append((index, recording_store.part_stem(part), t_ns))
    if not spans:
        raise ValueError(f"recording {recording_dir.name!r} has no frames to preview")
    duration_ns = int(round(window.duration_s * _NS_PER_S))
    last_ns = int(spans[-1][2][-1])
    if window.start_s is None:
        start_ns = max(0, (last_ns - duration_ns) // 2)
    else:
        start_ns = int(round(window.start_s * _NS_PER_S))
    for index, part_id, t_ns in spans:
        if start_ns <= int(t_ns[-1]):
            start_ns = max(start_ns, int(t_ns[0]))
            return (
                index,
                part_id,
                start_ns,
                min(start_ns + duration_ns, int(t_ns[-1]) + 1),
            )
    raise ValueError(
        f"a preview starting at {start_ns / _NS_PER_S:g} s is past the end of recording "
        f"{recording_dir.name!r} ({last_ns / _NS_PER_S:.0f} s of video)"
    )


def preview(
    sources: list[PreviewSource],
    variants: list[dict] | None = None,
    *,
    grid: bool = False,
    progress=None,
) -> PreviewResult:
    """Run detection alone, the current settings and each of ``variants`` (overrides on the
    current settings, at most :data:`PREVIEW_MAX_VARIANTS`) over each source's window, and
    measure each; with ``grid``, every ``tune_settings()`` setting in place of ``variants``,
    and the front marked. ``progress(stage, fraction, detail)`` hears ``assigning`` frame by
    frame, then ``tracking`` subject by subject and setting by setting, then ``measuring``.

    Raises:
        ValueError: No source, more than :data:`PREVIEW_MAX_RECORDINGS`, more than
            :data:`PREVIEW_MAX_VARIANTS` variants, variants with ``grid``, a grid search on a
            method other than ``TUNE_METHOD``, sources resolved to different methods, or a
            variant the method's model refuses.
    """
    variants = list(variants or [])
    if grid and variants:
        raise ValueError("a grid search tries its own settings; pass no variants")
    # Every frame reports, and the throttle forwards a few a second; each phase's first and
    # last report always go through, so the label never lags its phase.
    throttled = ThrottledProgress(None if progress is None else lambda e: progress(*e))

    def report(stage: str, fraction: float, detail: str, *, force=False) -> None:
        throttled((stage, fraction, detail), force=force)

    if not sources:
        raise ValueError("choose at least one recording to preview")
    if len(sources) > PREVIEW_MAX_RECORDINGS:
        raise ValueError(
            f"a preview covers at most {PREVIEW_MAX_RECORDINGS} recordings; got "
            f"{len(sources)}"
        )
    if len(variants) > PREVIEW_MAX_VARIANTS:
        raise ValueError(
            f"a preview tries at most {PREVIEW_MAX_VARIANTS} settings beside the current "
            f"ones; got {len(variants)}"
        )
    methods = {s.config["method"] for s in sources}
    if len(methods) > 1:
        raise ValueError(
            f"the recordings resolved to different track methods ({sorted(methods)}); "
            "choose one method"
        )
    method_id = methods.pop()
    if grid:
        if method_id != TUNE_METHOD:
            raise ValueError(
                f"the grid search covers {TUNE_METHOD} settings; these settings use "
                f"{method_id}. Choose {TUNE_METHOD} to search"
            )
        variants = tune_settings()
    method = KEYPOINT_TRACK_METHODS[method_id]()
    model = method.config_model
    settings = [PreviewSetting(DETECTION_ONLY, None), PreviewSetting(CURRENT, {})]
    for overrides in variants:
        unknown = sorted(set(overrides) - set(model.model_fields))
        if unknown:
            raise ValueError(f"{method_id} has no setting {', '.join(unknown)}")
        model.model_validate({**sources[0].config["config"], **overrides})
        settings.append(PreviewSetting(variant_name(overrides), dict(overrides)))

    # -- where each window lies, and how much assigning it is ----------------------- #
    windows: list[PreviewWindow] = []
    detects: dict[str, KeypointDetectRun] = {}
    margins: dict[str, tuple[int, int]] = {}
    totals: dict[str, int] = {}
    margin_ns = int(round(PREVIEW_MARGIN_S * _NS_PER_S))
    for source in sources:
        recording = recording_store.read_recording(source.recording_dir)
        index, part_id, start_ns, end_ns = resolve_window(
            source.recording_dir, recording, source.window
        )
        detect = KeypointDetectRun(source.recording_dir, source.config["detect_run_id"])
        rows_per_second, frame_wh = keypoint_track.rate_and_size(
            source.recording_dir, detect
        )
        window = PreviewWindow(
            source.recording_id,
            part_id,
            index,
            start_ns,
            end_ns,
            source.config["detect_run_id"],
            source.config["tracking_run_id"],
            rows_per_second,
            frame_wh,
        )
        windows.append(window)
        detects[source.recording_id] = detect
        margins[source.recording_id] = (start_ns - margin_ns, end_ns + margin_ns)
        totals[source.recording_id] = keypoint_track.window_subject_frames(
            source.recording_dir,
            detect,
            window.tracking_run_id,
            part_id,
            margins[source.recording_id],
        )

    # -- assign each window once ------------------------------------------------------ #
    total = sum(totals.values())
    done = {"frames": 0}
    report("assigning", 0.0, f"{total} subject-frames", force=True)

    def heard(n_subjects: int) -> None:
        done["frames"] += n_subjects
        report(
            "assigning",
            done["frames"] / total if total else 1.0,
            f"{done['frames']}/{total} subject-frames",
        )

    assigned: dict[str, dict] = {}
    for source, window in zip(sources, windows):
        match = source.config["match"]
        assigned[source.recording_id] = keypoint_track.subject_rows(
            source.recording_dir,
            detects[source.recording_id],
            window.tracking_run_id,
            window.part_id,
            {"min_inside": match["min_inside"], "mask_grow": match["mask_grow"]},
            window_ns=margins[source.recording_id],
            on_frame=heard,
        )
    report("assigning", 1.0, f"{total}/{total} subject-frames", force=True)

    # -- every setting over every subject, weighed by its rows ------------------------ #
    work = sum(
        len(rows["frame_idx"])
        for subjects in assigned.values()
        for rows, _scale in subjects.values()
    ) * len(settings)
    tracked_done = 0
    report("tracking", 0.0, f"{len(settings)} settings", force=True)
    tallies: dict[str, dict[str, diagnostics.Tally]] = {s.name: {} for s in settings}
    skeleton = detects[sources[0].recording_id].skeleton
    for source, window in zip(sources, windows):
        detect = detects[source.recording_id]
        base = source.config["config"]
        configs = [
            (
                None
                if s.overrides is None
                else model.model_validate({**base, **s.overrides})
            )
            for s in settings
        ]
        tracked = keypoint_track.tracked_columns(
            detect.skeleton, model.model_validate(base)
        )
        frame_wh = np.asarray(window.frame_wh, dtype=np.float64)
        for track_id, (rows, scale) in sorted(assigned[source.recording_id].items()):
            paths: dict[tuple, tuple] = {}
            keep = (rows["t_ns"] >= window.start_ns) & (rows["t_ns"] < window.end_ns)
            detected_px = np.asarray(rows["points"], dtype=np.float64)[keep] * frame_wh
            detected_scores = np.asarray(rows["point_scores"], dtype=np.float64)[keep]
            for setting, config in zip(settings, configs):
                points, scores = _run(
                    method, rows, scale, detect.skeleton, config, window, paths
                )
                setting.tracks.setdefault(source.recording_id, {})[int(track_id)] = (
                    PreviewTrack(
                        frame_idx=np.asarray(rows["frame_idx"])[keep],
                        t_ns=np.asarray(rows["t_ns"], dtype=np.int64)[keep],
                        points=np.asarray(points, dtype=np.float32)[keep],
                        scores=np.asarray(scores, dtype=np.float32)[keep],
                    )
                )
                tally = diagnostics.tally(
                    np.asarray(points, dtype=np.float64)[keep] * frame_wh,
                    np.asarray(rows["t_ns"], dtype=np.int64)[keep],
                    np.asarray(scale, dtype=np.float64)[keep],
                    detect.skeleton,
                    tracked,
                    window.rows_per_second,
                    detected_px=detected_px,
                    detected_scores=detected_scores,
                )
                tallies[setting.name].setdefault(
                    source.recording_id, diagnostics.Tally()
                ).add(tally)
                tracked_done += len(rows["frame_idx"])
                report(
                    "tracking",
                    tracked_done / work if work else 1.0,
                    f"{source.recording_id} track {track_id}: {setting.name}",
                )

    # -- the figures, per recording and pooled ---------------------------------------- #
    report("tracking", 1.0, f"{len(settings)} settings", force=True)
    for s, setting in enumerate(settings):
        report("measuring", s / len(settings), setting.name, force=s == 0)
        pooled = diagnostics.Tally()
        for recording_id, tally in tallies[setting.name].items():
            setting.figures[recording_id] = tally.figures()
            pooled.add(tally)
        setting.figures[ALL] = pooled.figures()
    report("measuring", 1.0, f"{len(settings)} settings", force=True)
    base = sources[0].config["config"]
    tracked_names = tuple(
        name
        for name, on in zip(
            skeleton.names,
            keypoint_track.tracked_columns(skeleton, model.model_validate(base)),
        )
        if on
    )
    return PreviewResult(
        method=method_id,
        base_config=dict(base),
        skeleton=skeleton,
        tracked=tracked_names,
        windows=windows,
        settings=settings,
        grid=grid,
        front=(
            diagnostics.pareto_front({s.name: s.figures[ALL] for s in settings[1:]})
            if grid
            else []
        ),
    )


def _run(method, rows, scale, skeleton, config, window: PreviewWindow, paths: dict):
    """``(points, scores)`` normalized: detection's own when ``config`` is None, else the
    method's — a path decoded once per path setting (``paths``) and filtered per setting.
    """
    if config is None:
        return rows["points"], rows["point_scores"]
    decode = getattr(method, "decode", None)
    if decode is None:
        part = method.track(
            rows,
            skeleton,
            config,
            rows_per_second=window.rows_per_second,
            frame_wh=window.frame_wh,
            body_scale_px=scale,
        )
        return part.points, part.point_scores
    key = path_key(config)
    if key not in paths:
        paths[key] = decode(
            rows,
            skeleton,
            config,
            rows_per_second=window.rows_per_second,
            frame_wh=window.frame_wh,
            body_scale_px=scale,
        )
    decoded = paths[key]
    part = method.finish(
        rows,
        decoded[0].copy(),
        decoded[1].copy(),
        decoded[2],
        config,
        window.rows_per_second,
        window.frame_wh,
        scale,
        keypoint_track.tracked_columns(skeleton, config),
    )
    return part.points, part.point_scores
