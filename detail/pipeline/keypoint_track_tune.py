"""Tune keypoint tracking (§26 *Tuning a track method*): which ``temporal_decode`` settings keep
a detect run's accuracy, scored against an example set's labels.

A track method's defaults are policy, not measurement (§26 open question 6), and the right
transition scale depends on the footage: on 25 fps oblique hot-plate video the default lost a
third of the paws that detection had placed, measured against hand labels. Tuning
re-runs the method **in memory** over the video parts that hold labelled frames, once per
setting in ``configs.keypoint_track.TUNE_GRID`` on top of the config the user has set, and scores
each against the labels with the audit scorer (``analysis.keypoint_audit.score``), detection
alone beside them. Nothing is written.

**Labels choose a setting; the choice is checked on labels it did not see.** The labelled
recordings are split in two by a seeded shuffle, balanced by frame count: the setting with the
best ``pck10`` over all points on the *pick* half is the recommendation, and the *check* half
scores it, detection and the user's current settings. The check figures are the ones to quote;
the pick figures are a best-of-``len(TUNE_GRID)`` and optimistic by construction. With labels in
only one recording there is no check half, and the result says so.

**Then which points to leave as detected** (``points_to_track``): the chosen setting against
detection alone, per pair or single point, on the pick half; the points detection placed better
are left out of ``track_points`` in one more scored setting, which is the recommendation only if
it also wins on the pick half.

**The detect run's own examples are never scored** (CLAUDE.md, *Labels score a method; they
never build one*): the example set the detect run learned from is refused outright, and any of
its frames that also sits in the chosen set is excluded and counted.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.analysis import keypoint_audit as scoring
from detail.configs.keypoint_track import (
    METHOD_CONFIGS,
    KeypointTrackConfig,
    TUNE_METHOD,
    tune_settings,
)
from detail.core.examples import ExampleSet
from detail.core.registries import KEYPOINT_TRACK_METHODS
from detail.pipeline import keypoint_track
from detail.storage import dirscan
from detail.storage import example_sets
from detail.storage import experiment as experiment_store
from detail.storage import keypoint_runs
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import KeypointDetectRun

DETECTION_ONLY = "detection only"
CURRENT = "current settings"
SPLIT_SEED = 0
#: The figures a setting carries per half: pck at both radii, over all points and the
#: paired / unpaired groups (paws / midline on a quadruped skeleton), and per keypoint.
SCORED_METRICS = ("pck10", "pck05")
#: The ``TemporalDecodeConfig`` fields the decoded path depends on; the rest are filters.
PATH_SETTINGS = (
    "transition_sigma_body",
    "missing_penalty",
    "mirror_weight",
    "decode_cutoff",
    "track_points",
)


@dataclass
class TuneSetting:
    """One row of the table: a name, the overrides on the base config (None: detection alone,
    no tracking), and its scores per half: ``{half: {"group:all": {...}, "keypoint:nose":
    {...}}}``, each figure ``{metric: {value, ci_low, ci_high, n, floor}}``."""

    name: str
    overrides: dict | None
    scores: dict[str, dict] = field(default_factory=dict)

    def figure(self, half: str, scope: str, name: str, metric: str = "pck10"):
        return self.scores.get(half, {}).get(f"{scope}:{name}", {}).get(metric)


@dataclass
class TuneResult:
    """What tuning found; derived, never stored."""

    detect_runs: dict[str, str]  # recording id -> its detect run id, labelled ones only
    example_set_id: str
    method: str
    base_config: dict  # the config every setting's overrides apply on
    pick_recordings: list[str]
    check_recordings: list[str]  # empty: no check half (labels in one recording)
    coverage: dict[str, int]
    settings: list[TuneSetting]
    best: str  # the name of the setting chosen on the pick half

    def setting(self, name: str) -> TuneSetting:
        return next(s for s in self.settings if s.name == name)

    def best_config(self) -> dict:
        """The full method config to run with: the base plus the best setting's overrides."""
        return {**self.base_config, **(self.setting(self.best).overrides or {})}

    @property
    def checked(self) -> bool:
        return bool(self.check_recordings)


def setting_name(overrides: dict) -> str:
    """``transition 3, smooth off, despike off`` — what a row says it is."""
    smooth = (
        f"smooth {overrides['smooth_window_s']:g} s"
        if overrides.get("smooth")
        else "smooth off"
    )
    despike = "despike on" if overrides.get("despike") else "despike off"
    return f"transition {overrides['transition_sigma_body']:g}, {smooth}, {despike}"


def _report(progress, stage: str, fraction: float, detail: str) -> None:
    if progress is not None:
        progress(stage, fraction, detail)


def _complete(recording_dir: Path, detect_run_id: str) -> bool:
    data = recording_store.read_recording(recording_dir)
    entry = {r["id"]: r for r in data.get("keypoint_detect_runs", []) or []}.get(
        detect_run_id
    )
    return entry is not None and entry.get("status") == "complete"


def split_recordings(
    frames_per_recording: dict[str, int], seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """``(pick, check)``: the recordings dealt largest first, each to the half with fewer
    frames so far (pick on a tie), the order among equal counts shuffled by ``seed``. A whole
    recording goes to one half, so the check half's frames come from videos the choice never
    saw. One recording: all pick, no check."""
    ids = sorted(frames_per_recording)
    if len(ids) < 2:
        return ids, []
    shuffled = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    order = sorted(shuffled, key=lambda r: -frames_per_recording[r])  # stable
    pick: list[str] = []
    check: list[str] = []
    totals = [0, 0]
    for recording_id in order:
        side = 0 if totals[0] <= totals[1] else 1
        (pick if side == 0 else check).append(recording_id)
        totals[side] += frames_per_recording[recording_id]
    return sorted(pick), sorted(check)


def tune(
    experiment_dir: Path | str,
    detect_runs: dict[str, str],
    example_set_id: str,
    *,
    tracking_runs: dict[str, str],
    match: dict | None = None,
    base_config: dict | None = None,
    seed: int = SPLIT_SEED,
    progress=None,
) -> TuneResult:
    """Score every tuning setting of ``temporal_decode`` over each recording's detect run
    (``detect_runs``: recording id -> run id; a recording missing from it is not covered)
    against ``example_set_id``'s labels. The instances reach the subjects the labels name
    through ``tracking_runs`` (recording id -> tracking run id, on the detect run's detection
    run) by the stage's own *Assignment* with ``match`` (``{min_inside, mask_grow}``; None takes
    ``KeypointTrackConfig``'s defaults), so a tuned setting meets the rows a run will.

    ``base_config`` is the track config the overrides apply on (the form's current settings);
    None takes the method's defaults. ``progress(stage, fraction, detail)`` hears reading,
    tracking and scoring.

    Raises:
        ValueError: The example set is the detect run's own; no labelled frame of the set lies
            in a recording with this detect run complete; the detect run stored one candidate
            per point (``temporal_decode`` needs several).
        FileNotFoundError: No such example set.
    """
    experiment_dir = Path(experiment_dir)
    labels = example_sets.read_set(experiment_dir, example_set_id)
    config_model = METHOD_CONFIGS[TUNE_METHOD]
    base = config_model.model_validate(base_config or {}).model_dump()
    method = KEYPOINT_TRACK_METHODS[TUNE_METHOD]()
    if match is None:
        fields = KeypointTrackConfig.model_fields
        match = {
            "min_inside": float(fields["match_min_inside"].default),
            "mask_grow": float(fields["match_mask_grow"].default),
        }

    labelled = [i for i in range(len(labels.frames)) if labels.labeled(i)]
    coverage = {
        "labelled": len(labelled),
        "excluded": 0,
        "uncovered": 0,
        "scored": 0,
    }
    by_recording: dict[str, list[int]] = {}
    detects: dict[str, KeypointDetectRun | None] = {}
    sources_read: set[str] = set()
    learned_from: set[tuple] = set()  # frame keys of the detect runs' own example sets
    for i in labelled:
        recording_id = labels.frames[i].recording_id
        if recording_id not in detects:
            rec_dir = experiment_store.recording_path(experiment_dir, recording_id)
            detect_run_id = detect_runs.get(recording_id)
            if (
                detect_run_id is None
                or recording_id not in tracking_runs
                or not keypoint_runs.detect_run_dir(rec_dir, detect_run_id).is_dir()
                or not _complete(rec_dir, detect_run_id)
            ):
                detects[recording_id] = None
            else:
                detect = KeypointDetectRun(rec_dir, detect_run_id)
                # Every set the run learned from — one per class for a per-class run.
                owns = sorted(
                    snap["id"]
                    for snap in keypoint_runs.detect_example_sets(detect.meta).values()
                    if snap.get("id")
                )
                if example_set_id in owns:
                    raise ValueError(
                        f"example set {example_set_id!r} is one detect run "
                        f"{detect_run_id!r} learned from; scoring on it would score the "
                        "answer key. Label frames the detect run did not learn from (another "
                        "example set) and tune on those."
                    )
                if detect.n_candidates <= 1:
                    raise ValueError(
                        f"detect run {detect_run_id!r} stored one candidate per point; "
                        f"{TUNE_METHOD} chooses among several, so there is nothing to tune"
                    )
                for own in owns:
                    if own in sources_read:
                        continue
                    sources_read.add(own)
                    try:
                        source = example_sets.read_set(experiment_dir, own)
                    except FileNotFoundError:
                        source = None
                    if source is not None:
                        learned_from.update(f.key() for f in source.frames)
                detects[recording_id] = detect
        if detects[recording_id] is None:
            coverage["uncovered"] += 1
            continue
        if labels.frames[i].key() in learned_from:
            coverage["excluded"] += 1
            continue
        by_recording.setdefault(recording_id, []).append(i)
    if not by_recording:
        raise ValueError(
            f"no labelled frame of example set {example_set_id!r} lies in a recording with a "
            "complete keypoint detect run of the chosen kind "
            f"({coverage['uncovered']} labelled frames in other recordings, "
            f"{coverage['excluded']} the detect run learned from)"
        )
    coverage["scored"] = sum(len(v) for v in by_recording.values())
    pick, check = split_recordings({r: len(v) for r, v in by_recording.items()}, seed)

    units = sorted(
        {
            (r, int(labels.frames[i].track_id), labels.frames[i].part_id)
            for r, idxs in by_recording.items()
            for i in idxs
        }
    )
    grid = [(DETECTION_ONLY, None), (CURRENT, {})] + [
        (setting_name(o), o) for o in tune_settings()
    ]
    settings = [TuneSetting(name, overrides) for name, overrides in grid]
    runs: dict[str, dict[str, scoring.RunRows]] = {}
    frame_wh: dict[str, tuple[int, int]] = {}
    # One assignment per part, shared by every setting and every subject on it.
    assigned: dict[tuple[str, str], dict] = {}

    def subject_part(recording_id: str, track_id: int, part_id: str):
        key = (recording_id, part_id)
        if key not in assigned:
            assigned[key] = keypoint_track.subject_rows(
                experiment_store.recording_path(experiment_dir, recording_id),
                detects[recording_id],
                tracking_runs[recording_id],
                part_id,
                match,
            )
        return assigned[key].get(int(track_id))

    def track(batch: list[TuneSetting], stage: str) -> None:
        """Re-run ``batch`` over every unit into ``runs``: each part's rows and inputs read
        once, each distinct path decoded once and filtered per setting."""
        configs = [
            (
                None
                if s.overrides is None
                else config_model.model_validate({**base, **s.overrides})
            )
            for s in batch
        ]
        for setting in batch:
            runs[setting.name] = {}
        for u, (recording_id, track_id, part_id) in enumerate(units):
            detect = detects[recording_id]
            rec_dir = experiment_store.recording_path(experiment_dir, recording_id)
            _report(
                progress,
                stage,
                u / len(units),
                f"{recording_id} {part_id} ({u + 1}/{len(units)})",
            )
            rows_per_second, wh = keypoint_track.rate_and_size(rec_dir, detect)
            frame_wh[recording_id] = wh
            found = subject_part(recording_id, track_id, part_id)
            if found is None:
                continue  # the subject is not tracked in this part: its labels go unmatched
            rows, scale = found
            skeleton = detect.skeleton
            paths: dict[tuple, tuple] = {}
            for s, (setting, config) in enumerate(zip(batch, configs)):
                if config is None:
                    points = np.asarray(rows["points"], dtype=np.float32)
                    scores = np.asarray(rows["point_scores"], dtype=np.float32)
                else:
                    key = path_key(config)
                    if key not in paths:
                        paths[key] = method.decode(
                            rows,
                            skeleton,
                            config,
                            rows_per_second=rows_per_second,
                            frame_wh=wh,
                            body_scale_px=scale,
                        )
                    decoded = paths[key]
                    part = method.finish(
                        rows,
                        decoded[0].copy(),
                        decoded[1].copy(),
                        decoded[2],
                        config,
                        rows_per_second,
                        wh,
                        scale,
                        keypoint_track.tracked_columns(skeleton, config),
                    )
                    points, scores = part.points, part.point_scores
                run = runs[setting.name].setdefault(
                    recording_id,
                    scoring.RunRows(
                        run_id=detect_runs[recording_id],
                        tracking_run_id=tracking_runs[recording_id],
                        names=tuple(skeleton.names),
                        parts={},
                    ),
                )
                run.parts[(track_id, part_id)] = scoring.RunPart(
                    t_ns=np.asarray(rows["t_ns"], dtype=np.int64),
                    points=np.asarray(points, dtype=np.float32),
                    scores=np.asarray(scores, dtype=np.float32),
                    visible=np.isfinite(np.asarray(points)).all(axis=-1),
                )
                _report(
                    progress,
                    stage,
                    (u + (s + 1) / len(batch)) / len(units),
                    f"{recording_id} {part_id}: {setting.name}",
                )

    names = list(labels.skeleton.names)
    name_map = {n: n for n in names}
    halves = {"pick": pick, "check": check, "all": pick + check}
    subsets = {
        half: dataclasses.replace(
            labels,
            frames=[
                labels.frames[i] for r in sorted(recs) for i in by_recording.get(r, [])
            ],
        )
        for half, recs in halves.items()
        if recs
    }

    def score(batch: list[TuneSetting]) -> None:
        for s, setting in enumerate(batch):
            _report(progress, "scoring", s / len(batch), setting.name)
            for half, subset in subsets.items():
                report = scoring.score(
                    subset, runs[setting.name], name_map, set(), frame_wh
                )
                setting.scores[half] = _figures(report)
        _report(progress, "scoring", 1.0, f"{len(batch)} settings")

    track(settings, "tracking")
    score(settings)
    best = choose(settings)

    # Which points the best setting should leave as detected: decided on the pick half alone,
    # then scored as one more setting, which is the recommendation only if it too wins there.
    skeleton = next(d for d in detects.values() if d is not None).skeleton
    tracked = [
        n
        for n, on in zip(
            skeleton.names,
            keypoint_track.tracked_columns(
                skeleton, config_model.model_validate({**base, **best.overrides})
            ),
        )
        if on
    ]
    kept = points_to_track(best, settings[0], skeleton, tracked)
    if kept is not None:
        left = [n for n in tracked if n not in kept]
        subset_setting = TuneSetting(
            f"{best.name}, as detected: {', '.join(left)}",
            {**best.overrides, "track_points": kept},
        )
        track([subset_setting], "tracking a subset")
        score([subset_setting])
        settings.append(subset_setting)
        if _pick(subset_setting) >= _pick(best):
            best = subset_setting
    return TuneResult(
        detect_runs={r: detect_runs[r] for r in sorted(by_recording)},
        example_set_id=example_set_id,
        method=TUNE_METHOD,
        base_config=base,
        pick_recordings=pick,
        check_recordings=check,
        coverage=coverage,
        settings=settings,
        best=best.name,
    )


def example_set_choices(
    experiment_dir: Path | str, detect_spec: dict | None
) -> list[dict]:
    """The example sets with labelled frames, each ``{id, name, labelled, usable, reason}``:
    not usable when it is the set the detect runs of ``detect_spec`` learned from (its digest is
    the spec's ``example_set_digest``: scoring on it scores the answer key) or when it was
    labelled on another skeleton (its skeleton's digest is not the spec's ``skeleton_digest``).
    What the Tune tracking window offers, so it never opens on a set it would refuse. Reads
    every set's images to hash them: call it off the UI thread."""
    experiment_dir = Path(experiment_dir)
    identity = (detect_spec or {}).get("identity") or {}
    skeleton = identity.get("skeleton_digest")
    # Every set the runs learned from: the one set's digest, or each class's.
    owns = {
        d
        for d in [
            identity.get("example_set_digest"),
            *(identity.get("class_example_digests") or {}).values(),
        ]
        if d
    }

    def choice(set_id: str) -> dict:
        es = example_sets.read_set(experiment_dir, set_id)
        labelled = es.counts()["labeled"]
        reason = None
        if skeleton and es.skeleton.digest() != skeleton:
            reason = "labelled on another skeleton than the detect run's"
        elif (
            labelled
            and owns
            and es.digest(example_sets.image_hashes(experiment_dir, es)) in owns
        ):
            reason = "the detect run learned from it"
        return {
            "id": es.id,
            "name": es.name,
            "labelled": labelled,
            "usable": reason is None,
            "reason": reason,
        }

    # A listing: one unreadable set costs its row, never the window (dirscan.read_each).
    listed = dirscan.read_each(example_sets.discover_sets(experiment_dir), choice)
    return [c for c in listed if c["labelled"]]


def choose(settings: list[TuneSetting]) -> TuneSetting:
    """The grid setting (not detection alone, not the current settings) with the best ``pck10``
    over all points on the **pick** half; the first in grid order on a tie, and a setting the
    scorer withheld a figure for counts as worst. Reads nothing of the check half: that is what
    keeps the check figures an honest estimate of the choice."""
    tuned = [s for s in settings if s.overrides]
    return max(tuned, key=_pick)


def _pick(setting: TuneSetting, scope: str = "group", name: str = "all") -> float:
    """``setting``'s pick-half PCK@0.1 over ``scope:name``; -1 when withheld."""
    value = (setting.figure("pick", scope, name) or {}).get("value")
    return -1.0 if value is None else value


def points_to_track(
    best: TuneSetting, detection: TuneSetting, skeleton, tracked: list[str]
) -> list[str] | None:
    """Of the points ``best`` tracks, the ones to keep tracking; ``None`` when that is all of
    them (or none, which leaves nothing to track). A unit is a left/right pair whose points
    are both tracked, or a single point otherwise; a unit stays tracked unless detection
    alone placed it better on the **pick** half (the mean of its points' PCK@0.1; a unit the
    scorer withheld a figure for stays tracked, for want of evidence). Reads nothing of the
    check half, like :func:`choose`. Names only: no body plan is assumed."""
    on = set(tracked)
    paired = {}
    for left, right in skeleton.pairs:
        if left in on and right in on:
            paired[left] = paired[right] = (left, right)
    units, seen = [], set()
    for name in skeleton.names:
        if name in on and name not in seen:
            unit = paired.get(name, (name,))
            units.append(unit)
            seen.update(unit)
    kept: list[str] = []
    for unit in units:
        tracked_value = [_pick(best, "keypoint", n) for n in unit]
        detected_value = [_pick(detection, "keypoint", n) for n in unit]
        if min(tracked_value + detected_value) < 0 or np.mean(tracked_value) >= np.mean(
            detected_value
        ):
            kept.extend(unit)
    kept = [n for n in skeleton.names if n in set(kept)]
    if not kept or len(kept) == len(tracked):
        return None
    return kept


def path_key(config) -> tuple:
    """What a decoded path depends on, hashable: a list (``track_points``) as a tuple."""
    return tuple(
        tuple(v) if isinstance(v, list) else v
        for v in (getattr(config, name) for name in PATH_SETTINGS)
    )


def _figures(report: scoring.AuditScores) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in report.rows:
        if row["scope"] in ("group", "keypoint") and row["metric"] in SCORED_METRICS:
            out.setdefault(f"{row['scope']}:{row['name']}", {})[row["metric"]] = {
                k: row.get(k) for k in ("value", "ci_low", "ci_high", "n", "floor")
            }
    return out
