"""The keypoint run viewers (§26 "Surfaces").

A **detect** run is reviewed the way a detection run is — the same frame scrubber, boxes and
masks, the per-frame count strip — over the **detection** run it read (``detection_run_id``),
with **every pose instance** on the frame drawn on top (``KeypointDetectRun.at``), coloured by
role (``core/visualize.py::draw_keypoints``). Since 2026-10-21 a detect run holds no subjects,
so its coverage line is per part: frames walked, frames with at least one pose, poses.

A **track** run is reviewed over the **tracking** run it assigned its poses to (its own
``tracking_run_id``): masks and boxes per subject, each subject's final points, a loss row per
subject, per-point coverage, and the assignment per part — how many poses were given to a
subject and how many were not.

Kept out of ``gui/run_preview.py`` so the keypoint work owns a whole file (§26 *Sequencing*).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QLabel

from detail.api.run_preview import locate_frame
from detail.core import visualize
from detail.core.visualize import draw_keypoints
from detail.gui import frame_source
from detail.gui.run_preview import (
    DetectionRunPreview,
    TrackingRunPreview,
    _TrackTimeline,
)
from detail.gui.status_color import WARN, set_label_color
from detail.storage import recording as recording_store


def _part_stems(recording) -> list[str]:
    """The scrubber's part index → the part id runs are keyed by (``part_001``)."""
    data = recording_store.read_recording(recording.dir)
    return [recording_store.part_stem(f) for f in recording_store.video_files(data)]


def _draw_instances(out, run, part_stems: list[str], part: int, local: int):
    """Every pose instance the detect run placed on this frame, drawn over ``out``."""
    if part >= len(part_stems):
        return out
    found = run.at(part_stems[part], int(local))
    skeleton = run.skeleton
    for points, scores in zip(found["points"], found["point_scores"]):
        out = draw_keypoints(out, points, skeleton, scores=scores)
    return out


def _draw_subject_points(
    out, run, part_stems: list[str], part: int, local: int, detections
):
    """A track run's final points for every subject tracked on this frame."""
    if part >= len(part_stems):
        return out
    stem = part_stems[part]
    skeleton = run.skeleton
    drawn: set[int] = set()
    for det in detections:
        track_id = det.object_id
        if track_id is None or track_id in drawn:
            continue
        drawn.add(int(track_id))
        found = run.at(int(track_id), stem, int(local))
        if found is None:
            continue
        points, scores = found
        out = draw_keypoints(out, points, skeleton, scores=scores)
    return out


def _compute_with_points(view, global_frame: int, draw):
    """Off-thread, for every keypoint viewer: decode, the overlay in the style picked
    (``DetectionRunPreview._overlay``), then ``draw(out, part, local, detections)`` — the
    run's points — over it."""
    if not view._total:
        return ("text", "(no video)")
    part, local = locate_frame(view._part_counts, global_frame)
    result, reason = frame_source.decode(
        lambda: view._preview.frame(
            part, local, with_masks=visualize.preview_needs_mask(view._style)
        )
    )
    if reason is not None:
        return ("text", frame_source.unavailable_text(reason))
    _actual, rgb, detections = result
    out = view._overlay(rgb, detections)
    return ("img", draw(out, part, local, detections))


class KeypointDetectRunPreview(DetectionRunPreview):
    """One recording's ``keypoint_detect`` run: detection's viewer plus every pose."""

    # The box alone, so a mask tint does not sit under the points and share the left side's
    # colour; the picker still offers every style.
    _DEFAULT_STYLE = visualize.BOX_ONLY

    def _make_preview(self, recording, run_id: str):
        # Off-thread (called from _load): the run's record names the detection run whose
        # frames this viewer shows.
        self._run = recording.keypoint_detect_run(run_id)
        self._part_stems = _part_stems(recording)
        return recording.detection_preview(  # off-thread (called from _load)
            self._run.detection_run_id
        )

    def _compute(self, global_frame: int):
        """Off-thread: decode + the detection overlay + every pose on the frame."""
        return _compute_with_points(
            self,
            global_frame,
            lambda out, part, local, _detections: _draw_instances(
                out, self._run, self._part_stems, part, local
            ),
        )

    def _summary_text(self, overview, fps: float | None) -> str:
        """Detection's line, then the run's: per part, the frames it walked, the frames
        with at least one pose and the poses — the numbers a detect run is asked for."""
        lines = [super()._summary_text(overview, fps)]
        meta = self._run.meta
        lines.append(
            f"keypoints: {meta['model_id']} · {len(self._run.skeleton)} points · "
            f"every {meta.get('every_n_frames', 1)} frame(s) · on detection run "
            f"{self._run.detection_run_id}"
            # the frames below are then the ones with a detection of that class
            + (f", {meta['class_name']} only" if meta.get("class_name") else "")
        )
        coverage = self._run.coverage()
        if not coverage:
            lines.append("coverage not recorded yet (the run is still going)")
        for part_id, part in sorted(coverage.items()):
            lines.append(
                f"{part_id.replace('_', ' ')} — {int(part['frames_with_instance']):,}/"
                f"{int(part['frames']):,} frames with a pose · "
                f"{int(part['instances']):,} poses"
            )
        return "\n".join(lines)


class LegacyKeypointsRunPreview(TrackingRunPreview):
    """One recording's run of the stage before §26 (``keypoints``, ``kp_NN``): tracking's
    viewer with the run's final points, read through ``keypoint_runs.LegacyKeypointsRun``.
    Nothing makes such a run any more; this is how an old one is still looked at."""

    _DEFAULT_STYLE = visualize.BOX_ONLY

    def _make_preview(self, recording, run_id: str):
        # Off-thread (called from _load), as the detect viewer's.
        from detail.storage.keypoint_runs import LegacyKeypointsRun

        self._run = LegacyKeypointsRun(recording.dir, run_id)
        data = recording_store.read_recording(recording.dir)
        self._part_stems = [
            recording_store.part_stem(f) for f in recording_store.video_files(data)
        ]
        return recording.tracking_preview(  # off-thread (called from _load)
            self._run.meta["tracking_run_id"]
        )

    def _compute(self, global_frame: int):
        """Off-thread: decode + the tracking overlay + the old run's points per subject."""
        return _compute_with_points(
            self,
            global_frame,
            lambda out, part, local, detections: _draw_subject_points(
                out, self._run, self._part_stems, part, local, detections
            ),
        )

    def _summary_text(self, overview, fps: float | None) -> str:
        meta = self._run.meta
        cutoff = self._run.min_score
        return "\n".join(
            [
                TrackingRunPreview._summary_text(self, overview, fps),
                f"keypoints (before the keypoint detect / track rebuild): "
                f"{meta.get('model_id')} · {len(self._run.skeleton)} points"
                + ("" if cutoff is None else f" · not visible below {cutoff:g}"),
                "a run of the old keypoints stage: it reads and exports, and a new one is "
                "made with keypoint detect and keypoint track",
            ]
        )


def legacy_preview_widget(recording, module_id: str, run_id: str):
    """The viewer for a run of a module this build no longer has, or None — what a run page
    shows where no module UI is registered (hiding governs launching, not reading)."""
    if module_id == "keypoints":
        return LegacyKeypointsRunPreview(recording, run_id)
    return None


#: A point placed on fewer than this share of a track's subject frames raises the alert
#: (§14 *Coverage and alerts*; the threshold is policy, not measured).
POINT_COVERAGE_ALERT = 0.80


class KeypointTrackRunPreview(TrackingRunPreview):
    """One recording's ``keypoint_track`` run: its tracking run's viewer with the final
    points, the assignment per part, a
    second loss row per track — frames the track had a subject but no keypoints — a per-point
    coverage line, and an amber alert naming any point under the coverage threshold with the
    one action that fixes it (§14 *Coverage and alerts*)."""

    # The box alone, so a mask tint does not sit under the points and share the left side's
    # colour; the picker still offers every style.
    _DEFAULT_STYLE = visualize.BOX_ONLY

    def _make_preview(self, recording, run_id: str):
        # Off-thread (called from _load): the tracking run is the track run's own — the one
        # whose subjects its poses were assigned to (§26 *Assignment*).
        self._run = recording.keypoint_track_run(run_id)
        self._part_stems = _part_stems(recording)
        return recording.tracking_preview(  # off-thread (called from _load)
            self._run.meta["tracking_run_id"]
        )

    def _compute(self, global_frame: int):
        return _compute_with_points(
            self,
            global_frame,
            lambda out, part, local, detections: _draw_subject_points(
                out, self._run, self._part_stems, part, local, detections
            ),
        )

    def _load_overview(self, preview):
        """Off-thread: tracking's presence, where each track has no keypoints although it had
        a subject, and each point's coverage."""
        track_ids, presence = preview.track_presence()  # off-thread (called from _load)
        counts = preview.part_frame_counts
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
        names = self._run.skeleton.names
        kp_presence: dict[int, np.ndarray] = {}
        per_point: dict[int, tuple[np.ndarray, int]] = {}
        total = int(sum(counts))
        for track_id in track_ids:
            has = np.zeros(total, dtype=bool)
            hits = np.zeros(len(names), dtype=np.int64)
            n_rows = 0
            for part_id in self._run.part_ids(track_id):
                if part_id not in self._part_stems:
                    continue
                rows = self._run.rows(track_id, part_id)
                finite = np.isfinite(rows["points"]).all(axis=2)  # (M, K)
                offset = int(offsets[self._part_stems.index(part_id)])
                placed = np.asarray(rows["frame_idx"])[finite.any(axis=1)] + offset
                has[placed[placed < total]] = True
                hits += finite.sum(axis=0)
                n_rows += int(finite.shape[0])
            # lost only where the track had a subject and no point was placed
            kp_presence[track_id] = has | ~np.asarray(presence[track_id], dtype=bool)
            per_point[track_id] = (hits, n_rows)
        return (track_ids, presence, kp_presence, per_point)

    def _build_overview(self, layout, overview) -> None:
        track_ids, presence, kp_presence, per_point = overview
        super()._build_overview(layout, (track_ids, presence))
        self._kp_timeline = _TrackTimeline(track_ids, kp_presence, self._total)
        self._kp_timeline.clicked.connect(lambda f: self._scrubber.set_frame(f))
        layout.addWidget(
            QLabel(
                "Frames the track had a subject but no keypoints (click to jump) — taller "
                "means more lost in that slice; empty means none:"
            )
        )
        layout.addWidget(self._kp_timeline)
        names = self._run.skeleton.names
        lines, alerts = [], []
        for track_id in track_ids:
            hits, n_rows = per_point[track_id]
            for name, hit in zip(names, hits.tolist()):
                share = hit / n_rows if n_rows else 0.0
                lines.append(f"track {track_id} · {name} {share:.0%}")
                if share < POINT_COVERAGE_ALERT:
                    alerts.append(f"{name} on track {track_id} ({share:.0%})")
        coverage = QLabel(
            "Per-point coverage (frames with the point placed / frames the track had a "
            "subject): " + (" · ".join(lines) if lines else "no tracks")
        )
        coverage.setWordWrap(True)
        layout.addWidget(coverage)
        self._alert = QLabel("")
        self._alert.setWordWrap(True)
        set_label_color(self._alert, WARN)
        self._alert.setVisible(bool(alerts))
        if alerts:
            self._alert.setText(
                f"{len(alerts)} point(s) under {POINT_COVERAGE_ALERT:.0%} coverage: "
                + ", ".join(alerts[:6])
                + (" …" if len(alerts) > 6 else "")
                + ". Mark more example frames showing them, then run keypoint detect "
                "again; or lower the decode cut-off in a new track run."
            )
        layout.addWidget(self._alert)

    def _set_overview_cursor(self, frame: int) -> None:
        super()._set_overview_cursor(frame)
        self._kp_timeline.set_cursor(frame)

    def _summary_text(self, overview, fps: float | None) -> str:
        track_ids, presence, _kp, _pp = overview
        lines = [super()._summary_text((track_ids, presence), fps)]
        meta = self._run.meta
        lines.append(
            f"keypoints: {meta['method']} over detect run {meta['detect_run_id']} · "
            f"{len(self._run.skeleton)} points · subjects from tracking run "
            f"{meta['tracking_run_id']}"
        )
        # The assignment (§26): how many of the detect run's poses went to a subject.
        for part_id, part in sorted((meta.get("assignment") or {}).items()):
            lines.append(
                f"{part_id.replace('_', ' ')} — {int(part['assigned']):,}/"
                f"{int(part['instances']):,} poses given to a subject · "
                f"{int(part['unassigned']):,} not"
            )
        coverage = self._run.coverage()
        if not coverage:
            lines.append("coverage not recorded yet (the run is still going)")
        for track, parts in sorted(coverage.items()):
            placed = sum(int(p["placed"]) for p in parts.values())
            total = sum(int(p["total"]) for p in parts.values())
            # A track run's coverage counts subject frames given a pose (§26 *Visibility*);
            # whether its points survived the decode is the per-point line below.
            lines.append(
                f"{track.replace('_', ' ')} — {placed:,}/{total:,} subject frames given a "
                "pose"
            )
        return "\n".join(lines)
