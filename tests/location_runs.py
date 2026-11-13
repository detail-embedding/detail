"""Completed location-export runs on disk, for the plot surfaces' tests.

Written through the producer's own writer (``storage/location_export.write_location_tables``)
with its own column list (``analysis/locations.TRACKS_COLUMNS``), so a column the plots read
that the export stops writing fails here rather than on a user's run.
"""

import math

import numpy as np
import yaml

from detail.analysis.locations import (
    KEYPOINT_COLUMNS,
    PAIR_COLUMNS,
    POINT_COLUMNS,
    ROI_COLUMNS,
    TRACKS_COLUMNS,
    LocationTables,
)
from detail.storage import analysis_runs as ar
from detail.storage import location_export


def circle(n: int, *, phase: float = 0.0, radius: float = 0.3):
    """``n`` normalized centroids going once round a circle centred in the frame."""
    angle = np.linspace(0.0, 2 * math.pi, n) + phase
    return 0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)


def tracks_rows(
    x_norm,
    y_norm,
    *,
    resolution=(640, 480),
    pixels_per_mm=None,
    fps: float = 30.0,
    object_id: int = 0,
    frame_offset: int = 0,
    skip: frozenset = frozenset(),
) -> list[tuple]:
    """``tracks.csv`` rows for one subject's path; ``skip`` drops those sample indices (a gap)."""
    w, h = resolution
    rows = []
    for i, (cx, cy) in enumerate(zip(x_norm, y_norm)):
        if i in skip:
            continue
        frame = frame_offset + i
        values = {name: "" for name in TRACKS_COLUMNS}
        values.update(
            frame_idx=frame,
            t_ns=int(round(frame * 1e9 / fps)),
            object_id=object_id,
            bbox_cx_norm=round(float(cx), 6),
            bbox_cy_norm=round(float(cy), 6),
            mask_cx_norm=round(float(cx), 6),
            mask_cy_norm=round(float(cy), 6),
            bbox_cx_px=round(float(cx) * w, 2),
            bbox_cy_px=round(float(cy) * h, 2),
            mask_cx_px=round(float(cx) * w, 2),
            mask_cy_px=round(float(cy) * h, 2),
        )
        if pixels_per_mm:
            values.update(
                mask_cx_mm=round(float(cx) * w / pixels_per_mm, 3),
                mask_cy_mm=round(float(cy) * h / pixels_per_mm, 3),
            )
        rows.append(tuple(values[name] for name in TRACKS_COLUMNS))
    return rows


def two_animals(*, ids=(0, 1), n=200, **kwargs) -> list[tuple]:
    """Rows for two animals circling out of phase, as track ``ids`` — a resident/intruder pair."""
    first = tracks_rows(*circle(n, radius=0.3), object_id=ids[0], **kwargs)
    second = tracks_rows(*circle(n, phase=3.0, radius=0.15), object_id=ids[1], **kwargs)
    return first + second


def keypoint_rows(
    object_id: int,
    points: dict,
    *,
    resolution=(640, 480),
    fps: float = 30.0,
    hidden: frozenset = frozenset(),
) -> list[tuple]:
    """``keypoints.csv`` rows for one animal: ``points`` is ``{name: (x_norm, y_norm)}`` arrays
    of one length; ``hidden`` is ``{(frame, name)}`` written as not visible (under the cut-off).
    """
    w, h = resolution
    rows = []
    n = len(next(iter(points.values()))[0])
    for frame in range(n):
        for name, (xs, ys) in points.items():
            x, y = float(xs[frame]), float(ys[frame])
            rows.append((
                frame, int(round(frame * 1e9 / fps)), object_id, name,
                round(x, 6), round(y, 6), round(x * w, 2), round(y * h, 2), "", "",
                0.9, 0 if (frame, name) in hidden else 1,
            ))  # fmt: skip
    return rows


def pair_rows(edges: dict, *, fps: float = 30.0) -> list[tuple]:
    """``pair_relations.csv`` rows carrying only a mask-edge distance: ``{(a, b): [px, ...]}``."""
    rows = []
    for (a, b), values in edges.items():
        for frame, px in enumerate(values):
            rows.append(
                (frame, int(round(frame * 1e9 / fps)), a, b)
                + ("",) * 6
                + (round(float(px), 2), "")
            )
    return rows


def write_recording(
    run_dir,
    recording_id,
    rows,
    *,
    resolution=(640, 480),
    pixels_per_mm=None,
    subject_labels=None,
    keypoints=None,
    pairs=None,
):
    """One recording's CSVs + ``locations.yaml`` under ``run_dir/recording_id``; ``keypoints``
    (``keypoint_rows``) writes a ``keypoints.csv`` and names its points in the manifest.
    """
    names = list(dict.fromkeys(r[3] for r in keypoints)) if keypoints else []
    tables = LocationTables(
        tracks=(TRACKS_COLUMNS, sorted(rows, key=lambda r: (r[0], r[2]))),
        roi_relations=(ROI_COLUMNS, []),
        point_relations=(POINT_COLUMNS, []),
        pair_relations=(PAIR_COLUMNS, pairs or []),
        keypoints=(KEYPOINT_COLUMNS, keypoints) if keypoints else None,
        source={
            "schema_version": 1,
            "tracking_run_id": "track_01",
            "video_resolution": list(resolution) if resolution else None,
            "pixels_per_mm": pixels_per_mm,
            "subject_labels": subject_labels,
            "keypoint_names": names,
        },
    )
    location_export.write_location_tables(run_dir / recording_id, tables)


def location_run(experiment, recordings: dict, *, run_id: str = "an_01"):
    """A complete location-export run over ``recordings``: ``{recording_id: kwargs}`` where
    kwargs may carry ``rows`` (default: a circle of 300 samples), ``resolution``,
    ``pixels_per_mm``, ``status``, ``keypoints`` / ``pairs`` (rows for those CSVs),
    ``subject_labels`` (written into the manifest, as an export does now) and ``tracking_labels`` (written only into the recording's tracking run, as for an
    export made before the manifest carried them). Creates each recording in ``experiment``.
    """
    run_dir = ar.analysis_dir(experiment.dir, run_id)
    per_recording = {}
    for index, (sid, spec) in enumerate(recordings.items()):
        if sid not in experiment.recording_names():
            experiment.create_recording(sid)
        resolution = spec.get("resolution", (640, 480))
        ppm = spec.get("pixels_per_mm")
        rows = spec.get("rows")
        if rows is None:
            rows = tracks_rows(
                *circle(300, phase=index), resolution=resolution, pixels_per_mm=ppm
            )
        status = spec.get("status", "complete")
        if status == "complete":
            write_recording(
                run_dir,
                sid,
                rows,
                resolution=resolution,
                pixels_per_mm=ppm,
                subject_labels=spec.get("subject_labels"),
                keypoints=spec.get("keypoints"),
                pairs=spec.get("pairs"),
            )
        if spec.get("tracking_labels") is not None:
            track_dir = experiment.resolve_recording(sid).dir / "tracking" / "track_01"
            track_dir.mkdir(parents=True, exist_ok=True)
            (track_dir / "track.yaml").write_text(
                yaml.dump({"subject_labels": spec["tracking_labels"]}), encoding="utf-8"
            )
        per_recording[sid] = {"status": status}
    ar.write_analysis_yaml(
        run_dir,
        {
            "id": run_id,
            "module_id": "location_export",
            "recording_ids": list(recordings),
            "status": "complete",
            "per_recording": per_recording,
            "tracking_run_ids": {sid: "track_01" for sid in recordings},
            "artifacts": [
                {"path": f"{sid}/tracks.csv", "kind": "tracks", "label": sid}
                for sid in recordings
            ],
        },
    )
    return run_dir
