"""A location-export run built by hand, for tests of what reads one (§24 *The locations source*).

Shared by ``tests/analysis/test_location_measures.py`` and
``tests/api/test_comparison_locations.py``: both need a run whose rows they chose, written
through the export's own writer, so what they read is the format a real run has.
"""

import numpy as np

from detail.analysis import location_measures as lm
from detail.analysis.locations import (
    BODY_COLUMNS,
    POINT_COLUMNS,
    ROI_COLUMNS,
    TRACKS_COLUMNS,
    LocationTables,
)
from detail.storage.location_export import write_location_tables

NS = 1_000_000_000


def track(x, y=None, *, hz=5.0, t0=0.0, t=None, length=180.0, width=75.0) -> lm.Track:
    """A track along ``x`` (mm), one sample every ``1 / hz`` seconds unless ``t`` is given."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x) if y is None else np.asarray(y, dtype=float)
    seconds = t0 + np.arange(len(x)) / hz if t is None else np.asarray(t, dtype=float)
    return lm.Track(
        t_ns=np.rint(seconds * NS).astype(np.int64),
        x_mm=x,
        y_mm=y,
        length_mm=np.full(len(x), float(length)),
        width_mm=np.full(len(x), float(width)),
        area_mm2=np.full(len(x), float(length) * float(width) * 0.8),
    )


def finished_run(
    experiment,
    plan: dict,
    *,
    run_id: str = "an_01",
    metadata: dict | None = None,
    failed: tuple[str, ...] = (),
    **write_options,
) -> dict[str, str]:
    """A finished location-export run of ``experiment`` over ``{recording name: {object_id:
    Track}}``: the recordings made by name, their files written by :func:`write_run`, and the
    run's record as the runner leaves it. ``metadata`` is ``{name: {column: value}}``;
    ``failed`` names recordings the run recorded as failed, with no files. Returns ``{name:
    recording_id}``.
    """
    from detail.storage import analysis_runs
    from detail.storage import metadata as md

    ids = {name: experiment.add_recording(name).id for name in plan}
    for name, values in (metadata or {}).items():
        md.upsert_row(experiment.dir, ids[name], values)
    run_dir = analysis_runs.analysis_dir(experiment.dir, run_id)
    write_run(
        run_dir,
        {ids[name]: tracks for name, tracks in plan.items() if name not in failed},
        **write_options,
    )
    per_recording = {
        ids[name]: {"status": "failed" if name in failed else "complete"}
        for name in plan
    }
    analysis_runs.write_analysis_yaml(
        run_dir,
        {
            "id": run_id,
            "module_id": "location_export",
            "status": "partial" if failed else "complete",
            "per_recording": per_recording,
            "artifacts": [
                {"path": f"{sid}/tracks.csv", "kind": "tracks", "label": sid}
                for name, sid in ids.items()
                if name not in failed
            ],
            "params": {
                "source_recording_ids": list(ids.values()),
                "sample_fps": 5.0,
                "remove_thin_parts": True,
                "thin_part_scale": 0.12,
            },
        },
    )
    return ids


def write_keypoints(
    recording_dir, tr: lm.Track, points: dict, *, pixels_per_mm=2.0, frame=None
) -> None:
    """A ``keypoints.csv`` beside a hand-built recording's tables, as the export writes it,
    and the manifest's ``keypoint_names``: one row per frame of ``tr`` per point, ``points``
    ``{name: (dx_mm, dy_mm, hidden)}`` — the point at the track's position plus that offset,
    and ``hidden`` a boolean array (or None) of frames it was under the cut-off."""
    import yaml

    from detail.analysis.locations import KEYPOINT_COLUMNS
    from detail.storage import location_export

    scale = pixels_per_mm or 1.0
    width, height = frame or FRAME
    rows = []
    for k in range(len(tr.t_ns)):
        for name, (dx, dy, hidden) in points.items():
            x_px = (float(tr.x_mm[k]) + dx) * scale
            y_px = (float(tr.y_mm[k]) + dy) * scale
            visible = 0 if hidden is not None and hidden[k] else 1
            rows.append(
                (
                    k, int(tr.t_ns[k]), 0, name,
                    round(x_px / width, 6), round(y_px / height, 6),
                    round(x_px, 2), round(y_px, 2), "", "", 0.9, visible,
                )
            )  # fmt: skip
    location_export._write_csv(
        recording_dir / location_export.KEYPOINTS_FILE, KEYPOINT_COLUMNS, rows
    )
    manifest = location_export.read_location_manifest(recording_dir)
    manifest["keypoint_run_id"] = "kt_01"
    manifest["keypoint_names"] = list(points)
    with open(
        recording_dir / location_export.MANIFEST_FILE, "w", encoding="utf-8"
    ) as f:
        yaml.dump(manifest, f, sort_keys=False)


#: The frame a hand-built run's recordings were filmed at, ``(width, height)`` in pixels.
FRAME = (640, 480)
#: How much larger the whole mask is than the body in a hand-built run: the tail.
MASK_OVER_BODY = 1.25


def write_run(
    run_dir, recordings: dict, *, pixels_per_mm=2.0, body=True, frame=FRAME
) -> None:
    """A location-export run's files for ``{recording_id: {object_id: Track}}``, through the
    export's own writer. ``body=False`` is a run from before the body was measured.

    A track's positions and sizes are in millimetres; every other column is filled from them
    as the export fills it — pixels at ``pixels_per_mm``, normalized by ``frame`` — with the
    mask's and the box's centres at the body's, the long axis along x, and the whole mask
    ``MASK_OVER_BODY`` times the body's area. ``pixels_per_mm=None`` is an uncalibrated run:
    its millimetre columns are blank, and the track's numbers are read as pixels.
    """
    columns = TRACKS_COLUMNS if body else TRACKS_COLUMNS[: -len(BODY_COLUMNS)]
    at = {name: i for i, name in enumerate(columns)}
    scale = pixels_per_mm or 1.0
    width, height = frame

    def mm(value: float) -> str | float:
        return round(value, 3) if pixels_per_mm else ""

    for recording_id, tracks in recordings.items():
        rows = []
        for object_id, tr in tracks.items():
            for k in range(len(tr.t_ns)):
                row = [""] * len(columns)
                row[at["frame_idx"]], row[at["t_ns"]] = k, int(tr.t_ns[k])
                row[at["object_id"]] = object_id
                x_px, y_px = float(tr.x_mm[k]) * scale, float(tr.y_mm[k]) * scale
                area_px = float(tr.area_mm2[k]) * scale * scale
                centres = {"x_norm": x_px / width, "y_norm": y_px / height}
                for src in ("mask", "bbox") + (("body",) if body else ()):
                    row[at[f"{src}_cx_norm"]] = round(centres["x_norm"], 6)
                    row[at[f"{src}_cy_norm"]] = round(centres["y_norm"], 6)
                    row[at[f"{src}_cx_px"]] = round(x_px, 2)
                    row[at[f"{src}_cy_px"]] = round(y_px, 2)
                    row[at[f"{src}_cx_mm"]] = mm(float(tr.x_mm[k]))
                    row[at[f"{src}_cy_mm"]] = mm(float(tr.y_mm[k]))
                row[at["mask_area_px"]] = round(area_px * MASK_OVER_BODY, 2)
                row[at["mask_area_norm"]] = round(
                    area_px * MASK_OVER_BODY / (width * height), 8
                )
                if body:
                    for name, value in (
                        ("body_length_px", float(tr.length_mm[k]) * scale),
                        ("body_width_px", float(tr.width_mm[k]) * scale),
                        ("body_area_px", area_px),
                        ("body_area_norm", area_px / (width * height)),
                        ("body_angle_deg", 0.0),
                    ):
                        row[at[name]] = "" if np.isnan(value) else round(value, 8)
                    for name, values in (
                        ("body_length_mm", tr.length_mm), ("body_width_mm", tr.width_mm),
                        ("body_area_mm2", tr.area_mm2),
                    ):  # fmt: skip
                        value = float(values[k])
                        row[at[name]] = "" if np.isnan(value) else mm(value)
                rows.append(tuple(row))
        write_location_tables(
            run_dir / recording_id,
            LocationTables(
                tracks=(list(columns), rows),
                roi_relations=(ROI_COLUMNS, []),
                point_relations=(POINT_COLUMNS, []),
                source={
                    "schema_version": 1,
                    "video_resolution": [width, height],
                    "pixels_per_mm": pixels_per_mm,
                    "sample_fps": 5.0,
                    "remove_thin_parts": True,
                    "thin_part_scale": 0.12,
                },
            ),
        )
