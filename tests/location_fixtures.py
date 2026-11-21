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


def write_run(run_dir, recordings: dict, *, pixels_per_mm=2.0, body=True) -> None:
    """A location-export run's files for ``{recording_id: {object_id: Track}}``, through the
    export's own writer. ``body=False`` is a run from before the body was measured."""
    columns = TRACKS_COLUMNS if body else TRACKS_COLUMNS[: -len(BODY_COLUMNS)]
    at = {name: i for i, name in enumerate(columns)}
    for recording_id, tracks in recordings.items():
        rows = []
        for object_id, tr in tracks.items():
            for k in range(len(tr.t_ns)):
                row = [""] * len(columns)
                row[at["frame_idx"]], row[at["t_ns"]] = k, int(tr.t_ns[k])
                row[at["object_id"]] = object_id
                if body:
                    for name, values in (
                        ("body_cx_mm", tr.x_mm), ("body_cy_mm", tr.y_mm),
                        ("body_length_mm", tr.length_mm), ("body_width_mm", tr.width_mm),
                        ("body_area_mm2", tr.area_mm2),
                    ):  # fmt: skip
                        value = float(values[k])
                        row[at[name]] = "" if np.isnan(value) else round(value, 3)
                rows.append(tuple(row))
        write_location_tables(
            run_dir / recording_id,
            LocationTables(
                tracks=(list(columns), rows),
                roi_relations=(ROI_COLUMNS, []),
                point_relations=(POINT_COLUMNS, []),
                source={
                    "schema_version": 1,
                    "pixels_per_mm": pixels_per_mm,
                    "sample_fps": 5.0,
                    "remove_thin_parts": True,
                    "thin_part_scale": 0.12,
                },
            ),
        )
