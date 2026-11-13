"""Location-export CSV + manifest writer/reader (analysis layer).

Tidy CSVs (``tracks`` / ``roi_relations`` / ``point_relations`` / ``pair_relations``, and
``keypoints`` when the export read a keypoint track run) keyed by
``t_ns`` so they join with the streams CSV (§3), plus a ``locations.yaml`` manifest
(``schema_version`` first — Tenet 7; full lineage so the export reproduces from
itself). Mirrors the canonical-stream CSV idiom in ``storage/streams.py``.

(Distinct from ``storage/locations.py``, which is the storage-locations config §2.)
"""

import csv
from pathlib import Path

import yaml

TRACKS_FILE = "tracks.csv"
ROI_FILE = "roi_relations.csv"
POINT_FILE = "point_relations.csv"
PAIR_FILE = "pair_relations.csv"
KEYPOINTS_FILE = "keypoints.csv"
MANIFEST_FILE = "locations.yaml"


def _write_csv(path: Path, columns: list[str], rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def written_tables(tables) -> dict[str, tuple[str, tuple]]:
    """``{kind: (file name, (columns, rows))}`` for what an export writes: the four tables
    always (``pair_relations`` header-only on a single-subject recording, so every export has
    the same files), and ``keypoints`` only when it was given a keypoint track run."""
    out = {
        "tracks": (TRACKS_FILE, tables.tracks),
        "roi_relations": (ROI_FILE, tables.roi_relations),
        "point_relations": (POINT_FILE, tables.point_relations),
        "pair_relations": (PAIR_FILE, tables.pair_relations),
    }
    if tables.keypoints is not None:
        out["keypoints"] = (KEYPOINTS_FILE, tables.keypoints)
    return out


def write_location_tables(out_dir: Path | str, tables) -> Path:
    """Write the CSVs + a schema-versioned ``locations.yaml`` into ``out_dir``.

    ``tables`` is an ``analysis.locations.LocationTables``. Returns ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = written_tables(tables)
    for name, table in written.values():
        _write_csv(out_dir / name, *table)

    manifest = {"schema_version": tables.source.get("schema_version", 1)}
    manifest.update({k: v for k, v in tables.source.items() if k != "schema_version"})
    manifest["files"] = {kind: name for kind, (name, _table) in written.items()}
    manifest["columns"] = {kind: table[0] for kind, (_name, table) in written.items()}
    with open(out_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, sort_keys=False)
    return out_dir


def read_location_csv(path: Path | str) -> tuple[list[str], list[tuple]]:
    """Read one location CSV into ``(columns, rows-of-strings)`` — for re-load / tests."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        columns = next(reader)
        rows = [tuple(row) for row in reader]
    return columns, rows
