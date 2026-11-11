"""Location-export CSV + manifest writer/reader (analysis layer).

Three tidy CSVs (``tracks`` / ``roi_relations`` / ``point_relations``) keyed by
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
MANIFEST_FILE = "locations.yaml"


def _write_csv(path: Path, columns: list[str], rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def write_location_tables(out_dir: Path | str, tables) -> Path:
    """Write the three CSVs + a schema-versioned ``locations.yaml`` into ``out_dir``.

    ``tables`` is an ``analysis.locations.LocationTables``. Returns ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / TRACKS_FILE, *tables.tracks)
    _write_csv(out_dir / ROI_FILE, *tables.roi_relations)
    _write_csv(out_dir / POINT_FILE, *tables.point_relations)

    manifest = {"schema_version": tables.source.get("schema_version", 1)}
    manifest.update({k: v for k, v in tables.source.items() if k != "schema_version"})
    manifest["files"] = {
        "tracks": TRACKS_FILE,
        "roi_relations": ROI_FILE,
        "point_relations": POINT_FILE,
    }
    manifest["columns"] = {
        "tracks": tables.tracks[0],
        "roi_relations": tables.roi_relations[0],
        "point_relations": tables.point_relations[0],
    }
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
