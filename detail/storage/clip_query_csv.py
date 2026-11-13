"""The clip-query result CSV — one self-describing schema in one place (§16).

One row per clip: a query example (``kind='query'``, rank 0) or a ranked match
(``kind='match'``), tagged with the named query it belongs to (``query_id`` + ``query_name``
— a run holds several distinct queries). Each row carries its subject (``recording_id`` +
``track_id``) and ``embed_run_id``, so a reader needs no per-subject overrides. Spans are
canonical in ``t_ns`` (Tenet 8 — the universal time reference, and what lets a match join
streams by time), with global ``frame_idx`` alongside for readability.

The **same** schema is the query-spec file, the results file, and the refine-from-results
import: point the query at any prior clip CSV and take its rows as new examples. Written
atomically (a background worker rewrites it while the GUI may read — the atomic-write
tenet); mirrors ``storage/location_export.py``.
"""

import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from detail.core import atomic

COLUMNS = [
    "query_id",
    "query_name",
    "kind",
    "rank",
    "recording_id",
    "track_id",
    "embed_run_id",
    "start_t_ns",
    "end_t_ns",
    "start_frame",
    "end_frame",
    "length",
    "score",
    "example_idx",
]

# Columns a reader must find. ``length`` is derivable from the frame span; the rest are
# optional so an older CSV (no query columns / no example_idx) still reads.
_REQUIRED = set(COLUMNS) - {"length", "query_id", "query_name", "example_idx"}


@dataclass
class ClipRow:
    """One clip-query CSV row: a query example (``kind='query'``) or a match, tagged with
    the named query it belongs to."""

    kind: str
    rank: int
    recording_id: str
    track_id: int
    embed_run_id: str
    start_t_ns: int
    end_t_ns: int
    start_frame: int
    end_frame: int
    score: float
    # Defaults let a pre-multi-query CSV read as one unnamed query.
    query_id: str = "q01"
    query_name: str = "query"
    example_idx: int = (
        0  # which example clip a match best matches (0 = the first/only one)
    )

    @property
    def length(self) -> int:
        """Inclusive frame length of the clip."""
        return self.end_frame - self.start_frame + 1


def write_clip_csv(path: Path | str, rows: Sequence[ClipRow]) -> None:
    """Write ``rows`` to ``path`` with the canonical header, atomically."""
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow(
                [
                    r.query_id,
                    r.query_name,
                    r.kind,
                    r.rank,
                    r.recording_id,
                    r.track_id,
                    r.embed_run_id,
                    r.start_t_ns,
                    r.end_t_ns,
                    r.start_frame,
                    r.end_frame,
                    r.length,
                    r.score,
                    r.example_idx,
                ]
            )
    atomic.replace(tmp, path)


def read_clip_csv(path: Path | str) -> list[ClipRow]:
    """Read a clip-query CSV into ``ClipRow``s.

    Raises
    ValueError: the file is missing an identifying/span column (not a clip CSV)
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = _REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"clip-query CSV {path} missing columns {sorted(missing)} "
                "(not a clip-query CSV, or regenerate it with the current tool)"
            )
        return [
            ClipRow(
                query_id=r.get("query_id") or "q01",
                query_name=r.get("query_name") or "query",
                kind=r["kind"],
                rank=int(r["rank"]),
                recording_id=r["recording_id"],
                track_id=int(r["track_id"]),
                embed_run_id=r["embed_run_id"],
                start_t_ns=int(r["start_t_ns"]),
                end_t_ns=int(r["end_t_ns"]),
                start_frame=int(r["start_frame"]),
                end_frame=int(r["end_frame"]),
                score=float(r["score"]),
                example_idx=int(r.get("example_idx") or 0),
            )
            for r in reader
        ]
