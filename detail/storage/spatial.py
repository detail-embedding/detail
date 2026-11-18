"""Arena, ROI, and POI keyframe storage (Section 10).

Spatial definitions live per-recording as top-level ``arenas:`` / ``rois:`` /
``points:`` entries in ``recording.yaml`` plus one ``keyframes.csv`` per definition
under ``<kind-dir>/<id>/``. ``frame_idx`` in the CSV is recording-global —
cumulative across video parts, with 0 at the first frame of ``part_001``.

Arenas are processing-time detection filters (always in effect). ROIs and POIs
are analysis-time definitions and may carry an optional ``active`` interval
(``{start, end}``, recording-global frames) outside which they are not present
(§10). A POI is a single-vertex polygon (``n_vertices = 1``); ROIs and arenas are
polygons (``>= 3`` vertices).
"""

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from detail.core import utils as core_utils
from detail.core.dataclasses import PolygonKeyframes
from detail.core.errors import UserInputError
from detail.importers import _csv_common
from detail.storage import dirscan
from detail.storage import id_allocator
from detail.storage import locations
from detail.storage import recording as recording_store

KEYFRAMES_FILENAME = "keyframes.csv"
ARENAS_DIRNAME = "arenas"

Keyframes = list[tuple[int, list[tuple[float, float]]]]
ActiveInterval = tuple[int, int] | None


@dataclass(frozen=True)
class _Kind:
    dirname: str  # subdir name + recording.yaml manifest key
    id_prefix: str  # id stem: "arena" -> "arena_001"
    min_vertices: int
    max_vertices: int | None  # None = unbounded (polygon)
    has_active: bool  # ROIs/POIs may be present for only part of the video


_KINDS: dict[str, _Kind] = {
    "arena": _Kind("arenas", "arena", 3, None, False),
    "roi": _Kind("rois", "roi", 3, None, True),
    "point": _Kind("points", "point", 1, 1, True),
}


def _kind(kind: str) -> _Kind:
    try:
        return _KINDS[kind]
    except KeyError:
        raise ValueError(
            f"unknown spatial kind {kind!r}; expected one of {sorted(_KINDS)}"
        ) from None


def _def_dir(recording_dir: Path | str, kind: str, def_id: str) -> Path:
    return Path(recording_dir) / _kind(kind).dirname / def_id


def _csv_header(n_vertices: int) -> list[str]:
    cols = ["frame_idx"]
    for i in range(n_vertices):
        cols.extend([f"v{i}x", f"v{i}y"])
    return cols


def _validate_keyframes(
    keyframes: Keyframes, min_vertices: int, max_vertices: int | None
) -> int:
    if not keyframes:
        raise ValueError("keyframes must be non-empty")
    n_vertices = len(keyframes[0][1])
    if max_vertices is not None and min_vertices == max_vertices:
        if n_vertices != min_vertices:
            raise ValueError(
                f"a point must have exactly {min_vertices} vertex, got {n_vertices}"
            )
    elif n_vertices < min_vertices:
        raise ValueError(
            f"polygon must have at least {min_vertices} vertices, got {n_vertices}"
        )
    last_frame = -1
    for frame_idx, vertices in keyframes:
        if frame_idx < 0:
            raise ValueError(f"frame_idx must be >= 0, got {frame_idx}")
        if frame_idx <= last_frame:
            raise ValueError(
                f"keyframes must have strictly increasing frame_idx; "
                f"{frame_idx} follows {last_frame}"
            )
        last_frame = frame_idx
        if len(vertices) != n_vertices:
            raise ValueError(
                f"keyframe at frame {frame_idx} has {len(vertices)} vertices; "
                f"expected {n_vertices} (all keyframes must agree)"
            )
        for x, y in vertices:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"vertex ({x}, {y}) at frame {frame_idx} is outside [0, 1]: coordinates "
                    "are fractions of the frame (0 to 1), so divide pixel coordinates by the "
                    "frame's width and height"
                )
    return n_vertices


def _validate_active(active: ActiveInterval) -> None:
    if active is None:
        return
    start, end = active
    if start < 0 or end < 0:
        raise ValueError(f"active interval frames must be >= 0, got {active}")
    if start > end:
        raise ValueError(f"active interval start {start} must be <= end {end}")


# --------------------------------------------------------------------------- #
# Generic, kind-parameterized core (kind in {"arena", "roi", "point"})
# --------------------------------------------------------------------------- #
def write_keyframes(
    recording_dir: Path | str, kind: str, def_id: str, keyframes: Keyframes
) -> int:
    """Write ``<kind-dir>/<def_id>/keyframes.csv``; returns the definition's
    n_vertices (1 for a point, >= 3 for an arena/ROI polygon).

    Raises:
        ValueError: If keyframes are empty, the vertex count violates the kind's
            constraint or is inconsistent across keyframes, coordinates fall
            outside ``[0, 1]``, or ``frame_idx`` is not strictly increasing from a
            non-negative start.
    """
    k = _kind(kind)
    n_vertices = _validate_keyframes(keyframes, k.min_vertices, k.max_vertices)
    target = _def_dir(recording_dir, kind, def_id)
    target.mkdir(parents=True, exist_ok=True)
    with open(target / KEYFRAMES_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_csv_header(n_vertices))
        for frame_idx, vertices in keyframes:
            row: list[object] = [frame_idx]
            for x, y in vertices:
                row.extend([float(x), float(y)])
            writer.writerow(row)
    return n_vertices


def keyframes_path(recording_dir: Path | str, kind: str, def_id: str) -> Path:
    """Where a definition's keyframes live — the file a reader stamps a cache on."""
    return _def_dir(recording_dir, kind, def_id) / KEYFRAMES_FILENAME


def arena_keyframes_path(recording_dir: Path | str, arena_id: str) -> Path:
    """``arenas/<arena_id>/keyframes.csv``."""
    return keyframes_path(recording_dir, "arena", arena_id)


def read_keyframes(
    recording_dir: Path | str, kind: str, def_id: str, n_vertices: int
) -> PolygonKeyframes:
    """Load ``<kind-dir>/<def_id>/keyframes.csv`` into a ``PolygonKeyframes``.

    Raises:
        FileNotFoundError: If the keyframes file is absent.
        ValueError: If the header column count does not match ``n_vertices``.
    """
    path = keyframes_path(recording_dir, kind, def_id)
    if not path.is_file():
        raise FileNotFoundError(f"no keyframes file at {path}")
    expected_cols = 1 + 2 * n_vertices
    keyframes: Keyframes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) != expected_cols:
            raise ValueError(
                f"{path}: expected {expected_cols} columns for n_vertices={n_vertices}, "
                f"got {len(header) if header else 0}"
            )
        for row in reader:
            frame_idx = int(row[0])
            vertices = [
                (float(row[1 + 2 * i]), float(row[2 + 2 * i]))
                for i in range(n_vertices)
            ]
            keyframes.append((frame_idx, vertices))
    return PolygonKeyframes(n_vertices=n_vertices, keyframes=keyframes)


def list_defs(recording_dir: Path | str, kind: str) -> list[dict]:
    """Return the ``recording.yaml`` manifest list for ``kind`` (empty if none)."""
    data = recording_store.read_recording(recording_dir)
    return list(data.get(_kind(kind).dirname, []))


def next_id(recording_dir: Path | str, kind: str) -> str:
    """Claim the next ``<prefix>_NNN`` id (3-digit zero-padded, 1-based) for a ROI or POI.

    Never handed out again once claimed, even after that region is deleted: a finished
    location run's ``roi_relations.csv`` names regions by this id, so a new region taking a
    deleted one's id would join to the old run's rows as if it were the same region. The mark
    is the recording's own (``storage/id_allocator.py``). Arenas keep the lowest free number
    (:func:`next_arena_id` says why).
    """
    if kind == "arena":
        return next_arena_id(recording_dir)
    k = _kind(kind)
    taken = [entry.get("id") for entry in list_defs(recording_dir, kind)]
    kind_dir = Path(recording_dir) / k.dirname
    taken += [p.name for p in dirscan.child_dirs(kind_dir)]
    return id_allocator.claim(recording_dir, k.id_prefix, digits=3, taken=taken)


def stored_name(name: str) -> str:
    """A definition's name as it is stored: without spaces at either end.

    The one rule both directions share. The GUI always stripped, the API stored what it
    was given, and the CSV import stripped, so a region named ``" padded "`` through the API
    exported with its spaces and came back as a different name (#31). The CSV reader refuses
    such a name instead of stripping it, because there a padded name is a typing mistake
    that can collide with another row's.
    """
    return str(name).strip()


def append_to_manifest(
    recording_dir: Path | str,
    kind: str,
    def_id: str,
    name: str,
    n_vertices: int,
    active: ActiveInterval = None,
) -> None:
    """Append the definition's entry to its ``recording.yaml`` manifest list.

    Arena entries are ``{id, name, n_vertices}``. ROI/POI entries also carry
    ``active`` (``{start, end}`` or ``null`` = whole video).

    Raises:
        ValueError: If ``name`` is empty, an entry with this ``id`` or ``name``
            already exists, ``active`` is given for a kind that does not support it
            (arenas), or the active interval is malformed.
    """
    k = _kind(kind)
    name = stored_name(name)
    if not name:
        raise ValueError(f"{k.id_prefix} name must be non-empty")
    if active is not None and not k.has_active:
        raise ValueError(f"{kind!r} does not support an active interval")
    _validate_active(active)
    data = recording_store.read_recording(recording_dir)
    existing = list(data.get(k.dirname, []))
    for entry in existing:
        if entry.get("id") == def_id:
            raise ValueError(f"{k.id_prefix} id {def_id!r} already exists in recording")
        if entry.get("name") == name:
            raise ValueError(f"{k.id_prefix} name {name!r} already exists in recording")
    entry = {"id": def_id, "name": name, "n_vertices": int(n_vertices)}
    if k.has_active:
        entry["active"] = (
            {"start": int(active[0]), "end": int(active[1])} if active else None
        )
    existing.append(entry)
    data[k.dirname] = existing
    recording_store.write_recording(recording_dir, data)


def delete_def(recording_dir: Path | str, kind: str, def_id: str) -> None:
    """Remove a definition's keyframes dir + its ``recording.yaml`` entry
    (idempotent). Already-computed runs are unaffected — the polygon was applied
    at run time."""
    k = _kind(kind)
    ddir = _def_dir(recording_dir, kind, def_id)
    if ddir.is_dir():
        shutil.rmtree(ddir)
    data = recording_store.read_recording(recording_dir)
    data[k.dirname] = [e for e in data.get(k.dirname, []) if e.get("id") != def_id]
    recording_store.write_recording(recording_dir, data)


# --------------------------------------------------------------------------- #
# Bulk replace — the multi-region manager's "Save all" (§10)
# --------------------------------------------------------------------------- #
def replace_defs(recording_dir: Path | str, kind: str, defs: list[dict]) -> list[str]:
    """Replace ALL definitions of ``kind`` in one operation — the bulk counterpart to the
    per-def define path, used by the ROI/POI manager's "Save all". Each entry is
    ``{"name": str, "keyframes": Keyframes, "active": ActiveInterval, "id": str}``
    (``active`` and ``id`` optional). Validates the whole set first (Tenet 10), then removes
    existing definitions of this kind and writes the new set; returns the ids, in order.

    **An entry carrying the ``id`` of a definition that exists keeps that id**, so a region
    the user did not delete is the same ``roi_NNN`` before and after a Save, and a location
    run joined on ``roi_id`` across the edit pairs like with like. An entry without one (a new
    region), or with an id no longer defined, gets a fresh id that was never used before
    (:func:`next_id`).

    Safe for ROIs/POIs (no run-lock); not for arenas, which guard against id churn.

    Raises:
        ValueError: An empty or duplicated name or id, or invalid keyframes / active.
    """
    k = _kind(kind)
    prepared: list[tuple[str | None, str, Keyframes, ActiveInterval, int]] = []
    seen: set[str] = set()
    seen_ids: set[str] = set()
    existing_ids = {entry["id"] for entry in list_defs(recording_dir, kind)}
    for d in defs:
        kept = d.get("id")
        if kept is not None:
            if kept in seen_ids:
                raise ValueError(f"{k.id_prefix} id {kept!r} is duplicated")
            seen_ids.add(kept)
            if kept not in existing_ids:
                kept = None
        name = stored_name(d.get("name") or "")
        if not name:
            raise ValueError(f"{k.id_prefix} name must be non-empty")
        if name in seen:
            raise ValueError(f"{k.id_prefix} name {name!r} is duplicated")
        seen.add(name)
        active = d.get("active")
        if active is not None and not k.has_active:
            raise ValueError(f"{kind!r} does not support an active interval")
        _validate_active(active)
        n_vertices = _validate_keyframes(d["keyframes"], k.min_vertices, k.max_vertices)
        prepared.append((kept, name, d["keyframes"], active, n_vertices))
    # New ids are claimed while the old definitions still exist, so even a recording with no
    # mark yet (made before it existed) cannot hand a new region a kept region's id.
    new_ids = [kept or next_id(recording_dir, kind) for kept, *_ in prepared]
    for entry in list_defs(recording_dir, kind):
        delete_def(recording_dir, kind, entry["id"])
    for def_id, (_kept, name, keyframes, active, n_vertices) in zip(new_ids, prepared):
        write_keyframes(recording_dir, kind, def_id, keyframes)
        append_to_manifest(recording_dir, kind, def_id, name, n_vertices, active)
    return new_ids


# --------------------------------------------------------------------------- #
# CSV export / import — spatial-setup carry-over across recordings (§10)
# --------------------------------------------------------------------------- #
# The four trailing columns carry the distance calibration, which has no vertices and so
# no place in the leading ones. They are appended rather than inserted so a CSV exported
# before they existed still parses (readers key by name, and the loader treats them as
# optional).
_SPATIAL_CSV_COLUMNS = [
    "kind",
    "name",
    "active_start",
    "active_end",
    "frame_idx",
    "vertex_index",
    "x",
    "y",
    "distance_mm",
    "pixels_per_mm",
    "ref_width",
    "ref_height",
]

# ``kind`` value for the calibration rows — not a polygon kind, so it is deliberately not
# in ``_KINDS`` and every polygon path skips it.
CALIBRATION_KIND = "calibration"

# Everything a recording's spatial setup consists of, in export order.
ALL_SPATIAL_KINDS = ("arena", "roi", "point")


def spatial_exports_root() -> Path:
    """Machine-global folder for exported spatial-setup CSVs (mirrors the tools workspace)."""
    root = locations.experiments_root() / "roi_exports"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _calibration_rows(recording_dir: Path | str) -> list[tuple]:
    """The calibration as CSV rows at the file's one-row-per-vertex grain: two rows (the
    normalized pick points) for a two-point calibration, one point-less row for a scalar
    one. The scalars repeat on each row; the loader reads them off the first."""
    cal = read_calibration(recording_dir)
    if cal is None:
        return []
    ref = cal.get("ref_resolution") or ["", ""]
    tail = (cal.get("distance_mm") or "", cal["pixels_per_mm"], ref[0], ref[1])
    points = cal.get("points")
    if not points:
        return [(CALIBRATION_KIND, CALIBRATION_KIND, "", "", 0, 0, "", "", *tail)]
    return [
        (CALIBRATION_KIND, CALIBRATION_KIND, "", "", 0, vi, x, y, *tail)
        for vi, (x, y) in enumerate(points)
    ]


def export_spatial(
    recording_dir: Path | str,
    out_path: Path | str,
    kinds: tuple[str, ...] = ALL_SPATIAL_KINDS,
    include_calibration: bool = True,
) -> None:
    """Write the recording's whole spatial setup to a flat, round-trippable CSV — one row
    per vertex per keyframe (columns ``_SPATIAL_CSV_COLUMNS``), plus the distance
    calibration. One file is the setup, so it carries the field of view (``arena``) and
    the calibration as well as the ROIs/POIs; narrow it with ``kinds`` /
    ``include_calibration``."""
    rows: list[tuple] = []
    for kind in kinds:
        for entry in list_defs(recording_dir, kind):
            path = keyframes_path(recording_dir, kind, entry["id"])
            if not path.is_file():
                raise UserInputError(
                    f"{kind} {entry['name']!r} is listed in recording.yaml but its keyframes "
                    f"file is missing ({path}). Draw it again, or delete it, then export"
                )
            pk = read_keyframes(recording_dir, kind, entry["id"], entry["n_vertices"])
            active = entry.get("active")
            a_start = active["start"] if active else ""
            a_end = active["end"] if active else ""
            for frame_idx, vertices in pk.keyframes:
                for vi, (x, y) in enumerate(vertices):
                    rows.append(
                        (kind, entry["name"], a_start, a_end, frame_idx, vi, x, y)
                        + ("", "", "", "")
                    )
    if include_calibration:
        rows += _calibration_rows(recording_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SPATIAL_CSV_COLUMNS)
        writer.writerows(rows)


# Only the polygon columns are required: the calibration ones were added later, so a CSV
# exported before them must still load (it simply carries no calibration).
_SPATIAL_CSV_REQUIRED = _SPATIAL_CSV_COLUMNS[:8]


def _parse_active(row: _csv_common.CsvRow) -> ActiveInterval:
    """The row's active window, ``None`` for "the whole video" (both bounds blank).

    One bound alone is refused: read as "no window", ``active_start=5`` with a blank end
    widened the region to the whole video, the worse of the two mistakes
    (:func:`prepare_for_arrival` says why) (#31).

    Raises:
        UserInputError: One bound blank, or a bound that is not an integer.
    """
    start = (row.get("active_start") or "").strip()
    end = (row.get("active_end") or "").strip()
    if start == "" and end == "":
        return None
    if start == "" or end == "":
        raise UserInputError(
            f"{row.source} row {row.line}: active_start and active_end are both set or "
            "both blank (blank means the whole video); this row sets only one"
        )
    return (
        _csv_common.int_cell(row, "active_start"),
        _csv_common.int_cell(row, "active_end"),
    )


def _read_spatial_csv(path: Path | str) -> list[_csv_common.CsvRow]:
    """The rows of an exported spatial CSV, its header checked (through the one opener for a
    CSV a user hands in, so a file saved in another encoding says so)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    fieldnames, rows = _csv_common.read_user_csv(p)
    _csv_common.reject_duplicate_columns(fieldnames)
    missing = set(_SPATIAL_CSV_REQUIRED) - set(fieldnames)
    if missing:
        raise UserInputError(f"spatial CSV missing columns: {sorted(missing)}")
    return rows


def load_calibration_csv(path: Path | str) -> dict | None:
    """The distance-calibration block from an exported spatial CSV, or ``None`` if it
    carries none (including every CSV exported before calibration rows existed).

    A separate reader rather than a second return value from ``load_spatial_csv`` — the
    calibration is not a polygon def, and keeping the signatures apart leaves every
    existing caller of that function untouched.

    Raises:
        FileNotFoundError: if the file is absent.
        ValueError: on a malformed calibration row.
    """
    rows = [
        r
        for r in _read_spatial_csv(path)
        if (r["kind"] or "").strip() == CALIBRATION_KIND
    ]
    if not rows:
        return None
    head = rows[0]
    try:
        pixels_per_mm = float(head["pixels_per_mm"])
        ref = [int(float(head["ref_width"])), int(float(head["ref_height"]))]
    except (KeyError, TypeError, ValueError) as exc:
        raise UserInputError(f"malformed calibration row in {path}: {exc}") from exc
    if ref[0] <= 0 or ref[1] <= 0:
        raise UserInputError(
            f"calibration in {path}: ref_width and ref_height must be positive (the video "
            f"size the calibration was measured at), got {ref[0]}x{ref[1]}"
        )
    points = [
        (float(r["x"]), float(r["y"]))
        for r in rows
        if (r.get("x") or "").strip() != "" and (r.get("y") or "").strip() != ""
    ]
    distance_raw = (head.get("distance_mm") or "").strip()
    distance_mm = float(distance_raw) if distance_raw else None
    if len(points) not in (0, 2):
        raise ValueError(
            f"calibration in {path} has {len(points)} point(s); expected 0 or 2"
        )
    return {
        "schema_version": 1,
        "points": [list(p) for p in points] if points else None,
        "distance_mm": distance_mm if points else None,
        "pixels_per_mm": pixels_per_mm,
        "ref_resolution": ref,
    }


def load_spatial_csv(path: Path | str) -> list[dict]:
    """Parse an exported spatial CSV back into defs ``{kind, name, keyframes, active}``,
    grouping rows by ``(kind, name)`` and rebuilding keyframes (ordered by ``frame_idx``,
    vertices by ``vertex_index``).

    Calibration rows are skipped — they carry no polygon; read them with
    ``load_calibration_csv``.

    A hand-edited file is read as strictly as it was written (#31): a name with a space at
    either end (``nest `` beside ``nest`` merged the two into one polygon), a definition
    whose rows disagree on the active window or set only one bound, a ``(kind, name,
    frame_idx, vertex_index)`` given twice (the later row won), and vertex indexes that
    skip one (``0, 1, 5`` made a triangle) are each refused, naming the row.

    Raises:
        FileNotFoundError: if the file is absent.
        ValueError: on structural problems (missing or repeated columns, unknown kind,
            empty name), or any of the refusals above.
    """
    groups: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for row in _read_spatial_csv(path):
        kind = (row["kind"] or "").strip()
        if kind == CALIBRATION_KIND:
            continue
        if kind not in _KINDS:
            raise UserInputError(
                f"{row.source} row {row.line}: unknown spatial kind {kind!r}; the kind "
                f"column holds one of {sorted([*_KINDS, CALIBRATION_KIND])}"
            )
        raw_name = row["name"] or ""
        name = raw_name.strip()
        if not name:
            raise UserInputError(f"{row.source} row {row.line}: the name is empty")
        if name != raw_name:
            raise UserInputError(
                f"{row.source} row {row.line}: name {raw_name!r} has a space at its start "
                "or end; remove it (a region's name never does)"
            )
        key = (kind, name)
        active = _parse_active(row)
        if key not in groups:
            groups[key] = {"active": active, "frames": {}, "line": row.line}
            order.append(key)
        elif groups[key]["active"] != active:
            raise UserInputError(
                f"{row.source} row {row.line}: {kind} {name!r} has active window "
                f"{active} here and {groups[key]['active']} on row {groups[key]['line']}; "
                "a definition has one window, the same on every row"
            )
        frame_idx = _csv_common.int_cell(row, "frame_idx")
        vi = _csv_common.int_cell(row, "vertex_index")
        frame = groups[key]["frames"].setdefault(frame_idx, {})
        if vi in frame:
            raise UserInputError(
                f"{row.source} row {row.line}: {kind} {name!r} frame {frame_idx} vertex "
                f"{vi} is given twice (first on row {frame[vi][1]}); keep one"
            )
        frame[vi] = (
            (_csv_common.float_cell(row, "x"), _csv_common.float_cell(row, "y")),
            row.line,
        )
    defs: list[dict] = []
    for kind, name in order:
        g = groups[(kind, name)]
        keyframes: Keyframes = []
        for frame_idx in sorted(g["frames"]):
            by_index = g["frames"][frame_idx]
            indexes = sorted(by_index)
            if indexes != list(range(len(indexes))):
                raise UserInputError(
                    f"{path}: {kind} {name!r} frame {frame_idx} has vertex indexes "
                    f"{indexes}; they count up from 0 with none skipped"
                )
            keyframes.append((frame_idx, [by_index[i][0] for i in indexes]))
        defs.append(
            {"kind": kind, "name": name, "keyframes": keyframes, "active": g["active"]}
        )
    return defs


def _unique_name(recording_dir: Path | str, kind: str, name: str) -> str:
    taken = {e.get("name") for e in list_defs(recording_dir, kind)}
    if name not in taken:
        return name
    i = 2
    while f"{name} ({i})" in taken:
        i += 1
    return f"{name} ({i})"


def import_spatial(recording_dir: Path | str, defs: list[dict]) -> int:
    """Append parsed defs (from ``load_spatial_csv`` or ``read_defs``) onto a recording
    as NEW ids — never clobbering existing definitions. A name that collides with an
    existing one gets a `` (2)`` suffix. Validates the whole set before writing any
    (all-or-nothing). Returns the number appended."""
    prepared: list[tuple[str, str, Keyframes, ActiveInterval, int]] = []
    for d in defs:
        kind = d["kind"]
        k = _kind(kind)
        active = d.get("active")
        if active is not None and not k.has_active:
            raise ValueError(f"{kind!r} does not support an active interval")
        _validate_active(active)
        n_vertices = _validate_keyframes(d["keyframes"], k.min_vertices, k.max_vertices)
        prepared.append((kind, d["name"], d["keyframes"], active, n_vertices))
    for kind, name, keyframes, active, n_vertices in prepared:
        def_id = next_id(recording_dir, kind)
        write_keyframes(recording_dir, kind, def_id, keyframes)
        append_to_manifest(
            recording_dir,
            kind,
            def_id,
            _unique_name(recording_dir, kind, stored_name(name)),
            n_vertices,
            active,
        )
    return len(prepared)


def prepare_for_arrival(
    defs: list[dict],
) -> tuple[list[dict], list[str], list[tuple[str, str]]]:
    """Apply the cross-recording policy to already-parsed ``defs``.

    Returns ``(defs, flattened, refused)``: what may travel, the names whose keyframe
    motion was dropped, and ``[(name, reason), ...]`` for what may not.

    **Two attributes of a definition are claims about one recording's own timeline**, and a
    frame index means nothing on a different recording — the two are not the same length
    and nothing happens at the same moment in both. They are handled differently on
    purpose, because the app can check one of them and cannot check the other:

    - **Keyframe motion is flattened** to the polygon at frame 0 (the first keyframe's
      vertices — ``polygon_at_frame`` clamps below it). A moving polygon tracks *this*
      camera's pan; a static one states the rig geometry, which is the part that carries
      over. The result is a plausible starting point that the scene-geometry editor draws,
      so a wrong one is visible before anybody trusts it.

    - **An active window refuses the trip.** Neither answer is safe: keeping ``(5000,
      9000)`` on a 500-frame recording leaves a region that exists and applies to no frame,
      and dropping it widens the user's claim to the whole video, inventing analysis rows
      where they meant none. Widening is the worse of the two — wrong information rather
      than missing — and unlike the geometry there is no way to *see* it happened, because
      the editor draws one frame at a time.

    **A field of view travels alone**: ``defs`` holding more than one is refused, naming them.

    Pure, and takes parsed defs rather than a directory, so a setup **pulled** from a CSV
    obeys the same rules as one **pushed** from a recording. They were three policies
    before this: import appended and suffixed a collision, the field-of-view import
    replaced, and only the push flattened or refused anything.

    Raises:
        ValueError: ``defs`` holds more than one field of view.
    """
    arenas = [d["name"] for d in defs if d["kind"] == "arena"]
    if len(arenas) > 1:
        # A recording has one field of view; two break the batch tracking picker, which
        # intersects arena ids across recordings. Taking every one made two (#31).
        raise ValueError(
            f"the source holds {len(arenas)} fields of view "
            f"({', '.join(map(repr, arenas))}); a recording has one, so keep the one you "
            "want and import again"
        )
    out: list[dict] = []
    flattened: list[str] = []
    refused: list[tuple[str, str]] = []
    for d in defs:
        name = d["name"]
        active = d.get("active")
        if active:
            start, end = (
                (active["start"], active["end"])
                if isinstance(active, dict)
                else (active[0], active[1])
            )
            refused.append(
                (
                    name,
                    f"has an active frame window ({start}–{end}), which belongs to one "
                    "recording's timeline — define it on that recording",
                )
            )
            continue
        keyframes = d["keyframes"]
        if len(keyframes) > 1:
            flattened.append(name)
        pk = PolygonKeyframes(n_vertices=len(keyframes[0][1]), keyframes=keyframes)
        vertices = [(float(x), float(y)) for x, y in core_utils.polygon_at_frame(pk, 0)]
        out.append({"kind": d["kind"], "name": name, "keyframes": [(0, vertices)]})
    return out, flattened, refused


def read_defs(source_dir: Path | str, kind: str) -> list[dict]:
    """``source_dir``'s definitions of ``kind`` as parsed dicts, policy **not** applied.

    The read half only. :func:`prepare_for_arrival` is the policy, and it runs once, at the
    destination — applying it here as well would flatten an already-flat set and report
    nothing the second time, losing the first pass's account of what it changed.
    """
    _kind(kind)
    out = []
    for entry in list_defs(source_dir, kind):
        pk = read_keyframes(source_dir, kind, entry["id"], entry["n_vertices"])
        out.append(
            {
                "kind": kind,
                "name": entry["name"],
                "keyframes": pk.keyframes,
                "active": entry.get("active"),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Distance calibration — pixels ↔ real-world units (§10)
# --------------------------------------------------------------------------- #
CALIBRATION_KEY = "calibration"


# A resize that rounds each side to an even pixel count moves the aspect ratio by well
# under 1%; a real change of shape (16:9 -> 4:3 is 25%) is far outside it.
ASPECT_TOLERANCE = 0.01


def _pixel_distance(points, width: int, height: int) -> float:
    (x1, y1), (x2, y2) = points
    return math.hypot((x2 - x1) * width, (y2 - y1) * height)


def _write_calibration(recording_dir: Path | str, block: dict) -> None:
    data = recording_store.read_recording(recording_dir)
    data[CALIBRATION_KEY] = block
    recording_store.write_recording(recording_dir, data)


def write_calibration_from_points(
    recording_dir: Path | str,
    points,
    distance_mm: float,
    resolution: tuple[int, int],
) -> dict:
    """Store a distance calibration from a two-point pick a known real distance apart.
    ``points`` are two normalized ``[0, 1]`` ``(x, y)``; ``resolution`` = ``(width,
    height)`` the pick was made at. Derives ``pixels_per_mm`` and writes the
    ``calibration`` block to recording.yaml; returns the block.

    Raises:
        ValueError: if the two points coincide or ``distance_mm <= 0``.
    """
    if distance_mm <= 0:
        raise ValueError(f"distance_mm must be > 0, got {distance_mm}")
    width, height = int(resolution[0]), int(resolution[1])
    pixel_dist = _pixel_distance(points, width, height)
    if pixel_dist <= 0:
        raise ValueError("the two calibration points must be distinct")
    block = {
        "schema_version": 1,
        "points": [
            [float(points[0][0]), float(points[0][1])],
            [float(points[1][0]), float(points[1][1])],
        ],
        "distance_mm": float(distance_mm),
        "pixels_per_mm": pixel_dist / distance_mm,
        "ref_resolution": [width, height],
    }
    _write_calibration(recording_dir, block)
    return block


def write_calibration_scalar(
    recording_dir: Path | str,
    pixels_per_mm: float,
    resolution: tuple[int, int],
) -> dict:
    """Store a distance calibration directly as a ``pixels_per_mm`` scalar (the CLI/API
    path — no two-point pick). ``resolution`` = ``(width, height)`` it is measured against.

    Raises:
        ValueError: if ``pixels_per_mm <= 0``.
    """
    if pixels_per_mm <= 0:
        raise ValueError(f"pixels_per_mm must be > 0, got {pixels_per_mm}")
    block = {
        "schema_version": 1,
        "points": None,
        "distance_mm": None,
        "pixels_per_mm": float(pixels_per_mm),
        "ref_resolution": [int(resolution[0]), int(resolution[1])],
    }
    _write_calibration(recording_dir, block)
    return block


def read_calibration(recording_dir: Path | str) -> dict | None:
    """The recording's ``calibration`` block, or ``None`` if not set."""
    return recording_store.read_recording(recording_dir).get(CALIBRATION_KEY)


def write_calibration_block(
    recording_dir: Path | str,
    block: dict,
    resolution: tuple[int, int] | None = None,
) -> dict:
    """Store a calibration block wholesale — the carry-over path, where the block is
    already derived (copied from another recording or parsed from a CSV) rather than being
    computed from a pick.

    The block keeps the ``ref_resolution`` it was measured at; ``pixels_per_mm_at``
    rescales on read, which is only honest for a video of the same shape. ``resolution``
    is the destination video's ``(width, height)``, or ``None`` when it has no video yet
    (the read then refuses instead).

    Raises:
        ValueError: if the block has no positive ``pixels_per_mm`` or no resolution, or
            its aspect ratio differs from ``resolution``'s.
    """
    if float(block.get("pixels_per_mm") or 0) <= 0:
        raise ValueError("calibration block needs a positive pixels_per_mm")
    if len(block.get("ref_resolution") or []) != 2:
        raise ValueError("calibration block needs a [width, height] ref_resolution")
    if resolution is not None:
        refusal = calibration_aspect_refusal(block, *resolution)
        if refusal is not None:
            raise ValueError(refusal)
    stored = {
        "schema_version": 1,
        "points": block.get("points"),
        "distance_mm": block.get("distance_mm"),
        "pixels_per_mm": float(block["pixels_per_mm"]),
        "ref_resolution": [
            int(block["ref_resolution"][0]),
            int(block["ref_resolution"][1]),
        ],
    }
    _write_calibration(recording_dir, stored)
    return stored


def copy_calibration(
    source_dir: Path | str,
    dest_dir: Path | str,
    resolution: tuple[int, int] | None = None,
) -> dict:
    """Copy the distance calibration from one recording onto another, replacing whatever
    the destination had. ``resolution`` is the destination video's — see
    ``write_calibration_block``.

    Raises:
        ValueError: if the source has no calibration, or its aspect ratio differs from
            ``resolution``'s.
    """
    block = read_calibration(source_dir)
    if block is None:
        raise ValueError("the source recording has no distance calibration")
    return write_calibration_block(dest_dir, block, resolution)


def delete_calibration(recording_dir: Path | str) -> None:
    """Remove the calibration block (idempotent)."""
    data = recording_store.read_recording(recording_dir)
    if CALIBRATION_KEY in data:
        del data[CALIBRATION_KEY]
        recording_store.write_recording(recording_dir, data)


def calibration_aspect_refusal(
    calibration: dict, width: int, height: int
) -> str | None:
    """Why ``calibration`` cannot give a scale for a ``(width, height)`` video, or
    ``None`` when it can.

    One ``pixels_per_mm`` is isotropic: it holds only while the video is a uniform resize
    of the one it was measured on. Across an aspect change the horizontal and vertical
    scales differ, so any single number is wrong in some direction — the old width-ratio
    rescale was far off for a vertical pick, and a pick recomputed from its
    points was right only along the ruler's own direction.
    """
    ref_w, ref_h = (int(v) for v in calibration["ref_resolution"])
    ratio = (width / height) / (ref_w / ref_h)
    if abs(ratio - 1.0) <= ASPECT_TOLERANCE:
        return None
    return (
        f"different video shape — calibrated on {ref_w}x{ref_h}, this video is "
        f"{width}x{height}. Another shape is another rig, and one pixels-per-mm cannot "
        "describe it: calibrate this recording on its own video instead"
    )


def pixels_per_mm_at(calibration: dict, width: int, height: int) -> float:
    """``pixels_per_mm`` for a video at ``(width, height)``: the stored scalar rescaled
    by the resize factor. A two-point pick and a typed scalar read the same, because
    both are the same number at ``ref_resolution``.

    Raises:
        ValueError: the video's aspect ratio differs from ``ref_resolution``'s — see
            ``calibration_aspect_refusal``.
    """
    refusal = calibration_aspect_refusal(calibration, width, height)
    if refusal is not None:
        raise UserInputError(refusal)
    return calibration["pixels_per_mm"] * (width / calibration["ref_resolution"][0])


# --------------------------------------------------------------------------- #
# Arena back-compat wrappers (kind="arena"; behavior unchanged)
# --------------------------------------------------------------------------- #
def write_arena_keyframes(
    recording_dir: Path | str, arena_id: str, keyframes: Keyframes
) -> int:
    """Write ``arenas/<arena_id>/keyframes.csv``; returns the polygon's n_vertices."""
    return write_keyframes(recording_dir, "arena", arena_id, keyframes)


def read_arena_keyframes(
    recording_dir: Path | str, arena_id: str, n_vertices: int
) -> PolygonKeyframes:
    """Load ``arenas/<arena_id>/keyframes.csv`` into a ``PolygonKeyframes``."""
    return read_keyframes(recording_dir, "arena", arena_id, n_vertices)


def append_arena_to_recording_yaml(
    recording_dir: Path | str, arena_id: str, name: str, n_vertices: int
) -> None:
    """Append ``{id, name, n_vertices}`` to top-level ``arenas:`` in recording.yaml."""
    append_to_manifest(recording_dir, "arena", arena_id, name, n_vertices)


def delete_arena(recording_dir: Path | str, arena_id: str) -> None:
    """Remove an arena's keyframes dir + its recording.yaml entry (idempotent)."""
    delete_def(recording_dir, "arena", arena_id)


def list_arenas(recording_dir: Path | str) -> list[dict]:
    """Return the ``arenas:`` list from recording.yaml (empty list if none)."""
    return list_defs(recording_dir, "arena")


def next_arena_id(recording_dir: Path | str) -> str:
    """Pick the lowest free ``arena_NNN`` id (3-digit zero-padded, 1-based).

    Deliberately not the never-reused allocator the other kinds use: every record naming an
    arena is a tracking run, and while one exists the arena can be neither deleted nor
    redefined (``Recording.delete_arena`` / ``clear_arenas`` refuse), so a freed arena id is
    named by nothing. The scene editor redefines the whole set on Save and relies on this
    handing the same ids back.
    """
    k = _kind("arena")
    taken = {entry.get("id") for entry in list_defs(recording_dir, "arena")}
    i = 1
    while f"{k.id_prefix}_{i:03d}" in taken:
        i += 1
    return f"{k.id_prefix}_{i:03d}"
