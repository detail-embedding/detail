"""Experiment export / import — portable single-archive bundles (Sections 1-2, 15).

An experiment is already self-contained with relative paths (Tenet 6), so copying it
"just works". Export packages an experiment's tree — optionally **slim** (no source
video, no datasets) — into one ``tar`` with a top-level ``export.yaml`` manifest (+
a payload sha256), and import validates that manifest and places the experiment into
the managed workspace. No path rewriting; this adds selectivity, an integrity
manifest, and safe placement (create, or replace a same-id experiment).

**Both directions report progress** (#916), because both are long and neither is a
run: a slim export of a real experiment is tens of thousands of files and many GB, read twice
(digest, then tar). They used to report nothing at all, so the only sign of an
export was a button reading "Exporting…" until it stopped.

**Read twice means the two reads must see the same bytes.** The digest is taken in one
pass and the tar written in a second, so a file rewritten between them is packed with a
digest that no longer describes it — and nothing notices until an import has extracted
every member and fails its integrity check. An export run made that certain rather than
possible: its own ``job.json`` lives in ``exports/`` and was rewritten by every progress
report. So ``exports/`` is left out, and any other file that changes between the passes
stops the export there, naming the file.
"""

import hashlib
import io
import os
import re
import shutil
import tarfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core.errors import UserInputError
from detail.core.progress import ThrottledProgress
from detail.storage import experiment as experiment_store
from detail.storage import integrity
from detail.storage import video_roots
from detail.storage.user_yaml import parse_user_yaml

SCHEMA_VERSION = 1
MANIFEST_NAME = "export.yaml"
_ROOT = "experiment"  # the experiment tree lives under this prefix inside the archive
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_TEXT_SUFFIXES = {".yaml", ".yml", ".csv"}  # scanned for absolute paths (Tenet 6)
#: Where an absolute path starts: a home folder on macOS or Linux, or any Windows drive. Not
#: inside a word, so a URL's ``/home/`` or a time's ``12:30`` is not one.
_ABS_PATH = re.compile(r"(?<![\w.])/(?:Users|home)/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
#: YAML keys whose values are text a person wrote, or a recorded error message: prose that
#: may quote a path, never a path DETAIL resolves.
_FREE_TEXT_KEYS = frozenset({"description", "notes", "error"})
#: The experiment's own metadata table: the user's columns, which DETAIL stores and shows but
#: never resolves as paths (a ``raw_file`` column naming where the video came from is data).
_USER_TABLES = frozenset({"metadata.csv"})
_CHUNK = 1 << 20

#: The phases an export reports, in order.
STAGE_SCANNING = "scanning"
STAGE_CHECKING = "checking"
STAGE_HASHING = "hashing"
STAGE_PACKING = "packing"
#: The phases an import reports, in order.
STAGE_READING = "reading"
STAGE_EXTRACTING = "extracting"
STAGE_VERIFYING = "verifying"
#: Only when a recording's video is referenced in place: looking where the exporting machine had
#: it. A few small reads per video folder, so not weighed in the bar.
STAGE_FINDING_VIDEO = "finding video"
STAGE_PLACING = "placing"

#: ``progress(stage, fraction, detail)`` — the synchronous shape the CLI's `_StageProgress`, the
#: GUI's `run_progress` and an export run's `report` all take.
Progress = Callable[[str, float, str], None]

#: An archive is written under this suffix and renamed when it is complete, so an export that
#: fails or is killed part-way never leaves a file carrying the name of a finished one.
_PARTIAL_SUFFIX = ".partial"


class ExperimentChangedError(RuntimeError):
    """A file changed between the digest pass and the pack pass of an export."""


@dataclass(frozen=True)
class _PassCost:
    """What one pass over the payload costs a file: a fixed overhead, plus its bytes at a rate.

    The bar's fraction is what every ETA extrapolates from, so it has to track *time* rather than
    count things — and neither count alone does. Measured on a real experiment, a
    file's fixed cost and an experiment's file count weigh about as much as its bytes. A
    bar of bytes alone would stall through every run of small zarr chunks; a bar of files
    alone through every large one. The absolute numbers belong to the box they were
    measured on; what the fraction needs is their *ratios*, which travel much better.
    """

    per_file_s: float
    bytes_per_s: float

    def of(self, size: int) -> float:
        return self.per_file_s + size / self.bytes_per_s


#: sha256 of every payload byte. The portability scan (read a text file, look for a marker)
#: measured near enough the same per file, so it is weighed the same.
_HASH = _PassCost(per_file_s=0.00040, bytes_per_s=1.7e9)
#: Plain tar: read every byte and write it again.
_PACK = _PassCost(per_file_s=0.00022, bytes_per_s=1.1e9)
#: gzip is CPU-bound: many times slower than a plain tar, for almost no compression on
#: zarr chunks that are already compressed. Weighed as it costs, or a gzip export's bar would
#: race through hashing and crawl through packing.
_PACK_GZ = _PassCost(per_file_s=0.00022, bytes_per_s=58e6)
#: Extraction makes a file per member and writes its bytes.
_EXTRACT = _PassCost(per_file_s=0.0010, bytes_per_s=840e6)
_EXTRACT_GZ = _PassCost(per_file_s=0.0010, bytes_per_s=594e6)


class _Bar:
    """One bar over every pass of an archive operation, credited in measured seconds of work.

    Throttled through `ThrottledProgress` — never a hand-rolled counter — so crediting every chunk
    is cheap. A phase change is forced past the throttle, so the label never lags the work.
    """

    def __init__(self, progress: Progress | None) -> None:
        self._emit = ThrottledProgress(
            None if progress is None else (lambda event: progress(*event))
        )
        self.total_s = 0.0
        self._done_s = 0.0
        self._stage = ""

    @property
    def fraction(self) -> float:
        if self.total_s <= 0:
            return 0.0
        return min(1.0, self._done_s / self.total_s)

    def phase(self, stage: str, detail: str = "") -> None:
        self._stage = stage
        self._emit((stage, self.fraction, detail), force=True)

    def note(self, detail: str) -> None:
        """Say something without moving the bar — a count while the total is still unknown."""
        self._emit((self._stage, self.fraction, detail))

    def credit(self, seconds: float, detail: str) -> None:
        self._done_s += seconds
        self._emit((self._stage, self.fraction, detail))

    def finish(self, stage: str, detail: str) -> None:
        self._stage = stage
        self._done_s = self.total_s
        self._emit((stage, 1.0, detail), force=True)


class _Meter:
    """Credit one pass's files to the bar: bytes as they are read, the per-file cost as each ends."""

    def __init__(self, bar: _Bar, cost: _PassCost, n_files: int) -> None:
        self._bar = bar
        self._cost = cost
        self._n_files = n_files
        self._files_done = 0

    def detail(self) -> str:
        return f"{self._files_done:,} / {self._n_files:,} files"

    def read(self, nbytes: int) -> None:
        self._bar.credit(nbytes / self._cost.bytes_per_s, self.detail())

    def done(self, *_args) -> None:
        self._files_done += 1
        self._bar.credit(self._cost.per_file_s, self.detail())


class _MeteredReader:
    """A source file that reports each read, so one large file moves the bar while it is packed.

    Without it a 1.5 GB member is one step: `tarfile` copies it in a loop of its own.
    """

    def __init__(self, handle, on_read: Callable[[int], None]) -> None:
        self._handle = handle
        self._on_read = on_read

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        if data:
            self._on_read(len(data))
        return data


class _PositionReader:
    """An archive that reports how far into itself it has been read.

    Extraction is measured by **archive bytes consumed**: `tarfile` gives no hook inside a
    member's copy, and the position in the file is the same measure whether the archive is
    plain or gzipped — a gzipped archive is read through this, compressed bytes and all. Only
    forward movement counts, because opening ``r:*`` probes the plain-tar reader first and
    seeks back when it fails.
    """

    def __init__(self, handle, on_advance: Callable[[int], None]) -> None:
        self._handle = handle
        self._on_advance = on_advance
        self._furthest = 0

    @property
    def name(self):
        return self._handle.name

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        position = self._handle.tell()
        if position > self._furthest:
            self._on_advance(position - self._furthest)
            self._furthest = position
        return data

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()


def _included_sizes(
    experiment_dir: Path,
    *,
    include_video: bool,
    include_datasets: bool,
    on_found: Callable[[int], None] | None = None,
) -> list[tuple[str, int]]:
    """``(relative posix path, size)`` of every file to pack, applying the slim filters.

    One ``os.scandir`` walk rather than ``rglob`` and then a ``stat`` per file: the size comes
    with the directory listing on Windows, and an experiment is tens of thousands of files — on a
    network share, tens of thousands of round trips. ``datasets/`` is not descended into at all
    when it is excluded. A symlinked directory is not descended into and a symlinked file is
    packed, which is what ``rglob`` did (Python 3.12 walks with ``follow_symlinks=False``).

    ``on_found(n)`` is called with the running count, because on a large experiment the walk is
    itself a wait, and it is the one phase whose total cannot be known in advance.

    ``exports/`` is never packed. It holds export **run records** — where this machine wrote
    copies, as root names that mean nothing on another one — and while an export runs, its own
    record is in there being rewritten by every progress report (see the module docstring).
    """
    root = Path(experiment_dir)
    left_out = {experiment_store.exports_dir(root)}
    if not include_datasets:
        left_out.add(root / "datasets")
    out: list[tuple[str, int]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if path in left_out:
                        continue
                    pending.append(path)
                    continue
                if not entry.is_file():
                    continue
                if not include_video and path.suffix.lower() in _VIDEO_SUFFIXES:
                    continue
                out.append((path.relative_to(root).as_posix(), entry.stat().st_size))
                if on_found is not None:
                    on_found(len(out))
    return sorted(out)


def _included_files(
    experiment_dir: Path, *, include_video: bool, include_datasets: bool
) -> list[str]:
    """Relative posix paths of the files to pack, applying the slim filters."""
    return [
        rel
        for rel, _size in _included_sizes(
            experiment_dir,
            include_video=include_video,
            include_datasets=include_datasets,
        )
    ]


def _is_foreign_owned(parts: tuple[str, ...]) -> bool:
    """True for files DETAIL stores but never parses, so a machine path inside one
    is the producer's business, not a Tenet 6 violation:

    - ``models/<id>/run/`` — trainer-owned scratch (Ultralytics bakes its own
      absolute paths into ``args.yaml``; §12).
    - ``recordings/<id>/streams/<name>/raw/`` — importer source files, stored
      verbatim by §3. BORIS writes the annotator's media path into its export
      columns. The pipeline reads ``streams/<name>/canonical/`` instead, which is
      DETAIL-written and stays relative.
    """
    if parts[:1] == ("models",) and "run" in parts:
        return True
    # Anchored by position so a stream *named* "raw" isn't mistaken for the dir.
    return (
        parts[:1] == ("recordings",)
        and parts[2:3] == ("streams",)
        and parts[4:5] == ("raw",)
    )


def _is_scanned(rel: str) -> bool:
    """Whether the portability scan reads ``rel`` — one answer for the scan and for its weight."""
    path = Path(rel)
    return (
        path.suffix.lower() in _TEXT_SUFFIXES
        and not _is_foreign_owned(path.parts)
        and rel not in _USER_TABLES
    )


def _path_strings(node, *, free_text: bool = False):
    """Every string in a parsed YAML that holds an absolute path, outside free-text keys."""
    if free_text:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _path_strings(key)
            yield from _path_strings(value, free_text=str(key) in _FREE_TEXT_KEYS)
    elif isinstance(node, list):
        for item in node:
            yield from _path_strings(item)
    elif isinstance(node, str) and _ABS_PATH.search(node):
        yield node


def _line_offenses(
    rel: str, lines: list[str], only: list[str] | None = None
) -> list[str]:
    """``rel line N: <path>`` for each line holding an absolute path.

    ``only`` limits the lines to those holding one of these paths (the strings a YAML parse
    found outside free-text keys); a path the parse found on no single line is still named.
    """
    found = []
    wanted = None if only is None else {_path_token(text) for text in only}
    for number, line in enumerate(lines, start=1):
        match = _ABS_PATH.search(line)
        if match is None:
            continue
        token = _path_token(line[match.start() :])
        if wanted is not None and not any(w in line for w in wanted):
            continue
        found.append(f"{rel} line {number}: {token}")
        if wanted is not None:
            wanted = {w for w in wanted if w not in line}
    found += [f"{rel}: {w}" for w in sorted(wanted or ())]
    return found


def _path_token(text: str) -> str:
    """The absolute path in ``text``, up to the first space or quote, cut to a readable size."""
    match = _ABS_PATH.search(text)
    token = re.match(r"[^\s'\",]+", text[match.start() :] if match else text)
    token = token.group(0) if token else text.strip()
    return token if len(token) <= 80 else token[:77] + "..."


def file_portability_offenses(rel: str, text: str) -> list[str]:
    """Where one experiment file holds an absolute path DETAIL would resolve (Tenet 6).

    A YAML is scanned value by value, past :data:`_FREE_TEXT_KEYS` (a description, a note, a
    recorded error may quote a folder and is never resolved), so a path in prose does not make
    an experiment unexportable. A CSV DETAIL writes, or a YAML that does not parse, is scanned
    line by line, as before. Each offense names the file, the line and the path.
    """
    lines = text.splitlines()
    if Path(rel).suffix.lower() in (".yaml", ".yml"):
        try:
            data = parse_user_yaml(text, rel)
        except UserInputError:
            return _line_offenses(rel, lines)
        strings = list(_path_strings(data))
        return _line_offenses(rel, lines, only=strings) if strings else []
    return _line_offenses(rel, lines)


def portability_offenses(
    experiment_dir: Path | str,
    included: list[str],
    *,
    on_file: Callable[[str], None] | None = None,
) -> list[str]:
    """Where DETAIL files hold an absolute path (Tenet 6 violation), as ``file line N: path``,
    scoped past files DETAIL only stores and never parses (see :func:`_is_foreign_owned`) and
    the user's own metadata table, and past free text (:func:`file_portability_offenses`).

    ``on_file(rel)`` is called after each file the scan reads."""
    experiment_dir = Path(experiment_dir)
    offenders: list[str] = []
    for rel in included:
        if not _is_scanned(rel):
            continue
        text = (experiment_dir / rel).read_text(errors="replace", encoding="utf-8-sig")
        offenders += file_portability_offenses(rel, text)
        if on_file is not None:
            on_file(rel)
    return offenders


def _payload_digest(
    base_dir: Path | str,
    relpaths: list[str],
    *,
    on_read: Callable[[int], None] | None = None,
    on_file: Callable[[str], None] | None = None,
    signatures: dict[str, tuple[int, int]] | None = None,
) -> str:
    """sha256 over the listed files (sorted by posix path, hashing path + bytes) —
    a transfer-integrity guard. Same ordering on export + import.

    ``on_read(nbytes)`` is called per chunk read and ``on_file(rel)`` after each file.
    ``signatures`` receives each file's `_signature` as it stood when hashed, for the pack pass
    to compare against.
    """
    base_dir = Path(base_dir)
    digest = hashlib.sha256()
    for rel in sorted(relpaths):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(base_dir / rel, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                digest.update(chunk)
                if on_read is not None:
                    on_read(len(chunk))
            if signatures is not None:
                signatures[rel] = _signature(f)
        if on_file is not None:
            on_file(rel)
    return digest.hexdigest()


def _referenced_recordings(experiment_dir: Path):
    """``(recording_dir, recording, parts)`` for each recording with a part referenced in place.

    ``parts`` are the referenced ones, as dicts inside ``recording``, so a caller that edits them
    and writes ``recording`` back has edited the file.
    """
    from detail.storage import recording as recording_store

    recordings_dir = experiment_dir / "recordings"
    if not recordings_dir.is_dir():
        return
    for directory in sorted(p for p in recordings_dir.iterdir() if p.is_dir()):
        try:
            data = recording_store.read_recording(directory)
        except (FileNotFoundError, ValueError):
            continue
        files = (data.get("video") or {}).get("files", [])
        parts = [part for part in files if part.get("source")]
        if parts:
            yield directory, data, parts


def referenced_video_recordings(experiment_dir: Path | str) -> list[str]:
    """Recordings whose video is **referenced in place** rather than held by DETAIL.

    Those files live under a machine-local video root, outside the experiment tree — so no
    setting of ``include_video`` can pack them, and a bundle carrying them is not what the
    recipient is going to get. Enumerated here so the manifest can say so.
    """
    return [d.name for d, _data, _parts in _referenced_recordings(Path(experiment_dir))]


def _video_root_locations(experiment_dir: Path) -> dict[str, str]:
    """Where this machine keeps each video root the experiment's recordings name.

    Written into the manifest, never into the experiment: an experiment file holds no machine path
    (Tenet 6), so the alias a part names means nothing on another machine. When the videos sit on
    a server both machines reach, this is the one fact the importing machine is missing (#916).
    """
    aliases = {
        part["source"].get("root")
        for _d, _data, parts in _referenced_recordings(experiment_dir)
        for part in parts
    }
    here = video_roots.list_video_roots()
    return {alias: here[alias].as_posix() for alias in sorted(aliases) if alias in here}


def export_experiment(
    experiment_dir: Path | str,
    dest: Path | str,
    *,
    include_video: bool = False,
    include_datasets: bool = False,
    gzip: bool = False,
    progress: Progress | None = None,
) -> Path:
    """Pack ``experiment_dir`` into a single ``tar`` at ``dest`` with an ``export.yaml``
    manifest. Default is **slim** (outputs + metadata + models; no source video, no
    datasets). Refuses if a DETAIL file holds an absolute path (Tenet 6).

    ``progress(stage, fraction, detail)`` reports ``scanning`` while the tree is walked (a
    running count of files, at 0% — the one phase whose size is not known in advance), then
    ``checking``, ``hashing`` and ``packing``: one fraction across all three, each file weighed
    by what each pass costs it (`_PassCost`), and moving inside a large file as well as between
    files.

    **The manifest records what was packed, not what was asked for.** ``video_included`` used
    to be the ``include_video`` argument, which is a different thing whenever a recording's
    video is *referenced in place*: those files live under a machine-local video root, outside
    this tree, so no setting can pack them. A recipient reading ``video_included: true`` and
    finding no video has been told something untrue by the file whose job is to say what is
    inside — the same shape as a status written before the work that justifies it.
    ``referenced_video`` names the recordings whose video could not travel, so the recipient
    knows which root to register rather than concluding the bundle is broken, and
    ``video_roots`` says where this machine keeps each of their roots, so an import can look
    there first (`import_experiment`).

    The archive is written as ``<dest>.partial`` and renamed when complete, so a failed,
    cancelled or killed export leaves nothing that looks like a finished one.

    Raises:
        ValueError: A DETAIL file holds an absolute path.
        ExperimentChangedError: A file changed between the digest and the pack — something
            is writing to the experiment — so the archive would fail its integrity check.
    """
    experiment_dir = Path(experiment_dir)
    if not (experiment_dir / experiment_store.CONFIG_FILENAME).is_file():
        raise FileNotFoundError(f"not an experiment (no config.yaml): {experiment_dir}")
    dest = Path(dest)
    bar = _Bar(progress)
    bar.phase(STAGE_SCANNING, "reading the experiment")
    sizes = dict(
        _included_sizes(
            experiment_dir,
            include_video=include_video,
            include_datasets=include_datasets,
            on_found=lambda n: bar.note(f"{n:,} files found"),
        )
    )
    # An archive written inside the tree it packs must not pack itself. `tar.add` used to skip
    # it silently — after the digest had already hashed it, so the bundle failed its own
    # integrity check on import. Dropped here, before either pass sees it.
    partial = dest.with_name(dest.name + _PARTIAL_SUFFIX)
    for written in (dest, partial):
        sizes.pop(_relative_inside(written, experiment_dir), None)
    included = list(sizes)
    scanned = [rel for rel in included if _is_scanned(rel)]
    pack_cost = _PACK_GZ if gzip else _PACK
    bar.total_s = sum(_HASH.of(sizes[rel]) for rel in scanned) + sum(
        _HASH.of(size) + pack_cost.of(size) for size in sizes.values()
    )

    checking = _Meter(bar, _HASH, len(scanned))
    bar.phase(STAGE_CHECKING, checking.detail())
    offenders = portability_offenses(
        experiment_dir,
        included,
        on_file=lambda rel: (checking.read(sizes[rel]), checking.done()),
    )
    if offenders:
        more = f" (and {len(offenders) - 5} more)" if len(offenders) > 5 else ""
        raise UserInputError(
            "absolute paths in experiment files, which would not exist on another machine "
            f"(Tenet 6): {'; '.join(offenders[:5])}{more}. A path DETAIL reads must be "
            "relative to the experiment: edit the file, or move the video under a video "
            "root (`detail video-root set`)"
        )
    config = experiment_store.read_config(experiment_dir)
    packed_video = any(Path(rel).suffix.lower() in _VIDEO_SUFFIXES for rel in included)
    packed_datasets = any(Path(rel).parts[:1] == ("datasets",) for rel in included)
    referenced = referenced_video_recordings(experiment_dir)
    hashing = _Meter(bar, _HASH, len(included))
    bar.phase(STAGE_HASHING, hashing.detail())
    hashed: dict[str, tuple[int, int]] = {}
    payload_sha256 = _payload_digest(
        experiment_dir,
        included,
        on_read=hashing.read,
        on_file=hashing.done,
        signatures=hashed,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "id": experiment_dir.name,
        "name": config.get("name", experiment_dir.name),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contents": {
            "video_included": packed_video,
            "video_requested": include_video,
            "referenced_video": referenced,
            "datasets_included": packed_datasets,
            "models_included": True,
        },
        "video_roots": _video_root_locations(experiment_dir),
        "n_files": len(included),
        "payload_sha256": payload_sha256,
    }
    blob = yaml.dump(manifest, sort_keys=False).encode("utf-8")
    packing = _Meter(bar, pack_cost, len(included))
    bar.phase(STAGE_PACKING, packing.detail())
    dest.parent.mkdir(parents=True, exist_ok=True)  # every check above has passed
    try:
        # `copybufsize` is the read size inside `tarfile`'s copy loop, and so how often a large
        # file reports: its default is 16 KiB, a callback every 16 KiB of a 17 GB payload.
        with tarfile.open(partial, "w:gz" if gzip else "w", copybufsize=_CHUNK) as tar:
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
            for rel in included:
                with open(experiment_dir / rel, "rb") as handle:
                    _refuse_if_changed(rel, hashed[rel], _signature(handle))
                    # From the open handle, so the header's size is the one just compared.
                    member = tar.gettarinfo(arcname=f"{_ROOT}/{rel}", fileobj=handle)
                    tar.addfile(member, _MeteredReader(handle, packing.read))
                    _refuse_if_changed(rel, hashed[rel], _signature(handle))
                packing.done()
        atomic.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    bar.finish(STAGE_PACKING, dest.name)
    return dest


def _signature(handle) -> tuple[int, int]:
    """``(size, mtime_ns)`` of an open file, from the file itself rather than a directory listing.

    ``fstat`` rather than the scan's ``DirEntry.stat``: on Windows a listing's entry is updated
    lazily, so it can disagree with a file nobody is touching and would refuse a sound export.
    """
    stat = os.fstat(handle.fileno())
    return stat.st_size, stat.st_mtime_ns


def _refuse_if_changed(rel: str, hashed: tuple[int, int], now: tuple[int, int]) -> None:
    """Stop the export when ``rel`` is not the file the digest was taken over."""
    if now != hashed:
        raise ExperimentChangedError(
            f"{rel} changed while the experiment was being exported, so the archive would fail "
            "its integrity check on import. Something is still writing to this experiment: let "
            "its running jobs finish, then export again."
        )


def _relative_inside(path: Path, root: Path) -> str | None:
    """``path`` as a posix path relative to ``root`` when it lies inside it, else ``None``."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _not_an_export(archive: Path | str, why: str | BaseException) -> UserInputError:
    """The one message for a file an import cannot read as a DETAIL export: a text file, an
    empty file, a ``.zip``, an archive cut short by an interrupted copy."""
    if isinstance(why, BaseException):
        text = str(why)
        if text.startswith("file could not be opened successfully"):
            # `tarfile.open(mode="r:*")` lists every compression it tried, one per line.
            empty = Path(archive).is_file() and Path(archive).stat().st_size == 0
            why = "the file is empty" if empty else "not a tar archive"
        else:
            why = text or type(why).__name__
    return UserInputError(
        f"{Path(archive).name} is not a DETAIL export, or it is incomplete ({why}); "
        "copy it again, or export the experiment again"
    )


def read_manifest(archive: Path | str) -> dict:
    """Read ``export.yaml`` from a bundle.

    Reads members only until it finds the manifest, which an export writes first. `getmember`
    loads every member header before answering, which on a gzipped bundle is a full
    decompression pass — paid before an import had reported anything at all.

    Raises:
        UserInputError: The file is not a DETAIL export (not a tar, empty, cut short, or
            without a manifest), or its manifest is malformed: not a mapping, an ``id``
            that is not a string, a ``schema_version`` that is not an integer.
    """
    try:
        with tarfile.open(archive, "r:*") as tar:
            member = next((m for m in tar if m.name == MANIFEST_NAME), None)
            if member is None:
                raise _not_an_export(archive, f"no {MANIFEST_NAME}")
            raw = tar.extractfile(member).read()
    except (tarfile.TarError, EOFError, zlib.error) as exc:
        raise _not_an_export(archive, exc) from exc
    data = parse_user_yaml(raw, f"{Path(archive).name}: {MANIFEST_NAME}")
    if not isinstance(data, dict) or "id" not in data:
        raise _not_an_export(archive, "malformed export manifest")
    if not isinstance(data["id"], str):
        raise _not_an_export(
            archive, f"its manifest's id is {data['id']!r}, not a name"
        )
    version = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise _not_an_export(
            archive, f"its manifest's schema_version is {version!r}, not a number"
        )
    return data


def _is_gzip(archive: Path) -> bool:
    with open(archive, "rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


@dataclass(frozen=True)
class _FoundRoot:
    """A video root the archive's recordings name, found where the exporting machine had it."""

    alias: str
    directory: Path
    #: This machine already uses ``alias`` for another folder. Re-pointing it would break the
    #: experiments here that use it, so the imported recordings are pointed elsewhere instead.
    taken: bool


def _holds_video(
    directory: Path, parts: list[dict], *, probe: int | None = None
) -> bool:
    """Whether ``directory`` holds these parts' videos.

    The first part whose file is there decides, by its recorded fingerprint: one matching file is
    the evidence a user's Locate gives (`api/recording.py::_relink_root`), so an import asks no
    more of a folder than that, and no less. The folder is checked first, so a server that is not
    reachable costs one lookup rather than one per part. ``probe`` caps how many parts are looked
    for, for a folder that is only a guess.
    """
    try:
        if not directory.is_dir():
            return False
        for part in parts[:probe]:
            path = directory / part["source"]["rel"]
            if path.is_file():
                current = integrity.file_fingerprint(path)
                break
        else:
            return False
    except OSError:  # a folder this machine may not read is not one it can use
        return False
    recorded = part.get("fingerprint") or {}
    return (current["size"], current["sha256_ends"]) == (
        recorded.get("size"),
        recorded.get("sha256_ends"),
    )


#: How many of a root's videos are looked for in a folder that is only a guess (`_beside`). The
#: right folder holds the first one it is asked for; a wrong one costs a stat per probe, on a
#: share, for every guess.
_BESIDE_PROBE = 3


class _Reachable:
    """Whether a path's drive or network share answers, asked once per share.

    A share this machine cannot reach (a server spelled by a name only the exporting machine
    resolves) makes every lookup under it wait out the network's timeout. Asked per root, that
    was one wait per day-folder of a study; asked per share, it is one.
    """

    def __init__(self) -> None:
        self._answers: dict[str, bool] = {}

    def __call__(self, path: Path) -> bool:
        anchor = path.anchor
        if anchor not in self._answers:
            try:
                self._answers[anchor] = Path(anchor).is_dir() if anchor else True
            except OSError:
                self._answers[anchor] = False
        return self._answers[anchor]


def _beside(anchors: list[Path], alias: str) -> list[Path]:
    """Where a root named ``alias`` would be, next to folders already known to hold video.

    An auto-registered root is named after its folder, deduplicated with a ``_2`` suffix
    (`storage/roots.py::_auto_alias`), and a lab keeps a study's video as one folder per day or
    cohort. So the folder for ``Day3`` sits beside the one ``Day1`` was found in, or under it,
    or is the same folder registered twice. Guesses only: `_holds_video` decides.
    """
    names = [alias]
    deduped = re.fullmatch(r"(.+)_\d+", alias)
    if deduped:
        names.append(deduped.group(1))
    out: list[Path] = []
    for anchor in anchors:
        guesses = [anchor, *(anchor / n for n in names)]
        guesses += [anchor.parent / n for n in names]
        if anchor.parent.name in names:
            guesses.append(anchor.parent)
        out += [g for g in guesses if g not in out]
    return out


def _find_referenced_video(staged: Path, located: dict, bar: _Bar) -> list[_FoundRoot]:
    """The video roots this machine cannot resolve, found where the exporting machine had them,
    or beside the ones that were found.

    Two places to look, in order. The manifest's ``video_roots``, where the exporting machine had
    each root. Then **beside every folder that did hold video** (`_beside`): a study's video is
    one folder per day, and the manifest is no help when the two machines spell the server
    differently, or when the archive predates it (#916). Each folder found is looked beside in
    turn, so one known folder is enough.

    Read-only: nothing is registered or rewritten here, so a cancel during the search leaves
    this machine as it was. A root this machine already resolves to the right videos is left
    alone.
    """
    by_alias: dict[str, list[dict]] = {}
    for _directory, _data, parts in _referenced_recordings(staged):
        for part in parts:
            by_alias.setdefault(part["source"].get("root"), []).append(part)
    found: list[_FoundRoot] = []
    anchors: list[Path] = []
    missing: dict[str, list[dict]] = {}
    reachable = _Reachable()
    for k, (alias, parts) in enumerate(sorted(by_alias.items())):
        bar.phase(STAGE_FINDING_VIDEO, f"{k + 1} / {len(by_alias)} video folders")
        here = video_roots.video_root(alias)
        if here is not None and reachable(here) and _holds_video(here, parts):
            anchors.append(here)
            continue
        there = located.get(alias)
        if (
            there
            and Path(there) != here
            and reachable(Path(there))
            and _holds_video(Path(there), parts)
        ):
            found.append(_FoundRoot(alias, Path(there), taken=here is not None))
            anchors.append(Path(there))
            continue
        missing[alias] = parts
    looked_beside = 0
    while missing and looked_beside < len(anchors):
        anchors_now = anchors[looked_beside:]
        looked_beside = len(anchors)
        for alias, parts in sorted(missing.items()):
            bar.phase(
                STAGE_FINDING_VIDEO, f"looking beside the folders found · {alias}"
            )
            for guess in _beside(anchors_now, alias):
                if _holds_video(guess, parts, probe=_BESIDE_PROBE):
                    here = video_roots.video_root(alias)
                    found.append(_FoundRoot(alias, guess, taken=here is not None))
                    anchors.append(guess)
                    del missing[alias]
                    break
    return found


def _connect_found_video(staged: Path, found: list[_FoundRoot]) -> None:
    """Register each found root on this machine, under its own name where that name is free.

    Where this machine already uses the name for another folder, the found folder gets a root of
    its own (or reuses one that already holds it), and the staged recordings are re-pointed at it.
    """
    from detail.storage import recording as recording_store

    renamed: dict[str, tuple[str, Path]] = {}
    for root in found:
        if not root.taken:
            video_roots.register_video_root(root.alias, root.directory)
            continue
        alias = video_roots.resolve_or_register_root(root.directory)
        inside = root.directory.resolve().relative_to(
            video_roots.video_root(alias).resolve()
        )
        renamed[root.alias] = (alias, inside)
    if not renamed:
        return
    for directory, data, parts in _referenced_recordings(staged):
        moved = [part for part in parts if part["source"].get("root") in renamed]
        for part in moved:
            alias, inside = renamed[part["source"]["root"]]
            rel = (inside / part["source"]["rel"]).as_posix()
            part["source"] = {"root": alias, "rel": rel}
        if moved:
            recording_store.write_recording(directory, data)


def import_experiment(
    archive: Path | str,
    dest_root: Path | str,
    *,
    replace: bool = False,
    name: str | None = None,
    progress: Progress | None = None,
) -> Path:
    """Validate + place an exported experiment under ``dest_root`` (the managed
    workspace). Stages to a temp dir, verifies the payload sha256, then moves into
    place atomically — nothing is placed on a validation error (Tenet 10).

    ``progress(stage, fraction, detail)`` reports ``reading``, ``extracting``, ``verifying``
    and ``placing``. It reaches 1.0 *before* the existing experiment is replaced and is not
    called again, so a callback that raises to cancel (the GUI's ``run_progress``) can stop an
    import at any point that leaves nothing behind, and never after the one step that cannot be
    undone. A cancel mid-extract leaves the staging folder, which the ``finally`` removes.

    **Video referenced in place is looked for where the exporting machine had it** (``finding
    video``, between ``verifying`` and ``placing``). A recording names its video by a root alias
    that resolves through this machine's ``storage.yaml``, so on another machine it resolved to
    nothing, even when both machines reach the same server, until a user located one file by
    hand. Each root this machine cannot resolve is looked for at the manifest's ``video_roots``
    path and registered when a video there matches its recorded fingerprint. A root name this
    machine already uses for another folder is never re-pointed: the imported recordings are
    pointed at a root of this machine's own instead.

    Raises:
        ValueError: Unsupported schema_version, malformed/corrupt archive, or a
            failed integrity check.
        FileExistsError: An experiment with the target id exists and ``replace`` is False.
    """
    archive = Path(archive)
    bar = _Bar(progress)
    bar.phase(STAGE_READING, "the archive's manifest")
    manifest = read_manifest(archive)
    version = manifest.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"export schema_version {version} is newer than supported {SCHEMA_VERSION}"
        )
    experiment_id = name or manifest["id"]
    # the id is a single directory name under the workspace — never a path that
    # could escape it (a malformed --name)
    if experiment_id in ("", ".", "..") or any(
        sep in experiment_id for sep in ("/", "\\")
    ):
        raise ValueError(
            f"invalid experiment name {experiment_id!r}: must be a bare name"
        )
    dest_root = Path(dest_root)
    dest = dest_root / experiment_id
    if dest.exists() and not replace:
        raise FileExistsError(
            f"experiment {experiment_id!r} already exists; pass replace=True to overwrite"
        )
    staging = dest_root / f".import-{experiment_id}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    # The archive's size stands in for the payload's when weighing the verify pass: exact for a
    # plain tar, and within a few percent for a gzipped one, since zarr chunks barely compress.
    n_files = int(manifest.get("n_files") or 0)
    archive_bytes = archive.stat().st_size
    extract_cost = _EXTRACT_GZ if _is_gzip(archive) else _EXTRACT
    bar.total_s = (
        n_files * (extract_cost.per_file_s + _HASH.per_file_s)
        + archive_bytes / extract_cost.bytes_per_s
        + archive_bytes / _HASH.bytes_per_s
    )
    try:
        extracting = _Meter(bar, extract_cost, n_files)
        bar.phase(STAGE_EXTRACTING, extracting.detail())
        with open(archive, "rb") as raw:
            try:
                with tarfile.open(
                    fileobj=_PositionReader(raw, extracting.read), mode="r:*"
                ) as tar:
                    # Member by member rather than `extractall`, so each finished file is
                    # counted; `extract` applies the same 3.12 "data" sandbox (no
                    # abs/traversal) per member.
                    for member in tar:
                        tar.extract(member, staging, filter="data")
                        if member.isfile() and member.name.startswith(f"{_ROOT}/"):
                            extracting.done()
            except (tarfile.ReadError, EOFError, zlib.error) as exc:
                # The manifest came first and read, so the rest was cut short in a copy.
                raise _not_an_export(archive, exc) from exc
        staged = staging / _ROOT
        if not (staged / experiment_store.CONFIG_FILENAME).is_file():
            raise ValueError("export is missing experiment/config.yaml")
        relpaths = [
            p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()
        ]
        expected = manifest.get("payload_sha256")
        verifying = _Meter(bar, _HASH, len(relpaths))
        bar.phase(STAGE_VERIFYING, verifying.detail())
        if (
            expected
            and _payload_digest(
                staged, relpaths, on_read=verifying.read, on_file=verifying.done
            )
            != expected
        ):
            raise ValueError("archive integrity check failed (payload sha256 mismatch)")
        found = _find_referenced_video(staged, manifest.get("video_roots") or {}, bar)
        # The last report, and the last point a raising callback can stop this cleanly:
        # nothing below may emit, or a cancel could land between the delete and the rename.
        bar.finish(
            STAGE_PLACING,
            "replacing the existing experiment" if dest.exists() else experiment_id,
        )
        _connect_found_video(staged, found)
        if dest.exists():
            shutil.rmtree(dest)
        staged.rename(dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest
