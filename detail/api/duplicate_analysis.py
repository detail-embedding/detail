"""Duplicate an analysis run: an exact copy under a new ``an_NN``, the original untouched (§16
"Duplicating a run").

The copy is the whole run directory — the fit, the tree, the node cache, the frame view, the
measures, the history, snapshots, and the profiles and suggestions beside a clustering run — so
the copy opens, and interprets, exactly as the original does. A person duplicates a run to try
something on it (relabel, rebuild its clips at another floor, run a long batch of profiles)
without risking the one they have.

**What is not copied** is what describes a *process* rather than the run: a ``cancel`` or
``stop`` request, a coordinator's pid file, an advisory ``*.lock`` and a half-written
``*.tmp``. The job envelope (``job.json``) is copied — every surface reads a run's status and
timing from it — with its id rewritten and its worker pid dropped, since that worker was the
original's.

**Ids are rewritten where a record names its own run.** Every YAML and JSON document (and
each line of a JSON-lines log) that contains the source id as a whole value has that value
replaced by the new id: ``analysis.yaml``'s ``id``, ``job.json``'s ``job_id``, the behaviour
artifact's ``producer.run_id``, each profile's ``run_id``, a snapshot's. Only a value that *is*
the id is rewritten — a string that merely contains it is left alone — and a document without
the id is copied byte for byte. Zarr stores are copied as bytes: none names its run.

**Visible only when whole.** The new directory is claimed through ``reserve_analysis_id``,
filled, and its ``analysis.yaml`` written last (atomically, ``schema_version`` first), so a
listing never shows a half-copied run; a copy that fails or is cancelled removes its
directory. It refuses while the source is running, since a run being written cannot be
copied whole.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.core.dataclasses import ACTIVE_STATUSES
from detail.core.utils import utc_timestamp
from detail.storage import analysis_runs as analysis_store

#: Files describing a process, not the run: never copied.
SKIPPED_NAMES = frozenset({"cancel", "stop", "coordinator.pid"})
SKIPPED_SUFFIXES = (".tmp", ".lock")
#: Documents whose values are read for the run's own id.
_YAML = (".yaml", ".yml")
_JSON = (".json",)
_JSON_LINES = (".jsonl",)
_CHUNK = 4 * 1024 * 1024
HISTORY_FILENAME = "history.jsonl"
JOB_FILENAME = "job.json"


def copied_files(run_dir: Path) -> list[Path]:
    """Every file of ``run_dir`` a duplicate copies, relative to it, in a stable order."""
    run_dir = Path(run_dir)
    out = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIPPED_NAMES or path.name.endswith(SKIPPED_SUFFIXES):
            continue
        out.append(path.relative_to(run_dir))
    return out


def _in_zarr(rel: Path) -> bool:
    return any(part.endswith(".zarr") for part in rel.parts[:-1])


def rewrite_ids(value, old: str, new: str):
    """``value`` with every string that **is** ``old`` replaced by ``new``, keys included."""
    if isinstance(value, dict):
        return {
            rewrite_ids(k, old, new): rewrite_ids(v, old, new) for k, v in value.items()
        }
    if isinstance(value, list):
        return [rewrite_ids(v, old, new) for v in value]
    if isinstance(value, str) and value == old:
        return new
    return value


def _rewritten(rel: Path, data: bytes, old: str, new: str) -> bytes | None:
    """The document rewritten for the new id, or ``None`` to copy it byte for byte."""
    suffix = rel.suffix.lower()
    if _in_zarr(rel) or suffix not in _YAML + _JSON + _JSON_LINES:
        return None
    if old.encode("utf-8") not in data and rel.name != JOB_FILENAME:
        return None
    text = data.decode("utf-8-sig")
    if suffix in _YAML:
        doc = rewrite_ids(yaml_load.safe_load(text), old, new)
        return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True).encode("utf-8")
    if suffix in _JSON:
        doc = rewrite_ids(json.loads(text), old, new)
        if rel.name == JOB_FILENAME and isinstance(doc, dict):
            doc["pid"] = None  # the original's worker, not this run's
        return json.dumps(doc, indent=2).encode("utf-8")
    lines = []
    for line in text.splitlines():
        lines.append(
            json.dumps(rewrite_ids(json.loads(line), old, new))
            if line.strip()
            else line
        )
    return ("\n".join(lines) + ("\n" if text.endswith("\n") else "")).encode("utf-8")


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    atomic.replace(tmp, path)


def _copy_bytes(src: Path, dst: Path, on_bytes) -> None:
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            chunk = fin.read(_CHUNK)
            if not chunk:
                break
            fout.write(chunk)
            on_bytes(len(chunk))
    shutil.copystat(src, dst)


def duplicate_analysis(
    experiment, run_id: str, *, name: str | None = None, progress=None
) -> str:
    """Copy analysis run ``run_id`` whole to a new ``an_NN``; the new id.

    ``name`` labels the copy (``"<the original's name or id> (copy)"`` when not given).
    ``progress(stage, fraction, detail)`` hears ``copying`` by bytes, file by file; an
    ``OperationCancelled`` raised from it stops the copy and removes what was copied.

    Raises:
        FileNotFoundError: No such analysis run.
        ValueError: The run is still running (or pending).
    """
    from detail.executor import rundir

    experiment.reconcile_analysis_run(run_id)
    src = experiment.analysis_run_dir(run_id)
    record = analysis_store.read_analysis_yaml(src)  # FileNotFoundError if absent
    try:
        envelope_status = rundir.read_status(src).value
    except (FileNotFoundError, KeyError, ValueError):
        envelope_status = None
    for status in (record.get("status"), envelope_status):
        if status in ACTIVE_STATUSES:
            raise ValueError(
                f"analysis run {run_id!r} is still {status}; wait for it to finish (or "
                "cancel it) before duplicating it"
            )
    files = [
        f for f in copied_files(src) if f.as_posix() != analysis_store.ANALYSIS_FILENAME
    ]
    total = sum((src / f).stat().st_size for f in files) or 1
    new_id = analysis_store.reserve_analysis_id(experiment.dir)
    dst = experiment.analysis_run_dir(new_id)
    done = {"bytes": 0}

    def report(detail: str) -> None:
        if progress is not None:
            progress("copying", min(done["bytes"] / total, 1.0), detail)

    try:
        report(f"{len(files)} files")
        for index, rel in enumerate(files):
            source, target = src / rel, dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = None
            if not _in_zarr(rel) and rel.suffix.lower() in _YAML + _JSON + _JSON_LINES:
                data = source.read_bytes()
                rewritten = _rewritten(rel, data, run_id, new_id)
                _write_atomic(target, rewritten if rewritten is not None else data)
                done["bytes"] += len(data)
            else:

                def on_bytes(n, _rel=rel, _i=index):
                    done["bytes"] += n
                    report(f"{_rel.as_posix()} ({_i + 1} of {len(files)})")

                _copy_bytes(source, target, on_bytes)
            report(f"{rel.as_posix()} ({index + 1} of {len(files)})")
        _append_history(dst, run_id)
        copy = rewrite_ids(dict(record), run_id, new_id)
        copy["id"] = new_id
        copy["name"] = name or f"{record.get('name') or run_id} (copy)"
        copy["duplicated_from"] = run_id
        copy["duplicated_at"] = utc_timestamp()
        analysis_store.write_analysis_yaml(dst, copy)  # last: the run exists from here
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    if progress is not None:
        progress("copying", 1.0, f"{run_id} copied to {new_id}")
    return new_id


def _append_history(run_dir: Path, source_id: str) -> None:
    """One provenance line in the copy's ``history.jsonl`` (§21), when it keeps one."""
    path = Path(run_dir) / HISTORY_FILENAME
    if not path.exists():
        return
    line = json.dumps({"at": utc_timestamp(), "op": "duplicate", "from": source_id})
    text = path.read_text(encoding="utf-8-sig")
    if text and not text.endswith("\n"):
        text += "\n"
    _write_atomic(path, (text + line + "\n").encode("utf-8"))
