"""Which finished runs a workspace has dismissed from Active Processes.

A run's own record never says whether someone has looked at it and moved on — that is a fact
about the *view*, not about the run, and it does not belong in `distill.yaml`/`model.yaml`/etc.
alongside data a resume or a reconciliation backstop reads. So it is its own small,
experiment-scoped store, the same idiom as `storage/stream_presets.py` (schema-first YAML, a
sibling ``.lock`` file, atomic write) with a different payload:

```yaml
schema_version: 1
archived:
  distill: [distill_002, distill_005]
  model: [model_001]
```

Archiving is purely a dismiss: it never touches the run's own record, so deleting, resuming, or
reconciling a run is unaffected by whether it is archived, and un-archiving is simply editing this
file back out (or deleting it) rather than an operation this module needs to expose.
"""

import contextlib
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load

SCHEMA_VERSION = 1
_FILENAME = "active_archive.yaml"


def _path(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / _FILENAME


@contextlib.contextmanager
def _lock(experiment_dir: Path | str):
    with atomic.file_lock(_path(experiment_dir).with_suffix(".yaml.lock")):
        yield


def archived_ids(experiment_dir: Path | str) -> dict[str, set[str]]:
    """``{kind: {run_id, ...}}`` for this experiment — ``{}`` if nothing has been archived."""
    path = _path(experiment_dir)
    if not path.is_file():
        return {}
    data = yaml_load.safe_load(atomic.read_text(path)) or {}
    return {kind: set(ids) for kind, ids in (data.get("archived") or {}).items() if ids}


def archive(experiment_dir: Path | str, kind: str, run_id: str) -> None:
    """Mark ``run_id`` (of ``kind``) archived. A no-op if it already is."""
    with _lock(experiment_dir):
        current = archived_ids(experiment_dir)
        ids = current.setdefault(kind, set())
        if run_id in ids:
            return
        ids.add(run_id)
        _write(experiment_dir, current)


def _write(experiment_dir: Path | str, archived: dict[str, set[str]]) -> None:
    path = _path(experiment_dir)
    tmp = path.parent / (path.name + ".tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "archived": {kind: sorted(ids) for kind, ids in archived.items() if ids},
    }
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, sort_keys=False)
    atomic.replace(tmp, path)
