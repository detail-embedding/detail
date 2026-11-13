"""Machine-global tool-run records: ``tools/<tool_id>/<run_id>/tool.yaml`` (§9).

A *tool run* is one execution of an experiment-free ``Tool`` (e.g. Quick segment).
Unlike processing runs, tool runs live outside any experiment — under a machine-global
``tools/`` tree beside the experiments in the managed workspace — but are managed like
project data (listed, opened, deleted). Each run owns a directory: a ``tool.yaml``
manifest plus whatever output the tool wrote (e.g. a ``detections.zarr``). Run ids are
unique across the whole tools workspace, so a single id names a run without its tool.
"""

import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.storage import dirscan
from detail.storage import locations

SCHEMA_VERSION = 1
RUN_FILENAME = "tool.yaml"


def tools_root() -> Path:
    """The machine-global tools workspace (sibling of the experiment dirs)."""
    return locations.experiments_root() / "tools"


def tool_dir(tool_id: str) -> Path:
    return tools_root() / tool_id


def tool_run_dir(tool_id: str, run_id: str) -> Path:
    return tool_dir(tool_id) / run_id


def _run_dirs(tool_id: str | None = None) -> list[Path]:
    """Every run directory (holding a ``tool.yaml``) for one tool, or all tools."""
    tool_dirs = (
        [tool_dir(tool_id)] if tool_id is not None else dirscan.child_dirs(tools_root())
    )
    runs: list[Path] = []
    for tdir in tool_dirs:
        runs += dirscan.dirs_containing(tdir, RUN_FILENAME)
    return runs


def next_run_id() -> str:
    """A ``run_NNN`` id unique across the whole tools workspace (past the highest)."""
    nums = []
    for rdir in _run_dirs():
        try:
            nums.append(int(rdir.name.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return f"run_{max(nums, default=0) + 1:03d}"


def write_tool_run(tool_id: str, run_id: str, record: dict) -> None:
    """Write ``tools/<tool_id>/<run_id>/tool.yaml`` with ``schema_version`` first (Tenet 7).

    Atomic (tmp + ``os.replace``): the GUI thread reads a run's manifest while the run
    worker rewrites it running → complete, so a reader never sees a torn file."""
    rdir = tool_run_dir(tool_id, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in record.items():
        if key != "schema_version":
            ordered[key] = value
    path = rdir / RUN_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, path)


def find_run_dir(run_id: str) -> Path | None:
    """The run directory for ``run_id`` (searching every tool), or None if absent."""
    for rdir in _run_dirs():
        if rdir.name == run_id:
            return rdir
    return None


def read_tool_run(run_id: str) -> dict:
    """Read one tool-run manifest by id.

    Raises:
        FileNotFoundError: If no run with that id exists.
    """
    rdir = find_run_dir(run_id)
    if rdir is None:
        raise FileNotFoundError(f"no tool run {run_id!r} in {tools_root()}")
    return yaml_load.safe_load(atomic.read_text(rdir / RUN_FILENAME))


def delete_tool_run(run_id: str) -> None:
    """Remove a tool-run directory by id (idempotent)."""
    rdir = find_run_dir(run_id)
    if rdir is not None:
        shutil.rmtree(rdir)


def list_tool_runs(tool_id: str | None = None) -> list[dict]:
    """All tool-run manifests (one tool or every tool), ordered by run id."""
    runs = dirscan.read_each(
        _run_dirs(tool_id),
        lambda rdir: yaml_load.safe_load(atomic.read_text(rdir / RUN_FILENAME)),
    )
    return sorted(runs, key=lambda r: r["id"])
