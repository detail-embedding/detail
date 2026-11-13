"""Machine-global Tools API (§9) — the experiment-free entry point for one-off runs.

A *tool* runs synchronously and saves a managed run under the machine-global tools
workspace (``storage/tool_runs.py``). This module is the surface the CLI and GUI both
drive: it resolves the tool, allocates a run directory, runs it (recording
running → complete/failed on the manifest), and lists / reads / deletes past runs.
Experiment-free, so it is a module of functions (like ``api/models.py``), not a class.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from detail.core.registries import TOOLS
from detail.core.utils import cuda_available, utc_timestamp
from detail.storage import tool_runs
from detail.tools import import_tools

_SAM3_MODEL_ID = "sam3"


@dataclass
class ToolInfo:
    """A registered tool, for listing in the CLI / GUI (no config/run coupling)."""

    tool_id: str
    label: str
    description: str
    category: str


def list_tools() -> list[ToolInfo]:
    """Every registered tool, ordered by id."""
    import_tools()
    return [
        ToolInfo(t.tool_id, t.label, t.description, t.category)
        for t in (TOOLS[k] for k in sorted(TOOLS))
    ]


def default_model_for_machine() -> str:
    """The detection model Quick segment auto-picks for this machine.

    SAM3 when an NVIDIA GPU is present. The no-GPU path (a shipped distilled model)
    is planned but not built (OQ-CFG-3), so on a machine without CUDA this raises with
    an actionable message rather than silently falling back to an unusable device."""
    if cuda_available():
        return _SAM3_MODEL_ID
    raise RuntimeError(
        "Quick segment needs an NVIDIA GPU that PyTorch can use (CUDA) to run SAM3 on "
        "this machine. A no-GPU shipped model is planned but not yet available."
    )


def run_tool(
    tool_id: str,
    config: dict,
    *,
    progress: Callable[[str, float, str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> dict:
    """Run a tool and record its managed run; return the final manifest.

    The manifest is written ``running`` before the work starts (so a crash leaves a
    visible record), then ``complete`` with the result — ``stopped`` when the user ended
    it early (the partial output is still there and still openable), or ``failed`` with
    the error, re-raised for the caller to surface.

    Raises:
        ValueError: Unknown ``tool_id``.
    """
    import_tools()
    tool_cls = TOOLS.get(tool_id)
    if tool_cls is None:
        raise ValueError(f"unknown tool {tool_id!r}; available: {sorted(TOOLS)}")
    tool = tool_cls()

    run_id = tool_runs.next_run_id()
    out_dir = tool_runs.tool_run_dir(tool_id, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Plural inputs with a singular alias: a run may span several clips and compare
    # several prompts, but ``source_video`` / ``prompt`` stay populated with the first so
    # manifests written before this — and anything reading them — keep working.
    videos = [Path(v) for v in (config.get("videos") or [config["video"]])]
    prompts = list(config.get("prompts") or [config["prompt"]])
    record = {
        "id": run_id,
        "tool_id": tool_id,
        "label": tool.label,
        "source_video": str(videos[0].resolve()),
        "source_videos": [str(v.resolve()) for v in videos],
        "model_id": config.get("model_id"),
        "model_variant": config.get("model_variant"),
        "prompt": prompts[0].to_dict(),
        "prompts": [p.to_dict() for p in prompts],
        "compute": config["compute"].to_dict(),
        # The tool's config model is the single source for its scalar parameters, so the
        # manifest records exactly those rather than a hand-listed subset that drifts.
        "config": {
            name: config[name]
            for name in tool.config_model.model_fields
            if name in config
        },
        "status": "running",
        "created_at": utc_timestamp(),
    }
    tool_runs.write_tool_run(tool_id, run_id, record)

    try:
        result = tool.run(config, out_dir, progress, stop_check)
    except Exception as exc:  # noqa: BLE001 — record the failure, then re-raise
        record["status"] = "failed"
        record["error"] = str(exc)
        tool_runs.write_tool_run(tool_id, run_id, record)
        raise
    record["status"] = "stopped" if result.stopped else "complete"
    record["result"] = result.to_dict()
    tool_runs.write_tool_run(tool_id, run_id, record)
    return record


def list_tool_runs(tool_id: str | None = None) -> list[dict]:
    """All tool-run manifests (one tool or every tool), ordered by run id."""
    return tool_runs.list_tool_runs(tool_id)


def read_tool_run(run_id: str) -> dict:
    """One tool-run manifest by id (raises ``FileNotFoundError`` if absent)."""
    return tool_runs.read_tool_run(run_id)


def delete_tool_run(run_id: str) -> None:
    """Delete a tool run by id (idempotent)."""
    tool_runs.delete_tool_run(run_id)
