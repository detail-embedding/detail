"""Machine-global check runner (§1) — the live checker framework's entry point.

Qt-free and experiment-free (like ``api/models.py`` / ``api/tools.py``). ``run_checks``
executes every registered check: the import-sensitive ones (``isolation="subprocess"``) in
a **single throwaway subprocess** so a mid-session install is reflected without restarting
the app, the rest in-process. ``in_process=True`` runs everything inline (tests, cheap
callers) with no spawn. A probe that can't answer degrades to ``UNKNOWN``; nothing raises.
"""

import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass

from detail.checks import import_checks
from detail.checks.commands import (
    SetupScript,
    SetupStep,
    current_script,
    detail_repo_root,
    setup_scripts,
    ssh_refresh_argv,
    ssh_terminal_supported,
    terminal_launch_argv,
    terminal_name,
    wsl_browser_argv,
    wsl_explorer_argv,
)
from detail.core.dataclasses import CheckResult, CheckStatus
from detail.core.registries import CHECKS

# Re-exported so the CLI (`detail doctor`) and the GUI (`Sam3BackendPage`,
# `RemoteTargetsPage`) reach the machine-tailored install commands through the
# one check API they already import.
__all__ = [
    "CheckInfo",
    "SetupScript",
    "SetupStep",
    "current_script",
    "detail_repo_root",
    "list_checks",
    "run_check",
    "run_checks",
    "setup_scripts",
    "ssh_refresh_argv",
    "ssh_terminal_supported",
    "terminal_launch_argv",
    "terminal_name",
    "wsl_browser_argv",
    "wsl_explorer_argv",
]

_PROBE_TIMEOUT_S = 60


@dataclass
class CheckInfo:
    """A registered check, for listing (no run coupling)."""

    check_id: str
    label: str
    group: str
    isolation: str
    fix_page: str | None
    fix_anchor: str | None


def list_checks() -> list[CheckInfo]:
    """Every machine-global check, ordered by id (per-recording checks are experiment-scoped
    and surfaced separately via ``run_recording_checks``)."""
    import_checks()
    infos = []
    for check_id in sorted(CHECKS):
        cls = CHECKS[check_id]
        if cls.target_kind != "machine":
            continue
        fix = getattr(cls, "fix", None)
        infos.append(
            CheckInfo(
                cls.check_id,
                cls.label,
                cls.group,
                cls.isolation,
                fix.page if fix is not None else None,
                fix.anchor if fix is not None else None,
            )
        )
    return infos


def run_checks(*, in_process: bool = False) -> list[CheckResult]:
    """Run every registered check and return the results, ordered by id.

    ``in_process=False`` (default) runs the ``subprocess`` checks in one fresh interpreter;
    ``in_process=True`` runs them inline (no spawn — for tests and cheap callers)."""
    import_checks()
    machine = [k for k in CHECKS if CHECKS[k].target_kind == "machine"]
    results: dict[str, CheckResult] = {}
    subprocess_ids = [k for k in machine if CHECKS[k].isolation == "subprocess"]
    if in_process:
        for check_id in subprocess_ids:
            results[check_id] = _run_inline(check_id)
    elif subprocess_ids:
        results.update(_run_in_subprocess(subprocess_ids))
    for check_id in machine:
        if CHECKS[check_id].isolation != "subprocess":
            results[check_id] = _run_inline(check_id)
    return [results[k] for k in sorted(results)]


def run_recording_checks(experiment_dir) -> list[CheckResult]:
    """Run every per-recording check against each recording in the experiment (in-process).

    Each result carries a ``target_id`` (the recording id) so the experiment view can show
    per-recording status. Reuses the same integrity logic the pre-run gate enforces.
    """
    import_checks()
    from detail.core.dataclasses import RecordingTarget
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    recording_ids = [k for k in sorted(CHECKS) if CHECKS[k].target_kind == "recording"]
    results: list[CheckResult] = []
    if not recording_ids:
        return results
    for sid in experiment_store.discover_recordings(experiment_dir):
        sdir = experiment_store.recording_path(experiment_dir, sid)
        try:
            sdata = recording_store.read_recording(sdir)
        except (FileNotFoundError, ValueError):
            continue  # a half-written recording is skipped, not fatal
        target = RecordingTarget(str(experiment_dir), sid, str(sdir), sdata)
        for check_id in recording_ids:
            results.append(_run_inline_target(check_id, target))
    return results


def run_check(check_id: str) -> CheckResult:
    """Run one check by id, in-process (raises KeyError for an unknown id)."""
    import_checks()
    if check_id not in CHECKS:
        raise KeyError(f"unknown check {check_id!r}")
    return _run_inline(check_id)


def _run_inline(check_id: str) -> CheckResult:
    try:
        return CHECKS[check_id]().run()
    except Exception as exc:  # isolation boundary — a probe never crashes the caller
        return CheckResult(
            check_id, CheckStatus.UNKNOWN, f"{type(exc).__name__}: {exc}"
        )


def _run_inline_target(check_id: str, target) -> CheckResult:
    try:
        result = CHECKS[check_id]().run(target)
    except Exception as exc:  # isolation boundary
        return CheckResult(
            check_id,
            CheckStatus.UNKNOWN,
            f"{type(exc).__name__}: {exc}",
            target.recording_id,
        )
    if result.target_id is None:  # a check that forgot to tag its target
        return dataclasses.replace(result, target_id=target.recording_id)
    return result


def _run_in_subprocess(check_ids: list[str]) -> dict[str, CheckResult]:
    """Run the import-sensitive checks in a fresh interpreter; parse the JSON they print."""
    from detail.executor.local import new_process_group_kwargs

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "detail.checks._probe"],
            capture_output=True,
            text=True,
            # The probe's own JSON is ASCII (`json.dumps` escapes), but backend import
            # chatter on the way past it is arbitrary bytes, and this decodes with the
            # locale codec — cp1252 has undefined slots, so a strict decode can raise
            # and `except Exception` below would report every check UNKNOWN.
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
            **new_process_group_kwargs(),
        )
        data = json.loads(_last_json_line(proc.stdout))
    except Exception:  # timeout, spawn failure, or unparseable output → all UNKNOWN
        return {
            cid: CheckResult(cid, CheckStatus.UNKNOWN, "readiness probe unavailable")
            for cid in check_ids
        }
    out = {}
    for cid in check_ids:
        raw = data.get(cid)
        out[cid] = (
            CheckResult.from_dict(raw)
            if raw
            else CheckResult(cid, CheckStatus.UNKNOWN, "no result from probe")
        )
    return out


def _last_json_line(stdout: str) -> str:
    """The last ``{...}`` line of the probe's stdout (robust to import chatter)."""
    for line in reversed(stdout.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            return stripped
    raise ValueError("no JSON line in probe output")
