"""Run a Python snippet on a remote target, and say why it failed (§6).

The one place the app runs Python on a cluster. Its callers (the shepherd's readiness probe, its
imported-model lookup and video-root registration, and the build comparison behind
``detail remote test``) all ask the cluster a question by importing DETAIL there, so they all fail
the same way when the cluster's DETAIL is not the one this app is running. Keeping the command in
one function is what lets that failure be named once: ``tests/executor/test_remote_version_skew.py``
scans the package and fails if a second place builds it.
"""

import re

from detail.core import interpreter
from detail.executor import ssh_transport
from detail.executor.schedulers.slurm import activation_prefix

#: What a cluster whose DETAIL is *older than this app* prints when a snippet imports something the
#: app has and the cluster's checkout does not (a module a newer release added, or a name it moved).
_VERSION_SKEW = re.compile(
    r"ModuleNotFoundError: No module named '(detail\.[\w.]+)'"
    r"|ImportError: cannot import name '(\w+)' from '(detail\.[\w.]+)'"
)


def remote_version_skew(output: str) -> str | None:
    """What the cluster's DETAIL lacks, if ``output`` is that failure; ``None`` otherwise.

    Matches only ``detail.<something>``: a bare ``No module named 'detail'`` means DETAIL is not
    installed in that environment at all, which is a different fix (the target's activation).
    """
    match = _VERSION_SKEW.search(output)
    if match is None:
        return None
    module, name, source = match.groups()
    return module or f"{source}.{name}"


def remote_python(
    target,
    script: str,
    failed: str,
    *,
    timeout_s: float | None = None,
    on_line=None,
):
    """Run ``script`` with the target's Python and return the result; a non-zero exit raises.

    ``failed`` names what was being attempted ("could not check model readiness on remote
    target ..."); the cause follows it, and the full output (an activated environment's banner and
    a traceback, which buried the cause in a run's table) goes on the lines after.

    A cluster whose DETAIL is older than this app is named as that, with the fix: its probes import
    every module the app lists, so a module added since the cluster was last updated fails every
    recording at once, and "No module named 'detail.pipeline.keypoint_dataset'" under a login
    banner does not say the cluster needs updating.

    ``on_line``: for a snippet that runs for minutes and prints as it goes (a model download
    reporting bytes), each line of its output reaches ``on_line`` as it arrives, and
    ``timeout_s`` does not apply — the snippet owns its own.

    Raises:
        RuntimeError: the snippet exited non-zero.
    """
    # `-P`: an ssh login starts in `$HOME`, where a checkout folder (`~/detail`) would otherwise
    # be imported as an empty namespace package (`core/interpreter.py`).
    command = activation_prefix(target) + interpreter.remote_python_command("-")
    if on_line is None:
        result = ssh_transport.run_remote(
            target.host,
            command,
            stdin_text=script,
            timeout_s=timeout_s,
            check=False,
        )
    else:
        result = ssh_transport.run_remote_lines(
            target.host, command, on_line, stdin_text=script
        )
    if result.returncode == 0:
        return result
    raise _failure(target, failed, result)


def _failure(target, failed: str, result) -> RuntimeError:
    detail = result.stderr.strip() or result.stdout.strip()
    missing = remote_version_skew(detail)
    if missing is not None:
        cause = (
            f"the DETAIL installed on {target.name!r} is older than this app and has no "
            f"{missing}. Update it ('git pull --ff-only' in its DETAIL folder, in the "
            "target's conda environment), then run again"
        )
    else:
        # The cause is the *last* line (an exception's own line).
        cause = detail.splitlines()[-1].strip() if detail else "no output"
    message = f"{failed}: {cause}"
    return RuntimeError(message if detail == cause else f"{message}\n{detail}")
