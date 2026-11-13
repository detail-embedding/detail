"""Experiment-independent remote (Slurm) target catalog (§6).

Remote targets are machine-global, not experiment state — configuring "my Tillicum account" is
a one-time, cross-experiment action, the same way installed models and credentials are. This is
the surface the CLI's ``detail remote`` group and the GUI's Configuration-mode remote-targets
page both call; neither talks to ``storage/remote_targets.py`` directly.
"""

import json
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import yaml

from detail.configs.remote import RemoteTargetConfig
from detail.core import buildinfo
from detail.core import yaml_load
from detail.executor import ssh_transport
from detail.executor.remote_python import remote_python
from detail.storage import credentials
from detail.storage import remote_targets as remote_targets_store

#: Default timeout for a connectivity check (§6's "shell out to the user's own ssh" — a hung
#: ControlMaster prompt must not hang the caller forever).
DEFAULT_TEST_TIMEOUT_S = 15


@dataclass
class RemoteTargetStatus:
    """Result of a connectivity check against a registered remote target."""

    ok: bool
    detail: str
    #: How the DETAIL on the target compares with this app's build, once the connection works.
    #: A mismatch does not make ``ok`` False: the cluster is reachable, it is just not running
    #: what this app is, and that is a warning with a fix, not a failed connection.
    build: buildinfo.BuildComparison | None = None


@dataclass(frozen=True)
class RemoteUpdate:
    """What updating the DETAIL on a remote target did (:func:`update_remote_build`).

    ``ok`` means the target is now the same build as this app, which is the point of updating it:
    a pull that succeeds and still leaves it on another commit (it tracks another branch, or this
    app is ahead of what its remote has) is ``ok=False``, with the reason in ``detail``.
    """

    ok: bool
    detail: str
    before: buildinfo.BuildComparison
    after: buildinfo.BuildComparison | None = None
    #: What git printed, trimmed to its last lines: the refusal's own words when it refused.
    output: str = ""
    #: ``pyproject.toml`` changed in what was pulled. The checkout is an editable install, so the
    #: new code is live, but a changed dependency is not installed until someone reinstalls.
    reinstall: bool = False


def list_remote_targets() -> list[RemoteTargetConfig]:
    """Every registered remote target, sorted by name."""
    return [
        cfg for _, cfg in sorted(remote_targets_store.list_remote_targets().items())
    ]


def get_remote_target(name: str) -> RemoteTargetConfig:
    """The registered remote target ``name``.

    Raises:
        KeyError: if no target is registered under that name.
    """
    return remote_targets_store.get_remote_target(name)


def analysis_compute(compute, what: str) -> dict | None:
    """A remote-capable analysis job's ``compute`` (§6 "Remote analysis") as a
    ``ComputeTarget`` dict: this machine, or a registered remote target. ``what`` names the
    job in the refusal.

    Raises:
        ValueError: A cloud target, a remote one with no name, or one not registered here.
    """
    from detail.core.dataclasses import ComputeTarget

    if compute is None:
        return None
    target = (
        compute
        if isinstance(compute, ComputeTarget)
        else ComputeTarget.from_dict(compute)
    )
    if target.backend == "cloud":
        raise ValueError(f"{what} on a cloud target is not supported")
    if target.backend == "remote":
        if not target.remote_target:
            raise ValueError("a remote run names its target (detail remote list)")
        try:
            get_remote_target(target.remote_target)
        except KeyError:
            raise ValueError(
                f"no remote target {target.remote_target!r} is registered on this machine "
                "(detail remote list)"
            ) from None
    return target.to_dict()


def add_remote_target(config: RemoteTargetConfig) -> RemoteTargetConfig:
    """Register (or re-point) a remote target. Returns the stored config."""
    remote_targets_store.register_remote_target(config)
    return config


def remove_remote_target(name: str) -> None:
    """Remove the registered remote target ``name``.

    Raises:
        KeyError: if no target is registered under that name.
    """
    remote_targets_store.remove_remote_target(name)


def test_remote_target(
    name: str, timeout_s: int = DEFAULT_TEST_TIMEOUT_S
) -> RemoteTargetStatus:
    """Check SSH reachability of target ``name`` and that `sbatch` is on its `PATH`.

    Shells out to the user's own `ssh` (§6) with `BatchMode=yes`, so a host that needs an
    interactive prompt (a password, an un-cached Duo push) fails fast rather than hanging —
    the check reports that as a failure with an actionable message rather than blocking.

    Raises:
        KeyError: if no target is registered under that name.
    """
    config = get_remote_target(name)
    try:
        with ssh_transport.session_slot(config.host):
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={timeout_s}",
                    config.host,
                    "command -v sbatch >/dev/null 2>&1 && echo DETAIL_SBATCH_OK "
                    "|| echo DETAIL_SBATCH_MISSING",
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s + 5,
            )
    except FileNotFoundError:
        return RemoteTargetStatus(
            ok=False, detail="no local `ssh` executable found on PATH"
        )
    except subprocess.TimeoutExpired:
        return RemoteTargetStatus(
            ok=False,
            detail=f"connection to {config.host!r} timed out after {timeout_s}s — check "
            "~/.ssh/config and that a Duo push isn't waiting",
        )
    if result.returncode != 0:
        if ssh_transport.login_failed(result.stderr):
            return RemoteTargetStatus(
                ok=False, detail=ssh_transport.expired_session_advice(config.host)
            )
        reason = result.stderr.strip() or f"ssh exited {result.returncode}"
        return RemoteTargetStatus(
            ok=False, detail=f"could not reach {config.host!r}: {reason}"
        )
    if "DETAIL_SBATCH_OK" not in result.stdout:
        return RemoteTargetStatus(
            ok=False,
            detail=f"connected to {config.host!r}, but `sbatch` is not on its PATH — check "
            "the scheduler is Slurm and its module/PATH setup",
        )
    return RemoteTargetStatus(
        ok=True,
        detail=f"connected to {config.host!r}; sbatch found",
        build=check_remote_build(name, timeout_s),
    )


def check_remote_build(
    name: str, timeout_s: int = DEFAULT_TEST_TIMEOUT_S
) -> buildinfo.BuildComparison:
    """Compare the DETAIL on target ``name`` with the build this app is running.

    A job imports this app's modules on the cluster, so a cluster on another commit runs code the
    app did not write, and one that is **behind** fails every recording at once the day the app
    uses something new (``No module named 'detail.pipeline.keypoint_dataset'``). The cluster is
    asked for its own commit (its ``buildinfo.report``, run in the target's environment), and the
    order is worked out with this checkout's git: nothing is fetched and nothing there changes.

    Never raises for a problem with the cluster or the answer: that comes back as
    ``state == "unknown"`` with the reason, because "could not tell" is a result of this check.

    Raises:
        KeyError: if no target is registered under that name.
    """
    config = get_remote_target(name)

    def unknown(reason: str) -> buildinfo.BuildComparison:
        return buildinfo.BuildComparison(
            buildinfo.UNKNOWN, target=config.name, host=config.host, reason=reason
        )

    script = (
        "import json\n"
        "from detail.core import buildinfo\n"
        "print(json.dumps(buildinfo.report()))\n"
    )
    try:
        result = remote_python(
            config,
            script,
            f"could not read the DETAIL build on {config.name!r}",
            timeout_s=timeout_s,
        )
    except RuntimeError as exc:
        return unknown(str(exc).splitlines()[0])
    except subprocess.TimeoutExpired:
        return unknown(f"it did not answer within {timeout_s}s")
    try:
        # The last line: an activated environment's banner comes before it.
        remote = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return unknown(f"its answer was not a build report: {result.stdout.strip()!r}")
    if not isinstance(remote, dict):
        return unknown(f"its answer was not a build report: {remote!r}")
    return buildinfo.compare_with(remote, target=config.name, host=config.host)


#: How long the update may take end to end: the ssh session, the target's environment
#: activating, and a ``git pull`` over the cluster's own network.
UPDATE_TIMEOUT_S = 180

#: Run on the target, in its own environment (so it does not depend on which shell the login
#: gives: tcsh spells ``$?`` and ``$(...)`` differently), after ``root = '<its checkout>'``.
#: ``git pull --ff-only`` only ever moves the branch forward: it refuses, and changes nothing, when
#: the checkout has commits of its own or would overwrite a local change.
_PULL_SCRIPT = """\
import json, os, subprocess

env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # a private remote must fail, not prompt


def git(*args, timeout=120):
    return subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, env=env, timeout=timeout
    )


old = git("rev-parse", "HEAD").stdout.strip()
pull = git("pull", "--ff-only", timeout=150)
new = git("rev-parse", "HEAD").stdout.strip()
changed = git("diff", "--name-only", old, new).stdout.split() if old and new and old != new else []
print(json.dumps({
    "old": old,
    "new": new,
    "returncode": pull.returncode,
    "output": (pull.stdout + pull.stderr).strip()[-800:],
    "pyproject": "pyproject.toml" in changed,
}))
"""


def update_remote_build(
    name: str,
    *,
    before: buildinfo.BuildComparison | None = None,
    timeout_s: int = UPDATE_TIMEOUT_S,
    progress: Callable[[str, float, str], None] | None = None,
) -> RemoteUpdate:
    """Bring the DETAIL on target ``name`` up to date with this app, when it is behind.

    The sequence is: compare (unless ``before`` is the comparison the caller already showed the
    user), refuse unless the target is **behind** (the same build needs nothing; ahead, diverged or
    unknown are not a fast-forward from here), ``git pull --ff-only`` in the target's own checkout,
    then compare again and say whether it now matches. Only the target's checkout moves; nothing on
    this machine does, and nothing is installed there.

    A run that is in flight on the target keeps importing modules as it goes, so updating while one
    runs can leave it on a mix of two commits; the caller says so before asking.

    Never raises for a problem with the target or with git: that comes back as ``ok=False`` and the
    reason, because a refused update is a result. ``progress(stage, fraction, detail)`` names each
    step as it starts.

    Raises:
        KeyError: if no target is registered under that name.
    """
    config = get_remote_target(name)

    def report(stage: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    if before is None:
        report("Checking", 0.0, f"DETAIL on {name!r}")
        before = check_remote_build(name)
    if before.state == buildinfo.SAME:
        return RemoteUpdate(
            True,
            f"Nothing to update: DETAIL on {name!r} is already the same build as this "
            f"app ({before.local_commit}).",
            before,
        )
    if before.state != buildinfo.BEHIND:
        return RemoteUpdate(False, f"Not updated. {before.describe()}", before)
    if not before.remote_root:
        return RemoteUpdate(
            False,
            f"Not updated: {name!r} did not say where its DETAIL checkout is.",
            before,
        )

    report("Updating", 0.3, f"git pull --ff-only in {before.remote_root}")
    script = f"root = {json.dumps(before.remote_root)}\n{_PULL_SCRIPT}"
    try:
        result = remote_python(
            config,
            script,
            f"could not update DETAIL on {name!r}",
            timeout_s=timeout_s,
        )
        pulled = json.loads(result.stdout.strip().splitlines()[-1])
    except RuntimeError as exc:
        return RemoteUpdate(False, str(exc).splitlines()[0], before)
    except subprocess.TimeoutExpired:
        return RemoteUpdate(
            False,
            f"The update on {name!r} did not finish within {timeout_s}s; check it with "
            "Test connection.",
            before,
        )
    except (json.JSONDecodeError, IndexError):
        return RemoteUpdate(
            False,
            f"The update on {name!r} gave an answer this app could not read: "
            f"{result.stdout.strip()[-200:]!r}",
            before,
        )
    output = str(pulled.get("output") or "")
    if pulled.get("returncode") != 0:
        reason = output.splitlines()[-1].strip() if output else "no output"
        return RemoteUpdate(
            False,
            f"git refused the update on {name!r}, which is unchanged: {reason}",
            before,
            output=output,
        )

    report("Verifying", 0.8, "")
    after = check_remote_build(name)
    old, new = str(pulled.get("old", ""))[:8], str(pulled.get("new", ""))[:8]
    moved = old != new
    reinstall = bool(pulled.get("pyproject"))
    note = (
        " pyproject.toml changed in this update: its dependencies are not reinstalled "
        "for you, so activate the target's environment there and run `pip install -e .`."
        if reinstall
        else ""
    )
    if after.matches:
        detail = (
            f"Updated DETAIL on {name!r}: {old} -> {new}. It is now the same build as "
            f"this app.{note}"
        )
    elif moved:
        detail = (
            f"Updated DETAIL on {name!r} ({old} -> {new}), but it is still not the same "
            f"build as this app. {after.describe()}{note}"
        )
    else:
        detail = (
            f"Nothing new to pull: the remote {name!r} pulls from has no commits past "
            f"{old}. {after.describe()}"
        )
        if after.state == buildinfo.BEHIND:
            detail += (
                " This app's newer commits are probably not pushed to the remote the "
                "cluster pulls from."
            )
    return RemoteUpdate(
        after.matches, detail, before, after=after, output=output, reinstall=reinstall
    )


def _remote_home(host: str, *, timeout_s: float | None = None) -> str:
    """The remote user's home directory, via the login shell's own ``$HOME`` —
    never a literal ``~`` handed to :func:`ssh_transport.write_remote_file` /
    :func:`ssh_transport.read_remote_file`, which ``shlex.quote`` their path
    argument: a quoted ``~`` is not expanded, so it would write a file named
    literally ``~`` in the ssh login shell's working directory instead of the
    home directory."""
    result = ssh_transport.run_remote(host, "echo $HOME", timeout_s=timeout_s)
    home = result.stdout.strip()
    if not home:
        raise RuntimeError(f"could not resolve $HOME on {host!r}")
    return home


@dataclass(frozen=True)
class CredentialSpec:
    """One credential this machine can push to a remote target: the key it lives
    under in ``credentials.yaml`` (local and remote — same file, same shape,
    everywhere), a display label for status/error text, and how to resolve its
    current value on *this* machine (``None`` when it isn't configured here)."""

    key: str
    label: str
    get_value: Callable[[], str | None]


#: Every credential ``push_credentials`` knows how to forward. One entry today
#: (Hugging Face, for a gated model's cluster-side access check and download);
#: a cloud provider (AWS, Azure, ...) joins this list the day this machine
#: actually manages one of *its* credentials locally — adding an entry here is
#: the only change ``push_credentials``, the CLI, and the GUI button need, since
#: all three already push "every configured credential" rather than a named one.
CREDENTIAL_SPECS: tuple[CredentialSpec, ...] = (
    # A lambda that looks the function up each call, not credentials.get_hf_token
    # bound here directly — the latter freezes the reference at import time, so
    # re-pointing credentials.get_hf_token afterwards (a test's monkeypatch, a
    # future runtime reconfiguration) would silently keep calling the original.
    CredentialSpec(
        "huggingface", "Hugging Face access", lambda: credentials.get_hf_token()
    ),
)


def push_credentials(
    name: str, timeout_s: int = DEFAULT_TEST_TIMEOUT_S
) -> RemoteTargetStatus:
    """Copy every credential this machine has configured (:data:`CREDENTIAL_SPECS`)
    to target ``name``'s own ``~/.config/detail/credentials.yaml`` — the same file,
    same keys, same 0600 permissions ``storage/credentials.py`` writes locally, so
    the remote install of this package resolves each one through its own identical
    local-resolution order with no code of its own. Merges into whatever is already
    there rather than overwriting it.

    A gated model (SAM 3, DINOv3) needs Hugging Face access on the cluster, not on
    this machine — the shepherd's pre-submission readiness probe and the worker
    both run *there*, and neither ever sees this machine's own token. Detection
    worked without this because its weights were already cached on the cluster
    from an earlier manual setup; a model that has never been fetched there needs
    a token to check access and download, which nothing before this pushed.

    Raises:
        KeyError: if no target is registered under that name.
        ValueError: if this machine has none of :data:`CREDENTIAL_SPECS` configured
            yet (Configuration → Hugging Face access sets the one that exists today).
    """
    target = get_remote_target(name)
    to_push = {
        spec.key: (spec.label, value)
        for spec in CREDENTIAL_SPECS
        if (value := spec.get_value())
    }
    if not to_push:
        raise ValueError(
            "no credentials are configured on this machine yet to push — set one "
            "first (Configuration → Hugging Face access), then push it here"
        )
    try:
        home = _remote_home(target.host, timeout_s=timeout_s)
        remote_path = f"{home}/.config/detail/credentials.yaml"
        try:
            existing = yaml_load.safe_load(
                ssh_transport.read_remote_file(
                    target.host, remote_path, timeout_s=timeout_s
                )
            )
        except ssh_transport.RemoteCommandError:
            existing = None  # nothing there yet — a fresh cluster account
        if not isinstance(existing, dict):
            existing = {}
        for key, (_label, value) in to_push.items():
            existing[key] = value
        ssh_transport.write_remote_file(
            target.host,
            remote_path,
            yaml.safe_dump(existing, sort_keys=False),
            timeout_s=timeout_s,
        )
        ssh_transport.run_remote(
            target.host, f"chmod 600 {shlex.quote(remote_path)}", timeout_s=timeout_s
        )
    except ssh_transport.RemoteCommandError as exc:
        return RemoteTargetStatus(
            ok=False,
            detail=f"could not push credentials to {target.host!r}: "
            f"{exc.stderr.strip() or exc.stdout.strip()}",
        )
    except subprocess.TimeoutExpired:
        return RemoteTargetStatus(
            ok=False,
            detail=f"connection to {target.host!r} timed out after {timeout_s}s — "
            "check ~/.ssh/config and that a Duo push isn't waiting",
        )
    pushed = ", ".join(label for label, _value in to_push.values())
    return RemoteTargetStatus(ok=True, detail=f"pushed to {target.host!r}: {pushed}")
