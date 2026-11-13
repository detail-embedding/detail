"""Remote (Slurm) cluster profiles — machine-global storage (§6).

A registered remote target names an SSH-reachable Slurm cluster (Tillicum, Hyak, a lab's own
box) a run's ``ComputeTarget`` can point at. Stored the same way as the other machine-global
config this package keeps under ``~/.config/detail/`` — ``credentials.yaml`` (secrets) and
``storage.yaml`` (paths/aliases) are its siblings; this is a third, separate file rather than a
new key in either, since neither is the right shape (not a secret, not a path alias).

No credential of any kind lives here or anywhere DETAIL writes: a target's ``host`` is expected to
match a ``Host`` entry in the user's own ``~/.ssh/config``, so authentication (keys, an agent,
ControlMaster, a cached Duo session) is entirely the user's existing SSH setup.
"""

import os
from contextlib import contextmanager
from pathlib import Path

from detail.configs.remote import RemoteTargetConfig
from detail.storage import machine_file

# Same machine config dir as credentials.yaml/storage.yaml, and honours the same
# ``DETAIL_CONFIG_DIR`` override so a worker — or a test — never writes into the real one.
CONFIG_DIR = Path(
    os.environ.get("DETAIL_CONFIG_DIR") or Path.home() / ".config" / "detail"
)
REMOTE_TARGETS_PATH = CONFIG_DIR / "remote_targets.yaml"


def list_remote_targets() -> dict[str, RemoteTargetConfig]:
    """All registered remote targets, keyed by name."""
    data = _read()
    return {
        name: RemoteTargetConfig.model_validate({"name": name, **fields})
        for name, fields in data.get("targets", {}).items()
    }


def get_remote_target(name: str) -> RemoteTargetConfig:
    """The registered remote target ``name``.

    Raises:
        KeyError: if no target is registered under that name.
    """
    targets = list_remote_targets()
    if name not in targets:
        raise KeyError(f"no remote target registered as {name!r}")
    return targets[name]


def require_registered(compute) -> None:
    """A remote ``compute`` names a target that is registered here (§6); any other backend
    passes.

    The one copy of the check every remote-capable stage's ``validate`` makes, so a typo'd or
    never-added target fails at launch rather than in a detached coordinator's first submit.

    Raises:
        ValueError: ``compute.backend`` is ``"remote"`` with no ``remote_target``, or one
            that names a target nothing has registered.
    """
    if compute is None or compute.backend != "remote":
        return
    if not compute.remote_target:
        raise ValueError(
            "compute.backend is 'remote' but no remote_target was given; "
            "pick a registered target (see `detail remote list`)"
        )
    try:
        get_remote_target(compute.remote_target)
    except KeyError as e:
        raise ValueError(str(e)) from e


def register_remote_target(config: RemoteTargetConfig) -> None:
    """Register (or replace) a remote target under ``config.name``."""
    with _editing(REMOTE_TARGETS_PATH) as data:
        data["targets"][config.name] = config.model_dump(exclude={"name"})


def remove_remote_target(name: str) -> None:
    """Remove the registered remote target ``name``.

    Raises:
        KeyError: if no target is registered under that name.
    """
    path = REMOTE_TARGETS_PATH  # once: the check and the edit are about one file
    if name not in _with_schema(machine_file.read(path))["targets"]:
        raise KeyError(f"no remote target registered as {name!r}")
    with _editing(path) as data:
        data["targets"].pop(name, None)


def _read() -> dict:
    return _with_schema(machine_file.read(REMOTE_TARGETS_PATH))


@contextmanager
def _editing(path: Path):
    """``path`` as a dict, ``schema_version`` first and ``targets`` present, written back to the
    file it was read from (``machine_file``)."""
    with machine_file.editing(path) as data:
        normal = _with_schema(dict(data))
        data.clear()
        data.update(normal)
        yield data


def _with_schema(data: dict) -> dict:
    return {
        "schema_version": 1,
        "targets": {},
        **{k: v for k, v in data.items() if k != "schema_version"},
    }
