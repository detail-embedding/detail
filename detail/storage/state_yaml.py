"""The authored-state file of a mutable run (§21) — atomic rewrite, ``schema_version`` first.

An authored run's state is the one thing in its directory that is *not* recomputable: it is
what a person decided. §21 gives it two rules, and both are here rather than in each module
because getting either subtly wrong is a data-loss bug rather than a tidiness one.

**Rewritten atomically, never in place.** A runner subprocess rewrites this while the GUI is
reading it, so a plain truncating write gives a reader an empty or half-written file — the
concurrently-read-file rule (CLAUDE.md). tmp + ``os.replace``, always.

**``schema_version`` is the first key** (Tenet 7), written before any content.

This is the second of §21's two lift-worthy utilities. ``AppendLog`` (``storage/append_log.py``)
came first, extracted when §19's ``audit.csv`` became its second consumer. §21 named this one as
worth lifting too but deliberately left it in ``annotation_runs.py`` while it had a single
implementation — Query is the second, which is the condition the section set, so it moves here
and ``annotation_runs`` is ported onto it.

A state file holds **no status and no counters**. Status lives in ``analysis.yaml`` and nowhere
else; anything countable is derived from the decision log. That is §21's one-writer invariant,
and it is enforced by a scan rather than by a behavioural test, because dual writers on one fact
stay well-formed while they disagree.
"""

from pathlib import Path

import yaml

from detail.core import atomic

SCHEMA_VERSION = 1


def write_state_yaml(path: Path | str, record: dict) -> Path:
    """Write an authored run's state to ``path`` atomically, ``schema_version`` first.

    Args:
        path: The state file itself (e.g. ``<run_dir>/query.yaml``), not the run directory.
        record: The authored state. Any ``schema_version`` it carries is replaced.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in record.items():
        if key != "schema_version":
            ordered[key] = value
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, path)
    return path


def read_state_yaml(path: Path | str) -> dict:
    """Read an authored run's state.

    Raises:
        FileNotFoundError: If the state file is absent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no {path.name} in {path.parent}")
    return yaml.safe_load(atomic.read_text(path)) or {}
