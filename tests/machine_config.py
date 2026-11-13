"""Reading the developer's real machine config across a test run, for the session backstop.

``tests/conftest.py`` snapshots the real ``~/.config/detail`` before the first fixture and
compares after the last worker; these are the two pure halves of that, kept here so
``tests/test_machine_config_isolation.py`` can hand them literals. The class and the rule are in
that file's docstring.
"""

from __future__ import annotations

from pathlib import Path


def snapshot(config_dir: Path) -> dict[str, bytes]:
    """The bytes of every YAML file directly in ``config_dir``."""
    files = {}
    for path in config_dir.glob("*.yaml"):
        try:
            files[path.name] = path.read_bytes()
        except (
            OSError
        ):  # replaced mid-read by the developer's own app: not ours to judge
            continue
    return files


def litter(
    before: dict[str, bytes], after: dict[str, bytes], basetemp: Path
) -> list[str]:
    """Names of config files that changed during the run to name a path under ``basetemp``.

    A path under this run's basetemp is something only a test writes, so matching on it rather
    than on "changed at all" lets the developer's own app edit the file mid-run. The trailing
    separator keeps ``pytest-22`` from matching an older run's ``pytest-2249``.
    """
    marker = basetemp.as_posix().rstrip("/").lower() + "/"
    return sorted(
        name
        for name, data in after.items()
        if data != before.get(name)
        and marker in data.decode("utf-8", "replace").replace("\\", "/").lower()
    )
