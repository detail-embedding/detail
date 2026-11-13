"""A run's meta-information: the YAML files its directory keeps, read as they are on disk.

An analysis run's settings, record and provenance are spread over several files in
``analysis/<an_NN>/``: ``analysis.yaml`` (the record, with every setting as resolved at launch,
Tenet 9), the module's own ``<module>.yaml`` (method parameters, and what the user has changed
since: labels, accepted names), ``machine.yaml`` and ``resources.yaml`` (where it ran and what it
used). The only way to see them was to open the folder. This reads every one of them into one
:class:`RunInfo` that the GUI shows in a window and the CLI prints, and :func:`export_run_info`
writes them to one zip, so what a run was set to can be sent to someone in one file.

The text is the file's own, unparsed: a viewer that re-dumped parsed YAML would reorder keys and
drop comments, and the point is to see what is on disk. Only the run directory's top level is
read; the folders under it (profiles, clips, caches) are results, not meta-information.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from detail.core import atomic

#: The suffixes read as meta-information.
YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class InfoFile:
    """One meta-information file: its name in the run directory, and its text."""

    name: str
    text: str


@dataclass(frozen=True)
class RunInfo:
    """Every meta-information file of one run, in reading order."""

    run_id: str
    files: tuple[InfoFile, ...]


def read_run_info(
    run_dir: Path | str, run_id: str, *, first: tuple[str, ...] = ()
) -> RunInfo:
    """Read every YAML file at the top of ``run_dir``.

    Args:
        run_dir: The run's directory.
        run_id: The run's id, which names the export.
        first: File names to put first, in this order, when present (the run's record and
            the module's own file); the rest follow alphabetically.

    Returns:
        The files and their text. A run directory with no YAML gives no files.

    Raises:
        FileNotFoundError: ``run_dir`` does not exist.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run directory at {run_dir}")
    paths = {
        p.name: p
        for p in run_dir.iterdir()
        if p.is_file() and p.suffix.lower() in YAML_SUFFIXES
    }
    order = [n for n in first if n in paths]
    order += sorted(n for n in paths if n not in order)
    return RunInfo(
        run_id=run_id,
        files=tuple(
            InfoFile(name, paths[name].read_text(encoding="utf-8-sig"))
            for name in order
        ),
    )


def export_run_info(info: RunInfo, path: Path | str) -> Path:
    """Write ``info``'s files to one zip at ``path``, each under a folder named for the run.

    Written beside ``path`` and then moved into place, so an export that fails part-way leaves
    no half-written zip under the name asked for.

    Returns:
        ``path``.

    Raises:
        ValueError: ``info`` holds no files.
    """
    if not info.files:
        raise ValueError(f"{info.run_id} has no meta-information files to export")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for f in info.files:
            archive.writestr(f"{info.run_id}/{f.name}", f.text)
    atomic.replace(tmp, path)
    return path
