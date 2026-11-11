"""A user's free-text notes on one experiment entry: ``<entry dir>/notes.md`` (§15).

Every entry the experiment tree shows — the experiment itself, a recording, a processing run, an
analysis run, a dataset, a model, a distill run, an export run, a comparison —
already owns a directory. The note lives in that directory, in a file of its own,
rather than as a field in the entry's record:

- **The record has another writer.** A processing run's coordinator rewrites ``run.yaml`` for the
  whole length of the run, and a note typed while it does would be lost to whichever write landed
  second. A sidecar has exactly one writer, so a note can be written while the run is live.
- **It goes where the entry goes.** Deleting an entry removes its directory, and an experiment
  export packs the tree, so neither needs to know notes exist.
- **Plain text, not YAML.** A note is prose the user typed, and DETAIL never parses it. It may
  quote a path (``C:\\Users\\…``), which in a YAML file is a Tenet 6 violation the experiment export
  refuses to pack; in ``notes.md`` it is what the user wrote. It also opens in any editor.

No note is no file: saving an empty note removes it, so the tree does not fill with empty files
from pages someone opened and never typed on.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from detail.core import atomic
from detail.storage import analysis_runs
from detail.storage import comparisons
from detail.storage import datasets
from detail.storage import distill_runs
from detail.storage import experiment as experiment_store
from detail.storage import export_runs
from detail.storage import models
from detail.storage import runs

NOTES_FILENAME = "notes.md"

#: Each entry kind and where its directory is. The kind names are the GUI's navigation kinds and
#: the export subject kinds where those exist (``run``, ``analysis_run``, ``distill``), so one
#: name means one thing on every surface. ``experiment`` takes no id.
ENTRY_DIRS: dict[str, Callable[[Path, str], Path]] = {
    "experiment": lambda experiment_dir, _entry_id: Path(experiment_dir),
    "recording": lambda experiment_dir, entry_id: (
        experiment_store.recordings_dir(experiment_dir) / entry_id
    ),
    "run": runs.run_dir,
    "analysis_run": analysis_runs.analysis_dir,
    "dataset": datasets.dataset_dir,
    "model": models.model_dir,
    "distill": distill_runs.run_dir,
    "export_run": export_runs.export_dir,
    "comparison": comparisons.comparison_dir,
}

KINDS: tuple[str, ...] = tuple(ENTRY_DIRS)


def entry_dir(experiment_dir: Path | str, kind: str, entry_id: str = "") -> Path:
    """The directory that holds ``kind``/``entry_id``'s record.

    Raises:
        ValueError: If ``kind`` is not an entry kind, or an id is missing or given wrongly.
        FileNotFoundError: If there is no such entry.
    """
    if kind not in ENTRY_DIRS:
        raise ValueError(
            f"no notes for a {kind!r}: notes are kept on {', '.join(KINDS)}"
        )
    if kind == "experiment" and entry_id:
        raise ValueError("the experiment's own notes take no entry id")
    if kind != "experiment" and not entry_id:
        raise ValueError(f"a {kind}'s notes need its id")
    directory = ENTRY_DIRS[kind](Path(experiment_dir), entry_id)
    if not directory.is_dir():
        raise FileNotFoundError(f"no {kind} {entry_id!r} in {experiment_dir}")
    return directory


def read_notes(directory: Path | str) -> str:
    """The note in ``directory``, or ``""`` when there is none."""
    try:
        return atomic.read_text(Path(directory) / NOTES_FILENAME)
    except FileNotFoundError:
        return ""


def write_notes(directory: Path | str, text: str) -> None:
    """Replace the note in ``directory`` with ``text``; an empty or blank note removes the file.

    Atomic, because the page that writes it autosaves while another window or the CLI may be
    reading it. Never creates ``directory``: a note saved after its entry was deleted must not
    bring back a directory that discovery would then mistake for a half-written entry.
    """
    path = Path(directory) / NOTES_FILENAME
    if not text.strip():
        path.unlink(missing_ok=True)
        return
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    atomic.replace(tmp, path)
