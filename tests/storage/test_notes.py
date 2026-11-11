"""`storage/notes.py`: one plain-text note per entry, beside the entry's record (§15)."""

import pytest

from detail.storage import export as export_store
from detail.storage import notes
from detail.storage import runs as runs_store


def _run(experiment_dir, run_id="run_001"):
    runs_store.write_run(experiment_dir, run_id, {"id": run_id, "status": "complete"})
    return runs_store.run_dir(experiment_dir, run_id)


def test_an_entry_with_no_note_reads_empty(experiment_dir):
    assert notes.read_notes(_run(experiment_dir)) == ""


def test_a_note_reads_back_as_written(experiment_dir):
    directory = _run(experiment_dir)
    text = "Black/white pair — box 3.\nRe-run with conf 0.2?\n"
    notes.write_notes(directory, text)
    assert notes.read_notes(directory) == text
    assert (directory / notes.NOTES_FILENAME).is_file()


def test_a_blank_note_removes_the_file(experiment_dir):
    directory = _run(experiment_dir)
    notes.write_notes(directory, "something")
    notes.write_notes(directory, "  \n ")
    assert not (directory / notes.NOTES_FILENAME).exists()
    assert notes.read_notes(directory) == ""


def test_a_note_is_never_written_into_a_missing_directory(experiment_dir):
    gone = runs_store.run_dir(experiment_dir, "run_009")
    with pytest.raises(FileNotFoundError):
        notes.write_notes(gone, "late save after a delete")
    assert not gone.exists()


def test_entry_dir_resolves_every_kind(experiment_dir):
    for kind in notes.KINDS:
        entry_id = "" if kind == "experiment" else "x_001"
        notes.ENTRY_DIRS[kind](experiment_dir, entry_id).mkdir(
            parents=True, exist_ok=True
        )
        assert notes.entry_dir(experiment_dir, kind, entry_id).is_dir()


def test_entry_dir_refuses_an_unknown_kind(experiment_dir):
    with pytest.raises(ValueError, match="no notes for a 'tool_run'"):
        notes.entry_dir(experiment_dir, "tool_run", "x")


def test_entry_dir_refuses_a_missing_or_extra_id(experiment_dir):
    with pytest.raises(ValueError, match="need its id"):
        notes.entry_dir(experiment_dir, "run")
    with pytest.raises(ValueError, match="take no entry id"):
        notes.entry_dir(experiment_dir, "experiment", "run_001")


def test_entry_dir_refuses_a_missing_entry(experiment_dir):
    with pytest.raises(FileNotFoundError, match="no run 'run_404'"):
        notes.entry_dir(experiment_dir, "run", "run_404")


def test_a_path_quoted_in_a_note_does_not_block_the_experiment_export(experiment_dir):
    """A note is prose; quoting `C:\\Users\\…` in one is not a Tenet 6 violation."""
    directory = _run(experiment_dir)
    notes.write_notes(directory, r"videos were in C:\Users\lab\raw before the move")
    rel = (directory / notes.NOTES_FILENAME).relative_to(experiment_dir).as_posix()
    assert export_store.portability_offenses(experiment_dir, [rel]) == []
