"""`Experiment.notes` / `set_notes`, and notes travelling into a catalogue export (§15)."""

import pytest

from detail.api import exporters
from detail.api.experiment import Experiment
from detail.api.exporters import ExportSubject
from detail.storage import notes as notes_store
from detail.storage import runs as runs_store


@pytest.fixture
def experiment(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    runs_store.write_run(
        experiment.dir, "run_001", {"id": "run_001", "status": "complete"}
    )
    return experiment


def test_notes_round_trip_on_the_experiment_and_on_a_run(experiment):
    experiment.set_notes("experiment", "", "pilot cohort")
    experiment.set_notes("run", "run_001", "conf 0.1 was too low")
    assert experiment.notes("experiment") == "pilot cohort"
    assert experiment.notes("run", "run_001") == "conf 0.1 was too low"


def test_set_notes_with_empty_text_clears_them(experiment):
    experiment.set_notes("run", "run_001", "draft")
    experiment.set_notes("run", "run_001", "")
    assert experiment.notes("run", "run_001") == ""


def test_notes_on_a_missing_entry_are_refused(experiment):
    with pytest.raises(FileNotFoundError, match="no run 'run_404'"):
        experiment.set_notes("run", "run_404", "x")


def test_every_export_subject_kind_is_a_notes_kind():
    """`write_export` copies the subject's note by kind name, so a subject kind with no notes
    kind would raise there — caught here, the day the target is registered."""
    missing = {t.subject_kind for t in exporters.all_targets()} - set(notes_store.KINDS)
    assert not missing


def test_an_export_carries_the_subjects_notes(experiment, tmp_path):
    experiment.set_notes("experiment", "", "what this cohort was for")
    subject = ExportSubject("experiment", experiment.dir)
    result = exporters.write_export(
        subject, [exporters.target("experiment", "metadata")], tmp_path / "out"
    )
    assert (result.dest / exporters.NOTES_NAME).read_text(encoding="utf-8-sig") == (
        "what this cohort was for"
    )
    assert exporters.NOTES_NAME in exporters.read_manifest(result.dest)["files"]


def test_an_export_of_an_entry_without_notes_writes_none(experiment, tmp_path):
    subject = ExportSubject("experiment", experiment.dir)
    result = exporters.write_export(
        subject, [exporters.target("experiment", "metadata")], tmp_path / "out"
    )
    assert not (result.dest / exporters.NOTES_NAME).exists()
