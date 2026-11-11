"""`detail notes show|set` (§15): the same notes the API and the GUI's Notes box read."""

from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import main
from detail.storage import runs as runs_store


def _run(*args, stdin=None):
    return CliRunner().invoke(main, list(args), input=stdin)


def _experiment(tmp_path):
    experiment = Experiment.create("ex", tmp_path)
    runs_store.write_run(
        experiment.dir, "run_001", {"id": "run_001", "status": "complete"}
    )
    return experiment


def test_set_then_show_a_runs_notes(tmp_path):
    root = str(_experiment(tmp_path).dir)
    result = _run(
        "notes", "set", "--experiment", root, "run", "run_001", "--text", "hi"
    )
    assert result.exit_code == 0, result.output
    assert "notes saved: run run_001" in result.output
    shown = _run("notes", "show", "--experiment", root, "run", "run_001")
    assert shown.output.strip() == "hi"


def test_set_reads_standard_input_and_clear_removes(tmp_path):
    experiment = _experiment(tmp_path)
    root = str(experiment.dir)
    result = _run(
        "notes",
        "set",
        "--experiment",
        root,
        "experiment",
        "--file",
        "-",
        stdin="a\nb\n",
    )
    assert result.exit_code == 0, result.output
    assert experiment.notes("experiment") == "a\nb\n"
    cleared = _run("notes", "set", "--experiment", root, "experiment", "--clear")
    assert "notes cleared: experiment" in cleared.output
    assert _run("notes", "show", "--experiment", root, "experiment").output.strip() == (
        "(no notes)"
    )


def test_set_takes_exactly_one_source(tmp_path):
    root = str(_experiment(tmp_path).dir)
    result = _run(
        "notes", "set", "--experiment", root, "run", "run_001", "--text", "x", "--clear"
    )
    assert result.exit_code != 0
    assert "exactly one of --text, --file or --clear" in result.output


def test_a_missing_entry_is_an_error_not_a_traceback(tmp_path):
    root = str(_experiment(tmp_path).dir)
    result = _run("notes", "show", "--experiment", root, "run", "run_404")
    assert result.exit_code != 0
    assert "no run 'run_404'" in result.output


def test_hidden_kinds_are_not_offered(tmp_path):
    root = str(_experiment(tmp_path).dir)
    result = _run("notes", "show", "--experiment", root, "clusterer", "c1")
    assert result.exit_code != 0
    assert "Invalid value" in result.output
