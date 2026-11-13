"""Class names through the CLI (§1 *Class names*): the same rename the API and GUI reach,
by either of a class's names."""

import pytest
from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import main
from detail.storage import runs as runs_store


def _run(*args):
    return CliRunner().invoke(main, list(args))


@pytest.fixture
def experiment(tmp_path):
    experiment = Experiment.create("cn", tmp_path)
    runs_store.write_run(
        experiment.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "config": {"model_id": "m", "class_conf_thresholds": {"A": 0.5, "B": 0.5}},
            "status": "complete",
        },
    )
    return experiment


def test_classes_lists_each_class_and_its_name(experiment):
    experiment.rename_class("A", "Intruder")
    result = _run("experiment", "classes", "--experiment", str(experiment.dir))
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["A → Intruder", "B"]


def test_rename_class_takes_either_name_and_reset_puts_back_the_stored_one(experiment):
    exp = str(experiment.dir)
    result = _run("experiment", "rename-class", "--experiment", exp, "A", "Intruder")
    assert result.exit_code == 0, result.output
    assert "A: called Intruder" in result.output
    result = _run(
        "experiment", "rename-class", "--experiment", exp, "Intruder", "Visitor"
    )
    assert result.exit_code == 0, result.output
    assert experiment.renamed_classes() == {"A": "Visitor"}
    result = _run("experiment", "rename-class", "--experiment", exp, "A", "--reset")
    assert result.exit_code == 0, result.output
    assert experiment.renamed_classes() == {}


def test_rename_class_refusals_are_clean_errors(experiment):
    exp = str(experiment.dir)
    result = _run("experiment", "rename-class", "--experiment", exp, "A", "B")
    assert result.exit_code != 0
    assert "'B' is already a class" in result.output
    result = _run("experiment", "rename-class", "--experiment", exp, "rat", "Rat")
    assert result.exit_code != 0
    assert "no class 'rat'" in result.output
    result = _run("experiment", "rename-class", "--experiment", exp, "A")
    assert result.exit_code != 0
    assert "give NAME, or --reset" in result.output
