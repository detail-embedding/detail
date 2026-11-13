"""Every remote-capable analysis module takes ``--remote-target`` at the CLI (§6 "Remote analysis").

**The class: a capability one surface of N can reach.** ``remote_capable`` is the module's own
declaration, and ``run_analysis`` takes ``compute`` for it, but each module's command is written
by hand: a module made remote-capable without the flag is remote in the API and the GUI and not
here, and nothing fails. The table below is the one place that names each module's command; a
remote-capable module missing from it fails, so a new one has to decide. The GUI half is
``tests/gui/test_run_on.py``.
"""

import pytest
from click.testing import CliRunner

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import Experiment
from detail.cli import main
from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

import_runner_modules(PIPELINE_RUNNERS)

#: module -> the ``detail analysis`` command that launches it.
COMMANDS = {
    "temporal_clustering": "temporal-clustering",
    "profile_describe": "profile-all",
    "cluster_lookalikes": "find-lookalikes",
}


def test_every_remote_capable_module_has_a_command_here():
    capable = {m for m, module in ANALYSIS_MODULES.items() if module.remote_capable}
    assert capable == set(COMMANDS)


@pytest.mark.parametrize("command", sorted(COMMANDS.values()))
def test_the_command_takes_a_remote_target(command):
    result = CliRunner().invoke(main, ["analysis", command, "--help"])
    assert result.exit_code == 0, result.output
    assert "--remote-target" in result.output


@pytest.mark.parametrize("command", ["temporal-clustering"])
def test_a_clustering_fit_sends_its_target_to_the_run(tmp_path, command):
    """The target reaches the module's validation: one this machine has not registered is
    refused by name, before any embedding is read."""
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    result = CliRunner().invoke(
        main,
        [
            "analysis",
            command,
            "--experiment",
            str(experiment.dir),
            "--recording",
            "recording_001",
            "--remote-target",
            "nowhere",
        ],
    )
    assert result.exit_code != 0
    assert "no remote target 'nowhere' is registered" in result.output
