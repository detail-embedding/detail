"""Every parameter a clustering method declares can be set from the command line.

**The class.** A CLI that hand-lists a method's knobs as options is complete on the day it is
written. ``temporal-clustering`` named seven (``--detector``, ``--min-bout``, ``--band``,
``--max-bouts``, ``--fit-share``, ``--codewords``, ``--min-run``) while its two methods declare
twenty, so ``seed`` (asked for by the user, 2026-11-08), ``window_s``, ``max_gap_s`` and most
of ``dtw_finch``'s knobs could be set in the GUI's launch form and nowhere in a terminal.
Nothing failed: the run simply used the default.

**The rule.** The command takes ``--method-param NAME=VALUE`` for anything the chosen method
declares, typed and bounded by the method's own model at ``validate()``, plus ``--seed`` by
name. A parameter given both ways is refused, and ``--seed`` for a method with no seed is
refused rather than dropped (``seed`` is a retired name the resolver drops, for old saved
configs). Guarded here over the registries, so a method registered tomorrow is covered.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from detail import cli
from detail.pipeline import temporal_clustering

COMMANDS = {
    "temporal-clustering": ("temporal_clustering", cli._temporal_cluster_methods),
}


def _fields(command: str, method: str) -> list[str]:
    del command
    return sorted(temporal_clustering.resolve_method(method).config_model.model_fields)


def _cases() -> list[tuple[str, str, str]]:
    return [
        (command, method, field)
        for command, (_module, methods) in COMMANDS.items()
        for method in methods()
        for field in _fields(command, method)
    ]


class _Captured(ValueError):
    pass


@pytest.fixture
def launched(monkeypatch):
    """What each command hands ``run_analysis``: the config, stopped there."""
    seen: dict = {}

    class _Experiment:
        def run_analysis(self, module_id, config):
            seen["module"], seen["config"] = module_id, config
            raise _Captured("captured")

    monkeypatch.setattr(cli, "_open_experiment", lambda _dir: _Experiment())
    monkeypatch.setattr(cli, "_resolve_recording_refs", lambda _p, refs: list(refs))
    monkeypatch.setattr(cli, "_embedding_selection", lambda *_a: {})
    monkeypatch.setattr(cli, "_remote_compute_config", lambda _t: {})
    return seen


def _invoke(command: str, method: str, *extra: str):
    return CliRunner().invoke(
        cli.main,
        ["analysis", command, "--experiment", ".", "--recording", "r1",
         "--method", method, *extra],
    )  # fmt: skip


@pytest.mark.parametrize("command,method,field", _cases())
def test_every_method_parameter_is_reachable(launched, command, method, field):
    result = _invoke(command, method, "--method-param", f"{field}=1")
    assert "captured" in result.output, result.output
    assert launched["config"]["method_params"][field] == "1"


@pytest.mark.parametrize("command", COMMANDS)
def test_seed_reaches_every_method_that_takes_one_and_is_refused_elsewhere(
    launched, command
):
    _module, methods = COMMANDS[command]
    for method in methods():
        launched.clear()
        result = _invoke(command, method, "--seed", "7")
        if "seed" in _fields(command, method):
            assert launched["config"]["method_params"]["seed"] == 7, result.output
        else:
            assert result.exit_code != 0 and not launched, method
            assert "seed" in result.output


def test_a_seed_from_the_command_line_is_typed_by_the_methods_own_model():
    from detail.configs.method_params import resolve_method_params

    method = temporal_clustering.resolve_method("codebook_viterbi")
    assert resolve_method_params(method, {"seed": "7"})["seed"] == 7


@pytest.mark.parametrize(
    "command,method,extra",
    [
        (
            "temporal-clustering",
            "codebook_viterbi",
            ("--seed", "7", "--method-param", "seed=8"),
        ),
        (
            "temporal-clustering",
            "codebook_viterbi",
            ("--codewords", "40", "--method-param", "codewords=50"),
        ),
    ],
)
def test_a_parameter_given_two_ways_is_refused(launched, command, method, extra):
    result = _invoke(command, method, *extra)
    assert result.exit_code != 0 and not launched
    assert "given twice" in result.output


def test_a_parameter_the_method_does_not_take_is_refused(launched):
    result = _invoke(
        "temporal-clustering", "dtw_finch", "--method-param", "codewords=40"
    )
    assert result.exit_code != 0 and not launched
    assert "not a parameter of method 'dtw_finch'" in result.output


def test_the_seed_help_names_exactly_the_methods_with_a_seed():
    named = {
        method
        for command, (_module, methods) in COMMANDS.items()
        for method in methods()
        if "seed" in _fields(command, method)
    }
    assert set(cli._seeded_methods().split(", ")) == named
    assert named, "the help has methods to name"
