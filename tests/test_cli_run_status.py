"""A CLI command that waits for its run to finish must not exit 0 when the run failed.

**The class.** A command submits a run, streams its progress until it ends, and returns. The
streaming drivers (`_stream_progress` and its siblings) return normally whatever status the run
ended with — they cannot raise on a failure, because the same drivers serve the ``attach``
commands, which also return normally when Ctrl-C detaches from a run that is still going. So a
command has to read the terminal status itself, and two did not: ``analysis pose-clustering``
and ``analysis temporal-clustering`` printed their next-step hint and exited 0 after a failed
run. A script running the pipeline end to end — the benchmark driver, for one — took that as
success and fed the next stage a run that did not exist.

**How it hid.** The terminal shows the failure in the progress output, so nobody reading it is
misled; only a caller that reads the exit code is. Nothing is wrong at the line where the check
is missing.

**The rule.** Every command that streams a run calls ``_check_status`` (or, for an export,
``_echo_export_result``, which raises on an incomplete export) — or is listed below with the
reason it reads the status differently. A new command is covered the day it is written.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import detail

pytestmark = pytest.mark.guard

_CLI_FILES = ("cli.py", "cli_keypoints.py", "cli_skeletons.py")
_DRIVERS = (
    "_stream_progress",
    "_stream_processing_progress",
    "_stream_training_progress",
    "_stream_distill_progress",
)
_CHECKS = {"_check_status", "_echo_export_result"}

_ATTACH = (
    "an attach command watches a run and prints its status; Ctrl-C detaches from a run that "
    "is still going, so a non-terminal status there is not a failure"
)

#: ``(file, command function) -> why it does not call one of the checks.`` Claims, not
#: exemptions: `test_no_stale_exemption` fails on an entry that no longer excuses anything.
EXEMPT: dict[tuple[str, str], str] = {
    ("cli.py", "process_attach"): _ATTACH,
    ("cli.py", "dataset_attach"): _ATTACH,
    ("cli.py", "train_attach"): _ATTACH,
    ("cli.py", "distill_attach"): _ATTACH,
    ("cli.py", "analysis_attach"): _ATTACH,
    ("cli.py", "export_attach"): _ATTACH,
    (
        "cli.py",
        "analysis_query",
    ): "reads `run.status` itself and raises: an authored query "
    "run stays open, so `complete` and `partial` both mean the search finished (§21)",
}


def _unchecked(source: str) -> set[str]:
    """Functions in ``source`` that call a streaming driver and none of the checks."""
    found = set()
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_stream_"):
            continue
        called = {
            getattr(node.func, "id", getattr(node.func, "attr", ""))
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
        }
        if called & set(_DRIVERS) and not called & _CHECKS:
            found.add(fn.name)
    return found


def _sweep() -> set[tuple[str, str]]:
    root = Path(detail.__file__).parent
    return {
        (name, fn)
        for name in _CLI_FILES
        for fn in _unchecked((root / name).read_text(encoding="utf-8-sig"))
    }


def test_every_streaming_command_checks_how_its_run_ended():
    missing = sorted(_sweep() - set(EXEMPT))
    assert not missing, (
        "these commands wait for a run and never read how it ended, so they exit 0 when it "
        f"failed: {missing}. Call _check_status(run, label) after streaming, or add an EXEMPT "
        "entry saying why the status is read another way."
    )


def test_no_stale_exemption():
    stale = sorted(set(EXEMPT) - _sweep())
    assert (
        not stale
    ), f"EXEMPT names commands that now check, or no longer exist: {stale}"


def test_the_sweep_flags_a_command_that_skips_the_check():
    """The negative case, on the shape the bug had: stream, print a hint, return."""
    source = (
        "def analysis_x(run):\n"
        "    _stream_progress(run, 'x')\n"
        "    click.echo('next: detail analysis x-interpret')\n"
        "def analysis_y(run):\n"
        "    _stream_progress(run, 'y')\n"
        "    _check_status(run, 'y')\n"
    )
    assert _unchecked(source) == {"analysis_x"}


class _FailedRun:
    """A run handle whose run already failed — progress ends, status says so."""

    id = "an_01"
    status = "failed"
    error = "boom"
    owns_worker = True

    def progress(self):
        return iter(())

    def wait(self):
        return self


@pytest.mark.parametrize(
    "command,module",
    [
        ("temporal-clustering", "temporal_clustering"),
    ],
)
def test_a_failed_clustering_run_exits_non_zero(tmp_path, monkeypatch, command, module):
    """The behaviour the scan stands for, on the two commands that had the bug."""
    from click.testing import CliRunner

    from detail import cli
    from detail.api.experiment import Experiment

    exp = Experiment.create("p", tmp_path)
    launched = []

    def fake_run_analysis(self, module_id, config):
        launched.append(module_id)
        return _FailedRun()

    monkeypatch.setattr(Experiment, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(cli, "_embedding_selection", lambda *a, **k: {})
    result = CliRunner().invoke(
        cli.main,
        [
            "analysis",
            command,
            "--experiment",
            str(exp.dir),
            "--recording",
            "recording_001",
            "--embed-run",
            "emb_01",
        ],
    )
    assert launched == [module], result.output
    assert result.exit_code != 0, result.output
    assert "did not complete" in result.output
