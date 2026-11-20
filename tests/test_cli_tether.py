"""The CLI reaches the tether option and the tracking override (§12, §13 ``exclude_classes``).

Each command hands its flags to the API by name; the fake experiment records what arrived and
then refuses, so nothing runs.
"""

import types

import pytest
from click.testing import CliRunner

from detail import cli
from detail.cli import main


class _Recorded(ValueError):
    pass


@pytest.fixture
def calls(monkeypatch, tmp_path):
    seen = {}

    def _record(name):
        def method(*args, **kwargs):
            seen[name] = (args, kwargs)
            raise _Recorded("stop here")

        return method

    recording = types.SimpleNamespace(
        id="recording_001", run_tracking=_record("run_tracking")
    )
    proj = types.SimpleNamespace(
        distill=_record("distill"),
        label_dataset=_record("label_dataset"),
        run_processing=_record("run_processing"),
        resolve_recording=lambda ref: recording,
    )
    monkeypatch.setattr(cli, "_open_experiment", lambda _d: proj)
    return seen


def _invoke(*args):
    return CliRunner().invoke(main, list(args))


def test_distill_run_passes_tether(calls, tmp_path):
    _invoke(
        "distill", "run", "--experiment", str(tmp_path), "--name", "m",
        "--class", "mouse:a mouse", "--n-frames", "4", "--tether",
    )  # fmt: skip
    assert calls["distill"][1]["tether"] is True


def test_distill_run_is_untethered_by_default(calls, tmp_path):
    _invoke(
        "distill", "run", "--experiment", str(tmp_path), "--name", "m",
        "--class", "mouse:a mouse", "--n-frames", "4",
    )  # fmt: skip
    assert calls["distill"][1]["tether"] is False


def test_dataset_label_passes_tether(calls, tmp_path):
    _invoke(
        "dataset", "label", "--experiment", str(tmp_path), "--name", "d",
        "--class", "mouse:a mouse", "--n-frames", "4", "--tether",
    )  # fmt: skip
    assert calls["label_dataset"][1]["tether"] is True


@pytest.mark.parametrize(
    "flags, expected",
    [
        ((), None),
        (
            ("--exclude-class", "cable", "--exclude-class", "pellet"),
            ["cable", "pellet"],
        ),
        (("--track-all-classes",), []),
    ],
)
def test_tracking_run_passes_exclude_classes(calls, tmp_path, flags, expected):
    _invoke(
        "tracking", "run", "--experiment", str(tmp_path), "--recording", "r",
        "--detection", "det_01", *flags,
    )  # fmt: skip
    assert calls["run_tracking"][1]["exclude_classes"] == expected


def test_process_tracking_puts_exclude_classes_in_the_config(calls, tmp_path):
    _invoke(
        "process", "tracking", "--experiment", str(tmp_path), "--recording", "r",
        "--detection", "det_01", "--track-all-classes",
    )  # fmt: skip
    args, _kwargs = calls["run_processing"]
    assert args[0] == "tracking"
    assert args[2]["exclude_classes"] == []


def test_the_two_tracking_flags_are_refused_together(calls, tmp_path):
    result = _invoke(
        "tracking", "run", "--experiment", str(tmp_path), "--recording", "r",
        "--detection", "det_01", "--exclude-class", "cable", "--track-all-classes",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "not both" in result.output
    assert "run_tracking" not in calls
