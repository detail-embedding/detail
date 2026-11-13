"""The camera view through the CLI (§10): the same setting the API and GUI reach.

`experiment create --camera-view` validates before creating anything, so a bad view leaves no
half-made experiment behind.
"""

from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import main


def _run(*args):
    return CliRunner().invoke(main, list(args))


def test_create_sets_the_default_and_status_shows_it(tmp_path):
    result = _run(
        "experiment", "create", "ex", "--dir", str(tmp_path), "--camera-view", "side"
    )
    assert result.exit_code == 0, result.output
    status = _run("experiment", "status", "--experiment", str(tmp_path / "ex"))
    assert "camera view (default): Side view" in status.output


def test_create_with_a_bad_view_makes_nothing(tmp_path):
    result = _run(
        "experiment", "create", "ex", "--dir", str(tmp_path), "--camera-view", "other"
    )
    assert result.exit_code != 0
    assert "needs a description" in result.output
    assert not (tmp_path / "ex").exists()


def test_a_recording_overrides_and_default_inherits_again(tmp_path):
    exp = Experiment.create("ex", tmp_path)
    exp.set_camera_view("side")
    exp.create_recording("r1")
    root = str(exp.dir)
    result = _run(
        "recording",
        "set-camera-view",
        "--experiment",
        root,
        "--recording",
        "r1",
        "other",
        "--note",
        "45 deg",
    )
    assert result.exit_code == 0, result.output
    status = _run("recording", "status", "--experiment", root, "--recording", "r1")
    assert "camera view: Other: 45 deg" in status.output
    result = _run(
        "recording",
        "set-camera-view",
        "--experiment",
        root,
        "--recording",
        "r1",
        "default",
    )
    assert "Side view (experiment default)" in result.output


def test_unset_removes_the_default(tmp_path):
    exp = Experiment.create("ex", tmp_path)
    exp.set_camera_view("side")
    result = _run(
        "experiment", "set-camera-view", "--experiment", str(exp.dir), "unset"
    )
    assert result.exit_code == 0, result.output
    assert "Not set" in result.output
    assert not exp.camera_view.is_set


def test_a_note_on_a_named_view_is_refused(tmp_path):
    exp = Experiment.create("ex", tmp_path)
    result = _run(
        "experiment",
        "set-camera-view",
        "--experiment",
        str(exp.dir),
        "side",
        "--note",
        "x",
    )
    assert result.exit_code != 0
    assert "only kept for 'other'" in result.output
