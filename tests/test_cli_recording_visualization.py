"""``detail analysis visualize-recording`` — the CLI over ``RecordingVisualization`` (§22)."""

from click.testing import CliRunner

from detail.cli import main
from detail.configs.recording_visualization import (
    RecordingVisualizationConfig,
    RecordingVisualizationExport,
)


def _invoke(*args):
    return CliRunner().invoke(
        main, ["analysis", "visualize-recording", *args], catch_exceptions=False
    )


def test_the_help_is_the_models_help():
    """Bounds and help live in the config models; the CLI reads them, never restates them."""
    result = CliRunner().invoke(
        main, ["analysis", "visualize-recording", "--help"], terminal_width=400
    )
    text = " ".join(result.output.split())
    for model, name in (
        (RecordingVisualizationConfig, "start_s"),
        (RecordingVisualizationConfig, "clusters"),
        (RecordingVisualizationConfig, "location"),
        (RecordingVisualizationExport, "speed"),
        (RecordingVisualizationExport, "width"),
    ):
        assert model.model_fields[name].description.split(".")[0][:30] in text, name


def test_save_then_list_then_estimate(visualizable_run):
    experiment, run = visualizable_run
    base = ["--experiment", str(experiment.dir), "--run", run.id]
    result = _invoke(
        *base, "--recording", "recording_001", "--from-s", "0.5", "--to-s", "1.5",
        "--no-location", "--save", "--estimate", "--width", "640",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "saved as viz_001" in result.output
    assert "MB per file" in result.output
    listed = _invoke(*base, "--list")
    assert "viz_001" in listed.output and "0.5–1.5 s" in listed.output


def test_export_without_out_is_refused(visualizable_run):
    experiment, run = visualizable_run
    result = CliRunner().invoke(
        main,
        ["analysis", "visualize-recording", "--experiment", str(experiment.dir), "--run", run.id,
         "--recording", "recording_001", "--from-s", "0.5", "--to-s", "1.0"],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "give --out" in result.output


def test_nothing_to_draw_is_refused(visualizable_run):
    experiment, run = visualizable_run
    result = CliRunner().invoke(
        main,
        [
            "analysis",
            "visualize-recording",
            "--experiment",
            str(experiment.dir),
            "--run",
            run.id,
        ],
    )
    assert result.exit_code != 0
    assert "give --recording" in result.output
