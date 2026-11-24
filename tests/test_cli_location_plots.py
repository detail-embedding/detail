"""``detail locations plot`` — a location-export run drawn from the command line (§16).

**The view model is the flags.** Every ``LocationPlotView`` field is a flag on ``locations plot``,
its help, default and choices read from the model (``cli._view_options``); none is written twice.
``test_every_view_field_is_a_flag_with_the_model_s_help`` fails when a field is added to the model
without reaching the command, or when a flag's help or default is typed by hand.
"""

import numpy as np
from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import locations_plot, main
from detail.configs.describe import describe_config
from detail.configs.location_plots import LocationPlotView
from tests.location_fixtures import finished_run, track


def _experiment(tmp_path):
    proj = Experiment.create("p", tmp_path / "p")
    walk = track(10.0 + np.arange(100) * 4.0, np.full(100, 50.0))
    finished_run(
        proj,
        {"a": {0: walk}, "b": {0: walk}},
        metadata={"a": {"genotype": "wt"}, "b": {"genotype": "ko"}},
    )
    return proj


def test_every_view_field_is_a_flag_with_the_model_s_help():
    params = {p.name: p for p in locations_plot.params}
    for doc in describe_config(LocationPlotView):
        param = params.get(doc["name"])
        assert param is not None, f"locations plot has no flag for {doc['name']}"
        assert param.help == doc["help"], doc["name"]
        if doc["type"] == "array":
            # A list is a repeatable flag named for one item: --point nose --point tail.
            assert param.multiple and param.opts == ["--" + doc["name"][:-1]]
            continue
        assert param.default == doc["default"], doc["name"]
        if doc["choices"]:
            assert list(param.type.choices) == [str(c) for c in doc["choices"]]


def test_a_bound_the_model_holds_is_refused_by_the_flag():
    result = CliRunner().invoke(
        main,
        ["locations", "plot", "--experiment", ".", "--run", "an_01", "--out", "x",
         "--posture-every-s", "0"],
    )  # fmt: skip
    assert result.exit_code != 0 and "0.0<x<=3600.0" in result.output


def test_plot_writes_each_format(tmp_path):
    proj = _experiment(tmp_path)
    out = tmp_path / "figs" / "paths"
    result = CliRunner().invoke(
        main,
        ["locations", "plot", "--experiment", str(proj.dir), "--run", "an_01",
         "--out", str(out), "--format", "png", "--format", "svg",
         "--posture-every-s", "5"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out.with_suffix(".png")).is_file() and (out.with_suffix(".svg")).is_file()
    assert f"wrote: {out.with_suffix('.png')}" in result.output


def test_plot_writes_an_interactive_page(tmp_path):
    proj = _experiment(tmp_path)
    out = tmp_path / "speed"
    result = CliRunner().invoke(
        main,
        ["locations", "plot", "--experiment", str(proj.dir), "--run", "an_01",
         "--out", str(out), "--kind", "speed", "--format", "html"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".html").is_file() and not out.with_suffix(".png").exists()


def test_a_body_plot_by_group_for_chosen_recordings(tmp_path):
    proj = _experiment(tmp_path)
    out = tmp_path / "body"
    result = CliRunner().invoke(
        main,
        ["locations", "plot", "--experiment", str(proj.dir), "--run", "an_01",
         "--out", str(out), "--kind", "body", "--body-measure", "width",
         "--group-by", "genotype", "--recording", "a", "--recording", "b"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".png").is_file()


def test_an_unknown_column_or_recording_is_a_message_not_a_traceback(tmp_path):
    proj = _experiment(tmp_path)
    base = ["locations", "plot", "--experiment", str(proj.dir), "--run", "an_01",
            "--out", str(tmp_path / "x")]  # fmt: skip
    result = CliRunner().invoke(main, base + ["--group-by", "strain"])
    assert result.exit_code == 1 and "columns: genotype" in result.output
    result = CliRunner().invoke(main, base + ["--recording", "nobody"])
    assert result.exit_code == 1 and "nobody" in result.output
    assert "Traceback" not in result.output


def test_keypoint_flags_are_the_model_s_and_belong_to_the_location_export(tmp_path):
    from detail.cli import analysis_run, locations_export
    from detail.configs.location_export import LocationExportConfig

    described = LocationExportConfig.model_fields["keypoint_run_id"].description
    exported = {p.name: p for p in locations_export.params}["keypoint_run_id"]
    assert exported.help == described and exported.opts == ["--keypoints"]
    assert {p.name: p for p in analysis_run.params}["keypoint_run_id"].help.endswith(
        described
    )
    proj = _experiment(tmp_path)
    base = ["analysis", "run", "--experiment", str(proj.dir), "--recording", "a",
            "--tracking", "track_01"]  # fmt: skip
    result = CliRunner().invoke(
        main, base + ["--module", "query", "--keypoints", "kt_01"]
    )
    assert result.exit_code != 0
    assert (
        "--keypoints belong to the location_export module, not to query"
        in result.output
    )
    result = CliRunner().invoke(
        main, base + ["--keypoints", "kt_01", "--keypoints-ref", "a:kt_01"]
    )
    assert result.exit_code != 0 and "at most one of --keypoints" in result.output


def test_several_points_are_a_repeated_flag(tmp_path):
    from tests.location_fixtures import write_keypoints

    proj = Experiment.create("k", tmp_path / "k")
    walk = track(10.0 + np.arange(100) * 4.0, np.full(100, 50.0))
    ids = finished_run(proj, {"a": {0: walk}})
    write_keypoints(
        proj.analysis_run_dir("an_01") / ids["a"],
        walk,
        {"nose": (5.0, 0.0, None), "tail_base": (-5.0, 0.0, None)},
    )
    out = tmp_path / "points"
    result = CliRunner().invoke(
        main,
        ["locations", "plot", "--experiment", str(proj.dir), "--run", "an_01",
         "--out", str(out), "--point", "nose", "--point", "tail_base"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".png").is_file()
