"""``detail locations plot`` — the CLI over ``Experiment.location_plots`` (§16)."""

from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import main
from detail.configs.describe import describe_config
from detail.configs.location_plots import LocationPlotView
from tests.location_runs import location_run, two_animals


def _invoke(*args):
    return CliRunner().invoke(
        main, ["locations", "plot", *args], catch_exceptions=False
    )


def test_every_view_field_is_a_flag_with_the_model_s_help():
    """Enumerated from the model, so a field added there is reachable here the same day."""
    result = CliRunner().invoke(
        main, ["locations", "plot", "--help"], terminal_width=400
    )
    for field in describe_config(LocationPlotView):
        flag = "--" + field["name"].replace("_", "-")
        if field["type"] == "array":
            flag = flag.removesuffix("s")  # a list field is a repeatable singular flag
        assert flag in result.output, flag
        assert field["help"].split(".")[0][:30] in " ".join(result.output.split())


def test_plot_writes_each_format_by_recording_name_and_group(tmp_path):
    proj = Experiment.create("p", tmp_path)
    location_run(proj, {"recording_001": {}, "recording_002": {}})
    proj.resolve_recording("recording_002").set_name("larva B")
    proj.set_recording_metadata("recording_001", {"genotype": "wt"})
    out = tmp_path / "figs" / "heat"
    result = _invoke(
        "--experiment", str(proj.dir), "--run", "an_01", "--out", str(out),
        "--kind", "heatmap", "--group-by", "genotype", "--format", "png",
        "--format", "svg", "--recording", "larva B", "--recording", "recording_001",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".png").is_file() and out.with_suffix(".svg").is_file()


def test_a_bad_view_is_a_clean_error_not_a_traceback(tmp_path):
    proj = Experiment.create("p", tmp_path)
    location_run(proj, {"recording_001": {}})
    result = CliRunner().invoke(
        main,
        [
            "locations", "plot", "--experiment", str(proj.dir), "--run", "an_01",
            "--out", str(tmp_path / "x"), "--group-by", "sex",
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "no Recording Information column 'sex'" in result.output
    assert "Traceback" not in result.output


def test_bounds_come_from_the_model(tmp_path):
    result = CliRunner().invoke(
        main,
        ["locations", "plot", "--experiment", str(tmp_path), "--run", "an_01",
         "--out", str(tmp_path / "x"), "--bins", "1"],
    )  # fmt: skip
    assert result.exit_code == 2 and "--bins" in result.output


def test_subject_and_split_by_subject_reach_the_plot(tmp_path):
    proj = Experiment.create("p", tmp_path)
    labels = {0: "resident", 1: "intruder"}
    location_run(
        proj,
        {
            "recording_001": {"rows": two_animals(), "subject_labels": labels},
            "recording_002": {"rows": two_animals(), "subject_labels": labels},
        },
    )
    out = tmp_path / "speed"
    result = _invoke(
        "--experiment", str(proj.dir), "--run", "an_01", "--out", str(out),
        "--kind", "speed", "--subject", "intruder", "--subject", "resident",
        "--split-by-subject",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".png").is_file()


# --------------------------------------------------------------------------- #
# The export's keypoint and pair inputs (§16 *Keypoints and pairs in the export*)
# --------------------------------------------------------------------------- #
def test_locations_export_writes_keypoints_and_pairs(tmp_path):
    from tests.analysis.test_locations import _mask_square, _setup
    from tests.experiments import write_keypoint_track_run

    recording = _setup(
        tmp_path,
        detections=[(0, 0, (0.1, 0.1, 0.5, 0.5), _mask_square(4, 20, 4, 20))],
        object_ids=[0],
    )
    write_keypoint_track_run(recording.dir, n_frames=1)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "locations", "export", "--experiment", str(recording.dir.parent.parent),
            "--recording", "recording_001", "--tracking", "track_01", "--out", str(out),
            "--keypoints", "kt_01", "--pair-edge-distance",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out / "keypoints.csv").is_file() and (out / "pair_relations.csv").is_file()


def test_locations_export_refuses_an_unknown_keypoint_run_cleanly(tmp_path):
    from tests.analysis.test_locations import _one_track

    recording = _one_track(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "locations", "export", "--experiment", str(recording.dir.parent.parent),
            "--recording", "recording_001", "--tracking", "track_01",
            "--out", str(tmp_path / "out"), "--keypoints", "kt_09",
        ],
    )  # fmt: skip
    assert result.exit_code != 0 and "kt_09" in result.output
    assert "Traceback" not in result.output


def test_analysis_run_takes_one_keypoint_selector(tmp_path, monkeypatch):
    from detail import cli

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001").set_name("r1")
    base = [
        "analysis", "run", "--experiment", str(proj.dir), "--recording", "r1",
        "--tracking", "track_01",
    ]  # fmt: skip
    both = CliRunner().invoke(
        main, base + ["--keypoints", "kt_01", "--keypoints-ref", "r1:kt_01"]
    )
    assert both.exit_code != 0 and "not both" in both.output

    seen = {}
    monkeypatch.setattr(
        cli,
        "_spec_from_ref",
        lambda _p, module_id, ref: {"module": module_id, "r": ref},
    )

    def capture(self, module_id, config):
        seen.update(config)
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "run_analysis", capture)
    CliRunner().invoke(
        main, base + ["--keypoints-ref", "r1:kt_01", "--pair-edge-distance"]
    )
    assert seen["keypoint_track_spec"] == {"module": "keypoint_track", "r": "r1:kt_01"}
    assert seen["pair_edge_distance"] is True


# --------------------------------------------------------------------------- #
# locations montage (§16 "Preview and montage")
# --------------------------------------------------------------------------- #
def test_every_montage_field_is_a_flag_with_the_model_s_help():
    from detail.configs.location_plots import LocationMontageView

    result = CliRunner().invoke(
        main, ["locations", "montage", "--help"], terminal_width=400
    )
    for field in describe_config(LocationMontageView):
        flag = "--" + field["name"].replace("_", "-")
        if field["type"] == "array":
            flag = flag.removesuffix("s")
        assert flag in result.output, flag
    assert "--no-mask" in result.output  # a field that defaults on can be turned off


def test_locations_montage_writes_a_gif_and_a_last_frame(tmp_path):
    from tests.api.test_location_montage import walked as walked_fixture

    experiment, _plots = walked_fixture.__wrapped__(tmp_path)
    for out, extra in (
        (tmp_path / "m.gif", []),
        (tmp_path / "last.png", ["--last-frame"]),
    ):
        result = CliRunner().invoke(
            main,
            [
                "locations", "montage", "--experiment", str(experiment.dir),
                "--run", "an_01", "--out", str(out), "--full-length", *extra,
            ],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert out.is_file()


def test_plot_writes_an_html_page(tmp_path):
    proj = Experiment.create("p", tmp_path)
    location_run(proj, {"recording_001": {}})
    out = tmp_path / "page"
    result = _invoke(
        "--experiment", str(proj.dir), "--run", "an_01", "--out", str(out),
        "--format", "html",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".html").is_file()
