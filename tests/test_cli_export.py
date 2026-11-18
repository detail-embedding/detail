"""`detail export` — the CLI half of the one catalogue (#336).

Export is the one feature that has to work identically headless: the destination is a GLM or a
collaborator's machine, not a click. So the CLI is first-class rather than a mirror written after
the GUI, and these tests drive the same catalogue the GUI menu reads.
"""

import csv

from click.testing import CliRunner

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters
from detail.api.exporters import ExportSubject
from detail.cli import main


def _run(*args):
    return CliRunner().invoke(main, list(args))


def _rows(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_list_names_the_files_that_come_with_each_artifact(authored_cohort):
    """`export list` prints the filenames that ship beside the artifact, not only the artifact."""
    result = _run(
        "export", "list",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "instances" in result.output
    assert "ethogram" in result.output
    assert (
        "coverage.csv" in result.output
    ), "the data files are what make this more than Save-As"
    assert "params.csv" in result.output and "export.yaml" in result.output


def test_list_says_why_a_target_cannot_run(authored_cohort):
    """Every target the subject *can* have is listed, so the reason has to be listed too.

    `docs/cli.md` has claimed this since the catalogue shipped and the command never did it: the
    GUI greys a row and names the missing thing, and the CLI printed the row with nothing to
    separate "DETAIL cannot export this" from "this run has none of it yet". The fixture records
    no embedding run, so `clips` is the target that cannot run.
    """
    result = _run(
        "export", "list",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    reason = exporters.unavailable_reason(
        exporters.target("analysis_run", "clips"),
        ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01"),
    )
    assert reason in result.output, "the row is printed; its reason is not"


def test_list_prints_no_reason_for_a_target_that_can_run(authored_cohort):
    """Guard the guard: a "cannot run" line under every row would say nothing at all.

    The fixture has both kinds — `instances` and `ethogram` run, `clips` and `performance`
    cannot — so the count has to sit strictly between nothing and everything.
    """
    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01")
    listed = exporters.targets_for(subject)
    runnable = exporters.available_targets(subject)
    assert runnable and len(runnable) < len(listed), "the fixture must have both kinds"

    result = _run(
        "export", "list",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01",
    )  # fmt: skip
    assert result.output.count("cannot run:") == len(listed) - len(
        runnable
    ), result.output


def test_list_on_a_tracking_run_offers_its_tracks(processed_recording):
    """The upstream half: a tracking run's own data, without going through an analysis."""
    result = _run(
        "export", "list",
        "--experiment", str(processed_recording.experiment.dir),
        "--run", processed_recording.tracking_run_id,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "tracks" in result.output
    assert "these come from an embedding run" in result.output


def test_run_writes_a_tracking_runs_trajectory(processed_recording, tmp_path):
    result = _run(
        "export", "run",
        "--experiment", str(processed_recording.experiment.dir),
        "--run", processed_recording.tracking_run_id,
        "--target", "tracks",
        "--out", str(tmp_path),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    written = sorted(tmp_path.glob("*/recording_001_tracks.csv"))
    assert written, result.output
    assert _rows(written[0])[0]["object_id"] == "0"


def test_list_on_an_experiment_shows_the_bundle_as_portable(authored_cohort):
    result = _run("export", "list", "--experiment", str(authored_cohort.experiment.dir))
    assert result.exit_code == 0, result.output
    assert "bundle" in result.output
    assert (
        "[portable]" in result.output
    ), "a bundle round-trips and an open format does not; the list has to say which"


def test_run_writes_into_its_own_named_folder(authored_cohort, tmp_path):
    parent = tmp_path / "out"
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01",
        "--target", "instances",
        "--out", str(parent),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    folders = list(parent.iterdir())
    assert len(folders) == 1, f"expected one export folder, got {folders}"
    dest = folders[0]
    assert "an_01" in dest.name and "export" in dest.name
    assert (dest / "instances.csv").is_file()
    assert (dest / "coverage.csv").is_file()
    assert (dest / "export.yaml").is_file()
    assert "Wrote" in result.output


def test_run_takes_several_targets_into_one_folder(authored_cohort, tmp_path):
    """The CLI half of the checklist: one export, one folder, however much you asked for."""
    parent = tmp_path / "out"
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01",
        "--target", "instances", "--target", "ethogram",
        "--out", str(parent),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    dest = next(iter(parent.iterdir()))
    assert (dest / "instances.csv").is_file()
    assert (dest / "r1_ethogram.csv").is_file()


def test_run_exports_a_selection(authored_cohort, tmp_path):
    """The same call with the rows named — "it might be export by individually selecting it"."""
    everything = tmp_path / "all"
    _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "instances", "--out", str(everything),
    )  # fmt: skip
    picked = [
        r["instance_id"]
        for r in _rows(next(iter(everything.iterdir())) / "instances.csv")[:2]
    ]

    parent = tmp_path / "some"
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "instances", "--out", str(parent),
        "--instance", picked[0], "--instance", picked[1],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    dest = next(iter(parent.iterdir()))
    assert [r["instance_id"] for r in _rows(dest / "instances.csv")] == picked


def test_run_writes_per_frame_labels_for_a_frame_range(authored_cohort, tmp_path):
    """The headless path to the GLM regressor: one row per frame, the range honoured."""
    parent = tmp_path / "out"
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "frame_labels",
        "--frames", "2:5", "--out", str(parent),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    rows = _rows(next(iter(parent.iterdir())) / "r1_subject0_frame_labels.csv")
    assert list(rows[0]) == ["frame_idx", "t_ns", "observed", "rearing"]
    assert [r["frame_idx"] for r in rows] == ["2", "3", "4", "5"]
    assert [r["rearing"] for r in rows] == ["0", "1", "1", "0"]


def test_an_empty_export_says_why_rather_than_leaving_an_empty_folder(
    authored_cohort, tmp_path
):
    """A cluster the run has, holding no instance: nothing to write, said so."""
    from detail.storage import behavior_output as bo

    run_dir = authored_cohort.run_dir()
    artifact = bo.read_behavior_output(run_dir)
    manifest = {
        **artifact.manifest,
        "names": {**artifact.manifest.get("names", {}), "c09": ["sleep"]},
        "vocabulary": [*artifact.manifest.get("vocabulary", []), "sleep"],
    }
    bo.write_behavior_output(
        run_dir, instances=artifact.instances, coverage=artifact.coverage,
        manifest=manifest,
    )  # fmt: skip
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "instances",
        "--out", str(tmp_path / "out"), "--cluster", "c09",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "nothing to write" in result.output
    assert "cluster c09" in result.output


def test_an_unknown_cluster_exits_non_zero_and_writes_nothing(
    authored_cohort, tmp_path
):
    """`--target ethogram --cluster zz` said "Wrote 2 file(s)" and exited 0."""
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "ethogram",
        "--out", str(tmp_path / "out"), "--cluster", "zz",
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "this run has no cluster 'zz'" in result.output
    assert not (tmp_path / "out").exists()


def test_a_scope_the_target_does_not_take_exits_non_zero(authored_cohort, tmp_path):
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "instances",
        "--out", str(tmp_path / "out"), "--frames", "0:3",
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "Behavior instances cannot be narrowed to a frame range" in result.output
    assert not (tmp_path / "out").exists()


def test_a_clip_without_frames_exits_non_zero(processed_recording, tmp_path):
    """It exited 0 with "Clip had nothing to write for everything."."""
    result = _run(
        "export", "run",
        "--experiment", str(processed_recording.experiment.dir),
        "--recording", "recording_001", "--target", "clip",
        "--embed-run", "emb_01", "--out", str(tmp_path / "out"),
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "Clip needs a frame range to export" in result.output
    assert not (tmp_path / "out").exists()


def test_an_unknown_target_lists_what_is_available(authored_cohort, tmp_path):
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "clips-please",
        "--out", str(tmp_path / "out"),
    )  # fmt: skip
    assert result.exit_code != 0
    assert "no export target 'clips-please'" in result.output
    assert "instances" in result.output, "an error names what you could have asked for"


def test_two_subject_flags_are_refused(authored_cohort, tmp_path):
    """Inference was rejected for exactly this: two ids and no way to tell which was meant."""
    result = _run(
        "export", "list",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--recording", "r1",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "exactly one of" in result.output


def test_a_subject_kind_without_its_id_is_refused(authored_cohort):
    result = _run(
        "export", "list",
        "--experiment", str(authored_cohort.experiment.dir),
        "--subject", "analysis_run",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "--analysis-run <id>" in result.output


def test_a_contradicting_subject_kind_is_refused(authored_cohort):
    result = _run(
        "export", "list",
        "--experiment", str(authored_cohort.experiment.dir),
        "--subject", "recording", "--analysis-run", "an_01",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "contradicts" in result.output


def test_a_distill_run_is_named_by_its_own_flag(authored_cohort):
    """``--distill-run`` is the subject, and the refusals name it by that flag rather than a
    spelling derived from the kind (``--distill``), which no command accepts."""
    exp = str(authored_cohort.experiment.dir)
    listed = _run("export", "list", "--experiment", exp, "--distill-run", "distill_001")
    assert listed.exit_code == 0, listed.output
    assert "performance" in listed.output
    bare = _run("export", "list", "--experiment", exp, "--subject", "distill")
    assert bare.exit_code != 0
    assert "--distill-run <id>" in bare.output


def test_export_streams_the_run_it_submitted(authored_cohort, tmp_path, monkeypatch):
    """A long op that reports nothing is indistinguishable from a hang (the long-op rule).

    The **seam moved** with #354: the CLI no longer calls `write_export` and hands it a
    callback — it submits a run and streams the handle, and the worker is what drives the
    export. So the wiring to assert is that the submitted handle reaches `_stream_progress`;
    spying on `write_export` now watches a function this process never calls, which is a test
    that passes for the wrong reason in one direction and fails for the wrong reason in the
    other.
    """
    from detail import cli as cli_mod

    streamed = []
    real = cli_mod._stream_progress

    def spy(handle, label, *a, **k):
        streamed.append((handle.id, label))
        return real(handle, label, *a, **k)

    monkeypatch.setattr(cli_mod, "_stream_progress", spy)
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01", "--target", "instances",
        "--out", str(tmp_path / "out"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert streamed, "the CLI submitted an export and never streamed it"
    assert streamed[0][1] == "export"
    assert "submitted export:" in result.output, "the run id is how you come back to it"


# --------------------------------------------------------------------------- #
# --theme: figures for print by default, for slides when asked
# --------------------------------------------------------------------------- #
def _corner(png):
    from tests.png_pixel import top_left_rgb

    return top_left_rgb(png)


def test_run_draws_its_figures_dark_when_asked(authored_cohort, tmp_path):
    from detail.core.chart_theme import DARK, LIGHT, rgb

    for theme, surface in (("light", LIGHT.surface), ("dark", DARK.surface)):
        parent = tmp_path / theme
        args = [
            "export", "run",
            "--experiment", str(authored_cohort.experiment.dir),
            "--analysis-run", "an_01",
            "--target", "instances",
            "--out", str(parent),
        ]  # fmt: skip
        if theme == "dark":
            args += ["--theme", "dark"]
        result = _run(*args)
        assert result.exit_code == 0, result.output
        dest = next(iter(parent.iterdir()))
        assert _corner(dest / "time_by_behavior.png") == rgb(surface), theme


def test_an_unknown_theme_is_refused_by_the_command_line(authored_cohort, tmp_path):
    result = _run(
        "export", "run",
        "--experiment", str(authored_cohort.experiment.dir),
        "--analysis-run", "an_01",
        "--target", "instances",
        "--out", str(tmp_path / "out"),
        "--theme", "sepia",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "sepia" in result.output


def test_the_api_refuses_an_unknown_theme_at_submission(authored_cohort, tmp_path):
    """Refused where the caller can see it, not minutes later in the worker as a failed run."""
    import pytest

    with pytest.raises(ValueError, match="chart theme must be one of"):
        authored_cohort.experiment.run_export(
            "analysis_run", "an_01", ["instances"], tmp_path / "out", theme="sepia"
        )


def test_every_figure_target_reads_the_theme_the_same_way():
    """One reading of the option, so a dark export is dark in every figure it writes."""
    assert export_targets._figure_style({}).theme == "light"
    assert export_targets._figure_style({"theme": "dark"}).theme == "dark"
    assert export_targets._figure_style({"theme": None}).theme == "light"
