"""The measures CLI (§23): the numbers without the group contrast on top of them."""

import csv

from click.testing import CliRunner

from detail.cli import main
from tests.api.test_comparison import UNBALANCED, _cohort


def _run(*args):
    return CliRunner().invoke(main, list(args))


def test_list_names_every_measure_with_its_unit():
    """`--measure` takes a name from a registry documented nowhere else a user looks, so the
    flag would otherwise ship with undiscoverable values. The unit is half the point: the column
    mixes nanoseconds, counts, ratios and per-second rates."""
    result = _run("measures", "list")

    assert result.exit_code == 0, result.output
    assert "n_instances" in result.output
    assert "1/s" in result.output, "a rate says it is per second"
    assert "ns" in result.output, "a duration says it is nanoseconds"


def test_export_writes_the_tables(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run(
        "measures", "export",
        "--experiment", str(exp.dir),
        "--run", "an_01",
        "--out", str(tmp_path / "out"),
        "--measure", "n_instances",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    with open(out / "measures.csv", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["recording_id"] for row in rows} == set(UNBALANCED)
    assert (out / "frames.csv").is_file()
    assert (out / "measures.yaml").is_file()


def test_a_sliding_window_takes_its_length_and_step(tmp_path):
    """The options that did not exist: `--windows sliding` used to silently take the model's
    defaults, with no way to say otherwise from the command line."""
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run(
        "measures", "export",
        "--experiment", str(exp.dir),
        "--run", "an_01",
        "--out", str(tmp_path / "out"),
        "--measure", "n_instances",
        "--windows", "sliding",
        "--length", "0.4",
        "--step", "0.2",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    with open(tmp_path / "out" / "measures.csv", encoding="utf-8-sig") as handle:
        windows = {row["window_id"] for row in csv.DictReader(handle)}
    # One second of recording, a 0.4 s window every 0.2 s: more than the one `whole` window,
    # and a count that tracks the options rather than the 300 s/60 s defaults (which would
    # yield exactly one).
    assert (
        len(windows) > 1
    ), f"the sliding options did not reach the window model: {windows}"


def test_an_unmeasurable_run_says_why(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run(
        "measures", "export",
        "--experiment", str(exp.dir),
        "--run", "an_77",
        "--out", str(tmp_path / "out"),
    )  # fmt: skip

    assert result.exit_code != 0
    assert "no behavior output" in result.output


def test_export_drives_a_progress_bar(monkeypatch, tmp_path):
    """A silent long op is indistinguishable from a hang, and measuring reads a frame view per
    unit. Asserts the callback is wired and survives a stage change (the test_cli.py idiom).
    """
    exp = _cohort(tmp_path, UNBALANCED)
    seen = []
    import detail.api.measures as api

    real = api.Measures.export

    def spy(self, out_dir, *, windows=None, measures=None, progress=None):
        assert progress is not None, "the CLI must pass a progress callback"
        return real(
            self,
            out_dir,
            windows=windows,
            measures=measures,
            progress=lambda s, f, d="": (seen.append(s), progress(s, f, d))[1],
        )

    monkeypatch.setattr(api.Measures, "export", spy)
    result = _run(
        "measures", "export",
        "--experiment", str(exp.dir),
        "--run", "an_01",
        "--out", str(tmp_path / "out"),
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert len(set(seen)) > 1, f"the bar never changed stage: {seen}"
