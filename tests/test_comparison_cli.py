"""The comparison CLI makes the unit of analysis impossible to leave unsaid (§24)."""

from click.testing import CliRunner

from detail.cli import main
from tests.api.test_comparison import UNBALANCED, _cohort


def _run(*args):
    return CliRunner().invoke(main, list(args))


def test_run_without_aggregate_to_is_refused_with_the_reason(tmp_path):
    """The one option with no default. A silent choice here is wrong about half the time and the
    output looks identical either way, so the CLI explains rather than picking."""
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
    )
    assert result.exit_code != 0
    assert "--aggregate-to is required" in result.output
    assert "different questions" in result.output


def test_none_is_how_you_say_do_not_aggregate(tmp_path):
    """``--aggregate-to none`` is a stated choice; an omitted flag is not the same thing."""
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
        "--aggregate-to",
        "none",
        "--measure",
        "n_instances",
    )
    assert result.exit_code == 0, result.output
    assert "one observation is: (one per recording)" in result.output


def test_run_by_animal_reports_the_unit(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
        "--aggregate-to",
        "animal",
        "--measure",
        "n_instances",
    )
    assert result.exit_code == 0, result.output
    assert "one observation is: animal" in result.output


def test_list_shows_the_unit_of_analysis(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
        "--aggregate-to",
        "animal",
        "--measure",
        "n_instances",
    )
    result = _run("comparison", "list", "--experiment", str(exp.dir))
    assert result.exit_code == 0, result.output
    assert "cmp_01" in result.output and "by=animal" in result.output


def test_list_is_explicit_when_empty(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run("comparison", "list", "--experiment", str(exp.dir))
    assert "(no comparisons)" in result.output


def test_export_writes_the_tree(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
        "--aggregate-to",
        "none",
        "--measure",
        "n_instances",
    )
    result = _run(
        "comparison",
        "export",
        "cmp_01",
        "--experiment",
        str(exp.dir),
        "--out",
        str(tmp_path / "out"),
        "--format",
        "png",
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "summary.md").is_file()
    assert (tmp_path / "out" / "stats.csv").is_file()


def test_run_drives_a_progress_bar(monkeypatch, tmp_path):
    """A silent long op is indistinguishable from a hang; measuring reads a frame view per unit.
    Asserts the callback is wired and survives a stage change (the test_cli.py idiom).
    """
    exp = _cohort(tmp_path, UNBALANCED)
    seen = []
    import detail.api.comparison as api

    real = api.run_comparison

    def spy(experiment, config, *, progress=None):
        assert progress is not None, "the CLI must pass a progress callback"
        return real(
            experiment,
            config,
            progress=lambda s, f, d: (seen.append(s), progress(s, f, d))[1],
        )

    monkeypatch.setattr(api, "run_comparison", spy)
    result = _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
        "--aggregate-to",
        "none",
        "--measure",
        "n_instances",
    )
    assert result.exit_code == 0, result.output
    assert len(set(seen)) > 1, f"the bar never changed stage: {seen}"


def test_delete_removes_the_comparison_and_its_exports(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
        "--aggregate-to",
        "none",
        "--measure",
        "n_instances",
    )
    _run(
        "comparison",
        "export",
        "cmp_01",
        "--experiment",
        str(exp.dir),
        "--out",
        str(exp.dir / "comparisons" / "cmp_01" / "export"),
        "--format",
        "png",
    )
    result = _run(
        "comparison", "delete", "cmp_01", "--experiment", str(exp.dir), "--yes"
    )
    assert result.exit_code == 0, result.output
    assert not (exp.dir / "comparisons" / "cmp_01").exists()
    assert (
        _run("comparison", "list", "--experiment", str(exp.dir)).output.strip()
        == "(no comparisons)"
    )


def test_deleting_an_unknown_comparison_says_what_exists(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    result = _run(
        "comparison", "delete", "cmp_99", "--experiment", str(exp.dir), "--yes"
    )
    assert result.exit_code != 0
    assert "cmp_99" in result.output


def test_delete_asks_before_destroying_work(tmp_path):
    """A comparison is minutes of compute and may be cited; deleting it is not undoable."""
    exp = _cohort(tmp_path, UNBALANCED)
    _run(
        "comparison",
        "run",
        "--experiment",
        str(exp.dir),
        "--source-run",
        "an_01",
        "--group-by",
        "drug",
        "--aggregate-to",
        "none",
        "--measure",
        "n_instances",
    )
    result = CliRunner().invoke(
        main,
        ["comparison", "delete", "cmp_01", "--experiment", str(exp.dir)],
        input="n\n",
    )
    assert result.exit_code != 0
    assert (
        exp.dir / "comparisons" / "cmp_01"
    ).is_dir(), "declining the prompt still deleted it"
