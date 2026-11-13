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


# --------------------------------------------------------------------------- #
# the design flags reach the config, and the design is printed (§24 *Designs*)
# --------------------------------------------------------------------------- #


def test_level_must_be_passed_twice(tmp_path):
    from tests.api.test_comparison import CROSSOVER

    exp = _cohort(tmp_path, CROSSOVER)
    result = _run(
        "comparison", "run", "--experiment", str(exp.dir), "--source-run", "an_01",
        "--group-by", "drug", "--aggregate-to", "animal", "--level", "veh",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "exactly twice" in result.output


def test_every_design_flag_reaches_the_config_and_the_design_is_printed(
    monkeypatch, tmp_path
):
    """Surface parity: what the API takes, the CLI passes — by name, every one of them."""
    from tests.api.test_comparison import CROSSOVER

    import detail.api.comparison as api

    exp = _cohort(tmp_path, CROSSOVER)
    captured = {}
    real = api.run_comparison

    def spy(experiment, config, *, progress=None):
        captured.update(config)
        return real(experiment, config, progress=progress)

    monkeypatch.setattr(api, "run_comparison", spy)
    result = _run(
        "comparison", "run", "--experiment", str(exp.dir), "--source-run", "an_01",
        "--group-by", "drug", "--aggregate-to", "animal",
        "--level", "veh", "--level", "fent", "--measure", "n_instances",
        "--measure", "total_time", "--presence-s", "2", "--pseudo-count-s", "0.5",
        "--min-units-fraction", "0.75", "--no-suppress-nesting",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert captured["source"] == "labels"
    assert captured["levels"] == ["veh", "fent"]
    assert captured["presence_s"] == 2.0
    assert captured["pseudo_count_s"] == 0.5
    assert captured["min_units_fraction"] == 0.75
    assert captured["suppress_nesting"] is False
    assert "design: paired" in result.output
    assert "fent against veh" in result.output
    assert "undefined below 3 present unit(s) of 4" in result.output


def test_a_refused_design_is_a_message_not_a_traceback(tmp_path):
    from tests.api.test_comparison import THREE_SESSIONS

    exp = _cohort(tmp_path, THREE_SESSIONS)
    result = _run(
        "comparison", "run", "--experiment", str(exp.dir), "--source-run", "an_01",
        "--group-by", "drug", "--aggregate-to", "animal", "--measure", "n_instances",
        "--level", "veh", "--level", "nope",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "not a value of 'drug'" in result.output
    assert "Traceback" not in result.output


def test_three_values_run_as_a_profile_and_the_design_is_printed(tmp_path):
    """No levels over three sessions is a profile over all of them, not a refusal."""
    from tests.api.test_comparison import THREE_SESSIONS

    exp = _cohort(tmp_path, THREE_SESSIONS)
    result = _run(
        "comparison", "run", "--experiment", str(exp.dir), "--source-run", "an_01",
        "--group-by", "drug", "--aggregate-to", "animal", "--measure", "n_instances",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "design: profile" in result.output
    assert "fent, nalox, veh" in result.output


def test_list_shows_the_design(tmp_path):
    from tests.api.test_comparison import CROSSOVER

    exp = _cohort(tmp_path, CROSSOVER)
    _run(
        "comparison", "run", "--experiment", str(exp.dir), "--source-run", "an_01",
        "--group-by", "drug", "--aggregate-to", "animal", "--measure", "n_instances",
    )  # fmt: skip
    result = _run("comparison", "list", "--experiment", str(exp.dir))
    assert "design=paired" in result.output


# --- --from-screen: the label records the screen; the comparison says what it scored ------


def _quiet_projection(monkeypatch):
    from detail.api.temporal_clustering import TemporalClusteringRun

    monkeypatch.setattr(TemporalClusteringRun, "_project", lambda self, **_: None)


def _branch(run) -> int:
    tree = run._tree()
    return int(tree.children[tree.n_nodes - 1 - tree.n_leaves][0])


def test_from_screen_records_the_origin_the_screen_gives(authored_cohort, monkeypatch):
    import detail.api.comparison as comparison_api
    from detail.storage import run_labels

    _quiet_projection(monkeypatch)
    run = authored_cohort.run()
    node = _branch(run)
    asked = []

    def fake_origins(experiment, comparison_id, wanted, *, ancestors_of):
        asked.append((comparison_id, list(wanted), [ancestors_of(n) for n in wanted]))
        return {
            n: run_labels.LabelOrigin(comparison_id, str(n), "kept", "drug", "x · y")
            for n in wanted
        }

    monkeypatch.setattr(comparison_api, "label_origins", fake_origins)
    result = _run(
        "analysis", "temporal-interpret", "--experiment", str(authored_cohort.experiment.dir),
        "--run", "an_01", "--name-node", f"{node}=climb", "--from-screen", "cmp_01",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "labelled 1 node(s), from screen cmp_01" in result.output
    assert asked == [("cmp_01", [node], [run.ancestors_of(node)])], "one ranking"
    assert run.state().label_origins[str(node)]["comparison"] == "cmp_01"


def test_from_screen_refuses_a_node_the_screen_did_not_point_at(
    authored_cohort, monkeypatch
):
    import detail.api.comparison as comparison_api

    _quiet_projection(monkeypatch)
    run = authored_cohort.run()
    node = _branch(run)
    monkeypatch.setattr(
        comparison_api, "label_origins", lambda _e, _c, nodes, **k: dict.fromkeys(nodes)
    )
    monkeypatch.setattr(comparison_api.Comparison, "kept_nodes", lambda self: [3, 5])
    result = _run(
        "analysis", "temporal-interpret", "--experiment", str(authored_cohort.experiment.dir),
        "--run", "an_01", "--name-node", f"{node}=climb", "--from-screen", "cmp_01",
    )  # fmt: skip
    assert result.exit_code != 0
    assert f"node {node} is not one screen cmp_01 kept" in result.output
    assert "it kept 3, 5" in result.output
    assert not run.state().node_labels, "refused before anything was written"


def test_from_screen_without_name_node_is_a_usage_error(authored_cohort):
    result = _run(
        "analysis", "temporal-interpret", "--experiment", str(authored_cohort.experiment.dir),
        "--run", "an_01", "--from-screen", "cmp_01",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "--from-screen records where --name-node labels came from" in result.output


def test_the_tree_marks_a_label_a_screen_chose(fitted_codebook):
    """``--tree`` needs the node cache, which the state-only cohort has none of: a real fit."""
    from detail.storage import run_labels

    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    node = int(view.children(view.root)[0])
    run.name_nodes(
        {node: "climb"},
        origins={node: run_labels.LabelOrigin("cmp_02", str(node), "kept", "drug")},
    )
    result = _run(
        "analysis", "temporal-interpret", "--experiment", str(experiment.dir),
        "--run", run_id, "--tree",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "climb [from cmp_02]" in result.output
    assert "[from cmp_NN] a label a difference screen chose" in result.output


def test_a_labels_comparison_prints_what_it_is_not_independent_of(
    authored_cohort, monkeypatch
):
    from detail.storage import metadata as md
    from detail.storage import run_labels

    _quiet_projection(monkeypatch)
    exp = authored_cohort.experiment
    md.upsert_row(exp.dir, "r1", {"animal": "A", "drug": "veh"})
    md.upsert_row(exp.dir, "r2", {"animal": "B", "drug": "fent"})
    run = authored_cohort.run()
    node = _branch(run)
    run.name_nodes(
        {node: "climb"},
        origins={node: run_labels.LabelOrigin("cmp_02", str(node), "kept", "drug")},
    )
    result = _run(
        "comparison", "run", "--experiment", str(exp.dir), "--source-run", "an_01",
        "--group-by", "drug", "--aggregate-to", "none",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (
        "not independent: climb (cmp_02) was chosen from a difference screen over drug"
        in (result.output)
    )
    listed = _run("comparison", "list", "--experiment", str(exp.dir))
    assert "screened=climb" in listed.output


def test_run_says_which_recordings_were_left_out(tmp_path):
    from tests.api.test_comparison import DOSES

    exp = _cohort(tmp_path, {**DOSES, "r0x": ({"animal": "A0", "dose": ""}, 3)})
    result = _run(
        "comparison", "run", "--experiment", str(exp.dir), "--source-run", "an_01",
        "--group-by", "dose", "--aggregate-to", "animal",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "design: trend" in result.output
    assert "left out" in result.output and "r0x" in result.output
