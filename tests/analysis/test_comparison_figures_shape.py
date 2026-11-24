"""One figure per measurement, and every figure says what one point is.

**The blow-up.** `_by_measure` bucketed on the whole `measure_key`, whose last element is
`window_id` — so the *window* was part of a figure's identity and a sliding-window comparison
wrote one two-point dot plot per window. Measured on a realistic 30-minute recording with a
5-minute window stepped by a minute: 30 windows × 9 applicable measures × 2 behaviors = **540
figures**, each showing two dots, where §24 asks for *"dot plots, time-resolved curves, the
difference ranking with its floor"*. The curve was not missing because it was hard; it was
missing because the data for it had been shattered across hundreds of files.

Nothing failed, which is why it survived: every figure was individually correct.

**The caption.** §24: *"The path is printed on every output — every table, every figure caption,
`summary.md`, and `stats.csv` as a column. **A plot that does not say what one point is cannot be
read.**"* It reached `stats.csv` and `summary.md`; no figure carried it. So every chart this tool
has ever exported omitted the line saying whether a dot is a recording or an animal — which is the
difference between two different answers to the same question, and §24's own reason for demanding
it. The requirement named a guard ("a source-scan-plus-render check"); there wasn't one.
"""

import csv
import pathlib

import pytest

from detail.analysis import comparison_report


def _run(exp, **overrides):
    config = {
        "source_run_id": "an_01",
        "group_by": ["drug"],
        "aggregate_to": [],
        "measures": ["n_instances"],
    }
    config.update(overrides)
    exp.run_comparison(config)
    return exp.comparisons[-1]


def _figures(comparison, tmp_path):
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))
    return sorted(p.name for p in (out / "figures").glob("*.png")), out


# --------------------------------------------------------------------------- #
# a window is not a figure
# --------------------------------------------------------------------------- #
def test_many_windows_make_one_curve_not_one_figure_each(tmp_path):
    """The number a user counts in their export folder."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(
        exp, windows={"kind": "normalized", "parts": 6}, measures=["n_instances"]
    )

    names, _out = _figures(comparison, tmp_path)

    per_measure = [n for n in names if n.startswith("n_instances")]
    assert per_measure == [
        "n_instances__rearing.png"
    ], f"six windows produced {len(per_measure)} figures for one measurement: {per_measure}"


def test_a_single_window_still_draws_the_dot_plot(tmp_path):
    """The negative case: collapsing the window must not turn a `whole`-window comparison — the
    common one — into a one-point line."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    names, _out = _figures(_run(exp), tmp_path)

    assert "n_instances__rearing.png" in names


def test_the_figure_stem_no_longer_carries_the_window(tmp_path):
    """A figure that spans every window cannot be named after one of them. The stem names the
    *measurement*; `stats.csv` and the table CSVs still carry the window as a column."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    names, _out = _figures(
        _run(exp, windows={"kind": "normalized", "parts": 3}), tmp_path
    )

    assert not any("part_0" in name for name in names), names


def test_the_windows_survive_in_the_data_even_though_the_figures_collapse(tmp_path):
    """Collapsing is a *rendering* decision. The long-form export keeps every window, because
    that is what someone re-plotting in R needs."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp, windows={"kind": "normalized", "parts": 3})

    with open(comparison.dir / "stats.csv", newline="", encoding="utf-8-sig") as handle:
        windows = {row["window_id"] for row in csv.DictReader(handle)}

    assert windows == {"part_00", "part_01", "part_02"}


# --------------------------------------------------------------------------- #
# every figure says what one point is
# --------------------------------------------------------------------------- #
def test_the_note_says_what_one_observation_is():
    assert (
        comparison_report.observation_note({"aggregate_to": ["animal"]})
        == "one observation is: animal"
    )
    assert "one per recording" in comparison_report.observation_note({})


def _texts_on(fig) -> list[str]:
    """Every free-standing text a figure carries — on the figure itself and on its axes.

    Both, because the caption moved from ``fig.text`` to an annotation on the axes (so constrained
    layout makes room for it) and a check reading only one of the two would have gone blind the
    day it moved.
    """
    return [t.get_text() for t in fig.texts] + [
        t.get_text() for ax in fig.axes for t in ax.texts
    ]


@pytest.mark.parametrize("aggregate_to", ([], ["animal"]))
def test_every_rendered_figure_carries_the_aggregation_path(tmp_path, aggregate_to):
    """Read off the **figure**, not off the call.

    The first version of this spied on `_caption` being invoked, and passed with the function's
    body replaced by `return` — proving only that a call site existed. §24 asks that the output
    carry the path, so this inspects the figure's texts at the moment each figure is saved.
    """
    from detail.report import figures as figlib
    from tests.api.test_comparison import _cohort

    # **A crossover, not `UNBALANCED`.** `UNBALANCED` is one group, so it drew neither a paired
    # figure nor a ranking — and this test passed with `_caption` deleted from both of those
    # functions, having only ever seen the dot plots. A fixture that does not reach a code path
    # cannot guard it.
    exp = _cohort(tmp_path, CROSSOVER)
    comparison = _run(exp, aggregate_to=aggregate_to, measures=["n_instances", "rate"])

    seen: list[list[str]] = []
    original = figlib.save_figure

    def spy(fig, out_stem, formats=("pdf", "svg", "png"), **kwargs):
        seen.append(_texts_on(fig))
        return original(fig, out_stem, formats, **kwargs)

    figlib.save_figure = spy
    try:
        names, _out = _figures(comparison, tmp_path)
    finally:
        figlib.save_figure = original

    expected = comparison_report.observation_note({"aggregate_to": aggregate_to})
    assert len(seen) == len(names)
    missing = [i for i, texts in enumerate(seen) if expected not in texts]
    assert not missing, (
        f"{len(missing)} of {len(seen)} figures were saved without the aggregation path "
        f"on them; §24 requires it on every one"
    )
    # And every *kind* of figure was actually drawn, or "every one" is a claim about a subset:
    # the per-measurement panels, the composite of them, and the ranking.
    assert any("difference_ranking" in n for n in names), names
    assert "panels.png" in names, names


def test_a_figure_saved_without_the_note_would_be_caught(tmp_path):
    """Guard the guard: the check above must be capable of failing, so prove a bare figure has
    no such text on it."""
    from detail.report import figures as figlib

    with figlib.styled():
        import matplotlib.pyplot as plt

        fig, _ax = plt.subplots()
        assert _texts_on(fig) == []
        comparison_report._caption(
            _ax, "one observation is: animal", figlib.DEFAULT_STYLE
        )
        assert _texts_on(fig) == ["one observation is: animal"]
        plt.close(fig)


# --------------------------------------------------------------------------- #
# a crossover is drawn as one
# --------------------------------------------------------------------------- #
CROSSOVER = {
    # One animal measured under both treatments — a single-animal crossover's shape.
    "r1": ({"animal": "F3", "drug": "veh"}, 9),
    "r2": ({"animal": "F3", "drug": "fent"}, 4),
    "r3": ({"animal": "F4", "drug": "veh"}, 8),
    "r4": ({"animal": "F4", "drug": "fent"}, 3),
}


def _joins_drawn(comparison, tmp_path) -> list[list[str]]:
    """The unit-joining lines each measurement figure drew: one list of unit labels per call
    of the bar primitive. A join is a line labelled with its unit; an error bar's cap is not.
    """
    from detail.report import figures as figlib

    drawn: list[list[str]] = []
    original = figlib.bars_with_units

    def spy(ax, group_labels, observations, **kwargs):
        units = original(ax, group_labels, observations, **kwargs)
        drawn.append(
            [line.get_label() for line in ax.lines if line.get_label() in units]
        )
        return units

    figlib.bars_with_units = spy
    try:
        _figures(comparison, tmp_path)
    finally:
        figlib.bars_with_units = original
    return drawn


def test_a_subject_in_both_groups_gets_a_line(tmp_path):
    """Two independent clouds of dots throw away the pairing that makes a crossover powerful:
    a consistent within-animal shift can read as heavy overlap between groups. The line is
    drawn in the measurement's own figure, not in a second one beside it (detail#76)."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, CROSSOVER)
    comparison = _run(exp, aggregate_to=["animal"])

    joins = _joins_drawn(comparison, tmp_path)

    assert joins and all(sorted(j) == ["F3", "F4"] for j in joins), joins
    names, _out = _figures(comparison, tmp_path)
    assert not any("paired" in name for name in names), names


def test_no_pairing_means_no_lines(tmp_path):
    """It fires on the structure, not on a setting. With one observation per recording, each
    point belongs to exactly one group and there is nothing to join."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, CROSSOVER)
    joins = _joins_drawn(_run(exp, aggregate_to=[]), tmp_path)

    assert joins and all(j == [] for j in joins), joins


def test_the_lines_are_built_from_observations_not_summaries(tmp_path):
    """`compare.summarize` buckets by group and sorts the values, dropping `Observation.key` —
    so the summaries cannot say which point is which animal, which is the whole content of a
    pairing. Asserted by the unit each dot is drawn for, which only the keyed data can give.
    """
    from detail.report import figures as figlib
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, CROSSOVER)
    comparison = _run(exp, aggregate_to=["animal"])

    drawn: list[list] = []
    original = figlib.bars_with_units

    def spy(ax, group_labels, observations, **kwargs):
        drawn.append(observations)
        return original(ax, group_labels, observations, **kwargs)

    figlib.bars_with_units = spy
    try:
        _figures(comparison, tmp_path)
    finally:
        figlib.bars_with_units = original

    assert drawn, "no bars were drawn"
    subjects = {unit for groups in drawn for group in groups for unit, _ in group}
    assert subjects == {"F3", "F4"}, subjects


# --------------------------------------------------------------------------- #
# a window is a window, wherever it is read
# --------------------------------------------------------------------------- #
def test_the_curve_runs_in_window_order_not_alphabetical(tmp_path):
    """`abs_9, abs_10, abs_100, abs_11` — the padding is a minimum width, so past the hundredth
    window plain string order interleaves the three-digit ids among the two-digit ones. A
    30-minute recording windowed every 10 seconds is 180 windows, so this is a setting a user
    reaches, not a corner. Asserted on the x-axis the curve is drawn against."""
    from detail.report import figures as figlib
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp, windows={"kind": "normalized", "parts": 120})

    drawn: list[list[str]] = []
    original = figlib.group_curves

    def spy(ax, x_labels, series, **kwargs):
        drawn.append(list(x_labels))
        return original(ax, x_labels, series, **kwargs)

    figlib.group_curves = spy
    try:
        _figures(comparison, tmp_path)
    finally:
        figlib.group_curves = original

    assert drawn, "no curve was drawn over 120 windows"
    axis = drawn[0]
    # `part_{i:02d}` pads to a *minimum* of two, so past 99 the ids get a third digit and
    # string order interleaves them: part_09, part_10, part_100, part_101, ..., part_11.
    assert len(axis) == 120 and axis[:2] == ["part_00", "part_01"], axis[:4]
    assert axis == sorted(axis, key=lambda w: int(w.rsplit("_", 1)[1])), axis[:12]


def test_a_pairing_is_not_drawn_from_one_window_labelled_as_all_of_them(tmp_path):
    """Every window wrote into the same (subject, group) slot and the last one won, so a paired
    figure over a windowed comparison showed the final window's numbers under a caption that
    named the whole measurement. Deliberately not drawn at all now — the curve carries the
    time-resolved reading, and one window's lines are not the measurement's."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, CROSSOVER)
    comparison = _run(
        exp, aggregate_to=["animal"], windows={"kind": "normalized", "parts": 3}
    )

    joins = _joins_drawn(comparison, tmp_path)

    assert joins == [], "a windowed measurement drew bars of one window as the whole"


def test_one_window_still_pairs(tmp_path):
    """The negative case: the ordinary whole-recording crossover keeps its lines."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, CROSSOVER)
    joins = _joins_drawn(_run(exp, aggregate_to=["animal"]), tmp_path)

    assert joins and all(sorted(j) == ["F3", "F4"] for j in joins), joins


# --------------------------------------------------------------------------- #
# a label is not a path
# --------------------------------------------------------------------------- #
def test_a_behavior_name_with_a_slash_does_not_become_a_directory(tmp_path):
    """A stem is built from the behavior's own name, and a name is a user's free text. `rear/sit`
    aborted the whole export with `FileNotFoundError` (there is no `figures/rear` directory), and
    `rear.sit` was worse: `.with_suffix('.csv')` cut it back to `n_instances__rear.csv`, so two
    different measurements silently wrote to one file."""
    from tests.api.test_comparison import _cohort

    # Few bouts: two behaviors have to fit in the fixture's frame axis, and the count is not
    # what is under test here.
    plan = {
        "r1": ({"animal": "A", "drug": "veh"}, 2),
        "r2": ({"animal": "B", "drug": "fent"}, 3),
    }
    exp = _cohort(tmp_path, plan, label_names=("rear/sit", "rear.sit"))
    comparison = _run(exp, measures=["n_instances"])
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))

    tables = sorted(p.name for p in (out / "tables").glob("n_instances__*.csv"))
    assert len(tables) == 2, f"two behaviors collapsed into {tables}"
    assert not any(p.is_dir() for p in (out / "figures").iterdir())


# --------------------------------------------------------------------------- #
# tables a manuscript can use
# --------------------------------------------------------------------------- #
def test_the_contrast_is_written_as_latex_markdown_and_csv(tmp_path):
    """§24 lists `tables.py`'s "LaTeX/Markdown/CSV" among its outputs; the renderers shipped,
    both benchmark exporters used them, and a comparison called only `write_tidy_csv`.
    """
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp)
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))

    tex = (out / "tables" / "comparison.tex").read_text(encoding="utf-8-sig")
    assert "\\begin{tabular}" in tex and "\\toprule" in tex
    # §24: the aggregation path reaches every output, a table included.
    assert "One observation is" in tex
    assert (out / "tables" / "comparison.md").is_file()
    assert (out / "tables" / "comparison.csv").is_file()


def test_the_latex_table_is_in_readable_units(tmp_path):
    """A duration in a manuscript reads in seconds. Nanoseconds would arrive as 4956702127.66.

    **Assert the number, not the unit label.** The first version checked that `(s)` appeared and
    that `e+09` did not — and both stay true when `_scale_for` returns 1.0, because the label
    comes from the measure and LaTeX renders 4956702127.66 without an exponent. It passed against
    its own subject reverted. The value is the thing the scaling changes, so this reads it.
    """
    import re

    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp, measures=["total_time"])
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))

    tex = (out / "tables" / "comparison.tex").read_text(encoding="utf-8-sig")
    assert "(s)" in tex
    body = tex.split("\\midrule", 1)[1]
    numbers = [float(n) for n in re.findall(r"\d+\.\d+", body)]

    assert numbers, f"no numbers in the table body: {body}"
    # The fixture's bouts are one frame-period long, so every total is a handful of seconds --
    # and nine orders of magnitude away from the nanoseconds they are stored in.
    assert max(numbers) < 1e3, f"raw nanoseconds reached the LaTeX table: {numbers}"


# --------------------------------------------------------------------------- #
# a re-export replaces rather than accumulates
# --------------------------------------------------------------------------- #
# a re-export replaces what it wrote, and nothing else
# --------------------------------------------------------------------------- #
def _manifest(out):
    import json

    return json.loads((out / "manifest.json").read_text(encoding="utf-8-sig"))


def _pretend_the_last_export_wrote(out, relative: str, body: bytes) -> pathlib.Path:
    """A file left by a previous export, recorded in its manifest — which is what makes it ours
    to delete. Writing the file alone would be somebody else's file that happens to look like
    one of ours, and the difference is the whole point."""
    import json

    path = out / relative
    path.write_bytes(body)
    manifest = _manifest(out)
    manifest["artifacts"][relative] = "sha-of-an-older-version"
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def test_a_re_export_drops_what_the_last_one_wrote(tmp_path):
    """The bug the window collapse introduced, and the reason a user said the page "looks the
    same": a figure is named after the measurement it shows, that name changed, and the old file
    stayed. The detail page globs the directory, so it went on showing twenty figures drawn by
    code that no longer existed."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp)
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))

    stale = _pretend_the_last_export_wrote(
        out, "figures/n_instances__rearing__whole.png", b"from an older version"
    )
    comparison.export(out, formats=("png",))

    assert not stale.exists(), "a figure from a previous export survived a re-render"
    assert (out / "figures" / "n_instances__rearing.png").is_file()


def test_a_re_export_leaves_a_users_own_files_alone(tmp_path):
    """Why this is manifest-driven and not a sweep by suffix.

    The suffix sweep deleted every `.png`, `.csv`, `.tex` and `.md` in `tables/` and `figures/`
    whoever wrote them — so exporting into a folder someone had chosen destroyed their work, and
    a `.png` of their own was indistinguishable from one of ours. A manifest is the only record
    of authorship there is, so a file missing from it is not ours to remove.
    """
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp)
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))

    mine = {
        out / "figures" / "notes.txt": b"mine",
        # The ones the suffix sweep took: our own extensions, somebody else's files.
        out / "figures" / "panel_for_fig3.png": b"a figure I made",
        out / "tables" / "hand_counts.csv": b"a,b\n1,2\n",
        out / "tables" / "draft.tex": b"\\section{mine}",
        out / "tables" / "notes.md": b"# mine",
    }
    for path, body in mine.items():
        path.write_bytes(body)

    comparison.export(out, formats=("png",))

    survived = {
        path: path.is_file() and path.read_bytes() == body
        for path, body in mine.items()
    }
    assert all(
        survived.values()
    ), f"the export deleted files it did not write: {survived}"


def test_a_first_export_into_a_full_directory_deletes_nothing(tmp_path):
    """No manifest means no claim of authorship, so nothing in the directory is ours."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp)
    out = tmp_path / "export"
    (out / "figures").mkdir(parents=True)
    collides = out / "figures" / "n_instances__rearing.png"
    collides.write_bytes(b"same name, not ours")
    untouched = out / "figures" / "their_analysis.png"
    untouched.write_bytes(b"nothing to do with this comparison")

    comparison.export(out, formats=("png",))

    # Overwritten by the figure of the same name -- but not *deleted* before the render, which
    # is the difference between replacing a file and emptying a directory.
    assert collides.is_file() and collides.read_bytes() != b"same name, not ours"
    assert untouched.read_bytes() == b"nothing to do with this comparison"


def test_a_failed_export_leaves_the_last_complete_one_standing(tmp_path):
    """Deletion runs after the render, not before it. Clearing first meant a run that raised
    part-way through left an empty directory where a complete older export had been."""
    from detail.analysis import comparison_report
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp)
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))
    before = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert before

    original = comparison_report._measure_figures

    def boom(*args, **kwargs):
        raise RuntimeError("the renderer fell over")

    comparison_report._measure_figures = boom
    try:
        with pytest.raises(RuntimeError):
            comparison.export(out, formats=("png",))
    finally:
        comparison_report._measure_figures = original

    assert sorted(p.name for p in (out / "figures").glob("*.png")) == before


# --------------------------------------------------------------------------- #
# a determinate bar that moves
# --------------------------------------------------------------------------- #
def test_the_export_reports_a_fraction_that_moves_within_the_figures(tmp_path):
    """Both surfaces offered a *determinate* bar for this and nothing ever drove it, so it sat
    at 0 for the whole render and jumped to done — which reads as a hang more loudly than no bar
    would. It is one matplotlib figure per measurement, minutes on a large comparison.

    Asserted on the **inner** pair with more than one figure to draw: a check that the fraction
    merely reaches 1.0 passes on a callback that only fires at the end.
    """
    from tests.api.test_comparison import _cohort

    # Two behaviors x three measures = six figures, so the fraction has somewhere to be.
    plan = {
        "r1": ({"animal": "A", "drug": "veh"}, 3),
        "r2": ({"animal": "B", "drug": "fent"}, 5),
    }
    exp = _cohort(tmp_path, plan, label_names=("rearing", "grooming"))
    comparison = _run(exp, measures=["n_instances", "rate", "total_time"])

    seen: list[tuple[str, float, str]] = []
    comparison.export(
        tmp_path / "export", formats=("png",), progress=lambda *e: seen.append(e)
    )

    figures = [f for stage, f, _d in seen if stage == "figures"]
    assert [
        f for f in figures if 0.0 < f < 1.0
    ], f"the figure bar never moved between its ends: {figures}"


def test_the_export_names_each_phase_as_it_reaches_it():
    """The fraction moving is not enough — a label frozen on one word through a multi-phase op
    reads as hung. The phases are distinct and in order."""
    import inspect

    from detail.analysis import comparison_report

    source = inspect.getsource(comparison_report.export_comparison)
    for stage in ("tables", "figures", "summary"):
        assert f'report("{stage}"' in source, stage


def test_the_export_phases_arrive_in_order(tmp_path):
    """Read off a real run rather than the source."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    comparison = _run(exp)

    stages: list[str] = []
    comparison.export(
        tmp_path / "export",
        formats=("png",),
        progress=lambda stage, _f, _d: stages.append(stage),
    )

    first_seen = list(dict.fromkeys(stages))
    assert first_seen[0] == "tables" and first_seen[-1] == "summary", first_seen
    assert "figures" in first_seen


# --------------------------------------------------------------------------- #
# a matrix is not a series of two-point plots
# --------------------------------------------------------------------------- #
def test_a_pairwise_measure_is_tabulated_rather_than_drawn(tmp_path):
    """`transitions` answers with one value per **ordered pair**, so a vocabulary of N behaviors
    is up to N x N figures — 400 for a 20-syllable clusterer, each a two-point dot plot of one
    cell. That is the blow-up collapsing the window out of a figure's identity just cured,
    arriving from the other direction.

    Not hidden — *tabulated*. The pairs are in `stats.csv`, in the per-measure tables and in the
    Measures tab, one row each. What a transition matrix should look like is a design question,
    and 400 wrong answers is not a way of answering it.
    """
    import csv

    from tests.api.test_comparison import _cohort

    plan = {
        "r1": ({"animal": "A", "drug": "veh"}, 3),
        "r2": ({"animal": "B", "drug": "fent"}, 4),
    }
    exp = _cohort(tmp_path, plan, label_names=("rearing", "grooming"))
    comparison = _run(exp, measures=["transitions", "n_instances"])
    names, out = _figures(comparison, tmp_path)

    assert not any(n.startswith("transitions") for n in names), names
    # The negative case, twice over: an ordinary measure is still drawn, and the transitions
    # themselves are still *reported* -- with their from-label, one row per ordered pair.
    assert any(n.startswith("n_instances") for n in names), names
    with open(comparison.dir / "stats.csv", newline="", encoding="utf-8-sig") as handle:
        pairs = {
            (row["label"], row["label_to"])
            for row in csv.DictReader(handle)
            if row["measure"] == "transitions"
        }
    assert pairs and all(a and b for a, b in pairs), pairs
    assert (out / "tables" / "comparison.csv").is_file()
