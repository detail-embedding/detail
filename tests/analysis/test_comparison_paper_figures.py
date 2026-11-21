"""The comparison figures a paper takes, and the statistics exported beside them (detail#76).

**A figure that showed a difference the test did not find, or hid one it did.** The old figures
carried no P at all, so a reader took the gap between two medians for the finding. The P is on
the plot now, for exactly the items the ranking's verdict keeps, and nothing is written for the
rest: a bracket drawn for an item below the floor would be a claim the test did not make, and
a missing one over a kept item hides the finding the comparison exists to report.

**A P that did not say which P it was.** A cluster found by screening a thousand carries the
family-wise P; a measure chosen in advance carries its own test's. Written the same, the first
reads as the second. The symbol says which (``P_FWE`` / ``P``), the shuffled zero is written as
the bound it is (``< 1/draws``), and an ordered variable is named rather than called a "trend",
which a reader takes to mean "not quite".

**Groups in string order, coloured by position.** Doses 5, 10 and 25 drew as 10, 25, 5, and a
missing group repainted the others. The order follows the design and the colour the group.

**Numbers computed and thrown away.** The per-unit effects a within-unit ranking averaged, and
the means and s.e.m. the figure drew, existed only inside the render. They are written beside
the figures by the run itself, and the Methods paragraph is generated from the record, so what
a paper says ran is what ran.
"""

import csv
import inspect
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from detail.analysis import compare, comparison_report
from detail.api.comparison import Comparison, run_comparison
from detail.report import figures as figlib
from detail.report.style import DEFAULT_STYLE
from tests.api.test_comparison_locations import PAIRED, cohort, paired_plan

SVG = "{http://www.w3.org/2000/svg}"


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    """One real comparison, exported once: eight animals under saline and drug, judged by each
    measure's own exact test (256 sign flips)."""
    exp = cohort(tmp_path_factory.mktemp("paired"), paired_plan())
    comp = run_comparison(exp, {**PAIRED, "draws": 200})
    saved: dict[str, object] = {}
    original = figlib.save_figure

    def spy(fig, out_stem, formats=("pdf", "svg", "png"), **kwargs):
        saved[str(out_stem).rsplit("\\", 1)[-1].rsplit("/", 1)[-1]] = (
            [ax for ax in fig.axes if ax.get_visible()],
            [t.get_text() for t in fig.texts],
            len(fig.legends),
        )
        return original(fig, out_stem, formats, **kwargs)

    figlib.save_figure = spy
    try:
        comp.export(formats=("png", "svg"))
    finally:
        figlib.save_figure = original
    return comp, comp.ranking(), saved


# --------------------------------------------------------------------------- #
# the P on the plot
# --------------------------------------------------------------------------- #
def _bracket_texts(axes) -> list[str]:
    """The P labels drawn on ``axes``: the texts a bracket writes (math text, italic P)."""
    return [t.get_text() for ax in axes for t in ax.texts if "mathit" in t.get_text()]


def test_a_p_is_drawn_for_exactly_the_items_the_verdict_keeps(paired):
    comp, ranking, saved = paired
    kept = {item.measure_key[0] for item in ranking.kept_items()}
    not_kept = {item.measure_key[0] for item in ranking.contrasts} - kept
    assert kept and not_kept, "the fixture must have both a finding and a non-finding"
    for measure in kept:
        assert _bracket_texts(saved[measure][0]), f"{measure} cleared and has no P"
    for measure in not_kept:
        assert not _bracket_texts(
            saved[measure][0]
        ), f"{measure} did not clear: {measure}"


def test_the_symbol_says_which_p_and_the_bound_where_it_is_one():
    item = compare.Effect(
        measure_key=("distance", "", "", "whole"),
        effect=-0.5,
        n_units=8,
        n_present=8,
        consistency=8,
        loo_sign_stable=True,
        pair=("saline", "drug"),
        family_p=0.0,
        own_p=0.004,
        q=0.01,
    )
    family = compare.Ranking(
        [item], 0.3, 200, 0, design=compare.PAIRED, levels=("saline", "drug")
    )
    own = comparison_report.by_own_test(family)
    assert comparison_report.p_symbol(family) == "P_FWE"
    assert comparison_report.p_symbol(own) == "P"
    # a shuffled family-wise P of zero is below the test's resolution, not zero
    assert comparison_report.item_p(family, item) == (0.0, 1 / 200)
    assert comparison_report.item_p(own, item) == (0.004, None)
    record = {"design_column": "dose"}
    assert comparison_report._bracket_text(family, item, record).endswith("< 0.005")
    assert "FWE" in comparison_report._bracket_text(family, item, record)
    assert comparison_report._bracket_text(own, item, record).endswith("= 0.004")
    # an ordered variable is named, never "trend"
    trend = compare.Ranking([item], 0.3, 200, 0, design=compare.TREND)
    text = comparison_report._bracket_text(trend, item, record)
    assert text.startswith("dose: ") and "trend" not in text
    # exact: a P of zero cannot occur, and no bound is written
    exact = compare.Ranking([item], 0.3, 256, 0, design=compare.PAIRED, exact=True)
    assert comparison_report.item_p(exact, item) == (0.0, None)


@pytest.mark.parametrize(
    ("p", "text"),
    [
        (0.0456, "= 0.046"),  # two decimals would print the threshold itself
        (0.004, "= 0.004"),
        (0.23, "= 0.23"),
        (0.0004, "< 0.001"),
        (0.05, "= 0.05"),
    ],
)
def test_p_formatting(p, text):
    assert comparison_report.p_label(p).endswith(text)


def test_p_label_is_italic_math_text():
    assert comparison_report.p_label(0.5) == r"$\mathit{P}$ = 0.50"
    assert comparison_report.p_label(0.5, "P_FWE").startswith(
        r"$\mathit{P}_{\mathrm{FWE}}$"
    )


# --------------------------------------------------------------------------- #
# group order and colour
# --------------------------------------------------------------------------- #
def test_groups_draw_in_design_order_not_string_order():
    groups = {("5",), ("10",), ("25",), ("control",)}
    assert comparison_report.group_order({}, None, groups) == [
        ("5",),
        ("10",),
        ("25",),
        ("control",),
    ]
    named = comparison_report.group_order(
        {"levels": ["drug", "veh"]}, None, groups | {("drug",), ("veh",)}
    )
    assert named[:2] == [("drug",), ("veh",)], "the named reference comes first"


def test_colour_follows_the_group_and_a_missing_group_repaints_nothing():
    style = DEFAULT_STYLE
    ranking = compare.Ranking([], 0.3, 200, 0, design=compare.TREND)
    record = {"groups": ["0", "25", "50", "75"]}
    every = comparison_report._all_groups(record, ranking, [])
    assert every == [("0",), ("25",), ("50",), ("75",)]
    colors = comparison_report.group_colors(record, ranking, every, style)
    assert colors[("0",)] == style.chart.text_muted, "the reference is grey"
    ramp = [colors[g] for g in every[1:]]
    figlib.check_ordered_ramp(ramp)  # one hue, light to dark
    # the same map whether or not a group is present in the summaries drawn
    summaries = compare.summarize(
        [
            compare.Observation(("A",), (g,), ("distance", "", "", "whole"), 1.0, 1)
            for g in ("25", "50", "75")
        ]
    )
    assert comparison_report._all_groups(record, ranking, summaries) == every
    # two levels: grey and the hue; more named values: distinct hues, no grey
    two = comparison_report.group_colors({}, None, [("veh",), ("drug",)], style)
    assert list(two.values()) == [style.chart.text_muted, style.chart.hue("blue")]
    three = comparison_report.group_colors({}, None, [("a",), ("b",), ("c",)], style)
    assert (
        len(set(three.values())) == 3 and style.chart.text_muted not in three.values()
    )


def test_a_ramp_that_turns_back_is_refused():
    with pytest.raises(ValueError, match="monotone"):
        figlib.check_ordered_ramp(["#dddddd", "#222222", "#888888"])
    assert len(figlib.one_hue_ramp("#0072B2", 4)) == 4


# --------------------------------------------------------------------------- #
# the bars
# --------------------------------------------------------------------------- #
def test_every_unit_is_drawn_and_joined_only_where_it_recurs():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    units = figlib.bars_with_units(
        ax,
        ["a", "b"],
        [[("A1", 1.0), ("A2", 2.0), ("A3", 0.0)], [("A1", 3.0), ("A2", 1.0)]],
        colors=["#777777", "#0072B2"],
    )
    assert units == ["A1", "A2", "A3"]
    joins = [line for line in ax.lines if line.get_label() in units]
    assert sorted(line.get_label() for line in joins) == ["A1", "A2"]
    from matplotlib.collections import PathCollection

    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    dots = sum(len(c.get_offsets()) for c in scatters)
    assert dots == 5, "a unit went missing between the data and the canvas"
    # nothing is clipped: a dot at zero is a whole dot
    assert not any(c.get_clip_on() for c in scatters)
    assert not any(line.get_clip_on() for line in joins)
    # the same unit sits at the same offset in every bar
    xs = {line.get_label(): [x - round(x) for x in line.get_xdata()] for line in joins}
    for offsets in xs.values():
        assert len({round(o, 6) for o in offsets}) == 1
    plt.close(fig)


def test_bars_are_deterministic():
    import matplotlib.pyplot as plt

    runs = []
    for _ in range(2):
        fig, ax = plt.subplots()
        figlib.bars_with_units(
            ax, ["a"], [[("u1", 1.0), ("u2", 1.0)]], colors=["#0072B2"]
        )
        runs.append([tuple(map(tuple, c.get_offsets())) for c in ax.collections])
        plt.close(fig)
    assert runs[0] == runs[1]


def test_a_ratio_grows_from_one_and_a_count_from_zero(paired):
    import matplotlib.pyplot as plt

    comp, ranking, _saved = paired
    obs = comparison_report._observations(comparison_report._read_stats(comp.dir))
    record = comp.record()
    colors = comparison_report.group_colors(
        record,
        ranking,
        comparison_report._all_groups(record, ranking, compare.summarize(obs)),
        DEFAULT_STYLE,
    )
    for measure, baseline in (("body_elongation", 1.0), ("distance", 0.0)):
        fig, ax = plt.subplots()
        comparison_report.draw_measurement(
            ax,
            (measure, "", ""),
            obs,
            record,
            ranking,
            style=DEFAULT_STYLE,
            colors=colors,
        )
        assert ax.get_ylim()[0] == baseline
        assert baseline in list(ax.get_yticks()), "the baseline is a labelled tick"
        plt.close(fig)


def test_headroom_is_reserved_only_for_a_bracket():
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2)
    figlib.fit_baseline_axis(a, 0.0, 10.0, annotated=True)
    figlib.fit_baseline_axis(b, 0.0, 10.0, annotated=False)
    assert a.get_ylim()[1] > b.get_ylim()[1]
    plt.close(fig)


def test_the_panel_label_names_the_thing_and_its_unit():
    assert (
        comparison_report.panel_label(("time_fraction", "rearing", "", "whole"), ())
        == "Rearing (% time)"
    )
    assert (
        comparison_report.panel_label(("n_instances", "rearing", "", "whole"), ())
        == "Rearing (bouts)"
    )
    assert (
        comparison_report.panel_label(("distance", "", "", "whole"), ())
        == "Distance travelled (m)"
    )
    assert (
        comparison_report.panel_label(("body_elongation", "", "", "whole"), ())
        == "Body length : width"
    )
    assert (
        comparison_report.panel_label(("time_fraction", "node:1354", "", "whole"), ())
        == "Node 1354 (% time)"
    )


# --------------------------------------------------------------------------- #
# the composite
# --------------------------------------------------------------------------- #
def test_panels_holds_every_measurement_with_letters_and_one_legend(paired):
    comp, ranking, saved = paired
    axes, texts, legends = saved[comparison_report.PANELS_STEM]
    assert len(axes) == len(ranking.contrasts) == 7
    assert legends == 1
    letters = [t for t in texts if len(t) == 1 and t.isupper()]
    assert letters == list("ABCDEFG")
    assert comparison_report.observation_note(comp.record()) in texts
    assert (comp.dir / "figures" / "panels.png").is_file()


# --------------------------------------------------------------------------- #
# the SVG a person edits
# --------------------------------------------------------------------------- #
def test_the_svg_has_no_clip_masks_no_shared_markers_and_one_text_per_p(paired):
    comp, _ranking, _saved = paired
    root = ET.parse(comp.dir / "figures" / "distance.svg").getroot()
    tags = {el.tag for el in root.iter()}
    assert f"{SVG}clipPath" not in tags and f"{SVG}use" not in tags
    assert not any("url(#" in v for el in root.iter() for v in el.attrib.values())
    p_texts = [
        el
        for el in root.iter(f"{SVG}text")
        if "".join(el.itertext()).replace(" ", "").startswith("P=")
    ]
    assert len(p_texts) == 1, "the P label is one text object, not loose characters"
    first = next(iter(p_texts[0]))
    assert first.get("font-style") == "italic" and first.text == "P"


def test_a_figure_with_a_pattern_fill_is_refused_by_the_rewrite(tmp_path):
    import matplotlib.pyplot as plt

    from detail.report import svg_edit

    fig, ax = plt.subplots()
    ax.bar([0], [1], hatch="//")
    raw = tmp_path / "hatched.svg"
    fig.savefig(raw, format="svg")
    plt.close(fig)
    with pytest.raises(svg_edit.SvgRewriteError):
        svg_edit.clean(raw, tmp_path / "out.svg")


# --------------------------------------------------------------------------- #
# the statistics beside the figures
# --------------------------------------------------------------------------- #
def test_the_run_writes_the_result_tables_before_any_export(tmp_path):
    exp = cohort(tmp_path, paired_plan())
    comp = run_comparison(exp, {**PAIRED, "draws": 200})
    for name in (
        comparison_report.RANKING_FILENAME,
        comparison_report.SIGNIFICANT_FILENAME,
        comparison_report.UNITS_FILENAME,
        comparison_report.UNIT_EFFECTS_FILENAME,
        comparison_report.METHODS_FILENAME,
    ):
        assert (comp.dir / name).is_file(), name
    assert not (comp.dir / "figures").exists(), "the run drew nothing"


def test_significant_holds_the_kept_items_with_the_means_the_figure_drew(paired):
    comp, ranking, _saved = paired
    rows = _rows(comp.dir / comparison_report.SIGNIFICANT_FILENAME)
    assert [r["measure"] for r in rows] == [
        i.measure_key[0] for i in ranking.kept_items()
    ]
    by_measure = {r["measure"]: r for r in rows}
    item = next(i for i in ranking.kept_items() if i.measure_key[0] == "distance")
    row = by_measure["distance"]
    assert float(row["p"]) == item.own_p and row["p_kind"] == "own"
    assert row["contrast"] == "drug vs saline" and row["unit"] == "m"
    assert float(row["a_mean"]) == pytest.approx(item.a.mean)
    assert float(row["a_sem"]) == pytest.approx(
        np.std(item.a.values, ddof=1) / np.sqrt(item.a.n)
    )
    assert row["in_direction"] == str(item.consistency) and row["n_units"] == "8"
    assert row["change"] == comparison_report._change_text(item)


def test_unit_effects_average_to_the_rankings_effect(paired):
    comp, ranking, _saved = paired
    rows = _rows(comp.dir / comparison_report.UNIT_EFFECTS_FILENAME)
    for item in ranking.contrasts:
        mine = [float(r["effect"]) for r in rows if r["measure"] == item.measure_key[0]]
        assert len(mine) == item.n_units
        assert np.mean(mine) == pytest.approx(item.effect)
    assert all(r["pair_a"] == "saline" and r["pair_b"] == "drug" for r in rows)


def test_units_is_wide_one_row_per_unit_with_a_column_per_group(paired):
    comp, ranking, _saved = paired
    with open(
        comp.dir / comparison_report.UNITS_FILENAME, newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    assert columns[-2:] == ["saline", "drug"]
    assert len(rows) == 7 * 8
    stats = _rows(comp.dir / "stats.csv")
    one = next(
        r
        for r in stats
        if r["measure"] == "distance"
        and r["observation"] == "A0"
        and r["group"] == "drug"
    )
    wide = next(
        r for r in rows if r["measure"] == "distance" and r["observation"] == "A0"
    )
    assert float(wide["drug"]) == pytest.approx(float(one["value"]))


def test_the_methods_paragraph_says_what_ran(paired):
    comp, ranking, _saved = paired
    text = (comp.dir / comparison_report.METHODS_FILENAME).read_text(encoding="utf-8")
    assert "every one of the 256 sign-flip configurations (an exact test)" in text
    assert "tested on its own" in text and "n = 8" in text
    assert "smallest P an exact test of 8 animals can report is 0.00781" in text
    assert text == comparison_report.methods_sentence(comp.record(), ranking) + "\n"
    # a family verdict over a sampled null reads differently, and says the bound
    family = compare.Ranking(
        list(ranking.contrasts),
        0.3,
        200,
        0,
        design=compare.PAIRED,
        levels=ranking.levels,
        n_units=8,
    )
    other = comparison_report.methods_sentence(comp.record(), family)
    assert "P_FWE is family-wise" in other and "200 random sign flips" in other
    assert "A P below 1/200 is reported as that bound" in other
    assert "nothing was contrasted" in comparison_report.methods_sentence(
        {"groups": ["a"]}, None
    )


def test_the_default_formats_are_one_constant_on_every_surface(monkeypatch):
    from detail import cli
    from detail.api import export_targets

    expected = comparison_report.DEFAULT_FORMATS
    assert (
        inspect.signature(Comparison.export).parameters["formats"].default == expected
    )
    option = next(p for p in cli.comparison_export.params if p.name == "formats")
    assert tuple(option.default) == expected
    seen = {}
    monkeypatch.setattr(
        comparison_report,
        "export_comparison",
        lambda *a, **k: seen.update(k) or {"artifacts": {}},
    )
    monkeypatch.setattr(
        export_targets, "_comparison_record", lambda subject: {"id": "cmp_01"}
    )
    export_targets._run_comparison(
        type("S", (), {"experiment_dir": ".", "id": "cmp_01"})(), ".", None, None
    )
    assert seen["formats"] == expected
