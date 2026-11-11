"""§24's figure primitives: show every observation, and never a ranking without its floor.

At the cohort sizes this project works at — three animals a group, sometimes two — a bar with an
error bar hides the entire dataset: the reader cannot tell six overlapping points from one outlier
dragging a mean. §24 requires the points themselves, so ``observation_dots`` is a **sibling** of
``grouped_dots`` rather than a flag on it. §18's benchmark exporters passed a value and an
optional error per group through ``grouped_dots``, which is why its signature was kept
unchanged; the signature test below still pins it, though those exporters have since been
removed.
"""

import inspect
from dataclasses import fields

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detail.analysis.compare import Ranking
from detail.report import figures


def _marks(ax):
    """Every drawn point across marker lines and scatter collections."""
    from_lines = sum(
        len(line.get_xdata())
        for line in ax.lines
        if line.get_marker() not in ("", "None")
    )
    return from_lines + sum(len(c.get_offsets()) for c in ax.collections)


def test_observation_dots_draws_every_observation():
    """The whole point of the figure: five animals must render five marks, not one mean."""
    fig, ax = plt.subplots()
    figures.observation_dots(
        ax, ["saline", "drug"], [[1.0, 2.0, 3.0, 4.0, 5.0], [7.0, 8.0]], ylabel="bouts"
    )
    assert (
        _marks(ax) == 7
    ), "an observation went missing between the data and the canvas"
    plt.close(fig)


def test_coincident_values_are_still_countable():
    """Four equal values stacked on one x read as a single animal. The spread is what makes n
    legible, so it has to survive the case where every value is identical."""
    fig, ax = plt.subplots()
    figures.observation_dots(ax, ["a"], [[2.0, 2.0, 2.0, 2.0]], ylabel="bouts")
    xs = [x for line in ax.lines if line.get_marker() == "o" for x in line.get_xdata()]
    assert len(set(xs)) == 4, f"four equal values landed on {len(set(xs))} x positions"
    plt.close(fig)


def test_the_spread_is_deterministic_not_jittered():
    """Re-export must be byte-identical (§18.5 L1), which rules out an unseeded random jitter —
    the obvious way to draw this figure and the one that would break determinism silently.
    """
    runs = []
    for _ in range(2):
        fig, ax = plt.subplots()
        figures.observation_dots(ax, ["a", "b"], [[1.0, 1.0, 1.0], [2.0, 2.0]])
        runs.append(
            [
                (tuple(line.get_xdata()), tuple(line.get_ydata()))
                for line in ax.lines
                if line.get_marker() == "o"
            ]
        )
        plt.close(fig)
    assert runs[0] == runs[1]


def test_a_single_observation_is_centered():
    """n=1 has no band to spread across; ``linspace`` over one point would put it at the edge."""
    fig, ax = plt.subplots()
    figures.observation_dots(ax, ["a"], [[5.0]])
    xs = [x for line in ax.lines if line.get_marker() == "o" for x in line.get_xdata()]
    assert xs == [0.0]
    plt.close(fig)


def test_an_empty_group_draws_nothing_and_keeps_its_tick():
    """A cohort with no observations for one group must still show that group as empty, rather
    than silently shifting the others left onto its label."""
    fig, ax = plt.subplots()
    figures.observation_dots(ax, ["saline", "drug"], [[], [3.0, 4.0]])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["saline", "drug"]
    assert _marks(ax) == 2
    plt.close(fig)


def test_grouped_dots_keeps_the_shape_section18_depends_on():
    """The reason this is a sibling and not a flag. §18's exporters were the callers, until
    benchmarks were removed."""
    params = list(inspect.signature(figures.grouped_dots).parameters)
    assert params[:3] == ["ax", "group_labels", "series"]


def test_the_floor_cannot_be_left_out_of_a_ranking_figure():
    """'A headline metric ships with its floor' made structural rather than remembered: ``floor``
    is a required positional parameter, so there is no call that omits it."""
    floor_param = inspect.signature(figures.ranked_bars_with_floor).parameters["floor"]
    assert floor_param.default is inspect.Parameter.empty
    assert floor_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_a_ranking_cannot_be_built_without_one():
    """Same rule one level up: the dataclass has no default for ``floor`` either, so a ranking
    carrying no floor cannot be constructed and then handed to the figure."""
    assert "floor" in {f.name for f in fields(Ranking)}
    floor_field = next(f for f in fields(Ranking) if f.name == "floor")
    import dataclasses

    assert floor_field.default is dataclasses.MISSING
    assert floor_field.default_factory is dataclasses.MISSING


def test_the_ranking_figure_draws_the_floor_as_a_rule():
    fig, ax = plt.subplots()
    figures.ranked_bars_with_floor(
        ax, ["rearing", "grooming"], [3.0, 1.0], 2.0, xlabel="difference"
    )
    rules = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert rules, "the floor was not drawn"
    assert rules[0].get_xdata()[0] == 2.0
    plt.close(fig)


def test_an_absent_floor_is_stated_not_omitted():
    """A floorless figure drawn as bare bars is indistinguishable from one whose bars all cleared
    the floor — the more dangerous of the two readings, so it has to say which it is."""
    fig, ax = plt.subplots()
    figures.ranked_bars_with_floor(ax, ["rearing"], [3.0], None, xlabel="difference")
    text = " ".join(t.get_text() for t in ax.texts).lower()
    assert "no floor" in text
    plt.close(fig)
