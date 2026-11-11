"""Ranking the measurements a person reads, and the statistic that gets it wrong.

**The class: a rank statistic that is not tie-safe, used on data full of ties.** The obvious
way to say where a cluster sits — what fraction of the run is below its median — counts strict
inequalities, so on a measurement whose median *is* its mode almost nothing counts. Two of this
project's nine measurements are small integer counts shaped exactly that way, and on a real
36-cluster run `postures` came out at the **0th percentile for every cluster in the run**.

Nothing fails. The panel renders, the column is a number, and an extreme rank looks like a
finding rather than an artefact — so the reading is not merely wrong, it is *interesting*, which
is the worst way for a number to be wrong. It is invisible to a test that only checks the
ranking returns something, and invisible on continuous fixtures, where ties never happen.

The fix is midranks, in both places the quantity appears: :func:`cliffs_delta` for the ordering
and :func:`midrank_percentile` for the number printed beside it. Both, because they are the same
claim rendered twice — a run where the delta says "typical" and the percentile says "lowest in
the run" is worse than either alone.

Second class here: **two statistics withheld on one condition, when they fail on different ones.**
Separation needs a *partition* — two groups for the grouping to explain anything. A selection's
ranking needs a non-empty *complement*, which is not the same thing. Tying them together meant one
group withheld everything, and four of five real query runs have one behavior: the band was dead
on the surface it mattered most on, and nothing failed. A background row — a span carrying no
group — is in the complement and out of the partition, which is what lets the two come apart.

The run-level absence is still reported **with its reason**, because a blank reads as "nothing
differed", a different and much more encouraging claim.
"""

import numpy as np
import pytest

from detail.analysis.descriptor_ranking import (
    STRONG,
    Ranked,
    cliffs_delta,
    midrank_percentile,
    rank_run,
    rank_selection,
    separation,
    separation_withheld,
)


def _brute(a, b) -> float:
    """The O(n*m) definition, which is what the fast version has to agree with."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (len(a) * len(b))


# --- the statistic -------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,a,b",
    [
        ("continuous", np.linspace(0, 1, 40), np.linspace(0.3, 1.3, 60)),
        ("disjoint", np.arange(10.0), np.arange(10.0) + 100),
        ("identical", np.ones(10), np.ones(20)),
        ("every pair tied", np.full(7, 3.0), np.full(11, 3.0)),
        ("small counts", np.array([1.0, 2, 2, 3, 1]), np.array([1.0, 1, 2, 3, 3, 2])),
        (
            "one dominant mode",
            np.r_[np.ones(48), 2.0, 3.0],
            np.r_[np.ones(90), [2.0] * 10],
        ),
        ("singletons", np.array([2.0]), np.array([5.0])),
    ],
)
def test_the_fast_delta_is_the_definition(name, a, b):
    """Midranks by two searchsorted lookups rather than counting pairs — same answer, and the
    tie cases are the ones the shortcut could plausibly get wrong."""
    assert cliffs_delta(a, b) == pytest.approx(_brute(a, b), abs=1e-12), name


def test_a_discrete_measurement_is_not_ranked_at_the_extreme():
    """The reported failure. `postures` is 1 for most bouts, so a cluster whose median is also 1
    is *typical* — and the naive percentile calls it the lowest in the run.

    Revert `midrank_percentile` to `mean(rest < value)` and this fails on the first assertion;
    revert `cliffs_delta` to a strict-inequality count and it fails on the second.
    """
    run = np.r_[np.ones(600), np.full(100, 2.0), np.full(33, 3.0)]
    mine = np.ones(26)  # a cluster sitting squarely on the mode

    naive = 100.0 * (run < np.median(mine)).mean()
    assert naive == pytest.approx(0.0), "the fixture must reproduce the broken reading"

    assert 20.0 < midrank_percentile(1.0, run) < 80.0
    assert abs(cliffs_delta(mine, run)) < STRONG
    # And the two agree about which side of the middle it is on, which is the whole point of
    # correcting both: a "typical" delta beside a "lowest in the run" percentile is worse than
    # either number alone.
    assert (midrank_percentile(1.0, run) > 50.0) == (cliffs_delta(mine, run) > 0)


def test_an_empty_side_has_no_standing():
    """A measurement no member could be computed for is `NaN`, not zero: zero is a real answer
    meaning "indistinguishable", and it would sort into the middle of the ranking as if measured.
    """
    assert np.isnan(cliffs_delta([], [1.0, 2.0]))
    assert np.isnan(cliffs_delta([np.nan, np.nan], [1.0]))
    assert np.isnan(midrank_percentile(1.0, []))
    assert np.isnan(midrank_percentile(float("nan"), [1.0, 2.0]))


def test_separation_is_a_property_of_the_partition():
    clean = separation([0.0, 0.0, 9.0, 9.0], ["a", "a", "b", "b"])
    none = separation([0.0, 9.0, 0.0, 9.0], ["a", "a", "b", "b"])
    assert clean == pytest.approx(1.0)
    assert none == pytest.approx(0.0)
    assert np.isnan(
        separation([1.0, 2.0, 3.0], ["a", "a", "a"])
    ), "one group explains nothing"
    assert np.isnan(
        separation([2.0, 2.0, 2.0], ["a", "b", "b"])
    ), "no variance to explain"


# --- the two rankings ------------------------------------------------------------------ #
@pytest.fixture
def run_values():
    """Two measurements over four groups: one the partition explains, one it does not."""
    # Sixty per group, not ten: at ten, one group's draw of pure noise landed at delta 0.41 and
    # the "not strong" assertion failed on the fixture rather than on the code. A guard whose
    # control condition sits within sampling error of the thing it controls for is not a control.
    per_group = 60
    labels = np.array(sum(([name] * per_group for name in "abcd"), []))
    rng = np.random.default_rng(0)
    return {
        "separated": np.concatenate(
            [rng.normal(centre, 0.1, per_group) for centre in (0, 5, 10, 15)]
        ),
        "noise": rng.normal(0, 1, per_group * 4),
    }, labels


def test_the_run_ranking_puts_the_separated_measurement_first(run_values):
    values, labels = run_values
    ranked = rank_run(values, labels)
    assert [name for name, _score in ranked] == ["separated", "noise"]
    assert ranked[0][1] > 0.9 and ranked[1][1] < 0.3


def test_the_run_ranking_does_not_move_with_the_selection(run_values):
    """It is a property of the partition, which is what makes it the answer to "which
    measurements is this clustering about" rather than "about this cluster"."""
    values, labels = run_values
    assert rank_run(values, labels) == rank_run(values, labels)


def test_the_selection_ranking_orders_by_effect_and_marks_the_strong(run_values):
    values, labels = run_values
    ranked = rank_selection(values, labels, "a")
    assert [r.name for r in ranked] == ["separated", "noise"]
    assert (
        ranked[0].strong and ranked[0].delta < 0
    ), "group a is the low end of a clean split"
    assert not ranked[1].strong
    assert [abs(r.delta) for r in ranked] == sorted(
        (abs(r.delta) for r in ranked), reverse=True
    )


def test_a_measurement_with_nothing_on_one_side_is_dropped_not_ranked(run_values):
    values, labels = run_values
    values = {**values, "unmeasurable": np.full(len(labels), np.nan)}
    assert "unmeasurable" not in {r.name for r in rank_selection(values, labels, "a")}
    assert "unmeasurable" not in {name for name, _s in rank_run(values, labels)}


def test_one_group_withholds_the_run_line_only(run_values):
    """The Query shape. With one group there is no partition, so separation has no answer — but
    with nothing else measured there is also no complement, so the selection cannot be ranked
    either. Both absent, for two different reasons."""
    values, _labels = run_values
    n = len(next(iter(values.values())))
    labels = np.array(["only"] * n, dtype=object)
    reason = separation_withheld(labels, "behavior")
    assert reason is not None and "only" in reason and "behavior" in reason
    assert rank_selection(values, labels, "only") == []


def test_background_rows_rank_a_lone_group_without_joining_the_partition(run_values):
    """The fix, and the thing that was dead. A run with one behavior plus background spans:
    the behavior *can* be ranked — against the recording — while separation still has nothing
    to explain, because a background row belongs to no group.

    Drop the `None` handling from `_labeled` and separation counts background as a second group,
    so the run line reports a number for a run with one behavior.
    """
    values, labels = run_values
    n = len(next(iter(values.values())))
    # One behavior over the first quarter; the rest is background carrying no group.
    lone = np.array(["q01" if i < n // 4 else None for i in range(n)], dtype=object)
    assert separation_withheld(lone, "behavior") is not None, "still one group"
    assert rank_run(values, lone) == [], "a partition of one explains nothing"

    ranked = rank_selection(values, lone, "q01")
    assert ranked, "the behavior must be rankable against the background"
    assert {r.name for r in ranked} == set(values)
    assert any(r.strong for r in ranked), "the separated measurement must still show"


def test_two_groups_are_not_withheld(run_values):
    values, labels = run_values
    assert separation_withheld(labels) is None
    assert rank_selection(values, labels, "a")


def test_ranked_reports_which_side_it_is_on():
    assert Ranked("x", 1.0, delta=0.5, percentile=90.0).above
    assert not Ranked("x", 1.0, delta=-0.5, percentile=10.0).above
    assert Ranked("x", 1.0, delta=STRONG, percentile=90.0).strong
    assert not Ranked("x", 1.0, delta=STRONG - 0.01, percentile=90.0).strong
