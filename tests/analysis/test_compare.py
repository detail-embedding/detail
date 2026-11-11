"""A comparison says what one point is, and never invents observations (§24).

Three classes guarded here, each of which renders as a clean chart while being wrong:

**A silently chosen aggregation path.** One animal contributes four sessions and another one; pool
recordings and the first votes four times. Both poolings are legitimate and they answer different
questions, so the tool must not pick — and the choice must reach every output row, because "one
point is one animal" and "one point is one session" are different claims about the same picture.

**A ranking with no floor.** With tens of measures and a handful of animals, ranking by difference
always produces a most-different one whether or not anything is there. The floor is what the same
procedure yields on shuffled labels — and it must shuffle whole *observations*, since shuffling the
underlying recordings breaks one animal into four independent draws and makes the floor far too
easy to clear.

**A window folded into a group mean.** ``window_id`` identifies *what* was measured, not who: drop
it and a group's mean averages the first third of every session with the last third, answering the
opposite of the question a windowed comparison was asked.
"""

import pytest

from detail.analysis import compare as c


def _rows(*specs):
    """``(recording, value)`` or ``(recording, value, window, measure)`` → measure rows."""
    out = []
    for spec in specs:
        recording, value = spec[0], spec[1]
        window = spec[2] if len(spec) > 2 else "whole"
        measure = spec[3] if len(spec) > 3 else "rate"
        out.append(
            {
                "recording_id": recording,
                "subject": 0,
                "window_id": window,
                "measure": measure,
                "label": "rearing",
                "label_to": "",
                "value": value,
            }
        )
    return out


# One animal with three sessions, one with one — the unbalanced shape a balanced fixture hides.
UNBALANCED_META = {
    "r1": {"animal": "A", "drug": "veh"},
    "r2": {"animal": "A", "drug": "veh"},
    "r3": {"animal": "A", "drug": "veh"},
    "r4": {"animal": "B", "drug": "veh"},
}
UNBALANCED_ROWS = _rows(("r1", 10.0), ("r2", 20.0), ("r3", 30.0), ("r4", 100.0))


def _one(summaries):
    assert len(summaries) == 1, summaries
    return summaries[0]


# --- the aggregation path ----------------------------------------------------------------- #


def test_the_path_changes_the_answer_on_an_unbalanced_design():
    """The headline guard. A path that silently pooled would pass a balanced fixture."""
    pooled = _one(
        c.summarize(
            c.observations(
                UNBALANCED_ROWS, UNBALANCED_META, group_by=["drug"], aggregate_to=[]
            )
        )
    )
    by_animal = _one(
        c.summarize(
            c.observations(
                UNBALANCED_ROWS,
                UNBALANCED_META,
                group_by=["drug"],
                aggregate_to=["animal"],
            )
        )
    )
    assert pooled.n == 4 and pooled.mean == pytest.approx(40.0)  # (10+20+30+100)/4
    assert by_animal.n == 2 and by_animal.mean == pytest.approx(60.0)  # (20+100)/2
    assert pooled.mean != by_animal.mean


def test_an_empty_path_leaves_every_row_its_own_observation():
    """``[]`` means no aggregation, not "collapse the group". An empty tuple key would report
    n = 1 for a cohort of any size, which looks like a legitimately small study."""
    obs = c.observations(
        UNBALANCED_ROWS, UNBALANCED_META, group_by=["drug"], aggregate_to=[]
    )
    assert len(obs) == 4
    assert all(o.n_underlying == 1 for o in obs)


def test_an_aggregated_observation_records_how_many_rows_it_averaged():
    """The sample-size trail: whether an animal-level point rests on three sessions or one."""
    obs = c.observations(
        UNBALANCED_ROWS, UNBALANCED_META, group_by=["drug"], aggregate_to=["animal"]
    )
    assert sorted(o.n_underlying for o in obs) == [1, 3]


def test_the_path_is_stamped_on_every_row():
    """A table or figure built from these is unreadable without it, and nothing else on the row
    distinguishes one-point-is-one-animal from one-point-is-one-session."""
    obs = c.observations(
        UNBALANCED_ROWS, UNBALANCED_META, group_by=["drug"], aggregate_to=["animal"]
    )
    rows = c.stats_rows(obs, aggregate_to=["animal"], group_by=["drug"])
    assert rows and all(r["aggregate_to"] == "animal" for r in rows)


def test_an_unknown_grouping_column_is_named_not_ignored():
    """Ignoring it groups everything into one bucket labelled with an empty string, which renders
    as a perfectly plausible single-group comparison."""
    with pytest.raises(c.ComparisonError, match="cage"):
        c.observations(
            UNBALANCED_ROWS, UNBALANCED_META, group_by=["cage"], aggregate_to=[]
        )
    with pytest.raises(c.ComparisonError, match="litter"):
        c.observations(
            UNBALANCED_ROWS, UNBALANCED_META, group_by=["drug"], aggregate_to=["litter"]
        )


# --- windows ------------------------------------------------------------------------------- #


def test_windows_stay_separate_measurements_unless_grouped_on():
    rows = _rows(("r1", 10.0, "abs_00"), ("r1", 20.0, "abs_01"))
    meta = {"r1": {"drug": "veh"}}
    keys = {
        s.measure_key
        for s in c.summarize(
            c.observations(rows, meta, group_by=["drug"], aggregate_to=[])
        )
    }
    assert len(keys) == 2, "two windows must not fold into one group mean"


def test_grouping_by_window_moves_it_into_the_group():
    """ "Early versus late" is the same operation as "drug A versus B", not a separate feature."""
    rows = _rows(("r1", 10.0, "abs_00"), ("r2", 40.0, "abs_00"), ("r1", 20.0, "abs_01"))
    meta = {"r1": {"drug": "veh"}, "r2": {"drug": "fent"}}
    summaries = c.summarize(
        c.observations(rows, meta, group_by=["window_id"], aggregate_to=[])
    )
    by_group = {s.group: s.values for s in summaries}
    assert by_group[("abs_00",)] == [10.0, 40.0]
    assert by_group[("abs_01",)] == [20.0]


# --- what a summary must carry --------------------------------------------------------------- #


def test_a_group_of_one_reports_no_spread():
    """Not zero, which reads as "no variation observed" when the truth is "not observable"."""
    obs = c.observations(
        _rows(("r1", 10.0)), {"r1": {"drug": "veh"}}, group_by=["drug"], aggregate_to=[]
    )
    assert _one(c.summarize(obs)).sd is None


def test_every_observation_survives_into_the_summary():
    """At these cohort sizes the points *are* the result; a centre and a spread hide the data."""
    obs = c.observations(
        UNBALANCED_ROWS, UNBALANCED_META, group_by=["drug"], aggregate_to=[]
    )
    assert _one(c.summarize(obs)).values == [10.0, 20.0, 30.0, 100.0]


def test_coverage_is_reported_alongside_n():
    """A group whose recordings were never analyzed is otherwise indistinguishable from a small
    group, and the mean of whatever arrived is reported with no sign anything is missing.
    """
    obs = c.observations(
        UNBALANCED_ROWS, UNBALANCED_META, group_by=["drug"], aggregate_to=[]
    )
    summary = _one(c.summarize(obs, coverage_by_group={("veh",): 10}))
    assert summary.n == 4 and summary.coverage == 10


def test_coverage_carries_the_signal_it_claims_to(tmp_path):
    """The test above hands `summarize` a number and checks it comes back out — which proves the
    field is *carried*, not that `coverage_by_group` *produces* anything readable.

    The reading the column exists for: `coverage > n` means this measure declined for some of the
    group's units, while a measure that answered for all of them reads `coverage == n`. Two units,
    two measures, one of which blanks on the second unit.
    """
    rows = _rows(
        ("r1", 10.0, "whole", "rate"),
        ("r2", 20.0, "whole", "rate"),
        ("r1", 1.0, "whole", "duration_mean"),
        ("r2", "", "whole", "duration_mean"),
    )
    meta = {"r1": {"drug": "veh"}, "r2": {"drug": "veh"}}
    obs = c.observations(rows, meta, group_by=["drug"], aggregate_to=[])

    coverage = c.coverage_by_group(obs)
    assert coverage == {("veh",): 2}

    by_measure = {
        s.measure_key[0]: s for s in c.summarize(obs, coverage_by_group=coverage)
    }
    assert by_measure["rate"].n == 2 and by_measure["rate"].coverage == 2
    assert (
        by_measure["duration_mean"].n == 1
    ), "the blank was coerced rather than dropped"
    assert by_measure["duration_mean"].coverage == 2, (
        "coverage collapsed to this measure's own n, so a measure that declined for half the "
        "group reads exactly like one that answered for all of it"
    )


def test_a_declined_measurement_is_dropped_not_coerced():
    """§23 blanks a value it will not answer (a point behavior's duration). Coercing it to zero
    would enter a group mean as a real observation of "never happened"."""
    rows = _rows(("r1", 10.0), ("r2", 20.0))
    rows[1]["value"] = ""
    obs = c.observations(
        rows,
        {"r1": {"drug": "veh"}, "r2": {"drug": "veh"}},
        group_by=["drug"],
        aggregate_to=[],
    )
    assert _one(c.summarize(obs)).values == [10.0]


# --- contrasts and the floor ------------------------------------------------------------------ #


def _two_group(values_a, values_b, measure="rate"):
    meta, rows = {}, []
    for i, v in enumerate(values_a):
        meta[f"a{i}"] = {"drug": "veh"}
        rows += _rows((f"a{i}", v, "whole", measure))
    for i, v in enumerate(values_b):
        meta[f"b{i}"] = {"drug": "fent"}
        rows += _rows((f"b{i}", v, "whole", measure))
    return c.observations(rows, meta, group_by=["drug"], aggregate_to=[])


def test_a_contrast_reports_difference_and_ratio():
    obs = _two_group([10.0, 10.0], [30.0, 30.0])
    ranking = c.rank_by_difference(obs, (("veh",), ("fent",)), draws=20)
    top = ranking.contrasts[0]
    assert top.difference == pytest.approx(20.0)
    assert top.ratio == pytest.approx(3.0)


def test_a_ratio_against_a_zero_reference_is_none_not_infinity():
    """ "Infinitely more common" is not a finding, it is a division."""
    obs = _two_group([0.0, 0.0], [5.0, 5.0])
    top = c.rank_by_difference(obs, (("veh",), ("fent",)), draws=20).contrasts[0]
    assert top.difference == pytest.approx(5.0)
    assert top.ratio is None


def test_a_ranking_always_carries_its_floor():
    """Computed in the same pass, so a ranking cannot be rendered without one."""
    ranking = c.rank_by_difference(
        _two_group([1.0, 2.0], [8.0, 9.0]), (("veh",), ("fent",)), draws=50
    )
    assert ranking.floor is not None
    assert ranking.draws == 50 and len(ranking.floor_distribution) == 50


def test_a_real_effect_clears_the_floor_and_noise_does_not():
    """Six per group: 924 distinct splits, so the true labelling is a small fraction of the null
    and a clean separation reaches the 95th percentile."""
    six = [1.0] * 6
    separated = c.rank_by_difference(
        _two_group(six, [50.0] * 6), (("veh",), ("fent",)), draws=400
    )
    assert separated.clears_floor(separated.contrasts[0])

    interleaved = c.rank_by_difference(
        _two_group(
            [1.0, 50.0, 1.0, 50.0, 1.0, 50.0], [50.0, 1.0, 50.0, 1.0, 50.0, 1.0]
        ),
        (("veh",), ("fent",)),
        draws=400,
    )
    assert not interleaved.clears_floor(interleaved.contrasts[0])


def test_a_cohort_too_small_cannot_clear_the_floor_however_large_the_effect():
    """Three per group gives twenty distinct splits, so the true labelling is 10% of the null and
    nothing reaches the 95th percentile. The floor is reporting that this cohort cannot distinguish
    any effect from chance — which is the answer, not a defect."""
    ranking = c.rank_by_difference(
        _two_group([1.0, 1.0, 1.0], [1000.0, 1000.0, 1000.0]),
        (("veh",), ("fent",)),
        draws=400,
    )
    assert ranking.contrasts[0].difference == pytest.approx(999.0)
    assert not ranking.clears_floor(ranking.contrasts[0])


def test_the_floor_is_a_percentile_and_not_the_maximum():
    """Pinned directly, because nothing else catches it.

    Every permutation distribution contains the observed labelling — shuffle enough times and one
    draw reproduces it — so a maximum floor is always at least the real effect and nothing can ever
    clear it. Reverting to ``max`` leaves the other floor tests passing (they simply become
    seed-dependent), which is a guard that passes forever.
    """
    graded = c.rank_by_difference(
        _two_group(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        ),
        (("veh",), ("fent",)),
        draws=400,
    )
    dist = graded.floor_distribution
    expected = dist[min(int(round(0.95 * (len(dist) - 1))), len(dist) - 1)]
    assert graded.floor == expected
    assert graded.floor < max(
        dist
    ), "a percentile floor sits below the distribution's maximum"


def test_the_floor_is_seeded_and_reproduces():
    obs = _two_group([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    a = c.rank_by_difference(obs, (("veh",), ("fent",)), draws=50, seed=7)
    b = c.rank_by_difference(obs, (("veh",), ("fent",)), draws=50, seed=7)
    assert a.floor == b.floor and a.floor_distribution == b.floor_distribution


def test_the_floor_shuffles_whole_observations_not_underlying_rows():
    """Shuffling recordings would break one animal into three independent draws, inflate the
    effective sample size, and make the floor far easier to clear than the ranking had to work for.
    Here the path aggregates to animal, so there are only two observations to permute — and a floor
    built from two units cannot exceed the one real difference between them."""
    meta = {
        f"r{i}": {"animal": "A" if i < 3 else "B", "drug": "veh" if i < 3 else "fent"}
        for i in range(6)
    }
    rows = _rows(*[(f"r{i}", 10.0 if i < 3 else 20.0) for i in range(6)])
    obs = c.observations(rows, meta, group_by=["drug"], aggregate_to=["animal"])
    assert len(obs) == 2, "the permutation universe is observations, not rows"
    ranking = c.rank_by_difference(obs, (("veh",), ("fent",)), draws=100)
    # Two units: every shuffle is the true split or its mirror, so the null is a single value and
    # no effect clears it. Shuffling the six underlying rows instead would manufacture a null with
    # many small differences and let this pass.
    assert not ranking.clears_floor(ranking.contrasts[0])


def test_a_contrast_needs_exactly_two_groups():
    with pytest.raises(c.ComparisonError, match="two groups"):
        c.rank_by_difference(_two_group([1.0], [2.0]), (("veh",),), draws=5)


def test_contrasting_different_measures_is_refused():
    a = c.GroupSummary(("veh",), ("rate", "rearing", "", "whole"), [1.0], 1)
    b = c.GroupSummary(("fent",), ("total_time", "rearing", "", "whole"), [2.0], 1)
    with pytest.raises(c.ComparisonError, match="different measures"):
        c.contrast(a, b)


def test_measures_are_ranked_by_absolute_difference():
    rows = _rows(
        ("r1", 1.0, "whole", "rate"),
        ("r2", 2.0, "whole", "rate"),
        ("r1", 1.0, "whole", "total_time"),
        ("r2", 90.0, "whole", "total_time"),
    )
    meta = {"r1": {"drug": "veh"}, "r2": {"drug": "fent"}}
    obs = c.observations(rows, meta, group_by=["drug"], aggregate_to=[])
    ranking = c.rank_by_difference(obs, (("veh",), ("fent",)), draws=20)
    assert ranking.contrasts[0].measure_key[0] == "total_time"
