"""Ranking measures that do not share a unit, and refusing a floor that cannot be exceeded.

**The class.** A ranking sorted by *raw difference* is only an ordering when every measure shares
a unit. Nothing enforced that, and nothing could: `compare.py` never imported the measure registry,
so it had no way to know that `total_time` is nanoseconds and `n_instances` is a count. On a real
comparison the top bar was 2.5e11 (nanoseconds) and rearing halving — 92 to 47, the most
interpretable result in the dataset — rendered as a bar of **zero width**. The chart was correct
arithmetic and useless information.

The same defect reached the floor. `_largest_difference` took the maximum across all measures,
which is a family-wise multiple-comparisons correction and deliberate — but it made the floor a
*nanosecond* quantity, so a count measure could never reach it at any sample size. Ranking on a
unit-free scale fixes both at once, because the null is then measured in the same unit-free scale
as the thing being ranked.

**The second class, in the same function.** `Ranking`'s docstring says a floor taken from the
maximum "would be unfalsifiable by construction" — and then the code shipped a data shape that
produces exactly that. With one observation per group, a swap leaves every magnitude unchanged, so
the null's maximum *is* the observed maximum and `clears_floor`'s `>` can never be satisfied. The
real comparison: floor equal to the observed maximum to six significant figures, 0 of 19 measures
clearing it. The rule now is that a floor is reported only when the permutation had something to
move, and withheld **with its reason** otherwise — a blank reads as "nothing differed", which is a
different claim.

**The third class: a permutation null drawn over a finer unit than the one the design
randomised.** A between design randomises animals, but the null shuffled the group label once per
(animal, measure) observation. So under the null a measure could be handed 0 to 8 "drug" animals
in a 4-against-4 design, an animal's measures stopped moving together (which the family-wise
maximum assumes), and a measure's own p depended on which *other* measures were in the
comparison: 0.0314 alone and 0.0574 beside twelve flat measures, on data whose exact
unit-permutation p is 0.0286. Every number was plausible and nothing failed. The rule now: one
unit → group map per draw, shuffled among the units of the contrasted groups and applied to every
measure (`compare.rank_by_difference`), as the within designs already permute within units. The
guards are the tests under *the between null permutes units*. **Seen failing:** the per-row
shuffle put back in `rank_by_difference` → 5 failed (own p unchanged by unrelated measures, copies
share one null, group sizes kept in every draw, own p near the exact p, other groups' units out
of the null), 12 passed; restored → 17 passed. The refusal of a unit in two groups is new
behaviour and does not depend on the shuffle.
"""

import math

import pytest

from detail.analysis import compare as c


def _obs(spec, measure="rate", label="rearing"):
    """`{group: [values]}` -> observations, one per value."""
    meta, rows = {}, []
    for group, values in spec.items():
        for i, value in enumerate(values):
            recording = f"{group}{i}"
            meta[recording] = {"drug": group}
            rows.append(
                {
                    "recording_id": recording,
                    "subject": 0,
                    "window_id": "whole",
                    "measure": measure,
                    "label": label,
                    "label_to": "",
                    "value": value,
                }
            )
    return c.observations(rows, meta, group_by=["drug"], aggregate_to=[])


def _merge(*groups):
    out = []
    for group in groups:
        out.extend(group)
    return out


GROUPS = (("veh",), ("fent",))


# --------------------------------------------------------------------------- #
# ranking on a scale where the measures are comparable
# --------------------------------------------------------------------------- #
def test_a_count_and_a_duration_with_the_same_fold_change_rank_together():
    """The whole point. `n_instances` doubling and `total_time` doubling are the same size of
    change; ranked by raw difference the nanosecond one wins by nine orders of magnitude.
    """
    counts = _obs({"veh": [10.0, 10.0], "fent": [20.0, 20.0]}, measure="n_instances")
    durations = _obs({"veh": [1e10, 1e10], "fent": [2e10, 2e10]}, measure="total_time")

    ranking = c.rank_by_difference(_merge(counts, durations), GROUPS, draws=20)

    magnitudes = [item.magnitude for item in ranking.contrasts]
    assert magnitudes[0] == pytest.approx(magnitudes[1]), (
        "a count and a duration that both doubled did not rank together: "
        f"{[(i.measure_key[0], i.magnitude) for i in ranking.contrasts]}"
    )


def test_the_bigger_fold_change_wins_even_though_its_raw_difference_is_tiny():
    """The real case, in miniature: a count that quadruples against a duration that moves 1%."""
    counts = _obs({"veh": [10.0, 10.0], "fent": [40.0, 40.0]}, measure="n_instances")
    durations = _obs(
        {"veh": [1e11, 1e11], "fent": [1.01e11, 1.01e11]}, measure="total_time"
    )

    ranking = c.rank_by_difference(_merge(counts, durations), GROUPS, draws=20)

    assert ranking.contrasts[0].measure_key[0] == "n_instances"


def test_the_ranking_does_not_depend_on_which_group_is_written_first():
    """92 -> 47 and 47 -> 92 are the same size of change. As percentages they are -49% and +96%,
    which is why the ranking is logarithmic and only the *display* is percent."""
    obs = _merge(
        _obs({"veh": [92.0, 90.0], "fent": [47.0, 45.0]}, measure="n_instances"),
        _obs({"veh": [1.0, 1.1], "fent": [1.4, 1.5]}, measure="rate"),
    )

    forward = c.rank_by_difference(obs, (("veh",), ("fent",)), draws=20)
    reverse = c.rank_by_difference(obs, (("fent",), ("veh",)), draws=20)

    assert [i.measure_key for i in forward.contrasts] == [
        i.measure_key for i in reverse.contrasts
    ]
    assert forward.contrasts[0].magnitude == pytest.approx(
        reverse.contrasts[0].magnitude
    )


def test_percent_change_is_what_a_reader_sees():
    obs = _obs({"veh": [92.0], "fent": [47.0]}, measure="n_instances")

    (item,) = c.rank_by_difference(obs, GROUPS, draws=20).contrasts

    assert item.percent_change == pytest.approx(-48.9, abs=0.1)
    assert item.magnitude == pytest.approx(abs(math.log(47 / 92)))


# --------------------------------------------------------------------------- #
# the edges of a fold change
# --------------------------------------------------------------------------- #
def test_no_change_at_zero_is_no_change_rather_than_an_error():
    """`unnamed_time` is zero throughout a query run — every cluster is named. A real row, and
    its fold change is 1, not undefined."""
    obs = _obs({"veh": [0.0], "fent": [0.0]}, measure="unnamed_time")

    (item,) = c.rank_by_difference(obs, GROUPS, draws=20).contrasts

    assert item.magnitude == 0.0
    assert not item.unbounded


def test_a_behavior_that_appears_is_ranked_first_and_not_given_a_number():
    """0 -> 34 is the largest relative change there is and has no finite magnitude. Reported as
    itself rather than dropped to the bottom or handed an invented value."""
    appeared = _obs({"veh": [0.0], "fent": [34.0]}, measure="n_instances")
    modest = _obs({"veh": [10.0], "fent": [12.0]}, measure="rate")

    ranking = c.rank_by_difference(_merge(appeared, modest), GROUPS, draws=20)

    top = ranking.contrasts[0]
    assert top.measure_key[0] == "n_instances"
    assert top.unbounded and top.magnitude is None
    assert not ranking.clears_floor(top), "an unbounded change is not a cleared floor"


def test_an_unbounded_change_does_not_clear_a_floor_that_exists():
    """The `not clears_floor` assertion above cannot fail: with one observation per group there
    is no floor at all, so *everything* returns False. This puts a real floor behind it — six per
    group, so the permutation has something to move — and keeps an appearing behavior beside a
    modest one. "Not shown to clear" is the answer, and the surface says why through
    `no_floor_reason` / the change column rather than reading this as a verdict.
    """
    appeared = _obs({"veh": [0.0] * 6, "fent": [34.0] * 6}, measure="n_instances")
    modest = _obs({"veh": [10.0] * 6, "fent": [12.0] * 6}, measure="rate")

    ranking = c.rank_by_difference(_merge(appeared, modest), GROUPS, draws=400)

    assert ranking.floor is not None, "the fixture has to have a floor to not clear"
    top = ranking.contrasts[0]
    assert top.unbounded
    assert not ranking.clears_floor(top)


# --------------------------------------------------------------------------- #
# the floor is withheld when the permutation had nothing to move
# --------------------------------------------------------------------------- #
def test_one_observation_per_group_gets_no_floor_and_a_reason():
    """The shape that started this. A swap leaves every magnitude unchanged, so the null's
    maximum is the observed maximum and `>` can never be satisfied."""
    obs = _merge(
        _obs({"veh": [92.0], "fent": [47.0]}, measure="n_instances"),
        _obs({"veh": [1.0], "fent": [3.0]}, measure="rate"),
    )

    ranking = c.rank_by_difference(obs, GROUPS, draws=200)

    assert ranking.floor is None
    assert "at most one observation per group" in (ranking.no_floor_reason or "")
    assert not ranking.clears_floor(ranking.contrasts[0])


def test_a_cohort_with_something_to_move_still_gets_a_floor():
    """The negative case — the gate must not swallow every floor. Six per group with a clean
    separation: the labelling that reproduces the effect is 2 of 924 splits, so the null tops out
    well below the observed and the floor discriminates."""
    obs = _obs({"veh": [1.0] * 6, "fent": [50.0] * 6})

    ranking = c.rank_by_difference(obs, GROUPS, draws=400)

    assert ranking.floor is not None and ranking.no_floor_reason is None
    assert ranking.clears_floor(ranking.contrasts[0])


def test_a_withheld_floor_is_distinguishable_from_a_cleared_one():
    """`clears_floor` is False in both cases, so a surface that renders it as a verdict would
    report "did not differ" for a comparison that was never tested. The reason is the seam that
    keeps those apart."""
    untestable = c.rank_by_difference(
        _obs({"veh": [1.0], "fent": [50.0]}), GROUPS, draws=200
    )
    # Interleaved, so the two groups have the same centre: a floor gets computed and the top
    # measure does not clear it. That is a *result*; the case above is not.
    tested = c.rank_by_difference(
        _obs(
            {
                "veh": [1.0, 50.0, 1.0, 50.0, 1.0, 50.0],
                "fent": [50.0, 1.0, 50.0, 1.0, 50.0, 1.0],
            }
        ),
        GROUPS,
        draws=400,
    )

    assert not untestable.clears_floor(untestable.contrasts[0])
    assert not tested.clears_floor(tested.contrasts[0])
    assert untestable.no_floor_reason and tested.no_floor_reason is None


# --------------------------------------------------------------------------- #
# guard the guard
# --------------------------------------------------------------------------- #
def test_ranking_by_raw_difference_would_fail_the_unit_test_above():
    """The assertion is capable of failing: sorted the old way, the nanosecond measure wins."""
    obs = _merge(
        _obs({"veh": [10.0, 10.0], "fent": [40.0, 40.0]}, measure="n_instances"),
        _obs({"veh": [1e11, 1e11], "fent": [1.01e11, 1.01e11]}, measure="total_time"),
    )
    ranking = c.rank_by_difference(obs, GROUPS, draws=20)

    by_raw = sorted(
        ranking.contrasts, key=lambda i: abs(i.difference or 0.0), reverse=True
    )

    assert by_raw[0].measure_key[0] == "total_time"
    assert ranking.contrasts[0].measure_key[0] == "n_instances"


# --------------------------------------------------------------------------------------------- #
# the between null permutes units, not (unit, measure) rows
# --------------------------------------------------------------------------------------------- #

_UNITS = [f"u{i}" for i in range(8)]
_DRUG, _VEH = ("drug",), ("veh",)
_GROUP = {u: (_DRUG if i < 4 else _VEH) for i, u in enumerate(_UNITS)}
_TARGET = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 30.0]
#: Twelve measures with nothing in them, fixed so the guard does not move with numpy's stream.
_FLAT = {
    f"m{k}": [5.0 + ((7 * k + 3 * i) % 11) / 11.0 for i in range(8)] for k in range(12)
}


def _unit_obs(measures, group=None, units=None):
    """``{measure: [value per unit]}`` -> one observation per (unit, measure); ``None`` skips."""
    group = group or _GROUP
    units = units or _UNITS
    return [
        c.Observation(
            key=(u,),
            group=group[u],
            measure_key=(m, "x", "", "whole"),
            value=float(v),
            n_underlying=1,
        )
        for m, vals in measures.items()
        for u, v in zip(units, vals)
        if v is not None
    ]


def _item(ranking, name):
    return next(i for i in ranking.contrasts if i.measure_key[0] == name)


def test_a_measures_own_p_does_not_move_when_unrelated_measures_are_added():
    """With one unit → group map per draw the draws depend only on the units and the seed, so
    a measure's own null — and its own p — is the same draw for draw however many other
    measures sit beside it. Shuffled per row, this was 0.0314 alone and 0.0574 with twelve.
    """
    alone = c.rank_by_difference(
        _unit_obs({"t": _TARGET}), (_DRUG, _VEH), draws=2000, seed=0
    )
    crowded = c.rank_by_difference(
        _unit_obs({"t": _TARGET, **_FLAT}), (_DRUG, _VEH), draws=2000, seed=0
    )
    assert _item(crowded, "t").own_p == _item(alone, "t").own_p


def test_a_copied_measure_moves_with_its_original_in_every_draw():
    """A unit's measures move together, which is what the family-wise maximum assumes: a measure
    and its exact copy have the same magnitude in every draw, so the family's largest is that
    measure's own null and the family p is the one it has alone.

    Not ``family_p == own_p``: the two are read by different conventions (the percentile rule
    the floor uses, and ``(k + 1) / (n + 1)``), which differ by a draw or two even for a
    measure ranked alone."""
    alone = c.rank_by_difference(
        _unit_obs({"t": _TARGET}), (_DRUG, _VEH), draws=2000, seed=0
    )
    copied = c.rank_by_difference(
        _unit_obs({"t": _TARGET, "t2": _TARGET}), (_DRUG, _VEH), draws=2000, seed=0
    )
    a, b = _item(copied, "t"), _item(copied, "t2")
    assert a.own_p == b.own_p == _item(alone, "t").own_p
    assert a.family_p == b.family_p == _item(alone, "t").family_p
    assert abs(a.family_p - a.own_p) <= 2 / 2000


def test_every_draw_keeps_each_measures_real_group_sizes(monkeypatch):
    """4 against 4 stays 4 against 4 for every measure in every draw, and a measure a unit has
    no value for keeps the sizes it really has. Per row, a measure was handed 0 to 8 "drug"
    animals and was not 4 in most draws."""
    sizes: list[dict] = []
    real = c._magnitudes

    def spy(keys, labels, values, groups, epsilon):
        per: dict = {}
        for key, label in zip(keys, labels):
            per.setdefault(key[0], {}).setdefault(label, 0)
            per[key[0]][label] += 1
        sizes.append(per)
        return real(keys, labels, values, groups, epsilon)

    monkeypatch.setattr(c, "_magnitudes", spy)
    missing_u0 = [None] + _TARGET[1:]
    c.rank_by_difference(
        _unit_obs({"t": _TARGET, "gap": missing_u0, **_FLAT}),
        (_DRUG, _VEH),
        draws=300,
        seed=0,
    )
    assert len(sizes) == 300
    for per in sizes:
        for name, counts in per.items():
            if name == "gap":
                assert sorted(counts.values()) == [3, 4], counts
            else:
                assert counts == {_DRUG: 4, _VEH: 4}, (name, counts)
    assert (
        len({tuple(sorted(p["gap"].items())) for p in sizes}) == 2
    ), "the unit missing a measure lands in either group across draws"


def test_the_between_own_p_matches_the_exact_unit_permutation_p():
    """The exact p for the target, counted over all 70 ways to split eight animals four and
    four (0.0286), and the sampled own p within sampling error of it (three standard errors at
    5000 draws). Shuffled per row, the same measure beside twelve flat ones read 0.0574.
    """
    import itertools

    def magnitude(drug_idx):
        drug = [_TARGET[i] for i in drug_idx]
        veh = [_TARGET[i] for i in range(8) if i not in drug_idx]
        a, b = sum(drug) / 4, sum(veh) / 4
        return abs(math.log(a / b))

    observed = magnitude((0, 1, 2, 3))
    splits = list(itertools.combinations(range(8), 4))
    exact = sum(magnitude(s) >= observed * (1 - 1e-12) for s in splits) / len(splits)
    assert round(exact, 4) == 0.0286
    ranking = c.rank_by_difference(
        _unit_obs({"t": _TARGET, **_FLAT}), (_DRUG, _VEH), draws=5000, seed=0
    )
    se = math.sqrt(exact * (1 - exact) / 5000)
    assert abs(_item(ranking, "t").own_p - exact) < 3 * se


def test_units_outside_the_contrasted_groups_stay_out_of_the_null():
    """A record's levels can pick two of three groups; the third group's animals are not in the
    contrast, so they are not in its null either — the p is the one the two groups give alone.
    """
    units = _UNITS + ["w0", "w1", "w2", "w3"]
    group = {**_GROUP, **{u: ("other",) for u in units[8:]}}
    two = c.rank_by_difference(
        _unit_obs({"t": _TARGET}), (_DRUG, _VEH), draws=2000, seed=0
    )
    with_third = c.rank_by_difference(
        _unit_obs({"t": _TARGET + [50.0, 60.0, 70.0, 80.0]}, group=group, units=units),
        (_DRUG, _VEH),
        draws=2000,
        seed=0,
    )
    assert _item(with_third, "t").own_p == _item(two, "t").own_p


def test_a_unit_in_two_groups_is_refused_in_a_between_design():
    obs = _unit_obs({"t": _TARGET}) + [
        c.Observation(
            key=("u0",),
            group=_VEH,
            measure_key=("other", "x", "", "whole"),
            value=1.0,
            n_underlying=1,
        )
    ]
    with pytest.raises(c.ComparisonError, match="is in two groups"):
        c.rank_by_difference(obs, (_DRUG, _VEH), draws=10, seed=0)
