"""The designs a comparison resolves to, and the null each one is allowed (§24 *Designs*).

**The class: a floor from a null that ignores the design.** A within-animal contrast — vehicle
against drug in the same ten animals — is a different experiment from ten vehicle animals against
ten drug animals, and the floor that calibrates it has to come from the same experiment: flip each
animal's own direction, or shuffle each animal's sessions within that animal. Relabelling the
sessions *across* animals breaks the pairing the design exists to use: an animal dealt two of one
level has no pair, each draw's statistic runs over a shrinking subset, the null's tail widens and
the floor rises past an effect every animal showed. The difference screen's pre-build study (kept outside the public tree) measured this on ~1,350
clustering nodes: the
between-unit fold change could clear nothing at a log ratio above 6, while the same nodes under a
within-animal contrast cleared a floor the shuffled control could not.

**The second class: an effect over absent animals.** A node three of eight rats ever showed has
a mean log ratio over eight, five of them zero, and the trend promoted one-to-three-minute nodes
on exactly that. An effect below ``min_units`` *present* units is undefined, never large.

**The third: a floor that moves with its seed.** With eight units the sign-flip null has 256
configurations; a sampled 95th percentile moved three nodes in and out of a kept set with its
seed. Ten or fewer units enumerate every flip.

**The fourth: a search the floor did not pay for.** A profile over three or more levels ranks
each measure by the largest of its pairwise contrasts, which is a search over pairs; a null that
took one fixed pair would set a floor the real search clears by luck. The null takes the same
largest pair under each within-unit shuffle, and the no-signal cohort shows it keeps nothing.

**The fifth: one cell, read two ways.** The design stripped a padded ``"fent "`` while the
observations grouped it raw, so a paired contrast resolved to ``fent`` and then found no group
by that name and ranked nothing; ``NaN`` was no value to the design and a value to the subject
inference. Every reader of a grouping cell reads it through ``cohort.cell`` (stripped, ``""``
when missing).

**The sixth: the observed value beating itself.** An enumerated null contains the observed
configuration, so its own p can never be 0. It came out 0 anyway: the null recomputes that
configuration as a matrix product where the effect is a ``nansum``, the two round differently,
and the draw that *is* the observed labelling landed one float step short of it. Compared with
an exact ``>=`` it reached nothing; compared with ``>``, the observed value cleared a floor that
was itself. Every comparison of an observed value against its null now goes through
``compare._reaches``, which allows for that rounding step.

Every guard here is shown to fail: the wrong null is run beside the right one, the planted node
is planted, the seed is varied.
"""

import math

import numpy as np
import pytest

from detail.analysis import compare as c

NS = 1_000_000_000
HOUR = 3600 * NS


def _rows(recording, label, share, denominator_ns=HOUR):
    """A `time_fraction` row and its `total_time` row, as the measures table emits them."""
    base = {
        "recording_id": recording,
        "subject": 0,
        "window_id": "whole",
        "label": label,
        "label_to": "",
        "denominator_ns": denominator_ns,
    }
    return [
        {**base, "measure": "time_fraction", "value": share},
        {**base, "measure": "total_time", "value": share * denominator_ns},
    ]


def _crossover(shares: dict, *, unit="animal", column="session"):
    """``{(animal, session): {label: share}}`` -> metadata, rows."""
    meta, rows = {}, []
    for i, ((animal, session), labels) in enumerate(shares.items()):
        recording = f"rec{i:03d}"
        meta[recording] = {unit: animal, column: session}
        for label, share in labels.items():
            rows += _rows(recording, label, share)
    return meta, rows


def _pseudo(seconds=1.0):
    def pseudo(measure_key, denominator_ns):
        if measure_key[0] == "time_fraction":
            return seconds * NS / denominator_ns if denominator_ns else 0.0
        return seconds * NS

    return pseudo


# --------------------------------------------------------------------------- #
# resolution: the user chooses the column, the tool says what it implies
# --------------------------------------------------------------------------- #


def _dose_cohort(n_animals=4, doses=("0", "25", "50", "75")):
    meta = {}
    for a in range(n_animals):
        for d in doses:
            meta[f"r{a}_{d}"] = {
                "animal": f"A{a}",
                "dose": d,
                "sex": "F" if a % 2 else "M",
            }
        meta[f"r{a}_nalox"] = {
            "animal": f"A{a}",
            "dose": "",
            "sex": "F" if a % 2 else "M",
        }
    return meta


def test_a_numeric_column_that_varies_within_units_is_a_trend():
    meta = _dose_cohort()
    design = c.resolve_design(
        meta, list(meta), group_by=["dose"], aggregate_to=["animal"]
    )
    assert design.kind == c.TREND
    assert design.column == "dose"
    assert "trend over 0, 25, 50, 75" in design.reason


def test_a_blank_value_sits_outside_the_trend():
    """An antagonist session has no dose. It is not a dose of zero, and it does not make the column
    text."""
    meta = _dose_cohort()
    design = c.resolve_design(
        meta, list(meta), group_by=["dose"], aggregate_to=["animal"]
    )
    assert design.kind == c.TREND
    assert "nalox" not in design.reason


def test_two_values_within_units_is_a_paired_contrast():
    meta, _ = _crossover(
        {("A", "veh"): {}, ("A", "fent"): {}, ("B", "veh"): {}, ("B", "fent"): {}}
    )
    design = c.resolve_design(
        meta, list(meta), group_by=["session"], aggregate_to=["animal"]
    )
    assert design.kind == c.PAIRED
    assert design.levels == ("fent", "veh")  # sorted; read as veh against fent
    assert "paired within animal" in design.reason


def test_a_constant_column_is_a_between_design():
    meta = _dose_cohort()
    design = c.resolve_design(
        meta, list(meta), group_by=["sex"], aggregate_to=["animal"]
    )
    assert design.kind == c.BETWEEN
    assert "constant within each animal" in design.reason


def test_no_unit_is_a_between_design_whatever_the_column_does():
    meta = _dose_cohort()
    design = c.resolve_design(meta, list(meta), group_by=["dose"], aggregate_to=[])
    assert design.kind == c.BETWEEN
    assert "each recording is one observation" in design.reason


def test_levels_on_a_numeric_column_contrast_those_two_doses():
    """The one lever a user keeps: naming two values turns the trend into a pair."""
    meta = _dose_cohort()
    design = c.resolve_design(
        meta, list(meta), group_by=["dose"], aggregate_to=["animal"], levels=["0", "75"]
    )
    assert design.kind == c.PAIRED
    assert design.levels == ("0", "75")


def test_three_or_more_text_values_within_units_is_a_profile_naming_them():
    """Three sessions within each animal are not a refusal to pick two of: the design is a
    profile over all of them, and naming two is still the lever that contrasts just those.
    """
    meta = {}
    for a in ("A", "B"):
        for s in ("saline", "fent", "nalox"):
            meta[f"{a}_{s}"] = {"animal": a, "session": s}
    design = c.resolve_design(
        meta, list(meta), group_by=["session"], aggregate_to=["animal"]
    )
    assert design.kind == c.PROFILE
    assert design.levels is None and design.column == "session"
    assert "fent, nalox, saline" in design.reason
    assert "largest difference between any two" in design.reason
    paired = c.resolve_design(
        meta,
        list(meta),
        group_by=["session"],
        aggregate_to=["animal"],
        levels=["saline", "fent"],
    )
    assert paired.kind == c.PAIRED and paired.levels == ("saline", "fent")


def test_a_level_the_column_does_not_hold_is_refused():
    meta = _dose_cohort()
    with pytest.raises(c.ComparisonError, match="not a value of 'dose'"):
        c.resolve_design(
            meta,
            list(meta),
            group_by=["dose"],
            aggregate_to=["animal"],
            levels=["0", "99"],
        )


def test_levels_need_one_grouping_column():
    meta = _dose_cohort()
    with pytest.raises(c.ComparisonError, match="one column"):
        c.resolve_design(
            meta,
            list(meta),
            group_by=["dose", "sex"],
            aggregate_to=["animal"],
            levels=["0", "75"],
        )


def test_the_rows_and_the_metadata_resolve_the_same_design():
    """One rule, two front doors: the launch dialog previews from `metadata.csv`, the run
    resolves from the measure rows, and they must say the same thing."""
    meta, rows = _crossover(
        {
            ("A", "veh"): {"rearing": 0.1},
            ("A", "fent"): {"rearing": 0.05},
            ("B", "veh"): {"rearing": 0.2},
            ("B", "fent"): {"rearing": 0.1},
        }
    )
    from_meta = c.resolve_design(
        meta, list(meta), group_by=["session"], aggregate_to=["animal"]
    )
    from_rows = c.resolve_design_from_rows(
        rows, meta, group_by=["session"], aggregate_to=["animal"]
    )
    assert from_rows == from_meta


# --------------------------------------------------------------------------- #
# the effects, by hand
# --------------------------------------------------------------------------- #


def _paired_ranking(shares, **kwargs):
    meta, rows = _crossover(shares)
    design = c.resolve_design(
        meta,
        list(meta),
        group_by=["session"],
        aggregate_to=["animal"],
        levels=kwargs.pop("levels", None),
    )
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    defaults = dict(
        draws=200,
        seed=0,
        min_units=1,
        presence_ns=0,
        pseudo=_pseudo(0.0),
        ranking_measure="time_fraction",
    )
    return c.rank_by_design(obs, design, **{**defaults, **kwargs})


def test_a_paired_effect_is_the_mean_per_unit_log_ratio():
    """Not the log ratio of the group means: each animal votes once, with its own ratio."""
    ranking = _paired_ranking(
        {
            ("A", "veh"): {"x": 0.10},
            ("A", "fent"): {"x": 0.20},
            ("B", "veh"): {"x": 0.30},
            ("B", "fent"): {"x": 0.30},
            ("C", "veh"): {"x": 0.05},
            ("C", "fent"): {"x": 0.40},
        },
        levels=["veh", "fent"],
    )
    (item,) = ranking.contrasts
    expected = (math.log(2.0) + math.log(1.0) + math.log(8.0)) / 3
    assert item.effect == pytest.approx(expected)
    assert item.n_units == 3
    assert (
        item.consistency == 2
    )  # B moved nowhere and does not count in either direction
    assert item.a.mean == pytest.approx(0.15) and item.b.mean == pytest.approx(0.30)


def test_a_trend_effect_is_the_mean_per_unit_spearman():
    meta, rows = {}, []
    monotone = {"0": 0.4, "25": 0.3, "50": 0.2, "75": 0.1}
    flat = {"0": 0.2, "25": 0.2, "50": 0.2, "75": 0.2}
    for animal, profile in (("A", monotone), ("B", monotone), ("C", flat)):
        for dose, share in profile.items():
            rid = f"{animal}_{dose}"
            meta[rid] = {"animal": animal, "dose": dose}
            rows += _rows(rid, "x", share)
    design = c.resolve_design(
        meta, list(meta), group_by=["dose"], aggregate_to=["animal"]
    )
    obs = c.observations(rows, meta, group_by=["dose"], aggregate_to=["animal"])
    ranking = c.rank_by_design(
        obs, design, draws=100, seed=0, min_units=1, ranking_measure="time_fraction"
    )
    (item,) = ranking.contrasts
    assert ranking.design == c.TREND
    assert item.effect == pytest.approx((-1.0 - 1.0 + 0.0) / 3)
    assert item.consistency == 2


# --------------------------------------------------------------------------- #
# the null respects the design
# --------------------------------------------------------------------------- #


def _baseline_spread_cohort(seed=0, n=8, effect=0.5):
    """Animals that differ a hundredfold at baseline, each shifted by `effect` under the drug."""
    rng = np.random.default_rng(seed)
    shares = {}
    for i in range(n):
        base = 0.002 * (10 ** rng.uniform(0, 2))
        shares[(f"A{i}", "veh")] = {"x": base}
        shares[(f"A{i}", "fent")] = {"x": base * math.exp(effect + rng.normal(0, 0.1))}
    return shares


def _across_unit_floor(obs, levels, draws=1000, seed=0):
    """The wrong null: the level labels shuffled across animals, which is what the between
    permutation does. A pair survives only where an animal happens to keep one of each level,
    so each draw's statistic is a sign flip over a shrinking subset of the animals."""
    rng = np.random.default_rng(seed)
    values = [o.value for o in obs]
    units = [o.key for o in obs]
    labels = [o.group[0] for o in obs]
    worst = []
    for _ in range(draws):
        dealt = rng.permutation(labels)
        per_unit: dict = {}
        for unit, label, value in zip(units, dealt, values):
            per_unit.setdefault(unit, {})[label] = value
        ratios = [
            math.log(have[levels[1]] / have[levels[0]])
            for have in per_unit.values()
            if set(levels) <= set(have)
        ]
        worst.append(abs(float(np.mean(ratios))) if ratios else 0.0)
    worst.sort()
    return worst[int(round(0.95 * (len(worst) - 1)))]


def test_the_permutation_respects_the_design():
    """Sign flips within animals, never a relabelling across them. Relabelling breaks the pairs:
    an animal dealt two of one level drops out of that draw, the null is a sign flip over a
    shrinking subset, its tail widens, and the floor rises past an effect every animal showed.
    §24's "the permutation respects the path", one level down.
    """
    shares = _baseline_spread_cohort()
    meta, rows = _crossover(shares)
    design = c.resolve_design(
        meta,
        list(meta),
        group_by=["session"],
        aggregate_to=["animal"],
        levels=["veh", "fent"],
    )
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    right = c.rank_by_design(
        obs, design, min_units=1, pseudo=_pseudo(0.0), ranking_measure="time_fraction"
    )
    wrong_floor = _across_unit_floor(
        [o for o in obs if o.measure_key[0] == "time_fraction"], design.levels
    )
    (item,) = right.contrasts
    assert right.exact and right.floor is not None
    assert right.clears_floor(item), (item.effect, right.floor)
    assert wrong_floor > 1.25 * right.floor, (wrong_floor, right.floor)
    assert (
        item.magnitude < wrong_floor
    ), "the across-animal null would have hidden the effect"


def test_the_negative_control_as_a_test():
    """A cohort with no signal keeps nothing in at least 19 of 20 seeds, and one planted node
    clears in every seed — the control Phase 0 ran on shuffled sessions, as a synthetic cohort
    whose sessions really are exchangeable."""
    kept_noise, planted_cleared = 0, 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        shares = {}
        for i in range(8):
            for session in ("veh", "fent"):
                labels = {f"n{k}": 0.05 * (10 ** rng.normal(0, 0.1)) for k in range(60)}
                labels["planted"] = 0.05 * (3.0 if session == "fent" else 1.0)
                shares[(f"A{i}", session)] = labels
        meta, rows = _crossover(shares)
        design = c.resolve_design(
            meta,
            list(meta),
            group_by=["session"],
            aggregate_to=["animal"],
            levels=["veh", "fent"],
        )
        obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
        ranking = c.rank_by_design(
            obs,
            design,
            min_units=4,
            presence_ns=NS,
            pseudo=_pseudo(),
            ranking_measure="time_fraction",
        )
        clearing = {item.measure_key[1] for item in ranking.clearing()}
        planted_cleared += "planted" in clearing
        kept_noise += bool(clearing - {"planted"})
    assert planted_cleared == 20
    assert kept_noise <= 1, f"noise cleared the floor in {kept_noise} of 20 seeds"


def test_the_exact_floor_does_not_move_with_its_seed():
    shares = _baseline_spread_cohort(n=8)
    a = _paired_ranking(shares, levels=["veh", "fent"], seed=0)
    b = _paired_ranking(shares, levels=["veh", "fent"], seed=7)
    assert a.exact and b.exact
    assert a.draws == 256 and a.floor == b.floor


def test_a_larger_cohort_samples_its_floor_and_says_so():
    shares = _baseline_spread_cohort(n=12)
    ranking = _paired_ranking(shares, levels=["veh", "fent"], draws=300)
    assert not ranking.exact and ranking.draws == 300


# --------------------------------------------------------------------------- #
# min_units, presence, the undefined
# --------------------------------------------------------------------------- #


def test_min_units_leaves_an_effect_undefined_not_large():
    """A node two of eight animals ever showed, with a huge ratio in those two, ranks last with
    no effect — not first with the largest."""
    shares = {}
    for i in range(8):
        present = i < 2
        shares[(f"A{i}", "veh")] = {"rare": 0.001 if present else 0.0, "common": 0.1}
        shares[(f"A{i}", "fent")] = {"rare": 0.1 if present else 0.0, "common": 0.12}
    ranking = _paired_ranking(
        shares, levels=["veh", "fent"], min_units=4, presence_ns=NS, pseudo=_pseudo()
    )
    by_label = {item.measure_key[1]: item for item in ranking.contrasts}
    assert by_label["rare"].effect is None
    assert by_label["rare"].n_present == 2
    assert by_label["common"].effect is not None
    assert ranking.contrasts[0].measure_key[1] == "common"
    assert "undefined" not in (ranking.no_floor_reason or "")


def test_presence_counts_animals_with_the_behavior_not_animals_with_a_value():
    """Every animal has a value (a real zero); only those over `presence_s` are present."""
    shares = {}
    for i in range(6):
        shares[(f"A{i}", "veh")] = {"x": 0.0}
        shares[(f"A{i}", "fent")] = {"x": 0.5 if i < 3 else 0.0}
    ranking = _paired_ranking(
        shares, levels=["veh", "fent"], min_units=1, presence_ns=NS, pseudo=_pseudo()
    )
    (item,) = ranking.contrasts
    assert item.n_units == 6 and item.n_present == 3
    assert (item.present_a, item.present_b) == (0, 3)


def test_one_unit_gets_no_floor_and_a_reason():
    ranking = _paired_ranking(
        {("A", "veh"): {"x": 0.1}, ("A", "fent"): {"x": 0.2}}, levels=["veh", "fent"]
    )
    assert ranking.floor is None
    assert ranking.no_floor_reason == (
        "only one animal (A) has both veh and fent, so a within-animal null has nothing "
        "to move and no floor can be computed"
    )
    assert not ranking.clears_floor(ranking.contrasts[0])


def test_a_unit_at_one_level_cannot_be_flipped_so_two_such_and_one_is_one():
    """Two animals, one of them with a single session: every flip of that one is the
    identity, so the null's maximum was the observed maximum, which nothing can exceed.
    The floor is withheld with the reason, naming the one animal that can move."""
    ranking = _paired_ranking(
        {
            ("A", "veh"): {"x": 0.1},
            ("A", "fent"): {"x": 0.2},
            ("B", "veh"): {"x": 0.3},
        },
        levels=["veh", "fent"],
    )
    assert ranking.floor is None
    assert ranking.no_floor_reason.startswith(
        "only one animal (A) has both veh and fent"
    )
    assert not ranking.clears_floor(ranking.contrasts[0])


def test_nothing_defined_gets_no_floor_and_says_why():
    shares = {}
    for i in range(4):
        shares[(f"A{i}", "veh")] = {"x": 0.0}
        shares[(f"A{i}", "fent")] = {"x": 0.0}
    ranking = _paired_ranking(
        shares, levels=["veh", "fent"], min_units=2, presence_ns=NS, pseudo=_pseudo()
    )
    assert ranking.floor is None
    assert "undefined" in ranking.no_floor_reason


# --------------------------------------------------------------------------- #
# nesting suppression, and the node source's pseudo-count
# --------------------------------------------------------------------------- #


def test_suppression_is_a_peak_pick():
    """Over a chain root > parent > child > grandchild all clearing, one is kept — the best — and
    a sibling branch keeps its own."""
    shares = {}
    for i in range(8):
        for session, k in (("veh", 1.0), ("fent", 3.0)):
            shares[(f"A{i}", session)] = {
                "parent": 0.10 * k,
                "child": 0.08 * k,
                "grandchild": 0.05 * (k * 1.2 if k > 1 else 1.0),
                "sibling": 0.02 * k,
            }
    ranking = _paired_ranking(shares, levels=["veh", "fent"], pseudo=_pseudo(0.0))
    assert len(ranking.clearing()) == 4
    ancestors = {
        "parent": set(),
        "child": {"parent"},
        "grandchild": {"parent", "child"},
        "sibling": set(),
    }
    kept = c.suppress_nesting(ranking, ancestors)
    labels = [k[1] for k in kept.kept]
    assert "grandchild" in labels, "the best of the chain is kept"
    assert "sibling" in labels and len(labels) == 2
    for a in labels:
        for b in labels:
            assert a == b or a not in ancestors[b]
    assert [i.measure_key[1] for i in kept.kept_items()] == labels
    assert [i.measure_key[1] for i in c.below_floor(kept, ancestors, 5)] == []


def test_a_label_source_keeps_appearance_unbounded_and_a_node_source_does_not():
    """A behavior that appears is reported as itself over labels; over the node source the
    pseudo-count makes it a finite ratio that cannot own the family-wise floor."""
    meta, rows = _crossover(
        {
            ("A", "veh"): {"x": 0.0},
            ("B", "veh"): {"x": 0.0},
            ("C", "fent"): {"x": 0.2},
            ("D", "fent"): {"x": 0.3},
        }
    )
    obs = [
        o
        for o in c.observations(rows, meta, group_by=["session"], aggregate_to=[])
        if o.measure_key[0] == "time_fraction"
    ]
    groups = (("fent",), ("veh",))
    plain = c.rank_by_difference(obs, groups, draws=50)
    shifted = c.rank_by_difference(obs, groups, draws=50, pseudo=_pseudo())
    assert plain.contrasts[0].unbounded and plain.contrasts[0].magnitude is None
    assert not shifted.contrasts[0].unbounded
    assert shifted.contrasts[0].magnitude is not None
    assert (
        shifted.contrasts[0].a.mean == plain.contrasts[0].a.mean
    ), "displayed means stay true"


def test_the_record_rebuilds_the_same_ranking_from_the_stored_rows():
    """`stats.csv` round trip: what `stats_rows` writes carries what `rank_by_design` needs."""
    from detail.analysis import comparison_report

    shares = _baseline_spread_cohort(n=6)
    meta, rows = _crossover(shares)
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    stored = c.stats_rows(obs, aggregate_to=["animal"], group_by=["session"])
    assert all(row["denominator_ns"] == HOUR for row in stored)
    rebuilt = comparison_report._observations(
        [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in stored]
    )
    record = {
        "design": c.PAIRED,
        "design_column": "session",
        "levels": ["veh", "fent"],
        "aggregate_to": ["animal"],
        "min_units": 3,
        "presence_s": 1.0,
        "pseudo_count_s": 1.0,
        "ranking_measure": "time_fraction",
        "seed": 0,
        "draws": 100,
    }
    first = comparison_report.rank_record(obs, record)
    second = comparison_report.rank_record(rebuilt, record)
    assert first.floor == second.floor
    assert [i.effect for i in first.contrasts] == [i.effect for i in second.contrasts]


def test_a_missing_token_is_no_value_either():
    """``NA``, ``nan``, ``None``: what a spreadsheet or a data frame writes for an empty cell.
    Each sits outside the trend with the blank -- and ``nan`` in particular does not parse as a
    dose, which it would under a bare ``float()``."""
    meta = _dose_cohort()
    tokens = iter(["NA", "nan", "None", "n/a"])
    for key in [k for k in meta if k.endswith("_nalox")]:
        meta[key]["dose"] = next(tokens)
    design = c.resolve_design(
        meta, list(meta), group_by=["dose"], aggregate_to=["animal"]
    )
    assert design.kind == c.TREND
    assert "trend over 0, 25, 50, 75" in design.reason
    assert not c._is_number("nan")
    assert not c._is_number("NA")
    assert not c._is_number("inf")
    assert c._is_number("25")


# --------------------------------------------------------------------------- #
# the profile design: three or more levels, the largest pair (§24 *Designs*)
# --------------------------------------------------------------------------- #


def _profile_ranking(shares, **kwargs):
    meta, rows = _crossover(shares)
    design = c.resolve_design(
        meta, list(meta), group_by=["session"], aggregate_to=["animal"]
    )
    assert design.kind == c.PROFILE
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    defaults = dict(
        draws=200,
        seed=0,
        min_units=1,
        presence_ns=0,
        pseudo=_pseudo(0.0),
        ranking_measure="time_fraction",
    )
    return c.rank_by_design(obs, design, **{**defaults, **kwargs})


def test_a_profile_effect_is_the_largest_pairwise_contrast_with_its_pair():
    """Over fent, nalox and saline (level order), every pair gets the paired contrast and the
    measure carries the largest, read as the later level against the earlier."""
    ranking = _profile_ranking(
        {
            ("A", "saline"): {"x": 0.10},
            ("A", "fent"): {"x": 0.20},
            ("A", "nalox"): {"x": 0.05},
            ("B", "saline"): {"x": 0.30},
            ("B", "fent"): {"x": 0.30},
            ("B", "nalox"): {"x": 0.10},
            ("C", "saline"): {"x": 0.05},
            ("C", "fent"): {"x": 0.40},
            ("C", "nalox"): {"x": 0.04},
        }
    )
    (item,) = ranking.contrasts
    # fent -> nalox, the largest of the three pairs (fent -> saline is 0.92, nalox -> saline
    # 0.67 in |mean log ratio|).
    fent_nalox = (
        math.log(0.05 / 0.20) + math.log(0.10 / 0.30) + math.log(0.04 / 0.40)
    ) / 3
    assert item.pair == ("fent", "nalox")
    assert item.effect == pytest.approx(fent_nalox)
    assert item.consistency == 3 and item.n_units == 3
    assert item.a.group == ("fent",) and item.b.group == ("nalox",)
    assert ranking.design == c.PROFILE and ranking.profile
    assert ranking.level_values == ("fent", "nalox", "saline")
    assert ranking.levels is None
    assert not ranking.exact, "(k!)^n configurations are sampled, never enumerated"


def test_a_profile_over_two_levels_is_the_paired_contrast():
    """With two levels there is one pair, so the profile's items are the paired design's, item
    for item -- the two-level designs are the profile's special case, and nothing saved changes.
    """
    shares = _baseline_spread_cohort(n=6)
    meta, rows = _crossover(shares)
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    paired = c.resolve_design(
        meta, list(meta), group_by=["session"], aggregate_to=["animal"]
    )
    assert paired.kind == c.PAIRED and paired.levels == ("fent", "veh")
    forced = c.Design(c.PROFILE, "session", None, ("animal",), "forced")
    kwargs = dict(min_units=1, pseudo=_pseudo(0.0), ranking_measure="time_fraction")
    a = c.rank_by_design(obs, paired, **kwargs)
    b = c.rank_by_design(obs, forced, **kwargs)
    assert [i.effect for i in a.contrasts] == pytest.approx(
        [i.effect for i in b.contrasts]
    )
    assert [i.pair for i in a.contrasts] == [i.pair for i in b.contrasts]
    assert [i.pair for i in b.contrasts] == [("fent", "veh")]
    assert [i.consistency for i in a.contrasts] == [i.consistency for i in b.contrasts]
    assert [i.n_present for i in a.contrasts] == [i.n_present for i in b.contrasts]


def test_the_profile_null_takes_the_same_statistic():
    """A cohort with no signal over three sessions keeps nothing in at least 19 of 20 seeds,
    and one node planted in one session clears in every seed: the null's largest pair is the
    same search the ranking ran, so choosing the pair is already paid for."""
    kept_noise, planted_cleared = 0, 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        shares = {}
        for i in range(8):
            for session in ("veh", "fent", "nalox"):
                labels = {f"n{k}": 0.05 * (10 ** rng.normal(0, 0.1)) for k in range(60)}
                labels["planted"] = 0.05 * (3.0 if session == "nalox" else 1.0)
                shares[(f"A{i}", session)] = labels
        ranking = _profile_ranking(
            shares, seed=seed, min_units=4, presence_ns=NS, pseudo=_pseudo()
        )
        clearing = {item.measure_key[1] for item in ranking.clearing()}
        planted_cleared += "planted" in clearing
        kept_noise += bool(clearing - {"planted"})
        planted = next(i for i in ranking.contrasts if i.measure_key[1] == "planted")
        assert planted.pair in {("fent", "nalox"), ("nalox", "veh")}
    assert planted_cleared == 20
    assert kept_noise <= 1, f"noise cleared the floor in {kept_noise} of 20 seeds"


def test_a_profile_pair_is_defined_only_with_enough_present_units():
    """A node two of eight animals ever show, under one session only: no pair reaches
    `min_units`, so the measure is undefined with no pair, never the largest."""
    shares = {}
    for i in range(8):
        present = i < 2
        for session in ("veh", "fent", "nalox"):
            shares[(f"A{i}", session)] = {
                "rare": 0.1 if (present and session == "nalox") else 0.0,
                "common": {"veh": 0.10, "fent": 0.12, "nalox": 0.11}[session],
            }
    ranking = _profile_ranking(shares, min_units=4, presence_ns=NS, pseudo=_pseudo())
    by_label = {item.measure_key[1]: item for item in ranking.contrasts}
    assert by_label["rare"].effect is None and by_label["rare"].pair is None
    assert by_label["rare"].n_present == 2
    assert by_label["common"].effect is not None
    assert by_label["common"].pair is not None
    assert ranking.contrasts[0].measure_key[1] == "common"


def test_a_between_design_over_three_groups_ranks_the_largest_gap():
    """Three strains, each animal in one: the item is the largest contrast between any two
    groups with the pair on it, and two groups are the one contrast there is, unchanged.
    """
    meta, rows = {}, []
    for strain, share in (("a", 0.10), ("b", 0.12), ("c", 0.30)):
        for i in range(4):
            rid = f"{strain}{i}"
            meta[rid] = {"animal": rid, "strain": strain}
            rows += _rows(rid, "x", share * (1 + 0.01 * i))
    design = c.resolve_design(
        meta, list(meta), group_by=["strain"], aggregate_to=["animal"]
    )
    assert design.kind == c.BETWEEN
    obs = [
        o
        for o in c.observations(
            rows, meta, group_by=["strain"], aggregate_to=["animal"]
        )
        if o.measure_key[0] == "time_fraction"
    ]
    ranking = c.rank_by_difference(obs, (("a",), ("b",), ("c",)), draws=200)
    (item,) = ranking.contrasts
    assert ranking.profile and ranking.level_values == ("a", "b", "c")
    assert ranking.levels is None, "no single pair names the whole ranking"
    assert item.pair == ("a", "c")
    assert item.magnitude == pytest.approx(math.log(item.b.mean / item.a.mean))
    assert item.magnitude == pytest.approx(math.log(3.0), abs=0.05)
    two = c.rank_by_difference(obs, (("a",), ("c",)), draws=200)
    assert two.levels == ("a", "c") and not two.profile
    assert two.contrasts[0].magnitude == pytest.approx(item.magnitude)
    assert two.contrasts[0].pair == ("a", "c")


def test_a_padded_cell_is_its_level_in_the_design_and_the_groups_alike():
    """``"fent "`` from a spreadsheet: the design read it stripped and the observations read
    it raw, so the paired contrast resolved to ``fent`` and then found no group by that name
    and ranked nothing. Every reader goes through ``cohort.cell``."""
    shares = {
        ("A", "veh"): {"x": 0.10},
        ("A", "fent"): {"x": 0.20},
        ("B", "veh"): {"x": 0.30},
        ("B", "fent"): {"x": 0.30},
        ("C", "veh"): {"x": 0.05},
        ("C", "fent"): {"x": 0.40},
    }
    clean = _paired_ranking(shares, levels=["veh", "fent"])
    meta, rows = _crossover(shares)
    for row in meta.values():
        row["session"] = f" {row['session']} "
        row["animal"] = f"{row['animal']} "
    design = c.resolve_design_from_rows(
        rows,
        meta,
        group_by=["session"],
        aggregate_to=["animal"],
        levels=["veh", "fent"],
    )
    assert design.kind == c.PAIRED and design.levels == ("veh", "fent")
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    assert {o.group for o in obs} == {("veh",), ("fent",)}
    padded = c.rank_by_design(
        obs,
        design,
        draws=200,
        seed=0,
        min_units=1,
        presence_ns=0,
        pseudo=_pseudo(0.0),
        ranking_measure="time_fraction",
    )
    (item,) = padded.contrasts
    assert item.effect == pytest.approx(clean.contrasts[0].effect)
    assert item.n_units == 3


def test_a_missing_token_is_no_value_in_the_groups_and_the_subject_inference():
    """``NaN`` was a level of its own in the observations and a value in the subject
    inference while the design and the left-out rule read it as no value at all."""
    from detail.analysis import cohort

    meta, rows = _crossover(
        {
            ("A", "veh"): {"x": 0.1},
            ("A", "fent"): {"x": 0.2},
            ("B", "veh"): {"x": 0.3},
            ("B", "fent"): {"x": 0.3},
        }
    )
    meta["rec003"]["session"] = "NaN"
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    assert ("NaN",) not in {o.group for o in obs}
    # `sex` determines nothing once NaN is no value: A is f twice, B is m once and NaN once.
    for recording, sex in zip(sorted(meta), ["f", "f", "m", "NaN"]):
        meta[recording]["sex"] = sex
    assert cohort._determines(meta, list(meta), "animal", "sex")
    # A blank and a NaN are one no-value: animal A is not across two sessions, B is.
    meta["rec000"]["session"] = ""
    meta["rec001"]["session"] = "NaN"
    meta["rec003"]["session"] = "fent"
    assert cohort._crossing(meta, list(meta), "animal", ["session"]) == 1


@pytest.mark.parametrize(
    ("measure", "on"),
    [
        ("time_fraction", 30 * NS / HOUR),
        ("total_time", 30 * NS),
        ("duration_mean", 30 * NS),
        ("n_instances", 1.0),
        ("rate", NS / HOUR),
    ],
)
def test_a_zero_pseudo_count_adds_nothing_to_any_measure(measure, on):
    """``pseudo_count_s: 0`` is recorded as ranking raw ratios. A rate kept one bout per
    recording on both sides whatever the setting, so the record said 0 and the numbers did
    not."""
    from detail.analysis.comparison_report import pseudo_count

    key = (measure, "node:1", "", "whole")
    assert pseudo_count(0.0)(key, HOUR) == 0.0
    assert pseudo_count(30.0)(key, HOUR) == pytest.approx(on)


# --------------------------------------------------------------------------- #
# the p's: the floor read as a number, and a strictness as a lookup (§22, §24)
# --------------------------------------------------------------------------- #


def test_each_item_carries_its_family_wise_p_its_own_p_and_its_q_by_hand():
    """Three animals, eight sign flips, two measures. ``x`` doubles in every animal: its own
    null reaches |ln 2| in 2 of 8 flips (all the same sign), so its own p is 2/8. ``y``
    quadruples in two and quarters in the third: its mean |ln 4|/3 is reached by every flip,
    own p 1. The family's largest per flip is {4 × ln4/3, 2 × ln2, 2 × ln4} sorted, so
    ``x`` clears at the smallest level whose floor sits below ln 2 — one half, to the
    rule's rounding — and ``y`` at none. The q's are Benjamini–Hochberg over the own p's.
    """
    ranking = _paired_ranking(
        {
            ("A", "veh"): {"x": 0.10, "y": 0.10},
            ("A", "fent"): {"x": 0.20, "y": 0.40},
            ("B", "veh"): {"x": 0.10, "y": 0.10},
            ("B", "fent"): {"x": 0.20, "y": 0.40},
            ("C", "veh"): {"x": 0.10, "y": 0.40},
            ("C", "fent"): {"x": 0.20, "y": 0.10},
        },
        levels=["veh", "fent"],
    )
    assert ranking.exact and ranking.draws == 8
    x, y = ranking.contrasts
    assert x.measure_key[1] == "x" and y.measure_key[1] == "y"
    assert x.own_p == pytest.approx(2 / 8)
    assert y.own_p == pytest.approx(1.0)
    assert sorted(ranking.floor_distribution) == pytest.approx(
        [math.log(4) / 3] * 4 + [math.log(2)] * 2 + [math.log(4)] * 2
    )
    assert x.family_p == pytest.approx(0.5)
    assert y.family_p == pytest.approx(1.0)
    assert x.q == pytest.approx(0.5) and y.q == pytest.approx(1.0)
    # Read as the rule reads it: `family_p <= alpha` is `magnitude > floor_at(alpha)` at
    # every level, the floor itself the one at the ranking's own level.
    assert ranking.floor_at(c.FLOOR_ALPHA) == ranking.floor
    for alpha in (*c.STRICTNESS_LEVELS, 0.49, 0.51, 0.6, 0.99):
        for item in (x, y):
            assert ranking.clears_at(item, alpha) == (
                item.magnitude > ranking.floor_at(alpha)
            ), (item.measure_key, alpha)
    assert not ranking.clears_at(x, 0.5) and ranking.clears_at(x, 0.51)


def test_the_p_is_read_through_the_floors_own_rule_not_a_bare_count():
    """The percentile rule rounds its position in the null, so a bare count over the draws
    sits one draw off it at some levels: at 256 flips, 229.5 rounds down and a cluster
    reached by 26 of 256 maxima clears at p ≤ 0.1 by the rule while its count says 0.102.
    The family-wise p is the smallest level the rule clears at, so a slider's threshold and
    a hover's p agree on every cluster at every level (§22)."""
    rng = np.random.default_rng(3)
    mismatches = 0
    for n in (8, 16, 51, 200, 256, 1024):
        for trial in range(12):
            ordered = sorted(rng.exponential(size=n).tolist())
            if trial % 3 == 0:
                ordered = sorted(float(round(v, 1)) for v in ordered)
            magnitudes = rng.exponential(size=30).tolist() + [ordered[0], ordered[-1]]
            for magnitude in magnitudes:
                p = c._family_p(ordered, magnitude)
                for alpha in (*c.STRICTNESS_LEVELS, *rng.uniform(0.001, 0.999, 8)):
                    by_rule = c._percentile_floor(ordered, alpha) < magnitude
                    mismatches += by_rule != (p <= alpha)
    assert mismatches == 0
    # The knife edge itself: at 256 flips the rule's position at p ≤ 0.1 is 229.5, which
    # rounds to 230 (half to even), so a value reached by 26 maxima sits exactly on the floor
    # there and does not clear it; its p is a float step above 0.1, where a bare count says
    # 0.102 either way. Read as the slider reads it, the two agree: not shown at p ≤ 0.1,
    # shown a step above.
    ordered = sorted(rng.exponential(size=256).tolist())
    reached_by_26 = (ordered[-26] + ordered[-27]) / 2
    p = c._family_p(ordered, reached_by_26)
    assert 0.1 < p < 0.1 + 1e-12
    assert not c._percentile_floor(ordered, 0.1) < reached_by_26
    assert c._percentile_floor(ordered, p) < reached_by_26
    reached_by_25 = (ordered[-25] + ordered[-26]) / 2
    assert c._family_p(ordered, reached_by_25) < 0.1
    assert c._percentile_floor(ordered, 0.1) < reached_by_25
    assert c._family_p(ordered, ordered[0] / 2) == 1.0, "no draw falls short of it"
    assert c._family_p(ordered, None) is None and c._family_p([], 1.0) is None


def test_a_draw_a_rounding_step_short_still_reaches_the_observed_value():
    """The sixth class, on a literal: the observed configuration's own draw, recomputed one
    float step below the observed value. Every reader of the null counts it as reaching —
    the own p, the family-wise p and the floor verdict — so none of them can say the
    observed value beat itself."""
    observed = 0.04763724318519285
    own_draw = math.nextafter(observed, 0.0)
    null = np.array([own_draw, 0.01, 0.02, 0.03])
    assert c._share_reaching(null, observed, True) == pytest.approx(1 / 4)
    assert c._share_reaching(null, observed, False) == pytest.approx(2 / 5)
    assert c._family_p(sorted(null.tolist()), observed) > 0.0
    item = c.Effect(
        measure_key=("time_fraction", "x", "", "whole"),
        effect=observed,
        n_units=2,
        n_present=2,
        consistency=2,
        loo_sign_stable=True,
    )
    ranking = c.Ranking([item], floor=own_draw, draws=4, seed=0)
    assert not ranking.clears_floor(item)
    # And a value that really is larger still clears: the tolerance is a rounding step, not
    # a margin.
    assert c.Ranking([item], floor=observed * (1 - 1e-9), draws=4, seed=0).clears_floor(
        item
    )


def test_an_enumerated_own_p_is_never_below_its_two_reaching_flips():
    """The sixth class, through the real ranking. Over an enumerated sign-flip null the
    observed flips and their mirror image are two of the ``2^U`` draws and both reach the
    observed |effect|, so every own p is at least ``2 / 2^U``; with four units that is 1/8,
    so nothing can clear a 5% floor. The null's matrix product and the effect's ``nansum``
    round differently, and compared exactly the own p came out 0 on random shares."""
    rng = np.random.default_rng(0)
    for _ in range(60):
        shares = {
            (f"A{u}", session): {
                f"n{k}": float(rng.uniform(0.01, 0.5)) for k in range(6)
            }
            for u in range(4)
            for session in ("veh", "fent")
        }
        ranking = _paired_ranking(shares, levels=["veh", "fent"])
        assert ranking.exact and ranking.draws == 16
        for item in ranking.contrasts:
            assert item.own_p >= 2 / 16, (item.measure_key, item.own_p)
            assert item.family_p > c.FLOOR_ALPHA, (item.measure_key, item.family_p)
            assert not ranking.clears_floor(item), item.measure_key


def test_a_strictness_is_a_lookup_over_the_one_ranking():
    """``kept_at`` reads the saved null at another level — no re-ranking, the same peak pick
    — and is ``kept_items`` at the ranking's own level; looser is a superset of stricter;
    the floor at a level is the null's quantile there; the record's kept set does not move.
    """
    shares = {}
    for i in range(8):
        for session, k in (("veh", 1.0), ("fent", 3.0)):
            shares[(f"A{i}", session)] = {
                "parent": 0.10 * k,
                "child": 0.08 * k,
                "sibling": 0.02 * k,
                "lead": 0.03 * (1.6 if k > 1 and i < 6 else 1.0),
                "noise": 0.05 * (1.0 + 0.01 * i),
            }
    ranking = _paired_ranking(shares, levels=["veh", "fent"], pseudo=_pseudo(0.0))
    ancestors = {"parent": set(), "child": {"parent"}, "sibling": set(), "lead": set()}
    kept = c.suppress_nesting(ranking, ancestors)
    assert kept.ancestors is ancestors
    assert kept.kept_at(c.FLOOR_ALPHA) == kept.kept_items()
    labels = lambda items: [i.measure_key[1] for i in items]  # noqa: E731
    assert "lead" not in labels(kept.kept_items()) and "noise" not in labels(
        kept.kept_items()
    )
    assert "lead" in labels(kept.kept_at(0.5)), "looser shows the lead"
    for item in kept.contrasts:
        assert item.family_p is not None and item.own_p is not None
    lead = next(i for i in kept.contrasts if i.measure_key[1] == "lead")
    assert c.FLOOR_ALPHA < lead.family_p <= 0.5
    assert (
        lead.own_p < 0.05
    ), "consistent in its own animals, hidden by the family floor"
    previous: set = set()
    for alpha in c.STRICTNESS_LEVELS:
        now = set(labels(kept.clearing_at(alpha)))
        assert previous <= now, f"stricter is a subset of looser at {alpha}"
        previous = now
        picked = labels(kept.kept_at(alpha))
        for a in picked:
            for b in picked:
                assert a == b or a not in ancestors.get(
                    b, ()
                ), "peak-picked at every level"
        assert kept.floor_at(alpha) == c._percentile_floor(
            kept.floor_distribution, alpha
        )
    assert kept.kept == c.suppress_nesting(ranking, ancestors).kept
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        kept.floor_at(1.0)


def test_a_ranking_with_no_floor_has_no_p_and_clears_at_no_level():
    shares = {("A", "veh"): {"x": 0.1}, ("A", "fent"): {"x": 0.3}}
    ranking = _paired_ranking(shares, levels=["veh", "fent"])
    assert ranking.floor is None and ranking.no_floor_reason
    (item,) = ranking.contrasts
    assert item.family_p is None and item.own_p is None and item.q is None
    assert ranking.floor_at(0.5) is None
    assert not ranking.clears_at(item, 0.5) and ranking.kept_at(0.5) == []


# --------------------------------------------------------------------------- #
# the statistic: the moderated t scores by agreement, the effect by size (§24)
# --------------------------------------------------------------------------- #


def _reference_t(values, s0: float) -> float:
    """The moderated t written out: ``mean · √n / (sd + s0)``, the sample sd over the units
    with an effect (0 for one unit)."""
    import statistics

    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return mean * math.sqrt(n) / (sd + s0)


def _agreement_cohort():
    """``w`` has the larger mean |log ratio| (10×, 9×, ÷8 and 2× in four animals) and ``z``
    the smaller (1.8×, 2.2×, 2× and 2.1×, every animal agreeing)."""
    shares = {}
    for i, (zr, wr) in enumerate(zip((1.8, 2.2, 2.0, 2.1), (10.0, 9.0, 1 / 8, 2.0))):
        shares[(f"A{i}", "veh")] = {"z": 0.10, "w": 0.10}
        shares[(f"A{i}", "fent")] = {"z": 0.10 * zr, "w": 0.10 * wr}
    return shares


def test_the_moderated_t_by_hand_over_three_animals_and_eight_flips():
    """Three animals, two measures. ``x`` doubles, quadruples and doubles: per-animal log
    ratios (ln 2, ln 4, ln 2); ``y`` quadruples twice and quarters once: (ln 4, ln 4, −ln 4).
    ``s0`` is the median of the two sample spreads, each t is ``mean · √3 / (sd + s0)``, and
    the null is the same t over every sign flip of each animal's own ratio with the family's
    largest |t| per flip. ``x``'s own p counts the flips whose |t| reaches its own, which
    only its mirror does: 2 of 8. The verdict at every level reads the t, never the effect.
    """
    import itertools
    import statistics

    ranking = _paired_ranking(
        {
            ("A", "veh"): {"x": 0.10, "y": 0.10},
            ("A", "fent"): {"x": 0.20, "y": 0.40},
            ("B", "veh"): {"x": 0.10, "y": 0.10},
            ("B", "fent"): {"x": 0.40, "y": 0.40},
            ("C", "veh"): {"x": 0.10, "y": 0.40},
            ("C", "fent"): {"x": 0.20, "y": 0.10},
        },
        levels=["veh", "fent"],
        statistic=c.STATISTIC_T,
    )
    assert ranking.statistic == c.STATISTIC_T and ranking.exact and ranking.draws == 8
    per_unit = {
        "x": (math.log(2), math.log(4), math.log(2)),
        "y": (math.log(4), math.log(4), -math.log(4)),
    }
    s0 = statistics.median(statistics.stdev(v) for v in per_unit.values())
    assert ranking.s0 == pytest.approx(s0)
    x, y = ranking.contrasts
    assert x.measure_key[1] == "x" and y.measure_key[1] == "y"
    for item in (x, y):
        values = per_unit[item.measure_key[1]]
        assert item.effect == pytest.approx(statistics.fmean(values))
        assert item.statistic == pytest.approx(_reference_t(values, s0))
        assert item.score == pytest.approx(abs(item.statistic))
    flips = list(itertools.product((1, -1), repeat=3))
    null = {
        k: [abs(_reference_t([s * d for s, d in zip(signs, v)], s0)) for signs in flips]
        for k, v in per_unit.items()
    }
    family = sorted(max(null["x"][i], null["y"][i]) for i in range(8))
    assert sorted(ranking.floor_distribution) == pytest.approx(family)
    assert ranking.floor == pytest.approx(family[round(0.95 * 7)])
    assert x.own_p == pytest.approx(2 / 8)
    assert y.own_p == pytest.approx(
        sum(t >= y.score * (1 - 1e-9) for t in null["y"]) / 8
    )
    for alpha in (*c.STRICTNESS_LEVELS, 0.6, 0.99):
        for item in (x, y):
            assert ranking.clears_at(item, alpha) == (
                not c._reaches(ranking.floor_at(alpha), item.score)
            ), (item.measure_key, alpha)


def test_the_t_ranks_agreement_over_size_where_the_effect_ranks_size():
    """The case the statistic exists for (§24 *The statistic*). On the effect ``w`` ranks
    first: the sparse, disagreeing measure owns the order and the floor. On the moderated t
    ``z`` ranks first with a smaller family-wise p, its effect unchanged beside it."""
    shares = _agreement_cohort()
    by_effect = _paired_ranking(
        shares, levels=["veh", "fent"], statistic=c.STATISTIC_EFFECT
    )
    by_t = _paired_ranking(shares, levels=["veh", "fent"], statistic=c.STATISTIC_T)
    assert [i.measure_key[1] for i in by_effect.contrasts] == ["w", "z"]
    assert [i.measure_key[1] for i in by_t.contrasts] == ["z", "w"]
    z_effect = next(i for i in by_effect.contrasts if i.measure_key[1] == "z")
    z_t, w_t = by_t.contrasts
    assert z_t.statistic > w_t.statistic > 0
    assert z_t.effect == pytest.approx(z_effect.effect), "the effect is what it was"
    assert z_t.family_p < z_effect.family_p
    assert by_effect.s0 is None and by_t.s0 > 0
    assert all(i.statistic is None for i in by_effect.contrasts)
    assert by_effect.statistic == c.STATISTIC_EFFECT


def test_a_profile_pair_is_the_one_its_units_agree_on_under_the_t():
    """Three sessions. fent → nalox is the biggest mean change but the animals disagree on
    it (two fall sharply, one rises); fent → saline is smaller and all three agree. On the
    effect the measure's pair is fent → nalox; on the t it is fent → saline, the pair chosen by
    the same statistic the floor is taken over."""
    shares = {
        ("A", "saline"): {"x": 0.10},
        ("A", "fent"): {"x": 0.19},
        ("A", "nalox"): {"x": 0.02},
        ("B", "saline"): {"x": 0.10},
        ("B", "fent"): {"x": 0.21},
        ("B", "nalox"): {"x": 0.025},
        ("C", "saline"): {"x": 0.10},
        ("C", "fent"): {"x": 0.20},
        ("C", "nalox"): {"x": 0.60},
    }
    (by_effect,) = _profile_ranking(shares, statistic=c.STATISTIC_EFFECT).contrasts
    (by_t,) = _profile_ranking(shares, statistic=c.STATISTIC_T).contrasts
    assert by_effect.pair == ("fent", "nalox")
    assert by_t.pair == ("fent", "saline")
    assert by_t.effect == pytest.approx(
        (math.log(0.10 / 0.19) + math.log(0.10 / 0.21) + math.log(0.10 / 0.20)) / 3
    )
    assert by_t.statistic < 0 and by_t.consistency == 3


def test_a_family_whose_units_all_agree_exactly_has_no_t_and_says_why():
    """Every animal doubles exactly: no measure has a spread, so there is no ``s0`` and no
    t; the ranking says so instead of dividing by nothing, and the effect is still there.
    """
    ranking = _paired_ranking(
        {
            ("A", "veh"): {"x": 0.10},
            ("A", "fent"): {"x": 0.20},
            ("B", "veh"): {"x": 0.10},
            ("B", "fent"): {"x": 0.20},
            ("C", "veh"): {"x": 0.30},
            ("C", "fent"): {"x": 0.60},
        },
        levels=["veh", "fent"],
        statistic=c.STATISTIC_T,
    )
    assert ranking.floor is None and ranking.s0 is None
    assert "no spread to divide by" in ranking.no_floor_reason
    (item,) = ranking.contrasts
    assert item.statistic is None and item.family_p is None
    assert item.effect == pytest.approx(math.log(2))


def test_units_that_agree_have_exactly_no_spread_whatever_the_effect_is():
    """**The class:** a quantity that is zero by arithmetic, computed as the difference of two
    equal floating-point sums and then asked whether it is positive. ``Σx² − n·mean²`` for
    units with one effect is a few ulps of ``Σx²`` either side of zero, so about one family in
    three with no disagreement at all was given a spread near 1e-8, an ``s0`` from it, a floor,
    and moderated t's in the hundreds of millions. It hid because the test above uses effects
    (``log 2`` three times) whose residue happens to be exactly zero. Swept here over the
    effects and cohort sizes a screen has; ``_SPREAD_RTOL`` is the rule."""
    rng = np.random.default_rng(0)
    residue = 0
    for n in range(2, 13):
        for value in rng.uniform(0.01, 4.0, 400):
            x = np.full(n, value)
            ss, mean = np.array([np.nansum(x**2)]), np.array([np.nansum(x) / n])
            residue += int((ss - n * mean**2)[0] > 0)
            spread = c._spread(ss, mean, np.array([float(n)]))
            assert spread[0] == 0.0, (n, value)
            assert c._s0(spread) is None
    assert residue > 400, "the sweep must contain the cases that left a residue"
    # A real spread, however small beside the effects, is still one.
    x = np.array([2.0, 2.0, 2.0, 2.00001])
    real = c._spread(np.array([np.sum(x**2)]), np.array([x.mean()]), np.array([4.0]))
    assert real[0] == pytest.approx(np.std(x, ddof=1), rel=1e-3) and real[0] > 0


def test_a_family_that_agrees_on_an_awkward_effect_still_has_no_floor():
    """The same through the ranking: every animal at one ratio whose log leaves a residue
    (the first such in the sweep above), where the floor used to come back a number."""
    rng = np.random.default_rng(0)
    for value in rng.uniform(1.2, 9.0, 400):
        x = np.full(3, math.log(value))
        if np.nansum(x**2) - 3 * (np.nansum(x) / 3) ** 2 > 0:
            break
    else:
        raise AssertionError("no effect in the sweep leaves a residue")
    ranking = _paired_ranking(
        {
            (animal, level): {"x": 0.05 * (value if level == "fent" else 1.0)}
            for animal in "ABC"
            for level in ("veh", "fent")
        },
        levels=["veh", "fent"],
        statistic=c.STATISTIC_T,
    )
    assert ranking.s0 is None and ranking.floor is None
    assert "no spread to divide by" in ranking.no_floor_reason
    (item,) = ranking.contrasts
    assert item.statistic is None
    assert item.effect == pytest.approx(math.log(value))


def test_an_unknown_statistic_is_refused():
    with pytest.raises(c.ComparisonError, match="statistic must be one of"):
        _paired_ranking(_agreement_cohort(), levels=["veh", "fent"], statistic="z")


def test_a_record_without_a_statistic_ranks_on_the_effect_and_with_one_on_the_t():
    """A record written before the field existed carries no ``ranking_statistic`` and ranks
    on the effect, as it did when it was written (§3 compatibility); one that names the t
    ranks on it, with the same effects beside."""
    from detail.analysis import comparison_report

    meta, rows = _crossover(_baseline_spread_cohort(n=6))
    obs = c.observations(rows, meta, group_by=["session"], aggregate_to=["animal"])
    record = {
        "design": c.PAIRED,
        "design_column": "session",
        "levels": ["veh", "fent"],
        "aggregate_to": ["animal"],
        "min_units": 3,
        "presence_s": 1.0,
        "pseudo_count_s": 1.0,
        "ranking_measure": "time_fraction",
        "seed": 0,
        "draws": 100,
    }
    old = comparison_report.rank_record(obs, record)
    new = comparison_report.rank_record(
        obs, {**record, "ranking_statistic": c.STATISTIC_T}
    )
    assert old.statistic == c.STATISTIC_EFFECT and old.s0 is None
    assert new.statistic == c.STATISTIC_T and new.s0 is not None
    by_key = lambda r: {i.measure_key: i.effect for i in r.contrasts}  # noqa: E731
    assert by_key(old) == pytest.approx(by_key(new))
    assert all(i.statistic is not None for i in new.contrasts if i.effect is not None)


def test_the_tables_the_csv_and_the_sentences_carry_the_t():
    """Every surface that prints a ranking prints the t where it ranked (§24): the tables'
    ``t`` column, ``ranking.csv``'s ``statistic``, the floor sentence in t, the statistic
    sentence with its ``s0``, and the record's note; none of them on a ranking by the
    effect, or for a between design."""
    from detail.analysis import comparison_report as cr

    shares = _agreement_cohort()
    by_t = _paired_ranking(shares, levels=["veh", "fent"], statistic=c.STATISTIC_T)
    by_effect = _paired_ranking(shares, levels=["veh", "fent"])
    record = {
        "design": c.PAIRED,
        "design_column": "session",
        "aggregate_to": ["animal"],
        "ranking_statistic": c.STATISTIC_T,
    }
    headers, rows = cr.ranking_rows(by_t, record)
    assert "t" in headers
    z = by_t.contrasts[0]
    assert rows[0][headers.index("t")] == f"{z.statistic:+.1f}"
    csv = cr.ranking_csv_rows(by_t, record)
    assert set(csv[0]) == set(cr.RANKING_COLUMNS)
    assert csv[0]["statistic"] == pytest.approx(z.statistic)
    assert csv[0]["clears"] == (abs(z.statistic) > by_t.floor)
    assert "largest |moderated t|" in cr.floor_sentence(by_t)
    assert f"s0 = {by_t.s0:.3g}" in cr.statistic_sentence(by_t)
    assert cr.statistic_note(record).startswith("ranked on the moderated t")
    assert cr.statistic_note({**record, "design": c.BETWEEN}) == ""
    assert cr.statistic_note({"design": c.PAIRED}) == ""

    headers, _ = cr.ranking_rows(by_effect, {**record, "ranking_statistic": "effect"})
    assert "t" not in headers
    assert cr.ranking_csv_rows(by_effect, record)[0]["statistic"] == ""
    assert "fold change" in cr.floor_sentence(by_effect)
    assert cr.statistic_sentence(by_effect).startswith("ranked on the effect")
