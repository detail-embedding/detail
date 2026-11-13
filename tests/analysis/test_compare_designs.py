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
    assert "single unit" in ranking.no_floor_reason
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
