"""A result certain to be empty from recording information alone is computed anyway.

**The class.** A difference screen exists to say which clusters clear the floor. Some shapes of
cohort cannot have a floor, and the shapes are counts of recordings and their metadata, knowable
before a single frame is measured: a within-unit design where fewer than two units hold the
levels its null flips or shuffles; a between design where every group holds at most one unit, so
relabelling changes nothing; fewer than two groups, so nothing is contrasted. The comparison
engine resolved the design *after* measuring, from the measure rows, and nothing acted on the
count, so a paired screen over two recordings from one animal measured for minutes, saved, and
was listed as a screen that could not be ticked (#39). The rule
(``compare.no_floor_before_measuring``) is asked before measuring: a screen is refused with the
reason and what would run, a comparison of labelled behaviors warns and runs.

**How it hid.** The after-measuring ranking said *no floor* with a reason, so every surface was
correct and the result looked like an answer about the data rather than a shape the launch could
have seen. The launch dialog previewed the design from the same metadata and never asked the
next question.

**The rule the guard holds.** The rule before measuring and the ranking after it must agree on
every shape the rule claims to know: a reason before measuring implies the same reason after it
(never a false refusal), and the knowable shapes are all caught (never a screen saved that the
rule could have refused). The counts before measuring are upper bounds on the ranking's (a unit
can lose a level to a blank measure, never gain one), which is what makes the first half true;
the second is checked shape by shape. The reasons only measuring can know (no measure present in
enough units, no spread) are shown to be *not* claimed.

**Shown able to fail.** Each branch of the rule removed in turn makes the agreement test fail
on that shape; the tightened unit count in ``rank_by_design`` (a unit at one level cannot be
flipped) reverted makes the two-units-one-of-them-single shape disagree.
"""

import pytest

from detail.analysis import compare as c
from tests.analysis.test_compare_designs import _crossover, _pseudo, _rows

NS = 1_000_000_000


def _between_cohort(plan: dict, *, group="drug", unit="animal"):
    """``{recording: (group value, unit value, share)}`` -> metadata, rows."""
    meta, rows = {}, []
    for recording, (g, u, share) in plan.items():
        meta[recording] = {group: g, unit: u}
        rows += _rows(recording, "x", share)
    return meta, rows


def _after_measuring(meta, rows, *, group_by, aggregate_to, levels=None, **kwargs):
    """The reason the ranking gives once measured, through the run's own steps: the design
    from the rows, the observations, the design's ranker. ``"nothing to contrast"`` where the
    record → ranking mapping returns ``None`` (fewer than two groups)."""
    design = c.resolve_design_from_rows(
        rows, meta, group_by=group_by, aggregate_to=aggregate_to, levels=levels
    )
    outside = set(c.left_out(meta, list(meta), group_by))
    wanted = set(design.levels) if design.levels and design.column else None
    obs = c.observations(
        rows,
        meta,
        group_by=group_by,
        aggregate_to=aggregate_to,
        predicate=lambda f: f["recording_id"] not in outside
        and (wanted is None or f.get(design.column, "") in wanted),
    )
    if design.within:
        ranking = c.rank_by_design(
            obs,
            design,
            min_units=kwargs.pop("min_units", 1),
            presence_ns=0,
            pseudo=_pseudo(0.0),
            ranking_measure="time_fraction",
            **kwargs,
        )
        return ranking.no_floor_reason
    family = [o for o in obs if o.measure_key[0] == "time_fraction"]
    groups = sorted({o.group for o in family})
    if len(groups) < 2:
        return "nothing to contrast"
    return c.rank_by_difference(family, tuple(groups)).no_floor_reason


#: Every shape the rule claims to know, and the two it must not claim. ``knowable`` is whether
#: the rule says *no floor* before measuring; the ranking afterwards must agree.
SHAPES = {
    "one animal with both sessions": (
        _crossover({("A", "veh"): {"x": 0.1}, ("A", "fent"): {"x": 0.2}}),
        dict(group_by=["session"], aggregate_to=["animal"]),
        True,
    ),
    "two animals with both sessions": (
        _crossover(
            {
                ("A", "veh"): {"x": 0.1},
                ("A", "fent"): {"x": 0.2},
                ("B", "veh"): {"x": 0.3},
                ("B", "fent"): {"x": 0.1},
            }
        ),
        dict(group_by=["session"], aggregate_to=["animal"]),
        False,
    ),
    "two animals, one of them at a single session": (
        _crossover(
            {
                ("A", "veh"): {"x": 0.1},
                ("A", "fent"): {"x": 0.2},
                ("B", "veh"): {"x": 0.3},
            }
        ),
        dict(group_by=["session"], aggregate_to=["animal"]),
        True,
    ),
    "a trend: two animals at three doses": (
        _crossover(
            {
                (a, d): {"x": 0.1 * (i + 1)}
                for a in ("A", "B")
                for i, d in enumerate(("0", "25", "50"))
            }
        ),
        dict(group_by=["session"], aggregate_to=["animal"]),
        False,
    ),
    "a trend: one animal at three doses, another at two": (
        _crossover(
            {
                ("A", "0"): {"x": 0.1},
                ("A", "25"): {"x": 0.2},
                ("A", "50"): {"x": 0.3},
                ("B", "0"): {"x": 0.1},
                ("B", "25"): {"x": 0.2},
            }
        ),
        dict(group_by=["session"], aggregate_to=["animal"]),
        True,
    ),
    "a profile: two animals over three named sessions": (
        _crossover(
            {
                (a, s): {"x": 0.1 * (i + 1)}
                for a in ("A", "B")
                for i, s in enumerate(("base", "drug", "wash"))
            }
        ),
        dict(group_by=["session"], aggregate_to=["animal"]),
        False,
    ),
    "a profile: one animal over three sessions, another at one": (
        _crossover(
            {
                ("A", "base"): {"x": 0.1},
                ("A", "drug"): {"x": 0.2},
                ("A", "wash"): {"x": 0.3},
                ("B", "base"): {"x": 0.1},
            }
        ),
        dict(group_by=["session"], aggregate_to=["animal"]),
        True,
    ),
    "between, one recording per group": (
        _between_cohort({"r1": ("veh", "A", 0.1), "r2": ("fent", "B", 0.3)}),
        dict(group_by=["drug"], aggregate_to=[], one_row_per_recording=True),
        True,
    ),
    "between, one animal per group over two recordings each": (
        _between_cohort(
            {
                "r1": ("veh", "A", 0.1),
                "r2": ("veh", "A", 0.2),
                "r3": ("fent", "B", 0.3),
                "r4": ("fent", "B", 0.4),
            }
        ),
        dict(group_by=["drug"], aggregate_to=["animal"]),
        True,
    ),
    "between, a group of two and a group of one": (
        _between_cohort(
            {"r1": ("veh", "A", 0.1), "r2": ("veh", "B", 0.2), "r3": ("fent", "C", 0.3)}
        ),
        dict(group_by=["drug"], aggregate_to=[], one_row_per_recording=True),
        False,
    ),
    "between over three groups, two animals in one of them": (
        _between_cohort(
            {
                "r1": ("veh", "A", 0.1),
                "r2": ("veh", "B", 0.2),
                "r3": ("low", "C", 0.3),
                "r4": ("high", "D", 0.4),
            }
        ),
        dict(group_by=["drug"], aggregate_to=["animal"]),
        False,
    ),
    "one group": (
        _between_cohort({"r1": ("veh", "A", 0.1), "r2": ("veh", "B", 0.2)}),
        dict(group_by=["drug"], aggregate_to=["animal"]),
        True,
    ),
    "one group once the blank recording is left out": (
        _between_cohort({"r1": ("veh", "A", 0.1), "r2": ("", "B", 0.2)}),
        dict(group_by=["drug"], aggregate_to=["animal"]),
        True,
    ),
    "two levels named out of three sessions, held by one animal": (
        _crossover(
            {
                ("A", "veh"): {"x": 0.1},
                ("A", "fent"): {"x": 0.2},
                ("A", "wash"): {"x": 0.3},
                ("B", "veh"): {"x": 0.3},
                ("B", "wash"): {"x": 0.1},
            }
        ),
        dict(group_by=["session"], aggregate_to=["animal"], levels=["veh", "fent"]),
        True,
    ),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_rule_before_measuring_and_the_ranking_after_it_agree(name):
    (meta, rows), spec, knowable = SHAPES[name]
    before = c.no_floor_before_measuring(meta, list(meta), **spec)
    after = _after_measuring(
        meta, rows, **{k: v for k, v in spec.items() if k != "one_row_per_recording"}
    )
    assert (before is not None) == knowable, (name, before)
    assert (after is not None) == knowable, (name, after)
    if (
        knowable
        and after != "nothing to contrast"
        and "every group" not in before.reason
    ):
        # The within-unit shapes carry the one sentence into the ranking; the between
        # shape's after-measuring sentence counts observations, which it alone can see.
        assert before.reason == after, name


def test_the_reason_names_the_one_unit_that_could_move():
    (meta, _), spec, _ = SHAPES["two animals, one of them at a single session"]
    found = c.no_floor_before_measuring(meta, list(meta), **spec)
    # No levels named, so the column's two values are read in sort order.
    assert found.reason == (
        "only one animal (A) has both fent and veh, so a within-animal null has nothing "
        "to move and no floor can be computed"
    )
    assert found.remedy == "Add a second animal that has both fent and veh."


def test_the_between_and_one_group_remedies_say_what_to_add():
    (meta, _), spec, _ = SHAPES[
        "between, one animal per group over two recordings each"
    ]
    found = c.no_floor_before_measuring(meta, list(meta), **spec)
    assert found.reason.startswith("every group holds at most one animal")
    assert found.remedy == "Add a second animal to a group."

    (meta, _), spec, _ = SHAPES["one group"]
    found = c.no_floor_before_measuring(meta, list(meta), **spec)
    assert found.reason == (
        "these recordings hold one value of 'drug' (veh), so there is nothing to contrast"
    )
    assert found.remedy == "Choose recordings with two values of 'drug'."


def test_the_reasons_only_measuring_can_know_are_not_claimed():
    """Four animals with both sessions can have a floor; a `min_units` above four cannot be
    seen from the metadata, so the rule says nothing and the ranking says why."""
    shares = {}
    for i in range(4):
        shares[(f"A{i}", "veh")] = {"x": 0.1}
        shares[(f"A{i}", "fent")] = {"x": 0.2}
    meta, rows = _crossover(shares)
    spec = dict(group_by=["session"], aggregate_to=["animal"])
    assert c.no_floor_before_measuring(meta, list(meta), **spec) is None
    assert "undefined" in _after_measuring(meta, rows, min_units=5, **spec)


def test_a_deferred_column_is_decided_when_it_runs():
    (meta, _), _, _ = SHAPES["one animal with both sessions"]
    assert (
        c.no_floor_before_measuring(
            meta, list(meta), group_by=["window_id"], aggregate_to=["animal"]
        )
        is None
    )
    assert (
        c.no_floor_before_measuring(
            meta, list(meta), group_by=["session"], aggregate_to=["subject"]
        )
        is None
    )


def test_one_recording_per_group_is_many_observations_under_windows():
    """With an empty aggregation path the observation is the row, and under windows one
    recording is many rows: the between shape is then not knowable, the one-group shape is.
    """
    (meta, _), _, _ = SHAPES["between, one recording per group"]
    spec = dict(group_by=["drug"], aggregate_to=[])
    assert c.no_floor_before_measuring(meta, list(meta), **spec) is None
    assert c.no_floor_before_measuring(
        meta, list(meta), one_row_per_recording=True, **spec
    )
    (meta, _), _, _ = SHAPES["one group"]
    assert c.no_floor_before_measuring(meta, list(meta), **spec).reason.endswith(
        "nothing to contrast"
    )
