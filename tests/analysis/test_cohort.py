"""What the cohort table says about itself, before anything is measured.

These are the facts a comparison dialog offers a user at the moment of choosing, so the thing
being pinned is a *judgement*, not a return value: which column identifies a repeated animal, and
whether the unit-of-analysis question arises at all. A test that only checked "it returned a
string" would pass while the surface confidently named the drug column as the animal.

The shapes below are the ones that actually occur. `CROSSOVER` is a synthetic stand-in for a
single-animal crossover study (one animal, two treatments); `BETWEEN` is the `UNBALANCED` fixture
that `tests/api/test_comparison.py` uses; `PAIRED_UNIQUE` is the trap where nothing repeats and
the question must not be asked.
"""

from detail.analysis import cohort
from detail.analysis.cohort import (
    describe_columns,
    group_preview,
    infer_subject_column,
)

# One animal, two treatments — the animal crosses the compared column. Synthetic values.
CROSSOVER = {
    "subject_01_baseline": {
        "animal": "S1",
        "sex": "X",
        "animal_id": "1",
        "treatment": "baseline",
        "date": "2000-01-01",
    },
    "subject_01_treatment": {
        "animal": "S1",
        "sex": "X",
        "animal_id": "1",
        "treatment": "treatment_low",
        "date": "2000-01-08",
    },
}

# Animal A contributed three recordings, animal B one — nested within one treatment arm.
BETWEEN = {
    "r1": {"animal": "A", "drug": "veh"},
    "r2": {"animal": "A", "drug": "veh"},
    "r3": {"animal": "A", "drug": "veh"},
    "r4": {"animal": "B", "drug": "veh"},
}

# Every animal appears once. Nothing repeats, so there is nothing to average within.
PAIRED_UNIQUE = {
    "r1": {"animal": "A", "drug": "veh"},
    "r2": {"animal": "B", "drug": "veh"},
    "r3": {"animal": "C", "drug": "drug_a"},
    "r4": {"animal": "D", "drug": "drug_a"},
}


def _all(metadata):
    return sorted(metadata), sorted({k for row in metadata.values() for k in row})


# --------------------------------------------------------------------------- #
# which column is the animal
# --------------------------------------------------------------------------- #
def test_a_crossover_names_the_animal_not_the_treatment():
    """The sharpest signal there is: `S1` appears under both treatments, so it identifies a
    subject. A treatment column essentially never partitions its own recordings across another
    treatment's levels *and* repeats."""
    recordings, columns = _all(CROSSOVER)

    picked = infer_subject_column(
        CROSSOVER, recordings, group_by=["treatment"], columns=columns
    )

    assert picked in ("animal", "animal_id"), picked


def test_nesting_within_one_arm_still_finds_the_animal():
    recordings, columns = _all(BETWEEN)

    picked = infer_subject_column(
        BETWEEN, recordings, group_by=["drug"], columns=columns
    )

    assert picked == "animal"


def test_no_repeats_means_no_question_to_ask():
    """The case the whole feature hinges on. Four animals, four recordings, nobody twice — the
    unit-of-analysis question does not arise and a surface must not raise it."""
    recordings, columns = _all(PAIRED_UNIQUE)

    picked = infer_subject_column(
        PAIRED_UNIQUE, recordings, group_by=["drug"], columns=columns
    )

    assert picked is None


def test_the_compared_column_is_never_its_own_subject():
    recordings, columns = _all(BETWEEN)

    picked = infer_subject_column(
        BETWEEN, recordings, group_by=["animal"], columns=columns
    )

    assert picked != "animal"


def test_a_column_with_a_blank_is_refused():
    """A blank is stored as `""` and `compare._factors` turns a missing value into that same
    `""`, so a half-filled column would pool every unfilled recording into one convincing extra
    animal. Refusing outright is the only safe reading."""
    holey = {
        "r1": {"animal": "A", "drug": "veh"},
        "r2": {"animal": "A", "drug": "veh"},
        "r3": {"animal": "", "drug": "drug_a"},
        "r4": {"animal": "", "drug": "drug_a"},
    }
    recordings, columns = _all(holey)

    assert (
        infer_subject_column(holey, recordings, group_by=["drug"], columns=columns)
        is None
    )


def test_the_choice_is_stable_when_two_columns_say_the_same_thing():
    """`animal` and `animal_id` are the same fact twice — common, and it must not flip between
    runs, or the same command reports a different unit tomorrow.

    **Vary the input order, not the call count.** The first version called the same pure function
    five times with identical arguments and asserted one answer came back, which no
    implementation can fail. What actually varies between runs is the order the columns arrive
    in — `metadata.csv`'s header, a `set` iteration — so that is what this permutes, holding
    everything the answer may legitimately depend on fixed.
    """
    import itertools

    recordings, columns = _all(CROSSOVER)

    picks = {
        infer_subject_column(
            CROSSOVER, recordings, group_by=["treatment"], columns=list(order)
        )
        for order in itertools.permutations(columns)
    }

    assert len(picks) == 1, f"the inferred subject depends on column order: {picks}"
    # And it is one of the two identity columns. Without this the assertion above passes on any
    # implementation that returns a *constant* -- including `max` over a tuple with the
    # tiebreaking name removed, which then indexes a number rather than a column.
    (pick,) = picks
    assert pick in {"animal", "animal_id"}, pick


def test_a_constant_column_that_explains_nothing_is_not_a_subject():
    """`subject` is a per-recording track index, `0` in every experiment that exists — and it is
    *structurally identical* to a single-animal crossover: it repeats, it has no blanks, and it
    crosses every group. Cardinality cannot tell them apart, so the rule is that a one-value
    column earns its place only by determining another. A track slot explains nothing.
    """
    flat = {
        "r1": {"subject": "0", "drug": "veh"},
        "r2": {"subject": "0", "drug": "drug_a"},
    }
    recordings, columns = _all(flat)

    picked = infer_subject_column(flat, recordings, group_by=["drug"], columns=columns)

    assert picked is None


def test_a_one_animal_cohort_is_still_found_because_it_explains_its_covariates():
    """The other half of the same rule, and the single-animal crossover shape: `animal` is
    also a single value, but it determines `sex` and `animal_id`, which is what an identity does
    and a track slot does not."""
    recordings, columns = _all(CROSSOVER)

    picked = infer_subject_column(
        CROSSOVER, recordings, group_by=["treatment"], columns=columns
    )

    assert picked is not None


# --------------------------------------------------------------------------- #
# what a column holds
# --------------------------------------------------------------------------- #
def test_a_column_reports_its_values_and_their_sizes():
    recordings, _ = _all(BETWEEN)

    (animal,) = describe_columns(BETWEEN, recordings, ["animal"])

    assert animal.counts == {"A": 3, "B": 1}
    assert animal.describe() == "A (3), B (1)"
    assert animal.usable_for_grouping is None
    assert animal.has_repeats


def test_a_constant_column_says_why_it_cannot_group():
    """`subject` is `0` in every experiment that exists, so offering it as a comparison axis is
    offering a guaranteed one-group result."""
    flat = {"r1": {"subject": "0"}, "r2": {"subject": "0"}}

    (column,) = describe_columns(flat, sorted(flat), ["subject"])

    assert column.usable_for_grouping == "every recording says '0'"


def test_a_small_study_is_not_refused():
    """Two recordings, one each — a pilot pair. Small is a fact to report, not a reason to
    refuse; a single-animal crossover study is exactly this."""
    recordings, _ = _all(CROSSOVER)

    (treatment,) = describe_columns(CROSSOVER, recordings, ["treatment"])

    assert treatment.usable_for_grouping is None
    assert not treatment.has_repeats


def test_many_values_collapse_to_a_shape():
    wide = {f"r{i}": {"date": f"2026-07-{i:02d}"} for i in range(1, 9)}

    (date,) = describe_columns(wide, sorted(wide), ["date"])

    assert date.describe() == "8 values, 1 each"


def test_blanks_are_counted_apart_from_values():
    holey = {"r1": {"drug": "veh"}, "r2": {"drug": ""}, "r3": {"drug": "veh"}}

    (drug,) = describe_columns(holey, sorted(holey), ["drug"])

    assert drug.counts == {"veh": 2}
    assert drug.blank == 1
    assert "1 blank" in drug.describe()


# --------------------------------------------------------------------------- #
# the groups, sized before the run
# --------------------------------------------------------------------------- #
def test_groups_are_sized_in_recordings_and_animals():
    recordings, _ = _all(PAIRED_UNIQUE)

    groups = group_preview(
        PAIRED_UNIQUE,
        recordings,
        group_by=["drug"],
        subject_column="animal",
    )

    assert [(g.label, g.n_recordings, g.n_subjects) for g in groups] == [
        ("drug_a", 2, 2),
        ("veh", 2, 2),
    ]


def test_unknown_subjects_are_none_not_the_recording_count():
    """ "one animal per recording" and "we do not know" are different facts, and rendering the
    second as the first is how a surface invents a denominator."""
    recordings, _ = _all(PAIRED_UNIQUE)

    groups = group_preview(PAIRED_UNIQUE, recordings, group_by=["drug"])

    assert all(g.n_subjects is None for g in groups)


def test_the_unbalanced_shape_is_visible_before_the_run():
    """The whole reason the question exists: animal A would outvote animal B three to one, and
    the dialog can say so without measuring anything."""
    recordings, _ = _all(BETWEEN)

    (group,) = group_preview(
        BETWEEN, recordings, group_by=["drug"], subject_column="animal"
    )

    assert group.n_recordings == 4 and group.n_subjects == 2


# --------------------------------------------------------------------------- #
# a column metadata.csv does not have
# --------------------------------------------------------------------------- #
DEFERRED_COHORT = {
    "r1": {"animal": "S1", "drug": "veh"},
    "r2": {"animal": "S1", "drug": "drug_a"},
    "r3": {"animal": "S2", "drug": "veh"},
}


def test_a_column_off_the_artifact_is_named_as_deferred():
    """`subject` and `window_id` are offered as grouping columns on purpose — they come from the
    behavior result's rows and from the window spec, not from `metadata.csv`."""
    recordings = list(DEFERRED_COHORT)

    assert cohort.deferred_columns(DEFERRED_COHORT, recordings, ["subject"]) == [
        "subject"
    ]
    assert cohort.deferred_columns(DEFERRED_COHORT, recordings, ["drug"]) == []
    assert cohort.deferred_columns(
        DEFERRED_COHORT, recordings, ["drug", "window_id"]
    ) == ["window_id"]


def test_a_deferred_column_previews_nothing_rather_than_one_blank_group():
    """Reading a column no row carries gave every recording `""`, so grouping by `subject`
    previewed as a single group labelled "(blank)" holding the whole cohort — and a comparison
    that will produce one group per subject was described as producing one. An empty preview is
    the honest answer; the surface says the sizes are known once it runs."""
    recordings = list(DEFERRED_COHORT)

    assert cohort.group_preview(DEFERRED_COHORT, recordings, group_by=["subject"]) == []
    # The negative case: a column that *is* there still previews.
    assert (
        len(cohort.group_preview(DEFERRED_COHORT, recordings, group_by=["drug"])) == 2
    )


def test_a_missing_token_counts_as_blank():
    """``NA`` and ``nan`` are how a spreadsheet writes an empty cell. A column summary reads them
    as blank, never as a value, so neither becomes a level to tick."""
    meta = {
        "r1": {"dose": "10"},
        "r2": {"dose": "NA"},
        "r3": {"dose": "nan"},
        "r4": {"dose": ""},
    }
    summary = cohort.describe_columns(meta, list(meta), ["dose"])[0]
    assert summary.counts == {"10": 1}
    assert summary.blank == 3


def test_a_recording_with_no_value_is_left_out_of_the_preview_and_named():
    """A recording with no value for a grouping column is not a group of its own: the preview
    leaves it out and names it, the same way the run will (section 24)."""
    meta = {
        "r1": {"animal": "A", "dose": "0"},
        "r2": {"animal": "A", "dose": "10"},
        "r3": {"animal": "A", "dose": ""},
        "r4": {"animal": "B", "dose": "NA"},
    }
    assert cohort.left_out(meta, list(meta), ["dose"]) == ["r3", "r4"]
    groups = cohort.group_preview(meta, list(meta), group_by=["dose"])
    assert [g.group for g in groups] == [("0",), ("10",)]
    # A deferred column leaves nothing out: its values are not knowable yet.
    assert cohort.left_out(meta, list(meta), ["subject"]) == []
