"""A measure carries a name a person can read, and an id nothing may rename.

**The class.** A registry entry whose only text is its key renders that key on every surface,
because there is nothing else to render. `MeasureSpec` had `measure_id` and `unit` and no display
name, so a comparison table read `n_instances__rearing__whole`, the launch dialog's measure
checklist listed `inter_bout_interval`, `summary.md`'s rows were stems, and every figure's y-axis
said `total_time`. A user looking at the result said the page had "too many `_`".

The newer half of this codebase already solved it — `Tool`, `Check`, `ModuleUI` and `SweepUI` each
pair an id with a `label` — so this is adopting the house pattern, not inventing one.

**The half that is easy to get wrong.** The id is not decoration: it is the `measure` column of
`stats.csv` and `measures.csv`, the `--measure` value scripts pass, the config value stored on a
comparison record, and the stem of every figure and table file (which `manifest.json` hashes by
path). Relabelling any of those would rename files a user has already cited. So the split is
enforced here in both directions: `_measure_stem` keeps producing ids, `measure_title` produces
labels, and the machine-readable surfaces are pinned to the id.
"""

import csv

import pytest

from detail.analysis import comparison_report
from detail.analysis.measures import MEASURES, MeasureError, register_measure


def test_there_are_measures_to_check():
    """Guard the guard: five assertions in this file are sweeps over `MEASURES`, and every one
    of them passes over an empty registry. An import that stopped populating it — the registry
    is filled by module import, so a moved file is enough — would turn this whole file green
    while saying nothing."""
    assert len(MEASURES) >= 11, (
        f"only {len(MEASURES)} measures registered; the sweeps below are near-vacuous. "
        f"Raise this floor when measures are added, never lower it."
    )


def test_every_measure_has_a_label_and_a_description():
    """Data-driven over the registry, so a measure added later is covered the day it exists."""
    missing = [
        measure_id
        for measure_id, spec in MEASURES.items()
        if not spec.label.strip() or not spec.description.strip()
    ]

    assert not missing, f"measures with no human text: {missing}"


def test_no_label_is_itself_an_id():
    """The point of the field is that a reader never meets an identifier. A label carrying an
    underscore is one that was pasted from the key.

    Note "Total time" for `total_time` is *correct*, not lazy — some ids are already the English
    words. What must not appear is the underscore."""
    idlike = [measure_id for measure_id, spec in MEASURES.items() if "_" in spec.label]

    assert not idlike, f"labels that are still ids: {idlike}"


def test_registering_without_a_label_is_refused():
    """Guard the guard, and the mechanism: defaulting to the id would have made every future
    measure render as one with nothing failing."""
    with pytest.raises(MeasureError, match="label is required"):

        @register_measure("nameless", rule="onset", description="x")
        def _nameless(sel, ctx):
            return 0.0

    with pytest.raises(MeasureError, match="description is required"):

        @register_measure("undescribed", rule="onset", label="x")
        def _undescribed(sel, ctx):
            return 0.0


# --------------------------------------------------------------------------- #
# the id keeps its jobs
# --------------------------------------------------------------------------- #
def test_the_stem_stays_an_id_and_the_title_becomes_english():
    key = ("n_instances", "rearing", "", "whole")

    assert comparison_report._measure_stem(key) == "n_instances__rearing__whole"
    assert comparison_report.measure_title(key) == "Bouts  ·  rearing"


def test_the_title_keeps_the_parts_that_are_already_human():
    """Only the measure id maps to a label. The behavior name is the user's own word, a
    transition's destination is another, and a window is a window."""
    title = comparison_report.measure_title(
        ("transitions", "rearing", "grooming", "abs_02")
    )

    assert title.startswith("Transitions")
    assert "rearing" in title and "grooming" in title and "abs_02" in title


def test_the_whole_window_is_not_worth_saying():
    """Every measure has a window; saying "whole" on all of them is noise."""
    assert "whole" not in comparison_report.measure_title(
        ("rate", "rearing", "", "whole")
    )


def test_an_unknown_measure_falls_back_to_its_id_rather_than_vanishing():
    """A `stats.csv` written by a future version must still render."""
    assert comparison_report.measure_title(
        ("not_a_measure", "x", "", "whole")
    ).startswith("not_a_measure")


def test_stats_csv_still_carries_ids(tmp_path):
    """The machine-readable half. A label here would break every script that filters on a
    measure name, and the round-trip `_observations` does back into `MEASURES`."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )

    with open(
        exp.comparisons[-1].dir / "stats.csv", newline="", encoding="utf-8-sig"
    ) as handle:
        measures = {row["measure"] for row in csv.DictReader(handle)}

    assert measures == {"n_instances"}


# --------------------------------------------------------------------------- #
# a measure is offered only where it can answer
# --------------------------------------------------------------------------- #
QUERY_MANIFEST = {"overlap": "allowed", "names": {"b1": ["rearing"], "b2": ["spin"]}}
CLUSTER_MANIFEST = {
    "overlap": "exclusive",
    # A clustering producer writes an empty list for a cluster nobody named.
    "names": {"0": ["rearing"], "1": [], "2": ["grooming"]},
}


def test_a_query_artifact_is_not_offered_clustering_measures():
    """The report that started this: a comparison of a *query* run carried a
    `Time in unnamed clusters` row of zeros. A query names every behavior by construction, so
    there is no unnamed time — the row could not have said anything."""
    from detail.analysis.measures import applicable_measures

    applicable = applicable_measures(QUERY_MANIFEST)

    assert "unnamed_time" not in applicable
    assert "n_instances" in applicable and "duration_mean" in applicable


def test_a_clustering_artifact_is_offered_all_of_them():
    """The negative case — the filter must not swallow measures that do apply."""
    from detail.analysis.measures import MEASURES, applicable_measures

    assert set(applicable_measures(CLUSTER_MANIFEST)) == set(MEASURES)


def test_transitions_needs_behaviors_that_cannot_overlap():
    """Already true, but silently: the run path skipped it with `continue` and every picker went
    on offering it, so a query comparison asked for `Transitions` and got nothing back.
    """
    from detail.analysis.measures import applicable_measures

    assert "transitions" not in applicable_measures(QUERY_MANIFEST)
    assert "transitions" in applicable_measures(CLUSTER_MANIFEST)


def test_a_refusal_carries_its_reason():
    """A missing measure with no reason reads as a bug. Every surface shows this string."""
    from detail.analysis.measures import MEASURES, unmet_condition

    why = unmet_condition(MEASURES["unnamed_time"], QUERY_MANIFEST)

    assert why and "named" in why
    assert unmet_condition(MEASURES["n_instances"], QUERY_MANIFEST) is None


def test_the_condition_is_the_fact_not_the_producer():
    """Keyed on the artifact's own names map, so a clustering module added later inherits the
    right answer without editing the measures registry."""
    from detail.analysis.measures import MEASURES, unmet_condition

    # No `producer` key at all, and the answer is still right.
    assert unmet_condition(MEASURES["unnamed_time"], {"names": {"0": []}}) is None
    assert (
        unmet_condition(MEASURES["unnamed_time"], {"names": {"0": ["x"]}}) is not None
    )


def test_every_declared_condition_exists():
    """Guard the guard: a typo'd condition name would raise at measure time, deep inside a run."""
    from detail.analysis.measures import CONDITIONS, MEASURES

    unknown = {
        (measure_id, condition)
        for measure_id, spec in MEASURES.items()
        for condition in spec.requires
        if condition not in CONDITIONS
    }

    assert not unknown, f"measures requiring conditions that do not exist: {unknown}"


def test_a_real_query_comparison_writes_no_unnamed_time_row(tmp_path):
    """Through the run path, not the predicate.

    The first version of this file tested `applicable_measures` directly and passed with the
    expansion point reverted — the filter existed and nothing used it. What the user reads is a
    row in `stats.csv`, so that is what this asserts.
    """
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    # `measures` unset — "all of them", which is the path that produced the useless row.
    exp.run_comparison(
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )

    with open(
        exp.comparisons[-1].dir / "stats.csv", newline="", encoding="utf-8-sig"
    ) as handle:
        written = {row["measure"] for row in csv.DictReader(handle)}

    assert "unnamed_time" not in written, (
        "a comparison of an artifact with no unnamed clusters still measured the time spent "
        f"in them; wrote {sorted(written)}"
    )
    assert "n_instances" in written, "the filter took measures that do apply"


def test_naming_an_inapplicable_measure_explicitly_still_writes_nothing(tmp_path):
    """The second filter's job. Asking for a measure this artifact cannot answer produces no
    rows rather than a column of zeros — the behaviour `pipeline/query.py` already documents
    for `transitions` ("it costs §23's transitions"), now true of every declared condition.
    """
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances", "unnamed_time"],
        }
    )

    with open(
        exp.comparisons[-1].dir / "stats.csv", newline="", encoding="utf-8-sig"
    ) as handle:
        written = {row["measure"] for row in csv.DictReader(handle)}

    assert written == {"n_instances"}
