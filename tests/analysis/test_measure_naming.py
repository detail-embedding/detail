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
    assert comparison_report.measure_title(key, ()) == "Bouts  ·  rearing"


def test_the_title_keeps_the_parts_that_are_already_human():
    """Only the measure id maps to a label. The behavior name is the user's own word, a
    transition's destination is another, and a window is a window."""
    title = comparison_report.measure_title(
        ("transitions", "rearing", "grooming", "abs_02"), ()
    )

    assert title.startswith("Transitions")
    assert "rearing" in title and "grooming" in title and "abs_02" in title


def test_the_whole_window_is_not_worth_saying():
    """Every measure has a window; saying "whole" on all of them is noise."""
    assert "whole" not in comparison_report.measure_title(
        ("rate", "rearing", "", "whole"), ()
    )


def test_an_unknown_measure_falls_back_to_its_id_rather_than_vanishing():
    """A `stats.csv` written by a future version must still render."""
    assert comparison_report.measure_title(
        ("not_a_measure", "x", "", "whole"), ()
    ).startswith("not_a_measure")


def test_a_remainder_label_reads_as_the_parents_other():
    """A parent label's row counts only what its children did not take (#24): its title says
    so, and a label that is not a parent keeps its plain name."""
    remainders = {"moving"}

    assert comparison_report.measure_title(
        ("total_time", "moving", "", "whole"), remainders
    ) == ("Total time  ·  moving (other)")
    assert comparison_report.measure_title(
        ("total_time", "fast", "", "whole"), remainders
    ) == ("Total time  ·  fast")


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
    # A clustering producer writes an empty list for its `unlabelled` source label. The real
    # producer's manifest is checked below (`test_a_real_clustering_run_*`): this hand-written
    # one was once true of no run on disk.
    "names": {"L1": ["rearing"], "L2": ["grooming"], "unlabelled": []},
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


# --------------------------------------------------------------------------- #
# a real clustering run: unnamed time is measured, a parent's row says what it counts
# --------------------------------------------------------------------------- #
def _measured(experiment, run_id) -> list[dict]:
    return experiment.measures(run_id).table()


def test_a_real_clustering_run_with_unnamed_clusters_measures_unnamed_time(
    fitted_codebook,
):
    """**The class: a consumer predicate written against an older shape of the producer's
    output.** ``unnamed_labels`` read "some ``names`` entry is an empty list", the fixture above
    (``CLUSTER_MANIFEST``) wrote one, and the real producer never did: its ``names`` is keyed by
    label id and holds named labels only, with unlabelled members under the ``unlabelled``
    source label and no entry at all. So no clustering run ever measured ``unnamed_time``, while
    ``docs/cli.md`` and ``docs/gui.md`` promised it. This runs the real producer instead.
    """
    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    left, _right = view.children(view.root)
    run.name_nodes({left: "rear"})

    rows = _measured(experiment, run_id)

    unnamed = [row for row in rows if row["measure"] == "unnamed_time"]
    assert (
        unnamed
    ), f"no unnamed_time row; measured {sorted({r['measure'] for r in rows})}"
    assert all(row["value"] > 0 for row in unnamed)


def test_a_clustering_run_with_nothing_labelled_still_measures_unnamed_time(
    fitted_codebook,
):
    """The edge: an empty vocabulary made the table header-only, so the one number that says
    "all of this run is still unnamed" was missing exactly when it is the whole answer.
    """
    experiment, run_id = fitted_codebook

    rows = _measured(experiment, run_id)

    assert {row["measure"] for row in rows} == {"unnamed_time"}
    assert all(row["value"] > 0 for row in rows)


def test_a_parent_labels_row_is_named_as_its_remainder(fitted_codebook, tmp_path):
    """**Owner decision (#24): a parent label keeps counting only its leftover, and the row says
    so.** Label the root "moving" and a child "fast": every frame is moving, but the row under
    "moving" counts only what "fast" did not take. The ``label`` column keeps the plain name,
    because it is the join key for a comparison (``compare.MEASURE_KEYS``) and for a user joining
    to ``instances.csv``; ``label_display`` carries "moving (other)" and ``label_is_remainder``
    marks the row, so a CSV reader can tell. No number changes."""
    from detail.analysis import measures as measures_lib

    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    child, _other = view.children(view.root)
    run.name_nodes({view.root: "moving", child: "fast"})

    rows = _measured(experiment, run_id)
    by_label = {}
    for row in rows:
        if row["measure"] == "total_time":
            by_label[row["label"]] = row

    assert set(by_label) == {"moving", "fast"}
    assert by_label["moving"]["label_display"] == "moving (other)"
    assert by_label["moving"]["label_is_remainder"] is True
    assert by_label["fast"]["label_display"] == "fast"
    assert by_label["fast"]["label_is_remainder"] is False

    out = experiment.measures(run_id).export(tmp_path / "m")
    with open(out / "measures.csv", newline="", encoding="utf-8-sig") as handle:
        written = list(csv.DictReader(handle))
    assert {"label_display", "label_is_remainder"} <= set(written[0])
    moving = [r for r in written if r["label"] == "moving"]
    assert moving and {r["label_display"] for r in moving} == {"moving (other)"}
    assert set(measures_lib.COLUMNS) >= {"label", "label_display", "label_is_remainder"}


def test_a_comparison_names_a_parent_labels_rows_as_its_remainder(fitted_codebook):
    """Through the run path: the record says which labels are remainders, ``stats.csv`` carries
    the display name and the flag beside the plain ``label`` it groups on, and the per-measure
    titles a person reads say "(other)". No number changes, so ``label`` is still the key.
    One recording, so the contrast is early against late (``window_id``)."""
    from detail.analysis import comparison_report

    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    child, _other = view.children(view.root)
    run.name_nodes({view.root: "moving", child: "fast"})
    # The fit fixture writes no video; a comparison's provenance fingerprints it.
    video = experiment.dir / "recordings/recording_001/video/part_001.mp4"
    video.write_bytes(b"not a real video")

    experiment.run_comparison(
        {
            "source_run_id": run_id,
            "group_by": ["window_id"],
            "aggregate_to": [],
            "windows": {"kind": "normalized", "parts": 2},
            "measures": ["total_time"],
        }
    )
    comparison = experiment.comparisons[-1]
    record = comparison.record()
    with open(comparison.dir / "stats.csv", newline="", encoding="utf-8-sig") as f:
        stats = list(csv.DictReader(f))

    assert record["remainder_labels"] == ["moving"]
    moving = [r for r in stats if r["label"] == "moving"]
    fast = [r for r in stats if r["label"] == "fast"]
    assert moving and fast
    assert {(r["label_display"], r["label_is_remainder"]) for r in moving} == {
        ("moving (other)", "True")
    }
    assert {(r["label_display"], r["label_is_remainder"]) for r in fast} == {
        ("fast", "False")
    }
    assert comparison_report.measure_title(
        ("total_time", "moving", "", "whole"), comparison_report.remainders_of(record)
    ).endswith("moving (other)")
    ranked = comparison_report.ranking_csv_rows(comparison.ranking(), record)
    assert {
        (r["label_display"], r["label_is_remainder"])
        for r in ranked
        if r["label"] == "moving"
    } == {("moving (other)", True)}, "ranking.csv names the remainder row too"
    assert set(ranked[0]) == set(comparison_report.RANKING_COLUMNS)
