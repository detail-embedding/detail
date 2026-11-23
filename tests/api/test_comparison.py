"""A comparison says what one point is, and records what produced it (§24).

Four classes guarded, all of which render as a clean chart:

**A silently chosen unit of analysis.** One animal contributes three recordings and another one;
pooling and averaging-within-animal give different group means and answer different questions. The
tool must not choose, and the choice must reach every output — "one point is one animal" and "one
point is one session" are different claims about the same picture.

**A pooled label space.** When a method labels each recording independently, its cluster 3 means
nothing in common across recordings, so grouping them averages unrelated things into a number that
looks exactly like a real one.

**A ranking with no floor.** With several measures and a handful of animals, ranking by difference
always produces a top hit. The figure takes a `Ranking`, which cannot be built without its floor.

**A recording with no value for a grouping column enters the comparison.** The design resolution
set a blank dose outside the trend and the ranking skipped it, but the recording's rows still
reached the observations, the group summaries, every per-measure figure and table and `stats.csv`
as a level with no name, and `n_observations` counted them: a trend over five doses drew a sixth,
unlabelled column. The spelling `nan` parsed as a dose under a bare `float()`. One predicate
(`cohort.missing`: blank, or a spelling of *not available*) is applied before anything is grouped,
the record names what was left out (`left_out`) and one sentence
(`comparison_report.left_out_sentence`) says so on every surface. A valued recording outside a
two-level contrast is not left out: it has a value, it sits outside the levels.
"""

import numpy as np
import pytest

from detail.analysis import measures
from detail.api.comparison import (
    Comparison,
    ComparisonRunError,
    run_comparison,
)
from detail.configs.comparison import ComparisonConfig
from detail.core.errors import UserInputError
from detail.configs.windows import WindowSpec
from detail.api.experiment import Experiment
from detail.storage import analysis_runs
from detail.storage import behavior_output as bo
from detail.storage import frame_view as fv
from detail.storage import metadata as md
from detail.storage import comparisons
from detail.storage import snapshots

PERIOD, N_FRAMES = 10_000_000, 100  # 100 Hz, one second


def _cohort(
    tmp_path, plan, *, label_space="pooled", run_id="an_01", label_names=("rearing",)
):
    """An experiment whose artifact holds ``{recording: (metadata, n_bouts)}``.

    ``label_names`` is what the behaviors are called. It is a parameter because a behavior name is
    a user's free text and reaches filenames, Markdown cells and LaTeX — so a fixture that only
    ever says ``rearing`` cannot exercise the characters those formats care about.
    """
    axis = (np.arange(N_FRAMES) * PERIOD).astype(np.int64)
    exp = Experiment.create("cohort", tmp_path)
    run_dir = analysis_runs.analysis_dir(exp.dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    instances, coverage = [], []
    for recording, (meta, n_bouts) in plan.items():
        exp.create_recording(recording)
        md.upsert_row(exp.dir, recording, meta)
        for i in range(n_bouts):
            for n, name in enumerate(label_names):
                start = (i * len(label_names) + n) * 3
                instances.append(
                    bo.BehaviorInstance(
                        recording,
                        0,
                        int(axis[start]),
                        int(axis[start + 1]),
                        start,
                        start + 1,
                        f"q{n:02d}",
                        name,
                    )
                )
        coverage.append(
            bo.CoverageRow(
                recording, 0, 0, N_FRAMES * PERIOD, N_FRAMES * PERIOD, N_FRAMES
            )
        )
    bo.write_behavior_output(
        run_dir,
        instances=instances,
        coverage=coverage,
        manifest={
            "producer": {"module_id": "test", "run_id": run_id},
            "origin": "predicted",
            "label_space": label_space,
            "vocabulary": list(label_names),
            "names": {f"q{n:02d}": [name] for n, name in enumerate(label_names)},
            "overlap": "exclusive",
            "behavior_types": {name: "state" for name in label_names},
        },
    )
    for recording in plan:
        fv.write_frame_view(
            run_dir,
            recording_id=recording,
            subject=0,
            t_ns=axis,
            observed=np.ones(N_FRAMES, bool),
            instances=[i for i in instances if i.recording_id == recording],
        )
    return exp


# One animal with three recordings, one with one — the shape a balanced fixture hides.
UNBALANCED = {
    "r1": ({"animal": "A", "drug": "veh"}, 2),
    "r2": ({"animal": "A", "drug": "veh"}, 4),
    "r3": ({"animal": "A", "drug": "veh"}, 6),
    "r4": ({"animal": "B", "drug": "veh"}, 20),
}


def _stats(comparison, measure="n_instances"):
    import csv

    with open(comparison.dir / "stats.csv", newline="", encoding="utf-8-sig") as handle:
        return [r for r in csv.DictReader(handle) if r["measure"] == measure]


def test_the_aggregation_path_changes_the_answer(tmp_path):
    """The §24 headline guard, through the config surface this time."""
    exp = _cohort(tmp_path, UNBALANCED)
    base = {"source_run_id": "an_01", "group_by": ["drug"], "measures": ["n_instances"]}

    pooled = run_comparison(exp, {**base, "aggregate_to": []})
    by_animal = run_comparison(exp, {**base, "aggregate_to": ["animal"]})

    pooled_values = sorted(float(r["value"]) for r in _stats(pooled))
    animal_values = sorted(float(r["value"]) for r in _stats(by_animal))
    assert pooled_values == [2.0, 4.0, 6.0, 20.0]  # one point per recording
    assert animal_values == [4.0, 20.0]  # A's three sessions average to 4
    assert sum(pooled_values) / len(pooled_values) != sum(animal_values) / len(
        animal_values
    )


def test_the_path_is_recorded_on_every_row_and_in_the_summary(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": ["animal"],
            "measures": ["n_instances"],
        },
    )
    assert all(r["aggregate_to"] == "animal" for r in _stats(comp))
    comp.export(formats=("png",))
    assert "one observation is**: animal" in (comp.dir / "summary.md").read_text(
        encoding="utf-8-sig"
    )


def test_an_aggregated_observation_records_how_many_rows_it_averaged(tmp_path):
    """Whether an animal-level point rests on three sessions or one."""
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": ["animal"],
            "measures": ["n_instances"],
        },
    )
    assert sorted(int(r["n_underlying"]) for r in _stats(comp)) == [1, 3]


def test_a_per_recording_label_space_is_not_pooled(tmp_path):
    """Cluster 3 here is unrelated to cluster 3 there, so a group mean over both is nonsense that
    renders as a tidy chart (§19, §22)."""
    exp = _cohort(tmp_path, UNBALANCED, label_space="per_recording")
    with pytest.raises(ComparisonRunError, match="per-recording label space"):
        run_comparison(
            exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
        )


def test_an_artifact_that_does_not_exist_is_named(tmp_path):
    exp = Experiment.create("empty", tmp_path)
    with pytest.raises(ComparisonRunError, match="no behavior output"):
        run_comparison(
            exp, {"source_run_id": "an_99", "group_by": ["drug"], "aggregate_to": []}
        )


def test_a_filter_matching_nothing_says_what_is_available(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    with pytest.raises(ComparisonRunError, match="r1"):
        run_comparison(
            exp,
            {
                "source_run_id": "an_01",
                "group_by": ["drug"],
                "aggregate_to": [],
                "recordings": ["nope"],
            },
        )


def test_the_export_tree_is_written_and_manifested(tmp_path):
    exp = _cohort(
        tmp_path,
        {
            "v1": ({"drug": "veh"}, 2),
            "v2": ({"drug": "veh"}, 3),
            "f1": ({"drug": "fent"}, 12),
            "f2": ({"drug": "fent"}, 14),
        },
    )
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances", "rate"],
            "draws": 50,
        },
    )
    manifest = comp.export(formats=("png",))
    assert (comp.dir / "summary.md").is_file()
    assert (comp.dir / "manifest.json").is_file()
    assert sorted(p.name for p in (comp.dir / "tables").glob("*.csv"))
    assert (comp.dir / "figures" / "difference_ranking.png").is_file()
    assert manifest["comparison"] == comp.id
    assert manifest["aggregate_to"] == []
    assert len(manifest["artifacts"]) >= 5


def test_re_export_is_byte_identical(tmp_path):
    """§18.5 L1: stored outputs re-render deterministically, which is what lets a figure be
    regenerated for a paper without re-deriving the numbers."""
    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
            "draws": 20,
        },
    )
    first = comp.export(tmp_path / "one", formats=("png",))
    second = comp.export(tmp_path / "two", formats=("png",))
    assert first["artifacts"] == second["artifacts"]


def test_the_export_tree_is_self_contained(tmp_path):
    """Exported elsewhere, the observations go with it — otherwise it is a set of figures whose
    source numbers live in another folder."""
    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    out = tmp_path / "elsewhere"
    comp.export(out, formats=("png",))
    assert (out / "stats.csv").is_file()


def test_a_fresh_comparison_has_no_drift(tmp_path):
    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    assert comp.drift() == []


def test_progress_reports_each_stage(tmp_path):
    """A silent long op is indistinguishable from a hang, and measuring reads a frame view per
    unit (CLAUDE.md)."""
    exp = _cohort(tmp_path, UNBALANCED)
    seen = []
    run_comparison(
        exp,
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []},
        progress=lambda stage, fraction, detail: seen.append(stage),
    )
    assert {"loading", "measuring", "grouping", "writing"} <= set(seen)


def test_the_measuring_stage_moves_within_the_pass(tmp_path):
    """The stage a caller actually waits on has to advance *inside* itself.

    Reporting only `measuring 0.0` and `measuring 1.0` passes a test that checks the stage was
    named, while the bar sits at 0% for a cohort's worth of frame-view reads — the "frozen at
    starting" shape CLAUDE.md names. `UNBALANCED` has four recordings, so a per-unit callback
    has to produce fractions strictly between the ends.
    """
    exp = _cohort(tmp_path, UNBALANCED)
    seen = []
    run_comparison(
        exp,
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []},
        progress=lambda stage, fraction, detail: seen.append((stage, fraction)),
    )

    measuring = [f for stage, f in seen if stage == "measuring"]
    assert measuring, "the measuring stage never reported"
    assert [f for f in measuring if 0.0 < f < 1.0], (
        f"measuring jumped straight from its start to its end, so the bar never moved "
        f"through the pass: {measuring}"
    )


def test_a_stopped_comparison_writes_nothing(tmp_path):
    """It used to write one, and that was the bug the name already described.

    Stopping is polled per unit, so a cancelled pass has measured *some prefix* of them — and
    the record built from it carried `status: complete` over a fraction of the cohort. Nothing
    downstream could tell it apart from a comparison that covered everything: the file is
    well-formed, the groups are real, the means are real means of the recordings that got
    measured. Losing the cancelled run is the cheaper of the two, since it is the one the user
    just asked to stop.
    """
    from detail.api.comparison import ComparisonStopped

    exp = _cohort(tmp_path, UNBALANCED)
    before = list(exp.comparisons)

    # Through the method, not the module function. The GUI holds an `Experiment`, and the
    # two are different callables: this shipped with `stop_check` reachable only on the
    # function, so pressing Cancel raised TypeError while this very test passed.
    with pytest.raises(ComparisonStopped, match="nothing was written"):
        exp.run_comparison(
            {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []},
            stop_check=lambda: True,
        )

    assert list(exp.comparisons) == before


def test_a_filter_naming_a_recording_the_artifact_lacks_is_refused(tmp_path):
    """A `--recording` the behavior result does not cover was silently dropped.

    `covered` is the intersection of the filter with the artifact's coverage, and only an
    *empty* intersection raised — so a typo, or a cohort where one recording was never analyzed,
    quietly compared a smaller set. The record made it worse: `recordings` stored the **request**
    (three ids) while provenance stored what was read (two), so the file claimed a recording that
    never entered a single number. That is Tenet 9's "persist the resolved value, not the ask",
    which the neighbouring `source_snapshot` line already gets right.
    """
    exp = _cohort(tmp_path, UNBALANCED)

    with pytest.raises(ComparisonRunError, match="typo_r9"):
        exp.run_comparison(
            {
                "source_run_id": "an_01",
                "group_by": ["drug"],
                "aggregate_to": [],
                "recordings": ["r1", "r2", "typo_r9"],
            }
        )

    assert exp.comparisons == [], "a refused comparison still wrote a record"


def test_a_filter_the_artifact_fully_covers_still_runs(tmp_path):
    """The negative case: the refusal is about *unmatched* names, not about filtering at all."""
    exp = _cohort(tmp_path, UNBALANCED)
    comparison = exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "recordings": ["r1", "r2"],
        }
    )

    assert comparison.record()["recordings"] == ["r1", "r2"]


def test_comparisons_are_discovered_and_ids_increment(tmp_path):
    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    cfg = {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    first, second = run_comparison(exp, cfg), run_comparison(exp, cfg)
    assert (
        [c.id for c in exp.comparisons] == [first.id, second.id] == ["cmp_01", "cmp_02"]
    )
    assert exp.get_comparison("cmp_01").record()["id"] == "cmp_01"


# --------------------------------------------------------------------------- #
# snapshots (§21) — a comparison cites what it read, not the slot it read from
# --------------------------------------------------------------------------- #
def _recut(exp, run_id="an_01", *, n_bouts):
    """Reproject the source run's artifact the way an authored edit does.

    Stands in for reopen -> set_granularity: the run directory keeps its id and its artifact
    changes underneath anyone still pointing at it, which is the whole defect.
    """
    axis = (np.arange(N_FRAMES) * PERIOD).astype(np.int64)
    run_dir = analysis_runs.analysis_dir(exp.dir, run_id)
    artifact = bo.read_behavior_output(run_dir)
    recordings = sorted({row.recording_id for row in artifact.coverage})
    instances = [
        bo.BehaviorInstance(
            recording,
            0,
            int(axis[i * 3]),
            int(axis[i * 3 + 1]),
            i * 3,
            i * 3 + 1,
            "q01",
            "rearing",
        )
        for recording in recordings
        for i in range(n_bouts)
    ]
    bo.write_behavior_output(
        run_dir,
        instances=instances,
        coverage=list(artifact.coverage),
        manifest=artifact.manifest,
    )
    for recording in recordings:
        fv.write_frame_view(
            run_dir,
            recording_id=recording,
            subject=0,
            t_ns=axis,
            observed=np.ones(N_FRAMES, bool),
            instances=[i for i in instances if i.recording_id == recording],
        )


def test_a_comparison_records_the_snapshot_it_read(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp,
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []},
    )
    snapshot_id = comp.record()["source_snapshot"]
    assert snapshot_id, "the resolved snapshot, not the user's None (Tenet 9)"
    assert snapshot_id in snapshots.discover_snapshots(
        analysis_runs.analysis_dir(exp.dir, "an_01")
    )


def test_a_comparisons_numbers_survive_the_source_being_re_cut(tmp_path):
    """The defect this whole change exists for. An authored run is reprojected on every edit and
    `reopen()` is a supported workflow, so a comparison that read `an_01` last week and one that
    reads it today can disagree with nothing failing. Pinning first is what makes the earlier
    numbers still reachable.
    """
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp,
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []},
    )
    before = _stats(comp)
    snapshot_id = comp.record()["source_snapshot"]

    _recut(exp, n_bouts=1)  # the run moves under everyone still pointing at it

    again = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "source_snapshot": snapshot_id,
            "group_by": ["drug"],
            "aggregate_to": [],
        },
    )
    assert _stats(again) == before

    # And the working artifact really did change, or the assertion above proves nothing.
    fresh = run_comparison(
        exp,
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []},
    )
    assert _stats(fresh) != before


def test_measuring_a_snapshot_matches_measuring_the_run_it_came_from(tmp_path):
    """A snapshot copies the frame view as well as the three artifact files, because §23's
    denominator is not regenerable from `instances.csv`. This is the assertion that justifies
    copying a directory rather than three files.
    """
    exp = _cohort(tmp_path, UNBALANCED)
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    snapshot_id = snapshots.pin(run_dir, reason="export")

    spec = WindowSpec()
    from_run = measures.measure_run(
        run_dir,
        windows_for=lambda c: spec.windows_for(c.start_t_ns, c.end_t_ns),
        measures=None,
    )
    from_snap = measures.measure_run(
        snapshots.snapshot_dir(run_dir, snapshot_id),
        windows_for=lambda c: spec.windows_for(c.start_t_ns, c.end_t_ns),
        measures=None,
    )
    assert from_snap == from_run


def test_two_comparisons_with_no_edit_between_share_one_snapshot(tmp_path):
    """Minted by difference, not by action — and copying the frame view must not break that."""
    exp = _cohort(tmp_path, UNBALANCED)
    first = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    second = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["animal"], "aggregate_to": []}
    )
    assert first.record()["source_snapshot"] == second.record()["source_snapshot"]
    assert (
        len(snapshots.discover_snapshots(analysis_runs.analysis_dir(exp.dir, "an_01")))
        == 1
    )


def test_a_snapshot_that_does_not_exist_names_the_ones_that_do(tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    with pytest.raises(ComparisonRunError, match="has no snapshot"):
        run_comparison(
            exp,
            {
                "source_run_id": "an_01",
                "source_snapshot": "snap_99",
                "group_by": ["drug"],
                "aggregate_to": [],
            },
        )


def test_a_record_written_before_snapshots_existed_still_reads(tmp_path):
    """Records already on disk carry no `source_snapshot` — the ~230 comparisons in a real
    experiment among them. They must keep reading and exporting rather than being stranded by a
    field that did not exist when they were written."""
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    record = comp.record()
    del record["source_snapshot"]  # the pre-snapshot shape
    comparisons.write_comparison_yaml(comp.dir, record)

    reread = Comparison(exp, comp.id)
    assert reread.record().get("source_snapshot") is None
    assert reread.drift() == []
    reread.export(tmp_path / "out")  # the publication tree still renders


# --------------------------------------------------------------------------- #
# drift over the source artifact (§24) — superseded, not invalid
# --------------------------------------------------------------------------- #
def test_drift_names_a_source_analysis_that_was_re_cut(tmp_path):
    """The change most likely to matter was the one `drift` could not see. `capture` recorded the
    code SHA, label hashes and video fingerprints — nothing about the artifact — so somebody
    reopening the source run and re-cutting the vocabulary was invisible to the mechanism built to
    catch exactly this (§24).
    """
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    assert comp.drift() == [], "nothing has moved yet"

    _recut(exp, n_bouts=1)

    changed = comp.drift()
    assert any("an_01" in line for line in changed), changed


def test_the_drift_message_does_not_claim_the_numbers_are_wrong(tmp_path):
    """A comparison measures a *pinned* snapshot, so its stored numbers stay exactly reproducible.
    The line has to say a re-run would differ, not that the saved result is broken — a warning read
    as "your result is corrupt" invites throwing away a good one.
    """
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    snapshot_id = comp.record()["source_snapshot"]
    _recut(exp, n_bouts=1)

    line = next(l for l in comp.drift() if "an_01" in l)
    assert snapshot_id in line, "name the snapshot the numbers came from"
    assert "re-run would differ" in line

    # And the claim is true: the pinned numbers really are still reachable.
    again = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "source_snapshot": snapshot_id,
            "group_by": ["drug"],
            "aggregate_to": [],
        },
    )
    assert _stats(again) == _stats(comp)


def test_editing_something_that_does_not_change_the_artifact_does_not_drift(tmp_path):
    """The negative half, so the guard above cannot pass on any reprojection at all: rewriting the
    artifact with identical content leaves the hash alone and says nothing."""
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    # Move the artifact, then put the original bytes back. The hash is over content, so a rewrite
    # that lands where it started must read as no change at all.
    _recut(exp, n_bouts=1)
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    snap = snapshots.snapshot_dir(run_dir, comp.record()["source_snapshot"])
    for name in (bo.INSTANCES_FILENAME, bo.COVERAGE_FILENAME, bo.MANIFEST_FILENAME):
        (run_dir / name).write_bytes((snap / name).read_bytes())

    assert not [l for l in comp.drift() if "an_01" in l]


def test_a_record_with_no_source_provenance_says_nothing_about_the_artifact(tmp_path):
    """Every comparison written before this field existed. It must drift without raising and
    without warning about something it never recorded."""
    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    record = comp.record()
    record["provenance"].pop("source", None)  # the pre-PR3 shape
    comparisons.write_comparison_yaml(comp.dir, record)

    _recut(exp, n_bouts=1)
    assert not [l for l in Comparison(exp, comp.id).drift() if "an_01" in l]


def test_a_comparison_against_an_older_snapshot_says_so_immediately(tmp_path):
    """Deliberately comparing against an earlier pin is legitimate, and it drifts from the moment
    it is created. That is correct rather than noisy: those numbers genuinely are not the current
    interpretation, and saying so is the point."""
    exp = _cohort(tmp_path, UNBALANCED)
    first = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    old_snapshot = first.record()["source_snapshot"]
    _recut(exp, n_bouts=1)

    against_old = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "source_snapshot": old_snapshot,
            "group_by": ["drug"],
            "aggregate_to": [],
        },
    )
    assert any("an_01" in line for line in against_old.drift())


def test_the_declared_stages_are_the_ones_the_run_emits():
    """`STAGES` is a claim about `run_comparison`'s own `report(...)` calls, and the GUI folds a
    per-stage fraction into a whole-run bar using its order.

    Read off the source, because the failure is silent: a renamed or reordered stage leaves the
    bar moving, to the wrong place. The GUI used to keep its own copy of this tuple, so the two
    could disagree with nothing to notice.
    """
    import ast
    import inspect
    import textwrap

    from detail.api.comparison import STAGES, prepare_comparison, run_comparison

    # The run is `prepare_comparison` (loading, measuring, grouping) then the write, so the
    # stages are read off both, in call order.
    emitted = []
    for fn in (prepare_comparison, run_comparison):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        # In source order: `ast.walk` is breadth-first, and a report inside a lambda sits
        # deeper than one at the function's top level whatever line it is on.
        calls = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "report"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        emitted += [node.args[0].value for node in calls]

    assert emitted, "no report() calls found; the AST shape changed"
    assert list(dict.fromkeys(emitted)) == list(
        STAGES
    ), f"run_comparison emits {list(dict.fromkeys(emitted))}, STAGES declares {list(STAGES)}"


def test_the_gui_folds_the_runners_stages_rather_than_its_own():
    """The other half: one declaration, not two that happen to agree today."""

    from detail.api.comparison import STAGES
    from detail.gui.comparison import _stages

    assert _stages() is STAGES


# --------------------------------------------------------------------------- #
# the figures' theme — light for print unless asked, and the choice is written down
# --------------------------------------------------------------------------- #
def _corner(png):
    from tests.png_pixel import top_left_rgb

    return top_left_rgb(png)


def test_an_export_is_light_unless_asked_and_dark_when_asked(tmp_path):
    """Read off the written PNG, not off the argument: the property is the file a reader gets."""
    from detail.core.chart_theme import DARK, LIGHT, rgb

    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
            "draws": 20,
        },
    )
    light = comp.export(tmp_path / "light", formats=("png",))
    dark = comp.export(tmp_path / "dark", formats=("png",), theme="dark")
    assert light["figure_theme"] == "light" and dark["figure_theme"] == "dark"
    name = next(n for n in light["artifacts"] if n.endswith(".png"))
    assert _corner(tmp_path / "light" / name) == rgb(LIGHT.surface)
    assert _corner(tmp_path / "dark" / name) == rgb(DARK.surface)


def test_an_unknown_theme_is_refused_before_anything_is_written(tmp_path):
    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
            "draws": 20,
        },
    )
    with pytest.raises(ValueError, match="chart theme must be one of"):
        comp.export(tmp_path / "out", theme="sepia")
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- #
# the design is resolved from the column (§24 *Designs*), and the node source
# --------------------------------------------------------------------------- #

#: Four animals, each measured on vehicle and on the drug — the crossover shape.
CROSSOVER = {
    "r1": ({"animal": "A", "drug": "veh"}, 2),
    "r2": ({"animal": "A", "drug": "fent"}, 6),
    "r3": ({"animal": "B", "drug": "veh"}, 3),
    "r4": ({"animal": "B", "drug": "fent"}, 9),
    "r5": ({"animal": "C", "drug": "veh"}, 4),
    "r6": ({"animal": "C", "drug": "fent"}, 10),
    "r7": ({"animal": "D", "drug": "veh"}, 2),
    "r8": ({"animal": "D", "drug": "fent"}, 7),
}

#: The same animals with a third session, so the column has to be told which two to contrast.
THREE_SESSIONS = {
    **CROSSOVER,
    "r9": ({"animal": "A", "drug": "nalox"}, 1),
    "r10": ({"animal": "B", "drug": "nalox"}, 2),
    "r11": ({"animal": "C", "drug": "nalox"}, 1),
    "r12": ({"animal": "D", "drug": "nalox"}, 3),
}

#: Three animals over four doses — a numeric column that varies within each animal.
DOSES = {
    f"r{a}{d}": ({"animal": f"A{a}", "dose": str(d)}, 12 - d // 10 - a)
    for a in range(3)
    for d in (0, 25, 50, 75)
}

#: One animal, two sessions: a paired design whose null has nothing to flip (#39). The
#: animal has a sex, so the launch dialog can tell it from a track slot and infer the unit
#: (`cohort.infer_subject_column`: a one-value column earns its place by determining another).
ONE_ANIMAL = {
    "r1": ({"animal": "A", "sex": "F", "drug": "veh"}, 2),
    "r2": ({"animal": "A", "sex": "F", "drug": "fent"}, 4),
}

#: Two animals, one per group: a between design relabelling cannot move.
ONE_PER_GROUP = {
    "r1": ({"animal": "A", "drug": "veh"}, 2),
    "r2": ({"animal": "B", "drug": "fent"}, 4),
}

_PAIRED = {
    "source_run_id": "an_01",
    "group_by": ["drug"],
    "aggregate_to": ["animal"],
    "measures": ["n_instances", "total_time"],
}


def test_a_column_that_varies_within_animals_resolves_to_a_paired_design(tmp_path):
    """Nothing in the config says `paired`: the column varies within the unit, so it is."""
    exp = _cohort(tmp_path, CROSSOVER)
    record = run_comparison(exp, _PAIRED).record()
    assert record["design"] == "paired"
    assert record["design_column"] == "drug"
    assert record["levels"] == ["fent", "veh"]
    assert record["n_units"] == 4 and record["min_units"] == 2
    assert "paired within animal" in record["design_reason"]
    assert record["source"] == "labels" and record["ranking_measure"] is None
    assert record["suppress_nesting"] is True and record["min_units_fraction"] == 0.5


def test_the_same_column_without_a_unit_is_a_between_design(tmp_path):
    exp = _cohort(tmp_path, CROSSOVER)
    record = run_comparison(exp, {**_PAIRED, "aggregate_to": []}).record()
    assert record["design"] == "between"
    # The contrast as run, resolved: the two groups, read as the second against the first.
    assert record["levels"] == ["fent", "veh"]
    assert "each recording is one observation" in record["design_reason"]


def test_a_live_screen_and_the_saved_comparison_rank_identically(tmp_path):
    """§22: the tab computes through §24's own code path, so the screen and the saved result
    cannot disagree — one `rank_record`, called on the same observations."""
    from detail.analysis import comparison_report
    from detail.api.comparison import screen_comparison

    exp = _cohort(tmp_path, CROSSOVER)
    before = len(exp.comparisons)
    screen = screen_comparison(exp, _PAIRED)
    assert len(exp.comparisons) == before, "a screen writes nothing"
    assert screen.ranking.design == "paired" and screen.record["design"] == "paired"

    comp = run_comparison(exp, _PAIRED)
    saved = comparison_report.rank_record(
        comparison_report._observations(comparison_report._read_stats(comp.dir)),
        comp.record(),
        comparison_report.read_hierarchy(comp.dir),
    )
    assert saved.floor == screen.ranking.floor
    assert saved.exact and saved.draws == 16  # 2^4 sign flips, enumerated
    assert [i.effect for i in saved.contrasts] == [
        i.effect for i in screen.ranking.contrasts
    ]
    assert [i.measure_key for i in saved.contrasts] == [
        i.measure_key for i in screen.ranking.contrasts
    ]


def test_three_sessions_run_as_a_profile_and_two_levels_contrast_only_those(tmp_path):
    """Three sessions within each animal are a profile over all three (§24 *Designs*): the
    record says so, the ranking carries each item's pair, the screen label names the column,
    and naming two levels still contrasts just those."""
    import csv

    from detail.api.comparison import screen_label

    exp = _cohort(tmp_path, THREE_SESSIONS)
    comp = run_comparison(exp, _PAIRED)
    record = comp.record()
    assert record["design"] == "profile"
    assert record["levels"] is None
    assert record["groups"] == ["fent", "nalox", "veh"]
    assert "a profile" in record["design_reason"]
    ranking = comp.ranking()
    assert ranking.design == "profile" and ranking.profile
    assert ranking.level_values == ("fent", "nalox", "veh")
    assert all(i.pair is not None for i in ranking.contrasts if i.effect is not None)
    assert screen_label(record) == "drug · profile"
    # Each row's effect is between its own pair: ranking.csv names it (§24 export tree).
    comp.export(formats=("png",))
    with open(comp.dir / "ranking.csv", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pairs = {
        (row["measure"], row["label"]): (row["pair_a"], row["pair_b"]) for row in rows
    }
    for item in ranking.contrasts:
        if item.pair is not None:
            assert pairs[item.measure_key[:2]] == item.pair
    record = run_comparison(exp, {**_PAIRED, "levels": ["veh", "nalox"]}).record()
    assert record["design"] == "paired"
    assert record["levels"] == ["veh", "nalox"]
    assert record["groups"] == [
        "nalox",
        "veh",
    ], "the third session is outside the contrast"
    assert "nalox against veh" in record["design_reason"]


def test_a_numeric_column_resolves_to_a_trend_and_exports_its_ranking(tmp_path):
    exp = _cohort(tmp_path, DOSES)
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "group_by": ["dose"],
            "aggregate_to": ["animal"],
            "measures": ["n_instances", "total_time"],
        },
    )
    record = comp.record()
    assert record["design"] == "trend" and record["levels"] is None
    assert "trend over 0, 25, 75" in record["design_reason"].replace("50, ", "")
    manifest = comp.export(formats=("png",))
    assert "ranking.csv" in manifest["artifacts"]
    summary = (comp.dir / "summary.md").read_text(encoding="utf-8-sig")
    assert "design**: trend" in summary
    assert "mean ρ" in summary


def test_the_node_source_measures_a_clustering_runs_nodes(tmp_path, monkeypatch):
    """Every candidate node becomes a label-like row, the hierarchy travels as `nodes.csv`, the
    ranking is on share of time and the export's `ranking.csv` carries every node."""
    import csv

    import detail.api.comparison as api
    from detail.analysis import node_measures
    from tests.analysis.test_node_measures import synthetic_view

    exp = _cohort(tmp_path, CROSSOVER)
    view = synthetic_view(recording_ids=tuple(CROSSOVER))
    monkeypatch.setattr(api, "_node_view", lambda experiment, run_id: view)
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "source": "nodes",
            "group_by": ["drug"],
            "aggregate_to": ["animal"],
            # The synthetic view's clusters hold 3 to 27 s of member time.
            "candidate_floor_s": 1.0,
        },
    )
    record = comp.record()
    assert record["source"] == "nodes"
    assert (
        record["source_snapshot"] is None
    ), "nothing is pinned: the cache is the source"
    assert record["candidate_floor_s"] == 1.0
    assert ComparisonConfig.model_fields["candidate_floor_s"].default == 30.0
    assert record["ranking_measure"] == "time_fraction"
    assert record["design"] == "paired"
    assert record["provenance"]["source"]["content_hash"] == api.node_source_hash(
        view.built_from
    )
    assert (comp.dir / api.NODES_FILENAME).is_file()
    with open(comp.dir / "stats.csv", newline="", encoding="utf-8-sig") as handle:
        stats = list(csv.DictReader(handle))
    assert {row["label"][:5] for row in stats} == {"node:"}
    assert {row["measure"] for row in stats} == set(node_measures.MEASURES)
    assert all(row["denominator_ns"] for row in stats)

    manifest = comp.export(formats=("png",))
    assert "ranking.csv" in manifest["artifacts"]
    with open(comp.dir / "ranking.csv", newline="", encoding="utf-8-sig") as handle:
        ranked = list(csv.DictReader(handle))
    assert {row["label"] for row in ranked} == {
        node_measures.node_label(n) for n in node_measures.candidate_nodes(view, 1.0)
    }
    assert {row["measure"] for row in ranked} == {"time_fraction"}
    assert all(row["suggestion"] for row in ranked)
    # The floor read as a number, each node's own p and the BH q over them (§24): what a
    # reader thresholds at a strictness of their own; blank only where the effect is.
    for row in ranked:
        if row["effect"] == "":
            assert row["family_p"] == row["own_p"] == row["q"] == ""
            continue
        family_p, own_p, q = (float(row[k]) for k in ("family_p", "own_p", "q"))
        assert 0.0 <= family_p <= 1.0 and 0.0 < own_p <= 1.0 and own_p <= q <= 1.0
        assert (row["clears"] == "True") == (family_p <= 0.05)
    assert comp.drift() == []


def test_the_node_source_refuses_windows(tmp_path, monkeypatch):
    import detail.api.comparison as api
    from tests.analysis.test_node_measures import synthetic_view

    exp = _cohort(tmp_path, CROSSOVER)
    monkeypatch.setattr(
        api, "_node_view", lambda e, r: synthetic_view(recording_ids=tuple(CROSSOVER))
    )
    with pytest.raises(ComparisonRunError, match="whole recording"):
        run_comparison(
            exp,
            {
                "source_run_id": "an_01",
                "source": "nodes",
                "group_by": ["drug"],
                "aggregate_to": ["animal"],
                "windows": {"kind": "normalized", "parts": 2},
            },
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"source_snapshot": "snap_01"}, "current node cache"),
        ({"measures": ["rate"]}, "always measures"),
        ({"measures": ["rate", "time_fraction", "total_time"]}, "always measures"),
    ],
)
def test_the_node_source_refuses_what_it_would_ignore(
    tmp_path, monkeypatch, override, match
):
    """A snapshot pins the labelled artifact, which the node source does not read; a screen
    without share of time ranks nothing, and without total time cannot tell presence. Each
    was silently ignored or silently broke the screen."""
    import detail.api.comparison as api
    from tests.analysis.test_node_measures import synthetic_view

    exp = _cohort(tmp_path, CROSSOVER)
    monkeypatch.setattr(
        api, "_node_view", lambda e, r: synthetic_view(recording_ids=tuple(CROSSOVER))
    )
    config = {
        "source_run_id": "an_01",
        "source": "nodes",
        "group_by": ["drug"],
        "aggregate_to": ["animal"],
        **override,
    }
    with pytest.raises(ComparisonRunError, match=match):
        run_comparison(exp, config)
    assert exp.comparisons == [], "refused before anything was written"


def test_the_node_source_takes_its_own_four_measures_named(tmp_path, monkeypatch):
    """A record seeds its measures back; the four the node source always takes are no
    subset."""
    from detail.analysis import node_measures

    _exp, _view, comp = _node_screen(
        tmp_path, monkeypatch, measures=list(reversed(node_measures.MEASURES))
    )
    assert comp.ranking() is not None


def test_the_node_source_refuses_a_run_that_has_no_nodes(tmp_path):
    exp = _cohort(tmp_path, CROSSOVER)
    with pytest.raises(ComparisonRunError, match="not a clustering run"):
        run_comparison(
            exp,
            {
                "source_run_id": "an_01",
                "source": "nodes",
                "group_by": ["drug"],
                "aggregate_to": ["animal"],
            },
        )


def test_a_stale_node_cache_is_refused_naming_what_changed(tmp_path, monkeypatch):
    import detail.api.comparison as api
    from tests.analysis.test_node_measures import synthetic_view

    exp = _cohort(tmp_path, CROSSOVER)
    view = synthetic_view(recording_ids=tuple(CROSSOVER))
    view.stale = "the fit file is 'new.npz' now but the clips were built from 'fit.npz'"

    class _Run:
        def has_node_cache(self):
            return True

        def node_view(self):
            return view

    monkeypatch.setattr(exp, "authored_run", lambda run_id: _Run())
    with pytest.raises(ComparisonRunError, match="stale.*new.npz"):
        run_comparison(
            exp,
            {
                "source_run_id": "an_01",
                "source": "nodes",
                "group_by": ["drug"],
                "aggregate_to": ["animal"],
            },
        )


def test_a_record_from_before_designs_still_ranks_as_it_did(tmp_path):
    """A comparison saved without `design` is a between design with no pseudo-count, so its
    numbers do not move under the new reader."""
    from detail.analysis import comparison_report

    exp = _cohort(tmp_path, UNBALANCED)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["animal"], "aggregate_to": []}
    )
    obs = comparison_report._observations(comparison_report._read_stats(comp.dir))
    record = comp.record()
    old = {
        k: v
        for k, v in record.items()
        if k not in ("design", "levels", "source", "pseudo_count_s", "min_units")
    }
    new = comparison_report.rank_record(obs, record)
    legacy = comparison_report.rank_record(obs, old)
    assert legacy.floor == new.floor
    assert [i.magnitude for i in legacy.contrasts] == [
        i.magnitude for i in new.contrasts
    ]


# --------------------------------------------------------------------------- #
# a saved screen read back: its ranking, its kept clusters, the run's list (§22)
# --------------------------------------------------------------------------- #


def _node_screen(tmp_path, monkeypatch, **overrides):
    import detail.api.comparison as api
    from tests.analysis.test_node_measures import synthetic_view

    exp = _cohort(tmp_path, CROSSOVER)
    view = synthetic_view(recording_ids=tuple(CROSSOVER))
    monkeypatch.setattr(api, "_node_view", lambda experiment, run_id: view)
    comp = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "source": "nodes",
            "group_by": ["drug"],
            "aggregate_to": ["animal"],
            "candidate_floor_s": 1.0,
            **overrides,
        },
    )
    return exp, view, comp


def test_a_saved_screen_ranks_and_names_its_kept_clusters(tmp_path, monkeypatch):
    from detail.analysis import node_measures
    from detail.api.comparison import screen_comparison

    exp, view, comp = _node_screen(tmp_path, monkeypatch)
    ranking = comp.ranking()
    assert ranking is not None and ranking.design == "paired"
    live = screen_comparison(exp, {**comp.record(), "candidate_floor_s": 1.0}).ranking
    assert [i.measure_key for i in ranking.contrasts] == [
        i.measure_key for i in live.contrasts
    ]
    kept = comp.kept_nodes()
    assert kept == [
        node_measures.node_of(item.measure_key[1]) for item in ranking.kept_items()
    ]
    assert set(kept) <= set(int(n) for n in node_measures.candidate_nodes(view, 1.0))
    # What the dock turns a screen into: the leaves under what it kept.
    leaves = view.leaves_under_nodes(kept)
    assert leaves <= frozenset(range(view.tree.n_leaves))


def test_the_runs_screens_are_listed_newest_first_and_labels_comparisons_are_not(
    tmp_path, monkeypatch
):
    from detail.api.comparison import node_screens, screen_label

    exp, _view, first = _node_screen(tmp_path, monkeypatch)
    run_comparison(exp, _PAIRED)  # a labels comparison over the same run
    second = run_comparison(
        exp,
        {
            "source_run_id": "an_01",
            "source": "nodes",
            "group_by": ["drug"],
            "aggregate_to": [],
            "candidate_floor_s": 1.0,
        },
    )
    listed = node_screens(exp, "an_01")
    assert [c.id for c in listed] == [second.id, first.id]
    assert node_screens(exp, "an_02") == []
    assert screen_label(first.record()) == "veh vs fent · paired"
    assert screen_label(second.record()) == "veh vs fent · between"
    assert screen_label({"design": "trend", "design_column": "dose"}) == "dose · trend"


def _all_stats(comparison):
    import csv

    with open(comparison.dir / "stats.csv", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_a_recording_with_no_value_in_the_column_is_left_out_and_named(tmp_path):
    """A blank dose is not a dose of zero and not a group of its own. The design already set
    such a recording outside the trend; its rows still reached the observations, the groups,
    the figures and `stats.csv` as a level with no name. It is left out before anything is
    grouped, and the record says which (section 24)."""
    plan = {
        **DOSES,
        "r0x": ({"animal": "A0", "dose": ""}, 3),
        "r1x": ({"animal": "A1", "dose": "NA"}, 4),
    }
    comp = run_comparison(
        _cohort(tmp_path, plan),
        {
            "source_run_id": "an_01",
            "group_by": ["dose"],
            "aggregate_to": ["animal"],
            "measures": ["n_instances", "total_time"],
        },
    )
    record = comp.record()
    assert record["design"] == "trend"
    assert record["left_out"] == ["r0x", "r1x"]
    assert record["groups"] == ["0", "25", "50", "75"]
    rows = _all_stats(comp)
    assert {r["group"] for r in rows} == {"0", "25", "50", "75"}
    assert record["n_observations"] == len(rows)
    comp.export(formats=("png",))
    summary = (comp.dir / "summary.md").read_text(encoding="utf-8-sig")
    assert "left out" in summary
    assert "r0x" in summary and "r1x" in summary


def test_a_valued_recording_outside_the_contrast_is_not_left_out(tmp_path):
    """The third session of a paired contrast has a value; it sits outside the two levels, not
    outside the data, and the record does not list it."""
    exp = _cohort(tmp_path, THREE_SESSIONS)
    record = run_comparison(exp, {**_PAIRED, "levels": ["veh", "nalox"]}).record()
    assert record["left_out"] == []


def test_the_record_resolves_the_statistic_it_ranked_on(tmp_path):
    """Tenet 9: the config's choice is written and, beside it, what actually ranked. A paired
    design ranks on the moderated t by default and on the effect when asked; a between design
    has no per-unit effects and ranks on the effect whatever was asked, and the record says
    so rather than carrying a choice that did not apply. The ranking read back agrees.
    """
    exp = _cohort(tmp_path, CROSSOVER)
    paired = run_comparison(exp, _PAIRED)
    record = paired.record()
    assert record["statistic"] == "moderated_t"
    assert record["ranking_statistic"] == "moderated_t"
    assert paired.ranking().statistic == "moderated_t"
    plain = run_comparison(exp, {**_PAIRED, "statistic": "effect"})
    assert plain.record()["ranking_statistic"] == "effect"
    assert plain.ranking().statistic == "effect"
    between = run_comparison(exp, {**_PAIRED, "aggregate_to": []})
    assert between.record()["statistic"] == "moderated_t"
    assert between.record()["ranking_statistic"] == "effect"
    assert between.ranking().statistic == "effect"


# --------------------------------------------------------------------------- #
# a screen that could have no floor is refused before measuring (#39)
# --------------------------------------------------------------------------- #
def _node_view_over(monkeypatch, plan):
    import detail.api.comparison as api
    from tests.analysis.test_node_measures import synthetic_view

    view = synthetic_view(recording_ids=tuple(plan))
    monkeypatch.setattr(api, "_node_view", lambda experiment, run_id: view)


@pytest.mark.parametrize(
    ("plan", "aggregate_to", "reason", "remedy"),
    [
        (
            ONE_ANIMAL,
            ["animal"],
            "only one animal (A) has both fent and veh",
            "Add a second animal that has both fent and veh.",
        ),
        (
            ONE_PER_GROUP,
            [],
            "every group holds at most one recording",
            "Add a second recording to a group.",
        ),
    ],
)
def test_a_screen_that_could_have_no_floor_is_refused_before_measuring(
    tmp_path, monkeypatch, plan, aggregate_to, reason, remedy
):
    """The design and its unit count were resolved after the measuring pass, so a paired
    screen over two recordings from one animal measured for minutes and saved a screen that
    kept nothing and could not be ticked (#39). The shapes a count of recordings decides are
    refused first, with the reason and what would run; nothing is measured or written.
    """
    from detail.api.comparison import screen_comparison

    exp = _cohort(tmp_path, plan)
    _node_view_over(monkeypatch, plan)
    config = {
        "source_run_id": "an_01",
        "source": "nodes",
        "group_by": ["drug"],
        "aggregate_to": aggregate_to,
        "candidate_floor_s": 1.0,
    }
    stages = []
    with pytest.raises(ComparisonRunError) as excinfo:
        run_comparison(
            exp, config, progress=lambda stage, fraction, detail: stages.append(stage)
        )
    message = str(excinfo.value)
    # The cohort is the user's choice: one `Error:` line from the CLI root (#52).
    assert isinstance(excinfo.value, UserInputError)
    assert message.startswith(reason), message
    assert "this screen is refused before measuring" in message
    assert remedy in message and "labelled behaviors instead" in message
    assert "measuring" not in stages, "refused before the measuring pass"
    assert exp.comparisons == [], "nothing is written"
    # The live screen is the same computation, so it is refused the same way (§22).
    with pytest.raises(ComparisonRunError, match="refused before measuring"):
        screen_comparison(exp, config)


def test_a_labelled_comparison_over_one_animal_runs_and_says_it_has_no_floor(tmp_path):
    """A comparison of named behaviors with no floor is still a description, reported as
    measured (§24): it is not refused, and the ranking carries the same sentence the launch
    would have warned with."""
    exp = _cohort(tmp_path, ONE_ANIMAL)
    comp = run_comparison(exp, _PAIRED)
    ranking = comp.ranking()
    assert ranking.floor is None
    assert ranking.no_floor_reason.startswith(
        "only one animal (A) has both fent and veh"
    )
    comp.export(formats=("png",))
    summary = (comp.dir / "summary.md").read_text(encoding="utf-8-sig")
    assert "No floor computed" in summary
