"""A comparison says what one point is, and records what produced it (§24).

Three classes guarded, all of which render as a clean chart:

**A silently chosen unit of analysis.** One animal contributes three recordings and another one;
pooling and averaging-within-animal give different group means and answer different questions. The
tool must not choose, and the choice must reach every output — "one point is one animal" and "one
point is one session" are different claims about the same picture.

**A pooled label space.** When a method labels each recording independently, its cluster 3 means
nothing in common across recordings, so grouping them averages unrelated things into a number that
looks exactly like a real one.

**A ranking with no floor.** With several measures and a handful of animals, ranking by difference
always produces a top hit. The figure takes a `Ranking`, which cannot be built without its floor.
"""

import numpy as np
import pytest

from detail.analysis import measures
from detail.api.comparison import (
    Comparison,
    ComparisonRunError,
    run_comparison,
)
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

    from detail.api.comparison import STAGES, run_comparison

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_comparison)))
    emitted = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "report"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            emitted.append(node.args[0].value)

    assert emitted, "no report() calls found; the AST shape changed"
    assert list(dict.fromkeys(emitted)) == list(
        STAGES
    ), f"run_comparison emits {list(dict.fromkeys(emitted))}, STAGES declares {list(STAGES)}"


def test_the_gui_folds_the_runners_stages_rather_than_its_own():
    """The other half: one declaration, not two that happen to agree today."""
    pytest.importorskip("PyQt6")

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
