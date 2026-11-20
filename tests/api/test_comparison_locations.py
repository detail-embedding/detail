"""A comparison over a location-export run: distance, speed and body shape by group (§24).

**A finding that could not be found (detail#70).** A comparison ranks its family against one
floor, the largest difference chance alone produces anywhere in it. That is right for hundreds
of clusters in one unit. Over a distance, a share of time and a body length it belongs to the
noisiest of them: on the experiment this was written for, every rat's body was shorter under
naloxone (10 of 10, by 6%), and no floor shared with a distance that halved could be cleared by
6% of anything. So a locations comparison reads each measure by its own test, and the test
below ranks the same observations both ways to show the difference.

**A source read as another kind.** A location-export run has no behavior artifact, no
snapshot and no node cache; asking a comparison for any of those of it is refused by name,
and so is asking for locations of a run that is not a location export.

**A comparison that does not say what it read.** Whether the body is the whole mask or has its
tail removed changes every body measure, so the record carries how the run was exported.
"""

import csv

import numpy as np
import pytest

from detail.analysis import compare
from detail.analysis import comparison_report
from detail.analysis import location_measures as lm
from detail.api.comparison import ComparisonRunError, run_comparison
from detail.api.experiment import Experiment
from detail.storage import analysis_runs
from detail.storage import metadata as md
from tests.location_fixtures import track, write_run

EXPORTED = {"sample_fps": 5.0, "remove_thin_parts": True, "thin_part_scale": 0.12}


def cohort(
    tmp_path, plan, *, run_id="an_01", module_id="location_export", status="complete"
):
    """An experiment with a finished location-export run over ``{recording: (metadata, track)}``."""
    exp = Experiment.create("cohort", tmp_path)
    run_dir = analysis_runs.analysis_dir(exp.dir, run_id)
    for recording, (meta, _track) in plan.items():
        exp.create_recording(recording)
        md.upsert_row(exp.dir, recording, meta)
    write_run(run_dir, {rec: {0: tr} for rec, (_meta, tr) in plan.items()})
    analysis_runs.write_analysis_yaml(
        run_dir,
        {
            "id": run_id,
            "module_id": module_id,
            "status": status,
            "recording_ids": list(plan),
            "per_recording": {rec: {"status": "complete"} for rec in plan},
            "params": dict(EXPORTED),
        },
    )
    return exp


def walk(speed_mm_s: float, *, length=180.0, width=75.0, seconds=60) -> lm.Track:
    """A straight walk at ``speed_mm_s``, five rows a second."""
    return track(np.arange(seconds * 5) * speed_mm_s / 5.0, length=length, width=width)


def paired_plan() -> dict:
    """Eight animals, each under saline and drug. Distance falls a lot and unevenly (one
    animal's rises); the body shortens by 5% in every animal; its width does not change.
    """
    change = [0.45, 0.5, 0.9, 0.4, 0.6, 1.3, 0.45, 0.7]  # the slowest still moves
    plan = {}
    for a, factor in enumerate(change):
        speed, length = 50.0 + 10.0 * a, 180.0 + 5.0 * a
        width = 75.0 + (0.2 if a % 2 else -0.2)
        plan[f"s{a}"] = (
            {"animal": f"A{a}", "session": "saline"},
            walk(speed, length=length),
        )
        plan[f"d{a}"] = (
            {"animal": f"A{a}", "session": "drug"},
            walk(speed * factor, length=length * 0.95, width=width),
        )
    return plan


PAIRED = {"source_run_id": "an_01", "source": "locations", "group_by": ["session"],
          "aggregate_to": ["animal"], "levels": ["saline", "drug"]}  # fmt: skip


def _items(comparison) -> dict:
    ranking = comparison.ranking()
    return {item.measure_key[0]: item for item in ranking.contrasts}


def test_the_record_says_what_was_read_and_how_it_is_judged(tmp_path):
    comp = run_comparison(cohort(tmp_path, paired_plan()), {**PAIRED, "draws": 200})
    record = comp.record()
    assert record["source"] == "locations" and record["design"] == compare.PAIRED
    assert record["verdict"] == compare.VERDICT_OWN
    # the moderated t borrows spread across the family, which means nothing across units
    assert record["statistic"] == "moderated_t"  # as asked
    assert record["ranking_statistic"] == compare.STATISTIC_EFFECT  # as ranked
    assert (
        record["draws"] == 10_000
    )  # never fewer: at 200 a trend's smallest p is 0.005
    assert record["location_export"] == EXPORTED
    assert record["measures"] == list(lm.MEASURES)
    assert record["source_snapshot"] is None
    for name in ("location_sample_fps", "location_smooth_s", "location_moving_threshold",
                 "location_jump_speed"):  # fmt: skip
        assert name in record, name
    assert "screened_labels" not in record and "remainder_labels" not in record


def test_a_small_change_every_animal_shares_is_a_finding_by_its_own_test(tmp_path):
    comp = run_comparison(cohort(tmp_path, paired_plan()), PAIRED)
    ranking = comp.ranking()
    items = _items(comp)
    length, width, distance = (
        items["body_length"],
        items["body_width"],
        items["distance"],
    )
    assert length.effect == pytest.approx(np.log(0.95), abs=1e-6)
    assert abs(distance.effect) > 8 * abs(length.effect)  # the far larger change
    # eight animals, every sign flip enumerated: all eight shortening is 2 of 256
    assert length.own_p == pytest.approx(2 / 256)
    assert ranking.verdict == compare.VERDICT_OWN
    assert ranking.clears_floor(length) and not ranking.clears_floor(width)
    kept = [i.measure_key[0] for i in ranking.kept_items()]
    assert "body_length" in kept and "body_width" not in kept
    # the class: against one floor shared with the distance, the same change cannot clear
    by_floor = comparison_report.rank_record(
        comparison_report._observations(comparison_report._read_stats(comp.dir)),
        {**comp.record(), "verdict": compare.VERDICT_FAMILY},
    )
    shared = {item.measure_key[0]: item for item in by_floor.contrasts}
    assert not by_floor.clears_floor(shared["body_length"])
    assert (
        shared["body_length"].own_p == length.own_p
    )  # the same test, read differently


def test_measures_are_ordered_by_evidence_not_by_size(tmp_path):
    ranking = run_comparison(cohort(tmp_path, paired_plan()), PAIRED).ranking()
    own = [item.own_p for item in ranking.contrasts if item.own_p is not None]
    assert own == sorted(own)
    sizes = [item.magnitude for item in ranking.contrasts if item.own_p is not None]
    assert sizes != sorted(sizes, reverse=True)  # by size, the distance would lead


def test_stats_csv_holds_one_value_per_animal_and_session_in_the_measures_unit(
    tmp_path,
):
    comp = run_comparison(cohort(tmp_path, paired_plan()), PAIRED)
    with open(comp.dir / "stats.csv", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8 * 2 * len(lm.MEASURES)
    saline_a0 = {
        r["measure"]: float(r["value"])
        for r in rows
        if r["observation"] == "A0" and r["group"] == "saline"
    }
    assert saline_a0["distance"] == pytest.approx(
        0.05 * 59.0, abs=0.06
    )  # 5 cm/s, metres
    assert saline_a0["speed_moving"] == pytest.approx(5.0, abs=0.01)  # cm/s
    assert saline_a0["body_length"] == 180.0 and saline_a0["time_moving"] == 1.0


def test_a_dose_series_is_a_trend_and_its_p_is_not_capped_at_the_default_draws(
    tmp_path,
):
    plan = {}
    for a in range(6):
        for dose in (0, 10, 20, 30):
            plan[f"a{a}_{dose}"] = (
                {"animal": f"A{a}", "dose": str(dose)},
                walk((60.0 + 5 * a) * (1.0 - dose / 40.0)),
            )
    comp = run_comparison(
        cohort(tmp_path, plan),
        {"source_run_id": "an_01", "source": "locations", "group_by": ["dose"],
         "aggregate_to": ["animal"], "measures": ["distance", "body_length"]},
    )  # fmt: skip
    record, items = comp.record(), _items(comp)
    assert record["design"] == compare.TREND and record["measures"] == [
        "distance",
        "body_length",
    ]
    assert items["distance"].effect == pytest.approx(
        -1.0
    )  # every animal, every step down
    assert items["distance"].own_p < 0.001  # 200 draws could not say less than 0.005
    assert comp.ranking().clears_floor(items["distance"])
    assert not comp.ranking().clears_floor(items["body_length"])


def test_two_groups_of_different_animals_are_compared_between(tmp_path):
    plan = {
        f"{strain}{a}": (
            {"animal": f"{strain}{a}", "strain": strain},
            walk(50.0 + a, length=length + 0.3 * a),
        )
        for strain, length in (("wt", 180.0), ("ko", 200.0))
        for a in range(4)
    }
    comp = run_comparison(
        cohort(tmp_path, plan),
        {"source_run_id": "an_01", "source": "locations", "group_by": ["strain"],
         "aggregate_to": ["animal"], "measures": ["body_length", "body_width"]},
    )  # fmt: skip
    ranking, items = comp.ranking(), _items(comp)
    assert comp.record()["design"] == compare.BETWEEN
    # four against four: 70 ways to split them, and this is the most extreme two
    assert items["body_length"].own_p == pytest.approx(2 / 70, abs=0.01)
    assert ranking.clears_floor(items["body_length"])
    assert not ranking.clears_floor(items["body_width"])


def test_windows_cut_each_animals_session_and_can_be_compared(tmp_path):
    comp = run_comparison(
        cohort(tmp_path, paired_plan()),
        {
            **PAIRED,
            "measures": ["distance"],
            "windows": {"kind": "normalized", "parts": 2},
        },
    )
    with open(comp.dir / "stats.csv", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["window_id"] for r in rows} == {"part_00", "part_01"}
    halves = [
        float(r["value"])
        for r in rows
        if r["observation"] == "A0" and r["group"] == "saline"
    ]
    assert sum(halves) == pytest.approx(0.05 * 59.0, abs=0.06)


def test_the_export_carries_the_verdicts_and_draws_no_chart_of_sizes(tmp_path):
    comp = run_comparison(cohort(tmp_path, paired_plan()), PAIRED)
    comp.export(formats=("png",))
    summary = (comp.dir / "summary.md").read_text(encoding="utf-8-sig")
    assert "Each measure is tested on its own" in summary
    assert "Floor (95th percentile" not in summary
    with open(comp.dir / "ranking.csv", newline="", encoding="utf-8-sig") as handle:
        rows = {r["measure"]: r for r in csv.DictReader(handle)}
    assert (
        rows["body_length"]["clears"] == "True"
        and rows["body_width"]["clears"] == "False"
    )
    assert float(rows["body_length"]["own_p"]) == pytest.approx(2 / 256)
    assert rows["body_length"]["q"] != ""
    assert {r["floor"] for r in rows.values()} == {""}  # no floor was the judge
    figures = sorted(p.name for p in (comp.dir / "figures").glob("*.png"))
    assert "difference_ranking.png" not in figures
    assert any(name.startswith("body_length") for name in figures)
    headers, cells = comparison_report.ranking_rows(comp.ranking(), comp.record())
    assert "p ≤ 0.05" in headers and comparison_report.OWN_P_HEADER in headers
    assert "clears floor" not in headers
    # a name a person reads, never the raw id
    names = {row[0] for row in cells}
    assert "Body length" in names and "Distance travelled" in names
    assert not any("_" in name for name in names)


def test_a_fresh_comparison_has_no_drift_and_a_replaced_run_has(tmp_path):
    plan = paired_plan()
    exp = cohort(tmp_path, plan)
    comp = run_comparison(exp, PAIRED)
    assert comp.drift() == []
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    write_run(run_dir, {"s0": {0: walk(90.0, seconds=30)}})  # the run's table, replaced
    assert comp.drift() != []


@pytest.mark.parametrize(
    "change, message",
    [
        ({"source_run_id": "an_99"}, "no analysis run 'an_99'"),
        ({"source_snapshot": "snap_01"}, "have no snapshots"),
        ({"measures": ["rate"]}, "has no measure rate; it has distance"),
        ({"recordings": ["s0", "nope"]}, "is not covered by 'an_01'"),
    ],
)
def test_what_a_location_run_cannot_answer_is_refused_by_name(
    tmp_path, change, message
):
    exp = cohort(tmp_path, paired_plan())
    with pytest.raises(ComparisonRunError, match=message):
        run_comparison(exp, {**PAIRED, **change})


def test_a_run_of_another_module_or_an_unfinished_one_is_refused(tmp_path):
    other = cohort(tmp_path / "a", paired_plan(), module_id="temporal_clustering")
    with pytest.raises(ComparisonRunError, match="an_01 is a temporal_clustering run"):
        run_comparison(other, PAIRED)
    running = cohort(tmp_path / "b", paired_plan(), status="running")
    with pytest.raises(ComparisonRunError, match="an_01 is running"):
        run_comparison(running, PAIRED)


def test_a_recording_the_run_did_not_finish_is_not_compared(tmp_path):
    exp = cohort(tmp_path, paired_plan(), status="partial")
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    record = analysis_runs.read_analysis_yaml(run_dir)
    record["per_recording"]["d7"] = {"status": "failed", "error": "ValueError: gone"}
    analysis_runs.write_analysis_yaml(run_dir, record)
    comp = run_comparison(exp, PAIRED)
    assert "d7" not in comp.record()["provenance"]["recordings"]
    # seven animals have both sessions; the eighth has only its saline one
    assert _items(comp)["body_length"].n_units == 7


def test_a_labels_comparison_is_still_judged_by_the_familys_floor(tmp_path):
    """The verdict is the source's, not a new default: every other comparison ranks as it did."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    assert comp.record()["verdict"] == compare.VERDICT_FAMILY
    ranking = comp.ranking()
    assert ranking is None or ranking.verdict == compare.VERDICT_FAMILY
