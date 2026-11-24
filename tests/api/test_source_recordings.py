"""What a comparison is described over before it runs is what it compares (§24).

**A preview over recordings the run never sees.** The launch dialog described every column
over every recording in the experiment, whatever the chosen run covered. Over a location
export of three recordings out of fifty it listed every column's values and every animal,
inferred repeats from recordings the run would never read, and the command line demanded
``--aggregate-to`` on the same count. The run itself compared only what it covered, so the
form described one comparison and launched another. One rule now says which recordings a run
covers (``api.comparison.source_recordings``); the run, the dialog and the command line all
ask it.

Data-driven over the config's ``source`` choices, so a source added later fails here until
it says what it covers.
"""

import typing

import pytest

from detail.api import comparison as api
from detail.configs.comparison import ComparisonConfig
from detail.storage import analysis_runs
from detail.storage import behavior_output as bo
from detail.storage import metadata as md
from detail.storage import node_cache as node_store
from detail.storage import snapshots
from tests.api import test_comparison_locations as locations
from tests.api.test_comparison import CROSSOVER, _cohort

SOURCES = typing.get_args(ComparisonConfig.model_fields["source"].annotation)

#: Recordings in the experiment that no run below covers, with values no covered one has.
OUTSIDE = {"x1": {"animal": "Z", "drug": "other"}, "x2": {"animal": "Z", "drug": "veh"}}


def _add_outside(exp, meta=OUTSIDE) -> None:
    for recording, values in meta.items():
        exp.create_recording(recording)
        md.upsert_row(exp.dir, recording, values)


def _labels(tmp_path, monkeypatch):
    exp = _cohort(tmp_path, CROSSOVER)
    _add_outside(exp)
    return exp, {
        "source_run_id": "an_01",
        "group_by": ["drug"],
        "aggregate_to": ["animal"],
    }


def _nodes(tmp_path, monkeypatch):
    """A node cache over six of the artifact's eight recordings, read from disk by the run."""
    from tests.analysis.test_node_measures import synthetic_view

    exp = _cohort(tmp_path, CROSSOVER)
    _add_outside(exp)
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    node_store.write_node_cache(
        run_dir,
        synthetic_view(recording_ids=("r1", "r2", "r3", "r4", "r5", "r6"))._cache,
    )
    # The run's view of the cache, built over the recordings the cache on disk holds.
    monkeypatch.setattr(
        api,
        "_node_view",
        lambda experiment, run_id: synthetic_view(
            recording_ids=node_store.read_node_cache(run_dir).leaves.recording_ids
        ),
    )
    config = {"source_run_id": "an_01", "source": "nodes", "group_by": ["drug"],
              "aggregate_to": ["animal"], "candidate_floor_s": 1.0}  # fmt: skip
    return exp, config


def _locations(tmp_path, monkeypatch):
    """A location export that finished all but one of its recordings."""
    plan = locations.paired_plan()
    exp = locations.cohort(tmp_path, plan)
    _add_outside(exp, {"x1": {"animal": "Z", "session": "other"}})
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    record = analysis_runs.read_analysis_yaml(run_dir)
    record["status"] = "partial"
    record["per_recording"]["d7"] = {"status": "failed"}
    analysis_runs.write_analysis_yaml(run_dir, record)
    return exp, {**locations.PAIRED, "draws": 200}


BUILDERS = {"labels": _labels, "nodes": _nodes, "locations": _locations}


def test_every_source_says_what_it_covers():
    assert set(BUILDERS) == set(SOURCES)


@pytest.mark.parametrize("source", SOURCES)
def test_what_a_source_covers_is_what_its_run_compares(tmp_path, monkeypatch, source):
    exp, config = BUILDERS[source](tmp_path, monkeypatch)
    covered = api.source_recordings(exp, "an_01", source=source)
    prepared = api.prepare_comparison(exp, config)
    assert sorted(covered) == sorted(set(prepared.covered))
    everything = {recording.id for recording in exp.recordings}
    assert (
        set(covered) < everything
    ), "the fixture must leave recordings outside the run"


def test_a_snapshot_covers_what_it_froze_and_not_what_the_run_holds_now(tmp_path):
    exp = _cohort(tmp_path, CROSSOVER)
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    frozen = snapshots.pin(run_dir, reason="comparison", code_sha="test")
    # The run's artifact is re-cut afterwards, over two fewer recordings.
    artifact = bo.read_behavior_output(run_dir)
    keep = {"r1", "r2", "r3", "r4", "r5", "r6"}
    bo.write_behavior_output(
        run_dir,
        instances=[i for i in artifact.instances if i.recording_id in keep],
        coverage=[c for c in artifact.coverage if c.recording_id in keep],
        manifest=artifact.manifest,
    )
    assert sorted(api.source_recordings(exp, "an_01")) == sorted(keep)
    then = api.source_recordings(exp, "an_01", snapshot=frozen)
    assert sorted(then) == sorted(CROSSOVER)
    config = {"source_run_id": "an_01", "source_snapshot": frozen, "group_by": ["drug"],
              "aggregate_to": ["animal"]}  # fmt: skip
    assert sorted(set(api.prepare_comparison(exp, config).covered)) == sorted(then)


@pytest.mark.parametrize(
    ("source", "said"),
    [
        ("labels", "no behavior output"),
        ("nodes", "no node cache"),
        ("locations", "location-export run"),
    ],
)
def test_a_run_that_cannot_offer_the_source_is_refused_by_name(tmp_path, source, said):
    """A behavior run is not a location export and has no node cache here; a location export
    has no behavior artifact."""
    if source == "labels":
        exp = locations.cohort(tmp_path, locations.paired_plan())
    else:
        exp = _cohort(tmp_path, CROSSOVER)
        analysis_runs.write_analysis_yaml(
            analysis_runs.analysis_dir(exp.dir, "an_01"),
            {"id": "an_01", "module_id": "behavior_scan", "status": "complete"},
        )
    with pytest.raises(api.ComparisonRunError, match=said):
        api.source_recordings(exp, "an_01", source=source)


def test_a_source_a_comparison_does_not_have_is_refused(tmp_path):
    exp = _cohort(tmp_path, CROSSOVER)
    with pytest.raises(ValueError, match="no source 'tracks'"):
        api.source_recordings(exp, "an_01", source="tracks")
