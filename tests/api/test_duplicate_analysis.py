"""Duplicate an analysis run (``api/duplicate_analysis.py``): an exact copy under a new id.

The classes guarded here:

- **A copy that is not the run.** Every file the run holds is in the copy, byte for byte,
  except the records that name their own run — and those name the copy. A record left naming
  the original (``producer.run_id``, a profile's ``run_id``, ``job.json``'s ``job_id``) makes
  the copy's artifact and profiles claim to be another run's.
- **A copy that harms the original.** Nothing of the source changes, not one byte.
- **A copy of a run mid-write.** A running run is refused.
- **Process state carried over.** A cancel request, a lock or a half-written file is not the
  run's; copied, a cancel marker would stop the copy's first job.
- **A half copy that looks whole.** The record is written last, and a cancelled copy leaves
  nothing behind.
- **A copy that does not open.** It opens as a clustering run whose node view is the
  original's.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from detail.api import duplicate_analysis
from detail.core.progress import OperationCancelled
from detail.storage import analysis_runs as analysis_store


def _digest(root) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def clustered(fitted_temporal, monkeypatch):
    """A temporal run with a node cache and one saved profile, so the copy has every kind of
    file a real clustering run carries."""
    import detail.api.cluster_profile as cluster_profile
    from tests.api.test_cluster_profile import _node_with_members, _track

    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    view = run.node_view()
    node = _node_with_members(view)
    by_unit: dict = {}
    for m in view.sample_runs(node, n=100_000, seed=0):
        by_unit.setdefault((m["recording_id"], int(m["subject"])), []).append(m)
    monkeypatch.setattr(
        cluster_profile,
        "_trajectory",
        lambda _e, key, _r, _ts: _track(by_unit, key, fast=True),
    )
    experiment.profile_cluster(
        str(run.id), node, {"explainer": "logistic_regression", "n_per_side": 40}
    )
    return experiment, str(run.id), node


def test_the_copy_is_every_file_with_only_its_own_id_changed(clustered):
    experiment, run_id, node = clustered
    source = experiment.analysis_run_dir(run_id)
    new_id = experiment.duplicate_analysis(run_id)
    copy = experiment.analysis_run_dir(new_id)
    assert new_id != run_id and re.fullmatch(r"an_\d+", new_id)
    original, copied = _digest(source), _digest(copy)
    assert set(copied) == set(original), "the copy has other files than the run"
    changed = {path for path in original if original[path] != copied[path]}
    # The documents that name the run; every other byte is the original's.
    assert changed <= {
        "analysis.yaml",
        "job.json",
        "behavior_output.yaml",
        "history.jsonl",
        f"profiles/node_{node}/profile.yaml",
    }, changed
    assert f"profiles/node_{node}/profile.yaml" in changed
    assert "node_cache.zarr/zarr.json" in copied


def test_no_record_of_the_copy_names_the_original(clustered):
    experiment, run_id, node = clustered
    new_id = experiment.duplicate_analysis(run_id)
    copy = experiment.analysis_run_dir(new_id)
    whole = re.compile(rf"(?<![\w]){re.escape(run_id)}(?![\w])")
    for path in sorted(copy.rglob("*")):
        if not path.is_file() or ".zarr" in path.as_posix():
            continue
        if path.suffix not in (".yaml", ".json", ".jsonl", ".csv", ".log"):
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not whole.search(line):
                continue
            # The three places that say where it came from, on purpose.
            assert line.startswith(("duplicated_from:", "name:")) or (
                '"op": "duplicate"' in line
            ), f"{path.relative_to(copy)} still names {run_id}: {line}"
    record = experiment.analysis_record(new_id)
    assert list(record)[0] == "schema_version"
    assert record["id"] == new_id
    assert record["name"] == f"{run_id} (copy)"
    assert record["duplicated_from"] == run_id
    assert experiment.cluster_profile(new_id, node).run_id == new_id
    envelope = json.loads((copy / "job.json").read_text(encoding="utf-8-sig"))
    assert envelope["job"]["job_id"] == new_id and envelope["pid"] is None


def test_the_original_is_untouched(clustered):
    experiment, run_id, _node = clustered
    source = experiment.analysis_run_dir(run_id)
    before = _digest(source)
    experiment.duplicate_analysis(run_id, name="trial")
    assert _digest(source) == before
    assert experiment.analysis_record(run_id).get("name") is None


def test_the_copy_opens_with_the_same_node_view(clustered):
    experiment, run_id, node = clustered
    new_id = experiment.duplicate_analysis(run_id)
    original_run = experiment.temporal_clustering_run(run_id)
    copied_run = experiment.temporal_clustering_run(new_id)
    original, copied = original_run.node_view(), copied_run.node_view()
    assert copied.stale is None, copied.stale
    assert (copied.node_times() == original.node_times()).all()
    # repr: the members' cx/cy are NaN, which never equals itself.
    assert repr(copied_run.typical_members(node)) == repr(
        original_run.typical_members(node)
    )
    assert new_id in [r.id for r in experiment.analysis_runs]
    ref = next(r for r in experiment.analysis_runs if r.id == new_id)
    assert ref.name == f"{run_id} (copy)" and ref.module_id == "temporal_clustering"


def test_a_running_run_is_refused(clustered):
    import os

    from detail.core.dataclasses import JobStatus
    from detail.executor import rundir

    experiment, run_id, _node = clustered
    source = experiment.analysis_run_dir(run_id)
    rundir.set_status(source, JobStatus.RUNNING)
    # A live worker (this process), so reconciling does not fail the run under the test.
    envelope = json.loads((source / "job.json").read_text(encoding="utf-8-sig"))
    envelope["pid"] = os.getpid()
    (source / "job.json").write_text(json.dumps(envelope), encoding="utf-8")
    record = experiment.analysis_record(run_id)
    analysis_store.write_analysis_yaml(source, {**record, "status": "running"})
    before = [r.id for r in experiment.analysis_runs]
    with pytest.raises(ValueError, match="still running"):
        experiment.duplicate_analysis(run_id)
    assert [r.id for r in experiment.analysis_runs] == before


def test_process_state_is_not_copied(clustered):
    experiment, run_id, _node = clustered
    source = experiment.analysis_run_dir(run_id)
    for name in ("cancel", "stop", "analysis.yaml.tmp", "frames.lock"):
        (source / name).write_text("", encoding="utf-8")
    new_id = experiment.duplicate_analysis(run_id)
    copy = experiment.analysis_run_dir(new_id)
    for name in ("cancel", "stop", "analysis.yaml.tmp", "frames.lock"):
        assert not (copy / name).exists(), name


def test_the_copy_reports_progress_and_a_cancel_leaves_nothing(clustered):
    experiment, run_id, _node = clustered
    heard: list = []
    experiment.duplicate_analysis(
        run_id, progress=lambda s, f, d: heard.append((s, f, d))
    )
    fractions = [f for _s, f, _d in heard]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any(0 < f < 1 for f in fractions), "the bar never moved between 0 and 1"

    before = set(analysis_store.discover_analyses(experiment.dir))
    claimed = {p.name for p in (experiment.dir / "analysis").iterdir()}

    def cancel(_stage, fraction, _detail):
        if fraction > 0.3:
            raise OperationCancelled("stop")

    with pytest.raises(OperationCancelled):
        experiment.duplicate_analysis(run_id, progress=cancel)
    assert set(analysis_store.discover_analyses(experiment.dir)) == before
    assert {p.name for p in (experiment.dir / "analysis").iterdir()} == claimed


def test_an_unknown_run_is_refused(clustered):
    experiment, _run_id, _node = clustered
    with pytest.raises(FileNotFoundError):
        experiment.duplicate_analysis("an_99")


def test_only_a_value_that_is_the_id_is_rewritten():
    doc = {"id": "an_1", "note": "from an_1", "list": ["an_1", "an_10"], "an_1": 3}
    assert duplicate_analysis.rewrite_ids(doc, "an_1", "an_7") == {
        "id": "an_7",
        "note": "from an_1",
        "list": ["an_7", "an_10"],
        "an_7": 3,
    }
