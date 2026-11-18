"""A query search that did not reach every unit says so, and cannot leave the run as if it had.

**The class.** A roll-up computed over the plan rather than over the evidence. A query run's
coverage — the §19 table that turns "no rows" into "a real zero" — was written for every
recording the search was *asked* to cover, before anything checked that the search had reached
it. A unit whose shard is missing (the search was cancelled before it, or one behavior's search
raised on it) therefore read as "searched, found nothing": its CSV export was header-only, every
rate was diluted by time nobody searched, and ``finalize`` turned the interrupted ``partial`` run
into ``complete`` on the user's say-so. Nothing was wrong at the line that appended the coverage
row; the error was only visible as a clean, confident, too-low number.

**How it hid.** Every fixture ran the search to completion, so the plan and the evidence always
agreed. The comment at the skipped shard ("its coverage stands") stated the bug as a rule.

**The rule.**

- Coverage holds a unit only where every behavior the artifact names (its vocabulary) has a
  shard for it. ``coverage.csv`` is per unit, not per behavior (§19), so a unit that one named
  behavior never searched cannot be stated as covered for any of them; its rows return when the
  search is re-run.
- A run whose search did not reach every unit — ``partial``, ``failed``, or ``open`` with a
  behavior in ``behaviors_failed`` — refuses to be finalized or exported (``pin``, which every
  number leaving the run goes through), naming the units it is missing and the way back:
  re-run the search. Calibrating it is still allowed: the curve over the searched units is
  real, and the artifact it projects now only states what was searched.

The cancel is simulated the way the review reproduced it: the runner's stop event is set after
the first unit's search, through a wrapper scoped to one test.
"""

import threading

import pytest

from detail.api.experiment import Experiment
from detail.pipeline.query import QueryModule
from detail.storage import analysis_runs as analysis_store

from tests.api.test_query import N_EMBEDDED, _add_video, _two_behaviors
from tests.pipeline.test_query_module import _config, _experiment


def _search(tmp_path, monkeypatch, *, behaviors=None, after_unit=None):
    """Run the real search over s1, s2 with ``after_unit(unit, stop)`` called after each unit."""
    proj, _rows = _experiment(tmp_path, ["s1", "s2"], n_rows=N_EMBEDDED)
    for sid in ("s1", "s2"):
        _add_video(proj.get_recording(sid).dir)
    module = QueryModule()
    normalized = module.validate(
        proj.dir, dict(_config(["s1", "s2"], behaviors=behaviors))
    )
    job = module.build_job("an_01", normalized)
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    stop = threading.Event()
    original = QueryModule._search_unit

    def wrapped(self, experiment_root, run_dir, job, method, state, unit, progress):
        if after_unit is not None:
            after_unit(unit, stop, before=True)
        original(self, experiment_root, run_dir, job, method, state, unit, progress)
        if after_unit is not None:
            after_unit(unit, stop, before=False)

    monkeypatch.setattr(QueryModule, "_search_unit", wrapped)
    module.runner(job, run_dir, lambda _e: None, stop)
    monkeypatch.setattr(QueryModule, "_search_unit", original)
    return Experiment(proj.dir).query_run("an_01")


def _cancel_after_first(unit, stop, *, before):
    if not before:
        stop.set()


def _fail_q02_on_s2(unit, stop, *, before):
    if before and unit.behavior_id == "q02" and unit.recording_id == "s2":
        raise RuntimeError("search fell over")


def test_a_cancelled_search_covers_only_the_units_it_reached(tmp_path, monkeypatch):
    run = _search(tmp_path, monkeypatch, after_unit=_cancel_after_first)
    assert run.status == "partial"

    run.set_cutoff("q01", 0.5)  # calibrating what was searched is still allowed

    output = run.artifact()
    assert {row.recording_id for row in output.coverage} == {"s1"}
    assert {row.recording_id for row in output.instances} <= {"s1"}


def test_a_cancelled_search_refuses_finalize_and_export(tmp_path, monkeypatch):
    run = _search(tmp_path, monkeypatch, after_unit=_cancel_after_first)
    run.set_cutoff("q01", 0.5)

    with pytest.raises(ValueError) as finalize:
        run.finalize()
    message = str(finalize.value)
    assert "s2" in message and "re-run the search" in message
    assert run.status == "partial"  # not turned into `complete`

    with pytest.raises(ValueError, match="re-run the search"):
        run.export(tmp_path / "export")
    assert not (tmp_path / "export" / "s2.csv").exists()


def test_a_behavior_that_failed_on_a_unit_takes_that_unit_out_of_coverage(
    tmp_path, monkeypatch
):
    """``behaviors_failed`` is the other way a unit goes unsearched: the run is ``open``, but
    the named behavior never looked at s2, so s2 cannot be a real zero for it."""
    run = _search(
        tmp_path,
        monkeypatch,
        behaviors=_two_behaviors(["s1", "s2"]),
        after_unit=_fail_q02_on_s2,
    )
    assert run.status == "open"

    run.set_cutoff("q01", 0.5)  # q01 searched both units: both are covered
    assert {row.recording_id for row in run.artifact().coverage} == {"s1", "s2"}

    run.set_cutoff("q02", 0.5)  # q02 never searched s2
    output = run.artifact()
    assert {row.recording_id for row in output.coverage} == {"s1"}
    assert {row.recording_id for row in output.instances} <= {"s1"}

    with pytest.raises(ValueError) as finalize:
        run.finalize()
    assert "q02" in str(finalize.value) and "s2" in str(finalize.value)
    assert run.status == "open"


def test_a_search_that_reached_every_unit_still_finalizes(tmp_path, monkeypatch):
    """The negative case: the refusal is about missing units, not about the method."""
    run = _search(tmp_path, monkeypatch)
    run.set_cutoff("q01", 0.5)
    run.finalize()
    assert run.status == "complete"


def test_the_cli_refuses_to_finalize_or_export_a_cancelled_search(
    tmp_path, monkeypatch
):
    """The two commands a user types say why, rather than printing a traceback or succeeding."""
    from click.testing import CliRunner

    from detail.cli import main

    run = _search(tmp_path, monkeypatch, after_unit=_cancel_after_first)
    experiment = str(run._experiment.dir)
    runner = CliRunner()
    finalize = runner.invoke(
        main,
        ["analysis", "query-calibrate", "--experiment", experiment, "--run", "an_01"]
        + ["--cutoff", "q01=0.5", "--finalize"],
    )
    assert finalize.exit_code == 1, finalize.output
    assert "s2" in finalize.output and "re-run the search" in finalize.output
    assert not isinstance(finalize.exception, ValueError)

    export = runner.invoke(
        main,
        ["analysis", "export-query", "--experiment", experiment, "--run", "an_01"]
        + ["--out", str(tmp_path / "export")],
    )
    assert export.exit_code == 1, export.output
    assert "re-run the search" in export.output
