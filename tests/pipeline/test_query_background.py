"""The background a §20 method is scored against — drawn by the module, never by the method.

The draw is the one place a query run could acquire something a user does not have. §20 puts
it in the module so every draw is recorded identically, and CLAUDE.md's "Labels score a method;
they never build one" is why the exclusion rule is exactly one line long: **the exemplar spans
come out and nothing else does.**

That is not a conservative simplification, it is the whole point. #191 measured what the other
choice costs: sampling negatives from the frames known *not* to be positives inflated a typical
exemplar's score far past what a user could ever reproduce, because at the moment
they run a search they have no labels to exclude by. In the tool there are none at all, so the
remainder *is* the unlabeled trajectory, positives and all. A future change that narrows the
pool by what merely *looks* like the behavior would rebuild that leak in a new place and would
look, from the outside, like an improvement.

The tests here therefore pin two different things: that the recorded draw is faithful (so a
result can be audited months later), and that the draw is a function of the user's own inputs
and nothing else.
"""

import threading

import numpy as np
import pytest

from detail.pipeline.query import QueryModule, draw_background, resolve_method
from detail.storage import analysis_runs as analysis_store
from detail.storage import query_runs

from tests.pipeline.test_query_module import PERIOD, _config, _experiment


def _two_behaviors(recording="recording_001"):
    """Two behaviors whose example clips sit at known, disjoint row ranges."""

    def clip(start, end):
        return {
            "recording_id": recording,
            "track_id": 0,
            "start_t_ns": start * PERIOD,
            "end_t_ns": end * PERIOD,
        }

    return [
        {"name": "rearing", "examples": [clip(100, 119)]},
        {"name": "grooming", "examples": [clip(300, 319)]},
    ]


def _job(tmp_path, method_id="soft_knn", **kw):
    proj, _ = _experiment(tmp_path, ["recording_001"])
    config = _config(
        ["recording_001"], behaviors=_two_behaviors(), method=method_id, **kw
    )
    module = QueryModule()
    normalized = module.validate(proj.dir, config)
    job = module.build_job("an_01", normalized)
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    return proj, job, run_dir


def _draw(tmp_path, **kw):
    proj, job, run_dir = _job(tmp_path, **kw)
    method = resolve_method(job.method)
    return draw_background(proj.dir, job, method, run_dir), run_dir


# --------------------------------------------------------------------------- #
# what comes out, and what stays in
# --------------------------------------------------------------------------- #
def test_no_drawn_row_falls_inside_any_behavior_s_examples(tmp_path):
    """*Any* behavior's, not just the one being prepared. §20 draws one background per run,
    so a per-behavior exclusion would make the draw depend on which behavior is being scored
    for no gain — and would quietly put behavior A's exemplars into the pool that calibrates
    behavior B."""
    sample, _run_dir = _draw(tmp_path)
    excluded = set(range(100, 120)) | set(range(300, 320))
    overlapping = [r for r in sample.rows if r in excluded]
    assert (
        not overlapping
    ), f"{len(overlapping)} drawn rows sit inside an example clip: {overlapping[:8]}"


def test_the_pool_is_the_rest_of_the_trajectory_and_not_a_clean_subset(tmp_path):
    """The positive half of the same rule. A draw that avoided anything *resembling* the
    exemplars would be the #191 leak wearing a different name, and it would show up here as
    a pool suspiciously smaller than the remainder."""
    sample, _run_dir = _draw(tmp_path)
    # 600 rows in the fixture, 40 inside example clips, so 560 are eligible and the request
    # (500) is satisfiable from them alone.
    assert len(sample) == 500
    assert (
        max(sample.rows) >= 500
    ), "the draw never reached the far end of the trajectory"


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
def test_the_draw_is_recorded_with_its_strategy_scope_size_and_seed(tmp_path):
    sample, run_dir = _draw(tmp_path)
    stored = query_runs.read_background(run_dir)
    assert (stored.strategy, stored.scope, stored.size) == ("uniform", "cohort", 500)
    assert stored.seed == 0
    assert len(stored.rows) == len(stored) == len(stored.recording_ids)
    np.testing.assert_array_equal(stored.vectors, sample.vectors)


def test_the_recorded_rows_are_the_vectors_that_were_handed_to_the_method(tmp_path):
    """The addresses are only worth recording if they describe the vectors actually used.
    Read the trajectory back and check row by row."""
    from detail.api.run_preview import load_track_trajectory
    from detail.storage import experiment as experiment_store

    proj, job, run_dir = _job(tmp_path)
    sample = draw_background(proj.dir, job, resolve_method(job.method), run_dir)
    emb, _frame_idx, _t_ns = load_track_trajectory(
        experiment_store.recording_path(proj.dir, "recording_001"), job.embed_run_id, 0
    )
    for index in (0, len(sample) // 2, len(sample) - 1):
        np.testing.assert_array_equal(sample.vectors[index], emb[sample.rows[index]])


def test_the_seed_reaches_the_run_record(tmp_path):
    """`job_params` derives the record from the job, so a field added to the job lands there
    by construction — but only if `build_job` carries it off the config."""
    from detail.executor.runners import job_params

    _proj, job, _run_dir = _job(tmp_path, background_seed=99)
    assert job.background_seed == 99
    assert job_params(job)["background_seed"] == 99


# --------------------------------------------------------------------------- #
# determinism and resume
# --------------------------------------------------------------------------- #
def test_the_same_seed_draws_the_same_rows(tmp_path):
    first, _ = _draw(tmp_path / "a", background_seed=4)
    second, _ = _draw(tmp_path / "b", background_seed=4)
    assert first.rows == second.rows


def test_a_different_seed_draws_different_rows(tmp_path):
    """Or the seed is inert and 'reproducible' means 'there was only ever one answer'."""
    first, _ = _draw(tmp_path / "a", background_seed=1)
    second, _ = _draw(tmp_path / "b", background_seed=2)
    assert first.rows != second.rows


def test_a_resumed_run_reuses_the_stored_draw_rather_than_redrawing(tmp_path):
    """A seed reproduces a draw only against an unchanged trajectory. Reading back what was
    written is what makes a resumed run score against the same background it started with —
    and what the recorded rows describe."""
    proj, job, run_dir = _job(tmp_path)
    method = resolve_method(job.method)
    first = draw_background(proj.dir, job, method, run_dir)

    # A redraw would notice this; a reload cannot.
    job.background_seed = 12345
    second = draw_background(proj.dir, job, method, run_dir)
    assert second.rows == first.rows
    assert second.seed == first.seed


# --------------------------------------------------------------------------- #
# the leakage guard, in the shape §20 needs
# --------------------------------------------------------------------------- #
def test_the_draw_consults_no_behavior_stream(tmp_path, monkeypatch):
    """`tests/pipeline/test_label_leakage.py` permutes a stream and compares predictions, but
    its method list is hand-written over the *eval* ids, so a §20 registry method is not
    covered by it. The §20 module needs a different form of the same question anyway: it never
    receives labels at all — exemplars arrive as user-chosen clip spans — so what has to hold
    is that the draw reads no label source whatsoever.

    Asserted by making every stream read raise. A draw that touched one would fail loudly
    instead of quietly returning a pool a user could not reproduce.
    """
    from detail.api.recording import Recording

    def refuse(*_args, **_kwargs):
        raise AssertionError(
            "the background draw read a behavior stream; the pool must be a function of the "
            "trajectories and the user's own clips, never of ground truth"
        )

    monkeypatch.setattr(Recording, "stream_events", refuse)
    monkeypatch.setattr(Recording, "streams", property(refuse))
    sample, _run_dir = _draw(tmp_path)
    assert len(sample) == 500


def test_a_method_that_wants_no_background_gets_none_and_writes_no_file(
    tmp_path, monkeypatch
):
    """A method that declares `sampling = None` has to be untouched by all of this. None is
    registered since `dtw` was removed, so the declaration is patched onto the default.
    """
    from detail.query_methods.soft_knn import SoftKnnQueryMethod

    monkeypatch.setattr(SoftKnnQueryMethod, "sampling", None)
    proj, job, run_dir = _job(tmp_path, method_id="soft_knn")
    assert draw_background(proj.dir, job, resolve_method("soft_knn"), run_dir) is None
    assert not query_runs.background_path(run_dir).is_file()


# --------------------------------------------------------------------------- #
# end to end through the runner
# --------------------------------------------------------------------------- #
def test_a_soft_knn_run_writes_its_background_and_searches(tmp_path):
    """Through `runner`, so the barrier ordering is exercised: the draw happens once, before
    any unit, and every unit is scored against the same one."""
    proj, job, run_dir = _job(tmp_path)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    events: list = []
    QueryModule().runner(job, run_dir, events.append, threading.Event())

    assert query_runs.background_path(run_dir).is_file()
    # `search/<behavior>/<recording>__<subject>.npz` — one directory per behavior.
    shards = sorted(query_runs.search_dir(run_dir).rglob("*.npz"))
    assert len(shards) == 2, f"expected one shard per behavior, got {shards}"
    assert any("preparing" in e.message for e in events), (
        "the background draw streams every searched subject with the fraction pinned at 0; "
        f"without its own phase message that reads as a hang: {[e.message for e in events]}"
    )
