"""Tests for the §20 Query search module (detail/pipeline/query.py).

Most of what is checked here is a claim the module makes that a smoke test would pass through:
that every bad behavior is reported together rather than one relaunch at a time; that a unit is
`(behavior x recording x subject)` and not a recording; that a chunked search agrees with an
unchunked one; and that the bar moves *inside* a single unit, which is the case a multi-unit
fixture cannot see.
"""

import threading
import zlib

import numpy as np
import pytest
import yaml

from detail.api.experiment import Experiment
from detail.configs.query_soft_knn import SoftKnnQueryConfig
from detail.core import progress as progress_mod
from detail.pipeline import query as query_module
from detail.pipeline.query import QueryModule, _plan_chunks, resolve_method
from detail.query_methods.soft_knn import SoftKnnQueryMethod
from detail.storage import analysis_runs as analysis_store
from detail.storage import query_runs
from detail.storage.zarr_io import EmbeddingPartWriter
from tests.pipeline.phase_messages import assert_messages_name_their_phases

DIM = 64
PERIOD = 1_000_000  # 1 ms between embedded frames


def _motif(length=20, seed=7):
    return np.random.default_rng(seed).normal(size=(length, DIM)).astype(np.float32)


def _write_embeddings(
    recording_dir, n_rows, plant=None, at=0, embed_run_id="emb_01", cx=None
):
    """A track's embedding run, optionally with `plant` written in at row `at`.

    Seeded off the recording's **name**, through a stable hash. Two things were wrong with
    the previous `abs(hash(str(recording_dir))) % 2**32`, and both defeated the intent of
    giving each recording its own reproducible draw: `hash()` on a `str` is salted per
    process by `PYTHONHASHSEED`, and `recording_dir` sits under a `tmp_path` that carries a
    fresh number every run. So every test built on this fixture scored a different draw each
    time, and a failure could not be reproduced from the command that produced it — which is
    how `test_browse_scores_are_distinguishable` failed roughly one run in five and passed
    every time it was re-run on its own.
    """
    rng = np.random.default_rng(zlib.crc32(recording_dir.name.encode()))
    emb = rng.normal(size=(n_rows, DIM)).astype(np.float32)
    if plant is not None:
        emb[at : at + len(plant)] = plant
    run_dir = recording_dir / "embeddings" / embed_run_id
    store = run_dir / "track_0" / "part_001.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    with EmbeddingPartWriter(store, output_dim=DIM) as w:
        for i in range(n_rows):
            w.add_row(i, i * PERIOD, 0, emb[i])
    # §14's kinematics sibling, only when a test asks for it — mirroring `make_embedding_run`.
    # A run made before kinematics existed has none, and that is the state most of this
    # module's fixtures should reproduce.
    if cx is not None:
        from detail.storage.zarr_io import KinematicsPartWriter, kinematics_part_path

        with KinematicsPartWriter(kinematics_part_path(store)) as kin:
            for i in range(n_rows):
                kin.add_row(
                    i,
                    i * PERIOD,
                    float(cx[i]),
                    float(cx[i]),
                    0.05,
                    [0.0, 0.0, 1.0, 1.0],
                )
    (run_dir / "embedding.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": embed_run_id,
                "tracking_run_id": "track_01",
                "model_id": "dinov3_convnext_base",
                "model_variant": "base",
                "constructor": "full_frame",
                "output_dim": DIM,
                "track_ids": [0],
                "renderer_spec": [{"type": "init_binary_mask"}],
                "status": "complete",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _experiment(tmp_path, recordings, n_rows=600, plant_at=100, cx=None):
    """An experiment whose recordings each carry the same motif at `plant_at`."""
    proj = Experiment.create("p", tmp_path)
    motif = _motif()
    for sid in recordings:
        sess = proj.create_recording(sid)
        _write_embeddings(sess.dir, n_rows, plant=motif, at=plant_at, cx=cx)
    return proj, motif


def _config(recordings, behaviors=None, plant_at=100, **kw):
    examples = [
        {
            "recording_id": recordings[0],
            "track_id": 0,
            "start_t_ns": plant_at * PERIOD,
            "end_t_ns": (plant_at + 19) * PERIOD,
        }
    ]
    config = {
        "source_recording_ids": list(recordings),
        "embed_run_id": "emb_01",
        "track_id": 0,
        "method": "soft_knn",
        "method_params": {},
        "behaviors": (
            behaviors
            if behaviors is not None
            else [{"name": "rearing", "examples": examples}]
        ),
    }
    config.update(kw)
    return config


def _run(proj, config, run_id="an_01", progress=None, module=None):
    """Validate, build and run, returning `(run_dir, job, events)`."""
    module = module or QueryModule()
    normalized = module.validate(proj.dir, dict(config))
    job = module.build_job(run_id, normalized)
    run_dir = analysis_store.analysis_dir(proj.dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": run_id, "module_id": "query", "status": "running"}
    )
    events = [] if progress is None else progress
    module.runner(job, run_dir, events.append, threading.Event())
    return run_dir, job, events


# --------------------------------------------------------------------------- #
# validation: everything wrong, reported at once
# --------------------------------------------------------------------------- #
def test_every_bad_behavior_is_reported_together(tmp_path):
    """Tenet 10, and the reason §20 states it for this module specifically.

    A user who marked five behaviors and got two of them wrong should see both, not discover
    the second after fixing the first and relaunching a cohort-scale search.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"])
    behaviors = [
        {"name": "good", "examples": [
            {"recording_id": "recording_001", "track_id": 0,
             "start_t_ns": 100 * PERIOD, "end_t_ns": 119 * PERIOD},
        ]},  # fmt: skip
        {"name": "no clips", "examples": []},
        {"name": "off recording", "examples": [
            {"recording_id": "recording_999", "track_id": 0,
             "start_t_ns": 0, "end_t_ns": 10 * PERIOD},
        ]},  # fmt: skip
    ]
    with pytest.raises(ValueError) as excinfo:
        QueryModule().validate(
            proj.dir, _config(["recording_001"], behaviors=behaviors)
        )
    message = str(excinfo.value)
    assert "no clips" in message and "off recording" in message, message
    assert "good" not in message  # the one that was fine is not named


def test_an_unknown_method_names_what_is_registered(tmp_path):
    proj, _ = _experiment(tmp_path, ["recording_001"])
    with pytest.raises(ValueError, match="registered:"):
        QueryModule().validate(
            proj.dir, _config(["recording_001"], method="not_a_method")
        )


def test_the_config_carries_no_threshold(tmp_path):
    """§20 makes the absence structural: a search that bakes in a cutoff means changing your
    mind costs another search."""
    from detail.configs.query import QueryConfig

    fields = set(QueryConfig.model_fields)
    assert not fields & {"threshold", "cutoff"}, fields


def test_the_default_method_is_one_that_is_registered():
    """§20 used to declare no default at all, because nothing separated the two methods. That
    held while nothing did; `soft_knn` is now measured at roughly thirty times `dtw`'s speed
    on a real recording, so the default is picked on **cost** and the surfaces say so.

    What has to stay true is that the default names a method that exists: a default resolving
    to nothing fails at launch for a user, on a field most of them never open.
    """
    from detail.configs.query import DEFAULT_METHOD, QueryConfig

    assert QueryConfig.model_fields["method"].default == DEFAULT_METHOD
    assert resolve_method(DEFAULT_METHOD).method_id == DEFAULT_METHOD


def test_a_run_that_names_no_method_gets_the_default(tmp_path):
    """The point of the default, end to end: a config with no `method` key still validates."""
    proj, _ = _experiment(tmp_path, ["recording_001"])
    config = _config(["recording_001"])
    config.pop("method")
    from detail.configs.query import DEFAULT_METHOD

    assert QueryModule().validate(proj.dir, config)["method"] == DEFAULT_METHOD


def _registered_query_methods():
    from detail.core.registries import QUERY_METHODS
    from detail.query_methods import import_query_methods

    import_query_methods()
    return sorted(QUERY_METHODS)


def _experiment_with_a_gap(tmp_path):
    """One recording whose track_0 embedding has a 150-frame hole between rows 50 and 200."""
    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    run_dir = sess.dir / "embeddings" / "emb_01"
    store = run_dir / "track_0" / "part_001.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = [*range(0, 50), *range(200, 250)]  # a 150-frame hole
    with EmbeddingPartWriter(store, output_dim=DIM) as w:
        for i in rows:
            w.add_row(i, i * PERIOD, 0, rng.normal(size=DIM).astype(np.float32))
    (run_dir / "embedding.yaml").write_text(
        yaml.dump({"schema_version": 1, "id": "emb_01", "tracking_run_id": "track_01",
                   "model_id": "dinov3_convnext_base", "model_variant": "base",
                   "constructor": "full_frame", "output_dim": DIM, "track_ids": [0],
                   "renderer_spec": [{"type": "init_binary_mask"}], "status": "complete"},
                  sort_keys=False)
    , encoding="utf-8")  # fmt: skip
    return proj


@pytest.mark.parametrize("method_id", _registered_query_methods())
def test_a_gap_crossing_exemplar_is_refused_only_where_the_method_needs_a_path(
    tmp_path, method_id
):
    """Whose rule this is, asserted against the method's own declaration.

    `dtw` (since removed) matched a path, so a clip straddling a break was two clips and it
    said so. `soft_knn` vstacks every exemplar frame into one unordered index and reads nothing
    back but the length, so the same clip is usable — and refusing it was the commonest way a
    user's marked clip was rejected. The rule lived in the module, which meant the default
    method inherited a requirement it does not have.

    Data-driven over the registry rather than written twice, so a method registered tomorrow
    is held to whatever its `ExemplarBounds` claims the day it is added.
    """
    proj = _experiment_with_a_gap(tmp_path)
    config = _config(["recording_001"], method=method_id)
    config["behaviors"][0]["examples"][0] = {
        "recording_id": "recording_001",
        "track_id": 0,
        "start_t_ns": 40 * PERIOD,
        "end_t_ns": 210 * PERIOD,  # straddles the hole
    }
    if resolve_method(method_id).exemplars.requires_contiguous:
        with pytest.raises(ValueError, match="break in tracking"):
            QueryModule().validate(proj.dir, config)
    else:
        resolved = QueryModule().validate(proj.dir, config)
        # Accepted *and* kept: a silent drop would leave the behavior with no examples and
        # read as success. The span snaps to the rows it covers, both sides of the hole.
        (example,) = resolved["behaviors"][0]["examples"]
        assert example["start_t_ns"] <= 49 * PERIOD
        assert example["end_t_ns"] >= 200 * PERIOD


def test_a_method_that_cannot_use_negatives_refuses_them(tmp_path, monkeypatch):
    # A method that has nothing to score non-examples against refuses them rather than
    # silently discarding them (the empty-affordance rule). No registered method declares
    # that today, so the declaration is patched onto the default one: the refusal reads the
    # method's own `negatives`, which is what this checks.
    from detail.core.dataclasses import NegativeSupport

    monkeypatch.setattr(
        SoftKnnQueryMethod, "negatives", NegativeSupport(support="unusable")
    )
    proj, _ = _experiment(tmp_path, ["recording_001"])
    config = _config(["recording_001"])
    config["behaviors"][0]["negatives"] = [
        {"recording_id": "recording_001", "track_id": 0,
         "start_t_ns": 300 * PERIOD, "end_t_ns": 319 * PERIOD},
    ]  # fmt: skip
    with pytest.raises(ValueError, match="cannot use non-example clips"):
        QueryModule().validate(proj.dir, config)


# --------------------------------------------------------------------------- #
# method parameters: resolved against the method's own model, written back
# --------------------------------------------------------------------------- #
def test_method_params_are_resolved_and_written_back(tmp_path):
    """Tenet 9 for the method's half of the config: the record carries the value that was
    used, not the request that asked for one. `query.yaml` documents "the method and its
    **resolved** params", and `validate` used to pass the raw dict straight through — so a
    run launched with nothing set recorded `{}` and left a reader to guess the defaults.

    Strings, because that is what `--method-param name=value` produces; the method's model
    is the only thing that knows each field's type.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"])
    config = _config(["recording_001"])
    config["method_params"] = {"k": "5", "duration_max_mult": "3"}
    resolved = QueryModule().validate(proj.dir, config)["method_params"]

    assert set(resolved) == set(
        SoftKnnQueryConfig.model_fields
    ), "a field went unrecorded"
    assert resolved["k"] == 5
    assert resolved["duration_max_mult"] == 3.0  # coerced, not left a string
    assert resolved["extent_fraction"] == SoftKnnQueryConfig().extent_fraction


def test_an_unknown_method_param_is_refused_rather_than_ignored(tmp_path):
    """Pydantic drops extras, so a typo validated clean and ran on the default — the same
    "the knob did nothing and nothing said so" defect this whole path exists to close.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"])
    config = _config(["recording_001"])
    config["method_params"] = {"kk": "5"}
    with pytest.raises(ValueError, match="soft_knn has no parameter"):
        QueryModule().validate(proj.dir, config)


def test_an_out_of_range_method_param_is_refused_with_the_bound(tmp_path):
    proj, _ = _experiment(tmp_path, ["recording_001"])
    config = _config(["recording_001"])
    config["method_params"] = {"extent_fraction": "5"}  # lt=1.0
    with pytest.raises(ValueError, match="extent_fraction"):
        QueryModule().validate(proj.dir, config)


# --------------------------------------------------------------------------- #
# the fan-out: a unit is (behavior x recording x subject)
# --------------------------------------------------------------------------- #
def test_a_unit_is_a_behavior_times_a_recording(tmp_path):
    """Two behaviors over three recordings is six searches, not three.

    §20 calls this out because the aggregate bar reads its outer count as units: labelling
    them recordings makes a two-behavior run over six recordings claim twelve recordings.
    """
    recordings = ["recording_001", "recording_002", "recording_003"]
    proj, _ = _experiment(tmp_path, recordings)
    behaviors = [
        {"name": n, "examples": [
            {"recording_id": "recording_001", "track_id": 0,
             "start_t_ns": 100 * PERIOD, "end_t_ns": 119 * PERIOD},
        ]}  # fmt: skip
        for n in ("rearing", "grooming")
    ]
    module = QueryModule()
    normalized = module.validate(proj.dir, _config(recordings, behaviors=behaviors))
    job = module.build_job("an_01", normalized)
    assert len(job.units) == 6
    assert {u.behavior_id for u in job.units} == {"q01", "q02"}


def test_the_search_finds_the_planted_motif_in_every_recording(tmp_path):
    # Assert content, not that the call returned: the whole module exists to find this.
    recordings = ["recording_001", "recording_002"]
    proj, _ = _experiment(tmp_path, recordings, plant_at=100)
    run_dir, _job, _events = _run(proj, _config(recordings))
    method = resolve_method("soft_knn")
    for sid in recordings:
        path = query_runs.shard_path(run_dir, "q01", sid, 0, method.shard_suffix)
        best, runner_up = method.instances_at(method.load(path), 0.0, {})[:2]
        assert abs(best.start_row - 100) <= 2, (sid, best.start_row)
        # The background is drawn from the unlabeled remainder, which holds the planted
        # motif too, so the peak is a vote rather than a certainty. What has to hold is that
        # the motif stands well clear of everything the noise produces.
        assert best.confidence > 2 * runner_up.confidence, (best, runner_up)


def test_a_completed_shard_is_never_recomputed(tmp_path):
    """Resume keys off the shard. Re-running must search nothing already on disk."""
    proj, _ = _experiment(tmp_path, ["recording_001"])
    module = QueryModule()
    run_dir, job, _ = _run(proj, _config(["recording_001"]), module=module)
    method = resolve_method("soft_knn")
    shard = query_runs.shard_path(
        run_dir, "q01", "recording_001", 0, method.shard_suffix
    )
    before = shard.stat().st_mtime_ns

    module.runner(job, run_dir, lambda _e: None, threading.Event())
    assert shard.stat().st_mtime_ns == before, "a completed shard was recomputed"


def test_one_behaviors_failure_leaves_the_others_reviewable(tmp_path, monkeypatch):
    """§20: one failure voiding a cohort-scale search is the outcome resume exists to prevent."""
    recordings = ["recording_001"]
    proj, _ = _experiment(tmp_path, recordings)
    behaviors = [
        {"name": n, "examples": [
            {"recording_id": "recording_001", "track_id": 0,
             "start_t_ns": 100 * PERIOD, "end_t_ns": 119 * PERIOD},
        ]}  # fmt: skip
        for n in ("good", "doomed")
    ]
    module = QueryModule()
    normalized = module.validate(proj.dir, _config(recordings, behaviors=behaviors))
    job = module.build_job("an_01", normalized)

    real = module._search_unit

    def explode(experiment_root, run_dir, job_, method, state, unit, progress):
        if unit.behavior_id == "q02":
            raise RuntimeError("this behavior's search failed")
        return real(experiment_root, run_dir, job_, method, state, unit, progress)

    monkeypatch.setattr(module, "_search_unit", explode)
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    module.runner(job, run_dir, lambda _e: None, threading.Event())

    record = analysis_store.read_analysis_yaml(run_dir)
    assert record["behaviors_searched"] == ["q01"]
    assert "q02" in record["behaviors_failed"]
    method = resolve_method("soft_knn")
    assert query_runs.shard_path(
        run_dir, "q01", "recording_001", 0, method.shard_suffix
    ).is_file()


# --------------------------------------------------------------------------- #
# the authored lifecycle (§21)
# --------------------------------------------------------------------------- #
def test_the_search_leaves_the_run_open_never_complete(tmp_path):
    """An authored run's compute writes the record back to `open`; only the user closes it."""
    proj, _ = _experiment(tmp_path, ["recording_001"])
    run_dir, _job, _ = _run(proj, _config(["recording_001"]))
    assert analysis_store.read_analysis_yaml(run_dir)["status"] == "open"


def test_the_runner_initializes_its_own_authored_state(tmp_path):
    """The generic GUI launch is module-agnostic and never pre-writes `query.yaml`, so the
    runner must be self-sufficient from the job — the reason clip_annotation did the same
    before it was removed.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"])
    run_dir, _job, _ = _run(proj, _config(["recording_001"]))
    state = query_runs.read_query_state(run_dir)
    assert [(b.id, b.name) for b in state.behaviors] == [("q01", "rearing")]
    assert state.behaviors[0].cutoff is None  # searched, not yet calibrated


def test_the_authored_state_carries_no_status_and_no_counters(tmp_path):
    """§21's one-writer invariant. Status lives in `analysis.yaml` and nowhere else; anything
    countable is derived. Dual writers on one fact drift while both files stay well-formed.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"])
    run_dir, _job, _ = _run(proj, _config(["recording_001"]))
    raw = yaml.safe_load(
        query_runs.query_yaml_path(run_dir).read_text(encoding="utf-8-sig")
    )
    assert "status" not in raw
    assert not [k for k in raw if k.startswith("n_") or k.endswith("_count")], raw


# --------------------------------------------------------------------------- #
# progress: the bar moves inside one unit
# --------------------------------------------------------------------------- #
def test_progress_moves_within_a_single_search_unit(tmp_path, monkeypatch):
    """The cadence guard, on the fixture the rule exists for: **one** unit.

    A run with several units satisfies `0 < fraction < 1` from the outer count alone, so it
    passes while the single-unit case — one behavior, one recording, the ordinary first run —
    sits dead at 0 and jumps to 100. Written over one unit so only the inner pair can move it.
    """
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    # The method reports once per scored block; the fixture's 600 rows would fit in one, so
    # the inner pair could not move and this would pass only on the outer count it excludes.
    monkeypatch.setattr("detail.query_methods.soft_knn.SCORE_BLOCK", 64)
    proj, _ = _experiment(tmp_path, ["recording_001"])
    _run_dir, job, events = _run(proj, _config(["recording_001"]))
    assert len(job.units) == 1

    fracs = [
        (
            e.recordings_done
            + (e.frames_done / e.frames_total if e.frames_total else 0.0)
        )
        / e.recordings_total
        for e in events
    ]
    assert any(0.0 < f < 1.0 for f in fracs), fracs
    assert fracs[-1] == 1.0
    assert any(e.eta_s is not None for e in events)


def test_phase_messages_advance_through_the_run(tmp_path, monkeypatch):
    """The number moving is not enough. `preparing` happens once before fan-out, where the
    fraction is pinned to a unit boundary and cannot move — a static label there reads as a
    hang however healthy the run is. Collapse the phases to one and this must fail.
    """
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    proj, _ = _experiment(tmp_path, ["recording_001"])
    _run_dir, _job, events = _run(proj, _config(["recording_001"]))
    messages = [e.message for e in events if e.message]
    # Derived from the module's own declaration (§16, issue #393), which is these same four
    # constants — a phase declared but never actually emitted fails right here rather than only
    # in a registry-wide guard that never runs the module.
    assert QueryModule.phases == (
        query_module.PHASE_LOADING,
        query_module.PHASE_PREPARING,
        query_module.PHASE_SEARCHING,
        query_module.PHASE_WRITING,
    )
    assert_messages_name_their_phases(messages, QueryModule.phases)
    assert "starting" not in messages
    assert len(set(messages)) >= len(QueryModule.phases)


# --------------------------------------------------------------------------- #
# chunking: split the curve, not the extraction
# --------------------------------------------------------------------------- #
def test_a_chunked_search_finds_what_an_unchunked_one_finds(tmp_path, monkeypatch):
    """§20's chunk equivalence, asserted on the **match set** rather than the curve.

    The curve is only equal to float tolerance (BLAS blocks a sliced matmul differently), so
    asserting costs fails on numerical noise while missing the failure it exists to catch:
    splitting silently changing answers on exactly the long recordings it exists to serve.

    The **module's own** merge is what runs here. It used to be the test's: the module left one
    shard per chunk and the test merged them itself, so the merge inside `_search_unit` — the
    one every real run takes — was never exercised by this. Calibration is why it moved: a
    reprojection would otherwise have to rebuild the prepared state to merge, reloading every
    exemplar recording's embeddings on each cutoff change for a step whose answer never varies.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"], n_rows=600, plant_at=400)
    method = resolve_method("soft_knn")

    whole_dir, _j, _e = _run(
        proj, _config(["recording_001"], plant_at=400), run_id="an_01"
    )
    whole = method.instances_at(
        method.load(
            query_runs.shard_path(
                whole_dir, "q01", "recording_001", 0, method.shard_suffix
            )
        ),
        0.5,
        {},
    )

    # How many chunks the run actually merged. Counting shards on disk would now report "no
    # chunking" for a run that chunked four ways, because the module deletes them once the
    # merged shard lands — a precondition read off a by-product rather than the thing itself.
    merged_counts: list[int] = []
    real_merge = SoftKnnQueryMethod.merge

    def spy(self, chunk_results, state, params=None):
        merged_counts.append(len(chunk_results))
        return real_merge(self, chunk_results, state, params)

    monkeypatch.setattr(SoftKnnQueryMethod, "merge", spy)

    split = QueryModule()
    split.chunk_target_rows = 250  # force several chunks over the same 600 rows
    chunk_dir, _j2, _e2 = _run(
        proj, _config(["recording_001"], plant_at=400), run_id="an_02", module=split
    )
    assert merged_counts and merged_counts[0] > 1, "the fixture no longer chunks"

    shards = sorted(query_runs.search_dir(chunk_dir).rglob(f"*{method.shard_suffix}"))
    assert len(shards) == 1, "a searched unit is one shard once its chunks are merged"
    chunked = method.instances_at(method.load(shards[0]), 0.5, {})

    assert [(c.start_row, c.end_row) for c in chunked] == [
        (c.start_row, c.end_row) for c in whole
    ]


def test_an_interrupted_merge_leaves_its_chunks_to_resume_from(tmp_path, monkeypatch):
    """Chunks are dropped only *after* the merged shard lands.

    Deleting them first and then failing would throw away a cohort-scale search that had
    already finished, leaving nothing to resume from and no sign that anything was lost — the
    run simply looks like it never searched that unit.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"], n_rows=600, plant_at=400)

    def boom(self, chunk_results, state, params=None):
        raise RuntimeError("merge interrupted")

    monkeypatch.setattr(SoftKnnQueryMethod, "merge", boom)
    split = QueryModule()
    split.chunk_target_rows = 250
    run_dir, _j, _e = _run(
        proj, _config(["recording_001"], plant_at=400), run_id="an_01", module=split
    )

    # The behavior is recorded as failed, and every chunk it had already searched survives.
    record = analysis_store.read_analysis_yaml(run_dir)
    assert "q01" in record["behaviors_failed"]
    chunks = sorted(query_runs.search_dir(run_dir).rglob("*__chunk_*.npz"))
    assert len(chunks) > 1

    # ...and resuming re-runs the merge without re-searching a single chunk.
    monkeypatch.undo()
    searches = _count_searches(monkeypatch)
    resumed = QueryModule()
    resumed.chunk_target_rows = 250
    resumed.runner(_j, run_dir, lambda _e: None, threading.Event())

    assert searches == [], "a resumed unit re-searched chunks it had already stored"
    assert not list(query_runs.search_dir(run_dir).rglob("*__chunk_*.npz"))
    assert query_runs.shard_path(run_dir, "q01", "recording_001", 0, ".npz").is_file()


def _count_searches(monkeypatch) -> list:
    """Record every `search` call a run makes, so resume can be asserted on work avoided."""
    calls: list[int] = []
    real_search = SoftKnnQueryMethod.search

    def counting_search(self, *args, **kw):
        calls.append(1)
        return real_search(self, *args, **kw)

    monkeypatch.setattr(SoftKnnQueryMethod, "search", counting_search)
    return calls


def test_a_completed_unit_is_not_searched_again(tmp_path, monkeypatch):
    """Resume keys off the shard: re-running a finished search must touch no method."""
    proj, _ = _experiment(tmp_path, ["recording_001"], n_rows=600, plant_at=400)
    run_dir, job, _e = _run(proj, _config(["recording_001"], plant_at=400))

    searches = _count_searches(monkeypatch)
    QueryModule().runner(job, run_dir, lambda _e: None, threading.Event())

    assert searches == []


def test_a_completed_chunked_unit_is_not_searched_again(tmp_path, monkeypatch):
    """The case the merged shard's resume check exists for.

    Once a chunked unit merges, its chunk shards are gone — so the per-chunk "already on disk"
    check finds nothing and a resume would re-search every chunk of a unit it had already
    finished. On a cohort-scale recording that is hours of recomputation producing a byte-
    identical answer, and nothing about the run looks wrong while it happens.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"], n_rows=600, plant_at=400)
    split = QueryModule()
    split.chunk_target_rows = 250
    run_dir, job, _e = _run(
        proj, _config(["recording_001"], plant_at=400), module=split
    )
    assert not list(query_runs.search_dir(run_dir).rglob("*__chunk_*.npz"))

    searches = _count_searches(monkeypatch)
    resumed = QueryModule()
    resumed.chunk_target_rows = 250
    resumed.runner(job, run_dir, lambda _e: None, threading.Event())

    assert searches == [], "a merged unit was searched again on resume"


def test_chunks_overlap_by_the_methods_declared_window():
    # A match ending near a seam needs its full window of context, so consecutive chunks
    # overlap by the longest match the method can produce; abutting chunks would cut one.
    chunks = _plan_chunks(1000, overlap=50, target=400)
    assert chunks[0] == (0, 400)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later[0] <= earlier[1] - 50
    assert chunks[-1][1] == 1000  # every row covered


def test_an_unsplittable_method_searches_the_unit_whole():
    # `None` means the method's recurrence carries state across the whole row axis, so a naive
    # split would cut matches at the seam.
    assert _plan_chunks(10_000_000, overlap=None, target=1000) == [(0, 10_000_000)]


def test_the_cli_routes_non_examples_to_the_behavior_that_names_them(tmp_path):
    """`--non-example` is new wiring, and the failure it could have is silent: a clip that
    parses fine and lands nowhere. Driven through the command rather than `validate`, because
    the grouping — a `name=` prefix picking the behavior — is the CLI's own step.

    Asserted on the config the command hands to `run_analysis`, which is the routing itself.
    It used to lean on `dtw` refusing non-examples — the refusal standing in as evidence they
    arrived — which tied this test to which method happens to refuse them *and* to that method
    being offered. Capturing the config says the thing directly and survives both.
    """
    from click.testing import CliRunner

    from detail.api.experiment import Experiment as _Experiment
    from detail.cli import main

    proj, _ = _experiment(tmp_path, ["recording_001"])
    captured: dict = {}

    class _Stop(Exception):
        """Halt once the config exists — running the search is not the point."""

    def capture(self, module_id, config):
        captured["config"] = config
        raise _Stop

    import unittest.mock

    with unittest.mock.patch.object(_Experiment, "run_analysis", capture):
        CliRunner().invoke(
            main,
            [
                "analysis", "query",
                "--experiment", str(proj.dir),
                "--search", "recording_001",
                "--embed-run", "emb_01",
                "--example", "rear=recording_001:0:100f-119f",
                "--non-example", "rear=recording_001:0:300f-319f",
            ],
        )  # fmt: skip

    (behavior,) = captured["config"]["behaviors"]
    assert behavior["name"] == "rear"
    assert len(behavior["examples"]) == 1
    # The clip that could silently land nowhere.
    assert len(behavior["negatives"]) == 1
    assert behavior["negatives"][0]["recording_id"] == "recording_001"
    assert behavior["negatives"] != behavior["examples"]


# --------------------------------------------------------------------------- #
# min match length: seconds in the config, rows only where the cadence is known
# --------------------------------------------------------------------------- #
def test_min_match_is_carried_to_the_job_in_ns(tmp_path):
    proj, _ = _experiment(tmp_path, ["recording_001"])
    normalized = QueryModule().validate(
        proj.dir, _config(["recording_001"], min_match_s=0.025)
    )
    assert "min_match_s" not in normalized
    assert normalized["min_match_ns"] == 25_000_000
    job = QueryModule().build_job("an_01", normalized)
    assert job.min_match_ns == 25_000_000
    assert query_module.QueryJob.from_dict(job.to_dict()).min_match_ns == 25_000_000


def test_a_job_from_before_the_minimum_loads_with_none():
    proj_job = (
        QueryModule()
        .build_job(
            "an_01",
            {
                "source_recording_ids": ["recording_001"],
                "embed_run_id": "emb_01",
                "track_id": 0,
                "method": "soft_knn",
                "behaviors": [],
            },
        )
        .to_dict()
    )
    proj_job.pop("min_match_ns")
    assert query_module.QueryJob.from_dict(proj_job).min_match_ns == 0


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_the_same_minimum_is_the_same_duration_at_every_cadence(fps):
    """A minimum in rows would be a different duration on every recording; converted from
    one timeline, one second is ``fps`` rows wherever it is applied."""
    period = round(1e9 / fps)
    t_ns = np.arange(500, dtype=np.int64) * period
    assert query_module.min_match_rows(1_000_000_000, t_ns) == fps
    assert query_module.min_match_rows(0, t_ns) == 0


def test_no_searched_match_is_shorter_than_the_minimum(tmp_path):
    """Through the module, so the seconds -> rows step and the hand-off to the method are
    both on the path: a minimum that never reached ``search`` would leave 20-row matches.
    """
    proj, _ = _experiment(tmp_path, ["recording_001"])
    config = _config(
        ["recording_001"],
        min_match_s=25 * PERIOD / 1e9,
        method_params={"duration_max_mult": 2.0},
    )
    run_dir, _job, _events = _run(proj, config)
    method = resolve_method("soft_knn")
    shard = query_runs.shard_path(run_dir, "q01", "recording_001", 0, ".npz")
    found = method.instances_at(method.load(shard), 0.0, config["method_params"])
    assert found
    assert min(c.n_rows for c in found) >= 25
    record = analysis_store.read_analysis_yaml(run_dir)
    assert record["params"]["min_match_ns"] == 25 * PERIOD


def test_the_cli_min_match_flag_reaches_the_config(tmp_path):
    from click.testing import CliRunner

    from detail.api.experiment import Experiment as _Experiment
    from detail.cli import main

    proj, _ = _experiment(tmp_path, ["recording_001"])
    captured: dict = {}

    class _Stop(Exception):
        """Halt once the config exists."""

    def capture(self, module_id, config):
        captured["config"] = config
        raise _Stop

    import unittest.mock

    with unittest.mock.patch.object(_Experiment, "run_analysis", capture):
        CliRunner().invoke(
            main,
            [
                "analysis", "query",
                "--experiment", str(proj.dir),
                "--search", "recording_001",
                "--embed-run", "emb_01",
                "--example", "rear=recording_001:0:100f-119f",
                "--min-match", "0.5",
            ],
        )  # fmt: skip

    assert captured["config"]["min_match_ns"] == 500_000_000
