"""Behavior-query eval end-to-end (§16) — evaluate_query + the CLI over a synthesized
embedding run and a multi-behavior ground-truth stream."""

import numpy as np
import pytest
from click.testing import CliRunner

from detail.api.recording import Recording
from detail.cli import main
from detail.pipeline.query_eval import QueryEvalModule, evaluate_query, null_at
from detail.storage import recording as recording_store
from tests.pipeline.phase_messages import assert_messages_name_their_phases

DIM = 8
PERIOD = 100_000_000  # 0.1 s / frame (10 Hz)
N = 60
_A = np.eye(DIM)[0]  # groom
_B = np.eye(DIM)[2]  # background
_C = np.eye(DIM)[1]  # bite
GROOM = [(5, 9), (20, 24), (35, 39), (50, 54)]  # state, 4 intervals
BITE = [15, 30, 45]  # point, 3 instants


def _trajectory():
    emb = np.tile(_B, (N, 1)).astype("float32")
    for start, end in GROOM:
        emb[start : end + 1] = _A
    for point in BITE:
        emb[point] = _C
    frame_idx = np.arange(N, dtype=np.int32)
    t_ns = np.arange(N, dtype=np.int64) * PERIOD
    return emb, frame_idx, t_ns


def _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path, labels=None):
    emb, frame_idx, t_ns = _trajectory()
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, emb, frame_idx, t_ns)
    lines = labels or (
        ["start_frame,end_frame,label"]
        + [f"{s},{e},groom" for s, e in GROOM]
        + [f"{p},{p},bite" for p in BITE]
    )
    csv = tmp_path / "labels.csv"
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Recording(experiment_dir, "recording_001").add_stream(
        "behavior", raw_path=csv, importer="behavior_csv"
    )


def _by_behavior(experiment_dir, config):
    return {r.behavior: r for r in evaluate_query(experiment_dir, config)}


def test_state_and_point_retrieved(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    results = _by_behavior(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom", "bite"], "band_s": 0.2},
    )
    assert results["groom"].behavior_type == "state"
    assert results["bite"].behavior_type == "point"
    # one instance held out as the exemplar; the rest are found
    assert results["groom"].metrics.n_gt == 3
    assert results["bite"].metrics.n_gt == 2
    assert results["groom"].metrics.recall == 1.0
    assert results["groom"].metrics.ap > 0.9
    assert results["bite"].metrics.ap > 0.9


def test_perframe_ap_scores_a_separable_state_behavior(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """The per-frame classification path: score every frame, rank, integrate a per-frame
    AP. On a separable fixture (groom frames are one pose, background another) soft_knn
    ranks the remaining groom frames at the top → AP near 1, beside the one analytic-floor
    row the per-frame path emits (the windowed path's multi-level overlap table is its
    own)."""
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    result = _by_behavior(
        experiment_dir,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 0,  # leave-one-out over the 4 groom intervals
            "n_negatives": 5,  # keep some background in the test set
        },
    )["groom"]
    assert result.metrics.ap > 0.95
    assert result.metrics.n_gt > 0  # per-frame positives (not spans)
    assert len(result.overlap) == 1  # the floor row, not a span-overlap sweep
    assert result.similarity is not None  # the scored curve is persisted for the figure


def test_perframe_point_window_widens_the_positive_frames(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """Point annotations are a single frame, so the per-frame label uses a ±point_window
    tolerance (mirroring the span path's point mode). A zero window marks only the exact
    annotated frames positive (the too-strict case); a wider window marks a band."""
    cfg = {
        "embed_run_id": "emb_01",
        "behaviors": ["bite"],
        "method": "soft_knn",
        "folds": 0,
        "n_negatives": 5,
    }
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    tight = _by_behavior(experiment_dir, {**cfg, "point_window_s": 0.0})["bite"]
    wide = _by_behavior(experiment_dir, {**cfg, "point_window_s": 0.5})["bite"]
    assert (
        tight.metrics.n_gt < wide.metrics.n_gt
    )  # the window expands the positive band
    # With only the exact bite pose labeled positive, it ranks at the very top → AP ~ 1.
    assert tight.metrics.ap > 0.95


def test_perframe_writes_gt_and_prediction_detail_for_figures(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """The per-frame path emits fold-0 GT spans + operating-point matches so the matches /
    similarity figures render (GT overlay + TP/FP spans) instead of blank."""
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    result = _by_behavior(
        experiment_dir,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 0,
            "n_negatives": 5,
        },
    )["groom"]
    assert {g.role for g in result.ground_truth} == {"scored", "exemplar"}
    assert all(g.fold == 0 for g in result.ground_truth)
    assert result.predictions  # operating-point spans, not blank
    assert {p.outcome for p in result.predictions} <= {"tp", "fp"}
    assert all(p.fold == 0 for p in result.predictions)


def test_fold0_similarity_curve_populated(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    results = _by_behavior(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom"], "band_s": 0.2},
    )
    sim = results["groom"].similarity
    assert sim is not None
    recording_id, t_ns, score = sim
    assert recording_id == "recording_001"
    assert len(t_ns) == len(score) == N  # one score per frame of the display recording
    assert score.dtype == np.float32


def test_behavior_not_scored_excluded(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    results = _by_behavior(
        experiment_dir, {"embed_run_id": "emb_01", "behaviors": ["climb"]}
    )
    assert results["climb"].metrics is None
    assert "no eligible" in results["climb"].skipped


def test_too_few_ground_truth_skipped(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    # groom has 4 intervals; holding out 5 leaves nothing to score
    results = _by_behavior(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom"], "n_exemplars": 5},
    )
    assert results["groom"].metrics is None
    assert "n_exemplars" in results["groom"].skipped


def test_empty_behaviors_raises(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    with pytest.raises(ValueError, match="at least one behavior"):
        evaluate_query(experiment_dir, {"embed_run_id": "emb_01", "behaviors": []})


def test_seed_is_deterministic(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    cfg = {"embed_run_id": "emb_01", "behaviors": ["groom"], "band_s": 0.2, "seed": 3}
    a = _by_behavior(experiment_dir, cfg)["groom"].metrics
    b = _by_behavior(experiment_dir, cfg)["groom"].metrics
    assert (a.ap, a.tp, a.n_gt) == (b.ap, b.tp, b.n_gt)


def test_evaluate_query_persists_per_clip_detail(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    # The per-clip detail the recorded run writes: predictions tagged TP/FP (a TP names
    # its GT span), and ground truth marked matched/FN plus the held-out exemplar.
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    groom = _by_behavior(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom"], "band_s": 0.2},
    )["groom"]
    assert "tp" in {r.outcome for r in groom.predictions}
    tps = [r for r in groom.predictions if r.outcome == "tp"]
    assert tps and all(r.matched_gt_start_t_ns is not None for r in tps)
    assert {r.role for r in groom.ground_truth} == {"scored", "exemplar"}
    assert sum(r.role == "exemplar" for r in groom.ground_truth) == 1
    # one scored GT row per scored interval; the matched/FN split matches the metrics
    scored = [r for r in groom.ground_truth if r.role == "scored"]
    assert len(scored) == groom.metrics.n_gt
    assert sum(r.outcome == "matched" for r in scored) == groom.metrics.tp


def test_leave_one_out_runs_a_fold_per_instance(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    # folds=0 rotates every groom interval (4) as the single query; each fold scores
    # against the other 3, and the aggregate is the mean over the 4 folds.
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    groom = _by_behavior(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom"], "band_s": 0.2, "folds": 0},
    )["groom"]
    assert groom.metrics.n_folds == 4
    assert len(groom.fold_metrics) == 4
    assert all(fm.n_gt == 3 for fm in groom.fold_metrics)  # 4 - 1 exemplar
    assert {p.fold for p in groom.predictions} == {0, 1, 2, 3}  # detail spans all folds


def test_k_folds_subsample_and_cap(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    two = _by_behavior(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom"], "band_s": 0.2, "folds": 2},
    )["groom"]
    assert two.metrics.n_folds == 2 and {p.fold for p in two.predictions} == {0, 1}
    # more folds than fit (4 instances) → capped
    capped = _by_behavior(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom"], "band_s": 0.2, "folds": 99},
    )["groom"]
    assert capped.metrics.n_folds == 4


def test_leave_one_out_rejected_at_config_and_validate():
    # The LOO ⇒ n_exemplars=1 rule lives in the config model (single source), so both the
    # model and the recorded module's validate() reject it up front — not buried in a
    # failed subprocess after the run record is written.
    from detail.configs.query_eval import QueryEvalConfig
    from detail.pipeline.query_eval import QueryEvalModule

    with pytest.raises(ValueError, match="leave-one-out"):
        QueryEvalConfig(embed_run_id="e", folds=0, n_exemplars=2)
    QueryEvalConfig(embed_run_id="e", folds=0, n_exemplars=1)  # valid
    with pytest.raises(ValueError, match="leave-one-out"):
        QueryEvalModule().validate(
            "/nope",
            {"embed_run_id": "e", "behaviors": ["x"], "folds": 0, "n_exemplars": 2},
        )


def test_make_folds_excludes_by_index_keeping_duplicate_gt(monkeypatch):
    # Two identical GT intervals (a duplicated annotation) must not collapse: holding one
    # out must leave the other scorable, so n_gt stays constant across folds. Regression
    # guard against value-based exclusion.
    import detail.pipeline.query_eval as qe_mod

    monkeypatch.setattr(qe_mod, "_resolvable", lambda *a: True)
    dup = (("s", 0), 100, 200)
    pooled = [dup, dup, (("s", 0), 300, 400)]
    cfg = type(
        "Cfg", (), {"seed": 0, "n_exemplars": 1, "n_holdout": None, "folds": 0}
    )()
    folds, k_max = qe_mod._make_folds(pooled, {}, cfg, None)
    assert k_max == 3
    assert all(
        len(ex) == 1 and len(rem) == 2 for ex, rem, _hold in folds
    )  # dup preserved


def test_make_folds_holds_out_more_than_it_queries(monkeypatch):
    """`n_holdout` fixes the scored set so an n_exemplars sweep compares like with like.

    Without it the positive set shrinks as the query grows and a curve against n measures its own
    denominator: every extra exemplar takes another instance out of the ground truth.
    """
    import detail.pipeline.query_eval as qe_mod

    monkeypatch.setattr(qe_mod, "_resolvable", lambda *a: True)
    pooled = [(("s", 0), i * 100, i * 100 + 50) for i in range(10)]

    scored = {}
    for n_ex in (1, 2, 5):
        cfg = type(
            "Cfg", (), {"seed": 0, "n_exemplars": n_ex, "n_holdout": 5, "folds": 1}
        )()
        folds, _k_max = qe_mod._make_folds(pooled, {}, cfg, None)
        exemplars, remaining, holdout = folds[0]
        assert len(exemplars) == n_ex
        assert len(holdout) == 5
        assert exemplars == holdout[:n_ex]  # the query is a prefix of the holdout
        scored[n_ex] = len(remaining)

    assert set(scored.values()) == {5}, (
        f"the scored set moved with n_exemplars: {scored} — that is the confound n_holdout "
        f"exists to remove"
    )


def test_without_n_holdout_the_scored_set_still_shrinks(monkeypatch):
    # Guard the guard: the old behaviour is what `n_holdout=None` must keep doing, and it is
    # what the test above would otherwise pass vacuously against.
    import detail.pipeline.query_eval as qe_mod

    monkeypatch.setattr(qe_mod, "_resolvable", lambda *a: True)
    pooled = [(("s", 0), i * 100, i * 100 + 50) for i in range(10)]
    sizes = []
    for n_ex in (1, 5):
        cfg = type(
            "Cfg", (), {"seed": 0, "n_exemplars": n_ex, "n_holdout": None, "folds": 1}
        )()
        folds, _k_max = qe_mod._make_folds(pooled, {}, cfg, None)
        _ex, remaining, _hold = folds[0]
        sizes.append(len(remaining))
    assert sizes == [9, 5]


def test_make_folds_loads_each_subject_once():
    # Regression (perf): resolvability must load each *distinct* subject's trajectory once,
    # not once per GT instance — the folds=1 reload storm where an 84-instance behavior
    # reloaded the full (~400MB) trajectory 84x on the uncached lazy store.
    import detail.pipeline.query_eval as qe_mod

    t_ns = np.arange(50, dtype=np.int64) * PERIOD

    class CountingStore:
        def __init__(self):
            self.loads: dict = {}

        def __contains__(self, subject):
            return True

        def __getitem__(self, subject):
            self.loads[subject] = self.loads.get(subject, 0) + 1
            emb = np.zeros((len(t_ns), DIM), dtype="float32")
            return emb, np.arange(len(t_ns), dtype=np.int32), t_ns

    store = CountingStore()
    subject = ("s", 0)
    # 10 GT instances of ONE subject, all resolvable within its single continuous run
    pooled = [(subject, i * PERIOD, (i + 1) * PERIOD) for i in range(10)]
    cfg = type(
        "Cfg", (), {"seed": 0, "n_exemplars": 1, "n_holdout": None, "folds": 0}
    )()
    _folds, k_max = qe_mod._make_folds(pooled, store, cfg, None)
    assert k_max == 10  # all resolvable
    assert store.loads == {subject: 1}  # loaded ONCE, not 10x


def test_evaluate_query_is_cancellable(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    # stop_check True from the outset → the behavior loop breaks immediately, so nothing is
    # scored (responsive cancel, not "run every behavior then stop").
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    results = evaluate_query(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom", "bite"], "band_s": 0.2},
        stop_check=lambda: True,
    )
    assert results == []


def _job(**over):
    from detail.pipeline.query_eval import QueryEvalModule

    config = {"embed_run_id": "emb_01", "behaviors": ["groom"], "band_s": 0.5}
    config.update(over)
    return QueryEvalModule().build_job("an_01", config)


def test_query_eval_job_roundtrips_folds():
    from detail.pipeline.query_eval import QueryEvalJob

    job = _job(folds=0)
    assert QueryEvalJob.from_dict(job.to_dict()).folds == 0  # survives serialization
    d = job.to_dict()
    del d["folds"]  # a pre-folds job.json (durable/reattach) defaults to a single split
    assert QueryEvalJob.from_dict(d).folds == 1


def test_the_recorded_job_carries_the_matcher():
    """A recorded run rebuilds its config from the job. A matcher field missing there
    does not fail — it silently reverts to `dtw`, so the run scores with a matcher the
    user never chose. Drop `method` from QueryEvalJob and this fails."""
    from detail.pipeline.query_eval import QueryEvalJob, _job_config

    job = _job(
        method="soft_knn",
        k=15,
        tau=0.05,
        n_negatives=300,
        keyframe_select="rare",
        n_keyframes=3,
    )
    assert job.method == "soft_knn"
    back = QueryEvalJob.from_dict(job.to_dict())  # survives job.json
    assert back.method == "soft_knn"
    assert back.k == 15 and back.n_negatives == 300
    config = _job_config(back)  # and reaches evaluate_query
    assert config["method"] == "soft_knn"
    assert config["k"] == 15 and config["tau"] == 0.05 and config["n_negatives"] == 300
    assert config["keyframe_select"] == "rare" and config["n_keyframes"] == 3


def test_every_config_field_reaches_the_job():
    """The structural guard behind the bug above: a new config parameter must not be
    left behind when the job is built."""
    from detail.configs.query_eval import QueryEvalConfig
    from detail.pipeline.query_eval import _job_config

    config = _job_config(_job())
    missing = set(QueryEvalConfig.model_fields) - set(config)
    assert not missing, f"config fields dropped by the job: {sorted(missing)}"


def test_job_reads_a_run_recorded_before_the_rename():
    """`max_gap_s` became `trajectory_split_gap_s`; an existing job.json keeps its value."""
    from detail.pipeline.query_eval import QueryEvalJob

    d = _job(trajectory_split_gap_s=2.5).to_dict()
    d["max_gap_s"] = d.pop("trajectory_split_gap_s")
    assert QueryEvalJob.from_dict(d).trajectory_split_gap_s == 2.5


def test_job_reads_a_run_recorded_before_the_matcher_existed():
    from detail.pipeline.query_eval import QueryEvalJob

    d = _job().to_dict()
    for key in ("method", "keyframe_select", "n_keyframes", "k", "tau", "n_negatives"):
        del d[key]
    back = QueryEvalJob.from_dict(d)
    assert back.method == "dtw"  # what such a run actually did
    assert back.k == 20 and back.tau == 0.07 and back.n_negatives == 500


def _run_module(experiment_dir, behaviors, progress=None, **over):
    import threading

    from detail.pipeline.query_eval import QueryEvalModule
    from detail.storage import analysis_runs as ar

    module = QueryEvalModule()
    config = module.validate(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": behaviors, "band_s": 0.2, **over},
    )
    job = module.build_job("an_01", config)
    run_dir = ar.analysis_dir(experiment_dir, "an_01")
    ar.write_analysis_yaml(
        run_dir,
        {
            "id": "an_01",
            "module_id": "query_eval",
            "recording_ids": config["source_recording_ids"],
            "status": "running",
        },
    )
    module.runner(job, run_dir, progress or (lambda e: None), threading.Event())
    return run_dir, config


def test_query_eval_module_writes_run_outputs(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    from detail.api.run_preview import QueryEvalPreview
    from detail.storage import analysis_runs as ar
    from detail.storage import query_eval_csv

    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    run_dir, config = _run_module(experiment_dir, ["groom", "bite"])
    assert config["source_recording_ids"] == ["recording_001"]  # validate resolved them

    for name in ("metrics.csv", "folds.csv", "predictions.csv", "ground_truth.csv"):
        assert (run_dir / name).is_file()
    assert (run_dir / "pr" / "groom" / "fold_00.csv").is_file()  # one PR curve per fold
    record = ar.read_analysis_yaml(run_dir)
    assert record["status"] == "complete"
    assert record["embed_run_id"] == "emb_01"
    kinds = {a["kind"] for a in record["artifacts"]}
    assert {
        "query_eval_metrics",
        "query_eval_folds",
        "query_eval_predictions",
        "query_eval_ground_truth",
    } <= kinds

    # read back through the API seam the GUI uses (Tenet 2)
    pv = QueryEvalPreview(experiment_dir)
    metrics = {m.behavior: m for m in pv.read_metrics("an_01")}
    assert set(metrics) == {"groom", "bite"}
    assert metrics["groom"].n_folds == 1  # default single split
    preds = pv.read_predictions("an_01", behavior="groom")
    assert preds and all(p.behavior == "groom" and p.fold == 0 for p in preds)
    assert any(g.role == "exemplar" for g in pv.read_ground_truth("an_01", "groom"))
    assert pv.read_pr("an_01", "groom").get(0)  # fold-0 PR points
    assert [f.fold for f in pv.read_folds("an_01", "groom")] == [0]
    assert query_eval_csv  # storage import used above


def test_query_eval_run_analysis_end_to_end(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    # The real executor path (subprocess worker) — exercises QueryEvalJob
    # to_dict/from_dict + progress-event registration that the direct-runner test
    # bypasses, and the run reconciles to a terminal status.
    from detail.api.experiment import Experiment

    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    proj = Experiment.open(experiment_dir)
    # folds=0 (LOO) so the cross-validation path — and QueryEvalJob's folds field —
    # survives serialization through the subprocess worker, not just the direct runner.
    run = proj.run_analysis(
        "query_eval",
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom", "bite"],
            "band_s": 0.2,
            "folds": 0,
        },
    )
    run.wait()
    rec = proj.analysis_record(run.id)
    assert run.status == "complete", rec.get("error")
    assert rec["status"] == "complete" and rec["embed_run_id"] == "emb_01"
    assert rec["recording_ids"] == ["recording_001"]  # validate resolved the search set
    base = proj.dir / "analysis" / run.id
    for name in ("metrics.csv", "folds.csv", "predictions.csv", "ground_truth.csv"):
        assert (base / name).is_file()
    from detail.api.run_preview import QueryEvalPreview

    metrics = {m.behavior: m for m in QueryEvalPreview(proj.dir).read_metrics(run.id)}
    assert metrics["groom"].n_folds == 4  # 4 groom intervals, leave-one-out


def test_query_eval_module_progress_ticks_mid_run(
    experiment_dir, recording_dir, make_embedding_run, tmp_path, monkeypatch
):
    # A recorded eval must stream progress that moves mid-run (not a lone terminal
    # event) and reaches 1.0, with an ETA once it is underway (long-op rule §6).
    import detail.core.progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)  # emit every tick
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    events = []
    _run_module(experiment_dir, ["groom", "bite"], progress=events.append)

    fracs = [
        (
            e.recordings_done
            + (e.frames_done / e.frames_total if e.frames_total else 0.0)
        )
        / e.recordings_total
        for e in events
    ]
    assert any(0.0 < f < 1.0 for f in fracs)  # movement before the end
    assert fracs[-1] == 1.0
    assert any(0.0 < f < 1.0 and e.eta_s is not None for e, f in zip(events, fracs))
    # The message is informative from the first tick (not the "frozen at starting" bug)
    # and advances through distinct phases (loading → per-fold querying → aggregating).
    messages = [e.message for e in events]
    assert messages[0] != "starting" and "loading" in messages[0]
    assert len(set(messages)) >= 3 and any("fold" in (m or "") for m in messages)
    # §16, issue #393: each message is drawn as the phase its runner meant — the querying
    # messages carry query_tracks' own "loading <subject>" in their note, which once put the
    # run page back on "loading" once per subject searched.
    assert_messages_name_their_phases(messages, QueryEvalModule.phases)


def test_evaluate_query_reports_progress(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    # A silent eval is indistinguishable from a hang (long-op hard rule): the run must
    # stream distinct phases (loading -> querying -> scoring) with a monotonic fraction
    # that moves mid-run and reaches 1.0 — not a single terminal event.
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    events = []
    evaluate_query(
        experiment_dir,
        {"embed_run_id": "emb_01", "behaviors": ["groom", "bite"], "band_s": 0.2},
        progress=lambda stage, frac, detail: events.append((stage, frac, detail)),
    )
    stages = [s for s, _f, _d in events]
    assert {"loading", "querying", "scoring"} <= set(stages)
    assert stages.index("loading") < stages.index("querying") < stages.index("scoring")
    assert stages.index("querying") < len(stages) - 1  # heavy phase reported mid-run
    fracs = [f for _s, f, _d in events]
    assert fracs == sorted(fracs)  # monotonic non-decreasing
    assert any(0.0 < f < 1.0 for f in fracs)  # movement before the end, not frozen
    assert fracs[-1] == 1.0
    assert any("groom" in d for _s, _f, d in events)  # a live, per-behavior message


def test_cli_query_eval_appends_rows(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    out = tmp_path / "metrics.csv"
    result = CliRunner().invoke(
        main,
        [
            "analysis",
            "query-eval",
            "--experiment",
            str(experiment_dir),
            "--embed-run",
            "emb_01",
            "--behavior",
            "groom",
            "--behavior",
            "bite",
            "--band",
            "0.2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "groom [state]: AP=" in result.output
    assert "bite [point]: AP=" in result.output
    from detail.storage import query_eval_csv

    rows = query_eval_csv.read_metrics_csv(out)
    assert {r.behavior for r in rows} == {"groom", "bite"}
    # a second call appends, accumulating the sweep table
    CliRunner().invoke(
        main,
        [
            "analysis",
            "query-eval",
            "--experiment",
            str(experiment_dir),
            "--embed-run",
            "emb_01",
            "--behavior",
            "groom",
            "--fusion",
            "mean",
            "--band",
            "0.2",
            "--out",
            str(out),
        ],
    )
    assert len(query_eval_csv.read_metrics_csv(out)) == 3


def test_cli_query_eval_writes_per_item_detail(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    # Surface parity: the per-clip detail the GUI explorer shows is reachable from the
    # scriptable path via --predictions-out / --ground-truth-out.
    from detail.storage import query_eval_csv

    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    preds = tmp_path / "predictions.csv"
    gt = tmp_path / "ground_truth.csv"
    result = CliRunner().invoke(
        main,
        [
            "analysis", "query-eval",
            "--experiment", str(experiment_dir),
            "--embed-run", "emb_01",
            "--behavior", "groom",
            "--band", "0.2",
            "--out", str(tmp_path / "metrics.csv"),
            "--predictions-out", str(preds),
            "--ground-truth-out", str(gt),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert query_eval_csv.read_predictions_csv(preds)
    assert any(r.role == "exemplar" for r in query_eval_csv.read_ground_truth_csv(gt))


def test_cli_query_eval_folds_aggregate(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    # --folds 0 (LOO) writes a mean±std metrics row + a per-fold folds.csv.
    from detail.storage import query_eval_csv

    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    out = tmp_path / "metrics.csv"
    folds_csv = tmp_path / "folds.csv"
    result = CliRunner().invoke(
        main,
        [
            "analysis", "query-eval",
            "--experiment", str(experiment_dir),
            "--embed-run", "emb_01",
            "--behavior", "groom",
            "--band", "0.2",
            "--folds", "0",
            "--out", str(out),
            "--folds-out", str(folds_csv),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert query_eval_csv.read_metrics_csv(out)[0].n_folds == 4
    assert len(query_eval_csv.read_folds_csv(folds_csv)) == 4


# --- frame→span end-to-end + the honest metric ------------------------------------- #


def _varied_trajectory():
    """Ground-truth stretches whose lengths vary several-fold — the case a fixed-length
    window cannot fit and the merge can."""
    emb = np.tile(_B, (200, 1)).astype("float32")
    spans = [(10, 14), (40, 62), (90, 96), (120, 158), (175, 180)]
    for start, end in spans:
        emb[start : end + 1] = _A
    frame_idx = np.arange(200, dtype=np.int32)
    t_ns = np.arange(200, dtype=np.int64) * PERIOD
    return emb, frame_idx, t_ns, spans


def _setup_varied(experiment_dir, recording_dir, make_embedding_run, tmp_path):
    emb, frame_idx, t_ns, spans = _varied_trajectory()
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, emb, frame_idx, t_ns)
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "\n".join(
            ["start_frame,end_frame,label"] + [f"{s},{e},climb" for s, e in spans]
        )
        + "\n",
        encoding="utf-8",
    )
    Recording(experiment_dir, "recording_001").add_stream(
        "behavior", raw_path=csv, importer="behavior_csv"
    )
    return spans


def _eval_varied(experiment_dir, method, **over):
    config = {
        "embed_run_id": "emb_01",
        "behaviors": ["climb"],
        "method": method,
        "band_s": 0.3,
        "folds": 3,
        "iou_threshold": 0.5,
    }
    config.update(over)
    return _by_behavior(experiment_dir, config)["climb"]


def test_overlap_metrics_reported_with_their_nulls(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """The windowed span path reports recall/precision/AP at each overlap level, each
    beside its null — the lift identities and level ordering hold for any windowed matcher.
    """
    _setup_varied(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    result = _eval_varied(experiment_dir, "dtw")
    levels = [round(m.iou, 3) for m in result.overlap]
    assert result.overlap  # windowed methods populate the overlap table
    assert levels == sorted(levels) and 0.1 in levels and 0.5 in levels
    for m in result.overlap:
        assert m.recall_lift == pytest.approx(m.recall - m.recall_null)
        assert m.ap_lift == pytest.approx(m.ap - m.ap_null)


def test_module_writes_overlap_csv(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    from detail.storage import query_eval_csv

    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    run_dir, _config = _run_module(experiment_dir, ["groom"])
    rows = query_eval_csv.read_overlap_csv(run_dir / "overlap.csv")
    assert rows and {r.behavior for r in rows} == {"groom"}
    assert {round(r.iou, 3) for r in rows} >= {0.1, 0.25, 0.5}


def test_metrics_csv_names_the_matcher_and_its_null(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """A sweep row that doesn't say which matcher produced it can't be compared."""
    from detail.storage import query_eval_csv

    _setup_varied(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    runner = CliRunner()
    out = tmp_path / "sweep.csv"
    result = runner.invoke(
        main,
        ["analysis", "query-eval", "--experiment", str(experiment_dir), "--embed-run",
         "emb_01", "--behavior", "climb", "--method", "dtw",
         "--out", str(out),
         "--overlap-out", str(tmp_path / "ov.csv")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (row,) = query_eval_csv.read_metrics_csv(out)
    assert row.method == "dtw"
    assert row.ap_lift == pytest.approx(row.ap - row.ap_null)
    assert query_eval_csv.read_overlap_csv(tmp_path / "ov.csv")


def test_cli_still_accepts_the_old_gap_flag(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """`--max-gap` was renamed to `--split-gap`; existing sweep scripts must keep running."""
    runner = CliRunner()
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    result = runner.invoke(
        main,
        ["analysis", "query-eval", "--experiment", str(experiment_dir), "--embed-run",
         "emb_01", "--behavior", "groom", "--max-gap", "1.5",
         "--out", str(tmp_path / "old.csv")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output


# --- per-frame review fixes: multi-recording, point detail, disk figures, negatives ------- #


def _second_recording(experiment_dir, make_embedding_run, tmp_path):
    """A second recording mirroring recording_001 (same trajectory + groom/bite labels)."""
    import yaml as _yaml

    s = experiment_dir / "recordings" / "recording_002"
    for d in ("video/timestamps/generated", "detections", "tracking", "embeddings"):
        (s / d).mkdir(parents=True)
    record = _yaml.safe_load(
        (experiment_dir / "recordings" / "recording_001" / "recording.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    record["id"] = record["name"] = "recording_002"
    record["streams"] = []  # start clean; make_embedding_run + add_stream repopulate
    record.pop("embedding_runs", None)
    (s / "recording.yaml").write_text(
        _yaml.dump(record, sort_keys=False), encoding="utf-8"
    )
    emb, frame_idx, t_ns = _trajectory()
    recording_store.write_timestamps_csv(
        s / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(s, emb, frame_idx, t_ns)
    lines = (
        ["start_frame,end_frame,label"]
        + [f"{a},{b},groom" for a, b in GROOM]
        + [f"{p},{p},bite" for p in BITE]
    )
    csv = tmp_path / "labels2.csv"
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Recording(experiment_dir, "recording_002").add_stream(
        "behavior", raw_path=csv, importer="behavior_csv"
    )


def test_perframe_multi_recording_pools_and_renders_display_subject(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """Two recordings: negatives are reservoir-sampled across both subjects, scores/labels pool
    into one ranked AP, and the fold-0 detail is the display subject (search[0])."""
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    _second_recording(experiment_dir, make_embedding_run, tmp_path)
    result = _by_behavior(
        experiment_dir,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 0,
            "n_negatives": 5,
        },
    )["groom"]
    assert result.n_recordings == 2
    assert result.metrics.ap > 0.9  # pooled across both recordings
    assert result.similarity is not None and result.similarity[0] == "recording_001"
    assert result.predictions
    assert {g.role for g in result.ground_truth} == {"scored", "exemplar"}


def test_perframe_point_detail_carries_offset_ns_end_to_end(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """The point branch of the detail rows runs: TP predictions carry offset_ns (not iou)."""
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    result = _by_behavior(
        experiment_dir,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["bite"],
            "method": "soft_knn",
            "folds": 0,
            "n_negatives": 5,
            "point_window_s": 0.5,
        },
    )["bite"]
    assert result.behavior_type == "point"
    tps = [p for p in result.predictions if p.outcome == "tp"]
    assert tps  # point predictions are not blanked by the min-duration filter
    assert all(p.offset_ns is not None and p.iou is None for p in tps)
    assert {g.role for g in result.ground_truth} == {"scored", "exemplar"}


def test_perframe_run_writes_nonblank_detail_to_disk(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """The real module runner writes non-blank predictions/GT + the similarity npz for a
    per-frame run, so the matches/similarity figures render (the blank-figure bug)."""
    from detail.storage import query_eval_csv

    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    run_dir, _config = _run_module(
        experiment_dir, ["groom"], method="soft_knn", folds=0, n_negatives=5
    )
    preds = query_eval_csv.read_predictions_csv(run_dir / "predictions.csv")
    gts = query_eval_csv.read_ground_truth_csv(run_dir / "ground_truth.csv")
    assert preds and {p.outcome for p in preds} <= {"tp", "fp"}
    assert {g.role for g in gts} == {"scored", "exemplar"}
    assert (run_dir / "similarity" / "groom.npz").is_file()


def test_perframe_soft_knn_negatives_do_not_fake_a_perfect_ap(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """A groom frame that looks like background is a hard positive that must rank among the
    (scored) background. With the old negative-exclusion a large n_negatives emptied the test
    set of negatives and faked AP=1.0; keeping negatives in the test set exposes it."""
    emb, frame_idx, t_ns = _trajectory()
    emb[5] = _B  # a labeled-groom frame that looks like background → a hard positive
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, emb, frame_idx, t_ns)
    lines = ["start_frame,end_frame,label"] + [f"{a},{b},groom" for a, b in GROOM]
    csv = tmp_path / "labels.csv"
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Recording(experiment_dir, "recording_001").add_stream(
        "behavior", raw_path=csv, importer="behavior_csv"
    )
    result = _by_behavior(
        experiment_dir,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 0,
            "n_negatives": 100_000,  # >> background; the bug would exclude all of it
        },
    )["groom"]
    assert result.metrics.ap < 0.999  # the hard positive is not hidden


def test_perframe_soft_knn_is_seed_deterministic(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """The seeded reservoir negative draw is reproducible — the 'fix the seed across compared
    runs' contract."""
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    cfg = {
        "embed_run_id": "emb_01",
        "behaviors": ["groom"],
        "method": "soft_knn",
        "folds": 0,
        "n_negatives": 5,
        "seed": 7,
    }
    a = _by_behavior(experiment_dir, cfg)["groom"].metrics
    b = _by_behavior(experiment_dir, cfg)["groom"].metrics
    assert (a.ap, a.tp, a.n_gt, a.n_pred) == (b.ap, b.tp, b.n_gt, b.n_pred)


def test_perframe_metrics_row_carries_its_floor(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """A per-frame AP shipped without a floor reads as a raw win. The span path's
    time-shift null cannot apply (there are no spans to shift), so the per-frame path
    emits the analytic floor — prevalence for AP, same-size coverage for recall — and
    ``metrics.csv`` reads it through the same ``null_at`` lookup the span path uses."""
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    result = _by_behavior(
        experiment_dir,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 0,
            "n_negatives": 5,
        },
    )["groom"]
    floor = null_at(result.overlap, 0.5)
    assert floor is not None, "per-frame run reported no floor"
    assert 0.0 < floor.ap_null < 1.0  # a real prevalence, not a placeholder zero
    assert floor.ap_lift == pytest.approx(result.metrics.ap - floor.ap_null)
    assert 0.0 < floor.recall_null < 1.0
    # a separable behavior must clear its own floor, or the floor is not measuring anything
    assert floor.ap_lift > 0.5


def test_n_holdout_keeps_the_scored_set_fixed_as_the_query_grows(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """The number the whole change exists for: `n_gt` constant across `n_exemplars`.

    Sweeping exemplars without a fixed holdout compares runs scored against different
    ground truth — each extra exemplar removes another instance — so a curve against n
    partly measures its own denominator. Asserting the count, not that the call returned:
    the failure mode here is a plausible number, not a crash.
    """
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)

    def n_gt_at(n_exemplars, n_holdout):
        config = {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 1,
            "n_negatives": 5,
            "n_exemplars": n_exemplars,
            "n_holdout": n_holdout,
        }
        return _by_behavior(experiment_dir, config)["groom"].metrics.n_gt

    fixed = [n_gt_at(n, 3) for n in (1, 2, 3)]
    assert len(set(fixed)) == 1, (
        f"n_gt moved with n_exemplars under a fixed holdout: {fixed} — the scored set is "
        f"supposed to be identical across the sweep"
    )

    # Guard the guard: without the holdout it moves, which is the confound being removed.
    drifting = [n_gt_at(n, None) for n in (1, 2, 3)]
    assert len(set(drifting)) == 3, drifting
    assert drifting[0] > drifting[-1]


def test_an_unqueried_held_out_instance_is_not_scored_as_a_negative(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """Held out but not queried means *out of the test set*, not "a negative".

    This is the half of the change that is invisible in the fold arithmetic: if the
    exclusion mask were still built from the exemplars, the two unqueried held-out grooms
    would be ranked as background — true positives labelled negative, right where the
    classifier scores highest — and AP would fall for a reason that has nothing to do
    with the query.
    """
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    result = _by_behavior(
        experiment_dir,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 1,
            "n_negatives": 5,
            "n_exemplars": 1,
            "n_holdout": 3,
        },
    )["groom"]
    assert result.metrics.ap > 0.95, (
        f"AP={result.metrics.ap:.3f} on a separable fixture — the held-out-but-unqueried "
        f"instances are being scored as negatives"
    )


def test_the_negative_pool_actually_changes_the_scores(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """Guard the guard: a field that changed nothing would pass every leakage test vacuously.

    Asserted on the score curve rather than on AP. This fixture is perfectly separable, so a
    contaminated negative pool lowers *every* positive frame's score by the same amount and AP —
    which is rank-based — cannot move. The mechanism still has to be visible: with the target
    behavior left in the background pool, its own frames sit in the negative class and vote
    against a true hit. On real data, where separability is partial, this does reach AP: a typical
    exemplar scores 0.92 clean and 0.50 unlabeled.
    """
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)

    def positive_score(pool):
        config = {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "method": "soft_knn",
            "folds": 1,
            "seed": 0,
            "n_negatives": 40,
            "negative_pool": pool,
        }
        result = _by_behavior(experiment_dir, config)["groom"]
        _recording_id, _t_ns, score = result.similarity
        # GROOM frames are the one distinct pose in this fixture, so their score is the thing the
        # negative pool moves.
        rows = np.concatenate([np.arange(s, e + 1) for s, e in GROOM])
        return float(np.asarray(score)[rows].mean())

    clean, unlabeled = positive_score("labeled_clean"), positive_score("unlabeled")
    assert clean != unlabeled, (
        f"both pools scored positives at {clean} — the sampling is not branching on "
        f"`negative_pool`, so every leakage test passes for the wrong reason"
    )
    assert clean > unlabeled, (
        f"the label-built pool ({clean:.3f}) should flatter a positive against the honest one "
        f"({unlabeled:.3f}) — it removes exactly the frames that would vote against a true hit"
    )


def test_negative_pool_defaults_to_the_reproducible_one():
    # The default is what an agent or a user gets without thinking about it, which is the whole
    # reason this is a field rather than a note in the docs.
    from detail.configs.query_eval import QueryEvalConfig

    assert QueryEvalConfig(embed_run_id="e").negative_pool == "unlabeled"
