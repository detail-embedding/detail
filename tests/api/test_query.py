"""Tests for §20 calibration and the §19 artifact it produces (detail/api/query.py).

The search half is `tests/pipeline/test_query_module.py`. What is checked here is the half a
smoke test would pass straight through: that an uncalibrated behavior is *absent* from the
artifact rather than present with no rows; that a reprojection is a pure function of the state
and the stored results; and that the numbers the artifact states — coverage spans, timestamps,
the windowed denominator — are the ones a reader will divide by.

The fixture runs the **real** search over synthetic 8-dimensional embeddings with a known motif
planted in them, the way `tests/api/test_annotation.py` does: a stubbed search would not
exercise the row-to-timestamp projection, which is where a wrong number would come from.
"""

import threading

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from detail.cli import main

from detail.api.experiment import Experiment
from detail.pipeline.query import QueryModule
from detail.storage import analysis_runs as analysis_store
from detail.storage import behavior_output, frame_view, query_runs
from detail.storage import recording as recording_store

from tests.pipeline.test_query_module import (
    DIM,
    PERIOD,
    _config,
    _experiment,
    _write_embeddings,
)

# The video runs past the embedded rows, so part of every recording is genuinely unobserved.
# A fixture whose video and embedding coincide makes `observed` all-True and cannot tell a
# correct windowed denominator from one that ignores the mask.
#
# Long against the 500-row background `soft_knn` draws from the cohort's unlabeled remainder.
# That remainder holds the other recordings' copies of the motif, so in a short fixture the
# background draws a large share of the very rows being scored and they vote against
# themselves; at 600 rows the planted motif peaked at 0.86, at 3000 it peaks at 0.999. The
# draw is the honest one (never the label-cleaned pool), so the fixture is sized to it.
N_EMBEDDED = 3000
N_FRAMES = 3200


def _add_video(recording_dir, n_frames: int = N_FRAMES, period: int = PERIOD) -> None:
    """Give a recording the video metadata + timestamp map the frame view's axis comes from."""
    csv_rel = "video/timestamps/generated/part_001.csv"
    path = recording_dir / csv_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "frame_idx,t_ns\n" + "".join(f"{i},{i * period}\n" for i in range(n_frames)),
        encoding="utf-8",
    )
    data = recording_store.read_recording(recording_dir)
    data["video"] = {
        "storage": "default",
        "fps_container": 1000.0,
        "fps_actual": 1000.0,
        "fps_source": "container",
        "resolution": [64, 48],
        "total_duration_s": n_frames * period / 1e9,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": csv_rel,
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    }
    recording_store.write_recording(recording_dir, data)


def _searched(
    tmp_path,
    recordings=("s1", "s2"),
    behaviors=None,
    with_video=True,
    cx=None,
    n_rows=N_EMBEDDED,
):
    """An experiment with a completed Query search over `recordings`. Returns (exp, run).

    ``cx`` writes §14 kinematics alongside the embeddings — off by default, because a run made
    before kinematics existed has none and that is the state most of these fixtures reproduce.
    ``n_rows`` is for a test that needs few candidates (an audit pool it can exhaust) more than
    a confident motif: a shorter fixture forms fewer noise peaks.
    """
    proj, _motif_rows = _experiment(tmp_path, list(recordings), n_rows=n_rows, cx=cx)
    if with_video:
        for sid in recordings:
            _add_video(proj.get_recording(sid).dir)
    config = _config(list(recordings), behaviors=behaviors)
    module = QueryModule()
    normalized = module.validate(proj.dir, dict(config))
    job = module.build_job("an_01", normalized)
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    module.runner(job, run_dir, lambda _e: None, threading.Event())
    exp = Experiment(proj.dir)
    return exp, exp.query_run("an_01")


def _write_gapped_embeddings(recording_dir, embed_run_id="emb_01") -> None:
    """Embeddings with a genuine tracking gap: rows 0-299 and 400-699, nothing between.

    The contiguous fixture cannot tell a coverage denominator that subtracts a gap from one
    that spans the whole recording — with one continuous run the two are the same number.
    """
    import yaml as _yaml

    from detail.storage.zarr_io import EmbeddingPartWriter

    rng = np.random.default_rng(11)
    frames = list(range(0, 300)) + list(range(400, 700))
    emb = rng.normal(size=(len(frames), DIM)).astype(np.float32)
    run_dir = recording_dir / "embeddings" / embed_run_id
    store = run_dir / "track_0" / "part_001.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    with EmbeddingPartWriter(store, output_dim=DIM) as w:
        for row, frame in enumerate(frames):
            w.add_row(frame, frame * PERIOD, 0, emb[row])
    (run_dir / "embedding.yaml").write_text(
        _yaml.dump(
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


def test_coverage_subtracts_a_tracking_gap_and_the_frame_view_agrees(tmp_path):
    """The denominator every rate ships with, on a subject that was not tracked throughout.

    600 embedded rows spanning 700 frames: `observed_ns` must be the 600, not the 700. And it
    must equal what the frame view independently computes from the dense axis and its mask —
    the two are the same rule applied to different inputs, so they agree by construction or
    one of them is wrong.
    """
    from detail.api.experiment import Experiment as _Experiment

    proj = _Experiment.create("p", tmp_path)
    recording = proj.create_recording("s1")
    _write_gapped_embeddings(recording.dir)
    _add_video(recording.dir)

    module = QueryModule()
    config = {
        "source_recording_ids": ["s1"],
        "embed_run_id": "emb_01",
        "track_id": 0,
        "method": "soft_knn",
        "method_params": {},
        "behaviors": [
            {
                "name": "rearing",
                "examples": [
                    {
                        "recording_id": "s1",
                        "track_id": 0,
                        "start_t_ns": 10 * PERIOD,
                        "end_t_ns": 29 * PERIOD,
                    }
                ],
            }
        ],
    }
    job = module.build_job("an_01", module.validate(proj.dir, dict(config)))
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    module.runner(job, run_dir, lambda _e: None, threading.Event())

    run = Experiment(proj.dir).query_run("an_01")
    row = next(r for r in run.artifact().coverage if r.recording_id == "s1")

    assert row.start_t_ns == 0
    assert (
        row.end_t_ns == 700 * PERIOD
    )  # the last embedded row's frame, plus its period
    assert row.n_observed == 600
    assert row.observed_ns == 600 * PERIOD  # the gap is subtracted
    assert row.observed_ns < row.end_t_ns - row.start_t_ns

    view = frame_view.read_frame_view(run_dir, "s1", 0)
    assert frame_view.observed_ns_in(view) == row.observed_ns
    # and the gap is where it actually is, which is the whole reason the view exists
    assert frame_view.observed_ns_in(view, 300 * PERIOD, 400 * PERIOD) == 0
    assert frame_view.observed_ns_in(view, 0, 300 * PERIOD) == 300 * PERIOD


def _two_behaviors(recordings, plant_at=100):
    examples = [
        {
            "recording_id": recordings[0],
            "track_id": 0,
            "start_t_ns": plant_at * PERIOD,
            "end_t_ns": (plant_at + 19) * PERIOD,
        }
    ]
    return [
        {"name": "rearing", "examples": examples},
        {"name": "grooming", "examples": examples},
    ]


# --- the rule the whole artifact rests on -------------------------------------------------


def test_an_uncalibrated_behavior_is_absent_not_a_zero(tmp_path):
    """§19 reads a vocabulary entry with no rows inside coverage as a **real zero** — looked
    for, never happened. A behavior nobody has thresholded yet has not been looked for, so
    listing it states something false in a table that stays perfectly well-formed.
    """
    exp, run = _searched(tmp_path, behaviors=_two_behaviors(["s1", "s2"]))
    run.set_cutoff("q01", 0.9)

    output = run.artifact()
    assert output.manifest["vocabulary"] == ["rearing"]
    assert "grooming" not in output.manifest["names"].values()
    assert "q02" not in output.manifest["names"]
    assert {row.source_label for row in output.instances} == {"q01"}
    assert "grooming" not in output.manifest["behavior_types"]

    # ...and calibrating it is what puts it there.
    run.set_cutoff("q02", 0.9)
    output = run.artifact()
    assert output.manifest["vocabulary"] == ["grooming", "rearing"]
    assert {row.source_label for row in output.instances} == {"q01", "q02"}


def test_zero_matches_is_a_first_class_state(tmp_path):
    """A calibrated behavior that returns nothing is *present* with no rows — which is exactly
    the real zero the rule above protects. It must be readable as one."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 1.0)  # nothing clears a perfect-confidence cutoff

    output = run.artifact()
    assert output.manifest["vocabulary"] == ["rearing"]
    assert output.instances == []
    # and the units were still analyzed, which is what separates this from "never looked".
    assert {row.recording_id for row in output.coverage} == {"s1", "s2"}


# --- when the artifact exists ------------------------------------------------------------


def test_the_search_writes_coverage_before_anything_is_calibrated(tmp_path):
    """§21: an `open` run *has* an artifact — the statistics are what the user is deciding
    against, so withholding it until finalize would invert the workflow."""
    exp, run = _searched(tmp_path)
    assert run.status == "open"

    output = run.artifact()
    assert output.manifest["vocabulary"] == []
    assert output.instances == []
    assert {row.recording_id for row in output.coverage} == {"s1", "s2"}


def test_an_interrupted_search_writes_no_artifact(tmp_path):
    """`partial` means interrupted compute, which §19 reads as "no artifact yet": a run that
    stopped early cannot honestly state coverage over recordings it never searched.

    Asserts the **files**, not that the call returned — the artifact's absence is the claim.
    """
    proj, _rows = _experiment(tmp_path, ["s1", "s2"], n_rows=N_EMBEDDED)
    module = QueryModule()
    normalized = module.validate(proj.dir, dict(_config(["s1", "s2"])))
    job = module.build_job("an_01", normalized)
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    stop = threading.Event()
    stop.set()
    module.runner(job, run_dir, lambda _e: None, stop)

    assert analysis_store.read_analysis_yaml(run_dir)["status"] == "partial"
    assert not behavior_output.has_behavior_output(run_dir)
    assert not (run_dir / "instances.csv").exists()
    assert not (run_dir / "coverage.csv").exists()


# --- reprojection is a pure function of state + stored results ----------------------------


def test_a_cutoff_change_reprojects_and_a_rename_does_not_square_the_rows(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    before = (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig")

    run.set_cutoff("q01", 0.1)  # low enough to admit peaks the noise forms
    assert (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig") != before

    # A rename re-points the label and leaves the row count alone: the writer fans a bout out
    # over the manifest's names, so a producer that also passed `label` would double them.
    n_rows = len(run.artifact().instances)
    run.name_behavior("q01", "rearing v2")
    after = run.artifact()
    assert len(after.instances) == n_rows
    assert {row.label for row in after.instances} == {"rearing v2"}


def test_reprojection_is_idempotent(tmp_path):
    """Reproject twice, byte-identical — the row-squaring failure one level up, where a
    re-expansion fans an already-expanded row out again."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.7)
    first = {
        name: (run._run_dir() / name).read_text(encoding="utf-8-sig")
        for name in ("instances.csv", "coverage.csv", "behavior_output.yaml")
    }
    run.reproject()
    for name, text in first.items():
        assert (run._run_dir() / name).read_text(encoding="utf-8-sig") == text


def test_the_artifact_round_trips(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    output = behavior_output.read_behavior_output(run._run_dir())

    assert output.manifest["origin"] == "predicted"
    assert output.manifest["label_space"] == "pooled"
    assert output.manifest["overlap"] == "allowed"
    assert output.manifest["producer"] == {"module_id": "query", "run_id": "an_01"}
    assert output.manifest["behavior_types"] == {"rearing": "state"}
    assert output.instances and all(row.label == "rearing" for row in output.instances)


def test_score_direction_is_higher_is_better(tmp_path):
    """A distance-scored producer says `lower_is_better` (§19's manifest example once did, for the
    removed `clip_query`). §20 makes confidence canonical — [0, 1], higher better — and confidence
    is what lands in the score column. Copying the example inverts every ranking built on the
    artifact, with nothing failing anywhere.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    output = run.artifact()

    assert output.manifest["score"]["direction"] == "higher_is_better"
    assert output.manifest["score"]["meaning"]
    # the planted motif is its own best match, so the top score is at the *high* end
    assert max(row.score for row in output.instances) > 0.9


# --- the numbers a reader divides by ------------------------------------------------------


def test_coverage_covers_every_searched_unit_including_one_that_matched_nothing(
    tmp_path,
):
    """A searched recording with no instances still gets a coverage row — that is exactly what
    makes its zero readable as a real one rather than as "never analyzed"."""
    proj, _rows = _experiment(tmp_path, ["s1", "s2"], n_rows=N_EMBEDDED)
    # A third recording with no motif planted in it: searched, and genuinely without the
    # behavior. The two-recording fixture cannot produce this, because it plants in every one.
    blank = proj.create_recording("s3")
    _write_embeddings(blank.dir, N_EMBEDDED)
    for sid in ("s1", "s2", "s3"):
        _add_video(proj.get_recording(sid).dir)

    module = QueryModule()
    normalized = module.validate(proj.dir, dict(_config(["s1", "s2", "s3"])))
    job = module.build_job("an_01", normalized)
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    module.runner(job, run_dir, lambda _e: None, threading.Event())

    run = Experiment(proj.dir).query_run("an_01")
    run.set_cutoff("q01", 0.95)
    output = run.artifact()

    assert {row.unit for row in output.coverage} == {("s1", 0), ("s2", 0), ("s3", 0)}
    assert {row.recording_id for row in output.instances} == {"s1", "s2"}
    # the blank recording is covered, and its coverage says real time was observed there
    blank_row = next(r for r in output.coverage if r.recording_id == "s3")
    assert blank_row.observed_ns > 0


def test_coverage_end_t_ns_is_the_span_end_not_the_last_timestamp(tmp_path):
    """A row covers its sampling interval, so a fully observed unit of N rows spans N periods
    while its last row sits at (N-1) periods.

    Writing the last timestamp instead claims more observed time than the span is long, and
    the writer refuses it — with a message about a deflated denominator that reads as baffling
    when the recording is perfectly fine. So the check is that `observed_ns` **equals** the
    span for a fully observed unit, which only holds at `t_ns[-1] + period`.
    """
    exp, run = _searched(tmp_path)
    row = next(r for r in run.artifact().coverage if r.recording_id == "s1")

    assert row.start_t_ns == 0
    assert row.end_t_ns == N_EMBEDDED * PERIOD
    assert row.n_observed == N_EMBEDDED
    assert row.observed_ns == row.end_t_ns - row.start_t_ns


def test_timeline_and_trajectory_index_rows_identically(tmp_path):
    """Every projected timestamp rests on this substitution.

    Reprojection reads `load_track_timeline` because the trajectory drags 589 MB of embeddings
    it does not need. That is only sound because the two walk the same sorted parts and skip
    empty ones the same way, so row k is the same row in both. If they ever diverge, every
    instance in every artifact is silently offset — a wrong number, not a crash.
    """
    from detail.api.run_preview import load_track_timeline, load_track_trajectory

    exp, run = _searched(tmp_path)
    recording_dir = exp.get_recording("s1").dir

    frame_idx, t_ns = load_track_timeline(recording_dir, "emb_01", 0)
    _emb, traj_frame_idx, traj_t_ns = load_track_trajectory(recording_dir, "emb_01", 0)

    assert np.array_equal(frame_idx, traj_frame_idx)
    assert np.array_equal(t_ns, traj_t_ns)


def test_instances_land_on_the_planted_motif(tmp_path):
    """The projection end to end: a motif planted at rows 100-119 must come back as the span
    those rows occupy, in both time and frames.

    Within a row rather than exactly: `soft_knn` sizes a match by where its smoothed score
    falls to a fraction of the peak, so the extent is the method's estimate of the motif, not
    its exact edges. What a projection error would break is the agreement between the row, its
    timestamp and its frame, and that is asserted exactly.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.95)
    output = run.artifact()

    assert {row.recording_id for row in output.instances} == {"s1", "s2"}
    for row in output.instances:
        assert abs(row.start_frame - 100) <= 1 and abs(row.end_frame - 119) <= 1, row
        assert row.t_ns == row.start_frame * PERIOD
        assert row.end_t_ns == row.end_frame * PERIOD


# --- the frame view, and the windowed denominator it exists for ---------------------------


def test_a_frame_view_is_written_for_every_searched_unit(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    run.bring_frame_views_current()  # a cutoff defers the views to their first reader (#325)

    assert sorted(frame_view.discover_frame_views(run._run_dir())) == [
        ("s1", 0),
        ("s2", 0),
    ]
    view = frame_view.read_frame_view(run._run_dir(), "s1", 0)
    assert view.n_frames == N_FRAMES  # the dense video axis, not the embedded rows
    assert view.source_labels == ["q01"]


def test_the_frame_view_marks_unembedded_frames_unobserved(tmp_path):
    """The video runs 200 frames past the embedded rows, so those frames were never analyzed.

    `coverage.observed_ns` records how much was observed and not *where*, so this is the only
    thing that can answer a windowed question — and `analysis/measures.py` refuses a windowed
    denominator without it rather than substituting the whole-recording number.
    """
    exp, run = _searched(tmp_path)
    view = frame_view.read_frame_view(run._run_dir(), "s1", 0)
    observed = np.asarray(view.observed[:], dtype=bool)

    assert observed[:N_EMBEDDED].all()
    assert not observed[N_EMBEDDED:].any()

    # The windowed denominator over the tail is zero, while the naive whole-recording scaling
    # would have called it a quarter of the observed time.
    assert frame_view.observed_ns_in(view, N_EMBEDDED * PERIOD, N_FRAMES * PERIOD) == 0
    assert (
        frame_view.observed_ns_in(view, 0, N_EMBEDDED * PERIOD) == N_EMBEDDED * PERIOD
    )


def test_the_frame_view_denominator_agrees_with_coverage(tmp_path):
    """Coverage's `observed_ns` is `observed_mask`'s rule transposed onto the analyzed rows,
    so the two must agree over the whole span — by construction, not coincidence."""
    exp, run = _searched(tmp_path)
    row = next(r for r in run.artifact().coverage if r.recording_id == "s1")
    view = frame_view.read_frame_view(run._run_dir(), "s1", 0)

    assert frame_view.observed_ns_in(view) == row.observed_ns


def test_the_frame_view_carries_the_instances(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.95)
    run.bring_frame_views_current()  # a cutoff defers the views to their first reader (#325)
    view = frame_view.read_frame_view(run._run_dir(), "s1", 0)
    column = np.asarray(view.root["q01"][:])
    (match,) = [r for r in run.artifact().instances if r.recording_id == "s1"]
    start, end = match.start_frame, match.end_frame + 1
    assert abs(start - 100) <= 1  # it is the planted motif

    assert column[start:end].all()  # the match's frames carry an ordinal
    assert not column[:start].any()
    assert not column[end:].any()


# --- the authored lifecycle ---------------------------------------------------------------


def test_a_finalized_run_refuses_an_edit_and_names_the_way_back(tmp_path):
    """§21 calls finalize "a statement by the user that they are done, not a lock" — and this
    is not one, since `reopen()` is a single call away. But a statement that can be silently
    overwritten is not a statement, so the error is what makes reopening discoverable.
    """
    exp, run = _searched(tmp_path)
    run.finalize()

    with pytest.raises(ValueError, match="reopen"):
        run.set_cutoff("q01", 0.5)
    with pytest.raises(ValueError, match="reopen"):
        run.name_behavior("q01", "something else")


def test_finalize_reopen_round_trips_and_the_artifact_matches_the_state(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    run.finalize()
    assert exp.query_run("an_01").status == "complete"

    run.reopen()
    assert exp.query_run("an_01").status == "open"
    run.set_cutoff("q01", 0.5)
    run.finalize()

    output = exp.query_run("an_01").artifact()
    assert output.manifest["method"]["cutoffs"] == {"q01": 0.5}
    assert len(output.instances) == len(
        [p for p in run.frequency_curve("q01") if p["cutoff"] == 0.5]
        and output.instances
    )


def test_every_authored_edit_survives_a_fresh_handle(tmp_path):
    """Re-read through a *fresh* `query_run(run_id)` rather than the object that made the
    change, so nothing passes on in-memory state."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.85)
    run.name_behavior("q01", "reared up")

    fresh = exp.query_run("an_01")
    behavior = fresh.state().behavior("q01")
    assert behavior.cutoff == 0.85
    assert behavior.name == "reared up"
    assert fresh.artifact().manifest["vocabulary"] == ["reared up"]


def test_a_rename_keeps_the_behavior_id_and_its_instance_ids(tmp_path):
    """§19 hashes `source_label` into every `instance_id`, and audit decisions are keyed by
    those. An id that moved when the user renamed would re-point every recorded judgment at a
    different behavior."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    before = sorted(row.instance_id for row in run.artifact().instances)

    run.name_behavior("q01", "totally different name")
    assert sorted(row.instance_id for row in run.artifact().instances) == before


def test_a_duplicate_behavior_name_is_refused(tmp_path):
    exp, run = _searched(tmp_path, behaviors=_two_behaviors(["s1", "s2"]))
    with pytest.raises(ValueError, match="already called"):
        run.name_behavior("q02", "rearing")


def test_a_cutoff_outside_the_confidence_range_is_refused(tmp_path):
    exp, run = _searched(tmp_path)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run.set_cutoff("q01", 1.5)


def test_clearing_a_cutoff_removes_the_behavior_again(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    assert run.artifact().manifest["vocabulary"] == ["rearing"]

    run.set_cutoff("q01", None)
    assert run.artifact().manifest["vocabulary"] == []
    assert run.artifact().instances == []


def test_an_unknown_behavior_id_names_the_ones_that_exist(tmp_path):
    exp, run = _searched(tmp_path)
    with pytest.raises(KeyError, match="q01"):
        run.set_cutoff("q99", 0.5)


# --- one writer per fact ------------------------------------------------------------------


def test_query_yaml_carries_no_status_and_no_counters(tmp_path):
    """§21's one-writer invariant. The lifecycle status lives in `analysis.yaml` and nowhere
    else, and every count is derived from the artifact — a stored second copy stays
    well-formed while it drifts, which is why nothing fails when it does."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    raw = yaml.safe_load(
        query_runs.query_yaml_path(run._run_dir()).read_text(encoding="utf-8-sig")
    )

    assert "status" not in raw
    forbidden = {"n_instances", "instance_count", "n_matches", "count", "audited"}
    assert not forbidden & set(raw)
    for behavior in raw["behaviors"]:
        assert not forbidden & set(behavior)


def test_history_records_each_operation_without_deciding_anything(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    run.name_behavior("q01", "reared")
    run.finalize()

    lines = (
        (run._run_dir() / query_runs.HISTORY_FILENAME)
        .read_text(encoding="utf-8-sig")
        .splitlines()
    )
    ops = [yaml.safe_load(line)["op"] for line in lines]
    assert ops == ["search", "set_cutoff", "name_behavior", "finalize"]


# --- the calibration curve ----------------------------------------------------------------


def test_the_frequency_curve_spans_the_confidence_range(tmp_path):
    exp, run = _searched(tmp_path)
    curve = run.frequency_curve("q01", steps=11)

    assert [round(p["cutoff"], 2) for p in curve] == [
        round(i / 10, 2) for i in range(11)
    ]
    assert curve[0]["n_instances"] > curve[-1]["n_instances"]
    assert all(p["capped"] is False for p in curve)  # soft_knn declares no cap


def test_the_frequency_curve_marks_a_point_its_cap_bound(tmp_path, monkeypatch):
    """§20: "a curve that flattens because storage ran out looks exactly like a curve that
    flattens because the animal stopped." The method declares its cap; the curve says which
    points it bound.

    `soft_knn` keeps every peak and declares no cap, so one is declared for it here: what is
    under test is the curve reading the method's declaration, not a method that has one.
    """
    from detail.query_methods.soft_knn import SoftKnnQueryMethod

    monkeypatch.setattr(SoftKnnQueryMethod, "candidate_cap", lambda self, params: 2)
    exp, run = _searched(tmp_path)

    curve = run.frequency_curve("q01", steps=11)
    assert any(p["capped"] for p in curve)
    # Marked only where a unit reached the cap: at the strictest cutoff each unit keeps its
    # single motif, and that point is not capped.
    assert curve[-2]["n_instances"] == 2 and not curve[-2]["capped"]


def test_the_frequency_curve_names_an_unknown_behavior(tmp_path):
    exp, run = _searched(tmp_path)
    with pytest.raises(KeyError, match="q01"):
        run.frequency_curve("q99")


# --- export -------------------------------------------------------------------------------


def test_export_writes_one_stream_csv_per_recording_carrying_origin(tmp_path):
    """§19's deliberate re-ingest. `origin: predicted` is what stops a prediction being read
    back as ground truth once it is a behavior stream."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.9)
    out = run.export(tmp_path / "export")

    assert sorted(p.name for p in out.iterdir()) == ["s1.csv", "s2.csv"]
    lines = (out / "s1.csv").read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "start_frame,end_frame,label,subject,origin"
    assert lines[1].endswith(",rearing,0,predicted")


def test_export_writes_an_empty_csv_for_a_recording_with_no_instances(tmp_path):
    """A searched recording that matched nothing gets a header-only file, not no file — the
    same real-zero distinction the vocabulary rule makes, one surface over."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 1.0)
    out = run.export(tmp_path / "export")

    for name in ("s1.csv", "s2.csv"):
        assert (out / name).read_text(encoding="utf-8-sig").splitlines() == [
            "start_frame,end_frame,label,subject,origin"
        ]


def test_export_before_the_search_completes_says_so(tmp_path):
    proj, _rows = _experiment(tmp_path, ["s1"], n_rows=N_EMBEDDED)
    run_dir = analysis_store.analysis_dir(proj.dir, "an_09")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_09", "module_id": "query", "status": "partial"}
    )
    query_runs.write_query_state(
        run_dir,
        query_runs.QueryState(
            embed_run_id="emb_01",
            track_id=0,
            method="soft_knn",
            behaviors=[query_runs.Behavior(id="q01", name="rearing")],
            search_recording_ids=["s1"],
        ),
    )
    # A `partial` run is refused before anything looks for the artifact, naming what the
    # search never reached (`tests/api/test_query_partial.py` has the class).
    with pytest.raises(ValueError, match=r"partial.*q01 on s1.*re-run the search"):
        Experiment(proj.dir).query_run("an_09").export(tmp_path / "export")


# --- provenance ---------------------------------------------------------------------------


def test_provenance_is_captured_once_and_carried_through_every_reprojection(tmp_path):
    """It fingerprints the run's *inputs*, which a cutoff does not change — so re-capturing
    per calibration would be work that cannot alter its own answer."""
    exp, run = _searched(tmp_path)
    record = analysis_store.read_analysis_yaml(run._run_dir())
    captured = record["provenance"]
    assert captured

    run.set_cutoff("q01", 0.9)
    assert run.artifact().manifest["provenance"] == captured
    run.set_cutoff("q01", 0.5)
    assert run.artifact().manifest["provenance"] == captured


# --- the handle without a live job --------------------------------------------------------


def test_a_handle_with_no_executor_job_is_inert_rather_than_broken(tmp_path):
    """`AnalysisRun` normally wraps a live executor handle; a run reached later to calibrate
    has none, and every inherited member has to survive that."""
    exp, run = _searched(tmp_path)

    assert run._inner is None
    assert run.error is None
    assert list(run.progress()) == []
    assert run.wait() is run
    run.cancel()  # a no-op, not an AttributeError
    assert run.status == "open"


# --- the CLI surface ----------------------------------------------------------------------


def _cli(*args):
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def test_cli_prints_the_frequency_curve_without_editing_anything(tmp_path):
    exp, run = _searched(tmp_path)
    before = query_runs.query_yaml_path(run._run_dir()).read_text(encoding="utf-8-sig")

    out = _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01",
        "--curve", "q01", "--steps", "5",
    )  # fmt: skip

    assert "cutoff" in out and "instances" in out
    assert out.count("\n") > 5
    assert (
        query_runs.query_yaml_path(run._run_dir()).read_text(encoding="utf-8-sig")
        == before
    )


def test_cli_sets_a_cutoff_and_reprojects(tmp_path):
    exp, run = _searched(tmp_path)
    out = _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01", "--cutoff", "q01=0.9",
    )  # fmt: skip

    assert "q01 cutoff 0.9" in out
    assert exp.query_run("an_01").state().behavior("q01").cutoff == 0.9
    assert exp.query_run("an_01").artifact().manifest["vocabulary"] == ["rearing"]


def test_cli_renames_a_behavior_and_reports_the_state(tmp_path):
    exp, run = _searched(tmp_path)
    out = _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01",
        "--name", "q01=reared up", "--cutoff", "q01=0.9",
    )  # fmt: skip

    assert "reared up" in out
    assert "run an_01 is open" in out
    assert exp.query_run("an_01").artifact().manifest["vocabulary"] == ["reared up"]


def test_cli_names_an_uncalibrated_behavior_as_such(tmp_path):
    exp, run = _searched(tmp_path, behaviors=_two_behaviors(["s1", "s2"]))
    out = _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01", "--cutoff", "q01=0.9",
    )  # fmt: skip

    assert "q02  grooming  (not calibrated)" in out
    assert "vocabulary ['rearing']" in out


def test_cli_finalize_then_reopen(tmp_path):
    exp, _run = _searched(tmp_path)
    _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01",
        "--cutoff", "q01=0.9", "--finalize",
    )  # fmt: skip
    assert exp.query_run("an_01").status == "complete"

    # ...and a further edit is refused until the run is reopened, in the same invocation.
    failed = CliRunner().invoke(
        main,
        ["analysis", "query-calibrate", "--experiment", str(exp.dir),
         "--run", "an_01", "--cutoff", "q01=0.5"],
    )  # fmt: skip
    assert failed.exit_code != 0
    assert "reopen" in failed.output

    out = _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01",
        "--reopen", "--cutoff", "q01=0.5",
    )  # fmt: skip
    assert "reopened an_01" in out
    assert exp.query_run("an_01").state().behavior("q01").cutoff == 0.5


def test_cli_clears_a_cutoff(tmp_path):
    exp, _run = _searched(tmp_path)
    _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01", "--cutoff", "q01=0.9",
    )  # fmt: skip
    out = _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01", "--cutoff", "q01=none",
    )  # fmt: skip

    assert "absent from the artifact" in out
    assert exp.query_run("an_01").artifact().manifest["vocabulary"] == []


def test_cli_exports_the_artifact(tmp_path):
    exp, _run = _searched(tmp_path)
    _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01", "--cutoff", "q01=0.9",
    )  # fmt: skip
    out_dir = tmp_path / "exported"
    _cli(
        "analysis", "export-query",
        "--experiment", str(exp.dir), "--run", "an_01", "--out", str(out_dir),
    )  # fmt: skip

    assert sorted(p.name for p in out_dir.iterdir()) == ["s1.csv", "s2.csv"]
    assert "predicted" in (out_dir / "s1.csv").read_text(encoding="utf-8-sig")


def test_cli_reports_a_bad_cutoff_spec_rather_than_traceback(tmp_path):
    exp, _run = _searched(tmp_path)
    result = CliRunner().invoke(
        main,
        ["analysis", "query-calibrate", "--experiment", str(exp.dir),
         "--run", "an_01", "--cutoff", "q01"],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "BEHAVIOR=CONFIDENCE" in result.output


# --- the reads a calibration surface needs ------------------------------------------------


def test_load_results_warms_the_extraction_once(tmp_path, monkeypatch):
    """The one genuinely sized step in calibration, done behind a named call with a bar.

    A method caches its candidate list on the result object, so whoever reads it first pays
    for the whole extraction. Left implicit, that cost lands on whichever redraw happened to
    come first and the surface freezes there instead of showing a bar.
    """
    exp, run = _searched(tmp_path)
    stages = []
    loaded = run.load_results("q01", progress=lambda s, f, d: stages.append((s, f)))

    assert set(loaded) == {("s1", 0), ("s2", 0)}
    assert stages and stages[0][0] == "loading"
    assert stages[-1][1] == 1.0
    assert 0 < min(f for _s, f in stages) or len(stages) > 1  # it moves, not one jump

    # ...and a second call is free: same objects, no reload.
    again = run.load_results("q01")
    assert all(again[k] is loaded[k] for k in loaded)


def test_the_curve_reuses_the_warmed_results(tmp_path, monkeypatch):
    """Building the curve twice must not re-read a shard — the drag redraws constantly."""
    exp, run = _searched(tmp_path)
    run.load_results("q01")

    from detail.query_methods.soft_knn import SoftKnnQueryMethod

    loads = []
    real_load = SoftKnnQueryMethod.load
    monkeypatch.setattr(
        SoftKnnQueryMethod,
        "load",
        lambda self, path: (loads.append(path), real_load(self, path))[1],
    )
    run.frequency_curve("q01", steps=5)
    run.frequency_curve("q01", steps=5)
    assert loads == []


def test_candidates_at_does_not_touch_the_artifact(tmp_path):
    """What the view shows while the user is still deciding: matches at a cutoff, uncommitted.

    This is the read half of §20's drag rule — the artifact is rewritten on release, never on
    the way there.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.5)
    before = (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig")

    loose = run.candidates_at("q01", 0.0)
    tight = run.candidates_at("q01", 0.99)

    assert sum(len(v) for v in loose.values()) > sum(len(v) for v in tight.values())
    assert (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig") == before


def test_score_distribution_covers_every_searched_recording(tmp_path):
    """§20 requires the per-recording distributions to be always visible — a recording whose
    search returned nothing must appear as an empty histogram, not be absent. An absent row
    reads as "not searched", which is the distinction coverage exists to preserve."""
    proj, _rows = _experiment(tmp_path, ["s1", "s2"], n_rows=N_EMBEDDED)
    blank = proj.create_recording("s3")
    _write_embeddings(blank.dir, N_EMBEDDED)
    for sid in ("s1", "s2", "s3"):
        _add_video(proj.get_recording(sid).dir)
    module = QueryModule()
    job = module.build_job(
        "an_01", module.validate(proj.dir, dict(_config(["s1", "s2", "s3"])))
    )
    run_dir = analysis_store.analysis_dir(proj.dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "query", "status": "running"}
    )
    module.runner(job, run_dir, lambda _e: None, threading.Event())

    run = Experiment(proj.dir).query_run("an_01")
    dist = run.score_distribution("q01")
    assert set(dist) == {"s1", "s2", "s3"}
    assert all(len(counts) == len(edges) - 1 for counts, edges in dist.values())
    # the planted recordings have mass at the top of the range; the blank one does not
    assert dist["s1"][0][-1] > 0

    # ...and a recording whose search never produced a shard still appears, all-zero. That is
    # the case the seeding exists for: `load_results` skips a unit with no shard, so without
    # it the row is absent — and an absent row reads as "not searched", which is exactly the
    # distinction coverage exists to preserve.
    query_runs.shard_path(run._run_dir(), "q01", "s2", 0, ".npz").unlink()
    fresh = Experiment(proj.dir).query_run("an_01").score_distribution("q01")
    assert set(fresh) == {"s1", "s2", "s3"}
    assert fresh["s2"][0].sum() == 0


def test_recording_duration_is_the_videos_not_the_matches(tmp_path):
    exp, run = _searched(tmp_path)
    assert run.recording_duration_ns("s1") == N_FRAMES * PERIOD


# --- the long-op rule ----------------------------------------------------------------------


def test_setting_a_cutoff_reports_progress(tmp_path):
    """Reprojection reads a shard and writes a frame-view zarr per unit. A synchronous API
    method doing real work must accept a callback and report through it — a silent long op is
    indistinguishable from a hang."""
    exp, run = _searched(tmp_path)
    seen = []
    run.set_cutoff("q01", 0.9, progress=lambda s, f, d: seen.append((s, f, d)))

    stages = [s for s, _f, _d in seen]
    assert seen, "set_cutoff reported nothing"
    # it names the phases it moves through, rather than holding one label to the end
    assert len(set(stages)) > 1, f"only one phase reported: {set(stages)}"
    assert "projecting" in stages
    # The frame views are the other long step, and a cutoff defers them to their first reader
    # (#325) — so that is where their progress has to come from.
    rebuilt = []
    assert run.bring_frame_views_current(progress=lambda s, f, d: rebuilt.append(s))
    assert any("frame view" in s for s in rebuilt), "the deferred rebuild ran silently"


def test_the_cli_drives_a_bar_through_a_phase_change(tmp_path, monkeypatch):
    """The caller side of the same rule: the CLI passes a real callback, and it survives the
    stage changing (the `_StageProgress` contract, which re-labels and resets on a new stage).
    """
    exp, _run = _searched(tmp_path)
    seen = []

    import detail.api.query as query_api

    real = query_api.QueryRun.set_cutoff

    def spy(self, behavior_id, cutoff, *, progress=None):
        assert progress is not None, "the CLI passed no progress callback"
        progress("checking", 0.0, "")
        progress("checking again", 0.5, "")  # a stage change must not raise
        seen.append(True)
        return real(self, behavior_id, cutoff, progress=progress)

    monkeypatch.setattr(query_api.QueryRun, "set_cutoff", spy)
    _cli(
        "analysis", "query-calibrate",
        "--experiment", str(exp.dir), "--run", "an_01", "--cutoff", "q01=0.9",
    )  # fmt: skip
    assert seen == [True]


def test_a_commit_reuses_results_already_in_memory(tmp_path, monkeypatch):
    """A calibration surface loads the results to draw its curve; committing a cutoff must not
    re-read and re-extract them. Before the extraction was made linear this was the difference
    between a usable slider and an unusable one; it is still the difference between one disk
    read and one per commit."""
    exp, run = _searched(tmp_path)
    run.load_results("q01")

    from detail.query_methods.soft_knn import SoftKnnQueryMethod

    loads = []
    real_load = SoftKnnQueryMethod.load
    monkeypatch.setattr(
        SoftKnnQueryMethod,
        "load",
        lambda self, path: (loads.append(path), real_load(self, path))[1],
    )
    run.set_cutoff("q01", 0.9)
    assert loads == []


# --- the shell reads the flag, not a list of statuses --------------------------------------


def test_an_authored_module_declares_itself_mutable():
    """§21: the flag lives on the AnalysisModule so the API and CLI know it too, not only the
    GUI module UI."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    assert ANALYSIS_MODULES["query"].mutable is True
    # ...and a computed module is not, by inheriting the default rather than restating it
    assert ANALYSIS_MODULES["location_export"].mutable is False


# --------------------------------------------------------------------------- #
# the reliability audit (§19)
# --------------------------------------------------------------------------- #
def test_the_audit_samples_at_no_cutoff_so_moving_one_costs_no_labels(tmp_path):
    """The property the whole audit rests on. `instances.csv` holds only what the current
    cutoff kept, so sampling *it* would make the evidence a function of a decision the user is
    still making — set-cutoff, audit, re-set, re-audit. Enumerating at zero instead means one
    audit serves every cutoff."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.99)  # a strict cutoff: the artifact is nearly empty
    kept = len(run.artifact().instances)

    drawn = run.audit_sample(label="q01", budget=10)
    assert len(drawn) > kept, (
        f"the sample ({len(drawn)}) came from the artifact ({kept} instances), not from the "
        "cutoff-free candidate set"
    )


def test_a_decision_survives_the_cutoff_moving(tmp_path):
    """`instance_id` is a content hash of the span, so a label recorded at one cutoff still
    attaches to its instance after the artifact is rebuilt at another."""
    from detail.storage import behavior_audit as audit

    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    drawn = run.audit_sample(label="q01", budget=5)
    for row in drawn:
        run.label_audit(row.instance_id, True)

    run.set_cutoff("q01", 0.5)
    decisions = {r.instance_id for r in audit.latest_decisions(run._run_dir())}
    assert {r.instance_id for r in drawn} <= decisions
    got = audit.precision_at_cutoff(
        run._run_dir(), run.artifact().instances, cutoff=None
    )
    assert got is not None and got["n"] >= 1


def test_an_unjudged_batch_is_offered_again_rather_than_burned(tmp_path):
    """The defect this replaced: a batch is recorded as drawn *before* it is judged, and the
    next draw used to exclude every logged row. Walk away mid-batch and those clips became
    unreachable for the life of the run — never judged, never offered again, the sample quietly
    smaller than the pool could give, and nothing failing anywhere.

    Offering them back is also what makes leaving a batch free, which is what lets the surface
    switch behavior mid-review without asking whether you meant to lose 18 clips.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    first = run.audit_sample(label="q01", budget=5)
    run.label_audit(first[0].instance_id, True)  # judge one, abandon four

    second = {r.instance_id for r in run.audit_sample(label="q01", budget=5)}
    abandoned = {r.instance_id for r in first[1:]}
    assert abandoned <= second, (
        f"{len(abandoned - second)} unjudged clips were never offered again; they are "
        "unreachable for the life of the run"
    )
    assert first[0].instance_id not in second, "a judged clip was offered again"


def test_a_clip_judged_unclear_is_not_handed_back_as_though_unseen(tmp_path):
    """An unclear **is** a decision, and for a long time the log could not say so: it wrote
    the same empty cell as a clip merely drawn. Nothing failed — every metric drops an unclear
    anyway, because it carries no evidence either way — but this draw offers undecided rows
    back first, so the one clip a reviewer could not call came round again, and again.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    first = run.audit_sample(label="q01", budget=5)
    for row in first:
        run.label_audit(row.instance_id, None if row is first[0] else True)

    second = {r.instance_id for r in run.audit_sample(label="q01", budget=5)}
    assert first[0].instance_id not in second, "the unclear was offered again as unseen"
    assert second, "and the draw still advances"


def test_a_batch_with_nothing_left_pending_draws_new_clips(tmp_path):
    """The other half: resume must not mean "never advance"."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    first = run.audit_sample(label="q01", budget=5)
    for row in first:
        run.label_audit(row.instance_id, True)

    second = {r.instance_id for r in run.audit_sample(label="q01", budget=5)}
    assert second and not (second & {r.instance_id for r in first})


def test_a_partly_judged_batch_is_topped_up_to_the_budget(tmp_path):
    """Resume first, then fill — so a batch is always the size that was asked for."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    first = run.audit_sample(label="q01", budget=5)
    for row in first[:3]:
        run.label_audit(row.instance_id, True)

    second = run.audit_sample(label="q01", budget=5)
    assert len(second) == 5
    assert {r.instance_id for r in first[3:]} <= {r.instance_id for r in second}


def test_auditing_a_finalized_run_is_refused_and_names_the_way_back(tmp_path):
    """Decided rather than inevitable: labelling a clip moves neither the instances, the cutoff
    nor the export. It is refused because precision is a number a reader sees, and a finalized
    run whose precision shifts underneath them is what `_refuse_if_finalized` exists to stop.
    `reopen()` is one call, and `AuthoredRun._record` is the only place that trade lives.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    run.finalize()

    with pytest.raises(ValueError, match="reopen"):
        run.audit_sample(label="q01", budget=5)
    with pytest.raises(ValueError, match="reopen"):
        run.label_audit("whatever", True)

    run.reopen()
    assert run.audit_sample(label="q01", budget=5), "reopening must actually restore it"


def test_the_budget_is_a_batch_not_a_cap(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    assert len(run.audit_sample(label="q01", budget=3)) == 3
    assert len(run.audit_sample(label="q01", budget=7)) == 7


# --------------------------------------------------------------------------- #
# the audit's read half
# --------------------------------------------------------------------------- #
def test_a_script_can_read_the_number_its_judging_bought(tmp_path):
    """The parity gap this closes: `audit_sample` and `label_audit` are the write half, and
    for a while there was no other. A script could spend an afternoon auditing and not
    retrieve the figure that afternoon produced — the number existed only inside the GUI,
    which reached past the API into storage for it."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    for row in run.audit_sample(label="q01", budget=10):
        run.label_audit(row.instance_id, True)

    estimate = run.audit_precision(label="q01", cutoff=0.0)
    assert estimate["n"] == 10
    assert estimate["precision"] == 1.0
    assert estimate["interval"][0] < 1.0 <= estimate["interval"][1]
    assert estimate["coverage"] == 1.0 and estimate["reason"] is None

    by_recording = run.audit_precision_by_recording(label="q01", cutoff=0.0)
    assert by_recording and sum(r["n"] for r in by_recording) == 10

    decisions = run.audit_decisions()
    assert len(decisions) == 10 and all(row.decided for row in decisions)


def test_the_read_half_says_nothing_rather_than_guessing(tmp_path):
    """`None` when nothing has been audited. A caller must say "no reliability estimate", not
    substitute one — the same contract the storage layer keeps."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    assert run.audit_precision(label="q01") is None
    assert run.audit_decisions() == []


def test_the_read_half_survives_the_cutoff_moving(tmp_path):
    """One audit serves every cutoff (§20), and the API half must inherit that rather than
    re-deriving coverage from whatever slice it happens to ask about."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    for row in run.audit_sample(label="q01", budget=10):
        run.label_audit(row.instance_id, True)

    run.set_cutoff("q01", 0.9)
    moved = run.audit_precision(label="q01", cutoff=0.9)
    assert moved["coverage"] == 1.0, "moving the cutoff read as a re-cut vocabulary"
    assert moved["reason"] is None


def test_cli_reports_the_audit_precision(tmp_path):
    """Surface parity for the read half: the GUI is not the only place the number lives."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    for row in run.audit_sample(label="q01", budget=10):
        run.label_audit(row.instance_id, True)

    out = _cli(
        "analysis", "audit",
        "--experiment", str(exp.dir), "--run", "an_01", "--behavior", "q01",
    )  # fmt: skip

    assert "precision 1.00" in out
    assert "n=10" in out, "the figure shipped without the sample size it rests on"
    assert "(" in out and "-" in out, "no interval beside the rate"


def test_cli_says_not_audited_rather_than_printing_a_number(tmp_path):
    exp, _run = _searched(tmp_path)
    out = _cli(
        "analysis", "audit", "--experiment", str(exp.dir), "--run", "an_01",
    )  # fmt: skip
    assert "not audited" in out
    assert "precision" not in out, "a run nobody audited must not report one"


def test_cli_splits_the_estimate_per_recording(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    for row in run.audit_sample(label="q01", budget=10):
        run.label_audit(row.instance_id, True)

    out = _cli(
        "analysis", "audit",
        "--experiment", str(exp.dir), "--run", "an_01", "--by-recording",
    )  # fmt: skip

    recordings = {row.recording_id for row in run.artifact().instances}
    assert any(name in out for name in recordings), "no per-recording rows"


# --- instance_features.csv — where each match happened (§19) ------------------------------


def test_a_run_writes_where_each_instance_happened(tmp_path):
    """Query is a §19 producer and owes the artifact its features, like temporal clustering.

    Position drifts left-to-right across the recording, so a match's mean position is a function
    of *when* it happened — a wrong span or a swapped unit then shows up as a wrong coordinate
    rather than passing quietly.
    """
    from detail.storage import behavior_output as bo

    cx = np.linspace(0.0, 1.0, N_EMBEDDED)
    exp, run = _searched(tmp_path, recordings=("s1",), cx=cx)
    run.set_cutoff("q01", 0.9)

    output = run.artifact()
    assert output.instances, "no matches; the features below would be vacuous"
    assert output.features, "a run with kinematics must write instance features"

    kin = exp.get_recording("s1").kinematics("emb_01", 0)
    for row in bo.bouts(output.instances):
        got = output.features[row.instance_id]
        inside = (kin["t_ns"] >= row.t_ns) & (kin["t_ns"] <= row.end_t_ns)
        assert inside.any()
        assert got[0] == pytest.approx(float(kin["cx_norm"][inside].mean()), abs=1e-6)


def test_a_run_without_kinematics_reports_features_absent_not_empty(tmp_path):
    """Absent and empty are different answers and only one is actionable: no file means the
    run predates §14's kinematics and `recording backfill-kinematics` fills it in.

    Asserted through ``artifact()`` rather than the raw reader, because that is what every
    surface actually holds — the GUI band and `instance_positions` both read this field.
    """
    exp, run = _searched(tmp_path, recordings=("s1",))
    run.set_cutoff("q01", 0.9)

    output = run.artifact()
    assert output.instances
    assert output.features is None


def test_placements_survive_a_behavior_id_that_is_not_a_number(tmp_path):
    """`instance_placements` bucketed by `int(row.source_label)` behind an
    `except ValueError: continue`, commented *"not a cluster id — another producer's artifact"*.

    Query's own labels are `q01`, `q02` (`storage/query_runs.py::behavior_id`), so that `int`
    raised on every row of every query run and the method returned `{}` — silently, and with
    `instance_positions` derived from it. The comment named the one case it was *not*: these
    rows are this producer's. A key that cannot hold a producer's labels is a hole, not a
    filter.

    Nothing failed, because `{}` and "no positions here" render identically.
    """
    import numpy as np

    exp, run = _searched(
        tmp_path, recordings=("s1",), cx=np.linspace(0.1, 0.9, N_EMBEDDED)
    )
    run.set_cutoff("q01", 0.0)

    placements = run.instance_placements()

    assert placements, "every row was dropped, which is what the int() coercion did"
    assert (
        "q01" in placements
    ), f"keyed by {sorted(placements)}, not the behavior's own id"
    rows = placements["q01"]["s1"]
    assert rows, "the behavior is present with no instances under it"
    first = rows[0]
    assert set(first) >= {"instance_id", "cx", "cy", "subject", "t_ns", "end_t_ns"}
    # Derived from the same call, so it inherits the fix rather than needing its own.
    assert run.instance_positions()["q01"]["s1"]


# --------------------------------------------------------------------------- #
# the user's own clips are never audited (§19)
# --------------------------------------------------------------------------- #
# A behavior's example and non-example clips are what the user told the method. A candidate on
# top of one is the method finding its own input, and it is near-certain to be judged a yes, so
# drawing it scores the answer key and nothing about the figure looks wrong. §19 always said such
# an instance is ineligible; the draw took no exclusions and the estimate counted every judgment.
def test_an_example_clip_is_never_drawn_and_never_counted(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    pool = run._audit_pool("q01")
    examples = run._audit_examples(pool)
    assert (
        examples
    ), "no candidate overlaps the fixture's example clip; this guards nothing"

    drawn = run.audit_sample(label="q01", budget=len(pool))
    assert (
        not {row.instance_id for row in drawn} & examples
    ), "an example clip was drawn"

    for row in drawn:
        run.label_audit(row.instance_id, True)
    # A judgment of an example clip, as an audit made before the exclusion could hold.
    for instance_id in examples:
        run.label_audit(instance_id, False)

    figure = run.audit_precision(label="q01", cutoff=0.0)
    assert figure["n"] == len(behavior_output.bouts(drawn))
    assert figure["precision"] == pytest.approx(
        1.0
    ), "a judgment of the user's own clip was counted"
    split = run.audit_precision_by_recording(label="q01", cutoff=0.0)
    assert sum(row["n"] for row in split) == figure["n"]


def test_an_example_clip_stays_in_the_artifact_marked_is_example(tmp_path):
    """Excluded from the audit, not from the run: the instance is still a match the user gets,
    and the artifact and its export say which ones their own clips overlap."""
    import csv

    from detail.api import export_targets
    from detail.api.exporters import ExportScope, ExportSubject

    exp, run = _searched(
        tmp_path, recordings=("s1",), cx=np.linspace(0.1, 0.9, N_EMBEDDED)
    )
    run.set_cutoff("q01", 0.0)
    output = run.artifact()
    examples = run._audit_examples(output.instances)
    assert examples, "no instance overlaps the example clip; this guards nothing"
    assert output.example_ids == examples
    assert examples <= {row.instance_id for row in output.instances}

    out = tmp_path / "export"
    out.mkdir()
    subject = ExportSubject(kind="analysis_run", experiment_dir=exp.dir, id="an_01")
    export_targets._run_instances(subject, out, ExportScope(), None)
    with open(out / "instances.csv", newline="", encoding="utf-8-sig") as handle:
        flagged = {
            row["instance_id"]
            for row in csv.DictReader(handle)
            if row["is_example"] == "true"
        }
    assert flagged == examples


def test_a_draw_records_the_bins_it_was_drawn_in(tmp_path):
    """The draw, every estimate and the confidence curve read one set of edges, recorded at the
    first draw, so a judgment is weighted by the bin it was drawn from."""
    import numpy as np

    from detail.storage import behavior_audit as audit

    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    assert audit.read_bins(run._run_dir()) == {}
    run.audit_sample(label="q01", budget=5)
    stored = audit.read_bins(run._run_dir())
    assert set(stored) == {"q01"}
    scores = [r.score for r in behavior_output.bouts(run._audit_pool("q01"))]
    assert np.array_equal(stored["q01"], audit.bin_edges(scores))
    curve = run.audit_precision_by_score_bin(label="q01")
    assert [b["score_low"] for b in curve] == [float(e) for e in stored["q01"][:-1]]


def test_a_cutoff_below_the_committed_one_is_estimated_over_its_own_instances(tmp_path):
    """The estimate is over the candidates the asked-for cutoff keeps, not the artifact, which
    holds only what the committed cutoff kept — so asking about a lower cutoff counts the
    judgments below the committed one instead of silently dropping them."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    for row in run.audit_sample(label="q01", budget=20):
        run.label_audit(row.instance_id, True)
    everywhere = run.audit_precision(label="q01", cutoff=0.0)

    run.set_cutoff("q01", 0.99)
    lower = run.audit_precision(label="q01", cutoff=0.0)
    assert lower["n"] == everywhere["n"] == 20
    assert lower["cutoff"] == 0.0


# --- data the search read, deleted since (#33 finding 21) ----------------------------------
def _audited(tmp_path):
    exp, run = _searched(tmp_path, n_rows=600)
    run.set_cutoff("q01", 0.0)
    for row in run.audit_sample(label="q01", budget=3):
        run.label_audit(row.instance_id, True)
    return exp


def _audit(exp):
    return CliRunner().invoke(
        main, ["analysis", "audit", "--experiment", str(exp.dir), "--run", "an_01"]
    )


def test_audit_after_a_searched_recording_was_deleted_names_it(tmp_path):
    exp = _audited(tmp_path)
    exp.delete_recording(exp.resolve_recording("s1").id)
    result = _audit(exp)
    assert result.exit_code == 1 and "Traceback" not in result.output
    assert result.output == (
        "Error: query run 'an_01' cannot read its candidates: the recording 's1' it "
        "searched was deleted. Run the query again over the recordings that are left\n"
    )


def test_audit_after_a_searched_embedding_was_deleted_names_it(tmp_path):
    import shutil

    exp = _audited(tmp_path)
    shutil.rmtree(exp.get_recording("s2").dir / "embeddings" / "emb_01")
    result = _audit(exp)
    assert result.exit_code == 1 and "Traceback" not in result.output
    assert (
        "embedding run 'emb_01' of recording 's2', which it searched, was deleted"
        in (result.output)
    )
