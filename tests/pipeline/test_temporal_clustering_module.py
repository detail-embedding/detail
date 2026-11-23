"""The §22 temporal-clustering module: what it refuses, what it leaves behind, and what it projects.

The classes of defect this guards, none of which raises where it is written:

- **An authored run that closes itself.** §21 says the compute never writes a terminal status —
  the user closes the run, not the fit. A runner that let the executor mark it ``complete`` gives
  a vocabulary nobody has interpreted, and every edit then refuses with a message about reopening.
  That shipped once in the pose module and only running the CLI caught it.
- **A default resolved from mutable state.** An embedding run picked because it sorted first makes
  the same command produce a different vocabulary the day another run is added (Tenet 9).
- **A subsampled fit that looks complete.** The DTW comparison is O(bouts²) and ``max_segments``
  is the only bound on it, so the run has to be able to say what it left out.
- **Coverage read as a bug.** Bouts do not tile a timeline, so a temporal run legitimately covers
  less than all of its observed time. A "fix" that made coverage total would be the defect.
"""

import numpy as np
import pytest

from detail.pipeline import temporal_clustering
from detail.storage import analysis_runs as analysis_store
from detail.storage import recording as recording_store
from detail.storage import temporal_runs
from tests.pipeline.phase_messages import assert_messages_name_their_phases

STEP = 17_000_000  # ~59 rows/s, the embed cadence this project's runs actually have

#: Frames the video runs on past the last embedded row. The frame view's axis is the dense
#: video timeline, so without a tail every frame is observed and a mask that came back
#: all-True would prove nothing.
VIDEO_TAIL = 60


def _module():
    return temporal_clustering.TemporalClusteringModule()


def _method_params(**overrides) -> dict:
    """Segmentation settings that actually cut a short synthetic trajectory at this cadence.

    The shipped defaults are tuned for a real recording, where a 0.27 s minimum bout is a few
    rows; on a 240-row fixture it is most of the trajectory. Stated rather than inherited so the
    bouts the test reasons about are the ones it asked for.
    """
    base = {
        "window_s": 0.05,
        "min_len_s": 0.1,
        "band_s": 0.05,
        "penalty": 1.0,
        "min_size_s": 0.05,
    }
    base.update(overrides)
    return base


def _bouty_trajectory(n_bouts=12, bout_rows=30, dim=8, kinds=3, seed=0):
    """A trajectory made of repeating bout kinds, so segmentation has transitions to find."""
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(kinds, 2, dim))
    rows = []
    for i in range(n_bouts):
        ramp = np.linspace(0.0, 1.0, bout_rows)[:, None]
        path = anchors[i % kinds, 0] * (1 - ramp) + anchors[i % kinds, 1] * ramp
        rows.append(path + 0.05 * rng.normal(size=path.shape))
    return np.vstack(rows).astype(np.float32)


# --- validation ---------------------------------------------------------------------------


def test_an_empty_selection_is_refused_with_the_reason(experiment_dir):
    with pytest.raises(ValueError, match="at least one recording"):
        _module().validate(experiment_dir, {"source_recording_ids": []})


def test_an_unknown_method_names_the_ones_there_are():
    with pytest.raises(ValueError, match="unknown temporal-clustering method"):
        temporal_clustering.resolve_method("no_such_method")


def test_the_pose_method_cannot_be_selected_here():
    """§22's whole reason for two registries: a bout method must not be selectable where a pose
    method belongs, and vice versa. Sharing one registry would make this a runtime surprise.
    """
    with pytest.raises(ValueError, match="unknown temporal-clustering method"):
        temporal_clustering.resolve_method("finch")


def test_a_parameter_the_method_does_not_have_is_refused_at_launch(experiment_dir):
    """Pydantic ignores an unknown field, so without this a knob meant for the other method — or a
    typo — would do nothing and the run would look as if it had honoured it."""
    with pytest.raises(
        ValueError, match=r"band_s is not a parameter.*codebook_viterbi"
    ):
        _module().validate(
            experiment_dir,
            {
                "source_recording_ids": ["recording_001"],
                "method": "codebook_viterbi",
                "method_params": {"band_s": 0.5},
            },
        )


def test_the_removed_fit_rows_is_refused_with_its_replacements(experiment_dir):
    """`fit_rows` did two jobs and became two settings (#573). A script still passing it would
    otherwise get the default sample and the default codeword count, silently."""
    with pytest.raises(ValueError, match=r"fit_rows.*fit_share.*codewords"):
        _module().validate(
            experiment_dir,
            {
                "source_recording_ids": ["recording_001"],
                "method": "codebook_viterbi",
                "method_params": {"fit_rows": 3000},
            },
        )


def test_a_run_recorded_with_fit_rows_still_reads_back():
    """The refusal is at launch only. A run fitted before #573 carries ``fit_rows`` in its stored
    params, and every read of it validates those params through the same model."""
    from detail.configs.temporal_codebook import TemporalCodebookConfig

    model = TemporalCodebookConfig.model_validate({"fit_rows": 3000, "min_run_s": 0.5})
    assert model.min_run_s == 0.5


def test_a_method_whose_extra_is_missing_is_refused_with_the_install_command(
    experiment_dir, monkeypatch
):
    """codebook_viterbi builds its codebook with k-means. A missing extra fails at validate,
    naming it, not as an ImportError in a worker after every trajectory has loaded."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "sklearn" else real(name, *a, **k),
    )
    with pytest.raises(ValueError, match=r"sklearn.*analysis"):
        _module().validate(
            experiment_dir,
            {"source_recording_ids": ["recording_001"], "method": "codebook_viterbi"},
        )


def test_the_cli_refuses_fit_rows_and_names_both_replacements(experiment_dir):
    from click.testing import CliRunner

    from detail.cli import main

    result = CliRunner().invoke(
        main,
        ["analysis", "temporal-clustering", "--experiment", str(experiment_dir),
         "--recording", "recording_001", "--method", "codebook_viterbi",
         "--fit-rows", "3000"],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "--fit-share" in result.output and "--codewords" in result.output


def test_the_job_carries_every_field_by_name():
    """A hand-listed params dict is complete until a field is added and nothing fails when it
    stops being — the defect `tests/configs/test_record_completeness.py` exists for."""
    job = _module().build_job(
        "an_01",
        {
            "source_recording_ids": ["rec_a"],
            "embed_run_id": "emb_01",
            "track_id": 2,
            "method": "dtw_finch",
            "method_params": {"band_s": 0.5},
        },
    )
    assert job.source_recording_ids == ["rec_a"]
    assert job.embed_run_id == "emb_01"
    assert job.track_id == 2
    assert job.method == "dtw_finch"
    assert job.method_params == {"band_s": 0.5}
    assert temporal_clustering.TemporalClusteringJob.from_dict(job.to_dict()) == job


# --- the authored state ---------------------------------------------------------------------


def test_the_state_round_trips_through_yaml(tmp_path):
    state = temporal_runs.TemporalClusteringState(
        embed_run_id="emb_01",
        track_id=1,
        method="dtw_finch",
        method_params={"band_s": 0.27},
        source_recording_ids=["rec_a", "rec_b"],
        granularity=30,
        labels={"L1": "rearing"},
        node_labels={"40": "L1", "41": "L1"},
        resolved_names={"L1": "rearing"},
    )
    temporal_runs.write_temporal_state(tmp_path, state)
    assert temporal_runs.read_temporal_state(tmp_path) == state


def test_an_unset_granularity_is_not_zero(tmp_path):
    """A fitted-but-uninterpreted run and a run set to zero clusters are different facts, and a
    dataclass that stored both as 0 could not tell them apart."""
    state = temporal_runs.TemporalClusteringState(
        embed_run_id="emb_01", track_id=0, method="dtw_finch"
    )
    temporal_runs.write_temporal_state(tmp_path, state)
    assert temporal_runs.read_temporal_state(tmp_path).granularity is None


def test_names_are_keyed_by_str_whichever_way_they_arrive(tmp_path):
    """A cluster id is an int in the method and a string in the CSV; YAML would round-trip them
    differently on the two paths without the coercion."""
    state = temporal_runs.TemporalClusteringState(
        embed_run_id="emb_01", track_id=0, method="dtw_finch", granularity=12
    )
    state.names[0] = "rearing"  # an int, the way a method hands one over
    temporal_runs.write_temporal_state(tmp_path, state)
    back = temporal_runs.read_temporal_state(tmp_path)
    assert back.names == {"0": "rearing"}
    assert back.name_of(0) == "rearing"
    assert back.named_count() == 1


# --- the manifest ---------------------------------------------------------------------------


def test_the_manifest_declares_exclusive_overlap_without_claiming_full_coverage():
    """Bouts are disjoint, so no two instances cover the same moment — but they do not tile, so
    exclusivity here does *not* mean the run accounts for all of its observed time. Both halves
    matter: §23's transition measure needs the first, and a reader must not infer the second.
    """
    state = temporal_runs.TemporalClusteringState(
        embed_run_id="emb_01",
        track_id=0,
        method="dtw_finch",
        granularity=4,
        labels={"L1": "rearing"},
        node_labels={"9": "L1"},
        resolved_names={"L1": "rearing"},
    )
    manifest = temporal_clustering._manifest(
        "an_01",
        {"provenance": {}},
        state,
        temporal_clustering.resolve_method("dtw_finch"),
    )
    assert manifest["overlap"] == "exclusive"
    assert manifest["origin"] == "predicted"
    assert manifest["label_space"] == "pooled"
    assert manifest["vocabulary"] == ["rearing"]
    # The unlabelled bucket gets an entry with no name: what `unnamed_time` is offered on (#24).
    assert manifest["names"] == {"L1": ["rearing"], "unlabelled": []}
    assert "score" not in manifest
    assert manifest["method"]["granularity"] == 4


def test_a_run_whose_fit_never_finished_says_so(tmp_path):
    method = temporal_clustering.resolve_method("dtw_finch")
    with pytest.raises(FileNotFoundError, match="no stored fit"):
        temporal_clustering._load_fit(tmp_path, method)


# --- a real fit through the module ----------------------------------------------------------


def _branches(handle) -> tuple[int, int]:
    """The root's two children: real nodes with members, the natural first labels (§22)."""
    tree = handle.hierarchy()
    a, b = tree.children[tree.n_nodes - 1 - tree.n_leaves]
    return int(a), int(b)


def _fitted_run(experiment_dir, recording_dir, make_embedding_run, **method_overrides):
    """One real fit, through the module, on a synthetic embedding run."""
    from detail.api.experiment import Experiment

    features = _bouty_trajectory()
    n = len(features)
    frame_idx = np.arange(n, dtype=np.int64)
    t_ns = frame_idx * STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.arange(n + VIDEO_TAIL, dtype=np.int64) * STEP,
    )
    make_embedding_run(recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01")

    proj = Experiment(experiment_dir)
    run = proj.run_analysis(
        "temporal_clustering",
        {
            "source_recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "method": "dtw_finch",
            "method_params": _method_params(**method_overrides),
        },
    )
    run.wait()
    return proj, run


def test_the_fit_leaves_the_run_open_rather_than_complete(
    experiment_dir, recording_dir, make_embedding_run
):
    """§21: an authored run's compute never writes a terminal status.

    The bug this exists for shipped once in the pose module and was caught only by running the
    CLI: the executor marked the run ``complete`` when the fit finished, so every interpretation
    refused with a message about reopening a run nobody had closed.
    """
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    run_dir = analysis_store.analysis_dir(proj.dir, run.id)
    assert analysis_store.read_analysis_yaml(run_dir)["status"] == "open"


def test_the_fit_records_what_it_was_launched_with(
    experiment_dir, recording_dir, make_embedding_run
):
    """Tenet 9, end to end: the `analysis.yaml` on disk carries the run's own settings.

    `tests/configs/test_record_completeness.py` asserts the *mechanism* by reading source, which
    is what catches a module that never writes params. This asserts the *file*, because the two
    can come apart: an assignment placed after an early return, or on a record that is then
    overwritten, satisfies a source scan and leaves the run on disk exactly as bare as before.
    Every real behavior-clustering run made before this fix has no params at all.
    """
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    run_dir = analysis_store.analysis_dir(proj.dir, run.id)
    params = analysis_store.read_analysis_yaml(run_dir).get("params")

    assert params, "the run recorded nothing it was launched with"
    assert params["method"], "the method that produced this vocabulary"
    assert "method_params" in params, "the method's own knobs"
    assert "track_id" in params
    # Job bookkeeping is not configuration and must not reach the record as if it were.
    assert "job_id" not in params


def test_an_open_run_already_has_an_artifact(
    experiment_dir, recording_dir, make_embedding_run
):
    """§21: the statistics are what the user is deciding against. With nothing labelled the
    artifact carries real coverage, an empty vocabulary and every bout as unlabelled (§22).
    """
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    artifact = handle.artifact()
    assert artifact.coverage, "an analyzed unit must have a coverage row"
    assert artifact.instances
    assert {i.source_label for i in artifact.instances} == {"unlabelled"}
    assert artifact.manifest["vocabulary"] == []


def test_labelling_one_branch_leaves_the_rest_unlabelled(
    experiment_dir, recording_dir, make_embedding_run
):
    """Unlabelled bouts stay in the output as one bucket; a labelled branch gets its label."""
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    left, _right = _branches(handle)
    handle.name_nodes({left: "rearing"})
    labels = {i.source_label: i.label for i in handle.artifact().instances}
    assert labels.get("L1") == "rearing"
    assert labels.get("unlabelled") == ""


def test_an_instance_is_a_bout_with_a_real_extent(
    experiment_dir, recording_dir, make_embedding_run
):
    """The difference from the pose module, asserted rather than assumed. A pose instance is a run
    of frames the method encoded; a temporal instance arrives as a span, so every row must have a
    genuine duration rather than one sampling interval."""
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    instances = handle.instances()
    assert instances
    assert all(i.end_t_ns > i.t_ns for i in instances)
    # Every instance is at least the minimum bout length the method was given.
    shortest = min((i.end_t_ns - i.t_ns) for i in instances)
    assert shortest >= 0.05 * 1e9, shortest
    assert all(i.end_frame >= i.start_frame for i in instances)


def test_coverage_is_the_observed_time_not_the_bout_time(
    experiment_dir, recording_dir, make_embedding_run
):
    """Bouts do not tile a timeline, so a temporal run's instances cover less than it observed.

    Asserted because the tempting "fix" is to make coverage sum the instances, which would report
    100% for every run and destroy the one number that says how much of a recording the
    vocabulary actually accounts for.
    """
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    artifact = handle.artifact()
    observed = sum(row.observed_ns for row in artifact.coverage)
    in_bouts = sum(i.end_t_ns - i.t_ns for i in artifact.instances)
    assert observed > 0
    assert in_bouts <= observed


def test_the_frame_view_denominator_agrees_with_coverage(
    experiment_dir, recording_dir, make_embedding_run
):
    """The invariant that makes a windowed denominator trustworthy, on this producer.

    `coverage.observed_ns` says how much time was observed; the frame view says *where*. If they
    disagree, every sub-window rate is wrong in proportion to where the gaps fall, and nothing
    raises -- the numbers stay plausible.
    """
    from detail.storage import frame_view

    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    run_dir = analysis_store.analysis_dir(proj.dir, run.id)

    units = frame_view.discover_frame_views(run_dir)
    assert (
        units
    ), "temporal clustering declares the frames artifact, so it must write one"
    for row in handle.artifact().coverage:
        view = frame_view.read_frame_view(run_dir, row.recording_id, row.subject)
        assert frame_view.observed_ns_in(view) == row.observed_ns

    observed = frame_view.read_frame_view(run_dir, *units[0]).observed[:]
    assert not observed[
        -1
    ], "the video's tail was never embedded, so it is not observed"
    assert observed.any(), "the analyzed span is observed"


def test_the_readers_bring_deferred_frame_views_current_themselves(
    experiment_dir, recording_dir, make_embedding_run
):
    """An edit defers the views (#325); every surface that reads them must not notice.

    Through the real entry points, a windowed measure and Finish, because the guarantee is that
    no user path can reach ``FrameViewsPendingError``. Each reports the rebuild through
    ``progress``, since it is the one long step either of them has.
    """
    from detail.configs.windows import WindowSpec
    from detail.storage import frame_view

    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    run_dir = analysis_store.analysis_dir(proj.dir, run.id)
    left, right = _branches(handle)

    handle.name_nodes({left: "rearing"})
    assert frame_view.is_pending(
        run_dir
    ), "a label moves bouts between labels; it defers"
    stages = []
    rows = proj.measures(run.id).table(
        windows=WindowSpec(kind="normalized", parts=2),
        measures=["rate"],
        progress=lambda stage, fraction, detail: stages.append(stage),
    )
    assert rows and not frame_view.is_pending(run_dir)
    assert "writing frame views" in stages, "the rebuild ran silently"

    handle.name_nodes({right: "grooming"})
    assert frame_view.is_pending(run_dir)
    stages.clear()
    handle.finalize(progress=lambda stage, fraction, detail: stages.append(stage))
    assert not frame_view.is_pending(run_dir)
    assert "writing frame views" in stages
    pinned = handle.snapshots()[-1]["snapshot_id"]
    snapshot = snapshots_store_dir(run_dir, pinned)
    assert frame_view.discover_frame_views(snapshot), "the snapshot froze current views"


def snapshots_store_dir(run_dir, snapshot_id):
    from detail.storage import snapshots

    return snapshots.snapshot_dir(run_dir, snapshot_id)


def test_renaming_a_label_rewrites_no_frame_view(
    experiment_dir, recording_dir, make_embedding_run
):
    """#325 through the real edit path: renaming a labelled node changes no bout and no
    source label, so no view moves, and the artifact it leaves is the one a full rebuild writes.
    """
    import filecmp
    import os

    from detail.storage import frame_view

    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    left, _right = _branches(handle)
    handle.name_nodes({left: "rearing"})
    assert (
        handle.bring_frame_views_current()
    ), "a first label moves bouts between labels"
    run_dir = analysis_store.analysis_dir(proj.dir, run.id)
    units = frame_view.discover_frame_views(run_dir)
    assert units, "no frame views; the assertion below would be vacuous"

    def identities():
        return {u: os.stat(frame_view.unit_path(run_dir, *u)).st_ino for u in units}

    before = identities()
    handle.name_nodes({left: "supported rearing"})
    assert (
        identities() == before
    ), "a rename rewrote a frame view whose bouts did not change"
    assert not frame_view.is_pending(
        run_dir
    ), "and left nothing for a reader to rebuild"
    assert "supported rearing" in {i.label for i in handle.artifact().instances}

    named = run_dir / "named.csv"
    named.write_bytes((run_dir / "instances.csv").read_bytes())

    def arrays_of(u):
        root = frame_view.read_frame_view(run_dir, *u).root
        return {k: root[k][:] for k in root.array_keys()}

    arrays = {u: arrays_of(u) for u in units}
    handle.reproject()
    assert filecmp.cmp(named, run_dir / "instances.csv", shallow=False)
    for u in units:
        rebuilt = arrays_of(u)
        assert sorted(rebuilt) == sorted(arrays[u])
        for k, kept in arrays[u].items():
            assert np.array_equal(rebuilt[k], kept), (u, k)


def test_several_nodes_take_one_name(experiment_dir, recording_dir, make_embedding_run):
    """§22: the many-to-one map is the normal case."""
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    left, right = _branches(handle)
    handle.name_nodes({left: "rearing", right: "rearing"})
    artifact = handle.artifact()
    assert artifact.manifest["vocabulary"] == ["rearing"]
    assert {i.label for i in artifact.instances} == {"rearing"}
    assert {i.source_label for i in artifact.instances} - {"unlabelled"} == {"L1"}


def test_a_finalized_run_refuses_an_edit_and_names_the_way_back(
    experiment_dir, recording_dir, make_embedding_run
):
    """§21: finish is a statement, not a lock, and the refusal has to say how to undo it."""
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    first, second = _branches(handle)
    # A run finishes only once a cluster is labelled (`AuthoredRun.finish_blocker`).
    handle.name_nodes({first: "rearing"})
    handle.finalize()
    with pytest.raises(ValueError, match="reopen"):
        handle.name_nodes({second: "grooming"})
    handle.reopen()
    handle.name_nodes({second: "grooming"})
    assert handle.state().node_names[str(second)] == "grooming"


# --- the subsample, stated -------------------------------------------------------------------


def test_a_full_fit_reports_no_subsample(
    experiment_dir, recording_dir, make_embedding_run
):
    """The negative case, so the guard below cannot pass vacuously."""
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    assert proj.temporal_clustering_run(run.id).subsample_note() is None


def test_a_capped_fit_states_it_on_the_run(
    experiment_dir, recording_dir, make_embedding_run
):
    """§22 requires the surface to say so, and the surface asks the run. A fit whose vocabulary
    came from part of the data is identical to a full one in every other respect."""
    proj, run = _fitted_run(
        experiment_dir, recording_dir, make_embedding_run, max_segments=4
    )
    note = proj.temporal_clustering_run(run.id).subsample_note()
    assert note is not None
    assert "4" in note
    assert "labeled" in note, "the note must say the rest are still labeled"


def test_a_capped_run_still_produces_an_instance_per_bout(
    experiment_dir, recording_dir, make_embedding_run
):
    """The whole point, at the artifact rather than at the method.

    The cap bounds the O(bouts²) comparison. If it also bounded the output, a discovery run would
    quietly return a fraction of the behavior it found — and the run would look completely normal:
    a well-formed artifact, a sensible vocabulary, coverage that reads plausibly.

    One fit, not two. ``make_embedding_run`` appends to the recording's store, so fitting twice in
    one experiment doubles the trajectory and produces overlapping bouts — the capped-versus-
    uncapped comparison lives in `tests/cluster_methods/test_dtw_finch.py`, where the method is
    called directly and no store is involved.
    """
    proj, run = _fitted_run(
        experiment_dir, recording_dir, make_embedding_run, max_segments=4
    )
    handle = proj.temporal_clustering_run(run.id)

    counts = handle.artifact().manifest["method"]
    total = counts["n_fit_bouts"] + counts["n_assigned_bouts"]
    assert counts["n_fit_bouts"] == 4, "the cap must still bound what is compared"
    assert counts["n_assigned_bouts"] > 0, "nothing was held out; this proves nothing"

    assert len(handle.instances()) == total
    # ...and the node cache counts every bout, not only what was compared.
    view = handle.node_view()
    assert view.summary(view.root).members == total


def test_the_manifest_separates_learned_from_assigned(
    experiment_dir, recording_dir, make_embedding_run
):
    """An assigned bout was matched to a codebook it had no part in building, which is a weaker
    claim than a clustered one's. A reader who cannot tell them apart cannot weigh that.
    """
    proj, run = _fitted_run(
        experiment_dir, recording_dir, make_embedding_run, max_segments=4
    )
    handle = proj.temporal_clustering_run(run.id)

    method = handle.artifact().manifest["method"]
    assert method["n_fit_bouts"] == 4
    assert method["n_assigned_bouts"] > 0
    assert method["n_fit_bouts"] + method["n_assigned_bouts"] == len(handle.instances())


# --- progress ---------------------------------------------------------------------------------


def test_the_runner_streams_distinct_phases(
    experiment_dir, recording_dir, make_embedding_run, monkeypatch
):
    """The real thing: run a fit and assert the messages advance rather than sitting on one.

    Modelled on a phase test of the removed syllable-discovery module.
    Collapse the phase emits back to one and this fails.
    """
    import detail.core.progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)

    features = _bouty_trajectory()
    n = len(features)
    frame_idx = np.arange(n, dtype=np.int64)
    t_ns = frame_idx * STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01")

    events: list = []
    module = _module()
    config = module.validate(
        experiment_dir,
        {
            "source_recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "method": "dtw_finch",
            "method_params": _method_params(),
        },
    )
    job = module.build_job("an_01", config)
    run_dir = analysis_store.analysis_dir(experiment_dir, "an_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        run_dir,
        {
            "id": "an_01",
            "module_id": "temporal_clustering",
            "recording_ids": ["recording_001"],
            "status": "running",
            "params": {},
        },
    )
    module.runner(job, run_dir, events.append, None)

    messages = [e.message for e in events]
    # A fit reports how long it has left (it never did: see the pose module's test).
    assert any(e.eta_s is not None for e in events), "no event ever carried an ETA"
    # Derived from the module's own declaration (§16, issue #393) rather than hand-typed here a
    # second time — a phase declared but never actually emitted fails right here.
    assert_messages_name_their_phases(messages, module.phases)
    seen = [m.split(" · ")[0] for m in messages]
    assert seen[0] == module.phases[0]
    assert len(set(seen)) >= len(
        module.phases
    ), f"only {set(seen)} — the label barely moved"

    # The inner pair moves within a phase, on a fixture with exactly one recording — the case a
    # `0 < recordings_done < recordings_total` assertion cannot cover.
    inner = [(e.frames_done, e.frames_total) for e in events]
    assert any(0 < done < total for done, total in inner), inner
    assert any(e.eta_s is not None for e in events)


def tmp_path_of(handle):
    """A scratch directory beside the run, so an export test needs no extra fixture."""
    return handle._run_dir().parent.parent


def test_export_writes_one_stream_csv_per_recording_carrying_origin(
    experiment_dir, recording_dir, make_embedding_run
):
    """§19's deliberate re-ingest. `origin: predicted` is what stops a prediction being read back
    as ground truth once it is a behavior stream.

    Neither temporal-clustering export had any test at all before snapshots were added — the paths were
    modified-but-unverified, which is the shape that lets a silent regression through.
    """
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    # A behavior-stream CSV needs a label, so an export before any labelling is header-only.
    handle.name_nodes({handle.hierarchy().n_nodes - 1: "rearing"})

    out = handle.export(tmp_path_of(handle) / "export")
    names = sorted(p.name for p in out.iterdir())
    assert names == [f"{r}.csv" for r in sorted(handle.state().source_recording_ids)]
    rows = [
        line
        for name in names
        for line in (out / name).read_text(encoding="utf-8-sig").splitlines()[1:]
    ]
    assert rows, "a named vocabulary must reach the CSV"
    assert all(
        (out / name).read_text(encoding="utf-8-sig").splitlines()[0]
        == "start_frame,end_frame,label,subject,origin"
        for name in names
    )
    assert all(line.endswith(",predicted") for line in rows)


def test_export_pins_what_it_wrote(experiment_dir, recording_dir, make_embedding_run):
    """An exported CSV must correspond to something still on disk (§21)."""
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    left, right = _branches(handle)
    handle.name_nodes({left: "rearing"})
    handle.export(tmp_path_of(handle) / "export")
    pinned = handle.snapshots()
    assert len(pinned) == 1 and pinned[0]["reason"] == "export"
    frozen = handle.snapshot_artifact(pinned[0]["snapshot_id"])
    handle.name_nodes({right: "grooming"})
    again = handle.snapshot_artifact(pinned[0]["snapshot_id"])
    assert [i.source_label for i in again.instances] == [
        i.source_label for i in frozen.instances
    ]


def test_exporting_twice_without_an_edit_pins_once(
    experiment_dir, recording_dir, make_embedding_run
):
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    handle.name_nodes({_branches(handle)[0]: "rearing"})
    handle.export(tmp_path_of(handle) / "a")
    handle.export(tmp_path_of(handle) / "b")
    assert len(handle.snapshots()) == 1


def test_a_second_export_into_one_folder_keeps_the_first(
    experiment_dir, recording_dir, make_embedding_run
):
    """Each export makes its own folder (#29): two into one ``--out`` are two folders."""
    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)
    handle.name_nodes({_branches(handle)[0]: "rearing"})
    parent = tmp_path_of(handle) / "export"
    first = handle.export(parent)
    second = handle.export(parent)
    assert first != second and first.parent == second.parent == parent
    assert sorted(p.name for p in first.iterdir()) == sorted(
        p.name for p in second.iterdir()
    )


def test_cli_exports_the_artifact_and_names_the_pin(
    experiment_dir, recording_dir, make_embedding_run
):
    """The command had no test at all before snapshots were added. It exercises the whole path a
    user takes — interpret, then export — and asserts the pinned id is echoed, because a figure
    built from these CSVs needs something citable that is still on disk (§21).
    """
    from click.testing import CliRunner

    from detail.cli import main

    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    handle = proj.temporal_clustering_run(run.id)

    interpret = CliRunner().invoke(
        main,
        ["analysis", "temporal-interpret", "--experiment", str(proj.dir),
         "--run", run.id, f"--name-node={_branches(handle)[0]}=walking"],
    )  # fmt: skip
    assert interpret.exit_code == 0, interpret.output
    # The coverage the edit produced, one line per recording, in this module's noun.
    assert "coverage: recording_001: analysed" in interpret.output, interpret.output
    assert "in a bout" in interpret.output

    # A label is what puts rows in the CSV.
    named = CliRunner().invoke(
        main,
        ["analysis", "temporal-interpret", "--experiment", str(proj.dir), "--run", run.id,
         f"--name-node={handle.hierarchy().n_nodes - 1}=rearing"],
    )  # fmt: skip
    assert named.exit_code == 0, named.output

    out_dir = proj.dir.parent / "exported"
    result = CliRunner().invoke(
        main,
        ["analysis", "export-temporal-clustering", "--experiment", str(proj.dir),
         "--run", run.id, "--out", str(out_dir)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    # A folder of its own inside --out, and the CLI says which (#29).
    (folder,) = out_dir.iterdir()
    assert f"exported to {folder}" in result.output
    written = sorted(folder.iterdir())
    assert written, "an export writes one CSV per analyzed recording"
    # Every file carries the header — a recording that produced nothing gets an empty file rather
    # than no file (§19) — and at least one row somewhere is a real instance marked predicted.
    assert all(
        p.read_text(encoding="utf-8-sig").splitlines()[0]
        == "start_frame,end_frame,label,subject,origin"
        for p in written
    )
    assert any("predicted" in p.read_text(encoding="utf-8-sig") for p in written)
    assert "pinned as snap_" in result.output


# --------------------------------------------------------------------------- #
# instance_features.csv — where each behavior happened (§19)
# --------------------------------------------------------------------------- #
def test_a_fit_writes_where_each_instance_happened(
    experiment_dir, recording_dir, make_embedding_run
):
    """The end of the spatial-map data path, asserted as numbers rather than as "a file exists".

    Position drifts left-to-right across the recording, so an instance's mean position is a
    function of *when* it happened — which makes a wrong span (or a swapped unit) show up as a
    wrong coordinate instead of passing quietly.
    """
    from detail.api.experiment import Experiment
    from detail.storage import behavior_output as bo

    features = _bouty_trajectory()
    n = len(features)
    frame_idx = np.arange(n, dtype=np.int64)
    t_ns = frame_idx * STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.arange(n + VIDEO_TAIL, dtype=np.int64) * STEP,
    )
    cx = np.linspace(0.0, 1.0, n)
    make_embedding_run(
        recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01", cx=cx, cy=cx
    )

    proj = Experiment(experiment_dir)
    run = proj.run_analysis(
        "temporal_clustering",
        {
            "source_recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "method": "dtw_finch",
            "method_params": _method_params(),
        },
    )
    run.wait()
    handle = proj.temporal_clustering_run(run.id)
    # Labels on the root's children, so instances fall at different times and places.
    left, right = _branches(handle)
    handle.name_nodes({left: "a", right: "b"})

    artifact = handle.artifact()
    assert artifact.instances, "no instances; the features below would be vacuous"
    features_by_id = bo.read_instance_features(
        analysis_store.analysis_dir(proj.dir, run.id)
    )
    assert features_by_id, "a run with kinematics must write instance features"

    # Every bout is covered, and each one's position is the mean of its own span.
    kin = proj.get_recording("recording_001").kinematics("emb_01", 0)
    for row in bo.bouts(artifact.instances):
        got = features_by_id[row.instance_id]
        inside = (kin["t_ns"] >= row.t_ns) & (kin["t_ns"] <= row.end_t_ns)
        assert inside.any()
        assert got[0] == pytest.approx(float(kin["cx_norm"][inside].mean()), abs=1e-6)

    # And the numbers actually differ between bouts — a constant would satisfy every
    # assertion above while telling you nothing about where anything happened.
    seen = {round(v[0], 4) for v in features_by_id.values()}
    assert len(seen) > 1, "every bout reported the same position"


def test_a_fit_over_an_embedding_without_kinematics_writes_no_features(
    experiment_dir, recording_dir, make_embedding_run
):
    """Absent, not empty. A run whose embedding predates §14's kinematics has no position to
    report, and that is fixable by backfilling — an empty file would say the opposite.
    """
    from detail.storage import behavior_output as bo

    proj, run = _fitted_run(experiment_dir, recording_dir, make_embedding_run)
    run_dir = analysis_store.analysis_dir(proj.dir, run.id)
    assert not (run_dir / bo.FEATURES_FILENAME).exists()
    # `None`, not `{}`: absent and empty are different answers and only absent is actionable
    # (`recording backfill-kinematics`). Sharing one encoding is what hid the difference from
    # every reader — the recorded class in tests/storage/test_behavior_audit.py.
    assert bo.read_instance_features(run_dir) is None
