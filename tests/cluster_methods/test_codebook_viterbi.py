"""The frame-codebook temporal method (§22) — the other shape a behavior vocabulary can take.

`test_dtw_finch.py` covers the contract both temporal methods share (its registry test is
data-driven, so this method is already in it). What this file covers is what is different, and the
differences are where the bugs would be:

- **A frame is not a bout.** This method never segments; it labels every frame and lets the
  stretches fall out of a decode, so an instance is a *run of frames* and the minimum-run floor is
  applied at the end rather than before the comparison.
- **The decode runs once, at the finest rung, and coarser settings inherit it.** That is a
  deliberate departure from the research protocol, taken so an authored knob stays instant, and it
  is only correct because the codebook's rungs nest. Asserted rather than assumed.
- **The sample and the codeword count are separate settings (#573).** ``fit_share`` samples the
  same share of every recording, so the sample grows with the study; ``codewords`` fixes how many
  states the decode carries whatever the sample's size. One ``fit_rows`` total used to do both,
  and could not be fixed without separating them: a share fed to FINCH made the state count, and
  the decode, grow with the study.
"""

import numpy as np
import pytest

from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.registries import TEMPORAL_CLUSTER_METHODS

STEP = 17_000_000  # ns between rows — the ~59 rows/s cadence the real embed runs at


def _method():
    import_temporal_cluster_methods()
    return TEMPORAL_CLUSTER_METHODS["codebook_viterbi"]()


def _trajectory(kinds=3, hold=120, repeats=2, dim=8, noise=0.15, seed=0):
    """A trajectory that holds one pose for a stretch, then moves to the next, and repeats.

    Structure a codebook can find and a decode can clean up — a fixture of independent noise would
    let every test here pass while the method did nothing.
    """
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(kinds, dim))
    block = np.repeat(anchors, hold, axis=0)
    rows = np.vstack(
        [block + noise * rng.normal(size=block.shape) for _ in range(repeats)]
    )
    t_ns = np.arange(len(rows), dtype=np.int64) * STEP
    return rows.astype(np.float32), t_ns


def _params(**overrides) -> dict:
    base = {
        "window_s": 0.05,
        "min_run_s": 0.1,
        "fit_share": 40.0,
        "codewords": 40,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def fitted():
    method = _method()
    features, t_ns = _trajectory()
    return method, method.fit([("recording_001", 0, features, t_ns)], _params())


def _rung(fit, at_least=3):
    """The planted kind count, a setting inside the fit's range (a cut of the tree, not a rung)."""
    spec = fit.granularity
    assert spec.minimum <= at_least <= spec.maximum, spec
    return at_least


def _settings(fit) -> list[int]:
    """A spread of settings across the knob, the rungs among them.

    Not the rungs alone: a two-step codebook has one, and a nesting check looping over one
    setting compares nothing with nothing and passes.
    """
    spec = fit.granularity
    spread = {spec.minimum, 2, 3, 5, 10, 20, spec.maximum, *spec.rungs}
    settings = sorted(g for g in spread if spec.minimum <= g <= spec.maximum)
    assert len(settings) >= 4, settings
    return settings


# --- an instance is a run of frames ----------------------------------------------------------


def test_instances_are_runs_of_frames_not_bouts(fitted):
    """The difference from `dtw_finch`, where a bout arrives already delimited. Here the extent
    comes out of the labeling, so an instance spans many rows and the rows it spans all carry its
    cluster."""
    method, fit = fitted
    granularity = _rung(fit)
    instances = method.instances_at(fit, granularity, _params())
    labels = fit.row_labels_at(granularity)

    assert instances
    assert any(i.end_row > i.start_row for i in instances), "every instance is one row"
    for instance in instances:
        span = labels[instance.start_row : instance.end_row + 1]
        assert set(span.tolist()) == {instance.cluster_id}


def test_a_run_shorter_than_the_floor_is_not_an_instance(fitted):
    """Without a floor every flicker the decode did not catch becomes a one-frame behavior."""
    method, fit = fitted
    granularity = _rung(fit)
    rows_per_second = 1e9 / STEP

    short = method.instances_at(fit, granularity, _params(min_run_s=0.0))
    long = method.instances_at(fit, granularity, _params(min_run_s=1.0))
    assert len(long) <= len(short)
    floor = int(round(1.0 * rows_per_second))
    assert all(i.end_row - i.start_row + 1 >= floor for i in long)


def test_the_planted_structure_is_found(fitted):
    """Assert the content. A codebook that put every frame in one cluster, or each in its own,
    would satisfy every structural test above."""
    method, fit = fitted
    granularity = _rung(fit)
    assert len(np.unique(fit.labels_at(granularity))) == granularity
    instances = method.instances_at(fit, granularity, _params())
    # The fixture holds each pose for a long stretch and repeats, so the runs should be few and
    # long rather than a spray of short ones.
    assert len(instances) <= 12, len(instances)
    assert max(i.end_row - i.start_row for i in instances) > 50


# --- the decode runs once and is inherited ----------------------------------------------------


def test_a_coarser_setting_inherits_the_decode_through_the_nesting(fitted):
    """The design's load-bearing claim, and the reason a granularity change is instant.

    The decode ran once at the finest rung. A coarser setting maps those labels upward, which is
    only correct because the codebook's rungs nest — so every row's coarse label must equal what
    the codebook says its fine codeword becomes. Get this wrong and rows land in a neighbouring
    cluster with nothing raising.
    """
    method, fit = fitted
    finest = fit.levels[0]
    for granularity in _settings(fit):
        coarse = fit.labels_at(granularity)
        rows = fit.row_labels_at(granularity)
        for fine in np.unique(fit.fine_labels):
            members = np.flatnonzero(finest == int(fine))
            assert len(members)
            expected = int(coarse[members[0]])
            got = rows[fit.fine_labels == int(fine)]
            assert set(got.tolist()) == {expected}, (
                f"fine codeword {fine} maps to {set(got.tolist())} at granularity "
                f"{granularity}, but the codebook puts it in {expected}"
            )


def test_the_codebook_rungs_are_nested(fitted):
    """What the inheritance rests on, checked directly: a coarser partition of the codebook is a
    union of finer ones."""
    method, fit = fitted
    settings = _settings(fit)
    for coarser, finer in zip(settings, settings[1:]):
        coarse = fit.labels_at(coarser)
        fine = fit.labels_at(finer)
        for cluster in np.unique(coarse):
            members = set(np.flatnonzero(coarse == cluster).tolist())
            for f in {int(x) for x in fine[sorted(members)]}:
                assert set(np.flatnonzero(fine == f).tolist()) <= members


def test_changing_granularity_re_labels_without_re_decoding(fitted):
    """§21: an authored knob must not trigger the fit. The stored decode is what makes that true,
    so it must not change when the granularity does."""
    method, fit = fitted
    before = np.array(fit.fine_labels, copy=True)
    for granularity in _settings(fit):
        method.instances_at(fit, granularity, _params())
    assert np.array_equal(fit.fine_labels, before)


# --- ranking a node ----------------------------------------------------------------------------


def _largest_cluster_node(method, fit, granularity):
    """The biggest cluster at ``granularity``, and its node of the tree."""
    from detail.api import authored_run

    labels = fit.row_labels_at(granularity)
    cluster = int(np.bincount(labels).argmax())
    nodes = authored_run.cluster_nodes(
        method.hierarchy_of(fit), fit.labels_at(granularity), granularity
    )
    return cluster, int(nodes[cluster])


def test_ranking_leads_with_the_most_typical_run(fitted):
    """Typicality, not length. Ranking by length would put a long drifting stretch above a short
    unambiguous one, which is the opposite of what someone naming a node wants to see.
    """
    method, fit = fitted
    _cluster, node = _largest_cluster_node(method, fit, _rung(fit))
    order = method.rank_members(fit, method.hierarchy_of(fit).leaves_under(node))
    assert order
    scores = [
        float(np.mean(fit.fine_distance[r.start_row : r.end_row + 1])) for r, _ in order
    ]
    assert scores == sorted(scores), scores
    assert [d for _, d in order] == pytest.approx(scores)


def test_ranked_runs_are_runs_the_user_can_also_see(fitted):
    """The protocol hands params to `instances_for` and not to this, so the floor comes off the
    fit — otherwise a ranked run could be one that is not in the list beside it.
    """
    method, fit = fitted
    granularity = _rung(fit)
    cluster, node = _largest_cluster_node(method, fit, granularity)
    shown = {
        (i.start_row, i.end_row)
        for i in method.instances_at(fit, granularity, {"min_run_s": fit.min_run_s})
        if i.cluster_id == cluster
    }
    order = method.rank_members(fit, method.hierarchy_of(fit).leaves_under(node))
    assert {(r.start_row, r.end_row) for r, _ in order} == shown


def test_a_node_with_no_leaves_ranks_to_nothing(fitted):
    method, fit = fitted
    assert method.rank_members(fit, np.zeros(0, dtype=np.int64)) == []


# --- the cost is bounded by a constant ---------------------------------------------------------


def test_the_codeword_count_is_the_setting_whatever_the_trajectory_length():
    """The decode's cost is quadratic in the state count, so the state count is what must not grow
    with the study. It is ``codewords``, however long the trajectory and however large the sample.

    A FINCH codebook fails this: its finest level is about a quarter of the rows sampled, so a
    share-based sample would carry the state count up with the study."""
    method = _method()
    short_features, short_t = _trajectory(repeats=2)
    long_features, long_t = _trajectory(repeats=8)

    short = method.fit([("r", 0, short_features, short_t)], _params(codewords=30))
    long = method.fit([("r", 0, long_features, long_t)], _params(codewords=30))

    assert len(long) > 2 * len(short), "the fixture did not actually grow"
    assert len(long.sample) > 2 * len(short.sample), "the sample is not a share"
    assert max(short.counts) == max(long.counts) == 30
    # ...and every row of the longer one is still labeled.
    assert len(long.fine_labels) == len(long)


def test_a_recording_is_sampled_as_densely_in_a_larger_study():
    """The thinning #573 was about: a fixed total spread over the study gave each video fewer
    rows as recordings were added, so a brief behavior was found in a small study and missed in a
    large one. A share gives a recording the same rows alone as beside three others."""
    method = _method()
    features, t_ns = _trajectory(repeats=2, seed=1)
    others = [_trajectory(repeats=4, seed=s) for s in (2, 3, 4)]

    alone = method.fit([("r", 0, features, t_ns)], _params(fit_share=10.0))
    study = method.fit(
        [("r", 0, features, t_ns)]
        + [(f"o{i}", 0, f, t) for i, (f, t) in enumerate(others)],
        _params(fit_share=10.0),
    )

    in_study = int((study.recording_index[study.sample_rows] == 0).sum())
    assert len(study.sample) > 3 * len(
        alone.sample
    ), "the fixture did not grow the study"
    assert in_study == len(alone.sample)


def test_the_sample_rows_are_where_the_sample_came_from():
    """The scatter resolves a clicked point to a frame through ``sample_rows``. Two recordings told
    apart by one feature's sign make a wrong row visible: every sample member must carry its own
    recording's sign.

    One recording is tiny, so its two-row minimum sample is out of proportion to the other's —
    the case an evenly spaced guess over the pooled rows (how a shard without ``sample_rows`` is
    read back) gets wrong. At a proportional sample the guess happens to land right, and a test
    built on one would pass either way."""
    method = _method()
    short_features, short_t = _trajectory(kinds=2, hold=15, repeats=1, seed=1)
    long_features, long_t = _trajectory(repeats=8, seed=2)
    short_features[:, 0] = 50.0
    long_features[:, 0] = -50.0

    fit = method.fit(
        [("short", 0, short_features, short_t), ("long", 0, long_features, long_t)],
        _params(fit_share=1.0),
    )

    from_short = fit.recording_index[fit.sample_rows] == 0
    assert from_short.sum() == 2, "the minimum sample did not apply"
    assert ((fit.sample[:, 0] > 0) == from_short).all()


def test_every_frame_is_labeled(fitted):
    """`has_noise = False`, made true: the decode moves labels around but assigns nothing to
    nothing."""
    method, fit = fitted
    assert len(fit.fine_labels) == len(fit)
    assert (fit.fine_labels >= 0).all()


# --- persistence ---------------------------------------------------------------------------------


def test_a_fit_round_trips_through_disk(fitted, tmp_path):
    method, fit = fitted
    path = tmp_path / f"fit{method.shard_suffix}"
    method.serialize(fit, path)
    back = method.load(path)

    granularity = _rung(fit)
    assert np.array_equal(back.fine_labels, fit.fine_labels)
    assert np.array_equal(back.fine_distance, fit.fine_distance)
    assert back.min_run_s == fit.min_run_s
    assert np.array_equal(back.sample_rows, fit.sample_rows)
    assert back.granularity.rungs == fit.granularity.rungs
    assert np.array_equal(
        back.row_labels_at(granularity), fit.row_labels_at(granularity)
    )
    assert len(method.instances_at(back, granularity, _params())) == len(
        method.instances_at(fit, granularity, _params())
    )


# --- refusals -------------------------------------------------------------------------------------


def test_no_trajectories_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="at least one trajectory"):
        _method().fit([], _params())


def test_every_trajectory_empty_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="every selected trajectory was empty"):
        _method().fit(
            [("r", 0, np.zeros((0, 8), dtype=np.float32), np.zeros(0, dtype=np.int64))],
            _params(),
        )


# --- memory scales with a recording, not with the pool -----------------------------------------


def test_pooling_many_recordings_never_scores_more_rows_than_one_block(monkeypatch):
    """The class of bug this guards: scoring every pooled row against the codebook in one dense
    matmul makes peak memory scale with the *sum* of every pooled recording's rows rather than
    with any single one of them — harmless on one recording, an out-of-memory kill on a
    multi-recording study before this fix. No
    continuity block crosses a recording boundary, so chunking the scoring by block means no
    single call should ever see more rows than the biggest recording in the pool.
    """
    import detail.analysis.decode as decode_mod

    seen_rows = []
    original = decode_mod.gaussian_emission

    def spy(features, centroids, sigma_sq):
        seen_rows.append(len(features))
        return original(features, centroids, sigma_sq)

    monkeypatch.setattr(decode_mod, "gaussian_emission", spy)

    recordings = [_trajectory(seed=i) for i in range(5)]
    trajectories = [
        (f"recording_{i:03d}", 0, features, t_ns)
        for i, (features, t_ns) in enumerate(recordings)
    ]
    largest = max(len(features) for features, _ in recordings)

    fit = _method().fit(trajectories, _params())

    assert seen_rows, "gaussian_emission was never called"
    assert max(seen_rows) <= largest, seen_rows
    assert (
        sum(seen_rows) > largest
    ), "the fixture did not actually pool multiple recordings"
    assert len(fit.fine_labels) == len(fit)
    assert len(fit) > largest, "the fit did not actually pool multiple recordings"
    assert (fit.fine_labels >= 0).all()


# --- the decode reports while it runs ----------------------------------------------------------


def test_the_decode_reports_progress_within_its_phase_on_one_recording():
    """The class of bug this guards: a phase that emits only at its boundary.

    The decode is O(rows x states^2) and the state count is the codebook's finest rung, so a
    pooled study spends minutes to hours here. Reporting only at 0.7 and again at 0.9 leaves the
    bar pinned through all of it, which is indistinguishable from the hang it is not (see
    `DESIGN_PLAN.md` section 6). The fixture is **one** recording, so the outer loop is cut to one
    and anything that moves the fraction is the decode's own loop over continuity blocks — the
    single-unit case a `0 < recordings_done < recordings_total` assertion cannot reach.

    No `PROGRESS_INTERVAL_S` patch: the method is handed a raw callback and its sole caller
    (`pipeline/temporal_clustering.py`) is what wraps one in `ThrottledProgress`.
    """
    features, t_ns = _trajectory()
    t_ns = t_ns.copy()
    for cut in (len(t_ns) // 4, len(t_ns) // 2, 3 * len(t_ns) // 4):
        t_ns[cut:] += (
            500 * STEP
        )  # a tracking gap, so the decode restarts and the loop iterates

    seen: list[tuple[float, str]] = []
    _method().fit(
        [("recording_001", 0, features, t_ns)],
        _params(),
        lambda fraction, phase: seen.append((fraction, phase)),
    )

    assert seen, "the fit reported nothing at all"
    fractions = [
        f for f, _ in seen if f < 1.0
    ]  # the terminal emit restates the whole op
    # A tolerance, not equality: a phase's last in-loop fraction and the next phase's boundary
    # are the same number reached by different arithmetic (0.55 + 0.15 vs a literal 0.7).
    assert all(
        later >= earlier - 1e-9 for earlier, later in zip(fractions, fractions[1:])
    ), f"the fraction went backwards: {seen}"

    within_decode = [f for f, phase in seen if phase == "decoding" and 0.7 < f < 0.9]
    assert len(within_decode) >= 2, (
        "the decode reported only at its phase boundaries — the bar sits at 70% for the whole "
        f"decode, which reads as a hang: {seen}"
    )

    within_labeling = [f for f, phase in seen if "labeling" in phase and 0.55 < f < 0.7]
    assert (
        len(within_labeling) >= 2
    ), f"the labeling loop reported only at its phase boundaries: {seen}"
