"""The DTW+FINCH temporal method (§22) — a vocabulary of behaviors, one cluster per kind of bout.

What is checked here is what separates a temporal method from the pose one, and what a registry
parity test cannot see:

- **A bout is already an instance.** No run-length encoding and no minimum-run floor, because
  ``min_len_s`` dropped the too-short ones before the comparison existed. Every bout the fit
  compared appears exactly once.
- **The rungs are nested.** A name can only survive a granularity change under a rule where a
  coarser setting merges clusters rather than reshuffling members, and §22 requires that be
  *tested* rather than declared: "carrying a name forward under a rule that only looks nested
  would hand a name to a cluster that is half something else".
- **A capped fit says so.** ``max_segments`` makes a fit over a fraction of the bouts identical to
  a full one on every screen, so ``subsample_note`` is the one thing standing between a user and
  a vocabulary quietly built on 20% of the data.

The fit is real — actual segmentation and actual banded DTW over synthetic trajectories with
planted structure. Nothing here is mocked.
"""

import numpy as np
import pytest

from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.registries import TEMPORAL_CLUSTER_METHODS

STEP = 17_000_000  # ns between rows — the ~59 rows/s cadence the real embed runs at


def _method():
    import_temporal_cluster_methods()
    return TEMPORAL_CLUSTER_METHODS["dtw_finch"]()


def _trajectory(n_bouts=12, bout_rows=30, dim=8, kinds=3, seed=0):
    """A trajectory built from repeating bout *kinds*, so there is structure to find.

    Each kind is a smooth path between two anchors, so bouts of one kind resemble each other and
    differ from the others — which is what a bout comparison is supposed to notice, and what a
    fixture of independent noise would not contain.
    """
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(kinds, 2, dim))
    rows = []
    for i in range(n_bouts):
        kind = i % kinds
        ramp = np.linspace(0.0, 1.0, bout_rows)[:, None]
        path = anchors[kind, 0] * (1 - ramp) + anchors[kind, 1] * ramp
        rows.append(path + 0.05 * rng.normal(size=path.shape))
    features = np.vstack(rows).astype(np.float32)
    t_ns = np.arange(len(features), dtype=np.int64) * STEP
    return features, t_ns


def _params(**overrides) -> dict:
    """Segmentation settings that actually cut this fixture, at its cadence.

    The shipped defaults are tuned for a real recording; on a 360-row synthetic one a 0.27 s
    minimum bout is most of the trajectory. Stated here rather than silently inherited so the
    fixture's bouts are the ones the test reasons about.
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


@pytest.fixture(scope="module")
def fitted():
    """One real fit, made once — segmentation plus a genuine O(K²) DTW pass."""
    method = _method()
    features, t_ns = _trajectory()
    fit = method.fit([("recording_001", 0, features, t_ns)], _params())
    return method, fit


# --- the registry -------------------------------------------------------------------------


def test_the_temporal_registry_is_not_the_pose_one():
    """§22 keeps the two apart so a bout method cannot be selected where a pose method belongs.
    Sharing a registry would make that mistake expressible, which is the whole point of the split.
    """
    from detail.cluster_methods import import_pose_cluster_methods
    from detail.core.registries import POSE_CLUSTER_METHODS

    import_pose_cluster_methods()
    import_temporal_cluster_methods()
    assert "dtw_finch" in TEMPORAL_CLUSTER_METHODS
    assert "dtw_finch" not in POSE_CLUSTER_METHODS
    assert "finch" not in TEMPORAL_CLUSTER_METHODS


def test_every_registered_method_declares_the_contract():
    """Data-driven over the registry, so a second temporal method is covered the day it lands."""
    import_temporal_cluster_methods()
    for method_id, cls in TEMPORAL_CLUSTER_METHODS.items():
        method = cls()
        assert method.method_id == method_id
        assert method.config_model is not None
        assert isinstance(method.has_noise, bool)
        assert method.shard_suffix.startswith(".")
        assert method.granularity.kind


# --- a bout is an instance ----------------------------------------------------------------


def test_every_bout_is_exactly_one_instance(fitted):
    """The temporal counterpart of the pose method's run-length encoding — and the absence of it.

    A bout already has an extent, so `instances_at` is a labeling, not a segmentation: as many
    instances as bouts, each appearing once. A method that re-encoded runs here would silently
    merge adjacent bouts that landed in the same cluster, which is a different vocabulary.
    """
    method, fit = fitted
    granularity = min(r for r in fit.granularity.rungs if r >= 2)
    instances = method.instances_at(fit, granularity, _params())

    assert len(instances) == len(fit)
    assert sorted(i.start_row for i in instances) == list(range(len(fit)))
    assert all(i.start_row == i.end_row for i in instances)


def test_no_bout_is_left_unassigned(fitted):
    """`has_noise = False` is a declaration, and this is what makes it true."""
    method, fit = fitted
    granularity = min(r for r in fit.granularity.rungs if r >= 2)
    instances = method.instances_at(fit, granularity, _params())
    assert all(i.cluster_id >= 0 for i in instances)
    assert method.has_noise is False


def test_the_planted_structure_is_found(fitted):
    """Assert the content, not that the call returned. A fit that put every bout in one cluster,
    or each in its own, would satisfy every structural test above and be useless."""
    method, fit = fitted
    counts = fit.granularity.rungs
    assert len(counts) >= 2, "a hierarchy with one level found no structure"
    assert max(counts) > min(counts)
    # The fixture has three kinds of bout; some rung should be in that neighbourhood rather than
    # the extremes.
    assert any(2 <= c <= 6 for c in counts), counts


# --- nesting, which is what lets a name survive -------------------------------------------


def test_the_rungs_are_nested(fitted):
    """§22 requires this be verified, not asserted: re-label at the coarser rung and every new
    cluster must be a **union** of finer ones.

    Carrying a name forward under a rule that only looks nested would hand a name to a cluster
    that is half something else, and it would look deliberate.
    """
    method, fit = fitted
    rungs = sorted(fit.granularity.rungs)
    assert len(rungs) >= 2

    # Ascending **count**, so the smaller number is the *coarser* setting — fewer clusters. Worth
    # stating because getting it backwards inverts the whole property and the test still runs.
    for coarser, finer in zip(rungs, rungs[1:]):
        coarse = fit.labels_at(coarser)
        fine = fit.labels_at(finer)
        # A coarser cluster is a union of finer ones exactly when no finer cluster is split across
        # two coarser ones.
        for cluster in np.unique(coarse):
            members = set(np.flatnonzero(coarse == cluster).tolist())
            for f in {int(x) for x in fine[sorted(members)]}:
                assert set(np.flatnonzero(fine == f).tolist()) <= members, (
                    f"cluster {f} at {finer} straddles cluster {cluster} at {coarser} — "
                    "the rungs are not nested, so a name could not survive the change"
                )


def _is_nested(coarse: np.ndarray, fine: np.ndarray) -> bool:
    """Whether every ``fine`` cluster sits wholly inside one ``coarse`` cluster."""
    for cluster in np.unique(coarse):
        members = set(np.flatnonzero(coarse == cluster).tolist())
        for f in {int(x) for x in fine[sorted(members)]}:
            if not set(np.flatnonzero(fine == f).tolist()) <= members:
                return False
    return True


def test_the_nesting_check_would_catch_a_partition_that_is_not_nested(fitted):
    """Guard the guard. A property test that cannot fail passes vacuously forever, and "these
    partitions are nested" is exactly the kind of claim that looks verified while checking
    nothing."""
    method, fit = fitted
    rungs = sorted(fit.granularity.rungs)
    coarse = fit.labels_at(rungs[0])
    fine = fit.labels_at(rungs[1])
    assert _is_nested(coarse, fine), "the real pair should pass"

    # Move one bout into another cluster and the union property must break.
    tampered = np.array(fine)
    victim = int(np.flatnonzero(tampered == tampered[0])[0])
    tampered[victim] = int(tampered.max()) + 1
    assert not _is_nested(coarse, tampered) or len(np.unique(coarse)) == 1


def test_a_setting_between_rungs_still_labels_every_bout(fitted):
    """Off-rung settings go through refinement rather than the hierarchy, and are not nested —
    that is why they discard names. They must still produce a complete labeling."""
    method, fit = fitted
    rungs = sorted(fit.granularity.rungs)
    between = next((k for a, b in zip(rungs, rungs[1:]) for k in range(a + 1, b)), None)
    if between is None:
        pytest.skip("this fit's rungs are adjacent; there is no setting between them")
    labels = fit.labels_at(between)
    assert len(labels) == len(fit)
    assert len(np.unique(labels)) == between


# --- ranking a node ----------------------------------------------------------------------


def test_ranking_a_node_leads_with_its_medoid(fitted):
    """Medoid rather than nearest-centroid: this is a distance-based rule and there is no centroid
    to be near, so what a node *is* is its most central member."""
    method, fit = fitted
    tree = method.hierarchy_of(fit)
    root = tree.n_nodes - 1
    node = max(
        (int(c) for c in tree.children[root - tree.n_leaves]),
        key=lambda c: len(tree.leaves_under(c)),
    )
    members = np.flatnonzero(np.isin(fit.levels[0], tree.leaves_under(node)))

    order = method.rank_members(fit, tree.leaves_under(node))
    assert sorted(r.start_row for r, _ in order) == sorted(members.tolist())

    # The first is the member with the smallest mean distance to the others.
    dist = np.asarray(fit.dist, dtype=np.float64)
    means = {
        int(m): float(dist[m, members].sum() / max(len(members) - 1, 1))
        for m in members
    }
    assert order[0][0].start_row == min(means, key=means.get)


def test_a_node_with_no_leaves_ranks_to_nothing(fitted):
    method, fit = fitted
    assert method.rank_members(fit, np.zeros(0, dtype=np.int64)) == []


# --- progress ------------------------------------------------------------------------------


def test_the_comparison_phase_counts_bouts_not_pairs():
    """The number on screen has to be the one the user set.

    `pairwise_dtw` reports `(pairs_done, pairs_total)`, and pairs are not bouts: a 20-bout fit
    compares 190 pairs. Passing the pair total straight into the message told a real run that its
    400-bout fit was "comparing 79,800 bouts", which reads as the cap having been ignored.

    Asserted on the *number*, not on the call returning, because the wrong number is a perfectly
    well-formed string (CLAUDE.md: "a wrong value the user reads → assert the number").
    """
    method = _method()
    features, t_ns = _trajectory()
    seen: list[tuple[float, str]] = []
    fit = method.fit(
        [("recording_001", 0, features, t_ns)],
        _params(),
        lambda fraction, phase: seen.append((fraction, phase)),
    )

    comparing = [phase for _f, phase in seen if phase.startswith("comparing")]
    assert comparing, "the comparison phase never reported"
    assert all(p == f"comparing {len(fit):,} bouts" for p in comparing), comparing
    # ...and the fraction still advanced through it, which is what the pair counts are for.
    fractions = [f for f, phase in seen if phase.startswith("comparing")]
    assert max(fractions) > min(fractions) or len(fractions) == 1


def test_the_fit_reports_its_phases_in_order():
    """The phase name is what moves while the fraction is pinned inside a long step (§22)."""
    method = _method()
    features, t_ns = _trajectory()
    seen: list[str] = []
    method.fit(
        [("recording_001", 0, features, t_ns)],
        _params(),
        lambda fraction, phase: seen.append(phase),
    )
    assert seen[0] == "loading"
    assert "segmenting" in seen
    assert any(p.startswith("comparing") for p in seen)
    assert "clustering" in seen


# --- the scale wall, stated ---------------------------------------------------------------


def test_a_full_fit_says_nothing_about_a_subsample(fitted):
    """The negative case. If this returned a note unconditionally the guard below would be
    meaningless."""
    method, fit = fitted
    assert method.subsample_note(fit) is None


def test_a_capped_fit_states_what_it_left_out():
    """§22: a fit over a fraction of the bouts is identical to a full one in every other respect,
    so the surface must be told. This is where that sentence comes from."""
    method = _method()
    features, t_ns = _trajectory()
    full = method.fit([("recording_001", 0, features, t_ns)], _params())
    capped = method.fit([("recording_001", 0, features, t_ns)], _params(max_segments=4))

    assert len(capped) == 4
    assert len(capped) < len(full)
    note = method.subsample_note(capped)
    assert note is not None
    assert "4" in note and f"{full.n_bouts:,}" in note


# --- a cap samples what the vocabulary is *learned from*, not what it labels ----------------


def test_a_capped_fit_still_labels_every_bout():
    """The point of a discovery module is the instances, so a cap must not delete any.

    ``max_segments`` bounds the O(bouts²) comparison by dropping bouts before the matrix is
    built. If those bouts were left there, they would carry no cluster, produce no instance and
    never reach the artifact — the knob would truncate the *output* rather than sample the *fit*,
    which for a method whose whole job is finding every occurrence is missing data rather than a
    quality trade.

    So the dropped bouts are assigned to the vocabulary afterwards, and the count of instances
    does not depend on the cap.
    """
    method = _method()
    features, t_ns = _trajectory()
    full = method.fit([("recording_001", 0, features, t_ns)], _params())
    capped = method.fit([("recording_001", 0, features, t_ns)], _params(max_segments=4))

    assert len(capped) == 4, "the cap must still bound what is compared"
    granularity = min(r for r in capped.granularity.rungs if r >= 2)
    instances = method.instances_at(capped, granularity, _params())

    assert len(instances) == len(full), (
        f"a capped fit produced {len(instances)} instances against the uncapped "
        f"{len(full)} — the cap is deleting bouts, not sampling them"
    )
    assert all(i.cluster_id >= 0 for i in instances)


def test_an_assigned_bout_inherits_through_the_nesting(fitted):
    """The property the whole design rests on, checked rather than assumed.

    An assigned bout is placed once, against the **finest** rung's medoids, and never
    re-compared. That is only correct because every granularity's partition is a union of
    finest-rung clusters: `_refine_to` starts from a level and only merges, and the levels
    themselves nest. So an assigned bout's label at any setting must equal what the kept bouts
    of its fine cluster carry there.

    Get this wrong and nothing raises — the bout simply lands in a neighbouring cluster, and the
    vocabulary looks fine.
    """
    method = _method()
    features, t_ns = _trajectory()
    fit = method.fit([("recording_001", 0, features, t_ns)], _params(max_segments=6))
    kept = len(fit)
    assert fit.n_assigned > 0, "nothing was assigned; this proves nothing"

    finest = fit.levels[0]
    for granularity in fit.granularity.rungs:
        labels = fit.all_labels_at(granularity)
        kept_labels = labels[:kept]
        for offset, fine in enumerate(fit.assigned_fine_cluster):
            members = np.flatnonzero(finest == int(fine))
            assert len(members), f"fine cluster {fine} has no kept members"
            expected = int(kept_labels[members[0]])
            assert int(labels[kept + offset]) == expected, (
                f"assigned bout {offset} reads {labels[kept + offset]} at granularity "
                f"{granularity} but its fine cluster {fine} is cluster {expected} there"
            )


def test_an_uncapped_fit_assigns_nothing(fitted):
    """The negative case. If a fit always carried assigned bouts, the guards above would be
    describing something other than the cap."""
    method, fit = fitted
    assert fit.n_bouts == fit.n_clustered
    assert fit.n_assigned == 0
    assert method.subsample_note(fit) is None


def test_the_assignment_phase_appears_only_when_bouts_were_dropped():
    """Assignment is a real part of the fit cost at study scale, not a rounding error, so it is
    its own phase rather than time hidden inside `clustering`."""
    method = _method()
    features, t_ns = _trajectory()

    full_phases: list[str] = []
    method.fit(
        [("recording_001", 0, features, t_ns)],
        _params(),
        lambda fraction, phase: full_phases.append(phase),
    )
    assert not any(p.startswith("assigning") for p in full_phases), full_phases

    capped_phases: list[str] = []
    method.fit(
        [("recording_001", 0, features, t_ns)],
        _params(max_segments=4),
        lambda fraction, phase: capped_phases.append(phase),
    )
    assert any(p.startswith("assigning") for p in capped_phases), capped_phases


# --- persistence --------------------------------------------------------------------------


def test_a_fit_round_trips_through_disk(fitted, tmp_path):
    """The fit is the expensive thing and the module stores it opaquely, so a lossy serializer
    would surface as a vocabulary that changed when a run was reopened."""
    method, fit = fitted
    path = tmp_path / f"fit{method.shard_suffix}"
    method.serialize(fit, path)
    back = method.load(path)

    granularity = min(r for r in fit.granularity.rungs if r >= 2)
    assert len(back) == len(fit)
    assert back.recording_ids == fit.recording_ids
    assert np.array_equal(back.start_t_ns, fit.start_t_ns)
    assert np.array_equal(back.end_t_ns, fit.end_t_ns)
    assert np.array_equal(back.subjects, fit.subjects)
    assert back.granularity.rungs == fit.granularity.rungs
    assert np.array_equal(back.labels_at(granularity), fit.labels_at(granularity))


# --- refusals -----------------------------------------------------------------------------


def test_no_trajectories_is_refused_with_the_reason():
    method = _method()
    with pytest.raises(ValueError, match="at least one trajectory"):
        method.fit([], _params())


def test_a_trajectory_with_no_bouts_says_which_knob_to_look_at():
    """A too-short trajectory produces no bouts at all. Failing with "no bouts were found" and
    naming the minimum length beats an IndexError three frames deeper."""
    method = _method()
    features, t_ns = _trajectory(n_bouts=1, bout_rows=4)
    with pytest.raises(ValueError, match="no bouts|minimum bout length"):
        method.fit([("recording_001", 0, features, t_ns)], _params(min_len_s=60.0))
