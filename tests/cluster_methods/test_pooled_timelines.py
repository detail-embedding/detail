"""A temporal method reads time **per recording**, never off the study's timestamps pooled.

**The class: a per-timeline statistic computed over many timelines pooled together.** Every
recording's clock starts at zero, so a study's timestamps concatenated (or worse, sorted) are not
one timeline: a cadence measured as the median step between them is the jitter between
recordings, and a gap rule run over them never sees the step *back* to zero where one video ends
and the next begins.

**How it hid.** ``codebook_viterbi`` measured its rows-per-second as
``median(diff(sort(fit.t_ns)))`` and cut its runs with ``_blocks_of(fit.t_ns)``. Every fixture fit
one recording, or several with *identical* clocks, where sorting merely duplicates steps that the
``> 0`` filter then drops — so both were right on every fixture. On a real multi-recording
study the clocks differ by microseconds: the cadence read far too many rows a second, a
minimum run became thousands of rows, and the only "runs" long enough were ones carried
across a recording seam, which ``resolve`` refuses. The run projected **no instances** and
its node cache listed **no clips for any node** — the greyed *Preview clips…* that reported it.

**The rule.** Measure cadence per unit (recording × subject), and cut blocks at unit seams before
any gap rule; ``finch``'s ``_unit_seams`` / ``_unit_blocks`` are the shape.

**The guard.** A study of several recordings whose clocks are offset by a millisecond or two (as
real ones are), each a trajectory that ends in the pose it began with — so a block running across
a seam would fuse the end of one video with the start of the next. Data-driven over
``TEMPORAL_CLUSTER_METHODS``: every method must place every instance and every ranked member in a
single recording, and give every recording its instances. ``PARAMS`` must name every registered
method, so a method added tomorrow is covered the day it lands.
"""

import numpy as np
import pytest

from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.registries import TEMPORAL_CLUSTER_METHODS

import_temporal_cluster_methods()

STEP = 17_000_000  # ns between rows, the ~59 rows/s a real embed run has
HOLD = 30  # rows a pose is held: 0.51 s at STEP
RECORDINGS = 6

#: Each method's settings at this fixture's cadence. A floor (``min_run_s`` / ``min_len_s``) of
#: 0.3 s keeps a 0.51 s hold — at the true cadence. At a pooled one it keeps nothing.
PARAMS = {
    "codebook_viterbi": {
        "window_s": 0.05,
        "min_run_s": 0.3,
        "fit_share": 40.0,
        "codewords": 12,
    },
    "dtw_finch": {
        "window_s": 0.05,
        "min_len_s": 0.1,
        "band_s": 0.05,
        "penalty": 1.0,
        "min_size_s": 0.05,
    },
}


def _trajectory(seed: int):
    """Poses A B C A, each held ``HOLD`` rows: it ends where it starts."""
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(3, 8))
    kinds = [0, 1, 2, 0]
    rows = np.vstack([np.repeat(anchors[k][None, :], HOLD, axis=0) for k in kinds])
    noise = np.random.default_rng(seed).normal(scale=0.05, size=rows.shape)
    return (rows + noise).astype(np.float32)


def _study():
    """``RECORDINGS`` trajectories whose clocks each start a little later than the last."""
    out = []
    for i in range(RECORDINGS):
        features = _trajectory(seed=i)
        t_ns = np.arange(len(features), dtype=np.int64) * STEP + i * 1_300_000
        out.append((f"recording_{i:03d}", 0, features, t_ns))
    return out


@pytest.fixture(scope="module", params=sorted(TEMPORAL_CLUSTER_METHODS))
def fitted(request):
    method = TEMPORAL_CLUSTER_METHODS[request.param]()
    params = PARAMS[request.param]
    return method, method.fit(_study(), params), params


def test_every_registered_method_has_settings_here():
    """A sweep over a registry passes by testing nothing when the table misses a member."""
    assert set(PARAMS) == set(TEMPORAL_CLUSTER_METHODS)


def _leaf_groups(fit, method):
    return np.arange(method.hierarchy_of(fit).n_leaves, dtype=np.int64)


def test_every_instance_sits_in_one_recording_and_every_recording_has_some(fitted):
    method, fit, params = fitted
    instances = method.instances_for(fit, _leaf_groups(fit, method), params)
    places = [method.resolve(fit, i) for i in instances]
    assert all(places), "an instance ran across a recording seam"
    found = {p["recording_id"] for p in places}
    assert found == {f"recording_{i:03d}" for i in range(RECORDINGS)}, (
        "a recording got no instances: its runs were measured against the wrong cadence"
    )  # fmt: skip


def test_every_ranked_member_can_be_placed(fitted):
    """What the node cache lists and the clip window plays: ``rank_members`` over the whole
    tree, every member placed."""
    method, fit, _params = fitted
    tree = method.hierarchy_of(fit)
    ranked = method.rank_members(fit, np.arange(tree.n_leaves))
    assert ranked, "the root ranked no members"
    assert all(method.resolve(fit, i) for i, _d in ranked)


def test_the_guard_would_catch_a_pooled_cadence(monkeypatch):
    """Revert the fix's shape — cadence off the pooled, sorted timestamps — and the first test
    must fail: the 0.3 s floor then outlasts every hold."""
    from detail.cluster_methods import codebook_viterbi

    def pooled(fit):
        steps = np.diff(np.sort(fit.t_ns))
        steps = steps[steps > 0]
        return 1e9 / float(np.median(steps))

    monkeypatch.setattr(codebook_viterbi, "_rows_per_second", pooled)
    method = TEMPORAL_CLUSTER_METHODS["codebook_viterbi"]()
    params = PARAMS["codebook_viterbi"]
    fit = method.fit(_study(), params)
    instances = method.instances_for(fit, _leaf_groups(fit, method), params)
    placed = {method.resolve(fit, i).get("recording_id") for i in instances}
    assert placed != {f"recording_{i:03d}" for i in range(RECORDINGS)}


def test_the_guard_would_catch_blocks_across_a_seam(monkeypatch):
    """Revert the other half — blocks over the pooled timeline — and a run fuses the end of
    one recording (pose A) with the start of the next (pose A)."""
    from detail.cluster_methods import codebook_viterbi

    monkeypatch.setattr(
        codebook_viterbi,
        "_unit_blocks",
        lambda fit, max_gap_ns: codebook_viterbi._blocks_of(fit.t_ns, max_gap_ns),
    )
    method = TEMPORAL_CLUSTER_METHODS["codebook_viterbi"]()
    params = PARAMS["codebook_viterbi"]
    fit = method.fit(_study(), params)
    instances = method.instances_for(fit, _leaf_groups(fit, method), params)
    assert not all(method.resolve(fit, i) for i in instances)
