"""The cluster profile (``api/cluster_profile.py``): a seeded random sample of a node's members
against a comparison sample, measured with the registered descriptors, explained by a classifier
scored on held-out recordings beside its shuffled floor.

The classes guarded here:

- **A confident explanation of nothing.** A classifier always finds features to lean on, so a
  profile must report its held-out score with a floor and say when the score does not clear it.
  A planted difference has to be found *and* separated; a cluster with nothing planted must come
  back not separated.
- **Labels build nothing.** The rows are told apart by cluster membership only; permuting the
  experiment's behavior labels cannot move a single number (`test_label_leakage.py`'s rule).
- **A saved profile reads back as what was saved**, so reopening shows the same answer.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import detail.api.cluster_profile as cluster_profile
from detail.analysis.descriptors import FrameSlice
from detail.configs.cluster_profile import ClusterProfileConfig


def _node_with_members(view, *, min_runs=12):
    candidates = [
        n
        for n in range(view.tree.n_nodes)
        if n != view.root and len(view.sample_runs(n, n=10_000, seed=0)) >= min_runs
    ]
    # The smallest qualifying node, so the rest of the run is large beside it.
    return min(candidates, key=lambda n: view.summary(n).members)


def _track(members_by_unit, unit, *, fast: bool, elsewhere: bool = False):
    """A whole track at 100 frames/s, finer than the fit's bouts so every member spans several
    frames: about 20 px a frame inside the node's runs when
    ``fast``, about 1 px a frame everywhere else, with a mask that varies a little. With
    ``elsewhere`` the node's runs differ only in *where* they are: 300 px further down the
    image, at the same speed and shape."""
    spans = members_by_unit.get(unit, [])
    end = max([int(s["end_t_ns"]) for s in spans] + [0]) + 600_000_000_000
    t_ns = np.arange(0, end, 10_000_000, dtype=np.int64)
    inside = np.zeros(len(t_ns), dtype=bool)
    if fast or elsewhere:
        for s in spans:
            lo, hi = s["start_t_ns"], s["end_t_ns"]
            inside |= (t_ns >= lo) & (t_ns <= hi)
    y = np.where(inside, 300.0, 0.0) if elsewhere else np.zeros(len(t_ns))
    if not fast:
        inside = np.zeros(len(t_ns), dtype=bool)
    # Seeded jitter, so a measure varies without carrying the membership.
    jitter = np.random.default_rng(zlib.crc32(repr(unit).encode())).uniform(
        0.5, 1.5, len(t_ns)
    )
    x = np.cumsum(np.where(inside, 20.0, 1.0) * jitter)
    area = 500.0 * np.random.default_rng(7).uniform(0.9, 1.1, len(t_ns))
    return FrameSlice(
        frame_idx=np.arange(len(t_ns)),
        t_ns=t_ns,
        stride=1,
        centroid=np.stack([x, y], axis=1),
        area=area,
        bbox=np.stack([x - 10, y, x + 10, y + 20.0], axis=1),
    )


@pytest.fixture
def profiled(fitted_profile, monkeypatch):
    experiment, run_id = fitted_profile
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    view = run.node_view()
    node = _node_with_members(view)
    members = view.sample_runs(node, n=100_000, seed=0)
    by_unit: dict = {}
    for m in members:
        by_unit.setdefault((m["recording_id"], int(m["subject"])), []).append(m)

    def stand_in(fast, elsewhere=False):
        # The fixture's embedding names a tracking run it never wrote, so tracks are stood in.
        monkeypatch.setattr(
            cluster_profile,
            "_trajectory",
            lambda _e, key, _r, _ts: _track(
                by_unit, key, fast=fast, elsewhere=elsewhere
            ),
        )

    return experiment, run, node, stand_in


def _config(**kw):
    base = dict(explainer="logistic_regression", n_per_side=40, seed=0)
    return ClusterProfileConfig(**{**base, **kw})


def test_a_planted_difference_is_found_and_separated(profiled):
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    heard = []
    profile = cluster_profile.profile_cluster(
        experiment, run, node, _config(), progress=lambda *e: heard.append(e[0])
    )
    e = profile.explanation
    assert e.separated, profile.headline
    assert e.accuracy > e.floor + e.margin
    assert e.n_cluster == e.n_comparison > 0, "the two sides are balanced"
    top = e.importances[0]
    assert "speed" in top.name and top.direction == 1
    assert top.cluster_median > top.comparison_median
    assert profile.basis.startswith("mask")
    assert {"sampling", "measuring", "fitting", "context"} <= set(heard)


def test_a_score_short_of_its_margin_is_reported_as_not_separating(profiled):
    """No planted speed: what is left (span lengths differ between the node's runs and the
    rest's) is real but weak, and with a demanding margin the profile must say it does not
    separate rather than present the chart as trustworthy."""
    experiment, run, node, stand_in = profiled
    stand_in(fast=False)
    profile = cluster_profile.profile_cluster(
        experiment, run, node, _config(margin=0.5)
    )
    e = profile.explanation
    assert e.accuracy < e.floor + 0.5
    assert not e.separated
    assert "don't separate" in profile.headline


def test_the_same_seed_gives_the_same_profile(profiled):
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    a = cluster_profile.profile_cluster(experiment, run, node, _config())
    b = cluster_profile.profile_cluster(experiment, run, node, _config())
    assert a.explanation.accuracy == b.explanation.accuracy
    assert [r["start_t_ns"] for r in a.rows] == [r["start_t_ns"] for r in b.rows]


def test_the_sibling_comparison_draws_from_the_sibling(profiled):
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    view = run.node_view()
    profile = cluster_profile.profile_cluster(
        experiment, run, node, _config(compare="sibling")
    )
    parent = view.ancestors(node)[-1]
    (sibling,) = [c for c in view.children(parent) if c != node]
    assert profile.comparison_node == sibling


def test_the_root_has_no_sibling(profiled):
    experiment, run, _node, _stand_in = profiled
    with pytest.raises(ValueError, match="the root has no sibling"):
        cluster_profile.profile_cluster(
            experiment, run, run.node_view().root, _config(compare="sibling")
        )


def test_a_saved_profile_reads_back(profiled, tmp_path):
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    profile = cluster_profile.profile_cluster(experiment, run, node, _config())
    out = cluster_profile.save_profile(tmp_path, profile)
    first = (out / "profile.yaml").read_text(encoding="utf-8-sig").splitlines()[0]
    assert first == "schema_version: 1"
    back = cluster_profile.load_profile(tmp_path, node)
    assert back.explanation.accuracy == profile.explanation.accuracy
    assert [i.name for i in back.explanation.importances] == [
        i.name for i in profile.explanation.importances
    ]
    assert back.config == profile.config
    assert (
        (out / "measures.csv")
        .read_text(encoding="utf-8-sig")
        .startswith("recording_id,subject,start_t_ns,end_t_ns,in_cluster")
    )
    assert cluster_profile.load_profile(tmp_path, node + 10_000) is None


def test_a_profile_with_no_held_out_score_says_why_and_keeps_it(profiled, tmp_path):
    """A cluster from one recording has no held-out score (``explainers.NO_SCORABLE_FOLD``):
    the headline gives that reason — not the fallback note beside it, and not a percentage —
    and a saved profile keeps it."""
    import dataclasses

    from detail.analysis import explainers

    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    profile = cluster_profile.profile_cluster(experiment, run, node, _config())
    unscored = dataclasses.replace(
        profile.explanation,
        accuracy=float("nan"),
        separated=False,
        note="shap is not installed",
        unscored=explainers.NO_SCORABLE_FOLD,
    )
    shown = dataclasses.replace(profile, explanation=unscored)
    assert shown.headline == explainers.NO_SCORABLE_FOLD
    cluster_profile.save_profile(tmp_path, shown)
    back = cluster_profile.load_profile(tmp_path, node)
    assert back.headline == explainers.NO_SCORABLE_FOLD
    assert back.explanation.note == "shap is not installed"


def test_context_clues_come_from_the_cache(profiled):
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    profile = cluster_profile.profile_cluster(experiment, run, node, _config())
    c = profile.context
    assert np.isfinite(c.bout_median_s) and c.bout_p10_s <= c.bout_median_s
    assert len(c.session_thirds) == 3 and abs(sum(c.session_thirds) - 1) < 1e-3
    for side in (c.before, c.after):
        assert abs(sum(n["share"] for n in side) - 1) < 1e-3 or len(side) == 3
        view = run.node_view()
        path = view.ancestors(node)
        branches = {
            other
            for parent, child in zip(path, [*path[1:], node])
            for other in view.children(parent)
            if other != child
        }
        assert {n["node"] for n in side} <= branches


def test_labels_do_not_build_the_profile(profiled, monkeypatch):
    """Permute what every label says and nothing a profile computes may move."""
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    before = cluster_profile.profile_cluster(experiment, run, node, _config())
    names = run.node_names()
    shuffled = dict(zip(names, reversed(list(names.values())))) if names else {}
    monkeypatch.setattr(type(run), "node_names", lambda self: shuffled)
    after = cluster_profile.profile_cluster(experiment, run, node, _config())

    def comparable(
        rows,
    ):  # NaN never equals NaN; a missing value is the same either way
        return [
            {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in r.items()
            }
            for r in rows
        ]

    assert comparable(before.rows) == comparable(after.rows)
    assert before.explanation.accuracy == after.explanation.accuracy
    assert [i.value for i in before.explanation.importances] == [
        i.value for i in after.explanation.importances
    ]


def test_a_side_only_measure_is_skipped_with_the_view_as_its_reason(profiled):
    """No camera view set: the height and floor measures are named with why, not reported as
    'missing on every row' (which would send the reader looking for a broken sidecar).
    """
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    profile = cluster_profile.profile_cluster(experiment, run, node, _config())
    dropped = profile.explanation.measures_dropped
    for name in ("mask_height", "floor_contact"):
        assert "needs a side view camera" in dropped[name], dropped.get(name)
        assert "not set" in dropped[name]


def test_an_over_represented_group_is_named_by_its_value():
    """`Distribution.top` is the value itself. Read as an index into `values`, it raised on
    the first real experiment whose recording information had a group the node was
    over-represented in, which no fixture without metadata could reach."""
    import numpy as np

    from detail.api.node_view import Distribution

    class _View:
        def neighbours(self, node):
            return {}, {}, 0

        def run_durations_s(self, node):
            return np.array([1.0, 2.0])

        def session_thirds(self, node):
            return [1.0, 0.0, 0.0]

        def distribution(self, node, groups):
            return Distribution(
                values=("drug", "vehicle"),
                node=np.array([0.8, 0.2]),
                run=np.array([0.5, 0.5]),
                top="drug",
                ratio=1.6,
                floor=1.2,
                flagged=True,
            )

    class _Experiment:
        metadata_columns = ["treatment"]
        recording_metadata = {
            "r1": {"treatment": "drug"},
            "r2": {"treatment": "vehicle"},
        }

    clues = cluster_profile.context_clues(_Experiment(), _View(), 3, {})
    assert clues.over_represented == [
        {"column": "treatment", "value": "drug", "ratio": 1.6, "floor": 1.2}
    ]


def test_a_skipped_measure_keeps_its_label(profiled):
    """A measure skipped for its camera view or a missing input is listed by its label, the
    way every other line names it, not by its internal name."""
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    profile = cluster_profile.profile_cluster(experiment, run, node, _config())
    for name in profile.explanation.measures_dropped:
        assert name in profile.labels, name
    assert profile.labels["mask_height"] == "mask height"


# --- another cluster, and where / which way ------------------------------------------- #
def _apart_from(view, node):
    """A node with members that neither holds ``node`` nor sits inside it."""
    related = set(view.ancestors(node))
    for n in sorted(range(view.tree.n_nodes), key=lambda n: -view.summary(n).members):
        if n == node or n in related or node in view.ancestors(n):
            continue
        if len(view.sample_runs(n, n=10_000, seed=0)) >= 12:
            return n
    raise AssertionError("no unrelated node with members in the fixture's tree")


def test_a_node_comparison_draws_only_from_the_named_cluster(profiled):
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    view = run.node_view()
    other = _apart_from(view, node)
    profile = cluster_profile.profile_cluster(
        experiment, run, node, _config(compare="node", against=other)
    )
    assert profile.compare == "node" and profile.comparison_node == other
    theirs = {
        (r["recording_id"], int(r["subject"]), int(r["start_t_ns"]))
        for r in view.sample_runs(other, n=100_000, seed=0)
    }
    comparison = [r for r in profile.rows if not r["in_cluster"]]
    assert comparison
    assert all(
        (r["recording_id"], int(r["subject"]), int(r["start_t_ns"])) in theirs
        for r in comparison
    )


@pytest.mark.parametrize("relation", ["ancestor", "descendant", "itself"])
def test_a_node_comparison_refuses_a_cluster_that_holds_or_is_inside_it(
    profiled, relation
):
    experiment, run, node, _stand_in = profiled
    view = run.node_view()
    if relation == "ancestor":
        other, fragment = view.ancestors(node)[-1], "sits inside"
    elif relation == "descendant":
        parent = view.ancestors(node)[-1]
        node, other, fragment = parent, node, "part of it"
    else:
        other, fragment = node, "with itself"
    with pytest.raises(ValueError, match=fragment):
        cluster_profile.profile_cluster(
            experiment, run, node, _config(compare="node", against=other)
        )


def test_against_goes_only_with_a_node_comparison():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="needs the other cluster"):
        ClusterProfileConfig(compare="node")
    with pytest.raises(ValidationError, match="only compare 'node' uses"):
        ClusterProfileConfig(compare="rest", against=3)


def test_a_batch_has_no_node_comparison():
    """Every cluster against one named cluster is not a batch question; the job's own
    `compare` offers the rest and the sibling only."""
    from typing import get_args

    from detail.configs.profile_describe import ProfileDescribeConfig

    choices = get_args(ProfileDescribeConfig.model_fields["compare"].annotation)
    assert set(choices) == {"rest", "sibling"}


def test_where_alone_separates_only_when_place_is_kept(profiled):
    """The node's members differ from the rest only in where they are. Left out (the
    default), position cannot separate them; kept, it does. Reverting the leave-out to a no-op
    fails the first half, and dropping the place measures from the registry fails the second.
    """
    experiment, run, node, stand_in = profiled
    stand_in(fast=False, elsewhere=True)
    # The margin of the not-separating test above: span lengths differ a little on their own.
    margin = 0.3
    left_out = cluster_profile.profile_cluster(
        experiment, run, node, _config(margin=margin)
    )
    e = left_out.explanation
    assert not ({"position_x", "position_y"} & set(e.measures_used))
    assert e.measures_dropped["position_y"] == cluster_profile.LEFT_OUT_PLACE
    assert np.isfinite(
        [r["position_y"] for r in left_out.rows]
    ).all(), "a place measure left out of the fit is still measured and kept"
    assert not e.separated, left_out.headline

    kept = cluster_profile.profile_cluster(
        experiment, run, node, _config(margin=margin, leave_out_place=False)
    )
    assert "position_y" in kept.explanation.measures_used
    assert kept.explanation.accuracy >= e.accuracy + 0.1, kept.headline
    assert kept.explanation.importances[0].name == "position_y"


def test_a_node_comparison_is_saved_beside_the_profile_not_over_it(profiled, tmp_path):
    experiment, run, node, stand_in = profiled
    stand_in(fast=True)
    other = _apart_from(run.node_view(), node)
    own = cluster_profile.profile_cluster(experiment, run, node, _config())
    pair = cluster_profile.profile_cluster(
        experiment, run, node, _config(compare="node", against=other)
    )
    cluster_profile.save_profile(tmp_path, own)
    out = cluster_profile.save_profile(tmp_path, pair)
    assert out == cluster_profile.profile_dir(tmp_path, node) / f"against_{other}"
    assert cluster_profile.load_profile(tmp_path, node).compare == "rest"
    back = cluster_profile.load_profile(tmp_path, node, other)
    assert back.compare == "node" and back.comparison_node == other


def test_a_node_comparison_fits_place_alone_and_keeps_it(profiled, tmp_path):
    """Against another cluster, the same rows are fitted on position and facing alone: the
    evidence that two clusters differ in where, not what. Only then, and only with place left
    out of the main fit; saved and read back with the profile."""
    experiment, run, node, stand_in = profiled
    stand_in(fast=False, elsewhere=True)
    other = _apart_from(run.node_view(), node)
    pair = cluster_profile.profile_cluster(
        experiment, run, node, _config(compare="node", against=other)
    )
    place = pair.place_explanation
    assert place is not None
    assert set(place.measures_used) <= {
        "position_x",
        "position_y",
        "facing_right",
        "facing_down",
    }
    assert place.separated, cluster_profile.place_text(pair)
    assert cluster_profile.place_text(pair).startswith("Position and facing alone:")

    cluster_profile.save_profile(tmp_path, pair)
    back = cluster_profile.load_profile(tmp_path, node, other)
    assert back.place_explanation.accuracy == place.accuracy
    assert back.place_explanation.measures_used == place.measures_used

    rest = cluster_profile.profile_cluster(experiment, run, node, _config())
    kept = cluster_profile.profile_cluster(
        experiment, run, node,
        _config(compare="node", against=other, leave_out_place=False),
    )  # fmt: skip
    assert rest.place_explanation is None and kept.place_explanation is None
    assert cluster_profile.place_text(rest) == ""
    cluster_profile.save_profile(tmp_path, rest)
    assert cluster_profile.load_profile(tmp_path, node).place_explanation is None
