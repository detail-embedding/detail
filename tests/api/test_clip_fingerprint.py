"""What sets a cluster apart, for the clip window (``api/clip_fingerprint.py``): the cluster
page's descriptors and ranking, over the cluster's most typical members against windows of
their own lengths from the same animals' recordings, with the clip on screen placed on each.
"""

from __future__ import annotations

import numpy as np

import detail.api.clip_fingerprint as clip_fingerprint
from detail.analysis.descriptors import FrameSlice
from detail.api.clip_fingerprint import _windows, cluster_fingerprint


def _node_with_members(view):
    return next(
        n for n in range(view.tree.n_nodes) if n != view.root and len(view.clips(n)) > 2
    )


def _fast_inside(members) -> FrameSlice:
    """A whole track at 10 frames/s that moves 20 px a frame inside the members' spans and
    1 px a frame elsewhere, with a steady mask: speed is what sets the cluster apart."""
    end = max(int(m["end_t_ns"]) for m in members) + 60_000_000_000
    t_ns = np.arange(0, end, 100_000_000, dtype=np.int64)
    inside = np.zeros(len(t_ns), dtype=bool)
    for m in members:
        start, end = clip_fingerprint.widened(m["start_t_ns"], m["end_t_ns"])
        inside |= (t_ns >= start) & (t_ns <= end)
    x = np.cumsum(np.where(inside, 20.0, 1.0))
    return FrameSlice(
        frame_idx=np.arange(len(t_ns)),
        t_ns=t_ns,
        stride=1,
        centroid=np.stack([x, np.zeros_like(x)], axis=1),
        area=np.full(len(t_ns), 500.0),
        bbox=np.stack(
            [x - 10, np.zeros_like(x), x + 10, np.full_like(x, 20.0)], axis=1
        ),
    )


def test_a_cluster_is_ranked_against_its_animals_recordings(
    fitted_temporal, monkeypatch
):
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    view = run.node_view()
    node = _node_with_members(view)
    members = run.typical_members(node)
    member = members[0]
    # The fixture's embedding names a tracking run it never wrote, so the track is stood in.
    track = _fast_inside(members)
    monkeypatch.setattr(clip_fingerprint, "_trajectory", lambda *a: track)
    clip = (
        member["recording_id"],
        int(member["subject"]),
        int(member["start_t_ns"]),
        int(member["end_t_ns"]),
    )
    heard = []
    print_ = cluster_fingerprint(
        experiment, run, node, clip, progress=lambda *e: heard.append(e)
    )
    assert print_.n_members > 0 and print_.n_units > 0 and print_.n_baseline > 0
    assert 0 < len(print_.rows) <= 3
    deltas = [abs(r.delta) for r in print_.rows]
    assert deltas == sorted(deltas, reverse=True), "most distinctive first"
    for row in print_.rows:
        assert len(row.cluster) == print_.n_members
        assert len(row.baseline) == print_.n_baseline
        assert row.label and row.unit
    assert any(r.clip is not None for r in print_.rows), "the clip is placed"
    top = print_.rows[0]
    assert "speed" in top.name and top.delta > 0.5, "the planted difference is found"
    assert heard[0][0] == "measuring" and heard[-1][1] == 1.0


def test_the_comparison_windows_take_the_members_lengths_and_are_seeded():
    t_ns = np.arange(0, 100_000_000_000, 33_000_000)
    lengths = np.array([1_000_000_000, 2_000_000_000])
    first = _windows(t_ns, lengths, 20, seed=3)
    assert first == _windows(t_ns, lengths, 20, seed=3)
    assert {end - start for start, end in first} <= {1_000_000_000, 2_000_000_000}
    assert first[0][0] == 0 and first[-1][1] <= int(t_ns[-1])
