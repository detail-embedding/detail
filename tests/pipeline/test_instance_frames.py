"""Guard: an instance's frame numbers are the video frames at its times, for every cluster method.

**The class:** a row position written as a video frame number. A method works in rows — positions
in a fit sample or in a trajectory — and a row position equals a video frame only when tracking
never lost a frame, nothing was strided, and the recording is one part. Pose clustering wrote
fit-sample row numbers as ``start_frame``/``end_frame`` (#505) and both temporal methods wrote
trajectory positions (``np.arange(len(features))``), so every clip exported from those runs came
from the wrong place on any recording tracking had lost frames on. Nothing failed: the numbers were
plausible, and every GUI surface places instances by ``t_ns``, which was right.

The mechanism is now one lookup, ``behavior_artifact.UnitFrames.frame_at``, used by both
pipelines. This test runs **every registered method** of both registries on a trajectory whose
first 50 frames and a 30-frame stretch were never embedded, and asserts each instance's frames are
the ones at its ``t_ns``. A method registered without an entry in ``PARAMS`` fails here rather
than going unchecked.

The lookup reads the embedding loaders' ``frame_idx``, so it is only as right as they are: past
the first video part they returned part-local numbers until #23. That next layer, a part-local
frame read as a recording-wide one, is guarded by ``tests/pipeline/test_part_frames.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from detail.cluster_methods import import_pose_cluster_methods
from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.registries import POSE_CLUSTER_METHODS
from detail.core.registries import TEMPORAL_CLUSTER_METHODS
from detail.storage import recording as recording_store

STEP = 17_000_000
FIRST_FRAME = 50
LOST = range(170, 200)
VIDEO_TAIL = 60

#: Settings that make each method cut a short synthetic trajectory at this cadence.
PARAMS = {
    ("pose_clustering", "finch"): {"fit_share": 80},
    ("pose_clustering", "two_step"): {"fit_share": 80, "n_groups": 40},
    ("temporal_clustering", "dtw_finch"): {
        "window_s": 0.05,
        "min_len_s": 0.1,
        "band_s": 0.05,
        "penalty": 1.0,
        "min_size_s": 0.05,
    },
    ("temporal_clustering", "codebook_viterbi"): {
        "fit_share": 20.0,
        "codewords": 20,
        "min_run_s": 0.05,
    },
}


def _registered() -> list[tuple[str, str]]:
    import_pose_cluster_methods()
    import_temporal_cluster_methods()
    return [("pose_clustering", m) for m in sorted(POSE_CLUSTER_METHODS)] + [
        ("temporal_clustering", m) for m in sorted(TEMPORAL_CLUSTER_METHODS)
    ]


def _trajectory(n_bouts=12, bout_rows=30, dim=8, kinds=3, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(kinds, 2, dim))
    rows = []
    for i in range(n_bouts):
        ramp = np.linspace(0.0, 1.0, bout_rows)[:, None]
        path = anchors[i % kinds, 0] * (1 - ramp) + anchors[i % kinds, 1] * ramp
        rows.append(path + 0.05 * rng.normal(size=path.shape))
    return np.vstack(rows).astype(np.float32)


@pytest.mark.parametrize("module_id, method", _registered())
def test_instance_frames_are_the_video_frames_at_their_times(
    module_id, method, experiment_dir, recording_dir, make_embedding_run
):
    from detail.api.experiment import Experiment

    assert (module_id, method) in PARAMS, (
        f"{module_id} method {method!r} has no entry in PARAMS — add settings that cut the "
        f"synthetic trajectory, so its frame numbers are checked"
    )
    features = _trajectory()
    embedded = [
        f
        for f in range(FIRST_FRAME, FIRST_FRAME + len(features) + len(LOST))
        if f not in LOST
    ]
    frame_idx = np.asarray(embedded[: len(features)], dtype=np.int64)
    t_ns = frame_idx * STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.arange(int(frame_idx[-1]) + VIDEO_TAIL, dtype=np.int64) * STEP,
    )
    make_embedding_run(recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01")

    proj = Experiment(experiment_dir)
    run = proj.run_analysis(
        module_id,
        {
            "source_recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "method": method,
            "method_params": PARAMS[(module_id, method)],
        },
    )
    run.wait()
    handle = (
        proj.pose_clustering_run(run.id)
        if module_id == "pose_clustering"
        else proj.temporal_clustering_run(run.id)
    )
    # Two labels on the root's two children, so the output has distinct labelled runs whose
    # frame numbers can each be checked (§22: labels, not a cut, define the instances).
    tree = handle.hierarchy()
    left, right = (int(n) for n in tree.children[tree.n_nodes - 1 - tree.n_leaves])
    handle.name_nodes({left: "a", right: "b"})

    instances = handle.artifact().instances
    assert instances, "the fixture must yield instances, or this checks nothing"
    for instance in instances:
        assert instance.start_frame == instance.t_ns // STEP
        assert instance.end_frame == instance.end_t_ns // STEP
