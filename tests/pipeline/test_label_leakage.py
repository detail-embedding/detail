"""Labels score a method; they never build one (CLAUDE.md).

The signature of this class is that a method's *output* moves when ground truth moves. Nothing
else about it is visible: the run is green, the number is plausible, and the method looks better
than it will in a user's hands — who has no labels at the moment they run it.

So the test is a permutation. Run the pipeline twice with identical user-supplied inputs — the
same exemplar clips, the same config — against two different ground-truth streams, and compare the
**predictions**. Metrics are expected to move (they are scored against different labels); the
predictions are not, because nothing a user could hand the method has changed.

This caught one live instance when it was written: ``soft_knn`` sampled its negative index from
``emb[~pos_mask & ~excl_mask]``, so which frames landed in the neighbour index depended on where
the positives were. A typical exemplar scored far higher that way than sampling from the
unlabeled remainder like a user must. That is now the ``negative_pool`` field, defaulting to ``unlabeled``;
the old behaviour is kept as ``labeled_clean`` to re-read runs recorded before it, and is the one
parametrisation below expected to fail.

It also cleared a suspect. Clustering's per-behavior F1 maps each cluster to its majority true
behavior on the scored recording, which has the same shape — but AMI/ARI/B-cubed are
label-invariant and never see that map, and majority-vote mapping is the standard definition of
purity-style per-class scores rather than a leak.
"""

import numpy as np
import pytest

from detail.api.recording import Recording
from detail.configs.query_eval import _PERFRAME_METHODS
from detail.pipeline.query_eval import evaluate_query
from detail.storage import recording as recording_store

N = 220
PERIOD = 16_666_667
_A = np.array([1.0, 0.0, 0.0], dtype="float32")
_B = np.array([0.0, 1.0, 0.0], dtype="float32")

# Query-eval draws its exemplar from the ground truth, so the invariant is narrower than "labels
# must not reach the method": the method may depend on the exemplar it was *handed*, and must not
# depend on any other label. `seeded_order(5, 0)` is `[2, 4, 3, 0, 1]`, so fold 0 queries with
# interval **index 2** — held identical in both streams below. Every other interval moves.
EXEMPLAR_INDEX = 2
GROOM = [(5, 14), (40, 49), (75, 84), (110, 119), (145, 154)]
GROOM_SHIFTED = [(20, 29), (55, 64), (75, 84), (125, 134), (185, 194)]


def _trajectory():
    emb = np.tile(_B, (N, 1)).astype("float32")
    for start, end in GROOM:
        emb[start : end + 1] = _A
    for start, end in GROOM_SHIFTED:
        emb[start : end + 1] = _A
    frame_idx = np.arange(N, dtype=np.int32)
    t_ns = np.arange(N, dtype=np.int64) * PERIOD
    return emb, frame_idx, t_ns


def _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path):
    """One trajectory, two ground-truth streams over it. Everything a user supplies is shared."""
    emb, frame_idx, t_ns = _trajectory()
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, emb, frame_idx, t_ns)
    recording = Recording(experiment_dir, "recording_001")
    for name, intervals in (("truth_a", GROOM), ("truth_b", GROOM_SHIFTED)):
        lines = ["start_frame,end_frame,label"] + [
            f"{s},{e},groom" for s, e in intervals
        ]
        csv_path = tmp_path / f"{name}.csv"
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        recording.add_stream(name, raw_path=csv_path, importer="behavior_csv")


def _method_output(experiment_dir, method, stream_name, negative_pool="unlabeled"):
    """What the method itself produced — which is a different object on the two paths.

    A windowed matcher's output is its ranked spans and their cost. The per-frame path's
    `predictions` are *not* its output: they come from thresholding the score curve at the best-F1
    operating point, which is computed from the labels, so they move with the ground truth by
    construction and would report a leak in any per-frame method. Its output is the score curve.
    """
    config = {
        "embed_run_id": "emb_01",
        "behaviors": ["groom"],
        "stream_name": stream_name,
        "method": method,
        "folds": 1,
        "seed": 0,
        "n_exemplars": 1,
        "n_negatives": 5,
        "negative_pool": negative_pool,
        "band_s": 0.1,
    }
    result = {r.behavior: r for r in evaluate_query(experiment_dir, config)}["groom"]
    if method in _PERFRAME_METHODS:
        recording_id, _t_ns, score = result.similarity
        return [(recording_id, i, round(float(v), 9)) for i, v in enumerate(score)]
    return [
        (p.recording_id, p.start_t_ns, p.end_t_ns, round(p.score, 9))
        for p in result.predictions
    ]


# `soft_knn` on `labeled_clean` is the one mode that fails, and it is kept rather than deleted
# because it is what every run recorded before `negative_pool` existed used. `strict=True` means
# that if it ever starts passing, the marker fails and has to come off rather than sitting there
# certifying nothing.
_LEAKS_BY_DESIGN = pytest.mark.xfail(
    strict=True,
    reason="the `labeled_clean` pool is sampled from frames known not to be positives, so it "
    "moves with the labels by construction; see CLAUDE.md, 'Labels score a method; they never "
    "build one'",
)


@pytest.mark.parametrize(
    "method,negative_pool",
    [
        ("dtw", "unlabeled"),
        ("clip_pooled_mean", "unlabeled"),
        ("keyframe", "unlabeled"),
        ("soft_knn", "unlabeled"),
        pytest.param("soft_knn", "labeled_clean", marks=_LEAKS_BY_DESIGN),
    ],
)
def test_predictions_do_not_move_when_the_labels_do(
    method, negative_pool, experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """A matcher sees exemplar clips and a trajectory. Moving the *other* labels must change
    nothing about what it retrieves — if it does, ground truth reached the method."""
    _setup(experiment_dir, recording_dir, make_embedding_run, tmp_path)
    # Same trajectory, same exemplar (index 2 is shared), every other label moved.
    first = _method_output(experiment_dir, method, "truth_a", negative_pool)
    second = _method_output(experiment_dir, method, "truth_b", negative_pool)

    assert first == second, (
        f"{method} on the {negative_pool!r} pool moved its predictions when the ground truth "
        f"moved, with every user-supplied "
        f"input held fixed — a label reached the method rather than only the metric. "
        f"{len(first)} vs {len(second)} output rows; first difference at "
        f"{next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), 'length')}"
    )


def test_the_permutation_would_catch_a_leak(
    experiment_dir, recording_dir, make_embedding_run, tmp_path
):
    """Guard the guard: a scan that cannot fail passes vacuously forever.

    The two label sets have to actually differ where it counts, or the comparison above proves
    nothing. Four of the five groom intervals move; the exemplar's does not.
    """
    from detail.analysis.query_eval import seeded_order

    assert seeded_order(len(GROOM), 0)[0] == EXEMPLAR_INDEX, (
        "fold 0 no longer queries with the interval this fixture pins — re-pin it, or the two "
        "runs below are comparing different exemplars and prove nothing"
    )
    assert (
        GROOM[EXEMPLAR_INDEX] == GROOM_SHIFTED[EXEMPLAR_INDEX]
    ), "the exemplar must be shared"
    moved = sum(
        a != b
        for i, (a, b) in enumerate(zip(GROOM, GROOM_SHIFTED))
        if i != EXEMPLAR_INDEX
    )
    assert moved == len(GROOM) - 1, f"only {moved} of the other intervals move"
