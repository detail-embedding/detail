"""A leaf run covers exactly its members' time, for every cluster method (§22, the node cache).

**The class: a duration that already includes a span, added to the span's end.** A member's
``time_ns`` (``MemberPlaces``) is the whole of what it stands for: a frame's gap to the next frame,
a bout's span **plus** one frame step. ``leaf_runs`` ended a run at its last member's *end* plus
that time. For a frame the two agree, because a frame starts where it ends, and every fixture the
node cache was first written against was a frame method; for a bout it counts the span twice. On a
twelve-recording ``dtw_finch`` fit, 0.37 s bouts ended 0.4 s past their last frame, on top of the
next bout, and up to 2 s past a recording's last frame. Nothing failed: the runs were well-formed,
every share was a plausible fraction, and the labels ethogram simply drew each bout twice as long.
A run now ends at its last member's **start** plus its time (``analysis/node_cache.covered_end``).

The guard is data-driven over every registered cluster method, pose (hidden, still guarded) and
temporal, each fitted directly on a small trajectory with a tracking gap, so a method added later
is covered the day it registers. The properties are the method-independent ones a reader relies
on, never the formula: a run's covered time is its members' summed time, a unit's runs never
overlap, and no run ends more than one frame step past its recording's last frame. A rule-1 cache
on disk is reported stale and rebuilt (``tests/api/test_node_cache.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from detail.analysis import node_cache as tables_lib
from detail.cluster_methods import import_pose_cluster_methods
from detail.cluster_methods import import_temporal_cluster_methods
from detail.core.dataclasses import MemberPlaces
from detail.core.registries import POSE_CLUSTER_METHODS
from detail.core.registries import TEMPORAL_CLUSTER_METHODS

STEP = 33_333_333
GAP_FRAMES = 40
BOUT_ROWS = 30
#: The small-trajectory settings of ``tests/conftest.py``'s temporal fit, applied where a method
#: has the field: the shipped defaults are tuned for real recordings.
_SMALL = {
    "window_s": 0.05,
    "min_len_s": 0.1,
    "band_s": 0.05,
    "penalty": 1.0,
    "min_size_s": 0.05,
    "min_run_s": 0.1,
    "codewords": 30,
    "n_groups": 40,
}

import_pose_cluster_methods()
import_temporal_cluster_methods()
_METHODS = sorted(
    [("pose", m) for m in POSE_CLUSTER_METHODS]
    + [("temporal", m) for m in TEMPORAL_CLUSTER_METHODS]
)


def _trajectories():
    """Two recordings of twelve 30-row bouts of three kinds, each with a tracking gap halfway."""
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(3, 2, 8))
    out = []
    for recording in ("recording_001", "recording_002"):
        rows = []
        for i in range(12):
            ramp = np.linspace(0.0, 1.0, BOUT_ROWS)[:, None]
            path = anchors[i % 3, 0] * (1 - ramp) + anchors[i % 3, 1] * ramp
            rows.append(path + 0.05 * rng.normal(size=path.shape))
        x = np.vstack(rows).astype(np.float32)
        t_ns = np.arange(len(x), dtype=np.int64) * STEP
        t_ns[len(x) // 2 :] += GAP_FRAMES * STEP
        out.append((recording, 0, x, t_ns))
    return out


@pytest.fixture(scope="module", params=_METHODS, ids=lambda p: p[1])
def fitted(request):
    """``(method, fit, places, scoring, tables, trajectories)`` for one registered method."""
    kind, method_id = request.param
    registry = POSE_CLUSTER_METHODS if kind == "pose" else TEMPORAL_CLUSTER_METHODS
    cls = registry[method_id]
    params = cls.config_model().model_dump() if cls.config_model else {}
    params.update({k: v for k, v in _SMALL.items() if k in params})
    method = cls()
    trajectories = _trajectories()
    fit = method.fit(trajectories, params)
    places = method.member_places(fit)
    scoring = method.clip_scoring(fit, params, trajectories)
    tree = method.hierarchy_of(fit)
    tables = tables_lib.leaf_tables(
        places,
        method.member_leaves(fit),
        tree.n_leaves,
        method.node_separations(fit),
        scoring=scoring,
    )
    return method, fit, places, scoring, tables, trajectories


def _overlapping(recording, subject, start, end) -> int:
    """How many runs start before the previous run of their unit has ended."""
    order = np.lexsort((start, subject, recording))
    r, s, a, b = (np.asarray(v)[order] for v in (recording, subject, start, end))
    same = (r[1:] == r[:-1]) & (s[1:] == s[:-1])
    return int((same & (a[1:] < b[:-1])).sum())


def _uncovered(tables, places) -> int:
    """How many leaves' runs cover a different time than their members stand for."""
    covered = np.bincount(
        tables.run_leaf.astype(np.int64),
        weights=(tables.run_end_ns - tables.run_start_ns).astype(np.float64),
        minlength=len(tables.leaf_time_ns),
    )
    return int((covered != tables.leaf_time_ns.astype(np.float64)).sum())


def test_every_method_s_runs_cover_exactly_their_members_time(fitted):
    """A run's covered time is its members' summed time: per leaf, and run by run."""
    _method, _fit, places, _scoring, tables, _traj = fitted
    assert len(tables.run_leaf), "the fit produced no runs"
    assert _uncovered(tables, places) == 0
    # Run by run: the members in run order (the order ``leaf_runs`` cuts them in).
    order = np.lexsort((places.start_ns, places.subject, places.recording_index))
    time = np.asarray(places.time_ns, dtype=np.int64)[order]
    heads = np.concatenate([[0], np.cumsum(tables.run_rows)[:-1]]).astype(np.int64)
    per_run = np.add.reduceat(time, heads)
    assert (tables.run_end_ns - tables.run_start_ns).tolist() == per_run.tolist()


def test_a_run_ends_one_step_after_its_last_member(fitted):
    """Every run ends at its last member's own end plus one frame step: a frame's gap to the next
    (the fixture's frames are regular, so exactly ``STEP``), a bout's last frame plus its step.
    """
    _method, _fit, _places, _scoring, tables, _traj = fitted
    past = tables.run_end_ns - tables.run_last_ns
    assert past.tolist() == [STEP] * len(past)


def test_consecutive_runs_of_one_unit_never_overlap(fitted):
    """Consecutive bouts touch, one ending where the next begins; under the old rule each bout ran
    a whole span into the next."""
    _method, _fit, _places, _scoring, t, _traj = fitted
    assert (
        _overlapping(t.run_recording, t.run_subject, t.run_start_ns, t.run_end_ns) == 0
    )


def test_no_run_ends_past_its_recording_s_last_frame_plus_one_step(fitted):
    _method, _fit, places, _scoring, tables, trajectories = fitted
    last = {rid: int(t_ns[-1]) for rid, _s, _x, t_ns in trajectories}
    ends = {
        places.recording_ids[int(r)]: int(
            tables.run_end_ns[tables.run_recording == r].max()
        )
        for r in np.unique(tables.run_recording)
    }
    assert ends, "no runs"
    for rid, end in ends.items():
        assert end <= last[rid] + STEP, (rid, (end - last[rid]) / 1e9)


def test_only_a_method_whose_members_have_a_span_leaves_its_runs_unjoined(fitted):
    """``NodeView.stale_reason`` reports a rule-1 cache only when its clips are unjoined, on the
    ground that only a bout method's members have a span and every bout method's clips are
    unjoined. That ground is checked here, per method, so a frame method that stopped joining or a
    bout method that started would fail rather than leave a wrong cache unreported."""
    _method, _fit, places, scoring, _tables, _traj = fitted
    spans = bool((np.asarray(places.end_ns) > np.asarray(places.start_ns)).any())
    assert spans == (not scoring.join)


def _two_bouts() -> MemberPlaces:
    """Two touching bouts of 10 frames each, frame 0 to 9 and 10 to 19, at ``STEP``."""
    start = np.array([0, 10 * STEP], dtype=np.int64)
    end = start + 9 * STEP
    return MemberPlaces(
        recording_ids=("recording_001",),
        recording_index=np.zeros(2, dtype=np.int64),
        subject=np.zeros(2, dtype=np.int64),
        start_ns=start,
        end_ns=end,
        time_ns=(end - start) + STEP,
    )


def test_a_bout_s_run_ends_one_step_after_its_last_frame():
    """By hand: frames 0-9 cover ``[0, 10·STEP)``, frames 10-19 ``[10·STEP, 20·STEP)``."""
    places = _two_bouts()
    runs = tables_lib.leaf_runs(places, np.array([0, 1]), join=False)
    assert runs[2].tolist() == [0, 10 * STEP]
    assert runs[3].tolist() == [10 * STEP, 20 * STEP]


def test_the_overlap_and_coverage_checks_would_catch_the_old_rule():
    """The predicates above, handed the runs the old rule made (end + time) for two touching
    bouts: they overlap, and they cover more than their members' time."""
    places = _two_bouts()
    old_end = places.end_ns + places.time_ns
    zeros = np.zeros(2, dtype=np.int64)
    assert _overlapping(zeros, zeros, places.start_ns, old_end) == 1
    assert (
        _overlapping(zeros, zeros, places.start_ns, places.start_ns + places.time_ns)
        == 0
    )

    class _Old:
        run_leaf = np.array([0, 1], dtype=np.int32)
        run_start_ns = places.start_ns
        run_end_ns = old_end
        leaf_time_ns = places.time_ns

    assert _uncovered(_Old, places) == 2
