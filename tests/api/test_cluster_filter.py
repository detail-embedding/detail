"""Time in session, the filter over each recording's whole video (§22, ``api/cluster_filter.py``).

The edges the rule is made of, each its own case: recordings of different lengths (the range is a
fraction of *each* recording), a multi-part recording (the span is first part's first frame to
last part's last), a run straddling either end of the range (a node's share counts only the time
inside; a clip matches by its midpoint), the whole range (no filter), a range too small to hold
anything, a recording with no timestamp map (it never matches), and the AND with a region and a
metadata condition. Built on a hand-made node cache so every expected number is arithmetic on
the page, not a re-run of the code under test.
"""

import dataclasses

import numpy as np
import pytest

from detail.analysis import node_cache as tables_lib
from detail.api import cluster_filter
from detail.storage import node_cache as store
from detail.storage import recording as recording_store

S = 1_000_000_000
#: recording_000 is 100 s long, recording_001 400 s: the range is a fraction of each.
SPANS = {"recording_000": (0, 100 * S), "recording_001": (0, 400 * S)}


def _view(positions: bool = False):
    """Three leaves, ``(0, 1) -> 3`` then ``(3, 2) -> 4``. Leaf 0 is ``[0, 40) s`` and leaf 1
    ``[40, 100) s`` of recording_000; leaf 2 is the whole of recording_001, ``[0, 400) s``.

    With ``positions``, each leaf-recording's members sit in one known cell: leaf 0 in the
    left half of the frame (cell (0, 0)), leaves 1 and 2 in the right (cell (15, 0)), one member
    per second.
    """
    from detail.api.node_view import NodeView
    from detail.core.dataclasses import ClusterTree

    tree = ClusterTree(
        children=np.array([[0, 1], [3, 2]], dtype=np.int32),
        height=np.array([1.0, 2.0]),
        size=np.array([1, 1, 1, 2, 3], dtype=np.int32),
        n_rows=np.array([1, 1, 1, 2, 3], dtype=np.int32),
        leaf_of_row=np.array([0, 1, 2], dtype=np.int32),
        rung_merges=(0,),
    )
    rec_time = np.array([[40, 0], [60, 0], [0, 400]], dtype=np.int64) * S
    pos = np.zeros((3, 2, 16, 16), dtype=np.uint32)
    if positions:
        pos[0, 0, 0, 0] = 40
        pos[1, 0, 0, 15] = 60
        pos[2, 1, 0, 15] = 400
    tables = tables_lib.LeafTables(
        recording_ids=tuple(SPANS),
        leaf_time_ns=rec_time.sum(axis=1),
        leaf_members=rec_time.sum(axis=1) // S,
        leaf_rec_time_ns=rec_time,
        leaf_rec_pos=pos,
        run_recording=np.array([0, 0, 1], dtype=np.int32),
        run_subject=np.zeros(3, dtype=np.int64),
        run_start_ns=np.array([0, 40, 0], dtype=np.int64) * S,
        run_end_ns=np.array([40, 100, 400], dtype=np.int64) * S,
        run_leaf=np.array([0, 1, 2], dtype=np.int32),
        separation=np.array([np.nan, np.nan, np.nan, 0.4, 0.2], dtype=np.float32),
        run_rows=np.array([40, 60, 400], dtype=np.int32),
        run_last_ns=np.array([39, 99, 399], dtype=np.int64) * S,
    )
    return NodeView(store.NodeCache(leaves=tables, built_from={}), tree)


def _clips(starts_s, ends_s, recordings, cx=None):
    """A ``ClipTable`` of the given clips, in seconds, positions unknown unless ``cx``."""
    from detail.api.node_view import ClipTable

    n = len(starts_s)
    cx = np.full(n, np.nan) if cx is None else np.asarray(cx, dtype=np.float64)
    return ClipTable(
        recording_ids=tuple(SPANS),
        recording=np.asarray(recordings, dtype=np.int32),
        subject=np.zeros(n, dtype=np.int64),
        start_ns=(np.asarray(starts_s, dtype=np.float64) * S).astype(np.int64),
        end_ns=(np.asarray(ends_s, dtype=np.float64) * S).astype(np.int64),
        cx=cx.astype(np.float32),
        cy=np.where(np.isfinite(cx), 0.01, np.nan).astype(np.float32),
        members=np.ones(n, dtype=np.int64),
        typicality=np.full(n, np.nan),
        leaf=np.zeros(n, dtype=np.int64),
    )


# --- the selection ------------------------------------------------------------------------


def test_the_whole_session_is_no_filter_and_a_bad_range_is_refused():
    whole = cluster_filter.Selection(session=(0, 1))
    assert whole.session is None and whole.is_empty and whole.keys() == []
    part = cluster_filter.Selection(session=(0.25, 0.5))
    assert not part.is_empty
    assert part.keys() == [cluster_filter.SESSION]
    assert part.without(cluster_filter.SESSION).is_empty
    both = cluster_filter.Selection(session=(0.25, 0.5), cells=frozenset({(0, 0)}))
    assert both.only(cluster_filter.SESSION) == part
    for bad in ((0.5, 0.25), (-0.1, 0.5), (0.2, 1.5)):
        with pytest.raises(ValueError, match="0 <= lo <= hi <= 1"):
            cluster_filter.Selection(session=bad)


def test_a_session_range_without_spans_is_refused():
    view = _view()
    with pytest.raises(ValueError, match="session span"):
        view.match_shares(cluster_filter.Selection(session=(0.25, 0.5)))
    with pytest.raises(ValueError, match="session span"):
        cluster_filter.clip_matches(
            cluster_filter.Selection(session=(0.25, 0.5)), _clips([0], [1], [0])
        )


# --- shares: only the time inside counts --------------------------------------------------


def test_a_share_counts_only_the_time_inside_each_recording_s_own_range():
    """``(0.25, 0.5)`` is 25–50 s of the 100 s recording and 100–200 s of the 400 s one. Leaf 0
    (0–40 s) straddles the low end: 15 of its 40 s are inside. Leaf 1 (40–100 s) straddles the
    high end: 10 of 60. Leaf 2 holds 100 of its 400. Normalising by absolute time instead would
    put leaf 2's window at 25–50 s and give it 25 of 400."""
    view = _view()
    selection = cluster_filter.Selection(session=(0.25, 0.5))
    matched, total = view.match_counts(selection, spans=SPANS)
    assert matched[:3].tolist() == pytest.approx([15 * S, 10 * S, 100 * S])
    assert total.tolist() == pytest.approx([40 * S, 60 * S, 400 * S, 100 * S, 500 * S])
    shares = view.match_shares(selection, spans=SPANS)
    assert shares.tolist() == pytest.approx([15 / 40, 10 / 60, 100 / 400, 0.25, 0.25])


def test_a_share_with_positions_is_counted_in_members_and_trimmed_the_same():
    view = _view(positions=True)
    shares = view.match_shares(
        cluster_filter.Selection(session=(0.25, 0.5)), spans=SPANS
    )
    assert shares.tolist() == pytest.approx([15 / 40, 10 / 60, 100 / 400, 0.25, 0.25])


def test_a_range_that_holds_nothing_and_a_recording_with_no_span():
    """A range inside a leaf's gap matches none of it; a recording missing from ``spans`` (no
    timestamp map) matches no session range at all, as a missing value matches no range.
    """
    view = _view()
    tiny = cluster_filter.Selection(session=(0.0, 0.001))
    shares = view.match_shares(tiny, spans=SPANS)
    # 0–0.1 s of recording_000 and 0–0.4 s of recording_001.
    assert shares[0] == pytest.approx(0.1 / 40) and shares[1] == 0.0
    assert shares[2] == pytest.approx(0.4 / 400)
    only_first = {"recording_000": SPANS["recording_000"]}
    late = cluster_filter.Selection(session=(0.5, 1.0))
    shares = view.match_shares(late, spans=only_first)
    assert shares[2] == 0.0, "recording_001 has no span, so none of it is in any range"
    assert shares[1] == pytest.approx(50 / 60)


def test_a_session_range_ands_with_a_region_and_with_metadata():
    """Leaf 0 is all in cell (0, 0) and leaves 1, 2 in (15, 0). With the left half of the frame
    picked, only leaf 0 can match, and only its 15 s inside the range; with a metadata value
    only recording_001 has, only leaf 2 can, and only its 100 s."""
    view = _view(positions=True)
    left = frozenset((x, y) for x in range(8) for y in range(16))
    both = cluster_filter.Selection(session=(0.25, 0.5), cells=left)
    matched, total = view.match_counts(both, spans=SPANS)
    assert matched[:3].tolist() == pytest.approx([15, 0, 0])
    metadata = {"recording_000": {"box": "a"}, "recording_001": {"box": "b"}}
    with_meta = cluster_filter.Selection(
        session=(0.25, 0.5), values={"box": frozenset({"b"})}
    )
    matched, _ = view.match_counts(with_meta, metadata, spans=SPANS)
    assert matched[:3].tolist() == pytest.approx([0, 0, 100])


def test_the_whole_session_leaves_every_share_where_it_was():
    view = _view(positions=True)
    region = cluster_filter.Selection(cells=frozenset({(0, 0)}))
    with_whole = cluster_filter.Selection(cells=frozenset({(0, 0)}), session=(0, 1))
    assert np.array_equal(
        view.match_shares(region), view.match_shares(with_whole, spans=SPANS)
    )


# --- clips: the midpoint decides ----------------------------------------------------------


def test_a_clip_matches_when_its_midpoint_is_inside():
    """Range 25–50 s of recording_000 and 100–200 s of recording_001. A clip 20–32 s (midpoint
    26) is in though it starts before the range; 20–28 s (midpoint 24) is out though most of a
    second of it is inside; 45–60 s (midpoint 52.5) is out though it starts inside. The 400 s
    recording's 90–150 s (midpoint 120) is in, and its 30–40 s is out: in the 100 s recording that
    stretch would be in, which is what normalising per recording means."""
    clips = _clips(
        [20, 20, 45, 90, 30],
        [32, 28, 60, 150, 40],
        [0, 0, 0, 1, 1],
    )
    selection = cluster_filter.Selection(session=(0.25, 0.5))
    got = cluster_filter.clip_matches(selection, clips, spans=SPANS)
    assert got.tolist() == [True, False, False, True, False]
    missing = cluster_filter.clip_matches(
        selection, clips, spans={"recording_000": SPANS["recording_000"]}
    )
    assert missing.tolist() == [True, False, False, False, False]


def test_a_clip_on_the_line_between_two_ranges_is_in_the_later_one_only():
    """Ranges are half-open, ``[lo, hi)``: a clip whose midpoint is exactly at 50 s of the 100 s
    recording is in ``[0.5, 1]`` and not ``[0, 0.5]``, and one a nanosecond earlier the other way
    round. Each of the five clips is in exactly one of the two halves."""
    clips = _clips([50, 49, 0, 99, 40], [50, 51, 1, 100, 60], [0, 0, 0, 0, 0])
    clips = dataclasses.replace(
        clips, end_ns=clips.end_ns - np.array([0, 2, 0, 0, 0], dtype=np.int64)
    )
    lower = cluster_filter.clip_matches(
        cluster_filter.Selection(session=(0.0, 0.5)), clips, spans=SPANS
    )
    upper = cluster_filter.clip_matches(
        cluster_filter.Selection(session=(0.5, 1.0)), clips, spans=SPANS
    )
    assert lower.tolist() == [False, True, True, False, False]
    assert upper.tolist() == [True, False, False, True, True]


def test_two_ranges_meet_at_one_nanosecond_when_the_line_falls_between_two():
    """A recording 101 ns long puts 0.5 at 50.5 ns. The end of ``[0, 0.5]`` and the start of
    ``[0.5, 1]`` are rounded by one rule, so they meet at one nanosecond: a midpoint at 50 or 51
    is in exactly one half. Rounding the end up and the start down put 50 in both."""
    spans = {"recording_000": (0, 101), "recording_001": (0, 400 * S)}
    clips = _clips([0, 0, 0], [0, 0, 0], [0, 0, 0])
    clips = dataclasses.replace(
        clips,
        start_ns=np.array([50, 51, 49], dtype=np.int64),
        end_ns=np.array([50, 51, 49], dtype=np.int64),
    )
    lower = cluster_filter.clip_matches(
        cluster_filter.Selection(session=(0.0, 0.5)), clips, spans=spans
    )
    upper = cluster_filter.clip_matches(
        cluster_filter.Selection(session=(0.5, 1.0)), clips, spans=spans
    )
    assert not (lower & upper).any()
    assert (lower | upper).all()


def test_a_run_of_no_length_on_the_line_counts_in_one_range():
    """``session_shares`` counts a zero-length run whole when its start is inside, by the same
    half-open rule: one at exactly 40 s is in ``[0.4, 1]`` and not in ``[0, 0.4]``."""
    view = _view()
    t = view._cache.leaves
    from detail.api.node_view import NodeView

    point = dataclasses.replace(
        t, run_end_ns=np.array([40, 100, 400], dtype=np.int64) * S
    )
    point = dataclasses.replace(
        point, run_start_ns=np.array([40, 40, 0], dtype=np.int64) * S
    )
    view = NodeView(store.NodeCache(leaves=point, built_from={}), view.tree)
    early = cluster_filter.session_window(
        cluster_filter.Selection(session=(0.0, 0.4)), view.recording_ids, SPANS
    )
    late = cluster_filter.session_window(
        cluster_filter.Selection(session=(0.4, 1.0)), view.recording_ids, SPANS
    )
    assert view.session_shares(early)[0, 0] == 0.0
    assert view.session_shares(late)[0, 0] == 1.0


def test_a_clip_must_meet_the_session_range_and_every_other_condition():
    clips = _clips([20, 20], [32, 32], [0, 0], cx=[0.01, 0.99])
    left = frozenset((x, y) for x in range(8) for y in range(16))
    selection = cluster_filter.Selection(session=(0.25, 0.5), cells=left)
    assert cluster_filter.clip_matches(selection, clips, spans=SPANS).tolist() == [
        True,
        False,
    ]
    nobody = cluster_filter.Selection(session=(0.25, 0.5), recording_ids=frozenset())
    assert not cluster_filter.clip_matches(nobody, clips, spans=SPANS).any()


# --- the spans: first frame to last, every part -------------------------------------------


def _recording(root, recording_id, part_lengths, fps=30.0, start_ns=0):
    """A recording.yaml with one part per length, each part's map continuing from the last
    (Tenet 8: ``t_ns`` runs from the start of the first part)."""
    sdir = root / "recordings" / recording_id
    files = []
    offset = start_ns
    for i, n in enumerate(part_lengths):
        rel = f"video/timestamps/generated/part_{i + 1:03d}.csv"
        (sdir / rel).parent.mkdir(parents=True, exist_ok=True)
        t_ns = offset + np.arange(n, dtype=np.int64) * int(1e9 / fps)
        recording_store.write_timestamps_csv(sdir / rel, t_ns)
        offset = int(t_ns[-1]) + int(1e9 / fps)
        files.append(
            {
                "index": i,
                "file": f"video/part_{i + 1:03d}.mp4",
                "timestamps": {"generated": rel, "user": None, "active": "generated"},
            }
        )
    recording_store.write_recording(
        sdir,
        {
            "id": recording_id,
            "name": recording_id,
            "created_at": "x",
            "video": {"fps_actual": fps, "files": files},
        },
    )
    return sdir


def test_spans_run_from_the_first_part_s_first_frame_to_the_last_part_s_last(tmp_path):
    """Two parts of 300 and 150 frames at 30 fps: the span is frame 0 to frame 449 of the whole
    recording, 0 to 449/30 s — not the first part's 299/30 s. A recording with no map has none.
    """
    step = int(1e9 / 30)
    _recording(tmp_path, "recording_001", [300, 150])
    _recording(tmp_path, "recording_002", [90])
    (tmp_path / "recordings" / "recording_003").mkdir(parents=True)
    spans = cluster_filter.session_spans(
        tmp_path, ["recording_001", "recording_002", "recording_003"]
    )
    assert spans == {
        "recording_001": (0, 449 * step),
        "recording_002": (0, 89 * step),
    }


def test_a_multi_part_recording_s_range_reaches_into_its_later_parts(tmp_path):
    """The last quarter of a two-part recording is in its second part: a clip there matches,
    and a clip at the end of the first part does not."""
    _recording(tmp_path, "recording_000", [300, 300])
    spans = cluster_filter.session_spans(tmp_path, ["recording_000"])
    clips = _clips([18, 9], [19, 10], [0, 0])
    clips = dataclasses.replace(clips, recording_ids=("recording_000",))
    late = cluster_filter.Selection(session=(0.75, 1.0))
    assert cluster_filter.clip_matches(late, clips, spans=spans).tolist() == [
        True,
        False,
    ]
