"""Time in session across twelve recordings of very different lengths (§22, ``api/cluster_filter``).

The rule is a fraction of **each recording's own** whole video, so the class of bug it invites is
one that is invisible on recordings of one length: a range normalised by the wrong length (a part's,
the longest recording's, the run's covered time), a multi-part recording whose later parts restart
the clock, or an edge that loses or double counts what sits on it. ``tests/api/test_cluster_filter.py``
pins each edge on a hand-made cache of two recordings; this runs the same rule over a real temporal
fit of twelve recordings from 8 s to 120 min (``tests/session_lengths.py``), two of them imported in
parts through the real importer, one on a user clock that starts at 7 s, one with dropped frames, one
with clips only at its start and one only at its end.

Every expected answer here is plain arithmetic in this file over the clips' times, the fit's own
bouts (``_bouts``: each covers its first frame to one frame step past its last) and the spans read
straight off the maps the fixture wrote (``layout.spans``), in exact fractions: never a call to
``session_window`` / ``session_shares``, and never the cache's run ends, which are under test
here too. A range is half-open, ``[lo, hi)``, so ``[0, x]`` and ``[x, 1]`` split every clip and
every second between them; the range ending at ``1`` runs to the end of the recording, so a run
whose covered time reaches one step past the last frame is wholly inside it: without that, the
end of a session could never hold all of itself.

The node cache once ended each bout's run a whole span late (its last frame plus the bout's
whole time), so these 0.37 s bouts ran 0.4 s past their ends, over the next bout and up to 2 s
past a recording's last frame; ``test_every_bout_covers_its_frames_and_one_step_and_no_more``
pins that, and the shares below are computed from the bouts rather than from the runs.
"""

from fractions import Fraction

import numpy as np
import pytest

from detail.api import cluster_filter
from detail.api.cluster_filter import Selection

#: The ranges every check runs over: the extremes, a sliver at each end, the middle, the halves.
RANGES = (
    (0.0, 0.01),
    (0.0, 0.1),
    (0.45, 0.55),
    (0.9, 1.0),
    (0.99, 1.0),
    (0.0, 0.5),
    (0.5, 1.0),
    (0.25, 0.75),
    (0.0, 0.99),
    (0.01, 1.0),
)


def _view(session_lengths):
    """``(spans, view, layout)``: ``spans`` read the way the page reads them
    (``session_spans``, the code under test), ``layout.spans`` the maps as written (the answer).
    """
    experiment, run_id, layout = session_lengths
    view = experiment.temporal_clustering_run(run_id).node_view()
    return (
        cluster_filter.session_spans(experiment.dir, view.recording_ids),
        view,
        layout,
    )


def _fraction(t_ns: int, span) -> Fraction:
    first, last = span
    return Fraction(int(t_ns) - first, last - first)


def _clip_fractions(clips, layout) -> list[Fraction]:
    """Each clip's midpoint as an exact fraction of its own recording's whole video."""
    out = []
    for i in range(len(clips.start_ns)):
        rid = clips.recording_ids[int(clips.recording[i])]
        mid = (int(clips.start_ns[i]) + int(clips.end_ns[i])) // 2
        out.append(_fraction(mid, layout.spans[rid]))
    return out


def _expected_clips(fractions, lo, hi) -> np.ndarray:
    """A clip matches when its midpoint is in ``[lo, hi)`` of its recording, ``hi = 1`` reaching
    the end (half-open, §22)."""
    lo, hi = Fraction(str(lo)), Fraction(str(hi))
    return np.array([lo <= f and (f < hi or hi == 1) for f in fractions], dtype=bool)


def _bouts(session_lengths) -> list[tuple[int, int, int, int]]:
    """``(leaf, recording index, start_ns, covered end)`` per bout, from the fit itself: a bout
    covers its first frame to one frame step past its last, its step its span over its frame
    count (§22 *Member time*). Every bout here is several frames, so each has its own step.
    """
    experiment, run_id, _layout = session_lengths
    run = experiment.temporal_clustering_run(run_id)
    fit = run._load_fit()
    leaves = run._method().member_leaves(fit)
    out = []
    for i in range(len(fit.start_t_ns)):
        start, last = int(fit.start_t_ns[i]), int(fit.end_t_ns[i])
        frames = int(fit.end_frames[i]) - int(fit.start_frames[i])
        assert frames > 0, "a one-frame bout: its step is not its own"
        step = (last - start) // frames
        out.append((int(leaves[i]), int(fit.recording_index[i]), start, last + step))
    return out


def _leaves_under(tree, node: int) -> list[int]:
    n_leaves = tree.n_leaves
    if node < n_leaves:
        return [node]
    a, b = (int(c) for c in tree.children[node - n_leaves])
    return _leaves_under(tree, a) + _leaves_under(tree, b)


def _expected_shares(view, layout, bouts, lo, hi) -> np.ndarray:
    """Every node's matching share, by hand: each bout's covered time (``_bouts``) inside its own
    recording's ``[lo, hi]`` (a range from ``0`` or to ``1`` is open at that end), over its covered
    time, per leaf and recording; each leaf-recording's members weighted by that share.
    """
    t = view._cache.leaves
    n_leaves, n_recs = t.leaf_rec_time_ns.shape
    inside = [[Fraction(0)] * n_recs for _ in range(n_leaves)]
    covered = [[0] * n_recs for _ in range(n_leaves)]
    lo, hi = Fraction(str(lo)), Fraction(str(hi))
    for leaf, rec, s, e in bouts:
        first, last = layout.spans[view.recording_ids[rec]]
        a = s if lo == 0 else max(s, first + lo * (last - first))
        b = e if hi == 1 else min(e, first + hi * (last - first))
        inside[leaf][rec] += max(Fraction(0), b - a)
        covered[leaf][rec] += e - s
    members = t.leaf_rec_pos.sum(axis=(2, 3), dtype=np.int64)
    out = []
    for node in range(view.tree.n_nodes):
        matched = total = Fraction(0)
        for leaf in _leaves_under(view.tree, node):
            for rec in range(n_recs):
                w = int(members[leaf, rec])
                total += w
                if covered[leaf][rec]:
                    matched += w * inside[leaf][rec] / covered[leaf][rec]
        out.append(float(matched / total) if total else np.nan)
    return np.array(out)


# --- the fixture is what it claims ---------------------------------------------------------


def test_the_recordings_differ_in_length_by_three_orders_of_magnitude(session_lengths):
    """Twelve recordings, 8 s to 2 h, every one with clips; the spans the page reads are the ones
    the maps were written with."""
    spans, view, layout = _view(session_lengths)
    assert len(layout.spans) == 12
    lengths = {rid: (last - first) / 1e9 for rid, (first, last) in layout.spans.items()}
    assert min(lengths.values()) < 8 and max(lengths.values()) > 7199
    assert spans == layout.spans
    clips = view.clips(view.root)
    assert set(np.unique(clips.recording)) == set(range(12)), "a recording has no clip"
    by_name = {layout.names[r]: r for r in layout.spans}
    assert layout.spans[by_name["two_minutes_user_clock"]][0] == 7 * 10**9


def test_an_imported_recording_s_later_parts_continue_its_clock(session_lengths):
    """Through ``Recording.add_video_part``: part 1 starts at 0 and each later part starts one
    frame after the last one ended, so the span is the whole recording, not a part. If a part's
    map restarted at 0 the span would end inside the last part's own length."""
    spans, view, layout = _view(session_lengths)
    imported = [r for r, spec in layout.specs.items() if spec.imported]
    assert {len(layout.part_maps[r]) for r in imported} == {2, 3}
    for rid in imported:
        maps = layout.part_maps[rid]
        assert maps[0][0] == 0
        for before, after in zip(maps, maps[1:]):
            step = (int(before[-1]) - int(before[0])) / (len(before) - 1)
            assert int(after[0]) - int(before[-1]) == pytest.approx(step, abs=1)
        whole_s = sum(layout.specs[rid].part_s)
        assert layout.spans[rid][1] / 1e9 == pytest.approx(whole_s, abs=0.05)


def test_a_clip_in_a_later_part_sits_at_its_place_in_the_whole_recording(
    session_lengths,
):
    """Every imported recording has clips past its first part; each one's fraction is past the
    first part's share of the recording, and the last range it is in agrees."""
    spans, view, layout = _view(session_lengths)
    clips = view.clips(view.root)
    fractions = _clip_fractions(clips, layout)
    for rid, spec in layout.specs.items():
        if not spec.imported:
            continue
        rec = view.recording_ids.index(rid)
        first_part_end = int(layout.part_maps[rid][0][-1])
        later = [
            i
            for i in range(len(fractions))
            if clips.recording[i] == rec and clips.start_ns[i] > first_part_end
        ]
        assert later, f"{spec.name} has no clip past its first part"
        boundary = _fraction(first_part_end, layout.spans[rid])
        assert all(fractions[i] > boundary for i in later)
        late = Selection(session=(float(boundary) + 0.01, 1.0))
        matched = cluster_filter.clip_matches(late, clips, spans=spans)
        assert all(matched[i] for i in later)


# --- clips: the midpoint, each recording normalised by its own length ----------------------


@pytest.mark.parametrize("lo, hi", RANGES)
def test_a_clip_matches_by_its_midpoint_in_its_own_recording(session_lengths, lo, hi):
    spans, view, layout = _view(session_lengths)
    clips = view.clips(view.root)
    expected = _expected_clips(_clip_fractions(clips, layout), lo, hi)
    actual = cluster_filter.clip_matches(
        Selection(session=(lo, hi)), clips, spans=spans
    )
    assert actual.tolist() == expected.tolist()
    assert 0 < expected.sum() < len(expected), "the range must split the clips"


def test_the_first_and_last_clip_of_every_recording_are_at_its_ends(session_lengths):
    """The first clip of a recording that starts with a burst is in its first 1% only when its
    midpoint is, whatever the recording's length; likewise the last clip and its last 1%. Long
    recordings have them there (a 0.4 s bout is inside 1% of 30 min), the 8 s one does not.
    """
    spans, view, layout = _view(session_lengths)
    clips = view.clips(view.root)
    fractions = _clip_fractions(clips, layout)
    start = cluster_filter.clip_matches(
        Selection(session=(0.0, 0.01)), clips, spans=spans
    )
    end = cluster_filter.clip_matches(
        Selection(session=(0.99, 1.0)), clips, spans=spans
    )
    seen_start, seen_end = set(), set()
    for rid, spec in layout.specs.items():
        rec = view.recording_ids.index(rid)
        mine = [i for i in range(len(fractions)) if clips.recording[i] == rec]
        first = min(mine, key=lambda i: fractions[i])
        last = max(mine, key=lambda i: fractions[i])
        assert bool(start[first]) == (fractions[first] <= Fraction(1, 100))
        assert bool(end[last]) == (fractions[last] >= Fraction(99, 100))
        long = (layout.spans[rid][1] - layout.spans[rid][0]) > 600 * 10**9
        if long and 0.0 in spec.bursts:
            assert start[first], spec.name
            seen_start.add(spec.name)
        if long and 1.0 in spec.bursts:
            assert end[last], spec.name
            seen_end.add(spec.name)
    assert seen_start and seen_end
    eight = view.recording_ids.index(
        next(r for r, s in layout.specs.items() if s.name == "eight_seconds")
    )
    assert not any(start[i] or end[i] for i in np.flatnonzero(clips.recording == eight))


def test_the_same_fraction_matches_and_the_same_absolute_time_does_not(session_lengths):
    """The first 1% of the 2 h recording is its first 72 s, and holds its clips at 0-3 s; the
    first 1% of the 45 s one is 0.45 s, and its clips at the same 0-3 s are outside it. The middle
    tenth holds the 45 s recording's clips at ~22 s and the 2 h one's at ~3600 s."""
    spans, view, layout = _view(session_lengths)
    clips = view.clips(view.root)
    by_name = {s.name: view.recording_ids.index(r) for r, s in layout.specs.items()}
    mid_s = (clips.start_ns + clips.end_ns) // 2 / 1e9

    def matched(lo, hi):
        return cluster_filter.clip_matches(
            Selection(session=(lo, hi)), clips, spans=spans
        )

    early = matched(0.0, 0.01)
    two_h = (clips.recording == by_name["two_hours"]) & (mid_s < 3)
    short = (
        (clips.recording == by_name["forty_five_seconds"]) & (mid_s > 0.5) & (mid_s < 3)
    )
    assert two_h.any() and short.any()
    assert early[two_h].all() and not early[short].any()

    middle = matched(0.45, 0.55)
    for name, around in (("forty_five_seconds", 22.5), ("two_hours", 3564.0)):
        near = (clips.recording == by_name[name]) & (np.abs(mid_s - around) < 2)
        assert near.any() and middle[near].all(), name
    assert not middle[(clips.recording == by_name["two_hours"]) & (mid_s < 60)].any()


def test_a_recording_with_clips_at_one_end_only(session_lengths):
    spans, view, layout = _view(session_lengths)
    clips = view.clips(view.root)
    by_name = {s.name: view.recording_ids.index(r) for r, s in layout.specs.items()}

    def matched(lo, hi, name):
        hit = cluster_filter.clip_matches(
            Selection(session=(lo, hi)), clips, spans=spans
        )
        return hit[clips.recording == by_name[name]]

    assert matched(0.0, 0.01, "fifteen_minutes_start_only").all()
    assert not matched(0.5, 1.0, "fifteen_minutes_start_only").any()
    assert matched(0.99, 1.0, "ninety_minutes_end_only").all()
    assert not matched(0.0, 0.5, "ninety_minutes_end_only").any()


def _line_through(clips, i, spans) -> float:
    """A fraction whose edge in clip ``i``'s recording falls exactly on that clip's midpoint: half
    a nanosecond past it, so the edge's rounding lands on the midpoint and not beside it.
    """
    rid = clips.recording_ids[int(clips.recording[i])]
    first, last = spans[rid]
    mid = (int(clips.start_ns[i]) + int(clips.end_ns[i])) // 2
    return (mid - first + 0.5) / (last - first)


def test_the_two_halves_split_every_clip_exactly_once(session_lengths):
    """``[0, x]`` and ``[x, 1]`` hold every clip between them and none twice: a range is
    half-open (§22). Run at round fractions and, per recording, at a line drawn exactly through
    a clip's midpoint, which goes to the later half — the case an inclusive end counted twice.
    """
    spans, view, layout = _view(session_lengths)
    clips = view.clips(view.root)
    lines = [(x, None) for x in (0.01, 0.25, 0.5, 0.75, 0.99)]
    for rec in np.unique(clips.recording):
        mine = np.flatnonzero(clips.recording == rec)
        i = int(mine[len(mine) // 2])
        lines.append((_line_through(clips, i, spans), i))
    on_a_line = 0
    for x, i in lines:
        lower = cluster_filter.clip_matches(
            Selection(session=(0.0, x)), clips, spans=spans
        )
        upper = cluster_filter.clip_matches(
            Selection(session=(x, 1.0)), clips, spans=spans
        )
        assert (lower | upper).all(), x
        assert not (lower & upper).any(), f"a clip in both halves at {x}"
        if i is not None:
            assert upper[i] and not lower[i], "the clip on the line is the later half's"
            on_a_line += 1
    assert on_a_line == 12


# --- shares: the time inside, each recording normalised by its own length ------------------


@pytest.mark.parametrize("lo, hi", RANGES)
def test_every_node_s_share_is_its_time_inside_each_recording_s_range(
    session_lengths, lo, hi
):
    spans, view, layout = _view(session_lengths)
    expected = _expected_shares(view, layout, _bouts(session_lengths), lo, hi)
    actual = view.match_shares(Selection(session=(lo, hi)), spans=spans)
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-9, equal_nan=True)
    assert 0 < expected[view.root] < 1


def test_complementary_ranges_split_every_node_s_time_with_nothing_lost(
    session_lengths,
):
    """``[0, x]`` and ``[x, 1]`` share out each node's whole time: they sum to 1 for every node,
    including the time a recording's last run covers past its last frame."""
    spans, view, layout = _view(session_lengths)
    for x in (0.01, 0.25, 0.5, 0.75, 0.99):
        lower = view.match_shares(Selection(session=(0.0, x)), spans=spans)
        upper = view.match_shares(Selection(session=(x, 1.0)), spans=spans)
        assert np.allclose(lower + upper, 1.0, atol=1e-9), x


def test_every_bout_covers_its_frames_and_one_step_and_no_more(session_lengths):
    """The cache's runs are the fit's bouts, each from its first frame to one step past its last
    (``_bouts``): consecutive bouts touch and never overlap (on a jittered clock, by less than its
    jitter), and none ends more than one step past its recording's last frame. The old rule ended
    each a whole bout later."""
    _spans, view, layout = _view(session_lengths)
    t = view._cache.leaves
    runs = sorted(
        zip(
            t.run_recording.tolist(),
            t.run_start_ns.tolist(),
            t.run_end_ns.tolist(),
            t.run_leaf.tolist(),
        )
    )
    bouts = sorted((rec, s, e, leaf) for leaf, rec, s, e in _bouts(session_lengths))
    assert runs == bouts
    overlaps: dict[str, int] = {}
    for (r0, _s0, e0, _l0), (r1, s1, _e1, _l1) in zip(runs, runs[1:]):
        if r0 == r1 and e0 > s1:
            name = layout.names[view.recording_ids[r0]]
            overlaps[name] = max(overlaps.get(name, 0), e0 - s1)
    # A bout's step is its span over its frame count, a mean, so on the jittered clock (±2 ms a
    # frame) a bout can end up to the jitter past the next one's first frame; on every regular
    # clock bouts touch exactly. The old rule overlapped each bout by its whole length (~0.37 s).
    assert set(overlaps) <= {"three_minutes_dropped"}, overlaps
    assert all(v < 4_000_000 for v in overlaps.values()), overlaps
    for rec, rid in enumerate(view.recording_ids):
        maps = layout.part_maps[rid]
        step = max(int(np.diff(m).max()) for m in maps if len(m) > 1)
        ends = [e for r, _s, e, _l in runs if r == rec]
        assert max(ends) <= layout.spans[rid][1] + step, layout.names[rid]


def test_the_end_of_the_session_holds_a_run_that_ends_on_the_last_frame(
    session_lengths,
):
    """A run inside the last 1% of its recording is wholly in ``[0.99, 1]``, though its covered
    time runs past the last frame; the end-only recording's every member is."""
    spans, view, layout = _view(session_lengths)
    rid = next(
        r for r, s in layout.specs.items() if s.name == "ninety_minutes_end_only"
    )
    t = view._cache.leaves
    rec = view.recording_ids.index(rid)
    _first, last = layout.spans[rid]
    assert (t.run_end_ns[t.run_recording == rec] > last).any(), "no run reaches past it"
    only = Selection(session=(0.99, 1.0), recording_ids=frozenset({rid}))
    matched, total = view.match_counts(only, spans=spans)
    whole, _ = view.match_counts(Selection(recording_ids=frozenset({rid})), spans=spans)
    assert matched[view.root] == pytest.approx(whole[view.root])
    assert matched[view.root] > 0
