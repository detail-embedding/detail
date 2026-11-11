"""Tests for the §19 frame view (detail/storage/frame_view.py).

The frame view exists for one number: a **windowed** observed-time denominator. `coverage.csv`
records how much was unobserved, not where, so applying it to a sub-window inflates or deflates
every rate in proportion to where the tracking gaps fall — and the resulting table looks fine.
So most of what is tested here is that the mask means what §23 needs it to mean, and that the
expansion addresses frames by time rather than by index.
"""

import numpy as np
import pytest
import zarr

from detail.storage import frame_view as fv
from detail.storage.behavior_output import BehaviorInstance

PERIOD = 1_000  # ns between frames, so a frame index reads as t_ns // 1000


def _axis(n=100, period=PERIOD, start=0):
    return start + np.arange(n, dtype=np.int64) * period


def _instance(start_t, end_t, source_label="q01", subject=0, **kw):
    data = dict(
        recording_id="recording_001",
        subject=subject,
        t_ns=start_t,
        end_t_ns=end_t,
        start_frame=start_t // PERIOD,
        end_frame=end_t // PERIOD,
        source_label=source_label,
    )
    data.update(kw)
    return BehaviorInstance(**data)


def _write(tmp_path, *, axis=None, observed=None, instances=(), subject=0):
    axis = _axis() if axis is None else axis
    observed = np.ones(len(axis), dtype=bool) if observed is None else observed
    fv.write_frame_view(
        tmp_path,
        recording_id="recording_001",
        subject=subject,
        t_ns=axis,
        observed=observed,
        instances=list(instances),
    )
    return fv.read_frame_view(tmp_path, "recording_001", subject)


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_rewriting_a_unit_replaces_it_while_a_reader_holds_the_old_one(tmp_path):
    """An authored run rewrites its frame view on every reprojection, and a surface may be
    reading it — the GUI measures panel is one.

    Written in place (`zarr.open_group(mode="w")`, which empties the directory first) that
    raced: intermittently `[Errno 66] Directory not empty`, roughly one run in four on macOS,
    and as a hard abort under CI's headless Qt. The store is now built beside the old one and
    swapped in, so a reader holding the displaced copy cannot break the writer.
    """
    axis = _axis()
    holder = _write(tmp_path, axis=axis, observed=np.ones(len(axis), bool),
                    instances=[_instance(axis[0], axis[4])], subject=0)  # fmt: skip
    assert holder.n_frames == len(axis)

    # Rewrite with different content while `holder` still references the old store.
    _write(tmp_path, axis=axis, observed=np.zeros(len(axis), bool),
           instances=[_instance(axis[2], axis[6])], subject=0)  # fmt: skip

    fresh = fv.read_frame_view(tmp_path, "recording_001", 0)
    assert not fresh.observed[:].any(), "the rewrite is what a reader sees next"
    leftovers = sorted(q.name for q in fv.frames_dir(tmp_path).iterdir())
    assert leftovers == [
        "recording_001__0.zarr"
    ], f"the swap left scratch behind: {leftovers}"


def test_schema_dtypes_chunks(tmp_path):
    _write(tmp_path, instances=[_instance(10_000, 14_000)])

    root = zarr.open_group(str(fv.unit_path(tmp_path, "recording_001", 0)), mode="r")
    assert root.attrs["schema_version"] == 1  # Tenet 7
    assert set(root.array_keys()) == {"t_ns", "observed", "instance_id", "q01"}
    assert root["t_ns"].dtype == np.int64
    assert root["observed"].dtype == np.bool_
    assert root["q01"].dtype == np.int32
    assert root["t_ns"].shape == (100,)
    assert root["q01"].shape == (100,)
    assert root["instance_id"].shape == (1,)
    assert root["t_ns"].chunks == (10_000,)
    assert root["q01"].chunks == (10_000,)


def test_the_store_is_named_for_its_unit(tmp_path):
    _write(tmp_path, subject=3)
    assert fv.unit_filename("recording_001", 3) == "recording_001__3.zarr"
    assert fv.unit_filename("recording_001", None) == "recording_001__none.zarr"
    assert fv.parse_unit("recording_001__3") == ("recording_001", 3)
    assert fv.parse_unit("recording_001__none") == ("recording_001", None)


def test_discovery_scans_the_frames_dir(tmp_path):
    _write(tmp_path, subject=0)
    _write(tmp_path, subject=1)
    assert fv.discover_frame_views(tmp_path) == [
        ("recording_001", 0),
        ("recording_001", 1),
    ]
    assert fv.discover_frame_views(tmp_path / "nope") == []


def test_discovery_orders_recordings_past_the_id_pad(tmp_path):
    """A recording id is a minted `recording_NNN`, so a text sort puts `recording_10` before
    `recording_9` once an experiment outgrows the pad — and every listing inherits it.
    """
    axis = _axis(10)
    for rid in ("recording_9", "recording_10", "recording_100"):
        fv.write_frame_view(
            tmp_path,
            recording_id=rid,
            subject=0,
            t_ns=axis,
            observed=np.ones(10, dtype=bool),
            instances=[],
        )
    assert [rid for rid, _ in fv.discover_frame_views(tmp_path)] == [
        "recording_9",
        "recording_10",
        "recording_100",
    ]


def test_a_subjectless_unit_round_trips(tmp_path):
    view = _write(tmp_path, subject=None)
    assert view.subject is None
    assert fv.discover_frame_views(tmp_path) == [("recording_001", None)]


# --------------------------------------------------------------------------- #
# observed: cadence is not a gap
# --------------------------------------------------------------------------- #
def test_cadence_does_not_reduce_observed_time():
    """The failure this guards is a rate that comes out N times too high.

    A run at every_n_frames=5 holds a fifth of the rows but covered the same wall clock. A mask
    that marked only sampled frames would divide the denominator by 5 and multiply every rate
    by 5, with nothing anywhere looking wrong.
    """
    axis = _axis(100)
    dense = fv.observed_mask(axis, axis[:75])
    sparse = fv.observed_mask(axis, axis[:75:5])
    assert dense.sum() == sparse.sum() == 75


def test_a_tracking_gap_is_unobserved():
    # ...but a real gap must still register: cadence 1 for 40 frames, a 30-frame hole, then more.
    axis = _axis(100)
    rows = np.concatenate([axis[:40], axis[70:]])
    mask = fv.observed_mask(axis, rows)
    assert mask[:40].all()
    assert not mask[40:70].any()
    assert mask[70:].all()


def test_an_empty_analysis_observes_nothing():
    axis = _axis(10)
    assert not fv.observed_mask(axis, np.empty(0, dtype=np.int64)).any()


def test_observed_mask_refuses_an_unsorted_axis():
    with pytest.raises(fv.FrameViewError, match="non-decreasing"):
        fv.observed_mask(np.array([0, 5, 3], dtype=np.int64), np.array([0]))


# --------------------------------------------------------------------------- #
# the windowed denominator — the reason the file exists
# --------------------------------------------------------------------------- #
def test_windowed_observed_time_uses_where_the_gap_is_not_just_how_much(tmp_path):
    """The whole point, stated as a test.

    60 frames observed of 100, but all 40 unobserved frames sit in the first half. A window over
    the first half must not be told "60% of it was observed" — which is exactly what scaling
    `coverage.observed_ns` by window length would say.
    """
    axis = _axis(100)
    observed = np.ones(100, dtype=bool)
    observed[10:50] = False  # the whole gap is inside the first half
    view = _write(tmp_path, axis=axis, observed=observed)

    first_half = fv.observed_ns_in(view, 0, 50 * PERIOD)
    second_half = fv.observed_ns_in(view, 50 * PERIOD, 100 * PERIOD)
    assert first_half == 10 * PERIOD  # frames 0-9 only
    assert second_half == 50 * PERIOD  # frames 50-99, all observed
    # and the naive scaling a whole-recording denominator would have given
    assert first_half != fv.observed_ns_in(view) // 2


def test_adjacent_windows_sum_to_the_window_that_spans_them(tmp_path):
    """Additivity, which the measure did not have.

    Both bounds were inclusive, so the frame on a shared boundary landed in *both* windows and
    was credited a full period to each. Two halves summed to more than the whole, and since
    this is §23's denominator, every rate over a sub-window was deflated by that much.
    """
    view = _write(tmp_path, axis=_axis(100), observed=np.ones(100, dtype=bool))
    whole = fv.observed_ns_in(view, 0, 100 * PERIOD)
    cuts = [0, 17 * PERIOD, 40 * PERIOD, 63 * PERIOD, 100 * PERIOD]
    pieces = [
        fv.observed_ns_in(view, lo, hi) for lo, hi in zip(cuts, cuts[1:])
    ]  # fmt: skip
    assert sum(pieces) == whole == 100 * PERIOD


def test_a_window_never_observes_more_time_than_it_is_long(tmp_path):
    # The same bug seen from one window: `[t, t + period]` reported `2 * period`, so a short
    # window could report a fraction above 1.0 with nothing raising.
    view = _write(tmp_path, axis=_axis(100), observed=np.ones(100, dtype=bool))
    for length in (1, 2, 5):
        window = length * PERIOD
        assert fv.observed_ns_in(view, 20 * PERIOD, 20 * PERIOD + window) == window


def test_observed_ns_over_the_whole_recording_matches_coverage(tmp_path):
    """The cross-artifact invariant: the frame view and coverage.csv describe one thing.

    If they disagree, one of them is lying about how much was seen — and every rate built on
    either is wrong by that much.
    """
    axis = _axis(100)
    observed = np.ones(100, dtype=bool)
    observed[30:45] = False
    view = _write(tmp_path, axis=axis, observed=observed)

    coverage_observed_ns = 85 * PERIOD  # what a producer would write to coverage.csv
    assert fv.observed_ns_in(view) == coverage_observed_ns


def test_a_window_outside_the_recording_observes_nothing(tmp_path):
    view = _write(tmp_path)
    assert fv.observed_ns_in(view, 10**9, 2 * 10**9) == 0


def test_an_analyzed_run_covering_no_frame_marks_no_frame(tmp_path):
    """`observed` is what §23 divides by, so a frame marked for a run that never reached it is
    observed time invented out of nothing.

    The mask forced at least one frame true per run. When no axis frame falls inside a run's
    span that forced frame is the first one *after* it, which can sit arbitrarily far away —
    here a run at 3-4 us marking a frame at 50 us.
    """
    axis = np.array([0, 1_000, 2_000, 50_000], dtype=np.int64)
    rows = np.array([3_000, 4_000, 5_000], dtype=np.int64)
    mask = fv.observed_mask(axis, rows)
    assert (
        not mask.any()
    ), f"marked {np.nonzero(mask)[0].tolist()} for a run covering nothing"


def test_a_run_that_does_cover_frames_still_marks_them(tmp_path):
    # The paired positive, so the fix above cannot be "mark nothing, ever".
    axis = _axis(10)
    rows = axis[3:7]
    mask = fv.observed_mask(axis, rows)
    assert mask[3:7].all() and not mask[:3].any()


# --------------------------------------------------------------------------- #
# the expansion
# --------------------------------------------------------------------------- #
def test_an_instance_covers_its_frames_inclusively(tmp_path):
    view = _write(tmp_path, instances=[_instance(10_000, 14_000)])
    column = view.root["q01"][:]
    assert (column[10:15] == 1).all()  # both ends inclusive
    assert column[9] == 0 and column[15] == 0


def test_abutting_instances_stay_two(tmp_path):
    """The failure a boolean array has by construction: two bouts of one label that touch merge
    into one rising edge, and the bout count — a headline measure — comes out one short.
    """
    view = _write(
        tmp_path,
        instances=[_instance(10_000, 14_000), _instance(15_000, 19_000)],
    )
    column = view.root["q01"][:]
    assert set(column[10:15]) == {1}
    assert set(column[15:20]) == {2}
    assert len(view.instance_ids()) == 2


def test_a_twice_named_cluster_takes_one_ordinal(tmp_path):
    """instances.csv is one row per (bout, label), so a cluster the user named twice appears
    twice over one span. It must not occupy two ordinals, or a consumer reading frames counts
    the same bout twice — the double count §19 exists to prevent."""
    bout = _instance(10_000, 14_000)
    twin = _instance(10_000, 14_000, label="near wall")
    assert bout.instance_id == twin.instance_id  # same span, same source_label
    view = _write(tmp_path, instances=[bout, twin])
    assert len(view.instance_ids()) == 1
    assert set(view.root["q01"][10:15]) == {1}


def test_several_source_labels_get_their_own_arrays(tmp_path):
    view = _write(
        tmp_path,
        instances=[_instance(10_000, 14_000), _instance(20_000, 24_000, "q02")],
    )
    assert view.source_labels == ["q01", "q02"]
    assert (view.root["q01"][10:15] > 0).all()
    assert (view.root["q02"][20:25] > 0).all()
    assert not (view.root["q01"][20:25] > 0).any()


def test_the_ordinal_joins_back_to_the_stable_instance_id(tmp_path):
    rows = [_instance(10_000, 14_000), _instance(20_000, 24_000)]
    view = _write(tmp_path, instances=rows)
    ids = view.instance_ids()
    column = view.root["q01"][:]
    for row in rows:
        ordinal = int(column[row.t_ns // PERIOD])
        assert ids[ordinal - 1] == row.instance_id


def test_expansion_round_trips_back_to_intervals(tmp_path):
    spans = [(10_000, 14_000), (30_000, 30_000), (50_000, 59_000)]
    view = _write(tmp_path, instances=[_instance(a, b) for a, b in spans])
    column = view.root["q01"][:]
    axis = view.t_ns[:]
    recovered = []
    for ordinal in range(1, len(spans) + 1):
        rows = np.nonzero(column == ordinal)[0]
        recovered.append((int(axis[rows[0]]), int(axis[rows[-1]])))
    assert recovered == spans


# --------------------------------------------------------------------------- #
# addressing: by time, never by frame index
# --------------------------------------------------------------------------- #
def test_a_multipart_recording_places_instances_by_time_not_frame_index(tmp_path):
    """An embedding part's `frame_idx` is part-local — no cross-part offset is applied — so a
    concatenated trajectory restarts its frame numbering at every part boundary. Addressing a
    recording-global axis by frame number therefore lands part 2's instances inside part 1.

    Here part 2 starts at global frame 60, and its instance carries the part-local
    `start_frame=5` it would really have. Time says frame 65; the frame index says 5.
    """
    axis = _axis(120)
    row = _instance(65 * PERIOD, 69 * PERIOD)
    row.start_frame, row.end_frame = 5, 9  # part-local, as an embedding part records it
    view = _write(tmp_path, axis=axis, instances=[row])

    column = view.root["q01"][:]
    assert (column[65:70] == 1).all(), "instance was not placed by time"
    assert not column[5:10].any(), "instance was placed at its part-local frame index"


def test_an_instance_off_the_axis_is_refused(tmp_path):
    with pytest.raises(fv.FrameViewError, match="covers no frame"):
        _write(tmp_path, instances=[_instance(10**9, 2 * 10**9)])


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_a_source_label_colliding_with_a_reserved_array_is_refused(tmp_path):
    with pytest.raises(fv.FrameViewError, match="reserved array name"):
        _write(tmp_path, instances=[_instance(10_000, 14_000, "observed")])


def test_an_unsafe_source_label_is_refused(tmp_path):
    with pytest.raises(fv.FrameViewError, match="safe zarr array name"):
        _write(tmp_path, instances=[_instance(10_000, 14_000, "q01/../etc")])


def test_a_mismatched_observed_mask_is_refused(tmp_path):
    with pytest.raises(fv.FrameViewError, match="observed has"):
        _write(tmp_path, axis=_axis(100), observed=np.ones(50, dtype=bool))


def test_an_unsorted_axis_is_refused(tmp_path):
    with pytest.raises(fv.FrameViewError, match="non-decreasing"):
        _write(tmp_path, axis=np.array([0, 5_000, 3_000], dtype=np.int64))


def test_reading_a_missing_unit_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no frame view"):
        fv.read_frame_view(tmp_path, "recording_001", 0)


# --------------------------------------------------------------------------- #
# frame durations
# --------------------------------------------------------------------------- #
def test_the_last_frame_gets_the_median_duration():
    # Assuming zero would silently shorten every window that ends at the recording's end.
    durations = fv.frame_durations(_axis(5))
    assert durations.tolist() == [PERIOD] * 5


def test_frame_durations_of_a_degenerate_axis():
    assert fv.frame_durations(np.empty(0, dtype=np.int64)).tolist() == []
    assert fv.frame_durations(np.array([7], dtype=np.int64)).tolist() == [0]


# --------------------------------------------------------------------------- #
# one frame view holds exactly one unit
# --------------------------------------------------------------------------- #
def test_another_units_instance_is_refused(tmp_path):
    """Nothing else here reads `recording_id` / `subject` — they name the file, not the rows.

    So a foreign instance whose span happened to land on this axis was written straight in and
    given an ordinal in this subject's store. It is not even collapsed away as a duplicate,
    because `instance_id` hashes the unit.
    """
    with pytest.raises(fv.FrameViewError, match="different"):
        _write(
            tmp_path,
            instances=[_instance(1_000, 5_000, subject=1)],  # written into subject 0
            subject=0,
        )


def test_another_recordings_instance_is_refused(tmp_path):
    with pytest.raises(fv.FrameViewError, match="different"):
        _write(
            tmp_path, instances=[_instance(1_000, 5_000, recording_id="recording_002")]
        )


def test_overlapping_bouts_of_one_label_are_refused_not_truncated(tmp_path):
    """Bouts are written in time order, so a later one overwrote the earlier's frames.

    Partial overlap truncated the earlier bout; full containment erased it entirely while it
    kept its slot in `instance_id`. Either way the frame view disagreed with `instances.csv`,
    which §19 calls authoritative, and a duration read off the frames came out short with
    nothing wrong on the face of it.
    """
    with pytest.raises(fv.FrameViewError, match="overlap"):
        _write(
            tmp_path,
            instances=[
                _instance(1_000, 6_000, source_label="q01"),
                _instance(5_000, 9_000, source_label="q01"),
            ],
        )


def test_overlapping_bouts_of_different_labels_are_fine(tmp_path):
    # The paired positive: §19 permits overlap *between* labels — the column is per
    # source_label, so two labels active at once is exactly what the layout is for.
    view = _write(
        tmp_path,
        instances=[
            _instance(1_000, 6_000, source_label="q01"),
            _instance(5_000, 9_000, source_label="q02"),
        ],
    )
    assert set(view.source_labels) == {"q01", "q02"}
