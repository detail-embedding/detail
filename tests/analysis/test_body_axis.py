"""The heading a keypoint track run gives each frame (§14 "Egocentric alignment"), on rows a
test can see: the set means, what makes a row undefined, interpolation across a gap the short
way round, the mirror hold, and the counts the run records."""

import numpy as np
import pytest

from detail.analysis import body_axis
from detail.analysis.body_axis import (
    SOURCE_INTERPOLATED,
    SOURCE_NONE,
    SOURCE_OWN,
    HeadingSeries,
    coverage_counts,
    heading_at,
    heading_series,
    longest_gap_s,
    mirror_flags,
    row_headings,
    validate_definition,
)

NAMES = ("nose", "left_ear", "right_ear", "tail_base")
WH = (200, 100)  # a non-square frame: pixels, not normalized units, decide the angle
S = 1_000_000_000


def test_a_definition_is_two_disjoint_non_empty_sets_of_known_points():
    checked = validate_definition({"front": "nose", "back": ["tail_base"]}, NAMES)
    assert checked == {"front": ["nose"], "back": ["tail_base"]}
    for bad, fragment in [
        ({"front": [], "back": ["tail_base"]}, "at least one 'front'"),
        ({"front": ["nose", "nose"], "back": ["tail_base"]}, "twice"),
        ({"front": ["nose"], "back": ["nose"]}, "both front and back"),
        ({"front": ["nose"], "back": ["hip"]}, "skeleton lacks: ['hip']"),
        ("nose", "mapping"),
    ]:
        with pytest.raises(ValueError, match=fragment.replace("[", r"\[")):
            validate_definition(bad, NAMES)
    assert validate_definition(
        {"front": ["nose"], "back": ["hip"]}
    )  # no skeleton: fine


def test_the_heading_is_the_pixel_angle_from_the_back_mean_to_the_front_mean():
    points = np.full((3, 4, 2), np.nan)
    points[0, 0] = (0.6, 0.5)  # nose right of the tail: 0°
    points[0, 3] = (0.4, 0.5)
    points[1, 1] = (0.5, 0.3)  # ears above the tail, nose missing: the front mean moves
    points[1, 2] = (0.7, 0.3)
    points[1, 3] = (0.6, 0.6)
    points[2, 0] = (0.5, 0.5)  # a front with no usable point: undefined
    points[2, 1] = (0.5, 0.5)
    points[2, 3] = (0.5, 0.5)
    scores = np.full((3, 4), 0.9)
    scores[2, 0] = 0.1
    scores[2, 1] = 0.1
    deg, length = row_headings(points, scores, [0, 1, 2], [3], WH, cutoff=0.5)
    assert deg[0] == pytest.approx(0.0) and length[0] == pytest.approx(40.0)
    assert deg[1] == pytest.approx(-90.0)  # straight up the image on a 2:1 frame
    assert length[1] == pytest.approx(30.0)
    assert np.isnan(deg[2]) and np.isnan(length[2])
    # normalized units would have said -56°: the aspect ratio is in the angle
    deg_square, _ = row_headings(points, scores, [0, 1, 2], [3], (100, 100), 0.5)
    assert deg_square[1] == pytest.approx(-90.0)
    assert row_headings(points, scores, [0], [3], WH, None)[0][2] == pytest.approx(
        0.0
    )  # no cut-off: the weak nose counts, back at (0.5,0.5)... same pixel: NaN length
    assert row_headings(points, scores, [0], [3], WH, None)[1][2] == pytest.approx(0.0)


def test_a_foreshortened_axis_is_undefined_against_the_tracks_typical_length():
    n = 40
    points = np.full((n, 4, 2), np.nan)
    points[:, 3] = (0.5, 0.5)
    points[:, 0, 1] = 0.5
    points[:, 0, 0] = (
        0.5 + 0.2
    )  # a 40 px axis, except where the animal faces the camera
    points[10:14, 0, 0] = 0.5 + 0.02  # 4 px: facing the camera
    t_ns = np.arange(n) * S // 30
    series = heading_series(
        t_ns,
        points,
        np.ones((n, 4)),
        [0],
        [3],
        WH,
        cutoff=None,
        facing_camera_ratio=0.5,
    )
    assert series.n_rows == n and series.n_defined == n - 4
    assert series.typical_length_px == pytest.approx(40.0)
    assert not np.isin(t_ns[10:14], series.t_ns).any()
    off = heading_series(
        t_ns,
        points,
        np.ones((n, 4)),
        [0],
        [3],
        WH,
        cutoff=None,
        facing_camera_ratio=0.0,
    )
    assert off.n_defined == n  # ratio 0 keeps every axis with any length


def test_heading_at_takes_the_row_interpolates_the_short_way_and_drops_long_gaps():
    series = HeadingSeries(
        t_ns=np.array([0, 1, 2, 10, 11], dtype=np.int64) * S,
        deg=np.array([170.0, -170.0, -150.0, 0.0, 10.0]),
        n_rows=12,
        typical_length_px=40.0,
    )
    times = np.array([0.0, 0.5, 1.0, 1.5, 5.0, 10.0, 12.0, -1.0]) * S
    deg, source = heading_at(series, times.astype(np.int64), max_gap_s=1.5)
    assert source.tolist() == [
        SOURCE_OWN,
        SOURCE_INTERPOLATED,
        SOURCE_OWN,
        SOURCE_INTERPOLATED,
        SOURCE_NONE,  # the 8 s gap is longer than max_gap_s
        SOURCE_OWN,
        SOURCE_NONE,  # after the last defined row
        SOURCE_NONE,  # before the first
    ]
    assert deg[0] == pytest.approx(170.0)
    assert deg[1] == pytest.approx(180.0) or deg[1] == pytest.approx(
        -180.0
    )  # short way
    assert deg[3] == pytest.approx(-160.0)
    assert np.isnan(deg[[4, 6, 7]]).all()
    assert deg.dtype == np.float32 and source.dtype == np.int8
    empty = heading_at(
        HeadingSeries(np.zeros(0, np.int64), np.zeros(0), 0, 0.0), times, 1.0
    )
    assert (empty[1] == SOURCE_NONE).all()


def test_mirror_flips_only_after_the_new_side_holds():
    t_ns = (np.arange(40) * S // 10).astype(np.int64)  # 10 Hz
    deg = np.full(40, 10.0)  # facing right
    deg[12] = 170.0  # one frame facing left: not held
    deg[20:] = 170.0  # from 2.0 s on, facing left for good
    deg[30] = np.nan  # a frame with no heading is never mirrored
    flags = mirror_flags(deg, t_ns, hold_s=0.5)
    assert not flags[:20].any()  # the flicker at 1.2 s never flipped the image
    # the hold looks both ways (§14): a turn that lasts flips from its first frame, so no
    # frame of it is embedded facing the wrong way
    assert flags[20:30].all()
    assert not flags[30] and flags[31:].all()
    short = np.full(6, 10.0)
    short[2:] = (
        170.0  # turns left and the series ends 0.3 s later: not held, not flipped
    )
    assert not mirror_flags(short, t_ns[:6], hold_s=0.5).any()
    flickers = np.array(
        [170.0, 10.0, 170.0, 10.0]
    )  # nothing holds: the first side stays
    assert mirror_flags(flickers, t_ns[:4], hold_s=0.5).tolist() == [True] * 4
    assert not mirror_flags(np.full(5, np.nan), t_ns[:5], 0.5).any()
    assert mirror_flags(np.full(5, 170.0), t_ns[:5], 0.5).all()  # left from the start


def test_coverage_counts_and_the_longest_gap():
    source = np.array([0, 1, -1, -1, -1, 0, -1, 0], dtype=np.int8)
    t_ns = (np.arange(8) * S).astype(np.int64)
    assert coverage_counts(source) == {
        "frames": 8,
        "own": 3,
        "interpolated": 1,
        "dropped": 4,
    }
    assert longest_gap_s(t_ns, source) == pytest.approx(4.0)  # 1 s to 5 s, bracketed
    assert longest_gap_s(t_ns[:3], source[:3]) == pytest.approx(1.0)  # an edge run
    assert longest_gap_s(t_ns, np.zeros(8, np.int8)) == 0.0


class _Run:
    """A stand-in for ``KeypointTrackRun``: one track, two parts, the second one facing the
    camera throughout (short axes) — judged against the whole run's typical length."""

    class _Skeleton:
        names = NAMES

    skeleton = _Skeleton()

    def part_ids(self, track_id):
        return ["part_001", "part_002"]

    def rows(self, track_id, part_id):
        n = 10
        points = np.full((n, 4, 2), np.nan)
        points[:, 3] = (0.5, 0.5)
        points[:, 0, 1] = 0.5
        points[:, 0, 0] = 0.7 if part_id == "part_001" else 0.52
        return {
            "frame_idx": np.arange(n),
            "t_ns": np.arange(n) * S // 30 + (0 if part_id == "part_001" else 10 * S),
            "points": points,
            "point_scores": np.ones((n, 4)),
        }


def test_track_series_measures_the_typical_length_over_the_whole_run():
    series = body_axis.track_series(
        _Run(),
        0,
        {"front": ["nose"], "back": ["tail_base"]},
        WH,
        cutoff=None,
        facing_camera_ratio=0.5,
    )
    assert set(series) == {"part_001", "part_002"}
    assert series["part_001"].n_defined == 10
    assert series["part_002"].n_defined == 0  # 4 px against the run's 40 px: undefined
    assert series["part_002"].typical_length_px == pytest.approx(40.0)
    with pytest.raises(ValueError, match="skeleton lacks"):
        body_axis.track_series(
            _Run(),
            0,
            {"front": ["nose"], "back": ["hip"]},
            WH,
            cutoff=None,
            facing_camera_ratio=0.5,
        )


# --- the body's shape (the clip popup's pose strips and aligned view) ------------------ #
SPINE = ("nose", "mid", "tail")


def _rows(*bodies):
    """``(points (M, 3, 2) normalized, scores (M, 3))`` from bodies given in pixels of WH."""
    px = np.asarray(bodies, dtype=np.float64)
    return px / np.array(WH, dtype=np.float64), np.full(px.shape[:2], 0.9)


def test_a_straight_body_has_no_bend_and_a_right_angle_reads_as_ninety():
    points, scores = _rows(
        [(30, 50), (20, 50), (10, 50)],  # straight, facing +x
        [(20, 60), (20, 50), (10, 50)],  # back -> mid along +x, mid -> front along +y
        [(20, 40), (20, 50), (10, 50)],  # the same turn the other way
    )
    bends = body_axis.row_bends(points, scores, [0], [2], [1], WH, None)
    # image y points down, so a turn from +x to +y is clockwise on screen: positive
    assert bends == pytest.approx([0.0, 90.0, -90.0], abs=1e-9)


def test_the_bend_is_taken_at_the_middle_point_furthest_off_the_axis():
    # nose, neck, hip, tail: the neck sits on the axis, the hip 10 px off it
    px = np.array([[(40, 50), (30, 50), (20, 60), (10, 50)]], dtype=np.float64)
    points, scores = px / np.array(WH, dtype=np.float64), np.full((1, 4), 0.9)
    bend = body_axis.row_bends(points, scores, [0], [3], [1, 2], WH, None)[0]
    # at the hip: tail -> hip is (10, 10), hip -> nose is (20, -10)
    assert bend == pytest.approx(np.degrees(np.arctan2(10 * -10 - 10 * 20, 200 - 100)))


def test_no_usable_middle_point_is_no_bend():
    points, scores = _rows([(30, 50), (20, 60), (10, 50)])
    assert np.isnan(body_axis.row_bends(points, scores, [0], [2], [], WH, None)).all()
    scores[0, 1] = 0.1
    assert np.isnan(body_axis.row_bends(points, scores, [0], [2], [1], WH, 0.5)).all()
    points[0, 0] = np.nan  # no front end, no axis
    assert np.isnan(body_axis.row_bends(points, scores, [0], [2], [1], WH, None)).all()


def test_the_midline_is_the_unpaired_points_between_the_ends():
    from detail.core.skeleton import Skeleton

    skeleton = Skeleton(
        names=("nose", "left_ear", "right_ear", "neck", "hip", "tail_base"),
        edges=(),
        pairs=(("left_ear", "right_ear"),),
    )
    definition = validate_definition(
        {"front": ["nose"], "back": ["tail_base"]}, skeleton.names
    )
    assert body_axis.midline_columns(skeleton, definition) == [3, 4]


def test_the_aligned_pose_faces_up_centred_at_the_scale_asked():
    points, scores = _rows([(30, 50), (20, 50), (10, 50)])  # facing +x, 20 px long
    front, back = body_axis.row_axis_ends(points, scores, [0], [2], WH, None)
    turned = body_axis.aligned_points(points[0], front[0], back[0], WH, 2.0)
    px = turned * np.array(WH)
    centre = np.array(WH) / 2
    assert px[1] == pytest.approx(
        centre
    ), "the middle of the axis is the frame's centre"
    assert px[0] == pytest.approx(
        centre + [0, -20]
    ), "the nose straight up, twice as far"
    assert px[2] == pytest.approx(centre + [0, 20])
