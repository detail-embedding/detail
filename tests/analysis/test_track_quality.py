"""Label-free tracking quality and the sweep's pick (``analysis/track_quality.py``), on
synthetic RLE masks: no subject is assumed, so the cases are shapes, not animals."""

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from detail.analysis import track_quality as tq

SIZE = (40, 60)


def _rle(r0, r1, c0, c1):
    m = np.zeros(SIZE, dtype=np.uint8)
    m[r0:r1, c0:c1] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _table(rows, n_frames):
    """``rows`` = (frame, confidence, rle, object_id)."""
    frame_idx, conf, rle, oid = (list(c) for c in zip(*rows)) if rows else ([],) * 4
    return tq.frame_table(
        n_frames,
        np.asarray(frame_idx),
        np.asarray(conf),
        rle,
        np.asarray(oid),
        SIZE,
    )


def test_the_grid_never_puts_a_secondary_above_its_anchor():
    grid = tq.grid()
    assert len(grid) == len(set(grid)) == 43
    assert all(s.secondary_conf <= s.conf_threshold for s in grid)
    assert all(
        s.secondary_conf == s.conf_threshold for s in grid if not s.max_secondary
    )
    assert tq.sweep_detection_conf() == min(tq.SECONDARY_CONFS)


def test_a_detection_run_caps_what_a_setting_can_keep():
    assert tq.effective_setting(0.5, 0.1, 0.05, 2) == tq.Setting(0.5, 0.5, 2)
    assert tq.effective_setting(0.05, 0.1, 0.05, 4) == tq.Setting(0.1, 0.05, 4)
    assert tq.Setting(0.7, 0.25, 2).detection_conf == 0.25
    assert tq.Setting(0.7, 0.7, 0).detection_conf == 0.7


def test_areas_come_from_the_rle_without_decoding():
    assert list(tq.rle_areas([_rle(0, 10, 0, 10), "", _rle(0, 2, 0, 3)], SIZE)) == [
        100,
        0,
        6,
    ]


def test_a_merged_subject_is_the_union_of_its_pieces():
    body, piece = _rle(0, 10, 0, 20), _rle(5, 10, 15, 30)  # 200 + 75, overlap 25
    table = _table([(0, 0.9, body, 0), (0, 0.4, piece, 0), (1, 0.8, body, 0)], 3)
    assert table.area[0] == 250
    assert table.anchor_area[0] == 200
    assert list(table.pieces) == [2, 1, 0]
    assert np.isnan(table.area[2])


def test_the_shares_are_of_the_frames_looked_at():
    body, piece = _rle(0, 10, 0, 20), _rle(0, 10, 20, 30)
    rows = []
    for f in range(10):
        rows.append((f, 0.9 if f < 8 else 0.3, body, 0))
        if f < 4:
            rows.append((f, 0.4, piece, 0))
    rows.append((9, 0.2, piece, -1))
    q = tq.quality([_table(rows, 12)], frame_rate=10.0)
    assert q.n_frames == 12
    assert q.tracked == pytest.approx(100 * 10 / 12)
    assert q.merged == pytest.approx(100 * 4 / 12)
    assert q.grew == pytest.approx(100 * 4 / 12)  # 300 against 200: grew by half
    assert q.low_conf == pytest.approx(100 * 2 / 10)
    assert q.unassigned_per_frame == pytest.approx(1 / 12)
    assert q.median_area == 200


def test_jumpy_is_judged_against_a_rolling_median_in_seconds():
    area = np.full(100, 1000.0)
    area[50] = 400.0  # one partial mask
    area[60] = np.nan  # a frame with no subject is never jumpy
    jumpy = tq.jumpy_frames(area, tq.rolling_rows(10.0))
    assert list(np.flatnonzero(jumpy)) == [50]
    assert tq.rolling_rows(10.0) == 11
    assert tq.rolling_rows(100.0) == 101  # the manual measurement's window at 100 fps


def test_a_slow_change_in_size_is_not_jumpy():
    area = np.linspace(500, 2000, 300)  # the subject comes closer over 30 s
    assert not tq.jumpy_frames(area, tq.rolling_rows(10.0)).any()


def test_the_rolling_median_stays_inside_one_window():
    small = tq.FrameTable(
        np.full(20, 100.0), np.full(20, 100.0), np.ones(20, int), np.full(20, 0.9), 0
    )
    large = tq.FrameTable(
        np.full(20, 900.0), np.full(20, 900.0), np.ones(20, int), np.full(20, 0.9), 0
    )
    assert tq.quality([small, large], frame_rate=10.0).jumpy == 0


def test_two_far_apart_confident_detections_read_as_two_subjects():
    frames = np.array([0, 0, 1, 1, 2, 2])
    conf = np.array([0.9, 0.8, 0.9, 0.8, 0.9, 0.3])
    apart = [[0, 0, 0.2, 0.2], [0.6, 0.6, 0.9, 0.9]]
    touching = [[0, 0, 0.5, 0.5], [0.4, 0.4, 0.9, 0.9]]
    bbox = np.array(apart + touching + apart)
    assert tq.multi_subject_share(frames, conf, bbox) == pytest.approx(1 / 3)


def _q(tracked, jumpy):
    return tq.Quality(100, tracked, 0, 0, jumpy, 500, 0, 0)


DEFAULT = tq.Setting(0.5, 0.5, 2)


def test_the_pick_takes_the_steadiest_setting_that_still_tracks():
    steady_but_loses = tq.Setting(0.7, 0.7, 0)
    best = tq.Setting(0.1, 0.05, 4)
    results = {
        DEFAULT: {"a": _q(100, 1.5), "b": _q(100, 1.0)},
        steady_but_loses: {"a": _q(98, 0.0), "b": _q(100, 0.0)},
        best: {"a": _q(99.8, 0.4), "b": _q(100, 0.2)},
    }
    chosen = tq.pick(results, DEFAULT)
    assert chosen.setting == best and not chosen.keep_defaults
    assert "1.25%" in chosen.reason  # the defaults' mean, for comparison


def test_a_tie_goes_to_fewer_secondaries_then_higher_confidence():
    results = {
        DEFAULT: {"a": _q(100, 5.0)},
        tq.Setting(0.1, 0.05, 8): {"a": _q(100, 0.30)},
        tq.Setting(0.1, 0.05, 2): {"a": _q(100, 0.33)},
        tq.Setting(0.25, 0.05, 2): {"a": _q(100, 0.34)},
    }
    assert tq.pick(results, DEFAULT).setting == tq.Setting(0.25, 0.05, 2)


def test_defaults_within_the_tie_margin_are_kept():
    results = {
        DEFAULT: {"a": _q(100, 0.33)},
        tq.Setting(0.1, 0.05, 4): {"a": _q(100, 0.30)},
    }
    chosen = tq.pick(results, DEFAULT)
    assert chosen.keep_defaults and chosen.setting == DEFAULT
    assert "keep them" in chosen.reason


def test_no_merging_leaves_only_settings_that_attach_nothing():
    results = {
        DEFAULT: {"a": _q(100, 3.0)},
        tq.Setting(0.25, 0.25, 0): {"a": _q(100, 2.0)},
        tq.Setting(0.1, 0.05, 4): {"a": _q(100, 0.1)},
    }
    chosen = tq.pick(results, DEFAULT, allow_merging=False)
    assert chosen.setting == tq.Setting(0.25, 0.25, 0)


def test_nothing_tracked_is_said_not_picked():
    results = {DEFAULT: {"a": _q(0, 0)}, tq.Setting(0.1, 0.1, 0): {"a": _q(0, 0)}}
    chosen = tq.pick(results, DEFAULT)
    assert chosen.setting is None
    assert "No setting tracked a subject" in chosen.reason
