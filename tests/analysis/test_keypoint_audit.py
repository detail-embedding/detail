"""Scoring a keypoint track run against an audit set (§26 "Auditing a keypoint track run").

Known answers on a synthetic frame: 800×600, the tracked box (200, 150)–(600, 450) in pixels, so
body size ``s = sqrt(400 * 300) = 346.41`` px. The labels sit at fixed pixels on every frame and
the run's predictions are offset from them by chosen amounts, so each figure can be worked out by
hand: 10 px is 0.029 s (inside PCK@0.05's 17.3 px), 30 px is 0.087 s (outside it, inside
PCK@0.1's 34.6 px).
"""

import math

import numpy as np
import pytest

from detail.analysis import keypoint_audit as ka
from detail.core.examples import ExampleFrame, ExampleMark, ExampleSet
from detail.core.skeleton import Skeleton

W, H = 800, 600
BBOX = (0.25, 0.25, 0.75, 0.75)
BODY = math.sqrt(400 * 300)
PART = "part_001"
SKELETON = Skeleton(
    names=("nose", "left_ear", "right_ear", "tail_base"),
    edges=(("nose", "tail_base"),),
    pairs=(("left_ear", "right_ear"),),
)
HEADING = {"front": ["nose"], "back": ["tail_base"]}
LABEL_PX = {
    "nose": (400, 200),
    "left_ear": (380, 230),
    "right_ear": (420, 230),
    "tail_base": (400, 400),
}
WH = {"recording_001": (W, H), "recording_002": (W, H)}


def _mark(name, offset=(0, 0), *, at=None) -> ExampleMark:
    x, y = at or LABEL_PX[name]
    return ExampleMark(name, x=(x + offset[0]) / W, y=(y + offset[1]) / H)


def _hidden(name) -> ExampleMark:
    return ExampleMark(name, visible=False)


def _frame(i, *, marks=None, recording="recording_001", tracking="track_01", **kw):
    return ExampleFrame(
        image=f"frames/{i:03d}.png",
        recording_id=recording,
        tracking_run_id=tracking,
        track_id=0,
        part_id=PART,
        frame_idx=10 * i,
        t_ns=1_000_000 * i,
        bbox=BBOX,
        marks=[_mark(n) for n in SKELETON.names] if marks is None else marks,
        round=1,
        **kw,
    )


def _audit(frames, skeleton=SKELETON) -> ExampleSet:
    return ExampleSet(
        id="ex_01",
        name="test",
        purpose="audit",
        created_at="",
        skeleton=skeleton,
        frames=list(frames),
        sources={"recording_001": "track_01"},
    )


def _run(n, offsets, *, names=SKELETON.names, unplaced=(), below=()) -> ka.RunRows:
    """A run with one row per audit time; ``offsets[name]`` moves that prediction,
    ``unplaced`` holds ``(row, name)`` cells the run left out and ``below`` the cells it placed
    under its cut-off (coordinates kept, not visible)."""
    xy = np.zeros((n, len(names), 2))
    for j, name in enumerate(names):
        x, y = LABEL_PX.get(name, (400, 300))
        dx, dy = offsets.get(name, (0, 0))
        xy[:, j] = ((x + dx) / W, (y + dy) / H)
    visible = np.ones((n, len(names)), bool)
    for i, name in unplaced:
        xy[i, names.index(name)] = np.nan
        visible[i, names.index(name)] = False
    for i, name in below:
        visible[i, names.index(name)] = False
    part = ka.RunPart(
        t_ns=np.arange(n, dtype=np.int64) * 1_000_000,
        points=xy,
        scores=np.ones((n, len(names))),
        visible=visible,
    )
    return ka.RunRows("kt_01", "track_01", tuple(names), {(0, PART): part})


def _identity() -> dict:
    return {n: n for n in SKELETON.names}


def _score(frames, run, mapping=None, excluded=(), audit_skeleton=SKELETON, **kw):
    return ka.score(
        _audit(frames, audit_skeleton),
        {"recording_001": run},
        _identity() if mapping is None else mapping,
        set(excluded),
        WH,
        **kw,
    )


@pytest.fixture
def report():
    """Ten frames. nose +10 px, left_ear +30 px, right_ear exact but unplaced on frames 8
    and 9, tail_base exact and labeled not visible on frame 9 (where the run still places it).
    """
    frames = [_frame(i) for i in range(10)]
    frames[9].marks[3] = _hidden("tail_base")
    run = _run(
        10,
        {"nose": (10, 0), "left_ear": (0, 30)},
        unplaced=[(8, "right_ear"), (9, "right_ear")],
    )
    return _score(frames, run, heading=HEADING, heading_name="nose-tail")


def test_each_keypoints_known_error_pck_and_misses(report):
    def value(name, metric):
        return report.row("keypoint", name, metric)["value"]

    assert value("nose", "error_px_median") == pytest.approx(10.0)
    assert value("nose", "error_body_median") == pytest.approx(10 / BODY)
    assert value("nose", "pck05") == 1.0
    assert value("left_ear", "pck05") == 0.0  # 30 px > 0.05 s
    assert value("left_ear", "pck10") == 1.0  # 30 px < 0.1 s
    # Unplaced on 2 of 10 frames: misses count against PCK, not in the error median.
    assert value("right_ear", "missed") == pytest.approx(0.2)
    assert value("right_ear", "pck10") == pytest.approx(0.8)
    assert value("right_ear", "error_px_median") == pytest.approx(0, abs=1e-6)
    assert report.row("keypoint", "right_ear", "error_px_median")["n"] == 8


def test_groups_pool_their_keypoints(report):
    def value(name, metric):
        return report.row("group", name, metric)["value"]

    # all: 39 visible cells (tail_base hidden once); hits at 0.1 s: 10 + 10 + 8 + 9.
    assert value("all", "pck10") == pytest.approx(37 / 39)
    assert value("all", "pck05") == pytest.approx(27 / 39)
    assert value("unpaired", "pck05") == 1.0
    assert value("paired", "pck05") == pytest.approx(8 / 20)
    # 17 zeros, ten 10s and ten 30s: the median of 37 is a 10.
    assert value("all", "error_px_median") == pytest.approx(10.0)
    assert report.row("recording", "recording_001", "pck10")["value"] == pytest.approx(
        37 / 39
    )


def test_a_proportion_carries_a_wilson_interval_not_a_collapsed_one(report):
    row = report.row("keypoint", "nose", "pck10")
    assert row["value"] == 1.0 and row["n"] == 10
    assert row["ci_high"] == 1.0
    assert 0.70 < row["ci_low"] < 0.75  # 10 of 10 is not certainty


def test_a_median_carries_a_seeded_frame_bootstrap(report):
    row = report.row("group", "all", "error_px_median")
    assert row["ci_low"] <= row["value"] <= row["ci_high"]
    assert row["ci_low"] < row["ci_high"], "37 values of 0/10/30 do not pin the median"
    again = ka.median_with_interval(
        np.array([0.0, 10.0, 30.0] * 4), np.repeat(np.arange(4), 3)
    )
    assert again == ka.median_with_interval(
        np.array([0.0, 10.0, 30.0] * 4), np.repeat(np.arange(4), 3)
    ), "seeded: the same report twice is the same report"


def test_a_figure_from_fewer_than_eight_frames_is_withheld(report):
    # One frame labels tail_base hidden: whether the run wrongly places such points rests on
    # one frame, so the figure is withheld and only n is reported.
    row = report.row("keypoint", "tail_base", "false_placement")
    assert row["n"] == 1
    assert row["value"] is None and row["ci_low"] is None and row["floor"] is None


def test_eight_frames_is_enough_and_seven_is_not():
    for n, shown in ((7, False), (8, True)):
        scored = _score([_frame(i) for i in range(n)], _run(n, {}))
        row = scored.row("keypoint", "nose", "pck10")
        assert row["n"] == n
        assert (row["value"] is not None) is shown, n


def test_the_heading_angle_error(report):
    # Labeled back->front (0, -200); predicted (10, -200): atan(10/200) = 2.86 deg. tail_base
    # is labeled not visible on frame 9, so 9 frames.
    row = report.row("heading", "nose-tail", "error_deg_median")
    assert row["value"] == pytest.approx(math.degrees(math.atan2(10, 200)), abs=1e-6)
    assert row["n"] == 9
    assert row["floor"] == pytest.approx(0.0, abs=1e-6), "every label at one box place"
    assert report.row("heading", "nose-tail", "within_15deg")["value"] == 1.0
    assert report.row("heading", "nose-tail", "flipped")["value"] == 0.0


def test_no_heading_definition_no_heading_rows():
    scored = _score([_frame(i) for i in range(8)], _run(8, {}))
    assert not [r for r in scored.rows if r["scope"] == "heading"]


def test_the_pair_its_floor_and_the_floor_on_identical_labels(report):
    correct = report.row("pair", "left_ear|right_ear", "correct")
    assert correct["value"] == 1.0 and correct["n"] == 8
    # The random-assignment rate: a coin decides which side each resolved frame lands on.
    assert correct["floor"] == pytest.approx(0.5)
    assert report.row("pair", "left_ear|right_ear", "swapped")["floor"] == 0.5
    assert report.row("pair", "left_ear|right_ear", "collapsed")["floor"] is None
    # Every label sits at the same place in the box, so the box alone already finds them.
    assert report.row("keypoint", "nose", "pck05")["floor"] == 1.0
    assert report.row("keypoint", "nose", "error_px_median")["floor"] == pytest.approx(
        0
    )


def test_the_floor_never_uses_a_frames_own_label():
    """Leave-one-out: seven noses at x=400 and one at 480. The odd frame's floor is the
    others' mean (400), 80 px away — 70 px if its own label had pulled the mean to 410.
    """
    frames = [_frame(i) for i in range(8)]
    frames[7].marks[0] = _mark("nose", (80, 0))
    scored = _score(frames, _run(8, {}))
    nose = [r for r in scored.frames if r["keypoint"] == "nose"]
    assert nose[7]["floor_error_body"] == pytest.approx(80 / BODY)
    assert nose[0]["floor_error_body"] == pytest.approx((80 / 7) / BODY)


def test_the_floor_is_computed_over_the_frames_the_runs_figure_used():
    """The run places the nose on the eight easy frames and misses it on twelve hard ones.
    Its error median rests on the eight; so must the floor's, or a run that skips its hard
    frames is compared against a floor that includes them."""
    frames = [_frame(i) for i in range(8)] + [
        _frame(
            i,
            marks=[_mark("nose", at=(600, 200))]
            + [_mark(n) for n in SKELETON.names[1:]],
        )
        for i in range(8, 20)
    ]
    run = _run(20, {}, unplaced=[(i, "nose") for i in range(8, 20)])
    scored = _score(frames, run)
    # an easy frame's floor: the other 7 easy (x=400) and 12 hard (x=600) noses
    easy = abs((7 * 400 + 12 * 600) / 19 - 400)
    row = scored.row("keypoint", "nose", "error_px_median")
    assert row["n"] == 8
    assert row["floor"] == pytest.approx(easy)
    hard = abs((8 * 400 + 11 * 600) / 19 - 600)
    assert hard < easy, "over all twenty the floor's median would be the hard frames'"


def test_pck_and_missed_floors_cover_the_same_visible_cells(report):
    nose = report.row("keypoint", "nose", "missed")
    assert nose["floor"] == 0.0, "the box alone places every point with another label"


@pytest.mark.parametrize(
    "pa, pb, outcome",
    [
        ((380, 230), (420, 230), "correct"),
        ((420, 230), (380, 230), "swapped"),
        ((381, 230), (385, 232), "collapsed"),
        ((380, 330), (420, 230), "off"),
    ],
)
def test_pair_outcomes(pa, pb, outcome):
    assert ka.pair_outcome(pa, pb, (380, 230), (420, 230), BODY) == outcome


def test_the_runs_own_example_frames_are_excluded_and_counted():
    """Labels score a method; they never build one."""
    frames = [_frame(i) for i in range(10)]
    scored = _score(
        frames,
        _run(10, {}),
        excluded={("recording_001", PART, 0, 0), ("recording_001", PART, 3_000_000, 0)},
    )
    assert scored.coverage["excluded"] == 2 and scored.coverage["scored"] == 8
    assert {r["frame_idx"] for r in scored.frames}.isdisjoint({0, 30})


def test_frames_the_run_does_not_cover_are_counted_not_scored():
    frames = [_frame(i) for i in range(8)] + [
        _frame(8, tracking="track_02"),  # the run read another tracking run
        _frame(9, recording="recording_002"),  # that recording has no such run
        _frame(10),  # no row at its time (a run made every n frames)
    ]
    scored = _score(frames, _run(10, {}))
    assert scored.coverage["uncovered"] == 3 and scored.coverage["scored"] == 8


def test_skipped_and_unfinished_frames_are_not_scored():
    frames = [_frame(i) for i in range(10)]
    frames[0].skipped = True
    del frames[1].marks[0]  # labeling stopped partway through this frame
    scored = _score(frames, _run(10, {}))
    assert scored.coverage["skipped"] == 1 and scored.coverage["pending"] == 1
    assert scored.coverage["scored"] == 8


def test_a_label_edited_after_the_report_is_counted_and_flagged():
    frames = [_frame(i) for i in range(8)]
    frames[2].marks[0].extra["edited_after_report"] = True
    scored = _score(frames, _run(8, {}))
    assert scored.coverage["edited_after_report"] == 1
    flagged = {r["frame_idx"] for r in scored.frames if r["edited_after_report"]}
    assert flagged == {20}


def test_a_name_map_scores_renamed_keypoints_and_reports_the_unmapped():
    audit_skeleton = Skeleton(names=("snout", "tail_base", "tail_tip"))
    marks = [
        _mark("snout", at=LABEL_PX["nose"]),
        _mark("tail_base"),
        _mark("tail_tip", at=(400, 540)),
    ]
    frames = [
        _frame(i, marks=[ExampleMark(**vars(m)) for m in marks]) for i in range(8)
    ]
    scored = _score(
        frames,
        _run(8, {"nose": (10, 0)}),
        {"snout": "nose", "tail_base": "tail_base", "tail_tip": None},
        audit_skeleton=audit_skeleton,
    )
    assert scored.row("keypoint", "snout", "error_px_median")["value"] == pytest.approx(
        10.0
    )
    assert scored.unmapped == ["tail_tip"]
    assert scored.row("keypoint", "tail_tip", "pck10") is None
    assert {r["run_keypoint"] for r in scored.frames} == {"nose", "tail_base"}


def test_runs_placing_different_keypoints_are_refused():
    runs = {
        "recording_001": _run(4, {}),
        "recording_002": _run(4, {}, names=("nose",)),
    }
    with pytest.raises(ValueError, match="different keypoints"):
        ka.score(_audit([_frame(0)]), runs, _identity(), set(), WH)


def test_a_map_naming_a_keypoint_the_run_lacks_is_refused():
    with pytest.raises(ValueError, match="does not place"):
        _score([_frame(0)], _run(1, {}), dict(_identity(), nose="snout"))


def test_a_point_below_the_runs_cutoff_is_missed():
    run = _run(8, {}, below=[(i, "nose") for i in range(8)])
    scored = _score([_frame(i) for i in range(8)], run)
    assert scored.row("keypoint", "nose", "missed")["value"] == 1.0
    assert scored.row("keypoint", "tail_base", "missed")["value"] == 0.0
