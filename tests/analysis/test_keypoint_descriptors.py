"""The three-keypoint measures (nose, mid back, tailbase) and the loader that joins them on.

Guards:

- **A skeleton names its points its own way.** ``tail_base``, ``TailBase`` and ``tailbase``
  are one point; a skeleton with none of a role's names offers none of its descriptors rather
  than reading the wrong point.
- **A point under the run's cut-off is missing, not zero** — a nose lost during rearing must
  not become a nose at the top of the frame.
- **Heading knows forward; the mask's axis does not** — a half-turn reads as 180°/step here.
"""

import numpy as np
import pytest

from detail.analysis.descriptors import (
    DESCRIPTORS,
    FrameSlice,
    in_scope,
    point_named,
)


def _slice(points, n=5, *, view="side", rate_hz=10.0):
    return FrameSlice(
        frame_idx=np.arange(n),
        t_ns=(np.arange(n) * (1e9 / rate_hz)).astype(np.int64),
        points={k: np.asarray(v, dtype=float) for k, v in points.items()},
        body_length=50.0,
        camera_view=view,
    )


def _still(x, y, n=5):
    return np.tile([x, y], (n, 1))


@pytest.mark.parametrize("name", ["tail_base", "TailBase", "tail-base", "tailbase"])
def test_a_role_is_found_whatever_the_skeleton_calls_it(name):
    assert point_named({name: _still(1, 2)}, "tailbase") is not None


def test_a_skeleton_without_the_role_offers_nothing_for_it():
    s = _slice({"left_ear": _still(0, 0), "right_ear": _still(1, 0)})
    assert point_named(s.points, "nose") is None
    assert np.isnan(DESCRIPTORS["bend"].fn(s))


def test_nose_height_is_up_from_the_tailbase_in_body_lengths():
    # y grows downward: a nose 25 px above the tailbase is half a body length up.
    s = _slice({"nose": _still(100, 75), "tail_base": _still(50, 100)})
    assert DESCRIPTORS["nose_height"].fn(s) == pytest.approx(0.5)
    assert not in_scope(DESCRIPTORS["nose_height"], _slice({}, view="top_down"))


def test_a_straight_back_has_no_bend_and_a_right_angle_has_ninety():
    straight = _slice(
        {"nose": _still(100, 0), "mid_back": _still(50, 0), "tailbase": _still(0, 0)}
    )
    bent = _slice(
        {"nose": _still(50, 50), "mid_back": _still(50, 0), "tailbase": _still(0, 0)}
    )
    assert DESCRIPTORS["bend"].fn(straight) == pytest.approx(0.0, abs=1e-6)
    assert DESCRIPTORS["bend"].fn(bent) == pytest.approx(90.0)


def test_heading_turning_knows_which_way_is_forward():
    n = 3
    tail = _still(0, 0, n)
    nose = np.array(
        [[10, 0], [-10, 0], [10, 0]], dtype=float
    )  # a full about-face per step
    s = _slice({"nose": nose, "tailbase": tail}, n=n)
    assert DESCRIPTORS["heading_turning"].fn(s) == pytest.approx(180.0 * 10.0)


def test_head_scanning_is_the_nose_moving_against_the_body():
    n = 4
    mid = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
    walking = _slice({"nose": mid + [20, 0], "mid_back": mid}, n=n)
    scanning = _slice(
        {
            "nose": np.array([[20, 0], [20, 5], [20, 0], [20, 5]]),
            "mid_back": _still(0, 0, n),
        },
        n=n,
    )
    assert DESCRIPTORS["head_scanning"].fn(walking) == pytest.approx(0.0)
    assert DESCRIPTORS["head_scanning"].fn(scanning) == pytest.approx(5 * 10 / 50)


# --- the loader ------------------------------------------------------------------------ #
def test_points_are_joined_on_the_slices_frames_in_pixels_and_scored(tmp_path):
    from detail.api.frame_slices import keypoint_run_for, keypoints_for
    from tests.api.test_frame_slices import _recording
    from tests.experiments import write_keypoint_track_run

    recording = _recording(tmp_path, 1)
    write_keypoint_track_run(recording.dir, names=("nose", "tail_base"), n_frames=10)
    frames = np.array(
        [3, 4, 12]
    )  # 12 is past the run's rows: NaN, not a neighbour's point
    points = keypoints_for(recording.dir, "kt_01", 0, frames)
    assert set(points) == {"nose", "tail_base"}
    assert points["nose"][0].tolist() == pytest.approx([50.0, 30.0])  # 100 x 100 frame
    assert np.isnan(points["nose"][2]).all()
    record = {"id": "emb_01", "tracking_run_id": "track_01"}
    assert keypoint_run_for(recording, record) == "kt_01"
    assert (
        keypoint_run_for(recording, {**record, "keypoint_run_id": "kt_07"}) == "kt_07"
    )


def test_facing_is_the_tailbase_to_nose_direction_as_two_numbers():
    # Nose to the right of the tailbase: facing right, not up or down the image.
    right = _slice({"nose": _still(100, 50), "tailbase": _still(50, 50)})
    assert DESCRIPTORS["facing_right"].fn(right) == pytest.approx(1.0)
    assert DESCRIPTORS["facing_down"].fn(right) == pytest.approx(0.0, abs=1e-9)
    down = _slice({"nose": _still(50, 100), "tailbase": _still(50, 50)})
    assert DESCRIPTORS["facing_down"].fn(down) == pytest.approx(1.0)


def test_facing_that_turns_back_on_itself_has_no_direction():
    """Half the bout facing left and half facing right averages to no direction, which is
    missing, not 'facing up' — an angle mean would have said 90 degrees."""
    nose = np.array([[100, 50], [100, 50], [0, 50], [0, 50]], dtype=float)
    tail = np.full((4, 2), [50.0, 50.0])
    s = _slice({"nose": nose, "tailbase": tail}, n=4)
    assert np.isnan(DESCRIPTORS["facing_right"].fn(s))
    assert np.isnan(DESCRIPTORS["facing_down"].fn(s))


def test_the_place_measures_are_marked_as_place_and_nothing_else_is():
    place = {name for name, d in DESCRIPTORS.items() if d.place}
    assert place == {"facing_right", "facing_down", "position_x", "position_y"}
