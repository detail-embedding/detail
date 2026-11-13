"""A per-class patch method matches each detection against its own class's example set.

The class: one bank for animals that look different. Two mice told apart at detection as two
classes — a white and a black one — were matched against one set of clicks, and the clicks on
the white mouse failed on the black one (§26 *One example set per detection class*). A
per-class run builds one bank per class on one backbone and picks each detection's by its
class. Nothing fails if that pick is wrong — every detection still gets a plausible pose — so
the check makes the two sets disagree on purpose: the same image, the same skeleton, the dots
named in opposite orders. Which set placed a detection's points is then visible in where its
``nose`` lands, and swapping the class-to-set map must swap the answers.

The backbone is the colour stand-in of ``test_patch_match.py`` (the one legitimate mock: the
external inference call).
"""

import numpy as np
import pytest

from detail.core.examples import ExampleFrame, ExampleMark, ExampleSet
from detail.core.keypoints import InstanceFrame
from detail.models.patch_match.dinov3_match import DINOv3Match
from tests.models.test_patch_match import BBOX, DOTS, _ColourBackbone, _frame

#: Red, green and blue dots, left to right, on every frame.
REF = [(200, 240), (320, 240), (440, 240)]


def _set(set_id: str, names_left_to_right, positions=(REF,)) -> ExampleSet:
    """A set whose clicks name the three dots, left to right, ``names_left_to_right``."""
    return ExampleSet(
        id=set_id,
        name=set_id,
        purpose="example",
        created_at="",
        skeleton=DOTS,
        frames=[
            ExampleFrame(
                image=f"frames/{set_id}_{i}.png",
                recording_id="recording_001",
                tracking_run_id="track_01",
                track_id=0,
                part_id="part_001",
                frame_idx=i,
                t_ns=i,
                bbox=BBOX,
                marks=[
                    ExampleMark(name, x=x / 640, y=y / 480)
                    for name, (x, y) in zip(names_left_to_right, pts)
                ],
            )
            for i, pts in enumerate(positions)
        ],
    )


#: Nose on the red (left) dot.
WHITE = ("nose", "mid_back", "tail_base")
#: Nose on the blue (right) dot: the same image read the other way round.
BLACK = ("tail_base", "mid_back", "nose")


def _model(monkeypatch, sets: dict[str, tuple], positions=(REF,)) -> DINOv3Match:
    images = [_frame(pts)[0] for pts in positions]
    model = DINOv3Match(
        class_examples={
            cls: (_set(f"ex_{cls}", names, positions), images)
            for cls, names in sets.items()
        },
        crop_px=448,
    )
    monkeypatch.setattr(model, "_make_backbone", lambda: _ColourBackbone())
    model.load("cpu")
    return model


def _noses(model, classes: dict[int, str]) -> dict[int, float]:
    """Each detection's placed nose x, in pixels, on a frame where the animal has moved."""
    frame, body = _frame([(230, 250), (330, 230), (420, 236)])
    [result] = model.predict_batch(
        [
            InstanceFrame(
                frame_idx=3,
                t_ns=30,
                frame=frame,
                boxes={row: BBOX for row in classes},
                masks={row: body for row in classes},
                classes=classes,
            )
        ]
    )
    j = DOTS.names.index("nose")
    return {
        int(row): float(result.points[i, j, 0] * 640)
        for i, row in enumerate(result.detection_row)
    }


def test_each_detection_is_matched_against_its_own_classs_set(monkeypatch):
    pytest.importorskip("torch")
    model = _model(monkeypatch, {"Resident": WHITE, "Intruder": BLACK})
    noses = _noses(model, {3: "Resident", 5: "Intruder"})
    assert (
        abs(noses[3] - 230) < 16
    ), "the Resident's nose is on the red dot, as its set says"
    assert (
        abs(noses[5] - 420) < 16
    ), "the Intruder's nose is on the blue dot, as its set says"


def test_swapping_the_classes_sets_swaps_the_placements(monkeypatch):
    """The counterexample: a method that matched every detection against one bank would
    place both noses alike, whichever class owns which set."""
    pytest.importorskip("torch")
    model = _model(monkeypatch, {"Resident": BLACK, "Intruder": WHITE})
    noses = _noses(model, {3: "Resident", 5: "Intruder"})
    assert abs(noses[3] - 420) < 16 and abs(noses[5] - 230) < 16


def test_a_detection_of_a_class_with_no_set_is_refused(monkeypatch):
    pytest.importorskip("torch")
    model = _model(monkeypatch, {"Resident": WHITE, "Intruder": BLACK})
    with pytest.raises(ValueError, match="'Pup', which has no example set"):
        _noses(model, {3: "Resident", 5: "Pup"})


def test_sets_on_different_skeletons_are_refused():
    from detail.core.skeleton import Skeleton

    other = _set("ex_other", WHITE)
    other.skeleton = Skeleton(names=("nose", "mid_back", "tail_base", "tail_tip"))
    images = [_frame(REF)[0]]
    with pytest.raises(ValueError, match="same skeleton"):
        DINOv3Match(
            class_examples={
                "Resident": (_set("ex_r", WHITE), images),
                "Intruder": (other, images),
            }
        )


def test_resolved_and_held_out_are_per_class(monkeypatch):
    """Each class learns its side rules from its own clicks, and the held-out check scores a
    class's frames against that class's other frames only."""
    pytest.importorskip("torch")
    positions = (REF, [(210, 250), (330, 230), (430, 236)])
    model = _model(monkeypatch, {"Resident": WHITE, "Intruder": BLACK}, positions)
    assert set(model.resolved()["by_class"]) == {"Resident", "Intruder"}
    rows = model.score_held_out()
    assert {r["class"] for r in rows} == {"Resident", "Intruder"}
    assert len(rows) == 2 * 2 * len(DOTS.names)  # per class: 2 frames x 3 points
    for r in rows:
        assert r["error_px"] < 16, r  # its own class's clicks, not the other's
