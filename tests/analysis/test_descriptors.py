"""Descriptors: the numbers that say what a cluster *is*.

The class these guard is **a value invented where there was no data**. Every descriptor here
averages a per-frame signal, and the signals carry ``NaN`` wherever they are undefined — across a
tracking gap, on a one-frame bout, on a mask with no pixels. Filling those with zero produces a
plausible number that says "nothing happened", which is indistinguishable from a real still
moment and wrong in the same direction every time. So the tests below drive the *edges* — one
frame, a gap, an absent input — rather than a comfortable ten-frame fixture.

The second class is **a surface that hardcodes what a module can answer**. Descriptors declare
the levels they mean anything at and the inputs they need; `describe` and `available` read those.
A panel listing them by hand would show `duration` on a pose cluster, where a member is a single
frame and a duration is not a thing.
"""

import numpy as np
import pytest

from detail.analysis import descriptors as d
from detail.analysis.descriptors import DESCRIPTORS, FrameSlice, available, describe


def _slice(n=10, *, gap_at=None, **kwargs) -> FrameSlice:
    """A straight-line walk at 2 px/frame, optionally with a tracking gap."""
    frame_idx = np.arange(n)
    if gap_at is not None:
        frame_idx = np.concatenate([np.arange(gap_at), np.arange(gap_at, n) + 100])
    fields = dict(
        embeddings=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        centroid=np.stack([np.arange(n) * 2.0, np.zeros(n)], axis=1),
        area=np.full(n, 100.0),
        # The box **moves with the subject**. A constant box made every box-derived descriptor
        # NaN or zero whatever the code did — `tortuosity` reported NaN because its path length
        # was zero, so the declaration guard below passed while the descriptor read one input
        # and declared another. Found by running the loader on a real recording, not here.
        bbox=np.stack(
            [
                np.arange(n) * 2.0 - 5.0,
                np.full(n, -10.0),
                np.arange(n) * 2.0 + 5.0,
                np.full(n, 10.0),
            ],
            axis=1,
        ),
    )
    fields.update(kwargs)
    return FrameSlice(
        frame_idx=frame_idx, t_ns=frame_idx.astype(np.int64) * 1_000_000_000, **fields
    )


def test_the_registry_is_populated():
    """A sweep over an empty registry passes by testing nothing."""
    assert len(DESCRIPTORS) >= 8, sorted(DESCRIPTORS)


# --- the numbers are right ------------------------------------------------------------ #
def test_a_straight_walk_reads_as_a_straight_walk():
    """The one case where every value is known by hand, so a sign or an off-by-one shows."""
    values = describe(_slice(10))
    assert values["mask_speed"] == pytest.approx(2.0)
    assert values["net_displacement"] == pytest.approx(18.0)
    assert values["tortuosity"] == pytest.approx(1.0)
    assert values["duration_s"] == pytest.approx(9.0)
    assert values["bbox_aspect"] == pytest.approx(2.0)
    assert values["mask_area"] == pytest.approx(100.0)


def test_tortuosity_separates_circling_from_travelling():
    """The descriptor that earns its place: two units with the *same* mean speed and opposite
    meaning. A metric set without it calls a tight circle and a walk across the arena the same
    thing, which is the confusion a person naming a cluster is trying to resolve."""
    n = 32
    angle = np.linspace(0, 2 * np.pi, n)
    circle = _slice(n, centroid=np.stack([np.cos(angle), np.sin(angle)], axis=1) * 10)
    line = _slice(n)
    assert describe(circle)["tortuosity"] < 0.2
    assert describe(line)["tortuosity"] == pytest.approx(1.0)


def test_appearance_change_is_zero_when_nothing_changes():
    """`pose_step` on identical embeddings is 0, not NaN: the frames *are* adjacent and the
    change between them is genuinely nothing."""
    assert describe(_slice(10))["pose_step"] == pytest.approx(0.0, abs=1e-9)


# --- the edges ------------------------------------------------------------------------- #
def test_a_one_frame_unit_has_no_steps_and_says_so():
    """Every step-based descriptor is undefined, and NaN is what says that. Zero would read as
    a still animal, which is a different claim and one this data cannot make."""
    values = describe(_slice(1))
    for name in ("mask_speed", "mask_area_step", "pose_step", "duration_s"):
        assert np.isnan(values[name]), f"{name} invented a value for a single frame"
    # What *is* defined from one frame stays defined.
    assert values["bbox_aspect"] == pytest.approx(2.0)


def test_a_gap_is_skipped_rather_than_stepped_across():
    """A step spanning a tracking gap compares frames 100 apart and reports it as one frame of
    motion — a large fabricated speed exactly where the data is missing.

    The subject **moves during the gap**, and that is the whole fixture. The first version of
    this test walked at a constant 2 px/frame, so a step leaking across the gap measured 2 as
    well and the assertion held whether or not the gap was honoured — it passed against the
    mutation it exists for. The position has to jump by more than a frame's worth for the
    difference to be visible at all.
    """
    positions = np.stack([np.arange(10) * 2.0, np.zeros(10)], axis=1)
    positions[5:] += 500.0  # 100 frames unobserved, and the animal went somewhere
    with_gap = describe(_slice(10, gap_at=5, centroid=positions))
    assert with_gap["mask_speed"] == pytest.approx(2.0), "the gap leaked into the mean"


def test_a_unit_that_is_all_gap_has_nothing_to_average():
    """The all-NaN slice. `np.nanmean` warns here and the suite runs warnings as errors, so
    this is also what keeps that from becoming a failure in the middle of a run."""
    every_other = FrameSlice(
        frame_idx=np.arange(0, 20, 2),  # stride 1, so no two rows are adjacent
        t_ns=np.arange(0, 20, 2, dtype=np.int64) * 1_000_000_000,
        centroid=np.zeros((10, 2)),
    )
    assert np.isnan(describe(every_other)["mask_speed"])


def test_an_absent_input_gives_nan_rather_than_raising():
    """A run embedded before the geometry existed answers fewer descriptors. Opening it must
    still work — fewer numbers is a state to render, not an error."""
    bare = FrameSlice(frame_idx=np.arange(5), t_ns=np.arange(5, dtype=np.int64))
    values = describe(bare)
    assert np.isnan(values["mask_speed"]) and np.isnan(values["pose_step"])
    assert values["duration_s"] == pytest.approx(4e-9)


def test_available_names_only_what_this_slice_can_answer():
    """So a surface shows the numbers it has rather than a column of blanks."""
    bare = FrameSlice(frame_idx=np.arange(5), t_ns=np.arange(5, dtype=np.int64))
    assert "mask_speed" not in available(bare)
    assert "duration_s" in available(bare)
    assert "mask_speed" in available(_slice(5))


# --- the declarations are what surfaces read ------------------------------------------- #
def test_duration_is_not_offered_at_the_frame_level():
    """A pose cluster's member is one frame, and a frame has no duration. The module says so;
    a panel that listed descriptors by hand would show it and render nonsense."""
    assert "frame" not in DESCRIPTORS["duration_s"].levels
    assert "duration_s" not in describe(_slice(5), level="frame")


@pytest.mark.parametrize("name", sorted(DESCRIPTORS))
def test_every_descriptor_declares_what_it_reads(name):
    """`needs` is what lets a surface say *why* a number is missing. A descriptor that reads an
    optional field without declaring it would silently return NaN with no explanation.
    """
    descriptor = DESCRIPTORS[name]
    assert descriptor.label and descriptor.unit
    # `position` is derived — the mask centroid when there is one, the box centre otherwise —
    # so satisfying it means clearing *both* sources rather than one.
    # Derived from the slice's optional fields rather than listed: the list was extended by
    # hand for each new input, which is the list going stale the day one is forgotten.
    import dataclasses

    optional = {f.name for f in dataclasses.fields(FrameSlice) if f.default is None} - {
        "camera_view",
        "points_run",
    }
    assert set(descriptor.needs) <= optional | {"position"}
    for need in descriptor.needs:
        cleared = (
            {"centroid": None, "bbox": None} if need == "position" else {need: None}
        )
        missing = _slice(5, **cleared)
        assert np.isnan(
            descriptor.fn(missing)
        ), f"{name} declares it needs {need} but produced a number without it"


def test_position_falls_back_to_the_box_centre():
    """A run with no mask geometry still answers where the subject went.

    The two sources are interchangeable for *displacement and path shape* and not for speed,
    which is why `mask_speed` and `bbox_speed` remain separate descriptors. This pins the
    fallback so a later change cannot quietly make one of them mean the other.
    """
    no_mask = _slice(10, centroid=None)
    assert no_mask.position is not None
    values = describe(no_mask)
    # The box travels 2 px/frame over nine steps, exactly as the mask centroid does — so the
    # fallback answers *where it went* with the real number rather than a zero.
    assert values["net_displacement"] == pytest.approx(18.0)
    assert values["tortuosity"] == pytest.approx(1.0)
    assert values["bbox_speed"] == pytest.approx(2.0)
    # And speed is not substituted: the mass centre carries deformation the box does not, which
    # is why these stay two descriptors rather than one with a fallback.
    assert np.isnan(values["mask_speed"]), "mask speed must not fall back to the box"


def test_a_descriptor_that_needs_an_input_it_does_not_declare_is_caught():
    """Guard the guard: the sweep above must fail on an undeclared dependency."""
    undeclared = d.Descriptor(
        name="x", label="x", unit="x", fn=lambda s: float(np.mean(s.area)), needs=()
    )
    assert not undeclared.needs
    with pytest.raises((TypeError, AttributeError)):
        undeclared.fn(FrameSlice(frame_idx=np.arange(3), t_ns=np.arange(3)))


# --- the part axis, tested rather than asserted ---------------------------------------- #
def test_a_descriptor_written_for_the_body_works_on_another_part():
    """The future-proofing claim, checked.

    A part is *another slice*, not a wider schema — so when a keypoint layer arrives, the
    descriptors written today apply to it unchanged. Two slices of the same shape with different
    `part` labels must produce values that differ only because the data does.
    """
    body = _slice(10, part="body")
    head = _slice(
        10, part="head", centroid=np.stack([np.arange(10) * 4.0, np.zeros(10)], axis=1)
    )
    assert body.part == "body" and head.part == "head"
    assert describe(head)["mask_speed"] == pytest.approx(
        2 * describe(body)["mask_speed"]
    )


def test_the_shape_helpers_are_the_ones_dynamics_uses():
    """One definition of a step, not two.

    `dynamics` is a research surface the pipeline may not import, so the formulas moved here and
    it aliases them. Two copies would let the segmentation video and the interpretation panel
    disagree about what mask speed means — and they would disagree slowly, in a way nobody would
    look for.
    """
    from detail.analysis import dynamics

    assert dynamics._step_curve is d.step_curve
    assert dynamics._cosine_step is d.cosine_step
    assert dynamics._adjacent is d.adjacent
    assert dynamics._bbox_center is d.bbox_center


# --- composition: the pose -> behavior bridge -------------------------------------------- #
def _with_labels(labels) -> FrameSlice:
    n = len(labels)
    return FrameSlice(
        frame_idx=np.arange(n),
        t_ns=np.arange(n, dtype=np.int64) * 1_000_000_000,
        labels=np.asarray(labels, dtype=np.int64),
    )


def test_a_bout_holding_one_posture_reads_as_one():
    values = describe(_with_labels([3, 3, 3, 3]))
    assert values["n_postures"] == 1
    assert values["dominant_posture_share"] == pytest.approx(1.0)
    assert values["posture_switch_rate"] == pytest.approx(0.0)


def test_a_bout_alternating_two_postures_reads_as_busy():
    values = describe(_with_labels([0, 1, 0, 1, 0]))
    assert values["n_postures"] == 2
    assert values["dominant_posture_share"] == pytest.approx(3 / 5)
    # Four changes over the four seconds between the first and last labelled frame.
    assert values["posture_switch_rate"] == pytest.approx(1.0)


def test_a_switch_across_an_unlabelled_gap_is_counted():
    """The real-data case that corrected this descriptor.

    A bout on a 30-minute recording carried labels ``1111......0000000``: the posture plainly
    changed, but across a stretch the pose run did not label. Requiring both frames of a pair to
    be labelled reported **2 postures and 0.00 switches/s** — a contradiction on its face, and
    the wrong answer. Counting over the labelled frames in sequence gives a lower bound, which
    a gap can only ever make conservative.
    """
    values = describe(_with_labels([1, 1, -1, -1, -1, -1, 0, 0]))
    assert values["n_postures"] == 2
    assert values["posture_switch_rate"] > 0, "a visible change reported as none"


def test_the_rate_is_over_labelled_time_not_the_whole_bout():
    """Half a bout unlabelled must not halve its measured busyness — the pose run's coverage is
    not a property of the behavior."""
    dense = describe(_with_labels([0, 1, 0, 1, 0]))
    padded = describe(_with_labels([0, 1, 0, 1, 0] + [-1] * 20))
    assert padded["posture_switch_rate"] == pytest.approx(dense["posture_switch_rate"])


def test_an_unlabelled_bout_has_no_composition():
    """A pose run that covers none of this bout says nothing about it — which is NaN, not zero
    postures. Zero would read as a bout with no posture at all."""
    values = describe(_with_labels([-1, -1, -1]))
    assert np.isnan(values["n_postures"])
    assert np.isnan(values["dominant_posture_share"])
    assert np.isnan(values["posture_switch_rate"])


def test_composition_is_absent_when_no_pose_run_covers_the_unit():
    """The ordinary case: a behavior run with no pose run beside it. The rows are not offered,
    rather than offered full of blanks."""
    assert "n_postures" not in available(_slice(5))
    assert "n_postures" in available(_with_labels([0, 1]))
