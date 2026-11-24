"""Skeletons with left/right pairs, for tests that need more than a midline.

The research method measured these two, and they shipped as presets
(``quadruped_overhead_12``, ``quadruped_side_5``) until 2026-11-22, when the presets were
trimmed to the two midlines. Tests of pairs, flips and the decode's side judgement still need
them, so they live here as test data rather than as something a user picks.
"""

from detail.core.skeleton import Skeleton

#: Overhead view, 12 points, three left/right pairs; ordered head to tail.
OVERHEAD_12 = Skeleton(
    names=(
        "nose",
        "left_ear",
        "right_ear",
        "neck_base",
        "left_shoulder",
        "right_shoulder",
        "mid_back",
        "left_hip",
        "right_hip",
        "tail_base",
        "tail_mid",
        "tail_tip",
    ),
    edges=(
        ("nose", "neck_base"),
        ("left_ear", "neck_base"),
        ("right_ear", "neck_base"),
        ("neck_base", "left_shoulder"),
        ("neck_base", "right_shoulder"),
        ("neck_base", "mid_back"),
        ("mid_back", "tail_base"),
        ("tail_base", "left_hip"),
        ("tail_base", "right_hip"),
        ("tail_base", "tail_mid"),
        ("tail_mid", "tail_tip"),
    ),
    pairs=(
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ),
    layout=(
        (0.5, 0.08),
        (0.41, 0.17),
        (0.59, 0.17),
        (0.5, 0.27),
        (0.38, 0.36),
        (0.62, 0.36),
        (0.5, 0.48),
        (0.4, 0.6),
        (0.6, 0.6),
        (0.5, 0.68),
        (0.5, 0.81),
        (0.5, 0.94),
    ),
)

#: Side view, 5 points, one pair (the ears).
SIDE_5 = Skeleton(
    names=("nose", "left_ear", "right_ear", "mid_back", "tail_base"),
    edges=(
        ("nose", "left_ear"),
        ("nose", "right_ear"),
        ("nose", "mid_back"),
        ("mid_back", "tail_base"),
    ),
    pairs=(("left_ear", "right_ear"),),
    layout=(
        (0.12, 0.5),
        (0.3, 0.3),
        (0.36, 0.38),
        (0.56, 0.36),
        (0.88, 0.5),
    ),
)
