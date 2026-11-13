"""Skeletons (§26): names are labels; edges and symmetry pairs are declared and checked.

There is no body axis and no side on a skeleton: the heading an embedding aligns to is a
heading definition chosen per run (§14), and which image side "left" falls on is learned from
the example clicks. A test here that asked for `front`, `back` or `side` would be asking for
the design #626 had and the review removed.
"""

import json

import pytest

from detail.core.skeleton import PRESETS, Skeleton
from detail.core.skeleton import check_against_model
from detail.core.skeleton import flip_index
from detail.core.skeleton import pairs_from_flip_index


def _rat(**overrides) -> Skeleton:
    fields = dict(
        names=("nose", "left_ear", "right_ear", "tail_base", "tail_tip"),
        edges=(("nose", "left_ear"), ("nose", "right_ear"), ("nose", "tail_base")),
        pairs=(("left_ear", "right_ear"),),
    )
    fields.update(overrides)
    return Skeleton(**fields)


def test_columns_pairs_and_midline():
    s = _rat()
    s.validate()
    assert s.names == ("nose", "left_ear", "right_ear", "tail_base", "tail_tip")
    assert len(s) == 5 and s.index("tail_base") == 3
    assert s.pair_indices() == [(1, 2)]
    # the midline is the points in no pair, in column order — nothing marks it
    assert s.midline_names() == ("nose", "tail_base", "tail_tip")


def test_names_are_arbitrary():
    s = Skeleton(names=("A", "zz 9"), edges=(("A", "zz 9"),))
    s.validate()
    assert s.index("zz 9") == 1


def test_a_skeleton_carries_no_axis_and_no_side():
    fields = set(Skeleton.__dataclass_fields__) - {"_index"}
    assert fields == {"names", "edges", "pairs", "layout"}
    for gone in ("axis", "front", "back", "side", "unset", "infer_roles"):
        assert not hasattr(Skeleton, gone), gone


def test_round_trips_through_a_dict_and_json():
    s = _rat()
    assert Skeleton.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert "layout" not in s.to_dict(), "a skeleton never drawn carries no layout"
    drawn = s.with_layout(tuple((0.125 * i, 0.5) for i in range(len(s))))
    back = Skeleton.from_dict(json.loads(json.dumps(drawn.to_dict())))
    assert back == drawn and back.layout == drawn.layout


def test_the_layout_is_how_it_was_drawn_not_what_it_means():
    """The editor's layout rides with the skeleton and outside its identity: two skeletons
    drawn differently are equal, digest the same, and validate only as a picture would.
    """
    s = _rat()
    drawn = s.with_layout(tuple((0.2, 0.1 * i) for i in range(len(s))))
    assert drawn == s and drawn.digest() == s.digest()
    assert Skeleton.from_dict(drawn.to_dict()).digest() == s.digest()
    drawn.validate()
    with pytest.raises(ValueError, match="places 1 point"):
        s.with_layout(((0.5, 0.5),)).validate()
    with pytest.raises(ValueError, match="outside the canvas"):
        s.with_layout(tuple((1.5, 0.5) for _ in range(len(s)))).validate()


def test_default_layout_puts_the_midline_in_the_middle_and_pairs_either_side():
    from detail.core.skeleton import default_layout

    s = _rat()  # nose, left_ear, right_ear, tail_base, tail_tip; one pair of ears
    layout = default_layout(s)
    assert len(layout) == len(s)
    nose, left, right, base, tip = layout
    assert nose[0] == base[0] == tip[0] == 0.5
    assert left[0] < 0.5 < right[0] and left[1] == right[1]
    assert nose[1] < left[1] < base[1] < tip[1], "head at the top, in column order"
    assert default_layout(Skeleton(names=())) == ()


def test_unknown_name_lookup_names_the_skeleton():
    with pytest.raises(KeyError, match="has \\['nose'"):
        _rat().index("snout")


@pytest.mark.parametrize(
    "skeleton,fragment",
    [
        (Skeleton(names=()), "at least one keypoint"),
        (Skeleton(names=("a", "a")), "appears twice"),
        (Skeleton(names=(" ",)), "non-empty"),
        (Skeleton(names=("a",), edges=(("a", "x"),)), "unknown keypoint 'x'"),
        (Skeleton(names=("a",), edges=(("a", "a"),)), "to itself"),
        (Skeleton(names=("a", "b"), edges=(("a", "b"), ("b", "a"))), "listed twice"),
        (Skeleton(names=("a",), pairs=(("a", "b"),)), "unknown keypoint 'b'"),
        (Skeleton(names=("a",), pairs=(("a", "a"),)), "with itself"),
        (
            Skeleton(names=("a", "b", "c"), pairs=(("a", "b"), ("a", "c"))),
            "in two pairs",
        ),
    ],
)
def test_validate_names_the_inconsistency(skeleton, fragment):
    with pytest.raises(ValueError, match=fragment):
        skeleton.validate()


def test_the_digest_is_the_structure_in_column_order():
    """A run counts its skeleton by this digest (§26 *Identity*): the same structure read back
    from disk digests the same; a reordered column or a dropped pair is a different skeleton.
    """
    s = _rat()
    assert len(s.digest()) == 16
    assert Skeleton.from_dict(s.to_dict()).digest() == s.digest()
    reordered = Skeleton(names=s.names[::-1], edges=s.edges, pairs=s.pairs)
    assert reordered.digest() != s.digest()
    assert _rat(pairs=()).digest() != s.digest()


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #


def test_the_presets_are_named_for_the_body_plan_and_view():
    """The user's naming (2026-10-09), ids and labels both: a species guess ("Fish or worm")
    read as a joke, and two presets share a view, so the count is in the id."""
    assert {pid: label for pid, (label, _sk) in PRESETS.items()} == {
        "quadruped_overhead_12": "Quadruped overhead view (12 points)",
        "quadruped_overhead_8": "Quadruped overhead view (8 points)",
        "quadruped_side_5": "Quadruped side view (5 points)",
        "midline_3": "Midline (3 points)",
    }
    for pid, (_label, sk) in PRESETS.items():
        assert pid.endswith(f"_{len(sk.names)}"), "the id carries the point count"


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_every_preset_is_valid(preset_id):
    label, skeleton = PRESETS[preset_id]
    skeleton.validate()
    assert label
    # a preset ships a layout, so the editor opens on a picture of the animal; each pair's
    # left member is drawn left of its right member
    assert skeleton.layout is not None and len(skeleton.layout) == len(skeleton)
    for left, right in skeleton.pairs:
        assert skeleton.layout[skeleton.index(left)][0] < (
            skeleton.layout[skeleton.index(right)][0]
        )


def test_the_overhead_preset_is_the_research_skeleton():
    _label, skeleton = PRESETS["quadruped_overhead_12"]
    assert len(skeleton) == 12
    assert skeleton.pairs == (
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    )
    assert skeleton.midline_names() == (
        "nose",
        "neck_base",
        "mid_back",
        "tail_base",
        "tail_mid",
        "tail_tip",
    )


# --- a skeleton against a pose checkpoint (§12 *Keypoint models in the catalog*) ---- #
# A checkpoint carries K and Ultralytics' flip_idx, not names; these are the checks an
# imported pose .pt's supplied skeleton must pass.
def test_flip_index_swaps_pairs_and_keeps_the_midline_at_its_new_home():
    assert flip_index(_rat()) == [0, 2, 1, 3, 4]
    assert flip_index(_rat(pairs=())) == [0, 1, 2, 3, 4]


def test_pairs_from_flip_index_reads_the_swaps_and_refuses_a_non_pairing():
    assert pairs_from_flip_index([0, 2, 1, 3]) == {frozenset((1, 2))}
    assert pairs_from_flip_index([0, 1]) == set()
    with pytest.raises(ValueError, match="must swap points in pairs"):
        pairs_from_flip_index([1, 2, 0])
    with pytest.raises(ValueError, match="outside 0..1"):
        pairs_from_flip_index([0, 5])


def test_a_skeleton_that_fits_the_model_passes():
    check_against_model(_rat(), 5, [0, 2, 1, 3, 4])
    # a skeleton with no pairs passes a model with no flip_idx: nothing to check
    check_against_model(_rat(pairs=()), 5, None)
    check_against_model(_rat(pairs=()), 5, [0, 1, 2, 3, 4])


def test_a_different_point_count_is_refused_naming_both():
    with pytest.raises(
        ValueError, match="model places 17 points but the skeleton has 5"
    ):
        check_against_model(_rat(), 17, list(range(17)))


def test_pairs_the_model_does_not_swap_are_refused_naming_both_sides():
    with pytest.raises(
        ValueError,
        match=r"model swaps \(nose, tail_base\).*skeleton pairs \(left_ear, right_ear\)",
    ):
        check_against_model(_rat(), 5, [3, 1, 2, 0, 4])
    # the model swaps a pair the skeleton does not declare
    with pytest.raises(ValueError, match="skeleton pairs none"):
        check_against_model(_rat(pairs=()), 5, [0, 2, 1, 3, 4])


def test_pairs_with_no_flip_idx_are_refused_because_they_cannot_be_checked():
    with pytest.raises(ValueError, match="no flip_idx.*use a skeleton without pairs"):
        check_against_model(_rat(), 5, None)


def test_a_flip_idx_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="flip_idx has 4 entries for 5 points"):
        check_against_model(_rat(), 5, [0, 2, 1, 3])


def test_a_pair_named_the_wrong_way_round_is_not_caught():
    # The stated limit (§12): (left, right) and (right, left) give the same flip_idx.
    check_against_model(_rat(pairs=(("right_ear", "left_ear"),)), 5, [0, 2, 1, 3, 4])


def test_no_reader_indexes_a_stored_skeletons_names_directly():
    """A stored skeleton is read through ``Skeleton.from_dict``, the one reader that knows
    both shapes on disk (§26's and the older ``keypoints`` one). Indexing ``["names"]`` on a
    stored dict is the bug that failed every catalog read on a student promoted before §26:
    ``KeyError: 'names'``. Read the skeleton, then its names."""
    import re
    from pathlib import Path

    import detail

    pattern = re.compile(r"""\[["']skeleton["']\]\s*\[["']names["']\]""")
    root = Path(detail.__file__).parent
    offenders = [
        f"{path.relative_to(root).as_posix()}:{n}"
        for path in sorted(root.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, offenders
    assert pattern.search('meta["skeleton"]["names"]')  # the check can fire
