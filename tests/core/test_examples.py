"""Example sets (§26): the marks a method learns from, and the audit frames that score one."""

import json

import pytest

from detail.core.examples import (
    ExampleFrame,
    ExampleMark,
    ExampleSet,
)
from detail.core.skeleton import PRESETS

SKELETON = PRESETS["quadruped_side_5"][
    1
]  # nose, left_ear, right_ear, mid_back, tail_base
IMAGE = "frames/a1f3c9.png"
HASHES = {IMAGE: "0" * 64, "frames/9c2e71.png": "1" * 64}


def _marks(**visible):
    out = []
    for i, name in enumerate(SKELETON.names):
        if visible.get(name, True):
            out.append(ExampleMark(name, x=0.1 * i + 0.1, y=0.5))
        else:
            out.append(ExampleMark(name, visible=False))
    return out


def _frame(image=IMAGE, marks=None, **overrides) -> ExampleFrame:
    fields = dict(
        image=image,
        recording_id="recording_000",
        tracking_run_id="track_01",
        track_id=0,
        part_id="part_001",
        frame_idx=12,
        t_ns=400_000_000,
        bbox=(0.1, 0.2, 0.5, 0.9),
        marks=_marks() if marks is None else marks,
    )
    fields.update(overrides)
    return ExampleFrame(**fields)


def _set(frames=None, **overrides) -> ExampleSet:
    fields = dict(
        id="ex_01",
        name="side rats",
        purpose="example",
        created_at="2026-10-07T10:00:00Z",
        skeleton=SKELETON,
        frames=[_frame()] if frames is None else frames,
    )
    fields.update(overrides)
    return ExampleSet(**fields)


def test_round_trips_through_json_with_schema_version_first():
    s = _set()
    d = s.to_dict(HASHES)
    assert list(d)[0] == "schema_version" and d["schema_version"] == 1
    assert d["digest"] == s.digest(HASHES)
    back = ExampleSet.from_dict(json.loads(json.dumps(d)))
    assert back == s
    assert back.frames[0].key() == ("recording_000", "part_001", 400_000_000, 0)


def test_unknown_fields_survive_a_round_trip():
    """Older code must never delete what newer code added (storage review, class D7): a key
    this version does not know rides along at every level and is written back."""
    d = _set().to_dict()
    d["later"] = {"added": "by a newer version"}
    d["frames"][0]["later_frame"] = 7
    d["frames"][0]["marks"][0]["later_mark"] = True
    out = ExampleSet.from_dict(d).to_dict()
    assert out["later"] == {"added": "by a newer version"}
    assert out["frames"][0]["later_frame"] == 7
    assert out["frames"][0]["marks"][0]["later_mark"] is True


def test_the_digest_follows_content_and_images_not_the_name():
    a, b = _set(), _set(name="another label")
    assert a.digest(HASHES) == b.digest(HASHES)
    swapped = dict(HASHES, **{IMAGE: "f" * 64})
    assert a.digest(swapped) != a.digest(HASHES), "an image swapped under its name"
    moved = _set(frames=[_frame(marks=_marks(nose=False))])
    assert moved.digest(HASHES) != a.digest(HASHES), "a changed mark"
    assert len(a.digest(HASHES)) == 16


KEY = ("recording_009", "part_001", 40_210_000_000, 0)


def test_a_rejection_is_per_bucket_and_a_null_bucket_is_every_bucket():
    s = _set()
    s.reject(KEY, "left-toward")
    assert s.is_rejected(KEY, "left-toward")
    assert not s.is_rejected(KEY, "away"), "turned down for one bucket, not for others"
    assert not s.is_rejected(KEY), "a bucketless strategy may still draw it"
    s.reject(KEY)
    assert s.is_rejected(KEY, "away") and s.is_rejected(KEY)
    s.reject(KEY, "left-toward")
    assert len(s.rejected) == 2, "a repeated rejection is recorded once"


def test_rejections_round_trip_and_stay_out_of_the_digest():
    s = _set()
    before = s.digest(HASHES)
    s.reject(KEY, "toward")
    d = s.to_dict(HASHES)
    assert d["rejected"] == [
        {
            "recording_id": "recording_009",
            "part_id": "part_001",
            "t_ns": 40_210_000_000,
            "track_id": 0,
            "bucket": "toward",
        }
    ]
    assert d["digest"] == before, "a refusal changes no run's input"
    d["rejected"][0]["later"] = 1
    back = ExampleSet.from_dict(json.loads(json.dumps(d)))
    assert back.is_rejected(KEY, "toward")
    assert back.to_dict()["rejected"][0]["later"] == 1
    assert "rejected" not in _set().to_dict(), "an empty list is not written"


def test_a_not_visible_mark_is_not_visible():
    s = _set(frames=[_frame(marks=_marks(tail_base=False))])
    assert s.visible_counts()["tail_base"] == 0
    assert s.visible_counts()["nose"] == 1
    assert s.labeled(0), "not visible is still a decision on that point"


@pytest.mark.parametrize(
    "example_set, fragment",
    [
        (_set(frames=[]), "has no frames"),
        (
            _set(
                frames=[_frame(marks=[*_marks(), ExampleMark("whisker", x=0.5, y=0.5)])]
            ),
            "does not name",
        ),
        (
            _set(frames=[_frame(marks=_marks(tail_base=False))]),
            "\\['tail_base'\\] visible",
        ),
        (_set(purpose="audit"), "never from audit labels"),
    ],
)
def test_check_runnable_names_what_to_fix(example_set, fragment):
    with pytest.raises(ValueError, match=fragment):
        example_set.check_runnable()


def test_a_runnable_set_passes():
    _set(
        frames=[_frame(), _frame(image="frames/9c2e71.png", t_ns=800_000_000)]
    ).check_runnable()


def test_a_visible_point_needs_coordinates_and_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="has no x, y"):
        ExampleMark("nose").validate()
    with pytest.raises(ValueError, match="unknown mark kind 'blob'"):
        ExampleMark.from_dict({"target": "nose", "kind": "blob"})


def test_reserved_kinds_are_carried_without_a_readout():
    mark = ExampleMark.from_dict(
        {"target": "head", "kind": "region", "rle": "3 4 5", "positive": False}
    )
    assert mark.kind == "region" and mark.positive is False
    assert mark.extra == {"rle": "3 4 5"}
    assert mark.to_dict()["rle"] == "3 4 5" and mark.to_dict()["positive"] is False


def test_audit_bookkeeping_rounds_skips_and_pending():
    frames = [
        _frame(round=1),
        _frame(image="frames/9c2e71.png", t_ns=800_000_000, round=1, marks=[]),
        _frame(
            image="frames/zz.png", t_ns=900_000_000, round=2, marks=[], skipped=True
        ),
    ]
    s = _set(frames=frames, purpose="audit", sources={"recording_000": "track_01"})
    assert s.labeled(0) and not s.labeled(1) and not s.labeled(2)
    assert s.pending() == [1]
    assert s.counts() == {"drawn": 3, "labeled": 1, "skipped": 1, "pending": 1}
    assert s.next_round() == 3
    d = s.to_dict()
    assert d["sources"] == {"recording_000": "track_01"}
    assert d["frames"][2]["skipped"] is True and "skipped" not in d["frames"][0]
    assert ExampleSet.from_dict(d) == s


def test_a_purpose_outside_the_two_is_refused():
    d = _set().to_dict()
    d["purpose"] = "training"
    with pytest.raises(ValueError, match="purpose 'training'"):
        ExampleSet.from_dict(d)
