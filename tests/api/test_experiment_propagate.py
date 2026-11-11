"""Experiment.propagate_spatial — pushing one recording's setup onto others (§10, #342).

Two rules carry the weight here, and both are invisible to a test that only checks the
copy landed:

- **It never replaces.** A target that already holds the category keeps what it has and is
  reported. So every one of these fixtures needs an *occupied* target as well as an empty
  one; a fixture where every target is empty passes with the refusal deleted.
- **A definition's timeline does not travel.** Keyframe motion is flattened to frame 0 and
  an active window refuses the trip, so the tests drive a *moving* polygon and a *windowed*
  region rather than the static ones the happy path uses.
"""

import pytest

from detail.api.experiment import Experiment
from detail.storage import recording as recording_store


def _full_frame():
    return [(0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]


def _square(offset=0.0):
    return [
        (0.1 + offset, 0.1),
        (0.9, 0.1),
        (0.9, 0.9),
        (0.1, 0.9 + offset),
    ]


def _experiment_with_recordings(tmp_path, *ids):
    exp = Experiment.create("p", tmp_path)
    for rid in ids:
        exp.create_recording(rid)
    return exp


def _set_resolution(recording, width, height):
    data = recording_store.read_recording(recording.dir)
    data["video"] = {"resolution": [width, height], "files": [], "storage": "default"}
    recording_store.write_recording(recording.dir, data)


def test_copy_lands_on_every_other_recording_by_default(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2", "r3")
    exp.resolve_recording("r1").define_arena("rig", [(0, _square())])

    result = exp.propagate_field_of_view("r1")

    assert sorted(result.copied) == ["r2", "r3"]
    assert result.skipped == []
    for rid in ("r2", "r3"):
        assert [a["name"] for a in exp.resolve_recording(rid).arenas] == ["rig"]


def test_copy_is_limited_to_explicit_targets(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2", "r3")
    exp.resolve_recording("r1").define_arena("rig", [(0, _square())])

    result = exp.propagate_field_of_view("r1", targets=["r2"])

    assert result.copied == ["r2"]
    assert exp.resolve_recording("r3").arenas == []


def test_copy_excludes_the_source_even_if_named(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2")
    exp.resolve_recording("r1").define_arena("rig", [(0, _square())])

    result = exp.propagate_field_of_view("r1", targets=["r1", "r2"])

    assert result.copied == ["r2"]
    assert [a["name"] for a in exp.resolve_recording("r1").arenas] == ["rig"]


def test_copy_requires_a_source(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2")
    with pytest.raises(ValueError, match="no field of view to copy"):
        exp.propagate_field_of_view("r1")


@pytest.mark.parametrize(
    "key, seed_source, seed_target, reads",
    [
        (
            "fov",
            lambda r: r.define_arena("rig", [(0, _square())]),
            lambda r: r.define_arena("its own", [(0, _square(0.05))]),
            lambda r: [a["name"] for a in r.arenas],
        ),
    ],
)
def test_a_kind_grained_target_that_has_one_keeps_it(
    tmp_path, key, seed_source, seed_target, reads
):
    """A field of view is refused whole, because a second one is not a richer setup.

    The batch tracking picker intersects arena *ids* across recordings, so two arenas mean
    one dropdown naming different polygons on different recordings.
    """
    exp = _experiment_with_recordings(tmp_path, "r1", "empty", "taken")
    seed_source(exp.resolve_recording("r1"))
    seed_target(exp.resolve_recording("taken"))

    result = exp.propagate_spatial(key, "r1")

    assert result.copied == ["empty"]
    assert [rid for rid, _ in result.skipped] == ["taken"]
    assert "already has" in result.skipped[0][1]
    assert reads(exp.resolve_recording("taken")) == ["its own"]


@pytest.mark.parametrize(
    "key, define, reads",
    [
        ("roi", "define_roi", lambda r: sorted(x["name"] for x in r.rois)),
        ("point", "define_point", lambda r: sorted(x["name"] for x in r.points)),
    ],
)
def test_a_set_grained_target_keeps_its_names_and_takes_the_rest(
    tmp_path, key, define, reads
):
    """Regions and points are sets, so a collision is judged per *name*.

    Refusing the whole kind because one name clashes would kill "push one more zone onto
    the cohort" — and on a real experiment it refuses nearly every target, since each
    recording usually has its own copy of the common region already.
    """
    exp = _experiment_with_recordings(tmp_path, "r1", "empty", "taken")
    src = exp.resolve_recording("r1")
    shape = [(0, [(0.5, 0.5)])] if key == "point" else _full_frame()
    getattr(src, define)("shared", shape)
    getattr(src, define)("extra", shape)
    getattr(exp.resolve_recording("taken"), define)("shared", shape)

    result = exp.propagate_spatial(key, "r1")

    assert sorted(result.copied) == ["empty", "taken"]
    assert reads(exp.resolve_recording("empty")) == ["extra", "shared"]
    # Its own "shared" is untouched; only "extra" arrived.
    assert reads(exp.resolve_recording("taken")) == ["extra", "shared"]
    assert [name for name, _ in result.refused] == ["shared"]
    assert "already exists" in result.refused[0][1]


def test_calibration_is_not_replaced_either(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "empty", "taken")
    for rid in ("r1", "empty", "taken"):
        _set_resolution(exp.resolve_recording(rid), 1920, 1080)
    exp.resolve_recording("r1").set_distance_calibration(
        [(0.25, 0.5), (0.75, 0.5)], 200.0
    )
    exp.resolve_recording("taken").set_pixels_per_mm(1.0)

    result = exp.propagate_calibration("r1")

    assert result.copied == ["empty"]
    assert [rid for rid, _ in result.skipped] == ["taken"]
    assert exp.resolve_recording("taken").pixels_per_mm == 1.0
    assert abs(exp.resolve_recording("empty").pixels_per_mm - 4.8) < 1e-9


def test_a_locked_target_names_the_runs_holding_it(tmp_path):
    """The refusal has to be actionable, and "delete it" is not the whole instruction
    when a completed tracking run is holding the arena — "delete these runs, then it" is.
    """
    exp = _experiment_with_recordings(tmp_path, "r1", "locked")
    exp.resolve_recording("r1").define_arena("rig", [(0, _square())])
    locked = exp.resolve_recording("locked")
    arena_id = locked.define_arena("in use", [(0, _square())])
    data = recording_store.read_recording(locked.dir)
    data["tracking_runs"] = [
        {"id": "trk_003", "arena_ids": [arena_id]},
        {"id": "trk_007", "arena_ids": [arena_id]},
    ]
    recording_store.write_recording(locked.dir, data)

    result = exp.propagate_field_of_view("r1")

    assert result.copied == []
    assert [rid for rid, _ in result.skipped] == ["locked"]
    reason = result.skipped[0][1]
    assert "trk_003" in reason and "trk_007" in reason
    assert [a["name"] for a in exp.resolve_recording("locked").arenas] == ["in use"]


def test_camera_motion_does_not_travel_and_is_reported(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2")
    exp.resolve_recording("r1").define_arena(
        "panning rig", [(0, _square()), (900, _square(0.05))]
    )

    result = exp.propagate_field_of_view("r1")

    assert result.copied == ["r2"] and result.flattened == ["panning rig"]
    target = exp.resolve_recording("r2")
    assert len(target.arena_keyframes(target.arenas[0]["id"])) == 1


def test_a_region_with_an_active_window_is_left_behind_and_named(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2")
    src = exp.resolve_recording("r1")
    src.define_roi("nest", _full_frame())
    src.define_roi("post-injection zone", _full_frame(), active=(5000, 9000))

    result = exp.propagate_spatial("roi", "r1")

    assert result.copied == ["r2"]
    assert [name for name, _ in result.refused] == ["post-injection zone"]
    assert [x["name"] for x in exp.resolve_recording("r2").rois] == ["nest"]


def test_progress_reports_within_the_batch(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2", "r3")
    exp.resolve_recording("r1").define_arena("rig", [(0, _square())])
    seen = []

    exp.propagate_field_of_view(
        "r1", progress=lambda stage, fraction, detail: seen.append((stage, fraction))
    )

    assert seen, "the batch reported no progress at all"
    assert {stage for stage, _ in seen} == {"copying field of view"}
    assert any(0 < fraction < 1 for _stage, fraction in seen)
    assert seen[-1][1] == 1.0


def test_an_unknown_category_is_refused(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "r2")
    with pytest.raises(ValueError, match="unknown spatial category"):
        exp.propagate_spatial("nonsense", "r1")


def test_calibration_skips_a_target_of_another_shape_with_the_reason(tmp_path):
    exp = _experiment_with_recordings(tmp_path, "r1", "same", "square")
    _set_resolution(exp.resolve_recording("r1"), 1920, 1080)
    _set_resolution(exp.resolve_recording("same"), 960, 540)
    _set_resolution(exp.resolve_recording("square"), 1280, 1024)
    exp.resolve_recording("r1").set_distance_calibration(
        [(0.5, 0.25), (0.5, 0.75)], 200.0
    )

    result = exp.propagate_calibration("r1")

    assert result.copied == ["same"]
    assert [rid for rid, _ in result.skipped] == ["square"]
    assert "different video shape" in result.skipped[0][1]
    assert exp.resolve_recording("square").distance_calibration is None
