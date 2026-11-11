"""An arena id is only unique within one recording (#377).

**The class.** A selection made once and applied to many recordings may only name something
that means the same thing on all of them. Arena ids are minted per recording — `arena_001` is
"whichever arena I drew first *here*" — so a cohort picked by id tracks a different region on
each recording, and nothing fails: every run completes, every record is well-formed, and the
numbers are simply about different parts of the frame.

**Why it survived.** It is invisible on the fixture everybody writes. Give two recordings one
arena each and their ids line up by construction, so an id-keyed selection is right by
accident. It needs a cohort where the ids *diverge* — which is exactly the cohort #350 made
reachable, since a field of view held by a tracking run now refuses to be replaced and the way
to correct it is to add a second arena beside it.
"""

import pytest

from detail.api.experiment import Experiment


def _square(dx=0.0):
    return [(0.1 + dx, 0.1), (0.9 + dx, 0.1), (0.9 + dx, 0.9), (0.1 + dx, 0.9)]


def _cohort(tmp_path):
    """Two recordings whose arena ids do NOT line up.

    `already_tracked` kept an old field of view (a tracking run holds it), so its corrected
    one is `arena_002`. `fresh` had none, so the same corrected field of view is `arena_001`.
    """
    exp = Experiment.create("p", tmp_path)
    tracked = exp.create_recording("already_tracked")
    tracked.define_arena("old rig", [(0, _square())])
    tracked.define_arena("rig A camera", [(0, _square(0.05))])
    fresh = exp.create_recording("fresh")
    fresh.define_arena("rig A camera", [(0, _square(0.05))])
    return exp, tracked, fresh


def test_the_ids_diverge_in_this_fixture(tmp_path):
    """Guard the guard: a fixture where the ids happen to line up proves nothing."""
    _exp, tracked, fresh = _cohort(tmp_path)
    assert [a["id"] for a in tracked.arenas] == ["arena_001", "arena_002"]
    assert [a["id"] for a in fresh.arenas] == ["arena_001"]
    by_name = lambda r: {a["name"]: a["id"] for a in r.arenas}  # noqa: E731
    assert by_name(tracked)["rig A camera"] != by_name(fresh)["rig A camera"]


def test_a_name_resolves_to_each_recordings_own_arena(tmp_path):
    """The fix, asserted as the resolved id per recording — not that validate returned."""
    from detail.pipeline.tracking import TrackingModule
    from detail.storage import recording as recording_store

    _exp, tracked, fresh = _cohort(tmp_path)
    module = TrackingModule()
    resolved = {}
    for recording in (tracked, fresh):
        data = recording_store.read_recording(recording.dir)
        resolved[recording.id] = module._resolve_arena(
            {"arena_name": "rig A camera"}, data, recording.id
        )
    assert resolved["already_tracked"] == ["arena_002"]
    assert resolved["fresh"] == ["arena_001"]


def test_the_same_name_is_the_same_polygon_on_every_recording(tmp_path):
    """The property the ids cannot carry.

    Asserted on the *keyframes*, because that is what the tracker actually filters against.
    Comparing resolved ids would pass on the bug — the whole point is that the ids differ.
    """
    from detail.pipeline.tracking import TrackingModule
    from detail.storage import recording as recording_store

    _exp, tracked, fresh = _cohort(tmp_path)
    module = TrackingModule()
    shapes = []
    for recording in (tracked, fresh):
        data = recording_store.read_recording(recording.dir)
        [arena_id] = module._resolve_arena(
            {"arena_name": "rig A camera"}, data, recording.id
        )
        shapes.append(recording.arena_keyframes(arena_id))
    assert (
        shapes[0] == shapes[1]
    ), "the same named field of view resolved to two polygons"

    # And the old id-keyed selection is what it replaced: `arena_001` on each recording is
    # two *different* polygons, which is the bug stated as a test.
    by_id = [r.arena_keyframes("arena_001") for r in (tracked, fresh)]
    assert by_id[0] != by_id[1]


def test_a_recording_without_that_name_is_refused_by_name(tmp_path):
    from detail.pipeline.tracking import TrackingModule
    from detail.storage import recording as recording_store

    exp, _tracked, _fresh = _cohort(tmp_path)
    other = exp.create_recording("no_arena")
    data = recording_store.read_recording(other.dir)
    with pytest.raises(ValueError, match="no field of view named 'rig A camera'"):
        TrackingModule()._resolve_arena({"arena_name": "rig A camera"}, data, other.id)


def test_an_id_is_still_accepted_so_old_scripts_keep_working(tmp_path):
    """`--arena arena_001` and every config already on disk go through the id path."""
    from detail.pipeline.tracking import TrackingModule
    from detail.storage import recording as recording_store

    _exp, tracked, _fresh = _cohort(tmp_path)
    data = recording_store.read_recording(tracked.dir)
    module = TrackingModule()
    assert module._resolve_arena({"arena_ids": ["arena_002"]}, data, tracked.id) == [
        "arena_002"
    ]
    # a bare id handed to --arena, which the option now types as a name
    assert module._resolve_arena({"arena_name": "arena_002"}, data, tracked.id) == [
        "arena_002"
    ]
    assert module._resolve_arena({}, data, tracked.id) == []
