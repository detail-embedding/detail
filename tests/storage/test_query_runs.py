"""Which instances sit on the user's own example clips (``query_runs.example_instance_ids``, §19).

An example names its subject by ``track_id`` and an instance by ``subject``; both are read
with ``core.utils.subject_key`` (``tests/core/test_subject_key.py``), so a spec stored without a
track or with a digit string still matches.
"""

import pytest

from detail.storage import behavior_output as bo
from detail.storage import query_runs

SEC = 1_000_000_000


def _instance(subject, start_s, end_s, *, behavior="q01", recording="recording_000"):
    return bo.BehaviorInstance(
        recording,
        subject,
        int(start_s * SEC),
        int(end_s * SEC),
        int(start_s * 30),
        int(end_s * 30),
        behavior,
        "",
    )


def _behavior(track_id, start_s, end_s):
    return query_runs.Behavior(
        id="q01",
        name="close contact",
        examples=[
            {
                "recording_id": "recording_000",
                "track_id": track_id,
                "start_t_ns": int(start_s * SEC),
                "end_t_ns": int(end_s * SEC),
            }
        ],
    )


def test_an_instance_on_its_example_is_matched():
    on_example = _instance(2, 131, 132)
    elsewhere = _instance(2, 200, 201)
    found = query_runs.example_instance_ids(
        [_behavior(2, 130, 133)], [on_example, elsewhere]
    )
    assert found == {on_example.instance_id}


@pytest.mark.parametrize(
    "example_track, instance_subject",
    [(0, 1), (1, 0)],
)
def test_an_example_matches_only_its_own_subject(example_track, instance_subject):
    """Two tracks are different subjects."""
    row = _instance(instance_subject, 131, 132)
    assert not query_runs.example_instance_ids(
        [_behavior(example_track, 130, 133)], [row]
    )


@pytest.mark.parametrize("example_track", [0, "0", None])
def test_one_animals_example_still_matches_however_it_was_stored(example_track):
    """A spec without ``track_id`` (or a digit string) is track 0, as it always was."""
    row = _instance(None, 131, 132)
    behavior = _behavior(example_track, 130, 133)
    if example_track is None:
        del behavior.examples[0]["track_id"]
    assert query_runs.example_instance_ids([behavior], [row]) == {row.instance_id}
