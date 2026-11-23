"""``query.yaml``'s behaviors — the removed clips a review adds (§20 *The audit*)."""

from detail.storage import query_runs

REMOVED = {
    "recording_id": "recording_001",
    "subject": 0,
    "t_ns": 510_000_000,
    "end_t_ns": 765_000_000,
    "at": "2026-11-22T10:00:00Z",
}


def test_a_behavior_written_before_removed_existed_reads_as_none_removed():
    old = {"id": "q01", "name": "rearing", "examples": [], "cutoff": 0.4}
    assert query_runs.Behavior.from_dict(old).removed == []


def test_removed_clips_round_trip_and_key_as_clip_refs(tmp_path):
    behavior = query_runs.Behavior("q01", "rearing", cutoff=0.4, removed=[REMOVED])
    state = query_runs.QueryState(
        embed_run_id="emb_01", track_id=0, method="soft_knn", behaviors=[behavior]
    )
    query_runs.write_query_state(tmp_path, state)
    back = query_runs.read_query_state(tmp_path).behavior("q01")
    assert back.removed == [REMOVED]
    assert back.removed_keys() == {("recording_001", 0, 510_000_000, 765_000_000)}


def test_told_kind_names_the_clip_a_span_sits_on_and_an_example_wins():
    """The one rule for "the method finding its own input", now saying which input: an example,
    a non-example, or neither — matched against the span's **own** behavior, recording and
    subject, by any shared time."""
    behavior = query_runs.Behavior(
        id="q01",
        name="rear",
        examples=[
            {"recording_id": "r1", "track_id": 0, "start_t_ns": 100, "end_t_ns": 200}
        ],
        negatives=[
            {"recording_id": "r1", "track_id": 0, "start_t_ns": 150, "end_t_ns": 400}
        ],
    )
    told = query_runs.told_kinds([behavior])
    assert query_runs.told_kind(told, "q01", "r1", 0, 120, 130) == query_runs.EXAMPLE
    assert query_runs.told_kind(told, "q01", "r1", 0, 160, 170) == query_runs.EXAMPLE
    assert (
        query_runs.told_kind(told, "q01", "r1", 0, 300, 350) == query_runs.NON_EXAMPLE
    )
    assert query_runs.told_kind(told, "q01", "r1", 0, 500, 600) is None
    assert (
        query_runs.told_kind(told, "q01", "r1", 1, 120, 130) is None
    ), "another subject"
    assert (
        query_runs.told_kind(told, "q01", "r2", 0, 120, 130) is None
    ), "another recording"
    assert (
        query_runs.told_kind(told, "q02", "r1", 0, 120, 130) is None
    ), "another behavior"
