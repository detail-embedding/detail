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
