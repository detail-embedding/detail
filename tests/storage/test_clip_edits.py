"""``storage/clip_edits.py`` and the ``clip_edits`` field of a temporal state (§22).

Fit-free: the address of a clip, the latest-wins list, and the field's round trip through
``temporal_clustering.yaml``. What the edits do to the instances is ``tests/api/test_clip_edits.py``.
"""

import pytest

from detail.storage import clip_edits as ce
from detail.storage import temporal_runs


def _state(**kw) -> temporal_runs.TemporalClusteringState:
    return temporal_runs.TemporalClusteringState(
        **{"embed_run_id": "emb_01", "track_id": 0, "method": "dtw_finch", **kw}
    )


def test_a_clip_ref_comes_from_a_clip_table_row_a_stored_row_or_a_tuple():
    row = {
        "recording_id": "recording_001",
        "subject": 0,
        "start_t_ns": 5,
        "end_t_ns": 9,
    }
    stored = {"recording_id": "recording_001", "subject": 0, "t_ns": 5, "end_t_ns": 9}
    expected = ce.ClipRef("recording_001", 0, 5, 9)
    assert ce.clip_ref(row) == ce.clip_ref(stored) == expected
    assert ce.clip_ref(("recording_001", 0, 5, 9)) == expected
    assert ce.clip_ref(expected) is expected
    with pytest.raises(TypeError, match="a clip is a ClipRef"):
        ce.clip_ref(5)


def test_the_latest_edit_of_a_clip_wins_and_put_back_removes_it():
    clip = ce.ClipRef("recording_001", 0, 5, 9)
    other = ce.ClipRef("recording_001", 0, 20, 30)
    edits = ce.taken_out([], [clip, other], "L1", None, "2026-11-08T10:00:00Z")
    edits = ce.taken_out(edits, [clip], "L1", "L2", "2026-11-08T10:01:00Z")
    assert len(edits) == 2
    assert [e["to_label"] for e in edits if ce.clip_ref(e) == clip] == ["L2"]
    # A hand-merged list carrying both entries still reads as one per clip, the later winning.
    doubled = [*edits, {**edits[-1], "to_label": None, "at": "later"}]
    assert [e["to_label"] for e in ce.latest(doubled) if ce.clip_ref(e) == clip] == [
        None
    ]
    kept, removed = ce.put_back(edits, [clip, ce.ClipRef("recording_001", 0, 99, 100)])
    assert removed == 1 and [ce.clip_ref(e) for e in kept] == [other]


def test_clip_edits_survive_a_write_and_read_with_schema_version_first(tmp_path):
    edits = ce.taken_out(
        [], [ce.ClipRef("recording_001", 0, 2**40, 2**40 + 17)], "L1", None, "t"
    )
    temporal_runs.write_temporal_state(tmp_path, _state(clip_edits=edits))
    text = temporal_runs.temporal_yaml_path(tmp_path).read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1\n")
    back = temporal_runs.read_temporal_state(tmp_path)
    assert back.clip_edits == edits
    assert isinstance(back.clip_edits[0]["t_ns"], int)
    assert list(back.clip_edits[0]) == list(ce.EDIT_FIELDS)


def test_a_state_written_before_clip_edits_reads_as_none(tmp_path):
    temporal_runs.write_temporal_state(tmp_path, _state())
    record = temporal_runs.read_state_dict(tmp_path)
    record.pop("clip_edits")
    assert temporal_runs.TemporalClusteringState.from_dict(record).clip_edits == []
