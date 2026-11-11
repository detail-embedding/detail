"""Selecting which behavior stream the clustering ground truth comes from.

These used to fall back to the recording's *first* behavior-role stream. On a recording
carrying several — an early hand-scored subset beside a full ethogram — that read whichever
was **added first**, and unlike query-eval (which checked a covered vocabulary and skipped)
they returned the wrong stream's intervals happily. A clustering run could score one
recording against one vocabulary and the rest against another and still produce a number.

That is no longer a hazard to document, because it fired: adding a 2-label stream beside a
7-label ethogram moved `clusterer diagnose` onto a different vocabulary than the eval it was
meant to explain, with nothing raised anywhere. ``resolve_behavior_stream`` now refuses to
choose. One stream still resolves silently — ambiguity is the trigger, not the absence of a
name — and no stream is still an empty result, which is a normal state for an unannotated
recording.
"""

import numpy as np
import pytest

from detail.analysis import behavior_labels as bl
from detail.api.recording import Recording
from detail.storage import recording as recording_store

PERIOD = 33_333_333  # ns per frame at 30 fps


@pytest.fixture
def two_stream_recording(experiment_dir, recording_dir, tmp_path):
    """`early` (added first, climb/groom) and `full` (a small ethogram)."""
    t_ns = np.arange(60, dtype=np.int64) * PERIOD
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    rec = Recording(experiment_dir, "recording_001")
    early = tmp_path / "early.csv"
    early.write_text(
        "start_frame,end_frame,label\n2,5,climb\n10,13,face groom\n", encoding="utf-8"
    )
    rec.add_stream("early", raw_path=early, importer="behavior_csv")
    full = tmp_path / "full.csv"
    full.write_text(
        "start_frame,end_frame,label\n20,25,rearing\n30,35,grooming\n40,45,rearing\n",
        encoding="utf-8",
    )
    rec.add_stream("full", raw_path=full, importer="behavior_csv")
    return Recording(experiment_dir, "recording_001")


def _labels(intervals):
    return sorted({iv[2] for iv in intervals})


def test_several_streams_and_no_selector_raises_rather_than_guessing(
    two_stream_recording,
):
    """What this file's docstring described as the hazard is now the error.

    The old default read whichever stream was added first. It fired for real: adding a
    2-label stream beside a 7-label ethogram silently moved `clusterer diagnose` onto a
    different vocabulary than the eval it was meant to explain, and nothing errored. The
    message has to name the streams, because the fix is for the caller to pick one.
    """
    with pytest.raises(
        bl.AmbiguousBehaviorStreamError, match="early.*full|full.*early"
    ):
        bl.recording_behavior_intervals(two_stream_recording)


def test_one_stream_still_resolves_silently(experiment_dir, recording_dir, tmp_path):
    """The common case is unchanged — ambiguity is the trigger, not the absence of a name."""
    t_ns = np.arange(60, dtype=np.int64) * PERIOD
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    rec = Recording(experiment_dir, "recording_001")
    only = tmp_path / "only.csv"
    only.write_text("start_frame,end_frame,label\n2,5,climb\n", encoding="utf-8")
    rec.add_stream("only", raw_path=only, importer="behavior_csv")
    assert _labels(bl.recording_behavior_intervals(rec)) == ["climb"]


def test_a_recording_with_no_behavior_stream_is_empty_not_an_error(
    experiment_dir, recording_dir
):
    rec = Recording(experiment_dir, "recording_001")
    assert bl.recording_behavior_intervals(rec) == []


def test_named_stream_reads_that_stream(two_stream_recording):
    intervals = bl.recording_behavior_intervals(
        two_stream_recording, stream_name="full"
    )
    assert _labels(intervals) == ["grooming", "rearing"]
    assert len(intervals) == 3


def test_each_stream_is_reachable_by_name(two_stream_recording):
    """Naming is now the only way in, so both must be reachable and distinct."""
    early = bl.recording_behavior_intervals(two_stream_recording, stream_name="early")
    full = bl.recording_behavior_intervals(two_stream_recording, stream_name="full")
    assert _labels(early) == ["climb", "face groom"]
    assert _labels(full) == ["grooming", "rearing"]


def test_behavior_filter_composes_with_the_stream(two_stream_recording):
    intervals = bl.recording_behavior_intervals(
        two_stream_recording, behaviors=["rearing"], stream_name="full"
    )
    assert _labels(intervals) == ["rearing"]
    assert len(intervals) == 2


def test_frame_labels_follow_the_selected_stream(two_stream_recording):
    """The whole point: per-frame ground truth must come from the chosen stream."""
    early = bl.recording_frame_labels(two_stream_recording, stream_name="early")
    chosen = bl.recording_frame_labels(two_stream_recording, stream_name="full")
    assert sorted(early.id_to_label.values()) == ["climb", "face groom"]
    assert sorted(chosen.id_to_label.values()) == ["grooming", "rearing"]
    # and they label disjoint frames — the two streams cover different spans
    assert not (
        set(np.nonzero(early.labels != -1)[0]) & set(np.nonzero(chosen.labels != -1)[0])
    )


def test_types_come_from_the_same_stream_as_the_intervals(two_stream_recording):
    """Reading intervals from one stream and point/state types from another would score a
    point behavior as a state. Both must follow ``stream_name``."""
    labels = bl.recording_frame_labels(two_stream_recording, stream_name="full")
    covered = set(labels.id_to_label.values())
    assert covered == set(two_stream_recording.behavior_types("full"))
    assert covered.isdisjoint(two_stream_recording.behavior_types("early"))


def test_unknown_stream_raises(two_stream_recording):
    with pytest.raises((FileNotFoundError, ValueError, KeyError, TypeError)):
        bl.recording_behavior_intervals(two_stream_recording, stream_name="nope")
