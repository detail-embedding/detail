"""Guard: a frame number and a time written together agree through the recording's timestamp map,
on every video part (issue #23).

**The class:** a part-local frame number read as a recording-wide one. Every per-part store
(detections, assignments, embeddings, kinematics, keypoints) counts ``frame_idx`` from 0 inside its
own video part (§5 ``FrameResult.frame_idx``), while ``t_ns`` is on the recording's clock. The two
embedding loaders, ``run_preview.load_track_trajectory`` and ``load_track_timeline``, concatenated
the parts with no offset, so a two-part track of 6 frames per part read ``[0..5, 0..5]``. Everything
that took a frame from them was wrong past part 1, and nothing failed, because the times beside
those frames were right and every GUI surface places an instance by time:

- query and clustering instances (``instances_from``, ``UnitFrames.frame_at``) wrote part 2's bouts
  with part 1's frame numbers, so ``instances.csv``, ``clips.csv``, the ethogram / stream CSV and
  every cut clip landed in part 1, and a re-imported stream (the importer reads recording frames,
  ``clocks.frame_to_t_ns``) came back in part 1;
- ``clip_query.frame_span_to_t_ns`` binary-searched the unsorted array: a part-1 span resolved into
  part 2 and a recording-wide part-2 span was refused;
- ``run_frame_ranges`` (the clip picker's coverage band) reported one part's run for a track that
  spans two;
- Measurements (``frame_slices.from_tracking``) joined recording-wide frames to the loaders'
  part-local ones, so every part-2 embedding was NaN; and, separately, it added each earlier
  part's elapsed time to a map that already holds recording-wide time, so part 2 ran late by the
  length of part 1;
- the NWB export started each part's video at 0 s while its poses use recording time.

It is the next layer of "a row position written as a video frame number" (#505,
``tests/pipeline/test_instance_frames.py``): that fix moved frames onto ``UnitFrames.frame_at``,
but the lookup read the loaders' part-local frames, so it stopped at the part boundary.

**The rule:** a reader that concatenates parts, or hands a stored frame to something that counts
across the recording, adds the part's start frame from ``recording_store.part_start_frames``
(keyed by part stem, the one mechanism) first, or carries the part beside the frame. A time comes
from the timestamp map as stored; it is already recording-wide.

**The fixture** makes part 2 store the same local frame numbers as part 1, which is what makes a
missing offset visible: a one-part fixture passes with or without the fix.

**Seen failing** (2026-11-17): the start frame dropped from ``_embedded_parts`` → 7 failed;
``elapsed_ns`` re-added to ``from_tracking`` → 1 failed; ``starting_time=0.0`` restored in
``write_nwb`` → 1 failed (``tests/api/test_keypoint_export.py``, needs pynwb).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from detail.analysis import clip_query
from detail.api import clip_render, frame_slices
from detail.api import run_preview
from detail.pipeline import query as query_pipeline
from detail.pipeline import temporal_clustering
from detail.pipeline.behavior_artifact import UnitFrames
from detail.storage import clocks
from detail.storage import recording as recording_store

PART = 6  # frames per part; both parts store local frames 0..5


@pytest.fixture
def two_parts(recording_dir, make_embedding_run):
    """``(recording_dir, whole map)``: two 6-frame video parts, a tracking run that tracked
    subject 0 on every frame, and an embedding run with a kinematics sidecar — every per-part
    store holding part-local frames 0..5 in both parts, as the stages write them."""
    from tests.experiments import add_tracking_parts, add_video_parts

    whole = add_video_parts(recording_dir, [PART, PART])
    add_tracking_parts(recording_dir)
    local = np.tile(np.arange(PART), 2)
    emb = np.arange(2 * PART, dtype=np.float32)[:, None] * np.ones(4, np.float32) + 1
    make_embedding_run(
        recording_dir,
        emb,
        local,
        whole,
        tracking_run_id="track_01",
        cx=np.full(2 * PART, 0.5),
        parts=[PART, PART],
    )
    return recording_dir, whole


def _agree(frames, t_ns, whole) -> None:
    """The invariant: each frame's time on the recording's map is the time written beside it."""
    frames = np.asarray(frames, dtype=np.int64)
    assert np.array_equal(
        whole[frames], np.asarray(t_ns, dtype=np.int64)
    ), f"frames {frames.tolist()} are not at times {np.asarray(t_ns).tolist()}"


def test_both_loaders_return_recording_wide_frames_in_order(two_parts):
    recording_dir, whole = two_parts
    frames, t_ns = run_preview.load_track_timeline(recording_dir, "emb_01", 0)
    _emb, traj_frames, traj_t_ns = run_preview.load_track_trajectory(
        recording_dir, "emb_01", 0
    )
    assert frames.tolist() == list(range(2 * PART))
    assert np.all(np.diff(frames) > 0)
    _agree(frames, t_ns, whole)
    # `unit_timeline` stands the timeline in for the trajectory: row k is the same row in both.
    assert np.array_equal(frames, traj_frames)
    assert np.array_equal(t_ns, traj_t_ns)


def test_a_frame_span_resolves_the_recording_frames_it_names(two_parts):
    recording_dir, whole = two_parts
    frames, t_ns = run_preview.load_track_timeline(recording_dir, "emb_01", 0)
    # a part-1 span stays in part 1 ...
    assert clip_query.frame_span_to_t_ns(frames, t_ns, 2, 4) == (whole[2], whole[4])
    # ... and a part-2 span written in recording frames resolves rather than being refused
    assert clip_query.frame_span_to_t_ns(frames, t_ns, 7, 9) == (whole[7], whole[9])


def test_a_frame_span_over_part_local_frames_is_refused():
    local = np.tile(np.arange(PART), 2)
    with pytest.raises(ValueError, match="not strictly increasing"):
        clip_query.frame_span_to_t_ns(local, np.arange(2 * PART), 2, 4)


def test_the_coverage_band_is_one_run_over_both_parts(two_parts):
    recording_dir, _whole = two_parts
    frames, t_ns = run_preview.load_track_timeline(recording_dir, "emb_01", 0)
    assert clip_query.run_frame_ranges(frames, t_ns) == [(0, 2 * PART - 1)]


def test_query_instances_carry_the_recording_frames_at_their_times(
    two_parts, experiment_dir
):
    recording_dir, whole = two_parts
    frames, t_ns = query_pipeline.unit_timeline(
        experiment_dir, "emb_01", recording_dir.name, 0
    )
    bouts = [types.SimpleNamespace(start_row=7, end_row=9, confidence=1.0)]
    (instance,) = query_pipeline.instances_from(
        bouts,
        recording_id=recording_dir.name,
        subject=0,
        behavior_id="b",
        t_ns=t_ns,
        frame_idx=frames,
    )
    assert (instance.start_frame, instance.end_frame) == (7, 9)
    _agree(
        [instance.start_frame, instance.end_frame],
        [instance.t_ns, instance.end_t_ns],
        whole,
    )
    # what the behavior-CSV importer does with an exported stream row: it lands where it was found
    assert clocks.frame_to_t_ns(instance.start_frame, whole) == instance.t_ns
    # and the clip is cut from part 2, at part-local frames 1..3
    part_index, job = clip_render.plan_job(
        [PART, PART], instance.start_frame, instance.end_frame, 0.1, recording_dir / "c"
    )
    assert (part_index, job["lo"], job["hi"]) == (1, 1, 3)


def test_clustering_instances_carry_the_recording_frames_at_their_times(
    two_parts, experiment_dir
):
    recording_dir, whole = two_parts
    # what `temporal_clustering.project_artifact` builds per recording, and `_row_for` reads
    frames, t_ns = temporal_clustering.load_timeline(
        experiment_dir, "emb_01", recording_dir.name, 0
    )
    unit = UnitFrames(t_ns, frames)
    found = [unit.frame_at(int(whole[f])) for f in range(2 * PART)]
    assert found == list(range(2 * PART))


def test_measurements_read_every_part_on_the_recordings_clock(two_parts):
    recording_dir, whole = two_parts
    tracked = frame_slices.from_tracking(
        recording_dir, tracking_run_id="track_01", track_id=0, embed_run_id="emb_01"
    )
    assert tracked.frame_idx.tolist() == list(range(2 * PART))
    _agree(tracked.frame_idx, tracked.t_ns, whole)
    assert not np.isnan(tracked.embeddings).any(), "an embedded frame read back as NaN"
    sidecar = frame_slices.from_kinematics(
        recording_dir, embed_run_id="emb_01", track_id=0
    )
    assert np.array_equal(sidecar.frame_idx, tracked.frame_idx)
    assert np.array_equal(sidecar.t_ns, tracked.t_ns)


def test_the_part_offsets_are_keyed_by_stem(two_parts):
    recording_dir, _whole = two_parts
    assert recording_store.part_start_frames(recording_dir) == {
        "part_001": 0,
        "part_002": PART,
    }
    assert recording_store.part_frame_offsets(recording_dir) == [0, PART]


def test_an_empty_first_part_does_not_shift_the_second(
    recording_dir, make_embedding_run
):
    """The loaders skip an empty part, so a positional offset would put part 2 at frame 0."""
    from tests.experiments import add_video_parts

    whole = add_video_parts(recording_dir, [PART, PART])
    make_embedding_run(
        recording_dir,
        np.ones((PART, 4), np.float32),
        np.arange(PART),
        whole[PART:],
        parts=[0, PART],
    )
    frames, t_ns = run_preview.load_track_timeline(recording_dir, "emb_01", 0)
    assert frames.tolist() == list(range(PART, 2 * PART))
    _agree(frames, t_ns, whole)


def test_a_part_the_recording_does_not_list_is_refused(
    recording_dir, make_embedding_run
):
    make_embedding_run(
        recording_dir,
        np.ones((2 * PART, 4), np.float32),
        np.tile(np.arange(PART), 2),
        np.arange(2 * PART),
        parts=[PART, PART],
    )
    with pytest.raises(
        ValueError, match="'part_002', which recording.yaml does not list"
    ):
        run_preview.load_track_timeline(recording_dir, "emb_01", 0)


def test_several_parts_without_a_video_manifest_are_refused(
    tmp_path, make_embedding_run
):
    """No ``recording.yaml``: one part is the only frame space there is; two cannot be placed."""
    make_embedding_run(
        tmp_path, np.ones((PART, 4), np.float32), np.arange(PART), np.arange(PART)
    )
    assert run_preview.load_track_timeline(tmp_path, "emb_01", 0)[0].tolist() == list(
        range(PART)
    )
    make_embedding_run(
        tmp_path,
        np.ones((2 * PART, 4), np.float32),
        np.tile(np.arange(PART), 2),
        np.arange(2 * PART),
        embed_run_id="emb_02",
        parts=[PART, PART],
    )
    with pytest.raises(ValueError, match="lists no video parts"):
        run_preview.load_track_timeline(tmp_path, "emb_02", 0)
