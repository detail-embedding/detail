"""Guard: a reader walks a recording's parts in play order, and indexes by the play index.

**The class:** a part's position in the sorted folder listing taken as its play position. Every
stage names a part's store after the part's stem (``part_002.zarr``), and *Move up / Move down*
(`Recording.reorder_video_parts`, allowed before any processing run) reorders
``recording.yaml``'s part list while every stem stays. After a move the sorted listing
(``part_001``, ``part_002``, …) is not the order the parts play in, so a reader that walks
``sorted(folder.glob("part_*.zarr"))`` and indexes the recording's part list, its frame offsets
or its timestamp maps by that position reads another part's. Nothing failed: every number was
plausible.

On origin/main (583d505), a recording of 3 parts (3, 5, 7 frames) moved to play order
part_002, part_001, part_003:

- the location export (``analysis/locations.py``) numbered 8 of 15 track rows with another
  part's offset;
- the COCO export (``export_targets._coco_for_recording``) did the same and, where two parts'
  numbers collided, merged their images: 13 images for 15 frames;
- embedding (``pipeline/embedding.py`` ``run`` → ``_build_units``) paired 2 of 3 parts'
  detections with another part's video, so it embedded the wrong pictures;
- the render preview and embedding's mean-colour pass (``render_preview._collect_candidates``)
  read 8 of 15 candidates from another part's video and timestamp map;
- the YOLO dataset export (``dataset_export._gather_parts``) paired 2 of 3 parts with another
  part's video;
- both trackers walked the parts out of play order, so they carried a track across parts in the
  wrong order and numbered its frames with another part's offset: a track on the part played
  first started at frame 5, not 0.

**The rule:** walk part stores with `recording_store.parts_in_play_order`, which returns
``(play index, path)`` in ``recording.yaml``'s order, and index ``video_files``,
``part_frame_offsets`` and ``load_active_timestamps`` with that play index; or look the part up
by stem (``part_start_frames``). A reader that only collects per-part values keyed by stem, or
counts, may walk in any order.

The fixture moves the parts so play order and name order disagree, which is what makes the bug
visible: on a recording that was never reordered they agree and every reader passes.

**Seen failing** (2026-11-17): ``parts_in_play_order`` replaced by the sorted listing (its old
behaviour, with positions) → 8 failed, 2 passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from detail.api.recording import Recording
from detail.core.dataclasses import ComputeTarget
from detail.core.registries import EMBEDDING_MODELS
from detail.pipeline import dataset_export
from detail.pipeline import embedding
from detail.pipeline import render_preview
from detail.pipeline import tracking
from detail.pipeline.embedding import EmbeddingJob
from detail.pipeline.tracking import TrackingJob
from detail.storage import recording as recording_store
from detail.storage.zarr_io import PartZarrReader

LENGTHS = [3, 5, 7]  # part_001, part_002, part_003
PLAY = ["part_002", "part_001", "part_003"]  # after the move


@pytest.fixture
def moved(experiment_dir, recording_dir):
    """``(recording, whole map)``: three parts moved so play order is part_002, part_001,
    part_003, then a complete ``det_01`` / ``track_01`` written by stem, as the stages do.
    """
    from tests.experiments import add_tracking_parts, add_video_parts

    add_video_parts(recording_dir, LENGTHS)
    recording = Recording(experiment_dir, recording_dir.name)
    recording.reorder_video_parts([1, 0, 2])
    data = recording_store.read_recording(recording_dir)
    assert [recording_store.part_stem(f) for f in data["video"]["files"]] == PLAY
    add_tracking_parts(recording_dir)
    return recording, recording.frame_timestamps()


def _at_their_times(frames, t_ns, whole) -> int:
    """How many ``(frame, t_ns)`` pairs disagree with the recording's map."""
    return sum(
        1 for f, t in zip(frames, t_ns) if f >= len(whole) or int(whole[f]) != int(t)
    )


def test_parts_are_walked_in_play_order_with_their_play_index(moved):
    recording, _whole = moved
    walked = recording_store.parts_in_play_order(
        recording.dir, recording.dir / "detections" / "det_01"
    )
    assert [(index, path.stem) for index, path in walked] == list(enumerate(PLAY))


def test_a_store_for_a_part_the_recording_does_not_list_is_refused(moved):
    recording, _whole = moved
    stray = recording.dir / "detections" / "det_01" / "part_009.zarr"
    stray.mkdir()
    with pytest.raises(
        ValueError, match=r"\['part_009'\] that recording.yaml does not"
    ):
        recording_store.parts_in_play_order(
            recording.dir, recording.dir / "detections" / "det_01"
        )


def test_without_a_manifest_the_sorted_listing_is_the_order(tmp_path):
    for stem in ("part_002", "part_001"):
        (tmp_path / f"{stem}.zarr").mkdir()
    walked = recording_store.parts_in_play_order(tmp_path, tmp_path)
    assert [(i, p.stem) for i, p in walked] == [(0, "part_001"), (1, "part_002")]


def test_the_location_export_numbers_every_part_by_its_place(moved):
    from detail.analysis import locations

    recording, whole = moved
    columns, rows = locations.compute_location_tables(recording, "track_01").tracks
    frame, t_ns = columns.index("frame_idx"), columns.index("t_ns")
    assert len(rows) == sum(LENGTHS)
    assert (
        _at_their_times([r[frame] for r in rows], [r[t_ns] for r in rows], whole) == 0
    )


def test_the_coco_export_numbers_every_part_by_its_place(moved, experiment_dir):
    from detail.api import export_targets

    recording, whole = moved
    document = export_targets._coco_for_recording(
        experiment_dir, recording.dir.name, recording.dir / "detections" / "det_01"
    )
    images = document["images"]
    assert len(images) == sum(LENGTHS), "two parts' frames were numbered alike"
    assert (
        _at_their_times([i["id"] for i in images], [i["t_ns"] for i in images], whole)
        == 0
    )


class _Planned(Exception):
    """Raised from inside `embedding.run` once its units are built: what is checked."""


def test_embedding_cuts_each_part_from_its_own_video(moved, monkeypatch):
    recording, _whole = moved
    data = recording_store.read_recording(recording.dir)
    video_of = {
        recording_store.part_stem(f): recording_store.resolve_part_video_path(
            recording.dir, f
        )
        for f in data["video"]["files"]
    }
    real = embedding._build_units

    def build_then_stop(**kwargs):
        raise _Planned(real(**kwargs))

    monkeypatch.setitem(EMBEDDING_MODELS, "probe", object)
    monkeypatch.setattr(embedding, "_build_units", build_then_stop)
    job = EmbeddingJob(
        job_id="emb_01",
        recording_id=recording.dir.name,
        embedding_run_id="emb_01",
        tracking_run_id="track_01",
        track_ids=[0],
        model_id="probe",
        renderer_spec={"steps": ["silhouette"]},
        batch_size=1,
        num_workers=0,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(_Planned) as planned:
        embedding.run(
            job, recording.dir / "embeddings" / "emb_01", lambda e: None, None
        )
    units = planned.value.args[0]
    assert [u["part_id"] for u in units] == PLAY
    assert {u["part_id"]: u["video_path"] for u in units} == video_of
    assert [u["part_index"] for u in units] == [0, 1, 2]


def test_preview_candidates_read_their_own_parts_map(moved):
    recording, _whole = moved
    data = recording_store.read_recording(recording.dir)
    candidates = render_preview._collect_candidates(
        recording.dir,
        data,
        recording.dir / "detections" / "det_01",
        recording.dir / "tracking" / "track_01",
        {0},
    )
    assert len(candidates) == sum(LENGTHS)
    maps = [
        recording_store.load_active_timestamps(recording.dir, data, i).values
        for i in range(len(PLAY))
    ]
    wrong = [
        c
        for c in candidates
        if c.frame_idx >= len(maps[c.part_index])
        or int(maps[c.part_index][c.frame_idx]) != c.t_ns
    ]
    assert not wrong, f"{len(wrong)} candidates read another part's video and map"


def test_the_dataset_export_pairs_each_part_with_its_own_video(moved, experiment_dir):
    recording, _whole = moved
    data = recording_store.read_recording(recording.dir)
    video_of = {
        recording_store.part_stem(f): recording_store.resolve_part_video_path(
            recording.dir, f
        )
        for f in data["video"]["files"]
    }
    record = {
        "per_recording": {
            recording.dir.name: {"status": "complete", "run_id": "track_01"}
        }
    }
    parts = dataset_export._gather_parts(experiment_dir, record, "tracking")
    assert {p["stem"]: p["video"] for p in parts} == video_of


@pytest.mark.parametrize(
    "tracker_id, hyperparams",
    [
        ("anchor_tracker", {"conf_threshold": 0.1, "require_contiguous": False}),
        ("iou_single_tracker", {}),
    ],
)
def test_a_tracker_walks_the_parts_as_they_play(moved, tracker_id, hyperparams):
    """Detections only on the part played first (part_002): its track starts at frame 0."""
    from tests.experiments import add_tracking_parts

    recording, _whole = moved
    add_tracking_parts(
        recording.dir,
        detection_run_id="det_first",
        tracking_run_id=None,
        detected={PLAY[0]},
    )
    run_dir = recording.dir / "tracking" / f"track_{tracker_id}"
    run_dir.mkdir(parents=True)
    job = TrackingJob(
        job_id="t",
        recording_id=recording.dir.name,
        track_id="t",
        detection_run_id="det_first",
        tracker_id=tracker_id,
        hyperparams=hyperparams,
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
    )
    tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    tracks = PartZarrReader(run_dir / "tracks.zarr").read()
    assert int(np.min(tracks["first_frame"])) == 0
