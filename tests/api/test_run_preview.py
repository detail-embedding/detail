"""DetectionPreview: per-frame boxes + counts from a detection run's zarr.

**The row-block cache's key (a recorded class): a cache keyed on less than what decides its
contents, or on an object's identity.** ``DetectionPreview._read_rows`` serves short spans from a
cached block of ``ROW_BLOCK`` rows. The block was keyed ``(id(part), block)`` and always read the
``mask_rle`` column, so a caller that asked for no masks still decompressed every row's RLE, and a
part dict freed and replaced could hand its ``id()`` to a different part and be served its rows.
Neither shows in a result that looks wrong: the masks are right when asked for, and ``_parts`` is
only built in ``__init__`` today, so the second is latent. ``main``'s #866 found both in its own
rebuild of the clip window; remote-detail, which kept its own (#954), had both. The rule: key a
cache on everything its stored value depends on (``with_masks``) and on a name that stays the same
for the cache's life (the part's stem), never on ``id()``. Pinned by the three tests under *the
row-block cache's key* below.
"""

import cv2
import numpy as np
import pytest
import yaml

from detail.api.run_preview import (
    _square_crop,
    _union_detection,
    locate_frame,
    pca_project_2d,
)
from detail.core.dataclasses import Detection
from detail.api.recording import Recording
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import video as video_store
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartWriter,
    EmbeddingPartWriter,
)


def _recording_with_detection(tmp_path, n_frames=5, write_video=None):
    """A recording with a tiny video + a hand-written detection run (no model).

    ``write_video`` overrides how the video is produced (e.g. the ``write_vfr_video``
    fixture); the timestamp map is then generated from the file's real PTS rather
    than assumed constant-rate.
    """
    proj_dir = experiment_store.create_experiment(tmp_path, "p")
    recording_store.create_recording(proj_dir, "s1")
    recording = Recording(proj_dir, "s1")
    sdir = recording.dir
    vid = sdir / "video" / "part_001.mp4"
    vid.parent.mkdir(parents=True, exist_ok=True)
    if write_video is not None:
        write_video(vid, n_frames=n_frames)
        t_ns = video_store.generate_timestamps(vid, 30.0)
        m = len(t_ns)
    else:
        writer = cv2.VideoWriter(
            str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
        )
        for i in range(n_frames):
            writer.write(np.full((48, 64, 3), (i * 40) % 256, np.uint8))
        writer.release()
        cap = cv2.VideoCapture(str(vid))
        m = 0
        while cap.read()[0]:
            m += 1
        cap.release()
        t_ns = np.arange(m, dtype=np.int64) * int(1e9 / 30)
    recording_store.write_timestamps_csv(
        sdir / "video" / "timestamps" / "generated" / "part_001.csv", t_ns
    )
    data = recording_store.read_recording(sdir)
    data["video"] = {
        "storage": "default",
        "fps_container": 30.0,
        "fps_actual": 30.0,
        "fps_source": "container",
        "resolution": [64, 48],
        "total_duration_s": m / 30,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    }
    from pycocotools import mask as mask_utils

    mask0 = np.zeros((48, 64), np.uint8)
    mask0[10:20, 10:20] = 1
    rle0 = mask_utils.encode(np.asfortranarray(mask0))["counts"].decode("ascii")
    ddir = sdir / "detections" / "det_01"
    ddir.mkdir(parents=True)
    with DetectionPartWriter(ddir / "part_001.zarr") as writer:
        # frame 0: two detections, frame 2: one, frames 1/3/4: none (no rows)
        writer.add_row(0, 0, 0, "mouse", 0.9, (0.1, 0.1, 0.2, 0.2), rle0)
        writer.add_row(0, 0, 1, "mouse", 0.8, (0.3, 0.3, 0.4, 0.4), "")
        writer.add_row(2, int(2e9 / 30), 0, "mouse", 0.7, (0.5, 0.5, 0.6, 0.6), "")
    data.setdefault("detection_runs", []).append(
        {"id": "det_01", "status": "complete", "downsampling_run_id": None}
    )
    recording_store.write_recording(sdir, data)
    return recording


def test_detection_preview_counts(tmp_path):
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    assert preview.part_frame_counts == [5]
    assert preview.counts(0).tolist() == [2, 0, 1, 0, 0]  # zero-detection frames stay 0


def test_detection_preview_frame_boxes(tmp_path):
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    actual, rgb, detections = preview.frame(0, 0)
    assert actual == 0
    assert rgb.shape == (48, 64, 3)
    assert len(detections) == 2
    assert detections[0].class_name == "mouse"
    assert detections[0].bbox == pytest.approx((0.1, 0.1, 0.2, 0.2), abs=1e-6)
    assert detections[0].mask is None  # boxes only in v1
    assert preview.frame(0, 1)[2] == []  # frame with no detections


def test_detection_preview_masks(tmp_path):
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    detections = preview.frame(0, 0, with_masks=True)[2]
    mask = detections[0].mask
    assert mask is not None and mask.shape == (48, 64) and mask.dtype == bool
    assert mask[15, 15] and not mask[0, 0]  # the 10:20 box round-trips through RLE
    assert preview.frame(0, 0)[2][0].mask is None  # default: boxes only


def test_detection_preview_unknown_run(tmp_path):
    recording = _recording_with_detection(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        recording.detection_preview("det_99")


# --- the row-block cache's key: (part stem, block, with_masks) ---------------------------- #
def _count_column_reads(monkeypatch) -> list:
    """Every zarr column read, by name — the cost ``_read_rows`` exists to bound."""
    from detail.storage.zarr_io import PartZarrReader

    reads: list = []
    real = PartZarrReader.column
    monkeypatch.setattr(
        PartZarrReader,
        "column",
        lambda self, name: reads.append(name) or real(self, name),
    )
    return reads


def test_a_plain_read_decompresses_no_mask_column(tmp_path, monkeypatch):
    """A caller that asked for no masks never pays for ``mask_rle``, not even to fill a block.
    A block keyed without ``with_masks`` read every row's RLE for every plain read."""
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    reads = _count_column_reads(monkeypatch)
    for lo in (0, 1, 2):
        preview._read_rows(preview._parts[0], lo, lo + 1, with_masks=False)
    assert reads.count("mask_rle") == 0
    assert reads.count("bbox") == 1, "one block serves every row of the span"


def test_a_masked_read_after_a_plain_one_still_carries_masks(tmp_path):
    """The other direction of the key: a block filled for a masks-off caller is never served
    to a masks-on one, which would hand back no masks for as long as the block stayed cached.
    """
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    plain = preview._read_rows(preview._parts[0], 0, 2, with_masks=False)
    assert plain["mask_rle"] is None
    masked = preview._read_rows(preview._parts[0], 0, 2, with_masks=True)
    assert masked["mask_rle"] is not None and len(masked["mask_rle"]) == 2


def test_row_blocks_survive_a_rebuilt_part_list(tmp_path, monkeypatch):
    """The block is keyed on the part's stem, not ``id(part)``: a preview whose part dicts are
    rebuilt (or whose freed dict's id is reused for another part) still finds its own blocks.
    """
    preview = _recording_with_detection(tmp_path).detection_preview("det_01")
    reads = _count_column_reads(monkeypatch)
    preview._read_rows(preview._parts[0], 0, 2, with_masks=True)
    preview._parts = [dict(part) for part in preview._parts]
    preview._read_rows(preview._parts[0], 0, 2, with_masks=True)
    assert reads.count("mask_rle") == 1


def test_detection_preview_overlay_matches_the_picture_on_a_vfr_video(
    tmp_path, monkeypatch, write_vfr_video
):
    """The reported bug, through the real preview surface.

    On a variable-frame-rate file no single fps maps frame number to time, so the
    old cv2 ``CAP_PROP_POS_FRAMES`` fallback returned the wrong picture *and*
    reported the requested index back — ``_detections_at`` then drew one frame's
    boxes on another frame's image, drifting with frame number (frame 25 returned
    the picture of 33; frame 40 returned 52). The preview now seeks through the
    recording's timestamp map, so picture and overlay always come from one frame.
    """
    import av

    recording = _recording_with_detection(
        tmp_path, n_frames=60, write_video=write_vfr_video
    )
    vid = recording.dir / "video" / "part_001.mp4"
    with av.open(str(vid)) as container:
        seq = [
            f.to_ndarray(format="bgr24")
            for f in container.decode(container.streams.video[0])
        ]
    # Disable the packet-PTS index so the map alone has to carry the seek.
    monkeypatch.setattr(video_store, "_cached_frame_pts", lambda *a, **k: None)

    preview = recording.detection_preview("det_01")
    for k in (0, 10, 25, 40, 55):
        actual, rgb, detections = preview.frame(0, k)
        assert actual == k
        assert np.array_equal(rgb, seq[k][:, :, ::-1])  # preview surface is RGB
        assert len(detections) == {0: 2, 2: 1}.get(k, 0)


def _write_track_yaml(tracking_dir, *, detection_run_id, class_name=None):
    """A real tracking run always writes one of these at completion
    (``trackers/_common.py::write_track_yaml``) — ``TrackingPreview`` reads it for
    the class filter the tracker actually applied, which is not always the same as
    the run's own record (a track_merge run's record carries a joined display
    label; its track.yaml correctly keeps ``None``). A fixture that skips this file
    is testing a run shape that cannot occur for a ``status: complete`` run."""
    tracking_dir.mkdir(parents=True, exist_ok=True)
    (tracking_dir / "track.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": tracking_dir.name,
                "detection_run_id": detection_run_id,
                "class_name": class_name,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _add_tracking(recording):
    """A tracking run over det_01: object_id parallel to its 3 detection rows."""
    sdir = recording.dir
    adir = sdir / "tracking" / "track_01" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        writer.add_object_id(0)  # frame 0, row 0 -> track 0
        writer.add_object_id(-1)  # frame 0, row 1 -> unassigned (dropped)
        writer.add_object_id(0)  # frame 2, row 0 -> track 0
    _write_track_yaml(sdir / "tracking" / "track_01", detection_run_id="det_01")
    data = recording_store.read_recording(sdir)
    data.setdefault("tracking_runs", []).append(
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    )
    recording_store.write_recording(sdir, data)
    return recording


def test_tracking_preview_track_ids(tmp_path):
    preview = _add_tracking(_recording_with_detection(tmp_path)).tracking_preview(
        "track_01"
    )
    # frame 0: the tracked row (track 0) is kept; the unassigned (-1) row is dropped
    _actual, _rgb, detections = preview.frame(0, 0)
    assert len(detections) == 1
    assert detections[0].object_id == 0
    assert preview.frame(0, 2)[2][0].object_id == 0  # frame 2 -> track 0


def test_tracking_preview_presence(tmp_path):
    preview = _add_tracking(_recording_with_detection(tmp_path)).tracking_preview(
        "track_01"
    )
    track_ids, presence = preview.track_presence()
    assert track_ids == [0]  # the -1 (unassigned) is not a track
    # track 0 is present where it has detections: frames 0 and 2
    assert presence[0].tolist() == [True, False, True, False, False]


def test_tracking_preview_survives_a_detection_run_torn_by_a_crash(tmp_path):
    """The reported bug: opening a tracking run's preview crashed with
    ``IndexError: boolean index did not match indexed array along axis 0`` when the
    underlying detection run had been interrupted mid-flush (a crash on a
    remote/multi-GPU worker), leaving one part's columns at unequal length until a
    *writer* reopens the store (``storage/zarr_io.py``). The tracking *run* itself
    (``tests/pipeline/test_tracking.py::test_class_filter_survives_a_detection_run_torn_by_a_crash``)
    is one exposed site; the preview — built from a *separate* detection reader and
    assignment reader that must agree on row count — is another, and crashes on
    ``track_presence()``/``frame()`` even when the tracking run itself completed
    cleanly before the tear.
    """
    import zarr

    recording = _add_tracking(_recording_with_detection(tmp_path))
    root = zarr.open_group(
        str(recording.dir / "detections" / "det_01" / "part_001.zarr"), mode="a"
    )
    # Torn: frame_idx (what _build_parts reads to index against the assignment's
    # object_id) is left ahead of every other column, including bbox — the
    # detection reader's own n_rows still correctly reports the shorter length,
    # but a caller reading frame_idx raw (pre-fix) would not see that.
    root["frame_idx"].resize((root["frame_idx"].shape[0] + 2,))

    preview = recording.tracking_preview("track_01")
    track_ids, presence = preview.track_presence()
    assert track_ids == [0]
    assert presence[0].tolist() == [True, False, True, False, False]
    assert preview.frame(0, 0)[2][0].object_id == 0


def test_tracking_preview_with_a_class_filter_is_not_a_truncated_prefix(tmp_path):
    """The actual reported bug (distinct from the torn-store one above): a
    class-filtered tracking run's assignment has one row per *matching* detection,
    not one row per detection overall (anchor_tracker et al. build ``object_ids``
    from ``filter_by_class``'d data, tests/pipeline/test_tracking.py). Reading "the
    first N rows" of the *unfiltered* detection reader — which is what pairing by
    raw position does — silently grabs the wrong rows: on a real 21,720-frame
    recording this read as tracking dropping the entire trailing ~41% of an
    otherwise fully-detected video, for either class filtered alone.

    Deliberately shaped so a white-mouse-only presence bar of ``[0, 1]`` (the
    prefix of the *raw*, both-classes row order) is a different, wrong answer
    from the real one, ``[0, 2, 3]``.
    """
    recording = _recording_with_detection(tmp_path)  # video + det_01 (unused here)
    sdir = recording.dir
    ddir = sdir / "detections" / "det_02"
    ddir.mkdir(parents=True)
    with DetectionPartWriter(ddir / "part_001.zarr") as writer:
        # frame 0: black + white; frame 1: black only; frame 2: white only;
        # frame 3: black + white — white mouse is at frames 0, 2, 3, never 1.
        writer.add_row(0, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        writer.add_row(0, 0, 1, "white mouse", 0.9, (0.3, 0.3, 0.4, 0.4), "")
        writer.add_row(1, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        writer.add_row(2, 0, 0, "white mouse", 0.9, (0.3, 0.3, 0.4, 0.4), "")
        writer.add_row(3, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        writer.add_row(3, 0, 1, "white mouse", 0.9, (0.3, 0.3, 0.4, 0.4), "")
    data = recording_store.read_recording(sdir)
    data["detection_runs"].append(
        {"id": "det_02", "status": "complete", "downsampling_run_id": None}
    )
    adir = sdir / "tracking" / "track_02" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        # One row per white-mouse detection only (3 rows), all assigned to track 0.
        writer.add_object_id(0)
        writer.add_object_id(0)
        writer.add_object_id(0)
    _write_track_yaml(
        sdir / "tracking" / "track_02",
        detection_run_id="det_02",
        class_name="white mouse",
    )
    data.setdefault("tracking_runs", []).append(
        {
            "id": "track_02",
            "status": "complete",
            "detection_run_id": "det_02",
            "class_name": "white mouse",
        }
    )
    recording_store.write_recording(sdir, data)

    preview = recording.tracking_preview("track_02")
    track_ids, presence = preview.track_presence()
    assert track_ids == [0]
    assert presence[0].tolist() == [True, False, True, True, False]


def test_track_merge_preview_shows_both_subjects_in_the_same_frame(tmp_path):
    """The actual point of track_merge (§13): one preview call over the merged run
    returns both animals' detections for the same frame, each carrying its own
    object_id and its own class_name — using TrackingPreview exactly as it already
    exists (object_id-generic, fixed earlier this session), no changes needed."""
    from detail.core.dataclasses import TrackingJob
    from detail.pipeline import track_merge, tracking

    recording = _recording_with_detection(tmp_path)  # video + det_01 (unused here)
    sdir = recording.dir
    ddir = sdir / "detections" / "det_02"
    ddir.mkdir(parents=True)
    (ddir / "job.json").write_text(
        '{"schema_version": 1, "mask_resolution": [48, 64]}', encoding="utf-8"
    )
    with DetectionPartWriter(ddir / "part_001.zarr") as writer:
        for f in range(4):
            writer.add_row(f, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
            writer.add_row(f, 0, 1, "white mouse", 0.8, (0.6, 0.6, 0.7, 0.7), "")
    data = recording_store.read_recording(sdir)
    data["detection_runs"].append(
        {"id": "det_02", "status": "complete", "downsampling_run_id": None}
    )
    recording_store.write_recording(sdir, data)

    for track_id, class_name in (
        ("track_a", "black mouse"),
        ("track_b", "white mouse"),
    ):
        job = TrackingJob(
            job_id=track_id,
            recording_id=recording.id,
            track_id=track_id,
            detection_run_id="det_02",
            tracker_id="anchor_tracker",
            hyperparams={"conf_threshold": 0.05},
            arena_ids=[],
            created_at="2026-01-01T00:00:00Z",
            class_name=class_name,
        )
        run_dir = sdir / "tracking" / track_id
        run_dir.mkdir(parents=True)
        tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
        data = recording_store.read_recording(sdir)
        data.setdefault("tracking_runs", []).append(
            {
                "id": track_id,
                "status": "complete",
                "detection_run_id": "det_02",
                "tracker_id": "anchor_tracker",
                "arena_ids": [],
                "class_name": class_name,
                "hyperparams": {"conf_threshold": 0.05},
                "every_n_frames": 1,
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
        recording_store.write_recording(sdir, data)

    merge_job = track_merge.TrackMergeJob(
        job_id="track_merged",
        recording_id=recording.id,
        track_id="track_merged",
        detection_run_id="det_02",
        source_run_ids=["track_a", "track_b"],
        created_at="2026-01-01T00:00:00Z",
        every_n_frames=1,
    )
    merge_run_dir = sdir / "tracking" / "track_merged"
    merge_run_dir.mkdir(parents=True)
    track_merge.run(
        merge_job, merge_run_dir, progress_cb=lambda e: None, stop_event=None
    )
    data = recording_store.read_recording(sdir)
    data["tracking_runs"].append(
        {
            "id": "track_merged",
            "status": "complete",
            "detection_run_id": "det_02",
            "tracker_id": "track_merge",
            "arena_ids": [],
            # The real identity TrackMergeModule.extra_record_fields writes (§13) —
            # not None — so this exercises the actual mismatch between the record's
            # joined label and track.yaml's real (None) class filter, the gap that
            # once made TrackingPreview read zero rows for every merged run.
            "class_name": "black mouse+white mouse",
            "hyperparams": {},
            "every_n_frames": 1,
            "source_run_ids": ["track_a", "track_b"],
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    recording_store.write_recording(sdir, data)

    preview = recording.tracking_preview("track_merged")
    track_ids, presence = preview.track_presence()
    assert track_ids == [0, 1]
    assert presence[0].tolist() == [True, True, True, True, False]  # black mouse
    assert presence[1].tolist() == [True, True, True, True, False]  # white mouse

    _actual, _rgb, detections = preview.frame(0, 0)
    assert {d.object_id for d in detections} == {0, 1}
    by_object_id = {d.object_id: d.class_name for d in detections}
    assert by_object_id == {0: "black mouse", 1: "white mouse"}


def test_tracking_preview_unknown_run(tmp_path):
    recording = _recording_with_detection(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        recording.tracking_preview("track_99")


_INPUT_SPEC = [
    {"type": "init_binary_mask"},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 32},
]


def test_input_image_preview_renders_for_track(tmp_path):
    recording = _add_tracking(_recording_with_detection(tmp_path))
    preview = recording.input_image_preview("track_01", _INPUT_SPEC)
    assert preview.part_frame_counts == [5]  # slider spans the whole video
    actual, rgb, model_input = preview.input_at(0, 0, 0)
    assert actual == 0
    assert rgb.shape == (48, 64, 3)  # original frame, full resolution
    assert model_input.shape == (32, 32, 3)  # the constructor's output
    assert (model_input == 255).any()  # the mask silhouette rendered


def test_input_image_preview_none_when_track_absent(tmp_path):
    recording = _add_tracking(_recording_with_detection(tmp_path))
    preview = recording.input_image_preview("track_01", _INPUT_SPEC)
    actual, rgb, model_input = preview.input_at(0, 1, 0)  # frame 1: no detections
    assert actual == 1
    assert rgb.shape == (48, 64, 3)  # original still shows
    assert model_input is None  # nothing to construct for this track here


def test_input_image_preview_filters_to_requested_track(tmp_path):
    recording = _add_tracking(_recording_with_detection(tmp_path))
    preview = recording.input_image_preview("track_01", _INPUT_SPEC)
    # frame 0 has track 0; track 99 was never tracked -> no input
    assert preview.input_at(0, 0, 99)[2] is None


def _add_embedding(recording, run_id="emb_01"):
    """A completed embedding run over track 0: two embedded frames (0 and 2)."""
    edir = recording.dir / "embeddings" / run_id
    (edir / "track_0").mkdir(parents=True)
    with EmbeddingPartWriter(edir / "track_0" / "part_001.zarr", output_dim=4) as w:
        w.add_row(0, 0, 0, np.array([1, 0, 0, 0], np.float32))  # det row 0, frame 0
        w.add_row(2, int(2e9 / 30), 2, np.array([0, 1, 0, 0], np.float32))  # frame 2
    (edir / "embedding.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": run_id,
                "tracking_run_id": "track_01",
                "model_id": "fake_embedder",
                "model_variant": None,
                "constructor": "full_frame",
                "output_dim": 4,
                "track_ids": [0],
                "renderer_spec": _INPUT_SPEC,
                "status": "complete",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return recording


def test_union_detection_unions_boxes_and_masks():
    """``source_crop_masked`` frames a location export's tile of several animals by one box —
    this is what stands in for a single detection there."""
    black = np.zeros((10, 10), dtype=bool)
    black[0:3, 0:3] = True
    white = np.zeros((10, 10), dtype=bool)
    white[7:10, 7:10] = True
    detections = [
        Detection(mask=black, bbox=(0.0, 0.0, 0.3, 0.3), class_name="black mouse",
                   confidence=0.9, object_id=0),
        Detection(mask=white, bbox=(0.7, 0.7, 1.0, 1.0), class_name="white mouse",
                   confidence=0.8, object_id=1),
    ]  # fmt: skip
    combined = _union_detection(detections)
    assert combined.bbox == (0.0, 0.0, 1.0, 1.0)
    assert combined.mask[1, 1] and combined.mask[8, 8]
    assert not combined.mask[5, 5]


def test_union_detection_is_none_with_no_detections():
    assert _union_detection([]) is None


def test_union_detection_skips_a_detection_with_no_box():
    """A detection decoded with no mask/bbox (e.g. boxes-only mode) contributes nothing to
    the union rather than crashing on a None bbox."""
    only = Detection(
        mask=None, bbox=None, class_name="black mouse", confidence=0.9, object_id=0
    )
    assert _union_detection([only]) is None


def test_pca_project_2d_shapes():
    assert pca_project_2d(np.random.rand(5, 8)).shape == (5, 2)
    assert pca_project_2d(np.random.rand(1, 8)).shape == (1, 2)  # degenerate, padded
    assert pca_project_2d(np.zeros((0, 8))).shape == (0, 2)  # empty


# --- locate_frame ------------------------------------------------------------ #
def test_locate_frame_within_the_first_part():
    assert locate_frame([5, 5], 3) == (0, 3)


def test_locate_frame_crosses_a_part_boundary():
    assert locate_frame([5, 5], 5) == (1, 0)
    assert locate_frame([5, 5], 7) == (1, 2)


def test_locate_frame_clamps_past_the_last_frame():
    """A run whose part shrank underfoot resolves to *something* on screen — the run
    preview's own scrub bar and a `--frames` export both hand this a number the caller
    already believed was in range."""
    assert locate_frame([5, 5], 999) == (1, 4)


def test_locate_frame_with_no_parts_clamps_to_zero():
    assert locate_frame([], 3) == (0, 0)


def test_embedding_preview_metadata_and_coverage(tmp_path):
    recording = _add_embedding(_add_tracking(_recording_with_detection(tmp_path)))
    preview = recording.embedding_preview("emb_01")
    meta = preview.metadata()
    assert meta["model_id"] == "fake_embedder"
    assert meta["constructor"] == "full_frame"
    assert preview.track_ids() == [0]
    assert preview.coverage() == [{"track_id": 0, "n_frames": 2, "n_parts": 1}]


def test_embedding_preview_pca_2d(tmp_path):
    recording = _add_embedding(_add_tracking(_recording_with_detection(tmp_path)))
    coords, tracks, locators = recording.embedding_preview("emb_01").pca_2d()
    assert coords.shape == (2, 2)
    assert tracks.tolist() == [0, 0]
    assert locators[0] == (0, "part_001", 0, 0)  # (track, stem, frame_idx, t_ns)


def test_embedding_preview_pca_2d_caps_points(tmp_path):
    recording = _add_tracking(_recording_with_detection(tmp_path))
    edir = recording.dir / "embeddings" / "emb_big"
    (edir / "track_0").mkdir(parents=True)
    rng = np.random.default_rng(1)
    with EmbeddingPartWriter(edir / "track_0" / "part_001.zarr", output_dim=4) as w:
        for f in range(10):
            w.add_row(
                f, int(f * 1e9 / 30), f, rng.standard_normal(4).astype(np.float32)
            )
    (edir / "embedding.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "emb_big",
                "tracking_run_id": "track_01",
                "model_id": "fake_embedder",
                "model_variant": None,
                "constructor": "full_frame",
                "output_dim": 4,
                "track_ids": [0],
                "renderer_spec": _INPUT_SPEC,
                "status": "complete",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    preview = recording.embedding_preview("emb_big")
    assert preview.pca_2d()[0].shape == (10, 2)  # no cap → every row
    coords, tracks, locators = preview.pca_2d(max_points=3)  # capped
    assert coords.shape == (3, 2) and len(tracks) == 3 and len(locators) == 3
    c2, _, l2 = preview.pca_2d(max_points=3)  # deterministic (seeded)
    assert np.array_equal(coords, c2) and locators == l2


def test_embedding_preview_frame_view(tmp_path):
    recording = _add_embedding(_add_tracking(_recording_with_detection(tmp_path)))
    actual, rgb, model_input = recording.embedding_preview("emb_01").frame_view(
        0, "part_001", 0
    )
    assert actual == 0
    assert rgb.shape == (48, 64, 3)
    assert model_input.shape == (32, 32, 3)  # the run's renderer_spec output
    assert (model_input == 255).any()  # the mask silhouette


def test_embedding_preview_frame_position(tmp_path):
    """A part-local frame, placed in the whole recording: the number the scrubbers show."""
    recording = _add_embedding(_add_tracking(_recording_with_detection(tmp_path)))
    preview = recording.embedding_preview("emb_01")
    assert preview.frame_position("part_001", 2) == (2, 5)
    assert preview.frame_position("part_009", 2) is None  # a stem the run does not have


def test_embedding_preview_frame_position_offsets_later_parts(tmp_path):
    """Every part before this one is counted in — the whole reason a locator's
    part-local index cannot be shown as it is."""
    import types

    recording = _add_embedding(_add_tracking(_recording_with_detection(tmp_path)))
    preview = recording.embedding_preview("emb_01")
    stems = ["part_001", "part_002", "part_003"]
    preview._input = types.SimpleNamespace(
        part_index_for_stem=lambda stem: stems.index(stem) if stem in stems else None,
        part_frame_counts=[5, 7, 4],
    )
    assert preview.frame_position("part_001", 3) == (3, 16)
    assert preview.frame_position("part_003", 1) == (13, 16)


def test_embedding_preview_unknown_run(tmp_path):
    recording = _add_tracking(_recording_with_detection(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        recording.embedding_preview("emb_99")


# --- _square_crop mask blending --------------------------------------------- #
def _blend_then_crop(rgb, mask, bbox, size, tint, alpha):
    """What the code did before: tint the whole frame, then throw all but the crop away."""
    from detail.core.visualize import draw_mask

    return _square_crop(draw_mask(rgb, mask, tint, alpha=alpha), bbox, size)


@pytest.mark.parametrize(
    "bbox",
    [
        (0.30, 0.30, 0.50, 0.60),  # wholly inside the frame
        (0.00, 0.00, 0.12, 0.15),  # clipped at the top-left corner
        (0.88, 0.80, 1.00, 1.00),  # clipped at the bottom-right corner
        (0.10, 0.45, 0.90, 0.55),  # wider than tall, so the square pads vertically
    ],
)
def test_blending_into_the_crop_gives_the_same_picture_as_blending_the_frame(bbox):
    """The optimization is that a mask blended over ~90k crop pixels beats one blended over
    ~2M frame pixels — 14.4 ms a frame, the largest single item in loading a clip. It is only
    an optimization if the picture is identical, including where the crop runs off the edge
    and the padding must stay unblended."""
    from detail.api.run_preview import _square_crop

    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=bool)
    mask[20:90, 30:120] = True
    tint, alpha = (200, 30, 90), 0.35

    fast = _square_crop(rgb, bbox, 64, mask=mask, tint=tint, alpha=alpha)
    slow = _blend_then_crop(rgb, mask, bbox, 64, tint, alpha)
    assert np.array_equal(fast, slow)


def test_layers_blend_each_animal_in_its_own_colour():
    """A location export's tile pictures its animals each in its own colour (``layers``), the
    same picture as blending each into the frame in turn and cropping.
    """
    from detail.api.run_preview import _square_crop

    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    first = np.zeros((120, 160), dtype=bool)
    first[20:90, 20:60] = True
    second = np.zeros((120, 160), dtype=bool)
    second[20:90, 100:140] = True
    bbox, alpha = (0.1, 0.15, 0.9, 0.8), 0.35
    blue, orange = (31, 119, 180), (230, 127, 14)

    fast = _square_crop(
        rgb, bbox, 64, alpha=alpha, layers=[(first, blue), (second, orange)]
    )
    from detail.core.visualize import draw_mask

    both = draw_mask(
        draw_mask(rgb, first, blue, alpha=alpha), second, orange, alpha=alpha
    )
    assert np.array_equal(fast, _square_crop(both, bbox, 64))


def test_a_crop_with_no_tint_is_untouched_by_the_mask():
    """A plain crop must not pay for masks it was not asked for, and must not show one."""
    from detail.api.run_preview import _square_crop

    rng = np.random.default_rng(8)
    rgb = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    mask = np.ones((60, 80), dtype=bool)
    bbox = (0.2, 0.2, 0.6, 0.7)
    assert np.array_equal(
        _square_crop(rgb, bbox, 32, mask=mask, tint=None),
        _square_crop(rgb, bbox, 32),
    )


def test_a_mask_of_the_wrong_shape_is_ignored_rather_than_raising():
    """A mask that does not describe this frame is a mismatched pair, and ``draw_mask``
    already declines one — the crop path has to decline it the same way."""
    from detail.api.run_preview import _square_crop

    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.7)
    assert np.array_equal(
        _square_crop(rgb, bbox, 32, mask=np.ones((7, 7), dtype=bool), tint=(1, 2, 3)),
        _square_crop(rgb, bbox, 32),
    )
