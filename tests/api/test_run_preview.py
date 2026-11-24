"""DetectionPreview: per-frame boxes + counts from a detection run's zarr — and the row-block
cache every clip surface reads them through.

**Two classes this file guards (``docs/recorded_classes.md``).**

*A cached read path keyed without the flag that decides its cost.* `DetectionPreview.read_rows`
serves a short span from a cached block of `ROW_BLOCK` rows, because each zarr column read is
its own ~2 ms call however few rows it returns. The port this replaced keyed the block on
``(part, block)`` and always read ``mask_rle``, so a caller that asked for no masks
(``frame_plain``, the one path with a published 4.68 ms number) paid the masks decode for every
block it touched. The flag is in the key, ``(part stem, block, with_masks)``, and a masks-off
read never names the column. Two arms, because each sees one direction: a masks-off walk counts
zero ``mask_rle`` reads (the negative arm the port's read-count guard lacked), and a masks-on
read after a masks-off one of the same block still carries masks (what the key alone decides; a
key change is invisible to the first arm, which is how the three-state check found the second).

*A cache keyed on ``id()`` of an object a caller may rebuild.* The same block cache keyed on
``id(part)``: the part list is rebuilt whenever a run's parts are re-read, so every cached block
missed after a rebuild, and an ``id`` is reused once its object is collected, so a block cached
for one part could answer for another. A key that is a value (``part["stem"]``) survives the
rebuild and names the part.
"""

import json

import cv2
import numpy as np
import pytest
import yaml

from detail.api.run_preview import _square_crop, pca_project_2d
from detail.api.recording import Recording
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import video as video_store
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartWriter,
    EmbeddingPartWriter,
)


def _recording_with_detection(tmp_path, n_frames=5, write_video=None, encoded=None):
    """A recording with a tiny video + a hand-written detection run (no model).

    ``write_video`` overrides how the video is produced (e.g. the ``write_vfr_video``
    fixture); the timestamp map is then generated from the file's real PTS rather
    than assumed constant-rate.

    ``encoded`` (``(h, w)``) encodes the masks at that size instead of the footage's 48x64,
    and writes the run's ``job.json`` saying so, as a run whose footage now decodes at another
    size than its detection did. The first mask is the same box either way: rows and columns
    10:20 of the footage.
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

    mask_h, mask_w = encoded or (48, 64)
    mask0 = np.zeros((mask_h, mask_w), np.uint8)
    y0, y1 = 10 * mask_h // 48, 20 * mask_h // 48
    x0, x1 = 10 * mask_w // 64, 20 * mask_w // 64
    mask0[y0:y1, x0:x1] = 1
    rle0 = mask_utils.encode(np.asfortranarray(mask0))["counts"].decode("ascii")
    ddir = sdir / "detections" / "det_01"
    ddir.mkdir(parents=True)
    if encoded is not None:
        (ddir / "job.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "mask_resolution": [mask_h, mask_w],
                    "parts": [{"id": "part_001", "mask_resolution": [mask_h, mask_w]}],
                }
            ),
            encoding="utf-8",
        )
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


def test_a_mask_encoded_at_another_size_lands_where_it_was_on_the_footage(tmp_path):
    """A run's masks decode at the size it encoded them at, then resize to the footage.

    Decoded at the footage's 48x64, a 24x32 mask's runs are laid down the columns of a grid
    twice as tall and the box comes out as stripes. Both readers are held to it: the
    overlay on a frame and the mask trail's per-frame masks.
    """
    recording = _add_tracking(_recording_with_detection(tmp_path, encoded=(24, 32)))
    preview = recording.detection_preview("det_01")
    mask = preview.frame(0, 0, with_masks=True)[2][0].mask
    assert mask.shape == (48, 64)
    expected = np.zeros((48, 64), bool)
    expected[10:20, 10:20] = True
    np.testing.assert_array_equal(mask, expected)

    trail = recording.tracking_preview("track_01").subject_masks(0, 0, [0], (48, 64))
    np.testing.assert_array_equal(trail[0], expected)


def test_detection_preview_unknown_run(tmp_path):
    recording = _recording_with_detection(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        recording.detection_preview("det_99")


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


def _add_tracking(recording):
    """A tracking run over det_01: object_id parallel to its 3 detection rows."""
    sdir = recording.dir
    adir = sdir / "tracking" / "track_01" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        writer.add_object_id(0)  # frame 0, row 0 -> track 0
        writer.add_object_id(-1)  # frame 0, row 1 -> unassigned (dropped)
        writer.add_object_id(0)  # frame 2, row 0 -> track 0
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


# --- EmbeddingPreview + PCA ------------------------------------------------- #
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


def test_pca_project_2d_shapes():
    assert pca_project_2d(np.random.rand(5, 8)).shape == (5, 2)
    assert pca_project_2d(np.random.rand(1, 8)).shape == (1, 2)  # degenerate, padded
    assert pca_project_2d(np.zeros((0, 8))).shape == (0, 2)  # empty


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


# --- the row-block reader: a caller pays only for the columns it asked for --------------- #
def _count_column_reads(monkeypatch) -> list:
    """Every zarr column read, by name — the cost ``read_rows`` exists to bound."""
    from detail.storage.zarr_io import PartZarrReader

    reads: list = []
    real = PartZarrReader.column
    monkeypatch.setattr(
        PartZarrReader,
        "column",
        lambda self, name: reads.append(name) or real(self, name),
    )
    return reads


def test_frame_plain_reads_no_mask_column(tmp_path, monkeypatch):
    """The negative arm of the block cache: a caller that asked for no masks never pays for
    the ``mask_rle`` column, not even to fill a block. A block keyed without ``with_masks``
    read every row's RLE for ``frame_plain``, the one path with a published 4.68 ms number.
    """
    from detail.api.run_preview import ClipQueryPreview

    from tests.clip_fixtures import walking  # local: that module imports this one

    recording = walking(tmp_path, keypoints=False, kinematics=False)
    preview = ClipQueryPreview(recording.dir.parent.parent)
    reads = _count_column_reads(monkeypatch)
    for frame in (0, 1, 2):
        assert preview.frame_plain(recording.id, "emb_01", 0, frame) is not None
    assert reads.count("mask_rle") == 0
    assert reads.count("bbox") == 1, "one block serves every frame of the span"


def test_frame_masked_reads_each_block_once(tmp_path, monkeypatch):
    """The positive arm: with masks, the column is read once for the block, not once a
    frame — the saving the block exists for."""
    from detail.api.run_preview import ClipQueryPreview

    from tests.clip_fixtures import walking  # local: that module imports this one

    recording = walking(tmp_path, keypoints=False, kinematics=False)
    preview = ClipQueryPreview(recording.dir.parent.parent)
    reads = _count_column_reads(monkeypatch)
    for frame in (0, 1, 2):
        assert preview.frame_masked(recording.id, "emb_01", 0, frame) is not None
    assert reads.count("mask_rle") == 1


def test_a_masked_read_after_a_plain_one_still_carries_masks(tmp_path):
    """The other direction of the key: a block filled for a masks-off caller is never served
    to a masks-on one. With ``with_masks`` out of the key it was, and every frame of the block
    came back with no masks for as long as the block stayed cached."""
    from detail.api.run_preview import DetectionPreview

    recording = _recording_with_detection(tmp_path)
    preview = DetectionPreview(recording.dir, "det_01")
    plain = preview.read_rows(preview._parts[0], 0, 2, with_masks=False)
    assert plain["mask_rle"] is None
    masked = preview.read_rows(preview._parts[0], 0, 2, with_masks=True)
    assert masked["mask_rle"] is not None and len(masked["mask_rle"]) == 2


def test_row_blocks_survive_a_rebuilt_part_list(tmp_path, monkeypatch):
    """The block is keyed on the part's stem, not ``id(part)``: a preview whose part dicts
    are rebuilt (or whose freed dict's id is reused) still finds its blocks."""
    from detail.api.run_preview import DetectionPreview

    recording = _recording_with_detection(tmp_path)
    preview = DetectionPreview(recording.dir, "det_01")
    reads = _count_column_reads(monkeypatch)
    preview.read_rows(preview._parts[0], 0, 2, with_masks=True)
    preview._parts = [dict(part) for part in preview._parts]
    preview.read_rows(preview._parts[0], 0, 2, with_masks=True)
    assert reads.count("mask_rle") == 1


# --- clip_boxes: the animal's boxes over a clip, off the rows ----------------------------- #
def test_clip_boxes_reads_the_subjects_box_per_frame_without_the_video(
    tmp_path, monkeypatch
):
    """What a montage fixes its crop from: one normalized box per frame of the clip, read off
    the detection rows — the mask column untouched, the video unopened."""
    from detail.api.run_preview import ClipQueryPreview

    from tests.clip_fixtures import H, W, count_decodes, walking

    recording = walking(tmp_path, keypoints=False, kinematics=False)
    preview = ClipQueryPreview(recording.dir.parent.parent)
    reads = _count_column_reads(monkeypatch)
    decodes = count_decodes(monkeypatch)
    boxes = preview.clip_boxes(recording.id, "emb_01", 0, 1, 3)
    assert boxes.shape == (3, 4) and boxes.dtype == np.float64
    for row, frame in enumerate((1, 2, 3)):
        x0 = 4 + 6 * frame
        np.testing.assert_allclose(
            boxes[row], (x0 / W, 4 / H, (x0 + 10) / W, 14 / H), atol=1e-6
        )
    other = preview.clip_boxes(recording.id, "emb_01", 1, 0, 0)
    np.testing.assert_allclose(other[0], (4 / W, 30 / H, 14 / W, 40 / H), atol=1e-6)
    assert reads.count("mask_rle") == 0 and decodes == []


def test_clip_boxes_leaves_an_untracked_frame_nan(tmp_path):
    from detail.api.run_preview import ClipQueryPreview

    from tests.clip_fixtures import two_animals

    recording = two_animals(tmp_path)  # track 0 on frames 0 and 2, nothing on frame 1
    preview = ClipQueryPreview(recording.dir.parent.parent)
    boxes = preview.clip_boxes(recording.id, "emb_01", 0, 0, 2)
    assert np.isfinite(boxes[0]).all() and np.isnan(boxes[1]).all()
    np.testing.assert_allclose(boxes[2], (0.5, 0.5, 0.6, 0.6), atol=1e-6)
    assert preview.clip_boxes(recording.id, "emb_01", 0, 2, 1).shape == (0, 4)
