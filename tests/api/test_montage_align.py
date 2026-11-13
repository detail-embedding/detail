"""A cluster montage aligned to heading, and the overlay (§22 *Align to heading, and Overlay*).

**What is pinned, and why.** An aligned tile turns its animal by **the embedding's heading** —
the embedding run's keypoint track run and heading definition, through
``pipeline.embedding.headings_for_run``, the one computation the embedding itself uses — so
the montage cannot disagree with what the clusters were built from. The synthetic animal here
is an asymmetric shape (a narrow body with a wide head) placed at a different, known heading on
every frame: aligned, every frame's tile is the **same** picture with the wide end up, and
unaligned they differ. A frame the embedding would write no row for (here, after the last
defined heading) is a grey *no heading* tile, never the unturned frame; a clip with no heading
definition says why on every tile.

The overlay draws each clip's outline in its own colour, centred and turned, with its skeleton
on top, and a legend; single-frame members make a still PNG and refuse a video.

The class these guard: a view that claims to be aligned while turning by something other than
the embedding's heading, or that quietly shows an unturned frame among turned ones — both look
plausible in a grid, and nothing else fails.
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.api import clip_montage as montage_api
from detail.api.clip_montage import ClusterMontage, MontageClip, MontageLayers
from detail.core.skeleton import Skeleton
from detail.core.visualize import MASK_TINT
from detail.storage import heading_definitions
from detail.storage import keypoint_runs
from detail.storage import recording as recording_store
from detail.storage.keypoint_zarr import KeypointTrackPartWriter
from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter
from tests.api.test_run_preview import (
    _add_embedding,
    _recording_with_detection,
    _write_track_yaml,
)

H, W = 48, 64
FRAME_NS = int(1e9 / 30)
#: Each frame's heading, image degrees (y down, 0 = facing +x).
HEADINGS = (0.0, 90.0, 22.5, -45.0, 135.0, -90.0)
#: Frame 2 has no keypoints (interpolated across, the short way from 90 to -45: 22.5, where
#: its body is); frame 5 none either (after the last defined row: no heading).
NO_POINTS = (2, 5)
SKELETON = Skeleton(names=("nose", "tail"), edges=(("nose", "tail"),))
CENTRE = (32.0, 24.0)


def _animal(deg: float) -> np.ndarray:
    """A narrow body (half-width 2) from tail to nose, 22 px long, with a wide head (half-width
    4) over its front quarter, facing ``deg`` about :data:`CENTRE`."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    a = np.radians(deg)
    dx, dy = xs - CENTRE[0], ys - CENTRE[1]
    along = dx * np.cos(a) + dy * np.sin(a)
    across = np.abs(-dx * np.sin(a) + dy * np.cos(a))
    body = (np.abs(along) <= 11) & (across <= 2)
    head = (along >= 6) & (along <= 11) & (across <= 4)
    return body | head


def _rle(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return (xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H)


def _turning(tmp_path, *, aligned_run=True, keypoints=True):
    """A recording of one animal (track 0) turning on the spot: frame ``i`` faces
    ``HEADINGS[i]``. The embedding run reads keypoint run ``kt_01`` (nose / tail on every frame
    but :data:`NO_POINTS`); ``aligned_run`` gives it the nose-front / tail-back heading, else
    it names none."""
    n = len(HEADINGS)
    recording = _recording_with_detection(tmp_path, n_frames=n)
    ddir = recording.dir / "detections" / "det_01"
    shutil.rmtree(ddir / "part_001.zarr")
    with DetectionPartWriter(ddir / "part_001.zarr") as writer:
        for i, deg in enumerate(HEADINGS):
            mask = _animal(deg)
            writer.add_row(i, i * FRAME_NS, 0, "fish", 0.9, _box(mask), _rle(mask))
    adir = recording.dir / "tracking" / "track_01" / "assignments"
    adir.mkdir(parents=True)
    with AssignmentPartWriter(adir / "part_001.zarr") as writer:
        for _ in HEADINGS:
            writer.add_object_id(0)
    _write_track_yaml(
        recording.dir / "tracking" / "track_01", detection_run_id="det_01"
    )
    data = recording_store.read_recording(recording.dir)
    data.setdefault("tracking_runs", []).append(
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    )
    recording_store.write_recording(recording.dir, data)
    recording = _add_embedding(recording)
    if not keypoints:
        return recording
    run_dir = keypoint_runs.track_run_dir(recording.dir, "kt_01")
    keypoint_runs.write_track_yaml(
        run_dir,
        {
            "schema_version": 1,
            "id": "kt_01",
            "created_at": "2026-11-08T00:00:00Z",
            "detect_run_id": "kd_01",
            "tracking_run_id": "track_01",
            "method": "despike_smooth",
            "config": {"decode_cutoff": None},
            "skeleton": SKELETON.to_dict(),
            "skeleton_digest": SKELETON.digest(),
        },
    )
    rows = [i for i in range(n) if i not in NO_POINTS]
    points = []
    for i in rows:
        a = np.radians(HEADINGS[i])
        tip = np.array([np.cos(a), np.sin(a)]) * 11.0
        nose = (np.array(CENTRE) + tip) / np.array([W, H])
        tail = (np.array(CENTRE) - tip) / np.array([W, H])
        points.append([nose, tail])
    path = keypoint_runs.track_part_path(run_dir, 0, "part_001")
    path.parent.mkdir(parents=True, exist_ok=True)
    with KeypointTrackPartWriter(path, SKELETON.names) as writer:
        writer.add_rows(
            np.array(rows),
            np.array(rows, np.int64) * FRAME_NS,
            np.zeros(len(rows), np.int32),
            np.array(points, np.float32),
            np.full((len(rows), 2), 0.9, np.float32),
            np.zeros((len(rows), 2), np.int8),
        )
    meta_path = recording.dir / "embeddings" / "emb_01" / "embedding.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig"))
    meta["keypoint_run_id"] = "kt_01"
    if aligned_run:
        meta.update(
            align="rotate",
            heading="nose_tail",
            heading_definition={"front": ["nose"], "back": ["tail"]},
            heading_cutoff=None,
            max_gap_s=0.5,
            mirror_hold_s=1.0,
            facing_camera_ratio=0.5,
        )
    meta_path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")
    return recording


def _clip(recording, *, rank=1, first=0, last=len(HEADINGS) - 1) -> MontageClip:
    return MontageClip(
        rank=rank,
        recording_id=recording.id,
        recording_name="s1",
        subject=0,
        embed_run_id="emb_01",
        start_t_ns=first * FRAME_NS,
        end_t_ns=last * FRAME_NS,
    )


def _montage(recording, clips=None) -> ClusterMontage:
    return ClusterMontage(
        recording.dir.parent.parent, clips or [_clip(recording)], workers=1
    )


ALIGNED = MontageLayers(mask=True, mask_trail=False, aligned=True)
UNALIGNED = MontageLayers(mask=True, mask_trail=False)
TILE = 96


def _filled(tile: np.ndarray) -> np.ndarray:
    """Where the Raw-off tile has the animal's tint (not white, not grey)."""
    return (tile != 255).any(axis=2)


def _wide_end_up(tile: np.ndarray) -> bool:
    """The animal's wider half is its upper half."""
    filled = _filled(tile)
    ys, _xs = np.nonzero(filled)
    mid = (ys.min() + ys.max()) / 2.0
    width = filled.sum(axis=1)
    return width[: int(mid)].max() > width[int(mid) + 1 :].max()


def test_every_frame_turns_by_its_own_heading_so_the_animal_faces_up(tmp_path):
    recording = _turning(tmp_path)
    montage = _montage(recording)
    tiles = {
        i: montage.tile(0, i, ALIGNED, TILE)
        for i in range(len(HEADINGS))
        if i != 5  # no heading: below
    }
    for i, tile in tiles.items():
        filled = _filled(tile)
        ys, xs = np.nonzero(filled)
        assert ys.max() - ys.min() > 1.5 * (
            xs.max() - xs.min()
        ), f"frame {i} ({HEADINGS[i]} deg): the body is upright"
        assert _wide_end_up(tile), f"frame {i}: its head is up"
    reference = _filled(tiles[0])
    for i, tile in tiles.items():
        overlap = (_filled(tile) & reference).sum() / reference.sum()
        assert overlap > 0.8, f"frame {i} is the same upright animal as frame 0"


def test_unaligned_tiles_show_the_turns(tmp_path):
    recording = _turning(tmp_path)
    montage = _montage(recording)
    across = _filled(montage.tile(0, 0, UNALIGNED, TILE))
    ys, xs = np.nonzero(across)
    assert xs.max() - xs.min() > ys.max() - ys.min(), "frame 0 faces +x, unturned"


def test_a_gap_within_max_gap_s_is_interpolated_like_the_embedding(tmp_path):
    """Frame 2 has no keypoints; frames 1 and 3 do, 67 ms apart (under ``max_gap_s``)."""
    recording = _turning(tmp_path)
    montage = _montage(recording)
    deg = montage._headings(0).deg
    assert np.isfinite(deg[2])
    # 90 -> -45 the short way round the circle (through 22.5)
    assert deg[2] == pytest.approx(22.5, abs=1e-3)
    assert _wide_end_up(montage.tile(0, 2, ALIGNED, TILE))


def test_a_frame_with_no_heading_is_a_grey_tile_never_the_unturned_frame(tmp_path):
    """Frame 5 comes after the last defined heading: the embedding writes no row for it."""
    recording = _turning(tmp_path)
    montage = _montage(recording)
    assert not np.isfinite(montage._headings(0).deg[5])
    tile = montage.tile(0, 5, ALIGNED, TILE)
    unturned = montage.tile(0, 5, UNALIGNED, TILE)
    assert not np.array_equal(tile, unturned)
    corner = tile[2, 2]
    assert corner[0] == corner[1] == corner[2] == montage_api._NO_PICTURE[0]
    assert not (tile == np.array(MASK_TINT, np.uint8)).all(axis=2).any(), "no animal"


def test_the_heading_is_the_embeddings_own_computation(tmp_path, monkeypatch):
    """Every heading the montage turns by comes from ``headings_for_run``, at every frame."""
    from detail.pipeline import embedding as embedding_pipeline

    calls = []
    real = embedding_pipeline.headings_for_run

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(embedding_pipeline, "headings_for_run", spy)
    recording = _turning(tmp_path)
    _montage(recording)._headings(0)
    assert calls, "the montage reads the embedding's heading"
    assert calls[0][3] == 1, "at every frame (every_n_frames=1)"


def test_an_unaligned_run_turns_by_the_experiments_one_fitting_definition(tmp_path):
    recording = _turning(tmp_path, aligned_run=False)
    montage = _montage(recording)
    assert montage._headings(0).note is not None, "no definition saved yet"
    assert montage.heading_note().startswith("No clip can be aligned")
    tile = montage.tile(0, 0, ALIGNED, TILE)
    assert tile[2, 2][0] == montage_api._NO_PICTURE[0]
    heading_definitions.save_definition(
        recording.dir.parent.parent, "nose_tail", ["nose"], ["tail"]
    )
    fresh = _montage(recording)
    assert fresh.heading_note() is None
    assert _wide_end_up(fresh.tile(0, 1, ALIGNED, TILE))


def test_a_clip_without_keypoints_says_why_on_every_tile(tmp_path):
    recording = _turning(tmp_path, keypoints=False)
    montage = _montage(recording)
    assert "used no keypoints" in montage.heading_note()


def test_the_turned_side_is_fixed_for_the_clip():
    """One side for every frame: the largest of the boxes turned by their own headings."""
    boxes = np.array([[0.0, 0.0, 0.5, 0.1], [0.0, 0.0, 0.1, 0.1]])
    upright = montage_api.turned_side(
        boxes, np.array([-90.0, 0.0]), (100, 100), margin=0
    )
    assert upright == 50  # facing up already: its own longer side
    turned = montage_api.turned_side(boxes, np.array([0.0, 0.0]), (100, 100), margin=0)
    assert turned == 50
    diagonal = montage_api.turned_side(
        boxes, np.array([45.0, 0.0]), (100, 100), margin=0
    )
    assert diagonal == int(np.ceil((50 + 10) / np.sqrt(2)))


# -- the overlay --------------------------------------------------------------------- #
def _two_clips(recording):
    return [
        _clip(recording, rank=1, first=0, last=1),
        _clip(recording, rank=2, first=3, last=4),
    ]


def test_the_overlay_draws_each_clip_in_its_own_colour_with_a_legend(tmp_path):
    recording = _turning(tmp_path)
    clips = _two_clips(recording)
    montage = _montage(recording, clips)
    labels = montage_api.montage_labels(clips, blind=False)
    picture = montage.overlay(0, labels, size=200)
    square = picture[:, :200]
    colours = montage_api.overlay_colours(2)
    assert colours[0] != colours[1]
    for colour in colours:
        ink = (np.abs(square.astype(int) - np.array(colour)).sum(axis=2) < 30).sum()
        assert ink > 20, f"clip in {colour} is outlined"
    legend = picture[:, 200:]
    for colour in colours:
        assert (legend == np.array(colour, np.uint8)).all(axis=2).sum() > 20
    assert picture.shape[0] == 200 and picture.shape[1] > 200


def test_overlay_colours_are_distinct_for_twelve():
    colours = montage_api.overlay_colours(montage_api.OVERLAY_CLIPS)
    assert len(set(colours)) == montage_api.OVERLAY_CLIPS == 12


def test_a_clip_without_a_heading_is_left_out_and_its_legend_says_so(tmp_path):
    recording = _turning(tmp_path)
    clips = [
        _clip(recording, rank=1, first=0, last=0),
        _clip(recording, rank=2, first=5, last=5),
    ]
    montage = _montage(recording, clips)
    members = [montage._overlay_member(i, f, False) for i, f in ((0, 0), (1, 5))]
    assert isinstance(members[0], montage_api._OverlayMember)
    assert members[1] == montage_api.NO_HEADING


def test_single_frame_members_make_a_still_png_and_refuse_a_video(tmp_path):
    recording = _turning(tmp_path)
    clips = [
        _clip(recording, rank=r, first=f, last=f) for r, f in ((1, 0), (2, 1), (3, 3))
    ]
    montage = _montage(recording, clips)
    assert montage.is_still
    labels = montage_api.montage_labels(clips, blind=False)
    out = montage.export_overlay(tmp_path / "overlay.png", labels, size=160)
    picture = cv2.imread(str(out))
    assert picture is not None and picture.shape[0] == 160
    with pytest.raises(ValueError, match="single frames"):
        montage.export_overlay(tmp_path / "overlay.mp4", labels, size=160)


def test_clips_make_an_overlay_video_on_the_montage_clock(tmp_path):
    import av

    recording = _turning(tmp_path)
    clips = _two_clips(recording)
    montage = _montage(recording, clips)
    out = montage.export_overlay(
        tmp_path / "overlay.mp4",
        montage_api.montage_labels(clips, blind=False),
        size=160,
    )
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
    assert frames == montage.n_steps


# -- one renderer, one heading: a fail-closed scan of the montage's module ----------------- #
#: What draws pixels, reads masks, keypoints or video, or computes a heading. The montage gets
#: every picture from ``clip_views`` and every heading from ``subject_headings``
#: (``run_preview``); importing one of these is a second path that can drift from them.
_FORBIDDEN_MODULES = (
    "detail.analysis",
    "detail.core.utils",
    "detail.models",
    "detail.pipeline",
    "detail.storage.keypoint_runs",
    "detail.storage.keypoint_zarr",
    "detail.storage.video",
    "detail.storage.zarr_io",
    "detail.transforms",
)
_FORBIDDEN_NAMES = {
    "VideoCapture",
    "decode_mask",
    "headings_for_run",
    "heading_at",
    "track_series",
    "row_headings",
}
#: What the montage may take from ``run_preview``: the renderer, its heading answer, a fade.
_FROM_RUN_PREVIEW = {"ClipQueryPreview", "ClipHeadings", "_MASK_TRAIL_ALPHA"}


def _second_paths(source: str) -> list[str]:
    import ast

    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in node.names]
            modules = [node.module, *(f"{node.module}.{n}" for n in names)]
            if any(m.startswith(_FORBIDDEN_MODULES) for m in modules):
                found.append(f"line {node.lineno}: from {node.module} import {names}")
            if node.module == "detail.api.run_preview":
                extra = sorted(set(names) - _FROM_RUN_PREVIEW)
                if extra:
                    found.append(f"line {node.lineno}: run_preview's {extra}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_FORBIDDEN_MODULES):
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            found.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            found.append(f"line {node.lineno}: {node.id}")
    return found


def test_the_montage_draws_and_turns_only_through_clip_views():
    from pathlib import Path

    source = Path(montage_api.__file__).read_text(encoding="utf-8-sig")
    assert _second_paths(source) == []
    assert (
        "subject_headings" in source and "clip_views" in source
    ), "the scan has a subject"


@pytest.mark.parametrize(
    "snippet",
    [
        "from detail.core.utils import decode_mask",
        "from detail.pipeline.embedding import headings_for_run",
        "from detail.analysis import body_axis",
        "import detail.storage.video",
        "from detail.api.run_preview import _ClipPoints",
        "cap = cv2.VideoCapture(path)",
        "deg = body_axis.heading_at(series, t, 0.5)",
    ],
)
def test_the_scan_catches_a_second_path(snippet):
    assert _second_paths(snippet), snippet
