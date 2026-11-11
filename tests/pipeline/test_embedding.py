"""The embedding stage (§14): resolution, the dataset, the runner and its records.

One recorded class lives here, guarded by ``test_a_red_subject_reaches_the_model_in_channel_0``
(``docs/recorded_classes.md``, "a frame handed across a boundary is in the channel order the far
side reads").

**The class.** A frame crosses a boundary in one channel order and is read on the far side in the
other. cv2 decodes BGR; ``RenderContext.frame`` and every embedding model read RGB (DINOv3 turns
each image into a PIL RGB image). ``DetectionMaskDataset`` handed the decoded
frame straight through, so a red animal reached the model in the blue channel, and the
"max-contrast" background — computed with ``COLOR_BGR2LAB`` and stored ``[B, G, R]`` — was the
right colour in the wrong order too, so a dark animal's blue background reached the model as red.

**How it hid.** Every image is a valid image. The worker and the in-process path shared the
dataset, the renderer's own tests drove it with uniform greys whose channels are equal, and
the GUI input preview converted its frames to RGB, so the picture a user inspected was not the
picture the model was given, and the swap changed which frames a frame's nearest
neighbours were.

**The rule.** Convert once, where the frame is read (``core.visualize.bgr_to_rgb`` in
``_iter_with_frames``); compute and store the background in RGB; record ``channel_order`` on
every run so a run from before the fix (no key, read as ``"bgr"``) never pools with one after.
"""

import contextlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import cv2
import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.core.dataclasses import ComputeTarget, EmbeddingJob, RenderContext
from detail.core.utils import decode_mask
from detail.core.protocols import EmbeddingModel
from detail.core.registries import (
    EMBEDDING_MODELS,
    construct_embedding_model,
    register_embedding_model,
)
from detail.pipeline import embedding
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartWriter,
    PartZarrReader,
    TracksSummaryWriter,
    kinematics_part_path,
)
from detail.transforms import basic as _basic  # noqa: F401 — registers transforms
from detail.transforms.renderer import MaskRenderer

H, W, D = 16, 20, 8
RENDERER = [
    {"type": "init_binary_mask"},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 32},
]


class _FakeEmbedder(EmbeddingModel):
    model_id = "fake_embedder"
    output_dim = D
    input_size = (
        64  # distinct from the ABC default (224) so resolve tests are unambiguous
    )
    # Both backends so the multi-GPU coordinator tests (which need backend=cuda
    # to take the pool path) and the original cpu tests share one fake.
    supported_backends = ["cpu", "cuda"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def embed(self, image):
        return np.full(D, float(np.asarray(image).mean()), dtype=np.float32)

    def embed_batch(self, images):
        return np.stack(
            [
                np.full(D, float(np.asarray(img).mean()), dtype=np.float32)
                for img in images
            ],
            axis=0,
        )


@contextlib.contextmanager
def _registered_embedder():
    register_embedding_model("fake_embedder")(_FakeEmbedder)
    try:
        yield
    finally:
        EMBEDDING_MODELS.pop("fake_embedder", None)


class _ResizableEmbedder(_FakeEmbedder):
    """A fake that accepts a per-run read-out + input_size, like DINOv3 (unlike _FakeEmbedder,
    which uses the default __init__), so validate() can thread both through it."""

    model_id = "resizable_embedder"
    READOUTS = ("pooler", "gap")

    def __init__(self, variant=None, readout="pooler", input_size=None):
        super().__init__(variant)
        self.readout = readout
        self.input_size = (
            int(input_size) if input_size is not None else type(self).input_size
        )


@contextlib.contextmanager
def _registered_resizable():
    register_embedding_model("resizable_embedder")(_ResizableEmbedder)
    try:
        yield
    finally:
        EMBEDDING_MODELS.pop("resizable_embedder", None)


def _mask_counts(f):
    m = np.zeros((H, W), np.uint8)
    top = f % (H - 4)
    m[top : top + 4, 2:8] = 1
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _write_test_video(
    path: Path, n_frames: int, color: tuple[int, int, int] | None = None
) -> None:
    """Synthetic MP4 — mp4v fourcc is fairly portable. Frames are uniform-color
    so embeddings differ per frame (the fake takes ``image.mean()``).

    ``color`` paints every frame one colour instead, given in the **BGR** order cv2 writes
    (so ``(0, 0, 255)`` is red), at the masks' own ``W`` x ``H`` so a mask indexes the frame
    directly: the colour variant the channel-order guard needs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (32, 32) if color is None else (W, H)
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path} with mp4v")
    for f in range(n_frames):
        if color is None:
            vw.write(np.full((32, 32, 3), (f * 7) % 250 + 1, dtype=np.uint8))
        else:
            vw.write(np.full((H, W, 3), color, dtype=np.uint8))
    vw.release()


def _setup(
    tmp_path,
    object_ids,
    *,
    n_video_frames: int | None = None,
    frames=None,
    boxes=None,
    color=None,
    masks=None,
):
    """Build a recording with one part: recording.yaml, a synthetic MP4, the
    detection zarr + envelope, and a tracking assignment zarr.

    ``frames`` optionally maps each detection row to a frame index (defaults to
    one row per frame); pass repeats to put several detections on one frame
    (e.g. ``anchor_tracker`` secondaries).

    ``boxes`` optionally gives each row its own normalized xyxy. Needed to tell a frame's
    **union** box from its representative one: with the default constant box those two are
    the same value, so a test written against it cannot fail on the difference.

    ``masks`` optionally gives each row its own RLE counts (default: a 4x6 box that moves
    down with the frame, the same size on every row)."""
    if frames is None:
        frames = list(range(len(object_ids)))
    if n_video_frames is None:
        n_video_frames = (max(frames) + 1) if frames else 0
    recording_dir = tmp_path / "recordings" / "recording_001"
    recording_dir.mkdir(parents=True)

    _write_test_video(
        recording_dir / "video" / "part_001.mp4", n_video_frames, color=color
    )

    (recording_dir / "recording.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "recording_001",
                "video": {
                    "files": [
                        {
                            "index": 0,
                            "file": "video/part_001.mp4",
                            "timestamps": {},
                        }
                    ],
                },
                "detection_runs": [{"id": "det_01", "downsampling_run_id": None}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    det_dir = recording_dir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    with DetectionPartWriter(det_dir / "part_001.zarr") as w:
        seen: dict[int, int] = {}
        for row_i, fr in enumerate(frames):
            det_idx = seen.get(fr, 0)
            seen[fr] = det_idx + 1
            w.add_row(
                fr,
                fr * 1000,
                det_idx,
                "mouse",
                0.9,
                (0.1, 0.2, 0.3, 0.4) if boxes is None else tuple(boxes[row_i]),
                _mask_counts(fr) if masks is None else masks[row_i],
            )
    (det_dir / "job.json").write_text(
        json.dumps(
            {"schema_version": 1, "status": "complete", "mask_resolution": [H, W]}
        ),
        encoding="utf-8",
    )

    track_dir = recording_dir / "tracking" / "track_01"
    (track_dir / "assignments").mkdir(parents=True)
    with AssignmentPartWriter(track_dir / "assignments" / "part_001.zarr") as w:
        for oid in object_ids:
            w.add_object_id(oid)
    (track_dir / "track.yaml").write_text(
        yaml.dump(
            {"schema_version": 1, "id": "track_01", "detection_run_id": "det_01"}
        ),
        encoding="utf-8",
    )
    return recording_dir


def _job(batch_size: int = 32):
    return EmbeddingJob(
        job_id="emb_01",
        recording_id="recording_001",
        embedding_run_id="emb_01",
        tracking_run_id="track_01",
        track_ids=[0],
        model_id="fake_embedder",
        renderer_spec=RENDERER,
        batch_size=batch_size,
        num_workers=0,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


# --------------------------------------------------------------------------- #
# Canvas size + constructor resolution
# --------------------------------------------------------------------------- #
def test_largest_mask_px_from_bboxes(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0, 0])
    # bbox (0.1,0.2,0.3,0.4): width 0.2*W=4.0, height 0.2*H=3.2 -> ceil(max)=4
    assert embedding.largest_mask_px(recording_dir, "track_01", [0]) == 4


def test_largest_mask_px_raises_without_detections(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0, 0])
    with pytest.raises(ValueError, match="no detections"):
        embedding.largest_mask_px(recording_dir, "track_01", [7])


def test_largest_mask_px_caches_by_track_set(tmp_path, monkeypatch):
    """A completed run is immutable, so the mask-size scan is computed once per track
    set and cached (a batch launch calls it per recording); a distinct track set is a
    distinct key, and a raising result is never memoized."""
    from detail.storage import read_cache

    read_cache.clear_read_cache()
    recording_dir = _setup(tmp_path, object_ids=[0, 0, 0])
    real = embedding._largest_mask_px
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(embedding, "_largest_mask_px", counting)
    assert embedding.largest_mask_px(recording_dir, "track_01", [0]) == 4
    assert embedding.largest_mask_px(recording_dir, "track_01", [0]) == 4
    assert calls["n"] == 1  # scanned once; the second call hits the cache

    with pytest.raises(ValueError):
        embedding.largest_mask_px(recording_dir, "track_01", [7])  # distinct key
    with pytest.raises(ValueError):
        embedding.largest_mask_px(recording_dir, "track_01", [7])
    assert calls["n"] == 3  # recomputed both times — a raised result is not cached


def test_resolve_renderer_spec_full_frame_skips_canvas(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    spec = embedding.resolve_renderer_spec(recording_dir, "track_01", [0], "full_frame")
    assert spec == [{"type": "init_frame"}, {"type": "pad_to_square"}]


def test_resolve_renderer_spec_centered_bakes_canvas(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    spec = embedding.resolve_renderer_spec(
        recording_dir, "track_01", [0], "silhouette_centered"
    )
    assert spec[-1] == {"type": "center_in_square", "size": 4}


def test_max_contrast_rgb_is_blue_for_a_dark_animal():
    # a dark warm-gray animal (RGB 56,52,48) -> the in-gamut color farthest in CIELAB is blue,
    # RGB 0,0,255, which validates the hand-picked background of the removed
    # masked_color_centered_blue constructor.
    assert embedding._max_contrast_rgb((56, 52, 48)) == [0, 0, 255]


def test_resolve_renderer_spec_contrast_threads_bg_color(tmp_path, monkeypatch):
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    # the color-sampling stat decodes video (absent in the fixture); stub it and assert the
    # resolved spec bakes the computed color into both the mask fill and the canvas padding.
    monkeypatch.setattr(embedding, "contrast_bg_color", lambda *a, **k: [0, 255, 0])
    spec = embedding.resolve_renderer_spec(
        recording_dir, "track_01", [0], "masked_color_centered_contrast"
    )
    assert {"type": "apply_mask", "background": [0, 255, 0]} in spec
    assert any(
        s.get("type") == "center_in_square" and s.get("fill") == [0, 255, 0]
        for s in spec
    )


def test_resolve_renderer_spec_unknown_constructor(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0])
    with pytest.raises(ValueError, match="unknown constructor"):
        embedding.resolve_renderer_spec(recording_dir, "track_01", [0], "nope")


def test_resolve_renderer_spec_defaults_resize_to_model_input(tmp_path):
    """With a model_id and no explicit resize, the spec ends with a resize to the
    model's input_size — so the GUI (which sends no resize) renders at the model
    input instead of the full canvas the processor re-downscales (§14)."""
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    with _registered_embedder():
        spec = embedding.resolve_renderer_spec(
            recording_dir,
            "track_01",
            [0],
            "silhouette_centered",
            model_id="fake_embedder",
        )
    assert spec[-1] == {"type": "resize", "size": 64}  # _FakeEmbedder.input_size


def test_resolve_renderer_spec_explicit_resize_overrides_model(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    with _registered_embedder():
        spec = embedding.resolve_renderer_spec(
            recording_dir,
            "track_01",
            [0],
            "full_frame",
            model_id="fake_embedder",
            resize=128,
        )
    assert spec[-1] == {"type": "resize", "size": 128}


def test_resolve_renderer_spec_input_size_sets_the_resize(tmp_path):
    # input_size (the model-resolution knob) drives the renderer's emitted size.
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    spec = embedding.resolve_renderer_spec(
        recording_dir, "track_01", [0], "full_frame", input_size=384
    )
    assert spec[-1] == {"type": "resize", "size": 384}


def test_resolve_renderer_spec_resize_overrides_input_size(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    spec = embedding.resolve_renderer_spec(
        recording_dir, "track_01", [0], "full_frame", resize=256, input_size=384
    )
    assert spec[-1] == {"type": "resize", "size": 256}  # explicit resize wins


def test_construct_embedding_model_threads_and_rejects_unsupported():
    from detail.core.registries import construct_embedding_model

    with _registered_resizable():
        m = construct_embedding_model(
            _ResizableEmbedder, None, readout="gap", input_size=384
        )
        assert (m.readout, m.input_size) == ("gap", 384)
    # a model whose __init__ takes neither (the default) rejects both with a clear message
    with pytest.raises(ValueError, match="fixed input size"):
        construct_embedding_model(_FakeEmbedder, None, input_size=384)
    with pytest.raises(ValueError, match="no selectable read-out"):
        construct_embedding_model(_FakeEmbedder, None, readout="gap")


def test_build_job_and_manifest_carry_readout_and_input_size(tmp_path):
    import yaml as _yaml

    module = embedding.EmbeddingModule()
    config = {
        "tracking_run_id": "track_01",
        "track_ids": [0],
        "model_id": "dinov3_vit",
        "model_variant": "vit_base",
        "constructor": "full_frame",
        "readout": "cls_patch",
        "input_size": 384,
        "renderer_spec": [{"type": "init_binary_mask"}],
        "batch_size": 32,
        "num_workers": 0,
        "compute": ComputeTarget("cpu", None, None, None),
    }
    job = module.build_job("recording_001", "emb_01", config)
    assert (job.readout, job.input_size) == ("cls_patch", 384)
    # and they survive to the recording run record + the per-run manifest
    fields = module.extra_record_fields(config)
    assert (fields["readout"], fields["input_size"]) == ("cls_patch", 384)
    path = tmp_path / "embedding.yaml"
    embedding._write_embedding_yaml(path, job, output_dim=1536, status="complete")
    data = _yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    assert (data["readout"], data["input_size"]) == ("cls_patch", 384)


def test_resolve_renderer_spec_populates_registry_on_a_miss(tmp_path):
    """A caller may resolve the spec before the model modules are imported (empty
    EMBEDDING_MODELS). Silently dropping the resize step then breaks the batch collate for
    a variable-size constructor (full_frame) — so a miss must re-import to recover the
    model's input_size. Regression: a 336 px model + a variable-size recipe produced un-resized frames.

    The cold state is a fresh interpreter, not this one with ``detail.models.*`` deleted from
    ``sys.modules``: re-importing a submodule rebinds it on its parent package, and restoring
    ``sys.modules`` afterwards does not undo that, so every later string-target patch in the
    worker (``"detail.models.sam3.is_importable"``) landed on a copy the code under test never
    reads. That failed two SAM 3 tests in 1 of 7 full runs on a box without the sam3 backend.
    """
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    code = textwrap.dedent("""
        import sys
        from detail.core.registries import EMBEDDING_MODELS
        from detail.pipeline import embedding

        assert not EMBEDDING_MODELS, "not a cold start: the models are already registered"
        spec = embedding.resolve_renderer_spec(
            sys.argv[1],
            "track_01",
            [0],
            "full_frame",  # variable-size -> must end in a resize to the model input
            model_id="dinov3_vit",
        )
        assert spec[-1] == {"type": "resize", "size": 224}, spec  # DINOv3 ViT's input
        assert "dinov3_vit" in EMBEDDING_MODELS  # the miss re-imported the models
        """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(recording_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# End-to-end runner
# --------------------------------------------------------------------------- #
def test_embedding_runner(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0, -1, 0])  # track 0 = rows 0,1,3
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    with _registered_embedder():
        embedding.run(_job(), run_dir, progress_cb=lambda e: None, stop_event=None)

    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["embedding"].shape == (3, D)
    assert emb["det_row"].tolist() == [0, 1, 3]
    assert emb["frame_idx"].tolist() == [0, 1, 3]
    assert emb["t_ns"].tolist() == [0, 1000, 3000]

    # §14 `kinematics/`: written beside the embedding, same rows, same order. Asserted here
    # rather than in its own test because "the sibling exists" and "the sibling is aligned with
    # the embedding it describes" are the same property, and only a real run shows it.
    kin = PartZarrReader(
        kinematics_part_path(run_dir / "track_0" / "part_001.zarr")
    ).read()
    assert kin["frame_idx"].tolist() == emb["frame_idx"].tolist()
    assert kin["t_ns"].tolist() == emb["t_ns"].tolist()
    # The fixture's mask for frame f is a 4x6 block at rows f%12..+4, cols 2..8 of a 16x20
    # frame, so the geometry is hand-computable — assert the value, not that a column exists.
    assert kin["mask_area_norm"].tolist() == [pytest.approx(24 / 320)] * 3
    assert kin["cx_norm"].tolist() == [pytest.approx(5.0 / 20)] * 3
    assert kin["cy_norm"].tolist() == [
        pytest.approx((f % 12 + 2.0) / 16) for f in (0, 1, 3)
    ]
    # Every detection row in this fixture carries the same box, so the union is that box.
    assert kin["bbox_norm"][0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])

    meta = yaml.safe_load((run_dir / "embedding.yaml").read_text(encoding="utf-8-sig"))
    assert meta["id"] == "emb_01"
    assert meta["model_id"] == "fake_embedder"
    assert meta["output_dim"] == D
    assert meta["track_ids"] == [0]
    assert meta["renderer_spec"] == RENDERER
    assert meta["model_variant"] is None  # provenance fields always written
    assert meta["constructor"] is None
    assert meta["status"] == "complete"


def test_embedding_unknown_model_raises(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    job = _job()
    job.model_id = "nope"
    with pytest.raises(ValueError, match="unknown model_id"):
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)


def test_inline_streams_units_with_loading_message(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 1, 0, 1])  # tracks 0 and 1
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    job = _job()
    job.track_ids = [0, 1]  # two (track, part) units
    events = []
    with _registered_embedder():
        embedding.run(job, run_dir, progress_cb=events.append, stop_event=None)
    assert events[0].message == "loading model"  # announced before the weight load
    assert all(e.units_total == 2 for e in events)  # real run-level denominator
    assert len(events) >= 2  # not just one event per unit at the end
    assert events[-1].units_done == 2  # both units completed


def test_progress_frames_total_counts_unique_frames_not_rows(tmp_path):
    """frames_total is the run-wide unique-frame count, not the detection-row
    count: a track with a secondary on a frame (two rows, one frame) OR-merges
    to one image, so it must not inflate the bar's denominator."""
    # track 0 on frames [0, 0, 1]: 3 rows, 2 unique frames (a secondary on f0).
    recording_dir = _setup(tmp_path, object_ids=[0, 0, 0], frames=[0, 0, 1])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    events = []
    with _registered_embedder():
        embedding.run(_job(), run_dir, progress_cb=events.append, stop_event=None)
    assert all(e.frames_total == 2 for e in events)  # unique frames, not 3 rows
    assert events[-1].frames_done == 2  # both unique frames embedded
    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["frame_idx"].tolist() == [0, 1]  # one embedding row per frame


def test_aggregate_event_populates_eta_on_midrun_frame_progress():
    """A mid-run embedding event (frames_done strictly between 0 and frames_total)
    carries a populated eta_s: the coordinator computes it via remaining_seconds, so
    every surface can render an ETA even on a single (track, part) unit (§6 cadence).
    The inline path streams these events through this builder while a unit is in flight
    (test_inline_streams_units_with_loading_message); this pins the ETA the builder
    attaches, deterministically."""
    start = time.monotonic() - 1.0  # ~1s elapsed so the ETA is a real number
    event = embedding._aggregate_event_for_track(
        track_id=0,
        results=[],  # no completed units yet — the in-flight unit is the only progress
        parts_total={0: 1},
        units_total=1,
        frames_total=4,
        start=start,
        live_frames=1,  # 1 of 4 frames rendered -> genuinely mid-run
    )
    assert 0 < event.frames_done < event.frames_total
    assert event.eta_s is not None and event.eta_s > 0


# --------------------------------------------------------------------------- #
# Batching edge cases (end-to-end via the runner)
# --------------------------------------------------------------------------- #
def test_batch_size_larger_than_rows(tmp_path):
    """3 rows, batch_size=10 — one batch, all rows written."""
    recording_dir = _setup(tmp_path, object_ids=[0, 0, 0])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    with _registered_embedder():
        embedding.run(
            _job(batch_size=10), run_dir, progress_cb=lambda e: None, stop_event=None
        )

    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["embedding"].shape == (3, D)
    assert emb["det_row"].tolist() == [0, 1, 2]


def test_batch_size_one(tmp_path):
    """batch_size=1 — degenerate per-frame batching, output still correct."""
    recording_dir = _setup(tmp_path, object_ids=[0, 0, 0, 0])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    with _registered_embedder():
        embedding.run(
            _job(batch_size=1), run_dir, progress_cb=lambda e: None, stop_event=None
        )

    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["embedding"].shape == (4, D)
    assert emb["det_row"].tolist() == [0, 1, 2, 3]


def test_batch_size_aligned_to_rows(tmp_path):
    """batch_size=2 with 6 rows — exactly 3 batches, no remainder."""
    recording_dir = _setup(tmp_path, object_ids=[0] * 6)
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    with _registered_embedder():
        embedding.run(
            _job(batch_size=2), run_dir, progress_cb=lambda e: None, stop_event=None
        )

    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["embedding"].shape == (6, D)
    assert emb["det_row"].tolist() == list(range(6))


def test_batch_with_remainder(tmp_path):
    """batch_size=2 with 5 rows — 2 full batches + 1 row."""
    recording_dir = _setup(tmp_path, object_ids=[0] * 5)
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    with _registered_embedder():
        embedding.run(
            _job(batch_size=2), run_dir, progress_cb=lambda e: None, stop_event=None
        )

    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["embedding"].shape == (5, D)
    assert emb["frame_idx"].tolist() == [0, 1, 2, 3, 4]


def test_empty_track_writes_empty_zarr(tmp_path):
    """No rows assigned to this track — runner writes an empty embedding zarr,
    no embed_batch call (so the loader stays cold)."""
    recording_dir = _setup(tmp_path, object_ids=[-1, -1, -1])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    calls = []

    class _CountingEmbedder(_FakeEmbedder):
        def embed_batch(self, images):
            calls.append(len(list(images)))
            return super().embed_batch(images)

    register_embedding_model("counting_embedder")(_CountingEmbedder)
    try:
        job = _job()
        job.model_id = "counting_embedder"
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    finally:
        EMBEDDING_MODELS.pop("counting_embedder", None)

    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["embedding"].shape == (0, D)
    assert emb["det_row"].tolist() == []
    assert calls == []  # zero embed_batch invocations on an empty track


def test_sparse_track_skips_gap_frames(tmp_path):
    """Track present at every other frame — dataset still reads sequentially,
    discarding the unwanted frames in between (Tenet 11)."""
    recording_dir = _setup(tmp_path, object_ids=[0, -1, 0, -1, 0])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    with _registered_embedder():
        embedding.run(_job(), run_dir, progress_cb=lambda e: None, stop_event=None)

    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["det_row"].tolist() == [0, 2, 4]
    assert emb["frame_idx"].tolist() == [0, 2, 4]


def test_per_frame_vs_batched_agree(tmp_path):
    """batch_size=1 and batch_size=N produce the same embedding values."""
    recording_dir = _setup(tmp_path, object_ids=[0, 0, 0, 0])

    # batch_size=1
    run_dir1 = recording_dir / "embeddings" / "emb_per_frame"
    run_dir1.mkdir(parents=True)
    job1 = _job(batch_size=1)
    job1.embedding_run_id = "emb_per_frame"
    with _registered_embedder():
        embedding.run(job1, run_dir1, progress_cb=lambda e: None, stop_event=None)
    per_frame = PartZarrReader(run_dir1 / "track_0" / "part_001.zarr").read()

    # batch_size=4
    run_dir4 = recording_dir / "embeddings" / "emb_batched"
    run_dir4.mkdir(parents=True)
    job4 = _job(batch_size=4)
    job4.embedding_run_id = "emb_batched"
    with _registered_embedder():
        embedding.run(job4, run_dir4, progress_cb=lambda e: None, stop_event=None)
    batched = PartZarrReader(run_dir4 / "track_0" / "part_001.zarr").read()

    np.testing.assert_array_equal(per_frame["embedding"], batched["embedding"])
    assert per_frame["det_row"].tolist() == batched["det_row"].tolist()


# --------------------------------------------------------------------------- #
# DetectionMaskDataset — direct tests
# --------------------------------------------------------------------------- #
def _binary_renderer():
    return MaskRenderer.from_spec(
        [
            {"type": "init_binary_mask"},
            {"type": "pad_to_square"},
            {"type": "resize", "size": 32},
        ]
    )


def _masked_color_renderer():
    # apply_mask reads ctx.frame -> needs_frame True -> takes the decode path.
    return MaskRenderer.from_spec(
        [{"type": "apply_mask", "background": 0}, {"type": "resize", "size": 32}]
    )


def test_dataset_iter_yields_in_frame_order(tmp_path):
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=5)
    rles = [_mask_counts(f) for f in range(3)]
    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([0, 2, 4]),
        t_ns=np.array([0, 2000, 4000]),
        det_rows=np.array([0, 1, 2]),
        mask_rles_per_frame=[[r] for r in rles],
        mask_size=(H, W),
        renderer=_binary_renderer(),
        track_id=0,
    )

    items = list(dataset)
    assert [it["frame_idx"] for it in items] == [0, 2, 4]
    assert [it["t_ns"] for it in items] == [0, 2000, 4000]
    assert [it["det_row"] for it in items] == [0, 1, 2]
    assert all(it["image"].shape == (32, 32, 3) for it in items)


def test_dataset_threads_bbox_into_context(tmp_path):
    """The per-frame bbox reaches ``ctx.bbox`` (was None before) and is reordered
    with the frames — so crop_to_bbox crops the subject (§14)."""
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=5)
    seen: list = []

    class _SpyRenderer:
        def render(self, ctx):
            seen.append(ctx.bbox)
            return np.zeros((4, 4, 3), np.uint8)

    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([4, 0, 2]),  # unsorted → reordered to 0,2,4
        t_ns=np.array([4000, 0, 2000]),
        det_rows=np.array([2, 0, 1]),
        mask_rles_per_frame=[[_mask_counts(f)] for f in (4, 0, 2)],
        bboxes=[(0.4, 0.4, 0.5, 0.5), (0.0, 0.0, 0.1, 0.1), (0.2, 0.2, 0.3, 0.3)],
        mask_size=(H, W),
        renderer=_SpyRenderer(),
        track_id=0,
    )
    list(dataset)
    # bboxes follow the frame sort (0, 2, 4)
    assert seen == [
        (0.0, 0.0, 0.1, 0.1),
        (0.2, 0.2, 0.3, 0.3),
        (0.4, 0.4, 0.5, 0.5),
    ]


def test_dataset_bbox_defaults_to_none(tmp_path):
    """No bboxes → ctx.bbox is None per frame (back-compat for callers/old specs)."""
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=3)
    seen: list = []

    class _SpyRenderer:
        def render(self, ctx):
            seen.append(ctx.bbox)
            return np.zeros((4, 4, 3), np.uint8)

    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([0, 1]),
        t_ns=np.array([0, 1000]),
        det_rows=np.array([0, 1]),
        mask_rles_per_frame=[[_mask_counts(0)], [_mask_counts(1)]],
        mask_size=(H, W),
        renderer=_SpyRenderer(),
        track_id=0,
    )
    list(dataset)
    assert seen == [None, None]


def test_dataset_silhouette_skips_video_decode(tmp_path, monkeypatch):
    """A mask-only renderer (needs_frame False) opens no cv2.VideoCapture — decoding
    a video whose frames it never touches is pure waste (§14). Proven by exploding
    VideoCapture."""

    def _boom(*a, **k):
        raise AssertionError("cv2.VideoCapture opened for a mask-only renderer")

    monkeypatch.setattr(embedding.cv2, "VideoCapture", _boom)
    dataset = embedding.DetectionMaskDataset(
        video_path=tmp_path / "does_not_exist.mp4",  # never opened
        frame_indices=np.array([0, 2, 4]),
        t_ns=np.array([0, 2000, 4000]),
        det_rows=np.array([0, 1, 2]),
        mask_rles_per_frame=[[_mask_counts(f)] for f in (0, 2, 4)],
        mask_size=(H, W),
        renderer=_binary_renderer(),
        track_id=0,
    )
    items = list(dataset)
    assert [it["frame_idx"] for it in items] == [0, 2, 4]
    assert all(it["image"].shape == (32, 32, 3) for it in items)


def test_dataset_masked_color_opens_video(tmp_path, monkeypatch):
    """A pixel-reading renderer (apply_mask) still opens the video (§14)."""
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=3)
    opened = {"n": 0}
    real_cls = cv2.VideoCapture

    class _CountCap:
        def __init__(self, path):
            opened["n"] += 1
            self._c = real_cls(path)

        def read(self):
            return self._c.read()

        def release(self):
            return self._c.release()

    monkeypatch.setattr(embedding.cv2, "VideoCapture", _CountCap)
    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([0, 1]),
        t_ns=np.array([0, 1000]),
        det_rows=np.array([0, 1]),
        mask_rles_per_frame=[[_mask_counts(0)], [_mask_counts(1)]],
        mask_size=(H, W),
        renderer=_masked_color_renderer(),
        track_id=0,
    )
    items = list(dataset)
    assert opened["n"] == 1  # video opened for the pixel-reading renderer
    assert len(items) == 2


def test_dataset_sorts_unsorted_input(tmp_path):
    """Caller may pass rows in any order; the dataset must sort by frame_idx
    so sequential ``cap.read()`` stays monotonic."""
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=5)
    rles = [_mask_counts(f) for f in [4, 0, 2]]
    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([4, 0, 2]),
        t_ns=np.array([4000, 0, 2000]),
        det_rows=np.array([2, 0, 1]),
        mask_rles_per_frame=[[r] for r in rles],
        mask_size=(H, W),
        renderer=_binary_renderer(),
        track_id=0,
    )

    items = list(dataset)
    assert [it["frame_idx"] for it in items] == [0, 2, 4]
    assert [it["det_row"] for it in items] == [0, 1, 2]


def test_dataset_empty_yields_nothing(tmp_path):
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=3)
    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([], dtype=np.int64),
        t_ns=np.array([], dtype=np.int64),
        det_rows=np.array([], dtype=np.int64),
        mask_rles_per_frame=[],
        mask_size=(H, W),
        renderer=_binary_renderer(),
        track_id=0,
    )
    assert list(dataset) == []
    assert len(dataset) == 0


def test_dataset_raises_when_video_ends_early(tmp_path):
    """Asking for a frame past the end of the video must raise — silent
    fallback (zeros, garbage, last frame) would corrupt embeddings."""
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=3)
    rles = [_mask_counts(0)]
    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([10]),
        t_ns=np.array([0]),
        det_rows=np.array([0]),
        mask_rles_per_frame=[[r] for r in rles],
        mask_size=(H, W),
        renderer=_masked_color_renderer(),  # decode path checks video length
        track_id=0,
    )
    with pytest.raises(RuntimeError, match="ended"):
        list(dataset)


def test_dataset_uses_real_frame_in_context(tmp_path):
    """The synthetic video frames are uniform colors. A renderer step that
    looks at ctx.frame should see those bytes (forward-compat for the future
    apply_mask transform). We verify by spying on the renderer."""
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=3)

    seen_frames: list[np.ndarray] = []

    class _SpyRenderer:
        def render(self, ctx: RenderContext):
            seen_frames.append(ctx.frame.copy())
            return np.zeros((4, 4, 3), dtype=np.uint8)

    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([0, 1]),
        t_ns=np.array([0, 1000]),
        det_rows=np.array([0, 1]),
        mask_rles_per_frame=[[_mask_counts(0)], [_mask_counts(1)]],
        mask_size=(H, W),
        renderer=_SpyRenderer(),
        track_id=0,
    )

    list(dataset)
    assert len(seen_frames) == 2
    for frame in seen_frames:
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8


class _SpyEmbedder(_FakeEmbedder):
    """Keeps every image it is handed, exactly as the model would receive it."""

    model_id = "spy_embedder"
    seen: list = []

    def embed_batch(self, images):
        type(self).seen.extend(np.array(img, copy=True) for img in images)
        return super().embed_batch(images)


def test_a_red_subject_reaches_the_model_in_channel_0(tmp_path):
    """The channel-order guard (docs/recorded_classes.md, "a frame handed across a boundary").

    cv2 decodes BGR and every embedding model reads RGB, so the frame is converted once where it
    is read. Before that, a red animal reached DINOv3 as a blue one and the "max-contrast" blue
    background reached it as red — every run, nothing failing, because every image is a valid
    image. This drives a pure-red subject through the real path: the background sampled from
    the video, the default constructor, ``DetectionMaskDataset`` and ``embedding.run``, to a
    model that keeps what it receives. Red must be channel 0, and the background must arrive as
    the colour that was computed for it.
    """
    from dataclasses import replace

    from detail.transforms.constructors import CONSTRUCTORS, DEFAULT_CONSTRUCTOR

    red_bgr = (0, 0, 255)  # the order cv2 writes
    recording_dir = _setup(
        tmp_path, object_ids=[0, 0], color=red_bgr, boxes=[(0.0, 0.0, 0.45, 0.35)] * 2
    )
    # The background is sampled the way a real run samples it: frames read by timestamp.
    from detail.storage import recording as recording_store

    csv = "video/timestamps/generated/part_001.csv"
    (recording_dir / csv).parent.mkdir(parents=True)
    recording_store.write_timestamps_csv(
        recording_dir / csv, np.array([0, 1000], dtype=np.int64)
    )
    data = recording_store.read_recording(recording_dir)
    data["video"]["files"][0]["timestamps"] = {
        "generated": csv,
        "user": None,
        "active": "generated",
    }
    recording_store.write_recording(recording_dir, data)
    mean = embedding._mean_mask_color_rgb(recording_dir, "track_01", [0], 30, 0)
    assert mean[0] > 200 and mean[2] < 60, f"the subject's mean is not red: {mean}"
    bg = embedding._max_contrast_rgb(mean)
    # No resize step: its interpolation would blend the subject's edge into the background.
    spec = CONSTRUCTORS[DEFAULT_CONSTRUCTOR].build(
        canvas_px=16, resize=None, bg_color=bg
    )
    job = replace(_job(), model_id="spy_embedder", renderer_spec=spec)
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    _SpyEmbedder.seen = []
    register_embedding_model("spy_embedder")(_SpyEmbedder)
    try:
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    finally:
        EMBEDDING_MODELS.pop("spy_embedder", None)
    assert _SpyEmbedder.seen, "the model was handed nothing"
    image = _SpyEmbedder.seen[0]
    is_bg = (image == np.array(bg, dtype=image.dtype)).all(axis=-1)
    assert is_bg.any(), f"the background {bg} never reached the model as computed"
    subject = image[~is_bg].astype(int)
    assert len(subject), "no subject pixel reached the model"
    red = subject[:, 0].mean()
    blue = subject[:, 2].mean()
    assert red > 200 and blue < 60, (
        f"a red subject reached the model as (ch0={red:.0f}, ch2={blue:.0f}): "
        "the frame is not RGB where the model reads it"
    )


def test_dataset_merges_multiple_masks_per_frame(tmp_path):
    """A track with two detections in one frame (anchor + secondary) yields a
    single image whose mask is the OR of both — full-silhouette recovery."""
    video = tmp_path / "v.mp4"
    _write_test_video(video, n_frames=3)

    seen_masks: list[np.ndarray] = []

    class _SpyRenderer:
        def render(self, ctx: RenderContext):
            seen_masks.append(ctx.mask.copy())
            return np.zeros((4, 4, 3), dtype=np.uint8)

    rle_a, rle_b = _mask_counts(0), _mask_counts(6)  # disjoint blocks (rows 0:4, 6:10)
    dataset = embedding.DetectionMaskDataset(
        video_path=video,
        frame_indices=np.array([1]),
        t_ns=np.array([1000]),
        det_rows=np.array([0]),
        mask_rles_per_frame=[[rle_a, rle_b]],
        mask_size=(H, W),
        renderer=_SpyRenderer(),
        track_id=0,
    )

    items = list(dataset)
    assert len(items) == 1  # one image per frame, not one per detection
    expected = decode_mask(rle_a, (H, W)) | decode_mask(rle_b, (H, W))
    assert np.array_equal(seen_masks[0], expected)
    # disjoint blocks -> merged area is the sum of the two
    assert int(seen_masks[0].sum()) == int(
        decode_mask(rle_a, (H, W)).sum() + decode_mask(rle_b, (H, W)).sum()
    )


# --------------------------------------------------------------------------- #
# Multi-GPU coordinator — tests with a synchronous fake pool
# --------------------------------------------------------------------------- #
# These tests stub ProcessPoolExecutor with a synchronous fake so we exercise
# the round-robin assignment, progress aggregation, stop semantics, and
# failure propagation without spawning real subprocesses or needing CUDA.

from concurrent.futures import Future

from detail.pipeline import _embedding_worker as _ew


def _completed_future(result):
    """Real concurrent.futures.Future, pre-marked done with a result.
    Needed so concurrent.futures.wait() sees the right internals."""
    fut = Future()
    fut.set_result(result)
    return fut


class _SyncPool:
    """Fake ProcessPoolExecutor that runs submitted callables synchronously
    in the test process. Bypasses the real initializer's CUDA pinning and
    populates worker state directly."""

    def __init__(self, max_workers=1, mp_context=None, initializer=None, initargs=()):
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs
        if initializer is not None:
            (
                gpu_idx,
                model_id,
                model_variant,
                renderer_spec,
                _runners,
                readout,
                input_size,
                share,
            ) = initargs
            model_cls = EMBEDDING_MODELS[model_id]
            model = construct_embedding_model(
                model_cls, model_variant, readout=readout, input_size=input_size
            )
            model.load("cpu")
            _ew._WORKER_STATE["model"] = model
            _ew._WORKER_STATE["renderer"] = MaskRenderer.from_spec(renderer_spec)
            _ew._WORKER_STATE["output_dim"] = int(model.output_dim)
            _ew._WORKER_STATE["gpu_idx"] = gpu_idx
            _ew._WORKER_STATE["share"] = share

    def submit(self, fn, *args, **kwargs):
        return _completed_future(fn(*args, **kwargs))

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _patch_pool(monkeypatch):
    """Swap ProcessPoolExecutor + the spawn mp context so the coordinator runs
    entirely in-process."""
    monkeypatch.setattr(embedding, "ProcessPoolExecutor", _SyncPool)

    class _NoopCtx:
        pass

    import multiprocessing as mp_mod

    monkeypatch.setattr(mp_mod, "get_context", lambda name: _NoopCtx())


def _setup_multi_part(tmp_path, parts_object_ids: list[list[int]]):
    """Multi-part variant of _setup. ``parts_object_ids[i]`` is the per-row
    object_id list for part i."""
    recording_dir = tmp_path / "recordings" / "recording_001"
    recording_dir.mkdir(parents=True)

    files = []
    for i, oids in enumerate(parts_object_ids, start=1):
        part_name = f"part_{i:03d}.mp4"
        _write_test_video(recording_dir / "video" / part_name, n_frames=len(oids))
        files.append({"index": i - 1, "file": f"video/{part_name}", "timestamps": {}})

    (recording_dir / "recording.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "recording_001",
                "video": {"files": files},
                "detection_runs": [{"id": "det_01", "downsampling_run_id": None}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    det_dir = recording_dir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    for i, oids in enumerate(parts_object_ids, start=1):
        with DetectionPartWriter(det_dir / f"part_{i:03d}.zarr") as w:
            for f in range(len(oids)):
                w.add_row(
                    f, f * 1000, 0, "mouse", 0.9, (0.1, 0.2, 0.3, 0.4), _mask_counts(f)
                )
    (det_dir / "job.json").write_text(
        json.dumps(
            {"schema_version": 1, "status": "complete", "mask_resolution": [H, W]}
        ),
        encoding="utf-8",
    )

    track_dir = recording_dir / "tracking" / "track_01"
    (track_dir / "assignments").mkdir(parents=True)
    for i, oids in enumerate(parts_object_ids, start=1):
        with AssignmentPartWriter(
            track_dir / "assignments" / f"part_{i:03d}.zarr"
        ) as w:
            for oid in oids:
                w.add_object_id(oid)
    (track_dir / "track.yaml").write_text(
        yaml.dump(
            {"schema_version": 1, "id": "track_01", "detection_run_id": "det_01"}
        ),
        encoding="utf-8",
    )
    return recording_dir


def _multi_gpu_job(device_indices, *, track_ids=(0,), batch_size=32):
    return EmbeddingJob(
        job_id="emb_01",
        recording_id="recording_001",
        embedding_run_id="emb_01",
        tracking_run_id="track_01",
        track_ids=list(track_ids),
        model_id="fake_embedder",
        renderer_spec=RENDERER,
        batch_size=batch_size,
        num_workers=0,
        compute=ComputeTarget("cuda", list(device_indices), None, None),
        created_at="2026-01-01T00:00:00Z",
    )


def test_multi_gpu_short_circuits_to_inline_for_one_device(tmp_path, monkeypatch):
    """device_indices=[0] → inline path; pool never instantiated."""
    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    pool_constructed = []

    class _Sentinel(_SyncPool):
        def __init__(self, *a, **kw):
            pool_constructed.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(embedding, "ProcessPoolExecutor", _Sentinel)
    with _registered_embedder():
        embedding.run(
            _multi_gpu_job([0]), run_dir, progress_cb=lambda e: None, stop_event=None
        )
    assert pool_constructed == []


def test_multi_gpu_cpu_backend_short_circuits_to_inline(tmp_path, monkeypatch):
    recording_dir = _setup_multi_part(tmp_path, [[0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    pool_constructed = []

    class _Sentinel(_SyncPool):
        def __init__(self, *a, **kw):
            pool_constructed.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(embedding, "ProcessPoolExecutor", _Sentinel)
    with _registered_embedder():
        embedding.run(_job(), run_dir, progress_cb=lambda e: None, stop_event=None)
    assert pool_constructed == []


def test_multi_gpu_constructs_one_pool_per_gpu(tmp_path, monkeypatch):
    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    constructed: list[int] = []
    shares: list[int] = []

    class _Recorder(_SyncPool):
        def __init__(self, *a, **kw):
            constructed.append(kw["initargs"][0])
            shares.append(kw["initargs"][-1])
            super().__init__(*a, **kw)

    monkeypatch.setattr(embedding, "ProcessPoolExecutor", _Recorder)

    class _NoopCtx:
        pass

    import multiprocessing as mp_mod

    monkeypatch.setattr(mp_mod, "get_context", lambda name: _NoopCtx())

    with _registered_embedder():
        embedding.run(
            _multi_gpu_job([0, 1]),
            run_dir,
            progress_cb=lambda e: None,
            stop_event=None,
        )
    assert constructed == [0, 1]  # one pool per GPU, in order
    assert shares == [2, 2]  # each worker's render pool takes its part of the cores


def test_multi_gpu_round_robin_unit_dispatch(tmp_path, monkeypatch):
    """4 units (1 track × 4 parts) across 2 GPUs → 2 units per pool, round-robin."""
    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0], [0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)

    dispatched: dict[int, list[int]] = {0: [], 1: []}

    class _Recording(_SyncPool):
        def __init__(self, *a, **kw):
            self._gpu = kw["initargs"][0]
            super().__init__(*a, **kw)

        def submit(self, fn, *args, **kwargs):
            dispatched[self._gpu].append(args[0]["part_index"])
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(embedding, "ProcessPoolExecutor", _Recording)

    class _NoopCtx:
        pass

    import multiprocessing as mp_mod

    monkeypatch.setattr(mp_mod, "get_context", lambda name: _NoopCtx())

    with _registered_embedder():
        embedding.run(
            _multi_gpu_job([0, 1]),
            run_dir,
            progress_cb=lambda e: None,
            stop_event=None,
        )
    # unit i → pool i % 2: parts 0,2 go to GPU 0; parts 1,3 to GPU 1.
    assert dispatched[0] == [0, 2]
    assert dispatched[1] == [1, 3]


def test_multi_gpu_writes_per_unit_zarrs(tmp_path, monkeypatch):
    recording_dir = _setup_multi_part(tmp_path, [[0, 0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)

    with _registered_embedder():
        embedding.run(
            _multi_gpu_job([0, 1]),
            run_dir,
            progress_cb=lambda e: None,
            stop_event=None,
        )

    emb1 = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    emb2 = PartZarrReader(run_dir / "track_0" / "part_002.zarr").read()
    assert emb1["embedding"].shape == (3, D)
    assert emb2["embedding"].shape == (2, D)
    assert emb1["det_row"].tolist() == [0, 1, 2]
    assert emb2["det_row"].tolist() == [0, 1]

    # Kinematics is per part too, each aligned to *its own* part's rows.
    #
    # The two `det_row` lists above are why this matters: both restart at 0, so a detection is
    # identified by `(track, part, det_row)` and the part is only implied by which file the row
    # is in. Anything that concatenates parts and then indexes by `det_row` resolves one part's
    # index against another's — silently, with a plausible answer. Capturing geometry during the
    # embed pass avoids that lookup existing at all; this asserts the shape that keeps it so, and
    # fails if the sibling ever becomes one flat file or drifts out of alignment.
    kin1 = PartZarrReader(
        kinematics_part_path(run_dir / "track_0" / "part_001.zarr")
    ).read()
    kin2 = PartZarrReader(
        kinematics_part_path(run_dir / "track_0" / "part_002.zarr")
    ).read()
    assert kin1["frame_idx"].tolist() == emb1["frame_idx"].tolist()
    assert kin2["frame_idx"].tolist() == emb2["frame_idx"].tolist()
    assert len(kin1["cx_norm"]) == 3 and len(kin2["cx_norm"]) == 2


def test_multi_gpu_progress_aggregated_per_track(tmp_path, monkeypatch):
    """Progress events scoped per-track; parts_done climbs as units complete."""
    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)
    events = []
    with _registered_embedder():
        embedding.run(
            _multi_gpu_job([0, 1]),
            run_dir,
            progress_cb=lambda e: events.append(e),
            stop_event=None,
        )
    assert all(e.track_id == 0 for e in events)
    assert all(e.parts_total == 2 for e in events)
    # final event for track 0 has parts_done=2, frames_done=4
    final = events[-1]
    assert final.parts_done == 2
    assert final.frames_done == 4


def test_multi_gpu_unknown_model_raises(tmp_path, monkeypatch):
    """Coordinator validates the model exists in this process before spawning."""
    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)
    job = _multi_gpu_job([0, 1])
    job.model_id = "nope_embedder"
    with pytest.raises(ValueError, match="unknown model_id"):
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)


def test_multi_gpu_worker_failure_raises_run_failed(tmp_path, monkeypatch):
    """A failed unit makes the coordinator raise after aggregation."""
    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)

    def fake_pool_run_one(args):
        return {
            "track_id": int(args["track_id"]),
            "part_index": int(args["part_index"]),
            "part_id": args["part_id"],
            "gpu": 0,
            "status": "failed",
            "frames_done": 0,
            "frames_total": 0,
            "output_dim": D,
            "error": "simulated worker failure",
        }

    monkeypatch.setattr(_ew, "pool_run_one", fake_pool_run_one)
    with _registered_embedder():
        with pytest.raises(RuntimeError, match="embedding failed"):
            embedding.run(
                _multi_gpu_job([0, 1]),
                run_dir,
                progress_cb=lambda e: None,
                stop_event=None,
            )


def test_multi_gpu_stop_event_marks_run_failed(tmp_path, monkeypatch):
    """Stop event → coordinator touches stop.flag; worker honors it and the
    run escalates to RuntimeError via _raise_if_any_failed."""
    recording_dir = _setup_multi_part(tmp_path, [[0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)

    class _StopEvent:
        def is_set(self):
            return True

    def fake_pool_run_one(args):
        # Mimic a worker that observed stop.flag at its first poll boundary.
        return {
            "track_id": int(args["track_id"]),
            "part_index": int(args["part_index"]),
            "part_id": args["part_id"],
            "gpu": 0,
            "status": "failed",
            "frames_done": 0,
            "frames_total": 0,
            "output_dim": D,
            "error": "stop.flag detected during embedding",
        }

    monkeypatch.setattr(_ew, "pool_run_one", fake_pool_run_one)
    with _registered_embedder():
        with pytest.raises(RuntimeError, match="embedding failed"):
            embedding.run(
                _multi_gpu_job([0, 1]),
                run_dir,
                progress_cb=lambda e: None,
                stop_event=_StopEvent(),
            )
    assert (run_dir / _ew.STOP_FLAG_NAME).exists()


def test_multi_gpu_unit_per_track_part(tmp_path, monkeypatch):
    """2 tracks × 2 parts = 4 units; results sorted by (track_id, part_index)."""
    recording_dir = _setup_multi_part(
        tmp_path, [[0, 1, 0], [0, 1, 0]]  # rows belonging to tracks 0 and 1
    )
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    _patch_pool(monkeypatch)

    with _registered_embedder():
        embedding.run(
            _multi_gpu_job([0, 1], track_ids=(0, 1)),
            run_dir,
            progress_cb=lambda e: None,
            stop_event=None,
        )

    # Both tracks × both parts → 4 output zarrs. Per part, object_ids=[0,1,0]
    # → track 0 = 2 rows (indices 0, 2), track 1 = 1 row (index 1).
    expected = {0: 2, 1: 1}
    for tid in (0, 1):
        for pidx in (1, 2):
            emb = PartZarrReader(
                run_dir / f"track_{tid}" / f"part_{pidx:03d}.zarr"
            ).read()
            assert emb["embedding"].shape == (expected[tid], D)


# --- real-subprocess integration (spawn pool, CPU-only fake model) -------- #
_FAKE_CUDA_EMBEDDER_SOURCE = '''
import numpy as np

from detail.core.protocols import EmbeddingModel
from detail.core.registries import register_embedding_model


@register_embedding_model("fake_cuda_embedder")
class FakeCudaEmbedder(EmbeddingModel):
    """Declares cuda so the coordinator picks the multi-GPU path; load/embed
    never actually touch torch.cuda — runs on CI without GPUs."""

    model_id = "fake_cuda_embedder"
    output_dim = 8
    supported_backends = ["cuda"]

    def load(self, device):
        self.device = device

    def is_ready(self):
        return True

    def embed(self, image):
        return np.full(8, float(np.asarray(image).mean()), dtype=np.float32)

    def embed_batch(self, images):
        return np.stack(
            [
                np.full(8, float(np.asarray(img).mean()), dtype=np.float32)
                for img in images
            ],
            axis=0,
        )
'''


def test_multi_gpu_real_spawn_pool(tmp_path, monkeypatch):
    """Spawns the actual ProcessPoolExecutor with device_indices=[0, 1] using
    a CPU-only fake. Verifies spawn-context plumbing + initializer + cross-
    process pickle work end-to-end on CI without real GPUs."""
    import importlib
    import os

    recording_dir = _setup_multi_part(tmp_path, [[0, 0, 0], [0, 0]])

    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    (mod_dir / "fake_cuda_embedder_mod.py").write_text(
        _FAKE_CUDA_EMBEDDER_SOURCE, encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(mod_dir))
    monkeypatch.setenv(
        "PYTHONPATH", str(mod_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

    from detail.api import _runner_modules

    monkeypatch.setattr(
        _runner_modules,
        "PIPELINE_RUNNERS",
        _runner_modules.PIPELINE_RUNNERS + ("fake_cuda_embedder_mod",),
    )
    importlib.import_module("fake_cuda_embedder_mod")

    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    job = EmbeddingJob(
        job_id="emb_01",
        recording_id="recording_001",
        embedding_run_id="emb_01",
        tracking_run_id="track_01",
        track_ids=[0],
        model_id="fake_cuda_embedder",
        renderer_spec=RENDERER,
        batch_size=32,
        num_workers=0,
        compute=ComputeTarget("cuda", [0, 1], None, None),
        created_at="2026-01-01T00:00:00Z",
    )
    try:
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    finally:
        EMBEDDING_MODELS.pop("fake_cuda_embedder", None)

    # Both per-(track, part) zarrs got populated by the spawn workers.
    e1 = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    e2 = PartZarrReader(run_dir / "track_0" / "part_002.zarr").read()
    assert e1["embedding"].shape == (3, 8)
    assert e2["embedding"].shape == (2, 8)
    meta = yaml.safe_load((run_dir / "embedding.yaml").read_text(encoding="utf-8-sig"))
    assert meta["status"] == "complete"
    assert meta["output_dim"] == 8


# --------------------------------------------------------------------------- #
# §14 Egocentric alignment — reads a keypoint track run
# --------------------------------------------------------------------------- #
SKELETON = {"names": ["nose", "tail"], "edges": [["nose", "tail"]], "pairs": []}


def _keypoint_runs(recording_dir, rows, *, tracking_run_id="track_01", cutoff=0.3):
    """A detect run record and a track run with one part of ``rows`` —
    ``[(frame_idx, t_ns, nose_xy, tail_xy)]`` — the way the stages write them since §26's
    2026-10-21 revision (the detect run on a detection run, the track run assigning to
    ``tracking_run_id``), plus the recording.yaml entries ``validate`` reads."""
    from detail.core.skeleton import Skeleton
    from detail.storage import keypoint_runs
    from detail.api import run_spec
    from detail.storage.keypoint_zarr import KeypointTrackPartWriter

    skeleton = Skeleton.from_dict(SKELETON)
    detect_dir = keypoint_runs.detect_run_dir(recording_dir, "kd_01")
    keypoint_runs.write_detect_yaml(
        detect_dir,
        {
            "schema_version": 1,
            "id": "kd_01",
            "created_at": "2026-01-01T00:00:00Z",
            "detection_run_id": "det_01",
            "min_pose_confidence": None,
            "model_id": "fake_kp",
            "skeleton": SKELETON,
            "skeleton_digest": skeleton.digest(),
            "n_candidates": 2,
            "every_n_frames": 1,
        },
    )
    track_dir = keypoint_runs.track_run_dir(recording_dir, "kt_01")
    keypoint_runs.write_track_yaml(
        track_dir,
        {
            "schema_version": 1,
            "id": "kt_01",
            "created_at": "2026-01-01T00:00:00Z",
            "detect_run_id": "kd_01",
            "tracking_run_id": tracking_run_id,
            "match": {"min_inside": 0.5, "mask_grow": 0.05},
            "method": "temporal_decode",
            "config": {"decode_cutoff": cutoff, "despike": True},
            "skeleton": SKELETON,
            "skeleton_digest": skeleton.digest(),
        },
    )
    tracks_path = recording_dir / "tracking" / "track_01" / "tracks.zarr"
    if (
        not tracks_path.exists()
    ):  # the base fixture writes no summary; validate reads it
        with TracksSummaryWriter(tracks_path) as summary:
            summary.add_track(0, "mouse", 0, 3, 3, 0, 0.9)
    part = keypoint_runs.track_part_path(track_dir, 0, "part_001")
    part.parent.mkdir(parents=True, exist_ok=True)
    if part.exists():  # the writer resumes an existing part; a second call rewrites it
        shutil.rmtree(part)
    with KeypointTrackPartWriter(part, ("nose", "tail")) as writer:
        points = np.array([[nose, tail] for _f, _t, nose, tail in rows], np.float32)
        writer.add_rows(
            np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.arange(len(rows), dtype=np.int32),
            points,
            np.full(points.shape[:2], 0.9, np.float32),
            np.zeros(points.shape[:2], np.int8),
        )
    data = yaml.safe_load(
        (recording_dir / "recording.yaml").read_text(encoding="utf-8-sig")
    )
    data["video"]["resolution"] = [W, H]
    data["video"]["fps_actual"] = 30.0
    data["detection_runs"] = [
        {"id": "det_01", "status": "complete", "downsampling_run_id": None}
    ]
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    ]
    data["keypoint_detect_runs"] = [
        {
            "id": "kd_01",
            "status": "complete",
            "model_id": "fake_kp",
            "detection_run_id": "det_01",
        }
    ]
    data["keypoint_track_runs"] = [
        {
            "id": "kt_01",
            "status": "complete",
            "detect_run_id": "kd_01",
            "tracking_run_id": tracking_run_id,
            "match": {"min_inside": 0.5, "mask_grow": 0.05},
            "method": "temporal_decode",
            "config": {"decode_cutoff": cutoff, "despike": True},
        }
    ]
    # the tracking run reaches the track run's identity as its spec, never its id
    data["keypoint_track_runs"][0]["tracking_identity"] = run_spec.spec_of(
        data, "tracking", data["tracking_runs"][0], recording_dir
    )
    (recording_dir / "recording.yaml").write_text(
        yaml.dump(data, sort_keys=False), encoding="utf-8"
    )


FACING_RIGHT = [
    (0, 0, (0.35, 0.3), (0.15, 0.3)),  # nose right of the tail: heading 0°
    (1, 1000, (0.35, 0.3), (0.15, 0.3)),
]
FACING_LEFT = [
    (0, 0, (0.15, 0.3), (0.35, 0.3)),
    (1, 1000, (0.15, 0.3), (0.35, 0.3)),
    (3, 3000, (0.15, 0.3), (0.35, 0.3)),
]


def _aligned_config(**overrides) -> dict:
    config = {
        "tracking_run_id": "track_01",
        "track_ids": [0],
        "all_tracks": False,
        "model_id": "fake_embedder",
        "model_variant": None,
        "constructor": "masked_color_centered",
        "batch_size": 8,
        "every_n_frames": None,
        "compute": ComputeTarget("cpu", None, None, None),
        "align": "rotate",
        "keypoint_run_id": "kt_01",
        "heading": "nose_to_tail",
        "max_gap_s": 0.5,
        "mirror_hold_s": 1.0,
        "facing_camera_ratio": 0.5,
    }
    config.update(overrides)
    return config


def _validate(recording_dir, config):
    from detail.storage import recording as recording_store

    module = embedding.EmbeddingModule()
    data = recording_store.read_recording(recording_dir)
    return module, module.validate(recording_dir, data, config)


@pytest.fixture
def aligned_recording(tmp_path, monkeypatch):
    from detail.storage import heading_definitions

    recording_dir = _setup(tmp_path, object_ids=[0, 0, -1, 0])  # track 0 = frames 0,1,3
    _keypoint_runs(recording_dir, FACING_RIGHT)  # a heading on frames 0 and 1 only
    heading_definitions.save_definition(tmp_path, "nose_to_tail", ["nose"], ["tail"])
    monkeypatch.setattr(
        "detail.storage.video_integrity.validate_source_videos", lambda *a, **k: []
    )
    return recording_dir


def test_an_aligned_run_turns_the_crop_records_the_heading_and_drops_frames_without_one(
    aligned_recording,
):
    recording_dir = aligned_recording
    with _registered_embedder():
        module, config = _validate(recording_dir, _aligned_config())
        types = [s["type"] for s in config["renderer_spec"]]
        assert types == [
            "crop_to_bbox",
            "apply_mask",
            "align_rotate",
            "center_in_square",
            "resize",
        ]
        assert config["heading_definition"] == {"front": ["nose"], "back": ["tail"]}
        assert config["heading_cutoff"] == 0.3  # the track run's decode cut-off
        assert config["keypoint_track_identity"]["module"] == "keypoint_track"
        assert (
            config["keypoint_track_identity"]["identity"]["method"] == "temporal_decode"
        )
        assert "kt_01" not in str(
            config["keypoint_track_identity"]
        )  # identity, never an id
        canvas = next(
            s for s in config["renderer_spec"] if s["type"] == "center_in_square"
        )
        assert canvas["size"] > 0  # sized from the turned masks
        job = module.build_job("recording_001", "emb_01", config)
        assert (
            job.align == "rotate"
            and job.heading_definition == config["heading_definition"]
        )
        run_dir = recording_dir / "embeddings" / "emb_01"
        run_dir.mkdir(parents=True)
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["frame_idx"].tolist() == [0, 1]  # frame 3: no heading within the gap
    assert emb["heading_deg"].tolist() == pytest.approx([0.0, 0.0])
    assert (
        emb["heading_source"].tolist() == [0, 0]
        and emb["heading_deg"].dtype == np.float32
    )
    assert emb["align_mirrored"].tolist() == [False, False]
    assert emb["embedding"].shape == (2, D)
    meta = yaml.safe_load((run_dir / "embedding.yaml").read_text(encoding="utf-8-sig"))
    assert meta["align"] == "rotate" and meta["keypoint_run_id"] == "kt_01"
    assert meta["heading_definition"] == {"front": ["nose"], "back": ["tail"]}
    assert meta["heading_coverage"]["track_0"]["part_001"] == {
        "frames": 3,
        "own": 2,
        "interpolated": 0,
        "dropped": 1,
        "longest_gap_s": 0.0,  # one dropped frame spans no time
    }
    record = module.extra_record_fields(config)
    assert record["align"] == "rotate" and record["keypoint_track_identity"] is not None
    assert set(module.identity_fields) <= set(record) | {"model_id"}


def test_an_unaligned_run_is_what_it_was_and_clears_every_alignment_field(
    aligned_recording,
):
    recording_dir = aligned_recording
    with _registered_embedder():
        module, config = _validate(
            recording_dir,
            _aligned_config(
                align="none", keypoint_run_id=None, heading=None, max_gap_s=9.0
            ),
        )
    assert "align_rotate" not in [s["type"] for s in config["renderer_spec"]]
    assert (
        config["keypoint_track_identity"] is None
        and config["heading_definition"] is None
    )
    assert (
        config["max_gap_s"] == 0.5
    )  # a setting that did nothing is not recorded as set
    job = module.build_job("recording_001", "emb_02", config)
    run_dir = recording_dir / "embeddings" / "emb_02"
    run_dir.mkdir(parents=True)
    with _registered_embedder():
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["frame_idx"].tolist() == [0, 1, 3] and "heading_deg" not in emb
    meta = yaml.safe_load((run_dir / "embedding.yaml").read_text(encoding="utf-8-sig"))
    assert meta["align"] == "none" and meta["heading_coverage"] == {}


def test_mirror_flips_a_subject_facing_left(aligned_recording):
    recording_dir = aligned_recording
    _keypoint_runs(recording_dir, FACING_LEFT)
    with _registered_embedder():
        module, config = _validate(recording_dir, _aligned_config(align="mirror"))
        assert "align_mirror" in [s["type"] for s in config["renderer_spec"]]
        job = module.build_job("recording_001", "emb_03", config)
        run_dir = recording_dir / "embeddings" / "emb_03"
        run_dir.mkdir(parents=True)
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    emb = PartZarrReader(run_dir / "track_0" / "part_001.zarr").read()
    assert emb["frame_idx"].tolist() == [0, 1, 3]
    assert emb["align_mirrored"].tolist() == [
        True,
        True,
        True,
    ]  # faced left from the start
    assert emb["heading_deg"].tolist() == pytest.approx([180.0, 180.0, 180.0])


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"keypoint_run_id": None}, "needs both a keypoint track run"),
        ({"heading": None}, "needs both"),
        ({"align": "none"}, "given with align 'none'"),
        ({"keypoint_run_id": "kt_99"}, "not found in recording"),
        ({"heading": "nope"}, "no heading definition named 'nope'"),
        ({"constructor": "full_frame"}, "does not crop to the subject"),
        ({"align": "sideways"}, "align"),
    ],
)
def test_alignment_validation_names_what_to_fix(aligned_recording, overrides, fragment):
    with _registered_embedder():
        with pytest.raises(ValueError, match=fragment):
            _validate(aligned_recording, _aligned_config(**overrides))


def test_alignment_refuses_a_keypoint_run_from_another_tracking_run_and_a_bad_skeleton(
    aligned_recording,
):
    from detail.storage import heading_definitions

    recording_dir = aligned_recording
    heading_definitions.save_definition(
        recording_dir.parent.parent, "hip_to_nose", ["nose"], ["hip"]
    )
    with _registered_embedder():
        with pytest.raises(ValueError, match="skeleton lacks: \\['hip'\\]"):
            _validate(recording_dir, _aligned_config(heading="hip_to_nose"))
        _keypoint_runs(recording_dir, FACING_RIGHT, tracking_run_id="track_99")
        with pytest.raises(
            ValueError, match="assigned its poses to tracking run 'track_99'"
        ):
            _validate(recording_dir, _aligned_config())


def test_an_explicit_renderer_spec_must_carry_the_step_align_names(aligned_recording):
    with _registered_embedder():
        with pytest.raises(ValueError, match="must carry exactly the step"):
            _validate(
                aligned_recording,
                _aligned_config(renderer_spec=list(RENDERER), constructor=None),
            )


def _counts(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _stroke(length: int, x0: int = 0) -> np.ndarray:
    """A one-pixel diagonal line: its box is square, its turned box at 45° is a line."""
    m = np.zeros((H, W), bool)
    for i in range(length):
        m[i, x0 + i] = True
    return m


def _heading_rows(degrees) -> list:
    """Keypoint rows whose nose-to-tail axis points along each angle, all one length."""
    rows = []
    for f, deg in enumerate(degrees):
        dx, dy = 3 * np.cos(np.radians(deg)), 3 * np.sin(np.radians(deg))
        nose = ((W / 2 + dx) / W, (H / 2 + dy) / H)
        tail = ((W / 2 - dx) / W, (H / 2 - dy) / H)
        rows.append((f, f * 1000, nose, tail))
    return rows


def _random_aligned_recording(tmp_path, seed: int):
    """Masks and headings that both vary per frame — L shapes, diagonal strokes, frames with
    two rows, rows with no mask or an empty one — so the frame with the largest box is not,
    in general, the frame whose box is largest once turned."""
    from detail.storage import heading_definitions

    rng = np.random.default_rng(seed)
    n = 24
    frames, masks = [], []
    for f in range(n):
        for _row in range(2 if f % 7 == 3 else 1):
            kind = int(rng.integers(4))
            if kind == 0:
                y0, x0 = int(rng.integers(H - 1)), int(rng.integers(W - 1))
                m = np.zeros((H, W), bool)
                m[y0 : y0 + int(rng.integers(1, H - y0 + 1)), x0 : x0 + 1] = True
                m[y0, x0 : x0 + int(rng.integers(1, W - x0 + 1))] = True
                masks.append(_counts(m))
            elif kind == 1:
                length = int(rng.integers(2, H + 1))
                masks.append(
                    _counts(_stroke(length, int(rng.integers(W - length + 1))))
                )
            elif kind == 2:
                masks.append("")
            else:
                masks.append(_counts(np.zeros((H, W), bool)))
            frames.append(f)
    recording_dir = _setup(tmp_path, [0] * len(frames), frames=frames, masks=masks)
    _keypoint_runs(recording_dir, _heading_rows(rng.uniform(0, 360, n)))
    heading_definitions.save_definition(tmp_path, "nose_to_tail", ["nose"], ["tail"])
    return recording_dir


def _alignment_of(recording_dir) -> dict:
    from detail.storage import recording as recording_store

    config = embedding._resolve_alignment(
        recording_dir,
        recording_store.read_recording(recording_dir),
        _aligned_config(),
        [0],
    )
    return embedding._alignment_settings(config)


def _turned_canvas_decoding_every_mask(recording_dir, settings) -> int:
    """The canvas pre-pass as first written: every mask decoded, every pixel turned."""
    from detail.analysis import body_axis

    def turned_mask_extent(mask, deg):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return 0, 0
        theta = np.radians(-deg)
        rx = xs * np.cos(theta) - ys * np.sin(theta)
        ry = xs * np.sin(theta) + ys * np.cos(theta)
        return (
            int(np.ceil(round(float(rx.max() - rx.min()), 6))) + 1,
            int(np.ceil(round(float(ry.max() - ry.min()), 6))) + 1,
        )

    unbounded = {**settings, "max_gap_s": float("inf"), "align": "rotate"}
    h = embedding.headings_for_run(
        recording_dir, "track_01", [0], 1, unbounded, (H, W)
    )[(0, "part_001")]
    deg = {
        int(f): float(d)
        for f, d, s in zip(h["frame_idx"], h["deg"], h["source"])
        if s != body_axis.SOURCE_NONE
    }
    det = PartZarrReader(
        recording_dir / "detections" / "det_01" / "part_001.zarr"
    ).read()
    union: dict[int, np.ndarray] = {}
    for f, counts in zip(det["frame_idx"].tolist(), det["mask_rle"].tolist()):
        if int(f) in deg:
            mask = decode_mask(str(counts), (H, W))
            union[int(f)] = union.get(int(f), np.zeros((H, W), bool)) | mask
    return max(max(turned_mask_extent(m, deg[f])) for f, m in union.items()) + 2


@pytest.mark.parametrize("seed", range(8))
def test_the_turned_canvas_is_the_largest_turned_box_over_every_mask(tmp_path, seed):
    """The pre-pass skips every frame whose box cannot beat the largest turned box so far.
    Skipping is only right if the bound really bounds: a bound that is too tight (the
    unturned box's longer side, say — a diagonal stroke turns to longer than that) drops the
    frame that sets the canvas, and the run then crops its largest animal. Measured against
    decoding every mask, over masks and headings that vary per frame."""
    recording_dir = _random_aligned_recording(tmp_path, seed)
    settings = _alignment_of(recording_dir)
    assert embedding._largest_turned_mask_px(
        recording_dir, "track_01", [0], settings
    ) == _turned_canvas_decoding_every_mask(recording_dir, settings)


def test_the_aligned_canvas_decodes_only_masks_that_could_be_the_largest(
    tmp_path, monkeypatch
):
    """A frame that cannot hold the largest mask is never decoded.

    Decoding every mask held a launch on one status line for a long time before a run
    began.
    A frame whose box diagonal cannot beat the largest turned box so far is never
    decoded. Through ``validate``: one long stroke sets the canvas, nine small squares
    cannot reach it, and only the stroke is decoded."""
    from detail.storage import heading_definitions

    square = np.zeros((H, W), bool)
    square[0:2, 0:2] = True
    masks = [_counts(_stroke(12))] + [_counts(square)] * 9
    recording_dir = _setup(tmp_path, [0] * 10, masks=masks)
    _keypoint_runs(recording_dir, _heading_rows([45.0] * 10))
    heading_definitions.save_definition(tmp_path, "nose_to_tail", ["nose"], ["tail"])
    monkeypatch.setattr(
        "detail.storage.video_integrity.validate_source_videos", lambda *a, **k: []
    )
    decoded = []
    real_decode = embedding.decode_mask
    monkeypatch.setattr(
        embedding,
        "decode_mask",
        lambda counts, size: decoded.append(counts) or real_decode(counts, size),
    )
    with _registered_embedder():
        _module, config = _validate(recording_dir, _aligned_config())
    canvas = next(s for s in config["renderer_spec"] if s["type"] == "center_in_square")
    # The 12-pixel stroke turned onto its own 45° axis spans 11·√2 = 15.6 px: 17 pixels,
    # then the canvas's 2 px of edge.
    assert canvas["size"] == 19
    assert decoded == [masks[0]]


def test_a_student_track_run_takes_the_detect_methods_display_cut_off(
    aligned_recording, monkeypatch
):
    """A `despike_smooth` config carries no `decode_cutoff`, so the heading falls back to
    the detect method's `score_spec.default_cutoff`; a `temporal_decode` config's own value,
    `None` included, is taken as written."""
    from detail.core.keypoints import ScoreSpec
    from detail.core.registries import KEYPOINT_MODELS
    from detail.storage import keypoint_runs

    recording_dir = aligned_recording

    class _Student:
        score_spec = ScoreSpec("detector confidence", 0.0, 1.0, 0.5)

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _Student)
    track_dir = keypoint_runs.track_run_dir(recording_dir, "kt_01")
    keypoint_runs.update_track_yaml(
        track_dir, method="despike_smooth", config={"despike": True, "smooth": True}
    )
    with _registered_embedder():
        _module, config = _validate(recording_dir, _aligned_config())
    assert config["heading_cutoff"] == 0.5
    keypoint_runs.update_track_yaml(
        track_dir, method="temporal_decode", config={"decode_cutoff": None}
    )
    with _registered_embedder():
        _module, config = _validate(recording_dir, _aligned_config())
    assert (
        config["heading_cutoff"] is None
    )  # the method's own reading, taken as written


# --------------------------------------------------------------------------- #
# One default per stage: an omitted model / constructor resolves and is written back
# --------------------------------------------------------------------------- #
def test_an_omitted_model_and_constructor_resolve_to_the_config_default():
    """The run record names what ran (Tenet 9): an omitted model and input image become the
    config's defaults in the config ``validate`` hands ``build_job``, not ``None``."""
    from detail.configs.embedding import EmbeddingConfig
    from detail.transforms.constructors import DEFAULT_CONSTRUCTOR

    config = {"tracking_run_id": "track_01"}
    embedding._resolve_defaults(config)
    assert config["model_id"] == EmbeddingConfig.model_fields["model_id"].default
    assert config["model_id"] == "dinov3_vit"
    assert config["constructor"] == DEFAULT_CONSTRUCTOR


def test_a_pinned_renderer_spec_never_gains_the_default_constructor(tmp_path):
    """A raw ``renderer_spec`` pins the input image itself; recording the default constructor
    beside it would make comparability treat the pinned spec as a recipe. The key is set to
    ``None`` *before* the experiment fills model defaults, so the stored record keeps it so.
    """
    from detail.api.experiment import _with_model_defaults

    module = embedding.EmbeddingModule()
    spec = [{"type": "init_binary_mask"}, {"type": "resize", "size": 224}]
    config = module.resolve_shared_config(
        tmp_path,
        ["recording_001"],
        {"tracking_run_id": "track_01", "renderer_spec": spec},
    )
    stored = _with_model_defaults(module, config)
    assert stored["constructor"] is None
    assert stored["model_id"] == "dinov3_vit"
