"""Render preview — pre-embedding renderer evaluation (Section 14).

Synchronous and in-process, just like prompt-test (§8). Samples N detections from
the requested tracks of a completed tracking run, decodes each one's frame + mask,
runs them through ``MaskRenderer.from_spec(renderer_spec)``, and returns
``list[RenderPreviewResult]`` directly. No executor job, no zarr, no recovery —
a throwaway inspection step a user runs to tune ``renderer_spec`` before
committing to a full embedding run.

Optional disk save (``save_as``) writes a ``preview.json`` plus two JPGs per sample
(source frame with mask overlay + rendered output) to
``recordings/<id>/embeddings/render_previews/<save_as>/`` for inspection in any
image viewer. Not recorded in ``recording.yaml`` — transient.

Frame seeking is permitted here (Tenet 11 exempts preview paths). It goes through
``video.read_frame_at`` with the part's timestamp map, so the decoded frame is the
requested presentation index; a sample whose decode lands elsewhere is skipped rather
than rendered through another frame's mask.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

from detail.core import atomic
from detail.core import yaml_load
from detail.core.dataclasses import (
    RenderContext,
    RenderPreviewJob,
    RenderPreviewResult,
)
from detail.core.registries import TRANSFORMS
from detail.core.utils import decode_mask
from detail.core.visualize import assign_colors, bgr_to_rgb, draw_mask, rgb_to_bgr
from detail.executor import rundir
from detail.storage import recording as recording_store
from detail.storage.video import read_frame_at
from detail.storage.zarr_io import (
    DetectionPartReader,
    PartZarrReader,
    resolve_assignment_rows,
)
from detail.transforms import basic as _basic  # noqa: F401 — registers transforms
from detail.transforms.renderer import MaskRenderer

_FRAMES_DIRNAME = "frames"
_RENDER_PREVIEWS_DIRNAME = "render_previews"


def run_render_preview(
    recording_dir: Path,
    job: RenderPreviewJob,
    save_as: str | None = None,
    seed: int | None = None,
) -> list[RenderPreviewResult]:
    """Sample detections from the requested tracks, render each, return results.

    Args:
        recording_dir: The recording directory (absolute path).
        job: Bundled parameters; see :class:`RenderPreviewJob`.
        save_as: If set, write source + rendered JPGs and ``preview.json`` to
            ``embeddings/render_previews/<save_as>/``. If None, results are
            returned in memory only.
        seed: Optional RNG seed for reproducible sampling + color assignment.

    Returns:
        One ``RenderPreviewResult`` per sampled detection. ``source_frame`` is
        the decoded video frame with the mask overlaid in the track's color;
        ``rendered_image`` is the model-facing output of the renderer pipeline.

    Raises:
        ValueError: If any transform in ``renderer_spec`` is unknown, if no
            detections exist for the requested tracks, or if a sampled frame
            cannot be decoded.
    """
    recording_dir = Path(recording_dir)
    _validate_transforms(job.renderer_spec)

    data = recording_store.read_recording(recording_dir)
    tracking_dir = recording_dir / "tracking" / job.tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    det_envelope = rundir.read_envelope(detection_dir)
    frame_h, frame_w = det_envelope["mask_resolution"]
    recording_store.raise_on_legacy_downsampling(det_envelope.get("job"))

    requested = set(int(t) for t in job.track_ids)
    candidates = _collect_candidates(
        detection_dir, tracking_dir, requested, track_meta.get("class_name")
    )
    if not candidates:
        raise ValueError(
            f"no detections found in tracking run {job.tracking_run_id!r} "
            f"for track_ids {sorted(requested)}"
        )
    samples = _sample(candidates, job.n_sample_frames, seed)

    renderer = MaskRenderer.from_spec(job.renderer_spec)
    track_colors = assign_colors([str(t) for t in sorted(requested)], seed=seed)

    files = recording_store.video_files(data)
    results: list[RenderPreviewResult] = []
    for cand in samples:
        tsmap = recording_store.load_active_timestamps_cached(
            recording_dir, data, cand.part_index
        )
        decoded = read_frame_at(
            recording_store.resolve_part_video_path(
                recording_dir, files[cand.part_index]
            ),
            cand.frame_idx,
            timestamps=tsmap.values,
        )
        if decoded is None:
            continue
        actual_frame, frame_bgr = decoded
        if actual_frame != cand.frame_idx:
            continue  # never pair a picture with another frame's stored mask
        frame_rgb = bgr_to_rgb(frame_bgr)
        mask = decode_mask(cand.mask_rles[0], (frame_h, frame_w))
        for extra in cand.mask_rles[1:]:  # OR-merge a frame's detections
            mask = mask | decode_mask(extra, (frame_h, frame_w))

        ctx = RenderContext(
            frame=frame_rgb,
            mask=mask,
            bbox=cand.bbox,
            image=None,
            frame_idx=actual_frame,
            t_ns=cand.t_ns,
            class_name=cand.class_name,
            confidence=cand.confidence,
            object_id=cand.track_id,
            aux={},
        )
        rendered = renderer.render(ctx)

        color = track_colors[str(cand.track_id)]
        source_overlay = draw_mask(frame_rgb, mask, color)

        results.append(
            RenderPreviewResult(
                frame_idx=actual_frame,
                t_ns=cand.t_ns,
                track_id=cand.track_id,
                source_frame=source_overlay,
                rendered_image=rendered,
                det_row=cand.det_row,
            )
        )

    if not results:
        raise ValueError("no frames could be decoded from the sampled positions")

    if save_as is not None:
        _save_results(recording_dir, save_as, job, samples, results, track_colors, seed)

    return results


# --------------------------------------------------------------------------- #
# Candidate enumeration + sampling
# --------------------------------------------------------------------------- #
class _Candidate:
    """One (track, part, frame) candidate. A frame may hold several detections of
    the subject (anchor secondaries) — ``mask_rles`` is the list to OR-merge, and
    the scalar fields come from the frame's highest-confidence (representative)
    row; ``bbox`` is the union over the frame's detections."""

    __slots__ = (
        "track_id",
        "part_index",
        "det_row",
        "frame_idx",
        "t_ns",
        "bbox",
        "mask_rles",
        "class_name",
        "confidence",
    )

    def __init__(
        self,
        track_id,
        part_index,
        det_row,
        frame_idx,
        t_ns,
        bbox,
        mask_rles,
        class_name,
        confidence,
    ):
        self.track_id = track_id
        self.part_index = part_index
        self.det_row = det_row
        self.frame_idx = frame_idx
        self.t_ns = t_ns
        self.bbox = bbox
        self.mask_rles = mask_rles
        self.class_name = class_name
        self.confidence = confidence


def _collect_candidates(
    detection_dir: Path,
    tracking_dir: Path,
    requested: set[int],
    class_name: str | None = None,
) -> list[_Candidate]:
    # Group detection rows by (track, part, frame); a group renders as one image
    # (its masks OR-merged), matching what the embedding stage embeds.
    groups: dict[tuple[int, int, int], list[int]] = {}
    det_parts = sorted(detection_dir.glob("part_*.zarr"))
    readers: dict[int, dict] = {}
    for part_index, part_path in enumerate(det_parts):
        det_reader = DetectionPartReader(part_path)
        assign_reader = PartZarrReader(tracking_dir / "assignments" / part_path.name)
        # object_id is parallel to the detection rows the tracker actually kept
        # when it ran (§13) — every row when this run had no class filter, only
        # one class's otherwise (anchor_tracker et al. build object_ids from
        # filter_by_class'd data). raw_rows recovers which reader row each
        # assignment row really is.
        raw_rows = resolve_assignment_rows(det_reader, assign_reader, class_name)
        det = {k: np.asarray(v)[raw_rows] for k, v in det_reader.read().items()}
        readers[part_index] = det
        object_id = assign_reader.read()["object_id"][: len(raw_rows)]
        for row in range(len(raw_rows)):
            tid = int(object_id[row])
            if tid not in requested:
                continue
            groups.setdefault((tid, part_index, int(det["frame_idx"][row])), []).append(
                row
            )

    out: list[_Candidate] = []
    for (tid, part_index, frame), rows in groups.items():
        det = readers[part_index]
        rep = max(rows, key=lambda r: float(det["confidence"][r]))
        boxes = [det["bbox"][r] for r in rows]
        union = (
            min(float(b[0]) for b in boxes),
            min(float(b[1]) for b in boxes),
            max(float(b[2]) for b in boxes),
            max(float(b[3]) for b in boxes),
        )
        out.append(
            _Candidate(
                track_id=tid,
                part_index=part_index,
                det_row=rep,
                frame_idx=frame,
                t_ns=int(det["t_ns"][rep]),
                bbox=union,
                mask_rles=[str(det["mask_rle"][r]) for r in rows],
                class_name=str(det["class_name"][rep]),
                confidence=float(det["confidence"][rep]),
            )
        )
    return out


def _sample(candidates: list[_Candidate], n: int, seed: int | None) -> list[_Candidate]:
    """Uniform sample without replacement; returns all when n >= len(candidates)."""
    if n <= 0:
        raise ValueError(f"n_sample_frames must be > 0, got {n}")
    if n >= len(candidates):
        return list(candidates)
    rng = random.Random(seed)
    return rng.sample(candidates, n)


# --------------------------------------------------------------------------- #
# Mask I/O
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Validation + save
# --------------------------------------------------------------------------- #
def _validate_transforms(renderer_spec: list[dict]) -> None:
    unknown = sorted({s["type"] for s in renderer_spec if s["type"] not in TRANSFORMS})
    if unknown:
        raise ValueError(
            f"unknown transform types: {unknown}; available: {sorted(TRANSFORMS)}"
        )


def _save_results(
    recording_dir: Path,
    save_as: str,
    job: RenderPreviewJob,
    samples: list[_Candidate],
    results: list[RenderPreviewResult],
    track_colors: dict[str, tuple[int, int, int]],
    seed: int | None,
) -> None:
    out = recording_dir / "embeddings" / _RENDER_PREVIEWS_DIRNAME / save_as
    frames_dir = out / _FRAMES_DIRNAME
    frames_dir.mkdir(parents=True, exist_ok=True)

    preview_json = {
        "tracking_run_id": job.tracking_run_id,
        "track_ids": list(job.track_ids),
        "renderer_spec": job.renderer_spec,
        "n_sample_frames": job.n_sample_frames,
        "seed": seed,
        "track_colors": {tid: list(rgb) for tid, rgb in track_colors.items()},
        "sampled_detections": [
            {
                "track_id": c.track_id,
                "part_index": c.part_index,
                "det_row": c.det_row,
                "frame_idx_requested": c.frame_idx,
            }
            for c in samples
        ],
    }
    (out / "preview.json").write_text(
        json.dumps(preview_json, indent=2), encoding="utf-8"
    )

    for result in results:
        stem = f"frame_{result.frame_idx:06d}_track_{result.track_id}"
        # cv2.imwrite expects BGR; convert back from RGB.
        cv2.imwrite(
            str(frames_dir / f"{stem}_src.jpg"), rgb_to_bgr(result.source_frame)
        )
        cv2.imwrite(
            str(frames_dir / f"{stem}_rendered.jpg"),
            (
                rgb_to_bgr(result.rendered_image)
                if result.rendered_image.ndim == 3
                else result.rendered_image
            ),
        )
