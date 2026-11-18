"""Quick segment (§27) — the first tool and the first-run "wow".

Run a segmentation model on raw video clip(s) — no experiment, no import — in one of two
modes:

- **sample** (default): frames spread evenly across each clip, every prompt run on each,
  compared side by side. This is the experiment prompt-test engine
  (``pipeline/prompt_test.py``) pointed at raw files. Sampling rather than reading the
  opening N frames is what makes the tool honest: a still animal or an empty lead-in shows
  nothing at all in the first few seconds.
- **consecutive**: read straight through from frame 0, writing a detection zarr (the same
  schema the pipeline uses), so overlay playback and mp4 export render from ``source
  frames + saved detections``. Uncapped — it can cover a whole recording — and therefore
  stoppable.

**Neither mode writes pictures.** A run stores detections (the pipeline's own schema) plus
each clip's timestamp map; overlays are drawn when a view asks for one, and images reach
disk only through an explicit export. A run therefore costs kilobytes, not a folder of JPGs
nobody asked for.

Both modes **generate the clip's timestamp map** (PTS from packets, no decode) and save it
beside the detections. That is what keeps the overlay honest: playback and export
hand the map back to ``read_frame_at``, so they resolve exactly the frames the run counted
— the single authoritative enumeration every other preview gets from its recording (Tenets
8/11). Without it ``read_frame_at`` derives its own packet index, which can drift from
cv2's sequential count and draw a mask on a neighbouring picture.
"""

import re
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.display import KEEP_ALL_PROPOSALS
from detail.configs.quick_segment import QuickSegmentConfig
from detail.core.dataclasses import Detection, ToolResult
from detail.core.errors import UserInputError
from detail.core.progress import ThrottledProgress
from detail.core.protocols import ModelNotReadyError, Tool
from detail.core.registries import register_tool
from detail.core.utils import decode_mask, resolve_device
from detail.core.visualize import (
    assign_colors,
    bgr_to_rgb,
    draw_detections,
    draw_detections_by_class,
    rgb_to_bgr,
)
from detail.executor.runners import import_runner_modules
from detail.pipeline._detection_worker import _NO_BBOX, _encode_mask
from detail.storage import video as video_store
from detail.storage.zarr_io import DetectionPartReader, DetectionPartWriter

DETECTIONS_FILENAME = "detections.zarr"
TIMESTAMPS_FILENAME = "timestamps.npy"
PROMPT_TEST_FILENAME = "test.json"
TIMESTAMPS_DIRNAME = "timestamps"


def load_timestamps(run_dir: Path, result: dict) -> np.ndarray | None:
    """A consecutive run's saved ``frame_idx -> t_ns`` map, or ``None`` for a run written
    before the map existed (its overlays fall back to read_frame_at's own index)."""
    name = (result.get("outputs") or {}).get("timestamps")
    if not name:
        return None
    path = Path(run_dir) / name
    return np.load(path) if path.is_file() else None


def approx_frame_count(video_path: Path | str) -> int:
    """The video's approximate total frame count via cv2 ``CAP_PROP_FRAME_COUNT``.

    Approximate (advertised fps × duration), which is acceptable in a tool/preview
    path — used only to warn when a clip is longer than the frame cap, never as an
    authoritative count (Tenet 8's carve-out for preview paths)."""
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


@register_tool("quick_segment")
class QuickSegment(Tool):
    """Segment the first N frames of a clip with a text-prompted model."""

    tool_id = "quick_segment"
    label = "Quick segment"
    description = "Run a segmentation model on a short clip."
    category = "Detection & segmentation"
    config_model = QuickSegmentConfig

    def run(
        self,
        config: dict,
        out_dir: Path,
        progress: Callable[[str, float, str], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> ToolResult:
        videos = [Path(v) for v in (config.get("videos") or [config["video"]])]
        prompts = list(config.get("prompts") or [config["prompt"]])
        mode = config.get("mode", "sample")

        model = self._resolve_model(config["model_id"], config.get("model_variant"))
        # The tool loads the model in the caller's process, so check readiness first: a
        # missing backend then reads as what to install, not a bare ModuleNotFoundError,
        # and the probe re-reads site-packages so a backend installed with the app open
        # imports here too.
        status = model.readiness()
        if not status.ready:
            raise ModelNotReadyError(status)
        model.load(resolve_device(config["compute"], model.supported_backends))
        if mode == "sample":
            return self._run_sampled(
                videos, prompts, model, config, out_dir, progress, stop_check
            )
        return self._run_consecutive(
            videos, prompts, model, config, out_dir, progress, stop_check
        )

    def _run_consecutive(
        self,
        videos: list[Path],
        prompts: list,
        model,
        config: dict,
        out_dir: Path,
        progress,
        stop_check,
    ) -> ToolResult:
        """Read straight through from frame 0, writing a detection zarr — the mode that
        can cover a whole recording and the only one an overlay video can be made from
        (sampled frames are not contiguous).

        One prompt only: N prompts over an uncapped sweep is N times an already unbounded
        cost, and comparison is what sampled mode is for.
        """
        video = videos[0]
        prompt = prompts[0]

        meta = video_store.probe_video(video)
        fps = float(meta["fps"])
        width, height = (int(x) for x in meta["resolution"])
        frames_total = approx_frame_count(video)
        if frames_total <= 0:
            raise ValueError(f"video {str(video)!r} has no readable frames")
        n = (
            frames_total
            if config.get("whole_video")
            else min(int(config["n_frames"]), frames_total)
        )

        # The authoritative frame_idx -> t_ns map for this clip, saved beside the
        # detections. Playback and export pass it back to read_frame_at, so they resolve
        # the same frames this loop counted — the one enumeration every other preview in
        # the app already enjoys via the recording's map (Tenets 8/11). Without it
        # read_frame_at derives its own packet index, which can drift from cv2's
        # sequential count and draw a mask on a neighbouring picture.
        timestamps = video_store.generate_timestamps(video, fps)
        np.save(out_dir / TIMESTAMPS_FILENAME, timestamps)

        model.set_conf_threshold(KEEP_ALL_PROPOSALS)
        emit = ThrottledProgress(
            None if progress is None else (lambda ev: progress(*ev))
        )
        class_counts: dict[str, int] = {}
        stopped = False
        cap = cv2.VideoCapture(str(video))
        try:
            processed = 0
            with DetectionPartWriter(out_dir / DETECTIONS_FILENAME) as writer:
                for frame_idx in range(n):
                    if stop_check is not None and stop_check():
                        stopped = True
                        break  # partial output stays openable (the ToolResult says so)
                    ok, frame_bgr = cap.read()  # sequential, no seek (Tenet 11)
                    if not ok:
                        break  # source held fewer frames than the header claimed
                    result = model.predict(frame_bgr, prompt)
                    # The map is authoritative now that we generate one, so the row's
                    # t_ns is the frame's real presentation time, not frame_idx / fps.
                    t_ns = (
                        int(timestamps[frame_idx])
                        if frame_idx < len(timestamps)
                        else (round(frame_idx * 1e9 / fps) if fps > 0 else frame_idx)
                    )
                    detection_idx = 0
                    for det in result.detections():
                        writer.add_row(
                            frame_idx,
                            t_ns,
                            detection_idx,
                            det.class_name,
                            float(det.confidence),
                            tuple(det.bbox) if det.bbox is not None else _NO_BBOX,
                            _encode_mask(det.mask),
                        )
                        class_counts[det.class_name] = (
                            class_counts.get(det.class_name, 0) + 1
                        )
                        detection_idx += 1
                    processed = frame_idx + 1
                    emit(("segmenting", processed / n, f"frame {processed}/{n}"))
        finally:
            cap.release()
        emit(("segmenting", 1.0, f"frame {processed}/{n}"), force=True)

        kept = sum(class_counts.values())
        note = " (stopped early)" if stopped else ""
        return ToolResult(
            n_processed=processed,
            frames_total=frames_total,
            fps=fps,
            resolution=(width, height),
            class_counts=class_counts,
            outputs={
                "detections": DETECTIONS_FILENAME,
                "timestamps": TIMESTAMPS_FILENAME,
            },
            # "proposal(s)", not "detection(s)": the run keeps everything the model found so
            # the threshold can be chosen afterwards, and this total counts all of it. What
            # a given threshold keeps is a live number the result view shows beside its
            # slider — a stored count can't answer a question asked after the run.
            summary=f"{processed} frame(s) segmented, {kept} proposal(s) kept{note}",
            stopped=stopped,
        )

    def _run_sampled(
        self,
        videos: list[Path],
        prompts: list,
        model,
        config: dict,
        out_dir: Path,
        progress,
        stop_check,
    ) -> ToolResult:
        """Spread the frames evenly across the clip(s) and run every prompt on each.

        This is the experiment prompt-test engine pointed at raw files — same sampling,
        same closed-set handling, same stored detections, so the same matrix viewer
        renders it.
        """
        from detail.core.visualize import assign_colors
        from detail.pipeline.prompt_test import run_prompt_test_on_parts

        n_frames = int(config["n_frames"])
        meta = video_store.probe_video(videos[0])
        fps = float(meta["fps"])
        width, height = (int(x) for x in meta["resolution"])
        frames_total = sum(approx_frame_count(v) for v in videos)
        if frames_total <= 0:
            raise ValueError(f"video {str(videos[0])!r} has no readable frames")

        emit = ThrottledProgress(
            None if progress is None else (lambda ev: progress(*ev))
        )
        stopped = False

        def on_progress(done: int, total: int) -> None:
            emit(("segmenting", done / max(total, 1), f"frame {done}/{total}"))

        # Sampling seeks, so each clip needs its map for the seek to land on the frame it
        # asked for — and the same map lets the viewer re-decode that exact frame later.
        emit(("reading timestamps", 0.0, f"0/{len(videos)}"))
        timestamps = []
        for i, video in enumerate(videos):
            timestamps.append(video_store.generate_timestamps(video, fps))
            emit(
                ("reading timestamps", (i + 1) / len(videos), f"{i + 1}/{len(videos)}")
            )

        results, _samples, labels = run_prompt_test_on_parts(
            videos,
            prompts,
            model,
            min(n_frames, frames_total),
            timestamps_for=lambda i: timestamps[i],
            sampling="even",
            progress=on_progress,
            stop_check=stop_check,
        )
        if stop_check is not None and stop_check():
            stopped = True
        colors = assign_colors(labels)
        _save_sampled(
            out_dir,
            videos,
            timestamps,
            prompts,
            labels,
            colors,
            results,
            {
                "model_id": config.get("model_id"),
                "n_sample_frames": len(results),
                "sampling": "even",
                "source_videos": [str(v) for v in videos],
                "compute": config["compute"].to_dict(),
            },
        )
        class_counts: dict[str, int] = {}
        for result in results:
            for label in labels:
                for det in result.results[label].detections():
                    class_counts[det.class_name] = (
                        class_counts.get(det.class_name, 0) + 1
                    )
        kept = sum(class_counts.values())
        note = " (stopped early)" if stopped else ""
        emit(("segmenting", 1.0, f"frame {len(results)}/{len(results)}"), force=True)
        return ToolResult(
            n_processed=len(results),
            frames_total=frames_total,
            fps=fps,
            resolution=(width, height),
            class_counts=class_counts,
            outputs={"prompt_test": PROMPT_TEST_FILENAME},
            summary=(
                f"{len(results)} sampled frame(s) × {len(prompts)} prompt(s), "
                f"{kept} proposal(s) kept{note}"
            ),
            stopped=stopped,
        )

    def _resolve_model(self, model_id: str, model_variant: str | None):
        import_runner_modules(
            PIPELINE_RUNNERS
        )  # populate MODELS + TRAINED_LOADERS (Tenet 1)
        from detail.models.resolve import resolve_pretrained_or_imported

        # A tool has no experiment scope: a built-in model (SAM 3) or a machine-global
        # imported/distilled model (a closed-set YOLO — also the no-GPU path).
        return resolve_pretrained_or_imported(model_id, model_variant)


# --------------------------------------------------------------------------- #
# Overlay rendering — reused by GUI playback and mp4 export (Qt-free)
# --------------------------------------------------------------------------- #
def load_detections(det_zarr_path: Path | str) -> dict:
    """Read a quick-segment run's detection columns into memory (small — ≤ the cap)."""
    return DetectionPartReader(det_zarr_path).read()


def class_colors(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    """One Okabe-Ito color per class name, stable across playback + export."""
    return assign_colors(sorted(set(class_names)))


def _kept(confidences, min_conf: float):
    """``confidences >= min_conf``, compared in the **stored** dtype.

    The zarr holds confidence as float32. Widening it to Python float turns a stored 0.9
    into 0.89999998, so a threshold of exactly 0.9 — a value the slider lands on
    directly — would drop the detections the user aimed at, and the count beside the
    slider would disagree with the overlay for no visible reason.
    """
    return np.asarray(confidences, dtype=np.float32) >= np.float32(min_conf)


def _frame_detections(
    columns: dict, frame_idx: int, size: tuple[int, int], min_conf: float = 0.0
) -> list:
    """Build ``Detection`` objects for one frame from the loaded zarr columns.

    ``min_conf`` filters on the confidence column **before** decoding masks. A run stores
    every proposal the model found, so at a display threshold most rows are never drawn and
    ``decode_mask`` is the expensive part — filtering after decoding would pay for all of
    them.
    """
    dets = []
    keep = _kept(columns["confidence"], min_conf)
    for row in np.nonzero((columns["frame_idx"] == frame_idx) & keep)[0]:
        confidence = float(columns["confidence"][row])
        dets.append(
            Detection(
                mask=decode_mask(str(columns["mask_rle"][row]), size),
                bbox=tuple(float(x) for x in columns["bbox"][row]),
                class_name=str(columns["class_name"][row]),
                confidence=confidence,
                object_id=None,
            )
        )
    return dets


def frame_detection_count(columns: dict, min_conf: float = 0.0) -> int:
    """How many stored detections a display threshold keeps — the count a view shows.

    Reads the confidence column only (no mask decode), so a view can label a grid without
    rendering it. A run's own ``class_counts`` counts every proposal and cannot answer this.
    """
    return int(np.count_nonzero(_kept(columns["confidence"], min_conf)))


def overlay_frame(
    source_video: Path | str,
    frame_idx: int,
    columns: dict,
    colors: dict[str, tuple[int, int, int]],
    timestamps: np.ndarray | None = None,
    min_conf: float = 0.0,
) -> np.ndarray | None:
    """Decode source ``frame_idx`` and draw its saved detections; return an RGB frame.

    ``None`` if the frame can't be decoded. Colors are keyed by class name; each class's
    detections are drawn in its own color (multi-class support). ``min_conf`` draws only
    detections at or above it.

    Pass the run's saved ``timestamps`` map: it is the enumeration the run itself counted
    with, so the decoded picture is the one the masks were computed on. Detections are
    still looked up at the index the decode reports rather than the one requested, so a
    run written before the map existed degrades to "consistent, possibly off-by-a-frame"
    instead of drawing one frame's masks on another.
    """
    decoded = video_store.read_frame_at(source_video, frame_idx, timestamps=timestamps)
    if decoded is None:
        return None
    actual, frame_bgr = decoded
    frame_rgb = bgr_to_rgb(frame_bgr)
    size = (frame_rgb.shape[0], frame_rgb.shape[1])  # (h, w) for mask decode
    dets = _frame_detections(columns, actual, size, min_conf)
    return draw_detections_by_class(frame_rgb, dets, colors)


def _save_sampled(
    out_dir: Path,
    videos: list[Path],
    timestamps: list,
    prompts: list,
    labels: list[str],
    colors: dict,
    results: list,
    meta: dict,
) -> None:
    """Persist a sampled run as **detections**, not pictures.

    One detection zarr per (prompt, clip) plus each clip's timestamp map and a
    ``test.json`` index. Nothing image-shaped is written: overlays are drawn when a view
    asks for them, the same way every run preview in the app works, so a run costs
    kilobytes instead of a folder of JPGs nobody asked for. Export is where images get
    made.
    """
    import json

    for part_index, ts in enumerate(timestamps):
        path = out_dir / TIMESTAMPS_DIRNAME / f"part_{part_index:03d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, ts)

    # Each result is filed under the part it names. The sampler's request list is not
    # parallel to `results` (a sample that does not decode is skipped), so pairing the
    # two by position put every result after a skipped sample under the wrong clip.
    by_part: dict[tuple[str, int], list] = {}
    for result in results:
        for label in labels:
            by_part.setdefault((label, result.part_index), []).append(result)
    for (label, part_index), part_results in by_part.items():
        store = out_dir / label / f"part_{part_index:03d}.zarr"
        store.parent.mkdir(parents=True, exist_ok=True)
        with DetectionPartWriter(store) as writer:
            for result in part_results:
                for i, det in enumerate(result.results[label].detections()):
                    writer.add_row(
                        result.frame_idx,
                        result.t_ns,
                        i,
                        det.class_name,
                        float(det.confidence),
                        tuple(det.bbox) if det.bbox is not None else _NO_BBOX,
                        _encode_mask(det.mask),
                    )

    (out_dir / PROMPT_TEST_FILENAME).write_text(
        json.dumps(
            {
                **meta,
                "sampled_frames": [
                    {"part_index": r.part_index, "frame_idx": r.frame_idx}
                    for r in results
                ],
                "prompts": [
                    {"label": label, "prompt": prompt.to_dict()}
                    for label, prompt in zip(labels, prompts)
                ],
                "prompt_colors": {label: list(colors[label]) for label in labels},
                "frame_results": [
                    {
                        "frame_idx": r.frame_idx,
                        "counts": {
                            label: len(r.results[label].detections())
                            for label in labels
                        },
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read_sampled(run_dir: Path | str) -> dict:
    """A sampled run's ``test.json`` index (prompts, sampled frames, per-frame counts)."""
    import json

    return json.loads(
        (Path(run_dir) / PROMPT_TEST_FILENAME).read_text(encoding="utf-8-sig")
    )


def sampled_frame(
    run_dir: Path | str, source_videos: list[str], sample_index: int
) -> tuple[np.ndarray | None, dict[str, dict], int]:
    """The **expensive** half of a sampled overlay: decode the frame, read the detections.

    Returns ``(frame_rgb, {label: columns}, frame_idx)`` — the decoded picture, each
    prompt's raw zarr columns, and the source frame index they were looked up at.
    ``(None, {}, frame_idx)`` if the frame can't be decoded.

    Split out from ``sampled_overlays`` so a view that re-draws at a new display threshold
    pays the video decode **once** rather than on every slider move. The detections come
    back as raw columns (RLE strings, kilobytes) rather than decoded ``Detection`` objects:
    a full-resolution mask is megabytes, and a run holds every proposal the model found, so
    caching them decoded would cost gigabytes. ``draw_overlays`` decodes only what a given
    threshold actually draws.

    The clip's saved timestamp map goes to ``read_frame_at``, so the picture is the one the
    detections were computed on.
    """
    run_dir = Path(run_dir)
    index = read_sampled(run_dir)
    entry = index["sampled_frames"][sample_index]
    part_index, frame_idx = int(entry["part_index"]), int(entry["frame_idx"])
    ts_path = run_dir / TIMESTAMPS_DIRNAME / f"part_{part_index:03d}.npy"
    timestamps = np.load(ts_path) if ts_path.is_file() else None
    decoded = video_store.read_frame_at(
        source_videos[part_index], frame_idx, timestamps=timestamps
    )
    per_label: dict[str, dict] = {}
    for prompt in index["prompts"]:
        label = prompt["label"]
        store = run_dir / label / f"part_{part_index:03d}.zarr"
        if store.is_dir():
            per_label[label] = load_detections(store)
    if decoded is None:
        return None, per_label, frame_idx
    _actual, frame_bgr = decoded
    return bgr_to_rgb(frame_bgr), per_label, frame_idx


def draw_overlays(
    frame_rgb: np.ndarray,
    per_label_columns: dict[str, dict],
    frame_idx: int,
    colors: dict[str, tuple[int, int, int]],
    min_conf: float = 0.0,
) -> dict[str, np.ndarray]:
    """The **cheap** half: draw each prompt's detections on a copy of one decoded frame.

    Pure and re-runnable — this is what a display-threshold change repeats, so it must not
    decode video or re-read the store. An N-prompt grid costs N frame copies.
    """
    size = (frame_rgb.shape[0], frame_rgb.shape[1])  # (h, w) for mask decode
    out: dict[str, np.ndarray] = {}
    for label, columns in per_label_columns.items():
        dets = _frame_detections(columns, frame_idx, size, min_conf)
        color = colors.get(label, (255, 255, 255))
        out[label] = draw_detections(frame_rgb, dets, color) if dets else frame_rgb
    return out


def sampled_confidences(run_dir: Path | str) -> dict[str, np.ndarray]:
    """Every stored detection's confidence, per prompt — ``{label: 1-D float array}``.

    Confidence columns only, no masks, so a view can hold them and recount at a new
    display threshold without touching disk. The counts in ``test.json`` can't serve this:
    they were counted once, over every proposal, at save time.
    """
    run_dir = Path(run_dir)
    index = read_sampled(run_dir)
    out: dict[str, np.ndarray] = {}
    for prompt in index.get("prompts", []):
        label = prompt["label"]
        parts = [
            np.asarray(load_detections(store)["confidence"], dtype=np.float32)
            for store in sorted((run_dir / label).glob("part_*.zarr"))
        ]
        out[label] = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return out


def sampled_prompt_colors(run_dir: Path | str) -> dict[str, tuple[int, int, int]]:
    """``{prompt label: RGB}`` from a sampled run's index."""
    index = read_sampled(run_dir)
    return {
        label: tuple(rgb) for label, rgb in (index.get("prompt_colors") or {}).items()
    }


def sampled_overlays(
    run_dir: Path | str,
    source_videos: list[str],
    sample_index: int,
    min_conf: float = 0.0,
) -> dict[str, np.ndarray]:
    """Overlays for one sampled position, one per prompt — ``{label: RGB}``.

    Decodes the source frame **once** and draws each prompt's saved detections on a copy,
    so an N-prompt grid costs N draws and a single decode, not N decodes. The composition
    of ``sampled_frame`` + ``draw_overlays``, kept as the one-shot form for callers that
    render a position exactly once (the frame export).
    """
    frame_rgb, per_label, frame_idx = sampled_frame(
        run_dir, source_videos, sample_index
    )
    if frame_rgb is None:
        return {}
    return draw_overlays(
        frame_rgb, per_label, frame_idx, sampled_prompt_colors(run_dir), min_conf
    )


def _prompt_filenames(prompts: list[dict]) -> dict[str, str]:
    """A filename-safe fragment per prompt label, from the prompt's own text.

    Exported stills are named after what was asked for — ``frame_023174_rat_with_tail`` —
    not the internal ``prompt_1``, which means nothing to whoever opens the folder. Two
    prompts that reduce to the same text keep their label so nothing overwrites.
    """
    out: dict[str, str] = {}
    seen: dict[str, int] = {}
    for prompt in prompts:
        label = prompt["label"]
        targets = (prompt.get("prompt") or {}).get("targets") or []
        text = "-".join(t.get("class_name", "") for t in targets) or label
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or label
        seen[safe] = seen.get(safe, 0) + 1
        out[label] = safe if seen[safe] == 1 else f"{safe}_{label}"
    return out


def export_basename(manifest: dict) -> str:
    """A self-describing stem for an export — the source clip, the tool, and the run id.

    An export is something you hand to someone else, so it has to say what it is once it
    has left the app: ``openfield_01_quick-segment_run_003``. Loose ``overlay.mp4`` /
    ``frame_000123_prompt_0.jpg`` files in a downloads folder say nothing a week later.
    """
    stem = Path(manifest["source_video"]).stem
    return f"{stem}_quick-segment_{manifest['id']}"


def export_run(
    run_dir: Path | str,
    manifest: dict,
    parent_dir: Path | str,
    progress: Callable[[str, float, str], None] | None = None,
    min_conf: float = 0.0,
) -> Path:
    """Export a run into a **new, named folder** under ``parent_dir``; return that folder.

    One folder is the shareable unit whichever mode produced the run: a sampled run
    renders its overlay stills plus ``test.json`` into it, a consecutive run renders its
    overlay clip into it. The caller picks where; the name comes from the run so two
    exports never collide or get mixed together.

    ``min_conf`` is the display threshold the pictures are drawn at. An exported image is
    the one form of this run that can't be re-thresholded, so the caller passes the
    threshold it was showing rather than letting the export pick its own.
    """
    out = Path(parent_dir) / export_basename(manifest)
    out.mkdir(parents=True, exist_ok=True)
    if (Path(run_dir) / PROMPT_TEST_FILENAME).is_file():
        export_frames(run_dir, manifest, out, progress, min_conf)
    else:
        export_mp4(
            run_dir,
            manifest,
            out / f"{export_basename(manifest)}.mp4",
            progress,
            min_conf,
        )
    return out


def export_frames(
    run_dir: Path | str,
    manifest: dict,
    out_dir: Path | str,
    progress: Callable[[str, float, str], None] | None = None,
    min_conf: float = 0.0,
) -> int:
    """Render a sampled run's overlays to JPGs in ``out_dir``; return the file count.

    Images exist only because you asked for them: the run stores detections, so this is
    the render, not a copy. Sampled frames are spread across the clip and are not
    contiguous, which is why the shareable artifact is a folder of stills, not a video.

    Raises:
        ValueError: if the run is not a sampled one (use ``export_mp4``).
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if not (run_dir / PROMPT_TEST_FILENAME).is_file():
        raise ValueError(
            "this run has no sampled frames: it read the clip consecutively, so its "
            "output is a video - use Export mp4 instead."
        )
    index = read_sampled(run_dir)
    sources = manifest.get("source_videos") or [manifest["source_video"]]
    _require_sources(sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    emit = ThrottledProgress(None if progress is None else (lambda ev: progress(*ev)))
    total = len(index["sampled_frames"])
    names = _prompt_filenames(index["prompts"])
    written = 0
    for i, entry in enumerate(index["sampled_frames"]):
        for label, rgb in sampled_overlays(run_dir, sources, i, min_conf).items():
            name = f"frame_{int(entry['frame_idx']):06d}_{names[label]}.jpg"
            cv2.imwrite(str(out_dir / name), rgb_to_bgr(rgb))
            written += 1
        emit(("exporting", (i + 1) / max(total, 1), f"frame {i + 1}/{total}"))
    shutil.copy2(run_dir / PROMPT_TEST_FILENAME, out_dir / PROMPT_TEST_FILENAME)
    written += 1
    emit(("exporting", 1.0, f"frame {total}/{total}"), force=True)
    return written


def _require_sources(sources: list[str]) -> None:
    """Refuse an export whose source clip is gone, before anything is written.

    Checked before the writer or the folder is made: a moved clip used to leave an empty
    ``.mp4`` (or folder) behind, with a message about video-excluded experiment exports,
    which a Quick segment run is not.

    Raises:
        UserInputError: A source clip is not at the path the run recorded.
    """
    for source in sources:
        if not Path(source).is_file():
            raise UserInputError(
                f"the clip this run was made from is no longer at {source} (moved or "
                "deleted); put it back there to export this run"
            )


def export_mp4(
    run_dir: Path,
    manifest: dict,
    out_path: Path | str,
    progress: Callable[[str, float, str], None] | None = None,
    min_conf: float = 0.0,
) -> Path:
    """Render a completed run's overlay clip to an ``.mp4`` at ``out_path``.

    Reads the source video + the saved detections (nothing overlay-shaped is stored),
    draws each frame's overlay, and writes an H.264 (or mp4v) clip via
    ``video_store.open_video_writer`` (cv2, no FFmpeg — Tenet 15)."""
    result = manifest["result"]
    if "detections" not in (result.get("outputs") or {}):
        raise ValueError(
            "this run has no overlay video to export: sampled frames are spread across "
            "the clip and are not contiguous, so there is no clip to render. Re-run in "
            "consecutive mode, or export the overlay frames instead."
        )
    source = manifest["source_video"]
    _require_sources([source])
    width, height = (int(x) for x in result["resolution"])
    fps = float(result["fps"]) or 30.0
    n = int(result["n_processed"])
    columns = load_detections(run_dir / result["outputs"]["detections"])
    colors = class_colors([str(c) for c in columns["class_name"]])
    timestamps = load_timestamps(run_dir, result)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emit = ThrottledProgress(None if progress is None else (lambda ev: progress(*ev)))
    writer, _codec = video_store.open_video_writer(out_path, fps, width, height)
    try:
        for frame_idx in range(n):
            frame_rgb = overlay_frame(
                source, frame_idx, columns, colors, timestamps, min_conf
            )
            if frame_rgb is None:
                continue
            writer.write(rgb_to_bgr(frame_rgb))  # cv2 wants BGR
            emit(("exporting", (frame_idx + 1) / n, f"frame {frame_idx + 1}/{n}"))
    finally:
        writer.release()
    emit(("exporting", 1.0, f"frame {n}/{n}"), force=True)
    return out_path
