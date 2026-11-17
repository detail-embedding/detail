"""Prompt-testing — pre-detection prompt evaluation (Section 8).

Unlike detection/tracking/embedding, prompt-test is **synchronous and in-process**:
the caller blocks while M sampled frames × N prompts run through the model and
``list[PromptTestResult]`` is returned directly. No zarr output, no executor job,
no recovery — it is a throwaway inspection step a user runs before committing to
a full detection run.

Optional disk save (``save_as``) writes a ``test.json`` plus per-frame, per-prompt
overlay JPGs to ``recordings/<id>/detections/prompt_tests/<save_as>/`` so users can
open the JPGs in any image viewer to compare prompts. Saved artifacts are not
recorded in ``recording.yaml`` — they are transient.

Frame seeking is permitted here (Tenet 11 explicitly exempts preview / prompt-test
paths). It goes through ``video.read_frame_at`` with the part's timestamp map, so the
decoded frame is the requested presentation index; ``PromptTestResult.frame_idx``
still reports what was actually decoded.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from detail.configs.display import KEEP_ALL_PROPOSALS
from detail.core.dataclasses import (
    FrameResult,
    Prompt,
    PromptTestJob,
    PromptTestResult,
)
from detail.core.utils import resolve_device
from detail.core.visualize import (
    assign_colors,
    bgr_to_rgb,
    draw_detections,
    rgb_to_bgr,
)
from detail.models.resolve import resolve_detection_model_auto
from detail.storage import recording as recording_store
from detail.storage.video import read_frame_at

_EMPTY_PROMPT = Prompt(targets=[])  # closed-set predict ignores it (all classes)


def run_prompt_test(
    recording_dir: Path,
    job: PromptTestJob,
    save_as: str | None = None,
    seed: int | None = None,
    min_conf: float = 0.0,
    on_result: Callable[[PromptTestResult], None] | None = None,
) -> list[PromptTestResult]:
    """Sample frames, run each prompt on each, return per-frame results.

    The model runs wide open, so the returned detections are **every proposal it found**,
    each carrying its ``confidence``. Filter them at whatever threshold you want to look
    at — that is the point, and it is why there is no confidence parameter on the job.

    Args:
        recording_dir: The recording directory (absolute path).
        job: Bundled parameters; see :class:`PromptTestJob`.
        save_as: If set, write overlay JPGs + ``test.json`` to
            ``detections/prompt_tests/<save_as>/``. If None, results are
            returned in memory only.
        seed: Optional RNG seed for reproducible sampling.
        min_conf: The threshold the **saved JPGs** are drawn at. A picture can't be
            re-thresholded once written, so the save path has to pick one; it does not
            affect the returned results, which always carry everything.
        on_result: Called with each frame's result as it is produced, instead of
            accumulating them (the return is then empty). For a caller that would
            otherwise hold every proposal's full-resolution mask at once — see
            ``run_prompt_test_on_parts``. Cannot be combined with ``save_as``.

    Returns:
        One ``PromptTestResult`` per sampled frame, keyed by ``prompt_0`` …
        ``prompt_{N-1}`` in ``results`` — or an empty list when ``on_result`` is given.

    Raises:
        ValueError: If the recording has no video parts, no frames could be decoded from
            any sample, or ``save_as`` and ``on_result`` are both given.
    """
    if save_as is not None and on_result is not None:
        raise ValueError(
            "save_as needs every result in hand to write its JPGs, and on_result exists "
            "to avoid holding them — pass one or the other"
        )
    recording_dir = Path(recording_dir)
    data = recording_store.read_recording(recording_dir)
    files = recording_store.video_files(data)
    if not files:
        raise ValueError(f"recording {data.get('id')!r} has no video parts")

    # Resolve through the shared resolver (not the MODELS registry) so a trained /
    # imported closed-set model id (model_NNN) loads too — the open-vocab path only
    # reached the pretrained registry.
    model, _ = resolve_detection_model_auto(
        recording_dir.parents[1], job.model_id, job.model_variant
    )
    model.load(resolve_device(job.compute, model.supported_backends))

    def _timestamps_for(part_index: int):
        # Loaded before the read, not after: it is also the seek index, so
        # read_frame_at resolves the frame without scanning the file's packets.
        return recording_store.load_active_timestamps_cached(
            recording_dir, data, part_index
        ).values

    results, samples, prompt_labels = run_prompt_test_on_parts(
        [recording_store.resolve_part_video_path(recording_dir, p) for p in files],
        job.prompts,
        model,
        job.n_sample_frames,
        timestamps_for=_timestamps_for,
        seed=seed,
        on_result=on_result,
    )

    if save_as is not None:
        colors = assign_colors(prompt_labels, seed=seed)
        save_prompt_test_results(
            recording_dir / "detections" / _PROMPT_TESTS_DIRNAME / save_as,
            {
                "model_id": job.model_id,
                "min_conf": min_conf,
                "n_sample_frames": job.n_sample_frames,
                "compute": job.compute.to_dict(),
            },
            job.prompts,
            samples,
            results,
            prompt_labels,
            colors,
            min_conf=min_conf,
        )
    return results


def run_prompt_test_on_parts(
    paths: list[Path],
    prompts: list[Prompt],
    model,
    n_sample_frames: int,
    *,
    timestamps_for: Callable[[int], np.ndarray | None] | None = None,
    sampling: str = "random",
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
    on_result: Callable[[PromptTestResult], None] | None = None,
) -> tuple[list[PromptTestResult], list[tuple[int, int]], list[str]]:
    """Sample across ``paths`` and run every prompt on each sampled frame.

    The recording-free core of ``run_prompt_test``: it takes plain video paths and an
    **already-loaded** model, so a project-free caller (the Quick segment tool) gets the
    same multi-prompt comparison without inventing its own. ``timestamps_for(part_index)``
    supplies that part's authoritative ``frame_idx -> t_ns`` map when the caller has one;
    returning ``None`` is allowed but means ``read_frame_at`` falls back to its own packet
    index — see the tool, which generates a map precisely to avoid that.

    The model runs **wide open** (``KEEP_ALL_PROPOSALS``) and every proposal comes back:
    SAM3 filters inside ``Sam3Processor`` before returning, so a threshold applied here
    would destroy the low-confidence rows rather than hide them, and no later view could
    get them back. The confidence a result is *shown* at is chosen afterwards
    (``configs/display.ConfidenceView``).

    ``on_result`` receives each frame's result as it is produced and, when given, turns
    off accumulation — the returned list comes back empty. Running wide open makes that
    matter: a mask is a full-resolution array (~2 MB at 1080p) and a frame can now carry
    every proposal the model found, so holding all of them across every sampled frame and
    prompt is gigabytes. A caller that only needs each result once (to compact it, or to
    write it to a store) should take this path instead of the list.

    Returns ``(results, samples, prompt_labels)`` — the extras are what
    ``save_prompt_test_results`` needs to write the viewer's artifacts.
    """
    part_frame_counts = [_frame_count(p) for p in paths]
    samples = _sample_frames(
        part_frame_counts, n_sample_frames, seed=seed, sampling=sampling
    )  # list[(part_index, local_frame_idx)]
    model.set_conf_threshold(KEEP_ALL_PROPOSALS)
    closed_set = not model.supported_cue_types

    prompt_labels = [f"prompt_{i}" for i in range(len(prompts))]
    results: list[PromptTestResult] = []
    decoded_frames = 0  # counted separately: on_result mode never fills ``results``
    stopped = False
    for done, (part_index, requested_frame) in enumerate(samples):
        if stop_check is not None and stop_check():
            stopped = True
            break
        tsmap = None if timestamps_for is None else timestamps_for(part_index)
        decoded = read_frame_at(
            paths[part_index],
            requested_frame,
            timestamps=tsmap,
        )
        if decoded is None:
            continue
        actual_frame, frame_bgr = decoded
        # cv2 reads BGR. Models receive BGR (the detection path's contract — each
        # backend handles channel order internally); RGB is only for the overlay.
        frame_rgb = bgr_to_rgb(frame_bgr)
        t_ns = (
            int(tsmap[actual_frame])
            if tsmap is not None and actual_frame < len(tsmap)
            else -1
        )

        per_prompt: dict[str, FrameResult] = {}
        if closed_set:
            # A closed-set model ignores the prompt and returns all its classes in one
            # pass; predict once, then narrow each column to the classes its prompt
            # names. An **empty** prompt means "whatever the model finds" — the tool
            # path sends one for a closed-set model, since there is nothing to type —
            # and a prompt may name several classes, so this filters on the whole set
            # rather than indexing targets[0] (which raised on the empty prompt).
            det_all = model.predict(frame_bgr, _EMPTY_PROMPT)
            found = det_all.detections()
            for label, prompt in zip(prompt_labels, prompts):
                wanted = {t.class_name for t in prompt.targets}
                per_prompt[label] = FrameResult.from_detections(
                    frame_idx=actual_frame,
                    t_ns=t_ns,
                    detections=(
                        found
                        if not wanted
                        else [d for d in found if d.class_name in wanted]
                    ),
                )
        else:
            for label, prompt in zip(prompt_labels, prompts):
                det = model.predict(frame_bgr, prompt)
                # The model fills frame_idx/t_ns with placeholders; stamp ours.
                det.frame_idx = actual_frame
                det.t_ns = t_ns
                per_prompt[label] = det

        result = PromptTestResult(
            frame_idx=actual_frame,
            t_ns=t_ns,
            frame_image=frame_rgb,
            results=per_prompt,
        )
        decoded_frames += 1
        if on_result is not None:
            on_result(result)  # the caller keeps what it needs; we drop our reference
        else:
            results.append(result)
        if progress is not None:
            progress(done + 1, len(samples))

    # A stop is not a decode failure (§27): a loop the user ended before its first frame
    # returns empty, and the caller records it stopped. Only a loop that ran out of samples
    # with nothing decoded says the video is unreadable.
    if not decoded_frames and not stopped:
        raise ValueError("no frames could be decoded from the sampled positions")
    return results, samples, prompt_labels


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _sample_frames(
    part_frame_counts: list[int],
    n_sample_frames: int,
    seed: int | None,
    sampling: str = "random",
) -> list[tuple[int, int]]:
    """Sample across parts, weighted by part frame count.

    Returns ``(part_index, local_frame_idx)`` pairs, without replacement across the
    cumulative frame range, so the same (part, frame) is never picked twice.

    ``sampling``:
    - ``"random"`` — uniform random (the recording path's long-standing behavior).
    - ``"even"`` — evenly spaced across the whole range. Deterministic, and it
      guarantees the start and end are represented: a random draw can miss the tail
      of a clip, which is exactly where "there's no animal at the very end" hides.
    """
    cumulative: list[int] = []
    total = 0
    for n in part_frame_counts:
        total += n
        cumulative.append(total)
    if n_sample_frames > total:
        raise ValueError(
            f"n_sample_frames ({n_sample_frames}) exceeds total recording frames ({total})"
        )
    if sampling == "even":
        # Midpoints of n equal buckets: spans the range without pinning frame 0, which
        # on real footage is often a blank lead-in.
        step = total / n_sample_frames
        global_indices = sorted(
            {min(int(step * (i + 0.5)), total - 1) for i in range(n_sample_frames)}
        )
    elif sampling == "random":
        global_indices = random.Random(seed).sample(range(total), n_sample_frames)
    else:
        raise ValueError(f"unknown sampling {sampling!r}; use 'random' or 'even'")
    samples: list[tuple[int, int]] = []
    for g in global_indices:
        part_index = 0
        while g >= cumulative[part_index]:
            part_index += 1
        prior = cumulative[part_index - 1] if part_index > 0 else 0
        samples.append((part_index, g - prior))
    samples.sort()  # group by part for cache-friendlier seeking
    return samples


# --------------------------------------------------------------------------- #
# Frame I/O
# --------------------------------------------------------------------------- #
def _frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


# --------------------------------------------------------------------------- #
# Save: test.json + overlay JPGs
# --------------------------------------------------------------------------- #
_FRAMES_DIRNAME = "frames"
_PROMPT_TESTS_DIRNAME = "prompt_tests"


def save_prompt_test_results(
    out: Path,
    meta: dict,
    prompts: list[Prompt],
    samples: list[tuple[int, int]],
    results: list[PromptTestResult],
    prompt_labels: list[str],
    colors: dict[str, tuple[int, int, int]],
    min_conf: float = 0.0,
) -> None:
    """Write ``test.json`` + one overlay JPG per (frame, prompt) under ``out``.

    Takes the destination directory rather than deriving it from a recording, so a
    project-free caller writes the same artifacts into its own directory. ``meta`` is
    folded into ``test.json`` verbatim — the caller decides which of its parameters are
    worth recording.

    ``min_conf`` is the threshold the images are drawn at, and the counts in ``test.json``
    are counted at the same one so the numbers describe the pictures beside them. A JPG is
    a decision that can't be revisited: the run kept every proposal, but whoever opens
    these files gets one threshold, so it has to be named rather than defaulted silently.
    """
    frames_dir = out / _FRAMES_DIRNAME
    frames_dir.mkdir(parents=True, exist_ok=True)

    test_json = {
        **meta,
        "sampled_frames": [
            {"part_index": p, "frame_idx_requested": f} for p, f in samples
        ],
        "prompts": [
            {"label": label, "prompt": prompt.to_dict()}
            for label, prompt in zip(prompt_labels, prompts)
        ],
        "prompt_colors": {label: list(colors[label]) for label in prompt_labels},
        "frame_results": [
            {
                "frame_idx": result.frame_idx,
                "counts": {
                    label: sum(
                        1
                        for d in result.results[label].detections()
                        if d.confidence >= min_conf
                    )
                    for label in prompt_labels
                },
            }
            for result in results
        ],
    }
    (out / "test.json").write_text(json.dumps(test_json, indent=2), encoding="utf-8")

    for result in results:
        for label in prompt_labels:
            overlay = draw_detections(
                result.frame_image,
                result.results[label].detections(),
                colors[label],
                min_conf=min_conf,
            )
            # cv2.imwrite expects BGR; convert back.
            cv2.imwrite(
                str(frames_dir / f"frame_{result.frame_idx:06d}_{label}.jpg"),
                rgb_to_bgr(overlay),
            )
