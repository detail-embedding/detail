"""Test a keypoint teacher before a distill (§12 *Distill keypoints*): what *Test cut-off…* and
*Test keypoints…* show.

The keypoint counterpart of detection's *Test prompts…*: **synchronous, in-process, writes
nothing**, so trying again is free. It draws frames **the way the distill would**
(``keypoint_dataset._gather`` → ``_share`` + ``_slices``: spread across the recordings in
proportion to their tracked frames, then over time, only frames on which every animal is tracked,
audit frames left out), so what the user judges is what the student would be taught. The patch
teacher places points on them (``keypoint_dataset._ModelTeacher``); a detect-run teacher's stored
instances are read. Either way the instances are given to the tracking run's subjects by §26
*Assignment*'s function, as the distill gives them (``keypoint_dataset._assign``). Points are as
the teacher placed them: nothing is smoothed (both teachers are raw, the user's decision
2026-10-09).

**The held-out check.** For a method that learns from an example set, each example frame is
scored by the *other* examples (``PatchMatch.score_held_out``): the user's own clicks are the
answer key, so :func:`cutoff_report` can say, at any cut-off, how far points above it land from
where a person put them, in **pixels** with % of body size beside (rigs record at different
sizes). Counts are always shown; a median over fewer than :data:`MIN_N` placements is withheld.

Frame seeking is permitted (Tenet 11 exempts preview paths): frames come through
``keypoint_dataset._decode`` (``read_frame_at`` with the part's timestamp map).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.core.dataclasses import ComputeTarget
from detail.core.keypoints import KeypointDatasetJob
from detail.core.skeleton import Skeleton
from detail.core.visualize import bgr_to_rgb
from detail.pipeline import keypoint_dataset as kds
from detail.storage import keypoint_runs

#: Frames a test draws unless asked otherwise: enough to see a teacher's habits, few enough to
#: wait for.
DEFAULT_TEST_FRAMES = 12
#: Below this many placements a median is withheld (the audit's MIN_N_FOR_ESTIMATE rule).
MIN_N = 8
#: "Close to the click": within this share of the subject's body size.
NEAR_FRAC = 0.05


@dataclass
class TestedFrame:
    """One tested frame: where it came from, the picture, and each track's placement."""

    __test__ = False  # named for what it is, not a pytest class

    recording_id: str
    part_id: str
    frame_idx: int
    t_ns: int
    image: np.ndarray  # RGB, the whole frame
    #: track id -> (bbox normalized xyxy, points (K, 2) normalized, scores (K,))
    tracks: dict[int, tuple]


@dataclass
class TeacherTest:
    """What a test found: the frames, the held-out check, and what it cost."""

    frames: list[TestedFrame]
    skeleton: Skeleton
    method: str
    score_meaning: str | None
    score_low: float | None
    score_high: float | None
    default_cutoff: float | None
    device: str | None
    load_seconds: float
    label_seconds: float
    requested: int
    #: ``PatchMatch.score_held_out`` rows, or None when the method has no example set to
    #: hold out (a student, a method without the hook).
    held_out: list[dict] | None = None
    sources: dict = field(default_factory=dict)

    def scores(self) -> np.ndarray:
        """Every placed point's score across the frames, ``(n,)``."""
        out = [
            np.asarray(t[2], float)
            for frame in self.frames
            for t in frame.tracks.values()
        ]
        if not out:
            return np.zeros(0)
        flat = np.concatenate(out)
        return flat[np.isfinite(flat)]

    def taught_visible(self, cutoff: float | None) -> tuple[int, int]:
        """``(points at or above the cut-off, points placed)`` over the tested frames."""
        scores = self.scores()
        if cutoff is None:
            return len(scores), len(scores)
        return int((scores >= cutoff).sum()), len(scores)


def _side(rows: list[dict]) -> dict:
    errors = np.asarray([r["error_px"] for r in rows], float)
    bodies = np.asarray([r["body_px"] for r in rows], float)
    placed = np.isfinite(errors)
    n = int(placed.sum())
    if n < MIN_N:
        return {"n": n, "median_px": None, "median_pct": None, "near": None}
    pct = errors[placed] / np.maximum(bodies[placed], 1e-9) * 100.0
    return {
        "n": n,
        "median_px": float(np.median(errors[placed])),
        "median_pct": float(np.median(pct)),
        "near": float(np.mean(pct <= NEAR_FRAC * 100.0)),
    }


def cutoff_report(held_out: list[dict], cutoff: float | None) -> dict:
    """At ``cutoff``, how far the held-out placements land from the user's clicks: overall
    and per point, ``{"above": side, "below": side}`` where a side is ``{n, median_px,
    median_pct, near}`` (``near``: share within :data:`NEAR_FRAC` of body size). A side with
    fewer than :data:`MIN_N` placements reports its ``n`` and withholds the rest (None).
    """

    def split(rows):
        if cutoff is None:
            return {"above": _side(rows), "below": _side([])}
        return {
            "above": _side([r for r in rows if r["score"] >= cutoff]),
            "below": _side([r for r in rows if not r["score"] >= cutoff]),
        }

    points = sorted({r["point"] for r in held_out})
    return {
        "overall": split(held_out),
        "points": {p: split([r for r in held_out if r["point"] == p]) for p in points},
    }


def run_teacher_test(
    experiment_root: Path | str,
    teacher: dict,
    *,
    compute: ComputeTarget | None = None,
    n_frames: int = DEFAULT_TEST_FRAMES,
    seed: int = 0,
    every_n_frames: int = 1,
    exclude_audit_s: float = 1.0,
    progress: Callable[[str, float, str], None] | None = None,
) -> TeacherTest:
    """Run ``teacher`` (as ``api.keypoint_distill.resolve_teacher`` returns it) on
    ``n_frames`` frames drawn the way the distill would draw them; ``seed`` picks another
    sample of the same kind.

    ``progress(stage, fraction, detail)`` names each phase as it starts.

    Raises:
        ValueError: ``n_frames`` below one, or no tracked frame to test on.
    """
    from detail.core.registries import KEYPOINT_MODELS
    from detail.models.resolve import keypoint_method_class

    if n_frames < 1:
        raise ValueError(f"n_frames must be 1 or more; got {n_frames}")
    if compute is not None and compute.backend == "remote":
        # The run's target says where the distill is submitted, not what device a preview
        # loads the model on (`core.utils.preview_compute`); `None` is the model's own default.
        compute = None
    root = Path(experiment_root)

    def report(stage: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    report("reading tracked frames", 0.0)
    job = KeypointDatasetJob(
        job_id="test",
        dataset_id="test",
        name="test",
        teacher=dict(teacher),
        n_frames=int(n_frames),
        val_split=0.0,
        score_cutoff=0.0,
        exclude_audit_s=float(exclude_audit_s),
        every_n_frames=int(every_n_frames),
        created_at="",
        compute=compute,
    )
    gathered = kds._gather(root, job, report)
    sources = gathered["sources"]
    rng = random.Random(seed)
    ids = list(sources)
    counts = [len(sources[r]["candidates"]) for r in ids]
    chosen = [
        cand
        for recording_id, k in zip(ids, kds._share(counts, n_frames))
        for cand in kds._slices(sources[recording_id]["candidates"], k, rng)
    ]
    if not chosen:
        raise ValueError(
            "no tracked frame to test on: every frame is unassigned, has no teacher "
            f"instance, or is near an audit frame ({gathered['excluded']})"
        )
    by_model = job.teacher_kind == "model"
    if not by_model:
        # The drawn frames' stored instances, given to the tracking run's subjects the way
        # the distill gives them (§26 *Assignment*).
        report("assigning teacher instances", 0.0)
        kds._teach(gathered, chosen)
    method = str(
        teacher["model_id"] if by_model else sources[ids[0]]["detect_meta"]["model_id"]
    )
    cls = keypoint_method_class(root, method) or KEYPOINT_MODELS.get(method)
    spec = getattr(cls, "score_spec", None)
    device = None
    load_s = 0.0
    held_out = None
    model = None
    start = time.monotonic()
    if by_model:
        # One step with no bar inside it: the weights, then the backbone over every example
        # frame. Said in the label, so a long wait reads as the model, not the frames.
        report(f"loading {method}", 0.0, "weights, then the example frames")
        teacher_obj = kds._ModelTeacher(job, root, sources, lambda *_a, **_k: None)
        model = teacher_obj._model
        device = getattr(teacher_obj, "device", None)
        load_s = time.monotonic() - start
    else:
        model, device, load_s = _load_for_held_out(root, sources, compute, report)

    frames: list[TestedFrame] = []
    label_start = time.monotonic()
    pending: list[tuple[dict, np.ndarray]] = []
    where = f" on {device}" if device else ""

    def placed() -> None:
        # Counted by what the model has returned, not what was queued: the teacher runs a
        # batch when it fills and at the end, and a bar driven by the queue sat at 100% while
        # the whole batch ran (reported from the app as "stuck on loading").
        if by_model:
            report(
                "placing points",
                len(pending) / len(chosen),
                f"{len(pending)}/{len(chosen)} frames{where}",
            )

    for n, cand in enumerate(
        sorted(chosen, key=lambda c: (c["recording_id"], c["part_id"], c["frame_idx"]))
    ):
        report("decoding frames", n / len(chosen), f"{n}/{len(chosen)}")
        bgr = kds._decode(
            sources[cand["recording_id"]], cand["part_id"], cand["frame_idx"]
        )
        if bgr is None:
            continue
        if by_model:
            labelled = teacher_obj.add(cand, bgr)
            if labelled:
                pending.extend(labelled)
                placed()
        else:
            pending.append((cand, bgr))
    if by_model:
        placed()
        pending.extend(teacher_obj.flush())
        placed()
    for cand, bgr in pending:
        frames.append(
            TestedFrame(
                recording_id=cand["recording_id"],
                part_id=cand["part_id"],
                frame_idx=int(cand["frame_idx"]),
                t_ns=int(cand["t_ns"]),
                image=bgr_to_rgb(bgr),
                tracks={
                    int(tid): (tuple(t[0]), np.asarray(t[1]), np.asarray(t[2]))
                    for tid, t in cand["tracks"].items()
                },
            )
        )
    label_s = time.monotonic() - label_start
    if model is not None and hasattr(model, "score_held_out"):
        held_out = model.score_held_out(
            progress=lambda done, total: report(
                "checking against your example clicks",
                done / max(total, 1),
                f"{done}/{total} example frames",
            )
        )
    report("complete", 1.0)
    return TeacherTest(
        frames=frames,
        skeleton=gathered["skeleton"],
        method=method,
        score_meaning=getattr(spec, "meaning", None),
        score_low=getattr(spec, "low", None),
        score_high=getattr(spec, "high", None),
        default_cutoff=getattr(spec, "default_cutoff", None),
        device=device,
        load_seconds=load_s,
        label_seconds=label_s,
        requested=int(n_frames),
        held_out=held_out,
        sources={
            r: s.get("detect_run_id") or s.get("tracking_run_id")
            for r, s in sources.items()
        },
    )


def _load_for_held_out(root: Path, sources: dict, compute, report):
    """For a detect-run teacher whose method learns from an example set: the method built
    from the first run's own copy of its set and settings, and loaded, so its examples can be
    scored held out. ``(model or None, device or None, load seconds)``; None for a method with
    no example set (a trained student), which has nothing to hold out."""
    from detail.core.utils import default_compute, resolve_device
    from detail.models.resolve import keypoint_method_class
    from detail.pipeline.keypoint_detect import METHOD_SETTINGS, build_method

    meta = next(iter(sources.values()))["detect_meta"]
    cls = keypoint_method_class(root, meta.get("model_id"))
    if cls is None or not getattr(cls, "needs_examples", False):
        return None, None, 0.0
    if not hasattr(cls, "score_held_out") or not keypoint_runs.detect_example_sets(
        meta
    ):
        return None, None, 0.0
    start = time.monotonic()
    report(f"loading {meta['model_id']}", 0.0)
    model = build_method(
        root,
        meta["model_id"],
        meta.get("model_variant"),
        {name: meta.get(name) for name in METHOD_SETTINGS},
        meta.get("example_set"),
        meta["skeleton"],
        example_sets_by_class=meta.get("example_sets"),
    )
    device = resolve_device(
        compute or default_compute(model.supported_backends), model.supported_backends
    )
    model.load(device)
    return model, device, time.monotonic() - start
