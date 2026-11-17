"""Shared, dependency-free helpers for the core layer.

Device resolution (Section 12), track-id resolution (Section 13), and canonical
``t_ns`` <-> seconds conversions for the universal int64-nanosecond time
reference (Tenet 8).
"""

from datetime import datetime, timezone
import math
import os
import re

import numpy as np

from detail.core.dataclasses import ComputeTarget, PolygonKeyframes

NS_PER_SECOND = 1_000_000_000


def safe_path_part(name: str, *, allow_dot: bool = False) -> str:
    """A user's free text, made safe to use as one path component.

    Behavior names, cluster names and group labels are typed by the person running the tool and
    end up in filenames. ``rear/sit`` is not a stem, it is a directory that does not exist, and
    the export aborts on it; ``rear.sit`` is worse, because ``Path.with_suffix`` treats everything
    from the dot as an extension and silently truncates two different measurements to one file.

    ``allow_dot`` keeps ``.`` for a **directory** name, where it is harmless. Leave it off for a
    stem that will have a suffix appended, which is the case that collides.
    """
    allowed = "0-9A-Za-z_.-" if allow_dot else "0-9A-Za-z_-"
    return re.sub(f"[^{allowed}]+", "_", str(name))


def safe_stem(name: str) -> str:
    """A user's free text, made safe to use as a filename stem **without merging two of them**.

    Sanitizing alone is not enough. Collapsing every awkward character to ``_`` maps ``rear/sit``
    and ``rear.sit`` onto one stem, so two behaviors write to one file and one of them is gone
    with nothing raised — the same silent merge as not sanitizing at all, one step further along.
    A name whose *path-meaningful* characters had to be replaced therefore carries a short digest
    of what it originally was.

    Whitespace is the exception, and deliberately so: ``climb wall`` → ``climb_wall`` is what
    every slug in the world does and what a reader expects to see, so it does not earn a digest.
    The characters that do are the ones that change what a path *means* — ``/`` makes a stem into
    a directory that does not exist, ``.`` is eaten by ``Path.with_suffix``.
    """
    import hashlib

    spaced = re.sub(r"\s+", "_", str(name))
    safe = safe_path_part(spaced)
    if safe == spaced:
        return safe
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:6]
    return f"{safe}-{digest}"


def utc_timestamp() -> str:
    """Current UTC time as the canonical ISO-8601 'Z' string for run/job metadata."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_downstream_every_n(requested: int | None, upstream: int) -> int:
    """The stride a downstream stage runs at, given the stride its input already used.

    Every stage counts **source** frames (§4), so a stage can only thin further by a whole
    multiple of what it received: after detection at 2, the reachable strides are 2, 4, 6, …
    Asking for 3 is not a rounding problem — those frames were never processed — so it raises
    rather than snapping to a neighbour (Tenet 17). ``None`` inherits the upstream cadence.

    Raises:
        ValueError: If ``requested`` is below the upstream stride or not a multiple of it.
    """
    upstream = max(1, int(upstream))
    if requested is None:
        return upstream
    requested = int(requested)
    reachable = ", ".join(str(upstream * n) for n in (1, 2, 3, 4))
    if requested < upstream:
        raise ValueError(
            f"frame stride {requested} is finer than the {upstream} its input ran at — "
            f"those frames were never processed. Reachable strides: {reachable}, …"
        )
    if requested % upstream:
        raise ValueError(
            f"frame stride {requested} is not a multiple of the {upstream} its input ran "
            f"at, so it would ask for frames that were never processed. Reachable "
            f"strides: {reachable}, …"
        )
    return requested


def transcode_parallelism() -> int:
    """Concurrent CPU transcodes (video import) to run at once.

    Decoders already self-thread, so this is a conservative floor that won't
    oversubscribe cores; override with ``DETAIL_MAX_TRANSCODE_PARALLELISM``.
    A GPU-box benchmark refines the default (planned)."""
    raw = os.environ.get("DETAIL_MAX_TRANSCODE_PARALLELISM")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, min(4, os.cpu_count() or 1))


def resolve_device(compute: ComputeTarget, supported_backends: list[str]) -> str:
    """Map a ComputeTarget to a device string.

    cuda + device_indices=[0, 1] -> "0,1"  (Ultralytics-compatible format)
    cuda + device_indices=None   -> "0"
    mps                          -> "mps"
    cpu                          -> "cpu"

    The "0,1" multi-GPU string is Ultralytics-compatible. PyTorch-native trainers
    that need a list of ints should parse this or read ``compute.device_indices``.

    Args:
        compute: The requested compute target.
        supported_backends: Backends this component accepts (e.g. ["cpu", "cuda"]).

    Returns:
        The resolved device string.

    Raises:
        ValueError: If the backend is not supported, or cannot be mapped to a
            local device string.
    """
    if compute.backend not in supported_backends:
        raise ValueError(
            f"backend {compute.backend!r} not supported; this component supports "
            f"{supported_backends}"
        )
    if compute.backend == "cuda":
        indices = compute.device_indices or [0]
        return ",".join(str(i) for i in indices)
    if compute.backend == "mps":
        return "mps"
    if compute.backend == "cpu":
        return "cpu"
    raise ValueError(
        f"resolve_device cannot map backend {compute.backend!r} to a local device "
        "string (expected one of: cuda, mps, cpu)"
    )


def torch_device(device: str) -> str:
    """Translate ``resolve_device`` output to a torch device string.

    ``resolve_device`` returns Ultralytics-style indices for cuda ("0", "0,1"); a
    single-process torch model uses the first index as ``cuda:N``. One copy, shared by
    every torch-backed model's ``load`` (it was three private copies).
    """
    if device in ("cpu", "mps"):
        return device
    return f"cuda:{device.split(',')[0]}"


def check_cuda_indices(device_indices: list[int]) -> None:
    """Validate that the requested cuda devices exist on this machine.

    Lazy torch import (Tenet 13) — only fires when a caller actually asks for
    specific cuda devices.

    Raises:
        RuntimeError: If torch is not installed.
        ValueError: If any requested index is out of range for this machine.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "torch is required for cuda backend; install with `pip install -e '.[torch]'`"
        ) from e
    available = torch.cuda.device_count()
    bad = [i for i in device_indices if not (0 <= i < available)]
    if bad:
        raise ValueError(
            f"requested cuda device(s) {bad} not available; this machine has "
            f"{available} cuda device(s) (indices 0..{max(0, available - 1)})"
        )


def cuda_available() -> bool:
    """Whether this machine has a usable CUDA device. Lazy torch import (Tenet 13);
    returns False when torch is absent or no device is present — used to pick a sensible
    default compute so a long job doesn't land on the CPU by accident."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def mps_available() -> bool:
    """Whether this machine has a usable Apple MPS (Metal) device. Lazy torch import
    (Tenet 13); returns False when torch is absent or MPS is unavailable."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def available_backends() -> list[str]:
    """The backends this machine can actually run, fastest first.

    What a compute picker may offer, and what decides whether a model can run here at all.
    ``best_supported_backend`` answers "which *one*", which is not the same question: asked
    for a cuda-only component on a Mac it still returns ``cpu``, a backend that component
    does not accept — so a picker built on it offered a device the machine did not have.
    """
    return (
        (["cuda"] if cuda_available() else [])
        + (["mps"] if mps_available() else [])
        + ["cpu"]  # always present
    )


def cuda_device_names() -> list[str]:
    """Names of the CUDA devices on this machine (empty when none, or torch absent).
    Lazy torch import (Tenet 13) — for the GPU / CUDA status surface."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def best_supported_backend(supported_backends: list[str]) -> str:
    """The fastest available backend a component also supports (cuda > mps > cpu).

    Intersects what this machine has with what the component accepts, so a default
    never lands on a backend the model/trainer can't run — e.g. a Mac (mps) picking a
    trainer that only supports cpu/cuda falls back to cpu. Falls back to cpu when no
    accelerator is available or supported. Over the full list it answers "the best device
    on this machine", which is why there is no separate function for that. Lazy torch
    import (Tenet 13) via the probes above.
    """
    for backend in ("cuda", "mps", "cpu"):
        if backend not in supported_backends:
            continue
        if backend == "cuda" and not cuda_available():
            continue
        if backend == "mps" and not mps_available():
            continue
        return backend
    return "cpu"


def default_compute(supported_backends: list[str] | None = None) -> ComputeTarget:
    """Build a ComputeTarget on the fastest supported+available backend.

    The default for every API entry point, so a long job lands on the fastest device
    present instead of the CPU (or a nonexistent NVIDIA GPU). Pass a component's
    ``supported_backends`` to keep the choice within what it can run; omit for the
    generic cuda > mps > cpu order.
    """
    backend = best_supported_backend(supported_backends or ["cuda", "mps", "cpu"])
    device_indices = [0] if backend == "cuda" else None
    return ComputeTarget(backend, device_indices, None, None)


def resolve_track_id(track_id: int | str, subject_labels: dict | None) -> int:
    """Resolve a track selector to a concrete integer track id.

    An ``int`` is returned unchanged. A ``str`` is treated as an animal id and
    resolved through ``subject_labels`` (track id -> animal id) from the tracking
    run (Section 13).

    Raises:
        ValueError: If a string id is given without ``subject_labels``, or the
            animal id is not present.
    """
    if isinstance(track_id, int):
        return track_id
    if not subject_labels:
        raise ValueError(
            f"Cannot resolve '{track_id}': no subject_labels on this tracking run"
        )
    matches = [tid for tid, aid in subject_labels.items() if aid == track_id]
    if not matches:
        raise ValueError(f"animal_id '{track_id}' not found in subject_labels")
    return matches[0]


def check_track_is_unambiguous(
    track_id: int, recording_id: str, available: list[int], *, run_id: str
) -> None:
    """Reject a bare integer track id in a recording that offers more than one track.

    Track ids are assigned **in order of first appearance within one tracking run** (§13), so an
    integer names a position, not an animal. With a single track that position is unambiguous —
    there is one subject and the int finds it, which is every recording in this project today.
    With several, ``track_id=0`` is "whichever the tracker happened to number first", and that
    can be a different animal in each recording. Pooling subjects across recordings by that
    integer then mixes them, silently and plausibly: every recording contributes bouts, the
    counts look right, and the vocabulary is fitted across two animals.

    So this raises at the point the ambiguity first exists rather than letting the run proceed.
    ``subject_labels`` on the tracking run and ``resolve_track_id`` above are the designed way
    out (§13), and are not built — deliberately, since nothing needs them yet. This is what will
    tell someone they do.

    Cross-recording animal *identity* is a separate thing and already solved: it is a
    recording-level fact carried by the §11 grouping variables (``metadata.csv``'s animal id
    column), which is the right level when one recording holds one animal.

    Raises:
        ValueError: If ``available`` holds more than one track.
    """
    if len(available) <= 1:
        return
    raise ValueError(
        f"recording {recording_id!r} has {len(available)} tracks ({sorted(available)}) in run "
        f"{run_id!r}, so track_id={track_id} is ambiguous: track ids are assigned in order of "
        "first appearance within each recording, so the same integer is not the same animal "
        "across recordings, and pooling them would mix subjects. Run this on one subject at a "
        "time, or assign subject_labels to the tracking run so a track can be named "
        "(DESIGN_PLAN §13 — not built yet)."
    )


def t_ns_to_seconds(t_ns: int) -> float:
    """Convert int64 nanoseconds (Tenet 8) to float seconds."""
    return t_ns / NS_PER_SECOND


#: The int64 range every ``t_ns`` must fit (Tenet 8): about ±292 years of nanoseconds.
T_NS_MIN = -(2**63)
T_NS_MAX = 2**63 - 1


def seconds_to_t_ns(seconds: float, *, what: str = "time") -> int:
    """Convert float seconds to int64 nanoseconds (Tenet 8), rounding to nearest ns.

    **The one checked seconds → ``t_ns`` conversion.** Every importer and every offset a user
    types goes through here, so a value that cannot be a time is refused where it enters,
    naming where it came from (``what``: ``"row 7, column 'seconds'"``), rather than stored and
    failing later. Unchecked, ``round(x * 1e9)`` happily returns a Python int far past int64:
    epoch *milliseconds* in a seconds column (a common logger mistake) stored as
    ``1788000000000000000000``, and the stream's preview then died with ``OverflowError``.

    Raises:
        ValueError: If ``seconds`` is not finite (``nan``/``inf``), or its nanoseconds do not
            fit in int64.
    """
    value = float(seconds)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{what}: {seconds!r} is not a finite number of seconds")
    t_ns = round(value * NS_PER_SECOND)
    if not T_NS_MIN <= t_ns <= T_NS_MAX:
        raise ValueError(
            f"{what}: {seconds!r} seconds is too large to be a time — it is beyond the "
            "int64 nanosecond range (about ±292 years). Does it hold milliseconds or an "
            "epoch timestamp rather than seconds?"
        )
    return t_ns


def check_t_ns(t_ns: int, *, what: str = "time") -> int:
    """``t_ns`` itself, refused if it does not fit in int64 (Tenet 8).

    For a value that arrives already in nanoseconds (a ``t_ns`` column), which has no
    multiplication to overflow in but can still be too large to store.

    Raises:
        ValueError: If ``t_ns`` is outside the int64 range.
    """
    if not T_NS_MIN <= int(t_ns) <= T_NS_MAX:
        raise ValueError(
            f"{what}: {t_ns} ns is beyond the int64 range (about ±292 years), so it cannot "
            "be a time"
        )
    return int(t_ns)


def polygon_at_frame(pk: PolygonKeyframes, frame_idx: int) -> np.ndarray:
    """Interpolate a polygon's vertices at a recording-global ``frame_idx`` (Section 10).

    Linearly interpolates each vertex between the bracketing keyframes; clamps
    to the first/last keyframe outside the keyframe range (no extrapolation).

    Returns:
        Array of shape ``(n_vertices, 2)`` with normalized ``(x, y)`` coords.

    Raises:
        ValueError: If ``pk`` has no keyframes.
    """
    if not pk.keyframes:
        raise ValueError("PolygonKeyframes is empty")
    first_frame, first_verts = pk.keyframes[0]
    if frame_idx <= first_frame:
        return np.asarray(first_verts, dtype=np.float64)
    last_frame, last_verts = pk.keyframes[-1]
    if frame_idx >= last_frame:
        return np.asarray(last_verts, dtype=np.float64)
    for i in range(len(pk.keyframes) - 1):
        f0, v0 = pk.keyframes[i]
        f1, v1 = pk.keyframes[i + 1]
        if f0 <= frame_idx <= f1:
            t = (frame_idx - f0) / (f1 - f0)
            a = np.asarray(v0, dtype=np.float64)
            b = np.asarray(v1, dtype=np.float64)
            return a + t * (b - a)
    raise AssertionError("unreachable: frame_idx fell outside all keyframe brackets")


def active_polygon_at_frame(
    pk: PolygonKeyframes,
    active: tuple[int, int] | None,
    frame_idx: int,
) -> np.ndarray | None:
    """The polygon/point at ``frame_idx`` if the definition is present there, else
    ``None`` (Section 10).

    ``active`` is an inclusive ``(start, end)`` recording-global frame interval, or
    ``None`` for whole-video presence (arenas, and ROIs/POIs without an interval).
    Outside the interval the definition is not present (returns ``None``); inside
    it (or when ``active`` is ``None``) the shape is ``polygon_at_frame`` — clamped
    at the keyframe ends. Works for a 1-vertex point (returns shape ``(1, 2)``).
    """
    if active is not None and not (active[0] <= frame_idx <= active[1]):
        return None
    return polygon_at_frame(pk, frame_idx)


def bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes.

    Scale-invariant, so it works directly on the normalized detection bboxes
    (§9). Returns ``0.0`` for disjoint or degenerate boxes.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def decode_mask(counts: str, size: tuple[int, int]) -> np.ndarray:
    """Decode an ASCII COCO-RLE mask (§9) to a ``(h, w)`` bool array.

    ``counts`` is the RLE string stored in the detection zarr's ``mask_rle``
    column; ``size`` is ``(height, width)``. Empty ``counts`` → an all-False mask.
    """
    if not counts:
        return np.zeros(tuple(size), dtype=bool)
    from pycocotools import mask as mask_utils

    return mask_utils.decode(
        {"counts": counts.encode("ascii"), "size": list(size)}
    ).astype(bool)


def mask_boxes(counts, size: tuple[int, int]) -> np.ndarray:
    """``(n, 4)`` float ``[x, y, w, h]`` pixel boxes tight to each ASCII COCO-RLE mask.

    Read from the runs without decoding a mask, so it is cheap over a whole part. An empty
    ``counts`` (no mask) is the all-zero box.
    """
    from pycocotools import mask as mask_utils

    rles = [{"counts": str(c).encode("ascii"), "size": list(size)} for c in counts]
    if not rles:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(mask_utils.toBbox(rles), dtype=np.float64).reshape(-1, 4)


def _splice_pieces(pieces: list[np.ndarray]) -> np.ndarray:
    """One closed ring through every piece, joined by zero-width corridors.

    Each piece is spliced into the growing ring at the closest pair of points, so the
    path runs out to the piece, around it, and back along the same line. The corridor
    encloses no area, so the ring rasterizes to the union of the pieces rather than to
    anything between them. This is the shape Ultralytics' own COCO converter writes
    (``data/converter.py::merge_multi_segment``); it is reimplemented here because
    ``detail.core`` may not import a backend (Tenet 13).
    """
    merged = pieces[0]
    for piece in pieces[1:]:
        gaps = np.linalg.norm(merged[:, None, :] - piece[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
        merged = np.concatenate(
            [merged[: i + 1], piece[j:], piece[: j + 1], merged[i:]]
        )
    return merged


def mask_to_polygon(mask: np.ndarray, min_area: int = 0) -> list[float] | None:
    """Every external contour of a binary mask as one normalized YOLO polygon.

    Returns a flat ``[x0, y0, x1, y1, ...]`` list in ``[0, 1]`` — one polygon per
    instance — or None when no contour has at least 3 points and area >= ``min_area``
    (px). Converts a detection mask into a training label (§12).

    A label line is one polygon and an instance is routinely more than one piece: an
    animal's mask splits at a thin dark neck, a tail, a limb crossing the body. Keeping
    only the largest piece drops whichever extremity detached — and Ultralytics derives
    the label's *box* from the polygon it is handed (``ops.segments2boxes``), so the box
    shrinks with it. Pieces are therefore spliced into one ring rather than discarded.
    """
    import cv2

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    scored = [(cv2.contourArea(c), c) for c in contours if len(c) >= 3]
    pieces = [
        c.reshape(-1, 2).astype(np.float64)
        for area, c in sorted(scored, key=lambda s: -s[0])
        if area >= min_area and area > 0
    ]
    if not pieces:
        return None
    return (_splice_pieces(pieces) / [w, h]).flatten().tolist()
