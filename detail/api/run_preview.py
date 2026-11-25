"""Read-only, per-frame accessors for previewing a completed run's results.

The GUI renders overlays itself (no annotated video is written, §15), so these
expose just what a frame viewer needs: the source frame to decode, the detections
on it (with their track id, for tracking), and a cheap per-frame count for an
overview. Lazy per-frame reads keep memory flat on hour-long videos — only a
frame-index column is held in memory; boxes/masks/track-ids are sliced per frame.
"""

import math
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from detail.analysis import clip_query
from detail.core import atomic
from detail.core.dataclasses import Detection, RenderContext
from detail.core.lru import CountLRU
from detail.core.utils import decode_mask
from detail.core.visualize import bgr_to_rgb
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import streams as streams_store
from detail.storage.video import read_frame_at, read_frames_range
from detail.storage.zarr_io import (
    DetectionPartReader,
    PartZarrReader,
    kinematics_part_path,
)
from detail.transforms import basic as _basic  # noqa: F401 — registers transforms
from detail.transforms.renderer import MaskRenderer

# Embedding-run fields that must match for two runs' embeddings to be comparable in a
# clip-query search: the input-image recipe and the embed model (dim follows the model).
_COMPARABILITY_KEYS = ("renderer_spec", "model_id", "model_variant")

# Identity of the *recipe* that produced the images, independent of the values it derives
# from each recording's own data. Two runs of the same named constructor, model and input
# resolution are comparable even when a per_recording-scoped parameter resolved differently
# (§14) — that difference is a deliberate per-recording normalization, not a mismatch.
# ``temporal_stack_s`` is a **retired** knob kept here on purpose. Temporal stacking built each
# input from three frames' greyscale in the colour channels; it was measured worse and
# removed from EmbeddingConfig. But runs
# made with it are still on disk recording ``temporal_stack_s: 0.2``, and they were shown
# something different from a single-frame run. Dropping the key would make those runs compare as
# interchangeable with plain ones and let a clustering run pool two incompatible spaces silently —
# exactly the failure the key was added to prevent. No new run writes it; every new run reads as
# 0.0 and is comparable with every other. See _recipe_value and tests/api/test_comparability_scope.
_RECIPE_KEYS = (
    "constructor",
    "model_id",
    "model_variant",
    "readout",
    "input_size",
    "temporal_stack_s",
    "every_n_frames",
    # A derived run's transform is fitted across a *set* of recordings, and the tag records
    # only the transform and its parameters — so two derived runs can carry byte-identical
    # provenance and hold different bases. ``_basis_digest`` was added to make that visible;
    # not checking it here left it visible and unread. Measured on the sweep: deriving
    # drop_top per recording gave one run id two different bases (0224086f / 15d8cce3), each
    # recording centred on its own mean, and every cross-recording comparison downstream
    # silently measured part of that basis gap instead of behavior. Absent on a run written
    # before the field existed, which matches every other such run.
    "derived_basis_digest",
    # §14 egocentric alignment: a turned picture is a different picture, and the keypoint
    # track run's *identity* (never its id) says which axis turned it. Defaulted for runs
    # written before alignment existed, which embedded the image as cropped.
    "align",
    "heading_definition",
    "max_gap_s",
    "mirror_hold_s",
    "facing_camera_ratio",
    "keypoint_track_identity",
    # The order the model was handed the pixels in. Runs before the channel-order fix rendered
    # and embedded BGR frames — a red subject reached the model as blue — and record nothing,
    # which reads as "bgr" (``_recipe_value``), so they never pool with an RGB run.
    "channel_order",
)

#: What a run recorded before the channel-order fix is read as: BGR frames reached the model.
_CHANNEL_ORDER_BEFORE_THE_FIX = "bgr"

#: What a run recorded before alignment existed is read as: unaligned, with the defaults.
_ALIGNMENT_DEFAULTS = {
    "align": "none",
    "heading_definition": None,
    "max_gap_s": 0.5,
    "mirror_hold_s": 1.0,
    "facing_camera_ratio": 0.5,
    "keypoint_track_identity": None,
}

# renderer_spec parameters a constructor derives from the data, and the scope field that
# says whether each was resolved per recording (may differ) or once across the selection
# (must match). Anything not listed here must always match.
#
# The max-contrast background is deliberately absent. A multi-recording run resolves it once
# for the whole selection, so it is compared verbatim on every step that carries it — the
# mask, the canvas fill, and the fill alignment's turn copies. Runs recorded before that, which
# say ``bg_color_scope: per_recording``, compare by the colour they actually resolved: the same
# colour is the same image, and two colours are a recording cue the encoder was shown.
_DERIVED_PARAMS = {
    ("center_in_square", "size"): "canvas_scope",
}


def _spec_without_per_recording(spec, scopes: dict) -> tuple:
    """``spec`` reduced to the parts that must match, given which scopes are per-recording.

    A derived parameter whose scope is ``per_recording`` is replaced by a placeholder, so
    two specs differing only there compare equal. Everything else — the step sequence, the
    resize target, every non-derived parameter — is compared verbatim.
    """
    out = []
    for step in spec or []:
        kind = step.get("type")
        reduced = {}
        for key, value in step.items():
            scope_field = _DERIVED_PARAMS.get((kind, key))
            if scope_field is not None and scopes.get(scope_field) == "per_recording":
                reduced[key] = "<per-recording>"
            else:
                reduced[key] = tuple(value) if isinstance(value, list) else value
        out.append((kind, tuple(sorted(reduced.items(), key=lambda kv: kv[0]))))
    return tuple(out)


def _scopes(meta: dict) -> dict:
    """The run's derived-parameter scopes that bear on comparability. A run written before
    the scopes existed resolved its canvas per recording, which is what absence means.
    ``bg_color_scope`` is not read: the background is compared by value (``_DERIVED_PARAMS``).
    """
    return {"canvas_scope": meta.get("canvas_scope", "per_recording")}


def _embedding_meta(recording_dir: Path, embed_run_id: str) -> dict:
    """The ``embedding.yaml`` for one run (carries lineage + comparability fields)."""
    path = Path(recording_dir) / "embeddings" / embed_run_id / "embedding.yaml"
    if not path.exists():
        raise ValueError(
            f"embedding run {embed_run_id!r} not found in recording {Path(recording_dir).name!r}"
        )
    return yaml.safe_load(atomic.read_text(path))


def iter_track_kinematics(
    recording_dir: Path | str, embed_run_id: str, track_id: int
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """A track's §14 ``kinematics/`` sidecar as one ``(part stem, columns)`` per part that has
    one, in part order; ``[]`` when the run never wrote any.

    ``columns`` is ``{frame_idx, t_ns, cx_norm, cy_norm, mask_area_norm, bbox_norm}`` with
    ``frame_idx`` **part-local**, as the sidecar stores it. Kept per part rather than flattened
    so a caller can place each part on the recording's timeline
    (:func:`detail.api.frame_slices.from_kinematics`) or keep a row with its own part: `det_row`
    restarts at zero in every part, so anything flattened first and indexed after resolves the
    wrong detection.
    """
    run_dir = Path(recording_dir) / "embeddings" / embed_run_id / f"track_{track_id}"
    columns = ("frame_idx", "t_ns", "cx_norm", "cy_norm", "mask_area_norm", "bbox_norm")
    parts: list[tuple[str, dict[str, np.ndarray]]] = []
    for part in sorted(run_dir.glob("part_*.zarr")):
        sidecar = kinematics_part_path(part)
        if not sidecar.exists():
            continue
        reader = PartZarrReader(sidecar)
        if reader.n_rows == 0:
            continue
        parts.append(
            (part.stem, {name: np.asarray(reader.column(name)[:]) for name in columns})
        )
    return parts


def load_track_kinematics(
    recording_dir: Path | str, embed_run_id: str, track_id: int
) -> dict | None:
    """A track's per-frame geometry (§14 ``kinematics/``), or ``None`` if it was never written.

    ``{frame_idx, t_ns, cx_norm, cy_norm, mask_area_norm, bbox_norm}``, concatenated over parts
    in frame order — the same rows and the same order as
    :func:`load_track_trajectory`, without loading the embedding column beside it.
    ``frame_idx`` stays part-local; :func:`iter_track_kinematics` is the per-part form.

    **``None``, not an exception.** A run made before kinematics existed simply has none, which is
    a state a caller renders ("re-run embedding, or backfill") rather than an error.
    """
    parts = iter_track_kinematics(recording_dir, embed_run_id, track_id)
    if not parts:
        return None
    return {
        name: np.concatenate([columns[name] for _stem, columns in parts])
        for name in parts[0][1]
    }


def tracking_entries(data: dict, tracking_run_id: str) -> tuple[dict, dict]:
    """A tracking run's entry in ``recording.yaml`` and its detection run's (``{}`` when the
    detection run is not listed).

    Raises:
        ValueError: the tracking run is not in the record.
    """
    entry = {r["id"]: r for r in data.get("tracking_runs", [])}.get(tracking_run_id)
    if entry is None:
        raise ValueError(f"tracking run {tracking_run_id!r} not found in recording")
    detection_entry = {r["id"]: r for r in data.get("detection_runs", [])}.get(
        entry["detection_run_id"], {}
    )
    return entry, detection_entry


def tracking_cadence(data: dict, tracking_run_id: str) -> int:
    """The ``every_n_frames`` a tracking run's rows were made at: its own, else its detection
    run's, else 1 — resolved once here for `TrackingPreview` and the kinematics loader alike,
    so the two cannot read one run at two cadences."""
    entry, detection_entry = tracking_entries(data, tracking_run_id)
    return int(
        entry.get("every_n_frames") or detection_entry.get("every_n_frames") or 1
    )


def embedding_cadence(recording_dir: Path | str, data: dict, embed_run_id: str) -> int:
    """The step, in video frames, between an embedding run's consecutive rows.

    The embed pass keeps a tracked frame when ``frame % every_n_frames == 0`` on top of the
    tracking run's own cadence (``pipeline/embedding.py``), so the rows sit at multiples of
    both: their step is the least common multiple. A run whose record has no
    ``every_n_frames`` embedded every tracked frame (the same default `_recipe_value` applies
    for comparability), so its step is the tracking run's.

    Raises:
        ValueError: the embedding run, or the tracking run it names, is not in the record.
    """
    meta = _embedding_meta(Path(recording_dir), embed_run_id)
    own = int(meta.get("every_n_frames") or 1)
    return math.lcm(tracking_cadence(data, meta["tracking_run_id"]), own)


def _recording_label(recording_dir: Path | str) -> str:
    """The recording's name for an error message, falling back to its folder's id.

    "no embedded rows for track 0" named neither the recording nor how to find it, so a run over
    sixty recordings left the user to guess which one had gone (#24).
    """
    try:
        name = recording_store.read_recording(recording_dir).get("name")
    except (FileNotFoundError, ValueError):
        name = None
    return repr(name or Path(recording_dir).name)


def _embedded_parts(
    recording_dir: Path, embed_run_id: str, track_id: int
) -> list[tuple[PartZarrReader, int]]:
    """A track's non-empty embedding parts as ``(reader, start frame)``, in recording order.

    The one walk both loaders share, so row ``k`` is the same row in each (`unit_timeline`
    relies on it). Each part zarr counts ``frame_idx`` from 0 inside its own part (§5, §14);
    the start frame from `recording_store.part_start_frames`, looked up **by part stem** (an
    empty part is skipped, so a position would be wrong), is what places it on the recording.

    A recording with no video manifest (a synthetic direct-call test) has one frame space only
    if it has one part: a single non-empty part is read as it is stored, and more than one is
    refused, since nothing says where the second starts.

    Raises
    ValueError: a part the recording does not list, or several parts and no manifest
    """
    run_dir = recording_dir / "embeddings" / embed_run_id / f"track_{track_id}"
    readers = [
        (part.stem, reader)
        for part in sorted(run_dir.glob("part_*.zarr"))
        if (reader := PartZarrReader(part)).n_rows
    ]
    starts = recording_store.part_start_frames(recording_dir)
    if starts is None:
        if len(readers) > 1:
            raise ValueError(
                f"{recording_dir.name}: embedding run {embed_run_id!r} track {track_id} has "
                f"{len(readers)} parts but recording.yaml lists no video parts, so their "
                "frames cannot be placed on the recording"
            )
        return [(reader, 0) for _stem, reader in readers]
    placed = []
    for stem, reader in readers:
        if stem not in starts:
            raise ValueError(
                f"{recording_dir.name}: embedding run {embed_run_id!r} has rows for part "
                f"{stem!r}, which recording.yaml does not list"
            )
        placed.append((reader, starts[stem]))
    return sorted(placed, key=lambda item: item[1])


def load_track_trajectory(
    recording_dir: Path | str, embed_run_id: str, track_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate a track's per-part embedding stores into one ``(M, D)`` trajectory.

    Detail stores ``(M, D)`` per track already (no rotation axis), so this is the
    concatenation of ``embeddings/<run>/track_<tid>/part_*.zarr`` in recording order.

    Returns
    (emb (M, D) float32, frame_idx (M,) int64, t_ns (M,) int64), ``frame_idx`` counted
    across the whole recording (the stored part-local value plus its part's start frame,
    `_embedded_parts`), so it is strictly increasing and ``frame_idx`` and ``t_ns`` agree
    through the recording's timestamp map.

    Raises
    ValueError: the track has no embedded rows in this run, or a part cannot be placed
    """
    embs: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for reader, start in _embedded_parts(Path(recording_dir), embed_run_id, track_id):
        embs.append(np.asarray(reader.column("embedding")[:], dtype=np.float32))
        frames.append(np.asarray(reader.column("frame_idx")[:], dtype=np.int64) + start)
        times.append(np.asarray(reader.column("t_ns")[:]))
    if not embs:
        raise ValueError(
            f"no embedded rows for track {track_id} of recording "
            f"{_recording_label(recording_dir)} in embedding run {embed_run_id!r}"
        )
    return np.concatenate(embs), np.concatenate(frames), np.concatenate(times)


class LazyTrajectories:
    """A subject → ``(emb, frame_idx, t_ns)`` mapping that loads each subject's trajectory
    from disk **on access**, so only one full trajectory is resident at a time.

    Preloading every searched subject (the previous behavior) holds all of them in memory at
    once — on a large run (many/long recordings, high-dim embeddings) that is gigabytes and OOMs
    the iteration subprocess. ``query_tracks`` touches each subject once per phase, releasing
    it before the next, so a plain lazy load streams the search. It is a drop-in ``Mapping``
    (``__contains__`` + ``__getitem__``), so the pure matcher and its tests are unchanged. No
    caching: repeated loads only matter once the example set grows large, which is a separate
    (deferred) concern."""

    def __init__(self, experiment_root: Path, embed_run_id: str) -> None:
        self._experiment_root = experiment_root
        self._embed_run_id = embed_run_id

    def _track_dir(self, subject: tuple[str, int]) -> Path:
        sid, tid = subject
        return (
            experiment_store.recording_path(self._experiment_root, sid)
            / "embeddings"
            / self._embed_run_id
            / f"track_{tid}"
        )

    def __contains__(self, subject: tuple[str, int]) -> bool:
        return self._track_dir(subject).is_dir()

    def __getitem__(self, subject: tuple[str, int]):
        sid, tid = subject
        return load_track_trajectory(
            experiment_store.recording_path(self._experiment_root, sid),
            self._embed_run_id,
            tid,
        )


def load_track_timeline(
    recording_dir: Path | str, embed_run_id: str, track_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """A track's ``(frame_idx, t_ns)`` **without** its embeddings — where the subject was
    embedded, not what it looked like.

    Continuity surfaces (the clip picker's coverage band, the CSV checker) need only the
    timeline, and the embedding column dwarfs it: a long track's embeddings are hundreds
    of MB, against under a megabyte for these two columns. Never reach for the full
    trajectory to ask *when* a track exists.

    Returns
    (frame_idx (M,) int64, t_ns (M,) int64), the same rows in the same order as
    :func:`load_track_trajectory`, ``frame_idx`` counted across the whole recording

    Raises
    ValueError: the track has no embedded rows in this run, or a part cannot be placed
    """
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for reader, start in _embedded_parts(Path(recording_dir), embed_run_id, track_id):
        frames.append(np.asarray(reader.column("frame_idx")[:], dtype=np.int64) + start)
        times.append(np.asarray(reader.column("t_ns")[:]))
    if not frames:
        raise ValueError(
            f"no embedded rows for track {track_id} of recording "
            f"{_recording_label(recording_dir)} in embedding run {embed_run_id!r}"
        )
    return np.concatenate(frames), np.concatenate(times)


def embedding_track_ids(recording_dir: Path | str, embed_run_id: str) -> list[int]:
    """Track ids embedded in one run (from its ``embedding.yaml``)."""
    return [
        int(t) for t in _embedding_meta(Path(recording_dir), embed_run_id)["track_ids"]
    ]


def validate_comparability(pairs: list[tuple[Path, str]]) -> tuple:
    """Confirm every ``(recording_dir, embed_run_id)`` shares the input-image recipe +
    embed model (so a cosine comparison across them is meaningful), and is complete.

    Comparison is on the *recipe* — the named constructor, model, variant, read-out and
    input size — plus every part of ``renderer_spec`` that the recipe did not derive from
    a single recording's data. A canvas whose recorded scope is ``per_recording`` is
    allowed to differ between recordings: resolving it per recording is a deliberate
    normalization, not a mismatch (§14). A ``shared`` one must be identical, and the scopes
    themselves must agree. The max-contrast background must always be identical.

    Runs with no recorded ``constructor`` (the raw-API path, which pins ``renderer_spec``
    verbatim and has no recipe identity to compare) keep strict whole-spec equality.

    Returns the common ``(renderer_spec, model_id, model_variant)`` — the first run's, so
    a per_recording-scoped value in it is that recording's.

    Raises
    ValueError: a run is missing/incomplete, or the runs are not comparable
    """
    metas = []
    for recording_dir, embed_run_id in pairs:
        meta = _embedding_meta(recording_dir, embed_run_id)
        if meta.get("status") != "complete":
            raise ValueError(
                f"embedding run {embed_run_id!r} in recording "
                f"{Path(recording_dir).name!r} is {meta.get('status')!r}, not complete"
            )
        metas.append(meta)
    first_key = _comparability_key(metas[0])
    for meta, (recording_dir, embed_run_id) in zip(metas[1:], pairs[1:]):
        if _comparability_key(meta) != first_key:
            raise ValueError(
                "embedding runs are not comparable — clip query requires the same "
                "input-image recipe and embed model across all recordings "
                f"(run {embed_run_id!r} in {Path(recording_dir).name!r} differs)"
            )
    # return in (renderer_spec, model_id, model_variant) declaration order
    return {k: metas[0].get(k) for k in _COMPARABILITY_KEYS}


def _comparability_key(meta: dict) -> tuple:
    """What two runs must share to be comparable.

    Strict whole-spec equality for a run with no named ``constructor`` (the raw-API path
    pins ``renderer_spec`` verbatim and has no recipe identity to compare); otherwise the
    recipe identity, the canvas scope, and the spec with a per_recording canvas masked out.
    The leading flag keeps a pinned-spec run from ever matching a constructor run.

    The fitted basis leads **both** branches: a derived run's transform is not described by
    its recipe or its spec — two runs can agree on every other field and still hold different
    bases — so checking it only on the constructor branch would leave the raw-API path, which
    is where derived runs of pinned-spec sources land, unguarded.
    """
    basis = meta.get("derived_basis_digest")
    if meta.get("constructor") is None:
        return (
            True,
            basis,
            _recipe_value(meta, "channel_order"),
            tuple(_hashable(meta.get(k)) for k in _COMPARABILITY_KEYS),
        )
    scopes = _scopes(meta)
    return (
        False,
        tuple(_recipe_value(meta, k) for k in _RECIPE_KEYS),
        tuple(sorted(scopes.items())),
        _spec_without_per_recording(meta.get("renderer_spec"), scopes),
    )


def _recipe_value(meta: dict, key: str):
    """A recipe field, defaulting the two that post-date the first runs.

    A run recorded before temporal stacking (or before ``every_n_frames``) existed has no key and
    used the single-frame, every-frame input, so it must compare equal to an explicit 0.0 / 1
    rather than to ``None`` — otherwise every old run becomes incomparable with every new one for
    no real reason.
    """
    if key == "temporal_stack_s":
        return float(meta.get(key) or 0.0)
    if key == "every_n_frames":
        return int(meta.get(key) or 1)
    if key == "channel_order":
        return meta.get(key) or _CHANNEL_ORDER_BEFORE_THE_FIX
    if key in _ALIGNMENT_DEFAULTS:
        value = meta.get(key)
        if value is None:
            return _ALIGNMENT_DEFAULTS[key]
        if key == "align" and value == "none":
            return "none"
        return value
    return meta.get(key)


def _hashable(value):
    """Lists (renderer_spec and its color triples) into tuples so keys compare by value."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _build_parts(
    recording_dir: Path,
    data: dict,
    detection_run_id: str,
    every_n_frames: int = 1,
    assignment_dir: Path | None = None,
) -> list[dict]:
    """Per-part readers for a detection run, optionally paired with a tracking
    run's per-part ``object_id`` assignments (parallel to the detection rows).

    ``every_n_frames`` is the run's cadence, carried through so a viewer can step between
    the frames the run actually processed: at a stride, a frame with no row may simply
    never have been looked at, which is not the same as "the model found nothing"."""
    return [
        _build_part(
            recording_dir, data, detection_run_id, i, every_n_frames, assignment_dir
        )
        for i in range(len(recording_store.video_files(data)))
    ]


def _build_part(
    recording_dir: Path,
    data: dict,
    detection_run_id: str,
    part_index: int,
    every_n_frames: int = 1,
    assignment_dir: Path | None = None,
    *,
    timestamps: np.ndarray | None = None,
) -> dict:
    """One part of :func:`_build_parts` — only its own video resolved, only its own map read.

    For a caller that needs one part (a preview's mask build): building every part resolved
    every part's video, so one part whose video is not on this machine failed a build of
    another, and re-parsed every part's timestamp CSV per call. ``timestamps`` is the part's
    map when the caller already holds it (``load_active_timestamps_cached``)."""
    file = recording_store.video_files(data)[part_index]
    stem = recording_store.part_stem(file)
    zarr_path = recording_dir / "detections" / detection_run_id / f"{stem}.zarr"
    reader = DetectionPartReader(zarr_path) if zarr_path.exists() else None
    frame_idx = (
        reader.column("frame_idx")[:]
        if reader is not None and reader.n_rows
        else np.empty(0, dtype=np.int64)
    )
    assignment = None
    if assignment_dir is not None:
        assignment_path = assignment_dir / f"{stem}.zarr"
        if assignment_path.exists():
            assignment = PartZarrReader(assignment_path)
    if timestamps is None:
        # Kept, not just measured: it is the authoritative frame_idx -> t_ns map, so
        # read_frame_at seeks by t_ns instead of scanning the whole file's packets.
        timestamps = recording_store.load_active_timestamps(
            recording_dir, data, part_index
        ).values
    return {
        "video": recording_store.resolve_part_video_path(recording_dir, file),
        "stem": stem,
        "reader": reader,
        "frame_idx": frame_idx,
        "n_frames": len(timestamps),
        "timestamps": timestamps,
        "assignment": assignment,
        "every_n_frames": max(1, int(every_n_frames or 1)),
    }


class DetectionPreview:
    """Per-frame detection overlays for one completed detection run in a recording."""

    def __init__(self, recording_dir: Path | str, run_id: str) -> None:
        self._dir = Path(recording_dir)
        data = recording_store.read_recording(self._dir)
        entry = {r["id"]: r for r in data.get("detection_runs", [])}.get(run_id)
        if entry is None:
            raise ValueError(f"detection run {run_id!r} not found in recording")
        recording_store.raise_on_legacy_downsampling(entry)
        self._parts = _build_parts(
            self._dir, data, run_id, int(entry.get("every_n_frames") or 1)
        )

    @property
    def part_frame_counts(self) -> list[int]:
        return [part["n_frames"] for part in self._parts]

    def timestamps(self) -> np.ndarray:
        """Every source frame's ``t_ns`` across the parts, in scrub order — the map already held.

        One entry per frame of ``sum(part_frame_counts)``, so a player can say what time the
        frame on screen is without deriving one from a frame rate.
        """
        if not self._parts:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(
            [np.asarray(part["timestamps"], dtype=np.int64) for part in self._parts]
        )

    @property
    def frames_processed(self) -> int:
        """How many frames this run actually looked at, across every part.

        Not the same as ``sum(part_frame_counts)``, which counts **source** frames. A run
        at ``every_n_frames=4`` never sees three frames in four, so measuring its coverage
        against the source count reports a healthy run as having missed 75% of the
        recording — a number that is alarming, wrong, and entirely an artifact of the
        denominator.
        """
        return sum(
            len(range(0, part["n_frames"], max(1, int(part["every_n_frames"] or 1))))
            for part in self._parts
        )

    def processed_mask(self) -> np.ndarray:
        """A bool per **source** frame over the global timeline: the frames this run looked at.

        Each part's own ``0, N, 2N, ...`` (§4), the frames :attr:`frames_processed` counts. What
        a coverage figure is measured over, so a frame never looked at is not read as missed.
        """
        out = []
        for part in self._parts:
            mask = np.zeros(int(part["n_frames"]), dtype=bool)
            mask[:: max(1, int(part["every_n_frames"] or 1))] = True
            out.append(mask)
        return np.concatenate(out) if out else np.zeros(0, dtype=bool)

    def part_index_for_stem(self, stem: str) -> int | None:
        """The part index for a video-file stem (e.g. ``part_001``), or None."""
        for index, part in enumerate(self._parts):
            if part["stem"] == stem:
                return index
        return None

    def counts(self, part_index: int) -> np.ndarray:
        """Detections per frame for one part (for the overview strip)."""
        part = self._parts[part_index]
        if not len(part["frame_idx"]):
            return np.zeros(part["n_frames"], dtype=int)
        return np.bincount(part["frame_idx"], minlength=part["n_frames"])

    def _cursor_for(self, part_index: int):
        """The reader for one part, keeping its place between calls.

        Playback asks for the *next* frame, over and over, and `read_frame_at` re-opens and
        re-seeks for each one, over an order of magnitude slower than a cursor on a 1080p
        part. This is the accessor behind the clip player, the input-image
        scrubber and the run viewers, so it is where the complaint that started #337 actually
        lands.

        **One cursor, not one per part.** It holds an open container; keeping one per part would
        make the number of open file handles a property of how many parts a recording happens to
        have. A surface scrubs within a part and crosses a boundary rarely, so a crossing closes
        the old one — costing exactly the seek it replaced.

        The cursor lives as long as this preview; its open file does not. It closes the container
        after ``FrameCursor.IDLE_CLOSE_S`` without a request, because on Windows an open handle
        stops the user moving or deleting the video while the viewer stays open.
        """
        from detail.storage.video import FrameCursor

        if getattr(self, "_cursor_part", None) != part_index:
            existing = getattr(self, "_cursor", None)
            if existing is not None:
                existing.close()
            part = self._parts[part_index]
            self._cursor = FrameCursor(part["video"], timestamps=part["timestamps"])
            self._cursor_part = part_index
        return self._cursor

    def frame(self, part_index: int, local_frame: int, with_masks: bool = False):
        """Decode a source frame + its detections.

        Returns ``(actual_frame_idx, rgb_frame, detections)`` or ``None`` if the
        frame could not be decoded. Detections are for the *actual* decoded frame,
        so overlays match the image even when a seek snaps to a keyframe. Masks
        (COCO RLE) are decoded only when ``with_masks`` — boxes are always present.
        """
        part = self._parts[part_index]
        decoded = self._cursor_for(part_index).frame(local_frame)
        if decoded is None:
            return None
        actual, frame_bgr = decoded
        rgb = bgr_to_rgb(frame_bgr)
        mask_size = rgb.shape[:2] if with_masks else None
        return actual, rgb, self._detections_at(part, actual, mask_size)

    def frames_range(
        self,
        part_index: int,
        first_local: int,
        last_local: int,
        *,
        with_masks: bool = False,
        step: int = 1,
    ) -> Iterator[tuple[int, np.ndarray, list[Detection]]]:
        """Decode a contiguous span of one part, with each frame's detections.

        ``frame`` re-opens and re-seeks the container per call — right for a scrub,
        ruinous for a clip, where every frame of a span is wanted. This walks the span
        once (``storage.video.read_frames_range``), over an order of magnitude faster, and
        yields as it goes so a caller can show the first picture
        without waiting for the last.

        Yields:
            ``(actual_frame_idx, rgb, detections)`` in presentation order.
        """
        part = self._parts[part_index]
        mask_size = None
        for actual, frame_bgr in read_frames_range(
            part["video"],
            first_local,
            last_local,
            timestamps=part["timestamps"],
            step=step,
        ):
            rgb = bgr_to_rgb(frame_bgr)
            if with_masks:
                mask_size = rgb.shape[:2]
            yield actual, rgb, self._detections_at(part, actual, mask_size)

    def detections_at(
        self, part_index: int, local_frame: int, *, mask_hw: tuple[int, int] | None
    ) -> tuple[int, list[Detection]]:
        """:meth:`frame`'s ``(actual_frame_idx, detections)`` for a caller that will not show
        the footage — the detections of ``local_frame`` itself, masks decoded at ``mask_hw``
        when it is given, and **no video opened**.

        ``local_frame`` is what :meth:`frame` lands on whenever the part's timestamp map
        describes its video, which is the map every surface passes (``FrameCursor``); the
        packet-PTS fallback, which can snap to a keyframe, is never reached without a decode.
        """
        part = self._parts[part_index]
        actual = int(local_frame)
        return actual, self._detections_at(part, actual, mask_hw)

    def _detections_at(self, part: dict, frame: int, mask_size=None) -> list[Detection]:
        reader = part["reader"]
        frame_idx = part["frame_idx"]
        if reader is None or not len(frame_idx):
            return []
        lo = int(np.searchsorted(frame_idx, frame, "left"))
        hi = int(np.searchsorted(frame_idx, frame, "right"))
        if lo >= hi:
            return []
        rows = self.read_rows(part, lo, hi, with_masks=mask_size is not None)
        bbox, class_name = rows["bbox"], rows["class_name"]
        confidence, mask_rle = rows["confidence"], rows["mask_rle"]
        object_id = rows["object_id"]
        detections = []
        for j in range(hi - lo):
            oid = int(object_id[j]) if object_id is not None else None
            if oid is not None and oid < 0:
                continue  # detection the tracker left unassigned (filtered)
            detections.append(
                Detection(
                    mask=(
                        decode_mask(str(mask_rle[j]), mask_size)
                        if mask_size is not None
                        else None
                    ),
                    bbox=tuple(float(v) for v in bbox[j]),
                    class_name=str(class_name[j]),
                    confidence=float(confidence[j]),
                    object_id=oid,
                )
            )
        return detections

    def subject_masks(
        self,
        part_index: int,
        track_id: int,
        local_frames: list[int],
        size: tuple[int, int],
    ) -> dict[int, np.ndarray]:
        """``{local_frame: bool mask (size)}`` of one subject on ``local_frames`` of a part,
        the frame's fragments OR-merged, from one read of their rows — what the clip window's
        onion skin is made of, with no video decoded."""
        part = self._parts[part_index]
        frame_idx = part["frame_idx"]
        if part["reader"] is None or not len(frame_idx) or not local_frames:
            return {}
        lo = int(np.searchsorted(frame_idx, min(local_frames), "left"))
        hi = int(np.searchsorted(frame_idx, max(local_frames), "right"))
        if lo >= hi:
            return {}
        rows = self.read_rows(part, lo, hi, with_masks=True)
        object_id = rows["object_id"]
        wanted = set(int(f) for f in local_frames)
        union: dict[int, np.ndarray] = {}
        for j in range(hi - lo):
            frame = int(frame_idx[lo + j])
            if frame not in wanted:
                continue
            oid = int(object_id[j]) if object_id is not None else None
            if oid is None or oid != int(track_id):
                continue
            counts = str(rows["mask_rle"][j])
            if not counts:
                continue
            mask = decode_mask(counts, size)
            union[frame] = (union[frame] | mask) if frame in union else mask
        return union

    #: Rows (positions in the part) per cached block: about one chunk of the ``mask_rle``
    #: column (§9's 1,000 rows), which a read of any one of its rows decompresses whole.
    ROW_BLOCK = 1024
    #: Blocks one preview keeps. A block is its rows' boxes, scores and (with masks) RLE
    #: strings — about a megabyte at 1080p with masks — so a surface holding a preview per
    #: recording stays in the tens of MB.
    ROW_BLOCKS_KEPT = 4

    def read_rows(self, part: dict, lo: int, hi: int, *, with_masks: bool) -> dict:
        """The columns of rows ``lo:hi`` of one part, with each row's track (``object_id``,
        ``None`` without a tracking run) read off the assignment row beside it.

        Each column read is its own zarr call, and zarr v3's async layer charges a fixed
        overhead for one however few rows it returns — five of them per frame cost more
        than decoding the frame, drawing on it and
        encoding it together. A caller that wants a span reads it here in one go.

        **A short span is served from a cached block of :data:`ROW_BLOCK` rows**, kept per
        ``(part, block, with_masks)``. A surface that walks a clip asks for one frame's rows
        and then the few frames before it (the mask trail), step after step, and each ask
        decompressed the same ``mask_rle`` chunk again, which was much of a montage's
        pass. ``with_masks`` is part of the key so a caller that asked for no masks pays
        for none. The rows of a part are fixed when the preview is built, so
        a block never goes stale; a span longer than a block is read directly.
        """
        if hi <= lo or hi - lo > self.ROW_BLOCK:
            return self._read_rows_now(part, lo, hi, with_masks=with_masks)
        size = self.ROW_BLOCK
        pieces = []
        for block in range(lo // size, (hi - 1) // size + 1):
            start = block * size
            rows = self._row_block(part, block, with_masks)
            pieces.append((rows, max(lo - start, 0), min(hi - start, size)))
        out = {}
        for name in ("bbox", "class_name", "confidence", "mask_rle", "object_id"):
            parts = [rows[name][a:b] for rows, a, b in pieces if rows[name] is not None]
            out[name] = (
                None
                if not parts
                else parts[0] if len(parts) == 1 else np.concatenate(parts)
            )
        return out

    def subject_boxes(
        self, part_index: int, track_id: int, lo: int, hi: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """One track's boxes on part ``part_index``'s local frames ``lo..hi`` inclusive:
        ``(frames, boxes)`` — the local frames it has a row on, ascending, and its normalized
        ``(x0, y0, x1, y1)`` on each. Read off the detection rows in blocks (:meth:`read_rows`,
        no masks), never by decoding video; empty without a tracking run."""
        part = self._parts[part_index]
        frame_idx = np.asarray(part["frame_idx"], dtype=np.int64)
        empty = np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
        if not len(frame_idx) or part.get("assignment") is None or hi < lo:
            return empty
        row_lo = int(np.searchsorted(frame_idx, int(lo), side="left"))
        row_hi = int(np.searchsorted(frame_idx, int(hi), side="right"))
        if row_hi <= row_lo:
            return empty
        rows = self.read_rows(part, row_lo, row_hi, with_masks=False)
        ids = rows["object_id"]
        if ids is None:
            return empty
        mine = np.asarray(ids) == int(track_id)
        return (
            frame_idx[row_lo:row_hi][mine],
            np.asarray(rows["bbox"], dtype=np.float64).reshape(-1, 4)[mine],
        )

    def _row_block(self, part: dict, block: int, with_masks: bool) -> dict:
        """Rows ``[block * ROW_BLOCK, (block + 1) * ROW_BLOCK)`` of ``part``, read once and
        kept in a small LRU (several threads may draw from one preview). Keyed on the part's
        stem — stable for the preview's life — never on ``id(part)``, which a rebuilt part
        list would silently reuse."""
        # Lazily, the way `_cursor_for` keeps its cursor: `TrackingPreview.__init__` builds
        # its parts without calling up, so an attribute set in `__init__` is not there.
        kept = getattr(self, "_row_blocks", None)
        if kept is None:
            kept = self._row_blocks = CountLRU(self.ROW_BLOCKS_KEPT)
        key = (str(part["stem"]), int(block), bool(with_masks))
        found = kept.get(key)
        if found is not None:
            return found
        start = int(block) * self.ROW_BLOCK
        stop = min(start + self.ROW_BLOCK, len(part["frame_idx"]))
        found = self._read_rows_now(part, start, stop, with_masks=with_masks)
        kept.put(key, found)
        return found

    def _read_rows_now(self, part: dict, lo: int, hi: int, *, with_masks: bool) -> dict:
        """:meth:`read_rows` straight from the zarr columns."""
        reader = part["reader"]
        assignment = part.get("assignment")
        return {
            "bbox": reader.column("bbox")[lo:hi],
            "class_name": reader.column("class_name")[lo:hi],
            "confidence": reader.column("confidence")[lo:hi],
            "mask_rle": reader.column("mask_rle")[lo:hi] if with_masks else None,
            "object_id": (
                assignment.column("object_id")[lo:hi]
                if assignment is not None
                else None
            ),
        }


class TrackingPreview(DetectionPreview):
    """Per-frame overlays for a tracking run: the detection boxes it tracked, each
    carrying its track id (``object_id``); untracked detections are dropped."""

    def __init__(self, recording_dir: Path | str, run_id: str) -> None:
        self._dir = Path(recording_dir)
        data = recording_store.read_recording(self._dir)
        entry, detection_entry = tracking_entries(data, run_id)
        assignment_dir = self._dir / "tracking" / run_id / "assignments"
        recording_store.raise_on_legacy_downsampling(detection_entry)
        self._parts = _build_parts(
            self._dir,
            data,
            entry["detection_run_id"],
            tracking_cadence(data, run_id),
            assignment_dir,
        )

    def track_presence(self) -> tuple[list[int], dict[int, np.ndarray]]:
        """``(track_ids, {track_id: bool array over the global timeline})`` marking
        where each track exists — for the presence timeline. Cheap: one pass over
        the object_id + frame columns, no decoding."""
        total = sum(self.part_frame_counts)
        presence: dict[int, np.ndarray] = {}
        offset = 0
        for part in self._parts:
            reader, assignment = part["reader"], part["assignment"]
            if reader is not None and assignment is not None and reader.n_rows:
                frame_idx = part["frame_idx"]
                object_id = assignment.column("object_id")[:]
                for track_id in np.unique(object_id[object_id >= 0]):
                    row = presence.setdefault(
                        int(track_id), np.zeros(total, dtype=bool)
                    )
                    row[offset + frame_idx[object_id == track_id]] = True
            offset += part["n_frames"]
        return sorted(presence), presence


def track_coverage(
    present: np.ndarray,
    fps: float | None = None,
    processed: np.ndarray | None = None,
) -> dict:
    """How much of the timeline this track actually holds, and the shape of what it misses.

    ``n_gaps`` in ``tracks.zarr`` is a count of missed *frames*. A thousand one-frame losses
    and one thousand-frame loss are the same number there and are completely different
    problems — the first is a per-frame gate rejecting the subject, the second is the animal
    leaving or the tracker losing it. So this reports the **runs**: how many, and the longest.

    Gaps are counted **inside the track's own span** (first present frame to last). Frames
    before it appears and after it leaves are not gaps in it; counting them would make a
    track that covers the back half of a recording look catastrophically broken.

    ``processed`` marks the frames the run looked at (`DetectionPreview.processed_mask`).
    Only those count: at ``every_n_frames`` 4 the three frames in four never looked at are
    not gaps, and counting them read a healthy strided run as "N gaps · longest 1 frames"
    (#32). ``frames_total`` and ``longest_gap`` are then in processed frames, like the
    coverage line beside them, and ``longest_gap_s`` is the time from the gap's first missed
    frame to the frame the track was seen again. ``None`` treats every frame as processed.
    """
    present = np.asarray(present, dtype=bool)
    positions = (
        np.arange(present.size)
        if processed is None
        else np.flatnonzero(np.asarray(processed, dtype=bool)[: present.size])
    )
    looked = present[positions]
    total = int(looked.size)
    covered = int(looked.sum())
    out = {
        "frames_total": total,
        "frames_covered": covered,
        "fraction": covered / total if total else 0.0,
        "n_gaps": 0,
        "longest_gap": 0,
        "longest_gap_s": None,
        "first_frame": None,
        "last_frame": None,
    }
    if not covered:
        return out
    where = np.flatnonzero(looked)
    first, last = int(where[0]), int(where[-1])
    out["first_frame"], out["last_frame"] = int(positions[first]), int(positions[last])
    missing = ~looked[first : last + 1]
    if missing.any():
        padded = np.concatenate(([False], missing, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[::2], edges[1::2]
        lengths = ends - starts
        out["n_gaps"] = int(lengths.size)
        longest = int(np.argmax(lengths))
        out["longest_gap"] = int(lengths[longest])
        if fps:
            # Source frames from the run's first miss to the frame it was seen again: the
            # time it was gone, at any stride (each run ends before the span's last frame).
            source = (
                positions[first + ends[longest]] - positions[first + starts[longest]]
            )
            out["longest_gap_s"] = float(source / fps)
    return out


class InputImagePreview(TrackingPreview):
    """Per-frame original frame + the constructed model input for one track.

    For scrubbing the whole video while building an embedding constructor (§14):
    the slider spans every source frame (``part_frame_counts``); on frames where
    the track has no detection the input is ``None`` and only the original frame
    shows. Renders the input with the run's exact ``renderer_spec``, so what you
    see is what the embedding model would receive.
    """

    def __init__(
        self,
        recording_dir: Path | str,
        run_id: str,
        renderer_spec: list[dict],
        *,
        alignment: dict | None = None,
    ) -> None:
        super().__init__(recording_dir, run_id)
        self._renderer = MaskRenderer.from_spec(renderer_spec)
        self._tracking_run_id = run_id
        # §14 alignment: the per-frame heading a run would use, computed once per track
        # the way the run computes it, so the preview turns what the run will turn and
        # shows no input where the run writes no row.
        self._alignment = (
            dict(alignment)
            if alignment and (alignment.get("align") or "none") != "none"
            else None
        )
        self._headings: dict[int, dict[str, dict]] = {}

    def heading_for(self, track_id: int, part_stem: str, frame_idx: int):
        """``(heading_deg, mirrored)`` at a frame, ``None`` when the run would not embed
        it; always ``(None, False)`` on an unaligned preview."""
        if self._alignment is None:
            return None, False
        from detail.analysis.body_axis import SOURCE_NONE
        from detail.pipeline.embedding import _alignment_settings, headings_for_run

        if int(track_id) not in self._headings:
            frame_h, frame_w = self._mask_size()
            per_part = headings_for_run(
                self._dir,
                self._tracking_run_id,
                [int(track_id)],
                1,
                _alignment_settings(self._alignment),
                (frame_h, frame_w),
            )
            self._headings[int(track_id)] = {
                part_id: {
                    int(f): (float(d), int(s), bool(m))
                    for f, d, s, m in zip(
                        h["frame_idx"], h["deg"], h["source"], h["mirrored"]
                    )
                }
                for (_t, part_id), h in per_part.items()
            }
        found = self._headings[int(track_id)].get(part_stem, {}).get(int(frame_idx))
        if found is None or found[1] == SOURCE_NONE:
            return None
        return found[0], found[2]

    def _mask_size(self) -> tuple[int, int]:
        """``(height, width)`` of the detection masks, from the detection run's envelope."""
        from detail.executor import rundir

        track_meta = yaml.safe_load(
            atomic.read_text(
                self._dir / "tracking" / self._tracking_run_id / "track.yaml"
            )
        )
        detection_dir = self._dir / "detections" / track_meta["detection_run_id"]
        frame_h, frame_w = rundir.read_envelope(detection_dir)["mask_resolution"]
        return int(frame_h), int(frame_w)

    def input_at(self, part_index: int, local_frame: int, track_id: int):
        """``(actual_frame_idx, original_rgb, input_image | None)`` for ``track_id``.

        Returns ``None`` if the frame could not be decoded. The detections are for
        the *actual* decoded frame, so the rendered input matches the frame shown
        even when a seek snaps to a keyframe. On an aligned preview a frame the run
        would not embed (no heading within the gap) has no input either.
        """
        result = self.frame(part_index, local_frame, with_masks=True)
        if result is None:
            return None
        actual, rgb, detections = result
        matches = [d for d in detections if d.object_id == int(track_id)]
        if not matches:
            return actual, rgb, None
        aux: dict = {}
        if self._alignment is not None:
            heading = self.heading_for(
                track_id, self._parts[part_index]["stem"], actual
            )
            if heading is None:
                return actual, rgb, None
            aux = {"heading_deg": heading[0], "align_mirrored": heading[1]}
        # Merge every detection this track has on the frame — OR the masks, union the
        # bboxes — so the preview renders exactly what the run embeds; ``embed_one``
        # merges the same way (§14). A single detection is unchanged.
        mask = matches[0].mask
        for det in matches[1:]:
            mask = np.logical_or(mask, det.mask)
        bbox = (
            min(d.bbox[0] for d in matches),
            min(d.bbox[1] for d in matches),
            max(d.bbox[2] for d in matches),
            max(d.bbox[3] for d in matches),
        )
        lead = matches[0]
        ctx = RenderContext(
            frame=rgb,
            mask=mask,
            bbox=bbox,
            image=None,
            frame_idx=actual,
            t_ns=0,
            class_name=lead.class_name,
            confidence=lead.confidence,
            object_id=int(track_id),
            aux=aux,
        )
        return actual, rgb, self._renderer.render(ctx)


def pca_project_2d(matrix: np.ndarray) -> np.ndarray:
    """Experiment ``(N, D)`` onto its top-2 principal components (centered SVD).

    Pure numpy — the lightweight inspection projection, distinct from the §16
    analysis stack. Degenerate inputs (``N`` or ``D`` < 2) are padded to two
    columns so callers always get an ``(N, 2)`` float array.
    """
    if matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centered = matrix.astype(np.float64) - matrix.astype(np.float64).mean(
        axis=0, keepdims=True
    )
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T  # (N, k), k = min(2, rank)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords.astype(np.float32)


class EmbeddingPreview:
    """Read-only inspection of a completed embedding run (§14): run metadata,
    per-track coverage, a 2-D PCA of the embeddings, and the original frame +
    model input behind any point (via the run's stored ``renderer_spec``)."""

    def __init__(self, recording_dir: Path | str, run_id: str) -> None:
        self._dir = Path(recording_dir)
        self._run_dir = self._dir / "embeddings" / run_id
        meta_path = self._run_dir / "embedding.yaml"
        if not meta_path.exists():
            raise ValueError(f"embedding run {run_id!r} not found in recording")
        self._meta = yaml.safe_load(atomic.read_text(meta_path))
        self._input: InputImagePreview | None = None  # lazily built for frame_view

    def metadata(self) -> dict:
        return dict(self._meta)

    def track_ids(self) -> list[int]:
        return sorted(
            int(p.name[len("track_") :])
            for p in self._run_dir.glob("track_*")
            if p.is_dir()
        )

    def _track_parts(self, track_id: int) -> list[Path]:
        return sorted((self._run_dir / f"track_{track_id}").glob("part_*.zarr"))

    def heading_loss(self, tracking: "TrackingPreview") -> dict:
        """Where an aligned run (§14) lost frames to heading: per track, over the tracking
        run's global timeline, ``True`` where the frame was embedded or the track had no
        subject — so ``False`` marks a subject frame with no heading within the gap, hence
        no embedding row. Plus the per-track counts the run recorded.

        ``{"track_ids", "kept": {track_id: bool array}, "total", "coverage"}``. Reads the
        parts' ``frame_idx`` column only; the embedding column is never touched.
        """
        track_ids, presence = tracking.track_presence()
        counts = tracking.part_frame_counts
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
        total = int(sum(counts))
        kept: dict[int, np.ndarray] = {}
        for track_id in track_ids:
            embedded = np.zeros(total, dtype=bool)
            for part_path in self._track_parts(track_id):
                part_index = tracking.part_index_for_stem(part_path.stem)
                if part_index is None:
                    continue
                frames = np.asarray(
                    PartZarrReader(part_path).column("frame_idx")[:], dtype=np.int64
                )
                frames = frames + int(offsets[part_index])
                embedded[frames[frames < total]] = True
            kept[track_id] = embedded | ~np.asarray(presence[track_id], dtype=bool)
        return {
            "track_ids": track_ids,
            "kept": kept,
            "total": total,
            "coverage": self._meta.get("heading_coverage") or {},
        }

    def coverage(self) -> list[dict]:
        """Per track: frames embedded + parts with rows — the "did it run" check."""
        rows = []
        for track_id in self.track_ids():
            n_frames, n_parts = 0, 0
            for part in self._track_parts(track_id):
                count = PartZarrReader(part).n_rows
                n_frames += count
                n_parts += 1 if count else 0
            rows.append(
                {"track_id": track_id, "n_frames": n_frames, "n_parts": n_parts}
            )
        return rows

    def pca_2d(self, track_ids: list[int] | None = None, max_points: int | None = None):
        """``(coords[N,2], track_id[N], locators)`` over the pooled embeddings of
        the given tracks (all by default). Each locator is
        ``(track_id, part_stem, frame_idx, t_ns)`` — one per embedded frame, so a
        scatter point maps straight back to a frame for ``frame_view``.

        ``max_points`` deterministically subsamples to at most that many rows
        **before** the SVD (the worst-scaling step) — for a responsive inspection
        scatter on a long run; ``None`` keeps every row.
        """
        ids = self.track_ids() if track_ids is None else [int(t) for t in track_ids]
        mats: list[np.ndarray] = []
        tracks: list[int] = []
        locators: list[tuple[int, str, int, int]] = []
        for track_id in ids:
            for part in self._track_parts(track_id):
                reader = PartZarrReader(part)
                if reader.n_rows == 0:
                    continue
                mats.append(reader.column("embedding")[:])
                frame_idx = reader.column("frame_idx")[:]
                t_ns = reader.column("t_ns")[:]
                tracks.extend([track_id] * len(frame_idx))
                locators.extend(
                    (track_id, part.stem, int(f), int(t))
                    for f, t in zip(frame_idx, t_ns)
                )
        if not mats:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=int), []
        matrix = np.concatenate(mats, axis=0)
        track_arr = np.asarray(tracks, dtype=int)
        if max_points is not None and len(matrix) > max_points:
            idx = np.sort(
                np.random.default_rng(0).choice(len(matrix), max_points, replace=False)
            )
            matrix = matrix[idx]
            track_arr = track_arr[idx]
            locators = [locators[i] for i in idx]
        coords = pca_project_2d(matrix)
        return coords, track_arr, locators

    def frame_view(self, track_id: int, part_stem: str, frame_idx: int):
        """``(actual_frame_idx, original_rgb, input_image | None)`` behind a point,
        rendered with the run's stored constructor — or None if undecodable."""
        preview = self._input_preview()
        part_index = preview.part_index_for_stem(part_stem)
        if part_index is None:
            return None
        return preview.input_at(part_index, int(frame_idx), int(track_id))

    def frame_position(self, part_stem: str, frame_idx: int) -> tuple[int, int] | None:
        """``(recording_frame, n_frames)`` — where a part's local ``frame_idx`` sits in the
        whole recording, and how many source frames the recording has.

        ``frame_view`` and every locator speak part-local indices; a reader wants the one
        number the scrubbers show. ``None`` for a stem the tracking run does not have.
        """
        preview = self._input_preview()
        part_index = preview.part_index_for_stem(part_stem)
        if part_index is None:
            return None
        counts = preview.part_frame_counts
        return sum(counts[:part_index]) + int(frame_idx), sum(counts)

    def _input_preview(self) -> "InputImagePreview":
        if self._input is None:
            self._input = InputImagePreview(
                self._dir,
                self._meta["tracking_run_id"],
                self._meta["renderer_spec"],
                # §14: an aligned run's clicked frame renders turned, as the run saw it.
                alignment={
                    key: self._meta.get(key)
                    for key in (
                        "align",
                        "keypoint_run_id",
                        "heading_definition",
                        "heading_cutoff",
                        "max_gap_s",
                        "mirror_hold_s",
                        "facing_camera_ratio",
                    )
                },
            )
        return self._input


class StreamPreview:
    """Read-only accessor for previewing an imported stream (Section 3).

    Exposes the canonical events in both stream-local and video-aligned time (the
    manifest ``offset_ns`` applied), grouped for a timeline raster. The frame
    decode for the video pane lives on ``Recording`` (``get_frame`` /
    ``frame_timestamps``); this only owns the event/alignment side.
    """

    def __init__(self, recording_dir: Path | str, name: str) -> None:
        data = recording_store.read_recording(recording_dir)
        streams = {s["name"]: s for s in data.get("streams", [])}
        if name not in streams:
            raise ValueError(f"no stream {name!r} in recording")
        self._manifest = streams[name]
        self._columns, self._rows = streams_store.read_canonical(
            recording_dir, name, dtypes=self._manifest.get("dtypes")
        )
        self._t_col = self._columns.index("t_ns")
        # ``None``, never ``-1``: a sentinel that is also a valid Python index reads the
        # *last* column, so a continuous stream's raster listed one lane per sample, named
        # after that channel's value. `labels()` returning ['2.0', '5.0'] for a sensor trace
        # is not a degraded view, it is a wrong one, and nothing about it looks wrong.
        self._label_col = (
            self._columns.index("label") if "label" in self._columns else None
        )

    @property
    def name(self) -> str:
        return self._manifest["name"]

    @property
    def type(self) -> str:
        return self._manifest["type"]

    @property
    def shape(self) -> str:
        """Temporal shape — event/interval/continuous (Section 3; legacy ``type`` fallback)."""
        return streams_store.stream_shape(self._manifest)

    @property
    def role(self) -> str | None:
        """Semantic role (e.g. ``behavior``), or ``None`` for an untyped signal."""
        return self._manifest.get("role")

    @property
    def offset_ns(self) -> int:
        return int((self._manifest.get("alignment") or {}).get("offset_ns", 0))

    @property
    def has_labels(self) -> bool:
        """Whether this stream carries event labels, and so has a raster to draw.

        A continuous stream does not: its rows are samples and its columns after ``t_ns``
        are channel values. A surface that draws lanes must ask before drawing.
        """
        return self._label_col is not None

    @property
    def channels(self) -> list[str]:
        """The value-column names of a continuous stream (``[]`` for an event stream)."""
        if self.has_labels:
            return []
        return [c for c in self._columns if c not in ("t_ns", "frame")]

    def labels(self) -> list[str]:
        """Distinct event labels in first-seen order (the timeline's lanes).

        Raises:
            ValueError: If the stream carries no ``label`` column.
        """
        self._require_labels()
        seen: list[str] = []
        for row in self._rows:
            label = str(row[self._label_col])
            if label not in seen:
                seen.append(label)
        return seen

    def events(self, aligned: bool = False) -> list[tuple[int, str]]:
        """``(t_ns, label)`` per event. ``aligned=True`` applies the video offset
        (``video_t_ns = stream_t_ns + offset_ns``); otherwise stream-local time.

        Raises:
            ValueError: If the stream carries no ``label`` column.
        """
        self._require_labels()
        offset = self.offset_ns if aligned else 0
        return [
            (int(row[self._t_col]) + offset, str(row[self._label_col]))
            for row in self._rows
        ]

    def times(self, aligned: bool = False) -> list[int]:
        """Every row's ``t_ns``, for a stream of any shape (a continuous one included)."""
        offset = self.offset_ns if aligned else 0
        return [int(row[self._t_col]) + offset for row in self._rows]

    def _require_labels(self) -> None:
        if self._label_col is None:
            raise ValueError(
                f"stream {self.name!r} is {self.shape!r} and carries no 'label' column, "
                f"so it has no events; its channels are {self.channels}"
            )


def _global_to_part(counts: list[int], global_frame: int) -> tuple[int, int]:
    """Map a recording-global frame index to ``(part_index, local_frame)``."""
    g = int(global_frame)
    for index, count in enumerate(counts):
        if g < count:
            return index, g
        g -= count
    raise IndexError(f"frame {global_frame} past the video ({sum(counts)} frames)")


def _square_crop(
    rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    size: int,
    pad: float = 1.3,
    *,
    mask: np.ndarray | None = None,
    tint: tuple[int, int, int] | None = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """A square, subject-centered crop of the raw frame, resized to ``size``×``size``.

    ``bbox`` is normalized ``x1,y1,x2,y2``; the crop is a square of the larger bbox side × ``pad``
    centered on the subject (so the subject stays framed as it moves), zero-padded where it runs
    off the frame edge so the output is always square (no aspect distortion) and a constant size
    (the video writer needs one size for the whole clip).

    ``mask`` + ``tint`` blend the subject's mask **into the crop** rather than into the frame it
    came from. Same picture either way, and it is the difference between blending over ~2M pixels
    and over ~90k: on a 1080p part, tinting the whole frame and then throwing all but the crop
    away was the largest single item in loading a clip.
    """
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 1.0) * pad
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    side_px = max(1, int(round(side)))
    sx1, sy1 = max(0, left), max(0, top)
    sx2, sy2 = min(w, left + side_px), min(h, top + side_px)
    canvas = np.zeros((side_px, side_px, 3), dtype=rgb.dtype)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - top : sy2 - top, sx1 - left : sx2 - left] = rgb[sy1:sy2, sx1:sx2]
        if mask is not None and tint is not None and mask.shape[:2] == (h, w):
            from detail.core.visualize import draw_mask

            mask_canvas = np.zeros((side_px, side_px), dtype=mask.dtype)
            mask_canvas[sy1 - top : sy2 - top, sx1 - left : sx2 - left] = mask[
                sy1:sy2, sx1:sx2
            ]
            canvas = draw_mask(canvas, mask_canvas, tint, alpha=alpha)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


# What a player falls back to when a recording carries too few timestamps to measure a rate
# (a one-frame part). Not a general default: every real recording reports its own.
DEFAULT_PREVIEW_FPS = 30.0


class SubjectNotDetectedError(LookupError):
    """The frame decoded, but the subject has no detection in it, so there is nothing to crop.

    Distinct from a frame that will not decode: the video is fine, tracking lost the subject
    there. A thumbnail that called this "could not decode" sent people looking at the video.
    """


class ClipQueryPreview:
    """Experiment-level read boundary for the shared clip engine (``analysis/clip_query.py``)
    — the API seam the GUI reads through (Tenet 2). Loads subject trajectories, runs an interactive
    preview search, scores example agreement, and renders a scrub frame (source + silhouette); the
    pure matcher lives in ``analysis/``."""

    def __init__(self, experiment_dir: Path | str) -> None:
        self._dir = Path(experiment_dir)
        # (recording_id, embed_run_id) -> (InputImagePreview, source_t_ns) for scrubbing.
        self._scrub: dict[tuple[str, str], tuple] = {}
        # One lock per key, so a build for one recording never blocks a question about
        # another — see `_scrub_ctx`. `_scrub_guard` only guards handing out the locks.
        self._scrub_locks: dict[tuple[str, str], threading.Lock] = {}
        self._scrub_guard = threading.Lock()
        # The clip window's reads, each once per key, under one lock: the keypoints an
        # embedding run read, the onion skin's past masks, a recording's frame size, and a
        # track's kinematics slice (for the centre trail).
        self._clip_guard = threading.Lock()
        self._points: dict[tuple[str, str], object] = {}
        self._mask_histories: dict[tuple[str, str], object] = {}
        self._frame_sizes: dict[str, tuple[int, int] | None] = {}
        self._kinematics: dict[tuple[str, str, int], object] = {}

    def load_databases(
        self, subjects: list[tuple[str, int]], embed_run_id: str
    ) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``(recording_id, track_id) -> (emb, frame_idx, t_ns)`` for the given subjects,
        all from the one ``embed_run_id``."""
        return {
            (sid, tid): load_track_trajectory(
                experiment_store.recording_path(self._dir, sid), embed_run_id, tid
            )
            for sid, tid in subjects
        }

    def track_continuity(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        max_gap_ns: int | None = None,
    ) -> dict:
        """Where a track has embeddings, for a surface that lets you pick clips over it.

        Embedding rows exist only on frames the subject was detected and tracked on (§9 —
        a frame with no detection leaves no row), so a scrubbable video frame need not have
        one. Returns the embedded frames themselves (shade a coverage band by their density),
        the continuous runs the example rule enforces, and how much of the recording is
        covered — the timeline only, never the embeddings.

        Returns
        ``{"frame_idx", "t_ns", "runs": [(first, last)], "n_rows", "n_frames", "coverage",
        "max_gap_ns"}``; ``runs`` are inclusive global-frame ranges. The timeline arrays come
        back so a caller can check spans against it (``clip_query.check_span``, pure) as the
        user scrubs, without re-reading disk per candidate span.
        """
        frame_idx, t_ns = load_track_timeline(
            experiment_store.recording_path(self._dir, recording_id),
            embed_run_id,
            track_id,
        )
        n_frames = self.recording_frame_count(recording_id, embed_run_id)
        return {
            "frame_idx": frame_idx,
            "t_ns": t_ns,
            "runs": clip_query.run_frame_ranges(frame_idx, t_ns, max_gap_ns),
            "n_rows": int(len(frame_idx)),
            "n_frames": n_frames,
            "coverage": (len(frame_idx) / n_frames) if n_frames else 0.0,
            "max_gap_ns": (
                clip_query.default_max_gap_ns(t_ns)
                if max_gap_ns is None
                else int(max_gap_ns)
            ),
        }

    def check_examples(
        self,
        embed_run_id: str,
        examples: Sequence[dict],
        max_gap_ns: int | None = None,
        *,
        require_contiguous: bool = True,
    ) -> list[dict]:
        """Verdict per candidate example clip — the check every picking surface shares.

        Each of ``examples`` is a ``{recording_id, track_id, ...}`` spec spanned either by
        ``start_t_ns``/``end_t_ns`` or by ``start_frame``/``end_frame`` — the same two
        spellings ``ClipQueryModule.validate`` accepts, so a CLI ``--example`` in frames can
        be screened too. Covers every way a clip can be unusable before the search sees it:
        an unknown recording, a track absent from this embed run, a span with no embedded
        frames, and a span crossing a break in tracking. Each subject's timeline loads once
        however many of its clips are checked.

        ``require_contiguous`` is the searching method's answer
        (``ExemplarBounds.requires_contiguous``); pass ``False`` for a method that pools
        exemplar frames rather than matching a path, and a gap-crossing clip comes back
        ``ok`` with its ``check.gaps`` still populated. The conservative default keeps a
        caller that does not know its method from silently screening nothing.

        Returns
        one ``{**spec, "start_t_ns", "end_t_ns", "ok", "reason", "check"}`` per input, in
        order — the span resolved to ``t_ns`` (Tenet 8) whichever spelling came in.
        ``check`` is the ``SpanCheck``, ``None`` when the span never resolved.
        """
        timelines: dict[tuple[str, int], tuple | str] = {}
        out = []
        for spec in examples:
            subject = (spec["recording_id"], int(spec["track_id"]))
            if subject not in timelines:
                timelines[subject] = self._timeline_or_reason(subject, embed_run_id)
            loaded = timelines[subject]
            if isinstance(loaded, str):
                out.append({**spec, "ok": False, "reason": loaded, "check": None})
                continue
            span = self._span_t_ns(spec, *loaded)
            if isinstance(span, str):
                out.append({**spec, "ok": False, "reason": span, "check": None})
                continue
            check = clip_query.check_span(
                *loaded, *span, max_gap_ns, require_contiguous=require_contiguous
            )
            out.append(
                {
                    **spec,
                    "start_t_ns": span[0],
                    "end_t_ns": span[1],
                    "ok": check.ok,
                    "reason": check.reason,
                    "check": check,
                }
            )
        return out

    @staticmethod
    def _span_t_ns(spec: dict, frame_idx: np.ndarray, t_ns: np.ndarray):
        """A spec's span as ``(start_t_ns, end_t_ns)``, or the reason it has none."""
        if spec.get("start_t_ns") is not None and spec.get("end_t_ns") is not None:
            return int(spec["start_t_ns"]), int(spec["end_t_ns"])
        if spec.get("start_frame") is not None and spec.get("end_frame") is not None:
            try:
                return clip_query.frame_span_to_t_ns(
                    frame_idx, t_ns, int(spec["start_frame"]), int(spec["end_frame"])
                )
            except ValueError as exc:
                return str(exc)
        return "needs a (start_t_ns, end_t_ns) or (start_frame, end_frame) span"

    def _timeline_or_reason(self, subject: tuple[str, int], embed_run_id: str):
        """``(frame_idx, t_ns)`` for a subject, or the reason it can't be read."""
        recording_id, track_id = subject
        recording_dir = experiment_store.recording_path(self._dir, recording_id)
        if not recording_dir.is_dir():
            return f"recording {recording_id!r} is not in this experiment"
        try:
            return load_track_timeline(recording_dir, embed_run_id, track_id)
        except (OSError, KeyError, ValueError):
            return (
                f"track {track_id} is not in embedding run {embed_run_id!r} of "
                f"recording {recording_id!r}"
            )

    def _scrub_ctx(self, recording_id: str, embed_run_id: str):
        """The decode context for one (recording, embedding run), built at most once.

        **A cache every caller races is a cache nobody has.** Thumbnail strips start one pool task
        per clip at once, and each task lands here first. The check and the fill used to be two
        unguarded statements, so on a cold strip every concurrent task missed, and every one of
        them built its own ``InputImagePreview`` and concatenated its own copy of the timestamps.
        Ten thumbnails across **two** recordings built the context **ten** times. Worse
        than the duplication is what it does under contention: a build that is quick alone
        inflates to
        seconds when ten threads run it at once.

        Locked **per key**, not globally. This is reached from the UI thread as well
        (``frame_rate`` off a selection change, ``recording_frame_count`` when a player opens), so
        one shared lock would let a worker's build for recording A freeze the window for
        somebody asking about recording B. Per key, a caller only ever waits for the context it
        actually wants — which it would have had to build itself anyway.

        The re-check inside the lock is not belt-and-braces: whoever waited at the door is there
        *because* somebody else was building this key, so by the time they get in it is normally
        done. Dict reads and writes are atomic under the GIL, so the fast path stays lock-free.
        """
        key = (recording_id, embed_run_id)
        ctx = self._scrub.get(key)
        if ctx is not None:
            return ctx
        with self._scrub_guard:
            lock = self._scrub_locks.setdefault(key, threading.Lock())
        with lock:
            ctx = self._scrub.get(key)
            if ctx is None:
                ctx = self._build_scrub_ctx(recording_id, embed_run_id)
                self._scrub[key] = ctx
        return ctx

    def _build_scrub_ctx(self, recording_id: str, embed_run_id: str) -> tuple:
        """Read the run's metadata and timestamps.

        Slow enough on a real recording that it is done once.
        """
        recording_dir = experiment_store.recording_path(self._dir, recording_id)
        meta = _embedding_meta(recording_dir, embed_run_id)
        data = recording_store.read_recording(recording_dir)
        tracking_run_id = meta["tracking_run_id"]
        detection_run_id = {r["id"]: r for r in data.get("tracking_runs", [])}[
            tracking_run_id
        ]["detection_run_id"]
        preview = InputImagePreview(
            recording_dir, tracking_run_id, meta["renderer_spec"]
        )
        # Source frame times concatenated over parts → one array aligned with the
        # scrub frames (every run's rows index the source video).
        parts_t_ns = [
            recording_store.load_active_timestamps(recording_dir, data, index).values
            for index in range(len(preview.part_frame_counts))
        ]
        source_t_ns = (
            np.concatenate(parts_t_ns) if parts_t_ns else np.zeros(0, dtype=np.int64)
        )
        return (preview, source_t_ns)

    def recording_frame_count(self, recording_id: str, embed_run_id: str) -> int:
        """Total scrubbable frames (the detection source's frame count)."""
        preview, _ = self._scrub_ctx(recording_id, embed_run_id)
        return int(sum(preview.part_frame_counts))

    def frame_t_ns(
        self, recording_id: str, embed_run_id: str, global_frame: int
    ) -> int:
        """The ``t_ns`` of a scrub frame — a cheap lookup (no decode) for mark capture."""
        _preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        if global_frame < 0 or global_frame >= len(source_t_ns):
            raise IndexError(f"frame {global_frame} out of range")
        return int(source_t_ns[global_frame])

    def timestamps(self, recording_id: str, embed_run_id: str) -> np.ndarray:
        """Every scrub frame's ``t_ns`` (int64, one per frame), for a player's time readout.

        The array the context already holds, read-only: a caller slices it rather than asking
        ``frame_t_ns`` once per frame, and must not be able to write into the shared cache.
        """
        _preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        view = np.asarray(source_t_ns, dtype=np.int64).view()
        view.flags.writeable = False
        return view

    def frame_rate(self, recording_id: str, embed_run_id: str) -> float:
        """Frames per second of the scrub source, measured from its own timestamps.

        Read rather than assumed: a player that advances on a fixed timer plays every
        recording at whatever rate that timer happens to be, which is only real time by
        coincidence. Uses the median step so a single dropped frame does not set the rate.
        """
        _preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        if len(source_t_ns) < 2:
            return DEFAULT_PREVIEW_FPS
        step = float(np.median(np.diff(np.asarray(source_t_ns, dtype=np.int64))))
        return 1e9 / step if step > 0 else DEFAULT_PREVIEW_FPS

    def t_ns_to_frame(self, recording_id: str, embed_run_id: str, t_ns: int) -> int:
        """The first scrub frame at or after ``t_ns`` (the last frame when past the end).

        Not the nearest: a time between two frames maps to the later one. That is exact for a
        span's inclusive ends (§19 stores ``end_t_ns`` as the last frame's own time) and one
        frame late for any time a caller made up between frames — which is how a one-row pose
        span written with an exclusive end showed every thumbnail one frame late.
        """
        _preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        if len(source_t_ns) == 0:
            return 0
        return int(np.clip(np.searchsorted(source_t_ns, t_ns), 0, len(source_t_ns) - 1))

    def frame_view(
        self, recording_id: str, embed_run_id: str, track_id: int, global_frame: int
    ):
        """``(rgb, silhouette | None, actual_frame, t_ns)`` for a scrub position — the
        source frame plus the pose silhouette the embedder saw (``None`` where the track
        has no detection). ``global_frame`` indexes the detection source's frames.

        Returns ``None`` when the position yields no picture at all, so a caller can tell
        "this frame did not decode" from "this track has no detection here" — they read
        the same in a blank pane but mean different things. A **missing** video file
        raises ``FileNotFoundError`` from below (a video-excluded export); the GUI
        classifies both through ``gui/frame_source.py``.
        """
        preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        t_ns = int(source_t_ns[global_frame]) if global_frame < len(source_t_ns) else 0
        result = preview.input_at(part_index, local, int(track_id))
        if result is None:
            return None
        actual, rgb, silhouette = result
        return rgb, silhouette, int(actual), t_ns

    def source_crop(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        global_frame: int,
        size: int,
    ) -> np.ndarray | None:
        """A ``size``×``size`` RGB square of the **raw source** frame cropped to ``track_id`` —
        the un-overlaid subject view the cluster-video export writes. ``None`` if the frame can't be
        decoded.

        Raises:
            SubjectNotDetectedError: the frame decoded but ``track_id`` has no detection in it.
        """
        return self.source_crop_masked(
            recording_id, embed_run_id, track_id, global_frame, size, tint=None
        )

    def frame_plain(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        global_frame: int,
    ):
        """``(rgb, actual_frame, t_ns)`` — the whole frame, nothing drawn on it.

        The cheap sibling of :meth:`frame_masked`, and the difference is not the drawing: it
        decodes the frame **without its masks**. A mask is COCO RLE per detection, and building
        the picture on top of it costs more than the frame did — about three times this
        path, the difference between fast playback and not reaching it.

        ``track_id`` is taken and unused, so a caller can switch between this and ``frame_masked``
        on a toggle without rewriting the call. That is the whole reason the pair exists: masks
        are worth waiting for when you are judging a segmentation and worth skipping when you are
        watching an animal move.
        """
        preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        result = preview.frame(part_index, local, with_masks=False)
        if result is None:
            return None
        actual, rgb, _detections = result
        t_ns = int(source_t_ns[global_frame]) if global_frame < len(source_t_ns) else 0
        return rgb, int(actual), t_ns

    # --- the clip window (§22): one read, three pictures, the strips ------------------- #
    def clip_frame(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        global_frame: int,
        settings,
        *,
        footage: bool = True,
    ):
        """One frame of a clip, read (``api.clip_views.ClipFrame``): the footage (one decode)
        or none, its detections with masks, the keypoints placed on it, and the trails
        ``settings`` ask for — or ``None`` when the position yields no frame.

        The points are **the keypoint run the embedding used** (``keypoint_run_id`` on its
        ``embedding.yaml``), never a run picked here, so the pose shown is the one the
        clusters were built with. Masks are decoded on every frame whatever the ticks: the
        masks pane needs them.

        ``footage=False`` is for a caller that shows only the masks and the keypoints — the
        enlarged pane, its export, a Raw-off montage tile — and **decodes no video**: the masks
        and points are stored data, and the picture's size is ``recording.yaml``'s.

        Raises:
            ValueError: ``footage=False`` on a recording whose record carries no
                ``video.resolution``: there is no size to draw the panes at without a decode,
                and this never decodes behind a caller's back. The caller decides.
        """
        from detail.api import clip_views as views_api

        preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        if footage:
            result = preview.frame(part_index, local, with_masks=True)
            if result is None:
                return None
            actual, rgb, detections = result
            frame_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        else:
            frame_hw = self._frame_hw(recording_id)
            if frame_hw is None:
                raise ValueError(
                    f"{recording_id}: recording.yaml records no video.resolution, so a "
                    "picture without the footage has no size; read the frame with "
                    "footage=True, or re-import the video so its size is recorded"
                )
            rgb = None
            actual, detections = preview.detections_at(
                part_index, local, mask_hw=frame_hw
            )
        t_ns = int(source_t_ns[global_frame]) if global_frame < len(source_t_ns) else 0
        counts = list(preview.part_frame_counts)
        shown = views_api.source_frame_of(counts, part_index, actual)
        span = (
            max(
                int(
                    round(
                        settings.trail_s * self.frame_rate(recording_id, embed_run_id)
                    )
                ),
                2,
            )
            if settings.trail_s > 0
            else 0
        )

        points = self.clip_points(recording_id, embed_run_id)
        has_points = points.run is not None
        placed = (
            points.at(points.part_stems[part_index], int(actual)) if has_points else {}
        )
        axis = points.axis(int(track_id)) if has_points else None
        # A side view's flip is read only for what draws one: Aligned, or Align pose.
        wants_flip = axis is not None and (settings.aligned or settings.pose_aligned)
        point_trail = None
        centre_trail = None
        if settings.trail and span:
            if has_points:
                names = list(points.skeleton.names)
                wanted = set(
                    names if settings.trail_points is None else settings.trail_points
                )
                kept = np.array([name in wanted for name in names], dtype=bool)
                xy, scores = points.window(
                    int(track_id), counts, max(shown - span, 0), shown
                )
                from detail.core.visualize import KEYPOINT_COLORS, keypoint_roles

                colours = tuple(
                    KEYPOINT_COLORS[role] for role in keypoint_roles(points.skeleton)
                )
                flips = (
                    points.flip_window(
                        int(track_id), counts, max(shown - span, 0), shown
                    )
                    if wants_flip and settings.pose_aligned
                    else None
                )
                point_trail = views_api.PointTrail(xy, scores, colours, kept, flips)
            else:
                centre_trail = self._centre_trail(
                    recording_id,
                    embed_run_id,
                    int(track_id),
                    max(shown - span, 0),
                    shown,
                )
        mask_history = None
        if settings.mask_trail and span:
            mask_history = self._mask_history(recording_id, embed_run_id).layers(
                preview,
                part_index,
                int(track_id),
                list(range(max(int(actual) - span, 0), int(actual))),
                frame_hw,
            )
        return views_api.ClipFrame(
            rgb=rgb,
            frame_hw=frame_hw,
            detections=detections,
            actual=int(actual),
            shown=int(shown),
            t_ns=t_ns,
            placed=placed,
            skeleton=points.skeleton if has_points else None,
            pose_note=points.note,
            axis=axis,
            align_note=points.axis_note,
            point_trail=point_trail,
            centre_trail=centre_trail,
            mask_history=mask_history,
            flip=(
                self._flip(points, int(track_id), part_index, int(actual))
                if wants_flip
                else None
            ),
        )

    @staticmethod
    def _flip(points, track_id: int, part_index: int, actual: int) -> float | None:
        """A side-view embedding's stored flip on this frame (``ClipFrame.flip``), ``None``
        for any other run."""
        found = points.flips(track_id, points.part_stems[part_index], [actual])
        return None if found is None else float(found[0])

    def clip_views(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        global_frame: int,
        settings,
        *,
        footage: bool = True,
    ):
        """The three pictures of one frame the clip window shows
        (``api.clip_views.ClipViews``), from one :meth:`clip_frame`; ``None`` when the
        position yields no frame. With ``footage=False`` the footage picture is ``None``.
        """
        from detail.api import clip_views as views_api

        frame = self.clip_frame(
            recording_id,
            embed_run_id,
            track_id,
            global_frame,
            settings,
            footage=footage,
        )
        if frame is None:
            return None
        picture, masks, pose, _turned = views_api.turn_views(
            frame,
            settings,
            track_id,
            views_api.draw_footage(frame, settings, track_id) if footage else None,
            views_api.draw_masks_pane(frame, settings, track_id),
            views_api.draw_pose_pane(frame, settings, track_id),
        )
        return views_api.ClipViews(
            frame=picture,
            masks=masks,
            pose=pose,
            pose_note=frame.pose_note,
            actual=frame.actual,
            t_ns=frame.t_ns,
            align_note=frame.align_note,
            skeleton=frame.skeleton,
            trail_note=views_api.trail_note(frame, settings),
        )

    def clip_curves(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        first: int,
        last: int,
    ):
        """What the clip's animal is doing over the window ``[first, last]`` (inclusive global
        frames): ``api.clip_curves.read_clip_curves`` with the recording's calibration and,
        when the embedding read keypoints and a heading definition names the body's ends, its
        keypoints over the window for the body's length, bend and turning."""
        from detail.api import clip_curves as curves_api
        from detail.api.recording import Recording

        recording_dir = experiment_store.recording_path(self._dir, recording_id)
        try:
            ppm = Recording(self._dir, recording_id).pixels_per_mm
        except ValueError:  # a calibration that cannot describe this video: no mm
            ppm = None
        points = self.clip_points(recording_id, embed_run_id)
        pose = None
        if points.run is not None:
            axis = points.axis(int(track_id))
            if axis is not None:
                preview, _t = self._scrub_ctx(recording_id, embed_run_id)
                xy, scores = points.window(
                    int(track_id),
                    list(preview.part_frame_counts),
                    int(first),
                    int(last),
                )
                pose = (xy, scores, axis)
        return curves_api.read_clip_curves(
            recording_dir,
            embed_run_id=embed_run_id,
            track_id=int(track_id),
            first=int(first),
            last=int(last),
            pixels_per_mm=ppm,
            pose=pose,
        )

    def clip_points(self, recording_id: str, embed_run_id: str):
        """The keypoints an embedding run read (``api.clip_views.ClipPoints``), opened once
        per (recording, run)."""
        from detail.api.clip_views import ClipPoints

        key = (recording_id, embed_run_id)
        with self._clip_guard:
            found = self._points.get(key)
            if found is None:
                recording_dir = experiment_store.recording_path(self._dir, recording_id)
                found = ClipPoints.of(
                    self._dir,
                    recording_dir,
                    _embedding_meta(recording_dir, embed_run_id),
                    embed_run_id,
                )
                self._points[key] = found
        return found

    def clip_has_points(self, recording_id: str, embed_run_id: str) -> str | None:
        """``None`` when the embedding run read a keypoint run the clip window can draw, else
        why not — for a control that only means something when there are points."""
        return self.clip_points(recording_id, embed_run_id).note

    def _mask_history(self, recording_id: str, embed_run_id: str):
        """The masks pane's onion-skin cache for one (recording, run)."""
        from detail.api.clip_views import MaskHistory

        key = (recording_id, embed_run_id)
        with self._clip_guard:
            found = self._mask_histories.get(key)
            if found is None:
                found = self._mask_histories[key] = MaskHistory()
        return found

    def frame_size(self, recording_id: str) -> tuple[int, int] | None:
        """The footage's ``(height, width)`` as ``recording.yaml`` records it, or ``None``
        when it does not — what :meth:`clip_frame` needs to draw the panes with
        ``footage=False``. A window asks this first and decides; the API never decodes
        behind a caller's back."""
        return self._frame_hw(recording_id)

    def clip_boxes(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        first: int,
        last: int,
    ) -> np.ndarray:
        """The clip's animal's normalized ``(x0, y0, x1, y1)`` box on each global frame
        ``first..last`` inclusive — ``(n, 4)`` float64, a NaN row where it has no detection —
        read off the detection rows (``DetectionPreview.subject_boxes``), never by decoding
        video. What a montage fixes its crop from (``api.clip_montage.crop_box``)."""
        preview, _ = self._scrub_ctx(recording_id, embed_run_id)
        first, last = int(first), int(last)
        out = np.full((max(last - first + 1, 0), 4), np.nan, dtype=np.float64)
        offset = 0
        for part_index, count in enumerate(preview.part_frame_counts):
            lo, hi = max(first, offset), min(last, offset + int(count) - 1)
            if lo <= hi:
                frames, boxes = preview.subject_boxes(
                    part_index, int(track_id), lo - offset, hi - offset
                )
                out[frames + offset - first] = boxes
            offset += int(count)
        return out

    def _frame_hw(self, recording_id: str) -> tuple[int, int] | None:
        """The footage's ``(height, width)`` from ``recording.yaml``'s ``video.resolution``
        (``[width, height]``, every part's, §11), read once; ``None`` when not recorded.
        """
        with self._clip_guard:
            if recording_id not in self._frame_sizes:
                data = recording_store.read_recording(
                    experiment_store.recording_path(self._dir, recording_id)
                )
                resolution = (data.get("video") or {}).get("resolution") or (0, 0)
                width, height = (int(v) for v in resolution[:2])
                self._frame_sizes[recording_id] = (
                    (height, width) if width > 0 and height > 0 else None
                )
            return self._frame_sizes[recording_id]

    def _centre_trail(
        self, recording_id: str, embed_run_id: str, track_id: int, first: int, last: int
    ) -> np.ndarray | None:
        """The subject's centroid over global frames ``[first, last]`` from the kinematics
        sidecar (``frame_slices.from_kinematics``, read once per track), ``(n, 2)`` normalized
        oldest first with a NaN row where two rows are not adjacent at the run's stride, so a
        trail breaks across a gap; ``None`` when the run has no sidecar."""
        from detail.api import frame_slices

        key = (recording_id, embed_run_id, int(track_id))
        with self._clip_guard:
            if key not in self._kinematics:
                self._kinematics[key] = frame_slices.from_kinematics(
                    experiment_store.recording_path(self._dir, recording_id),
                    embed_run_id=embed_run_id,
                    track_id=int(track_id),
                )
            slice_ = self._kinematics[key]
        if slice_ is None or slice_.centroid is None:
            return None
        frame_hw = self._frame_hw(recording_id)
        if frame_hw is None:
            return None
        frame_idx = np.asarray(slice_.frame_idx, dtype=np.int64)
        lo = int(np.searchsorted(frame_idx, int(first), side="left"))
        hi = int(np.searchsorted(frame_idx, int(last), side="right"))
        if hi <= lo:
            return None
        frames = frame_idx[lo:hi]
        centroid = np.asarray(slice_.centroid, dtype=np.float64)[lo:hi] / np.array(
            [frame_hw[1], frame_hw[0]], dtype=np.float64
        )
        rows = [centroid[0]]
        for i in range(1, len(frames)):
            if int(frames[i]) - int(frames[i - 1]) != int(slice_.stride):
                rows.append(np.array([np.nan, np.nan]))
            rows.append(centroid[i])
        return np.asarray(rows, dtype=np.float64)

    def frame_masked(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        global_frame: int,
        *,
        tint: tuple[int, int, int] | None = None,
        alpha: float = 0.35,
    ):
        """``(rgb_with_mask, actual_frame, t_ns)`` — the **whole frame** with the subject's
        mask blended in, or ``None`` when the position yields no picture.

        The framing sibling of :meth:`source_crop_masked`: that one crops to the subject, this
        keeps the arena so a reviewer can see *where* the animal is as well as what it is
        doing. Both blend through ``core/visualize.draw_mask``, so there is one implementation
        of "the subject, marked" and two framings of it rather than two renderers.

        A frame the track has no detection in comes back **unmasked rather than as nothing**:
        the animal is still visible and a judgment about it is still meaningful, where a blank
        pane would read as a decode failure. ``mask_drawn`` in the caller distinguishes them if
        it needs to.
        """
        from detail.core.visualize import MASK_TINT, draw_mask

        preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        result = preview.frame(part_index, local, with_masks=True)
        if result is None:
            return None
        actual, rgb, detections = result
        t_ns = int(source_t_ns[global_frame]) if global_frame < len(source_t_ns) else 0
        det = next((d for d in detections if d.object_id == int(track_id)), None)
        if det is not None and det.mask is not None:
            colour = tuple(int(c) for c in (tint or MASK_TINT))
            rgb = draw_mask(rgb, det.mask, colour, alpha=alpha)
        return rgb, int(actual), t_ns

    def source_crop_masked(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        global_frame: int,
        size: int,
        *,
        tint: tuple[int, int, int] | None,
        alpha: float = 0.35,
    ) -> np.ndarray | None:
        """``source_crop`` with the subject's mask blended in ``tint``, or plain when ``None``.

        The tint is the *caller's* colour, not one this picks: a cluster surface tints every
        thumbnail in that cluster's own colour so a picture, its point in the scatter and its band
        in the ethogram read as one thing. Masks are decoded only when a tint is asked for — they
        are the expensive half of a frame read (COCO RLE per detection), and a plain crop must not
        pay for them.

        Raises:
            SubjectNotDetectedError: the frame decoded but ``track_id`` has no detection in it.
        """
        preview, _ = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        result = preview.frame(part_index, local, with_masks=tint is not None)
        if result is None:
            return None
        actual, rgb, detections = result
        det = next((d for d in detections if d.object_id == int(track_id)), None)
        if det is None or det.bbox is None:
            raise SubjectNotDetectedError(
                f"track {int(track_id)} has no detection at frame {actual} of "
                f"{recording_id}"
            )
        return _square_crop(
            rgb,
            det.bbox,
            size,
            mask=det.mask,
            tint=None if tint is None else tuple(int(c) for c in tint),
            alpha=alpha,
        )

    def source_crops_masked(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        first_frame: int,
        last_frame: int,
        size: int,
        *,
        tint: tuple[int, int, int] | None,
        alpha: float = 0.35,
        step: int = 1,
    ) -> Iterator[np.ndarray]:
        """Every crop of a span, decoded in one pass — ``source_crop_masked`` over a clip.

        Calling the single-frame form in a loop re-opens and re-seeks the container per frame:
        over an order of magnitude slower than here. A span that runs
        past a part boundary is decoded part by part, since a part is a separate file.

        Frames where the track has no detection are skipped rather than yielded blank — the
        subject is genuinely not located there, and a hole in a clip is better than a picture
        of the wrong thing.

        Yields:
            The crops in presentation order.
        """
        preview, _ = self._scrub_ctx(recording_id, embed_run_id)
        counts = preview.part_frame_counts
        first_part, first_local = _global_to_part(counts, first_frame)
        last_part, last_local = _global_to_part(counts, last_frame)
        wanted = int(track_id)
        # Span frames already emitted, so a stride keeps its phase across a part
        # boundary instead of restarting and doubling up a frame there.
        offset = 0
        for part_index in range(first_part, last_part + 1):
            lo = first_local if part_index == first_part else 0
            hi = last_local if part_index == last_part else counts[part_index] - 1
            start = lo + (-offset % step)
            for actual, rgb, detections in preview.frames_range(
                part_index,
                start,
                hi,
                with_masks=tint is not None,
                step=step,
            ):
                det = next((d for d in detections if d.object_id == wanted), None)
                if det is None or det.bbox is None:
                    continue
                yield _square_crop(
                    rgb,
                    det.bbox,
                    size,
                    mask=det.mask,
                    tint=None if tint is None else tuple(int(c) for c in tint),
                    alpha=alpha,
                )
            offset += hi - lo + 1


class QueryEvalPreview:
    """Experiment-level read boundary for the behavior-query eval (§16) — the API seam the
    GUI reads through (Tenet 2). Reads a completed run's metrics + per-clip detail
    (predictions / ground truth) + PR-curve points from ``analysis/<run_id>/``; the
    compute lives in ``pipeline/query_eval``. The scrub-frame panel reuses
    ``ClipQueryPreview.frame_view`` (same embed-run silhouette seam)."""

    def __init__(self, experiment_dir: Path | str) -> None:
        self._dir = Path(experiment_dir)

    def read_metrics(self, run_id: str):
        """The per-behavior ``MetricRow``s of a completed run."""
        from detail.storage import analysis_runs as analysis_store
        from detail.storage import query_eval_csv

        return query_eval_csv.read_metrics_csv(
            analysis_store.analysis_dir(self._dir, run_id) / "metrics.csv"
        )

    def read_predictions(self, run_id: str, behavior: str | None = None):
        """Ranked prediction outcomes (``PredictionRow``s), optionally one behavior."""
        from detail.storage import analysis_runs as analysis_store
        from detail.storage import query_eval_csv

        rows = query_eval_csv.read_predictions_csv(
            analysis_store.analysis_dir(self._dir, run_id) / "predictions.csv"
        )
        return [r for r in rows if behavior is None or r.behavior == behavior]

    def read_ground_truth(self, run_id: str, behavior: str | None = None):
        """Ground-truth outcomes (``GroundTruthRow``s), optionally one behavior."""
        from detail.storage import analysis_runs as analysis_store
        from detail.storage import query_eval_csv

        rows = query_eval_csv.read_ground_truth_csv(
            analysis_store.analysis_dir(self._dir, run_id) / "ground_truth.csv"
        )
        return [r for r in rows if behavior is None or r.behavior == behavior]

    def read_pr(
        self, run_id: str, behavior: str
    ) -> dict[int, list[tuple[float, float]]]:
        """Per-fold ``(recall, precision)`` PR-curve points for one behavior, keyed by
        fold index (``pr/<behavior>/fold_NN.csv``). Empty if the behavior was skipped.
        """
        from detail.storage import analysis_runs as analysis_store
        from detail.storage import query_eval_csv

        pr_dir = analysis_store.analysis_dir(self._dir, run_id) / "pr" / behavior
        if not pr_dir.is_dir():
            return {}
        return {
            int(f.stem.split("_")[1]): query_eval_csv.read_pr_points(f)
            for f in sorted(pr_dir.glob("fold_*.csv"))
        }

    def read_folds(self, run_id: str, behavior: str | None = None):
        """Per-fold ``FoldMetricRow``s of a completed run, optionally one behavior."""
        from detail.storage import analysis_runs as analysis_store
        from detail.storage import query_eval_csv

        rows = query_eval_csv.read_folds_csv(
            analysis_store.analysis_dir(self._dir, run_id) / "folds.csv"
        )
        return [r for r in rows if behavior is None or r.behavior == behavior]

    def read_overlap(self, run_id: str, behavior: str | None = None):
        """Per-overlap-level ``OverlapRow``s (each score beside its null) of a completed
        run, optionally one behavior. Raises ``FileNotFoundError`` for a run recorded
        before the nulls existed."""
        from detail.storage import analysis_runs as analysis_store
        from detail.storage import query_eval_csv

        rows = query_eval_csv.read_overlap_csv(
            analysis_store.analysis_dir(self._dir, run_id) / "overlap.csv"
        )
        return [r for r in rows if behavior is None or r.behavior == behavior]
