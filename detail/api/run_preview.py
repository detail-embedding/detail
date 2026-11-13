"""Read-only, per-frame accessors for previewing a completed run's results.

The GUI renders overlays itself (no annotated video is written, §15), so these
expose just what a frame viewer needs: the source frame to decode, the detections
on it (with their track id, for tracking), and a cheap per-frame count for an
overview. Lazy per-frame reads keep memory flat on hour-long videos — only a
frame-index column is held in memory; boxes/masks/track-ids are sliced per frame.
"""

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detail.analysis import body_axis, clip_query
from detail.core import atomic
from detail.core import yaml_load
from detail.core.dataclasses import Detection, RenderContext
from detail.core.skeleton import Skeleton
from detail.core.utils import decode_mask

from detail.core.utils import (
    coerce_series_key,
    decode_mask,
    parse_series_dirname,
    series_dirname,
    subject_key,
)
from detail.core import visualize
from detail.core.visualize import bgr_to_rgb, draw_keypoints
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import streams as streams_store
from detail.storage.video import read_frame_at, read_frames_range
from detail.storage.zarr_io import (
    DetectionPartReader,
    PartZarrReader,
    kinematics_part_path,
    resolve_assignment_rows,
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
    # §14 pose in the embedding: drawn points or an appended pose block is a different vector.
    "pose",
    "pose_weight",
    "pose_min_score",
    # The order the model was handed the pixels in. Runs before the channel-order fix rendered
    # and embedded BGR frames — a red subject reached the model as blue — and record nothing,
    # which reads as "bgr" (``_recipe_value``), so they never pool with an RGB run.
    "channel_order",
)

#: What a run recorded before the channel-order fix is read as: BGR frames reached the model.
_CHANNEL_ORDER_BEFORE_THE_FIX = "bgr"

#: What a run recorded before alignment (or pose) existed is read as: unaligned, no pose.
_ALIGNMENT_DEFAULTS = {
    "align": "none",
    "heading_definition": None,
    "max_gap_s": 0.5,
    "mirror_hold_s": 1.0,
    "facing_camera_ratio": 0.5,
    "keypoint_track_identity": None,
    "pose": "none",
    "pose_weight": None,
    "pose_min_score": None,
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


@dataclass(frozen=True)
class ClipSource:
    """Where a clip's footage, masks and keypoints come from when no embedding run stands behind
    it — a location export's tiles (§16 "Location plots"). Accepted wherever the clip machinery
    takes an ``embed_run_id``: the embedding run was only ever a lookup for its tracking run and
    its keypoint run, which this names directly.

    ``subject_colors`` (``((object_id, (r, g, b)), ...)``) are the animals a tile shows, each in
    its own colour (:func:`_tile_colors`): the tile draws every one of them, whatever the clip's
    subject, and each leaves its own trail. Hashable, so it keys every cache an embedding run id
    keys.
    """

    tracking_run_id: str
    keypoint_track_run_id: str | None = None
    subject_colors: tuple = ()

    @property
    def subject_ids(self) -> tuple[int, ...]:
        return tuple(int(oid) for oid, _rgb in self.subject_colors)


def _tile_colors(source) -> dict[int, tuple[int, int, int]] | None:
    """``{object_id: RGB}`` when ``source`` is a ``ClipSource`` naming the animals its tile shows
    (a location export's tile), each drawn in its own colour; ``None`` for a clip of one animal,
    drawn in the caller's tint."""
    if isinstance(source, ClipSource) and source.subject_colors:
        return dict(source.subject_colors)
    return None


def _source_meta(recording_dir: Path, key) -> dict:
    """What a clip key says about its source: an embedding run's ``embedding.yaml``, or for a
    ``ClipSource`` the two run ids it names (no renderer, no alignment).
    """
    if isinstance(key, ClipSource):
        return {
            "tracking_run_id": key.tracking_run_id,
            "keypoint_run_id": key.keypoint_track_run_id,
        }
    return _embedding_meta(recording_dir, key)


def _source_name(key) -> str:
    """How a note names a clip's source."""
    if isinstance(key, ClipSource):
        return f"tracking run {key.tracking_run_id}"
    return f"embedding run {key}"


def _embedding_meta(recording_dir: Path, embed_run_id: str) -> dict:
    """The ``embedding.yaml`` for one run (carries lineage + comparability fields)."""
    path = Path(recording_dir) / "embeddings" / embed_run_id / "embedding.yaml"
    if not path.exists():
        raise ValueError(
            f"embedding run {embed_run_id!r} not found in recording {Path(recording_dir).name!r}"
        )
    return yaml_load.safe_load(atomic.read_text(path))


def load_track_kinematics(
    recording_dir: Path | str, embed_run_id: str, track_id: int
) -> dict | None:
    """A track's per-frame geometry (§14 ``kinematics/``), or ``None`` if it was never written.

    ``{frame_idx, t_ns, cx_norm, cy_norm, mask_area_norm, bbox_norm}``, concatenated over parts
    in frame order — the same rows and the same order as
    :func:`load_track_trajectory`, without loading the embedding column beside it.

    **``None``, not an exception.** A run made before kinematics existed simply has none, which is
    a state a caller renders ("re-run embedding, or backfill") rather than an error. Concatenating
    per part rather than reading one flat store is what keeps a row with its own part: `det_row`
    restarts at zero in every part, so anything flattened first and indexed after resolves the
    wrong detection.
    """
    run_dir = Path(recording_dir) / "embeddings" / embed_run_id / f"track_{track_id}"
    columns = ("frame_idx", "t_ns", "cx_norm", "cy_norm", "mask_area_norm", "bbox_norm")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    for part in sorted(run_dir.glob("part_*.zarr")):
        sidecar = kinematics_part_path(part)
        if not sidecar.exists():
            continue
        reader = PartZarrReader(sidecar)
        if reader.n_rows == 0:
            continue
        for name in columns:
            chunks[name].append(np.asarray(reader.column(name)[: reader.n_rows]))
    if not chunks["frame_idx"]:
        return None
    return {name: np.concatenate(chunks[name]) for name in columns}


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


def load_track_trajectory(
    recording_dir: Path | str, embed_run_id: str, track_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate a track's per-part embedding stores into one ``(M, D)`` trajectory.

    Detail stores ``(M, D)`` per track already (no rotation axis), so this is just the
    ordered concatenation of ``embeddings/<run>/track_<tid>/part_*.zarr``.

    Returns
    (emb (M, D) float32, frame_idx (M,) int32, t_ns (M,) int64) in frame order

    Raises
    ValueError: the track has no embedded rows in this run
    """
    run_dir = Path(recording_dir) / "embeddings" / embed_run_id / f"track_{track_id}"
    embs: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for part in sorted(run_dir.glob("part_*.zarr")):
        reader = PartZarrReader(part)
        if reader.n_rows == 0:
            continue
        embs.append(
            np.asarray(reader.column("embedding")[: reader.n_rows], dtype=np.float32)
        )
        frames.append(np.asarray(reader.column("frame_idx")[: reader.n_rows]))
        times.append(np.asarray(reader.column("t_ns")[: reader.n_rows]))
    if not embs:
        raise ValueError(
            f"no embedded rows for track {track_id} in embedding run {embed_run_id!r}"
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
    (frame_idx (M,) int32, t_ns (M,) int64) in frame order

    Raises
    ValueError: the track has no embedded rows in this run
    """
    run_dir = Path(recording_dir) / "embeddings" / embed_run_id / f"track_{track_id}"
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for part in sorted(run_dir.glob("part_*.zarr")):
        reader = PartZarrReader(part)
        if reader.n_rows == 0:
            continue
        frames.append(np.asarray(reader.column("frame_idx")[: reader.n_rows]))
        times.append(np.asarray(reader.column("t_ns")[: reader.n_rows]))
    if not frames:
        raise ValueError(
            f"no embedded rows for track {track_id} in embedding run {embed_run_id!r}"
        )
    return np.concatenate(frames), np.concatenate(times)


def embedding_track_ids(recording_dir: Path | str, embed_run_id: str) -> list[int]:
    """Selectable series in one embedding run (from its ``embedding.yaml``)."""
    meta = _embedding_meta(Path(recording_dir), embed_run_id)
    return [coerce_series_key(t) for t in meta["track_ids"]]


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
    class_name: str | None = None,
) -> list[dict]:
    """Per-part readers for a detection run, optionally paired with a tracking
    run's per-part ``object_id`` assignments (parallel to the detection rows).

    ``every_n_frames`` is the run's cadence, carried through so a viewer can step between
    the frames the run actually processed: at a stride, a frame with no row may simply
    never have been looked at, which is not the same as "the model found nothing".

    ``class_name`` is the *tracking run's own* class filter (§13), if it had one — not
    optional to pass when pairing with an assignment: a class-filtered tracker writes
    one assignment row per *matching* detection row, not one per detection row overall
    (``anchor_tracker`` etc. build ``object_ids`` from ``filter_by_class``'d data). Row i
    of the assignment is then the i-th *filtered* detection, not the i-th detection —
    reading "the first N rows" of the raw, unfiltered reader silently pairs the wrong
    rows together, which reads as tracking dropping a whole trailing span of an
    otherwise fully-detected video. ``raw_rows`` (below) is what recovers the real
    detection row behind each filtered position.
    """
    run_dir = recording_dir / "detections" / detection_run_id
    mask_hw = _encoded_mask_hw(run_dir)
    return [
        _build_part(
            recording_dir,
            data,
            detection_run_id,
            i,
            every_n_frames,
            assignment_dir,
            class_name=class_name,
            mask_hw=mask_hw,
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
    class_name: str | None = None,
    mask_hw: tuple[int, int] | None = None,
    timestamps: np.ndarray | None = None,
) -> dict:
    """One part of :func:`_build_parts` — only its own video resolved, only its own map read.

    For a caller that needs one part (a preview's mask build): building every part resolved
    every part's video, so one part whose video is not on this machine failed a build of
    another, and re-parsed every part's timestamp CSV per call. ``timestamps`` is the part's
    map when the caller already holds it (``load_active_timestamps_cached``). ``class_name``
    is the tracking run's own class filter, as in :func:`_build_parts`; ``mask_hw`` the size
    its detection run encoded masks at, read from the run when not given."""
    file = recording_store.video_files(data)[part_index]
    stem = recording_store.part_stem(file)
    run_dir = recording_dir / "detections" / detection_run_id
    if mask_hw is None:
        mask_hw = _encoded_mask_hw(run_dir)
    zarr_path = run_dir / f"{stem}.zarr"
    reader = DetectionPartReader(zarr_path) if zarr_path.exists() else None
    assignment = None
    if assignment_dir is not None:
        assignment_path = assignment_dir / f"{stem}.zarr"
        if assignment_path.exists():
            assignment = PartZarrReader(assignment_path)
    # raw_rows[k] is the reader's own row index behind filtered position k — the
    # identity map when this run had no class filter, so the unfiltered path below
    # is unchanged (zarr_io.resolve_assignment_rows).
    raw_rows = (
        resolve_assignment_rows(reader, assignment, class_name)
        if reader is not None
        else np.empty(0, dtype=np.int64)
    )
    n_rows = len(raw_rows)
    frame_idx = (
        np.asarray(reader.column("frame_idx")[: reader.n_rows])[raw_rows]
        if reader is not None and n_rows
        else np.empty(0, dtype=np.int64)
    )
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
        "raw_rows": raw_rows,
        "n_rows": n_rows,
        "n_frames": len(timestamps),
        "timestamps": timestamps,
        "assignment": assignment,
        "every_n_frames": max(1, int(every_n_frames or 1)),
        # The size the run's mask RLEs were encoded at (`decode_mask(encoded=)`).
        "mask_hw": mask_hw,
    }


def _encoded_mask_hw(run_dir: Path) -> tuple[int, int] | None:
    """``(height, width)`` a detection run encoded its masks at (its envelope's
    ``mask_resolution``); ``None`` when it does not say, and masks decode as asked."""
    from detail.executor import rundir

    try:
        hw = rundir.read_envelope(run_dir).get("mask_resolution")
    except (FileNotFoundError, ValueError, OSError):
        return None
    return (int(hw[0]), int(hw[1])) if hw else None


def locate_frame(part_frame_counts: Sequence[int], frame: int) -> tuple[int, int]:
    """A recording-**global** frame number as ``(part_index, local_frame)``.

    Clamped to the last frame of the last part rather than raising: a scrub bar and an
    export's ``--frames`` both hand this a number a caller already validated is in
    range, and a run whose last part is shorter than expected should still resolve to
    something on screen rather than crash a widget mid-drag.
    """
    for part, count in enumerate(part_frame_counts):
        if frame < count:
            return part, frame
        frame -= count
    last = len(part_frame_counts) - 1
    return max(last, 0), max((part_frame_counts[-1] if part_frame_counts else 1) - 1, 0)


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

    @property
    def frames_with_detections(self) -> int:
        """How many frames carry at least one detection, across every part."""
        return sum(len(np.unique(part["frame_idx"])) for part in self._parts)

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

    def blank_frame(
        self, part_index: int, local_frame: int, frame_hw: tuple[int, int]
    ) -> tuple[int, np.ndarray, list[Detection]]:
        """:meth:`frame`'s ``(actual_frame_idx, rgb, detections)`` for a caller that will not
        show the footage: a white ``frame_hw`` canvas in place of the picture, the detections
        (masks decoded at ``frame_hw``) of ``local_frame`` itself, and **no video opened**.

        ``local_frame`` is what :meth:`frame` lands on whenever the part's timestamp map
        describes its video, which is the map every surface passes (``FrameCursor``); the
        packet-PTS fallback, which can snap to a keyframe, is never reached without a decode.
        """
        part = self._parts[part_index]
        canvas = np.full((int(frame_hw[0]), int(frame_hw[1]), 3), 255, dtype=np.uint8)
        actual = int(local_frame)
        return actual, canvas, self._detections_at(part, actual, canvas.shape[:2])

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

    def _detections_at(self, part: dict, frame: int, mask_size=None) -> list[Detection]:
        frame_idx = part["frame_idx"]
        if part["reader"] is None or not len(frame_idx):
            return []
        lo = int(np.searchsorted(frame_idx, frame, "left"))
        hi = int(np.searchsorted(frame_idx, frame, "right"))
        if lo >= hi:
            return []
        rows = self._read_rows(part, lo, hi, with_masks=mask_size is not None)
        return _detections_from_rows(
            rows, 0, hi - lo, mask_size, encoded_hw=part.get("mask_hw")
        )

    #: Rows (filtered positions) per cached block: about one chunk of the ``mask_rle`` column
    #: (§9's 1,000 rows), which a read of any one of its rows decompresses whole.
    ROW_BLOCK = 1024
    #: Blocks one preview keeps. A block is its rows' boxes, scores and RLE strings — about a
    #: megabyte at 1080p — so a montage holding a preview per recording stays in the tens of MB.
    ROW_BLOCKS_KEPT = 4

    def _read_rows(self, part: dict, lo: int, hi: int, *, with_masks: bool) -> dict:
        """The columns of rows ``lo:hi`` (positions in the part's filtered space).

        Each column read is its own zarr call, and zarr v3's async layer charges a fixed
        overhead for one however few rows it returns — five of them per frame cost more
        than decoding the frame, drawing on it and
        encoding it together. A caller that wants a span reads it here in one go.

        **A short span is served from a cached block of :data:`ROW_BLOCK` rows**, kept per
        ``(part stem, block, with_masks)``. A surface that walks a clip asks for one frame's
        rows and then the few frames before it (the mask trail), step after step, and each ask
        decompressed the same ``mask_rle`` chunk again, which was much of a montage's pass.
        ``with_masks`` is part of the key so a caller that asked for no masks decompresses
        none. The rows of a part are fixed when the preview is built, so a block never goes
        stale; a span longer than a block is read directly.
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

    def _row_block(self, part: dict, block: int, with_masks: bool) -> dict:
        """Rows ``[block * ROW_BLOCK, (block + 1) * ROW_BLOCK)`` of ``part``, read once and
        kept in a small LRU (several threads may draw from one preview). Keyed on the part's
        stem — stable for the preview's life — never on ``id(part)``, which a rebuilt part
        list would silently reuse for different rows.
        """
        from collections import OrderedDict

        kept = self.__dict__.setdefault("_row_blocks", OrderedDict())
        guard = self.__dict__.setdefault("_row_blocks_guard", threading.Lock())
        key = (str(part["stem"]), int(block), bool(with_masks))
        with guard:
            found = kept.get(key)
            if found is not None:
                kept.move_to_end(key)
                return found
        start = int(block) * self.ROW_BLOCK
        stop = min(start + self.ROW_BLOCK, len(part["raw_rows"]))
        found = self._read_rows_now(part, start, stop, with_masks=with_masks)
        with guard:
            kept[key] = found
            while len(kept) > self.ROW_BLOCKS_KEPT:
                kept.popitem(last=False)
        return found

    def _read_rows_now(self, part: dict, lo: int, hi: int, *, with_masks: bool) -> dict:
        """:meth:`_read_rows` straight from the zarr columns."""
        # part["frame_idx"] is positioned in *filtered* space (§13's class filter, if
        # this run had one) — raw_rows maps each filtered position back to the
        # reader's own row, the identity map when there was no filter. Reading
        # reader.column(name)[lo:hi] directly would be a contiguous slice of the
        # *unfiltered* reader, which is a different — and wrong — set of rows the
        # moment a class filter narrowed what the tracker actually saw.
        reader = part["reader"]
        raw = part["raw_rows"][lo:hi]
        assignment = part.get("assignment")
        return {
            "bbox": reader.column("bbox")[raw],
            "class_name": reader.column("class_name")[raw],
            "confidence": reader.column("confidence")[raw],
            "mask_rle": reader.column("mask_rle")[raw] if with_masks else None,
            "object_id": (
                assignment.column("object_id")[lo:hi]
                if assignment is not None
                else None
            ),
        }

    def detection_blocks(
        self, part_index: int, mask_size=None, *, block: int = 1024
    ) -> "DetectionBlocks":
        """A sequential reader of one part's detections, ``block`` frames per read."""
        return DetectionBlocks(self, self._parts[part_index], mask_size, block)


def _detections_from_rows(
    rows: dict, first: int, stop: int, mask_size=None, *, encoded_hw=None
) -> list[Detection]:
    """``Detection``s for rows ``first:stop`` of what ``DetectionPreview._read_rows`` read.

    A mask is decoded here, one row at a time, because a span's worth of decoded masks is a
    frame-sized array each: a thousand frames of 1080p is gigabytes, where their RLE strings
    are kilobytes.

    An RLE only decodes at the size it was encoded at (``encoded_hw``, the run's
    ``mask_resolution``); decoded at any other it comes out striped and repeated. So a mask
    asked for at another size (the footage's own, when a video decodes larger than the
    detection run saw it) is decoded at its own and resized, nearest-neighbour.
    """
    if first >= stop:
        return []
    object_id = rows["object_id"]
    detections = []
    for j in range(first, stop):
        oid = int(object_id[j]) if object_id is not None else None
        if oid is not None and oid < 0:
            continue  # detection the tracker left unassigned (filtered)
        detections.append(
            Detection(
                mask=(
                    decode_mask(str(rows["mask_rle"][j]), mask_size, encoded=encoded_hw)
                    if mask_size is not None
                    else None
                ),
                bbox=tuple(float(v) for v in rows["bbox"][j]),
                class_name=str(rows["class_name"][j]),
                confidence=float(rows["confidence"][j]),
                object_id=oid,
            )
        )
    return detections


class DetectionBlocks:
    """One part's detections for a caller that walks its frames in order, read a block at a time.

    ``at(frame)`` is what ``DetectionPreview._detections_at`` answers, without paying five zarr
    reads for each frame: the columns of the next ``block`` frames are read once and a frame is
    sliced out of them. Frames may be asked for in any order, but only a forward walk is cheap —
    a frame outside the block in hand reads the block that starts at it.
    """

    def __init__(
        self, preview: DetectionPreview, part: dict, mask_size, block: int
    ) -> None:
        self._preview = preview
        self._part = part
        self._mask_size = mask_size
        self._encoded_hw = part.get("mask_hw")
        self._block = max(1, int(block))
        self._first = self._stop = 0
        self._frames = np.zeros(0, dtype=np.int64)
        self._rows: dict = {}

    def at(self, frame: int) -> list[Detection]:
        frame_idx = self._part["frame_idx"]
        if self._part["reader"] is None or not len(frame_idx):
            return []
        if not self._first <= frame < self._stop:
            self._load(frame)
        lo = int(np.searchsorted(self._frames, frame, "left"))
        hi = int(np.searchsorted(self._frames, frame, "right"))
        return _detections_from_rows(
            self._rows, lo, hi, self._mask_size, encoded_hw=self._encoded_hw
        )

    def _load(self, first: int) -> None:
        frame_idx = self._part["frame_idx"]
        stop = first + self._block
        lo = int(np.searchsorted(frame_idx, first, "left"))
        hi = int(np.searchsorted(frame_idx, stop, "left"))
        self._first, self._stop = first, stop
        self._frames = np.asarray(frame_idx[lo:hi])
        self._rows = (
            self._preview._read_rows(
                self._part, lo, hi, with_masks=self._mask_size is not None
            )
            if hi > lo
            else {}
        )


class TrackingPreview(DetectionPreview):
    """Per-frame overlays for a tracking run: the detection boxes it tracked, each
    carrying its track id (``object_id``); untracked detections are dropped."""

    def __init__(self, recording_dir: Path | str, run_id: str) -> None:
        self._dir = Path(recording_dir)
        data = recording_store.read_recording(self._dir)
        entry = {r["id"]: r for r in data.get("tracking_runs", [])}.get(run_id)
        if entry is None:
            raise ValueError(f"tracking run {run_id!r} not found in recording")
        detection_run_id = entry["detection_run_id"]
        detection_entry = {r["id"]: r for r in data.get("detection_runs", [])}.get(
            detection_run_id, {}
        )
        assignment_dir = self._dir / "tracking" / run_id / "assignments"
        recording_store.raise_on_legacy_downsampling(detection_entry)
        # The row filter must be the class this run's own tracker actually applied
        # (track.yaml), not the record's `class_name` — a track_merge run gives that
        # record field a joined identity label (e.g. "black mouse+white mouse", §13)
        # for display, while its own track.yaml correctly keeps `None` ("no single
        # class filter applies"). Filtering detection rows by the joined label matches
        # nothing and silently reads as an empty run.
        track_yaml = yaml_load.safe_load(
            atomic.read_text(self._dir / "tracking" / run_id / "track.yaml")
        )
        self._parts = _build_parts(
            self._dir,
            data,
            detection_run_id,
            int(
                entry.get("every_n_frames")
                or detection_entry.get("every_n_frames")
                or 1
            ),
            assignment_dir,
            track_yaml.get("class_name"),
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
            n_rows = part["n_rows"]
            if reader is not None and assignment is not None and n_rows:
                frame_idx = part["frame_idx"]  # already bounded to n_rows
                object_id = assignment.column("object_id")[:n_rows]
                for track_id in np.unique(object_id[object_id >= 0]):
                    row = presence.setdefault(
                        int(track_id), np.zeros(total, dtype=bool)
                    )
                    row[offset + frame_idx[object_id == track_id]] = True
            offset += part["n_frames"]
        return sorted(presence), presence


def track_coverage(present: np.ndarray, fps: float | None = None) -> dict:
    """How much of the timeline this track actually holds, and the shape of what it misses.

    ``n_gaps`` in ``tracks.zarr`` is ``(last - first + 1) - n_present``: a count of missing
    *frames*. A thousand one-frame losses and one thousand-frame loss are the same number
    there and are completely different problems — the first is a per-frame gate rejecting
    the subject, the second is the animal leaving or the tracker losing it. So this reports
    the **runs**: how many, and the longest.

    Gaps are counted **inside the track's own span** (first present frame to last). Frames
    before it appears and after it leaves are not gaps in it; counting them would make a
    track that covers the back half of a recording look catastrophically broken.
    """
    present = np.asarray(present, dtype=bool)
    total = int(present.size)
    covered = int(present.sum())
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
    where = np.flatnonzero(present)
    first, last = int(where[0]), int(where[-1])
    out["first_frame"], out["last_frame"] = first, last
    missing = ~present[first : last + 1]
    if missing.any():
        edges = np.flatnonzero(
            np.concatenate(([False], missing, [False]))[1:]
            != np.concatenate(([False], missing, [False]))[:-1]
        )
        lengths = edges[1::2] - edges[::2]
        out["n_gaps"] = int(lengths.size)
        out["longest_gap"] = int(lengths.max())
        if fps:
            out["longest_gap_s"] = float(out["longest_gap"] / fps)
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
        recorded_turns: Path | str | None = None,
    ) -> None:
        super().__init__(recording_dir, run_id)
        self._renderer = MaskRenderer.from_spec(renderer_spec)
        self._tracking_run_id = run_id
        # An embedding run aligned by the stage before §26 read its axis from a ``kp_NN`` run
        # that §26 has no reader for the heading of; it recorded the turn it applied on every
        # row (``align_deg`` / ``align_mirrored``), so its preview replays those rather than
        # computing one. The run's own directory; None for every other preview.
        self._recorded_turns = None if recorded_turns is None else Path(recorded_turns)
        self._turns: dict[tuple[int, str], dict[int, tuple[float, bool]]] = {}
        # §14 alignment: the per-frame heading a run would use, computed once per track
        # the way the run computes it, so the preview turns what the run will turn and
        # shows no input where the run writes no row.
        self._alignment = (
            dict(alignment)
            if alignment and (alignment.get("align") or "none") != "none"
            else None
        )
        self._headings: dict[int, dict[str, dict]] = {}
        # §14 pose: a drawn pose reads each frame's points from the keypoint track run, the
        # way the worker hands them to the ``draw_pose`` step. ``append`` changes the vector,
        # not the image, so its preview is the mask's.
        self._pose_run = None
        if (
            alignment
            and alignment.get("pose") == "draw"
            and alignment.get("keypoint_run_id")
        ):
            from detail.storage.keypoint_runs import KeypointTrackRun

            self._pose_run = KeypointTrackRun(
                self._dir, str(alignment["keypoint_run_id"])
            )

    #: The embedding.yaml keys an aligned run's preview needs to turn a frame as the run did.
    _ALIGNMENT_KEYS = (
        "align",
        "keypoint_run_id",
        "heading_definition",
        "heading_cutoff",
        "max_gap_s",
        "mirror_hold_s",
        "facing_camera_ratio",
        "pose",
    )

    @classmethod
    def for_embedding_run(
        cls, recording_dir: Path | str, meta: dict
    ) -> "InputImagePreview":
        """The inputs a finished embedding run embedded, from its ``embedding.yaml`` — its
        tracking run, its stored spec, and its alignment (§14). The one way to re-render a run,
        so no reader shows an aligned run's frames unaligned."""
        from detail.storage.keypoint_runs import is_legacy_run_id

        legacy = (meta.get("align") or "none") != "none" and is_legacy_run_id(
            str(meta.get("keypoint_run_id") or "")
        )
        return cls(
            recording_dir,
            meta["tracking_run_id"],
            meta["renderer_spec"],
            alignment={key: meta.get(key) for key in cls._ALIGNMENT_KEYS},
            recorded_turns=(
                Path(recording_dir) / "embeddings" / str(meta["id"]) if legacy else None
            ),
        )

    def heading_for(self, track_id: int, part_stem: str, frame_idx: int):
        """``(heading_deg, mirrored)`` at a frame, ``None`` when the run would not embed
        it; always ``(None, False)`` on an unaligned preview."""
        if self._alignment is None:
            return None, False
        if self._recorded_turns is not None:
            return self._recorded_turn(int(track_id), part_stem, int(frame_idx))
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

    def _recorded_turn(self, track_id: int, part_stem: str, frame_idx: int):
        """``(deg, mirrored)`` the run recorded for this frame, or None where it wrote no row
        (a frame it did not embed shows no input, as on every aligned preview)."""
        key = (track_id, part_stem)
        if key not in self._turns:
            path = self._recorded_turns / f"track_{track_id}" / f"{part_stem}.zarr"
            turns: dict[int, tuple[float, bool]] = {}
            if path.is_dir():
                columns = PartZarrReader(path).read()
                deg = columns.get("align_deg", columns.get("heading_deg"))
                mirrored = columns.get("align_mirrored")
                for i, frame in enumerate(np.asarray(columns["frame_idx"]).tolist()):
                    turns[int(frame)] = (
                        float(deg[i]) if deg is not None else float("nan"),
                        bool(mirrored[i]) if mirrored is not None else False,
                    )
            self._turns[key] = turns
        return self._turns[key].get(frame_idx)

    def _mask_size(self) -> tuple[int, int]:
        """``(height, width)`` of the detection masks, from the detection run's envelope."""
        from detail.executor import rundir

        track_meta = yaml_load.safe_load(
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
        if self._pose_run is not None:
            placed = self._pose_run.at(
                int(track_id), self._parts[part_index]["stem"], int(actual)
            )
            if placed is not None:
                aux["pose"] = {"xy": placed[0], "score": placed[1]}
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
        t_ns = int(self._parts[part_index]["timestamps"][actual])
        ctx = RenderContext(
            frame=rgb,
            mask=mask,
            bbox=bbox,
            image=None,
            frame_idx=actual,
            t_ns=t_ns,
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
        self._meta = yaml_load.safe_load(atomic.read_text(meta_path))
        self._input: InputImagePreview | None = None  # lazily built for frame_view

    def metadata(self) -> dict:
        return dict(self._meta)

    def track_ids(self) -> list[int]:
        """The series actually on disk."""
        return sorted(
            parse_series_dirname(p.name)
            for p in self._run_dir.glob("track_*")
            if p.is_dir()
        )

    def _track_parts(self, track_id: int) -> list[Path]:
        return sorted((self._run_dir / series_dirname(track_id)).glob("part_*.zarr"))

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
        ids = (
            self.track_ids()
            if track_ids is None
            else [coerce_series_key(t) for t in track_ids]
        )
        mats: list[np.ndarray] = []
        tracks: list[int] = []
        locators: list[tuple[int, str, int, int]] = []
        for track_id in ids:
            for part in self._track_parts(track_id):
                reader = PartZarrReader(part)
                if reader.n_rows == 0:
                    continue
                mats.append(reader.column("embedding")[: reader.n_rows])
                frame_idx = reader.column("frame_idx")[: reader.n_rows]
                t_ns = reader.column("t_ns")[: reader.n_rows]
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
            # §14: an aligned run's clicked frame renders turned, as the run saw it.
            self._input = InputImagePreview.for_embedding_run(self._dir, self._meta)
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


def _union_detection(detections: list[Detection]) -> Detection | None:
    """A synthetic combined detection standing in for every animal in the frame — how a
    location export's tile of several animals (:func:`_tile_colors`) is framed by
    ``source_crop_masked``. ``None`` when the frame has no detections at all.
    """
    boxes = [d.bbox for d in detections if d.bbox is not None]
    if not boxes:
        return None
    masks = [d.mask for d in detections if d.mask is not None]
    union = None
    for mask in masks:
        union = mask.copy() if union is None else union | mask
    return Detection(
        mask=union,
        bbox=(
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ),
        class_name="",
        confidence=1.0,
        object_id=None,
    )


def _colour_layers(
    detections: list[Detection], colors: dict[int, tuple[int, int, int]]
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Each animal of a location export's tile as ``(mask, RGB)``, in its own colour
    (:func:`_tile_colors`). An animal with no colour of its own takes the usual tint, so nothing
    in the frame goes unmarked.
    """
    from detail.core.visualize import MASK_TINT

    return [
        (
            d.mask,
            (
                MASK_TINT
                if d.object_id is None
                else colors.get(int(d.object_id), MASK_TINT)
            ),
        )
        for d in detections
        if d.mask is not None
    ]


def _square_crop(
    rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    size: int,
    pad: float = 1.3,
    *,
    mask: np.ndarray | None = None,
    tint: tuple[int, int, int] | None = None,
    alpha: float = 0.35,
    layers: Sequence[tuple[np.ndarray, tuple[int, int, int]]] = (),
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

    ``layers`` are further ``(mask, tint)`` pairs blended the same way, one after another: a
    location export's tile of several animals, each in its own colour."""
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
        drawn = [(mask, tint), *layers] if tint is not None else list(layers)
        for one, colour in drawn:
            if one is None or colour is None or one.shape[:2] != (h, w):
                continue
            from detail.core.visualize import draw_mask

            mask_canvas = np.zeros((side_px, side_px), dtype=one.dtype)
            mask_canvas[sy1 - top : sy2 - top, sx1 - left : sx2 - left] = one[
                sy1:sy2, sx1:sx2
            ]
            canvas = draw_mask(canvas, mask_canvas, colour, alpha=alpha)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


# What a player falls back to when a recording carries too few timestamps to measure a rate
# (a one-frame part). Not a general default: every real recording reports its own.
DEFAULT_PREVIEW_FPS = 30.0


class SubjectNotDetectedError(LookupError):
    """The frame decoded, but the subject has no detection in it, so there is nothing to crop.

    Distinct from a frame that will not decode: the video is fine, tracking lost the subject
    there. A thumbnail that called this "could not decode" sent people looking at the video.
    """


#: How far back the clip popup's trails reach: the point trails, the box-centre trail and the
#: masks pane's onion skin all show this much of the animal's past.
CLIP_TRAIL_S = 0.5
#: The onion skin's most opaque layer (the frame just before the one shown); older layers
#: fade linearly towards nothing — ``visualize``'s one fade rule, which draws the trails.
_MASK_TRAIL_ALPHA = visualize.ONION_ALPHA
#: The trail drawn without keypoints: the animal's box centre, in white.
_CENTRE_TRAIL = (255, 255, 255)


class _MaskHistory:
    """The clip's animal's masks on the frames before the one shown, for the masks pane's
    onion skin — **read from the detection run's rows, never by decoding video**, decoded from
    their RLE once, cropped to the mask, and kept for the frames playback asks for again.

    Bounded (an LRU of :attr:`capacity` frames), so a long playback holds a few hundred small
    crops rather than a frame-sized array for every frame it passed.
    """

    capacity = 256

    def __init__(self) -> None:
        from collections import OrderedDict

        self._kept: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def layers(
        self,
        preview: "DetectionPreview",
        part_index: int,
        subject: int | None,
        local_frames: list[int],
        size: tuple[int, int],
    ) -> list:
        """One ``(y0, x0, crop)`` or None per frame of ``local_frames`` (one part's):
        ``subject``'s mask, or every animal's together when ``None``."""
        subject = None if subject is None else subject_key(subject)
        key = (int(part_index), str(subject), tuple(int(v) for v in size))
        found: dict[int, object] = {}
        with self._lock:
            for frame in local_frames:
                if (key, frame) in self._kept:
                    self._kept.move_to_end((key, frame))
                    found[frame] = self._kept[(key, frame)]
        missing = [f for f in local_frames if f not in found]
        if missing:
            read = self._read(preview, part_index, subject, missing, size)
            with self._lock:
                for frame in missing:
                    found[frame] = self._kept[(key, frame)] = read.get(frame)
                while len(self._kept) > self.capacity:
                    self._kept.popitem(last=False)
        return [found[frame] for frame in local_frames]

    @staticmethod
    def _read(preview, part_index, subject, frames, size) -> dict:
        """``{local_frame: (y0, x0, crop)}`` for ``frames``, from one read of their rows."""
        part = preview._parts[part_index]
        frame_idx = part["frame_idx"]
        if part["reader"] is None or not len(frame_idx):
            return {}
        lo = int(np.searchsorted(frame_idx, min(frames), "left"))
        hi = int(np.searchsorted(frame_idx, max(frames), "right"))
        if lo >= hi:
            return {}
        rows = preview._read_rows(part, lo, hi, with_masks=True)
        object_id = rows["object_id"]
        wanted = set(frames)
        union: dict[int, np.ndarray] = {}
        for j in range(hi - lo):
            frame = int(frame_idx[lo + j])
            if frame not in wanted:
                continue
            oid = int(object_id[j]) if object_id is not None else None
            if oid is not None and oid < 0:
                continue  # a detection the tracker left unassigned
            if subject is not None and oid != subject:
                continue
            counts = str(rows["mask_rle"][j])
            if not counts:
                continue
            mask = decode_mask(counts, size, encoded=part.get("mask_hw"))
            union[frame] = (union[frame] | mask) if frame in union else mask
        out = {}
        for frame, mask in union.items():
            # The mask's extent from two axis reductions, not `np.nonzero`, which lists every
            # pixel of the animal across a full frame to find four numbers.
            rows = np.flatnonzero(mask.any(axis=1))
            if not len(rows):
                continue
            cols = np.flatnonzero(mask[rows[0] : rows[-1] + 1].any(axis=0))
            y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            out[frame] = (int(y0), int(x0), mask[y0:y1, x0:x1].copy())
        return out


def source_frame_of(preview, part_index: int, local: int) -> int:
    """A part-local frame as the recording-global frame §10's keyframes are numbered in."""
    return int(sum(preview.part_frame_counts[:part_index])) + int(local)


class _ClipArena:
    """A recording's field of view and regions of interest (§10), for the masks pane:
    ``(kind, name, keyframes, active)`` each, the shape interpolated to the frame shown.
    """

    def __init__(self, shapes: list) -> None:
        self.shapes = shapes

    @classmethod
    def of(cls, recording_dir: Path) -> "_ClipArena":
        from detail.storage import spatial

        shapes = []
        for kind in ("arena", "roi"):
            for entry in spatial.list_defs(recording_dir, kind):
                try:
                    keyframes = spatial.read_keyframes(
                        recording_dir, kind, entry["id"], int(entry["n_vertices"])
                    )
                except (FileNotFoundError, KeyError, ValueError):
                    continue  # a half-written region is left off, not a failed frame
                shapes.append(
                    (
                        kind,
                        str(entry.get("name") or entry["id"]),
                        keyframes,
                        _active(entry.get("active")),
                    )
                )
        return cls(shapes)

    def at(self, frame: int) -> list[tuple[str, str, np.ndarray]]:
        """``(kind, name, vertices (n, 2) normalized)`` of each shape live on ``frame``."""
        from detail.core.utils import polygon_at_frame

        out = []
        for kind, name, keyframes, active in self.shapes:
            if active is not None and not active[0] <= frame <= active[1]:
                continue
            if keyframes.keyframes:
                out.append((kind, name, polygon_at_frame(keyframes, frame)))
        return out


def _active(value) -> tuple[int, int] | None:
    if not value:
        return None
    if isinstance(value, dict):
        return int(value["start"]), int(value["end"])
    return int(value[0]), int(value[1])


#: The masks pane's field of view and regions: a muted blue, and the line to the nearest animal.
_ARENA_INK = (90, 130, 190)
_REGION_INK = (150, 160, 200)
#: The tag behind a note written on a pane (regions entered, "no body axis").
_NOTE_TAG = (60, 60, 60)


def _draw_arena(
    masks: np.ndarray, shapes: list, subjects: list, others: list
) -> np.ndarray:
    """The field of view and regions outlined on the masks pane, the regions the clip's animal
    is in named in its corner, and a line from it to the nearest other animal."""
    out = masks.copy()
    h, w = out.shape[:2]
    thickness = max(1, round(min(h, w) / 400))
    scale = np.asarray([w, h], dtype=np.float64)
    centre = _centre_px([d.bbox for d in subjects], w, h)
    inside = []
    for kind, name, vertices in shapes:
        polygon = np.round(np.asarray(vertices) * scale).astype(np.int32)
        ink = _ARENA_INK if kind == "arena" else _REGION_INK
        cv2.polylines(out, [polygon], True, ink, thickness, cv2.LINE_AA)
        if (
            kind == "roi"
            and centre is not None
            and cv2.pointPolygonTest(
                polygon, (float(centre[0]), float(centre[1])), False
            )
            >= 0
        ):
            inside.append(name)
    others_c = [_centre_px([d.bbox], w, h) for d in others]
    others_c = [c for c in others_c if c is not None]
    if centre is not None and others_c:
        nearest = min(others_c, key=lambda c: float(np.hypot(*(c - centre))))
        cv2.line(
            out,
            (int(centre[0]), int(centre[1])),
            (int(nearest[0]), int(nearest[1])),
            _ARENA_INK,
            thickness,
            cv2.LINE_AA,
        )
    if inside:
        visualize.draw_text_label(
            out,
            "in: " + ", ".join(inside),
            (int(6 * thickness), int(6 * thickness)),
            _NOTE_TAG,
            scale=max(min(h, w) / 450, 0.45),
            above=False,
        )
    return out


def _centre_px(boxes: list, w: int, h: int) -> np.ndarray | None:
    """The centre of the union of normalized boxes, in pixels; None with no box."""
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.asarray(
        [
            (b[:, 0].min() + b[:, 2].max()) / 2 * w,
            (b[:, 1].min() + b[:, 3].max()) / 2 * h,
        ]
    )


#: The clip popup's masks pane: an animal that is not the clip's subject, on white.
_OTHER_ANIMAL = (150, 150, 150)
#: The clip popup's pose pane draws the skeleton's edges dark: the default is near-white.
_POSE_EDGE = (110, 110, 110)


@dataclass(frozen=True)
class ClipViews:
    """One frame as the clip popup shows it (:meth:`ClipQueryPreview.clip_views`): RGB
    pictures of the frame's size, ``pose`` ``None`` (with ``pose_note``) when there are no
    keypoints to draw."""

    frame: np.ndarray
    masks: np.ndarray
    pose: np.ndarray | None
    pose_note: str | None
    actual: int
    t_ns: int
    #: Why the pose cannot be drawn body-aligned for this run (None when it can).
    align_note: str | None = None
    #: The keypoints' skeleton (None without keypoints), for choosing which points trail.
    skeleton: Skeleton | None = None
    #: What ``trail`` draws for this clip, or why it draws nothing — for the tick's tooltip.
    trail_note: str | None = None
    #: With ``mask_outline``: the subject's mask (bool, the frame's size), or None without one.
    subject_mask: np.ndarray | None = None
    #: With ``mask_outline`` and ``mask_trail_on_frame``: the onion skin's layers, oldest first,
    #: each ``(y0, x0, bool crop)`` or None — what ``_draw_onion_skin`` would have tinted.
    mask_history: list | None = None
    #: With ``mask_outline`` on a ClipSource's tile: ``[(mask, RGB)]``, each chosen animal's
    #: mask and its own colour, in place of ``subject_mask``.
    subject_outlines: list | None = None
    #: The same with ``mask_trail_on_frame``: ``[(layers, RGB)]``, each animal's onion skin.
    history_outlines: list | None = None
    #: With ``subject_data``: the subject's ``(points (K, 2) normalized, scores (K,))`` on this
    #: frame, or None when it has none here (or the run read no keypoints).
    subject_points: tuple | None = None


@dataclass(frozen=True)
class ClipHeadings:
    """A clip's heading on each of its frames, by the embedding's own rule (§22 *Align to
    heading*; :meth:`ClipQueryPreview.subject_headings`)."""

    #: ``(n,)`` image degrees (y down, 0 = facing +x), NaN on a frame with no heading.
    deg: np.ndarray
    #: ``(n,)`` bool: where a ``mirror`` run flips its picture.
    mirrored: np.ndarray
    #: ``"rotate"`` or ``"mirror"``: how the clip's embedding aligns (unaligned: rotate).
    mode: str
    #: Why no frame has a heading, when none can (no keypoint run, no single fitting
    #: definition, a tile of several animals); None otherwise.
    note: str | None = None


#: The share of the pose pane's height the animal's typical body length fills, body-aligned.
_ALIGNED_LENGTH = 0.4


def _aligned_window(
    points, xy: np.ndarray, scores: np.ndarray, track_id, canvas_wh: tuple[int, int]
) -> np.ndarray:
    """``xy`` (``(n, K, 2)`` normalized) with each frame turned by **its own** body axis, the
    way :func:`_draw_aligned_pose` turns the frame shown — NaN where a frame has no axis.

    So a trail on the body-aligned pose is the path each point took *relative to the body*:
    a paw stepping, a head turning, not the animal crossing the arena (which the footage's
    trail shows, and which aligning is there to remove)."""
    from detail.analysis import body_axis

    front, back, _middle = points.axis_columns
    wh = points.frame_wh if min(points.frame_wh) > 0 else canvas_wh
    typical = points.typical_length(int(track_id))
    out = np.full_like(xy, np.nan)
    if typical <= 0:
        return out
    front_xy, back_xy = body_axis.row_axis_ends(
        xy, scores, front, back, wh, points.cutoff
    )
    scale = _ALIGNED_LENGTH * wh[1] / typical
    for i in range(len(xy)):
        if np.isfinite(front_xy[i]).all() and np.isfinite(back_xy[i]).all():
            out[i] = body_axis.aligned_points(xy[i], front_xy[i], back_xy[i], wh, scale)
    return out


def _turn_clip_views(points, subject, frame, masks, pose, *, footage, pose_turned):
    """``(frame, masks, pose)`` in the clip window's **Aligned** orientation
    (``ClipQueryPreview.clip_views``' ``aligned``): each turned by the one
    ``clip_views.turn_matrix`` of ``subject``'s body axis on this frame, at the frame's size.
    ``subject`` is its ``(points (K, 2) normalized, scores (K,))``, or ``None`` when it has
    none here; a frame with no axis comes back as recorded, the footage (or, without it, the
    masks pane) marked. ``pose_turned``: the pose pane is body-aligned already."""
    from detail.analysis import body_axis
    from detail.api import clip_views

    h, w = frame.shape[:2]
    ends = None
    if subject is not None:
        front, back, _middle = points.axis_columns
        xy, scores = subject
        front_xy, back_xy = body_axis.row_axis_ends(
            np.asarray(xy, dtype=np.float64)[None],
            np.asarray(scores, dtype=np.float64)[None],
            list(front),
            list(back),
            (w, h),
            points.cutoff,
        )
        front_xy, back_xy = front_xy[0], back_xy[0]
        if (
            np.isfinite(front_xy).all()
            and np.isfinite(back_xy).all()
            and not np.allclose(front_xy, back_xy)
        ):
            ends = (front_xy, back_xy)
    if ends is None:
        if footage:
            return clip_views.mark_no_axis(frame), masks, pose
        return frame, clip_views.mark_no_axis(masks), pose
    matrix = clip_views.turn_matrix(ends[0], ends[1], h, (w, h))
    white = (255, 255, 255)
    frame = clip_views.turn_picture(
        frame, matrix, (w, h), (0, 0, 0) if footage else white
    )
    masks = clip_views.turn_picture(masks, matrix, (w, h), white)
    if pose is not None and not pose_turned:
        pose = clip_views.turn_picture(pose, matrix, (w, h), white)
    return frame, masks, pose


def _draw_aligned_pose(canvas: np.ndarray, points, subject, track_id) -> np.ndarray:
    """The clip's animal on ``canvas``, its body axis up (``ClipQueryPreview.clip_views``'s
    ``pose_aligned``); a line instead when this frame gives no axis."""
    from detail.analysis import body_axis

    front, back, _middle = points.axis_columns
    h, w = canvas.shape[:2]
    # Measured in the recording's pixels, as the typical length was; drawn normalized.
    wh = points.frame_wh if min(points.frame_wh) > 0 else (w, h)
    if subject is not None:
        xy, scores = subject
        front_xy, back_xy = body_axis.row_axis_ends(
            xy[None], scores[None], front, back, wh, points.cutoff
        )
        typical = points.typical_length(int(track_id))
        if np.isfinite(front_xy).all() and np.isfinite(back_xy).all() and typical > 0:
            turned = body_axis.aligned_points(
                xy, front_xy[0], back_xy[0], wh, _ALIGNED_LENGTH * wh[1] / typical
            )
            return draw_keypoints(
                canvas, turned, points.skeleton, scores=scores, edge_color=_POSE_EDGE
            )
    visualize.draw_text_label(
        canvas,
        "no body axis on this frame",
        (8, h // 2),
        _NOTE_TAG,
        scale=max(h / 600.0, 0.4),
        above=False,
    )
    return canvas


class _ClipPoints:
    """The keypoint run an embedding run read, held for drawing: each (subject, part) series
    is read whole the first time a frame of it is asked for, so playback looks a frame up
    rather than reopening a store per frame.

    It also holds the **heading definition** the body's axis is measured by (which points are
    the front and which the back): the embedding run's own when it was aligned, else the
    experiment's one saved definition that fits this skeleton. Several that fit is a choice
    this window does not make for you, so it uses none and ``axis_note`` says why.
    """

    def __init__(
        self,
        run,
        note: str | None,
        part_stems: list[str],
        *,
        frame_wh: tuple[int, int] = (0, 0),
        definition: dict | None = None,
        axis_note: str | None = None,
        cutoff: float | None = None,
    ) -> None:
        self.run = run
        self.note = note
        self.part_stems = part_stems
        self.frame_wh = (int(frame_wh[0]), int(frame_wh[1]))
        self.definition = definition
        self.axis_note = note if run is None else axis_note
        self.cutoff = cutoff
        self._series: dict[tuple[int, str], tuple | None] = {}
        self._typical: dict[int, float] = {}
        self._lock = threading.Lock()

    @property
    def skeleton(self) -> Skeleton:
        return self.run.skeleton

    @property
    def axis_columns(self) -> tuple[list[int], list[int], list[int]] | None:
        """``(front, back, middle)`` skeleton columns, or None without a definition."""
        if self.run is None or self.definition is None:
            return None
        from detail.analysis import body_axis

        names = tuple(self.skeleton.names)
        front, back = body_axis.definition_columns(self.definition, names)
        return front, back, body_axis.midline_columns(self.skeleton, self.definition)

    @classmethod
    def of(
        cls, experiment_dir: Path, recording_dir: Path, embed_run_id: str
    ) -> "_ClipPoints":
        from detail.storage.keypoint_runs import (
            KeypointTrackRun,
            LegacyKeypointsRun,
            is_legacy_run_id,
        )

        meta = _source_meta(recording_dir, embed_run_id)
        data = recording_store.read_recording(recording_dir)
        stems = [
            recording_store.part_stem(f) for f in recording_store.video_files(data)
        ]
        resolution = (data.get("video") or {}).get("resolution") or (0, 0)
        run_id = str(meta.get("keypoint_run_id") or "")
        if not run_id and isinstance(embed_run_id, ClipSource):
            return cls(
                None,
                "This export read no keypoint track run, so there is no pose to show. "
                "Export again with a keypoint track run to draw one.",
                stems,
            )
        if not run_id:
            return cls(
                None,
                f"Embedding run {embed_run_id} used no keypoints, so there is no pose to "
                "show. An embedding aligned by, or drawing, a keypoint track run carries one.",
                stems,
            )
        try:
            run = (
                LegacyKeypointsRun(recording_dir, run_id)
                if is_legacy_run_id(run_id)
                else KeypointTrackRun(recording_dir, run_id)
            )
        except (FileNotFoundError, ValueError) as exc:
            return cls(
                None,
                f"The keypoint run {_source_name(embed_run_id)} reads ({run_id}) cannot be "
                f"read: {exc}",
                stems,
            )
        definition, axis_note = _clip_heading_definition(
            experiment_dir, meta, tuple(run.skeleton.names)
        )
        cutoff = meta.get("heading_cutoff")
        if cutoff is None:
            cutoff = meta.get("pose_min_score")
        return cls(
            run,
            None,
            stems,
            frame_wh=(int(resolution[0]), int(resolution[1])),
            definition=definition,
            axis_note=axis_note,
            cutoff=None if cutoff is None else float(cutoff),
        )

    def window(
        self, track_id: int, part_frame_counts: list[int], first: int, last: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(points (n, K, 2), scores (n, K))`` for one subject over the inclusive global
        frames ``[first, last]``, NaN on a frame it has no keypoints."""
        n = int(last) - int(first) + 1
        k = len(self.skeleton.names)
        points = np.full((n, k, 2), np.nan, dtype=np.float64)
        scores = np.full((n, k), np.nan, dtype=np.float64)
        start = 0
        for stem, count in zip(self.part_stems, part_frame_counts):
            lo, hi = max(int(first), start), min(int(last), start + int(count) - 1)
            if lo <= hi:
                series = self._load(int(track_id), stem)
                if series is not None:
                    frames, pts, scs = series
                    keep = (frames >= lo - start) & (frames <= hi - start)
                    at = frames[keep].astype(np.int64) + start - int(first)
                    points[at] = pts[keep]
                    scores[at] = scs[keep]
            start += int(count)
        return points, scores

    def typical_length(self, track_id: int) -> float:
        """The subject's median body-axis length in pixels over the whole run (0 when it has
        none): what ``body length`` is relative to, and what the aligned pose is scaled by,
        so neither changes with the window."""
        columns = self.axis_columns
        if columns is None:
            return 0.0
        with self._lock:
            found = self._typical.get(int(track_id))
        if found is not None:
            return found
        from detail.analysis import body_axis

        front, back, _middle = columns
        lengths = []
        for stem in self.part_stems:
            series = self._load(int(track_id), stem)
            if series is not None:
                lengths.append(
                    body_axis.row_headings(
                        series[1], series[2], front, back, self.frame_wh, self.cutoff
                    )[1]
                )
        every = np.concatenate(lengths) if lengths else np.zeros(0)
        every = every[np.isfinite(every) & (every > 0)]
        typical = float(np.median(every)) if every.size else 0.0
        with self._lock:
            self._typical[int(track_id)] = typical
        return typical

    def at(
        self, part_id: str, frame_idx: int
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """``{subject: (points (K, 2) normalized, scores (K,))}`` on one frame of a part."""
        out = {}
        for track_id in self.run.track_ids:
            series = self._load(int(track_id), part_id)
            if series is None:
                continue
            frames, points, scores = series
            i = int(np.searchsorted(frames, frame_idx))
            if i < len(frames) and int(frames[i]) == int(frame_idx):
                out[int(track_id)] = (points[i], scores[i])
        return out

    def _load(self, track_id: int, part_id: str):
        key = (track_id, part_id)
        with self._lock:
            if key not in self._series:
                if part_id in self.run.part_ids(track_id):
                    rows = self.run.rows(track_id, part_id)
                    self._series[key] = (
                        np.asarray(rows["frame_idx"]),
                        np.asarray(rows["points"], dtype=np.float32),
                        np.asarray(rows["point_scores"], dtype=np.float32),
                    )
                else:
                    self._series[key] = None
            return self._series[key]


def _clip_heading_definition(
    experiment_dir: Path, meta: dict, names: tuple[str, ...]
) -> tuple[dict | None, str | None]:
    """``(definition, None)`` or ``(None, why there is none)`` for the clip popup's body axis:
    the aligned embedding's own, else the experiment's single saved definition that fits.
    """
    from detail.analysis import body_axis
    from detail.storage import heading_definitions

    own = meta.get("heading_definition")
    if own:
        try:
            return body_axis.validate_definition(dict(own), names), None
        except ValueError as exc:
            return None, f"The embedding's heading definition does not fit: {exc}"
    fitting = [
        d
        for d in heading_definitions.list_definitions(experiment_dir)
        if heading_definitions.fits(d, names) is None
    ]
    if len(fitting) == 1:
        return body_axis.validate_definition(fitting[0], names), None
    if not fitting:
        return None, (
            "No heading definition says which keypoints are the front and the back. Make "
            "one with Define... beside Heading on the embedding form (it is saved for the "
            "experiment) to measure the body's length, bend and turning and to align the "
            "pose."
        )
    return None, (
        "Several saved heading definitions fit this skeleton "
        f"({', '.join(sorted(str(d.get('name')) for d in fitting))}) and the embedding was "
        "not aligned by one, so none is chosen for you."
    )


def _heading_settings(recording_dir: Path, meta: dict, points: "_ClipPoints") -> dict:
    """``headings_for_run``'s settings for a montage's turn (§22 *Align to heading*): the
    embedding run's own when it carries a heading, else its keypoint run and the definition
    the clip window chose, at the embedding's defaults — ``align`` is how the run aligned,
    and ``rotate`` for a run that did not."""
    from detail.pipeline import embedding as embedding_pipeline

    if meta.get("heading_definition"):
        settings = embedding_pipeline._alignment_settings(meta)
        if settings["align"] == "none":
            settings["align"] = "rotate"
        return settings
    from detail.storage.keypoint_runs import KeypointDetectRun

    # `point_cutoff` reads the detect run only when the track method names no cut-off.
    detect = (
        None
        if "decode_cutoff" in (points.run.meta.get("config") or {})
        else KeypointDetectRun(recording_dir, points.run.meta["detect_run_id"])
    )
    defaults = embedding_pipeline._ALIGNMENT_DEFAULTS
    return {
        **embedding_pipeline._alignment_settings({}),
        "align": "rotate",
        "keypoint_run_id": str(meta["keypoint_run_id"]),
        "heading_definition": {
            "front": list(points.definition["front"]),
            "back": list(points.definition["back"]),
        },
        "heading_cutoff": embedding_pipeline.point_cutoff(points.run, detect),
        "max_gap_s": float(defaults["max_gap_s"]),
        "mirror_hold_s": float(defaults["mirror_hold_s"]),
        "facing_camera_ratio": float(defaults["facing_camera_ratio"]),
    }


def _clip_headings(
    recording_dir: Path,
    tracking_run_id: str,
    track_id: int,
    settings: dict,
    frame_hw: tuple[int, int],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{part_id: (local frame_idx, deg, mirrored)}`` for one subject at every frame —
    ``headings_for_run`` with ``every_n_frames=1``, cached against the tracking run."""
    from detail.pipeline.embedding import headings_for_run
    from detail.storage import read_cache

    key = (
        "montage_clip_headings",
        str(recording_dir),
        tracking_run_id,
        int(track_id),
        repr(sorted((k, repr(v)) for k, v in settings.items())),
        tuple(int(v) for v in frame_hw),
    )

    def compute():
        per_part = headings_for_run(
            recording_dir, tracking_run_id, [int(track_id)], 1, settings, frame_hw
        )
        return {
            part_id: (
                np.asarray(h["frame_idx"], dtype=np.int64),
                np.asarray(h["deg"], dtype=np.float32),
                np.asarray(h["mirrored"], dtype=bool),
            )
            for (_track, part_id), h in per_part.items()
        }

    stamp = Path(recording_dir) / "tracking" / tracking_run_id / "track.yaml"
    return read_cache.cached_read(key, stamp, compute)


class ClipQueryPreview:
    """Experiment-level read boundary for the clip-query analysis (§16) — the API seam the
    GUI reads through (Tenet 2). Loads subject trajectories, runs an interactive preview
    search, scores example agreement, and renders a scrub frame (source + silhouette); the
    pure matcher lives in ``analysis/``."""

    def __init__(self, experiment_dir: Path | str) -> None:
        self._dir = Path(experiment_dir)
        # (recording_id, embed_run_id) -> (InputImagePreview, source_t_ns) for scrubbing.
        self._scrub: dict[tuple[str, str], tuple] = {}
        # One lock per key, so a build for one recording never blocks a question about
        # another — see `_scrub_ctx`. `_scrub_guard` only guards handing out the locks.
        self._scrub_locks: dict[tuple[str, str], threading.Lock] = {}
        self._scrub_guard = threading.Lock()
        # (recording_id, embed_run_id) -> the keypoints that run read, for the clip popup.
        self._points: dict[tuple[str, str], _ClipPoints] = {}
        # (recording_id, embed_run_id) -> the reader of its clip signals, parts kept.
        self._signals: dict[tuple[str, str], object] = {}
        # recording_id -> its field of view and regions, for the masks pane.
        self._arenas: dict[str, _ClipArena] = {}
        # (recording_id, embed_run_id) -> the subject masks the masks pane's trail has read.
        self._mask_histories: dict[tuple[str, str], _MaskHistory] = {}
        # recording_id -> its footage's (height, width), or None when the record has none.
        self._frame_sizes: dict[str, tuple[int, int] | None] = {}

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
            subject = (spec["recording_id"], coerce_series_key(spec["track_id"]))
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
        meta = _source_meta(recording_dir, embed_run_id)
        data = recording_store.read_recording(recording_dir)
        tracking_run_id = meta["tracking_run_id"]
        detection_run_id = {r["id"]: r for r in data.get("tracking_runs", [])}[
            tracking_run_id
        ]["detection_run_id"]
        # Frames, masks and parts are the tracking run's either way; only an embedding run's
        # own input image (`frame_view`) needs its renderer, which a ClipSource has none of.
        preview = (
            TrackingPreview(recording_dir, tracking_run_id)
            if isinstance(embed_run_id, ClipSource)
            else InputImagePreview.for_embedding_run(recording_dir, meta)
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

        A location export's tile of several animals (:func:`_tile_colors`) blends each in its
        own colour rather than in ``tint``.
        """
        from detail.core.visualize import MASK_TINT, draw_mask

        preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        result = preview.frame(part_index, local, with_masks=True)
        if result is None:
            return None
        actual, rgb, detections = result
        t_ns = int(source_t_ns[global_frame]) if global_frame < len(source_t_ns) else 0
        several = _tile_colors(embed_run_id)
        if several is not None:
            for mask, colour in _colour_layers(detections, several):
                rgb = draw_mask(rgb, mask, colour, alpha=alpha)
            return rgb, int(actual), t_ns
        det = next((d for d in detections if d.object_id == int(track_id)), None)
        if det is not None and det.mask is not None:
            colour = tuple(int(c) for c in (tint or MASK_TINT))
            rgb = draw_mask(rgb, det.mask, colour, alpha=alpha)
        return rgb, int(actual), t_ns

    def clip_views(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        global_frame: int,
        settings=None,
        *,
        masks_on_frame: bool | None = None,
        points_on_frame: bool | None = None,
        trail: bool = False,
        trail_points: Sequence[str] | None = None,
        mask_trail: bool = False,
        pose_aligned: bool = False,
        aligned: bool = False,
        mask_trail_on_frame: bool = False,
        mask_outline: bool = False,
        footage: bool = True,
        trail_s: float | None = None,
        subject_data: bool = False,
    ) -> "ClipViews | None":
        """The three pictures of one frame the clip popup shows, from **one** decode, or
        ``None`` when the position yields no picture.

        - ``frame``: the footage, with the subject's mask (``masks_on_frame``, as
          :meth:`frame_masked` draws it) and the subject's keypoints (``points_on_frame``)
          drawn on it — or neither.
        - ``masks``: every tracked animal's mask alone on white, the subject in the mask tint
          and the others grey, so a second animal is visible as one without the footage. A
          location export's tile of several animals (:func:`_tile_colors`) has each in its own
          colour, here, on the footage and in both onion skins.
        - ``pose``: every subject's keypoints alone on white, the subject in its role colours
          and the others greyed — or ``None`` with ``pose_note`` saying why, when the embedding
          run read no keypoint run.

        The points are **the keypoint run the embedding used** (``keypoint_run_id`` on its
        ``embedding.yaml``: an aligned or pose-carrying run), never a run picked here, so the
        pose shown is the one the clusters were built with. Masks are decoded on every frame
        whatever the toggles, because the masks pane needs them.

        ``trail`` draws where the subject has just been, over the last :data:`CLIP_TRAIL_S`,
        as lines that fade into the past: with keypoints, **each point's own path** in its
        role colour (only ``trail_points``, by name, when given; every point when ``None``) on
        the footage and on the pose pane; without keypoints, the centre of its box, in white,
        on the footage. ``trail_note`` says which.

        ``mask_trail`` draws the subject's masks over the same span **under** the current
        ones on the masks pane, fading with age — an onion skin of how its outline moved and
        changed. They are read from the detection run's rows and decoded from their RLE
        (:class:`_MaskHistory`, kept between frames); no extra video frame is decoded.

        ``mask_trail_on_frame`` draws the same onion skin on the **footage**, under the subject's
        tint — what a clip montage's Raw + Mask trail tile shows, where there is no masks pane to
        carry it.

        ``mask_outline`` hands the subject's mask (``subject_mask``) and that onion skin
        (``mask_history``) back **as data** instead of tinting them onto the footage, for a
        caller that draws them as outlines at its own scale — the clip montage, whose tiles are
        cropped and shrunk after this, so a line drawn here at the frame's resolution would be
        a different width in every tile.

        ``pose_aligned`` draws the pose pane **body-aligned**: the clip's animal alone, its body
        axis (the heading definition's back to front) pointing up, its middle at the centre and
        its typical length a fixed share of the pane, so two clips' postures compare directly
        whatever way the animals faced. Without a heading definition the pane is drawn as
        usual and ``align_note`` says why; a frame where the axis cannot be found is left
        blank with a line saying so. Its trails turn each past frame by **that** frame's
        axis (:func:`_aligned_window`): the points' paths relative to the body.

        ``aligned`` is the clip window's **Aligned** orientation: the footage, the masks pane
        and the pose pane each turned by the one ``api.clip_views.turn_matrix`` of this frame's
        body axis, so the animal points up about its middle at the picture's centre, at the
        frame's own size. A frame with no axis for the subject is drawn as recorded and marked
        (``clip_views.mark_no_axis``); a run with no axis at all (no keypoints, no heading
        definition, a tile of several animals) is untouched, with ``align_note`` saying why. The pose pane is
        not turned again when ``pose_aligned`` has drawn it body-aligned already.

        ``footage`` off is for a caller that shows only the masks and the keypoints — a Raw-off
        montage tile, a side pane's export, the enlarged pane — and **decodes no video**: the
        masks and points are stored data, and the frame's size is ``recording.yaml``'s.
        ``frame`` is then a white canvas with nothing drawn on it (the ``*_on_frame`` marks and
        the footage's trail are skipped); ``masks`` and ``pose`` are what the decode draws,
        pixel for pixel, at the frame the decode lands on when the timestamp map describes the
        video. A recording with no recorded resolution is decoded as before.

        ``subject_data`` is for a caller that draws the clip's animal itself — the montage's
        overlay (§22 *Overlay*): the subject's mask (``subject_mask``), its onion skin with
        ``mask_trail`` (``mask_history``) and its keypoints (``subject_points``) come back as
        data, and neither pane is drawn. A tile of several animals has no one subject and gets
        none.

        ``settings`` (a ``clip_pane_export.ClipViewSettings``) gives the view ticks as one object —
        ``masks_on_frame``, ``points_on_frame``, ``trail``, ``trail_points``, ``mask_trail``,
        ``pose_aligned`` and ``aligned`` — in place of the keywords of the same names; without
        it ``masks_on_frame`` and ``points_on_frame`` must be given.

        Raises:
            TypeError: neither ``settings`` nor both of ``masks_on_frame`` and
                ``points_on_frame``.
        """
        from detail.core.visualize import (
            KEYPOINT_COLORS,
            MASK_TINT,
            draw_mask,
            keypoint_roles,
        )

        if settings is not None:
            masks_on_frame = settings.masks_on_frame
            points_on_frame = settings.points_on_frame
            trail = settings.trail
            trail_points = settings.trail_points
            mask_trail = settings.mask_trail
            pose_aligned = settings.pose_aligned
            aligned = settings.aligned
        elif masks_on_frame is None or points_on_frame is None:
            raise TypeError(
                "clip_views needs settings, or both masks_on_frame and points_on_frame"
            )

        aligned_view = bool(aligned)
        preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        frame_hw = None if footage else self._frame_hw(recording_id)
        if frame_hw is None:
            footage = True
            result = preview.frame(part_index, local, with_masks=True)
        else:
            masks_on_frame = points_on_frame = mask_trail_on_frame = False
            result = preview.blank_frame(part_index, local, frame_hw)
        if result is None:
            return None
        actual, rgb, detections = result
        t_ns = int(source_t_ns[global_frame]) if global_frame < len(source_t_ns) else 0
        shown = int(source_frame_of(preview, part_index, actual))
        trail_seconds = CLIP_TRAIL_S if trail_s is None else float(trail_s)
        span = (
            max(
                int(round(trail_seconds * self.frame_rate(recording_id, embed_run_id))),
                2,
            )
            if trail or mask_trail or mask_trail_on_frame
            else 0
        )

        # A ClipSource names its animals and their colours itself (a location export's tile):
        # each in its own colour, on the footage, the masks pane and both onion skins.
        pair = _tile_colors(embed_run_id)
        own = dict(pair or {})
        several = pair is not None

        def is_subject(det: Detection) -> bool:
            if several:
                return det.object_id is not None and int(det.object_id) in own
            return det.object_id is not None and int(det.object_id) == int(track_id)

        def colour_of(det: Detection) -> tuple[int, int, int]:
            if not is_subject(det):
                return _OTHER_ANIMAL
            if pair is None or det.object_id is None:
                return MASK_TINT
            return own.get(int(det.object_id), MASK_TINT)

        if subject_data:
            return self._subject_data(
                recording_id,
                embed_run_id,
                track_id,
                preview,
                part_index,
                result,
                t_ns,
                span if mask_trail else 0,
            )

        frame = rgb
        onion = None
        skins: list[tuple[list, tuple[int, int, int]]] = []
        if mask_trail or mask_trail_on_frame:
            history = self._mask_history(recording_id, embed_run_id)
            before = list(range(max(int(actual) - span, 0), int(actual)))

            def skin_of(subject: int | None) -> list:
                return history.layers(
                    preview, part_index, subject, before, rgb.shape[:2]
                )

            if pair is None or mask_outline:
                onion = skin_of(None if several else track_id)
            skins = (
                [(skin_of(oid), colour) for oid, colour in own.items()]
                if pair is not None
                else [(onion, MASK_TINT)]
            )
        if mask_trail_on_frame and not mask_outline:
            for layers, colour in skins:
                frame = visualize.draw_onion_skin(frame, layers, colour)
        subject_mask = None
        # A ClipSource's tile outlines each chosen animal in its own colour, as it fills them
        # without the footage.
        own_outlines = mask_outline and several
        subject_outlines = None
        if masks_on_frame and own_outlines:
            subject_outlines = [
                (np.asarray(det.mask, dtype=bool), own[int(det.object_id)])
                for det in detections
                if is_subject(det) and det.mask is not None
            ]
        elif masks_on_frame and pair is not None and not mask_outline:
            for mask, colour in _colour_layers(detections, own):
                frame = draw_mask(frame, mask, colour, alpha=0.35)
        elif masks_on_frame:
            subject = next((d for d in detections if is_subject(d)), None)
            if subject is not None and subject.mask is not None:
                if mask_outline:
                    subject_mask = np.asarray(subject.mask, dtype=bool)
                else:
                    frame = draw_mask(frame, subject.mask, MASK_TINT, alpha=0.35)
        masks = np.full_like(rgb, 255)
        if mask_trail:
            for layers, colour in skins:
                masks = visualize.draw_onion_skin(masks, layers, colour)
        for det in sorted(detections, key=is_subject):  # the subject on top
            if det.mask is not None:
                masks = draw_mask(masks, det.mask, colour_of(det), alpha=1.0)
        masks = _draw_arena(
            masks,
            self._clip_arena(recording_id).at(shown),
            [d for d in detections if is_subject(d)],
            [d for d in detections if not is_subject(d)],
        )

        points = self._clip_points(recording_id, embed_run_id)
        trail_note = None
        # ((n, K, 2) normalized, (n, K) scores, colour per point, kept per point)
        point_trail = None
        if trail and several:
            # One trail per animal, in that animal's colour: two animals' paths in the
            # per-point colours could not be told apart.
            for oid, colour in own.items():
                if points.run is not None:
                    xy, _scores = points.window(
                        int(oid),
                        list(preview.part_frame_counts),
                        max(shown - span, 0),
                        shown,
                    )
                    names = list(points.skeleton.names)
                    wanted = set(names if trail_points is None else trail_points)
                    kept = np.array([name in wanted for name in names], dtype=bool)
                    paths, colours = xy[:, kept], [colour] * int(kept.sum())
                else:
                    centre = (
                        self._clip_signal_reader(recording_id, embed_run_id)
                        .read(int(oid), max(shown - span, 0), shown, source_t_ns)
                        .centre
                    )
                    paths, colours = centre[:, None, :], [colour]
                if footage:
                    frame = visualize.draw_fading_paths(frame, paths, colours)
            trail_note = "Each animal's path, in its colour, on the footage."
        elif trail and points.run is not None:
            names = list(points.skeleton.names)
            wanted = set(names if trail_points is None else trail_points)
            kept = np.array([name in wanted for name in names], dtype=bool)
            xy, scores = points.window(
                int(track_id),
                list(preview.part_frame_counts),
                max(shown - span, 0),
                shown,
            )
            colours = [
                KEYPOINT_COLORS[role] for role in keypoint_roles(points.skeleton)
            ]
            point_trail = (xy, scores, colours, kept)
            if footage:
                frame = visualize.draw_fading_paths(
                    frame, xy[:, kept], [c for c, k in zip(colours, kept) if k]
                )
            trail_note = (
                "Each ticked keypoint's path over the last half second, in its colour, on "
                "the footage and the Keypoints pane."
            )
        elif trail:
            if footage:
                centre = (
                    self._clip_signal_reader(recording_id, embed_run_id)
                    .read(int(track_id), max(shown - span, 0), shown, source_t_ns)
                    .centre
                )
                frame = visualize.draw_fading_paths(
                    frame, centre[:, None, :], [_CENTRE_TRAIL]
                )
            trail_note = (
                "This run has no keypoints, so the trail is the centre of the animal's "
                "box over the last half second, on the footage."
            )

        pose = None
        aligned = pose_aligned and not several and points.axis_columns is not None
        subject_placed = None
        if points.run is not None:
            stem = points.part_stems[part_index]
            pose = np.full_like(rgb, 255)
            placed = points.at(stem, int(actual))
            subject_placed = None if several else placed.get(int(track_id))
            if point_trail is not None:
                xy, scores, colours, kept = point_trail
                if aligned:
                    xy = _aligned_window(
                        points, xy, scores, track_id, (rgb.shape[1], rgb.shape[0])
                    )
                pose = visualize.draw_fading_paths(
                    pose, xy[:, kept], [c for c, k in zip(colours, kept) if k]
                )
            if aligned:
                pose = _draw_aligned_pose(
                    pose, points, placed.get(int(track_id)), track_id
                )
                placed = {}
            for subject_id in sorted(placed, key=lambda t: several or t == track_id):
                xy, scores = placed[subject_id]
                own = several or subject_id == track_id
                pose = draw_keypoints(
                    pose,
                    xy,
                    points.skeleton,
                    scores=scores if own else np.zeros_like(scores),
                    edge_color=_POSE_EDGE,
                )
                if own and points_on_frame:
                    frame = draw_keypoints(frame, xy, points.skeleton, scores=scores)
        if aligned_view and not several and points.axis_columns is not None:
            frame, masks, pose = _turn_clip_views(
                points,
                subject_placed,
                frame,
                masks,
                pose,
                footage=footage,
                pose_turned=aligned,
            )
        return ClipViews(
            frame=frame,
            masks=masks,
            pose=pose,
            pose_note=points.note,
            actual=int(actual),
            t_ns=t_ns,
            align_note=(
                "A tile of several animals follows no one animal, so there is no body to "
                "align."
                if several and points.run is not None
                else points.axis_note
            ),
            skeleton=points.skeleton if points.run is not None else None,
            trail_note=trail_note,
            subject_mask=subject_mask,
            mask_history=onion if (mask_outline and mask_trail_on_frame) else None,
            subject_outlines=subject_outlines,
            history_outlines=(
                skins if (own_outlines and mask_trail_on_frame) else None
            ),
        )

    def _subject_data(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        preview,
        part_index: int,
        result,
        t_ns: int,
        span: int,
    ) -> "ClipViews":
        """:meth:`clip_views`' ``subject_data`` answer: the subject's mask, onion skin and
        points on one frame, as data, with no pane drawn."""
        actual, rgb, detections = result
        points = self._clip_points(recording_id, embed_run_id)
        mask = history = placed = None
        if _tile_colors(embed_run_id) is None:
            subject = next(
                (
                    d
                    for d in detections
                    if d.object_id is not None and int(d.object_id) == int(track_id)
                ),
                None,
            )
            if subject is not None and subject.mask is not None:
                mask = np.asarray(subject.mask, dtype=bool)
            if span:
                history = self._mask_history(recording_id, embed_run_id).layers(
                    preview,
                    part_index,
                    track_id,
                    list(range(max(int(actual) - span, 0), int(actual))),
                    rgb.shape[:2],
                )
            if points.run is not None:
                placed = points.at(points.part_stems[part_index], int(actual)).get(
                    int(track_id)
                )
        return ClipViews(
            frame=rgb,
            masks=rgb,
            pose=None,
            pose_note=points.note,
            actual=int(actual),
            t_ns=int(t_ns),
            skeleton=points.skeleton if points.run is not None else None,
            subject_mask=mask,
            mask_history=history,
            subject_points=placed,
        )

    def subject_headings(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        first: int,
        last: int,
    ) -> ClipHeadings:
        """The subject's heading on each global frame ``first..last`` (inclusive), by the
        embedding's own rule (§22 *Align to heading*): the run's keypoint track run, its heading
        definition (else the experiment's one that fits, as the clip window's body axis
        chooses), and ``pipeline.embedding.headings_for_run`` at every frame, with the run's own
        gap, facing-camera and cut-off settings when it has a heading and the embedding's
        defaults when it does not. NaN where the embedding would write no row.

        Read once per (recording, tracking run, subject, settings) and cached against the
        tracking run, so every page and every clip of a recording share it.
        """
        n = max(int(last) - int(first) + 1, 0)

        def none(note: str | None, mode: str = "rotate") -> ClipHeadings:
            return ClipHeadings(
                np.full(n, np.nan, dtype=np.float32),
                np.zeros(n, dtype=bool),
                mode,
                note,
            )

        if _tile_colors(embed_run_id) is not None:
            return none("A tile of several animals has no one heading.")
        points = self._clip_points(recording_id, embed_run_id)
        if points.run is None:
            return none(points.note)
        if points.definition is None:
            return none(points.axis_note)
        from detail.storage.keypoint_runs import is_legacy_run_id

        recording_dir = experiment_store.recording_path(self._dir, recording_id)
        meta = _source_meta(recording_dir, embed_run_id)
        if is_legacy_run_id(str(meta.get("keypoint_run_id") or "")):
            return none(
                f"{_source_name(embed_run_id)} read a keypoints run from before keypoint "
                "track runs (§26), which has no heading series to turn by."
            )
        settings = _heading_settings(recording_dir, meta, points)
        frame_hw = self._frame_hw(recording_id)
        if frame_hw is None:
            frame_hw = (points.frame_wh[1], points.frame_wh[0])
        per_part = _clip_headings(
            recording_dir,
            str(meta["tracking_run_id"]),
            int(track_id),
            settings,
            frame_hw,
        )
        preview, _t = self._scrub_ctx(recording_id, embed_run_id)
        deg = np.full(n, np.nan, dtype=np.float32)
        mirrored = np.zeros(n, dtype=bool)
        start = 0
        for stem, count in zip(points.part_stems, preview.part_frame_counts):
            lo, hi = max(int(first), start), min(int(last), start + int(count) - 1)
            found = per_part.get(stem)
            if lo <= hi and found is not None:
                frames, part_deg, part_mirrored = found
                local = np.arange(lo, hi + 1, dtype=np.int64) - start
                at = np.searchsorted(frames, local)
                inside = at < frames.size
                hit = inside.copy()
                hit[inside] = frames[at[inside]] == local[inside]
                rows = np.flatnonzero(hit) + (lo - int(first))
                deg[rows] = part_deg[at[hit]]
                mirrored[rows] = part_mirrored[at[hit]]
            start += int(count)
        return ClipHeadings(deg, mirrored, settings["align"], None)

    def subject_boxes(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        first: int,
        last: int,
    ) -> np.ndarray:
        """``(n, 4)`` normalized boxes of the clip's animal over ``[first, last]`` (inclusive
        global frames), NaN where it was not tracked; a location export's tile of several
        animals (:func:`_tile_colors`) gets their union. Read from the tracking run's rows;
        nothing is decoded.
        """
        reader = self._clip_signal_reader(recording_id, embed_run_id)
        if _tile_colors(embed_run_id) is not None:
            # The chosen animals' union, not every animal's: an animal left out of the tile
            # must not widen its crop.
            boxes = [
                reader.boxes(oid, int(first), int(last))
                for oid in embed_run_id.subject_ids
            ]
            union = boxes[0].copy()
            for more in boxes[1:]:
                union[:, :2] = np.fmin(union[:, :2], more[:, :2])
                union[:, 2:] = np.fmax(union[:, 2:], more[:, 2:])
            return union
        return reader.boxes(int(track_id), int(first), int(last))

    def clip_has_points(self, recording_id: str, embed_run_id: str) -> str | None:
        """``None`` when the embedding run read a keypoint run the popup can draw, else why
        not — for a control that only means something when there are points."""
        return self._clip_points(recording_id, embed_run_id).note

    def clip_signals(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        first: int,
        last: int,
    ):
        """What the clip's animal is doing over the window ``[first, last]`` (inclusive
        global frames): speed, mask area, distance to the nearest other animal, and the
        frames not to trust (``api.clip_signals``) — then, when the embedding read keypoints
        and a heading definition names the body's ends, its length, bend and turning
        (``clip_signals.pose_signals``). A location export's tile of several animals
        (:func:`_tile_colors`) has no one animal to follow, so it gets none: ``None``.
        """
        from dataclasses import replace

        from detail.api.clip_signals import pose_signals

        if _tile_colors(embed_run_id) is not None:
            return None
        reader = self._clip_signal_reader(recording_id, embed_run_id)
        preview, source_t_ns = self._scrub_ctx(recording_id, embed_run_id)
        signals = reader.read(int(track_id), int(first), int(last), source_t_ns)
        points = self._clip_points(recording_id, embed_run_id)
        columns = points.axis_columns
        if columns is None:
            return signals
        front, back, middle = columns
        xy, scores = points.window(
            int(track_id), list(preview.part_frame_counts), int(first), int(last)
        )
        shape = pose_signals(
            xy,
            scores,
            signals.t_ns,
            front=front,
            back=back,
            middle=middle,
            frame_wh=points.frame_wh,
            cutoff=points.cutoff,
            typical_length_px=points.typical_length(int(track_id)),
        )
        return replace(signals, signals=[*signals.signals, *shape])

    def _clip_signal_reader(self, recording_id: str, embed_run_id: str):
        """The reader of a run's clip signals, opened once per (recording, run)."""
        key = (recording_id, embed_run_id)
        reader = self._signals.get(key)
        if reader is None:
            reader = self._signal_reader(recording_id, embed_run_id)
            self._signals[key] = reader
        return reader

    def _mask_history(self, recording_id: str, embed_run_id: str) -> "_MaskHistory":
        """The masks pane's onion-skin cache for one (recording, run)."""
        key = (recording_id, embed_run_id)
        found = self._mask_histories.get(key)
        if found is None:
            found = self._mask_histories.setdefault(key, _MaskHistory())
        return found

    def _signal_reader(self, recording_id: str, embed_run_id: str):
        from detail.api.clip_signals import ClipSignalReader
        from detail.api.recording import Recording

        recording_dir = experiment_store.recording_path(self._dir, recording_id)
        meta = _source_meta(recording_dir, embed_run_id)
        data = recording_store.read_recording(recording_dir)
        preview, _ = self._scrub_ctx(recording_id, embed_run_id)
        try:
            ppm = Recording(self._dir, recording_id).pixels_per_mm
        except ValueError:  # a calibration that cannot describe this video: no mm
            ppm = None
        resolution = (data.get("video") or {}).get("resolution") or (0, 0)
        return ClipSignalReader(
            recording_dir,
            str(meta["tracking_run_id"]),
            [recording_store.part_stem(f) for f in recording_store.video_files(data)],
            list(preview.part_frame_counts),
            (int(resolution[0]), int(resolution[1])),
            ppm,
        )

    def _frame_hw(self, recording_id: str) -> tuple[int, int] | None:
        """The footage's ``(height, width)`` from ``recording.yaml``'s ``video.resolution``
        (``[width, height]``, every part's, §11), read once; ``None`` when it is not recorded.
        """
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

    def _clip_arena(self, recording_id: str) -> "_ClipArena":
        """The recording's field of view and regions, read once (§10)."""
        found = self._arenas.get(recording_id)
        if found is None:
            found = _ClipArena.of(
                experiment_store.recording_path(self._dir, recording_id)
            )
            self._arenas[recording_id] = found
        return found

    def _clip_points(self, recording_id: str, embed_run_id: str) -> "_ClipPoints":
        """The keypoints an embedding run read, opened once per (recording, run)."""
        key = (recording_id, embed_run_id)
        found = self._points.get(key)
        if found is None:
            recording_dir = experiment_store.recording_path(self._dir, recording_id)
            found = _ClipPoints.of(self._dir, recording_dir, embed_run_id)
            self._points[key] = found
        return found

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

        A location export's tile of several animals (:func:`_tile_colors`) frames the union of
        every detection in the frame (``_union_detection``), and when a tint is asked for, each
        animal is blended in its own colour instead.

        Raises:
            SubjectNotDetectedError: the frame decoded but ``track_id`` has no detection in it.
        """
        preview, _ = self._scrub_ctx(recording_id, embed_run_id)
        part_index, local = _global_to_part(preview.part_frame_counts, global_frame)
        result = preview.frame(part_index, local, with_masks=tint is not None)
        if result is None:
            return None
        actual, rgb, detections = result
        crop = self._subject_crop(
            recording_id, embed_run_id, track_id, rgb, detections, size, tint, alpha
        )
        if crop is None:
            raise SubjectNotDetectedError(
                f"track {track_id} has no detection at frame {actual} of "
                f"{recording_id}"
            )
        return crop

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
        of the wrong thing. A location export's tile of several animals is framed and coloured
        as ``source_crop_masked`` does it, and a frame with no detections at all is skipped the
        same way.

        Yields:
            The crops in presentation order.
        """
        preview, _ = self._scrub_ctx(recording_id, embed_run_id)
        counts = preview.part_frame_counts
        first_part, first_local = _global_to_part(counts, first_frame)
        last_part, last_local = _global_to_part(counts, last_frame)
        # Span frames already emitted, so a stride keeps its phase across a part
        # boundary instead of restarting and doubling up a frame there.
        offset = 0
        for part_index in range(first_part, last_part + 1):
            lo = first_local if part_index == first_part else 0
            hi = last_local if part_index == last_part else counts[part_index] - 1
            start = lo + (-offset % step)
            for _actual, rgb, detections in preview.frames_range(
                part_index,
                start,
                hi,
                with_masks=tint is not None,
                step=step,
            ):
                crop = self._subject_crop(
                    recording_id,
                    embed_run_id,
                    track_id,
                    rgb,
                    detections,
                    size,
                    tint,
                    alpha,
                )
                if crop is not None:
                    yield crop
            offset += hi - lo + 1

    def _subject_crop(
        self,
        recording_id: str,
        embed_run_id: str,
        track_id: int,
        rgb: np.ndarray,
        detections: list[Detection],
        size: int,
        tint: tuple[int, int, int] | None,
        alpha: float,
    ) -> np.ndarray | None:
        """One frame's crop around the subject for both crop paths, or ``None`` when the
        subject has no detection in it: one animal in ``tint``, a location export's tile of
        several animals each in its colour.
        """
        several = _tile_colors(embed_run_id)
        if several is not None:
            det = _union_detection(detections)
        else:
            det = next((d for d in detections if d.object_id == int(track_id)), None)
        if det is None or det.bbox is None:
            return None
        if several is not None and tint is not None:
            return _square_crop(
                rgb,
                det.bbox,
                size,
                alpha=alpha,
                layers=_colour_layers(detections, several),
            )
        return _square_crop(
            rgb,
            det.bbox,
            size,
            mask=det.mask,
            tint=None if tint is None else tuple(int(c) for c in tint),
            alpha=alpha,
        )


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
