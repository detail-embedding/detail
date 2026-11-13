"""Keypoint methods of the Python API (§26), delegated to from ``Experiment``.

The seam that keeps three PRs out of one region of ``api/experiment.py`` (§26 *Sequencing*):
the functions here take the ``Experiment`` as their first argument, and ``Experiment`` grows
one-line methods that import this module locally and forward to them — the shape
``Experiment.query_run`` has (``from detail.api.query import QueryRun``).
Each PR fills its own block below and touches no other.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from detail.configs.defaults import none_means_default
from detail.core import camera_view
from detail.core import yaml_load
from detail.core.examples import (
    HEADING_BUCKETS,
    ExampleFrame,
    ExampleMark,
    ExampleSet,
)
from detail.core.skeleton import Skeleton
from detail.pipeline import subject_frames
from detail.storage import example_sets as example_store

if TYPE_CHECKING:
    from detail.api.experiment import Experiment

# -- keypoints (§26): PR 2 -- example sets (a skeleton plus the frames marked on it) ------ #


def resolve_skeleton(
    experiment: "Experiment", skeleton
) -> tuple[Skeleton, dict | None]:
    """A skeleton from what a caller may hand over, and the ``skeleton_ref`` a copy of it
    records.

    ``skeleton`` is a ``Skeleton`` or its dict, a preset or library id (``api/skeletons.py``),
    an example set's id (its copy and its ref), or a YAML file holding a skeleton. The ref is
    the library entry it came from, the source set's own ref, or None for a ``Skeleton`` or a
    file, which name no library entry.

    Raises:
        ValueError: An unknown reference, or a file without a skeleton in it.
    """
    from detail.api import skeletons as library

    if isinstance(skeleton, (Skeleton, dict)):
        return library.resolve_skeleton(skeleton), None
    text = str(skeleton)
    if text in example_store.discover_sets(experiment.dir):
        source = example_store.read_set(experiment.dir, text)
        return source.skeleton, source.skeleton_ref
    path = Path(text)
    if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
        return library.resolve_skeleton(path), None
    try:
        return library.skeleton(text)["skeleton"], library.ref(text)
    except FileNotFoundError:
        raise ValueError(
            f"no skeleton preset, library skeleton, example set or file {text!r}"
        ) from None


def example_sets(
    experiment: "Experiment", purpose: str | None = "example"
) -> list[ExampleSet]:
    """The experiment's example sets of one ``purpose`` (``None`` for every purpose), id order."""
    return example_store.list_sets(experiment.dir, purpose=purpose)


def example_set(experiment: "Experiment", set_id: str) -> ExampleSet:
    """Raises ``FileNotFoundError`` naming the sets that do exist."""
    return example_store.read_set(experiment.dir, set_id)


def create_example_set(
    experiment: "Experiment", name: str, skeleton, *, purpose: str = "example"
) -> ExampleSet:
    """A new, empty set holding a **copy** of ``skeleton`` (anything :func:`resolve_skeleton`
    takes: a library or preset id, another set's id, a ``Skeleton``, or a skeleton YAML path),
    with ``skeleton_ref`` naming the library entry it came from. The copy is fixed: a set's
    skeleton never changes after it is made (§26 *Example sets follow the model*)."""
    sk, ref = resolve_skeleton(experiment, skeleton)
    created = example_store.create_set(experiment.dir, name, sk, purpose=purpose)
    if ref is not None:
        created.skeleton_ref = dict(ref)
        example_store.write_set(experiment.dir, created)
    return created


def _only_tracking_run(experiment: "Experiment", recording_id: str) -> str:
    """The one complete tracking run of a recording. Raises rather than guesses when there
    are several (Tenet 9's resolve-and-record rule), or none."""
    recording = experiment.get_recording(recording_id)
    complete = [
        r["id"]
        for r in recording.tracking_runs
        if str(r.get("status", "complete")) == "complete"
    ]
    name = experiment.recording_names().get(recording_id, recording_id)
    if not complete:
        raise ValueError(
            f"recording {name!r} has no complete tracking run; run tracking first"
        )
    if len(complete) > 1:
        raise ValueError(
            f"recording {name!r} has {len(complete)} tracking runs ({', '.join(complete)}); "
            "say which one to sample from"
        )
    return complete[0]


def resolve_sources(
    experiment: "Experiment", sources: dict[str, str | None] | list[str]
) -> dict[str, str]:
    """``{recording_id: tracking_run_id}`` from what a caller typed: recording names or ids,
    each with a tracking run id or ``None`` for the recording's only complete one.

    Raises:
        KeyError: An unknown recording reference.
        ValueError: A recording with no complete tracking run, or several and none named.
    """
    pairs = (
        {ref: None for ref in sources}
        if isinstance(sources, (list, tuple))
        else sources
    )
    out: dict[str, str] = {}
    for ref, run_id in pairs.items():
        recording_id = experiment.resolve_recording(ref).id
        out[recording_id] = (
            str(run_id) if run_id else _only_tracking_run(experiment, recording_id)
        )
    return out


def heading_applies(experiment: "Experiment", recording_ids) -> bool:
    """Whether the heading sampler (§26 *Headings*) means anything for these recordings.

    Its directions are relative to the camera — toward, away — and a camera looking down
    sees none of them, so it applies unless every recording resolves to ``top_down`` (§10).
    With no recordings, the experiment's default view decides. A view that is not set is
    unknown, and the sampler stays offered.
    """
    ids = list(recording_ids)
    views = (
        [experiment.get_recording(rid).camera_view for rid in ids]
        if ids
        else [experiment.camera_view]
    )
    return not camera_view.all_top_down(views)


def _refuse_heading_from_above(experiment: "Experiment", recording_ids) -> None:
    if not heading_applies(experiment, recording_ids):
        raise ValueError(
            "sampling by heading reads the direction the animal faces relative to the "
            "camera (toward, away), which a top-down camera does not see, and every "
            "recording here is Top down; sample 'varied' or 'spread' instead, or set the "
            "camera view (`experiment set-camera-view` / `recording set-camera-view`) if "
            "that is wrong"
        )


def sample_example_frames(
    experiment: "Experiment",
    set_id: str,
    sources: dict[str, str | None] | list[str],
    n_frames: int = 5,
    *,
    strategy: str = subject_frames.DEFAULT_STRATEGY,
    n_typical: int = subject_frames.DEFAULT_N_TYPICAL,
    from_recordings: int | None = subject_frames.DEFAULT_FROM_RECORDINGS,
    spread_by: str | None = None,
    buckets: list[str] | None = None,
    floor_class: str | None = None,
    track_ids: list[int] | None = None,
    seed: int = 0,
    device: str | None = None,
    progress=None,
) -> ExampleSet:
    """Copy ``n_frames`` tracked frames from ``sources`` into the set (``pipeline.subject_frames``)
    and return the set with them appended, unmarked.

    ``sources`` is recording names or ids, each with the tracking run to read or ``None``
    for the recording's only complete one (:func:`resolve_sources`). ``strategy`` is ``varied``
    (postures that differ; the default), ``spread`` (one per time slice) or ``heading``
    (``n_frames`` per heading bucket of ``buckets``, all when None; ``floor_class`` is the
    detection class that is the floor, when the rig has one — §26 *Headings*). ``varied`` and
    ``heading`` read at most ``from_recordings`` of the recordings (``None``: every one), split
    evenly across the values of the metadata column ``spread_by`` when one is named. The
    frames are decoded exactly, so this takes seconds — minutes for ``heading`` the first time
    it reads a recording: ``progress(stage, fraction, detail)`` names each phase. Audit sets
    (``purpose: audit``) were drawn by the keypoint audit, since removed, and are refused here.

    Raises:
        ValueError: An audit set, a bad request (including a ``spread_by`` that is not a
            metadata column, or a ``floor_class`` no recording detects), or no frames left.
        FileNotFoundError: A recording's video is not on this machine.
    """
    current = example_store.read_set(experiment.dir, set_id)
    if current.purpose != "example":
        raise ValueError(
            f"{set_id!r} is an audit set, made by the keypoint audit (since removed); "
            "frames can only be sampled into an example set"
        )
    resolved = resolve_sources(experiment, sources)
    if strategy == "heading":
        _refuse_heading_from_above(experiment, resolved)
    frames = subject_frames.sample_frames(
        experiment.dir,
        set_id,
        resolved,
        n_frames,
        strategy=strategy,
        n_typical=n_typical,
        from_recordings=from_recordings,
        spread_by=spread_by,
        buckets=buckets,
        floor_class=floor_class,
        track_ids=track_ids,
        seed=seed,
        device=device,
        progress=progress,
    )
    current = example_store.read_set(experiment.dir, set_id)
    current.frames.extend(frames)
    example_store.write_set(experiment.dir, current)
    return current


def heading_pool(
    experiment: "Experiment",
    sources: dict[str, str | None] | list[str],
    *,
    from_recordings: int | None = subject_frames.DEFAULT_FROM_RECORDINGS,
    spread_by: str | None = None,
    floor_class: str | None = None,
    seed: int = 0,
    device: str | None = None,
    progress=None,
):
    """Every frame's heading bucket over the recording subset (``pipeline.headings.Pool``),
    computing each missing cache first — what the review window holds while it is open, so
    paging and rejecting cost nothing. The subset is chosen as ``sample_example_frames``
    chooses it.

    Raises:
        ValueError: A bad request (see ``sample_example_frames``).
        FileNotFoundError: A recording to compute has no video on this machine.
    """
    from detail.pipeline import headings

    resolved = resolve_sources(experiment, sources)
    _refuse_heading_from_above(experiment, resolved)
    groups = (
        subject_frames.recording_groups(experiment.dir, spread_by)
        if spread_by
        else None
    )
    if from_recordings is not None:
        chosen = subject_frames.choose_recordings(
            sorted(resolved), from_recordings, seed, groups
        )
        resolved = {r: resolved[r] for r in chosen}
    return headings.build_pool(
        experiment.dir,
        resolved,
        floor_class=floor_class,
        groups=groups,
        device=device,
        progress=progress,
    )


def heading_candidates(
    experiment: "Experiment", set_id: str, pool, bucket: str, n: int = 8
) -> list[dict]:
    """``bucket``'s next ``n`` candidates from ``pool``, best first, skipping the set's frames
    and those rejected for the bucket (``pipeline.headings.candidates``)."""
    from detail.pipeline import headings

    if bucket not in HEADING_BUCKETS:
        raise ValueError(
            f"unknown heading bucket {bucket!r}; expected one of {HEADING_BUCKETS}"
        )
    current = example_store.read_set(experiment.dir, set_id)
    return headings.candidates(pool, current, bucket, n)


def add_heading_candidates(
    experiment: "Experiment", set_id: str, chosen: list[dict], bucket: str
) -> ExampleSet:
    """Copy candidates ``heading_candidates`` offered into the set, each ``sampled_by``
    ``heading:<bucket>:<source>``, and return the set."""
    current = example_store.read_set(experiment.dir, set_id)
    held = {f.key() for f in current.frames}

    def add_image(rgb):
        return example_store.add_frame_image(experiment.dir, set_id, rgb)

    for cand in chosen:
        if tuple(cand["key"]) in held:
            continue
        copied = subject_frames._copy_frame(
            experiment.dir,
            cand["recording_id"],
            cand["tracking_run_id"],
            cand,
            add_image,
            sampled_by=f"heading:{bucket}:{cand.get('source') or 'none'}",
        )
        if copied is not None:
            current.frames.append(copied[0])
    example_store.write_set(experiment.dir, current)
    return current


def heading_coverage(experiment: "Experiment", set_id: str, pool=None) -> dict:
    """Per bucket, how many of the set's frames fall in it and how many are marked
    (``pipeline.headings.coverage``) — what the compass draws."""
    from detail.pipeline import headings

    return headings.coverage(pool, example_store.read_set(experiment.dir, set_id))


def add_example_frame(
    experiment: "Experiment",
    set_id: str,
    recording: str,
    tracking_run_id: str,
    part_id: str,
    frame_idx: int,
    track_id: int,
) -> ExampleSet:
    """Copy the one frame a user navigated to into the set (``sampled_by`` None: hand-picked)
    and return the set with it appended.

    Raises:
        ValueError: The track has no subject on that frame, or the set already holds it.
    """
    recording_id = experiment.resolve_recording(recording).id
    frame = subject_frames.pick_frame(
        experiment.dir,
        set_id,
        recording_id,
        tracking_run_id,
        part_id,
        int(frame_idx),
        int(track_id),
    )
    current = example_store.read_set(experiment.dir, set_id)
    current.frames.append(frame)
    example_store.write_set(experiment.dir, current)
    return current


def _frame_at(current: ExampleSet, frame_index: int) -> ExampleFrame:
    if not 0 <= int(frame_index) < len(current.frames):
        raise IndexError(
            f"example set {current.id!r} has {len(current.frames)} frames; no frame "
            f"{frame_index}"
        )
    return current.frames[int(frame_index)]


def set_example_marks(
    experiment: "Experiment", set_id: str, frame_index: int, marks
) -> ExampleSet:
    """Replace the marks on one frame and write the set atomically (the labeling screen
    saves every click through here). ``marks`` are ``ExampleMark``s or their dicts; a
    target must be one of the set's skeleton points, a visible point needs ``x``, ``y`` in
    ``[0, 1]`` (normalized full-frame), and a target appears at most once.

    Raises:
        ValueError: A mark the skeleton does not name, a duplicate target, a visible point
            without a position or with one off the frame.
        IndexError: No such frame.
    """
    current = example_store.read_set(experiment.dir, set_id)
    frame = _frame_at(current, frame_index)
    parsed = [
        m if isinstance(m, ExampleMark) else ExampleMark.from_dict(m) for m in marks
    ]
    names = set(current.skeleton.names)
    seen: set[str] = set()
    for mark in parsed:
        if mark.target not in names:
            raise ValueError(
                f"no keypoint {mark.target!r} in example set {set_id!r}; it has "
                f"{list(current.skeleton.names)}"
            )
        if mark.target in seen:
            raise ValueError(f"keypoint {mark.target!r} is marked twice on one frame")
        seen.add(mark.target)
        mark.validate()
        if mark.kind == "point" and mark.visible:
            if not (0.0 <= float(mark.x) <= 1.0 and 0.0 <= float(mark.y) <= 1.0):
                raise ValueError(
                    f"mark on {mark.target!r} at ({mark.x}, {mark.y}) is off the frame; "
                    "positions are normalized to [0, 1]"
                )
    frame.marks = parsed
    example_store.write_set(experiment.dir, current)
    return current


def remove_example_frame(
    experiment: "Experiment", set_id: str, frame_index: int
) -> ExampleSet:
    """Drop one frame from the set. Its PNG stays (a run may list it; §26 "Frames are
    immutable"). Raises ``IndexError`` for no such frame."""
    current = example_store.read_set(experiment.dir, set_id)
    _frame_at(current, frame_index)
    del current.frames[int(frame_index)]
    example_store.write_set(experiment.dir, current)
    return current


def reject_example_frame(
    experiment: "Experiment",
    set_id: str,
    frame_index: int,
    bucket: str | None = None,
) -> ExampleSet:
    """Drop one frame from the set and record it in ``rejected`` so no sampler offers it
    again — for ``bucket`` (a heading bucket, §26 *Headings*), or for every one when ``None``.
    Its PNG stays, as ``remove_example_frame``'s does. Raises ``IndexError`` for no such frame.
    """
    current = example_store.read_set(experiment.dir, set_id)
    frame = _frame_at(current, frame_index)
    del current.frames[int(frame_index)]
    current.reject(frame.key(), bucket)
    example_store.write_set(experiment.dir, current)
    return current


def reject_candidate(
    experiment: "Experiment",
    set_id: str,
    key: tuple[str, str, int, int],
    bucket: str | None = None,
) -> ExampleSet:
    """Record a frame that is not in the set — ``(recording_id, part_id, t_ns, track_id)``, a
    candidate a sampler offered — as refused for ``bucket`` (every bucket when ``None``).
    """
    current = example_store.read_set(experiment.dir, set_id)
    current.reject(tuple(key), bucket)
    example_store.write_set(experiment.dir, current)
    return current


def example_frame_image(
    experiment: "Experiment", set_id: str, frame_index: int
) -> np.ndarray:
    """One frame's picture as RGB uint8 (``frames/<sha>.png``)."""
    current = example_store.read_set(experiment.dir, set_id)
    frame = _frame_at(current, frame_index)
    return example_store.load_image(experiment.dir, set_id, frame.image)


def import_example_set(experiment: "Experiment", yaml_path: Path | str) -> ExampleSet:
    """Read a prepared set (the ``examples.yaml`` body, images beside it as its ``image``
    paths say) into this experiment under a fresh id, copying the images in by content.

    The scripted route to a set (§26 "Example sets"): a labmate's export, or a YAML a script
    wrote. Marks are checked against the skeleton as ``set_example_marks`` checks them.

    Raises:
        FileNotFoundError: The YAML, or an image it names, is missing.
        ValueError: A malformed file, the wrong ``schema_version``, or marks the skeleton
            does not name.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"no example set file at {path}")
    data = yaml_load.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    if "schema_version" not in data and "names" in data:
        # A bare skeleton file (names / edges / pairs / layout): an empty set holding it.
        return create_example_set(experiment, path.stem, Skeleton.from_dict(data))
    if data.get("schema_version") != example_store.SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version is {data.get('schema_version')!r}, expected "
            f"{example_store.SCHEMA_VERSION}"
        )
    data.setdefault("id", "imported")
    data.setdefault("name", path.stem)
    source = ExampleSet.from_dict(data)
    source.skeleton.validate()
    names = set(source.skeleton.names)
    for i, frame in enumerate(source.frames):
        if not (path.parent / frame.image).is_file():
            raise FileNotFoundError(
                f"{path} frame {i} names image {frame.image!r}, which is not beside it"
            )
        stale = sorted({m.target for m in frame.marks} - names)
        if stale:
            raise ValueError(
                f"{path} frame {i} has marks for {stale}, which the skeleton does not name"
            )
        for mark in frame.marks:
            mark.validate()
    created = example_store.create_set(
        experiment.dir, source.name, source.skeleton, purpose=source.purpose
    )
    for frame in source.frames:
        rgb = _read_rgb(path.parent / frame.image)
        frame.image = example_store.add_frame_image(experiment.dir, created.id, rgb)
    created.frames = source.frames
    created.sources = source.sources
    created.extra = source.extra
    example_store.write_set(experiment.dir, created)
    return created


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"frame image missing or unreadable: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def example_set_runs(experiment: "Experiment", set_id: str) -> list[tuple[str, str]]:
    """``(recording_id, run_id)`` of every keypoint detect run that lists the set (a set a run
    lists cannot be deleted)."""
    return example_store.runs_using(experiment.dir, set_id)


def update_example_set(
    experiment: "Experiment", set_id: str, *, name: str | None = None, skeleton=None
) -> ExampleSet:
    """Rename a set, and/or change its skeleton **until its first mark**.

    ``skeleton`` is anything :func:`resolve_skeleton` takes (a library or preset id, …); the
    set gets that copy and its ``skeleton_ref``. Once any frame carries a mark the skeleton
    is fixed (§26 *Example sets follow the model that needs them*): marks name points, so a
    different skeleton is a new set. Frames without marks are kept; they hold no points.

    Raises:
        ValueError: Nothing to change, an empty name, an unknown skeleton, or a skeleton change
            on a set that has marks.
    """
    if name is None and skeleton is None:
        raise ValueError("nothing to change: give a name or a skeleton")
    current = example_store.read_set(experiment.dir, set_id)
    if name is not None:
        if not str(name).strip():
            raise ValueError("an example set needs a name")
        current.name = str(name).strip()
    if skeleton is not None:
        sk, ref = resolve_skeleton(experiment, skeleton)
        marked = sum(1 for f in current.frames if f.marks)
        if marked and sk != current.skeleton:
            raise ValueError(
                f"example set {current.name!r} has marks on {marked} frame(s), so its "
                "skeleton is fixed; make a new example set to use another skeleton"
            )
        current.skeleton = sk
        current.skeleton_ref = dict(ref) if ref is not None else None
    example_store.write_set(experiment.dir, current)
    return current


def duplicate_example_set(
    experiment: "Experiment", set_id: str, *, name: str | None = None
) -> ExampleSet:
    """A new, empty set holding a copy of this set's skeleton and its ``skeleton_ref``."""
    source = example_store.read_set(experiment.dir, set_id)
    copy = example_store.create_set(
        experiment.dir, name or f"{source.name} copy", source.skeleton
    )
    if source.skeleton_ref is not None:
        copy.skeleton_ref = dict(source.skeleton_ref)
        example_store.write_set(experiment.dir, copy)
    return copy


def export_example_set(
    experiment: "Experiment", set_id: str, out_dir: Path | str
) -> Path:
    """Write the set to ``out_dir`` as ``examples.yaml`` with its frame images beside it under
    ``frames/`` (what :func:`import_example_set` reads); a set with no frames is just the
    skeleton. Returns the YAML's path.

    Raises:
        FileExistsError: ``out_dir`` already holds an ``examples.yaml``.
    """
    import shutil

    from detail.core import atomic

    current = example_store.read_set(experiment.dir, set_id)
    out = Path(out_dir)
    target = out / example_store.EXAMPLES_FILENAME
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose an empty folder")
    out.mkdir(parents=True, exist_ok=True)
    source_dir = example_store.set_dir(experiment.dir, set_id)
    for frame in current.frames:
        dest = out / frame.image
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / frame.image, dest)
    body = current.to_dict(
        image_hashes=example_store.image_hashes(experiment.dir, current)
    )
    tmp = out / (target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(body, handle, sort_keys=False)
    atomic.replace(tmp, target)
    return target


def delete_example_set(experiment: "Experiment", set_id: str) -> None:
    """Delete a set no ``keypoint_detect`` run lists (``storage.example_sets.delete_set``)."""
    example_store.delete_set(experiment.dir, set_id)


# -- keypoints (§26): PR 4 -- keypoint detect: test_keypoints (a few frames, no run)


# -- keypoints (§26): tuning a track method -------------------------------------------- #
def tune_example_sets(
    experiment: "Experiment", keypoint_detect_spec: dict | None = None
) -> list[dict]:
    """The example sets Tune tracking could score with, each marked usable or not with the
    reason (``pipeline.keypoint_track_tune.example_set_choices``). Hashes every set's images:
    call it off the UI thread."""
    from detail.pipeline import keypoint_track_tune

    return keypoint_track_tune.example_set_choices(experiment.dir, keypoint_detect_spec)


def tune_keypoint_tracking(
    experiment: "Experiment",
    example_set_id: str,
    *,
    detect_run_id: str | None = None,
    keypoint_detect_spec: dict | None = None,
    tracking_run_id: str | None = None,
    tracking_spec: dict | None = None,
    match: dict | None = None,
    base_config: dict | None = None,
    seed: int = 0,
    progress=None,
):
    """Which ``temporal_decode`` settings keep a detect run's accuracy, scored against an
    example set the detect run did not learn from (``pipeline.keypoint_track_tune``); derived,
    never stored. Name the detect runs by one ``detect_run_id`` (``kd_NN``, the same in every
    recording) or by a ``keypoint_detect_spec`` (each recording's own match, as the tracking
    form chooses them). The poses reach the labelled subjects through a tracking run on each
    detect run's detection run, named the same two ways (``tracking_run_id`` /
    ``tracking_spec``) or, with neither, the one complete tracking run there (several are
    refused, as the track stage refuses them), and assigned with ``match`` (§26 *Assignment*;
    None: the defaults).
    ``base_config`` is the track config the tried settings apply on.

    Raises:
        ValueError: Neither or both of ``detect_run_id`` / ``keypoint_detect_spec``; both of
            ``tracking_run_id`` / ``tracking_spec``, or neither while a recording has several
            tracking runs on the detection run; the example set is the detect run's own; no
            labelled frame lies under a complete detect run.
        FileNotFoundError: No such example set.
    """
    from detail.api import run_spec
    from detail.pipeline import keypoint_track_tune
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    if (detect_run_id is None) == (keypoint_detect_spec is None):
        raise ValueError(
            "name the detect runs by detect_run_id or by keypoint_detect_spec (one of them)"
        )
    if tracking_run_id is not None and tracking_spec is not None:
        raise ValueError(
            "name the tracking runs by tracking_run_id or by tracking_spec, not both"
        )
    labels = example_store.read_set(experiment.dir, example_set_id)

    def resolve(recording_id: str, run_id: str | None, module: str, spec: dict | None):
        if run_id is not None:
            return run_id
        rec_dir = experiment_store.recording_path(experiment.dir, recording_id)
        return run_spec.resolve_spec_in_recording(
            recording_store.read_recording(rec_dir), module, spec, rec_dir
        )

    def only_tracking(recording_id: str, detect_id: str) -> str:
        """The track stage's own default: the one tracking run on the detection run."""
        from detail.pipeline.keypoint_track import _resolve_tracking_run
        from detail.storage.keypoint_runs import KeypointDetectRun

        rec_dir = experiment_store.recording_path(experiment.dir, recording_id)
        return _resolve_tracking_run(
            rec_dir,
            recording_store.read_recording(rec_dir),
            KeypointDetectRun(rec_dir, detect_id),
            None,
        )["id"]

    detect_runs: dict[str, str] = {}
    tracking_runs: dict[str, str] = {}
    for recording_id in sorted({f.recording_id for f in labels.frames}):
        found = resolve(
            recording_id, detect_run_id, "keypoint_detect", keypoint_detect_spec
        )
        if found is None:
            continue
        detect_runs[recording_id] = found
        if tracking_run_id is None and tracking_spec is None:
            tracked = only_tracking(recording_id, found)
        else:
            tracked = resolve(recording_id, tracking_run_id, "tracking", tracking_spec)
        if tracked is not None:
            tracking_runs[recording_id] = tracked
    return keypoint_track_tune.tune(
        experiment.dir,
        detect_runs,
        example_set_id,
        tracking_runs=tracking_runs,
        match=match,
        base_config=base_config,
        seed=seed,
        progress=progress,
    )


# -- keypoints (§26): previewing a track method ------------------------------------------ #
def preview_keypoint_tracking(
    experiment: "Experiment",
    config: dict,
    recordings: list[str] | None = None,
    *,
    starts: dict[str, float | None] | None = None,
    duration_s: float | None = None,
    variants: list[dict] | None = None,
    grid: bool = False,
    progress=None,
):
    """Run the keypoint track stage over a few seconds of a few recordings, in memory, for
    the current settings, detection alone and up to two ``variants`` (overrides on the
    current method settings), and describe each (``pipeline.keypoint_track_preview``);
    derived, never stored. With ``grid``, every *Tune tracking* grid setting runs in place
    of ``variants`` and the result marks the provisional front (``result.front``, shown in
    ``result.ranked()`` order).

    ``config`` is the track stage's launch config, as ``run_processing("keypoint_track",
    config)`` takes it (a ``keypoint_detect_spec`` or ``detect_run_id``, the tracking run, the
    match, ``method`` and ``config``), and each recording is validated exactly as a launch
    validates it. ``recordings`` are ids or names (at most three); None takes two of those
    the config covers, spread across the list. ``starts`` maps a recording to its window's
    start in seconds (missing or None: the middle of the recording); ``duration_s`` is every
    window's length (``KeypointTrackPreviewWindow``'s default when None).

    Raises:
        ValueError: A recording the config does not cover, more than three, none covered,
            a window past a recording's end, a variant the method's model refuses, variants
            with ``grid``, or ``grid`` on a method other than temporal_decode.
        KeyError: No recording by that id or name.
    """
    from detail.configs.keypoint_track import (
        PREVIEW_DEFAULT_RECORDINGS,
        KeypointTrackPreviewWindow,
    )
    from detail.pipeline import keypoint_track_preview
    from detail.storage import experiment as experiment_store

    module = _processing_module_for("keypoint_track")
    if recordings is None:
        covered = [
            sid
            for sid in experiment.recording_ids
            if _covers(experiment, module, sid, config)
        ]
        if not covered:
            from detail.storage import guidance

            raise ValueError(
                "no recording has the keypoint detect run and tracking run these settings "
                f"read; {guidance.missing_keypoints_hint('keypoint detect')}"
            )
        recording_ids = subject_frames.choose_recordings(
            covered, PREVIEW_DEFAULT_RECORDINGS
        )
    else:
        recording_ids = [experiment.resolve_recording(ref).id for ref in recordings]
    starts = {
        experiment.resolve_recording(ref).id: start
        for ref, start in (starts or {}).items()
    }
    window_fields = {} if duration_s is None else {"duration_s": duration_s}
    _recorded, validated = experiment._validate_processing(
        module, recording_ids, dict(config)
    )
    sources = [
        keypoint_track_preview.PreviewSource(
            recording_id=sid,
            recording_dir=experiment_store.recording_path(experiment.dir, sid),
            config=validated[sid],
            window=KeypointTrackPreviewWindow(start_s=starts.get(sid), **window_fields),
        )
        for sid in recording_ids
    ]
    return keypoint_track_preview.preview(
        sources, variants, grid=grid, progress=progress
    )


def _processing_module_for(module_id: str):
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import PROCESSING_MODULES, import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    return PROCESSING_MODULES[module_id]


def _covers(experiment: "Experiment", module, recording_id: str, config: dict) -> bool:
    """Whether a launch of ``config`` would accept ``recording_id``."""
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    rec_dir = experiment_store.recording_path(experiment.dir, recording_id)
    data = recording_store.read_recording(rec_dir)
    config = none_means_default(module.config_model, config)
    try:
        module.validate(
            rec_dir, data, module.resolve_upstream_spec(rec_dir, data, config)
        )
    except (ValueError, FileNotFoundError):
        return False
    return True
